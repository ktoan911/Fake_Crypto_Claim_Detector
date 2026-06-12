#!/usr/bin/env python3
"""
Sequential fusion training:
  Phase 1 — train on train.csv  (general fact-checking)
  Phase 2 — fine-tune on time_dataset.json  (temporal / financial)

Usage:
  python scripts/train_sequential.py \
      --train_csv data/train.csv \
      --time_json data/time_dataset.json \
      --phase1_save models/fusion_phase1.pt \
      --final_save  models/fusion_final.pt
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from loguru import logger

from src.data_process.csv_loader import CSVLabeledLoader
from src.training.fusion_trainer import FusionTrainingConfig, train_fusion_from_dataframe
from src.utils import normalize_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_kb(unique_docs: dict):
    return list(unique_docs.values())


def _load_csv(path: str):
    """Load labeled CSV and extract knowledge-base docs."""
    labeled_df = CSVLabeledLoader(path).load()
    evidences = labeled_df["evidence"].tolist()
    timestamps = (
        labeled_df["timestamp"].tolist()
        if "timestamp" in labeled_df.columns
        else [None] * len(evidences)
    )
    unique_docs = {}
    for evidence, ts in zip(evidences, timestamps):
        for article in str(evidence).split("|||"):
            article = article.strip()
            if len(article) > 10:
                key = normalize_text(article)
                if key not in unique_docs:
                    unique_docs[key] = {"text": article, "timestamp": ts, "source": "csv"}
                elif ts is not None and unique_docs[key]["timestamp"] is None:
                    unique_docs[key]["timestamp"] = ts
    return labeled_df, unique_docs


def _load_json(path: str):
    """Load time_dataset.json and extract (labeled_df, kb unique_docs)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    unique_docs = {}
    rows = []
    for d in data:
        claim = d.get("claim") or d.get("text", "")
        label = d.get("label", "")
        evs = d.get("evidence", [])
        parts = []
        for ev in evs:
            if isinstance(ev, dict) and "content" in ev:
                content = ev["content"]
                raw_ts = ev.get("timestamp")
                timestamp = None
                if raw_ts:
                    try:
                        timestamp = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                    except Exception:
                        timestamp = datetime.now(timezone.utc)
                parts.append(content)
                if content.strip() and len(content.strip()) > 10:
                    key = normalize_text(content.strip())
                    if key not in unique_docs:
                        unique_docs[key] = {"text": content.strip(), "timestamp": timestamp, "source": "json"}
            elif isinstance(ev, str):
                parts.append(ev)
                if ev.strip() and len(ev.strip()) > 10:
                    key = normalize_text(ev.strip())
                    if key not in unique_docs:
                        unique_docs[key] = {"text": ev.strip(), "timestamp": None, "source": "json"}

        rows.append({"text": claim, "label": label, "evidence": "|||".join(parts)})

    return pd.DataFrame(rows), unique_docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sequential fusion training: CSV → JSON fine-tune")

    # Data paths
    parser.add_argument("--train_csv",   required=True, help="Path to train.csv")
    parser.add_argument("--time_json",   required=True, help="Path to time_dataset.json")

    # Checkpoint paths
    parser.add_argument("--phase1_save", default="models/fusion_phase1.pt",
                        help="Where to save the Phase-1 checkpoint (default: models/fusion_phase1.pt)")
    parser.add_argument("--final_save",  default="models/fusion_final.pt",
                        help="Where to save the final checkpoint (default: models/fusion_final.pt)")

    # Model paths
    parser.add_argument("--model_path",
                        default=os.getenv("LORA_MODEL_PATH", "ktoan911/Qwen3-4B-factcheck-finetune"),
                        help="LoRA model path/HF repo")
    parser.add_argument("--retriever_model",
                        default=os.getenv("RETRIEVER_MODEL_PATH", "AITeamVN/Vietnamese_Embedding"),
                        help="Dense retriever model path/HF repo")

    # Device
    parser.add_argument("--device", default="cuda",
                        help="Compute device: cuda or cpu (default: cuda)")

    # Phase 1 hyper-params
    parser.add_argument("--p1_epochs",     type=int,   default=15,   help="Phase 1 epochs (default: 15)")
    parser.add_argument("--p1_lr",         type=float, default=1e-4, help="Phase 1 learning rate (default: 1e-4)")
    parser.add_argument("--p1_batch_size", type=int,   default=8,    help="Phase 1 batch size (default: 8)")
    parser.add_argument("--p1_llm_batch",  type=int,   default=8,    help="Phase 1 LLM batch size (default: 8)")

    # Phase 2 hyper-params
    parser.add_argument("--p2_epochs",     type=int,   default=10,   help="Phase 2 epochs (default: 10)")
    parser.add_argument("--p2_lr",         type=float, default=5e-5, help="Phase 2 learning rate — lower for fine-tune (default: 5e-5)")
    parser.add_argument("--p2_batch_size", type=int,   default=8,    help="Phase 2 batch size (default: 8)")
    parser.add_argument("--p2_llm_batch",  type=int,   default=8,    help="Phase 2 LLM batch size (default: 8)")

    # NLI
    parser.add_argument("--no_nli", action="store_true", help="Disable NLI stance features")
    parser.add_argument("--nli_model",
                        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                        help="NLI model name/path")
    parser.add_argument("--nli_batch_size", type=int, default=64, help="NLI scoring batch size")

    args = parser.parse_args()

    use_nli = not args.no_nli

    # -----------------------------------------------------------------------
    # Phase 1: train on train.csv
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 1 — Training on train.csv")
    logger.info("=" * 60)

    labeled_df_p1, unique_docs_p1 = _load_csv(args.train_csv)
    kb_p1 = _build_kb(unique_docs_p1)
    logger.info(f"Phase 1 data: {len(labeled_df_p1)} samples, {len(kb_p1)} KB docs")

    config_p1 = FusionTrainingConfig(
        model_name=args.model_path,
        retriever_model=args.retriever_model,
        device=args.device,
        batch_size=args.p1_batch_size,
        llm_batch_size=args.p1_llm_batch,
        epochs=args.p1_epochs,
        learning_rate=args.p1_lr,
        evidence_mode="gold",
        use_nli=use_nli,
        nli_model=args.nli_model,
        nli_batch_size=args.nli_batch_size,
    )

    train_fusion_from_dataframe(
        knowledge_base=kb_p1,
        labeled_df=labeled_df_p1,
        config=config_p1,
        save_path=args.phase1_save,
    )
    logger.info(f"Phase 1 complete — checkpoint: {args.phase1_save}")

    # -----------------------------------------------------------------------
    # Phase 2: fine-tune on time_dataset.json
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 2 — Fine-tuning on time_dataset.json")
    logger.info("=" * 60)

    labeled_df_p2, unique_docs_p2 = _load_json(args.time_json)
    kb_p2 = _build_kb(unique_docs_p2)
    logger.info(f"Phase 2 data: {len(labeled_df_p2)} samples, {len(kb_p2)} KB docs")

    config_p2 = FusionTrainingConfig(
        model_name=args.model_path,
        retriever_model=args.retriever_model,
        device=args.device,
        batch_size=args.p2_batch_size,
        llm_batch_size=args.p2_llm_batch,
        epochs=args.p2_epochs,
        learning_rate=args.p2_lr,
        evidence_mode="retrieved",
        use_nli=use_nli,
        nli_model=args.nli_model,
        nli_batch_size=args.nli_batch_size,
        align_runtime_with_resume_checkpoint=False,
    )

    train_fusion_from_dataframe(
        knowledge_base=kb_p2,
        labeled_df=labeled_df_p2,
        config=config_p2,
        save_path=args.final_save,
        resume_checkpoint_path=args.phase1_save,
    )
    logger.info(f"Phase 2 complete — final model: {args.final_save}")


if __name__ == "__main__":
    main()
