#!/usr/bin/env python3
"""
Train Fusion MLP + beta only using CSV labeled data.
Required CSV columns: text/claim, evidence, label
"""

import argparse
import os
import sys

import torch

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
from src.data_process.csv_loader import CSVLabeledLoader

from src.training.fusion_trainer import (
    FusionTrainingConfig,
    train_fusion_from_dataframe,
)
from src.utils import normalize_text


def _add_doc_if_new(unique_docs: dict, text: str, timestamp, source: str) -> None:
    text = text.strip()
    if not text or len(text) <= 10:
        return
    key = normalize_text(text)
    if key not in unique_docs:
        unique_docs[key] = {"text": text, "timestamp": timestamp, "source": source}
    elif timestamp is not None and unique_docs[key]["timestamp"] is None:
        unique_docs[key]["timestamp"] = timestamp


def main():
    parser = argparse.ArgumentParser(
        description="Train Fusion MLP + beta only using CSV labeled data."
    )
    parser.add_argument(
        "--labeled_csv",
        type=str,
        required=True,
        help="Path to the labeled CSV file (text,evidence,label)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--llm_batch_size", type=int, default=8, help="Batch size for LLM"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.getenv("LORA_MODEL_PATH", "ktoan911/Qwen3-4B-factcheck-finetune"),
        help="Path to the LoRA-trained model (default: models/lora_llm)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.getenv("FUSION_OUTPUT_PATH"),
        help="Path to save the fusion model",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default=os.getenv("RETRIEVER_MODEL_PATH", "AITeamVN/Vietnamese_Embedding"),
        help="Path to trained dense retrieval model (default: models/retriever_model)",
    )
    parser.add_argument(
        "--evidence_mode",
        type=str,
        default=os.getenv("FUSION_EVIDENCE_MODE", "retrieved"),
        choices=["gold", "retrieved"],
        help="Evidence source: 'gold' = dùng cột evidence từ CSV, 'retrieved' = retriever tìm kiếm "
        "(default: retrieved, phải khớp với evidence thực tế lúc serving)",
    )
    parser.add_argument(
        "--llm_evidence_top_k",
        type=int,
        default=int(os.getenv("FUSION_LLM_EVIDENCE_TOP_K", "5")),
        help="Số bằng chứng đưa vào LLM judge, phải khớp FUSION_LLM_EVIDENCE_TOP_K lúc serving",
    )

    args = parser.parse_args()

    logger.info(f"Loading labeled data from {args.labeled_csv}...")
    file_ext = os.path.splitext(args.labeled_csv)[1].lower()

    unique_docs = {}
    import pandas as pd

    if file_ext in [".json", ".jsonl"]:
        import json
        from datetime import datetime, timezone

        with open(args.labeled_csv, "r", encoding="utf-8") as f:
            if file_ext == ".jsonl":
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]

        df_rows = []
        for d in data:
            claim = d.get("claim", "")
            label = d.get("label", "")
            evs = d.get("evidence", [])
            parts = []

            for ev in evs:
                if isinstance(ev, dict) and "content" in ev:
                    content = ev["content"]
                    raw_ts = ev.get("timestamp", None)
                    timestamp = None
                    if raw_ts:
                        try:
                            timestamp = datetime.fromisoformat(
                                str(raw_ts).replace("Z", "+00:00")
                            )
                            if timestamp.tzinfo is None:
                                timestamp = timestamp.replace(tzinfo=timezone.utc)
                        except Exception:
                            timestamp = datetime.now(timezone.utc)

                    parts.append(content)
                    _add_doc_if_new(unique_docs, content, timestamp, "json")
                elif isinstance(ev, str):
                    parts.append(ev)
                    _add_doc_if_new(unique_docs, ev, None, "json")

            evidence_str = "|||".join(parts)
            df_rows.append({"text": claim, "label": label, "evidence": evidence_str})

        labeled_df = pd.DataFrame(df_rows)
    else:
        labeled_df = CSVLabeledLoader(args.labeled_csv).load()

        evidences = labeled_df["evidence"].tolist()
        timestamps = (
            labeled_df["timestamp"].tolist()
            if "timestamp" in labeled_df.columns
            else [None] * len(evidences)
        )

        for evidence, ts in zip(evidences, timestamps):
            evidence_str = str(evidence)
            articles = evidence_str.split("|||")

            for article in articles:
                _add_doc_if_new(unique_docs, article, ts, "csv")

    logger.info(f"Labeled data: {len(labeled_df)} samples")
    kb_docs = list(unique_docs.values())
    logger.info(
        f"Knowledge base built: {len(kb_docs)} unique documents (deduplicated from {len(labeled_df)} labeled samples)"
    )

    fusion_config = FusionTrainingConfig(
        model_name=args.model_path,
        retriever_model=args.retriever_model,
        device=args.device,
        batch_size=args.batch_size,
        llm_batch_size=args.llm_batch_size,
        epochs=args.epochs,
        evidence_mode=args.evidence_mode,
    )

    train_fusion_from_dataframe(
        knowledge_base=kb_docs,
        labeled_df=labeled_df,
        config=fusion_config,
        save_path=args.save_path,
    )

    logger.info(f"Fusion training complete. Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
