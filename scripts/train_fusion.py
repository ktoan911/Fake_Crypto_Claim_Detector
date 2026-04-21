#!/usr/bin/env python3
"""
Train Fusion MLP + beta only using CSV labeled data.
Required CSV columns: text/claim, evidence, label
"""

import argparse
import os
import sys

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
from src.data_process.csv_loader import CSVLabeledLoader

from src.training.fusion_trainer import (
    FusionTrainingConfig,
    train_fusion_from_dataframe,
)
from src.utils import normalize_text



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
        default=os.getenv("LORA_MODEL_PATH", "ktoan911/fact-check-Qwen3-4B-finetune"),
        help="Path to the LoRA-trained model (default: models/lora_llm)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
        if os.getenv("CUDA_VISIBLE_DEVICES")
        or os.system("nvidia-smi > /dev/null 2>&1") == 0
        else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.getenv("FUSION_OUTPUT_PATH", "models/fusion_model.pt"),
        help="Path to save the fusion model",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default=os.getenv("RETRIEVER_MODEL_PATH", "AITeamVN/Vietnamese_Embedding"),
        help="Path to trained dense retrieval model (default: models/retriever_model)",
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

                    if content.strip() and len(content.strip()) > 10:
                        norm_key = normalize_text(content.strip())
                        if norm_key not in unique_docs:
                            unique_docs[norm_key] = {
                                "text": content.strip(),
                                "timestamp": timestamp,
                                "source": "json",
                            }
                elif isinstance(ev, str):
                    parts.append(ev)
                    if ev.strip() and len(ev.strip()) > 10:
                        norm_key = normalize_text(ev.strip())
                        if norm_key not in unique_docs:
                            unique_docs[norm_key] = {
                                "text": ev.strip(),
                                "timestamp": None,
                                "source": "json",
                            }

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
                article = article.strip()
                if len(article) > 10:
                    norm_key = normalize_text(article)

                    if norm_key not in unique_docs:
                        unique_docs[norm_key] = {
                            "text": article,
                            "timestamp": ts,
                            "source": "csv",
                        }
                    else:
                        if (
                            ts is not None
                            and unique_docs[norm_key]["timestamp"] is None
                        ):
                            unique_docs[norm_key]["timestamp"] = ts

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
