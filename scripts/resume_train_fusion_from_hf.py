#!/usr/bin/env python3
"""
Resume fusion training from a Hugging Face checkpoint (.pt).

Example:
python3 scripts/resume_train_fusion_from_hf.py \
  --labeled_path data/time_dataset.json \
  --hf_repo ktoan911/fact-check-fusion-model \
  --epochs 10 \
  --save_path models/fusion_model_resumed.pt
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_process.csv_loader import CSVLabeledLoader
from src.training.fusion_trainer import FusionTrainingConfig, train_fusion_from_dataframe
from src.utils import normalize_text


def _load_labeled_and_kb(labeled_path: str) -> Tuple[pd.DataFrame, List[Dict]]:
    file_ext = os.path.splitext(labeled_path)[1].lower()
    unique_docs = {}

    if file_ext in [".json", ".jsonl"]:
        with open(labeled_path, "r", encoding="utf-8") as f:
            if file_ext == ".jsonl":
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]

        df_rows = []
        for item in data:
            claim = item.get("claim", "")
            label = item.get("label", "")
            evidences = item.get("evidence", [])
            evidence_parts = []

            for ev in evidences:
                if isinstance(ev, dict) and "content" in ev:
                    content = str(ev["content"]).strip()
                    raw_ts = ev.get("timestamp")
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

                    evidence_parts.append(content)
                    if content and len(content) > 10:
                        norm_key = normalize_text(content)
                        if norm_key not in unique_docs:
                            unique_docs[norm_key] = {
                                "text": content,
                                "timestamp": timestamp,
                                "source": "json",
                            }
                elif isinstance(ev, str):
                    content = ev.strip()
                    evidence_parts.append(content)
                    if content and len(content) > 10:
                        norm_key = normalize_text(content)
                        if norm_key not in unique_docs:
                            unique_docs[norm_key] = {
                                "text": content,
                                "timestamp": None,
                                "source": "json",
                            }

            df_rows.append(
                {
                    "text": claim,
                    "label": label,
                    "evidence": "|||".join(evidence_parts),
                }
            )

        labeled_df = pd.DataFrame(df_rows)
    else:
        labeled_df = CSVLabeledLoader(labeled_path).load()
        evidences = labeled_df["evidence"].tolist()
        timestamps = (
            labeled_df["timestamp"].tolist()
            if "timestamp" in labeled_df.columns
            else [None] * len(evidences)
        )

        for evidence, ts in zip(evidences, timestamps):
            for article in str(evidence).split("|||"):
                article = article.strip()
                if len(article) <= 10:
                    continue
                norm_key = normalize_text(article)
                if norm_key not in unique_docs:
                    unique_docs[norm_key] = {
                        "text": article,
                        "timestamp": ts,
                        "source": "csv",
                    }
                elif ts is not None and unique_docs[norm_key]["timestamp"] is None:
                    unique_docs[norm_key]["timestamp"] = ts

    kb_docs = list(unique_docs.values())
    return labeled_df, kb_docs


def _download_with_hf_cli(
    repo_id: str, filename: str, local_dir: str, revision: str = ""
) -> str:
    cmd = [
        "hf",
        "download",
        repo_id,
        filename,
        "--type",
        "model",
        "--local-dir",
        local_dir,
        "--quiet",
    ]
    if revision:
        cmd.extend(["--revision", revision])

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = (result.stdout or "").strip()
    candidates = []
    if stdout:
        candidates.append(stdout.splitlines()[-1].strip())
    candidates.append(str(Path(local_dir) / filename))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        f"Downloaded '{filename}' but could not resolve local path from output."
    )


def _download_with_hf_hub(
    repo_id: str, filename: str, local_dir: str, revision: str = ""
) -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        revision=revision or None,
        local_dir=local_dir,
    )
    return os.path.abspath(path)


def _resolve_resume_checkpoint(
    repo_id: str,
    filenames: List[str],
    local_dir: str,
    revision: str = "",
    local_checkpoint: str = "",
) -> str:
    if local_checkpoint:
        if not os.path.isfile(local_checkpoint):
            raise FileNotFoundError(f"Local checkpoint not found: {local_checkpoint}")
        return os.path.abspath(local_checkpoint)

    os.makedirs(local_dir, exist_ok=True)
    errors = []

    for filename in filenames:
        try:
            path = _download_with_hf_cli(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                revision=revision,
            )
            logger.info(f"Downloaded checkpoint via hf CLI: {path}")
            return path
        except Exception as exc:
            errors.append(f"hf-cli {filename}: {exc}")
            logger.warning(f"hf CLI download failed for {filename}: {exc}")

    for filename in filenames:
        try:
            path = _download_with_hf_hub(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                revision=revision,
            )
            logger.info(f"Downloaded checkpoint via huggingface_hub: {path}")
            return path
        except Exception as exc:
            errors.append(f"hf-hub {filename}: {exc}")
            logger.warning(f"huggingface_hub download failed for {filename}: {exc}")

    raise RuntimeError(
        "Could not download any candidate checkpoint filename from HF repo.\n"
        f"repo={repo_id}, filenames={filenames}, revision={revision or 'main'}\n"
        + "\n".join(errors)
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a fusion .pt checkpoint from Hugging Face and continue training."
    )
    parser.add_argument(
        "--labeled_path",
        type=str,
        default="data/time_dataset.json",
        help="Path to labeled training file (.json/.jsonl/.csv).",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=os.getenv("FUSION_MODEL_PATH", "ktoan911/fact-check-fusion-model"),
        help="HF model repo id containing fusion checkpoint (.pt).",
    )
    parser.add_argument(
        "--hf_filenames",
        type=str,
        nargs="+",
        default=["model.pt", "fusion_model.pt"],
        help="Candidate checkpoint filenames to try in HF repo.",
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default="",
        help="Optional HF revision/branch/tag/commit.",
    )
    parser.add_argument(
        "--local_ckpt_dir",
        type=str,
        default="models/hf_downloads/fusion_checkpoint",
        help="Where to store downloaded checkpoint locally.",
    )
    parser.add_argument(
        "--local_checkpoint",
        type=str,
        default="",
        help="Use an existing local checkpoint path instead of downloading.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for fusion training."
    )
    parser.add_argument(
        "--llm_batch_size", type=int, default=8, help="Batch size for LLM scoring."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Fusion learning rate."
    )
    parser.add_argument(
        "--beta_lr_multiplier",
        type=float,
        default=5.0,
        help="Multiply beta gate learning rate relative to --learning_rate.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k retrieved docs for retrieval features.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
        if os.getenv("CUDA_VISIBLE_DEVICES")
        or os.system("nvidia-smi > /dev/null 2>&1") == 0
        else "cpu",
        help="Device to use (cuda/cpu).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.getenv("LORA_MODEL_PATH", "models/lora_llm"),
        help="Path/repo for LoRA model used by LLM scorer.",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default=os.getenv("RETRIEVER_MODEL_PATH", "AITeamVN/Vietnamese_Embedding"),
        help="Retriever embedding model path/repo.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="models/fusion_model_resumed.pt",
        help="Path to save resumed fusion checkpoint.",
    )
    parser.add_argument(
        "--evidence_mode",
        type=str,
        default="retrieved",
        choices=["retrieved", "gold"],
        help="Evidence mode used during LLM scoring.",
    )
    parser.add_argument(
        "--resume_strict",
        action="store_true",
        help="Strictly enforce checkpoint keys when loading resume checkpoint.",
    )
    parser.add_argument(
        "--disable_resume_runtime_alignment",
        action="store_true",
        help="Do not override model/retriever/evidence settings from checkpoint config.",
    )
    parser.add_argument(
        "--disable_logit_normalization",
        action="store_true",
        help="Disable branch-logit normalization inside fusion layer.",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    labeled_path = os.path.abspath(args.labeled_path)
    if not os.path.isfile(labeled_path):
        raise FileNotFoundError(f"Labeled file not found: {labeled_path}")

    logger.info(f"Loading labeled data from: {labeled_path}")
    labeled_df, kb_docs = _load_labeled_and_kb(labeled_path)
    logger.info(f"Labeled data: {len(labeled_df)} samples")
    logger.info(f"Knowledge base built: {len(kb_docs)} unique documents")

    ckpt_path = _resolve_resume_checkpoint(
        repo_id=args.hf_repo,
        filenames=args.hf_filenames,
        local_dir=os.path.abspath(args.local_ckpt_dir),
        revision=args.hf_revision,
        local_checkpoint=args.local_checkpoint,
    )
    logger.info(f"Using resume checkpoint: {ckpt_path}")

    fusion_config = FusionTrainingConfig(
        model_name=args.model_path,
        retriever_model=args.retriever_model,
        device=args.device,
        batch_size=args.batch_size,
        llm_batch_size=args.llm_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta_lr_multiplier=args.beta_lr_multiplier,
        top_k=args.top_k,
        evidence_mode=args.evidence_mode,
        align_runtime_with_resume_checkpoint=not args.disable_resume_runtime_alignment,
        normalize_branch_logits=not args.disable_logit_normalization,
    )

    output_path = train_fusion_from_dataframe(
        knowledge_base=kb_docs,
        labeled_df=labeled_df,
        config=fusion_config,
        save_path=args.save_path,
        resume_checkpoint_path=ckpt_path,
        resume_strict=args.resume_strict,
    )
    logger.info(f"Resumed fusion training complete. Model saved to: {output_path}")


if __name__ == "__main__":
    main()
