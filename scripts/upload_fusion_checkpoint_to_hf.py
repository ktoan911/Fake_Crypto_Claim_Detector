#!/usr/bin/env python3
"""
Upload a local fusion checkpoint (.pt) to a Hugging Face model repo.

Examples:
  python3 scripts/upload_fusion_checkpoint_to_hf.py \
    --local_file models/fusion_model_resumed.pt \
    --repo_id toan911/fact-check-fusion-model \
    --path_in_repo fusion_model.pt
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


def _upload_with_hf_cli(
    repo_id: str,
    local_file: str,
    path_in_repo: str,
    repo_type: str,
    revision: str,
    commit_message: str,
) -> None:
    cmd = [
        "hf",
        "upload",
        repo_id,
        local_file,
        path_in_repo,
        "--type",
        repo_type,
        "--commit-message",
        commit_message,
    ]
    if revision:
        cmd.extend(["--revision", revision])

    logger.info(f"Uploading with hf CLI: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _upload_with_hf_hub(
    repo_id: str,
    local_file: str,
    path_in_repo: str,
    repo_type: str,
    revision: str,
    commit_message: str,
) -> None:
    from huggingface_hub import HfApi

    logger.info("hf CLI upload failed, retrying with huggingface_hub.HfApi...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_file,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision or None,
        commit_message=commit_message,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local fusion checkpoint .pt file to Hugging Face Hub."
    )
    parser.add_argument(
        "--local_file",
        type=str,
        default="models/fusion_model_resumed.pt",
        help="Path to local .pt file.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=os.getenv("FUSION_MODEL", "toan911/fact-check-fusion-model"),
        help="HF model repo id (namespace/repo_name).",
    )
    parser.add_argument(
        "--path_in_repo",
        type=str,
        default="fusion_model.pt",
        help="Target file path inside HF repo.",
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
        help="HF repository type.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Target branch/revision.",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload fusion checkpoint",
        help="Commit message for upload.",
    )
    parser.add_argument(
        "--no_fallback",
        action="store_true",
        help="Disable fallback upload via huggingface_hub if hf CLI fails.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    local_file = os.path.abspath(args.local_file)
    if not os.path.isfile(local_file):
        raise FileNotFoundError(f"Local checkpoint file not found: {local_file}")

    if Path(local_file).suffix.lower() != ".pt":
        logger.warning(
            f"File extension is not .pt ({local_file}). Proceeding anyway."
        )

    token_exists = bool(os.getenv("HF_TOKEN"))
    logger.info(
        "Starting upload "
        f"| local_file={local_file} "
        f"| repo_id={args.repo_id} "
        f"| path_in_repo={args.path_in_repo} "
        f"| revision={args.revision} "
        f"| hf_token={'set' if token_exists else 'missing'}"
    )

    try:
        _upload_with_hf_cli(
            repo_id=args.repo_id,
            local_file=local_file,
            path_in_repo=args.path_in_repo,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message=args.commit_message,
        )
    except Exception as exc:
        if args.no_fallback:
            raise RuntimeError(f"hf CLI upload failed and fallback disabled: {exc}") from exc

        _upload_with_hf_hub(
            repo_id=args.repo_id,
            local_file=local_file,
            path_in_repo=args.path_in_repo,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message=args.commit_message,
        )

    logger.info("Upload completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error(f"Upload failed: {exc}")
        sys.exit(1)
