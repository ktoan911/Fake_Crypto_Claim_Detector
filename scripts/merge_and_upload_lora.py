#!/usr/bin/env python3
"""
Merge LoRA adapter into the base model and upload the merged model to Hugging Face.

Usage:
  python scripts/merge_and_upload_lora.py \
    --adapter_dir models/lora_llm \
    --repo_id your-username/your-repo-name \
    --merged_dir models/merged_lora    # optional, temp dir for merged weights
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model and upload to Hugging Face Hub."
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="models/lora_llm",
        help="Path to the LoRA adapter directory (contains adapter_config.json).",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=os.getenv("HF_REPO_ID", ""),
        help="Hugging Face repo id to upload to (e.g. username/model-name).",
    )
    parser.add_argument(
        "--merged_dir",
        type=str,
        default="models/merged_lora",
        help="Local directory to save the merged model before uploading.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name/path. Auto-read from adapter_config.json if not set.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Target branch on Hugging Face Hub.",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload merged LoRA model",
    )
    parser.add_argument(
        "--skip_upload",
        action="store_true",
        help="Only merge and save locally, skip the upload step.",
    )
    parser.add_argument(
        "--keep_merged_dir",
        action="store_true",
        help="Keep the merged_dir after upload (default: delete it).",
    )
    return parser.parse_args()


def _read_base_model_from_adapter_config(adapter_dir: str) -> str:
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path") or cfg.get("model_name_or_path")
    if not base:
        raise ValueError(
            "Cannot find base model name in adapter_config.json. "
            "Pass --base_model explicitly."
        )
    return base


def merge(adapter_dir: str, base_model: str, merged_dir: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Base model : {base_model}")
    logger.info(f"Adapter    : {adapter_dir}")
    logger.info(f"Output     : {merged_dir}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir, trust_remote_code=False
    )

    logger.info(f"Loading base model in {dtype} ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    base.config.use_cache = True

    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, adapter_dir)

    logger.info("Merging adapter weights into base model...")
    model = model.merge_and_unload()
    model.eval()

    os.makedirs(merged_dir, exist_ok=True)
    logger.info(f"Saving merged model to {merged_dir} ...")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    logger.info("Merge complete.")


def upload(merged_dir: str, repo_id: str, revision: str, commit_message: str) -> None:
    from huggingface_hub import HfApi

    if not repo_id:
        raise ValueError("--repo_id is required for upload. Use --skip_upload to merge only.")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN environment variable not set. "
            "Run: export HF_TOKEN=hf_... or add it to .env"
        )

    api = HfApi(token=token)

    logger.info(f"Creating repo {repo_id} if it does not exist...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    logger.info(f"Uploading {merged_dir} -> {repo_id} (branch={revision}) ...")
    api.upload_folder(
        folder_path=merged_dir,
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        commit_message=commit_message,
    )
    logger.info(f"Upload complete: https://huggingface.co/{repo_id}")


def main() -> None:
    load_dotenv()
    args = parse_args()

    adapter_dir = os.path.abspath(args.adapter_dir)
    merged_dir = os.path.abspath(args.merged_dir)

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    base_model = args.base_model or _read_base_model_from_adapter_config(adapter_dir)

    merge(adapter_dir, base_model, merged_dir)

    if not args.skip_upload:
        upload(merged_dir, args.repo_id, args.revision, args.commit_message)

        if not args.keep_merged_dir:
            logger.info(f"Removing temporary merged dir {merged_dir} ...")
            shutil.rmtree(merged_dir, ignore_errors=True)
    else:
        logger.info(f"--skip_upload set. Merged model is at {merged_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error(f"Failed: {exc}")
        sys.exit(1)
