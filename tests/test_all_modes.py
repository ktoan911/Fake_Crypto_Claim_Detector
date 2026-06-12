#!/usr/bin/env python3
"""
Comprehensive Test Script: LoRA vs Fusion (Gold vs Retrieval)
Evaluates 4 modes:
1. LoRA + Retrieval Evidence
2. LoRA + Gold Evidence
3. Fusion + Retrieval Evidence
4. Fusion + Gold Evidence

Model sources support both local paths and HuggingFace repo IDs.
Default HF repos are read from env vars (same as .env / fusion_inference.py):
  LLM_FINETUNE        -> LoRA model (e.g. ktoan911/Qwen3-4B-factcheck-finetune)
  FUSION_MODEL        -> Fusion checkpoint (e.g. ktoan911/fact-check-fusion-model)
  RETRIEVER_MODEL     -> Retriever embedding model (e.g. AITeamVN/Vietnamese_Embedding)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import LABEL_LIST, PROMPT_TEMPLATE
from src.data_process.csv_loader import CSVLabeledLoader
from src.llm_scorer import LLMScorer
from src.models.fusion import ConfidenceAwareFusion, RetrievalFeatureEncoder
from src.retrieval.retrieval import KnowledgeAugmentedRetriever
from src.training.fusion_trainer import (  # Re-use helper
    _build_retrieval_features,
)
from src.utils import normalize_text

# ---------------------------------------------------------------------------
# Env-var defaults (mirrors .env / fusion_inference.py convention)
# ---------------------------------------------------------------------------
_DEFAULT_LLM_MODEL = os.getenv("LLM_FINETUNE", "ktoan911/Qwen3-4B-factcheck-finetune")
_DEFAULT_FUSION_MODEL = os.getenv("FUSION_MODEL", "ktoan911/fact-check-fusion-model")
_DEFAULT_RETRIEVER_MODEL = os.getenv("RETRIEVER_MODEL", "AITeamVN/Vietnamese_Embedding")

# Label mapping for metrics
LABEL_MAP = {idx: label for idx, label in enumerate(LABEL_LIST)}


def calculate_metrics(y_true, y_pred, mode_name, label_list=LABEL_LIST):
    """Calculate and print metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class_f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
        labels=list(range(len(label_list))),
    )

    logger.info(f"--- Results for {mode_name} ---")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1 Macro:  {f1_macro:.4f}")
    logger.info(f"F1 Weighted: {f1_weighted:.4f}")
    for idx, label_name in enumerate(label_list):
        logger.info(f"F1 {label_name}: {per_class_f1[idx]:.4f}")

    metrics = {
        "Mode": mode_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Macro": f1_macro,
        "F1_Weighted": f1_weighted,
    }
    for idx, label_name in enumerate(label_list):
        metrics[f"F1_{label_name}"] = per_class_f1[idx]

    return metrics


def _resolve_fusion_model_path(path_or_repo: str, filename: str = "acf_fusion_model_v2.pt") -> str:
    """
    Resolve a fusion model path — mirrors fusion_inference._resolve_fusion_model_path.
    Accepts a local .pt file path OR a HuggingFace repo id.
    """
    if os.path.isfile(path_or_repo):
        return path_or_repo
    try:
        from huggingface_hub import hf_hub_download

        logger.info(
            f"[load_fusion_model] Downloading '{filename}' from HF repo '{path_or_repo}'..."
        )
        local_path = hf_hub_download(repo_id=path_or_repo, filename=filename)
        logger.info(f"[load_fusion_model] Downloaded to {local_path}")
        return local_path
    except Exception as exc:
        raise FileNotFoundError(
            f"Cannot resolve fusion model path '{path_or_repo}': {exc}"
        ) from exc


def load_fusion_model(model_path, device, num_classes=None):
    """
    Load trained fusion model components.
    model_path can be:
      - A local .pt file path
      - A HuggingFace repo id (e.g. 'ktoan911/fact-check-fusion-model')
    """
    resolved_path = _resolve_fusion_model_path(model_path)
    logger.info(f"[load_fusion_model] Loading checkpoint from {resolved_path}")

    checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
    fusion_state = checkpoint["fusion"]

    # Load config from checkpoint if available, else standard
    saved_config = checkpoint.get("config", {})
    if num_classes is None:
        num_classes = saved_config.get("num_classes", len(LABEL_LIST))

    retrieval_encoder = RetrievalFeatureEncoder(
        num_retrieved=saved_config.get("top_k", 10),
        score_features=int(saved_config.get("score_features", 5)),
        hidden_dim=64,
        output_dim=64,
        interaction_dim=int(saved_config.get("interaction_dim", 0)),
    ).to(device)

    has_adaptive_state = any(k.startswith("beta_gate.") for k in fusion_state.keys())
    saved_adaptive = bool(saved_config.get("adaptive_beta", has_adaptive_state))
    if saved_adaptive != has_adaptive_state:
        logger.warning(
            "adaptive_beta in config and fusion state_dict are inconsistent. "
            f"config={saved_adaptive}, state_has_beta_gate={has_adaptive_state}. "
            "Using state_dict architecture for safe loading."
        )
    effective_adaptive_beta = has_adaptive_state

    fusion = ConfidenceAwareFusion(
        retrieval_input_dim=64,
        hidden_dim=128,
        num_classes=num_classes,
        initial_beta=saved_config.get("initial_beta", 0.5),
        lambda_reg=float(saved_config.get("lambda_reg", 0.01)),
        normalize_branch_logits=bool(
            saved_config.get("normalize_branch_logits", False)
        ),
        adaptive_beta=effective_adaptive_beta,
    ).to(device)

    retrieval_encoder.load_state_dict(checkpoint["retrieval_encoder"])
    fusion.load_state_dict(fusion_state)

    retrieval_encoder.eval()
    fusion.eval()

    return retrieval_encoder, fusion, saved_config


def main():
    parser = argparse.ArgumentParser(
        description="Test LoRA and Fusion models in all modes. "
        "Model args accept local paths OR HuggingFace repo ids."
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to test CSV or JSON/JSONL"
    )
    parser.add_argument(
        "--lora_model",
        type=str,
        default=_DEFAULT_LLM_MODEL,
        help=(
            "Path to LoRA/fine-tuned model (local dir or HF repo id). "
            f"Default: {_DEFAULT_LLM_MODEL!r} (from env LLM_FINETUNE)"
        ),
    )
    parser.add_argument(
        "--fusion_model",
        type=str,
        default=_DEFAULT_FUSION_MODEL,
        help=(
            "Path to Fusion checkpoint (.pt) or HF repo id containing acf_fusion_model_v2.pt. "
            f"Default: {_DEFAULT_FUSION_MODEL!r} (from env FUSION_MODEL)"
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples for testing"
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default=None,
        help=(
            "Override dense retrieval embedding model (local path or HF repo id). "
            "If not provided, will fall back to fusion checkpoint config, "
            f"then env RETRIEVER_MODEL ({_DEFAULT_RETRIEVER_MODEL!r})."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for retrieval feature batching",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=int(os.getenv("LLM_INFER_BATCH_SIZE", "32")),
        help="LLM inference batch size (default from env LLM_INFER_BATCH_SIZE or 32)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    logger.info(f"Testing on device: {args.device}")

    # Fusion Model
    retrieval_encoder, fusion_layer, fusion_config = load_fusion_model(
        args.fusion_model, args.device
    )
    top_k = fusion_config.get("top_k", 10)
    # Resolve retriever model: CLI arg > checkpoint config > env var default
    retriever_model = (
        args.retriever_model
        or fusion_config.get("retriever_model")
        or _DEFAULT_RETRIEVER_MODEL
    )

    # Extract dynamic label properties
    dynamic_label_list = fusion_config.get(
        "label_list", ["Đúng", "Sai", "Chưa chắc chắn"]
    )

    def local_normalize_label_to_id(val):
        v = str(val).lower().strip()
        if v in ["true", "đúng", "dung", "supported", "legit", "0"]:
            return (
                dynamic_label_list.index("Đúng") if "Đúng" in dynamic_label_list else 0
            )
        if v in ["false", "sai", "refuted", "scam", "fake", "1"]:
            return dynamic_label_list.index("Sai") if "Sai" in dynamic_label_list else 1
        if v in [
            "nei",
            "thieu",
            "thiếu",
            "thiếu thông tin",
            "chưa chắc chắn",
            "chua chac chan",
            "not enough info",
            "insufficient",
            "2",
        ]:
            if "Chưa chắc chắn" in dynamic_label_list:
                return dynamic_label_list.index("Chưa chắc chắn")
            if "Thiếu" in dynamic_label_list:
                return dynamic_label_list.index("Thiếu")
            return 2
        for idx, lbl in enumerate(dynamic_label_list):
            if v == lbl.lower().strip():
                return idx
        return 0

    # 1. Load Data
    logger.info(f"Loading data from {args.csv}...")
    gold_evidences = []
    unique_docs = {}

    if args.csv.endswith(".json") or args.csv.endswith(".jsonl"):
        import json

        with open(args.csv, "r", encoding="utf-8") as f:
            if args.csv.endswith(".jsonl"):
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]

        if args.limit:
            data = data[: args.limit]
            logger.info(f"Limited to {args.limit} samples")

        texts = [d["claim"] for d in data]
        labels = [local_normalize_label_to_id(d["label"]) for d in data]

        # Build Knowledge Base for Retrieval
        from datetime import datetime, timezone

        for d in data:
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
                            }
                elif isinstance(ev, str):
                    parts.append(ev)
                    if ev.strip() and len(ev.strip()) > 10:
                        norm_key = normalize_text(ev.strip())
                        if norm_key not in unique_docs:
                            unique_docs[norm_key] = {
                                "text": ev.strip(),
                                "timestamp": None,
                            }
            gold_evidences.append("|||".join(parts))

    else:
        df = CSVLabeledLoader(args.csv).load()
        if args.limit:
            df = df.head(args.limit)
            logger.info(f"Limited to {args.limit} samples")

        texts = df["text"].tolist()
        raw_evidences = df["evidence"].tolist()
        labels = [local_normalize_label_to_id(v) for v in df["label"].tolist()]

        # Build Knowledge Base for Retrieval
        for evidence in raw_evidences:
            gold_evidences.append(evidence)
            if pd.isna(evidence):
                continue
            parts = str(evidence).split("|||")
            for part in parts:
                part = part.strip()
                if len(part) > 10:
                    norm_key = normalize_text(part)
                    if norm_key not in unique_docs:
                        unique_docs[norm_key] = {
                            "text": part,
                            "timestamp": None,
                        }

    kb_docs = list(unique_docs.values())
    logger.info(
        f"Built temporary KB with {len(kb_docs)} unique documents from test set evidence (deduplicated)"
    )

    kb_docs = list(unique_docs.values())
    logger.info(
        f"Built temporary KB with {len(kb_docs)} unique documents from test set evidence (deduplicated)"
    )

    # 2. Initialize Components
    # Retriever
    logger.info(f"Using retriever model: {retriever_model}")
    retriever = KnowledgeAugmentedRetriever(embedding_model=retriever_model, rrf_k=60)
    retriever.index_documents(kb_docs, text_field="text", timestamp_field="timestamp")

    # LLM Scorer
    llm = LLMScorer(
        model_name=args.lora_model,
        device=args.device,
        max_length=2048,
        labels=dynamic_label_list,
        prompt_template=PROMPT_TEMPLATE,
    )

    # 3. Running Inference
    results_summary = []

    # Store predictions
    preds_lora_retrieval = []
    preds_lora_gold = []
    preds_fusion_retrieval = []
    preds_fusion_gold = []

    # We need to process in batches
    num_samples = len(texts)

    # Pre-compute retrieval features for all samples (needed for Fusion)
    # Note: For Fusion + Gold, we still use retrieval features from retriever
    # but feed Gold Evidence to LLM.

    logger.info("Step 1/3: Running Retrieval...")
    all_retrieval_base = []
    all_retrieval_interactions = []
    all_retrieved_evidences = []
    all_evidence_raw = []  # raw texts for NLI

    for text in tqdm(texts, desc="Retrieving"):
        feats, interaction, retrieved_evidence_list = _build_retrieval_features(
            retriever, text, top_k
        )
        all_retrieval_base.append(feats)
        all_retrieval_interactions.append(interaction)
        all_retrieved_evidences.append(retrieved_evidence_list)
        all_evidence_raw.append(retrieved_evidence_list)

    # Append NLI features if checkpoint was trained with NLI
    all_retrieval_features = all_retrieval_base
    nli_model_name = fusion_config.get("nli_model") or None
    if nli_model_name:
        from src.models.nli_scorer import NLIScorer
        import gc as _gc

        logger.info(f"Running NLI scoring with {nli_model_name}...")
        _nli = NLIScorer(model_name=nli_model_name, device=args.device).load()

        flat_docs, flat_claims = [], []
        for text, evidences in zip(texts, all_evidence_raw):
            for doc in evidences:
                flat_docs.append(doc)
                flat_claims.append(text)

        nli_flat = _nli.score(premises=flat_docs, hypotheses=flat_claims)
        _nli.unload(); del _nli; _gc.collect()

        cursor = 0
        all_retrieval_features = []
        for base_feats, evidences in zip(all_retrieval_base, all_evidence_raw):
            n_real = len(evidences)
            nli_padded = np.full((top_k, 3), 1.0 / 3.0, dtype=np.float32)
            if n_real > 0:
                nli_padded[:n_real] = nli_flat[cursor : cursor + n_real]
            cursor += n_real
            all_retrieval_features.append(
                np.concatenate([base_feats, nli_padded], axis=-1)
            )
        logger.info(f"NLI merged. score_features=8, pairs={len(flat_docs)}")

    tensor_retrieval_features = torch.tensor(
        np.array(all_retrieval_features), dtype=torch.float32
    ).to(args.device)

    tensor_interactions = None
    _non_none = [x for x in all_retrieval_interactions if x is not None]
    if _non_none:
        emb_shape = _non_none[0].shape[0]
        _filled = [
            x if x is not None else np.zeros(emb_shape, dtype=np.float32)
            for x in all_retrieval_interactions
        ]
        tensor_interactions = torch.tensor(
            np.array(_filled, dtype=np.float32), dtype=torch.float32
        ).to(args.device)

    logger.info("Step 2/3: Running LLM Inference (Retrieval & Gold)...")

    # Only need logits for Fusion, but we can get probs/preds from logits too
    logits_retrieval = []
    logits_gold = []

    # Batch processing for LLM
    for i in tqdm(range(0, num_samples, args.batch_size), desc="LLM Scoring"):
        batch_texts = texts[i : i + args.batch_size]
        batch_gold_evidences = gold_evidences[i : i + args.batch_size]
        batch_retrieved_evidences = all_retrieved_evidences[i : i + args.batch_size]

        # A. Mode: Retrieval Evidence (Micro-batching)
        sub_logits_ret = []
        for j in range(0, len(batch_texts), args.llm_batch_size):
            sub_texts = batch_texts[j : j + args.llm_batch_size]
            sub_evs = batch_retrieved_evidences[j : j + args.llm_batch_size]
            # Truncate each retrieved evidence list to top 3 to reduce context noise (matches fusion_inference.py)
            sub_evs_truncated = [ev[:3] for ev in sub_evs]
            sub_logits_ret.append(llm.score_logits(sub_texts, sub_evs_truncated))

        if sub_logits_ret:
            logits_retrieval.append(torch.cat(sub_logits_ret, dim=0))

        # B. Mode: Gold Evidence (Micro-batching)
        sub_logits_gold = []
        for j in range(0, len(batch_texts), args.llm_batch_size):
            sub_texts = batch_texts[j : j + args.llm_batch_size]
            sub_evs = batch_gold_evidences[j : j + args.llm_batch_size]
            sub_logits_gold.append(llm.score_logits(sub_texts, sub_evs))

        if sub_logits_gold:
            logits_gold.append(torch.cat(sub_logits_gold, dim=0))

    tensor_logits_retrieval = torch.cat(logits_retrieval, dim=0).to(args.device)
    tensor_logits_gold = torch.cat(logits_gold, dim=0).to(args.device)

    logger.info("Step 3/3: Computing Metrics...")

    # --- Mode 1: LoRA + Retrieval ---
    # Preds = argmax(logits)
    preds_lora_retrieval = torch.argmax(tensor_logits_retrieval, dim=1).cpu().numpy()
    results_summary.append(
        calculate_metrics(
            labels, preds_lora_retrieval, "LoRA + Retrieval", dynamic_label_list
        )
    )

    # --- Mode 2: LoRA + Gold ---
    preds_lora_gold = torch.argmax(tensor_logits_gold, dim=1).cpu().numpy()
    results_summary.append(
        calculate_metrics(labels, preds_lora_gold, "LoRA + Gold", dynamic_label_list)
    )

    # --- Mode 3: Fusion + Retrieval ---
    # Fusion(logits_retrieval, features_retrieval)
    with torch.no_grad():
        encoded_feats = retrieval_encoder(tensor_retrieval_features, tensor_interactions)
        fusion_out_ret = fusion_layer(tensor_logits_retrieval, encoded_feats)
        # Final probs or logits? Fusion returns FusionOutput with final_probs
        # Use simple argmax on final_probs
        preds_fusion_retrieval = (
            torch.argmax(fusion_out_ret.final_probs, dim=1).cpu().numpy()
        )

    results_summary.append(
        calculate_metrics(
            labels, preds_fusion_retrieval, "Fusion + Retrieval", dynamic_label_list
        )
    )

    # --- Mode 4: Fusion + Gold ---
    # Fusion(logits_gold, features_retrieval)
    # Rationale: LLM sees Gold evidence (perfect context), but Fusion layer still sees
    # "how hard was it to retrieve info" features. This tests if Fusion improves
    # even with perfect LLM context (unlikely, but requested).
    with torch.no_grad():
        # Reuse encoded_feats from retrieval
        fusion_out_gold = fusion_layer(tensor_logits_gold, encoded_feats)
        preds_fusion_gold = (
            torch.argmax(fusion_out_gold.final_probs, dim=1).cpu().numpy()
        )

    results_summary.append(
        calculate_metrics(
            labels, preds_fusion_gold, "Fusion + Gold", dynamic_label_list
        )
    )

    # Summary Table
    print("\n" + "=" * 110)
    headers = [f"{'Mode':<18}"] + [
        f"{h:<8}"
        for h in ["Acc", "Prec", "Rec", "F1_Mac", "F1_W"]
        + [f"F1_{l}" for l in dynamic_label_list]
    ]
    print(" | ".join(headers))
    print("-" * 110)
    for res in results_summary:
        row = (
            [f"{res['Mode']:<18}"]
            + [
                f"{res['Accuracy']:<8.4f}",
                f"{res['Precision']:<8.4f}",
                f"{res['Recall']:<8.4f}",
                f"{res['F1_Macro']:<8.4f}",
                f"{res['F1_Weighted']:<8.4f}",
            ]
            + [f"{res[f'F1_{l}']:<8.4f}" for l in dynamic_label_list]
        )
        print(" | ".join(row))
    print("=" * 110 + "\n")

    logger.info("Done.")


if __name__ == "__main__":
    main()
