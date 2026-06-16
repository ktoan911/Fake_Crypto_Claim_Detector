#!/usr/bin/env python3
"""
Ablation Study: Temporal Scoring & Beta-Gate
Compares F1-score across 3 configurations:
  1. w/o Temporal Scoring  — retrieval score = pure RRF, no recency/cyclicity
  2. w/o Beta-Gate         — fusion uses fixed β (adaptive_beta=False)
  3. Full Model            — temporal scoring + adaptive β gate

Memory-efficient design (3 sequential phases, never 2 large models at once):
  Phase 1 — Retrieval: load retriever → compute features for both temporal configs → delete
  Phase 2 — LLM:       load LLM       → compute logits for all claims             → delete
  Phase 3 — Fusion:    load fusion (tiny) → run 3 configs against cached tensors   → done

Usage:
  python tests/ablation_study.py \\
      --csv data/test.csv \\
      --fusion_model ktoan911/fact-check-fusion-model \\
      --lora_model  ktoan911/Qwen3-4B-factcheck-finetune \\
      [--limit 300] [--llm_batch_size 8] [--device cpu] [--output results/ablation.csv]
"""

from __future__ import annotations

import argparse
import ast
import gc
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import LABEL_LIST, PROMPT_TEMPLATE
from src.models.fusion import ConfidenceAwareFusion, RetrievalFeatureEncoder

_DEFAULT_LLM_MODEL = os.getenv("LLM_FINETUNE", "ktoan911/Qwen3-4B-factcheck-finetune")
_DEFAULT_FUSION_MODEL = os.getenv("FUSION_MODEL", "ktoan911/fact-check-fusion-model")
_DEFAULT_RETRIEVER_MODEL = os.getenv("RETRIEVER_MODEL", "AITeamVN/Vietnamese_Embedding")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free(obj) -> None:
    """Delete object, collect garbage, empty CUDA cache."""
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_fusion_path(
    path_or_repo: str, filename: str = "acf_fusion_model.pt"
) -> str:
    if os.path.isfile(path_or_repo):
        return path_or_repo
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id=path_or_repo, filename=filename)
    logger.info(f"Downloaded {filename} → {local}")
    return local


def _normalize_label(val) -> int:
    """Map raw label strings -> integer index into LABEL_LIST = ["A", "B", "C"].

    LABEL_LIST (src/config.py):
      A (index 0) = Supported  / Đúng
      B (index 1) = Contradicted / Sai
      C (index 2) = Insufficient evidence / Chưa chắc chắn
    """
    v = str(val).strip()
    # Direct A/B/C mapping (current format in config.py)
    if v.upper() in ("A",):
        return 0
    if v.upper() in ("B",):
        return 1
    if v.upper() in ("C",):
        return 2
    # Numeric string
    if v == "0":
        return 0
    if v == "1":
        return 1
    if v == "2":
        return 2
    # Legacy Vietnamese/English labels (backward compat with old CSVs)
    v_lower = v.lower()
    if v_lower in {"true", "supported", "legit", "\u0111u\u1ed3ng", "dung"}:
        return 0  # A
    if v_lower in {"false", "refuted", "scam", "fake", "sai"}:
        return 1  # B
    return 2      # C = default (Insufficient)


def _parse_evidence(field) -> List[str]:
    if isinstance(field, list):
        return [str(x) for x in field]
    try:
        parsed = ast.literal_eval(str(field))
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [str(field)]


def _pseudo_timestamps(n: int, days_spread: int = 365) -> List[datetime]:
    now = datetime.now(timezone.utc)
    rng = random.Random(42)
    return [
        now - timedelta(days=rng.randint(0, days_spread), hours=rng.randint(0, 23))
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Phase 1: Retrieval — compute features, then free retriever
# ---------------------------------------------------------------------------


def phase1_retrieval(
    claims: List[str],
    gold_evidences: List[List[str]],
    retriever_model: str,
    saved_config: dict,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        feats_temporal     : float32 [N, top_k, 4]  — with temporal scoring
        feats_no_temporal  : float32 [N, top_k, 4]  — pure RRF only
    """
    from src.retrieval.retrieval import KnowledgeAugmentedRetriever

    alpha = float(saved_config.get("alpha", 0.7))
    lambda_decay = float(saved_config.get("lambda_decay", 0.1))
    gamma = float(saved_config.get("gamma", 0.5))
    rrf_k = int(saved_config.get("rrf_k", 60))

    logger.info("Phase 1 — building retriever & indexing KB …")
    retriever = KnowledgeAugmentedRetriever(
        embedding_model=retriever_model,
        alpha=alpha,
        lambda_decay=lambda_decay,
        gamma=gamma,
        use_query_expansion=True,
        rrf_k=rrf_k,
    )

    # Index gold evidence with pseudo-timestamps for realistic temporal scoring
    all_docs = []
    timestamps = _pseudo_timestamps(
        sum(len(ev) for ev in gold_evidences), days_spread=365
    )
    ts_cursor = 0
    for ev_list in gold_evidences:
        for ev_text in ev_list:
            all_docs.append(
                {
                    "id": f"d{ts_cursor}",
                    "text": str(ev_text),
                    "timestamp": timestamps[ts_cursor],
                }
            )
            ts_cursor += 1

    retriever.index_documents(all_docs)
    logger.info(f"  Indexed {len(all_docs)} passages")

    def _feats(claim: str, use_temporal: bool):
        """Returns (score_feats [top_k, 5], interaction [2*emb_dim] or None)."""
        results = retriever.retrieve(claim, top_k=top_k, use_temporal=use_temporal)
        doc_embs = []
        rows = []
        for r in results:
            rows.append(
                [
                    r.score,
                    r.rrf_score,
                    r.recency_score,
                    r.cyclicity_score,
                    r.cosine_similarity,
                ]
            )
            if r.embedding is not None:
                doc_embs.append(r.embedding)
        while len(rows) < top_k:
            rows.append([0.0, 0.0, 0.0, 0.0, 0.0])
        score_arr = np.array(rows[:top_k], dtype=np.float32)

        interaction = None
        q_emb = getattr(retriever, "_last_query_embedding", None)
        if q_emb is not None and doc_embs:
            mean_d = np.array(doc_embs, dtype=np.float32).mean(axis=0)
            interaction = np.concatenate(
                [q_emb * mean_d, np.abs(q_emb - mean_d)], dtype=np.float32
            )
        return score_arr, interaction

    # Determine interaction_dim from first sample
    _s0, _i0 = _feats(claims[0], use_temporal=True)
    interaction_dim = _i0.shape[0] if _i0 is not None else 0

    feats_temporal = np.zeros((len(claims), top_k, 5), dtype=np.float32)
    feats_no_temporal = np.zeros((len(claims), top_k, 5), dtype=np.float32)
    interactions_temporal = (
        np.zeros((len(claims), interaction_dim), dtype=np.float32)
        if interaction_dim > 0
        else None
    )
    interactions_no_temporal = (
        np.zeros((len(claims), interaction_dim), dtype=np.float32)
        if interaction_dim > 0
        else None
    )

    for i, claim in enumerate(tqdm(claims, desc="Phase 1 — retrieval")):
        s_t, i_t = _feats(claim, use_temporal=True)
        s_nt, i_nt = _feats(claim, use_temporal=False)
        feats_temporal[i] = s_t
        feats_no_temporal[i] = s_nt
        if interactions_temporal is not None and i_t is not None:
            interactions_temporal[i] = i_t
        if interactions_no_temporal is not None and i_nt is not None:
            interactions_no_temporal[i] = i_nt

    _free(retriever)
    logger.info("Phase 1 done — retriever freed")
    return (
        feats_temporal,
        feats_no_temporal,
        interactions_temporal,
        interactions_no_temporal,
    )


# ---------------------------------------------------------------------------
# Phase 2: LLM — compute logits, then free LLM
# ---------------------------------------------------------------------------


def phase2_llm(
    claims: List[str],
    gold_evidences: List[List[str]],
    lora_model: str,
    device: str,
    llm_top_k: int,
    llm_batch_size: int,
) -> np.ndarray:
    """
    Returns llm_logits: float32 [N, num_classes]
    """
    from src.llm_scorer import LLMScorer

    logger.info(f"Phase 2 — loading LLM: {lora_model}")
    llm = LLMScorer(
        model_name=lora_model,
        device=device,
        labels=LABEL_LIST,
        prompt_template=PROMPT_TEMPLATE,
    )

    n = len(claims)
    num_classes = len(LABEL_LIST)
    all_logits = np.zeros((n, num_classes), dtype=np.float32)

    for start in tqdm(range(0, n, llm_batch_size), desc="Phase 2 — LLM scoring"):
        end = min(start + llm_batch_size, n)
        batch_claims = claims[start:end]
        batch_evs = [gold_evidences[i][:llm_top_k] for i in range(start, end)]

        with torch.inference_mode():
            logits = llm.score_logits(batch_claims, batch_evs)
            all_logits[start:end] = logits.cpu().float().numpy()

    _free(llm)
    logger.info("Phase 2 done — LLM freed")
    return all_logits


# ---------------------------------------------------------------------------
# Phase 3: Fusion — 3 configs, tiny model, use cached tensors
# ---------------------------------------------------------------------------


def _build_fusion(
    ckpt: dict,
    saved_config: dict,
    device: str,
    adaptive_beta: Optional[bool],
) -> Tuple[RetrievalFeatureEncoder, ConfidenceAwareFusion]:
    fusion_state = ckpt.get("fusion", {})
    has_gate = any(k.startswith("beta_gate.") for k in fusion_state.keys())
    effective = has_gate if adaptive_beta is None else adaptive_beta

    num_classes = saved_config.get("num_classes", len(LABEL_LIST))
    top_k = saved_config.get("top_k", 10)

    enc = RetrievalFeatureEncoder(
        num_retrieved=top_k,
        score_features=int(saved_config.get("score_features", 5)),
        hidden_dim=64,
        output_dim=64,
        interaction_dim=int(saved_config.get("interaction_dim", 0)),
    ).to(device)
    enc.load_state_dict(ckpt["retrieval_encoder"])
    enc.eval()

    fus = ConfidenceAwareFusion(
        retrieval_input_dim=64,
        hidden_dim=128,
        num_classes=num_classes,
        initial_beta=float(saved_config.get("initial_beta", 0.8)),
        lambda_reg=float(saved_config.get("lambda_reg", 0.01)),
        normalize_branch_logits=bool(
            saved_config.get("normalize_branch_logits", False)
        ),
        adaptive_beta=effective,
    ).to(device)

    if not effective and has_gate:
        # Drop beta_gate.* keys — simulate fixed-β (weights other than gate are loaded)
        compatible = {
            k: v for k, v in fusion_state.items() if not k.startswith("beta_gate.")
        }
        fus.load_state_dict(compatible, strict=False)
    else:
        fus.load_state_dict(fusion_state, strict=(effective == has_gate))

    fus.eval()
    return enc, fus


def _eval_config(
    config_name: str,
    feats_np: np.ndarray,
    llm_logits_np: np.ndarray,
    y_true: List[int],
    enc: RetrievalFeatureEncoder,
    fus: ConfidenceAwareFusion,
    device: str,
    batch_size: int = 64,
    interactions_np: Optional[np.ndarray] = None,
) -> dict:
    n = len(y_true)
    preds: List[int] = []

    for start in tqdm(range(0, n, batch_size), desc=config_name):
        end = min(start + batch_size, n)
        feat_t = torch.tensor(feats_np[start:end], dtype=torch.float32, device=device)
        llm_t = torch.tensor(
            llm_logits_np[start:end], dtype=torch.float32, device=device
        )
        int_t = (
            torch.tensor(interactions_np[start:end], dtype=torch.float32, device=device)
            if interactions_np is not None
            else None
        )

        with torch.inference_mode():
            enc_out = enc(feat_t, int_t)
            fus_out = fus(llm_t, enc_out)
            batch_preds = torch.argmax(fus_out.final_probs, dim=-1).cpu().tolist()

        preds.extend(batch_preds)

    label_ids = list(range(len(LABEL_LIST)))
    f1_macro = f1_score(
        y_true, preds, average="macro", zero_division=0, labels=label_ids
    )
    f1_weighted = f1_score(
        y_true, preds, average="weighted", zero_division=0, labels=label_ids
    )
    per_class = f1_score(y_true, preds, average=None, zero_division=0, labels=label_ids)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(
        y_true, preds, average="macro", zero_division=0, labels=label_ids
    )
    rec = recall_score(
        y_true, preds, average="macro", zero_division=0, labels=label_ids
    )

    row: dict = {
        "Configuration": config_name,
        "Accuracy": round(acc, 4),
        "Precision (macro)": round(prec, 4),
        "Recall (macro)": round(rec, 4),
        "F1 (macro)": round(f1_macro, 4),
        "F1 (weighted)": round(f1_weighted, 4),
    }
    for i, lbl in enumerate(LABEL_LIST):
        if i < len(per_class):
            row[f"F1_{lbl}"] = round(float(per_class[i]), 4)

    logger.info(
        f"[{config_name}] F1-macro={f1_macro:.4f} | "
        f"F1-weighted={f1_weighted:.4f} | Accuracy={acc:.4f}"
    )
    return row


def phase3_fusion(
    ckpt: dict,
    saved_config: dict,
    feats_temporal: np.ndarray,
    feats_no_temporal: np.ndarray,
    interactions_temporal: "Optional[np.ndarray]",
    interactions_no_temporal: "Optional[np.ndarray]",
    llm_logits: np.ndarray,
    y_true: List[int],
    device: str,
) -> List[dict]:
    # Fusion model is tiny (~few MB): keep it for all 3 configs, one at a time
    ablation_configs = [
        {
            "name": "w/o Temporal Scoring",
            "feats": feats_no_temporal,
            "interactions": interactions_no_temporal,
            "adaptive_beta": None,
        },
        {
            "name": "w/o Beta-Gate (Fixed β)",
            "feats": feats_temporal,
            "interactions": interactions_temporal,
            "adaptive_beta": False,
        },
        {
            "name": "Full Model",
            "feats": feats_temporal,
            "interactions": interactions_temporal,
            "adaptive_beta": None,
        },
    ]

    results = []
    for cfg in ablation_configs:
        logger.info(f"\n{'=' * 60}\nPhase 3 — config: {cfg['name']}\n{'=' * 60}")
        enc, fus = _build_fusion(ckpt, saved_config, device, cfg["adaptive_beta"])
        row = _eval_config(
            config_name=cfg["name"],
            feats_np=cfg["feats"],
            interactions_np=cfg["interactions"],
            llm_logits_np=llm_logits,
            y_true=y_true,
            enc=enc,
            fus=fus,
            device=device,
        )
        results.append(row)
        _free(enc)
        _free(fus)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation Study — Temporal Scoring & Beta-Gate (memory-efficient)"
    )
    parser.add_argument("--csv", default="data/test.csv")
    parser.add_argument("--fusion_model", default=_DEFAULT_FUSION_MODEL)
    parser.add_argument("--lora_model", default=_DEFAULT_LLM_MODEL)
    parser.add_argument("--retriever_model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--llm_evidence_top_k", type=int, default=3)
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=int(os.getenv("LLM_INFER_BATCH_SIZE", "4")),
        help="LLM inference batch size (default 4; lower = less RAM)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", default="results/ablation_results.csv")
    args = parser.parse_args()

    # Load test CSV
    logger.info(f"Loading {args.csv}")
    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42).reset_index(
            drop=True
        )

    claims: List[str] = df["claim"].tolist()
    y_true: List[int] = [_normalize_label(lbl) for lbl in df["label"].tolist()]
    gold_evidences: List[List[str]] = [
        _parse_evidence(e) for e in df["evidence"].tolist()
    ]
    logger.info(
        f"{len(claims)} samples | labels: {pd.Series(y_true).value_counts().to_dict()}"
    )

    # Resolve paths once
    fusion_path = _resolve_fusion_path(args.fusion_model)
    ckpt = torch.load(fusion_path, map_location="cpu", weights_only=False)
    saved_config: dict = ckpt.get("config", {})
    top_k: int = saved_config.get("top_k", 10)

    retriever_model = (
        args.retriever_model
        or saved_config.get("retriever_model")
        or _DEFAULT_RETRIEVER_MODEL
    )

    # ---- Phase 1: Retrieval (SentenceTransformer + FAISS + BM25) ----
    (
        feats_temporal,
        feats_no_temporal,
        interactions_temporal,
        interactions_no_temporal,
    ) = phase1_retrieval(
        claims=claims,
        gold_evidences=gold_evidences,
        retriever_model=retriever_model,
        saved_config=saved_config,
        top_k=top_k,
    )

    # ---- Phase 2: LLM (Qwen3-4B) ----
    llm_logits = phase2_llm(
        claims=claims,
        gold_evidences=gold_evidences,
        lora_model=args.lora_model,
        device=args.device,
        llm_top_k=args.llm_evidence_top_k,
        llm_batch_size=args.llm_batch_size,
    )

    # ---- Phase 3: Fusion (tiny) × 3 configs ----
    all_results = phase3_fusion(
        ckpt=ckpt,
        saved_config=saved_config,
        feats_temporal=feats_temporal,
        feats_no_temporal=feats_no_temporal,
        interactions_temporal=interactions_temporal,
        interactions_no_temporal=interactions_no_temporal,
        llm_logits=llm_logits,
        y_true=y_true,
        device=args.device,
    )

    # ---- Print & save ----
    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("ABLATION STUDY — F1-Score Comparison")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)

    best_idx = results_df["F1 (macro)"].idxmax()
    print(
        f"\nBest F1 (macro): {results_df.loc[best_idx, 'Configuration']}"
        f" → {results_df.loc[best_idx, 'F1 (macro)']:.4f}"
    )

    full_f1_rows = results_df.loc[
        results_df["Configuration"] == "Full Model", "F1 (macro)"
    ]
    if not full_f1_rows.empty:
        full_f1 = full_f1_rows.values[0]
        print("\nΔ F1 (macro) vs Full Model:")
        for _, row in results_df.iterrows():
            if row["Configuration"] != "Full Model":
                delta = row["F1 (macro)"] - full_f1
                sign = "+" if delta >= 0 else ""
                print(f"  {row['Configuration']}: {sign}{delta:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    results_df.to_csv(args.output, index=False)
    logger.info(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
