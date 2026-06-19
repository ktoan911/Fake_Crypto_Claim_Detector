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

import torch
import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

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
    gold_timestamps: Optional[List[List[datetime]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        feats_temporal           : float32 [N, top_k, score_features]  — with temporal scoring
        feats_no_temporal        : float32 [N, top_k, score_features]  — pure RRF only
        interactions_temporal    : float32 [N, interaction_dim] or None
        interactions_no_temporal : float32 [N, interaction_dim] or None
        retrieved_texts_temporal : List[List[str]]  — top-k retrieved texts per claim (temporal)

    gold_timestamps: real per-evidence timestamps from dataset; falls back to pseudo if None.
    Leave-one-out: each claim retrieves from KB built without its own evidence.
    """
    from src.retrieval.retrieval import KnowledgeAugmentedRetriever

    alpha = float(saved_config.get("alpha", 0.7))
    lambda_decay = float(saved_config.get("lambda_decay", 0.1))
    gamma = float(saved_config.get("gamma", 0.5))
    rrf_k = int(saved_config.get("rrf_k", 60))
    score_features = int(saved_config.get("score_features", 5))
    nli_model_name: Optional[str] = saved_config.get("nli_model") or None
    nli_top_k = min(5, top_k)

    logger.info("Phase 1 — building retriever & indexing KB …")
    retriever = KnowledgeAugmentedRetriever(
        embedding_model=retriever_model,
        alpha=alpha,
        lambda_decay=lambda_decay,
        gamma=gamma,
        use_query_expansion=True,
        rrf_k=rrf_k,
    )

    # Load NLI scorer if checkpoint was trained with NLI features
    nli_scorer = None
    if score_features > 5 and nli_model_name:
        try:
            from src.models.nli_scorer import NLIScorer
            nli_device = os.getenv("NLI_DEVICE", "cpu")
            nli_max_length = int(os.getenv("NLI_MAX_LENGTH", "256"))
            nli_scorer = NLIScorer(
                model_name=nli_model_name,
                device=nli_device,
                max_length=nli_max_length,
            )
            logger.info(f"  NLI scorer loaded: {nli_model_name} (device={nli_device})")
        except Exception as e:
            logger.warning(f"  NLI scorer failed to load ({e}); padding NLI dims with 1/3")

    # Build per-evidence timestamps: use real ones when available, else pseudo-random
    if gold_timestamps is not None:
        flat_timestamps: List[datetime] = [
            ts for tss in gold_timestamps for ts in tss
        ]
        logger.info("  Using real timestamps from dataset")
    else:
        flat_timestamps = _pseudo_timestamps(
            sum(len(ev) for ev in gold_evidences), days_spread=365
        )
        logger.info("  Using pseudo-random timestamps (no real timestamps provided)")

    # Build docs with structured IDs: "c{claim_idx}_e{ev_idx}" for leave-one-out filtering
    all_docs = []
    for claim_idx, ev_list in enumerate(gold_evidences):
        for ev_idx, ev_text in enumerate(ev_list):
            flat_pos = sum(len(gold_evidences[j]) for j in range(claim_idx)) + ev_idx
            all_docs.append(
                {
                    "id": f"c{claim_idx}_e{ev_idx}",
                    "text": str(ev_text),
                    "timestamp": flat_timestamps[flat_pos],
                }
            )

    retriever.index_documents(all_docs)
    logger.info(f"  Indexed {len(all_docs)} passages")

    # Max own-evidence a claim can have (used to over-fetch for leave-one-out filtering)
    max_own_ev = max(len(ev) for ev in gold_evidences) if gold_evidences else 0

    def _feats(
        claim_idx: int, claim: str, use_temporal: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """Returns (score_feats [top_k, score_features], interaction or None, retrieved_texts).

        Leave-one-out: over-fetches then strips documents belonging to claim_idx's own evidence,
        so the retriever cannot trivially return the answer from the KB.
        """
        own_prefix = f"c{claim_idx}_"
        fetch_k = top_k + max_own_ev + 5  # extra buffer for leave-one-out filtering
        raw_results = retriever.retrieve(claim, top_k=fetch_k, use_temporal=use_temporal)
        results = [r for r in raw_results if not r.document_id.startswith(own_prefix)][:top_k]

        doc_embs = []
        doc_texts = []
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
            if r.text:
                doc_texts.append(r.text)
        while len(rows) < top_k:
            rows.append([0.0, 0.0, 0.0, 0.0, 0.0])
        score_arr = np.array(rows[:top_k], dtype=np.float32)

        # Append NLI features if the checkpoint was trained with them
        if score_features > 5:
            nli_padded = np.full((top_k, 3), 1.0 / 3.0, dtype=np.float32)
            if nli_scorer is not None and doc_texts:
                real_docs = doc_texts[:nli_top_k]
                try:
                    nli_scores = nli_scorer.score(
                        premises=real_docs,
                        hypotheses=[claim] * len(real_docs),
                    )
                    nli_padded[: len(real_docs)] = nli_scores
                except Exception as e:
                    logger.warning(f"  NLI scoring failed ({e}); using neutral 1/3")
            score_arr = np.concatenate([score_arr, nli_padded], axis=-1)

        interaction = None
        q_emb = getattr(retriever, "_last_query_embedding", None)
        if q_emb is not None and doc_embs:
            mean_d = np.array(doc_embs, dtype=np.float32).mean(axis=0)
            interaction = np.concatenate(
                [q_emb * mean_d, np.abs(q_emb - mean_d)], dtype=np.float32
            )
        return score_arr, interaction, doc_texts

    # Determine interaction_dim from first sample
    _s0, _i0, _ = _feats(0, claims[0], use_temporal=True)
    interaction_dim = _i0.shape[0] if _i0 is not None else 0

    feats_temporal = np.zeros((len(claims), top_k, score_features), dtype=np.float32)
    feats_no_temporal = np.zeros((len(claims), top_k, score_features), dtype=np.float32)
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
    retrieved_texts_temporal: List[List[str]] = []

    for i, claim in enumerate(tqdm(claims, desc="Phase 1 — retrieval")):
        s_t, i_t, texts_t = _feats(i, claim, use_temporal=True)
        s_nt, i_nt, _ = _feats(i, claim, use_temporal=False)
        feats_temporal[i] = s_t
        feats_no_temporal[i] = s_nt
        retrieved_texts_temporal.append(texts_t)
        if interactions_temporal is not None and i_t is not None:
            interactions_temporal[i] = i_t
        if interactions_no_temporal is not None and i_nt is not None:
            interactions_no_temporal[i] = i_nt

    _free(retriever)
    if nli_scorer is not None:
        _free(nli_scorer)
    logger.info("Phase 1 done — retriever freed")
    return (
        feats_temporal,
        feats_no_temporal,
        interactions_temporal,
        interactions_no_temporal,
        retrieved_texts_temporal,
    )


# ---------------------------------------------------------------------------
# Phase 2: LLM — compute logits, then free LLM
# ---------------------------------------------------------------------------


def phase2_llm(
    claims: List[str],
    evidence_lists: List[List[str]],
    lora_model: str,
    device: str,
    llm_top_k: int,
    llm_batch_size: int,
) -> np.ndarray:
    """
    Returns llm_logits: float32 [N, num_classes]

    evidence_lists: retrieved evidence texts (from phase1), keeping LLM and retriever
    consistent — both branches now see the same input.
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
        batch_evs = [evidence_lists[i][:llm_top_k] for i in range(start, end)]

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

    if str(device) != "cpu" and hasattr(torch, "compile"):
        try:
            enc = torch.compile(enc)
            fus = torch.compile(fus)
            logger.info("  torch.compile enabled for fusion models (H200 optimization)")
        except Exception as e:
            logger.warning(f"  torch.compile failed (non-fatal): {e}")

    return enc, fus


def _eval_config(
    config_name: str,
    feats_np: np.ndarray,
    llm_logits_np: np.ndarray,
    y_true: List[int],
    enc: RetrievalFeatureEncoder,
    fus: ConfidenceAwareFusion,
    device: str,
    batch_size: int = 512,
    interactions_np: Optional[np.ndarray] = None,
) -> dict:
    n = len(y_true)
    preds: List[int] = []
    _use_amp = str(device) != "cpu" and torch.cuda.is_available()

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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=_use_amp):
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
    fusion_batch_size: int = 512,
) -> List[dict]:
    score_features = int(saved_config.get("score_features", 5))

    # Fusion model is tiny (~few MB): keep it for all configs, one at a time
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

    # NLI Re-ranking ablation: replace NLI feature columns with uniform prior (1/3).
    # Only meaningful when the checkpoint was trained with NLI features (score_features > 5).
    if score_features > 5:
        feats_no_nli = feats_temporal.copy()
        feats_no_nli[:, :, 5:] = 1.0 / 3.0
        ablation_configs.append(
            {
                "name": "w/o NLI Re-ranking",
                "feats": feats_no_nli,
                "interactions": interactions_temporal,
                "adaptive_beta": None,
            }
        )

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
            batch_size=fusion_batch_size,
        )
        results.append(row)
        _free(enc)
        _free(fus)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import json as _json

    parser = argparse.ArgumentParser(
        description="Ablation Study — Temporal Scoring & Beta-Gate (memory-efficient)"
    )
    parser.add_argument("--csv", default=None, help="CSV with claim/label/evidence columns")
    parser.add_argument(
        "--json", default=None,
        help="JSON dataset (preferred): preserves real evidence timestamps for temporal ablation",
    )
    parser.add_argument("--fusion_model", default=_DEFAULT_FUSION_MODEL)
    parser.add_argument("--lora_model", default=_DEFAULT_LLM_MODEL)
    parser.add_argument("--retriever_model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--llm_evidence_top_k", type=int, default=3)
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=int(os.getenv("LLM_INFER_BATCH_SIZE", "64" if torch.cuda.is_available() else "4")),
        help="LLM inference batch size (default 64 on GPU, 4 on CPU)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--fusion_batch_size",
        type=int,
        default=512,
        help="Fusion inference batch size (default 512 — tiny model, H200 handles large batches)",
    )
    parser.add_argument("--output", default="results/ablation_results.csv")
    args = parser.parse_args()

    if args.json is None and args.csv is None:
        args.csv = "data/test.csv"

    # H200 / Ampere+ optimizations
    if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 matmul + cuDNN precision for H200 GPU.")

    gold_timestamps: Optional[List[List[datetime]]] = None

    if args.json:
        # JSON path: extract real timestamps (Fix 1)
        logger.info(f"Loading {args.json}")
        with open(args.json, encoding="utf-8") as f:
            raw_data = _json.load(f)
        if args.limit:
            import random as _random
            _random.seed(42)
            raw_data = _random.sample(raw_data, min(args.limit, len(raw_data)))

        claims: List[str] = [row["claim"] for row in raw_data]
        y_true: List[int] = [_normalize_label(row["label"]) for row in raw_data]
        gold_evidences: List[List[str]] = []
        gold_timestamps = []
        for row in raw_data:
            texts, tss = [], []
            for e in row.get("evidence", []):
                if isinstance(e, dict):
                    texts.append(e.get("content", str(e)))
                    raw_ts = e.get("timestamp")
                    if raw_ts:
                        try:
                            ts = datetime.fromisoformat(str(raw_ts)).replace(tzinfo=timezone.utc)
                        except ValueError:
                            ts = datetime.now(timezone.utc)
                    else:
                        ts = datetime.now(timezone.utc)
                else:
                    texts.append(str(e))
                    ts = datetime.now(timezone.utc)
                tss.append(ts)
            gold_evidences.append(texts)
            gold_timestamps.append(tss)
    else:
        # CSV fallback (no real timestamps)
        logger.info(f"Loading {args.csv}")
        df = pd.read_csv(args.csv)
        if args.limit:
            df = df.sample(n=min(args.limit, len(df)), random_state=42).reset_index(drop=True)
        claims = df["claim"].tolist()
        y_true = [_normalize_label(lbl) for lbl in df["label"].tolist()]
        gold_evidences = [_parse_evidence(e) for e in df["evidence"].tolist()]

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
        retrieved_texts_temporal,
    ) = phase1_retrieval(
        claims=claims,
        gold_evidences=gold_evidences,
        retriever_model=retriever_model,
        saved_config=saved_config,
        top_k=top_k,
        gold_timestamps=gold_timestamps,
    )

    # ---- Phase 2: LLM (Qwen3-4B) — uses retrieved evidence (Fix 3: consistent inputs) ----
    llm_logits = phase2_llm(
        claims=claims,
        evidence_lists=retrieved_texts_temporal,
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
        fusion_batch_size=args.fusion_batch_size,
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
