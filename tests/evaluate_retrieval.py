#!/usr/bin/env python3
"""
Đánh giá hệ thống Retrieval trên bộ dữ liệu 10k_finance_news_dataset.csv
Tối ưu cho GPU mạnh (H200 / A100) — toàn bộ corpus 10 000 bài báo.

═══════════════════════════════════════════════════════════
Setup
  Corpus      : 10 000 bài báo tài chính (indexed bằng "content")
  Query       : "title" của mỗi bài báo
  Ground truth: "content" tương ứng với title (relevant document)

Ablation modes
  1. bm25_only           – BM25 thuần tuý, không có semantic / temporal
  2. semantic_only       – Dense (FAISS) thuần tuý, không temporal
  3. hybrid_rrf          – BM25 + Dense kết hợp RRF, không temporal
  4. hybrid_rrf_temporal – Full system: RRF + Temporal scoring

Metrics
  Hit Rate@K  MRR@K  Recall@K  NDCG@K  Precision@K  MAP@K
═══════════════════════════════════════════════════════════

Chạy nhanh (BM25 only, 200 queries):
  python tests/evaluate_retrieval.py --modes bm25_only --n_queries 200

Chạy đầy đủ trên H200 (mặc định):
  python tests/evaluate_retrieval.py

Lưu FAISS index để chạy lại nhanh:
  python tests/evaluate_retrieval.py --save_index tests/eval_results/faiss_10k.pkl
  python tests/evaluate_retrieval.py --index_path  tests/eval_results/faiss_10k.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── project root (phải trước mọi import nặng) ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Early GPU guard ────────────────────────────────────────────────────────────
# Nếu user truyền --device cpu hoặc --no_semantic, ẩn CUDA TRƯỚC khi torch load
# để SentenceTransformer không cố mount model lên GPU (tránh OOM khi GPU bận)
def _early_device_patch() -> None:
    hide = False
    if "--no_semantic" in sys.argv:
        hide = True
    if "--device" in sys.argv:
        idx = sys.argv.index("--device")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1].lower() == "cpu":
            hide = True
    if hide:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_early_device_patch()

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.retrieval.retrieval import KnowledgeAugmentedRetriever


# ══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def _hit(ids: List[str], rel: str, k: int) -> float:
    return float(rel in ids[:k])

def _mrr(ids: List[str], rel: str, k: int) -> float:
    for r, d in enumerate(ids[:k], 1):
        if d == rel:
            return 1.0 / r
    return 0.0

def _ndcg(ids: List[str], rel: str, k: int) -> float:
    for r, d in enumerate(ids[:k], 1):
        if d == rel:
            return 1.0 / np.log2(r + 1)
    return 0.0

def _prec(ids: List[str], rel: str, k: int) -> float:
    return sum(1 for d in ids[:k] if d == rel) / k

def _map(ids: List[str], rel: str, k: int) -> float:
    for r, d in enumerate(ids[:k], 1):
        if d == rel:
            return _prec(ids, rel, r)
    return 0.0

def compute_metrics(retrieved_ids: List[str], relevant_id: str, k_values: List[int]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for k in k_values:
        result[f"hit@{k}"]    = _hit(retrieved_ids, relevant_id, k)
        result[f"mrr@{k}"]    = _mrr(retrieved_ids, relevant_id, k)
        result[f"recall@{k}"] = _hit(retrieved_ids, relevant_id, k)   # single-relevant: recall = hit
        result[f"ndcg@{k}"]   = _ndcg(retrieved_ids, relevant_id, k)
        result[f"prec@{k}"]   = _prec(retrieved_ids, relevant_id, k)
        result[f"map@{k}"]    = _map(retrieved_ids, relevant_id, k)
    return result

def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(csv_path: str, max_rows: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["title", "content"])
    df["title"]   = df["title"].astype(str).str.strip()
    df["content"] = df["content"].astype(str).str.strip()
    df = df[(df["title"] != "") & (df["content"] != "")].reset_index(drop=True)
    if max_rows and max_rows < len(df):
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        logger.info(f"Sampled {max_rows} rows from dataset")
    logger.info(f"Dataset: {len(df)} rows | cols: {df.columns.tolist()}")
    return df


def build_documents(df: pd.DataFrame) -> List[Dict]:
    docs = []
    for idx, row in df.iterrows():
        ts = pd.to_datetime(row.get("time"), errors="coerce")
        if pd.isnull(ts):
            ts = pd.Timestamp.now()
        docs.append({
            "id":        str(idx),
            "text":      row["content"],
            "timestamp": ts,
            "title":     row["title"],
            "category":  row.get("category", ""),
            "url":       row.get("url", ""),
        })
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation configs (ablation)
# ══════════════════════════════════════════════════════════════════════════════

EVAL_CONFIGS: Dict[str, Dict] = {
    "bm25_only": dict(
        use_semantic=False,
        use_temporal=False,
        expand_query=False,
    ),
    "semantic_only": dict(
        use_semantic=True,
        use_temporal=False,
        expand_query=False,
    ),
    "hybrid_rrf": dict(
        use_semantic=True,
        use_temporal=False,
        expand_query=True,
    ),
    "hybrid_rrf_temporal": dict(
        use_semantic=True,
        use_temporal=True,
        expand_query=True,
        use_nli=False,
    ),
    "hybrid_rrf_temporal_nli": dict(
        use_semantic=True,
        use_temporal=True,
        expand_query=True,
        use_nli=True,
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Core evaluation runner
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    retriever: KnowledgeAugmentedRetriever,
    queries: List[Tuple[str, str]],
    mode_name: str,
    retrieve_kwargs: Dict,
    k_values: List[int],
    top_k: int,
    log_every: int = 500,
    nli_scorer: Optional[Any] = None,
) -> Dict[str, float]:

    logger.info(f"\n{'='*64}")
    logger.info(f"  MODE: {mode_name}")
    logger.info(f"  retrieve kwargs: {retrieve_kwargs}")
    logger.info(f"  queries={len(queries)} | K={k_values} | top_k={top_k}")
    logger.info(f"{'='*64}")

    all_metrics: List[Dict[str, float]] = []
    latencies: List[float] = []
    failures = 0

    for qi, (query_text, relevant_id) in enumerate(
        tqdm(queries, desc=f"[{mode_name}]", ncols=100), 1
    ):
        try:
            t0 = time.perf_counter()
            # Bỏ use_nli ra khỏi kwargs truyền cho retriever (vì retriever không biết tham số này)
            use_nli = retrieve_kwargs.pop("use_nli", False)
            results = retriever.retrieve(query=query_text, top_k=top_k, **retrieve_kwargs)
            # Khôi phục lại dict nếu loop sau cần dùng
            retrieve_kwargs["use_nli"] = use_nli

            if use_nli and nli_scorer is not None and results:
                premises = [r.text for r in results]
                hypotheses = [query_text] * len(results)
                nli_probs = nli_scorer.score(premises, hypotheses)
                for i, r in enumerate(results):
                    # NLI prob ở index 0 là entailment. Cộng thẳng vào score (hoặc có thể tuỳ chỉnh weight)
                    r.score = r.score + float(nli_probs[i, 0])
                results.sort(key=lambda x: x.score, reverse=True)

            latencies.append(time.perf_counter() - t0)
            retrieved_ids = [r.document_id for r in results]
            all_metrics.append(compute_metrics(retrieved_ids, relevant_id, k_values))
        except Exception as exc:
            logger.warning(f"  ⚠ query#{qi} failed: {exc}")
            failures += 1
            all_metrics.append(compute_metrics([], relevant_id, k_values))

        if qi % log_every == 0:
            partial = aggregate(all_metrics)
            k_primary = max(k_values)
            logger.info(
                f"  [{mode_name}] {qi}/{len(queries)} | "
                f"hit@{k_primary}={partial.get(f'hit@{k_primary}', 0):.4f} | "
                f"mrr@{k_primary}={partial.get(f'mrr@{k_primary}', 0):.4f} | "
                f"avg_lat={np.mean(latencies)*1000:.1f}ms"
            )

    agg = aggregate(all_metrics)
    agg["avg_latency_ms"] = float(np.mean(latencies) * 1000) if latencies else 0.0
    agg["p50_latency_ms"] = float(np.percentile(latencies, 50) * 1000) if latencies else 0.0
    agg["p95_latency_ms"] = float(np.percentile(latencies, 95) * 1000) if latencies else 0.0
    agg["p99_latency_ms"] = float(np.percentile(latencies, 99) * 1000) if latencies else 0.0
    agg["failures"]       = failures
    agg["n_queries"]      = len(queries)

    logger.info(f"\n📊 [{mode_name}] Final results:")
    for k, v in sorted(agg.items()):
        logger.info(f"     {k:<28}: {v:.4f}")

    return agg


# ══════════════════════════════════════════════════════════════════════════════
# Report / save helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_table(all_results: Dict[str, Dict[str, float]], k_values: List[int]) -> None:
    primary = []
    for k in k_values:
        primary += [f"hit@{k}", f"mrr@{k}", f"ndcg@{k}", f"recall@{k}"]
    primary += ["avg_latency_ms", "p95_latency_ms", "failures"]

    M = 28   # metric column width
    C = 22   # data column width
    modes = list(all_results.keys())
    sep = "=" * (M + C * len(modes))

    print(f"\n{sep}")
    print("RETRIEVAL EVALUATION — ABLATION COMPARISON TABLE")
    print(f"  Dataset: 10k_finance_news_dataset.csv")
    print(f"  Query=title  |  Relevant=content  |  K={k_values}")
    print(sep)
    header = f"{'Metric':<{M}}" + "".join(f"{m:<{C}}" for m in modes)
    print(header)
    print("-" * (M + C * len(modes)))
    for metric in primary:
        row = f"{metric:<{M}}"
        for m in modes:
            val = all_results[m].get(metric, 0.0)
            row += f"{val:<{C}.4f}"
        print(row)
    print(sep)

    # Delta vs baseline (bm25_only)
    if "bm25_only" in all_results and len(all_results) > 1:
        print("\nΔ vs bm25_only (primary metric: hit@10 & mrr@10):")
        base = all_results["bm25_only"]
        for m in modes:
            if m == "bm25_only":
                continue
            dh = all_results[m].get("hit@10", 0) - base.get("hit@10", 0)
            dm = all_results[m].get("mrr@10", 0) - base.get("mrr@10", 0)
            dn = all_results[m].get("ndcg@10", 0) - base.get("ndcg@10", 0)
            sign = lambda x: ("+" if x >= 0 else "")
            print(f"  {m:<30}: hit@10={sign(dh)}{dh:.4f}  mrr@10={sign(dm)}{dm:.4f}  ndcg@10={sign(dn)}{dn:.4f}")
    print()


def save_json(all_results: Dict, output_path: str, extra: Optional[Dict] = None) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": all_results, "meta": extra or {}}, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON → {output_path}")


def save_csv(all_results: Dict[str, Dict], output_path: str) -> None:
    rows = [{"mode": m, **v} for m, v in all_results.items()]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info(f"CSV  → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieval evaluation on 10k_finance_news_dataset.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    p.add_argument("--data", default=str(PROJECT_ROOT / "data" / "10k_finance_news_dataset.csv"),
                   help="Path to CSV dataset")
    p.add_argument("--corpus_size", type=int, default=None,
                   help="Cap corpus size (None = all 10k). Useful for quick tests.")
    p.add_argument("--n_queries", type=int, default=-1,
                   help="Number of queries (-1 = all documents in corpus)")
    p.add_argument("--seed", type=int, default=42)

    # ── Eval ─────────────────────────────────────────────────────────────────
    p.add_argument("--modes", nargs="+",
                   choices=list(EVAL_CONFIGS.keys()) + ["all"],
                   default=["all"],
                   help="Ablation modes to evaluate")
    p.add_argument("--k_values", nargs="+", type=int, default=[1, 3, 5, 10, 20],
                   help="K values for metrics")
    p.add_argument("--top_k", type=int, default=20,
                   help="Max retrieved docs per query (must be >= max(k_values))")
    p.add_argument("--no_semantic", action="store_true",
                   help="Disable semantic (dense) retrieval — BM25 only modes")

    # ── Model / GPU ──────────────────────────────────────────────────────────
    p.add_argument("--embedding_model", default="AITeamVN/Vietnamese_Embedding",
                   help="SentenceTransformer model name or local path")
    p.add_argument("--nli_model", default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                   help="NLI model name for hybrid_rrf_temporal_nli mode")
    p.add_argument("--device", default=None,
                   help="torch device: cuda / cuda:0 / cpu (auto-detected if None)")
    p.add_argument("--embed_batch_size", type=int, default=512,
                   help="Batch size for encoding documents (H200 can handle 512+)")
    p.add_argument("--rrf_k", type=int, default=60,
                   help="RRF constant k")
    p.add_argument("--alpha", type=float, default=0.7,
                   help="α weight: final = α·RRF + (1-α)·Temporal")

    # ── Index ────────────────────────────────────────────────────────────────
    p.add_argument("--index_path", default=None,
                   help="Load pre-built FAISS index pickle (skip re-embedding)")
    p.add_argument("--save_index", default=None,
                   help="Save FAISS index after building")

    # ── Output ───────────────────────────────────────────────────────────────
    p.add_argument("--output_dir",
                   default=str(PROJECT_ROOT / "tests" / "eval_results"),
                   help="Directory for result files")
    p.add_argument("--log_every", type=int, default=1000,
                   help="Log partial metrics every N queries")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Resolve modes ────────────────────────────────────────────────────────
    selected_modes = list(EVAL_CONFIGS.keys()) if "all" in args.modes else args.modes
    if args.no_semantic:
        selected_modes = [m for m in selected_modes if "semantic" not in m and m != "hybrid_rrf" and m != "hybrid_rrf_temporal"]
        logger.info("--no_semantic: dense modes excluded")
    logger.info(f"Modes to evaluate: {selected_modes}")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.data}")
    df = load_dataset(args.data, max_rows=args.corpus_size, seed=args.seed)
    documents = build_documents(df)
    logger.info(f"Corpus size: {len(documents)} documents")

    # ── Build query list ─────────────────────────────────────────────────────
    all_queries: List[Tuple[str, str]] = [(d["title"], d["id"]) for d in documents]
    if args.n_queries == -1 or args.n_queries >= len(all_queries):
        eval_queries = all_queries
        logger.info(f"Using ALL {len(eval_queries)} documents as queries")
    else:
        eval_queries = random.sample(all_queries, args.n_queries)
        logger.info(f"Sampled {len(eval_queries)} queries from {len(all_queries)}")

    # ── Device ───────────────────────────────────────────────────────────────
    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} | VRAM: {gpu_mem:.1f} GB")

    # ── Init retriever ───────────────────────────────────────────────────────
    logger.info("Initializing KnowledgeAugmentedRetriever...")
    retriever = KnowledgeAugmentedRetriever(
        embedding_model=args.embedding_model,
        alpha=args.alpha,
        rrf_k=args.rrf_k,
        use_query_expansion=True,
    )

    # Patch embedding batch size nếu encoder tồn tại
    if retriever.encoder is not None and hasattr(retriever.encoder, "_target_device"):
        logger.info(f"Embedding model loaded. Batch size for indexing: {args.embed_batch_size}")

    # ── Index / load ─────────────────────────────────────────────────────────
    if args.index_path and os.path.exists(args.index_path):
        logger.info(f"Loading pre-built index: {args.index_path}")
        retriever.load_index(args.index_path)
    else:
        logger.info(f"Building index for {len(documents)} documents...")
        t0 = time.time()
        # Patch batch size để tận dụng H200 VRAM
        _orig_encode = None
        if retriever.encoder is not None:
            def _patched_index(documents, text_field="text", id_field="id", timestamp_field="timestamp"):
                """Wrapper that uses large batch_size for encoding on target device."""
                from datetime import datetime
                logger.info(f"  Indexing {len(documents)} docs (embed_batch={args.embed_batch_size}, device={device})...")
                retriever.documents = []
                texts = []
                for doc in documents:
                    entry = {
                        "id":        doc.get(id_field, str(len(retriever.documents))),
                        "text":      doc.get(text_field, ""),
                        "timestamp": doc.get(timestamp_field, datetime.now()),
                        "metadata":  {k: v for k, v in doc.items() if k not in [text_field, id_field, timestamp_field]},
                    }
                    retriever.documents.append(entry)
                    texts.append(entry["text"])
                # BM25
                tokenized = [retriever._tokenize(t) for t in tqdm(texts, desc="BM25 tokenize", ncols=90)]
                from rank_bm25 import BM25Okapi
                retriever.bm25 = BM25Okapi(tokenized)
                # Dense embeddings — move encoder to target device, use large batch
                try:
                    retriever.encoder.to(device)
                    logger.info(f"  Encoder moved to {device}")
                except Exception as _e:
                    logger.warning(f"  Could not move encoder to {device}: {_e}")
                retriever.document_embeddings = retriever.encoder.encode(
                    texts,
                    batch_size=args.embed_batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                )
                # FAISS
                try:
                    import faiss
                    retriever.faiss_index = faiss.IndexFlatIP(retriever.embedding_dim)
                    faiss.normalize_L2(retriever.document_embeddings)
                    retriever.faiss_index.add(retriever.document_embeddings)
                    logger.info(f"  FAISS index: {retriever.faiss_index.ntotal} vectors")
                except Exception as _fe:
                    logger.warning(f"  FAISS unavailable ({_fe}), dense fallback to numpy similarity")
            retriever.index_documents = _patched_index

        retriever.index_documents(
            documents=documents,
            text_field="text",
            id_field="id",
            timestamp_field="timestamp",
        )
        logger.info(f"Indexing done in {time.time() - t0:.1f}s")

        if args.save_index:
            retriever.save_index(args.save_index)
            logger.info(f"Index saved → {args.save_index}")

    # ── Init NLI Scorer ──────────────────────────────────────────────────────
    nli_scorer = None
    if any(m == "hybrid_rrf_temporal_nli" for m in selected_modes):
        if not args.no_semantic:
            logger.info(f"Initializing NLIScorer ({args.nli_model})...")
            from src.models.nli_scorer import NLIScorer
            nli_scorer = NLIScorer(model_name=args.nli_model, device=device).load()
        else:
            logger.warning("--no_semantic disables NLI. Removing hybrid_rrf_temporal_nli mode.")
            selected_modes.remove("hybrid_rrf_temporal_nli")

    # ── Run ablation ─────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    all_results: Dict[str, Dict[str, float]] = {}
    top_k = max(args.top_k, max(args.k_values))
    t_total = time.time()

    for mode in selected_modes:
        cfg = dict(EVAL_CONFIGS[mode])
        if args.no_semantic:
            cfg["use_semantic"] = False
            cfg["use_nli"] = False

        t_mode = time.time()
        metrics = run_evaluation(
            retriever=retriever,
            queries=eval_queries,
            mode_name=mode,
            retrieve_kwargs=cfg,
            k_values=args.k_values,
            top_k=top_k,
            log_every=args.log_every,
            nli_scorer=nli_scorer,
        )
        metrics["wall_time_s"] = round(time.time() - t_mode, 2)
        all_results[mode] = metrics

    logger.info(f"\nTotal evaluation time: {time.time() - t_total:.1f}s")

    # ── Report ────────────────────────────────────────────────────────────────
    print_table(all_results, args.k_values)

    # ── Save ─────────────────────────────────────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    meta = {
        "dataset":         args.data,
        "corpus_size":     len(documents),
        "n_queries":       len(eval_queries),
        "k_values":        args.k_values,
        "top_k":           top_k,
        "modes":           selected_modes,
        "embedding_model": args.embedding_model,
        "device":          device,
        "alpha":           args.alpha,
        "rrf_k":           args.rrf_k,
        "seed":            args.seed,
        "timestamp":       ts,
    }
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)

    json_out = os.path.join(args.output_dir, f"retrieval_eval_{ts}.json")
    csv_out  = os.path.join(args.output_dir, f"retrieval_eval_{ts}.csv")
    save_json(all_results, json_out, meta)
    save_csv(all_results, csv_out)

    logger.info("✅ Evaluation complete!")
    logger.info(f"   JSON → {json_out}")
    logger.info(f"   CSV  → {csv_out}")


if __name__ == "__main__":
    main()
