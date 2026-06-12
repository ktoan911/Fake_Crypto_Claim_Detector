from __future__ import annotations

import os
import sys

# Giới hạn số thread cho BLAS/OMP TRƯỚC khi import torch/numpy/bertopic,
# nếu không các env này sẽ bị bỏ qua. Cluster yếu thì oversubscribe gây thrash.
_THREADS = os.getenv("CLUSTER_NUM_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
    os.environ.setdefault(_v, _THREADS if _v != "TOKENIZERS_PARALLELISM" else "false")

import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv

from loguru import logger
from opensearchpy.helpers import scan

from src.database.opensearch import OpenSearchKB
from src.llm_call import generate_cluster_content_with_llm

load_dotenv()

try:
    import numpy as np
    import torch
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
except ImportError as exc:
    raise ImportError(
        "sentence-transformers, bertopic, hdbscan, numpy are required. Install dependencies from requirements.txt."
    ) from exc

try:
    torch.set_num_threads(int(_THREADS))
except Exception:
    pass

_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_MODEL_LOCK = Lock()


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    """Cache SentenceTransformer để tránh tải lại model ~120MB mỗi request."""
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached is None:
            cached = SentenceTransformer(model_name)
            _MODEL_CACHE[model_name] = cached
        return cached

def cluster_claims(
    claims: List[str],
    model_name: str,
    max_k: int = 10,
    random_state: int = 42,
    llm_workers: int = 2,
) -> Dict:
    cleaned_claims = [c.strip() for c in claims if isinstance(c, str) and c.strip()]
    if not cleaned_claims:
        raise ValueError("Input claims list is empty after cleaning.")

    # e5 models cần prefix "passage: " cho clustering để activate correct embedding space
    _is_e5 = "e5" in model_name.lower()
    encode_claims = [f"passage: {c}" for c in cleaned_claims] if _is_e5 else cleaned_claims

    n_samples = len(cleaned_claims)

    if n_samples < 3:
        # Quá ít claim, gộp chung vào 1 cluster — không cần load embedding model
        rep_claim = cleaned_claims[0]
        gen_content = generate_cluster_content_with_llm(cleaned_claims, rep_claim)
        return {
            "num_input_claims": n_samples,
            "num_clusters": 1,
            "model_name": model_name,
            "clusters": [
                {
                    "cluster_id": 0,
                    "size": n_samples,
                    "representative_claim": rep_claim,
                    "cluster_content": gen_content,
                    "claims": cleaned_claims,
                }
            ],
        }

    model = _get_embedding_model(model_name)

    # Tính embedding một lần và normalize → cosine sim = dot product, dùng được
    # cho cả UMAP (metric=cosine), centroid và cohesion filter ở dưới.
    embeddings = model.encode(
        encode_claims,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # n_neighbors: giữ cấu trúc local, claim khác chủ đề ít bị kéo lại gần.
    # n_components: tăng lên 10 để giữ nhiều cấu trúc topic hơn trước khi HDBSCAN
    # chạy — 5 chiều trước đây mất signal phân biệt các chủ đề gần nhau.
    # min_topic_size adaptive theo sqrt(N).
    n_neighbors = max(5, min(15, n_samples - 1))
    n_components = max(2, min(10, n_samples // 5))
    min_topic_size = max(2, int(math.sqrt(n_samples) / 2))
    min_samples = max(2, min(min_topic_size, int(math.sqrt(n_samples) / 2)))

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )

    # cluster_selection_method="eom" cho cluster ổn định hơn "leaf" khi
    # min_samples đã đủ chặt — leaf dễ vỡ vụn thành nhiều cluster nhỏ kém chất lượng.
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )

    topic_model = BERTopic(
        embedding_model=model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=False,
    )

    # Truyền embeddings đã tính sẵn để BERTopic không encode lại — vừa nhanh hơn
    # vừa đảm bảo cluster + cohesion filter dùng cùng một không gian vector.
    topics, _ = topic_model.fit_transform(cleaned_claims, embeddings=embeddings)

    cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(topics):
        cluster_to_indices[int(label)].append(idx)

    # Hậu kiểm cohesion: với mỗi cluster, tính centroid trên embedding chuẩn hoá
    # rồi loại claim có cosine sim < ngưỡng. Đây là chốt chặn cuối.
    # Ngưỡng 0.55: hai claim cùng chủ đề tài chính tiếng Việt thường đạt >0.6,
    # khác chủ đề thường < 0.5. Dùng 0.45 trước đây quá lỏng — cluster 3 bị lẫn
    # đô thị/hạ tầng vào vì cosine sim ~0.47-0.50 vẫn pass ngưỡng cũ.
    cohesion_threshold = float(os.getenv("CLUSTER_COHESION_THRESHOLD", "0.55"))

    refined: Dict[int, Dict] = {}
    for cid, indices in cluster_to_indices.items():
        if cid == -1 or len(indices) < min_topic_size:
            continue
        cluster_emb = embeddings[indices]
        centroid = cluster_emb.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm < 1e-9:
            continue
        centroid = centroid / norm
        sims = cluster_emb @ centroid  # cosine vì đã normalize
        keep_mask = sims >= cohesion_threshold
        kept_indices = [indices[i] for i, k in enumerate(keep_mask) if k]
        if len(kept_indices) < min_topic_size:
            # Cluster còn lại quá nhỏ sau khi lọc → drop, ưu tiên chất lượng
            # hơn coverage như user yêu cầu.
            continue
        kept_emb = embeddings[kept_indices]
        kept_centroid = kept_emb.mean(axis=0)
        kept_centroid = kept_centroid / max(np.linalg.norm(kept_centroid), 1e-9)
        kept_sims = kept_emb @ kept_centroid
        # Representative = claim gần centroid nhất → ổn định và đại diện hơn
        # c-TF-IDF mặc định của BERTopic (vốn token hoá theo space, lệch với tiếng Việt).
        rep_local_idx = int(np.argmax(kept_sims))
        refined[cid] = {
            "indices": kept_indices,
            "rep_global_idx": kept_indices[rep_local_idx],
            "mean_sim": float(kept_sims.mean()),
        }

    # Sắp xếp theo size giảm dần, tie-break bằng cohesion (mean_sim) để cluster
    # chặt được ưu tiên khi cùng size. Lấy tối đa max_k cluster.
    sorted_cluster_ids = sorted(
        refined.keys(),
        key=lambda cid: (len(refined[cid]["indices"]), refined[cid]["mean_sim"]),
        reverse=True,
    )[:max_k]

    prepared = []
    for cluster_id in sorted_cluster_ids:
        info = refined[cluster_id]
        indices = info["indices"]
        cluster_claim_list = [cleaned_claims[i] for i in indices]
        representative_claim = cleaned_claims[info["rep_global_idx"]]
        prepared.append((cluster_id, indices, cluster_claim_list, representative_claim))

    # LLM call là I/O-bound (HTTP tới Together). Chạy song song để tổng latency
    # gần bằng 1 call thay vì N call tuần tự. Mặc định 2 worker — nhiều hơn dễ
    # đẩy Together vào rate-limit ngầm và làm cả batch cùng timeout.
    def _summarize(item):
        _cid, _idx, _claims, _rep = item
        try:
            return generate_cluster_content_with_llm(
                cluster_claims=_claims,
                representative_claim=_rep,
            )
        except Exception as e:
            # Một cluster fail không được phép kéo cả pipeline xuống.
            logger.error(f"cluster {_cid} summarize failed: {e}, fallback rep claim")
            return _rep

    workers = max(1, min(llm_workers, len(prepared)))
    if workers == 1 or len(prepared) <= 1:
        contents = [_summarize(p) for p in prepared]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            contents = list(pool.map(_summarize, prepared))

    clusters = []
    for (cluster_id, indices, cluster_claim_list, representative_claim), generated_content in zip(
        prepared, contents
    ):
        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": len(indices),
                "representative_claim": representative_claim,
                "cluster_content": generated_content,
                "claims": cluster_claim_list[:10],
            }
        )

    return {
        "num_input_claims": n_samples,
        "num_clusters": len(clusters),
        "model_name": model_name,
        "clusters": clusters,
    }


def load_claims_from_opensearch(timestamp_seconds: int = 86400) -> List[str]:
    kb = OpenSearchKB(
        index_name=os.getenv("OP_CLAIMS_INDEX", "claims"),
        embedding_dim=1,
    )
    client = kb.client
    index = kb.index

    now_utc = datetime.now(timezone.utc)
    cutoff_time = (now_utc - timedelta(seconds=timestamp_seconds)).isoformat()

    try:
        hits_iter = scan(
            client,
            index=index,
            query={
                "bool": {
                    "must": [{"range": {"checked_at": {"gte": cutoff_time}}}],
                    "filter": [{"terms": {"verdict.keyword": ["Sai"]}}],
                }
            },
            _source=["claim"],
            size=1000,
        )
        return [
            h["_source"]["claim"]
            for h in hits_iter
            if h.get("_source", {}).get("claim")
        ]
    except Exception as e:
        logger.error(f"Error querying OpenSearch: {e}")
        return []


def run_clustering_pipeline(
    timestamp_seconds: int = 86400,
    model_name: str = "intfloat/multilingual-e5-base",
) -> Dict:
    """
    Hàm để gọi trực tiếp từ các file khác.
    Bạn có thể import:
    from scripts.cluster_claims import run_clustering_pipeline

    result = run_clustering_pipeline(timestamp_seconds=864000)
    """
    claims = load_claims_from_opensearch(timestamp_seconds)
    if not claims:
        logger.info(f"No claims found in the last {timestamp_seconds} seconds.")
        return {"error": "No claims found", "clusters": []}

    logger.info(f"Đã lấy được {len(claims)} claim từ OpenSearch cho việc cluster claim.")

    return cluster_claims(claims=claims, model_name=model_name)
