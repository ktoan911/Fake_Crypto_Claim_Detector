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

from src.database.opensearch import OpenSearchKB
from src.llm_call import generate_cluster_content_with_llm

load_dotenv()

try:
    import torch
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
except ImportError as exc:
    raise ImportError(
        "sentence-transformers, bertopic, hdbscan are required. Install dependencies from requirements.txt."
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
    llm_workers: int = 4,
) -> Dict:
    cleaned_claims = [c.strip() for c in claims if isinstance(c, str) and c.strip()]
    if not cleaned_claims:
        raise ValueError("Input claims list is empty after cleaning.")

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

    # n_neighbors thấp → UMAP giữ cấu trúc local, claim khác chủ đề ít bị kéo lại gần.
    # n_components cao hơn (5 → 10-15) để không collapse phân biệt semantic khi giảm chiều.
    # min_topic_size adaptive theo sqrt(N): vừa đủ để có cluster, không quá lớn ép gom nhầm.
    n_neighbors = max(3, min(10, n_samples - 1))
    n_components = max(2, min(15, n_samples - 2))
    min_topic_size = max(2, int(math.sqrt(n_samples) / 2))

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )

    # Tách HDBSCAN ra để chỉnh cluster_selection_method="leaf" (cluster mịn hơn
    # so với "eom" mặc định — tránh over-merge các topic gần nhau).
    # min_samples=1 giảm strictness về density, ít noise hơn.
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="leaf",
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

    topics, _ = topic_model.fit_transform(cleaned_claims)

    cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(topics):
        cluster_to_indices[int(label)].append(idx)

    # Loại bỏ cluster nhiễu (id=-1 của BERTopic), sắp xếp theo size giảm dần
    # rồi chỉ lấy tối đa max_k cluster lớn nhất trước khi gọi LLM
    sorted_cluster_ids = sorted(
        (cid for cid in cluster_to_indices if cid != -1),
        key=lambda cid: len(cluster_to_indices[cid]),
        reverse=True,
    )[:max_k]

    prepared = []
    for cluster_id in sorted_cluster_ids:
        indices = cluster_to_indices[cluster_id]
        cluster_claim_list = [cleaned_claims[i] for i in indices]

        rep_docs = topic_model.get_representative_docs(cluster_id)
        if rep_docs and len(rep_docs) > 0:
            representative_claim = rep_docs[0]
        else:
            representative_claim = cluster_claim_list[0]

        prepared.append((cluster_id, indices, cluster_claim_list, representative_claim))

    # LLM call là I/O-bound (HTTP tới Together). Chạy song song để tổng latency
    # gần bằng 1 call thay vì N call tuần tự.
    def _summarize(item):
        _cid, _idx, _claims, _rep = item
        return generate_cluster_content_with_llm(
            cluster_claims=_claims,
            representative_claim=_rep,
        )

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
                "claims": cluster_claim_list[:5],
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
        resp = client.search(
            index=index,
            body={
                "size": 10000,
                "query": {
                    "bool": {
                        "must": [{"range": {"checked_at": {"gte": cutoff_time}}}],
                        "filter": [
                            {"terms": {"verdict.keyword": ["Sai"]}}
                        ],
                    }
                },
                "_source": ["claim"],
            },
        )
        hits = resp.get("hits", {}).get("hits", [])
        return [
            h["_source"].get("claim")
            for h in hits
            if h.get("_source") and h["_source"].get("claim")
        ]
    except Exception as e:
        print(f"Error querying OpenSearch: {e}")
        return []


def run_clustering_pipeline(
    timestamp_seconds: int = 86400,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> Dict:
    """
    Hàm để gọi trực tiếp từ các file khác.
    Bạn có thể import:
    from scripts.cluster_claims import run_clustering_pipeline

    result = run_clustering_pipeline(timestamp_seconds=864000)
    """
    claims = load_claims_from_opensearch(timestamp_seconds)
    if not claims:
        print(f"No claims found in the last {timestamp_seconds} seconds.")
        return {"error": "No claims found", "clusters": []}

    print(f"Đã lấy được {len(claims)} claim từ OpenSearch.")

    return cluster_claims(claims=claims, model_name=model_name)
