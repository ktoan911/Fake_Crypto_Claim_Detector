from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv

from src.database.opensearch import OpenSearchKB
from src.llm_call import generate_cluster_content_with_llm

load_dotenv()

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
except ImportError as exc:
    raise ImportError(
        "sentence-transformers and bertopic are required. Install dependencies from requirements.txt."
    ) from exc

def cluster_claims(
    claims: List[str],
    model_name: str,
    num_clusters: Optional[int] = None,
    min_k: int = 2,
    max_k: int = 10,
    random_state: int = 42,
) -> Dict:
    cleaned_claims = [c.strip() for c in claims if isinstance(c, str) and c.strip()]
    if not cleaned_claims:
        raise ValueError("Input claims list is empty after cleaning.")

    n_samples = len(cleaned_claims)

    model = SentenceTransformer(model_name)

    if n_samples < 3:
        # Quá ít claim, gộp chung vào 1 cluster
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

    n_neighbors = min(15, n_samples - 1)
    n_components = min(5, max(1, n_samples - 2))
    min_topic_size = min(10, max(2, n_samples // 3))

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=42,
    )

    topic_model = BERTopic(
        embedding_model=model,
        umap_model=umap_model,
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

    clusters = []
    for cluster_id in sorted_cluster_ids:
        indices = cluster_to_indices[cluster_id]
        cluster_claim_list = [cleaned_claims[i] for i in indices]

        rep_docs = topic_model.get_representative_docs(cluster_id)
        if rep_docs and len(rep_docs) > 0:
            representative_claim = rep_docs[0]
        else:
            representative_claim = cluster_claim_list[0]

        generated_content = generate_cluster_content_with_llm(
            cluster_claims=cluster_claim_list,
            representative_claim=representative_claim,
        )

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
                            {"terms": {"verdict.keyword": ["Sai", "Chưa chắc chắn"]}}
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
