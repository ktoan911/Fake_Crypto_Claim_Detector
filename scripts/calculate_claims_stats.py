import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cluster_claims import run_clustering_pipeline
from src.database.opensearch import OpenSearchKB


def _get_claims_kb() -> OpenSearchKB:
    """Return an OpenSearchKB pointed at the 'claims' index."""
    return OpenSearchKB(
        index_name=os.getenv("OP_CLAIMS_INDEX", "claims"),
        embedding_dim=1,
    )


def _get_stats_kb() -> OpenSearchKB:
    """Return an OpenSearchKB pointed at the 'stats' index."""
    return OpenSearchKB(
        index_name=os.getenv("OP_STATS_INDEX", "stats"),
        embedding_dim=1,
    )


def calculate_and_save_stats(target_date: datetime = None):
    try:
        if target_date is None:
            target_date = datetime.now(timezone.utc)

        kb = _get_claims_kb()
        client = kb.client
        index = kb.index

        cutoff_24h = (target_date - timedelta(hours=24)).isoformat()

        # ── 1. 10 claims gần nhất ───────────────────────────────────────────
        recent_resp = client.search(
            index=index,
            body={
                "size": 10,
                "sort": [{"checked_at": {"order": "desc"}}],
                "_source": ["claim", "verdict", "checked_at"],
            },
        )
        recent_claims = [
            h["_source"] for h in recent_resp.get("hits", {}).get("hits", [])
        ]

        # ── 2 & 3. Thống kê 24h qua — dùng terms aggregation ───────────────
        stats_resp = client.search(
            index=index,
            body={
                "size": 0,
                "query": {
                    "range": {
                        "checked_at": {
                            "gte": cutoff_24h,
                            "lte": target_date.isoformat(),
                        }
                    }
                },
                "aggs": {
                    "by_verdict": {
                        "terms": {
                            "field": "verdict.keyword",
                            "size": 10,
                        }
                    }
                },
            },
        )
        verdict_buckets = (
            stats_resp.get("aggregations", {}).get("by_verdict", {}).get("buckets", [])
        )
        counts = {b["key"]: b["doc_count"] for b in verdict_buckets}
        total_24h = sum(counts.values()) or 1  # tránh chia 0

        dung = counts.get("Đúng", 0)
        sai = counts.get("Sai", 0)
        ccc = counts.get("Chưa chắc chắn", 0)

        stats_24h = {
            "đúng": dung,
            "sai": sai,
            "chưa chắc chắn": ccc,
            "percent_đúng": round(dung / total_24h * 100, 2),
            "percent_sai": round(sai / total_24h * 100, 2),
        }

        # Lấy trực tiếp dữ liệu 6 ngày trước từ index stats thay vì query lại claims
        stats_kb = _get_stats_kb()
        stats_client = stats_kb.client
        stats_index = stats_kb.index

        target_date_str = target_date.strftime("%Y-%m-%d")
        past_dates = [
            (target_date - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(6, 0, -1)
        ]

        daily_total = {}
        daily_false = {}

        if stats_client.indices.exists(index=stats_index):
            try:
                mget_resp = stats_client.mget(
                    index=stats_index,
                    body={"ids": past_dates},
                    _source_includes=["stats_24h"],
                )
                for doc in mget_resp.get("docs", []):
                    date_key = doc["_id"]
                    if doc.get("found"):
                        past_stats = doc.get("_source", {}).get("stats_24h", {})
                        d_total = (
                            past_stats.get("đúng", 0)
                            + past_stats.get("sai", 0)
                            + past_stats.get("chưa chắc chắn", 0)
                        )
                        d_false = past_stats.get("sai", 0)
                        daily_total[date_key] = d_total
                        daily_false[date_key] = d_false
                    else:
                        daily_total[date_key] = 0
                        daily_false[date_key] = 0
            except Exception as e:
                logger.warning(f"Failed to mget past stats, using 0: {e}")
                for d in past_dates:
                    daily_total[d] = 0
                    daily_false[d] = 0
        else:
            for d in past_dates:
                daily_total[d] = 0
                daily_false[d] = 0

        # Thêm dữ liệu ngày hiện tại từ kết quả stats_24h vừa tính ở trên
        daily_total[target_date_str] = (
            stats_24h.get("đúng", 0)
            + stats_24h.get("sai", 0)
            + stats_24h.get("chưa chắc chắn", 0)
        )
        daily_false[target_date_str] = stats_24h.get("sai", 0)

        result_dict_cluster = run_clustering_pipeline(
            timestamp_seconds=864000,
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

        doc_id = target_date.strftime("%Y-%m-%d")

        result = {
            "id": doc_id,  # dùng làm _id trong OpenSearch
            "date": doc_id,
            "timestamp": target_date.isoformat(),
            "recent_claims": recent_claims,
            "stats_24h": stats_24h,
            "daily_total": daily_total,
            "daily_false": daily_false,
            "cluster": result_dict_cluster[:5],
        }

        # Lưu vào OpenSearch index stats qua insert_many() của OpenSearchKB
        res = stats_kb.insert_many([result], upsert=True)
        logger.info(
            f"Successfully saved stats for {doc_id} to index '{stats_index}': {res}"
        )

    except Exception as e:
        logger.error(f"Error calculating stats:\n{e}")
        import traceback

        logger.error(traceback.format_exc())
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate and save claims stats to OpenSearch."
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Target date in YYYY-MM-DD format (UTC). Defaults to today.",
        default=None,
    )
    args = parser.parse_args()

    if args.date:
        target = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        target = datetime.now(timezone.utc)

    calculate_and_save_stats(target)
