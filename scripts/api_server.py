import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.opensearch import OpenSearchKB
from src.models.fusion_inference import FusionClaimVerifier, _resolve_fusion_model_path

# ── Global verifier (pre-warmed at startup) ─────────────────────────────────
_verifier: FusionClaimVerifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at container startup so requests never cold-start."""
    global _verifier
    logger.info("[startup] Pre-warming FusionClaimVerifier …")
    try:
        fusion_path = _resolve_fusion_model_path(
            os.getenv("FUSION_MODEL", "models/fusion_model.pt")
        )
        _verifier = FusionClaimVerifier(
            fusion_model_path=fusion_path,
            opensearch_index=os.getenv("OPENSEARCH_INDEX_NAME")
            or os.getenv("OP_KB_NAME", "news_kb"),
            llm_model_path=os.getenv("LLM_FINETUNE"),
            retriever_model_path=os.getenv(
                "RETRIEVER_MODEL", "AITeamVN/Vietnamese_Embedding"
            ),
            device=os.getenv("DEVICE", "cpu"),
            llm_evidence_top_k=int(os.getenv("FUSION_LLM_EVIDENCE_TOP_K", "3")),
            debug=True,
        )
        logger.info("[startup] FusionClaimVerifier ready ✓")
    except Exception:
        import traceback

        logger.error(f"[startup] Failed to load verifier:\n{traceback.format_exc()}")
        # Keep _verifier = None; requests will return a clear error instead of hanging.
    yield
    # shutdown: nothing to clean up
    logger.info("[shutdown] API server stopping.")


app = FastAPI(title="Fake Crypto Claim Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClaimRequest(BaseModel):
    claim: str


@app.get("/health")
def health(request: Request):
    logger.info(f"[health] domain={request.headers.get('host', 'unknown')}")
    return {"status": "ok", "model_loaded": _verifier is not None}


@app.post("/verify")
def verify_claim(request: ClaimRequest, http_request: Request):
    if _verifier is None:
        import traceback

        return {
            "verdict": "Lỗi xử lý",
            "status": "error",
            "error": "Verifier chưa được khởi tạo (xem log startup để biết lý do).",
        }
    try:
        domain = http_request.headers.get("host", "unknown")
        logger.info(f"[verify] domain={domain} claim={request.claim!r}")
        prediction = _verifier.predict(request.claim)
        return {
            "verdict": prediction.verdict,
            "status": "success",
            "evidence": prediction.evidence,
            "source_links": prediction.source_links,
            "confidence": prediction.confidence,
        }
    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        print(f"API Error: {error_traceback}", flush=True)
        return {
            "verdict": "Lỗi xử lý",
            "status": "error",
            "error": str(e),
            "traceback": error_traceback,
        }


# ── Claims dashboard ────────────────────────────────────────────────────────


def _get_claims_kb() -> OpenSearchKB:
    """Return an OpenSearchKB pointed at the 'claims' index."""
    return OpenSearchKB(
        index_name=os.getenv("OPENSEARCH_CLAIMS_INDEX", "claims"),
        embedding_dim=1,  # không dùng vector search ở đây
    )


@app.get("/claims/stats")
def claims_stats():
    """
    Trả về dashboard thống kê cho index 'claims':
      - recent_claims   : 10 claims gần nhất (sort by checked_at desc)
      - stats_24h       : đếm + % theo verdict trong 24h qua
      - daily_total     : tổng claims mỗi ngày trong 7 ngày qua
      - daily_false     : claims 'Sai' mỗi ngày trong 7 ngày qua

    Giả định: checked_at lưu dưới dạng ISO-8601 UTC string.
    """
    try:
        kb = _get_claims_kb()
        client = kb.client
        index = kb.index

        now_utc = datetime.now(timezone.utc)
        cutoff_24h = (now_utc - timedelta(hours=24)).isoformat()
        # 7 ngày bao gồm hôm nay và 6 ngày trước
        cutoff_7d = (now_utc - timedelta(days=6)).isoformat()

        # epoch ms — format-agnostic, safe for all OpenSearch date params
        def _ms(dt: datetime) -> int:
            return int(dt.timestamp() * 1000)

        now_ms = _ms(now_utc)
        cutoff_7d_ms = _ms(now_utc - timedelta(days=6))

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
                "query": {"range": {"checked_at": {"gte": cutoff_24h}}},
                "aggs": {
                    # đếm theo verdict
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

        daily_histogram = {
            "field": "checked_at",
            "calendar_interval": "day",
            "format": "yyyy-MM-dd",
            "time_zone": "UTC",
            "min_doc_count": 0,
            "extended_bounds": {"min": cutoff_7d_ms, "max": now_ms},
        }

        daily_resp = client.search(
            index=index,
            body={
                "size": 0,
                "query": {"range": {"checked_at": {"gte": cutoff_7d}}},
                "aggs": {
                    "daily_total": {"date_histogram": daily_histogram},
                    "daily_false": {
                        "filter": {"term": {"verdict.keyword": "Sai"}},
                        "aggs": {"by_day": {"date_histogram": daily_histogram}},
                    },
                },
            },
        )

        aggs = daily_resp.get("aggregations", {})

        daily_total: dict = {
            b["key_as_string"]: b["doc_count"]
            for b in aggs.get("daily_total", {}).get("buckets", [])
        }

        daily_false: dict = {
            b["key_as_string"]: b["doc_count"]
            for b in aggs.get("daily_false", {}).get("by_day", {}).get("buckets", [])
        }

        return {
            "recent_claims": recent_claims,
            "stats_24h": stats_24h,
            "daily_total": daily_total,
            "daily_false": daily_false,
        }

    except Exception as e:
        import traceback

        logger.error(f"[claims/stats] {traceback.format_exc()}")
        return {"status": "error", "error": str(e)}
