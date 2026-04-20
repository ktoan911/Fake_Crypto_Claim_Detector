import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone

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


def _get_stats_kb() -> OpenSearchKB:
    """Return an OpenSearchKB pointed at the 'stats' index."""
    return OpenSearchKB(
        index_name=os.getenv("OP_STATS_INDEX", "stats"),
        embedding_dim=1,  # không dùng vector search ở đây
    )


@app.get("/claims/stats")
def claims_stats(date: str = None):
    """
    Đọc dữ liệu dashboard thống kê từ index 'stats'.
    - Input: `date` định dạng YYYY-MM-DD. Nếu không cung cấp, mặc định là ngày hôm nay theo giờ UTC.
    """
    try:
        stats_kb = _get_stats_kb()
        client = stats_kb.client
        index = stats_kb.index

        if date is None:
            # mặc định lấy theo ngày hiện tại UTC
            target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        else:
            target_date = date

        if not client.indices.exists(index=index):
            return {
                "status": "error",
                "error": f"Index '{index}' not found. Please run calculate_claims_stats.py first.",
            }

        try:
            resp = client.get(index=index, id=target_date)
            return resp.get("_source", {})
        except Exception:
            # Nếu không tìm thấy bằng id (404), có thể do ngày chưa được tính thì ưu tiên lấy cái lớn nhất hiện có
            logger.warning(
                f"Stats for date {target_date} not found. Looking for the most recent stats..."
            )
            search_resp = client.search(
                index=index,
                body={
                    "size": 1,
                    "sort": [{"date": {"order": "desc"}}],
                },
            )
            hits = search_resp.get("hits", {}).get("hits", [])
            if hits:
                return hits[0]["_source"]
            else:
                return {
                    "status": "error",
                    "error": f"No stats data available in index '{index}'.",
                }

    except Exception as e:
        import traceback

        logger.error(f"[claims/stats] {traceback.format_exc()}")
        return {"status": "error", "error": str(e)}
