import asyncio
import hashlib
import os
import sys
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.opensearch import OpenSearchKB
from src.models.fusion_inference import FusionClaimVerifier, _resolve_fusion_model_path


class _TTLCache:
    """Thread-safe LRU cache với TTL, không cần thư viện ngoài."""

    def __init__(self, maxsize: int = 500, ttl: float = 3600.0):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            if key not in self._cache:
                return None
            value, ts = self._cache[key]
            if time.monotonic() - ts > self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = (value, time.monotonic())


# ── Global state (khởi tạo trong lifespan) ──────────────────────────────────
_verifier: FusionClaimVerifier | None = None
_inference_sem: asyncio.Semaphore | None = None
_claim_cache: _TTLCache = _TTLCache(maxsize=500, ttl=3600.0)
_stats_kb: OpenSearchKB | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model và khởi tạo tài nguyên dùng chung một lần khi startup."""
    global _verifier, _inference_sem, _stats_kb

    logger.info("[startup] Pre-warming FusionClaimVerifier …")
    try:
        fusion_path = _resolve_fusion_model_path(os.getenv("FUSION_MODEL"))
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
        # _verifier = None; requests sẽ trả lỗi rõ ràng thay vì treo.

    max_concurrent = int(os.getenv("MAX_CONCURRENT_INFERENCES", "1"))
    _inference_sem = asyncio.Semaphore(max_concurrent)
    logger.info(f"[startup] Inference semaphore: max_concurrent={max_concurrent}")

    _stats_kb = OpenSearchKB(
        index_name=os.getenv("OP_STATS_INDEX", "stats"),
        embedding_dim=1,
    )
    logger.info("[startup] Stats OpenSearchKB connection ready ✓")

    yield

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
async def verify_claim(request: ClaimRequest, http_request: Request):
    if _verifier is None:
        return {
            "verdict": "Lỗi xử lý",
            "status": "error",
            "error": "Verifier chưa được khởi tạo (xem log startup để biết lý do).",
        }

    domain = http_request.headers.get("host", "unknown")
    logger.info(f"[verify] domain={domain} claim={request.claim!r}")

    cache_key = hashlib.sha1(
        request.claim.encode("utf-8", errors="replace")
    ).hexdigest()

    cached = _claim_cache.get(cache_key)
    if cached is not None:
        logger.info(f"[verify] cache_hit key={cache_key[:8]}…")
        return cached

    try:
        async with _inference_sem:
            # Double-check sau khi lấy semaphore: request trước có thể đã cache rồi.
            cached = _claim_cache.get(cache_key)
            if cached is not None:
                logger.info(f"[verify] cache_hit (post-sem) key={cache_key[:8]}…")
                return cached

            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(
                None, _verifier.predict, request.claim
            )

        result = {
            "verdict": prediction.verdict,
            "status": "success",
            "evidence": prediction.evidence,
            "source_links": prediction.source_links,
            "confidence": prediction.confidence,
        }
        _claim_cache.set(cache_key, result)
        return result

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


@app.get("/claims/stats")
async def claims_stats(date: str = None):
    """
    Đọc dữ liệu dashboard thống kê từ index 'stats'.
    - Input: `date` định dạng YYYY-MM-DD. Nếu không cung cấp, mặc định là ngày hôm nay theo giờ UTC.
    """
    try:
        client = _stats_kb.client
        index = _stats_kb.index

        target_date = (
            date if date is not None
            else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )

        if not client.indices.exists(index=index):
            return {
                "status": "error",
                "error": f"Index '{index}' not found. Please run calculate_claims_stats.py first.",
            }

        try:
            resp = client.get(index=index, id=target_date)
            return resp.get("_source", {})
        except Exception:
            logger.warning(
                f"Stats for date {target_date} not found. Looking for the most recent stats..."
            )
            search_resp = client.search(
                index=index,
                body={"size": 1, "sort": [{"date": {"order": "desc"}}]},
            )
            hits = search_resp.get("hits", {}).get("hits", [])
            if hits:
                return hits[0]["_source"]
            return {
                "status": "error",
                "error": f"No stats data available in index '{index}'.",
            }

    except Exception as e:
        import traceback

        logger.error(f"[claims/stats] {traceback.format_exc()}")
        return {"status": "error", "error": str(e)}