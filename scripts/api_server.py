import asyncio
import hashlib
import os
import sys
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
_claim_cache: _TTLCache = _TTLCache(maxsize=500, ttl=3600.0)
_stats_kb: OpenSearchKB | None = None
_batch_scheduler: "_BatchScheduler | None" = None

# Counter các request đang in-flight (đang chờ batch + đang inference). Dùng để
# từ chối sớm khi queue quá dài, tránh tích lũy RAM khi burst.
_pending_lock = threading.Lock()
_pending_count = 0
_max_pending = int(os.getenv("MAX_PENDING_REQUESTS", "64"))
_inference_timeout_s = float(os.getenv("INFERENCE_TIMEOUT_S", "120"))
_batch_max_size = int(os.getenv("BATCH_MAX_SIZE", "8"))
_batch_max_wait_ms = float(os.getenv("BATCH_MAX_WAIT_MS", "50"))


class _BatchScheduler:
    """Dynamic micro-batching cho FusionClaimVerifier.

    Gom các request đến trong cửa sổ ngắn (`max_wait_ms`) thành 1 batch
    tối đa `max_batch` claims rồi gọi `verifier.predict_batch` một lần.
    Tận dụng retrieval/LLM/fusion đã được batch sẵn → tăng throughput
    đáng kể khi nhiều request đến cùng lúc, đổi lại một chút latency
    cho request lẻ.
    """

    def __init__(self, verifier: FusionClaimVerifier, max_batch: int, max_wait_ms: float):
        self.verifier = verifier
        self.max_batch = max(1, max_batch)
        self.max_wait_s = max(0.0, max_wait_ms / 1000.0)
        self.queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="batch-scheduler")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def submit(self, claim: str):
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self.queue.put((claim, fut))
        return await fut

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                first = await self.queue.get()
            except asyncio.CancelledError:
                return
            batch: list[tuple[str, asyncio.Future]] = [first]

            # Gom thêm các request đến trong cửa sổ chờ.
            deadline = loop.time() + self.max_wait_s
            while len(batch) < self.max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    for _, fut in batch:
                        if not fut.done():
                            fut.cancel()
                    return

            claims = [c for c, _ in batch]
            logger.info(f"[batch] running size={len(claims)}")
            try:
                predictions = await loop.run_in_executor(
                    None, self.verifier.predict_batch, claims
                )
            except Exception as exc:
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue

            if len(predictions) != len(batch):
                # predict_batch có thể bỏ qua claim rỗng → đã validate ở /verify,
                # nhưng vẫn fail-safe.
                err = RuntimeError(
                    f"predict_batch trả {len(predictions)} kết quả cho {len(batch)} claims"
                )
                logger.error(f"[batch] mismatch: {err}")
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(err)
                continue

            for (_, fut), pred in zip(batch, predictions):
                if not fut.done():
                    fut.set_result(pred)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model và khởi tạo tài nguyên dùng chung một lần khi startup."""
    global _verifier, _stats_kb, _batch_scheduler

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

    if _verifier is not None:
        _batch_scheduler = _BatchScheduler(
            verifier=_verifier,
            max_batch=_batch_max_size,
            max_wait_ms=_batch_max_wait_ms,
        )
        _batch_scheduler.start()
        logger.info(
            f"[startup] BatchScheduler started | max_batch={_batch_max_size} "
            f"max_wait_ms={_batch_max_wait_ms} max_pending={_max_pending} "
            f"timeout={_inference_timeout_s}s"
        )

    _stats_kb = OpenSearchKB(
        index_name=os.getenv("OP_STATS_INDEX", "stats"),
        embedding_dim=1,
    )
    logger.info("[startup] Stats OpenSearchKB connection ready ✓")

    yield

    if _batch_scheduler is not None:
        await _batch_scheduler.stop()
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
    with _pending_lock:
        pending = _pending_count
    queued = _batch_scheduler.queue.qsize() if _batch_scheduler is not None else 0
    return {
        "status": "ok",
        "model_loaded": _verifier is not None,
        "pending": pending,
        "max_pending": _max_pending,
        "queued": queued,
        "batch_max_size": _batch_max_size,
        "batch_max_wait_ms": _batch_max_wait_ms,
    }


@app.post("/verify")
async def verify_claim(request: ClaimRequest, http_request: Request):
    global _pending_count

    if _verifier is None or _batch_scheduler is None:
        return {
            "verdict": "Lỗi xử lý",
            "status": "error",
            "error": "Verifier chưa được khởi tạo (xem log startup để biết lý do).",
        }

    claim_text = (request.claim or "").strip()
    if not claim_text:
        return JSONResponse(
            status_code=400,
            content={
                "verdict": "Lỗi xử lý",
                "status": "error",
                "error": "Claim rỗng.",
            },
        )

    domain = http_request.headers.get("host", "unknown")
    logger.info(f"[verify] domain={domain} claim={claim_text!r}")

    cache_key = hashlib.sha1(claim_text.encode("utf-8", errors="replace")).hexdigest()

    # Cache hit không tốn inference → trả ngay, không tính vào pending.
    cached = _claim_cache.get(cache_key)
    if cached is not None:
        logger.info(f"[verify] cache_hit key={cache_key[:8]}…")
        return cached

    # Reject sớm khi queue quá dài để tránh tích luỹ RAM và client-timeout hàng loạt.
    with _pending_lock:
        if _pending_count >= _max_pending:
            current = _pending_count
            logger.warning(
                f"[verify] reject pending={current}/{_max_pending} key={cache_key[:8]}…"
            )
            return JSONResponse(
                status_code=503,
                content={
                    "verdict": "Hệ thống đang quá tải",
                    "status": "error",
                    "error": f"Quá nhiều request đang chờ ({current}/{_max_pending}). Vui lòng thử lại sau.",
                },
            )
        _pending_count += 1

    try:
        # Double-check cache trước khi submit (request trùng có thể vừa cache xong).
        cached = _claim_cache.get(cache_key)
        if cached is not None:
            logger.info(f"[verify] cache_hit (post-pending) key={cache_key[:8]}…")
            return cached

        prediction = await asyncio.wait_for(
            _batch_scheduler.submit(claim_text),
            timeout=_inference_timeout_s,
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

    except asyncio.TimeoutError:
        logger.error(
            f"[verify] timeout sau {_inference_timeout_s}s key={cache_key[:8]}…"
        )
        return JSONResponse(
            status_code=504,
            content={
                "verdict": "Lỗi xử lý",
                "status": "error",
                "error": f"Inference quá {_inference_timeout_s}s — vui lòng thử lại.",
            },
        )
    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        logger.error(f"[verify] error: {error_traceback}")
        return {
            "verdict": "Lỗi xử lý",
            "status": "error",
            "error": str(e),
            "traceback": error_traceback,
        }
    finally:
        with _pending_lock:
            _pending_count -= 1


@app.get("/claims/stats")
async def claims_stats(date: str = None):
    """
    Đọc dữ liệu dashboard thống kê từ index 'stats'.
    - Input: `date` định dạng YYYY-MM-DD. Nếu không cung cấp, trả về bản ghi
      stats mới nhất hiện có (không mặc định "hôm nay" để tránh trả rỗng khi
      crawler/stats chưa kịp chạy xong cho ngày hiện tại).
    """
    try:
        client = _stats_kb.client
        index = _stats_kb.index

        if not client.indices.exists(index=index):
            return {
                "status": "error",
                "error": f"Index '{index}' not found. Please run calculate_claims_stats.py first.",
            }

        def _latest_source():
            search_resp = client.search(
                index=index,
                body={"size": 1, "sort": [{"date": {"order": "desc"}}]},
            )
            hits = search_resp.get("hits", {}).get("hits", [])
            return hits[0]["_source"] if hits else None

        if date is None:
            latest = _latest_source()
            if latest is not None:
                return latest
            return {
                "status": "error",
                "error": f"No stats data available in index '{index}'.",
            }

        try:
            resp = client.get(index=index, id=date)
            return resp.get("_source", {})
        except Exception:
            logger.warning(
                f"Stats for date {date} not found. Looking for the most recent stats..."
            )
            latest = _latest_source()
            if latest is not None:
                return latest
            return {
                "status": "error",
                "error": f"No stats data available in index '{index}'.",
            }

    except Exception as e:
        import traceback

        logger.error(f"[claims/stats] {traceback.format_exc()}")
        return {"status": "error", "error": str(e)}