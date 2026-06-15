import asyncio
import hashlib
import json
import os
import sys
import threading
import time
import traceback

# Phải set trước khi import torch để có hiệu lực.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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
_inference_executor: ThreadPoolExecutor | None = None

# ── crawl log via OpenSearch ─────────────────────────────────────────────────
_CRAWL_LOGS_INDEX = "crawl_logs"

# ── Server log (in-memory ring buffer) ───────────────────────────────────────
_SERVER_LOG_MAX = 500
_server_log_buf: deque[str] = deque(maxlen=_SERVER_LOG_MAX)

_inference_timeout_s = float(os.getenv("INFERENCE_TIMEOUT_S", "300"))
_max_claim_chars = int(os.getenv("MAX_CLAIM_CHARS", "0"))



def _server_log_sink(message) -> None:
    """Loguru sink: ghi thẳng vào ring buffer."""
    line = str(message).rstrip("\n")
    if line.strip():
        _server_log_buf.append(line)


def _setup_server_log() -> None:
    """Đăng ký loguru sink + stdlib handler để capture uvicorn logs vào ring buffer."""
    import logging as _stdlib_logging
    from datetime import datetime

    class _StdlibSink(_stdlib_logging.Handler):
        def emit(self, record: _stdlib_logging.LogRecord) -> None:
            ts = datetime.now().strftime("%H:%M:%S")
            line = f"{ts} - {record.levelname} - {record.getMessage()}"
            _server_log_buf.append(line)

    _sink = _StdlibSink()
    for name in ("uvicorn.access", "uvicorn.error", "uvicorn"):
        _stdlib_logging.getLogger(name).addHandler(_sink)

    logger.add(
        _server_log_sink,
        format="{time:HH:mm:ss} - {level} - {message}",
        level="INFO",
        enqueue=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model và khởi tạo tài nguyên dùng chung một lần khi startup."""
    global _verifier, _stats_kb, _inference_executor

    _inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
    logger.info("[startup] Inference executor (max_workers=1) ready ✓")

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
        logger.info(f"[startup] FusionClaimVerifier ready ✓ | timeout={_inference_timeout_s}s")
        logger.info("[startup] Running model warmup (torch.compile kernel compilation) …")
        _verifier.warmup()
        logger.info("[startup] Model warmup complete ✓")
    except Exception:
        import traceback
        logger.error(f"[startup] Failed to load verifier:\n{traceback.format_exc()}")

    _stats_kb = OpenSearchKB(
        index_name=os.getenv("OP_STATS_INDEX", "stats"),
        embedding_dim=1,
    )
    logger.info("[startup] Stats OpenSearchKB connection ready ✓")

    try:
        _setup_server_log()
        logger.info("[startup] Server log handler registered ✓")
    except Exception as e:
        logger.warning(f"[startup] Server log handler failed (non-fatal): {e}")

    yield

    _inference_executor.shutdown(wait=False)
    logger.info("[shutdown] API server stopping.")


app = FastAPI(title="Fake Claim Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClaimRequest(BaseModel):
    claim: str


class AdminLoginRequest(BaseModel):
    username: str
    password: str


@app.post("/admin/login")
async def admin_login(request: AdminLoginRequest):
    import bcrypt
    from opensearchpy.exceptions import ConnectionError as OSConnectionError

    try:
        client = _crawl_client()
        if not client.indices.exists(index="admin"):
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": "Hệ thống xác thực chưa được khởi tạo.",
                },
            )

        # Chỉ query theo username, verify hash trong Python để tránh timing attack
        resp = client.search(
            index="admin",
            body={"size": 1, "query": {"term": {"username": request.username}}},
        )
        hits = resp.get("hits", {}).get("hits", [])

        _INVALID = JSONResponse(
            status_code=401,
            content={
                "success": False,
                "error": "Tên đăng nhập hoặc mật khẩu không đúng.",
            },
        )

        if not hits:
            # Vẫn chạy checkpw trên dummy hash để giữ thời gian phản hồi đồng đều
            bcrypt.checkpw(b"dummy", bcrypt.hashpw(b"dummy", bcrypt.gensalt()))
            return _INVALID

        stored_hash: str = hits[0]["_source"].get("password", "")
        try:
            match = bcrypt.checkpw(
                request.password.encode("utf-8"),
                stored_hash.encode("utf-8"),
            )
        except Exception:
            return _INVALID

        if match:
            logger.info(f"[admin/login] success user={request.username!r}")
            return {"success": True}

        return _INVALID

    except OSConnectionError as e:
        logger.warning(f"[admin/login] OpenSearch unreachable: {e}")
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "Không thể kết nối đến hệ thống."},
        )
    except Exception as e:
        logger.error(f"[admin/login] error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Lỗi máy chủ nội bộ."},
        )


@app.get("/health")
def health(request: Request):
    logger.info(f"[health] domain={request.headers.get('host', 'unknown')}")
    return {
        "status": "ok",
        "model_loaded": _verifier is not None,
    }


@app.post("/verify")
async def verify_claim(request: ClaimRequest, http_request: Request):
    t_api_start = time.perf_counter()

    if _verifier is None:
        return JSONResponse(
            status_code=503,
            content={
                "verdict": "Lỗi xử lý",
                "status": "error",
                "error": "Verifier chưa được khởi tạo (xem log startup để biết lý do).",
            },
        )

    claim_text = (request.claim or "").strip()
    if not claim_text:
        return JSONResponse(
            status_code=400,
            content={"verdict": "Lỗi xử lý", "status": "error", "error": "Claim rỗng."},
        )

    if _max_claim_chars > 0 and len(claim_text) > _max_claim_chars:
        logger.warning(
            f"[verify] claim quá dài ({len(claim_text)} chars > {_max_claim_chars}), truncate"
        )
        claim_text = claim_text[:_max_claim_chars]

    domain = http_request.headers.get("host", "unknown")
    logger.info(f"[verify] domain={domain} claim={claim_text!r}")

    t_cache0 = time.perf_counter()
    cache_key = hashlib.sha1(claim_text.encode("utf-8", errors="replace")).hexdigest()
    cached = _claim_cache.get(cache_key)
    t_cache1 = time.perf_counter()
    if cached is not None:
        cache_ms = round(1000.0 * (t_cache1 - t_cache0), 1)
        api_total_ms = round(1000.0 * (time.perf_counter() - t_api_start), 1)
        logger.info(f"[verify] cache_hit key={cache_key[:8]}… cache_ms={cache_ms}")
        return {
            **cached,
            "timing_ms": {
                "cache_hit": True,
                "cache_check_ms": cache_ms,
                "api_total_ms": api_total_ms,
            },
        }

    try:
        loop = asyncio.get_running_loop()
        t_inference0 = time.perf_counter()
        prediction = await asyncio.wait_for(
            loop.run_in_executor(_inference_executor, _verifier.predict, claim_text),
            timeout=_inference_timeout_s,
        )
        t_inference1 = time.perf_counter()

        api_timing = {
            "cache_check_ms": round(1000.0 * (t_cache1 - t_cache0), 1),
            "executor_queue_ms": round(1000.0 * (t_inference0 - t_cache1), 1),
            "inference_ms": round(1000.0 * (t_inference1 - t_inference0), 1),
            "api_total_ms": round(1000.0 * (time.perf_counter() - t_api_start), 1),
        }
        logger.info(
            f"[verify] timing"
            f" | cache_ms={api_timing['cache_check_ms']}"
            f" | queue_ms={api_timing['executor_queue_ms']}"
            f" | inference_ms={api_timing['inference_ms']}"
            f" | api_total_ms={api_timing['api_total_ms']}"
        )

        inference_timing = prediction.timing_ms or {}
        result = {
            "verdict": prediction.verdict,
            "status": "success",
            "evidence": prediction.evidence,
            "source_links": prediction.source_links,
            "confidence": prediction.confidence,
            "timing_ms": {**inference_timing, **api_timing},
        }
        _claim_cache.set(cache_key, result)
        return result

    except asyncio.TimeoutError:
        logger.error(f"[verify] timeout sau {_inference_timeout_s}s key={cache_key[:8]}…")
        return JSONResponse(
            status_code=504,
            content={
                "verdict": "Lỗi xử lý",
                "status": "error",
                "error": f"Inference quá {_inference_timeout_s}s — vui lòng thử lại.",
            },
        )
    except Exception:
        logger.error(f"[verify] error: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"verdict": "Lỗi xử lý", "status": "error", "error": "Lỗi máy chủ nội bộ."},
        )


@app.get("/claims/stats")
async def claims_stats(date: str | None = None):
    """
    Đọc dữ liệu dashboard thống kê từ index 'stats'.
    - Input: `date` định dạng YYYY-MM-DD. Nếu không cung cấp, trả về bản ghi
      stats mới nhất hiện có (không mặc định "hôm nay" để tránh trả rỗng khi
      crawler/stats chưa kịp chạy xong cho ngày hiện tại).
    """
    from opensearchpy.exceptions import ConnectionError as OSConnectionError

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
        except OSConnectionError:
            raise
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

    except OSConnectionError as e:
        logger.warning(f"[claims/stats] OpenSearch unreachable: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": "OpenSearch service is unreachable. Please check the connection.",
            },
        )
    except Exception as e:
        import traceback

        logger.error(f"[claims/stats] {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)},
        )


# ── Crawler log endpoints ─────────────────────────────────────────────────────

_crawl_os_client = None


def _crawl_client():
    global _crawl_os_client
    if _crawl_os_client is None:
        from opensearchpy import OpenSearch, RequestsHttpConnection

        _crawl_os_client = OpenSearch(
            hosts=[
                {
                    "host": os.getenv("OP_HOST"),
                    "port": int(os.getenv("OP_PORT", "9200")),
                    "scheme": "https",
                }
            ],
            http_auth=(os.getenv("OP_AUTH_USERNAME"), os.getenv("OP_AUTH_PASSWORD")),
            verify_certs=True,
            http_compress=True,
            timeout=10,
            connection_class=RequestsHttpConnection,
        )
    return _crawl_os_client


def _crawl_get_doc_sync(doc_id: str | None) -> dict | None:
    client = _crawl_client()
    if not client.indices.exists(index=_CRAWL_LOGS_INDEX):
        return None
    if doc_id:
        try:
            r = client.get(index=_CRAWL_LOGS_INDEX, id=doc_id)
            return {"doc_id": r["_id"], **r["_source"]}
        except Exception:
            return None
    resp = client.search(
        index=_CRAWL_LOGS_INDEX,
        body={
            "size": 1,
            "sort": [{"start_ts": {"order": "desc"}}],
            "query": {"match_all": {}},
        },
    )
    hits = resp["hits"]["hits"]
    if not hits:
        return None
    doc = {"doc_id": hits[0]["_id"], **hits[0]["_source"]}
    if "content" in doc:
        lines = doc["content"].splitlines()
        if len(lines) > _SERVER_LOG_MAX:
            doc["content"] = "\n".join(lines[-_SERVER_LOG_MAX:]) + "\n"
    return doc


_CRAWL_INFO_INDEX = "crawl_info"


def _source_display_name(url: str) -> str:
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = parsed.netloc or parsed.path.split("/")[0]
        if host.startswith("www."):
            host = host[4:]
        return host or url
    except Exception:
        return url


def _crawl_info_sync(days: int) -> dict:
    client = _crawl_client()
    if not client.indices.exists(index=_CRAWL_INFO_INDEX):
        return {"crawl_by_day": [], "per_source": []}

    resp = client.search(
        index=_CRAWL_INFO_INDEX,
        body={
            "size": days,
            "sort": [{"crawled_at": {"order": "desc"}}],
            "_source": ["crawled_at", "total_articles"],
        },
    )
    crawl_by_day = [
        {
            "day": hit["_source"].get("crawled_at", "")[:10],
            "total_crawl": hit["_source"].get("total_articles", 0),
        }
        for hit in resp["hits"]["hits"]
    ]

    agg_resp = client.search(
        index=_CRAWL_INFO_INDEX,
        body={
            "size": 0,
            "aggs": {
                "sources": {
                    "nested": {"path": "per_source"},
                    "aggs": {
                        "by_url": {
                            "terms": {"field": "per_source.source_url", "size": 30},
                            "aggs": {"total": {"sum": {"field": "per_source.count"}}},
                        }
                    },
                }
            },
        },
    )
    buckets = (
        agg_resp.get("aggregations", {})
        .get("sources", {})
        .get("by_url", {})
        .get("buckets", [])
    )
    merged: dict[str, int] = {}
    for b in buckets:
        name = _source_display_name(b["key"])
        merged[name] = merged.get(name, 0) + int(b["total"]["value"])
    per_source = [
        {"name": name, "value": total}
        for name, total in sorted(merged.items(), key=lambda x: -x[1])
    ]

    return {"crawl_by_day": crawl_by_day, "per_source": per_source}


@app.get("/crawler/info")
async def get_crawler_info(days: int = 7):
    """Thống kê từ index 'crawl_info': số bài theo ngày và phân bố nguồn crawl."""
    from opensearchpy.exceptions import ConnectionError as OSConnectionError

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _crawl_info_sync, days)
        return result
    except OSConnectionError as e:
        logger.warning(f"[crawler/info] OpenSearch unreachable: {e}")
        return JSONResponse(
            status_code=503,
            content={"crawl_by_day": [], "per_source": [], "error": "OpenSearch unreachable"},
        )
    except Exception:
        logger.error(f"[crawler/info] {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"crawl_by_day": [], "per_source": []},
        )


@app.get("/crawler/logs")
async def get_crawler_logs(doc_id: str | None = None):
    """Trả log run mới nhất (hoặc theo doc_id). Field `running` = status=="running"."""
    loop = asyncio.get_running_loop()
    doc = await loop.run_in_executor(None, _crawl_get_doc_sync, doc_id)
    if doc is None:
        return JSONResponse(status_code=404, content={"error": "Chưa có log nào."})
    return {**doc, "running": doc.get("status") == "running"}


@app.get("/crawler/logs/stream")
async def stream_crawler_logs(request: Request, doc_id: str | None = None):
    """SSE — stream log realtime. Poll mỗi 5s; đóng khi status=="done"."""
    loop = asyncio.get_running_loop()

    async def event_generator():
        sent_len = 0
        while not await request.is_disconnected():
            try:
                doc = await loop.run_in_executor(None, _crawl_get_doc_sync, doc_id)
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                await asyncio.sleep(5)
                continue

            if doc is None:
                yield ": waiting\n\n"
                await asyncio.sleep(5)
                continue

            running = doc.get("status") == "running"
            content: str = doc.get("content", "")

            if len(content) > sent_len:
                for line in content[sent_len:].split("\n"):
                    if line.strip():
                        yield f"data: {json.dumps({'line': line, 'running': running}, ensure_ascii=False)}\n\n"
                sent_len = len(content)

            if not running:
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            yield ": heartbeat\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Server logs endpoints ─────────────────────────────────────────────────────

@app.get("/server/logs")
async def get_server_logs():
    """Trả toàn bộ ring buffer server log dưới dạng content string."""
    content = "\n".join(_server_log_buf)
    return {"content": content, "running": True}
