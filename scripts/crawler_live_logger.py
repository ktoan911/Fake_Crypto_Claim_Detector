"""
Reads from stdin, forwards to stdout, and pushes content to crawl_logs in OpenSearch realtime.
Usage: some_command 2>&1 | python3 crawler_live_logger.py [doc_id]
"""

import os
import queue
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# dotenv từ cwd (= $REPO_DIR sau khi cd)
try:
    from dotenv import load_dotenv
    for _p in (Path.cwd(), Path.cwd().parent):
        if (_p / ".env").exists():
            load_dotenv(_p / ".env")
            break
except ImportError:
    pass

_OP_HOST = os.getenv("OP_HOST", "").strip()
_OP_PORT = os.getenv("OP_PORT", "").strip()
_OP_USER = os.getenv("OP_AUTH_USERNAME", "").strip()
_OP_PASS = os.getenv("OP_AUTH_PASSWORD", "").strip()
_INDEX = "crawl_logs"
_FLUSH_EVERY = 5  # dòng/lần push

if not all([_OP_HOST, _OP_PORT, _OP_USER, _OP_PASS]):
    sys.stderr.write("[live_logger] Thiếu biến OpenSearch — chỉ passthrough stdout.\n")
    for line in sys.stdin:
        sys.stdout.write(line)
        sys.stdout.flush()
    sys.exit(0)

from opensearchpy import OpenSearch, RequestsHttpConnection  # noqa: E402

_client = OpenSearch(
    hosts=[{"host": _OP_HOST, "port": int(_OP_PORT), "scheme": "https"}],
    http_auth=(_OP_USER, _OP_PASS),
    verify_certs=True,
    http_compress=True,
    timeout=5,
    max_retries=0,
    retry_on_timeout=False,
    connection_class=RequestsHttpConnection,
)

DOC_ID = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%dT%H:%M")


def _ensure_index() -> None:
    try:
        if not _client.indices.exists(index=_INDEX):
            _client.indices.create(
                index=_INDEX,
                body={
                    "mappings": {
                        "properties": {
                            "start_ts": {"type": "date"},
                            "status":   {"type": "keyword"},
                            "content":  {"type": "text", "index": False},
                        }
                    }
                },
            )
    except Exception as e:
        sys.stderr.write(f"[live_logger] index create failed: {e}\n")


def _init_doc() -> None:
    try:
        _client.index(
            index=_INDEX,
            id=DOC_ID,
            body={
                "start_ts": datetime.now(timezone.utc).isoformat(),
                "status": "running",
                "content": "",
            },
        )
    except Exception as e:
        sys.stderr.write(f"[live_logger] init doc failed: {e}\n")


threading.Thread(target=lambda: (_ensure_index(), _init_doc()), daemon=True).start()

_lines: deque = deque(maxlen=2000)
_flush_q: queue.Queue = queue.Queue()


def _flush_worker() -> None:
    while True:
        content, status = _flush_q.get()
        try:
            _client.update(
                index=_INDEX,
                id=DOC_ID,
                body={"doc": {"content": content, "status": status}},
            )
        except Exception as e:
            sys.stderr.write(f"[live_logger] flush failed: {e}\n")
        finally:
            _flush_q.task_done()


# daemon=False để flush cuối hoàn thành trước khi process thoát
threading.Thread(target=_flush_worker, daemon=False).start()

_pending = 0
for _line in sys.stdin:
    sys.stdout.write(_line)
    sys.stdout.flush()
    stripped = _line.rstrip("\n")
    if stripped:
        _lines.append(stripped)
        _pending += 1
        if _pending >= _FLUSH_EVERY:
            _flush_q.put(("\n".join(_lines) + "\n", "running"))
            _pending = 0

# Final flush — đánh dấu done
_flush_q.put(("\n".join(_lines) + "\n", "done"))
_flush_q.join()
