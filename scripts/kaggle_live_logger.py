"""
Mỗi lần chạy tạo 1 document duy nhất, id = ngày giờ bắt đầu (VD: 2024-01-15T08:30).
Mỗi print() append thêm dòng vào field `content`, flush lên OpenSearch mỗi 3 dòng.
Không dùng Painless script — chỉ dùng doc update với full content.
"""

import os
import queue
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# Khi chạy trực tiếp: __file__ = .../scripts/kaggle_live_logger.py → parents[1] = project root
# Khi exec() từ notebook cell: __file__ không tồn tại → dùng cwd (notebook phải ở project root)
try:
    _root = Path(__file__).resolve().parents[1]
except NameError:
    _root = Path.cwd()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_OP_HOST = os.getenv("OP_HOST")
_OP_PORT = os.getenv("OP_PORT")
_OP_USER = os.getenv("OP_AUTH_USERNAME")
_OP_PASS = os.getenv("OP_AUTH_PASSWORD")

if not all([_OP_HOST, _OP_PORT, _OP_USER, _OP_PASS]):
    sys.__stdout__.write("[live_logger] Thiếu biến môi trường OpenSearch — live logger bị tắt.\n")
    sys.__stdout__.flush()
    # Không thay thế sys.stdout, không làm gì thêm
else:
    from opensearchpy import OpenSearch, RequestsHttpConnection

    # Client riêng cho logger: timeout ngắn, không retry, không block lâu
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

    _INDEX = "crawl_logs"

    def _setup_index():
        try:
            if not _client.indices.exists(index=_INDEX):
                _client.indices.create(
                    index=_INDEX,
                    body={
                        "mappings": {
                            "properties": {
                                "start_ts": {"type": "date"},
                                "content":  {"type": "text", "index": False},
                            }
                        }
                    },
                )
        except Exception as e:
            sys.__stdout__.write(f"[live_logger] setup index failed: {e}\n")

    # ID = ngày giờ bắt đầu run (giờ local) — dùng làm document ID
    DOC_ID = datetime.now().strftime("%Y-%m-%dT%H:%M")

    def _init_doc():
        try:
            _client.index(
                index=_INDEX,
                id=DOC_ID,
                body={"start_ts": datetime.now(timezone.utc).isoformat(), "content": ""},
            )
        except Exception as e:
            sys.__stdout__.write(f"[live_logger] init doc failed: {e}\n")

    # Chạy setup và init doc trong background thread để không block exec()
    _setup_thread = threading.Thread(target=lambda: (_setup_index(), _init_doc()), daemon=True)
    _setup_thread.start()

    _FLUSH_EVERY = 3  # flush lên OpenSearch mỗi N dòng


    class _LiveLogger:
        def __init__(self):
            self._line_buf = ""                           # buffer dòng chưa kết thúc
            self._lines: deque[str] = deque(maxlen=2000)  # ring buffer
            self._pending = 0                             # số dòng chưa flush
            self._lock = threading.Lock()
            self._flush_queue: queue.Queue[str] = queue.Queue()
            threading.Thread(target=self._flush_worker, daemon=True).start()

        def _flush_worker(self) -> None:
            """Single background thread — consumes flush snapshots one at a time."""
            while True:
                snap = self._flush_queue.get()
                try:
                    _client.update(
                        index=_INDEX,
                        id=DOC_ID,
                        body={"doc": {"content": snap}},
                    )
                except Exception as e:
                    sys.__stdout__.write(f"[live_logger] flush failed: {e}\n")
                finally:
                    self._flush_queue.task_done()

        def _fire_flush(self, content_snapshot: str) -> None:
            self._flush_queue.put(content_snapshot)

        def write(self, msg: str):
            sys.__stdout__.write(msg)
            with self._lock:
                self._line_buf += msg
                while "\n" in self._line_buf:
                    line, self._line_buf = self._line_buf.split("\n", 1)
                    if line.strip():
                        self._lines.append(line)
                        self._pending += 1
                        if self._pending >= _FLUSH_EVERY:
                            self._fire_flush("\n".join(self._lines) + "\n")
                            self._pending = 0

        def flush(self):
            sys.__stdout__.flush()
            with self._lock:
                if self._pending > 0:
                    self._fire_flush("\n".join(self._lines) + "\n")
                    self._pending = 0

        def isatty(self):
            return False


    sys.stdout = _LiveLogger()
    sys.__stdout__.write(f"[live_logger] ON — doc_id={DOC_ID}\n")
    sys.__stdout__.flush()
