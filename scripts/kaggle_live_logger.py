"""
Mỗi lần chạy tạo 1 document duy nhất, id = ngày giờ bắt đầu (VD: 2024-01-15T08:30).
Mỗi print() append thêm dòng vào field `content`, flush lên OpenSearch mỗi 3 dòng.
Không dùng Painless script — chỉ dùng doc update với full content.
"""

import sys
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

from src.database.opensearch import OpenSearchKB  # noqa: E402

_kb = OpenSearchKB(index_name="crawl_logs", embedding_dim=1)

if not _kb.client.indices.exists(index="crawl_logs"):
    _kb.client.indices.create(
        index="crawl_logs",
        body={
            "mappings": {
                "properties": {
                    "start_ts": {"type": "date"},
                    "content":  {"type": "text", "index": False},
                }
            }
        },
    )

# ID = ngày giờ bắt đầu run (giờ local) — dùng làm document ID
DOC_ID = datetime.now().strftime("%Y-%m-%dT%H:%M")

_kb.client.index(
    index="crawl_logs",
    id=DOC_ID,
    body={"start_ts": datetime.now(timezone.utc).isoformat(), "content": ""},
)

_FLUSH_EVERY = 3  # flush lên OpenSearch mỗi N dòng


class _LiveLogger:
    def __init__(self):
        self._line_buf = ""   # buffer dòng chưa kết thúc
        self._content = ""    # full content đã tích lũy
        self._pending = 0     # số dòng chưa flush

    def write(self, msg: str):
        sys.__stdout__.write(msg)
        self._line_buf += msg
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            if line.strip():
                self._content += line + "\n"
                self._pending += 1
                if self._pending >= _FLUSH_EVERY:
                    self._flush()

    def _flush(self):
        try:
            _kb.client.update(
                index="crawl_logs",
                id=DOC_ID,
                body={"doc": {"content": self._content}},
            )
            self._pending = 0
        except Exception as e:
            sys.__stdout__.write(f"[live_logger] flush failed: {e}\n")

    def flush(self):
        sys.__stdout__.flush()
        if self._pending > 0:
            self._flush()

    def isatty(self):
        return False


sys.stdout = _LiveLogger()
print(f"[live_logger] ON — doc_id={DOC_ID}")
