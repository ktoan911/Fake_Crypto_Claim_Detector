"""
Mỗi lần chạy tạo 1 document duy nhất, id = ngày giờ bắt đầu (VD: 2024-01-15T08:30).
Mỗi print() append thêm dòng vào field `content`, cách nhau bằng \\n.
"""

import sys
from datetime import datetime, timezone

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

# ID = ngày giờ bắt đầu run — dùng làm document ID
DOC_ID = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")

_kb.client.index(
    index="crawl_logs",
    id=DOC_ID,
    body={"start_ts": datetime.now(timezone.utc).isoformat(), "content": ""},
)


class _LiveLogger:
    _buf = ""

    def write(self, msg: str):
        sys.__stdout__.write(msg)
        self._buf += msg
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._append(line)

    def _append(self, line: str):
        try:
            _kb.client.update(
                index="crawl_logs",
                id=DOC_ID,
                body={
                    "script": {
                        "source": "ctx._source.content += params.line",
                        "lang": "painless",
                        "params": {"line": line + "\n"},
                    }
                },
            )
        except Exception as e:
            sys.__stdout__.write(f"[live_logger] append failed: {e}\n")

    def flush(self):
        sys.__stdout__.flush()

    def isatty(self):
        return False


sys.stdout = _LiveLogger()
print(f"[live_logger] ON — doc_id={DOC_ID}")
