"""
Test script for GET /kaggle/logs and GET /kaggle/logs/stream endpoints.
Usage:
    python tests/test_kaggle_logs.py [BASE_URL]

Default BASE_URL: http://localhost:8000
"""

import json
import sys

import requests

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
LOGS_URL   = f"{BASE_URL}/kaggle/logs"
STREAM_URL = f"{BASE_URL}/kaggle/logs/stream"


def check(condition: bool, msg: str) -> bool:
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}  {msg}")
    return condition


# ── /kaggle/logs ──────────────────────────────────────────────────────────────

def test_logs():
    print(f"\n🔍 Testing GET {LOGS_URL}\n")

    try:
        resp = requests.get(LOGS_URL, timeout=15)
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {LOGS_URL}. Is the server running?")
        sys.exit(1)

    ok_status = check(
        resp.status_code in (200, 404),
        f"HTTP status in {{200, 404}} (got {resp.status_code})",
    )
    if not ok_status:
        return

    data = resp.json()
    print(f"\n  Raw response (truncated):\n  {json.dumps(data, ensure_ascii=False, indent=2)[:600]}\n")

    if resp.status_code == 404:
        check("error" in data, "404 response has 'error' field")
        print("  ℹ️  No log document found yet (index empty or not created).")
        return

    # 200 — normal response
    for field in ("doc_id", "start_ts", "content", "running"):
        check(field in data, f"field '{field}' present")

    check(isinstance(data.get("running"), bool), "'running' is bool")
    check(isinstance(data.get("content"), str),  "'content' is str")
    check(isinstance(data.get("doc_id"), str),   "'doc_id' is str")

    # doc_id must look like ISO datetime prefix YYYY-MM-DDTHH:MM
    doc_id: str = data.get("doc_id", "")
    check(
        len(doc_id) >= 16 and "T" in doc_id,
        f"'doc_id' looks like ISO datetime (got {doc_id!r})",
    )


# ── /kaggle/logs/stream (SSE) ─────────────────────────────────────────────────

def test_stream():
    print(f"\n🔍 Testing GET {STREAM_URL} (SSE, read first 3 events)\n")

    try:
        with requests.get(STREAM_URL, stream=True, timeout=20) as resp:
            check(resp.status_code == 200, f"HTTP status == 200 (got {resp.status_code})")
            check(
                "text/event-stream" in resp.headers.get("Content-Type", ""),
                f"Content-Type is text/event-stream (got {resp.headers.get('Content-Type')})",
            )

            events_seen = 0
            has_data_event = False
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                print(f"  raw: {raw_line[:120]}")

                if raw_line.startswith("data:"):
                    payload_str = raw_line[len("data:"):].strip()
                    try:
                        payload = json.loads(payload_str)
                        has_data_event = True
                        # Payload must have one of: line, done, error
                        check(
                            any(k in payload for k in ("line", "done", "error")),
                            f"data event has 'line'/'done'/'error' key: {list(payload.keys())}",
                        )
                    except json.JSONDecodeError:
                        check(False, f"data event is valid JSON: {payload_str[:80]!r}")

                events_seen += 1
                if events_seen >= 3:
                    break

            check(events_seen > 0, f"at least 1 SSE event received (got {events_seen})")

    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {STREAM_URL}. Is the server running?")
        sys.exit(1)
    except requests.exceptions.Timeout:
        check(False, "stream timed out before first event")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    test_logs()
    test_stream()
    print("\nDone.\n")


if __name__ == "__main__":
    main()
