"""
Đo thời gian gọi POST /verify cho một claim.

Chạy:
    python tests/test_verify_latency.py
    python tests/test_verify_latency.py --url http://localhost:7860 --claim "Bitcoin sẽ đạt 1 triệu USD vào năm 2025"
"""

import argparse
import json
import time
import urllib.request
import urllib.error

DEFAULT_URL = "http://localhost:7860"
DEFAULT_CLAIM = "Bitcoin sẽ tăng lên 200,000 USD vào cuối năm 2025 theo dự báo của các chuyên gia tài chính hàng đầu."


def verify(base_url: str, claim: str) -> tuple[dict, float]:
    payload = json.dumps({"claim": claim}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/verify",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    return body, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL của API server")
    parser.add_argument("--claim", default=DEFAULT_CLAIM, help="Claim cần kiểm chứng")
    args = parser.parse_args()

    print(f"URL  : {args.url}/verify")
    print(f"Claim: {args.claim}")
    print("-" * 60)

    try:
        result, elapsed = verify(args.url, args.claim)
    except urllib.error.URLError as e:
        print(f"Lỗi kết nối: {e.reason}")
        print("Kiểm tra server đã chạy chưa: uvicorn scripts.api_server:app --port 7860")
        return

    print(f"Tổng thời gian : {elapsed:.2f}s")
    print(f"Verdict        : {result.get('verdict', 'N/A')}")
    print(f"Confidence     : {result.get('confidence', 'N/A')}")
    print(f"Status         : {result.get('status', 'N/A')}")

    timing = result.get("timing_ms")
    if timing:
        retrieval = timing.get("retrieval_ms", 0)
        llm = timing.get("llm_ms", 0)
        total = timing.get("total_ms", 0)
        other = total - retrieval - llm
        print("\nTiming breakdown:")
        print(f"  Retrieval : {retrieval/1000:.2f}s")
        print(f"  LLM       : {llm/1000:.2f}s")
        print(f"  Khác      : {other/1000:.2f}s")
        print(f"  Total     : {total/1000:.2f}s  (server-side)")

    evidence = result.get("evidence", [])
    if evidence:
        print(f"\nEvidence ({len(evidence)} mục):")
        for i, ev in enumerate(evidence[:3], 1):
            snippet = ev[:120].replace("\n", " ")
            print(f"  [{i}] {snippet}{'…' if len(ev) > 120 else ''}")

    if result.get("status") == "error":
        print(f"Lỗi: {result.get('error', '')}")


if __name__ == "__main__":
    main()
