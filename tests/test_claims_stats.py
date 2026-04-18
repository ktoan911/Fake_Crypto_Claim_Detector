"""
Test script for GET /claims/stats endpoint.
Usage:
    python scripts/test_claims_stats.py [BASE_URL]

Default BASE_URL: http://localhost:8000
"""

import json
import sys

import requests

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
URL = f"{BASE_URL}/claims/stats"


def check(condition: bool, msg: str):
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}  {msg}")
    return condition


def main():
    print(f"\n🔍 Testing {URL}\n")

    # ── Request ──────────────────────────────────────────────────────────────
    try:
        resp = requests.get(URL, timeout=15)
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {URL}. Is the server running?")
        sys.exit(1)

    check(resp.status_code == 200, f"HTTP status == 200 (got {resp.status_code})")

    data = resp.json()
    print(
        f"\n  Raw response (truncated):\n  {json.dumps(data, ensure_ascii=False, indent=2)[:800]}\n"
    )

    # ── Top-level keys ───────────────────────────────────────────────────────
    for key in ("recent_claims", "stats_24h", "daily_total", "daily_false"):
        check(key in data, f"key '{key}' present in response")

    if "status" in data and data.get("status") == "error":
        print(f"\n  ⚠️  API returned an error: {data.get('error')}")
        sys.exit(1)

    # ── recent_claims ─────────────────────────────────────────────────────────
    rc = data.get("recent_claims", [])
    check(isinstance(rc, list), "recent_claims is a list")
    check(len(rc) <= 10, f"recent_claims len <= 10 (got {len(rc)})")
    if rc:
        first = rc[0]
        for field in ("claim", "verdict", "checked_at"):
            check(field in first, f"recent_claims[0] has field '{field}'")

    # ── stats_24h ────────────────────────────────────────────────────────────
    s = data.get("stats_24h", {})
    check(isinstance(s, dict), "stats_24h is a dict")
    for field in ("đúng", "sai", "chưa chắc chắn", "percent_đúng", "percent_sai"):
        check(field in s, f"stats_24h has field '{field}'")

    total = s.get("đúng", 0) + s.get("sai", 0) + s.get("chưa chắc chắn", 0)
    if total > 0:
        pct_ok = 0 <= s.get("percent_đúng", -1) <= 100
        pct_sai = 0 <= s.get("percent_sai", -1) <= 100
        check(pct_ok, f"percent_đúng in [0, 100] (got {s.get('percent_đúng')})")
        check(pct_sai, f"percent_sai in [0, 100] (got {s.get('percent_sai')})")

    # ── daily_total / daily_false ─────────────────────────────────────────────
    dt = data.get("daily_total", {})
    df = data.get("daily_false", {})
    check(isinstance(dt, dict), "daily_total is a dict")
    check(isinstance(df, dict), "daily_false is a dict")
    check(len(dt) <= 7, f"daily_total has <= 7 entries (got {len(dt)})")

    # daily_false count should never exceed daily_total for same day
    for day, false_cnt in df.items():
        total_cnt = dt.get(day, 0)
        check(
            false_cnt <= total_cnt,
            f"daily_false[{day}]={false_cnt} <= daily_total[{day}]={total_cnt}",
        )

    print("\nDone.\n")


if __name__ == "__main__":
    main()
