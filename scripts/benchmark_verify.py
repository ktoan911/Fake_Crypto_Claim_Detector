"""
Benchmark thời gian từng bước trong verify API.

Chạy:
    python scripts/benchmark_verify.py
    python scripts/benchmark_verify.py --url http://localhost:7860 --n 3
    python scripts/benchmark_verify.py --claim "Bitcoin sẽ lên 200k USD năm 2025"
"""

import argparse
import statistics
import sys
import time

import requests

DEFAULT_CLAIMS = [
    "Bitcoin dự kiến sẽ tăng lên 200,000 USD vào cuối năm 2025.",
    "Ngân hàng Nhà nước Việt Nam đã tăng lãi suất cơ bản lên 6% trong tháng 3 năm 2024.",
    "VN-Index đạt mức kỷ lục 1,500 điểm trong năm 2024.",
]

STEP_ORDER = [
    "split_ms",
    "retrieval_ms",
    "nli_ms",
    "llm_ms",
    "fusion_ms",
    "total_ms",
    "cache_check_ms",
    "executor_queue_ms",
    "inference_ms",
    "api_total_ms",
]

STEP_LABELS = {
    "split_ms": "Sub-claim split (LLM)",
    "retrieval_ms": "Retrieval (BM25 + vector)",
    "nli_ms": "NLI features",
    "llm_ms": "LLM scoring",
    "fusion_ms": "Fusion model",
    "total_ms": "Total inference",
    "cache_check_ms": "Cache check",
    "executor_queue_ms": "Executor queue wait",
    "inference_ms": "Inference in executor",
    "api_total_ms": "API total (round-trip server)",
    "cache_hit": "Cache hit",
}


def _bar(ms: float, max_ms: float, width: int = 30) -> str:
    if max_ms <= 0:
        return ""
    filled = int(round(width * ms / max_ms))
    return "█" * filled + "░" * (width - filled)


def call_verify(url: str, claim: str, timeout: int = 400) -> dict:
    t0 = time.perf_counter()
    resp = requests.post(
        f"{url}/verify",
        json={"claim": claim},
        timeout=timeout,
    )
    client_ms = round(1000.0 * (time.perf_counter() - t0), 1)
    resp.raise_for_status()
    data = resp.json()
    data["_client_total_ms"] = client_ms
    return data


def print_timing(timing: dict, client_ms: float, run_idx: int, cache_hit: bool = False):
    tag = "  [CACHE HIT]" if cache_hit else ""
    print(f"\n  Run #{run_idx}{tag}")
    if cache_hit:
        cache_ms = timing.get("cache_check_ms", 0.0)
        api_ms = timing.get("api_total_ms", 0.0)
        print(f"    {'Cache check':<30} {cache_ms:>8.1f} ms")
        print(f"    {'API total (round-trip server)':<30} {api_ms:>8.1f} ms")
        print(f"    {'Client round-trip':<30} {client_ms:>8.1f} ms")
        return

    all_steps = STEP_ORDER + [k for k in timing if k not in STEP_ORDER]
    max_ms = max(
        (
            timing.get(k, 0)
            for k in all_steps
            if isinstance(timing.get(k), (int, float))
        ),
        default=1,
    )

    for key in all_steps:
        val = timing.get(key)
        if val is None or not isinstance(val, (int, float)):
            continue
        label = STEP_LABELS.get(key, key)
        bar = _bar(val, max_ms)
        print(f"    {label:<30} {val:>8.1f} ms  {bar}")
    print(f"    {'Client round-trip':<30} {client_ms:>8.1f} ms")


def benchmark(url: str, claim: str, n: int, warm_cache: bool):
    print(f"\n{'=' * 65}")
    print(f"Claim : {claim!r}")
    print(f"URL   : {url}/verify")
    print(f"Runs  : {n}  |  warm_cache={warm_cache}")
    print(f"{'=' * 65}")

    all_runs = []
    for i in range(1, n + 1):
        try:
            result = call_verify(url, claim)
        except requests.exceptions.Timeout:
            print(f"  Run #{i}  TIMEOUT")
            continue
        except Exception as exc:
            print(f"  Run #{i}  ERROR: {exc}")
            continue

        if result.get("status") != "success":
            print(f"  Run #{i}  API error: {result.get('error')}")
            continue

        timing = result.get("timing_ms") or {}
        client_ms = result.pop("_client_total_ms", 0.0)
        is_cache_hit = timing.get("cache_hit", False)
        print_timing(timing, client_ms, i, cache_hit=is_cache_hit)
        if not is_cache_hit:
            all_runs.append((timing, client_ms))

        if not warm_cache:
            # Small pause between cold runs so cache doesn't kick in
            time.sleep(0.2)

    if len(all_runs) >= 2:
        print(f"\n  {'─' * 55}")
        print(f"  Summary ({len(all_runs)} runs):")
        all_steps = STEP_ORDER + [
            k for k in (all_runs[0][0] if all_runs else {}) if k not in STEP_ORDER
        ]
        for key in all_steps:
            vals = [r[0].get(key) for r in all_runs if r[0].get(key) is not None]
            if not vals:
                continue
            label = STEP_LABELS.get(key, key)
            print(
                f"    {label:<30}  min={min(vals):>8.1f}  "
                f"avg={statistics.mean(vals):>8.1f}  "
                f"max={max(vals):>8.1f} ms"
            )
        client_vals = [r[1] for r in all_runs]
        print(
            f"    {'Client round-trip':<30}  min={min(client_vals):>8.1f}  "
            f"avg={statistics.mean(client_vals):>8.1f}  "
            f"max={max(client_vals):>8.1f} ms"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark /verify endpoint timing")
    parser.add_argument(
        "--url", default="http://localhost:7860", help="Base URL của API server"
    )
    parser.add_argument(
        "--claim",
        default=None,
        help="Claim cần kiểm tra (nếu không truyền, dùng 3 claim mẫu)",
    )
    parser.add_argument("--n", type=int, default=2, help="Số lần gọi mỗi claim")
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Không sleep giữa các run (test warm cache)",
    )
    args = parser.parse_args()

    # Health check
    try:
        r = requests.get(f"{args.url}/health", timeout=5)
        health = r.json()
        print(f"Health: {health}")
        if not health.get("model_loaded"):
            print("WARN: model_loaded=False — verifier chưa load xong.")
    except Exception as exc:
        print(f"Cannot reach {args.url}/health: {exc}")
        sys.exit(1)

    claims = [args.claim] if args.claim else DEFAULT_CLAIMS
    for claim in claims:
        benchmark(args.url, claim, n=args.n, warm_cache=args.warm_cache)

    print()


if __name__ == "__main__":
    main()
