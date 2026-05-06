from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, List

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@dataclass
class RunMetric:
    run: int
    claims: int
    elapsed_sec: float
    qps: float
    step_details: List[dict[str, Any]] = field(default_factory=list)


def _print_header(title: str) -> None:
    print(f"\n{title}")


def _print_step(step_no: int, text: str) -> None:
    print(f"[Bước {step_no}] {text}")


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * p
    lower = int(idx)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = idx - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


def _parse_json_claims(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in f if line.strip()]
        else:
            rows = json.load(f)

    if isinstance(rows, dict):
        rows = [rows]

    claims: List[str] = []
    for row in rows:
        if isinstance(row, dict):
            value = row.get("claim") or row.get("text") or row.get("message")
        else:
            value = row
        text = str(value or "").strip()
        if text:
            claims.append(text)
    return claims


def load_claims(claims_file: str | None, inline_claims: List[str], limit: int) -> List[str]:
    claims = [str(c).strip() for c in inline_claims if str(c).strip()]

    if claims_file:
        path = Path(claims_file)
        if not path.exists():
            raise FileNotFoundError(f"Claims file not found: {claims_file}")
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            claims.extend(_parse_json_claims(path))
        else:
            with path.open("r", encoding="utf-8") as f:
                claims.extend(line.strip() for line in f if line.strip())

    if not claims:
        claims = [
            "Saigonbank đặt mục tiêu lợi nhuận trước thuế 310 tỷ đồng trong năm 2026.",
        ]

    unique_claims: List[str] = []
    seen = set()
    for claim in claims:
        if claim not in seen:
            seen.add(claim)
            unique_claims.append(claim)

    if limit > 0:
        unique_claims = unique_claims[:limit]
    return unique_claims


def run_pipeline_once_with_logs(
    verifier: Any,
    claims: List[str],
    mode: str,
    batch_size: int,
    run_name: str,
) -> tuple[int, float, List[dict[str, Any]]]:
    t_round0 = perf_counter()
    step_details: List[dict[str, Any]] = []
    processed_claims = 0

    if mode == "single":
        total_steps = len(claims)
        for idx, claim in enumerate(claims, start=1):
            print(f"  - {run_name} | bước {idx}/{total_steps}: chạy predict cho 1 claim")
            t0 = perf_counter()
            pred = verifier.predict(claim)
            elapsed = perf_counter() - t0
            processed_claims += 1
            print(
                f"    Thời gian: {elapsed:.4f} giây | Kết quả: {getattr(pred, 'verdict', 'N/A')}"
            )
            step_details.append(
                {
                    "step": idx,
                    "step_type": "predict",
                    "claims_processed": 1,
                    "elapsed_sec": elapsed,
                    "verdict": getattr(pred, "verdict", None),
                }
            )
    else:
        total_batches = (len(claims) + batch_size - 1) // batch_size
        for batch_idx, start in enumerate(range(0, len(claims), batch_size), start=1):
            batch = claims[start : start + batch_size]
            print(
                f"  - {run_name} | bước {batch_idx}/{total_batches}: chạy predict_batch cho {len(batch)} claim"
            )
            t0 = perf_counter()
            preds = verifier.predict_batch(batch)
            elapsed = perf_counter() - t0
            batch_processed = len(preds)
            processed_claims += batch_processed
            print(
                f"    Thời gian: {elapsed:.4f} giây | Đã xử lý: {batch_processed} claim"
            )
            step_details.append(
                {
                    "step": batch_idx,
                    "step_type": "predict_batch",
                    "claims_processed": batch_processed,
                    "elapsed_sec": elapsed,
                }
            )

    elapsed_total = perf_counter() - t_round0
    print(
        f"  => {run_name} hoàn tất: {elapsed_total:.4f} giây | tổng claim xử lý: {processed_claims}"
    )
    return processed_claims, elapsed_total, step_details


def benchmark_pipeline(
    verifier: Any,
    claims: List[str],
    mode: str,
    batch_size: int,
    warmup: int,
    runs: int,
) -> tuple[List[float], List[RunMetric]]:
    warmup_times: List[float] = []
    for i in range(1, max(0, warmup) + 1):
        _print_header(f"[Warmup {i}/{max(0, warmup)}]")
        _, warmup_elapsed, _ = run_pipeline_once_with_logs(
            verifier=verifier,
            claims=claims,
            mode=mode,
            batch_size=batch_size,
            run_name=f"Warmup {i}",
        )
        warmup_times.append(warmup_elapsed)

    metrics: List[RunMetric] = []
    for run in range(1, runs + 1):
        _print_header(f"[Lần chạy benchmark {run}/{runs}]")
        processed, elapsed, step_details = run_pipeline_once_with_logs(
            verifier,
            claims,
            mode=mode,
            batch_size=batch_size,
            run_name=f"Lần {run}",
        )
        qps = processed / elapsed if elapsed > 0 else 0.0
        metrics.append(
            RunMetric(
                run=run,
                claims=processed,
                elapsed_sec=elapsed,
                qps=qps,
                step_details=step_details,
            )
        )
    return warmup_times, metrics


def summarize(metrics: List[RunMetric]) -> dict[str, Any]:
    elapsed_values = [m.elapsed_sec for m in metrics]
    qps_values = [m.qps for m in metrics]
    return {
        "runs": len(metrics),
        "claims_per_run": metrics[0].claims if metrics else 0,
        "avg_sec": statistics.mean(elapsed_values) if elapsed_values else 0.0,
        "median_sec": statistics.median(elapsed_values) if elapsed_values else 0.0,
        "p95_sec": _percentile(elapsed_values, 0.95),
        "min_sec": min(elapsed_values) if elapsed_values else 0.0,
        "max_sec": max(elapsed_values) if elapsed_values else 0.0,
        "avg_qps": statistics.mean(qps_values) if qps_values else 0.0,
        "min_qps": min(qps_values) if qps_values else 0.0,
        "max_qps": max(qps_values) if qps_values else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark full runtime of fusion_inference pipeline"
    )
    parser.add_argument("--claims-file", type=str, default=None)
    parser.add_argument(
        "--claim",
        action="append",
        default=[],
        help="Inline claim text. Can pass multiple times.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Max claims to run")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--mode", choices=["single", "batch"], default="batch")
    parser.add_argument("--batch-size", type=int, default=4)

    parser.add_argument("--fusion-model", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--retriever-model", type=str, default=None)
    parser.add_argument("--opensearch-index", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--llm-evidence-top-k", type=int, default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        default=None,
        help="Enable fusion_inference debug logs",
    )
    parser.add_argument(
        "--no-debug",
        action="store_false",
        dest="debug",
        help="Disable fusion_inference debug logs",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write benchmark report to JSON file",
    )
    parser.add_argument(
        "--max-avg-sec",
        type=float,
        default=None,
        help="Optional threshold: fail if avg runtime/run is above this value",
    )
    parser.add_argument(
        "--max-p95-sec",
        type=float,
        default=None,
        help="Optional threshold: fail if p95 runtime/run is above this value",
    )
    return parser.parse_args()


def main() -> int:
    t_script0 = perf_counter()
    step_no = 1

    _print_header("===== Benchmark thời gian full pipeline fusion_inference =====")
    _print_step(step_no, "Nạp biến môi trường từ file .env")
    t0 = perf_counter()
    load_dotenv()
    print(f"  Hoàn tất trong {perf_counter() - t0:.4f} giây")
    step_no += 1

    _print_step(step_no, "Đọc tham số dòng lệnh")
    t0 = perf_counter()
    args = parse_args()
    print(f"  Hoàn tất trong {perf_counter() - t0:.4f} giây")
    step_no += 1

    _print_step(step_no, "Import module fusion_inference")
    t0 = perf_counter()
    try:
        from src.models.fusion_inference import (
            FusionClaimVerifier,
            _resolve_fusion_model_path,
        )
    except Exception as exc:
        print(f"Cannot import FusionClaimVerifier: {exc}")
        return 1
    print(f"  Hoàn tất trong {perf_counter() - t0:.4f} giây")
    step_no += 1

    _print_step(step_no, "Nạp danh sách claim để benchmark")
    t0 = perf_counter()
    claims = load_claims(args.claims_file, args.claim, limit=args.limit)
    load_claims_sec = perf_counter() - t0
    print(f"  Số claim: {len(claims)} | thời gian: {load_claims_sec:.4f} giây")
    step_no += 1

    if not claims:
        print("No valid claims to test.")
        return 1

    _print_step(step_no, "Đọc cấu hình model và OpenSearch (mặc định theo api_server.py)")
    t0 = perf_counter()
    fusion_model_input = args.fusion_model or os.getenv("FUSION_MODEL")
    if not fusion_model_input:
        print(
            "Missing fusion model path. Set --fusion-model or env FUSION_MODEL (same as scripts/api_server.py)."
        )
        return 1
    try:
        fusion_model = _resolve_fusion_model_path(fusion_model_input)
    except Exception as exc:
        print(f"Cannot resolve fusion model: {exc}")
        return 1

    llm_model = args.llm_model or os.getenv("LLM_FINETUNE")
    retriever_model = args.retriever_model or os.getenv(
        "RETRIEVER_MODEL", "AITeamVN/Vietnamese_Embedding"
    )
    opensearch_index = (
        args.opensearch_index
        or os.getenv("OPENSEARCH_INDEX_NAME")
        or os.getenv("OP_KB_NAME", "news_kb")
    )
    device = args.device or os.getenv("DEVICE", "cpu")
    llm_evidence_top_k = (
        args.llm_evidence_top_k
        if args.llm_evidence_top_k is not None
        else int(os.getenv("FUSION_LLM_EVIDENCE_TOP_K", "3"))
    )
    debug = True if args.debug is None else bool(args.debug)
    config_sec = perf_counter() - t0
    print(f"  Hoàn tất trong {config_sec:.4f} giây")
    step_no += 1

    _print_step(step_no, "Khởi tạo FusionClaimVerifier (cold start)")
    t_init0 = perf_counter()
    verifier = FusionClaimVerifier(
        fusion_model_path=fusion_model,
        opensearch_index=opensearch_index,
        llm_model_path=llm_model,
        retriever_model_path=retriever_model,
        device=device,
        llm_evidence_top_k=llm_evidence_top_k,
        debug=debug,
    )
    init_sec = perf_counter() - t_init0
    print(f"  Hoàn tất trong {init_sec:.4f} giây")
    step_no += 1

    _print_step(
        step_no,
        "Chạy warmup + benchmark, log thời gian chi tiết cho từng bước chạy",
    )
    t_bench0 = perf_counter()
    warmup_times, metrics = benchmark_pipeline(
        verifier=verifier,
        claims=claims,
        mode=args.mode,
        batch_size=max(1, args.batch_size),
        warmup=max(0, args.warmup),
        runs=max(1, args.runs),
    )
    total_benchmark_sec = perf_counter() - t_bench0
    print(f"  Hoàn tất trong {total_benchmark_sec:.4f} giây")
    step_no += 1

    summary = summarize(metrics)

    _print_step(step_no, "In tổng kết thời gian")
    print("\n===== Cấu hình benchmark =====")
    print(f"Số claim: {len(claims)}")
    print(f"Chế độ chạy: {args.mode}")
    print(f"Batch size: {max(1, args.batch_size)}")
    print(f"Số lần warmup: {max(0, args.warmup)}")
    print(f"Số lần benchmark: {max(1, args.runs)}")
    print(f"Fusion model: {fusion_model}")
    print(f"LLM model: {llm_model}")
    print(f"Retriever model: {retriever_model}")
    print(f"OpenSearch index: {opensearch_index}")
    print(f"Device: {device}")
    print(f"LLM evidence top k: {llm_evidence_top_k}")
    print(f"Debug: {debug}")

    print("\n===== Thời gian theo bước lớn =====")
    print(f"Bước nạp claim: {load_claims_sec:.4f} giây")
    print(f"Bước đọc cấu hình: {config_sec:.4f} giây")
    print(f"Bước khởi tạo verifier (cold start): {init_sec:.4f} giây")
    if warmup_times:
        for i, wt in enumerate(warmup_times, start=1):
            print(f"Warmup {i}: {wt:.4f} giây")
    print(f"Tổng thời gian benchmark (không tính init): {total_benchmark_sec:.4f} giây")

    print("\n===== Thời gian từng lần benchmark =====")
    for m in metrics:
        print(
            f"Lần {m.run}: {m.elapsed_sec:.4f} giây | claims xử lý: {m.claims} | tốc độ: {m.qps:.2f} claim/giây"
        )

    print("\n===== Tổng kết cuối =====")
    print(f"Thời gian trung bình mỗi lần chạy: {summary['avg_sec']:.4f} giây")
    print(f"Thời gian trung vị mỗi lần chạy: {summary['median_sec']:.4f} giây")
    print(f"Thời gian p95 mỗi lần chạy: {summary['p95_sec']:.4f} giây")
    print("Giải thích p95: 95% lần chạy có thời gian <= giá trị này.")
    print(f"Thời gian nhanh nhất: {summary['min_sec']:.4f} giây")
    print(f"Thời gian chậm nhất: {summary['max_sec']:.4f} giây")
    print(f"Tốc độ trung bình: {summary['avg_qps']:.2f} claim/giây")
    print(f"Tốc độ thấp nhất: {summary['min_qps']:.2f} claim/giây")
    print(f"Tốc độ cao nhất: {summary['max_qps']:.2f} claim/giây")
    print(f"Tổng thời gian toàn script: {perf_counter() - t_script0:.4f} giây")

    if args.output_json:
        _print_step(step_no + 1, "Ghi báo cáo JSON")
        report = {
            "config": {
                "claims_count": len(claims),
                "mode": args.mode,
                "batch_size": max(1, args.batch_size),
                "warmup": max(0, args.warmup),
                "runs": max(1, args.runs),
                "fusion_model": fusion_model,
                "llm_model": llm_model,
                "retriever_model": retriever_model,
                "opensearch_index": opensearch_index,
                "device": device,
                "llm_evidence_top_k": llm_evidence_top_k,
                "debug": debug,
            },
            "cold_start_init_sec": init_sec,
            "warmup_times_sec": warmup_times,
            "runs": [asdict(m) for m in metrics],
            "summary": summary,
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  Đã ghi report vào: {output_path}")

    failed = False
    if args.max_avg_sec is not None and summary["avg_sec"] > args.max_avg_sec:
        print(
            f"\nKhông đạt ngưỡng: thời gian trung bình {summary['avg_sec']:.4f} > {args.max_avg_sec:.4f} giây"
        )
        failed = True
    if args.max_p95_sec is not None and summary["p95_sec"] > args.max_p95_sec:
        print(
            f"Không đạt ngưỡng: thời gian p95 {summary['p95_sec']:.4f} > {args.max_p95_sec:.4f} giây"
        )
        failed = True

    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
