from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable

import psutil


UVICORN_PATTERN = ("uvicorn", "scripts.api_server")
NPM_DEV_PATTERN = ("npm", "run", "dev")


@dataclass
class GroupStats:
    label: str
    pids: set[int] = field(default_factory=set)
    cpu_samples: list[float] = field(default_factory=list)
    rss_samples: list[int] = field(default_factory=list)

    def add(self, cpu: float, rss: int, pids: Iterable[int]) -> None:
        self.cpu_samples.append(cpu)
        self.rss_samples.append(rss)
        self.pids = set(pids)

    def summary(self) -> dict[str, float]:
        if not self.cpu_samples:
            return {"avg_cpu": 0.0, "max_cpu": 0.0, "avg_rss_mb": 0.0, "max_rss_mb": 0.0}
        return {
            "avg_cpu": sum(self.cpu_samples) / len(self.cpu_samples),
            "max_cpu": max(self.cpu_samples),
            "avg_rss_mb": sum(self.rss_samples) / len(self.rss_samples) / 1024 / 1024,
            "max_rss_mb": max(self.rss_samples) / 1024 / 1024,
        }


def cmdline_matches(cmdline: list[str], needles: tuple[str, ...]) -> bool:
    joined = " ".join(cmdline)
    return all(n in joined for n in needles)


def find_root_processes() -> tuple[list[psutil.Process], list[psutil.Process]]:
    uvicorn_roots: list[psutil.Process] = []
    npm_roots: list[psutil.Process] = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = p.info["cmdline"] or []
            if not cmd:
                continue
            if cmdline_matches(cmd, UVICORN_PATTERN):
                uvicorn_roots.append(p)
            elif cmdline_matches(cmd, NPM_DEV_PATTERN):
                npm_roots.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return uvicorn_roots, npm_roots


def expand_tree(roots: list[psutil.Process]) -> list[psutil.Process]:
    seen: dict[int, psutil.Process] = {}
    for root in roots:
        try:
            seen[root.pid] = root
            for child in root.children(recursive=True):
                seen[child.pid] = child
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return list(seen.values())


def sample_group(procs: list[psutil.Process]) -> tuple[float, int, list[int]]:
    cpu_total = 0.0
    rss_total = 0
    pids: list[int] = []
    for p in procs:
        try:
            cpu_total += p.cpu_percent(interval=None)
            rss_total += p.memory_info().rss
            pids.append(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return cpu_total, rss_total, pids


def prime_cpu(procs: list[psutil.Process]) -> None:
    for p in procs:
        try:
            p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def system_used_mb() -> float:
    vm = psutil.virtual_memory()
    # "used" theo cách Linux hiểu: total - available (loại bỏ buffer/cache có thể giải phóng).
    return (vm.total - vm.available) / 1024 / 1024


def wait_for_processes(timeout: float = 60.0) -> tuple[list[psutil.Process], list[psutil.Process]]:
    deadline = time.monotonic() + timeout
    last = ([], [])
    while time.monotonic() < deadline:
        uv, npm = find_root_processes()
        last = (uv, npm)
        if uv and npm:
            return uv, npm
        time.sleep(0.5)
    return last


def fmt_row(t: float, uv: GroupStats, npm: GroupStats,
            sys_used_now: float, sys_baseline: float | None,
            cpu_now: float, cpu_baseline: float | None) -> str:
    uv_cpu = uv.cpu_samples[-1] if uv.cpu_samples else 0.0
    uv_mem = uv.rss_samples[-1] / 1024 / 1024 if uv.rss_samples else 0.0
    npm_cpu = npm.cpu_samples[-1] if npm.cpu_samples else 0.0
    npm_mem = npm.rss_samples[-1] / 1024 / 1024 if npm.rss_samples else 0.0
    proc_total_cpu = uv_cpu + npm_cpu
    proc_total_mem = uv_mem + npm_mem

    line = (
        f"t={t:6.1f}s | "
        f"uvicorn cpu={uv_cpu:6.1f}% ram={uv_mem:7.1f}MB | "
        f"npm cpu={npm_cpu:6.1f}% ram={npm_mem:7.1f}MB | "
        f"2 lệnh: cpu={proc_total_cpu:6.1f}% ram={proc_total_mem:7.1f}MB"
    )
    if sys_baseline is not None and cpu_baseline is not None:
        delta_mem = sys_used_now - sys_baseline
        delta_cpu = cpu_now - cpu_baseline
        line += f" | Δsystem ram=+{delta_mem:6.1f}MB cpu=+{delta_cpu:5.1f}%"
    return line


def print_summary(uv: GroupStats, npm: GroupStats, num_cpus: int,
                  sys_baseline: float | None, sys_used_samples: list[float],
                  cpu_baseline: float | None, cpu_samples: list[float]) -> None:
    print("\n=== Tổng kết ===")
    print(f"Logical CPUs: {num_cpus}")
    print(f"(CPU% tính theo psutil: 100% = 1 core full, nên có thể vượt 100%)")
    print()
    print("→ RAM/CPU thực dùng bởi 2 process tree (per-process, không tính task khác):")
    for g in (uv, npm):
        s = g.summary()
        print(
            f"  {g.label:>10}: "
            f"avg cpu={s['avg_cpu']:6.1f}%  max cpu={s['max_cpu']:6.1f}%  | "
            f"avg ram={s['avg_rss_mb']:7.1f}MB  max ram={s['max_rss_mb']:7.1f}MB"
        )
    uv_s, npm_s = uv.summary(), npm.summary()
    tot_avg_cpu = uv_s["avg_cpu"] + npm_s["avg_cpu"]
    tot_max_cpu = uv_s["max_cpu"] + npm_s["max_cpu"]
    tot_avg_mem = uv_s["avg_rss_mb"] + npm_s["avg_rss_mb"]
    tot_max_mem = uv_s["max_rss_mb"] + npm_s["max_rss_mb"]
    print(
        f"  {'TỔNG':>10}: "
        f"avg cpu={tot_avg_cpu:6.1f}%  max cpu={tot_max_cpu:6.1f}%  | "
        f"avg ram={tot_avg_mem:7.1f}MB  max ram={tot_max_mem:7.1f}MB"
    )

    if sys_baseline is not None and sys_used_samples:
        print()
        print("→ Delta hệ thống (bắt cả overhead gián tiếp: tmpfs, kernel cache, GPU driver…):")
        deltas_mem = [s - sys_baseline for s in sys_used_samples]
        deltas_cpu = [c - (cpu_baseline or 0.0) for c in cpu_samples]
        avg_dm = sum(deltas_mem) / len(deltas_mem)
        max_dm = max(deltas_mem)
        avg_dc = sum(deltas_cpu) / len(deltas_cpu) if deltas_cpu else 0.0
        max_dc = max(deltas_cpu) if deltas_cpu else 0.0
        print(f"  baseline RAM used: {sys_baseline:.1f} MB | baseline CPU: {cpu_baseline:.1f}%")
        print(f"  Δ RAM  : avg=+{avg_dm:.1f} MB   max=+{max_dm:.1f} MB")
        print(f"  Δ CPU% : avg=+{avg_dc:.1f}%    max=+{max_dc:.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--interval", type=float, default=1.0, help="khoảng sample (s), mặc định 1")
    ap.add_argument("-d", "--duration", type=float, default=0.0, help="tổng thời gian đo (s), 0 = chạy đến khi Ctrl-C")
    ap.add_argument("--csv", type=str, default=None, help="ghi log từng sample ra CSV")
    ap.add_argument("--baseline", action="store_true",
                    help="đo baseline TRƯỚC khi 2 lệnh chạy, sau đó tính delta hệ thống")
    args = ap.parse_args()

    num_cpus = psutil.cpu_count(logical=True) or 1

    sys_baseline: float | None = None
    cpu_baseline: float | None = None
    if args.baseline:
        print("CHẾ ĐỘ BASELINE — đảm bảo uvicorn và npm CHƯA chạy.")
        input("Nhấn Enter để chụp baseline RAM/CPU hệ thống… ")
        # Mồi cpu_percent toàn hệ thống (lần đầu trả 0.0).
        psutil.cpu_percent(interval=None)
        time.sleep(1.0)
        cpu_baseline = psutil.cpu_percent(interval=None)
        sys_baseline = system_used_mb()
        print(f"  baseline: RAM used = {sys_baseline:.1f} MB | CPU = {cpu_baseline:.1f}%")
        print("→ Bây giờ start 2 lệnh ở 2 terminal khác.")
        print("→ Script đang đợi 2 process xuất hiện (tối đa 60s)…")
        uv_roots, npm_roots = wait_for_processes(timeout=60.0)
    else:
        uv_roots, npm_roots = find_root_processes()

    if not uv_roots:
        print("! Không thấy `uvicorn scripts.api_server`.")
    else:
        print(f"  uvicorn roots: {[p.pid for p in uv_roots]}")
    if not npm_roots:
        print("! Không thấy `npm run dev`.")
    else:
        print(f"  npm run dev roots: {[p.pid for p in npm_roots]}")
    if not uv_roots and not npm_roots:
        return 1

    uv_procs = expand_tree(uv_roots)
    npm_procs = expand_tree(npm_roots)
    prime_cpu(uv_procs + npm_procs)
    if args.baseline:
        psutil.cpu_percent(interval=None)  # mồi lại để delta CPU hệ thống chính xác
    time.sleep(args.interval)

    uv_stats = GroupStats("uvicorn")
    npm_stats = GroupStats("npm-dev")
    sys_used_samples: list[float] = []
    cpu_samples: list[float] = []

    csv_writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "t_seconds", "uvicorn_cpu_pct", "uvicorn_rss_mb",
            "npm_cpu_pct", "npm_rss_mb",
            "proc_total_cpu_pct", "proc_total_rss_mb",
            "system_used_mb", "system_cpu_pct",
            "delta_system_used_mb", "delta_system_cpu_pct",
        ])

    stop = False

    def _stop(_sig, _frm):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    start = time.monotonic()
    try:
        while not stop:
            uv_procs = expand_tree(uv_roots)
            npm_procs = expand_tree(npm_roots)

            uv_cpu, uv_rss, uv_pids = sample_group(uv_procs)
            npm_cpu, npm_rss, npm_pids = sample_group(npm_procs)
            uv_stats.add(uv_cpu, uv_rss, uv_pids)
            npm_stats.add(npm_cpu, npm_rss, npm_pids)

            sys_used = system_used_mb()
            sys_cpu = psutil.cpu_percent(interval=None)
            sys_used_samples.append(sys_used)
            cpu_samples.append(sys_cpu)

            elapsed = time.monotonic() - start
            print(fmt_row(elapsed, uv_stats, npm_stats,
                          sys_used, sys_baseline, sys_cpu, cpu_baseline))

            if csv_writer:
                d_mem = (sys_used - sys_baseline) if sys_baseline is not None else ""
                d_cpu = (sys_cpu - cpu_baseline) if cpu_baseline is not None else ""
                csv_writer.writerow([
                    f"{elapsed:.2f}", f"{uv_cpu:.2f}", f"{uv_rss / 1024 / 1024:.2f}",
                    f"{npm_cpu:.2f}", f"{npm_rss / 1024 / 1024:.2f}",
                    f"{(uv_cpu + npm_cpu):.2f}", f"{(uv_rss + npm_rss) / 1024 / 1024:.2f}",
                    f"{sys_used:.2f}", f"{sys_cpu:.2f}",
                    f"{d_mem:.2f}" if isinstance(d_mem, float) else "",
                    f"{d_cpu:.2f}" if isinstance(d_cpu, float) else "",
                ])
                csv_file.flush()

            if args.duration and elapsed >= args.duration:
                break
            time.sleep(args.interval)
    finally:
        if csv_file:
            csv_file.close()
        print_summary(uv_stats, npm_stats, num_cpus,
                      sys_baseline, sys_used_samples, cpu_baseline, cpu_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
