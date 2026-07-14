#!/usr/bin/env python3
"""
Runner: so sánh TẤT CẢ model (Together AI + LoRA + Fusion) trên
data/time_dataset.json, tối ưu cho 1 GPU NVIDIA H200 (141GB VRAM, Hopper sm_90).

Chế độ: GOLD evidence cho tất cả. Timestamp chỉ vào fusion (qua feature recency),
KHÔNG đưa vào text của LoRA/Together.

Set env cấu hình H200 TRƯỚC khi import torch, rồi gọi thẳng main() của
test_together_comparison.

Cách dùng:
  python tests/run_comparison_h200.py                 # full dataset
  python tests/run_comparison_h200.py --limit 100     # 100 mẫu cân bằng 3 nhãn
  python tests/run_comparison_h200.py --skip_together # chỉ LoRA + Fusion
  # mọi flag của test_together_comparison đều truyền thẳng được
"""

import os
import sys

# --- Cấu hình GPU H200 (phải set TRƯỚC khi torch/transformers được import) ---
# 141GB VRAM: LoRA (Qwen3-4B) chạy full fp16, KHÔNG cần 4-bit quantization.
# H200 dư VRAM → batch LLM lớn để chạy nhanh.
os.environ.setdefault("LLM_INFER_BATCH_SIZE", "32")
# Context dài cho gold evidence nhiều đoạn.
os.environ.setdefault("LLM_MAX_LENGTH", "6144")
# Chỉ dùng GPU 0 (đổi nếu máy nhiều card).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# Giảm phân mảnh VRAM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_together_comparison import main  # noqa: E402


def _inject_default_args():
    """Bơm sẵn các flag mặc định cho H200 nếu user chưa truyền.
    Vẫn cho phép override thủ công qua CLI (vd: --limit 100)."""
    argv = sys.argv[1:]

    def _has(flag):
        return any(a == flag or a.startswith(flag + "=") for a in argv)

    defaults = []
    if not _has("--run_local") and not _has("--skip_together"):
        defaults.append("--run_local")  # mặc định chạy cả local
    if not _has("--device"):
        defaults += ["--device", "cuda"]
    if not _has("--together_workers"):
        defaults += ["--together_workers", "16"]
    if not _has("--output_csv"):
        os.makedirs("results", exist_ok=True)
        defaults += ["--output_csv", "results/together_comparison_h200.csv"]

    sys.argv = [sys.argv[0]] + defaults + argv


if __name__ == "__main__":
    _inject_default_args()
    print("=" * 60)
    print(" H200 full comparison (gold mode)")
    print(f" LLM_INFER_BATCH_SIZE={os.environ['LLM_INFER_BATCH_SIZE']}"
          f"  4bit={os.environ['LLM_LOAD_IN_4BIT']}")
    print(f" args: {' '.join(sys.argv[1:])}")
    print("=" * 60)
    main()
