"""Single-shot sigmoid kernel runner for msprof op profiling.

msprof op captures exactly one kernel launch. This script runs the kernel
once (with proper warmup so the kernel is compiled & cached) and exits,
leaving the profiling data for analysis.

Usage:
    msprof op --kernel-name=main_kernel --output=./msprof_output \
        python custom/sigmoid/perf_tuning/run_once_for_msprof.py
"""

import os
import sys

import tilelang
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, OP_DIR)
from sigmoid import sigmoid  # noqa: E402

SHAPE = (1024, 8192)
DTYPE = "float16"
BLOCK = (128, 128)


def main():
    tilelang.disable_cache()
    torch.manual_seed(0)

    M, N = SHAPE
    block_M, block_N = BLOCK

    # Compile (this happens outside the profiled region)
    kernel_fn = sigmoid(M, N, block_M, block_N, dtype=DTYPE)

    # Prepare input on NPU
    dt = getattr(torch, DTYPE)
    x = torch.randn(M, N, dtype=dt, device="npu")

    # Warmup once (so the single captured launch is steady-state)
    _ = kernel_fn(x)
    torch.npu.synchronize()

    # The single profiled launch
    y = kernel_fn(x)
    torch.npu.synchronize()

    # Quick correctness sanity (will not affect profiling data)
    ref = torch.sigmoid(x)
    max_abs = (y.cpu().float() - ref.cpu().float()).abs().max().item()
    print(f"[run_once] shape={SHAPE} dtype={DTYPE} max_abs={max_abs:.3e}")


if __name__ == "__main__":
    main()
