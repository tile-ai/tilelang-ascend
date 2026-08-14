"""Mish cann-bench 20-case performance benchmark.

Measures end-to-end latency (host adapter + kernel launch + sync) for both
TileLang kernel and PyTorch torch.nn.functional.mish baseline, aligned with
cann-bench HAP scoring methodology.

Note: cann-bench timing includes host-side overhead, so local bench measures
end-to-end (not msprof op kernel-only). For element-wise ops, host-side tiling
is the primary bottleneck (see tilelang-perf-optimization Step 5.5).

Usage:
    source set_env.sh
    python custom/mish/perf_tuning/bench_perf.py [--label baseline|iterN]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

# Add custom/mish to path for kernel import
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
OP_DIR = HERE.parent
sys.path.insert(0, str(OP_DIR))
from mish import mish_forward  # noqa: E402


# ========== cann-bench 20 standard cases ==========
# (case_id, shape, dtype_str, value_range_for_perf)
# Special-value cases (12: inf, 13: nan, 14: zero) use [-1,1] for perf measurement
CANN_BENCH_CASES = [
    (1, [1024, 1024], "float16", (-1, 1)),
    (2, [2048, 2048], "float32", (-2, 2)),
    (3, [4096, 4096], "bfloat16", (-3, 3)),
    (4, [8192, 8192], "float16", (-10, 10)),
    (5, [8192, 8192], "float32", (-100, 100)),
    (6, [1023, 1023], "bfloat16", (-0.1, 0.1)),
    (7, [1009, 1021], "float16", (-1, 2)),
    (8, [1537, 769], "float32", (-5, 10)),
    (9, [363, 367, 373], "bfloat16", (-50, 100)),
    (10, [2049, 513], "float16", (-65504, 65504)),
    (11, [3, 7, 13, 4001], "float32", (-88, 88)),
    (12, [1000003], "bfloat16", (-1, 1)),       # original [-inf,inf] -> [-1,1] for perf
    (13, [11, 13, 17, 67, 67], "float32", (-1, 1)),  # original [nan,nan] -> [-1,1] for perf
    (14, [3, 7, 11, 13, 1009], "float16", (-1, 1)),  # original [0,0] -> [-1,1] for perf
    (15, [512, 2049], "float32", (-0.5, 0.5)),
    (16, [255, 8193], "bfloat16", (-1, 3)),
    (17, [4097, 511], "float16", (-1000, 1000)),
    (18, [2, 511, 2049], "float32", (-0.2, 0.2)),
    (19, [4, 255, 2049], "bfloat16", (-3, 6)),
    (20, [2, 3, 17, 1024, 101], "float32", (-20, 40)),
]

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

WARMUP_ITERS = 5
TIMED_ITERS = 20


def _gen_input(shape, dtype_str, vrange, seed=42):
    """Generate uniform random input on NPU."""
    dt = _DTYPE_MAP[dtype_str]
    lo, hi = vrange
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(shape, generator=gen, dtype=torch.float32) * (hi - lo) + lo
    return x.to(dt).npu()


def _measure_latency(fn, x, warmup=WARMUP_ITERS, iters=TIMED_ITERS):
    """Measure end-to-end latency (ms), return median of `iters` runs."""
    # Warmup
    for _ in range(warmup):
        _ = fn(x)
    torch.npu.synchronize()
    # Timed
    times_ms = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _ = fn(x)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    times_ms.sort()
    return times_ms[len(times_ms) // 2]  # median


def _torch_mish(x):
    """PyTorch baseline: torch.nn.functional.mish (in-place on NPU)."""
    return torch.nn.functional.mish(x)


def run_bench(label="baseline"):
    """Run 20-case benchmark, return results list and summary dict."""
    results = []
    for case_id, shape, dtype_str, vrange in CANN_BENCH_CASES:
        x = _gen_input(shape, dtype_str, vrange)
        numel = 1
        for d in shape:
            numel *= d
        try:
            kernel_ms = _measure_latency(lambda xx: mish_forward(xx), x)
            baseline_ms = _measure_latency(_torch_mish, x)
            speedup = baseline_ms / kernel_ms if kernel_ms > 0 else 0.0
            results.append({
                "case_id": case_id,
                "shape": shape,
                "dtype": dtype_str,
                "numel": numel,
                "kernel_ms": round(kernel_ms, 4),
                "baseline_ms": round(baseline_ms, 4),
                "speedup": round(speedup, 4),
                "status": "ok",
            })
            print(f"[case {case_id:2d}] shape={str(shape):25s} {dtype_str:8s} "
                  f"kernel={kernel_ms:8.4f}ms baseline={baseline_ms:8.4f}ms "
                  f"speedup={speedup:.4f}x")
        except Exception as e:
            results.append({
                "case_id": case_id,
                "shape": shape,
                "dtype": dtype_str,
                "numel": numel,
                "kernel_ms": None,
                "baseline_ms": None,
                "speedup": None,
                "status": f"error: {type(e).__name__}: {e}",
            })
            print(f"[case {case_id:2d}] ERROR: {type(e).__name__}: {e}")

    # Summary
    ok_results = [r for r in results if r["status"] == "ok"]
    speedups = [r["speedup"] for r in ok_results]
    mean_speedup = sum(speedups) / len(speedups) if speedups else 0.0
    summary = {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_cases": len(results),
        "ok_cases": len(ok_results),
        "mean_speedup": round(mean_speedup, 4),
        "min_speedup": round(min(speedups), 4) if speedups else 0.0,
        "max_speedup": round(max(speedups), 4) if speedups else 0.0,
        "results": results,
    }
    print(f"\n[{label}] mean_speedup={mean_speedup:.4f}x "
          f"(min={min(speedups):.4f}, max={max(speedups):.4f}, {len(ok_results)}/{len(results)} ok)")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Mish cann-bench 20-case perf benchmark")
    parser.add_argument("--label", default="baseline", help="Label for this run (e.g. baseline, iter1)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: perf_tuning/{label}.json)")
    args = parser.parse_args()

    torch.manual_seed(0)
    # Use NPU device 0
    torch.npu.set_device(0)

    out_path = args.output or str(HERE / f"{args.label}.json")
    summary = run_bench(label=args.label)

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
