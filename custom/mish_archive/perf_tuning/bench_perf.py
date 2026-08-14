"""Mish performance benchmark: tilelang kernel vs torch.nn.functional.mish baseline.

Usage:
    python custom/mish/perf_tuning/bench_perf.py
    python custom/mish/perf_tuning/bench_perf.py --warmup 30 --iters 100

Measures DESIGN.md §12 perf_target shape set:
    - (1024, 1024) float16 / float32 / bfloat16  -- S aligned
    - (2048, 2048) float16 / float32              -- M aligned
    - (8192, 8192) float16 / float32              -- L aligned
    - (1023, 1023) bfloat16                       -- S non-aligned
    - (1537, 769)  float32                        -- S prime non-aligned

Each config: warmup (default 30) + timed iters (default 100).
Reports: latency (ms), speedup vs torch.nn.functional.mish, mean speedup.

Target: mean_speedup >= 0.6x (per DESIGN.md §12).
"""

import argparse
import json
import os
import sys
import time

import tilelang
import torch

# Make sibling mish.py importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, OP_DIR)
from mish import mish  # noqa: E402


# ========== DESIGN.md §12 perf target shape set ==========
# (shape, dtype, block, tag)
BENCH_CONFIGS = [
    # S aligned
    ((1024, 1024), "float16", (128, 128), "S_aligned_fp16"),
    ((1024, 1024), "float32", (128, 128), "S_aligned_fp32"),
    ((1024, 1024), "bfloat16", (128, 128), "S_aligned_bf16"),
    # M aligned
    ((2048, 2048), "float16", (128, 128), "M_aligned_fp16"),
    ((2048, 2048), "float32", (128, 128), "M_aligned_fp32"),
    # L aligned
    ((8192, 8192), "float16", (128, 128), "L_aligned_fp16"),
    ((8192, 8192), "float32", (128, 128), "L_aligned_fp32"),
    # S non-aligned
    ((1023, 1023), "bfloat16", (128, 128), "S_nonalign_bf16"),
    # S prime non-aligned
    ((1537, 769), "float32", (128, 128), "S_prime_fp32"),
]


def median_ms(times_ms):
    s = sorted(times_ms)
    n = len(s)
    if n == 0:
        return float("inf")
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def bench_fn(fn, x, warmup, iters):
    """Benchmark fn(x) on NPU. Returns (median_ms, mean_ms)."""
    for _ in range(warmup):
        _ = fn(x)
    torch.npu.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn(x)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    return median_ms(times), sum(times) / len(times)


def run_config(shape, dtype, block, tag, warmup, iters):
    """Run one (shape, dtype) config: tilelang vs torch.nn.functional.mish."""
    M, N = shape
    block_M, block_N = block
    print(f"\n=== {tag}: shape={shape} dtype={dtype} block={block} ===", flush=True)

    # Compile tilelang kernel
    print("[compile] tilelang kernel ...", flush=True)
    t_compile_start = time.perf_counter()
    kernel_fn = mish(M, N, block_M, block_N, dtype=dtype)
    t_compile = time.perf_counter() - t_compile_start
    print(f"[compile] done in {t_compile:.2f}s")

    # Generate input (same value range as L0 basic: [-1, 1] for normal)
    dt = getattr(torch, dtype)
    x = torch.rand(shape, dtype=torch.float32, device="npu")
    x = (x * 2.0 - 1.0).to(dt)  # uniform in [-1, 1]

    # 1. tilelang kernel
    tl_med, tl_mean = bench_fn(kernel_fn, x, warmup, iters)
    # 2. torch.nn.functional.mish baseline
    pt_med, pt_mean = bench_fn(torch.nn.functional.mish, x, warmup, iters)

    speedup_med = pt_med / tl_med if tl_med > 0 else 0.0
    print(f"  torch.mish        : median={pt_med:.4f} ms  mean={pt_mean:.4f} ms")
    print(f"  tilelang kernel   : median={tl_med:.4f} ms  mean={tl_mean:.4f} ms")
    print(f"  speedup (pt/tl)   : {speedup_med:.3f}x  ({'FASTER' if speedup_med > 1.0 else 'SLOWER'})")

    return {
        "tag": tag,
        "shape": list(shape),
        "dtype": dtype,
        "block": list(block),
        "tl_median_ms": tl_med,
        "tl_mean_ms": tl_mean,
        "pt_median_ms": pt_med,
        "pt_mean_ms": pt_mean,
        "speedup": speedup_med,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", type=str, default=None, help="output JSON path (default: bench_perf_result.json)")
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    results = []
    for shape, dtype, block, tag in BENCH_CONFIGS:
        try:
            r = run_config(shape, dtype, block, tag, args.warmup, args.iters)
            results.append(r)
        except Exception as e:
            print(f"[ERROR] {tag}: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "tag": tag,
                    "shape": list(shape),
                    "dtype": dtype,
                    "block": list(block),
                    "error": str(e),
                    "speedup": 0.0,
                }
            )

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'tag':<22} {'dtype':<10} {'pt_ms':<10} {'tl_ms':<10} {'speedup':<10} {'verdict'}")
    valid_speedups = []
    for r in results:
        if "error" in r:
            print(f"{r['tag']:<22} {r['dtype']:<10} ERROR: {r['error']}")
            continue
        verdict = "FASTER" if r["speedup"] > 1.03 else ("SLOWER" if r["speedup"] < 0.97 else "PARITY")
        print(f"{r['tag']:<22} {r['dtype']:<10} {r['pt_median_ms']:<10.4f} {r['tl_median_ms']:<10.4f} {r['speedup']:<10.3f} {verdict}")
        valid_speedups.append(r["speedup"])

    if valid_speedups:
        mean_speedup = sum(valid_speedups) / len(valid_speedups)
        target = 0.6
        print(f"\nMean speedup: {mean_speedup:.3f}x  (target >= {target}x: {'MET' if mean_speedup >= target else 'NOT MET'})")
    else:
        mean_speedup = 0.0

    out_json = args.out or os.path.join(SCRIPT_DIR, "bench_perf_result.json")
    summary = {
        "warmup": args.warmup,
        "iters": args.iters,
        "target_speedup": 0.6,
        "mean_speedup": mean_speedup,
        "target_met": mean_speedup >= 0.6,
        "results": results,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON saved to: {out_json}")


if __name__ == "__main__":
    main()
