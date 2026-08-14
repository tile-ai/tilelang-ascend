"""Sigmoid performance benchmark: tilelang kernel vs torch.sigmoid baseline.

Usage:
    python custom/sigmoid/perf_tuning/bench_perf.py
    python custom/sigmoid/perf_tuning/bench_perf.py --warmup 50 --iters 200

Measures:
- Main shape: (1024, 8192) float16  -- primary benchmark
- Aux  shape: (512, 512)  float32   -- secondary benchmark

Each config: warmup (default 20) + timed iters (default 100).
Reports: latency (us), throughput (GB/s for read+write), speedup vs torch.sigmoid.
"""

import argparse
import os
import sys
import time

import tilelang
import torch

# Make sibling sigmoid.py importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, OP_DIR)
from sigmoid import sigmoid  # noqa: E402


def median_us_ms(times_ms):
    """Return median latency (ms) from a list of per-iter times (ms)."""
    s = sorted(times_ms)
    n = len(s)
    if n == 0:
        return float("inf")
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def bench_fn(fn, x, warmup, iters):
    """Benchmark a function `fn(x)` on NPU. Returns (median_ms, mean_ms)."""
    # Warmup
    for _ in range(warmup):
        _ = fn(x)
    torch.npu.synchronize()
    # Timed
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn(x)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    return median_us_ms(times), sum(times) / len(times)


def fmt_throughput_gb_s(elem_count, dtype_bytes, latency_ms):
    """Read+Write bytes / time. For sigmoid: 1 input + 1 output = 2 * data."""
    total_bytes = 2 * elem_count * dtype_bytes
    return total_bytes / (latency_ms * 1e-3) / 1e9  # GB/s


def run_config(shape, dtype, block, warmup, iters):
    """Run one (shape, dtype) config and print comparison."""
    M, N = shape
    block_M, block_N = block
    dtype_bytes = 2 if dtype == "float16" else 4
    elem_count = M * N

    print(f"\n=== shape={shape} dtype={dtype} block={block} ===")

    # Compile tilelang kernel
    print("[compile] tilelang kernel ...", flush=True)
    t_compile_start = time.perf_counter()
    kernel_fn = sigmoid(M, N, block_M, block_N, dtype=dtype)
    t_compile = time.perf_counter() - t_compile_start
    print(f"[compile] done in {t_compile:.2f}s")

    # Generate input
    dt = getattr(torch, dtype)
    x = torch.randn(M, N, dtype=dt, device="npu")

    # 1. tilelang kernel
    tl_med, tl_mean = bench_fn(kernel_fn, x, warmup, iters)
    # 2. torch.sigmoid baseline
    pt_med, pt_mean = bench_fn(torch.sigmoid, x, warmup, iters)

    speedup_med = pt_med / tl_med if tl_med > 0 else 0.0
    tl_tput = fmt_throughput_gb_s(elem_count, dtype_bytes, tl_med)
    pt_tput = fmt_throughput_gb_s(elem_count, dtype_bytes, pt_med)

    print(f"  torch.sigmoid   : median={pt_med:.4f} ms  mean={pt_mean:.4f} ms  tput={pt_tput:.2f} GB/s")
    print(f"  tilelang kernel : median={tl_med:.4f} ms  mean={tl_mean:.4f} ms  tput={tl_tput:.2f} GB/s")
    print(f"  speedup (pt/tl) : {speedup_med:.3f}x  ({'kernel FASTER' if speedup_med > 1.0 else 'kernel SLOWER'})")

    return {
        "shape": shape,
        "dtype": dtype,
        "block": block,
        "tl_median_ms": tl_med,
        "tl_mean_ms": tl_mean,
        "pt_median_ms": pt_med,
        "pt_mean_ms": pt_mean,
        "speedup": speedup_med,
        "tl_throughput_gbs": tl_tput,
        "pt_throughput_gbs": pt_tput,
    }


def dump_kernel_source(shape, dtype, block):
    """Print translated Ascend C source to determine kernel type (AIC/AIV)."""
    M, N = shape
    block_M, block_N = block
    print(f"\n[get_kernel_source] shape={shape} dtype={dtype}")
    try:
        fn = sigmoid(M, N, block_M, block_N, dtype=dtype)
        src = fn.get_kernel_source()
        # Print only the diagnostic header lines (avoid dumping whole file)
        has_aic = "IS_ASCEND_AIC" in src
        has_aiv = "IS_ASCEND_AIV" in src
        has_mix = "KERNEL_TYPE_MIX" in src
        ktype = "MIX" if (has_aic and has_aiv) else ("AIC" if has_aic else ("AIV" if has_aiv else "UNKNOWN"))
        print(f"  kernel_type: {ktype} (AIC={has_aic} AIV={has_aiv} MIX={has_mix})")
        # Save full source for later inspection
        src_path = os.path.join(SCRIPT_DIR, f"kernel_source_{dtype}_{M}x{N}.cpp")
        with open(src_path, "w") as f:
            f.write(src)
        print(f"  full source saved to: {src_path}")
        return ktype
    except Exception as e:
        print(f"  [WARN] get_kernel_source failed: {e}")
        return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--source-only", action="store_true", help="only dump kernel source then exit")
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    configs = [
        # Primary: large float16
        ((1024, 8192), "float16", (128, 128)),
        # Secondary: float32
        ((512, 512), "float32", (128, 128)),
    ]

    # Always dump kernel source for type diagnosis
    ktype = dump_kernel_source(configs[0][0], configs[0][1], configs[0][2])
    if args.source_only:
        return

    results = []
    for shape, dtype, block in configs:
        r = run_config(shape, dtype, block, args.warmup, args.iters)
        results.append(r)

    print("\n=== SUMMARY ===")
    print(f"{'shape':<20} {'dtype':<10} {'pt_ms':<10} {'tl_ms':<10} {'speedup':<10} {'verdict'}")
    for r in results:
        verdict = "FASTER" if r["speedup"] > 1.03 else ("SLOWER" if r["speedup"] < 0.97 else "PARITY")
        print(
            f"{str(r['shape']):<20} {r['dtype']:<10} {r['pt_median_ms']:<10.4f} {r['tl_median_ms']:<10.4f} {r['speedup']:<10.3f} {verdict}"
        )

    # Save json summary
    import json

    summary = {
        "kernel_type": ktype,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }
    out_json = os.path.join(SCRIPT_DIR, "bench_perf_result.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON saved to: {out_json}")


if __name__ == "__main__":
    main()
