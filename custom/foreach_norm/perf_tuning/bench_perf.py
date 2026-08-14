"""ForeachNorm cann-bench performance benchmark.

Measures baseline (torch.norm per-tensor) vs ours (foreach_norm) on the 20
cann-bench cases.yaml representative cases. Outputs per-case + average speedup.

Methodology (per DESIGN.md S12.2):
  - warmup: 5 iters (not timed)
  - measure: 20 iters, take median
  - speedup = baseline_time / our_time  (>1 = we are faster)

Usage:
    python custom/foreach_norm/perf_tuning/bench_perf.py
        [--iters N] [--warmup N] [--out JSON_PATH] [--label LABEL]
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import yaml

# Make foreach_norm importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _OP_DIR)

import tilelang  # noqa: E402

tilelang.enable_cache()  # cache compiled kernels across calls

from foreach_norm import foreach_norm  # noqa: E402

# ---------------------------------------------------------------------------
# Cases (from cann-bench-master/tasks/level1/foreach_norm/cases.yaml)
# ---------------------------------------------------------------------------

CASES_YAML = "/mnt/workspace/gitCode/cann/cann-bench-master/tasks/level1/foreach_norm/cases.yaml"

_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def load_cases(path: str = CASES_YAML) -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["cases"]


def gen_tensor(shape: tuple[int, ...], dtype_str: str, vrange: tuple[float, float], scalar: float) -> torch.Tensor:
    """Generate one input tensor with controlled value range."""
    dt = _TORCH_DTYPE[dtype_str]
    lo, hi = vrange
    # value_range [0, 0] -> all zeros
    if lo == 0 and hi == 0:
        return torch.zeros(shape, dtype=dt, device="npu")
    # value_range [-.inf, .inf] -> standard normal (any finite value)
    if math.isinf(lo) and math.isinf(hi):
        t = torch.randn(shape, dtype=dt, device="npu")
        return t
    if math.isinf(lo) or math.isinf(hi):
        # one-sided inf; use randn scaled by a sane factor
        t = torch.randn(shape, dtype=dt, device="npu")
        scale = max(abs(lo) if math.isfinite(lo) else 1.0, abs(hi) if math.isfinite(hi) else 1.0)
        return t * scale
    # uniform in [lo, hi]
    t = torch.empty(shape, dtype=dt, device="npu").uniform_(lo, hi)
    # negative p: inputs must be non-zero (desc.md constraint)
    if scalar < 0 and scalar != float("-inf"):
        t[t == 0] = 1.0
    return t


def to_compute_dtype(t: torch.Tensor) -> torch.Tensor:
    """Match golden: FP16/BF16 upcast to FP32 for baseline torch.norm."""
    if t.dtype in (torch.float16, torch.bfloat16):
        return t.to(torch.float32)
    return t


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _sync():
    torch.npu.synchronize()


def measure_baseline(x_list: list[torch.Tensor], scalar: float, warmup: int, iters: int) -> float:
    """torch.norm per-tensor baseline (matches golden: FP16/BF16 upcast FP32)."""
    # Pre-upcast so timing reflects the norm computation itself (the golden
    # upcasts before torch.norm; we measure the same compute the golden does).
    x_compute = [to_compute_dtype(t) for t in x_list]
    for _ in range(warmup):
        for t in x_compute:
            _ = torch.norm(t, p=scalar)
    _sync()
    times = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        for t in x_compute:
            _ = torch.norm(t, p=scalar)
        _sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]


def measure_ours(x_list: list[torch.Tensor], scalar: float, warmup: int, iters: int) -> float:
    """foreach_norm (host dispatch + kernel) end-to-end."""
    for _ in range(warmup):
        _ = foreach_norm(x_list, scalar)
    _sync()
    times = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        _ = foreach_norm(x_list, scalar)
        _sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_benchmark(warmup: int, iters: int, label: str, out_path: str, cases: list[dict] = None):
    if cases is None:
        cases = load_cases()

    results = []
    sum_speedup = 0.0
    n_valid = 0

    print(f"\n=== ForeachNorm bench (label={label}, warmup={warmup}, iters={iters}) ===")
    print(f"{'cid':>3} {'shape':>30} {'dtype':>9} {'scalar':>6} {'tl':>3} {'base_us':>10} {'ours_us':>10} {'speedup':>8}")
    print("-" * 90)

    for case in cases:
        cid = case["case_id"]
        # input_shape is a list wrapping the TensorList: [[shape0, shape1, ...]]
        # input_shape[0] is the TensorList (list of per-tensor shapes).
        tensorlist_shapes = case["input_shape"][0]
        dtype_str = case["dtype"][0]
        scalar = case["attrs"]["scalar"]
        vrange = tuple(case["value_range"])
        tl_len = len(tensorlist_shapes)

        # scalar may be .inf / -.inf from YAML
        if isinstance(scalar, str):
            scalar = float(scalar)

        # Build TensorList
        try:
            x_list = [gen_tensor(tuple(s), dtype_str, vrange, scalar) for s in tensorlist_shapes]
        except Exception as e:
            print(f"  case {cid}: input gen failed: {e}")
            results.append({"case_id": cid, "error": str(e)})
            continue

        shape_str = str(tensorlist_shapes[0]) + (f" x{tl_len}" if tl_len > 1 else "")

        try:
            base_t = measure_baseline(x_list, scalar, warmup, iters)
            our_t = measure_ours(x_list, scalar, warmup, iters)
        except Exception as e:
            print(f"  case {cid}: measure failed: {e}")
            results.append({"case_id": cid, "error": str(e)})
            continue

        base_us = base_t * 1e6
        our_us = our_t * 1e6
        # Handle near-zero times (e.g., all-zero input -> tiny work)
        if our_us < 1e-3 and base_us < 1e-3:
            speedup = 1.0
        elif our_us < 1e-3:
            speedup = 99.0
        else:
            speedup = base_us / our_us

        sum_speedup += speedup
        n_valid += 1

        print(f"{cid:>3} {shape_str:>30} {dtype_str:>9} {str(scalar):>6} {tl_len:>3} {base_us:>10.2f} {our_us:>10.2f} {speedup:>8.3f}")

        results.append(
            {
                "case_id": cid,
                "shape": tensorlist_shapes,
                "dtype": dtype_str,
                "scalar": float(scalar) if not math.isinf(scalar) else str(scalar),
                "list_len": tl_len,
                "value_range": list(vrange),
                "baseline_us": base_us,
                "ours_us": our_us,
                "speedup": speedup,
            }
        )

    avg = sum_speedup / n_valid if n_valid else 0.0
    print("-" * 90)
    print(f"  AVG speedup = {avg:.4f}  (target >= 0.6)  |  cases: {n_valid}/{len(cases)}")
    target_met = avg >= 0.6
    print(f"  TARGET {'MET' if target_met else 'NOT MET'} (avg={avg:.4f}, target=0.6)")

    summary = {
        "label": label,
        "warmup": warmup,
        "iters": iters,
        "n_cases": len(cases),
        "n_valid": n_valid,
        "avg_speedup": avg,
        "target": 0.6,
        "target_met": target_met,
        "cases": results,
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  -> saved {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--label", default="ours")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if not args.out:
        args.out = os.path.join(_HERE, f"{args.label}.json")

    torch.manual_seed(0)
    run_benchmark(args.warmup, args.iters, args.label, args.out)


if __name__ == "__main__":
    main()
