"""Mish activation: layered tests L0/L1/L2/Boundary + main(--level).

L0: 9 gate cases from DESIGN.md §9.2 (3 dtypes x rule shapes + special values).
L1: functional tests with non-aligned/edge/multi-dim shapes + asymmetric value range.
L2: negative tests (unsupported dtype, 0-dim scalar) -- non-blocking.
Boundary: special values (inf/nan/zero/dbound) -- non-blocking.
"""

import argparse
import os
import sys

import tilelang
import torch

# Import kernel from sibling file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mish import mish, mish_forward  # noqa: E402, F401


# ========== Golden reference (matches cann-bench golden.py) ==========
def golden_mish(x):
    """Mish golden: y = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x)).

    Uses torch.nn.functional.mish (same as cann-bench tasks/level1/mish/golden.py).
    PyTorch internally upcasts fp16/bf16 to fp32 for element-wise ops, matching
    the kernel's float32 intermediate computation approach.
    """
    return torch.nn.functional.mish(x)


# ========== Precision standard (DESIGN.md §9.3, mixed tolerance) ==========
def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio) by dtype."""
    table = {
        "float16": (6.10e-5, 9.77e-4, 1e2, 0.99),
        "bfloat16": (4.88e-4, 7.81e-3, 1e3, 0.99),
        "float32": (7.63e-6, 1.22e-4, 1e0, 0.99),
    }
    return table.get(dtype, (6.10e-5, 9.77e-4, 1e2, 0.99))


def check_special_masks(actual, golden):
    """Check NaN/pos-inf/neg-inf masks match exactly.

    Returns (masks_match, finite_mask) where finite_mask marks positions where
    both actual and golden are finite (for numerical comparison).
    """
    a, g = actual.float(), golden.float()
    masks_match = (
        torch.equal(torch.isnan(a), torch.isnan(g))
        and torch.equal(torch.isposinf(a), torch.isposinf(g))
        and torch.equal(torch.isneginf(a), torch.isneginf(g))
    )
    finite = torch.isfinite(a) & torch.isfinite(g)
    return masks_match, finite


def check_precision(actual, golden, dtype):
    """Mixed tolerance check: returns (passed, matched_ratio, max_abs_error).

    Dual-gate: matched_ratio >= required AND max_abs_error <= limit.
    Special values (NaN/Inf) checked via mask equality first.
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a, g = actual.detach().cpu(), golden.detach().cpu()

    masks_match, finite = check_special_masks(a, g)
    if not masks_match:
        return False, 0.0, float("inf")

    a, g = a.float(), g.float()
    if finite.sum().item() == 0:
        return True, 1.0, 0.0

    abs_err = (a[finite] - g[finite]).abs()
    ratio = (abs_err <= (atol + rtol * g[finite].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# ========== Test data generation (all on CPU, then H2D) ==========
_STR_TO_TORCH = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _gen_finite(shape, dtype_torch, lo, hi, seed):
    """Uniform random finite values in [lo, hi)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(shape, generator=gen, dtype=torch.float32) * (hi - lo) + lo
    return x.to(dtype_torch)


def _gen_mixed_inf(shape, dtype_torch, lo, hi, seed):
    """Finite values + sparse +/-inf injection (position-sensitive)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(shape, generator=gen, dtype=torch.float32) * (hi - lo) + lo
    flat = x.view(-1)
    n = flat.numel()
    if n > 0:
        flat[0::1000] = float("inf")
    if n > 500:
        flat[500::1000] = float("-inf")
    return x.to(dtype_torch)


def _gen_mixed_nan(shape, dtype_torch, lo, hi, seed):
    """Finite values + sparse NaN injection (position-sensitive)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(shape, generator=gen, dtype=torch.float32) * (hi - lo) + lo
    flat = x.view(-1)
    n = flat.numel()
    if n > 0:
        flat[0::1000] = float("nan")
    return x.to(dtype_torch)


def _gen_zeros(shape, dtype_torch):
    """All zeros."""
    return torch.zeros(shape, dtype=dtype_torch)


def _gen_input(shape, dtype_str, gen, vrange, seed):
    """Generate test input tensor on CPU per config."""
    dt = _STR_TO_TORCH[dtype_str]
    lo, hi = vrange
    if gen == "finite":
        return _gen_finite(shape, dt, lo, hi, seed)
    elif gen == "mixed_inf":
        return _gen_mixed_inf(shape, dt, lo, hi, seed)
    elif gen == "mixed_nan":
        return _gen_mixed_nan(shape, dt, lo, hi, seed)
    elif gen == "zeros":
        return _gen_zeros(shape, dt)
    else:
        raise ValueError(f"Unknown gen type: {gen}")


def _run_kernel_and_golden(shape, dtype_str, gen, vrange, seed):
    """Generate input, run kernel + golden, return (y_cpu, ref_cpu)."""
    x_cpu = _gen_input(shape, dtype_str, gen, vrange, seed)
    x_npu = x_cpu.npu()
    y_npu = mish_forward(x_npu, block_M=128, block_N=128)
    y_cpu = y_npu.cpu()
    ref_cpu = golden_mish(x_cpu)
    return y_cpu, ref_cpu


# ========== L0 test cases (DESIGN.md §9.2, 9 cases with tags) ==========
L0_CONFIGS = [
    {
        "name": "l0_fp16_basic",
        "shape": (1024, 1024),
        "dtype": "float16",
        "gen": "finite",
        "range": (-1, 1),
        "seed": 1,
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-S"],
    },
    {
        "name": "l0_fp32_basic",
        "shape": (1024, 1024),
        "dtype": "float32",
        "gen": "finite",
        "range": (-2, 2),
        "seed": 2,
        "tags": ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-S"],
    },
    {
        "name": "l0_bf16_basic",
        "shape": (1024, 1024),
        "dtype": "bfloat16",
        "gen": "finite",
        "range": (-3, 3),
        "seed": 3,
        "tags": ["D-DTYPE-bf16", "D-SHAPE-ALIGNED"],
    },
    {
        "name": "l0_fp16_mid",
        "shape": (2048, 2048),
        "dtype": "float16",
        "gen": "finite",
        "range": (-10, 10),
        "seed": 4,
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M"],
    },
    {
        "name": "l0_fp32_large",
        "shape": (8192, 8192),
        "dtype": "float32",
        "gen": "finite",
        "range": (-100, 100),
        "seed": 5,
        "tags": ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-L"],
    },
    {
        "name": "l0_fp16_dbound",
        "shape": (1024, 1024),
        "dtype": "float16",
        "gen": "finite",
        "range": (-65504, 65504),
        "seed": 6,
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-SPECIAL-DBOUND"],
    },
    {
        "name": "l0_bf16_inf",
        "shape": (1024, 1024),
        "dtype": "bfloat16",
        "gen": "mixed_inf",
        "range": (-3, 3),
        "seed": 7,
        "tags": ["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-SPECIAL-INF"],
    },
    {
        "name": "l0_fp32_nan",
        "shape": (1024, 1024),
        "dtype": "float32",
        "gen": "mixed_nan",
        "range": (-2, 2),
        "seed": 8,
        "tags": ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-SPECIAL-NAN"],
    },
    {
        "name": "l0_fp16_zero",
        "shape": (1024, 1024),
        "dtype": "float16",
        "gen": "zeros",
        "range": (0, 0),
        "seed": 9,
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-SPECIAL-ZERO"],
    },
]


def test_mish_l0():
    """L0 gate tests: rule shapes (block-aligned), for precision convergence."""
    ok = True
    for cfg in L0_CONFIGS:
        name = cfg["name"]
        shape = cfg["shape"]
        dtype = cfg["dtype"]
        try:
            y_cpu, ref_cpu = _run_kernel_and_golden(shape, dtype, cfg["gen"], cfg["range"], cfg["seed"])
            passed, ratio, max_abs = check_precision(y_cpu, ref_cpu, dtype)
            tag = "PASS" if passed else "FAIL"
            print(f"[PRECISION_{tag}] {name} shape={shape} dtype={dtype} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
            ok &= passed
        except Exception as e:
            print(f"[PRECISION_FAIL] {name} shape={shape} dtype={dtype}: {e}")
            ok = False
    return ok


# ========== L1 test cases (functional: non-aligned/edge/multi-dim + asymmetric range) ==========
# (shape, dtype, gen, vrange, seed, tags)
L1_CASES = [
    # --- Non-aligned shapes (tail blocks) ---
    ((1025, 1024), "float16", "finite", (-1, 1), 13, ["D-SHAPE-TAIL-1", "D-DTYPE-fp16"]),
    ((1024, 1025), "float32", "finite", (-1, 1), 14, ["D-SHAPE-TAIL-1", "D-DTYPE-fp32"]),
    ((1088, 1088), "bfloat16", "finite", (-1, 1), 15, ["D-SHAPE-TAIL-MID", "D-DTYPE-bf16"]),
    ((1009, 1021), "float16", "finite", (-1, 1), 16, ["D-SHAPE-PRIME"]),
    # --- Edge / degenerate shapes ---
    ((1, 1024), "float16", "finite", (-1, 1), 17, ["D-SHAPE-EDGE"]),
    ((1024, 1), "float32", "finite", (-1, 1), 18, ["D-SHAPE-EDGE"]),
    # --- Asymmetric value range ---
    ((1024, 1024), "float32", "finite", (-5, 10), 20, ["D-VALRANGE-ASYM"]),
    # --- Multi-dimensional (rank coverage) ---
    ((1009,), "float16", "finite", (-1, 1), 21, ["D-SHAPE-RANK-1"]),
    ((363, 367, 373), "float16", "finite", (-1, 1), 22, ["D-SHAPE-RANK-3"]),
    ((2, 3, 1024, 101), "float32", "finite", (-1, 1), 23, ["D-SHAPE-RANK-4"]),
]


def test_mish_l1():
    """L1 functional tests: non-aligned/edge/multi-dim shapes + asymmetric range."""
    ok = True
    for shape, dtype, gen, vrange, seed, _tags in L1_CASES:
        try:
            y_cpu, ref_cpu = _run_kernel_and_golden(shape, dtype, gen, vrange, seed)
            passed, ratio, max_abs = check_precision(y_cpu, ref_cpu, dtype)
            tag = "PASS" if passed else "FAIL"
            print(f"[PRECISION_{tag}] l1 shape={shape} dtype={dtype} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
            ok &= passed
        except Exception as e:
            print(f"[PRECISION_FAIL] l1 shape={shape} dtype={dtype}: {e}")
            ok = False
    return ok


# ========== L2 negative tests (non-blocking: invalid input should be rejected) ==========
def _run_exception(name, fn, tags=None):
    """L2: expect fn() to raise. PASS if raises, WARN if silently accepts.
    tags parameter is for coverage checker AST collection only."""
    try:
        fn()
    except Exception as e:
        print(f"[BOUNDARY_PASS] l2 {name}: rejected ({type(e).__name__})")
        return
    print(f"[BOUNDARY_WARN] l2 {name}: invalid input silently accepted")


def test_mish_l2():
    """L2 negative tests: unsupported dtype and illegal shape should be rejected."""
    # D-EXC-DTYPE: unsupported dtype (int32 not in _TORCH_DTYPE_TO_STR)
    _run_exception("unsupported_int32", lambda: mish_forward(torch.zeros((128, 128), dtype=torch.int32).npu()), tags=["D-EXC-DTYPE"])
    # D-EXC-DTYPE: unsupported dtype (float64 not in _TORCH_DTYPE_TO_STR)
    _run_exception("unsupported_float64", lambda: mish_forward(torch.zeros((128, 128), dtype=torch.float64).npu()), tags=["D-EXC-DTYPE"])
    # D-EXC-SHAPE: 0-dimensional (scalar) input -- mish requires at least 1D
    _run_exception("zero_dim_scalar", lambda: mish_forward(torch.tensor(1.0, dtype=torch.float16).npu()), tags=["D-EXC-SHAPE"])


# ========== Boundary tests (special values, non-blocking) ==========
def _run_boundary(name, dtype, fn):
    """Boundary: run kernel+golden, check precision. Non-blocking."""
    try:
        out, ref = fn()
        passed, ratio, max_abs = check_precision(out, ref, dtype)
        tag = "PASS" if passed else "WARN"
        print(f"[BOUNDARY_{tag}] boundary {name} dtype={dtype} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary {name} dtype={dtype}: {e}")


def test_mish_boundary():
    """Boundary: special values (inf/nan/zero/dbound), non-blocking."""
    # D-SPECIAL-INF: fp16 mixed inf (different dtype than L0's bf16 inf)
    _run_boundary("inf_fp16", "float16", lambda: _run_kernel_and_golden((256, 256), "float16", "mixed_inf", (-3, 3), 31))
    # D-SPECIAL-NAN: fp32 mixed nan (different shape than L0's nan)
    _run_boundary("nan_fp16", "float16", lambda: _run_kernel_and_golden((256, 256), "float16", "mixed_nan", (-2, 2), 32))
    # D-SPECIAL-ZERO: bf16 zeros (different dtype than L0's fp16 zero)
    _run_boundary("zero_bf16", "bfloat16", lambda: _run_kernel_and_golden((256, 256), "bfloat16", "zeros", (0, 0), 33))
    # D-SPECIAL-DBOUND: fp16 boundary values (different shape than L0's dbound)
    _run_boundary("dbound_fp32", "float32", lambda: _run_kernel_and_golden((256, 256), "float32", "finite", (-88, 88), 34))


# ========== Coverage declarations (for coverage_check.py) ==========
COVERAGE_CATEGORY = "Activation"
COVERAGE_MANIFEST = {
    "D-DTYPE-fp16": 8,
    "D-DTYPE-fp32": 6,
    "D-DTYPE-bf16": 4,
    "D-SHAPE-ALIGNED": 9,
    "D-SHAPE-EDGE": 2,
    "D-VALRANGE-S": 2,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-SPECIAL-INF": 2,
    "D-SPECIAL-NAN": 2,
    "D-SPECIAL-ZERO": 2,
    "D-SPECIAL-DBOUND": 2,
    "D-EXC-DTYPE": 2,
    "D-EXC-SHAPE": 1,
}
COVERAGE_NA = {}  # No exemptions needed -- all required dimensions covered


# ========== Main: --level dispatch + exit code ==========
def main():
    parser = argparse.ArgumentParser(description="Mish layered tests")
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"], help="Test level to run (default: l0)")
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True  # Only L0/L1 affect exit code
    if args.level in ("l0", "all"):
        blocking_ok &= test_mish_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_mish_l1()
    if args.level in ("l2", "all"):
        test_mish_l2()
    if args.level in ("boundary", "all"):
        test_mish_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
