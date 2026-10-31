"""Transpose operator layered tests: L0/L1/L2/Boundary + main(--level)."""

import argparse
import math
import os
import sys

import tilelang
import torch

# Import kernel from sibling file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transpose import transpose  # noqa: E402

# ========== Coverage declarations (for coverage_check.py) ==========
COVERAGE_CATEGORY = "Vector"
COVERAGE_MANIFEST = {}
COVERAGE_NA = {}


# ========== Golden reference ==========
def golden_transpose(x: torch.Tensor, perm: list) -> torch.Tensor:
    """PyTorch golden: torch.permute."""
    return torch.permute(x, perm)


# ========== Precision standard (mixed tolerance, by dtype) ==========
def get_precision(dtype_str):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio).

    Float: mixed tolerance; Integer: exact match (0 error).
    """
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    if dtype_str in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype_str, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype_str):
    """Mixed tolerance dual-gate check: return (passed, matched_ratio, max_abs_error).

    For float dtypes: checks inf/nan position consistency, then compares finite values
    with mixed tolerance (atol + rtol*|golden|, matched_ratio, max_abs_error_limit).
    For integer dtypes: exact match (0 error).
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype_str)
    a, g = actual.detach().cpu(), golden.detach().cpu()

    if atol == 0.0 and rtol == 0.0:  # Integer exact match
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))

    a, g = a.float(), g.float()

    # Check inf position consistency
    g_inf = torch.isinf(g)
    a_inf = torch.isinf(a)
    if not torch.equal(g_inf, a_inf):
        return False, 0.0, float("inf")

    # Check nan position consistency
    g_nan = torch.isnan(g)
    a_nan = torch.isnan(a)
    if not torch.equal(g_nan, a_nan):
        return False, 0.0, float("inf")

    # Compare finite values only
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# ========== Dtype helpers ==========
DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _sf(v):
    """Parse special string values to float."""
    if isinstance(v, str):
        return {"inf": float("inf"), "-inf": float("-inf"), "nan": float("nan")}[v]
    return v


def make_input(shape, dtype_str, value_range):
    """Create test input on CPU, then move to NPU.

    Supports special values: inf, -inf, nan, all-zeros.
    """
    torch_dtype = DTYPE_MAP[dtype_str]
    lo, hi = _sf(value_range[0]), _sf(value_range[1])

    if dtype_str in ("float16", "float32", "bfloat16"):
        if isinstance(lo, float) and isinstance(hi, float) and math.isnan(lo) and math.isnan(hi):
            # Match cann-bench semantics: [nan, nan] means a deterministic
            # mixture of finite values and NaNs.  An all-NaN tensor cannot
            # expose values moved to the wrong positions by a layout kernel.
            generator = torch.Generator().manual_seed(0)
            base = torch.rand(shape, dtype=torch.float32, generator=generator) * 2.0 - 1.0
            nan_mask = torch.rand(shape, dtype=torch.float32, generator=generator) < 0.5
            flat_mask = nan_mask.view(-1)
            if flat_mask.numel() > 0:
                flat_mask[0] = True
            if flat_mask.numel() > 1:
                flat_mask[1] = False
            base[nan_mask] = float("nan")
            return base.to(torch_dtype).npu()
        if isinstance(lo, float) and (math.isinf(lo) or math.isinf(hi)):
            # Contains inf: generate random + inject inf/-inf
            base = (torch.rand(shape, dtype=torch.float32) * 2.0 - 1.0).to(torch_dtype)
            flat = base.view(-1)
            if flat.numel() > 0:
                flat[0] = float("inf")
            if flat.numel() > 1:
                flat[1] = float("-inf")
            return base.npu()
        if lo == hi:
            # All same value (e.g., all zeros)
            return torch.full(shape, float(lo), dtype=torch_dtype).npu()
        return (torch.rand(shape, dtype=torch.float32) * (hi - lo) + lo).to(torch_dtype).npu()
    elif dtype_str == "int8":
        x_cpu = torch.randint(int(lo), int(hi) + 1, shape, dtype=torch.int32).to(torch.int8)
        return x_cpu.npu()
    elif dtype_str == "int16":
        x_cpu = torch.randint(int(lo), int(hi) + 1, shape, dtype=torch.int32).to(torch.int16)
        return x_cpu.npu()
    else:
        x_cpu = torch.randint(int(lo), int(hi) + 1, shape, dtype=torch_dtype)
        return x_cpu.npu()


def run_case(case_id, shape, dtype_str, perm, value_range, tags=None):
    """Run a single transpose test case and print result."""
    x = make_input(shape, dtype_str, value_range)
    y = transpose(x, perm)
    golden = golden_transpose(x.cpu(), perm)
    passed, ratio, max_abs = check_precision(y, golden, dtype_str)
    tag = "PASS" if passed else "FAIL"
    print(
        f"[PRECISION_{tag}] case_{case_id:02d} shape={shape} dtype={dtype_str} perm={perm} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
    )
    return passed


# ========== L0 tests: threshold (from DESIGN.md §9.2) ==========
def test_transpose_l0():
    """L0 threshold tests: regular shapes (block-divisible), for precision convergence."""
    test_configs = [
        ("l0_2d_fp16", (1024, 1024), "float16", [1, 0], (-1.0, 1.0)),
        ("l0_2d_fp32", (2048, 2048), "float32", [1, 0], (-2.0, 2.0)),
        ("l0_2d_bf16", (1024, 1024), "bfloat16", [1, 0], (-1.0, 1.0)),
        ("l0_2d_int8", (1024, 1024), "int8", [1, 0], (-128, 127)),
        ("l0_2d_int16", (1024, 1024), "int16", [1, 0], (-1000, 1000)),
        ("l0_2d_int32", (2048, 2048), "int32", [1, 0], (-10000, 10000)),
        ("l0_2d_int64", (1024, 1024), "int64", [1, 0], (-100000, 100000)),
        ("l0_4d_fp16", (64, 32, 512, 128), "float16", [0, 2, 1, 3], (-1.0, 1.0)),
        ("l0_3d_fp16", (128, 128, 128), "float16", [2, 0, 1], (-1.0, 1.0)),
        ("l0_5d_fp16", (8, 8, 8, 8, 8), "float16", [4, 3, 2, 1, 0], (-1.0, 1.0)),
    ]

    ok = True
    for name, shape, dtype_str, perm, value_range in test_configs:
        try:
            x = make_input(shape, dtype_str, value_range)
            y = transpose(x, perm)
            golden = golden_transpose(x.cpu(), perm)
            passed, ratio, max_abs = check_precision(y, golden, dtype_str)
            tag = "PASS" if passed else "FAIL"
            print(f"[PRECISION_{tag}] {name} shape={shape} dtype={dtype_str} perm={perm} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
            ok &= passed
        except Exception as e:
            print(f"[PRECISION_FAIL] {name} shape={shape} dtype={dtype_str}: {e}")
            ok = False
    return ok


# ========== L1 tests: functional (cases.csv 20 cases) ==========
L1_CASES = [
    # (case_id, shape, dtype_str, perm, value_range, tags)
    (1, (64, 32, 512, 128), "float16", [0, 2, 1, 3], (-1.0, 1.0), ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-perm"]),
    (2, (2048, 2048), "float32", [1, 0], (-2.0, 2.0), ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-perm"]),
    (3, (4096, 4096), "bfloat16", [1, 0], (-3.0, 3.0), ["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-VALRANGE-L", "D-PARAM-perm"]),
    (4, (8192, 8192), "int32", [1, 0], (-10000, 10000), ["D-DTYPE-int32", "D-SHAPE-ALIGNED", "D-VALRANGE-L", "D-PARAM-perm"]),
    (5, (4096, 8192), "int64", [1, 0], (-100000, 100000), ["D-DTYPE-int64", "D-SHAPE-ALIGNED", "D-VALRANGE-L", "D-PARAM-perm"]),
    (6, (2, 9, 256, 256), "int16", [0, 2, 3, 1], (-1000, 1000), ["D-DTYPE-int16", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-perm"]),
    (7, (1023, 1023), "float16", [1, 0], (-0.1, 0.1), ["D-DTYPE-fp16", "D-SHAPE-EDGE", "D-SHAPE-PRIME", "D-VALRANGE-S", "D-PARAM-perm"]),
    (8, (1009, 1021), "float32", [1, 0], (-1.0, 2.0), ["D-DTYPE-fp32", "D-SHAPE-EDGE", "D-SHAPE-PRIME", "D-VALRANGE-ASYM", "D-PARAM-perm"]),
    (
        9,
        (1537, 769),
        "bfloat16",
        [1, 0],
        (-5.0, 10.0),
        ["D-DTYPE-bf16", "D-SHAPE-EDGE", "D-SHAPE-PRIME", "D-VALRANGE-ASYM", "D-PARAM-perm"],
    ),
    (
        10,
        (363, 367, 373),
        "int32",
        [2, 0, 1],
        (-50, 100),
        ["D-DTYPE-int32", "D-SHAPE-EDGE", "D-SHAPE-PRIME", "D-VALRANGE-ASYM", "D-PARAM-perm"],
    ),
    (11, (2049, 513), "float16", [1, 0], (-65504.0, 65504.0), ["D-DTYPE-fp16", "D-SHAPE-EDGE", "D-SPECIAL-DBOUND", "D-PARAM-perm"]),
    (12, (3, 7, 13, 4001), "float32", [0, 3, 1, 2], (-88.0, 88.0), ["D-DTYPE-fp32", "D-SHAPE-EDGE", "D-SHAPE-PRIME", "D-PARAM-perm"]),
    (13, (2, 7, 256, 256), "bfloat16", [0, 1, 3, 2], (-0.01, 0.01), ["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-VALRANGE-S", "D-PARAM-perm"]),
    (
        14,
        (2, 511, 7, 127),
        "float32",
        [0, 2, 1, 3],
        (float("-inf"), float("inf")),
        ["D-DTYPE-fp32", "D-SHAPE-EDGE", "D-SHAPE-PRIME", "D-SPECIAL-INF", "D-PARAM-perm"],
    ),
    (
        15,
        (11, 13, 17, 67, 67),
        "float16",
        [4, 3, 2, 1, 0],
        (float("nan"), float("nan")),
        ["D-DTYPE-fp16", "D-SHAPE-EDGE", "D-SHAPE-PRIME", "D-SPECIAL-NAN", "D-EXC-SHAPE", "D-PARAM-perm"],
    ),
    (
        16,
        (3, 7, 11, 13, 1013),
        "int64",
        [4, 3, 2, 1, 0],
        (0, 0),
        ["D-DTYPE-int64", "D-SHAPE-EDGE", "D-SHAPE-PRIME", "D-SPECIAL-ZERO", "D-EXC-SHAPE", "D-PARAM-perm"],
    ),
    (17, (512, 2049), "float32", [1, 0], (-0.5, 0.5), ["D-DTYPE-fp32", "D-SHAPE-EDGE", "D-VALRANGE-S", "D-PARAM-perm"]),
    (18, (255, 8193), "bfloat16", [1, 0], (-1.0, 3.0), ["D-DTYPE-bf16", "D-SHAPE-EDGE", "D-VALRANGE-ASYM", "D-PARAM-perm"]),
    (19, (4097, 511), "int8", [1, 0], (-128, 127), ["D-DTYPE-int8", "D-SHAPE-EDGE", "D-EXC-DTYPE", "D-PARAM-perm"]),
    (
        20,
        (2, 511, 2049),
        "float16",
        [2, 1, 0],
        (-3.0, 6.0),
        ["D-DTYPE-fp16", "D-SHAPE-EDGE", "D-SHAPE-PRIME", "D-VALRANGE-ASYM", "D-PARAM-perm"],
    ),
]


def test_transpose_l1():
    """L1 functional tests: cases.csv 20 cases."""
    ok = True
    for case_id, shape, dtype_str, perm, value_range, tags in L1_CASES:
        try:
            passed = run_case(case_id, shape, dtype_str, perm, value_range, tags)
            ok &= passed
        except Exception as e:
            print(f"[PRECISION_FAIL] case_{case_id:02d} shape={shape} dtype={dtype_str}: {e}")
            ok = False
    return ok


# ========== L2 tests: negative (invalid perm should be rejected) ==========
def test_transpose_l2():
    """L2 negative tests: invalid perm should raise exception."""

    def _expect_reject(desc, fn):
        try:
            fn()
            print(f"[BOUNDARY_WARN] {desc}: silently accepted (should have raised)")
        except (ValueError, AssertionError, RuntimeError):
            print(f"[BOUNDARY_PASS] {desc}: correctly rejected")

    # perm length mismatch
    _expect_reject(
        "perm_len_mismatch",
        lambda: transpose(torch.randn(3, 4, dtype=torch.float32).npu(), [0, 1, 2]),
    )
    # perm out of range
    _expect_reject(
        "perm_out_of_range",
        lambda: transpose(torch.randn(3, 4, dtype=torch.float32).npu(), [0, 2]),
    )
    # perm duplicate
    _expect_reject(
        "perm_duplicate",
        lambda: transpose(torch.randn(3, 4, dtype=torch.float32).npu(), [0, 0]),
    )
    # perm negative value
    _expect_reject(
        "perm_negative",
        lambda: transpose(torch.randn(3, 4, dtype=torch.float32).npu(), [0, -1]),
    )


# ========== Boundary tests: special values ==========
def test_transpose_boundary():
    """Boundary special value tests: inf/nan/zero/int8-range/fp16-boundary."""
    boundary_cases = [
        (14, (2, 511, 7, 127), "float32", [0, 2, 1, 3], (float("-inf"), float("inf")), "inf special"),
        (15, (11, 13, 17, 67, 67), "float16", [4, 3, 2, 1, 0], (float("nan"), float("nan")), "nan special"),
        (16, (3, 7, 11, 13, 1013), "int64", [4, 3, 2, 1, 0], (0, 0), "all zeros"),
        (19, (4097, 511), "int8", [1, 0], (-128, 127), "int8 full range"),
        (11, (2049, 513), "float16", [1, 0], (-65504.0, 65504.0), "fp16 boundary"),
    ]

    for case_id, shape, dtype_str, perm, value_range, note in boundary_cases:
        try:
            x = make_input(shape, dtype_str, value_range)
            y = transpose(x, perm)
            golden = golden_transpose(x.cpu(), perm)
            passed, ratio, max_abs = check_precision(y, golden, dtype_str)
            tag = "PASS" if passed else "WARN"
            print(
                f"[BOUNDARY_{tag}] case_{case_id:02d} ({note}) shape={shape} "
                f"dtype={dtype_str} perm={perm} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
            )
        except Exception as e:
            print(f"[BOUNDARY_WARN] case_{case_id:02d} ({note}): {e}")


# ========== Main: --level dispatch + exit code ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()  # Disable compile cache to avoid stale artifacts
    torch.manual_seed(0)

    blocking_ok = True  # Only L0/L1 count toward blocking
    if args.level in ("l0", "all"):
        blocking_ok &= test_transpose_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_transpose_l1()
    if args.level in ("l2", "all"):
        test_transpose_l2()  # L2: correct rejection=PASS, silent accept=WARN, non-blocking
    if args.level in ("boundary", "all"):
        test_transpose_boundary()  # Boundary: precision fail=WARN, non-blocking

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
