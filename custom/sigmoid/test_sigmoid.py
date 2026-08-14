"""Sigmoid layered tests: L0/L1/L2/Boundary + main(--level)."""

import argparse
import os
import sys

import tilelang
import torch

# Import kernel from sibling module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sigmoid import sigmoid  # noqa: E402

# ========== Coverage category (for coverage_check.py) ==========
COVERAGE_CATEGORY = "Activation"


# ========== Golden reference ==========
def golden_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid reference: y = 1 / (1 + exp(-x))."""
    return torch.sigmoid(x)


# ========== Precision standard (mixed tolerance) ==========
def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio).

    Float: mixed tolerance; Integer: exact match (0 error).
    See tilelang-op-test-design/references/precision-standard.md.
    """
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Mixed tolerance dual-threshold: return (passed, matched_ratio, max_abs_error).

    Float: dual-threshold (matched_ratio + max_abs_error).
    Integer: exact match (0 error).
    Non-finite golden positions: structural comparison (nan==nan, inf==inf).
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a, g = actual.detach().cpu(), golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # integer exact match
        mism = (a != g).sum().item()
        return mism == 0, 1.0 - mism / max(a.numel(), 1), (0.0 if mism == 0 else float("inf"))
    a, g = a.float(), g.float()
    # Structural comparison for non-finite golden (inf/nan) positions
    special = ~torch.isfinite(g)
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        return False, 0.0, float("inf")
    # Compare finite golden positions: actual inf/nan at finite-golden = fail
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# ========== Input generation ==========
def _make_input(shape, dtype, vrange=None):
    """Generate input tensor with optional uniform value range [low, high]."""
    dt = getattr(torch, dtype)
    if vrange is None:
        return torch.randn(shape, dtype=dt, device="npu")
    low, high = vrange
    # Generate in float32 then cast (reliable for controlled ranges across dtypes)
    x = torch.rand(shape, dtype=torch.float32, device="npu")
    x = (x * (high - low) + low).to(dt)
    return x


# ========== L0/L1 helper: precision case (blocking) ==========
def _run_precision(level, shape, dtype, block, vrange=None):
    """L0/L1 single case: compile kernel, run, compare with golden.
    Returns True if passed. Prints [PRECISION_PASS] or [PRECISION_FAIL]."""
    try:
        M, N = shape
        block_M, block_N = block
        kernel = sigmoid(M, N, block_M, block_N, dtype=dtype)
        x = _make_input(shape, dtype, vrange)
        y = kernel(x)
        ref = golden_sigmoid(x)
        passed, ratio, max_abs = check_precision(y, ref, dtype)
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] {level} shape={shape} dtype={dtype} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] {level} shape={shape} dtype={dtype}: {e}")
        return False


# ========== L2 helper: exception case (non-blocking) ==========
def _run_exception(name, fn):
    """L2: fn() feeds illegal input, expect rejection.
    Raises = [BOUNDARY_PASS] (correctly rejected); no raise = [BOUNDARY_WARN]."""
    try:
        fn()
    except Exception as e:
        print(f"[BOUNDARY_PASS] l2 {name}: rejected ({type(e).__name__})")
        return
    print(f"[BOUNDARY_WARN] l2 {name}: illegal input not rejected")


# ========== Boundary helper: special value precision (non-blocking) ==========
def _run_boundary(name, dtype, fn):
    """Boundary: fn() returns (out, ref); compare precision.
    Pass = [BOUNDARY_PASS]; fail or exception = [BOUNDARY_WARN]. Non-blocking."""
    try:
        out, ref = fn()
        passed, ratio, max_abs = check_precision(out, ref, dtype)
        tag = "PASS" if passed else "WARN"
        print(f"[BOUNDARY_{tag}] boundary {name} dtype={dtype} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary {name} dtype={dtype}: {e}")


# ========== L0: gate tests (from DESIGN.md §9.2) ==========
def test_sigmoid_l0():
    """L0 gate tests: regular shapes (block-aligned) for precision convergence."""
    test_configs = [
        # (dtype, shape, block) -- from DESIGN.md §9.2
        ("float16", (128, 128), (128, 128)),
        ("float16", (256, 256), (64, 64)),
        ("float16", (512, 512), (128, 128)),
        ("float16", (1024, 1024), (128, 128)),
        ("float16", (1024, 8192), (128, 128)),
        ("float32", (256, 256), (64, 64)),
        ("float32", (512, 512), (128, 128)),
    ]
    ok = True
    for dtype, shape, block in test_configs:
        ok &= _run_precision("l0", shape, dtype, block)
    return ok


# ========== L1: functional tests (deterministic shapes, §6 coverage matrix) ==========
# (shape, dtype, block, vrange, tags)
L1_CASES = [
    # Aligned shapes — dtype + valrange coverage
    ((512, 512), "float16", (128, 128), (-1, 1), ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-S"]),
    ((512, 512), "float32", (128, 128), (-1, 1), ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-S"]),
    ((512, 512), "float16", (128, 128), (-10, 10), ["D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    ((512, 512), "float32", (128, 128), (-10, 10), ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    ((512, 512), "float16", (128, 128), (-50, 50), ["D-SHAPE-ALIGNED", "D-VALRANGE-L"]),
    ((512, 512), "float16", (128, 128), (-5, 10), ["D-SHAPE-ALIGNED", "D-VALRANGE-ASYM"]),
    # Tail block shapes (non-divisible)
    ((513, 512), "float16", (128, 128), (-1, 1), ["D-DTYPE-fp16", "D-SHAPE-TAIL-1"]),
    ((576, 576), "float16", (128, 128), (-1, 1), ["D-DTYPE-fp16", "D-SHAPE-TAIL-MID"]),
    # Prime shapes (fully non-aligned)
    ((509, 503), "float16", (128, 128), (-1, 1), ["D-DTYPE-fp16", "D-SHAPE-PRIME"]),
    # Edge / degenerate shapes
    ((1, 512), "float16", (2, 128), (-1, 1), ["D-DTYPE-fp16", "D-SHAPE-EDGE"]),
]


def test_sigmoid_l1():
    """L1 functional tests: dtype/shape/valrange coverage with deterministic
    non-aligned shapes. Returns True if all pass (blocking)."""
    ok = True
    for shape, dtype, block, vrange, _tags in L1_CASES:
        ok &= _run_precision("l1", shape, dtype, block, vrange)
    return ok


# ========== L2: negative tests (illegal input should be rejected) ==========
def test_sigmoid_l2():
    """L2 negative tests: unsupported dtype / illegal shape should be rejected.
    Correct rejection = [BOUNDARY_PASS]; silent accept = [BOUNDARY_WARN]."""

    # D-EXC-DTYPE: unsupported dtype (int32 not in proto)
    def _unsupported_dtype():
        sigmoid(128, 128, 128, 128, dtype="int32")

    _run_exception("unsupported_dtype_int32", _unsupported_dtype)

    # D-EXC-SHAPE: 1D tensor passed to 2D kernel
    def _illegal_shape_1d():
        kernel = sigmoid(128, 128, 128, 128, dtype="float16")
        x = torch.randn(128, dtype=torch.float16, device="npu")  # 1D, wrong
        kernel(x)

    _run_exception("illegal_shape_1d", _illegal_shape_1d)


# ========== Boundary: special value tests (legal extremes, non-blocking) ==========
def test_sigmoid_boundary():
    """Boundary tests: INF/NAN/zero/dtype-boundary special values.
    Compare precision with golden; non-blocking (WARN on failure)."""
    M, N = 128, 128
    block = (128, 128)

    # D-SPECIAL-ZERO: all zeros
    def _zero():
        kernel = sigmoid(M, N, *block, dtype="float16")
        x = torch.zeros(M, N, dtype=torch.float16, device="npu")
        return kernel(x), golden_sigmoid(x)

    _run_boundary("zero", "float16", _zero)

    # D-SPECIAL-INF: mix of +inf and -inf
    def _inf():
        kernel = sigmoid(M, N, *block, dtype="float16")
        x = torch.zeros(M, N, dtype=torch.float16, device="npu")
        x[: M // 2] = float("inf")
        x[M // 2 :] = float("-inf")
        return kernel(x), golden_sigmoid(x)

    _run_boundary("inf", "float16", _inf)

    # D-SPECIAL-NAN: all nan
    def _nan():
        kernel = sigmoid(M, N, *block, dtype="float16")
        x = torch.full((M, N), float("nan"), dtype=torch.float16, device="npu")
        return kernel(x), golden_sigmoid(x)

    _run_boundary("nan", "float16", _nan)

    # D-SPECIAL-DBOUND: dtype boundary values (fp16 max, exp boundary)
    def _dbound():
        kernel = sigmoid(M, N, *block, dtype="float16")
        x = torch.zeros(M, N, dtype=torch.float16, device="npu")
        x[:32] = 65504.0  # fp16 max
        x[32:64] = -65504.0  # fp16 min
        x[64:96] = 11.0  # exp boundary (exp(11) ~ 60000)
        x[96:] = -11.0  # exp boundary
        return kernel(x), golden_sigmoid(x)

    _run_boundary("dbound", "float16", _dbound)


# ========== Coverage manifest (for coverage_check.py) ==========
COVERAGE_MANIFEST = {
    "D-DTYPE-fp16": 8,  # L0: 5 + L1: 8 (incl. boundary overlap)
    "D-DTYPE-fp32": 4,  # L0: 2 + L1: 2
    "D-SHAPE-ALIGNED": 6,  # L1: 6 (512x512 with 128x128)
    "D-SHAPE-EDGE": 1,  # L1: (1, 512)
    "D-SHAPE-TAIL-1": 1,  # L1: (513, 512)
    "D-SHAPE-TAIL-MID": 1,  # L1: (576, 576)
    "D-SHAPE-PRIME": 1,  # L1: (509, 503)
    "D-VALRANGE-S": 2,  # L1: [-1, 1] x2
    "D-VALRANGE-M": 2,  # L1: [-10, 10] x2
    "D-VALRANGE-L": 1,  # L1: [-50, 50]
    "D-VALRANGE-ASYM": 1,  # L1: [-5, 10]
    "D-SPECIAL-ZERO": 1,  # Boundary
    "D-SPECIAL-INF": 1,  # Boundary
    "D-SPECIAL-NAN": 1,  # Boundary
    "D-SPECIAL-DBOUND": 1,  # Boundary
    "D-EXC-DTYPE": 1,  # L2
    "D-EXC-SHAPE": 1,  # L2
}

COVERAGE_NA = {}  # All required dimensions covered; no exemptions needed


# ========== Main: --level dispatch + exit code ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()  # disable compile cache to avoid stale artifacts
    torch.manual_seed(0)

    blocking_ok = True  # only L0/L1 count toward blocking
    if args.level in ("l0", "all"):
        blocking_ok &= test_sigmoid_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_sigmoid_l1()
    if args.level in ("l2", "all"):
        test_sigmoid_l2()
    if args.level in ("boundary", "all"):
        test_sigmoid_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
