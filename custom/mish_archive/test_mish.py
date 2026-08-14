"""Mish layered tests: L0/L1/L2/Boundary + main(--level).

L0 gate tests follow DESIGN.md §9.2 (8 cases: float16/float32/bfloat16 + boundary/inf/nan/zero).
Precision thresholds use cann-bench-derived values (DESIGN.md §9.3), stricter than
precision-standard.md defaults for rtol, with larger max_abs_error_limit to accommodate
mish(x) ~= x for large x.
"""

import argparse
import os
import sys

import tilelang
import torch

# Import kernel from sibling module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mish import mish  # noqa: E402

# ========== Coverage category (for coverage_check.py) ==========
COVERAGE_CATEGORY = "Activation"


# ========== Golden reference ==========
def golden_mish(x: torch.Tensor) -> torch.Tensor:
    """Mish reference: y = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x)).

    Uses torch.nn.functional.mish (same as cann-bench golden.py).
    """
    return torch.nn.functional.mish(x)


# ========== Precision standard (cann-bench-derived, DESIGN.md §9.3) ==========
def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio).

    Float: mixed tolerance; Integer: exact match (0 error).
    Thresholds derived from cann-bench (rtol = Threshold, atol = Threshold/16).
    See custom/mish/DESIGN.md §9.3.
    """
    # cann-bench-derived: rtol=Threshold, atol=Threshold/16
    fp_table = {
        # dtype       : (atol,    rtol,    max_abs_error_limit, required_matched_ratio)
        "float16": (2**-14, 2**-10, 1e2, 0.99),  # atol 6.10e-5, rtol 9.77e-4
        "bfloat16": (2**-11, 2**-7, 1e3, 0.99),  # atol 4.88e-4, rtol 7.81e-3
        "float32": (2**-17, 2**-13, 1e0, 0.99),  # atol 7.63e-6, rtol 1.22e-4
        "hifloat32": (2**-17, 2**-13, 1e0, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-10, 1e2, 0.99))


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
    """Generate input tensor with optional uniform value range [low, high].

    vrange can be:
      - None: randn
      - (low, high): uniform in [low, high]
      - "inf": mix of +inf, -inf, and finite values
      - "nan": all nan
    """
    dt = getattr(torch, dtype)
    if vrange is None:
        return torch.randn(shape, dtype=dt, device="npu")
    if vrange == "inf":
        x = torch.zeros(shape, dtype=dt, device="npu")
        # Mix: half +inf, half -inf (finite values already 0)
        flat = x.view(-1)
        n = flat.numel()
        flat[: n // 2] = float("inf")
        flat[n // 2 :] = float("-inf")
        return x
    if vrange == "nan":
        return torch.full(shape, float("nan"), dtype=dt, device="npu")
    low, high = vrange
    # Generate in float32 then cast (reliable for controlled ranges across dtypes)
    x = torch.rand(shape, dtype=torch.float32, device="npu")
    x = (x * (high - low) + low).to(dt)
    return x


# ========== L0/L1 helper: precision case (blocking) ==========
def _run_precision(level, shape, dtype, block, vrange=None, name=None):
    """L0/L1 single case: compile kernel, run, compare with golden.

    Returns True if passed. Prints [PRECISION_PASS] or [PRECISION_FAIL].
    """
    label = name or f"shape={shape} dtype={dtype}"
    try:
        M, N = shape
        block_M, block_N = block
        kernel = mish(M, N, block_M, block_N, dtype=dtype)
        x = _make_input(shape, dtype, vrange)
        y = kernel(x)
        ref = golden_mish(x)
        passed, ratio, max_abs = check_precision(y, ref, dtype)
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] {level} {label} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] {level} {label}: {e}")
        return False


# ========== L2 helper: exception case (non-blocking) ==========
def _run_exception(name, fn):
    """L2: fn() feeds illegal input, expect rejection.

    Raises = [BOUNDARY_PASS] (correctly rejected); no raise = [BOUNDARY_WARN].
    """
    try:
        fn()
    except Exception as e:
        print(f"[BOUNDARY_PASS] l2 {name}: rejected ({type(e).__name__})")
        return
    print(f"[BOUNDARY_WARN] l2 {name}: illegal input not rejected")


# ========== Boundary helper: special value precision (non-blocking) ==========
def _run_boundary(name, dtype, fn):
    """Boundary: fn() returns (out, ref); compare precision.

    Pass = [BOUNDARY_PASS]; fail or exception = [BOUNDARY_WARN]. Non-blocking.
    """
    try:
        out, ref = fn()
        passed, ratio, max_abs = check_precision(out, ref, dtype)
        tag = "PASS" if passed else "WARN"
        print(f"[BOUNDARY_{tag}] boundary {name} dtype={dtype} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary {name} dtype={dtype}: {e}")


# ========== L0: gate tests (from DESIGN.md §9.2, 8 cases) ==========
# (dtype, shape, block, vrange, name, tags)
L0_CASES = [
    ("float16", (1024, 1024), (128, 128), (-1, 1), "l0_fp16_basic", ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-S"]),
    ("float32", (1024, 1024), (128, 128), (-2, 2), "l0_fp32_basic", ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-S"]),
    ("bfloat16", (1024, 1024), (128, 128), (-3, 3), "l0_bf16_basic", ["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-VALRANGE-S"]),
    ("float16", (2048, 2048), (128, 128), (-10, 10), "l0_fp16_mid", ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    ("float16", (1024, 1024), (128, 128), (-65504, 65504), "l0_fp16_maxval", ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-SPECIAL-DBOUND"]),
    ("bfloat16", (1024, 1024), (128, 128), "inf", "l0_bf16_inf", ["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-SPECIAL-INF"]),
    ("float32", (1024, 1024), (128, 128), "nan", "l0_fp32_nan", ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-SPECIAL-NAN"]),
    ("float16", (1024, 1024), (128, 128), (0, 0), "l0_fp16_zero", ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-SPECIAL-ZERO"]),
]


def test_mish_l0():
    """L0 gate tests: regular shapes (block-aligned) for precision convergence.

    8 cases from DESIGN.md §9.2: float16/float32/bfloat16 + boundary/inf/nan/zero.
    """
    ok = True
    for dtype, shape, block, vrange, name, _tags in L0_CASES:
        ok &= _run_precision("l0", shape, dtype, block, vrange, name=name)
    return ok


# ========== L1/L2/Boundary: expanded by tilelang-op-test-design (scenario B) ==========
# L1 cases: (shape, dtype, block, vrange, name, tags)
# Covers forced dims: D-SHAPE-EDGE, D-VALRANGE-{M,L,ASYM} + extra dtype/shape coverage.
# D-SHAPE-TAIL-*/PRIME included for thoroughness (optional for element-wise Activation).
L1_CASES = [
    # D-VALRANGE-M (symmetric medium range, not yet in L0)
    ((512, 512), "float16", (128, 128), (-10, 10), "l1_fp16_mid", ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    ((512, 512), "float32", (128, 128), (-10, 10), "l1_fp32_mid", ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    ((512, 512), "bfloat16", (128, 128), (-10, 10), "l1_bf16_mid", ["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    # D-VALRANGE-L (symmetric large range, near dtype upper bound)
    (
        (512, 512),
        "float16",
        (128, 128),
        (-65504, 65504),
        "l1_fp16_large",
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-L", "D-SPECIAL-DBOUND"],
    ),
    ((256, 256), "float32", (128, 128), (-1e4, 1e4), "l1_fp32_large", ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-L"]),
    # D-VALRANGE-ASYM (asymmetric range)
    ((512, 512), "float16", (128, 128), (-5, 10), "l1_fp16_asym", ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-ASYM"]),
    ((512, 512), "bfloat16", (128, 128), (-3, 20), "l1_bf16_asym", ["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-VALRANGE-ASYM"]),
    # D-SHAPE-EDGE (degenerate: 1×N and N×1)
    ((1, 512), "float16", (128, 128), (-1, 1), "l1_fp16_edge_1xN", ["D-DTYPE-fp16", "D-SHAPE-EDGE", "D-VALRANGE-S"]),
    ((512, 1), "float32", (128, 128), (-2, 2), "l1_fp32_edge_Nx1", ["D-DTYPE-fp32", "D-SHAPE-EDGE", "D-VALRANGE-S"]),
    # D-SHAPE-TAIL-1 (remainder=1, most likely to expose boundary bugs)
    ((513, 512), "float16", (128, 128), (-1, 1), "l1_fp16_tail1", ["D-DTYPE-fp16", "D-SHAPE-TAIL-1", "D-VALRANGE-S"]),
    ((512, 513), "bfloat16", (128, 128), (-3, 3), "l1_bf16_tail1", ["D-DTYPE-bf16", "D-SHAPE-TAIL-1", "D-VALRANGE-S"]),
    # D-SHAPE-TAIL-MID (remainder=block_M//2, mid-size tail)
    ((576, 576), "float16", (128, 128), (-1, 1), "l1_fp16_tailmid", ["D-DTYPE-fp16", "D-SHAPE-TAIL-MID", "D-VALRANGE-S"]),
    ((576, 512), "float32", (128, 128), (-2, 2), "l1_fp32_tailmid", ["D-DTYPE-fp32", "D-SHAPE-TAIL-MID", "D-VALRANGE-S"]),
    # D-SHAPE-PRIME (fully non-aligned, prime dimensions)
    ((509, 503), "bfloat16", (128, 128), (-3, 3), "l1_bf16_prime", ["D-DTYPE-bf16", "D-SHAPE-PRIME", "D-VALRANGE-S"]),
    ((509, 503), "float16", (128, 128), (-10, 10), "l1_fp16_prime", ["D-DTYPE-fp16", "D-SHAPE-PRIME", "D-VALRANGE-M"]),
]


def test_mish_l1():
    """L1 functional tests: regular + irregular shapes (tail/prime/edge) + value ranges.

    Forced coverage: D-SHAPE-EDGE, D-VALRANGE-{M,L,ASYM} (not in L0).
    Optional coverage: D-SHAPE-TAIL-{1,MID}, D-SHAPE-PRIME (element-wise, but included
    for thoroughness — T.ceildiv + T.copy tail handling).
    """
    ok = True
    for shape, dtype, block, vrange, name, _tags in L1_CASES:
        ok &= _run_precision("l1", shape, dtype, block, vrange, name=name)
    return ok


# ========== L2: negative tests (illegal input should be rejected, non-blocking) ==========
def test_mish_l2():
    """L2 negative tests: unsupported dtype / illegal shape should be rejected.

    Correct rejection (raises) = [BOUNDARY_PASS]; silent acceptance = [BOUNDARY_WARN].
    Non-blocking, does not affect exit code.
    """

    # D-EXC-DTYPE: int8 is not a supported mish dtype (proto: float16/float32/bfloat16)
    def _bad_dtype():
        M, N = 128, 128
        kernel = mish(M, N, 128, 128, dtype="int8")
        x = torch.randn((M, N), dtype=torch.float32, device="npu").to(torch.int8)
        kernel(x)

    _run_exception("unsupported_dtype_int8", _bad_dtype)

    # D-EXC-SHAPE: 3D tensor directly passed to 2D kernel (without host-side flatten)
    def _bad_shape_3d():
        kernel = mish(128, 128, 128, 128, dtype="float16")
        x = torch.randn((16, 8, 128), dtype=torch.float16, device="npu")
        kernel(x)

    _run_exception("illegal_shape_3d", _bad_shape_3d)


# ========== Boundary: special value precision tests (non-blocking) ==========
def test_mish_boundary():
    """Boundary tests: legal special values (INF/NAN/zero/dtype-bound), precision compared.

    Precision pass = [BOUNDARY_PASS]; precision fail or exception = [BOUNDARY_WARN].
    Non-blocking, does not affect exit code.
    """

    # D-SPECIAL-INF: ±inf input (mish(+inf)=+inf, mish(-inf)=nan per IEEE -inf*0=nan)
    def _inf_case():
        kernel = mish(128, 128, 128, 128, dtype="float16")
        x = _make_input((128, 128), "float16", "inf")
        return kernel(x), golden_mish(x)

    _run_boundary("inf", "float16", _inf_case)

    # D-SPECIAL-NAN: all-nan input (nan propagates through all 12 steps)
    def _nan_case():
        kernel = mish(128, 128, 128, 128, dtype="float32")
        x = _make_input((128, 128), "float32", "nan")
        return kernel(x), golden_mish(x)

    _run_boundary("nan", "float32", _nan_case)

    # D-SPECIAL-ZERO: all-zero input (mish(0)=0 precisely)
    def _zero_case():
        kernel = mish(128, 128, 128, 128, dtype="float16")
        x = _make_input((128, 128), "float16", (0, 0))
        return kernel(x), golden_mish(x)

    _run_boundary("zero", "float16", _zero_case)

    # D-SPECIAL-DBOUND: float16 boundary values (±65504, finite max)
    def _dbound_case():
        kernel = mish(128, 128, 128, 128, dtype="float16")
        x = _make_input((128, 128), "float16", (-65504, 65504))
        return kernel(x), golden_mish(x)

    _run_boundary("dbound", "float16", _dbound_case)


# ========== Coverage manifest (for coverage_check.py) ==========
# L0 covers: D-DTYPE-{fp16,fp32,bf16}, D-SHAPE-ALIGNED, D-VALRANGE-S, D-SPECIAL-{DBOUND,INF,NAN,ZERO}
# L1 adds: D-SHAPE-{EDGE,TAIL-1,TAIL-MID,PRIME}, D-VALRANGE-{M,L,ASYM}
# L2 adds: D-EXC-{DTYPE,SHAPE}
# Boundary adds: D-SPECIAL-{INF,NAN,ZERO,DBOUND} (precision-checked, non-blocking)
COVERAGE_MANIFEST = {
    # D-DTYPE-*: counted from L0 + L1 tags (checker auto-walks all list literals)
    "D-DTYPE-fp16": 10,  # L0: 4 + L1: 6
    "D-DTYPE-fp32": 6,  # L0: 2 + L1: 4
    "D-DTYPE-bf16": 6,  # L0: 2 + L1: 4
    # D-SHAPE-*
    "D-SHAPE-ALIGNED": 17,  # L0: 8 + L1: 9 (all non-edge/tail/prime L1 cases)
    "D-SHAPE-EDGE": 2,  # L1: 1xN + Nx1
    "D-SHAPE-TAIL-1": 2,  # L1: (513,512) + (512,513)
    "D-SHAPE-TAIL-MID": 2,  # L1: (576,576) + (576,512)
    "D-SHAPE-PRIME": 2,  # L1: (509,503) × 2 dtypes
    # D-VALRANGE-*
    "D-VALRANGE-S": 8,  # L0: 3 + L1: 5
    "D-VALRANGE-M": 5,  # L1: 3 (mid) + 2 (prime with mid range)
    "D-VALRANGE-L": 2,  # L1: fp16 large + fp32 large
    "D-VALRANGE-ASYM": 2,  # L1: fp16 asym + bf16 asym
    # D-SPECIAL-* (L0 + Boundary)
    "D-SPECIAL-DBOUND": 2,  # L0: 1 + L1: 1 + Boundary: 1
    "D-SPECIAL-INF": 2,  # L0: 1 + Boundary: 1
    "D-SPECIAL-NAN": 2,  # L0: 1 + Boundary: 1
    "D-SPECIAL-ZERO": 2,  # L0: 1 + Boundary: 1
    # D-EXC-* (L2)
    "D-EXC-DTYPE": 1,  # L2: int8
    "D-EXC-SHAPE": 1,  # L2: 3D tensor
}

# Mish is element-wise Activation (no cross-element dependency).
# D-SHAPE-TAIL-*/PRIME are optional per coverage-matrix.md §二 Activation row,
# but we include them anyway for thoroughness (T.ceildiv + T.copy tail handling).
# No COVERAGE_NA needed — all forced dims are covered.
COVERAGE_NA = {}


# ========== Main: --level dispatch + exit code ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()  # disable compile cache to avoid stale artifacts
    torch.manual_seed(0)

    blocking_ok = True  # only L0/L1 count toward blocking
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
