"""SituGluGrad operator layered tests: L0/L1/L2/Boundary + main(--level)."""

import argparse
import os
import sys

import tilelang
import torch

# Import kernel/golden from the sibling implementation file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_situ_glu_grad import golden_situ_glu_grad, situ_glu_grad  # noqa: E402

# pytest collects the test functions without calling main(), so disable the
# compile cache at import time (also avoids stale kernels across branches).
tilelang.disable_cache()

# ========== Coverage declarations (for coverage_check.py) ==========
COVERAGE_CATEGORY = "Vector"
COVERAGE_MANIFEST = {}
COVERAGE_NA = {}


# ========== Precision standard (mixed tolerance, by dtype) ==========
def get_precision(dtype_str):
    """Return (atol, rtol, max_abs_error_limit | None, required_matched_ratio).

    No absolute-error gate for this op: grad magnitudes scale with beta /
    linear_beta (Kimi K3 uses up to 25), so the mixed atol + rtol * |golden|
    tolerance is the meaningful criterion and a fixed absolute cap would only
    flag legitimate low-precision rounding at large magnitudes.
    """
    fp_table = {
        "float16": (2**-14, 2**-9, None, 0.99),
        "bfloat16": (2**-10, 2**-6, None, 0.99),
        "float32": (2**-16, 2**-10, None, 0.99),
    }
    return fp_table.get(dtype_str, fp_table["float16"])


def check_precision(actual, golden, dtype_str):
    """Mixed tolerance dual-gate check: return (passed, matched_ratio, max_abs_error)."""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype_str)
    a, g = actual.detach().cpu(), golden.detach().cpu()

    a, g = a.float(), g.float()

    # inf / nan position consistency
    if not torch.equal(torch.isinf(g), torch.isinf(a)):
        return False, 0.0, float("inf")
    if not torch.equal(torch.isnan(g), torch.isnan(a)):
        return False, 0.0, float("inf")

    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    within_abs = max_abs_limit is None or max_abs <= max_abs_limit
    return (ratio >= required_ratio and within_abs), ratio, max_abs


# ========== Input generation ==========
DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def make_input(shape, dtype_str, value_range=None, seed=42):
    """Random input on CPU; `value_range=(lo, hi)` uses uniform, else normal."""
    dt = DTYPE_MAP[dtype_str]
    g = torch.Generator().manual_seed(seed)
    if value_range is None:
        t = torch.randn(shape, generator=g)
    else:
        lo, hi = value_range
        t = torch.rand(shape, generator=g) * (hi - lo) + lo
    return t.to(dt)


def make_grad_y(shape, dim, dtype_str, value_range=None, seed=43):
    d = dim if dim >= 0 else dim + len(shape)
    gy_shape = list(shape)
    gy_shape[d] //= 2
    return make_input(tuple(gy_shape), dtype_str, value_range, seed=seed)


def run_case(name, shape, dtype_str, dim, beta, linear_beta, activate_left=True, value_range=None):
    """Run one case: NPU kernel vs fp64 golden, mixed-tolerance verdict."""
    x = make_input(shape, dtype_str, value_range).npu()
    grad_y = make_grad_y(shape, dim, dtype_str, value_range).npu()
    out = situ_glu_grad(x, grad_y, dim=dim, beta=beta, linear_beta=linear_beta, activate_left=activate_left)
    torch.npu.synchronize()
    golden = golden_situ_glu_grad(x.cpu(), grad_y.cpu(), dim=dim, beta=beta, linear_beta=linear_beta, activate_left=activate_left)
    passed, ratio, max_abs = check_precision(out, golden.to(out.dtype), dtype_str)
    tag = "PASS" if passed else "FAIL"
    print(
        f"[PRECISION_{tag}] {name}: shape={shape} dim={dim} dtype={dtype_str} "
        f"beta={beta} linear_beta={linear_beta} activate_left={activate_left} "
        f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
    )
    return passed


# ========== L0 tests: threshold (dtype x default contract) ==========
def test_situ_glu_grad_l0():
    """L0 threshold tests: one case per dtype + the default parameter set."""
    cases = [
        ("l0_bf16_kimi", (256, 6144), "bfloat16", -1, 4.0, 25.0, True),
        ("l0_fp16_kimi", (256, 6144), "float16", -1, 4.0, 25.0, True),
        ("l0_fp32_kimi", (256, 6144), "float32", -1, 4.0, 25.0, True),
        ("l0_bf16_default", (256, 6144), "bfloat16", -1, 1.0, 0.0, True),
    ]
    ok = True
    for name, shape, dtype_str, dim, beta, linear_beta, al in cases:
        try:
            ok &= run_case(name, shape, dtype_str, dim, beta, linear_beta, al)
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] {name}: {type(e).__name__}: {e}")
            ok = False
    assert ok, "L0 precision tests failed"
    return ok


# ========== L1 tests: functional (dim / params / shapes) ==========
def test_situ_glu_grad_l1():
    """L1 functional tests: dim variants, parameter branches, unaligned shapes."""
    cases = [
        # (name, shape, dtype, dim, beta, linear_beta, activate_left, value_range)
        ("l1_kimi_m4096", (4096, 6144), "bfloat16", -1, 4.0, 25.0, True, None),
        ("l1_activate_left_false", (256, 6144), "bfloat16", -1, 4.0, 25.0, False, None),
        ("l1_beta100", (256, 6144), "bfloat16", -1, 100.0, 25.0, True, None),
        ("l1_lb0_activate_left_false", (256, 6144), "bfloat16", -1, 4.0, 0.0, False, None),
        ("l1_lb_negative", (256, 6144), "bfloat16", -1, 4.0, -1.0, True, None),
        ("l1_dim0_pre1", (6144, 256), "bfloat16", 0, 4.0, 25.0, True, None),
        ("l1_dim1_3d", (8, 6144, 64), "bfloat16", 1, 4.0, 25.0, True, None),
        ("l1_4d_last", (2, 4, 8, 6144), "bfloat16", -1, 4.0, 25.0, True, None),
        ("l1_unaligned_cut", (512, 2050), "bfloat16", -1, 4.0, 25.0, True, None),
        ("l1_unaligned_rows", (1025, 2048), "bfloat16", -1, 4.0, 25.0, True, None),
        ("l1_m1", (1, 6144), "bfloat16", -1, 4.0, 25.0, True, None),
        ("l1_tiny_cut", (128, 4), "bfloat16", -1, 4.0, 25.0, True, None),
        ("l1_fp16_unaligned", (33, 1026), "float16", -1, 4.0, 25.0, True, None),
        ("l1_fp32_3d", (4, 32, 6144), "float32", -2, 4.0, 25.0, True, None),
        ("l1_medium_vals", (512, 6144), "bfloat16", -1, 4.0, 25.0, True, (-50.0, 50.0)),
        ("l1_small", (16, 1152), "bfloat16", -1, 4.0, 25.0, True, None),
    ]
    ok = True
    for name, shape, dtype_str, dim, beta, linear_beta, al, vr in cases:
        try:
            ok &= run_case(name, shape, dtype_str, dim, beta, linear_beta, al, vr)
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] {name}: {type(e).__name__}: {e}")
            ok = False
    # Non-contiguous inputs (NPU slices) are materialized by the host wrapper.
    try:
        base_x = torch.randn(2 * 512, 6144, dtype=torch.bfloat16).npu()
        base_gy = torch.randn(2 * 512, 3072, dtype=torch.bfloat16).npu()
        x_nc = base_x[::2]
        gy_nc = base_gy[::2]
        out = situ_glu_grad(x_nc, gy_nc, dim=-1, beta=4.0, linear_beta=25.0, activate_left=True)
        torch.npu.synchronize()
        golden = golden_situ_glu_grad(x_nc.cpu(), gy_nc.cpu(), dim=-1, beta=4.0, linear_beta=25.0, activate_left=True)
        passed, ratio, max_abs = check_precision(out, golden.to(out.dtype), "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] l1_non_contiguous: shape={tuple(out.shape)} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        ok &= passed
    except Exception as e:  # noqa: BLE001
        print(f"[PRECISION_FAIL] l1_non_contiguous: {type(e).__name__}: {e}")
        ok = False

    assert ok, "L1 functional tests failed"
    return ok


# ========== L2 tests: negative (invalid inputs must be rejected) ==========
def _expect_raise(fn, name):
    try:
        fn()
    except (ValueError, TypeError):
        print(f"[NEGATIVE_PASS] {name}: rejected as expected")
        return True
    print(f"[NEGATIVE_FAIL] {name}: silently accepted")
    return False


def test_situ_glu_grad_l2():
    """L2 negative tests: odd cut / beta=0 / grad_y shape mismatch / bad dim."""
    x_odd = torch.randn(4, 7, dtype=torch.bfloat16)
    gy_odd = torch.randn(4, 3, dtype=torch.bfloat16)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    gy = torch.randn(4, 4, dtype=torch.bfloat16)

    ok = True
    ok &= _expect_raise(lambda: situ_glu_grad(x_odd, gy_odd, dim=-1), "l2_odd_cut")
    ok &= _expect_raise(lambda: situ_glu_grad(x, gy, dim=-1, beta=0.0), "l2_beta_zero")
    ok &= _expect_raise(lambda: situ_glu_grad(x, torch.randn(4, 3, dtype=torch.bfloat16), dim=-1), "l2_grad_y_shape")
    ok &= _expect_raise(lambda: situ_glu_grad(x, gy, dim=2), "l2_bad_dim")
    # Non-blocking: report only (framework convention)
    print(f"[NEGATIVE {'PASS' if ok else 'WARN'}] L2 negative tests")
    return ok


# ========== Boundary tests: special values / degenerate shapes ==========
def test_situ_glu_grad_boundary():
    """Boundary tests: empty tensor, saturation range, tiny values, zero grad_y."""
    ok = True

    # Empty tensor: must early-return before any kernel build
    x = torch.empty(0, 6144, dtype=torch.bfloat16)
    grad_y = torch.empty(0, 3072, dtype=torch.bfloat16)
    out = situ_glu_grad(x, grad_y)
    if out.shape == x.shape and out.numel() == 0:
        print("[BOUNDARY_PASS] b_empty: shape preserved")
    else:
        print(f"[BOUNDARY_FAIL] b_empty: got {tuple(out.shape)}")
        ok = False

    # Saturation range: |gate / beta| up to 25, sigmoid/tanh extremes
    try:
        ok &= run_case("b_saturation", (256, 6144), "bfloat16", -1, 4.0, 25.0, True, (-100.0, 100.0))
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_FAIL] b_saturation: {type(e).__name__}: {e}")
        ok = False

    # Tiny values: near-zero inputs
    try:
        ok &= run_case("b_tiny", (256, 6144), "bfloat16", -1, 4.0, 25.0, True, (-1e-6, 1e-6))
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_FAIL] b_tiny: {type(e).__name__}: {e}")
        ok = False

    print(f"[BOUNDARY {'PASS' if ok else 'WARN'}] boundary tests")
    return ok


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="SituGluGrad NPU kernel tests")
    parser.add_argument("--level", default="all", choices=["all", "l0", "l1", "l2", "boundary"])
    args = parser.parse_args()

    torch.manual_seed(0)

    blocking_ok = True
    if args.level in ("all", "l0"):
        blocking_ok &= test_situ_glu_grad_l0()
    if args.level in ("all", "l1"):
        blocking_ok &= test_situ_glu_grad_l1()
    if args.level in ("all", "l2"):
        test_situ_glu_grad_l2()  # non-blocking
    if args.level in ("all", "boundary"):
        test_situ_glu_grad_boundary()  # non-blocking

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
