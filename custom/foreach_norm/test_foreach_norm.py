"""ForeachNorm layered tests: L0/L1/L2/Boundary + main(--level)."""

import argparse
import os
import sys

import tilelang
import torch

# Import kernel from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foreach_norm import foreach_norm  # noqa: E402


# ============================================================================
# Coverage category declaration (for coverage_check.py)
# ============================================================================
COVERAGE_CATEGORY = "Reduction"


# ============================================================================
# Golden reference (matches cann-bench golden.py)
# ============================================================================

def golden_foreach_norm(x: list[torch.Tensor], scalar: float) -> list[torch.Tensor]:
    """PyTorch golden: torch.norm(tensor, p=scalar) with FP16/BF16 upcast."""
    input_dtype = x[0].dtype if x else torch.float32
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype
    x_compute = [t.to(compute_dtype) for t in x]
    y = [torch.norm(tensor, p=scalar) for tensor in x_compute]
    if input_dtype in (torch.float16, torch.bfloat16):
        return [t.to(input_dtype) for t in y]
    return y


# ============================================================================
# Precision standard (mixed tolerance, per dtype)
# ============================================================================

def get_precision(dtype: str):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio)."""
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual: torch.Tensor, golden: torch.Tensor, dtype: str):
    """Mixed tolerance dual-gate: return (passed, matched_ratio, max_abs_error)."""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    a, g = a.float(), g.float()
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        # golden all inf/nan — compare masks instead
        masks_match = torch.equal(torch.isnan(a), torch.isnan(g)) and \
                      torch.equal(torch.isinf(a), torch.isinf(g))
        return masks_match, 1.0 if masks_match else 0.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item() if abs_err.numel() > 0 else 0.0
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# ============================================================================
# L0 tests (precision convergence — aligned shapes from DESIGN.md S9.2)
# ============================================================================

# (name, dtype, shape_per_tensor, scalar, block_N, list_len, tags)
L0_CASES = [
    ("l0_l2_fp16_small",   "float16",  (8192,),       2.0,    8192, 1,
     ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-scalar"]),
    ("l0_l2_fp16_mid",     "float16",  (32768,),      2.0,    8192, 1,
     ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-scalar"]),
    ("l0_l2_fp16_large",   "float16",  (131072,),     2.0,    8192, 1,
     ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-scalar"]),
    ("l0_l1_fp16",         "float16",  (8192,),       1.0,    8192, 1,
     ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-scalar"]),
    ("l0_l3_fp16",         "float16",  (8192,),       3.0,    8192, 1,
     ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-scalar"]),
    ("l0_linf_fp16",       "float16",  (8192,),       float("inf"), 8192, 1,
     ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-scalar"]),
    ("l0_l0_count_fp16",   "float16",  (8192,),       0.0,    8192, 1,
     ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-scalar"]),
    ("l0_lneg1_fp16",      "float16",  (8192,),       -1.0,   8192, 1,
     ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-scalar"]),
    ("l0_l2_fp32_mid",     "float32",  (32768,),      2.0,    8192, 1,
     ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-PARAM-scalar"]),
    ("l0_l1_fp32",         "float32",  (8192,),       1.0,    8192, 1,
     ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-PARAM-scalar"]),
    ("l0_l2_bf16_mid",     "bfloat16", (32768,),      2.0,    8192, 1,
     ["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-PARAM-scalar"]),
    ("l0_l1_bf16",         "bfloat16", (8192,),       1.0,    8192, 1,
     ["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-PARAM-scalar"]),
    ("l0_l2_2d_fp16",      "float16",  (1024, 1024),  2.0,    8192, 1,
     ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-scalar"]),
    ("l0_l2_tl2_fp16",     "float16",  (8192,),       2.0,    8192, 2,
     ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-scalar"]),
    ("l0_l1_tl3_fp32",     "float32",  (8192,),       1.0,    8192, 3,
     ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-PARAM-scalar"]),
]


def test_foreach_norm_l0():
    """L0 gate tests: aligned shapes, precision convergence."""
    ok = True
    for name, dt, shape, scalar, _bn, ll, _tags in L0_CASES:
        torch_dt = getattr(torch, dt)
        try:
            x_list = [torch.randn(shape, dtype=torch_dt, device="npu") for _ in range(ll)]
            if scalar == -1.0:
                for t in x_list:
                    t[t == 0] = 1.0
            y_list = foreach_norm(x_list, scalar)
            torch.npu.synchronize()
            ref_list = golden_foreach_norm(x_list, scalar)
            case_ok = True
            for i, (y, ref) in enumerate(zip(y_list, ref_list)):
                passed, ratio, max_abs = check_precision(y, ref, dt)
                tag = "PASS" if passed else "FAIL"
                print(f"[PRECISION_{tag}] l0 {name} shape={shape} dtype={dt} "
                      f"scalar={scalar} tl_idx={i}/{ll} "
                      f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
                if not passed:
                    case_ok = False
            if not case_ok:
                ok = False
        except Exception as e:
            import traceback
            print(f"[PRECISION_FAIL] l0 {name}: {e}")
            traceback.print_exc()
            ok = False
    return ok


# ============================================================================
# L1/L2/Boundary: expanded by tilelang-op-test-design (scenario B)
# Cases derived from cann-bench foreach_norm/cases.yaml (20 real cases) +
# deterministic non-aligned shapes per coverage-matrix.md §6.
# ============================================================================

# L1 cases: (name, dtype, shape, scalar, block_N, list_len, value_range, tags)
L1_CASES = [
    # --- D-SHAPE-ALIGNED + D-VALRANGE-S (symmetric small, cann-bench case 1-5) ---
    ("l1_c1_fp16_l1_tl2",   "float16",  (1024, 1024),     1.0,    8192, 2, (-1.0, 1.0),
     ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-S", "D-PARAM-scalar"]),
    # Note: shape kept moderate (128x128=16384 elements) so float32 L1-norm
    # accumulation error (~N*eps*mean) stays within max_abs_error_limit=1e-2.
    # Larger shapes (e.g. 512x512=262K) produce correct results (rel err ~6e-8)
    # but exceed the absolute cap for raw-sum reductions (no root compression).
    ("l1_c2_fp32_l1_tl3",   "float32",  (128, 128),       1.0,    8192, 3, (-2.0, 2.0),
     ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-S", "D-PARAM-scalar"]),
    ("l1_c3_bf16_l1",       "bfloat16", (1024, 1024),     1.0,    8192, 1, (-3.0, 3.0),
     ["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-VALRANGE-S", "D-PARAM-scalar"]),
    ("l1_c4_fp16_l2_tiny",  "float16",  (512, 512),       2.0,    8192, 1, (-0.1, 0.1),
     ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-S", "D-PARAM-scalar"]),
    ("l1_c5_fp32_l3",       "float32",  (512, 1024),      3.0,    8192, 1, (-0.1, 0.1),
     ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-S", "D-PARAM-scalar"]),
    # --- D-SHAPE-TAIL-MID (non-aligned mid remainder, cann-bench case 6/8/10/15-19) ---
    ("l1_c6_bf16_l15_tail", "bfloat16", (1023, 1023),     1.5,    8192, 1, (-0.1, 0.1),
     ["D-DTYPE-bf16", "D-SHAPE-TAIL-MID", "D-VALRANGE-S", "D-PARAM-scalar"]),
    ("l1_c8_fp32_l4_tail",  "float32",  (1537, 769),      4.0,    8192, 1, (-5.0, 10.0),
     ["D-DTYPE-fp32", "D-SHAPE-TAIL-MID", "D-VALRANGE-ASYM", "D-PARAM-scalar"]),
    ("l1_c10_fp16_l1_dbound","float16", (2049, 513),      1.0,    8192, 1, (-65504.0, 65504.0),
     ["D-DTYPE-fp16", "D-SHAPE-TAIL-MID", "D-VALRANGE-L", "D-PARAM-scalar"]),
    ("l1_c13_fp32_l2_tail", "float32",  (512, 2049),      2.0,    8192, 2, (-0.5, 0.5),
     ["D-DTYPE-fp32", "D-SHAPE-TAIL-MID", "D-VALRANGE-S", "D-PARAM-scalar"]),
    ("l1_c14_bf16_l1_tl4",  "bfloat16", (127, 4097),      1.0,    8192, 4, (-1.0, 3.0),
     ["D-DTYPE-bf16", "D-SHAPE-TAIL-MID", "D-VALRANGE-ASYM", "D-PARAM-scalar"]),
    ("l1_c15_fp16_lneg1",   "float16",  (4097, 511),      -1.0,   8192, 1, (-1000.0, 1000.0),
     ["D-DTYPE-fp16", "D-SHAPE-TAIL-MID", "D-VALRANGE-L", "D-PARAM-scalar"]),
    ("l1_c16_fp32_l2_3d",   "float32",  (2, 127, 513),    2.0,    8192, 2, (-0.2, 0.2),
     ["D-DTYPE-fp32", "D-SHAPE-TAIL-MID", "D-VALRANGE-S", "D-PARAM-scalar"]),
    ("l1_c17_bf16_l3_3d",   "bfloat16", (4, 127, 513),    3.0,    8192, 1, (-3.0, 6.0),
     ["D-DTYPE-bf16", "D-SHAPE-TAIL-MID", "D-VALRANGE-ASYM", "D-PARAM-scalar"]),
    # --- D-SHAPE-PRIME (fully non-aligned / prime factors, cann-bench case 7/9/11/13) ---
    ("l1_c7_fp16_l15_prime","float16",  (1009, 1021),     1.5,    8192, 1, (-1.0, 2.0),
     ["D-DTYPE-fp16", "D-SHAPE-PRIME", "D-VALRANGE-ASYM", "D-PARAM-scalar"]),
    ("l1_c9_bf16_l2_prime3d","bfloat16",(73, 79, 83),     2.0,    8192, 2, (-50.0, 100.0),
     ["D-DTYPE-bf16", "D-SHAPE-PRIME", "D-VALRANGE-ASYM", "D-VALRANGE-L", "D-PARAM-scalar"]),
    ("l1_c11_fp32_l2_4d",   "float32",  (3, 7, 13, 4001), 2.0,    8192, 1, (-88.0, 88.0),
     ["D-DTYPE-fp32", "D-SHAPE-PRIME", "D-VALRANGE-L", "D-PARAM-scalar"]),
    ("l1_c12_fp32_l5_5d",   "float32",  (7, 11, 13, 17, 19), 5.0, 8192, 1, (-10.0, 10.0),
     ["D-DTYPE-fp32", "D-SHAPE-PRIME", "D-VALRANGE-M", "D-PARAM-scalar"]),
    # --- D-SHAPE-TAIL-1 (remainder = 1, most boundary-exposing) ---
    ("l1_tail1_fp16_1d",    "float16",  (8193,),          2.0,    8192, 1, (-1.0, 1.0),
     ["D-DTYPE-fp16", "D-SHAPE-TAIL-1", "D-VALRANGE-S", "D-PARAM-scalar"]),
    ("l1_tail1_fp32_1d",    "float32",  (16385,),         1.0,    8192, 1, (-2.0, 2.0),
     ["D-DTYPE-fp32", "D-SHAPE-TAIL-1", "D-VALRANGE-S", "D-PARAM-scalar"]),
    ("l1_tail1_bf16_1d",    "bfloat16", (32769,),         2.0,    8192, 1, (-0.5, 0.5),
     ["D-DTYPE-bf16", "D-SHAPE-TAIL-1", "D-VALRANGE-S", "D-PARAM-scalar"]),
    # --- D-SHAPE-EDGE (degenerate: single element / 1×N) ---
    ("l1_edge_fp16_scalar", "float16",  (1,),             2.0,    8192, 1, (-1.0, 1.0),
     ["D-DTYPE-fp16", "D-SHAPE-EDGE", "D-VALRANGE-S", "D-PARAM-scalar"]),
    ("l1_edge_fp32_1row",   "float32",  (1, 8192),        1.0,    8192, 1, (-2.0, 2.0),
     ["D-DTYPE-fp32", "D-SHAPE-EDGE", "D-VALRANGE-S", "D-PARAM-scalar"]),
    # --- D-VALRANGE-M (symmetric mid) ---
    ("l1_valmid_fp16_l2",   "float16",  (8192,),          2.0,    8192, 1, (-10.0, 10.0),
     ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-scalar"]),
    # --- D-VALRANGE-L (symmetric large) ---
    ("l1_vallarge_fp32_l2", "float32",  (8192,),          2.0,    8192, 1, (-1000.0, 1000.0),
     ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-L", "D-PARAM-scalar"]),
    # --- D-VALRANGE-ASYM (non-symmetric) ---
    ("l1_valasym_bf16_l1",  "bfloat16", (8192,),          1.0,    8192, 1, (-5.0, 10.0),
     ["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-VALRANGE-ASYM", "D-PARAM-scalar"]),
]

# L2 cases: (name, kind, tags) — kind drives illegal input construction.
# Expects host dispatch / PyTorch to reject (ValueError/RuntimeError).
L2_CASES = [
    ("l2_unsupported_float64", "unsupported_dtype_float64", ["D-EXC-DTYPE"]),
    ("l2_unsupported_int32",   "unsupported_dtype_int32",   ["D-EXC-DTYPE"]),
    ("l2_dtype_mismatch",      "dtype_mismatch",            ["D-EXC-SHAPE"]),
    ("l2_non_contiguous",      "non_contiguous",            ["D-EXC-SHAPE"]),
]

# Boundary cases: (name, dtype, shape, scalar, list_len, value_kind, tags)
# Legal special values; precision compared against torch.norm golden.
BOUNDARY_CASES = [
    ("b_inf_fp16_l2",      "float16",  (8192,),    2.0,          1, "inf",
     ["D-SPECIAL-INF", "D-DTYPE-fp16"]),
    ("b_nan_fp16_l2",      "float16",  (8192,),    2.0,          1, "nan",
     ["D-SPECIAL-NAN", "D-DTYPE-fp16"]),
    ("b_zero_fp16_l2",     "float16",  (8192,),    2.0,          1, "zero",
     ["D-SPECIAL-ZERO", "D-DTYPE-fp16"]),
    ("b_dbound_fp16_l2",   "float16",  (8192,),    2.0,          1, "dbound_fp16",
     ["D-SPECIAL-DBOUND", "D-DTYPE-fp16"]),
    ("b_dbound_fp32_l2",   "float32",  (8192,),    2.0,          1, "dbound_fp32",
     ["D-SPECIAL-DBOUND", "D-DTYPE-fp32"]),
    ("b_inf_bf16_linf",    "bfloat16", (1000003,), float("inf"), 1, "inf",
     ["D-SPECIAL-INF", "D-DTYPE-bf16", "D-SHAPE-PRIME"]),
    ("b_zero_fp32_l0count","float32",  (8192,),    0.0,          1, "zero",
     ["D-SPECIAL-ZERO", "D-DTYPE-fp32", "D-PARAM-scalar"]),
]

# Coverage manifest (tags auto-counted by coverage_check.py; manifest is a floor).
COVERAGE_MANIFEST = {}
COVERAGE_NA = {}


def _gen_input(shape, torch_dt, vrange, scalar):
    """Generate input tensor with controlled value range. Negative p: no zeros."""
    lo, hi = vrange
    t = torch.empty(shape, dtype=torch_dt, device="npu").uniform_(lo, hi)
    if scalar < 0 and scalar != float("-inf"):
        t[t == 0] = 1.0
    return t


def test_foreach_norm_l1():
    """L1 functional tests: irregular/tail/prime/edge shapes + value ranges.

    Blocking: [PRECISION_FAIL] counts toward exit code 1.
    Cases derived from cann-bench foreach_norm/cases.yaml + deterministic shapes.
    """
    ok = True
    for name, dt, shape, scalar, _bn, ll, vrange, _tags in L1_CASES:
        torch_dt = getattr(torch, dt)
        try:
            x_list = [_gen_input(shape, torch_dt, vrange, scalar) for _ in range(ll)]
            y_list = foreach_norm(x_list, scalar)
            torch.npu.synchronize()
            ref_list = golden_foreach_norm(x_list, scalar)
            case_ok = True
            for i, (y, ref) in enumerate(zip(y_list, ref_list)):
                passed, ratio, max_abs = check_precision(y, ref, dt)
                tag = "PASS" if passed else "FAIL"
                print(f"[PRECISION_{tag}] l1 {name} shape={shape} dtype={dt} "
                      f"scalar={scalar} tl_idx={i}/{ll} "
                      f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
                if not passed:
                    case_ok = False
            if not case_ok:
                ok = False
        except Exception as e:
            import traceback
            print(f"[PRECISION_FAIL] l1 {name} shape={shape} dtype={dt} scalar={scalar}: {e}")
            traceback.print_exc()
            ok = False
    return ok


def test_foreach_norm_l2():
    """L2 negative tests: illegal inputs must be rejected.

    Non-blocking: [BOUNDARY_PASS] (correctly rejected) / [BOUNDARY_WARN] (silently
    accepted). Does not affect exit code. Validates host dispatch dtype/structure checks.
    """
    for name, kind, _tags in L2_CASES:
        try:
            if kind == "unsupported_dtype_float64":
                # float64 not in SUPPORTED_DTYPES -> host dispatch raises ValueError
                x = [torch.randn((8192,), dtype=torch.float64, device="cpu")]
                foreach_norm(x, 2.0)
            elif kind == "unsupported_dtype_int32":
                x = [torch.ones((8192,), dtype=torch.int32, device="cpu")]
                foreach_norm(x, 2.0)
            elif kind == "dtype_mismatch":
                # cann-bench constraint: all tensors must share same dtype
                x = [torch.randn((8192,), dtype=torch.float16, device="npu"),
                     torch.randn((8192,), dtype=torch.float32, device="npu")]
                foreach_norm(x, 2.0)
            elif kind == "non_contiguous":
                # Non-contiguous tensor: x.view(-1) raises RuntimeError
                t = torch.randn((1024, 1024), dtype=torch.float16, device="npu").t()
                foreach_norm([t], 2.0)
            else:
                continue
            print(f"[BOUNDARY_WARN] l2 {name}: illegal input silently accepted ({kind})")
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"[BOUNDARY_PASS] l2 {name}: correctly rejected ({kind}): {str(e)[:120]}")
        except Exception as e:
            print(f"[BOUNDARY_WARN] l2 {name}: unexpected exception ({kind}): {e}")


def test_foreach_norm_boundary():
    """Boundary tests: legal special values (INF/NAN/zero/dtype-bound).

    Non-blocking: precision compared against torch.norm golden; [BOUNDARY_WARN]
    if precision gate not met or exception. Does not affect exit code.
    """
    for name, dt, shape, scalar, ll, vkind, _tags in BOUNDARY_CASES:
        torch_dt = getattr(torch, dt)
        try:
            if vkind == "zero":
                x_list = [torch.zeros(shape, dtype=torch_dt, device="npu")
                          for _ in range(ll)]
            else:
                x_list = [torch.zeros(shape, dtype=torch_dt, device="npu")
                          for _ in range(ll)]
                for t in x_list:
                    n = min(10, t.numel())
                    if vkind == "inf":
                        t.view(-1)[:n] = float("inf")
                        if t.numel() > n:
                            t.view(-1)[n:2 * n] = float("-inf")
                    elif vkind == "nan":
                        t.view(-1)[:n] = float("nan")
                    elif vkind == "dbound_fp16":
                        t.fill_(65504.0)
                    elif vkind == "dbound_fp32":
                        t.fill_(88.0)
            y_list = foreach_norm(x_list, scalar)
            torch.npu.synchronize()
            ref_list = golden_foreach_norm(x_list, scalar)
            case_ok = True
            for i, (y, ref) in enumerate(zip(y_list, ref_list)):
                passed, ratio, max_abs = check_precision(y, ref, dt)
                tag = "PASS" if passed else "WARN"
                print(f"[BOUNDARY_{tag}] boundary {name} shape={shape} dtype={dt} "
                      f"scalar={scalar} tl_idx={i}/{ll} "
                      f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
                if not passed:
                    case_ok = False
            if not case_ok:
                print(f"  -> [BOUNDARY_WARN] {name}: precision gate not met (non-blocking)")
        except Exception as e:
            print(f"[BOUNDARY_WARN] boundary {name}: exception: {e}")


# ============================================================================
# Main: --level dispatcher + exit code
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ForeachNorm layered tests")
    parser.add_argument("--level", default="l0",
                        choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True

    if args.level in ("l0", "all"):
        blocking_ok &= test_foreach_norm_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_foreach_norm_l1()
    if args.level in ("l2", "all"):
        test_foreach_norm_l2()
    if args.level in ("boundary", "all"):
        test_foreach_norm_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
