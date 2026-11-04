"""SituGlu layered tests: L0/L1/L2/Boundary + main(--level).

Golden references and test data are prepared entirely on CPU; the runtime
path is H2D -> TileLang kernel -> D2H only, with no aclnn dependency.
Layer conventions (all four layers assert their guarantees so the pytest
gate fails for real -- pytest ignores boolean return values):
- L0/L1 precision cases: [PRECISION_PASS] / [PRECISION_FAIL]
- L2 negative cases (invalid inputs must be rejected) and Boundary
  special-value cases: [BOUNDARY_PASS] / [BOUNDARY_WARN]

Coverage annotation: L1/L2/Boundary cases carry D-* dimension tags;
COVERAGE_CATEGORY / COVERAGE_MANIFEST / COVERAGE_NA feed the coverage gate.
"""

import argparse
import os
import sys
import time

import tilelang
import torch

# Import the kernel under test from the sibling file (add this dir to sys.path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from situ_glu import situ_glu  # noqa: E402

TORCH_TO_TL_DTYPE = {
    torch.float32: "float",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}

SUB_M = 32


def _pick_blocks(m, n_half):
    """Metadata-only tile heuristic (calibrated on Ascend 910_9362, 40 AIV).

    Large shapes use large blocks (deeper pipeline, longer DMA segments);
    small shapes use small blocks to keep cores busy.
    """
    if m >= 4096:
        return 512, 256
    if m >= 1024:
        return 256, 256
    return 128, 128


# Coverage gate declaration: activation category (pure Vector, multi-step).
# TAIL/PRIME are exemptible, but this operator's tiling has real tail-clamp
# paths, so they are covered voluntarily (no COVERAGE_NA exemptions).
COVERAGE_CATEGORY = "Activation"
COVERAGE_NA = {}


# ========== Golden reference (computed on CPU) ==========
def golden_situ_glu(x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True):
    assert x.shape[dim] % 2 == 0, "x.shape[dim] must be even"
    h = x.shape[dim] // 2
    gate, up = torch.split(x, h, dim=dim)
    if not activate_left:
        gate, up = up, gate
    gate = gate.double()
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = up.double()
    if linear_beta > 0:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


# ========== Precision standard (per-dtype mixed tolerance: atol + rtol*|golden|, dual matched-ratio / max-abs gates) ==========
def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio).

    Float: mixed tolerance; integer: exact match with zero error.
    """
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Precision check: return (passed, matched_ratio, max_abs_error).

    Float dual gate: matched_ratio >= required and max_abs_error <= limit;
    integer: element-wise exact equality; inf/nan positions are compared
    structurally (NaN mask equality + Inf sign-exact equality) and excluded
    from the numeric tolerance.
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a, g = actual.detach().cpu(), golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # integer: exact match
        mism = (a != g).sum().item()
        return mism == 0, 1.0 - mism / max(a.numel(), 1), (0.0 if mism == 0 else float("inf"))
    a, g = a.float(), g.float()
    special = ~torch.isfinite(g)
    if special.any():
        # position-sensitive: NaN positions match, Inf positions equal (including sign)
        nan_ok = torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        inf_g = torch.isinf(g[special])
        inf_ok = bool(inf_g.numel() == 0) or torch.equal(a[special][inf_g], g[special][inf_g])
        if not (nan_ok and inf_ok):
            return False, 0.0, float("inf")
    m = torch.isfinite(g)  # compare all finite-golden positions: inf/nan actual fails
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()  # actual inf/nan gives inf/nan err -> fails and raises max_abs
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# ========== Test data construction (all on CPU) ==========
def make_input(shape, torch_dtype, vrange=None):
    """Build an input by value range: randn if vrange is None, else uniform(vrange) cast to dtype."""
    if vrange is None:
        return torch.randn(*shape, dtype=torch_dtype)
    lo, hi = vrange
    return torch.empty(*shape).uniform_(lo, hi).to(torch_dtype)


def make_special(shape, torch_dtype, specials):
    """Finite base values + sparse special-value injection (deterministic
    positions; the seed is fixed in main).

    specials: [(flat_fraction, value), ...]; value may be inf/-inf/nan or a
    finite extreme. At least one special value is guaranteed (the randn base
    provides finite values).
    """
    base = torch.randn(*shape)
    flat = base.flatten()
    n = flat.numel()
    for frac, val in specials:
        flat[int(frac * (n - 1))] = val
    return flat.reshape(shape).to(torch_dtype)


def make_paired(shape, torch_dtype, pairs):
    """Build an input with explicit (gate, up) values at the same output positions.

    Pairs cycle across ALL rows (gate at column 0, up at column half, so the
    two values multiply into the same output element). Filling every row keeps
    the number of affected elements above the matched_ratio gate (>= 1% of the
    tensor), so a regression cannot hide behind the 99% pass ratio.
    """
    x = torch.randn(*shape)
    half = shape[-1] // 2
    for r in range(shape[0]):
        gv, uv = pairs[r % len(pairs)]
        x[r, 0] = gv
        x[r, half] = uv
    return x.to(torch_dtype)


# ========== Host adapter: metadata-only validation + merge view + kernel call ==========
def run_situ_glu(x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True, block_M=None, block_N=None):
    """Run SituGlu on the NPU tensor x; return the output with the original
    dims (dim halved).

    The host side does metadata validation and zero-copy views only (x must
    be contiguous); all computation happens inside the kernel.
    """
    if x.dim() < 1 or x.dim() > 8:
        raise ValueError(f"x rank should be in [1, 8], but got {x.dim()}")
    if x.dtype not in TORCH_TO_TL_DTYPE:
        raise ValueError(f"unsupported dtype {x.dtype}, expected one of {list(TORCH_TO_TL_DTYPE)}")
    if beta == 0.0:
        raise ValueError("beta must be non-zero")
    actual_dim = dim if dim >= 0 else dim + x.dim()
    if actual_dim < 0 or actual_dim >= x.dim():
        raise ValueError(f"dim should be in [{-x.dim()}, {x.dim() - 1}], but got {dim}")
    if x.shape[actual_dim] % 2 != 0:
        raise ValueError(f"x size at dim {actual_dim} should be even, but got {x.shape[actual_dim]}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous (v1)")
    if x.numel() == 0:
        out_shape = list(x.shape)
        out_shape[actual_dim] //= 2
        return x.new_empty(out_shape)
    # Merge to (pre, cut) -- zero-copy view on a contiguous tensor
    pre = 1
    for s in x.shape[:actual_dim]:
        pre *= s
    cut = x.numel() // pre
    x_2d = x.reshape(pre, cut)

    if block_M is None or block_N is None:
        block_M, block_N = _pick_blocks(pre, cut // 2)

    func = situ_glu(block_M, block_N, beta, linear_beta, activate_left, dtype=TORCH_TO_TL_DTYPE[x.dtype], sub_M=SUB_M)
    out_2d = func(x_2d)

    # Restore the original dims (dim halved) -- pure view, kernel output is contiguous
    out_shape = list(x.shape)
    out_shape[actual_dim] //= 2
    return out_2d.reshape(out_shape)


def run_case(name, shape, torch_dtype, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True, vrange=None, x_cpu=None):
    """One case: CPU data + golden -> H2D -> kernel -> D2H -> precision check."""
    # 1. Prepare input and golden on CPU (randoms / specials / golden never run on NPU)
    if x_cpu is None:
        x_cpu = make_input(shape, torch_dtype, vrange)
    ref = golden_situ_glu(x_cpu, dim=dim, beta=beta, linear_beta=linear_beta, activate_left=activate_left)

    # 2. H2D -> kernel -> D2H
    x_npu = x_cpu.npu()
    start = time.time()
    out = run_situ_glu(x_npu, dim=dim, beta=beta, linear_beta=linear_beta, activate_left=activate_left)
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    elapsed = time.time() - start
    actual = out.cpu()

    # 3. Precision check
    passed, ratio, max_abs = check_precision(actual, ref, TORCH_TO_TL_DTYPE[torch_dtype])
    tag = "PASS" if passed else "FAIL"
    print(
        f"[PRECISION_{tag}] {name} shape={tuple(shape)} dtype={TORCH_TO_TL_DTYPE[torch_dtype]} "
        f"dim={dim} beta={beta} linear_beta={linear_beta} activate_left={activate_left} "
        f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e} elapsed={elapsed:.2f}s"
    )
    return passed


# ========== L0: threshold tests (regular shapes, fast convergence) ==========
def test_situ_glu_l0():
    """L0 threshold tests: regular shapes (block-divisible), fast convergence.

    The last case, perf_worst_path, is the largest LLM FFN shape (bf16 +
    largest workload) and doubles as the performance regression case.
    """
    test_configs = [
        # (name, shape, torch_dtype, dim, beta, linear_beta, activate_left)
        ("l0_basic_fp32", (256, 256), torch.float32, -1, 1.0, 0.0, True),
        ("l0_basic_fp16", (256, 256), torch.float16, -1, 1.0, 0.0, True),
        ("l0_basic_bf16", (256, 256), torch.bfloat16, -1, 1.0, 0.0, True),
        ("l0_typical_bf16", (1024, 2048), torch.bfloat16, -1, 1.0, 0.0, True),
        ("l0_activate_right", (256, 256), torch.float32, -1, 1.0, 0.0, False),
        ("l0_beta", (256, 256), torch.float32, -1, 0.7, 0.0, True),
        ("l0_linear_beta", (256, 256), torch.float32, -1, 1.0, 1.5, True),
        ("l0_dim_mid_3d", (2, 128, 512), torch.float32, 1, 1.0, 0.0, True),
        ("perf_worst_path", (4096, 18432), torch.bfloat16, -1, 1.0, 0.0, True),
    ]
    ok = True
    for name, shape, torch_dtype, dim, beta, linear_beta, activate_left in test_configs:
        try:
            ok &= run_case(name, shape, torch_dtype, dim=dim, beta=beta, linear_beta=linear_beta, activate_left=activate_left)
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] {name} shape={tuple(shape)}: {e}")
            ok = False
    assert ok, "L0 precision tests failed"
    return ok


# ========== L1 case table (tags = coverage dimension IDs, reverse-checked by coverage_check.py) ==========
# (name, shape, dtype, vrange, dim, beta, linear_beta, activate_left, tags)
# Shapes follow the coverage matrix: block=(64,64), k=4 baseline + full rank coverage.
L1_CASES = [
    # --- dtype x aligned x value range ---
    ("l1_aligned_fp32", (256, 256), torch.float32, (-1, 1), -1, 1.0, 0.0, True, ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-S"]),
    ("l1_aligned_fp16", (256, 256), torch.float16, (-10, 10), -1, 1.0, 0.0, True, ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    ("l1_aligned_bf16", (256, 256), torch.bfloat16, (-1, 1), -1, 1.0, 0.0, True, ["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-VALRANGE-S"]),
    # --- tail blocks (real clamp paths in this tiling, covered voluntarily) ---
    ("l1_tail1_rows", (257, 256), torch.float16, (-1, 1), -1, 1.0, 0.0, True, ["D-SHAPE-TAIL-1"]),
    ("l1_tail1_cols", (256, 258), torch.float32, (-1, 1), -1, 1.0, 0.0, True, ["D-SHAPE-TAIL-1"]),
    ("l1_tail_mid", (288, 288), torch.bfloat16, (-10, 10), -1, 1.0, 0.0, True, ["D-SHAPE-TAIL-MID"]),
    ("l1_prime", (251, 502), torch.float16, (-1, 1), -1, 1.0, 0.0, True, ["D-SHAPE-PRIME"]),
    # --- degenerate shapes ---
    ("l1_edge_one_row", (1, 256), torch.float32, (-1, 1), -1, 1.0, 0.0, True, ["D-SHAPE-EDGE"]),
    ("l1_edge_min_cols", (256, 2), torch.bfloat16, (-1, 1), -1, 1.0, 0.0, True, ["D-SHAPE-EDGE"]),
    ("l1_edge_single_pair", (1, 2), torch.float16, (-1, 1), -1, 1.0, 0.0, True, ["D-SHAPE-EDGE"]),
    # --- value ranges ---
    ("l1_valrange_large_fp16", (256, 256), torch.float16, (-60000, 60000), -1, 1.0, 0.0, True, ["D-VALRANGE-L"]),
    ("l1_valrange_large_fp32", (256, 256), torch.float32, (-1e30, 1e30), -1, 1.0, 0.0, True, ["D-VALRANGE-L"]),
    ("l1_valrange_asym", (256, 256), torch.float32, (-5, 10), -1, 1.0, 0.0, True, ["D-VALRANGE-ASYM"]),
    # --- rank coverage (proto declares 1-8 dims) ---
    ("l1_rank1", (502,), torch.float32, (-1, 1), 0, 1.0, 0.0, True, ["D-SHAPE-RANK-1", "D-PARAM-dim"]),
    ("l1_rank3", (2, 34, 130), torch.float32, (-1, 1), 1, 1.0, 0.0, True, ["D-SHAPE-RANK-3", "D-PARAM-dim"]),
    ("l1_rank4", (2, 3, 17, 66), torch.bfloat16, (-1, 1), -1, 1.0, 0.0, True, ["D-SHAPE-RANK-4"]),
    ("l1_rank5", (2, 2, 3, 9, 34), torch.float16, (-1, 1), -1, 1.0, 0.0, True, ["D-SHAPE-RANK-5"]),
    ("l1_rank6", (2, 2, 2, 3, 9, 34), torch.float32, (-1, 1), -1, 1.0, 0.0, True, ["D-SHAPE-RANK-6"]),
    ("l1_rank7", (2, 2, 2, 2, 3, 9, 34), torch.bfloat16, (-1, 1), -1, 1.0, 0.0, True, ["D-SHAPE-RANK-7"]),
    ("l1_rank8", (2, 2, 2, 2, 2, 3, 9, 34), torch.float16, (-1, 1), -1, 1.0, 0.0, True, ["D-SHAPE-RANK-8"]),
    # --- non-default attrs ---
    ("l1_param_beta", (300, 258), torch.float32, (-10, 10), -1, 0.7, 0.0, True, ["D-PARAM-beta", "D-SHAPE-TAIL-1"]),
    ("l1_param_linear_beta", (257, 258), torch.float32, (-1, 1), -1, 1.0, 1.5, True, ["D-PARAM-linear_beta", "D-SHAPE-TAIL-1"]),
    ("l1_param_activate_right", (251, 256), torch.float32, (-1, 1), -1, 1.0, 0.0, False, ["D-PARAM-activate_left", "D-SHAPE-PRIME"]),
    ("l1_param_dim_neg", (256, 256), torch.float32, (-1, 1), -2, 1.0, 0.0, True, ["D-PARAM-dim"]),
]

# Coverage summary (consistent with the tags, for manual review; the checker takes the max of tags and manifest)
COVERAGE_MANIFEST = {
    "D-DTYPE-fp16": 7,
    "D-DTYPE-fp32": 11,
    "D-DTYPE-bf16": 5,
    "D-SHAPE-ALIGNED": 3,
    "D-SHAPE-TAIL-1": 4,
    "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 2,
    "D-SHAPE-EDGE": 3,
    "D-SHAPE-RANK-1": 1,
    "D-SHAPE-RANK-3": 1,
    "D-SHAPE-RANK-4": 1,
    "D-SHAPE-RANK-5": 1,
    "D-SHAPE-RANK-6": 1,
    "D-SHAPE-RANK-7": 1,
    "D-SHAPE-RANK-8": 1,
    "D-VALRANGE-S": 2,
    "D-VALRANGE-M": 2,
    "D-VALRANGE-L": 2,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-dim": 3,
    "D-PARAM-beta": 1,
    "D-PARAM-linear_beta": 1,
    "D-PARAM-activate_left": 1,
    "D-SPECIAL-INF": 2,
    "D-SPECIAL-NAN": 2,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-DBOUND": 2,
    "D-EXC-DTYPE": 2,
    "D-EXC-SHAPE": 3,
}


# ========== L1: functional tests (deterministic unaligned shapes: tails/prime/degenerate/rank) ==========
def test_situ_glu_l1():
    """L1 functional tests: parameter combinations, mandatory tail/prime shapes. Returns all-pass."""
    ok = True
    for name, shape, torch_dtype, vrange, dim, beta, linear_beta, activate_left, _tags in L1_CASES:
        try:
            ok &= run_case(
                name, shape, torch_dtype, dim=dim, beta=beta, linear_beta=linear_beta, activate_left=activate_left, vrange=vrange
            )
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] {name} shape={tuple(shape)}: {e}")
            ok = False
    assert ok, "L1 functional tests failed"
    return ok


# ========== L2: negative tests (non-blocking) -- invalid inputs must be rejected ==========
def _run_exception(name, fn):
    """One L2 case: fn() feeds an invalid input that must be rejected.

    Raises -> [BOUNDARY_PASS] (correctly rejected); silent -> [BOUNDARY_WARN]
    (should have been rejected). Non-blocking either way.
    """
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_PASS] l2 {name}: rejected as expected ({type(e).__name__}: {e})")
        return True
    print(f"[BOUNDARY_WARN] l2 {name}: invalid input silently accepted")
    return False


def test_situ_glu_l2():
    """L2 negative tests: unsupported dtypes / invalid shapes / invalid
    parameters must be rejected. Correct raise = PASS; a silently accepted
    invalid input fails the test.
    """
    # (name, tags, fn)
    l2_cases = [
        ("unsupported_dtype_int32", ["D-EXC-DTYPE"], lambda: run_situ_glu(torch.randn(64, 64).to(torch.int32).npu())),
        ("unsupported_dtype_int64", ["D-EXC-DTYPE"], lambda: run_situ_glu(torch.zeros(64, 64, dtype=torch.int64).npu())),
        ("odd_dim", ["D-EXC-SHAPE"], lambda: run_situ_glu(torch.randn(64, 63).npu())),
        ("dim_out_of_range", ["D-EXC-SHAPE"], lambda: run_situ_glu(torch.randn(64, 64).npu(), dim=2)),
        ("dim_negative_out_of_range", ["D-EXC-SHAPE"], lambda: run_situ_glu(torch.randn(64, 64).npu(), dim=-3)),
        ("beta_zero", [], lambda: run_situ_glu(torch.randn(64, 64).npu(), beta=0.0)),
        ("non_contiguous", [], lambda: run_situ_glu(torch.randn(64, 64).npu().t())),
        ("rank_zero_scalar", [], lambda: run_situ_glu(torch.tensor(3.0).npu())),
        ("rank_nine", [], lambda: run_situ_glu(torch.randn(2, 2, 2, 2, 2, 2, 2, 2, 4).npu())),
    ]
    ok = True
    for name, _tags, fn in l2_cases:
        ok &= _run_exception(name, fn)
    assert ok, "L2 negative tests failed (some invalid input was silently accepted)"
    return ok


# ========== Boundary: special values (legal extremes, precision-checked, non-blocking) ==========
def _run_boundary(name, shape, torch_dtype, tags, make_fn, **kwargs):
    """One Boundary case: legal special values (INF/NAN/extremes/empty).

    Checked against the same precision standard as L0/L1 (check_precision,
    per-dtype thresholds): pass -> [BOUNDARY_PASS]; fail or raise ->
    [BOUNDARY_WARN] and the failure propagates to the caller.
    """
    try:
        x_cpu = make_fn()
        ref = golden_situ_glu(x_cpu, **kwargs)
        out = run_situ_glu(x_cpu.npu(), **kwargs)
        actual = out.cpu()
        passed, ratio, max_abs = check_precision(actual, ref, TORCH_TO_TL_DTYPE[torch_dtype])
        tag = "PASS" if passed else "WARN"
        print(
            f"[BOUNDARY_{tag}] boundary {name} shape={tuple(shape)} "
            f"dtype={TORCH_TO_TL_DTYPE[torch_dtype]} matched_ratio={ratio:.4f} "
            f"max_abs={max_abs:.3e}"
        )
        return passed
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_WARN] boundary {name} shape={tuple(shape)}: {e}")
        return False


def test_situ_glu_boundary():
    """Boundary tests: sparse +/-inf / nan / all-zero / dtype bounds / empty.

    Special values use deterministic data (finite base + sparse injection);
    see check_precision for the position-sensitive comparison.
    """
    inf = float("inf")
    # Injection positions are flat-index based so both the gate half (col < 64) and the up half (col >= 64) are hit.
    boundary_cases = [
        (
            "inf_mixed",
            (64, 128),
            torch.float32,
            ["D-SPECIAL-INF"],
            lambda: make_special((64, 128), torch.float32, [(0.0, inf), (0.3, -inf), (0.059, -inf), (0.171, inf)]),
            {},
        ),
        (
            "inf_up_bounded",
            (64, 128),
            torch.float32,
            ["D-SPECIAL-INF"],
            lambda: make_special((64, 128), torch.float32, [(0.1, -inf), (0.4, inf), (0.9, inf)]),
            {"linear_beta": 2.0},
        ),
        (
            "nan_mixed",
            (64, 128),
            torch.float32,
            ["D-SPECIAL-NAN"],
            lambda: make_special((64, 128), torch.float32, [(0.05, float("nan")), (0.6, float("nan"))]),
            {},
        ),
        (
            "nan_mixed_fp16",
            (64, 128),
            torch.float16,
            ["D-SPECIAL-NAN"],
            lambda: make_special((64, 128), torch.float16, [(0.2, float("nan")), (0.8, float("nan"))]),
            {},
        ),
        ("all_zero", (64, 128), torch.float32, ["D-SPECIAL-ZERO"], lambda: torch.zeros(64, 128), {}),
        (
            "dbound_fp16",
            (64, 128),
            torch.float16,
            ["D-SPECIAL-DBOUND"],
            lambda: make_special((64, 128), torch.float16, [(0.0, 65504.0), (0.3, -65504.0), (0.6, 65504.0)]),
            {},
        ),
        (
            "dbound_fp32",
            (64, 128),
            torch.float32,
            ["D-SPECIAL-DBOUND"],
            lambda: make_special((64, 128), torch.float32, [(0.0, 3.4e38), (0.3, -3.4e38), (0.6, 3.4e38)]),
            {},
        ),
        ("empty_rows", (0, 128), torch.float32, [], lambda: torch.randn(0, 128), {}),
        # Targeted regression for the tanh small-argument blend: the sigmoid
        # composition alone quantizes tanh(z) to 0 for |z| < ~1.2e-7 (fp32 ulp
        # at 0.5), which a huge up value amplifies into a visible error
        # (gate=1e-8, up=1e8 would yield 0 instead of ~0.5).
        (
            "tiny_gate_huge_up",
            (64, 128),
            torch.float32,
            ["D-SPECIAL-DBOUND"],
            lambda: make_paired((64, 128), torch.float32, [(1e-8, 1e8), (-1e-8, -1e8), (1e-8, -1e8), (5e-8, 1e8), (1e-9, 3.4e38)]),
            {},
        ),
        (
            "tiny_gate_huge_up_bf16",
            (64, 128),
            torch.bfloat16,
            ["D-SPECIAL-DBOUND"],
            lambda: make_paired((64, 128), torch.bfloat16, [(1e-8, 1e8), (-1e-8, -1e8)]),
            {},
        ),
        # Up-path blend: up/linear_beta inside the dead zone (|up/lb| < ~2.5e-8
        # zeroes the composed tanh). beta=100 amplifies situ_a so the y error
        # (~2e-4) clearly exceeds the fp32 atol.
        (
            "tiny_up_bounded",
            (64, 128),
            torch.float32,
            ["D-SPECIAL-DBOUND"],
            lambda: make_paired((64, 128), torch.float32, [(10.0, 2e-5), (10.0, -2e-5)]),
            {"beta": 100.0, "linear_beta": 1000.0},
        ),
    ]
    ok = True
    for name, shape, torch_dtype, _tags, make_fn, kwargs in boundary_cases:
        ok &= _run_boundary(name, shape, torch_dtype, _tags, make_fn, **kwargs)
    assert ok, "Boundary tests failed"
    return ok


# ========== Main: --level dispatch + exit code ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()  # disable the compile cache to avoid stale artifacts
    torch.manual_seed(0)

    blocking_ok = True  # every executed layer asserts its guarantees
    try:
        if args.level in ("l0", "all"):
            blocking_ok &= test_situ_glu_l0()
        if args.level in ("l1", "all"):
            blocking_ok &= test_situ_glu_l1()
        if args.level in ("l2", "all"):
            blocking_ok &= test_situ_glu_l2()
        if args.level in ("boundary", "all"):
            blocking_ok &= test_situ_glu_boundary()
    except AssertionError as exc:
        print(f"Test Failed: {exc}")
        sys.exit(1)

    if blocking_ok:
        print("Test Passed!")  # all L0/L1 passed (exit code + this line are the verdict)
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
