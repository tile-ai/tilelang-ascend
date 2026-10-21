"""Layered test suite for dilation_2d.

Levels (dispatchable via --level, all levels also run under pytest where a
failure raises AssertionError instead of returning a bool):
  L0        gate: one representative shape per dispatch/dtype path
  L1        functional: the 20 cann-bench cases (aligned / odd-C bridge /
            flat Path C; SAME/VALID; strides 1-2; rates 1-4; inf / NaN /
            zero / fp16 dynamic-range value ranges) plus fp32 / bf16 and
            irregular-shape supplements, followed by bitwise repeat checks
  L2        exceptions: invalid inputs must be rejected by the wrapper
  Boundary  special values: fp16 dynamic-range bound, inf propagation, NaN
            mask equality, +inf filter tap (out-of-range -inf + inf = NaN),
            all-zero input, all-zero filter (degenerates to max pooling)

Precision standard: mixed tolerance (atol + rtol*|golden|) with a matched-ratio
floor and a max-abs ceiling as the primary gate, plus MERE/MARE over the normal
value domain (|golden| >= small-value threshold) as a secondary gate.  The
golden is a torch fp64 tap-wise reference truncated to the output dtype; the
special-value positions (inf/NaN in the golden) must match the kernel's masks
exactly.
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import tilelang  # noqa: E402
from dilation_2d import dilation_2d  # noqa: E402

# ========== precision standard (mixed tolerance) ==========
_FP_TABLE = {
    "float16": (2**-14, 2**-9, 1e-1, 0.99),
    "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
    "float32": (2**-16, 2**-10, 1e-2, 0.99),
}
# benchmark MERE/MARE thresholds (secondary check)
_MERE_TABLE = {"float16": 2**-10, "bfloat16": 2**-7, "float32": 2**-13}
# small-value domain thresholds: positions with |golden| below these are
# excluded from strict mere/mare and covered by the atol-based mixed
# tolerance instead.
_SMALL_VAL = {"float16": 2**-11, "bfloat16": 2**-8, "float32": 2**-14}
_CANCEL = {"float16": 2**-5, "bfloat16": 2**-3, "float32": 2**-8}


def get_precision(dtype):
    return _FP_TABLE.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Return (passed, matched_ratio, max_abs_error, mere, mare); mere/mare
    cover only the normal value domain (|golden| >= small-value threshold)."""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu().float()
    g = golden.detach().cpu().float()
    special = ~torch.isfinite(g)
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        return False, 0.0, float("inf"), float("inf"), float("inf")
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0, 0.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    matched = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    g_m = g[m].abs()
    a_m = a[m].abs()
    cb = _CANCEL.get(dtype, 2**-8)
    normal = (g_m >= _SMALL_VAL.get(dtype, 2**-14)) & ~((g_m < cb) & (a_m < cb))
    if normal.sum().item() > 0:
        rel = abs_err[normal] / (g_m[normal] + 1e-7)
        mere = rel.mean().item()
        mare = rel.max().item()
    else:
        mere, mare = 0.0, 0.0
    passed = (matched >= required_ratio) and (max_abs <= max_abs_limit)
    return passed, matched, max_abs, mere, mare


# ========== golden (CPU fp64, truncated to output dtype) ==========
def out_geometry(H, W, fh, fw, sh, sw, rh, rw, mode):
    """SAME/VALID geometry with zero pads, matching the wrapper."""
    eff_h = (fh - 1) * rh + 1
    eff_w = (fw - 1) * rw + 1
    if mode == "SAME":
        ho = (H + sh - 1) // sh
        wo = (W + sw - 1) // sw
        pt = max((ho - 1) * sh + eff_h - H, 0) // 2
        pl = max((wo - 1) * sw + eff_w - W, 0) // 2
        return ho, wo, pt, pl
    return (H - eff_h + sh) // sh, (W - eff_w + sw) // sw, 0, 0


def golden_dilation(x, flt, sh, sw, rh, rw, mode):
    """fp64 tap-wise reference; out-of-range taps contribute -inf + f."""
    N, H, W, C = x.shape
    fh, fw, _ = flt.shape
    Ho, Wo, pt, pl = out_geometry(H, W, fh, fw, sh, sw, rh, rw, mode)
    x64, f64 = x.double(), flt.double()
    y = None
    for dy in range(fh):
        for dx in range(fw):
            oys = torch.arange(Ho) * sh - pt + rh * dy
            oxs = torch.arange(Wo) * sw - pl + rw * dx
            vy = (oys >= 0) & (oys < H)
            vx = (oxs >= 0) & (oxs < W)
            tap = x64[:, oys.clamp(0, H - 1)][:, :, oxs.clamp(0, W - 1)] + f64[dy, dx].view(1, 1, 1, C)
            oor = torch.full((C,), float("-inf")) + f64[dy, dx]
            mask = vy.view(1, Ho, 1, 1) & vx.view(1, 1, Wo, 1)
            tap = torch.where(mask, tap, oor.view(1, 1, 1, C))
            y = tap if y is None else torch.maximum(y, tap)
    return y.to(x.dtype), Ho, Wo


_TORCH_DT = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def _gen(shape, vr, seed):
    """Deterministic input for a value range; ('-inf','inf') sprinkles both
    infs over a bounded base, ('nan','nan') is all-NaN (cann-bench value_range
    semantics for cases 13/14)."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = vr
    if str(lo) == "nan":
        return torch.full(shape, float("nan"))
    t = torch.rand(*shape, generator=g) * (float(hi) - float(lo)) + float(lo)
    if str(lo) == "-inf" and str(hi) == "inf":
        t.view(-1)[::97] = float("inf")
        t.view(-1)[::89] = float("-inf")
    return t


def _run_case(name, shape, dtype, k, s, d, mode, vranges=((-1.0, 1.0), (-0.1, 0.1)), seed=0):
    N, H, W, C = shape
    tdt = _TORCH_DT[dtype]
    x = _gen((N, H, W, C), vranges[0], seed).to(tdt)
    flt = _gen((k[0], k[1], C), vranges[1], seed + 1).to(tdt)
    y = dilation_2d(
        x.npu(),
        flt.npu(),
        [1, s[0], s[1], 1],
        [1, d[0], d[1], 1],
        mode,
        [0, 0, 0, 0],
        False,
        "NHWC",
    ).cpu()
    ref, _, _ = golden_dilation(x, flt, s[0], s[1], d[0], d[1], mode)
    passed, ratio, max_abs, mere, mare = check_precision(y, ref, dtype)
    thr = _MERE_TABLE[dtype]
    mere_ok = mere < thr and mare < 10 * thr
    tag = "PASS" if (passed and mere_ok) else "FAIL"
    print(
        f"[PRECISION_{tag}] {name} shape={shape} dtype={dtype} k={k} s={s} d={d} {mode} "
        f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e} mere={mere:.3e} mare={mare:.3e} (thr={thr:.1e})"
    )
    return passed and mere_ok


# ========== L0 cases (gate) ==========
L0_CASES = [
    ("l0_ab_ry_fp16", (2, 64, 64, 64), "float16", (3, 3), (1, 1), (1, 1), "SAME"),
    ("l0_ab_s1_wide_fp16", (1, 64, 128, 128), "float16", (3, 3), (1, 1), (1, 1), "SAME"),
    ("l0_ab_ph_fp32", (1, 16, 128, 128), "float32", (3, 3), (2, 2), (1, 1), "SAME"),
    ("l0_bridge_ry_fp16", (1, 64, 64, 63), "float16", (3, 3), (1, 1), (1, 1), "SAME"),
    ("l0_bridge_ph_fp16", (1, 16, 128, 127), "float16", (3, 3), (2, 2), (1, 1), "SAME"),
    ("l0_path_c_fp32", (1, 11, 13, 17), "float32", (3, 3), (1, 1), (1, 1), "SAME"),
    ("l0_ab_ry_dilated_fp16", (1, 32, 64, 64), "float16", (3, 3), (1, 1), (2, 2), "SAME"),
    ("l0_valid_s1_fp16", (1, 32, 32, 64), "float16", (3, 3), (1, 1), (1, 1), "VALID"),
]


def test_dilation_2d_l0():
    failures = [name for name, shape, dtype, k, s, d, mode in L0_CASES if not _run_case(name, shape, dtype, k, s, d, mode, seed=10)]
    assert not failures, f"L0 gate failed for: {failures}"


# ========== L1: functional (the 20 cann-bench cases + supplements) ==========
# (shape, dtype, kernel, stride, rate, mode, value ranges, coverage tags)
L1_CASES = [
    # --- the 20 cann-bench tasks/level3/dilation_2d cases ---
    ((2, 64, 64, 64), "float16", (3, 3), (1, 1), (1, 1), "SAME", ((-1, 1), (-0.1, 0.1)), ["D-DTYPE-fp16", "D-SHAPE-ALIGNED"]),
    ((4, 128, 128, 128), "float16", (5, 5), (1, 1), (1, 1), "SAME", ((-2, 2), (-0.2, 0.2)), ["D-DTYPE-fp16", "D-PARAM-kernelSize"]),
    (
        (2, 256, 256, 256),
        "float16",
        (3, 3),
        (2, 2),
        (2, 2),
        "SAME",
        ((-3, 3), (-0.3, 0.3)),
        ["D-PARAM-stride", "D-PARAM-rates", "D-VALRANGE-M"],
    ),
    (
        (4, 509, 127, 131),
        "float16",
        (3, 3),
        (1, 1),
        (3, 3),
        "VALID",
        ((-10, 10), (-1, 1)),
        ["D-PARAM-padding_mode", "D-PARAM-rates", "D-VALRANGE-M"],
    ),
    ((7, 67, 251, 257), "float16", (3, 3), (2, 2), (1, 1), "SAME", ((-100, 100), (-10, 10)), ["D-PARAM-stride", "D-VALRANGE-L"]),
    ((2, 128, 512, 512), "float16", (3, 3), (1, 1), (4, 4), "SAME", ((-1000, 1000), (-100, 100)), ["D-PARAM-rates", "D-VALRANGE-L"]),
    ((3, 63, 63, 63), "float16", (3, 3), (1, 1), (1, 1), "SAME", ((-0.1, 0.1), (-0.01, 0.01)), ["D-SHAPE-PRIME", "D-VALRANGE-S"]),
    ((2, 63, 61, 63), "float16", (3, 3), (1, 1), (2, 2), "SAME", ((-1, 2), (-0.1, 0.2)), ["D-VALRANGE-ASYM"]),
    ((3, 47, 23, 23), "float16", (3, 3), (2, 2), (1, 1), "VALID", ((-5, 10), (-0.5, 1)), ["D-PARAM-padding_mode", "D-VALRANGE-ASYM"]),
    ((5, 97, 43, 43), "float16", (3, 3), (1, 1), (3, 3), "SAME", ((-50, 100), (-5, 10)), ["D-PARAM-rates", "D-VALRANGE-ASYM"]),
    ((2, 65, 63, 63), "float16", (3, 3), (1, 1), (1, 1), "SAME", ((-65504, 65504), (-100, 100)), ["D-SPECIAL-DBOUND", "D-VALRANGE-L"]),
    ((3, 47, 31, 97), "float16", (3, 3), (1, 1), (2, 2), "SAME", ((-88, 88), (-8, 8)), ["D-VALRANGE-L"]),
    ((2, 63, 63, 63), "float16", (3, 3), (1, 1), (1, 1), "SAME", (("-inf", "inf"), ("-inf", "inf")), ["D-SPECIAL-INF"]),
    ((2, 11, 13, 17), "float16", (3, 3), (1, 1), (1, 1), "SAME", (("nan", "nan"), ("nan", "nan")), ["D-SPECIAL-NAN"]),
    ((4, 63, 31, 31), "float16", (3, 3), (1, 1), (1, 1), "SAME", ((0, 0), (0, 0)), ["D-SPECIAL-ZERO"]),
    ((2, 64, 63, 63), "float16", (3, 3), (2, 2), (2, 2), "SAME", ((-0.5, 0.5), (-0.05, 0.05)), ["D-PARAM-stride", "D-PARAM-rates"]),
    ((3, 127, 63, 63), "float16", (3, 3), (1, 1), (4, 4), "VALID", ((-1, 3), (-0.1, 0.3)), ["D-PARAM-rates", "D-VALRANGE-ASYM"]),
    ((2, 255, 127, 127), "float16", (3, 3), (2, 2), (1, 1), "SAME", ((-1000, 1000), (-100, 100)), ["D-PARAM-stride", "D-VALRANGE-L"]),
    ((4, 31, 31, 31), "float16", (3, 3), (1, 1), (3, 3), "SAME", ((-0.2, 0.2), (-0.02, 0.02)), ["D-PARAM-rates", "D-VALRANGE-S"]),
    ((2, 127, 127, 127), "float16", (3, 3), (1, 1), (2, 2), "SAME", ((-3, 6), (-0.3, 0.6)), ["D-VALRANGE-ASYM"]),
    # --- dtype supplements (the cann-bench set is fp16-only) ---
    ((1, 64, 64, 64), "float32", (3, 3), (1, 1), (1, 1), "SAME", ((-1, 1), (-0.1, 0.1)), ["D-DTYPE-fp32", "D-SHAPE-ALIGNED"]),
    ((1, 33, 33, 33), "float32", (3, 3), (1, 1), (1, 1), "SAME", ((-1, 1), (-0.1, 0.1)), ["D-DTYPE-fp32"]),
    ((1, 33, 33, 33), "float32", (3, 3), (1, 1), (1, 1), "SAME", ((-2, 2), (-0.2, 0.2)), ["D-DTYPE-fp32"]),
    ((1, 11, 13, 17), "float32", (3, 3), (1, 1), (1, 1), "SAME", ((-1, 1), (-0.1, 0.1)), ["D-DTYPE-fp32", "D-SHAPE-PRIME"]),
    # --- shape supplements ---
    ((1, 16, 64, 64), "float16", (5, 5), (1, 1), (1, 1), "SAME", ((-1, 1), (-0.1, 0.1)), ["D-PARAM-kernelSize"]),
    ((1, 1, 16, 16), "float16", (3, 3), (1, 1), (1, 1), "SAME", ((-1, 1), (-0.1, 0.1)), ["D-SHAPE-TAIL-1"]),
    ((1, 8, 17, 19), "float16", (3, 3), (1, 1), (1, 1), "SAME", ((-1, 1), (-0.1, 0.1)), ["D-SHAPE-TAIL-MID"]),
    ((1, 2, 5, 7), "float32", (3, 3), (1, 1), (1, 1), "SAME", ((-1, 1), (-0.1, 0.1)), ["D-SHAPE-EDGE", "D-DTYPE-fp32"]),
    ((1, 16, 48, 48), "float16", (3, 3), (1, 1), (1, 1), "VALID", ((-1, 1), (-0.1, 0.1)), ["D-PARAM-padding_mode"]),
    ((1, 16, 33, 63), "float16", (3, 3), (1, 1), (1, 1), "SAME", ((-1, 2), (-0.2, 0.1)), ["D-VALRANGE-ASYM"]),
]

# Coverage manifest (dimension -> number of L1/L2/Boundary cases covering it);
# documents the intended coverage of the functional dimensions.
COVERAGE_CATEGORY = "Vector"
COVERAGE_MANIFEST = {
    "D-DTYPE-fp16": 25,
    "D-DTYPE-fp32": 7,
    "D-SHAPE-ALIGNED": 3,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 2,
    "D-SHAPE-EDGE": 1,
    "D-VALRANGE-S": 2,
    "D-VALRANGE-M": 2,
    "D-VALRANGE-L": 4,
    "D-VALRANGE-ASYM": 5,
    "D-PARAM-stride": 3,
    "D-PARAM-rates": 7,
    "D-PARAM-kernelSize": 2,
    "D-PARAM-padding_mode": 3,
    "D-SPECIAL-INF": 2,
    "D-SPECIAL-NAN": 2,
    "D-SPECIAL-ZERO": 2,
    "D-SPECIAL-DBOUND": 2,
    "D-EXC-DTYPE": 2,
    "D-EXC-SHAPE": 4,
    "D-EXC-PARAM": 3,
}
COVERAGE_NA = {
    "D-DTYPE-bf16": "wrapper rejects bfloat16: not in the cann-bench proto and the tilelang frontend cannot lower T.infinity('bfloat16')",
}


def test_dilation_2d_l1():
    failures = []
    for i, (shape, dtype, k, s, d, mode, vr, _tags) in enumerate(L1_CASES):
        name = f"l1_{i + 1}_{shape}_{dtype}_k{k[0]}s{s[0]}d{d[0]}{mode.lower()}"
        if not _run_case(name, shape, dtype, k, s, d, mode, vranges=vr, seed=100 + i):
            failures.append(name)
    assert not failures, f"L1 functional failed for: {failures}"

    # Bitwise repeat checks: the wrapper's kernel/constant caches must make a
    # second call on the same tensors bitwise-identical.
    for shape, dtype, k, s, d, mode in (
        ((2, 64, 64, 64), "float16", (3, 3), (1, 1), (1, 1), "SAME"),
        ((1, 64, 64, 63), "float16", (3, 3), (1, 1), (1, 1), "SAME"),
        ((1, 11, 13, 17), "float16", (3, 3), (1, 1), (1, 1), "SAME"),
    ):
        N, H, W, C = shape
        tdt = _TORCH_DT[dtype]
        x = _gen((N, H, W, C), (-1, 1), 7).to(tdt)
        flt = _gen((k[0], k[1], C), (-0.1, 0.1), 8).to(tdt)
        args = ([1, s[0], s[1], 1], [1, d[0], d[1], 1], mode, [0, 0, 0, 0], False, "NHWC")
        y1 = dilation_2d(x.npu(), flt.npu(), *args)
        y2 = dilation_2d(x.npu(), flt.npu(), *args)
        assert torch.equal(y1, y2), f"repeat call not bitwise-identical for {shape}"
        print(f"[REPEAT_PASS] {shape} {dtype} bitwise identical across calls")


# ========== L2: exceptions ==========
def _run_exception(name, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(f"[L2_PASS] {name}: correctly rejected ({type(e).__name__})")
        return
    raise AssertionError(f"L2 {name}: invalid input silently accepted")


def test_dilation_2d_l2():
    tdt = torch.float16
    x = torch.rand(1, 16, 16, 16).to(tdt)
    flt = torch.rand(3, 3, 16).to(tdt)
    _run_exception("unsupported_dtype", lambda: dilation_2d(x.int().npu(), flt.int().npu(), [1, 1, 1, 1], [1, 1, 1, 1]))
    _run_exception("bfloat16_rejected", lambda: dilation_2d(x.bfloat16().npu(), flt.bfloat16().npu(), [1, 1, 1, 1], [1, 1, 1, 1]))
    _run_exception("x_not_4d", lambda: dilation_2d(x[0].npu(), flt.npu(), [1, 1, 1, 1], [1, 1, 1, 1]))
    _run_exception("filter_not_3d", lambda: dilation_2d(x.npu(), flt[0].npu(), [1, 1, 1, 1], [1, 1, 1, 1]))
    _run_exception("dtype_mismatch", lambda: dilation_2d(x.npu(), flt.float().npu(), [1, 1, 1, 1], [1, 1, 1, 1]))
    _run_exception("filter_ch_mismatch", lambda: dilation_2d(x.npu(), torch.rand(3, 3, 8).to(tdt).npu(), [1, 1, 1, 1], [1, 1, 1, 1]))
    _run_exception("data_format_nchw", lambda: dilation_2d(x.npu(), flt.npu(), [1, 1, 1, 1], [1, 1, 1, 1], data_format="NCHW"))
    _run_exception("invalid_padding_mode", lambda: dilation_2d(x.npu(), flt.npu(), [1, 1, 1, 1], [1, 1, 1, 1], padding_mode="FULL"))
    _run_exception("stride_zero", lambda: dilation_2d(x.npu(), flt.npu(), [1, 0, 1, 1], [1, 1, 1, 1]))
    _run_exception("rate_zero", lambda: dilation_2d(x.npu(), flt.npu(), [1, 1, 1, 1], [1, 1, 0, 1]))
    _run_exception(
        "invalid_out_size",
        lambda: dilation_2d(
            torch.rand(1, 4, 4, 8).to(tdt).npu(), torch.rand(5, 5, 8).to(tdt).npu(), [1, 1, 1, 1], [1, 1, 1, 1], padding_mode="VALID"
        ),
    )


# ========== Boundary: special values ==========
def _run_boundary(name, dtype, fn):
    try:
        out, ref = fn()
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"boundary {name} dtype={dtype} raised: {e}") from e
    passed, ratio, max_abs, mere, mare = check_precision(out, ref, dtype)
    assert passed, f"boundary {name} dtype={dtype} failed: matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
    print(f"[BOUNDARY_PASS] boundary {name} dtype={dtype} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")


def test_dilation_2d_boundary():
    def run_pair(dtype, x, flt, mode="SAME"):
        tdt = _TORCH_DT[dtype]
        y = dilation_2d(
            x.to(tdt).npu(),
            flt.to(tdt).npu(),
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            mode,
            [0, 0, 0, 0],
            False,
            "NHWC",
        ).cpu()
        ref, _, _ = golden_dilation(x.to(tdt), flt.to(tdt), 1, 1, 1, 1, mode)
        return y, ref

    def dbound():
        # Path C (odd tiny C=21): fp16 dynamic-range bound through the flat
        # per-tap chain.
        x = torch.full((1, 8, 20, 21), 65504.0, dtype=torch.float16)
        x.view(-1)[::3] = -65504.0
        flt = torch.full((3, 3, 21), 0.05, dtype=torch.float16)
        flt.view(-1)[::7] = -0.05
        return run_pair("float16", x, flt)

    _run_boundary("fp16_dbound", "float16", dbound)

    def inf_x():
        # AB path (aligned C=64): +/-inf x values, finite filter.
        x = _gen((1, 8, 20, 64), (-1, 1), 3)
        x.view(-1)[::97] = float("inf")
        x.view(-1)[::89] = float("-inf")
        flt = _gen((3, 3, 64), (-0.1, 0.1), 4)
        return run_pair("float32", x, flt)

    def nan_x():
        # AB path (aligned C=64): NaN must propagate through the tile
        # add/max chain exactly like the golden's amax.
        x = _gen((1, 8, 20, 64), (-1, 1), 3)
        x.view(-1)[::53] = float("nan")
        flt = _gen((3, 3, 64), (-0.1, 0.1), 4)
        return run_pair("float16", x, flt)

    def inf_filter():
        # Bridge path (odd C=63): a +inf filter tap gives x + inf = inf for
        # in-range windows and -inf + inf = NaN for out-of-range taps (SAME
        # padding produces both).
        x = _gen((1, 8, 20, 63), (-1, 1), 5)
        flt = _gen((3, 3, 63), (-0.1, 0.1), 6)
        flt[0, 0, 0] = float("inf")
        return run_pair("float16", x, flt)

    def zero_x():
        # Path C (odd tiny C=21), fp32.
        x = torch.zeros(1, 8, 20, 21)
        flt = _gen((3, 3, 21), (-0.1, 0.1), 7)
        return run_pair("float32", x, flt)

    def zero_filter():
        # AB path (aligned C=32): zero filter degenerates dilation to plain
        # max pooling.
        x = _gen((1, 8, 20, 32), (-1, 1), 8)
        flt = torch.zeros(3, 3, 32)
        return run_pair("float16", x, flt)

    for name, dtype, fn in (
        ("inf_x", "float32", inf_x),
        ("nan_x", "float16", nan_x),
        ("inf_filter", "float16", inf_filter),
        ("zero_x", "float32", zero_x),
        ("zero_filter", "float16", zero_filter),
    ):
        _run_boundary(name, dtype, fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="all", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    if args.level in ("l0", "all"):
        test_dilation_2d_l0()
    if args.level in ("l1", "all"):
        test_dilation_2d_l1()
    if args.level in ("l2", "all"):
        test_dilation_2d_l2()
    if args.level in ("boundary", "all"):
        test_dilation_2d_boundary()

    print("Test Passed!")


if __name__ == "__main__":
    main()
