# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms of the CANN Open Software License Agreement Version 2.0 (the
# "License"). You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED.
# ----------------------------------------------------------------------------

"""Layered test suite for depthwise_conv_2d.

Levels (dispatchable via --level, all levels also run under pytest where a
failure raises AssertionError instead of returning a bool):
  L0        gate: one representative shape per dispatch/dtype path
  L1        functional: deterministic irregular shapes, dtypes, value ranges
  L2        exceptions: invalid inputs must be rejected by the wrapper
  Boundary  special values: inf / NaN / all-zero / fp16 dynamic-range bound

Precision standard: mixed tolerance (atol + rtol*|golden|) with a matched-ratio
floor and a max-abs ceiling as the primary gate, plus MERE/MARE over the normal
value domain (|golden| >= small-value threshold) as a secondary gate.  The
golden is a torch fp64 conv2d truncated to the output dtype.
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import tilelang  # noqa: E402
from depthwise_conv_2d import depthwise_conv_2d  # noqa: E402

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
def golden_depthwise_conv_2d(x, weight, bias, kernelSize, stride, padding, dilation, groups):
    w = weight.double().unsqueeze(1)
    y = torch.nn.functional.conv2d(
        x.double(), w, bias.double(), stride=tuple(stride), padding=tuple(padding), dilation=tuple(dilation), groups=groups
    )
    return y.to(x.dtype)


_TORCH_DT = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}

# ========== L0 cases (gate) ==========
L0_CASES = [
    ("l0_sw1_fp16", (2, 64, 64, 64), "float16", (3, 3), (1, 1), (1, 1), (1, 1)),
    ("l0_sw1_k5_fp32", (1, 128, 64, 64), "float32", (5, 5), (1, 1), (2, 2), (1, 1)),
    ("l0_sw1_d2_fp32", (1, 16, 64, 64), "float32", (3, 3), (1, 1), (2, 2), (2, 2)),
    ("l0_sw1_k1_fp16", (1, 64, 64, 64), "float16", (1, 1), (1, 1), (0, 0), (1, 1)),
    ("l0_sw2_bf16", (2, 32, 128, 128), "bfloat16", (3, 3), (2, 2), (1, 1), (1, 1)),
    ("l0_sw2_k5_fp16", (1, 16, 128, 128), "float16", (5, 5), (2, 2), (2, 2), (1, 1)),
    ("l0_sw4_fp16", (1, 16, 256, 256), "float16", (3, 3), (4, 4), (2, 2), (1, 1)),
    ("l0_sw3_scalar", (1, 8, 192, 192), "float16", (3, 3), (3, 3), (1, 1), (1, 1)),
]


def _run_case(name, shape, dtype, k, s, p, d, vranges=(-1.0, 1.0), seed=0):
    torch.manual_seed(seed)
    N, C, H, W = shape
    tdt = _TORCH_DT[dtype]
    x = (torch.rand(N, C, H, W) * (vranges[1] - vranges[0]) + vranges[0]).to(tdt)
    weight = (torch.rand(C, k[0], k[1]) * 0.6 - 0.3).to(tdt)
    bias = (torch.rand(C) * 0.2 - 0.1).to(tdt)
    x_npu = x.npu()
    w_npu = weight.npu()
    b_npu = bias.npu()
    out = depthwise_conv_2d(x_npu, w_npu, b_npu, list(k), list(s), list(p), list(d), C).cpu()
    ref = golden_depthwise_conv_2d(x, weight, bias, list(k), list(s), list(p), list(d), C)
    passed, ratio, max_abs, mere, mare = check_precision(out, ref, dtype)
    thr = _MERE_TABLE[dtype]
    mere_ok = mere < thr and mare < 10 * thr
    tag = "PASS" if (passed and mere_ok) else "FAIL"
    print(
        f"[PRECISION_{tag}] {name} shape={shape} dtype={dtype} "
        f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e} "
        f"mere={mere:.3e} mare={mare:.3e} (thr={thr:.1e})"
    )
    return passed and mere_ok


def test_depthwise_conv_2d_l0():
    failures = [name for name, shape, dtype, k, s, p, d in L0_CASES if not _run_case(name, shape, dtype, k, s, p, d)]
    assert not failures, f"L0 gate failed for: {failures}"


# ========== L1: functional (deterministic irregular shapes) ==========
# (shape, dtype, k, s, p, d, value_range, coverage tags)
L1_CASES = [
    ((1, 64, 64, 64), "float16", (3, 3), (1, 1), (1, 1), (1, 1), (-1, 1), ["D-DTYPE-fp16", "D-SHAPE-ALIGNED"]),
    ((1, 64, 64, 64), "float32", (3, 3), (1, 1), (1, 1), (1, 1), (-10, 10), ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    ((1, 64, 64, 64), "bfloat16", (3, 3), (1, 1), (1, 1), (1, 1), (-1, 1), ["D-DTYPE-bf16", "D-SHAPE-ALIGNED"]),
    ((1, 9, 32, 32), "float16", (3, 3), (1, 1), (1, 1), (1, 1), (-1, 1), ["D-SHAPE-TAIL-1"]),
    ((1, 12, 33, 33), "float16", (3, 3), (1, 1), (1, 1), (1, 1), (-1, 1), ["D-SHAPE-TAIL-MID"]),
    ((2, 7, 13, 17), "float32", (3, 3), (1, 1), (1, 1), (1, 1), (-1, 2), ["D-SHAPE-PRIME", "D-VALRANGE-ASYM"]),
    ((1, 1, 8, 8), "float16", (3, 3), (1, 1), (1, 1), (1, 1), (-1, 1), ["D-SHAPE-EDGE"]),
    ((1, 16, 32, 32), "float16", (3, 3), (1, 1), (1, 1), (1, 1), (-50, 50), ["D-VALRANGE-L"]),
    ((1, 16, 48, 48), "bfloat16", (5, 5), (2, 2), (2, 2), (1, 1), (-1, 1), ["D-PARAM-stride", "D-PARAM-kernelSize", "D-PARAM-groups"]),
    ((1, 16, 40, 40), "float32", (3, 3), (1, 1), (2, 2), (3, 3), (-1, 1), ["D-PARAM-dilation", "D-PARAM-padding"]),
    ((1, 20, 24, 24), "float16", (1, 1), (1, 1), (0, 0), (1, 1), (-1, 1), ["D-PARAM-kernelSize", "D-PARAM-groups"]),
]

# Coverage manifest (dimension -> number of L1/L2/Boundary cases covering it);
# documents the intended coverage of the functional dimensions.
COVERAGE_CATEGORY = "Vector"
COVERAGE_MANIFEST = {
    "D-DTYPE-fp16": 6,
    "D-DTYPE-fp32": 3,
    "D-DTYPE-bf16": 4,
    "D-SHAPE-ALIGNED": 3,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 1,
    "D-SHAPE-EDGE": 1,
    "D-VALRANGE-S": 5,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-stride": 1,
    "D-PARAM-kernelSize": 2,
    "D-PARAM-dilation": 1,
    "D-PARAM-padding": 1,
    "D-PARAM-groups": 2,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 2,
}
COVERAGE_NA = {}


def test_depthwise_conv_2d_l1():
    failures = []
    for shape, dtype, k, s, p, d, vr, _tags in L1_CASES:
        name = f"l1_{shape}_{dtype}_k{k[0]}s{s[0]}p{p[0]}d{d[0]}"
        if not _run_case(name, shape, dtype, k, s, p, d, vranges=vr, seed=1):
            failures.append(name)
    assert not failures, f"L1 functional failed for: {failures}"


def _run_exception(name, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(f"[L2_PASS] {name}: correctly rejected ({type(e).__name__})")
        return
    raise AssertionError(f"L2 {name}: invalid input silently accepted")


def test_depthwise_conv_2d_l2():
    tdt = torch.float16
    x = torch.rand(1, 8, 16, 16).to(tdt)
    w = torch.rand(8, 3, 3).to(tdt)
    b = torch.rand(8).to(tdt)
    _run_exception("groups_ne_C", lambda: depthwise_conv_2d(x.npu(), w.npu(), b.npu(), [3, 3], [1, 1], [1, 1], [1, 1], 4))
    _run_exception("kernel_size_mismatch", lambda: depthwise_conv_2d(x.npu(), w.npu(), b.npu(), [5, 5], [1, 1], [1, 1], [1, 1], 8))
    _run_exception("dtype_mismatch", lambda: depthwise_conv_2d(x.npu(), w.float().npu(), b.npu(), [3, 3], [1, 1], [1, 1], [1, 1], 8))
    _run_exception("x_not_4d", lambda: depthwise_conv_2d(x[0].npu(), w.npu(), b.npu(), [3, 3], [1, 1], [1, 1], [1, 1], 8))
    _run_exception(
        "non_contiguous", lambda: depthwise_conv_2d(x.permute(0, 1, 3, 2).npu(), w.npu(), b.npu(), [3, 3], [1, 1], [1, 1], [1, 1], 8)
    )
    _run_exception(
        "unsupported_dtype", lambda: depthwise_conv_2d(x.int().npu(), w.int().npu(), b.int().npu(), [3, 3], [1, 1], [1, 1], [1, 1], 8)
    )


def _run_boundary(name, dtype, fn):
    try:
        out, ref = fn()
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"boundary {name} dtype={dtype} raised: {e}") from e
    passed, ratio, max_abs, mere, mare = check_precision(out, ref, dtype)
    assert passed, f"boundary {name} dtype={dtype} failed: matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
    print(f"[BOUNDARY_PASS] boundary {name} dtype={dtype} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")


def test_depthwise_conv_2d_boundary():
    def make_special(dtype, special):
        torch.manual_seed(3)
        tdt = _TORCH_DT[dtype]
        x = (torch.rand(1, 8, 20, 20) * 2 - 1).to(tdt)
        w = (torch.rand(8, 3, 3) * 0.6 - 0.3).to(tdt)
        b = (torch.rand(8) * 0.2 - 0.1).to(tdt)
        if special == "inf":
            xf = x.float()
            xf.view(-1)[::97] = float("inf")
            xf.view(-1)[::89] = float("-inf")
            x = xf.to(tdt)
        elif special == "nan":
            xf = x.float()
            xf.view(-1)[::53] = float("nan")
            x = xf.to(tdt)
        elif special == "zero":
            x = torch.zeros_like(x)
        return x, w, b

    def dbound():
        x = torch.full((1, 8, 16, 16), 65504.0, dtype=torch.float16)
        x.view(-1)[::3] = -65504.0
        w = torch.full((8, 3, 3), 0.01, dtype=torch.float16)
        b = torch.zeros(8, dtype=torch.float16)
        out = depthwise_conv_2d(x.npu(), w.npu(), b.npu(), [3, 3], [1, 1], [1, 1], [1, 1], 8).cpu()
        ref = golden_depthwise_conv_2d(x, w, b, [3, 3], [1, 1], [1, 1], [1, 1], 8)
        return out, ref

    _run_boundary("fp16_dbound", "float16", dbound)

    for special, dt in (("inf", "float32"), ("nan", "float16"), ("zero", "bfloat16")):

        def fn(special=special, dt=dt):
            x, w, b = make_special(dt, special)
            out = depthwise_conv_2d(x.npu(), w.npu(), b.npu(), [3, 3], [1, 1], [1, 1], [1, 1], 8).cpu()
            ref = golden_depthwise_conv_2d(x, w, b, [3, 3], [1, 1], [1, 1], [1, 1], 8)
            return out, ref

        _run_boundary(special, dt, fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="all", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    if args.level in ("l0", "all"):
        test_depthwise_conv_2d_l0()
    if args.level in ("l1", "all"):
        test_depthwise_conv_2d_l1()
    if args.level in ("l2", "all"):
        test_depthwise_conv_2d_l2()
    if args.level in ("boundary", "all"):
        test_depthwise_conv_2d_boundary()

    print("Test Passed!")


if __name__ == "__main__":
    main()
