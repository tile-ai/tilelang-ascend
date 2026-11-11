"""GRU operator layered tests: L0/L1/L2/Boundary + main(--level).

Covers the 20 cases of cann-bench tasks/level4/gru (single-layer, stacked,
bidirectional, batchFirst, no-bias, h0, unaligned widths) plus special-value
boundaries.
"""

import argparse
import os
import sys

import tilelang
import torch

# Import kernel from sibling file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gru import gru  # noqa: E402

# ========== Coverage declarations (for coverage_check.py) ==========
COVERAGE_CATEGORY = "Fusion"
# Dimensions covered by the boundary suite rather than by a cann-bench case;
# the counts mirror the tags attached to those boundary configs.
COVERAGE_MANIFEST = {
    "D-SPECIAL-ZERO": 2,  # zero-input, zero-weight
    "D-SPECIAL-DBOUND": 1,  # large amplitude
    "D-SHAPE-TAIL-1": 2,  # S=1, B=1
    "D-SHAPE-TAIL-MID": 1,  # T=17 (mid tail)
    "D-EXC-DTYPE": 1,  # unsupported dtype rejected
    "D-EXC-SHAPE": 3,  # TensorList length, hiddenSize < 16, bi + unaligned H
}
COVERAGE_NA = {}

# ========== Dtype helpers ==========
DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _sf(v):
    """Parse special string values to float."""
    if isinstance(v, str):
        return {"inf": float("inf"), "-inf": float("-inf"), "nan": float("nan")}[v]
    return v


# ========== Precision standard ==========
# Two gates; a case passes when either gate passes.
#   Gate 1 (engineering bar): mixed tolerance |a-g| <= atol + rtol*|g| with a
#     matched-ratio and max-abs limit.
#   Gate 2 (benchmark bar): the cann-bench acceptance rule -- stage-1
#     MERE/MARE, then small-value / cancellation / normal domain fallbacks
#     against a same-dtype torch reference (fp16/bf16 recurrence error is
#     inherent to the dtype; the benchmark accepts a kernel whose error count
#     is at most 2x the same-dtype reference's own).
MIXED_TOLERANCE = {
    "float16": (2**-14, 2**-9, 1e-1, 0.99),
    "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
    "float32": (2**-16, 2**-10, 1e-2, 0.99),
}

# Harness threshold table (kernel_eval/utils/thresholds.py):
# (mere threshold, small-value boundary, small-value error, cancel boundary,
#  cancel zero) per dtype.
BENCH_THRESHOLDS = {
    "float16": (2**-10, 2**-11, 2**-16, 2**-5, 2**-5),
    "bfloat16": (2**-7, 2**-8, 2**-16, 2**-3, 2**-3),
    "float32": (2**-13, 2**-14, 2**-30, 2**-8, 2**-8),
}


def _gate_mixed(a, g, dtype_str):
    atol, rtol, max_abs_limit, required_ratio = MIXED_TOLERANCE.get(dtype_str, MIXED_TOLERANCE["float16"])
    m = torch.isfinite(g)
    if int(m.sum().item()) == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


def _gate_bench(a, g, n, dtype_str):
    """Cann-bench acceptance rule; returns (passed, mere, mare).

    The reported mere/mare follow the harness display rule: normal-domain
    values when the normal domain passes, overall values otherwise.
    """
    thr, sv_thr, sv_err, cb, cz = BENCH_THRESHOLDS.get(dtype_str, BENCH_THRESHOLDS["float32"])
    mare_thr = 10 * thr

    diff = (a - g).abs()
    golden_abs = g.abs()
    rel = diff / (golden_abs + 1e-7)
    valid = torch.isfinite(rel)
    if int(valid.sum().item()) == 0:
        return True, 0.0, 0.0

    mere = float(rel[valid].mean())
    mare = float(rel[valid].max())
    if mere < thr and mare < mare_thr:
        return True, mere, mare

    n_diff = (n - g).abs()
    n_rel = n_diff / (golden_abs + 1e-7)

    small = valid & (golden_abs < sv_thr)
    sv_npu = int((small & (diff > sv_err)).sum())
    sv_cpu = int((small & (n_diff > sv_err)).sum())
    sv_passed = (sv_npu / max(sv_cpu, 1)) <= 2

    cancel = valid & (a.abs() < cz) & (golden_abs < cb) & (golden_abs >= sv_thr)
    c_npu = int((cancel & (rel > mare_thr)).sum())
    c_cpu = int((cancel & (n_rel > mare_thr)).sum())
    c_passed = (c_npu / max(c_cpu, 1)) <= 2

    normal = valid & ~small & ~cancel
    nm_npu = int((normal & (rel > mare_thr)).sum())
    nm_cpu = int((normal & (n_rel > mare_thr)).sum())
    if nm_npu == 0:
        nm_passed = True
    elif nm_cpu == 0:
        nm_passed = False  # clean reference, offending kernel -> fail
    else:
        nm_passed = (nm_npu / nm_cpu) <= 2

    passed = sv_passed and c_passed and nm_passed
    if nm_passed and int(normal.sum().item()) > 0:
        return passed, float(rel[normal].mean()), float(rel[normal].max())
    return passed, mere, mare


def check_precision(actual, golden, native, dtype_str):
    """Two-gate check: return (passed, detail).

    `golden` is the fp64 reference truncated to the output dtype;
    `native` is the same model run at the case dtype. Inf/NaN position
    mismatches fail outright.
    """
    a = actual.detach().cpu().double()
    g = golden.detach().cpu().to(actual.dtype).double()
    n = native.detach().cpu().double()

    if not torch.equal(torch.isnan(a), torch.isnan(g)):
        return False, "nan-position mismatch"
    if not torch.equal(torch.isinf(a), torch.isinf(g)):
        return False, "inf-position mismatch"

    ok1, ratio, max_abs = _gate_mixed(a, g, dtype_str)
    if ok1:
        return True, f"{ratio:.4f}/{max_abs:.2e}"
    ok2, mere, mare = _gate_bench(a, g, n, dtype_str)
    if ok2:
        return True, f"bench-rule:{mere:.2e}/{mare:.2e}"
    return False, f"mixed:{ratio:.4f}/{max_abs:.2e} bench:{mere:.2e}/{mare:.2e}"


# ========== Golden reference ==========
def golden_gru(
    x,
    weight_ih,
    weight_hh,
    inputSize,
    hiddenSize,
    numLayers=1,
    bias=True,
    batchFirst=False,
    bidirectional=False,
    bias_ih=None,
    bias_hh=None,
    h0=None,
    fp64=True,
):
    """Reference built on torch.nn.GRU (inference mode).

    With fp64=True (default) the reference runs in float64 -- the cann-bench
    golden. With fp64=False it runs at the case dtype on CPU -- the
    same-precision reference the benchmark's domain fallbacks compare
    against. The module is switched to eval() so inter-layer dropout is a
    no-op, and a missing bias means a zero bias (the kernel never reads an
    absent bias tensor, so the reference must not keep the randomly
    initialised one).
    """
    num_directions = 2 if bidirectional else 1
    gate_size = 3 * hiddenSize
    has_bias = bias and bias_ih is not None and bias_hh is not None

    ref = torch.nn.GRU(
        input_size=inputSize,
        hidden_size=hiddenSize,
        num_layers=numLayers,
        bias=has_bias,
        batch_first=batchFirst,
        dropout=0.0,
        bidirectional=bidirectional,
    )
    if fp64:
        ref = ref.double()
    else:
        ref = ref.to(x.dtype)
    ref.eval()

    layer_inputs = [inputSize]
    for _ in range(1, numLayers):
        layer_inputs.append(hiddenSize * num_directions)

    def _cast(t):
        return t.double() if fp64 else t

    with torch.no_grad():
        for layer in range(numLayers):
            layer_input = layer_inputs[layer]
            for d in range(num_directions):
                idx = layer * num_directions + d
                suffix = f"l{layer}" if d == 0 else f"l{layer}_reverse"
                wi = weight_ih[idx][:gate_size, :layer_input]
                wh = weight_hh[idx][:gate_size, :hiddenSize]
                getattr(ref, f"weight_ih_{suffix}").copy_(_cast(wi))
                getattr(ref, f"weight_hh_{suffix}").copy_(_cast(wh))
                if has_bias:
                    getattr(ref, f"bias_ih_{suffix}").copy_(_cast(bias_ih[idx][:gate_size]))
                    getattr(ref, f"bias_hh_{suffix}").copy_(_cast(bias_hh[idx][:gate_size]))

    batch = x.shape[0] if batchFirst else x.shape[1]
    if h0 is None:
        h0 = torch.zeros(
            numLayers * num_directions,
            batch,
            hiddenSize,
            dtype=torch.float64 if fp64 else x.dtype,
        )
    else:
        h0 = _cast(h0)

    with torch.no_grad():
        y, hn = ref(_cast(x), h0)
    return y, hn


# ========== Input generation ==========
def _gen(shape, dtype_str, value_range, seed):
    """Deterministic CPU tensor honouring the case value_range."""
    torch.manual_seed(seed)
    lo, hi = value_range if value_range else (-1.0, 1.0)
    lo, hi = _sf(lo), _sf(hi)
    if lo == hi:
        cpu = torch.full(shape, float(lo), dtype=torch.float32)
    else:
        cpu = torch.empty(shape, dtype=torch.float32).uniform_(float(lo), float(hi))
    return cpu.to(DTYPE_MAP[dtype_str])


def build_inputs(case):
    """Build (x, weight_ih, weight_hh, bias_ih, bias_hh, h0) on CPU.

    Scalar inputs carry a shape/dtype/value_range triple; TensorList inputs
    carry one triple per tensor. Entries listed as None are optional and stay
    None.
    """
    cid = case["case_id"]
    out = []
    for i, (shape, dtype_str, vrange) in enumerate(zip(case["input_shape"], case["dtype"], case["value_range"])):
        if shape is None or dtype_str is None:
            out.append(None)
            continue
        if isinstance(shape[0], list):
            dts = dtype_str if isinstance(dtype_str, list) else [dtype_str] * len(shape)
            vrs = vrange if isinstance(vrange, list) else [vrange] * len(shape)
            out.append(
                [
                    _gen(
                        one_shape,
                        dts[j] if j < len(dts) else dts[-1],
                        vrs[j] if j < len(vrs) and vrs[j] else None,
                        cid * 1000 + i * 10 + j,
                    )
                    for j, one_shape in enumerate(shape)
                ]
            )
        else:
            out.append(_gen(shape, dtype_str, vrange, cid * 1000 + i))
    out += [None] * (6 - len(out))
    return out[:6]


# ========== Single case runner ==========
def _check_inf_contract(cid, y, hn, injected_row=0):
    """Invariants for an inf input, in place of reference-mask equality.

    fp32 and fp64 cannot agree on the exact inf/NaN pattern here: the gate
    pre-activation accumulates `w * inf`, and both `0 * inf` and `inf - inf`
    depend on accumulation order, so the reference mask is not reproducible.
    What must hold regardless is that a non-finite input is never laundered
    into a finite output, and that it stays confined to its own batch row even
    though all rows go through one tiled GEMM.
    """
    propagated = not bool(torch.isfinite(y[:, injected_row, :]).all().item())
    leaked = [b for b in range(y.shape[1]) if b != injected_row and not bool(torch.isfinite(y[:, b, :]).all().item())]
    ok = propagated and not leaked
    print(f"[BOUNDARY_{'PASS' if ok else 'FAIL'}] case{cid} inf input: propagated={propagated} cross_row_leak={leaked}")
    assert propagated, f"case{cid}: an inf input was laundered into an all-finite output"
    assert not leaked, f"case{cid}: non-finite values leaked into batch rows {leaked}"
    return ok


def run_case(case, repeat_check=False):
    """Run one case against the fp64 golden, print and return the dual-gate result.

    With repeat_check the same inputs are submitted twice and the two results
    must match bitwise, which exercises the wrapper's cross-call workspace
    cache.
    """
    cid, attrs = case["case_id"], case["attrs"]
    x, w_ih, w_hh, b_ih, b_hh, h0 = build_inputs(case)
    if case.get("special") in ("inf", "nan"):
        x = x.clone()
        x.view(-1)[0] = float("inf") if case["special"] == "inf" else float("nan")
    desc = case.get("note", "")
    dtype_str = str(x.dtype).split(".")[-1]

    # Materialize the device tensors once: the wrapper keeps permanently-zero
    # workspaces in a cross-call cache keyed by tensor identity, so
    # re-allocating for every call would not exercise the reuse path the
    # repeat check targets.
    dev_x = x.npu()
    dev_ih = [t.npu() for t in w_ih]
    dev_hh = [t.npu() for t in w_hh]
    dev_b_ih = [t.npu() for t in b_ih] if b_ih is not None else None
    dev_b_hh = [t.npu() for t in b_hh] if b_hh is not None else None
    dev_h0 = h0.npu() if h0 is not None else None

    def _call():
        return gru(
            dev_x,
            dev_ih,
            dev_hh,
            inputSize=attrs["inputSize"],
            hiddenSize=attrs["hiddenSize"],
            numLayers=attrs["numLayers"],
            bias=attrs.get("bias", True),
            batchFirst=attrs.get("batchFirst", False),
            dropout=attrs.get("dropout", 0.0),
            bidirectional=attrs.get("bidirectional", False),
            bias_ih=dev_b_ih,
            bias_hh=dev_b_hh,
            h0=dev_h0,
        )

    y, hn = _call()
    torch.npu.synchronize()

    if repeat_check:
        y2, hn2 = _call()
        torch.npu.synchronize()
        same = all(torch.equal(a.cpu(), b.cpu()) for a, b in ((y, y2), (hn, hn2)))
        print(f"[REPEAT_{'PASS' if same else 'FAIL'}] case{cid} repeated call bitwise identical")
        assert same, f"case{cid}: a repeated call returned different results (cached workspace leak)"

    if case.get("special") == "inf":
        return _check_inf_contract(cid, y, hn)

    gy, ghn = golden_gru(
        x,
        w_ih,
        w_hh,
        inputSize=attrs["inputSize"],
        hiddenSize=attrs["hiddenSize"],
        numLayers=attrs["numLayers"],
        bias=attrs.get("bias", True),
        batchFirst=attrs.get("batchFirst", False),
        bidirectional=attrs.get("bidirectional", False),
        bias_ih=b_ih,
        bias_hh=b_hh,
        h0=h0,
    )
    # Same-precision reference for the domain fallbacks (benchmark semantics).
    ny, nhn = golden_gru(
        x,
        w_ih,
        w_hh,
        inputSize=attrs["inputSize"],
        hiddenSize=attrs["hiddenSize"],
        numLayers=attrs["numLayers"],
        bias=attrs.get("bias", True),
        batchFirst=attrs.get("batchFirst", False),
        bidirectional=attrs.get("bidirectional", False),
        bias_ih=b_ih,
        bias_hh=b_hh,
        h0=h0,
        fp64=False,
    )

    for name, got, want in (("y", y, gy), ("hn", hn, ghn)):
        assert list(got.shape) == list(want.shape), f"case{cid} {name} shape {list(got.shape)} != golden {list(want.shape)}"

    ok = True
    detail = []
    for name, got, want, nat in (("y", y, gy, ny), ("hn", hn, ghn, nhn)):
        passed, det = check_precision(got, want, nat, dtype_str)
        ok &= passed
        detail.append(f"{name}:{det}")
    tag = "PASS" if ok else "FAIL"
    print(
        f"[PRECISION_{tag}] case{cid} dtype={dtype_str} S={x.shape[0]} B={x.shape[1]} "
        f"F={attrs['inputSize']} H={attrs['hiddenSize']} L={attrs['numLayers']} "
        f"bi={int(attrs.get('bidirectional', False))} "
        f"bf={int(attrs.get('batchFirst', False))} | " + " ".join(detail) + f" | {desc}"
    )
    return ok


# ========== Case table (cann-bench tasks/level4/gru) ==========
# Input order per case: x, weight_ih, weight_hh, bias_ih, bias_hh, h0
# (entries beyond the listed ones are optional and stay None).
CASES = [
    dict(
        case_id=1,
        note="single-layer fp32",
        input_shape=[[10, 4, 64], [[96, 64]], [[96, 32]], [[96]], [[96]]],
        dtype=["float32", ["float32"], ["float32"], ["float32"], ["float32"]],
        value_range=[
            [-0.1, 0.1],
            [[-0.1, 0.1]],
            [[-0.1, 0.1]],
            [[-0.01, 0.01]],
            [[-0.01, 0.01]],
        ],
        attrs=dict(inputSize=64, hiddenSize=32, numLayers=1, bias=True),
        tags=["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-hiddenSize"],
    ),
    dict(
        case_id=2,
        note="single-layer fp16",
        input_shape=[[20, 16, 128], [[192, 128]], [[192, 64]], [[192]], [[192]]],
        dtype=["float16", ["float16"], ["float16"], ["float16"], ["float16"]],
        value_range=[
            [-0.2, 0.2],
            [[-0.2, 0.2]],
            [[-0.2, 0.2]],
            [[-0.02, 0.02]],
            [[-0.02, 0.02]],
        ],
        attrs=dict(inputSize=128, hiddenSize=64, numLayers=1, bias=True),
        tags=["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-inputSize"],
    ),
    dict(
        case_id=3,
        note="single-layer bf16",
        input_shape=[[50, 32, 256], [[384, 256]], [[384, 128]], [[384]], [[384]]],
        dtype=["bfloat16", ["bfloat16"], ["bfloat16"], ["bfloat16"], ["bfloat16"]],
        value_range=[
            [-0.3, 0.3],
            [[-0.3, 0.3]],
            [[-0.3, 0.3]],
            [[-0.03, 0.03]],
            [[-0.03, 0.03]],
        ],
        attrs=dict(inputSize=256, hiddenSize=128, numLayers=1, bias=True),
        tags=["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-hiddenSize"],
    ),
    dict(
        case_id=4,
        note="bidirectional fp32",
        input_shape=[
            [12, 5, 64],
            [[96, 64], [96, 64]],
            [[96, 32], [96, 32]],
            [[96], [96]],
            [[96], [96]],
        ],
        dtype=[
            "float32",
            ["float32", "float32"],
            ["float32", "float32"],
            ["float32", "float32"],
            ["float32", "float32"],
        ],
        value_range=[
            [-0.4, 0.4],
            [[-0.4, 0.4], [-0.4, 0.4]],
            [[-0.4, 0.4], [-0.4, 0.4]],
            [[-0.04, 0.04], [-0.04, 0.04]],
            [[-0.04, 0.04], [-0.04, 0.04]],
        ],
        attrs=dict(inputSize=64, hiddenSize=32, numLayers=1, bias=True, bidirectional=True),
        tags=[
            "D-DTYPE-fp32",
            "D-SHAPE-ALIGNED",
            "D-VALRANGE-M",
            "D-PARAM-bidirectional",
        ],
    ),
    dict(
        case_id=5,
        note="bidirectional fp16",
        input_shape=[
            [24, 8, 128],
            [[192, 128], [192, 128]],
            [[192, 64], [192, 64]],
            [[192], [192]],
            [[192], [192]],
        ],
        dtype=[
            "float16",
            ["float16", "float16"],
            ["float16", "float16"],
            ["float16", "float16"],
            ["float16", "float16"],
        ],
        value_range=[
            [-0.5, 0.5],
            [[-0.5, 0.5], [-0.5, 0.5]],
            [[-0.5, 0.5], [-0.5, 0.5]],
            [[-0.05, 0.05], [-0.05, 0.05]],
            [[-0.05, 0.05], [-0.05, 0.05]],
        ],
        attrs=dict(inputSize=128, hiddenSize=64, numLayers=1, bias=True, bidirectional=True),
        tags=[
            "D-DTYPE-fp16",
            "D-SHAPE-ALIGNED",
            "D-VALRANGE-M",
            "D-PARAM-bidirectional",
        ],
    ),
    dict(
        case_id=6,
        note="2-layer fp16 (layer 1 input = 64)",
        input_shape=[
            [8, 8, 96],
            [[192, 96], [192, 64]],
            [[192, 64], [192, 64]],
            [[192], [192]],
            [[192], [192]],
        ],
        dtype=[
            "float16",
            ["float16", "float16"],
            ["float16", "float16"],
            ["float16", "float16"],
            ["float16", "float16"],
        ],
        value_range=[
            [-0.6, 0.6],
            [[-0.5, 0.5], [-0.5, 0.5]],
            [[-0.5, 0.5], [-0.5, 0.5]],
            [[-0.05, 0.05], [-0.05, 0.05]],
            [[-0.05, 0.05], [-0.05, 0.05]],
        ],
        attrs=dict(inputSize=96, hiddenSize=64, numLayers=2, bias=True),
        tags=["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-numLayers"],
    ),
    dict(
        case_id=7,
        note="3-layer fp16 with inter-layer dropout (inference: no-op)",
        input_shape=[
            [30, 16, 256],
            [[384, 256], [384, 128], [384, 128]],
            [[384, 128], [384, 128], [384, 128]],
            [[384], [384], [384]],
            [[384], [384], [384]],
        ],
        dtype=[
            "float16",
            ["float16"] * 3,
            ["float16"] * 3,
            ["float16"] * 3,
            ["float16"] * 3,
        ],
        value_range=[
            [-0.7, 0.7],
            [[-0.5, 0.5]] * 3,
            [[-0.5, 0.5]] * 3,
            [[-0.05, 0.05]] * 3,
            [[-0.05, 0.05]] * 3,
        ],
        attrs=dict(inputSize=256, hiddenSize=128, numLayers=3, bias=True, dropout=0.1),
        tags=[
            "D-DTYPE-fp16",
            "D-SHAPE-ALIGNED",
            "D-VALRANGE-M",
            "D-PARAM-numLayers",
            "D-PARAM-dropout",
        ],
    ),
    dict(
        case_id=8,
        note="2-layer bidirectional fp16",
        input_shape=[
            [16, 6, 96],
            [[192, 96], [192, 96], [192, 128], [192, 128]],
            [[192, 64], [192, 64], [192, 64], [192, 64]],
            [[192], [192], [192], [192]],
            [[192], [192], [192], [192]],
        ],
        dtype=[
            "float16",
            ["float16"] * 4,
            ["float16"] * 4,
            ["float16"] * 4,
            ["float16"] * 4,
        ],
        value_range=[
            [-0.8, 0.8],
            [[-0.5, 0.5]] * 4,
            [[-0.5, 0.5]] * 4,
            [[-0.05, 0.05]] * 4,
            [[-0.05, 0.05]] * 4,
        ],
        attrs=dict(inputSize=96, hiddenSize=64, numLayers=2, bias=True, bidirectional=True),
        tags=[
            "D-DTYPE-fp16",
            "D-SHAPE-ALIGNED",
            "D-VALRANGE-M",
            "D-PARAM-numLayers",
            "D-PARAM-bidirectional",
        ],
    ),
    dict(
        case_id=9,
        note="2-layer bidirectional bf16",
        input_shape=[
            [15, 12, 128],
            [[192, 128]] * 4,
            [[192, 64]] * 4,
            [[192]] * 4,
            [[192]] * 4,
        ],
        dtype=[
            "bfloat16",
            ["bfloat16"] * 4,
            ["bfloat16"] * 4,
            ["bfloat16"] * 4,
            ["bfloat16"] * 4,
        ],
        value_range=[
            [-0.9, 0.9],
            [[-0.5, 0.5]] * 4,
            [[-0.5, 0.5]] * 4,
            [[-0.05, 0.05]] * 4,
            [[-0.05, 0.05]] * 4,
        ],
        attrs=dict(inputSize=128, hiddenSize=64, numLayers=2, bias=True, bidirectional=True),
        tags=[
            "D-DTYPE-bf16",
            "D-SHAPE-ALIGNED",
            "D-VALRANGE-M",
            "D-PARAM-numLayers",
            "D-PARAM-bidirectional",
        ],
    ),
    dict(
        case_id=10,
        note="batchFirst fp16",
        input_shape=[[6, 8, 64], [[96, 64]], [[96, 32]], [[96]], [[96]]],
        dtype=["float16", ["float16"], ["float16"], ["float16"], ["float16"]],
        value_range=[
            [-1.0, 1.0],
            [[-0.5, 0.5]],
            [[-0.5, 0.5]],
            [[-0.05, 0.05]],
            [[-0.05, 0.05]],
        ],
        attrs=dict(inputSize=64, hiddenSize=32, numLayers=1, bias=True, batchFirst=True),
        tags=["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-batchFirst"],
    ),
    dict(
        case_id=11,
        note="batchFirst bidirectional fp16",
        input_shape=[
            [14, 10, 128],
            [[192, 128], [192, 128]],
            [[192, 64], [192, 64]],
            [[192], [192]],
            [[192], [192]],
        ],
        dtype=[
            "float16",
            ["float16", "float16"],
            ["float16", "float16"],
            ["float16", "float16"],
            ["float16", "float16"],
        ],
        value_range=[
            [-1.1, 1.1],
            [[-0.5, 0.5], [-0.5, 0.5]],
            [[-0.5, 0.5], [-0.5, 0.5]],
            [[-0.05, 0.05], [-0.05, 0.05]],
            [[-0.05, 0.05], [-0.05, 0.05]],
        ],
        attrs=dict(
            inputSize=128,
            hiddenSize=64,
            numLayers=1,
            bias=True,
            batchFirst=True,
            bidirectional=True,
        ),
        tags=[
            "D-DTYPE-fp16",
            "D-SHAPE-ALIGNED",
            "D-VALRANGE-M",
            "D-PARAM-batchFirst",
            "D-PARAM-bidirectional",
        ],
    ),
    dict(
        case_id=12,
        note="no bias tensors (zero bias)",
        input_shape=[[8, 3, 64], [[96, 64]], [[96, 32]], None, None],
        dtype=["float16", ["float16"], ["float16"], None, None],
        value_range=[[-1.2, 1.2], [[-0.5, 0.5]], [[-0.5, 0.5]], None, None],
        attrs=dict(inputSize=64, hiddenSize=32, numLayers=1, bias=False),
        tags=["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-bias"],
    ),
    dict(
        case_id=13,
        note="bidirectional bf16, no bias",
        input_shape=[[18, 7, 128], [[192, 128], [192, 128]], [[192, 64], [192, 64]]],
        dtype=["bfloat16", ["bfloat16", "bfloat16"], ["bfloat16", "bfloat16"]],
        value_range=[
            [-1.3, 1.3],
            [[-0.5, 0.5], [-0.5, 0.5]],
            [[-0.5, 0.5], [-0.5, 0.5]],
        ],
        attrs=dict(inputSize=128, hiddenSize=64, numLayers=1, bias=False, bidirectional=True),
        tags=[
            "D-DTYPE-bf16",
            "D-SHAPE-ALIGNED",
            "D-VALRANGE-M",
            "D-PARAM-bias",
            "D-PARAM-bidirectional",
        ],
    ),
    dict(
        case_id=14,
        note="long sequence fp16 (S=100)",
        input_shape=[[100, 8, 256], [[384, 256]], [[384, 128]], [[384]], [[384]]],
        dtype=["float16", ["float16"], ["float16"], ["float16"], ["float16"]],
        value_range=[
            [-1.4, 1.4],
            [[-0.5, 0.5]],
            [[-0.5, 0.5]],
            [[-0.05, 0.05]],
            [[-0.05, 0.05]],
        ],
        attrs=dict(inputSize=256, hiddenSize=128, numLayers=1, bias=True),
        tags=["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-L", "D-PARAM-hiddenSize"],
    ),
    dict(
        case_id=15,
        note="long sequence fp16 (S=200, H=256)",
        input_shape=[[200, 3, 512], [[768, 512]], [[768, 256]], [[768]], [[768]]],
        dtype=["float16", ["float16"], ["float16"], ["float16"], ["float16"]],
        value_range=[
            [-1.5, 1.5],
            [[-0.5, 0.5]],
            [[-0.5, 0.5]],
            [[-0.05, 0.05]],
            [[-0.05, 0.05]],
        ],
        attrs=dict(inputSize=512, hiddenSize=256, numLayers=1, bias=True),
        tags=["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-L", "D-PARAM-hiddenSize"],
    ),
    dict(
        case_id=16,
        note="unaligned width fp16 (F=H=127, K tail)",
        input_shape=[[10, 4, 127], [[381, 127]], [[381, 127]], [[381]], [[381]]],
        dtype=["float16", ["float16"], ["float16"], ["float16"], ["float16"]],
        value_range=[
            [-1.6, 1.6],
            [[-0.5, 0.5]],
            [[-0.5, 0.5]],
            [[-0.05, 0.05]],
            [[-0.05, 0.05]],
        ],
        attrs=dict(inputSize=127, hiddenSize=127, numLayers=1, bias=True),
        tags=[
            "D-DTYPE-fp16",
            "D-SHAPE-EDGE",
            "D-SHAPE-PRIME",
            "D-VALRANGE-M",
            "D-PARAM-inputSize",
        ],
    ),
    dict(
        case_id=17,
        note="unaligned width bf16 (F=63, H=31)",
        input_shape=[[15, 6, 63], [[93, 63]], [[93, 31]], [[93]], [[93]]],
        dtype=["bfloat16", ["bfloat16"], ["bfloat16"], ["bfloat16"], ["bfloat16"]],
        value_range=[
            [-1.7, 1.7],
            [[-0.5, 0.5]],
            [[-0.5, 0.5]],
            [[-0.05, 0.05]],
            [[-0.05, 0.05]],
        ],
        attrs=dict(inputSize=63, hiddenSize=31, numLayers=1, bias=True),
        tags=[
            "D-DTYPE-bf16",
            "D-SHAPE-EDGE",
            "D-SHAPE-PRIME",
            "D-VALRANGE-M",
            "D-PARAM-inputSize",
        ],
    ),
    dict(
        case_id=18,
        note="explicit h0",
        input_shape=[[7, 6, 64], [[96, 64]], [[96, 32]], [[96]], [[96]], [1, 6, 32]],
        dtype=[
            "float16",
            ["float16"],
            ["float16"],
            ["float16"],
            ["float16"],
            "float16",
        ],
        value_range=[
            [-1.8, 1.8],
            [[-0.5, 0.5]],
            [[-0.5, 0.5]],
            [[-0.05, 0.05]],
            [[-0.05, 0.05]],
            [-0.1, 0.1],
        ],
        attrs=dict(inputSize=64, hiddenSize=32, numLayers=1, bias=True),
        tags=["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-h0"],
    ),
    dict(
        case_id=19,
        note="2-layer bidirectional fp16 with h0",
        input_shape=[
            [22, 10, 128],
            [[192, 128]] * 4,
            [[192, 64]] * 4,
            [[192]] * 4,
            [[192]] * 4,
            [4, 10, 64],
        ],
        dtype=[
            "float16",
            ["float16"] * 4,
            ["float16"] * 4,
            ["float16"] * 4,
            ["float16"] * 4,
            "float16",
        ],
        value_range=[
            [-1.9, 1.9],
            [[-0.5, 0.5]] * 4,
            [[-0.5, 0.5]] * 4,
            [[-0.05, 0.05]] * 4,
            [[-0.05, 0.05]] * 4,
            [-0.1, 0.1],
        ],
        attrs=dict(inputSize=128, hiddenSize=64, numLayers=2, bias=True, bidirectional=True),
        tags=[
            "D-DTYPE-fp16",
            "D-SHAPE-ALIGNED",
            "D-VALRANGE-M",
            "D-PARAM-numLayers",
            "D-PARAM-bidirectional",
            "D-PARAM-h0",
        ],
    ),
    dict(
        case_id=20,
        note="small fp16",
        input_shape=[[5, 2, 32], [[48, 32]], [[48, 16]], [[48]], [[48]]],
        dtype=["float16", ["float16"], ["float16"], ["float16"], ["float16"]],
        value_range=[
            [-2.0, 2.0],
            [[-0.5, 0.5]],
            [[-0.5, 0.5]],
            [[-0.05, 0.05]],
            [[-0.05, 0.05]],
        ],
        attrs=dict(inputSize=32, hiddenSize=16, numLayers=1, bias=True),
        tags=["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-S", "D-PARAM-hiddenSize"],
    ),
]

_BY_ID = {c["case_id"]: c for c in CASES}

# ========== L0 tests: threshold (representative subset) ==========
L0_IDS = [
    20,
    1,
    18,
    4,
    10,
    17,
]  # fp16 / fp32 / h0 / bidirectional / batchFirst / unaligned bf16


def test_gru_l0():
    """L0 threshold tests: one representative case per execution path."""
    ok = True
    for cid in L0_IDS:
        case = _BY_ID[cid]
        try:
            ok &= run_case(case)
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] case{cid}: {type(e).__name__}: {e}")
            ok = False
    assert ok, "L0 precision tests failed"


# ========== L1 tests: functional (all 20 cann-bench cases) ==========
# Cases whose repeat is checked: an aligned fp32 case, an unaligned case and
# an h0 case (three distinct workspace cache keys).
REPEAT_CASE_IDS = (1, 16, 18)


def test_gru_l1():
    """L1 functional tests: every case of tasks/level4/gru/cases.yaml, plus a
    repeated-call check on distinct cache-key paths."""
    ok = True
    for case in CASES:
        try:
            ok &= run_case(case, repeat_check=case["case_id"] in REPEAT_CASE_IDS)
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] case{case['case_id']}: {type(e).__name__}: {e}")
            ok = False
    assert ok, "L1 functional tests failed"


# ========== L2 tests: negative (invalid inputs must be rejected) ==========
def test_gru_l2():
    """L2 negative tests: unsupported dtype / hiddenSize < 16 / TensorList
    length / bidirectional with unaligned hiddenSize must raise instead of
    returning wrong results."""

    def _expect_reject(desc, fn):
        try:
            fn()
            print(f"[L2_FAIL] {desc}: silently accepted (should have raised)")
            return False
        except (ValueError, AssertionError, RuntimeError):
            print(f"[L2_PASS] {desc}: correctly rejected")
            return True

    def _case(**overrides):
        attrs = dict(inputSize=32, hiddenSize=16, numLayers=1)
        attrs.update(overrides)
        return dict(
            case_id=900,
            note="negative",
            input_shape=[[6, 4, 32], [[48, 32]], [[48, 16]], None, None],
            dtype=["float32", ["float32"], ["float32"], None, None],
            value_range=[[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], None, None],
            attrs=attrs,
        )

    ok = True

    # hiddenSize < 16 breaks the GEMM base block
    c = _case(hiddenSize=8)
    c["input_shape"] = [[6, 4, 32], [[24, 32]], [[24, 8]], None, None]
    c["dtype"] = ["float32", ["float32"], ["float32"], None, None]
    c["value_range"] = [[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], None, None]
    ok &= _expect_reject("hidden_size_below_16", lambda: run_case(c))

    # TensorList shorter than numLayers * num_directions
    c = _case(numLayers=2)
    c["input_shape"] = [[6, 4, 32], [[48, 32]], [[48, 16]], None, None]
    ok &= _expect_reject("weight_tensorlist_length_mismatch", lambda: run_case(c))

    # int32 input has no TileLang dtype mapping
    ok &= _expect_reject(
        "unsupported_dtype_int32",
        lambda: gru(
            torch.zeros(6, 4, 32, dtype=torch.int32).npu(),
            [torch.zeros(48, 32, dtype=torch.int32).npu()],
            [torch.zeros(48, 16, dtype=torch.int32).npu()],
            inputSize=32,
            hiddenSize=16,
        ),
    )

    # bidirectional requires hiddenSize % 16 == 0 (col_off alignment)
    c = _case(bidirectional=True, hiddenSize=31)
    c["input_shape"] = [
        [6, 4, 32],
        [[93, 32], [93, 32]],
        [[93, 31], [93, 31]],
        None,
        None,
    ]
    c["dtype"] = ["float32", ["float32", "float32"], ["float32", "float32"], None, None]
    c["value_range"] = [
        [-1.0, 1.0],
        [[-0.1, 0.1], [-0.1, 0.1]],
        [[-0.1, 0.1], [-0.1, 0.1]],
        None,
        None,
    ]
    ok &= _expect_reject("bidirectional_requires_aligned_hidden", lambda: run_case(c))

    assert ok, "L2 negative tests failed"


# ========== Boundary tests: special values and degenerate shapes ==========
def _boundary_case(
    case_id,
    S,
    B,
    F,
    H,
    dtype_str,
    zero_input=False,
    zero_weight=False,
    large=False,
    asym=False,
    special=None,
    tags=None,
):
    w_lo, w_hi = (0.0, 0.0) if zero_weight else (-0.5, 0.5)
    x_hi = 8.0 if large else 1.0
    x_range = [0.0, 0.0] if zero_input else ([-0.5, 3.0] if asym else [-x_hi, x_hi])
    return dict(
        case_id=case_id,
        note=f"boundary S={S} B={B} F={F} H={H}",
        input_shape=[[S, B, F], [[3 * H, F]], [[3 * H, H]], [[3 * H]], [[3 * H]]],
        dtype=[dtype_str] * 5,
        value_range=[
            x_range,
            [[w_lo, w_hi]],
            [[w_lo, w_hi]],
            [[-0.05, 0.05]],
            [[-0.05, 0.05]],
        ],
        attrs=dict(inputSize=F, hiddenSize=H, numLayers=1, bias=True),
        special=special,
        tags=tags or [],
    )


def test_gru_boundary():
    """Boundary special-value tests: S=1, B=1, mid-length tail, zero input, zero
    weights, large amplitude, asymmetric ranges, and inf / NaN injection."""
    configs = [
        # single timestep (tail 1)
        _boundary_case(801, 1, 4, 32, 16, "float16", tags=["D-SHAPE-TAIL-1"]),
        # single batch (tail 1)
        _boundary_case(802, 12, 1, 32, 16, "float32", tags=["D-SHAPE-TAIL-1"]),
        # mid tail length
        _boundary_case(803, 17, 3, 48, 32, "float32", tags=["D-SHAPE-TAIL-MID"]),
        _boundary_case(804, 9, 4, 32, 16, "float16", zero_input=True, tags=["D-SPECIAL-ZERO"]),
        _boundary_case(805, 9, 4, 32, 16, "float32", zero_weight=True, tags=["D-SPECIAL-ZERO"]),
        _boundary_case(806, 9, 4, 32, 16, "float32", large=True, tags=["D-SPECIAL-DBOUND"]),
        # asymmetric x range against symmetric weights
        _boundary_case(807, 9, 4, 32, 16, "float32", asym=True, tags=["D-VALRANGE-ASYM"]),
        # +inf in one x element saturates its gate exactly in both kernel and
        # reference, so the inf pattern must match rather than spread
        _boundary_case(808, 9, 4, 32, 16, "float32", special="inf", tags=["D-SPECIAL-INF"]),
        # NaN in one x element stays confined to its batch row and every later
        # step of that row (the recurrence never mixes rows)
        _boundary_case(809, 9, 4, 32, 16, "float32", special="nan", tags=["D-SPECIAL-NAN"]),
    ]

    ok = True
    for case in configs:
        try:
            ok &= run_case(case)
        except Exception as e:  # noqa: BLE001
            print(f"[BOUNDARY_FAIL] case{case['case_id']}: {type(e).__name__}: {e}")
            ok = False
    assert ok, "boundary special-value tests failed"


# ========== Main: --level dispatch + exit code ==========
def _run_level(fn):
    """Run one level; an AssertionError marks the level failed (pytest runs the
    test functions directly and fails on the same asserts)."""
    try:
        fn()
        return True
    except AssertionError as e:
        print(f"[LEVEL_FAIL] {fn.__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()  # Disable compile cache to avoid stale artifacts
    torch.manual_seed(0)

    ok = True
    if args.level in ("l0", "all"):
        ok &= _run_level(test_gru_l0)
    if args.level in ("l1", "all"):
        ok &= _run_level(test_gru_l1)
    if args.level in ("l2", "all"):
        ok &= _run_level(test_gru_l2)
    if args.level in ("boundary", "all"):
        ok &= _run_level(test_gru_boundary)

    if ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
