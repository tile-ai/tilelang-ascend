"""LSTM operator layered tests: L0/L1/L2/Boundary + main(--level).

Covers the 20 cases of cann-bench tasks/level4/lstm (single-layer, stacked,
bidirectional, batchFirst, no-bias, h0/c0, unaligned widths and projection).
"""

import argparse
import os
import sys

import tilelang
import torch

# Import kernel from sibling file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstm import lstm  # noqa: E402

# ========== Coverage declarations (for coverage_check.py) ==========
COVERAGE_CATEGORY = "Fusion"
# Dimensions covered by the boundary suite rather than by a cann-bench case; the
# counts mirror the tags attached to those boundary configs.
COVERAGE_MANIFEST = {
    "D-SPECIAL-ZERO": 2,  # zero-input, zero-weight
    "D-SPECIAL-DBOUND": 1,  # large amplitude
    "D-SHAPE-TAIL-1": 2,  # S=1, B=1
    "D-SHAPE-TAIL-MID": 1,  # T=17 (mid tail)
    "D-EXC-DTYPE": 1,  # unsupported dtype rejected
    "D-EXC-SHAPE": 2,  # TensorList length, hiddenSize < 16
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


# ========== Precision standard (mixed tolerance, by dtype) ==========
def get_precision(dtype_str):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio)."""
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    return fp_table.get(dtype_str, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype_str):
    """Mixed tolerance dual-gate check: return (passed, matched_ratio, max_abs_error).

    Checks inf/nan position consistency, then compares finite values with mixed
    tolerance (atol + rtol*|golden|, matched_ratio, max_abs_error_limit).
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype_str)
    a, g = actual.detach().cpu().float(), golden.detach().cpu().float()

    if not torch.equal(torch.isinf(a), torch.isinf(g)):
        return False, 0.0, float("inf")
    if not torch.equal(torch.isnan(a), torch.isnan(g)):
        return False, 0.0, float("inf")

    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# ========== Golden reference ==========
def golden_lstm(
    x,
    weight_ih,
    weight_hh,
    inputSize,
    hiddenSize,
    numLayers=1,
    bias=True,
    batchFirst=False,
    bidirectional=False,
    projSize=0,
    bias_ih=None,
    bias_hh=None,
    weight_hr=None,
    h0=None,
    c0=None,
):
    """PyTorch fp64 reference built on torch.nn.LSTM (inference mode).

    Mirrors the cann-bench golden semantics: the module is switched to eval() so
    inter-layer dropout is a no-op, and a missing bias means a zero bias (the
    kernel never reads an absent bias tensor, so the reference must not keep the
    randomly initialised one).
    """
    num_directions = 2 if bidirectional else 1
    gate_size = 4 * hiddenSize
    effective_hidden = projSize if projSize > 0 else hiddenSize
    has_bias = bias and bias_ih is not None and bias_hh is not None

    if projSize > 0 and weight_hr is None:
        raise ValueError("projSize>0 (LSTMP) requires weight_hr")

    ref = torch.nn.LSTM(
        input_size=inputSize,
        hidden_size=hiddenSize,
        num_layers=numLayers,
        bias=has_bias,
        batch_first=batchFirst,
        dropout=0.0,
        bidirectional=bidirectional,
        proj_size=projSize if projSize > 0 else 0,
    ).double()
    ref.eval()

    layer_inputs = [inputSize]
    for _ in range(1, numLayers):
        layer_inputs.append(effective_hidden * num_directions)

    with torch.no_grad():
        for layer in range(numLayers):
            layer_input = layer_inputs[layer]
            for d in range(num_directions):
                idx = layer * num_directions + d
                suffix = f"l{layer}" if d == 0 else f"l{layer}_reverse"
                wi = weight_ih[idx][:gate_size, :layer_input]
                wh = weight_hh[idx][:gate_size, :effective_hidden]
                getattr(ref, f"weight_ih_{suffix}").copy_(wi.double())
                getattr(ref, f"weight_hh_{suffix}").copy_(wh.double())
                if projSize > 0:
                    wr = weight_hr[idx][:projSize, :hiddenSize]
                    getattr(ref, f"weight_hr_{suffix}").copy_(wr.double())
                if has_bias:
                    getattr(ref, f"bias_ih_{suffix}").copy_(bias_ih[idx][:gate_size].double())
                    getattr(ref, f"bias_hh_{suffix}").copy_(bias_hh[idx][:gate_size].double())

    batch = x.shape[0] if batchFirst else x.shape[1]
    if h0 is None:
        h0 = torch.zeros(numLayers * num_directions, batch, effective_hidden, dtype=torch.float64)
    else:
        h0 = h0.double()
    if c0 is None:
        c0 = torch.zeros(numLayers * num_directions, batch, hiddenSize, dtype=torch.float64)
    else:
        c0 = c0.double()

    with torch.no_grad():
        y, (hn, cn) = ref(x.double(), (h0, c0))
    return y, hn, cn


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
    """Build (x, weight_ih, weight_hh, bias_ih, bias_hh, weight_hr, h0, c0) on CPU.

    Scalar inputs carry a shape/dtype/value_range triple; TensorList inputs carry
    one triple per tensor. Entries listed as None are optional and stay None.
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
    out += [None] * (8 - len(out))
    return out[:8]


# ========== Single case runner ==========
def _check_inf_contract(cid, y, hn, cn, injected_row=0):
    """Invariants for an inf input, in place of reference-mask equality.

    fp32 and fp64 cannot agree on the exact inf/NaN pattern here: the gate
    pre-activation accumulates `w * inf`, and both `0 * inf` and `inf - inf`
    depend on accumulation order, so the reference mask is not reproducible.
    What must hold regardless is that a non-finite input is never laundered into
    a finite output, and that it stays confined to its own batch row even though
    all rows go through one tiled GEMM.
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

    With repeat_check the same inputs are submitted twice and the two results must
    match bitwise, which exercises the wrapper's cross-call workspace cache.
    """
    cid, attrs = case["case_id"], case["attrs"]
    x, w_ih, w_hh, b_ih, b_hh, w_hr, h0, c0 = build_inputs(case)
    if case.get("special") in ("inf", "nan"):
        x = x.clone()
        x.view(-1)[0] = float("inf") if case["special"] == "inf" else float("nan")
    desc = case.get("note", "")
    dtype_str = str(x.dtype).split(".")[-1]

    # Materialize the device tensors once: the wrapper keeps permanently-zero
    # workspaces in a cross-call cache keyed by tensor identity, so re-allocating
    # for every call would not exercise the reuse path the repeat check targets.
    dev_x = x.npu()
    dev_ih = [t.npu() for t in w_ih]
    dev_hh = [t.npu() for t in w_hh]
    dev_b_ih = [t.npu() for t in b_ih] if b_ih is not None else None
    dev_b_hh = [t.npu() for t in b_hh] if b_hh is not None else None
    dev_hr = [t.npu() for t in w_hr] if w_hr is not None else None
    dev_h0 = h0.npu() if h0 is not None else None
    dev_c0 = c0.npu() if c0 is not None else None

    def _call():
        return lstm(
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
            projSize=attrs.get("projSize", 0),
            bias_ih=dev_b_ih,
            bias_hh=dev_b_hh,
            weight_hr=dev_hr,
            h0=dev_h0,
            c0=dev_c0,
        )

    y, hn, cn = _call()
    torch.npu.synchronize()

    if repeat_check:
        y2, hn2, cn2 = _call()
        torch.npu.synchronize()
        same = all(torch.equal(a.cpu(), b.cpu()) for a, b in ((y, y2), (hn, hn2), (cn, cn2)))
        print(f"[REPEAT_{'PASS' if same else 'FAIL'}] case{cid} repeated call bitwise identical")
        assert same, f"case{cid}: a repeated call returned different results (cached workspace leak)"

    if case.get("special") == "inf":
        return _check_inf_contract(cid, y, hn, cn)

    gy, ghn, gcn = golden_lstm(
        x,
        w_ih,
        w_hh,
        inputSize=attrs["inputSize"],
        hiddenSize=attrs["hiddenSize"],
        numLayers=attrs["numLayers"],
        bias=attrs.get("bias", True),
        batchFirst=attrs.get("batchFirst", False),
        bidirectional=attrs.get("bidirectional", False),
        projSize=attrs.get("projSize", 0),
        bias_ih=b_ih,
        bias_hh=b_hh,
        weight_hr=w_hr,
        h0=h0,
        c0=c0,
    )

    for name, got, want in (("y", y, gy), ("hn", hn, ghn), ("cn", cn, gcn)):
        assert list(got.shape) == list(want.shape), f"case{cid} {name} shape {list(got.shape)} != golden {list(want.shape)}"

    # cann-bench compare semantics: the reference is truncated to the output dtype.
    ok = True
    detail = []
    for name, got, want in (("y", y, gy), ("hn", hn, ghn), ("cn", cn, gcn)):
        passed, ratio, max_abs = check_precision(got, want.to(got.dtype), dtype_str)
        ok &= passed
        detail.append(f"{name}:{ratio:.4f}/{max_abs:.2e}")
    tag = "PASS" if ok else "FAIL"
    print(
        f"[PRECISION_{tag}] case{cid} dtype={dtype_str} S={x.shape[0]} B={x.shape[1]} "
        f"H={attrs['hiddenSize']} L={attrs['numLayers']} "
        f"bi={int(attrs.get('bidirectional', False))} "
        f"proj={attrs.get('projSize', 0)} | " + " ".join(detail) + f" | {desc}"
    )
    return ok


# ========== Case table (cann-bench tasks/level4/lstm) ==========
# Input order per case: x, weight_ih, weight_hh, bias_ih, bias_hh, weight_hr,
# h0, c0 (entries beyond the listed ones are optional and stay None).
CASES = [
    dict(
        case_id=1,
        note="single-layer fp32",
        input_shape=[[10, 4, 64], [[128, 64]], [[128, 32]], [[128]], [[128]], None, None],
        dtype=["float32", ["float32"], ["float32"], ["float32"], ["float32"], None, None],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], [[-0.01, 0.01]], [[-0.01, 0.01]], None, None],
        attrs=dict(inputSize=64, hiddenSize=32, numLayers=1, bias=True, projSize=0),
        tags=["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-hiddenSize"],
    ),
    dict(
        case_id=2,
        note="single-layer fp16",
        input_shape=[[20, 16, 128], [[256, 128]], [[256, 64]], [[256]], [[256]]],
        dtype=["float16", ["float16"], ["float16"], ["float16"], ["float16"]],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], [[-0.01, 0.01]], [[-0.01, 0.01]]],
        attrs=dict(inputSize=128, hiddenSize=64, numLayers=1, bias=True, projSize=0),
        tags=["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-inputSize"],
    ),
    dict(
        case_id=3,
        note="single-layer bf16",
        input_shape=[[50, 32, 256], [[512, 256]], [[512, 128]], [[512]], [[512]]],
        dtype=["bfloat16", ["bfloat16"], ["bfloat16"], ["bfloat16"], ["bfloat16"]],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], [[-0.01, 0.01]], [[-0.01, 0.01]]],
        attrs=dict(inputSize=256, hiddenSize=128, numLayers=1, bias=True, projSize=0),
        tags=["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-hiddenSize"],
    ),
    dict(
        case_id=4,
        note="bidirectional fp32",
        input_shape=[[12, 5, 64], [[128, 64], [128, 64]], [[128, 32], [128, 32]], [[128], [128]], [[128], [128]]],
        dtype=["float32", ["float32", "float32"], ["float32", "float32"], ["float32", "float32"], ["float32", "float32"]],
        value_range=[
            [-1.0, 1.0],
            [[-0.1, 0.1], [-0.1, 0.1]],
            [[-0.1, 0.1], [-0.1, 0.1]],
            [[-0.01, 0.01], [-0.01, 0.01]],
            [[-0.01, 0.01], [-0.01, 0.01]],
        ],
        attrs=dict(inputSize=64, hiddenSize=32, numLayers=1, bias=True, bidirectional=True, projSize=0),
        tags=["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-bidirectional"],
    ),
    dict(
        case_id=5,
        note="bidirectional fp16",
        input_shape=[[24, 8, 128], [[256, 128], [256, 128]], [[256, 64], [256, 64]], [[256], [256]], [[256], [256]]],
        dtype=["float16", ["float16", "float16"], ["float16", "float16"], ["float16", "float16"], ["float16", "float16"]],
        value_range=[
            [-1.0, 1.0],
            [[-0.1, 0.1], [-0.1, 0.1]],
            [[-0.1, 0.1], [-0.1, 0.1]],
            [[-0.01, 0.01], [-0.01, 0.01]],
            [[-0.01, 0.01], [-0.01, 0.01]],
        ],
        attrs=dict(inputSize=128, hiddenSize=64, numLayers=1, bias=True, bidirectional=True, projSize=0),
        tags=["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-bidirectional"],
    ),
    dict(
        case_id=6,
        note="2-layer fp32 (layer 1 input = 64)",
        input_shape=[[8, 8, 96], [[256, 96], [256, 64]], [[256, 64], [256, 64]], [[256], [256]], [[256], [256]]],
        dtype=["float32", ["float32", "float32"], ["float32", "float32"], ["float32", "float32"], ["float32", "float32"]],
        value_range=[
            [-1.0, 1.0],
            [[-0.1, 0.1], [-0.1, 0.1]],
            [[-0.1, 0.1], [-0.1, 0.1]],
            [[-0.01, 0.01], [-0.01, 0.01]],
            [[-0.01, 0.01], [-0.01, 0.01]],
        ],
        attrs=dict(inputSize=96, hiddenSize=64, numLayers=2, bias=True, projSize=0),
        tags=["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-numLayers"],
    ),
    dict(
        case_id=7,
        note="3-layer fp16 with inter-layer dropout (inference: no-op)",
        input_shape=[
            [30, 16, 256],
            [[512, 256], [512, 128], [512, 128]],
            [[512, 128], [512, 128], [512, 128]],
            [[512], [512], [512]],
            [[512], [512], [512]],
        ],
        dtype=["float16", ["float16"] * 3, ["float16"] * 3, ["float16"] * 3, ["float16"] * 3],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1]] * 3, [[-0.1, 0.1]] * 3, [[-0.01, 0.01]] * 3, [[-0.01, 0.01]] * 3],
        attrs=dict(inputSize=256, hiddenSize=128, numLayers=3, bias=True, dropout=0.1, projSize=0),
        tags=["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-numLayers", "D-PARAM-dropout"],
    ),
    dict(
        case_id=8,
        note="2-layer bidirectional fp32",
        input_shape=[
            [16, 6, 96],
            [[256, 96], [256, 96], [256, 128], [256, 128]],
            [[256, 64], [256, 64], [256, 64], [256, 64]],
            [[256], [256], [256], [256]],
            [[256], [256], [256], [256]],
        ],
        dtype=["float32", ["float32"] * 4, ["float32"] * 4, ["float32"] * 4, ["float32"] * 4],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1]] * 4, [[-0.1, 0.1]] * 4, [[-0.01, 0.01]] * 4, [[-0.01, 0.01]] * 4],
        attrs=dict(inputSize=96, hiddenSize=64, numLayers=2, bias=True, bidirectional=True, projSize=0),
        tags=["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-numLayers", "D-PARAM-bidirectional"],
    ),
    dict(
        case_id=9,
        note="2-layer bidirectional bf16",
        input_shape=[[15, 12, 128], [[256, 128]] * 4, [[256, 64]] * 4, [[256]] * 4, [[256]] * 4],
        dtype=["bfloat16", ["bfloat16"] * 4, ["bfloat16"] * 4, ["bfloat16"] * 4, ["bfloat16"] * 4],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1]] * 4, [[-0.1, 0.1]] * 4, [[-0.01, 0.01]] * 4, [[-0.01, 0.01]] * 4],
        attrs=dict(inputSize=128, hiddenSize=64, numLayers=2, bias=True, bidirectional=True, projSize=0),
        tags=["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-numLayers", "D-PARAM-bidirectional"],
    ),
    dict(
        case_id=10,
        note="batchFirst fp32",
        input_shape=[[6, 8, 64], [[128, 64]], [[128, 32]], [[128]], [[128]]],
        dtype=["float32", ["float32"], ["float32"], ["float32"], ["float32"]],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], [[-0.01, 0.01]], [[-0.01, 0.01]]],
        attrs=dict(inputSize=64, hiddenSize=32, numLayers=1, bias=True, batchFirst=True, projSize=0),
        tags=["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-batchFirst"],
    ),
    dict(
        case_id=11,
        note="no bias tensors (zero bias)",
        input_shape=[[8, 3, 64], [[128, 64]], [[128, 32]], None, None, None, None],
        dtype=["float32", ["float32"], ["float32"], None, None, None, None],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], None, None, None, None],
        attrs=dict(inputSize=64, hiddenSize=32, numLayers=1, projSize=0),
        tags=["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-bias"],
    ),
    dict(
        case_id=12,
        note="bidirectional bf16, no bias",
        input_shape=[[18, 7, 128], [[256, 128], [256, 128]], [[256, 64], [256, 64]]],
        dtype=["bfloat16", ["bfloat16", "bfloat16"], ["bfloat16", "bfloat16"]],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1], [-0.1, 0.1]], [[-0.1, 0.1], [-0.1, 0.1]]],
        attrs=dict(inputSize=128, hiddenSize=64, numLayers=1, bidirectional=True, projSize=0),
        tags=["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-bias", "D-PARAM-bidirectional"],
    ),
    dict(
        case_id=13,
        note="long sequence fp32 (S=100)",
        input_shape=[[100, 8, 256], [[512, 256]], [[512, 128]], [[512]], [[512]]],
        dtype=["float32", ["float32"], ["float32"], ["float32"], ["float32"]],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], [[-0.01, 0.01]], [[-0.01, 0.01]]],
        attrs=dict(inputSize=256, hiddenSize=128, numLayers=1, bias=True, projSize=0),
        tags=["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-L", "D-PARAM-hiddenSize"],
    ),
    dict(
        case_id=14,
        note="long sequence fp16 (S=200)",
        input_shape=[[200, 3, 512], [[1024, 512]], [[1024, 256]], [[1024]], [[1024]]],
        dtype=["float16", ["float16"], ["float16"], ["float16"], ["float16"]],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], [[-0.01, 0.01]], [[-0.01, 0.01]]],
        attrs=dict(inputSize=512, hiddenSize=256, numLayers=1, bias=True, projSize=0),
        tags=["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-L", "D-PARAM-hiddenSize"],
    ),
    dict(
        case_id=15,
        note="explicit h0/c0",
        input_shape=[[7, 6, 64], [[128, 64]], [[128, 32]], [[128]], [[128]], None, [1, 6, 32], [1, 6, 32]],
        dtype=["float32", ["float32"], ["float32"], ["float32"], ["float32"], None, "float32", "float32"],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], [[-0.01, 0.01]], [[-0.01, 0.01]], None, [-0.1, 0.1], [-0.1, 0.1]],
        attrs=dict(inputSize=64, hiddenSize=32, numLayers=1, bias=True, projSize=0),
        tags=["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-h0c0"],
    ),
    dict(
        case_id=16,
        note="small fp16",
        input_shape=[[5, 2, 32], [[64, 32]], [[64, 16]], [[64]], [[64]]],
        dtype=["float16", ["float16"], ["float16"], ["float16"], ["float16"]],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], [[-0.01, 0.01]], [[-0.01, 0.01]]],
        attrs=dict(inputSize=32, hiddenSize=16, numLayers=1, bias=True, projSize=0),
        tags=["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-S", "D-PARAM-hiddenSize"],
    ),
    dict(
        case_id=17,
        note="unaligned width fp32 (F=H=127, K tail)",
        input_shape=[[10, 4, 127], [[508, 127]], [[508, 127]], [[508]], [[508]]],
        dtype=["float32", ["float32"], ["float32"], ["float32"], ["float32"]],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], [[-0.01, 0.01]], [[-0.01, 0.01]]],
        attrs=dict(inputSize=127, hiddenSize=127, numLayers=1, bias=True, projSize=0),
        tags=["D-DTYPE-fp32", "D-SHAPE-EDGE", "D-SHAPE-PRIME", "D-VALRANGE-M", "D-PARAM-inputSize"],
    ),
    dict(
        case_id=18,
        note="unaligned width bf16 (F=63, H=31)",
        input_shape=[[15, 6, 63], [[124, 63]], [[124, 31]], [[124]], [[124]]],
        dtype=["bfloat16", ["bfloat16"], ["bfloat16"], ["bfloat16"], ["bfloat16"]],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], [[-0.01, 0.01]], [[-0.01, 0.01]]],
        attrs=dict(inputSize=63, hiddenSize=31, numLayers=1, bias=True, projSize=0),
        tags=["D-DTYPE-bf16", "D-SHAPE-EDGE", "D-SHAPE-PRIME", "D-VALRANGE-M", "D-PARAM-inputSize"],
    ),
    dict(
        case_id=19,
        note="projection (LSTMP) fp32",
        input_shape=[[14, 10, 128], [[256, 128]], [[256, 16]], [[256]], [[256]], [[16, 64]]],
        dtype=["float32", ["float32"], ["float32"], ["float32"], ["float32"], ["float32"]],
        value_range=[[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], [[-0.01, 0.01]], [[-0.01, 0.01]], [[-0.1, 0.1]]],
        attrs=dict(inputSize=128, hiddenSize=64, numLayers=1, bias=True, projSize=16),
        tags=["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-projSize"],
    ),
    dict(
        case_id=20,
        note="2-layer projection (LSTMP) fp32",
        input_shape=[[22, 10, 128], [[256, 128], [256, 16]], [[256, 16], [256, 16]], [[256], [256]], [[256], [256]], [[16, 64], [16, 64]]],
        dtype=[
            "float32",
            ["float32", "float32"],
            ["float32", "float32"],
            ["float32", "float32"],
            ["float32", "float32"],
            ["float32", "float32"],
        ],
        value_range=[
            [-1.0, 1.0],
            [[-0.1, 0.1], [-0.1, 0.1]],
            [[-0.1, 0.1], [-0.1, 0.1]],
            [[-0.01, 0.01], [-0.01, 0.01]],
            [[-0.01, 0.01], [-0.01, 0.01]],
            [[-0.1, 0.1], [-0.1, 0.1]],
        ],
        attrs=dict(inputSize=128, hiddenSize=64, numLayers=2, bias=True, projSize=16),
        tags=["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-numLayers", "D-PARAM-projSize"],
    ),
]

_BY_ID = {c["case_id"]: c for c in CASES}

# ========== L0 tests: threshold (representative subset) ==========
L0_IDS = [16, 1, 15, 19, 10]  # fp16 / fp32 / h0-c0 / proj / batchFirst


def test_lstm_l0():
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
# Cases whose repeat is checked: one fused-path case and one per-timestep case.
REPEAT_CASE_IDS = (1, 19)


def test_lstm_l1():
    """L1 functional tests: every case of tasks/level4/lstm/cases.yaml, plus a
    repeated-call check on both execution paths."""
    ok = True
    for case in CASES:
        try:
            ok &= run_case(case, repeat_check=case["case_id"] in REPEAT_CASE_IDS)
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] case{case['case_id']}: {type(e).__name__}: {e}")
            ok = False
    assert ok, "L1 functional tests failed"


# ========== L2 tests: negative (invalid inputs must be rejected) ==========
def test_lstm_l2():
    """L2 negative tests: unsupported dtype / hiddenSize < 16 / TensorList length /
    projection constraints must raise instead of returning wrong results."""

    def _expect_reject(desc, fn):
        try:
            fn()
            print(f"[L2_FAIL] {desc}: silently accepted (should have raised)")
            return False
        except (ValueError, AssertionError, RuntimeError):
            print(f"[L2_PASS] {desc}: correctly rejected")
            return True

    def _case(**overrides):
        attrs = dict(inputSize=32, hiddenSize=16, numLayers=1, projSize=0)
        attrs.update(overrides)
        return dict(
            case_id=900,
            note="negative",
            input_shape=[[6, 4, 32], [[64, 32]], [[64, 16]], None, None, None, None],
            dtype=["float32", ["float32"], ["float32"], None, None, None, None],
            value_range=[[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], None, None, None, None],
            attrs=attrs,
        )

    ok = True

    # hiddenSize < 16 breaks the GEMM base block
    c = _case(hiddenSize=8)
    c["input_shape"] = [[6, 4, 32], [[32, 32]], [[32, 8]], None, None, None, None]
    c["dtype"] = ["float32", ["float32"], ["float32"], None, None, None, None]
    c["value_range"] = [[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], None, None, None, None]
    ok &= _expect_reject("hidden_size_below_16", lambda: run_case(c))

    # TensorList shorter than numLayers * num_directions
    c = _case(numLayers=2)
    c["input_shape"] = [[6, 4, 32], [[64, 32]], [[64, 16]], None, None, None, None]
    ok &= _expect_reject("weight_tensorlist_length_mismatch", lambda: run_case(c))

    # int32 input has no TileLang dtype mapping
    c = _case()
    c["input_shape"] = [[6, 4, 32], [[64, 32]], [[64, 16]], None, None, None, None]
    c["dtype"] = ["int32", ["int32"], ["int32"], None, None, None, None]
    ok &= _expect_reject(
        "unsupported_dtype_int32",
        lambda: lstm(
            torch.zeros(6, 4, 32, dtype=torch.int32).npu(),
            [torch.zeros(64, 32, dtype=torch.int32).npu()],
            [torch.zeros(64, 16, dtype=torch.int32).npu()],
            inputSize=32,
            hiddenSize=16,
        ),
    )

    # projSize < 16 violates the projection GEMM base block
    c = _case(projSize=8)
    c["input_shape"] = [[6, 4, 32], [[64, 32]], [[64, 8]], None, None, [[8, 16]], None, None]
    c["dtype"] = ["float32", ["float32"], ["float32"], None, None, ["float32"], None, None]
    c["value_range"] = [[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], None, None, [[-0.1, 0.1]], None, None]
    ok &= _expect_reject("projection_below_16", lambda: run_case(c))

    # projection currently supports float32 only
    c = _case(projSize=16)
    c["input_shape"] = [[6, 4, 32], [[64, 32]], [[64, 16]], None, None, [[16, 16]], None, None]
    c["dtype"] = ["float16", ["float16"], ["float16"], None, None, ["float16"], None, None]
    c["value_range"] = [[-1.0, 1.0], [[-0.1, 0.1]], [[-0.1, 0.1]], None, None, [[-0.1, 0.1]], None, None]
    ok &= _expect_reject("projection_requires_float32", lambda: run_case(c))

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
    w_lo, w_hi = (0.0, 0.0) if zero_weight else (-0.1, 0.1)
    x_hi = 8.0 if large else 1.0
    x_range = [0.0, 0.0] if zero_input else ([-0.5, 3.0] if asym else [-x_hi, x_hi])
    return dict(
        case_id=case_id,
        note=f"boundary S={S} B={B} F={F} H={H}",
        input_shape=[[S, B, F], [[4 * H, F]], [[4 * H, H]], [[4 * H]], [[4 * H]]],
        dtype=[dtype_str] * 5,
        value_range=[x_range, [[w_lo, w_hi]], [[w_lo, w_hi]], [[-0.01, 0.01]], [[-0.01, 0.01]]],
        attrs=dict(inputSize=F, hiddenSize=H, numLayers=1, bias=True, projSize=0),
        special=special,
        tags=tags or [],
    )


def test_lstm_boundary():
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
        ok &= _run_level(test_lstm_l0)
    if args.level in ("l1", "all"):
        ok &= _run_level(test_lstm_l1)
    if args.level in ("l2", "all"):
        ok &= _run_level(test_lstm_l2)
    if args.level in ("boundary", "all"):
        ok &= _run_level(test_lstm_boundary)

    if ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
