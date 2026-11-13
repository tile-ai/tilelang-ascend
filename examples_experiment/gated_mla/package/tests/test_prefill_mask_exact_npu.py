"""Analytic and self-consistency oracles for the C4 prefill attention core.

The faults these exist to catch are a column off by one, a reversed bit order
inside the packed mask, and a tile that writes over a neighbouring batch. Each
produces a small, plausible-looking difference that the operator's
rtol/atol/normalized budget would admit, so none of these assertions uses it.

Two kinds of oracle appear here, and they are deliberately not held to the same
standard.

The analytic one fixes QL = QA = 0, so every visible score is zero and after the
online maximum is subtracted the probability of each visible key is exp(0). With
C[k, :] = k the PV accumulation is sum(0..n-1) = n(n-1)/2 and the divide by the
normaliser yields (n-1)/2, where n is the number of keys visible to that query.
Every such value is a multiple of 0.5, and a mask that admits one key too many or
too few moves a whole row by exactly 0.5, so the primary assertion rounds to the
nearest half and compares exactly: there is no tolerance to argue about. In
exact arithmetic the ratio is an integer one below 2**24, but on device it passes
through the vector exponential and an FP32 divide; measured across both backends,
both dtypes, both mask paths and every window below, the worst deviation is
1.02 ULP of FP32, i.e. 3.8e-06 against the 0.5 that a mask error produces. A
second assertion bounds that drift at 8 ULP so the rounding cannot hide it.

The self-consistency ones compare two routes through the same arithmetic, so
they are held to bit-identity with no tolerance at all.

Shapes here never force block_n. `public_prefill_block_n` is the host selector
the public path uses, and it returns 128 for exactly one profile
(batch 1, Tq 128, 96 heads, Tk 128) and 64 everywhere else. Forcing the other
combination would compile a kernel no public call can reach, and would ask the
cold-compile plan to prepare a factory production never builds, so the two mask
paths are reached by shape instead. The paired batch cases use Tq 64 so that the
two-batch arm and its single-batch control select the same block_n; at Tq 128
they would not, and bit-identity would fail for a legitimate reason.

Lane split: wall clock on this board is a linear function of the number of
distinct shapes, not of the number of cases, because a new shape costs seconds of
JIT while a repeat costs a fraction of one. The core lane therefore keeps one
representative per property, chosen so that only the paired-batch shape and its
control are new to the cold-compile plan; every other core case is a metadata
variation of a profile the suite already prepares. The remaining dtype, backend
and window combinations assert the same properties and carry `low_priority` for
the daily full run.
"""

import pytest
import torch

from gmla._contracts import public_prefill_block_n
from gmla.kernels.prefill import DEFAULT_SCALE, build_cube_attention, cube_prefill

HEADS = 96
WIDTH = 512
AUX = 64
TILE = 32
# Eight ULP of FP32 at the oracle magnitude; the measured worst case is 1.02.
DRIFT = 2.0**-21

# batch, q_len, kv_len. WHOLE_TILE is the only profile the selector gives 128.
WHOLE_TILE = (1, 128, 128)
ROW_WISE = (1, 64, 128)
PAIRED = (2, 64, 128)

SHAPE_LABEL = {WHOLE_TILE: "whole_tile_mask", ROW_WISE: "row_wise_mask", PAIRED: "paired"}


def _label(value):
    if value in SHAPE_LABEL:
        return SHAPE_LABEL[value]
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    return str(value)


def lanes(combinations, core):
    """One representative per property in the core lane; the rest run daily."""
    assert core in combinations, core
    cases = []
    for combination in combinations:
        marks = () if combination == core else (pytest.mark.low_priority,)
        values = combination if isinstance(combination, tuple) else (combination,)
        cases.append(pytest.param(*values, marks=marks, id="-".join(_label(value) for value in values)))
    return cases


def selected_block_n(batch, q_len, kv_len):
    return public_prefill_block_n(batch, q_len, HEADS, kv_len, DEFAULT_SCALE)


def _zero_query(device, dtype, tq, batch=1):
    ql = torch.zeros(batch, tq, HEADS, WIDTH, dtype=dtype, device=device)
    qa = torch.zeros(batch, tq, HEADS, AUX, dtype=dtype, device=device)
    residual = torch.zeros_like(ql) if dtype is torch.bfloat16 else None
    return ql, qa, residual


def _ramp_cache(device, dtype, tk, batch=1):
    ramp = torch.arange(tk, dtype=torch.float32).view(1, tk, 1)
    c = ramp.expand(batch, tk, WIDTH).to(dtype).to(device).contiguous()
    e = torch.zeros(batch, tk, AUX, dtype=dtype, device=device)
    return c, e


def _run(ql, qa, c, e, metadata, target, residual):
    batch, q_len = ql.shape[0], ql.shape[1]
    out = cube_prefill(
        ql,
        qa,
        c,
        e,
        metadata,
        block_m=TILE,
        block_n=selected_block_n(batch, q_len, c.shape[1]),
        target=target,
        max_blocks=20,
        ql_residual=residual,
    )
    # Every assertion below reads the result, so settle the launch here rather
    # than relying on whatever the comparison happens to synchronise.
    torch.npu.synchronize()
    return out


def _assert_matches_quantised_oracle(got, want):
    got, want = got.float(), want.float()
    assert torch.equal(torch.round(got * 2) / 2, want), "oracle value moved by at least a quarter"
    drift = (got - want).abs().max().item()
    assert drift <= want.abs().max().item() * DRIFT, drift


def _window_oracle(out, visible):
    _assert_matches_quantised_oracle(out, ((visible - 1) / 2).view(1, -1, 1, 1).expand_as(out))


WINDOW_CASES = lanes(
    [
        (target, dtype, shape)
        for target in ("pto", "ascendc")
        for dtype in (torch.float16, torch.bfloat16)
        for shape in (WHOLE_TILE, ROW_WISE)
    ],
    core=("pto", torch.float16, WHOLE_TILE),
)


@pytest.mark.npu
@pytest.mark.parametrize("target,dtype,shape", WINDOW_CASES)
def test_causal_window_matches_the_analytic_value(npu, target, dtype, shape):
    """Row q must average exactly the first q + 1 cache rows, on both mask paths."""
    device = "npu:0"
    _, tq, tk = shape
    ql, qa, residual = _zero_query(device, dtype, tq)
    c, e = _ramp_cache(device, dtype, tk)
    metadata = torch.tensor([[tq, tk, 0, 0, 0, 0, 0, 0]], dtype=torch.int32, device=device)
    out = _run(ql, qa, c, e, metadata, target, residual)
    visible = torch.clamp(torch.arange(1, tq + 1, dtype=torch.float32, device=device), max=tk)
    _window_oracle(out, visible)


START_CASES = lanes([(target, start) for target in ("pto", "ascendc") for start in (1, 7, 64)], core=("pto", 7))


@pytest.mark.npu
@pytest.mark.parametrize("target,start", START_CASES)
def test_query_start_shifts_the_window(npu, target, start):
    """query_start moves every window by that many keys, then clamps at kv_len."""
    device = "npu:0"
    _, tq, tk = WHOLE_TILE
    ql, qa, residual = _zero_query(device, torch.float16, tq)
    c, e = _ramp_cache(device, torch.float16, tk)
    metadata = torch.tensor([[tq, tk, start, 0, 0, 0, 0, 0]], dtype=torch.int32, device=device)
    out = _run(ql, qa, c, e, metadata, target, residual)
    visible = torch.clamp(torch.arange(1 + start, tq + 1 + start, dtype=torch.float32, device=device), max=tk)
    _window_oracle(out, visible)


KV_CASES = lanes([(target, kv_len) for target in ("pto", "ascendc") for kv_len in (1, 40, 127)], core=("pto", 40))


@pytest.mark.npu
@pytest.mark.parametrize("target,kv_len", KV_CASES)
def test_kv_length_clamps_the_window(npu, target, kv_len):
    """A short valid cache clamps every window, including rows past its end."""
    device = "npu:0"
    _, tq, tk = WHOLE_TILE
    ql, qa, residual = _zero_query(device, torch.float16, tq)
    c, e = _ramp_cache(device, torch.float16, tk)
    metadata = torch.tensor([[tq, kv_len, 0, 0, 0, 0, 0, 0]], dtype=torch.int32, device=device)
    out = _run(ql, qa, c, e, metadata, target, residual)
    visible = torch.clamp(torch.arange(1, tq + 1, dtype=torch.float32, device=device), max=kv_len)
    _window_oracle(out, visible)


@pytest.mark.npu
@pytest.mark.parametrize("target", lanes(["pto", "ascendc"], core="pto"))
def test_inactive_batch_writes_exact_zeros(npu, target):
    """A zero-length query leaves exact zeros over its physical rows."""
    device = "npu:0"
    _, tq, tk = WHOLE_TILE
    ql, qa, residual = _zero_query(device, torch.float16, tq)
    c, e = _ramp_cache(device, torch.float16, tk)
    metadata = torch.tensor([[0, tk, 0, 0, 0, 0, 0, 0]], dtype=torch.int32, device=device)
    out = _run(ql, qa, c, e, metadata, target, residual)
    assert torch.equal(out, torch.zeros_like(out))


PAIR_CASES = lanes(
    [(target, dtype) for target in ("pto", "ascendc") for dtype in (torch.float16, torch.bfloat16)],
    core=("pto", torch.float16),
)


@pytest.mark.npu
@pytest.mark.parametrize("target,dtype", PAIR_CASES)
def test_batches_do_not_influence_each_other(npu, target, dtype):
    """A batch computed beside another must be bit-identical to computing it alone.

    Exact, not budgeted: a tile that writes past its own rows into a neighbour
    produces a bounded, plausible difference that a tolerance would admit.

    Both arms select the same block_n at this Tq, so the two routes run the same
    arithmetic in the same order and bit-identity is the right bar.
    """
    device = "npu:0"
    _, tq, tk = PAIRED
    assert selected_block_n(2, tq, tk) == selected_block_n(1, tq, tk)
    torch.manual_seed(0)
    pair_ql = (torch.randn(2, tq, HEADS, WIDTH) * 0.1).to(dtype).to(device)
    pair_qa = (torch.randn(2, tq, HEADS, AUX) * 0.1).to(dtype).to(device)
    pair_c = (torch.randn(2, tk, WIDTH) * 0.1).to(dtype).to(device)
    pair_e = (torch.randn(2, tk, AUX) * 0.1).to(dtype).to(device)
    pair_res = torch.zeros_like(pair_ql) if dtype is torch.bfloat16 else None
    meta_pair = torch.tensor([[tq, tk, 0, 0, 0, 0, 0, 0]] * 2, dtype=torch.int32, device=device)
    together = _run(pair_ql, pair_qa, pair_c, pair_e, meta_pair, target, pair_res)

    meta_one = torch.tensor([[tq, tk, 0, 0, 0, 0, 0, 0]], dtype=torch.int32, device=device)
    for index in (0, 1):
        alone = _run(
            pair_ql[index : index + 1].contiguous(),
            pair_qa[index : index + 1].contiguous(),
            pair_c[index : index + 1].contiguous(),
            pair_e[index : index + 1].contiguous(),
            meta_one,
            target,
            None if pair_res is None else pair_res[index : index + 1].contiguous(),
        )
        assert torch.equal(together[index : index + 1], alone)


@pytest.mark.npu
@pytest.mark.parametrize("target", lanes(["pto", "ascendc"], core="pto"))
def test_an_inactive_neighbour_does_not_disturb_an_active_batch(npu, target):
    """An empty batch beside a live one must not change the live result."""
    device = "npu:0"
    _, tq, tk = PAIRED
    torch.manual_seed(17)
    pair_ql = (torch.randn(2, tq, HEADS, WIDTH) * 0.1).half().to(device)
    pair_qa = (torch.randn(2, tq, HEADS, AUX) * 0.1).half().to(device)
    pair_c = (torch.randn(2, tk, WIDTH) * 0.1).half().to(device)
    pair_e = (torch.randn(2, tk, AUX) * 0.1).half().to(device)
    meta_mixed = torch.tensor([[tq, tk, 0, 0, 0, 0, 0, 0], [0, tk, 0, 0, 0, 0, 0, 0]], dtype=torch.int32, device=device)
    mixed = _run(pair_ql, pair_qa, pair_c, pair_e, meta_mixed, target, None)

    meta_one = torch.tensor([[tq, tk, 0, 0, 0, 0, 0, 0]], dtype=torch.int32, device=device)
    alone = _run(
        pair_ql[0:1].contiguous(),
        pair_qa[0:1].contiguous(),
        pair_c[0:1].contiguous(),
        pair_e[0:1].contiguous(),
        meta_one,
        target,
        None,
    )
    assert torch.equal(mixed[0:1], alone)
    assert torch.equal(mixed[1:2], torch.zeros_like(mixed[1:2]))


def _entry_source(batch, query_capacity, cache_capacity):
    kernel = build_cube_attention(
        batch=batch,
        heads=HEADS,
        cache_capacity=cache_capacity,
        query_capacity=query_capacity,
        block_m=TILE,
        block_n=selected_block_n(batch, query_capacity, cache_capacity),
        dtype="float16",
        target="pto",
        has_ql_residual=False,
        max_blocks=20,
    )
    for attribute in ("get_kernel_source", "kernel_source"):
        value = getattr(kernel, attribute, None)
        if callable(value):
            return value()
        if isinstance(value, str):
            return value
    raise AssertionError("the compiled kernel exposes no generated source to inspect")


@pytest.mark.npu
def test_the_pipelined_entry_is_compiled_only_inside_its_documented_domain(npu):
    """A failure here means the documented domain changed, not that a value drifted.

    The pipelined entry issues the QK of one wave before the PV of the previous
    one. C and E are L1-resident under a batch tag, so across a batch change the
    earlier QK would reload them into the buffers the pending PV still reads.
    Lifting the batch bound needs a second C/E slot, about 144 KiB more L1, so
    this asserts the bound still holds rather than trusting the comment.

    Every shape below is one another selected test already compiles, so this adds
    no factory to the cold-compile plan.
    """
    assert "main_pipelined" in _entry_source(*WHOLE_TILE)
    for reason, batch, query_capacity, cache_capacity in (
        ("more than one batch", 2, PAIRED[1], PAIRED[2]),
        ("more than one KV block", 1, 17, 129),
    ):
        assert "main_pipelined" not in _entry_source(batch, query_capacity, cache_capacity), reason
