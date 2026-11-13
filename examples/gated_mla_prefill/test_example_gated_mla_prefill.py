"""Correctness gates for the Gated MLA prefill example.

Three kinds of check appear here and they are deliberately not held to the same
standard.

The budgeted one compares the chained forward against the FP32 reference in the
example module, position sensitively: rows outside the valid query length must
be exactly zero before any tolerance is applied to the rest.

The analytic one fixes QL = QA = 0, so every visible score is zero and after the
online maximum is subtracted the probability of each visible key is exp(0). With
latent_kv[k, :] = k the PV accumulation is sum(0..n-1) = n(n-1)/2 and the divide
by the normaliser yields (n-1)/2, where n is the number of keys visible to that
row. Every such value is a multiple of 0.5, and a mask that admits one key too
many or too few moves a whole row by exactly 0.5, so the assertion rounds to the
nearest half and compares exactly: there is no tolerance to argue about. On
device the ratio passes through the vector exponential and an FP32 divide, so a
second assertion bounds that drift at 8 ULP of FP32 rather than hiding it.

The self-consistency ones compare two routes through the same arithmetic and are
held to bit-identity with no tolerance at all. They exist to catch a tile that
writes past its own rows into a neighbouring batch, which produces a bounded,
plausible-looking difference that a tolerance would admit.

Shapes are kept to the measured profile and one row-wise-mask companion. block_n
follows the production selector, so forcing the other combination would compile a
kernel no public call can reach.
"""

import pytest
import torch

from example_gated_mla_prefill import (
    DEFAULT_SCALE,
    build_cube_attention,
    cube_prefill,
    gated_mla_prefill,
    public_prefill_block_n,
    reference_gated_mla_prefill,
)

HEADS = 96
WIDTH = 512
AUX = 64
DEVICE = "npu:0"


@pytest.fixture(scope="module")
def npu_device():
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        pytest.fail("NPU tests require an Ascend device; no silent device skips")
    torch.npu.set_device(0)


# Eight ULP of FP32 at the oracle magnitude; the measured worst case is near one.
DRIFT = 2.0**-21

# batch, q_len, kv_len. The selector returns 128 for exactly the first profile.
WHOLE_TILE = (1, 128, 128)
ROW_WISE = (1, 64, 128)
PAIRED = (2, 64, 128)


def selected_block_n(batch, q_len, kv_len):
    return public_prefill_block_n(batch, q_len, HEADS, kv_len, DEFAULT_SCALE)


def _inputs(batch, q_len, kv_len, dtype, seed=0):
    torch.manual_seed(seed)
    return {
        "q_content": (torch.randn(batch, q_len, HEADS, 128) * 0.1).to(dtype),
        "q_aux": (torch.randn(batch, q_len, HEADS, AUX) * 0.1).to(dtype),
        "latent_kv": (torch.randn(batch, kv_len, WIDTH) * 0.1).to(dtype),
        "key_aux": (torch.randn(batch, kv_len, AUX) * 0.1).to(dtype),
        "w_k": (torch.randn(HEADS, 128, WIDTH) * 0.05).to(dtype),
        "w_v": (torch.randn(HEADS, 128, WIDTH) * 0.05).to(dtype),
        "gate_logits": (torch.randn(batch, q_len, HEADS, 128) * 0.5).to(dtype),
    }


def _to_device(cpu):
    return {name: value.npu() for name, value in cpu.items()}


def _metadata(rows):
    return torch.tensor([row + [0] * (8 - len(row)) for row in rows], dtype=torch.int32)


def _attention_only(batch, q_len, kv_len, dtype, metadata, ql, qa, c, e):
    out = cube_prefill(
        ql,
        qa,
        c,
        e,
        metadata,
        block_m=32,
        block_n=selected_block_n(batch, q_len, kv_len),
        target="pto",
        max_blocks=20,
    )
    torch.npu.synchronize()
    return out


def _zero_query_window(batch, q_len, kv_len, dtype):
    """QL = QA = 0 and a ramp cache: every visible key contributes exp(0)."""
    ql = torch.zeros(batch, q_len, HEADS, WIDTH, dtype=dtype, device=DEVICE)
    qa = torch.zeros(batch, q_len, HEADS, AUX, dtype=dtype, device=DEVICE)
    ramp = torch.arange(kv_len, dtype=torch.float32).view(1, kv_len, 1)
    c = ramp.expand(batch, kv_len, WIDTH).to(dtype).to(DEVICE).contiguous()
    e = torch.zeros(batch, kv_len, AUX, dtype=dtype, device=DEVICE)
    return ql, qa, c, e


def _assert_quantised_oracle(got, want):
    got, want = got.float(), want.float()
    assert torch.equal(torch.round(got * 2) / 2, want), "an oracle row moved by at least a quarter"
    drift = (got - want).abs().max().item()
    assert drift <= want.abs().max().item() * DRIFT, drift


def _window_oracle(out, visible):
    _assert_quantised_oracle(out, ((visible - 1) / 2).view(1, -1, 1, 1).expand_as(out))


# --- budgeted gate against the FP32 reference --------------------------------


@pytest.mark.parametrize("target", ["pto", "ascendc"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chained_forward_matches_the_reference(npu_device, target, dtype):
    """The whole chain, Q absorption through the gate, against FP32."""
    batch, q_len, kv_len = WHOLE_TILE
    cpu = _inputs(batch, q_len, kv_len, dtype)
    metadata = _metadata([[q_len, kv_len, 0]] * batch)
    expected = reference_gated_mla_prefill(**cpu, metadata=metadata)
    actual = gated_mla_prefill(**_to_device(cpu), metadata=metadata.npu(), target=target).cpu()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2**-9, atol=2**-10)


@pytest.mark.parametrize("target", ["pto", "ascendc"])
def test_rows_past_the_valid_query_length_are_exactly_zero(npu_device, target):
    """Position sensitive: the inactive tail is exact, not merely small."""
    batch, q_len, kv_len = WHOLE_TILE
    valid = 40
    cpu = _inputs(batch, q_len, kv_len, torch.float16)
    metadata = _metadata([[valid, kv_len, 0]])
    actual = gated_mla_prefill(**_to_device(cpu), metadata=metadata.npu(), target=target).cpu()
    assert torch.equal(actual[:, valid:], torch.zeros_like(actual[:, valid:]))
    assert not torch.equal(actual[:, :valid], torch.zeros_like(actual[:, :valid]))


# --- analytic causal-mask oracle ---------------------------------------------


@pytest.mark.parametrize("shape", [WHOLE_TILE, ROW_WISE], ids=["whole_tile_mask", "row_wise_mask"])
def test_causal_window_matches_the_analytic_value(npu_device, shape):
    """Row q averages exactly the first q + 1 cache rows, on both mask paths."""
    batch, q_len, kv_len = shape
    ql, qa, c, e = _zero_query_window(batch, q_len, kv_len, torch.float16)
    out = _attention_only(batch, q_len, kv_len, torch.float16, _metadata([[q_len, kv_len, 0]]).npu(), ql, qa, c, e)
    visible = torch.clamp(torch.arange(1, q_len + 1, dtype=torch.float32, device=DEVICE), max=kv_len)
    _window_oracle(out, visible)


@pytest.mark.parametrize("start", [1, 7, 64])
def test_query_start_shifts_the_window(npu_device, start):
    """query_start moves every window by that many keys, then clamps at kv_len."""
    batch, q_len, kv_len = WHOLE_TILE
    ql, qa, c, e = _zero_query_window(batch, q_len, kv_len, torch.float16)
    out = _attention_only(batch, q_len, kv_len, torch.float16, _metadata([[q_len, kv_len, start]]).npu(), ql, qa, c, e)
    visible = torch.clamp(torch.arange(1 + start, q_len + 1 + start, dtype=torch.float32, device=DEVICE), max=kv_len)
    _window_oracle(out, visible)


@pytest.mark.parametrize("kv_len", [1, 40, 127])
def test_kv_length_clamps_the_window(npu_device, kv_len):
    """A short valid cache clamps every window, including rows past its end."""
    batch, q_len, capacity = WHOLE_TILE
    ql, qa, c, e = _zero_query_window(batch, q_len, capacity, torch.float16)
    out = _attention_only(batch, q_len, capacity, torch.float16, _metadata([[q_len, kv_len, 0]]).npu(), ql, qa, c, e)
    visible = torch.clamp(torch.arange(1, q_len + 1, dtype=torch.float32, device=DEVICE), max=kv_len)
    _window_oracle(out, visible)


# --- bit-exact self consistency ----------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batches_do_not_influence_each_other(npu_device, dtype):
    """A batch computed beside another must be bit-identical to computing it alone.

    This also pins the pipelined entry at batch one. That entry issues the QK of
    one wave before the PV of the previous one, and the QK may reload C and E for
    a different batch into the L1 buffers the pending PV still reads, so it is
    compiled for a single batch only.
    """
    batch, q_len, kv_len = PAIRED
    assert selected_block_n(batch, q_len, kv_len) == selected_block_n(1, q_len, kv_len)
    cpu = _inputs(batch, q_len, kv_len, dtype)
    metadata = _metadata([[q_len, kv_len, 0]] * batch)
    together = gated_mla_prefill(**_to_device(cpu), metadata=metadata.npu(), target="pto").cpu()
    for index in range(batch):
        alone_cpu = dict(cpu)
        for name in ("q_content", "q_aux", "latent_kv", "key_aux", "gate_logits"):
            alone_cpu[name] = cpu[name][index : index + 1].contiguous()
        alone = gated_mla_prefill(**_to_device(alone_cpu), metadata=_metadata([[q_len, kv_len, 0]]).npu(), target="pto").cpu()
        assert torch.equal(together[index : index + 1], alone)


def test_an_inactive_neighbour_does_not_disturb_an_active_batch(npu_device):
    """An empty batch beside a live one must not change the live result."""
    batch, q_len, kv_len = PAIRED
    cpu = _inputs(batch, q_len, kv_len, torch.float16, seed=17)
    mixed = gated_mla_prefill(**_to_device(cpu), metadata=_metadata([[q_len, kv_len, 0], [0, kv_len, 0]]).npu(), target="pto").cpu()
    alone_cpu = dict(cpu)
    for name in ("q_content", "q_aux", "latent_kv", "key_aux", "gate_logits"):
        alone_cpu[name] = cpu[name][0:1].contiguous()
    alone = gated_mla_prefill(**_to_device(alone_cpu), metadata=_metadata([[q_len, kv_len, 0]]).npu(), target="pto").cpu()
    assert torch.equal(mixed[0:1], alone)
    assert torch.equal(mixed[1:2], torch.zeros_like(mixed[1:2]))


# --- the documented domain, asserted rather than trusted ----------------------


def _entry_source(batch, query_capacity, cache_capacity):
    kernel = build_cube_attention(
        batch=batch,
        heads=HEADS,
        cache_capacity=cache_capacity,
        query_capacity=query_capacity,
        block_m=32,
        block_n=selected_block_n(batch, query_capacity, cache_capacity),
        dtype="float16",
        target="pto",
        has_ql_residual=False,
        max_blocks=20,
    )
    return kernel.get_kernel_source()


def test_the_pipelined_entry_is_compiled_only_inside_its_documented_domain(npu_device):
    """A failure here means the documented domain changed, not that a value drifted.

    Lifting the batch bound needs a second C/E slot, about 144 KiB more L1, so
    this asserts the bound still holds rather than trusting the comment. Every
    shape below is one another test already compiles.
    """
    assert "main_pipelined" in _entry_source(*WHOLE_TILE)
    for reason, batch, query_capacity, cache_capacity in (
        ("more than one batch", PAIRED[0], PAIRED[1], PAIRED[2]),
        ("more than one KV block", 1, 128, 129),
    ):
        assert "main_pipelined" not in _entry_source(batch, query_capacity, cache_capacity), reason


# --- negative gates: invalid input is rejected, not quietly accepted ----------


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda kw: kw.update(q_content=kw["q_content"][..., :64].contiguous()), "q_content must have shape"),
        # A slice of the last axis is a view, so the contiguity gate fires first:
        # a strided operand is rejected rather than silently read with the wrong
        # stride, which is a separate guarantee from the shape check above.
        (lambda kw: kw.update(q_content=kw["q_content"][..., :64]), "q_content must be contiguous"),
        (lambda kw: kw.update(q_content=kw["q_content"].float()), "float16 or bfloat16"),
        (lambda kw: kw.update(w_v=kw["w_v"].float()), "w_v must be float16 or bfloat16"),
    ],
    ids=["q_content_width", "q_content_strided", "q_content_dtype", "w_v_dtype"],
)
def test_invalid_operands_are_rejected(npu_device, mutate, message):
    batch, q_len, kv_len = WHOLE_TILE
    kwargs = _to_device(_inputs(batch, q_len, kv_len, torch.float16))
    kwargs["metadata"] = _metadata([[q_len, kv_len, 0]]).npu()
    mutate(kwargs)
    with pytest.raises((ValueError, TypeError), match=message):
        gated_mla_prefill(**kwargs, target="pto")
