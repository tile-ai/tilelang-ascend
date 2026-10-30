"""Ascend device tests and independent CPU mathematical reference checks."""

import os
from pathlib import Path

import pytest
import torch

from gated_mla_golden import check_case, make_decode_inputs


@pytest.fixture(scope="module")
def npu_device():
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        pytest.fail("NPU tests require an Ascend device; no silent device skips")
    torch.npu.set_device(0)


def destination():
    path = os.environ.get("GMLA_OUTPUT")
    return Path(path) if path else None


@pytest.mark.parametrize("seed", [0, 1, 17])
def test_decode_full_chain(npu_device, seed):
    check_case(
        make_decode_inputs(seed),
        dict(q_lens=(1,), kv_lens=(64,), query_start=(63,)),
        case_name=f"seed_{seed}",
        output_dir=destination(),
        repeat=3,
    )


@pytest.mark.parametrize(
    "kv_len,position",
    [
        pytest.param(1, 0, marks=pytest.mark.low_priority),
        (17, 16),
        pytest.param(63, 62, marks=pytest.mark.low_priority),
        (64, 16),
    ],
)
def test_decode_valid_length_and_causal(npu_device, kv_len, position):
    inputs = make_decode_inputs(3)
    # Large finite tail values expose missing masks without violating input contract.
    inputs["latent_kv"][:, position + 1 :] = 32
    inputs["key_aux"][:, position + 1 :] = -32
    check_case(
        inputs,
        dict(q_lens=(1,), kv_lens=(kv_len,), query_start=(position,)),
        case_name=f"mask_{kv_len}_{position}",
        output_dir=destination(),
    )


def test_decode_gate_zero_and_saturation(npu_device):
    inputs = make_decode_inputs(2)
    inputs["gate_logits"].zero_()
    inputs["gate_logits"][..., :32] = -20
    inputs["gate_logits"][..., 96:] = 20
    check_case(inputs, dict(q_lens=(1,), kv_lens=(64,), query_start=(63,)), case_name="gate", output_dir=destination())


def test_decode_inactive_query(npu_device):
    check_case(make_decode_inputs(4), dict(q_lens=(0,), kv_lens=(0,), query_start=(0,)), case_name="inactive", output_dir=destination())


def test_decode_zero_query_reads_mean(npu_device):
    inputs = make_decode_inputs(5)
    inputs["q_content"].zero_()
    inputs["q_aux"].zero_()
    inputs["gate_logits"].zero_()
    check_case(inputs, dict(q_lens=(1,), kv_lens=(17,), query_start=(16,)), case_name="zero_q_mean", output_dir=destination())


def test_decode_future_and_padding_invariance(npu_device):
    inputs = make_decode_inputs(6)
    meta = dict(q_lens=(1,), kv_lens=(64,), query_start=(16,))
    first, _ = check_case(inputs, meta, case_name="invariance_before")
    inputs["latent_kv"][:, 17:] = 40
    inputs["key_aux"][:, 17:] = 40
    after, _ = check_case(inputs, meta, case_name="invariance_after")
    torch.testing.assert_close(first, after, rtol=0, atol=0)


@pytest.mark.parametrize("visible", [1, pytest.param(7, marks=pytest.mark.low_priority), 17, 63, 64])
def test_latent_readout_prefix_mask(npu_device, visible):
    """Exercise aligned whole-row tail clearing independently of softmax zeros."""
    import torch_npu  # noqa: F401
    from example_gated_mla_decode import latent_readout_kernel

    cache = make_decode_inputs(19)["latent_kv"]
    cache[:, visible:] = 32
    # Deliberately nonzero tail weights expose missing or mis-offset clearing.
    probabilities = torch.linspace(0.125, 0.25, 64, dtype=torch.float32)
    expected = (probabilities[:visible].double()[:, None] * cache[0, :visible].double()).sum(dim=0)
    device_cache, device_probabilities = cache.npu(), probabilities.npu()
    actual = latent_readout_kernel(visible)(device_probabilities, device_cache)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=1e-3, atol=5e-5)
    torch.testing.assert_close(device_cache.cpu(), cache, rtol=0, atol=0)
    torch.testing.assert_close(device_probabilities.cpu(), probabilities, rtol=0, atol=0)


@pytest.mark.low_priority
def test_decode_extreme_scale_single_key(npu_device):
    inputs = make_decode_inputs(7)
    inputs["q_content"].fill_(-1)
    inputs["w_k"].fill_(1)
    inputs["latent_kv"].fill_(1)
    inputs["q_aux"].zero_()
    inputs["key_aux"].zero_()
    inputs["w_v"].fill_(1 / 512)
    inputs["gate_logits"].zero_()
    # A finite -1e30 mask would exceed the valid score -6.5536e31 here.
    # Official FP16 scores overflow at this scale; compare FP32 official separately.
    import torch_npu  # noqa: F401
    from example_gated_mla_decode import gated_mla_core

    device_inputs = {name: t.npu() for name, t in inputs.items()}
    meta = dict(q_lens=(1,), kv_lens=(1,), query_start=(0,), scale=1e27)
    result, intermediates, _ = gated_mla_core(
        **device_inputs,
        **meta,
        return_intermediates=True,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), torch.full((1, 1, 1, 128), 0.5).half(), rtol=0, atol=0)
    assert intermediates["probs"].cpu()[0] == 1
    assert torch.count_nonzero(intermediates["probs"].cpu()[1:]) == 0


def test_decode_contract_rejects_before_kernel(npu_device):
    import torch_npu  # noqa: F401
    from example_gated_mla_decode import gated_mla_core

    inputs = {name: t.npu() for name, t in make_decode_inputs().items()}
    meta = dict(q_lens=(1,), kv_lens=(64,), query_start=(63,))
    with pytest.raises(ValueError, match="exceeds"):
        gated_mla_core(**inputs, **{**meta, "query_start": (64,)})
    bad_dtype = {name: t.float() for name, t in inputs.items()}
    with pytest.raises(TypeError, match="unsupported"):
        gated_mla_core(**bad_dtype, **meta)
    wrong_shape = dict(inputs)
    wrong_shape["latent_kv"] = inputs["latent_kv"][:, :32].contiguous()
    wrong_shape["key_aux"] = inputs["key_aux"][:, :32].contiguous()
    with pytest.raises(NotImplementedError, match="L0 decode"):
        gated_mla_core(**wrong_shape, q_lens=(1,), kv_lens=(32,), query_start=(31,))


@pytest.mark.parametrize("seed", [0, 17])
@pytest.mark.parametrize("visible", [1, 17, 63, 64])
def test_cpu_independent_references(seed, visible):
    from gated_mla_golden import absorbed_reference, expanded_reference

    inputs = make_decode_inputs(seed)
    metadata = dict(q_lens=(1,), kv_lens=(64,), query_start=(visible - 1,))
    expected = expanded_reference(**inputs, **metadata)
    actual = absorbed_reference(**inputs, **metadata)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("route", ["expanded_reference", "absorbed_reference"])
def test_cpu_single_key_hand_computed(route):
    import gated_mla_golden

    inputs = make_decode_inputs(0)
    inputs["latent_kv"].fill_(1)
    inputs["w_v"].fill_(1 / 512)
    inputs["gate_logits"].zero_()
    actual = getattr(gated_mla_golden, route)(**inputs, q_lens=(1,), kv_lens=(1,), query_start=(0,))
    torch.testing.assert_close(actual, torch.full((1, 1, 1, 128), 0.5).double(), rtol=0, atol=0)


def test_cpu_negative_infinity_mask_counterexample():
    valid_score = -6.5536e31
    correct = torch.softmax(torch.tensor([valid_score, -torch.inf], dtype=torch.float64), dim=0)
    finite_sentinel = torch.softmax(torch.tensor([valid_score, -1e30], dtype=torch.float64), dim=0)
    torch.testing.assert_close(correct, torch.tensor([1.0, 0.0]).double(), rtol=0, atol=0)
    assert finite_sentinel[1] == 1
