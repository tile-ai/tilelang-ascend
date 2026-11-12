import pytest
import torch

from gmla import DEFAULT_SCALE, reference

from . import fixtures
from .helpers import cpu_inputs, execute_case, metadata, refs, runtime_for, verify_result


def _assert_input_bytes(device, expected, device_meta, cpu_meta):
    for name, value in {**device, "metadata": device_meta}.items():
        original = cpu_meta if name == "metadata" else expected[name]
        assert value.dtype == original.dtype and tuple(value.shape) == tuple(original.shape), name
        assert torch.equal(value.detach().cpu().contiguous().view(torch.uint8), original.contiguous().view(torch.uint8)), name


def _payload_inputs(inputs, rows):
    """New CPU mappings only: cpu_inputs() returns shared LRU-cached tensors."""
    joint = dict(inputs)
    joint["latent_kv"] = inputs["latent_kv"].clone().add_(0.125)
    joint["key_aux"] = inputs["key_aux"].clone().sub_(0.25)
    # A uniform E displacement cancels in softmax. Perturb one visible key.
    auxiliary = dict(inputs)
    auxiliary["key_aux"] = inputs["key_aux"].clone()
    auxiliary["key_aux"][:, 0, 0].add_(0.75)
    query = dict(inputs)
    query["q_content"] = inputs["q_content"].clone()
    active = [(b, row[0] - 1) for b, row in enumerate(rows) if row[0] and min(row[1], row[2] + row[0]) > 1]
    assert active, "Q probe needs an active query with at least two visible keys"
    batch, token = active[-1]
    query["q_content"][batch, token, 0, 0].add_(0.75)
    return (
        ("joint_cache_aux", joint, ("latent_kv", "key_aux")),
        ("auxiliary_only", auxiliary, ("key_aux",)),
        ("query_only", query, ("q_content",)),
    )


def _check_live_prefill_payloads(inputs, rows, target, runtime, device, device_meta, initial_result, *, graph):
    cpu_meta = metadata(rows, inputs)
    original_refs = refs(inputs, cpu_meta)
    original_latent = initial_result[1]["latent_out"].detach().cpu().clone()
    pointers = {name: value.data_ptr() for name, value in {**device, "metadata": device_meta}.items()}

    def invoke():
        return runtime.forward(**device, metadata=device_meta, return_intermediates=True)

    def check(result, expected, expected_inputs, *, changed, expected_rows=None, expected_meta=None):
        output, intermediate, audit = result
        torch.npu.synchronize()
        verify_result(output, intermediate, audit, expected, inputs["q_content"].dtype, target)
        latent = intermediate["latent_out"].detach().cpu()
        live_rows = rows if expected_rows is None else expected_rows
        live_meta = cpu_meta if expected_meta is None else expected_meta
        for batch, row in enumerate(live_rows):
            assert torch.count_nonzero(output[batch, row[0] :].cpu()) == 0
            assert torch.count_nonzero(latent[batch, row[0] :]) == 0
        if changed:
            assert not torch.equal(latent, original_latent), "live payload update was ignored"
        assert pointers == {name: value.data_ptr() for name, value in {**device, "metadata": device_meta}.items()}
        _assert_input_bytes(device, expected_inputs, device_meta, live_meta)

    with torch.no_grad():
        handle, captured = None, None
        if graph:
            handle = torch.npu.NPUGraph()
            with torch.npu.graph(handle):
                captured = invoke()
            handle.replay()
            check(captured, original_refs, inputs, changed=False)
        for label, updated, names in _payload_inputs(inputs, rows):
            # Both ordinary and replay checks use this independently calculated
            # FP64 result, with the exact LP bytes copied from CPU to the NPU.
            expected = refs(updated, cpu_meta)
            assert torch.count_nonzero(expected["latent_out"] - original_refs["latent_out"]) > 0, label
            try:
                for name in names:
                    device[name].copy_(updated[name])
                check(invoke(), expected, updated, changed=True)
                if handle is not None:
                    handle.replay()
                    check(captured, expected, updated, changed=True)
            finally:
                for name in names:
                    device[name].copy_(inputs[name])
                torch.npu.synchronize()
            check(invoke(), original_refs, inputs, changed=False)
            if handle is not None:
                handle.replay()
                check(captured, original_refs, inputs, changed=False)
        if handle is not None:
            # Metadata values remain device inputs, never factory parameters.
            # Reuse the existing graph and the same Bx8 storage on both profiles.
            capacities = (inputs["q_content"].shape[1], inputs["latent_kv"].shape[1])
            changed_rows = {(128, 128): [[64, 128, 32]], (17, 17): [[8, 17, 4]]}[capacities]
            changed_meta = metadata(changed_rows, inputs)
            changed_refs = refs(inputs, changed_meta)
            assert torch.count_nonzero(changed_refs["latent_out"] - original_refs["latent_out"]) > 0
            try:
                device_meta.copy_(changed_meta)
                check(invoke(), changed_refs, inputs, changed=True, expected_rows=changed_rows, expected_meta=changed_meta)
                handle.replay()
                check(captured, changed_refs, inputs, changed=True, expected_rows=changed_rows, expected_meta=changed_meta)
            finally:
                device_meta.copy_(cpu_meta)
                torch.npu.synchronize()
            check(invoke(), original_refs, inputs, changed=False)
            handle.replay()
            check(captured, original_refs, inputs, changed=False)


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("target", [None, "ascendc"], ids=["default_pto", "ascendc"])
def test_h96_t17_tail_reset_full_chain(npu, dtype, target):
    inputs = cpu_inputs(1, 96, 17, 17, dtype)
    rows = [[17, 17, 0]]
    runtime, device, device_meta, output, intermediate, audit = execute_case(inputs, rows, target)
    assert audit["selected_attention"] == "c4_prefill"
    assert "cache_tail" in audit["kernels"]
    if target == "ascendc":
        _check_live_prefill_payloads(inputs, rows, target, runtime, device, device_meta, (output, intermediate, audit), graph=True)


@pytest.mark.npu
@pytest.mark.parametrize("target", [None, "ascendc"], ids=["default_pto", "ascendc"])
def test_original_bf16_visible17_failure_and_metadata_reuse(npu, target):
    inputs = cpu_inputs(1, 96, 1, 4096, torch.bfloat16, True)
    runtime = runtime_for(target)
    device = {k: v.npu() for k, v in inputs.items()}
    device_meta = metadata([[1, 4096, 4095]], inputs).npu()
    first_kernels = None
    address = device_meta.data_ptr()
    with torch.no_grad():
        # Original failure used [1,4096,16]. [1,17,16] has the same
        # causal visible prefix min(kv_len,start+1)=17 with identical seven
        # physical inputs. The capacity remains4096; it is not a short-cache case.
        for rows in ([[1, 4096, 4095]], [[1, 17, 16]]):
            cpu_meta = metadata(rows, inputs)
            device_meta.copy_(cpu_meta)
            expected = refs(inputs, cpu_meta)
            previous = None
            for _ in range(3):
                output, intermediate, audit = runtime.forward(**device, metadata=device_meta, return_intermediates=True)
                torch.npu.synchronize()
                verify_result(output, intermediate, audit, expected, torch.bfloat16, target)
                assert intermediate["q_latent_lo"] is not None
                if previous is not None:
                    torch.testing.assert_close(output.cpu(), previous, rtol=0, atol=0)
                previous = output.cpu()
            if first_kernels is None:
                first_kernels = audit["kernels"]
            else:
                assert all(first_kernels[name] is kernel for name, kernel in audit["kernels"].items())
            assert device_meta.data_ptr() == address
            fixtures.verify_read_only(device, inputs, device_meta, cpu_meta)


@pytest.mark.npu
@pytest.mark.parametrize("dtype,target", [(torch.float16, None), (torch.bfloat16, "ascendc")])
def test_empty_splits_inactive_batch_and_tails(npu, dtype, target):
    inputs = cpu_inputs(2, 97, 1, 129, dtype)
    runtime, device, device_meta, output, intermediate, _ = execute_case(inputs, [[1, 129, 16], [0, 0, 0]], target)
    assert torch.count_nonzero(output[1].cpu()) == 0
    high, low = intermediate["q_latent_hi"], intermediate["q_latent_lo"]
    with torch.no_grad():
        latent, partial, _ = runtime.kernel_bundle.decode_module.cube_decode(
            high.view(2, 97, 512),
            device["q_aux"].view(2, 97, 64),
            device["latent_kv"],
            device["key_aux"],
            device_meta,
            ql_residual=None if low is None else low.view(2, 97, 512),
            target=target or "pto",
            return_partials=True,
        )
        torch.npu.synchronize()
    reference.assert_fp16_close(latent, intermediate["latent_out"].view(2, 97, 512).cpu().double())
    maximum, normalizer, weighted = (partial[k].cpu() for k in ("maximum", "normalizer", "weighted_latent"))
    for batch in range(2):
        for head_tile in range(4):
            for split in range(6):
                if batch == 0 and split == 0:
                    continue
                task = (batch * 4 + head_tile) * 6 + split
                assert torch.isneginf(maximum[task]).all()
                assert torch.count_nonzero(normalizer[task]) == 0 and torch.count_nonzero(weighted[task]) == 0


@pytest.mark.npu
def test_continuation_equals_actual_token_decode(npu):
    inputs = cpu_inputs(1, 4, 17, 81, torch.float16)
    runtime, _, _, output, _, _ = execute_case(inputs, [[17, 81, 64]], None)
    parts = []
    with torch.no_grad():
        for token in range(17):
            cpu = fixtures.token_inputs(inputs, token)
            meta = metadata([[1, 81, 64 + token]], cpu)
            device = {k: v.npu() for k, v in cpu.items()}
            parts.append(runtime.forward(**device, metadata=meta.npu()).cpu())
    reference.assert_fp16_close(torch.cat(parts, dim=1), output.cpu().double())


@pytest.mark.npu
@pytest.mark.low_priority
@pytest.mark.parametrize("tq", [127, 128, 129])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("target", [None, "ascendc"])
def test_prefill_127_128_129(npu, tq, dtype, target):
    execute_case(cpu_inputs(1, 96, tq, 129, dtype), [[tq, 129, 0]], target)


@pytest.mark.npu
@pytest.mark.low_priority
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("target", [None, "ascendc"])
def test_h96_full_chain_16k(npu, dtype, target):
    execute_case(cpu_inputs(1, 96, 1, 16384, dtype, True), [[1, 16384, 16383]], target)


@pytest.mark.npu
@pytest.mark.low_priority
@pytest.mark.parametrize("target", [None, "ascendc"])
def test_decode_custom_scale_routes_prefill(npu, target):
    *_, audit = execute_case(cpu_inputs(1, 4, 1, 65, torch.bfloat16), [[1, 65, 64]], target, scale=DEFAULT_SCALE * 0.75)
    assert audit["selected_attention"] == "c4_prefill"


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prefill128_public_profile_core(npu, dtype):
    """The exact new public profile is Tq=Tk128, distinct from old Tk129 cases."""
    inputs = cpu_inputs(1, 96, 128, 128, dtype)
    rows = [[128, 128, 0]]
    runtime, device, device_meta, output, intermediate, audit = execute_case(inputs, rows, None)
    assert runtime.target == "pto" and audit["selected_attention"] == "c4_prefill"
    # The base diagnostic bundle remains immutable N64; only public.forward
    # chooses the separately cached N128 profile for this geometry/default scale.
    assert runtime.kernel_bundle.block_n == 64
    expected = runtime.kernel_bundle.prefill_module.build_cube_attention(
        batch=1,
        heads=96,
        query_capacity=128,
        cache_capacity=128,
        block_m=32,
        block_n=128,
        dtype="float16" if dtype == torch.float16 else "bfloat16",
        target="pto",
        max_blocks=20,
        scale=DEFAULT_SCALE,
        has_ql_residual=dtype == torch.bfloat16,
    )
    actual = audit["kernels"]["attention"]
    assert actual is expected and str(actual.target) == "pto"
    assert actual.get_kernel_source() == expected.get_kernel_source()
    _check_live_prefill_payloads(inputs, rows, None, runtime, device, device_meta, (output, intermediate, audit), graph=True)


@pytest.mark.npu
@pytest.mark.parametrize("dtype,target", [(torch.float16, None), (torch.bfloat16, "ascendc")])
def test_prefill_one_block_cache_batch_switch(npu, dtype, target):
    inputs = cpu_inputs(3, 96, 3, 63, dtype)
    rows = [[3, 63, 60], [0, 0, 0], [2, 33, 31]]
    runtime, device, device_meta, output, intermediate, audit = execute_case(inputs, rows, target)
    assert audit["selected_attention"] == "c4_prefill"
    assert runtime.kernel_bundle.block_n == 64
    _check_live_prefill_payloads(inputs, rows, target, runtime, device, device_meta, (output, intermediate, audit), graph=False)


@pytest.mark.npu
def test_prefill_multikv_uniform_heads_fallback(npu):
    inputs = cpu_inputs(1, 96, 17, 129, torch.float16)
    rows = [[17, 129, 112]]
    runtime, device, device_meta, output, intermediate, audit = execute_case(inputs, rows, None)
    assert audit["selected_attention"] == "c4_prefill" and "cache_tail" in audit["kernels"]
    assert runtime.kernel_bundle.block_n == 64
    _check_live_prefill_payloads(inputs, rows, None, runtime, device, device_meta, (output, intermediate, audit), graph=False)
