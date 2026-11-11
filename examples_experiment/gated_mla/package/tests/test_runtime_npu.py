import pytest
import torch

from gmla import DEFAULT_SCALE, reference

from . import fixtures
from .helpers import cpu_inputs, execute_case, metadata, refs, runtime_for, verify_result


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("target", [None, "ascendc"], ids=["default_pto", "ascendc"])
def test_h96_t17_tail_reset_full_chain(npu, dtype, target):
    inputs = cpu_inputs(1, 96, 17, 17, dtype)
    *_, audit = execute_case(inputs, [[17, 17, 0]], target)
    assert audit["selected_attention"] == "c4_prefill"
    assert "cache_tail" in audit["kernels"]


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
    runtime, _, _, _, _, audit = execute_case(inputs, [[128, 128, 0]], None)
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
