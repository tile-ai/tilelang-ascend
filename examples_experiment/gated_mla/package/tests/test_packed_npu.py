import pytest
import torch
from pathlib import Path
import subprocess
import sys

from gmla import build_packed_adapter, create_cache, reference

from .helpers import cpu_inputs, metadata, refs, verify_result


@pytest.mark.npu
@pytest.mark.parametrize("dtype,target", [(torch.float16, None), (torch.bfloat16, "ascendc")])
def test_packed_h96_tail_prefill_with_selected_providers(npu, dtype, target):
    cpu = cpu_inputs(1, 96, 17, 17, dtype)
    expected = refs(cpu, metadata([[17, 17, 0]], cpu))
    packed_q = torch.cat((cpu["q_content"], cpu["q_aux"]), dim=-1).view(1, 17, 96 * 192).npu()
    packed_w = torch.cat((cpu["w_k"], cpu["w_v"]), dim=1).view(96 * 256, 512).npu()
    gate = cpu["gate_logits"].view(1, 17, 96 * 128).npu()
    c = cpu["latent_kv"].npu()
    raw = torch.cat((torch.zeros_like(cpu["latent_kv"]), cpu["key_aux"]), dim=-1).npu()
    with torch.no_grad():
        owner = create_cache(1, 17, dtype=dtype, **({} if target is None else {"target": target}))
        adapter = build_packed_adapter(owner, heads=96)
        output, intermediate, audit = adapter.append_and_forward(packed_q, packed_w, c, raw, gate, [17], return_intermediates=True)
        verify_result(output, intermediate, audit, expected, dtype, target)
        assert audit["route"] == "c4_prefill" and owner.lengths == (17,) and owner.generation == 1
        torch.testing.assert_close(intermediate["qa"].cpu(), cpu["q_aux"], rtol=0, atol=0)
        again = adapter.read_existing_forward(packed_q, packed_w, gate, [17], [0])
        reference.assert_fp16_close(again, expected["expanded"])
        assert owner.generation == 1
    torch.testing.assert_close(owner.cache_c.cpu(), cpu["latent_kv"], rtol=0, atol=0)
    torch.testing.assert_close(owner.cache_e.cpu(), cpu["key_aux"], rtol=0, atol=0)


@pytest.mark.npu
@pytest.mark.parametrize("dtype,target", [(torch.float16, None), (torch.bfloat16, "ascendc")])
def test_owner_two_appends_inactive_and_no_overwrite(npu, dtype, target):
    gen = torch.Generator().manual_seed(13)
    c1 = torch.randn((2, 17, 512), generator=gen).to(dtype)
    raw1 = torch.randn((2, 17, 576), generator=gen).to(dtype)
    c2 = torch.randn((2, 17, 512), generator=gen).to(dtype)
    raw2 = torch.randn((2, 17, 576), generator=gen).to(dtype)
    with torch.no_grad():
        owner = create_cache(2, 65, dtype=dtype, **({} if target is None else {"target": target}))
        assert torch.count_nonzero(owner.cache_c.cpu()) == 0 and torch.count_nonzero(owner.cache_e.cpu()) == 0
        owner.append(c1.npu(), raw1.npu(), [17, 0])
        owner.append(c2.npu(), raw2.npu(), [9, 17])
        assert owner.lengths == (26, 17) and owner.generation == 2
        before = owner.launches
        owner.read_existing([1, 0], [25, 17], Qcap=1)
        assert owner.launches == before
        with pytest.raises(ValueError):
            owner.append(torch.empty((2, 40, 512), dtype=dtype).npu(), torch.empty((2, 40, 576), dtype=dtype).npu(), [40, 0])
        assert owner.launches == before and owner.lengths == (26, 17)
    observed_c, observed_e = owner.cache_c.cpu(), owner.cache_e.cpu()
    for b, old, count in ((0, 17, 9), (1, 0, 17)):
        torch.testing.assert_close(observed_c[b, old : old + count], c2[b, :count], rtol=0, atol=0)
        torch.testing.assert_close(observed_e[b, old : old + count], raw2[b, :count, 512:], rtol=0, atol=0)
        assert torch.count_nonzero(observed_c[b, old + count :]) == 0 and torch.count_nonzero(observed_e[b, old + count :]) == 0
    torch.testing.assert_close(observed_c[0, :17], c1[0], rtol=0, atol=0)
    torch.testing.assert_close(observed_e[0, :17], raw1[0, :, 512:], rtol=0, atol=0)


@pytest.mark.npu
def test_direct_owner_rejects_actual_nz_before_commit_in_child(npu):
    code = r"""
import torch, torch_npu
from gmla import build_runtime, create_cache
from gmla.layout import InternalFormatError, require_dense_base

runtime = build_runtime()
assert not torch.npu.is_initialized()
_ = runtime.kernel_bundle
assert not torch.npu.is_initialized()
torch.npu.set_device(0)
# Process-local fixture option only; the test parent/production option is unchanged.
torch_npu.npu.config.allow_internal_format = True
with torch.no_grad():
    owner = create_cache(1, 65, dtype=torch.float16)
    c = torch.ones((1, 17, 512), dtype=torch.float16).npu()
    raw = torch.ones((1, 17, 576), dtype=torch.float16).npu()
    require_dense_base(c, name='positive Cnorm')
    require_dense_base(raw, name='positive rawKV')
    baseline = (owner.lengths, owner.generation, owner.launches)
    for name, original in [('Cnorm', c), ('rawKV', raw)]:
        # Real NZ allocation only: uninitialized payload is never read/computed.
        # The metadata guard must reject before any append launch or commit.
        nz = torch_npu.empty_with_format(original.shape, dtype=original.dtype, device=original.device, acl_format=torch_npu.Format.FRACTAL_NZ)
        assert torch_npu.get_npu_format(nz) == torch_npu.Format.FRACTAL_NZ
        assert nz.shape == original.shape and nz.dtype == original.dtype and nz.device == original.device
        assert nz.is_contiguous(), 'Only the physical format may be invalid'
        try:
            owner.append(nz if name == 'Cnorm' else c, nz if name == 'rawKV' else raw, [17])
        except InternalFormatError as error:
            assert name in str(error)
        else:
            raise AssertionError('Direct owner accepted NZ ' + name)
        assert (owner.lengths, owner.generation, owner.launches) == baseline
        assert torch.count_nonzero(owner.cache_c.cpu()) == 0
        assert torch.count_nonzero(owner.cache_e.cpu()) == 0
    owner.read_existing([0], [0], Qcap=17)
    assert (owner.lengths, owner.generation, owner.launches) == baseline
print('DIRECT_OWNER_NZ_REJECTED_WITHOUT_COMMIT')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DIRECT_OWNER_NZ_REJECTED_WITHOUT_COMMIT" in result.stdout
