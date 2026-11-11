from pathlib import Path
import subprocess
import sys

import pytest
import torch

from .helpers import cpu_inputs, metadata, runtime_for


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("target", [None, "ascendc"])
def test_empty_query_and_zero_cache_outputs(npu, dtype, target):
    runtime = runtime_for(target)
    with torch.no_grad():
        for tq, tk in ((0, 17), (1, 0)):
            cpu = cpu_inputs(1, 96, tq, tk, dtype)
            device = {k: v.npu() for k, v in cpu.items()}
            meta = metadata([[0, 0, 0]], cpu).npu()
            output, _, audit = runtime.forward(**device, metadata=meta, return_intermediates=True)
            torch.npu.synchronize()
            assert output.shape == (1, tq, 96, 128) and output.dtype == dtype
            if tq == 0:
                assert not audit["kernels"] and audit["selected_attention"] == "empty_query"
            else:
                assert set(audit["kernels"]) == {"empty_cache_zero"}
                assert torch.count_nonzero(output.cpu()) == 0
                poison = torch.full_like(output, 7)
                runtime.kernel_bundle.empty_cache_module.empty_cache_forward(**device, metadata=meta, target=target or "pto", out=poison)
                torch.npu.synchronize()
                assert torch.count_nonzero(poison.cpu()) == 0


@pytest.mark.npu
def test_actual_nz_rejected_in_separate_process(npu):
    code = r"""
import torch, torch_npu
from gmla import build_runtime, metadata_for_reference, prepare_metadata, validate_runtime_metadata_cpu
from gmla.layout import InternalFormatError, require_dense_base
from gmla.reference import validate_inputs

# Establish the pinned TileLang/TVM native libraries before the isolated
# format-cast operation can load another libtvm with the same library name.
# This diagnostic property imports the selected modules; it runs no factory.
runtime = build_runtime()
assert not torch.npu.is_initialized()
bundle = runtime.kernel_bundle
assert bundle.target == "pto" and not torch.npu.is_initialized()
torch.npu.set_device(0)
torch_npu.npu.config.allow_internal_format = True
# Single invalid condition: only q_content changes physical format. Every other
# tensor, and all logical properties of q_content, satisfy the real contract.
shapes = {
    "q_content": (1, 1, 96, 128),
    "q_aux": (1, 1, 96, 64),
    "latent_kv": (1, 1, 512),
    "key_aux": (1, 1, 64),
    "w_k": (96, 128, 512),
    "w_v": (96, 128, 512),
    "gate_logits": (1, 1, 96, 128),
}
cpu = {name: torch.ones(shape, dtype=torch.float16) for name, shape in shapes.items()}
cpu_meta = prepare_metadata([[1, 1, 0]], batch=1, query_capacity=1, cache_capacity=1)
validate_runtime_metadata_cpu(cpu_meta, B=1, Tq=1, Tk=1)
lengths = metadata_for_reference(cpu_meta)
validate_inputs(**cpu, **lengths)
device = {name: value.npu() for name, value in cpu.items()}
metadata = cpu_meta.to(device["q_content"].device)
validate_inputs(**device, **lengths, require_npu=True, allowed_dtypes=(torch.float16,))
for name, value in device.items():
    require_dense_base(value, name=name, shape=shapes[name], dtype=torch.float16, device=device["q_content"].device)
require_dense_base(metadata, name="metadata", shape=(1, 8), dtype=torch.int32, device=device["q_content"].device)
dense = device["q_content"]
# Public allocator creates a real physical NZ tensor without a legacy TBE cast.
# Payload is uninitialized and never filled, copied, read or computed; only
# metadata reaches the guard, which must reject before the first kernel/view.
nz = torch_npu.empty_with_format(dense.shape, dtype=dense.dtype, device=dense.device, acl_format=torch_npu.Format.FRACTAL_NZ)
assert torch_npu.get_npu_format(nz) == torch_npu.Format.FRACTAL_NZ
assert nz.shape == dense.shape and nz.dtype == dense.dtype and nz.device == dense.device
assert nz.is_contiguous(), "NZ fixture must retain valid logical row-major layout"
device["q_content"] = nz
validate_inputs(**device, **lengths, require_npu=True, allowed_dtypes=(torch.float16,))
try:
    with torch.no_grad():
        runtime.forward(**device, metadata=metadata)
except InternalFormatError as error:
    assert "q_content" in str(error), "Only the query physical format may trigger rejection"
    print('ACTUAL_NZ_REJECTED_BEFORE_FIRST_VIEW')
else:
    raise AssertionError('NZ accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ACTUAL_NZ_REJECTED_BEFORE_FIRST_VIEW" in result.stdout
