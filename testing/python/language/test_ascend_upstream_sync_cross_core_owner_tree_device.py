# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""A3 acceptance of one closed publication with strided multi-workgroup output."""

import re

import pytest
import torch

import tilelang
from tilelang import tvm
from test_ascend_upstream_sync_cross_core_owner_tree import (
    _OWNER_TREE_PASS_CONFIGS,
    _owner_tree_gemm,
)


pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.mark.parametrize(
    "workgroups,in_place,input_alias",
    [(2, False, False), (3, False, True), (2, True, False), (3, True, True)],
)
def test_closed_owner_trees_complete_strided_publication_and_fresh_cython_roles(
    workgroups,
    in_place,
    input_alias,
):
    program = _owner_tree_gemm(workgroups, in_place=in_place)
    assert "tir.noalias" not in program.attrs
    tilelang.disable_cache()
    kernel = tilelang.compile(
        program,
        out_idx=[4, 5],
        workspace_idx=[3, 6],
        execution_backend="cython",
        target="ascendc",
        platform="A3",
        pass_configs=_OWNER_TREE_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    source = kernel.get_kernel_source()
    assert len(re.findall(r"CrossCoreSetFlag<(?:2|0x2), PIPE_FIX>", source)) == 1
    assert source.count("CrossCoreWaitFlag(") == 1
    assert "HardEvent::S_MTE2" in source
    total_rows, total_k, columns, stride = workgroups * 64, 128, 16, 32
    for phase in (0, 2, 5):
        first = ((torch.arange(total_rows * total_k) + phase) % 7) - 3
        first = first.reshape(total_rows, total_k).half() / 16
        first_device = first.npu()
        if input_alias:
            second = first.view(-1)[: total_k * columns].view(total_k, columns)
            second_device = first_device.view(-1)[: total_k * columns].view(total_k, columns)
            first_storage = first_device.untyped_storage().data_ptr()
            assert second_device.untyped_storage().data_ptr() == first_storage
        else:
            second = ((torch.arange(total_k * columns) + 3 * phase) % 7) - 3
            second = second.reshape(total_k, columns).half() / 8
            second_device = second.npu()
        bias = (((torch.arange(columns) + phase) % 7) - 3).reshape(1, columns).float() / 64
        bias_device = bias.npu()
        # All values and products are dyadic. The bounded short reduction and
        # bias addition are exact in FP32 and the resulting FP16 output.
        expected = first.float() @ second.float() + bias * 2
        result, half_result = kernel(first_device, second_device, bias_device)
        torch.npu.synchronize()
        assert result.shape == half_result.shape == (total_rows, stride)
        assert result.dtype == torch.float32 and half_result.dtype == torch.float16
        input_tensors = (first_device, second_device, bias_device)
        inputs = {tensor.untyped_storage().data_ptr() for tensor in input_tensors}
        outputs = {tensor.untyped_storage().data_ptr() for tensor in (result, half_result)}
        assert len(outputs) == 2 and inputs.isdisjoint(outputs)
        # Padding columns are not logical outputs. The host byte oracle audits
        # their exact exclusion from each strided DMA and cross-instance write.
        torch.testing.assert_close(result[:, :columns].cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(half_result[:, :columns].cpu(), expected.half(), rtol=0, atol=0)
        torch.testing.assert_close(first_device.cpu(), first, rtol=0, atol=0)
        torch.testing.assert_close(bias_device.cpu(), bias, rtol=0, atol=0)
