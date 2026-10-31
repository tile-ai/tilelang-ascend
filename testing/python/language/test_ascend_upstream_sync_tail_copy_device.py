# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Exercise both internal paths of one GM-to-UB helper with CCE sync off."""

import pytest
import torch

from tilelang import language as T
from tilelang import tvm

from test_ascend_upstream_sync_device import _compile_device_kernel


pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)


@pytest.fixture(scope="module")
def dynamic_tail_copy_kernel():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")
    n = T.symbolic("n")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), O: T.Tensor((2, 2, 64), "float32")):
        with T.Kernel(1, is_npu=True) as (_cid, vid):
            values = T.alloc_ub((64,), "float32")
            with T.Scope("V"):
                if n > 0 and n <= 64:
                    T.tile.fill(values, -7)
                    T.copy(values, O[vid, 0, :])
                    T.copy(A[0:64], values)
                    T.copy(values, O[vid, 1, :])

    return _compile_device_kernel(main.with_attr("tir.noalias", True))


def test_dynamic_full_and_tail_copy_preserves_vector_and_mte3_order(dynamic_tail_copy_kernel):
    # The initial store must complete before the helper's whole-tile Duplicate.
    # Alternating full/tail also exercises the path that skips hidden V events.
    for count in (64, 33, 1, 32, 63, 64, 7):
        source = torch.arange(count, dtype=torch.float32) + 17
        output = torch.full((2, 2, 64), -999.0, dtype=torch.float32, device="npu")
        for _invocation in range(3):
            dynamic_tail_copy_kernel(source.npu(), output)
            expected = torch.zeros((2, 2, 64), dtype=torch.float32)
            expected[:, 0, :] = -7
            expected[:, 1, :count] = source
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_dynamic_empty_columns_keep_nonzero_padding_and_completion():
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), Count: T.Tensor((1,), "int32"), O: T.Tensor((2, 2, 64), "float32")):
        with T.Kernel(1, is_npu=True) as (_cid, vid):
            values = T.alloc_ub((64,), "float32")
            with T.Scope("V"):
                count = Count[0]
                if count >= 0 and count <= 64:
                    T.tile.fill(values, -7)
                    T.copy(values, O[vid, 0, :])
                    T.copy(A[0:count], values, pad_value=5)
                    T.copy(values, O[vid, 1, :])

    kernel = _compile_device_kernel(main.with_attr("tir.noalias", True))
    source = torch.arange(64, dtype=torch.float32) + 17
    for count in (64, 0, 33, 1, 0, 32, 63, 64, 7):
        expected = torch.full((2, 2, 64), 5.0, dtype=torch.float32)
        expected[:, 0, :] = -7
        expected[:, 1, :count] = source[:count]
        for _invocation in range(3):
            output = torch.full(expected.shape, -999.0, dtype=torch.float32, device="npu")
            kernel(source.npu(), torch.tensor([count], dtype=torch.int32, device="npu"), output)
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_empty_vector_subtile_preserves_padding_and_prior_mte3_reads():
    @T.prim_func
    def main(A: T.Tensor((34, 8), "float32"), O: T.Tensor((2, 2, 2, 16, 8), "float32")):
        with T.Kernel(2, is_npu=True) as (cid, vid):
            values = T.alloc_ub((16, 8), "float32")
            with T.Scope("V"):
                T.tile.fill(values, -7)
                T.copy(values, O[cid, vid, 0, :, :])
                T.copy(A[cid * 32 + vid * 16, 0], values, pad_value=5)
                T.copy(values, O[cid, vid, 1, :, :])

    kernel = _compile_device_kernel(main.with_attr("tir.noalias", True))
    source = torch.arange(34 * 8, dtype=torch.float32).reshape(34, 8)
    expected = torch.full((2, 2, 2, 16, 8), 5, dtype=torch.float32)
    expected[:, :, 0, :, :] = -7
    for cid in range(2):
        for vid in range(2):
            start = cid * 32 + vid * 16
            count = max(0, min(16, 34 - start))
            expected[cid, vid, 1, :count, :] = source[start : start + count, :]
    for _invocation in range(3):
        output = torch.full(expected.shape, -999.0, dtype=torch.float32, device="npu")
        kernel(source.npu(), output)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_real_int32_tail_copy_preserves_values_and_zero_padding():
    # AscendCopy::Lower emits copy_gm_to_ub<int, ...> for this DSL dtype.
    # Its template alias must retain the audited int32 Duplicate contract.
    @T.prim_func
    def main(A: T.Tensor((33,), "int32"), O: T.Tensor((2, 64), "int32")):
        with T.Kernel(1, is_npu=True) as (_cid, vid):
            values = T.alloc_ub((64,), "int32")
            with T.Scope("V"):
                T.copy(A[0:64], values)
                T.copy(values, O[vid, :])

    kernel = _compile_device_kernel(main.with_attr("tir.noalias", True))
    source = torch.arange(33, dtype=torch.int32) - 17
    expected = torch.zeros((2, 64), dtype=torch.int32)
    expected[:, :33] = source
    for _invocation in range(3):
        output = torch.full((2, 64), -999, dtype=torch.int32, device="npu")
        kernel(source.npu(), output)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
