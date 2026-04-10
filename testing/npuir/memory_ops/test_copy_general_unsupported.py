# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch_npu  # noqa: F401
import tvm

import tilelang
import tilelang.language as T


pytestmark = [
    pytest.mark.op("copy_general"),
    pytest.mark.mode("Developer"),
]

DTYPE = "float16"


@pytest.fixture(autouse=True)
def _force_expert_backend(monkeypatch):
    monkeypatch.setenv("TILELANG_ASCEND_MODE", "expert")


def _compile(func):
    return tilelang.compile(func, target="npuir")


def _assert_compile_fails(func, match):
    with pytest.raises((tvm.error.InternalError, tvm.error.DiagnosticError), match=match):
        _compile(func)


# src.shape = [8, 16, 32], src.range = [:, :, :], src.slice = [8, 16, 32]
# dst.shape = [8, 16], dst.range = [:, :], dst.slice = [8, 16]
# Tests unsupported backbone-count mismatch.
@T.prim_func
def reject_backbone_count_mismatch_kernel(
    A: T.Tensor((8, 16, 32), DTYPE), B: T.Tensor((8, 16, 32), DTYPE)
):
    with T.Kernel(1, is_npu=True):
        UB = T.alloc_ub((8, 16), DTYPE)
        T.copy(A, UB)
        T.copy(UB, B)


# src.shape = [64, 128], src.range = [:, :], src.slice = [64, 128]
# dst.shape = [128, 64], dst.range = [:, :], dst.slice = [128, 64]
# Tests unsupported logical permutation.
@T.prim_func
def reject_permute_kernel(A: T.Tensor((64, 128), DTYPE), B: T.Tensor((64, 128), DTYPE)):
    with T.Kernel(1, is_npu=True):
        UB = T.alloc_ub((128, 64), DTYPE)
        T.copy(A, UB)
        T.copy(UB, B)


# src.shape = [64, 128], src.range = [:, :], src.slice = [64, 128]
# dst.shape = [64, 64], dst.range = [:, :], dst.slice = [64, 64]
# Tests unsupported static backbone mismatch.
@T.prim_func
def reject_static_dim_mismatch_kernel(
    A: T.Tensor((64, 128), DTYPE), B: T.Tensor((64, 128), DTYPE)
):
    with T.Kernel(1, is_npu=True):
        UB = T.alloc_ub((64, 64), DTYPE)
        T.copy(A, UB)
        T.copy(UB, B[:, 0:64])


dynamic_gm_active_dynamic_s = T.symbolic("dynamic_gm_active_dynamic_s")
dynamic_backbone_tail = T.symbolic("dynamic_backbone_tail")


# src.shape = [4, S, 16, 32]
# src.range = [i:i+1, idx1:idx1+8, j:j+1, idx2:idx2+tail], src.slice = [1, 8, 1, tail]
# dst.shape = [1, 8, 1, 16], dst.range = [:, :, :, 0:tail], dst.slice = [1, 8, 1, tail]
# Tests unsupported dynamic GM shape where an active projected stride remains dynamic.
@T.prim_func
def reject_dynamic_gm_active_stride_dynamic_kernel(
    A: T.Tensor((4, dynamic_gm_active_dynamic_s, 16, 32), DTYPE),
    B: T.Tensor((1, 8, 1, 16), DTYPE),
    i: T.int32,
    idx1: T.int32,
    j: T.int32,
    idx2: T.int32,
):
    with T.Kernel(1, is_npu=True):
        tail = T.min(32 - idx2, 16)
        T.copy(A[i : i + 1, idx1 : idx1 + 8, j : j + 1, idx2 : idx2 + tail], B[:, :, :, 0:tail])


# src.shape = [1, 8, tail], src.range = [:, :, :], src.slice = [1, 8, tail]
# dst.shape = [1, 8, 16], dst.range = [:, :, 0:tail], dst.slice = [1, 8, tail]
# Tests the currently unsupported case where a dynamic last-axis GM shape keeps
# projected strides dynamic even though the logical slice shapes align.
@T.prim_func
def reject_dynamic_backbone_tail_kernel(
    A: T.Tensor((1, 8, dynamic_backbone_tail), DTYPE),
    B: T.Tensor((1, 8, 16), DTYPE),
):
    with T.Kernel(1, is_npu=True):
        UB = T.alloc_ub((1, 8, 16), DTYPE)
        T.copy(A, UB)
        T.copy(UB, B)


# src.shape = [64, 128], src.range = [:, :], src.slice = [64, 128]
# dst.shape = [64, 128], dst.range = [:, :], dst.slice = [64, 128]
# Tests the explicitly unsupported contract where the source buffer carries
# explicit TIR strides instead of using the default contiguous layout.
@T.prim_func
def reject_explicit_stride_buffer_kernel(
    A: T.Buffer((64, 128), DTYPE, strides=[256, 1]),
    B: T.Tensor((64, 128), DTYPE),
):
    with T.Kernel(1, is_npu=True):
        UB = T.alloc_ub((64, 128), DTYPE)
        T.copy(A[0:64, 0:128], UB)
        T.copy(UB, B)


def test_copy_general_reject_backbone_count_mismatch():
    _assert_compile_fails(reject_backbone_count_mismatch_kernel, "generic T.copy backbone mismatch")


def test_copy_general_reject_permute():
    _assert_compile_fails(reject_permute_kernel, "generic T.copy backbone dimension mismatch")


def test_copy_general_reject_static_dim_mismatch():
    _assert_compile_fails(reject_static_dim_mismatch_kernel, "generic T.copy backbone dimension mismatch")


def test_copy_general_reject_dynamic_gm_active_stride_dynamic():
    _assert_compile_fails(
        reject_dynamic_gm_active_stride_dynamic_kernel,
        "generic T.copy requires statically-known projected strides",
    )


def test_copy_general_reject_dynamic_backbone_tail():
    _assert_compile_fails(
        reject_dynamic_backbone_tail_kernel,
        "generic T.copy requires statically-known projected strides",
    )


def test_copy_general_reject_explicit_stride_buffer():
    _assert_compile_fails(
        reject_explicit_stride_buffer_kernel,
        "explicit buffer strides are unsupported",
    )
