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


def test_copy_general_reject_explicit_stride_buffer():
    _assert_compile_fails(
        reject_explicit_stride_buffer_kernel,
        "explicit buffer strides are unsupported",
    )
