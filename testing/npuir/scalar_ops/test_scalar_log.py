# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.


import pytest
import torch

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [pytest.mark.op("tir_log_scalar_codegen"), pytest.mark.mode("Developer")]


@tilelang.jit(target="npuir")
def tir_log_scalar_kernel(M, N, dtype="float16"):
    @T.prim_func
    def tir_log_scalar_(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(M, is_npu=True) as (cid, _):
            i = cid
            A_ub = T.alloc_shared((1, N), dtype)
            B_ub = T.alloc_shared((1, N), dtype)

            T.copy(A[i, :], A_ub)
            for j in T.serial(N):
                # Keep scalar expression form so codegen handles tir.exp path.
                B_ub[0, j] = T.log(A_ub[0, j])
            T.copy(B_ub, B[i, :])

    return tir_log_scalar_


@tilelang.jit(target="npuir")
def tir_log_scalar_kernel2(M, N, dtype="float16"):
    @T.prim_func
    def tir_log_scalar2_(B: T.Tensor((M, N), dtype)):
        with T.Kernel(M, is_npu=True) as (cid, _):
            i = cid
            B_ub = T.alloc_shared((1, N), dtype)
            for j in T.serial(N):
                # Keep scalar expression form so codegen handles tir.exp path.
                two = 2.0
                B_ub[0, j] = T.log(two)
            T.copy(B_ub, B[i, :])

    return tir_log_scalar2_


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_tir_log_scalar_codegen(dtype):
    M, N = 16, 32
    kernel = tir_log_scalar_kernel(M, N, dtype)

    a = gen_tensor((M, N), dtype, kind="randn", low=0.1, high=10.0)
    b = torch.empty_like(a)

    kernel(a, b)

    ref = torch.log(a)
    assert_close(b, ref, dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_tir_log_scalar_codegen_const(dtype):
    M, N = 16, 32
    kernel = tir_log_scalar_kernel2(M, N, dtype)

    a = 2 * gen_tensor((M, N), dtype, kind="ones")
    b = torch.empty_like(a)

    kernel(b)

    ref = torch.log(a)
    assert_close(b, ref, dtype=dtype, rtol=1e-2, atol=1e-2)
