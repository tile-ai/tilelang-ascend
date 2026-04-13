# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.


import pytest
import torch

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [pytest.mark.op("tir_exp2_scalar_codegen"), pytest.mark.mode("Developer")]


@tilelang.jit(target="npuir")
def tir_exp2_scalar_kernel(M, N, dtype="float16"):
    @T.prim_func
    def tir_exp2_scalar_(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(M, is_npu=True) as (cid, _):
            i = cid
            A_ub = T.alloc_shared((1, N), dtype)
            B_ub = T.alloc_shared((1, N), dtype)

            T.copy(A[i, :], A_ub)
            for j in T.serial(N):
                # Keep scalar expression form so codegen handles tir.exp path.
                B_ub[0, j] = T.exp2(A_ub[0, j])
            T.copy(B_ub, B[i, :])

    return tir_exp2_scalar_


@tilelang.jit(target="npuir")
def tir_exp2_scalar_kernel2(M, N, dtype="float16"):
    @T.prim_func
    def tir_exp2_scalar2_(B: T.Tensor((M, N), dtype)):
        with T.Kernel(M, is_npu=True) as (cid, _):
            i = cid
            B_ub = T.alloc_shared((1, N), dtype)
            for j in T.serial(N):
                # Keep scalar expression form so codegen handles tir.exp path.
                one = 1.0
                B_ub[0, j] = T.exp2(one)
            T.copy(B_ub, B[i, :])

    return tir_exp2_scalar2_


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_tir_exp2_scalar_codegen(dtype):
    M, N = 16, 32
    kernel = tir_exp2_scalar_kernel(M, N, dtype)

    a = gen_tensor((M, N), dtype, kind="randn")
    b = torch.empty_like(a)

    kernel(a, b)

    ref = torch.exp2(a)
    assert_close(b, ref, dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_tir_exp2_scalar_codegen_const(dtype):
    M, N = 16, 32
    kernel = tir_exp2_scalar_kernel2(M, N, dtype)

    a = gen_tensor((M, N), dtype, kind="ones")
    b = torch.empty_like(a)

    kernel(b)

    ref = torch.exp2(a)
    assert_close(b, ref, dtype=dtype, rtol=1e-2, atol=1e-2)
