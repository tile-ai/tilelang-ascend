# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [pytest.mark.op("bf16_unary"), pytest.mark.mode("Developer")]

DTYPES = ["bfloat16"]


@tilelang.jit(target="npuir")
def unary_bf16_kernel(M, N, block_M, block_N, dtype):
    grid_M = (M + block_M - 1) // block_M
    grid_N = (N + block_N - 1) // block_N

    @T.prim_func
    def unary_bf16(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(grid_M * grid_N, is_npu=True) as (cid, _):
            blockx = cid % grid_N
            bx = blockx * block_M
            blocky = cid // grid_N
            by = blocky * block_N

            A_VEC = T.alloc_shared([block_M, block_N], dtype)
            B_VEC = T.alloc_shared([block_M, block_N], dtype)

            T.copy(A[by : by + block_M, bx : bx + block_N], A_VEC)
            T.vabs(A_VEC, B_VEC)
            T.copy(B_VEC, B[by : by + block_M, bx : bx + block_N])

    return unary_bf16


@tilelang.jit(target="npuir")
def unary_bf16_multi(M, N, block_M, block_N, dtype):
    grid_M = (M + block_M - 1) // block_M
    grid_N = (N + block_N - 1) // block_N

    @T.prim_func
    def unary_bf16_multi(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_M * grid_N, is_npu=True) as (cid, _):
            blockx = cid % grid_N
            bx = blockx * block_M
            blocky = cid // grid_N
            by = blocky * block_N

            A_VEC = T.alloc_shared([block_M, block_N], dtype)
            B_VEC = T.alloc_shared([block_M, block_N], dtype)
            C_VEC = T.alloc_shared([block_M, block_N], dtype)

            T.copy(A[by : by + block_M, bx : bx + block_N], A_VEC)
            T.vabs(A_VEC, B_VEC)
            T.vexp(A_VEC, C_VEC)
            T.copy(B_VEC, B[by : by + block_M, bx : bx + block_N])
            T.copy(C_VEC, C[by : by + block_M, bx : bx + block_N])

    return unary_bf16_multi


@pytest.mark.parametrize("dtype", DTYPES)
def test_bf16_unary(dtype):
    M, N = 128, 128
    kernel = unary_bf16_kernel(M, N, 32, 32, dtype)
    A = gen_tensor((M, N), dtype, kind="randn")
    B = torch.empty_like(A)
    kernel(A, B)
    assert_close(B, torch.abs(A))


@pytest.mark.parametrize("dtype", DTYPES)
def test_bf16_unary_multi(dtype):
    M, N = 128, 128
    kernel = unary_bf16_multi(M, N, 32, 32, dtype)
    A = gen_tensor((M, N), dtype, kind="randn")
    B = torch.empty_like(A)
    C = torch.empty_like(A)
    kernel(A, B, C)
    assert_close(B, torch.abs(A))
    assert_close(C, torch.exp(A))
