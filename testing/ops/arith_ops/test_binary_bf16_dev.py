# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [pytest.mark.op("bf16_binary"), pytest.mark.mode("Developer")]

DTYPES = ["bfloat16"]


@tilelang.jit(target="npuir")
def binary_bf16_kernel(M, N, block_M, block_N, dtype):
    grid_M = (M + block_M - 1) // block_M
    grid_N = (N + block_N - 1) // block_N

    @T.prim_func
    def binary_bf16(
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
            T.copy(B[by : by + block_M, bx : bx + block_N], B_VEC)
            T.vadd(A_VEC, B_VEC, C_VEC)
            T.copy(C_VEC, C[by : by + block_M, bx : bx + block_N])

    return binary_bf16


@tilelang.jit(target="npuir")
def binary_bf16_multi(M, N, block_M, block_N, dtype):
    grid_M = (M + block_M - 1) // block_M
    grid_N = (N + block_N - 1) // block_N

    @T.prim_func
    def binary_bf16_mul(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        D: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_M * grid_N, is_npu=True) as (cid, _):
            blockx = cid % grid_N
            bx = blockx * block_M
            blocky = cid // grid_N
            by = blocky * block_N

            A_VEC = T.alloc_shared([block_M, block_N], dtype)
            B_VEC = T.alloc_shared([block_M, block_N], dtype)
            C_VEC = T.alloc_shared([block_M, block_N], dtype)
            D_VEC = T.alloc_shared([block_M, block_N], dtype)

            T.copy(A[by : by + block_M, bx : bx + block_N], A_VEC)
            T.copy(B[by : by + block_M, bx : bx + block_N], B_VEC)
            T.vadd(A_VEC, B_VEC, C_VEC)
            T.vdiv(B_VEC, A_VEC, D_VEC)
            T.copy(C_VEC, C[by : by + block_M, bx : bx + block_N])
            T.copy(D_VEC, D[by : by + block_M, bx : bx + block_N])

    return binary_bf16_mul


@pytest.mark.parametrize("dtype", DTYPES)
def test_bf16_binary(dtype):
    M, N = 128, 128
    kernel = binary_bf16_kernel(M, N, 32, 32, dtype)

    a = gen_tensor([M, N], dtype, kind="randn")
    b = gen_tensor([M, N], dtype, kind="randn")
    c = gen_tensor([M, N], dtype, kind="zeros")

    kernel(a, b, c)

    expected = a + b

    assert_close(c, expected, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bf16_binary_multi(dtype):
    M, N = 128, 128
    kernel = binary_bf16_multi(M, N, 32, 32, dtype)

    a = gen_tensor([M, N], dtype, kind="randn")
    b = gen_tensor([M, N], dtype, kind="randn")
    c = gen_tensor([M, N], dtype, kind="zeros")
    d = gen_tensor([M, N], dtype, kind="zeros")

    kernel(a, b, c, d)

    expected_c = a + b
    expected_d = b / a

    assert_close(c, expected_c, dtype=dtype)
    assert_close(d, expected_d, dtype=dtype)
