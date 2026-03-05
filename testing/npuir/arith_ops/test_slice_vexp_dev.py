# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T

import torch

import pytest
from testcommon import assert_close, gen_tensor

pytestmark = [pytest.mark.mode("Developer")]

DTYPES = ["float32", "float16"]


@tilelang.jit(target="npuir")
def vec_unary_op(block_M, block_N, dtype="float32"):
    M = T.symbolic("M")
    N = T.symbolic("N")

    @T.prim_func
    def vexp_slice(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (
            cid,
            _,
        ):
            blockx = cid % T.ceildiv(N, block_N)
            bx = blockx * block_M
            blocky = cid // T.ceildiv(N, block_N)
            by = blocky * block_N

            A_VEC = T.alloc_shared([block_M // 2, block_N], dtype)
            C_VEC = T.alloc_shared([block_M, block_N], dtype)

            T.copy(A[bx : bx + block_M // 2, by : by + block_N], A_VEC)
            T.vexp(A_VEC, C_VEC[block_M // 2 : block_M, :])

            T.copy(C_VEC[:block_M, :block_N], C[bx : bx + block_M, by : by + block_N])

    return vexp_slice


@pytest.mark.op("vexp_slice")
@pytest.mark.parametrize("dtype", DTYPES)
def test_unary_op(dtype):

    M, N = 32, 32

    A = gen_tensor([M, N], dtype, kind="randn")
    C = gen_tensor([M, N], dtype, kind="zeros")
    expected = torch.exp(A)

    func = vec_unary_op(32, 32, dtype)

    func(A, C)

    exp_slice = expected.cpu()[:16, :]
    c_slice = C.cpu()[16:, :]

    assert_close(c_slice, exp_slice, dtype=dtype)


@tilelang.jit(target="npuir")
def unary_op_broadcast(block_M, block_N, dtype="float32"):
    M = T.symbolic("M")
    N = T.symbolic("N")

    @T.prim_func
    def vexp_broadcast(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (
            cid,
            _,
        ):
            blockx = cid % T.ceildiv(N, block_N)
            bx = blockx * block_M
            blocky = cid // T.ceildiv(N, block_N)
            by = blocky * block_N

            A_VEC = T.alloc_shared([block_M, block_N], dtype)
            C_VEC = T.alloc_shared([block_M, block_N], dtype)

            T.copy(A[bx : bx + block_M, by : by + block_N], A_VEC)
            T.vexp(A_VEC[0:1, :], C_VEC)

            T.copy(C_VEC, C[bx : bx + block_M, by : by + block_N])

    return vexp_broadcast


@pytest.mark.op("vexp_broadcast")
@pytest.mark.parametrize("dtype", DTYPES)
def test_unary_op_broadcast(dtype):

    M, N = 32, 32

    A = gen_tensor([M, N], dtype, kind="ones")
    C = gen_tensor([M, N], dtype, kind="zeros")
    expected = torch.exp(A[0, :].expand(M, N))

    func = unary_op_broadcast(32, 32, dtype)

    func(A, C)

    assert_close(C.cpu(), expected.cpu(), dtype=dtype)
