# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("atomic_add"),
    pytest.mark.mode("Expert"),
]

DTYPES = ["float32"]
ATOMIC_ADD_1D_CASES = [(64, 32)]
ATOMIC_ADD_2D_CASES = [(256, 256, 16, 16)]


def vec_atomic_add_1d(N, block_size, dtype="float32"):
    n_blocks = N // block_size

    @T.prim_func
    def vecAtomicAdd1DExp(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(n_blocks, is_npu=True) as (bid, _):
            A_VEC = T.alloc_ub((block_size,), dtype)
            start = bid * block_size
            t0 = shape - start
            tail_size = T.min(block_size, t0)
            T.copy(A[start : start + tail_size], A_VEC[0:tail_size])
            T.npuir_atomic_add(B[start], A_VEC, [tail_size])

    return vecAtomicAdd1DExp


def vec_atomic_add_2d(M, N, block_M, block_N, dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def vecAtomicAdd2DExp(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        shape_M: T.int32,
        shape_N: T.int32,
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            blockx = cid // n_num
            bx = blockx * block_M
            blocky = cid % n_num
            by = blocky * block_N
            A_VEC = T.alloc_ub((block_M, block_N), dtype)

            t0 = shape_M - bx
            tile_size_M = T.min(block_M, t0)

            t0 = shape_N - by
            tile_size_N = T.min(block_N, t0)
            T.copy(
                A[bx : bx + tile_size_M, by : by + tile_size_N],
                A_VEC[0:tile_size_M, 0:tile_size_N],
            )
            T.npuir_atomic_add(B[bx, by], A_VEC, [tile_size_M, tile_size_N])

    return vecAtomicAdd2DExp


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N, block_size", ATOMIC_ADD_1D_CASES)
def test_vec_atomic_add_1d(dtype, N, block_size):
    A = gen_tensor((N,), dtype, kind="randn")
    B = gen_tensor((N,), dtype, kind="randn")
    expected = A + B

    func = vec_atomic_add_1d(N, block_size=block_size, dtype=dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")
    compiled_kernel(A, B, N)

    assert_close(B.cpu(), expected.cpu(), dtype=dtype, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M, block_N", ATOMIC_ADD_2D_CASES)
def test_vec_atomic_add_2d(dtype, M, N, block_M, block_N):
    A = gen_tensor((M, N), dtype, kind="randn")
    B = gen_tensor((M, N), dtype, kind="randn")
    expected = A + B

    func = vec_atomic_add_2d(M, N, block_M=block_M, block_N=block_N, dtype=dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")
    compiled_kernel(A, B, M, N)

    assert_close(B.cpu(), expected.cpu(), dtype=dtype, rtol=1e-5, atol=1e-8)
