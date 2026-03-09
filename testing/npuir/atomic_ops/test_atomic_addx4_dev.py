# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("atomic_addx4"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float32"]
ATOMIC_ADDX4_CASES = [(256, 256, 16, 16)]


def run_atomic_addx4(M, N, block_M, block_N, dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def atomicAddx4Program(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        shape_M: T.int32,
        shape_N: T.int32,
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            blockx = cid // n_num
            blocky = cid % n_num
            A_VEC = T.alloc_shared((1, 4), dtype)

            for i in T.serial(block_M):
                for j in T.serial(block_N // 4):
                    bx = blockx * block_M + i
                    by = blocky * block_N + j * 4
                    t0 = shape_M - bx
                    tile_size_M = T.min(block_M, t0)

                    t0 = shape_N - by
                    tile_size_N = T.min(block_N, t0)
                    T.copy(A[bx:bx + 1, by:by + 4], A_VEC[0:1, 0:4])
                    T.npuir_atomic_addx4(B[bx, by], A_VEC, [1, 4])

    return atomicAddx4Program


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M, block_N", ATOMIC_ADDX4_CASES)
def test_vec_atomic_addx4_dev(dtype, M, N, block_M, block_N):
    A = gen_tensor((M, N), dtype, kind="randn")
    B = gen_tensor((M, N), dtype, kind="zeros")
    ref_B = B.clone()

    for i in range(M):
        for j in range(0, N - 3, 4):
            ref_B[i, j] += A[i, j]
            ref_B[i, j + 1] += A[i, j + 1]
            ref_B[i, j + 2] += A[i, j + 2]
            ref_B[i, j + 3] += A[i, j + 3]

    func = run_atomic_addx4(M, N, block_M=block_M, block_N=block_N, dtype=dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")
    compiled_kernel(A, B, M, N)

    assert_close(B.cpu(), ref_B.cpu(), dtype=dtype, rtol=1e-3, atol=1e-3)
