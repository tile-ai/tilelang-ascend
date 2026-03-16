# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest

import tilelang

from testcommon import npuir_compile_to_bin

pytestmark = [pytest.mark.mode("Developer")]
import tilelang.language as T


tilelang.cache.clear_cache()

dtype = "float32"


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

            for i, j in T.Parallel(block_M, block_N // 4):
                bx = blockx * block_M + i
                by = blocky * block_N + j * 4
                T.copy(A[bx : bx + 1, by : by + 4], A_VEC[0:1, 0:4])
                T.npuir_atomic_addx4(A_VEC, B[bx, by], [1, 4])

    return atomicAddx4Program


def test_vec_atomic_addx4():
    M, N = 256, 256
    func = run_atomic_addx4(M, N, block_M=16, block_N=16)
    kernel = tilelang.engine.lower(func, target="npuir")
    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, (
        "npuir compile failed or returned empty"
    )


if __name__ == "__main__":
    test_vec_atomic_addx4()
