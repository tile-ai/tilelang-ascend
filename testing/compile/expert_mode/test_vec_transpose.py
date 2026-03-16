# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import pytest

import tilelang

from testcommon import npuir_compile_to_bin

pytestmark = [pytest.mark.mode("Expert")]
import tilelang.language as T

tilelang.cache.clear_cache()

M = 512
N = 512

def vec_transpose(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    BLOCK_SIZE = 20

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((N, M), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_* block_N

            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_N, block_M), dtype)
            
            for i in T.serial(T.ceildiv(m_num*n_num, BLOCK_SIZE)):
                block_id_base = i * BLOCK_SIZE
                block_id = block_id_base + cid
                block_id_m = block_id // n_num
                block_id_n = block_id % n_num
                bx = block_id_m * block_M
                by = block_id_n * block_N
                T.copy(A[bx, by], A_VEC)
                T.npuir_transpose(A_VEC, B_VEC, [1, 0])
                T.copy(B_VEC, B[by, bx])

    return main

def test_vec_transpose():
    func = vec_transpose(M, N, 128, 256)
    kernel = tilelang.engine.lower(func)
    # print(kernel)

    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, "npuir compile failed or returned empty"

if __name__ == "__main__":
    test_vec_transpose()
