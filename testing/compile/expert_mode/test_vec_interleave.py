# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import pytest

import tilelang

from testcommon import npuir_compile_to_bin

pytestmark = [pytest.mark.mode("Expert")]
import tilelang.language as T

tilelang.cache.clear_cache()

M = 128
N = 64

def vec_interleave(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    BLOCK_SIZE = 8

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, 2*N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_* block_N

            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, 2*block_N), dtype)
            for i in T.serial(T.ceildiv(m_num*n_num, BLOCK_SIZE)):
                block_id_base = i * BLOCK_SIZE
                block_id = block_id_base + cid
                block_id_m = block_id // n_num
                block_id_n = block_id % n_num
                bx = block_id_m * block_M
                by = block_id_n * block_N
                T.copy(A[bx, by], A_VEC)
                T.copy(B[bx, by], B_VEC)
                T.npuir_interleave(A_VEC, B_VEC, C_VEC)
                T.npuir_deinterleave(C_VEC, A_VEC, B_VEC)
                T.npuir_deinterleave(C_VEC, A_VEC, index_mode = "CHANNEL_0")
                T.npuir_deinterleave(C_VEC, B_VEC, index_mode = "CHANNEL_1")
                T.npuir_deinterleave(C_VEC, A_VEC, B_VEC, index_mode = "ALL_CHANNELS")
                T.copy(C_VEC, C[bx, 2*by])

    return main

def test_vec_interleave():
    func = vec_interleave(M, N, 16, 32)
    kernel = tilelang.engine.lower(func)
    # print(kernel)

    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, "npuir compile failed or returned empty"

if __name__ == "__main__":
    test_vec_interleave()
