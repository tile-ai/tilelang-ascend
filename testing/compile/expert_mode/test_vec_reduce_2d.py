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

def vec_reduce(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, 1), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_ * block_N

            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, 1), dtype)
            T.copy(A[bx, by], A_VEC)
            T.npuir_reduce(A_VEC, B_VEC, [1], "max", [128, 128])
            T.copy(B_VEC[0:128, 0:1], B[bx : bx + 128, 0 : 1])

    return main

def test_vec_reduce():
    func = vec_reduce(M, N, 128, 256)
    kernel = tilelang.engine.lower(func)
    # print(kernel)

    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, "npuir compile failed or returned empty"

if __name__ == "__main__":
    test_vec_reduce()
