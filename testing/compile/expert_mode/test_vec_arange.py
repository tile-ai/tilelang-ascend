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

def vec_arange(M, N, block_M, block_N, src_dtype="float32", dst_dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dst_dtype),
            B: T.Tensor((M, N), dst_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_ * block_N

            A_VEC = T.alloc_ub((block_M, block_N), dst_dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dst_dtype)
            strides= [1, 2]
            T.npuir_arange(A_VEC, strides, offset=1)
            T.npuir_arange(B_VEC, strides)
            T.copy(A_VEC, A[bx, by])
            T.copy(B_VEC, B[bx, by])

    return main

def test_vec_arange():
    func = vec_arange(M, N, 128, 256)
    kernel = tilelang.engine.lower(func)
    # print(kernel)

    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, "npuir compile failed or returned empty"

if __name__ == "__main__":
    test_vec_arange()
