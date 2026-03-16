# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import pytest

import tilelang

from testcommon import npuir_compile_to_bin

pytestmark = [pytest.mark.mode("Expert")]
import tilelang.language as T

tilelang.cache.clear_cache()

M = 1024
N = 1024
K = 1024

def vec_add_2d_var(M, N, K, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_* block_N

            event_id = cid % 2

            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
            with T.rs("PIPE_MTE2"):
                T.copy(A[bx, by], A_VEC)
                T.copy(B[bx, by], B_VEC)
                T.set_flag("PIPE_V", event_id)

            with T.rs("PIPE_V"):
                T.wait_flag("PIPE_MTE2", event_id)
                T.npuir_add(A_VEC, B_VEC, C_VEC)
                T.set_flag("PIPE_MTE3", event_id)

            with T.rs("PIPE_MTE3"):
                T.wait_flag("PIPE_V", event_id)
                T.copy(C_VEC, C[bx, by])

    return main

def test_vec_add_2d_var():
    func = vec_add_2d_var(M, N, K, 128, 256)
    kernel = tilelang.engine.lower(func)
    # print(kernel)

    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, "npuir compile failed or returned empty"

if __name__ == "__main__":
    test_vec_add_2d_var()
