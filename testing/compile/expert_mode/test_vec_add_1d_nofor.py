# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import pytest

import tilelang

from testcommon import npuir_compile_to_bin

pytestmark = [pytest.mark.mode("Expert")]
import tilelang.language as T

tilelang.cache.clear_cache()

N = 1024

def vec_add_1d(N, block_N, dtype="float32"):
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((N), dtype),
            B: T.Tensor((N), dtype),
            C: T.Tensor((N), dtype),
            shape: T.int32,
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_N), dtype)
            # min(block_N, shape - cid * block_N)
            t0 = cid * block_N
            t0 = shape - t0
            tail_size = T.min(block_N, t0)
            T.copy(A[cid * block_N : cid * block_N + tail_size], A_VEC[0:tail_size])
            T.copy(B[cid * block_N : cid * block_N + tail_size], B_VEC[0:tail_size])

            T.npuir_add(A_VEC, B_VEC, C_VEC)
            T.copy(C_VEC[0:tail_size], C[cid * block_N : cid * block_N + tail_size])

    return main

def test_vec_add_1d():
    func = vec_add_1d(N, 1024)
    kernel = tilelang.engine.lower(func)
    # print(kernel)

    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, "npuir compile failed or returned empty"

if __name__ == "__main__":
    test_vec_add_1d()
