# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest

import tilelang

from testcommon import npuir_compile_to_bin

pytestmark = [pytest.mark.mode("Expert")]
import tilelang.language as T

tilelang.cache.clear_cache()

M = 256
N = 256


def vec_select(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            Cond_VEC = T.alloc_ub((block_M, block_N), "bool")
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M, by * block_N], A_VEC)
            T.copy(B[bx * block_M, by * block_N], B_VEC)

            T.npuir_cmp(A_VEC, B_VEC, Cond_VEC, "ge")
            T.npuir_select(Cond_VEC, A_VEC, B_VEC, C_VEC)

            T.copy(C_VEC, C[bx * block_M, by * block_N])

    return main


def test_vec_select():
    func = vec_select(M, N, 32, 64)
    kernel = tilelang.engine.lower(func)
    # print(kernel)

    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, (
        "npuir compile failed or returned empty"
    )


if __name__ == "__main__":
    test_vec_select()
