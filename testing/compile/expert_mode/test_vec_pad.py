# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest

import tilelang

from testcommon import npuir_compile_to_bin

pytestmark = [pytest.mark.mode("Expert")]
import tilelang.language as T

tilelang.cache.clear_cache()

M = 64
N = 64


def vec_pad(M, N, block_M, block_N, src_dtype="float32", dst_dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), src_dtype),
        B: T.Tensor((2 * M, 2 * N), dst_dtype),
        C: T.Tensor((M, N), dst_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_ * block_N

            A_VEC = T.alloc_ub((block_M, block_N), src_dtype)
            B_VEC = T.alloc_ub((2 * block_M, block_N), dst_dtype)
            C_VEC = T.alloc_ub((block_M + 2 * m_num * n_num, block_N), dst_dtype)
            T.copy(A[bx, by], A_VEC)
            T.npuir_pad(A_VEC, B_VEC, 0.0, [block_M / 2, 0], [block_M / 2, 0])
            T.npuir_pad(A_VEC, C_VEC, 0.0, [cid, 0], [cid, 0])
            T.copy(B_VEC, B[2 * bx, by])
            T.copy(C_VEC, C[bx, by])

    return main


def test_vec_pad():
    func = vec_pad(M, N, 16, 32)
    kernel = tilelang.engine.lower(func)
    # print(kernel)

    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, (
        "npuir compile failed or returned empty"
    )


if __name__ == "__main__":
    test_vec_pad()
