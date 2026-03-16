# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest

import tilelang

from testcommon import npuir_compile_to_bin

pytestmark = [pytest.mark.mode("Expert")]
import tilelang.language as T

tilelang.cache.clear_cache()

M = 512
N = 512


def vec_gather(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    BLOCK_SIZE = 20

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_ * block_N

            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            indices = T.alloc_ub((block_M, block_N), "int32")
            value_one = 1
            T.npuir_brc(value_one, indices)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id_base = i * BLOCK_SIZE
                block_id = block_id_base + cid
                block_id_m = block_id // n_num
                block_id_n = block_id % n_num
                bx = block_id_m * block_M
                by = block_id_n * block_N
                T.copy(A[bx, by], A_VEC)
                T.npuir_gather(A_VEC, B_VEC, indices)
                T.copy(B_VEC, B[bx, by])

    return main


def test_vec_gather():
    func = vec_gather(M, N, 32, 64)
    kernel = tilelang.engine.lower(func)
    # print(kernel)

    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, (
        "npuir compile failed or returned empty"
    )


if __name__ == "__main__":
    test_vec_gather()
