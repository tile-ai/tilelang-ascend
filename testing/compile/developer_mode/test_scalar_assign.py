# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest

import tilelang

from testcommon import npuir_compile_to_bin

pytestmark = [pytest.mark.mode("Developer")]
import tilelang.language as T

tilelang.cache.clear_cache()

N = 1024


def impl(N, block_N, dtype="float32"):
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((N), dtype),
        B: T.Tensor((N), "int32"),
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            tmp = 0.0000000001
            tmp1 = 200.0
            tmp2 = tmp + tmp1
            A[cid * block_N] = tmp2
            A[cid * block_N + 1] = 0.0
            tmp3 = 100
            tmp4 = 200
            tmp5 = tmp3 + tmp4
            B[cid * block_N] = tmp5

    return main


def test_scalar_assgin():
    func = impl(N, 1024)
    kernel = tilelang.engine.lower(func)
    # print(kernel)

    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, (
        "npuir compile failed or returned empty"
    )


if __name__ == "__main__":
    test_scalar_assgin()
