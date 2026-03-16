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
            B: T.Tensor((N), dtype),
            C: T.Tensor((N), dtype),
            shape: T.int32,
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            a = A[cid * block_N]
            b = B[cid * block_N]
            c = a + b
            C[cid * block_N] = c

    return main

def test_scalar_load():
    func = impl(N, 1024)
    kernel = tilelang.engine.lower(func)
    # print(kernel)

    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, "npuir compile failed or returned empty"

if __name__ == "__main__":
    test_scalar_load()
