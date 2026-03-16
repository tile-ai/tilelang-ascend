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
K = 512
 
def barrier(M, N, K, block_M, block_N, dtype="float16"):
    BLOCK_SIZE = 20
 
    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            with T.rs("PIPE_FIX"):
                T.sync_block_set(1)
                T.sync_block_wait(1)
                T.subblock_barrier(1)
                T.block_barrier(1)
 
 
    return main
 
def test_barrier():
    func = barrier(M, N, K, 128, 256)
    kernel = tilelang.engine.lower(func)
    # print(kernel)
 
    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, "npuir compile failed or returned empty"
 
if __name__ == "__main__":
    test_barrier()