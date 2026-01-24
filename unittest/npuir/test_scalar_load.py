# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import filecmp

import tilelang
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
            # min(block_N, shape - cid * block_N)
            t0 = cid * block_N
            t0 = shape - t0
            tail_size = T.min(block_N, t0)

            a = A[cid * block_N]
            b = B[cid * block_N]
            c = a + b
            C[cid * block_N] = c

    return main

def test_scalar_load():
    func = impl(N, 1024)
    kernel = tilelang.engine.lower(func)
    # print(kernel)

    curr_name = os.path.splitext(os.path.basename(__file__))[0][5:] + ".mlir"
    # Export to .mlir file
    output_file = './output/' + curr_name
    with open(output_file, 'w') as f:
        f.write(kernel)
    
    ref_file = "./mlir_files/" + curr_name
    # filecmp.cmp returns True if files are identical, False otherwise
    are_identical = filecmp.cmp(output_file, ref_file , shallow=False)
    # assertion for pytest
    assert are_identical, f"'{output_file}' and '{ref_file}' are not identical"

if __name__ == "__main__":
    test_scalar_load()
