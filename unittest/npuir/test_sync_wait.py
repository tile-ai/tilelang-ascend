# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import filecmp
 
import tilelang
import tilelang.language as T
 
tilelang.cache.clear_cache()
 
M = 512
N = 512
K = 512
 
def barrier(M, N, K, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
 
    VEC_NUM = 2
    BLOCK_SIZE = 20
 
    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            with T.rs("PIPE_FIX"):
                T.sync_block_wait(1)

            flag_id = cid % 2
            flag_id = flag_id + 5
            with T.rs("PIPE_MTE2"):
                T.sync_block_wait(flag_id)
 
 
    return main
 
def test_barrier():
    func = barrier(M, N, K, 128, 256)
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
    test_barrier()