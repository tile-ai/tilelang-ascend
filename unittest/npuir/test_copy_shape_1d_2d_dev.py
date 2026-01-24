# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T
import filecmp


tilelang.cache.clear_cache()
dtype="float16"

def copy_shape_1d_2d(M, N, block_M, block_N):

    @T.prim_func
    def copyShape(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            shape_M: T.int32, shape_N: T.int32,
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, _):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            A_BUF = T.alloc_shared((block_N), dtype)

            for i in T.Parallel(block_M):
                bx = blockx * block_M + i
                T.copy(A[bx, by], A_BUF) 
                T.copy(A_BUF, B[bx, by])

    return copyShape

def test_copy_shape_1d_2d():
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    # In the futrue, Developer mode and Expert Mode will transition smoothly without
    # requiring explicit declarations.
    M = 256
    N = 1024
    func = copy_shape_1d_2d(M, N, block_M=32, block_N=32)
    kernel = tilelang.engine.lower(func, target='npuir')
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
    test_copy_shape_1d_2d()