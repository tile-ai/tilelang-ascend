# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import filecmp

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

M = 1024
N = 1024
K = 512

FFTS_FLAG_THRESHOLD = 15

def sync_op(M, N, K, block_M, block_N, dtype="float16", inner_dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
            D: T.Tensor((M, N), dtype),
            shape_M: T.int32, shape_N: T.int32,
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, subid):
            with T.Scope("Cube"):
                blockx = cid // n_num
                bx = blockx * block_M
                blocky = cid % n_num
                by = blocky * block_N

                A_BUF = T.alloc_L1((block_M, K), dtype)
                B_BUF = T.alloc_L1((K, block_N), dtype)

                # min(block_M, shape_M - cid // n_num * block_M)
                t0 = shape_M - bx
                tile_size_M = T.min(block_M, t0)
                # min(block_N, shape_N - cid % n_num * block_N)
                t0 = shape_N - by
                tile_size_N = T.min(block_N, t0)

                T.npuir_load_nd2nz(A[bx, 0], A_BUF, [tile_size_M, K])
                T.pipe_barrier("PIPE_MTE2")
                T.npuir_load_nd2nz(B[0, by], B_BUF, [K, tile_size_N])

                C_BUF = T.alloc_L0C((block_M, block_N), inner_dtype)
                T.npuir_dot(A_BUF, B_BUF, C_BUF, [tile_size_M, K, tile_size_N], initC = True)

                with T.rs("PIPE_FIX"):
                    T.sync_block_wait(1)
                    T.npuir_store_fixpipe(C_BUF, C[bx, by], [tile_size_M, tile_size_N], enable_nz2nd = True)
                    T.sync_block_set(0)
                    T.block_barrier(0)

                with T.rs("PIPE_FIX"):
                    for i in range(0, FFTS_FLAG_THRESHOLD):
                        T.sync_block_wait(1)

            with T.Scope("Vector"):
                # two vector cores consume (block_M, block_N) together.
                # fixme: block_M % 2 == 0. Odd number is not handled yet in usercase.
                blockx = cid // n_num
                bx = blockx * block_M
                subblock_M = subid * (block_M // 2)
                bx = bx + subblock_M
                blocky = cid % n_num
                by = blocky * block_N

                with T.rs("PIPE_MTE2"):
                    for i in range(0, FFTS_FLAG_THRESHOLD):
                        T.sync_block_set(1)

                C_VEC = T.alloc_ub((block_M // 2, block_N), dtype)
                D_VEC = T.alloc_ub((block_M // 2, block_N), dtype)

                # min(subblock_M, shape_M - cid * block_M - subblock_M)
                t0 = shape_M - bx
                tile_size_M = T.min(block_M//2, t0)
                # min(block_N, shape_N - cid * block_N)
                t0 = shape_N - by
                tile_size_N = T.min(block_N, t0)

                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(0)
                    T.copy(C[bx, by], C_VEC, [tile_size_M, tile_size_N])
                    T.sync_block_set(1)
                    T.pipe_barrier("PIPE_MTE2")
                    T.subblock_barrier(0)

                T.npuir_exp(C_VEC, D_VEC)
                T.copy(D_VEC, D[bx, by], [tile_size_M, tile_size_N])

    return main

def test_sync_op():
    func = sync_op(M, N, K, 128, 256)
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
    test_sync_op()
