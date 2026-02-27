# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os

import tilelang
import tilelang.language as T

import torch
import torch_npu

torch.npu.set_device(0)

tilelang.cache.clear_cache()

M = 1024
N = 1024
K = 512

FFTS_FLAG_THRESHOLD = 15

def minicv(M, N, K, block_M, block_N, dtype="float16", inner_dtype="float32"):
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
                T.npuir_load_nd2nz(B[0, by], B_BUF, [K, tile_size_N])

                C_BUF = T.alloc_L0C((block_M, block_N), inner_dtype)
                T.npuir_dot(A_BUF, B_BUF, C_BUF, [tile_size_M, K, tile_size_N], initC = True)

                with T.rs("PIPE_FIX"):
                    T.sync_block_wait(1)
                    T.npuir_store_fixpipe(C_BUF, C[bx, by], [tile_size_M, tile_size_N], enable_nz2nd = True)
                    T.sync_block_set(0)

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
                    T.copy(C[bx : bx + tile_size_M, by : by + tile_size_N], C_VEC[0:tile_size_M, 0:tile_size_N])
                    T.sync_block_set(1)

                T.npuir_exp(C_VEC, D_VEC)
                T.copy(D_VEC[0:tile_size_M, 0:tile_size_N], D[bx : bx + tile_size_M, by : by + tile_size_N])

    return main

def test_minicv():
    dtype = "float16"
    func = minicv(M, N, K, 128, 256)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = torch.randn(size=[M, K], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[K, N], dtype=eval("torch." + dtype)).npu()
    v3 = torch.zeros(size=[M, N], dtype=eval("torch." + dtype)).npu()
    v4 = torch.zeros(size=[M, N], dtype=eval("torch." + dtype)).npu()

    y_ref = torch.exp(v1 @ v2)
    compiled_kernel(v1, v2, v3, v4, M, N)

    print(y_ref)
    print(v4)
    torch.testing.assert_close(v4, y_ref, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":
    test_minicv()