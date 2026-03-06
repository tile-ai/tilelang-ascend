# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os

import tilelang
import tilelang.language as T

import torch
import torch_npu

torch.npu.set_device(0)
tilelang.cache.clear_cache()

dtype = "float32"

@tilelang.jit(target="npuir")
def vec_add_2d(block_M, block_N, dtype="float32"):
    M = T.symbolic("M")
    N = T.symbolic("N")

    @T.prim_func
    def vecAdd2D(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            blockx = cid % T.ceildiv(N, block_N)
            bx = blockx * block_M
            blocky = cid // T.ceildiv(N, block_N)
            by = blocky * block_N
            A_VEC = T.alloc_shared([block_M, block_N], dtype)
            B_VEC = T.alloc_shared([block_M, block_N], dtype)
            C_VEC = T.alloc_shared([block_M, block_N], dtype)

            t0 = M - bx
            tile_size_M = T.min(block_M, t0)

            t0 = N - by
            tile_size_N = T.min(block_N, t0)
            T.copy(A[bx:bx + tile_size_M, by:by + tile_size_N], A_VEC[:tile_size_M, :tile_size_N])
            T.copy(B[bx:bx + tile_size_M, by:by + tile_size_N], B_VEC[:tile_size_M, :tile_size_N])
            T.npuir_add(A_VEC, B_VEC, C_VEC)
            T.print(C_VEC[:4,:4])
            T.copy(C_VEC[:tile_size_M, :tile_size_N], C[bx:bx + tile_size_M, by:by + tile_size_N])

    return vecAdd2D


def test_vec_add_2d():

    M, N = 256, 256

    A = torch.ones(size=[M, N], dtype=eval("torch." + dtype)).npu()
    B = torch.ones(size=[M, N], dtype=eval("torch." + dtype)).npu()
    C = torch.zeros(size=[M, N], dtype=eval("torch." + dtype)).npu()
    expected = A + B

    func = vec_add_2d(32, 32)

    print("\nRunning vadd with sliced npuir_print")
    func(A, B, C)

    print(f"All elements equal to expected: {torch.allclose(C, expected)}")
    print("test npuir_print with sliced buffer success")


if __name__ == "__main__":
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    torch.manual_seed(42)
    test_vec_add_2d()
    