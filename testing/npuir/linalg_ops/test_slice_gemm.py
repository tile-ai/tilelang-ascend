# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os

import tilelang
import tilelang.language as T

import torch
import torch_npu

torch.npu.set_device(0)
tilelang.cache.clear_cache()

def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    num_stages,
):
    A_shape = (M, K)
    B_shape = (K, N)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_K, block_N)

    @T.prim_func
    def main(
            A: T.Tensor(A_shape, in_dtype),
            B: T.Tensor(B_shape, in_dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            blockx = cid % T.ceildiv(N, block_N)
            bx = blockx * block_M
            blocky = cid // T.ceildiv(N, block_N)
            by = blocky * block_N
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_shared((block_M, block_N), out_dtype)
            value_zero = 0
            T.npuir_brc(value_zero, C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by, k * block_K], A_shared, size=[block_M,block_K])
                T.copy(B[k * block_K, bx], B_shared, size=[block_K,block_N])
                T.gemm(A_shared[:block_M,:block_K],
                       B_shared[:block_K,:block_N],
                       C_local[:block_M,:block_N])
            T.copy(C_local, C[by, bx], size=[block_M,block_N])
    return main

def test_matmul():
    M,K,N = 256, 512, 256
    in_type = "float16"
    out_type = "float32"
    program = matmul(M=M,
                     N=N,
                     K=K,
                     block_M=32,
                     block_K=32,
                     block_N=32,
                     in_dtype=in_type,
                     out_dtype=out_type,
                     num_stages=0)

    kernel = tilelang.compile(program, target="npuir")

    torch.manual_seed(42)
    A = torch.randn([M, K],dtype=torch.float16).npu()
    B = torch.randn([K, N],dtype=torch.float16).npu()
    C = torch.zeros([M, N],dtype=torch.float32).npu()
    kernel(A, B, C)
    ref_C = torch.matmul(A, B).float()
    torch.testing.assert_close(C, ref_C, rtol=1e-2, atol=1e-2)
    print("test gemm with sliced input success")

if __name__ == "__main__":
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    torch.manual_seed(42) 
    test_matmul()  