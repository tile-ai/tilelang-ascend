import argparse

import tilelang
import tilelang.language as T
import torch
from tilelang.intrinsics import make_zn_layout, make_nz_layout

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=512, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=256, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((block_N, K_L1), dtype)

            T.annotate_layout(
                {
                    A_L1: make_zn_layout(A_L1),
                    B_L1: make_zn_layout(B_L1),
                }
            )
            
            A_L0 = T.alloc_L0A((block_M, K_L1), dtype)
            B_L0 = T.alloc_L0B((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M : (bx + 1) * block_M, 
                    k * K_L1 : (k + 1) * K_L1], A_L1)
                    T.copy(B[by * block_N : (by + 1) * block_N,
                    k * K_L1 : (k + 1) * K_L1], B_L1)

                    T.barrier_all()

                    # T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0), transpose_B=True)
                    
                    # copy L1
                    T.copy(A_L1, A_L0)
                    T.copy(B_L1, B_L0, transpose=True)
                    T.barrier_all()
                    T.mma(A_L0, B_L0, C_L0, init=(k == 0))

                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M : (bx + 1) * block_M, 
                    by * block_N : (by + 1) * block_N])

    return main


func = matmul(M, N, K, 256, 128, 64)

torch.manual_seed(42)

a = torch.randn(M, K).half().npu()
b = torch.randn(N, K).half().npu()
c = torch.empty(M, N).half().npu()
print("init successful!")

c = func(a, b)

ref_c = a @ b.T

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
# print(func.get_kernel_source())
