import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k


@tilelang.jit(out_idx=[-2])
def matmul_add(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2
    vec_proc = 4

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
            D: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)

            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N // vec_proc), dtype)
            d_ub = T.alloc_ub((block_M // VEC_NUM, block_N // vec_proc), dtype)
            e_ub = T.alloc_ub((block_M // VEC_NUM, block_N // vec_proc), dtype)

            with T.Scope("C"):

                loop_k = T.ceildiv(K, block_K)
                for k in T.Pipelined(loop_k, num_stages=3):
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    T.copy(B[k * block_K, by * block_N], B_L1)

                    T.barrier_all()
                    if k == 0:
                        T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                    else:
                        T.gemm_v0(A_L1, B_L1, C_L0)

                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                T.wait_cross_flag(0)

                for i in T.Pipelined(vec_proc, num_stages=2):
                    T.copy(C[bx * block_M + vid * block_M // VEC_NUM, by * block_N + i * block_N // vec_proc], c_ub)
                    T.copy(D[bx * block_M + vid * block_M // VEC_NUM, by * block_N + i * block_N // vec_proc], d_ub)

                    T.barrier_all()
                    for (j, k) in T.Parallel(block_M // VEC_NUM, block_N // vec_proc):
                        e_ub[j, k] = c_ub[j, k] + d_ub[j, k]
                    T.barrier_all()

                    T.copy(e_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N + i * block_N // vec_proc])
                    T.barrier_all()

    return main


func = matmul_add(M, N, K, 128, 256, 64)

torch.manual_seed(0)

a = torch.randn(M, K).half().npu()
b = torch.randn(K, N).half().npu()
d = torch.randn(M, N).half().npu()
print("init successful!")

c = func(a, b, d)

ref_c = a @ b + d

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
