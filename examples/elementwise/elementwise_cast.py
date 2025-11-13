import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n

@tilelang.jit(out_idx=[-1])
def cast_tl(M, N, block_M, block_N, mode, count, scale):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 1

    @T.prim_func
    def main(
            A: T.Tensor((M, N), "float"),
            B: T.Tensor((M, N), "float"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "float")
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "float")
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

                T.barrier_all()

                T.cast_tl(b_ub, a_ub, mode, count, scale)

                T.barrier_all()

                T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main

func = cast_tl(M, N, 1024, 1024, "CAST_RINT", 256, 1.0)

torch.manual_seed(0)

# a = torch.randn(M, N).npu()

a = torch.full((M, N), 0.5, dtype=torch.float).npu()
print("------src value--------")
print(a)

torch.npu.synchronize()
print("init successful!")

b = func(a)
print("------ascend c value--------")
print(b)

ref_b = a.to(torch.float).npu()
print("------true value--------")
print(ref_b)

torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
