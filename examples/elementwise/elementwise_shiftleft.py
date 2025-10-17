import argparse

import tilelang
import tilelang.language as T
import torch
import random

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n
scalarvalue = random.randint(1, 32)


@tilelang.jit(out_idx=[-1])
def shiftleft(M, N, block_M, block_N, scalarvalue, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

                T.barrier_all()
                T.shiftleft(b_ub, a_ub, scalarvalue)
                T.barrier_all()

                T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


func = shiftleft(M, N, 128, 256, scalarvalue)

torch.manual_seed(0)

a = torch.randint(low=1, high=101, size=(M, N), dtype=torch.int32).npu()

torch.npu.synchronize()
print("init successful!")

b = func(a)

ref_b = pow(2, scalarvalue) * a

torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
