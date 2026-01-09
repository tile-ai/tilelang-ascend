import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--s", type=float, default=4.0, help="scalar")
args = parser.parse_args()

M = args.m
N = args.n
scalar = args.s


# @tilelang.jit(out_idx=[-1], target="pto")
def vec_add(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M, by * block_N], a_ub)
                T.barrier_all()
                T.tile.mul(b_ub, a_ub, scalar)
                T.barrier_all()

                T.copy(b_ub, B[bx * block_M, by * block_N])

    return main

func = vec_add(M, N, 64, 32)

kernel = tilelang.engine.lower(func, target="pto")
print(kernel.kernel_source)