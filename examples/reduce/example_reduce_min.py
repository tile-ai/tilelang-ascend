import argparse

import tilelang
from tilelang import language as T
import torch

@tilelang.jit(out_idx=[1])
def reduce_min(M, N, block_M, dtype="float"):
    m_num = M // block_M
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),
        B: T.Tensor([M], dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((sub_block_M, N), dtype)
            b_ub = T.alloc_ub((sub_block_M), dtype)
            tmp = T.alloc_ub((2 * sub_block_M, N), "uint8")

            row_base = cid * block_M + vid * sub_block_M
            with T.Scope("V"):
                T.copy(A[row_base : row_base + sub_block_M, :], a_ub)

                T.barrier_all()
                T.tile.reduce_min(b_ub, a_ub, tmp, dim=-1)
                T.barrier_all()

                T.copy(b_ub, B[row_base : row_base + sub_block_M])

    return main

if __name__ == "__main__":
    tilelang.cache.clear_cache()

    parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
    parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=256, help="Matrix N dimension")
    args = parser.parse_args()

    M = args.m
    N = args.n

    func = reduce_min(M, N, 128)

    torch.manual_seed(0)

    a = torch.randn(M, N).npu()

    torch.npu.synchronize()
    print("init successful!")

    c = func(a)

    ref_c = torch.min(a, dim=-1).values

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")