import argparse

import tilelang
from tilelang import language as T
import torch

@tilelang.jit(out_idx=[1])
def reduce_max(M, N, block_M, block_N, n, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),
        B: T.Tensor([M], dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            tmp = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.dump_tensor(a_ub, 111, (block_M // VEC_NUM * block_N))

                T.barrier_all()
                T.tile.reduce_max_mask(b_ub, a_ub, tmp, n)
                # T.dump_tensor(b_ub, 111, (block_M // VEC_NUM * block_N))
                # T.barrier_all()

                T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM])

    return main

if __name__ == "__main__":
    tilelang.cache.clear_cache()

    parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
    parser.add_argument("--m", type=int, default=8, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=8, help="Matrix N dimension")
    args = parser.parse_args()

    M = args.m
    N = args.n

    func = reduce_max(M, N, 8, 8, 10)

    torch.manual_seed(0)

    a = torch.tensor(
        [[1,1,1,1,1,1,1,1],
        [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],
        [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],
        [0.5,0.5,0.5,0.5,0.5,0.5,0.5,10]], dtype=torch.float).npu()

    torch.npu.synchronize()
    print("init successful!")

    b = func(a)
    print("b=", b)
    if b[0] == 1:
        print("b Kernel Output Match!")
    
