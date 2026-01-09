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

@tilelang.jit(out_idx=-1, target="pto")
def exp(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            with T.Scope("V"):
                T.tile.exp(a_ub, 10.0)
                T.barrier_all()
                T.copy(a_ub, A[bx * block_M, by * block_N])

    return main

func = exp(M, N, 64, 32)

torch.manual_seed(0)

torch.npu.synchronize()
print("init successful!")

b = func()

ref_b = torch.full((M, N), 10.0).npu()

torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")