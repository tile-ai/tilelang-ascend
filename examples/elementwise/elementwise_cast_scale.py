import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=64, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=64, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n

@tilelang.jit(out_idx=[-1])
def cast_2(M, N, block_M, block_N, mode, count, scale):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 1

    @T.prim_func
    def main(
            A: T.Tensor((M, N), "int32"),
            B: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "int32")
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "float16")
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

                T.barrier_all()

                T.set_deq_scale(scale)

                T.tile.cast(b_ub, a_ub, mode, count)

                T.barrier_all()

                T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main

func_2 = cast_2(M, N, 16, 16, "CAST_RINT", 4096, 1.0)

torch.manual_seed(0)

# should setdeqscale
a2 = torch.full((M, N), 1, dtype=torch.int32).npu()

torch.npu.synchronize()
print("init successful!")

b2 = func_2(a2)

ref_b2 = torch.full((M, N), 1.0, dtype=torch.float16).npu()

torch.testing.assert_close(b2, ref_b2, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
