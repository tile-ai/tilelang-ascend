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
M = 1
N = 1024
block_M = 1
block_N = 1024
firstValue = 0

@tilelang.jit(out_idx=[0])
def createvecindex(M, N, block_M, block_N, firstValue, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 1

    @T.prim_func
    def main(
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):

                T.barrier_all()
                T.createvecindex(c_ub, firstValue)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


func = createvecindex(M, N, block_M, block_N, firstValue)

torch.manual_seed(0)

torch.npu.synchronize()
print("init successful!")

c = func()

ref_c = torch.arange(start = firstValue, end = firstValue + block_N, dtype = torch.int32).reshape(M, N)
ref_c = ref_c.npu()

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
