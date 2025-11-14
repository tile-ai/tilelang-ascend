import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
args = parser.parse_args()

M = 2
N = 512
block_M = 2
block_N = 128
dataBlockHalfNum = 16
mask = 128
repeat = 1
dstRepStride = 1
srcBlkStride = 1
srcRepStride = 8


@tilelang.jit(out_idx=[-1])
def block_reduce_sum(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N // dataBlockNum), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N // dataBlockNum), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

                T.barrier_all()
                T.block_reduce_sum(b_ub, a_ub, repeat, mask, dstRepStride, srcBlkStride, srcRepStride)
                T.barrier_all()

                T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N // dataBlockNum])

    return main


func = block_reduce_sum(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockHalfNum)

torch.manual_seed(0)

a = torch.randn(M, N, dtype=torch.float16).npu()

torch.npu.synchronize()
print("init successful!")

b = func(a)

num_groups = M * N // dataBlockHalfNum
ref_b = torch.zeros((1, num_groups)).to(torch.float16)
a_flag = a.reshape(-1)
for i in range(num_groups):
    start = i * dataBlockHalfNum
    end = start + dataBlockHalfNum
    group = a_flag[start:end]
    sum_val = torch.sum(group).item()
    ref_b[0, i] = sum_val
ref_b = ref_b.reshape(M, N // dataBlockHalfNum)
ref_b = ref_b.npu().to(dtype=torch.float16)

torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")