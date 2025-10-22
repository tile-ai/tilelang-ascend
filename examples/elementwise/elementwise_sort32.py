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
block_M = 64
block_N = 128  #block_M,block_N与repeatimes有关，repeatimes要求范围[0,255]

@tilelang.jit(out_idx=[-1])
def sort32(M, N, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), "float"),
            B: T.Tensor((M, N), "uint32"),
            C: T.Tensor((M, 2 * N), "float"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N // VEC_NUM), "float")
            b_ub = T.alloc_ub((block_M, block_N // VEC_NUM), "uint32")
            c_ub = T.alloc_ub((block_M, 2 * block_N // VEC_NUM), "float")
            with T.Scope("V"):
                T.copy(A[bx * block_M, by * block_N + vid * block_N // VEC_NUM],a_ub)
                T.copy(B[bx * block_M, by * block_N + vid * block_N // VEC_NUM],b_ub)

                T.barrier_all()
                T.sort32(c_ub, a_ub, b_ub)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M, 2 * (by * block_N + vid * block_N // VEC_NUM)])

    return main


func = sort32(M, N, block_M, block_N)

torch.manual_seed(0)

a = torch.randint(low=1, high=101, size=(M,N), dtype=torch.float).npu()
b = torch.zeros((M,N), dtype=torch.uint32).npu()

torch.npu.synchronize()
print("init successful!")

c = func(a, b)
#计算ref_c
group_size = 32
total_elements = M * N
b_ref = torch.zeros((M,N), dtype=torch.int32, device="npu")
src0_flat = a.flatten()
src1_flat = b_ref.flatten()

total_groups = (total_elements + group_size - 1) // group_size
for i in range(total_groups):
    start = i * group_size
    end = min((i + 1) * group_size, total_elements)

    group_src0 = src0_flat[start:end]
    group_src1 = src1_flat[start:end]
    sorted_indices = torch.argsort(group_src0, descending=True)

    src0_flat[start:end] = group_src0[sorted_indices]
    src1_flat[start:end] = group_src1[sorted_indices]

sorted_src0 = src0_flat.reshape(M, N)
sorted_src1 = src1_flat.reshape(M, N)

ref_c = torch.empty((M, 2 * N), dtype=torch.float)
ref_c[:, ::2] = sorted_src0
ref_c[:, 1::2] = sorted_src1

torch.testing.assert_close(c, ref_c.npu(), rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
