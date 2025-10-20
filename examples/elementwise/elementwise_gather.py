import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=128, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n

def generate_golden(a, b):
    result = torch.zeros(a.size(0), a.size(1), dtype=torch.int32)
    for i in range(a.size(0)):
        tmp_result = torch.zeros(1, a.size(1), dtype=torch.int32)
        for j in range(a.size(1)):
            index = b[i, j].to(torch.int32) / 4 # 4: sizeof(int)
            index = index.long()
            tmp_result[0, j] = a[i, index]
        result[i:] = tmp_result
    return result
        

@tilelang.jit(out_idx=[-1])
def gather(M, N, block_M, block_N, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), "uint32"),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "uint32")
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                T.barrier_all()
                T.gather(c_ub, a_ub, b_ub, 0)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


func = gather(M, N, 16, N)

torch.manual_seed(0)

a = torch.arange(N, dtype=torch.int32).unsqueeze(0).expand(M, -1).npu()
all_multiples = torch.arange(0, 4 * N, 4)  # 4: sizeof(int)
random_indices = torch.randperm(len(all_multiples))[:N]
random_multiples = all_multiples[random_indices].to(torch.uint32)
tmp_tensor = random_multiples.reshape(1, N)
tensor_cpu = tmp_tensor.repeat(M, 1)
b = tensor_cpu.npu()

torch.npu.synchronize()
print("init successful!")

c = func(a, b)

ref_c = generate_golden(a, b).npu()

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
