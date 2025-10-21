import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=64, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n

def generate_golden(a, b, N, max_index):
    result = torch.zeros(1, N, dtype=torch.int16)
    for i in range(max_index):
        start = b[0, i].to(torch.int32) // 2  # 2: sizeof(uint16)
        for j in range(16):  # 16: 8 * 32 // (2 * 8) 一个DataBlock有16个数
            result[0, i * 16 + j] = start + j
    return result

@tilelang.jit(out_idx=[-1])
def gatherb(M, N, block_M, block_N, b_len, repeat_time, dtype="uint16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, b_len), "uint32"),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, b_len), "uint32")
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * b_len], b_ub)

                T.barrier_all()
                T.gatherb(c_ub, a_ub, b_ub, repeat_time, 1, 8)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


max_index = (N * 2 - 1) // 32 + 1
repeat = N * 2 // (32 * 8)
func = gatherb(M, N, 2, N, max_index, repeat)

torch.manual_seed(0)

a = torch.arange(N, dtype=torch.int16).to(torch.uint16).unsqueeze(0).expand(M, -1).npu()
tmp_tensor = torch.zeros(1, max_index, dtype=torch.uint32)
for i in range(max_index):
    tmp_tensor[0, max_index - 1 - i] = i * 32
b = tmp_tensor.expand(M, -1).npu()

torch.npu.synchronize()
print("init successful!")

c = func(a, b)

ref_c = generate_golden(a, b, N, max_index).expand(M, -1).npu()

torch.testing.assert_close(c.to(torch.int16), ref_c.to(torch.int16), rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
