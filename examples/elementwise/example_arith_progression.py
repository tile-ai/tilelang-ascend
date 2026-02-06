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


@tilelang.jit(out_idx=[-1])
def generate_arithmetic_progression(M, N, block_M, block_N, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            output: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num


            seq_ub = T.alloc_ub((block_N,), dtype)
            
            with T.Scope("V"):
                T.tile.arith_progression(seq_ub, 0, 1, block_N)

                for row in range(block_M):
                    for col in range(block_N):
                        if (bx * block_M + row < M) and (by * block_N + col < N):
                            output[bx * block_M + row, by * block_N + col] = seq_ub[col]
    return main


func = generate_arithmetic_progression(M, N, 64, 32)

output = torch.zeros(M, N, dtype=torch.int32).npu()

torch.npu.synchronize()
print("init successful!")

result = func(output)

ref_result = torch.zeros(M, N, dtype=torch.int32).npu()
for i in range(0, M, 64):
    for j in range(0, N, 32):
        for row in range(min(64, M-i)):
            for col in range(min(32, N-j)):
                ref_result[i+row, j+col] = col

torch.testing.assert_close(result, ref_result, rtol=0, atol=0)
print("Kernel Output Match!")