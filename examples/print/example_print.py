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
N = 16

block_M = 2
block_N = 16


@tilelang.jit(out_idx=[-1])
def and_tl(M, N, block_M, block_N, dtype="int16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        T.printf("===========A:\n")
        T.dump_tensor(A, 111, M * N, (M, N))
        T.printf("===========B:\n")
        T.dump_tensor(B, 111, M * N, (M, N))
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            T.printf("-----cid:%d-------------vid:%d--------------------------------------\n", cid, vid)
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            T.printf("===========original c_ub:\n")
            T.dump_tensor(c_ub, 222, block_M // VEC_NUM * block_N, (block_M // VEC_NUM, block_N))
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.printf("===========a_ub after copy:\n")
                T.dump_tensor(a_ub, 222, block_M // VEC_NUM * block_N, (block_M // VEC_NUM, block_N))
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
                T.barrier_all()
                T.printf("===========b_ub after copy:\n")
                T.dump_tensor(b_ub, 222, block_M // VEC_NUM * block_N, (block_M // VEC_NUM, block_N))
                
                T.barrier_all()
                T.tile.and_tl(c_ub, a_ub, b_ub)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])
                
                T.barrier_all()
                T.printf("===========c_ub after processing:\n")
                T.dump_tensor(c_ub, 222, block_M // VEC_NUM * block_N, (block_M // VEC_NUM, block_N))

        T.printf("===========C:\n")
        T.dump_tensor(C, 111, M * N, (M, N))
    return main


func = and_tl(M, N, block_M, block_N)

a = torch.ones(M, N, dtype=torch.int16).npu()
b = torch.ones(M, N, dtype=torch.int16).npu() * 2

torch.npu.synchronize()
print("init successful!")

c = func(a, b)
print(f"*******c:")
print(c)
ref_c = a & b

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")