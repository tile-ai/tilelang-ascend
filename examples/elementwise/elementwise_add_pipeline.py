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
def vec_add_pipeline(M, N, block_M, block_N, sub_M, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2
    stages = 2

    @T.macro
    def init_flag():
        T.set_flag("mte3", "mte2", 0)
        T.set_flag("mte3", "mte2", 1)
  
    @T.macro
    def clear_flag():
        T.wait_flag("mte3", "mte2", 0)
        T.wait_flag("mte3", "mte2", 1)

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
          
            vec_proc = block_M // sub_M

            a_ub = T.alloc_ub((stages, sub_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((stages, sub_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((stages, sub_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):
                init_flag()
                T.wait_flag("mte3", "mte2", 0)
                T.copy(A[bx * block_M + vid * sub_M // VEC_NUM + 0 * sub_M, by * block_N], a_ub[0, :, :])
                T.copy(B[bx * block_M + vid * sub_M // VEC_NUM + 0 * sub_M, by * block_N], b_ub[0, :, :])
                T.set_flag("mte2", "v", 0)

                for mm in T.serial(vec_proc):
                    cur = mm % stages
                    nxt = (mm + 1) % stages

                    if mm < vec_proc -1:
                        T.wait_flag("mte3", "mte2", nxt)
                        T.copy(A[bx * block_M + vid * sub_M // VEC_NUM + (mm + 1) * sub_M, by * block_N], a_ub[nxt, :, :])
                        T.copy(B[bx * block_M + vid * sub_M // VEC_NUM + (mm + 1) * sub_M, by * block_N], b_ub[nxt, :, :])
                        T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    for (i, j) in T.Parallel(sub_M // VEC_NUM, block_N):
                        c_ub[cur, i, j] = a_ub[cur, i, j] + b_ub[cur, i, j]
                    T.set_flag("v", "mte3", cur)
                    T.wait_flag("v", "mte3", cur)

                    T.copy(c_ub[cur, :, :], C[bx * block_M + vid * sub_M // VEC_NUM + mm * sub_M, by * block_N])
                    T.set_flag("mte3", "mte2", cur)


                clear_flag()

    return main


func = vec_add_pipeline(M, N, 128, 128, 32)

torch.manual_seed(0)

a = torch.randn(M, N).npu()
b = torch.randn(M, N).npu()

torch.npu.synchronize()
print("init successful!")

c = func(a, b)

torch.npu.synchronize()
torch.set_printoptions(threshold = float('inf'), sci_mode = False)

ref_c = a + b

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
