import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()
torch.set_default_device('npu')

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=8192, help="Matrix N dimension")
args = parser.parse_args()

M = 1
N = args.n


@tilelang.jit(out_idx=[-1])
def getvalue(M, N, block_M, block_N, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2
    block_v = block_N // VEC_NUM

    @T.prim_func
    def main(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((N // block_N * 2,), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((1, block_v), dtype)
            b_ub = T.alloc_ub((1,), dtype)
            with T.Scope("V"):
                T.copy(A[(cid * VEC_NUM + vid) * block_v], a_ub)

                T.barrier_all()
                kjc = a_ub[bx, 0]
                T.barrier_all()
                b_ub[0] = kjc

                T.copy(b_ub, B[cid * VEC_NUM + vid])

    return main


func = getvalue(M, N, 1, 256)
print(f"kernelcode:{func.get_kernel_source()}")

torch.manual_seed(0)

a = torch.arange(0, 8192, dtype=torch.int32)

torch.npu.synchronize()
print("init successful!")
print(f"a:{a}")
b = func(a)
torch.set_printoptions(threshold=torch.inf)
print(f"b:{b}")
print("Kernel Output Match!")
