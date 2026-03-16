import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()
torch.set_default_device('npu')

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--n", type=int, default=8192, help="Matrix N dimension")
args = parser.parse_args()

@tilelang.jit(out_idx=[-1])
def alloc_var(N, block_N, dtype="int32"):
    VEC_NUM = 2
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
    ):
        with T.Kernel(N // block_N, is_npu=True) as (cid, vid):
            a_ub = T.alloc_shared(block_N // VEC_NUM, dtype)

            flag = T.alloc_var("bool", init=False)
            a = T.alloc_var(dtype, init=1)
            b = T.alloc_var(dtype, init=a)

            T.tile.fill(a_ub, 0.0)
            a_ub[0] = b
            flag = True
            if flag:
                a = 2
                a_ub[1] = a   
            else:
                a_ub[1] = a
                        
            flag = False
            if flag:
                a_ub[2] = a
            else:
                a += 1
                a_ub[2] = a
            T.copy(a_ub, A[cid * block_N + vid * block_N // VEC_NUM])
    return main

N = 32
block_N = 16
func = alloc_var(N, block_N)
code = func.get_kernel_source()


torch.manual_seed(0)


torch.npu.synchronize()
print("init successful!")
a = func()
torch.set_printoptions(threshold=torch.inf)
print(f"b:{a}")
# print(code)
if "flag =" in code and "a =" in code and "b =" in code:
    print("Kernel Output Match!")
else:
    print("T.alloc_var failed")