import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")

M = 16
N = 16


@tilelang.jit(out_idx=[-1])
def transpose(M, N, block_M, block_N, dtype="int16"):

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((M, N), dtype)

            with T.Scope("V"):
                T.copy(A, a_ub)

                T.barrier_all()
                T.transpose(b_ub, a_ub)
                T.barrier_all()

                T.copy(b_ub, B)

    return main


func = transpose(M, N, 16, 16)

torch.manual_seed(0)

a = torch.randn(M, N).npu().to(torch.int16)

torch.npu.synchronize()
print("init successful!")

b = func(a)

ref_b = a.T

torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
