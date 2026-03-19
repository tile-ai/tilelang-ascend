# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import argparse
import torch

import tilelang
import tilelang.language as T

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--M", type=int, default=512, help="")
parser.add_argument("--N", type=int, default=512, help="")
parser.add_argument("--block_M", type=int, default=128, help="")
parser.add_argument("--block_N", type=int, default=128, help="")


@tilelang.jit(out_idx=[-1], target="npuir")
def vec_add(M, N, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N
    dtype = "float16"
    BLOCK_SIZE = 20

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.copy(A[bx, by], A_VEC)
                    T.copy(B[bx, by], B_VEC)
                    T.vadd(A_VEC, B_VEC, C_VEC)
                    T.copy(C_VEC, C[bx, by])

    return main


def run_test(main_args):
    kernel = vec_add(
        main_args.M,
        main_args.N,
        main_args.block_M,
        main_args.block_N,
    )

    shape = [main_args.M, main_args.N]

    torch.manual_seed(88888888)  # set the random seed for torch
    dtype = "float16"

    a = torch.randn(shape, dtype=eval("torch." + dtype), device="npu")
    b = torch.randn(shape, dtype=eval("torch." + dtype), device="npu")

    ref_output = a + b
    c = kernel(a, b)

    torch.testing.assert_close(c, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    torch.npu.set_device(0)
    tilelang.cache.clear_cache()
    args = parser.parse_args()
    run_test(args)
