# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import sys
import os
import argparse
import torch

import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--M", type=int, default=512, help="")
parser.add_argument("--N", type=int, default=512, help="")
parser.add_argument("--K", type=int, default=512, help="")
parser.add_argument("--block_M", type=int, default=128, help="")
parser.add_argument("--block_N", type=int, default=256, help="")


def vec_brc(M, N, K, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N
    dtype = "float16"
    BLOCK_SIZE = 20

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num*n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.copy(A[bx, by], A_VEC)
                    brc_value = 3
                    T.npuir_brc(brc_value, A_VEC)
                    T.copy(A_VEC, A[bx, by])

    return main


def generate_tensor(shape, dtype, clear=False):
    """generate tensor"""
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        return torch.randn(size=shape, dtype=eval("torch." + dtype))
    if dtype in ("int32", "int64", "int16"):
        return torch.randint(low=0, high=2000, size=shape, dtype=eval("torch." + dtype))
    if dtype == "int8":
        return torch.randint(low=0, high=127, size=shape, dtype=eval("torch." + dtype))
    if dtype == "bool":
        return torch.randint(low=0, high=2, size=shape).bool()
    raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def run_test(main_args):
    func = vec_brc(
        main_args.M,
        main_args.N,
        main_args.K,
        main_args.block_M,
        main_args.block_N,
    )
    compiled_kernel = tilelang.compile(func, target='npuir')

    shape = [main_args.M, main_args.K]
    torch.manual_seed(88888888)  # set the random seed for torch
    dtype = "float16"

    a = generate_tensor(shape, dtype).npu()

    ref_output = torch.full(tuple(shape), fill_value=3, dtype=torch.float16, device='npu')
    compiled_kernel(a)
    print("Actual Result:")
    print(a)
    print("Expected Result:")
    print(ref_output)
    torch.testing.assert_close(a, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":
    args = parser.parse_args()
    run_test(args)