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
parser.add_argument("--block_M", type=int, default=32, help="")
parser.add_argument("--block_N", type=int, default=32, help="")


def vec_brc_dev(M, N, K, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N
    dtype = "float16"
    BLOCK_SIZE = 20

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((1, block_N), dtype),
            C: T.Tensor((M, block_N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            B_VEC = T.alloc_shared((1, block_N), dtype)
            C_VEC = T.alloc_shared((M, block_N), dtype)
            T.copy(B,B_VEC)
            T.copy(C,C_VEC)
            for i in T.serial(T.ceildiv(m_num*n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    # T.npuir_brc(C_VEC[0,0], A_VEC)
                    # T.npuir_brc(bx+1, A_VEC)
                    # T.npuir_brc(i+1, A_VEC)
                    # T.npuir_brc(B_VEC, A_VEC)
                    T.npuir_brc(C_VEC[0:1,:], A_VEC)
                    T.copy(A_VEC, A[bx, by])

    return main

def vec_brc_exp(M, N, K, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N
    dtype = "float16"
    BLOCK_SIZE = 20

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((1, block_N), dtype),
            C: T.Tensor((M, block_N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((1, block_N), dtype)
            C_VEC = T.alloc_ub((M, block_N), dtype)
            T.copy(B,B_VEC)
            T.copy(C,C_VEC)
            for i in T.serial(T.ceildiv(m_num*n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    # T.npuir_brc(C_VEC[0,0], A_VEC)
                    # T.npuir_brc(bx+1, A_VEC)
                    # T.npuir_brc(i+1, A_VEC)
                    # T.npuir_brc(B_VEC, A_VEC)
                    T.npuir_brc(C_VEC[0:1,:], A_VEC)
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
    if os.environ['TILELANG_ASCEND_MODE'] == 'Dev':
        func = vec_brc_dev(
            main_args.M,
            main_args.N,
            main_args.K,
            main_args.block_M,
            main_args.block_N,
        )
    else:
        func = vec_brc_exp(
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
    b = generate_tensor((1, main_args.block_N), dtype).npu()
    c = generate_tensor((main_args.M, main_args.block_N), dtype).npu()

    compiled_kernel(a,b,c)
    print("====a====")
    print(a)
    print("====b====")
    print(b)
    print("====c====")
    print(c)

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ['TILELANG_ASCEND_MODE'] = 'dev'
    run_test(args)
    os.environ['TILELANG_ASCEND_MODE'] = 'Expert'
    run_test(args)