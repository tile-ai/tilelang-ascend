# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import torch_npu
from typing import Tuple

import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

@tilelang.jit(target="npuir")
def if_have_yield(M, block_N, dtype="float16", indexType="int32"):
    BLOCK_SIZE = 1
    N = T.symbolic("N")
    @T.prim_func
    def if_have_yield_(
        Input: T.Tensor((M, N), dtype),
        Index: T.Tensor((M), indexType),
        Output: T.Tensor((1, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src = T.alloc_shared((1, block_N), dtype)
            dst = T.alloc_shared((1, block_N), dtype)
            idx = T.alloc_shared((M), indexType)
            T.copy(Index, idx)
            n_num = T.ceildiv(N, block_N)

            for idx_n in T.serial(n_num):
                value_zero = 0
                T.npuir_brc(value_zero, dst)
                offset_n = idx_n * block_N
                remain_n = T.min(N - offset_n, block_N)
                for i in T.serial(M):
                    if idx[i] == 1:
                        T.copy(Input[i : i + 1, offset_n : offset_n + remain_n], src)
                        T.npuir_add(src, dst, dst)
                T.copy(dst, Output[0 : 1, offset_n : offset_n + remain_n])
                
    return if_have_yield_


def if_have_yield_torch(Input: torch.Tensor, Index: torch.Tensor, block_N: int) -> torch.Tensor:
    """ Pytorch implement the logic corresponding to if_have_yield_ kernel"""
    M, N = Input.shape
    Output = torch.zeros((1, N), dtype=Input.dtype, device=Input.device)
    n_num = (N + block_N - 1) // block_N
    for idx_n in range(n_num):
        offset_n = idx_n * block_N
        remain_n = min(N - offset_n, block_N)
        dst = torch.zeros(remain_n, dtype=Input.dtype, device=Input.device)
        for i in range(M):
            if Index[i].item() == 1:
                src = Input[i, offset_n:offset_n+remain_n]
                dst = dst + src
        
        Output[0, offset_n:offset_n+remain_n] = dst

    return Output

def if_have_yield_generate(M: int, N: int, block_N: int, dtype: torch.dtype = torch.float16) -> Tuple[torch.Tensor, torch.Tensor]:
    Input = torch.randn(M, N, dtype=dtype).npu()
    Index = torch.randint(0, 2, (M,), dtype=torch.int32).npu()
    return Input, Index

def if_have_yield_test():
    print("=== testing if_have_yield_ kernel ===")
    M, block_N = 20, 32
    if_have_yield_kernel = if_have_yield(M, block_N)
    test_cases =[
        (64),
        (133),
        (397),
        (499)
    ]

    for test_idx, (N) in enumerate(test_cases, 1):
        print(f"\ntest case {test_idx}: N={N}")
        Input, Index = if_have_yield_generate(M, N, block_N)
        Ouput_ref = if_have_yield_torch(Input, Index, block_N)
        Ouput = torch.randn(1, N, dtype=torch.float16).npu()
        if_have_yield_kernel(Input, Index, Ouput)
        print("Ouput: ", Ouput)
        print("Ref: ", Ouput_ref)
        torch.testing.assert_close(Ouput, Ouput_ref, rtol=1e-2, atol=1e-2)
    
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    # Set TileLang developer mode
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    torch.manual_seed(0)
    if_have_yield_test()