# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import torch_npu
import tilelang
import tilelang.language as T
import torch.nn.functional as F
from typing import Tuple, Optional


tilelang.set_log_level("WARNING")

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"

@tilelang.jit(target="npuir")
def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc 
    hc_full = tilelang.cdiv(hc * 4, 32) * 32 // 4
    dtype = FP32
    block_M = 48 
    n_num = tilelang.cdiv(n, block_M) 

    @T.prim_func
    def hc_split_sinkhorn_kernel_(
        mixes: T.Tensor((n, mix_hc), dtype),
        hc_scale: T.Tensor((3,), dtype), 
        hc_base: T.Tensor((mix_hc,), dtype), 
        pre: T.Tensor((n, hc), dtype),
        post: T.Tensor((n, hc), dtype),
        comb: T.Tensor((n, hc, hc), dtype),
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _): 
            mixes_shared = T.alloc_shared((block_M, hc), dtype)
            mixes_shared_2 = T.alloc_shared((block_M, mix_hc - 2 * hc), dtype)
            hc_scaled = T.alloc_shared((3,), dtype)
            hc_based = T.alloc_shared((1, hc), dtype)
            hc_based_2 = T.alloc_shared((1, mix_hc - 2 * hc), dtype)
            pre_ub = T.alloc_shared((block_M, hc), dtype)
            post_ub = T.alloc_shared((block_M, hc), dtype)
            comb_frag = T.alloc_shared((block_M, hc, hc_full), dtype)
            
            BLOCK = T.min(block_M, n - cid * block_M)
            # calculate pre
            T.copy(hc_scale, hc_scaled)
            T.copy(mixes[cid * block_M, 0], mixes_shared, size=[BLOCK, hc])
            T.copy(hc_base[ : hc], hc_based[0, :])
            for i, j in T.Parallel(block_M, hc):
                pre_ub[i, j] = T.sigmoid(mixes_shared[i, j] * hc_scaled[0] + hc_based[0, j]) + eps
            T.copy(pre_ub, pre[cid * block_M , 0], size=[BLOCK, hc])
            
            # calculate post
            T.copy(mixes[cid * block_M , hc], mixes_shared, size=[BLOCK, hc])
            T.copy(hc_base[hc : hc * 2], hc_based[0, :])
            for i, j in T.Parallel(block_M, hc):
                post_ub[i, j] = 2 * T.sigmoid(mixes_shared[i, j] * hc_scaled[1] + hc_based[0, j]) 
            T.copy(post_ub, post[cid * block_M , 0], size=[BLOCK, hc])
            
            # calculate comb
            T.copy(mixes[cid * block_M , hc * 2], mixes_shared_2, size=[BLOCK, mix_hc - 2 * hc])
            T.copy(hc_base[hc * 2 : ], hc_based_2[0, :])
            for k, i, j in T.Parallel(block_M, hc, hc):
                comb_frag[k, i, j] = mixes_shared_2[k, i * hc + j] * hc_scaled[2] + hc_based_2[0, i * hc + j]

            row_sum = T.alloc_shared((block_M, hc, 1), dtype)
            col_sum = T.alloc_shared((block_M, 1, hc_full), dtype)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_shared((block_M, hc, 1), dtype)
            T.reduce_max(comb_frag, row_max, dim=2, size=[BLOCK, hc, hc])
            for k, i, j in T.Parallel(block_M, hc, hc_full):
                comb_frag[k, i, j] = T.exp(comb_frag[k, i, j] - row_max[k, i, 0])
            T.reduce_sum(comb_frag, row_sum, dim=2, size=[BLOCK, hc, hc])
            for k, i, j in T.Parallel(block_M, hc, hc_full):
                comb_frag[k, i, j] = comb_frag[k, i, j] / row_sum[k, i, 0] + eps

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(comb_frag, col_sum, dim=1, size=[BLOCK, hc, hc])
            for k, i, j in T.Parallel(block_M, hc, hc_full):
                comb_frag[k, i, j] = comb_frag[k, i, j] / (col_sum[k, 0, j] + eps)

            for _ in T.serial(sinkhorn_iters - 1):
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(comb_frag, row_sum, dim=2, size=[BLOCK, hc, hc])
                T.npuir_add(row_sum, eps, row_sum)
                T.npuir_div(comb_frag, row_sum, comb_frag)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_frag, col_sum, dim=1, size=[BLOCK, hc, hc])
                T.npuir_add(col_sum, eps, col_sum)
                T.npuir_div(comb_frag, col_sum, comb_frag)

            T.copy(comb_frag[0, 0, 0], comb[cid * block_M, 0, 0], size=[BLOCK, hc, hc])

    return hc_split_sinkhorn_kernel_

def singleton(cls):
    _instances = {}
    def wrapper(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]
    return wrapper

@singleton
class HcKernel:
    def __init__(self, hc_mult, sinkhorn_iters, eps):
        self.kernel = hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
    
    def __call__(self, mixes, hc_scale, hc_base, pre, post, comb):
        self.kernel(mixes, hc_scale, hc_base, pre, post, comb)


def hc_split_sinkhorn(
    mixes: torch.Tensor, 
    hc_scale: torch.Tensor, 
    hc_base: torch.Tensor,
    hc_mult: int = 4, 
    sinkhorn_iters: int = 20, 
    eps: float = 1e-6, 
    n: int = 32
):
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    b, s, _ = mixes.size()
    pre = mixes.new_empty(b, s, hc_mult)
    post = mixes.new_empty(b, s, hc_mult)
    comb = mixes.new_empty(b, s, hc_mult, hc_mult)
    kernel = HcKernel(hc_mult, sinkhorn_iters, eps)
    kernel(mixes.view(-1, (2 + hc_mult) * hc_mult), hc_scale, hc_base, pre, post, comb)
    return pre, post, comb

def hc_split_sinkhorn_torch(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    # mixes: [b, s, mix_hc], hc_scale: [3], hc_base: [mix_hc]
    # mix_hc = (hc + 2) * hc
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))

    pre = (
        F.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0)) + eps
    )
    post = 2 * F.sigmoid(
        post * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult].unsqueeze(0).unsqueeze(0)
    )
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].view(hc_mult, hc_mult).unsqueeze(
        0
    ).unsqueeze(0)

    comb = comb.softmax(-1) + eps
    col_sum = comb.sum(-2, keepdim=True)
    comb = comb / (col_sum + eps)
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(-2, keepdim=True)
        comb = comb / (col_sum + eps)
    return pre, post, comb

def test_hc(hc_mult: int = 4):
    # n = batch_size * seq_len
    dtype = FP32
    mix_hc = (2 + hc_mult) * hc_mult

    print("Start Testing: batch_size = 2, seq_len = 1024")
    batch_size = 2
    seq_len = 1024
    mixes = torch.randn(size=[batch_size, seq_len, mix_hc], dtype=eval("torch." + dtype)).npu()
    hc_scale  = torch.randn(size=[3], dtype=eval("torch." + dtype)).npu()
    hc_base = torch.randn(size=[mix_hc], dtype=eval("torch." + dtype)).npu()
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base)
    pre_cpu, post_cpu, comb_cpu = hc_split_sinkhorn_torch(mixes, hc_scale, hc_base)
    
    torch.testing.assert_close(pre, pre_cpu, rtol=1e-3, atol=1e-3)
    print("\033[92m pre check passed!\033[0m")
    torch.testing.assert_close(post, post_cpu, rtol=1e-3, atol=1e-3)
    print("\033[92m post check passed!\033[0m")
    torch.testing.assert_close(comb, comb_cpu, rtol=1e-3, atol=1e-3)
    print("\033[92m comb check passed!\033[0m")

    print("Start Testing: batch_size = 2, seq_len = 27")
    batch_size = 2
    seq_len = 27
    mixes = torch.randn(size=[batch_size, seq_len, mix_hc], dtype=eval("torch." + dtype)).npu()
    hc_scale  = torch.randn(size=[3], dtype=eval("torch." + dtype)).npu()
    hc_base = torch.randn(size=[mix_hc], dtype=eval("torch." + dtype)).npu()
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base)
    pre_cpu, post_cpu, comb_cpu = hc_split_sinkhorn_torch(mixes, hc_scale, hc_base)
   
    torch.testing.assert_close(pre, pre_cpu, rtol=1e-3, atol=1e-3)
    print("\033[92m pre check passed!\033[0m")
    torch.testing.assert_close(post, post_cpu, rtol=1e-3, atol=1e-3)
    print("\033[92m post check passed!\033[0m")
    torch.testing.assert_close(comb, comb_cpu, rtol=1e-3, atol=1e-3)
    print("\033[92m comb check passed!\033[0m")
    

if __name__ == "__main__":
    torch.manual_seed(888)
    test_hc()