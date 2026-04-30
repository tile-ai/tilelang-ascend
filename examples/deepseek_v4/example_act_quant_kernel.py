# Copyright (c) Huawei Technologies Co., Ltd. 2026.
import os
import tilelang
import tilelang.language as T
from tilelang import DataType
import torch
import torch_npu
from typing import Tuple, Optional, Literal

tilelang.cache.clear_cache()

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"

@tilelang.jit(target="npuir")
def act_quant_kernel(
    N: int,
    block_M: int = 32, block_N: int = 32,
    round_scale: bool = False
):
    M = T.symbolic("M")
    
    # INT8 quant attributes
    int8_min = -128
    int8_max = 127
    int8_abs_max = 127.0
    
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        X: T.Tensor([M, N], "bfloat16"),      # input BF16
        Y: T.Tensor([M, N], "int8"),         # output INT8
        S: T.Tensor([M, 1], "float32"),      # output scale [M, 1]
    ):
        
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bm = cid // n_num  
            bn = cid % n_num   
            
            x_ub = T.alloc_shared([block_M, block_N], "bfloat16")  
            x_ub_fp = T.alloc_shared([block_M, block_N], "float32")  
            x_ub_half = T.alloc_shared([block_M, block_N], "float16")  
            x_ub_fp_abs = T.alloc_shared([block_M, block_N], "float32") 
            y_ub = T.alloc_shared([block_M, block_N], "int8")     
            
            max_ub = T.alloc_shared([block_M, 1], "float32")         
            scale_ub = T.alloc_shared([block_M, 1], "float32")       
            
            # x_ub_fp_1 = T.alloc_shared([block_M, block_N], "float32")  
            
            T.copy(X[bm * block_M, bn * block_N], x_ub)
            T.vcast(x_ub, x_ub_fp) 
            T.vabs(x_ub_fp, x_ub_fp_abs)
            T.reduce_max(x_ub_fp_abs, max_ub, dim=1)
        
            for i in T.Parallel(block_M):
                scale_ub[i, 0] = max_ub[i, 0] / int8_abs_max
        
            for i, j in T.Parallel(block_M, block_N):
                x_ub_fp[i, j] = x_ub_fp[i, j] / scale_ub[i, 0]

            T.vclamp(x_ub_fp, x_ub_fp, -127.0, 127.0)
            T.vcast(x_ub_fp, x_ub_fp, round_mode="round")

            T.vcast(x_ub_fp, x_ub_half)
            T.vcast(x_ub_half, y_ub)

            T.copy(y_ub, Y[bm * block_M, bn * block_N], size = [block_M, block_N])
            T.copy(scale_ub, S[bm * block_M, 0], size = [block_M, 1])
    
    return main

def act_quant(
    x: torch.Tensor, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"

    N = x.size(-1)
    y = torch.empty_like(x, dtype=torch.int8)
    #s = x.new_empty(*x.size()[:-1], N, dtype=torch.float32)
    s = x.new_empty(N, 1, dtype=torch.float32)
    kernel = act_quant_kernel(N, round_scale=scale_fmt is not None)
    kernel(x.view(-1, N), y.view(-1, N), s.view(-1, N))
    return y, s

def act_quant_torch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of act_quant_kernel.
    
    Args:
        x: Input tensor of shape [M, N], dtype torch.bfloat16.
    
    Returns:
        y: Quantized tensor of shape [M, N], dtype torch.int8.
        s: Scale factors of shape [M, 1], dtype torch.float32.
    """
    # Convert to float32 for accurate arithmetic (kernel does bf16->fp32)
    x_f32 = x.float()                     # [M, N]
    
    # Step 1: per‑row maximum of absolute values
    abs_max = x_f32.abs().max(dim=1, keepdim=True)[0]   # [M, 1]
    # Step 2: compute scale = max / 127.0
    # Avoid division by zero (kernel does not handle zero, but we add safety)
    scale = abs_max / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    # Step 3: scale the row values
    x_scaled = x_f32 / scale              # [M, N]
    # Step 4: clamp to [-127, 127]
    x_clamped = torch.clamp(x_scaled, -127.0, 127.0)
    # Step 5: round to nearest integer
    x_rounded = torch.round(x_clamped)    # float32 with integer values
    # Step 6: convert to int8
    y = x_rounded.to(torch.int8)          # [M, N]
    s = scale                          
    
    return y, s

def test_act_quant():
    M = 64
    N = 32
    dtype = "bfloat16"
    x = torch.randn(size=[M, N], dtype=eval("torch." + dtype)).npu()
    y, s = act_quant(x)
    y_ref, s_ref = act_quant_torch(x)
    print("Start Testing: M = 64, N = 32")
    print("y:", y)
    print("y_ref:", y_ref)
    print("s:", s)
    print("s_ref:", s_ref)
    assert torch.all(y == y_ref)
    torch.testing.assert_close(s, s_ref, atol=1e-3, rtol=1e-3)
    print("Comparison passed.")

if __name__ == "__main__":
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    torch.manual_seed(888)
    test_act_quant()