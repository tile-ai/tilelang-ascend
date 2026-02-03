# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
from tilelang import language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

def discrete_copy_tiled(total_h, width, stride=2, block_h=32):
    """
    Args:
        total_h: Input height
        width: Input width
        stride: 2 (Take 1 line, skip 1 line)
        block_h: Processing tile size (Common scenario: 32 or 64)
    """
    assert total_h % stride == 0
    out_h = total_h // stride
    assert out_h % block_h == 0, "Output height must be divisible by block_h"
    
    dtype = "float16"
    
    shape_in = [total_h, width]
    shape_out = [out_h, width]
    
    num_blocks = out_h // block_h

    @T.prim_func
    def main(
        In: T.Tensor(shape_in, dtype),
        Out: T.Tensor(shape_out, dtype)
    ):
        with T.Kernel(1, is_npu=True):

            ub_frag = T.alloc_fragment([block_h, width], dtype)
            
            for block_idx in T.serial(num_blocks):
                
                out_block_offset = block_idx * block_h
                in_block_offset = out_block_offset * stride

                for i in T.serial(block_h):
                    row_in = in_block_offset + i * stride
                    T.copy(In[row_in, 0], ub_frag[i, 0], size=[1, width])
                
                T.copy(ub_frag, Out[out_block_offset, 0], size=[block_h, width])

    return main

def run_test():
    print("=" * 40)
    print("Testing Tiled Discrete Copy (Block-based)")
    print("=" * 40)

    H = 1024
    W = 256
    STRIDE = 2
    BLOCK_H = 32
    
    print(f"Config: H={H}, W={W}, Stride={STRIDE}, Block_H={BLOCK_H}")

    func = discrete_copy_tiled(H, W, STRIDE, BLOCK_H)
    compiled_kernel = tilelang.compile(func, target='npuir')
    
    torch.manual_seed(123)
    inp = torch.randn(H, W).npu().half()
    out = torch.zeros(H // STRIDE, W).npu().half()

    compiled_kernel(inp, out)

    ref_out = inp[::STRIDE, :].contiguous()
    
    try:
        torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)
        print("\nInput Slice (Rows 0-3):")
        print(inp[:4, :4])
        print("\nOutput Slice (Rows 0-1, from Inp 0, 2):")
        print(out[:2, :4])
        print("\n\033[92m[Success] Tiled Strided Copy check passed!\033[0m")
    except Exception as e:
        print("\n\033[91m[Failed] Result mismatch!\033[0m")
        print(e)

if __name__ == "__main__":
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'

    run_test()