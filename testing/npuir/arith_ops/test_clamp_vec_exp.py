import torch
import torch_npu
import argparse
import tilelang
import tilelang.language as T
import os

import pytest
import testcommon as tc
# -------------------------
# NPU Configuration
# -------------------------
M, N = 4, 4
CLAMP_MIN = 0.0
CLAMP_MAX = 100.0
DTYPE_CASES = [
    ("float16", "float16"),
    ("float32", "float32"),
]
# -------------------------
# TileLang clamp kernel (support min/max as tensors)
# -------------------------
def clamp_kernel(M, N, min_tensor, max_tensor, dtype, accum_dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def clampVecExpKernel(src: T.Tensor((M, N), dtype),
             dst: T.Tensor((M, N), accum_dtype),
             min_val: T.Tensor((M, N), dtype),
             max_val: T.Tensor((M, N), dtype)):

        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            # Allocate UB memory
            src_ub = T.alloc_ub((M, N), dtype)
            dst_ub = T.alloc_ub((M, N), accum_dtype)
            min_ub = T.alloc_ub((M, N), dtype)
            max_ub = T.alloc_ub((M, N), dtype)

            # Copy from GM to UB
            T.copy(src, src_ub)

            # Clamp operation
            
            T.copy(min_val, min_ub)
            T.copy(max_val, max_ub)
            T.vclamp(src_ub, dst_ub, min_ub, max_ub)
        

            # Copy back from UB to GM
            T.copy(dst_ub, dst)

    return clampVecExpKernel



# -------------------------
# Generate test tensor
# -------------------------
def generate_tensor(shape, dtype, clear=False):
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        return (torch.rand(size=shape, dtype=eval("torch." + dtype)) * 200.0) - 50.0
    raise ValueError(f'Unsupported dtype: {dtype}')

@pytest.mark.mode("Expert")
@pytest.mark.op("vclamp_vec_exp")
@pytest.mark.parametrize("intype, accumtype", DTYPE_CASES)
def test_clamp_vec_exp(intype, accumtype):
    shape = (M, N)
    src = generate_tensor(shape, intype).npu()
    dst = generate_tensor(shape, accumtype, clear=True).npu()
    min_tensor = generate_tensor(shape, intype).npu()
    # Ensure max_tensor > min_tensor by adding a positive offset
    positive_offset = torch.rand(shape, dtype=min_tensor.cpu().dtype) * 50.0 + 1.0
    max_tensor = (min_tensor.cpu() + positive_offset).npu()

    func = clamp_kernel(M, N, 
                        min_tensor=min_tensor, 
                        max_tensor=max_tensor,
                        dtype=intype,
                        accum_dtype=accumtype)
    compiled_kernel = tilelang.compile(func, target='npuir')
    compiled_kernel(src, dst, min_tensor, max_tensor)
    ref = torch.clamp(src.cpu(), min=min_tensor.cpu(), max=max_tensor.cpu())
    tc.assert_close(dst.cpu(), ref, rtol=1e-3, atol=1e-3, equal_nan=True)

