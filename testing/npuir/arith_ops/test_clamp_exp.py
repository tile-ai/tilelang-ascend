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
# TileLang clamp kernel
# -------------------------
def clamp_kernel(M, N, dtype, accum_dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def clampExpKernel(src: T.Tensor((M, N), dtype),
             dst: T.Tensor((M, N), accum_dtype)):

        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            # Allocate UB (Unified Buffer) memory
            src_ub = T.alloc_ub((M, N), dtype)
            dst_ub = T.alloc_ub((M, N), accum_dtype)

            # Copy from GM (Global Memory) to UB
            T.copy(src, src_ub)

            # Clamp operation
            T.vclamp(src_ub, dst_ub, CLAMP_MIN, CLAMP_MAX)

            # Copy back from UB to GM
            T.copy(dst_ub, dst)

    return clampExpKernel

# -------------------------
# Generate test data
# -------------------------
def generate_tensor(shape, dtype, clear=False):
    """Generate tensor"""
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        # Input range [-50, 150] to ensure clamp boundaries are triggered
        return (torch.rand(size=shape, dtype=eval("torch." + dtype)) * 200.0) - 50.0
    raise ValueError(f'Unsupported dtype: {dtype}')

@pytest.mark.mode("Expert")
@pytest.mark.op("vclamp_exp")
@pytest.mark.parametrize("intype, accumtype", DTYPE_CASES)
def test_clamp_exp(intype, accumtype):
    func = clamp_kernel(M, N, intype, accumtype)
    compiled_kernel = tilelang.compile(func, target='npuir')

    shape = (M, N)
    src = generate_tensor(shape, intype).npu()
    dst = generate_tensor(shape, accumtype, clear=True).npu()

    compiled_kernel(src, dst)
    ref = torch.clamp(src.cpu(), min=CLAMP_MIN, max=CLAMP_MAX)
    tc.assert_close(dst.cpu(), ref, rtol=1e-3, atol=1e-3, equal_nan=True)