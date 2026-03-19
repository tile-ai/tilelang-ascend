import torch
import torch_npu
import argparse
import tilelang
import tilelang.language as T
import os

import pytest
import testcommon as tc

M, N = 4, 4
DTYPE_CASES = ["float16", "float32"]

def sigmoid_kernel(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(src: T.Tensor((M, N), dtype),
             dst: T.Tensor((M, N), dtype)):
        
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
 
            src_ub = T.alloc_ub((M, N), dtype)
            dst_ub = T.alloc_ub((M, N), dtype)
            
            T.copy(src, src_ub)
            T.npuir_sigmoid(src_ub, dst_ub)
            T.copy(dst_ub, dst)
    
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

@pytest.mark.mode("Developer")
@pytest.mark.op("vsigmoid_dev")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_sigmoid_dev(dtype):
    func = sigmoid_kernel(M, N, dtype)
    compiled_kernel = tilelang.compile(func, target='npuir')

    src = generate_tensor((M, N), dtype).npu()
    dst = generate_tensor((M, N), dtype, clear=True).npu()

    ref = torch.sigmoid(src.cpu())

    compiled_kernel(src, dst)

    tc.assert_close(dst.cpu(), ref, atol=1e-3, rtol=1e-3)

@pytest.mark.mode("Expert")
@pytest.mark.op("vsigmoid_exp")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_sigmoid_exp(dtype):
    func = sigmoid_kernel(M, N, dtype)
    compiled_kernel = tilelang.compile(func, target='npuir')

    src = generate_tensor((M, N), dtype).npu()
    dst = generate_tensor((M, N), dtype, clear=True).npu()

    ref = torch.sigmoid(src.cpu())

    compiled_kernel(src, dst)

    tc.assert_close(dst.cpu(), ref, atol=1e-3, rtol=1e-3)