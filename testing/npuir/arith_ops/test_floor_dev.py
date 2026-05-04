import os

import pytest
import torch
import torch_npu

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()
os.environ["TILELANG_ASCEND_MODE"] = "Developer"


@pytest.fixture(
    params=[
        ((1024, 1024), "float16"),
        ((1024, 4096), "float16"),
        ((1024, 10240), "float32"),
        ((1024, 16384), "float32"),
    ]
)
def floor_case(request):
    return request.param

block_M = 16
block_N = 1024

def floor_kernel(M, N, dtype):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def floorKernel(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = block_M * (cid // n_num)
            by = block_N * (cid % n_num)
            
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            B_VEC = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bx:bx+block_M, by:by+block_N], A_VEC)
            T.npuir_floor(A_VEC, B_VEC)
            T.copy(B_VEC, B[bx:bx+block_M, by:by+block_N])

    return floorKernel


def generate_tensor(shape, dtype, clear=False, positive=False):
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16"):
        t = torch.randn(size=shape, dtype=eval("torch." + dtype))
        if positive:
            t = torch.abs(t) + 0.1
        return t
    raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def test_floor_dev(floor_case):
    shape, dtype = floor_case

    func = floor_kernel(*shape, dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")

    src = generate_tensor(shape, dtype).npu()
    dst = generate_tensor(shape, dtype, clear=True).npu()

    ref = torch.floor(src.cpu())
    compiled_kernel(src, dst)

    assert torch.allclose(dst.cpu(), ref, rtol=1e-5, atol=1e-5)
