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
        ((4, 64), "float16", 0),
        ((64, 64), "float16", 0),
        ((128, 64), "float32", 0),
        ((64, 256), "float32", 0),
        ((4, 64), "float16", 1),
        ((64, 64), "float16", 1),
        ((128, 64), "float32", 1),
        ((64, 256), "float32", 1),
    ]
)
def flip_case(request):
    return request.param


def flip_kernel(M, N, axis, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            
            A_VEC = T.alloc_shared((M, N), dtype)
            B_VEC = T.alloc_shared((M, N), dtype)

            T.copy(A, A_VEC)
            T.flip(A_VEC, B_VEC, axis)
            T.copy(B_VEC, B)

    return main


def generate_tensor(shape, dtype, clear=False, positive=False):
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16"):
        t = torch.randn(size=shape, dtype=eval("torch." + dtype))
        if positive:
            t = torch.abs(t) + 0.1
        return t
    raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def test_flip_kernel(flip_case):
    shape, dtype, axis = flip_case

    func = flip_kernel(*shape, axis, dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")

    src = generate_tensor(shape, dtype).npu()
    dst = generate_tensor(shape, dtype, clear=True).npu()

    ref = torch.flip(src.cpu(), dims=[axis])
    compiled_kernel(src, dst)

    assert torch.allclose(dst.cpu(), ref, rtol=1e-5, atol=1e-5)
