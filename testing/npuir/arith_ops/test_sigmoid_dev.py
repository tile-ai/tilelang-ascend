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
        # ((1, 1), "float16"),    # FIXME: (X, 1) shape will have compilation error in hfusion-flatten-ops pass in NPUIR. Skip them for now.
        ((1, 4), "float16"),
        ((1, 15), "float16"),
        # ((4, 1), "float16"),
        ((4, 4), "float16"),
        ((4, 15), "float16"),
        # ((15, 1), "float16"),
        ((15, 4), "float16"),
        ((15, 15), "float16"),
    ]
)
def sigmoid_case(request):
    return request.param


def sigmoid_kernel(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def sigmoidKernel(src: T.Tensor((M, N), dtype), dst: T.Tensor((M, N), dtype)):
        with T.Kernel(BLOCK_SIZE, is_npu=True):
            src_ub = T.alloc_ub((M, N), dtype)
            dst_ub = T.alloc_ub((M, N), dtype)

            T.copy(src, src_ub)
            T.npuir_sigmoid(src_ub, dst_ub)
            T.copy(dst_ub, dst)

    return sigmoidKernel


def generate_tensor(shape, dtype, clear=False):
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

def test_sigmoid_dev(sigmoid_case):
    shape, dtype = sigmoid_case

    func = sigmoid_kernel(*shape, dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")

    src = generate_tensor(shape, dtype).npu()
    dst = generate_tensor(shape, dtype, clear=True).npu()

    ref = torch.sigmoid(src.cpu())
    compiled_kernel(src, dst)

    assert torch.allclose(dst.cpu(), ref, rtol=1e-3, atol=1e-3)
