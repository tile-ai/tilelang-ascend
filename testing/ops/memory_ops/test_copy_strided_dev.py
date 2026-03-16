# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
from tilelang import language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("copy"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16"]
STRIDED_COPY_CASES = [(1024, 256, 2, 32)]


def discrete_copy_tiled(total_h, width, stride=2, block_h=32, dtype="float16"):
    assert total_h % stride == 0
    out_h = total_h // stride
    assert out_h % block_h == 0, "Output height must be divisible by block_h"

    shape_in = [total_h, width]
    shape_out = [out_h, width]

    num_blocks = out_h // block_h

    @T.prim_func
    def discrete_copy_tiled(
        In: T.Tensor(shape_in, dtype),
        Out: T.Tensor(shape_out, dtype),
    ):
        with T.Kernel(1, is_npu=True):
            ub_frag = T.alloc_fragment([block_h, width], dtype)

            for block_idx in T.serial(num_blocks):
                out_block_offset = block_idx * block_h
                in_block_offset = out_block_offset * stride

                for i in T.serial(block_h):
                    row_in = in_block_offset + i * stride
                    T.copy(In[row_in:row_in + 1, 0:width], ub_frag[i:i + 1, 0:width])

                T.copy(ub_frag[0:block_h, 0:width], Out[out_block_offset:out_block_offset + block_h, 0:width])

    return discrete_copy_tiled


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("H, W, stride, block_h", STRIDED_COPY_CASES)
def test_copy_strided_tiled_dev(dtype, H, W, stride, block_h):
    func = discrete_copy_tiled(H, W, stride, block_h, dtype=dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")

    inp = gen_tensor((H, W), dtype, kind="randn")
    out = gen_tensor((H // stride, W), dtype, kind="zeros")

    compiled_kernel(inp, out)

    ref_out = inp[::stride, :].contiguous()
    assert_close(out.cpu(), ref_out.cpu(), dtype=dtype, rtol=1e-3, atol=1e-3)
