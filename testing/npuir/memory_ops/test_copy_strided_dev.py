# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
from tilelang import language as T

from testcommon import ascend_mode, assert_close, gen_tensor


def discrete_copy_tiled(total_h, width, stride=2, block_h=32):
    assert total_h % stride == 0
    out_h = total_h // stride
    assert out_h % block_h == 0, "Output height must be divisible by block_h"

    dtype = "float16"

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


@pytest.mark.copy
@pytest.mark.op("copy")
@pytest.mark.dtype("float16")
@pytest.mark.mode("Developer")
def test_copy_strided_tiled_dev():
    H = 1024
    W = 256
    STRIDE = 2
    BLOCK_H = 32

    with ascend_mode("Developer"):
        func = discrete_copy_tiled(H, W, STRIDE, BLOCK_H)
        compiled_kernel = tilelang.compile(func, target="npuir")

        inp = gen_tensor((H, W), "float16", kind="randn")
        out = gen_tensor((H // STRIDE, W), "float16", kind="zeros")

        compiled_kernel(inp, out)

    ref_out = inp[::STRIDE, :].contiguous()
    assert_close(out.cpu(), ref_out.cpu(), dtype="float16", rtol=1e-3, atol=1e-3)
