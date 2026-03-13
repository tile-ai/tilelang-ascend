# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("copy"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["bfloat16"]
COPY_1D_CASES = [(1024, 256)]


@tilelang.jit(target="npuir")
def copy_1d_demo(L, block_L, dtype="bfloat16", calc_type="float32"):
    @T.prim_func
    def simple_copy_1d(
        In: T.Tensor((L,), dtype),
        A: T.Tensor((L,), dtype),
        B: T.Tensor((L,), dtype),
        C: T.Tensor((L,), calc_type),
    ):
        with T.Kernel(T.ceildiv(L, block_L), is_npu=True) as (cid, _):
            start_idx = cid * block_L

            in_frag = T.alloc_fragment((block_L,), dtype)
            A_frag = T.alloc_fragment((block_L,), calc_type)
            B_frag = T.alloc_fragment((block_L,), calc_type)
            out_frag = T.alloc_fragment((block_L,), dtype)

            T.copy(In[start_idx], in_frag)
            T.vcast(in_frag, A_frag, round_mode="rint")
            T.npuir_add(A_frag, A_frag, B_frag)
            T.vcast(B_frag, out_frag, round_mode="rint")

            T.copy(A_frag, A[start_idx])
            T.copy(B_frag, B[start_idx])
            T.copy(out_frag, C[start_idx])

    return simple_copy_1d


@tilelang.jit(target="npuir")
def copy_1d_bf16(L, block_L, dtype="bfloat16"):
    @T.prim_func
    def simple_copy_1d(
        In: T.Tensor((L,), dtype),
        A: T.Tensor((L,), dtype),
        B: T.Tensor((L,), dtype),
        C: T.Tensor((L,), dtype),
    ):
        with T.Kernel(T.ceildiv(L, block_L), is_npu=True) as (cid, _):
            start_idx = cid * block_L

            in_frag = T.alloc_fragment((block_L,), dtype)
            A_frag = T.alloc_fragment((block_L,), dtype)
            B_frag = T.alloc_fragment((block_L,), dtype)
            out_frag = T.alloc_fragment((block_L,), dtype)

            T.copy(In[start_idx], in_frag)
            T.copy(in_frag, A_frag)
            T.npuir_add(A_frag, A_frag, B_frag)
            T.copy(B_frag, out_frag)

            T.copy(A_frag, A[start_idx])
            T.copy(B_frag, B[start_idx])
            T.copy(out_frag, C[start_idx])

    return simple_copy_1d


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("L, block_L", COPY_1D_CASES)
def test_copy_1d_demo(dtype, L, block_L):
    kernel = copy_1d_demo(L, block_L, dtype=dtype)

    input_tensor = gen_tensor((L,), dtype, kind="randn")
    a = gen_tensor((L,), dtype, kind="zeros")
    b = gen_tensor((L,), dtype, kind="zeros")
    c = gen_tensor((L,), "float32", kind="zeros")

    kernel(input_tensor, a, b, c)

    expected_a = input_tensor
    expected_b = (input_tensor.to(torch.float32) * 2).to(getattr(torch, dtype))
    expected_c = expected_b.to(torch.float32)

    assert_close(a.cpu(), expected_a.cpu(), dtype=dtype)
    assert_close(b.cpu(), expected_b.cpu(), dtype=dtype)
    assert_close(c.cpu(), expected_c.cpu(), dtype="float32")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("L, block_L", COPY_1D_CASES)
def test_copy_1d_bf16(dtype, L, block_L):
    kernel = copy_1d_bf16(L, block_L, dtype=dtype)

    input_tensor = gen_tensor((L,), dtype, kind="randn")
    a = gen_tensor((L,), dtype, kind="zeros")
    b = gen_tensor((L,), dtype, kind="zeros")
    c = gen_tensor((L,), dtype, kind="zeros")

    kernel(input_tensor, a, b, c)

    expected_a = input_tensor
    expected_b = input_tensor * 2
    expected_c = expected_b

    assert_close(a.cpu(), expected_a.cpu(), dtype=dtype)
    assert_close(b.cpu(), expected_b.cpu(), dtype=dtype)
    assert_close(c.cpu(), expected_c.cpu(), dtype=dtype)
