# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import ascend_mode, assert_close, gen_tensor


@tilelang.jit(target="npuir")
def simple_copy_1d(L, block_L, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def simple_copy_1d(
        In: T.Tensor((L,), dtype),
        A: T.Tensor((L,), dtype),
        B: T.Tensor((L,), dtype),
        C: T.Tensor((L,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(L, block_L), is_npu=True) as (cid, _):
            start_idx = cid * block_L

            A_frag = T.alloc_fragment((block_L,), dtype)
            B_frag = T.alloc_fragment((block_L,), dtype)
            C_frag = T.alloc_fragment((block_L,), accum_dtype)

            T.copy(In[start_idx], A_frag)
            T.copy(A_frag, B_frag)
            T.npuir_add(A_frag, B_frag, B_frag)
            T.copy(B_frag, C_frag)

            T.copy(A_frag, A[start_idx])
            T.copy(B_frag, B[start_idx])
            T.copy(C_frag, C[start_idx])

    return simple_copy_1d


@tilelang.jit(target="npuir")
def simple_copy_2d(M, N, block_M, block_N, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def simple_copy_2d(
        In: T.Tensor((M, N), dtype),
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            A_frag = T.alloc_fragment((block_M, block_N), dtype)
            B_frag = T.alloc_fragment((block_M, block_N), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.copy(In[by * block_M, bx * block_N], A_frag)
            T.copy(A_frag, B_frag)
            T.npuir_add(A_frag, B_frag, B_frag)
            T.copy(B_frag, C_frag)

            T.copy(A_frag, A[by * block_M, bx * block_N])
            T.copy(B_frag, B[by * block_M, bx * block_N])
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return simple_copy_2d


@tilelang.jit(target="npuir")
def simple_copy_3d(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def simple_copy_3d(
        In: T.Tensor((M, N, K), dtype),
        A: T.Tensor((M, N, K), dtype),
        B: T.Tensor((M, N, K), dtype),
        C: T.Tensor((M, N, K), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N) * T.ceildiv(K, block_K), is_npu=True) as (cid, _):
            num_blocks_N = T.ceildiv(N, block_N)
            num_blocks_K = T.ceildiv(K, block_K)

            bk = cid % num_blocks_K
            bn = (cid // num_blocks_K) % num_blocks_N
            bm = cid // (num_blocks_N * num_blocks_K)

            start_m = bm * block_M
            start_n = bn * block_N
            start_k = bk * block_K

            A_frag = T.alloc_fragment((block_M, block_N, block_K), dtype)
            B_frag = T.alloc_fragment((block_M, block_N, block_K), dtype)
            C_frag = T.alloc_fragment((block_M, block_N, block_K), accum_dtype)

            T.copy(In[start_m, start_n, start_k], A_frag)
            T.copy(A_frag, B_frag)
            T.npuir_add(A_frag, B_frag, B_frag)
            T.copy(B_frag, C_frag)

            T.copy(A_frag, A[start_m, start_n, start_k])
            T.copy(B_frag, B[start_m, start_n, start_k])
            T.copy(C_frag, C[start_m, start_n, start_k])

    return simple_copy_3d


@pytest.mark.copy
@pytest.mark.op("copy")
@pytest.mark.dtype("float16")
@pytest.mark.mode("Developer")
def test_copy_simple_1d_dev():
    with ascend_mode("Developer"):
        kernel = simple_copy_1d(1024, 256)

        input_tensor = gen_tensor((1024,), "float16", kind="ones")
        a = gen_tensor((1024,), "float16", kind="zeros")
        b = gen_tensor((1024,), "float16", kind="zeros")
        c = gen_tensor((1024,), "float32", kind="zeros")

        kernel(input_tensor, a, b, c)

    assert_close(a.cpu(), input_tensor.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    assert_close(b.cpu(), (a * 2).cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    assert_close(c.cpu(), b.to(torch.float32).cpu(), dtype="float32", rtol=1e-5, atol=1e-5)


@pytest.mark.copy
@pytest.mark.op("copy")
@pytest.mark.dtype("float16")
@pytest.mark.mode("Developer")
def test_copy_simple_2d_dev():
    with ascend_mode("Developer"):
        kernel = simple_copy_2d(1024, 1024, 128, 128)

        input_tensor = gen_tensor((1024, 1024), "float16", kind="randn")
        a = gen_tensor((1024, 1024), "float16", kind="zeros")
        b = gen_tensor((1024, 1024), "float16", kind="zeros")
        c = gen_tensor((1024, 1024), "float32", kind="zeros")

        kernel(input_tensor, a, b, c)

    assert_close(a.cpu(), input_tensor.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    assert_close(b.cpu(), (a * 2).cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    assert_close(c.cpu(), b.to(torch.float32).cpu(), dtype="float32", rtol=1e-5, atol=1e-5)


@pytest.mark.copy
@pytest.mark.op("copy")
@pytest.mark.dtype("float16")
@pytest.mark.mode("Developer")
def test_copy_simple_3d_dev():
    M, N, K = 64, 128, 256
    block_M, block_N, block_K = 16, 32, 32

    assert M % block_M == 0, f"M({M}) must be divisible by block_M({block_M})"
    assert N % block_N == 0, f"N({N}) must be divisible by block_N({block_N})"
    assert K % block_K == 0, f"K({K}) must be divisible by block_K({block_K})"

    with ascend_mode("Developer"):
        kernel = simple_copy_3d(M, N, K, block_M, block_N, block_K)

        input_tensor = gen_tensor((M, N, K), "float16", kind="randn")
        a = gen_tensor((M, N, K), "float16", kind="zeros")
        b = gen_tensor((M, N, K), "float16", kind="zeros")
        c = gen_tensor((M, N, K), "float32", kind="zeros")

        kernel(input_tensor, a, b, c)

    assert_close(a.cpu(), input_tensor.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    assert_close(b.cpu(), (a * 2).cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    assert_close(c.cpu(), b.to(torch.float32).cpu(), dtype="float32", rtol=1e-5, atol=1e-5)
