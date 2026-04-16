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

DTYPES = ["float16"]
COPY_1D_CASES = [(1024, 256)]
COPY_2D_CASES = [(1024, 1024, 128, 128)]
COPY_3D_CASES = [(64, 128, 256, 16, 32, 32)]


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
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (
            cid,
            _,
        ):
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
def simple_copy_3d(
    M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"
):
    @T.prim_func
    def simple_copy_3d(
        In: T.Tensor((M, N, K), dtype),
        A: T.Tensor((M, N, K), dtype),
        B: T.Tensor((M, N, K), dtype),
        C: T.Tensor((M, N, K), accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(M, block_M) * T.ceildiv(N, block_N) * T.ceildiv(K, block_K),
            is_npu=True,
        ) as (cid, _):
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


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("L, block_L", COPY_1D_CASES)
def test_copy_simple_1d_dev(dtype, L, block_L):
    kernel = simple_copy_1d(L, block_L, dtype=dtype)

    input_tensor = gen_tensor((L,), dtype, kind="ones")
    a = gen_tensor((L,), dtype, kind="zeros")
    b = gen_tensor((L,), dtype, kind="zeros")
    c = gen_tensor((L,), "float32", kind="zeros")

    kernel(input_tensor, a, b, c)

    assert_close(a.cpu(), input_tensor.cpu(), dtype=dtype, rtol=1e-5, atol=1e-5)
    assert_close(b.cpu(), (a * 2).cpu(), dtype=dtype, rtol=1e-5, atol=1e-5)
    assert_close(
        c.cpu(), b.to(torch.float32).cpu(), dtype="float32", rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M, block_N", COPY_2D_CASES)
def test_copy_simple_2d_dev(dtype, M, N, block_M, block_N):
    kernel = simple_copy_2d(M, N, block_M, block_N, dtype=dtype)

    input_tensor = gen_tensor((M, N), dtype, kind="randn")
    a = gen_tensor((M, N), dtype, kind="zeros")
    b = gen_tensor((M, N), dtype, kind="zeros")
    c = gen_tensor((M, N), "float32", kind="zeros")

    kernel(input_tensor, a, b, c)

    assert_close(a.cpu(), input_tensor.cpu(), dtype=dtype, rtol=1e-5, atol=1e-5)
    assert_close(b.cpu(), (a * 2).cpu(), dtype=dtype, rtol=1e-5, atol=1e-5)
    assert_close(
        c.cpu(), b.to(torch.float32).cpu(), dtype="float32", rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, K, block_M, block_N, block_K", COPY_3D_CASES)
def test_copy_simple_3d_dev(dtype, M, N, K, block_M, block_N, block_K):
    assert M % block_M == 0, f"M({M}) must be divisible by block_M({block_M})"
    assert N % block_N == 0, f"N({N}) must be divisible by block_N({block_N})"
    assert K % block_K == 0, f"K({K}) must be divisible by block_K({block_K})"

    kernel = simple_copy_3d(M, N, K, block_M, block_N, block_K, dtype=dtype)

    input_tensor = gen_tensor((M, N, K), dtype, kind="randn")
    a = gen_tensor((M, N, K), dtype, kind="zeros")
    b = gen_tensor((M, N, K), dtype, kind="zeros")
    c = gen_tensor((M, N, K), "float32", kind="zeros")

    kernel(input_tensor, a, b, c)

    assert_close(a.cpu(), input_tensor.cpu(), dtype=dtype, rtol=1e-5, atol=1e-5)
    assert_close(b.cpu(), (a * 2).cpu(), dtype=dtype, rtol=1e-5, atol=1e-5)
    assert_close(
        c.cpu(), b.to(torch.float32).cpu(), dtype="float32", rtol=1e-5, atol=1e-5
    )


@tilelang.jit(target="npuir")
def implicit_cast_copy_1d(L, block_L, dtype="float16", mid_dtype="float32"):

    @T.prim_func
    def implicit_cast_copy_1d(
        In: T.Tensor((L,), dtype),
        Out: T.Tensor((L,), dtype),
    ):
        with T.Kernel(T.ceildiv(L, block_L), is_npu=True) as (cid, _):
            start_idx = cid * block_L
            # Alloc UB with mid_dtype (different from dtype)
            A_frag = T.alloc_fragment((block_L,), mid_dtype)
            # GM(dtype) -> UB(mid_dtype) : Implicit Cast during GM2UB
            T.copy(In[start_idx], A_frag)
            # UB(mid_dtype) -> GM(dtype) : Implicit Cast during UB2GM
            T.copy(A_frag, Out[start_idx])

    return implicit_cast_copy_1d


@tilelang.jit(target="npuir")
def implicit_cast_copy_1d_dynamic(L, block_L, dtype="float16", mid_dtype="float32"):

    @T.prim_func
    def implicit_cast_copy_1d_dynamic(
        In: T.Tensor((L,), dtype),
        Out: T.Tensor((L,), dtype),
        shape_L: T.int32,
    ):
        with T.Kernel(T.ceildiv(L, block_L), is_npu=True) as (cid, _):
            start_idx = cid * block_L
            tile_size = T.min(block_L, shape_L - start_idx)

            A_frag = T.alloc_fragment((block_L,), mid_dtype)

            T.copy(In[start_idx : start_idx + tile_size], A_frag[0:tile_size])
            T.copy(A_frag[0:tile_size], Out[start_idx : start_idx + tile_size])

    return implicit_cast_copy_1d_dynamic


@pytest.mark.parametrize(
    "dtype, mid_dtype", [("float16", "float32"), ("float32", "float16")]
)
def test_copy_implicit_cast_dev(dtype, mid_dtype):
    L, block_L = 1024, 256
    kernel = implicit_cast_copy_1d(L, block_L, dtype=dtype, mid_dtype=mid_dtype)

    input_tensor = gen_tensor((L,), dtype, kind="randn")
    output_tensor = gen_tensor((L,), dtype, kind="zeros")

    kernel(input_tensor, output_tensor)

    expected_tensor = input_tensor.to(getattr(torch, mid_dtype)).to(
        getattr(torch, dtype)
    )
    assert_close(output_tensor.cpu(), expected_tensor.cpu(), dtype=dtype)


def test_copy_implicit_cast_dynamic_dev():
    dtype, mid_dtype = "float16", "float32"
    L, block_L = 1000, 256
    kernel = implicit_cast_copy_1d_dynamic(L, block_L, dtype=dtype, mid_dtype=mid_dtype)

    input_tensor = gen_tensor((L,), dtype, kind="randn")
    output_tensor = gen_tensor((L,), dtype, kind="zeros")

    kernel(input_tensor, output_tensor, L)

    expected_tensor = input_tensor.to(getattr(torch, mid_dtype)).to(
        getattr(torch, dtype)
    )
    assert_close(output_tensor.cpu(), expected_tensor.cpu(), dtype=dtype)
