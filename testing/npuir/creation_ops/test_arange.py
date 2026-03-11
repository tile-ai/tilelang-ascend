import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


pytestmark = [pytest.mark.op("arange")]

DTYPES = ["float16"]
ARANGE_CASES = [(256, 256, 32, 32)]


def arange_demo_dev(M, N, block_M, block_N, dtype="float16"):
    block_size = 1
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def arangeDemoDev(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            a = T.alloc_shared((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num * n_num, block_size)):
                block_id = i * block_size + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.npuir_arange(a, [bx, by], bx)
                    T.copy(a, A[bx:bx + block_M, by:by + block_N])

    return arangeDemoDev


def arange_demo_exp(M, N, block_M, block_N, dtype="float16"):
    block_size = 1
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def arangeDemoExpert(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            a = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num * n_num, block_size)):
                block_id = i * block_size + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.npuir_arange(a, [bx, by], bx)
                    T.copy(a, A[bx:bx + block_M, by:by + block_N])

    return arangeDemoExpert


def tile_arange(A, block_M, block_N):
    M, N = A.shape
    out = torch.empty_like(A)

    for by in range(0, M, block_M):
        for bx in range(0, N, block_N):
            stride_y = by
            stride_x = bx
            offset = by

            m = torch.arange(block_M, device=A.device).view(block_M, 1)
            n = torch.arange(block_N, device=A.device).view(1, block_N)

            block = offset + m * stride_y + n * stride_x
            out[by:by + block_M, bx:bx + block_N] = block.to(dtype=A.dtype)

    return out


@pytest.mark.mode("Developer")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M, block_N", ARANGE_CASES)
def test_arange_dev(dtype, M, N, block_M, block_N):
    kernel = tilelang.compile(arange_demo_dev(M, N, block_M, block_N, dtype), target="npuir")

    a = gen_tensor((M, N), dtype, kind="randn")
    ref = tile_arange(a, block_M, block_N)

    kernel(a)
    assert_close(a.cpu(), ref.cpu(), dtype=dtype, rtol=1e-3, atol=1e-3)


@pytest.mark.mode("Expert")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M, block_N", ARANGE_CASES)
def test_arange_expert(dtype, M, N, block_M, block_N):
    kernel = tilelang.compile(arange_demo_exp(M, N, block_M, block_N, dtype), target="npuir")

    a = gen_tensor((M, N), dtype, kind="randn")
    ref = tile_arange(a, block_M, block_N)

    kernel(a)
    assert_close(a.cpu(), ref.cpu(), dtype=dtype, rtol=1e-3, atol=1e-3)
