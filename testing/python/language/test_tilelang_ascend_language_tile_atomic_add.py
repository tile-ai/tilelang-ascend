import pytest
import torch

import tilelang
import tilelang.language as T


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

VEC_NUM = 2


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


def _compile(program):
    return tilelang.compile(program, pass_configs=PASS_CONFIGS, target="ascendc")


def _torch_dtype(dtype):
    if dtype == "float16":
        return torch.float16
    return torch.float32


def _tile_atomic_add_1d_kernel(num_blocks=4, tile_n=32, dtype="float32"):
    @T.prim_func
    def main(C: T.Tensor((tile_n,), dtype)):  # type: ignore
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((tile_n,), dtype)

            T.tile.fill(src_ub, 1.0)
            T.tile.atomic_add(C[0], src_ub)

    return main


def _tile_atomic_add_2d_kernel(num_blocks=4, tile_m=4, tile_n=32, dtype="float32"):
    @T.prim_func
    def main(C: T.Tensor((tile_m, tile_n), dtype)):  # type: ignore
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((tile_m, tile_n), dtype)

            T.tile.fill(src_ub, 1.0)
            T.tile.atomic_add(C[0, 0], src_ub)

    return main


def _run_atomic_add_case(program, shape, dtype, num_blocks):
    kernel = _compile(program)
    torch_dtype = _torch_dtype(dtype)

    out = torch.empty(shape, dtype=torch_dtype, device="npu")
    out.zero_()
    torch.npu.synchronize()

    kernel(out)
    torch.npu.synchronize()

    expected = torch.full(
        shape,
        num_blocks * VEC_NUM,
        dtype=torch_dtype,
        device="npu",
    )
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="tile atomic_add correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_tile_atomic_add_1d_accumulates_multiple_blocks_after_zeroing_gm(dtype):
    num_blocks = 4
    tile_n = 32
    program = _tile_atomic_add_1d_kernel(
        num_blocks=num_blocks,
        tile_n=tile_n,
        dtype=dtype,
    )
    _run_atomic_add_case(program, (tile_n,), dtype, num_blocks)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="tile atomic_add correctness requires an Ascend NPU runtime",
)
def test_tile_atomic_add_2d_region_accumulates_multiple_blocks_after_zeroing_gm():
    num_blocks = 4
    tile_m, tile_n = 4, 32
    dtype = "float32"
    program = _tile_atomic_add_2d_kernel(
        num_blocks=num_blocks,
        tile_m=tile_m,
        tile_n=tile_n,
        dtype=dtype,
    )
    _run_atomic_add_case(program, (tile_m, tile_n), dtype, num_blocks)


def _tile_atomic_add_l0c_gemm_kernel(num_blocks=4, block_M=16, block_N=16, block_K=16, dtype="float16", accum_dtype="float"):
    """Test L0C atomic_add with GEMM"""

    @T.prim_func
    def main(
        A: T.Tensor((block_M, block_K), dtype),  # type: ignore
        B: T.Tensor((block_K, block_N), dtype),  # type: ignore
        C: T.Tensor((block_M, block_N), accum_dtype),  # type: ignore
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            T.copy(A, A_L1)
            T.copy(B, B_L1)

            T.gemm_v0(A_L1, B_L1, C_L0, init=True)

            T.tile.atomic_add(C, C_L0)

    return main


def _run_atomic_add_l0c_gemm_case(program, block_M, block_N, block_K, dtype, accum_dtype, num_blocks):
    kernel = _compile(program)
    torch_dtype = _torch_dtype(dtype)
    torch_accum_dtype = _torch_dtype(accum_dtype)

    # all-one matrix
    a = torch.ones((block_M, block_K), dtype=torch_dtype, device="npu")
    b = torch.ones((block_K, block_N), dtype=torch_dtype, device="npu")

    c = torch.empty((block_M, block_N), dtype=torch_accum_dtype, device="npu")
    c.zero_()
    torch.npu.synchronize()

    kernel(a, b, c)
    torch.npu.synchronize()

    expected_value = num_blocks * block_K  # for every value in c
    expected = torch.full((block_M, block_N), expected_value, dtype=torch_accum_dtype, device="npu")

    torch.testing.assert_close(c, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="tile atomic_add correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("dtype", ["float16"])
def test_tile_atomic_add_l0c_gemm_accumulates_multiple_blocks(dtype):
    num_blocks = 4
    block_M = 16
    block_N = 16
    block_K = 16
    accum_dtype = "float"

    program = _tile_atomic_add_l0c_gemm_kernel(
        num_blocks=num_blocks,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )
    _run_atomic_add_l0c_gemm_case(program, block_M, block_N, block_K, dtype, accum_dtype, num_blocks)
