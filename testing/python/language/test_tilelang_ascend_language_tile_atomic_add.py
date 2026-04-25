import pytest
import torch

import tilelang
import tilelang.language as T


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

VEC_NUM = 2


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
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
