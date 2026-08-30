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


def _compile(program, target):
    return tilelang.compile(program, pass_configs=PASS_CONFIGS, target=target)


def _torch_dtype(dtype):
    return {
        "float16": torch.float16,
        "float": torch.float32,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int32": torch.int32,
    }[dtype]


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


def _run_atomic_add_case(program, shape, dtype, num_blocks, target):
    kernel = _compile(program, target)
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
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_tile_atomic_add_1d_accumulates_multiple_blocks_after_zeroing_gm(target, dtype):
    num_blocks = 4
    tile_n = 32
    program = _tile_atomic_add_1d_kernel(
        num_blocks=num_blocks,
        tile_n=tile_n,
        dtype=dtype,
    )
    _run_atomic_add_case(program, (tile_n,), dtype, num_blocks, target)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="tile atomic_add correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_tile_atomic_add_2d_region_accumulates_multiple_blocks_after_zeroing_gm(target):
    num_blocks = 4
    tile_m, tile_n = 4, 32
    dtype = "float32"
    program = _tile_atomic_add_2d_kernel(
        num_blocks=num_blocks,
        tile_m=tile_m,
        tile_n=tile_n,
        dtype=dtype,
    )
    _run_atomic_add_case(program, (tile_m, tile_n), dtype, num_blocks, target)


def _tile_atomic_add_l0c_gemm_kernel(num_blocks=4, block_M=16, block_N=16, block_K=16, dtype="float16", accum_dtype="float", out_dtype=None):
    """Test L0C atomic_add with GEMM.

    ``out_dtype`` is the GM dtype of C; it defaults to ``accum_dtype`` (the L0C
    dtype) so existing callers are unchanged.  Passing a different ``out_dtype``
    exercises the L0C -> GM dtype-converting atomic writeback (e.g. fp32 L0C ->
    bf16 GM).
    """
    if out_dtype is None:
        out_dtype = accum_dtype

    @T.prim_func
    def main(
        A: T.Tensor((block_M, block_K), dtype),  # type: ignore
        B: T.Tensor((block_K, block_N), dtype),  # type: ignore
        C: T.Tensor((block_M, block_N), out_dtype),  # type: ignore
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


def _run_atomic_add_l0c_gemm_case(program, block_M, block_N, block_K, dtype, accum_dtype, num_blocks, target, out_dtype=None):
    if out_dtype is None:
        out_dtype = accum_dtype
    kernel = _compile(program, target)
    torch_dtype = _torch_dtype(dtype)
    torch_out_dtype = _torch_dtype(out_dtype)

    # all-one matrix
    a = torch.ones((block_M, block_K), dtype=torch_dtype, device="npu")
    b = torch.ones((block_K, block_N), dtype=torch_dtype, device="npu")

    c = torch.empty((block_M, block_N), dtype=torch_out_dtype, device="npu")
    c.zero_()
    torch.npu.synchronize()

    kernel(a, b, c)
    torch.npu.synchronize()

    expected_value = num_blocks * block_K  # for every value in c
    expected = torch.full((block_M, block_N), expected_value, dtype=torch_out_dtype, device="npu")

    torch.testing.assert_close(c, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="tile atomic_add correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_tile_atomic_add_l0c_gemm_accumulates_multiple_blocks(target, dtype):
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
    _run_atomic_add_l0c_gemm_case(program, block_M, block_N, block_K, dtype, accum_dtype, num_blocks, target)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="tile atomic_add correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_tile_atomic_add_l0c_fp32_to_bf16_gemm(target, dtype):
    """fp32 L0C -> bf16 GM: SetAtomicAdd() converts the fp32 L0C value to bf16,
    then performs the GM atomic add as bf16 (AscendC mm enAtomic=1 semantics).

    A/B use fp16: fp32 A/B gemm_v0 is independently broken on the ascendc
    backend (garbage L0C even with a plain L0C->GM copy, no atomic_add), which
    is out of scope for this dtype-pair change."""
    num_blocks = 4
    block_M = 16
    block_N = 16
    block_K = 16
    accum_dtype = "float"
    out_dtype = "bfloat16"
    program = _tile_atomic_add_l0c_gemm_kernel(
        num_blocks=num_blocks,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        dtype=dtype,
        accum_dtype=accum_dtype,
        out_dtype=out_dtype,
    )
    _run_atomic_add_l0c_gemm_case(
        program, block_M, block_N, block_K, dtype, accum_dtype, num_blocks,
        target, out_dtype=out_dtype)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="tile atomic_add correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_tile_atomic_add_l0c_int32_to_int32_gemm(target):
    """int8 x int8 GEMM accumulates into an int32 L0C; the int32 -> int32 L0C
    atomic writeback must stay legal (regression: it must not be rejected by
    the L0C dtype-pair check)."""
    if target == "pto":
        # Lowering correctly accepts the int32->int32 pair, but the PTO backend
        # then fails its own tile-layout static_assert (pto_tile.hpp: "Layout
        # rows/cols must be divisible by inner box rows/cols") for the int32
        # accumulator tile at this block size.  That is a PTO tile fractal
        # alignment constraint, unrelated to the atomic_add dtype-pair check
        # (which the ascendc leg below exercises end to end).
        pytest.xfail(
            "PTO backend rejects the int32 L0C tile layout (inner-box "
            "divisibility static_assert); not a lowering dtype-pair issue")
    num_blocks = 4
    block_M = 16
    block_N = 16
    block_K = 32  # int8 fractal needs K >= 32
    dtype = "int8"
    accum_dtype = "int32"
    program = _tile_atomic_add_l0c_gemm_kernel(
        num_blocks=num_blocks,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )
    _run_atomic_add_l0c_gemm_case(program, block_M, block_N, block_K, dtype, accum_dtype, num_blocks, target)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="tile atomic_add correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_tile_atomic_add_l0c_fp32_to_int32_rejected(target):
    """fp32 L0C -> int32 GM is not a supported pair (no Catlass specialization
    and PTO cannot express the float->int conversion atomically); lowering must
    reject it with a clear error."""
    program = _tile_atomic_add_l0c_gemm_kernel(
        num_blocks=4,
        block_M=16,
        block_N=16,
        block_K=16,
        dtype="float16",
        accum_dtype="float",
        out_dtype="int32",
    )
    with pytest.raises(Exception, match="dtype pairs"):
        _compile(program, target)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
