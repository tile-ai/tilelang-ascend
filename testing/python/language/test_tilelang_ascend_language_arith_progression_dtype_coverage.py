"""
T.tile.arith_progression dtype coverage tests.

Tests dtype coverage, parameter combinations, partial buffer writes,
and unsupported dtype compilation errors for T.tile.arith_progression.

Existing coverage in test_tilelang_ascend_language_elementwise.py:
- test_generate_arithmetic_progression: int32, shape=1024, step=1 (low_priority)

This file supplements with:
1. Supported dtype coverage (float16/float32/int16/int32 x ascendc/pto, uint16/uint32 x pto)
2. Parameter combinations on ascendc (first_value/diff_value variations)
3. Partial buffer writes on ascendc (count < buffer size)
4. Unsupported dtype compilation errors (uint16/uint32 x ascendc, float16 x pto)

Note: pto parameter combinations and partial writes are not tested here because
pto codegen has known bugs (does not pass diff_value/count to TCI).
See docs/api_docs/T.tile.arith_progression.md section 2.4.1 for details.
"""

import pytest
import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

DTYPE_TORCH_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "int16": torch.int16,
    "int32": torch.int32,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
}


def _torch_arange(start, end, step, dtype):
    if dtype in (torch.uint16, torch.uint32):
        ref = torch.arange(start, end, step, dtype=torch.int32)
        return ref.to(dtype)
    return torch.arange(start, end, step, dtype=dtype)


def _build_seq(first_value, diff_value, count, torch_dtype):
    if diff_value == 0:
        return torch.full((count,), first_value, dtype=torch_dtype)
    return _torch_arange(first_value, first_value + count * diff_value, diff_value, torch_dtype)


def _make_kernel(N, block_size, dtype, first_value, diff_value, count):
    num_blocks = N // block_size
    VEC_NUM = 2

    @T.prim_func
    def main(output: T.Tensor((N,), dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            start_idx = cid * block_size + vid * block_size // VEC_NUM
            seq_ub = T.alloc_shared((block_size // VEC_NUM,), dtype)
            T.tile.arith_progression(seq_ub, first_value, diff_value, count)
            T.copy(seq_ub, output[start_idx])

    return main


def _make_partial_write_kernel(buf_size, write_count, dtype, first_value, diff_value):
    @T.prim_func
    def main(output: T.Tensor((buf_size,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            seq_ub = T.alloc_shared((buf_size,), dtype)
            T.tile.arith_progression(seq_ub, first_value, diff_value, write_count)
            T.copy(seq_ub, output[0])

    return main


def _run_one(dtype, target, N=1024, block_size=64, first_value=0, diff_value=1):
    torch_dtype = DTYPE_TORCH_MAP[dtype]
    count = block_size // 2
    func = _make_kernel(N, block_size, dtype, first_value, diff_value, count)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    output = torch.zeros(N, dtype=torch_dtype).npu()
    torch.npu.synchronize()
    result = func(output)

    seq = _build_seq(first_value, diff_value, count, torch_dtype)
    ref_result = seq.repeat(N // count)
    torch.testing.assert_close(result.cpu(), ref_result, rtol=0, atol=0)


def _run_partial_write(dtype, target, buf_size=256, write_count=100, first_value=5, diff_value=2):
    torch_dtype = DTYPE_TORCH_MAP[dtype]
    func = _make_partial_write_kernel(buf_size, write_count, dtype, first_value, diff_value)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    output = torch.zeros(buf_size, dtype=torch_dtype).npu()
    torch.npu.synchronize()
    result = func(output)

    ref = _build_seq(first_value, diff_value, write_count, torch_dtype)
    torch.testing.assert_close(result.cpu()[:write_count], ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
        pytest.param("int32", marks=pytest.mark.low_priority),
        pytest.param("int16", marks=pytest.mark.low_priority),
    ],
)
@pytest.mark.parametrize(
    "target",
    [
        "ascendc",
        pytest.param("pto", marks=pytest.mark.low_priority),
    ],
)
def test_arith_progression_dtype(dtype, target):
    _run_one(dtype, target)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float16", marks=pytest.mark.low_priority),
    ],
)
@pytest.mark.parametrize("target", ["ascendc"])
def test_arith_progression_dtype_float16_ascendc(dtype, target):
    _run_one(dtype, target)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("uint16", marks=pytest.mark.low_priority),
        pytest.param("uint32", marks=pytest.mark.low_priority),
    ],
)
@pytest.mark.parametrize("target", ["pto"])
def test_arith_progression_dtype_uint_pto(dtype, target):
    _run_one(dtype, target)


@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
        "int32",
        pytest.param("int16", marks=pytest.mark.low_priority),
    ],
)
@pytest.mark.parametrize("target", ["ascendc"])
@pytest.mark.parametrize(
    "first_value,diff_value",
    [
        pytest.param(10, 2, marks=pytest.mark.low_priority),
        pytest.param(0, 0, marks=pytest.mark.low_priority),
        pytest.param(100, 5, marks=pytest.mark.low_priority),
    ],
)
def test_arith_progression_param_combinations(dtype, target, first_value, diff_value):
    _run_one(dtype, target, first_value=first_value, diff_value=diff_value)


@pytest.mark.low_priority
@pytest.mark.parametrize("dtype", ["float32", "int32"])
@pytest.mark.parametrize("target", ["ascendc"])
def test_arith_progression_partial_write(dtype, target):
    _run_partial_write(dtype, target, buf_size=256, write_count=100, first_value=5, diff_value=2)


@pytest.mark.parametrize(
    "dtype,target",
    [
        ("uint16", "ascendc"),
        pytest.param("uint32", "ascendc", marks=pytest.mark.low_priority),
        pytest.param("float16", "pto", marks=pytest.mark.low_priority),
    ],
)
def test_arith_progression_unsupported_dtype_raises(dtype, target):
    @T.prim_func
    def main(output: T.Tensor((64,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            seq_ub = T.alloc_shared((64,), dtype)
            T.tile.arith_progression(seq_ub, 0, 1, 64)
            T.copy(seq_ub, output[0])

    with pytest.raises(RuntimeError, match="Compilation Failed"):  # noqa: B017
        tilelang.compile(main, out_idx=[-1], pass_configs=pass_configs, target=target)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
