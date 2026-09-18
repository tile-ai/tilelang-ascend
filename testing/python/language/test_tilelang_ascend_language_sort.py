import argparse

import pytest
import torch

import tilelang
import tilelang.language as T

"""
Test suite for T.tile.sort API.

Covers:
  - Basic functionality: dtype (float16/float32) × actual_num (aligned/non-aligned)
  - 2D buffer support
  - Boundary cases: actual_num=1 (min), actual_num=4096 (large, repeatTimes=128;
    8160 max repeatTimes=255 exceeds practical UB capacity and segfaults)
  - src in-place modification verification
  - float16 index precision loss for actual_num > 2048 (known limitation)
  - Output format: interleaved (value, index) pairs, both of dst dtype

Reference: docs/api_docs/T.tile.sort.md
"""

TORCH_DTYPE = {"float16": torch.float16, "float32": torch.float32}
RTOL = {"float16": 1e-3, "float32": 1e-3}
ATOL = {"float16": 1e-3, "float32": 1e-3}

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _aligned_count(actual_num):
    """Compute aligned_count = ((actual_num + 31) // 32) * 32"""
    return ((actual_num + 31) // 32) * 32


# -----------------------------------------------------------------------------
# Kernel builders
# -----------------------------------------------------------------------------


def sort_kernel_1d(actual_num, ub_N, dtype="float16"):
    """1D sort kernel: src (ub_N,), dst (ub_N*2,), actual_num elements valid."""
    m_num = 1
    n_num = 1

    @T.prim_func
    def main(
        A: T.Tensor((1, ub_N), dtype),
        B: T.Tensor((1, ub_N * 2), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((1, ub_N), dtype)
            dst_ub = T.alloc_ub((1, ub_N * 2), dtype)
            T.copy(A, src_ub)
            T.tile.sort(dst_ub, src_ub, actual_num)
            T.copy(dst_ub, B)

    return main


def sort_kernel_2d(M, actual_num, total_ub, dtype="float16"):
    """2D sort kernel: src (M, ub_N), dst (M, ub_N*2).

    T.tile.sort treats 2D buffer as flattened 1D array.
    actual_num = total valid elements across all rows.
    total_ub = M * ub_N (total buffer size, must be 32-aligned).
    """
    m_num = 1
    n_num = 1

    @T.prim_func
    def main(
        A: T.Tensor((M, total_ub // M), dtype),
        B: T.Tensor((M, (total_ub // M) * 2), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((M, total_ub // M), dtype)
            dst_ub = T.alloc_ub((M, (total_ub // M) * 2), dtype)
            T.copy(A, src_ub)
            T.tile.sort(dst_ub, src_ub, actual_num)
            T.copy(dst_ub, B)

    return main


def sort_kernel_with_src_output(M, actual_num, ub_N, dtype="float16"):
    """Sort kernel that also copies src back to GM for in-place modification check."""
    m_num = 1
    n_num = 1

    @T.prim_func
    def main(
        A: T.Tensor((M, ub_N), dtype),
        B: T.Tensor((M, ub_N * 2), dtype),
        S: T.Tensor((M, ub_N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((M, ub_N), dtype)
            dst_ub = T.alloc_ub((M, ub_N * 2), dtype)
            T.copy(A, src_ub)
            T.tile.sort(dst_ub, src_ub, actual_num)
            T.copy(dst_ub, B)
            T.copy(src_ub, S)

    return main


# -----------------------------------------------------------------------------
# Reference computation (CPU)
# -----------------------------------------------------------------------------


def _cpu_sort_ref(a_cpu, actual_num, aligned_count):
    """Compute reference sort result on CPU.

    Returns (out_values, out_indices) of length actual_num, descending order.
    Pads with -inf up to aligned_count to match hardware behavior.
    """
    a_flat = a_cpu.float().reshape(-1)
    a_padded = torch.full((aligned_count,), float("-inf"))
    a_padded[:actual_num] = a_flat[:actual_num]
    ref_vals, ref_index = torch.sort(a_padded, descending=True)
    return ref_vals[:actual_num], ref_index[:actual_num]


# -----------------------------------------------------------------------------
# Run-test helpers
# -----------------------------------------------------------------------------


def run_test_sort_basic(actual_num, dtype, target):
    """Three-fold verification: compile + run + precision for 1D sort."""
    ub_N = _aligned_count(actual_num)
    torch_dtype = TORCH_DTYPE[dtype]

    kernel = sort_kernel_1d(actual_num, ub_N, dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    perm = torch.randperm(actual_num, dtype=torch.int32)
    a = torch.full((1, ub_N), float("-inf"), dtype=torch_dtype).npu()
    a[:, :actual_num] = perm.to(torch_dtype)

    b = func(a)
    torch.npu.synchronize()

    b_cpu = b.cpu().float().reshape(-1)
    out_values = b_cpu[0::2][:actual_num]
    out_indices = b_cpu[1::2][:actual_num]

    ref_vals, ref_index = _cpu_sort_ref(a.cpu(), actual_num, ub_N)

    torch.testing.assert_close(out_values, ref_vals, rtol=RTOL[dtype], atol=ATOL[dtype])
    torch.testing.assert_close(out_indices, ref_index.float(), rtol=RTOL[dtype], atol=ATOL[dtype])


def run_test_sort_2d(M, per_row_N, dtype, target):
    """Three-fold verification for 2D buffer sort.

    T.tile.sort treats the 2D buffer as a flat 1D array (row-major).
    actual_num = M * per_row_N (total valid elements across all rows).
    Uses unique values across the entire buffer to avoid index ambiguity
    from duplicate values.
    """
    ub_N = _aligned_count(per_row_N)
    total_ub = M * ub_N
    total_actual = M * per_row_N
    torch_dtype = TORCH_DTYPE[dtype]

    kernel = sort_kernel_2d(M, total_actual, total_ub, dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    a = torch.full((M, ub_N), float("-inf"), dtype=torch_dtype).npu()
    total_perm = torch.randperm(total_actual, dtype=torch.int32)
    a_flat = a.view(-1)
    a_flat[:total_actual] = total_perm.to(torch_dtype)

    b = func(a)
    torch.npu.synchronize()

    b_cpu = b.cpu().float().reshape(-1)
    out_values = b_cpu[0::2][:total_actual]
    out_indices = b_cpu[1::2][:total_actual]

    ref_vals, ref_index = _cpu_sort_ref(a.cpu(), total_actual, total_ub)

    torch.testing.assert_close(out_values, ref_vals, rtol=RTOL[dtype], atol=ATOL[dtype])
    torch.testing.assert_close(out_indices, ref_index.float(), rtol=RTOL[dtype], atol=ATOL[dtype])


def run_test_sort_src_inplace(M, actual_num, dtype, target):
    """Verify src in-place modification behavior.

    float32: src IS modified in-place (tail region padded with -inf).
    float16: src is NOT modified (implementation casts to float32 in tmp,
             padding happens on the float copy, not original src).
    """
    ub_N = _aligned_count(actual_num)
    torch_dtype = TORCH_DTYPE[dtype]

    kernel = sort_kernel_with_src_output(M, actual_num, ub_N, dtype)
    func = tilelang.compile(kernel, out_idx=[1, 2], pass_configs=PASS_CONFIGS, target=target)

    perm = torch.randperm(actual_num, dtype=torch.int32)
    a = torch.full((M, ub_N), 0.0, dtype=torch_dtype).npu()
    a[:, :actual_num] = perm.to(torch_dtype)

    b, s = func(a)
    torch.npu.synchronize()

    s_cpu = s.cpu()
    if actual_num < ub_N:
        tail = s_cpu[0, actual_num:ub_N].float()
        if dtype == "float32":
            assert torch.all(tail == float("-inf")), (
                f"src tail [{actual_num}:{ub_N}] should be -inf after in-place padding, "
                f"got min={tail.min().item()}, max={tail.max().item()}"
            )
        else:
            assert torch.all(tail == 0.0), (
                f"float16 src tail [{actual_num}:{ub_N}] should remain unchanged (0.0), "
                f"got min={tail.min().item()}, max={tail.max().item()}"
            )


def run_test_sort_fp16_index_large(actual_num, dtype, target):
    """Test float16 index precision for actual_num > 2048.

    float16 can exactly represent integers 0-2048. Beyond that, index values
    may lose precision due to half rounding. This test verifies the behavior
    and reports the mismatch rate (expected to be non-zero).
    """
    ub_N = _aligned_count(actual_num)
    torch_dtype = TORCH_DTYPE[dtype]

    kernel = sort_kernel_1d(actual_num, ub_N, dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    perm = torch.randperm(actual_num, dtype=torch.int32)
    a = torch.full((1, ub_N), float("-inf"), dtype=torch_dtype).npu()
    a[:, :actual_num] = perm.to(torch_dtype)

    b = func(a)
    torch.npu.synchronize()

    b_cpu = b.cpu().float().reshape(-1)
    out_values = b_cpu[0::2][:actual_num]
    out_indices = b_cpu[1::2][:actual_num]

    ref_vals, ref_index = _cpu_sort_ref(a.cpu(), actual_num, ub_N)

    torch.testing.assert_close(out_values, ref_vals, rtol=RTOL[dtype], atol=ATOL[dtype])

    mismatched = (out_indices != ref_index.float()).sum().item()
    mismatch_rate = mismatched / actual_num
    print(f"  fp16 index mismatch: {mismatched}/{actual_num} ({mismatch_rate:.1%})")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before tests."""
    tilelang.cache.clear_cache()
    yield


@pytest.fixture
def setup_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(0)
    yield


# -----------------------------------------------------------------------------
# Test cases
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 1. Basic functionality: dtype × actual_num (1D)
# -----------------------------------------------------------------------------


basic_params = [
    pytest.param(32, marks=pytest.mark.low_priority),  # 1 sort32 block, no merge
    pytest.param(64, marks=pytest.mark.low_priority),  # 2 blocks, triggers merge
    pytest.param(128, marks=pytest.mark.low_priority),  # 4 blocks
    256,  # 8 blocks
    131,  # non-aligned actual_num, padded to 160
]


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize(
    "dtype",
    ["float32", pytest.param("float16", marks=pytest.mark.low_priority)],
)
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
@pytest.mark.parametrize("actual_num", basic_params)
def test_sort_basic(dtype, target, actual_num):
    """Basic 1D sort: compile + run + precision for each dtype and actual_num."""
    run_test_sort_basic(actual_num, dtype, target)


# -----------------------------------------------------------------------------
# 2. 2D buffer support
# -----------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize(
    "dtype",
    ["float32", pytest.param("float16", marks=pytest.mark.low_priority)],
)
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
@pytest.mark.parametrize(
    "M,per_row_N",
    [
        pytest.param(1, 131, marks=pytest.mark.low_priority),  # single row, non-aligned
        (4, 128),  # multi-row, each row aligned (total 512 elements)
        pytest.param(2, 64, marks=pytest.mark.low_priority),  # multi-row, small (total 128 elements)
    ],
)
def test_sort_2d(dtype, target, M, per_row_N):
    """2D buffer sort: buffer is flattened and sorted as one array."""
    run_test_sort_2d(M, per_row_N, dtype, target)


# -----------------------------------------------------------------------------
# 3. Boundary: actual_num=1 (minimum valid)
# -----------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize(
    "dtype",
    ["float32", pytest.param("float16", marks=pytest.mark.low_priority)],
)
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
def test_sort_min_actual_num(dtype, target):
    """Boundary: actual_num=1 (minimum valid value, repeatTimes=1)."""
    run_test_sort_basic(1, dtype, target)


# -----------------------------------------------------------------------------
# 4. Boundary: large actual_num (4096, repeatTimes=128)
#    Note: hardware upper limit is 8160 (repeatTimes=255) but exceeds UB capacity.
# -----------------------------------------------------------------------------


@pytest.mark.low_priority
@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
def test_sort_large_actual_num(dtype, target):
    """Large input: actual_num=4096 (repeatTimes=128).

    The hardware upper limit is repeatTimes=255 (actual_num=8160), but
    actual_num=8160 exceeds practical UB capacity and causes segfault.
    This test uses 4096 as a large-but-feasible size.

    Only float32 is tested here because float16 index precision loss
    beyond 2048 elements causes index mismatch (covered separately by
    test_sort_fp16_index_precision).
    """
    run_test_sort_basic(4096, dtype, target)


# -----------------------------------------------------------------------------
# 5. src in-place modification verification
#    Note: pto pads only the 32-byte-aligned part of the tail with -inf
#    (e.g. tail [131:160) -> 24/29 -inf, first 5 elements left as 0.0),
#    unlike ascendc which pads the whole tail. Sorting values are correct
#    on both backends; the padding assertion is ascendc-specific, so the
#    pto combos are ci_skip.
# -----------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize(
    "dtype",
    ["float32", pytest.param("float16", marks=pytest.mark.low_priority)],
)
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.ci_skip)],
)
@pytest.mark.parametrize(
    "actual_num",
    [
        131,  # non-aligned: padding region [131:160)
        pytest.param(100, marks=pytest.mark.low_priority),  # non-aligned: padding region [100:128)
    ],
)
def test_sort_src_inplace(dtype, target, actual_num):
    """Verify src in-place modification behavior:
    float32 - tail [actual_num:ub_N) padded with -inf;
    float16 - src unchanged (padding happens on internal float32 copy).
    """
    run_test_sort_src_inplace(1, actual_num, dtype, target)


# -----------------------------------------------------------------------------
# 6. float16 index precision for actual_num > 2048
#    Known limitation: half can only exactly represent integers 0-2048.
#    This test reports mismatch rate but does NOT fail on index precision.
#    Values are still verified for correctness.
# -----------------------------------------------------------------------------


@pytest.mark.low_priority
@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
@pytest.mark.parametrize("actual_num", [2049, 4096])
def test_sort_fp16_index_precision(dtype, target, actual_num):
    """float16 index precision loss for actual_num > 2048.

    Per docs/api_docs/T.tile.sort.md section 2.3.4:
      'Index values beyond 2048 may lose exactness due to half-precision rounding.'

    This test verifies:
      - Values are still correct (asserted)
      - Index mismatch rate is reported (not asserted, known limitation)
    """
    run_test_sort_fp16_index_large(actual_num, dtype, target)


# -----------------------------------------------------------------------------
# 7. actual_num=0 triggers hardware exception (constraint verification)
#    Marked ci_skip because it crashes NPU aicore and affects subsequent tests.
# -----------------------------------------------------------------------------


@pytest.mark.ci_skip
@pytest.mark.parametrize(
    "dtype",
    ["float32", pytest.param("float16", marks=pytest.mark.low_priority)],
)
def test_sort_actual_num_zero_crashes(dtype):
    """Verify actual_num=0 triggers hardware exception (constraint #5: repeatTimes >= 1).

    Per docs/api_docs/T.tile.sort.md constraint #5:
      'repeatTimes in [1, 255], actual_num in [1, 8160]'

    actual_num=0 -> repeatTimes=0 -> NPU aicore exception (error 507015).
    Marked ci_skip because it crashes the NPU and affects subsequent tests.
    """
    ub_N = 32

    @T.prim_func
    def main_zero(
        A: T.Tensor((1, ub_N), dtype),
        B: T.Tensor((1, ub_N * 2), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((1, ub_N), dtype)
            dst_ub = T.alloc_ub((1, ub_N * 2), dtype)
            T.copy(A, src_ub)
            T.tile.sort(dst_ub, src_ub, 0)
            T.copy(dst_ub, B)

    torch_dtype = TORCH_DTYPE[dtype]
    a = torch.full((1, ub_N), float("-inf"), dtype=torch_dtype).npu()

    func = tilelang.compile(main_zero, out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")
    with pytest.raises(RuntimeError):
        func(a)
        torch.npu.synchronize()


# -----------------------------------------------------------------------------
# Standalone command-line entry point
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T.tile.sort test suite")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument("--target", type=str, choices=["ascendc", "pto"], default="ascendc")
    parser.add_argument("--actual-num", type=int, default=131, help="Number of valid elements")
    parser.add_argument("--m", type=int, default=1, help="Number of rows (for 2D test)")
    args = parser.parse_args()

    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    print("=" * 60)
    print(f"T.tile.sort test: dtype={args.dtype}, target={args.target}")
    print("=" * 60)

    print(f"\n[1D basic] actual_num={args.actual_num}")
    run_test_sort_basic(args.actual_num, args.dtype, args.target)
    print("  PASSED")

    print(f"\n[2D] M={args.m}, per_row_N={args.actual_num}")
    run_test_sort_2d(args.m, args.actual_num, args.dtype, args.target)
    print("  PASSED")

    ub_N = _aligned_count(args.actual_num)
    if args.actual_num < ub_N:
        print(f"\n[src in-place] actual_num={args.actual_num}, ub_N={ub_N}")
        run_test_sort_src_inplace(1, args.actual_num, args.dtype, args.target)
        print("  PASSED")

    print("\nAll tests passed.")
