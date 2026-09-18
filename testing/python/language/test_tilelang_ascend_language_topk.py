import argparse

import pytest
import torch

import tilelang
import tilelang.language as T

"""
Test suite for T.tile.topk API.

Covers:
  - Basic functionality: dtype (float16/float32) x actual_num (aligned/non-aligned)
  - K boundary: K = actual_num (maximum), K = 1 (minimum)
  - 2D buffer support (sum-of-dims semantics)
  - src in-place modification verification
  - float16 index precision loss for actual_num > 2048 (known limitation)

"""

TORCH_DTYPE = {
    "float": torch.float32,
    "float16": torch.float16,
    "float32": torch.float32,
}
RTOL = {"float": 1e-3, "float16": 1e-3, "float32": 1e-3}
ATOL = {"float": 1e-3, "float16": 1e-3, "float32": 1e-3}

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _aligned_topk(K, dtype):
    """aligned_topk = ceil(2*K / elems_per_block) * elems_per_block."""
    elems_per_block = 16 if dtype == "float16" else 8
    return ((2 * K + elems_per_block - 1) // elems_per_block) * elems_per_block


# -----------------------------------------------------------------------------
# Kernel builders
# -----------------------------------------------------------------------------


def topk_kernel_1d(buffer_size, K, actual_num, dtype="float16", with_src_out=False):
    """1D topk kernel: src (buffer_size,), dst (aligned_topk,)."""
    dst_size = _aligned_topk(K, dtype)

    if with_src_out:

        @T.prim_func
        def main(
            A: T.Tensor((buffer_size,), dtype),
            B: T.Tensor((dst_size,), dtype),
            S: T.Tensor((buffer_size,), dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                src_ub = T.alloc_ub((buffer_size,), dtype)
                dst_ub = T.alloc_ub((dst_size,), dtype)
                T.copy(A, src_ub)
                T.tile.topk(dst_ub, src_ub, K, actual_num)
                T.copy(dst_ub, B)
                T.copy(src_ub, S)

        return main

    @T.prim_func
    def main(
        A: T.Tensor((buffer_size,), dtype),
        B: T.Tensor((dst_size,), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((buffer_size,), dtype)
            dst_ub = T.alloc_ub((dst_size,), dtype)
            T.copy(A, src_ub)
            T.tile.topk(dst_ub, src_ub, K, actual_num)
            T.copy(dst_ub, B)

    return main


def topk_kernel_2d(M, per_row, K, actual_num, dtype="float16"):
    """2D topk kernel: src (M, per_row), dst (aligned_topk,).

    The number of elements involved equals the sum of the shape dimensions
    (M + per_row); only the first actual_num elements are valid.
    """
    dst_size = _aligned_topk(K, dtype)

    @T.prim_func
    def main(
        A: T.Tensor((M, per_row), dtype),
        B: T.Tensor((dst_size,), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((M, per_row), dtype)
            dst_ub = T.alloc_ub((dst_size,), dtype)
            T.copy(A, src_ub)
            T.tile.topk(dst_ub, src_ub, K, actual_num)
            T.copy(dst_ub, B)

    return main


# -----------------------------------------------------------------------------
# Reference computation (CPU)
# -----------------------------------------------------------------------------


def _cpu_topk_ref(a_cpu, actual_num, K):
    """Reference: top-K values and indices of the first actual_num elements."""
    a_flat = a_cpu.float().reshape(-1)
    ref_vals, ref_index = torch.sort(a_flat[:actual_num], descending=True)
    return ref_vals[:K], ref_index[:K].float()


# -----------------------------------------------------------------------------
# Run-test helpers
# -----------------------------------------------------------------------------


def run_test_topk_basic(buffer_size, actual_num, K, dtype, target):
    """Three-fold verification: compile + run + precision for 1D topk."""
    torch_dtype = TORCH_DTYPE[dtype]

    kernel = topk_kernel_1d(buffer_size, K, actual_num, dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    perm = torch.randperm(actual_num, dtype=torch.int32)
    a = torch.full((buffer_size,), float("-inf"), dtype=torch_dtype).npu()
    a[:actual_num] = perm.to(torch_dtype)

    b = func(a)
    torch.npu.synchronize()

    b_cpu = b.cpu().float().reshape(-1)
    out_values = b_cpu[0::2][:K]
    out_indices = b_cpu[1::2][:K]

    ref_vals, ref_indices = _cpu_topk_ref(a.cpu(), actual_num, K)

    torch.testing.assert_close(out_values, ref_vals, rtol=RTOL[dtype], atol=ATOL[dtype])
    torch.testing.assert_close(out_indices, ref_indices, rtol=RTOL[dtype], atol=ATOL[dtype])


def run_test_topk_2d(M, per_row, K, actual_num, dtype, target):
    """Three-fold verification for 2D buffer topk.

    The top-K is computed over the first actual_num elements of the flattened
    (row-major) buffer.
    """
    torch_dtype = TORCH_DTYPE[dtype]

    kernel = topk_kernel_2d(M, per_row, K, actual_num, dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    perm = torch.randperm(actual_num, dtype=torch.int32)
    a = torch.full((M, per_row), float("-inf"), dtype=torch_dtype).npu()
    a_flat = a.reshape(-1)
    a_flat[:actual_num] = perm.to(torch_dtype)

    b = func(a)
    torch.npu.synchronize()

    b_cpu = b.cpu().float().reshape(-1)
    out_values = b_cpu[0::2][:K]
    out_indices = b_cpu[1::2][:K]

    ref_vals, ref_indices = _cpu_topk_ref(a.cpu(), actual_num, K)

    torch.testing.assert_close(out_values, ref_vals, rtol=RTOL[dtype], atol=ATOL[dtype])
    torch.testing.assert_close(out_indices, ref_indices, rtol=RTOL[dtype], atol=ATOL[dtype])


def run_test_topk_src_inplace(buffer_size, actual_num, K, dtype, target):
    """Verify src in-place modification behavior.

    float32: src IS modified in-place (tail region padded with -inf).
             On ascendc the whole tail [actual_num:buffer_size) is padded;
             on pto only block-aligned positions are guaranteed padded
             in-place (the partial block may keep its original values).
    float16: src is NOT modified (implementation casts to float32 in tmp,
             padding happens on the float copy, not the original src).
    """
    torch_dtype = TORCH_DTYPE[dtype]

    kernel = topk_kernel_1d(buffer_size, K, actual_num, dtype, with_src_out=True)
    func = tilelang.compile(kernel, out_idx=[1, 2], pass_configs=PASS_CONFIGS, target=target)

    perm = torch.randperm(actual_num, dtype=torch.int32)
    a = torch.full((buffer_size,), 0.0, dtype=torch_dtype).npu()
    a[:actual_num] = perm.to(torch_dtype)

    b, s = func(a)
    torch.npu.synchronize()

    s_cpu = s.cpu()
    if actual_num < buffer_size:
        tail = s_cpu[actual_num:buffer_size].float()
        if dtype in ("float", "float32"):
            if target == "ascendc":
                assert torch.all(tail == float("-inf")), (
                    f"src tail [{actual_num}:{buffer_size}] should be -inf after in-place padding, "
                    f"got min={tail.min().item()}, max={tail.max().item()}"
                )
            else:
                assert (tail == float("-inf")).any(), (
                    f"pto src tail [{actual_num}:{buffer_size}] should be partially padded with -inf, "
                    f"got min={tail.min().item()}, max={tail.max().item()}"
                )
        else:
            assert torch.all(tail == 0.0), (
                f"float16 src tail [{actual_num}:{buffer_size}] should remain unchanged (0.0), "
                f"got min={tail.min().item()}, max={tail.max().item()}"
            )


def run_test_topk_fp16_index_large(buffer_size, actual_num, K, dtype, target):
    """Test float16 index precision for actual_num > 2048.

    float16 can exactly represent integers 0-2048. Beyond that, index values
    may lose precision due to half rounding. Values are still verified for
    correctness.
    """
    torch_dtype = TORCH_DTYPE[dtype]

    kernel = topk_kernel_1d(buffer_size, K, actual_num, dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    perm = torch.randperm(actual_num, dtype=torch.int32)
    a = torch.full((buffer_size,), float("-inf"), dtype=torch_dtype).npu()
    a[:actual_num] = perm.to(torch_dtype)

    b = func(a)
    torch.npu.synchronize()

    b_cpu = b.cpu().float().reshape(-1)
    out_values = b_cpu[0::2][:K]
    out_indices = b_cpu[1::2][:K]

    ref_vals, ref_indices = _cpu_topk_ref(a.cpu(), actual_num, K)

    torch.testing.assert_close(out_values, ref_vals, rtol=RTOL[dtype], atol=ATOL[dtype])

    mismatched = (out_indices != ref_indices).sum().item()
    mismatch_rate = mismatched / K
    print(f"  fp16 index mismatch: {mismatched}/{K} ({mismatch_rate:.1%})")


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
# 1. Basic functionality: dtype x actual_num (1D, buffer size is a multiple of 32)
# -----------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize(
    "dtype",
    [
        "float",
        pytest.param("float16", marks=pytest.mark.low_priority),
    ],
)
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
@pytest.mark.parametrize(
    "actual_num",
    [
        128,  # aligned: actual_num == buffer size
        51,  # non-aligned: padded to 128
        pytest.param(100, marks=pytest.mark.low_priority),  # non-aligned: covered by 51
    ],
)
def test_topk_basic(dtype, target, actual_num):
    """Basic 1D topk: compile + run + precision for each dtype and actual_num."""
    run_test_topk_basic(128, actual_num, K=10, dtype=dtype, target=target)


# -----------------------------------------------------------------------------
# 2. K boundary
# -----------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize(
    "dtype",
    [
        "float",
        pytest.param("float16", marks=pytest.mark.low_priority),
    ],
)
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
@pytest.mark.parametrize("K", [1, 64])
def test_topk_k_boundary(dtype, target, K):
    """Boundary: K=1 (minimum) and K=actual_num (maximum)."""
    run_test_topk_basic(128, 64, K=K, dtype=dtype, target=target)


# -----------------------------------------------------------------------------
# 3. 2D buffer support (sum-of-dims semantics)
# -----------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize(
    "dtype",
    [
        "float",
        pytest.param("float16", marks=pytest.mark.low_priority),
    ],
)
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
def test_topk_2d(dtype, target):
    """2D buffer topk: only the first sum(shape) elements are involved."""
    run_test_topk_2d(M=2, per_row=16, K=6, actual_num=18, dtype=dtype, target=target)


# -----------------------------------------------------------------------------
# 4. src in-place modification verification
# -----------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize(
    "dtype",
    [
        "float",
        pytest.param("float16", marks=pytest.mark.low_priority),
    ],
)
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
def test_topk_src_inplace(dtype, target):
    """Verify src in-place modification behavior:
    float32 - tail [actual_num:buffer_size) padded with -inf;
    float16 - src unchanged (padding happens on internal float32 copy).
    """
    run_test_topk_src_inplace(128, 51, K=10, dtype=dtype, target=target)


# -----------------------------------------------------------------------------
# 5. Large actual_num (4096, low priority)
# -----------------------------------------------------------------------------


@pytest.mark.low_priority
@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
def test_topk_large_actual_num(dtype, target):
    """Large input: actual_num=4096 (repeatTimes=128).

    The hardware upper limit is repeatTimes=255 (buffer size 8160), but the
    practical size is limited by UB capacity.
    """
    run_test_topk_basic(4096, 4096, K=10, dtype=dtype, target=target)


# -----------------------------------------------------------------------------
# 6. float16 index precision for actual_num > 2048 (low priority)
#    Known limitation: half can only exactly represent integers 0-2048.
# -----------------------------------------------------------------------------


@pytest.mark.low_priority
@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
@pytest.mark.parametrize("actual_num", [2049, 4096])
def test_topk_fp16_index_precision(dtype, target, actual_num):
    """float16 index precision loss for actual_num > 2048.

    Values are verified for correctness; index mismatch rate is reported but
    not asserted (known limitation documented in the API doc).
    """
    run_test_topk_fp16_index_large(actual_num, actual_num, K=10, dtype=dtype, target=target)


# -----------------------------------------------------------------------------
# Standalone command-line entry point
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T.tile.topk test suite")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument("--target", type=str, choices=["ascendc", "pto"], default="ascendc")
    parser.add_argument("--buffer-size", type=int, default=128, help="src buffer size (multiple of 32)")
    parser.add_argument("--actual-num", type=int, default=51, help="Number of valid elements")
    parser.add_argument("--k", type=int, default=10, help="Number of top elements to extract")
    args = parser.parse_args()

    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    print("=" * 60)
    print(f"T.tile.topk test: dtype={args.dtype}, target={args.target}")
    print("=" * 60)

    print(f"\n[1D basic] buffer_size={args.buffer_size}, actual_num={args.actual_num}, K={args.k}")
    run_test_topk_basic(args.buffer_size, args.actual_num, args.k, args.dtype, args.target)
    print("  PASSED")

    print("\nAll tests passed.")
