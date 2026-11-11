import argparse

import pytest
import torch

import tilelang
import tilelang.language as T

"""
Test suite for T.tile.merge_sort API.

Covers:
  - Basic functionality: 2/3/4-way merge (verify 4-way is supported)
  - dtype: float32 only (float16 is NOT supported by the current implementation:
    ascendc raises aicore exception, pto produces wrong results)
  - Unequal block lengths (ascendc only; pto compile fails)
  - BufferRegion slices as sources (2D buffer row slices)
  - Stability: equal scores keep source order (src0 -> src1 -> ...) and
    in-block order
  - blockLen boundaries per backend: ascendc [1, 4095], pto [4, 4088]

low_priority trimming:
  - pto: ascendc executes by default; pto is marked low_priority
  - large merge sizes / stability / boundaries: low_priority when overlapping
"""

TORCH_DTYPE = {"float32": torch.float32}
RTOL = 1e-3
ATOL = 1e-3

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# -----------------------------------------------------------------------------
# Kernel builders
# -----------------------------------------------------------------------------


def merge_sort_kernel_2way(n0, n1, dtype="float32"):
    """2-way merge_sort kernel: src sizes 2*n0, 2*n1; dst 2*(n0+n1)."""

    @T.prim_func
    def main(
        A0: T.Tensor((2 * n0,), dtype),  # type: ignore
        A1: T.Tensor((2 * n1,), dtype),  # type: ignore
        B: T.Tensor((2 * (n0 + n1),), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src0 = T.alloc_ub((2 * n0,), dtype)
            src1 = T.alloc_ub((2 * n1,), dtype)
            dst = T.alloc_ub((2 * (n0 + n1),), dtype)
            T.copy(A0, src0)
            T.copy(A1, src1)
            T.tile.merge_sort(dst, src0, src1)
            T.copy(dst, B)

    return main


def merge_sort_kernel_3way(n, dtype="float32"):
    """3-way merge_sort kernel: all src sizes 2*n; dst 6*n."""

    @T.prim_func
    def main(
        A0: T.Tensor((2 * n,), dtype),  # type: ignore
        A1: T.Tensor((2 * n,), dtype),  # type: ignore
        A2: T.Tensor((2 * n,), dtype),  # type: ignore
        B: T.Tensor((6 * n,), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src0 = T.alloc_ub((2 * n,), dtype)
            src1 = T.alloc_ub((2 * n,), dtype)
            src2 = T.alloc_ub((2 * n,), dtype)
            dst = T.alloc_ub((6 * n,), dtype)
            T.copy(A0, src0)
            T.copy(A1, src1)
            T.copy(A2, src2)
            T.tile.merge_sort(dst, src0, src1, src2)
            T.copy(dst, B)

    return main


def merge_sort_kernel_4way(n, dtype="float32"):
    """4-way merge_sort kernel: all src sizes 2*n; dst 8*n."""

    @T.prim_func
    def main(
        A0: T.Tensor((2 * n,), dtype),  # type: ignore
        A1: T.Tensor((2 * n,), dtype),  # type: ignore
        A2: T.Tensor((2 * n,), dtype),  # type: ignore
        A3: T.Tensor((2 * n,), dtype),  # type: ignore
        B: T.Tensor((8 * n,), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src0 = T.alloc_ub((2 * n,), dtype)
            src1 = T.alloc_ub((2 * n,), dtype)
            src2 = T.alloc_ub((2 * n,), dtype)
            src3 = T.alloc_ub((2 * n,), dtype)
            dst = T.alloc_ub((8 * n,), dtype)
            T.copy(A0, src0)
            T.copy(A1, src1)
            T.copy(A2, src2)
            T.copy(A3, src3)
            T.tile.merge_sort(dst, src0, src1, src2, src3)
            T.copy(dst, B)

    return main


def merge_sort_kernel(sizes, dtype="float32"):
    """N-way merge_sort kernel from element-count list `sizes`.

    Each source buffer has `2 * size` elements (value, index pairs).
    dst has `2 * sum(sizes)` elements.
    """
    assert len(sizes) in (2, 3, 4)
    if len(sizes) == 2:
        return merge_sort_kernel_2way(sizes[0], sizes[1], dtype)
    if len(sizes) == 3:
        assert sizes[0] == sizes[1] == sizes[2]
        return merge_sort_kernel_3way(sizes[0], dtype)
    assert sizes[0] == sizes[1] == sizes[2] == sizes[3]
    return merge_sort_kernel_4way(sizes[0], dtype)


def merge_sort_kernel_region(M, per_row, dtype="float32"):
    """2D src buffer (M, per_row); sources are row slices of it.
    Requires M == 2 (i.e., a 2-way merge on two row slices).
    """
    total = M * per_row

    @T.prim_func
    def main(A: T.Tensor((M, per_row), dtype), B: T.Tensor((total,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src = T.alloc_ub((M, per_row), dtype)
            dst = T.alloc_ub((total,), dtype)
            T.copy(A, src)
            T.tile.merge_sort(dst, src[0, :], src[1, :])
            T.copy(dst, B)

    return main


# -----------------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------------


def make_sorted_block(n, dtype="float32", seed=0, k_distinct=None):
    """Create a block of n (value, index) pairs sorted in descending order.

    With k_distinct, values are drawn from k_distinct distinct ints so that
    ties are guaranteed (for stability testing).
    """
    torch.manual_seed(seed)
    if k_distinct is not None:
        vals = torch.randint(0, k_distinct, (n,)).float()
    else:
        vals = torch.randn(n)
    idx = torch.argsort(vals, descending=True, stable=True)
    pairs = torch.zeros(n * 2, dtype=TORCH_DTYPE[dtype])
    pairs[0::2] = vals[idx]
    pairs[1::2] = idx.float()
    return pairs


def cpu_merge_ref(blocks):
    """Stable k-way merge reference matching hardware MrgSort semantics.

    Equal (value) scores are ordered by source order first (all elements of
    src0 before any element of src1), then by in-block original order. Since
    each block is pre-sorted by (descending value, ascending original index),
    the merged order is: sort by (value desc, block order, in-block order).
    """
    elems = []
    for bi, b in enumerate(blocks):
        for j in range(len(b) // 2):
            elems.append((b[2 * j].item(), bi, b[2 * j + 1].float().item()))
    elems.sort(key=lambda e: (-e[0], e[1], e[2]))
    result = torch.zeros(2 * len(elems), dtype=torch.float32)
    result[0::2] = torch.tensor([v for v, _, _ in elems])
    result[1::2] = torch.tensor([i for _, _, i in elems])
    return result


# -----------------------------------------------------------------------------
# Run-test helpers
# -----------------------------------------------------------------------------


def run_test_merge(block_len, num_ways, dtype="float32", target="ascendc", seeds=None):
    """Three-fold verification: compile + run + precision for N-way merge."""
    sizes = [block_len] * num_ways
    seeds = seeds or list(range(num_ways))
    blocks = [make_sorted_block(s, dtype, seed=seeds[i]) for i, s in enumerate(sizes)]

    kernel = merge_sort_kernel(sizes, dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    out = func(*[b.npu() for b in blocks])
    torch.npu.synchronize()

    out_cpu = out.cpu().float().reshape(-1)
    ref = cpu_merge_ref(blocks)

    torch.testing.assert_close(out_cpu[0::2], ref[0::2], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(out_cpu[1::2], ref[1::2], rtol=0, atol=0)


def run_test_merge_unequal(sizes, dtype="float32", target="ascendc", seeds=None):
    """Merge with different block lengths per source."""
    seeds = seeds or list(range(len(sizes)))
    blocks = [make_sorted_block(s, dtype, seed=seeds[i]) for i, s in enumerate(sizes)]

    kernel = merge_sort_kernel(sizes, dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    out = func(*[b.npu() for b in blocks])
    torch.npu.synchronize()

    out_cpu = out.cpu().float().reshape(-1)
    ref = cpu_merge_ref(blocks)

    torch.testing.assert_close(out_cpu[0::2], ref[0::2], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(out_cpu[1::2], ref[1::2], rtol=0, atol=0)


def run_test_merge_stability(block_len, dtype="float32", target="ascendc"):
    """Verify stability: equal values keep source order + in-block order.

    Values are drawn from a small integer range to force ties.
    """
    blocks = [make_sorted_block(block_len, dtype, seed=i, k_distinct=8) for i in range(3)]

    kernel = merge_sort_kernel([block_len] * 3, dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    out = func(*[b.npu() for b in blocks])
    torch.npu.synchronize()

    out_cpu = out.cpu().float().reshape(-1)
    ref = cpu_merge_ref(blocks)

    torch.testing.assert_close(out_cpu[0::2], ref[0::2], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(out_cpu[1::2], ref[1::2], rtol=0, atol=0)


def run_test_merge_region(M, per_row, dtype="float32", target="ascendc"):
    """Use row slices of a 2D buffer as the sources."""
    per_row = per_row // 2 * 2  # even: block_len = per_row // 2
    seeds = list(range(M))
    blocks = [make_sorted_block(per_row // 2, dtype, seed=seeds[i]) for i in range(M)]
    a = torch.stack(blocks)

    kernel = merge_sort_kernel_region(M, per_row, dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    out = func(a.npu())
    torch.npu.synchronize()

    out_cpu = out.cpu().float().reshape(-1)
    ref = cpu_merge_ref(blocks)

    torch.testing.assert_close(out_cpu[0::2], ref[0::2], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(out_cpu[1::2], ref[1::2], rtol=0, atol=0)


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


@pytest.mark.usefixtures("setup_random_seed")
def test_merge_sort_basic_ascendc():
    """2/3/4-way merge on ascendc: compile + run + precision."""
    for num_ways in (2, 3, 4):
        run_test_merge(block_len=64, num_ways=num_ways, target="ascendc")


@pytest.mark.usefixtures("setup_random_seed")
def test_merge_sort_sizes_ascendc():
    """Block lengths and unequal sources on ascendc."""
    for block_len in (16, 64):
        run_test_merge(block_len=block_len, num_ways=2, target="ascendc")
    run_test_merge_unequal([16, 8], target="ascendc")
    run_test_merge_unequal([48, 16], target="ascendc")


@pytest.mark.usefixtures("setup_random_seed")
def test_merge_sort_edges_ascendc():
    """Boundary shapes on ascendc: min blockLen and BufferRegion slices."""
    run_test_merge(block_len=1, num_ways=2, target="ascendc")
    run_test_merge_region(M=2, per_row=128, target="ascendc")


@pytest.mark.low_priority
@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize("num_ways", [2, 3, 4])
def test_merge_sort_basic_pto(num_ways):
    """2/3/4-way merge on pto (low_priority)."""
    run_test_merge(block_len=64, num_ways=num_ways, target="pto")


@pytest.mark.low_priority
@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize(
    "target,block_len",
    [
        ("ascendc", 128),  # ascendc large block len
        ("pto", 16),
        ("pto", 64),
        ("pto", 128),
    ],
)
def test_merge_sort_2way_sizes_low(target, block_len):
    """2-way merge with block lengths on both backends (low_priority)."""
    run_test_merge(block_len=block_len, num_ways=2, target=target)


@pytest.mark.low_priority
@pytest.mark.usefixtures("setup_random_seed")
def test_merge_sort_region_pto():
    """BufferRegion row slices as sources on pto (low_priority)."""
    run_test_merge_region(M=2, per_row=128, target="pto")


@pytest.mark.low_priority
@pytest.mark.usefixtures("setup_random_seed")
def test_merge_sort_min_blocklen_pto():
    """Minimum block length for pto (>=4, 32-byte tile) (low_priority)."""
    run_test_merge(block_len=4, num_ways=2, target="pto")


@pytest.mark.low_priority
@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize(
    "target",
    ["ascendc", "pto"],
)
def test_merge_sort_stability(target):
    """Equal values keep source order and in-block order (stable merge)."""
    run_test_merge_stability(block_len=64, target=target)


@pytest.mark.low_priority
@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize(
    "target,block_len",
    [
        ("ascendc", 4095),  # hardware elementLengths upper bound
        pytest.param("pto", 4088, marks=pytest.mark.low_priority),  # pto segfaults for block_len >= 4092
    ],
)
def test_merge_sort_max_blocklen(target, block_len):
    """Maximum block length. pto limit is lower (tested 4088 OK, >=4092 crashes)."""
    run_test_merge(block_len=block_len, num_ways=2, target=target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T.tile.merge_sort test suite")
    parser.add_argument("--target", type=str, choices=["ascendc", "pto"], default="ascendc")
    parser.add_argument("--num-ways", type=int, choices=[2, 3, 4], default=3)
    parser.add_argument("--block-len", type=int, default=64, help="elements per source block")
    args = parser.parse_args()

    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    print("=" * 60)
    print(f"T.tile.merge_sort test: target={args.target}, {args.num_ways}-way, block_len={args.block_len}")
    print("=" * 60)

    run_test_merge(block_len=args.block_len, num_ways=args.num_ways, target=args.target)
    print("  merge PASSED")

    run_test_merge_unequal([args.block_len, args.block_len // 2], target=args.target)
    print("  unequal PASSED")

    run_test_merge_stability(block_len=args.block_len, target=args.target)
    print("  stability PASSED")

    print("\nAll tests passed.")
