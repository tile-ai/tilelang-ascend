"""Regression tests for issue #1194: T.Parallel compact-to-aligned UB store.

A T.Parallel copy from a compact UB layout to an aligned/padded UB layout
(different inner/row width, e.g. dst[g, inner] = src[g, inner] with src [G, 8]
and dst [G, 16]) used to be miscompiled into a copy_ub_to_ub whose source width
was taken from the destination, packing two source rows into one destination row
and producing wrong results.

The fix refuses to vectorize such UB->UB stride-mismatched copies and lowers any
T.Parallel that cannot be vectorized to a scalar serial loop (codegen emits
SetValue / GetValue element by element).  The same serial fallback also fixes
the "Find undefined Variable v_thread" codegen error for parallel loops whose
body cannot be vectorized (e.g. a data-dependent if/else).

All row widths here are 32B-aligned (int32 multiples of 8, float16 multiples of
16) on purpose, so the surrounding GM<->UB T.copy is valid and only the
T.Parallel compact->aligned lowering is exercised.  Sub-32B T.copy is a separate
issue and intentionally not covered here.

NOTE: these kernels target Ascend NPU and must be run on NPU hardware.
"""

import pytest
import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


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


# ---------------------------------------------------------------------------
# Compact -> aligned UB store: dst[g, inner] = src[g, inner] for inner < src_cols,
# the padding columns [src_cols, dst_cols) stay zero.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def compact_to_aligned_kernel(G, src_cols, dst_cols, dtype="int32"):
    @T.prim_func
    def main(SRC: T.Tensor((G, src_cols), dtype), DST: T.Tensor((G, dst_cols), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src = T.alloc_ub((G, src_cols), dtype)
            dst = T.alloc_ub((G, dst_cols), dtype)
            with T.Scope("V"):
                T.copy(SRC, src)
                T.tile.fill(dst, 0)
                for g, inner in T.Parallel(G, src_cols):
                    dst[g, inner] = src[g, inner]
                T.copy(dst, DST)

    return main


@pytest.mark.parametrize(
    "src_cols, dst_cols, dtype",
    [
        (8, 16, "int32"),  # 32B -> 64B, stride-mismatched: wrong results before the fix
        (16, 32, "int32"),  # 64B -> 128B, stride-mismatched
        (16, 32, "float16"),  # 32B -> 64B, stride-mismatched, different dtype
        (16, 16, "int32"),  # same-width control (no padding): always correct
    ],
)
def test_parallel_compact_to_aligned(setup_random_seed, src_cols, dst_cols, dtype):
    G = 256
    func = compact_to_aligned_kernel(G, src_cols, dst_cols, dtype)
    if dtype == "int32":
        src = torch.arange(G * src_cols, dtype=torch.int32).reshape(G, src_cols).npu()
    else:
        src = torch.randn(G, src_cols).to(torch.float16).npu()

    torch.npu.synchronize()
    out = func(src)

    torch.testing.assert_close(out[:, :src_cols], src, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out[:, src_cols:], torch.zeros_like(out[:, src_cols:]), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Data-dependent if/else in a T.Parallel body. The body cannot be vectorized;
# before the fix the surviving kParallel loop tripped "undefined Variable
# v_thread" at codegen. The serial fallback now lowers it to a scalar loop
# (equivalent to relu) and it runs correctly.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def relu_if_else_1d_kernel(N, block_N, dtype="float"):
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(n_num, is_npu=True) as (cid, vid):
            a = T.alloc_ub((block_N,), dtype)
            c = T.alloc_ub((block_N,), dtype)
            with T.Scope("V"):
                T.copy(A[cid * block_N], a)
                for j in T.Parallel(block_N):
                    if a[j] > 0:
                        c[j] = a[j]
                    else:
                        c[j] = 0.0
                T.copy(c, C[cid * block_N])

    return main


def test_parallel_if_else_serial_fallback(setup_random_seed):
    N, block_N = 1024, 128
    func = relu_if_else_1d_kernel(N, block_N)
    a = torch.randn(N).npu()
    torch.npu.synchronize()
    out = func(a)
    torch.testing.assert_close(out, torch.relu(a), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "0"])
