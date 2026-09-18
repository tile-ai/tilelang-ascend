import pytest
import tilelang
import tilelang.language as T
import torch

"""
Regression test for flattening a >2D BufferRegion in the AscendC GM<->UB copy
lowering (src/op/ascend.cc :: AscendCopy::Lower). Issue #1231.

Feature under test
------------------
``copy_gm_to_ub`` / ``copy_ub_to_gm`` model their transfer as a 2D (rows x
cols) DMA rectangle. The lowering used to derive that rectangle from the LAST
two active dims of the region only: a 3D region (16, 4, 8) was flattened to
(4, 8), silently DROPPING the leading dims. The batch DMA then copied only 32
elements (token 0) and left the other 15 tokens of the UB tile uninitialized
-- silent data corruption for any 3D block copy.

The fix: when the region is memory-uniform (every active row dim except the
outermost fully covers the buffer dims it spans -- e.g. ``A[t:t+16, :, :]`` on
a ``(T, 4, 8)`` tensor), the leading dims fold into the row count, giving the
"total plane" (64, 8): the DMA copies ``prod(leading extents)`` bursts of the
last-dim width, and the tail-block clamp is applied to the outermost dim
before folding.

How this test triggers it
-------------------------
Round-trips a 3D (and 4D) block through UB: ``T.copy(A[t:t+BT, :, :], ub)``
followed by ``T.copy(ub, C[t:t+BT, :, :])``. Both the gm2ub load (UB is the
destination) and the ub2gm store (UB is the source) exercise the fold. Sizes
are chosen so the last dim is 32B-aligned for float32 (pad = 8) and float16
(pad = 16). ``num_tokens`` is used both divisible (full blocks) and
non-divisible (tail blocks, runtime clamp on the outermost dim) by the block
size. Before the fix every token except the first of each block is garbage;
after it, the full block round-trips exactly.

This targets the ascendc backend only (the fix is in the non-PTO copy
lowering; the PTO codegen consumes the same validRow/validCol args).
"""

TARGET = "ascendc"

VEC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


def _torch_dtype(dtype):
    return {
        "float": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def gm_ub_gm_nd_copy(block_shape, dtype):
    """Round-trip one N-D block per kernel through a same-shaped UB tile."""
    num_tokens = T.symbolic("num_tokens")
    block_tokens = block_shape[0]

    @T.prim_func
    def main(
        A: T.Tensor((num_tokens,) + tuple(block_shape[1:]), dtype),
        C: T.Tensor((num_tokens,) + tuple(block_shape[1:]), dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_tokens), is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub(tuple(block_shape), dtype)
            row_base = cid * block_tokens
            # Full-rank block slices: (row_base : row_base + block_tokens, :, ...)
            # with one ":" per trailing dim.
            if len(block_shape) == 3:
                T.copy(A[row_base : row_base + block_tokens, :, :], a_ub[:, :, :])
                T.copy(a_ub[:, :, :], C[row_base : row_base + block_tokens, :, :])
            else:
                T.copy(A[row_base : row_base + block_tokens, :, :, :], a_ub[:, :, :, :])
                T.copy(a_ub[:, :, :, :], C[row_base : row_base + block_tokens, :, :, :])

    return main


def run_test_nd_copy_roundtrip(block_shape, dtype, num_tokens_list):
    torch.manual_seed(0)
    tilelang.disable_cache()
    try:
        func = gm_ub_gm_nd_copy(block_shape, dtype)
        func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=TARGET)
    finally:
        tilelang.enable_cache()
    td = _torch_dtype(dtype)
    for num_tokens in num_tokens_list:
        shape = (num_tokens,) + tuple(block_shape[1:])
        a = torch.randn(shape, dtype=td).npu()
        torch.npu.synchronize()
        c = func(a)
        torch.npu.synchronize()
        # The full 3D block (not just its first (h, pad) plane) must round-trip.
        torch.testing.assert_close(c.cpu(), a.cpu(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
def test_3d_copy_full_blocks(dtype):
    # (16, 4, 8): the exact shape from issue #1231. 64 = 100 % 32 tail tokens.
    run_test_nd_copy_roundtrip((16, 4, 8), dtype, num_tokens_list=[64])


@pytest.mark.parametrize("dtype", ["float", "float16"])
def test_3d_copy_tail_blocks(dtype):
    # 100 tokens / 16 per block -> a tail block of 4 tokens: the outermost-dim
    # clamp folds into validRow and the remaining UB rows are pad-filled.
    run_test_nd_copy_roundtrip((16, 4, 8), dtype, num_tokens_list=[100])


def test_4d_copy_full_blocks():
    # A 4D region (2, 8, 4, 8) exercises multi-level leading-dim folding.
    run_test_nd_copy_roundtrip((2, 8, 4, 8), "float", num_tokens_list=[8])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
