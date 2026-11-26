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

The fix folds the leading dims into the row count (the "total plane", e.g.
(16, 4, 8) -> (64, 8)) under four guards:

1. Memory uniformity: every active row dim except the outermost fully AND
   legally covers the buffer dims it spans (e.g. ``A[t:t+16, :, :]`` on
   ``(T, 4, 8)``) -- its extent equals the group span AND its range starts
   at the group base (min == 0), so consecutive row starts keep a constant
   pitch and the flattened plane is the contiguous group span. The
   min check rejects a shifted slice whose extent still equals the full
   span (``A[t:t+16, 1:5, :]`` over a size-4 dim, an out-of-bounds region):
   extent-only checking folded that into a contiguous DMA whose bursts
   walked across the neighboring tokens and past the end of the buffer.
   The tail-block clamp is applied to the outermost dim before folding.
2. UB-side tail-gap freedom: on the UB side, the dims between the last row
   dim and the col dim must all have shape 1. The UB burst pitch is bound to
   the last-dim template (dstN/srcN), so a singleton dim with shape > 1
   there (e.g. shape ``(T, 4, 2, 8)`` with region ``[16, 4, 1, 8]``:
   physical row pitch 2*8 = 16) makes the fold unrepresentable and must be
   rejected. The GM side needs no such check -- its pitch rides the free
   strideN argument.
3. Both sides must fold: the fold commits the DMA to the flattened row count
   on both sides, so it is all-or-nothing; a mismatch (a clean GM region
   landing in a differently shaped UB buffer) falls back to the unfolded 2D
   form.
4. DMA row-count limit: ``DataCopyExtParams.blockCount`` is a 12-bit
   hardware field (at most 4095 bursts per ``DataCopyPad``), but a folded
   region can easily exceed it -- a (64, 64, 8) float32 block is only
   128 KiB of UB yet folds to 4096 rows. The copy helpers therefore split
   the transfer into batches of at most 4095 rows
   (``MAX_DMA_ROWS`` in src/tl_templates/ascend/common.h), advancing the
   src/dst tensors by the per-batch element offset.

How this test triggers it
-------------------------
Full-block round-trips (``T.copy(A[t:t+BT, :, :], ub)`` and back) cover the
fold for 3D and 4D blocks, with ``num_tokens`` both divisible (full blocks)
and non-divisible (tail blocks, runtime clamp on the outermost dim) by the
block size. The 4D sliced cases pin guard 2: ``[t:t+16, 0:4, 0:1, 0:8]`` on
``(T, 4, 2, 8)`` must NOT fold when the UB tile mirrors the GM buffer (the
reviewer's counterexample -- a folded DMA would scatter rows across wrong UB
offsets), but must still fold and round-trip exactly when the UB tile
squeezes the singleton dim (``(16, 4, 1, 8)``, GM pitch via strideN). A
contiguous 4D full block keeps folding as the over-correction guard.

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


def gm_ub_gm_4d_sliced_copy(ub_tail_shape, dtype):
    """4D copy with a singleton dim between the row dim and the col dim.

    GM buffer (num_tokens, 4, 2, 8); the region fixes dim2 to one element
    ([t:t+16, 0:4, 0:1, 0:8]), so the physical row pitch is 2*8 = 16, not 8.
    ``ub_tail_shape`` is the UB tile's trailing shape: (4, 2, 8) mirrors the
    GM buffer (the reviewer's counterexample -- unrepresentable on the UB
    side, must NOT fold), while (4, 1, 8) squeezes the singleton dim (the
    pitch is 8 there, and the GM-side gap rides the free strideN arg, so
    the fold is legal and must stay enabled).
    """
    num_tokens = T.symbolic("num_tokens")

    @T.prim_func
    def main(
        A: T.Tensor((num_tokens, 4, 2, 8), dtype),
        C: T.Tensor((num_tokens, 4, 2, 8), dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, 16), is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((16,) + tuple(ub_tail_shape), dtype)
            row_base = cid * 16
            T.copy(A[row_base : row_base + 16, 0:4, 0:1, 0:8], a_ub[:, 0:4, 0:1, 0:8])
            T.copy(a_ub[:, 0:4, 0:1, 0:8], C[row_base : row_base + 16, 0:4, 0:1, 0:8])

    return main


def test_4d_sliced_copy_must_not_fold():
    """A singleton dim with shape > 1 between the last row dim and the col
    dim makes the fold unrepresentable on the UB side: the physical row
    pitch is 16 while the dstN template (last-dim bound) can only express 8.
    The lowering must reject the fold and keep the pre-existing unfolded 2D
    form -- folding here would scatter rows across wrong UB offsets."""
    torch.manual_seed(0)
    tilelang.disable_cache()
    try:
        func = gm_ub_gm_4d_sliced_copy((4, 2, 8), "float")
        func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=TARGET)
    finally:
        tilelang.enable_cache()
    src = func.get_kernel_source()
    copy_lines = [ln for ln in src.splitlines() if "copy_gm_to_ub" in ln or "copy_ub_to_gm" in ln]
    assert copy_lines, "expected copy calls in the generated source"
    # The folded form would carry the flattened row count (16*4 = 64) in the
    # template / maskShapeM; it must be absent.
    assert not any("<float, 8, 64>" in ln for ln in copy_lines), (
        "4D sliced copy with a shape>1 singleton between row and col dims must not fold: " + copy_lines[0].strip()
    )


def test_4d_sliced_copy_gm_gap_ub_clean_roundtrip():
    """Same sliced region, but the UB tile squeezes the singleton dim
    ((16, 4, 1, 8)): the UB row pitch is 8, matching the last-dim template,
    while the GM-side 16 pitch rides the free strideN argument. The fold is
    legal and the sliced block must round-trip exactly."""
    torch.manual_seed(0)
    tilelang.disable_cache()
    try:
        func = gm_ub_gm_4d_sliced_copy((4, 1, 8), "float")
        func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=TARGET)
    finally:
        tilelang.enable_cache()
    for num_tokens in [64, 100]:
        a = torch.randn(num_tokens, 4, 2, 8, dtype=torch.float32).npu()
        torch.npu.synchronize()
        c = func(a)
        torch.npu.synchronize()
        # Only the s=0 sub-row of each (t, h) is round-tripped.
        torch.testing.assert_close(c[:, :, 0:1, :].cpu(), a[:, :, 0:1, :].cpu(), rtol=1e-2, atol=1e-2)


def test_4d_contiguous_full_block_roundtrip():
    """A contiguous 4D full block (no singleton gap) must keep folding: (16,
    4, 2, 8) flattens to a (128, 8) plane. Guards against over-correcting
    the sliced rejection above."""
    torch.manual_seed(0)
    num_tokens = T.symbolic("num_tokens")

    @T.prim_func
    def main(
        A: T.Tensor((num_tokens, 4, 2, 8), "float"),
        C: T.Tensor((num_tokens, 4, 2, 8), "float"),
    ):
        with T.Kernel(T.ceildiv(num_tokens, 16), is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((16, 4, 2, 8), "float")
            row_base = cid * 16
            T.copy(A[row_base : row_base + 16, :, :, :], a_ub[:, :, :, :])
            T.copy(a_ub[:, :, :, :], C[row_base : row_base + 16, :, :, :])

    tilelang.disable_cache()
    try:
        func = tilelang.compile(main, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=TARGET)
    finally:
        tilelang.enable_cache()
    for num_tokens in [64, 100]:
        a = torch.randn(num_tokens, 4, 2, 8, dtype=torch.float32).npu()
        torch.npu.synchronize()
        c = func(a)
        torch.npu.synchronize()
        torch.testing.assert_close(c.cpu(), a.cpu(), rtol=1e-2, atol=1e-2)


def gm_ub_gm_3d_mid_offset_copy(dtype):
    """3D copy whose middle dim slice starts off the group base.

    GM buffer (num_tokens, 4, 8); the region slices the middle (head) dim as
    [1:5] -- extent 4 == shape 4, but min is 1, an out-of-bounds region.
    The extent-only uniformity check used to fold it into a 64-row
    contiguous DMA: each token's bursts crossed the head boundary into the
    next token and the final burst read 32 bytes past the end of the buffer
    (the reviewer's counterexample). The fold must be rejected and the
    copy fall back to the unfolded 2D form."""
    num_tokens = T.symbolic("num_tokens")

    @T.prim_func
    def main(
        A: T.Tensor((num_tokens, 4, 8), dtype),
        C: T.Tensor((num_tokens, 4, 8), dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, 16), is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((16, 4, 8), dtype)
            row_base = cid * 16
            T.copy(A[row_base : row_base + 16, 1:5, :], a_ub[:, 1:5, :])
            T.copy(a_ub[:, 1:5, :], C[row_base : row_base + 16, 1:5, :])

    return main


def test_3d_mid_dim_offset_must_not_fold():
    """A middle-dim slice with min != 0 whose extent still equals the full
    span must not fold: the flattened DMA would read across the token
    boundary and past the end of the buffer. The lowering must reject the
    fold and keep the pre-existing unfolded 2D form."""
    torch.manual_seed(0)
    tilelang.disable_cache()
    try:
        func = gm_ub_gm_3d_mid_offset_copy("float")
        func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=TARGET)
    finally:
        tilelang.enable_cache()
    src = func.get_kernel_source()
    copy_lines = [ln for ln in src.splitlines() if "copy_gm_to_ub" in ln or "copy_ub_to_gm" in ln]
    assert copy_lines, "expected copy calls in the generated source"
    # The folded form would carry the flattened row count (16*4 = 64) in the
    # template / maskShapeM; it must be absent.
    assert not any("<float, 8, 64>" in ln for ln in copy_lines), (
        "3D copy with a middle-dim slice that starts off the group base (min != 0) must not fold: " + copy_lines[0].strip()
    )


@pytest.mark.parametrize(
    "block_shape, dtype, num_tokens_list",
    [
        # (455, 9, 8): 455*9 = 4095 folded rows -- exactly the
        # DataCopyExtParams.blockCount limit, still a single batch.
        ((455, 9, 8), "float", [455, 500]),
        # (64, 64, 8): 64*64 = 4096 folded rows -- one row past the limit
        # (the reviewer's counterexample, 128 KiB of UB), split into
        # 4095 + 1.
        ((64, 64, 8), "float", [128, 100]),
        # (65, 64, 16) fp16: 4160 folded rows, split into 4095 + 65; the
        # tail block of num_tokens=194 keeps 64 tokens -> 4096 rows, so
        # both the full and the tail block take the multi-batch path. 16
        # fp16 cols make 32B rows (the memory planner models unaligned
        # fp16 rows at double footprint, which is a pre-existing quirk
        # this shape avoids).
        ((65, 64, 16), "float16", [130, 194]),
    ],
)
def test_3d_copy_folded_rows_beyond_dma_limit_roundtrip(block_shape, dtype, num_tokens_list):
    """A folded row count above the single-DMA blockCount limit (4095) must
    still copy every element: the helpers batch the transfer, and both the
    full blocks (row count above the limit) and tail blocks (runtime clamp
    below it) must round-trip exactly."""
    run_test_nd_copy_roundtrip(block_shape, dtype, num_tokens_list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
