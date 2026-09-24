"""UB->GM ``T.copy`` path coverage (issue #1682).

``copy_ub_to_gm`` always emitted one multi-burst ``DataCopyPad``. The MTE3
engine implements the system-wide "narrow UB rows sit in 32B slots" layout
convention: the source address advances ``ceil(blockLen/32) + srcStride``
whole 32B blocks per burst, which matches every producer that writes rows on
32B boundaries (MTE-filled buffers, the packed-mask helpers, and any srcPitch
that is a 32B multiple - the GM-side stride is byte granular, so per-burst
destination addresses need no alignment, as the long-standing tail-block
tests prove).

What the engine cannot express is a *compactly packed* sub-32B-pitch source
with more than one row -- e.g. the ``(M, 1)`` fp32 keepdim output of
``T.reduce_max/min/sum`` written straight back to GM (issue #1682): V-pipe
writes rows 4B apart while the engine steps 32B per burst.

The helper now picks between a single flat burst (packed src+dst, fixes
#1682) and the classic multi-burst (every 32B-slotted source). These tests
pin each path down, plus the loop-carried UB reuse that needs an explicit
producer sync once the copy sits under a runtime conditional.
"""

import pytest
import torch

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[0], target="ascendc")
def copy_packed(M: int = 16, N: int = 1, dtype: str = "float32"):
    @T.prim_func
    def main(B: T.Tensor((M, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((M, N), dtype)
            T.tile.arith_progression(a_ub, 0.0, 1.0, M * N)
            T.copy(a_ub, B)

    return main


@tilelang.jit(out_idx=[0], target="ascendc")
def copy_strided(
    M: int = 16,
    N: int = 96,
    src_pitch: int = 96,
    dst_pitch: int = 128,
    dtype: str = "float32",
):
    """Copy an (M, N) tile: the UB buffer and the GM rows both have their own
    pitch, so realdstN/srcN exercise the stride arithmetic."""

    @T.prim_func
    def main(B: T.Tensor((M, dst_pitch), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((M, src_pitch), dtype)
            T.tile.arith_progression(a_ub, 0.0, 1.0, M * src_pitch)
            T.copy(a_ub, B[0:M, 0:N])

    return main


@tilelang.jit(
    out_idx=[0],
    target="ascendc",
    pass_configs={tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True},
)
def copy_rows(M: int = 8, W: int = 126, pitch: int = 126, dtype: str = "float32"):
    """Copy a 1D row into each row of a (M, pitch) tensor. With W=126 fp32 the
    row destinations alternate between aligned (multi-burst) and misaligned
    (scalar fallback) addresses."""

    @T.prim_func
    def main(B: T.Tensor((M, pitch), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((W,), dtype)
            for m in T.serial(M):
                T.tile.arith_progression(a_ub, T.cast(m, dtype), 1.0, W)
                T.copy(a_ub, B[m, 0])

    return main


@tilelang.jit(
    out_idx=[0],
    target="ascendc",
    pass_configs={tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True},
)
def copy_keepdim_reduce(M: int = 16, N: int = 128, dtype: str = "float32"):
    """The issue #1682 repro: reduce_max keepdim output copied straight to GM
    with 4-byte rows (blockLen < 32B)."""

    @T.prim_func
    def main(B: T.Tensor((M, 1), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((M, 1), dtype)
            T.tile.arith_progression(a_ub, 0.0, 1.0, M * N)
            T.reduce_max(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B)

    return main


@tilelang.jit(
    out_idx=[0],
    target="ascendc",
    pass_configs={tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True},
)
def copy_reuse_ub_in_loop(M: int = 512, N: int = 128, blk: int = 64, dtype: str = "float32"):
    """One UB buffer reused across loop iterations: iteration i+1's producer
    must not overwrite data iteration i's copy still reads."""

    @T.prim_func
    def main(B: T.Tensor((M, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((blk, N), dtype)
            for it in T.serial(M // blk):
                T.tile.arith_progression(a_ub, T.cast(it, dtype), 1.0, blk * N)
                T.copy(a_ub, B[it * blk : (it + 1) * blk, :])

    return main


@pytest.mark.parametrize(
    "M,N,dtype",
    [
        (16, 1, "float32"),  # issue #1682: 4-byte rows
        (3, 1, "float32"),  # flat copy below 32B total
        (5, 3, "float32"),  # odd width, neither side 32B-aligned
        (16, 100, "float32"),  # packed, non-32B-multiple row
        (16, 1, "float16"),  # 2-byte rows
    ],
)
def test_copy_ub_to_gm_packed(M, N, dtype):
    out = copy_packed(M, N, dtype)()
    torch.npu.synchronize()
    ref = torch.arange(M * N, dtype=torch.float32).reshape(M, N)
    if dtype == "float16":
        ref = ref.half()
    assert torch.equal(out.cpu(), ref)


def test_copy_ub_to_gm_keepdim_reduce():
    out = copy_keepdim_reduce(16, 128)()
    torch.npu.synchronize()
    ref = torch.arange(16 * 128, dtype=torch.float32).reshape(16, 128)
    ref = ref.max(dim=-1, keepdim=True).values
    assert torch.equal(out.cpu(), ref)


def test_copy_ub_to_gm_aligned_strided():
    out = copy_strided(16, 96, 96, 128)()
    torch.npu.synchronize()
    ref = torch.arange(16 * 96, dtype=torch.float32).reshape(16, 96)
    assert torch.equal(out.cpu()[:, :96], ref)


def test_copy_ub_to_gm_misaligned_dst_pitch():
    out = copy_strided(16, 112, 112, 126)()
    torch.npu.synchronize()
    ref = torch.arange(16 * 112, dtype=torch.float32).reshape(16, 112)
    assert torch.equal(out.cpu()[:, :112], ref)


def test_copy_ub_to_gm_rows_mixed_alignment():
    M, W = 8, 126
    out = copy_rows(M, W, W)()
    torch.npu.synchronize()
    ref = torch.arange(M, dtype=torch.float32).unsqueeze(1) + torch.arange(W, dtype=torch.float32).unsqueeze(0)
    assert torch.equal(out.cpu(), ref)


def test_copy_ub_to_gm_reuse_ub_in_loop():
    M, N, blk = 512, 128, 64
    out = copy_reuse_ub_in_loop(M, N, blk)()
    torch.npu.synchronize()
    ref = torch.stack([torch.arange(blk * N, dtype=torch.float32).reshape(blk, N) + it for it in range(M // blk)]).reshape(M, N)
    assert torch.equal(out.cpu(), ref)
