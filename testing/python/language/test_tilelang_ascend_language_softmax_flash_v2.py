import pytest
import tilelang
import tilelang.language as T
import torch

"""
Regression test for the non-PTO ``softmax_flash_v2`` primitive on the AscendC
backend (src/tl_templates/ascend/common.h).

Feature under test
------------------
``softmax_flash_v2`` is a flash-attention online-softmax primitive: a single call
does the variable-N softmax over a window with a running max/sum, wrapping the
AscendC ``SoftmaxFlashV2`` library. For an ``[M, N]``-strided score tile whose
first ``actual_col`` columns are the live window it computes, per row ``i``::

    new_max      = max(in_max[i], max_j score[i, j])          for j in [0, actual_col)
    dst[i, j]    = exp(score[i, j] - new_max)                 for j in [0, col_count)
    out_sum[i]   = in_sum[i] * exp(in_max[i] - new_max) + sum_j exp(score[i,j]-new_max)
    out_max[i]   = new_max
    expmax[i]    = exp(in_max[i] - new_max)                   (flash rescale factor)

``in_max`` / ``in_sum`` are the running max / sum carried in from earlier blocks
(seeded by a per-row sink and 1.0 for the first block), so the same primitive
serves both a single softmax pass and the multi-block online merge of flash
attention. The window width is a *runtime* pair: ``col_count`` is the (16-aligned)
buffer stride the library runs on and ``actual_col`` (<= col_count) is the number
of columns actually reduced -- the ``[actual_col, col_count)`` alignment tail is
exp'd but excluded from the reduce, and the uninitialised ``[col_count, N)`` tail
is never touched (the template compacts ``src[:, 0:col_count]`` into a contiguous
tile first, so the library stays in its designed range).

Existing decomposed ``reduce_max`` / ``reduce_sum`` cannot express this: their N is
a compile-time template constant, so a variable window forces a full-width reduce
plus an explicit mask. This primitive fills that gap.

How this test exercises it
--------------------------
A GM->UB load feeds one ``softmax_flash_v2`` call (dst aliases src, in place) whose
P / out_sum / out_max / expmax are stored UB->GM and checked against a torch reference. Two
regimes are covered: ``seed`` (in_sum = 1.0, first block) and ``update`` (in_sum /
in_max carried from a prior block -- the flash online-merge path). ``col_count`` <
N and ``actual_col`` < col_count exercise the variable window and the alignment
tail. This targets the ascendc backend only (the PTO path has no such primitive
and is untouched).

The per-row scalars (in/out sum, in/out max, expmax) are carried as a contiguous
``[1, M]`` row rather than an ``[M, 1]`` column: the primitive writes M contiguous
values either way, but an ``[M, 1]`` column makes the UB->GM ``copy_ub_to_gm`` read
at a 32B-block stride (one element per 8-float block), which shuffles the rows in
the copied-out result. A ``[1, M]`` row copies as one contiguous block.
"""

TARGET = "ascendc"
SOFTMAX_TMP_BYTES = 32768

VEC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear the tilelang cache before the session (a stale kernel could mask a
    codegen regression -- a rebuilt kernel could otherwise return a cached one)."""
    tilelang.cache.clear_cache()
    yield


def _torch_dtype(dtype):
    return {
        "float": torch.float32,
        "float16": torch.float16,
    }[dtype]


def softmax_flash_v2_kernel(M, N, col_count, actual_col, dtype):
    """Load a score tile [M, N] plus running max/sum seeds (each a [1, M] row) into
    UB, run one ``softmax_flash_v2`` (dst aliases src, in place), store P / out_sum /
    out_max. ``compact`` is the [M, N]-capacity contiguous scratch the template
    compacts the window into; ``tmp`` is the library's uint8 shared scratch."""

    @T.prim_func
    def main(
        Src: T.Tensor((M, N), dtype),  # type: ignore  score, first actual_col cols live
        InMax: T.Tensor((1, M), dtype),  # type: ignore  per-row running-max seed (sink / prev)
        InSum: T.Tensor((1, M), dtype),  # type: ignore  per-row running-sum seed (1.0 / prev)
        Dst: T.Tensor((M, N), dtype),  # type: ignore  P = exp(score - new_max)
        OutSum: T.Tensor((1, M), dtype),  # type: ignore  per-row new running sum
        OutMax: T.Tensor((1, M), dtype),  # type: ignore  per-row new running max
        ExpMax: T.Tensor((1, M), dtype),  # type: ignore  per-row rescale exp(in_max-new_max)
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s_ub = T.alloc_ub((M, N), dtype)
            # Per-row scalars as [1, M] rows -- see the module docstring: an [M, 1]
            # column would be shuffled by the 32B-block-strided UB<->GM copy.
            inmax_ub = T.alloc_ub((1, M), dtype)
            insum_ub = T.alloc_ub((1, M), dtype)
            outsum_ub = T.alloc_ub((1, M), dtype)
            outmax_ub = T.alloc_ub((1, M), dtype)
            expmax_ub = T.alloc_ub((1, M), dtype)
            compact = T.alloc_ub((M, N), dtype)
            tmp = T.alloc_ub((SOFTMAX_TMP_BYTES,), "uint8")
            T.copy(Src, s_ub)
            T.copy(InMax, inmax_ub)
            T.copy(InSum, insum_ub)
            # dst == src == s_ub: the library runs in place (isReuseSource).
            T.tile.softmax_flash_v2(
                s_ub,
                outsum_ub,
                outmax_ub,
                expmax_ub,
                s_ub,
                insum_ub,
                inmax_ub,
                tmp,
                compact,
                col_count,
                actual_col,
            )
            T.copy(s_ub, Dst)
            T.copy(outsum_ub, OutSum)
            T.copy(outmax_ub, OutMax)
            T.copy(expmax_ub, ExpMax)

    return main


def run_test_softmax_flash_v2(M, N, col_count, actual_col, mode, dtype):
    torch.manual_seed(0)
    func = softmax_flash_v2_kernel(M, N, col_count, actual_col, dtype)
    func = tilelang.compile(func, out_idx=[3, 4, 5, 6], pass_configs=VEC_PASS_CONFIGS, target=TARGET)
    td = _torch_dtype(dtype)

    # Scores kept modest so exp() stays well inside range for both dtypes. Per-row
    # seeds are [1, M] rows (element j is row j's seed).
    src = torch.randn(M, N, dtype=td).npu()
    if mode == "seed":
        # First block: sum seed is 1.0, max seed is a per-row sink (below the row max).
        in_max = (torch.randn(1, M, dtype=td) - 4.0).npu()
        in_sum = torch.ones(1, M, dtype=td).npu()
    else:  # update: max/sum carried in from a prior block (the online-merge path).
        in_max = torch.randn(1, M, dtype=td).npu()
        in_sum = (torch.rand(1, M, dtype=td) + 1.0).npu()
    torch.npu.synchronize()
    dst, out_sum, out_max, expmax = func(src, in_max, in_sum)

    # torch reference (float math), only over the live window [0, actual_col).
    # Reshape the [1, M] rows to [M, 1] columns for the per-row broadcast.
    s = src.float()
    imx = in_max.float().reshape(M, 1)
    ism = in_sum.float().reshape(M, 1)
    row_max = s[:, :actual_col].max(dim=1, keepdim=True).values
    new_max = torch.maximum(imx, row_max)
    p = torch.exp(s[:, :actual_col] - new_max)
    ref_sum = ism * torch.exp(imx - new_max) + p.sum(dim=1, keepdim=True)
    ref_expmax = torch.exp(imx - new_max)

    rtol, atol = 2e-2, 2e-2
    torch.testing.assert_close(dst.float().cpu()[:, :actual_col], p.cpu(), rtol=rtol, atol=atol)
    torch.testing.assert_close(out_sum.float().cpu().reshape(M, 1), ref_sum.cpu(), rtol=rtol, atol=atol)
    torch.testing.assert_close(out_max.float().cpu().reshape(M, 1), new_max.cpu(), rtol=rtol, atol=atol)
    torch.testing.assert_close(expmax.float().cpu().reshape(M, 1), ref_expmax.cpu(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("mode", ["seed", "update"])
def test_softmax_flash_v2(mode, dtype):
    # M=64, N=128; col_count=64 (< N, variable window) and actual_col=60
    # (< col_count, a >=16 alignment tail). col_count and N are multiples of
    # 32/sizeof(dtype) as the template requires.
    run_test_softmax_flash_v2(64, 128, 64, 60, mode, dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
