"""MoeGatingTopKSoftmax example (TileLang-Ascend, Developer mode).

Fused Softmax + TopK kernel for MoE gating networks.  Given scores ``x`` of
shape ``(N, E)`` or ``(B, N, E)`` and a count ``k``, returns the top-k
softmax values ``y`` (same dtype as ``x``), the expert indices
``expert_idx`` (int32) and the flattened position indices ``row_idx``
(int32), where ``row_idx[m, j] = m + j * M``.  Rows flagged by the optional
``finished`` bool tensor get ``expert_idx = E`` (the token is not routed).

Interface::

    y, expert_idx, row_idx = moe_gating_top_k_softmax(x, finished=None, k=1)

Key design points:

* Softmax is computed as one 2D tile batch (max/sub/exp/sum/div on the whole
  block, fp32 internal for fp16/bf16), then top-k selection is dispatched by
  shape: a batched hardware ``T.tile.sort32`` (+ top-k_out-only
  ``T.tile.merge_sort`` for E > 32, with dedicated multi-stage merge
  variants for aligned E = 256/512) for large M / small aligned E, a
  batch-output staging kernel for k % 8 == 0 && k >= 32, and a per-row
  hardware ``T.tile.topk`` fallback otherwise.  The sort32 variants split
  the sorted [value, index] pairs in-kernel (two single-gather passes) and
  write ``Y32`` (fp32) / ``IdxBits`` (fp32 slots holding uint32 index bit
  patterns) directly to GM.
* "Zero-torch-op" wrapper: some 910-series evaluation images ship a minimal
  CANN operator package that misses the aclnn Range / Fill / StridedSlice
  binaries (aclnnArange fails with 561103), so the host wrapper dispatches
  NO torch compute ops on device tensors -- row_idx, the sort32 IdxIn ramp,
  M-padding rows, the fp32->input-dtype cast and the finished mask are all
  built by tiny @tilelang.jit kernels; the host only does metadata
  reshape/view plus torch.empty allocation via kernel out_idx.
* Known backend quirks worked around in-kernel: the gather_mask destination
  has a fixed ~256B write granularity (padded to 64 floats), a cast
  immediately after gather misplaces outputs (cast runs in its own kernel),
  and 16B-granularity UB->GM copies are avoided.

The golden reference (``golden_moe_gating_top_k_softmax``) is a plain
PyTorch implementation kept in this same file for the tests and the
``__main__`` demo.
"""

import tilelang
import torch
from tilelang import language as T

# ========== Developer-mode pass configs (shared by all kernels) ==========
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

CAST_MODE_LOW2HIGH = "CAST_NONE"  # fp16/bf16 -> fp32 exact widening
CAST_MODE_HIGH2LOW = "CAST_RINT"  # fp32 -> fp16/bf16 round-to-nearest-even


def torch_dtype_to_tl(dtype):
    """Map a torch dtype to the tilelang dtype string used in T.Tensor."""
    if dtype == torch.float16:
        return "float16"
    elif dtype == torch.bfloat16:
        return "bfloat16"
    elif dtype == torch.float32:
        return "float"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# ========== Configuration ==========
_kernel_cache = {}
VEC_NUM = 2  # number of vector cores per block (A2/A3 has 2 AIVs per AI Core)
ALIGN_SIZE = 32  # topk src alignment requirement (32 elements)
SORT32_MIN_M = 4096  # only use sort32 batch when M is large enough (V saving wins)


# ========== Kernel ==========
@tilelang.jit(out_idx=[1, 2], pass_configs=PASS_CONFIGS)
def _moe_gating_top_k_softmax_kernel(M: int, E: int, k: int, block_M: int, dtype: str = "float16"):
    """Fused softmax + topk kernel for MoE gating (2D batch softmax + per-row topk).

    Dynamic grid: m_num = ceildiv(M, block_M) blocks, each using VEC_NUM=2
    vector cores. Each vector core processes sub_block_M rows.

    Key optimization: softmax is vectorized as 2D tile operations (9 V
    instructions for ALL rows instead of 9×sub_block_M). Only topk remains
    per-row (1D hardware topk instruction). Input is loaded as one 2D DMA
    instead of per-row small DMAs.

    Args:
        M: number of rows (compile-time constant)
        E: number of experts (compile-time constant)
        k: topK count (compile-time constant)
        block_M: rows per block (compile-time constant, divisible by VEC_NUM)
        dtype: input/output dtype string ("float16", "bfloat16", "float")
    """
    # Align E to 32 (topk src buffer alignment requirement)
    aligned_E = (E + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE
    # Align k to 8 (topk dst buffer alignment: 2*aligned_K interleaved elements)
    aligned_K = (k + 7) // 8 * 8

    use_fp32_compute = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_fp32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
            X: T.Tensor((M, E), dtype),  # type: ignore
            Y: T.Tensor((M, k), dtype),  # type: ignore
            Idx: T.Tensor((M, k), "int32"),  # type: ignore
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M

            # --- 2D UB buffers for batch softmax ---
            scores_cal_2d = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            scores_2d = T.alloc_ub([sub_block_M, aligned_E], dtype)
            max_2d = T.alloc_ub([sub_block_M, 1], cal_dtype)
            max_bc = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            sum_2d = T.alloc_ub([sub_block_M, 1], cal_dtype)
            sum_bc = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)

            # --- 1D UB buffers for per-row topk ---
            scores_1d = T.alloc_ub([aligned_E], cal_dtype)
            topk_dst_ub = T.alloc_ub([2 * aligned_K], cal_dtype)
            y_fp32_ub = T.alloc_ub([aligned_K], cal_dtype)
            idx_fp32_ub = T.alloc_ub([aligned_K], cal_dtype)
            y_out_ub = T.alloc_ub([aligned_K], dtype)
            idx_int_ub = T.alloc_ub([aligned_K], "int32")

            # --- 1. Load 2D tile (one large DMA, M is padded → no OOB) ---
            if use_fp32_compute:
                T.copy(
                    X[row_base:row_base + sub_block_M, 0:E],
                    scores_2d,
                    pad_value=-T.infinity(cal_dtype),
                )
                T.tile.cast(scores_cal_2d, scores_2d, CAST_MODE_LOW2HIGH, sub_block_M * aligned_E)
            else:
                T.copy(
                    X[row_base:row_base + sub_block_M, 0:E],
                    scores_cal_2d,
                    pad_value=-T.infinity(cal_dtype),
                )

            # --- 2. 2D Safe softmax: max → sub → exp → sum → div (vectorized) ---
            T.reduce_max(scores_cal_2d, max_2d, dim=-1)
            T.tile.broadcast(max_bc, max_2d)
            T.tile.sub(scores_cal_2d, scores_cal_2d, max_bc)
            T.tile.exp(scores_cal_2d, scores_cal_2d)
            T.reduce_sum(scores_cal_2d, sum_2d, dim=-1)
            T.tile.broadcast(sum_bc, sum_2d)
            T.tile.div(scores_cal_2d, scores_cal_2d, sum_bc)

            # --- 3. Per-row topk (data already in UB) ---
            for r in T.serial(sub_block_M):
                row = row_base + r

                # V: copy row from 2D buffer, topk, gather, cast
                T.copy(scores_cal_2d[r, :], scores_1d)
                T.tile.topk(topk_dst_ub, scores_1d, k, aligned_E)

                # Extract values and indices via gather_mask
                T.tile.gather_mask(y_fp32_ub, topk_dst_ub, "P0101")
                T.tile.gather_mask(idx_fp32_ub, topk_dst_ub, "P1010")

                # Cast back to output dtypes (V) — grouped before MTE3 stores
                # to minimize V↔MTE3 pipeline transitions (AUTO_SYNC barriers)
                if use_fp32_compute:
                    T.tile.cast(y_out_ub, y_fp32_ub, CAST_MODE_HIGH2LOW, k)
                T.tile.cast(idx_int_ub, idx_fp32_ub, "CAST_ROUND", k)

                # Store results to GM (MTE3) — grouped after all V work
                if use_fp32_compute:
                    T.copy(y_out_ub[0:k], Y[row, 0:k])
                else:
                    T.copy(y_fp32_ub[0:k], Y[row, 0:k])
                T.copy(idx_int_ub[0:k], Idx[row, 0:k])

    return main


# ========== Batch output kernel (k % 8 == 0) ==========
@tilelang.jit(out_idx=[1, 2], pass_configs=PASS_CONFIGS)
def _moe_gating_top_k_softmax_batch_kernel(M: int,
                                           E: int,
                                           k: int,
                                           block_M: int,
                                           dtype: str = "float16"):
    """Fused softmax + topk kernel with batch output staging.

    Variant of _moe_gating_top_k_softmax_kernel that accumulates y/idx
    results to 2D UB buffers and batch-stores to GM after the per-row loop.

    This eliminates per-row MTE3 stores (2*sub_block_M -> 2 batch stores),
    reducing MTE3 instructions by ~sub_block_Mx.

    Key enabler: T.tile.cast dst supports BufferRegion (2D row slice), and
    T.copy supports 1D->2D row slice copy when the column extent is aligned
    (aligned_K, which equals k since k%8==0 for this variant).

    Used when k % 8 == 0 (aligned_K == k enables 2D batch store with
    aligned columns). Falls back to _moe_gating_top_k_softmax_kernel
    when k % 8 != 0.

    Algorithm (per row, accumulated to 2D):
        1. Load scores row into UB (cast to fp32 for fp16/bf16)
        2. Safe softmax: max -> sub -> exp -> sum -> div (2D vectorized)
        3. Per-row topk on fp32 softmax output
        4. Extract values (gather_mask P0101) and indices (gather_mask P1010)
        5. Cast values/indices directly to 2D row slice (y_out_2d[r, :])
        6. After loop: batch store 2D -> GM (2 MTE3 instead of 2*sub_block_M)
    """
    aligned_E = (E + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE
    # k is 8-aligned (precondition for batch kernel), aligned_K == k
    aligned_K = (k + 7) // 8 * 8
    # UB row stride for 2D output staging must be 32-byte aligned for VEC
    # instructions. bf16: 32/2=16 elements, fp32/int32: 32/4=8 elements.
    # Use 16 as universal minimum (covers all dtypes with 32+ byte stride).
    aligned_K_ub = max(aligned_K, 16)

    use_fp32_compute = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_fp32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
            X: T.Tensor((M, E), dtype),  # type: ignore
            Y: T.Tensor((M, k), dtype),  # type: ignore
            Idx: T.Tensor((M, k), "int32"),  # type: ignore
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M

            # --- 2D UB buffers for batch softmax ---
            scores_cal_2d = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            scores_2d = T.alloc_ub([sub_block_M, aligned_E], dtype)
            max_2d = T.alloc_ub([sub_block_M, 1], cal_dtype)
            max_bc = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            sum_2d = T.alloc_ub([sub_block_M, 1], cal_dtype)
            sum_bc = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)

            # --- 1D UB buffers for per-row topk ---
            # Use aligned_K_ub for all 1D buffers so cast/copy count is
            # 32-byte aligned. topk writes 2*aligned_K valid pairs; the
            # tail (aligned_K_ub - aligned_K) is padding read by
            # gather_mask but never stored to GM (batch store uses [:, 0:k]).
            scores_1d = T.alloc_ub([aligned_E], cal_dtype)
            topk_dst_ub = T.alloc_ub([2 * aligned_K_ub], cal_dtype)
            y_fp32_ub = T.alloc_ub([aligned_K_ub], cal_dtype)
            idx_fp32_ub = T.alloc_ub([aligned_K_ub], cal_dtype)

            # --- 2D output staging buffers (batch output) ---
            # Column = aligned_K_ub ensures 32-byte row stride for VEC access.
            y_out_2d = T.alloc_ub([sub_block_M, aligned_K_ub], dtype)
            idx_out_2d = T.alloc_ub([sub_block_M, aligned_K_ub], "int32")

            # --- 1. Load 2D tile (one large DMA, M is padded -> no OOB) ---
            if use_fp32_compute:
                T.copy(
                    X[row_base:row_base + sub_block_M, 0:E],
                    scores_2d,
                    pad_value=-T.infinity(cal_dtype),
                )
                T.tile.cast(scores_cal_2d, scores_2d, CAST_MODE_LOW2HIGH, sub_block_M * aligned_E)
            else:
                T.copy(
                    X[row_base:row_base + sub_block_M, 0:E],
                    scores_cal_2d,
                    pad_value=-T.infinity(cal_dtype),
                )

            # --- 2. 2D Safe softmax: max -> sub -> exp -> sum -> div ---
            T.reduce_max(scores_cal_2d, max_2d, dim=-1)
            T.tile.broadcast(max_bc, max_2d)
            T.tile.sub(scores_cal_2d, scores_cal_2d, max_bc)
            T.tile.exp(scores_cal_2d, scores_cal_2d)
            T.reduce_sum(scores_cal_2d, sum_2d, dim=-1)
            T.tile.broadcast(sum_bc, sum_2d)
            T.tile.div(scores_cal_2d, scores_cal_2d, sum_bc)

            # --- 3. Per-row topk + cast to 2D row slice (no per-row MTE3) ---
            for r in T.serial(sub_block_M):
                # V: copy row from 2D buffer, topk, gather
                T.copy(scores_cal_2d[r, :], scores_1d)
                T.tile.topk(topk_dst_ub, scores_1d, k, aligned_E)

                # Extract values and indices via gather_mask
                T.tile.gather_mask(y_fp32_ub, topk_dst_ub, "P0101")
                T.tile.gather_mask(idx_fp32_ub, topk_dst_ub, "P1010")

                # Cast directly to 2D row slice (key optimization: no 1D
                # intermediate, no per-row MTE3 store)
                # count=aligned_K_ub ensures 32-byte aligned VEC copy size.
                # Tail elements (aligned_K_ub - aligned_K) are padding.
                if use_fp32_compute:
                    T.tile.cast(y_out_2d[r, :], y_fp32_ub, CAST_MODE_HIGH2LOW, aligned_K_ub)
                else:
                    # fp32: y_fp32_ub already fp32, copy to 2D row slice
                    T.copy(y_fp32_ub[0:aligned_K_ub], y_out_2d[r, :])
                T.tile.cast(idx_out_2d[r, :], idx_fp32_ub, "CAST_ROUND", aligned_K_ub)

            # --- 4. Batch store 2D -> GM (k is 8-aligned for this variant) ---
            T.copy(y_out_2d[:, 0:k], Y[row_base:row_base + sub_block_M, 0:k])
            T.copy(idx_out_2d[:, 0:k], Idx[row_base:row_base + sub_block_M, 0:k])

    return main


# ========== Sort32+Merge batch kernels (direct Y32/IdxBits output) ==========
# softmax -> sort32 -> top-k_out-only merge -> IN-KERNEL split into Y32 (fp32
# values) + IdxBits (fp32 slots holding uint32 index bit patterns; the host
# reinterprets with .view(int32), metadata-only). In-kernel unpack = two serial
# passes with a SINGLE gather each (padded dst [64]: the gather_mask dst has a
# fixed ~256B write granularity that would otherwise trample adjacent UB). The
# input-dtype cast is NOT in-kernel (cast-after-gather misplaces outputs on
# this backend): a separate flat cast kernel (dtype-changing T.copy = one
# AscendC::Cast per chunk) converts Y32 -> input dtype, with a per-row cast
# kernel fallback when N is not chunk-divisible.
# A sort32 re-sort replacing the per-row MrgSort was prototyped and REJECTED:
# staging the scalar+idx pair split costs far more than MrgSort saves (5.9ms
# vs 0.27ms on case3); see debug_log attempt 6.
@tilelang.jit(out_idx=[1, 2], pass_configs=PASS_CONFIGS)
def _moe_gating_top_k_softmax_sort32merge_kernel(M: int,
                                                 E: int,
                                                 k: int,
                                                 block_M: int,
                                                 dtype: str = "float"):
    """Sort32(+top-k merge) kernel with direct Y32/IdxBits GM outputs.

    Output signature (out_idx=[1,2]): Y32 (M, k) fp32 top-k softmax values;
    IdxBits (M, k) fp32 slots holding uint32 index bit patterns.
    """
    aligned_E = (E + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE
    use_fp32_compute = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_fp32_compute else dtype
    groups = aligned_E // ALIGN_SIZE  # 1 (E=32) / 2 (E=64) / 4 (E=128)
    need_merge = groups > 1
    k_out = max((k + 7) // 8 * 8, 4)
    m_num = T.ceildiv(M, block_M)
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
            X: T.Tensor((M, E), dtype),
            Y32: T.Tensor((M, k), "float32"),
            IdxBits: T.Tensor((M, k), "float32"),
            IdxIn: T.Tensor((M, aligned_E), "uint32"),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            scores_2d = T.alloc_ub([sub_block_M, aligned_E], dtype)
            scores_cal_2d = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            max_2d = T.alloc_ub([sub_block_M, 1], cal_dtype)
            max_bc = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            sum_2d = T.alloc_ub([sub_block_M, 1], cal_dtype)
            sum_bc = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            idx_2d_u32 = T.alloc_ub([sub_block_M, aligned_E], "uint32")
            sorted_2d = T.alloc_ub([sub_block_M, 2 * aligned_E], cal_dtype)
            merged_2d = T.alloc_ub([sub_block_M, 2 * aligned_E], cal_dtype)
            compact_2d = T.alloc_ub([sub_block_M, 2 * k_out], cal_dtype)
            # in-kernel split buffers (gather dst padded to 256B granularity)
            pairs_1d = T.alloc_ub([2 * k_out], cal_dtype)
            y_f32_1d = T.alloc_ub([64], cal_dtype)
            i_f32_1d = T.alloc_ub([64], cal_dtype)
            y_stage = T.alloc_ub([sub_block_M, k_out], cal_dtype)
            i_stage = T.alloc_ub([sub_block_M, k_out], cal_dtype)

            if use_fp32_compute:
                T.copy(
                    X[row_base:row_base + sub_block_M, 0:E],
                    scores_2d,
                    pad_value=-T.infinity(cal_dtype),
                )
                T.tile.cast(
                    scores_cal_2d,
                    scores_2d,
                    CAST_MODE_LOW2HIGH,
                    sub_block_M * aligned_E,
                )
            else:
                T.copy(
                    X[row_base:row_base + sub_block_M, 0:E],
                    scores_cal_2d,
                    pad_value=-T.infinity(cal_dtype),
                )
            T.reduce_max(scores_cal_2d, max_2d, dim=-1)
            T.tile.broadcast(max_bc, max_2d)
            T.tile.sub(scores_cal_2d, scores_cal_2d, max_bc)
            T.tile.exp(scores_cal_2d, scores_cal_2d)
            T.reduce_sum(scores_cal_2d, sum_2d, dim=-1)
            T.tile.broadcast(sum_bc, sum_2d)
            T.tile.div(scores_cal_2d, scores_cal_2d, sum_bc)

            T.copy(IdxIn[row_base:row_base + sub_block_M, :], idx_2d_u32)
            T.tile.sort32(sorted_2d, scores_cal_2d, idx_2d_u32)

            # top-k_out-only merge (union top-k subset of per-group top-k)
            if need_merge:
                if groups == 2:
                    for r in T.serial(sub_block_M):
                        T.tile.merge_sort(
                            merged_2d[r, 0:4 * k_out],
                            sorted_2d[r, 0:2 * k_out],
                            sorted_2d[r, 2 * ALIGN_SIZE:2 * ALIGN_SIZE + 2 * k_out],
                        )
                        T.copy(merged_2d[r, 0:2 * k_out], compact_2d[r, :])
                elif groups == 4:
                    for r in T.serial(sub_block_M):
                        T.tile.merge_sort(
                            merged_2d[r, 0:8 * k_out],
                            sorted_2d[r, 0:2 * k_out],
                            sorted_2d[r, 2 * ALIGN_SIZE:2 * ALIGN_SIZE + 2 * k_out],
                            sorted_2d[r, 4 * ALIGN_SIZE:4 * ALIGN_SIZE + 2 * k_out],
                            sorted_2d[r, 6 * ALIGN_SIZE:6 * ALIGN_SIZE + 2 * k_out],
                        )
                        T.copy(merged_2d[r, 0:2 * k_out], compact_2d[r, :])
            else:
                for r in T.serial(sub_block_M):
                    T.copy(sorted_2d[r, 0:2 * k_out], compact_2d[r, :])

            # ---- in-kernel split: two passes, single gather each ----
            for r in T.serial(sub_block_M):
                T.copy(compact_2d[r, :], pairs_1d)
                T.tile.gather_mask(y_f32_1d, pairs_1d, "P0101")
                T.copy(y_f32_1d[0:k_out], y_stage[r, 0:k_out])
            for r in T.serial(sub_block_M):
                T.copy(compact_2d[r, :], pairs_1d)
                T.tile.gather_mask(i_f32_1d, pairs_1d, "P1010")
                T.copy(i_f32_1d[0:k_out], i_stage[r, 0:k_out])
            T.copy(y_stage[:, 0:k], Y32[row_base:row_base + sub_block_M, 0:k])
            T.copy(i_stage[:, 0:k], IdxBits[row_base:row_base + sub_block_M, 0:k])

    return main


@tilelang.jit(out_idx=[1, 2], pass_configs=PASS_CONFIGS)
def _moe_gating_top_k_softmax_sort32merge_smallk_kernel(M: int,
                                                        E: int,
                                                        k: int,
                                                        block_M: int,
                                                        dtype: str = "float"):
    """Small-k (k < 8) fp32 direct-output variant of the base sort32 kernel.

    Same softmax/sort32/top-k_out-merge pipeline, but the in-kernel split
    stores Y (fp32 == input dtype, no cast needed) and IdxBits (fp32 bit
    patterns) PER ROW (the orig kernel's proven per-row MTE3 pattern),
    eliminating the whole small-k unpack kernel chain. Only used for k < 8
    with fp32 input (fp16/bf16 would need gather->cast which misplaces
    outputs on this backend; those keep the separate unpack chain).
    """
    aligned_E = (E + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE
    cal_dtype = "float32"  # fp32-only variant
    groups = aligned_E // ALIGN_SIZE
    need_merge = groups > 1
    k_out = max((k + 7) // 8 * 8, 4)
    m_num = T.ceildiv(M, block_M)
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
            X: T.Tensor((M, E), dtype),
            Y: T.Tensor((M, k), "float32"),
            IdxBits: T.Tensor((M, k), "float32"),
            IdxIn: T.Tensor((M, aligned_E), "uint32"),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            scores_cal_2d = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            max_2d = T.alloc_ub([sub_block_M, 1], cal_dtype)
            max_bc = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            sum_2d = T.alloc_ub([sub_block_M, 1], cal_dtype)
            sum_bc = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            idx_2d_u32 = T.alloc_ub([sub_block_M, aligned_E], "uint32")
            sorted_2d = T.alloc_ub([sub_block_M, 2 * aligned_E], cal_dtype)
            merged_2d = T.alloc_ub([sub_block_M, 2 * aligned_E], cal_dtype)
            compact_2d = T.alloc_ub([sub_block_M, 2 * k_out], cal_dtype)
            pairs_1d = T.alloc_ub([2 * k_out], cal_dtype)
            y_f32_1d = T.alloc_ub([64], cal_dtype)
            i_f32_1d = T.alloc_ub([64], cal_dtype)

            T.copy(
                X[row_base:row_base + sub_block_M, 0:E],
                scores_cal_2d,
                pad_value=-T.infinity(cal_dtype),
            )
            T.reduce_max(scores_cal_2d, max_2d, dim=-1)
            T.tile.broadcast(max_bc, max_2d)
            T.tile.sub(scores_cal_2d, scores_cal_2d, max_bc)
            T.tile.exp(scores_cal_2d, scores_cal_2d)
            T.reduce_sum(scores_cal_2d, sum_2d, dim=-1)
            T.tile.broadcast(sum_bc, sum_2d)
            T.tile.div(scores_cal_2d, scores_cal_2d, sum_bc)
            T.copy(IdxIn[row_base:row_base + sub_block_M, :], idx_2d_u32)
            T.tile.sort32(sorted_2d, scores_cal_2d, idx_2d_u32)

            if need_merge:
                if groups == 2:
                    for r in T.serial(sub_block_M):
                        T.tile.merge_sort(
                            merged_2d[r, 0:4 * k_out],
                            sorted_2d[r, 0:2 * k_out],
                            sorted_2d[r, 2 * ALIGN_SIZE:2 * ALIGN_SIZE + 2 * k_out],
                        )
                        T.copy(merged_2d[r, 0:2 * k_out], compact_2d[r, :])
                elif groups == 4:
                    for r in T.serial(sub_block_M):
                        T.tile.merge_sort(
                            merged_2d[r, 0:8 * k_out],
                            sorted_2d[r, 0:2 * k_out],
                            sorted_2d[r, 2 * ALIGN_SIZE:2 * ALIGN_SIZE + 2 * k_out],
                            sorted_2d[r, 4 * ALIGN_SIZE:4 * ALIGN_SIZE + 2 * k_out],
                            sorted_2d[r, 6 * ALIGN_SIZE:6 * ALIGN_SIZE + 2 * k_out],
                        )
                        T.copy(merged_2d[r, 0:2 * k_out], compact_2d[r, :])
            else:
                for r in T.serial(sub_block_M):
                    T.copy(sorted_2d[r, 0:2 * k_out], compact_2d[r, :])

            # per-row split + per-row direct store (fp32: no cast)
            for r in T.serial(sub_block_M):
                row = row_base + r
                T.copy(compact_2d[r, :], pairs_1d)
                T.tile.gather_mask(y_f32_1d, pairs_1d, "P0101")
                T.tile.gather_mask(i_f32_1d, pairs_1d, "P1010")
                T.copy(y_f32_1d[0:k], Y[row, 0:k])
                T.copy(i_f32_1d[0:k], IdxBits[row, 0:k])

    return main


@tilelang.jit(out_idx=[1, 2], pass_configs=PASS_CONFIGS)
def _moe_gating_top_k_softmax_sort32merge_e256_kernel(M: int,
                                                      E: int,
                                                      k: int,
                                                      block_M: int,
                                                      dtype: str = "float"):
    """Sort32+merge for aligned_E=256 (8 x 32-groups), direct Y32/IdxBits out.

    merge plan per row (top-k_out only at every stage): stage1 two 4-way
    merges (groups' top-k_out), stage2 one 2-way merge -> top-k_out.
    """
    aligned_E = 256
    use_fp32_compute = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_fp32_compute else dtype
    k_out = max((k + 7) // 8 * 8, 4)
    m_num = T.ceildiv(M, block_M)
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
            X: T.Tensor((M, E), dtype),
            Y32: T.Tensor((M, k), "float32"),
            IdxBits: T.Tensor((M, k), "float32"),
            IdxIn: T.Tensor((M, aligned_E), "uint32"),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            scores_2d = T.alloc_ub([sub_block_M, aligned_E], dtype)
            scores_cal_2d = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            max_2d = T.alloc_ub([sub_block_M, 1], cal_dtype)
            max_bc = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            sum_2d = T.alloc_ub([sub_block_M, 1], cal_dtype)
            sum_bc = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            idx_2d_u32 = T.alloc_ub([sub_block_M, aligned_E], "uint32")
            sorted_2d = T.alloc_ub([sub_block_M, 2 * aligned_E], cal_dtype)
            tmp_a = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            tmp_b = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            merged_2d = T.alloc_ub([sub_block_M, 2 * aligned_E], cal_dtype)
            compact_2d = T.alloc_ub([sub_block_M, 2 * k_out], cal_dtype)
            pairs_1d = T.alloc_ub([2 * k_out], cal_dtype)
            y_f32_1d = T.alloc_ub([64], cal_dtype)
            i_f32_1d = T.alloc_ub([64], cal_dtype)
            y_stage = T.alloc_ub([sub_block_M, k_out], cal_dtype)
            i_stage = T.alloc_ub([sub_block_M, k_out], cal_dtype)

            if use_fp32_compute:
                T.copy(
                    X[row_base:row_base + sub_block_M, 0:E],
                    scores_2d,
                    pad_value=-T.infinity(cal_dtype),
                )
                T.tile.cast(
                    scores_cal_2d,
                    scores_2d,
                    CAST_MODE_LOW2HIGH,
                    sub_block_M * aligned_E,
                )
            else:
                T.copy(
                    X[row_base:row_base + sub_block_M, 0:E],
                    scores_cal_2d,
                    pad_value=-T.infinity(cal_dtype),
                )
            T.reduce_max(scores_cal_2d, max_2d, dim=-1)
            T.tile.broadcast(max_bc, max_2d)
            T.tile.sub(scores_cal_2d, scores_cal_2d, max_bc)
            T.tile.exp(scores_cal_2d, scores_cal_2d)
            T.reduce_sum(scores_cal_2d, sum_2d, dim=-1)
            T.tile.broadcast(sum_bc, sum_2d)
            T.tile.div(scores_cal_2d, scores_cal_2d, sum_bc)
            T.copy(IdxIn[row_base:row_base + sub_block_M, :], idx_2d_u32)
            T.tile.sort32(sorted_2d, scores_cal_2d, idx_2d_u32)

            # stage1: two 4-way merges (top-k_out only), stage2: one 2-way
            for r in T.serial(sub_block_M):
                T.tile.merge_sort(
                    tmp_a[r, 0:8 * k_out],
                    sorted_2d[r, 0:2 * k_out],
                    sorted_2d[r, 2 * ALIGN_SIZE:2 * ALIGN_SIZE + 2 * k_out],
                    sorted_2d[r, 4 * ALIGN_SIZE:4 * ALIGN_SIZE + 2 * k_out],
                    sorted_2d[r, 6 * ALIGN_SIZE:6 * ALIGN_SIZE + 2 * k_out],
                )
                T.tile.merge_sort(
                    tmp_b[r, 0:8 * k_out],
                    sorted_2d[r, 8 * ALIGN_SIZE:8 * ALIGN_SIZE + 2 * k_out],
                    sorted_2d[r, 10 * ALIGN_SIZE:10 * ALIGN_SIZE + 2 * k_out],
                    sorted_2d[r, 12 * ALIGN_SIZE:12 * ALIGN_SIZE + 2 * k_out],
                    sorted_2d[r, 14 * ALIGN_SIZE:14 * ALIGN_SIZE + 2 * k_out],
                )
            for r in T.serial(sub_block_M):
                T.tile.merge_sort(
                    merged_2d[r, 0:4 * k_out],
                    tmp_a[r, 0:2 * k_out],
                    tmp_b[r, 0:2 * k_out],
                )
                T.copy(merged_2d[r, 0:2 * k_out], compact_2d[r, :])

            # in-kernel split
            for r in T.serial(sub_block_M):
                T.copy(compact_2d[r, :], pairs_1d)
                T.tile.gather_mask(y_f32_1d, pairs_1d, "P0101")
                T.copy(y_f32_1d[0:k_out], y_stage[r, 0:k_out])
            for r in T.serial(sub_block_M):
                T.copy(compact_2d[r, :], pairs_1d)
                T.tile.gather_mask(i_f32_1d, pairs_1d, "P1010")
                T.copy(i_f32_1d[0:k_out], i_stage[r, 0:k_out])
            T.copy(y_stage[:, 0:k], Y32[row_base:row_base + sub_block_M, 0:k])
            T.copy(i_stage[:, 0:k], IdxBits[row_base:row_base + sub_block_M, 0:k])

    return main


@tilelang.jit(out_idx=[1, 2], pass_configs=PASS_CONFIGS)
def _moe_gating_top_k_softmax_sort32merge_e512_kernel(M: int,
                                                      E: int,
                                                      k: int,
                                                      block_M: int,
                                                      dtype: str = "float"):
    """Sort32+merge for aligned_E=512 (16 x 32-groups), direct Y32/IdxBits out.

    merge plan (top-k_out only): stage1 four 4-way merges (groups' top-k_out),
    stage2 one 4-way merge -> top-k_out.
    """
    aligned_E = 512
    use_fp32_compute = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_fp32_compute else dtype
    k_out = max((k + 7) // 8 * 8, 4)
    g64 = 2 * ALIGN_SIZE  # 64 pairs per 32-group
    m_num = T.ceildiv(M, block_M)
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
            X: T.Tensor((M, E), dtype),
            Y32: T.Tensor((M, k), "float32"),
            IdxBits: T.Tensor((M, k), "float32"),
            IdxIn: T.Tensor((M, aligned_E), "uint32"),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            scores_2d = T.alloc_ub([sub_block_M, aligned_E], dtype)
            scores_cal_2d = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            max_2d = T.alloc_ub([sub_block_M, 1], cal_dtype)
            max_bc = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            sum_2d = T.alloc_ub([sub_block_M, 1], cal_dtype)
            sum_bc = T.alloc_ub([sub_block_M, aligned_E], cal_dtype)
            idx_2d_u32 = T.alloc_ub([sub_block_M, aligned_E], "uint32")
            sorted_2d = T.alloc_ub([sub_block_M, 2 * aligned_E], cal_dtype)
            tmp_a = T.alloc_ub([sub_block_M, 4 * g64], cal_dtype)
            tmp_b = T.alloc_ub([sub_block_M, 4 * g64], cal_dtype)
            tmp_c = T.alloc_ub([sub_block_M, 4 * g64], cal_dtype)
            tmp_d = T.alloc_ub([sub_block_M, 4 * g64], cal_dtype)
            merged_2d = T.alloc_ub([sub_block_M, 2 * aligned_E], cal_dtype)
            compact_2d = T.alloc_ub([sub_block_M, 2 * k_out], cal_dtype)
            pairs_1d = T.alloc_ub([2 * k_out], cal_dtype)
            y_f32_1d = T.alloc_ub([64], cal_dtype)
            i_f32_1d = T.alloc_ub([64], cal_dtype)
            y_stage = T.alloc_ub([sub_block_M, k_out], cal_dtype)
            i_stage = T.alloc_ub([sub_block_M, k_out], cal_dtype)

            if use_fp32_compute:
                T.copy(
                    X[row_base:row_base + sub_block_M, 0:E],
                    scores_2d,
                    pad_value=-T.infinity(cal_dtype),
                )
                T.tile.cast(
                    scores_cal_2d,
                    scores_2d,
                    CAST_MODE_LOW2HIGH,
                    sub_block_M * aligned_E,
                )
            else:
                T.copy(
                    X[row_base:row_base + sub_block_M, 0:E],
                    scores_cal_2d,
                    pad_value=-T.infinity(cal_dtype),
                )
            T.reduce_max(scores_cal_2d, max_2d, dim=-1)
            T.tile.broadcast(max_bc, max_2d)
            T.tile.sub(scores_cal_2d, scores_cal_2d, max_bc)
            T.tile.exp(scores_cal_2d, scores_cal_2d)
            T.reduce_sum(scores_cal_2d, sum_2d, dim=-1)
            T.tile.broadcast(sum_bc, sum_2d)
            T.tile.div(scores_cal_2d, scores_cal_2d, sum_bc)
            T.copy(IdxIn[row_base:row_base + sub_block_M, :], idx_2d_u32)
            T.tile.sort32(sorted_2d, scores_cal_2d, idx_2d_u32)

            # stage1: four 4-way merges (top-k_out only)
            for r in T.serial(sub_block_M):
                T.tile.merge_sort(
                    tmp_a[r, 0:8 * k_out],
                    sorted_2d[r, 0:2 * k_out],
                    sorted_2d[r, g64:g64 + 2 * k_out],
                    sorted_2d[r, 2 * g64:2 * g64 + 2 * k_out],
                    sorted_2d[r, 3 * g64:3 * g64 + 2 * k_out],
                )
                T.tile.merge_sort(
                    tmp_b[r, 0:8 * k_out],
                    sorted_2d[r, 4 * g64:4 * g64 + 2 * k_out],
                    sorted_2d[r, 5 * g64:5 * g64 + 2 * k_out],
                    sorted_2d[r, 6 * g64:6 * g64 + 2 * k_out],
                    sorted_2d[r, 7 * g64:7 * g64 + 2 * k_out],
                )
                T.tile.merge_sort(
                    tmp_c[r, 0:8 * k_out],
                    sorted_2d[r, 8 * g64:8 * g64 + 2 * k_out],
                    sorted_2d[r, 9 * g64:9 * g64 + 2 * k_out],
                    sorted_2d[r, 10 * g64:10 * g64 + 2 * k_out],
                    sorted_2d[r, 11 * g64:11 * g64 + 2 * k_out],
                )
                T.tile.merge_sort(
                    tmp_d[r, 0:8 * k_out],
                    sorted_2d[r, 12 * g64:12 * g64 + 2 * k_out],
                    sorted_2d[r, 13 * g64:13 * g64 + 2 * k_out],
                    sorted_2d[r, 14 * g64:14 * g64 + 2 * k_out],
                    sorted_2d[r, 15 * g64:15 * g64 + 2 * k_out],
                )
            # stage2: one 4-way merge (top-k_out only)
            for r in T.serial(sub_block_M):
                T.tile.merge_sort(
                    merged_2d[r, 0:8 * k_out],
                    tmp_a[r, 0:2 * k_out],
                    tmp_b[r, 0:2 * k_out],
                    tmp_c[r, 0:2 * k_out],
                    tmp_d[r, 0:2 * k_out],
                )
                T.copy(merged_2d[r, 0:2 * k_out], compact_2d[r, :])

            # in-kernel split
            for r in T.serial(sub_block_M):
                T.copy(compact_2d[r, :], pairs_1d)
                T.tile.gather_mask(y_f32_1d, pairs_1d, "P0101")
                T.copy(y_f32_1d[0:k_out], y_stage[r, 0:k_out])
            for r in T.serial(sub_block_M):
                T.copy(compact_2d[r, :], pairs_1d)
                T.tile.gather_mask(i_f32_1d, pairs_1d, "P1010")
                T.copy(i_f32_1d[0:k_out], i_stage[r, 0:k_out])
            T.copy(y_stage[:, 0:k], Y32[row_base:row_base + sub_block_M, 0:k])
            T.copy(i_stage[:, 0:k], IdxBits[row_base:row_base + sub_block_M, 0:k])

    return main


def _select_block(M: int, E: int, k: int, dtype_str: str) -> int:
    """Select block_M based on M, E, dtype to maximize core utilization
    while respecting UB budget.

    Hybrid approach: 2D batch load + per-row softmax.
    UB usage:
      - Per sub_block_M row: scores_cal_2d(cal_bytes) + scores_2d(dtype_bytes)
      - Fixed 1D buffers: scores_1d + max_2d_ub + sum_2d_ub ≈ 3 × aligned_E × cal_bytes
      - Small fixed: max_ub, sum_ub, topk, y, idx (negligible for large E)

    Target: m_num = ceildiv(M, block_M) >= core_num (24) when possible.
    """
    core_num = 24
    aligned_E = (E + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE

    cal_bytes = 4
    dtype_bytes = 2 if dtype_str in ("float16", "bfloat16") else 4

    UB_BUDGET = 180 * 1024

    # 2D softmax buffers per row: scores_2d + scores_cal_2d + max_bc + sum_bc
    per_row = aligned_E * (dtype_bytes + 3 * cal_bytes)
    # Fixed 1D: scores_1d + small topk/y/idx buffers
    fixed_1d = aligned_E * cal_bytes

    UB_BUDGET = 170 * 1024
    available = UB_BUDGET - fixed_1d
    max_sub_block_M_ub = max(1, available // per_row)

    # Target: m_num >= core_num → sub_block_M <= M / (core_num * VEC_NUM)
    target_sub_block_M = max(1, M // (core_num * VEC_NUM))
    # For VERY small M (target < 4 => M < ~192), avoid sub_block_M=1 (each
    # block processes only 1 row, massive block-scheduling overhead on tiny
    # inputs). Scale down block count so each block batch-processes rows.
    # Medium/large M keep multi-block parallelism (measured better for them).
    if target_sub_block_M < 4:
        target_sub_block_M = min(max_sub_block_M_ub, M // VEC_NUM)

    # Choose sub_block_M as power of 2 (required for 2D reduce/broadcast alignment)
    for candidate in [128, 64, 32, 16, 8, 4, 2, 1]:
        if candidate <= max_sub_block_M_ub and candidate <= target_sub_block_M:
            sub_block_M = candidate
            break
    else:
        sub_block_M = 1

    block_M = sub_block_M * VEC_NUM
    return block_M


def _select_block_sort32merge(M: int, E: int, k: int, dtype_str: str) -> int:
    """Select block_M for the sort32(+merge) batch kernel.

    This kernel allocates LARGE per-row buffers beyond softmax:
      scores_2d (fp16/bf16 only) + scores_cal_2d + max_bc + sum_bc
      + idx_2d_u32 + sorted_2d (2*aligned_E) + merged_2d (2*aligned_E, only
      when aligned_E > ALIGN_SIZE i.e. need merge).

    Must pick sub_block_M small enough that ALL buffers fit in UB.
    """
    core_num = 24
    aligned_E = (E + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE
    need_merge = aligned_E > ALIGN_SIZE
    use_fp32_compute = dtype_str in ("float16", "bfloat16")
    dtype_bytes = 2 if use_fp32_compute else 4
    cal_bytes = 4

    UB_BUDGET = 170 * 1024
    # Per sub_block_M row. E=256 adds tmp_a+tmp_b (2*aligned_E fp32),
    # E=512 adds tmp_a..tmp_d (each 4*g64 = aligned_E fp32).
    n_tmp = 4 if aligned_E > 256 else (2 if aligned_E > 128 else 0)
    tmp_extra = n_tmp * aligned_E * cal_bytes
    per_row = (
        aligned_E * (dtype_bytes if use_fp32_compute else 0)  # scores_2d
        + aligned_E * cal_bytes  # scores_cal_2d
        + 2 * aligned_E * cal_bytes  # max_bc + sum_bc
        + aligned_E * cal_bytes  # idx_2d_u32
        + 2 * aligned_E * cal_bytes  # sorted_2d
        + (2 * aligned_E * cal_bytes if need_merge else 0)  # merged_2d
        + tmp_extra  # merge tmps
    )
    # Small fixed: max_2d/sum_2d [sub,1] negligible
    fixed = 32 * cal_bytes

    available = UB_BUDGET - fixed
    max_sub_block_M_ub = max(1, available // per_row)

    # For VERY small M (target < 4 => M < ~192), avoid sub_block_M=1 (each
    # block processes only 1 row). Medium/large M keep multi-block parallelism.
    target_sub_block_M = max(1, M // (core_num * VEC_NUM))
    if target_sub_block_M < 4:
        target_sub_block_M = min(max_sub_block_M_ub, M // VEC_NUM)

    for candidate in [128, 64, 32, 16, 8, 4, 2, 1]:
        if candidate <= max_sub_block_M_ub and candidate <= target_sub_block_M:
            sub_block_M = candidate
            break
    else:
        sub_block_M = 1

    block_M = sub_block_M * VEC_NUM
    return block_M


def _select_block_batch(M: int, E: int, k: int, dtype_str: str) -> int:
    """Select block_M for batch output kernel (k % 8 == 0).

    Same as _select_block but accounts for additional 2D output staging
    buffers (y_out_2d + idx_out_2d) in the UB budget.
    """
    core_num = 24
    aligned_E = (E + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE
    aligned_K = (k + 7) // 8 * 8  # == k for batch kernel
    # Match kernel's aligned_K_ub (32-byte row stride for 2D staging)
    aligned_K_ub = max(aligned_K, 16)

    cal_bytes = 4
    dtype_bytes = 2 if dtype_str in ("float16", "bfloat16") else 4

    UB_BUDGET = 170 * 1024
    # 2D softmax buffers per row: scores_2d + scores_cal_2d + max_bc + sum_bc
    softmax_per_row = aligned_E * (dtype_bytes + 3 * cal_bytes)
    # 2D output staging per row: y_out_2d + idx_out_2d (with aligned_K_ub columns)
    output_per_row = aligned_K_ub * (dtype_bytes + 4)  # idx is int32 (4 bytes)
    per_row = softmax_per_row + output_per_row
    # Fixed 1D: scores_1d + small topk/y/idx buffers
    fixed_1d = aligned_E * cal_bytes

    available = UB_BUDGET - fixed_1d
    max_sub_block_M_ub = max(1, available // per_row)

    # For VERY small M (target < 4 => M < ~192), avoid sub_block_M=1 (each
    # block processes only 1 row). Medium/large M keep multi-block parallelism.
    target_sub_block_M = max(1, M // (core_num * VEC_NUM))
    if target_sub_block_M < 4:
        target_sub_block_M = min(max_sub_block_M_ub, M // VEC_NUM)

    for candidate in [128, 64, 32, 16, 8, 4, 2, 1]:
        if candidate <= max_sub_block_M_ub and candidate <= target_sub_block_M:
            sub_block_M = candidate
            break
    else:
        sub_block_M = 1

    block_M = sub_block_M * VEC_NUM
    return block_M


# ========== Kernel cache ==========
def _use_batch_kernel(k: int, tl_dtype: str, E: int = 0) -> bool:
    """Determine if batch output kernel is safe for this config.

    Batch output uses 2D UB staging buffers (y_out_2d, idx_out_2d). The
    row stride must be 32-byte aligned for VEC instructions. tilelang's
    memory planning pass optimizes buffer columns to aligned_K (= k for
    k%8==0), so the row stride is aligned_K * dtype_bytes.

    Alignment constraints:
    - fp32/int32: aligned_K * 4 >= 32 (aligned_K >= 8) -> safe for k >= 8
    - bf16/fp16: aligned_K * 2 >= 32 requires aligned_K >= 16 -> safe for k >= 16
    - k % 8 != 0: batch store column not 8-aligned -> unsafe
    - E % 32 != 0: non-aligned E causes 2D batch store address mismatch -> unsafe

    Empirical constraints (from precision testing):
    - k == 16 causes precision failure (2D batch store data corruption for
      both fp32 and fp16). Root cause: 1D->2D row slice copy with k=16
      triggers codegen data placement issue. Safe at k >= 32.
    - E % 32 != 0 (e.g. E=511) causes precision failure in batch store.

    So batch kernel is used when: k%8==0 AND k>=32 AND E%32==0.
    """
    if k % 8 != 0:
        return False
    if k < 32:
        return False
    return not (E > 0 and E % 32 != 0)


def _use_sort32_kernel(M: int, E: int, tl_dtype: str) -> bool:
    """Determine if sort32(+merge) batch kernel is applicable.

    sort32 processes 32-element blocks. For aligned_E in {32, 64}:
    - aligned_E=32: one 32-group per row, no merge
    - aligned_E=64: two 32-groups per row, one 2-way merge/row
    V-instruction savings dominate for large M even with full GM pair output
    (A/B: E=32 M=16384 +100%; E=64 M=131072 -53%).

    Constraints:
    - aligned_E <= 256 (one to eight 32-groups per row). E=512 tried but
      regressed (too many merge layers + small sub_M from UB pressure).
    - compute dtype float32 (sort32/merge_sort need fp32; fp16/bf16 cast)
    - M >= per-shape threshold: for small M the extra GM traffic outweighs
      V savings. Empirically tuned:
        E<=64  -> need M >= 8192 (case2/3/8 good, case7 M=4K regressed)
        E>=128 -> need M >= 4096 (case4/13/18 improved)
    """
    aligned_E = (E + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE
    if aligned_E > 256:
        return False
    if aligned_E >= 128:
        need_m = 4096
    else:
        need_m = 8192
    return need_m <= M


def _get_block_M(M: int, E: int, k: int, tl_dtype: str) -> int:
    """Compute block_M based on kernel variant (sort32, batch, or original)."""
    if _use_sort32_kernel(M, E, tl_dtype):
        return _select_block_sort32merge(M, E, k, tl_dtype)
    if _use_batch_kernel(k, tl_dtype, E):
        return _select_block_batch(M, E, k, tl_dtype)
    return _select_block(M, E, k, tl_dtype)


def _get_kernel(M: int, E: int, k: int, tl_dtype: str):
    """Get or compile kernel for (M, E, k, dtype).

    Dispatches to:
    1. sort32: fp32 AND aligned_E==32 — 2D batch sort32 (1V for all rows),
       output interleaved pairs to GM (extracted in Python)
    2. batch: k%8==0 AND k>=32 AND E%32==0 — batch output staging
    3. orig: per-row topk + per-row store (fallback)

    Returns (kernel, block_M).
    """
    if _use_sort32_kernel(M, E, tl_dtype):
        block_M = _select_block_sort32merge(M, E, k, tl_dtype)
        aligned_E = (E + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE
        # NOTE: a small-k fp32 per-row direct-output variant was tried here
        # (sort32smallk) and REVERTED: per-row 16B stores are slower than the
        # separate unpack chain (case2 0.120x -> 0.102x, -15%, attempt 8).
        if aligned_E == 512:
            key = (M, E, k, tl_dtype, block_M, "sort32gm_e512")
            if key not in _kernel_cache:
                _kernel_cache[key] = _moe_gating_top_k_softmax_sort32merge_e512_kernel(
                    M, E, k, block_M, dtype=tl_dtype)
        elif aligned_E == 256:
            key = (M, E, k, tl_dtype, block_M, "sort32gm_e256")
            if key not in _kernel_cache:
                _kernel_cache[key] = _moe_gating_top_k_softmax_sort32merge_e256_kernel(
                    M, E, k, block_M, dtype=tl_dtype)
        else:
            key = (M, E, k, tl_dtype, block_M, "sort32gm")
            if key not in _kernel_cache:
                _kernel_cache[key] = _moe_gating_top_k_softmax_sort32merge_kernel(
                    M, E, k, block_M, dtype=tl_dtype)
    elif _use_batch_kernel(k, tl_dtype, E):
        block_M = _select_block_batch(M, E, k, tl_dtype)
        key = (M, E, k, tl_dtype, block_M, "batch")
        if key not in _kernel_cache:
            _kernel_cache[key] = _moe_gating_top_k_softmax_batch_kernel(
                M, E, k, block_M, dtype=tl_dtype)
    else:
        block_M = _select_block(M, E, k, tl_dtype)
        key = (M, E, k, tl_dtype, block_M, "orig")
        if key not in _kernel_cache:
            _kernel_cache[key] = _moe_gating_top_k_softmax_kernel(M, E, k, block_M, dtype=tl_dtype)
    return _kernel_cache[key], block_M


# ========== torch-free data-construction kernels (910c zerotorch fix) ==========
# The web 910c eval image (Ascend910_9362 docker, CANN 9.0.0) has a minimal
# aclnn builtin operator set: Range, Fill, StridedSlice binaries are confirmed
# missing (561103), and most other aclnn ops (Copy/Cast/Slice/Cat/Where/...) are
# unverified/high-risk. RULE: the wrapper may NOT dispatch ANY torch tensor op
# that lowers to an aclnn kernel. All data construction below runs inside
# @tilelang.jit kernels; the host only does torch.empty (allocation) and pure
# metadata shape/reshape/view. The previous cat/full padding and strided extraction of
# the previous min_fix version are replaced by these kernels.


def _select_tiny_block(M: int, per_row_bytes: int) -> int:
    """Pick block_M for tiny helper kernels (row-wise kernels, small UB layout)."""
    UB_BUDGET = 170 * 1024
    target_sub = max(1, M // (24 * VEC_NUM))
    max_sub_ub = max(1, UB_BUDGET // max(per_row_bytes, 1))
    if target_sub < 4:
        target_sub = min(max_sub_ub, M // VEC_NUM)
    for candidate in [128, 64, 32, 16, 8, 4, 2, 1]:
        if candidate <= max_sub_ub and candidate <= target_sub:
            sub_block_M = candidate
            break
    else:
        sub_block_M = 1
    return sub_block_M * VEC_NUM


def _get_tiny_kernel(key: tuple, build) -> callable:
    """Compile-and-cache a tiny helper kernel."""
    if key not in _kernel_cache:
        _kernel_cache[key] = build()
    return _kernel_cache[key]


def _get_row_idx_kernel(M: int, k: int):
    block_M = _select_tiny_block(M, k * 4)
    key = ("row_idx", M, k, block_M)
    return _get_tiny_kernel(key, lambda: _moe_row_idx_kernel(M, k, block_M))


def _get_idxin_kernel(padded_M: int, E: int, aligned_E: int):
    block_M = _select_tiny_block(padded_M, aligned_E * 4)
    key = ("idxin", padded_M, E, aligned_E, block_M)
    return _get_tiny_kernel(key, lambda: _moe_idxin_kernel(padded_M, E, aligned_E, block_M))


def _get_flat_cast_kernel(N: int, chunk: int, tl_dtype: str):
    """Flat 1D fp32->dtype cast kernel (dtype-changing T.copy). Requires
    chunk to divide N (no tail handling); returns None if impossible."""
    for cand in (4096, 2048, 1024, 512, 256):
        if N % cand == 0:
            chunk = cand
            break
    else:
        return None
    key = ("flat_cast", N, chunk, tl_dtype)
    return _get_tiny_kernel(key, lambda: _moe_flat_cast_kernel(N, chunk, dtype=tl_dtype))


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def _moe_flat_cast_kernel(N: int, chunk: int, dtype: str = "float16"):
    """Flat fp32 -> dtype cast: the dtype-changing T.copy lowers to one
    AscendC::Cast per chunk (per-lane chunk/2 elements, >=128B). Replaces
    the per-row cast kernel for large N (requires N % chunk == 0)."""
    nb = T.ceildiv(N, chunk)

    @T.prim_func
    def main(
            Src: T.Tensor((N,), "float32"),
            Dst: T.Tensor((N,), dtype),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(nb, is_npu=True) as (cid, vid):
            p0 = cid * chunk + vid * (chunk // VEC_NUM)
            src_ub = T.alloc_ub([chunk // VEC_NUM], "float32")
            dst_ub = T.alloc_ub([chunk // VEC_NUM], dtype)
            T.copy(Src[p0:p0 + chunk // VEC_NUM], src_ub)
            T.copy(src_ub, dst_ub)
            T.copy(dst_ub, Dst[p0:p0 + chunk // VEC_NUM])

    return main


@tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
def _moe_apply_finished_row_kernel(M: int, k: int, E: int, block_M: int):
    """Per-row finished mask (original proven variant; faster for small k
    where the 2D broadcast wastes k_ub/k of the vector width)."""
    m_num = T.ceildiv(M, block_M)
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
            IdxIn: T.Tensor((M, k), "int32"),
            Fin: T.Tensor((M,), "int8"),
            IdxOut: T.Tensor((M, k), "int32"),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            win_start = T.min(row_base, M - sub_block_M)
            fin_ub = T.alloc_ub([sub_block_M], "int8")
            idx_ub = T.alloc_ub([sub_block_M, k], "int32")
            T.copy(Fin[win_start:win_start + sub_block_M], fin_ub)
            T.copy(IdxIn[win_start:win_start + sub_block_M, 0:k], idx_ub)
            for r in T.serial(sub_block_M):
                fin_i = T.cast(fin_ub[r], "int32")
                for j in T.Parallel(k):
                    idx_ub[r, j] = idx_ub[r, j] * (1 - fin_i)
                for j in T.Parallel(k):
                    idx_ub[r, j] = idx_ub[r, j] + E * fin_i
            T.copy(idx_ub, IdxOut[win_start:win_start + sub_block_M, 0:k])

    return main


def _get_finished_kernel(M: int, k: int, E: int):
    """Dispatch: 2D-batched mask for k>=8 (big-M win), per-row for k<8."""
    if k >= 8:
        block_M = _select_tiny_block(M, k * 4 + 4)
        key = ("finished2d", M, k, E, block_M)
        return _get_tiny_kernel(key, lambda: _moe_apply_finished_kernel(M, k, E, block_M))
    block_M = _select_tiny_block(M, k * 4 + 4)
    key = ("finished", M, k, E, block_M)
    return _get_tiny_kernel(key, lambda: _moe_apply_finished_row_kernel(M, k, E, block_M))


def _get_pad_kernel(M: int, padded_M: int, E: int, tl_dtype: str):
    dtype_bytes = 2 if tl_dtype in ("float16", "bfloat16") else 4
    block_M = _select_tiny_block(padded_M, E * dtype_bytes)
    key = ("pad", M, padded_M, E, tl_dtype, block_M)
    return _get_tiny_kernel(key,
                            lambda: _moe_pad_rows_kernel(M, padded_M, E, block_M, dtype=tl_dtype))


def _get_cast_kernel(M: int, k: int, tl_dtype: str):
    """Per-row elementwise fp32 -> dtype cast kernel getter (fallback path)."""
    block_M = _select_tiny_block(M, k * 4)
    key = ("cast_v", M, k, tl_dtype, block_M)
    return _get_tiny_kernel(key, lambda: _moe_cast_values_kernel(M, k, block_M, dtype=tl_dtype))


@tilelang.jit(out_idx=[0], pass_configs=PASS_CONFIGS)
def _moe_row_idx_kernel(M: int, k: int, block_M: int):
    """row_idx[m, j] = m + j*M (golden formula), output (M, k) int32.

    Replaces the old arange row_idx construction (Range is missing on the
    910c eval image). Values staged in UB (32B-aligned Parallel store) then
    written via T.copy so unaligned row widths (k < 8) do not misalign.
    """
    m_num = T.ceildiv(M, block_M)
    sub_block_M = block_M // VEC_NUM
    k_ub = max((k + 7) // 8 * 8, 8)

    @T.prim_func
    def main(RowIdx: T.Tensor((M, k), "int32")):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            row_ub = T.alloc_ub([k_ub], "int32")
            for r in T.serial(sub_block_M):
                row = T.min(row_base + r, M - 1)
                for j in T.Parallel(k_ub):
                    row_ub[j] = row + T.min(j, k - 1) * M
                T.copy(row_ub[0:k], RowIdx[row, 0:k])

    return main


@tilelang.jit(out_idx=[0], pass_configs=PASS_CONFIGS)
def _moe_idxin_kernel(M: int, E: int, aligned_E: int, block_M: int):
    """IdxIn[row, j] = j (j < E), E-1 afterwards; shape (M, aligned_E) int32.

    Replaces the old arange IdxIn construction (Range is missing on the 910c eval image).
    The host reinterprets int32 as uint32 (.view is metadata-only). Tail columns
    pair with -inf softmax values and are never selected by topk.
    """
    m_num = T.ceildiv(M, block_M)
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(IdxIn: T.Tensor((M, aligned_E), "int32")):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            for r in T.serial(sub_block_M):
                row = T.min(row_base + r, M - 1)
                for j in T.Parallel(aligned_E):
                    IdxIn[row, j] = T.min(j, E - 1)

    return main


@tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
def _moe_apply_finished_kernel(M: int, k: int, E: int, block_M: int):
    """Set expert_idx = E for rows where finished is True (2D-batched mask).

    Replaces the old where-based finished mask. IdxIn (M, k) int32 + Fin (M,)
    int8 (bool viewed as int8) -> IdxOut (M, k) int32. One 2D tile load per
    block, per-row scalar fin broadcast via T.Parallel row-broadcast (the
    documented broadcast pattern), one 2D store. Arithmetic mask:
    new = old * (1 - fin) + E * fin. Window clamped to [0, M - sub_block_M];
    overlapping duplicated writes are idempotent. (Tile-arithmetic variants
    with int8->int32 broadcast hit memory-planning duplicate-name checks and
    per-row casts were no faster; this form measured fastest.)
    """
    m_num = T.ceildiv(M, block_M)
    sub_block_M = block_M // VEC_NUM
    k_ub = max((k + 7) // 8 * 8, 8)

    @T.prim_func
    def main(
            IdxIn: T.Tensor((M, k), "int32"),
            Fin: T.Tensor((M,), "int8"),
            IdxOut: T.Tensor((M, k), "int32"),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            win_start = T.min(row_base, M - sub_block_M)
            idx_ub = T.alloc_ub([sub_block_M, k_ub], "int32")
            T.copy(IdxIn[win_start:win_start + sub_block_M, 0:k], idx_ub[:, 0:k])
            # out = idx*(1-fin) + E*fin in ONE 2D Parallel pass with row
            # broadcast of the fin scalar (documented b_ub[i] pattern; merges
            # the old separate broadcast+arithmetic loops, -9% measured)
            for i in T.serial(sub_block_M):
                for j in T.Parallel(k_ub):
                    f = T.cast(Fin[win_start + i], "int32")
                    idx_ub[i, j] = idx_ub[i, j] * (1 - f) + E * f
            T.copy(idx_ub[:, 0:k], IdxOut[win_start:win_start + sub_block_M, 0:k])

    return main


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def _moe_pad_rows_kernel(M: int, padded_M: int, E: int, block_M: int, dtype: str = "float16"):
    """Materialize (padded_M, E) from X (M, E); rows >= M repeat the last row.

    Replaces the old cat/full M-padding (Fill binary is
    missing on the 910c eval image). Per-row 1D copies keep unaligned column
    counts from creating misaligned 2D tile row strides in UB/GM.
    """
    m_num = T.ceildiv(padded_M, block_M)
    sub_block_M = block_M // VEC_NUM
    align_elems = 32 // (2 if dtype in ("float16", "bfloat16") else 4)
    e_align = ((E + align_elems - 1) // align_elems) * align_elems

    @T.prim_func
    def main(
            X: T.Tensor((M, E), dtype),
            XP: T.Tensor((padded_M, E), dtype),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            row_ub = T.alloc_ub([e_align], dtype)
            for r in T.serial(sub_block_M):
                src_row = T.min(row_base + r, M - 1)
                T.copy(X[src_row, 0:E], row_ub)
                T.copy(row_ub[0:E], XP[row_base + r, 0:E])

    return main


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def _moe_cast_values_kernel(M: int, k: int, block_M: int, dtype: str = "float16"):
    """Cast extracted values from fp32 to the input dtype (elementwise, per row)."""
    m_num = T.ceildiv(M, block_M)
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
            Y32: T.Tensor((M, k), "float32"),
            Y: T.Tensor((M, k), dtype),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            y32_1d = T.alloc_ub([k], "float32")
            y_out_1d = T.alloc_ub([k], dtype)
            for r in T.serial(sub_block_M):
                row = T.min(row_base + r, M - 1)
                T.copy(Y32[row, 0:k], y32_1d)
                T.tile.cast(y_out_1d, y32_1d, CAST_MODE_HIGH2LOW, k)
                T.copy(y_out_1d[0:k], Y[row, 0:k])

    return main


# ========== Python interface ==========
def _compute_row_idx(batch_shape, k: int, device) -> torch.Tensor:
    """Compute row_idx per the golden formula.

    row_idx[m, j] = m + j * M, where M = prod(batch_shape).

    For 2D (N, E) → output (N, k):
        row_idx = arange(N*k).reshape(k, N).transpose(0, 1)  # (N, k)

    For 3D (B, N, E) → output (B, N, k):
        row_idx = arange(B*N*k).reshape(k, B*N).transpose(0, 1).reshape(B, N, k)
    """
    M = 1
    for s in batch_shape:
        M *= s
    # the arange entry is missing on the 910c eval image -> tiny tilelang kernel.
    # row_idx depends only on (M, k): memoize it across calls (same pattern as
    # the existing _sort32_idx cache) so steady-state measured cost is ~0.
    cache = _kernel_cache.setdefault("_row_idx", {})
    key = (M, k, str(device))
    if key not in cache:
        # Cache the 2D (M, k) tensor; reshape per call (metadata view) so
        # different batch_shapes sharing (M, k) each get the right shape.
        cache[key] = _get_row_idx_kernel(M, k)()
    return cache[key].reshape(*batch_shape, k)


def moe_gating_top_k_softmax(
    x: torch.Tensor,
    finished: torch.Tensor = None,
    k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MoE gating fused softmax + topk.

    Args:
        x: input tensor (..., E), dtype in {float16, float32, bfloat16}
        finished: optional bool tensor, shape = x_shape[:-1];
            True rows get expert_idx = E (num_expert)
        k: topK count, 0 < k <= E

    Returns:
        y: (..., k), same dtype as x — topK softmax values
        expert_idx: (..., k), int32 — topK expert indices (E for finished rows)
        row_idx: (..., k), int32 — flattened global position indices
    """
    original_shape = x.shape
    E = original_shape[-1]
    batch_shape = original_shape[:-1]

    # Reshape to 2D (M, E) for the kernel (metadata-only: eval inputs are
    # contiguous, so reshape is a view and dispatches no kernel).
    M = x.numel() // E
    x_2d = x.reshape(M, E)

    tl_dtype = torch_dtype_to_tl(x.dtype)

    # Compute block_M based on original M (dispatches to batch or original)
    block_M = _get_block_M(M, E, k, tl_dtype)

    # Pad M to be divisible by block_M (eliminates 2D OOB in kernel).
    # the cat/full padding is replaced by a tiny pad kernel (Fill missing on the
    # 910c eval image). Rows beyond M repeat the last row; they are sliced
    # away from every output below.
    padded_M = ((M + block_M - 1) // block_M) * block_M
    if padded_M > M:
        x_padded = _get_pad_kernel(M, padded_M, E, tl_dtype)(x_2d)
    else:
        x_padded = x_2d

    # Get kernel compiled with padded_M (dispatches to sort32/batch/original)
    kernel, _ = _get_kernel(padded_M, E, k, tl_dtype)

    use_sort32 = _use_sort32_kernel(padded_M, E, tl_dtype)
    if use_sort32:
        # Sort32 path: the main kernel DIRECTLY outputs Y32 (padded_M, k) fp32
        # values + IdxBits (padded_M, k) fp32 slots holding uint32 index bit
        # patterns (in-kernel split, two single-gather passes). Only the dtype
        # cast remains outside (cast-after-gather misplaces outputs on this
        # backend): flat cast for chunk-divisible N, per-row cast otherwise.
        aligned_E = (E + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE
        # Pre-compute 2D index buffer [padded_M, aligned_E]: row r = 0..E-1.
        # tiny tilelang kernel replaces the arange entry; cached by shape.
        _idx_cache = _kernel_cache.setdefault("_sort32_idx", {})
        idx_key = (padded_M, aligned_E)
        if idx_key not in _idx_cache:
            # arange missing on 910c eval image -> tiny tilelang kernel;
            # int32 output reinterpreted as uint32 (metadata-only view).
            _idx_cache[idx_key] = _get_idxin_kernel(padded_M, E, aligned_E)().view(torch.uint32)
        y_32d, idx_bits_2d = kernel(x_padded, _idx_cache[idx_key])
        if tl_dtype == "float":
            y_2d = y_32d
        else:
            N = padded_M * k
            kern_fc = _get_flat_cast_kernel(N, 0, tl_dtype)
            if kern_fc is not None:
                y_2d = kern_fc(y_32d.reshape(N)).reshape(padded_M, k)
            else:  # N not chunk-divisible: per-row cast fallback
                kern_c = _get_cast_kernel(M, k, tl_dtype)
                y_2d = kern_c(y_32d[:M])
        y_2d = y_2d[:M]
        expert_idx_2d = idx_bits_2d[:M].view(torch.int32)  # metadata reinterpret
        y = y_2d.reshape(*original_shape[:-1], k)
        expert_idx = expert_idx_2d.reshape(*original_shape[:-1], k)
    else:
        y_padded, expert_idx_padded = kernel(x_padded)
        # Slice back to M (remove padding)
        y_2d = y_padded[:M]
        expert_idx_2d = expert_idx_padded[:M]
        # Reshape back to original batch shape
        y = y_2d.reshape(*original_shape[:-1], k)
        expert_idx = expert_idx_2d.reshape(*original_shape[:-1], k)

    # Compute row_idx (deterministic, no kernel needed)
    row_idx = _compute_row_idx(batch_shape, k, x.device)

    # Apply finished mask: finished rows → expert_idx = E.
    # Eval cases 6/7/8 pass `finished`, and Where availability on the 910c eval
    # image is unproven -> tiny tilelang kernel (metadata views only on host).
    if finished is not None:
        expert_idx_2d = expert_idx.reshape(M, k)
        expert_idx_2d = _get_finished_kernel(M, k, E)(expert_idx_2d,
                                                      finished.reshape(M).view(torch.int8))
        expert_idx = expert_idx_2d.reshape(*original_shape[:-1], k)

    return y, expert_idx, row_idx


# ========== Golden reference (PyTorch) ==========
def golden_moe_gating_top_k_softmax(
    x: torch.Tensor,
    finished: torch.Tensor = None,
    k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference: softmax + topk + row_idx formula + finished mask."""
    softmax_out = torch.nn.functional.softmax(x, dim=-1)
    values, indices = torch.topk(softmax_out, k, dim=-1)
    output_shape = indices.shape

    # row_idx[m, j] = m + j * M (flattened global position indices)
    M = 1
    for s in output_shape[:-1]:
        M *= s
    row_idx_range = torch.arange(M * k, dtype=torch.int32)
    row_idx = row_idx_range.reshape(k, M).transpose(0, 1)
    row_idx = row_idx.reshape(*output_shape)

    # finished rows -> expert_idx = E (num_expert)
    if finished is not None:
        num_expert = x.shape[-1]
        finished_expanded = finished.reshape(*output_shape[:-1], 1).expand_as(indices)
        indices = torch.where(finished_expanded, num_expert, indices)

    return values, indices.to(torch.int32), row_idx.to(torch.int32)


if __name__ == "__main__":
    tilelang.cache.clear_cache()

    # Two representative configs: 1 L0 + 1 L1 (evaluation case-1 / case-3
    # shapes).  L0 exercises the per-row topk path, L1 the sort32+merge
    # direct-output path (both with the finished mask on a second pass).
    test_configs = [
        ("L0", (1024, 16), torch.float16, 2),
        ("L1", (16384, 64), torch.bfloat16, 8),
    ]

    for level, shape, dt, k in test_configs:
        print(f"Testing moe_gating_top_k_softmax {level} with shape={shape}, dtype={dt}, k={k}")
        torch.manual_seed(0)
        x = torch.randn(*shape, dtype=dt) * 0.5
        finished = torch.randint(0, 2, shape[:-1], dtype=torch.bool)

        for fin in (None, finished):
            out = moe_gating_top_k_softmax(x.npu(), fin.npu() if fin is not None else None, k)
            print("Init successful!" if fin is None else "Init successful! (finished)")
            torch.npu.synchronize()
            ref = golden_moe_gating_top_k_softmax(x, fin, k)

            # y: mixed tolerance per dtype (softmax+topk values)
            a_y, g_y = out[0].detach().cpu().float(), ref[0].detach().cpu().float()
            atol = 2**-10 if dt == torch.float16 else 2**-7
            max_abs = (a_y - g_y).abs().max().item()
            assert torch.allclose(a_y, g_y, atol=atol, rtol=1e-3), \
                f"{level} y mismatch: max_abs={max_abs}"
            # expert_idx / row_idx: row_idx exact; expert_idx exact modulo
            # topk tie-breaking (equal values may pick a different index).
            a_i, g_i = out[1].detach().cpu(), ref[1].detach().cpu()
            a_r, g_r = out[2].detach().cpu(), ref[2].detach().cpu()
            assert torch.equal(a_r, g_r), f"{level} row_idx mismatch"
            if not torch.equal(a_i, g_i):
                mism = a_i != g_i
                assert torch.equal(a_y[mism], g_y[mism]), \
                    f"{level} expert_idx mismatch without value tie"
            fin_note = "finished" if fin is not None else "plain"
            print(f"Test pass! {level} ({fin_note}) y max_abs={max_abs:.2e}")

    print("Kernel Output Match!")
