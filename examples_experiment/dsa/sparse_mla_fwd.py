# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Varlen sparse MLA (multi-head latent attention) forward for Ascend 910B3.

Given packed queries Q [S, H, 576], shared KV [S_kv, G, 576] and per-row
top-k indices Indices [S, G, topk] over a varlen layout described by cu_seqlens
Offsets, computes Output [S, H, d_v] bfloat16 and log-sum-exp Lse [S, H] fp32.
The kernel runs a persistent grid over (packed row, kv group) tasks: gather
the top-k KV rows, causal + invalid arithmetic mask, QK, streaming online
softmax, PV.

Semantics (kernel-side contract):
  * repeated indices are NOT deduplicated (their contribution is kept);
  * causal masking uses the in-segment position s_i = TokenIndices[s_row, 1];
  * out-of-range / -1 indices: rows are clamped for addressing safety only,
    their contribution is zeroed by the mask chain (score - BIGC, exp
    underflows to 0);
  * first-row self-attention falls out of the mask math naturally;
  * no assumption on index distribution.

Two kernel paths behind one host interface:
  * portable (tl_sparse_mla_fwd): Developer mode with full AUTO pass_configs,
    dual backend (target="ascendc" | "pto"); covers the whole support domain
    (head_kv = heads // kv_group <= 128, bounded by the portable kernel's UB
    budget — see _derived_head_layout).
  * fast (tl_sparse_mla_fwd_fast): Expert mode (manual T.Scope("C")/("V"),
    manual intra-core flags + cross-flag ring protocol), ascendc only; a
    two-pass BI=256 ring-pipelined kernel restricted to (dim, tail_dim) =
    (512, 64), heads=128, kv_group=1, topk % 256 == 0 and NI = topk // 256
    in [1, 6].

Host interface:
    sparse_mla_fwd_interface(q, kv, indices, offsets, sm_scale=None,
        return_p_sum=False, d_v=512, block_I=128, num_stages=2, threads=128,
        target="ascendc") -> (Output, Lse)
"""

import tilelang
from tilelang import language as T

__all__ = [
    "tl_sparse_mla_fwd",
    "tl_sparse_mla_fwd_fast",
    "sparse_mla_fwd_interface",
    "prepare_token_indices",
    "get_cube_core_num",
]

BIGC = 1.0e30  # mask invalid-slot negative bias (finite; exp underflows to 0)

_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


def _derived_head_layout(heads, kv_group):
    """Head layout shared by builder and host (single source of truth).

    Support domain: head_kv = heads // kv_group with (16 <= head_kv <= 64 and
    power of 2) or head_kv % 64 == 0, bounded by head_kv <= 128 (portable
    kernel UB capacity, see the assert below); padded_H != head_kv only
    allowed for kv_group == 1.
    """
    head_kv = heads // kv_group
    assert kv_group >= 1 and heads % kv_group == 0, f"heads={heads} must be divisible by kv_group={kv_group}"
    assert (16 <= head_kv <= 64 and head_kv == tilelang.math.next_power_of_2(head_kv)) or head_kv % 64 == 0, (
        f"head_kv={head_kv} out of supported domain: need (16<=head_kv<=64 and power of 2) or head_kv % 64 == 0"
    )
    # UB capacity bound: the portable kernel's fp32 output accumulator
    # acc_o_all [N_SEG, v_block, DS] spans N_SEG * DS = dim = 512 columns,
    # i.e. 2 KB per Hseg head row, and stays live across the whole softmax
    # loop. head_kv = 128 (v_block = 64 -> 128 KB, the shipped main config)
    # is the largest layout that fits the ~192 KB AIV UB budget together
    # with the softmax scratch; head_kv = 192 needs 192 KB for acc_o_all
    # alone and head_kv = 256 needs 256 KB (262,144 B > 196,352 B usable
    # UB). Reject up front with a clear message instead of letting the
    # build/launch fail (or misbehave) deep inside the kernel.
    assert head_kv <= 128, (
        f"head_kv={head_kv} (heads={heads}, kv_group={kv_group}) exceeds the portable kernel's UB budget: "
        f"acc_o_all alone needs {1024 * head_kv} B of UB, over the 196352 B available per AIV; "
        f"use kv_group > 1 to shrink head_kv, or split heads across calls"
    )
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != head_kv:
        assert kv_group == 1, "H padding path only supports kv_group == 1"
    if head_kv > 64:
        RH = head_kv // 64
        M_SLICE = 64
    else:
        RH = 1
        M_SLICE = padded_H
    Hseg = RH * M_SLICE
    v_block = Hseg // 2
    return head_kv, padded_H, RH, M_SLICE, Hseg, v_block


def _make_jit_builder(target):
    """Closure factory: one builder definition, one @tilelang.jit instance per
    target (the decorator fixes `target` at definition time)."""

    @tilelang.jit(pass_configs=_pass_configs, target=target)
    def _sparse_mla_fwd_builder(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group=1,
        sm_scale=None,
        block_I=128,
        grid_size=20,
    ):
        # ---- compile-time asserts (support domain) ----
        assert dim == tilelang.math.next_power_of_2(dim), f"dim must be a power of 2, dim={dim}"
        assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"tail_dim must be a power of 2, tail_dim={tail_dim}"
        assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
        assert block_I % 2 == 0 and block_I >= 32, f"block_I must be even and >= 32 (AIV half-split), got {block_I}"
        # (dim, tail_dim) = (512, 64) is the only supported pair: the segment
        # loop needs BI | dim, dim/tail must be powers of 2, DDT = 576.
        assert dim == 512 and tail_dim == 64, f"(dim={dim}, tail_dim={tail_dim}) unsupported: layout requires (512, 64)"
        head_kv, padded_H, RH, M_SLICE, Hseg, v_block = _derived_head_layout(heads, kv_group)
        if sm_scale is None:
            sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        BI = block_I
        NI = topk // BI
        HALF = BI // 2
        DDT = dim + tail_dim
        DS = BI  # KV/V segment width (QK K-seg == PV N-seg == V o-chunk)
        N_SEG = dim // DS  # 4 (BI=128) or 8 (BI=64)
        # Gather staging rows per batch: HALF rows are gathered in 2 explicit
        # batches of GKB = HALF // 2; GKB is bounded by the UB budget
        # (kv_stage [GKB, DDT] bf16 aliases the softmax-phase scratch via
        # MEMORY_PLANNING liveness).
        assert HALF % 2 == 0, f"block_I={block_I} must be a multiple of 4 (two-batch gather staging)"
        GKB = HALF // 2
        # Softmax broadcast group span: fixed 8 rows (EG must divide v_block;
        # 8 keeps the 1D slice sources 32-byte aligned for the VEC pipe),
        # variable group count. The planner-allocated reduce workspace plus
        # this broadcast scratch must stay clear of the cast_bf16 window in
        # the UB budget. v_block is a power of two >= 8 (Hseg >= 16).
        assert v_block % 8 == 0, f"v_block={v_block} must be a multiple of 8"
        EG = 8
        NGRP = v_block // 8

        # ---- symbolic dynamic dims (one build covers any S / batch) ----
        seq_len = T.symbolic("seq_len")  # Sigma S
        seq_len_kv = T.symbolic("seq_len_kv")  # Sigma S_kv
        batch_plus_one = T.symbolic("batch_plus_one")

        dtype = "bfloat16"
        indices_dtype = "int32"
        accum_dtype = "float32"

        q_shape = [seq_len, heads, DDT]
        kv_shape = [seq_len_kv, kv_group, DDT]
        indices_shape = [seq_len, kv_group, topk]
        o_shape = [seq_len, heads, dim]
        lse_shape = [seq_len, heads]
        offsets_shape = [batch_plus_one]
        token_indices_shape = [seq_len, 2]

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Offsets: T.Tensor(offsets_shape, indices_dtype),  # type: ignore
            TokenIndices: T.Tensor(token_indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
            workspace_kv: T.Tensor([grid_size, BI, DDT], dtype),  # type: ignore
            workspace_s: T.Tensor([grid_size, Hseg, BI], accum_dtype),  # type: ignore
            workspace_p: T.Tensor([grid_size, Hseg, BI], dtype),  # type: ignore
            workspace_o: T.Tensor([grid_size, Hseg, dim], accum_dtype),  # type: ignore
        ):
            with T.Kernel(grid_size, is_npu=True) as (cid, vid):
                # ================= Cube side: L1 / L0C =================
                # v_all[s] = ws_kv columns [s*DS, (s+1)*DS) as [BI, DS].
                # Serves QK (transpose_B, K=DS) AND PV (non-transposed, N=DS).
                v_all = T.alloc_L1([N_SEG, BI, DS], dtype)
                tail_l1 = T.alloc_L1([BI, tail_dim], dtype)
                l1_dummy = T.alloc_L1([16, 16], dtype)  # CV-pairing dummy read
                q_all = T.alloc_L1([N_SEG, Hseg, DS], dtype)
                qt_l1 = T.alloc_L1([Hseg, tail_dim], dtype)
                p_l1 = T.alloc_L1([Hseg, BI], dtype)
                acc_s_l0c = T.alloc_L0C([Hseg, BI], accum_dtype)
                acc_o_l0c = T.alloc_L0C([Hseg, DS], accum_dtype)

                # ================= Vector side: UB =================
                # AIV-vid owns head rows [vid*v_block, (vid+1)*v_block) of Hseg.
                # acc_o_all[d] = output chunk [v_block, DS]; softmax scratch in
                # acc_s_ub [v_block, BI] (also the ws_o chunk read slot, DS==BI).
                acc_o_all = T.alloc_ub([N_SEG, v_block, DS], accum_dtype)
                acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
                cast_bf16 = T.alloc_ub([v_block, BI], dtype)  # p out / o out cast
                m_i = T.alloc_ub([v_block], accum_dtype)
                m_prev = T.alloc_ub([v_block], accum_dtype)
                sumexp = T.alloc_ub([v_block], accum_dtype)
                sumexp_i = T.alloc_ub([v_block], accum_dtype)
                indices_ub = T.alloc_ub([BI], indices_dtype)
                rows_int = T.alloc_ub([BI], indices_dtype)
                idx_f = T.alloc_ub([BI], accum_dtype)
                t_ub = T.alloc_ub([BI], accum_dtype)
                m1_ub = T.alloc_ub([BI], accum_dtype)
                m2_ub = T.alloc_ub([BI], accum_dtype)
                mask_ub = T.alloc_ub([BI], accum_dtype)
                kv_stage = T.alloc_ub([GKB, DDT], dtype)  # gather staging
                # group-row broadcast scratch for the vectorized softmax
                # chain (sizing rationale at the EG constant above)
                bcast_h = T.alloc_ub([EG, BI], accum_dtype)

                total = seq_len * kv_group
                ntask = T.max(0, T.ceildiv(total - cid, grid_size))
                for k in T.serial(ntask):
                    pid = k * grid_size + cid
                    s_row = pid % seq_len  # row-major: same-g rows adjacent
                    g_i = pid // seq_len
                    b_i = TokenIndices[s_row, 0]
                    s_i = TokenIndices[s_row, 1]
                    bos = Offsets[b_i]
                    H0 = g_i * padded_H  # Q head-segment base for this task

                    # ---- [C] Q load, once per task (whole head rows, per K seg)
                    for s in range(N_SEG):
                        T.copy(
                            Q[s_row, H0 : H0 + Hseg, s * DS : (s + 1) * DS],
                            q_all[s, :, :],
                        )
                    T.copy(Q[s_row, H0 : H0 + Hseg, dim:DDT], qt_l1)

                    # ---- [V] softmax state init (per task) ----
                    # m floor = -BIGC/2: low enough to sit BELOW every finite
                    # score of practical bf16 inputs (a -2**30 floor sits ABOVE
                    # legal extremes, e.g. Q=-8192/KV=8192 with the default
                    # scale gives scores ~-1.6e9, which made every exp
                    # underflow and the epilogue divide 0/0 -> NaN), yet high
                    # enough that all-invalid rows' masked scores (-BIGC)
                    # still underflow to 0 against it. A finite floor also
                    # keeps alpha = exp(m_prev - m_i) free of -inf - -inf NaN.
                    for d in range(N_SEG):
                        T.tile.fill(acc_o_all[d, :, :], 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(m_i, -(BIGC * 0.5))

                    for i in T.serial(NI):
                        # ---- [V] indices block + mask arithmetic chain ----
                        # mask = 1{0 <= idx <= s_i}; bias = BIGC * (1 - mask)
                        # (masked score -> x*mask - bias). mask via the min
                        # form: clamp01(min(idx + 1, s_i + 1 - idx)) — both
                        # window terms are >= 1 inside [0, s_i] and either is
                        # <= 0 outside, so min-then-clamp01 is exact. bias =
                        # mask * (-BIGC) + BIGC == BIGC * (1 - mask) exactly
                        # for mask in {0, 1}. All-fp32 scalar chains (bf16
                        # scalar arithmetic unsupported); sub with a scalar
                        # LEFT operand replaced by mul(-1)+add.
                        T.copy(Indices[s_row, g_i, i * BI : (i + 1) * BI], indices_ub)
                        T.copy(indices_ub, idx_f)  # int32 -> fp32 auto-cast
                        # t_ub = idx + 1; m1_ub = (s_i + 1) - idx (mul+add form)
                        T.tile.add(t_ub, idx_f, T.float32(1.0))
                        T.tile.mul(m1_ub, idx_f, T.float32(-1.0))
                        T.tile.add(m1_ub, m1_ub, T.float32(s_i + 1))
                        # mask = clamp01(min(t_ub, m1_ub)); m2_ub holds the min
                        T.tile.min(m2_ub, t_ub, m1_ub)
                        T.tile.max(mask_ub, m2_ub, T.float32(0.0))
                        T.tile.min(mask_ub, mask_ub, T.float32(1.0))
                        # bias = mask * (-BIGC) + BIGC (m1_ub reused)
                        T.tile.mul(m1_ub, mask_ub, T.float32(-BIGC))
                        T.tile.add(m1_ub, m1_ub, T.float32(BIGC))
                        # rows = clamp(idx + bos, 0, seq_len_kv - 1): addressing
                        # guard only (contribution zeroed by mask). fp32 domain
                        # min/max with symbolic bound, cast back to int32.
                        T.tile.add(idx_f, idx_f, T.float32(bos))
                        T.tile.max(idx_f, idx_f, T.float32(0.0))
                        T.tile.min(idx_f, idx_f, T.float32(seq_len_kv - 1))
                        T.copy(idx_f, rows_int)  # fp32 -> int32 (values exact)

                        # ---- [V] gather: AIV-vid fetches its HALF rows into
                        #      kv_stage [GKB, DDT] and writes each batch out
                        #      as ONE whole-2D MTE3 burst. Whole-2D T.copy
                        #      sources only (2D column-slice sources are not
                        #      supported on ascendc). AUTO_CV_SYNC pairs
                        #      statements in order and each C statement may
                        #      only depend on its paired (or earlier) V
                        #      statement: the C-side s-loop/tail reads need
                        #      ALL rows, so V emits [batch0, batch1, mini]
                        #      where the trailing idempotent 1-row rewrite
                        #      (kv_stage still holds the last batch-1 row)
                        #      pairs with the C-side tail read — 3:3
                        #      statements. Rows come from the clamped
                        #      rows_int (addressing safety); invalid slots
                        #      are masked to 0 by the mask chain.
                        for j in range(GKB):
                            kv_row = rows_int[vid * HALF + j]
                            T.copy(KV[kv_row, g_i, :], kv_stage[j, :])
                        T.copy(
                            kv_stage,
                            workspace_kv[cid, vid * HALF : vid * HALF + GKB, :],
                        )
                        for j in range(GKB):
                            kv_row = rows_int[vid * HALF + GKB + j]
                            T.copy(KV[kv_row, g_i, :], kv_stage[j, :])
                        T.copy(
                            kv_stage,
                            workspace_kv[cid, vid * HALF + GKB : (vid + 1) * HALF, :],
                        )
                        T.copy(
                            kv_stage[GKB - 1, :],
                            workspace_kv[cid, (vid + 1) * HALF - 1, :],
                        )

                        # ---- [C] ws_kv -> L1 segments + tail (3D regions) ----
                        # stmt 1: tiny dummy read of rows [0:16] x cols [0:16]
                        # (produced by V stmt 1 / batch 0). AUTO_CV_SYNC pairs
                        # statements in order; without this dummy the s-loop
                        # read would pair with V stmt 1 (batch 0 only) and
                        # race batch 1's write.
                        T.copy(workspace_kv[cid, 0:16, 0:16], l1_dummy)
                        for s in range(N_SEG):
                            T.copy(
                                workspace_kv[cid, 0:BI, s * DS : (s + 1) * DS],
                                v_all[s, :, :],
                            )
                        T.copy(workspace_kv[cid, 0:BI, dim:DDT], tail_l1)

                        # ---- [C] QK: K = DS * N_SEG + tail_dim, M = Hseg ----
                        T.gemm_v0(
                            q_all[0, :, :],
                            v_all[0, :, :],
                            acc_s_l0c,
                            transpose_B=True,
                            init=True,
                        )
                        for s in range(1, N_SEG):
                            T.gemm_v0(
                                q_all[s, :, :],
                                v_all[s, :, :],
                                acc_s_l0c,
                                transpose_B=True,
                                init=False,
                            )
                        T.gemm_v0(qt_l1, tail_l1, acc_s_l0c, transpose_B=True, init=False)
                        T.copy(acc_s_l0c, workspace_s[cid, 0:Hseg, :])

                        # ---- [C] PV: N split into N_SEG x DS, M = Hseg ----
                        T.copy(workspace_p[cid, 0:Hseg, :], p_l1)
                        for s in range(N_SEG):
                            T.gemm_v0(p_l1, v_all[s, :, :], acc_o_l0c, init=True)
                            T.copy(
                                acc_o_l0c,
                                workspace_o[cid, 0:Hseg, s * DS : (s + 1) * DS],
                            )

                        # ---- [V] online softmax on head rows [vid*v_block, ...) ----
                        T.copy(m_i, m_prev)
                        T.copy(
                            workspace_s[cid, vid * v_block : (vid + 1) * v_block, :],
                            acc_s_ub,
                        )
                        T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)  # scaled domain
                        # mask apply: x_p = x * mask - bias (-BIGC for invalid
                        # slots; exp underflows to 0, no -inf/NaN hazard; bias
                        # lives in m1_ub after the 1D chain above). Vectorized
                        # through the group broadcast scratch: axis=0
                        # broadcast for the per-slot mask/bias + row-group
                        # region ops.
                        T.tile.broadcast(bcast_h, mask_ub, axis=0)  # [BI] -> [EG, BI]
                        for g in range(NGRP):
                            T.tile.mul(
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                bcast_h,
                            )
                        T.tile.broadcast(bcast_h, m1_ub, axis=0)  # bias (overwrite)
                        for g in range(NGRP):
                            T.tile.sub(
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                bcast_h,
                            )
                        # m_i update + alpha. Reduce split into per-group
                        # calls: the planner-sized reduce workspace shrinks
                        # from [v_block, BI] to one group tile, keeping the
                        # softmax-phase UB footprint clear of the
                        # reduce/broadcast/cast window conflicts.
                        for g in range(NGRP):
                            T.reduce_max(
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                m_i[g * EG : (g + 1) * EG],
                                dim=-1,
                            )
                        T.tile.max(m_i, m_i, m_prev)
                        T.tile.sub(m_prev, m_prev, m_i)
                        T.tile.exp(m_prev, m_prev)  # alpha
                        # p = exp(x_p - m_i): per-row scalars broadcast along
                        # columns (axis=1, 1D slice sources), one group at a time
                        for g in range(NGRP):
                            T.tile.broadcast(bcast_h, m_i[g * EG : (g + 1) * EG], axis=1)
                            T.tile.sub(
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                bcast_h,
                            )
                        T.tile.exp(acc_s_ub, acc_s_ub)
                        # sumexp = sumexp * alpha + sum(p) (per-group)
                        for g in range(NGRP):
                            T.reduce_sum(
                                acc_s_ub[g * EG : (g + 1) * EG, :],
                                sumexp_i[g * EG : (g + 1) * EG],
                                dim=-1,
                            )
                        T.tile.mul(sumexp, sumexp, m_prev)
                        T.tile.add(sumexp, sumexp, sumexp_i)
                        # acc_o *= alpha: 3D middle-dim region muls (d-outer/
                        # g-inner keeps broadcast+mul as siblings in the
                        # inner loop).
                        for d in range(N_SEG):
                            for g in range(NGRP):
                                T.tile.broadcast(bcast_h, m_prev[g * EG : (g + 1) * EG], axis=1)
                                T.tile.mul(
                                    acc_o_all[d, g * EG : (g + 1) * EG, :],
                                    acc_o_all[d, g * EG : (g + 1) * EG, :],
                                    bcast_h,
                                )
                        # p -> bf16 -> ws_p
                        T.copy(acc_s_ub, cast_bf16)
                        T.copy(
                            cast_bf16,
                            workspace_p[cid, vid * v_block : (vid + 1) * v_block, :],
                        )
                        # acc_o += ws_o chunks (acc_s_ub reused as read slot)
                        for d in range(N_SEG):
                            T.copy(
                                workspace_o[
                                    cid,
                                    vid * v_block : (vid + 1) * v_block,
                                    d * DS : (d + 1) * DS,
                                ],
                                acc_s_ub,
                            )
                            T.tile.add(acc_o_all[d, :, :], acc_o_all[d, :, :], acc_s_ub)

                    # ---- [V tail] normalize + Lse + writeback ----
                    # Empty-row guard: a row whose top-k slots are ALL
                    # invalid has sumexp == 0 (its masked scores -BIGC all
                    # underflow against the m floor above). Divide by
                    # max(sumexp, 1) so such rows emit Output = 0 instead of
                    # 0/0 NaN; for non-empty rows sumexp >= 1 (the global-max
                    # slot contributes exp(0) = 1), so the clamp is an exact
                    # no-op. m_prev is dead past the NI loop — reuse it as
                    # the safe-denominator buffer. Lse below still uses the
                    # raw sumexp: ln(0) = -inf gives empty rows -inf.
                    T.tile.max(m_prev, sumexp, T.float32(1.0))
                    for d in range(N_SEG):
                        for h in range(v_block):
                            T.tile.div(acc_o_all[d, h, :], acc_o_all[d, h, :], m_prev[h])
                    T.tile.ln(sumexp, sumexp)
                    T.tile.add(sumexp, sumexp, m_i)  # Lse = ln(sumexp) + m_i
                    for d in range(N_SEG):
                        T.copy(acc_o_all[d, :, :], acc_s_ub)  # 3D -> 2D fp32
                        T.copy(acc_s_ub, cast_bf16)  # fp32 -> bf16
                        T.copy(
                            cast_bf16,
                            Output[
                                s_row,
                                H0 + vid * v_block : H0 + (vid + 1) * v_block,
                                d * DS : (d + 1) * DS,
                            ],
                        )
                    T.copy(
                        sumexp,
                        Lse[s_row, H0 + vid * v_block : H0 + (vid + 1) * v_block],
                    )

        return main

    return _sparse_mla_fwd_builder


_JIT_BUILDERS: dict = {}


def tl_sparse_mla_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    block_I=128,
    grid_size=20,
    target="ascendc",
):
    """Build the sparse_mla_fwd kernel for the requested backend.

    Args:
        heads: total head count H (head_kv = heads // kv_group, see domain asserts).
        dim: QK main / V dimension D (must be 512).
        tail_dim: QK tail dimension (must be 64).
        topk: selected positions per row (topk % block_I == 0).
        kv_group: KV group count G.
        sm_scale: defaults to (dim + tail_dim) ** -0.5.
        block_I: topk-direction tile BI in {64, 128}.
        grid_size: persistent grid (host queries cube_core_num; 910B3 = 20).
        target: "ascendc" (default) or "pto".

    Returns:
        Callable kernel: func(Q, KV, Indices, Offsets, TokenIndices, Output, Lse,
        workspace_kv, workspace_s, workspace_p, workspace_o) — all buffers
        explicit (host torch.zeros), no framework-allocated outputs.
    """
    if target not in _JIT_BUILDERS:
        _JIT_BUILDERS[target] = _make_jit_builder(target)
    return _JIT_BUILDERS[target](heads, dim, tail_dim, topk, kv_group, sm_scale, block_I, grid_size)


# ============================================================================
# Host side: data preparation + call entry (TokenIndices precompute on CPU
# with a single H2D copy, torch.zeros allocations, and the kernel call).
# ============================================================================


def _tensor_cache(fn):
    """Single-entry cache keyed on input tensor identity + mutation version.

    Keeps prepare_token_indices' CPU->H2D copy out of steady-state benchmark
    loops. Identity alone is NOT a valid key: cu_seqlens updated in place
    (offsets.copy_(...), offsets[1] = ...) keeps the same Python object while
    its content changes, and a stale TokenIndices would silently mix the old
    segment mapping with the new boundaries (wrong gather, wrong causal mask,
    wrong output). torch bumps a per-storage version counter on every in-place
    mutation, so (identity, _version) catches those updates with pure
    metadata checks — no device sync, so the steady-state fast path is kept.
    Residual blind spot: writes through ``tensor.data`` bypass the version
    counter (autograd escape hatch); pass a fresh tensor (or use regular
    in-place ops) when mutating cu_seqlens that way.
    """
    last_args = None
    last_versions = None
    last_result = None

    def wrapper(*args):
        nonlocal last_args, last_versions, last_result
        versions = tuple(getattr(a, "_version", None) for a in args)
        if (
            last_args is not None
            and len(args) == len(last_args)
            and all(a is b for a, b in zip(args, last_args))
            and versions == last_versions
        ):
            return last_result
        result = fn(*args)
        last_args, last_versions, last_result = args, versions, result
        return result

    return wrapper


def get_cube_core_num(default=20):
    """Query device cube core count (grid_size compile constant; 910B3 = 20)."""
    try:
        import torch

        if torch.npu.is_available():
            n = torch.npu.get_device_properties(0).cube_core_num
            if isinstance(n, int) and n > 0:
                return n
    except Exception:  # pragma: no cover - query failure falls back to default
        pass
    return default


@_tensor_cache
def prepare_token_indices(offsets):
    """Build TokenIndices [Sigma S, 2] int32 ((batch_id, in-segment position)).

    CPU construction -> single H2D copy.
    """
    import torch

    off = offsets.detach().cpu().tolist()
    batch_ids = []
    positions = []
    for b in range(len(off) - 1):
        n = int(off[b + 1]) - int(off[b])
        batch_ids.append(torch.full((n,), b, dtype=torch.int32))
        positions.append(torch.arange(n, dtype=torch.int32))
    if batch_ids:
        ti = torch.stack([torch.cat(batch_ids), torch.cat(positions)], dim=1)
    else:
        ti = torch.zeros((0, 2), dtype=torch.int32)
    return ti.to(offsets.device)


_BUILD_CACHE: dict = {}


def sparse_mla_fwd_interface(
    q,
    kv,
    indices,
    offsets,
    sm_scale=None,
    return_p_sum=False,
    d_v=512,
    block_I=128,
    num_stages=2,
    threads=128,
    target="ascendc",
):
    """Host entry (main-repo signature compatible).

    Args:
        q: [Sigma S, heads, dim+tail_dim] bfloat16 (NPU); head_kv =
            heads // kv_group must satisfy the kernel domain, in particular
            head_kv <= 128 (portable-path UB budget; see _derived_head_layout)
        kv: [Sigma S_kv, kv_group, dim+tail_dim] bfloat16 (NPU)
        indices: [Sigma S, kv_group, topk] int32 (NPU), values in {-1} U int32
            (out-of-range clamped for addressing, contribution masked to 0)
        offsets: [batch+1] int32 cu_seqlens
        sm_scale: defaults to (dim + tail_dim) ** -0.5
        return_p_sum: must be False (fwd-only file)
        d_v: V dimension (dim_plus_tail_dim - d_v = tail_dim)
        block_I: topk tile for the portable path; explicit 64/128 respected,
            other values fall back to the default 128. The fast path ignores
            it: that kernel's internal BI=256 is structural — the dispatch
            gate judges (topk, H, G, dim, tail).
        num_stages / threads: kept for signature compatibility, unused on Ascend
        target: "ascendc" (default) or "pto"

    Returns:
        (Output [Sigma S, heads, d_v] bfloat16, Lse [Sigma S, heads] float32)
    """
    import torch

    assert return_p_sum == False, "This kernel file is for fwd only"  # noqa: E712  (main-repo verbatim)
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    seq_len, heads, dim_plus_tail_dim = q.shape
    seq_len_kv, kv_group, _ = kv.shape
    assert seq_len == seq_len_kv
    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v
    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    _, _, topk = indices.shape
    assert indices.shape == (seq_len, kv_group, topk)

    if block_I in (64, 128):
        bi = block_I
    else:
        bi = 128  # default; non-{64,128} user values not honoured

    grid_size = get_cube_core_num()

    # ---- fast/generic dispatch: the two-pass fast kernel is the ONLY fast
    # path; out-of-domain shapes (topk % 256 != 0, NI > 6, kv_group=2,
    # heads=64, pto backend, ...) go to the portable path. The gate is
    # structural: user block_I is a portable-path tiling hint only — the
    # fast kernel's internal BI=256 is passed to the gate, which judges
    # (topk, H, G, dim, tail). No value-range predicate on Q/KV is applied:
    # the fast path is bf16 all-chain (only fp32->bf16 rounding on the P
    # store), so an fp16 overflow bound would be meaningless; dispatch is on
    # the gate domain alone.
    use_fast = target == "ascendc" and _fast_gate_ok(heads, dim, tail_dim, topk, kv_group, 256)
    if use_fast:
        ni = topk // 256
        fkey = ("fastv2", heads, topk, sm_scale, grid_size)
        ffunc = _BUILD_CACHE.get(fkey)
        if ffunc is None:
            ffunc = tl_sparse_mla_fwd_fast(
                heads=heads,
                dim=dim,
                tail_dim=tail_dim,
                topk=topk,
                kv_group=kv_group,
                sm_scale=sm_scale,
                block_I=256,
                grid_size=grid_size,
            )
            _BUILD_CACHE[fkey] = ffunc
        token_indices = prepare_token_indices(offsets)  # CPU build -> one H2D
        out = torch.zeros((seq_len, heads, dim), dtype=q.dtype, device=q.device)
        lse = torch.zeros((seq_len, heads), dtype=torch.float32, device=q.device)
        # fast-path workspaces (two-pass): ws_kv ring with DEPTH=NI+1 slots of
        # [256, 576]; ws_s / ws_p with NI slots of [HSEG=128, BI=256]; ws_o
        # one [HSEG, DIM] fp32 accumulator per core (single flush per task)
        workspace_kv = torch.zeros((grid_size, (ni + 1) * 256, dim + tail_dim), dtype=torch.bfloat16, device=q.device)
        workspace_s = torch.zeros((grid_size, ni * 128, 256), dtype=torch.float32, device=q.device)
        workspace_p = torch.zeros((grid_size, ni * 128, 256), dtype=torch.bfloat16, device=q.device)
        workspace_o = torch.zeros((grid_size, 128, dim), dtype=torch.float32, device=q.device)
        ffunc(q, kv, indices, offsets, token_indices, out, lse, workspace_kv, workspace_s, workspace_p, workspace_o)
        return out, lse

    key = (heads, dim, tail_dim, topk, kv_group, sm_scale, bi, grid_size, target)
    func = _BUILD_CACHE.get(key)
    if func is None:
        func = tl_sparse_mla_fwd(
            heads=heads,
            dim=dim,
            tail_dim=tail_dim,
            topk=topk,
            kv_group=kv_group,
            sm_scale=sm_scale,
            block_I=bi,
            grid_size=grid_size,
            target=target,
        )
        _BUILD_CACHE[key] = func

    _, _, _, _, Hseg, _ = _derived_head_layout(heads, kv_group)
    token_indices = prepare_token_indices(offsets)  # CPU build -> one H2D

    out = torch.zeros((seq_len, heads, dim), dtype=q.dtype, device=q.device)
    lse = torch.zeros((seq_len, heads), dtype=torch.float32, device=q.device)
    workspace_kv = torch.zeros((grid_size, bi, dim + tail_dim), dtype=torch.bfloat16, device=q.device)
    workspace_s = torch.zeros((grid_size, Hseg, bi), dtype=torch.float32, device=q.device)
    workspace_p = torch.zeros((grid_size, Hseg, bi), dtype=torch.bfloat16, device=q.device)
    workspace_o = torch.zeros((grid_size, Hseg, dim), dtype=torch.float32, device=q.device)
    func(q, kv, indices, offsets, token_indices, out, lse, workspace_kv, workspace_s, workspace_p, workspace_o)
    return out, lse


# ============================================================================
# Fast path: Expert mode, ascendc only — two-pass ring-pipelined kernel.
#
# KV route: the vector side gathers top-k KV rows into a GM ring buffer
# ws_kv (one [BI, DDT] slot per block); the cube side bulk-reads each slot
# into L1 segments (3D L1 regions as gemm_v0 operands).
#
# Two-pass softmax per task (NI = topk // BI blocks, BI = 256):
#   * pass 1 (cube): all NI QK blocks -> ws_s (NI slots, no intra-task
#     reuse); vector reduces the task-global row max m_fin (NO online
#     rescale) and saves each block's mask in UB mask_mem;
#   * pass 2 (vector): re-reads ws_s, recomputes P = exp(s_masked - m_fin)
#     -> ws_p; (cube): accumulates PV across blocks in L0C acc3d[2, 64, 256]
#     (init=False) and flushes ONE ws_o band per (task, N-seg);
#   * M-slice = vid (HSEG 128 heads = 2 x 64 per AIV).
#
# ws_kv ring depth DEPTH = NI + 1, not NI: task k's NI slots must stay intact
# from the pass-1 write until the pass-2 re-read (fenced by the F2 chain),
# while the LEAD=1 lookahead gather of task k+1's block 0 fires during V's
# pass-1 iteration NI-1 — NI live blocks + 1 lookahead = NI + 1 in flight.
#
# Cross-flag budget: F0[2] (ws_kv, V->C) + F1A[NI] (ws_s, C->V) + F2[NI]
# (ws_p, V->C) + F3[2] (ws_o flush, C->V) = 2*NI + 4 <= 16 hardware flag
# slots -> NI <= 6 (asserted). F0 uses block-parity ids (n % 2): the F1A
# chain keeps V at most 2 blocks ahead of C's QK consumption, so a 2-id
# set/wait alternation covers any ring depth.
#
# All-invalid-row corner: the m_fin floor -BIGC/2 sits below every finite
# score of practical bf16 inputs (a -2**30 floor would sit ABOVE legal
# extremes like Q=-8192/KV=8192 scores ~-1.6e9 and NaN the epilogue), yet
# above -BIGC, so all-invalid rows' masked scores still underflow to 0 and
# sumexp stays 0; the epilogue's max(sumexp, 1) guard then emits Output = 0
# / Lse = -inf instead of 0/0 -> NaN.
# ============================================================================


def _fast_gate_ok(heads, dim, tail_dim, topk, kv_group, block_I):
    """Host mirror of the fast-path compile-time domain."""
    ni = topk // block_I if topk % block_I == 0 else 99
    return (
        dim == 512
        and tail_dim == 64
        and block_I == 256
        and kv_group == 1
        and heads == 128
        and topk % block_I == 0
        and 1 <= ni <= 6  # flag budget: 2*NI + 4 <= 16
    )


@tilelang.jit(target="ascendc")  # Expert mode (manual sync)
def tl_sparse_mla_fwd_fast(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    block_I=256,
    grid_size=20,
):
    """Build the two-pass BI=256 fast kernel (Expert mode, ascendc only; see
    the section header above).

    Domain (asserts mirror _fast_gate_ok): (dim, tail_dim) == (512, 64),
    block_I == 256, (heads, kv_group) == (128, 1), topk % 256 == 0 and
    NI = topk // 256 in [1, 6] (cross-flag budget 2*NI + 4 <= 16).

    Returns:
        Callable kernel: func(Q, KV, Indices, Offsets, TokenIndices, Output,
        Lse, ws_kv [grid, (NI+1)*256, 576] bf16, ws_s [grid, NI*128, 256] fp32,
        ws_p [grid, NI*128, 256] bf16, ws_o [grid, 128, 512] fp32).
    """
    # ---- compile-time asserts (fast domain; host mirror: _fast_gate_ok) ----
    assert dim == 512 and tail_dim == 64, f"(dim, tail_dim) must be (512, 64), got ({dim}, {tail_dim})"
    assert block_I == 256, f"fast path requires block_I=256, got {block_I}"
    assert kv_group == 1 and heads == 128, f"fast path requires (heads=128, kv_group=1), got ({heads}, {kv_group})"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    NI = topk // block_I
    assert 1 <= NI <= 6, f"fast path flag budget requires NI=topk/256 in [1,6], got {NI}"
    if sm_scale is None:
        sm_scale = (dim + tail_dim) ** -0.5

    # structural constants (fixed by the domain asserts above)
    HSEG, V_BLOCK, BI = 128, 64, 256  # heads, per-vid head slice, block cols
    HALF = 128  # per-vid gather rows (BI / 2 vids)
    DDT, DIM, TAIL = 576, 512, 64
    DS, N_SEG = 256, 2  # QK K-seg / PV N-seg / o-chunk width (DS*2 + TAIL = DDT)
    GKB = 16  # gather staging rows per batch
    NBATCH = HALF // GKB  # 8
    EG = 8  # softmax group rows (VEC 32B alignment)
    NGRP = V_BLOCK // EG  # 8
    DEPTH = NI + 1  # ws_kv ring slots: task k's NI blocks live from the
    # pass-1 write until the pass-2 re-read (F2-chain fenced) + 1 LEAD
    # lookahead block of task k+1 gathered during V's pass-1 iter NI-1
    # (pigeonhole: NI live blocks + 1 lookahead; see the section header)
    IDXSLOTS = 2  # indices ring (pass1 lookahead only: pass2 reuses the
    # per-block masks saved in mask_mem — no Indices GM re-read, no ring
    # interaction)
    KL0 = 64  # gemm_v0 kL0Size (B slot 256*64*2 = 32KB == multi-step budget)
    # flag families: F0 ws_kv gather (V->C, BLOCK-PARITY ids n % 2 — the F1A
    # chain keeps V <= 2 blocks ahead of C's QK consumption, so a 2-id
    # set/wait alternation covers ANY ring depth), F1A ws_s slot (C->V, NI),
    # F2 ws_p slot (V->C, NI), F3 ws_o flush (C->V, BOTH ids set after the
    # complete pass-2; V waits BOTH — its next-task gathers may only fire
    # past C's whole pass 2, ws_kv ring safety)
    F0_BASE, F1A_BASE, F2_BASE, F3_BASE = 0, 2, 2 + NI, 2 + 2 * NI

    dtype = "bfloat16"
    indices_dtype = "int32"
    accum_dtype = "float32"

    seq_len = T.symbolic("seq_len")  # Sigma S
    seq_len_kv = T.symbolic("seq_len_kv")  # Sigma S_kv
    batch_plus_one = T.symbolic("batch_plus_one")

    @T.prim_func
    def main(
        Q: T.Tensor([seq_len, heads, DDT], dtype),  # type: ignore
        KV: T.Tensor([seq_len_kv, kv_group, DDT], dtype),  # type: ignore
        Indices: T.Tensor([seq_len, kv_group, topk], indices_dtype),  # type: ignore
        Offsets: T.Tensor([batch_plus_one], indices_dtype),  # type: ignore
        TokenIndices: T.Tensor([seq_len, 2], indices_dtype),  # type: ignore
        Output: T.Tensor([seq_len, heads, DIM], dtype),  # type: ignore
        Lse: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        ws_kv: T.Tensor([grid_size, DEPTH * BI, DDT], dtype),  # type: ignore
        ws_s: T.Tensor([grid_size, NI * HSEG, BI], accum_dtype),  # type: ignore
        ws_p: T.Tensor([grid_size, NI * HSEG, BI], dtype),  # type: ignore
        ws_o: T.Tensor([grid_size, HSEG, DIM], accum_dtype),  # type: ignore
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            # ---- Cube side: L1 464KB / L0C 128KB (== full) ----
            v_all = T.alloc_L1([N_SEG, BI, DS], dtype)  # KV resident: 256KB
            tail_l1 = T.alloc_L1([BI, TAIL], dtype)  # 32KB
            q_a = T.alloc_L1([N_SEG, V_BLOCK, DS], dtype)  # ms=0 Q: 64KB
            q_b = T.alloc_L1([N_SEG, V_BLOCK, DS], dtype)  # ms=1 Q: 64KB
            qt_all = T.alloc_L1([2, V_BLOCK, TAIL], dtype)  # 16KB
            p_l1 = T.alloc_L1([V_BLOCK, BI], dtype)  # 32KB, reused per block
            # pass1 QK -> slot 0; pass2 PV accumulates slot0/slot1 across
            # blocks (3D-slot + init=False form)
            acc3d = T.alloc_L0C([2, V_BLOCK, BI], accum_dtype)  # 128KB
            # ---- Vector side: UB ~103KB + NI*1KB per vid (<= 109KB @NI=6,
            #      budget 196352B) ----
            s_ub = T.alloc_ub([V_BLOCK, BI], accum_dtype)  # 64KB ws_s slot
            kv_stage = T.alloc_ub([GKB, DDT], dtype)  # 18KB gather staging
            cast_g = T.alloc_ub([EG, BI], dtype)  # 4KB cast slot
            bcast_h = T.alloc_ub([EG, BI], accum_dtype)  # 8KB broadcast
            indices_ub = T.alloc_ub([IDXSLOTS, BI], indices_dtype)  # 2KB
            rows_ub = T.alloc_ub([BI], indices_dtype)  # 1KB
            idx_f = T.alloc_ub([BI], accum_dtype)  # 1KB idx -> fp32
            t_ub = T.alloc_ub([BI], accum_dtype)  # 1KB mask chain
            m1_ub = T.alloc_ub([BI], accum_dtype)  # 1KB window term
            mask_mem = T.alloc_ub([NI, BI], accum_dtype)  # <=6KB pass-1 masks
            m_fin = T.alloc_ub([V_BLOCK], accum_dtype)  # task-global max
            sumexp = T.alloc_ub([V_BLOCK], accum_dtype)  # task denom
            sumexp_i = T.alloc_ub([V_BLOCK], accum_dtype)  # rowmax (pass1) /
            # rowsum (pass2) scratch — V-pipe overwrite order, no hazard

            total = seq_len * kv_group
            ntask = T.max(0, T.ceildiv(total - cid, grid_size))
            nblk = ntask * NI  # flat block count (flag protocol domain)

            # ================= Cube stream =================
            # (intra-core L1/L0C hazards covered by gemm_v0 internal flags)
            with T.Scope("C"):
                for k in T.serial(ntask):
                    pid = k * grid_size + cid
                    s_row = pid % seq_len  # g == 0 in the fast domain
                    # ---- pass 1: Q load (column segments) ----
                    # (literal 2-slice unroll: range() loops are TIR loops,
                    # so python-level buffer selection is not available)
                    for s in range(N_SEG):
                        T.copy(
                            Q[s_row, 0:V_BLOCK, s * DS : (s + 1) * DS],
                            q_a[s, :, :],
                        )
                    T.copy(Q[s_row, 0:V_BLOCK, DIM:DDT], qt_all[0, :, :])
                    for s in range(N_SEG):
                        T.copy(
                            Q[s_row, V_BLOCK : 2 * V_BLOCK, s * DS : (s + 1) * DS],
                            q_b[s, :, :],
                        )
                    T.copy(Q[s_row, V_BLOCK : 2 * V_BLOCK, DIM:DDT], qt_all[1, :, :])
                    # ---- pass 1: all NI QKs -> ws_s ----
                    for t in T.serial(NI):
                        n = k * NI + t
                        slot = n % DEPTH
                        # F0 parity id (n % 2), NOT the slot id: ring depth
                        # NI+1 > flag count 2 — the F1A lag keeps the
                        # set/wait alternation exact (see F0_BASE note)
                        T.wait_cross_flag(F0_BASE + n % 2)
                        for s in range(N_SEG):
                            T.copy(
                                ws_kv[cid, slot * BI : (slot + 1) * BI, s * DS : (s + 1) * DS],
                                v_all[s, :, :],
                            )
                        T.copy(ws_kv[cid, slot * BI : (slot + 1) * BI, DIM:DDT], tail_l1)
                        # QK per M-slice: 3-segment init chain (kL0Size=64
                        # -> B slot 256*64*2 = 32KB == budget)
                        # ms=0 (heads 0:64)
                        T.gemm_v0(
                            q_a[0, :, :],
                            v_all[0, :, :],
                            acc3d[0, :, :],
                            transpose_B=True,
                            init=True,
                            kL0Size=KL0,
                        )
                        T.gemm_v0(
                            q_a[1, :, :],
                            v_all[1, :, :],
                            acc3d[0, :, :],
                            transpose_B=True,
                            init=False,
                            kL0Size=KL0,
                        )
                        T.gemm_v0(
                            qt_all[0, :, :],
                            tail_l1,
                            acc3d[0, :, :],
                            transpose_B=True,
                            init=False,
                            kL0Size=KL0,
                        )
                        T.copy(
                            acc3d[0, :, :],
                            ws_s[cid, t * HSEG : t * HSEG + V_BLOCK, :],
                        )
                        # ms=1 (heads 64:128)
                        T.gemm_v0(
                            q_b[0, :, :],
                            v_all[0, :, :],
                            acc3d[0, :, :],
                            transpose_B=True,
                            init=True,
                            kL0Size=KL0,
                        )
                        T.gemm_v0(
                            q_b[1, :, :],
                            v_all[1, :, :],
                            acc3d[0, :, :],
                            transpose_B=True,
                            init=False,
                            kL0Size=KL0,
                        )
                        T.gemm_v0(
                            qt_all[1, :, :],
                            tail_l1,
                            acc3d[0, :, :],
                            transpose_B=True,
                            init=False,
                            kL0Size=KL0,
                        )
                        T.copy(
                            acc3d[0, :, :],
                            ws_s[cid, t * HSEG + V_BLOCK : t * HSEG + 2 * V_BLOCK, :],
                        )
                        T.set_cross_flag("FIX", F1A_BASE + t)
                    # ---- pass 2: PV accumulation, N-seg outer ----
                    # acc3d[ms] = ms-th M-slice's seg-s partial, accumulated
                    # across t (init on t==0; 3D-slot + init=False
                    # cross-block form). s is the OUTER loop: L0C's 128KB
                    # holds only 2 x [64,256] fp32 = both M-slices of ONE
                    # N-seg; each s pass re-reads its 256-col ws_kv band
                    # (per task 2*NI*128KB KV + 2*NI*64KB P — cheaper than
                    # the M-outer variant's full-width KV re-reads).
                    # F2 waits live in the s=0 pass ONLY (1:1 with V's
                    # pass-2 sets; the s=1 pass re-reads ws_p/ws_kv in C
                    # program order — both live until V's task k+1 pass 2,
                    # chain-fenced via F1A(task k+1) set only after this
                    # pass). t=0 hoisted: gemm init= must be a literal.
                    n0 = k * NI
                    slot0 = n0 % DEPTH
                    for s in range(N_SEG):
                        if s == 0:
                            T.wait_cross_flag(F2_BASE + 0)
                        T.copy(
                            ws_kv[cid, slot0 * BI : (slot0 + 1) * BI, s * DS : (s + 1) * DS],
                            v_all[s, :, :],
                        )
                        T.copy(ws_p[cid, 0:V_BLOCK, :], p_l1)
                        T.gemm_v0(p_l1, v_all[s, :, :], acc3d[0, :, :], init=True, kL0Size=KL0)
                        T.copy(ws_p[cid, V_BLOCK:HSEG, :], p_l1)
                        T.gemm_v0(p_l1, v_all[s, :, :], acc3d[1, :, :], init=True, kL0Size=KL0)
                        for t in T.serial(NI - 1):
                            n = k * NI + t + 1
                            slot = n % DEPTH
                            if s == 0:
                                T.wait_cross_flag(F2_BASE + t + 1)
                            T.copy(
                                ws_kv[
                                    cid,
                                    slot * BI : (slot + 1) * BI,
                                    s * DS : (s + 1) * DS,
                                ],
                                v_all[s, :, :],
                            )
                            T.copy(
                                ws_p[
                                    cid,
                                    (t + 1) * HSEG : (t + 1) * HSEG + V_BLOCK,
                                    :,
                                ],
                                p_l1,
                            )
                            T.gemm_v0(p_l1, v_all[s, :, :], acc3d[0, :, :], init=False, kL0Size=KL0)
                            T.copy(
                                ws_p[
                                    cid,
                                    (t + 1) * HSEG + V_BLOCK : (t + 1) * HSEG + HSEG,
                                    :,
                                ],
                                p_l1,
                            )
                            T.gemm_v0(p_l1, v_all[s, :, :], acc3d[1, :, :], init=False, kL0Size=KL0)
                        # seg-s band of both M-slices -> ws_o
                        T.copy(acc3d[0, :, :], ws_o[cid, 0:V_BLOCK, s * DS : (s + 1) * DS])
                        T.copy(acc3d[1, :, :], ws_o[cid, V_BLOCK:HSEG, s * DS : (s + 1) * DS])
                    # both N-seg bands of both M-slices flushed -> release V
                    T.set_cross_flag("FIX", F3_BASE + 0)
                    T.set_cross_flag("FIX", F3_BASE + 1)

            # ================= Vector stream =================
            with T.Scope("V"):
                if ntask > 0:
                    # ---- prologue: gather(0) (LEAD=1 lookahead:
                    #      8 x [16,576] batches/vid) ----
                    s_row0 = cid % seq_len
                    b0 = TokenIndices[s_row0, 0]
                    bos0 = Offsets[b0]
                    T.copy(Indices[s_row0, 0, 0:BI], indices_ub[0, :])
                    T.set_flag("mte2", "v", 0)
                    T.wait_flag("mte2", "v", 0)
                    # vectorized pre-clamp rows (int32 domain):
                    # rows = clamp(idx + bos, 0, S_kv - 1); clamp protects
                    # addressing only, contributions are masked
                    T.tile.add(rows_ub, indices_ub[0, :], bos0)
                    T.tile.clamp(rows_ub, rows_ub, 0, seq_len_kv - 1, BI)
                    T.set_flag("v", "s", 0)
                    T.wait_flag("v", "s", 0)
                    for b in range(NBATCH):
                        T.set_flag("mte3", "mte2", 0)
                        T.wait_flag("mte3", "mte2", 0)
                        for j in range(GKB):
                            kv_row = rows_ub[vid * HALF + b * GKB + j]
                            T.copy(KV[kv_row, 0, :], kv_stage[j, :])
                        T.set_flag("mte2", "mte3", 0)
                        T.wait_flag("mte2", "mte3", 0)
                        T.copy(
                            kv_stage,
                            ws_kv[cid, vid * HALF + b * GKB : vid * HALF + (b + 1) * GKB, :],
                        )
                    T.set_cross_flag("MTE3", F0_BASE + 0)

                    for k in T.serial(ntask):
                        pid = k * grid_size + cid
                        s_row = pid % seq_len
                        b_i = TokenIndices[s_row, 0]
                        s_i = TokenIndices[s_row, 1]
                        bos = Offsets[b_i]
                        # lookahead (task k+1) scalars — clamped memory-safe
                        pid_n = pid + grid_size
                        s_row_n = T.min(pid_n % seq_len, seq_len - 1)
                        b_n = TokenIndices[s_row_n, 0]
                        bos_n = Offsets[b_n]
                        # state init: m_fin floor = -BIGC/2 (below every
                        # finite practical score, above -BIGC; finite floor
                        # also avoids -inf - -inf NaN — see section header)
                        T.tile.fill(m_fin, -(BIGC * 0.5))
                        T.tile.fill(sumexp, 0.0)
                        # ---- pass 1: NI-block m_final reduction ----
                        # (each block's reduction is preceded by the
                        # gather(n+1) LEAD=1 lookahead)
                        for t in T.serial(NI):
                            n = k * NI + t
                            # ---- gather(n+1): lookahead, 8x[16-row] batches ----
                            if n + 1 < nblk:
                                gn = n + 1
                                nxt = t + 1 >= NI  # gn belongs to task k+1?
                                nslot = gn % DEPTH
                                islot = gn % IDXSLOTS
                                tg = (t + 1) % NI  # gn's block index in its task
                                s_row_g = T.if_then_else(nxt, s_row_n, s_row)
                                # indices_ub rewrite protection: prior V
                                # readers = post1(n-1)'s mask chain (same slot)
                                T.set_flag("v", "mte2", 0)
                                T.wait_flag("v", "mte2", 0)
                                T.copy(
                                    Indices[s_row_g, 0, tg * BI : (tg + 1) * BI],
                                    indices_ub[islot, :],
                                )
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                # rows_ub rewrite protection: prior S-pipe
                                # GetValue readers = gather(n)'s row loop
                                T.set_flag("s", "v", 0)
                                T.wait_flag("s", "v", 0)
                                if nxt:
                                    # GM scalars must stay in direct-read form
                                    # (select-wrapped scalars break the Adds
                                    # codegen)
                                    T.tile.add(rows_ub, indices_ub[islot, :], bos_n)
                                else:
                                    T.tile.add(rows_ub, indices_ub[islot, :], bos)
                                T.tile.clamp(rows_ub, rows_ub, 0, seq_len_kv - 1, BI)
                                T.set_flag("v", "s", 0)
                                T.wait_flag("v", "s", 0)
                                for b in range(NBATCH):
                                    # kv_stage WAR: drain the previous bulk
                                    # (previous batch here; the previous
                                    # gather's last bulk for b == 0)
                                    T.set_flag("mte3", "mte2", 0)
                                    T.wait_flag("mte3", "mte2", 0)
                                    for j in range(GKB):
                                        kv_row = rows_ub[vid * HALF + b * GKB + j]
                                        T.copy(KV[kv_row, 0, :], kv_stage[j, :])
                                    T.set_flag("mte2", "mte3", 0)
                                    T.wait_flag("mte2", "mte3", 0)
                                    T.copy(
                                        kv_stage,
                                        ws_kv[
                                            cid,
                                            nslot * BI + vid * HALF + b * GKB : nslot * BI + vid * HALF + (b + 1) * GKB,
                                            :,
                                        ],
                                    )
                                T.set_cross_flag("MTE3", F0_BASE + gn % 2)  # parity id (F0 note)
                            # post1(n): m_final step (pass 1 computes m
                            # ONLY — no online sumexp/alpha rescale here)
                            T.wait_cross_flag(F1A_BASE + t)
                            # s_ub MTE2 write vs prior V readers (the
                            # previous post's chain; pass2's loads below)
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.copy(
                                ws_s[
                                    cid,
                                    t * HSEG + vid * V_BLOCK : t * HSEG + (vid + 1) * V_BLOCK,
                                    :,
                                ],
                                s_ub,
                            )
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            # mask chain (group broadcast):
                            # mask = clamp01(min(idx + 1, s_i + 1 - idx));
                            # bias = BIGC * (1 - mask)
                            T.copy(indices_ub[n % IDXSLOTS, :], idx_f)  # int32 -> fp32
                            T.tile.add(t_ub, idx_f, T.float32(1.0))
                            T.tile.mul(m1_ub, idx_f, T.float32(-1.0))
                            T.tile.add(m1_ub, m1_ub, T.float32(s_i + 1))
                            T.tile.min(t_ub, t_ub, m1_ub)
                            T.tile.max(t_ub, t_ub, T.float32(0.0))
                            T.tile.min(t_ub, t_ub, T.float32(1.0))
                            # save this block's mask for pass 2 (pure V->V,
                            # no flags; avoids an Indices GM re-read — both
                            # ring slots are clobbered by lookahead gathers
                            # by the time pass 2 starts)
                            T.copy(t_ub, mask_mem[t, :])
                            T.tile.mul(m1_ub, t_ub, T.float32(-BIGC))
                            T.tile.add(m1_ub, m1_ub, T.float32(BIGC))
                            # s_masked = s * scale * mask - BIGC * (1 - mask)
                            T.tile.mul(s_ub, s_ub, sm_scale)
                            T.tile.broadcast(bcast_h, t_ub, axis=0)
                            for g in range(NGRP):
                                T.tile.mul(
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    bcast_h,
                                )
                            T.tile.broadcast(bcast_h, m1_ub, axis=0)
                            for g in range(NGRP):
                                T.tile.sub(
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    bcast_h,
                                )
                            # m_fin = max(m_fin, rowmax(s_masked));
                            # sumexp_i doubles as the rowmax scratch (its
                            # pass-2 rowsum use overwrites it — no hazard)
                            for g in range(NGRP):
                                T.reduce_max(
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    sumexp_i[g * EG : (g + 1) * EG],
                                    dim=-1,
                                )
                            T.tile.max(m_fin, m_fin, sumexp_i)
                        # ---- pass 2: P = exp(s_masked - m_fin) ----
                        # ws_s RE-READ per block. No cross-flag wait needed:
                        # ws_s[t] was published by F1A+t in pass 1, and C
                        # cannot overwrite it for task k+1 before consuming
                        # ALL F2 flags of task k (chain-fenced via the F2
                        # waits in the cube pass 2).
                        for t in T.serial(NI):
                            # s_ub MTE2 write vs prior V readers (the
                            # previous pass2 block's chain / exp)
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.copy(
                                ws_s[
                                    cid,
                                    t * HSEG + vid * V_BLOCK : t * HSEG + (vid + 1) * V_BLOCK,
                                    :,
                                ],
                                s_ub,
                            )
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            # restore this block's mask (saved in pass 1) +
                            # recompute the bias (2 vector ops — cheaper than
                            # a second [NI, BI] save array)
                            T.copy(mask_mem[t, :], t_ub)
                            T.tile.mul(m1_ub, t_ub, T.float32(-BIGC))
                            T.tile.add(m1_ub, m1_ub, T.float32(BIGC))
                            # s_masked = s * scale * mask - BIGC * (1 - mask)
                            T.tile.mul(s_ub, s_ub, sm_scale)
                            T.tile.broadcast(bcast_h, t_ub, axis=0)
                            for g in range(NGRP):
                                T.tile.mul(
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    bcast_h,
                                )
                            T.tile.broadcast(bcast_h, m1_ub, axis=0)
                            for g in range(NGRP):
                                T.tile.sub(
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    bcast_h,
                                )
                            # P = exp(s_masked - m_fin); masked-out entries
                            # are exp(-BIGC - m_fin) -> natural underflow 0
                            for g in range(NGRP):
                                T.tile.broadcast(bcast_h, m_fin[g * EG : (g + 1) * EG], axis=1)
                                T.tile.sub(
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    bcast_h,
                                )
                            T.tile.exp(s_ub, s_ub)
                            for g in range(NGRP):
                                T.reduce_sum(
                                    s_ub[g * EG : (g + 1) * EG, :],
                                    sumexp_i[g * EG : (g + 1) * EG],
                                    dim=-1,
                                )
                            T.tile.add(sumexp, sumexp, sumexp_i)
                            # p store: per-group INTERLEAVED cast + store (a
                            # single cast_g slot cannot batch the two loops —
                            # every store would read the final group's cast)
                            for g in range(NGRP):
                                T.set_flag("mte3", "v", 0)
                                T.wait_flag("mte3", "v", 0)
                                T.copy(s_ub[g * EG : (g + 1) * EG, :], cast_g)
                                T.set_flag("v", "mte3", 0)
                                T.wait_flag("v", "mte3", 0)
                                T.copy(
                                    cast_g,
                                    ws_p[
                                        cid,
                                        t * HSEG + vid * V_BLOCK + g * EG : t * HSEG + vid * V_BLOCK + (g + 1) * EG,
                                        :,
                                    ],
                                )
                            T.set_cross_flag("MTE3", F2_BASE + t)
                        # ---- epilogue (task tail): ws_o read, normalize,
                        #      Output + Lse ----
                        # BOTH F3 ids: (a) this vid's ws_o rows span both
                        # N-seg bands, (b) ws_kv ring safety — the next
                        # task's gathers may only fire past C's COMPLETE
                        # pass 2 (both bands, both M-slices flushed).
                        T.wait_cross_flag(F3_BASE + 0)
                        T.wait_cross_flag(F3_BASE + 1)
                        # Empty-row guard: all-invalid rows have sumexp == 0
                        # (masked scores -BIGC underflow against the m_fin
                        # floor); divide by max(sumexp, 1) to emit zeros
                        # instead of 0/0 NaN. Non-empty rows have sumexp >= 1
                        # (global-max slot -> exp(0) = 1), so the clamp is an
                        # exact no-op. sumexp_i is dead past pass 2 (rowsum
                        # scratch) — reuse it as the denominator buffer. Lse
                        # below still uses raw sumexp (ln(0) = -inf).
                        T.tile.max(sumexp_i, sumexp, T.float32(1.0))
                        for s in range(N_SEG):
                            # s_ub MTE2 write vs prior V readers (pass-2
                            # exp/store chain)
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.copy(
                                ws_o[
                                    cid,
                                    vid * V_BLOCK : (vid + 1) * V_BLOCK,
                                    s * DS : (s + 1) * DS,
                                ],
                                s_ub,
                            )
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            # per-row /sumexp (per-task 1/8 frequency, not
                            # worth a broadcast chain)
                            for h in range(V_BLOCK):
                                T.tile.div(s_ub[h, :], s_ub[h, :], sumexp_i[h])
                            for g in range(NGRP):
                                T.set_flag("mte3", "v", 0)
                                T.wait_flag("mte3", "v", 0)
                                T.copy(s_ub[g * EG : (g + 1) * EG, :], cast_g)
                                T.set_flag("v", "mte3", 0)
                                T.wait_flag("v", "mte3", 0)
                                T.copy(
                                    cast_g,
                                    Output[
                                        s_row,
                                        vid * V_BLOCK + g * EG : vid * V_BLOCK + (g + 1) * EG,
                                        s * DS : (s + 1) * DS,
                                    ],
                                )
                        # Lse = ln(sumexp) + m_fin (global-max form)
                        T.tile.ln(sumexp, sumexp)
                        T.tile.add(sumexp, sumexp, m_fin)
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.copy(sumexp, Lse[s_row, vid * V_BLOCK : (vid + 1) * V_BLOCK])
                        # drain the Lse store (MTE3 reads sumexp) BEFORE the
                        # next task's V-pipe state fills: without this pair
                        # fill(sumexp, 0) can overwrite the UB data before
                        # the DMA reads it.
                        T.set_flag("mte3", "v", 0)
                        T.wait_flag("mte3", "v", 0)

    return main
