"""Expert v2: BF16 decode attention with a shared prefix and two per-beam tokens.

Adapted from xa's tl_xattn_v3q. See README.md for the algorithm and input shapes.
"""

import argparse

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout
import torch

tilelang.disable_cache()

# ===========================================================================
# Lock declarations
# ===========================================================================

# Cube scope

# Vector scope: shared load/store buffers across softmax, O_acc and the
# inlined unshared+merge tail.

# The unshared+merge is INLINED into vec_o_acc's last-KV-block tail, so it
# reuses phase-1's buffers BY CATEGORY -- which makes the lock rule ("aliased
# DMA-touched buffers MUST share the same lock") hold by construction:
#   loads  (MTE2->V)  -> pinned inside the LD_BUF region [81920,147456), LD_BUF
#   store  (V->MTE3)  -> pinned inside the ST_ON  region [32768, 49152), ST_ON
#   scratch (V only, no DMA) -> may alias freely, in two classes:
#     * "safe"       [0,32768)  -- score_local's region; NO DMA queue ever
#                    touches it, so a buffer there may stay live across a
#                    REL V LD_BUF.
#     * "under-lock" [106496,139264) -- aliases ld_o_acc/ld_o_partial, which
#                    ARE MTE2-written under LD_BUF, so these are touched ONLY
#                    while V holds LD_BUF and MTE2 cannot clobber them.

# ===========================================================================
# Constants
# ===========================================================================
NUM_CORES = 24
DIM = 128

TILE_Q_L2 = 256
TILE_KV_L1 = 1024
TILE_KV_L2 = 1024

TILE_M = 128
TILE_K = 128

TILE_Q_UB = 8
TILE_Q_ACC = 64

NUM_STAGES = 2

M_ITERS = TILE_Q_L2 // TILE_M
N_ITERS = TILE_KV_L1 // TILE_K
K_ITERS = TILE_KV_L1 // TILE_K

HALF_Q = TILE_Q_L2 // 2
SOFTMAX_STRIPS = HALF_Q // TILE_Q_UB
O_ACC_STRIPS = HALF_Q // TILE_Q_ACC
SOFTMAX_STRIPS_PER_ACC = TILE_Q_ACC // TILE_Q_UB

GROUP_SIZE = 4
KV_HEADS = 8

BEAMS_PER_M = TILE_M // GROUP_SIZE  # 32
BEAMS_PER_Q_BLOCK = TILE_Q_L2 // GROUP_SIZE  # 64
GROUPS_PER_STRIP = TILE_Q_ACC // BEAMS_PER_M  # 2

SEM_CUBE = 0
SEM_VEC = 1

# FP32 scalar broadcast uses 8 elements per 32-byte block.
ELEM_PER_BLK = 8
SFM_WORKSPACE_BYTES = 8384  # Reduction scratch for [8,1024], pinned inside ST_BUF.
ROW_EXPAND_VECTOR_ELEMS = 64  # one latest-codegen fp32 call = 256 bytes per row

# Merge tile = one 32-row HALF of a vec_o_acc strip. In the group-major layout
# st_o_acc rows are [gi*BEAMS_PER_M + beam], so a half is exactly ONE group-head
# x BEAMS_PER_M DISTINCT beams -> K_u/V_u load straight as [32,128] with NO
# group replication (v3f needed 4x replication: its tile spanned 4 group-heads
# of 8 beams each).
MERGE_ROWS = BEAMS_PER_M  # 32

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False,
}


# ===========================================================================
# Layer 2: GEMM macros (Cube scope) -- identical to tl_stream_shared_only_v3.
# ===========================================================================


@T.macro(hygienic=False)
def gemm_qk(request, kv_head, beam_base, kv_offset, stage, K_3d, Q_5d, a_l1, cid, kv_l1, l0a, l0b, l0c, ws_sp):
    T.wait_flag("MTE1", "MTE2", 2)
    for ni in T.serial(N_ITERS):
        T.copy(K_3d[request, kv_offset + ni * TILE_K : kv_offset + (ni + 1) * TILE_K, kv_head, :], kv_l1[ni, :, :])
    T.set_flag("MTE2", "MTE1", 2)

    T.wait_flag("MTE2", "MTE1", 2)
    for mi in T.serial(M_ITERS):
        mi_side = mi % 2
        T.wait_flag("MTE1", "MTE2", 0 + mi_side)
        for g in T.serial(GROUP_SIZE):
            T.copy(
                Q_5d[request, beam_base + mi * BEAMS_PER_M : beam_base + (mi + 1) * BEAMS_PER_M, kv_head, g, :],
                a_l1[mi_side, g * BEAMS_PER_M : (g + 1) * BEAMS_PER_M, :],
            )
        T.set_flag("MTE2", "MTE1", 0 + mi_side)

        T.wait_flag("MTE2", "MTE1", 0 + mi_side)
        for ni in T.serial(N_ITERS):
            side = ni % 2
            T.wait_flag("M", "MTE1", 3 + side)
            if ni < 2:
                T.copy(a_l1[mi_side, :, :], l0a[side, :, :])
            T.copy(kv_l1[ni, :, :], l0b[side, :, :], transpose=True)
            T.set_flag("MTE1", "M", 3 + side)

            T.wait_flag("MTE1", "M", 3 + side)
            T.wait_flag("FIX", "M", 5 + side)
            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
            T.set_flag("M", "MTE1", 3 + side)
            T.set_flag("M", "FIX", 5 + side)
            T.wait_flag("M", "FIX", 5 + side)
            T.copy(l0c[side, :, :], ws_sp[cid, stage, mi * TILE_M : (mi + 1) * TILE_M, ni * TILE_K : (ni + 1) * TILE_K])
            T.set_flag("FIX", "M", 5 + side)
        T.set_flag("MTE1", "MTE2", 0 + mi_side)
    T.set_flag("MTE1", "MTE2", 2)


@T.macro(hygienic=False)
def gemm_pv(request, kv_head, kv_offset, stage, V_3d, a_l1, cid, kv_l1, l0a, l0b, l0c, ws_op, ws_sp):
    T.wait_flag("MTE1", "MTE2", 2)
    for ki in T.serial(K_ITERS):
        T.copy(V_3d[request, kv_offset + ki * TILE_K : kv_offset + (ki + 1) * TILE_K, kv_head, :], kv_l1[ki, :, :])
    T.set_flag("MTE2", "MTE1", 2)

    T.wait_flag("MTE2", "MTE1", 2)
    for mi in T.serial(M_ITERS):
        c_side = mi % 2
        T.wait_flag("FIX", "M", 5 + c_side)
        for ki in T.serial(K_ITERS):
            side = ki % 2
            T.wait_flag("MTE1", "MTE2", 0 + side)
            T.copy(ws_sp[cid, stage, mi * TILE_M : (mi + 1) * TILE_M, ki * TILE_K : (ki + 1) * TILE_K], a_l1[side, :, :])
            T.set_flag("MTE2", "MTE1", 0 + side)

            T.wait_flag("M", "MTE1", 3 + side)
            T.wait_flag("MTE2", "MTE1", 0 + side)
            T.copy(a_l1[side, :, :], l0a[side, :, :])
            T.set_flag("MTE1", "MTE2", 0 + side)
            T.copy(kv_l1[ki, :, :], l0b[side, :, :])
            T.set_flag("MTE1", "M", 3 + side)

            T.wait_flag("MTE1", "M", 3 + side)
            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[c_side, :, :], init=(ki == 0))
            T.set_flag("M", "MTE1", 3 + side)
        T.set_flag("M", "FIX", 5 + c_side)
        T.wait_flag("M", "FIX", 5 + c_side)
        T.copy(l0c[c_side, :, :], ws_op[cid, stage, mi * TILE_M : (mi + 1) * TILE_M, :])
        T.set_flag("FIX", "M", 5 + c_side)
    T.set_flag("MTE1", "MTE2", 2)


# ===========================================================================
# Layer 3: Vector macros -- SoftmaxFlashV2 + row_expand_*.
# ===========================================================================


# ---------------------------------------------------------------------------
# vec_softmax: TASK 2 -- S -> P via SoftmaxFlashV2, in-place, early release.
# expMax = exp(old_max - new_max) is written straight into scale_ring (the
# O_acc rescale factor).
# ---------------------------------------------------------------------------
@T.macro(hygienic=False)
def softmax_composed_update(r, q_ring, l_panel, m_stats_ring, score_local, sfm_brcb, sfm_row, sfm_tmp):
    T.pipe_barrier("v")
    T.reduce_max(score_local, sfm_row, dim=-1, tmp=sfm_tmp)
    T.pipe_barrier("v")
    T.tile.brcb_experiment(sfm_brcb, sfm_row, 1, 1, 8)
    T.pipe_barrier("v")
    T.tile.max(sfm_brcb, sfm_brcb, m_stats_ring[q_ring, r * TILE_Q_UB : (r + 1) * TILE_Q_UB, :])
    T.pipe_barrier("v")
    T.copy(sfm_brcb, m_stats_ring[q_ring, r * TILE_Q_UB : (r + 1) * TILE_Q_UB, :])
    for chunk in T.unroll(TILE_KV_L2 // ROW_EXPAND_VECTOR_ELEMS):
        col = chunk * ROW_EXPAND_VECTOR_ELEMS
        T.tile.row_expand_sub_experiment(
            score_local[:, col : col + ROW_EXPAND_VECTOR_ELEMS], score_local[:, col : col + ROW_EXPAND_VECTOR_ELEMS], sfm_brcb
        )
    T.pipe_barrier("v")
    T.tile.exp(score_local, score_local)
    T.pipe_barrier("v")
    T.reduce_sum(score_local, l_panel[r * TILE_Q_UB : (r + 1) * TILE_Q_UB, :], dim=-1, tmp=sfm_tmp)


@T.macro(hygienic=False)
def finish_softmax_stats(acc_strip, q_ring, scale_bank, l_panel, l_stats_ring, m_prev, m_stats_ring, sum_brcb):
    T.tile.sub(
        scale_bank,
        m_prev[acc_strip * TILE_Q_ACC : (acc_strip + 1) * TILE_Q_ACC, :],
        m_stats_ring[q_ring, acc_strip * TILE_Q_ACC : (acc_strip + 1) * TILE_Q_ACC, :],
    )
    T.pipe_barrier("v")
    T.tile.exp(scale_bank, scale_bank)
    T.tile.brcb_experiment(sum_brcb, l_panel[acc_strip * TILE_Q_ACC : (acc_strip + 1) * TILE_Q_ACC, :], TILE_Q_ACC // 8, 1, 8)
    T.pipe_barrier("v")
    T.tile.mul(
        l_stats_ring[q_ring, acc_strip * TILE_Q_ACC : (acc_strip + 1) * TILE_Q_ACC, :],
        l_stats_ring[q_ring, acc_strip * TILE_Q_ACC : (acc_strip + 1) * TILE_Q_ACC, :],
        scale_bank,
    )
    T.pipe_barrier("v")
    T.tile.add(
        l_stats_ring[q_ring, acc_strip * TILE_Q_ACC : (acc_strip + 1) * TILE_Q_ACC, :],
        l_stats_ring[q_ring, acc_strip * TILE_Q_ACC : (acc_strip + 1) * TILE_Q_ACC, :],
        sum_brcb,
    )
    T.pipe_barrier("v")


@T.macro(hygienic=False)
def vec_softmax(
    step,
    my_start,
    num_q_blocks,
    num_kv_blocks,
    num_q_stages,
    sm_scale,
    cid,
    l_panel,
    l_stats_ring,
    ld_score,
    m_prev,
    m_stats_ring,
    scale_s0_r0,
    scale_s0_r1,
    scale_s1_r0,
    scale_s1_r1,
    score_local,
    sfm_brcb,
    sfm_row,
    sfm_tmp,
    st_prob,
    sum_brcb,
    vid,
    ws_sp,
):
    stage = step % NUM_STAGES
    global_q = my_start + step // num_kv_blocks
    kv_idx = step % num_kv_blocks
    q_ring = global_q % num_q_stages
    if kv_idx == 0:
        T.tile.fill(m_stats_ring[q_ring, :, :], -(2**30))
        T.tile.fill(l_stats_ring[q_ring, :, :], 0.0)
        T.pipe_barrier("v")
    T.copy(m_stats_ring[q_ring, :, :], m_prev)
    for r in T.serial(SOFTMAX_STRIPS):
        row = vid * HALF_Q + r * TILE_Q_UB
        T.wait_flag("V", "MTE2", 0)
        T.copy(ws_sp[cid, stage, row : row + TILE_Q_UB, :], ld_score)
        T.set_flag("MTE2", "V", 0)
        T.wait_flag("MTE2", "V", 0)
        T.pipe_barrier("v")
        T.copy(ld_score, score_local)
        T.set_flag("V", "MTE2", 0)
        T.pipe_barrier("v")
        T.tile.mul(score_local, score_local, sm_scale)
        # Scratch occupies ST_BUF's unused upper half; own it before reducing.
        T.wait_flag("MTE3", "V", 1)
        softmax_composed_update(
            r,
            q_ring,
            l_panel=l_panel,
            m_stats_ring=m_stats_ring,
            score_local=score_local,
            sfm_brcb=sfm_brcb,
            sfm_row=sfm_row,
            sfm_tmp=sfm_tmp,
        )
        T.copy(score_local, st_prob)
        T.set_flag("V", "MTE3", 1)
        T.wait_flag("V", "MTE3", 1)
        T.copy(st_prob, ws_sp[cid, stage, row : row + TILE_Q_UB, :])
        T.set_flag("MTE3", "V", 1)
    T.set_cross_flag("MTE3", SEM_CUBE)
    if stage == 0:
        finish_softmax_stats(
            0, q_ring, scale_s0_r0, l_panel=l_panel, l_stats_ring=l_stats_ring, m_prev=m_prev, m_stats_ring=m_stats_ring, sum_brcb=sum_brcb
        )
        finish_softmax_stats(
            1, q_ring, scale_s0_r1, l_panel=l_panel, l_stats_ring=l_stats_ring, m_prev=m_prev, m_stats_ring=m_stats_ring, sum_brcb=sum_brcb
        )
    else:
        finish_softmax_stats(
            0, q_ring, scale_s1_r0, l_panel=l_panel, l_stats_ring=l_stats_ring, m_prev=m_prev, m_stats_ring=m_stats_ring, sum_brcb=sum_brcb
        )
        finish_softmax_stats(
            1, q_ring, scale_s1_r1, l_panel=l_panel, l_stats_ring=l_stats_ring, m_prev=m_prev, m_stats_ring=m_stats_ring, sum_brcb=sum_brcb
        )


@T.macro(hygienic=False)
def merge_half(
    request,
    kv_head,
    beam_start,
    ghead,
    gi,
    sr,
    q_ring,
    bpr,
    decode_step,
    sm_scale,
    preload,
    next_ghead,
    r,
    kv_once,
    O_5d,
    Q_5d,
    cor_s_b,
    cor_u_b,
    gl_mrg_b,
    gl_u_b,
    gm_mrg_b,
    gm_u,
    gm_u_b,
    ku_ld,
    kv_f32,
    kv_f32_3,
    kv_f32_flat,
    l_stats_ring,
    m_stats_ring,
    o_st,
    o_u,
    os_half,
    q_bf,
    q_ld,
    score_buf,
    score_flat,
    uk_4d,
    ured,
    uv_4d,
    vu_ld,
    vu_ld_flat,
    w_blk,
    w_blk_flat,
    w_u,
):
    token_start = request * bpr + beam_start
    T.pipe_barrier("v")  # Prior output cast must stop reading reused safe scratch.

    # D2: K_u/V_u are invariant in r as well as gi (D1) -- beam_start has no r
    # term either. When kv_once (num_kv_blocks == 1) they therefore only need
    # loading on the FIRST strip of the step: ld_o_acc, the sole other writer of
    # their [81920,114688) window, is DEAD in that regime (kv_idx = old_step % 1
    # is always 0, so the branch that loads it cannot execute), and ld_o_partial
    # / the merge scratch live above 114688. So the r=0 copy survives into r=1
    # untouched -> loaded 1x per step instead of 2x (and 4x in v3p).
    #   The two regimes are written as NESTED PARSE-TIME `if` STATEMENTS on the
    # Python bool kv_once, NOT as `cond = (r == 0) if kv_once else True`: a
    # ternary is parsed as a doc.IfExp and evaluated EAGERLY by TVMScript, so the
    # `r == 0` operand (an EqualOp proxy, not a plain bool) makes it raise
    # "Cannot use and/or/not operator to Expr". An `if` STATEMENT on a Python
    # bool folds at parse time and is safe; only the inner `if r == 0` survives
    # as a TIR branch.

    # --- LEADING MTE2 SLOT --------------------------------------------------
    # D1: K_u/V_u are indexed by (token_start, kv_head, t) ONLY. token_start =
    # request*bpr + beam_start and beam_start = local_q*BEAMS_PER_Q_BLOCK +
    # vid*BEAMS_PER_M depend on NEITHER r NOR gi -- only `ghead` (= r*2+gi)
    # varies across the 4 merge_half calls of a step, and ghead feeds ONLY the Q
    # load. So all 4 calls were loading BYTE-IDENTICAL K_u and V_u: a 4x
    # redundant DMA on the measured-exposed path. Hoisting them to the gi=0 call
    # (preload=True) halves the traffic; both halves then read the same resident
    # copy. (Hoisting out of the r loop too would quarter it, but ku_ld/vu_ld
    # alias ld_o_acc/ld_o_partial, which every r iteration re-loads, and no other
    # 32 KB of UB is free -- see the pin map.)
    T.wait_flag("V", "MTE2", 0)
    if preload:
        if kv_once:
            if r == 0:
                for t in range(decode_step):
                    T.copy(uk_4d[token_start : token_start + MERGE_ROWS, kv_head, t, :], ku_ld[t, :, :])
        else:
            for t in range(decode_step):
                T.copy(uk_4d[token_start : token_start + MERGE_ROWS, kv_head, t, :], ku_ld[t, :, :])
        T.copy(Q_5d[request, beam_start : beam_start + MERGE_ROWS, kv_head, ghead, :], q_ld)
    T.set_flag("MTE2", "V", 0)

    # --- PHASE A: compute the per-key scaled scores (Q + K_u resident) -------
    T.wait_flag("MTE2", "V", 0)
    T.copy(q_ld, q_bf)
    for t in range(decode_step):
        T.copy(ku_ld[t, :, :], kv_f32)
        T.pipe_barrier("v")
        T.tile.mul(kv_f32, kv_f32, q_bf)
        T.pipe_barrier("v")
        T.reduce_sum(kv_f32, score_buf[t, :, :], dim=-1, tmp=ured)
        T.pipe_barrier("v")
    T.set_flag("V", "MTE2", 0)

    # C2: scale the SCORES, not q. v3n folded sm_scale into q_bf to save
    # (decode_step-1) ops -- right under an op-count model, BACKWARDS under an
    # element model: that mul is 4096 elements. score_flat is a 2-D alias over
    # ALL of score_buf, so ONE mul over decode_step*MERGE_ROWS = 64 elements
    # replaces it. Exactly equivalent: (sm_scale*q).K == sm_scale*(q.K).
    # Safe-region scratch -> outside the lock, so it overlaps the next DMA.
    T.tile.mul(score_flat, score_flat, sm_scale)

    # --- MID MTE2 SLOT: everything that can hide under the scalar chain ------
    # The ablation says a DMA issued HERE costs ~13x less than the same bytes
    # issued in the leading slot (V_u: 2.4% of merge; Q+K_u: 21.3%) -- because
    # the C3 scalar chain below runs on V while this MTE2 burst is in flight,
    # whereas the leading slot has NOTHING to overlap with. So load here
    # everything not needed by PHASE A:
    #   * V_u  -- only the accumulation (below) reads it, and it is gi-invariant
    #             (D1), so gi=0 loads it once for both halves.
    #   * the NEXT half's Q -- q_ld is dead the moment PHASE A's cast to q_bf
    #     retires (inside the V section that just closed), so MTE2 may refill it
    #     now. gi=1's leading slot is then EMPTY and its Q DMA is fully hidden.
    T.wait_flag("V", "MTE2", 0)
    if preload:
        if kv_once:
            if r == 0:
                for t in range(decode_step):
                    T.copy(uv_4d[token_start : token_start + MERGE_ROWS, kv_head, t, :], vu_ld[t, :, :])
        else:
            for t in range(decode_step):
                T.copy(uv_4d[token_start : token_start + MERGE_ROWS, kv_head, t, :], vu_ld[t, :, :])
    if next_ghead is not None:
        T.copy(Q_5d[request, beam_start : beam_start + MERGE_ROWS, kv_head, next_ghead, :], q_ld)
    T.set_flag("MTE2", "V", 0)

    # ---- C3: ALL the small math now runs HERE, while the V_u DMA above is
    # still in flight, and BEFORE the o_u accumulation -- so the final scale
    # a_u can be folded into the per-t weights on [32,8] instead of costing a
    # [32,128] row_expand_mul in phase C. Safe-region scratch only -> no lock.
    #
    # gm_u = max_t s_t
    T.pipe_barrier("v")
    T.copy(score_buf[0, :, :], gm_u)
    for t in range(1, decode_step):
        T.pipe_barrier("v")
        T.tile.max(gm_u, gm_u, score_buf[t, :, :])

    # w_t = exp(s_t - gm_u), kept as genuine 8-lane blocks; gl_u = sum_t w_t
    # accumulated in 8-lane layout straight off the broadcast (all lanes equal),
    # so gl_u_b needs no broadcast below.
    for t in range(decode_step):
        T.pipe_barrier("v")
        T.tile.sub(w_u, score_buf[t, :, :], gm_u)
        T.pipe_barrier("v")
        T.tile.exp(w_u, w_u)
        T.pipe_barrier("v")
        T.tile.brcb_experiment(w_blk[t, :, :], w_u, MERGE_ROWS // 8, 1, 8)
        T.pipe_barrier("v")
        if t == 0:
            T.copy(w_blk[0, :, :], gl_u_b)
        else:
            T.tile.add(gl_u_b, gl_u_b, w_blk[t, :, :])

    # --- PHASE C: LSE merge, END-TO-END IN 8-LANE BLOCK LAYOUT (safe-region
    #     scratch only; load lock released so MTE2 can prefetch the next
    #     half's Q/K_u under it) ---------------------------------------------
    #
    # m_stats_ring/l_stats_ring arrive from softmax_flash_v2 as GENUINE 8-lane
    # broadcast blocks -- ALL 8 LANES HOLD THE ROW VALUE. (AscendC
    # softmax_flashv2_basic_block_impl.h fp32 path: Brcb() broadcasts the row
    # max/sum, then every update is elementwise over reduceSize=M*8 against an
    # all-lanes-equal inMax/inExpSum, so the property is inductive; the base
    # case is vec_softmax's T.tile.fill over all 8 lanes. Verified on device: a
    # reduce_min-vs-reduce_max differential over the lanes is a no-op.)
    #
    # So there is no reason to leave block layout at all. v3m collapsed the
    # blocks to [32,1] with 2 reduces, did the merge on scalars, then broadcast
    # back to [32,8] 3x because row_expand_* REQUIRES an 8-lane src1. Staying in
    # [32,8] end-to-end deletes 2 copies + 2 reduces + 3 broadcasts. The math is
    # elementwise, so it just runs on all 8 lanes at once -- and small vector ops
    # are issue-overhead-bound, not element-bound, so [32,8] costs the same as
    # [32,1]. The rings are read directly as src (never written).
    T.pipe_barrier("v")
    T.tile.brcb_experiment(gm_u_b, gm_u, MERGE_ROWS // 8, 1, 8)
    T.pipe_barrier("v")
    T.tile.max(gm_mrg_b, m_stats_ring[q_ring, sr : sr + MERGE_ROWS, :], gm_u_b)
    T.pipe_barrier("v")
    T.tile.sub(cor_s_b, m_stats_ring[q_ring, sr : sr + MERGE_ROWS, :], gm_mrg_b)
    T.tile.sub(cor_u_b, gm_u_b, gm_mrg_b)
    T.pipe_barrier("v")
    T.tile.exp(cor_s_b, cor_s_b)
    T.tile.exp(cor_u_b, cor_u_b)
    T.pipe_barrier("v")
    # gm_mrg_b is DEAD from here on -> gl_mrg_b is pinned to its address (both
    # are V-only safe scratch with DISJOINT lifetimes: the write below is the
    # first gl_mrg_b access and strictly follows the last gm_mrg_b read above,
    # in V-pipe program order, with no DMA queue touching either).
    T.tile.mul(gl_mrg_b, cor_s_b, l_stats_ring[q_ring, sr : sr + MERGE_ROWS, :])
    T.pipe_barrier("v")
    # gl_mrg += gl_u*cor_u -- mul_add_dst fuses v3m's separate mul+add (2 -> 1).
    T.tile.mul_add_dst(gl_mrg_b, gl_u_b, cor_u_b)
    T.pipe_barrier("v")

    # C1: sink the normalization into the correction factors ON [32,8].
    #   O = (O_s*cor_s + O_u*cor_u)/gl_mrg == O_s*(cor_s/gl_mrg) + O_u*(cor_u/gl_mrg)
    # v3n spent a [32,128] row_expand_DIV (4096 elements, and a divide) to
    # normalize at the end. Doing the two divides on [32,8] instead costs 2
    # SMALL ops and deletes one BIG op -- and the surviving big ops are all
    # multiplies. gl_mrg is a sum of exps -> strictly > 0, so the divide is
    # always safe; only float re-association changes.
    # In-place over cor_s_b/cor_u_b: they have no further use as corrections.
    T.tile.div(cor_s_b, cor_s_b, gl_mrg_b)  # a_s = cor_s/gl_mrg
    T.tile.div(cor_u_b, cor_u_b, gl_mrg_b)  # a_u = cor_u/gl_mrg

    # C3: fold a_u into every per-t weight ON [32,8]. decode_step tiny muls buy
    # the deletion of a [32,128] row_expand_mul(o_u, o_u, a_u) below.
    T.pipe_barrier("v")
    for t in range(decode_step):
        T.tile.mul(w_blk[t, :, :], w_blk[t, :, :], cor_u_b)
    T.pipe_barrier("v")

    # ---- o_u = sum_t (w_t*a_u) * V_u[t] -- BIG ops ONLY -------------------
    # C4: all decode_step t's as ONE double-width op. kv_ld / kv_f32 / w_blk are
    # all [decode_step, MERGE_ROWS, *], so their flat 2-D views index (t, beam)
    # in the SAME row order (row j == t=j//MERGE_ROWS, beam=j%MERGE_ROWS) -> one
    # cast + one row_expand over decode_step*MERGE_ROWS rows replaces decode_step
    # of each, at an IDENTICAL element count. row_expand's src1 stays a genuine
    # 8-lane block (w_blk_flat is [decode_step*MERGE_ROWS, ELEM_PER_BLK]).
    T.wait_flag("MTE2", "V", 0)
    T.copy(vu_ld_flat, kv_f32_flat)
    T.pipe_barrier("v")
    for vu_mul_chunk in T.serial(DIM // ROW_EXPAND_VECTOR_ELEMS):
        vu_mul_col = vu_mul_chunk * ROW_EXPAND_VECTOR_ELEMS
        T.tile.row_expand_mul_experiment(
            kv_f32_flat[:, vu_mul_col : vu_mul_col + ROW_EXPAND_VECTOR_ELEMS],
            kv_f32_flat[:, vu_mul_col : vu_mul_col + ROW_EXPAND_VECTOR_ELEMS],
            w_blk_flat,
        )
    T.pipe_barrier("v")
    T.tile.add(o_u, kv_f32_3[0, :, :], kv_f32_3[1, :, :])
    for t in range(2, decode_step):
        T.pipe_barrier("v")
        T.tile.add(o_u, o_u, kv_f32_3[t, :, :])
    T.set_flag("V", "MTE2", 0)

    # O = O_s*a_s + O_u*a_u -- o_u ALREADY carries a_u, so this is 2 big ops.
    # Every row_expand_* src1 is a genuine 8-lane block by construction.
    for os_mul_chunk in T.serial(DIM // ROW_EXPAND_VECTOR_ELEMS):
        os_mul_col = os_mul_chunk * ROW_EXPAND_VECTOR_ELEMS
        T.tile.row_expand_mul_experiment(
            os_half[gi, :, os_mul_col : os_mul_col + ROW_EXPAND_VECTOR_ELEMS],
            os_half[gi, :, os_mul_col : os_mul_col + ROW_EXPAND_VECTOR_ELEMS],
            cor_s_b,
        )
    T.pipe_barrier("v")
    T.tile.add(o_u, o_u, os_half[gi, :, :])
    T.pipe_barrier("v")

    # --- store (V -> MTE3) --------------------------------------------------
    T.wait_flag("MTE3", "V", 2 + gi)
    T.copy(o_u, o_st[gi, :, :])
    T.set_flag("V", "MTE3", 2 + gi)
    T.wait_flag("V", "MTE3", 2 + gi)
    T.copy(o_st[gi, :, :], O_5d[request, beam_start : beam_start + MERGE_ROWS, kv_head, ghead, :])
    T.set_flag("MTE3", "V", 2 + gi)


# ---------------------------------------------------------------------------
# vec_o_acc: TASK 1 -- O_acc update (consume O_partial). At the LAST KV block
# the unshared+merge is inlined per 32-row half.
# ---------------------------------------------------------------------------
@T.macro(hygienic=False)
def vec_o_acc(
    step,
    my_start,
    num_q_blocks,
    num_kv_blocks,
    num_q_stages,
    bpr,
    decode_step,
    sm_scale,
    kv_once,
    O_5d,
    Q_5d,
    cid,
    cor_s_b,
    cor_u_b,
    gl_mrg_b,
    gl_u_b,
    gm_mrg_b,
    gm_u,
    gm_u_b,
    ku_ld,
    kv_f32,
    kv_f32_3,
    kv_f32_flat,
    l_stats_ring,
    ld_o_acc,
    ld_o_partial,
    m_stats_ring,
    o_st,
    o_u,
    os_half,
    q_bf,
    q_ld,
    scale_s0_r0,
    scale_s0_r1,
    scale_s1_r0,
    scale_s1_r1,
    score_buf,
    score_flat,
    st_o_acc,
    uk_4d,
    ured,
    uv_4d,
    vid,
    vu_ld,
    vu_ld_flat,
    w_blk,
    w_blk_flat,
    w_u,
    ws_oa,
    ws_op,
):
    old_step = step - NUM_STAGES
    stage = step % NUM_STAGES
    old_q_task = old_step // num_kv_blocks
    old_global_q = my_start + old_q_task
    bh = old_global_q // num_q_blocks
    local_q = old_global_q % num_q_blocks
    kv_idx = old_step % num_kv_blocks
    q_ring = old_global_q % num_q_stages

    request = bh // KV_HEADS
    kv_head = bh % KV_HEADS

    for r in T.serial(O_ACC_STRIPS):
        row = vid * HALF_Q + r * TILE_Q_ACC

        if kv_idx == 0:
            T.wait_flag("V", "MTE2", 0)
            T.copy(ws_op[cid, stage, row : row + TILE_Q_ACC, :], ld_o_partial)
            T.set_flag("MTE2", "V", 0)

            T.wait_flag("MTE3", "V", 1)
            T.wait_flag("MTE2", "V", 0)
            T.copy(ld_o_partial, st_o_acc)
            T.set_flag("V", "MTE2", 0)
        else:
            T.wait_flag("V", "MTE2", 0)
            T.copy(ws_oa[cid, row : row + TILE_Q_ACC, :], ld_o_acc)
            T.copy(ws_op[cid, stage, row : row + TILE_Q_ACC, :], ld_o_partial)
            T.set_flag("MTE2", "V", 0)

            T.wait_flag("MTE2", "V", 0)
            T.wait_flag("MTE3", "V", 1)
            if stage == 0:
                if r == 0:
                    for o_mul_s0_r0_chunk in T.serial(DIM // ROW_EXPAND_VECTOR_ELEMS):
                        o_mul_s0_r0_col = o_mul_s0_r0_chunk * ROW_EXPAND_VECTOR_ELEMS
                        T.tile.row_expand_mul_experiment(
                            st_o_acc[:, o_mul_s0_r0_col : o_mul_s0_r0_col + ROW_EXPAND_VECTOR_ELEMS],
                            ld_o_acc[:, o_mul_s0_r0_col : o_mul_s0_r0_col + ROW_EXPAND_VECTOR_ELEMS],
                            scale_s0_r0,
                        )
                else:
                    for o_mul_s0_r1_chunk in T.serial(DIM // ROW_EXPAND_VECTOR_ELEMS):
                        o_mul_s0_r1_col = o_mul_s0_r1_chunk * ROW_EXPAND_VECTOR_ELEMS
                        T.tile.row_expand_mul_experiment(
                            st_o_acc[:, o_mul_s0_r1_col : o_mul_s0_r1_col + ROW_EXPAND_VECTOR_ELEMS],
                            ld_o_acc[:, o_mul_s0_r1_col : o_mul_s0_r1_col + ROW_EXPAND_VECTOR_ELEMS],
                            scale_s0_r1,
                        )
            else:
                if r == 0:
                    for o_mul_s1_r0_chunk in T.serial(DIM // ROW_EXPAND_VECTOR_ELEMS):
                        o_mul_s1_r0_col = o_mul_s1_r0_chunk * ROW_EXPAND_VECTOR_ELEMS
                        T.tile.row_expand_mul_experiment(
                            st_o_acc[:, o_mul_s1_r0_col : o_mul_s1_r0_col + ROW_EXPAND_VECTOR_ELEMS],
                            ld_o_acc[:, o_mul_s1_r0_col : o_mul_s1_r0_col + ROW_EXPAND_VECTOR_ELEMS],
                            scale_s1_r0,
                        )
                else:
                    for o_mul_s1_r1_chunk in T.serial(DIM // ROW_EXPAND_VECTOR_ELEMS):
                        o_mul_s1_r1_col = o_mul_s1_r1_chunk * ROW_EXPAND_VECTOR_ELEMS
                        T.tile.row_expand_mul_experiment(
                            st_o_acc[:, o_mul_s1_r1_col : o_mul_s1_r1_col + ROW_EXPAND_VECTOR_ELEMS],
                            ld_o_acc[:, o_mul_s1_r1_col : o_mul_s1_r1_col + ROW_EXPAND_VECTOR_ELEMS],
                            scale_s1_r1,
                        )
            T.pipe_barrier("v")
            T.tile.add(st_o_acc, st_o_acc, ld_o_partial)
            T.set_flag("V", "MTE2", 0)

        if kv_idx == num_kv_blocks - 1:
            T.pipe_barrier("v")
            # st_o_acc now holds O_s UN-normalized and m_stats_ring/l_stats_ring
            # hold gm_s/gl_s -> merge the unshared decode attention right here.
            beam_start = local_q * BEAMS_PER_Q_BLOCK + vid * BEAMS_PER_M
            g_base = r * GROUPS_PER_STRIP
            # The two halves are written out EXPLICITLY, not as `for gi in
            # range(GROUPS_PER_STRIP)`: TVMScript parses `range` into a T.serial
            # LOOP (so gi would be a runtime tir.Var), and `preload`/`next_ghead`
            # must fold at PARSE time -- the whole point is that gi=0 and gi=1
            # get DIFFERENT DMA payloads (gi=0 preloads K_u/V_u and prefetches
            # gi=1's Q; gi=1 loads nothing at all).
            merge_half(
                request,
                kv_head,
                beam_start,
                g_base + 0,
                0,
                r * TILE_Q_ACC + 0 * BEAMS_PER_M,
                q_ring,
                bpr,
                decode_step,
                sm_scale,
                preload=True,
                next_ghead=g_base + 1,
                r=r,
                kv_once=kv_once,
                O_5d=O_5d,
                Q_5d=Q_5d,
                cor_s_b=cor_s_b,
                cor_u_b=cor_u_b,
                gl_mrg_b=gl_mrg_b,
                gl_u_b=gl_u_b,
                gm_mrg_b=gm_mrg_b,
                gm_u=gm_u,
                gm_u_b=gm_u_b,
                ku_ld=ku_ld,
                kv_f32=kv_f32,
                kv_f32_3=kv_f32_3,
                kv_f32_flat=kv_f32_flat,
                l_stats_ring=l_stats_ring,
                m_stats_ring=m_stats_ring,
                o_st=o_st,
                o_u=o_u,
                os_half=os_half,
                q_bf=q_bf,
                q_ld=q_ld,
                score_buf=score_buf,
                score_flat=score_flat,
                uk_4d=uk_4d,
                ured=ured,
                uv_4d=uv_4d,
                vu_ld=vu_ld,
                vu_ld_flat=vu_ld_flat,
                w_blk=w_blk,
                w_blk_flat=w_blk_flat,
                w_u=w_u,
            )
            merge_half(
                request,
                kv_head,
                beam_start,
                g_base + 1,
                1,
                r * TILE_Q_ACC + 1 * BEAMS_PER_M,
                q_ring,
                bpr,
                decode_step,
                sm_scale,
                preload=False,
                next_ghead=None,
                r=r,
                kv_once=kv_once,
                O_5d=O_5d,
                Q_5d=Q_5d,
                cor_s_b=cor_s_b,
                cor_u_b=cor_u_b,
                gl_mrg_b=gl_mrg_b,
                gl_u_b=gl_u_b,
                gm_mrg_b=gm_mrg_b,
                gm_u=gm_u,
                gm_u_b=gm_u_b,
                ku_ld=ku_ld,
                kv_f32=kv_f32,
                kv_f32_3=kv_f32_3,
                kv_f32_flat=kv_f32_flat,
                l_stats_ring=l_stats_ring,
                m_stats_ring=m_stats_ring,
                o_st=o_st,
                o_u=o_u,
                os_half=os_half,
                q_bf=q_bf,
                q_ld=q_ld,
                score_buf=score_buf,
                score_flat=score_flat,
                uk_4d=uk_4d,
                ured=ured,
                uv_4d=uv_4d,
                vu_ld=vu_ld,
                vu_ld_flat=vu_ld_flat,
                w_blk=w_blk,
                w_blk_flat=w_blk_flat,
                w_u=w_u,
            )

        T.set_flag("V", "MTE3", 1)
        T.wait_flag("V", "MTE3", 1)
        if kv_idx != num_kv_blocks - 1:
            T.copy(st_o_acc, ws_oa[cid, row : row + TILE_Q_ACC, :])
        T.set_flag("MTE3", "V", 1)


# ===========================================================================
# Main kernel
# ===========================================================================
@tilelang.jit(
    out_idx=[5],
    workspace_idx=[6, 7, 8],
    pass_configs=pass_configs,
    compile_flags=["--cce-auto-sync=off", "-O3"],
)
def flash_attention_xattn_fwd(
    request_num,
    bpr,
    kv_heads,
    group_size,
    kv_seq,
    decode_step,
    dim,
    *,
    kernel_name: str = "main_kernel",
):
    assert dim == DIM
    assert request_num > 0 and bpr > 0 and kv_seq > 0
    assert decode_step == 2, "This example supports exactly two per-beam tokens"
    assert DIM % ROW_EXPAND_VECTOR_ELEMS == 0
    assert kv_heads == KV_HEADS
    assert group_size == GROUP_SIZE
    assert kv_seq % TILE_KV_L2 == 0
    assert bpr * group_size % TILE_Q_L2 == 0

    dtype = "bfloat16"
    accum_dtype = "float"

    sm_scale = (1.0 / dim) ** 0.5

    # ---- merge scratch pin map, inside score_local's DMA-free region --------
    # Computed (not magic numbers) so it stays correct if decode_step changes,
    # and asserted against the region size -- overflowing it would silently push
    # the auto-placed stats rings past the UB budget and fault at runtime.
    _F32 = 4
    _BLK_B = MERGE_ROWS * ELEM_PER_BLK * _F32  # 1024
    _P_O_U = 0
    _P_URED = _P_O_U + MERGE_ROWS * DIM * _F32  # 16384
    _P_SCORE = _P_URED + MERGE_ROWS * DIM * 2  # 24576 (ured is 8192 B)
    _P_GM_U = _P_SCORE + decode_step * MERGE_ROWS * _F32
    _P_W_U = _P_GM_U + MERGE_ROWS * _F32
    _P_W_BLK = _P_W_U + MERGE_ROWS * _F32
    _P_GM_U_B = _P_W_BLK + decode_step * _BLK_B
    _P_GL_U_B = _P_GM_U_B + _BLK_B
    _P_GM_MRG_B = _P_GL_U_B + _BLK_B
    _P_GL_MRG_B = _P_GM_MRG_B  # alias: gm_mrg_b is dead before gl_mrg_b's
    # first write (see merge_half PHASE C)
    _P_COR_S_B = _P_GM_MRG_B + _BLK_B
    _P_COR_U_B = _P_COR_S_B + _BLK_B
    # C4 fuses the t-loop pairwise and spans q_bf+kv_f32 with kv_f32_flat.
    assert decode_step >= 2, "C4's fused accumulation assumes decode_step >= 2"

    # ---- D1 load-region pin map: [_LD_BASE, _LD_END) aliases ld_o_acc +
    # ld_o_partial, so everything here is MTE2-touched (or under-lock V scratch)
    # under the SINGLE LD_BUF lock -- the aliasing rule holds by construction.
    _LD_BASE = 81920
    _LD_END = 147456  # == ld_o_partial's end
    _KVB = decode_step * MERGE_ROWS * DIM * 2  # one bf16 K_u/V_u block
    _P_KU = _LD_BASE
    _P_VU = _P_KU + _KVB
    _P_Q_BF = _P_VU + _KVB
    _P_KV_F32 = _P_Q_BF + MERGE_ROWS * DIM * _F32
    _LD_USED_END = _P_KV_F32 + MERGE_ROWS * DIM * _F32
    assert _LD_USED_END <= _LD_END, (
        f"merge load region needs {_LD_USED_END - _LD_BASE} B but only {_LD_END - _LD_BASE} B alias ld_o_acc/ld_o_partial"
    )
    # kv_f32_flat (the C4 fused accumulation) spans q_bf+kv_f32 -- both are
    # PHASE-A-only and dead by then, and all three are under-lock LD_BUF scratch.
    assert decode_step * MERGE_ROWS * DIM * _F32 <= _LD_USED_END - _P_Q_BF, "kv_f32_flat overflows the q_bf+kv_f32 under-lock span"
    # q_ld no longer fits the 64 KB alias window, so it takes a pin of its OWN
    # just above it. That region is touched by NOTHING but merge's MTE2 under
    # LD_BUF, so it is a clean single-lock domain -- but it PUSHES the
    # auto-placed buffers up by its size (auto starts above the highest pin end),
    # which is what the UB-budget assert below guards.
    _P_Q_LD = _LD_END
    _AUTO_BASE = _P_Q_LD + MERGE_ROWS * DIM * 2
    assert all(a % 32 == 0 for a in (_P_KU, _P_VU, _P_Q_BF, _P_KV_F32, _P_Q_LD)), "UB pins must be 32 B aligned"
    _MERGE_SCRATCH_END = _P_COR_U_B + _BLK_B
    _SAFE_REGION_END = TILE_Q_UB * TILE_KV_L2 * _F32  # score_local: 32768
    assert _MERGE_SCRATCH_END <= _SAFE_REGION_END, (
        f"merge scratch {_MERGE_SCRATCH_END} B overflows score_local's DMA-free "
        f"region ({_SAFE_REGION_END} B) -- it would collide with o_st @32768"
    )
    assert all(
        a % 32 == 0
        for a in (_P_O_U, _P_URED, _P_SCORE, _P_GM_U, _P_W_U, _P_W_BLK, _P_GM_U_B, _P_GL_U_B, _P_GM_MRG_B, _P_COR_S_B, _P_COR_U_B)
    ), "UB pins must be 32 B aligned"

    N = request_num * bpr
    num_heads = kv_heads * group_size
    q_seq = bpr * group_size
    num_bh = request_num * kv_heads

    num_q_blocks = q_seq // TILE_Q_L2
    num_kv_blocks = kv_seq // TILE_KV_L2
    global_q_tasks = num_bh * num_q_blocks

    num_q_stages = 1 + (NUM_STAGES + num_kv_blocks - 1) // num_kv_blocks

    # D2 (see merge_half): with a single KV block, kv_idx = old_step % 1 is
    # always 0, so vec_o_acc's kv_idx!=0 path -- the ONLY writer of ld_o_acc, and
    # hence of ku_ld/vu_ld's [81920,114688) window -- can never execute. The
    # merge's K_u/V_u then survive the whole r loop and load once per step.
    # Computed HERE, in the builder, and passed down as a macro ARG: a
    # `_KV_ONCE = ...` assignment *inside* a parsed macro body gets intercepted
    # by TVMScript and turned into a TIR Var (verified), which would silently
    # demote this parse-time switch into a runtime branch. Macro ARGS keep their
    # Python identity (that is why `preload=True` folds correctly).
    _KV_ONCE = num_kv_blocks == 1

    # ---- UB budget: the auto-placer puts every un-pinned buffer ABOVE the
    # highest pin end, NOT in the gaps below it. D1's new q_ld pin raises that
    # ceiling by 8 KB, so the stats rings + SoftmaxFlashV2 workspace all shift up
    # and could silently run past the 196,352 B UB budget -- which faults at
    # RUNTIME (aicore exception), not at compile time, and may only surface under
    # sustained profiling. Assert it here instead.
    _AUTO_BYTES = (
        2 * num_q_stages * HALF_Q * ELEM_PER_BLK * _F32  # m_/l_stats_ring
        + NUM_STAGES * HALF_Q * ELEM_PER_BLK * _F32  # scale_ring
        + HALF_Q * ELEM_PER_BLK * _F32  # m_prev
        + HALF_Q * _F32  # l_panel
        + TILE_Q_ACC * ELEM_PER_BLK * _F32
    )  # sum_brcb; arena is pinned
    _UB_BUDGET = 196352
    assert _AUTO_BASE + _AUTO_BYTES <= _UB_BUDGET, (
        f"UB overflow: pins end at {_AUTO_BASE} B, auto-placed buffers need "
        f"{_AUTO_BYTES} B -> {_AUTO_BASE + _AUTO_BYTES} B > {_UB_BUDGET} B budget"
    )

    q_tasks_per_core = global_q_tasks // NUM_CORES
    r_tasks = global_q_tasks % NUM_CORES

    shape_q = [N, num_heads, dim]
    shape_kv = [request_num * kv_seq, kv_heads, dim]
    shape_unshared = [N, kv_heads, decode_step, dim]

    @T.prim_func
    def main(
        Q: T.Tensor(shape_q, dtype),
        K_shared: T.Tensor(shape_kv, dtype),
        V_shared: T.Tensor(shape_kv, dtype),
        unshared_key: T.Tensor(shape_unshared, dtype),
        unshared_value: T.Tensor(shape_unshared, dtype),
        Output: T.Tensor(shape_q, dtype),
        ws_sp: T.Tensor([NUM_CORES, NUM_STAGES, TILE_Q_L2, TILE_KV_L2], dtype),
        ws_op: T.Tensor([NUM_CORES, NUM_STAGES, TILE_Q_L2, DIM], accum_dtype),
        ws_oa: T.Tensor([NUM_CORES, TILE_Q_L2, DIM], accum_dtype),
    ):
        T.func_attr({"global_symbol": kernel_name})
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            Q_5d = T.decl_buffer([request_num, bpr, kv_heads, group_size, dim], dtype, data=Q.data, scope="global")
            O_5d = T.decl_buffer([request_num, bpr, kv_heads, group_size, dim], dtype, data=Output.data, scope="global")
            K_3d = T.decl_buffer([request_num, kv_seq, kv_heads, dim], dtype, data=K_shared.data, scope="global")
            V_3d = T.decl_buffer([request_num, kv_seq, kv_heads, dim], dtype, data=V_shared.data, scope="global")
            uk_4d = T.decl_buffer([N, kv_heads, decode_step, dim], dtype, data=unshared_key.data, scope="global")
            uv_4d = T.decl_buffer([N, kv_heads, decode_step, dim], dtype, data=unshared_value.data, scope="global")

            a_l1 = T.alloc_L1([2, TILE_M, TILE_K], dtype)
            kv_l1 = T.alloc_L1([N_ITERS, TILE_K, DIM], dtype)
            T.annotate_layout({a_l1: make_zn_layout(a_l1), kv_l1: make_nz_layout(kv_l1)})

            l0a = T.alloc_L0A([2, TILE_M, TILE_K], dtype)
            l0b = T.alloc_L0B([2, TILE_K, DIM], dtype)
            l0c = T.alloc_L0C([2, TILE_M, DIM], accum_dtype)

            # --- UB layout -------------------------------------------------
            # [0K,   32K)  V-scratch, NO DMA -- score_local | merge "safe" scratch
            # [32K,  48K)  ST_ON  -- o_st[2,32,128] bf16 (merged O store)
            # [48K,  80K)  ST_BUF -- st_prob / st_o_acc / os_half (alias)
            # [80K, 144K)  LD_BUF -- ld_score / ld_o_acc / ld_o_partial;
            #                        merge loads (q_ld, kv_ld) + under-lock scratch
            # [144K, ...)  auto   -- stats rings + SoftmaxFlashV2 workspace (~34K)
            score_local = T.alloc_ub([TILE_Q_UB, TILE_KV_L2], accum_dtype)  # 32KB

            st_prob = T.alloc_ub([TILE_Q_UB, TILE_KV_L2], dtype)  # 16KB
            st_o_acc = T.alloc_ub([TILE_Q_ACC, DIM], accum_dtype)  # 32KB (alias st_prob)
            # 3-D alias VIEW of st_o_acc: os_half[gi] == st_o_acc rows
            # [gi*32, gi*32+32) == one group-head x 32 beams. A leading-dim
            # slice is the form tile ops index correctly (unlike a row slice
            # of a 2-D buffer, whose extent inference is less certain).
            os_half = T.alloc_ub([GROUPS_PER_STRIP, BEAMS_PER_M, DIM], accum_dtype)

            ld_score = T.alloc_ub([TILE_Q_UB, TILE_KV_L2], dtype)  # 16KB
            ld_o_acc = T.alloc_ub([TILE_Q_ACC, DIM], accum_dtype)  # 32KB (alias ld_score)
            ld_o_partial = T.alloc_ub([TILE_Q_ACC, DIM], accum_dtype)  # 32KB

            # --- Auto-placed: stats + SoftmaxFlashV2 workspace (~34K) ---
            # m_stats ringed (like l_stats_ring) so the running max survives
            # pipeline overlap and is still valid at the old-step vec_o_acc.
            m_stats_ring = T.alloc_ub([num_q_stages, HALF_Q, ELEM_PER_BLK], accum_dtype)
            l_stats_ring = T.alloc_ub([num_q_stages, HALF_Q, ELEM_PER_BLK], accum_dtype)
            scale_s0_r0 = T.alloc_ub([TILE_Q_ACC, ELEM_PER_BLK], accum_dtype)
            scale_s0_r1 = T.alloc_ub([TILE_Q_ACC, ELEM_PER_BLK], accum_dtype)
            scale_s1_r0 = T.alloc_ub([TILE_Q_ACC, ELEM_PER_BLK], accum_dtype)
            scale_s1_r1 = T.alloc_ub([TILE_Q_ACC, ELEM_PER_BLK], accum_dtype)
            sfm_tmp = T.alloc_ub([SFM_WORKSPACE_BYTES], "uint8")
            sfm_row = T.alloc_ub([8], accum_dtype)
            sfm_brcb = T.alloc_ub([8, ELEM_PER_BLK], accum_dtype)
            m_prev = T.alloc_ub([HALF_Q, ELEM_PER_BLK], accum_dtype)
            l_panel = T.alloc_ub([HALF_Q, 1], accum_dtype)
            sum_brcb = T.alloc_ub([TILE_Q_ACC, ELEM_PER_BLK], accum_dtype)
            T.annotate_address({sfm_tmp: 65536, sfm_row: 73920, sfm_brcb: 73952})

            # ---- merge buffers (overlay phase-1 UB, BY CATEGORY) ----
            # store (V->MTE3, ST_ON): 2 slots, slot = gi
            o_st = T.alloc_ub([GROUPS_PER_STRIP, MERGE_ROWS, DIM], dtype)  # 16KB
            # loads (MTE2->V, LD_BUF). D1: K_u and V_u are now BOTH resident for
            # the whole strip (they are gi-invariant), so they get separate
            # buffers instead of v3p's single re-used kv_ld.
            q_ld = T.alloc_ub([MERGE_ROWS, DIM], dtype)  # 8KB
            ku_ld = T.alloc_ub([decode_step, MERGE_ROWS, DIM], dtype)  # 16KB
            vu_ld = T.alloc_ub([decode_step, MERGE_ROWS, DIM], dtype)  # 16KB
            vu_ld_flat = T.alloc_ub([decode_step * MERGE_ROWS, DIM], dtype)
            # under-lock V-scratch (aliases ld_o_acc / ld_o_partial)
            q_bf = T.alloc_ub([MERGE_ROWS, DIM], accum_dtype)  # 16KB
            kv_f32 = T.alloc_ub([MERGE_ROWS, DIM], accum_dtype)  # 16KB
            # C4 flat views for the fused o_u accumulation. kv_f32_flat/kv_f32_3
            # span BOTH q_bf's and kv_f32's slots (decode_step*16KB = the whole
            # 32KB under-lock region): legal because q_bf and kv_f32 are PHASE-A
            # only and dead by the accumulation, and all three are under-lock
            # V-scratch guarded by the same LD_BUF lock.
            kv_f32_flat = T.alloc_ub([decode_step * MERGE_ROWS, DIM], accum_dtype)
            kv_f32_3 = T.alloc_ub([decode_step, MERGE_ROWS, DIM], accum_dtype)
            # safe V-scratch (score_local's region; live across REL V LD_BUF)
            o_u = T.alloc_ub([MERGE_ROWS, DIM], accum_dtype)  # 16KB
            ured = T.alloc_ub([MERGE_ROWS * DIM * 2], "uint8")  # 8KB reduce tmp
            score_buf = T.alloc_ub([decode_step, MERGE_ROWS, 1], accum_dtype)
            # Flat 2-D alias over ALL of score_buf (same bytes, same row-major
            # layout: score_flat[t*MERGE_ROWS+i, 0] == score_buf[t, i, 0]) so the
            # sm_scale fold is a single tiny op. V-only safe scratch -> aliasing
            # is free. The per-t 3-D leading-dim slices stay on score_buf, which
            # is the form tile ops index correctly.
            score_flat = T.alloc_ub([decode_step * MERGE_ROWS, 1], accum_dtype)
            # [32,1] scalars: only the two that MUST match score_buf's shape.
            gm_u = T.alloc_ub([MERGE_ROWS, 1], accum_dtype)
            w_u = T.alloc_ub([MERGE_ROWS, 1], accum_dtype)
            # 8-lane (ELEM_PER_BLK) blocks: the whole merge scalar chain lives
            # here, so every row_expand_* src1 is a genuine block by
            # construction and no broadcast/reduce round-trip is needed.
            # Per-t weight blocks: replace v3n's single `blk` scratch, because
            # C3 needs all decode_step weights live at once (they are computed
            # before the LSE chain, then scaled by a_u, then consumed).
            w_blk = T.alloc_ub([decode_step, MERGE_ROWS, ELEM_PER_BLK], accum_dtype)
            w_blk_flat = T.alloc_ub([decode_step * MERGE_ROWS, ELEM_PER_BLK], accum_dtype)
            gm_u_b = T.alloc_ub([MERGE_ROWS, ELEM_PER_BLK], accum_dtype)
            gl_u_b = T.alloc_ub([MERGE_ROWS, ELEM_PER_BLK], accum_dtype)
            gm_mrg_b = T.alloc_ub([MERGE_ROWS, ELEM_PER_BLK], accum_dtype)
            cor_s_b = T.alloc_ub([MERGE_ROWS, ELEM_PER_BLK], accum_dtype)
            cor_u_b = T.alloc_ub([MERGE_ROWS, ELEM_PER_BLK], accum_dtype)
            gl_mrg_b = T.alloc_ub([MERGE_ROWS, ELEM_PER_BLK], accum_dtype)

            T.annotate_address(
                {
                    # ---- phase-1 pins (unchanged except st_o_norm -> o_st) ----
                    score_local: 0,
                    st_prob: 49152,
                    st_o_acc: 49152,
                    os_half: 49152,
                    ld_score: 81920,
                    ld_o_acc: 81920,
                    ld_o_partial: 114688,
                    # ---- merge: store -> ST_ON region [32768,49152) ----
                    o_st: 32768,
                    # ---- merge: loads + under-lock scratch -> LD_BUF region ----
                    # D1 re-lays-out [81920,147456): K_u and V_u are now both live
                    # for the whole strip, so the 64 KB is EXACTLY full --
                    # ku(16)+vu(16)+[q_bf(16)|kv_f32(16) == kv_f32_flat(32)].
                    # q_ld is evicted to its own pin just above (see _P_Q_LD).
                    ku_ld: _P_KU,
                    vu_ld: _P_VU,
                    vu_ld_flat: _P_VU,
                    q_bf: _P_Q_BF,
                    kv_f32: _P_KV_F32,
                    kv_f32_flat: _P_Q_BF,
                    kv_f32_3: _P_Q_BF,  # C4: span q_bf+kv_f32
                    q_ld: _P_Q_LD,
                    # ---- merge: safe scratch -> score_local region (no DMA) ----
                    # Computed + asserted above (see the pin map next to sm_scale):
                    # o_u 16K + ured 8K + score_buf 256B + 2x [32,1] + w_blk
                    # (decode_step x 1K) + 5 distinct [32,8] blocks -> ends at 32256
                    # < 32768, so the whole merge scratch still fits score_local's
                    # DMA-free region and nothing is pushed into the auto-placed
                    # area above the pins.
                    o_u: _P_O_U,
                    ured: _P_URED,
                    score_buf: _P_SCORE,
                    score_flat: _P_SCORE,
                    gm_u: _P_GM_U,
                    w_u: _P_W_U,
                    w_blk: _P_W_BLK,
                    w_blk_flat: _P_W_BLK,
                    gm_u_b: _P_GM_U_B,
                    gl_u_b: _P_GL_U_B,
                    gm_mrg_b: _P_GM_MRG_B,
                    gl_mrg_b: _P_GL_MRG_B,  # aliased: disjoint lifetimes
                    cor_s_b: _P_COR_S_B,
                    cor_u_b: _P_COR_U_B,
                }
            )

            my_start = T.alloc_var("int32", init=0)
            my_count = T.alloc_var("int32", init=0)
            if cid < r_tasks:
                my_start = cid * q_tasks_per_core + cid
                my_count = q_tasks_per_core + 1
            else:
                my_start = cid * q_tasks_per_core + r_tasks
                my_count = q_tasks_per_core
            my_total_steps = my_count * num_kv_blocks

            with T.Scope("C"):
                T.set_flag("MTE1", "MTE2", 0)
                T.set_flag("MTE1", "MTE2", 1)
                T.set_flag("MTE1", "MTE2", 2)
                T.set_flag("M", "MTE1", 3)
                T.set_flag("M", "MTE1", 4)
                T.set_flag("FIX", "M", 5)
                T.set_flag("FIX", "M", 6)

                for step in T.serial(my_total_steps + NUM_STAGES):
                    T.wait_cross_flag(SEM_CUBE)
                    stage = step % NUM_STAGES

                    if step >= NUM_STAGES:
                        old_step = step - NUM_STAGES
                        old_global_q = my_start + old_step // num_kv_blocks
                        bh_a = old_global_q // num_q_blocks
                        request_a = bh_a // kv_heads
                        kv_head_a = bh_a % kv_heads
                        kv_off_a = (old_step % num_kv_blocks) * TILE_KV_L2
                        gemm_pv(
                            request_a,
                            kv_head_a,
                            kv_off_a,
                            stage,
                            V_3d=V_3d,
                            a_l1=a_l1,
                            cid=cid,
                            kv_l1=kv_l1,
                            l0a=l0a,
                            l0b=l0b,
                            l0c=l0c,
                            ws_op=ws_op,
                            ws_sp=ws_sp,
                        )

                    if step < my_total_steps:
                        global_q = my_start + step // num_kv_blocks
                        bh_b = global_q // num_q_blocks
                        local_q_b = global_q % num_q_blocks
                        request_b = bh_b // kv_heads
                        kv_head_b = bh_b % kv_heads
                        beam_base_b = local_q_b * BEAMS_PER_Q_BLOCK
                        kv_off_b = (step % num_kv_blocks) * TILE_KV_L2
                        gemm_qk(
                            request_b,
                            kv_head_b,
                            beam_base_b,
                            kv_off_b,
                            stage,
                            K_3d=K_3d,
                            Q_5d=Q_5d,
                            a_l1=a_l1,
                            cid=cid,
                            kv_l1=kv_l1,
                            l0a=l0a,
                            l0b=l0b,
                            l0c=l0c,
                            ws_sp=ws_sp,
                        )

                    T.set_cross_flag("FIX", SEM_VEC)

                T.wait_flag("MTE1", "MTE2", 0)
                T.wait_flag("MTE1", "MTE2", 1)
                T.wait_flag("MTE1", "MTE2", 2)
                T.wait_flag("M", "MTE1", 3)
                T.wait_flag("M", "MTE1", 4)
                T.wait_flag("FIX", "M", 5)
                T.wait_flag("FIX", "M", 6)

            with T.Scope("V"):
                for _ in range(NUM_STAGES):
                    T.set_cross_flag("MTE2", SEM_CUBE)
                T.set_flag("V", "MTE2", 0)
                T.set_flag("MTE3", "V", 1)
                T.set_flag("MTE3", "V", 2)
                T.set_flag("MTE3", "V", 3)
                T.set_flag("MTE3", "MTE2", 4)

                for step in T.serial(my_total_steps + NUM_STAGES):
                    T.wait_cross_flag(SEM_VEC)

                    if step >= NUM_STAGES:
                        T.wait_flag("MTE3", "MTE2", 4)
                        vec_o_acc(
                            step,
                            my_start,
                            num_q_blocks,
                            num_kv_blocks,
                            num_q_stages,
                            bpr,
                            decode_step,
                            sm_scale,
                            _KV_ONCE,
                            O_5d=O_5d,
                            Q_5d=Q_5d,
                            cid=cid,
                            cor_s_b=cor_s_b,
                            cor_u_b=cor_u_b,
                            gl_mrg_b=gl_mrg_b,
                            gl_u_b=gl_u_b,
                            gm_mrg_b=gm_mrg_b,
                            gm_u=gm_u,
                            gm_u_b=gm_u_b,
                            ku_ld=ku_ld,
                            kv_f32=kv_f32,
                            kv_f32_3=kv_f32_3,
                            kv_f32_flat=kv_f32_flat,
                            l_stats_ring=l_stats_ring,
                            ld_o_acc=ld_o_acc,
                            ld_o_partial=ld_o_partial,
                            m_stats_ring=m_stats_ring,
                            o_st=o_st,
                            o_u=o_u,
                            os_half=os_half,
                            q_bf=q_bf,
                            q_ld=q_ld,
                            scale_s0_r0=scale_s0_r0,
                            scale_s0_r1=scale_s0_r1,
                            scale_s1_r0=scale_s1_r0,
                            scale_s1_r1=scale_s1_r1,
                            score_buf=score_buf,
                            score_flat=score_flat,
                            st_o_acc=st_o_acc,
                            uk_4d=uk_4d,
                            ured=ured,
                            uv_4d=uv_4d,
                            vid=vid,
                            vu_ld=vu_ld,
                            vu_ld_flat=vu_ld_flat,
                            w_blk=w_blk,
                            w_blk_flat=w_blk_flat,
                            w_u=w_u,
                            ws_oa=ws_oa,
                            ws_op=ws_op,
                        )
                        T.set_flag("MTE3", "MTE2", 4)

                    if step < my_total_steps:
                        vec_softmax(
                            step,
                            my_start,
                            num_q_blocks,
                            num_kv_blocks,
                            num_q_stages,
                            sm_scale,
                            cid=cid,
                            l_panel=l_panel,
                            l_stats_ring=l_stats_ring,
                            ld_score=ld_score,
                            m_prev=m_prev,
                            m_stats_ring=m_stats_ring,
                            scale_s0_r0=scale_s0_r0,
                            scale_s0_r1=scale_s0_r1,
                            scale_s1_r0=scale_s1_r0,
                            scale_s1_r1=scale_s1_r1,
                            score_local=score_local,
                            sfm_brcb=sfm_brcb,
                            sfm_row=sfm_row,
                            sfm_tmp=sfm_tmp,
                            st_prob=st_prob,
                            sum_brcb=sum_brcb,
                            vid=vid,
                            ws_sp=ws_sp,
                        )

                T.wait_flag("V", "MTE2", 0)
                T.wait_flag("MTE3", "V", 1)
                T.wait_flag("MTE3", "V", 2)
                T.wait_flag("MTE3", "V", 3)
                T.wait_flag("MTE3", "MTE2", 4)

    return main


def reference_attention(query, shared_key, shared_value, unshared_key, unshared_value, request_num, beams, seq_len):
    """FP32 reference: one softmax over the shared and per-beam keys."""
    q = query.float().reshape(request_num, beams, KV_HEADS, GROUP_SIZE, DIM)
    sk = shared_key.float().reshape(request_num, seq_len, KV_HEADS, DIM)
    sv = shared_value.float().reshape(request_num, seq_len, KV_HEADS, DIM)
    uk = unshared_key.float().reshape(request_num, beams, KV_HEADS, 2, DIM)
    uv = unshared_value.float().reshape(request_num, beams, KV_HEADS, 2, DIM)
    output = torch.empty_like(q)
    for request in range(request_num):
        for head in range(KV_HEADS):
            for first in range(0, beams, 128):
                last = min(first + 128, beams)
                q_part = q[request, first:last, head]
                shared_scores = torch.matmul(q_part, sk[request, :, head].transpose(0, 1))
                per_beam_scores = torch.einsum("bgd,btd->bgt", q_part, uk[request, first:last, head])
                scores = torch.cat((shared_scores, per_beam_scores), dim=-1) * (DIM**-0.5)
                weights = torch.softmax(scores, dim=-1)
                shared_output = torch.matmul(weights[..., :seq_len], sv[request, :, head])
                per_beam_output = torch.einsum("bgt,btd->bgd", weights[..., seq_len:], uv[request, first:last, head])
                output[request, first:last, head] = shared_output + per_beam_output
    return output.reshape(request_num * beams, KV_HEADS * GROUP_SIZE, DIM).to(query.dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-num", type=int)
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--kv-seqlen", type=int)
    parser.add_argument("--all-cases", action="store_true", help="check the original 18-shape matrix")
    args = parser.parse_args()
    supplied = (args.request_num, args.beam_size, args.kv_seqlen)
    if args.all_cases and any(x is not None for x in supplied):
        parser.error("--all-cases cannot be combined with a single shape")
    if args.all_cases:
        cases = [(r, b, s) for r, s in ((1, 1024), (2, 1024), (2, 2048)) for b in (128, 256, 512, 1024, 2048, 4096)]
    elif any(x is not None for x in supplied):
        cases = [
            (
                1 if args.request_num is None else args.request_num,
                128 if args.beam_size is None else args.beam_size,
                1024 if args.kv_seqlen is None else args.kv_seqlen,
            )
        ]
    else:
        cases = [(1, 128, 1024), (2, 256, 2048)]

    torch.set_default_device("npu")
    for request_num, beams, seq_len in cases:
        torch.manual_seed(42)
        tokens = request_num * beams
        shapes = [
            (tokens, 32, DIM),
            (request_num * seq_len, KV_HEADS, DIM),
            (request_num * seq_len, KV_HEADS, DIM),
            (tokens, KV_HEADS, 2, DIM),
            (tokens, KV_HEADS, 2, DIM),
        ]
        inputs = [torch.empty(shape, dtype=torch.bfloat16).uniform_(-1, 1) for shape in shapes]
        kernel = flash_attention_xattn_fwd(request_num, beams, KV_HEADS, GROUP_SIZE, seq_len, 2, DIM)
        expected = reference_attention(*inputs, request_num, beams, seq_len)
        for _ in range(3):
            output = kernel(*inputs)
            torch.npu.synchronize()
            torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
        print(f"Checked R={request_num}, B={beams}, shared length={seq_len}")
    print("Kernel Output Match!")
