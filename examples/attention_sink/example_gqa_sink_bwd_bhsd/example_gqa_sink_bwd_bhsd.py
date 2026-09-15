"""GQA Sink Attention (BHSD) for Ascend NPU — on-chip single-kernel backward.

Layout: BHSD (Batch, Heads, SeqLen, Dim). Supports GQA (grouped-query attention),
an attention sink token, and an optional sliding window mask. fp16, Developer mode.

Architecture (preprocess merged into bwd kernel — 3 kernels, 1 host sync):
  Kernel 1 (fwd, Hybrid):           Forward (online softmax + sink + window) -> O, lse
  Kernel 2 (bwd, Developer):        Single kernel: Phase0 Delta + KV-loop(5 GEMM + softmax + dS) + Phase5 dSink
  Kernel 3 (postprocess, Vector):   dK+dV fp32->fp16 dual-output cast

3 kernels total (preprocess merged into bwd Phase 0 + Phase 5):
  1. flashattn_fwd:               Forward (online softmax + sink + window) -> O, lse
  2. flashattn_bwd:               Single kernel: Phase0(Delta) + KV-loop(5 GEMM+softmax+dS) + Phase5(dSink)
  3. flashattn_bwd_postprocess:   dK+dV fp32 -> fp16 (dual-output, dQ direct fp16 from bwd)

preprocess merged into bwd (eliminates preprocess->bwd host sync):
  - Phase 0 (before KV loop): Delta = sum(O * dO, dim=-1) computed in Vector scope
    using UB buffers (o_ub, do_ub, prod_ub, do_fp32). Delta stays in delta_ub (UB)
    for Phase 4 dS compute — no GM roundtrip. Also written to Delta_out (GM) for
    precision verification. Processes block_M rows in two halves (hm=block_M//2)
    to reduce UB pressure.
  - Phase 5 (after KV loop): dSink = -exp(sink - lse) * Delta computed in Vector
    scope using sink_val_ub, sink_exp_ub. Uses delta_ub (from Phase 0) and lse_ub
    (loaded at kernel start). Outputs dSinks to GM.
  - New bwd inputs: O (fwd output), Sinks (sink values).
  - New bwd outputs: Delta_out (fp32), dSinks (fp32).
  - Removed: Delta input (now computed internally).
  - Host syncs: 2->1 (preprocess->bwd sync eliminated; bwd->postprocess sync remains).

On-chip CV transfer:
  - alloc_shared / alloc_fragment replace alloc_L1 / alloc_ub / alloc_L0C.
  - 6 GM workspace buffers eliminated via on-chip direct T.copy(fragment, shared):
    workspace_s, workspace_p, workspace_dp, workspace_ds -> L0C->UB / UB->L1 direct,
    AND ws_p_delta, ws_ds_delta -> UB->L1 direct (p_delta_half / ds_delta_half are
    same size as p_half which already does UB->L1 direct).
  - threads=1 (no vid), Kernel(bwd_block_num, threads=1, is_npu=True) as (cid).
  - CompGEMM via L0C accumulation: main GEMM init=True + corr GEMM init=False
    on same L0C.

Key design decisions:
  - No T.Scope, no set_flag/wait_flag, no cross_flag, no barrier_all — pure
    AUTO_CV_SYNC + AUTO_SYNC automatic synchronization (Developer mode).
  - Compensated GEMM: fp16 GEMM result corrected by a second GEMM on the fp16
    quantization residual (p_delta for GEMM2corr, ds_delta for GEMM4corr).
  - Single loop (no split-loop mask skip) — simplifies CV sync validation.
  - dQ written as fp16 directly from bwd kernel (L0C fp32 -> GM fp16 auto-cast).
  - P_fp32 retained in UB (s_ub), NOT written to GM — eliminates workspace_p_fp32.
  - Precision: 169-line standard (atol=6.10e-5, rtol=1.95e-3 for fp16).
"""

import tilelang
import torch
from tilelang import language as T

# ============================================================================
# pass_configs (4 explicit keys per config)
# ============================================================================

# Hybrid mode for fwd + single-kernel bwd (flashattn_bwd) — AUTO_CV_COMBINE/SYNC
# needed for L0C->GM->UB two-hop accumulation pattern and intra-kernel CV transfer.
# bwd kernel also includes Phase 0 (Delta) + Phase 5 (dSink) in Vector scope.
_hybrid_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# ============================================================================
# Kernel 1: Forward (online softmax + attention sink + sliding window)
# ============================================================================


@tilelang.jit(out_idx=[3, 4], workspace_idx=[6, 7, 8], pass_configs=_hybrid_pass_configs)
def flashattn_fwd(batch, heads, seq_len, dim, groups, window_size, block_M=64, block_N=64):
    """Forward: produces O [B,H,N,dim] fp16 and lse [B,H,N] fp32.

    Online softmax with attention sink: the sink logit participates in the max
    and normalizer, so O = sum(P * V) / (sum(P) + exp(sink - m)). Mask is
    vectorized as 2D tile ops (broadcast + compare + select) instead of a
    per-row Python loop.
    """
    assert seq_len % block_M == 0
    assert seq_len % block_N == 0
    sm_scale = (1.0 / dim) ** 0.5
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"

    q_shape = [batch, heads, seq_len, dim]
    kv_shape = [batch, head_kv, seq_len, dim]
    o_shape = [batch, heads, seq_len, dim]
    lse_shape = [batch, heads, seq_len]
    block_num = (seq_len // block_M) * heads * batch

    if window_size is not None:
        assert window_size % block_N == 0

    window_eff = window_size if window_size is not None else seq_len * 2
    hm = block_M // 2

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(kv_shape, dtype),  # type: ignore
        V: T.Tensor(kv_shape, dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        Sinks: T.Tensor([heads], dtype),  # type: ignore
        workspace_1: T.Tensor([block_num, block_M, block_N], accum_dtype),  # type: ignore
        workspace_2: T.Tensor([block_num, block_M, block_N], dtype),  # type: ignore
        workspace_3: T.Tensor([block_num, block_M, dim], accum_dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch
            kv_by = by // groups

            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)
            acc_s_l1 = T.alloc_L1([block_M, block_N], dtype)
            acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_M, dim], accum_dtype)

            acc_o = T.alloc_ub([hm, dim], accum_dtype)
            sumexp = T.alloc_ub([hm], accum_dtype)
            m_i = T.alloc_ub([hm], accum_dtype)
            acc_s_ub = T.alloc_ub([hm, block_N], accum_dtype)
            m_i_prev = T.alloc_ub([hm], accum_dtype)
            acc_s_ub_ = T.alloc_ub([hm, block_N], accum_dtype)
            sumexp_i_ub = T.alloc_ub([hm], accum_dtype)
            acc_s_half = T.alloc_ub([hm, block_N], dtype)
            acc_o_ub = T.alloc_ub([hm, dim], accum_dtype)
            acc_o_half = T.alloc_ub([hm, dim], dtype)
            col_pos = T.alloc_ub([block_N], accum_dtype)
            row_pos = T.alloc_ub([hm], accum_dtype)
            row_pos_2d = T.alloc_ub([hm, block_N], accum_dtype)
            mask_2d = T.alloc_ub([hm, block_N], accum_dtype)
            causal_mask_2d = T.alloc_ub([hm, block_N], accum_dtype)
            sink_ub = T.alloc_ub([hm], accum_dtype)
            sink_exp_ub = T.alloc_ub([hm], accum_dtype)
            sink_scalar = T.alloc_ub([1], dtype)

            loop_st = T.max(0, (bx * block_M - window_eff) // block_N)
            loop_ed = T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N))

            v_row = vid * hm
            q_row = bx * block_M

            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -(2**30))

            T.copy(Sinks[by : by + 1], sink_scalar)
            T.tile.fill(sink_ub, sink_scalar[0])

            T.copy(Q[bz, by, q_row : q_row + block_M, :], q_l1)

            T.tile.arith_progression(row_pos, q_row + v_row, 1, hm)

            for k in T.serial(loop_st, loop_ed):
                kv = k * block_N

                # Cube: QK^T -> workspace_1
                T.copy(K[bz, kv_by, kv : kv + block_N, :], k_l1)
                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                T.copy(acc_s_l0c, workspace_1[cid, :, :])

                # Vector: softmax + mask -> workspace_2
                T.tile.fill(acc_s_ub, 0.0)
                T.copy(m_i, m_i_prev)
                T.copy(workspace_1[cid, v_row : v_row + hm, :], acc_s_ub_)
                T.tile.axpy(acc_s_ub, acc_s_ub_, sm_scale)

                T.tile.arith_progression(col_pos, kv, 1, block_N)
                T.tile.broadcast(acc_s_ub_, col_pos, axis=0)
                T.tile.broadcast(row_pos_2d, row_pos, axis=1)
                if window_size is not None:
                    T.tile.compare(causal_mask_2d, acc_s_ub_, row_pos_2d, "LE")
                    T.tile.sub(row_pos_2d, row_pos_2d, window_size)
                    T.tile.compare(mask_2d, acc_s_ub_, row_pos_2d, "GT")
                    T.tile.bitwise_and(mask_2d, causal_mask_2d, mask_2d)
                else:
                    T.tile.compare(mask_2d, acc_s_ub_, row_pos_2d, "LE")
                T.tile.select(
                    acc_s_ub,
                    mask_2d,
                    acc_s_ub,
                    -T.infinity(accum_dtype),
                    "VSEL_TENSOR_SCALAR_MODE",
                )

                T.reduce_max(acc_s_ub, m_i, dim=-1)
                T.tile.max(m_i, m_i, m_i_prev)
                T.tile.sub(m_i_prev, m_i_prev, m_i)
                T.tile.exp(m_i_prev, m_i_prev)
                T.tile.broadcast(acc_s_ub_, m_i, axis=1)
                T.tile.sub(acc_s_ub, acc_s_ub, acc_s_ub_)
                T.tile.exp(acc_s_ub, acc_s_ub)
                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)
                T.tile.broadcast(acc_o_ub, m_i_prev, axis=1)
                T.tile.mul(acc_o, acc_o, acc_o_ub)

                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, workspace_2[cid, v_row : v_row + hm, :])

                # Cube: PV -> workspace_3
                T.copy(workspace_2[cid, :, :], acc_s_l1)
                T.copy(V[bz, kv_by, kv : kv + block_N, :], v_l1)
                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, workspace_3[cid, :, :])

                # Vector: accumulate acc_o
                T.copy(workspace_3[cid, v_row : v_row + hm, :], acc_o_ub)
                T.tile.add(acc_o, acc_o, acc_o_ub)

            # Attention Sink: sumexp += exp(sink - m_i)
            T.tile.sub(sink_exp_ub, sink_ub, m_i)
            T.tile.exp(sink_exp_ub, sink_exp_ub)
            T.tile.add(sumexp, sumexp, sink_exp_ub)

            T.tile.broadcast(acc_o_ub, sumexp, axis=1)
            T.tile.div(acc_o, acc_o, acc_o_ub)

            T.copy(acc_o, acc_o_half)
            T.copy(acc_o_half, Output[bz, by, q_row + v_row : q_row + v_row + hm, :])

            T.tile.ln(sumexp, sumexp)
            T.tile.add(sumexp, sumexp, m_i)
            T.copy(sumexp, lse[bz, by, q_row + v_row : q_row + v_row + hm])

    return main


# ============================================================================
# Kernel 2 (preprocess merged): flashattn_bwd
#   Merges qk_softmax + dv_dp_ds + dk_dq + preprocess (Delta + dSink) into one kernel.
#   Phase 0 (Vector): Delta = sum(O * dO, dim=-1) — full block_M computation.
#   Phase 1-5 (KV loop): 5 GEMM + softmax + dS, per-iter C-V-C-V-C.
#   Phase 6 (Vector): dSink = -exp(sink - lse) * Delta — uses delta_ub from Phase 0.
#   alloc_shared/fragment replace alloc_L1/ub/L0C; 6 GM workspace buffers
#   eliminated via on-chip direct T.copy(fragment, shared) (p_delta/ds_delta
#   now also UB->L1 direct, no GM roundtrip).
#   Developer + combineCV (4 True): zero T.Scope, zero manual flag, threads=1.
# ============================================================================


@tilelang.jit(pass_configs=_hybrid_pass_configs)
def flashattn_bwd(batch, heads, seq_len, dim_qk, dim_v, window_size, block_M, block_N, groups=1):
    """Single-kernel backward (on-chip, preprocess merged).

    preprocess (Delta + dSink) merged into Phase 0 + Phase 6:
      - Phase 0 (before KV loop): Delta = sum(O * dO, dim=-1) in Vector scope.
        Full block_M computation. Delta stays in delta_ub (UB) for Phase 4 dS
        compute — no GM roundtrip. Also written to Delta_out (GM) for verification.
      - Phase 6 (after KV loop): dSink = -exp(sink - lse) * Delta in Vector scope.
        Uses delta_ub (Phase 0) + lse_ub (loaded at start). Outputs dSinks to GM.
      - New inputs: O (fwd output), Sinks (sink values).
      - New outputs: Delta_out (fp32), dSinks (fp32).
      - Removed: Delta input (now computed internally).

    6 GM workspace buffers (s/p/dp/ds/p_delta/ds_delta) eliminated via on-chip
    direct T.copy(fragment, shared). p_delta_half and ds_delta_half are
    alloc_shared [block_M, block_N] fp16 (8KB) — same size as p_half which
    already does UB->L1 direct, so they can too (no UB overflow).
    alloc_shared/fragment replace alloc_L1/ub/L0C. threads=1, no vid.

    Compensated GEMM: p + p_delta (1:1 each) for GEMM2corr, ds + ds_delta
    (1:1 each) for GEMM4corr, via L0C init=False accumulation.
    """
    assert seq_len % block_M == 0
    assert seq_len % block_N == 0
    assert dim_qk % 128 == 0, f"dim_qk must be multiple of 128 (got {dim_qk}); otherwise dim_qk_padded != dim_qk causes shape mismatch"
    sm_scale = (1.0 / dim_qk) ** 0.5
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"
    dim_qk_padded = ((dim_qk + 127) // 128) * 128

    q_shape = [batch, heads, seq_len, dim_qk_padded]
    k_shape = [batch, head_kv, seq_len, dim_qk_padded]
    v_shape = [batch, head_kv, seq_len, dim_v]
    do_shape = [batch, heads, seq_len, dim_v]
    o_shape = [batch, heads, seq_len, dim_v]  # fwd output for Delta
    dk_shape = [batch, head_kv, seq_len, dim_qk_padded]
    dv_shape = [batch, head_kv, seq_len, dim_v]
    dq_shape = [batch, heads, seq_len, dim_qk_padded]
    lse_shape = [batch, heads, seq_len]
    delta_shape = [batch, heads, seq_len]
    sinks_shape = [heads]  # sink values for dSink
    bwd_block_num = (seq_len // block_M) * heads * batch

    if window_size is not None:
        assert window_size % block_N == 0
    window_eff = window_size if window_size is not None else seq_len * 2

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        dO: T.Tensor(do_shape, dtype),  # type: ignore
        O: T.Tensor(o_shape, dtype),  # type: ignore  # fwd output for Delta
        lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        Sinks: T.Tensor(sinks_shape, dtype),  # type: ignore  # sink values for dSink
        Delta_out: T.Tensor(delta_shape, accum_dtype),  # type: ignore  # Delta output
        dSinks: T.Tensor(delta_shape, accum_dtype),  # type: ignore  # dSinks output
        dQ: T.Tensor(dq_shape, dtype),  # type: ignore
        dK: T.Tensor(dk_shape, accum_dtype),  # type: ignore
        dV: T.Tensor(dv_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(bwd_block_num, threads=1, is_npu=True) as (cid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch
            kv_by = by // groups

            loop_st = T.max(0, (bx * block_M - window_eff) // block_N)
            loop_ed = T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N))

            q_row = bx * block_M

            # === L1/UB buffers (alloc_shared, compiler maps to L1 or UB) — fp16 ===
            q_l1 = T.alloc_shared([block_M, dim_qk_padded], dtype)
            k_l1 = T.alloc_shared([block_N, dim_qk_padded], dtype)
            do_l1 = T.alloc_shared([block_M, dim_v], dtype)
            v_l1 = T.alloc_shared([block_N, dim_v], dtype)
            p_l1 = T.alloc_shared([block_M, block_N], dtype)
            p_delta_l1 = T.alloc_shared([block_M, block_N], dtype)
            ds_l1 = T.alloc_shared([block_M, block_N], dtype)
            ds_delta_l1 = T.alloc_shared([block_M, block_N], dtype)

            # === L0C buffers (alloc_fragment) — fp32 ===
            l0c_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            l0c_dp = T.alloc_fragment([block_M, block_N], accum_dtype)
            l0c_dv = T.alloc_fragment([block_N, dim_v], accum_dtype)
            l0c_dk = T.alloc_fragment([block_N, dim_qk_padded], accum_dtype)
            l0c_dq = T.alloc_fragment([block_M, dim_qk_padded], accum_dtype)

            # === UB buffers (alloc_shared, Vector side) — s_ub retains P_fp32 across phases ===
            s_ub = T.alloc_shared([block_M, block_N], accum_dtype)
            lse_ub = T.alloc_shared([block_M], accum_dtype)
            lse_2d = T.alloc_shared([block_M, block_N], accum_dtype)
            col_pos = T.alloc_shared([block_N], accum_dtype)
            row_1d = T.alloc_shared([block_M], accum_dtype)
            row_2d = T.alloc_shared([block_M, block_N], accum_dtype)
            mask_2d = T.alloc_shared([block_M, block_N], accum_dtype)
            p_half = T.alloc_shared([block_M, block_N], dtype)
            p_delta_half = T.alloc_shared([block_M, block_N], dtype)
            p_delta_ub = T.alloc_shared([block_M, block_N], accum_dtype)
            dp_ub = T.alloc_shared([block_M, block_N], accum_dtype)
            delta_ub = T.alloc_shared([block_M], accum_dtype)
            delta_2d = T.alloc_shared([block_M, block_N], accum_dtype)
            ds_half = T.alloc_shared([block_M, block_N], dtype)
            ds_delta_half = T.alloc_shared([block_M, block_N], dtype)
            ds_delta_ub = T.alloc_shared([block_M, block_N], accum_dtype)
            ds_rec_ub = T.alloc_shared([block_M, block_N], accum_dtype)

            # === Phase 0 buffers (Delta = sum(O * dO, dim=-1)) — Vector scope ===
            # Full block_M computation. Use alloc_ub to force UB placement and avoid
            # memory planner overlap issues with KV loop's alloc_shared buffers.
            o_ub_p0 = T.alloc_ub([block_M, dim_v], dtype)  # fp16, 64*128*2 = 16KB
            do_ub_p0 = T.alloc_ub([block_M, dim_v], dtype)  # fp16, 16KB
            prod_ub_p0 = T.alloc_ub([block_M, dim_v], accum_dtype)  # fp32, 32KB
            do_fp32_p0 = T.alloc_ub([block_M, dim_v], accum_dtype)  # fp32, 32KB
            sum_ub_p0 = T.alloc_ub([block_M, dim_v], accum_dtype)  # fp32, 32KB

            # === Phase 6 buffers (dSink = -exp(sink - lse) * Delta) — Vector scope ===
            sink_val_ub = T.alloc_ub([block_M], accum_dtype)  # fp32, 256B
            sink_exp_ub = T.alloc_ub([block_M], accum_dtype)  # fp32, 256B
            sink_scalar = T.alloc_ub([1], dtype)  # fp16, 2B

            # === Phase 0 (Vector): Delta = sum(O * dO, dim=-1) ===
            # Computed FIRST (before Q/dO/lse loads) to isolate Phase 0 flag assignments
            # from KV loop flag assignments. Full block_M computation.
            T.copy(O[bz, by, q_row : q_row + block_M, :], o_ub_p0)
            T.copy(dO[bz, by, q_row : q_row + block_M, :], do_ub_p0)
            T.copy(o_ub_p0, prod_ub_p0)  # fp16 -> fp32 auto-cast
            T.copy(do_ub_p0, do_fp32_p0)  # fp16 -> fp32 auto-cast
            T.tile.mul(sum_ub_p0, prod_ub_p0, do_fp32_p0)
            T.reduce_sum(sum_ub_p0, delta_ub, dim=-1)
            T.copy(delta_ub, Delta_out[bz, by, q_row : q_row + block_M])

            # === Loop-invariant loads ===
            T.copy(Q[bz, by, bx * block_M : bx * block_M + block_M, :], q_l1)
            T.copy(dO[bz, by, bx * block_M : bx * block_M + block_M, :], do_l1)
            T.copy(lse[bz, by, q_row : q_row + block_M], lse_ub)
            T.tile.arith_progression(row_1d, q_row, 1, block_M)

            for k in T.serial(loop_st, loop_ed):
                kv = k * block_N

                # === Phase 1 (Cube): GEMM1 S = Q @ K^T ===
                T.copy(K[bz, kv_by, kv : kv + block_N, :], k_l1)
                T.gemm_v0(q_l1, k_l1, l0c_s, transpose_B=True, init=True)

                # === Phase 2 (Vector): softmax P + p_delta ===
                T.copy(l0c_s, s_ub)  # on-chip direct
                T.tile.fill(lse_2d, 0.0)
                T.tile.axpy(lse_2d, s_ub, sm_scale)
                T.tile.broadcast(s_ub, lse_ub, axis=1)
                T.tile.sub(lse_2d, lse_2d, s_ub)
                T.tile.exp(s_ub, lse_2d)

                # Mask (causal + window) — unified path using window_eff.
                # Always use the window mask branch (even for w=None, where
                # window_eff=seq_len*2 makes the window mask all-True). This avoids
                # a compiler flag assignment bug in combineCV mode where the
                # else-branch (w=None) has incorrect flag synchronization when
                # Phase 0 is present before the KV loop.
                T.tile.arith_progression(col_pos, kv, 1, block_N)
                T.tile.broadcast(lse_2d, col_pos, axis=0)
                T.tile.broadcast(row_2d, row_1d, axis=1)
                T.tile.compare(mask_2d, lse_2d, row_2d, "LE")
                T.tile.sub(row_2d, row_2d, window_eff)
                T.tile.compare(p_delta_ub, lse_2d, row_2d, "GT")
                T.tile.bitwise_and(mask_2d, mask_2d, p_delta_ub)
                T.tile.select(s_ub, mask_2d, s_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")

                # P_fp32 in s_ub (retained for Phase 4!)
                T.copy(s_ub, p_half)
                T.copy(p_half, p_l1)  # on-chip direct
                T.copy(p_half, p_delta_ub)
                T.tile.sub(p_delta_ub, s_ub, p_delta_ub)
                T.copy(p_delta_ub, p_delta_half)
                # p_delta_half UB->L1 direct

                # === Phase 3 (Cube): GEMM2 dV + GEMM3 dP ===
                T.gemm_v0(p_l1, do_l1, l0c_dv, transpose_A=True, init=True)
                T.copy(p_delta_half, p_delta_l1)  # on-chip direct
                T.gemm_v0(p_delta_l1, do_l1, l0c_dv, transpose_A=True, init=False)
                T.tile.atomic_add(dV[bz, kv_by, kv : kv + block_N, :], l0c_dv)
                T.copy(V[bz, kv_by, kv : kv + block_N, :], v_l1)
                T.gemm_v0(do_l1, v_l1, l0c_dp, transpose_B=True, init=True)

                # === Phase 4 (Vector): dS compute ===
                T.copy(l0c_dp, dp_ub)  # on-chip direct
                # s_ub still has P_fp32 from Phase 2 (retained in UB!)
                T.tile.broadcast(delta_2d, delta_ub, axis=1)
                T.tile.sub(dp_ub, dp_ub, delta_2d)
                T.tile.mul(s_ub, s_ub, dp_ub)
                T.tile.mul(s_ub, s_ub, sm_scale)

                # Mask (causal + window) — unified path using window_eff.
                # Same unified path as Phase 2 (see comment above).
                T.tile.arith_progression(col_pos, kv, 1, block_N)
                T.tile.broadcast(delta_2d, col_pos, axis=0)
                T.tile.broadcast(row_2d, row_1d, axis=1)
                T.tile.compare(mask_2d, delta_2d, row_2d, "LE")
                T.tile.sub(row_2d, row_2d, window_eff)
                T.tile.compare(ds_delta_ub, delta_2d, row_2d, "GT")
                T.tile.bitwise_and(mask_2d, mask_2d, ds_delta_ub)
                T.tile.select(s_ub, mask_2d, s_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")

                # dS_fp32 in s_ub
                T.copy(s_ub, ds_half)
                T.copy(ds_half, ds_l1)  # on-chip direct
                T.copy(ds_half, ds_rec_ub)
                T.tile.sub(ds_delta_ub, s_ub, ds_rec_ub)
                T.copy(ds_delta_ub, ds_delta_half)
                # ds_delta_half UB->L1 direct

                # === Phase 5 (Cube): GEMM4 dK + GEMM5 dQ ===
                T.gemm_v0(ds_l1, q_l1, l0c_dk, transpose_A=True, init=True)
                T.copy(ds_delta_half, ds_delta_l1)  # on-chip direct
                T.gemm_v0(ds_delta_l1, q_l1, l0c_dk, transpose_A=True, init=False)
                T.tile.atomic_add(dK[bz, kv_by, kv : kv + block_N, :], l0c_dk)
                T.copy(K[bz, kv_by, kv : kv + block_N, :], k_l1)
                T.gemm_v0(ds_l1, k_l1, l0c_dq, init=(k == loop_st))
                T.gemm_v0(ds_delta_l1, k_l1, l0c_dq, init=False)

            # After loop: write dQ from L0C -> GM (fp32 -> fp16 auto-cast)
            T.copy(l0c_dq, dQ[bz, by, q_row : q_row + block_M, :])

            # === Phase 6 (Vector): dSink = -exp(sink - lse) * Delta ===
            # Uses delta_ub (computed in Phase 0, still in UB) + lse_ub (loaded at start).
            # Outputs dSinks to GM.
            T.copy(Sinks[by : by + 1], sink_scalar)
            T.tile.fill(sink_val_ub, sink_scalar[0])
            T.tile.sub(sink_exp_ub, sink_val_ub, lse_ub)
            T.tile.exp(sink_exp_ub, sink_exp_ub)
            T.tile.mul(sink_exp_ub, sink_exp_ub, delta_ub)
            T.tile.mul(sink_exp_ub, sink_exp_ub, -1.0)
            T.copy(sink_exp_ub, dSinks[bz, by, q_row : q_row + block_M])

    return main


# ============================================================================
# Autograd Function (end-to-end wrapper) — 3-kernel call chain (1 host sync)
# ============================================================================


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, window_size, groups):
        def maybe_contiguous(x):
            return x if x.stride(-1) == 1 else x.contiguous()

        q, k, v, sinks = [maybe_contiguous(x) for x in (q, k, v, sinks)]
        B, H, N, D = q.shape
        block_M, block_N = 64, 64

        fwd_mod = flashattn_fwd(B, H, N, D, groups, window_size, block_M, block_N)
        o, lse = fwd_mod(q, k, v, sinks)

        ctx.save_for_backward(q, k, v, sinks, o, lse)
        ctx.window_size = window_size
        ctx.groups = groups
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, sinks, o, lse = ctx.saved_tensors
        B, H, N, D = q.shape
        groups = ctx.groups
        window_size = ctx.window_size
        H_kv = H // groups
        block_M, block_N = 64, 64
        dim_qk_padded = ((D + 127) // 128) * 128

        # preprocess merged into bwd kernel — no separate preprocess call, no host sync.
        # bwd kernel Phase 0 computes Delta internally, Phase 6 computes dSinks.

        # Output tensors (bwd kernel outputs: Delta_out, dSinks, dQ, dK, dV)
        delta_out = torch.zeros(B, H, N, dtype=torch.float32, device=q.device)
        dSinks = torch.zeros(B, H, N, dtype=torch.float32, device=q.device)
        dQ = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float16, device=q.device)
        dK = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float32, device=q.device)
        dV = torch.zeros(B, H_kv, N, D, dtype=torch.float32, device=q.device)

        # Kernel 2: single flashattn_bwd (Phase 0 Delta + KV-loop + Phase 6 dSink)
        # All 6 GM workspace buffers eliminated (on-chip direct UB->L1).
        # preprocess merged — O and Sinks now passed as inputs, Delta_out/dSinks as outputs.
        bwd_args = (B, H, N, D, D, window_size, block_M, block_N, groups)
        bwd_mod = flashattn_bwd(*bwd_args)
        bwd_mod(
            q,
            k,
            v,
            do,
            o,
            lse,
            sinks,
            delta_out,
            dSinks,
            dQ,
            dK,
            dV,
        )
        torch.npu.synchronize()

        # postprocess kernel removed — host .half() cast (dQ already fp16 from bwd)
        dQ = dQ[..., :D]
        dK = dK[..., :D].half()
        dV = dV[..., :D].half()

        # dSinks: host sum over B and N -> [H] fp32 (dSinks output by bwd kernel Phase 6)
        dsinks_sum = dSinks.cpu().sum(0).sum(1)

        return dQ, dK, dV, dsinks_sum, None, None


attention = _attention.apply


# ============================================================================
# Main: smoke test (CI self-contained — verifies kernel runs and output shape)
# ============================================================================


if __name__ == "__main__":
    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(42)

    B, H, groups, N, D = 1, 4, 2, 128, 128
    H_kv = H // groups

    Q = torch.randn(B, H, N, D, dtype=torch.float16, device="npu")
    K = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="npu")
    V = torch.randn_like(K)
    sinks = torch.randn(H, dtype=torch.float16, device="npu")
    dO = torch.randn_like(Q)

    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)

    O = attention(Q, K, V, sinks, None, groups)
    O.backward(dO)
    torch.npu.synchronize()

    assert O.shape == (B, H, N, D)
    assert Q.grad is not None
    assert K.grad is not None
    assert V.grad is not None
    assert torch.isfinite(O).all()
    assert torch.isfinite(Q.grad).all()

    print("Test Passed!")
