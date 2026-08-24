"""GQA Sink Attention (BHSD) for Ascend NPU — combineCV single-kernel bwd merge (rev2).

Layout: BHSD (Batch, Heads, SeqLen, Dim). Supports GQA (grouped-query attention),
an attention sink token, and an optional sliding window mask. fp16, Developer mode.

Architecture (ref: DESIGN.md rev2 — single-kernel bwd merge):
  Kernel 1 (fwd, Hybrid):           Forward (online softmax + sink + window) -> O, lse
  Kernel 2 (preprocess, Vector):    Delta = sum(O*dO) + dSink = -exp(sink-lse)*Delta
  Kernel 3 (bwd, Hybrid):           **Single kernel** merges qk_softmax + dv_dp_ds + dk_dq
  Kernel 4 (postprocess, Vector):   dK+dV fp32->fp16 dual-output cast

4 kernels total (down from rev0's 6):
  1. flashattn_fwd:               Forward (online softmax + sink + window) -> O, lse
  2. flashattn_bwd_preprocess:    Delta = sum(O*dO) + dSink = -exp(sink-lse)*Delta
  3. flashattn_bwd:         Single kernel: 5 GEMM + softmax + dS, per-iter C-V-C-V-C
  4. flashattn_bwd_postprocess: dK+dV fp32 -> fp16 (dual-output, dQ direct fp16 from bwd)

combineCV sync mechanism (DESIGN.md rev2 §4.3/§9.3):
  - 6 intra-kernel CV transfer buffers named ``workspace_*`` (contain "workspace"
    substring) → combineCV FetchWorkspaceName collects sync points →
    auto set_flag/wait_flag insertion (ascend_combinecv.cc:190-202).
  - Per-iteration C-V-C-V-C: each workspace is 1 write + 1 read per iteration.
  - P_fp32 retained in UB (s_ub), NOT written to GM — eliminates workspace_p_fp32.
  - CompGEMM dual-read: workspace_p + workspace_p_delta (1:1 each), workspace_ds +
    workspace_ds_delta (1:1 each).

Key design decisions (preserved from rev0):
  - No T.Scope, no set_flag/wait_flag, no cross_flag, no barrier_all — pure
    AUTO_CV_SYNC + AUTO_SYNC automatic synchronization (Developer mode).
  - Compensated GEMM: fp16 GEMM result corrected by a second GEMM on the fp16
    quantization residual (p_delta for GEMM2corr, ds_delta for GEMM4corr).
  - Single loop (no split-loop mask skip) — simplifies combineCV sync validation.
  - dQ written as fp16 directly from bwd kernel (L0C fp32 -> GM fp16 auto-cast).
  - Precision: 169-line standard (atol=6.10e-5, rtol=1.95e-3 for fp16).
"""

import sys

import tilelang
import torch
from tilelang import language as T

# ============================================================================
# pass_configs (D11: 4 explicit keys per config)
# ============================================================================

# Hybrid mode for fwd + single-kernel bwd (flashattn_bwd) — AUTO_CV_COMBINE/SYNC — AUTO_CV_COMBINE/SYNC
# needed for L0C->GM->UB two-hop accumulation pattern and intra-kernel CV transfer.
_hybrid_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Vector mode for preprocess / postprocess (pure element-wise, no GEMM).
_vector_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# ============================================================================
# Kernel 1: Forward (online softmax + attention sink + sliding window) — unchanged
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
# Kernel 2 (merged): flashattn_bwd_preprocess
#   Delta = sum(O * dO, dim=-1)  +  dSink = -exp(sink - lse) * Delta
#   Pure Vector. Merges baseline preprocess + dsink (方案 E).
#   Delta computed in UB, used directly for dSink (no GM read-back), then written
#   to GM for merged_dv_dp_ds consumer.
# ============================================================================


@tilelang.jit(out_idx=[2, 3], pass_configs=_vector_pass_configs)
def flashattn_bwd_preprocess(batch, heads, seq_len, dim, blk=32):
    """Merged preprocess + dsink: Delta = sum(O*dO) and dSink = -exp(sink-lse)*Delta.

    Pure Vector kernel. Each block handles ``blk // 2`` rows of one (batch, head).
    Delta is computed in UB and used directly for dSink (no inter-kernel GM
    read-back). Both Delta and dSinks are written to GM as outputs.

    Outputs:
      Delta [B, H, N] fp32 — consumed by merged_dv_dp_ds
      dsinks [B, H, N] fp32 — host sums to [H] for final dSinks
    """
    assert seq_len % blk == 0
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, heads, seq_len, dim]
    delta_shape = [batch, heads, seq_len]
    block_num = heads * (seq_len // blk) * batch

    @T.prim_func
    def main(
        O: T.Tensor(shape, dtype),  # type: ignore
        dO: T.Tensor(shape, dtype),  # type: ignore
        Delta: T.Tensor(delta_shape, accum_dtype),  # type: ignore
        dsinks: T.Tensor(delta_shape, accum_dtype),  # type: ignore
        Sinks: T.Tensor([heads], dtype),  # type: ignore
        lse: T.Tensor(delta_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            by = cid % (seq_len // blk)
            bx = cid // (seq_len // blk) % heads
            bz = cid // (seq_len // blk) // heads % batch

            o_ub = T.alloc_ub([blk // 2, dim], dtype)
            do_ub = T.alloc_ub([blk // 2, dim], dtype)
            sum_ub = T.alloc_ub([blk // 2, dim], accum_dtype)
            prod_ub = T.alloc_ub([blk // 2, dim], accum_dtype)
            do_fp32 = T.alloc_ub([blk // 2, dim], accum_dtype)
            delta_ub = T.alloc_ub([blk // 2], accum_dtype)
            lse_ub = T.alloc_ub([blk // 2], accum_dtype)
            sink_exp_ub = T.alloc_ub([blk // 2], accum_dtype)
            sink_val_ub = T.alloc_ub([blk // 2], accum_dtype)
            sink_scalar = T.alloc_ub([1], dtype)

            row_st = by * blk + vid * blk // 2
            row_ed = row_st + blk // 2

            # --- Delta = sum(O * dO, dim=-1) ---
            T.copy(O[bz, bx, row_st:row_ed, :], o_ub)
            T.copy(dO[bz, bx, row_st:row_ed, :], do_ub)
            T.copy(o_ub, prod_ub)
            T.copy(do_ub, do_fp32)
            T.tile.mul(sum_ub, prod_ub, do_fp32)
            T.reduce_sum(sum_ub, delta_ub, dim=-1)
            T.copy(delta_ub, Delta[bz, bx, row_st:row_ed])

            # --- dSink = -exp(sink - lse) * Delta (uses delta_ub directly) ---
            T.copy(Sinks[bx : bx + 1], sink_scalar)
            T.tile.fill(sink_val_ub, sink_scalar[0])
            T.copy(lse[bz, bx, row_st:row_ed], lse_ub)
            T.tile.sub(sink_exp_ub, sink_val_ub, lse_ub)
            T.tile.exp(sink_exp_ub, sink_exp_ub)
            T.tile.mul(sink_exp_ub, sink_exp_ub, delta_ub)
            T.tile.mul(sink_exp_ub, sink_exp_ub, -1.0)
            T.copy(sink_exp_ub, dsinks[bz, bx, row_st:row_ed])

    return main


# ============================================================================
# Kernel 3 (rev2 single-kernel merge): flashattn_bwd
#   Merges rev0's qk_softmax (#3) + dv_dp_ds (#4) + dk_dq (#5) into one kernel.
#   Per-iteration C-V-C-V-C structure with 6 intra-kernel workspace_* buffers.
#   P_fp32 retained in UB (s_ub), NOT written to GM.
#   Developer + combineCV (4 True): zero T.Scope, zero manual flag.
# ============================================================================


@tilelang.jit(pass_configs=_hybrid_pass_configs)
def flashattn_bwd(batch, heads, seq_len, dim_qk, dim_v, window_size, block_M, block_N, groups=1):
    """Single-kernel bwd: merges qk_softmax + dv_dp_ds + dk_dq.

    Per-iteration C-V-C-V-C (5 GEMM + softmax + dS) with 6 intra-kernel
    workspace_* buffers for combineCV auto-sync. P_fp32 retained in UB.

    Compensated GEMM: workspace_p + workspace_p_delta (1:1 each) for GEMM2corr,
    workspace_ds + workspace_ds_delta (1:1 each) for GEMM4corr.
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
    dk_shape = [batch, head_kv, seq_len, dim_qk_padded]
    dv_shape = [batch, head_kv, seq_len, dim_v]
    dq_shape = [batch, heads, seq_len, dim_qk_padded]
    lse_shape = [batch, heads, seq_len]
    delta_shape = [batch, heads, seq_len]
    bwd_block_num = (seq_len // block_M) * heads * batch

    if window_size is not None:
        assert window_size % block_N == 0
    window_eff = window_size if window_size is not None else seq_len * 2

    ws_2d = [bwd_block_num, block_M, block_N]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        dO: T.Tensor(do_shape, dtype),  # type: ignore
        lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        Delta: T.Tensor(delta_shape, accum_dtype),  # type: ignore
        dQ: T.Tensor(dq_shape, dtype),  # type: ignore
        dK: T.Tensor(dk_shape, accum_dtype),  # type: ignore
        dV: T.Tensor(dv_shape, accum_dtype),  # type: ignore
        workspace_s: T.Tensor(ws_2d, accum_dtype),  # type: ignore
        workspace_p: T.Tensor(ws_2d, dtype),  # type: ignore
        workspace_p_delta: T.Tensor(ws_2d, dtype),  # type: ignore
        workspace_dp: T.Tensor(ws_2d, accum_dtype),  # type: ignore
        workspace_ds: T.Tensor(ws_2d, dtype),  # type: ignore
        workspace_ds_delta: T.Tensor(ws_2d, dtype),  # type: ignore
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch
            kv_by = by // groups

            loop_st = T.max(0, (bx * block_M - window_eff) // block_N)
            loop_ed = T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N))

            q_row = bx * block_M

            # === L1 buffers (Cube side) — fp16 ===
            q_l1 = T.alloc_L1([block_M, dim_qk_padded], dtype)
            k_l1 = T.alloc_L1([block_N, dim_qk_padded], dtype)
            do_l1 = T.alloc_L1([block_M, dim_v], dtype)
            v_l1 = T.alloc_L1([block_N, dim_v], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)
            p_delta_l1 = T.alloc_L1([block_M, block_N], dtype)
            ds_l1 = T.alloc_L1([block_M, block_N], dtype)
            ds_delta_l1 = T.alloc_L1([block_M, block_N], dtype)

            # === L0C buffers (Cube side) — fp32 ===
            l0c_s = T.alloc_L0C([block_M, block_N], accum_dtype)
            l0c_dp = T.alloc_L0C([block_M, block_N], accum_dtype)
            l0c_dv = T.alloc_L0C([block_N, dim_v], accum_dtype)
            l0c_dk = T.alloc_L0C([block_N, dim_qk_padded], accum_dtype)
            l0c_dq = T.alloc_L0C([block_M, dim_qk_padded], accum_dtype)

            # === UB buffers (Vector side) — s_ub retains P_fp32 across phases ===
            s_ub = T.alloc_ub([block_M, block_N], accum_dtype)
            lse_ub = T.alloc_ub([block_M], accum_dtype)
            lse_2d = T.alloc_ub([block_M, block_N], accum_dtype)
            col_pos = T.alloc_ub([block_N], accum_dtype)
            row_1d = T.alloc_ub([block_M], accum_dtype)
            row_2d = T.alloc_ub([block_M, block_N], accum_dtype)
            mask_2d = T.alloc_ub([block_M, block_N], accum_dtype)
            p_half = T.alloc_ub([block_M, block_N], dtype)
            p_delta_half = T.alloc_ub([block_M, block_N], dtype)
            p_delta_ub = T.alloc_ub([block_M, block_N], accum_dtype)
            dp_ub = T.alloc_ub([block_M, block_N], accum_dtype)
            delta_ub = T.alloc_ub([block_M], accum_dtype)
            delta_2d = T.alloc_ub([block_M, block_N], accum_dtype)
            ds_half = T.alloc_ub([block_M, block_N], dtype)
            ds_delta_half = T.alloc_ub([block_M, block_N], dtype)
            ds_delta_ub = T.alloc_ub([block_M, block_N], accum_dtype)
            ds_rec_ub = T.alloc_ub([block_M, block_N], accum_dtype)

            # === Loop-invariant loads ===
            T.copy(Q[bz, by, bx * block_M : bx * block_M + block_M, :], q_l1)
            T.copy(dO[bz, by, bx * block_M : bx * block_M + block_M, :], do_l1)
            T.copy(Delta[bz, by, q_row : q_row + block_M], delta_ub)
            T.copy(lse[bz, by, q_row : q_row + block_M], lse_ub)
            T.tile.arith_progression(row_1d, q_row, 1, block_M)

            for k in T.serial(loop_st, loop_ed):
                kv = k * block_N

                # === Phase 1 (Cube): GEMM1 S = Q @ K^T ===
                T.copy(K[bz, kv_by, kv : kv + block_N, :], k_l1)
                T.gemm_v0(q_l1, k_l1, l0c_s, transpose_B=True, init=True)
                T.copy(l0c_s, workspace_s[cid, :, :])

                # === Phase 2 (Vector): softmax P + p_delta ===
                T.copy(workspace_s[cid, :, :], s_ub)
                T.tile.fill(lse_2d, 0.0)
                T.tile.axpy(lse_2d, s_ub, sm_scale)
                T.tile.broadcast(s_ub, lse_ub, axis=1)
                T.tile.sub(lse_2d, lse_2d, s_ub)
                T.tile.exp(s_ub, lse_2d)

                # Mask (causal + optional window)
                T.tile.arith_progression(col_pos, kv, 1, block_N)
                T.tile.broadcast(lse_2d, col_pos, axis=0)
                T.tile.broadcast(row_2d, row_1d, axis=1)
                if window_size is not None:
                    T.tile.compare(mask_2d, lse_2d, row_2d, "LE")
                    T.tile.sub(row_2d, row_2d, window_size)
                    T.tile.compare(p_delta_ub, lse_2d, row_2d, "GT")
                    T.tile.bitwise_and(mask_2d, mask_2d, p_delta_ub)
                else:
                    T.tile.compare(mask_2d, lse_2d, row_2d, "LE")
                T.tile.select(s_ub, mask_2d, s_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")

                # P_fp32 in s_ub (retained for Phase 4!)
                T.copy(s_ub, p_half)
                T.copy(p_half, workspace_p[cid, :, :])
                T.copy(p_half, p_delta_ub)
                T.tile.sub(p_delta_ub, s_ub, p_delta_ub)
                T.copy(p_delta_ub, p_delta_half)
                T.copy(p_delta_half, workspace_p_delta[cid, :, :])

                # === Phase 3 (Cube): GEMM2 dV + GEMM3 dP ===
                T.copy(workspace_p[cid, :, :], p_l1)
                T.gemm_v0(p_l1, do_l1, l0c_dv, transpose_A=True, init=True)
                T.copy(workspace_p_delta[cid, :, :], p_delta_l1)
                T.gemm_v0(p_delta_l1, do_l1, l0c_dv, transpose_A=True, init=False)
                T.tile.atomic_add(dV[bz, kv_by, kv : kv + block_N, :], l0c_dv)
                T.copy(V[bz, kv_by, kv : kv + block_N, :], v_l1)
                T.gemm_v0(do_l1, v_l1, l0c_dp, transpose_B=True, init=True)
                T.copy(l0c_dp, workspace_dp[cid, :, :])

                # === Phase 4 (Vector): dS compute ===
                T.copy(workspace_dp[cid, :, :], dp_ub)
                # s_ub still has P_fp32 from Phase 2 (retained in UB!)
                T.tile.broadcast(delta_2d, delta_ub, axis=1)
                T.tile.sub(dp_ub, dp_ub, delta_2d)
                T.tile.mul(s_ub, s_ub, dp_ub)
                T.tile.mul(s_ub, s_ub, sm_scale)

                # Mask (causal + optional window)
                T.tile.arith_progression(col_pos, kv, 1, block_N)
                T.tile.broadcast(delta_2d, col_pos, axis=0)
                T.tile.broadcast(row_2d, row_1d, axis=1)
                if window_size is not None:
                    T.tile.compare(mask_2d, delta_2d, row_2d, "LE")
                    T.tile.sub(row_2d, row_2d, window_size)
                    T.tile.compare(ds_delta_ub, delta_2d, row_2d, "GT")
                    T.tile.bitwise_and(mask_2d, mask_2d, ds_delta_ub)
                else:
                    T.tile.compare(mask_2d, delta_2d, row_2d, "LE")
                T.tile.select(s_ub, mask_2d, s_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")

                # dS_fp32 in s_ub
                T.copy(s_ub, ds_half)
                T.copy(ds_half, workspace_ds[cid, :, :])
                T.copy(ds_half, ds_rec_ub)
                T.tile.sub(ds_delta_ub, s_ub, ds_rec_ub)
                T.copy(ds_delta_ub, ds_delta_half)
                T.copy(ds_delta_half, workspace_ds_delta[cid, :, :])

                # === Phase 5 (Cube): GEMM4 dK + GEMM5 dQ ===
                T.copy(workspace_ds[cid, :, :], ds_l1)
                T.gemm_v0(ds_l1, q_l1, l0c_dk, transpose_A=True, init=True)
                T.copy(workspace_ds_delta[cid, :, :], ds_delta_l1)
                T.gemm_v0(ds_delta_l1, q_l1, l0c_dk, transpose_A=True, init=False)
                T.tile.atomic_add(dK[bz, kv_by, kv : kv + block_N, :], l0c_dk)
                T.copy(K[bz, kv_by, kv : kv + block_N, :], k_l1)
                T.gemm_v0(ds_l1, k_l1, l0c_dq, init=(k == loop_st))
                T.gemm_v0(ds_delta_l1, k_l1, l0c_dq, init=False)

            # After loop: write dQ from L0C -> GM (fp32 -> fp16 auto-cast)
            T.copy(l0c_dq, dQ[bz, by, q_row : q_row + block_M, :])

    return main


# ============================================================================
# Kernel 6 (merged): flashattn_bwd_postprocess
#   dK fp32 -> fp16  +  dV fp32 -> fp16  (dual-output, single kernel)
#   Pure Vector. Merges baseline 2× postprocess calls (方案 F').
#   dQ does not need this — bwd writes dQ as fp16 directly.
# ============================================================================


@tilelang.jit(out_idx=[2, 3], pass_configs=_vector_pass_configs)
def flashattn_bwd_postprocess(batch, heads, seq_len, dim_k, dim_v, blk=64):
    """Merged postprocess: cast dK and dV from fp32 GM to fp16 GM in one kernel.

    dQ does not need this — bwd writes dQ as fp16 directly (L0C fp32 -> GM fp16
    auto-cast). dK/dV stay fp32 in GM because they receive atomic_add from
    multiple Q blocks.

    Supports different dims for dK (dim_k, e.g. dim_qk_padded) and dV (dim_v, e.g. D).
    """
    assert seq_len % blk == 0
    dtype = "float16"
    accum_dtype = "float"
    dk_shape = [batch, heads, seq_len, dim_k]
    dv_shape = [batch, heads, seq_len, dim_v]
    block_num = (seq_len // blk) * heads * batch

    @T.prim_func
    def main(
        dK: T.Tensor(dk_shape, accum_dtype),  # type: ignore
        dV: T.Tensor(dv_shape, accum_dtype),  # type: ignore
        dK_out: T.Tensor(dk_shape, dtype),  # type: ignore
        dV_out: T.Tensor(dv_shape, dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // blk)
            by = cid // (seq_len // blk) % heads
            bz = cid // (seq_len // blk) // heads % batch

            row_st = bx * blk + vid * blk // 2
            row_ed = row_st + blk // 2

            # Cast dK fp32 -> fp16
            dk_ub = T.alloc_ub([blk // 2, dim_k], accum_dtype)
            dk_half = T.alloc_ub([blk // 2, dim_k], dtype)
            T.copy(dK[bz, by, row_st:row_ed, :], dk_ub)
            T.copy(dk_ub, dk_half)
            T.copy(dk_half, dK_out[bz, by, row_st:row_ed, :])

            # Cast dV fp32 -> fp16
            dv_ub = T.alloc_ub([blk // 2, dim_v], accum_dtype)
            dv_half = T.alloc_ub([blk // 2, dim_v], dtype)
            T.copy(dV[bz, by, row_st:row_ed, :], dv_ub)
            T.copy(dv_ub, dv_half)
            T.copy(dv_half, dV_out[bz, by, row_st:row_ed, :])

    return main


# ============================================================================
# Golden Reference (PyTorch CPU) — unchanged from baseline
# ============================================================================


def ref_fwd(Q, K, V, Sinks, window_size=None, groups=1):
    """Forward golden (CPU): GQA + Attention Sink + optional sliding window."""
    B, H, N, D = Q.shape
    sm_scale = 1.0 / D**0.5

    K_rep = K.float().repeat_interleave(groups, dim=1)
    V_rep = V.float().repeat_interleave(groups, dim=1)

    S = torch.matmul(Q.float(), K_rep.transpose(-2, -1)) * sm_scale

    pos_q = torch.arange(N, device=Q.device).float()
    pos_k = torch.arange(N, device=Q.device).float()
    causal_mask = pos_k[None, :] <= pos_q[:, None]
    if window_size is not None:
        window_mask = pos_k[None, :] > (pos_q[:, None] - window_size)
        mask = causal_mask & window_mask
    else:
        mask = causal_mask
    S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    m = S.max(dim=-1, keepdim=True).values
    sinks_b = Sinks.view(1, H, 1, 1).float()
    m_with_sink = torch.maximum(sinks_b, m)

    P = torch.exp(S - m_with_sink)
    sinks_exp = torch.exp(sinks_b - m_with_sink)
    normalizer = P.sum(dim=-1, keepdim=True) + sinks_exp
    P = P / normalizer

    O = torch.matmul(P, V_rep)
    return O.half()


def ref_bwd(Q, K, V, Sinks, dO, window_size=None, groups=1):
    """Backward golden (CPU autograd). Returns dQ, dK, dV (fp16), dSinks (fp32)."""
    Q_f = Q.float().requires_grad_(True)
    K_f = K.float().requires_grad_(True)
    V_f = V.float().requires_grad_(True)
    Sinks_f = Sinks.float().requires_grad_(True)

    B, H, N, D = Q_f.shape
    sm_scale = 1.0 / D**0.5

    K_rep = K_f.repeat_interleave(groups, dim=1)
    V_rep = V_f.repeat_interleave(groups, dim=1)

    S = torch.matmul(Q_f, K_rep.transpose(-2, -1)) * sm_scale

    pos_q = torch.arange(N, device=Q_f.device).float()
    pos_k = torch.arange(N, device=Q_f.device).float()
    causal_mask = pos_k[None, :] <= pos_q[:, None]
    if window_size is not None:
        window_mask = pos_k[None, :] > (pos_q[:, None] - window_size)
        mask = causal_mask & window_mask
    else:
        mask = causal_mask
    S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    m = S.max(dim=-1, keepdim=True).values
    sinks_b = Sinks_f.view(1, H, 1, 1)
    m_with_sink = torch.maximum(sinks_b, m)
    P = torch.exp(S - m_with_sink)
    sinks_exp = torch.exp(sinks_b - m_with_sink)
    normalizer = P.sum(dim=-1, keepdim=True) + sinks_exp
    P = P / normalizer

    O = torch.matmul(P, V_rep)
    O.backward(dO.float())

    return Q_f.grad.half(), K_f.grad.half(), V_f.grad.half(), Sinks_f.grad


# ============================================================================
# Autograd Function (end-to-end wrapper) — 4-kernel call chain (rev2)
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

        # Kernel 2: preprocess — Delta + dSink
        preprocess_mod = flashattn_bwd_preprocess(B, H, N, D, blk=32)
        delta, dsinks = preprocess_mod(o, do, sinks, lse)
        torch.npu.synchronize()

        # Output tensors
        dQ = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float16, device=q.device)
        dK = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float32, device=q.device)
        dV = torch.zeros(B, H_kv, N, D, dtype=torch.float32, device=q.device)

        # Intra-kernel CV transfer workspace (combineCV auto-sync manages visibility)
        bwd_block_num = H * (N // block_M) * B
        ws_2d_shape = (bwd_block_num, block_M, block_N)
        workspace_s = torch.empty(*ws_2d_shape, dtype=torch.float32, device=q.device)
        workspace_p = torch.empty(*ws_2d_shape, dtype=torch.float16, device=q.device)
        workspace_p_delta = torch.empty(*ws_2d_shape, dtype=torch.float16, device=q.device)
        workspace_dp = torch.empty(*ws_2d_shape, dtype=torch.float32, device=q.device)
        workspace_ds = torch.empty(*ws_2d_shape, dtype=torch.float16, device=q.device)
        workspace_ds_delta = torch.empty(*ws_2d_shape, dtype=torch.float16, device=q.device)

        # Kernel 3: single flashattn_bwd (merges qk_softmax + dv_dp_ds + dk_dq)
        bwd_args = (B, H, N, D, D, window_size, block_M, block_N, groups)
        bwd_mod = flashattn_bwd(*bwd_args)
        bwd_mod(
            q,
            k,
            v,
            do,
            lse,
            delta,
            dQ,
            dK,
            dV,
            workspace_s,
            workspace_p,
            workspace_p_delta,
            workspace_dp,
            workspace_ds,
            workspace_ds_delta,
        )
        torch.npu.synchronize()

        # Kernel 4: postprocess — dK+dV fp32 -> fp16 (dQ already fp16 from bwd)
        postprocess_mod = flashattn_bwd_postprocess(B, H_kv, N, dim_qk_padded, D, blk=64)
        dK, dV = postprocess_mod(dK, dV)
        dQ = dQ[..., :D]
        dK = dK[..., :D]

        # dSinks: host sum over B and N -> [H] fp32
        dsinks_sum = dsinks.cpu().sum(0).sum(1)

        return dQ, dK, dV, dsinks_sum, None, None


attention = _attention.apply


# ============================================================================
# Main: smoke test (CI §5.1 — self-contained gen/prepare/golden/check)
# ============================================================================


if __name__ == "__main__":
    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(42)

    # Minimal L0 config (fast smoke test)
    B, H, groups, N, D = 1, 4, 2, 128, 128
    window_size = None

    # Inputs on CPU (golden) + NPU copies (kernel)
    Q_cpu = torch.randn(B, H, N, D, dtype=torch.float16, device="cpu")
    K_cpu = torch.randn(B, H // groups, N, D, dtype=torch.float16, device="cpu")
    V_cpu = torch.randn_like(K_cpu)
    sinks_cpu = torch.randn(H, dtype=torch.float16, device="cpu")
    dO_cpu = torch.randn_like(Q_cpu)

    Q = Q_cpu.npu()
    K = K_cpu.npu()
    V = V_cpu.npu()
    sinks = sinks_cpu.npu()
    dO = dO_cpu.npu()

    # Enable autograd on inputs (needed for backward to populate .grad)
    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)

    # Golden (CPU)
    O_golden = ref_fwd(Q_cpu, K_cpu, V_cpu, sinks_cpu, window_size, groups)
    dQ_golden, dK_golden, dV_golden, _ = ref_bwd(Q_cpu, K_cpu, V_cpu, sinks_cpu, dO_cpu, window_size, groups)

    # NPU forward + backward (end-to-end via autograd wrapper)
    O_npu = attention(Q, K, V, sinks, window_size, groups)
    O_npu.backward(dO)
    dQ = Q.grad
    dK = K.grad
    dV = V.grad
    torch.npu.synchronize()

    # Precision check (169-line standard: atol=6.10e-5, rtol=1.95e-3 for fp16)
    def _check(actual, golden, name):
        atol, rtol = 6.10e-5, 1.95e-3
        a = actual.detach().cpu().float()
        g = golden.detach().cpu().float()
        # INF/NAN structural comparison (precision-standard.md §3.1)
        special = ~torch.isfinite(g)
        if special.any():  # noqa: SIM102
            if not torch.equal(torch.isnan(a[special]), torch.isnan(g[special])) or not torch.equal(
                torch.isinf(a[special]), torch.isinf(g[special])
            ):
                print(f"  {name:10s}: [PRECISION_FAIL] inf/nan structure mismatch")
                return False
        m = torch.isfinite(g)
        if m.sum().item() == 0:
            print(f"  {name:10s}: [PRECISION_PASS] ratio=1.0000 max_abs=0.000e+00 (all inf/nan)")
            return True
        abs_err = (a[m] - g[m]).abs()
        threshold = atol + rtol * g[m].abs()
        matched_ratio = (abs_err <= threshold).float().mean().item()
        max_abs = abs_err.max().item()
        passed = matched_ratio >= 0.99 and max_abs <= 0.1
        tag = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"
        print(f"  {name:10s}: {tag} ratio={matched_ratio:.4f} max_abs={max_abs:.3e}")
        return passed

    print(f"Smoke test: B={B} H={H} N={N} D={D} g={groups} w={window_size}")
    ok = True
    ok &= _check(O_npu, O_golden, "fwd_O")
    ok &= _check(dQ, dQ_golden, "bwd_dQ")
    ok &= _check(dK, dK_golden, "bwd_dK")
    ok &= _check(dV, dV_golden, "bwd_dV")

    if ok:
        print("\nTest Passed!")
    else:
        print("\nTest FAILED — see [PRECISION_FAIL] above")
        sys.exit(1)
