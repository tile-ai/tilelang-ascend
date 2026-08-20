"""GQA Sink Attention (BHSD) for Ascend NPU — no-scope Developer mode (5-kernel bwd split).

Layout: BHSD (Batch, Heads, SeqLen, Dim). Supports GQA (grouped-query attention),
an attention sink token, and an optional sliding window mask. fp16, Developer mode.

Architecture (ref: examples/deepseek_nsa/example_tilelang_nsa_bwd/):
  k1 (Cube):   S = Q @ K^T                     -> ws_s          [fp32]
  k2 (Vector): P = exp(S*scale - lse) + mask    -> ws_p, ws_p_delta, ws_p_fp32
  k3 (Cube):   dV = P^T @ dO (Comp GEMM) + dP   -> dV[atomic], ws_dp
  k4 (Vector): dS = P*(dP-Delta)*scale + mask   -> ws_ds, ws_ds_delta
  k5 (Cube):   dK = dS^T @ Q (Comp GEMM) + dQ   -> dK[atomic], dQ[L0C accumulate]

9 kernels total:
  1. flashattn_fwd:                Forward (online softmax + sink + window) -> O, lse
  2. flashattn_bwd_preprocess:     Delta = sum(O * dO, dim=-1)
  3. flashattn_bwd_k1_qk_recompute: S = Q @ K^T (recompute) -> ws_s
  4. flashattn_bwd_k2_softmax_p:    P = softmax(S) + mask + p_delta
  5. flashattn_bwd_k3_dv_dp:        dV (Compensated GEMM) + dP -> ws_dp
  6. flashattn_bwd_k4_ds_compute:   dS = P*(dP-Delta)*scale + mask + ds_delta
  7. flashattn_bwd_k5_dk_dq:        dK (Compensated GEMM) + dQ (L0C accumulate)
  8. flashattn_bwd_postprocess:     dK/dV fp32 -> fp16 (dQ direct fp16 from k5)
  9. flashattn_bwd_dsink:           dSink = -exp(sink - lse) * Delta

Key design decisions:
  - No T.Scope, no set_flag/wait_flag, no cross_flag, no barrier_all — pure
    AUTO_CV_SYNC + AUTO_SYNC automatic synchronization (Developer mode).
  - Compensated GEMM: fp16 GEMM result corrected by a second GEMM on the fp16
    quantization residual (p_delta in k3, ds_delta in k5). Recovers fp32-equivalent
    accuracy while keeping the main GEMM in fp16 throughput path.
  - Split-loop mask skip in k2/k4: when a KV block is fully above the causal
    diagonal (all positions pass the mask), the mask computation is skipped via a
    Python-level split into two T.serial loops. TIR if/else inside T.serial is
    avoided because Ascend codegen handles branches inside loops poorly.
  - dQ written as fp16 directly from k5 (L0C fp32 -> GM fp16 auto-cast), skipping
    the postprocess cast that dK/dV still need (they stay fp32 in GM for atomic_add).
  - Precision: 169-line standard (atol=6.10e-5, rtol=1.95e-3 for fp16).
"""

import sys

import tilelang
import torch
from tilelang import language as T

# ============================================================================
# pass_configs (D11: 4 explicit keys per config)
# ============================================================================

# Hybrid mode for fwd + Cube kernels (k1, k3, k5) — AUTO_CV_COMBINE/SYNC needed
# for L0C->GM->UB two-hop accumulation pattern.
_hybrid_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Vector mode for preprocess/k2/k4/postprocess/dsink (pure element-wise, no GEMM).
_vector_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
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
            # 2D mask buffers: row_pos [hm] broadcast to row_pos_2d [hm, block_N],
            # col_pos broadcast to [hm, block_N] via acc_s_ub_ reuse. mask_2d holds
            # the final per-element mask; causal_mask_2d is the window-case intermediate.
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

            # Loop-invariant: Q token positions depend only on q_row and v_row.
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
                # axpy(dst, src, scalar) = scalar*src + dst; with dst=0 yields sm_scale * S.
                T.tile.axpy(acc_s_ub, acc_s_ub_, sm_scale)

                # 2D mask via broadcast + compare + select (replaces per-row Python loop).
                # acc_s_ub_ reused as col_pos_2d (free after the axpy above).
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
                # Broadcast m_i to 2D and subtract (replaces per-row loop).
                T.tile.broadcast(acc_s_ub_, m_i, axis=1)
                T.tile.sub(acc_s_ub, acc_s_ub, acc_s_ub_)
                T.tile.exp(acc_s_ub, acc_s_ub)
                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)
                # Broadcast m_i_prev to 2D and rescale acc_o (replaces per-row loop).
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

            # Normalize: O /= sumexp (broadcast sumexp to 2D, replaces per-row loop).
            T.tile.broadcast(acc_o_ub, sumexp, axis=1)
            T.tile.div(acc_o, acc_o, acc_o_ub)

            T.copy(acc_o, acc_o_half)
            T.copy(acc_o_half, Output[bz, by, q_row + v_row : q_row + v_row + hm, :])

            T.tile.ln(sumexp, sumexp)
            T.tile.add(sumexp, sumexp, m_i)
            T.copy(sumexp, lse[bz, by, q_row + v_row : q_row + v_row + hm])

    return main


# ============================================================================
# Kernel 2: Backward Preprocess — Delta = sum(O * dO, dim=-1)
# ============================================================================


@tilelang.jit(out_idx=[2], pass_configs=_vector_pass_configs)
def flashattn_bwd_preprocess(batch, heads, seq_len, dim, blk=32):
    assert seq_len % blk == 0
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, heads, seq_len, dim]
    block_num = heads * (seq_len // blk) * batch

    @T.prim_func
    def main(
        O: T.Tensor(shape, dtype),  # type: ignore
        dO: T.Tensor(shape, dtype),  # type: ignore
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
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

            T.copy(O[bz, bx, by * blk + vid * blk // 2 : by * blk + vid * blk // 2 + blk // 2, :], o_ub)
            T.copy(dO[bz, bx, by * blk + vid * blk // 2 : by * blk + vid * blk // 2 + blk // 2, :], do_ub)
            T.copy(o_ub, prod_ub)
            T.copy(do_ub, do_fp32)
            T.tile.mul(sum_ub, prod_ub, do_fp32)
            T.reduce_sum(sum_ub, delta_ub, dim=-1)
            T.copy(delta_ub, Delta[bz, bx, by * blk + vid * blk // 2 : by * blk + vid * blk // 2 + blk // 2])

    return main


# ============================================================================
# Kernel 3 (k1): flashattn_bwd_k1_qk_recompute — S = Q @ K^T (pure Cube)
# ============================================================================


@tilelang.jit(pass_configs=_hybrid_pass_configs)
def flashattn_bwd_k1_qk_recompute(batch, heads, seq_len, dim_qk, dim_v, window_size, block_M, block_N, groups=1):
    """k1: S = Q @ K^T (recompute) -> ws_s [fp32].

    Pure Cube kernel. Each block handles one Q block, loops over KV blocks within
    the window. Output: ws_s[bwd_block_num, max_kv_per_q, block_M, block_N] fp32
    (unscaled — scale is applied in k2 UB).
    """
    assert seq_len % block_M == 0
    assert seq_len % block_N == 0
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"
    dim_qk_padded = ((dim_qk + 127) // 128) * 128

    q_shape = [batch, heads, seq_len, dim_qk_padded]
    k_shape = [batch, head_kv, seq_len, dim_qk_padded]
    bwd_block_num = (seq_len // block_M) * heads * batch

    if window_size is not None:
        assert window_size % block_N == 0
    window_eff = window_size if window_size is not None else seq_len * 2
    if window_size is not None:
        max_kv_per_q = min(window_size // block_N + 1, seq_len // block_N)
    else:
        max_kv_per_q = seq_len // block_N

    ws_shape = [bwd_block_num, max_kv_per_q, block_M, block_N]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        ws_s: T.Tensor(ws_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch
            kv_by = by // groups

            loop_st = T.max(0, (bx * block_M - window_eff) // block_N)
            loop_ed = T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N))

            q_l1 = T.alloc_L1([block_M, dim_qk_padded], dtype)
            k_l1 = T.alloc_L1([block_N, dim_qk_padded], dtype)
            l0c_s = T.alloc_L0C([block_M, block_N], accum_dtype)

            # Load loop-invariant Q block
            T.copy(Q[bz, by, bx * block_M : bx * block_M + block_M, :], q_l1)

            for k in T.serial(loop_st, loop_ed):
                kv = k * block_N

                T.copy(K[bz, kv_by, kv : kv + block_N, :], k_l1)
                # S = Q @ K^T -> l0c_s (unscaled — scale applied in k2 UB)
                T.gemm_v0(q_l1, k_l1, l0c_s, transpose_B=True, init=True)

                kv_iter = k - loop_st
                T.copy(l0c_s, ws_s[cid, kv_iter, :, :])

    return main


# ============================================================================
# Kernel 4 (k2): flashattn_bwd_k2_softmax_p — P = softmax(S) + mask + p_delta
# ============================================================================


@tilelang.jit(pass_configs=_vector_pass_configs)
def flashattn_bwd_k2_softmax_p(batch, heads, seq_len, dim_qk, dim_v, window_size, block_M, block_N, groups=1):
    """k2: P = exp(S*scale - lse) + causal/window mask -> ws_p + ws_p_delta + ws_p_fp32.

    Pure Vector kernel. Reads ws_s (fp32, raw S from k1) and lse (fp32, from fwd).
    Writes ws_p (fp16), ws_p_delta (fp16), ws_p_fp32 (fp32).

    Compensated GEMM: p_delta = P_fp32 - cast(P_fp16, fp32) captures the fp16
    storage loss. k3 uses main GEMM (P_fp16) + correction GEMM (p_delta).

    Split-loop mask skip: for non-window (causal only) case, KV blocks fully
    above the causal diagonal (all positions pass) skip the mask computation.
    Implemented as two T.serial loops at Python level — TIR if/else inside
    T.serial causes Ascend codegen issues. Window case uses a single loop
    (window mask is bidirectional, cannot prove all-pass).
    """
    assert seq_len % block_M == 0
    assert seq_len % block_N == 0
    sm_scale = (1.0 / dim_qk) ** 0.5
    dtype = "float16"
    accum_dtype = "float"

    lse_shape = [batch, heads, seq_len]
    bwd_block_num = (seq_len // block_M) * heads * batch

    if window_size is not None:
        assert window_size % block_N == 0
    window_eff = window_size if window_size is not None else seq_len * 2
    if window_size is not None:
        max_kv_per_q = min(window_size // block_N + 1, seq_len // block_N)
    else:
        max_kv_per_q = seq_len // block_N

    ws_shape = [bwd_block_num, max_kv_per_q, block_M, block_N]

    @T.prim_func
    def main(
        ws_s: T.Tensor(ws_shape, accum_dtype),  # type: ignore
        lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        ws_p: T.Tensor(ws_shape, dtype),  # type: ignore
        ws_p_delta: T.Tensor(ws_shape, dtype),  # type: ignore
        ws_p_fp32: T.Tensor(ws_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch

            loop_st = T.max(0, (bx * block_M - window_eff) // block_N)
            loop_ed = T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N))

            q_row = bx * block_M

            # UB buffers
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

            # Loop-invariant: Q token positions + lse
            T.tile.arith_progression(row_1d, q_row, 1, block_M)
            T.copy(lse[bz, by, q_row : q_row + block_M], lse_ub)

            if window_size is None:
                # Split-loop: first k where mask IS needed is mask_k_start.
                # Blocks before mask_k_start are fully above the causal diagonal
                # (all positions pass) — mask computation skipped.
                mask_k_start = (bx * block_M + 1) // block_N

                # --- Loop 1: no mask (all elements pass causal) ---
                for k in T.serial(loop_st, mask_k_start):
                    kv = k * block_N
                    kv_iter = k - loop_st

                    T.copy(ws_s[cid, kv_iter, :, :], s_ub)
                    # S_scaled = S * scale (axpy: fill 0 + axpy sm_scale)
                    T.tile.fill(lse_2d, 0.0)
                    T.tile.axpy(lse_2d, s_ub, sm_scale)
                    # P_fp32 = exp(S_scaled - lse)
                    T.tile.broadcast(s_ub, lse_ub, axis=1)
                    T.tile.sub(lse_2d, lse_2d, s_ub)
                    T.tile.exp(s_ub, lse_2d)

                    # Write P_fp32 (exact, for k4 dS compute)
                    T.copy(s_ub, ws_p_fp32[cid, kv_iter, :, :])
                    # Write P_fp16 (lossy fp32->fp16 cast, for k3 GEMM)
                    T.copy(s_ub, p_half)
                    T.copy(p_half, ws_p[cid, kv_iter, :, :])
                    # Compensated GEMM: p_delta = P_fp32 - cast(P_fp16, fp32)
                    T.copy(p_half, p_delta_ub)
                    T.tile.sub(p_delta_ub, s_ub, p_delta_ub)
                    T.copy(p_delta_ub, p_delta_half)
                    T.copy(p_delta_half, ws_p_delta[cid, kv_iter, :, :])

                # --- Loop 2: with causal mask ---
                for k in T.serial(mask_k_start, loop_ed):
                    kv = k * block_N
                    kv_iter = k - loop_st

                    T.copy(ws_s[cid, kv_iter, :, :], s_ub)
                    T.tile.fill(lse_2d, 0.0)
                    T.tile.axpy(lse_2d, s_ub, sm_scale)
                    T.tile.broadcast(s_ub, lse_ub, axis=1)
                    T.tile.sub(lse_2d, lse_2d, s_ub)
                    T.tile.exp(s_ub, lse_2d)

                    # Causal mask: col <= row
                    T.tile.arith_progression(col_pos, kv, 1, block_N)
                    T.tile.broadcast(lse_2d, col_pos, axis=0)
                    T.tile.broadcast(row_2d, row_1d, axis=1)
                    T.tile.compare(mask_2d, lse_2d, row_2d, "LE")
                    T.tile.select(s_ub, mask_2d, s_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")

                    T.copy(s_ub, ws_p_fp32[cid, kv_iter, :, :])
                    T.copy(s_ub, p_half)
                    T.copy(p_half, ws_p[cid, kv_iter, :, :])
                    T.copy(p_half, p_delta_ub)
                    T.tile.sub(p_delta_ub, s_ub, p_delta_ub)
                    T.copy(p_delta_ub, p_delta_half)
                    T.copy(p_delta_half, ws_p_delta[cid, kv_iter, :, :])

            else:
                # --- Window case: single loop with causal + window mask ---
                for k in T.serial(loop_st, loop_ed):
                    kv = k * block_N
                    kv_iter = k - loop_st

                    T.copy(ws_s[cid, kv_iter, :, :], s_ub)
                    T.tile.fill(lse_2d, 0.0)
                    T.tile.axpy(lse_2d, s_ub, sm_scale)
                    T.tile.broadcast(s_ub, lse_ub, axis=1)
                    T.tile.sub(lse_2d, lse_2d, s_ub)
                    T.tile.exp(s_ub, lse_2d)

                    # Causal + window mask
                    T.tile.arith_progression(col_pos, kv, 1, block_N)
                    T.tile.broadcast(lse_2d, col_pos, axis=0)
                    T.tile.broadcast(row_2d, row_1d, axis=1)
                    T.tile.compare(mask_2d, lse_2d, row_2d, "LE")
                    T.tile.sub(row_2d, row_2d, window_size)
                    T.tile.compare(p_delta_ub, lse_2d, row_2d, "GT")
                    T.tile.bitwise_and(mask_2d, mask_2d, p_delta_ub)
                    T.tile.select(s_ub, mask_2d, s_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")

                    T.copy(s_ub, ws_p_fp32[cid, kv_iter, :, :])
                    T.copy(s_ub, p_half)
                    T.copy(p_half, ws_p[cid, kv_iter, :, :])
                    T.copy(p_half, p_delta_ub)
                    T.tile.sub(p_delta_ub, s_ub, p_delta_ub)
                    T.copy(p_delta_ub, p_delta_half)
                    T.copy(p_delta_half, ws_p_delta[cid, kv_iter, :, :])

    return main


# ============================================================================
# Kernel 5 (k3): flashattn_bwd_k3_dv_dp — dV (Compensated GEMM) + dP
# ============================================================================


@tilelang.jit(pass_configs=_hybrid_pass_configs)
def flashattn_bwd_k3_dv_dp(batch, heads, seq_len, dim_qk, dim_v, window_size, block_M, block_N, groups=1):
    """k3: dV = P^T @ dO (Compensated GEMM) + dP = dO @ V^T -> ws_dp.

    Hybrid kernel. Per (Q block, KV block):
      GEMM2 main:  dV = P_fp16^T @ dO  (init=True, fresh)
      GEMM2 corr:  dV += p_delta^T @ dO (init=False, Compensated GEMM)
      atomic_add:  dV[kv] += l0c_dv (L0C -> GM atomic, fp32)
      GEMM3:       dP = dO @ V^T -> ws_dp (init=True, fresh)
    """
    assert seq_len % block_M == 0
    assert seq_len % block_N == 0
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"

    do_shape = [batch, heads, seq_len, dim_v]
    v_shape = [batch, head_kv, seq_len, dim_v]
    dv_shape = [batch, head_kv, seq_len, dim_v]
    bwd_block_num = (seq_len // block_M) * heads * batch

    if window_size is not None:
        assert window_size % block_N == 0
    window_eff = window_size if window_size is not None else seq_len * 2
    if window_size is not None:
        max_kv_per_q = min(window_size // block_N + 1, seq_len // block_N)
    else:
        max_kv_per_q = seq_len // block_N

    ws_shape = [bwd_block_num, max_kv_per_q, block_M, block_N]

    @T.prim_func
    def main(
        ws_p: T.Tensor(ws_shape, dtype),  # type: ignore
        ws_p_delta: T.Tensor(ws_shape, dtype),  # type: ignore
        dO: T.Tensor(do_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        dV: T.Tensor(dv_shape, accum_dtype),  # type: ignore
        ws_dp: T.Tensor(ws_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch
            kv_by = by // groups

            loop_st = T.max(0, (bx * block_M - window_eff) // block_N)
            loop_ed = T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N))

            q_row = bx * block_M

            # L1 buffers (Cube scope) — fp16
            p_l1 = T.alloc_L1([block_M, block_N], dtype)
            p_delta_l1 = T.alloc_L1([block_M, block_N], dtype)
            do_l1 = T.alloc_L1([block_M, dim_v], dtype)
            v_l1 = T.alloc_L1([block_N, dim_v], dtype)

            # L0C buffers (Cube scope) — fp32
            l0c_dv = T.alloc_L0C([block_N, dim_v], accum_dtype)
            l0c_dp = T.alloc_L0C([block_M, block_N], accum_dtype)

            # Load loop-invariant dO block
            T.copy(dO[bz, by, q_row : q_row + block_M, :], do_l1)

            for k in T.serial(loop_st, loop_ed):
                kv = k * block_N
                kv_iter = k - loop_st

                # Load P_fp16 from GM workspace
                T.copy(ws_p[cid, kv_iter, :, :], p_l1)

                # GEMM2 main: dV = P_fp16^T @ dO -> l0c_dv (init=True, fresh)
                T.gemm_v0(p_l1, do_l1, l0c_dv, transpose_A=True, init=True)

                # GEMM2 correction: dV += p_delta^T @ dO (Compensated GEMM)
                T.copy(ws_p_delta[cid, kv_iter, :, :], p_delta_l1)
                T.gemm_v0(p_delta_l1, do_l1, l0c_dv, transpose_A=True, init=False)

                # atomic_add dV contribution to GM (L0C -> GM, fp32)
                T.tile.atomic_add(dV[bz, kv_by, kv : kv + block_N, :], l0c_dv)

                # GEMM3: dP = dO @ V^T -> l0c_dp (init=True, fresh)
                T.copy(V[bz, kv_by, kv : kv + block_N, :], v_l1)
                T.gemm_v0(do_l1, v_l1, l0c_dp, transpose_B=True, init=True)

                # Write dP to GM workspace (fp32)
                T.copy(l0c_dp, ws_dp[cid, kv_iter, :, :])

    return main


# ============================================================================
# Kernel 6 (k4): flashattn_bwd_k4_ds_compute — dS = P*(dP-Delta)*scale + mask
# ============================================================================


@tilelang.jit(pass_configs=_vector_pass_configs)
def flashattn_bwd_k4_ds_compute(batch, heads, seq_len, dim_qk, dim_v, window_size, block_M, block_N, groups=1):
    """k4: dS = P_fp32 * (dP - Delta) * scale + mask -> ws_ds + ws_ds_delta.

    Pure Vector kernel. Reads ws_p_fp32 (P, fp32), ws_dp (dP, fp32), Delta (fp32,
    from preprocess). Writes ws_ds (fp16), ws_ds_delta (fp16).

    Compensated GEMM: ds_delta = dS_fp32 - cast(dS_fp16, fp32) captures the fp16
    storage loss. k5 uses main GEMM (dS_fp16) + correction GEMM (ds_delta).
    """
    assert seq_len % block_M == 0
    assert seq_len % block_N == 0
    sm_scale = (1.0 / dim_qk) ** 0.5
    dtype = "float16"
    accum_dtype = "float"

    delta_shape = [batch, heads, seq_len]
    bwd_block_num = (seq_len // block_M) * heads * batch

    if window_size is not None:
        assert window_size % block_N == 0
    window_eff = window_size if window_size is not None else seq_len * 2
    if window_size is not None:
        max_kv_per_q = min(window_size // block_N + 1, seq_len // block_N)
    else:
        max_kv_per_q = seq_len // block_N

    ws_shape = [bwd_block_num, max_kv_per_q, block_M, block_N]

    @T.prim_func
    def main(
        ws_p_fp32: T.Tensor(ws_shape, accum_dtype),  # type: ignore
        ws_dp: T.Tensor(ws_shape, accum_dtype),  # type: ignore
        Delta: T.Tensor(delta_shape, accum_dtype),  # type: ignore
        ws_ds: T.Tensor(ws_shape, dtype),  # type: ignore
        ws_ds_delta: T.Tensor(ws_shape, dtype),  # type: ignore
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch

            loop_st = T.max(0, (bx * block_M - window_eff) // block_N)
            loop_ed = T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N))

            q_row = bx * block_M

            # UB buffers
            p_ub = T.alloc_ub([block_M, block_N], accum_dtype)
            dp_ub = T.alloc_ub([block_M, block_N], accum_dtype)
            delta_ub = T.alloc_ub([block_M], accum_dtype)
            delta_2d = T.alloc_ub([block_M, block_N], accum_dtype)
            col_pos = T.alloc_ub([block_N], accum_dtype)
            row_1d = T.alloc_ub([block_M], accum_dtype)
            row_2d = T.alloc_ub([block_M, block_N], accum_dtype)
            mask_2d = T.alloc_ub([block_M, block_N], accum_dtype)
            ds_half = T.alloc_ub([block_M, block_N], dtype)
            ds_delta_half = T.alloc_ub([block_M, block_N], dtype)
            ds_delta_ub = T.alloc_ub([block_M, block_N], accum_dtype)
            ds_rec_ub = T.alloc_ub([block_M, block_N], accum_dtype)

            # Loop-invariant: Q token positions + Delta
            T.tile.arith_progression(row_1d, q_row, 1, block_M)
            T.copy(Delta[bz, by, q_row : q_row + block_M], delta_ub)

            for k in T.serial(loop_st, loop_ed):
                kv = k * block_N
                kv_iter = k - loop_st

                # Read P_fp32 (exact, no fp16 quantization) and dP from GM workspace
                T.copy(ws_p_fp32[cid, kv_iter, :, :], p_ub)
                T.copy(ws_dp[cid, kv_iter, :, :], dp_ub)

                # dS = P * (dP - Delta) * scale
                T.tile.broadcast(delta_2d, delta_ub, axis=1)
                T.tile.sub(dp_ub, dp_ub, delta_2d)
                T.tile.mul(p_ub, p_ub, dp_ub)
                T.tile.mul(p_ub, p_ub, sm_scale)  # p_ub = dS_fp32

                # Apply mask (causal, optionally + window)
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
                T.tile.select(p_ub, mask_2d, p_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                # p_ub = dS_fp32 (masked)

                # Write dS_fp16 (lossy fp32->fp16 cast)
                T.copy(p_ub, ds_half)
                T.copy(ds_half, ws_ds[cid, kv_iter, :, :])

                # Compensated GEMM: ds_delta = dS_fp32 - cast(dS_fp16, fp32)
                T.copy(ds_half, ds_rec_ub)
                T.tile.sub(ds_delta_ub, p_ub, ds_rec_ub)
                T.copy(ds_delta_ub, ds_delta_half)
                T.copy(ds_delta_half, ws_ds_delta[cid, kv_iter, :, :])

    return main


# ============================================================================
# Kernel 7 (k5): flashattn_bwd_k5_dk_dq — dK (Compensated GEMM) + dQ (accumulate)
# ============================================================================


@tilelang.jit(pass_configs=_hybrid_pass_configs)
def flashattn_bwd_k5_dk_dq(batch, heads, seq_len, dim_qk, dim_v, window_size, block_M, block_N, groups=1):
    """k5: dK = dS^T @ Q (Compensated GEMM, atomic_add) + dQ = dS @ K (Comp GEMM, L0C accumulate).

    Hybrid kernel. Per (Q block, KV block):
      GEMM4 main:  dK = dS_fp16^T @ Q  (init=True, fresh per KV block)
      GEMM4 corr:  dK += ds_delta^T @ Q (init=False, Compensated GEMM)
      atomic_add:  dK[kv] += l0c_dk (L0C -> GM atomic, fp32)
      GEMM5 main:  dQ = dS_fp16 @ K  (init=(k==loop_st), accumulate across KV blocks)
      GEMM5 corr:  dQ += ds_delta @ K (init=False, Compensated GEMM accumulate)
    After loop: write dQ from L0C -> GM (fp32 -> fp16 auto-cast).
    """
    assert seq_len % block_M == 0
    assert seq_len % block_N == 0
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"
    dim_qk_padded = ((dim_qk + 127) // 128) * 128

    q_shape = [batch, heads, seq_len, dim_qk_padded]
    k_shape = [batch, head_kv, seq_len, dim_qk_padded]
    dk_shape = [batch, head_kv, seq_len, dim_qk_padded]
    dq_shape = [batch, heads, seq_len, dim_qk_padded]
    bwd_block_num = (seq_len // block_M) * heads * batch

    if window_size is not None:
        assert window_size % block_N == 0
    window_eff = window_size if window_size is not None else seq_len * 2
    if window_size is not None:
        max_kv_per_q = min(window_size // block_N + 1, seq_len // block_N)
    else:
        max_kv_per_q = seq_len // block_N

    ws_shape = [bwd_block_num, max_kv_per_q, block_M, block_N]

    @T.prim_func
    def main(
        ws_ds: T.Tensor(ws_shape, dtype),  # type: ignore
        ws_ds_delta: T.Tensor(ws_shape, dtype),  # type: ignore
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        dK: T.Tensor(dk_shape, accum_dtype),  # type: ignore
        dQ: T.Tensor(dq_shape, dtype),  # type: ignore
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch
            kv_by = by // groups

            loop_st = T.max(0, (bx * block_M - window_eff) // block_N)
            loop_ed = T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N))

            q_row = bx * block_M

            # L1 buffers (Cube scope) — fp16
            ds_l1 = T.alloc_L1([block_M, block_N], dtype)
            ds_delta_l1 = T.alloc_L1([block_M, block_N], dtype)
            q_l1 = T.alloc_L1([block_M, dim_qk_padded], dtype)
            k_l1 = T.alloc_L1([block_N, dim_qk_padded], dtype)

            # L0C buffers (Cube scope) — fp32
            l0c_dk = T.alloc_L0C([block_N, dim_qk_padded], accum_dtype)
            l0c_dq = T.alloc_L0C([block_M, dim_qk_padded], accum_dtype)

            # Load loop-invariant Q block
            T.copy(Q[bz, by, q_row : q_row + block_M, :], q_l1)

            for k in T.serial(loop_st, loop_ed):
                kv = k * block_N
                kv_iter = k - loop_st

                # Load dS_fp16 from GM workspace
                T.copy(ws_ds[cid, kv_iter, :, :], ds_l1)

                # GEMM4 main: dK = dS_fp16^T @ Q -> l0c_dk (init=True, fresh per KV block)
                T.gemm_v0(ds_l1, q_l1, l0c_dk, transpose_A=True, init=True)

                # GEMM4 correction: dK += ds_delta^T @ Q (Compensated GEMM)
                T.copy(ws_ds_delta[cid, kv_iter, :, :], ds_delta_l1)
                T.gemm_v0(ds_delta_l1, q_l1, l0c_dk, transpose_A=True, init=False)

                # atomic_add dK contribution to GM (L0C -> GM, fp32)
                T.tile.atomic_add(dK[bz, kv_by, kv : kv + block_N, :], l0c_dk)

                # Load K block for GEMM5
                T.copy(K[bz, kv_by, kv : kv + block_N, :], k_l1)

                # GEMM5 main: dQ = dS_fp16 @ K -> l0c_dq (accumulate across KV blocks)
                T.gemm_v0(ds_l1, k_l1, l0c_dq, init=(k == loop_st))

                # GEMM5 correction: dQ += ds_delta @ K (Compensated GEMM, accumulate)
                T.gemm_v0(ds_delta_l1, k_l1, l0c_dq, init=False)

            # After loop: write accumulated dQ from L0C -> GM (fp32 -> fp16 auto-cast)
            T.copy(l0c_dq, dQ[bz, by, q_row : q_row + block_M, :])

    return main


# ============================================================================
# Kernel 8: Backward Postprocess — fp32 -> fp16 cast (for dK/dV)
# ============================================================================


@tilelang.jit(out_idx=[1], pass_configs=_vector_pass_configs)
def flashattn_bwd_postprocess(batch, heads, seq_len, dim_qk, blk=64):
    """Cast dK or dV from fp32 GM to fp16 GM. dQ does not need this — k5 writes
    dQ as fp16 directly (L0C fp32 -> GM fp16 auto-cast). dK/dV stay fp32 in GM
    because they receive atomic_add from multiple Q blocks.
    """
    assert seq_len % blk == 0
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, heads, seq_len, dim_qk]
    block_num = (seq_len // blk) * heads * batch

    @T.prim_func
    def main(
        dQ: T.Tensor(shape, accum_dtype),  # type: ignore
        dQ_out: T.Tensor(shape, dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // blk)
            by = cid // (seq_len // blk) % heads
            bz = cid // (seq_len // blk) // heads % batch

            dq_ub = T.alloc_ub([blk // 2, dim_qk], accum_dtype)
            dq_half = T.alloc_ub([blk // 2, dim_qk], dtype)

            T.copy(dQ[bz, by, bx * blk + vid * blk // 2 : bx * blk + vid * blk // 2 + blk // 2, :], dq_ub)
            T.copy(dq_ub, dq_half)
            T.copy(dq_half, dQ_out[bz, by, bx * blk + vid * blk // 2 : bx * blk + vid * blk // 2 + blk // 2, :])

    return main


# ============================================================================
# Kernel 9: Dsink — dSink = -exp(sink - lse) * Delta, fp32 output
# ============================================================================


@tilelang.jit(out_idx=-1, pass_configs=_vector_pass_configs)
def flashattn_bwd_dsink(batch, heads, seq_len, block=128):
    """dSink = -exp(sink - lse) * Delta. Output is fp32 (matches golden)."""
    assert seq_len % block == 0
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, heads, seq_len]
    block_num = heads * (seq_len // block) * batch

    @T.prim_func
    def main(
        Sinks: T.Tensor([heads], dtype),  # type: ignore
        Delta: T.Tensor(shape, accum_dtype),  # type: ignore
        lse: T.Tensor(shape, accum_dtype),  # type: ignore
        dsinks: T.Tensor(shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % heads
            by = cid // heads % (seq_len // block)
            bz = cid // heads // (seq_len // block) % batch

            lse_ub = T.alloc_ub([block // 2], accum_dtype)
            delta_ub = T.alloc_ub([block // 2], accum_dtype)
            sink_exp_ub = T.alloc_ub([block // 2], accum_dtype)
            sink_val_ub = T.alloc_ub([block // 2], accum_dtype)
            sink_scalar = T.alloc_ub([1], dtype)

            T.copy(Sinks[bx : bx + 1], sink_scalar)
            T.tile.fill(sink_val_ub, sink_scalar[0])

            T.copy(lse[bz, bx, by * block + vid * block // 2 : by * block + vid * block // 2 + block // 2], lse_ub)
            T.copy(Delta[bz, bx, by * block + vid * block // 2 : by * block + vid * block // 2 + block // 2], delta_ub)

            T.tile.sub(sink_exp_ub, sink_val_ub, lse_ub)
            T.tile.exp(sink_exp_ub, sink_exp_ub)
            T.tile.mul(sink_exp_ub, sink_exp_ub, delta_ub)
            T.tile.mul(sink_exp_ub, sink_exp_ub, -1.0)

            T.copy(sink_exp_ub, dsinks[bz, bx, by * block + vid * block // 2 : by * block + vid * block // 2 + block // 2])

    return main


# ============================================================================
# Golden Reference (PyTorch CPU)
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
# Autograd Function (end-to-end wrapper)
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

        prep_mod = flashattn_bwd_preprocess(B, H, N, D, blk=32)
        delta = prep_mod(o, do)
        torch.npu.synchronize()

        # Output tensors
        dQ = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float16, device=q.device)
        dK = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float32, device=q.device)
        dV = torch.zeros(B, H_kv, N, D, dtype=torch.float32, device=q.device)

        # GM workspace tensors (inter-kernel communication)
        bwd_block_num = H * (N // block_M) * B
        if window_size is not None:
            max_kv_per_q = min(window_size // block_N + 1, N // block_N)
        else:
            max_kv_per_q = N // block_N
        ws_s = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float32, device=q.device)
        ws_p = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device=q.device)
        ws_p_delta = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device=q.device)
        ws_p_fp32 = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float32, device=q.device)
        ws_dp = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float32, device=q.device)
        ws_ds = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device=q.device)
        ws_ds_delta = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device=q.device)

        bwd_args = (B, H, N, D, D, window_size, block_M, block_N, groups)

        # k1: S = Q @ K^T -> ws_s
        k1_mod = flashattn_bwd_k1_qk_recompute(*bwd_args)
        k1_mod(q, k, ws_s)
        torch.npu.synchronize()

        # k2: P = softmax(S) + mask + p_delta -> ws_p, ws_p_delta, ws_p_fp32
        k2_mod = flashattn_bwd_k2_softmax_p(*bwd_args)
        k2_mod(ws_s, lse, ws_p, ws_p_delta, ws_p_fp32)
        torch.npu.synchronize()

        # k3: dV (Compensated GEMM, atomic_add) + dP -> ws_dp
        k3_mod = flashattn_bwd_k3_dv_dp(*bwd_args)
        k3_mod(ws_p, ws_p_delta, do, v, dV, ws_dp)
        torch.npu.synchronize()

        # k4: dS = P*(dP-Delta)*scale + mask + ds_delta -> ws_ds, ws_ds_delta
        k4_mod = flashattn_bwd_k4_ds_compute(*bwd_args)
        k4_mod(ws_p_fp32, ws_dp, delta, ws_ds, ws_ds_delta)
        torch.npu.synchronize()

        # k5: dK (Compensated GEMM, atomic_add) + dQ (L0C accumulate)
        k5_mod = flashattn_bwd_k5_dk_dq(*bwd_args)
        k5_mod(ws_ds, ws_ds_delta, q, k, dK, dQ)
        torch.npu.synchronize()

        # Postprocess: fp32 -> fp16 for dK, dV (dQ already fp16 from k5)
        post_dk = flashattn_bwd_postprocess(B, H_kv, N, dim_qk_padded, blk=64)
        dK = post_dk(dK)[..., :D]
        post_dv = flashattn_bwd_postprocess(B, H_kv, N, D, blk=64)
        dV = post_dv(dV)
        dQ = dQ[..., :D]

        # Dsink (fp32 output, sum on CPU after D2H)
        dsink_mod = flashattn_bwd_dsink(B, H, N, block=128)
        dsinks = dsink_mod(sinks, delta, lse).sum(0).sum(1)

        return dQ, dK, dV, dsinks, None, None


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
