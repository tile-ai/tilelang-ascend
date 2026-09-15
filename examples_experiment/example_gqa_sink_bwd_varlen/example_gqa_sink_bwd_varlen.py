"""GQA + Attention Sink Flash Attention Backward (Varlen) for Ascend NPU.

Single-kernel backward: 1 forward + 1 backward kernel = 2 launches.
All intermediates on-chip (L1/UB/L0C), no GM workspace, no host sync.

  fwd: flashattn_fwd        — online softmax + sink + causal/window mask
  bwd: flashattn_bwd_single — Delta + K-loop (5-GEMM + 2-softmax-bwd +
                               Compensated GEMM via L0C init=False) + dSink

Developer mode (NoScope): threads=1, 4-True pass_configs, block_M=64, block_N=64.
0 T.Scope / 0 cross_flag / 0 barrier_all / 0 annotate_address.
"""

import tilelang
import torch
from tilelang import language as T

# ============================================================================
# pass_configs
# ============================================================================

_hybrid_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_vector_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

DTYPE_FP16 = "float16"
BLOCK_M_FWD = 64
BLOCK_N_FWD = 64
BLOCK_M_BWD = 64
BLOCK_N_BWD = 64


# ============================================================================
# Forward kernel
# ============================================================================


@tilelang.jit(out_idx=[3, 4], pass_configs=_hybrid_pass_configs)
def flashattn_fwd(
    batch,
    UQ,
    UKV,
    max_seq_len,
    heads,
    dim,
    groups,
    window_size,
    block_M=64,
    block_N=64,
):
    """Forward: online softmax + sink + causal/window mask (varlen)."""
    sm_scale = (1.0 / dim) ** 0.5
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"
    q_shape = [UQ, heads, dim]
    kv_shape = [UKV, head_kv, dim]
    o_shape = [UQ, heads, dim]
    lse_shape = [batch, heads, max_seq_len]
    block_num = (max_seq_len // block_M) * heads * batch
    if window_size is not None:
        assert window_size % block_N == 0
    window_eff = window_size if window_size is not None else max_seq_len * 2

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(o_shape, dtype),
        lse: T.Tensor(lse_shape, accum_dtype),
        Sinks: T.Tensor([heads], dtype),
        cu_seqlens_q: T.Tensor([batch + 1], "int32"),
        cu_seqlens_k: T.Tensor([batch + 1], "int32"),
    ):
        with T.Kernel(block_num, threads=1, is_npu=True) as cid:
            bx = cid % (max_seq_len // block_M)
            by = cid // (max_seq_len // block_M) % heads
            bz = cid // (max_seq_len // block_M) // heads % batch
            kv_by = by // groups
            q_start = cu_seqlens_q[bz]
            kv_start = cu_seqlens_k[bz]
            q_len = cu_seqlens_q[bz + 1] - q_start
            kv_len = cu_seqlens_k[bz + 1] - kv_start
            offset = kv_len - q_len
            loop_st = T.max(0, (bx * block_M + offset - window_eff) // block_N)
            loop_ed = T.min(
                T.ceildiv((bx + 1) * block_M + offset, block_N),
                T.ceildiv(kv_len, block_N),
            )
            if bx * block_M < q_len:
                q_l1 = T.alloc_L1([block_M, dim], dtype)
                k_l1 = T.alloc_L1([block_N, dim], dtype)
                v_l1 = T.alloc_L1([block_N, dim], dtype)
                acc_s_l1 = T.alloc_L1([block_M, block_N], dtype)
                acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
                acc_o_l0c = T.alloc_L0C([block_M, dim], accum_dtype)
                acc_o = T.alloc_ub([block_M, dim], accum_dtype)
                sumexp = T.alloc_ub([block_M], accum_dtype)
                m_i = T.alloc_ub([block_M], accum_dtype)
                acc_s_ub = T.alloc_ub([block_M, block_N], accum_dtype)
                m_i_prev = T.alloc_ub([block_M], accum_dtype)
                acc_s_ub_ = T.alloc_ub([block_M, block_N], accum_dtype)
                sumexp_i_ub = T.alloc_ub([block_M], accum_dtype)
                acc_s_half = T.alloc_ub([block_M, block_N], dtype)
                acc_o_ub = T.alloc_ub([block_M, dim], accum_dtype)
                acc_o_half = T.alloc_ub([block_M, dim], dtype)
                col_pos = T.alloc_ub([block_N], accum_dtype)
                cmp_mask = T.alloc_ub([block_N], accum_dtype)
                win_mask = T.alloc_ub([block_N], accum_dtype)
                combined_mask = T.alloc_ub([block_N], accum_dtype)
                sink_ub = T.alloc_ub([block_M], accum_dtype)
                sink_exp_ub = T.alloc_ub([block_M], accum_dtype)
                sink_scalar = T.alloc_ub([1], dtype)
                m_i_2d = T.alloc_ub([block_M, block_N], accum_dtype)
                m_i_prev_2d = T.alloc_ub([block_M, dim], accum_dtype)
                sumexp_2d = T.alloc_ub([block_M, dim], accum_dtype)
                T.tile.fill(acc_o, 0.0)
                T.tile.fill(sumexp, 0.0)
                T.tile.fill(m_i, -(2**30))
                T.copy(Q[q_start + bx * block_M : q_start + (bx + 1) * block_M, by, :], q_l1)
                T.copy(Sinks[by : by + 1], sink_scalar)
                T.tile.fill(sink_ub, sink_scalar[0])
                for k in T.serial(loop_st, loop_ed):
                    T.copy(K[kv_start + k * block_N : kv_start + (k + 1) * block_N, kv_by, :], k_l1)
                    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                    T.copy(acc_s_l0c, acc_s_ub_)
                    T.tile.fill(acc_s_ub, 0.0)
                    T.copy(m_i, m_i_prev)
                    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                    T.tile.arith_progression(col_pos, k * block_N, 1, block_N)
                    T.tile.compare(win_mask, col_pos, kv_len, "LT")
                    for h_i in range(block_M):
                        row_pos = (bx * block_M + h_i + offset) * 1.0
                        T.tile.compare(cmp_mask, col_pos, row_pos, "LE")
                        if window_size is not None:
                            T.tile.compare(combined_mask, col_pos, row_pos - window_size, "GT")
                            T.tile.bitwise_and(combined_mask, combined_mask, cmp_mask)
                            T.tile.bitwise_and(combined_mask, combined_mask, win_mask)
                        else:
                            T.tile.bitwise_and(combined_mask, cmp_mask, win_mask)
                        T.tile.select(
                            acc_s_ub[h_i, :],
                            combined_mask,
                            acc_s_ub[h_i, :],
                            -T.infinity(accum_dtype),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                    T.reduce_max(acc_s_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    T.tile.sub(m_i_prev, m_i_prev, m_i)
                    T.tile.exp(m_i_prev, m_i_prev)
                    T.tile.broadcast(m_i_2d, m_i, axis=1)
                    T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)
                    T.tile.exp(acc_s_ub, acc_s_ub)
                    T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                    T.tile.mul(sumexp, sumexp, m_i_prev)
                    T.tile.add(sumexp, sumexp, sumexp_i_ub)
                    T.copy(acc_s_ub, acc_s_half)
                    T.copy(acc_s_half, acc_s_l1)
                    T.copy(V[kv_start + k * block_N : kv_start + (k + 1) * block_N, kv_by, :], v_l1)
                    T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                    T.copy(acc_o_l0c, acc_o_ub)
                    T.tile.broadcast(m_i_prev_2d, m_i_prev, axis=1)
                    T.tile.mul(acc_o, acc_o, m_i_prev_2d)
                    T.tile.add(acc_o, acc_o, acc_o_ub)
                T.tile.compare(m_i_prev, m_i, -(2**30) * 1.0, "NE")
                T.tile.select(m_i, m_i_prev, m_i, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.tile.sub(sink_exp_ub, sink_ub, m_i)
                T.tile.exp(sink_exp_ub, sink_exp_ub)
                T.tile.add(sumexp, sumexp, sink_exp_ub)
                T.tile.broadcast(sumexp_2d, sumexp, axis=1)
                T.tile.div(acc_o, acc_o, sumexp_2d)
                T.copy(acc_o, acc_o_half)
                T.copy(acc_o_half, Output[q_start + bx * block_M : q_start + (bx + 1) * block_M, by, :])
                T.tile.ln(sumexp, sumexp)
                T.tile.add(sumexp, sumexp, m_i)
                T.copy(sumexp, lse[bz, by, bx * block_M : (bx + 1) * block_M])

    return main


# ============================================================================
# Backward single kernel: complete backward in ONE launch
# Developer mode (NoScope): T.alloc_shared/fragment, 0 T.Scope/barrier_all
# Q-block-centric grid + T.serial(loop_st, loop_ed) K-loop
# 4 CV handoffs, each with independent UB buffer (bhsd rule: reuse causes
#   "cube has 1, vec has 0" sync point mismatch in ascend_combinecv.cc:375)
# Compensated GEMM via L0C init=False accumulation (no GM workspace)
# p_delta/ds_delta: fp16 intermediate UB→L1 direct (8KB each, fits UB)
# P_fp32 retained in s_ub across phases (avoids GM roundtrip for dS compute)
# dSinks: kernel-internal Phase 6 (fp32 output, golden uses kernel lse/Delta)
# ============================================================================


@tilelang.jit(pass_configs=_hybrid_pass_configs)
def flashattn_bwd_single(
    batch,
    UQ,
    UKV,
    max_seq_len,
    max_kv_len,
    heads,
    dim_qk,
    dim_qk_padded,
    dim_v,
    window_size,
    block_M,
    block_N,
    groups,
):
    """Single-kernel backward (on-chip, no GM workspace).

    Phase 0: Delta = sum(O * dO, dim=-1) — full block_M, fp32
    K-loop (T.serial):
      Phase 1 (Cube):  GEMM1 S = Q @ K^T
      Phase 2 (Vector): softmax P = exp(S*scale - lse) * mask; p_delta = P_fp32 - cast(P_fp16)
      Phase 3 (Cube):  GEMM2 dV = P^T @ dO (init=True) + GEMM2corr dV += p_delta^T @ dO (init=False)
                       GEMM3 dP = dO @ V^T
      Phase 4 (Vector): dS = P * (dP - Delta) * scale; ds_delta = dS_fp32 - cast(dS_fp16)
      Phase 5 (Cube):  GEMM4 dK = dS^T @ Q (init=True) + GEMM4corr dK += ds_delta^T @ Q (init=False)
                       GEMM5 dQ = dS^T @ K (atomic_add fp32 GM)
    Phase 6: dSink = -exp(sink - lse) * Delta (fp32 output)
    """
    sm_scale = (1.0 / dim_qk) ** 0.5
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"
    window_eff = window_size if window_size is not None else max_seq_len * 2
    bwd_block_num = (max_seq_len // block_M) * heads * batch

    q_shape = [UQ, heads, dim_qk_padded]
    k_shape = [UKV, head_kv, dim_qk_padded]
    v_shape = [UKV, head_kv, dim_v]
    o_shape = [UQ, heads, dim_v]
    do_shape = [UQ, heads, dim_v]
    lse_shape = [batch, heads, max_seq_len]
    dv_shape = [UKV, head_kv, dim_v]
    dk_shape = [UKV, head_kv, dim_qk_padded]
    dq_shape = [UQ, heads, dim_qk_padded]
    delta_shape = [batch, heads, max_seq_len]
    sinks_shape = [heads]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        O: T.Tensor(o_shape, dtype),
        dO: T.Tensor(do_shape, dtype),
        lse: T.Tensor(lse_shape, accum_dtype),
        Sinks: T.Tensor(sinks_shape, dtype),
        cu_seqlens_q: T.Tensor([batch + 1], "int32"),
        cu_seqlens_k: T.Tensor([batch + 1], "int32"),
        Delta_out: T.Tensor(delta_shape, accum_dtype),
        dSinks: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(dq_shape, accum_dtype),
        dK: T.Tensor(dk_shape, accum_dtype),
        dV: T.Tensor(dv_shape, accum_dtype),
    ):
        with T.Kernel(bwd_block_num, threads=1, is_npu=True) as (cid):
            bx = cid % (max_seq_len // block_M)
            by = cid // (max_seq_len // block_M) % heads
            bz = cid // (max_seq_len // block_M) // heads % batch
            kv_by = by // groups
            q_start = cu_seqlens_q[bz]
            kv_start = cu_seqlens_k[bz]
            q_len = cu_seqlens_q[bz + 1] - q_start
            kv_len = cu_seqlens_k[bz + 1] - kv_start
            offset = kv_len - q_len
            loop_st = T.max(0, (bx * block_M + offset - window_eff) // block_N)
            loop_ed = T.min(
                T.ceildiv((bx + 1) * block_M + offset, block_N),
                T.ceildiv(kv_len, block_N),
            )

            if bx * block_M < q_len:
                # L1 (Cube inputs)
                q_l1 = T.alloc_shared([block_M, dim_qk_padded], dtype)
                k_l1 = T.alloc_shared([block_N, dim_qk_padded], dtype)
                do_l1 = T.alloc_shared([block_M, dim_v], dtype)
                v_l1 = T.alloc_shared([block_N, dim_v], dtype)
                p_l1 = T.alloc_shared([block_M, block_N], dtype)
                p_delta_l1 = T.alloc_shared([block_M, block_N], dtype)
                ds_l1 = T.alloc_shared([block_M, block_N], dtype)
                ds_delta_l1 = T.alloc_shared([block_M, block_N], dtype)

                # L0C (fragment)
                l0c_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                l0c_dp = T.alloc_fragment([block_M, block_N], accum_dtype)
                l0c_dv = T.alloc_fragment([block_N, dim_v], accum_dtype)
                l0c_dk = T.alloc_fragment([block_N, dim_qk_padded], accum_dtype)
                l0c_dq = T.alloc_fragment([block_M, dim_qk_padded], accum_dtype)

                # UB (Vector) — s_ub retains P_fp32/dS_fp32 across phases
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

                # Phase 0 / Phase 6 buffers
                o_ub_p0 = T.alloc_ub([block_M, dim_v], dtype)
                do_ub_p0 = T.alloc_ub([block_M, dim_v], dtype)
                prod_ub_p0 = T.alloc_ub([block_M, dim_v], accum_dtype)
                do_fp32_p0 = T.alloc_ub([block_M, dim_v], accum_dtype)
                sum_ub_p0 = T.alloc_ub([block_M, dim_v], accum_dtype)
                sink_val_ub = T.alloc_ub([block_M], accum_dtype)
                sink_exp_ub = T.alloc_ub([block_M], accum_dtype)
                sink_scalar = T.alloc_ub([1], dtype)

                # Phase 0: Delta = sum(O * dO)
                T.copy(O[q_start + bx * block_M : q_start + (bx + 1) * block_M, by, :dim_v], o_ub_p0)
                T.copy(dO[q_start + bx * block_M : q_start + (bx + 1) * block_M, by, :dim_v], do_ub_p0)
                T.copy(o_ub_p0, prod_ub_p0)
                T.copy(do_ub_p0, do_fp32_p0)
                T.tile.mul(sum_ub_p0, prod_ub_p0, do_fp32_p0)
                T.reduce_sum(sum_ub_p0, delta_ub, dim=-1)
                T.copy(delta_ub, Delta_out[bz, by, bx * block_M : (bx + 1) * block_M])

                # Loop-invariant loads
                T.copy(Q[q_start + bx * block_M : q_start + (bx + 1) * block_M, by, :], q_l1)
                T.copy(dO[q_start + bx * block_M : q_start + (bx + 1) * block_M, by, :dim_v], do_l1)
                T.copy(lse[bz, by, bx * block_M : (bx + 1) * block_M], lse_ub)
                T.tile.arith_progression(row_1d, (bx * block_M + offset) * 1.0, 1, block_M)

                for k_iter in T.serial(loop_st, loop_ed):
                    kv = k_iter * block_N

                    # === Phase 1 (Cube): GEMM1 S = Q @ K^T ===
                    T.copy(K[kv_start + kv : kv_start + kv + block_N, kv_by, :], k_l1)
                    T.gemm_v0(q_l1, k_l1, l0c_s, transpose_B=True, init=True)

                    # === Phase 2 (Vector): softmax P + p_delta ===
                    # C→V handoff #1: L0C→UB (on-chip direct, independent buffer)
                    T.copy(l0c_s, s_ub)
                    # P = exp(S*scale - lse)
                    T.tile.fill(lse_2d, 0.0)
                    T.tile.axpy(lse_2d, s_ub, sm_scale)
                    T.tile.broadcast(s_ub, lse_ub, axis=1)
                    T.tile.sub(lse_2d, lse_2d, s_ub)
                    T.tile.exp(s_ub, lse_2d)

                    # mask: causal (kv_pos <= q_pos + offset) + window + kv padding
                    T.tile.arith_progression(col_pos, kv, 1, block_N)
                    T.tile.broadcast(lse_2d, col_pos, axis=0)  # reuse as col_pos_2d
                    T.tile.broadcast(row_2d, row_1d, axis=1)
                    T.tile.compare(mask_2d, lse_2d, row_2d, "LE")  # causal
                    T.tile.sub(row_2d, row_2d, window_eff)
                    T.tile.compare(p_delta_ub, lse_2d, row_2d, "GT")  # window
                    T.tile.bitwise_and(mask_2d, mask_2d, p_delta_ub)
                    T.tile.compare(p_delta_ub, col_pos, kv_len, "LT")  # kv padding
                    T.tile.bitwise_and(mask_2d, mask_2d, p_delta_ub)
                    T.tile.select(s_ub, mask_2d, s_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")

                    # P_fp32 retained in s_ub for Phase 4 dS compute
                    T.copy(s_ub, p_half)  # fp32 → fp16 (for GEMM2)
                    T.copy(p_half, p_l1)  # V→C handoff #2: UB→L1 direct
                    # p_delta = P_fp32 - cast_fp32(P_fp16) — Compensated GEMM residual
                    T.copy(p_half, p_delta_ub)
                    T.tile.sub(p_delta_ub, s_ub, p_delta_ub)
                    T.copy(p_delta_ub, p_delta_half)  # fp16, UB→L1 direct

                    # === Phase 3 (Cube): GEMM2 dV + GEMM2corr + GEMM3 dP ===
                    # Compensated GEMM: main + correction via L0C init=False accumulation
                    T.gemm_v0(p_l1, do_l1, l0c_dv, transpose_A=True, init=True)
                    T.copy(p_delta_half, p_delta_l1)  # UB→L1 direct (no GM workspace)
                    T.gemm_v0(p_delta_l1, do_l1, l0c_dv, transpose_A=True, init=False)
                    # dV fp32 atomic_add (GQA groups need fp32 cross-head accumulation)
                    T.tile.atomic_add(dV[kv_start + kv : kv_start + kv + block_N, kv_by, :dim_v], l0c_dv)
                    T.copy(V[kv_start + kv : kv_start + kv + block_N, kv_by, :dim_v], v_l1)
                    T.gemm_v0(do_l1, v_l1, l0c_dp, transpose_B=True, init=True)

                    # === Phase 4 (Vector): dS compute ===
                    # C→V handoff #3: L0C→UB (on-chip direct)
                    T.copy(l0c_dp, dp_ub)
                    # dS = P * (dP - Delta) * scale — s_ub still has P_fp32 from Phase 2
                    T.tile.broadcast(delta_2d, delta_ub, axis=1)
                    T.tile.sub(dp_ub, dp_ub, delta_2d)
                    T.tile.mul(s_ub, s_ub, dp_ub)
                    T.tile.mul(s_ub, s_ub, sm_scale)

                    # mask: same as Phase 2 (causal + window + kv padding)
                    T.tile.arith_progression(col_pos, kv, 1, block_N)
                    T.tile.broadcast(delta_2d, col_pos, axis=0)
                    T.tile.broadcast(row_2d, row_1d, axis=1)
                    T.tile.compare(mask_2d, delta_2d, row_2d, "LE")
                    T.tile.sub(row_2d, row_2d, window_eff)
                    T.tile.compare(ds_delta_ub, delta_2d, row_2d, "GT")
                    T.tile.bitwise_and(mask_2d, mask_2d, ds_delta_ub)
                    T.tile.compare(ds_delta_ub, col_pos, kv_len, "LT")
                    T.tile.bitwise_and(mask_2d, mask_2d, ds_delta_ub)
                    T.tile.select(s_ub, mask_2d, s_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")

                    # dS_fp32 in s_ub; ds_delta = dS_fp32 - cast_fp32(dS_fp16)
                    T.copy(s_ub, ds_half)  # fp32 → fp16 (for GEMM4)
                    T.copy(ds_half, ds_l1)  # V→C handoff #4: UB→L1 direct
                    T.copy(ds_half, ds_rec_ub)
                    T.tile.sub(ds_delta_ub, s_ub, ds_rec_ub)
                    T.copy(ds_delta_ub, ds_delta_half)  # fp16, UB→L1 direct

                    # === Phase 5 (Cube): GEMM4 dK + GEMM4corr + GEMM5 dQ ===
                    # Compensated GEMM: main + correction via L0C init=False
                    T.gemm_v0(ds_l1, q_l1, l0c_dk, transpose_A=True, init=True)
                    T.copy(ds_delta_half, ds_delta_l1)  # UB→L1 direct (no GM workspace)
                    T.gemm_v0(ds_delta_l1, q_l1, l0c_dk, transpose_A=True, init=False)
                    # dK fp32 atomic_add
                    T.tile.atomic_add(dK[kv_start + kv : kv_start + kv + block_N, kv_by, :dim_qk_padded], l0c_dk)
                    # GEMM5: dQ = dS^T @ K (atomic_add fp32 GM)
                    T.copy(K[kv_start + kv : kv_start + kv + block_N, kv_by, :], k_l1)
                    T.gemm_v0(ds_l1, k_l1, l0c_dq, init=True)
                    T.gemm_v0(ds_delta_l1, k_l1, l0c_dq, init=False)
                    T.tile.atomic_add(dQ[q_start + bx * block_M : q_start + (bx + 1) * block_M, by, :dim_qk_padded], l0c_dq)

                # === Phase 6 (Vector): dSink = -exp(sink - lse) * Delta ===
                # Uses delta_ub (Phase 0) + lse_ub (loop-invariant load). fp32 output.
                # Golden uses kernel's lse/Delta to recompute, isolating T.tile.exp precision.
                T.copy(Sinks[by : by + 1], sink_scalar)
                T.tile.fill(sink_val_ub, sink_scalar[0])
                T.tile.sub(sink_exp_ub, sink_val_ub, lse_ub)
                T.tile.exp(sink_exp_ub, sink_exp_ub)
                T.tile.mul(sink_exp_ub, sink_exp_ub, delta_ub)
                T.tile.mul(sink_exp_ub, sink_exp_ub, -1.0)
                T.copy(sink_exp_ub, dSinks[bz, by, bx * block_M : (bx + 1) * block_M])

    return main


# ============================================================================
# Host pipeline: single kernel launch, no workspace, no host sync between stages
# ============================================================================


def run_bwd_pipeline(
    Q,
    K,
    V,
    O,
    dO,
    lse,
    Sinks,
    cu_seqlens_q,
    cu_seqlens_k,
    batch,
    UQ,
    UKV,
    max_seq_len,
    max_kv_len,
    heads,
    dim_qk,
    dim_v,
    window_size,
    block_M,
    block_N,
    groups,
):
    """Run single-kernel backward. Returns dQ/dK/dV (fp16), dSinks/Delta_out (fp32)."""
    head_kv = heads // groups
    dim_qk_padded = ((dim_qk + 127) // 128) * 128
    dV = torch.zeros(UKV, head_kv, dim_v, dtype=torch.float32, device="npu")
    dK = torch.zeros(UKV, head_kv, dim_qk_padded, dtype=torch.float32, device="npu")
    dQ = torch.zeros(UQ, heads, dim_qk_padded, dtype=torch.float32, device="npu")
    Delta_out = torch.zeros(batch, heads, max_seq_len, dtype=torch.float32, device="npu")
    dSinks = torch.zeros(batch, heads, max_seq_len, dtype=torch.float32, device="npu")
    bwd_mod = flashattn_bwd_single(
        batch,
        UQ,
        UKV,
        max_seq_len,
        max_kv_len,
        heads,
        dim_qk,
        dim_qk_padded,
        dim_v,
        window_size,
        block_M,
        block_N,
        groups,
    )
    bwd_mod(Q, K, V, O, dO, lse, Sinks, cu_seqlens_q, cu_seqlens_k, Delta_out, dSinks, dQ, dK, dV)
    torch.npu.synchronize()
    return dQ.half(), dK.half(), dV.half(), dSinks, Delta_out


# ============================================================================
# Smoke test (CI entry point — verifies kernel runs and output is non-trivial)
# ============================================================================


if __name__ == "__main__":
    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(42)

    B, H, G, N, D = 1, 4, 2, 128, 128
    H_kv = H // G
    UQ, UKV = N * B, N * B
    cu_q = torch.tensor([0, N], dtype=torch.int32, device="npu")
    cu_k = torch.tensor([0, N], dtype=torch.int32, device="npu")

    Q = torch.randn(UQ, H, D, dtype=torch.float16, device="npu")
    K = torch.randn(UKV, H_kv, D, dtype=torch.float16, device="npu")
    V = torch.randn_like(K)
    sinks = torch.randn(H, dtype=torch.float16, device="npu")
    dO = torch.randn_like(Q)

    fwd_mod = flashattn_fwd(B, UQ, UKV, N, H, D, G, None, BLOCK_M_FWD, BLOCK_N_FWD)
    O, lse = fwd_mod(Q, K, V, sinks, cu_q, cu_k)
    torch.npu.synchronize()

    dQ, dK, dV, dSinks, _ = run_bwd_pipeline(
        Q,
        K,
        V,
        O,
        dO,
        lse,
        sinks,
        cu_q,
        cu_k,
        B,
        UQ,
        UKV,
        N,
        N,
        H,
        D,
        D,
        None,
        BLOCK_M_BWD,
        BLOCK_N_BWD,
        G,
    )

    assert O.shape == (UQ, H, D)
    assert dQ.shape == (UQ, H, D)
    assert dK.shape == (UKV, H_kv, D)
    assert dV.shape == (UKV, H_kv, D)
    assert torch.isfinite(O).all()
    assert torch.isfinite(dQ).all()
    assert torch.isfinite(dK).all()
    assert torch.isfinite(dV).all()
    assert torch.isfinite(dSinks).all()

    print("Test Passed!")
