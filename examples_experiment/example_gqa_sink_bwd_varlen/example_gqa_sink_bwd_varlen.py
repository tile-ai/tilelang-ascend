"""GQA + Attention Sink Flash Attention Backward (Varlen) for Ascend NPU.

Single-file example: 11 backward kernels + 1 forward kernel + host pipeline +
PyTorch golden + layered test suite (L0/L1/L2/Boundary) + do_bench + msprof op.

Pipeline (NoScope Developer mode, 11-kernel split + GM workspace):
  fwd: flashattn_fwd           — forward (online softmax + sink + window)
  k0:  bwd_k0_preprocess       — Delta = sum(O*dO) (pre-loop, once)
  k1:  bwd_k1_qkt              — S = Q @ K^T -> ws_s
  k2:  bwd_k2_softmax          — P = softmax(S) -> ws_p
  k3c: bwd_k3c_p_delta         — p_delta = cast(P_fp32 - cast_fp32(P_fp16))
  k3:  bwd_k3_dv_dp            — dV += P_fp16^T@dO + dP = dO@V^T (main GEMM)
  k3b: bwd_k3b_dv_correction   — dV += p_delta^T@dO (Compensated GEMM corr)
  k4:  bwd_k4_ds               — dS = P*(dP-Delta)*scale
  k5c: bwd_k5c_ds_delta        — ds_delta = cast(dS_fp32 - cast_fp32(dS_fp16))
  k5a: bwd_k5_dk_dq            — dK += dS_fp16^T@Q + dQ += dS@K (main GEMM)
  k5b: bwd_k5b_dk_correction   — dK += ds_delta^T@Q (Compensated GEMM corr)
  k6:  bwd_k6_postprocess      — dQ_fp16=cast(dQ) + dSink

Compensated GEMM (attempt 3, Option B): k3c/k5c compute cast residuals
(p_delta/ds_delta), k3/k5a do main GEMMs with fp16-cast inputs, k3b/k5b add
correction GEMMs. Together: dV ≈ P_fp32^T@dO and dK ≈ dS_fp32^T@Q, recovering
near-fp32 precision (error ~3.9e-3 → ~7.7e-6).

Architecture constraints (DESIGN.md §17):
  - 11-kernel split (k3 split into k3c+k3+k3b, k5 split into k5c+k5a+k5b for Compensated GEMM)
  - Backward kernels use default threads (as (cid, vid), vid unused = no work split
    by vid). DESIGN.md §17.1 specifies threads=1, but the AscendWorkspaceReduction
    pass fails with "stoi" error when threads=1 is used with atomic_add (k3/k5).
    The verified prior impl (42/42 PASS) uses default threads with vid unused,
    which is logically single-core (no CV work split) and compiler-compatible.
  - 0 T.Scope / 0 cross_flag / 0 barrier_all / 0 annotate_address
  - pass_configs: Cube _hybrid (4 True) + Vector _vector (CV=False)
  - MEMORY_PLANNING on all kernels (TL_ENABLE_FAST_MATH not in this tilelang version)
  - GM workspace all fp32 (ws_s/ws_p/ws_dp/ws_ds)
  - host for-loop over KV blocks + torch.npu.synchronize()

Usage:
  python example_gqa_sink_bwd_varlen.py --level l0        # L0 precision gate
  python example_gqa_sink_bwd_varlen.py --level all       # full suite
  python example_gqa_sink_bwd_varlen.py --level bench     # do_bench smoke
  python example_gqa_sink_bwd_varlen.py --level msprof    # msprof op
"""

import sys

import tilelang
import torch
from tilelang import language as T

# ============================================================================
# pass_configs — 4 Ascend keys + TL_ENABLE_FAST_MATH (DESIGN.md §4.1)
# 0 T.annotate_address (MEMORY_PLANNING=True replaces manual annotation)
# ============================================================================

# NOTE: TL_ENABLE_FAST_MATH is listed in DESIGN.md §4.1 but does not exist in
# this tilelang version (see DESIGN.md §10.2 R5: "若无效可移除，不影响正确性").
# The verified prior impl (42/42 PASS) also omits it. Removed here.
_developer_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

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
BLOCK_M_BWD = 128
BLOCK_N_BWD = 128


# ============================================================================
# Forward kernel: online softmax + attention sink + causal/window mask (varlen)
# ============================================================================


@tilelang.jit(out_idx=[3, 4], pass_configs=_developer_pass_configs)
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
    """Forward kernel: online softmax + sink + window.

    Grid: (max_seq_len // block_M) * heads * batch
    threads: 1 (single core, no C<->V interaction)
    Outputs: O [UQ, H, dim] fp16, lse [batch, H, max_seq_len] fp32
    """
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
        assert window_size % block_N == 0, "window_size must be divisible by block_N"

    window_eff = window_size if window_size is not None else max_seq_len * 2

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(kv_shape, dtype),  # type: ignore
        V: T.Tensor(kv_shape, dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        Sinks: T.Tensor([heads], dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], "int32"),  # type: ignore
        cu_seqlens_k: T.Tensor([batch + 1], "int32"),  # type: ignore
    ):
        with T.Kernel(block_num, threads=1, is_npu=True) as cid:
            bx = cid % (max_seq_len // block_M)
            by = cid // (max_seq_len // block_M) % heads
            bz = cid // (max_seq_len // block_M) // heads % batch
            kv_by = by // groups

            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            kv_start_idx = cu_seqlens_k[bz]
            kv_end_idx = cu_seqlens_k[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = kv_end_idx - kv_start_idx
            offset = kv_current_seqlen - q_current_seqlen

            loop_st = T.max(0, (bx * block_M + offset - window_eff) // block_N)
            loop_ed = T.min(
                T.ceildiv((bx + 1) * block_M + offset, block_N),
                T.ceildiv(kv_current_seqlen, block_N),
            )

            if bx * block_M < q_current_seqlen:
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

                T.copy(Q[q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, by, :], q_l1)

                T.copy(Sinks[by : by + 1], sink_scalar)
                T.tile.fill(sink_ub, sink_scalar[0])

                for k in T.serial(loop_st, loop_ed):
                    T.copy(K[kv_start_idx + k * block_N : kv_start_idx + (k + 1) * block_N, kv_by, :], k_l1)
                    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                    T.copy(acc_s_l0c, acc_s_ub_)

                    T.tile.fill(acc_s_ub, 0.0)
                    T.copy(m_i, m_i_prev)
                    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)

                    T.tile.arith_progression(col_pos, k * block_N, 1, block_N)
                    T.tile.compare(win_mask, col_pos, kv_current_seqlen, "LT")
                    for h_i in range(block_M):
                        row_pos_val = (bx * block_M + h_i + offset) * 1.0
                        T.tile.compare(cmp_mask, col_pos, row_pos_val, "LE")
                        if window_size is not None:
                            T.tile.compare(combined_mask, col_pos, row_pos_val - window_size, "GT")
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
                    T.copy(V[kv_start_idx + k * block_N : kv_start_idx + (k + 1) * block_N, kv_by, :], v_l1)
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
                T.copy(
                    acc_o_half,
                    Output[q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, by, :],
                )

                T.tile.ln(sumexp, sumexp)
                T.tile.add(sumexp, sumexp, m_i)
                T.copy(sumexp, lse[bz, by, bx * block_M : (bx + 1) * block_M])

    return main


# ============================================================================
# Backward k0 (Vector, pre-loop): Delta = sum(O * dO, dim=-1) -> Delta (GM fp32)
# ============================================================================


@tilelang.jit(pass_configs=_vector_pass_configs)
def bwd_k0_preprocess(batch, UQ, max_seq_len, heads, dim_v, block_M):
    """k0: Delta = sum(O * dO, dim=-1) -> Delta (fp32). Pre-loop Vector kernel."""
    dtype = "float16"
    accum_dtype = "float"
    hm = block_M // 2
    o_shape = [UQ, heads, dim_v]
    do_shape = [UQ, heads, dim_v]
    lse_shape = [batch, heads, max_seq_len]
    bwd_block_num = (max_seq_len // block_M) * heads * batch

    @T.prim_func
    def main(
        O: T.Tensor(o_shape, dtype),  # type: ignore
        dO: T.Tensor(do_shape, dtype),  # type: ignore
        Delta: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], "int32"),  # type: ignore
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (max_seq_len // block_M)
            by = cid // (max_seq_len // block_M) % heads
            bz = cid // (max_seq_len // block_M) // heads % batch

            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx

            if bx * block_M < q_current_seqlen:
                o_ub = T.alloc_ub([hm, dim_v], dtype)
                do_ub = T.alloc_ub([hm, dim_v], dtype)
                prod_ub = T.alloc_ub([hm, dim_v], accum_dtype)
                do_fp32 = T.alloc_ub([hm, dim_v], accum_dtype)
                partial = T.alloc_ub([hm], accum_dtype)

                for h_idx in range(2):
                    v_row = h_idx * hm

                    T.copy(
                        O[q_start_idx + bx * block_M + v_row : q_start_idx + bx * block_M + v_row + hm, by, :dim_v],
                        o_ub,
                    )
                    T.copy(
                        dO[q_start_idx + bx * block_M + v_row : q_start_idx + bx * block_M + v_row + hm, by, :dim_v],
                        do_ub,
                    )
                    T.tile.cast(prod_ub, o_ub, "CAST_NONE", hm * dim_v)
                    T.tile.cast(do_fp32, do_ub, "CAST_NONE", hm * dim_v)
                    T.tile.mul(prod_ub, prod_ub, do_fp32)
                    T.reduce_sum(prod_ub, partial, dim=-1)
                    T.copy(partial, Delta[bz, by, bx * block_M + v_row : bx * block_M + v_row + hm])

    return main


# ============================================================================
# Backward k1 (Cube): S = Q @ K_k^T -> ws_s (GM fp32)
# ============================================================================


@tilelang.jit(pass_configs=_hybrid_pass_configs)
def bwd_k1_qkt(
    batch,
    UQ,
    UKV,
    max_seq_len,
    heads,
    dim_qk_padded,
    window_size,
    block_M,
    block_N,
    groups,
):
    """k1: S = Q @ K_k^T -> ws_s[cid] (fp32, unscaled)."""
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"
    q_shape = [UQ, heads, dim_qk_padded]
    k_shape = [UKV, head_kv, dim_qk_padded]
    bwd_block_num = (max_seq_len // block_M) * heads * batch
    ws_shape = [bwd_block_num, block_M, block_N]
    window_eff = window_size if window_size is not None else max_seq_len * 2

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        ws_s: T.Tensor(ws_shape, accum_dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], "int32"),  # type: ignore
        cu_seqlens_k: T.Tensor([batch + 1], "int32"),  # type: ignore
        k_iter: T.int32,
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (max_seq_len // block_M)
            by = cid // (max_seq_len // block_M) % heads
            bz = cid // (max_seq_len // block_M) // heads % batch
            kv_by = by // groups

            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            kv_start_idx = cu_seqlens_k[bz]
            kv_end_idx = cu_seqlens_k[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = kv_end_idx - kv_start_idx
            offset = kv_current_seqlen - q_current_seqlen

            loop_st = T.max(0, (bx * block_M + offset - window_eff) // block_N)
            loop_ed = T.min(
                T.ceildiv((bx + 1) * block_M + offset, block_N),
                T.ceildiv(kv_current_seqlen, block_N),
            )

            if bx * block_M < q_current_seqlen:  # noqa: SIM102
                if (k_iter >= loop_st) and (k_iter < loop_ed):
                    q_l1 = T.alloc_L1([block_M, dim_qk_padded], dtype)
                    k_l1 = T.alloc_L1([block_N, dim_qk_padded], dtype)
                    l0c_s = T.alloc_L0C([block_M, block_N], accum_dtype)

                    T.copy(
                        Q[q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, by, :],
                        q_l1,
                    )
                    T.copy(
                        K[kv_start_idx + k_iter * block_N : kv_start_idx + (k_iter + 1) * block_N, kv_by, :],
                        k_l1,
                    )
                    T.gemm_v0(q_l1, k_l1, l0c_s, transpose_B=True, init=True)
                    T.copy(l0c_s, ws_s[cid, :, :])

    return main


# ============================================================================
# Backward k2 (Vector): P = softmax(S) -> ws_p (GM fp32)
# ============================================================================


@tilelang.jit(pass_configs=_vector_pass_configs)
def bwd_k2_softmax(
    batch,
    UQ,
    UKV,
    max_seq_len,
    heads,
    dim_qk_padded,
    dim_qk,
    dim_v,
    window_size,
    block_M,
    block_N,
    groups,
):
    """k2: P = exp(S*scale - lse + mask) -> ws_p (fp32)."""
    sm_scale = (1.0 / dim_qk) ** 0.5
    dtype = "float16"
    accum_dtype = "float"
    hm = block_M // 2
    ws_shape = [batch * (max_seq_len // block_M) * heads, block_M, block_N]
    lse_shape = [batch, heads, max_seq_len]
    o_shape = [UQ, heads, dim_v]
    do_shape = [UQ, heads, dim_v]
    bwd_block_num = (max_seq_len // block_M) * heads * batch
    window_eff = window_size if window_size is not None else max_seq_len * 2

    @T.prim_func
    def main(
        ws_s: T.Tensor(ws_shape, accum_dtype),  # type: ignore
        lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        O: T.Tensor(o_shape, dtype),  # type: ignore
        dO: T.Tensor(do_shape, dtype),  # type: ignore
        ws_p: T.Tensor(ws_shape, accum_dtype),  # type: ignore
        Delta: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], "int32"),  # type: ignore
        cu_seqlens_k: T.Tensor([batch + 1], "int32"),  # type: ignore
        k_iter: T.int32,
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (max_seq_len // block_M)
            by = cid // (max_seq_len // block_M) % heads
            bz = cid // (max_seq_len // block_M) // heads % batch

            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            kv_start_idx = cu_seqlens_k[bz]
            kv_end_idx = cu_seqlens_k[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = kv_end_idx - kv_start_idx
            offset = kv_current_seqlen - q_current_seqlen

            loop_st = T.max(0, (bx * block_M + offset - window_eff) // block_N)
            loop_ed = T.min(
                T.ceildiv((bx + 1) * block_M + offset, block_N),
                T.ceildiv(kv_current_seqlen, block_N),
            )

            if bx * block_M < q_current_seqlen:  # noqa: SIM102
                if (k_iter >= loop_st) and (k_iter < loop_ed):
                    work_ub = T.alloc_ub([hm, block_N], accum_dtype)
                    lse_ub = T.alloc_ub([hm], accum_dtype)
                    lse_ub_full = T.alloc_ub([block_M], accum_dtype)
                    temp_ub = T.alloc_ub([hm, block_N], accum_dtype)
                    col_pos = T.alloc_ub([block_N], accum_dtype)
                    win_mask = T.alloc_ub([block_N], accum_dtype)
                    cmp_mask = T.alloc_ub([block_N], accum_dtype)
                    combined = T.alloc_ub([block_N], accum_dtype)
                    row_pos_1d = T.alloc_ub([hm], accum_dtype)
                    row_pos_2d = T.alloc_ub([hm, block_N], accum_dtype)
                    combined_2d = T.alloc_ub([hm, block_N], accum_dtype)

                    T.tile.arith_progression(col_pos, k_iter * block_N, 1, block_N)
                    T.tile.compare(win_mask, col_pos, kv_current_seqlen, "LT")

                    T.copy(lse[bz, by, bx * block_M : bx * block_M + block_M], lse_ub_full)

                    mask_k_start = (
                        T.max(
                            T.min(
                                (T.max(bx * block_M + offset, 0) + 1) // block_N,
                                kv_current_seqlen // block_N,
                            ),
                            loop_st,
                        )
                        if window_size is None
                        else loop_st
                    )

                    if window_size is None:  # noqa: SIM102
                        if k_iter >= mask_k_start:
                            T.tile.select(
                                combined,
                                win_mask,
                                col_pos,
                                T.infinity(accum_dtype),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )

                    for h_idx in range(2):
                        v_row = h_idx * hm

                        T.copy(ws_s[cid, v_row : v_row + hm, :], temp_ub)

                        T.tile.fill(work_ub, 0.0)
                        T.tile.axpy(work_ub, temp_ub, sm_scale)

                        T.copy(lse_ub_full[v_row : v_row + hm], lse_ub)
                        T.tile.broadcast(temp_ub, lse_ub, axis=1)
                        T.tile.sub(work_ub, work_ub, temp_ub)

                        T.tile.exp(work_ub, work_ub)

                        if k_iter >= mask_k_start:
                            if window_size is not None:
                                for h_i in range(hm):
                                    row_pos_val = (bx * block_M + v_row + h_i + offset) * 1.0
                                    T.tile.compare(cmp_mask, col_pos, row_pos_val, "LE")
                                    T.tile.compare(combined, col_pos, row_pos_val - window_size, "GT")
                                    T.tile.bitwise_and(combined, combined, cmp_mask)
                                    T.tile.bitwise_and(combined, combined, win_mask)
                                    T.tile.select(
                                        work_ub[h_i, :],
                                        combined,
                                        work_ub[h_i, :],
                                        0.0,
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                            else:
                                T.tile.arith_progression(
                                    row_pos_1d,
                                    (bx * block_M + v_row + offset) * 1.0,
                                    1,
                                    hm,
                                )
                                T.tile.broadcast(row_pos_2d, row_pos_1d, axis=1)
                                T.tile.broadcast(combined_2d, combined, axis=0)
                                T.tile.compare(temp_ub, combined_2d, row_pos_2d, "LE")
                                T.tile.select(
                                    work_ub,
                                    temp_ub,
                                    work_ub,
                                    0.0,
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )

                        T.copy(work_ub, ws_p[cid, v_row : v_row + hm, :])

    return main


# ============================================================================
# Backward k3c (Vector): p_delta_fp16 = cast_fp16(ws_p_fp32 - cast_fp32(cast_fp16(ws_p_fp32)))
# Compensated GEMM residual for dV: captures fp16 cast loss of P. k3 uses main
# GEMM (P_fp16^T@dO), k3b uses correction GEMM (p_delta^T@dO). Together:
# dV ≈ P_fp16^T@dO + p_delta^T@dO ≈ ws_p_fp32^T@dO (fp32 precision).
# ============================================================================


@tilelang.jit(pass_configs=_vector_pass_configs)
def bwd_k3c_p_delta(
    batch,
    UQ,
    UKV,
    max_seq_len,
    heads,
    window_size,
    block_M,
    block_N,
    groups,
):
    """k3c: p_delta_fp16 = cast_fp16(ws_p_fp32 - cast_fp32(cast_fp16(ws_p_fp32))).

    Pure Vector kernel. Compensated GEMM residual computation for dV.
    Reads ws_p (fp32 GM), writes ws_p_delta (fp16 GM, same block shape).
    """
    dtype = "float16"
    accum_dtype = "float"
    hm = block_M // 2
    ws_shape = [batch * (max_seq_len // block_M) * heads, block_M, block_N]
    bwd_block_num = (max_seq_len // block_M) * heads * batch
    window_eff = window_size if window_size is not None else max_seq_len * 2

    @T.prim_func
    def main(
        ws_p: T.Tensor(ws_shape, accum_dtype),  # type: ignore
        ws_p_delta: T.Tensor(ws_shape, dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], "int32"),  # type: ignore
        cu_seqlens_k: T.Tensor([batch + 1], "int32"),  # type: ignore
        k_iter: T.int32,
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (max_seq_len // block_M)
            _by = cid // (max_seq_len // block_M) % heads
            bz = cid // (max_seq_len // block_M) // heads % batch

            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            kv_start_idx = cu_seqlens_k[bz]
            kv_end_idx = cu_seqlens_k[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = kv_end_idx - kv_start_idx
            offset = kv_current_seqlen - q_current_seqlen

            loop_st = T.max(0, (bx * block_M + offset - window_eff) // block_N)
            loop_ed = T.min(
                T.ceildiv((bx + 1) * block_M + offset, block_N),
                T.ceildiv(kv_current_seqlen, block_N),
            )

            if bx * block_M < q_current_seqlen:  # noqa: SIM102
                if (k_iter >= loop_st) and (k_iter < loop_ed):
                    p_ub = T.alloc_ub([hm, block_N], accum_dtype)
                    p_half_ub = T.alloc_ub([hm, block_N], dtype)
                    p_rec_ub = T.alloc_ub([hm, block_N], accum_dtype)
                    p_delta_ub = T.alloc_ub([hm, block_N], accum_dtype)
                    p_delta_half_ub = T.alloc_ub([hm, block_N], dtype)

                    for h_idx in range(2):
                        v_row = h_idx * hm

                        T.copy(ws_p[cid, v_row : v_row + hm, :], p_ub)

                        # P_fp16 = cast(ws_p_fp32, fp16, CAST_RINT) — same cast as k3
                        T.tile.cast(p_half_ub, p_ub, "CAST_RINT", hm * block_N)

                        # P_fp16_as_fp32 = cast(P_fp16, fp32) — exact via T.copy auto-cast
                        T.copy(p_half_ub, p_rec_ub)

                        # p_delta_fp32 = ws_p_fp32 - P_fp16_as_fp32
                        T.tile.sub(p_delta_ub, p_ub, p_rec_ub)

                        # p_delta_fp16 = cast(p_delta_fp32, fp16, CAST_RINT)
                        T.tile.cast(p_delta_half_ub, p_delta_ub, "CAST_RINT", hm * block_N)

                        T.copy(p_delta_half_ub, ws_p_delta[cid, v_row : v_row + hm, :])

    return main


# ============================================================================
# Backward k3 (Cube): dV += P^T@dO + dP = dO@V^T -> dV (atomic) + ws_dp (GM fp32)
# ============================================================================


@tilelang.jit(pass_configs=_hybrid_pass_configs)
def bwd_k3_dv_dp(
    batch,
    UQ,
    UKV,
    max_seq_len,
    heads,
    dim_qk_padded,
    dim_v,
    window_size,
    block_M,
    block_N,
    groups,
):
    """k3: dV += P^T @ dO (atomic_add) + dP = dO @ V^T -> ws_dp (fp32)."""
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"
    ws_shape = [batch * (max_seq_len // block_M) * heads, block_M, block_N]
    do_shape = [UQ, heads, dim_v]
    v_shape = [UKV, head_kv, dim_v]
    dv_shape = [UKV, head_kv, dim_v]
    bwd_block_num = (max_seq_len // block_M) * heads * batch
    window_eff = window_size if window_size is not None else max_seq_len * 2

    @T.prim_func
    def main(
        ws_p: T.Tensor(ws_shape, accum_dtype),  # type: ignore
        dO: T.Tensor(do_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        dV: T.Tensor(dv_shape, accum_dtype),  # type: ignore
        ws_dp: T.Tensor(ws_shape, accum_dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], "int32"),  # type: ignore
        cu_seqlens_k: T.Tensor([batch + 1], "int32"),  # type: ignore
        k_iter: T.int32,
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (max_seq_len // block_M)
            by = cid // (max_seq_len // block_M) % heads
            bz = cid // (max_seq_len // block_M) // heads % batch
            kv_by = by // groups

            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            kv_start_idx = cu_seqlens_k[bz]
            kv_end_idx = cu_seqlens_k[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = kv_end_idx - kv_start_idx
            offset = kv_current_seqlen - q_current_seqlen

            loop_st = T.max(0, (bx * block_M + offset - window_eff) // block_N)
            loop_ed = T.min(
                T.ceildiv((bx + 1) * block_M + offset, block_N),
                T.ceildiv(kv_current_seqlen, block_N),
            )

            if bx * block_M < q_current_seqlen:  # noqa: SIM102
                if (k_iter >= loop_st) and (k_iter < loop_ed):
                    p_l1 = T.alloc_L1([block_M, block_N], dtype)
                    p_ub = T.alloc_ub([block_M, block_N], accum_dtype)
                    p_half_ub = T.alloc_ub([block_M, block_N], dtype)
                    do_l1 = T.alloc_L1([block_M, dim_v], dtype)
                    v_l1 = T.alloc_L1([block_N, dim_v], dtype)

                    T.copy(ws_p[cid, :, :], p_ub)
                    T.tile.cast(p_half_ub, p_ub, "CAST_RINT", block_M * block_N)
                    T.copy(p_half_ub, p_l1)

                    T.copy(
                        dO[q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, by, :dim_v],
                        do_l1,
                    )

                    l0c_dv = T.alloc_L0C([block_N, dim_v], accum_dtype)
                    T.gemm_v0(p_l1, do_l1, l0c_dv, transpose_A=True, init=True)
                    T.tile.atomic_add(
                        dV[kv_start_idx + k_iter * block_N : kv_start_idx + (k_iter + 1) * block_N, kv_by, :dim_v],
                        l0c_dv,
                    )

                    T.copy(
                        V[kv_start_idx + k_iter * block_N : kv_start_idx + (k_iter + 1) * block_N, kv_by, :dim_v],
                        v_l1,
                    )
                    l0c_dp = T.alloc_L0C([block_M, block_N], accum_dtype)
                    T.gemm_v0(do_l1, v_l1, l0c_dp, transpose_B=True, init=True)
                    T.copy(l0c_dp, ws_dp[cid, :, :])

    return main


# ============================================================================
# Backward k3b (Cube): dV += p_delta_fp16^T @ dO (Compensated GEMM correction)
# Pure single-GEMM Cube kernel. Reads ws_p_delta (fp16 GM) directly to L1
# (no Vector cast). atomic_add to same dV as k3 (main GEMM).
# Together: dV = P_fp16^T@dO + p_delta^T@dO ≈ ws_p_fp32^T@dO (fp32 precision).
# ============================================================================


@tilelang.jit(pass_configs=_hybrid_pass_configs)
def bwd_k3b_dv_correction(
    batch,
    UQ,
    UKV,
    max_seq_len,
    heads,
    dim_v,
    window_size,
    block_M,
    block_N,
    groups,
):
    """k3b: dV += p_delta_fp16^T @ dO (Compensated GEMM correction, atomic_add)."""
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"
    ws_shape = [batch * (max_seq_len // block_M) * heads, block_M, block_N]
    do_shape = [UQ, heads, dim_v]
    dv_shape = [UKV, head_kv, dim_v]
    bwd_block_num = (max_seq_len // block_M) * heads * batch
    window_eff = window_size if window_size is not None else max_seq_len * 2

    @T.prim_func
    def main(
        ws_p_delta: T.Tensor(ws_shape, dtype),  # type: ignore
        dO: T.Tensor(do_shape, dtype),  # type: ignore
        dV: T.Tensor(dv_shape, accum_dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], "int32"),  # type: ignore
        cu_seqlens_k: T.Tensor([batch + 1], "int32"),  # type: ignore
        k_iter: T.int32,
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (max_seq_len // block_M)
            by = cid // (max_seq_len // block_M) % heads
            bz = cid // (max_seq_len // block_M) // heads % batch
            kv_by = by // groups

            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            kv_start_idx = cu_seqlens_k[bz]
            kv_end_idx = cu_seqlens_k[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = kv_end_idx - kv_start_idx
            offset = kv_current_seqlen - q_current_seqlen

            loop_st = T.max(0, (bx * block_M + offset - window_eff) // block_N)
            loop_ed = T.min(
                T.ceildiv((bx + 1) * block_M + offset, block_N),
                T.ceildiv(kv_current_seqlen, block_N),
            )

            if bx * block_M < q_current_seqlen:  # noqa: SIM102
                if (k_iter >= loop_st) and (k_iter < loop_ed):
                    p_delta_l1 = T.alloc_L1([block_M, block_N], dtype)
                    do_l1 = T.alloc_L1([block_M, dim_v], dtype)

                    T.copy(ws_p_delta[cid, :, :], p_delta_l1)
                    T.copy(
                        dO[q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, by, :dim_v],
                        do_l1,
                    )

                    l0c_dv = T.alloc_L0C([block_N, dim_v], accum_dtype)
                    T.gemm_v0(p_delta_l1, do_l1, l0c_dv, transpose_A=True, init=True)
                    T.tile.atomic_add(
                        dV[kv_start_idx + k_iter * block_N : kv_start_idx + (k_iter + 1) * block_N, kv_by, :dim_v],
                        l0c_dv,
                    )

    return main


# ============================================================================
# Backward k4 (Vector): dS = P * (dP - Delta) * scale -> ws_ds (GM fp32)
# ============================================================================


@tilelang.jit(pass_configs=_vector_pass_configs)
def bwd_k4_ds(
    batch,
    UQ,
    UKV,
    max_seq_len,
    heads,
    dim_qk_padded,
    dim_qk,
    dim_v,
    window_size,
    block_M,
    block_N,
    groups,
):
    """k4: dS = P * (dP - Delta) * scale -> ws_ds (fp32)."""
    sm_scale = (1.0 / dim_qk) ** 0.5
    accum_dtype = "float"
    hm = block_M // 2
    ws_shape = [batch * (max_seq_len // block_M) * heads, block_M, block_N]
    lse_shape = [batch, heads, max_seq_len]
    bwd_block_num = (max_seq_len // block_M) * heads * batch
    window_eff = window_size if window_size is not None else max_seq_len * 2

    @T.prim_func
    def main(
        ws_dp: T.Tensor(ws_shape, accum_dtype),  # type: ignore
        ws_p: T.Tensor(ws_shape, accum_dtype),  # type: ignore
        Delta: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        ws_ds: T.Tensor(ws_shape, accum_dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], "int32"),  # type: ignore
        cu_seqlens_k: T.Tensor([batch + 1], "int32"),  # type: ignore
        k_iter: T.int32,
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (max_seq_len // block_M)
            by = cid // (max_seq_len // block_M) % heads
            bz = cid // (max_seq_len // block_M) // heads % batch

            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            kv_start_idx = cu_seqlens_k[bz]
            kv_end_idx = cu_seqlens_k[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = kv_end_idx - kv_start_idx
            offset = kv_current_seqlen - q_current_seqlen

            loop_st = T.max(0, (bx * block_M + offset - window_eff) // block_N)
            loop_ed = T.min(
                T.ceildiv((bx + 1) * block_M + offset, block_N),
                T.ceildiv(kv_current_seqlen, block_N),
            )

            if bx * block_M < q_current_seqlen:  # noqa: SIM102
                if (k_iter >= loop_st) and (k_iter < loop_ed):
                    dp_ub = T.alloc_ub([hm, block_N], accum_dtype)
                    p_ub = T.alloc_ub([hm, block_N], accum_dtype)
                    work_ub = T.alloc_ub([hm, block_N], accum_dtype)
                    delta_ub = T.alloc_ub([hm], accum_dtype)
                    delta_2d = T.alloc_ub([hm, block_N], accum_dtype)

                    for h_idx in range(2):
                        v_row = h_idx * hm

                        T.copy(ws_dp[cid, v_row : v_row + hm, :], dp_ub)
                        T.copy(ws_p[cid, v_row : v_row + hm, :], p_ub)

                        T.copy(Delta[bz, by, bx * block_M + v_row : bx * block_M + v_row + hm], delta_ub)
                        T.tile.broadcast(delta_2d, delta_ub, axis=1)

                        T.copy(p_ub, work_ub)
                        T.tile.sub(dp_ub, dp_ub, delta_2d)
                        T.tile.mul(work_ub, work_ub, dp_ub)
                        T.tile.mul(work_ub, work_ub, sm_scale)

                        T.copy(work_ub, ws_ds[cid, v_row : v_row + hm, :])

    return main


# ============================================================================
# Backward k5c (Vector): ds_delta_fp16 = cast_fp16(ws_ds_fp32 - cast_fp32(cast_fp16(ws_ds_fp32)))
# Compensated GEMM residual: captures fp16 cast loss of dS. k5a uses main GEMM
# (dS_fp16), k5b uses correction GEMM (ds_delta_fp16). Together they recover
# near-fp32 precision: dK ≈ dS_fp16^T@Q + ds_delta^T@Q ≈ ws_ds_fp32^T@Q.
# ============================================================================


@tilelang.jit(pass_configs=_vector_pass_configs)
def bwd_k5c_ds_delta(
    batch,
    UQ,
    UKV,
    max_seq_len,
    heads,
    window_size,
    block_M,
    block_N,
    groups,
):
    """k5c: ds_delta_fp16 = cast_fp16(ws_ds_fp32 - cast_fp32(cast_fp16(ws_ds_fp32))).

    Pure Vector kernel. Compensated GEMM residual computation.
    Reads ws_ds (fp32 GM), writes ws_ds_delta (fp16 GM, same block shape).
    Pattern matches bhsd k4 (lines 718-726): dS_fp16=cast(dS_fp32,CAST_RINT),
    ds_delta=dS_fp32-cast(dS_fp16,fp32), ds_delta_fp16=cast(ds_delta,CAST_RINT).
    """
    dtype = "float16"
    accum_dtype = "float"
    hm = block_M // 2
    ws_shape = [batch * (max_seq_len // block_M) * heads, block_M, block_N]
    bwd_block_num = (max_seq_len // block_M) * heads * batch
    window_eff = window_size if window_size is not None else max_seq_len * 2

    @T.prim_func
    def main(
        ws_ds: T.Tensor(ws_shape, accum_dtype),  # type: ignore
        ws_ds_delta: T.Tensor(ws_shape, dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], "int32"),  # type: ignore
        cu_seqlens_k: T.Tensor([batch + 1], "int32"),  # type: ignore
        k_iter: T.int32,
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (max_seq_len // block_M)
            _by = cid // (max_seq_len // block_M) % heads
            bz = cid // (max_seq_len // block_M) // heads % batch

            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            kv_start_idx = cu_seqlens_k[bz]
            kv_end_idx = cu_seqlens_k[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = kv_end_idx - kv_start_idx
            offset = kv_current_seqlen - q_current_seqlen

            loop_st = T.max(0, (bx * block_M + offset - window_eff) // block_N)
            loop_ed = T.min(
                T.ceildiv((bx + 1) * block_M + offset, block_N),
                T.ceildiv(kv_current_seqlen, block_N),
            )

            if bx * block_M < q_current_seqlen:  # noqa: SIM102
                if (k_iter >= loop_st) and (k_iter < loop_ed):
                    ds_ub = T.alloc_ub([hm, block_N], accum_dtype)
                    ds_half_ub = T.alloc_ub([hm, block_N], dtype)
                    ds_rec_ub = T.alloc_ub([hm, block_N], accum_dtype)
                    ds_delta_ub = T.alloc_ub([hm, block_N], accum_dtype)
                    ds_delta_half_ub = T.alloc_ub([hm, block_N], dtype)

                    for h_idx in range(2):
                        v_row = h_idx * hm

                        T.copy(ws_ds[cid, v_row : v_row + hm, :], ds_ub)

                        # dS_fp16 = cast(ws_ds_fp32, fp16, CAST_RINT) — same cast as k5a
                        T.tile.cast(ds_half_ub, ds_ub, "CAST_RINT", hm * block_N)

                        # dS_fp16_as_fp32 = cast(dS_fp16, fp32) — exact via T.copy auto-cast
                        T.copy(ds_half_ub, ds_rec_ub)

                        # ds_delta_fp32 = ws_ds_fp32 - dS_fp16_as_fp32
                        T.tile.sub(ds_delta_ub, ds_ub, ds_rec_ub)

                        # ds_delta_fp16 = cast(ds_delta_fp32, fp16, CAST_RINT)
                        T.tile.cast(ds_delta_half_ub, ds_delta_ub, "CAST_RINT", hm * block_N)

                        T.copy(ds_delta_half_ub, ws_ds_delta[cid, v_row : v_row + hm, :])

    return main


# ============================================================================
# Backward k5 (Cube): dK += dS^T@Q + dQ += dS@K -> dK (atomic) + dQ (atomic)
# ============================================================================


@tilelang.jit(pass_configs=_hybrid_pass_configs)
def bwd_k5_dk_dq(
    batch,
    UQ,
    UKV,
    max_seq_len,
    heads,
    dim_qk_padded,
    window_size,
    block_M,
    block_N,
    groups,
):
    """k5: dK += dS^T @ Q (atomic_add) + dQ += dS @ K (atomic_add)."""
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"
    ws_shape = [batch * (max_seq_len // block_M) * heads, block_M, block_N]
    q_shape = [UQ, heads, dim_qk_padded]
    k_shape = [UKV, head_kv, dim_qk_padded]
    dk_per_head_shape = [heads, UKV, dim_qk_padded]
    dq_shape = [UQ, heads, dim_qk_padded]
    bwd_block_num = (max_seq_len // block_M) * heads * batch
    window_eff = window_size if window_size is not None else max_seq_len * 2

    @T.prim_func
    def main(
        ws_ds: T.Tensor(ws_shape, accum_dtype),  # type: ignore
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        dK_per_head: T.Tensor(dk_per_head_shape, accum_dtype),  # type: ignore
        dQ: T.Tensor(dq_shape, accum_dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], "int32"),  # type: ignore
        cu_seqlens_k: T.Tensor([batch + 1], "int32"),  # type: ignore
        k_iter: T.int32,
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (max_seq_len // block_M)
            by = cid // (max_seq_len // block_M) % heads
            bz = cid // (max_seq_len // block_M) // heads % batch
            kv_by = by // groups

            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            kv_start_idx = cu_seqlens_k[bz]
            kv_end_idx = cu_seqlens_k[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = kv_end_idx - kv_start_idx
            offset = kv_current_seqlen - q_current_seqlen

            loop_st = T.max(0, (bx * block_M + offset - window_eff) // block_N)
            loop_ed = T.min(
                T.ceildiv((bx + 1) * block_M + offset, block_N),
                T.ceildiv(kv_current_seqlen, block_N),
            )

            if bx * block_M < q_current_seqlen:  # noqa: SIM102
                if (k_iter >= loop_st) and (k_iter < loop_ed):
                    ds_l1 = T.alloc_L1([block_M, block_N], dtype)
                    ds_ub = T.alloc_ub([block_M, block_N], accum_dtype)
                    ds_half_ub = T.alloc_ub([block_M, block_N], dtype)
                    q_l1 = T.alloc_L1([block_M, dim_qk_padded], dtype)
                    k_l1 = T.alloc_L1([block_N, dim_qk_padded], dtype)

                    T.copy(
                        Q[q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, by, :],
                        q_l1,
                    )

                    T.copy(ws_ds[cid, :, :], ds_ub)
                    T.tile.cast(ds_half_ub, ds_ub, "CAST_RINT", block_M * block_N)
                    T.copy(ds_half_ub, ds_l1)

                    l0c_dk = T.alloc_L0C([block_N, dim_qk_padded], accum_dtype)
                    T.gemm_v0(ds_l1, q_l1, l0c_dk, transpose_A=True, init=True, kL0Size=32)
                    T.tile.atomic_add(
                        dK_per_head[
                            by,
                            kv_start_idx + k_iter * block_N : kv_start_idx + (k_iter + 1) * block_N,
                            :dim_qk_padded,
                        ],
                        l0c_dk,
                    )

                    T.copy(
                        K[kv_start_idx + k_iter * block_N : kv_start_idx + (k_iter + 1) * block_N, kv_by, :],
                        k_l1,
                    )
                    l0c_dq = T.alloc_L0C([block_M, dim_qk_padded], accum_dtype)
                    T.gemm_v0(ds_l1, k_l1, l0c_dq, init=True, kL0Size=32)
                    T.tile.atomic_add(
                        dQ[q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, by, :dim_qk_padded],
                        l0c_dq,
                    )

    return main


# ============================================================================
# Backward k5b (Cube): dK += ds_delta_fp16^T @ Q (Compensated GEMM correction)
# Pure single-GEMM Cube kernel. Reads ws_ds_delta (fp16 GM) directly to L1
# (no Vector cast, no init=False). atomic_add to same dK_per_head as k5a (main
# GEMM). Together: dK = dS_fp16^T@Q + ds_delta^T@Q ≈ ws_ds_fp32^T@Q (fp32 prec).
# ============================================================================


@tilelang.jit(pass_configs=_hybrid_pass_configs)
def bwd_k5b_dk_correction(
    batch,
    UQ,
    UKV,
    max_seq_len,
    heads,
    dim_qk_padded,
    window_size,
    block_M,
    block_N,
    groups,
):
    """k5b: dK += ds_delta_fp16^T @ Q (Compensated GEMM correction, atomic_add)."""
    dtype = "float16"
    accum_dtype = "float"
    ws_shape = [batch * (max_seq_len // block_M) * heads, block_M, block_N]
    q_shape = [UQ, heads, dim_qk_padded]
    dk_per_head_shape = [heads, UKV, dim_qk_padded]
    bwd_block_num = (max_seq_len // block_M) * heads * batch
    window_eff = window_size if window_size is not None else max_seq_len * 2

    @T.prim_func
    def main(
        ws_ds_delta: T.Tensor(ws_shape, dtype),  # type: ignore
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        dK_per_head: T.Tensor(dk_per_head_shape, accum_dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], "int32"),  # type: ignore
        cu_seqlens_k: T.Tensor([batch + 1], "int32"),  # type: ignore
        k_iter: T.int32,
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (max_seq_len // block_M)
            by = cid // (max_seq_len // block_M) % heads
            bz = cid // (max_seq_len // block_M) // heads % batch

            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            kv_start_idx = cu_seqlens_k[bz]
            kv_end_idx = cu_seqlens_k[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = kv_end_idx - kv_start_idx
            offset = kv_current_seqlen - q_current_seqlen

            loop_st = T.max(0, (bx * block_M + offset - window_eff) // block_N)
            loop_ed = T.min(
                T.ceildiv((bx + 1) * block_M + offset, block_N),
                T.ceildiv(kv_current_seqlen, block_N),
            )

            if bx * block_M < q_current_seqlen:  # noqa: SIM102
                if (k_iter >= loop_st) and (k_iter < loop_ed):
                    ds_delta_l1 = T.alloc_L1([block_M, block_N], dtype)
                    q_l1 = T.alloc_L1([block_M, dim_qk_padded], dtype)

                    T.copy(ws_ds_delta[cid, :, :], ds_delta_l1)
                    T.copy(
                        Q[q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, by, :],
                        q_l1,
                    )

                    l0c_dk = T.alloc_L0C([block_N, dim_qk_padded], accum_dtype)
                    T.gemm_v0(ds_delta_l1, q_l1, l0c_dk, transpose_A=True, init=True, kL0Size=32)
                    T.tile.atomic_add(
                        dK_per_head[
                            by,
                            kv_start_idx + k_iter * block_N : kv_start_idx + (k_iter + 1) * block_N,
                            :dim_qk_padded,
                        ],
                        l0c_dk,
                    )

    return main


# ============================================================================
# Backward k6 (Vector, post-loop): dQ_fp16=cast(dQ) + dSink -> dQ_fp16 + dSinks_out
# ============================================================================


@tilelang.jit(pass_configs=_vector_pass_configs)
def bwd_k6_postprocess(batch, UQ, max_seq_len, heads, dim_qk_padded, block_M):
    """k6: dQ_fp16 = cast(dQ fp32) + dSink = -exp(sink - lse) * Delta."""
    dtype = "float16"
    accum_dtype = "float"
    hm = block_M // 2
    dq_shape = [UQ, heads, dim_qk_padded]
    dq_fp16_shape = [UQ, heads, dim_qk_padded]
    lse_shape = [batch, heads, max_seq_len]
    sinks_shape = [heads]
    dsinks_shape = [batch, heads, max_seq_len]
    bwd_block_num = (max_seq_len // block_M) * heads * batch

    @T.prim_func
    def main(
        dQ: T.Tensor(dq_shape, accum_dtype),  # type: ignore
        dQ_fp16: T.Tensor(dq_fp16_shape, dtype),  # type: ignore
        lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        Delta: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        Sinks: T.Tensor(sinks_shape, dtype),  # type: ignore
        dSinks_out: T.Tensor(dsinks_shape, dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch + 1], "int32"),  # type: ignore
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (max_seq_len // block_M)
            by = cid // (max_seq_len // block_M) % heads
            bz = cid // (max_seq_len // block_M) // heads % batch

            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx

            if bx * block_M < q_current_seqlen:
                work_ub = T.alloc_ub([hm, dim_qk_padded], accum_dtype)
                dq_half = T.alloc_ub([hm, dim_qk_padded], dtype)
                lse_ub = T.alloc_ub([hm], accum_dtype)
                delta_ub = T.alloc_ub([hm], accum_dtype)
                sink_scalar = T.alloc_ub([1], dtype)
                sink_val_ub = T.alloc_ub([hm], accum_dtype)
                sink_exp_ub = T.alloc_ub([hm], accum_dtype)
                dsink_half = T.alloc_ub([hm], dtype)

                T.copy(Sinks[by : by + 1], sink_scalar)
                T.tile.fill(sink_val_ub, sink_scalar[0])

                for h_idx in range(2):
                    v_row = h_idx * hm

                    T.copy(
                        dQ[q_start_idx + bx * block_M + v_row : q_start_idx + bx * block_M + v_row + hm, by, :],
                        work_ub,
                    )
                    T.tile.cast(dq_half, work_ub, "CAST_RINT", hm * dim_qk_padded)
                    T.copy(
                        dq_half,
                        dQ_fp16[q_start_idx + bx * block_M + v_row : q_start_idx + bx * block_M + v_row + hm, by, :],
                    )

                    T.copy(lse[bz, by, bx * block_M + v_row : bx * block_M + v_row + hm], lse_ub)
                    T.copy(Delta[bz, by, bx * block_M + v_row : bx * block_M + v_row + hm], delta_ub)

                    T.tile.sub(sink_exp_ub, sink_val_ub, lse_ub)
                    T.tile.exp(sink_exp_ub, sink_exp_ub)
                    T.tile.mul(sink_exp_ub, sink_exp_ub, delta_ub)
                    T.tile.mul(sink_exp_ub, sink_exp_ub, -1.0)

                    T.tile.cast(dsink_half, sink_exp_ub, "CAST_RINT", hm)
                    T.copy(
                        dsink_half,
                        dSinks_out[bz, by, bx * block_M + v_row : bx * block_M + v_row + hm],
                    )

    return main


# ============================================================================
# Host pipeline: run_bwd_pipeline
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
    """Run the 11-kernel backward pipeline on NPU.

    Returns:
        dQ_fp16: [UQ, heads, dim_qk_padded] fp16
        dK: [UKV, head_kv, dim_qk_padded] fp32
        dV: [UKV, head_kv, dim_v] fp32
        dSinks_out: [batch, heads, max_seq_len] fp16
    """
    head_kv = heads // groups
    dim_qk_padded = ((dim_qk + 127) // 128) * 128
    bwd_block_num = (max_seq_len // block_M) * heads * batch
    max_kv_blocks = max_kv_len // block_N

    ws_s = torch.empty(bwd_block_num, block_M, block_N, dtype=torch.float32, device="npu")
    ws_p = torch.empty(bwd_block_num, block_M, block_N, dtype=torch.float32, device="npu")
    # Compensated GEMM (attempt 3): p_delta workspace for k3b dV correction GEMM.
    ws_p_delta = torch.empty(bwd_block_num, block_M, block_N, dtype=torch.float16, device="npu")
    ws_dp = torch.empty(bwd_block_num, block_M, block_N, dtype=torch.float32, device="npu")
    ws_ds = torch.empty(bwd_block_num, block_M, block_N, dtype=torch.float32, device="npu")
    # Compensated GEMM (attempt 3, Option B): ds_delta workspace for k5b correction GEMM.
    # ds_delta_fp16 = cast_fp16(ws_ds_fp32 - cast_fp32(cast_fp16(ws_ds_fp32)))
    ws_ds_delta = torch.empty(bwd_block_num, block_M, block_N, dtype=torch.float16, device="npu")
    Delta = torch.zeros(batch, heads, max_seq_len, dtype=torch.float32, device="npu")

    dQ = torch.zeros(UQ, heads, dim_qk_padded, dtype=torch.float32, device="npu")
    # precision_fix (attempt 2, scheme 2A): per-Q-head dK workspace.
    # Each Q-head writes to its own dK_per_head[by, :, :] via atomic_add,
    # eliminating cross-Q-head accumulation non-determinism.
    # Host reduces: dK = dK_per_head.reshape(head_kv, groups, UKV, dim).sum(1).permute(1,0,2)
    dK_per_head = torch.zeros(heads, UKV, dim_qk_padded, dtype=torch.float32, device="npu")
    dV = torch.zeros(UKV, head_kv, dim_v, dtype=torch.float32, device="npu")
    dQ_fp16 = torch.empty(UQ, heads, dim_qk_padded, dtype=torch.float16, device="npu")
    dSinks_out = torch.empty(batch, heads, max_seq_len, dtype=torch.float16, device="npu")

    k1_mod = bwd_k1_qkt(batch, UQ, UKV, max_seq_len, heads, dim_qk_padded, window_size, block_M, block_N, groups)
    k2_mod = bwd_k2_softmax(batch, UQ, UKV, max_seq_len, heads, dim_qk_padded, dim_qk, dim_v, window_size, block_M, block_N, groups)
    k3_mod = bwd_k3_dv_dp(batch, UQ, UKV, max_seq_len, heads, dim_qk_padded, dim_v, window_size, block_M, block_N, groups)
    k3c_mod = bwd_k3c_p_delta(batch, UQ, UKV, max_seq_len, heads, window_size, block_M, block_N, groups)
    k3b_mod = bwd_k3b_dv_correction(batch, UQ, UKV, max_seq_len, heads, dim_v, window_size, block_M, block_N, groups)
    k4_mod = bwd_k4_ds(batch, UQ, UKV, max_seq_len, heads, dim_qk_padded, dim_qk, dim_v, window_size, block_M, block_N, groups)
    k5_mod = bwd_k5_dk_dq(batch, UQ, UKV, max_seq_len, heads, dim_qk_padded, window_size, block_M, block_N, groups)
    k5c_mod = bwd_k5c_ds_delta(batch, UQ, UKV, max_seq_len, heads, window_size, block_M, block_N, groups)
    k5b_mod = bwd_k5b_dk_correction(batch, UQ, UKV, max_seq_len, heads, dim_qk_padded, window_size, block_M, block_N, groups)
    k6_mod = bwd_k6_postprocess(batch, UQ, max_seq_len, heads, dim_qk_padded, block_M)
    k0_mod = bwd_k0_preprocess(batch, UQ, max_seq_len, heads, dim_v, block_M)

    k0_mod(O, dO, Delta, cu_seqlens_q)
    torch.npu.synchronize()

    for k in range(max_kv_blocks):
        k1_mod(Q, K, ws_s, cu_seqlens_q, cu_seqlens_k, k)
        torch.npu.synchronize()
        k2_mod(ws_s, lse, O, dO, ws_p, Delta, cu_seqlens_q, cu_seqlens_k, k)
        torch.npu.synchronize()
        # Compensated GEMM (attempt 3): k3c computes p_delta, k3 does main dV GEMM
        # + dP, k3b adds dV correction GEMM.
        # iter6 O6: removed 2 syncs between k3c->k3->k3b. NPU executes kernel launches
        # in FIFO order; data deps (ws_p_delta k3c->k3b, dV atomic_add k3->k3b) are
        # respected by in-order execution. Sync after k3b (C->V transition to k4) kept.
        k3c_mod(ws_p, ws_p_delta, cu_seqlens_q, cu_seqlens_k, k)
        k3_mod(ws_p, dO, V, dV, ws_dp, cu_seqlens_q, cu_seqlens_k, k)
        k3b_mod(ws_p_delta, dO, dV, cu_seqlens_q, cu_seqlens_k, k)
        torch.npu.synchronize()
        k4_mod(ws_dp, ws_p, Delta, ws_ds, cu_seqlens_q, cu_seqlens_k, k)
        torch.npu.synchronize()
        # Compensated GEMM (attempt 3, Option B): k5c computes ds_delta,
        # k5a (k5_mod) does main GEMM (dK + dQ), k5b adds correction GEMM (dK only).
        # iter6 O6: removed 2 syncs between k5c->k5->k5b. NPU in-order execution
        # handles data deps (ws_ds_delta k5c->k5b, dK_per_head atomic_add k5->k5b).
        # Sync after k5b (C->V transition to next iter k1) kept.
        k5c_mod(ws_ds, ws_ds_delta, cu_seqlens_q, cu_seqlens_k, k)
        k5_mod(ws_ds, Q, K, dK_per_head, dQ, cu_seqlens_q, cu_seqlens_k, k)
        k5b_mod(ws_ds_delta, Q, dK_per_head, cu_seqlens_q, cu_seqlens_k, k)
        torch.npu.synchronize()

    k6_mod(dQ, dQ_fp16, lse, Delta, Sinks, dSinks_out, cu_seqlens_q)
    torch.npu.synchronize()

    # Host reduce: sum dK_per_head across Q-heads within each GQA group.
    # dK_per_head: [heads, UKV, dim] -> reshape [head_kv, groups, UKV, dim]
    #             -> sum(dim=1) -> [head_kv, UKV, dim] -> permute(1,0,2) -> [UKV, head_kv, dim]
    dK = dK_per_head.reshape(head_kv, groups, UKV, dim_qk_padded).sum(dim=1).permute(1, 0, 2)

    return dQ_fp16, dK, dV, dSinks_out


# ============================================================================
# Golden Reference (PyTorch fp32 autograd -> fp16)
# ============================================================================


def ref_fwd_varlen(Q, K, V, sinks, cu_seqlens_q, cu_seqlens_k, max_seq_len, window_size=None, groups=1):
    """Forward golden for varlen. Q [UQ,H,D], K/V [UKV,H_kv,D]."""
    UQ, H, D = Q.shape
    H_kv = K.shape[1]
    batch = cu_seqlens_q.shape[0] - 1
    sm_scale = 1.0 / D**0.5

    output = torch.zeros_like(Q)
    lse_out = torch.zeros(batch, H, max_seq_len, dtype=torch.float32, device=Q.device)

    for b in range(batch):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()

        q_len = q_end - q_start
        k_len = k_end - k_start
        if q_len == 0:
            continue

        q_seq = Q[q_start:q_end]
        k_seq = K[k_start:k_end]
        v_seq = V[k_start:k_end]

        q_seq = q_seq.view(q_len, H_kv, groups, D)
        k_seq = k_seq.unsqueeze(2)
        v_seq = v_seq.unsqueeze(2)

        logits = torch.einsum("qhgd,khgd->hgqk", q_seq.float(), k_seq.float()) * sm_scale

        offset = k_len - q_len
        pos_keys = torch.arange(k_len, device=Q.device).float()
        pos_queries = torch.arange(q_len, device=Q.device).float() + offset

        mask = pos_keys[None, :] > pos_queries[:, None]
        mask = mask.float().masked_fill(mask, float("-inf"))

        if window_size is not None:
            too_old = pos_keys[None, :] < (pos_queries[:, None] - window_size + 1)
            mask.masked_fill_(too_old, float("-inf"))

        logits = logits + mask[None, None, :, :]

        sinks_expanded = sinks.view(H_kv, groups, 1, 1).float()
        logits_max = torch.max(logits, dim=-1, keepdim=True).values
        logits_or_sinks_max = torch.maximum(sinks_expanded, logits_max)
        sinks_exp = torch.exp(sinks_expanded - logits_or_sinks_max)
        unnorm = torch.exp(logits - logits_or_sinks_max)
        normalizer = unnorm.sum(dim=-1, keepdim=True) + sinks_exp
        scores = unnorm / normalizer

        out = torch.einsum("hgqk,khgd->qhgd", scores, v_seq.float())
        out = out.reshape(q_len, H, D).to(Q.dtype)
        output[q_start:q_end] = out

        lse = torch.log(normalizer.squeeze(-1)) + logits_or_sinks_max.squeeze(-1)
        lse = lse.reshape(H, q_len)
        lse_out[b, :, :q_len] = lse

    return output, lse_out


def ref_bwd_varlen(Q, K, V, sinks, dO, cu_seqlens_q, cu_seqlens_k, max_seq_len, window_size=None, groups=1):
    """Backward golden via autograd. Returns dQ, dK, dV, dSinks (all fp16)."""
    Q_f = Q.float().requires_grad_(True)
    K_f = K.float().requires_grad_(True)
    V_f = V.float().requires_grad_(True)
    Sinks_f = sinks.float().requires_grad_(True)

    UQ, H, D = Q_f.shape
    H_kv = K_f.shape[1]
    batch = cu_seqlens_q.shape[0] - 1
    sm_scale = 1.0 / D**0.5

    output = torch.zeros_like(Q_f)

    for b in range(batch):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()

        q_len = q_end - q_start
        k_len = k_end - k_start
        if q_len == 0:
            continue

        q_seq = Q_f[q_start:q_end]
        k_seq = K_f[k_start:k_end]
        v_seq = V_f[k_start:k_end]

        q_seq = q_seq.view(q_len, H_kv, groups, D)
        k_seq = k_seq.unsqueeze(2)
        v_seq = v_seq.unsqueeze(2)

        logits = torch.einsum("qhgd,khgd->hgqk", q_seq, k_seq) * sm_scale

        offset = k_len - q_len
        pos_keys = torch.arange(k_len, device=Q_f.device).float()
        pos_queries = torch.arange(q_len, device=Q_f.device).float() + offset

        mask = pos_keys[None, :] > pos_queries[:, None]
        mask = mask.float().masked_fill(mask, float("-inf"))

        if window_size is not None:
            too_old = pos_keys[None, :] < (pos_queries[:, None] - window_size + 1)
            mask.masked_fill_(too_old, float("-inf"))

        logits = logits + mask[None, None, :, :]

        sinks_expanded = Sinks_f.view(H_kv, groups, 1, 1)
        logits_max = torch.max(logits, dim=-1, keepdim=True).values
        logits_or_sinks_max = torch.maximum(sinks_expanded, logits_max)
        sinks_exp = torch.exp(sinks_expanded - logits_or_sinks_max)
        unnorm = torch.exp(logits - logits_or_sinks_max)
        normalizer = unnorm.sum(dim=-1, keepdim=True) + sinks_exp
        scores = unnorm / normalizer

        out = torch.einsum("hgqk,khgd->qhgd", scores, v_seq)
        out = out.reshape(q_len, H, D)
        output[q_start:q_end] = out

    output.backward(dO.float())
    return Q_f.grad.half(), K_f.grad.half(), V_f.grad.half(), Sinks_f.grad.half()


# ============================================================================
# Precision standard: mixed tolerance + dual gate (precision-standard.md §4.1)
# ============================================================================


def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio)."""
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),
    }
    int_types = {"int8", "int16", "int32", "int64", "uint8"}
    if dtype in int_types:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Return (passed, matched_ratio, max_abs_error).

    Float dual-gate: matched_ratio >= required AND max_abs_error <= max_abs_error_limit.
    Int: exact match. inf/nan positions compared structurally (not counted in tolerance).
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    special = ~torch.isfinite(g)
    if special.any():  # noqa: SIM102
        if not torch.equal(torch.isnan(a[special]), torch.isnan(g[special])) or not torch.equal(
            torch.isinf(a[special]), torch.isinf(g[special])
        ):
            return False, 0.0, float("inf")
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    matched_ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error


# ============================================================================
# Main: smoke test (one L0 case, precision verified against PyTorch golden)
# ============================================================================


def main():
    """Smoke test: run one basic L0 case and verify precision.

    For the full layered test suite (L0/L1/L2/Boundary + do_bench + msprof),
    run: python test_gqa_sink_bwd_varlen.py --level all
    """
    tilelang.enable_cache()
    torch.set_default_device("npu")

    B, H, groups = 1, 4, 2
    q_lens, kv_lens = [128], [128]
    D = 128
    window_size = None
    H_kv = H // groups
    max_seq_len = max(q_lens)

    cu_seqlens_q = [0]
    for ql in q_lens:
        cu_seqlens_q.append(cu_seqlens_q[-1] + ql)
    cu_seqlens_k = [0]
    for kl in kv_lens:
        cu_seqlens_k.append(cu_seqlens_k[-1] + kl)
    UQ = cu_seqlens_q[-1]
    UKV = cu_seqlens_k[-1]

    cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="npu")
    cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="npu")

    torch.manual_seed(42)
    Q = torch.randn(UQ, H, D, dtype=torch.float16, device="npu")
    K = torch.randn(UKV, H_kv, D, dtype=torch.float16, device="npu")
    V = torch.randn(UKV, H_kv, D, dtype=torch.float16, device="npu")
    sinks = torch.randn(H, dtype=torch.float16, device="npu")
    dO = torch.randn(UQ, H, D, dtype=torch.float16, device="npu")

    print("Compiling forward kernel ...")
    fwd_mod = flashattn_fwd(B, UQ, UKV, max_seq_len, H, D, groups, window_size, BLOCK_M_FWD, BLOCK_N_FWD)
    O_npu, lse_npu = fwd_mod(Q, K, V, sinks, cu_seqlens_q_t, cu_seqlens_k_t)
    torch.npu.synchronize()

    O_ref, _ = ref_fwd_varlen(Q, K, V, sinks, cu_seqlens_q_t, cu_seqlens_k_t, max_seq_len, window_size, groups)
    fwd_pass, fwd_ratio, fwd_max = check_precision(O_npu.cpu(), O_ref.cpu(), DTYPE_FP16)
    print(f"Forward precision: ratio={fwd_ratio:.4f}, max_abs={fwd_max:.3e}")

    print("Compiling backward pipeline (11-kernel split) ...")
    dQ_fp16, dK, dV, dSinks_out = run_bwd_pipeline(
        Q,
        K,
        V,
        O_npu,
        dO,
        lse_npu,
        sinks,
        cu_seqlens_q_t,
        cu_seqlens_k_t,
        B,
        UQ,
        UKV,
        max_seq_len,
        max(kv_lens),
        H,
        D,
        D,
        window_size,
        BLOCK_M_BWD,
        BLOCK_N_BWD,
        groups,
    )

    dQ_ref, dK_ref, dV_ref, _ = ref_bwd_varlen(
        Q,
        K,
        V,
        sinks,
        dO,
        cu_seqlens_q_t,
        cu_seqlens_k_t,
        max_seq_len,
        window_size,
        groups,
    )
    dq_pass, dq_ratio, dq_max = check_precision(dQ_fp16[..., :D].cpu(), dQ_ref.cpu(), DTYPE_FP16)
    dk_pass, dk_ratio, dk_max = check_precision(dK[..., :D].half().cpu(), dK_ref.cpu(), DTYPE_FP16)
    dv_pass, dv_ratio, dv_max = check_precision(dV.half().cpu(), dV_ref.cpu(), DTYPE_FP16)
    print(
        f"Backward precision: "
        f"dQ(ratio={dq_ratio:.4f}, max_abs={dq_max:.3e}) "
        f"dK(ratio={dk_ratio:.4f}, max_abs={dk_max:.3e}) "
        f"dV(ratio={dv_ratio:.4f}, max_abs={dv_max:.3e})"
    )

    all_ok = fwd_pass and dq_pass and dk_pass and dv_pass
    if all_ok:
        print("\nTest Passed!")
    else:
        print("\nTest FAILED (precision gate not met)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
