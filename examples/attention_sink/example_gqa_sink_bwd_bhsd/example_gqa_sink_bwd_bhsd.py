"""
GQA + Attention Sink Flash Attention (BHSD) for Ascend NPU — Expert Mode.

Layout: BHSD (Batch, Heads, SeqLen, Dim). Supports GQA (grouped-query attention),
an attention sink token, and an optional sliding window mask.

Based on:
  - GPU source: tilelang/examples/attention_sink/example_gqa_sink_bwd_bhsd.py
  - NPU reference: tilelang-ascend/examples_experiment/example_gqa_bwd/example_gqa_bwd.py

5 kernels:
  1. flashattn_fwd:            Forward (online softmax + sink + window) -> O, lse
  2. flashattn_bwd_preprocess: Delta = sum(O * dO, dim=-1)
  3. flashattn_bwd:            Backward main (5-GEMM + softmax recompute) -> dQ/dK/dV
  4. flashattn_bwd_postprocess: dQ fp32 -> fp16
  5. flashattn_bwd_dsink:      dSink = -exp(sink - lse) * Delta

Expert mode: explicit L1/UB/L0C + T.Scope("C"/"V") + manual sync + workspace.
"""

import tilelang
import torch
from tilelang import DataType, language as T

# ============================================================================
# Common pass_configs for Expert mode (all auto passes OFF)
# ============================================================================

_expert_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

# combineCV mode for pure-Vector kernels (preprocess/postprocess/dsink):
# no Cube/GEMM, so AUTO_CV_COMBINE/SYNC not needed. Enable auto sync +
# memory planning to drop manual T.Scope/barrier_all/annotate_address.
_vector_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# ============================================================================
# Kernel 1: Forward (online softmax + attention sink + sliding window)
# ============================================================================


@tilelang.jit(out_idx=[3, 4], workspace_idx=[6, 7, 8], pass_configs=_expert_pass_configs)
def flashattn_fwd(batch, heads, seq_len, dim, groups, window_size, block_M=64, block_N=64):
    """Forward: produces O [B,H,N,dim] fp16 and lse [B,H,N] fp32.

    Args:
        window_size: None for causal-only, or int (must be divisible by block_N).
    """
    assert seq_len % block_M == 0, f"seq_len ({seq_len}) must be divisible by block_M ({block_M})"
    assert seq_len % block_N == 0, f"seq_len ({seq_len}) must be divisible by block_N ({block_N})"
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
        assert window_size % block_N == 0, "window_size must be divisible by block_N"

    # Effective window for dynamic loop bounds: None → very large so loop_st = 0
    # (all KV blocks from 0 are valid for causal-only). Same trick as backward
    # kernel: for window=128 / N=4096 / block_N=64, reduces iters 64 → ~3.
    window_eff = window_size if window_size is not None else seq_len * 2

    # ---- Address layout precomputation ----
    # UB/L1 use time-share layout (non-compact: buffers with non-overlapping
    # lifetimes share address space). Offsets are running-accumulated; values
    # are identical to the original explicit sum expressions.
    hm = block_M // 2  # V scope processes half the rows
    b16 = DataType(dtype).bits // 8  # 2 bytes (fp16)
    b32 = DataType(accum_dtype).bits // 8  # 4 bytes (fp32)

    # L1 running offset
    l1_off = 0
    l1_off += block_M * dim * b16  # q_l1 [block_M, dim] fp16
    l1_q, l1_k_accs = 0, l1_off  # k_l1 & acc_s_l1 time-share @ l1_off
    l1_off += block_M * block_N * b16  # acc_s_l1 gap (actual: block_N*dim*b16)
    l1_v = l1_off  # v_l1 [block_N, dim] fp16

    # UB running offset
    u = 0
    u += hm * dim * b32  # acc_o [hm, dim] fp32
    u_sumexp = u
    u += hm * b32  # sumexp [hm] fp32
    u_m_i = u
    u += hm * b32  # m_i [hm] fp32
    u_acc_s_ub = u
    u += hm * block_N * b32  # acc_s_ub [hm, block_N] fp32
    u_m_i_prev = u
    u += hm * b32  # m_i_prev [hm] fp32
    u_acc_s_ub_ = u
    u += hm * b32  # acc_s_ub_ [hm, block_N] fp32 (time-share)
    u_sumexp_i = u
    u += hm * block_N * b32  # sumexp_i [hm] fp32 (gap)
    u_acc_s_half = u
    u += hm * block_N * b16  # acc_s_half [hm, block_N] fp16
    u_acc_o_ub = u
    u += hm * dim * b32  # acc_o_ub [hm, dim] fp32
    u_acc_o_half = u
    u += hm * dim * b16  # acc_o_half [hm, dim] fp16
    u_col_pos = u
    u += block_N * b32  # col_pos [block_N] fp32
    u_cmp = u
    u += block_N * b32  # cmp_mask [block_N] fp32
    u_win = u
    u += block_N * b32  # win_mask [block_N] fp32
    u_comb = u
    u += block_N * b32  # combined_mask [block_N] fp32
    u_sink = u
    u += hm * b32  # sink_ub [hm] fp32
    u_sink_exp = u
    u += hm * b32  # sink_exp_ub [hm] fp32
    u_sink_scalar = u  # sink_scalar [1] fp16 (last)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(o_shape, dtype),
        lse: T.Tensor(lse_shape, accum_dtype),
        Sinks: T.Tensor([heads], dtype),
        workspace_1: T.Tensor([block_num, block_M, block_N], accum_dtype),
        workspace_2: T.Tensor([block_num, block_M, block_N], dtype),
        workspace_3: T.Tensor([block_num, block_M, dim], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch
            kv_by = by // groups

            # ---- L1 buffers ----
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)
            acc_s_l1 = T.alloc_L1([block_M, block_N], dtype)
            acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_M, dim], accum_dtype)

            # ---- UB buffers ----
            acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)
            sumexp = T.alloc_ub([block_M // 2], accum_dtype)
            m_i = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            m_i_prev = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_ub_ = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            sumexp_i_ub = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)
            acc_o_ub = T.alloc_ub([block_M // 2, dim], accum_dtype)
            acc_o_half = T.alloc_ub([block_M // 2, dim], dtype)
            col_pos = T.alloc_ub([block_N], accum_dtype)
            cmp_mask = T.alloc_ub([block_N], accum_dtype)
            win_mask = T.alloc_ub([block_N], accum_dtype)
            combined_mask = T.alloc_ub([block_N], accum_dtype)
            # Attention sink buffers
            sink_ub = T.alloc_ub([block_M // 2], accum_dtype)
            sink_exp_ub = T.alloc_ub([block_M // 2], accum_dtype)
            sink_scalar = T.alloc_ub([1], dtype)

            T.annotate_address(
                {
                    # L1: q_l1 @ 0; k_l1 & acc_s_l1 time-share; v_l1 after gap
                    q_l1: l1_q,
                    k_l1: l1_k_accs,
                    acc_s_l1: l1_k_accs,
                    v_l1: l1_v,
                    # L0C (acc_s_l0c and acc_o_l0c share addr 0 — time-share)
                    acc_s_l0c: 0,
                    acc_o_l0c: 0,
                    # UB (time-share layout, offsets precomputed above)
                    acc_o: 0,
                    sumexp: u_sumexp,
                    m_i: u_m_i,
                    acc_s_ub: u_acc_s_ub,
                    m_i_prev: u_m_i_prev,
                    acc_s_ub_: u_acc_s_ub_,
                    sumexp_i_ub: u_sumexp_i,
                    acc_s_half: u_acc_s_half,
                    acc_o_ub: u_acc_o_ub,
                    acc_o_half: u_acc_o_half,
                    col_pos: u_col_pos,
                    cmp_mask: u_cmp,
                    win_mask: u_win,
                    combined_mask: u_comb,
                    sink_ub: u_sink,
                    sink_exp_ub: u_sink_exp,
                    sink_scalar: u_sink_scalar,
                }
            )

            # ---- Dynamic loop bounds: skip fully-invalid KV blocks ----
            # For Q block bx (Q rows [bx*block_M, (bx+1)*block_M-1]):
            #   causal: valid k_idx <= (bx+1)*block_M - 1  → last KV block = bx
            #   window: valid k_idx > bx*block_M - window_size  → skip early blocks
            # window_eff = window_size (or seq_len*2 for causal-only → loop_st=0)
            # C scope and V scope MUST use identical bounds (they are paired
            # via set_cross_flag/wait_cross_flag per iteration).
            loop_st = T.max(0, (bx * block_M - window_eff) // block_N)
            loop_ed = T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N))

            with T.Scope("C"):
                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_l1)
                T.barrier_all()
                for k in T.serial(loop_st, loop_ed):
                    T.copy(K[bz, kv_by, k * block_N : (k + 1) * block_N, :], k_l1)
                    T.barrier_all()
                    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                    T.barrier_all()
                    T.copy(acc_s_l0c, workspace_1[cid, :, :])
                    T.barrier_all()
                    T.set_cross_flag("FIX", 0)
                    T.wait_cross_flag(1)
                    T.barrier_all()
                    T.copy(workspace_2[cid, :, :], acc_s_l1)
                    T.copy(V[bz, kv_by, k * block_N : (k + 1) * block_N, :], v_l1)
                    T.barrier_all()
                    T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                    T.barrier_all()
                    T.copy(acc_o_l0c, workspace_3[cid, :, :])
                    T.barrier_all()
                    T.set_cross_flag("FIX", 2)
                    T.wait_cross_flag(3)

            with T.Scope("V"):
                T.tile.fill(acc_o, 0.0)
                T.tile.fill(sumexp, 0.0)
                T.tile.fill(m_i, -(2**30))
                T.barrier_all()

                # Load attention sink value and broadcast
                T.copy(Sinks[by : by + 1], sink_scalar)
                T.barrier_all()
                T.tile.fill(sink_ub, sink_scalar[0])
                T.barrier_all()

                for _k in T.serial(loop_st, loop_ed):
                    T.tile.fill(acc_s_ub, 0.0)
                    T.copy(m_i, m_i_prev)
                    T.barrier_all()
                    T.wait_cross_flag(0)
                    T.copy(workspace_1[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], acc_s_ub_)
                    T.barrier_all()
                    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)

                    # Causal + window mask
                    T.tile.arith_progression(col_pos, _k * block_N, 1, block_N)
                    for h_i in range(block_M // 2):
                        row_pos_val = (bx * block_M + vid * block_M // 2 + h_i) * 1.0
                        T.tile.compare(cmp_mask, col_pos, row_pos_val, "LE")
                        if window_size is not None:
                            T.tile.compare(win_mask, col_pos, row_pos_val - window_size, "GT")
                            T.tile.bitwise_and(combined_mask, cmp_mask, win_mask)
                            T.tile.select(
                                acc_s_ub[h_i, :], combined_mask, acc_s_ub[h_i, :], -T.infinity(accum_dtype), "VSEL_TENSOR_SCALAR_MODE"
                            )
                        else:
                            T.tile.select(acc_s_ub[h_i, :], cmp_mask, acc_s_ub[h_i, :], -T.infinity(accum_dtype), "VSEL_TENSOR_SCALAR_MODE")

                    # Online softmax: max -> rescale -> exp -> sum
                    T.reduce_max(acc_s_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    T.tile.sub(m_i_prev, m_i_prev, m_i)
                    T.tile.exp(m_i_prev, m_i_prev)
                    for h_i in range(block_M // 2):
                        T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
                    T.tile.exp(acc_s_ub, acc_s_ub)
                    T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                    T.tile.mul(sumexp, sumexp, m_i_prev)
                    T.tile.add(sumexp, sumexp, sumexp_i_ub)
                    for h_i in range(block_M // 2):
                        T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])

                    # Write P to workspace for C scope GEMM2
                    T.copy(acc_s_ub, acc_s_half)
                    T.barrier_all()
                    T.copy(acc_s_half, workspace_2[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :])
                    T.barrier_all()
                    T.set_cross_flag("MTE3", 1)
                    T.wait_cross_flag(2)
                    T.barrier_all()
                    T.copy(workspace_3[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], acc_o_ub)
                    T.barrier_all()
                    T.tile.add(acc_o, acc_o, acc_o_ub)
                    T.barrier_all()
                    T.set_cross_flag("V", 3)
                    T.barrier_all()

                # Attention Sink: sumexp += exp(sink - m_i)
                T.tile.sub(sink_exp_ub, sink_ub, m_i)
                T.tile.exp(sink_exp_ub, sink_exp_ub)
                T.tile.add(sumexp, sumexp, sink_exp_ub)

                # Normalize: O /= sumexp
                for h_i in range(block_M // 2):
                    T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

                # Output O (fp16)
                T.copy(acc_o, acc_o_half)
                T.barrier_all()
                T.copy(acc_o_half, Output[bz, by, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + block_M // 2, :])

                # lse = ln(sumexp) + m_i
                T.barrier_all()
                T.tile.ln(sumexp, sumexp)
                T.tile.add(sumexp, sumexp, m_i)
                T.barrier_all()
                T.copy(sumexp, lse[bz, by, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + block_M // 2])
                T.barrier_all()

    return main


# ============================================================================
# Kernel 2: Backward Preprocess — Delta = sum(O * dO, dim=-1)
# ============================================================================


@tilelang.jit(out_idx=[2], pass_configs=_vector_pass_configs)
def flashattn_bwd_preprocess(batch, heads, seq_len, dim, blk=32):
    assert seq_len % blk == 0, f"seq_len ({seq_len}) must be divisible by blk ({blk})"
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, heads, seq_len, dim]
    block_num = heads * (seq_len // blk) * batch

    @T.prim_func
    def main(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
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

            T.tile.fill(sum_ub, 0.0)
            T.copy(O[bz, bx, by * blk + vid * blk // 2 : by * blk + vid * blk // 2 + blk // 2, :], o_ub)
            T.copy(dO[bz, bx, by * blk + vid * blk // 2 : by * blk + vid * blk // 2 + blk // 2, :], do_ub)
            T.copy(o_ub, prod_ub)
            T.copy(do_ub, do_fp32)
            T.tile.mul(prod_ub, prod_ub, do_fp32)
            T.tile.add(sum_ub, sum_ub, prod_ub)
            T.reduce_sum(sum_ub, delta_ub, dim=-1)
            T.copy(delta_ub, Delta[bz, bx, by * blk + vid * blk // 2 : by * blk + vid * blk // 2 + blk // 2])

    return main


# ============================================================================
# Kernel 3: Backward Main — 5-GEMM + causal + optional window mask
# ============================================================================
# Originally in gqa_sink_bwd_kernel.py (separated to avoid TVM parser overflow).
# Merged here to follow the single-file convention (see examples_experiment/example_gqa_bwd).
#
# Grid over Q blocks (block_M), inner loop over KV blocks (block_N).
# dK/dV use T.tile.atomic_add to fp32 output tensors.
# dQ accumulated in L0C across KV iterations.
#
# 5 GEMMs per inner iteration:
#   GEMM1: S = Q @ K^T           -> [M, N]       (uses dim_qk_padded)
#   GEMM2: dV_partial = P^T @ dO  -> [N, dim_v]   (uses dim_v)
#   GEMM3: dP = dO @ V^T          -> [M, N]       (uses dim_v)
#   GEMM4: dK_partial = dS^T @ Q  -> [N, dim_qk_padded]
#   GEMM5: dQ_partial = dS @ K    -> [M, dim_qk_padded]
#
# Causal mask is always applied. When window_size is not None, a sliding window
# mask is also applied (causal AND window). Both masks use T.tile.compare which
# requires block_N >= 64 (256-byte alignment for fp32).


@tilelang.jit(pass_configs=_expert_pass_configs)
def flashattn_bwd(batch, heads, seq_len, dim_qk, dim_v, window_size, block_M, block_N, groups=1):
    """Backward main kernel with 5-GEMM + causal + optional window mask.

    Args:
        window_size: None for causal-only, or int (must be divisible by block_N)
                     for causal + sliding window.
    """
    assert seq_len % block_M == 0, f"seq_len ({seq_len}) must be divisible by block_M ({block_M})"
    assert seq_len % block_N == 0, f"seq_len ({seq_len}) must be divisible by block_N ({block_N})"
    sm_scale = (1.0 / dim_qk) ** 0.5
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"
    # gemm_v0 requires N <= 128 or N % 128 == 0 for non-transpose-B.
    dim_qk_padded = ((dim_qk + 127) // 128) * 128

    q_shape = [batch, heads, seq_len, dim_qk_padded]
    k_shape = [batch, head_kv, seq_len, dim_qk_padded]
    v_shape = [batch, head_kv, seq_len, dim_v]
    do_shape = [batch, heads, seq_len, dim_v]
    dq_shape_padded = [batch, heads, seq_len, dim_qk_padded]
    dk_shape_padded = [batch, head_kv, seq_len, dim_qk_padded]
    bwd_block_num = (seq_len // block_M) * heads * batch

    if window_size is not None:
        assert window_size % block_N == 0, "window_size must be divisible by block_N"

    # Effective window for dynamic loop bounds: None → very large so loop_st = 0
    # (all KV blocks from 0 are valid for causal-only)
    window_eff = window_size if window_size is not None else seq_len * 2

    # ---- Address layout precomputation (time-share, non-compact) ----
    hm = block_M // 2  # V scope processes half the rows
    max_dim = max(dim_qk_padded, dim_v)
    b16 = DataType(dtype).bits // 8  # 2 bytes (fp16)
    b32 = DataType(accum_dtype).bits // 8  # 4 bytes (fp32)

    # L1 running offset (compact)
    l1 = 0
    l1_q = l1
    l1 += block_M * dim_qk_padded * b16  # q_l1
    l1_do = l1
    l1 += block_M * dim_v * b16  # do_l1
    l1_k = l1
    l1 += block_N * dim_qk_padded * b16  # k_l1
    l1_v = l1
    l1 += block_N * dim_v * b16  # v_l1
    l1_mn = l1  # mn_l1

    # L0C: l0c_mn @ 0; l0c_nd_v/l0c_nd_qk time-share; l0c_dq after both
    c_mn = 0
    c_nd = block_M * block_N * b32
    c_dq = (block_M * block_N + block_N * max_dim) * b32

    # UB running offset (compact: each buffer occupies its actual size)
    # Layout: work_ub | dp_ub | p_half | lse_ub | delta_ub | dv_tmp
    #       | col_pos | cmp_mask | win_mask | combined_mask | lse_2d | delta_2d
    u = 0
    u_work = u
    u += hm * block_N * b32  # work_ub [hm, block_N] fp32
    u_dp = u
    u += hm * block_N * b32  # dp_ub [hm, block_N] fp32
    u_p = u
    u += hm * block_N * b16  # p_half [hm, block_N] fp16
    u_lse = u
    u += hm * b32  # lse_ub [hm] fp32
    u_delta = u
    u += hm * b32  # delta_ub [hm] fp32
    u_dv = u
    u += (block_N // 2) * max_dim * b32  # dv_tmp [block_N//2, max_dim] fp32
    u_col = u
    u += block_N * b32  # col_pos [block_N] fp32
    u_cmp = u
    u += block_N * b32  # cmp_mask [block_N] fp32
    u_win = u
    u += block_N * b32  # win_mask [block_N] fp32
    u_comb = u
    u += block_N * b32  # combined_mask [block_N] fp32
    u_lse2d = u
    u += hm * block_N * b32  # lse_2d [hm, block_N] fp32
    u_delta2d = u  # delta_2d [hm, block_N] fp32 (last)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        dO: T.Tensor(do_shape, dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        dQ: T.Tensor(dq_shape_padded, accum_dtype),
        dK: T.Tensor(dk_shape_padded, accum_dtype),
        dV: T.Tensor(v_shape, accum_dtype),
        ws_s_dp: T.Tensor([bwd_block_num, block_M, block_N], accum_dtype),
        ws_p_ds: T.Tensor([bwd_block_num, block_M, block_N], dtype),
        ws_dv_dk: T.Tensor([bwd_block_num, block_N, max(dim_qk_padded, dim_v)], accum_dtype),
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch
            kv_by = by // groups

            # ---- Dynamic loop bounds: skip fully-invalid KV blocks ----
            # For Q block bx (Q rows [bx*block_M, (bx+1)*block_M-1]):
            #   causal: valid k_idx <= (bx+1)*block_M - 1  → last KV block = bx
            #   window: valid k_idx > bx*block_M - window_size  → skip early blocks
            # window_eff = window_size (or seq_len*2 for causal-only → loop_st=0)
            # This reduces iterations from seq_len/block_N to ~3 for window=128.
            loop_st = T.max(0, (bx * block_M - window_eff) // block_N)
            loop_ed = T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N))

            # ---- L1 buffers ----
            q_l1 = T.alloc_L1([block_M, dim_qk_padded], dtype)
            do_l1 = T.alloc_L1([block_M, dim_v], dtype)
            k_l1 = T.alloc_L1([block_N, dim_qk_padded], dtype)
            v_l1 = T.alloc_L1([block_N, dim_v], dtype)
            mn_l1 = T.alloc_L1([block_M, block_N], dtype)

            # ---- L0C buffers ----
            l0c_mn = T.alloc_L0C([block_M, block_N], accum_dtype)
            l0c_nd_v = T.alloc_L0C([block_N, dim_v], accum_dtype)
            l0c_nd_qk = T.alloc_L0C([block_N, dim_qk_padded], accum_dtype)
            l0c_dq = T.alloc_L0C([block_M, dim_qk_padded], accum_dtype)

            # ---- UB buffers ----
            work_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            dp_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            p_half = T.alloc_ub([block_M // 2, block_N], dtype)
            lse_ub = T.alloc_ub([block_M // 2], accum_dtype)
            delta_ub = T.alloc_ub([block_M // 2], accum_dtype)
            dv_tmp = T.alloc_ub([block_N // 2, max(dim_qk_padded, dim_v)], accum_dtype)
            col_pos = T.alloc_ub([block_N], accum_dtype)
            cmp_mask = T.alloc_ub([block_N], accum_dtype)
            # Always allocate (avoids conditional T.annotate_address which
            # TVM parser doesn't support)
            win_mask = T.alloc_ub([block_N], accum_dtype)
            combined_mask = T.alloc_ub([block_N], accum_dtype)
            # P2: 2D broadcast buffers for lse/delta sub (replace per-row
            # for loop with whole-tile sub). axis=1 broadcast only (axis=0
            # col-broadcast TCOLEXPAND is unreliable in codegen).
            lse_2d = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            delta_2d = T.alloc_ub([block_M // 2, block_N], accum_dtype)

            T.annotate_address(
                {
                    # L1: q_l1, do_l1 persistent; k_l1, v_l1, mn_l1 separate
                    q_l1: l1_q,
                    do_l1: l1_do,
                    k_l1: l1_k,
                    v_l1: l1_v,
                    mn_l1: l1_mn,
                    # L0C: l0c_mn @ 0; l0c_nd_v/l0c_nd_qk share addr; l0c_dq after both
                    l0c_mn: c_mn,
                    l0c_nd_v: c_nd,
                    l0c_nd_qk: c_nd,
                    l0c_dq: c_dq,
                    # UB (time-share layout, offsets precomputed above)
                    work_ub: u_work,
                    dp_ub: u_dp,
                    p_half: u_p,
                    lse_ub: u_lse,
                    delta_ub: u_delta,
                    dv_tmp: u_dv,
                    col_pos: u_col,
                    cmp_mask: u_cmp,
                    win_mask: u_win,
                    combined_mask: u_comb,
                    # P2: 2D broadcast buffers for lse/delta sub
                    lse_2d: u_lse2d,
                    delta_2d: u_delta2d,
                }
            )

            with T.Scope("C"):
                # Load Q and dO (persistent across KV iterations)
                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_l1)
                T.copy(dO[bz, by, bx * block_M : (bx + 1) * block_M, :], do_l1)
                T.barrier_all()

                for k in T.serial(loop_st, loop_ed):
                    # === GEMM1: S = Q @ K^T -> l0c_mn [M, N] ===
                    T.copy(K[bz, kv_by, k * block_N : (k + 1) * block_N, :], k_l1)
                    T.barrier_all()
                    T.gemm_v0(q_l1, k_l1, l0c_mn, transpose_B=True, init=True)
                    T.barrier_all()
                    T.copy(l0c_mn, ws_s_dp[cid, :, :])
                    T.barrier_all()
                    T.set_cross_flag("FIX", 0)  # signal V: S ready

                    # Wait for P from V scope
                    T.wait_cross_flag(1)
                    T.barrier_all()
                    T.copy(ws_p_ds[cid, :, :], mn_l1)

                    # === GEMM2: dV_partial = P^T @ dO -> l0c_nd_v [N, dim_v] ===
                    T.barrier_all()
                    T.gemm_v0(mn_l1, do_l1, l0c_nd_v, transpose_A=True, init=True)
                    T.barrier_all()
                    T.copy(l0c_nd_v, ws_dv_dk[cid, :, :])
                    T.barrier_all()

                    # === GEMM3: dP = dO @ V^T -> l0c_mn [M, N] ===
                    T.copy(V[bz, kv_by, k * block_N : (k + 1) * block_N, :], v_l1)
                    T.barrier_all()
                    T.gemm_v0(do_l1, v_l1, l0c_mn, transpose_B=True, init=True)
                    T.barrier_all()
                    T.copy(l0c_mn, ws_s_dp[cid, :, :])
                    T.barrier_all()
                    T.set_cross_flag("FIX", 2)  # signal V: dV + dP ready

                    # Wait for dS from V scope
                    T.wait_cross_flag(3)
                    T.barrier_all()
                    T.copy(ws_p_ds[cid, :, :], mn_l1)

                    # === GEMM4: dK_partial = dS^T @ Q -> l0c_nd_qk [N, dim_qk] ===
                    T.barrier_all()
                    T.gemm_v0(mn_l1, q_l1, l0c_nd_qk, transpose_A=True, init=True)
                    T.barrier_all()
                    T.copy(l0c_nd_qk, ws_dv_dk[cid, :, :])
                    T.barrier_all()
                    T.set_cross_flag("FIX", 4)  # signal V: dK ready

                    # === GEMM5: dQ_partial = dS @ K -> l0c_dq [M, dim_qk] ===
                    T.barrier_all()
                    T.gemm_v0(mn_l1, k_l1, l0c_dq, init=(k == loop_st))
                    T.barrier_all()

                # After KV loop: write accumulated dQ to global memory
                T.barrier_all()
                T.copy(l0c_dq, dQ[bz, by, bx * block_M : (bx + 1) * block_M, :])
                T.barrier_all()

            with T.Scope("V"):
                for _k in T.serial(loop_st, loop_ed):
                    # ---- Step 1: Receive S, compute P = exp(S * scale - lse) ----
                    T.wait_cross_flag(0)
                    T.barrier_all()
                    T.copy(ws_s_dp[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], work_ub)
                    T.barrier_all()

                    # Scale: S * sm_scale
                    T.tile.mul(work_ub, work_ub, sm_scale)

                    # Subtract LSE and exp: P = exp(S * scale - lse)
                    T.copy(lse[bz, by, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + block_M // 2], lse_ub)
                    T.barrier_all()
                    # P2: broadcast lse to 2D then whole-tile sub (replaces
                    # per-row for loop). axis=1 broadcast is supported.
                    T.tile.broadcast(lse_2d, lse_ub, axis=1)
                    T.tile.sub(work_ub, work_ub, lse_2d)
                    T.tile.exp(work_ub, work_ub)

                    # Apply causal + window mask: zero out P where invalid
                    # (block_N >= 64 required for T.tile.compare)
                    T.tile.arith_progression(col_pos, _k * block_N, 1, block_N)
                    for h_i in range(block_M // 2):
                        row_pos_val = (bx * block_M + vid * block_M // 2 + h_i) * 1.0
                        # causal: col_pos <= row_pos
                        T.tile.compare(cmp_mask, col_pos, row_pos_val, "LE")
                        if window_size is not None:
                            # window: col_pos > row_pos - window_size
                            T.tile.compare(win_mask, col_pos, row_pos_val - window_size, "GT")
                            T.tile.bitwise_and(combined_mask, cmp_mask, win_mask)
                            T.tile.select(work_ub[h_i, :], combined_mask, work_ub[h_i, :], 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        else:
                            T.tile.select(work_ub[h_i, :], cmp_mask, work_ub[h_i, :], 0.0, "VSEL_TENSOR_SCALAR_MODE")

                    # Cast P to fp16 and write to workspace
                    T.copy(work_ub, p_half)
                    T.barrier_all()
                    T.copy(p_half, ws_p_ds[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :])
                    T.barrier_all()
                    T.set_cross_flag("MTE3", 1)  # signal C: P ready

                    # ---- Step 2: Receive dV + dP, compute dS ----
                    T.wait_cross_flag(2)
                    T.barrier_all()

                    # atomic_add dV to global memory
                    T.copy(ws_dv_dk[cid, vid * block_N // 2 : vid * block_N // 2 + block_N // 2, :], dv_tmp)
                    T.barrier_all()
                    T.tile.atomic_add(
                        dV[bz, kv_by, _k * block_N + vid * block_N // 2 : _k * block_N + vid * block_N // 2 + block_N // 2, :], dv_tmp
                    )

                    # Load dP from workspace (fp32)
                    T.copy(ws_s_dp[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], dp_ub)
                    T.barrier_all()

                    # P is still in p_half from Step 1
                    T.copy(p_half, work_ub)  # fp16 -> fp32

                    # Load Delta
                    T.copy(Delta[bz, by, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + block_M // 2], delta_ub)
                    T.barrier_all()

                    # Compute dS = P * (dP - Delta) * sm_scale
                    # P2: broadcast delta to 2D then whole-tile sub (replaces
                    # per-row for loop). axis=1 broadcast is supported.
                    T.tile.broadcast(delta_2d, delta_ub, axis=1)
                    T.tile.sub(dp_ub, dp_ub, delta_2d)
                    T.tile.mul(work_ub, work_ub, dp_ub)
                    T.tile.mul(work_ub, work_ub, sm_scale)

                    # Cast dS to fp16 and write to workspace (overwrite P)
                    T.copy(work_ub, p_half)
                    T.barrier_all()
                    T.copy(p_half, ws_p_ds[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :])
                    T.barrier_all()
                    T.set_cross_flag("V", 3)  # signal C: dS ready

                    # ---- Step 3: Receive dK and atomic_add ----
                    T.wait_cross_flag(4)
                    T.barrier_all()
                    T.copy(ws_dv_dk[cid, vid * block_N // 2 : vid * block_N // 2 + block_N // 2, :], dv_tmp)  # reuse dv_tmp for dK
                    T.barrier_all()
                    T.tile.atomic_add(
                        dK[bz, kv_by, _k * block_N + vid * block_N // 2 : _k * block_N + vid * block_N // 2 + block_N // 2, :], dv_tmp
                    )
                    T.barrier_all()

    return main


# ============================================================================
# Kernel 4: Backward Postprocess — dQ fp32 -> fp16
# ============================================================================


@tilelang.jit(out_idx=[1], pass_configs=_vector_pass_configs)
def flashattn_bwd_postprocess(batch, heads, seq_len, dim_qk, blk=64):
    assert seq_len % blk == 0, f"seq_len ({seq_len}) must be divisible by blk ({blk})"
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, heads, seq_len, dim_qk]
    block_num = (seq_len // blk) * heads * batch

    @T.prim_func
    def main(
        dQ: T.Tensor(shape, accum_dtype),
        dQ_out: T.Tensor(shape, dtype),
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
# Kernel 5: Dsink — dSink = -exp(sink - lse) * Delta, sum over (B, N)
# ============================================================================


@tilelang.jit(out_idx=-1, pass_configs=_vector_pass_configs)
def flashattn_bwd_dsink(batch, heads, seq_len, block=128):
    assert seq_len % block == 0, f"seq_len ({seq_len}) must be divisible by block ({block})"
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, heads, seq_len]
    block_num = heads * (seq_len // block) * batch

    @T.prim_func
    def main(
        Sinks: T.Tensor([heads], dtype),
        Delta: T.Tensor(shape, accum_dtype),
        lse: T.Tensor(shape, accum_dtype),
        dsinks: T.Tensor(shape, dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % heads
            by = cid // heads % (seq_len // block)
            bz = cid // heads // (seq_len // block) % batch

            lse_ub = T.alloc_ub([block // 2], accum_dtype)
            delta_ub = T.alloc_ub([block // 2], accum_dtype)
            sink_exp_ub = T.alloc_ub([block // 2], accum_dtype)
            dsink_ub = T.alloc_ub([block // 2], dtype)
            sink_val_ub = T.alloc_ub([block // 2], accum_dtype)
            sink_scalar = T.alloc_ub([1], dtype)

            # Load sink value and broadcast
            T.copy(Sinks[bx : bx + 1], sink_scalar)
            T.tile.fill(sink_val_ub, sink_scalar[0])

            # Load lse and Delta
            T.copy(lse[bz, bx, by * block + vid * block // 2 : by * block + vid * block // 2 + block // 2], lse_ub)
            T.copy(Delta[bz, bx, by * block + vid * block // 2 : by * block + vid * block // 2 + block // 2], delta_ub)

            # dsink = -exp(sink - lse) * delta
            T.tile.sub(sink_exp_ub, sink_val_ub, lse_ub)
            T.tile.exp(sink_exp_ub, sink_exp_ub)
            T.tile.mul(sink_exp_ub, sink_exp_ub, delta_ub)
            T.tile.mul(sink_exp_ub, sink_exp_ub, -1.0)

            # Cast to fp16 and write output
            T.copy(sink_exp_ub, dsink_ub)
            T.copy(dsink_ub, dsinks[bz, bx, by * block + vid * block // 2 : by * block + vid * block // 2 + block // 2])

    return main


# ============================================================================
# Golden Reference (PyTorch)
# ============================================================================


def ref_fwd(Q, K, V, Sinks, window_size=None, groups=1):
    """Forward golden: GQA + Attention Sink + optional sliding window.

    Q: [B, H, N, D] fp16, K/V: [B, H_kv, N, D] fp16, Sinks: [H] fp16.
    Returns: O [B, H, N, D] fp16.
    """
    B, H, N, D = Q.shape
    sm_scale = 1.0 / D**0.5

    # GQA: repeat K, V for each group
    K_rep = K.float().repeat_interleave(groups, dim=1)  # [B, H, N, D]
    V_rep = V.float().repeat_interleave(groups, dim=1)

    # S = Q @ K^T * scale
    S = torch.matmul(Q.float(), K_rep.transpose(-2, -1)) * sm_scale  # [B, H, N, N]

    # Causal + window mask
    pos_q = torch.arange(N, device=Q.device).float()
    pos_k = torch.arange(N, device=Q.device).float()
    causal_mask = pos_k[None, :] <= pos_q[:, None]  # k <= q (causal)
    if window_size is not None:
        window_mask = pos_k[None, :] > (pos_q[:, None] - window_size)
        mask = causal_mask & window_mask
    else:
        mask = causal_mask
    S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Softmax with attention sink
    m = S.max(dim=-1, keepdim=True).values  # [B, H, N, 1]
    sinks_b = Sinks.view(1, H, 1, 1).float()  # [1, H, 1, 1]
    m_with_sink = torch.maximum(sinks_b, m)  # numerical stability

    P = torch.exp(S - m_with_sink)  # [B, H, N, N]
    sinks_exp = torch.exp(sinks_b - m_with_sink)  # [B, H, N, 1]
    normalizer = P.sum(dim=-1, keepdim=True) + sinks_exp  # [B, H, N, 1]
    P = P / normalizer

    O = torch.matmul(P, V_rep)  # [B, H, N, D]
    return O.half()


def ref_bwd(Q, K, V, Sinks, dO, window_size=None, groups=1):
    """Backward golden via autograd. Returns dQ, dK, dV, dSinks (all fp16)."""
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

    return Q_f.grad.half(), K_f.grad.half(), V_f.grad.half(), Sinks_f.grad.half()


# ============================================================================
# Autograd Function (end-to-end wrapper: attention(q, k, v, sinks, ...))
# Matches GPU source _attention class and example_gqa_bwd convention.
# Users can call ``attention(q, k, v, sinks, window_size, groups)`` directly
# and get gradients via ``O.backward(dO)`` without manually orchestrating
# the 5 kernels.
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

        # Preprocess: Delta = sum(O * dO, dim=-1)
        prep_mod = flashattn_bwd_preprocess(B, H, N, D, blk=32)
        delta = prep_mod(o, do)

        # Backward main (pad Q/K to dim_qk_padded for gemm_v0 alignment)
        Q_pad = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float16, device=q.device)
        Q_pad[..., :D] = q
        K_pad = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float16, device=q.device)
        K_pad[..., :D] = k

        # dK/dV use atomic_add — must zero before call.
        dQ = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float32, device=q.device)
        dK = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float32, device=q.device)
        dV = torch.zeros(B, H_kv, N, D, dtype=torch.float32, device=q.device)

        bwd_block_num = H * (N // block_M) * B
        ws_s_dp = torch.empty(bwd_block_num, block_M, block_N, dtype=torch.float32, device=q.device)
        ws_p_ds = torch.empty(bwd_block_num, block_M, block_N, dtype=torch.float16, device=q.device)
        ws_dv_dk = torch.empty(bwd_block_num, block_N, max(dim_qk_padded, D), dtype=torch.float32, device=q.device)

        bwd_mod = flashattn_bwd(B, H, N, D, D, window_size, block_M, block_N, groups)
        bwd_mod(Q_pad, K_pad, v, do, lse, delta, dQ, dK, dV, ws_s_dp, ws_p_ds, ws_dv_dk)

        # Postprocess: dQ fp32 -> fp16, trim padded columns
        post_mod = flashattn_bwd_postprocess(B, H, N, dim_qk_padded, blk=64)
        dQ = post_mod(dQ)[..., :D]

        # dK/dV: trim padded columns, fp32 -> fp16
        dK = dK[..., :D].half()
        dV = dV.half()

        # dSinks: sum over (B, N) to get [H]
        dsink_mod = flashattn_bwd_dsink(B, H, N, block=64)
        dsinks = dsink_mod(sinks, delta, lse).sum(0).sum(1)

        return dQ, dK, dV, dsinks, None, None


attention = _attention.apply


# ============================================================================
# __main__: minimal L0 smoke test (CI bench_test.sh runs this directly).
# stdout must contain "Test Passed!" for CI to mark the script PASSED.
# Minimal L0 config: B=1, H=4, groups=2, N=128, D=64, causal-only (window=None).
# Uses small config for fast CI turnaround; golden config (H=64, N=4096, D=128,
# window=128) is benchmarked via ``python test_gqa_sink_bwd_bhsd.py --level bench``.
# ============================================================================


if __name__ == "__main__":
    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(42)

    # Minimal L0 config (causal-only GQA + attention sink)
    B, H, groups, N, D = 1, 4, 2, 128, 64
    H_kv = H // groups
    window_size = None
    # Forward is tighter (5e-3) to match test_gqa_sink_bwd_bhsd.py::_run_case;
    # backward uses 1e-2 (5-GEMM pipeline accumulates more error).
    fwd_atol = 5e-3
    bwd_atol = 1e-2

    Q = torch.randn(B, H, N, D, dtype=torch.float16, device="npu")
    K = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="npu")
    V = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="npu")
    sinks = torch.randn(H, dtype=torch.float16, device="npu")
    dO = torch.randn(B, H, N, D, dtype=torch.float16, device="npu")

    block_M, block_N = 64, 64
    dim_qk_padded = ((D + 127) // 128) * 128

    # --- Forward smoke test ---
    fwd_mod = flashattn_fwd(B, H, N, D, groups, window_size, block_M, block_N)
    O_npu, lse_npu = fwd_mod(Q, K, V, sinks)
    torch.npu.synchronize()

    O_ref = ref_fwd(Q, K, V, sinks, window_size, groups)
    fwd_max_diff = (O_npu.float() - O_ref.float()).abs().max().item()
    assert fwd_max_diff < fwd_atol, f"Forward precision check failed: max_diff={fwd_max_diff} >= atol={fwd_atol}"

    # --- Backward preprocess ---
    prep_mod = flashattn_bwd_preprocess(B, H, N, D, blk=32)
    Delta_npu = prep_mod(O_npu, dO)
    torch.npu.synchronize()

    # --- Backward main ---
    # Pad Q and K to dim_qk_padded for backward kernel
    Q_pad = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float16, device="npu")
    Q_pad[..., :D] = Q
    K_pad = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float16, device="npu")
    K_pad[..., :D] = K

    # dK/dV use atomic_add — must zero before each call.
    dQ = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float32, device="npu")
    dK = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float32, device="npu")
    dV = torch.zeros(B, H_kv, N, D, dtype=torch.float32, device="npu")

    bwd_block_num = H * (N // block_M) * B
    ws_s_dp = torch.empty(bwd_block_num, block_M, block_N, dtype=torch.float32, device="npu")
    ws_p_ds = torch.empty(bwd_block_num, block_M, block_N, dtype=torch.float16, device="npu")
    ws_dv_dk = torch.empty(bwd_block_num, block_N, max(dim_qk_padded, D), dtype=torch.float32, device="npu")

    bwd_mod = flashattn_bwd(B, H, N, D, D, window_size, block_M, block_N, groups)
    bwd_mod(Q_pad, K_pad, V, dO, lse_npu, Delta_npu, dQ, dK, dV, ws_s_dp, ws_p_ds, ws_dv_dk)
    torch.npu.synchronize()

    # --- Postprocess dQ (fp32 -> fp16) ---
    post_mod = flashattn_bwd_postprocess(B, H, N, dim_qk_padded, blk=64)
    dQ_fp16 = post_mod(dQ)
    torch.npu.synchronize()

    # --- Dsink ---
    dsink_mod = flashattn_bwd_dsink(B, H, N, block=64)
    dSinks_npu = dsink_mod(sinks, Delta_npu, lse_npu).sum(0).sum(1)
    torch.npu.synchronize()

    # --- Compare with golden backward ---
    dQ_ref, dK_ref, dV_ref, dSinks_ref = ref_bwd(Q, K, V, sinks, dO, window_size, groups)

    bwd_max_diff = max(
        (dQ_fp16[..., :D].float() - dQ_ref.float()).abs().max().item(),
        (dK[..., :D].half().float() - dK_ref.float()).abs().max().item(),
        (dV.half().float() - dV_ref.float()).abs().max().item(),
        (dSinks_npu.float() - dSinks_ref.float()).abs().max().item(),
    )
    assert bwd_max_diff < bwd_atol, f"Backward precision check failed: max_diff={bwd_max_diff} >= atol={bwd_atol}"

    print(f"max_diff: fwd={fwd_max_diff:.6e} bwd={bwd_max_diff:.6e}")
    print("Test Passed!")
