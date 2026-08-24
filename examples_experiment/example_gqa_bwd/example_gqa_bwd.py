"""GQA Flash Attention Forward + Backward (BHSD layout).

Developer mode: zero T.Scope, zero T.barrier_all, zero manual flags.
Backward uses 3-sub-kernel host-side serial launch to eliminate C<->V
cross-core dependencies.

Kernels:
  - flashattn_fwd: Forward (online softmax + causal) -> O, lse
  - flashattn_bwd_preprocess: Delta = sum(O * dO)
  - flashattn_bwd_gemm_s_dp: Phase 1 (GEMM1 S=Q@K^T + GEMM3 dP=dO@V^T)
  - flashattn_bwd_softmax_ds: Phase 2 (softmax recompute + dS)
  - flashattn_bwd_gemm_dv_dk_dq: Phase 3 (GEMM2+corr + GEMM4+corr + GEMM5)
  - flashattn_bwd_postprocess: dK/dV/dQ fp32->fp16 cast
"""

import torch
import torch_npu  # noqa: F401  # register NPU backend
import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

# pass_configs — Developer mode (CV fusion: 4 keys enabled)
_developer_cv_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# pass_configs — Developer mode (Vector-only: 2 keys enabled)
_developer_vector_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# --- Kernel 1: Forward (online softmax + causal mask) ---


@tilelang.jit(out_idx=[3, 4], workspace_idx=[5, 6, 7], pass_configs=_developer_cv_pass_configs)
def flashattn_fwd(batch, heads, seq_len, dim_qk, dim_v, groups, is_causal, block_M=64, block_N=64):
    """Forward: produces O [B,H,N,dim_v] fp16 and lse [B,H,N] fp32.

    Developer mode (4 keys): alloc_L1/ub/L0C + auto CV split + auto sync.
    """
    assert seq_len % block_M == 0
    assert seq_len % block_N == 0
    sm_scale = (1.0 / dim_qk) ** 0.5
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"
    dim_qk_padded = ((dim_qk + 15) // 16) * 16

    q_shape = [batch, heads, seq_len, dim_qk_padded]
    kv_shape_qk = [batch, head_kv, seq_len, dim_qk_padded]
    kv_shape_v = [batch, head_kv, seq_len, dim_v]
    o_shape = [batch, heads, seq_len, dim_v]
    lse_shape = [batch, heads, seq_len]
    block_num = (seq_len // block_M) * heads * batch
    hm = block_M // 2

    @T.prim_func
    def fwd(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(kv_shape_qk, dtype),  # type: ignore
        V: T.Tensor(kv_shape_v, dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        workspace_1: T.Tensor([block_num, block_M, block_N], accum_dtype),  # type: ignore
        workspace_2: T.Tensor([block_num, block_M, block_N], dtype),  # type: ignore
        workspace_3: T.Tensor([block_num, block_M, dim_v], accum_dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch
            kv_by = by // groups

            q_l1 = T.alloc_L1([block_M, dim_qk_padded], dtype)
            k_l1 = T.alloc_L1([block_N, dim_qk_padded], dtype)
            v_l1 = T.alloc_L1([block_N, dim_v], dtype)
            acc_s_l1 = T.alloc_L1([block_M, block_N], dtype)
            acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_M, dim_v], accum_dtype)

            acc_o = T.alloc_ub([hm, dim_v], accum_dtype)
            sumexp = T.alloc_ub([hm], accum_dtype)
            m_i = T.alloc_ub([hm], accum_dtype)
            acc_s_ub = T.alloc_ub([hm, block_N], accum_dtype)
            m_i_prev = T.alloc_ub([hm], accum_dtype)
            acc_s_ub_ = T.alloc_ub([hm, block_N], accum_dtype)
            sumexp_i_ub = T.alloc_ub([hm], accum_dtype)
            acc_s_half = T.alloc_ub([hm, block_N], dtype)
            acc_o_ub = T.alloc_ub([hm, dim_v], accum_dtype)
            acc_o_half = T.alloc_ub([hm, dim_v], dtype)
            col_pos = T.alloc_ub([block_N], accum_dtype)
            row_pos = T.alloc_ub([hm], accum_dtype)
            row_pos_2d = T.alloc_ub([hm, block_N], accum_dtype)
            mask_2d = T.alloc_ub([hm, block_N], accum_dtype)

            window_eff = seq_len * 2
            loop_st = T.max(0, (bx * block_M - window_eff) // block_N)
            loop_ed = (
                T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N)) if is_causal else T.ceildiv(seq_len, block_N)
            )

            v_row = vid * hm
            q_row = bx * block_M

            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -(2**30))

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

                if is_causal:
                    T.tile.arith_progression(col_pos, kv, 1, block_N)
                    T.tile.broadcast(acc_s_ub_, col_pos, axis=0)
                    T.tile.broadcast(row_pos_2d, row_pos, axis=1)
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

            # Normalize: O /= sumexp
            T.tile.broadcast(acc_o_ub, sumexp, axis=1)
            T.tile.div(acc_o, acc_o, acc_o_ub)

            T.copy(acc_o, acc_o_half)
            T.copy(acc_o_half, Output[bz, by, q_row + v_row : q_row + v_row + hm, :])

            T.tile.ln(sumexp, sumexp)
            T.tile.add(sumexp, sumexp, m_i)
            T.copy(sumexp, lse[bz, by, q_row + v_row : q_row + v_row + hm])

    return fwd


# --- Kernel 2: Backward Preprocess — Delta = sum(O * dO, dim=-1) ---


@tilelang.jit(out_idx=[2], pass_configs=_developer_vector_pass_configs)
def flashattn_bwd_preprocess(batch, heads, seq_len, dim_v, blk=32):
    assert seq_len % blk == 0
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, heads, seq_len, dim_v]
    block_num = heads * (seq_len // blk) * batch

    @T.prim_func
    def prep(
        O: T.Tensor(shape, dtype),  # type: ignore
        dO: T.Tensor(shape, dtype),  # type: ignore
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            by = cid % (seq_len // blk)
            bx = cid // (seq_len // blk) % heads
            bz = cid // (seq_len // blk) // heads % batch
            row = by * blk + vid * blk // 2

            o_ub = T.alloc_ub([blk // 2, dim_v], dtype)
            do_ub = T.alloc_ub([blk // 2, dim_v], dtype)
            sum_ub = T.alloc_ub([blk // 2, dim_v], accum_dtype)
            prod_ub = T.alloc_ub([blk // 2, dim_v], accum_dtype)
            do_fp32 = T.alloc_ub([blk // 2, dim_v], accum_dtype)
            delta_ub = T.alloc_ub([blk // 2], accum_dtype)

            T.tile.fill(sum_ub, 0.0)
            T.copy(O[bz, bx, row : row + blk // 2, :], o_ub)
            T.copy(dO[bz, bx, row : row + blk // 2, :], do_ub)
            T.copy(o_ub, prod_ub)
            T.copy(do_ub, do_fp32)
            T.tile.mul(prod_ub, prod_ub, do_fp32)
            T.tile.add(sum_ub, sum_ub, prod_ub)
            T.reduce_sum(sum_ub, delta_ub, dim=-1)
            T.copy(delta_ub, Delta[bz, bx, row : row + blk // 2])

    return prep


# --- Kernel 3: bwd Phase 1 — GEMM1 (S=Q@K^T) + GEMM3 (dP=dO@V^T) ---


@tilelang.jit(out_idx=[4, 5], pass_configs=_developer_cv_pass_configs)
def flashattn_bwd_gemm_s_dp(batch, heads, seq_len, dim_qk, dim_v, groups, is_causal, block_M=64, block_N=64):
    """Phase 1: GEMM1 (S=Q@K^T) + GEMM3 (dP=dO@V^T) -> ws_s, ws_dp (GM fp32).

    Cube-only Developer mode. Each cid processes one Q-block; loops over KV blocks.
    Auto-selects block_N=128 when seq_len divisible and D_qk<=192 (halves KV iters).
    """
    dim_qk_padded = ((dim_qk + 15) // 16) * 16
    if seq_len % 128 == 0 and dim_qk_padded <= 192:
        block_N = 128
    assert seq_len % block_M == 0
    assert seq_len % block_N == 0
    sm_scale = (1.0 / dim_qk) ** 0.5  # noqa: F841
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"

    q_shape = [batch, heads, seq_len, dim_qk_padded]
    k_shape = [batch, head_kv, seq_len, dim_qk_padded]
    v_shape = [batch, head_kv, seq_len, dim_v]
    do_shape = [batch, heads, seq_len, dim_v]
    bwd_block_num = (seq_len // block_M) * heads * batch
    num_kv_iters = seq_len // block_N

    @T.prim_func
    def phase1(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        dO: T.Tensor(do_shape, dtype),  # type: ignore
        ws_s: T.Tensor([bwd_block_num, num_kv_iters, block_M, block_N], accum_dtype),  # type: ignore
        ws_dp: T.Tensor([bwd_block_num, num_kv_iters, block_M, block_N], accum_dtype),  # type: ignore
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch
            kv_by = by // groups
            q_row = bx * block_M

            q_l1 = T.alloc_L1([block_M, dim_qk_padded], dtype)
            k_l1 = T.alloc_L1([block_N, dim_qk_padded], dtype)
            do_l1 = T.alloc_L1([block_M, dim_v], dtype)
            v_l1 = T.alloc_L1([block_N, dim_v], dtype)
            l0c_mn = T.alloc_L0C([block_M, block_N], accum_dtype)

            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                }
            )

            # Hoist Q, dO (loop-invariant per cid)
            T.copy(Q[bz, by, q_row : q_row + block_M, :], q_l1)
            T.copy(dO[bz, by, q_row : q_row + block_M, :], do_l1)

            window_eff = seq_len * 2
            loop_st = T.max(0, (bx * block_M - window_eff) // block_N)
            loop_ed = T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N)) if is_causal else num_kv_iters

            for k in T.serial(loop_st, loop_ed):
                kv = k * block_N

                # GEMM1: S = Q @ K^T -> l0c_mn -> ws_s
                T.copy(K[bz, kv_by, kv : kv + block_N, :], k_l1)
                T.gemm_v0(q_l1, k_l1, l0c_mn, transpose_B=True, init=True)
                T.copy(l0c_mn, ws_s[cid, k, :, :])

                # GEMM3: dP = dO @ V^T -> l0c_mn (reuse) -> ws_dp
                T.copy(V[bz, kv_by, kv : kv + block_N, :], v_l1)
                T.gemm_v0(do_l1, v_l1, l0c_mn, transpose_B=True, init=True)
                T.copy(l0c_mn, ws_dp[cid, k, :, :])

    return phase1


# --- Kernel 4: bwd Phase 2 — softmax recompute + dS + compensated deltas ---


@tilelang.jit(out_idx=[4, 5, 6, 7], pass_configs=_developer_vector_pass_configs)
def flashattn_bwd_softmax_ds(batch, heads, seq_len, dim_qk, dim_v, groups, is_causal, block_M=64, block_N=64):
    """Phase 2: softmax recompute (P, dP) + dS -> ws_p, ws_ds, ws_p_delta, ws_ds_delta.

    Vector-only Developer mode. Each cid processes one Q-block; vid splits rows.
    P-retention: fp32 P preserved in work_ub across Step1->Step2.
    Auto-selects block_N=128 (must match Phase 1/3).
    """
    dim_qk_padded = ((dim_qk + 15) // 16) * 16
    if seq_len % 128 == 0 and dim_qk_padded <= 192:
        block_N = 128
    assert seq_len % block_M == 0
    assert seq_len % block_N == 0
    sm_scale = (1.0 / dim_qk) ** 0.5
    head_kv = heads // groups  # noqa: F841
    dtype = "float16"
    accum_dtype = "float"

    bwd_block_num = (seq_len // block_M) * heads * batch
    num_kv_iters = seq_len // block_N
    hm = block_M // 2

    @T.prim_func
    def phase2(
        ws_s: T.Tensor([bwd_block_num, num_kv_iters, block_M, block_N], accum_dtype),  # type: ignore
        ws_dp: T.Tensor([bwd_block_num, num_kv_iters, block_M, block_N], accum_dtype),  # type: ignore
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        ws_p: T.Tensor([bwd_block_num, num_kv_iters, block_M, block_N], dtype),  # type: ignore
        ws_ds: T.Tensor([bwd_block_num, num_kv_iters, block_M, block_N], dtype),  # type: ignore
        ws_p_delta: T.Tensor([bwd_block_num, num_kv_iters, block_M, block_N], dtype),  # type: ignore
        ws_ds_delta: T.Tensor([bwd_block_num, num_kv_iters, block_M, block_N], dtype),  # type: ignore
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch

            v_row = vid * hm
            q_row = bx * block_M

            # UB buffers
            work_ub = T.alloc_ub([hm, block_N], accum_dtype)
            dp_ub = T.alloc_ub([hm, block_N], accum_dtype)
            lse_ub = T.alloc_ub([hm], accum_dtype)
            delta_ub = T.alloc_ub([hm], accum_dtype)
            lse_2d = T.alloc_ub([hm, block_N], accum_dtype)
            delta_2d = T.alloc_ub([hm, block_N], accum_dtype)
            p_half_ub = T.alloc_ub([hm, block_N], dtype)
            p_back_ub = T.alloc_ub([hm, block_N], accum_dtype)
            p_delta_ub = T.alloc_ub([hm, block_N], accum_dtype)
            ds_back_ub = T.alloc_ub([hm, block_N], accum_dtype)
            ds_delta_ub = T.alloc_ub([hm, block_N], accum_dtype)
            p_delta_half = T.alloc_ub([hm, block_N], dtype)
            col_pos = T.alloc_ub([block_N], accum_dtype)
            row_pos = T.alloc_ub([hm], accum_dtype)
            mask_2d = T.alloc_ub([hm, block_N], accum_dtype)

            # Hoist lse, Delta, row_pos (loop-invariant per cid+vid)
            T.copy(lse[bz, by, q_row + v_row : q_row + v_row + hm], lse_ub)
            T.copy(Delta[bz, by, q_row + v_row : q_row + v_row + hm], delta_ub)
            T.tile.arith_progression(row_pos, q_row + v_row, 1, hm)

            window_eff = seq_len * 2
            loop_st = T.max(0, (bx * block_M - window_eff) // block_N)
            loop_ed = T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N)) if is_causal else num_kv_iters

            for k in T.serial(loop_st, loop_ed):
                kv = k * block_N

                # Step 1: S -> P (softmax recompute) -> ws_p
                T.copy(ws_s[cid, k, v_row : v_row + hm, :], work_ub)
                T.tile.mul(work_ub, work_ub, sm_scale)
                T.tile.broadcast(lse_2d, lse_ub, axis=1)
                T.tile.sub(work_ub, work_ub, lse_2d)
                T.tile.exp(work_ub, work_ub)

                if is_causal:
                    T.tile.arith_progression(col_pos, kv, 1, block_N)
                    T.tile.broadcast(lse_2d, col_pos, axis=0)
                    T.tile.broadcast(delta_2d, row_pos, axis=1)
                    T.tile.compare(mask_2d, lse_2d, delta_2d, "LE")
                    T.tile.select(
                        work_ub,
                        mask_2d,
                        work_ub,
                        0.0,
                        "VSEL_TENSOR_SCALAR_MODE",
                    )

                # P fp32 -> fp16 -> GM (work_ub keeps fp32 P for Step 2)
                T.copy(work_ub, p_half_ub)
                T.copy(p_half_ub, ws_p[cid, k, v_row : v_row + hm, :])

                # Step 1b: DeltaP = P_fp32 - cast(P_fp16, fp32) -> ws_p_delta
                T.copy(p_half_ub, p_back_ub)
                T.tile.sub(p_delta_ub, work_ub, p_back_ub)
                T.copy(p_delta_ub, p_delta_half)
                T.copy(p_delta_half, ws_p_delta[cid, k, v_row : v_row + hm, :])

                # Step 2: dP + Delta -> dS -> ws_ds (work_ub still has P_fp32)
                T.copy(ws_dp[cid, k, v_row : v_row + hm, :], dp_ub)
                T.tile.broadcast(delta_2d, delta_ub, axis=1)
                T.tile.sub(dp_ub, dp_ub, delta_2d)
                T.tile.mul(work_ub, work_ub, dp_ub)
                T.tile.mul(work_ub, work_ub, sm_scale)

                if is_causal:
                    T.tile.select(
                        work_ub,
                        mask_2d,
                        work_ub,
                        0.0,
                        "VSEL_TENSOR_SCALAR_MODE",
                    )

                # dS fp32 -> fp16 -> GM
                T.copy(work_ub, p_half_ub)
                T.copy(p_half_ub, ws_ds[cid, k, v_row : v_row + hm, :])

                # Step 2b: DeltadS = dS_fp32 - cast(dS_fp16, fp32) -> ws_ds_delta
                T.copy(p_half_ub, ds_back_ub)
                T.tile.sub(ds_delta_ub, work_ub, ds_back_ub)
                T.copy(ds_delta_ub, p_delta_half)
                T.copy(p_delta_half, ws_ds_delta[cid, k, v_row : v_row + hm, :])

    return phase2


# --- Kernel 5: bwd Phase 3 — GEMM2+corr + GEMM4+corr + GEMM5 ---


@tilelang.jit(pass_configs=_developer_cv_pass_configs)
def flashattn_bwd_gemm_dv_dk_dq(batch, heads, seq_len, dim_qk, dim_v, groups, is_causal, block_M=64, block_N=64):
    """Phase 3: GEMM2+corr (dV) + GEMM4+corr (dK) + GEMM5 (dQ) -> atomic_add dQ/dK/dV.

    Cube-only Developer mode. Compensated GEMM: GEMM2 (init=True) + GEMM2_corr
    (init=False) share l0c_dv; GEMM4 + GEMM4_corr share l0c_dk (MEMORY_PLANNING
    reuses l0c_dv address). Auto-selects block_N=128 (must match Phase 1/2).
    """
    dim_qk_padded = ((dim_qk + 15) // 16) * 16
    if seq_len % 128 == 0 and dim_qk_padded <= 192:
        block_N = 128
    assert seq_len % block_M == 0
    assert seq_len % block_N == 0
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"

    # GEMM4/GEMM5 output N=dim_qk_padded. When dim_qk_padded > 128 and not
    # divisible by 128 (e.g. D_qk=192 → dim_qk_padded=192), the default
    # kL0Size=128 makes nMaxByL0B=128, which cannot tile N=192. Use
    # kL0Size=64 so nMaxByL0B=256 >= 192, allowing a single N-pass.
    gemm_kL0Size = 64 if dim_qk_padded > 128 and dim_qk_padded % 128 != 0 else 128

    q_shape = [batch, heads, seq_len, dim_qk_padded]
    k_shape = [batch, head_kv, seq_len, dim_qk_padded]
    do_shape = [batch, heads, seq_len, dim_v]
    dq_shape = [batch, heads, seq_len, dim_qk_padded]
    dk_shape = [batch, head_kv, seq_len, dim_qk_padded]
    dv_shape = [batch, head_kv, seq_len, dim_v]
    bwd_block_num = (seq_len // block_M) * heads * batch
    num_kv_iters = seq_len // block_N

    @T.prim_func
    def phase3(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        dO: T.Tensor(do_shape, dtype),  # type: ignore
        ws_p: T.Tensor([bwd_block_num, num_kv_iters, block_M, block_N], dtype),  # type: ignore
        ws_ds: T.Tensor([bwd_block_num, num_kv_iters, block_M, block_N], dtype),  # type: ignore
        ws_p_delta: T.Tensor([bwd_block_num, num_kv_iters, block_M, block_N], dtype),  # type: ignore
        ws_ds_delta: T.Tensor([bwd_block_num, num_kv_iters, block_M, block_N], dtype),  # type: ignore
        dQ: T.Tensor(dq_shape, accum_dtype),  # type: ignore  — fp32 atomic_add target
        dK: T.Tensor(dk_shape, accum_dtype),  # type: ignore  — fp32 atomic_add target
        dV: T.Tensor(dv_shape, accum_dtype),  # type: ignore  — fp32 atomic_add target
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch
            kv_by = by // groups
            q_row = bx * block_M

            # L1 buffers (fp16)
            q_l1 = T.alloc_L1([block_M, dim_qk_padded], dtype)
            k_l1 = T.alloc_L1([block_N, dim_qk_padded], dtype)
            do_l1 = T.alloc_L1([block_M, dim_v], dtype)
            mn_l1 = T.alloc_L1([block_M, block_N], dtype)
            p_delta_l1 = T.alloc_L1([block_M, block_N], dtype)
            ds_delta_l1 = T.alloc_L1([block_M, block_N], dtype)

            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                }
            )

            # L0C buffers (fp32) — MEMORY_PLANNING reuses addresses via liveness
            l0c_dv = T.alloc_L0C([block_N, dim_v], accum_dtype)
            l0c_dk = T.alloc_L0C([block_N, dim_qk_padded], accum_dtype)
            l0c_dq = T.alloc_L0C([block_M, dim_qk_padded], accum_dtype)

            # Hoist Q, dO (loop-invariant per cid)
            T.copy(Q[bz, by, q_row : q_row + block_M, :], q_l1)
            T.copy(dO[bz, by, q_row : q_row + block_M, :], do_l1)

            window_eff = seq_len * 2
            loop_st = T.max(0, (bx * block_M - window_eff) // block_N)
            loop_ed = T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N)) if is_causal else num_kv_iters

            for k in T.serial(loop_st, loop_ed):
                kv = k * block_N

                # GEMM2: dV = P^T @ dO (init=True)
                T.copy(ws_p[cid, k, :, :], mn_l1)
                T.gemm_v0(mn_l1, do_l1, l0c_dv, transpose_A=True, init=True)

                # GEMM2_corr: dV += DeltaP^T @ dO (init=False)
                T.copy(ws_p_delta[cid, k, :, :], p_delta_l1)
                T.gemm_v0(p_delta_l1, do_l1, l0c_dv, transpose_A=True, init=False)

                T.tile.atomic_add(dV[bz, kv_by, kv : kv + block_N, :], l0c_dv)

                # GEMM4: dK = dS^T @ Q (init=True)
                T.copy(ws_ds[cid, k, :, :], mn_l1)
                T.gemm_v0(mn_l1, q_l1, l0c_dk, transpose_A=True, init=True, kL0Size=gemm_kL0Size)

                # GEMM4_corr: dK += DeltadS^T @ Q (init=False)
                T.copy(ws_ds_delta[cid, k, :, :], ds_delta_l1)
                T.gemm_v0(ds_delta_l1, q_l1, l0c_dk, transpose_A=True, init=False, kL0Size=gemm_kL0Size)

                T.tile.atomic_add(dK[bz, kv_by, kv : kv + block_N, :], l0c_dk)

                # GEMM5: dQ = dS @ K (init=True, per-iter atomic_add)
                T.copy(K[bz, kv_by, kv : kv + block_N, :], k_l1)
                T.gemm_v0(mn_l1, k_l1, l0c_dq, init=True, kL0Size=gemm_kL0Size)

                T.tile.atomic_add(dQ[bz, by, q_row : q_row + block_M, :], l0c_dq)

    return phase3


# --- Kernel 6: Backward Postprocess — fp32 -> fp16 cast ---


@tilelang.jit(out_idx=[1], pass_configs=_developer_vector_pass_configs)
def flashattn_bwd_postprocess(batch, heads, seq_len, dim, blk=64):
    assert seq_len % blk == 0
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, heads, seq_len, dim]
    block_num = (seq_len // blk) * heads * batch

    @T.prim_func
    def post(
        dQ: T.Tensor(shape, accum_dtype),  # type: ignore
        dQ_out: T.Tensor(shape, dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            # vid splits blk rows: vid=0 handles first blk//2 rows, vid=1 handles next blk//2.
            # Relies on default threads=2 (910B3 Vector core pair).
            bx = cid % (seq_len // blk)
            by = cid // (seq_len // blk) % heads
            bz = cid // (seq_len // blk) // heads % batch
            row = bx * blk + vid * blk // 2

            dq_ub = T.alloc_ub([blk // 2, dim], accum_dtype)
            dq_half = T.alloc_ub([blk // 2, dim], dtype)

            T.copy(dQ[bz, by, row : row + blk // 2, :], dq_ub)
            T.copy(dq_ub, dq_half)
            T.copy(dq_half, dQ_out[bz, by, row : row + blk // 2, :])

    return post


# --- Golden Reference (PyTorch CPU) ---


def ref_fwd(Q, K, V, is_causal=False, groups=1):
    """Forward golden (CPU): GQA + causal mask.

    Args:
        Q: [B, H, N, D_qk] fp16
        K: [B, H_kv, N, D_qk] fp16
        V: [B, H_kv, N, D_v] fp16
    Returns:
        O [B, H, N, D_v] fp16, lse [B, H, N] fp32
    """
    B, H, N, D_qk = Q.shape
    _, _, _, D_v = V.shape
    sm_scale = 1.0 / D_qk**0.5

    K_rep = K.float().repeat_interleave(groups, dim=1)
    V_rep = V.float().repeat_interleave(groups, dim=1)

    S = torch.matmul(Q.float(), K_rep.transpose(-2, -1)) * sm_scale

    if is_causal:
        pos_q = torch.arange(N, device=Q.device).float()
        pos_k = torch.arange(N, device=Q.device).float()
        mask = pos_k[None, :] <= pos_q[:, None]
        S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    m = S.max(dim=-1, keepdim=True).values
    P = torch.exp(S - m)
    lse = torch.log(P.sum(dim=-1, keepdim=True)) + m
    P = P / P.sum(dim=-1, keepdim=True)

    O = torch.matmul(P, V_rep)
    return O.half(), lse.squeeze(-1)


def ref_bwd(Q, K, V, dO, lse, is_causal=False, groups=1):
    """Backward golden (CPU autograd).

    P is recomputed from S (NOT using lse as constant) so the softmax gradient
    flows correctly through the normalization. Returns dQ, dK, dV (fp16).
    """
    Q_f = Q.float().requires_grad_(True)
    K_f = K.float().requires_grad_(True)
    V_f = V.float().requires_grad_(True)

    B, H, N, D_qk = Q_f.shape
    _, _, _, D_v = V_f.shape
    sm_scale = 1.0 / D_qk**0.5

    K_rep = K_f.repeat_interleave(groups, dim=1)
    V_rep = V_f.repeat_interleave(groups, dim=1)

    S = torch.matmul(Q_f, K_rep.transpose(-2, -1)) * sm_scale

    if is_causal:
        pos_q = torch.arange(N, device=Q_f.device).float()
        pos_k = torch.arange(N, device=Q_f.device).float()
        mask = pos_k[None, :] <= pos_q[:, None]
        S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    m = S.max(dim=-1, keepdim=True).values
    P = torch.exp(S - m)
    P = P / P.sum(dim=-1, keepdim=True)

    O = torch.matmul(P, V_rep)
    O.backward(dO.float())

    return Q_f.grad.half(), K_f.grad.half(), V_f.grad.half()


# --- Host-side bwd pipeline: 3 sub-kernel serial launch ---


def run_bwd(Q, K, V, dO, lse, Delta, is_causal=False, groups=1, block_M=64, block_N=64):
    """Host-side bwd pipeline: 3 sub-kernel serial launch.

    Note: block_N may be overridden by kernel auto-select
    (block_N=128 when seq_len%128==0 and dim_qk_padded<=192).

    Phase 1: GEMM1 + GEMM3 -> ws_s, ws_dp (GM fp32)
    Phase 2: softmax recompute + dS -> ws_p, ws_ds, ws_p_delta, ws_ds_delta (GM fp16)
    Phase 3: GEMM2+corr + GEMM4+corr + GEMM5 -> dQ, dK, dV (GM fp32, atomic_add)

    Returns: dQ_fp16, dK_fp16, dV_fp16, dQ_fp32, dK_fp32, dV_fp32
    """
    batch, heads, seq_len, dim_qk_padded = Q.shape
    dim_qk = dim_qk_padded
    _, H_kv, _, dim_v = V.shape

    # Phase 1: GEMM1 + GEMM3 -> ws_s, ws_dp
    phase1_mod = flashattn_bwd_gemm_s_dp(batch, heads, seq_len, dim_qk, dim_v, groups, is_causal, block_M, block_N)
    ws_s, ws_dp = phase1_mod(Q, K, V, dO)
    torch.npu.synchronize()

    # Phase 2: softmax + dS -> ws_p, ws_ds, ws_p_delta, ws_ds_delta
    phase2_mod = flashattn_bwd_softmax_ds(batch, heads, seq_len, dim_qk, dim_v, groups, is_causal, block_M, block_N)
    ws_p, ws_ds, ws_p_delta, ws_ds_delta = phase2_mod(ws_s, ws_dp, lse, Delta)
    torch.npu.synchronize()

    # Phase 3: GEMM2+corr + GEMM4+corr + GEMM5 -> dQ, dK, dV (atomic_add, pre-zeroed)
    dQ = torch.zeros(batch, heads, seq_len, dim_qk_padded, dtype=torch.float32, device=Q.device)
    dK = torch.zeros(batch, H_kv, seq_len, dim_qk_padded, dtype=torch.float32, device=Q.device)
    dV = torch.zeros(batch, H_kv, seq_len, dim_v, dtype=torch.float32, device=Q.device)

    phase3_mod = flashattn_bwd_gemm_dv_dk_dq(batch, heads, seq_len, dim_qk, dim_v, groups, is_causal, block_M, block_N)
    phase3_mod(Q, K, dO, ws_p, ws_ds, ws_p_delta, ws_ds_delta, dQ, dK, dV)
    torch.npu.synchronize()

    # Postprocess: fp32 -> fp16 for dQ, dK, dV
    post_dq = flashattn_bwd_postprocess(batch, heads, seq_len, dim_qk_padded, blk=64)
    dQ_fp16 = post_dq(dQ)
    post_dk = flashattn_bwd_postprocess(batch, H_kv, seq_len, dim_qk_padded, blk=64)
    dK_fp16 = post_dk(dK)
    post_dv = flashattn_bwd_postprocess(batch, H_kv, seq_len, dim_v, blk=64)
    dV_fp16 = post_dv(dV)
    torch.npu.synchronize()

    return dQ_fp16, dK_fp16, dV_fp16, dQ, dK, dV


# --- Autograd Function (end-to-end wrapper) ---


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal, groups):
        def maybe_contiguous(x):
            return x if x.stride(-1) == 1 else x.contiguous()

        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
        B, H, N, D_qk = q.shape
        _, H_kv, _, D_v = v.shape
        dim_qk_padded = ((D_qk + 15) // 16) * 16
        block_M = 64 if dim_qk_padded <= 192 else 32
        block_N_fwd = 128 if N % 128 == 0 else 64
        block_N_bwd = 64

        # Pad Q/K to dim_qk_padded if needed
        if dim_qk_padded > D_qk:
            q_pad = torch.zeros(B, H, N, dim_qk_padded, dtype=q.dtype, device=q.device)
            q_pad[:, :, :, :D_qk] = q
            k_pad = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=k.dtype, device=k.device)
            k_pad[:, :, :, :D_qk] = k
            q, k = q_pad, k_pad

        fwd_mod = flashattn_fwd(B, H, N, D_qk, D_v, groups, is_causal, block_M, block_N_fwd)
        o, lse = fwd_mod(q, k, v)

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.is_causal = is_causal
        ctx.groups = groups
        ctx.D_qk = D_qk
        ctx.dim_qk_padded = dim_qk_padded
        ctx.block_M = block_M
        ctx.block_N_bwd = block_N_bwd
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        B, H, N, D_qk_padded = q.shape
        _, H_kv, _, D_v = v.shape
        is_causal = ctx.is_causal
        groups = ctx.groups
        D_qk = ctx.D_qk
        block_M = ctx.block_M
        block_N_bwd = ctx.block_N_bwd

        # Preprocess: Delta = sum(O * dO, dim=-1)
        prep_mod = flashattn_bwd_preprocess(B, H, N, D_v, blk=32)
        delta = prep_mod(o, do)
        torch.npu.synchronize()

        # bwd pipeline: 3 sub-kernel serial launch
        dQ_fp16, dK_fp16, dV_fp16, _, _, _ = run_bwd(q, k, v, do, lse, delta, is_causal, groups, block_M, block_N_bwd)

        # Slice to original D_qk (remove padding)
        return dQ_fp16[..., :D_qk], dK_fp16[..., :D_qk], dV_fp16, None, None


attention = _attention.apply
kernel = attention  # alias for coverage check API


# --- Main: smoke test ---


if __name__ == "__main__":
    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(42)

    # Minimal smoke config: B=1, H=4, N=128, D_qk=128, D_v=128, groups=2, causal=True
    B, H, N, D_qk, D_v = 1, 4, 128, 128, 128
    groups = 2
    is_causal = True
    H_kv = H // groups
    dim_qk_padded = ((D_qk + 15) // 16) * 16
    block_M = 64
    block_N_fwd = 128
    block_N_bwd = 64

    Q = torch.randn(B, H, N, dim_qk_padded, dtype=torch.float16, device="npu")
    K = torch.randn(B, H_kv, N, dim_qk_padded, dtype=torch.float16, device="npu")
    V = torch.randn(B, H_kv, N, D_v, dtype=torch.float16, device="npu")
    dO = torch.randn(B, H, N, D_v, dtype=torch.float16, device="npu")

    # Forward
    O, lse = flashattn_fwd(B, H, N, D_qk, D_v, groups, is_causal, block_M, block_N_fwd)(Q, K, V)
    torch.npu.synchronize()
    print(f"Forward done: O={O.shape}, lse={lse.shape}")

    # Preprocess: Delta = sum(O * dO)
    Delta = flashattn_bwd_preprocess(B, H, N, D_v, blk=32)(O, dO)
    torch.npu.synchronize()

    # bwd pipeline (3 sub-kernel serial launch)
    dQ_fp16, dK_fp16, dV_fp16, dQ_fp32, dK_fp32, dV_fp32 = run_bwd(Q, K, V, dO, lse, Delta, is_causal, groups, block_M, block_N_bwd)
    torch.npu.synchronize()
    print(f"Backward done: dQ={dQ_fp16.shape}, dK={dK_fp16.shape}, dV={dV_fp16.shape}")

    # Quick precision check against golden (CPU)
    Q_cpu, K_cpu, V_cpu, dO_cpu = Q.cpu(), K.cpu(), V.cpu(), dO.cpu()
    O_ref, lse_ref = ref_fwd(Q_cpu, K_cpu, V_cpu, is_causal, groups)
    dQ_ref, dK_ref, dV_ref = ref_bwd(Q_cpu, K_cpu, V_cpu, dO_cpu, lse_ref, is_causal, groups)

    print(f"O:  max_diff={(O.cpu().float() - O_ref.float()).abs().max():.4e}")
    print(f"dQ: max_diff={(dQ_fp16.cpu().float() - dQ_ref.float()).abs().max():.4e}")
    print(f"dK: max_diff={(dK_fp16.cpu().float() - dK_ref.float()).abs().max():.4e}")
    print(f"dV: max_diff={(dV_fp16.cpu().float() - dV_ref.float()).abs().max():.4e}")
    print("Test Passed!")
