"""NSA Backward (Native Sparse Attention Backward) for Ascend NPU.

2 @tilelang.jit kernels + pipeline runner + 1 CI smoke case. CPU block_mask for fwd only.
Golden reference and precision testing: see test_tilelang_nsa_bwd.py.
Performance benchmarking (do_bench / msprof): see perf_bench_nsa_bwd.py.

Architecture (fresh redesign — single-kernel bwd, bhsd pattern):
  1. nsa_fwd           (Developer/hybrid) — Forward (for e2e tests)
  2. nsa_bwd_single    (Developer/hybrid) — Single-kernel bwd (5 GEMM + softmax + Delta + dS)

nsa_bwd_single directly follows bhsd flashattn_bwd pattern:
  - alloc_shared / alloc_fragment (NOT alloc_L1/alloc_ub/alloc_L0C + T.annotate_layout)
  - threads=1 (no vid)
  - Developer + combineCV (4 True)
  - On-chip direct T.copy (L0C->UB / UB->L1) — eliminates 9 GM workspace buffers
  - 2-level Compensated GEMM preserved (ds_delta + ds_delta2)

Per-K-block loop structure (NOT reversed):
  cid -> i_s (K block), i_b, i_h (KV head)
  loop: for i in T.serial(i_s*BS, seq_len)  # Q tokens (causal: Q >= K block start)

5 GEMMs per iter + softmax + Delta + dS (all on-chip):
  GEMM1 (qkT) -> softmax -> GEMM2 (dsT) + GEMM3 (dV)
    -> Delta + dS -> GEMM4 (dK) + GEMM5 (dQ)

Constraint: NS=1 (T in [BS, 2*BS-1]) required for correctness. The bwd kernel
does not check block_mask; Delta is computed per-iter from the single K block's
P and dsT. For NS>1, Delta would need summation across K blocks (not supported).

Key NPU adaptations vs GPU reference:
  - T.exp2 -> T.tile.exp (natural log domain; scale = 1/sqrt(D))
  - T.gemm -> T.gemm_v0 (no policy, has init)
  - T.atomic_add -> T.copy (NV=1, no competition)
  - T.Pipelined -> T.serial
  - T.Kernel(NV, NS, B*H, threads=32) -> T.Kernel(block_num, threads=1, is_npu=True)
  - block_mask computed on CPU for fwd (NPU scalar GM read unreliable); bwd uses kernel-internal causal mask
"""

import tilelang
import torch
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

# ============================================================================
# pass_configs
# ============================================================================

# Hybrid mode for fwd + bwd_single — all 4 CV keys True + TAIL_MASK.
# AUTO_CV_COMBINE/SYNC needed for on-chip L0C->UB / UB->L1 direct T.copy.
# TAIL_MASK enables causal boundary handling for non-aligned seq_len.
_hybrid_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
}


# ============================================================================
# CPU Function: block_mask — Build binary mask from block_indices + block_counts
# ============================================================================


def _compute_block_mask_cpu(block_indices, block_counts, BS):
    """Compute block_mask on CPU (int32) for fwd sparse selection.

    NPU scalar GM BufferLoad is unreliable for conditional logic. With NS=1
    (T in [BS, 2*BS-1]) and S=1, block_mask is all-ones for the single selected
    block, so CPU compute is simple and reliable. Result H2D to NPU as GM int32
    tensor. bwd_single does not use block_mask (uses kernel-internal causal mask).
    """
    B, T_len, H, S = block_indices.shape
    NS = T_len // BS
    block_mask = torch.zeros(B, T_len, H, NS, dtype=torch.int32)
    for b in range(B):
        for t in range(T_len):
            for h in range(H):
                for s in range(S):
                    b_i = block_indices[b, t, h, s].item()
                    b_c = block_counts[b, t, h].item()
                    ns = b_i  # K block index
                    if 0 <= ns < NS and b_i * BS <= t and s < b_c:
                        block_mask[b, t, h, ns] = 1
    return block_mask


# ============================================================================
# Test data generator (shared by __main__ smoke test and test_tilelang_nsa_bwd)
# ============================================================================


def _generate_test_data(B=1, T=32, H=1, HQ=16, D=32, S=1, BS=32, dtype=torch.float16):
    """Generate deterministic test data on CPU (then H2D in caller).

    Must use device='cpu' explicitly — torch.set_default_device('npu') would
    create NPU tensors, causing the golden to run on NPU with different precision.
    """
    torch.manual_seed(0)
    q = torch.randn(B, T, HQ, D, dtype=dtype, device="cpu")
    k = torch.randn(B, T, H, D, dtype=dtype, device="cpu")
    v = torch.randn(B, T, H, D, dtype=dtype, device="cpu")
    do_slc = torch.randn(B, T, HQ, D, dtype=dtype, device="cpu")

    # block_indices: each Q token selects from [0, t//BS) blocks
    block_indices = torch.full((B, T, H, S), T, dtype=torch.int32, device="cpu")
    for b in range(B):
        for t in range(T):
            for h in range(H):
                avail = max(1, t // BS)
                perm = torch.randperm(avail)[:S]
                block_indices[b, t, h, : len(perm)] = perm
    block_indices = block_indices.sort(-1)[0]

    # block_counts: 1..S+1
    block_counts = torch.randint(1, S + 1, (B, T, H), dtype=torch.int32, device="cpu")

    return q, k, v, do_slc, block_indices, block_counts


# ============================================================================
# Kernel 1: nsa_fwd — Forward (online softmax + sparse K block selection)
# ============================================================================


@tilelang.jit(out_idx=[4, 5], workspace_idx=[6, 7, 8], pass_configs=_hybrid_pass_configs)
def nsa_fwd(
    batch,
    seq_len,
    heads,
    heads_kv,
    dim,
    groups,
    selected_blocks,
    block_size,
):
    """Forward: produces O_slc [B, T, HQ, D] fp16 and LSE_slc [B, T, HQ] fp32.

    Hybrid mode (Developer + AUTO_CV_COMBINE/SYNC + T.serial).
    Each block handles 1 Q token x 1 V slice. Iterates over S selected K blocks.
    Uses T.tile.exp (natural log domain); scale = 1/sqrt(D) (no log2(e)).
    LSE fix: m_i is already scaled, so no additional sm_scale multiplication.
    """
    sm_scale = (1.0 / dim) ** 0.5
    G = groups
    S = selected_blocks
    BS = block_size
    BK = dim
    BV = dim
    NV = 1
    hm = G // 2  # per-vid Q head count
    dtype = "float16"
    accum_dtype = "float"

    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, heads_kv, dim]
    o_shape = [batch, seq_len, heads, dim]
    lse_shape = [batch, seq_len, heads]
    bi_shape = [batch, seq_len, heads_kv, S]
    block_num = seq_len * NV * batch * heads_kv

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(kv_shape, dtype),  # type: ignore
        V: T.Tensor(kv_shape, dtype),  # type: ignore
        BlockIndices: T.Tensor(bi_shape, "int32"),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        workspace_1: T.Tensor([block_num, G, BS], accum_dtype),  # type: ignore
        workspace_2: T.Tensor([block_num, G, BS], dtype),  # type: ignore
        workspace_3: T.Tensor([block_num, G, BV], accum_dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            i_t = cid // (NV * batch * heads_kv)
            _i_v = (cid // (batch * heads_kv)) % NV
            i_bh = cid % (batch * heads_kv)
            i_b = i_bh // heads_kv
            i_h = i_bh % heads_kv

            # L1 buffers (Cube scope) — full G dimension
            q_l1 = T.alloc_L1([G, BK], dtype)
            k_l1 = T.alloc_L1([BS, BK], dtype)
            v_l1 = T.alloc_L1([BS, BV], dtype)
            acc_s_l1 = T.alloc_L1([G, BS], dtype)
            acc_s_l0c = T.alloc_L0C([G, BS], accum_dtype)
            acc_o_l0c = T.alloc_L0C([G, BV], accum_dtype)

            # OP-R12: ZN/NZ fractal layout for L1 buffers (borrowed from
            # NSA-Forward-Varlen iter5 dir4 + mla_decode_paged). Each L1 buffer
            # is used by only ONE GEMM role (no layout conflict):
            #   q_l1: GEMM1 A input (transpose_B=True) → ZN
            #   k_l1: GEMM1 B input (transpose_B=True) → NZ
            #   acc_s_l1: GEMM2 A input (no transpose) → ZN
            #   v_l1: GEMM2 B input (no transpose) → NZ
            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                    acc_s_l1: make_zn_layout(acc_s_l1),
                    v_l1: make_nz_layout(v_l1),
                }
            )

            # UB buffers (Vector scope) — per-vid half of G
            v_row = vid * hm
            acc_o = T.alloc_ub([hm, BV], accum_dtype)
            sumexp = T.alloc_ub([hm], accum_dtype)
            m_i = T.alloc_ub([hm], accum_dtype)
            acc_s_ub = T.alloc_ub([hm, BS], accum_dtype)
            m_i_prev = T.alloc_ub([hm], accum_dtype)
            acc_s_ub_ = T.alloc_ub([hm, BS], accum_dtype)
            sumexp_i_ub = T.alloc_ub([hm], accum_dtype)
            acc_s_half = T.alloc_ub([hm, BS], dtype)
            acc_o_ub = T.alloc_ub([hm, BV], accum_dtype)
            acc_o_half = T.alloc_ub([hm, BV], dtype)
            col_pos = T.alloc_ub([BS], accum_dtype)
            row_pos = T.alloc_ub([hm], accum_dtype)
            row_pos_2d = T.alloc_ub([hm, BS], accum_dtype)
            mask_2d = T.alloc_ub([hm, BS], accum_dtype)

            # Load Q (loop-invariant for this Q token)
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], q_l1)

            # row_pos = i_t for all hm rows (Q token position)
            T.tile.arith_progression(row_pos, i_t, 0, hm)

            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -(2**30))

            # Iterate over S selected K blocks
            for i_s_iter in T.serial(S):
                kv_start = BlockIndices[i_b, i_t, i_h, i_s_iter] * BS

                # Cube: QK^T -> acc_s_l0c -> workspace_1
                T.copy(K[i_b, kv_start : kv_start + BS, i_h, :], k_l1)
                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                T.copy(acc_s_l0c, workspace_1[cid, :, :])

                # Vector: softmax + causal mask
                T.tile.fill(acc_s_ub, 0.0)
                T.copy(m_i, m_i_prev)
                T.copy(workspace_1[cid, v_row : v_row + hm, :], acc_s_ub_)
                # acc_s_ub = sm_scale * acc_s_ub_ (scaled scores)
                T.tile.mul(acc_s_ub, acc_s_ub_, sm_scale)

                # Causal mask: K_token (kv_start + col) <= Q_token (i_t)
                T.tile.arith_progression(col_pos, kv_start, 1, BS)
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

                # Online softmax: max -> sub -> exp -> sum
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
                # Rescale acc_o
                T.tile.broadcast(acc_o_ub, m_i_prev, axis=1)
                T.tile.mul(acc_o, acc_o, acc_o_ub)

                # Copy P (fp16) per-vid slice to workspace_2
                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, workspace_2[cid, v_row : v_row + hm, :])

                # Cube: PV -> acc_o_l0c (read full G from workspace_2)
                T.copy(workspace_2[cid, :, :], acc_s_l1)
                T.copy(V[i_b, kv_start : kv_start + BS, i_h, :], v_l1)
                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)

                # Vector: accumulate acc_o (read per-vid slice from workspace_3)
                T.copy(acc_o_l0c, workspace_3[cid, :, :])
                T.copy(workspace_3[cid, v_row : v_row + hm, :], acc_o_ub)
                T.tile.add(acc_o, acc_o, acc_o_ub)

            # Normalize: O /= sumexp (guard against div-by-zero when block_counts=0)
            T.tile.broadcast(acc_o_ub, sumexp, axis=1)
            T.tile.max(acc_o_ub, acc_o_ub, 1e-30)
            T.tile.div(acc_o, acc_o, acc_o_ub)

            # Write O_slc (per-vid slice)
            T.copy(acc_o, acc_o_half)
            T.copy(
                acc_o_half,
                Output[
                    i_b,
                    i_t,
                    i_h * G + v_row : i_h * G + v_row + hm,
                    :,
                ],
            )

            # LSE = ln(sumexp) + m_i (natural log domain)
            T.tile.ln(sumexp, sumexp)
            T.tile.add(sumexp, sumexp, m_i)
            T.copy(sumexp, lse[i_b, i_t, i_h * G + v_row : i_h * G + v_row + hm])

    return main


# ============================================================================
# Kernel 2: nsa_bwd_single — Single-kernel backward (on-chip, 5 GEMM merged)
# ============================================================================


@tilelang.jit(pass_configs=_hybrid_pass_configs)
def nsa_bwd_single(batch, seq_len, heads, heads_kv, dim, groups, block_size):
    """Single-kernel NSA backward (on-chip, 5 GEMM merged from scratch).

    Directly follows bhsd flashattn_bwd pattern: alloc_shared/fragment,
    threads=1, Developer + combineCV (4 True), on-chip direct T.copy.

    Per-K-block loop structure (NOT reversed):
      cid -> i_s (K block), i_b, i_h
      loop: for i in T.serial(i_s*BS, seq_len)  # Q tokens

    5 GEMMs per iter + softmax + Delta + dS (all on-chip):
      GEMM1 (qkT) -> softmax -> GEMM2 (dsT) + GEMM3 (dV)
        -> Delta + dS -> GEMM4 (dK) + GEMM5 (dQ)

    Inputs:  Q [B,T,HQ,D] fp16, K [B,T,H,D] fp16, V [B,T,H,D] fp16,
             DO_slc [B,T,HQ,D] fp16, LSE_slc [B,T,HQ] fp32
    Outputs: DQ [B,T,HQ,D] fp16, DK [B,T,H,D] fp16, DV [B,T,H,D] fp16
             (caller pre-allocates DQ/DK/DV as zeros; kernel writes via T.copy)

    Constraint: NS=1 required for correctness (Delta computed per-iter from
    single K block's P and dsT; for NS>1 would need cross-block sum).
    """
    sm_scale = (1.0 / dim) ** 0.5
    G = groups
    BS = block_size
    BK = dim
    BV = dim
    NS = seq_len // BS
    NV = 1
    dtype = "float16"
    accum_dtype = "float"

    q_shape = [batch, seq_len, heads, dim]
    k_shape = [batch, seq_len, heads_kv, dim]
    v_shape = [batch, seq_len, heads_kv, dim]
    do_shape = [batch, seq_len, heads, dim]
    lse_shape = [batch, seq_len, heads]
    dk_shape = [batch, seq_len, heads_kv, dim]
    dv_shape = [batch, seq_len, heads_kv, dim]
    dq_shape = [batch, seq_len, heads, dim]
    bwd_block_num = NV * NS * batch * heads_kv

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        DO_slc: T.Tensor(do_shape, dtype),  # type: ignore
        LSE_slc: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        DQ: T.Tensor(dq_shape, dtype),  # type: ignore
        DK: T.Tensor(dk_shape, dtype),  # type: ignore
        DV: T.Tensor(dv_shape, dtype),  # type: ignore
    ):
        with T.Kernel(bwd_block_num, threads=1, is_npu=True) as (cid):
            # cid decoding (per-K-block)
            _i_v = cid // (NS * batch * heads_kv)
            i_s = (cid // (batch * heads_kv)) % NS
            i_bh = cid % (batch * heads_kv)
            i_b = i_bh // heads_kv
            i_h = i_bh % heads_kv

            # loop bounds
            loop_st = i_s * BS
            loop_ed = seq_len

            # L1 buffers (alloc_shared, fp16)
            k_l1 = T.alloc_shared([BS, BK], dtype)
            v_l1 = T.alloc_shared([BS, BV], dtype)
            q_l1 = T.alloc_shared([G, BK], dtype)
            do_l1 = T.alloc_shared([G, BV], dtype)
            p_l1 = T.alloc_shared([BS, G], dtype)
            p_delta_l1 = T.alloc_shared([BS, G], dtype)
            ds_l1 = T.alloc_shared([BS, G], dtype)
            ds_delta_l1 = T.alloc_shared([BS, G], dtype)
            ds_delta2_l1 = T.alloc_shared([BS, G], dtype)

            # L0C buffers (alloc_fragment, fp32)
            l0c_qkt = T.alloc_fragment([BS, G], accum_dtype)
            l0c_dst = T.alloc_fragment([BS, G], accum_dtype)
            l0c_dv = T.alloc_fragment([BS, BV], accum_dtype)
            l0c_dk = T.alloc_fragment([BS, BK], accum_dtype)
            l0c_dq = T.alloc_fragment([G, BK], accum_dtype)

            # UB buffers (alloc_shared)
            qkt_ub = T.alloc_shared([BS, G], accum_dtype)  # P_fp32 -> dS_fp32
            lse_ub = T.alloc_shared([G], accum_dtype)
            lse_2d = T.alloc_shared([BS, G], accum_dtype)
            col_pos = T.alloc_shared([BS], accum_dtype)
            row_1d = T.alloc_shared([BS], accum_dtype)
            row_2d = T.alloc_shared([BS, G], accum_dtype)
            mask_2d = T.alloc_shared([BS, G], accum_dtype)
            p_half = T.alloc_shared([BS, G], dtype)
            p_delta_ub = T.alloc_shared([BS, G], accum_dtype)
            p_delta_half = T.alloc_shared([BS, G], dtype)
            dst_ub = T.alloc_shared([BS, G], accum_dtype)
            delta_ub = T.alloc_shared([G], accum_dtype)
            delta_2d = T.alloc_shared([BS, G], accum_dtype)
            ds_half = T.alloc_shared([BS, G], dtype)
            ds_delta_ub = T.alloc_shared([BS, G], accum_dtype)
            ds_delta_half = T.alloc_shared([BS, G], dtype)
            ds_delta2_ub = T.alloc_shared([BS, G], accum_dtype)
            ds_delta2_half = T.alloc_shared([BS, G], dtype)
            ds_rec_ub = T.alloc_shared([BS, G], accum_dtype)

            # loop-invariant loads
            T.copy(K[i_b, i_s * BS : (i_s + 1) * BS, i_h, :BK], k_l1)
            T.copy(V[i_b, i_s * BS : (i_s + 1) * BS, i_h, :BV], v_l1)

            # col_pos (loop-invariant): K token positions
            T.tile.arith_progression(col_pos, i_s * BS, 1, BS)

            # loop over Q tokens (causal: Q >= K block start)
            for i in T.serial(loop_st, loop_ed):
                # per-iter loads
                T.copy(Q[i_b, i, i_h * G : (i_h + 1) * G, :BK], q_l1)
                T.copy(DO_slc[i_b, i, i_h * G : (i_h + 1) * G, :BV], do_l1)
                T.copy(LSE_slc[i_b, i, i_h * G : (i_h + 1) * G], lse_ub)

                # GEMM1: qkT = K @ Q^T (transpose_B=True, init=True)
                T.gemm_v0(k_l1, q_l1, l0c_qkt, transpose_B=True, init=True)

                # L0C -> UB (on-chip direct)
                T.copy(l0c_qkt, qkt_ub)

                # softmax scale + exp
                T.tile.fill(lse_2d, 0.0)
                T.tile.axpy(lse_2d, qkt_ub, sm_scale)  # lse_2d = sm_scale * qkt
                T.tile.broadcast(qkt_ub, lse_ub, axis=0)  # [G] -> [BS, G]
                T.tile.sub(lse_2d, lse_2d, qkt_ub)  # scaled_qkt - lse
                T.tile.exp(qkt_ub, lse_2d)  # qkt_ub = P_fp32

                # causal mask
                T.tile.arith_progression(row_1d, i, 0, BS)
                T.tile.broadcast(lse_2d, col_pos, axis=1)  # [BS] -> [BS, G]
                T.tile.broadcast(row_2d, row_1d, axis=1)  # [BS] -> [BS, G]
                T.tile.compare(mask_2d, lse_2d, row_2d, "LE")
                T.tile.select(qkt_ub, mask_2d, qkt_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")

                # P_fp16 + p_delta
                T.copy(qkt_ub, p_half)  # fp32 -> fp16
                T.copy(p_half, p_l1)  # UB -> L1
                T.copy(p_half, p_delta_ub)  # fp16 -> fp32
                T.tile.sub(p_delta_ub, qkt_ub, p_delta_ub)  # p_delta = P - cast(P_fp16)
                T.copy(p_delta_ub, p_delta_half)  # fp32 -> fp16
                T.copy(p_delta_half, p_delta_l1)  # UB -> L1

                # GEMM2: dsT = V @ dO^T (transpose_B=True, init=True)
                T.gemm_v0(v_l1, do_l1, l0c_dst, transpose_B=True, init=True)
                T.copy(l0c_dst, dst_ub)  # L0C -> UB

                # GEMM3: dV (1-level CompGEMM)
                T.gemm_v0(p_l1, do_l1, l0c_dv, init=(i == loop_st))
                T.gemm_v0(p_delta_l1, do_l1, l0c_dv, init=False)

                # Delta
                T.tile.mul(delta_2d, qkt_ub, dst_ub)  # P * dsT
                T.reduce_sum(delta_2d, delta_ub, dim=0)  # sum over BS

                # dS = P * (dsT - Delta) * scale
                T.tile.broadcast(delta_2d, delta_ub, axis=0)  # [G] -> [BS, G]
                T.tile.sub(dst_ub, dst_ub, delta_2d)  # dsT - Delta
                T.tile.mul(qkt_ub, qkt_ub, dst_ub)  # P * (dsT - Delta)
                T.tile.mul(qkt_ub, qkt_ub, sm_scale)  # qkt_ub = dS_fp32

                # dS_fp16 + ds_delta + ds_delta2 (2-level CompGEMM)
                T.copy(qkt_ub, ds_half)  # fp32 -> fp16
                T.copy(ds_half, ds_l1)  # UB -> L1
                T.copy(ds_half, ds_rec_ub)  # fp16 -> fp32
                # Level 1: ds_delta = dS_fp32 - cast(dS_fp16)
                T.tile.sub(ds_delta_ub, qkt_ub, ds_rec_ub)
                T.copy(ds_delta_ub, ds_delta_half)  # fp32 -> fp16
                T.copy(ds_delta_half, ds_delta_l1)  # UB -> L1
                T.copy(ds_delta_half, ds_rec_ub)  # fp16 -> fp32
                # Level 2: ds_delta2 = ds_delta - cast(ds_delta_fp16)
                T.tile.sub(ds_delta2_ub, ds_delta_ub, ds_rec_ub)
                T.copy(ds_delta2_ub, ds_delta2_half)  # fp32 -> fp16
                T.copy(ds_delta2_half, ds_delta2_l1)  # UB -> L1

                # GEMM4: dK (2-level CompGEMM, accumulate, kL0Size=32)
                T.gemm_v0(ds_l1, q_l1, l0c_dk, init=(i == loop_st), kL0Size=32)
                T.gemm_v0(ds_delta_l1, q_l1, l0c_dk, init=False, kL0Size=32)
                T.gemm_v0(ds_delta2_l1, q_l1, l0c_dk, init=False, kL0Size=32)

                # GEMM5: dQ (2-level CompGEMM, fresh per token, kL0Size=32)
                T.gemm_v0(ds_l1, k_l1, l0c_dq, transpose_A=True, init=True, kL0Size=32)
                T.gemm_v0(ds_delta_l1, k_l1, l0c_dq, transpose_A=True, init=False, kL0Size=32)
                T.gemm_v0(ds_delta2_l1, k_l1, l0c_dq, transpose_A=True, init=False, kL0Size=32)

                # write dQ (fp32 -> fp16 auto-cast)
                T.copy(l0c_dq, DQ[i_b, i, i_h * G : (i_h + 1) * G, :BK])

            # after loop: write dV, dK (fp32 -> fp16 auto-cast)
            T.copy(l0c_dv, DV[i_b, i_s * BS : (i_s + 1) * BS, i_h, :BV])
            T.copy(l0c_dk, DK[i_b, i_s * BS : (i_s + 1) * BS, i_h, :BK])

    return main


# ============================================================================
# Pipeline runners
# ============================================================================


def _run_nsa_pipeline(q, k, v, do_slc, block_indices, block_counts, B, T, H, HQ, D, S, BS):
    """Run the full NSA pipeline on NPU (fwd + bwd_single).

    Returns (o_slc, lse_slc, dq, dk, dv, block_mask).
    block_mask is for fwd sparse selection (bwd_single uses kernel-internal causal mask).
    Single-kernel bwd: no GM workspace needed (on-chip direct T.copy).

    OP-R20: Input validation — assert seq_len >= 1 to reject empty KV sequences
    (NS=0 would produce zero-block kernel grids, silently returning empty tensors).
    """
    assert T >= 1, "seq_len (T) must be >= 1 (empty KV sequence not supported)"
    assert BS >= 1, "block_size (BS) must be >= 1"
    assert S >= 1, "selected_blocks (S) must be >= 1"
    assert HQ % H == 0, f"HQ ({HQ}) must be a multiple of H ({H})"
    G = HQ // H
    assert G % 2 == 0, f"G ({G}) must be even (fwd kernel uses vid split with hm=G//2)"
    assert G >= 16, f"G ({G}) must be >= 16 (T.tile.compare minimum dim requirement)"
    assert T // BS == 1, (
        f"NS=1 required (T={T} // BS={BS} = {T // BS}), bwd kernel doesn't support NS>1 (Delta computed per single K block)"
    )

    # Step 1: block_mask for fwd (CPU computation — NPU scalar GM read unreliable)
    block_mask_cpu = _compute_block_mask_cpu(block_indices, block_counts, BS)
    block_mask = block_mask_cpu.to("npu")

    # Step 2: fwd
    fwd_mod = nsa_fwd(B, T, HQ, H, D, G, S, BS)
    o_slc, lse_slc = fwd_mod(q, k, v, block_indices)
    torch.npu.synchronize()

    # Step 3: nsa_bwd_single (no workspace — on-chip direct T.copy)
    dq = torch.zeros(B, T, HQ, D, dtype=torch.float16, device="npu")
    dk = torch.zeros(B, T, H, D, dtype=torch.float16, device="npu")
    dv = torch.zeros(B, T, H, D, dtype=torch.float16, device="npu")

    bwd_mod = nsa_bwd_single(B, T, HQ, H, D, G, BS)
    bwd_mod(q, k, v, do_slc, lse_slc, dq, dk, dv)
    torch.npu.synchronize()

    return o_slc, lse_slc, dq, dk, dv, block_mask


def _run_bwd_pipeline(q, k, v, o_slc, lse_slc, do_slc, B, T, H, HQ, D, S, BS):
    """Run only the bwd pipeline (nsa_bwd_single) on NPU.

    Takes fwd outputs (o_slc, lse_slc) as inputs. o_slc is unused by bwd_single
    (kernel recomputes forward internally), but kept for API compatibility.

    OP-R20: Input validation — assert seq_len >= 1 to reject empty KV sequences.
    """
    assert T >= 1, "seq_len (T) must be >= 1 (empty KV sequence not supported)"
    assert BS >= 1, "block_size (BS) must be >= 1"
    assert S >= 1, "selected_blocks (S) must be >= 1"
    assert HQ % H == 0, f"HQ ({HQ}) must be a multiple of H ({H})"
    G = HQ // H
    assert G % 2 == 0, f"G ({G}) must be even (fwd kernel uses vid split with hm=G//2)"
    assert G >= 16, f"G ({G}) must be >= 16 (T.tile.compare minimum dim requirement)"
    assert T // BS == 1, (
        f"NS=1 required (T={T} // BS={BS} = {T // BS}), bwd kernel doesn't support NS>1 (Delta computed per single K block)"
    )

    dq = torch.zeros(B, T, HQ, D, dtype=torch.float16, device="npu")
    dk = torch.zeros(B, T, H, D, dtype=torch.float16, device="npu")
    dv = torch.zeros(B, T, H, D, dtype=torch.float16, device="npu")

    bwd_mod = nsa_bwd_single(B, T, HQ, H, D, G, BS)
    bwd_mod(q, k, v, do_slc, lse_slc, dq, dk, dv)
    torch.npu.synchronize()

    return dq, dk, dv


# ============================================================================
# __main__: CI smoke test (shape + finite check — precision in test file)
# ============================================================================

if __name__ == "__main__":
    import sys

    tilelang.disable_cache()
    torch.manual_seed(0)

    # Minimal L0 smoke config (matches test_tilelang_nsa_bwd l0_default).
    # NOTE: use SEQ_LEN (not T) to avoid shadowing tilelang.language as T.
    B, SEQ_LEN, H, HQ, D, S, BS = 1, 32, 1, 16, 32, 1, 32
    dtype = torch.float16

    q, k, v, do_slc, block_indices, block_counts = _generate_test_data(B, SEQ_LEN, H, HQ, D, S, BS, dtype)

    q_npu = q.to("npu")
    k_npu = k.to("npu")
    v_npu = v.to("npu")
    do_npu = do_slc.to("npu")
    bi_npu = block_indices.to("npu")
    bc_npu = block_counts.to("npu")

    o_slc, lse_slc, dq, dk, dv, _bm = _run_nsa_pipeline(q_npu, k_npu, v_npu, do_npu, bi_npu, bc_npu, B, SEQ_LEN, H, HQ, D, S, BS)

    # CI smoke check: 5 outputs finite + shape OK
    ok = True
    for name, out, expected_shape in [
        ("o_slc", o_slc, (B, SEQ_LEN, HQ, D)),
        ("lse_slc", lse_slc, (B, SEQ_LEN, HQ)),
        ("dq", dq, (B, SEQ_LEN, HQ, D)),
        ("dk", dk, (B, SEQ_LEN, H, D)),
        ("dv", dv, (B, SEQ_LEN, H, D)),
    ]:
        out_cpu = out.cpu()
        shape_ok = tuple(out_cpu.shape) == expected_shape
        finite_ok = torch.isfinite(out_cpu).all().item()
        tag = "OK" if (shape_ok and finite_ok) else "FAIL"
        print(f"[SMOKE_{tag}] {name} shape={tuple(out_cpu.shape)} finite={finite_ok}")
        ok = ok and shape_ok and finite_ok

    if ok:
        print("Test Passed!")
    else:
        sys.exit(1)
