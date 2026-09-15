"""chunk_o_bwd (chunk-based linear-attention O backward) for Ascend NPU.

Computes gradients dq/dk/dw/dg for the chunk-based gated linear attention forward.
Single-kernel on-chip fusion (bhsd-style block-internal chunk loop), Developer mode.

Key design:
  - Single @tilelang.jit, grid=256, chunks_per_block=32, 0 GM workspace.
  - block_DK=64, block_DV=128, kL0Size=32, GM layout [B,H,S,D] (bh-major).
  - T.gemm_v0 (NOT T.gemm); no T.Scope/flag/barrier_all (Developer + combineCV).
  - T.mma explicit L0 staging for GEMM1/2 + Stage-3 GEMMs (shared L0A/L0B).
  - Compensated GEMM (init=True + init=False same L0C) for bf16 precision.
  - Host precompute: dO *= scale, dv = -dv (eliminate in-kernel ops).
  - G_T [B,H,S] transposed input; dg output [NK,B,H,S] (host sum(dim=0) merge).

Pipeline (per chunk):
  Stage 1 (Cube, 5 GEMM): ds=dO@V^T; dq1=dO@h^T; dk1=V@dh^T; dw1=dv@h^T; ds_pos=q@k^T
  Stage 2 (Vector): gate dq/dk/ds; dg reduces; compensated GEMM prep (cast+transpose)
  Stage 3 (Cube, 2+2 compensated GEMM): dq2=ds_gated@k+ds_delta@k; dk2=ds_gated_T@q+ds_delta_T@q
  Stage 4 (Vector): merge + output cast + GM write
"""

import math

import tilelang
import torch
from tilelang import language as T

_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[10, 11, 12, 13], pass_configs=_pass_configs)
def chunk_o_bwd(
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    scale,
    core_num,
    use_g=True,
    use_dw=True,
    block_DK=64,
    block_DV=128,
    chunks_per_block=32,
):
    """Single-kernel on-chip fusion (bhsd-style block-internal chunk loop).

    grid = B * H * NK * num_chunk_groups = 256 (for B=1,S=32768,H=8,DK=128,cpb=32)
    Each block processes chunks_per_block consecutive chunks.
    0 GM workspace, 0 host ATen ops.
    """
    block_S = chunk_size
    # The gate path reuses G_2d_ub as both the col-broadcast target and the
    # last-row select source, requiring a square gate tile (block_DK == block_S).
    if use_g:
        assert block_DK == block_S, "block_DK must equal block_S when use_g=True (G_2d_ub doubles as the col-broadcast target)"
    BS = S // block_S
    NK = math.ceil(DK / block_DK)
    num_chunk_groups = math.ceil(BS / chunks_per_block)
    grid = B * H * NK * num_chunk_groups
    # Batch the per-chunk dg GM write into one DMA per block via a flat UB
    # staging buffer, zero-initialized before the chunk loop and flushed once
    # after it. Gated on staging size (<= 8KB).
    dg_batch_write = use_g and chunks_per_block * block_S * 4 <= 8192
    # Declared unconditionally at prim_func top level (length 1 when off).
    dg_batch_len = chunks_per_block * block_S if dg_batch_write else 1

    @T.prim_func
    def kernel(
        # GM layout [B,H,S,D] (bh-major) for contiguous per-chunk DMA slices.
        # h/dh: [B,H,BS,DK,DV]. G/G_T/dg unchanged.
        Q: T.Tensor((B, H, S, DK), input_dtype),  # type: ignore
        K: T.Tensor((B, H, S, DK), input_dtype),  # type: ignore
        V: T.Tensor((B, H, S, DV), input_dtype),  # type: ignore
        h: T.Tensor((B, H, BS, DK, DV), input_dtype),  # type: ignore
        G: T.Tensor((B, S, H), gate_dtype),  # type: ignore
        G_T: T.Tensor((B, H, S), gate_dtype),  # type: ignore
        dO: T.Tensor((B, H, S, DV), input_dtype),  # type: ignore
        dh: T.Tensor((B, H, BS, DK, DV), input_dtype),  # type: ignore
        dv: T.Tensor((B, H, S, DV), input_dtype),  # type: ignore
        W: T.Tensor((B, H, S, DK), input_dtype),  # type: ignore
        dq: T.Tensor((B, H, S, DK), output_dtype),  # type: ignore
        dk: T.Tensor((B, H, S, DK), output_dtype),  # type: ignore
        dw: T.Tensor((B, H, S, DK), output_dtype),  # type: ignore
        dg: T.Tensor((NK, B, H, S), gate_dtype),  # type: ignore
    ):
        with T.Kernel(grid, threads=1, is_npu=True) as (cid):
            chunk_group = cid % num_chunk_groups
            bk = (cid // num_chunk_groups) % NK
            bh = (cid // num_chunk_groups // NK) % H
            bb = cid // (num_chunk_groups * NK * H)
            d0 = bk * block_DK

            # --- L1 buffers (bf16) ---
            # alloc_L1 for T.mma operands; alloc_shared for gemm_v0 operands.
            v_l1 = T.alloc_shared((block_S, block_DV), input_dtype)
            do_l1 = T.alloc_L1((block_S, block_DV), input_dtype)
            h_l1 = T.alloc_shared((block_DK, block_DV), input_dtype)
            dh_l1 = T.alloc_shared((block_DK, block_DV), input_dtype)
            q_l1 = T.alloc_shared((block_S, block_DK), input_dtype)
            k_l1 = T.alloc_shared((block_S, block_DK), input_dtype)
            dv_l1 = T.alloc_shared((block_S, block_DV), input_dtype)
            ds_gated_l1 = T.alloc_L1((block_S, block_S), output_dtype)
            ds_delta_l1 = T.alloc_L1((block_S, block_S), output_dtype)
            ds_gated_T_l1 = T.alloc_L1((block_S, block_S), output_dtype)
            ds_delta_T_l1 = T.alloc_L1((block_S, block_S), output_dtype)

            # --- L0 buffers for T.mma staging ---
            # GEMM1+GEMM2 share A=dO; GEMM6+GEMM7 share B=k; GEMM8+GEMM9 share B=q.
            do_l0a = T.alloc_L0A((block_S, block_DV), input_dtype)  # [M,K] shared GEMM1+GEMM2
            v_l0b = T.alloc_L0B((block_DV, block_S), input_dtype)  # [K,N] GEMM1 (V^T)
            h_l0b = T.alloc_L0B(
                (block_DV, block_DK), input_dtype
            )  # [K,N] shared GEMM2+GEMM4... (GEMM4 stays gemm_v0, h_l0b for GEMM2 only)

            # Stage-3 L0A (4 operands, [block_S, block_S] bf16 = 8KB each)
            ds_gated_l0a = T.alloc_L0A((block_S, block_S), output_dtype)
            ds_delta_l0a = T.alloc_L0A((block_S, block_S), output_dtype)
            ds_gated_T_l0a = T.alloc_L0A((block_S, block_S), output_dtype)
            ds_delta_T_l0a = T.alloc_L0A((block_S, block_S), output_dtype)
            # Stage-3 L0B (2 shared, [block_S, block_S] bf16 = 8KB each)
            k_l0b = T.alloc_L0B((block_S, block_S), input_dtype)  # shared GEMM6+GEMM7
            q_l0b = T.alloc_L0B((block_S, block_S), input_dtype)  # shared GEMM8+GEMM9

            # --- L0C fragments (fp32) ---
            ds_frag = T.alloc_L0C((block_S, block_S), accum_dtype)
            dq1_frag = T.alloc_L0C((block_S, block_DK), accum_dtype)
            dk1_frag = T.alloc_fragment((block_S, block_DK), accum_dtype)
            dw1_frag = T.alloc_fragment((block_S, block_DK), accum_dtype)
            ds_pos_frag = T.alloc_fragment((block_S, block_S), accum_dtype)
            dq2_frag = T.alloc_L0C((block_S, block_DK), accum_dtype)
            dk2_frag = T.alloc_L0C((block_S, block_DK), accum_dtype)

            # --- UB buffers (fp32) ---
            ds_ub = T.alloc_ub((block_S, block_S), accum_dtype)
            dq_ub = T.alloc_ub((block_S, block_DK), accum_dtype)
            dk_ub = T.alloc_ub((block_S, block_DK), accum_dtype)
            dw_ub = T.alloc_ub((block_S, block_DK), accum_dtype)
            q_ub = T.alloc_ub((block_S, block_DK), accum_dtype)
            k_ub = T.alloc_ub((block_S, block_DK), accum_dtype)
            q_ub_bf16 = T.alloc_shared((block_S, block_DK), input_dtype)
            k_ub_bf16 = T.alloc_shared((block_S, block_DK), input_dtype)
            dq_gemm_ub = T.alloc_ub((block_S, block_DK), accum_dtype)
            dk_gemm_ub = T.alloc_ub((block_S, block_DK), accum_dtype)
            ds_pos_ub = T.alloc_ub((block_S, block_S), accum_dtype)

            # --- Gate UB buffers (fp32) ---
            G_1d_ub = T.alloc_ub((block_S,), gate_dtype)
            G_2d_ub = T.alloc_ub((block_S, block_DK), gate_dtype)
            exp_G_ub = T.alloc_ub((block_S, block_DK), gate_dtype)
            G_last_2d = T.alloc_ub((block_S, block_DK), gate_dtype)
            diff_2d = T.alloc_ub((block_S, block_DK), gate_dtype)
            mask_2d = T.alloc_ub((block_S, block_DK), gate_dtype)
            exp_diff_2d = T.alloc_ub((block_S, block_DK), gate_dtype)
            dg_reduce_tmp = T.alloc_ub((block_S, block_DK), gate_dtype)
            dg_1 = T.alloc_ub((block_S,), gate_dtype)
            dg_2 = T.alloc_ub((block_S,), gate_dtype)
            dg_final = T.alloc_ub((block_S,), gate_dtype)
            dg_last_0_1d = T.alloc_ub((block_S,), gate_dtype)
            dg_last_1_1d = T.alloc_ub((block_S,), gate_dtype)
            # 1D scratch for dg_last_0 = g_last * exp(g_last). Shape (block_S,)
            # matches dg_last_0_1d (block_DK == block_S asserted above).
            exp_g_last_1d = T.alloc_ub((block_S,), gate_dtype)
            # Positive dk-contribution row sums (negated later via 1D sub in dg_final)
            dg_from_dk_1d = T.alloc_ub((block_S,), gate_dtype)
            # Flat dg batch staging buffer (1D for correct DMA behavior).
            dg_batch_ub = T.alloc_ub((dg_batch_len,), gate_dtype)

            # --- Gate 2D UB buffers (fp32) ---
            # G_diff_2d is reused for prologue staging (dead until the first
            # chunk's ds-gate sub overwrites it).
            G_row_2d = T.alloc_ub((block_S, block_S), gate_dtype)
            G_diff_2d = T.alloc_ub((block_S, block_S), gate_dtype)
            G_diff_exp_2d = T.alloc_ub((block_S, block_S), gate_dtype)
            G_diff_mask_2d = T.alloc_ub((block_S, block_S), gate_dtype)
            # Loop-invariant last-row mask, written once before the chunk loop.
            # Dedicated buffer (mask_2d is overwritten per chunk by the dk-gate compare).
            last_row_mask = T.alloc_ub((block_S, block_S), gate_dtype)

            # --- Compensated prep buffers (bf16 + fp32) ---
            ds_gated_bf16 = T.alloc_ub((block_S, block_S), output_dtype)
            ds_delta_ub = T.alloc_ub((block_S, block_S), accum_dtype)
            ds_delta_bf16 = T.alloc_ub((block_S, block_S), output_dtype)
            ds_gated_T_bf16 = T.alloc_ub((block_S, block_S), output_dtype)
            ds_delta_T_bf16 = T.alloc_ub((block_S, block_S), output_dtype)

            # --- Output cast buffers (bf16) ---
            dq_out_bf16 = T.alloc_ub((block_S, block_DK), output_dtype)
            dk_out_bf16 = T.alloc_ub((block_S, block_DK), output_dtype)
            dw_ub_bf16 = T.alloc_ub((block_S, block_DK), output_dtype)

            # --- Tril mask buffers (fp32, use_g=False) ---
            row_1d = T.alloc_ub((block_S,), accum_dtype)
            col_1d = T.alloc_ub((block_S,), accum_dtype)
            row_2d = T.alloc_ub((block_S, block_S), accum_dtype)
            col_2d = T.alloc_ub((block_S, block_S), accum_dtype)
            tril_mask = T.alloc_ub((block_S, block_S), accum_dtype)

            # --- Loop-invariant precomputation (hoisted before chunk loop) ---
            # use_g=False: tril mask (i >= j), computed once (not overwritten in else branch).
            # use_g=True: last-row mask for dg_last_0 (i == block_S-1), staged
            # through G_diff_2d (reused later by the ds-gate sub). dg_2 is
            # dual-use: prologue constant consumed before the first per-chunk
            # reduce_sum overwrites it.
            if not use_g:
                T.tile.arith_progression(row_1d, 0, 1, block_S)
                T.tile.arith_progression(col_1d, 0, 1, block_S)
                T.tile.broadcast(row_2d, row_1d, axis=1)
                T.tile.broadcast(col_2d, col_1d, axis=0)
                T.tile.compare(tril_mask, row_2d, col_2d, "GE")
            # Zero-init the batch buffer: invalid-chunk rows hold 0 (matches
            # golden's torch.zeros semantics), and the fill extends liveness
            # to the prologue for correct UB placement.
            if dg_batch_write:
                T.tile.fill(dg_batch_ub, 0.0)
            if use_g:
                T.tile.arith_progression(dg_2, 0, 1, block_S)
                # Stage the i-index broadcast through G_diff_2d (dead until
                # the first chunk's ds-gate sub overwrites it).
                T.tile.broadcast(G_diff_2d, dg_2, axis=1)
                T.tile.compare(last_row_mask, G_diff_2d, float(block_S - 1), "EQ")

            # --- Block-internal chunk loop (bhsd style) ---
            for c in T.serial(chunks_per_block):
                bs = chunk_group * chunks_per_block + c
                if bs < BS:
                    s0 = bs * block_S

                    # Stage 1 (Cube, 5 GEMM)
                    # All GM->L1 loads issued back-to-back before the GEMM chain
                    # for maximum MTE2 overlap.
                    T.copy(V[bb, bh, s0 : s0 + block_S, :], v_l1)
                    T.copy(dO[bb, bh, s0 : s0 + block_S, :], do_l1)
                    T.copy(h[bb, bh, bs, d0 : d0 + block_DK, :], h_l1)
                    T.copy(dh[bb, bh, bs, d0 : d0 + block_DK, :], dh_l1)
                    T.copy(dv[bb, bh, s0 : s0 + block_S, :], dv_l1)
                    T.copy(Q[bb, bh, s0 : s0 + block_S, d0 : d0 + block_DK], q_l1)
                    T.copy(K[bb, bh, s0 : s0 + block_S, d0 : d0 + block_DK], k_l1)
                    # GEMM1 (ds = dO @ V^T) via explicit L0 staging + T.mma.
                    T.copy(do_l1, do_l0a)
                    T.copy(v_l1, v_l0b, transpose=True)
                    T.mma(do_l0a, v_l0b, ds_frag, init=True)
                    # GEMM2 (dq1 = dO @ h^T) reuses do_l0a (shared A=dO with GEMM1).
                    T.copy(h_l1, h_l0b, transpose=True)
                    T.mma(do_l0a, h_l0b, dq1_frag, init=True)
                    T.gemm_v0(v_l1, dh_l1, dk1_frag, transpose_B=True, init=True, kL0Size=32)
                    # Always run dw1 + ds_pos GEMMs (even when use_dw/use_g=False)
                    # to keep C/V workspace balanced.
                    T.gemm_v0(dv_l1, h_l1, dw1_frag, transpose_B=True, init=True, kL0Size=32)
                    T.gemm_v0(q_l1, k_l1, ds_pos_frag, transpose_B=True, init=True, kL0Size=32)

                    # C->V on-chip direct (L0C->UB)
                    T.copy(ds_frag, ds_ub)
                    T.copy(dq1_frag, dq_ub)
                    T.copy(dk1_frag, dk_ub)
                    T.copy(ds_pos_frag, ds_pos_ub)
                    T.copy(dw1_frag, dw_ub)

                    # Stage 2 (Vector): gate dq/dk/ds; dg reduces; compensated GEMM prep
                    if use_dw:
                        T.copy(dw_ub, dw_ub_bf16)
                    else:
                        T.tile.mul(dw_ub, dw_ub, 0.0)
                        T.copy(dw_ub, dw_ub_bf16)

                    if use_g:
                        # Two-step load: GM bf16 -> UB bf16 -> UB fp32 (avoids L1->UB cast issue).
                        # All 3 GM->UB loads (G_T, Q, K) issue back-to-back before the first Vector op.
                        T.copy(G_T[bb, bh, s0 : s0 + block_S], G_1d_ub)
                        T.copy(Q[bb, bh, s0 : s0 + block_S, d0 : d0 + block_DK], q_ub_bf16)
                        T.copy(q_ub_bf16, q_ub)
                        T.copy(K[bb, bh, s0 : s0 + block_S, d0 : d0 + block_DK], k_ub_bf16)
                        T.copy(k_ub_bf16, k_ub)

                        # dq gate: dq *= exp(G) (scale is pre-multiplied into dO on the host)
                        T.tile.broadcast(G_2d_ub, G_1d_ub, axis=1)
                        T.tile.exp(exp_G_ub, G_2d_ub)
                        T.tile.mul(dq_ub, dq_ub, exp_G_ub)

                        # dg_last_0 = g_last * exp(g_last) (last-row mask hoisted before chunk loop)
                        T.tile.select(diff_2d, last_row_mask, G_2d_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.reduce_sum(diff_2d, dg_last_0_1d, dim=0, clear=True)
                        # G_last_2d broadcast MUST stay: consumed by the dk-gate
                        # sub below (diff = g_last - G)
                        T.tile.broadcast(G_last_2d, dg_last_0_1d, axis=0)
                        # dg_last_0_1d is already [g_last]*block_DK (all elements
                        # identical after the dim=0 reduce), so the 2D path
                        # collapses to 1D ops. Math: g_last * exp(g_last).
                        T.tile.exp(exp_g_last_1d, dg_last_0_1d)
                        T.tile.mul(dg_last_0_1d, dg_last_0_1d, exp_g_last_1d)

                        # dg_from_dq = sum(dq_gated * q, dim=-1)
                        T.tile.mul(dg_reduce_tmp, dq_ub, q_ub)
                        T.reduce_sum(dg_reduce_tmp, dg_1, dim=-1, clear=True)

                        # dk gate factors (for dg computation only -- dk_ub stays ungated for output).
                        # Golden: dk = dk1_ungated + ds_g^T @ q; dk_gated only used for dg.
                        # Direct tensor-tensor compare: (g_last - G[i]) <= 0 is
                        # elementwise-equivalent to g_last <= G[i] for finite fp32.
                        T.tile.compare(mask_2d, G_last_2d, G_2d_ub, "LE")
                        T.tile.sub(diff_2d, G_last_2d, G_2d_ub)
                        T.tile.exp(exp_diff_2d, diff_2d)

                        # dg_from_dk + dg_last_1: compute dk1*exp(diff)*mask*k once
                        T.tile.mul(dg_reduce_tmp, dk_ub, exp_diff_2d)
                        T.tile.mul(dg_reduce_tmp, dg_reduce_tmp, k_ub)
                        T.tile.select(dg_reduce_tmp, mask_2d, dg_reduce_tmp, 0.0, "VSEL_TENSOR_SCALAR_MODE")

                        # One reduce serves both consumers: row sums land directly
                        # in dg_from_dk_1d, and dg_last_1 reads from it via broadcast.
                        T.reduce_sum(dg_reduce_tmp, dg_from_dk_1d, dim=-1, clear=True)

                        # dg_last_1 = grand total: broadcast row sums, then reduce dim=1.
                        T.tile.broadcast(diff_2d, dg_from_dk_1d, axis=0)
                        T.reduce_sum(diff_2d, dg_last_1_1d, dim=1, clear=True)

                        # dg_from_dk kept positive; negation folded into dg_final as a 1D sub.

                        # ds gate: ds *= exp(Gi - Gj) * mask (scale folded into dO on host)
                        # G_2d_ub already holds broadcast(G_1d_ub, axis=1) from the
                        # dq gate above -- reused here (requires block_DK == block_S).
                        # Direct tensor-tensor compare: (G[i]-G[j]) <= 0 is
                        # elementwise-equivalent to G[i] <= G[j] for finite fp32.
                        T.tile.broadcast(G_row_2d, G_1d_ub, axis=0)
                        T.tile.compare(G_diff_mask_2d, G_2d_ub, G_row_2d, "LE")
                        T.tile.sub(G_diff_2d, G_2d_ub, G_row_2d)
                        T.tile.exp(G_diff_exp_2d, G_diff_2d)
                        T.tile.mul(ds_ub, ds_ub, G_diff_exp_2d)
                        T.tile.select(ds_ub, G_diff_mask_2d, ds_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")

                        # ds_pos = ds_gated * (q @ k^T)
                        T.tile.mul(ds_pos_ub, ds_ub, ds_pos_ub)

                        # dg1 = sum(ds_pos, dim=1); dg2 = sum(ds_pos, dim=0)
                        T.reduce_sum(ds_pos_ub, dg_1, dim=1, clear=False)
                        # dg2 via reduce dim=0 (sum over rows)
                        T.reduce_sum(ds_pos_ub, dg_2, dim=0, clear=True)

                        # dg_final = dg_1 - dg_2 + dg_last_0 + dg_last_1 - dg_from_dk_pos
                        if dg_batch_write:
                            # Compute directly into the batch buffer's chunk-c row.
                            b0 = c * block_S
                            T.tile.sub(dg_batch_ub[b0 : b0 + block_S], dg_1, dg_2)
                            T.tile.add(
                                dg_batch_ub[b0 : b0 + block_S],
                                dg_batch_ub[b0 : b0 + block_S],
                                dg_last_0_1d,
                            )
                            T.tile.add(
                                dg_batch_ub[b0 : b0 + block_S],
                                dg_batch_ub[b0 : b0 + block_S],
                                dg_last_1_1d,
                            )
                            T.tile.sub(
                                dg_batch_ub[b0 : b0 + block_S],
                                dg_batch_ub[b0 : b0 + block_S],
                                dg_from_dk_1d,
                            )
                        else:
                            T.tile.sub(dg_final, dg_1, dg_2)
                            T.tile.add(dg_final, dg_final, dg_last_0_1d)
                            T.tile.add(dg_final, dg_final, dg_last_1_1d)
                            # Subtract the positive dk contribution (1D)
                            T.tile.sub(dg_final, dg_final, dg_from_dk_1d)
                            T.copy(dg_final, dg[bk, bb, bh, s0 : s0 + block_S])

                        # Compensated GEMM prep: cast fp32->bf16 and compute delta.
                        # ds_gated_bf16 = bf16(ds_ub); ds_delta = ds_ub - fp32(ds_gated_bf16)
                        T.copy(ds_ub, ds_gated_bf16)
                        T.copy(ds_gated_bf16, ds_delta_ub)
                        T.tile.sub(ds_delta_ub, ds_ub, ds_delta_ub)
                        T.copy(ds_delta_ub, ds_delta_bf16)
                        # Transpose fp32 then cast to bf16.
                        T.tile.transpose(ds_pos_ub, ds_ub)
                        T.copy(ds_pos_ub, ds_gated_T_bf16)
                        T.tile.transpose(ds_pos_ub, ds_delta_ub)
                        T.copy(ds_pos_ub, ds_delta_T_bf16)

                        # V->C on-chip direct (UB->L1)
                        T.copy(ds_gated_bf16, ds_gated_l1)
                        T.copy(ds_delta_bf16, ds_delta_l1)
                        T.copy(ds_gated_T_bf16, ds_gated_T_l1)
                        T.copy(ds_delta_T_bf16, ds_delta_T_l1)

                        # Stage 3 (Cube, 2+2 compensated GEMM)
                        # Reordered: dq2 group -> copy -> dk2 group -> copy,
                        # so dq2_frag and dk2_frag have non-overlapping
                        # lifetimes -> memory planner can alias them.
                        # GEMM6/7: dq2 = ds_gated@K + ds_delta@K
                        T.copy(ds_gated_l1, ds_gated_l0a)
                        T.copy(k_l1, k_l0b)
                        T.mma(ds_gated_l0a, k_l0b, dq2_frag, init=True)
                        T.copy(ds_delta_l1, ds_delta_l0a)
                        T.mma(ds_delta_l0a, k_l0b, dq2_frag, init=False)
                        T.copy(dq2_frag, dq_gemm_ub)
                        # GEMM8/9: dk2 = ds_gated_T@Q + ds_delta_T@Q
                        T.copy(ds_gated_T_l1, ds_gated_T_l0a)
                        T.copy(q_l1, q_l0b)
                        T.mma(ds_gated_T_l0a, q_l0b, dk2_frag, init=True)
                        T.copy(ds_delta_T_l1, ds_delta_T_l0a)
                        T.mma(ds_delta_T_l0a, q_l0b, dk2_frag, init=False)
                        T.copy(dk2_frag, dk_gemm_ub)

                        # Stage 4 (merge + output)
                        # cast-cast-write-write order lets MTE3 GM writes overlap
                        # the next chunk's Stage 1.
                        T.tile.add(dq_ub, dq_ub, dq_gemm_ub)
                        T.tile.add(dk_ub, dk_ub, dk_gemm_ub)
                        T.copy(dq_ub, dq_out_bf16)
                        T.copy(dk_ub, dk_out_bf16)
                        T.copy(dq_out_bf16, dq[bb, bh, s0 : s0 + block_S, d0 : d0 + block_DK])
                        T.copy(dk_out_bf16, dk[bb, bh, s0 : s0 + block_S, d0 : d0 + block_DK])
                        # Deferred dw GM write (see Stage 2)
                        T.copy(dw_ub_bf16, dw[bb, bh, s0 : s0 + block_S, d0 : d0 + block_DK])

                    else:
                        # use_g=False: tril mask path (no gating).
                        # Consume ds_pos_ub C->V transfer with mul-by-0 (read+write).
                        T.tile.mul(ds_pos_ub, ds_pos_ub, 0.0)
                        T.tile.select(ds_ub, tril_mask, ds_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")

                        # Compensated GEMM prep (same as use_g=True branch)
                        T.copy(ds_ub, ds_gated_bf16)
                        T.copy(ds_gated_bf16, ds_delta_ub)
                        T.tile.sub(ds_delta_ub, ds_ub, ds_delta_ub)
                        T.copy(ds_delta_ub, ds_delta_bf16)
                        T.tile.transpose(ds_pos_ub, ds_ub)
                        T.copy(ds_pos_ub, ds_gated_T_bf16)
                        T.tile.transpose(ds_pos_ub, ds_delta_ub)
                        T.copy(ds_pos_ub, ds_delta_T_bf16)

                        # V->C
                        T.copy(ds_gated_bf16, ds_gated_l1)
                        T.copy(ds_delta_bf16, ds_delta_l1)
                        T.copy(ds_gated_T_bf16, ds_gated_T_l1)
                        T.copy(ds_delta_T_bf16, ds_delta_T_l1)

                        # Stage 3 compensated GEMM (reordered for L0C aliasing)
                        T.copy(ds_gated_l1, ds_gated_l0a)
                        T.copy(k_l1, k_l0b)
                        T.mma(ds_gated_l0a, k_l0b, dq2_frag, init=True)
                        T.copy(ds_delta_l1, ds_delta_l0a)
                        T.mma(ds_delta_l0a, k_l0b, dq2_frag, init=False)
                        T.copy(dq2_frag, dq_gemm_ub)
                        T.copy(ds_gated_T_l1, ds_gated_T_l0a)
                        T.copy(q_l1, q_l0b)
                        T.mma(ds_gated_T_l0a, q_l0b, dk2_frag, init=True)
                        T.copy(ds_delta_T_l1, ds_delta_T_l0a)
                        T.mma(ds_delta_T_l0a, q_l0b, dk2_frag, init=False)
                        T.copy(dk2_frag, dk_gemm_ub)

                        # merge + output (dq = dq1 + dq2; dk = dk1 + dk2)
                        T.tile.add(dq_ub, dq_ub, dq_gemm_ub)
                        T.tile.add(dk_ub, dk_ub, dk_gemm_ub)
                        T.copy(dq_ub, dq_out_bf16)
                        T.copy(dk_ub, dk_out_bf16)
                        T.copy(dq_out_bf16, dq[bb, bh, s0 : s0 + block_S, d0 : d0 + block_DK])
                        T.copy(dk_out_bf16, dk[bb, bh, s0 : s0 + block_S, d0 : d0 + block_DK])
                        # Deferred dw GM write (see Stage 2)
                        T.copy(dw_ub_bf16, dw[bb, bh, s0 : s0 + block_S, d0 : d0 + block_DK])

            # Flush dg batch buffer to GM (one DMA per block)
            if dg_batch_write:
                s_start = chunk_group * chunks_per_block * block_S
                T.copy(
                    dg_batch_ub,
                    dg[bk, bb, bh, s_start : s_start + chunks_per_block * block_S],
                )

    return kernel


# ============================================================================
# Input preparation (shared by smoke test and test file)
# ============================================================================


def _prepare_inputs(B, S, H, DK, DV, chunk_size, device="npu"):
    """Prepare inputs for chunk_o_bwd kernel.

    Layout: [B,H,S,D] (bh-major) so per-chunk slices are contiguous.
    h/dh: [B,H,BS,DK,DV]. G stays [B,S,H].

    Host precompute (folded into inputs to reduce in-kernel ops):
    - dO *= scale (fp32 mul + bf16 cast)
    - dv = -dv (pre-negate to eliminate in-kernel dw negate)
    """
    BS = S // chunk_size
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, DK, dtype=torch.bfloat16, device=device)
    K = torch.randn(B, H, S, DK, dtype=torch.bfloat16, device=device)
    V = torch.randn(B, H, S, DV, dtype=torch.bfloat16, device=device)
    h_t = torch.randn(B, H, BS, DK, DV, dtype=torch.bfloat16, device=device)
    G = torch.randn(B, S, H, dtype=torch.float32, device=device)
    dO = torch.randn(B, H, S, DV, dtype=torch.bfloat16, device=device)
    scale = DK**-0.5
    dO = (dO.float() * scale).bfloat16()
    dh = torch.randn(B, H, BS, DK, DV, dtype=torch.bfloat16, device=device)
    dv = torch.randn(B, H, S, DV, dtype=torch.bfloat16, device=device)
    dv = -dv
    W = torch.randn(B, H, S, DK, dtype=torch.bfloat16, device=device)
    G_T = G.cpu().permute(0, 2, 1).contiguous().to(device)
    return Q, K, V, h_t, G, G_T, dO, dh, dv, W


# ============================================================================
# Smoke test (verify kernel runs + output shapes correct)
# Golden comparison lives in test_chunk_o_bwd.py.
# ============================================================================


if __name__ == "__main__":
    tilelang.disable_cache()
    torch.set_default_device("npu")
    B, S, H, DK, DV, cs = 1, 512, 8, 128, 128, 64
    scale = DK**-0.5
    core_num = int(torch.npu.get_device_properties("npu").cube_core_num)
    Q, K, V, h_t, G, G_T, dO, dh, dv, W = _prepare_inputs(B, S, H, DK, DV, cs)
    kernel = chunk_o_bwd(
        B,
        S,
        H,
        DK,
        DV,
        "bfloat16",
        "bfloat16",
        "float32",
        "float32",
        "float32",
        cs,
        scale,
        core_num,
        True,
        True,
        64,
        128,
    )
    dq_o, dk_o, dw_o, dg_o = kernel(Q, K, V, h_t, G, G_T, dO, dh, dv, W)
    torch.npu.synchronize()
    # Verify output shapes only (golden comparison in test_chunk_o_bwd.py)
    assert dq_o.shape == (B, H, S, DK), f"dq shape mismatch: {dq_o.shape}"
    assert dk_o.shape == (B, H, S, DK), f"dk shape mismatch: {dk_o.shape}"
    assert dw_o.shape == (B, H, S, DK), f"dw shape mismatch: {dw_o.shape}"
    NK = math.ceil(DK / 64)
    assert dg_o.shape == (NK, B, H, S), f"dg shape mismatch: {dg_o.shape}"
    assert not torch.isnan(dq_o).any(), "dq NaN"
    assert not torch.isnan(dk_o).any(), "dk NaN"
    print(f"Shapes: dq={dq_o.shape}, dk={dk_o.shape}, dw={dw_o.shape}, dg={dg_o.shape}")
    print("Test Passed!")
