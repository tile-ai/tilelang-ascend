"""
Block Sparse MQA Attention Kernel (Expert Mode, A5 CV通路 / no-workspace)

A5-only variant of block_sparse_mqa_attn_expert_test.py.

The base kernel routes each [H, kv] score tile through a GM workspace:
    C:  MMA -> L0C -> T.copy(l0c, workspace_1[token, n_i])   (L0C -> GM)
    V:  T.copy(workspace_1[token, n_i], s_ub column)         (GM  -> UB)

This version uses the A5-exclusive CV通路 (L0C -> UB direct via TMOV,
`T.copy_op.copy_cv_experiment`) to hand the score tile straight from the
cube's L0C to the vector core's UB, eliminating the workspace round-trip.

Key differences vs. the base kernel (everything else is kept identical):
  1. `workspace_1` argument removed (and `workspace_idx`); no GM staging.
  2. C scope: each `T.copy(l0c, workspace_1[...])` becomes
     `copy_cv_experiment(l0c, s_ub_j, SingleVec{0|1})`.
       - token_a -> vector core 0 (SingleVec0), token_b -> core 1 (SingleVec1),
         matching `token_idx = bx*ntpk + pair_i*2 + by`.
       - The block index within the group (0..3) selects s_ub_0..3.
       - The surrounding M<->FIX (SIG_L0C) flags are unchanged: the CV copy is
         a FIX-pipe op reading L0C, exactly like the old L0C->GM copy.
  3. Because the 4 per-block UB score buffers are *reused* across n_outer
     (unlike the uniquely-addressed GM workspace), a V->C "buffer-free" credit
     (CROSS_V2C) is added so C never overwrites a tile V has not yet consumed.
  4. V scope: per-block relu / row_expand_mul / reduce_sum into logits_4x
     columns (the merged [H, 4*kv] buffer can't be filled by CV copies, since
     TMOV writes a slice as contiguous). Masking/output are unchanged.

Flag conventions (C-scope pipe flags identical to base):
  - MTE1<->MTE2 for L1 buffer management
  - M<->MTE1 for L0A/L0B management
  - M<->FIX for L0C management
  - V<->MTE2 for UB (weights) buffer management
  - V<->MTE3 for output buffer management
  - Cross: FIX->V (CROSS_C2V) data-ready, V->FIX (CROSS_V2C) buffer-free
"""

import tilelang
from tilelang import language as T
import argparse
import torch
import sys

tilelang.disable_cache()


@tilelang.jit(
    out_idx=[3],
    target="pto",
)
def block_sparse_mqa_attn_return_logits(
    seq_len: int,
    seq_len_kv: int,
    kv_block_size: int,
    topk: int,
    heads: int,
    index_dim: int,
    block_N: int = 8,
    num_stages: int = 2,  # noqa: ARG001
    threads: int = 2,  # noqa: ARG001
    grid_size: int = 24,
):
    dtype = "float16"
    accum_dtype = "float32"
    index_dtype = "int32"

    # grid_size is the controllable input. num_pairs is derived from it:
    # each block must cover ceildiv(seq_len, grid_size) tokens, rounded up to
    # whole pairs so that num_tokens_per_kernel == 2*num_pairs. Any tokens
    # provisioned beyond seq_len are guarded by the `if token_* < seq_len` checks
    # (and by the per-core num_pairs_bx loop bound, see kernel body).
    tokens_per_block = (seq_len + grid_size - 1) // grid_size
    num_pairs = (tokens_per_block + 1) // 2
    num_tokens_per_kernel = 2 * num_pairs

    index_q_shape = [seq_len, heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    logits_shape = [seq_len, topk, kv_block_size]

    H_per_block = heads
    assert kv_block_size % block_N == 0, "block_N must divide kv_block_size"
    assert topk % 4 == 0, "topk must be divisible by 4 for 4x unrolled loop"
    topk_groups = topk // 4

    # ---------- Signal IDs ----------
    # C scope: MTE1↔MTE2
    SIG_Q_L1 = 0  # Q L1 buffer (single)
    SIG_K_L1_0 = 1  # K L1 buffer ping
    SIG_K_L1_1 = 2  # K L1 buffer pong
    # C scope: M↔MTE1
    SIG_L0AB_0 = 0  # L0A/L0B ping
    SIG_L0AB_1 = 1  # L0A/L0B pong
    # C scope: FIX↔M
    SIG_L0C_0 = 0  # L0C ping
    SIG_L0C_1 = 1  # L0C pong
    # V scope
    SIG_W_UB = 1  # weights_ub: V↔MTE2
    SIG_LOGITS = 0  # logits: V↔MTE3
    # Cross-scope
    CROSS_C2V_0 = 0  # C→V data-ready for even n_outer
    CROSS_C2V_1 = 1  # C→V data-ready for odd n_outer
    CROSS_V2C = 2  # V→C buffer-free credit (s_ub_0..3 consumed)

    # CV通路 destination-core mode (plain int; must live outside prim_func body,
    # otherwise TVMScript binds it to a Var and int(mode) fails).
    V0 = int(T.copy_op.CopyCVMode.SingleVec0)  # token_a → vector core 0
    V1 = int(T.copy_op.CopyCVMode.SingleVec1)  # token_b → vector core 1

    @T.prim_func
    def kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([seq_len, topk], index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(grid_size, is_npu=True) as (bx, by):
            # Per-core valid pair count. Compile-time num_pairs is the upper
            # bound (buffer alloc / unroll); each core loops only over the pairs
            # it actually owns. This is what makes an arbitrary grid_size safe:
            # the tail core (fewer valid tokens) runs fewer iterations instead of
            # entering fully-empty pairs. A fully-empty pair would still execute
            # the unconditional cross-flag handshake (CROSS_C2V / CROSS_V2C) with
            # no buffer work to pace it, desyncing the intra-block cube<->vec
            # handshake -> device hang. Both C and V loops MUST use this same
            # expression so the C2V/V2C credit ping-pong stays balanced per core.
            num_pairs_bx = T.min(num_pairs, T.ceildiv(seq_len - bx * num_tokens_per_kernel, 2))
            # ---- V scope: UB allocations ----
            # 4 per-block score buffers (CV通路 destinations). Each block's
            # [H, kv] L0C tile is copied here directly by the cube. Total UB
            # footprint == the base kernel's single s_ub_4x [H, 4*kv].
            s_ub_0 = T.alloc_ub([H_per_block, kv_block_size], accum_dtype)
            s_ub_1 = T.alloc_ub([H_per_block, kv_block_size], accum_dtype)
            s_ub_2 = T.alloc_ub([H_per_block, kv_block_size], accum_dtype)
            s_ub_3 = T.alloc_ub([H_per_block, kv_block_size], accum_dtype)
            logits_4x = T.alloc_ub([1, kv_block_size * 4], accum_dtype)
            # Per-block reduce temporaries (reduce_sum needs a full-buffer out;
            # a sliced BufferRegion out is not accepted, so reduce here then
            # T.copy into the logits_4x columns).
            logits_t0 = T.alloc_ub([1, kv_block_size], accum_dtype)
            logits_t1 = T.alloc_ub([1, kv_block_size], accum_dtype)
            logits_t2 = T.alloc_ub([1, kv_block_size], accum_dtype)
            logits_t3 = T.alloc_ub([1, kv_block_size], accum_dtype)
            weights_ub = T.alloc_ub([heads], dtype)
            weights = T.alloc_ub([heads], accum_dtype)
            # Mask buffers (4 blocks merged → [4*kv//8] = 64 uint8 ≥ 32)
            kvpi_a = T.alloc_ub([kv_block_size], "int32")
            kvpi_b = T.alloc_ub([kv_block_size], "int32")
            kvpi_c = T.alloc_ub([kv_block_size], "int32")
            kvpi_d = T.alloc_ub([kv_block_size], "int32")
            kvpf_4x = T.alloc_ub([kv_block_size * 4], "float")
            mask1_ub = T.alloc_ub([kv_block_size * 4 // 8], "uint8")
            mask2_ub = T.alloc_ub([kv_block_size * 4 // 8], "uint8")

            # ---- C scope: L1 / L0A / L0B / L0C allocations ----
            q_l1 = T.alloc_L1([H_per_block, index_dim], dtype)
            # Double-buffered K L1
            k_l1_0 = T.alloc_L1([kv_block_size, index_dim], dtype)
            k_l1_1 = T.alloc_L1([kv_block_size, index_dim], dtype)
            # Double-buffered L0A (token_a uses l0a_0, token_b uses l0a_1)
            l0a_0 = T.alloc_L0A([H_per_block, index_dim], dtype)
            l0a_1 = T.alloc_L0A([H_per_block, index_dim], dtype)
            # Double-buffered L0B
            l0b_0 = T.alloc_L0B([index_dim, kv_block_size], dtype)
            l0b_1 = T.alloc_L0B([index_dim, kv_block_size], dtype)
            # Double-buffered L0C
            l0c_0 = T.alloc_L0C([H_per_block, kv_block_size], accum_dtype)
            l0c_1 = T.alloc_L0C([H_per_block, kv_block_size], accum_dtype)

            # ================================================================
            # C scope: double-buffered L0B/L0C/k_l1 4-stage SW pipeline.
            # L0C tiles go straight to V's UB via CV通路 (no GM workspace).
            # ================================================================
            with T.Scope("C"):
                # Init: all buffers start free (ping + pong)
                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                T.set_flag("MTE1", "MTE2", SIG_K_L1_0)
                T.set_flag("MTE1", "MTE2", SIG_K_L1_1)
                T.set_flag("M", "MTE1", SIG_L0AB_0)
                T.set_flag("M", "MTE1", SIG_L0AB_1)
                T.set_flag("FIX", "M", SIG_L0C_0)
                T.set_flag("FIX", "M", SIG_L0C_1)

                for pair_i in T.serial(num_pairs_bx):
                    for n_outer in T.serial(topk_groups):
                        n_i0 = n_outer * 4 + 0
                        n_i1 = n_outer * 4 + 1
                        n_i2 = n_outer * 4 + 2
                        n_i3 = n_outer * 4 + 3

                        # Back-pressure: wait until V has consumed the previous
                        # group's s_ub_0..3 before overwriting them via CV通路.
                        T.wait_cross_flag(CROSS_V2C, "FIX")

                        # ====================================================
                        # token_a → vector core 0 (SingleVec0)
                        # ====================================================
                        t_a = pair_i * 2
                        token_a = bx * num_tokens_per_kernel + t_a
                        if token_a < seq_len:
                            # ---- Wave 0: DMA K[0] → k_l1_0 ----
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1_0)
                            T.copy(
                                IndexK[
                                    TopKBlockIndex[token_a, n_i0] * kv_block_size : TopKBlockIndex[token_a, n_i0] * kv_block_size
                                    + kv_block_size,
                                    :,
                                ],
                                k_l1_0,
                            )
                            T.set_flag("MTE2", "MTE1", SIG_K_L1_0)

                            # ---- Wave 1: DMA K[1]→k_l1_1 | Stage K[0]→l0b_0 ----
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1_1)
                            T.copy(
                                IndexK[
                                    TopKBlockIndex[token_a, n_i1] * kv_block_size : TopKBlockIndex[token_a, n_i1] * kv_block_size
                                    + kv_block_size,
                                    :,
                                ],
                                k_l1_1,
                            )
                            T.set_flag("MTE2", "MTE1", SIG_K_L1_1)

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1_0)
                            T.wait_flag("M", "MTE1", SIG_L0AB_0)
                            T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                            T.copy(IndexQ[token_a, :, :], q_l1)
                            T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                            T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                            T.copy(q_l1, l0a_0)
                            T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                            T.copy(k_l1_0, l0b_0, transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1_0)
                            T.set_flag("MTE1", "M", SIG_L0AB_0)

                            # ---- Wave 2: DMA K[2]→k_l1_0 | Stage K[1]→l0b_1 | MMA K[0]→l0c_0 ----
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1_0)
                            T.copy(
                                IndexK[
                                    TopKBlockIndex[token_a, n_i2] * kv_block_size : TopKBlockIndex[token_a, n_i2] * kv_block_size
                                    + kv_block_size,
                                    :,
                                ],
                                k_l1_0,
                            )
                            T.set_flag("MTE2", "MTE1", SIG_K_L1_0)

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1_1)
                            T.wait_flag("M", "MTE1", SIG_L0AB_1)
                            T.copy(k_l1_1, l0b_1, transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1_1)
                            T.set_flag("MTE1", "M", SIG_L0AB_1)

                            T.wait_flag("MTE1", "M", SIG_L0AB_0)
                            T.wait_flag("FIX", "M", SIG_L0C_0)
                            T.mma(l0a_0, l0b_0, l0c_0, init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB_0)
                            T.set_flag("M", "FIX", SIG_L0C_0)

                            # ---- Wave 3: DMA K[3]→k_l1_1 | Stage K[2]→l0b_0 | MMA K[1]→l0c_1 | CV l0c_0→s_ub_0 ----
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1_1)
                            T.copy(
                                IndexK[
                                    TopKBlockIndex[token_a, n_i3] * kv_block_size : TopKBlockIndex[token_a, n_i3] * kv_block_size
                                    + kv_block_size,
                                    :,
                                ],
                                k_l1_1,
                            )
                            T.set_flag("MTE2", "MTE1", SIG_K_L1_1)

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1_0)
                            T.wait_flag("M", "MTE1", SIG_L0AB_0)
                            T.copy(k_l1_0, l0b_0, transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1_0)
                            T.set_flag("MTE1", "M", SIG_L0AB_0)

                            T.wait_flag("MTE1", "M", SIG_L0AB_1)
                            T.wait_flag("FIX", "M", SIG_L0C_1)
                            T.mma(l0a_0, l0b_1, l0c_1, init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB_1)
                            T.set_flag("M", "FIX", SIG_L0C_1)

                            T.wait_flag("M", "FIX", SIG_L0C_0)
                            T.copy_op.copy_cv_experiment(l0c_0, s_ub_0, V0)
                            T.set_flag("FIX", "M", SIG_L0C_0)

                            # ---- Wave 4: Stage K[3]→l0b_1 | MMA K[2]→l0c_0 | CV l0c_1→s_ub_1 ----
                            T.wait_flag("MTE2", "MTE1", SIG_K_L1_1)
                            T.wait_flag("M", "MTE1", SIG_L0AB_1)
                            T.copy(k_l1_1, l0b_1, transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1_1)
                            T.set_flag("MTE1", "M", SIG_L0AB_1)

                            T.wait_flag("MTE1", "M", SIG_L0AB_0)
                            T.wait_flag("FIX", "M", SIG_L0C_0)
                            T.mma(l0a_0, l0b_0, l0c_0, init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB_0)
                            T.set_flag("M", "FIX", SIG_L0C_0)

                            T.wait_flag("M", "FIX", SIG_L0C_1)
                            T.copy_op.copy_cv_experiment(l0c_1, s_ub_1, V0)
                            T.set_flag("FIX", "M", SIG_L0C_1)

                            # ---- Wave 5: MMA K[3]→l0c_1 | CV l0c_0→s_ub_2 (drain) ----
                            T.wait_flag("MTE1", "M", SIG_L0AB_1)
                            T.wait_flag("FIX", "M", SIG_L0C_1)
                            T.mma(l0a_0, l0b_1, l0c_1, init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB_1)
                            T.set_flag("M", "FIX", SIG_L0C_1)

                            T.wait_flag("M", "FIX", SIG_L0C_0)
                            T.copy_op.copy_cv_experiment(l0c_0, s_ub_2, V0)
                            T.set_flag("FIX", "M", SIG_L0C_0)

                            # ---- Wave 6: CV l0c_1→s_ub_3 (drain) ----
                            T.wait_flag("M", "FIX", SIG_L0C_1)
                            T.copy_op.copy_cv_experiment(l0c_1, s_ub_3, V0)
                            T.set_flag("FIX", "M", SIG_L0C_1)

                        # ====================================================
                        # token_b → vector core 1 (SingleVec1)
                        # ====================================================
                        t_b = pair_i * 2 + 1
                        token_b = bx * num_tokens_per_kernel + t_b
                        if token_b < seq_len:
                            # ---- Wave 0: DMA K[0] → k_l1_0 ----
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1_0)
                            T.copy(
                                IndexK[
                                    TopKBlockIndex[token_b, n_i0] * kv_block_size : TopKBlockIndex[token_b, n_i0] * kv_block_size
                                    + kv_block_size,
                                    :,
                                ],
                                k_l1_0,
                            )
                            T.set_flag("MTE2", "MTE1", SIG_K_L1_0)

                            # ---- Wave 1: DMA K[1]→k_l1_1 | Stage K[0]→l0b_0 ----
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1_1)
                            T.copy(
                                IndexK[
                                    TopKBlockIndex[token_b, n_i1] * kv_block_size : TopKBlockIndex[token_b, n_i1] * kv_block_size
                                    + kv_block_size,
                                    :,
                                ],
                                k_l1_1,
                            )
                            T.set_flag("MTE2", "MTE1", SIG_K_L1_1)

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1_0)
                            T.wait_flag("M", "MTE1", SIG_L0AB_0)
                            T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                            T.copy(IndexQ[token_b, :, :], q_l1)
                            T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                            T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                            T.copy(q_l1, l0a_1)
                            T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                            T.copy(k_l1_0, l0b_0, transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1_0)
                            T.set_flag("MTE1", "M", SIG_L0AB_0)

                            # ---- Wave 2: DMA K[2]→k_l1_0 | Stage K[1]→l0b_1 | MMA K[0]→l0c_0 ----
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1_0)
                            T.copy(
                                IndexK[
                                    TopKBlockIndex[token_b, n_i2] * kv_block_size : TopKBlockIndex[token_b, n_i2] * kv_block_size
                                    + kv_block_size,
                                    :,
                                ],
                                k_l1_0,
                            )
                            T.set_flag("MTE2", "MTE1", SIG_K_L1_0)

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1_1)
                            T.wait_flag("M", "MTE1", SIG_L0AB_1)
                            T.copy(k_l1_1, l0b_1, transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1_1)
                            T.set_flag("MTE1", "M", SIG_L0AB_1)

                            T.wait_flag("MTE1", "M", SIG_L0AB_0)
                            T.wait_flag("FIX", "M", SIG_L0C_0)
                            T.mma(l0a_1, l0b_0, l0c_0, init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB_0)
                            T.set_flag("M", "FIX", SIG_L0C_0)

                            # ---- Wave 3: DMA K[3]→k_l1_1 | Stage K[2]→l0b_0 | MMA K[1]→l0c_1 | CV l0c_0→s_ub_0 ----
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1_1)
                            T.copy(
                                IndexK[
                                    TopKBlockIndex[token_b, n_i3] * kv_block_size : TopKBlockIndex[token_b, n_i3] * kv_block_size
                                    + kv_block_size,
                                    :,
                                ],
                                k_l1_1,
                            )
                            T.set_flag("MTE2", "MTE1", SIG_K_L1_1)

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1_0)
                            T.wait_flag("M", "MTE1", SIG_L0AB_0)
                            T.copy(k_l1_0, l0b_0, transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1_0)
                            T.set_flag("MTE1", "M", SIG_L0AB_0)

                            T.wait_flag("MTE1", "M", SIG_L0AB_1)
                            T.wait_flag("FIX", "M", SIG_L0C_1)
                            T.mma(l0a_1, l0b_1, l0c_1, init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB_1)
                            T.set_flag("M", "FIX", SIG_L0C_1)

                            T.wait_flag("M", "FIX", SIG_L0C_0)
                            T.copy_op.copy_cv_experiment(l0c_0, s_ub_0, V1)
                            T.set_flag("FIX", "M", SIG_L0C_0)

                            # ---- Wave 4: Stage K[3]→l0b_1 | MMA K[2]→l0c_0 | CV l0c_1→s_ub_1 ----
                            T.wait_flag("MTE2", "MTE1", SIG_K_L1_1)
                            T.wait_flag("M", "MTE1", SIG_L0AB_1)
                            T.copy(k_l1_1, l0b_1, transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1_1)
                            T.set_flag("MTE1", "M", SIG_L0AB_1)

                            T.wait_flag("MTE1", "M", SIG_L0AB_0)
                            T.wait_flag("FIX", "M", SIG_L0C_0)
                            T.mma(l0a_1, l0b_0, l0c_0, init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB_0)
                            T.set_flag("M", "FIX", SIG_L0C_0)

                            T.wait_flag("M", "FIX", SIG_L0C_1)
                            T.copy_op.copy_cv_experiment(l0c_1, s_ub_1, V1)
                            T.set_flag("FIX", "M", SIG_L0C_1)

                            # ---- Wave 5: MMA K[3]→l0c_1 | CV l0c_0→s_ub_2 (drain) ----
                            T.wait_flag("MTE1", "M", SIG_L0AB_1)
                            T.wait_flag("FIX", "M", SIG_L0C_1)
                            T.mma(l0a_1, l0b_1, l0c_1, init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB_1)
                            T.set_flag("M", "FIX", SIG_L0C_1)

                            T.wait_flag("M", "FIX", SIG_L0C_0)
                            T.copy_op.copy_cv_experiment(l0c_0, s_ub_2, V1)
                            T.set_flag("FIX", "M", SIG_L0C_0)

                            # ---- Wave 6: CV l0c_1→s_ub_3 (drain) ----
                            T.wait_flag("M", "FIX", SIG_L0C_1)
                            T.copy_op.copy_cv_experiment(l0c_1, s_ub_3, V1)
                            T.set_flag("FIX", "M", SIG_L0C_1)

                        # Per-n_outer sync: both token_a and token_b tiles are in UB
                        if n_outer % 2 == 0:
                            T.set_cross_flag("FIX", CROSS_C2V_0)
                        else:
                            T.set_cross_flag("FIX", CROSS_C2V_1)

                # Destroy: consume outstanding init-direction flags
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                T.wait_flag("MTE1", "MTE2", SIG_K_L1_0)
                T.wait_flag("MTE1", "MTE2", SIG_K_L1_1)
                T.wait_flag("M", "MTE1", SIG_L0AB_0)
                T.wait_flag("M", "MTE1", SIG_L0AB_1)
                T.wait_flag("FIX", "M", SIG_L0C_0)
                T.wait_flag("FIX", "M", SIG_L0C_1)

            # ================================================================
            # V scope: s_ub (from CV通路) → ReLU → mul weights → reduce → mask → Logits
            # ================================================================
            kv = kv_block_size  # shorthand

            with T.Scope("V"):
                # Init: UB buffers start free
                T.set_flag("V", "MTE2", SIG_W_UB)
                T.set_flag("MTE3", "V", SIG_LOGITS)
                # Pre-arm buffer-free credit so C's first n_outer can proceed.
                T.set_cross_flag("V", CROSS_V2C)

                for pair_i in T.serial(num_pairs_bx):
                    for n_outer in T.serial(topk_groups):
                        # Per-n_outer sync: wait for C to finish this topk tile
                        if n_outer % 2 == 0:
                            T.wait_cross_flag(CROSS_C2V_0, "V")
                        else:
                            T.wait_cross_flag(CROSS_C2V_1, "V")

                        t_a = pair_i * 2
                        token_idx = bx * num_tokens_per_kernel + t_a + by
                        if token_idx < seq_len:
                            n_i_base = n_outer * 4

                            # -- DMA weights once (shared across 4 blocks) --
                            T.wait_flag("V", "MTE2", SIG_W_UB)
                            T.copy(Weights[token_idx, :], weights_ub)
                            T.set_flag("MTE2", "V", SIG_W_UB)
                            T.wait_flag("MTE2", "V", SIG_W_UB)
                            T.copy(weights_ub, weights)
                            T.set_flag("V", "MTE2", SIG_W_UB)

                            # -- Vector ops per block on [H, kv] (s_ub from CV通路) --
                            T.pipe_barrier("v")
                            T.tile.relu(s_ub_0, s_ub_0)
                            T.tile.relu(s_ub_1, s_ub_1)
                            T.tile.relu(s_ub_2, s_ub_2)
                            T.tile.relu(s_ub_3, s_ub_3)
                            T.tile.row_expand_mul(s_ub_0, s_ub_0, weights)
                            T.pipe_barrier("v")
                            T.tile.row_expand_mul(s_ub_1, s_ub_1, weights)
                            T.pipe_barrier("v")
                            T.tile.row_expand_mul(s_ub_2, s_ub_2, weights)
                            T.pipe_barrier("v")
                            T.tile.row_expand_mul(s_ub_3, s_ub_3, weights)
                            T.pipe_barrier("v")

                            # -- Reduce sum per block: [H, kv] → [kv] temp --
                            T.reduce_sum(s_ub_0, logits_t0, dim=0, clear=True)
                            T.pipe_barrier("v")
                            T.reduce_sum(s_ub_1, logits_t1, dim=0, clear=True)
                            T.pipe_barrier("v")
                            T.reduce_sum(s_ub_2, logits_t2, dim=0, clear=True)
                            T.pipe_barrier("v")
                            T.reduce_sum(s_ub_3, logits_t3, dim=0, clear=True)
                            T.pipe_barrier("v")

                            # -- Merge temps into logits_4x columns --
                            T.wait_flag("MTE3", "V", SIG_LOGITS)
                            T.copy(logits_t0, logits_4x[0, 0 * kv : 1 * kv])
                            T.copy(logits_t1, logits_4x[0, 1 * kv : 2 * kv])
                            T.copy(logits_t2, logits_4x[0, 2 * kv : 3 * kv])
                            T.copy(logits_t3, logits_4x[0, 3 * kv : 4 * kv])

                            # ================================================
                            # Mask: 4 blocks merged → [64] uint8 (≥32 ✓)
                            # ================================================
                            n_i_0 = n_i_base + 0
                            n_i_1 = n_i_base + 1
                            n_i_2 = n_i_base + 2
                            n_i_3 = n_i_base + 3

                            # (1) create position vectors (4 different block_start)
                            T.tile.createvecindex(kvpi_a, TopKBlockIndex[token_idx, n_i_0] * kv)
                            T.tile.createvecindex(kvpi_b, TopKBlockIndex[token_idx, n_i_1] * kv)
                            T.tile.createvecindex(kvpi_c, TopKBlockIndex[token_idx, n_i_2] * kv)
                            T.tile.createvecindex(kvpi_d, TopKBlockIndex[token_idx, n_i_3] * kv)
                            # (2) copy int32→float32, concatenate into [4*kv]
                            T.copy(kvpi_a, kvpf_4x[0 * kv : 1 * kv])
                            T.copy(kvpi_b, kvpf_4x[1 * kv : 2 * kv])
                            T.copy(kvpi_c, kvpf_4x[2 * kv : 3 * kv])
                            T.copy(kvpi_d, kvpf_4x[3 * kv : 4 * kv])

                            # (3) compare: GE cu_seqlen_ks, LT cu_seqlen_ke
                            cu_k_s_min = CuSeqLenKS[token_idx]
                            cu_k_e_max = CuSeqLenKE[token_idx]
                            T.tile.compare(mask1_ub, kvpf_4x, T.float32(cu_k_s_min), "GE")
                            T.tile.compare(mask2_ub, kvpf_4x, T.float32(cu_k_e_max), "LT")
                            T.pipe_barrier("v")
                            T.tile.bitwise_and(mask1_ub, mask1_ub, mask2_ub)

                            # (4) select: mask out-of-range → -inf
                            T.tile.select(logits_4x[0, :], mask1_ub, logits_4x[0, :], -T.infinity(accum_dtype), "VSEL_TENSOR_SCALAR_MODE")

                            T.set_flag("V", "MTE3", SIG_LOGITS)

                            # -- DMA logits → output: 4 × [kv] slices --
                            T.wait_flag("V", "MTE3", SIG_LOGITS)
                            T.copy(
                                logits_4x[0, 0 * kv : 1 * kv],
                                Logits[token_idx, n_i_base + 0, :],
                            )
                            T.copy(
                                logits_4x[0, 1 * kv : 2 * kv],
                                Logits[token_idx, n_i_base + 1, :],
                            )
                            T.copy(
                                logits_4x[0, 2 * kv : 3 * kv],
                                Logits[token_idx, n_i_base + 2, :],
                            )
                            T.copy(
                                logits_4x[0, 3 * kv : 4 * kv],
                                Logits[token_idx, n_i_base + 3, :],
                            )
                            T.set_flag("MTE3", "V", SIG_LOGITS)

                            # All UB ops done (masking + DMA issued) →
                            # release s_ub_0..3 so C can overwrite via CV通路.
                            T.set_cross_flag("V", CROSS_V2C)
                        else:
                            # token out of range: still release the credit so C's
                            # per-n_outer wait_cross_flag count stays balanced.
                            T.set_cross_flag("V", CROSS_V2C)

                # Destroy: consume outstanding init-direction flags
                T.wait_flag("V", "MTE2", SIG_W_UB)
                T.wait_flag("MTE3", "V", SIG_LOGITS)

    return kernel


def ref_block_sparse_mqa_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    topk_block_indices: torch.Tensor,
    kv_block_size: int,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation of sparse MQA attention using torch_npu vectorization."""
    seq_len, heads, index_dim = q.shape
    seq_len_kv = k.shape[0]
    topk = topk_block_indices.shape[1]

    q = q.float()
    k = k.float()
    weights = weights.float()

    block_indices = topk_block_indices.unsqueeze(-1) * kv_block_size + torch.arange(kv_block_size, device=q.device)
    block_indices = block_indices.long()

    k_gathered = k[block_indices.view(-1)]
    k_gathered = k_gathered.view(seq_len, topk, kv_block_size, index_dim)

    scores = torch.einsum("qhd,qkbd->qkbh", q, k_gathered)

    weights_expanded = weights.unsqueeze(1).unsqueeze(2)
    scores = scores.relu() * weights_expanded

    logits = scores.sum(dim=-1)

    pos_out_of_bounds = block_indices >= seq_len_kv
    cu_seqlen_ks_exp = cu_seqlen_ks.unsqueeze(1).unsqueeze(2)
    cu_seqlen_ke_exp = cu_seqlen_ke.unsqueeze(1).unsqueeze(2)
    pos_invalid = (block_indices < cu_seqlen_ks_exp) | (block_indices >= cu_seqlen_ke_exp)
    invalid_mask = pos_out_of_bounds | pos_invalid
    logits = logits.masked_fill(invalid_mask, float("-inf"))

    return logits


def test_block_sparse_mqa_attn(
    seq_len: int,
    seq_len_kv: int,
    heads: int,
    index_dim: int,
    kv_block_size: int,
    topk: int,
    dtype: str = "float16",
    grid_size: int = 24,
):
    """Test sparse MQA attention kernel with golden validation."""
    kernel = block_sparse_mqa_attn_return_logits(
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        kv_block_size=kv_block_size,
        topk=topk,
        heads=heads,
        index_dim=index_dim,
        grid_size=grid_size,
    )
    print(kernel.get_kernel_source())

    q = torch.rand((seq_len, heads, index_dim), dtype=torch.float16)
    k = torch.rand((seq_len_kv, index_dim), dtype=torch.float16)
    weights = torch.rand((seq_len, heads), dtype=torch.float16)

    cu_seqlen_ks = torch.zeros(seq_len, dtype=torch.int32)
    cu_seqlen_ke = torch.arange(1, seq_len + 1, dtype=torch.int32) * (seq_len_kv // seq_len)
    cu_seqlen_ke = cu_seqlen_ke.clamp(max=seq_len_kv)

    max_block_id = seq_len_kv // kv_block_size
    topk_block_indices = torch.randint(0, max_block_id, (seq_len, topk), dtype=torch.int32)

    logits = torch.empty((seq_len, topk * kv_block_size), dtype=torch.float32).npu()
    torch.npu.synchronize()
    logits = kernel(
        q.npu(),
        k.npu(),
        topk_block_indices.npu(),
        weights.npu(),
        cu_seqlen_ks.npu(),
        cu_seqlen_ke.npu(),
    )
    torch.npu.synchronize()
    ref_logits = ref_block_sparse_mqa_attn(
        q,
        k,
        topk_block_indices,
        kv_block_size,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(ref_logits, logits, rtol=1e-2, atol=1e-2)

    print(f"Test passed! seq_len={seq_len}, seq_len_kv={seq_len_kv}, heads={heads}, topk={topk}")
    print(f"  Q shape: {q.shape}")
    print(f"  K shape: {k.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  kv_block_size: {kv_block_size}")

    return logits


def get_npu_core_num() -> int:
    """Query the current NPU's AI Cube core count (== the kernel's grid_size).

    Each kernel block uses 1 cube + 2 vector sub-cores (by in {0,1}), so the
    number of blocks we launch is the cube core count. Raises if the device
    can't be queried -- grid_size is compiled into the kernel, so a wrong guess
    would silently mis-shape it; failing loudly is safer.
    """
    import torch_npu  # noqa: F401

    props = torch.npu.get_device_properties(0)
    cube = int(getattr(props, "cube_core_num", 0))
    if cube <= 0:
        raise RuntimeError(f"could not determine NPU cube_core_num (got {cube!r})")
    return cube


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Block Sparse MQA Attention Kernel Test (A5 CV通路)")
    parser.add_argument("--seq_len", type=int, default=1024, help="Query sequence length")
    parser.add_argument("--seq_len_kv", type=int, default=128 * 1024, help="KV sequence length")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--index_dim", type=int, default=128, help="Index dimension")
    parser.add_argument("--kv_block_size", type=int, default=128, help="KV block size")
    parser.add_argument("--topk", type=int, default=64, help="Number of top blocks (must be divisible by 4)")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type")
    args = parser.parse_args()

    # A5-only guard: this kernel requires A5 CV通路 (TMOV).
    from tilelang.utils.target import determine_platform

    if determine_platform() != "A5":
        print(f"[SKIP] This kernel requires A5 CV通路, treat it as Kernel Output Match, detected: {determine_platform()}")
        sys.exit(0)

    torch.set_default_device("npu")
    torch.manual_seed(42)
    tilelang.disable_cache()

    # grid_size is always the current NPU's cube core count (one block per cube).
    args.grid_size = get_npu_core_num()
    print(f"[grid_size] NPU cube core count -> grid_size={args.grid_size}")

    print("=" * 60)
    print("Block Sparse MQA Attention Kernel Test (A5 CV通路)")
    print("=" * 60)
    print("Configuration:")
    print(f"  seq_len: {args.seq_len}")
    print(f"  seq_len_kv: {args.seq_len_kv}")
    print(f"  heads: {args.heads}")
    print(f"  index_dim: {args.index_dim}")
    print(f"  kv_block_size: {args.kv_block_size}")
    print(f"  topk: {args.topk}")
    print(f"  grid_size: {args.grid_size}")
    _tokens_per_block = (args.seq_len + args.grid_size - 1) // args.grid_size
    _num_pairs = (_tokens_per_block + 1) // 2
    print(f"  derived num_pairs: {_num_pairs}")
    print(f"  tokens_per_block: {2 * _num_pairs}")
    print(f"  dtype: {args.dtype}")
    print()

    test_block_sparse_mqa_attn(
        seq_len=args.seq_len,
        seq_len_kv=args.seq_len_kv,
        heads=args.heads,
        index_dim=args.index_dim,
        kv_block_size=args.kv_block_size,
        topk=args.topk,
        grid_size=args.grid_size,
        dtype=args.dtype,
    )
    print("Kernel Output Match!")
