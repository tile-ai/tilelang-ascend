"""
Block Sparse MQA Attention Kernel (Expert Mode)

Manual CV scope, manual sync, explicit L0A/L0B/MMA.
Pure expert mode: all auto passes disabled.

Flag conventions:
  - MTE1↔MTE2 for L1 buffer management
  - M↔MTE1 for L0A/L0B management
  - M↔FIX for L0C management
  - V↔MTE2 for UB buffer management
  - V↔MTE3 for output buffer management
"""

import tilelang
from tilelang import language as T
import argparse
import torch

tilelang.disable_cache()


@tilelang.jit(
    out_idx=[3],
    workspace_idx=[-1],
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
    # provisioned beyond seq_len are guarded by the `if token_* < seq_len` checks.
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
    SIG_S_UB = 0  # s_ub: V↔MTE2
    SIG_W_UB = 1  # weights_ub: V↔MTE2
    SIG_LOGITS = 0  # logits: V↔MTE3
    # Cross-scope (ping-pong flags for n_outer-level pipelining)
    CROSS_FLAG_C2V_0 = 0  # C→V for even n_outer
    CROSS_FLAG_C2V_1 = 1  # C→V for odd n_outer

    @T.prim_func
    def kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([seq_len, topk], index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
        workspace_1: T.Tensor([seq_len, topk, H_per_block, kv_block_size], accum_dtype),
    ):
        with T.Kernel(grid_size, is_npu=True) as (bx, by):
            # Per-core valid pair count. Compile-time num_pairs is the upper
            # bound (buffer alloc / unroll); each core loops only over the pairs
            # it actually owns. This is what makes an arbitrary grid_size safe:
            # the tail core (fewer valid tokens) runs fewer iterations instead of
            # entering fully-empty pairs. A fully-empty pair would still execute
            # the unconditional set/wait_cross_flag with no buffer work to pace
            # it, desyncing the intra-block cube<->vec handshake -> device hang.
            # T.min clamps to num_pairs; negative arg on an over-shot core yields
            # a 0-trip loop (harmless; our derivation never produces empty cores).
            num_pairs_bx = T.min(num_pairs, T.ceildiv(seq_len - bx * num_tokens_per_kernel, 2))
            # ---- V scope: UB allocations (4 blocks merged) ----
            s_ub_4x = T.alloc_ub([H_per_block, kv_block_size * 4], accum_dtype)
            logits_4x = T.alloc_ub([1, kv_block_size * 4], accum_dtype)
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
            # C scope: double-buffered L0B/L0C/k_l1 4-stage SW pipeline
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

                        # ====================================================
                        # token_a: 4 blocks, double-buffered pipeline
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

                            # ---- Wave 3: DMA K[3]→k_l1_1 | Stage K[2]→l0b_0 | MMA K[1]→l0c_1 | Copy l0c_0→ws ----
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
                            T.copy(l0c_0, workspace_1[token_a, n_i0, :, :])
                            T.set_flag("FIX", "M", SIG_L0C_0)

                            # ---- Wave 4: Stage K[3]→l0b_1 | MMA K[2]→l0c_0 | Copy l0c_1→ws ----
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
                            T.copy(l0c_1, workspace_1[token_a, n_i1, :, :])
                            T.set_flag("FIX", "M", SIG_L0C_1)

                            # ---- Wave 5: MMA K[3]→l0c_1 | Copy l0c_0→ws (drain) ----
                            T.wait_flag("MTE1", "M", SIG_L0AB_1)
                            T.wait_flag("FIX", "M", SIG_L0C_1)
                            T.mma(l0a_0, l0b_1, l0c_1, init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB_1)
                            T.set_flag("M", "FIX", SIG_L0C_1)

                            T.wait_flag("M", "FIX", SIG_L0C_0)
                            T.copy(l0c_0, workspace_1[token_a, n_i2, :, :])
                            T.set_flag("FIX", "M", SIG_L0C_0)

                            # ---- Wave 6: Copy l0c_1→ws (drain) ----
                            T.wait_flag("M", "FIX", SIG_L0C_1)
                            T.copy(l0c_1, workspace_1[token_a, n_i3, :, :])
                            T.set_flag("FIX", "M", SIG_L0C_1)

                        # ====================================================
                        # token_b: 4 blocks, double-buffered pipeline
                        # ====================================================
                        t_b = pair_i * 2 + 1
                        token_b = bx * num_tokens_per_kernel + t_b
                        if token_b < seq_len:
                            n_i1 = n_outer * 4 + 1
                            n_i2 = n_outer * 4 + 2
                            n_i3 = n_outer * 4 + 3

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

                            # ---- Wave 3: DMA K[3]→k_l1_1 | Stage K[2]→l0b_0 | MMA K[1]→l0c_1 | Copy l0c_0→ws ----
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
                            T.copy(l0c_0, workspace_1[token_b, n_i0, :, :])
                            T.set_flag("FIX", "M", SIG_L0C_0)

                            # ---- Wave 4: Stage K[3]→l0b_1 | MMA K[2]→l0c_0 | Copy l0c_1→ws ----
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
                            T.copy(l0c_1, workspace_1[token_b, n_i1, :, :])
                            T.set_flag("FIX", "M", SIG_L0C_1)

                            # ---- Wave 5: MMA K[3]→l0c_1 | Copy l0c_0→ws (drain) ----
                            T.wait_flag("MTE1", "M", SIG_L0AB_1)
                            T.wait_flag("FIX", "M", SIG_L0C_1)
                            T.mma(l0a_1, l0b_1, l0c_1, init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB_1)
                            T.set_flag("M", "FIX", SIG_L0C_1)

                            T.wait_flag("M", "FIX", SIG_L0C_0)
                            T.copy(l0c_0, workspace_1[token_b, n_i2, :, :])
                            T.set_flag("FIX", "M", SIG_L0C_0)

                            # ---- Wave 6: Copy l0c_1→ws (drain) ----
                            T.wait_flag("M", "FIX", SIG_L0C_1)
                            T.copy(l0c_1, workspace_1[token_b, n_i3, :, :])
                            T.set_flag("FIX", "M", SIG_L0C_1)

                        # Per-n_outer sync: both token_a and token_b ready
                        if n_outer % 2 == 0:
                            T.set_cross_flag("FIX", CROSS_FLAG_C2V_0)
                        else:
                            T.set_cross_flag("FIX", CROSS_FLAG_C2V_1)

                # Destroy: consume outstanding init-direction flags
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                T.wait_flag("MTE1", "MTE2", SIG_K_L1_0)
                T.wait_flag("MTE1", "MTE2", SIG_K_L1_1)
                T.wait_flag("M", "MTE1", SIG_L0AB_0)
                T.wait_flag("M", "MTE1", SIG_L0AB_1)
                T.wait_flag("FIX", "M", SIG_L0C_0)
                T.wait_flag("FIX", "M", SIG_L0C_1)

            # ================================================================
            # V scope: workspace → s_ub → ReLU → mul weights → reduce → Logits
            # ================================================================
            kv = kv_block_size  # shorthand

            with T.Scope("V"):
                # Init: UB buffers start free
                T.set_flag("V", "MTE2", SIG_S_UB)
                T.set_flag("V", "MTE2", SIG_W_UB)
                T.set_flag("MTE3", "V", SIG_LOGITS)

                for pair_i in T.serial(num_pairs_bx):
                    for n_outer in T.serial(topk_groups):
                        # Per-n_outer sync: wait for C scope to finish this topk tile
                        if n_outer % 2 == 0:
                            T.wait_cross_flag(CROSS_FLAG_C2V_0, "MTE2")
                        else:
                            T.wait_cross_flag(CROSS_FLAG_C2V_1, "MTE2")

                        t_a = pair_i * 2
                        token_idx = bx * num_tokens_per_kernel + t_a + by
                        if token_idx < seq_len:
                            n_i_base = n_outer * 4

                            # -- DMA 4 workspace blocks → s_ub_4x columns --
                            T.wait_flag("V", "MTE2", SIG_S_UB)
                            T.copy(
                                workspace_1[token_idx, n_i_base + 0, :, :],
                                s_ub_4x[:, 0 * kv : 1 * kv],
                            )
                            T.copy(
                                workspace_1[token_idx, n_i_base + 1, :, :],
                                s_ub_4x[:, 1 * kv : 2 * kv],
                            )
                            T.copy(
                                workspace_1[token_idx, n_i_base + 2, :, :],
                                s_ub_4x[:, 2 * kv : 3 * kv],
                            )
                            T.copy(
                                workspace_1[token_idx, n_i_base + 3, :, :],
                                s_ub_4x[:, 3 * kv : 4 * kv],
                            )
                            T.set_flag("MTE2", "V", SIG_S_UB)
                            T.wait_flag("MTE2", "V", SIG_S_UB)

                            # -- DMA weights once (shared across 4 blocks) --
                            T.wait_flag("V", "MTE2", SIG_W_UB)
                            T.copy(Weights[token_idx, :], weights_ub)
                            T.set_flag("MTE2", "V", SIG_W_UB)
                            T.wait_flag("MTE2", "V", SIG_W_UB)
                            T.copy(weights_ub, weights)
                            T.pipe_barrier("v")
                            T.set_flag("V", "MTE2", SIG_W_UB)

                            # -- Vector ops once on [H, 4*kv] --
                            T.tile.relu(s_ub_4x, s_ub_4x)
                            T.tile.row_expand_mul(s_ub_4x, s_ub_4x, weights)

                            # -- Reduce sum: [H, 4*kv] → [4*kv] --
                            T.wait_flag("MTE3", "V", SIG_LOGITS)
                            T.reduce_sum(s_ub_4x, logits_4x, dim=0, clear=True)
                            T.set_flag("V", "MTE2", SIG_S_UB)

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
                            T.pipe_barrier("v")

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

                # Destroy: consume outstanding init-direction flags
                T.wait_flag("V", "MTE2", SIG_S_UB)
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
    parser = argparse.ArgumentParser(description="Block Sparse MQA Attention Kernel Test")
    parser.add_argument("--seq_len", type=int, default=1024, help="Query sequence length")
    parser.add_argument("--seq_len_kv", type=int, default=128 * 1024, help="KV sequence length")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--index_dim", type=int, default=128, help="Index dimension")
    parser.add_argument("--kv_block_size", type=int, default=128, help="KV block size")
    parser.add_argument("--topk", type=int, default=64, help="Number of top blocks (must be divisible by 4)")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type")
    args = parser.parse_args()

    torch.set_default_device("npu")
    torch.manual_seed(42)
    tilelang.disable_cache()

    # grid_size is always the current NPU's cube core count (one block per cube).
    args.grid_size = get_npu_core_num()
    print(f"[grid_size] NPU cube core count -> grid_size={args.grid_size}")

    print("=" * 60)
    print("Block Sparse MQA Attention Kernel Test")
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
