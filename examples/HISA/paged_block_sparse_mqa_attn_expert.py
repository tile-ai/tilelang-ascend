"""1
Paged Block Sparse MQA Attention Kernel — Decode, Flat Grid
[Optimization 1] 4×K L1 + 4×L0C + MTE2∥V overlap + DMA reorder

Strategy:
  - 4×K L1 slots: K[0]+Q DMA'd first (early staging priority), then K[1..3]
  - 4×L0C slots: each MMA has its own l0c, no FIX↔M contention
  - 2×L0B ping-pong (L0B 64KB hardware limit, unchanged)
  - Removed unnecessary MTE1→MTE2 signals (no L1 buffer reuse)
  - Removed `if token < total_tokens` guard (decode: always true)
  - 4 ws slots + per-block cross-flags:
      Wave 2: ws[0]→BLK0   Wave 3: ws[1]→BLK1
      Wave 4: ws[2]→BLK2   Wave 5: ws[3]→BLK3
  - V double-buffered s_ub_e/s_ub_l per AIV
  - Late DMA enqueued BEFORE early mask+output → MTE2 ∥ V overlap
  - Tail-fill replaces mask pipeline

C scope (6 waves, DMA reorder for reduced pipeline fill):
    W0: DMA K[0]+Q first → K[1..3] (MTE2 queue: K0,Q,K1,K2,K3)
    W1: wait(K0,Q) → Stage Q+K[0] → MMA K[0]
    W2: L0C0→ws[0]→BLK0 | Stage K[1] | MMA K[1]
    W3: L0C1→ws[1]→BLK1 | Stage K[2] | MMA K[2]
    W4: L0C2→ws[2]→BLK2 | Stage K[3] | MMA K[3]
    W5: L0C3→ws[3]→BLK3 (drain)

Pipeline fill improvement:
    Original: 5 DMA (K0..K3+Q) must complete before MTE1 starts
    Optimized: only K0+Q before MTE1 starts (K1..K3 overlap with staging/MMA)

V scope (MTE2 ∥ V):
    early: DMA_in → compute → [enqueue late DMA] → mask → output
    late:                                        [wait DMA] → compute → mask → output
                    ↑ late MTE2 DMA runs while V does early mask+output ↑

Key invariants:
  - Grid unchanged: batch * topk_groups (16 blocks for batch=1, topk=64)
  - 16 cores, 2 AIVs per core, no head split
  - Same output shape, same reference function
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
def paged_block_sparse_mqa_attn_return_logits(
    batch: int,
    seq_len: int,
    num_phys_blocks: int,
    kv_block_size: int,
    topk: int,
    heads: int,
    index_dim: int,
    max_blocks: int,
    num_stages: int = 2,  # noqa: ARG001
    threads: int = 2,  # noqa: ARG001
):
    dtype = "float16"
    accum_dtype = "float32"
    index_dtype = "int32"

    assert topk % 4 == 0, "topk must be divisible by 4"
    topk_groups = topk // 4

    total_tokens = batch * seq_len
    grid_size = batch * topk_groups

    index_q_shape = [total_tokens, heads, index_dim]
    topk_index_shape = [total_tokens, topk]
    logits_shape = [total_tokens, topk, kv_block_size]
    weights_shape = [total_tokens, heads]

    kv_cache_shape = [num_phys_blocks, kv_block_size, 1, index_dim]
    block_tables_shape = [batch, max_blocks]
    context_lens_shape = [batch]

    H_per_block = heads
    kv = kv_block_size

    # ---------- Signal IDs ----------
    # C scope: MTE2→MTE1 only (L1 data readiness — MTE1→MTE2 removed, no reuse)
    SIG_Q_L1 = 0
    SIG_K_L1_0 = 1
    SIG_K_L1_1 = 2
    SIG_K_L1_2 = 3
    SIG_K_L1_3 = 4
    # C scope: M↔MTE1  (L0A/L0B — 2-way ping-pong, L0B 64KB limit)
    SIG_L0AB_0 = 0
    SIG_L0AB_1 = 1
    # C scope: FIX↔M  (L0C — 4 independent, no reuse)
    SIG_L0C_0 = 0
    SIG_L0C_1 = 1
    SIG_L0C_2 = 2
    SIG_L0C_3 = 3
    # V scope: V↔MTE2  (s_ub double-buffered + weights)
    SIG_S_E = 0   # s_ub_e — early block (0 or 1)
    SIG_S_L = 1   # s_ub_l — late  block (2 or 3)
    SIG_W_UB = 2
    # V scope: MTE3↔V  (logits output)
    SIG_LOGITS_E = 0
    SIG_LOGITS_L = 1
    # Cross-scope: per-block flags
    FLAG_BLK0 = 0
    FLAG_BLK1 = 1
    FLAG_BLK2 = 2
    FLAG_BLK3 = 3

    @T.prim_func
    def kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        KvCache: T.Tensor(kv_cache_shape, dtype),  # type: ignore
        TopKBlockIndex: T.Tensor(topk_index_shape, index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(weights_shape, dtype),  # type: ignore
        ContextLens: T.Tensor(context_lens_shape, index_dtype),  # type: ignore
        BlockTables: T.Tensor(block_tables_shape, index_dtype),  # type: ignore
        ws_buf: T.Tensor([4, total_tokens, topk, H_per_block, kv], accum_dtype),
    ):
        with T.Kernel(grid_size, is_npu=True) as (bx, by):
            token = bx // topk_groups
            n_outer = bx - token * topk_groups
            n_i_base = n_outer * 4
            n_i0 = n_i_base + 0
            n_i1 = n_i_base + 1
            n_i2 = n_i_base + 2
            n_i3 = n_i_base + 3

            # ---- V scope: UB allocations ----
            # Double-buffered s_ub for DMA↔compute overlap
            s_ub_e = T.alloc_ub([H_per_block, kv], accum_dtype)
            logits_e = T.alloc_ub([1, kv], accum_dtype)
            s_ub_l = T.alloc_ub([H_per_block, kv], accum_dtype)
            logits_l = T.alloc_ub([1, kv], accum_dtype)
            weights_ub = T.alloc_ub([heads], dtype)
            weights = T.alloc_ub([heads], accum_dtype)
            # Mask buffers eliminated — scalar tail-fill replaces mask pipeline

            # ---- C scope: L1 / L0A / L0B / L0C ----
            # 4×K L1 slots + 4×L0C slots — no reuse, fewer flag waits
            q_l1 = T.alloc_L1([H_per_block, index_dim], dtype)
            k_l1_0 = T.alloc_L1([kv, index_dim], dtype)
            k_l1_1 = T.alloc_L1([kv, index_dim], dtype)
            k_l1_2 = T.alloc_L1([kv, index_dim], dtype)
            k_l1_3 = T.alloc_L1([kv, index_dim], dtype)
            l0a = T.alloc_L0A([H_per_block, index_dim], dtype)
            l0b_0 = T.alloc_L0B([index_dim, kv], dtype)
            l0b_1 = T.alloc_L0B([index_dim, kv], dtype)
            l0c_0 = T.alloc_L0C([H_per_block, kv], accum_dtype)
            l0c_1 = T.alloc_L0C([H_per_block, kv], accum_dtype)
            l0c_2 = T.alloc_L0C([H_per_block, kv], accum_dtype)
            l0c_3 = T.alloc_L0C([H_per_block, kv], accum_dtype)

            # ================================================================
            # C scope: 6-wave pipeline, DMA reorder + signal cleanup
            #
            # Wave 0: DMA K[0]+Q first → K[1..3] (MTE2 queue: K0,Q,K1,K2,K3)
            # Wave 1: wait(K0,Q) → Stage Q+K[0] → MMA K[0]
            # Wave 2: L0C0→ws[0]→BLK0, Stage K[1], MMA K[1]
            # Wave 3: L0C1→ws[1]→BLK1, Stage K[2], MMA K[2]
            # Wave 4: L0C2→ws[2]→BLK2, Stage K[3], MMA K[3]
            # Wave 5: L0C3→ws[3]→BLK3
            #
            # Removed: MTE1→MTE2 signals (no L1 reuse), if guard (decode always true)
            # Pipeline fill: only K0+Q before MTE1 starts (K1..K3 overlap staging)
            # ================================================================
            with T.Scope("C"):
                T.set_flag("M", "MTE1", SIG_L0AB_0)
                T.set_flag("M", "MTE1", SIG_L0AB_1)
                T.set_flag("FIX", "M", SIG_L0C_0)
                T.set_flag("FIX", "M", SIG_L0C_1)
                T.set_flag("FIX", "M", SIG_L0C_2)
                T.set_flag("FIX", "M", SIG_L0C_3)

                b = token // seq_len

                # ---- Wave 0: DMA K[0]+Q first, then K[1..3] ----
                T.copy(
                    KvCache[BlockTables[b, TopKBlockIndex[token, n_i0]], :, 0, :],
                    k_l1_0,
                )
                T.set_flag("MTE2", "MTE1", SIG_K_L1_0)

                T.copy(IndexQ[token, :, :], q_l1)
                T.set_flag("MTE2", "MTE1", SIG_Q_L1)

                T.copy(
                    KvCache[BlockTables[b, TopKBlockIndex[token, n_i1]], :, 0, :],
                    k_l1_1,
                )
                T.set_flag("MTE2", "MTE1", SIG_K_L1_1)

                T.copy(
                    KvCache[BlockTables[b, TopKBlockIndex[token, n_i2]], :, 0, :],
                    k_l1_2,
                )
                T.set_flag("MTE2", "MTE1", SIG_K_L1_2)

                T.copy(
                    KvCache[BlockTables[b, TopKBlockIndex[token, n_i3]], :, 0, :],
                    k_l1_3,
                )
                T.set_flag("MTE2", "MTE1", SIG_K_L1_3)

                # ---- Wave 1: wait(K0,Q) → Stage Q+K[0] → MMA K[0] ----
                T.wait_flag("MTE2", "MTE1", SIG_K_L1_0)
                T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB_0)

                T.copy(q_l1, l0a)
                T.copy(k_l1_0, l0b_0, transpose=True)
                T.set_flag("MTE1", "M", SIG_L0AB_0)

                T.wait_flag("MTE1", "M", SIG_L0AB_0)
                T.wait_flag("FIX", "M", SIG_L0C_0)
                T.mma(l0a, l0b_0, l0c_0, init=True)
                T.set_flag("M", "MTE1", SIG_L0AB_0)
                T.set_flag("M", "FIX", SIG_L0C_0)

                # ---- Wave 2: L0C0→ws[0]→BLK0 | Stage K[1] | MMA K[1] ----
                T.wait_flag("M", "FIX", SIG_L0C_0)
                T.copy(l0c_0, ws_buf[0, token, n_i0, :, :])
                T.set_flag("FIX", "M", SIG_L0C_0)
                T.set_cross_flag("FIX", FLAG_BLK0)

                T.wait_flag("MTE2", "MTE1", SIG_K_L1_1)
                T.wait_flag("M", "MTE1", SIG_L0AB_1)
                T.copy(k_l1_1, l0b_1, transpose=True)
                T.set_flag("MTE1", "M", SIG_L0AB_1)

                T.wait_flag("MTE1", "M", SIG_L0AB_1)
                T.wait_flag("FIX", "M", SIG_L0C_1)
                T.mma(l0a, l0b_1, l0c_1, init=True)
                T.set_flag("M", "MTE1", SIG_L0AB_1)
                T.set_flag("M", "FIX", SIG_L0C_1)

                # ---- Wave 3: L0C1→ws[1]→BLK1 | Stage K[2] | MMA K[2] ----
                T.wait_flag("M", "FIX", SIG_L0C_1)
                T.copy(l0c_1, ws_buf[1, token, n_i1, :, :])
                T.set_flag("FIX", "M", SIG_L0C_1)
                T.set_cross_flag("FIX", FLAG_BLK1)

                T.wait_flag("MTE2", "MTE1", SIG_K_L1_2)
                T.wait_flag("M", "MTE1", SIG_L0AB_0)
                T.copy(k_l1_2, l0b_0, transpose=True)
                T.set_flag("MTE1", "M", SIG_L0AB_0)

                T.wait_flag("MTE1", "M", SIG_L0AB_0)
                T.wait_flag("FIX", "M", SIG_L0C_2)
                T.mma(l0a, l0b_0, l0c_2, init=True)
                T.set_flag("M", "MTE1", SIG_L0AB_0)
                T.set_flag("M", "FIX", SIG_L0C_2)

                # ---- Wave 4: L0C2→ws[2]→BLK2 | Stage K[3] | MMA K[3] ----
                T.wait_flag("M", "FIX", SIG_L0C_2)
                T.copy(l0c_2, ws_buf[2, token, n_i2, :, :])
                T.set_flag("FIX", "M", SIG_L0C_2)
                T.set_cross_flag("FIX", FLAG_BLK2)

                T.wait_flag("MTE2", "MTE1", SIG_K_L1_3)
                T.wait_flag("M", "MTE1", SIG_L0AB_1)
                T.copy(k_l1_3, l0b_1, transpose=True)
                T.set_flag("MTE1", "M", SIG_L0AB_1)

                T.wait_flag("MTE1", "M", SIG_L0AB_1)
                T.wait_flag("FIX", "M", SIG_L0C_3)
                T.mma(l0a, l0b_1, l0c_3, init=True)
                T.set_flag("M", "MTE1", SIG_L0AB_1)
                T.set_flag("M", "FIX", SIG_L0C_3)

                # ---- Wave 5: L0C3→ws[3]→BLK3 (drain) ----
                T.wait_flag("M", "FIX", SIG_L0C_3)
                T.copy(l0c_3, ws_buf[3, token, n_i3, :, :])
                T.set_flag("FIX", "M", SIG_L0C_3)
                T.set_cross_flag("FIX", FLAG_BLK3)

                # Destroy (no MTE1→MTE2 waits — L1 buffers not reused)
                T.wait_flag("M", "MTE1", SIG_L0AB_0)
                T.wait_flag("M", "MTE1", SIG_L0AB_1)
                T.wait_flag("FIX", "M", SIG_L0C_0)
                T.wait_flag("FIX", "M", SIG_L0C_1)
                T.wait_flag("FIX", "M", SIG_L0C_2)
                T.wait_flag("FIX", "M", SIG_L0C_3)

            # ================================================================
            # V scope: tail-fill replaces mask pipeline, MTE2 ∥ V overlap
            #
            # Mask pipeline eliminated — instead, compute valid_count per block
            # and use a scalar loop to fill the tail with -inf.
            #
            # valid_count = max(0, min(kv, cu_k_e_max - block_start))
            # tail positions [valid_count .. kv) → -inf
            #
            # Late DMA enqueued BEFORE early tail-fill+output (MTE2 ∥ V).
            # ================================================================
            with T.Scope("V"):
                T.set_flag("V", "MTE2", SIG_S_E)
                T.set_flag("V", "MTE2", SIG_S_L)
                T.set_flag("V", "MTE2", SIG_W_UB)
                T.set_flag("MTE3", "V", SIG_LOGITS_E)
                T.set_flag("MTE3", "V", SIG_LOGITS_L)

                if token < total_tokens:
                    b_v = token // seq_len
                    cu_k_e_max = ContextLens[b_v]

                    # ---- Weights DMA (shared) ----
                    T.wait_flag("V", "MTE2", SIG_W_UB)
                    T.copy(Weights[token, :], weights_ub)
                    T.set_flag("MTE2", "V", SIG_W_UB)
                    T.wait_flag("MTE2", "V", SIG_W_UB)
                    T.copy(weights_ub, weights)
                    T.pipe_barrier("v")
                    T.set_flag("V", "MTE2", SIG_W_UB)

                    # ============================================
                    # Early: DMA_in → compute → tail-fill → output
                    # ============================================
                    early_flag = FLAG_BLK0 + by
                    early_n_i = n_i_base + by
                    early_ws_slot = by

                    T.wait_cross_flag(early_flag, "MTE2")

                    T.wait_flag("V", "MTE2", SIG_S_E)
                    T.copy(
                        ws_buf[early_ws_slot, token, early_n_i, :, :],
                        s_ub_e[:, :],
                    )
                    T.set_flag("MTE2", "V", SIG_S_E)
                    T.wait_flag("MTE2", "V", SIG_S_E)

                    T.tile.relu(s_ub_e, s_ub_e)
                    T.tile.row_expand_mul(s_ub_e, s_ub_e, weights)
                    T.wait_flag("MTE3", "V", SIG_LOGITS_E)
                    T.reduce_sum(s_ub_e, logits_e, dim=0, clear=True)
                    T.set_flag("V", "MTE2", SIG_S_E)

                    # ---- Tail-fill early block ----
                    e_block_start = TopKBlockIndex[token, early_n_i] * kv
                    e_valid = cu_k_e_max - e_block_start
                    e_limit = T.max(
                        T.cast(0, index_dtype),
                        T.min(T.cast(kv, index_dtype), e_valid),
                    )
                    for i in T.serial(kv):
                        if T.cast(i, index_dtype) >= e_limit:
                            logits_e[0, i] = -T.infinity(accum_dtype)

                    # ============================================
                    # Enqueue late DMA NOW (MTE2), before early output
                    # ============================================
                    late_flag = FLAG_BLK0 + 2 + by
                    late_n_i = n_i_base + 2 + by
                    late_ws_slot = 2 + by

                    T.wait_cross_flag(late_flag, "MTE2")
                    T.wait_flag("V", "MTE2", SIG_S_L)
                    T.copy(
                        ws_buf[late_ws_slot, token, late_n_i, :, :],
                        s_ub_l[:, :],
                    )
                    T.set_flag("MTE2", "V", SIG_S_L)

                    # ---- Early output ----
                    T.set_flag("V", "MTE3", SIG_LOGITS_E)
                    T.wait_flag("V", "MTE3", SIG_LOGITS_E)
                    T.copy(
                        logits_e[0, 0 * kv : 1 * kv],
                        Logits[token, early_n_i, :],
                    )
                    T.set_flag("MTE3", "V", SIG_LOGITS_E)

                    # ============================================
                    # Late: wait DMA → compute → tail-fill → output
                    # ============================================
                    T.wait_flag("MTE2", "V", SIG_S_L)

                    T.tile.relu(s_ub_l, s_ub_l)
                    T.tile.row_expand_mul(s_ub_l, s_ub_l, weights)
                    T.wait_flag("MTE3", "V", SIG_LOGITS_L)
                    T.reduce_sum(s_ub_l, logits_l, dim=0, clear=True)
                    T.set_flag("V", "MTE2", SIG_S_L)

                    # ---- Tail-fill late block ----
                    l_block_start = TopKBlockIndex[token, late_n_i] * kv
                    l_valid = cu_k_e_max - l_block_start
                    l_limit = T.max(
                        T.cast(0, index_dtype),
                        T.min(T.cast(kv, index_dtype), l_valid),
                    )
                    for i in T.serial(kv):
                        if T.cast(i, index_dtype) >= l_limit:
                            logits_l[0, i] = -T.infinity(accum_dtype)

                    # ---- Late output ----
                    T.set_flag("V", "MTE3", SIG_LOGITS_L)
                    T.wait_flag("V", "MTE3", SIG_LOGITS_L)
                    T.copy(
                        logits_l[0, 0 * kv : 1 * kv],
                        Logits[token, late_n_i, :],
                    )
                    T.set_flag("MTE3", "V", SIG_LOGITS_L)

                # Destroy
                T.wait_flag("V", "MTE2", SIG_S_E)
                T.wait_flag("V", "MTE2", SIG_S_L)
                T.wait_flag("V", "MTE2", SIG_W_UB)
                T.wait_flag("MTE3", "V", SIG_LOGITS_E)
                T.wait_flag("MTE3", "V", SIG_LOGITS_L)

    return kernel


# ============================================================================
# Reference — unchanged
# ============================================================================
def ref_paged_block_sparse_mqa_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_block_indices: torch.Tensor,
    kv_block_size: int,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
) -> torch.Tensor:
    """Reference: paged sparse MQA attention."""
    batch, seq_len, heads, index_dim = q.shape
    topk = topk_block_indices.shape[2]

    q = q.float()
    kv_cache = kv_cache.float()
    weights = weights.float()

    logits_out = torch.zeros(
        (batch, seq_len, topk, kv_block_size),
        dtype=torch.float32,
        device=q.device,
    )

    for b in range(batch):
        ctx_len = context_lens[b].item()
        for s in range(seq_len):
            for k_i in range(topk):
                logical_block = topk_block_indices[b, s, k_i].item()
                phys_block = block_tables[b, logical_block].item()
                k_block = kv_cache[phys_block, :, 0, :]

                scores = q[b, s, :, :] @ k_block.T
                scores = scores.relu()
                scores = scores * weights[b, s, :].unsqueeze(1)
                logits_val = scores.sum(dim=0)

                block_start = logical_block * kv_block_size
                pos = torch.arange(kv_block_size, device=q.device) + block_start
                pos_out = pos >= ctx_len
                logits_val[pos_out] = float("-inf")

                logits_out[b, s, k_i, :] = logits_val

    return logits_out.view(batch, seq_len, topk * kv_block_size)


def test_paged_block_sparse_mqa_attn(
    batch: int,
    seq_len: int,
    num_phys_blocks: int,
    heads: int,
    index_dim: int,
    kv_block_size: int,
    topk: int,
    max_blocks: int,
    dtype: str = "float16",
):
    """Test paged sparse MQA attention (decode, flat grid)."""
    kernel = paged_block_sparse_mqa_attn_return_logits(
        batch=batch,
        seq_len=seq_len,
        num_phys_blocks=num_phys_blocks,
        kv_block_size=kv_block_size,
        topk=topk,
        heads=heads,
        index_dim=index_dim,
        max_blocks=max_blocks,
    )
    print(kernel.get_kernel_source())

    device = "npu"
    total_tokens = batch * seq_len
    topk_groups = topk // 4

    # Generate random tensors on CPU first to work around dynamic kernel
    # init failures (aclnnInplaceRandom, aclnnMax, etc.) on this CANN
    # version, then copy to NPU via plain H2D transfers.
    with torch.device("cpu"):
        q_cpu = torch.rand((batch, seq_len, heads, index_dim), dtype=torch.float16)
        kv_cache_cpu = torch.rand((num_phys_blocks, kv_block_size, 1, index_dim), dtype=torch.float16)
        weights_cpu = torch.rand((batch, seq_len, heads), dtype=torch.float16)

        context_lens_cpu = torch.randint(
            kv_block_size, num_phys_blocks * kv_block_size + 1, (batch,), dtype=torch.int32
        )

        block_tables_cpu = (
            torch.arange(max_blocks, dtype=torch.int32)
            .unsqueeze(0)
            .expand(batch, -1)
            .contiguous()
        )

        max_logical_block = (context_lens_cpu.max().item() + kv_block_size - 1) // kv_block_size
        max_logical_block = min(max_logical_block, max_blocks)
        topk_block_indices_cpu = torch.randint(
            0, max_logical_block, (batch, seq_len, topk), dtype=torch.int32
        )

    # Copy to NPU (plain H2D, no compute kernel needed)
    q = q_cpu.to(device)
    kv_cache = kv_cache_cpu.to(device)
    weights = weights_cpu.to(device)
    context_lens = context_lens_cpu.to(device)
    block_tables = block_tables_cpu.to(device)
    topk_block_indices = topk_block_indices_cpu.to(device)

    # Flatten batch×seq_len
    q_flat = q.reshape(total_tokens, heads, index_dim).contiguous()
    topk_flat = topk_block_indices.reshape(total_tokens, topk).contiguous()
    weights_flat = weights.reshape(total_tokens, heads).contiguous()

    torch.npu.synchronize()
    logits = kernel(
        q_flat,
        kv_cache,
        topk_flat,
        weights_flat,
        context_lens,
        block_tables,
    )
    torch.npu.synchronize()

    ref_logits = ref_paged_block_sparse_mqa_attn(
        q,
        kv_cache,
        topk_block_indices,
        kv_block_size,
        weights,
        context_lens,
        block_tables,
    )
    torch.npu.synchronize()

    logits_flat = logits.view(batch, seq_len, topk * kv_block_size)
    torch.testing.assert_close(ref_logits, logits_flat, rtol=1e-2, atol=1e-2)

    print(f"Test passed! batch={batch}, seq_len={seq_len}, heads={heads}, topk={topk}")
    print(f"  grid: [{batch}, {topk_groups}]  Q: {q.shape}  Logits: {logits_flat.shape}")

    return logits_flat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paged Sparse MQA Attention — Opt1: WS Double-buffer + Padded Mask"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--num_phys_blocks", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--index_dim", type=int, default=128)
    parser.add_argument("--kv_block_size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--max_blocks", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()

    torch.set_default_device("npu")
    torch.manual_seed(42)
    tilelang.disable_cache()
    assert args.topk % 4 == 0

    print("=" * 60)
    print("Paged Sparse MQA — Opt1: Per-block flags + Double-buffered V UB")
    print("=" * 60)
    print(f"  batch={args.batch}, heads={args.heads}, index_dim={args.index_dim}")
    print(f"  kv_block_size={args.kv_block_size}, topk={args.topk}")
    print(f"  grid={args.batch * args.topk // 4} tasks")
    print(f"  cross flags: BLK0..BLK3 (per-block, immediate)")
    print(f"  V UB: s_ub_e + s_ub_l per AIV (DMA ∥ compute)")
    print()

    test_paged_block_sparse_mqa_attn(
        batch=args.batch, seq_len=args.seq_len,
        num_phys_blocks=args.num_phys_blocks, heads=args.heads,
        index_dim=args.index_dim, kv_block_size=args.kv_block_size,
        topk=args.topk, max_blocks=args.max_blocks,
    )
    print("Kernel Output Match!")

