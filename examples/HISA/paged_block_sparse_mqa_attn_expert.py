"""Persistent paged block-sparse MQA decode kernel for A3 PTO.

The math and C -> GM workspace -> V data path are unchanged from the corrected
A3 kernel.  The launch grid is capped to the physical AIC count and each
resident AIC/AIV cluster processes multiple logical tasks persistently.  The
GM workspace is already private per logical task, so no extra UB score slot is
needed; a V -> C release event bounds the cross-core semaphore queue to one
published task while C can still compute the next task ahead of that release.

The context mask has three paths: full blocks bypass masking, invalid blocks
use one Vector fill, and partial blocks use Scalar ``SetValue`` only for the
invalid tail.  The partial path explicitly synchronizes
``TCOLSUM(V) -> Scalar -> MTE3`` so both producer/consumer dependencies are
visible without constructing a full position vector.
"""

import tilelang
from tilelang import language as T
import argparse
import torch

tilelang.disable_cache()

A3_9392_AICORE_NUM = 24
PERSISTENT_QUEUE_DEPTH = 1


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
    aicore_num: int = A3_9392_AICORE_NUM,
):
    dtype = "float16"
    accum_dtype = "float32"
    index_dtype = "int32"

    assert topk % 4 == 0, "topk must be divisible by 4"
    assert aicore_num > 0, "aicore_num must be positive"
    topk_groups = topk // 4

    total_tokens = batch * seq_len
    grid_size = total_tokens * topk_groups
    kernel_grid_size = min(grid_size, aicore_num)

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
    SIG_Q_L1 = 0
    SIG_K_L1_0 = 1
    SIG_K_L1_1 = 2
    SIG_K_L1_2 = 3
    SIG_K_L1_3 = 4
    SIG_L0AB_0 = 0
    SIG_L0AB_1 = 1
    SIG_L0C_0 = 0
    SIG_L0C_1 = 1
    SIG_L0C_2 = 2
    SIG_L0C_3 = 3
    SIG_S_E = 0
    SIG_S_L = 1
    SIG_W_UB = 2
    SIG_LOGITS_E = 0
    SIG_LOGITS_L = 1
    FLAG_BLK0 = 0
    FLAG_BLK1 = 1
    FLAG_BLK2 = 2
    FLAG_BLK3 = 3
    FLAG_SCORE_RELEASE = 8
    # Partial-tail dependency: TCOLSUM(V) -> SetValue(S) -> output(MTE3).
    SIG_V_TO_S = 0
    SIG_S_TO_MTE3 = 0

    @T.prim_func
    def kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),
        KvCache: T.Tensor(kv_cache_shape, dtype),
        TopKBlockIndex: T.Tensor(topk_index_shape, index_dtype),
        Logits: T.Tensor(logits_shape, accum_dtype),
        Weights: T.Tensor(weights_shape, dtype),
        ContextLens: T.Tensor(context_lens_shape, index_dtype),
        BlockTables: T.Tensor(block_tables_shape, index_dtype),
        ws_buf: T.Tensor([4, total_tokens, topk, H_per_block, kv], accum_dtype),
    ):
        with T.Kernel(kernel_grid_size, is_npu=True) as (bx, by):
            # ---- V scope: UB allocations ----
            s_ub_e = T.alloc_ub([H_per_block, kv], accum_dtype)
            logits_e = T.alloc_ub([1, kv], accum_dtype)
            s_ub_l = T.alloc_ub([H_per_block, kv], accum_dtype)
            logits_l = T.alloc_ub([1, kv], accum_dtype)
            weights_ub = T.alloc_ub([heads], dtype)
            weights = T.alloc_ub([heads], accum_dtype)

            # ---- C scope: L1 / L0A / L0B / L0C ----
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

            with T.Scope("C"):
                T.set_flag("M", "MTE1", SIG_L0AB_0)
                T.set_flag("M", "MTE1", SIG_L0AB_1)
                T.set_flag("FIX", "M", SIG_L0C_0)
                T.set_flag("FIX", "M", SIG_L0C_1)
                T.set_flag("FIX", "M", SIG_L0C_2)
                T.set_flag("FIX", "M", SIG_L0C_3)

                for task in T.Persistent([grid_size], kernel_grid_size, bx):
                    token = task // topk_groups
                    n_outer = task - token * topk_groups
                    n_i_base = n_outer * 4
                    n_i0 = n_i_base + 0
                    n_i1 = n_i_base + 1
                    n_i2 = n_i_base + 2
                    n_i3 = n_i_base + 3

                    b = token // seq_len

                    # Finish both previous L1 -> L0/MMA users before the next
                    # iteration can overwrite their shared Q/K L1 buffers.
                    T.wait_flag("M", "MTE1", SIG_L0AB_0)
                    T.wait_flag("M", "MTE1", SIG_L0AB_1)

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
                    # A3 cross-core flags are counting semaphores.  Keep at most
                    # one logical task in flight so their 4-bit counters cannot
                    # overflow, while retaining C/V overlap through distinct GM
                    # workspace addresses.
                    if task >= PERSISTENT_QUEUE_DEPTH * kernel_grid_size:
                        T.wait_cross_flag(FLAG_SCORE_RELEASE)
                    T.set_cross_flag("FIX", FLAG_BLK0)

                    T.wait_flag("MTE2", "MTE1", SIG_K_L1_1)
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

                # Destroy
                T.wait_flag("M", "MTE1", SIG_L0AB_0)
                T.wait_flag("M", "MTE1", SIG_L0AB_1)
                T.wait_flag("FIX", "M", SIG_L0C_0)
                T.wait_flag("FIX", "M", SIG_L0C_1)
                T.wait_flag("FIX", "M", SIG_L0C_2)
                T.wait_flag("FIX", "M", SIG_L0C_3)

            with T.Scope("V"):
                T.set_flag("V", "MTE2", SIG_S_E)
                T.set_flag("V", "MTE2", SIG_S_L)
                T.set_flag("V", "MTE2", SIG_W_UB)
                T.set_flag("MTE3", "V", SIG_LOGITS_E)
                T.set_flag("MTE3", "V", SIG_LOGITS_L)

                for task in T.Persistent([grid_size], kernel_grid_size, bx):
                    token = task // topk_groups
                    n_outer = task - token * topk_groups
                    n_i_base = n_outer * 4
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
                    # full: no-op; invalid: one Vector fill; partial: write
                    # only [valid, kv) on Scalar with explicit V -> S -> MTE3.
                    e_block_start = TopKBlockIndex[token, early_n_i] * kv
                    if e_block_start + kv > cu_k_e_max:
                        if e_block_start >= cu_k_e_max:
                            T.tile.fill(logits_e, -T.infinity(accum_dtype))
                        else:
                            e_valid = cu_k_e_max - e_block_start
                            T.set_flag("V", "S", SIG_V_TO_S)
                            T.wait_flag("V", "S", SIG_V_TO_S)
                            for i in T.serial(e_valid, kv):
                                logits_e[0, i] = -T.infinity(accum_dtype)
                            T.set_flag("S", "MTE3", SIG_S_TO_MTE3)
                            T.wait_flag("S", "MTE3", SIG_S_TO_MTE3)

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

                    # Mode2 broadcasts each C ready event to both AIVs.  Each
                    # AIV consumes its two data-bearing events above; drain the
                    # partner AIV's copies as well so no 4-bit semaphore count
                    # accumulates across many persistent iterations.
                    T.wait_cross_flag(FLAG_BLK1 - by)
                    T.wait_cross_flag(FLAG_BLK3 - by)

                    # Both AIVs publish one release.  The paired AIC wait has
                    # reduce semantics and resumes only after both are done.
                    if task + PERSISTENT_QUEUE_DEPTH * kernel_grid_size < grid_size:
                        T.set_cross_flag("V", FLAG_SCORE_RELEASE)

                    # ---- Tail-fill late block: same three-way policy ----
                    l_block_start = TopKBlockIndex[token, late_n_i] * kv
                    if l_block_start + kv > cu_k_e_max:
                        if l_block_start >= cu_k_e_max:
                            T.tile.fill(logits_l, -T.infinity(accum_dtype))
                        else:
                            l_valid = cu_k_e_max - l_block_start
                            T.set_flag("V", "S", SIG_V_TO_S)
                            T.wait_flag("V", "S", SIG_V_TO_S)
                            for i in T.serial(l_valid, kv):
                                logits_l[0, i] = -T.infinity(accum_dtype)
                            T.set_flag("S", "MTE3", SIG_S_TO_MTE3)
                            T.wait_flag("S", "MTE3", SIG_S_TO_MTE3)

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
    aicore_num: int = A3_9392_AICORE_NUM,
):
    """Test paged sparse MQA attention (decode, persistent grid)."""
    kernel = paged_block_sparse_mqa_attn_return_logits(
        batch=batch,
        seq_len=seq_len,
        num_phys_blocks=num_phys_blocks,
        kv_block_size=kv_block_size,
        topk=topk,
        heads=heads,
        index_dim=index_dim,
        max_blocks=max_blocks,
        aicore_num=aicore_num,
    )
    device = "npu"
    total_tokens = batch * seq_len
    topk_groups = topk // 4

    with torch.device("cpu"):
        q_cpu = torch.rand((batch, seq_len, heads, index_dim), dtype=torch.float16)
        kv_cache_cpu = torch.rand((num_phys_blocks, kv_block_size, 1, index_dim), dtype=torch.float16)
        weights_cpu = torch.rand((batch, seq_len, heads), dtype=torch.float16)

        context_lens_cpu = torch.randint(kv_block_size, num_phys_blocks * kv_block_size + 1, (batch,), dtype=torch.int32)

        block_tables_cpu = torch.arange(max_blocks, dtype=torch.int32).unsqueeze(0).expand(batch, -1).contiguous()

        max_logical_block = (context_lens_cpu.max().item() + kv_block_size - 1) // kv_block_size
        max_logical_block = min(max_logical_block, max_blocks)
        topk_block_indices_cpu = torch.randint(0, max_logical_block, (batch, seq_len, topk), dtype=torch.int32)

    q = q_cpu.to(device)
    kv_cache = kv_cache_cpu.to(device)
    weights = weights_cpu.to(device)
    context_lens = context_lens_cpu.to(device)
    block_tables = block_tables_cpu.to(device)
    topk_block_indices = topk_block_indices_cpu.to(device)

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
    parser = argparse.ArgumentParser(description="Paged Sparse MQA Attention")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--num_phys_blocks", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--index_dim", type=int, default=128)
    parser.add_argument("--kv_block_size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--max_blocks", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument(
        "--aicore_num",
        type=int,
        default=A3_9392_AICORE_NUM,
        help="physical AIC count (9392=24; 9362=20)",
    )
    args = parser.parse_args()

    torch.set_default_device("npu")
    torch.manual_seed(42)
    tilelang.disable_cache()
    assert args.topk % 4 == 0

    print("=" * 60)
    print("Paged Sparse MQA — A3 persistent Expert kernel")
    print("=" * 60)
    print(f"  batch={args.batch}, heads={args.heads}, index_dim={args.index_dim}")
    print(f"  kv_block_size={args.kv_block_size}, topk={args.topk}")
    logical_grid = args.batch * args.topk // 4
    print(f"  grid={logical_grid} logical tasks, {min(logical_grid, args.aicore_num)} resident AICs")
    print("  mask: full bypass + invalid Vector fill + partial Scalar tail")
    print()

    test_paged_block_sparse_mqa_attn(
        batch=args.batch,
        seq_len=args.seq_len,
        num_phys_blocks=args.num_phys_blocks,
        heads=args.heads,
        index_dim=args.index_dim,
        kv_block_size=args.kv_block_size,
        topk=args.topk,
        max_blocks=args.max_blocks,
        aicore_num=args.aicore_num,
    )
    print("Kernel Output Match!")
