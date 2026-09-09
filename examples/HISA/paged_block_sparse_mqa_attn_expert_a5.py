"""Paged block-sparse MQA decode kernel for A5 PTO.

This file intentionally follows the corrected A3 implementation.  The
four-MMA schedule, L1/L0 tiling, per-block readiness events, AIV early/late
order, post-processing, context mask, output, reference and driver are kept
the same.

Only the score transport changes: A3 uses ``L0C -> GM workspace -> UB``;
A5 uses four targeted ``FixPipe -> AIV UB`` copies.  Each copy immediately
publishes its own readiness event, and AIV bridges that event from MTE2 to V
before reading the direct score tile.  A5 additionally keeps at most 28
physical blocks resident and uses two score slots to overlap C/V work across
logical tasks; a one-wave B1 specialization statically folds to one slot.
"""

import tilelang
from tilelang import language as T
import argparse
import torch

tilelang.disable_cache()

A5_9579_AICORE_NUM = 28
PERSISTENT_SCORE_SLOTS = 2
A5_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_PTO_USE_PIPE_IN_CV_COPY: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


@tilelang.jit(
    out_idx=[3],
    target="pto",
    platform="A5",
    pass_configs=A5_PASS_CONFIGS,
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
    aicore_num: int = A5_9579_AICORE_NUM,
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
    score_slots = PERSISTENT_SCORE_SLOTS if grid_size > kernel_grid_size else 1

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

    @T.macro
    def process_score(score_ub, logits_ub, weights_fp32, logits_event):
        """AIV post-processing shared by both persistent score slots."""
        T.tile.relu(score_ub, score_ub)
        T.tile.row_expand_mul(score_ub, score_ub, weights_fp32)
        T.wait_flag("MTE3", "V", logits_event)
        T.reduce_sum(score_ub, logits_ub, dim=0, clear=True)

    @T.prim_func
    def kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),
        KvCache: T.Tensor(kv_cache_shape, dtype),
        TopKBlockIndex: T.Tensor(topk_index_shape, index_dtype),
        Logits: T.Tensor(logits_shape, accum_dtype),
        Weights: T.Tensor(weights_shape, dtype),
        ContextLens: T.Tensor(context_lens_shape, index_dtype),
        BlockTables: T.Tensor(block_tables_shape, index_dtype),
    ):
        with T.Kernel(kernel_grid_size, is_npu=True) as (bx, by):
            # [A5 persistent] Two score slots let C produce the next logical
            # task while V consumes the current one. B1 statically uses slot0.
            s_ub_e0 = T.alloc_ub([H_per_block, kv], accum_dtype)
            s_ub_l0 = T.alloc_ub([H_per_block, kv], accum_dtype)
            s_ub_e1 = T.alloc_ub([H_per_block, kv], accum_dtype)
            s_ub_l1 = T.alloc_ub([H_per_block, kv], accum_dtype)
            logits_e = T.alloc_ub([1, kv], accum_dtype)
            logits_l = T.alloc_ub([1, kv], accum_dtype)
            weights_ub = T.alloc_ub([heads], dtype)
            weights = T.alloc_ub([heads], accum_dtype)

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
                    score_slot = (task // kernel_grid_size) % score_slots
                    b = token // seq_len

                    # Keep the A3 issue order: resolve each K at its DMA.
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

                    T.wait_flag("MTE2", "MTE1", SIG_K_L1_0)
                    T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                    # L0A is shared by all four MMAs. Both previous ping-pong
                    # users must release it before the next task overwrites Q.
                    T.wait_flag("M", "MTE1", SIG_L0AB_0)
                    T.wait_flag("M", "MTE1", SIG_L0AB_1)
                    T.copy(q_l1, l0a)
                    T.copy(k_l1_0, l0b_0, transpose=True)
                    T.set_flag("MTE1", "M", SIG_L0AB_0)

                    T.wait_flag("MTE1", "M", SIG_L0AB_0)
                    T.wait_flag("FIX", "M", SIG_L0C_0)
                    T.mma(l0a, l0b_0, l0c_0, init=True)
                    T.set_flag("M", "MTE1", SIG_L0AB_0)
                    T.set_flag("M", "FIX", SIG_L0C_0)

                    T.wait_flag("M", "FIX", SIG_L0C_0)
                    # [A5 persistent] Reusing slot N requires both AIVs to
                    # publish its PIPE_V release to AIC PIPE_FIX first.
                    if task >= 2 * kernel_grid_size:
                        T.wait_cross_flag(FLAG_SCORE_RELEASE + score_slot, "FIX")
                    if score_slot != 0:
                        T.copy_op.copy_cv_experiment(
                            l0c_0,
                            s_ub_e1,
                            T.copy_op.CopyCVMode.SingleVec0,
                        )
                    else:
                        T.copy_op.copy_cv_experiment(
                            l0c_0,
                            s_ub_e0,
                            T.copy_op.CopyCVMode.SingleVec0,
                        )
                    T.set_flag("FIX", "M", SIG_L0C_0)
                    if score_slot == 0:
                        T._src_code(f"set_intra_block(PIPE_FIX, {FLAG_BLK0});")
                    else:
                        T._src_code(f"set_intra_block(PIPE_FIX, {FLAG_BLK0 + 4});")

                    T.wait_flag("MTE2", "MTE1", SIG_K_L1_1)
                    T.copy(k_l1_1, l0b_1, transpose=True)
                    T.set_flag("MTE1", "M", SIG_L0AB_1)
                    T.wait_flag("MTE1", "M", SIG_L0AB_1)
                    T.wait_flag("FIX", "M", SIG_L0C_1)
                    T.mma(l0a, l0b_1, l0c_1, init=True)
                    T.set_flag("M", "MTE1", SIG_L0AB_1)
                    T.set_flag("M", "FIX", SIG_L0C_1)

                    T.wait_flag("M", "FIX", SIG_L0C_1)
                    if score_slot != 0:
                        T.copy_op.copy_cv_experiment(
                            l0c_1,
                            s_ub_e1,
                            T.copy_op.CopyCVMode.SingleVec1,
                        )
                    else:
                        T.copy_op.copy_cv_experiment(
                            l0c_1,
                            s_ub_e0,
                            T.copy_op.CopyCVMode.SingleVec1,
                        )
                    T.set_flag("FIX", "M", SIG_L0C_1)
                    if score_slot == 0:
                        T._src_code(f"set_intra_block(PIPE_FIX, {FLAG_BLK1 + 16});")
                    else:
                        T._src_code(f"set_intra_block(PIPE_FIX, {FLAG_BLK1 + 20});")

                    T.wait_flag("MTE2", "MTE1", SIG_K_L1_2)
                    T.wait_flag("M", "MTE1", SIG_L0AB_0)
                    T.copy(k_l1_2, l0b_0, transpose=True)
                    T.set_flag("MTE1", "M", SIG_L0AB_0)
                    T.wait_flag("MTE1", "M", SIG_L0AB_0)
                    T.wait_flag("FIX", "M", SIG_L0C_2)
                    T.mma(l0a, l0b_0, l0c_2, init=True)
                    T.set_flag("M", "MTE1", SIG_L0AB_0)
                    T.set_flag("M", "FIX", SIG_L0C_2)

                    T.wait_flag("M", "FIX", SIG_L0C_2)
                    if score_slot != 0:
                        T.copy_op.copy_cv_experiment(
                            l0c_2,
                            s_ub_l1,
                            T.copy_op.CopyCVMode.SingleVec0,
                        )
                    else:
                        T.copy_op.copy_cv_experiment(
                            l0c_2,
                            s_ub_l0,
                            T.copy_op.CopyCVMode.SingleVec0,
                        )
                    T.set_flag("FIX", "M", SIG_L0C_2)
                    if score_slot == 0:
                        T._src_code(f"set_intra_block(PIPE_FIX, {FLAG_BLK2});")
                    else:
                        T._src_code(f"set_intra_block(PIPE_FIX, {FLAG_BLK2 + 4});")

                    T.wait_flag("MTE2", "MTE1", SIG_K_L1_3)
                    T.wait_flag("M", "MTE1", SIG_L0AB_1)
                    T.copy(k_l1_3, l0b_1, transpose=True)
                    T.set_flag("MTE1", "M", SIG_L0AB_1)
                    T.wait_flag("MTE1", "M", SIG_L0AB_1)
                    T.wait_flag("FIX", "M", SIG_L0C_3)
                    T.mma(l0a, l0b_1, l0c_3, init=True)
                    T.set_flag("M", "MTE1", SIG_L0AB_1)
                    T.set_flag("M", "FIX", SIG_L0C_3)

                    T.wait_flag("M", "FIX", SIG_L0C_3)
                    if score_slot != 0:
                        T.copy_op.copy_cv_experiment(
                            l0c_3,
                            s_ub_l1,
                            T.copy_op.CopyCVMode.SingleVec1,
                        )
                    else:
                        T.copy_op.copy_cv_experiment(
                            l0c_3,
                            s_ub_l0,
                            T.copy_op.CopyCVMode.SingleVec1,
                        )
                    T.set_flag("FIX", "M", SIG_L0C_3)
                    if score_slot == 0:
                        T._src_code(f"set_intra_block(PIPE_FIX, {FLAG_BLK3 + 16});")
                    else:
                        T._src_code(f"set_intra_block(PIPE_FIX, {FLAG_BLK3 + 20});")

                T.wait_flag("M", "MTE1", SIG_L0AB_0)
                T.wait_flag("M", "MTE1", SIG_L0AB_1)
                T.wait_flag("FIX", "M", SIG_L0C_0)
                T.wait_flag("FIX", "M", SIG_L0C_1)
                T.wait_flag("FIX", "M", SIG_L0C_2)
                T.wait_flag("FIX", "M", SIG_L0C_3)

            with T.Scope("V"):
                T.set_flag("V", "MTE2", SIG_W_UB)
                T.set_flag("MTE3", "V", SIG_LOGITS_E)
                T.set_flag("MTE3", "V", SIG_LOGITS_L)

                for task in T.Persistent([grid_size], kernel_grid_size, bx):
                    token = task // topk_groups
                    n_outer = task - token * topk_groups
                    n_i_base = n_outer * 4
                    score_slot = (task // kernel_grid_size) % score_slots
                    b_v = token // seq_len
                    cu_k_e_max = ContextLens[b_v]

                    T.wait_flag("V", "MTE2", SIG_W_UB)
                    T.copy(Weights[token, :], weights_ub)
                    T.set_flag("MTE2", "V", SIG_W_UB)
                    T.wait_flag("MTE2", "V", SIG_W_UB)
                    T.copy(weights_ub, weights)
                    T.pipe_barrier("v")
                    T.set_flag("V", "MTE2", SIG_W_UB)

                    early_flag = score_slot * 4 + FLAG_BLK0 + by
                    early_n_i = n_i_base + by
                    T.wait_cross_flag(early_flag, "MTE2")
                    T.set_flag("MTE2", "V", SIG_S_E)
                    T.wait_flag("MTE2", "V", SIG_S_E)
                    if score_slot != 0:
                        process_score(s_ub_e1, logits_e, weights, SIG_LOGITS_E)
                    else:
                        process_score(s_ub_e0, logits_e, weights, SIG_LOGITS_E)

                    # full: no-op; invalid: Vector fill; partial: Scalar tail.
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

                    late_flag = score_slot * 4 + FLAG_BLK2 + by
                    late_n_i = n_i_base + 2 + by
                    T.wait_cross_flag(late_flag, "MTE2")
                    T.set_flag("MTE2", "V", SIG_S_L)

                    T.set_flag("V", "MTE3", SIG_LOGITS_E)
                    T.wait_flag("V", "MTE3", SIG_LOGITS_E)
                    T.copy(logits_e[0, :], Logits[token, early_n_i, :])
                    T.set_flag("MTE3", "V", SIG_LOGITS_E)

                    T.wait_flag("MTE2", "V", SIG_S_L)
                    if score_slot != 0:
                        process_score(s_ub_l1, logits_l, weights, SIG_LOGITS_L)
                    else:
                        process_score(s_ub_l0, logits_l, weights, SIG_LOGITS_L)

                    # [A5 persistent] Score UB is free after late reduction;
                    # both AIVs publish the same logical slot release.
                    if task + 2 * kernel_grid_size < grid_size:
                        T.set_cross_flag("V", FLAG_SCORE_RELEASE + score_slot)

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

                    T.set_flag("V", "MTE3", SIG_LOGITS_L)
                    T.wait_flag("V", "MTE3", SIG_LOGITS_L)
                    T.copy(logits_l[0, :], Logits[token, late_n_i, :])
                    T.set_flag("MTE3", "V", SIG_LOGITS_L)

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
    aicore_num: int = A5_9579_AICORE_NUM,
):
    """Test paged sparse MQA attention with the A5 persistent grid."""
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
    parser = argparse.ArgumentParser(description="Paged Sparse MQA Attention — A5 FixPipe-to-UB")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--num_phys_blocks", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--index_dim", type=int, default=128)
    parser.add_argument("--kv_block_size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--max_blocks", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--aicore_num", type=int, default=A5_9579_AICORE_NUM)
    args = parser.parse_args()

    torch.set_default_device("npu")
    torch.manual_seed(42)
    tilelang.disable_cache()
    assert args.topk % 4 == 0

    print("=" * 60)
    print("Paged Sparse MQA — A5 FixPipe-to-UB")
    print("=" * 60)
    print(f"  batch={args.batch}, heads={args.heads}, index_dim={args.index_dim}")
    print(f"  kv_block_size={args.kv_block_size}, topk={args.topk}")
    logical_grid = args.batch * args.seq_len * args.topk // 4
    print(f"  grid={logical_grid} logical tasks -> {min(logical_grid, args.aicore_num)} persistent blocks")
    print("  mask: full bypass + invalid Vector fill + partial Scalar tail")
    print("  C2V: targeted FixPipe-to-UB; persistent C/V overlap")
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
