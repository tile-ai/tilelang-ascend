"""DeepSeek MLA (Multi-Latent Attention) Decode kernel for Ascend NPU.

Migrated from tilelang CUDA example to tilelang-ascend.
Implements flash-attention for MLA decode, where Q has two components
(Q and Q_pe) whose attention scores are summed before softmax.

Supports two modes:
  - num_split=1: single kernel processes the full KV sequence (default)
  - num_split>1: split kernel processes KV chunks in parallel, then a
    combine kernel reduces partial results via log-sum-exp

Key adaptations for Ascend NPU:
  - Cube core (gemm_v0) computes QK^T and Q_pe K_pe^T, and PV matmul
  - Vector core (T.tile.*) handles online softmax and output accumulation
  - Workspace tensors mediate Cube <-> Vector communication
  - Auto-sync pass configs handle Cube/Vector synchronization
"""

import argparse

import tilelang
from tilelang import DataType, language as T
import torch

torch.set_default_device("npu")
torch.manual_seed(0)
tilelang.disable_cache()

NUM_STAGES = 2
L0_MAX_SIZE = 64 * 1024  # 64 KB

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(
    out_idx=[4],
    workspace_idx=[5, 6, 7, 8],
    pass_configs=pass_configs,
)
def flashattn_no_split(
    batch,
    heads,
    kv_head_num,
    seqlen_kv,
    dim,
    pe_dim,
    block_N,
    block_H,
    num_split,
    softmax_scale,
):
    assert num_split == 1, "Use flashattn_split_kernel for num_split > 1"
    assert kv_head_num == 1, "kv_head_num must be 1"

    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert heads % VALID_BLOCK_H == 0, "heads must be divisible by VALID_BLOCK_H"
    assert VALID_BLOCK_H % 2 == 0, "VALID_BLOCK_H must be even"
    assert seqlen_kv % block_N == 0, "seqlen_kv must be divisible by block_N"

    block_num = batch * (heads // VALID_BLOCK_H)

    n_num = max((block_N * dim * DataType(dtype).bits // 8 + L0_MAX_SIZE - 1) // L0_MAX_SIZE, 1)
    assert block_N % n_num == 0, "block_N must be divisible by n_num"
    assert dim % n_num == 0, "dim must be divisible by n_num"
    block_K = block_N // n_num
    block_D = dim // n_num
    num_stages = min((seqlen_kv + block_N - 1) // block_N, NUM_STAGES)
    v_block = VALID_BLOCK_H // 2  # Compute at outer scope as concrete Python int

    @T.prim_func
    def main_no_split(
        Q: T.Tensor([batch, heads, dim], dtype),  # type: ignore
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),  # type: ignore
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),  # type: ignore
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),  # type: ignore
        Output: T.Tensor([batch, heads, dim], dtype),  # type: ignore
        # Workspace buffers (auto-allocated via workspace_idx, double-buffered for pipelining)
        workspace_qk: T.Tensor([2, block_num, VALID_BLOCK_H, block_N], accum_dtype),  # QK scores (fp32)
        workspace_pe: T.Tensor([2, block_num, VALID_BLOCK_H, block_N], accum_dtype),  # Q_pe*K_pe scores (fp32)
        workspace_attn: T.Tensor([2, block_num, VALID_BLOCK_H, block_N], dtype),  # softmax scores (fp16)
        workspace_pv: T.Tensor([2, block_num, VALID_BLOCK_H, dim], accum_dtype),  # PV output (fp32)
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            hid = cid % (heads // VALID_BLOCK_H)
            bid = cid // (heads // VALID_BLOCK_H)
            cur_kv_head = 0  # kv_head_num == 1

            # ---- Cube core buffers (L1 / L0C) ----
            q_l1 = T.alloc_L1([VALID_BLOCK_H, dim], dtype)
            q_pe_l1 = T.alloc_L1([VALID_BLOCK_H, pe_dim], dtype)
            kv_l1 = T.alloc_L1([block_K, dim], dtype)  # K tiles for QK gemm
            k_pe_l1 = T.alloc_L1([block_N, pe_dim], dtype)  # K_pe tiles for PE gemm
            v_l1 = T.alloc_L1([block_N, block_D], dtype)  # V tiles for PV gemm
            attn_l1 = T.alloc_L1([VALID_BLOCK_H, block_N], dtype)

            acc_s_l0c = T.alloc_L0C([VALID_BLOCK_H, block_K], accum_dtype)  # QK gemm output
            acc_pe_l0c = T.alloc_L0C([VALID_BLOCK_H, block_N], accum_dtype)  # PE gemm output
            acc_o_l0c = T.alloc_L0C([VALID_BLOCK_H, block_D], accum_dtype)  # PV gemm output

            # ---- Vector core buffers (UB) ----

            acc_o = T.alloc_ub([v_block, dim], accum_dtype)
            scores_max = T.alloc_ub([v_block], accum_dtype)
            scores_max_prev = T.alloc_ub([v_block], accum_dtype)
            scores_max_2d = T.alloc_ub([v_block, block_N], accum_dtype)
            acc_s_ub = T.alloc_ub([v_block, block_N], accum_dtype)
            qk_ub = T.alloc_ub([v_block, block_N], accum_dtype)
            pe_ub = T.alloc_ub([v_block, block_N], accum_dtype)
            sumexp = T.alloc_ub([v_block], accum_dtype)
            sumexp_i_ub = T.alloc_ub([v_block], accum_dtype)
            acc_s_half = T.alloc_ub([v_block, block_N], dtype)
            acc_o_ub = T.alloc_ub([v_block, dim], accum_dtype)
            acc_o_half = T.alloc_ub([v_block, dim], dtype)

            # ---- Load Q and Q_pe to L1 ----
            T.copy(Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], q_l1)
            T.copy(Q_pe[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], q_pe_l1)

            # ---- Initialize accumulators ----
            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(scores_max, -(2.0**30))

            loop_range = T.ceildiv(seqlen_kv, block_N)

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                # ============ Cube: compute attention scores ============

                # QK^T gemm (tiled along seq dim when n_num > 1)
                for n_i in T.serial(n_num):
                    kv_start = k * block_N + n_i * block_K
                    T.copy(
                        KV[bid, kv_start : kv_start + block_K, cur_kv_head, :],
                        kv_l1,
                    )
                    T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True)
                    T.copy(
                        acc_s_l0c,
                        workspace_qk[k % 2, cid, :, n_i * block_K : (n_i + 1) * block_K],
                    )

                # Q_pe * K_pe^T gemm (full block_N, not tiled)
                T.copy(
                    K_pe[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :],
                    k_pe_l1,
                )
                T.gemm_v0(q_pe_l1, k_pe_l1, acc_pe_l0c, transpose_B=True, init=True)
                T.copy(acc_pe_l0c, workspace_pe[k % 2, cid, :, :])

                # ============ Vector: online softmax ============

                T.tile.fill(acc_s_ub, 0.0)
                T.copy(scores_max, scores_max_prev)

                # Load QK scores into UB
                T.copy(
                    workspace_qk[k % 2, cid, vid * v_block : vid * v_block + v_block, :],
                    qk_ub,
                )
                T.tile.add(acc_s_ub, acc_s_ub, qk_ub)

                # Load and add PE scores
                T.copy(
                    workspace_pe[k % 2, cid, vid * v_block : vid * v_block + v_block, :],
                    pe_ub,
                )
                T.tile.add(acc_s_ub, acc_s_ub, pe_ub)

                # Scale by softmax_scale (natural exp, no log2(e) factor)
                T.tile.mul(acc_s_ub, acc_s_ub, softmax_scale)

                # Online softmax: update running max
                T.reduce_max(acc_s_ub, scores_max, dim=-1)
                T.tile.max(scores_max, scores_max, scores_max_prev)

                # Rescale factor: exp(old_max - new_max)
                T.tile.sub(scores_max_prev, scores_max_prev, scores_max)
                T.tile.exp(scores_max_prev, scores_max_prev)

                # Subtract new max and exp
                T.tile.broadcast(scores_max_2d, scores_max)
                T.tile.sub(acc_s_ub, acc_s_ub, scores_max_2d)
                T.tile.exp(acc_s_ub, acc_s_ub)

                # Sum of exp values
                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, scores_max_prev)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)

                # Rescale running output accumulator
                for h_i in range(v_block):
                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], scores_max_prev[h_i])

                # Cast softmax scores to fp16 and write back for PV gemm
                T.copy(acc_s_ub, acc_s_half)
                T.copy(
                    acc_s_half,
                    workspace_attn[k % 2, cid, vid * v_block : vid * v_block + v_block, :],
                )
                if vid == 0:
                    T.copy(workspace_attn[k % 2, cid, :, :], attn_l1)

                # ============ Cube: PV gemm ============
                for n_i in T.serial(n_num):
                    T.copy(
                        KV[bid, k * block_N : (k + 1) * block_N, cur_kv_head, n_i * block_D : (n_i + 1) * block_D],
                        v_l1,
                    )
                    T.gemm_v0(attn_l1, v_l1, acc_o_l0c, init=True)
                    T.copy(
                        acc_o_l0c,
                        workspace_pv[k % 2, cid, :, n_i * block_D : (n_i + 1) * block_D],
                    )

                # ============ Vector: accumulate PV output ============
                T.copy(
                    workspace_pv[k % 2, cid, vid * v_block : vid * v_block + v_block, :],
                    acc_o_ub,
                )
                T.tile.add(acc_o, acc_o, acc_o_ub)

            # ---- Final normalization: acc_o /= sumexp ----
            for h_i in range(v_block):
                T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

            # ---- Write output ----
            T.copy(acc_o, acc_o_half)
            out_start = hid * VALID_BLOCK_H + vid * v_block
            T.copy(
                acc_o_half,
                Output[bid, out_start : out_start + v_block, :],
            )

    return main_no_split


@tilelang.jit(
    out_idx=[8, 9],
    workspace_idx=[4, 5, 6, 7],
    pass_configs=pass_configs,
)
def flashattn_split_kernel(
    batch,
    heads,
    kv_head_num,
    seqlen_kv,
    dim,
    pe_dim,
    block_N,
    block_H,
    num_split,
    softmax_scale,
):
    """Split-phase kernel: flash attention on a chunk of the KV sequence.

    Each block processes seqlen_kv // num_split tokens and writes its partial
    output and log-sum-exp to workspace buffers for the combine kernel.
    """
    assert kv_head_num == 1, "kv_head_num must be 1"
    assert seqlen_kv % num_split == 0, "seqlen_kv must be divisible by num_split"

    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert heads % VALID_BLOCK_H == 0, "heads must be divisible by VALID_BLOCK_H"
    assert VALID_BLOCK_H % 2 == 0, "VALID_BLOCK_H must be even"

    base_block_num = batch * (heads // VALID_BLOCK_H)
    split_block_num = base_block_num * num_split
    seq_per_split = seqlen_kv // num_split
    assert seq_per_split % block_N == 0, "seq_per_split must be divisible by block_N"

    n_num = max((block_N * dim * DataType(dtype).bits // 8 + L0_MAX_SIZE - 1) // L0_MAX_SIZE, 1)
    assert block_N % n_num == 0, "block_N must be divisible by n_num"
    assert dim % n_num == 0, "dim must be divisible by n_num"
    block_K = block_N // n_num
    block_D = dim // n_num
    num_stages = min((seq_per_split + block_N - 1) // block_N, NUM_STAGES)
    v_block = VALID_BLOCK_H // 2

    @T.prim_func
    def main_split(
        Q: T.Tensor([batch, heads, dim], dtype),  # type: ignore
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),  # type: ignore
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),  # type: ignore
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),  # type: ignore
        # Workspace buffers (auto-allocated via workspace_idx, double-buffered for pipelining)
        workspace_qk: T.Tensor([2, split_block_num, VALID_BLOCK_H, block_N], accum_dtype),  # QK scores (fp32)
        workspace_pe: T.Tensor([2, split_block_num, VALID_BLOCK_H, block_N], accum_dtype),  # Q_pe*K_pe scores (fp32)
        workspace_attn: T.Tensor([2, split_block_num, VALID_BLOCK_H, block_N], dtype),  # softmax scores (fp16)
        workspace_pv: T.Tensor([2, split_block_num, VALID_BLOCK_H, dim], accum_dtype),  # PV output (fp32)
        # Combine workspace (auto-allocated as outputs via out_idx)
        ws_glse: T.Tensor([split_block_num, VALID_BLOCK_H], accum_dtype),
        ws_partial: T.Tensor([split_block_num, VALID_BLOCK_H, dim], accum_dtype),
    ):
        with T.Kernel(split_block_num, is_npu=True) as (cid, vid):
            # Decompose global block id into split index and base block
            bz = cid // base_block_num
            base_cid = cid % base_block_num
            hid = base_cid % (heads // VALID_BLOCK_H)
            bid = base_cid // (heads // VALID_BLOCK_H)
            cur_kv_head = 0  # kv_head_num == 1

            # ---- Cube core buffers (L1 / L0C) ----
            q_l1 = T.alloc_L1([VALID_BLOCK_H, dim], dtype)
            q_pe_l1 = T.alloc_L1([VALID_BLOCK_H, pe_dim], dtype)
            kv_l1 = T.alloc_L1([block_K, dim], dtype)
            k_pe_l1 = T.alloc_L1([block_N, pe_dim], dtype)
            v_l1 = T.alloc_L1([block_N, block_D], dtype)
            attn_l1 = T.alloc_L1([VALID_BLOCK_H, block_N], dtype)

            acc_s_l0c = T.alloc_L0C([VALID_BLOCK_H, block_K], accum_dtype)
            acc_pe_l0c = T.alloc_L0C([VALID_BLOCK_H, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([VALID_BLOCK_H, block_D], accum_dtype)

            # ---- Vector core buffers (UB) ----
            acc_o = T.alloc_ub([v_block, dim], accum_dtype)
            scores_max = T.alloc_ub([v_block], accum_dtype)
            scores_max_prev = T.alloc_ub([v_block], accum_dtype)
            scores_max_2d = T.alloc_ub([v_block, block_N], accum_dtype)
            acc_s_ub = T.alloc_ub([v_block, block_N], accum_dtype)
            qk_ub = T.alloc_ub([v_block, block_N], accum_dtype)
            pe_ub = T.alloc_ub([v_block, block_N], accum_dtype)
            sumexp = T.alloc_ub([v_block], accum_dtype)
            sumexp_i_ub = T.alloc_ub([v_block], accum_dtype)
            acc_s_half = T.alloc_ub([v_block, block_N], dtype)
            acc_o_ub = T.alloc_ub([v_block, dim], accum_dtype)

            # ---- Load Q and Q_pe to L1 ----
            T.copy(Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], q_l1)
            T.copy(Q_pe[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], q_pe_l1)

            # ---- Initialize accumulators ----
            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(scores_max, -(2.0**30))

            split_kv_start = bz * seq_per_split
            loop_range = T.ceildiv(seq_per_split, block_N)

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                # ============ Cube: compute attention scores ============
                for n_i in T.serial(n_num):
                    kv_start = split_kv_start + k * block_N + n_i * block_K
                    T.copy(
                        KV[bid, kv_start : kv_start + block_K, cur_kv_head, :],
                        kv_l1,
                    )
                    T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True)
                    T.copy(
                        acc_s_l0c,
                        workspace_qk[k % 2, cid, :, n_i * block_K : (n_i + 1) * block_K],
                    )
                pe_start = split_kv_start + k * block_N
                T.copy(
                    K_pe[bid, pe_start : pe_start + block_N, cur_kv_head, :],
                    k_pe_l1,
                )
                T.gemm_v0(q_pe_l1, k_pe_l1, acc_pe_l0c, transpose_B=True, init=True)
                T.copy(acc_pe_l0c, workspace_pe[k % 2, cid, :, :])

                # ============ Vector: online softmax ============
                T.tile.fill(acc_s_ub, 0.0)
                T.copy(scores_max, scores_max_prev)
                T.copy(
                    workspace_qk[k % 2, cid, vid * v_block : vid * v_block + v_block, :],
                    qk_ub,
                )
                T.tile.add(acc_s_ub, acc_s_ub, qk_ub)
                T.copy(
                    workspace_pe[k % 2, cid, vid * v_block : vid * v_block + v_block, :],
                    pe_ub,
                )
                T.tile.add(acc_s_ub, acc_s_ub, pe_ub)
                T.tile.mul(acc_s_ub, acc_s_ub, softmax_scale)
                T.reduce_max(acc_s_ub, scores_max, dim=-1)
                T.tile.max(scores_max, scores_max, scores_max_prev)
                T.tile.sub(scores_max_prev, scores_max_prev, scores_max)
                T.tile.exp(scores_max_prev, scores_max_prev)
                T.tile.broadcast(scores_max_2d, scores_max)
                T.tile.sub(acc_s_ub, acc_s_ub, scores_max_2d)
                T.tile.exp(acc_s_ub, acc_s_ub)
                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, scores_max_prev)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)
                for h_i in range(v_block):
                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], scores_max_prev[h_i])
                T.copy(acc_s_ub, acc_s_half)
                T.copy(
                    acc_s_half,
                    workspace_attn[k % 2, cid, vid * v_block : vid * v_block + v_block, :],
                )
                if vid == 0:
                    T.copy(workspace_attn[k % 2, cid, :, :], attn_l1)

                # ============ Cube: PV gemm ============
                for n_i in T.serial(n_num):
                    v_start = split_kv_start + k * block_N
                    T.copy(
                        KV[bid, v_start : v_start + block_N, cur_kv_head, n_i * block_D : (n_i + 1) * block_D],
                        v_l1,
                    )
                    T.gemm_v0(attn_l1, v_l1, acc_o_l0c, init=True)
                    T.copy(
                        acc_o_l0c,
                        workspace_pv[k % 2, cid, :, n_i * block_D : (n_i + 1) * block_D],
                    )

                # ============ Vector: accumulate PV output ============
                T.copy(
                    workspace_pv[k % 2, cid, vid * v_block : vid * v_block + v_block, :],
                    acc_o_ub,
                )
                T.tile.add(acc_o, acc_o, acc_o_ub)

            # ---- Normalize partial output for combine ----
            # Combine expects partial_k = O_k / sumexp_k
            for h_i in range(v_block):
                T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

            # ---- Compute LSE = ln(sumexp) + max_score ----
            T.tile.ln(sumexp, sumexp)
            T.tile.add(sumexp, sumexp, scores_max)

            T.copy(sumexp, ws_glse[cid, vid * v_block : vid * v_block + v_block])
            T.copy(acc_o, ws_partial[cid, vid * v_block : vid * v_block + v_block, :])

    return main_split


@tilelang.jit(
    out_idx=[2],
    pass_configs=pass_configs,
)
def flashattn_combine_kernel(
    batch,
    heads,
    kv_head_num,
    seqlen_kv,
    dim,
    pe_dim,
    block_N,
    block_H,
    num_split,
):
    """Combine-phase kernel: reduce partial results across splits via LSE.

    Vector-core only (no Cube ops). For each (batch, head_group) block:
    1. Load num_split LSE values, find max LSE
    2. Weight each normalized partial by exp(lse_k - lse_max)
    3. Accumulate and divide by sum of weights
    """
    assert kv_head_num == 1, "kv_head_num must be 1"

    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)

    block_num = batch * (heads // VALID_BLOCK_H)
    split_block_num = block_num * num_split
    v_block = VALID_BLOCK_H // 2

    @T.prim_func
    def main_combine(
        ws_glse: T.Tensor([split_block_num, VALID_BLOCK_H], accum_dtype),
        ws_partial: T.Tensor([split_block_num, VALID_BLOCK_H, dim], accum_dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            hid = cid % (heads // VALID_BLOCK_H)
            bid = cid // (heads // VALID_BLOCK_H)

            # ---- Vector core buffers (UB) ----
            acc_o = T.alloc_ub([v_block, dim], accum_dtype)
            lse_val = T.alloc_ub([v_block], accum_dtype)
            lse_max = T.alloc_ub([v_block], accum_dtype)
            logsum = T.alloc_ub([v_block], accum_dtype)
            weight = T.alloc_ub([v_block], accum_dtype)
            partial_ub = T.alloc_ub([v_block, dim], accum_dtype)
            acc_o_half = T.alloc_ub([v_block, dim], dtype)

            # ---- Step 1: find max LSE across splits ----
            T.tile.fill(lse_max, -(2.0**30))
            for k in T.serial(num_split):
                lse_idx = k * block_num + cid
                T.copy(
                    ws_glse[lse_idx, vid * v_block : vid * v_block + v_block],
                    lse_val,
                )
                T.tile.max(lse_max, lse_max, lse_val)

            # ---- Step 2: compute logsum and weighted accumulation ----
            T.tile.fill(logsum, 0.0)
            T.tile.fill(acc_o, 0.0)
            for k in T.serial(num_split):
                lse_idx = k * block_num + cid
                # Load LSE and compute weight = exp(lse[k] - lse_max)
                T.copy(
                    ws_glse[lse_idx, vid * v_block : vid * v_block + v_block],
                    weight,
                )
                T.tile.sub(weight, weight, lse_max)
                T.tile.exp(weight, weight)
                T.tile.add(logsum, logsum, weight)

                # Load normalized partial and accumulate: acc_o += partial * weight
                T.copy(
                    ws_partial[lse_idx, vid * v_block : vid * v_block + v_block, :],
                    partial_ub,
                )
                for h_i in range(v_block):
                    T.tile.mul(partial_ub[h_i, :], partial_ub[h_i, :], weight[h_i])
                T.tile.add(acc_o, acc_o, partial_ub)

            # Normalize: acc_o /= logsum
            for h_i in range(v_block):
                T.tile.div(acc_o[h_i, :], acc_o[h_i, :], logsum[h_i])

            # ---- Write final output ----
            T.copy(acc_o, acc_o_half)
            out_start = hid * VALID_BLOCK_H + vid * v_block
            T.copy(
                acc_o_half,
                Output[bid, out_start : out_start + v_block, :],
            )

    return main_combine


def flashattn(
    batch,
    heads,
    kv_head_num,
    seqlen_kv,
    dim,
    pe_dim,
    block_N,
    block_H,
    num_split,
    softmax_scale,
):
    """Public API: dispatches to no-split or split+combine kernels.

    Returns a callable that takes (q, q_pe, kv, k_pe) and returns the output tensor.
    """
    if num_split <= 1:
        kernel = flashattn_no_split(
            batch,
            heads,
            kv_head_num,
            seqlen_kv,
            dim,
            pe_dim,
            block_N,
            block_H,
            num_split,
            softmax_scale,
        )
        return kernel
    else:
        split_kernel = flashattn_split_kernel(
            batch,
            heads,
            kv_head_num,
            seqlen_kv,
            dim,
            pe_dim,
            block_N,
            block_H,
            num_split,
            softmax_scale,
        )
        combine_kernel = flashattn_combine_kernel(
            batch,
            heads,
            kv_head_num,
            seqlen_kv,
            dim,
            pe_dim,
            block_N,
            block_H,
            num_split,
        )

        def wrapper(q, q_pe, kv, k_pe):
            # Phase 1: split — returns [ws_glse, ws_partial]
            ws_glse, ws_partial = split_kernel(q, q_pe, kv, k_pe)

            # Phase 2: combine — reduce partial results via LSE.
            return combine_kernel(ws_glse, ws_partial)

        return wrapper


def ref_program(q, q_pe, kv, k_pe):
    """Reference implementation using pure PyTorch (no einops).

    Args:
        q:     [batch, heads, dim]
        q_pe:  [batch, heads, pe_dim]
        kv:    [batch, seqlen_kv, kv_head_num, dim]
        k_pe:  [batch, seqlen_kv, kv_head_num, pe_dim]

    Returns:
        output: [batch, heads, dim]
    """
    dim = q.shape[-1]
    pe_dim = q_pe.shape[-1]
    batch = q.shape[0]
    heads = q.shape[1]
    kv_head_num = kv.shape[2]
    num_head_groups = heads // kv_head_num

    scale = (dim + pe_dim) ** 0.5

    q_f = q.float()
    q_pe_f = q_pe.float()
    kv_f = kv.float()
    k_pe_f = k_pe.float()

    # Reshape for grouped attention
    # q: [batch, heads, dim] -> [batch, groups, kv_heads, dim]
    q_r = q_f.view(batch, num_head_groups, kv_head_num, dim)
    q_pe_r = q_pe_f.view(batch, num_head_groups, kv_head_num, pe_dim)
    # kv: [batch, seqlen_kv, kv_heads, dim] -> [batch, kv_heads, seqlen_kv, dim]
    kv_r = kv_f.permute(0, 2, 1, 3)
    k_pe_r = k_pe_f.permute(0, 2, 1, 3)

    # Concatenate Q and Q_pe along dim
    query = torch.cat([q_r, q_pe_r], dim=-1)  # [B, G, H, D+Dpe]
    key = torch.cat([kv_r, k_pe_r], dim=-1)  # [B, H, S, D+Dpe]

    # Attention scores: [B, G, H, D+Dpe] x [B, H, S, D+Dpe] -> [B, G, H, S]
    scores = torch.einsum("bghd,bhsd->bghs", query, key) / scale

    # Softmax
    attention = torch.softmax(scores, dim=-1)  # [B, G, H, S]

    # Weighted sum with V (= KV in MLA latent representation)
    # [B, G, H, S] x [B, H, S, D] -> [B, G, H, D]
    out = torch.einsum("bghs,bhsd->bghd", attention, kv_r)

    # Reshape: [B, G, H, D] -> [B, G*H, D] = [B, heads, dim]
    out = out.reshape(batch, heads, dim)
    return out.to(torch.float16)


def main(
    batch=1,
    heads=128,
    kv_heads=1,
    kv_ctx=8192,
    dim=512,
    pe_dim=64,
    num_split=1,
    no_check=False,
):
    BLOCK_N = 64
    BLOCK_H = min(64, heads // kv_heads)
    softmax_scale = (dim + pe_dim) ** -0.5

    kernel = flashattn(batch, heads, kv_heads, kv_ctx, dim, pe_dim, BLOCK_N, BLOCK_H, num_split, softmax_scale)

    # Build inputs
    q = torch.randn(batch, heads, dim, dtype=torch.float16)
    q_pe = torch.randn(batch, heads, pe_dim, dtype=torch.float16)
    kv = torch.randn(batch, kv_ctx, kv_heads, dim, dtype=torch.float16)
    k_pe = torch.randn(batch, kv_ctx, kv_heads, pe_dim, dtype=torch.float16)

    torch.npu.synchronize()
    print(f"init successful! (num_split={num_split})")

    output = kernel(q, q_pe, kv, k_pe)

    if not no_check:
        torch.npu.synchronize()
        ref_output = ref_program(q, q_pe, kv, k_pe)
        torch.npu.synchronize()
        torch.testing.assert_close(output, ref_output, rtol=5e-2, atol=5e-2)
        print("Test Passed!")
    else:
        torch.npu.synchronize()
        print("Reference check skipped.")

    # Simple latency measurement
    from tilelang.profiler import do_bench

    latency = do_bench(lambda: kernel(q, q_pe, kv, k_pe))
    qk_flops = 2 * batch * heads * kv_ctx * (dim + pe_dim)
    pv_flops = 2 * batch * heads * kv_ctx * dim
    total_flops = qk_flops + pv_flops
    print(f"Latency: {latency:.3f} ms")
    print(f"TFlops: {total_flops / latency * 1e-9:.2f} TFlops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek MLA Decode on Ascend NPU")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--heads", type=int, default=128, help="q heads number")
    parser.add_argument("--kv_heads", type=int, default=1, help="kv heads number")
    parser.add_argument("--kv_ctx", type=int, default=8192, help="kv context length")
    parser.add_argument("--dim", type=int, default=512, help="head dim")
    parser.add_argument("--pe_dim", type=int, default=64, help="pe head dim")
    parser.add_argument("--num-split", type=int, default=1, help="number of KV sequence splits (1 = no split)")
    parser.add_argument("--no-check", action="store_true", help="skip correctness check")
    args = parser.parse_args()

    main(args.batch, args.heads, args.kv_heads, args.kv_ctx, args.dim, args.pe_dim, args.num_split, args.no_check)
