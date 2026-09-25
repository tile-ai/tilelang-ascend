# ruff: noqa
import math

import torch
import torch.nn.functional as F

import tilelang
from tilelang import language as T

torch.set_default_device("npu")
torch.manual_seed(0)

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[4], pass_configs=PASS_CONFIGS)
def sparse_gqa_decode_varlen_mask(batch, heads, heads_kv, dim, dim_v, block_N, block_H, num_blocks, NI):
    """JIT-compiled Developer mode kernel (rev3: host pre-sorted K_sorted/V_sorted).

    Args (all compile-time Python constants):
        batch: batch size
        heads: number of Q attention heads
        heads_kv: number of K/V attention heads (GQA: heads // heads_kv = kv_group_num)
        dim: head dimension for Q/K
        dim_v: head dimension for V
        block_N: block size along KV sequence dimension (= block_size)
        block_H: block size along Q head dimension (padded to fractal minimum 16)
        num_blocks: total number of KV blocks (= max_cache_seqlen // block_N)
        NI: max_valid_blocks (compile-time constant, = ceil(num_blocks*(1-sparse)*1.25))
    """
    # Ascend uses exp (not exp2); scale keeps 1/sqrt(d) without log2(e) factor.
    sm_scale = (1.0 / dim) ** 0.5
    dtype = "float16"
    accum_dtype = "float"

    kv_group_num = heads // heads_kv
    valid_block_H = min(block_H, kv_group_num)

    # Q and Output are padded to padded_heads so each block uses full block_H
    # T.copy (AUTO_CV_COMBINE needs full block_H for correct vid splitting + L1 fractal).
    # Host side pads Q and crops Output to [:heads, :].
    padded_heads = (heads // valid_block_H) * block_H
    shape_q = [batch, padded_heads, dim]
    # rev3: K_sorted/V_sorted are host pre-sorted contiguous GM buffers.
    # Stored as 4D [batch, heads_kv, NI*block_N, dim] to match the verified
    # flash_attn_bshd_developer.py slice pattern: T.copy(K[bz,by,k*block_N:(k+1)*block_N,:], k_l1).
    # Kernel indexes via slice idx*block_N:(idx+1)*block_N (static affine offset, loop variable).
    shape_k_sorted = [batch, heads_kv, NI * block_N, dim]
    shape_v_sorted = [batch, heads_kv, NI * block_N, dim_v]
    shape_scoremask = [batch, heads_kv, NI, block_N]
    shape_o = [batch, padded_heads, dim_v]

    # 1D grid: each cid handles one (batch, query-head-group) pair
    block_num = batch * (heads // valid_block_H)

    @T.prim_func
    def main(
        Q: T.Tensor(shape_q, dtype),  # type: ignore
        K_sorted: T.Tensor(shape_k_sorted, dtype),  # type: ignore
        V_sorted: T.Tensor(shape_v_sorted, dtype),  # type: ignore
        ScoreMask: T.Tensor(shape_scoremask, accum_dtype),  # type: ignore
        Output: T.Tensor(shape_o, dtype),  # type: ignore
    ):
        with T.Kernel(block_num, threads=2, is_npu=True) as (cid):
            bid = cid // (heads // valid_block_H)
            hid = cid % (heads // valid_block_H)
            cur_kv_head = hid // (kv_group_num // valid_block_H)

            # ---- on-chip buffers (Developer: shared=L1/UB, fragment=L0C) ----
            # L1 (GEMM operands)
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim_v], dtype)
            acc_s_l1 = T.alloc_shared([block_H, block_N], dtype)  # P matrix (GEMM2 input)
            # L0C (GEMM output)
            acc_s_l0c = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o_l0c = T.alloc_fragment([block_H, dim_v], accum_dtype)
            # UB (element-wise + softmax)
            acc_s_ub = T.alloc_shared([block_H, block_N], accum_dtype)
            acc_s_ub_ = T.alloc_shared([block_H, block_N], accum_dtype)  # L0C->UB temp
            acc_o = T.alloc_shared([block_H, dim_v], accum_dtype)  # running accumulator
            acc_o_ub = T.alloc_shared([block_H, dim_v], accum_dtype)  # GEMM2 output (L0C->UB)
            acc_s_half = T.alloc_shared([block_H, block_N], dtype)  # fp32->fp16 P
            acc_o_half = T.alloc_shared([block_H, dim_v], dtype)  # fp32->fp16 output
            # 1D row buffers (matching flash_attn_bshd_developer.py scalar loop pattern;
            # broadcast approach with 2D [block_H,1] produced wrong per-row results)
            m_i = T.alloc_shared([block_H], accum_dtype)
            m_i_prev = T.alloc_shared([block_H], accum_dtype)
            sumexp = T.alloc_shared([block_H], accum_dtype)
            sumexp_i = T.alloc_shared([block_H], accum_dtype)
            pad_mask_1d = T.alloc_shared([1, block_N], accum_dtype)  # ScoreMask row [1, block_N]

            # === Load Q (GM -> L1), padded to block_H ===
            T.copy(Q[bid, hid * block_H : (hid + 1) * block_H, :], Q_shared)

            # === Init accumulators ===
            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -(2.0**30))

            # === Main loop: iterate valid blocks (NO conditional branch, NO dynamic gather) ===
            # AUTO_CV_COMBINE compatible: idx is T.serial loop variable -> static affine
            # offset. K_sorted/V_sorted loaded via static offset T.copy (rev3 core fix).
            # Invalid blocks contribute 0 (ScoreMask=-inf -> exp(-inf)=0).
            # Standard flash attention order (load K -> GEMM1 -> softmax -> load V -> GEMM2),
            # matching flash_attn_bshd_developer.py (NO pair pipeline).
            # Softmax uses scalar loop per-row ops (matching flash_attn_bshd_developer.py:96-119).
            for idx in T.serial(NI):
                # 1. Load K_sorted block (GM->L1, STATIC AFFINE OFFSET idx)
                #    Slice idx*block_N:(idx+1)*block_N matches flash_attn_bshd_developer.py:80
                #    pattern (loop variable offset slice -> compiler-analyzable affine
                #    expression -> AUTO_CV_COMBINE correctly analyzes buffer lifetime)
                T.copy(K_sorted[bid, cur_kv_head, idx * block_N : (idx + 1) * block_N, :], K_shared)

                # 2. GEMM1: Q x K_sorted^T -> L0C (init=True, overwrite each iter)
                T.gemm_v0(Q_shared, K_shared, acc_s_l0c, transpose_B=True, init=True)

                # 3. L0C -> UB (via temp + fill + add, matching flash_attn_bshd_developer.py:82-86)
                #    Direct T.copy(acc_s_l0c, acc_s_ub) fails with AUTO_CV_COMBINE buffer
                #    lifetime issues; intermediate step forces proper Cube/Vector sync.
                T.copy(acc_s_l0c, acc_s_ub_)
                T.tile.fill(acc_s_ub, 0.0)
                T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)

                # 4. Apply pre-computed ScoreMask (scalar loop per-row add, avoids broadcast)
                T.copy(ScoreMask[bid, cur_kv_head, idx : idx + 1, :], pad_mask_1d)
                for h_i in range(block_H):
                    T.tile.add(acc_s_ub[h_i, :], acc_s_ub[h_i, :], pad_mask_1d[0, :])

                # 5. Scale
                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)

                # 6. Online softmax (scalar loop, matching flash_attn_bshd_developer.py:85-102)
                # rolling max
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s_ub, m_i, dim=-1)
                T.tile.max(m_i, m_i, m_i_prev)
                T.tile.sub(m_i_prev, m_i_prev, m_i)
                T.tile.exp(m_i_prev, m_i_prev)

                # P = exp(acc_s - m_i) (scalar loop per-row sub)
                for h_i in range(block_H):
                    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
                T.tile.exp(acc_s_ub, acc_s_ub)

                # update sumexp
                T.reduce_sum(acc_s_ub, sumexp_i, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)
                T.tile.add(sumexp, sumexp, sumexp_i)

                # 7. Rescale acc_o (scalar loop per-row mul, matching flash_attn:113-114)
                for h_i in range(block_H):
                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])

                # 8. P -> L1 on-chip direct (GEMM2 input)
                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, acc_s_l1)

                # 9. Load V_sorted block (GM->L1, STATIC AFFINE OFFSET idx)
                #    Slice idx*block_N:(idx+1)*block_N (same pattern as K_sorted)
                T.copy(V_sorted[bid, cur_kv_head, idx * block_N : (idx + 1) * block_N, :], V_shared)

                # 10. GEMM2: P x V_sorted -> L0C (init=True, overwrite each iter)
                T.gemm_v0(acc_s_l1, V_shared, acc_o_l0c, init=True)

                # 11. L0C -> UB on-chip direct + UB accumulate
                T.copy(acc_o_l0c, acc_o_ub)
                T.tile.add(acc_o, acc_o, acc_o_ub)

            # === Normalize: O = acc_o / sumexp (scalar loop per-row div) ===
            # L0 guarantees sumexp >= 1 (at least one valid block with valid tokens).
            # Boundary (all-invalid) may produce NaN; host wrapper guards by returning zeros.
            for h_i in range(block_H):
                T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

            # === Output: fp32 -> fp16, UB -> GM ===
            T.copy(acc_o, acc_o_half)
            T.copy(acc_o_half, Output[bid, hid * block_H : (hid + 1) * block_H, :])

    return main


# ==================== Host-side two-stage preprocessing (rev3) ====================
def materialize_block_mask(block_mask, cache_seqlens, num_blocks, block_N, NI):
    """Stage 1: Pre-materialize block_mask + cache_seqlens into Indices + ScoreMask.

    block_mask: [batch, heads_kv, num_blocks] int8 (1=valid, 0=invalid)
    cache_seqlens: [batch] int32 (actual KV length per sample)
    Returns:
        Indices: [batch, heads_kv, NI] int32
            Compact list of valid block indices (block-level, not token-level).
            Padding indices (idx >= valid_count) are 0 (loads block 0 data,
            masked by ScoreMask=-inf so contributes 0 to softmax).
        ScoreMask: [batch, heads_kv, NI, block_N] fp32 (0.0 valid / -inf invalid)
            Encodes both index validity (padding idx -> -inf) and varlen padding
            (pos >= cache_seqlens -> -inf).
    """
    batch, heads_kv, _ = block_mask.shape
    device = block_mask.device
    Indices = torch.zeros((batch, heads_kv, NI), dtype=torch.int32, device=device)
    ScoreMask = torch.full((batch, heads_kv, NI, block_N), float("-inf"), dtype=torch.float32, device=device)
    for b in range(batch):
        cache_len = int(cache_seqlens[b].item())
        for h in range(heads_kv):
            valid_blocks = []
            for k in range(num_blocks):
                if bool(block_mask[b, h, k].item()) and k * block_N < cache_len:
                    valid_blocks.append(k)
            for idx, k in enumerate(valid_blocks[:NI]):
                Indices[b, h, idx] = k
                block_start = k * block_N
                for j in range(block_N):
                    if block_start + j < cache_len:
                        ScoreMask[b, h, idx, j] = 0.0
            # Padding indices (idx >= len(valid_blocks)): Indices=0 (default),
            # ScoreMask=-inf (default). pre_sort_kv copies block 0 data but -inf
            # mask makes exp(-inf)=0, contributing 0 to acc_o. No OOB (block 0 in range).
    return Indices, ScoreMask


def pre_sort_kv(K, V, Indices, block_N):
    """Stage 2 (rev3 NEW): Pre-sort K/V into contiguous GM buffers indexed by valid block order.

    Eliminates kernel-side gather (AUTO_CV_COMBINE buffer lifetime bug, Issue 3/4).
    Uses vectorized torch advanced indexing (no Python for-loops).

    K: [batch, max_cache_seqlen, heads_kv, dim] fp16
    V: [batch, max_cache_seqlen, heads_kv, dim_v] fp16
    Indices: [batch, heads_kv, NI] int32 (block indices from materialize_block_mask)
    block_N: int (block size)

    Returns:
        K_sorted: [batch, heads_kv, NI, block_N, dim] fp16 (contiguous)
        V_sorted: [batch, heads_kv, NI, block_N, dim_v] fp16 (contiguous)
    """
    batch, _, heads_kv, dim = K.shape
    _, _, _, dim_v = V.shape
    NI = Indices.shape[2]
    device = K.device

    # Build token-level indices: [batch, heads_kv, NI, block_N]
    # token_indices[b, h, idx, j] = Indices[b, h, idx] * block_N + j
    offsets = torch.arange(block_N, device=device, dtype=Indices.dtype)  # [block_N]
    token_indices = Indices.unsqueeze(-1) * block_N + offsets  # [batch, heads_kv, NI, block_N]

    # Advanced indexing: K[batch_idx, token_indices, kv_head_idx, :]
    # batch_idx broadcasts to [batch, heads_kv, NI, block_N]
    batch_idx = torch.arange(batch, device=device).view(batch, 1, 1, 1)
    kv_head_idx = torch.arange(heads_kv, device=device).view(1, heads_kv, 1, 1)

    K_sorted = K[batch_idx, token_indices, kv_head_idx, :]  # [batch, heads_kv, NI, block_N, dim]
    V_sorted = V[batch_idx, token_indices, kv_head_idx, :]  # [batch, heads_kv, NI, block_N, dim_v]

    # Reshape to 4D [batch, heads_kv, NI*block_N, dim] to match flash_attn_bshd_developer.py
    # slice pattern (T.copy with idx*block_N:(idx+1)*block_N offset).
    K_sorted = K_sorted.reshape(batch, heads_kv, NI * block_N, dim).contiguous()
    V_sorted = V_sorted.reshape(batch, heads_kv, NI * block_N, dim_v).contiguous()

    return K_sorted, V_sorted


def run_developer_kernel(Q, K, V, block_mask, cache_seqlens, batch, heads, heads_kv, dim, dim_v, block_size, NI, block_H=16):
    """Host-side wrapper: pads Q, runs two-stage preprocessing, invokes kernel, crops Output.

    Args:
        Q: [batch, heads, dim] fp16
        K: [batch, max_cache_seqlen, heads_kv, dim] fp16
        V: [batch, max_cache_seqlen, heads_kv, dim_v] fp16
        block_mask: [batch, heads_kv, num_blocks] int8
        cache_seqlens: [batch] int32
        batch, heads, heads_kv, dim, dim_v: shape params
        block_size: KV block size (= block_N)
        NI: max_valid_blocks (compile-time constant for this kernel)
        block_H: Q head block size (default 16)

    Returns:
        output: [batch, heads, dim_v] fp16 (on CPU)
    """
    max_cache_seqlen = K.shape[1]
    num_blocks = max_cache_seqlen // block_size

    kv_group_num = heads // heads_kv
    valid_block_H = min(block_H, kv_group_num)
    padded_heads = (heads // valid_block_H) * block_H

    # Stage 1: materialize_block_mask -> Indices + ScoreMask
    Indices, ScoreMask = materialize_block_mask(block_mask, cache_seqlens, num_blocks, block_size, NI)

    # Guard: all-invalid block case -> return zeros (avoid kernel NaN from sumexp=0)
    if not (ScoreMask == 0.0).any():
        return torch.zeros((batch, heads, dim_v), dtype=torch.float16, device="cpu")

    # Stage 2: pre_sort_kv -> K_sorted + V_sorted (rev3 core: eliminates kernel gather)
    K_sorted, V_sorted = pre_sort_kv(K, V, Indices, block_size)

    # Pad Q: each hid gets block_H rows, first valid_block_H are real data
    Q_padded = torch.zeros((batch, padded_heads, dim), dtype=Q.dtype, device=Q.device)
    for hid in range(heads // valid_block_H):
        Q_padded[:, hid * block_H : hid * block_H + valid_block_H, :] = Q[:, hid * valid_block_H : (hid + 1) * valid_block_H, :]

    kernel = sparse_gqa_decode_varlen_mask(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        dim_v=dim_v,
        block_N=block_size,
        block_H=block_H,
        num_blocks=num_blocks,
        NI=NI,
    )

    output_padded = kernel(Q_padded, K_sorted, V_sorted, ScoreMask)
    torch.npu.synchronize()

    # Crop output: extract valid_block_H rows per hid
    out_padded = output_padded.cpu()
    out = torch.zeros((batch, heads, dim_v), dtype=out_padded.dtype, device="cpu")
    for hid in range(heads // valid_block_H):
        out[:, hid * valid_block_H : (hid + 1) * valid_block_H, :] = out_padded[:, hid * block_H : hid * block_H + valid_block_H, :]

    return out


# ==================== Golden (PyTorch reference) ====================
def golden_sparse_gqa_decode_varlen_mask(query, key, value, block_mask, cache_seqlens, block_size):
    """PyTorch reference. Uses original block_mask + cache_seqlens + GQA semantics.
    Kernel uses host-pre-sorted K_sorted/V_sorted + ScoreMask; both are mathematically
    equivalent (Indices = compact block_mask, K_sorted/V_sorted = pre-sorted K/V by
    Indices, ScoreMask encodes block_mask + cache_seqlens mask info).

    query:  [batch, heads, dim]            (decode, no seq_len dim)
    key:    [batch, max_cache_seqlen, heads_kv, dim]
    value:  [batch, max_cache_seqlen, heads_kv, dim_v]
    block_mask: [batch, heads_kv, num_blocks] bool/int8
    cache_seqlens: [batch] int32
    Returns: [batch, heads, dim_v] float16
    """
    batch, heads, dim = query.shape
    heads_kv = key.shape[2]
    num_blocks = block_mask.shape[2]
    num_head_groups = heads // heads_kv
    scale = dim**0.5

    qf = query.float()
    kf = key.float()
    vf = value.float()

    # GQA rearrange: "b (h g) d -> b g h d" with g=num_head_groups
    q_g = qf.reshape(batch, heads_kv, num_head_groups, dim).permute(0, 2, 1, 3)  # [b, g, h_kv, dim]
    k_r = kf.permute(0, 2, 1, 3)  # [b, h_kv, seqlen, dim]
    v_r = vf.permute(0, 2, 1, 3)  # [b, h_kv, seqlen, dim_v]

    scores = torch.einsum("bghd,bhsd->bghs", q_g, k_r)  # [b, g, h_kv, seqlen]

    # Sparse block_mask: expand to token level
    sparse_mask = torch.zeros_like(scores)
    for b in range(batch):
        for h in range(heads_kv):
            for idx in range(num_blocks):
                if bool(block_mask[b, h, idx].item()):
                    sparse_mask[b, :, h, idx * block_size : (idx + 1) * block_size] = 1.0
    scores = scores.masked_fill(sparse_mask == 0, float("-inf"))

    # Varlen padding mask: pos >= cache_seqlens[b] -> -inf
    range_len = torch.arange(scores.shape[-1], device=query.device).unsqueeze(0)  # [1, seqlen]
    pad_mask = range_len >= cache_seqlens.unsqueeze(1)  # [b, seqlen]
    scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))

    attention = F.softmax(scores / scale, dim=-1)  # [b, g, h_kv, seqlen]
    out = torch.einsum("bghs,bhsd->bghd", attention, v_r)  # [b, g, h_kv, dim_v]
    # rearrange "b g h d -> b (h g) d"
    out = out.permute(0, 2, 1, 3).reshape(batch, heads, -1)  # [b, heads, dim_v]
    return out.to(torch.float16)


def _smoke_test():
    """Quick smoke test: minimal case, verifies kernel compiles and runs."""
    batch, heads, heads_kv = 1, 16, 8
    dim, dim_v = 128, 128
    max_cache_seqlen, block_size = 128, 128
    sparse_ratio = 0.0
    block_H = 16
    num_blocks = max_cache_seqlen // block_size
    NI = int(math.ceil(num_blocks * (1 - sparse_ratio) * 1.25))

    device = "npu"
    dtype = torch.float16
    Q = torch.randn((batch, heads, dim), dtype=dtype, device=device)
    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=dtype, device=device)
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=dtype, device=device)
    cache_seqlens = torch.tensor([max_cache_seqlen], dtype=torch.int32, device=device)
    block_mask = torch.ones((batch, heads_kv, num_blocks), dtype=torch.int8, device=device)

    out = run_developer_kernel(
        Q,
        K,
        V,
        block_mask,
        cache_seqlens,
        batch,
        heads,
        heads_kv,
        dim,
        dim_v,
        block_size,
        NI,
        block_H,
    )
    ref = golden_sparse_gqa_decode_varlen_mask(Q, K, V, block_mask, cache_seqlens, block_size)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3)
    max_diff = (ref.cpu().float() - out.cpu().float()).abs().max().item()
    print(f"[SMOKE_PASS] output shape={tuple(out.shape)}, max_diff={max_diff:.6e}")
    # CI gate (bench_test.sh matches "TEST PASSED!" case-insensitively)
    print("Test Passed!")


if __name__ == "__main__":
    tilelang.disable_cache()
    _smoke_test()
