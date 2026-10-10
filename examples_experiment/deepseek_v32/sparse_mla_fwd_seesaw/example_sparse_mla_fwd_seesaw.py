"""Sparse MLA Forward (DeepSeek V3.2) kernel for Ascend NPU.

Math:
    S = Q_nope @ K_nope^T + Q_pe @ K_pe^T        # 576 = 512 (nope) + 64 (pe)
    P = softmax(S * sm_scale + causal_mask)
    O = P @ V_nope                                # V = K[..., :512]
    Lse = log(sumexp) + max_scaled

Sparsity: ``Indices[b, s, 0, :topk]`` selects top-K KV positions per query.
    KV is split into nope (512) and pe (64) L1 buffers to avoid L1 slicing
    in T.gemm_v0 (which requires full L1 buffer inputs).

    GEMM3 (P@V) runs at M = 2 * block_H (one call per head pair) instead of
    M = block_H (one call per head): the B operand (kv_nope, 128KB per stage)
    is reloaded from L1 to L0B once per head pair instead of once per head,
    halving the dominant L1->L0 traffic.

For the layered test suite (L0/L1/L2/Boundary), see
``test_sparse_mla_fwd_seesaw.py`` in the same directory.
"""

import torch
import tilelang
from tilelang import language as T

_developer_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[3, 4], workspace_idx=[5, 6, 7], pass_configs=_developer_pass_configs)
def sparse_mla_fwd(
    batch,
    seq_len,
    seq_len_kv,
    heads,
    dim,
    tail_dim,
    topk,
    kv_stride,
    kv_group,
    core_num,
    q_start_index_s,
    sm_scale=None,
    block_I=64,
    block_H=32,
    hid_tiles=2,
    num_stages=1,
    dim_split=1,
):
    """Sparse MLA forward kernel.

    Args:
        batch:           batch size
        seq_len:         query sequence length
        seq_len_kv:      KV cache sequence length
        heads:           number of query heads (H)
        dim:             value/nope head dim (512 for DeepSeek V3 MLA)
        tail_dim:        RoPE/pe head dim (64 for DeepSeek V3 MLA)
        topk:            number of selected KV positions per query
        kv_stride:       KV storage stride (usually 1)
        kv_group:        number of KV groups (must be 1)
        core_num:        number of AI Cube cores
        q_start_index_s: query start position in global sequence
        sm_scale:        softmax scale (1/sqrt(dim+tail_dim))
        block_I:         KV block size for attention
        block_H:         query head block size
        hid_tiles:       number of hid_blocks per tile
        num_stages:      multi-buffer pipeline depth
        dim_split:       split output dim for UB-fit when hid_tiles>2
    """
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5
    dtype = "bfloat16"
    accum_dtype = "float"
    indices_dtype = "int32"

    assert kv_group == 1, "kv_group must be 1 (HKV=1 for DeepSeek V3 MLA)"
    assert heads % block_H == 0, f"heads ({heads}) must be divisible by block_H ({block_H})"
    assert topk % block_I == 0, f"topk ({topk}) must be divisible by block_I ({block_I})"
    assert block_H % 2 == 0, (
        f"block_H ({block_H}) must be even: the two vid lanes split each head "
        "block into row halves of block_H // 2, an odd block_H silently drops "
        "the last row of every head block"
    )
    assert (dim + tail_dim) == 576
    assert dim % dim_split == 0, "dim must be divisible by dim_split"
    assert hid_tiles % 2 == 0, (
        f"hid_tiles ({hid_tiles}) must be even: GEMM3 merges head blocks into "
        "pairs (M = 2 * block_H) to halve the kv_nope L0B reloads; an odd "
        "hid_tiles would leave the last head block unpaired"
    )

    dim_per_split = dim // dim_split

    num_h_blocks = heads // block_H
    assert num_h_blocks % hid_tiles == 0, (
        f"tiling divisibility violated: num_h_blocks ({num_h_blocks} = heads "
        f"({heads}) // block_H ({block_H})) must be divisible by hid_tiles "
        f"({hid_tiles}), otherwise the trailing head blocks are silently "
        "skipped by the grid decomposition (and hid_tiles > num_h_blocks "
        "yields num_h_block_pairs = 0 and a divide-by-zero in the tile map)"
    )
    num_h_block_pairs = num_h_blocks // hid_tiles
    NI = topk // block_I
    hm = block_H // 2
    concat_H = hid_tiles * block_H
    pair_H = 2 * block_H
    num_hid_pairs = hid_tiles // 2
    ws_slots = num_stages
    num_outer = T.ceildiv(NI, num_stages)
    total_tiles = batch * num_h_block_pairs * seq_len
    waves = T.ceildiv(total_tiles, core_num)
    NEG_INF = -(2.0**30)

    @T.prim_func
    def main(
        Q: T.Tensor([batch, seq_len, heads, dim + tail_dim], dtype),
        KV: T.Tensor([batch, seq_len_kv, kv_group, dim + tail_dim], dtype),
        SafeIdx: T.Tensor([batch, seq_len, kv_group, topk], indices_dtype),
        Output: T.Tensor([batch, seq_len, heads, dim], dtype),
        Lse: T.Tensor([batch, seq_len, heads], accum_dtype),
        workspace_1: T.Tensor([core_num, ws_slots, concat_H, block_I], accum_dtype),
        workspace_2: T.Tensor([core_num, ws_slots, concat_H, block_I], dtype),
        workspace_3: T.Tensor([core_num, ws_slots, concat_H, dim], dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            q_nope_concat = T.alloc_L1([concat_H, dim], dtype)
            q_pe_concat = T.alloc_L1([concat_H, tail_dim], dtype)
            kv_nope_l1 = T.alloc_L1([num_stages, block_I, dim], dtype)
            kv_pe_l1 = T.alloc_L1([num_stages, block_I, tail_dim], dtype)
            acc_s_l1 = T.alloc_L1([pair_H, block_I], dtype)
            acc_s_l0c = T.alloc_L0C([concat_H, block_I], accum_dtype)
            acc_o_l0c = T.alloc_L0C([pair_H, dim], accum_dtype)

            indices_ub = T.alloc_ub([block_I], indices_dtype)
            mask_col = T.alloc_ub([block_I], accum_dtype)
            cmp_mask = T.alloc_ub([block_I], "uint8")
            zero_buf = T.alloc_ub([block_I], accum_dtype)
            acc_o_all = T.alloc_ub([hid_tiles, dim_split, hm, dim_per_split], accum_dtype)
            acc_o = T.alloc_ub([hm, dim_per_split], accum_dtype)
            m_i_all = T.alloc_ub([hid_tiles, hm], accum_dtype)
            logsum_all = T.alloc_ub([hid_tiles, hm], accum_dtype)
            m_i = T.alloc_ub([hm], accum_dtype)
            m_i_prev = T.alloc_ub([hm], accum_dtype)
            logsum = T.alloc_ub([hm], accum_dtype)
            acc_s_ub = T.alloc_ub([hm, block_I], accum_dtype)
            acc_s_ub_ = T.alloc_ub([hm, block_I], accum_dtype)
            mask_2d = T.alloc_ub([hm, block_I], accum_dtype)
            acc_s_half = T.alloc_ub([hm, block_I], dtype)
            acc_o_ub = T.alloc_ub([hm, dim_per_split], accum_dtype)
            acc_o_half = T.alloc_ub([hm, dim_per_split], dtype)
            scale_ub = T.alloc_ub([hm, dim_per_split], accum_dtype)
            r_factors = T.alloc_ub([num_stages, hid_tiles, hm], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, hid_tiles, hm], accum_dtype)

            v_row = vid * hm

            T.tile.fill(zero_buf, 0.0)

            # Cube: GEMM1 (Q@K^T) + GEMM3 (P@V)
            for w in T.serial(waves):
                tile_id = core_num * w + cid
                bid = tile_id // (num_h_block_pairs * seq_len)
                rem = tile_id % (num_h_block_pairs * seq_len)
                hid_block = rem // seq_len
                s_i = rem % seq_len

                if bid < batch:
                    for hid_local in T.serial(hid_tiles):
                        h_start = (hid_block * hid_tiles + hid_local) * block_H
                        T.copy(
                            Q[bid, s_i, h_start : h_start + block_H, :dim],
                            q_nope_concat[hid_local * block_H : (hid_local + 1) * block_H, :],
                        )
                        T.copy(
                            Q[bid, s_i, h_start : h_start + block_H, dim:],
                            q_pe_concat[hid_local * block_H : (hid_local + 1) * block_H, :],
                        )

                    for k_outer in T.serial(num_outer):
                        _remaining = NI - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        for i in T.serial(batch_iters):
                            i_i = k_outer * num_stages + i
                            for bi_i in T.serial(block_I):
                                idx = SafeIdx[bid, s_i, 0, i_i * block_I + bi_i]
                                T.copy(KV[bid, idx, 0, :dim], kv_nope_l1[i, bi_i, :])
                                T.copy(KV[bid, idx, 0, dim:], kv_pe_l1[i, bi_i, :])

                        for i in T.serial(batch_iters):
                            T.gemm_v0(
                                q_nope_concat,
                                kv_nope_l1[i, :, :],
                                acc_s_l0c,
                                transpose_B=True,
                                init=True,
                            )
                            T.gemm_v0(
                                q_pe_concat,
                                kv_pe_l1[i, :, :],
                                acc_s_l0c,
                                transpose_B=True,
                                init=False,
                                kL0Size=32,
                            )
                            T.copy(acc_s_l0c, workspace_1[cid, i, :, :])

                        for i in T.serial(batch_iters):
                            for hid_pair in T.serial(num_hid_pairs):
                                p_off = hid_pair * pair_H
                                T.copy(
                                    workspace_2[cid, i, p_off : p_off + pair_H, :],
                                    acc_s_l1,
                                )
                                T.gemm_v0(
                                    acc_s_l1,
                                    kv_nope_l1[i, :, :],
                                    acc_o_l0c,
                                    init=True,
                                    kL0Size=32,
                                )
                                T.copy(
                                    acc_o_l0c,
                                    workspace_3[cid, i, p_off : p_off + pair_H, :],
                                )

            # Vector: online softmax + O accumulate
            for w in T.serial(waves):
                tile_id = core_num * w + cid
                bid = tile_id // (num_h_block_pairs * seq_len)
                rem = tile_id % (num_h_block_pairs * seq_len)
                hid_block = rem // seq_len
                s_i = rem % seq_len

                if bid < batch:
                    q_i = q_start_index_s + s_i
                    max_kv_i = (q_i + 1 - kv_stride) // kv_stride

                    for hid_local in T.serial(hid_tiles):
                        for d in T.serial(dim_split):
                            T.tile.fill(acc_o, 0.0)
                            T.copy(acc_o, acc_o_all[hid_local, d, :, :])
                        T.tile.fill(m_i, -(2.0**30))
                        T.copy(m_i, m_i_all[hid_local, :])
                        T.tile.fill(logsum, 0.0)
                        T.copy(logsum, logsum_all[hid_local, :])

                    T.copy(SafeIdx[bid, s_i, 0, 0:block_I], indices_ub)
                    T.tile.cast(mask_col, indices_ub, "CAST_NONE", block_I)
                    T.tile.compare(cmp_mask, mask_col, max_kv_i, "LE")
                    T.tile.select(mask_col, cmp_mask, zero_buf, NEG_INF, "VSEL_TENSOR_SCALAR_MODE")
                    T.tile.broadcast(mask_2d, mask_col, axis=0)

                    for k_outer in T.serial(num_outer):
                        _remaining = NI - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        for i in T.serial(batch_iters):
                            i_i = k_outer * num_stages + i

                            for hid_local in T.serial(hid_tiles):
                                h_off = hid_local * block_H

                                T.copy(m_i_all[hid_local, :], m_i)
                                T.copy(m_i, m_i_prev)

                                T.copy(
                                    workspace_1[cid, i, h_off + v_row : h_off + v_row + hm, :],
                                    acc_s_ub,
                                )

                                T.tile.add(acc_s_ub, acc_s_ub, mask_2d)

                                T.reduce_max(acc_s_ub, m_i, dim=-1)
                                T.tile.max(m_i, m_i, m_i_prev)
                                T.tile.sub(m_i_prev, m_i_prev, m_i)
                                T.copy(m_i_prev, r_factors[i, hid_local, :])

                                T.tile.broadcast(acc_s_ub_, m_i, axis=1)
                                T.tile.sub(acc_s_ub, acc_s_ub, acc_s_ub_)
                                T.tile.exp(acc_s_ub, acc_s_ub)

                                T.reduce_sum(acc_s_ub, m_i_prev, dim=-1)
                                T.copy(m_i_prev, sumexp_is[i, hid_local, :])

                                T.tile.cast(acc_s_half, acc_s_ub, "CAST_RINT", hm * block_I)
                                T.copy(
                                    acc_s_half,
                                    workspace_2[cid, i, h_off + v_row : h_off + v_row + hm, :],
                                )

                                T.copy(m_i, m_i_all[hid_local, :])

                            next_i_i = T.if_then_else(
                                i < batch_iters - 1,
                                i_i + 1,
                                T.if_then_else(k_outer < num_outer - 1, (k_outer + 1) * num_stages, i_i),
                            )
                            T.copy(
                                SafeIdx[bid, s_i, 0, next_i_i * block_I : (next_i_i + 1) * block_I],
                                indices_ub,
                            )
                            T.tile.cast(mask_col, indices_ub, "CAST_NONE", block_I)
                            T.tile.compare(cmp_mask, mask_col, max_kv_i, "LE")
                            T.tile.select(mask_col, cmp_mask, zero_buf, NEG_INF, "VSEL_TENSOR_SCALAR_MODE")
                            T.tile.broadcast(mask_2d, mask_col, axis=0)

                        for i in T.serial(batch_iters):
                            for hid_local in T.serial(hid_tiles):
                                h_off = hid_local * block_H

                                T.copy(logsum_all[hid_local, :], logsum)

                                T.copy(r_factors[i, hid_local, :], m_i_prev)
                                T.tile.exp(m_i_prev, m_i_prev)
                                T.tile.mul(logsum, logsum, m_i_prev)

                                T.tile.broadcast(scale_ub, m_i_prev, axis=1)

                                for d in T.serial(dim_split):
                                    d_start = d * dim_per_split
                                    T.copy(acc_o_all[hid_local, d, :, :], acc_o)
                                    T.tile.mul(acc_o, acc_o, scale_ub)

                                    T.copy(
                                        workspace_3[
                                            cid,
                                            i,
                                            h_off + v_row : h_off + v_row + hm,
                                            d_start : d_start + dim_per_split,
                                        ],
                                        acc_o_half,
                                    )
                                    T.tile.cast(acc_o_ub, acc_o_half, "CAST_NONE", hm * dim_per_split)
                                    T.tile.add(acc_o, acc_o, acc_o_ub)

                                    T.copy(acc_o, acc_o_all[hid_local, d, :, :])

                                T.copy(sumexp_is[i, hid_local, :], m_i_prev)
                                T.tile.add(logsum, logsum, m_i_prev)

                                T.copy(logsum, logsum_all[hid_local, :])

                    for hid_local in T.serial(hid_tiles):
                        global_hid = hid_block * hid_tiles + hid_local
                        h_start = global_hid * block_H

                        T.copy(logsum_all[hid_local, :], logsum)
                        T.copy(m_i_all[hid_local, :], m_i)

                        T.copy(logsum, m_i_prev)
                        T.tile.ln(m_i_prev, m_i_prev)
                        T.tile.add(m_i_prev, m_i_prev, m_i)
                        T.copy(m_i_prev, Lse[bid, s_i, h_start + v_row : h_start + v_row + hm])

                        T.tile.max(logsum, logsum, 1e-30)
                        T.tile.broadcast(acc_o_ub, logsum, axis=1)

                        for d in T.serial(dim_split):
                            d_start = d * dim_per_split
                            T.copy(acc_o_all[hid_local, d, :, :], acc_o)
                            T.tile.div(acc_o, acc_o, acc_o_ub)

                            T.tile.cast(acc_o_half, acc_o, "CAST_RINT", hm * dim_per_split)
                            T.copy(
                                acc_o_half,
                                Output[
                                    bid,
                                    s_i,
                                    h_start + v_row : h_start + v_row + hm,
                                    d_start : d_start + dim_per_split,
                                ],
                            )

    return main


def _get_core_num():
    """Query the device's cube core count at runtime."""
    try:
        return int(torch.npu.get_device_properties(0).cube_core_num)
    except Exception:
        return 20


def sparse_mla_fwd_interface(q, kv, indices, q_start_index_s, kv_stride=1, sm_scale=None, return_kernel=False):
    """Host-side interface for sparse_mla_fwd."""
    assert q.dtype == torch.bfloat16, f"q must be bfloat16, got {q.dtype}"
    assert kv.dtype == torch.bfloat16, f"kv must be bfloat16, got {kv.dtype}"
    assert indices.dtype == torch.int32, f"indices must be int32, got {indices.dtype}"
    assert q.device.type == "npu", f"q must be on NPU, got {q.device}"
    assert kv.device.type == "npu", f"kv must be on NPU, got {kv.device}"
    assert indices.device.type == "npu", f"indices must be on NPU, got {indices.device}"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    assert q.ndim == 4, f"q must be 4D [batch, seq, heads, dqk], got ndim={q.ndim}"
    assert kv.ndim == 4, f"kv must be 4D [batch, seq_kv, hk, dqk], got ndim={kv.ndim}"
    assert indices.ndim == 4, f"indices must be 4D, got ndim={indices.ndim}"

    batch, seq_len, heads, dqk = q.shape
    _, seq_len_kv, hk, _ = kv.shape
    _, _, _, topk = indices.shape

    dim = 512
    tail_dim = dqk - dim
    assert dqk == 576, f"dqk must be 576, got {dqk}"
    assert kv.shape[-1] == dqk, f"kv last dim must match q ({dqk}), got {kv.shape[-1]}"
    assert hk == 1, f"hk must be 1, got {hk}"
    assert heads % 64 == 0, (
        f"heads must be a multiple of 64 (supported: 64, 128, 192, ...; got "
        f"{heads}). Head tiling uses block_H=32 with hid_tiles in {{2, 4}}, and "
        "the head-block grid is lossless only when heads % 64 == 0; other "
        "multiples of 16 silently truncate head blocks or break divisibility"
    )
    assert kv_stride >= 1, f"kv_stride must be a positive integer (>= 1), got {kv_stride}"
    assert dim % 16 == 0, f"dim must be 16-aligned, got {dim}"
    assert indices.shape[0] == batch and indices.shape[1] == seq_len

    # Degenerate inputs must be rejected before indices.min()/indices.max():
    # the reductions raise RuntimeError (not AssertionError) on an empty tensor.
    assert seq_len >= 1, f"seq_len must be >= 1, got {seq_len} (an empty query sequence is meaningless; the tile map divides by seq_len)"
    assert seq_len_kv >= 1, f"seq_len_kv must be >= 1, got {seq_len_kv} (indices.clamp(0, seq_len_kv - 1) is illegal when seq_len_kv == 0)"
    assert q_start_index_s >= 0, (
        f"q_start_index_s must be >= 0, got {q_start_index_s} (the global query start position must be non-negative)"
    )
    assert topk >= 1, (
        f"topk must be >= 1, got {topk} (topk == 0 passes the '0 % 128 == 0' "
        "BLOCK_I derivation and causes an out-of-bounds prefetch in the "
        "Vector stage during lowering)"
    )
    assert indices.shape[2] == 1, f"indices kv_group dim must be 1 (HKV == 1), got {indices.shape[2]}"
    idx_min = int(indices.min())
    assert idx_min >= 0, f"indices contain negative values (min = {idx_min}); legal range is [0, {seq_len_kv})"
    idx_max = int(indices.max())
    assert idx_max < seq_len_kv, (
        f"indices out of range (max = {idx_max} >= seq_len_kv = {seq_len_kv}); "
        "this operator requires every index to be a legal KV position in "
        "[0, seq_len_kv). Out-of-range padding/sentinel values are no longer "
        "supported: when q_start is late they get clamped to seq_len_kv - 1 "
        "and silently (and wrongly) counted into the softmax"
    )

    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

    CORE_NUM = _get_core_num()
    BLOCK_I = 128 if topk % 128 == 0 else 64
    BLOCK_H = 32
    HID_TILES = 4 if (heads // BLOCK_H) % 4 == 0 else 2
    DIM_SPLIT = 4 if HID_TILES == 4 else 1
    NUM_STAGES = 2

    q_scaled = q * sm_scale
    safe_idx = indices.clamp(0, seq_len_kv - 1).to(torch.int32)

    kernel = sparse_mla_fwd(
        batch,
        seq_len,
        seq_len_kv,
        heads,
        dim,
        tail_dim,
        topk,
        kv_stride,
        hk,
        CORE_NUM,
        q_start_index_s,
        1.0,
        BLOCK_I,
        BLOCK_H,
        HID_TILES,
        NUM_STAGES,
        DIM_SPLIT,
    )

    if return_kernel:
        return kernel

    out, lse = kernel(q_scaled, kv, safe_idx)
    return out, lse


def smoke_test():
    """Minimal smoke test: verify kernel runs and output shapes are correct."""
    torch.manual_seed(42)

    B, S, SKV, H, HKV, DQK, TOPK = 1, 4, 64, 128, 1, 576, 64
    Q_START_S = 63
    KV_STRIDE = 1

    q = torch.randn(B, S, H, DQK, dtype=torch.bfloat16) / 10
    kv = torch.randn(B, SKV, HKV, DQK, dtype=torch.bfloat16) / 10
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.zeros(B, S, HKV, TOPK, dtype=torch.int32)
    for bi in range(B):
        for si in range(S):
            perm = torch.randperm(SKV)[:TOPK]
            indices[bi, si, 0, :TOPK] = perm

    tl_out, tl_lse = sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), Q_START_S, KV_STRIDE)

    assert tl_out.shape == (B, S, H, 512)
    assert tl_lse.shape == (B, S, H)
    assert not torch.isnan(tl_out).any()
    assert not torch.isnan(tl_lse).any()

    print(f"[smoke] output shape: {tuple(tl_out.shape)}")
    print(f"[smoke] lse shape:    {tuple(tl_lse.shape)}")
    print("[smoke] no NaN in outputs")
    print("Test Passed!")


if __name__ == "__main__":
    tilelang.disable_cache()
    smoke_test()
