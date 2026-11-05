import tilelang
from tilelang import language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[3, 4], workspace_idx=[5, 6, 7, 8, 9], pass_configs=pass_configs)
def sparse_mla_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
):

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    assert dim == tilelang.math.next_power_of_2(dim)
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim)
    assert is_causal
    assert topk % block_I == 0

    sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 if sm_scale is None else sm_scale

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != head_kv:
        assert kv_group == 1

    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64
    v_block = H_per_block // 2

    block_num = seq_len * REPLICATE_H * batch * kv_group

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        workspace_1: T.Tensor([block_num, BI, D], dtype),  # KV dim segment
        workspace_2: T.Tensor([block_num, BI, D_tail], dtype),  # KV tail segment
        workspace_3: T.Tensor([block_num, H_per_block, BI], accum_dtype),  # acc_s
        workspace_4: T.Tensor([block_num, H_per_block, BI], dtype),  # S_shared bf16
        workspace_5: T.Tensor([block_num, H_per_block, D], accum_dtype),  # acc_o
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len * REPLICATE_H)
            by = cid // (seq_len * REPLICATE_H) % batch
            bz = cid // (seq_len * REPLICATE_H) // batch % kv_group

            b_i = by
            g_i = bz
            s_i = bx // REPLICATE_H

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            # Cube-side L1 buffers
            q_l1 = T.alloc_L1([H_per_block, D], dtype)
            q_tail_l1 = T.alloc_L1([H_per_block, D_tail], dtype)
            # L1 double-buffer for MTE2||M overlap — 2 slots so MTE2 can
            # load the next iteration's KV while M computes the current
            # GEMM. AUTO_SYNC=True lets the compiler insert the pipeline
            # sync points.
            # Capacity: kv_l1 [2,64,512] bf16=128KB, kv_tail_l1 [2,64,64]
            # bf16=16KB, acc_s_l1 [2,64,64] bf16=16KB → +64KB over single-buffer,
            # total L1 ~232KB < 512KB.
            kv_l1 = T.alloc_L1([2, BI, D], dtype)
            kv_tail_l1 = T.alloc_L1([2, BI, D_tail], dtype)
            acc_s_l1 = T.alloc_L1([2, H_per_block, BI], dtype)

            # L0C accumulators
            acc_s_l0c = T.alloc_L0C([H_per_block, BI], accum_dtype)
            acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

            # Vector-side UB buffers
            acc_o = T.alloc_ub([v_block, D], accum_dtype)
            sumexp = T.alloc_ub([v_block], accum_dtype)
            m_i = T.alloc_ub([v_block], accum_dtype)
            indices_ub = T.alloc_ub([BI], indices_dtype)
            indices_ub_float = T.alloc_ub([BI], accum_dtype)
            mask_ub = T.alloc_ub([BI // 8], "uint8")
            # Multi-row gather buffers — collect all BI//2 rows into UB
            # first, then bulk-write to workspace. Eliminates 62 per-row
            # MTE3 writes + scalar loop overhead.
            kv_ub_gather = T.alloc_ub([BI // 2, D], dtype)
            kv_tail_ub_gather = T.alloc_ub([BI // 2, D_tail], dtype)
            # Double-buffered kv_full_ub for MTE2||V overlap: 2 slots so
            # MTE2 can load row[i+1] while V copies row[i]. The MTE2→V
            # set_flag provides memory visibility for the V-side T.copy
            # (UB→UB) reads.
            # Buffer: [2, 576] bf16 = 2.3KB, +1.15KB over single-buffer.
            kv_full_ub = T.alloc_ub([2, D + D_tail], dtype)
            acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
            m_i_prev = T.alloc_ub([v_block], accum_dtype)
            acc_s_ub_ = T.alloc_ub([v_block, BI], accum_dtype)
            sumexp_i_ub = T.alloc_ub([v_block], accum_dtype)
            acc_s_half = T.alloc_ub([v_block, BI], dtype)
            acc_o_ub = T.alloc_ub([v_block, D], accum_dtype)
            acc_o_half = T.alloc_ub([v_block, D], dtype)
            lse_ub = T.alloc_ub([v_block], accum_dtype)
            # Per-row scalar broadcast temp (reused for sub/mul/div;
            # replaces per-row scalar loops with whole-tile ops)
            bcast_2d = T.alloc_ub([v_block, BI], accum_dtype)

            # Load Q (resident for entire block lifetime)
            T.copy(Q[b_i, s_i, H0:H1, :D], q_l1)
            T.copy(Q[b_i, s_i, H0:H1, D:], q_tail_l1)

            # Cube: KV iteration loop (GEMM1 + GEMM2, store to workspace)
            # L1 double-buffer — alternate between slot 0 and slot 1 so
            # MTE2 (workspace→L1) can overlap with M (GEMM) on the Cube.
            # Pipeline transitions (MTE2→M, M→FIX, FIX→MTE2) are synced
            # automatically by AUTO_SYNC.
            for i_i in T.serial(NI):
                side = i_i % 2
                T.copy(workspace_1[cid, 0:BI, 0:D], kv_l1[side, :, :])
                T.copy(workspace_2[cid, 0:BI, 0:D_tail], kv_tail_l1[side, :, :])

                # Two-segment GEMM accumulate (dim=512 init, tail_dim=64 accumulate)
                T.gemm_v0(q_l1, kv_l1[side, :, :], acc_s_l0c, transpose_B=True, init=True)
                T.gemm_v0(q_tail_l1, kv_tail_l1[side, :, :], acc_s_l0c, transpose_B=True)

                # Store acc_s to workspace for Vector to read
                T.copy(acc_s_l0c, workspace_3[cid, 0:H_per_block, 0:BI])

                # Vector provides S_shared (bf16 softmax weights) in workspace_4
                T.copy(workspace_4[cid, 0:H_per_block, 0:BI], acc_s_l1[side, :, :])

                # GEMM2: P @ KV -> acc_o
                T.gemm_v0(acc_s_l1[side, :, :], kv_l1[side, :, :], acc_o_l0c, init=True)

                # Store acc_o to workspace for Vector to read
                T.copy(acc_o_l0c, workspace_5[cid, 0:H_per_block, 0:D])

            # Initialize Vector accumulators
            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -(2.0**30))

            # Vector: KV iteration loop (indirect load, mask, softmax, rescale).
            # All pipeline transitions are synced automatically by AUTO_SYNC,
            # except the gather loop below which uses an explicit MTE2||V
            # double-buffer pipeline (set_flag/wait_flag) to overlap the
            # indirect KV loads with the V-side copies.
            SIG_GATHER = 0  # base signal ID for gather double-buffer (slots 0,1)

            for i_i in T.serial(NI):
                # Load indices (MTE2: GM→UB)
                T.copy(Indices[b_i, s_i, g_i, i_i * BI : i_i * BI + BI], indices_ub)
                T.copy(indices_ub, indices_ub_float)

                # ---- MTE2||V gather pipeline (double-buffer) ----
                # Double-buffer kv_full_ub[2, D+D_tail]. MTE2 loads row[i+1]
                # to slot nxt while V copies row[i] from slot cur.
                # V does T.copy (UB→UB) — set_flag("MTE2","V") provides
                # memory visibility for T.copy reads.
                # Flag flow per slot: V→MTE2 (free) → MTE2→V (loaded) →
                # V consumes → V→MTE2 (free for reuse).
                # Init: pretend V released both slots (MTE2 can start)
                T.set_flag("V", "MTE2", SIG_GATHER)
                T.set_flag("V", "MTE2", SIG_GATHER + 1)

                # Prefetch slot 0
                T.wait_flag("V", "MTE2", SIG_GATHER)
                T.copy(
                    KV[b_i, indices_ub[vid * (BI // 2)], g_i, :],
                    kv_full_ub[0, :],
                )
                T.set_flag("MTE2", "V", SIG_GATHER)

                # Main body: prefetch [i+1] while consuming [i]
                for bi_i in T.serial(BI // 2 - 1):
                    cur = bi_i % 2
                    nxt = (bi_i + 1) % 2
                    # Prefetch next row (MTE2 loads to slot nxt)
                    T.wait_flag("V", "MTE2", nxt)
                    T.copy(
                        KV[b_i, indices_ub[(bi_i + 1) + vid * (BI // 2)], g_i, :],
                        kv_full_ub[nxt, :],
                    )
                    T.set_flag("MTE2", "V", nxt)
                    # Consume current row (V copies from slot cur)
                    T.wait_flag("MTE2", "V", cur)
                    T.copy(kv_full_ub[cur, :D], kv_ub_gather[bi_i, :])
                    T.copy(kv_full_ub[cur, D:], kv_tail_ub_gather[bi_i, :])
                    T.set_flag("V", "MTE2", cur)

                # Epilogue: consume last prefetched row
                T.wait_flag("MTE2", "V", (BI // 2 - 1) % 2)
                T.copy(
                    kv_full_ub[(BI // 2 - 1) % 2, :D],
                    kv_ub_gather[BI // 2 - 1, :],
                )
                T.copy(
                    kv_full_ub[(BI // 2 - 1) % 2, D:],
                    kv_tail_ub_gather[BI // 2 - 1, :],
                )
                T.set_flag("V", "MTE2", (BI // 2 - 1) % 2)

                # Drain: consume outstanding init-direction flags
                T.wait_flag("V", "MTE2", SIG_GATHER)
                T.wait_flag("V", "MTE2", SIG_GATHER + 1)

                # Bulk write gathered KV to workspace (UB→GM)
                T.copy(
                    kv_ub_gather,
                    workspace_1[cid, vid * (BI // 2) : (vid + 1) * (BI // 2), :],
                )
                T.copy(
                    kv_tail_ub_gather,
                    workspace_2[cid, vid * (BI // 2) : (vid + 1) * (BI // 2), :],
                )

                # Load GEMM result from workspace (Cube wrote this)
                T.copy(workspace_3[cid, vid * v_block : vid * v_block + v_block, :], acc_s_ub_)

                # Causal mask: compare + select
                T.tile.compare(mask_ub, indices_ub_float, T.float32(s_i), "LE")
                T.tile.fill(acc_s_ub, 0.0)
                for h_i in T.serial(v_block):
                    T.tile.select(
                        acc_s_ub[h_i, :],
                        mask_ub,
                        acc_s_ub_[h_i, :],
                        -T.infinity(accum_dtype),
                        "VSEL_TENSOR_SCALAR_MODE",
                    )

                # Online softmax (flash attention style — scale BEFORE reduce_max)
                T.copy(m_i, m_i_prev)
                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                T.reduce_max(acc_s_ub, m_i, dim=-1)
                T.tile.max(m_i, m_i, m_i_prev)

                # alpha = exp(m_i_prev - m_i) — NO extra sm_scale (already scaled)
                T.tile.sub(m_i_prev, m_i_prev, m_i)
                T.tile.exp(m_i_prev, m_i_prev)

                # P = exp(scaled_score - m_i) — broadcast + whole-tile sub
                T.tile.broadcast(bcast_2d, m_i, axis=1)
                T.tile.sub(acc_s_ub, acc_s_ub, bcast_2d)
                T.tile.exp(acc_s_ub, acc_s_ub)

                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)

                # dtype convert fp32 -> bf16, store to workspace for Cube GEMM2
                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, workspace_4[cid, vid * v_block : vid * v_block + v_block, :])

                # Rescale and accumulate (mul kept per-row in-loop: a 3rd
                # [v_block,D] 64KB bcast temp cannot coexist with acc_o +
                # acc_o_ub both live inside the loop)
                for h_i in T.serial(v_block):
                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])

                # Load GEMM2 result from workspace
                T.copy(workspace_5[cid, vid * v_block : vid * v_block + v_block, :], acc_o_ub)
                T.tile.add(acc_o, acc_o, acc_o_ub)

            # Normalize (add epsilon to avoid division by zero / ln(0))
            T.tile.add(sumexp, sumexp, T.float32(1e-30))
            # div kept per-row: acc_o_ub is the workspace_5 bridge buffer and
            # cannot be reused as broadcast scratch. A separate
            # [v_block,D] 64KB temp won't fit UB.
            for h_i in T.serial(v_block):
                T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

            # Lse = (ln(sumexp) + m_i) * log2(e)
            # m_i is already scaled (sm_scale applied before reduce_max)
            T.tile.ln(lse_ub, sumexp)
            T.tile.add(lse_ub, lse_ub, m_i)
            T.tile.mul(lse_ub, lse_ub, T.float32(1.44269504))

            # Output: convert then write to GM
            T.copy(acc_o, acc_o_half)
            T.copy(acc_o_half, Output[b_i, s_i, H0 + vid * v_block : H0 + v_block + vid * v_block, :])
            T.copy(lse_ub, Lse[b_i, s_i, H0 + vid * v_block : H0 + v_block + vid * v_block])

    return main


def sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, d_v=512, block_I=64):
    """Wrapper for the Developer-mode sparse MLA forward kernel.

    Handles shape parsing, host-side KV padding (out-of-bounds sentinel),
    and parameter forwarding to the compiled kernel.
    """
    is_casual = True
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v
    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    # Pad KV with one extra zero row (out-of-bounds sentinel: indices equal
    # to seq_len_kv point at this row, and the causal mask maps them to -inf)
    kv_padded = torch.zeros(
        (batch, seq_len_kv + 1, kv_group, dim + tail_dim),
        dtype=kv.dtype,
        device=kv.device,
    )
    kv_padded[:, :seq_len_kv, :, :] = kv
    kv = kv_padded

    kernel = sparse_mla_fwd(
        heads=heads,
        dim=dim,
        tail_dim=tail_dim,
        topk=topk,
        kv_group=kv_group,
        sm_scale=sm_scale,
        is_causal=is_casual,
        block_I=block_I,
    )
    out, lse = kernel(q, kv, indices)
    return out, lse


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None):
    """PyTorch reference (ported from GPU source, device-agnostic).

    Only outputs O (not Lse); the tests verify O precision.
    """
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32, device=q.device).view(-1, 1) >= torch.arange(
        0, sk, 1, dtype=torch.int32, device=q.device
    ).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, 0, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim)
    return o.to(torch.bfloat16)


def make_indices(B, S, SKV, HKV, topk, device="npu"):
    """Construct causal sparse indices.

    Each query position s only indexes key positions <= s.
    Positions beyond available are filled with SKV (out-of-bounds sentinel,
    kernel masks as -inf).
    """
    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device=device)
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i
    return indices


if __name__ == "__main__":
    torch.set_default_device("npu")
    torch.manual_seed(0)
    tilelang.disable_cache()

    # Smoke test: minimal shape (B=1 S=128 SKV=128 H=16 HKV=1 DQK=576 DV=512 topk=64)
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 128, 128, 16, 1, 576, 512, 64
    dtype = torch.bfloat16

    try:
        q = torch.randn((B, S, H, DQK), dtype=dtype)
        kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype)
        indices = make_indices(B, S, SKV, HKV, topk)

        torch.npu.synchronize()

        tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices)
        torch.npu.synchronize()

        ref_out = ref_sparse_mla_fwd_interface(q, kv, indices)
        torch.npu.synchronize()

        torch.testing.assert_close(tl_out, ref_out, rtol=1e-2, atol=1e-2)
        print("[SMOKE_PASS]")
        print("Test Passed!")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[SMOKE_FAIL]: {e}")
