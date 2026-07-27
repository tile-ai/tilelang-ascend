import math
import torch

import tilelang
import tilelang.language as T
import torch.nn.functional as F


def get_sparse_attn_mask_from_topk(x, topk, use_dense_for_last_block=False):
    bsz, num_head, downsample_len, _ = x.shape
    sparse_index = torch.topk(x, topk, dim=-1).indices
    dense_mask = torch.full([bsz, num_head, downsample_len, downsample_len], False, dtype=torch.bool, device=x.device)
    dense_mask.scatter_(-1, sparse_index, True)
    if use_dense_for_last_block:
        dense_mask[:, :, -2:, :] = True
    dense_mask.tril_()
    return dense_mask


def get_sparse_attn_mask_from_threshold(x, threshold, use_dense_for_last_block=False):
    dense_mask = x > threshold
    if use_dense_for_last_block:
        dense_mask[:, :, -2:, :] = True
    dense_mask.tril_()
    return dense_mask


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7], pass_configs=pass_configs)
def blocksparse_flashattn_ascend(batch, heads, seq_q, seq_kv, dim, downsample_len, is_causal):
    block_M = 64
    block_N = 64
    dtype = "float16"
    accum_dtype = "float"
    block_mask_dtype = "int8"
    sm_scale = (1.0 / dim) ** 0.5

    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    block_mask_shape = [batch, heads, downsample_len, downsample_len]
    block_num = ((seq_q + block_M - 1) // block_M) * heads * batch

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        BlockSparseMask: T.Tensor(block_mask_shape, block_mask_dtype),
        Output: T.Tensor(q_shape, dtype),
        workspace_1: T.Tensor([block_num, block_M, block_N], accum_dtype),
        workspace_2: T.Tensor([block_num, block_M, block_N], dtype),
        workspace_3: T.Tensor([block_num, block_M, dim], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % ((seq_q + block_M - 1) // block_M)
            by = cid // ((seq_q + block_M - 1) // block_M) % heads
            bz = cid // ((seq_q + block_M - 1) // block_M) // heads % batch

            # L1 buffers (Cube core)
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)
            acc_s_l1 = T.alloc_L1([block_M, block_N], dtype)
            acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_M, dim], accum_dtype)

            # UB buffers (Vector core) for online softmax
            acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)
            sumexp = T.alloc_ub([block_M // 2], accum_dtype)
            m_i = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            m_i_prev = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_ub_ = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            sumexp_i_ub = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)
            acc_o_ub = T.alloc_ub([block_M // 2, dim], accum_dtype)
            acc_o_half = T.alloc_ub([block_M // 2, dim], dtype)

            # UB buffers for block-sparse mask and causal mask
            mask_row = T.alloc_ub([downsample_len], block_mask_dtype)
            col_idx = T.alloc_ub([block_N], "int32")
            col_idx_f = T.alloc_ub([block_N], "float")
            cmp_mask = T.alloc_ub([block_N // 8], "uint8")

            # === Cube: Load Q, run GEMM pipeline for all KV blocks ===
            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_l1)

            loop_range = (seq_kv + block_N - 1) // block_N
            for k in T.serial(loop_range):
                T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], k_l1)
                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                T.copy(acc_s_l0c, workspace_1[cid, :, :])

                T.copy(workspace_2[cid, :, :], acc_s_l1)
                T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], v_l1)
                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, workspace_3[cid, :, :])

            # === Vector: Online softmax + output accumulation ===
            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -T.infinity(accum_dtype))

            T.copy(BlockSparseMask[bz, by, bx, :], mask_row)

            past_len = seq_kv - seq_q

            for _k in T.serial(loop_range):
                T.tile.fill(acc_s_ub, 0.0)
                T.copy(m_i, m_i_prev)

                # Load S = Q@K^T from Cube workspace
                T.copy(workspace_1[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], acc_s_ub_)

                # Block sparsity mask
                if mask_row[_k] == 0:
                    T.tile.fill(acc_s_ub_, -T.infinity(accum_dtype))

                # Element-level causal mask via compare+select (float32)
                elif is_causal:
                    T.tile.createvecindex(col_idx, _k * block_N)
                    T.copy(col_idx, col_idx_f)

                    for h_i in range(block_M // 2):
                        q_pos = bx * block_M + vid * (block_M // 2) + h_i + past_len
                        T.tile.compare(cmp_mask, col_idx_f, T.float32(q_pos), "LE")
                        T.tile.select(acc_s_ub_[h_i, :], cmp_mask, acc_s_ub_[h_i, :], -T.infinity(accum_dtype), "VSEL_TENSOR_SCALAR_MODE")

                T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)

                T.reduce_max(acc_s_ub, m_i, dim=-1)
                T.tile.max(m_i, m_i, m_i_prev)
                T.tile.sub(m_i_prev, m_i_prev, m_i)
                T.tile.exp(m_i_prev, m_i_prev)

                for h_i in range(block_M // 2):
                    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])

                T.tile.exp(acc_s_ub, acc_s_ub)
                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)

                for h_i in range(block_M // 2):
                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])

                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, workspace_2[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :])

                T.copy(workspace_3[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], acc_o_ub)
                T.tile.add(acc_o, acc_o, acc_o_ub)

            # Final normalization: acc_o /= sumexp
            for h_i in range(block_M // 2):
                T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

            T.copy(acc_o, acc_o_half)
            T.copy(acc_o_half, Output[bz, by, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + block_M // 2, :])

    return main


def test_topk_sparse_attention():
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = 4, 2, 256, 64
    TOPK = 2
    BLOCK = 64
    torch.manual_seed(0)

    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="npu", dtype=torch.float16)
    k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="npu", dtype=torch.float16)
    v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="npu", dtype=torch.float16)

    sm_scale = 1.0 / (D_HEAD**0.5)

    downsample_factor = BLOCK
    downsample_len = math.ceil(SEQ_LEN / downsample_factor)
    x_ds = torch.randn([BATCH, N_HEADS, downsample_len, downsample_len], device="npu", dtype=torch.float16)
    x_ds[:, :, :, 0] = 100
    block_mask = get_sparse_attn_mask_from_topk(x_ds, topk=TOPK)

    kernel = blocksparse_flashattn_ascend(BATCH, N_HEADS, SEQ_LEN, SEQ_LEN, D_HEAD, downsample_len, is_causal=True)
    torch.npu.synchronize()
    tilelang_output = kernel(q, k, v, block_mask.to(torch.int8))
    torch.npu.synchronize()

    # Reference with FULL element-level causal + block sparsity
    full_mask = torch.kron(block_mask.float(), torch.ones(BLOCK, BLOCK, device="npu"))
    full_mask = full_mask[..., :SEQ_LEN, :SEQ_LEN].bool()
    full_mask = full_mask & torch.tril(torch.ones_like(full_mask))

    attn = torch.einsum("bhsd,bhtd->bhst", q.float(), k.float()) * sm_scale
    attn = attn.masked_fill(~full_mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    ref_output = torch.einsum("bhst,bhtd->bhsd", attn, v.float()).to(torch.float16)

    torch.testing.assert_close(tilelang_output, ref_output, atol=1e-2, rtol=1e-2)
    print("Pass topk sparse attention test with qlen == klen")


def test_topk_sparse_attention_qlen_lt_klen():
    BATCH, N_HEADS = 1, 1
    Q_LEN, K_LEN, D_HEAD = 128, 256, 64
    TOPK = 1
    BLOCK = 64
    torch.manual_seed(0)

    q = torch.randn(BATCH, N_HEADS, Q_LEN, D_HEAD, device="npu", dtype=torch.float16)
    k = torch.randn(BATCH, N_HEADS, K_LEN, D_HEAD, device="npu", dtype=torch.float16)
    v = torch.randn(BATCH, N_HEADS, K_LEN, D_HEAD, device="npu", dtype=torch.float16)
    sm_scale = 1.0 / (D_HEAD**0.5)

    downsample_factor = BLOCK
    downsample_len = math.ceil(K_LEN / downsample_factor)
    x_ds = torch.randn(BATCH, N_HEADS, downsample_len, downsample_len, device="npu", dtype=torch.float16)
    x_ds[:, :, :, 0] = 100
    block_mask = get_sparse_attn_mask_from_topk(x_ds, topk=TOPK)

    kernel = blocksparse_flashattn_ascend(BATCH, N_HEADS, Q_LEN, K_LEN, D_HEAD, downsample_len, is_causal=True)
    torch.npu.synchronize()
    tilelang_output = kernel(q, k, v, block_mask.to(torch.int8))
    torch.npu.synchronize()

    past_len = K_LEN - Q_LEN

    attn = torch.einsum("bhsd,bhtd->bhst", q.float(), k.float()) * sm_scale

    full_mask_full = torch.kron(block_mask.float(), torch.ones(BLOCK, BLOCK, device="npu")).bool()
    full_mask_full = full_mask_full[..., :K_LEN, :K_LEN]
    effective_mask = full_mask_full[..., past_len:K_LEN, :]

    i_global = torch.arange(past_len, K_LEN, device=k.device).unsqueeze(1)
    j_global = torch.arange(K_LEN, device=k.device).unsqueeze(0)
    causal_mask = j_global <= i_global

    final_mask = effective_mask & causal_mask

    attn = attn.masked_fill(~final_mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    ref_output = torch.einsum("bhst,bhtd->bhsd", attn, v.float()).to(torch.float16)

    torch.testing.assert_close(tilelang_output, ref_output, atol=1e-2, rtol=1e-2)
    print("Pass topk sparse attention test with qlen < klen")


def main():
    test_topk_sparse_attention()
    test_topk_sparse_attention_qlen_lt_klen()
    print("Test passed!")


if __name__ == "__main__":
    main()
