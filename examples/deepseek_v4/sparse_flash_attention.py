# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import logging

import tilelang
from tilelang import DataType, language as T


torch.set_default_device("npu")
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# kernel
@tilelang.jit(out_idx=[2], workspace_idx=[5, 6, 7, 8], pass_configs=pass_configs)
def sparse_attn_kernel(h: int, d: int, scale=None):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    scale = (1.0 / d) ** 0.5 if scale is None else scale

    dtype = "bfloat16"
    accum_dtype = "float"
    indices_dtype = "int32"

    block = 64
    v_block = h // 2
    num_topk = tilelang.cdiv(topk, block)
    block_num = b * m

    q_shape = [b, m, h, d]
    kv_shape = [b, n, d]
    attn_sink_shape = [h]
    topk_idxs_shape = [b, m, topk]

    accum_bits = DataType(accum_dtype).bits // 8
    v_block_pad = tilelang.cdiv(v_block * accum_bits, 32) * (32 // accum_bits)

    @T.prim_func
    def sparse_attn_kernel_(
        q: T.Tensor(q_shape, dtype),
        kv: T.Tensor(kv_shape, dtype),
        output: T.Tensor((b, m, h, d), accum_dtype),
        attn_sink: T.Tensor(attn_sink_shape, accum_dtype),
        topk_idxs: T.Tensor(topk_idxs_shape, indices_dtype),
        workspace_1: T.Tensor([block_num, block, d], dtype),  # kv idx copy
        workspace_2: T.Tensor([block_num, h, block], accum_dtype),  # acc_s copy
        workspace_3: T.Tensor([block_num, h, block], dtype),  # acc_s_half copy
        workspace_4: T.Tensor([block_num, h, d], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % m
            by = cid // m

            q_l1 = T.alloc_L1([h, d], dtype)
            kv_l1 = T.alloc_L1([block, d], dtype)
            acc_s_l1 = T.alloc_L1([h, block], dtype)

            acc_s_l0c = T.alloc_L0C([h, block], accum_dtype)
            acc_o_l0c = T.alloc_L0C([h, d], accum_dtype)

            kv_ub = T.alloc_ub([block, d], dtype)
            kv_ub_tmp = T.alloc_ub([d], dtype)
            acc_s_ub = T.alloc_ub([v_block, block], accum_dtype)
            acc_s_ub_singledim = T.alloc_ub([1, block], accum_dtype)
            acc_s_ub_ = T.alloc_ub([v_block, block], accum_dtype)
            acc_s_half = T.alloc_ub([v_block, block], dtype)
            acc_o_ub = T.alloc_ub([v_block, d], accum_dtype)
            acc_o_ub_ = T.alloc_ub([v_block, d], accum_dtype)
            idxs_ub = T.alloc_ub([block], indices_dtype)

            scores_max = T.alloc_ub([v_block_pad, 1], accum_dtype)
            scores_max_brd = T.alloc_ub([v_block_pad, block], accum_dtype)
            scores_max_prev = T.alloc_ub([v_block_pad, 1], accum_dtype)
            scores_max_prev_brd = T.alloc_ub([v_block_pad, d], accum_dtype)
            scores_sum = T.alloc_ub([v_block_pad], accum_dtype)
            sum_exp = T.alloc_ub([v_block_pad, 1], accum_dtype)
            sum_exp_brd = T.alloc_ub([v_block_pad, d], accum_dtype)
            attn_sink_ub = T.alloc_ub([v_block_pad], accum_dtype)

            T.copy(q[by, bx, :, :], q_l1)

            T.tile.fill(acc_o_ub, 0.0)
            T.tile.fill(sum_exp, 0.0)
            T.tile.fill(kv_ub_tmp, 0.0)
            T.tile.fill(scores_max, -T.infinity(accum_dtype))

            for t in T.serial(num_topk):
                T.tile.fill(acc_s_ub_singledim, 0.0)

                T.copy(topk_idxs[by, bx, t * block : t * block + block], idxs_ub)
                for i in T.serial(block):
                    if t * block + i >= topk:
                        idxs_ub[i] = -1

                for b_i in T.serial(block):
                    idx_num = idxs_ub[b_i]
                    if idx_num != -1:
                        T.copy(kv[by, idx_num, :], kv_ub[b_i, :])
                    else:
                        T.copy(kv_ub_tmp, kv_ub[b_i, :])
                T.copy(kv_ub, workspace_1[cid, :, :])

                T.copy(workspace_1[cid, 0:block, 0:d], kv_l1)
                T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True)
                T.copy(acc_s_l0c, workspace_2[cid, 0:h, 0:block])

                for i in T.serial(block):
                    if idxs_ub[i] == -1:
                        acc_s_ub_singledim[0, i] = -T.infinity(accum_dtype)
                T.tile.broadcast(acc_s_ub, acc_s_ub_singledim)

                T.copy(workspace_2[cid, vid * v_block : vid * v_block + v_block, :], acc_s_ub_)

                T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                T.tile.mul(acc_s_ub, acc_s_ub, scale)

                T.copy(scores_max, scores_max_prev)

                T.reduce_max(acc_s_ub, scores_max, dim=-1)
                T.tile.max(scores_max, scores_max, scores_max_prev)
                T.tile.sub(scores_max_prev, scores_max_prev, scores_max)
                T.tile.exp(scores_max_prev, scores_max_prev)
                T.tile.broadcast(scores_max_brd, scores_max)
                T.tile.sub(acc_s_ub, acc_s_ub, scores_max_brd)
                T.tile.exp(acc_s_ub, acc_s_ub)
                T.reduce_sum(acc_s_ub, scores_sum, dim=-1)
                T.tile.mul(sum_exp, sum_exp, scores_max_prev)
                T.tile.add(sum_exp, sum_exp, scores_sum)

                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, workspace_3[cid, vid * v_block : vid * v_block + v_block, :])

                T.copy(workspace_3[cid, 0:h, 0:block], acc_s_l1)
                T.gemm_v0(acc_s_l1, kv_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, workspace_4[cid, 0:h, 0:d])

                T.tile.broadcast(scores_max_prev_brd, scores_max_prev)
                T.tile.mul(acc_o_ub, acc_o_ub, scores_max_prev_brd)

                T.copy(workspace_4[cid, vid * v_block : vid * v_block + v_block, :], acc_o_ub_)
                T.tile.add(acc_o_ub, acc_o_ub, acc_o_ub_)

            T.copy(attn_sink[vid * v_block : vid * v_block + v_block], attn_sink_ub)

            T.tile.sub(attn_sink_ub, attn_sink_ub, scores_max[:, 0])
            T.tile.exp(attn_sink_ub, attn_sink_ub)
            T.tile.add(sum_exp[:, 0], sum_exp[:, 0], attn_sink_ub)
            T.tile.broadcast(sum_exp_brd, sum_exp)
            T.tile.div(acc_o_ub, acc_o_ub, sum_exp_brd)

            T.copy(acc_o_ub, output[by, bx, vid * v_block : vid * v_block + v_block, :])

    return sparse_attn_kernel_


# golden
def gather_sparse_kv(kv_states: torch.Tensor, topk_idxs: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, topk = topk_idxs.shape
    batch_idx = torch.arange(batch_size, device=kv_states.device).view(batch_size, 1, 1).expand(-1, seq_len, topk)
    safe_topk_idxs = torch.where(topk_idxs == -1, 0, topk_idxs).long()
    gathered = kv_states[batch_idx, safe_topk_idxs, :]
    gathered_mask = (topk_idxs != -1).unsqueeze(-1).to(gathered.dtype)
    return gathered * gathered_mask


def sparse_softmax_with_sink(scores: torch.Tensor, attn_sink: torch.Tensor, head_dim: int, softmax_dim: int = -1) -> torch.Tensor:
    max_vals = torch.max(scores, dim=softmax_dim, keepdim=True).values
    exp_scores = torch.exp(scores - max_vals)
    sum_exp = torch.sum(exp_scores, dim=softmax_dim, keepdim=True)

    sink_view_shape = [1] * scores.dim()
    sink_view_shape[head_dim if head_dim >= 0 else scores.dim() + head_dim] = scores.shape[head_dim]
    sink_term = torch.exp(attn_sink.view(sink_view_shape) - max_vals)
    return exp_scores / (sum_exp + sink_term)


def sparse_attn(
    query_states: torch.Tensor, kv_states: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
):
    sparse_kv = gather_sparse_kv(kv_states, topk_idxs)
    score_mask = torch.where((topk_idxs == -1).unsqueeze(-2), -torch.inf, 0.0).to(device=query_states.device, dtype=torch.float32)
    scores = torch.matmul(query_states, sparse_kv.transpose(-2, -1)).to(torch.float32)
    probs = sparse_softmax_with_sink(scores * softmax_scale + score_mask, attn_sink, head_dim=-2)
    return torch.matmul(probs, sparse_kv.to(torch.float32))


def make_random_test_inputs(b: int, m: int, n: int, h: int, d: int, topk: int, dtype: torch.dtype):
    return {
        "q": torch.randn((b, m, h, d), dtype=dtype),
        "kv": torch.randn((b, n, d), dtype=dtype),
        "attn_sink": torch.randn((h,), dtype=torch.float32),
        "topk_idxs": torch.randint(0, n, (b, m, topk), dtype=torch.int32, device="npu"),
    }


def test():
    # Input data dtype and shape
    dtype = torch.bfloat16
    b, m, n, h, d, topk = 1, 256, 256, 64, 512, 128  # Shape 1
    # b, m, n, h, d, topk = 1, 6, 6, 16, 512, 6  # Shape 2

    inputs = make_random_test_inputs(b, m, n, h, d, topk, dtype)
    q = inputs["q"]
    kv = inputs["kv"]
    attn_sink = inputs["attn_sink"]
    topk_idxs = inputs["topk_idxs"]
    output_golden = torch.zeros((b, m, h, d), dtype=dtype)
    softmax_scale = 512**-0.5

    func = sparse_attn_kernel(h=h, d=d, scale=softmax_scale)

    logging.info("init successful!")

    output = func(q, kv, attn_sink, topk_idxs)
    torch.npu.synchronize()

    output_golden = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

    torch.testing.assert_close(output_golden, output, rtol=1e-2, atol=1e-2)
    logging.info("Kernel Output Match!")


if __name__ == "__main__":
    test()
