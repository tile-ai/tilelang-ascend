import torch
import torch.nn.functional as F
import tilelang
from tilelang import language as T
from einops import rearrange, einsum
import argparse

torch.set_default_device("npu")
torch.manual_seed(0)

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6], pass_configs=pass_configs)
def flashattn(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, block_N, block_H, softmax_scale):
    sm_scale = softmax_scale
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert kv_head_num == 1, "kv_head_num must be 1 for MLA decode"

    block_num = batch * (heads // VALID_BLOCK_H)

    @T.prim_func
    def main(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            Output: T.Tensor([batch, heads, dim], dtype),
            workspace_1: T.Tensor([block_num, VALID_BLOCK_H, block_N], accum_dtype),
            workspace_2: T.Tensor([block_num, VALID_BLOCK_H, dim], accum_dtype),
    ):
        with T.Kernel(batch * (heads // VALID_BLOCK_H), is_npu=True) as (cid, vid):
            bid = cid // (heads // VALID_BLOCK_H)
            hid = cid % (heads // VALID_BLOCK_H)

            Q_shared = T.alloc_shared([VALID_BLOCK_H, dim], dtype)
            Q_pe_shared = T.alloc_shared([VALID_BLOCK_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)

            acc_s_l0c = T.alloc_fragment([VALID_BLOCK_H, block_N], accum_dtype)
            acc_o_l0c = T.alloc_fragment([VALID_BLOCK_H, dim], accum_dtype)

            acc_o = T.alloc_shared([VALID_BLOCK_H, dim], accum_dtype)
            acc_o_half = T.alloc_shared([VALID_BLOCK_H, dim], dtype)
            acc_o_ub = T.alloc_shared([VALID_BLOCK_H, dim], accum_dtype)
            acc_s = T.alloc_shared([VALID_BLOCK_H, block_N], accum_dtype)
            acc_s_half = T.alloc_shared([VALID_BLOCK_H, block_N], dtype)
            scores_max = T.alloc_shared([VALID_BLOCK_H], accum_dtype)
            scores_max_prev = T.alloc_shared([VALID_BLOCK_H], accum_dtype)
            scores_scale = T.alloc_shared([VALID_BLOCK_H], accum_dtype)
            scores_sum = T.alloc_shared([VALID_BLOCK_H], accum_dtype)
            logsum = T.alloc_shared([VALID_BLOCK_H], accum_dtype)

            cur_kv_head = 0

            T.copy(Q[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, :], Q_shared)
            T.copy(Q_pe[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, :], Q_pe_shared)
            T.tile.fill(acc_o, 0.0)
            T.tile.fill(logsum, 0.0)
            T.tile.fill(scores_max, -(2.0**30))

            loop_range = T.ceildiv(seqlen_kv, block_N)
            for k in T.Pipelined(loop_range, num_stages=2):
                kv_start = k * block_N
                kv_end = (k + 1) * block_N
                T.copy(KV[bid, kv_start:kv_end, cur_kv_head, :], KV_shared)
                T.copy(K_pe[bid, kv_start:kv_end, cur_kv_head, :], K_pe_shared)

                T.gemm_v0(Q_shared, KV_shared, acc_s_l0c, transpose_B=True, init=True)
                T.gemm_v0(Q_pe_shared, K_pe_shared, acc_s_l0c, transpose_B=True)
                T.copy(acc_s_l0c, workspace_1[cid, :, :])

                T.copy(workspace_1[cid, :, :], acc_s)

                T.copy(scores_max, scores_max_prev)
                T.tile.fill(scores_max, -(2.0**30))
                T.reduce_max(acc_s, scores_max, dim=-1, clear=False)
                T.tile.max(scores_max, scores_max, scores_max_prev)

                T.tile.sub(scores_max_prev, scores_max_prev, scores_max)
                T.tile.mul(scores_max_prev, scores_max_prev, sm_scale)
                T.tile.exp(scores_scale, scores_max_prev)

                T.tile.sub(acc_s, acc_s, scores_max)
                T.tile.mul(acc_s, acc_s, sm_scale)
                T.tile.exp(acc_s, acc_s)

                T.reduce_sum(acc_s, scores_sum, dim=-1, clear=False)

                T.tile.mul(logsum, logsum, scores_scale)
                T.tile.add(logsum, logsum, scores_sum)

                T.tile.mul(acc_o, acc_o, scores_scale)

                T.copy(acc_s, acc_s_half)
                T.gemm_v0(acc_s_half, KV_shared, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, workspace_2[cid, :, :])

                T.copy(workspace_2[cid, :, :], acc_o_ub)
                T.tile.add(acc_o, acc_o, acc_o_ub)

            for i, j in T.Parallel(VALID_BLOCK_H, dim):
                acc_o[i, j] /= logsum[i]

            T.copy(acc_o, acc_o_half)
            T.copy(acc_o_half, Output[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, :])

    return main


def ref_program(q, q_pe, kv, k_pe):
    dim = q.shape[-1]
    pe_dim = q_pe.shape[-1]
    num_head_groups = q.shape[1] // kv.shape[2]
    scale = (dim + pe_dim)**0.5
    q = rearrange(q, "b (h g) d -> b g h d", g=num_head_groups)
    q_pe = rearrange(q_pe, "b (h g) d -> b g h d", g=num_head_groups)
    kv = rearrange(kv, "b n h d -> b h n d")
    k_pe = rearrange(k_pe, "b n h d -> b h n d")
    query = torch.concat([q, q_pe], dim=-1)
    key = torch.concat([kv, k_pe], dim=-1)
    scores = einsum(query, key, "b g h d, b h s d -> b g h s")
    attention = F.softmax(scores / scale, dim=-1)
    out = einsum(attention, kv, "b g h s, b h s d -> b g h d")
    out = rearrange(out, "b g h d -> b (h g) d")
    return out


def main(
    batch=1,
    heads=128,
    kv_heads=1,
    kv_ctx=8192,
    dim=512,
    pe_dim=64,
):
    BLOCK_N = 64
    BLOCK_H = min(64, heads // kv_heads)
    softmax_scale = (dim + pe_dim)**-0.5

    kernel = flashattn(batch, heads, kv_heads, kv_ctx, dim, pe_dim, BLOCK_N, BLOCK_H, softmax_scale)

    q = torch.randn(batch, heads, dim, dtype=torch.float16)
    q_pe = torch.randn(batch, heads, pe_dim, dtype=torch.float16)
    kv = torch.randn(batch, kv_ctx, kv_heads, dim, dtype=torch.float16)
    k_pe = torch.randn(batch, kv_ctx, kv_heads, pe_dim, dtype=torch.float16)

    output = kernel(q, q_pe, kv, k_pe)

    ref_output = ref_program(q.cpu(), q_pe.cpu(), kv.cpu(), k_pe.cpu())
    torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-2, atol=1e-2)

    print("Test passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--heads", type=int, default=128, help="q heads number")
    parser.add_argument("--kv_heads", type=int, default=1, help="kv heads number")
    parser.add_argument("--kv_ctx", type=int, default=8192, help="kv context length")
    parser.add_argument("--dim", type=int, default=512, help="head dim")
    parser.add_argument("--pe_dim", type=int, default=64, help="pe head dim")
    args = parser.parse_args()
    batch, heads, kv_heads, kv_ctx, dim, pe_dim = args.batch, args.heads, args.kv_heads, args.kv_ctx, args.dim, args.pe_dim
    main(batch, heads, kv_heads, kv_ctx, dim, pe_dim)
