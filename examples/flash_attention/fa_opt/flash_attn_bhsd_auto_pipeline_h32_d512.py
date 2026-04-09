import argparse
import tilelang
from tilelang import DataType, language as T

import torch

torch.set_default_device("npu")
torch.manual_seed(0)

tilelang.disable_cache()

NUM_STAGES = 2
L0_MAX_SIZE = 64 * 1024  # 64KB

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flash_attention_fwd(
    batch,
    seq_len,
    heads_q,
    heads_kv,
    dim,
):
    assert heads_q % heads_kv == 0, "heads_q must be a multiple of heads_kv"
    block_M, block_N = 64, 128

    dtype = "float16"
    accum_dtype = "float"

    sm_scale = (1.0 / dim) ** 0.5

    shape_q = [batch, heads_q, seq_len, dim]
    shape_kv = [batch, heads_kv, seq_len, dim]

    block_num = seq_len // block_M * heads_q * batch

    n_num = T.max(T.ceildiv(block_N * dim * DataType(dtype).bits // 8, L0_MAX_SIZE), 1)
    block_K = T.ceildiv(block_N, n_num)
    block_D = T.ceildiv(dim, n_num)
    num_stages = T.min(T.ceildiv(seq_len, block_N), NUM_STAGES).value

    @T.prim_func
    def main(
        Q: T.Tensor(shape_q, dtype),  # type: ignore
        K: T.Tensor(shape_kv, dtype),  # type: ignore
        V: T.Tensor(shape_kv, dtype),  # type: ignore
        Output: T.Tensor(shape_q, dtype),  # type: ignore
        workspace_1: T.Tensor([block_num, block_M, block_N], accum_dtype),
        workspace_2: T.Tensor([block_num, block_M, block_N], dtype),
        workspace_3: T.Tensor([block_num, block_M, dim], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads_q
            bz = cid // (seq_len // block_M) // heads_q % batch

            kv_by = by // (heads_q // heads_kv)

            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_K, dim], dtype)
            v_l1 = T.alloc_L1([block_N, block_D], dtype)

            acc_s_l1 = T.alloc_L1([block_M, block_N], dtype)

            acc_s_l0c = T.alloc_L0C([block_M, block_K], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_M, block_D], accum_dtype)

            acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)
            sumexp = T.alloc_ub([block_M // 2, 1], accum_dtype)

            m_i = T.alloc_ub([block_M // 2, 1], accum_dtype)
            m_i_2d = T.alloc_ub([block_M // 2, block_N], accum_dtype)

            acc_s_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            m_i_prev = T.alloc_ub([block_M // 2, 1], accum_dtype)

            acc_s_ub_ = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            sumexp_i_ub = T.alloc_ub([block_M // 2, 1], accum_dtype)
            acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)
            acc_o_ub = T.alloc_ub([block_M // 2, dim], accum_dtype)
            acc_o_half = T.alloc_ub([block_M // 2, dim], dtype)

            # with T.Scope("C"):
            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -(2**30))
            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_l1)

            for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=num_stages):
                for n_i in T.serial(n_num):
                    T.copy(K[bz, kv_by, k * block_N + n_i * block_K : k * block_N + (n_i + 1) * block_K, :], k_l1)
                    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                    T.copy(acc_s_l0c, workspace_1[cid, :, n_i * block_K : (n_i + 1) * block_K])

                T.tile.fill(acc_s_ub, 0.0)
                T.copy(m_i, m_i_prev)
                T.copy(workspace_1[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], acc_s_ub_)
                T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                T.reduce_max(acc_s_ub, m_i, dim=-1)
                T.tile.max(m_i, m_i, m_i_prev)
                T.tile.sub(m_i_prev, m_i_prev, m_i)
                T.tile.exp(m_i_prev, m_i_prev)

                T.tile.broadcast(m_i_2d, m_i)
                T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)

                T.tile.exp(acc_s_ub, acc_s_ub)
                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)

                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, workspace_2[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :])
                T.copy(workspace_2[cid, :, :], acc_s_l1)

                for n_i in T.serial(n_num):
                    T.copy(V[bz, kv_by, k * block_N : (k + 1) * block_N, n_i * block_D : (n_i + 1) * block_D], v_l1)
                    T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                    T.copy(acc_o_l0c, workspace_3[cid, :, n_i * block_D : (n_i + 1) * block_D])

                for h_i in range(block_M // 2):
                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i, 0])

                T.copy(workspace_3[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], acc_o_ub)
                T.tile.add(acc_o, acc_o, acc_o_ub)

            for h_i in range(block_M // 2):
                T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i, 0])

            T.copy(acc_o, acc_o_half)
            T.copy(acc_o_half, Output[bz, by, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + block_M // 2, :])

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4, help="batch size")
    parser.add_argument("--S", type=int, default=4096, help="seq len")
    parser.add_argument("--H", type=int, default=32, help="attention head size")
    parser.add_argument("--q-heads", type=int, default=None, help="query head size")
    parser.add_argument("--kv-heads", type=int, default=None, help="kv head size")
    parser.add_argument("--D", type=int, default=512, help="hidden dim")
    parser.add_argument("--no-check", action="store_true", help="disable reference check")
    args = parser.parse_args()
    B, S, H, D = args.B, args.S, args.H, args.D
    Q_H = args.q_heads or H
    KV_H = args.kv_heads or H

    func = flash_attention_fwd(
        batch=B,
        seq_len=S,
        heads_q=Q_H,
        heads_kv=KV_H,
        dim=D,
    )

    def ref_flash_attn(q, k, v):
        # GQA/MQA support: torch.einsum does not support MQA/GQA broadcasting, so we must manually repeat k/v heads
        if k.shape[1] != q.shape[1]:
            n_rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        q = q.float()
        k = k.float()
        v = v.float()

        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

        return output.to(torch.float16)

    q = torch.randn((B, Q_H, S, D), dtype=torch.float16)
    k = torch.randn((B, KV_H, S, D), dtype=torch.float16)
    v = torch.randn((B, KV_H, S, D), dtype=torch.float16)

    torch.npu.synchronize()
    print("init successful!")

    output = func(q, k, v)

    if not args.no_check:
        ref_output = ref_flash_attn(q, k, v)
        torch.npu.synchronize()
        torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)

        print("Test Passed!")
    else:
        torch.npu.synchronize()
        print("Reference check skipped.")
