import tilelang
import tilelang.language as T
import torch
import argparse
from einops import rearrange, einsum
import torch.nn.functional as F

torch.set_default_device('npu')
torch.manual_seed(0)

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flashattn_gqa_decode(batch, heads, groups, seqlen_kv, dim, block_H=16, block_N=64):
    kv_group_num = heads // groups
    H_per_block = block_H

    dtype = "float16"
    accum_dtype = "float"
    scale = (1.0 / dim) ** 0.5

    shape_q = [batch, heads, dim]
    shape_k = [batch, seqlen_kv, groups, dim]
    shape_v = [batch, seqlen_kv, groups, dim]
    shape_o = [batch, heads, dim]

    block_num = batch * groups * (kv_group_num // H_per_block)

    @T.prim_func
    def main(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        Output: T.Tensor(shape_o, dtype),
        workspace_1: T.Tensor([block_num, H_per_block, block_N], accum_dtype),
        workspace_2: T.Tensor([block_num, H_per_block, block_N], dtype),
        workspace_3: T.Tensor([block_num, H_per_block, dim], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bid = cid // (groups * (kv_group_num // H_per_block))
            gid = (cid // (kv_group_num // H_per_block)) % groups
            h_block_id = cid % (kv_group_num // H_per_block)

            group_start = gid * kv_group_num + h_block_id * H_per_block

            q_l1 = T.alloc_shared([H_per_block, dim], dtype)
            k_l1 = T.alloc_shared([block_N, dim], dtype)
            v_l1 = T.alloc_shared([block_N, dim], dtype)
            acc_s_l1 = T.alloc_shared([H_per_block, block_N], dtype)

            acc_s_l0c = T.alloc_fragment([H_per_block, block_N], accum_dtype)
            acc_o_l0c = T.alloc_fragment([H_per_block, dim], accum_dtype)

            # Vector core buffers: split into H_per_block // 2 halves
            # Each vector core processes one half via vid
            acc_o = T.alloc_shared([H_per_block // 2, dim], accum_dtype)
            sumexp = T.alloc_shared([H_per_block // 2], accum_dtype)
            m_i = T.alloc_shared([H_per_block // 2], accum_dtype)

            acc_s_ub = T.alloc_shared([H_per_block // 2, block_N], accum_dtype)
            m_i_prev = T.alloc_shared([H_per_block // 2], accum_dtype)
            sumexp_i_ub = T.alloc_shared([H_per_block // 2], accum_dtype)
            acc_s_half = T.alloc_shared([H_per_block // 2, block_N], dtype)
            acc_o_ub = T.alloc_shared([H_per_block // 2, dim], accum_dtype)
            acc_o_half = T.alloc_shared([H_per_block // 2, dim], dtype)

            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -2**30)
            T.copy(Q[bid, group_start : group_start + H_per_block, :], q_l1)

            loop_range = T.ceildiv(seqlen_kv, block_N)
            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bid, k * block_N : (k + 1) * block_N, gid, :], k_l1)
                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                T.copy(acc_s_l0c, workspace_1[cid, :, :])

                T.copy(m_i, m_i_prev)
                T.copy(
                    workspace_1[cid, vid * H_per_block // 2 : vid * H_per_block // 2 + H_per_block // 2, :],
                    acc_s_ub)

                T.tile.mul(acc_s_ub, acc_s_ub, scale)

                T.reduce_max(acc_s_ub, m_i, dim=-1)
                T.tile.max(m_i, m_i, m_i_prev)

                T.tile.sub(m_i_prev, m_i_prev, m_i)
                T.tile.exp(m_i_prev, m_i_prev)

                for h_i in range(H_per_block // 2):
                    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
                T.tile.exp(acc_s_ub, acc_s_ub)

                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)

                T.copy(acc_s_ub, acc_s_half)
                T.copy(
                    acc_s_half,
                    workspace_2[cid, vid * H_per_block // 2 : vid * H_per_block // 2 + H_per_block // 2, :])

                T.copy(workspace_2[cid, :, :], acc_s_l1)
                T.copy(V[bid, k * block_N : (k + 1) * block_N, gid, :], v_l1)
                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, workspace_3[cid, :, :])

                for h_i in range(H_per_block // 2):
                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])
                T.copy(
                    workspace_3[cid, vid * H_per_block // 2 : vid * H_per_block // 2 + H_per_block // 2, :],
                    acc_o_ub)
                T.tile.add(acc_o, acc_o, acc_o_ub)

            for h_i in range(H_per_block // 2):
                T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

            T.copy(acc_o, acc_o_half)
            T.copy(
                acc_o_half,
                Output[bid, group_start + vid * H_per_block // 2 : group_start + vid * H_per_block // 2 + H_per_block // 2, :])

    return main


def ref_program(query, key, value):
    dim = query.shape[-1]
    num_head_groups = query.shape[1] // key.shape[2]
    scale = dim**0.5

    key = rearrange(key, "b n h d -> b h n d")
    value = rearrange(value, "b n h d -> b h n d")
    query = rearrange(query, "b (h g) d -> b g h d", g=num_head_groups)

    scores = einsum(query, key, "b g h d, b h s d -> b g h s")

    attention = F.softmax(scores / scale, dim=-1)
    out = einsum(attention, value, "b g h s, b h s d -> b g h d")
    out = rearrange(out, "b g h d -> b (h g) d")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--heads", type=int, default=128, help="query heads")
    parser.add_argument("--groups", type=int, default=8, help="kv heads")
    parser.add_argument("--kv_seqlen", type=int, default=8192, help="kv sequence length")
    parser.add_argument("--dim", type=int, default=128, help="hidden dim")
    parser.add_argument("--block_H", type=int, default=16, help="block size for H dimension")
    args = parser.parse_args()

    batch, heads, groups, kv_seqlen, dim, block_H = args.batch, args.heads, args.groups, args.kv_seqlen, args.dim, args.block_H

    kv_group_num = heads // groups
    
    # Hardware constraint check
    # Ascend vector unit requires H_per_block // 2 >= 8 for reduce operations
    MIN_V_BLOCK = 8
    if kv_group_num < 16:
        separator = "=" * 80
        error_msg = f"""
{separator}
Hardware Constraint Error: Current parameters violate Ascend vector unit constraints
{separator}
Problem Analysis:
  - kv_group_num = {heads} // {groups} = {kv_group_num}
  - H_per_block max = kv_group_num = {kv_group_num}
  - H_per_block // 2 = {kv_group_num // 2} < {MIN_V_BLOCK} (hardware requirement)
  - Ascend vector unit reduce operations require minimum dimension >= {MIN_V_BLOCK}

Solutions:
  Option 1: Increase heads
    python examples/flash_decoding/example_gqa_decode.py --batch {batch} --heads {groups * 16} --groups {groups} --kv_seqlen {kv_seqlen}
    (kv_group_num = 16, H_per_block//2 = 8, satisfies constraint)

  Option 2: Decrease groups
    python examples/flash_decoding/example_gqa_decode.py --batch {batch} --heads {heads} --groups {heads // 16} --kv_seqlen {kv_seqlen}
    (kv_group_num = 16, H_per_block//2 = 8, satisfies constraint)

  Option 3: Use PyTorch standard GQA (no hardware constraints)
    python examples/flash_decoding/gqa_decode_hw_constraint_analysis.py
{separator}
"""
        print(error_msg)
        import sys
        sys.exit(1)
    
    # Ensure block_H does not exceed kv_group_num
    if block_H > kv_group_num:
        block_H = kv_group_num
        print(f"INFO: Adjusted block_H to {block_H} (kv_group_num={kv_group_num})")
    
    assert kv_group_num % block_H == 0, f"kv_group_num ({kv_group_num}) must be divisible by block_H ({block_H})"

    block_N = 128
    
    print(f"INFO: kv_group_num={kv_group_num}, block_H={block_H}, block_N={block_N}")
    print(f"      H_per_block={block_H}, H_per_block//2={block_H//2} >= {MIN_V_BLOCK} ✓")

    func = flashattn_gqa_decode(batch, heads, groups, kv_seqlen, dim, block_H, block_N)

    q = torch.randn(batch, heads, dim, dtype=torch.float16)
    k = torch.randn(batch, kv_seqlen, groups, dim, dtype=torch.float16)
    v = torch.randn(batch, kv_seqlen, groups, dim, dtype=torch.float16)

    block_num = batch * groups * (kv_group_num // block_H)
    workspace_1 = torch.zeros((block_num, block_H, block_N), dtype=torch.float)
    workspace_2 = torch.zeros((block_num, block_H, block_N), dtype=torch.float16)
    workspace_3 = torch.zeros((block_num, block_H, dim), dtype=torch.float)

    torch.npu.synchronize()
    print("init successful!")

    output = func(q, k, v, workspace_1, workspace_2, workspace_3)
    ref_output = ref_program(q, k, v)

    torch.npu.synchronize()

    print("TileLang output shape:", output.shape)
    print("Reference output shape:", ref_output.shape)

    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print("Test Passed!")