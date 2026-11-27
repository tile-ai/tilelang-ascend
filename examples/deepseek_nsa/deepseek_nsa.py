import tilelang
from tilelang import DataType, language as T
import torch
from reference import naive_nsa

torch.set_default_device('npu')
torch.manual_seed(0)

tilelang.disable_cache()

core_num = 24


@tilelang.jit(out_idx=[4], workspace_idx=[5,6,7,8,9])
def deepseek_nsa_fwd(
    heads,
    dim,
    topk,
    kv_stride,
    kv_head=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    dtype="bfloat16"
):
    # NOTE: ascend only support exp interface instead of exp2
    sm_scale = (1.0 / (dim))**0.5 if sm_scale is None else sm_scale

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")

    seq_len_kv = T.symbolic("seq_len_kv")
    groups = heads // kv_head
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len_kv, kv_head, dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_head, topk]
    # lse_shape = [batch, seq_len, heads]
    indices_dtype = "int32"
    accum_dtype = "float"

    H = groups

    padded_H = max(tilelang.math.next_power_of_2(groups), 16)  # matmul对m轴的长度有要求
    if padded_H != H:  # 原因不清楚, 但是应该走不进去
        assert kv_head == 1, 'here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)'

    BI = block_I
    NI = tilelang.cdiv(topk, block_I)  # 类似于S2轴循环的次数
    D = dim

    if groups > 64:
        assert groups % 64 == 0, 'head_kv should be a multiple of 64'
        REPLICATE_H = groups // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64  # G > 64时按照64切块

    v_block = H_per_block // 2  # 两个vector分别拿到的计算量

    block_num = [batch, seq_len, REPLICATE_H, kv_head]  # 总任务数

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            K: T.Tensor(kv_shape, dtype),  # type: ignore
            V: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            workspace_1: T.Tensor([core_num, BI, D], dtype),  # T.Tensor([block_num, BI, D], dtype),
            workspace_2: T.Tensor([core_num, BI, D], dtype),  # T.Tensor([block_num, BI, D_tail], dtype),
            workspace_3: T.Tensor([core_num, H_per_block, BI], accum_dtype),  # T.Tensor([block_num, H_per_block, BI], accum_dtype),
            workspace_4: T.Tensor([core_num, H_per_block, BI], dtype),  # T.Tensor([block_num, H_per_block, BI], dtype),
            workspace_5: T.Tensor([core_num, H_per_block, D], accum_dtype),  # T.Tensor([block_num, H_per_block, D], accum_dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            # Alloc Memory
            q_l1 = T.alloc_L1([H_per_block, D], dtype)
            # q_tail_l1 = T.alloc_L1([H_per_block, D_tail], dtype)
            k_l1 = T.alloc_L1([BI, D], dtype)
            v_l1 = T.alloc_L1([BI, D], dtype)
            # kv_tail_l1 = T.alloc_L1([BI, D_tail], dtype)
            acc_s_l1 = T.alloc_L1([H_per_block, BI], dtype)

            acc_s_l0c = T.alloc_L0C([H_per_block, BI], accum_dtype)
            acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

            ## 2. Vector
            acc_o = T.alloc_ub([v_block, D], accum_dtype)
            sumexp = T.alloc_ub([v_block], accum_dtype)
            m_i = T.alloc_ub([v_block], accum_dtype)
            indices_ub_ = T.alloc_ub([BI], indices_dtype)
            indices_ub_float = T.alloc_ub([BI], "float")
            k_ub = T.alloc_ub([D], dtype)
            v_ub = T.alloc_ub([D], dtype)
            # kv_tail_ub = T.alloc_ub([D_tail], dtype)
            acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
            m_i_prev = T.alloc_ub([v_block], accum_dtype)
            acc_s_ub_ = T.alloc_ub([v_block, BI], accum_dtype)
            tmp_ub = T.alloc_ub([3 * DataType(accum_dtype).bits // 8 * v_block * BI], "uint8")
            sumexp_i_ub = T.alloc_ub([v_block], accum_dtype)
            acc_s_half = T.alloc_ub([v_block, BI], dtype)
            acc_o_ub = T.alloc_ub([v_block, D], accum_dtype)
            acc_o_half = T.alloc_ub([v_block, D], dtype)
            mask_ub = T.alloc_ub([BI // 8], "uint8")


            # Currently manually set the address.
            T.annotate_address({
                # L1 address
                q_l1: 0,                    # 64 * 128
                k_l1: 8192,                 # 64 * 128
                v_l1: 16384,                # 64 * 128
                acc_s_l1: 24576,            # 64 * 64

                # L0C address
                acc_s_l0c: 0,
                acc_o_l0c: 0,

                ## ub address
                acc_o: 0,                   # 16384
                sumexp: 16384,              # 128
                m_i: 16512,                 # 128
                indices_ub_: 16640,         # 256
                indices_ub_float: 16896,    # 256
                k_ub: 17152,                # 256
                v_ub: 17408,                # 256
                acc_s_ub: 17664,            # 8192
                m_i_prev: 25856,            # 128
                acc_s_ub_: 25984,           # 8192
                tmp_ub: 34176,              # 24576
                sumexp_i_ub: 58752,         # 128
                acc_s_half: 58880,          # 4096
                acc_o_ub: 62976,            # 16384
                acc_o_half: 79360,          # 8192
                mask_ub: 87552,
            })

            # fixed core
            for core_index in T.serial(T.ceildiv(seq_len * REPLICATE_H * batch * kv_head, core_num)):
                pid = core_index * core_num + cid
                if pid < seq_len * REPLICATE_H * batch * kv_head:
                    bx = pid % (seq_len * REPLICATE_H)
                    by = pid // (seq_len * REPLICATE_H) % batch
                    bz = pid // (seq_len * REPLICATE_H) // batch % kv_head
                    
                    b_i = by
                    kvn_i = bz

                    s_i = (bx // REPLICATE_H)
                    h_i = (bx % REPLICATE_H)

                    H0 = kvn_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
                    H1 = H0 + H_per_block

                    with T.Scope("C"):
                        T.copy(Q[b_i, s_i, H0:H1, :D], q_l1)
                        T.barrier_all()
                        for _ in T.serial(NI):
                            T.wait_cross_flag(0)
                            T.barrier_all()
                            T.copy(workspace_1[cid, 0:BI, 0:D], k_l1)
                            T.barrier_all()
                            T.copy(workspace_2[cid, 0:BI, 0:D], v_l1)
                            T.barrier_all()

                            T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                            T.barrier_all()

                            T.copy(acc_s_l0c, workspace_3[cid, 0:H_per_block, 0:BI])
                            T.barrier_all()
                            T.set_cross_flag("FIX", 1)

                            T.wait_cross_flag(2)
                            T.barrier_all()

                            T.copy(workspace_4[cid, 0:H_per_block, 0:BI], acc_s_l1)
                            T.barrier_all()

                            T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                            T.barrier_all()

                            T.copy(acc_o_l0c, workspace_5[cid, 0:H_per_block, 0:D])
                            T.barrier_all()

                            T.set_cross_flag("FIX", 3)
                            T.wait_cross_flag(4)

                    with T.Scope("V"):
                        T.tile.fill(acc_o, 0.0)
                        T.tile.fill(sumexp, 0.0)
                        T.tile.fill(m_i, -2.0**30)
                        T.barrier_all()

                        for i_i in range(NI):
                            T.copy(Indices[b_i, s_i, kvn_i, i_i * BI:i_i * BI + BI], indices_ub_)
                            T.barrier_all()
                            T.copy(indices_ub_, indices_ub_float)
                            T.barrier_all()
                            T.tile.compare(mask_ub, indices_ub_float, T.float32(s_i), "LE")
                            T.barrier_all()

                            for bi_i in range(BI // 2):
                                T.copy(K[b_i, indices_ub_[bi_i + vid * BI // 2], kvn_i, :D], k_ub)
                                T.copy(V[b_i, indices_ub_[bi_i + vid * BI // 2], kvn_i, :D], v_ub)
                                # T.copy(KV[b_i, indices_ub_[bi_i + vid * BI // 2], g_i, D:], kv_tail_ub)
                                T.barrier_all()
                                T.copy(k_ub, workspace_1[cid, bi_i + vid * BI // 2, :])
                                T.copy(v_ub, workspace_2[cid, bi_i + vid * BI // 2, :])
                                T.barrier_all()

                            T.set_cross_flag("MTE3", 0)

                            T.tile.fill(acc_s_ub_, 0.0)
                            T.barrier_all()

                            for i in T.serial(v_block):
                                # T.barrier_all()
                                T.tile.select(acc_s_ub[i, :], mask_ub, acc_s_ub_[i, :], -T.infinity(accum_dtype), "VSEL_TENSOR_SCALAR_MODE")
                                T.barrier_all()

                            T.copy(m_i, m_i_prev)
                            T.barrier_all()

                            T.wait_cross_flag(1)
                            T.copy(
                                workspace_3[cid, vid * v_block:vid * v_block + v_block, :],
                                acc_s_ub_)
                            T.barrier_all()

                            T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                            T.barrier_all()

                            T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                            T.barrier_all()

                            T.tile.reduce_max(m_i, acc_s_ub, tmp_ub, dim=-1)
                            T.barrier_all()

                            T.tile.max(m_i, m_i, m_i_prev)
                            T.barrier_all()

                            # alpha_ub = m_i_prev

                            T.tile.sub(m_i_prev, m_i_prev, m_i)
                            T.barrier_all()

                            T.tile.exp(m_i_prev, m_i_prev)
                            T.barrier_all()

                            for h_i in range(v_block):
                                T.barrier_all()
                                T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])  # -
                                T.barrier_all()

                            T.tile.exp(acc_s_ub, acc_s_ub)
                            T.barrier_all()

                            T.tile.reduce_sum(sumexp_i_ub, acc_s_ub, tmp_ub, dim=-1)
                            T.barrier_all()

                            T.tile.mul(sumexp, sumexp, m_i_prev)  # check
                            T.barrier_all()

                            T.tile.add(sumexp, sumexp, sumexp_i_ub)
                            T.barrier_all()

                            for h_i in range(v_block):
                                T.barrier_all()
                                T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])
                                T.barrier_all()

                            T.copy(acc_s_ub, acc_s_half)
                            T.barrier_all()

                            T.copy(
                                acc_s_half, workspace_4[cid,
                                                        vid * v_block:vid * v_block + v_block, :])
                            T.barrier_all()

                            T.set_cross_flag("MTE3", 2)

                            T.wait_cross_flag(3)
                            T.barrier_all()

                            T.copy(
                                workspace_5[cid, vid * v_block:vid * v_block + v_block, :],
                                acc_o_ub)
                            T.barrier_all()

                            T.tile.add(acc_o, acc_o, acc_o_ub)
                            T.barrier_all()

                            T.set_cross_flag("V", 4)
                            T.barrier_all()

                        for h_i in range(v_block):
                            T.barrier_all()
                            T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])
                            T.barrier_all()

                        T.copy(acc_o, acc_o_half)
                        T.barrier_all()
                        T.copy(acc_o_half, Output[b_i, s_i, H0 + vid * v_block:H1 + vid * v_block, :])

    return main

if __name__ == "__main__":
    B, S, SKV, H, HKV, DQK, DV = 1, 1024, 1024, 64, 1, 128, 128
    BLOCK_SIZE = 64
    BLOCK_SELECT = 16
    topk = BLOCK_SIZE * BLOCK_SELECT

    dtype = torch.bfloat16

    scale = (1.0 / (DQK))**0.5

    func = deepseek_nsa_fwd(
        heads=64,
        dim=128,
        topk=1024,
        kv_stride=1,
        kv_head=1,
        sm_scale=scale   
    )

    q = torch.randn((B, S, H, DQK), dtype=dtype)
    k = torch.randn((B, SKV, HKV, DQK), dtype=dtype)
    v = torch.randn((B, SKV, HKV, DV), dtype=dtype)
    g_slc = torch.ones((B, S, H), dtype=dtype)
    g_swa = torch.ones((B, S, H), dtype=dtype)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32)
    block_indices = torch.full((B, S, HKV, BLOCK_SELECT), SKV, dtype=torch.int32)
    block_counts = torch.full((B, S, HKV), SKV, dtype=torch.int32)

    # 生成索引表
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, (t // BLOCK_SIZE)))[:BLOCK_SELECT]
                block_indices[b, t, h, :len(i_i)] = i_i
                block_counts[b, t, h] = (block_indices[b, t, h] != SKV).sum().item()
    block_indices = block_indices.sort(-1)[0]

    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                for i in range(BLOCK_SELECT):
                    if block_indices[b, t, h, i] != SKV:
                        indices[b, t, h, i * BLOCK_SIZE : i * BLOCK_SIZE + BLOCK_SIZE] = block_indices[b, t, h, i] * BLOCK_SIZE + torch.arange(BLOCK_SIZE)

    torch.npu.synchronize()

    ref = naive_nsa(
        q=q,
        k=k,
        v=v,
        g_slc=g_slc,
        g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=BLOCK_SIZE,
        scale=scale,
    )


    # output = torch.empty((B, S, H, DV), dtype=dtype)
    # workspace_1 = torch.zeros((core_num, 64, 512), dtype=dtype)
    # workspace_2 = torch.zeros((core_num, 64, 64), dtype=dtype)
    # workspace_3 = torch.zeros((core_num, 64, 64), dtype=torch.float)
    # workspace_4 = torch.zeros((core_num, 64, 64), dtype=dtype)
    # workspace_5 = torch.zeros((core_num, 64, 512), dtype=torch.float)

    torch.npu.synchronize()
    print("init successful!")

    output = func(q, k, v, indices)

    torch.npu.synchronize()

    torch.testing.assert_close(ref, output, rtol=1e-2, atol=1e-2)
    print("Test Passed!")
