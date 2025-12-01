import tilelang
from tilelang import DataType, language as T
import torch

torch.set_default_device('npu')
torch.manual_seed(0)

tilelang.disable_cache()


@tilelang.jit(out_idx=[3])
def sparse_attention_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_stride,
    kv_group=8,
    sm_scale=None,
    is_causal=True,
    block_I=64,
):
    assert dim == tilelang.math.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    assert is_causal, 'non-casual is not supported'
    assert topk % block_I == 0, 'otherwise will load some index=0 thus causing wrong kv to be loaded'

    # NOTE: ascend only support exp interface instead of exp2
    sm_scale = (1.0 / (dim + tail_dim))**0.5 if sm_scale is None else sm_scale

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")
    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    indices_dtype = "int32"
    dtype = "float16"
    accum_dtype = "float"

    H = head_kv

    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, 'head_kv should be a multiple of 64'
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = head_kv if REPLICATE_H == 1 else 64
    v_block = H_per_block // 2
    ub_len = max(32 // (DataType(accum_dtype).bits // 8), v_block)   # UB need 32B align

    block_num = [batch, seq_len, REPLICATE_H, kv_group]

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            workspace_1: T.Tensor([*block_num, BI, D], dtype),
            workspace_2: T.Tensor([*block_num, H_per_block, BI], accum_dtype),
            workspace_3: T.Tensor([*block_num, H_per_block, BI], dtype),
            workspace_4: T.Tensor([*block_num, H_per_block, D], accum_dtype),
    ):
        with T.Kernel(seq_len * REPLICATE_H * batch * kv_group, is_npu=True) as (cid, vid):
            bx = cid % (seq_len * REPLICATE_H)   # S
            by = cid // (seq_len * REPLICATE_H) % batch  # B
            bz = cid // (seq_len * REPLICATE_H) // batch % kv_group  # H

            q_l1 = T.alloc_L1([H_per_block, D], dtype)
            kv_l1 = T.alloc_L1([BI, D], dtype)
            acc_s_l1 = T.alloc_L1([H_per_block, BI], dtype)

            acc_s_l0c = T.alloc_L0C([H_per_block, BI], accum_dtype)
            acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

            ## 2. Vector
            acc_o = T.alloc_ub([v_block, D], accum_dtype)
            sumexp = T.alloc_ub([ub_len], accum_dtype)
            m_i = T.alloc_ub([ub_len], accum_dtype)
            indices_ub_ = T.alloc_ub([BI], indices_dtype)
            kv_ub = T.alloc_ub([D], dtype)
            acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
            m_i_prev = T.alloc_ub([ub_len], accum_dtype)
            acc_s_ub_ = T.alloc_ub([v_block, BI], accum_dtype)
            tmp_ub = T.alloc_ub([3 * DataType(accum_dtype).bits // 8 * v_block * BI], "uint8")
            sumexp_i_ub = T.alloc_ub([ub_len], accum_dtype)
            acc_s_half = T.alloc_ub([v_block, BI], dtype)
            acc_o_ub = T.alloc_ub([v_block, D], accum_dtype)
            acc_o_half = T.alloc_ub([v_block, D], dtype)

            # Currently manually set the address.
            T.annotate_address({
                # L1 address
                q_l1: 0,
                kv_l1: 2048,
                acc_s_l1: 18432,

                # L0C address
                acc_s_l0c: 0,
                acc_o_l0c: 0,

                ## ub address
                acc_o: 0,
                sumexp: 4096,
                m_i: 4160,
                indices_ub_: 4224,
                kv_ub: 4480,
                acc_s_ub: 4736,
                m_i_prev: 6784,
                acc_s_ub_: 6848,
                tmp_ub: 8960,
                sumexp_i_ub: 12032,
                acc_s_half: 12032,
                acc_o_ub: 12032,
                acc_o_half: 12032
            })

            b_i = by
            g_i = bz

            s_i = (bx // REPLICATE_H)
            h_i = (bx % REPLICATE_H)

            heads_per_group = heads // kv_group
            group_start = g_i * heads_per_group
            group_end = (g_i + 1) * heads_per_group
            H0 = group_start
            H1 = group_end

            if REPLICATE_H != 1:
                blocks_in_group = tilelang.cdiv(heads_per_group, H_per_block)
                block_idx_in_group = bx % blocks_in_group
                H0 = group_start + block_idx_in_group * H_per_block
                H1 = T.if_then_else(H0 + H_per_block > group_end, group_end, H0 + H_per_block)

            with T.Scope("C"):
                T.copy(Q[b_i, s_i, H0:H1, :D], q_l1)
                T.barrier_all()
                for _ in T.serial(NI):
                    T.wait_cross_flag(0)
                    T.barrier_all()
                    T.copy(workspace_1[b_i, s_i, h_i, g_i, 0:BI, 0:D], kv_l1)
                    T.barrier_all()

                    T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True)
                    T.barrier_all()

                    T.copy(acc_s_l0c, workspace_2[b_i, s_i, h_i, g_i, 0:heads_per_group, 0:BI])
                    T.barrier_all()
                    T.set_cross_flag("FIX", 1)

                    T.wait_cross_flag(2)
                    T.barrier_all()

                    T.copy(workspace_3[b_i, s_i, h_i, g_i, 0:H_per_block, 0:BI], acc_s_l1)
                    T.barrier_all()

                    T.gemm_v0(acc_s_l1, kv_l1, acc_o_l0c, init=True)
                    T.barrier_all()

                    T.copy(acc_o_l0c, workspace_4[b_i, s_i, h_i, g_i, 0:H_per_block, 0:D])
                    T.barrier_all()

                    T.set_cross_flag("FIX", 3)
                    T.wait_cross_flag(4)
                T.wait_cross_flag(8)

            with T.Scope("V"):

                T.tile.fill(acc_o, 0.0)
                T.tile.fill(sumexp, 0.0)
                T.tile.fill(m_i, -2.0**30)
                T.barrier_all()

                for i_i in range(NI):
                    T.copy(Indices[b_i, s_i, g_i, i_i * BI:i_i * BI + BI], indices_ub_)
                    T.barrier_all()

                    for bi_i in range(BI // 2):
                        T.copy(KV[b_i, indices_ub_[bi_i + vid * BI // 2], g_i, :D], kv_ub)
                        T.barrier_all()
                        T.copy(kv_ub, workspace_1[b_i, s_i, h_i, g_i, bi_i + vid * BI // 2, :])
                        T.barrier_all()

                    T.set_cross_flag("MTE3", 0)

                    T.tile.fill(acc_s_ub, 0.0)
                    T.barrier_all()

                    T.copy(m_i, m_i_prev)
                    T.barrier_all()

                    T.wait_cross_flag(1)
                    T.copy(
                        workspace_2[b_i, s_i, h_i, g_i, vid * v_block:vid * v_block + v_block, :],
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
                        acc_s_half, workspace_3[b_i, s_i, h_i, g_i,
                                                vid * v_block:vid * v_block + v_block, :])
                    T.barrier_all()

                    T.set_cross_flag("MTE3", 2)

                    T.wait_cross_flag(3)
                    T.barrier_all()

                    T.copy(
                        workspace_4[b_i, s_i, h_i, g_i, vid * v_block:vid * v_block + v_block, :],
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

                T.barrier_all()

                T.set_cross_flag("MTE3", 8)

    return main


func = sparse_attention_fwd(
    heads=64,
    dim=128,
    tail_dim=0,
    topk=2048,
    kv_stride=1,
    kv_group=8,
)


def ref_sparse_attention_fwd_interface_gqa(q,
                                           kv,
                                           indices,
                                           q_start_index_s,
                                           kv_stride=4,
                                           sm_scale=None,
                                           is_casual=True):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)    # [b, g, sq, topk]
    b, sq, h_q, dim_q = q.shape
    b, sk, h_kv, _ = kv.shape
    if q_start_index_s is None:
        q_start_index_s = sk * kv_stride - sq

    assert kv.shape[-1] == 128, 'dim should be 128 for GQA'
    dim = 128
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    
    groups = h_q // h_kv
    g_index = h_kv
    h_index = groups
    compressed_casual_mask = torch.arange(
        q_start_index_s, sq + q_start_index_s, dtype=torch.int32).view(-1, 1) >= torch.arange(
            kv_stride - 1, sk * kv_stride, kv_stride, dtype=torch.int32).view(1, -1)

    # create mask, shape [b, g_index, sq, sk]
    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, :kv_stride - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)  # broadcast

    # [b, sq, g, h_per_group, dim_q]
    q = q.view(b, sq, g_index, h_index, dim_q)
    # [b, g, h, sq, sk]
    score = torch.einsum("bqghd,bkgd->bghqk", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    # softmax
    p = score.softmax(dim=-1)  # [b, g, h, sq, sk]
    # [b, g, h, sq, sk] -> [b, h_q, sq, sk]
    p = p.reshape(b, h_q, sq, sk)
    # each kv head repeat h_index times
    v_expanded = v.repeat_interleave(h_index, dim=2)  # [b, sk, h_q, dim_v]
    # output: [b, h_q, sq, dim_v]
    o = torch.einsum("bhqs,bshd->bqhd", p.type(v.dtype), v_expanded)

    return o.to(torch.float16)


B, S, SKV, H_Q, H_KV, DIM, topk = 2, 273, 44444, 64, 8, 128, 2048
kv_group = H_Q // H_KV
dtype = torch.float16

KV_stride = 1
q_start_s_index = 4096 * 7

# create input
q = torch.randn((B, S, H_Q, DIM), dtype=dtype)
kv = torch.zeros((B, SKV, H_KV, DIM), dtype=dtype)
# value for KV head i
for kv_head in range(H_KV):
    kv[:, :, kv_head, :] = torch.rand(DIM, dtype=torch.float16)

# create indice
indices = torch.full((B, S, H_KV, topk), SKV, dtype=torch.int32)
for b in range(B):
    for t in range(S):
        for h in range(H_KV):
            i_i = torch.randperm(max(1, ((t + q_start_s_index) // KV_stride)))[:topk]
            indices[b, t, h, :len(i_i)] = i_i

# output = torch.empty((B, S, H_Q, DIM), dtype=dtype)
workspace_1 = torch.zeros((B, S, 1, H_KV, 64, DIM), dtype=dtype)
workspace_2 = torch.zeros((B, S, 1, H_KV, kv_group, 64), dtype=torch.float)
workspace_3 = torch.zeros((B, S, 1, H_KV, kv_group, 64), dtype=dtype)
workspace_4 = torch.zeros((B, S, 1, H_KV, kv_group, DIM), dtype=torch.float)

torch.npu.synchronize()
print("init successful!")

output = func(q, kv, indices, workspace_1, workspace_2, workspace_3, workspace_4)

torch.npu.synchronize()

ref_output = ref_sparse_attention_fwd_interface_gqa(q, kv, indices, q_start_s_index, KV_stride)
torch.npu.synchronize()
torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)

print("Test Passed!")
