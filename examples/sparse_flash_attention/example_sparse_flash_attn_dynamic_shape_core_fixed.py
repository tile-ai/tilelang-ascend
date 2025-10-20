import tilelang
from tilelang import DataType, language as T
import torch

torch.set_default_device('npu')
torch.manual_seed(0)

tilelang.disable_cache()

core_num = 24


@tilelang.jit(out_idx=[3])
def sparse_attention_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_stride,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
):
    assert dim == tilelang.math.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal, 'non-casual is not supported'
    assert topk % block_I == 0, 'otherwise will load some index=0 thus causing wrong kv to be loaded'

    # NOTE: ascend only support exp interface instead of exp2
    sm_scale = (1.0 / (dim + tail_dim))**0.5 if sm_scale is None else sm_scale

    batch = T.symbolic("batch")
    # batch = 2
    seq_len = T.symbolic("seq_len")
    # seq_len = 273

    # seq_len_kv = 44444 # T.symbolic("seq_len_kv")
    seq_len_kv = T.symbolic("seq_len_kv")
    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    # lse_shape = [batch, seq_len, heads]
    indices_dtype = "int32"
    dtype = "float16"
    accum_dtype = "float"

    H = head_kv

    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, 'here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)'

    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, 'head_kv should be a multiple of 64'
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    v_block = H_per_block // 2

    block_num = [batch, seq_len, REPLICATE_H, kv_group]

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            workspace_1: T.Tensor([core_num, BI, D],
                                  dtype),  # T.Tensor([block_num, BI, D], dtype),
            workspace_2: T.Tensor([core_num, BI, D_tail],
                                  dtype),  # T.Tensor([block_num, BI, D_tail], dtype),
            workspace_3: T.Tensor(
                [core_num, H_per_block, BI],
                accum_dtype),  # T.Tensor([block_num, H_per_block, BI], accum_dtype),
            workspace_4: T.Tensor([core_num, H_per_block, BI],
                                  dtype),  # T.Tensor([block_num, H_per_block, BI], dtype),
            workspace_5: T.Tensor(
                [core_num, H_per_block, D],
                accum_dtype),  # T.Tensor([block_num, H_per_block, D], accum_dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            # Alloc Memory
            q_l1 = T.alloc_L1([H_per_block, D], dtype)
            q_tail_l1 = T.alloc_L1([H_per_block, D_tail], dtype)
            kv_l1 = T.alloc_L1([BI, D], dtype)
            kv_tail_l1 = T.alloc_L1([BI, D_tail], dtype)
            acc_s_l1 = T.alloc_L1([H_per_block, BI], dtype)

            acc_s_l0c = T.alloc_L0C([H_per_block, BI], accum_dtype)
            acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

            ## 2. Vector
            acc_o = T.alloc_ub([v_block, D], accum_dtype)
            sumexp = T.alloc_ub([v_block], accum_dtype)
            m_i = T.alloc_ub([v_block], accum_dtype)
            indices_ub_ = T.alloc_ub([BI], indices_dtype)
            kv_ub = T.alloc_ub([D], dtype)
            kv_tail_ub = T.alloc_ub([D_tail], dtype)
            acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
            m_i_prev = T.alloc_ub([v_block], accum_dtype)
            acc_s_ub_ = T.alloc_ub([v_block, BI], accum_dtype)
            tmp_ub = T.alloc_ub([3 * DataType(accum_dtype).bits // 8 * v_block * BI], "uint8")
            sumexp_i_ub = T.alloc_ub([v_block], accum_dtype)
            acc_s_half = T.alloc_ub([v_block, BI], dtype)
            acc_o_ub = T.alloc_ub([v_block, D], accum_dtype)
            acc_o_half = T.alloc_ub([v_block, D], dtype)

            # Currently manually set the address.
            T.annotate_address({
                # L1 address
                q_l1: 0,
                q_tail_l1: 65536,
                kv_l1: 73728,
                kv_tail_l1: 139264,
                acc_s_l1: 139264,

                # L0C address
                acc_s_l0c: 0,
                acc_o_l0c: 0,

                ## ub address
                acc_o: 0,
                sumexp: 65536,
                m_i: 65664,
                indices_ub_: 65792,
                kv_ub: 66048,
                kv_tail_ub: 67072,
                acc_s_ub: 66048,
                m_i_prev: 74240,
                acc_s_ub_: 74368,
                tmp_ub: 74368,
                sumexp_i_ub: 98944,
                acc_s_half: 98944,
                acc_o_ub: 98944,
                acc_o_half: 98944
            })

            # fixed core
            for core_index in T.serial(T.ceildiv(seq_len * REPLICATE_H * batch * kv_group, core_num)):
                pid = core_index * core_num + cid
                if pid < seq_len * REPLICATE_H * batch * kv_group:
                    bx = pid % (seq_len * REPLICATE_H)
                    by = pid // (seq_len * REPLICATE_H) % batch
                    bz = pid // (seq_len * REPLICATE_H) // batch % kv_group
                    
                    b_i = by
                    g_i = bz

                    s_i = (bx // REPLICATE_H)
                    h_i = (bx % REPLICATE_H)

                    H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
                    H1 = H0 + H_per_block

                    with T.Scope("C"):
                        T.copy(Q[b_i, s_i, H0:H1, :D], q_l1)
                        T.copy(Q[b_i, s_i, H0:H1, D:], q_tail_l1)
                        T.barrier_all()
                        for _ in T.serial(NI):
                            T.wait_cross_flag(0)
                            T.barrier_all()
                            T.copy(workspace_1[cid, 0:BI, 0:D], kv_l1)
                            T.copy(workspace_2[cid, 0:BI, 0:D_tail], kv_tail_l1)
                            T.barrier_all()

                            T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True)
                            T.barrier_all()
                            T.gemm_v0(q_tail_l1, kv_tail_l1, acc_s_l0c, transpose_B=True)
                            T.barrier_all()

                            T.copy(acc_s_l0c, workspace_3[cid, 0:H_per_block, 0:BI])
                            T.barrier_all()
                            T.set_cross_flag("FIX", 1)

                            T.wait_cross_flag(2)
                            T.barrier_all()

                            T.copy(workspace_4[cid, 0:H_per_block, 0:BI], acc_s_l1)
                            T.barrier_all()

                            T.gemm_v0(acc_s_l1, kv_l1, acc_o_l0c, init=True)
                            T.barrier_all()

                            T.copy(acc_o_l0c, workspace_5[cid, 0:H_per_block, 0:D])
                            T.barrier_all()

                            T.set_cross_flag("FIX", 3)
                            T.wait_cross_flag(4)
                        # T.wait_cross_flag(8)

                    with T.Scope("V"):

                        T.fill(acc_o, 0.0)
                        T.fill(sumexp, 0.0)
                        T.fill(m_i, -2.0**30)
                        T.barrier_all()

                        for i_i in range(NI):
                            T.copy(Indices[b_i, s_i, g_i, i_i * BI:i_i * BI + BI], indices_ub_)
                            T.barrier_all()

                            for bi_i in range(BI // 2):
                                T.copy(KV[b_i, indices_ub_[bi_i + vid * BI // 2], g_i, :D], kv_ub)
                                T.copy(KV[b_i, indices_ub_[bi_i + vid * BI // 2], g_i, D:], kv_tail_ub)
                                T.barrier_all()
                                T.copy(kv_ub, workspace_1[cid, bi_i + vid * BI // 2, :])
                                T.copy(kv_tail_ub, workspace_2[cid, bi_i + vid * BI // 2, :])
                                T.barrier_all()

                            T.set_cross_flag("MTE3", 0)

                            T.fill(acc_s_ub, 0.0)
                            T.barrier_all()

                            T.copy(m_i, m_i_prev)
                            T.barrier_all()

                            T.wait_cross_flag(1)
                            T.copy(
                                workspace_3[cid, vid * v_block:vid * v_block + v_block, :],
                                acc_s_ub_)
                            T.barrier_all()

                            T.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                            T.barrier_all()

                            T.mul(acc_s_ub, acc_s_ub, sm_scale)
                            T.barrier_all()

                            T.reduce_max(m_i, acc_s_ub, tmp_ub, dim=-1)
                            T.barrier_all()

                            T.max(m_i, m_i, m_i_prev)
                            T.barrier_all()

                            # alpha_ub = m_i_prev

                            T.sub(m_i_prev, m_i_prev, m_i)
                            T.barrier_all()

                            T.exp(m_i_prev, m_i_prev)
                            T.barrier_all()

                            for h_i in range(v_block):
                                T.barrier_all()
                                T.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])  # -
                                T.barrier_all()

                            T.exp(acc_s_ub, acc_s_ub)
                            T.barrier_all()

                            T.reduce_sum(sumexp_i_ub, acc_s_ub, tmp_ub, dim=-1)
                            T.barrier_all()

                            T.mul(sumexp, sumexp, m_i_prev)  # check
                            T.barrier_all()

                            T.add(sumexp, sumexp, sumexp_i_ub)
                            T.barrier_all()

                            for h_i in range(v_block):
                                T.barrier_all()
                                T.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])
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

                            T.add(acc_o, acc_o, acc_o_ub)
                            T.barrier_all()

                            T.set_cross_flag("V", 4)
                            T.barrier_all()

                        for h_i in range(v_block):
                            T.barrier_all()
                            T.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])
                            T.barrier_all()

                        T.copy(acc_o, acc_o_half)
                        T.barrier_all()
                        T.copy(acc_o_half, Output[b_i, s_i, H0 + vid * v_block:H1 + vid * v_block, :])

                        T.barrier_all()

                        # T.set_cross_flag("MTE3", 8)

    return main


func = sparse_attention_fwd(
    heads=128,
    dim=512,
    tail_dim=64,
    topk=2048,
    kv_stride=1,
)


def ref_sparse_attention_fwd_interface(q,
                                       kv,
                                       indices,
                                       q_start_index_s,
                                       kv_stride=4,
                                       sm_scale=None,
                                       is_casual=True):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape
    if q_start_index_s is None:
        q_start_index_s = sk * kv_stride - sq

    assert kv.shape[-1] == 576, 'you should assign dim otherwise'
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    # num_kv_per_index = 1
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(
        q_start_index_s, sq + q_start_index_s, dtype=torch.int32).view(-1, 1) >= torch.arange(
            kv_stride - 1, sk * kv_stride, kv_stride, dtype=torch.int32).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, :kv_stride - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.float16)


B, S, SKV, H, HKV, DQK, DV, topk = 2, 273, 44444, 128, 1, 576, 512, 2048
dtype = torch.float16

KV_stride = 1
q_start_s_index = 4096 * 7

q = torch.randn((B, S, H, DQK), dtype=dtype)
kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype)
indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32)
for b in range(B):
    for t in range(S):
        for h in range(HKV):
            i_i = torch.randperm(max(1, ((t + q_start_s_index) // KV_stride)))[:topk]
            indices[b, t, h, :len(i_i)] = i_i

# output = torch.empty((B, S, H, DV), dtype=dtype)
workspace_1 = torch.zeros((core_num, 64, 512), dtype=dtype)
workspace_2 = torch.zeros((core_num, 64, 64), dtype=dtype)
workspace_3 = torch.zeros((core_num, 64, 64), dtype=torch.float)
workspace_4 = torch.zeros((core_num, 64, 64), dtype=dtype)
workspace_5 = torch.zeros((core_num, 64, 512), dtype=torch.float)

torch.npu.synchronize()
print("init successful!")

output = func(q, kv, indices, workspace_1, workspace_2, workspace_3, workspace_4, workspace_5)

torch.npu.synchronize()

ref_output = ref_sparse_attention_fwd_interface(q, kv, indices, q_start_s_index, KV_stride)
torch.npu.synchronize()
print(f"obviously cmp ref and out, ref:{ref_output}, out:{output}")
torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
print("Test Passed!")
