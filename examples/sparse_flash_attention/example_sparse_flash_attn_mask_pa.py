import tilelang
from tilelang import DataType, language as T
import torch
import sfa_golden as ref
import numpy as np

torch.set_default_device('npu')
torch.manual_seed(0)

tilelang.disable_cache()

@tilelang.jit(out_idx=[3])
def sparse_attention_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    dtype="bfloat16",
    block_num = 516,
    block_size = 128,
    core_num = 24,
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
    seq_len = T.symbolic("seq_len")

    block_table_len = T.symbolic("block_table_len")

    seq_len_kv = T.symbolic("seq_len_kv")
    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [seq_len, kv_group, topk]
    # lse_shape = [batch, seq_len, heads]
    indices_dtype = "int32"
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

    kv_shape = [block_num, block_size, 1, D + D_tail]

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            actual_q_len: T.Tensor([batch], indices_dtype),
            actual_kv_len: T.Tensor([batch], indices_dtype),
            block_table: T.Tensor([batch, block_table_len], indices_dtype),
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
            indices_ub_float = T.alloc_ub([BI], "float")
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
            mask_ub = T.alloc_ub([BI // 8], "uint8")
            mask_ub_2 = T.alloc_ub([BI // 8], "uint8")


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
                indices_ub_float: 66048,
                kv_ub: 66048,
                kv_tail_ub: 67072,
                acc_s_ub: 66048,
                m_i_prev: 74240,
                acc_s_ub_: 74368,
                tmp_ub: 74368,
                sumexp_i_ub: 98944,
                acc_s_half: 98944,
                acc_o_ub: 98944,
                acc_o_half: 98944,
                mask_ub: 164480,
                mask_ub_2: 164512,
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
                    act_q_len = actual_q_len[b_i]
                    # T.barrier_all()
                    if s_i < act_q_len:
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

                        with T.Scope("V"):

                            T.tile.fill(acc_o, 0.0)
                            T.tile.fill(sumexp, 0.0)
                            T.tile.fill(m_i, -2.0**30)
                            T.barrier_all()

                            for i_i in range(NI):
                                T.copy(Indices[s_i, g_i, i_i * BI:i_i * BI + BI], indices_ub_)
                                T.barrier_all()
                                T.copy(indices_ub_, indices_ub_float)
                                T.barrier_all()
                                actual_len = actual_kv_len[b_i]
                                T.barrier_all()
                                valid_kv_len = T.Min(T.float32(s_i), T.float32(actual_len))
                                T.barrier_all()
                                T.tile.compare(mask_ub, indices_ub_float, T.float32(actual_len - act_q_len + s_i), "LE")
                                T.tile.compare(mask_ub_2, indices_ub_float, T.float32(-1.0), "NE")
                                T.barrier_all()
                                T.tile.and_tl(mask_ub, mask_ub, mask_ub_2)

                                for bi_i in range(BI // 2):
                                    index_i = indices_ub_[bi_i + vid * BI // 2]
                                    T.barrier_all()
                                    if index_i > -1:
                                        block_idx = index_i // block_size
                                        block_i = block_table[b_i, block_idx]
                                        block_inter = index_i % block_size
                                        T.barrier_all()
                                        T.copy(KV[block_i, block_inter, 0, :D], kv_ub)
                                        T.copy(KV[block_i, block_inter, 0, D:], kv_tail_ub)
                                    else:
                                        T.tile.fill(kv_ub, 0.0)
                                        T.tile.fill(kv_tail_ub, 0.0)
                                    T.barrier_all()
                                    T.copy(kv_ub, workspace_1[cid, bi_i + vid * BI // 2, :])
                                    T.copy(kv_tail_ub, workspace_2[cid, bi_i + vid * BI // 2, :])
                                    T.barrier_all()

                                T.set_cross_flag("MTE3", 0)

                                T.tile.fill(acc_s_ub_, 0.0)
                                T.barrier_all()

                                for i in T.serial(v_block):
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


core_num = 24

block_num = 516
block_size = 128

func = sparse_attention_fwd(
    heads=128,
    dim=512,
    tail_dim=64,
    topk=2048,
    core_num=core_num,
    block_num=block_num,
    block_size=block_size,
)

print(func.get_kernel_source())
# exit(0)

def ref_sparse_attention_fwd_interface(q,
                                       kv,
                                       indices,
                                       sm_scale=None,):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(
        0, sq, dtype=torch.int32).view(-1, 1) >= torch.arange(
            1 - 1, sk * 1, 1, dtype=torch.int32).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, :1 - 1, 0] = True
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
    return o.to(torch.bfloat16)

B, S, SKV, H, HKV, DQK, DV, topk = 1, 512, 32 * 1024, 128, 1, 576, 512, 2048
dtype = torch.bfloat16


q = torch.randn((B, S, H, DQK), dtype=dtype)
kv = torch.randn((block_num, block_size, 1, DQK), dtype=dtype)
indices = torch.full((S, HKV, topk), -1, dtype=torch.int32)

for t in range(S):
    for h in range(HKV):
        i_i = torch.randperm(max(1, t))[:topk]
        indices[t, h, :len(i_i)] = i_i
torch.npu.synchronize()

workspace_1 = torch.zeros((core_num, 64, 512), dtype=dtype)
workspace_2 = torch.zeros((core_num, 64, 64), dtype=dtype)
workspace_3 = torch.zeros((core_num, 64, 64), dtype=torch.float)
workspace_4 = torch.zeros((core_num, 64, 64), dtype=dtype)
workspace_5 = torch.zeros((core_num, 64, 512), dtype=torch.float)

block_table = torch.zeros((B, SKV // block_size), dtype=torch.int32)

actual_q_len = torch.tensor([S] * B, dtype=torch.int32)
actual_kv_len = torch.tensor([SKV] * B, dtype=torch.int32)

torch.npu.synchronize()
print("init successful!")

output = func(q, kv, indices, actual_q_len, actual_kv_len, block_table, workspace_1, workspace_2, workspace_3, workspace_4, workspace_5)
torch.npu.synchronize()

indices_ref = indices.unsqueeze(0)
import math
scale_value = math.sqrt(1 / DQK)
sparse_block_size = 1
cpu_out = ref.cpu_sparse_flash_attention(
            q[..., 0:512], kv[..., 0:512], kv[..., 0:512], indices_ref, scale_value, sparse_block_size,
            actual_seq_lengths_query=actual_q_len, actual_seq_lengths_kv=actual_kv_len,
            query_rope=q[..., 512:], key_rope=kv[..., 512:],
            layout_query='BSND', layout_kv='PA_BSND', sparse_mode=3, block_table=block_table)

cpu_out = torch.from_numpy(cpu_out).to(dtype).to("npu")

print(f"output:{output}, \nshape:{output.shape}")
print(f"cpu_out:{cpu_out}, \nshape:{cpu_out.shape}")

torch.testing.assert_close(cpu_out, output, rtol=1e-2, atol=1e-2)
print("Test Passed!")