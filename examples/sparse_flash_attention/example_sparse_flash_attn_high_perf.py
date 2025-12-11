import tilelang
from tilelang import DataType, language as T
import torch
from tilelang.profiler import do_bench

torch.set_default_device('npu')
torch.manual_seed(0)

tilelang.disable_cache()


@tilelang.jit(workspace_idx=[4, 5, 6, 7, 8])
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
    block_total_size = batch * seq_len * REPLICATE_H * kv_group
    core_num = 24
    pre_total_size = batch * seq_len
    db = 2
    pre_loop_size = ((144) * 1024 // 2 // ((D + D_tail) * db))
    pre_ub_size = pre_loop_size * D
    pre_rope_ub_size = pre_loop_size * D_tail

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            workspace_1: T.Tensor([batch, seq_len, kv_group, topk, D], dtype),  # T.Tensor([block_num, BI, D], dtype),
            workspace_2: T.Tensor([batch, seq_len, kv_group, topk, D_tail], dtype),  # T.Tensor([block_num, BI, D_tail], dtype),
            workspace_3: T.Tensor([*block_num, H_per_block, BI], accum_dtype),  # T.Tensor([block_num, H_per_block, BI], accum_dtype),
            workspace_4: T.Tensor([*block_num, H_per_block, BI], dtype),  # T.Tensor([block_num, H_per_block, BI], dtype),
            workspace_5: T.Tensor([*block_num, H_per_block, D], accum_dtype),  # T.Tensor([block_num, H_per_block, D], accum_dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
	        #pre
            pre_ub = T.alloc_ub((db, pre_loop_size, D), dtype)
            pre_rope_ub = T.alloc_ub((db, pre_loop_size, D_tail), dtype)
            precore_size = T.ceildiv(pre_total_size, core_num)
            pre_corenum = T.ceildiv(pre_total_size, precore_size)
            pretail_size = pre_total_size - (pre_corenum - 1) * precore_size
            inner_loop_size = pre_loop_size
            inner_loop_count = topk // inner_loop_size // 2
            bs1_start_idx = cid * precore_size
            bs1_end_idx = T.if_then_else(cid == pre_corenum - 1, bs1_start_idx + pretail_size, bs1_start_idx + precore_size)
            with T.Scope("V"):
                if cid < pre_corenum:
                    for bs_idx in T.serial(bs1_start_idx, bs1_end_idx):
                        b_idx = bs_idx // seq_len
                        s_idx = bs_idx % seq_len
                        g_idx = bs_idx // (seq_len * batch) % kv_group
                        for loop in T.serial(inner_loop_count):
                            topk_start_idx = loop * inner_loop_size + vid * inner_loop_count * inner_loop_size
                            topk_end_idx = topk_start_idx + inner_loop_size
                            for topk_idx in T.serial(topk_start_idx, topk_end_idx):
                                T.set_flag("mte2", "s", loop % db)
                                T.wait_flag("mte2", "s", loop % db)
                                T.copy(KV[b_idx, Indices[b_idx, s_idx, g_idx, topk_idx], g_idx, : D], pre_ub[loop % db, (topk_idx - topk_start_idx), :])
                                T.copy(KV[b_idx, Indices[b_idx, s_idx, g_idx, topk_idx], g_idx, D : ], pre_rope_ub[loop % db, (topk_idx - topk_start_idx), :])
    
                            T.set_flag("mte2", "mte3", loop % db)
                            T.wait_flag("mte2", "mte3", loop % db)
                            T.copy(pre_ub[loop % db, 0, 0], workspace_1[b_idx, s_idx, g_idx, topk_start_idx : topk_start_idx + pre_loop_size, :])
                            T.copy(pre_rope_ub[loop % db, 0, 0], workspace_2[b_idx, s_idx, g_idx, topk_start_idx : topk_start_idx + pre_loop_size, :])
 
            T.sync_all()
            q_l1 = T.alloc_L1([H_per_block, D], dtype)
            q_tail_l1 = T.alloc_L1([H_per_block, D_tail], dtype)
            kv_l1 = T.alloc_L1([BI, D], dtype)
            kv_tail_l1 = T.alloc_L1([BI, D_tail], dtype)
            acc_s_l1 = T.alloc_L1([H_per_block, BI], dtype)

            acc_s_l0c = T.alloc_L0C([H_per_block, BI], accum_dtype)
            acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

            # 2. Vector
            acc_o = T.alloc_ub([v_block, D], accum_dtype)
            sumexp = T.alloc_ub([v_block], accum_dtype)
            m_i = T.alloc_ub([v_block], accum_dtype)
            acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
            m_i_prev = T.alloc_ub([v_block], accum_dtype)
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
                acc_s_ub: 66048,
                m_i_prev: 74240,
                tmp_ub: 74368,
                sumexp_i_ub: 98944,
                acc_s_half: 98944,
                acc_o_ub: 98944,
                acc_o_half: 98944
            })

            normcore_size = T.ceildiv(block_total_size, core_num)
            used_corenum = T.ceildiv(block_total_size, normcore_size)
            tailcore_size = block_total_size - (used_corenum - 1) * normcore_size
            block_start_idx = cid * normcore_size
            block_end_idx = T.if_then_else(cid == used_corenum - 1, block_start_idx + tailcore_size, block_start_idx + normcore_size)
            if cid < used_corenum:
                for block_idx in T.serial(block_start_idx, block_end_idx):
                    bx = block_idx % (seq_len * REPLICATE_H)
                    by = block_idx // (seq_len * REPLICATE_H) % batch
                    bz = block_idx // (seq_len * REPLICATE_H) // batch % kv_group

                    b_i = by
                    g_i = bz

                    s_i = (bx // REPLICATE_H)
                    h_i = (bx % REPLICATE_H)

                    H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
                    H1 = H0 + H_per_block

                    with T.Scope("C"):
                        # T.wait_cross_flag(0)
                        T.copy(Q[b_i, s_i, H0:H1, :D], q_l1)
                        T.copy(Q[b_i, s_i, H0:H1, D:], q_tail_l1)
                        # T.barrier_all()
                        T.set_flag("m", "mte2", 0)
                        T.set_flag("m", "mte2", 1)
                        for _ in T.serial(NI):
                            T.wait_flag("m", "mte2", 0)
                            T.copy(workspace_1[b_i, s_i, g_i, _ * BI:_ * BI + BI, 0:D], kv_l1)
                            T.copy(workspace_2[b_i, s_i, g_i, _ * BI:_ * BI + BI, 0:D_tail], kv_tail_l1)
                            # T.barrier_all()
                            T.set_flag("mte2", "mte1", 0)
                            T.wait_flag("mte2", "mte1", 0)

                            T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True)
                            # T.barrier_all()
                            T.gemm_v0(q_tail_l1, kv_tail_l1, acc_s_l0c, transpose_B=True)
                            # T.barrier_all()
                            T.set_flag("m", "fix", 0)
                            T.wait_flag("m", "fix", 0)
                            T.set_flag("m", "mte2", 0)

                            T.copy(acc_s_l0c, workspace_3[b_i, s_i, h_i, g_i, 0:H_per_block, 0:BI])
                            # T.barrier_all()
                            T.set_cross_flag("FIX", 1)

                            T.wait_cross_flag(2)
                            # T.barrier_all()
                            T.wait_flag("m", "mte2", 1)

                            T.copy(workspace_4[b_i, s_i, h_i, g_i, 0:H_per_block, 0:BI], acc_s_l1)
                            T.set_flag("mte2", "mte1", 1)
                            T.wait_flag("mte2", "mte1", 1)

                            T.gemm_v0(acc_s_l1, kv_l1, acc_o_l0c, init=True)
                            # T.barrier_all()
                            T.set_flag("m", "fix", 1)
                            T.wait_flag("m", "fix", 1)
                            T.set_flag("m", "mte2", 1)

                            T.copy(acc_o_l0c, workspace_5[b_i, s_i, h_i, g_i, 0:H_per_block, 0:D])
                            # T.barrier_all()

                            T.set_cross_flag("FIX", 3)
                            T.wait_cross_flag(4)
                        T.wait_flag("m", "mte2", 0)
                        T.wait_flag("m", "mte2", 1)
                        T.wait_cross_flag(8)

                    with T.Scope("V"):
                        T.tile.fill(acc_o, 0.0)
                        T.tile.fill(sumexp, 0.0)
                        T.tile.fill(m_i, -2.0**30)

                        for i_i in range(NI):

                            T.pipe_barrier("v")

                            T.copy(m_i, m_i_prev)

                            T.wait_cross_flag(1)
                            T.copy(
                                workspace_3[b_i, s_i, h_i, g_i, vid * v_block:vid * v_block + v_block, :],
                                acc_s_ub)
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)

                            T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                            T.pipe_barrier("v")

                            T.tile.reduce_max(m_i, acc_s_ub, tmp_ub, dim=-1)
                            T.pipe_barrier("v")


                            T.tile.max(m_i, m_i, m_i_prev)
                            T.pipe_barrier("v")

                            # alpha_ub = m_i_prev

                            T.tile.sub(m_i_prev, m_i_prev, m_i)
                            T.pipe_barrier("v")

                            T.tile.exp(m_i_prev, m_i_prev)
                            T.set_flag("v", "s", 0)
                            T.wait_flag("v", "s", 0)

                            for h_i in range(v_block):
                                T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])  # -

                            T.pipe_barrier("v")
                            T.tile.exp(acc_s_ub, acc_s_ub)
                            T.pipe_barrier("v")

                            T.tile.reduce_sum(sumexp_i_ub, acc_s_ub, tmp_ub, dim=-1)
                            T.pipe_barrier("v")

                            T.tile.mul(sumexp, sumexp, m_i_prev)  # check
                            T.pipe_barrier("v")

                            T.tile.add(sumexp, sumexp, sumexp_i_ub)

                            for h_i in range(v_block):
                                T.set_flag("v", "s", 0)
                                T.wait_flag("v", "s", 0)
                                T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])

                            T.copy(acc_s_ub, acc_s_half)
                            T.set_flag("v", "mte3", 0)
                            T.wait_flag("v", "mte3", 0)

                            T.copy(
                                acc_s_half, workspace_4[b_i, s_i, h_i, g_i,
                                                        vid * v_block:vid * v_block + v_block, :])
                            # T.barrier_all()

                            T.set_cross_flag("MTE3", 2)

                            T.wait_cross_flag(3)
                            # T.barrier_all()

                            T.copy(
                                workspace_5[b_i, s_i, h_i, g_i, vid * v_block:vid * v_block + v_block, :],
                                acc_o_ub)
                            T.set_flag("mte2", "v", 1)
                            T.wait_flag("mte2", "v", 1)

                            T.tile.add(acc_o, acc_o, acc_o_ub)
                            # T.barrier_all()

                            T.set_cross_flag("V", 4)
                            # T.barrier_all()
                        T.set_flag("v", "s", 1)
                        T.wait_flag("v", "s", 1)
                        for h_i in range(v_block):
                            T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])
                            # T.barrier_all()

                        T.copy(acc_o, acc_o_half)
                        # T.barrier_all()
                        T.set_flag("v", "mte3", 1)
                        T.wait_flag("v", "mte3", 1)
                        T.copy(acc_o_half, Output[b_i, s_i, H0 + vid * v_block:H1 + vid * v_block, :])

                        T.set_cross_flag("MTE3", 8)

    return main


func = sparse_attention_fwd(
    heads=128,
    dim=512,
    tail_dim=64,
    topk=2048,
    kv_stride=1,
)
print(f"kernel code in sparseFA.py:{func.get_kernel_source()}")


def ref_sparse_attention_fwd_interface(q,
                                       kv,
                                       indices,
                                       q_start_index_s,
                                       kv_stride=4,
                                       sm_scale=None,
                                       is_casual=True):
    q = q.float()
    kv = kv.float()
    # print(f"indices shape:{indices.shape}")
    indices = indices.transpose(1, 2)
    # print(f"indices shape aft trans:{indices.shape}")
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
    # print(f"before qk, q.shape:{q.shape}, k.shape{k.shape}")
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    # print(f"score shape:{score.shape}")
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
# workspace_1 = torch.zeros((2, 273, 1, 2048, 512), dtype=dtype)
# workspace_2 = torch.zeros((2, 273, 1, 2048, 64), dtype=dtype)
# workspace_3 = torch.zeros((2, 273, 2, 1, 64, 64), dtype=torch.float)
# workspace_4 = torch.zeros((2, 273, 2, 1, 64, 64), dtype=dtype)
# workspace_5 = torch.zeros((2, 273, 2, 1, 64, 512), dtype=torch.float)

torch.npu.synchronize()
print("init successful!")

output = torch.empty((B, S, H, DV), dtype=dtype)
func(q, kv, indices, output)

# torch.npu.synchronize()
# execute_time = do_bench(lambda : func(q, kv, indices, output, workspace_1, workspace_2, workspace_3, workspace_4, workspace_5))
# torch.npu.synchronize()
# print(f"execute_time:{execute_time}")

ref_output = ref_sparse_attention_fwd_interface(q, kv, indices, q_start_s_index, KV_stride)
torch.npu.synchronize()
print(f"obviously cmp ref and out, ref:{ref_output}, out:{output}")
torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
print("Test Passed!")
