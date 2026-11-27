import tilelang
from tilelang import DataType, language as T
import torch
from reference import naive_nsa

torch.set_default_device('npu')
torch.manual_seed(0)

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True
}

@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def native_sparse_attention(
    batch, kv_heads, seq_len, dim, is_causal, scale=None, block_size=64, groups=1, selected_blocks=1
):

    dtype = "float16"
    accum_dtype = "float"
    block_indices_dtype = "int32"

    q_shape = [batch, seq_len, kv_heads * groups, dim]  # B,S,G*KVN,D
    kv_shape = [batch, seq_len, kv_heads, dim]  # B,S,KVN,D
    block_indices_shape = [batch, seq_len, kv_heads, selected_blocks]  # query的每个token选取的block的ID
    
    # 开启的任务数：B * KVN * QS
    # 每个query的token对应的KV的block不一定，因此需要每个token单独计算
    block_num = batch * seq_len * kv_heads

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            K: T.Tensor(kv_shape, dtype),  # type: ignore
            V: T.Tensor(kv_shape, dtype),  # type: ignore
            BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
            Output: T.Tensor(q_shape, dtype),  # type: ignore
            workspace_1: T.Tensor([block_num, groups, block_size], accum_dtype),  # 存放L0C的mm1结果
            workspace_2: T.Tensor([block_num, groups, block_size], dtype),  # 存放softmax结果
            workspace_3: T.Tensor([block_num, groups, dim], accum_dtype),  # 存放mm2结果
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % seq_len  # seq index
            by = cid // seq_len % kv_heads  # head index
            bz = cid // seq_len // kv_heads % batch  # batch index

            q_l1 = T.alloc_L1([groups, dim], dtype)
            k_l1 = T.alloc_L1([block_size, dim], dtype)
            v_l1 = T.alloc_L1([block_size, dim], dtype)

            acc_p_l1 = T.alloc_L1([groups, block_size], dtype)  # mm2左矩阵，P矩阵

            acc_mm1_l0c = T.alloc_L0C([groups, block_size], accum_dtype)  # mm1结果
            acc_mm2_l0c = T.alloc_L0C([groups, dim], accum_dtype)  # mm2结果

            acc_vec2_ub = T.alloc_ub([groups // 2, dim], accum_dtype)  # vec2结果[16*128*4 = 8192]
            sumexp = T.alloc_ub([groups // 2], accum_dtype)  # softmax sum[16*4 = 64]
            max_i = T.alloc_ub([groups // 2], accum_dtype)  # softmax max,每一行的max[16*4 = 64]
            max_i_prev = T.alloc_ub([groups // 2], accum_dtype)  # 用于更新flash结果[16*4 = 64]

            acc_vec1_ub = T.alloc_ub([groups // 2, block_size], accum_dtype)  # [16*64*4 = 4096]
            acc_vec1_ub_ = T.alloc_ub([groups // 2, block_size], accum_dtype) # vec1运算，将mm1结果从GM搬到UB中[16*64*4 = 4096]

            tmp_ub = T.alloc_ub([3 * DataType(accum_dtype).bits // 8 * groups // 2 * block_size], "uint8") # [3*4*16*64 = 12288]
            sumexp_i_ub = T.alloc_ub([groups // 2], accum_dtype)  # [16*4 = 64]
            acc_s_half = T.alloc_ub([groups // 2, block_size], dtype)  # P矩阵从FP32 --> FP16 [16*2 = 32]
            acc_o_ub = T.alloc_ub([groups // 2, dim], accum_dtype)  # 存放mm2结果UB [16*128*4 = 8192]
            acc_o_half = T.alloc_ub([groups // 2, dim], dtype)  # 存放vec2输出结果 [16*128*2 = 4096]

            T.annotate_address({
                # L1 address
                q_l1: 0,
                k_l1: groups * dim * DataType(dtype).bits // 8,  # 单位Byte
                v_l1: (groups + block_size) * dim * DataType(dtype).bits // 8,
                acc_p_l1: (groups + 2 * block_size) * dim * DataType(dtype).bits // 8,

                # L0C address
                acc_mm1_l0c: 0,
                acc_mm2_l0c: 0,

                ## ub address
                acc_vec2_ub: 0,
                sumexp: 8192,
                max_i: 8256,
                max_i_prev: 8320,
                acc_vec1_ub: 8384,
                acc_vec1_ub_: 12480,
                tmp_ub: 12480,
                sumexp_i_ub: 24768,
                acc_s_half: 24832,
                acc_o_ub: 24864,
                acc_o_half: 33056
            })


            # layout BSND
            T.copy(Q[bz, bx, by * groups:(by + 1) * groups, :], q_l1)
            for k in T.serial(selected_blocks):  # 循环读取block_size大小块
                block_start = BlockIndices[bz, bx, by, k] * block_size  # 每个block的起点
                T.copy(K[bz, bx, block_start:block_start + block_size, :], k_l1)
                T.gemm_v0(q_l1, k_l1, acc_mm1_l0c, transpose_B=True, init=True)
                T.copy(acc_mm1_l0c, workspace_1[cid, :, :])  # fixpipe？mm1L0c --> workspace1

                # mm2
                T.copy(workspace_2[cid, :, :], acc_p_l1)  # 拷贝P矩阵从GM-->L1
                T.copy(K[bz, bx, block_start:block_start + block_size, :], v_l1)  # [B,S,N,D]
                T.gemm_v0(acc_p_l1, v_l1, acc_mm2_l0c, init=True)
                T.copy(acc_mm2_l0c, workspace_3[cid, :, :])

            # with T.Scope("V"):
            T.tile.fill(acc_vec2_ub, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(max_i, -2**30)
            for _k in T.serial(selected_blocks): # 迭代S2，nsa里面是按照根据block_table去循环
                block_start = BlockIndices[bz, bx, by, k] * block_size  # 每个block的起点
                T.tile.fill(acc_vec1_ub, 0.0)
                # 在此设置mask掩码
                # if is_causal:
                #     for i, j in T.Parallel(groups // 2, block_size):
                #         acc_vec1_ub[i, j] = T.if_then_else(bx >= (block_start + j), 0, -T.infinity(accum_dtype))
                T.copy(max_i, max_i_prev)  # 备份上一轮的全局最大值
                T.copy(workspace_1[cid, vid * groups // 2:vid * groups // 2 + groups // 2, :], acc_vec1_ub_)  # 2个v核各算一半，vid = 0/1
                T.tile.add(acc_vec1_ub, acc_vec1_ub, acc_vec1_ub_)  # mask操作
                T.tile.mul(acc_vec1_ub, acc_vec1_ub, scale)  # mul scale
                T.tile.reduce_max(max_i, acc_vec1_ub, tmp_ub, dim=-1)  # 求每一行的最大值
                T.tile.max(max_i, max_i, max_i_prev) # 全局最大值，max_i
                T.tile.sub(max_i_prev, max_i_prev, max_i)  # max_prev - x_max
                T.tile.exp(max_i_prev, max_i_prev)  # exp(max_prev - x_max)

                for h_i in range(groups // 2):
                    T.tile.sub(acc_vec1_ub[h_i, :], acc_vec1_ub[h_i, :], max_i[h_i])  # x - x_max

                T.tile.exp(acc_vec1_ub, acc_vec1_ub)  # exp(x - x_max) P矩阵
                T.tile.reduce_sum(sumexp_i_ub, acc_vec1_ub, tmp_ub, dim=-1) # 求每一行的和
                T.tile.mul(sumexp, sumexp, max_i_prev)  # prev_sum = prev_sum * exp(max_prev - x_max)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)  # prev_sum + cur_sum

                for h_i in range(groups // 2):
                    T.tile.mul(acc_vec2_ub[h_i, :], acc_vec2_ub[h_i, :], max_i_prev[h_i])  # flash update操作，mul

                T.copy(acc_vec1_ub, acc_s_half) # cast FP32-->FP16
                T.copy(acc_s_half, workspace_2[cid, vid * groups // 2:vid * groups // 2 + groups // 2, :])  # copy UB --> GM
                
                # vec2
                T.copy(workspace_3[cid, vid * groups // 2:vid * groups // 2 + groups // 2, :], acc_o_ub)
                T.tile.add(acc_vec2_ub, acc_vec2_ub, acc_o_ub)  # flash update操作, add

            for h_i in range(groups // 2):
                T.tile.div(acc_vec2_ub[h_i, :], acc_vec2_ub[h_i, :], sumexp[h_i])

            T.copy(acc_vec2_ub, acc_o_half)
            T.copy(acc_o_half, Output[bz, bx, by * groups + vid * groups // 2:by * groups + vid * groups // 2 + groups // 2, :])

    return main

def main():
    torch.random.manual_seed(0)
    B, SEQ_LEN, QN, KVN, D, SEL_BLK = 2, 64, 16, 1, 32, 1
    BLOCK_SIZE = 32
    G = QN // KVN

    scale = (1.0 / D)**0.5

    kernel = native_sparse_attention(
        batch=B,
        heads=KVN,
        seq_len=SEQ_LEN,
        dim=D,
        is_causal=True,
        block_size=BLOCK_SIZE,
        groups=QN // KVN,
        selected_blocks=SEL_BLK,
        scale=scale,
    )

    print(kernel.get_kernel_source())
    
    Q = torch.randn((B, SEQ_LEN, QN, D), dtype=torch.float16)
    K = torch.randn((B, SEQ_LEN, KVN, D), dtype=torch.float16)
    V = torch.randn((B, SEQ_LEN, KVN, D), dtype=torch.float16)

    g_slc = torch.ones((B, SEQ_LEN, QN), dtype=torch.float16)
    g_swa = torch.ones((B, SEQ_LEN, QN), dtype=torch.float16)

    task_nums = B * SEQ_LEN * KVN  # 开启的任务数量
    workspace_1 = torch.zeros((task_nums, 64, 64), dtype=torch.float)
    workspace_2 = torch.zeros((task_nums, 64, 64), dtype=torch.float16)
    workspace_3 = torch.zeros((task_nums, 64, 512), dtype=torch.float)

    block_indices = torch.full((B, SEQ_LEN, KVN, SEL_BLK), SEQ_LEN, dtype=torch.long)
    block_counts = torch.zeros((B, SEQ_LEN, KVN), dtype=torch.long)

    # 生成索引表
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(KVN):
                i_i = torch.randperm(max(1, (t // BLOCK_SIZE)))[:SEL_BLK]
                block_indices[b, t, h, :len(i_i)] = i_i
                block_counts[b, t, h] = (block_indices[b, t, h] != SEQ_LEN).sum().item()
    block_indices = block_indices.sort(-1)[0]

    out = kernel(Q, K, V, block_indices.to(torch.int32), workspace_1, workspace_2, workspace_3)

    ref = naive_nsa(
        q=Q,
        k=K,
        v=V,
        g_slc=g_slc,
        g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=BLOCK_SIZE,
        scale=scale,
    )

    print("out", out)
    print("ref", ref)
    torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)

    print("Test Passed!")

if __name__ == "__main__":
    main()