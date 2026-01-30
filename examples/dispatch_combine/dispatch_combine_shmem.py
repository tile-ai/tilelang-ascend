import argparse
import tilelang
import tilelang.language as T
import torch
import shmem as aclshmem_module
import multiprocessing as mp
import random
from multiprocessing import Barrier
tilelang.cache.clear_cache()
G_IP_PORT = "tcp://xxx.xxx.xxx.xxx:xxxx"    # 填写IP和端口
g_ash_size = 1024 * 1024 * 1024
pass_configs = {
    tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
#--- 1. 实现 Dispatch 算子逻辑 ---
@tilelang.jit(out_idx=[4,5,6,7], pass_configs=pass_configs)
def moe_dispatch_kernel(
    Bs,     # token总数量
    H,      # token长度
    K,      # 每个token发送moe专家数
    ep_world_size,  # 总die数
    local_expert_num,   # 每个rank的moe专家数
    rank,   # 当前rank id
    ub_size,    # win数据区单行size
    aiv_num,    # 启用v核数
):
    total_expert_num = ep_world_size * local_expert_num     # 总moe专家数
    assist_size = 3     # 三元组size
    ub_align = 32   # ub需要32字节对齐
    ub_float_int32_align = 8    
    status_per_core = (total_expert_num + aiv_num - 1) // aiv_num   # 每个v核要处理的状态数，整除向上对齐
    # 计算当前发往的moe专家当前准备收到的token数
    @T.macro
    def cal_token_send_expert_cnt(
        dst_expert_id,
        cal_cnt,
        dst_expert_id_ub: T.Tensor([Bs * K], "int32"),
        sub_ub: T.Tensor([Bs * K], "int32"),
        expert_ids_ub: T.Tensor([Bs * K], "int32"),
        tmp_fp_32: T.Tensor([Bs * K], "float"),
        tmp_out_fp_32: T.Tensor([Bs * K], "float"),
        work_local_ub: T.Tensor([total_expert_num * ub_float_align], "float"),
    ):
        T.tile.fill(dst_expert_id_ub, dst_expert_id)
        T.barrier_all()
        T.tile.sub_experiment(sub_ub, expert_ids_ub, dst_expert_id_ub, cal_cnt)
        T.barrier_all()
        T.reinterpretcast(tmp_fp_32, sub_ub, "float")
        T.reinterpretcast(tmp_out_fp_32, dst_expert_id_ub, "float")
        T.tile.abs_experiment(tmp_out_fp_32, tmp_fp_32, cal_cnt)
        T.tile.mins_experiment(sub_ub, dst_expert_id_ub, 1, cal_cnt) 
        T.barrier_all()
        T.tile.reduce_sum_experiment(tmp_out_fp_32, tmp_fp_32, work_local_ub, cal_cnt)
        T.barrier_all()
    # 不同流水间同步函数
    @T.macro
    def sync_func(src, dst, event_id: "str"):
        T.set_flag(src, dst, event_id)
        T.wait_flag(src, dst, event_id)
    @T.prim_func
    def main_dispatch(
        x: T.Tensor([Bs, H], "bfloat16"),   # 发送的token
        expert_ids: T.Tensor([Bs, K], "int32"),     # token要发往的moe expert的index
        win_data: T.Tensor([total_expert_num * Bs, ub_size], "bfloat16"),   # 接收其他rank发送token的shmem空间
        win_status: T.Tensor([total_expert_num, ub_float_int32_align], "float"),    # 接收其他rank发送status的shmem空间
        expand_x_out: T.Tensor([ep_world_size * Bs * local_expert_num, H], "bfloat16"),     # dispatch输出：本卡接收token
        expand_ids: T.Tensor([ep_world_size * Bs * local_expert_num, assist_size], "int32"),    # dispatch输出：本卡接收token的三元组
        ep_receive_count: T.Tensor([total_expert_num], "int32"),    # 当前rank接收token的前缀和+自身接收token量
        expert_token_nums_out: T.Tensor([local_expert_num], "int64"),   # 当前rank的moe expert各自接收token量
        workspace: T.Tensor([aiv_num, ub_float_int32_align], "int32"),  # 存储rank接收token前缀和的Global buffer
    ):
        with T.Kernel(aiv_num // 2, is_npu=True) as (cid, vid):     # 启用kernel逻辑，第一个参数为AI Core数量
            # 分配ub空间
            x_ub = T.alloc_ub([H + (32 + 12) // 2], "bfloat16")     # 将要dispatch的数据，H为token长度，32位预留的量化参数空间，12为三元组大小（4 byte * 3）
            x_ub_cast32 = T.alloc_ub([H + (32 + 12) // 2], "int32")
            x_win_ub = T.alloc_ub([H], "bfloat16")  # 本地win区token——>ub
            expert_ids_ub = T.alloc_ub([Bs* K], "int32")   
            dst_expert_id_ub = T.alloc_ub([Bs* K], "int32") # 用于cal_token_send_expert_cnt，填充目标moe expert id
            sub_ub = T.alloc_ub([Bs* K], "int32")   
            tmp_fp_32 = T.alloc_ub([Bs, K], "float")
            tmp_out_fp_32 = T.alloc_ub([Bs* K], "float")
            work_local_ub = T.alloc_ub([Bs* K], "float")
            win_status_ub = T.alloc_ub([status_per_core * ub_float_int32_align], "int32")    # 存储要发送到其他rank的win状态区的状态
            win_status_fp_ub = T.alloc_ub([status_per_core * ub_float_int32_align], "float")     
            status_sum_ub = T.alloc_ub([status_per_core * ub_float_int32_align], "float")
            status_sum_int_ub = T.alloc_ub([status_per_core * ub_float_int32_align], "int32")
            status_sum_out = T.alloc_ub([status_per_core * ub_float_int32_align], "float")
            status_sum_out_tmp_ub = T.alloc_ub([status_per_core * ub_float_int32_align], "float")
            gather_mask_out_ub = T.alloc_ub([status_per_core], "float")
            receive_count_max_ub = T.alloc_ub([status_per_core], "int32")
            sum_local_ub = T.alloc_ub([aiv_num * ub_align // 4], "int32")
            sum_continue_ub = T.alloc_ub([aiv_num], "int32")
            sum_continue_fp_ub = T.alloc_ub([aiv_num], "float")
            win_status_ub_single = T.alloc_ub([ub_float_int32_align], "float")
            status_sum_on_core_ub = T.alloc_ub([ub_float_int32_align], "float")
            status_sum_on_core_int_ub = T.alloc_ub([ub_float_int32_align], "int32")
            gather_sum_pattern_ub = T.alloc_ub([ub_float_int32_align], "uint32")
            recv_cnt_sum_out_ub = T.alloc_ub([ub_float_int32_align], "float")
            out_count_ub = T.alloc_ub([ub_float_int32_align], "int32")
            status_local_data = T.alloc_ub([assist_size * 2], "bfloat16")
            tmp_triple = T.alloc_ub([assist_size], "int32")
            token_repeat_num = T.alloc_ub([1], "int32")
            cur_vid = T.alloc_ub([1], "int32")
            sum_of_flag = T.alloc_ub([1], "float")
            count = T.alloc_ub([1], "int32")
            win_data_offset = T.alloc_ub([1], "int32")
            state_reset_ub = T.alloc_ub([status_per_core, ub_float_int32_align], "float")
            gather_tmp_ub = T.alloc_ub([1], "uint32")
            begin_idx_ub = T.alloc_ub([1], "int32")
            state_reset_floor_ub = T.alloc_ub([status_per_core - 1, ub_float_int32_align], "float")
            receive_count_floor_ub = T.alloc_ub([status_per_core - 1], "int32")
            cur_vid[0] = (vid + 2 * cid)
            cur_send_token_cnt = Bs * K
            with T.Scope("C"):
                T.sync_all()
                T.sync_all()
                T.sync_all()
            with T.Scope("V"):
                # 发送数据分核
                send_token_num = cur_send_token_cnt // aiv_num
                remainder_token_num = cur_send_token_cnt % aiv_num
                start_send_token_id = send_token_num * cur_vid[0]
                start_send_token_id = T.if_then_else(cur_vid[0] < remainder_token_num, start_send_token_id + cur_vid[0], start_send_token_id + remainder_token_num)
                send_token_num = T.if_then_else(cur_vid[0] < remainder_token_num, send_token_num + 1, send_token_num)
                T.tile.fill(state_reset_ub, 0.0)
                T.tile.fill(state_reset_floor_ub, 0.0)
                T.copy(expert_ids[0,0], expert_ids_ub)
                sync_func("mte2", "s", 0)
                token_repeat_num[0] = 0
                # 发送数据AlltoAllDispatch
                for cur_send_token_id in range(start_send_token_id, start_send_token_id + send_token_num):
                    cal_token_send_expert_cnt(expert_ids_ub[cur_send_token_id], cur_send_token_id, dst_expert_id_ub, sub_ub, expert_ids_ub, tmp_fp_32, tmp_out_fp_32, work_local_ub)
                    token_repeat_num[0] = cur_send_token_id - dst_expert_id_ub[0]
                    if (cur_send_token_id == 0):
                        token_repeat_num[0] = 0
                    dest_rank_id = expert_ids_ub[cur_send_token_id] // local_expert_num
                    dest_expert_id = expert_ids_ub[cur_send_token_id] % local_expert_num
                    sync_func("s", "mte2", 1)
                    T.copy(x[cur_send_token_id // K, 0], x_ub)
                    # 计算三元组
                    token_in_topkid = cur_send_token_id % K
                    sync_func("mte2", "v", 2)
                    T.tile.reinterpretcast(x_ub_cast32, x_ub, "int32_t")
                    sync_func("s", "mte3", 3)
                    x_ub_cast32[(H + 16) // 2] = rank
                    x_ub_cast32[(H + 16) // 2 + 1] = cur_send_token_id // K
                    x_ub_cast32[(H + 16) // 2 + 2] = token_in_topkid
                    sync_func("s", "mte3", 4)
                    T.shmem_ub_put_nbi(x_ub, win_data, ub_size, dest_rank_id, (rank * Bs * local_expert_num + dest_expert_id * Bs + token_repeat_num[0]) * ub_size)
                # 发状态SetStatus
                # 状态分核
                aiv_expert_num = total_expert_num // aiv_num
                remainder_expert_num = total_expert_num % aiv_num
                start_expert_id = aiv_expert_num * cur_vid[0]
                start_expert_id = T.if_then_else(cur_vid[0] < remainder_expert_num, start_expert_id + cur_vid[0], start_expert_id + remainder_expert_num)
                aiv_expert_num = T.if_then_else(cur_vid[0] < remainder_expert_num, aiv_expert_num + 1, aiv_expert_num)
                
                total_send_token_num = Bs * K
                for cur_expert_id in range(start_expert_id, start_expert_id + aiv_expert_num):
                    cal_token_send_expert_cnt(cur_expert_id, total_send_token_num, dst_expert_id_ub, sub_ub, expert_ids_ub, tmp_fp_32, tmp_out_fp_32, work_local_ub)                    
                    cnt_pos_index = (cur_expert_id - start_expert_id) * 8
                    win_status_ub[cnt_pos_index + 1] = total_send_token_num - dst_expert_id_ub[0]   # 状态区第二位填发送token数
                    win_status_ub[cnt_pos_index] = 1    # 状态区第一位为flag标志位
                T.barrier_all()
                T.sync_all()    # 保证前面发数据已经全核完毕
                T.reinterpretcast(win_status_fp_ub, win_status_ub, "float")
                T.barrier_all()
                for cur_expert_id in range(start_expert_id, start_expert_id + aiv_expert_num):
                    dest_rank_id = cur_expert_id // local_expert_num    # 状态目标rank
                    local_expert_id = cur_expert_id % local_expert_num  # 状态目标rank的moe expert
                    index = (cur_expert_id - start_expert_id) * 8
                    T.copy(win_status_fp_ub[index:index+8], win_status_ub_single)   # shmem的限制
                    T.shmem_ub_put_nbi(win_status_ub_single, win_status, 8, dest_rank_id, local_expert_id * ep_world_size * 8 + rank * 8)
                    sync_func("mte3", "s", 5)
                # 循环等状态 WaitDispatch
                aiv_expert_fp_num = T.reinterpret("float", aiv_expert_num)
                sum_of_flag[0] = 0.0
                mask = 1
                start_status_index = start_expert_id
                status_num_per_core = aiv_expert_num
                sync_func("mte3", "s", 6)
                while (sum_of_flag[0] != aiv_expert_fp_num):
                    T.copy(win_status[start_status_index, 0], status_sum_ub)
                    sync_func("mte2", "v", 7)
                    T.pipe_barrier("v")
                    T.tile.reduce_sum_mask_experiment(status_sum_out, status_sum_ub, status_sum_out_tmp_ub, mask, status_num_per_core, 1)  # 各核累加所负责状态区的flag位
                    sync_func("v", "s", 8)
                    sum_of_flag[0] = status_sum_out[0]
                sync_func("v", "mte3", 9)
                # 清理状态区
                if status_num_per_core > 0 and status_num_per_core == status_per_core:
                    T.copy(state_reset_ub, win_status[start_status_index, 0])
                elif status_num_per_core > 0 and status_num_per_core == status_per_core - 1:
                    T.copy(state_reset_floor_ub, win_status[start_status_index, 0])
                # SyncCntOnCore 计算当前核的token数量sum的结果，以便GetCumSum计算前缀和使用
                gather_tmp_ub[0] = 2
                mask = 2
                sync_func("s", "v", 10)
                T.tile.gathermask_experiment(gather_mask_out_ub, status_sum_ub, gather_tmp_ub, True, mask, [1, aiv_expert_num, 1, 0], 0)   # 各核累加所负责状态区的token cnt位
                T.pipe_barrier("v")
                rec_status_num_per_core_inner = (aiv_expert_num * 4 + ub_align - 1) // ub_align * ub_align // 4
                T.tile.sum_experiment(status_sum_on_core_ub, gather_mask_out_ub, [1, rec_status_num_per_core_inner, aiv_expert_num])
                T.reinterpretcast(status_sum_on_core_int_ub, status_sum_on_core_ub, "int32_t")
                sync_func("v", "mte3", 11)
                T.copy(status_sum_on_core_int_ub, workspace[cur_vid[0], 0])
                T.barrier_all()
                T.sync_all()
                # GetCumSum 计算当前状态的前缀和，以便知晓前序token量，为本地数据拷贝偏移计算做准备
                T.copy(workspace[0, 0], sum_local_ub)
                T.barrier_all()
                gather_sum_pattern_ub[0] = 1
                T.tile.gathermask_experiment(sum_continue_ub, sum_local_ub, gather_sum_pattern_ub, True, 1, [1, cur_vid[0], 1, 0], 0)
                T.barrier_all()
                T.reinterpretcast(sum_continue_fp_ub, sum_continue_ub, "float")
                T.barrier_all()
                inner_sum_params = (cur_vid[0] * 4 + ub_align - 1) // ub_align * ub_align // 4
                T.tile.sum_experiment(recv_cnt_sum_out_ub, sum_continue_fp_ub, [1, inner_sum_params, cur_vid[0]])
                T.reinterpretcast(out_count_ub, recv_cnt_sum_out_ub, "int32_t")
                T.reinterpretcast(status_sum_int_ub, status_sum_ub, "int32_t")
                if cur_vid[0] == 0:
                    out_count_ub[0] = 0
                begin_idx_ub[0] = out_count_ub[0]
                # 本地数据拷贝LocalWindowCopy
                for i in range(status_num_per_core):
                    begin_idx = begin_idx_ub[0]
                    count = status_sum_int_ub[i * 8 + 1]
                    receive_count_max_ub[i] = begin_idx + count
                    if status_num_per_core == status_per_core - 1:
                        receive_count_floor_ub[i] = begin_idx + count
                    win_data_offset[0] = (i + start_status_index) % ep_world_size * (Bs * local_expert_num) + (i + start_status_index) // ep_world_size * Bs
                    for j in range(count):
                        T.copy(win_data[win_data_offset[0] + j, 0:H], x_win_ub)
                        # 拆解三元组
                        T.copy(win_data[win_data_offset[0] + j, H+16:H+22], status_local_data)
                        sync_func("mte2", "v", 12)
                        T.reinterpretcast(tmp_triple, status_local_data, "int32_t")
                        sync_func("v", "mte3", 13)
                        T.barrier_all()
                        T.copy(tmp_triple, expand_ids[begin_idx + j, 0])
                        T.copy(x_win_ub, expand_x_out[begin_idx + j, 0])
                        T.barrier_all()
                    begin_idx_ub[0] = begin_idx + count # 更新前缀和，以获取输出ep_receive_count
                T.barrier_all()
                # 获取ep_receive_count
                if status_num_per_core > 0 and status_num_per_core == status_per_core:
                    T.copy(receive_count_max_ub, ep_receive_count[start_status_index])
                elif status_num_per_core > 0 and status_num_per_core == status_per_core - 1:
                    T.copy(receive_count_floor_ub, ep_receive_count[start_status_index])
                T.pipe_barrier("mte3")
                # UpdateTokenNumsOut，获取expert_token_nums_out，用更新后的前缀和差值计算
                T.sync_all()
                last_core = T.if_then_else(total_expert_num < aiv_num, total_expert_num - 1, aiv_num - 1)
                if cur_vid[0] == last_core:
                    T.tile.datacachecleanandinvalid_experiment(ep_receive_count, "SINGLE_CACHE_LINE", "CACHELINE_OUT")
                    first_moe_cnt = ep_receive_count[ep_world_size - 1]
                    expert_token_nums_out[0] = first_moe_cnt
                    T.tile.datacachecleanandinvalid_experiment(expert_token_nums_out, "SINGLE_CACHE_LINE", "CACHELINE_OUT")
                    for local_moe_index in range(1, local_expert_num):
                        pre_offset = ep_world_size * (local_moe_index - 1) + ep_world_size - 1
                        cur_offset = ep_world_size * local_moe_index + ep_world_size - 1
                        T.tile.datacachecleanandinvalid_experiment(ep_receive_count, "SINGLE_CACHE_LINE", "CACHELINE_OUT")
                        T.tile.datacachecleanandinvalid_experiment(ep_receive_count, "SINGLE_CACHE_LINE", "CACHELINE_OUT")
                        pre_moe_index_cnt = ep_receive_count[pre_offset]
                        cur_moe_index_cnt = ep_receive_count[cur_offset]
                        token_sums = cur_moe_index_cnt - pre_moe_index_cnt
                        expert_token_nums_out[local_moe_index] = token_sums
                        T.tile.datacachecleanandinvalid_experiment(expert_token_nums_out, "SINGLE_CACHE_LINE", "CACHELINE_OUT")
    return main_dispatch

# --- 2. 实现 Combine 算子逻辑 ---
@tilelang.jit(out_idx=[5], pass_configs=pass_configs)
def moe_combine_kernel(
    Bs,
    send_token_cnt,
    H,
    K,
    ep_world_size,
    local_expert_num,
    rank,
    aiv_num,
):
    assist_size = 3
    token_per_core = (send_token_cnt + aiv_num - 1) // aiv_num
    float_align_ub = 8
    @T.prim_func
    def main_combine(
        expand_x: T.Tensor([send_token_cnt, H], "bfloat16"),    # 当前rank需返还的经过moe expert处理后数据
        assist_info_combine: T.Tensor([send_token_cnt, assist_size], "int32"),      # 返还token的三元组信息
        ep_send_counts: T.Tensor([local_expert_num * ep_world_size], "int32"),
        win_data: T.Tensor([Bs * K, H], "bfloat16"),
        win_status: T.Tensor([Bs * K, float_align_ub], "float"),
        combine_out: T.Tensor([Bs, H], "bfloat16")
    ):
        with T.Kernel(aiv_num // 2, is_npu=True) as (cid, vid):
            cur_vid[0] = vid + 2 * cid
            # 分配ub
            x_ub = T.alloc_ub([H], "bfloat16")
            assist_ub = T.alloc_ub([token_per_core * assist_size], "int32")
            status_ub = T.alloc_ub([float_align_ub], "float")
            state_ub = T.alloc_ub([K * float_align_ub], "float")
            work_local_ub = T.alloc_ub([K * float_align_ub], "float")
            state_sum_out = T.alloc_ub([K * float_align_ub], "float")
            state_reset = T.alloc_ub([K * float_align_ub], "float")
            win_data_ub_bfloat = T.alloc_ub([H], "bfloat16")
            cur_vid = T.alloc_ub([1], "int32")
            sum_of_flag = T.alloc_ub([1], "float")
            # 返还token分核
            aiv_send_token_num = send_token_cnt // aiv_num
            remainder_send_token_num = send_token_cnt % aiv_num
            start_send_token_id = aiv_send_token_num * cur_vid[0]
            start_send_token_id = T.if_then_else(cur_vid[0] < remainder_send_token_num, start_send_token_id + cur_vid[0], start_send_token_id + remainder_send_token_num)
            send_token_num = T.if_then_else(cur_vid[0] < remainder_send_token_num, send_token_num + 1, send_token_num)
            with T.Scope("V"):
                T.tile.fill(status_ub, 1.0)
                T.tile.fill(state_reset, 0.0)
                T.barrier_all()
                T.copy(assist_info_combine[start_send_token_id, 0], assist_ub)
                T.barrier_all()
                for loop in range(send_token_num):
                    tk_index = start_send_token_id + ((loop + rank) % send_token_num)
                    base_offset = (tk_index - start_send_token_id) * assist_size
                    to_rank_id = assist_ub[base_offset]
                    token_id = assist_ub[base_offset + 1]
                    topk_id = assist_ub[base_offset + 2]
                    T.copy(expand_x[tk_index, 0], x_ub)
                    T.barrier_all()
                    win_gm = token_id * K + topk_id
                    T.shmem_ub_put_nbi(x_ub, win_data, H, to_rank_id, win_gm * H)   # 返还数据
                    T.barrier_all()
                    T.shmem_ub_put_nbi(status_ub, win_status, float_align_ub, to_rank_id, win_gm * float_align_ub)  # 返还状态
                T.barrier_all()
                # 本地token分核，以Bs为细粒度
                aiv_token_num = Bs // aiv_num
                remainder_token_num = Bs % aiv_num
                start_token_id = aiv_token_num * cur_vid[0]
                start_token_id = T.if_then_else(cur_vid[0] < remainder_token_num, start_token_id + cur_vid[0], start_token_id + remainder_token_num)
                token_num = T.if_then_else(cur_vid[0] < remainder_send_token_num, token_num + 1, token_num)
                compare_target = K * float_align_ub
                # 循环处理combine返还数据，从win到ub再到global输出
                for cur_idx in range(start_token_id, start_token_id + token_num)
                    sum_of_flag[0] = -1.0
                    state_gm = cur_idx * K
                    while((sum_of_flag[0] < (compare_target - 0.5)) or (sum_of_flag[0] > (compare_target + 0.5))):
                        T.copy(win_status[state_gm, 0], state_ub)
                        T.barrier_all()
                        T.tile.reduce_sum_experiment(state_sum_out, state_ub, work_local_ub, compare_target)
                        sum_of_flag[0] = state_sum_out[0]
                        T.barrier_all()
                    T.copy(state_reset, win_status[state_gm, 0])
                    T.barrier_all()
                    token_index_offset = cur_idx * K
                    T.copy(win_data[token_index_offset, 0], win_data_ub_bfloat)
                    T.barrier_all()
                    T.copy(win_data_ub_bfloat, combine_out[cur_idx, 0])
    return main_combine

def worker(rank, barrier, x, expert_ids):
    print(f"Rank {rank}: Setting device")
    torch.npu.set_device(rank)
    x = x.npu()
    expert_ids = expert_ids.npu()
    token_byte = (H * byte_bf16 + 31) // 32 * 32
    quant_byte = 32
    threeinfo_byte = 3 * 4
    ub_byte = (token_byte + quant_byte + threeinfo_byte + 511) // 512 * 512
    ub_size = ub_byte // 2

    ret = aclshmem_module.set_conf_store_tls(False, "")
    if ret == 0:
        print(f"Rank {rank}: Initialization successful")
        torch.manual_seed(0)
        # 初始化shmem的tensor
        tensorData_dispatch = aclshmem_module.aclshmem_create_tensor([epWorldSize * localExpertNum * Bs, ub_size], dtype = torch.bfloat16, device_id = rank)
        tensorStatus_dispatch = aclshmem_module.aclshmem_create_tensor([epWorldSize * localExpertNum, 8], dtype = torch.float, device_id = rank)
        tensor_combine = aclshmem_module.aclshmem_create_tensor([Bs * K, H], dtype=torch.bfloat16, device_id = rank)
        tensor_combine = aclshmem_module.aclshmem_create_tensor([Bs * K, 8], dtype=torch.float, device_id=rank)
        win_dispatch = tensorData_dispatch.fill_(0)
        winStatus_dispatch = tensorStatus_dispatch.fill_(0)
        win_combine = tensor_combine.fill_(0)
        winStatus_combine = tensorStatus_combine.fill_(0)
        # dispatch
        func_dispatch = moe_dispatch_kernel(Bs, H, K, epWorldSize, localExpertNum, rank, ub_size, aiCoreNum)
        expand_x, expand_idx, ep_recv_counts, expert_token_nums = func_dispatch(x, expert_ids, win_dispatch, winStatus_dispatch)
        barrier.wait()
        # combine
        expand_x = expand_x[:ep_recv_counts[-1].item(), :]
        expand_idx = expand_idx[:ep_recv_counts[-1].item(), :]
        x_out = func_combine(expand_x, expand_idx, ep_recv_counts, win_combine, winStatus_combine)
        barrier.wait()
        print("combine out is", x_out)
        aclshmem_module.aclshmem_free_tensor(tensorData_dispatch)
        aclshmem_module.aclshmem_free_tensor(tensorStatus_dispatch)
        aclshmem_module.aclshmem_free_tensor(tensor_combine)
        aclshmem_module.aclshmem_free_tensor(tensorStatus_combine)
    else:
        print(f"Rank {rank}: Initialization failed with code {ret}")
    aclshmem_module.aclshmem_finialize()
    print(f"Rank {rank}: Finalization")
    # golden
    x_ref = x
    torch.testing.assert_close(x_out, x_ref, rtol=1e-2, atol=1e-2)
    print("Kernel Output Mathc!")

# 计算输入dispatch输入
def init_input(rank, Bs, H, K, epWorldSize, localExpertNum):
    start = rank * Bs + 1
    end = start + Bs
    x = torch.tensor([[i] for i in range(start, edn)], dtype=torch.bfloat16)
    x = x.repeat(1, H)
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    expert_ids_list = []
    for _ in range(Bs):
        full_range = list(range(0, epWorldSize * localExpertNum))
        random.shuffle(full_range)
        row = full_range[:K]
        expert_ids_list.append(row)
    expert_ids = torch.tensor(expert_ids_list, dtype=torch.int32)
    return x, expert_ids

if __name__ == '__main__':
    Bs = 64     
    H = 7168
    K = 4
    epWorldSize = 16
    localExpertNum = 3
    aivNum = 48
    num_processes = 16
    barrier = Barrier(num_processes)
    processes = []
    for rank in range(num_processes):
        x, expert_ids = init_input(rank, Bs, H, K, epWorldSize, localExpertNum)
        p = mp.Process(target=worker, args=(rank, barrier, x, expert_ids))
        p.start()
        process.append(p)
    for p in process:
        p.join()
    print("All processes completed")