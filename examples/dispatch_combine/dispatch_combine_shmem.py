import argparse
import tilelang
import tilelang.language as T
import torch
import shmem as aclshmem_module
import multiprocessing as mp
import random
from multiprocessing import Barrier
tilelang.cache.clear_cache()
G_IP_PORT = "tcp://xxx.xxx.xxx.xxx:xxxx"    # Enter IP and Port
g_ash_size = 1024 * 1024 * 1024
pass_configs = {
    tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
#--- 1. Implement Dispatch Operator ---
@tilelang.jit(out_idx=[4,5,6,7], workspace_idx=[8], pass_configs=pass_configs)
def moe_dispatch_kernel(
    Bs,     # Total number of tokens
    H,      # token length
    K,      # Number of MOE experts to send per token
    ep_world_size,  # Total number of dies
    local_expert_num,   # Number of MOE experts per rank
    rank,   # Current rank ID
    ub_size,    # Single row size of the win data area
    aiv_num,    # v-core count
):
    total_expert_num = ep_world_size * local_expert_num     # Total MOE experts count
    assist_size = 3     # Triple size
    ub_align = 32   # UB requires 32-byte alignment
    ub_float_int32_align = 8    
    status_per_core = (total_expert_num + aiv_num - 1) // aiv_num   # Number of states to be processed per v-core, rounded up to the nearest integer
    # Calculate the number of tokens currently ready to be received by the MOE expert being dispatched to
    @T.macro
    def cal_token_send_expert_cnt(
        dst_expert_id,
        cal_cnt,
        dst_expert_id_ub: T.Tensor([Bs * K], "int32"),
        sub_ub: T.Tensor([Bs * K], "int32"),
        expert_ids_ub: T.Tensor([Bs * K], "int32"),
        tmp_fp_32: T.Tensor([Bs * K], "float"),
        tmp_out_fp_32: T.Tensor([Bs * K], "float"),
        work_local_ub: T.Tensor([total_expert_num * ub_float_int32_align], "float"),
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
    # Synchronization function between different pipelines
    @T.macro
    def sync_func(src, dst, event_id: "str"):
        T.set_flag(src, dst, event_id)
        T.wait_flag(src, dst, event_id)
    @T.prim_func
    def main_dispatch(
        x: T.Tensor([Bs, H], "bfloat16"),   # Tokens sent
        expert_ids: T.Tensor([Bs, K], "int32"),     # Index of the MOE expert to which the token is to be sent
        win_data: T.Tensor([total_expert_num * Bs, ub_size], "bfloat16"),   # Shared memory space for receiving tokens sent from other ranks
        win_status: T.Tensor([total_expert_num, ub_float_int32_align], "float"),    # Shared memory space for receiving status sent from other ranks
        expand_x_out: T.Tensor([ep_world_size * Bs * local_expert_num, H], "bfloat16"),     # Dispatch output: tokens received by this card
        expand_ids: T.Tensor([ep_world_size * Bs * local_expert_num, assist_size], "int32"),    # Dispatch output: triple of tokens received by this card
        ep_receive_count: T.Tensor([total_expert_num], "int32"),    # Prefix sum of tokens received by the current rank + number of tokens received by itself
        expert_token_nums_out: T.Tensor([local_expert_num], "int64"),   # Number of tokens received by each MOE expert of the current rank
        workspace: T.Tensor([aiv_num, ub_float_int32_align], "int32"),  # Global buffer for storing the prefix sum of tokens received by ranks
    ):
        with T.Kernel(aiv_num // 2, is_npu=True) as (cid, vid):     # Enable kernel logic, with the first parameter being the number of AI Cores
            # Allocate ub space
            x_ub = T.alloc_ub([H + (32 + 12) // 2], "bfloat16")     # Data to be dispatched, H is the token length, 32-bit reserved quantization parameter space, 12 is the triple size (4 bytes * 3)
            x_ub_cast32 = T.alloc_ub([H + (32 + 12) // 2], "int32")
            x_win_ub = T.alloc_ub([H], "bfloat16")  # Local win area token -> ub
            expert_ids_ub = T.alloc_ub([Bs* K], "int32")   
            dst_expert_id_ub = T.alloc_ub([Bs* K], "int32") # Used for cal_token_send_expert_cnt, filling in the target MOE expert ID
            sub_ub = T.alloc_ub([Bs* K], "int32")   
            tmp_fp_32 = T.alloc_ub([Bs* K], "float")
            tmp_out_fp_32 = T.alloc_ub([Bs* K], "float")
            work_local_ub = T.alloc_ub([Bs* K], "float")
            win_status_ub = T.alloc_ub([status_per_core * ub_float_int32_align], "int32")    # Store the status to be sent to the win status area of other ranks
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
                # Send data distributed across cores
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
                # Send data:AlltoAllDispatch
                for cur_send_token_id in range(start_send_token_id, start_send_token_id + send_token_num):
                    cal_token_send_expert_cnt(expert_ids_ub[cur_send_token_id], cur_send_token_id, dst_expert_id_ub, sub_ub, expert_ids_ub, tmp_fp_32, tmp_out_fp_32, work_local_ub)
                    token_repeat_num[0] = cur_send_token_id - dst_expert_id_ub[0]
                    if (cur_send_token_id == 0):
                        token_repeat_num[0] = 0
                    dest_rank_id = expert_ids_ub[cur_send_token_id] // local_expert_num
                    dest_expert_id = expert_ids_ub[cur_send_token_id] % local_expert_num
                    sync_func("s", "mte2", 1)
                    T.copy(x[cur_send_token_id // K, 0], x_ub)
                    # Calculate triple
                    token_in_topkid = cur_send_token_id % K
                    sync_func("mte2", "v", 2)
                    T.reinterpretcast(x_ub_cast32, x_ub, "int32_t")
                    sync_func("v", "s", 3)
                    x_ub_cast32[(H + 16) // 2] = rank
                    x_ub_cast32[(H + 16) // 2 + 1] = cur_send_token_id // K
                    x_ub_cast32[(H + 16) // 2 + 2] = token_in_topkid
                    sync_func("s", "mte3", 4)
                    T.shmem_ub_put_nbi(x_ub, win_data, ub_size, dest_rank_id, (rank * Bs * local_expert_num + dest_expert_id * Bs + token_repeat_num[0]) * ub_size)
                # Send status:SetStatus
                # Status distributed across cores
                aiv_expert_num = total_expert_num // aiv_num
                remainder_expert_num = total_expert_num % aiv_num
                start_expert_id = aiv_expert_num * cur_vid[0]
                start_expert_id = T.if_then_else(cur_vid[0] < remainder_expert_num, start_expert_id + cur_vid[0], start_expert_id + remainder_expert_num)
                aiv_expert_num = T.if_then_else(cur_vid[0] < remainder_expert_num, aiv_expert_num + 1, aiv_expert_num)
                
                total_send_token_num = Bs * K
                for cur_expert_id in range(start_expert_id, start_expert_id + aiv_expert_num):
                    cal_token_send_expert_cnt(cur_expert_id, total_send_token_num, dst_expert_id_ub, sub_ub, expert_ids_ub, tmp_fp_32, tmp_out_fp_32, work_local_ub)                    
                    cnt_pos_index = (cur_expert_id - start_expert_id) * 8
                    win_status_ub[cnt_pos_index + 1] = total_send_token_num - dst_expert_id_ub[0]   # The second field in the status area is filled with the number of tokens sent.
                    win_status_ub[cnt_pos_index] = 1    # The first field in the status area is the flag indicator.
                T.barrier_all()
                T.sync_all()    # Ensure that all cores have completed sending data previously.
                T.reinterpretcast(win_status_fp_ub, win_status_ub, "float")
                T.barrier_all()
                for cur_expert_id in range(start_expert_id, start_expert_id + aiv_expert_num):
                    dest_rank_id = cur_expert_id // local_expert_num    # Status target rank
                    local_expert_id = cur_expert_id % local_expert_num  # MOE expert of the status target rank
                    index = (cur_expert_id - start_expert_id) * 8
                    T.copy(win_status_fp_ub[index:index+8], win_status_ub_single)  
                    T.shmem_ub_put_nbi(win_status_ub_single, win_status, 8, dest_rank_id, local_expert_id * ep_world_size * 8 + rank * 8)
                    sync_func("mte3", "s", 5)
                # Loop waiting for status WaitDispatch
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
                    T.tile.reduce_sum_mask_experiment(status_sum_out, status_sum_ub, status_sum_out_tmp_ub, mask, status_num_per_core, 1)  # Each core accumulates the flag bits of the status area it is responsible for.
                    sync_func("v", "s", 8)
                    sum_of_flag[0] = status_sum_out[0]
                sync_func("v", "mte3", 9)
                # Clear status area
                if status_num_per_core > 0 and status_num_per_core == status_per_core:
                    T.copy(state_reset_ub, win_status[start_status_index, 0])
                elif status_num_per_core > 0 and status_num_per_core == status_per_core - 1:
                    T.copy(state_reset_floor_ub, win_status[start_status_index, 0])
                # SyncCntOnCore: Calculate the sum result of the token count for the current core, to be used by GetCumSum for prefix sum calculation.
                gather_tmp_ub[0] = 2
                mask = 2
                sync_func("s", "v", 10)
                T.tile.gathermask_experiment(gather_mask_out_ub, status_sum_ub, gather_tmp_ub, True, mask, [1, aiv_expert_num, 1, 0], 0)   
                T.pipe_barrier("v")
                rec_status_num_per_core_inner = (aiv_expert_num * 4 + ub_align - 1) // ub_align * ub_align // 4
                T.tile.sum_experiment(status_sum_on_core_ub, gather_mask_out_ub, [1, rec_status_num_per_core_inner, aiv_expert_num])
                T.reinterpretcast(status_sum_on_core_int_ub, status_sum_on_core_ub, "int32_t")
                sync_func("v", "mte3", 11)
                T.copy(status_sum_on_core_int_ub, workspace[cur_vid[0], 0])
                T.barrier_all()
                T.sync_all()
                # GetCumSum: Calculate the prefix sum of the current status to determine the preceding token quantity, preparing for the offset calculation in local data copying.
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
                # Local data copy 
                for i in range(status_num_per_core):
                    begin_idx = begin_idx_ub[0]
                    count = status_sum_int_ub[i * 8 + 1]
                    receive_count_max_ub[i] = begin_idx + count
                    if status_num_per_core == status_per_core - 1:
                        receive_count_floor_ub[i] = begin_idx + count
                    win_data_offset[0] = (i + start_status_index) % ep_world_size * (Bs * local_expert_num) + (i + start_status_index) // ep_world_size * Bs
                    for j in range(count):
                        T.copy(win_data[win_data_offset[0] + j, 0:H], x_win_ub)
                        # Decompose triple
                        T.copy(win_data[win_data_offset[0] + j, H+16:H+22], status_local_data)
                        sync_func("mte2", "v", 12)
                        T.reinterpretcast(tmp_triple, status_local_data, "int32_t")
                        sync_func("v", "mte3", 13)
                        T.barrier_all()
                        T.copy(tmp_triple, expand_ids[begin_idx + j, 0])
                        T.copy(x_win_ub, expand_x_out[begin_idx + j, 0])
                        T.barrier_all()
                    begin_idx_ub[0] = begin_idx + count # Update prefix sum to obtain output ep_receive_count
                T.barrier_all()
                # Obtain ep_receive_count
                if status_num_per_core > 0 and status_num_per_core == status_per_core:
                    T.copy(receive_count_max_ub, ep_receive_count[start_status_index])
                elif status_num_per_core > 0 and status_num_per_core == status_per_core - 1:
                    T.copy(receive_count_floor_ub, ep_receive_count[start_status_index])
                T.pipe_barrier("mte3")
                # UpdateTokenNumsOut，Obtain expert_token_nums_out, calculated using the difference of the updated prefix sum.
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

# --- 2. Implement the Combine operator ---
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
        expand_x: T.Tensor([send_token_cnt, H], "bfloat16"),    # Data processed by the MOE expert to be returned by the current rank
        assist_info_combine: T.Tensor([send_token_cnt, assist_size], "int32"),      # Triple information of returned tokens
        ep_send_counts: T.Tensor([local_expert_num * ep_world_size], "int32"),
        win_data: T.Tensor([Bs * K, H], "bfloat16"),
        win_status: T.Tensor([Bs * K, float_align_ub], "float"),
        combine_out: T.Tensor([Bs, H], "bfloat16")
    ):
        with T.Kernel(aiv_num // 2, is_npu=True) as (cid, vid):
            # Allocate ub
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

            cur_vid[0] = vid + 2 * cid
            # Logic for distributing returned tokens across cores
            send_token_num = send_token_cnt // aiv_num
            remainder_send_token_num = send_token_cnt % aiv_num
            start_send_token_id = send_token_num * cur_vid[0]
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
                    T.shmem_ub_put_nbi(x_ub, win_data, H, to_rank_id, win_gm * H)   # Return data
                    T.barrier_all()
                    T.shmem_ub_put_nbi(status_ub, win_status, float_align_ub, to_rank_id, win_gm * float_align_ub)  # Return status
                T.barrier_all()
                # Logic for distributing local tokens across cores, with Bs as the granularity.
                token_num = Bs // aiv_num
                remainder_token_num = Bs % aiv_num
                start_token_id = token_num * cur_vid[0]
                start_token_id = T.if_then_else(cur_vid[0] < remainder_token_num, start_token_id + cur_vid[0], start_token_id + remainder_token_num)
                token_num = T.if_then_else(cur_vid[0] < remainder_token_num, token_num + 1, token_num)
                compare_target = K * float_align_ub
                # Loop processing combine returned data, from win to ub to global output.
                for cur_idx in range(start_token_id, start_token_id + token_num):
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

def worker(rank, barrier, x, expert_ids, aiv_num, ep_world_size, local_expert_num, Bs):
    print(f"Rank {rank}: Setting device")
    torch.npu.set_device(rank)
    x = x.npu()
    expert_ids = expert_ids.npu()
    byte_bf16 = torch.tensor([], dtype=torch.float16).element_size()
    token_byte = (H * byte_bf16 + 31) // 32 * 32
    quant_byte = 32
    threeinfo_byte = 3 * 4
    ub_byte = (token_byte + quant_byte + threeinfo_byte + 511) // 512 * 512
    ub_size = ub_byte // 2

    ret = aclshmem_module.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError("[ERROR] set_conf_store_tls failed")
    # Create initialization attribute object
    attributes = aclshmem_module.InitAttr()
    npu_num = num_processes
    attributes.my_rank = rank
    attributes.n_ranks = npu_num
    attributes.local_mem_size = g_ash_size
    attributes.ip_port = G_IP_PORT
    attributes.option_attr.data_op_engine_type = aclshmem_module.OpEngineType.MTE
    # Initialize aclshmem
    ret = aclshmem_module.aclshmem_init(attributes)
    if ret == 0:
        print(f"Rank {rank}: Initialization successful")
        torch.manual_seed(0)
        # Initialize shmem tensor
        tensorData_dispatch = aclshmem_module.aclshmem_create_tensor([ep_world_size * local_expert_num * Bs, ub_size], dtype = torch.bfloat16, device_id = rank)
        tensorStatus_dispatch = aclshmem_module.aclshmem_create_tensor([ep_world_size * local_expert_num, 8], dtype = torch.float, device_id = rank)
        tensor_combine = aclshmem_module.aclshmem_create_tensor([Bs * K, H], dtype=torch.bfloat16, device_id = rank)
        tensorStatus_combine = aclshmem_module.aclshmem_create_tensor([Bs * K, 8], dtype=torch.float, device_id=rank)
        win_dispatch = tensorData_dispatch.fill_(0)
        win_status_dispatch = tensorStatus_dispatch.fill_(0)
        win_combine = tensor_combine.fill_(0)
        win_status_combine = tensorStatus_combine.fill_(0)
        # dispatch
        func_dispatch = moe_dispatch_kernel(Bs, H, K, ep_world_size, local_expert_num, rank, ub_size, aiv_num)
        expand_x, expand_idx, ep_recv_counts, expert_token_nums = func_dispatch(x, expert_ids, win_dispatch, win_status_dispatch)
        barrier.wait()
        # combine
        expand_x = expand_x[:ep_recv_counts[-1].item(), :]
        expand_idx = expand_idx[:ep_recv_counts[-1].item(), :]
        func_combine = moe_combine_kernel(Bs, ep_recv_counts[-1].item(), H, K, ep_world_size, local_expert_num, rank, aiv_num)
        x_out = func_combine(expand_x, expand_idx, ep_recv_counts, win_combine, win_status_combine)
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
    print("Kernel Output Match!")

# Construct input
def init_input(rank, Bs, H, K, ep_world_size, local_expert_num):
    start = rank * Bs + 1
    end = start + Bs
    x = torch.tensor([[i] for i in range(start, end)], dtype=torch.bfloat16)
    x = x.repeat(1, H)
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    expert_ids_list = []
    for _ in range(Bs):
        full_range = list(range(0, ep_world_size * local_expert_num))
        random.shuffle(full_range)
        row = full_range[:K]
        expert_ids_list.append(row)
    expert_ids = torch.tensor(expert_ids_list, dtype=torch.int32)
    return x, expert_ids

if __name__ == '__main__':
    Bs = 64     
    H = 7168
    K = 4
    ep_world_size = 16
    local_expert_num = 3
    aiv_num = 48
    num_processes = 16
    barrier = Barrier(num_processes)
    processes = []
    for rank in range(num_processes):
        x, expert_ids = init_input(rank, Bs, H, K, ep_world_size, local_expert_num)
        p = mp.Process(target=worker, args=(rank, barrier, x, expert_ids, aiv_num, ep_world_size, local_expert_num, Bs))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("All processes completed")