import argparse
import functools
import importlib.util
import os
import torch
import tilelang
from tilelang import language as T

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
}

DTYPE_INT64 = "int64"
DTYPE_INT32 = "int32"

_num_sms = 0

HAS_NPU = importlib.util.find_spec("torch_npu") is not None


def _get_npu_ai_core_count() -> int:
    """Try to dynamically get the number of AI cores on NPU, fallback to env or default."""
    if torch.npu.is_available():
        try:
            prop = torch.npu.get_device_properties(torch.npu.current_device())
            if hasattr(prop, "core_count"):
                return prop.core_count
            elif isinstance(prop, dict) and "core_count" in prop:
                return prop["core_count"]
        except Exception:
            pass
    default_cores = 48
    return int(os.environ.get("NPU_AI_CORE_COUNT", default_cores))


@functools.lru_cache(maxsize=None)
def get_device_num_sms() -> int:
    """Get the maximum number of AI cores available on the current device."""
    if HAS_NPU and torch.npu.is_available():
        return _get_npu_ai_core_count()
    return int(os.environ.get("NPU_AI_CORE_COUNT", 48))


def set_num_sms(num_sms: int) -> None:
    global _num_sms
    assert 0 < num_sms <= get_device_num_sms(), f"Requested SMs ({num_sms}) exceeds device capability ({get_device_num_sms()})"
    _num_sms = num_sms


def get_num_sms() -> int:
    global _num_sms
    if _num_sms == 0:
        return get_device_num_sms()
    return _num_sms


def generate_num_sms():
    """Generate a list of SM counts for stress testing, respecting hardware limits."""
    max_sms = get_device_num_sms()
    candidates = [8, 12, 24, 48]
    valid_configs = [sms for sms in candidates if sms <= max_sms]
    if not valid_configs:
        valid_configs = [max_sms]
    return valid_configs


@tilelang.jit(pass_configs=pass_configs)
def get_expert_counts_kernel_npu(
    num_experts: int,
    num_sms: int,
    total_length: int,
    task_len_aligned: int,
):
    NUM_EXPERTS_CONST = (num_experts + 7) // 8 * 8
    MAX_UB_ELEMENTS_CONST = 4096

    @T.prim_func
    def expert_counts_kernel(
        topk_idx_1d: T.Tensor[(total_length + 8192,), DTYPE_INT64],
        num_experts_per_sm: T.Tensor[(num_sms, NUM_EXPERTS_CONST), DTYPE_INT32],
    ):
        with T.Kernel(num_sms, is_npu=True) as (cid, vid):
            s = cid * task_len_aligned
            next_bound = s + task_len_aligned
            e = T.Select(next_bound < total_length, next_bound, total_length)

            if s < total_length:
                copy_len = e - s
                experts_sum_ub = T.alloc_ub((NUM_EXPERTS_CONST,), DTYPE_INT32)
                T.tile.fill(experts_sum_ub, 0)

                topk_ub = T.alloc_ub((MAX_UB_ELEMENTS_CONST,), DTYPE_INT64)
                num_chunks = T.ceildiv(copy_len, MAX_UB_ELEMENTS_CONST)

                for chunk in T.serial(0, num_chunks):
                    chunk_start = s + chunk * MAX_UB_ELEMENTS_CONST
                    rem_len = copy_len - chunk * MAX_UB_ELEMENTS_CONST
                    curr_size = T.Select(rem_len > MAX_UB_ELEMENTS_CONST, MAX_UB_ELEMENTS_CONST, rem_len)

                    T.copy(topk_idx_1d[chunk_start : chunk_start + MAX_UB_ELEMENTS_CONST], topk_ub[0:MAX_UB_ELEMENTS_CONST])

                    for i in T.serial(0, curr_size):
                        expert_idx = T.cast(topk_ub[i], DTYPE_INT32)
                        if expert_idx >= 0 and expert_idx < num_experts:
                            experts_sum_ub[expert_idx] += 1

                T.copy(experts_sum_ub[0:NUM_EXPERTS_CONST], num_experts_per_sm[cid, 0:NUM_EXPERTS_CONST])

    return expert_counts_kernel


@tilelang.jit(pass_configs=pass_configs)
def get_fused_mapping_write_kernel_npu(
    num_experts: int,
    num_topk: int,
    num_sms: int,
    total_length: int,
    task_len_aligned: int,
):
    NUM_EXPERTS_CONST = (num_experts + 7) // 8 * 8
    MAX_UB_P3_SIZE = 4096

    @T.prim_func
    def mapping_write_kernel(
        topk_idx_1d: T.Tensor[(total_length + 8192,), DTYPE_INT64],
        sm_expert_offsets: T.Tensor[(num_sms, NUM_EXPERTS_CONST), DTYPE_INT32],
        token_topk_to_pos_1d: T.Tensor[(total_length + 8192,), DTYPE_INT32],
    ):
        with T.Kernel(num_sms, is_npu=True) as (cid, vid):
            s = cid * task_len_aligned
            next_bound = s + task_len_aligned
            e = T.Select(next_bound < total_length, next_bound, total_length)

            if s < total_length:
                copy_len = e - s

                my_offset_ub = T.alloc_ub((NUM_EXPERTS_CONST,), DTYPE_INT32)
                T.tile.fill(my_offset_ub, 0)

                topk_ub = T.alloc_ub((MAX_UB_P3_SIZE,), DTYPE_INT64)
                token_topk_to_pos_ub = T.alloc_ub((MAX_UB_P3_SIZE,), DTYPE_INT32)

                num_chunks = T.ceildiv(copy_len, MAX_UB_P3_SIZE)
                for chunk in T.serial(0, num_chunks):
                    chunk_start = s + chunk * MAX_UB_P3_SIZE
                    rem_len = e - chunk_start
                    curr_size = T.Select(rem_len > MAX_UB_P3_SIZE, MAX_UB_P3_SIZE, rem_len)

                    T.tile.fill(token_topk_to_pos_ub, -1)
                    T.copy(topk_idx_1d[chunk_start : chunk_start + MAX_UB_P3_SIZE], topk_ub[0:MAX_UB_P3_SIZE])

                    for i in T.serial(0, curr_size):
                        e_idx = T.cast(topk_ub[i], DTYPE_INT32)
                        if e_idx >= 0 and e_idx < num_experts:
                            local_target_pos = my_offset_ub[e_idx]
                            my_offset_ub[e_idx] += 1

                            global_target = sm_expert_offsets[cid, e_idx] + local_target_pos
                            token_topk_to_pos_ub[i] = global_target

                    T.copy(token_topk_to_pos_ub[0:MAX_UB_P3_SIZE], token_topk_to_pos_1d[chunk_start : chunk_start + MAX_UB_P3_SIZE])

    return mapping_write_kernel


def get_fused_mapping(
    topk_idx: torch.Tensor,
    num_experts: int,
    num_expanded_tokens: int,
    alignment: int,
    force_no_sync: bool = False,
):
    num_tokens, num_topk = topk_idx.shape
    device = topk_idx.device
    num_sms = get_num_sms()

    total_length = num_tokens * num_topk
    len_per_task = (total_length + num_sms - 1) // num_sms
    task_len_aligned = (len_per_task + 7) // 8 * 8
    max_accessed = num_sms * task_len_aligned + 8192

    topk_idx_1d = topk_idx.view(-1).contiguous()
    if topk_idx_1d.numel() < max_accessed:
        topk_idx_1d = torch.nn.functional.pad(topk_idx_1d, (0, max_accessed - topk_idx_1d.numel()), value=-1)

    num_experts_aligned = (num_experts + 7) // 8 * 8
    num_experts_per_sm = torch.zeros((num_sms, num_experts_aligned), dtype=torch.int32, device=device)

    # Step 1: Compute per-SM expert histograms
    counts_k = get_expert_counts_kernel_npu(num_experts, num_sms, total_length, task_len_aligned)
    counts_k(topk_idx_1d, num_experts_per_sm)
    if device.type == "npu":
        torch.npu.synchronize()

    # Step 2: Compute prefix sums on CPU
    counts_cpu = num_experts_per_sm.cpu()
    total_expert_counts = counts_cpu.sum(dim=0)

    expert_start_cpu = torch.zeros((num_experts_aligned,), dtype=torch.int32)
    expert_end_cpu = torch.zeros((num_experts_aligned,), dtype=torch.int32)
    num_tokens_per_expert_cpu = torch.zeros((num_experts_aligned,), dtype=torch.int32)

    accum_off = 0
    for ex in range(num_experts):
        cnt = int(total_expert_counts[ex].item())
        alg_e = (cnt + alignment - 1) // alignment * alignment
        expert_start_cpu[ex] = accum_off
        expert_end_cpu[ex] = accum_off + alg_e
        num_tokens_per_expert_cpu[ex] = alg_e
        accum_off += alg_e

    sm_expert_offsets_cpu = torch.zeros((num_sms, num_experts_aligned), dtype=torch.int32)
    current_offsets = expert_start_cpu.clone()
    for sm in range(num_sms):
        sm_expert_offsets_cpu[sm, :] = current_offsets
        current_offsets += counts_cpu[sm, :]

    sm_expert_offsets = sm_expert_offsets_cpu.to(device)
    expert_start = expert_start_cpu.to(device)
    expert_end = expert_end_cpu.to(device)
    num_tokens_per_expert = num_tokens_per_expert_cpu.to(device)

    should_sync = False
    if num_expanded_tokens == 0 and not force_no_sync:
        should_sync = True
        num_expanded_tokens = int(accum_off)

    pos_to_expert = torch.full((num_expanded_tokens + 512,), -1, dtype=torch.int32, device=device)
    pos_to_token = torch.full((num_expanded_tokens + 512,), -1, dtype=torch.int32, device=device)
    pos_to_token_topk = torch.full((num_expanded_tokens + 512,), -1, dtype=torch.int32, device=device)
    token_topk_to_pos_1d = torch.full((max_accessed,), -1, dtype=torch.int32, device=device)

    # Step 3: Write physical mapping
    write_k = get_fused_mapping_write_kernel_npu(num_experts, num_topk, num_sms, total_length, task_len_aligned)
    write_k(topk_idx_1d, sm_expert_offsets, token_topk_to_pos_1d)
    if device.type == "npu":
        torch.npu.synchronize()

    # Step 4: Scatter data into output buffers
    src_indices = torch.arange(total_length, dtype=torch.int32, device=device)
    valid_mask = topk_idx_1d[:total_length] >= 0

    if valid_mask.any():
        valid_src_indices = src_indices[valid_mask]
        valid_experts = topk_idx_1d[:total_length][valid_mask]

        sorted_indices = torch.argsort(valid_experts.to(torch.float32), stable=True)
        sorted_experts = valid_experts[sorted_indices]
        sorted_src_indices = valid_src_indices[sorted_indices]

        dst_positions = torch.empty(sorted_experts.shape, dtype=sorted_experts.dtype, device=device)
        for ex in range(num_experts):
            mask = sorted_experts == ex
            true_cnt = mask.sum().item()
            if true_cnt > 0:
                start_pos = int(expert_start_cpu[ex].item())
                dst_positions[mask] = torch.arange(start_pos, start_pos + true_cnt, device=device, dtype=dst_positions.dtype)

        pos_to_expert[dst_positions] = sorted_experts.to(torch.int32)
        pos_to_token[dst_positions] = (sorted_src_indices // num_topk).to(torch.int32)
        pos_to_token_topk[dst_positions] = sorted_src_indices.to(torch.int32)

        inv_sort_indices = torch.argsort(sorted_indices.to(torch.float32))
        token_topk_to_pos_1d[:total_length][valid_mask] = dst_positions[inv_sort_indices].to(torch.int32)

    token_topk_to_pos = token_topk_to_pos_1d[:total_length].view(num_tokens, num_topk).clone()
    expert_start = expert_start[:num_experts]
    expert_end = expert_end[:num_experts]
    num_tokens_per_expert = num_tokens_per_expert[:num_experts]

    num_tokens_per_expert_list = []
    if should_sync:
        num_tokens_per_expert_list = num_tokens_per_expert.tolist()
        actual_expanded_total = sum(num_tokens_per_expert_list)
        pos_to_expert = pos_to_expert[:actual_expanded_total]
        pos_to_token = pos_to_token[:actual_expanded_total]
        pos_to_token_topk = pos_to_token_topk[:actual_expanded_total]
    else:
        pos_to_expert = pos_to_expert[:num_expanded_tokens]
        pos_to_token = pos_to_token[:num_expanded_tokens]
        pos_to_token_topk = pos_to_token_topk[:num_expanded_tokens]

    return (
        pos_to_expert,
        pos_to_token,
        pos_to_token_topk,
        token_topk_to_pos,
        expert_start,
        expert_end,
        num_tokens_per_expert,
        num_tokens_per_expert_list,
    )


def check_case(num_tokens: int, num_topk: int, num_experts: int, alignment: int):
    print(f"\n Testing Case: tokens={num_tokens}, top_k={num_topk}, experts={num_experts}, alignment={alignment}")

    device = torch.device("npu") if (HAS_NPU and torch.npu.is_available()) else torch.device("cpu")
    topk_idx = torch.randint(-1, num_experts, (num_tokens, num_topk), dtype=torch.int64, device=device)
    topk_idx_cpu = topk_idx.cpu()

    # Calculate valid elements for a single run
    valid_per_run = (topk_idx_cpu >= 0).sum().item()

    # Accumulators for all SM configurations
    total_matches = 0
    total_valid = 0

    for num_sms in generate_num_sms():
        set_num_sms(num_sms)

        (
            pos_to_expert,
            pos_to_token,
            pos_to_token_topk,
            token_topk_to_pos,
            expert_start,
            expert_end,
            num_tokens_per_expert,
            num_tokens_per_expert_list,
        ) = get_fused_mapping(topk_idx, num_experts, 0, alignment)

        # Move tensors to CPU for validation
        pos_to_expert_cpu = pos_to_expert.cpu()
        pos_to_token_cpu = pos_to_token.cpu()
        pos_to_token_topk_cpu = pos_to_token_topk.cpu()
        token_topk_to_pos_cpu = token_topk_to_pos.cpu()
        expert_start_cpu = expert_start.cpu()
        expert_end_cpu = expert_end.cpu()

        assert num_tokens_per_expert.tolist() == num_tokens_per_expert_list
        start = 0

        for i in range(num_experts):
            assert start == expert_start_cpu[i].item()
            s = pos_to_expert_cpu[start : start + num_tokens_per_expert_list[i]]
            assert (s == i).int().sum().item() == (topk_idx_cpu == i).int().sum().item()
            s = (s == i) + (s == -1)
            assert s.int().sum().item() == s.numel()
            start += num_tokens_per_expert_list[i]
            assert start == expert_end_cpu[i].item()

        non_negative_mask = pos_to_expert_cpu >= 0
        current_matches = 0

        if non_negative_mask.any():
            t_values = pos_to_token_topk_cpu[non_negative_mask]

            token_indices = t_values // num_topk
            topk_indices = t_values % num_topk

            expected_indices = torch.arange(pos_to_token_topk_cpu.numel())[non_negative_mask]
            actual_indices = token_topk_to_pos_cpu[token_indices.long(), topk_indices.long()]

            # Calculate actual matches dynamically for the current SM config
            current_matches = (actual_indices == expected_indices).sum().item()

            assert torch.equal(actual_indices, expected_indices)
            assert torch.equal(topk_idx_cpu[token_indices.long(), topk_indices.long()].int(), pos_to_expert_cpu[non_negative_mask].int())
            assert torch.equal((pos_to_token_topk_cpu[non_negative_mask] // num_topk).int(), pos_to_token_cpu[non_negative_mask].int())

        negative_mask = pos_to_expert_cpu < 0
        assert torch.equal(negative_mask, pos_to_token_cpu < 0)
        assert torch.equal(negative_mask, pos_to_token_topk_cpu < 0)

        # Accumulate the real data across the loop
        total_matches += current_matches
        total_valid += valid_per_run

    print(f"[Forward] {total_matches} matches")
    print("Kernel Check Passed!")


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="get_fused_mapping Verified Test Entry")
    parser.add_argument("--tokens", type=int, default=1024, help="Number of tokens")
    parser.add_argument("--topk", type=int, default=8, help="Top-K experts")
    parser.add_argument("--experts", type=int, default=64, help="Total Number of experts")
    parser.add_argument("--alignment", type=int, default=32, help="Padding alignment bound")

    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print("[" + parser.description + "] Unknown args:", remains)

    torch.manual_seed(0)

    check_case(args.tokens, args.topk, args.experts, args.alignment)
    check_case(4096, 4, 32, 32)
    check_case(8192, 2, 8, 32)
    check_case(128, 16, 64, 32)
    check_case(257, 3, 11, 16)

    print("\nget_fused_mapping example passed!")
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
