import argparse
from typing import Literal
from collections import Counter

import tilelang
import torch
from tilelang import language as T

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tilelang.jit(pass_configs=pass_configs)
def get_topk_sum_and_topk_group_idx_kernel(B: int, N: int, top_k: int, block_N: int, topk_sum: int, dtype: Literal["float32"] = "float32"):
    INDEX_DTYPE = "int64"
    num_threads = 32
    total_N = N * block_N
    aligned_total_N = (total_N + num_threads - 1) // num_threads * num_threads
    aligned_N = (N + num_threads - 1) // num_threads * num_threads
    aligned_block_N = (block_N + num_threads - 1) // num_threads * num_threads
    dynamic_B = T.symbolic("B")
    assert N <= 32, f"num_groups ({N}) must be <= 32"

    @T.prim_func
    def topk_sum_and_topk_group_idx_kernel(
        scores: T.Tensor[(dynamic_B, total_N), dtype], group_topk_idx: T.Tensor[(dynamic_B, top_k), INDEX_DTYPE]
    ):
        with T.Kernel(dynamic_B, is_npu=True) as (cid, _):
            scores_ub = T.alloc_ub((aligned_total_N,), dtype)
            group_experts_ub = T.alloc_ub((aligned_block_N,), dtype)
            group_scores_ub = T.alloc_ub((aligned_N,), dtype)
            top1_ub = T.alloc_ub((1,), dtype)
            top2_ub = T.alloc_ub((1,), dtype)
            amax_ub = T.alloc_ub((1,), dtype)
            tmp_calc_ub = T.alloc_ub((aligned_N,), dtype)
            min_idx_res_ub = T.alloc_ub((1,), dtype)
            topk_group_idx_out_ub = T.alloc_ub((top_k,), INDEX_DTYPE)

            # 1. Load scores and pad
            T.copy(scores[cid, 0:total_N], scores_ub[0:total_N])
            for i in T.serial(aligned_total_N - total_N):
                scores_ub[total_N + i] = -1e10

            # 2. Compute group scores (topk_sum within each group)
            for g in T.serial(N):
                group_start = g * block_N
                for e in T.serial(block_N):
                    group_experts_ub[e] = scores_ub[group_start + e]
                for e in T.serial(aligned_block_N - block_N):
                    group_experts_ub[block_N + e] = -1e10

                T.reduce_max(group_experts_ub, top1_ub, dim=0)
                for e in T.serial(aligned_block_N):
                    if group_experts_ub[e] == top1_ub[0]:
                        group_experts_ub[e] = -1e10
                T.reduce_max(group_experts_ub, top2_ub, dim=0)

                if topk_sum == 1:
                    group_scores_ub[g] = top1_ub[0]
                else:
                    group_scores_ub[g] = top1_ub[0] + top2_ub[0]

            for g in T.serial(aligned_N - N):
                group_scores_ub[N + g] = -1e10

            # 3. Select top-k groups
            for k in T.serial(top_k):
                T.reduce_max(group_scores_ub, amax_ub, dim=0)
                for i in T.serial(aligned_N):
                    if group_scores_ub[i] == amax_ub[0]:
                        tmp_calc_ub[i] = T.cast(i, dtype)
                    else:
                        tmp_calc_ub[i] = T.cast(aligned_N, dtype)
                T.reduce_min(tmp_calc_ub, min_idx_res_ub, dim=0)
                found_idx_int32 = T.cast(min_idx_res_ub[0], "int32")
                topk_group_idx_out_ub[k] = T.cast(min_idx_res_ub[0], INDEX_DTYPE)
                group_scores_ub[found_idx_int32] = -1e10

            # 4. Write back results
            T.copy(topk_group_idx_out_ub[0:top_k], group_topk_idx[cid, 0:top_k])

    return topk_sum_and_topk_group_idx_kernel


def topk_sum_and_topk_group_idx(scores: torch.Tensor, num_topk_sum: int, num_topk_groups: int) -> torch.Tensor:
    assert scores.dim() == 3 and scores.is_contiguous() and scores.dtype == torch.float32
    num_tokens, num_groups, num_experts_per_group = scores.shape
    assert num_topk_sum <= num_experts_per_group and num_topk_sum in (1, 2) and num_topk_groups <= num_groups

    kernel = get_topk_sum_and_topk_group_idx_kernel(
        B=num_tokens, N=num_groups, top_k=num_topk_groups, block_N=num_experts_per_group, topk_sum=num_topk_sum, dtype="float32"
    )
    topk_group_idx = torch.empty(num_tokens, num_topk_groups, dtype=torch.int64, device=scores.device)
    if num_tokens == 0:
        return topk_group_idx
    kernel(scores.view(num_tokens, -1), topk_group_idx)
    return topk_group_idx


@tilelang.jit(pass_configs=pass_configs)
def get_topk_sum_and_topk_group_idx_backward_kernel(
    B: int, N: int, top_k: int, block_N: int, topk_sum: int, dtype: Literal["float32"] = "float32"
):
    INDEX_DTYPE = "int64"
    num_threads = 32
    total_N = N * block_N
    aligned_total_N = (total_N + num_threads - 1) // num_threads * num_threads
    aligned_block_N = (block_N + num_threads - 1) // num_threads * num_threads
    aligned_top_k = (top_k + 31) // 32 * 32
    dynamic_B = T.symbolic("B")

    @T.prim_func
    def topk_sum_and_topk_group_idx_backward_kernel(
        grad_out: T.Tensor[(dynamic_B, aligned_top_k), dtype],
        scores: T.Tensor[(dynamic_B, aligned_total_N), dtype],
        group_topk_idx: T.Tensor[(dynamic_B, aligned_top_k), INDEX_DTYPE],
        grad_scores: T.Tensor[(dynamic_B, aligned_total_N), dtype],
    ):
        with T.Kernel(dynamic_B, is_npu=True) as (cid, _):
            grad_out_ub = T.alloc_ub((aligned_top_k,), "float32")
            scores_ub = T.alloc_ub((aligned_total_N,), "float32")
            group_topk_idx_ub = T.alloc_ub((aligned_top_k,), INDEX_DTYPE)
            grad_scores_ub = T.alloc_ub((aligned_total_N,), "float32")
            group_experts_ub = T.alloc_ub((aligned_block_N,), "float32")
            tmp_calc_ub = T.alloc_ub((aligned_block_N,), "float32")
            top_val_ub = T.alloc_ub((1,), "float32")
            min_idx_res_ub = T.alloc_ub((1,), "float32")

            for i in T.serial(aligned_total_N):
                grad_scores_ub[i] = 0.0

            T.copy(grad_out[cid, 0:aligned_top_k], grad_out_ub[0:aligned_top_k])
            T.copy(scores[cid, 0:aligned_total_N], scores_ub[0:aligned_total_N])
            T.copy(group_topk_idx[cid, 0:aligned_top_k], group_topk_idx_ub[0:aligned_top_k])

            for k in T.serial(top_k):
                g_int32 = T.cast(group_topk_idx_ub[k], "int32")
                if g_int32 >= 0 and g_int32 < N:
                    group_start = g_int32 * block_N
                    current_grad = grad_out_ub[k]

                    for e in T.serial(block_N):
                        group_experts_ub[e] = scores_ub[group_start + e]
                    for e in T.serial(aligned_block_N - block_N):
                        group_experts_ub[block_N + e] = -1e10

                    for _step in T.serial(topk_sum):
                        T.reduce_max(group_experts_ub, top_val_ub, dim=0)
                        for i in T.serial(aligned_block_N):
                            if group_experts_ub[i] == top_val_ub[0]:
                                tmp_calc_ub[i] = T.cast(i, "float32")
                            else:
                                tmp_calc_ub[i] = T.cast(aligned_block_N, "float32")
                        T.reduce_min(tmp_calc_ub, min_idx_res_ub, dim=0)
                        found_e_idx = T.cast(min_idx_res_ub[0], "int32")
                        if found_e_idx < block_N:
                            grad_scores_ub[group_start + found_e_idx] = grad_scores_ub[group_start + found_e_idx] + current_grad
                        group_experts_ub[found_e_idx] = -1e10

            T.copy(grad_scores_ub[0:aligned_total_N], grad_scores[cid, 0:aligned_total_N])

    return topk_sum_and_topk_group_idx_backward_kernel


def topk_sum_and_topk_group_idx_backward(
    grad_out: torch.Tensor, scores: torch.Tensor, group_topk_idx: torch.Tensor, num_topk_sum: int
) -> torch.Tensor:
    assert grad_out.is_contiguous() and grad_out.dtype == torch.float32
    assert scores.is_contiguous() and scores.dtype == torch.float32
    assert group_topk_idx.is_contiguous() and group_topk_idx.dtype == torch.int64

    num_tokens, num_groups, num_experts_per_group = scores.shape
    num_topk_groups = group_topk_idx.shape[1]
    num_experts = num_groups * num_experts_per_group
    num_aligned_experts = (num_experts + 31) // 32 * 32
    num_aligned_topk_groups = (num_topk_groups + 31) // 32 * 32

    if num_experts < num_aligned_experts:
        scores_padded = torch.nn.functional.pad(
            scores.view(num_tokens, -1), (0, num_aligned_experts - num_experts), value=-1e10
        ).contiguous()
    else:
        scores_padded = scores.view(num_tokens, -1).contiguous()

    if num_topk_groups < num_aligned_topk_groups:
        grad_out_padded = torch.nn.functional.pad(grad_out, (0, num_aligned_topk_groups - num_topk_groups), value=0.0).contiguous()
        group_topk_idx_padded = torch.nn.functional.pad(
            group_topk_idx, (0, num_aligned_topk_groups - num_topk_groups), value=-1
        ).contiguous()
    else:
        grad_out_padded = grad_out.contiguous()
        group_topk_idx_padded = group_topk_idx.contiguous()

    kernel = get_topk_sum_and_topk_group_idx_backward_kernel(
        B=num_tokens, N=num_groups, top_k=num_topk_groups, block_N=num_experts_per_group, topk_sum=num_topk_sum, dtype="float32"
    )
    grad_scores_padded = torch.zeros(num_tokens, num_aligned_experts, dtype=torch.float32, device=scores.device)
    if num_tokens > 0:
        kernel(grad_out_padded, scores_padded, group_topk_idx_padded, grad_scores_padded)
    return grad_scores_padded[:, :num_experts].view(num_tokens, num_groups, num_experts_per_group).contiguous()


def stable_topk(x, top_k):
    _, sorted_indices = torch.sort(x, dim=-1, descending=True, stable=True)
    return sorted_indices[..., :top_k].contiguous()


def ref_topk_sum_and_topk_group_idx(scores: torch.Tensor, num_group_sum_topk: int, num_topk_groups: int) -> torch.Tensor:
    group_scores_ref = scores.topk(num_group_sum_topk, dim=-1, sorted=False).values.sum(-1)
    return stable_topk(group_scores_ref, num_topk_groups)


def count_per_row_mismatches(indices: torch.Tensor, ref_indices: torch.Tensor):
    row_mismatches = 0
    for i in range(indices.shape[0]):
        ref_indices_np = ref_indices[i].to(torch.int32).numpy()
        indices_np = indices[i].to(torch.int32).numpy()
        ref_indices_set = set(ref_indices_np)
        indices_set = set(indices_np)
        intersection = ref_indices_set & indices_set
        if len(intersection) != len(ref_indices_set):
            row_mismatches += 1
        indices_ratio = len(intersection) / len(indices_set)
        ref_indices_ratio = len(intersection) / len(ref_indices_set)
        assert indices_ratio == ref_indices_ratio and indices_ratio > 0.99, (
            f"Row {i} check failed: {indices_ratio = }, {ref_indices_ratio = }; {row_mismatches = }"
        )
    return row_mismatches


def count_total_mismatches(indices: torch.Tensor, ref_indices: torch.Tensor):
    assert indices.shape[-1] == ref_indices.shape[-1], "last dimension must match"
    total_mismatches = 0
    for i in range(indices.shape[0]):
        indices_row = indices[i].tolist()
        ref_indices_row = ref_indices[i].tolist()
        indices_counter = Counter(indices_row)
        ref_indices_counter = Counter(ref_indices_row)
        diff = (indices_counter - ref_indices_counter) + (ref_indices_counter - indices_counter)
        total_mismatches += sum(diff.values())
    return total_mismatches


def check_case(B: int, N: int, block_N: int, top_k: int, topk_sum: int):
    print(f"\n Testing Case: B={B}, N={N}, block_N={block_N}, top_k={top_k}, topk_sum={topk_sum} ")

    scores = torch.randn(B, N, block_N).to(torch.float32).npu()
    indices = topk_sum_and_topk_group_idx(scores, topk_sum, top_k)
    ref_indices = ref_topk_sum_and_topk_group_idx(scores, topk_sum, top_k)

    grad_out = torch.randn(B, top_k).to(torch.float32).npu()
    grad_scores = topk_sum_and_topk_group_idx_backward(grad_out, scores, indices, topk_sum)

    scores_ref = scores.clone().detach().requires_grad_(True)
    group_scores = scores_ref.topk(topk_sum, dim=-1, sorted=False).values.sum(-1)
    chosen_group_scores = group_scores.gather(1, indices)
    chosen_group_scores.backward(grad_out)
    ref_grad_scores = scores_ref.grad

    indices_cpu = indices.cpu()
    ref_indices_cpu = ref_indices.cpu()

    row_mismatches = count_per_row_mismatches(indices_cpu, ref_indices_cpu)
    print(f"[Forward] Row matches: {B - row_mismatches} / {B}")

    total = B * top_k
    total_mismatches = count_total_mismatches(indices_cpu, ref_indices_cpu)
    accuracy = 1 - total_mismatches / total
    print(f"[Forward] Total matches: {total - total_mismatches} / {total}")
    print(f"[Forward] Accuracy: {accuracy}")
    assert accuracy > 0.99, "Forward accuracy check failed against torch.topk!"

    torch.testing.assert_close(grad_scores, ref_grad_scores, rtol=1e-5, atol=1e-5)
    print("[Backward] Scatter Check Passed! (PyTorch Reference Match!)")


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="topk_sum_and_topk_group_idx with Backward Test")
    parser.add_argument("--b", type=int, default=64, help="Batch Size / num_tokens")
    parser.add_argument("--n", type=int, default=16, help="Number of groups (N)")
    parser.add_argument("--block_n", type=int, default=32, help="Experts per group")
    parser.add_argument("--top_k", type=int, default=4, help="Number of selected groups")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}]", "Unknown args:", remains)

    B, N, block_N, top_k = args.b, args.n, args.block_n, args.top_k

    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    check_case(B, N, block_N, top_k, topk_sum=2)
    check_case(32, 8, 16, 2, topk_sum=1)
    check_case(123, 14, 23, 5, topk_sum=2)

    print("\ntopk_sum_and_topk_group_idx example passed!")
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
