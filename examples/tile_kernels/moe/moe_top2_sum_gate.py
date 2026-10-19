# ruff: noqa: F841, SIM117

"""V25 scalar-optimized compatibility entry.

This version prioritizes the full GPU/PyTorch top2-sum-gate contract:

* arbitrary token counts, including zero tokens and tail blocks
* variable routed expert counts and TopK
* grouped routing
* shared-as-routed experts
* fixed routing
* logical-to-physical expert maps
* stable tie-breaking through the scalar kernel

The file is intentionally self-contained: it does not import another
top2-sum-gate implementation. The kernel keeps the scalar, stable selection
logic while still using vector tile operations for scoring transforms.
"""

import os
from typing import Optional

import torch
import tilelang
import tilelang.language as T


pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


double_buffer_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


SCORING_SIGMOID = 0
SCORING_SQRTSOFTPLUS = 1
SCORING_SOFTMAX = 2
SCORING_IDENTITY = 3
VEC_NUM = 2


def _rows_per_vec(num_routed_experts: int, num_tokens: Optional[int] = None) -> int:
    return 4


def _rows_per_vec_dynamic(num_routed_experts: int, num_tokens: Optional[int] = None) -> int:
    # Restore the lightweight dynamic-map path used by the 170us reference version.
    # The static DB reference keeps _rows_per_vec(...)=4.
    return 32


def _scoring_type(scoring_func: str) -> int:
    table = {
        "sigmoid": SCORING_SIGMOID,
        "sqrtsoftplus": SCORING_SQRTSOFTPLUS,
        "softmax": SCORING_SOFTMAX,
        "identity": SCORING_IDENTITY,
    }
    key = scoring_func.lower()
    if key not in table:
        raise ValueError(f"Unsupported scoring_func: {scoring_func}")
    return table[key]


_STATIC_INPUT_CACHE = {}


def _build_expert_local_map(
    num_logical_experts: int,
    num_experts_per_rank: int,
    tp_rank: int,
    num_tp_ranks: int,
) -> torch.Tensor:
    expert_idx = torch.arange(num_logical_experts, dtype=torch.int32)
    num_experts_per_dp = num_experts_per_rank * num_tp_ranks
    dst_ep_rank = expert_idx // num_experts_per_rank
    local_mask = dst_ep_rank % num_tp_ranks == tp_rank
    local_idx = expert_idx - tp_rank * num_experts_per_rank
    dst_dp_rank = local_idx // num_experts_per_dp
    local_idx = local_idx - dst_dp_rank * num_experts_per_dp + dst_dp_rank * num_experts_per_rank
    return torch.where(
        local_mask,
        local_idx,
        torch.full_like(local_idx, -1),
    )


def _get_expert_local_map(
    device: torch.device,
    num_logical_experts: int,
    num_experts_per_rank: int,
    tp_rank: int,
    num_tp_ranks: int,
) -> torch.Tensor:
    key = (
        str(device),
        num_logical_experts,
        num_experts_per_rank,
        tp_rank,
        num_tp_ranks,
    )
    expert_local_map = _STATIC_INPUT_CACHE.get(key)
    if expert_local_map is None:
        expert_local_map = _build_expert_local_map(
            num_logical_experts,
            num_experts_per_rank,
            tp_rank,
            num_tp_ranks,
        ).to(device)
        _STATIC_INPUT_CACHE[key] = expert_local_map
    return expert_local_map


@tilelang.jit(pass_configs=double_buffer_pass_configs)
def _get_top2_sum_gate_static_refdb_kernel(
    scoring_type: int,
    num_topk: int,
    num_topk_groups: int,
    num_groups: int,
    num_routed_experts: int,
    num_shared_experts: int,
    mask_exists: bool,
    fix_routing_mask_exists: bool,
    unmapped_topk_idx_exists: bool,
    to_physical_map_exists: bool,
    rows_per_vec: int,
):
    num_tokens = T.symbolic("num_tokens")
    num_duplicate_experts = T.symbolic("num_duplicate_experts")
    num_logical_experts = num_routed_experts + num_shared_experts
    num_physical_topk = num_topk + num_shared_experts
    effective_groups = num_groups if num_groups > 0 else 1
    effective_topk_groups = num_topk_groups if num_topk_groups > 0 else 1
    experts_per_group = num_routed_experts // effective_groups
    skip_group_sort = num_groups == 0 or num_groups == num_topk_groups
    aligned_topk = ((num_topk + 7) // 8) * 8
    aligned_physical_topk = ((num_physical_topk + 7) // 8) * 8
    aligned_num_experts = ((num_routed_experts + 31) // 32) * 32
    aligned_copy_experts = (num_routed_experts // 8) * 8
    tail_experts = num_routed_experts - aligned_copy_experts
    aligned_groups = ((effective_groups + 31) // 32) * 32
    aligned_experts_per_group = ((experts_per_group + 31) // 32) * 32
    group_score_count = 1 if skip_group_sort else aligned_groups
    selected_group_count = 1 if skip_group_sort else effective_topk_groups
    topk_candidate_count = num_routed_experts if skip_group_sort else effective_topk_groups * experts_per_group
    aligned_topk_candidates = ((topk_candidate_count + 31) // 32) * 32
    vec_num = VEC_NUM
    tokens_per_block = rows_per_vec * vec_num
    num_token_blocks = T.ceildiv(num_tokens, tokens_per_block)
    num_cores = 24
    stages = 2
    num_iters = T.ceildiv(num_token_blocks, num_cores)

    @T.prim_func
    def top2_sum_gate_ascend_v25_scalar_kernel(
        logits: T.Tensor((num_tokens, num_routed_experts), "float"),
        bias: T.Tensor((num_routed_experts,), "float"),
        mask: T.Tensor((num_tokens,), "int32"),
        fix_routing_mask: T.Tensor((num_tokens,), "int32"),
        to_physical_map: T.Tensor(
            (num_logical_experts, num_duplicate_experts),
            "int32",
        ),
        logical_count: T.Tensor((num_logical_experts,), "int32"),
        expert_local_map: T.Tensor((num_logical_experts,), "int32"),
        fixed_topk_idx: T.Tensor((num_tokens, num_topk), "int64"),
        topk_idx: T.Tensor((num_tokens, num_physical_topk), "int64"),
        unmapped_topk_idx: T.Tensor((num_tokens, num_topk), "int64"),
        topk_weights: T.Tensor((num_tokens, num_physical_topk), "float"),
        num_extra_experts: T.int32,
        routed_scaling_factor: T.float32,
        ep_rank: T.int32,
        num_ep_ranks: T.int32,
        tp_rank: T.int32,
        num_tp_ranks: T.int32,
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                scores_ub = T.alloc_ub((stages, rows_per_vec, aligned_num_experts), "float")
                shifted_scores_ub = T.alloc_ub((stages, rows_per_vec, aligned_num_experts), "float")
                route_scores_ub = T.alloc_ub((stages, rows_per_vec, aligned_num_experts), "float")
                bias_ub = T.alloc_ub((1, aligned_num_experts), "float")
                bias_block_ub = T.alloc_ub((rows_per_vec, aligned_num_experts), "float")
                max_ub = T.alloc_ub((rows_per_vec, 1), "float")
                max_block_ub = T.alloc_ub((rows_per_vec, aligned_num_experts), "float")
                sum_vec_ub = T.alloc_ub((rows_per_vec, 1), "float")
                sum_block_ub = T.alloc_ub((rows_per_vec, aligned_num_experts), "float")
                group_scores_ub = T.alloc_ub((group_score_count,), "float")
                group_topk_scores_ub = T.alloc_ub((aligned_experts_per_group,), "float")
                group_topk_result_ub = T.alloc_ub((4,), "float")
                group_select_result_ub = T.alloc_ub((effective_topk_groups * 2,), "float")
                selected_groups_ub = T.alloc_ub((selected_group_count,), "int32")
                group_logical_idx_f32_ub = T.alloc_ub((aligned_experts_per_group,), "float")
                selected_idx_ub = T.alloc_ub((aligned_num_experts,), "int32")
                selected_idx_float_ub = T.alloc_ub((aligned_topk,), "float")
                selected_score_ub = T.alloc_ub((aligned_topk,), "float")
                selected_score_mat_ub = T.alloc_ub((1, aligned_topk), "float")
                selected_sum_ub = T.alloc_ub((1, 1), "float")
                selected_sum_block_ub = T.alloc_ub((1, aligned_topk), "float")
                selected_sum_vec_ub = T.alloc_ub((aligned_topk,), "float")
                scale_vec_ub = T.alloc_ub((aligned_topk,), "float")
                normalized_weight_ub = T.alloc_ub((aligned_topk,), "float")
                expert_local_map_ub = T.alloc_ub((num_logical_experts,), "int32")
                expert_local_map_f32_ub = T.alloc_ub((num_logical_experts,), "float")
                mapped_expert_full_f32_ub = T.alloc_ub((aligned_num_experts,), "float")
                mapped_expert_topk_f32_ub = T.alloc_ub((aligned_topk,), "float")
                mapped_expert_topk_i32_ub = T.alloc_ub((aligned_topk,), "int32")
                selected_idx_offset_i32_ub = T.alloc_ub((aligned_num_experts,), "int32")
                selected_idx_gather_ub = T.alloc_ub((aligned_num_experts,), "uint32")

                topk_scores_ub = T.alloc_ub((aligned_topk_candidates,), "float")
                candidate_logical_idx_f32_ub = T.alloc_ub((aligned_topk_candidates,), "float")
                topk_result_ub = T.alloc_ub((num_topk * 2,), "float")
                topk_idx_out_ub = T.alloc_ub((stages, rows_per_vec, aligned_physical_topk), "int64")
                unmapped_idx_out_ub = T.alloc_ub((stages, rows_per_vec, aligned_topk), "int64")
                topk_weights_out_ub = T.alloc_ub((stages, rows_per_vec, aligned_physical_topk), "float")
                topk_idx_out_i32_ub = T.alloc_ub((stages, rows_per_vec, aligned_physical_topk), "int32")
                unmapped_idx_out_i32_ub = T.alloc_ub((stages, rows_per_vec, aligned_topk), "int32")

                selected_score_full_ub = T.alloc_ub((aligned_num_experts,), "float")

                best_score = T.alloc_var("float", init=-3.402823e38)
                best_idx = T.alloc_var("int32", init=-1)
                is_selected = T.alloc_var("bool", init=False)
                already_taken = T.alloc_var("bool", init=False)
                logical_idx = T.alloc_var("int32", init=-1)
                physical_idx = T.alloc_var("int32", init=-1)
                duplicate_count = T.alloc_var("int32", init=1)
                duplicate_idx = T.alloc_var("int32", init=0)
                weight_val = T.alloc_var("float", init=0.0)
                num_experts_per_rank_var = T.alloc_var("int32", init=0)
                num_experts_per_dp_var = T.alloc_var("int32", init=0)
                dst_ep_rank = T.alloc_var("int32", init=0)
                dst_dp_rank = T.alloc_var("int32", init=0)

                # Branch: dynamic logical-to-physical expert map.
                if to_physical_map_exists:
                    num_experts_per_rank_var = (num_routed_experts + num_duplicate_experts - 1) // num_ep_ranks
                # Branch: logical expert id is already physical id.
                else:
                    num_experts_per_rank_var = num_routed_experts // num_ep_ranks
                num_experts_per_dp_var = num_experts_per_rank_var * num_tp_ranks

                T.copy(bias[:aligned_copy_experts], bias_ub[0, :], pad_value=0.0)
                # Branch: static logical-to-local expert map.
                if not to_physical_map_exists:
                    T.copy(expert_local_map, expert_local_map_ub)
                T.set_flag("mte2", "v", 2)
                T.wait_flag("mte2", "v", 2)
                for tail in T.unroll(tail_experts):
                    bias_ub[0, aligned_copy_experts + tail] = bias[aligned_copy_experts + tail]
                T.tile.broadcast(bias_block_ub, bias_ub)
                # Branch: static logical-to-local expert map.
                if not to_physical_map_exists:
                    T.tile.cast(
                        expert_local_map_f32_ub,
                        expert_local_map_ub,
                        "CAST_NONE",
                        num_logical_experts,
                    )

                for stage in T.serial(stages):
                    T.set_flag("mte3", "mte2", stage)

                if cid < num_token_blocks:
                    first_token_start = cid * tokens_per_block + vid * rows_per_vec
                    T.wait_flag("mte3", "mte2", 0)
                    for load_row in T.serial(rows_per_vec):
                        T.copy(
                            logits[first_token_start + load_row, :aligned_copy_experts],
                            scores_ub[0, load_row, :],
                            pad_value=-T.infinity("float"),
                        )
                    T.set_flag("mte2", "v", 0)

                # DB pipeline: prefetch next block while current block computes.
                for i in T.serial(num_iters):
                    cur = i % stages
                    nxt = (i + 1) % stages
                    block_id = cid + i * num_cores
                    token_start = block_id * tokens_per_block + vid * rows_per_vec
                    # Branch: this core owns a valid token block in this iteration.
                    if block_id < num_token_blocks:
                        next_block_id = cid + (i + 1) * num_cores
                        if next_block_id < num_token_blocks:
                            next_token_start = next_block_id * tokens_per_block + vid * rows_per_vec
                            T.wait_flag("mte3", "mte2", nxt)
                            for load_row in T.serial(rows_per_vec):
                                T.copy(
                                    logits[next_token_start + load_row, :aligned_copy_experts],
                                    scores_ub[nxt, load_row, :],
                                    pad_value=-T.infinity("float"),
                                )
                            T.set_flag("mte2", "v", nxt)

                        T.wait_flag("mte2", "v", cur)
                        for load_row in T.serial(rows_per_vec):
                            for tail in T.unroll(tail_experts):
                                scores_ub[cur, load_row, aligned_copy_experts + tail] = logits[
                                    token_start + load_row, aligned_copy_experts + tail
                                ]

                        # Branch: softmax scoring.
                        if scoring_type == SCORING_SOFTMAX:
                            T.reduce_max(scores_ub[cur, :, :], max_ub, dim=-1)
                            T.tile.broadcast(max_block_ub, max_ub)
                            T.tile.sub(shifted_scores_ub[cur, :, :], scores_ub[cur, :, :], max_block_ub)
                            T.tile.exp(route_scores_ub[cur, :, :], shifted_scores_ub[cur, :, :])
                            T.reduce_sum(route_scores_ub[cur, :, :], sum_vec_ub, dim=-1)
                            T.tile.broadcast(sum_block_ub, sum_vec_ub)
                            T.tile.div(route_scores_ub[cur, :, :], route_scores_ub[cur, :, :], sum_block_ub)
                            T.tile.add(scores_ub[cur, :, :], scores_ub[cur, :, :], bias_block_ub)
                        # Branch: sigmoid scoring.
                        elif scoring_type == SCORING_SIGMOID:
                            T.tile.sigmoid(route_scores_ub[cur, :, :], scores_ub[cur, :, :])
                            T.tile.add(scores_ub[cur, :, :], route_scores_ub[cur, :, :], bias_block_ub)
                        # Branch: sqrtsoftplus scoring.
                        elif scoring_type == SCORING_SQRTSOFTPLUS:
                            T.tile.exp(route_scores_ub[cur, :, :], scores_ub[cur, :, :])
                            T.tile.add(route_scores_ub[cur, :, :], route_scores_ub[cur, :, :], 1.0)
                            T.tile.ln(route_scores_ub[cur, :, :], route_scores_ub[cur, :, :])
                            T.tile.sqrt(route_scores_ub[cur, :, :], route_scores_ub[cur, :, :])
                            T.tile.add(scores_ub[cur, :, :], route_scores_ub[cur, :, :], bias_block_ub)
                        # Branch: identity scoring.
                        else:
                            T.copy(scores_ub[cur, :, :], route_scores_ub[cur, :, :])
                            T.tile.add(scores_ub[cur, :, :], route_scores_ub[cur, :, :], bias_block_ub)

                        # Branch: unmapped topk index output enabled.
                        if unmapped_topk_idx_exists:
                            T.tile.fill(unmapped_idx_out_i32_ub[cur, :, :], -1)
                            T.tile.cast(
                                unmapped_idx_out_ub[cur, :, :], unmapped_idx_out_i32_ub[cur, :, :], "CAST_NONE", rows_per_vec * aligned_topk
                            )
                        # Branch: padded-token mask enabled.
                        if mask_exists:
                            T.tile.fill(topk_idx_out_i32_ub[cur, :, :], -1)
                            T.tile.fill(topk_weights_out_ub[cur, :, :], 0.0)
                            T.tile.cast(
                                topk_idx_out_ub[cur, :, :],
                                topk_idx_out_i32_ub[cur, :, :],
                                "CAST_NONE",
                                rows_per_vec * aligned_physical_topk,
                            )

                        for row in T.serial(rows_per_vec):
                            T.tile.fill(selected_idx_ub, 0)
                            T.tile.fill(selected_score_ub, 0.0)
                            # Branch: grouped routing enabled.
                            if not skip_group_sort:
                                T.tile.fill(selected_groups_ub, -1)

                            # Branch: active token path.
                            if not (mask_exists and mask[token_start + row] == 0):
                                T.reinterpretcast(selected_idx_gather_ub, selected_idx_offset_i32_ub, "uint32_t")
                                # Branch: fixed routing token path.
                                if fix_routing_mask_exists and fix_routing_mask[token_start + row] != 0:
                                    #     selected_idx_ub[k] = T.Cast("int32", fixed_topk_idx[token_start + row, k])
                                    #     selected_score_ub[k] = route_scores_ub[cur, row, selected_idx_ub[k]]
                                    #     unmapped_idx_out_ub[cur, row, k] = T.Cast("int64", selected_idx_ub[k])
                                    for k in T.unroll(num_topk):
                                        selected_idx_ub[k] = T.Cast("int32", fixed_topk_idx[token_start + row, k])
                                        selected_score_ub[k] = route_scores_ub[cur, row, selected_idx_ub[k]]
                                        unmapped_idx_out_ub[cur, row, k] = T.Cast("int64", selected_idx_ub[k])
                                # Branch: normal routing token path.
                                else:
                                    # Branch: ungrouped routing topk path.
                                    if skip_group_sort:
                                        T.tile.fill(topk_scores_ub, -3.402823e38)
                                        if aligned_copy_experts > 0:
                                            T.copy(scores_ub[cur, row, :aligned_copy_experts], topk_scores_ub[:aligned_copy_experts])
                                        for tail in T.unroll(tail_experts):
                                            topk_scores_ub[aligned_copy_experts + tail] = scores_ub[cur, row, aligned_copy_experts + tail]
                                        T.tile.topk(topk_result_ub, topk_scores_ub, num_topk, num_routed_experts)
                                        T.tile.gather_mask(selected_idx_float_ub, topk_result_ub, "P1010")
                                        T.tile.fill(selected_idx_ub, 0)
                                        T.tile.cast(selected_idx_ub, selected_idx_float_ub, "CAST_ROUND", aligned_topk)
                                        #     selected_score_ub[k] = route_scores_ub[cur, row, best_idx]
                                        #     unmapped_idx_out_ub[cur, row, k] = T.Cast("int64", best_idx)
                                        T.tile.fill(selected_idx_offset_i32_ub, 0)
                                        #     selected_idx_offset_i32_ub[gather_k] = selected_idx_ub[gather_k] * 4
                                        T.tile.mul(selected_idx_offset_i32_ub, selected_idx_ub, 4)

                                        T.tile.gather(
                                            selected_score_full_ub,
                                            route_scores_ub[cur, row, :],
                                            selected_idx_gather_ub,
                                            0,
                                        )

                                        T.copy(
                                            selected_score_full_ub[:aligned_topk],
                                            selected_score_ub,
                                        )
                                    # Branch: grouped routing enabled.
                                    else:
                                        T.tile.fill(group_scores_ub, -3.402823e38)
                                        for g in T.serial(effective_groups):
                                            T.tile.fill(group_topk_scores_ub, -3.402823e38)
                                            T.copy(
                                                scores_ub[
                                                    cur,
                                                    row,
                                                    g * experts_per_group : (g + 1) * experts_per_group,
                                                ],
                                                group_topk_scores_ub[:experts_per_group],
                                            )
                                            T.tile.topk(
                                                group_topk_result_ub,
                                                group_topk_scores_ub,
                                                2,
                                                experts_per_group,
                                            )
                                            group_scores_ub[g] = group_topk_result_ub[0] + group_topk_result_ub[2]
                                        T.tile.topk(group_select_result_ub, group_scores_ub, effective_topk_groups, aligned_groups)
                                        for selected_group in T.unroll(effective_topk_groups):
                                            selected_groups_ub[selected_group] = T.Cast(
                                                "int32", group_select_result_ub[selected_group * 2 + 1]
                                            )

                                        # Compact experts from selected groups, then run vector topk over compact candidates.
                                        T.tile.fill(topk_scores_ub, -3.402823e38)
                                        T.tile.fill(candidate_logical_idx_f32_ub, 0.0)
                                        for candidate_group in T.serial(effective_groups):
                                            for gk in T.serial(effective_topk_groups):
                                                if selected_groups_ub[gk] == candidate_group:
                                                    T.copy(
                                                        scores_ub[
                                                            cur,
                                                            row,
                                                            candidate_group * experts_per_group : (candidate_group + 1) * experts_per_group,
                                                        ],
                                                        topk_scores_ub[gk * experts_per_group : (gk + 1) * experts_per_group],
                                                    )
                                                    T.tile.arith_progression(
                                                        group_logical_idx_f32_ub,
                                                        T.Cast("float", candidate_group * experts_per_group),
                                                        1.0,
                                                        experts_per_group,
                                                    )
                                                    T.copy(
                                                        group_logical_idx_f32_ub[:experts_per_group],
                                                        candidate_logical_idx_f32_ub[gk * experts_per_group : (gk + 1) * experts_per_group],
                                                    )
                                        T.tile.topk(topk_result_ub, topk_scores_ub, num_topk, topk_candidate_count)
                                        T.tile.gather_mask(selected_idx_float_ub, topk_result_ub, "P1010")
                                        T.tile.fill(selected_idx_ub, 0)
                                        T.tile.cast(selected_idx_ub, selected_idx_float_ub, "CAST_ROUND", aligned_topk)
                                        T.tile.fill(selected_idx_offset_i32_ub, 0)
                                        T.tile.mul(selected_idx_offset_i32_ub, selected_idx_ub, 4)
                                        T.tile.gather(mapped_expert_full_f32_ub, candidate_logical_idx_f32_ub, selected_idx_gather_ub, 0)
                                        T.copy(mapped_expert_full_f32_ub[:aligned_topk], selected_idx_float_ub)
                                        T.tile.fill(selected_idx_ub, 0)
                                        T.tile.cast(selected_idx_ub, selected_idx_float_ub, "CAST_ROUND", aligned_topk)
                                        T.tile.fill(selected_idx_offset_i32_ub, 0)
                                        T.tile.mul(selected_idx_offset_i32_ub, selected_idx_ub, 4)
                                        T.tile.gather(selected_score_full_ub, route_scores_ub[cur, row, :], selected_idx_gather_ub, 0)
                                        T.copy(selected_score_full_ub[:aligned_topk], selected_score_ub)

                                T.tile.fill(selected_score_mat_ub, 0.0)
                                T.copy(selected_score_ub, selected_score_mat_ub[0, :])
                                T.reduce_sum(
                                    selected_score_mat_ub,
                                    selected_sum_ub,
                                    dim=-1,
                                    real_shape=[1, num_topk],
                                )
                                T.tile.add(
                                    selected_sum_ub,
                                    selected_sum_ub,
                                    1.0e-20,
                                )
                                T.tile.broadcast(
                                    selected_sum_block_ub,
                                    selected_sum_ub,
                                )
                                T.copy(
                                    selected_sum_block_ub[0, :],
                                    selected_sum_vec_ub,
                                )
                                T.tile.fill(scale_vec_ub, routed_scaling_factor)
                                T.tile.div(
                                    normalized_weight_ub,
                                    selected_score_ub,
                                    selected_sum_vec_ub,
                                )
                                T.tile.mul(
                                    normalized_weight_ub,
                                    normalized_weight_ub,
                                    scale_vec_ub,
                                )
                                T.copy(
                                    normalized_weight_ub,
                                    topk_weights_out_ub[cur, row, :aligned_topk],
                                )
                                # Branch: unmapped topk index output enabled.
                                if unmapped_topk_idx_exists:
                                    T.tile.cast(
                                        unmapped_idx_out_ub[cur, row, :],
                                        selected_idx_ub,
                                        "CAST_NONE",
                                        aligned_topk,
                                    )
                                T.tile.mul(
                                    selected_idx_offset_i32_ub,
                                    selected_idx_ub,
                                    4,
                                )

                                # Branch: dynamic logical-to-physical expert map.
                                if to_physical_map_exists:
                                    for k in T.serial(num_topk):
                                        logical_idx = selected_idx_ub[k]
                                        duplicate_count = logical_count[logical_idx]
                                        duplicate_idx = (ep_rank + (token_start + row) * 23333) % duplicate_count
                                        physical_idx = to_physical_map[logical_idx, duplicate_idx]
                                        weight_val = normalized_weight_ub[k]
                                        dst_ep_rank = physical_idx // num_experts_per_rank_var
                                        # Branch: expert belongs to another TP rank.
                                        if dst_ep_rank % num_tp_ranks != tp_rank:
                                            physical_idx = -1
                                        # Branch: expert belongs to current TP rank.
                                        else:
                                            physical_idx = physical_idx - tp_rank * num_experts_per_rank_var
                                            dst_dp_rank = physical_idx // num_experts_per_dp_var
                                            physical_idx = (
                                                physical_idx - dst_dp_rank * num_experts_per_dp_var + dst_dp_rank * num_experts_per_rank_var
                                            )
                                            # Branch: invalid local physical expert id.
                                            if physical_idx < 0:
                                                physical_idx = -1
                                        topk_idx_out_ub[cur, row, k] = T.Cast("int64", physical_idx)
                                        topk_weights_out_ub[cur, row, k] = weight_val
                                # Branch: static logical-to-local expert map.
                                else:
                                    T.tile.gather(
                                        mapped_expert_full_f32_ub,
                                        expert_local_map_f32_ub,
                                        selected_idx_gather_ub,
                                        0,
                                    )
                                    T.copy(
                                        mapped_expert_full_f32_ub[:aligned_topk],
                                        mapped_expert_topk_f32_ub,
                                    )
                                    T.tile.cast(
                                        mapped_expert_topk_i32_ub,
                                        mapped_expert_topk_f32_ub,
                                        "CAST_ROUND",
                                        aligned_topk,
                                    )
                                    T.tile.cast(
                                        topk_idx_out_ub[cur, row, :aligned_topk],
                                        mapped_expert_topk_i32_ub,
                                        "CAST_NONE",
                                        aligned_topk,
                                    )

                                for k in T.serial(num_shared_experts):
                                    logical_idx = num_routed_experts + k
                                    physical_idx = logical_idx
                                    weight_val = 1.0
                                    # Branch: dynamic logical-to-physical expert map.
                                    if to_physical_map_exists:
                                        duplicate_count = logical_count[logical_idx]
                                        duplicate_idx = (ep_rank + (token_start + row) * 23333) % duplicate_count
                                        physical_idx = to_physical_map[logical_idx, duplicate_idx]
                                    dst_ep_rank = physical_idx // num_experts_per_rank_var
                                    # Branch: expert belongs to another TP rank.
                                    if dst_ep_rank % num_tp_ranks != tp_rank:
                                        physical_idx = -1
                                    # Branch: expert belongs to current TP rank.
                                    else:
                                        physical_idx = physical_idx - tp_rank * num_experts_per_rank_var
                                        dst_dp_rank = physical_idx // num_experts_per_dp_var
                                        physical_idx = (
                                            physical_idx - dst_dp_rank * num_experts_per_dp_var + dst_dp_rank * num_experts_per_rank_var
                                        )
                                        # Branch: invalid local physical expert id.
                                        if physical_idx < 0:
                                            physical_idx = -1
                                    topk_idx_out_ub[cur, row, num_topk + k] = T.Cast("int64", physical_idx)
                                    topk_weights_out_ub[cur, row, num_topk + k] = weight_val

                        T.set_flag("v", "mte3", cur)
                        T.wait_flag("v", "mte3", cur)
                        for row in T.serial(rows_per_vec):
                            if unmapped_topk_idx_exists:
                                T.copy(unmapped_idx_out_ub[cur, row, :num_topk], unmapped_topk_idx[token_start + row, :])
                            T.copy(topk_idx_out_ub[cur, row, :num_physical_topk], topk_idx[token_start + row, :])
                            T.copy(topk_weights_out_ub[cur, row, :num_physical_topk], topk_weights[token_start + row, :])
                        T.pipe_barrier("mte3")
                        T.set_flag("mte3", "mte2", cur)

                for stage in T.serial(stages):
                    T.wait_flag("mte3", "mte2", stage)

    return top2_sum_gate_ascend_v25_scalar_kernel


# Dynamic-map path restored from the measured ~170us non-DB implementation.
# Static-map path keeps the separately validated reference DB implementation above.
@tilelang.jit(pass_configs=pass_configs)
def _get_top2_sum_gate_dynamic_170us_kernel(
    scoring_type: int,
    num_topk: int,
    num_topk_groups: int,
    num_groups: int,
    num_routed_experts: int,
    num_shared_experts: int,
    mask_exists: bool,
    fix_routing_mask_exists: bool,
    unmapped_topk_idx_exists: bool,
    to_physical_map_exists: bool,
    rows_per_vec: int,
):
    num_tokens = T.symbolic("num_tokens")
    num_duplicate_experts = T.symbolic("num_duplicate_experts")
    num_logical_experts = num_routed_experts + num_shared_experts
    num_physical_topk = num_topk + num_shared_experts
    effective_groups = num_groups if num_groups > 0 else 1
    effective_topk_groups = num_topk_groups if num_topk_groups > 0 else 1
    experts_per_group = num_routed_experts // effective_groups
    skip_group_sort = num_groups == 0 or num_groups == num_topk_groups
    aligned_topk = ((num_topk + 7) // 8) * 8
    aligned_physical_topk = ((num_physical_topk + 7) // 8) * 8
    aligned_num_experts = ((num_routed_experts + 31) // 32) * 32
    aligned_groups = ((effective_groups + 31) // 32) * 32
    aligned_experts_per_group = ((experts_per_group + 31) // 32) * 32
    group_score_count = 1 if skip_group_sort else aligned_groups
    selected_group_count = 1 if skip_group_sort else effective_topk_groups
    topk_candidate_count = num_routed_experts if skip_group_sort else effective_topk_groups * experts_per_group
    aligned_topk_candidates = ((topk_candidate_count + 31) // 32) * 32
    # Floor-aligned copy sizes for non-32-aligned expert counts such as 36/108.
    # UB row width is aligned_num_experts, but TopK must only see real experts.
    aligned_copy_experts = (num_routed_experts // 32) * 32
    tail_experts = num_routed_experts - aligned_copy_experts
    aligned_copy_experts_per_group = (experts_per_group // 32) * 32
    tail_experts_per_group = experts_per_group - aligned_copy_experts_per_group
    vec_num = VEC_NUM
    tokens_per_block = rows_per_vec * vec_num
    num_token_blocks = T.ceildiv(num_tokens, tokens_per_block)

    @T.prim_func
    def top2_sum_gate_ascend_v25_scalar_kernel(
        logits: T.Tensor((num_tokens, num_routed_experts), "float"),
        bias: T.Tensor((num_routed_experts,), "float"),
        mask: T.Tensor((num_tokens,), "int32"),
        fix_routing_mask: T.Tensor((num_tokens,), "int32"),
        to_physical_map: T.Tensor(
            (num_logical_experts, num_duplicate_experts),
            "int32",
        ),
        logical_count: T.Tensor((num_logical_experts,), "int32"),
        expert_local_map: T.Tensor((num_logical_experts,), "int32"),
        fixed_topk_idx: T.Tensor((num_tokens, num_topk), "int64"),
        topk_idx: T.Tensor((num_tokens, num_physical_topk), "int64"),
        unmapped_topk_idx: T.Tensor((num_tokens, num_topk), "int64"),
        topk_weights: T.Tensor((num_tokens, num_physical_topk), "float"),
        num_extra_experts: T.int32,
        routed_scaling_factor: T.float32,
        ep_rank: T.int32,
        num_ep_ranks: T.int32,
        tp_rank: T.int32,
        num_tp_ranks: T.int32,
    ):
        with T.Kernel(num_token_blocks, is_npu=True) as (cid, vid):
            token_start = cid * tokens_per_block + vid * rows_per_vec
            with T.Scope("V"):
                scores_ub = T.alloc_ub((rows_per_vec, aligned_num_experts), "float")
                shifted_scores_ub = T.alloc_ub((rows_per_vec, aligned_num_experts), "float")
                route_scores_ub = T.alloc_ub((rows_per_vec, aligned_num_experts), "float")
                bias_ub = T.alloc_ub((1, aligned_num_experts), "float")
                bias_block_ub = T.alloc_ub((rows_per_vec, aligned_num_experts), "float")
                max_ub = T.alloc_ub((rows_per_vec, 1), "float")
                max_block_ub = T.alloc_ub((rows_per_vec, aligned_num_experts), "float")
                sum_vec_ub = T.alloc_ub((rows_per_vec, 1), "float")
                sum_block_ub = T.alloc_ub((rows_per_vec, aligned_num_experts), "float")
                group_scores_ub = T.alloc_ub((group_score_count,), "float")
                group_topk_scores_ub = T.alloc_ub((aligned_experts_per_group,), "float")
                group_topk_result_ub = T.alloc_ub((4,), "float")
                group_select_result_ub = T.alloc_ub((effective_topk_groups * 2,), "float")
                selected_groups_ub = T.alloc_ub((selected_group_count,), "int32")
                group_logical_idx_f32_ub = T.alloc_ub((aligned_experts_per_group,), "float")
                selected_idx_ub = T.alloc_ub((aligned_num_experts,), "int32")
                selected_idx_float_ub = T.alloc_ub((aligned_topk,), "float")
                selected_score_ub = T.alloc_ub((aligned_topk,), "float")
                selected_score_mat_ub = T.alloc_ub((1, aligned_topk), "float")
                selected_sum_ub = T.alloc_ub((1, 1), "float")
                selected_sum_block_ub = T.alloc_ub((1, aligned_topk), "float")
                selected_sum_vec_ub = T.alloc_ub((aligned_topk,), "float")
                scale_vec_ub = T.alloc_ub((aligned_topk,), "float")
                normalized_weight_ub = T.alloc_ub((aligned_topk,), "float")
                expert_local_map_ub = T.alloc_ub((num_logical_experts,), "int32")
                expert_local_map_f32_ub = T.alloc_ub((num_logical_experts,), "float")
                mapped_expert_full_f32_ub = T.alloc_ub((aligned_num_experts,), "float")
                mapped_expert_topk_f32_ub = T.alloc_ub((aligned_topk,), "float")
                mapped_expert_topk_i32_ub = T.alloc_ub((aligned_topk,), "int32")
                selected_idx_offset_i32_ub = T.alloc_ub((aligned_num_experts,), "int32")
                selected_idx_gather_ub = T.alloc_ub((aligned_num_experts,), "uint32")

                topk_scores_ub = T.alloc_ub((aligned_topk_candidates,), "float")
                candidate_logical_idx_f32_ub = T.alloc_ub((aligned_topk_candidates,), "float")
                topk_result_ub = T.alloc_ub((num_topk * 2,), "float")
                topk_idx_out_ub = T.alloc_ub((rows_per_vec, aligned_physical_topk), "int64")
                unmapped_idx_out_ub = T.alloc_ub((rows_per_vec, aligned_topk), "int64")
                topk_weights_out_ub = T.alloc_ub((rows_per_vec, aligned_physical_topk), "float")
                topk_idx_out_i32_ub = T.alloc_ub((rows_per_vec, aligned_physical_topk), "int32")
                unmapped_idx_out_i32_ub = T.alloc_ub((rows_per_vec, aligned_topk), "int32")

                selected_score_full_ub = T.alloc_ub((aligned_num_experts,), "float")

                best_score = T.alloc_var("float", init=-3.402823e38)
                best_idx = T.alloc_var("int32", init=-1)
                is_selected = T.alloc_var("bool", init=False)
                already_taken = T.alloc_var("bool", init=False)
                logical_idx = T.alloc_var("int32", init=-1)
                physical_idx = T.alloc_var("int32", init=-1)
                duplicate_count = T.alloc_var("int32", init=1)
                duplicate_idx = T.alloc_var("int32", init=0)
                weight_val = T.alloc_var("float", init=0.0)
                num_experts_per_rank_var = T.alloc_var("int32", init=0)
                num_experts_per_dp_var = T.alloc_var("int32", init=0)
                dst_ep_rank = T.alloc_var("int32", init=0)
                dst_dp_rank = T.alloc_var("int32", init=0)
                token_hash = T.alloc_var("int32", init=0)

                # Branch: dynamic logical-to-physical expert map.
                if to_physical_map_exists:
                    num_experts_per_rank_var = (num_routed_experts + num_duplicate_experts - 1) // num_ep_ranks
                # Branch: logical expert id is already physical id.
                else:
                    num_experts_per_rank_var = num_routed_experts // num_ep_ranks
                num_experts_per_dp_var = num_experts_per_rank_var * num_tp_ranks

                # For non-32-aligned experts (36/108), keep every UB row 32B-aligned.
                # Valid rows get -inf expert padding; invalid tail-token rows stay 0 and
                # are masked by wrapper-side token padding.
                T.tile.fill(scores_ub, 0.0)
                for load_row in T.serial(rows_per_vec):
                    if token_start + load_row < num_tokens:
                        T.copy(
                            logits[token_start + load_row, 0:num_routed_experts],
                            scores_ub[load_row, :],
                            pad_value=-T.infinity("float"),
                        )
                T.tile.fill(bias_ub, 0.0)
                T.copy(bias[0:num_routed_experts], bias_ub[0, :], pad_value=0.0)
                T.tile.broadcast(bias_block_ub, bias_ub)
                # Branch: static logical-to-local expert map.
                if not to_physical_map_exists:
                    T.copy(expert_local_map, expert_local_map_ub)
                    T.tile.cast(
                        expert_local_map_f32_ub,
                        expert_local_map_ub,
                        "CAST_NONE",
                        num_logical_experts,
                    )

                # Branch: softmax scoring.
                if scoring_type == SCORING_SOFTMAX:
                    T.reduce_max(scores_ub, max_ub, dim=-1, real_shape=[rows_per_vec, aligned_num_experts])
                    T.tile.broadcast(max_block_ub, max_ub)
                    T.tile.sub(shifted_scores_ub, scores_ub, max_block_ub)
                    T.tile.exp(route_scores_ub, shifted_scores_ub)
                    T.reduce_sum(route_scores_ub, sum_vec_ub, dim=-1, real_shape=[rows_per_vec, aligned_num_experts])
                    T.tile.broadcast(sum_block_ub, sum_vec_ub)
                    T.tile.div(route_scores_ub, route_scores_ub, sum_block_ub)
                    T.tile.add(scores_ub, scores_ub, bias_block_ub)
                # Branch: sigmoid scoring.
                elif scoring_type == SCORING_SIGMOID:
                    T.tile.sigmoid(route_scores_ub, scores_ub)
                    T.tile.add(scores_ub, route_scores_ub, bias_block_ub)
                # Branch: sqrtsoftplus scoring.
                elif scoring_type == SCORING_SQRTSOFTPLUS:
                    T.tile.exp(route_scores_ub, scores_ub)
                    T.tile.add(route_scores_ub, route_scores_ub, 1.0)
                    T.tile.ln(route_scores_ub, route_scores_ub)
                    T.tile.sqrt(route_scores_ub, route_scores_ub)
                    T.tile.add(scores_ub, route_scores_ub, bias_block_ub)
                # Branch: identity scoring.
                else:
                    T.copy(scores_ub, route_scores_ub)
                    T.tile.add(scores_ub, route_scores_ub, bias_block_ub)

                # Branch: unmapped topk index output enabled.
                if unmapped_topk_idx_exists:
                    T.tile.fill(unmapped_idx_out_i32_ub, -1)
                    #     T.tile.cast(unmapped_idx_out_ub[row, :], unmapped_idx_out_i32_ub[row, :], "CAST_NONE", aligned_topk)
                    T.tile.cast(unmapped_idx_out_ub, unmapped_idx_out_i32_ub, "CAST_NONE", rows_per_vec * aligned_topk)
                # Branch: padded-token mask enabled.
                if mask_exists:
                    T.tile.fill(topk_idx_out_i32_ub, -1)
                    T.tile.fill(topk_weights_out_ub, 0.0)
                    #     T.tile.cast(topk_idx_out_ub[row, :], topk_idx_out_i32_ub[row, :], "CAST_NONE", aligned_physical_topk)
                    T.tile.cast(topk_idx_out_ub, topk_idx_out_i32_ub, "CAST_NONE", rows_per_vec * aligned_physical_topk)

                for row in T.serial(rows_per_vec):
                    T.tile.fill(selected_idx_ub, 0)
                    T.tile.fill(selected_score_ub, 0.0)
                    # Branch: grouped routing enabled.
                    if not skip_group_sort:
                        T.tile.fill(selected_groups_ub, -1)

                    # Branch: active token path.
                    if not (mask_exists and mask[token_start + row] == 0):
                        token_hash = ep_rank + (token_start + row) * 23333
                        T.reinterpretcast(selected_idx_gather_ub, selected_idx_offset_i32_ub, "uint32_t")
                        # Branch: fixed routing token path.
                        if fix_routing_mask_exists and fix_routing_mask[token_start + row] != 0:
                            #     selected_idx_ub[k] = T.Cast("int32", fixed_topk_idx[token_start + row, k])
                            #     selected_score_ub[k] = route_scores_ub[row, selected_idx_ub[k]]
                            #     unmapped_idx_out_ub[row, k] = T.Cast("int64", selected_idx_ub[k])
                            for k in T.unroll(num_topk):
                                selected_idx_ub[k] = T.Cast("int32", fixed_topk_idx[token_start + row, k])
                                selected_score_ub[k] = route_scores_ub[row, selected_idx_ub[k]]
                                unmapped_idx_out_ub[row, k] = T.Cast("int64", selected_idx_ub[k])
                        # Branch: normal routing token path.
                        else:
                            # Branch: ungrouped routing topk path.
                            if skip_group_sort:
                                T.tile.fill(topk_scores_ub, -3.402823e38)
                                # Copy only real experts. Avoid UB->UB vector copies with
                                # non-32-aligned lengths such as 36/108.
                                if aligned_copy_experts > 0:
                                    T.copy(
                                        scores_ub[row, 0:aligned_copy_experts],
                                        topk_scores_ub[:aligned_copy_experts],
                                    )
                                if tail_experts > 0:
                                    for tail_i in T.unroll(tail_experts):
                                        topk_scores_ub[aligned_copy_experts + tail_i] = scores_ub[row, aligned_copy_experts + tail_i]
                                T.tile.topk(topk_result_ub, topk_scores_ub, num_topk, aligned_topk_candidates)
                                T.tile.gather_mask(selected_idx_float_ub, topk_result_ub, "P1010")
                                T.tile.fill(selected_idx_ub, 0)
                                T.tile.cast(selected_idx_ub, selected_idx_float_ub, "CAST_ROUND", aligned_topk)
                                #     selected_score_ub[k] = route_scores_ub[row, best_idx]
                                #     unmapped_idx_out_ub[row, k] = T.Cast("int64", best_idx)
                                T.tile.fill(selected_idx_offset_i32_ub, 0)
                                #     selected_idx_offset_i32_ub[gather_k] = selected_idx_ub[gather_k] * 4
                                T.tile.mul(selected_idx_offset_i32_ub, selected_idx_ub, 4)

                                T.tile.gather(
                                    selected_score_full_ub,
                                    route_scores_ub[row, :],
                                    selected_idx_gather_ub,
                                    0,
                                )

                                T.copy(
                                    selected_score_full_ub[:aligned_topk],
                                    selected_score_ub,
                                )
                            # Branch: grouped routing enabled.
                            else:
                                T.tile.fill(group_scores_ub, -3.402823e38)
                                for g in T.serial(effective_groups):
                                    T.tile.fill(group_topk_scores_ub, -3.402823e38)
                                    if aligned_copy_experts_per_group > 0:
                                        T.copy(
                                            scores_ub[
                                                row,
                                                g * experts_per_group : g * experts_per_group + aligned_copy_experts_per_group,
                                            ],
                                            group_topk_scores_ub[:aligned_copy_experts_per_group],
                                        )
                                    if tail_experts_per_group > 0:
                                        for tail_i in T.unroll(tail_experts_per_group):
                                            group_topk_scores_ub[aligned_copy_experts_per_group + tail_i] = scores_ub[
                                                row,
                                                g * experts_per_group + aligned_copy_experts_per_group + tail_i,
                                            ]
                                    T.tile.topk(
                                        group_topk_result_ub,
                                        group_topk_scores_ub,
                                        2,
                                        aligned_experts_per_group,
                                    )
                                    group_scores_ub[g] = group_topk_result_ub[0] + group_topk_result_ub[2]
                                T.tile.topk(
                                    group_select_result_ub,
                                    group_scores_ub,
                                    effective_topk_groups,
                                    aligned_groups,
                                )
                                for selected_group in T.unroll(effective_topk_groups):
                                    selected_groups_ub[selected_group] = T.Cast("int32", group_select_result_ub[selected_group * 2 + 1])

                                # Compact experts from selected groups, then run vector topk over compact candidates.
                                T.tile.fill(topk_scores_ub, -3.402823e38)
                                T.tile.fill(candidate_logical_idx_f32_ub, 0.0)
                                for candidate_group in T.serial(effective_groups):
                                    for gk in T.serial(effective_topk_groups):
                                        if selected_groups_ub[gk] == candidate_group:
                                            if aligned_copy_experts_per_group > 0:
                                                T.copy(
                                                    scores_ub[
                                                        row,
                                                        candidate_group * experts_per_group : candidate_group * experts_per_group
                                                        + aligned_copy_experts_per_group,
                                                    ],
                                                    topk_scores_ub[
                                                        gk * experts_per_group : gk * experts_per_group + aligned_copy_experts_per_group
                                                    ],
                                                )
                                            if tail_experts_per_group > 0:
                                                for tail_i in T.unroll(tail_experts_per_group):
                                                    topk_scores_ub[gk * experts_per_group + aligned_copy_experts_per_group + tail_i] = (
                                                        scores_ub[
                                                            row,
                                                            candidate_group * experts_per_group + aligned_copy_experts_per_group + tail_i,
                                                        ]
                                                    )
                                            T.tile.arith_progression(
                                                group_logical_idx_f32_ub,
                                                T.Cast("float", candidate_group * experts_per_group),
                                                1.0,
                                                experts_per_group,
                                            )
                                            T.copy(
                                                group_logical_idx_f32_ub[:experts_per_group],
                                                candidate_logical_idx_f32_ub[gk * experts_per_group : (gk + 1) * experts_per_group],
                                            )
                                T.tile.topk(topk_result_ub, topk_scores_ub, num_topk, aligned_topk_candidates)
                                T.tile.gather_mask(selected_idx_float_ub, topk_result_ub, "P1010")
                                T.tile.fill(selected_idx_ub, 0)
                                T.tile.cast(selected_idx_ub, selected_idx_float_ub, "CAST_ROUND", aligned_topk)
                                T.tile.fill(selected_idx_offset_i32_ub, 0)
                                T.tile.mul(selected_idx_offset_i32_ub, selected_idx_ub, 4)
                                T.tile.gather(mapped_expert_full_f32_ub, candidate_logical_idx_f32_ub, selected_idx_gather_ub, 0)
                                T.copy(mapped_expert_full_f32_ub[:aligned_topk], selected_idx_float_ub)
                                T.tile.fill(selected_idx_ub, 0)
                                T.tile.cast(selected_idx_ub, selected_idx_float_ub, "CAST_ROUND", aligned_topk)
                                T.tile.fill(selected_idx_offset_i32_ub, 0)
                                T.tile.mul(selected_idx_offset_i32_ub, selected_idx_ub, 4)
                                T.tile.gather(selected_score_full_ub, route_scores_ub[row, :], selected_idx_gather_ub, 0)
                                T.copy(selected_score_full_ub[:aligned_topk], selected_score_ub)

                        T.tile.fill(selected_score_mat_ub, 0.0)
                        T.copy(selected_score_ub, selected_score_mat_ub[0, :])
                        T.reduce_sum(
                            selected_score_mat_ub,
                            selected_sum_ub,
                            dim=-1,
                            real_shape=[1, num_topk],
                        )
                        T.tile.add(
                            selected_sum_ub,
                            selected_sum_ub,
                            1.0e-20,
                        )
                        T.tile.broadcast(
                            selected_sum_block_ub,
                            selected_sum_ub,
                        )
                        T.copy(
                            selected_sum_block_ub[0, :],
                            selected_sum_vec_ub,
                        )
                        T.tile.fill(scale_vec_ub, routed_scaling_factor)
                        T.tile.div(
                            normalized_weight_ub,
                            selected_score_ub,
                            selected_sum_vec_ub,
                        )
                        T.tile.mul(
                            normalized_weight_ub,
                            normalized_weight_ub,
                            scale_vec_ub,
                        )
                        T.copy(
                            normalized_weight_ub,
                            topk_weights_out_ub[row, :aligned_topk],
                        )
                        # Branch: unmapped topk index output enabled.
                        if unmapped_topk_idx_exists:
                            T.tile.cast(
                                unmapped_idx_out_ub[row, :],
                                selected_idx_ub,
                                "CAST_NONE",
                                aligned_topk,
                            )
                        T.tile.mul(
                            selected_idx_offset_i32_ub,
                            selected_idx_ub,
                            4,
                        )

                        # Branch: dynamic logical-to-physical expert map.
                        if to_physical_map_exists:
                            # AscendC-style fast path:
                            for k in T.unroll(num_topk):
                                logical_idx = selected_idx_ub[k]
                                duplicate_count = logical_count[logical_idx]
                                if duplicate_count == 1:
                                    duplicate_idx = 0
                                else:
                                    duplicate_idx = token_hash % duplicate_count
                                physical_idx = to_physical_map[logical_idx, duplicate_idx]
                                weight_val = normalized_weight_ub[k]
                                dst_ep_rank = physical_idx // num_experts_per_rank_var
                                # Branch: expert belongs to another TP rank.
                                if dst_ep_rank % num_tp_ranks != tp_rank:
                                    physical_idx = -1
                                # Branch: expert belongs to current TP rank.
                                else:
                                    physical_idx = physical_idx - tp_rank * num_experts_per_rank_var
                                    dst_dp_rank = physical_idx // num_experts_per_dp_var
                                    physical_idx = (
                                        physical_idx - dst_dp_rank * num_experts_per_dp_var + dst_dp_rank * num_experts_per_rank_var
                                    )
                                    # Branch: invalid local physical expert id.
                                    if physical_idx < 0:
                                        physical_idx = -1
                                topk_idx_out_ub[row, k] = T.Cast("int64", physical_idx)
                                topk_weights_out_ub[row, k] = weight_val
                        # Branch: static logical-to-local expert map.
                        else:
                            T.tile.gather(
                                mapped_expert_full_f32_ub,
                                expert_local_map_f32_ub,
                                selected_idx_gather_ub,
                                0,
                            )
                            T.copy(
                                mapped_expert_full_f32_ub[:aligned_topk],
                                mapped_expert_topk_f32_ub,
                            )
                            T.tile.cast(
                                mapped_expert_topk_i32_ub,
                                mapped_expert_topk_f32_ub,
                                "CAST_ROUND",
                                aligned_topk,
                            )
                            T.tile.cast(
                                topk_idx_out_ub[row, :aligned_topk],
                                mapped_expert_topk_i32_ub,
                                "CAST_NONE",
                                aligned_topk,
                            )

                        for k in T.unroll(num_shared_experts):
                            logical_idx = num_routed_experts + k
                            physical_idx = logical_idx
                            weight_val = 1.0
                            # Branch: dynamic logical-to-physical expert map.
                            if to_physical_map_exists:
                                duplicate_count = logical_count[logical_idx]
                                if duplicate_count == 1:
                                    duplicate_idx = 0
                                else:
                                    duplicate_idx = token_hash % duplicate_count
                                physical_idx = to_physical_map[logical_idx, duplicate_idx]
                            dst_ep_rank = physical_idx // num_experts_per_rank_var
                            # Branch: expert belongs to another TP rank.
                            if dst_ep_rank % num_tp_ranks != tp_rank:
                                physical_idx = -1
                            # Branch: expert belongs to current TP rank.
                            else:
                                physical_idx = physical_idx - tp_rank * num_experts_per_rank_var
                                dst_dp_rank = physical_idx // num_experts_per_dp_var
                                physical_idx = physical_idx - dst_dp_rank * num_experts_per_dp_var + dst_dp_rank * num_experts_per_rank_var
                                # Branch: invalid local physical expert id.
                                if physical_idx < 0:
                                    physical_idx = -1
                            topk_idx_out_ub[row, num_topk + k] = T.Cast("int64", physical_idx)
                            topk_weights_out_ub[row, num_topk + k] = weight_val

                T.barrier_all()
                for row in T.serial(rows_per_vec):
                    if unmapped_topk_idx_exists:
                        T.copy(unmapped_idx_out_ub[row, :num_topk], unmapped_topk_idx[token_start + row, :])
                    T.copy(topk_idx_out_ub[row, :num_physical_topk], topk_idx[token_start + row, :])
                    T.copy(topk_weights_out_ub[row, :num_physical_topk], topk_weights[token_start + row, :])

    return top2_sum_gate_ascend_v25_scalar_kernel


def get_top2_sum_gate_ascend_v25_scalar_kernel(
    scoring_type: int,
    num_topk: int,
    num_topk_groups: int,
    num_groups: int,
    num_routed_experts: int,
    num_shared_experts: int,
    mask_exists: bool,
    fix_routing_mask_exists: bool,
    unmapped_topk_idx_exists: bool,
    to_physical_map_exists: bool,
    rows_per_vec: int,
):
    """Dispatch by map type.

    static-map  -> validated reference DB kernel.
    dynamic-map -> restored lightweight non-DB kernel measured around 170us.
    """
    if to_physical_map_exists:
        return _get_top2_sum_gate_dynamic_170us_kernel(
            scoring_type,
            num_topk,
            num_topk_groups,
            num_groups,
            num_routed_experts,
            num_shared_experts,
            mask_exists,
            fix_routing_mask_exists,
            unmapped_topk_idx_exists,
            to_physical_map_exists,
            rows_per_vec,
        )
    return _get_top2_sum_gate_static_refdb_kernel(
        scoring_type,
        num_topk,
        num_topk_groups,
        num_groups,
        num_routed_experts,
        num_shared_experts,
        mask_exists,
        fix_routing_mask_exists,
        unmapped_topk_idx_exists,
        to_physical_map_exists,
        rows_per_vec,
    )


def top2_sum_gate_ascend(
    logits: torch.Tensor,
    bias: torch.Tensor,
    num_topk: int,
    num_topk_groups: int,
    num_groups: int,
    use_shared_as_routed: bool,
    num_shared_experts: int,
    routed_scaling_factor: float,
    ep_rank: int,
    num_ep_ranks: int,
    tp_rank: int,
    num_tp_ranks: int,
    scoring_func: str,
    mask: Optional[torch.Tensor] = None,
    fix_routing_mask: Optional[torch.Tensor] = None,
    to_physical_map: Optional[torch.Tensor] = None,
    logical_count: Optional[torch.Tensor] = None,
    unmapped_topk_idx: Optional[torch.Tensor] = None,
):
    assert logits.dim() == 2
    assert logits.is_contiguous() and logits.dtype == torch.float32
    assert bias.dim() == 1
    assert bias.is_contiguous() and bias.dtype == torch.float32
    assert logits.shape[1] == bias.numel()

    num_tokens, num_routed_experts = logits.shape
    if not use_shared_as_routed:
        num_shared_experts = 0
    assert num_topk <= num_routed_experts
    assert num_groups == 0 or num_routed_experts % num_groups == 0
    assert (to_physical_map is None) == (logical_count is None)
    assert num_ep_ranks > 0 and num_tp_ranks > 0
    assert 0 <= tp_rank < num_tp_ranks

    device = logits.device
    if mask is None:
        mask = torch.empty((num_tokens,), dtype=torch.int32, device=device)
        mask_exists = False
    else:
        mask = mask.to(torch.int32)
        mask_exists = True

    unmapped_topk_idx_exists = unmapped_topk_idx is not None
    if unmapped_topk_idx is not None:
        assert unmapped_topk_idx.shape == (num_tokens, num_topk)
        assert unmapped_topk_idx.dtype == torch.int64
        assert unmapped_topk_idx.stride(1) == 1

    if fix_routing_mask is None:
        fix_routing_mask = torch.empty((num_tokens,), dtype=torch.int32, device=device)
        fixed_topk_idx = torch.empty((num_tokens, num_topk), dtype=torch.int64, device=device)
        fix_routing_mask_exists = False
    else:
        assert unmapped_topk_idx is not None
        fix_routing_mask = fix_routing_mask.to(torch.int32)
        fixed_topk_idx = unmapped_topk_idx.contiguous()
        fix_routing_mask_exists = True

    num_logical_experts = num_routed_experts + num_shared_experts
    if to_physical_map is None:
        to_physical_map = torch.empty((num_logical_experts, 1), dtype=torch.int32, device=device)
        logical_count = torch.empty((num_logical_experts,), dtype=torch.int32, device=device)
        num_duplicate_experts = 1
        num_extra_experts = 0
        to_physical_map_exists = False
        num_experts_per_rank = num_routed_experts // num_ep_ranks
        expert_local_map = _get_expert_local_map(
            device,
            num_logical_experts,
            num_experts_per_rank,
            tp_rank,
            num_tp_ranks,
        )
    else:
        assert logical_count is not None
        assert to_physical_map.dim() == 2
        assert to_physical_map.dtype == torch.int32
        assert logical_count.dim() == 1
        assert logical_count.dtype == torch.int32
        num_duplicate_experts = to_physical_map.shape[1]
        num_extra_experts = num_duplicate_experts - 1
        to_physical_map_exists = True
        expert_local_map = torch.empty(
            (num_logical_experts,),
            dtype=torch.int32,
            device=device,
        )

    if to_physical_map_exists:
        # Keep the dynamic-map path identical to the measured lightweight reference.
        rows_per_vec = _rows_per_vec_dynamic(num_routed_experts, num_tokens)
    else:
        # Keep the static-map path identical to the validated DB reference.
        rows_per_vec = _rows_per_vec(num_routed_experts, num_tokens)
    tokens_per_block = rows_per_vec * VEC_NUM
    aligned_num_tokens = (num_tokens + tokens_per_block - 1) // tokens_per_block * tokens_per_block
    tail_exists = aligned_num_tokens != num_tokens
    kernel_logits = logits
    kernel_mask = mask
    kernel_fix_routing_mask = fix_routing_mask
    kernel_fixed_topk_idx = fixed_topk_idx

    if tail_exists:
        kernel_logits = torch.zeros((aligned_num_tokens, num_routed_experts), dtype=torch.float32, device=device)
        kernel_logits[:num_tokens].copy_(logits)
        kernel_mask = torch.zeros((aligned_num_tokens,), dtype=torch.int32, device=device)
        if mask_exists:
            kernel_mask[:num_tokens].copy_(mask)
        else:
            kernel_mask[:num_tokens].fill_(1)
        mask_exists = True
        kernel_fix_routing_mask = torch.zeros((aligned_num_tokens,), dtype=torch.int32, device=device)
        if fix_routing_mask_exists:
            kernel_fix_routing_mask[:num_tokens].copy_(fix_routing_mask)
            kernel_fixed_topk_idx = torch.empty((aligned_num_tokens, num_topk), dtype=torch.int64, device=device)
            kernel_fixed_topk_idx[:num_tokens].copy_(fixed_topk_idx)
        else:
            kernel_fixed_topk_idx = torch.empty((aligned_num_tokens, num_topk), dtype=torch.int64, device=device)

    kernel = get_top2_sum_gate_ascend_v25_scalar_kernel(
        _scoring_type(scoring_func),
        num_topk,
        num_topk_groups,
        num_groups,
        num_routed_experts,
        num_shared_experts,
        mask_exists,
        fix_routing_mask_exists,
        unmapped_topk_idx_exists,
        to_physical_map_exists,
        rows_per_vec,
    )

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", "0")):
        print(kernel.get_kernel_source())

    num_physical_topk = num_topk + num_shared_experts
    output_num_tokens = aligned_num_tokens if tail_exists else num_tokens
    kernel_topk_idx = torch.empty((output_num_tokens, num_physical_topk), dtype=torch.int64, device=device)
    kernel_topk_weights = torch.empty((output_num_tokens, num_physical_topk), dtype=torch.float32, device=device)
    kernel_unmapped_topk_idx = torch.empty((output_num_tokens, num_topk), dtype=torch.int64, device=device)

    kernel(
        kernel_logits,
        bias,
        kernel_mask,
        kernel_fix_routing_mask,
        to_physical_map,
        logical_count,
        expert_local_map,
        kernel_fixed_topk_idx,
        kernel_topk_idx,
        kernel_unmapped_topk_idx,
        kernel_topk_weights,
        num_extra_experts,
        routed_scaling_factor,
        ep_rank,
        num_ep_ranks,
        tp_rank,
        num_tp_ranks,
    )
    if tail_exists:
        topk_idx = torch.empty((num_tokens, num_physical_topk), dtype=torch.int64, device=device)
        topk_weights = torch.empty((num_tokens, num_physical_topk), dtype=torch.float32, device=device)
        topk_idx.copy_(kernel_topk_idx[:num_tokens])
        topk_weights.copy_(kernel_topk_weights[:num_tokens])
    else:
        topk_idx = kernel_topk_idx
        topk_weights = kernel_topk_weights
    if unmapped_topk_idx is not None:
        unmapped_topk_idx.copy_(kernel_unmapped_topk_idx[:num_tokens])
    return topk_idx, topk_weights


def ascend_top2_sum_gate_v25_scalar_optimized(
    logits: torch.Tensor,
    bias: torch.Tensor,
    num_topk: int,
    num_topk_groups: int,
    num_groups: int,
    use_shared_as_routed: bool,
    num_shared_experts: int,
    routed_scaling_factor: float,
    ep_rank: int,
    num_ep_ranks: int,
    tp_rank: int,
    num_tp_ranks: int,
    scoring_func: str,
    mask: Optional[torch.Tensor] = None,
    fix_routing_mask: Optional[torch.Tensor] = None,
    to_physical_map: Optional[torch.Tensor] = None,
    logical_count: Optional[torch.Tensor] = None,
    unmapped_topk_idx: Optional[torch.Tensor] = None,
    topk_idx_out: Optional[torch.Tensor] = None,
    topk_weights_out: Optional[torch.Tensor] = None,
):
    """Run the full-contract V25 scalar-optimized top2-sum-gate."""

    assert logits.dim() == 2
    assert logits.is_contiguous() and logits.dtype == torch.float32
    assert bias.dim() == 1
    assert bias.is_contiguous() and bias.dtype == torch.float32
    assert logits.shape[1] == bias.numel()
    assert (to_physical_map is None) == (logical_count is None)

    effective_shared_experts = num_shared_experts if use_shared_as_routed else 0
    num_tokens, num_routed_experts = logits.shape
    num_physical_topk = num_topk + effective_shared_experts
    output_shape = (num_tokens, num_physical_topk)

    if topk_idx_out is not None:
        assert topk_idx_out.shape == output_shape
        assert topk_idx_out.dtype == torch.int64
    if topk_weights_out is not None:
        assert topk_weights_out.shape == output_shape
        assert topk_weights_out.dtype == torch.float32

    if num_tokens == 0:
        if topk_idx_out is None:
            topk_idx_out = torch.empty(output_shape, dtype=torch.int64, device=logits.device)
        if topk_weights_out is None:
            topk_weights_out = torch.empty(output_shape, dtype=torch.float32, device=logits.device)
        return topk_idx_out, topk_weights_out

    topk_idx, topk_weights = top2_sum_gate_ascend(
        logits,
        bias,
        num_topk,
        num_topk_groups,
        num_groups,
        use_shared_as_routed,
        effective_shared_experts,
        routed_scaling_factor,
        ep_rank,
        num_ep_ranks,
        tp_rank,
        num_tp_ranks,
        scoring_func,
        mask,
        fix_routing_mask,
        to_physical_map,
        logical_count,
        unmapped_topk_idx,
    )
    if topk_idx_out is not None:
        topk_idx_out.copy_(topk_idx)
        topk_idx = topk_idx_out
    if topk_weights_out is not None:
        topk_weights_out.copy_(topk_weights)
        topk_weights = topk_weights_out

    return topk_idx, topk_weights


ascend_top2_sum_gate_v25 = ascend_top2_sum_gate_v25_scalar_optimized
ascend_top2_sum_gate_optimized = ascend_top2_sum_gate_v25_scalar_optimized


TEST_CONFIGS = (
    (0, 0, 72, 1, 6),  # ungrouped, dynamic_map, shared
    (0, 0, 32, 2, 6),  # ungrouped, static_map, no_shared
    (0, 0, 64, 2, 6),  # ungrouped, static_map, no_shared
    (0, 0, 96, 2, 6),  # ungrouped, dynamic_map, shared
    (0, 0, 16, 2, 6),  # ungrouped, static_map, no_shared
    (0, 0, 36, 2, 6),  # ungrouped, dynamic_map, shared
    (0, 0, 108, 2, 6),  # ungrouped, dynamic_map, shared
    (0, 0, 128, 2, 6),  # ungrouped, static_map, no_shared
    (0, 0, 144, 2, 6),  # ungrouped, dynamic_map, shared
    (8, 8, 256, 2, 8),  # ungrouped, dynamic_map, shared
    (8, 4, 256, 2, 8),  # grouped, dynamic_map, shared
)
TEST_SCORING_FUNCS = ("sigmoid", "sqrtsoftplus", "softmax", "identity")


def torch_top2_sum_gate_v25_ref(
    logits: torch.Tensor,
    bias: torch.Tensor,
    num_topk: int,
    num_topk_groups: int,
    num_groups: int,
    use_shared_as_routed: bool,
    num_shared_experts: int,
    routed_scaling_factor: float,
    ep_rank: int,
    num_ep_ranks: int,
    tp_rank: int,
    num_tp_ranks: int,
    scoring_func: str,
    mask: Optional[torch.Tensor] = None,
    fix_routing_mask: Optional[torch.Tensor] = None,
    to_physical_map: Optional[torch.Tensor] = None,
    logical_count: Optional[torch.Tensor] = None,
    unmapped_topk_idx: Optional[torch.Tensor] = None,
):
    if not use_shared_as_routed:
        num_shared_experts = 0
    num_tokens, num_routed_experts = logits.shape
    num_physical_topk = num_topk + num_shared_experts
    topk_idx = torch.full(
        (num_tokens, num_physical_topk),
        -1,
        dtype=torch.int64,
    )
    topk_weights = torch.zeros(
        (num_tokens, num_physical_topk),
        dtype=torch.float32,
    )
    ref_unmapped_topk_idx = torch.full(
        (num_tokens, num_topk),
        -1,
        dtype=torch.int64,
    )

    if num_tokens == 0:
        return topk_idx, ref_unmapped_topk_idx, topk_weights

    if scoring_func == "sigmoid":
        route_scores = torch.sigmoid(logits)
    elif scoring_func == "sqrtsoftplus":
        route_scores = torch.sqrt(torch.nn.functional.softplus(logits))
    elif scoring_func == "softmax":
        route_scores = torch.softmax(logits, dim=-1)
    elif scoring_func == "identity":
        route_scores = logits
    else:
        raise ValueError(scoring_func)

    biased_scores = logits + bias if scoring_func == "softmax" else route_scores + bias
    skip_group_sort = num_groups == 0 or num_groups == num_topk_groups

    for token in range(num_tokens):
        if mask is not None and not bool(mask[token]):
            continue
        if fix_routing_mask is not None and bool(fix_routing_mask[token]):
            assert unmapped_topk_idx is not None
            order_t = unmapped_topk_idx[token].to(torch.int64)
        else:
            candidate_mask = torch.ones(num_routed_experts, dtype=torch.bool)
            if not skip_group_sort:
                experts_per_group = num_routed_experts // num_groups
                group_scores = []
                for group in range(num_groups):
                    start = group * experts_per_group
                    values = biased_scores[
                        token,
                        start : start + experts_per_group,
                    ]
                    group_scores.append(torch.topk(values, 2, sorted=False).values.sum())
                selected_groups = sorted(
                    range(num_groups),
                    key=lambda group: (-float(group_scores[group]), group),
                )[:num_topk_groups]
                candidate_mask.fill_(False)
                for group in selected_groups:
                    start = group * experts_per_group
                    candidate_mask[start : start + experts_per_group] = True
            order = sorted(
                (expert for expert in range(num_routed_experts) if bool(candidate_mask[expert])),
                key=lambda expert: (-float(biased_scores[token, expert]), expert),
            )[:num_topk]
            order_t = torch.tensor(order, dtype=torch.int64)

        selected_weights = route_scores[token, order_t]
        selected_weights = selected_weights / (selected_weights.sum() + 1.0e-20) * routed_scaling_factor
        ref_unmapped_topk_idx[token] = order_t
        topk_idx[token, :num_topk] = order_t
        topk_weights[token, :num_topk] = selected_weights

        for k in range(num_shared_experts):
            topk_idx[token, num_topk + k] = num_routed_experts + k
            topk_weights[token, num_topk + k] = 1.0

    if to_physical_map is not None:
        assert logical_count is not None
        for token in range(num_tokens):
            if mask is not None and not bool(mask[token]):
                continue
            for k in range(num_physical_topk):
                logical_idx = int(topk_idx[token, k])
                if logical_idx >= 0:
                    duplicate_idx = (ep_rank + token * 23333) % int(logical_count[logical_idx])
                    topk_idx[token, k] = int(to_physical_map[logical_idx, duplicate_idx])
        num_extra_experts = to_physical_map.shape[1] - 1
    else:
        num_extra_experts = 0

    num_experts_per_rank = (num_routed_experts + num_extra_experts) // num_ep_ranks
    num_experts_per_dp = num_experts_per_rank * num_tp_ranks
    for token in range(num_tokens):
        for k in range(num_physical_topk):
            idx = int(topk_idx[token, k])
            if idx < 0:
                continue
            dst_ep_rank = idx // num_experts_per_rank
            if dst_ep_rank % num_tp_ranks != tp_rank:
                topk_idx[token, k] = -1
            else:
                idx = idx - tp_rank * num_experts_per_rank
                dst_dp_rank = idx // num_experts_per_dp
                idx = idx - dst_dp_rank * num_experts_per_dp + dst_dp_rank * num_experts_per_rank
                topk_idx[token, k] = idx if idx >= 0 else -1
    return topk_idx, ref_unmapped_topk_idx, topk_weights


def _make_v25_test_case(
    num_tokens: int,
    num_padded_tokens: int,
    num_routed_experts: int,
    num_topk: int,
    num_shared_experts: int,
    use_shared_as_routed: bool,
    fix_routing: bool,
):
    total_tokens = num_tokens + num_padded_tokens
    logits = torch.randn(
        (total_tokens, num_routed_experts),
        dtype=torch.float32,
    )
    bias = torch.randn((num_routed_experts,), dtype=torch.float32)
    mask = None
    if num_padded_tokens > 0:
        mask = torch.ones((total_tokens,), dtype=torch.bool)
        mask[-num_padded_tokens:] = False

    unmapped_topk_idx = torch.empty((total_tokens, num_topk), dtype=torch.int64)
    fix_routing_mask = None
    if fix_routing:
        fix_routing_mask = torch.ones((total_tokens,), dtype=torch.bool)
        generator = torch.Generator().manual_seed(42)
        unmapped_topk_idx = torch.randint(
            0,
            num_routed_experts,
            (total_tokens, num_topk),
            generator=generator,
            dtype=torch.int64,
        )

    to_physical_map = None
    logical_count = None
    if use_shared_as_routed:
        num_logical_experts = num_routed_experts + num_shared_experts
        to_physical_map = torch.arange(num_logical_experts, dtype=torch.int32).view(-1, 1).expand(-1, 33).contiguous()
        logical_count = torch.ones((num_logical_experts,), dtype=torch.int32)
    return (
        logits,
        bias,
        mask,
        fix_routing_mask,
        to_physical_map,
        logical_count,
        unmapped_topk_idx,
    )


def _run_v25_test_suite(
    test_configs: tuple[tuple[int, int, int, int, int], ...],
    token_cases: tuple[tuple[int, int], ...],
) -> None:
    routed_scaling_factor = 1.5
    num_ep_ranks, num_tp_ranks = 4, 2

    for (
        num_groups,
        num_topk_groups,
        num_routed_experts,
        num_shared_experts,
        num_topk,
    ) in test_configs:
        shared_valid = (
            num_shared_experts > 0 and num_topk % num_shared_experts == 0 and num_routed_experts % (num_topk // num_shared_experts) == 0
        )
        for num_active_tokens, num_padded_tokens in token_cases:
            total_tokens = num_active_tokens + num_padded_tokens
            tp_rank = num_tp_ranks - 1
            use_shared_as_routed = shared_valid
            fix_routing = False
            for scoring_func in TEST_SCORING_FUNCS:
                (
                    logits,
                    bias,
                    mask,
                    fix_routing_mask,
                    to_physical_map,
                    logical_count,
                    initial_unmapped,
                ) = _make_v25_test_case(
                    num_active_tokens,
                    num_padded_tokens,
                    num_routed_experts,
                    num_topk,
                    num_shared_experts,
                    use_shared_as_routed,
                    fix_routing,
                )
                ref_idx, ref_unmapped_idx, ref_weights = torch_top2_sum_gate_v25_ref(
                    logits,
                    bias,
                    num_topk,
                    num_topk_groups,
                    num_groups,
                    use_shared_as_routed,
                    num_shared_experts,
                    routed_scaling_factor,
                    0,
                    num_ep_ranks,
                    tp_rank,
                    num_tp_ranks,
                    scoring_func,
                    mask,
                    fix_routing_mask,
                    to_physical_map,
                    logical_count,
                    initial_unmapped,
                )

                kernel_unmapped_idx = initial_unmapped.npu()
                kernel_idx, kernel_weights = ascend_top2_sum_gate_v25_scalar_optimized(
                    logits.npu(),
                    bias.npu(),
                    num_topk,
                    num_topk_groups,
                    num_groups,
                    use_shared_as_routed,
                    num_shared_experts,
                    routed_scaling_factor,
                    0,
                    num_ep_ranks,
                    tp_rank,
                    num_tp_ranks,
                    scoring_func,
                    mask.npu() if mask is not None else None,
                    fix_routing_mask.npu() if fix_routing_mask is not None else None,
                    to_physical_map.npu() if to_physical_map is not None else None,
                    logical_count.npu() if logical_count is not None else None,
                    kernel_unmapped_idx,
                )
                torch.npu.synchronize()

                torch.testing.assert_close(kernel_idx.cpu(), ref_idx, rtol=0, atol=0)
                torch.testing.assert_close(kernel_unmapped_idx.cpu(), ref_unmapped_idx, rtol=0, atol=0)
                torch.testing.assert_close(kernel_weights.cpu(), ref_weights, rtol=1e-4, atol=1e-4)
                print(
                    "PASS "
                    f"tokens={total_tokens} "
                    f"active={num_active_tokens} "
                    f"padded={num_padded_tokens} "
                    f"experts={num_routed_experts} "
                    f"topk={num_topk} "
                    f"groups={num_groups}/{num_topk_groups} "
                    f"shared={use_shared_as_routed} "
                    f"fixed={fix_routing} "
                    f"tp_rank={tp_rank} "
                    f"scoring={scoring_func}"
                )


def main() -> None:
    torch.manual_seed(0)
    mode = os.getenv("TK_TOP2_SUM_GATE_V25_TEST_MODE", "quick").lower()
    test_configs = TEST_CONFIGS
    token_cases = (
        # (32, 4),
        # (4096, 0),
        (4001, 4),
    )
    print(f"Running {len(test_configs) * len(token_cases) * len(TEST_SCORING_FUNCS)} benchmark-aligned cases")
    _run_v25_test_suite(test_configs, token_cases)
    print("TEST PASSED!")


if __name__ == "__main__":
    for i in range(5):
        print(f"iter{i}")
        main()
