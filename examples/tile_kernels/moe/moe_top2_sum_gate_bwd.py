# ruff: noqa: SIM102, SIM117

import os
from typing import Optional

import torch
import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


SCORING_SIGMOID = 0
SCORING_SQRTSOFTPLUS = 1
SCORING_SOFTMAX = 2
SCORING_IDENTITY = 3
_PHYSICAL_LOCAL_CACHE = {}
_DUMMY_MAP_CACHE = {}


def _backward_rows_per_vec(num_routed_experts: int) -> int:
    if num_routed_experts <= 32:
        return 32
    if num_routed_experts <= 64:
        return 16
    if num_routed_experts <= 128:
        return 8
    return 4


def _get_physical_local_map(
    num_physical_experts: int, num_ep_ranks: int, tp_rank: int, num_tp_ranks: int, device: torch.device
) -> torch.Tensor:
    aligned_num_physical_experts = (num_physical_experts + 7) // 8 * 8
    key = (str(device), num_physical_experts, num_ep_ranks, tp_rank, num_tp_ranks)
    if key not in _PHYSICAL_LOCAL_CACHE:
        physical_idx = torch.arange(aligned_num_physical_experts, dtype=torch.int32)
        num_experts_per_rank = num_physical_experts // num_ep_ranks
        dst_ep_rank = physical_idx // num_experts_per_rank
        local_map = torch.where(dst_ep_rank % num_tp_ranks == tp_rank, physical_idx, -1)
        local_map[num_physical_experts:] = -1
        _PHYSICAL_LOCAL_CACHE[key] = local_map.to(device=device).contiguous()
    return _PHYSICAL_LOCAL_CACHE[key]


def _get_dummy_maps(num_logical_experts: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), num_logical_experts)
    if key not in _DUMMY_MAP_CACHE:
        to_physical_map = torch.arange(num_logical_experts, dtype=torch.int32).reshape(num_logical_experts, 1)
        logical_count = torch.ones((num_logical_experts,), dtype=torch.int32)
        _DUMMY_MAP_CACHE[key] = (to_physical_map.to(device=device), logical_count.to(device=device))
    return _DUMMY_MAP_CACHE[key]


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


@tilelang.jit(out_idx=[7, 8], pass_configs=pass_configs)
def get_top2_sum_gate_ascend_backward_kernel(
    scoring_type: int,
    num_topk: int,
    num_physical_topk: int,
    num_routed_experts: int,
    num_logical_experts: int,
    num_duplicate_experts: int,
    use_mask: bool,
    use_physical_map: bool,
    rows_per_vec: int,
    num_tokens: int,
):
    vec_num = 2
    tokens_per_block = rows_per_vec * vec_num
    num_token_blocks = T.ceildiv(num_tokens, tokens_per_block)
    num_cores = 24
    stages = 2
    num_iters = T.ceildiv(num_token_blocks, num_cores)
    aligned_num_experts = ((num_routed_experts + 31) // 32) * 32
    aligned_copy_experts = ((num_routed_experts + 7) // 8) * 8
    aligned_topk = ((num_topk + 7) // 8) * 8
    aligned_physical_topk = ((num_physical_topk + 7) // 8) * 8
    num_physical_experts = num_routed_experts + num_duplicate_experts - 1
    aligned_num_physical_experts = ((num_physical_experts + 7) // 8) * 8

    @T.prim_func
    def top2_sum_gate_ascend_backward_kernel(
        logits: T.Tensor((num_tokens, aligned_copy_experts), "float"),
        mask: T.Tensor((num_tokens,), "int32"),
        unmapped_topk_idx: T.Tensor((num_tokens, aligned_topk), "int64"),
        d_topk_weights: T.Tensor((num_tokens, aligned_physical_topk), "float"),
        to_physical_map: T.Tensor((num_logical_experts, num_duplicate_experts), "int32"),
        logical_count: T.Tensor((num_logical_experts,), "int32"),
        physical_local_map: T.Tensor((aligned_num_physical_experts,), "int32"),
        dlogits: T.Tensor((num_tokens, aligned_copy_experts), "float"),
        dbias: T.Tensor((aligned_copy_experts,), "float"),
        routed_scaling_factor: T.float32,
        ep_rank: T.int32,
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                logits_ub = T.alloc_ub((stages, rows_per_vec, aligned_num_experts), "float")
                route_scores_ub = T.alloc_ub((rows_per_vec, aligned_num_experts), "float")
                grad_scores_ub = T.alloc_ub((rows_per_vec, aligned_num_experts), "float")
                dlogits_out_ub = T.alloc_ub((stages, rows_per_vec, aligned_num_experts), "float")
                work_ub = T.alloc_ub((rows_per_vec, aligned_num_experts), "float")
                reduce_ub = T.alloc_ub((rows_per_vec, 1), "float")
                reduce_broadcast_ub = T.alloc_ub((rows_per_vec, aligned_num_experts), "float")
                selected_idx_ub = T.alloc_ub((rows_per_vec, aligned_topk), "int32")
                unmapped_idx_i64_ub = T.alloc_ub((stages, rows_per_vec, aligned_topk), "int64")
                physical_idx_ub = T.alloc_ub((rows_per_vec, aligned_topk), "int32")
                duplicate_count_ub = T.alloc_ub((rows_per_vec, aligned_topk), "int32")
                selected_score_ub = T.alloc_ub((rows_per_vec, aligned_topk), "float")
                grad_topk_ub = T.alloc_ub((rows_per_vec, aligned_topk), "float")
                d_topk_weights_ub = T.alloc_ub((stages, rows_per_vec, aligned_physical_topk), "float")
                grad_product_ub = T.alloc_ub((rows_per_vec, aligned_topk), "float")
                selected_sum_ub = T.alloc_ub((rows_per_vec, 1), "float")
                grad_dot_ub = T.alloc_ub((rows_per_vec, 1), "float")
                physical_local_ub = T.alloc_ub((aligned_num_physical_experts,), "int32")
                dbias_out_ub = T.alloc_ub((aligned_num_experts,), "float")

                score_val = T.alloc_var("float", init=0.0)
                grad_norm = T.alloc_var("float", init=0.0)
                duplicate_idx = T.alloc_var("int32", init=0)

                T.copy(physical_local_map, physical_local_ub)
                T.set_flag("mte2", "v", 2)
                T.wait_flag("mte2", "v", 2)
                for stage in T.serial(stages):
                    T.set_flag("mte3", "mte2", stage)

                if cid < num_token_blocks:
                    first_token_start = cid * tokens_per_block + vid * rows_per_vec
                    T.wait_flag("mte3", "mte2", 0)
                    for row in T.serial(rows_per_vec):
                        T.copy(logits[first_token_start + row, :aligned_copy_experts], logits_ub[0, row, :], pad_value=-T.infinity("float"))
                        T.copy(unmapped_topk_idx[first_token_start + row, :aligned_topk], unmapped_idx_i64_ub[0, row, :])
                        T.copy(d_topk_weights[first_token_start + row, :aligned_physical_topk], d_topk_weights_ub[0, row, :])
                    T.set_flag("mte2", "v", 0)

                for i in T.serial(num_iters):
                    cur = i % stages
                    nxt = (i + 1) % stages
                    block_id = cid + i * num_cores
                    token_start = block_id * tokens_per_block + vid * rows_per_vec
                    if block_id < num_token_blocks:
                        next_block_id = cid + (i + 1) * num_cores
                        if next_block_id < num_token_blocks:
                            next_token_start = next_block_id * tokens_per_block + vid * rows_per_vec
                            T.wait_flag("mte3", "mte2", nxt)
                            for row in T.serial(rows_per_vec):
                                T.copy(
                                    logits[next_token_start + row, :aligned_copy_experts],
                                    logits_ub[nxt, row, :],
                                    pad_value=-T.infinity("float"),
                                )
                                T.copy(unmapped_topk_idx[next_token_start + row, :aligned_topk], unmapped_idx_i64_ub[nxt, row, :])
                                T.copy(d_topk_weights[next_token_start + row, :aligned_physical_topk], d_topk_weights_ub[nxt, row, :])
                            T.set_flag("mte2", "v", nxt)

                        T.wait_flag("mte2", "v", cur)
                        T.tile.fill(grad_scores_ub, 0.0)
                        T.tile.fill(dlogits_out_ub[cur, :, :], 0.0)
                        T.tile.fill(selected_score_ub, 0.0)
                        T.tile.fill(grad_topk_ub, 0.0)
                        T.tile.cast(selected_idx_ub, unmapped_idx_i64_ub[cur, :, :], "CAST_NONE", rows_per_vec * aligned_topk)

                        if scoring_type == SCORING_SOFTMAX:
                            T.reduce_max(logits_ub[cur, :, :], reduce_ub, dim=-1, real_shape=[rows_per_vec, aligned_num_experts])
                            T.tile.broadcast(reduce_broadcast_ub, reduce_ub)
                            T.tile.sub(work_ub, logits_ub[cur, :, :], reduce_broadcast_ub)
                            T.tile.exp(route_scores_ub, work_ub)
                            T.reduce_sum(route_scores_ub, reduce_ub, dim=-1, real_shape=[rows_per_vec, aligned_num_experts])
                            T.tile.broadcast(reduce_broadcast_ub, reduce_ub)
                            T.tile.div(route_scores_ub, route_scores_ub, reduce_broadcast_ub)
                        elif scoring_type == SCORING_SIGMOID:
                            T.tile.sigmoid(route_scores_ub, logits_ub[cur, :, :])
                        elif scoring_type == SCORING_IDENTITY:
                            T.tile.add(route_scores_ub, logits_ub[cur, :, :], 0.0)

                        for row in T.serial(rows_per_vec):
                            if not use_mask or mask[token_start + row] != 0:
                                for k in T.unroll(num_topk):
                                    if selected_idx_ub[row, k] >= 0 and selected_idx_ub[row, k] < num_routed_experts:
                                        if scoring_type == SCORING_SQRTSOFTPLUS:
                                            score_val = logits_ub[cur, row, selected_idx_ub[row, k]]
                                            if score_val <= 20.0:
                                                if score_val < 0.0:
                                                    score_val = 0.0
                                                score_val = score_val + 0.69314718
                                            selected_score_ub[row, k] = score_val
                                        else:
                                            selected_score_ub[row, k] = route_scores_ub[row, selected_idx_ub[row, k]]
                                        physical_idx_ub[row, k] = selected_idx_ub[row, k]
                                        if use_physical_map:
                                            duplicate_count_ub[row, k] = logical_count[selected_idx_ub[row, k]]
                                            duplicate_idx = (ep_rank + (token_start + row) * 23333) % duplicate_count_ub[row, k]
                                            physical_idx_ub[row, k] = to_physical_map[selected_idx_ub[row, k], duplicate_idx]
                                        if physical_idx_ub[row, k] >= 0 and physical_idx_ub[row, k] < num_physical_experts:
                                            if physical_local_ub[physical_idx_ub[row, k]] >= 0:
                                                grad_topk_ub[row, k] = d_topk_weights_ub[cur, row, k]

                        T.tile.mul(grad_product_ub, grad_topk_ub, selected_score_ub)
                        T.reduce_sum(selected_score_ub, selected_sum_ub, dim=-1, real_shape=[rows_per_vec, aligned_topk])
                        T.reduce_sum(grad_product_ub, grad_dot_ub, dim=-1, real_shape=[rows_per_vec, aligned_topk])
                        T.tile.add(selected_sum_ub, selected_sum_ub, 1.0e-20)
                        for row in T.serial(rows_per_vec):
                            if not use_mask or mask[token_start + row] != 0:
                                for k in T.unroll(num_topk):
                                    if selected_idx_ub[row, k] >= 0 and selected_idx_ub[row, k] < num_routed_experts:
                                        grad_norm = (
                                            routed_scaling_factor
                                            / selected_sum_ub[row, 0]
                                            * (grad_topk_ub[row, k] - grad_dot_ub[row, 0] / selected_sum_ub[row, 0])
                                        )
                                        grad_scores_ub[row, selected_idx_ub[row, k]] = (
                                            grad_scores_ub[row, selected_idx_ub[row, k]] + grad_norm
                                        )

                        if scoring_type == SCORING_SOFTMAX:
                            T.tile.mul(work_ub, grad_scores_ub, route_scores_ub)
                            T.reduce_sum(work_ub, reduce_ub, dim=-1, real_shape=[rows_per_vec, aligned_num_experts])
                            T.tile.broadcast(reduce_broadcast_ub, reduce_ub)
                            T.tile.sub(work_ub, grad_scores_ub, reduce_broadcast_ub)
                            T.tile.mul(dlogits_out_ub[cur, :, :], route_scores_ub, work_ub)
                        elif scoring_type == SCORING_SIGMOID:
                            T.tile.add(work_ub, route_scores_ub, -1.0)
                            T.tile.mul(work_ub, route_scores_ub, work_ub)
                            T.tile.mul(dlogits_out_ub[cur, :, :], grad_scores_ub, work_ub)
                            T.tile.mul(dlogits_out_ub[cur, :, :], dlogits_out_ub[cur, :, :], -1.0)
                        elif scoring_type == SCORING_IDENTITY:
                            T.tile.add(dlogits_out_ub[cur, :, :], grad_scores_ub, 0.0)
                        else:
                            for row in T.serial(rows_per_vec):
                                if not use_mask or mask[token_start + row] != 0:
                                    for k in T.unroll(num_topk):
                                        if selected_idx_ub[row, k] >= 0 and selected_idx_ub[row, k] < num_routed_experts:
                                            if logits_ub[cur, row, selected_idx_ub[row, k]] >= 0.0:
                                                dlogits_out_ub[cur, row, selected_idx_ub[row, k]] = grad_scores_ub[
                                                    row, selected_idx_ub[row, k]
                                                ]

                        T.set_flag("v", "mte3", cur)
                        T.wait_flag("v", "mte3", cur)
                        for row in T.serial(rows_per_vec):
                            T.copy(dlogits_out_ub[cur, row, :aligned_copy_experts], dlogits[token_start + row, :aligned_copy_experts])
                        T.pipe_barrier("mte3")
                        T.set_flag("mte3", "mte2", cur)

                for stage in T.serial(stages):
                    T.wait_flag("mte3", "mte2", stage)
                if cid == 0:
                    T.tile.fill(dbias_out_ub, 0.0)
                    T.copy(dbias_out_ub[:aligned_copy_experts], dbias[:aligned_copy_experts])

    return top2_sum_gate_ascend_backward_kernel


def top2_sum_gate_ascend_backward(
    logits: torch.Tensor,
    bias: torch.Tensor,
    unmapped_topk_idx: torch.Tensor,
    d_topk_weights: torch.Tensor,
    num_topk: int,
    use_shared_as_routed: bool,
    num_shared_experts: int,
    routed_scaling_factor: float,
    ep_rank: int,
    num_ep_ranks: int,
    tp_rank: int,
    num_tp_ranks: int,
    scoring_func: str,
    mask: Optional[torch.Tensor] = None,
    to_physical_map: Optional[torch.Tensor] = None,
    logical_count: Optional[torch.Tensor] = None,
):
    assert logits.dim() == 2 and logits.is_contiguous() and logits.dtype == torch.float32
    assert bias.dim() == 1 and bias.is_contiguous() and bias.dtype == torch.float32
    assert unmapped_topk_idx.dim() == 2 and unmapped_topk_idx.dtype == torch.int64
    assert d_topk_weights.dim() == 2 and d_topk_weights.dtype == torch.float32
    assert logits.shape[1] == bias.numel()
    assert unmapped_topk_idx.shape == (logits.shape[0], num_topk)

    num_tokens, num_routed_experts = logits.shape
    if not use_shared_as_routed:
        num_shared_experts = 0
    num_physical_topk = num_topk + num_shared_experts
    assert d_topk_weights.shape == (num_tokens, num_physical_topk)
    assert (to_physical_map is None) == (logical_count is None)

    device = logits.device
    if mask is None:
        mask = torch.ones((num_tokens,), dtype=torch.int32, device=device)
        use_mask = False
    else:
        mask = mask.to(torch.int32)
        use_mask = True

    num_logical_experts = num_routed_experts + num_shared_experts
    if to_physical_map is None:
        to_physical_map, logical_count = _get_dummy_maps(num_logical_experts, device)
        num_duplicate_experts = 1
        num_extra_experts = 0
        use_physical_map = False
    else:
        assert logical_count is not None
        assert to_physical_map.dim() == 2 and to_physical_map.dtype == torch.int32
        assert logical_count.dim() == 1 and logical_count.dtype == torch.int32
        assert to_physical_map.shape[0] == num_logical_experts
        assert logical_count.shape[0] == num_logical_experts
        num_duplicate_experts = to_physical_map.shape[1]
        num_extra_experts = num_duplicate_experts - 1
        use_physical_map = True

    rows_per_vec = _backward_rows_per_vec(num_routed_experts)
    tokens_per_block = rows_per_vec * 2
    aligned_num_tokens = (num_tokens + tokens_per_block - 1) // tokens_per_block * tokens_per_block
    aligned_copy_experts = (num_routed_experts + 7) // 8 * 8
    aligned_topk = (num_topk + 7) // 8 * 8
    aligned_physical_topk = (num_physical_topk + 7) // 8 * 8
    kernel_logits = logits
    kernel_mask = mask
    if aligned_num_tokens != num_tokens or aligned_copy_experts != num_routed_experts:
        use_mask = True
        kernel_logits = torch.full((aligned_num_tokens, aligned_copy_experts), float("-inf"), dtype=logits.dtype, device=device)
        kernel_logits[:num_tokens, :num_routed_experts].copy_(logits)
        kernel_mask = torch.zeros((aligned_num_tokens,), dtype=torch.int32, device=device)
        kernel_mask[:num_tokens].copy_(mask)
    kernel_unmapped_topk_idx = unmapped_topk_idx
    if aligned_num_tokens != num_tokens or aligned_topk != num_topk:
        kernel_unmapped_topk_idx = torch.empty((aligned_num_tokens, aligned_topk), dtype=torch.int64, device=device)
        kernel_unmapped_topk_idx[:num_tokens, :num_topk].copy_(unmapped_topk_idx)
    kernel_d_topk_weights = d_topk_weights
    if aligned_num_tokens != num_tokens or aligned_physical_topk != num_physical_topk:
        kernel_d_topk_weights = torch.empty((aligned_num_tokens, aligned_physical_topk), dtype=d_topk_weights.dtype, device=device)
        kernel_d_topk_weights[:num_tokens, :num_physical_topk].copy_(d_topk_weights)
    physical_local_map = _get_physical_local_map(num_routed_experts + num_extra_experts, num_ep_ranks, tp_rank, num_tp_ranks, device)

    kernel = get_top2_sum_gate_ascend_backward_kernel(
        _scoring_type(scoring_func),
        num_topk,
        num_physical_topk,
        num_routed_experts,
        num_logical_experts,
        num_duplicate_experts,
        use_mask,
        use_physical_map,
        rows_per_vec,
        aligned_num_tokens,
    )

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", "0")):
        print(kernel.get_kernel_source())

    dlogits, dbias = kernel(
        kernel_logits,
        kernel_mask,
        kernel_unmapped_topk_idx,
        kernel_d_topk_weights,
        to_physical_map,
        logical_count,
        physical_local_map,
        routed_scaling_factor,
        ep_rank,
    )
    return dlogits[:num_tokens, :num_routed_experts], dbias[:num_routed_experts]


def torch_top2_sum_gate_backward_ref(
    logits: torch.Tensor,
    bias: torch.Tensor,
    unmapped_topk_idx: torch.Tensor,
    d_topk_weights: torch.Tensor,
    num_topk: int,
    use_shared_as_routed: bool,
    num_shared_experts: int,
    routed_scaling_factor: float,
    ep_rank: int,
    num_ep_ranks: int,
    tp_rank: int,
    num_tp_ranks: int,
    scoring_func: str,
    mask: Optional[torch.Tensor] = None,
    to_physical_map: Optional[torch.Tensor] = None,
    logical_count: Optional[torch.Tensor] = None,
):
    if not use_shared_as_routed:
        num_shared_experts = 0

    num_tokens, num_routed_experts = logits.shape
    num_logical_experts = num_routed_experts + num_shared_experts
    dlogits = torch.zeros_like(logits)
    dbias = torch.zeros_like(bias)

    if to_physical_map is None:
        to_physical_map = torch.arange(num_logical_experts, dtype=torch.int32).reshape(num_logical_experts, 1)
        logical_count = torch.ones((num_logical_experts,), dtype=torch.int32)
        num_extra_experts = 0
        use_physical_map = False
    else:
        assert logical_count is not None
        num_extra_experts = to_physical_map.shape[1] - 1
        use_physical_map = True

    if scoring_func == "sigmoid":
        route_scores = torch.sigmoid(logits)
    elif scoring_func == "softmax":
        route_scores = torch.softmax(logits, dim=-1)
    elif scoring_func == "identity":
        route_scores = logits
    elif scoring_func == "sqrtsoftplus":
        route_scores = logits.clone()
        route_scores = torch.where(route_scores < 0, torch.zeros_like(route_scores), route_scores)
        route_scores = route_scores + 0.69314718
    else:
        raise ValueError(scoring_func)

    for token in range(num_tokens):
        if mask is not None and not bool(mask[token]):
            continue

        grad_scores = torch.zeros((num_routed_experts,), dtype=torch.float32)
        idx = unmapped_topk_idx[token]
        valid = idx >= 0
        if not bool(valid.any()):
            continue

        idx_valid = idx[valid]
        selected_scores = route_scores[token, idx_valid]
        grad_topk = d_topk_weights[token, :num_topk][valid]
        for local_k, logical_idx in enumerate(idx_valid.tolist()):
            physical_idx = int(logical_idx)
            if use_physical_map:
                duplicate_count = int(logical_count[physical_idx])
                duplicate_idx = (ep_rank + token * 23333) % duplicate_count
                physical_idx = int(to_physical_map[physical_idx, duplicate_idx])

            num_experts_per_rank = (num_routed_experts + num_extra_experts) // num_ep_ranks
            num_experts_per_dp = num_experts_per_rank * num_tp_ranks
            dst_ep_rank = physical_idx // num_experts_per_rank
            route_is_local = True
            if dst_ep_rank % num_tp_ranks != tp_rank:
                route_is_local = False
            else:
                physical_idx = physical_idx - tp_rank * num_experts_per_rank
                dst_dp_rank = physical_idx // num_experts_per_dp
                physical_idx = physical_idx - dst_dp_rank * num_experts_per_dp + dst_dp_rank * num_experts_per_rank
                if physical_idx < 0:
                    route_is_local = False
            if not route_is_local:
                grad_topk[local_k] = 0.0

        selected_sum = selected_scores.sum() + 1.0e-20
        grad_dot = (grad_topk * selected_scores).sum()
        grad_scores[idx_valid] = routed_scaling_factor / selected_sum * (grad_topk - grad_dot / selected_sum)

        if scoring_func == "sigmoid":
            dlogits[token] = grad_scores * route_scores[token] * (1.0 - route_scores[token])
        elif scoring_func == "softmax":
            softmax_dot = (grad_scores * route_scores[token]).sum()
            dlogits[token] = route_scores[token] * (grad_scores - softmax_dot)
        elif scoring_func == "identity":
            dlogits[token] = grad_scores
        elif scoring_func == "sqrtsoftplus":
            dlogits[token] = torch.where(logits[token] < 0, torch.zeros_like(grad_scores), grad_scores)

    return dlogits, dbias


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


def main():
    torch.manual_seed(0)
    num_active_tokens, num_padded_tokens = 4001, 4
    num_tokens = num_active_tokens + num_padded_tokens
    routed_scaling_factor = 1.5
    num_ep_ranks, num_tp_ranks = 4, 2
    ep_rank, tp_rank = 0, num_tp_ranks - 1

    for num_groups, num_topk_groups, num_routed_experts, num_shared_experts, num_topk in TEST_CONFIGS:
        use_shared_as_routed = num_topk % num_shared_experts == 0 and num_routed_experts % (num_topk // num_shared_experts) == 0
        for scoring_func in TEST_SCORING_FUNCS:
            logits = torch.randn((num_tokens, num_routed_experts), dtype=torch.float32)
            bias = torch.randn((num_routed_experts,), dtype=torch.float32)
            mask = torch.ones((num_tokens,), dtype=torch.bool)
            mask[-num_padded_tokens:] = False
            to_physical_map = None
            logical_count = None
            if use_shared_as_routed:
                num_logical_experts = num_routed_experts + num_shared_experts
                to_physical_map = torch.arange(num_logical_experts, dtype=torch.int32).view(-1, 1).expand(-1, 33).contiguous()
                logical_count = torch.ones((num_logical_experts,), dtype=torch.int32)

            unmapped_topk_idx_cpu = logits.topk(num_topk, dim=-1, sorted=False).indices
            unmapped_topk_idx_cpu[-num_padded_tokens:] = -1
            unmapped_topk_idx = unmapped_topk_idx_cpu.npu()
            num_physical_topk = num_topk + (num_shared_experts if use_shared_as_routed else 0)
            d_topk_weights_cpu = torch.randn((num_tokens, num_physical_topk), dtype=torch.float32)
            d_topk_weights = d_topk_weights_cpu.npu()
            ref_dlogits, ref_dbias = torch_top2_sum_gate_backward_ref(
                logits,
                bias,
                unmapped_topk_idx_cpu,
                d_topk_weights_cpu,
                num_topk,
                use_shared_as_routed,
                num_shared_experts,
                routed_scaling_factor,
                ep_rank,
                num_ep_ranks,
                tp_rank,
                num_tp_ranks,
                scoring_func,
                mask=mask,
                to_physical_map=to_physical_map,
                logical_count=logical_count,
            )
            kernel_dlogits, kernel_dbias = top2_sum_gate_ascend_backward(
                logits.npu(),
                bias.npu(),
                unmapped_topk_idx,
                d_topk_weights,
                num_topk,
                use_shared_as_routed,
                num_shared_experts,
                routed_scaling_factor,
                ep_rank,
                num_ep_ranks,
                tp_rank,
                num_tp_ranks,
                scoring_func,
                mask=mask.npu(),
                to_physical_map=None if to_physical_map is None else to_physical_map.npu(),
                logical_count=None if logical_count is None else logical_count.npu(),
            )
            torch.npu.synchronize()
            torch.testing.assert_close(kernel_dlogits.cpu(), ref_dlogits, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(kernel_dbias.cpu(), ref_dbias, rtol=0, atol=0)
            print(
                f"PASS tokens={num_tokens} experts={num_routed_experts} topk={num_topk} groups={num_groups}/{num_topk_groups} shared={use_shared_as_routed} tp_rank={tp_rank} scoring={scoring_func}"
            )

    print("TEST PASSED!")


if __name__ == "__main__":
    main()
