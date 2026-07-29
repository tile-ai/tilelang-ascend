import os
import sys
import pytest
import numpy as np
import torch
import tilelang
from tilelang import language as T


tilelang.cache.clear_cache()

# Apply pass configs to remove auto sync
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

FLOAT32 = "float32"
INDEX_TYPE = "int64"
INT32 = "int32"


@tilelang.jit(pass_configs=pass_configs)
def get_topk_sum_and_topk_group_idx_kernel(num_groups: int, num_experts_per_group: int, num_topk_groups: int, num_topk_sum: int):
    ALIGNMENT = 32
    num_experts = num_experts_per_group * num_groups
    # num_aligned_experts is not used in forward kernel, removed
    num_aligned_groups = (num_groups + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    num_aligned_epg = (num_experts_per_group + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    # Align int64 output to 8 elements for 32 byte MTE3 alignment
    num_aligned_topk_groups = (num_topk_groups + 7) // 8 * 8

    num_tokens = T.symbolic("num_tokens")

    assert num_groups <= 32, f"num_groups {num_groups} must be less than or equal to 32"

    @T.prim_func
    def topk_sum_and_topk_group_idx_kernel(
        scores: T.Tensor[(num_tokens, num_experts), FLOAT32],
        group_topk_idx: T.Tensor[(num_tokens, num_aligned_topk_groups), INDEX_TYPE],
    ):
        VEC_NUM = 2
        num_blocks = 128
        chunk_size = num_blocks * VEC_NUM
        num_batches = (num_tokens + chunk_size - 1) // chunk_size

        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            topk_group_idx_out_ub = T.alloc_ub((num_aligned_topk_groups,), INDEX_TYPE)

            # Two 1D buffers for double buffering
            group_experts_ub_ping = T.alloc_ub((num_aligned_epg,), FLOAT32)
            group_experts_ub_pong = T.alloc_ub((num_aligned_epg,), FLOAT32)

            group_scores_ub = T.alloc_ub((num_aligned_groups,), FLOAT32)

            # Split topk_res_ub buffer to break compiler's data dependency analysis between ping-pong branches
            topk_res_ub_ping = T.alloc_ub((64,), FLOAT32)
            topk_res_ub_pong = T.alloc_ub((64,), FLOAT32)

            group_topk_res_ub = T.alloc_ub(((num_topk_groups * 2 + 31) // 32 * 32,), FLOAT32)

            # Temporary efficient buffer for float32 indices extracted from group_topk_res_ub
            topk_index_f32_ub = T.alloc_ub((num_aligned_topk_groups,), FLOAT32)

            for batch_idx in T.Pipelined(num_batches):
                token_idx = (batch_idx * num_blocks + cid) * VEC_NUM + vid
                if token_idx < num_tokens:
                    # Process first group (group 0) outside the loop
                    T.copy(
                        scores[token_idx, 0:num_experts_per_group],
                        group_experts_ub_ping,
                        pad_value=-T.infinity(FLOAT32),
                    )
                    T.set_flag("MTE2", "V", 0)

                    # If group 1 exists, preload to pong
                    if num_groups > 1:
                        T.copy(
                            scores[token_idx, num_experts_per_group : 2 * num_experts_per_group],
                            group_experts_ub_pong,
                            pad_value=-T.infinity(FLOAT32),
                        )
                    # Perform topk on group 0 and assign
                    T.wait_flag("MTE2", "V", 0)
                    T.tile.topk(topk_res_ub_ping, group_experts_ub_ping, num_topk_sum, num_experts_per_group)
                    T.pipe_barrier("V")

                    # Process remaining groups (starting from group 1) in the loop
                    for g in T.serial(1, num_groups):
                        T.barrier_all()

                        if g % 2 == 1:
                            # Currently using data in pong, load next group to ping
                            if g + 1 < num_groups:
                                next_group_start_ping = (g + 1) * num_experts_per_group
                                T.copy(
                                    scores[token_idx, next_group_start_ping : next_group_start_ping + num_experts_per_group],
                                    group_experts_ub_ping,
                                    pad_value=-T.infinity(FLOAT32),
                                )
                            # Process current group (pong)
                            T.tile.topk(topk_res_ub_pong, group_experts_ub_pong, num_topk_sum, num_experts_per_group)
                            T.pipe_barrier("V")
                            if num_topk_sum == 1:
                                group_scores_ub[g - 1] = topk_res_ub_ping[0]
                            else:
                                group_scores_ub[g - 1] = topk_res_ub_ping[0] + topk_res_ub_ping[2]
                        else:
                            # g is even (>=2), currently using data in ping, load next group to pong
                            if g + 1 < num_groups:
                                next_group_start_pong = (g + 1) * num_experts_per_group
                                T.copy(
                                    scores[token_idx, next_group_start_pong : next_group_start_pong + num_experts_per_group],
                                    group_experts_ub_pong,
                                    pad_value=-T.infinity(FLOAT32),
                                )
                            # Process current group (ping)
                            T.tile.topk(topk_res_ub_ping, group_experts_ub_ping, num_topk_sum, num_experts_per_group)
                            T.pipe_barrier("V")
                            if num_topk_sum == 1:
                                group_scores_ub[g - 1] = topk_res_ub_pong[0]
                            else:
                                group_scores_ub[g - 1] = topk_res_ub_pong[0] + topk_res_ub_pong[2]

                        T.barrier_all()

                    if num_topk_sum == 1:
                        group_scores_ub[num_groups - 1] = topk_res_ub_pong[0] if (num_groups - 1) % 2 == 1 else topk_res_ub_ping[0]
                    else:
                        if (num_groups - 1) % 2 == 1:
                            group_scores_ub[num_groups - 1] = topk_res_ub_pong[0] + topk_res_ub_pong[2]
                        else:
                            group_scores_ub[num_groups - 1] = topk_res_ub_ping[0] + topk_res_ub_ping[2]

                    T.pipe_barrier("V")

                    T.tile.topk(group_topk_res_ub, group_scores_ub, num_topk_groups, num_groups)
                    T.pipe_barrier("V")

                    # Vectorized extraction and conversion
                    T.tile.gather_mask(topk_index_f32_ub, group_topk_res_ub, "P1010")
                    T.pipe_barrier("V")

                    T.tile.cast(topk_group_idx_out_ub, topk_index_f32_ub, "CAST_ROUND", num_topk_groups)
                    T.pipe_barrier("V")

                    T.copy(topk_group_idx_out_ub, group_topk_idx[token_idx, 0:num_aligned_topk_groups])

    return topk_sum_and_topk_group_idx_kernel


def topk_sum_and_topk_group_idx(scores: torch.Tensor, num_topk_sum: int, num_topk_groups: int) -> torch.Tensor:
    """Return top num topk groups group indices ranked by intra group top k sum"""
    assert scores.dim() == 3 and scores.is_contiguous() and scores.dtype == torch.float32
    num_tokens, num_groups, num_experts_per_group = scores.shape
    assert num_topk_sum <= num_experts_per_group and num_topk_sum in (1, 2) and num_topk_groups <= num_groups

    kernel = get_topk_sum_and_topk_group_idx_kernel(num_groups, num_experts_per_group, num_topk_groups, num_topk_sum)
    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    num_aligned_topk_groups = (num_topk_groups + 7) // 8 * 8
    topk_group_idx_padded = torch.empty(num_tokens, num_aligned_topk_groups, dtype=torch.int64, device=scores.device)

    if num_tokens == 0:
        return topk_group_idx_padded[:, :num_topk_groups].contiguous()

    kernel(scores.view(num_tokens, -1), topk_group_idx_padded)

    # Slice to original shape after aligned writeback
    return topk_group_idx_padded[:, :num_topk_groups].contiguous()


@tilelang.jit(pass_configs=pass_configs)
def get_topk_sum_and_topk_group_idx_backward_kernel(num_groups: int, num_experts_per_group: int, num_topk_groups: int, num_topk_sum: int):
    ALIGNMENT = 32
    num_experts = num_experts_per_group * num_groups
    num_aligned_experts = (num_experts + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    num_aligned_epg = (num_experts_per_group + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    num_aligned_topk_groups = (num_topk_groups + 7) // 8 * 8
    num_tokens = T.symbolic("num_tokens")

    @T.prim_func
    def topk_sum_and_topk_group_idx_backward_kernel(
        grad_out: T.Tensor[(num_tokens, num_topk_groups), FLOAT32],
        scores: T.Tensor[(num_tokens, num_experts), FLOAT32],
        group_topk_idx: T.Tensor[(num_tokens, num_topk_groups), INDEX_TYPE],
        grad_scores: T.Tensor[(num_tokens, num_aligned_experts), FLOAT32],
    ):
        VEC_NUM = 2
        num_blocks = 128
        chunk_size = num_blocks * VEC_NUM
        num_batches = (num_tokens + chunk_size - 1) // chunk_size

        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            # Token-level double buffering for base inputs
            grad_out_ub = T.alloc_ub((2, num_aligned_topk_groups), FLOAT32)
            group_topk_idx_ub = T.alloc_ub((2, num_aligned_topk_groups), INDEX_TYPE)

            group_experts_ub_ping = T.alloc_ub((num_aligned_epg,), FLOAT32)
            group_experts_ub_pong = T.alloc_ub((num_aligned_epg,), FLOAT32)
            topk_res_ub_ping = T.alloc_ub((64,), FLOAT32)
            topk_res_ub_pong = T.alloc_ub((64,), FLOAT32)

            group_grad_ub_ping = T.alloc_ub((num_aligned_epg,), FLOAT32)
            group_grad_ub_pong = T.alloc_ub((num_aligned_epg,), FLOAT32)

            for batch_idx in T.Pipelined(num_batches):
                token_idx = (batch_idx * num_blocks + cid) * VEC_NUM + vid
                ping_pong_idx = batch_idx % 2

                if token_idx < num_tokens:
                    # Fetch token metadata
                    T.copy(
                        grad_out[token_idx, 0:num_topk_groups],
                        grad_out_ub[ping_pong_idx, 0:num_topk_groups],
                        pad_value=0.0,
                    )
                    T.copy(
                        group_topk_idx[token_idx, 0:num_topk_groups],
                        group_topk_idx_ub[ping_pong_idx, 0:num_topk_groups],
                        pad_value=-1,
                    )
                    T.barrier_all()

                    if num_topk_groups > 0:
                        # Backward three-stage software pipeline: Prologue (k = 0)
                        g_0 = T.cast(group_topk_idx_ub[ping_pong_idx, 0], INT32)
                        if g_0 >= 0 and g_0 < num_groups:
                            group_start_0 = g_0 * num_experts_per_group
                            T.copy(
                                scores[token_idx, group_start_0 : group_start_0 + num_experts_per_group],
                                group_experts_ub_ping,
                                pad_value=-T.infinity(FLOAT32),
                            )

                        if num_topk_groups > 1:
                            g_1 = T.cast(group_topk_idx_ub[ping_pong_idx, 1], INT32)
                            if g_1 >= 0 and g_1 < num_groups:
                                group_start_1 = g_1 * num_experts_per_group
                                T.copy(
                                    scores[token_idx, group_start_1 : group_start_1 + num_experts_per_group],
                                    group_experts_ub_pong,
                                    pad_value=-T.infinity(FLOAT32),
                                )

                        T.barrier_all()

                        if g_0 >= 0 and g_0 < num_groups:
                            T.tile.topk(topk_res_ub_ping, group_experts_ub_ping, num_topk_sum, num_experts_per_group)
                            T.barrier_all()
                            # Zero out small UB
                            for e in T.serial(num_aligned_epg):
                                group_grad_ub_ping[e] = 0.0

                            current_grad_0 = grad_out_ub[ping_pong_idx, 0]
                            for step in T.serial(num_topk_sum):
                                idx_f32 = topk_res_ub_ping[step * 2 + 1]
                                found_e_idx = T.cast(idx_f32, INT32)
                                if found_e_idx < num_experts_per_group:
                                    group_grad_ub_ping[found_e_idx] += current_grad_0

                        # Backward three-stage software pipeline: Main Loop
                        for k in T.serial(1, num_topk_groups):
                            T.pipe_barrier("V")
                            if k % 2 == 1:
                                g_prev = T.cast(group_topk_idx_ub[ping_pong_idx, k - 1], INT32)
                                if g_prev >= 0 and g_prev < num_groups:
                                    group_start_prev = g_prev * num_experts_per_group
                                    T.copy(
                                        group_grad_ub_ping,
                                        grad_scores[token_idx, group_start_prev : group_start_prev + num_experts_per_group],
                                    )
                                    T.pipe_barrier("V")

                                if k + 1 < num_topk_groups:
                                    g_next = T.cast(group_topk_idx_ub[ping_pong_idx, k + 1], INT32)
                                    if g_next >= 0 and g_next < num_groups:
                                        group_start_next = g_next * num_experts_per_group
                                        T.copy(
                                            scores[token_idx, group_start_next : group_start_next + num_experts_per_group],
                                            group_experts_ub_ping,
                                            pad_value=-T.infinity(FLOAT32),
                                        )

                                g_curr = T.cast(group_topk_idx_ub[ping_pong_idx, k], INT32)
                                if g_curr >= 0 and g_curr < num_groups:
                                    T.tile.topk(topk_res_ub_pong, group_experts_ub_pong, num_topk_sum, num_experts_per_group)
                                    T.barrier_all()
                                    for e in T.serial(num_aligned_epg):
                                        group_grad_ub_pong[e] = 0.0

                                    current_grad = grad_out_ub[ping_pong_idx, k]
                                    for step in T.serial(num_topk_sum):
                                        idx_f32 = topk_res_ub_pong[step * 2 + 1]
                                        found_e_idx = T.cast(idx_f32, INT32)
                                        if found_e_idx < num_experts_per_group:
                                            group_grad_ub_pong[found_e_idx] += current_grad

                            else:
                                g_prev = T.cast(group_topk_idx_ub[ping_pong_idx, k - 1], INT32)
                                if g_prev >= 0 and g_prev < num_groups:
                                    group_start_prev = g_prev * num_experts_per_group
                                    T.copy(
                                        group_grad_ub_pong,
                                        grad_scores[token_idx, group_start_prev : group_start_prev + num_experts_per_group],
                                    )
                                    T.pipe_barrier("V")

                                if k + 1 < num_topk_groups:
                                    g_next = T.cast(group_topk_idx_ub[ping_pong_idx, k + 1], INT32)
                                    if g_next >= 0 and g_next < num_groups:
                                        group_start_next = g_next * num_experts_per_group
                                        T.copy(
                                            scores[token_idx, group_start_next : group_start_next + num_experts_per_group],
                                            group_experts_ub_pong,
                                            pad_value=-T.infinity(FLOAT32),
                                        )

                                g_curr = T.cast(group_topk_idx_ub[ping_pong_idx, k], INT32)
                                if g_curr >= 0 and g_curr < num_groups:
                                    T.tile.topk(topk_res_ub_ping, group_experts_ub_ping, num_topk_sum, num_experts_per_group)
                                    T.barrier_all()
                                    for e in T.serial(num_aligned_epg):
                                        group_grad_ub_ping[e] = 0.0

                                    current_grad = grad_out_ub[ping_pong_idx, k]
                                    for step in T.serial(num_topk_sum):
                                        idx_f32 = topk_res_ub_ping[step * 2 + 1]
                                        found_e_idx = T.cast(idx_f32, INT32)
                                        if found_e_idx < num_experts_per_group:
                                            group_grad_ub_ping[found_e_idx] += current_grad

                            T.barrier_all()

                        # Backward three-stage software pipeline: Epilogue (handle last group write)
                        k_last = num_topk_groups - 1
                        g_last = T.cast(group_topk_idx_ub[ping_pong_idx, k_last], INT32)
                        if g_last >= 0 and g_last < num_groups:
                            group_start_last = g_last * num_experts_per_group
                            if k_last % 2 == 0:
                                T.copy(
                                    group_grad_ub_ping,
                                    grad_scores[token_idx, group_start_last : group_start_last + num_experts_per_group],
                                )
                            else:
                                T.copy(
                                    group_grad_ub_pong,
                                    grad_scores[token_idx, group_start_last : group_start_last + num_experts_per_group],
                                )

    return topk_sum_and_topk_group_idx_backward_kernel


def topk_sum_and_topk_group_idx_backward(
    grad_out: torch.Tensor, scores: torch.Tensor, group_topk_idx: torch.Tensor, num_topk_sum: int
) -> torch.Tensor:
    """Backward pass"""
    assert grad_out.is_contiguous() and grad_out.dtype == torch.float32
    assert scores.is_contiguous() and scores.dtype == torch.float32
    assert group_topk_idx.is_contiguous() and group_topk_idx.dtype == torch.int64

    num_tokens, num_groups, num_experts_per_group = scores.shape
    num_topk_groups = group_topk_idx.shape[1]
    num_experts = num_groups * num_experts_per_group

    ALIGNMENT = 32
    num_aligned_experts = (num_experts + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    kernel = get_topk_sum_and_topk_group_idx_backward_kernel(num_groups, num_experts_per_group, num_topk_groups, num_topk_sum)

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    # Host-side aligned memory allocation (already initialized to zero, perfect for sparse update)
    grad_scores_padded = torch.zeros(num_tokens, num_aligned_experts, dtype=torch.float32, device=scores.device)

    if num_tokens > 0:
        kernel(grad_out, scores.view(num_tokens, -1), group_topk_idx, grad_scores_padded)

    # Slice to original shape after aligned writeback
    return grad_scores_padded[:, :num_experts].view(num_tokens, num_groups, num_experts_per_group).contiguous()


def ref_topk_sum_and_topk_group_idx(scores: torch.Tensor, num_topk_sum: int, num_topk_groups: int) -> torch.Tensor:
    num_tokens, num_groups, num_experts_per_group = scores.shape
    group_scores = torch.empty(num_tokens, num_groups, dtype=scores.dtype, device=scores.device)
    for g in range(num_groups):
        group_experts = scores[:, g, :]
        if num_topk_sum == 1:
            group_scores[:, g] = torch.topk(group_experts, 1, dim=1).values.squeeze(1)
        else:
            topk_vals = torch.topk(group_experts, 2, dim=1).values
            group_scores[:, g] = topk_vals.sum(dim=1)
    _, topk_group_idx = torch.topk(group_scores, num_topk_groups, dim=1)
    return topk_group_idx.to(torch.int64)


def ref_topk_sum_and_topk_group_idx_backward(
    grad_out: torch.Tensor, scores: torch.Tensor, group_topk_idx: torch.Tensor, num_topk_sum: int
) -> torch.Tensor:
    num_tokens, num_groups, num_experts_per_group = scores.shape
    num_topk_groups = group_topk_idx.shape[1]
    grad_scores = torch.zeros_like(scores)
    for t in range(num_tokens):
        for k in range(num_topk_groups):
            g = group_topk_idx[t, k].item()
            if g < 0 or g >= num_groups:
                continue
            experts = scores[t, g, :]
            if num_topk_sum == 1:
                _, top1_idx = torch.topk(experts, 1)
                grad_scores[t, g, top1_idx.item()] += grad_out[t, k].item()
            else:
                _, top2_idx = torch.topk(experts, 2)
                for e in top2_idx.tolist():
                    grad_scores[t, g, e] += grad_out[t, k].item()
    return grad_scores


def generate_test_params() -> list[dict]:
    return [
        {
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "num_groups": num_groups,
            "num_group_sum_topk": num_group_sum_topk,
            "num_topk_groups": num_topk_groups,
        }
        for num_tokens in [1, 4001, 8001]
        for num_experts in (72, 256)
        for num_groups in (4, 8, 12, 16)
        if num_experts % num_groups == 0
        for num_group_sum_topk in (1, 2)
        for num_topk_groups in (2, 4)
    ]


def make_param_id(params: dict) -> str:
    return (
        f"tok={params['num_tokens']}_exp={params['num_experts']}_"
        f"grp={params['num_groups']}_sumk={params['num_group_sum_topk']}_"
        f"topk={params['num_topk_groups']}"
    )


@pytest.mark.parametrize("params", generate_test_params(), ids=make_param_id)
def test_topk_sum_and_topk_group_idx(params):
    num_tokens = params["num_tokens"]
    num_experts = params["num_experts"]
    num_groups = params["num_groups"]
    num_group_sum_topk = params["num_group_sum_topk"]
    num_topk_groups = params["num_topk_groups"]
    num_experts_per_group = num_experts // num_groups

    torch.manual_seed(42)
    scores = torch.randn((num_tokens, num_groups, num_experts_per_group), dtype=torch.float32)
    topk_idx_ref = ref_topk_sum_and_topk_group_idx(scores, num_group_sum_topk, num_topk_groups)

    if hasattr(torch, "npu") and torch.npu.is_available():
        scores = scores.to("npu").contiguous()
    elif torch.cuda.is_available():
        scores = scores.to("cuda").contiguous()

    topk_idx = topk_sum_and_topk_group_idx(scores, num_group_sum_topk, num_topk_groups)
    np.testing.assert_equal(topk_idx.cpu().numpy(), topk_idx_ref.cpu().numpy())
    print(f"Forward Test passed for {make_param_id(params)}")


@pytest.mark.parametrize("params", generate_test_params(), ids=make_param_id)
def test_topk_sum_and_topk_group_idx_backward(params):
    num_tokens = params["num_tokens"]
    num_experts = params["num_experts"]
    num_groups = params["num_groups"]
    num_group_sum_topk = params["num_group_sum_topk"]
    num_topk_groups = params["num_topk_groups"]
    num_experts_per_group = num_experts // num_groups

    torch.manual_seed(42)
    scores = torch.randn((num_tokens, num_groups, num_experts_per_group), dtype=torch.float32)
    grad_out = torch.randn((num_tokens, num_topk_groups), dtype=torch.float32)

    topk_idx = ref_topk_sum_and_topk_group_idx(scores, num_group_sum_topk, num_topk_groups)
    grad_scores_ref = ref_topk_sum_and_topk_group_idx_backward(grad_out, scores, topk_idx, num_group_sum_topk)

    if hasattr(torch, "npu") and torch.npu.is_available():
        scores = scores.to("npu").contiguous()
        grad_out = grad_out.to("npu").contiguous()
        topk_idx = topk_idx.to("npu").contiguous()
    elif torch.cuda.is_available():
        scores = scores.to("cuda").contiguous()
        grad_out = grad_out.to("cuda").contiguous()
        topk_idx = topk_idx.to("cuda").contiguous()

    grad_scores_tl = topk_sum_and_topk_group_idx_backward(grad_out, scores, topk_idx, num_group_sum_topk)

    np.testing.assert_allclose(
        grad_scores_tl.cpu().numpy(),
        grad_scores_ref.cpu().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    print(f"Backward Test passed for {make_param_id(params)}")


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v"])
    if exit_code == pytest.ExitCode.OK:
        print("All topk_sum_and_topk_group_idx tests passed! Kernel Output Match!")
    sys.exit(exit_code)
