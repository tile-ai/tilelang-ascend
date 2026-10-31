"""Original MoE backward factory, TA 0deb8b93 lines 175--352.

Only imports and the external JIT decorator are adapted; the entire factory
algorithm is preserved for the flat-Bind/CV migration regression.
"""

from tilelang.ascend.a2a3 import language as T

FLOAT32 = "float32"
INDEX_TYPE = "int64"
INT32 = "int32"


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
