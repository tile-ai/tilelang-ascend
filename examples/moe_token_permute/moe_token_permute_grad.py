import math
import tilelang
import tilelang.language as T
import torch
import torch_npu


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

PASS_CONFIGS_EXPERT = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

CAL_DTYPE = "float32"
CAST_LOW2HIGH = "CAST_NONE"
CAST_HIGH2LOW = "CAST_RINT"


def _is_fp32(dtype: str) -> bool:
    return dtype in ("float32", "float")


def _pad_last_dim(tensor: torch.Tensor, target_cols: int) -> torch.Tensor:
    if tensor.shape[-1] >= target_cols:
        return tensor
    out = torch.zeros((*tensor.shape[:-1], target_cols), dtype=tensor.dtype, device=tensor.device)
    out[..., : tensor.shape[-1]] = tensor
    return out


def _build_gather_reduce_kernel_cast_group_pipelined(
    num_tokens,
    topK,
    hidden_size,
    E,
    padded_E,
    actual_cores,
    tokens_per_core,
    TILE_H,
    HALF_H,
    BATCH_T,
    n_batches,
    dtype,
    idx_dtype,
):
    stages = 2

    assert topK == 8, "group-pipelined kernel is specialized for topK == 8"
    assert dtype != CAL_DTYPE, "group-pipelined kernel is for non-fp32 dtypes"
    assert hidden_size == TILE_H, "group-pipelined kernel assumes single h-tile"
    assert HALF_H * 2 == TILE_H, "HALF_H must be TILE_H/2"

    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS_EXPERT)
    def _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        TILE_H,
        HALF_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
        stages,
    ):
        @T.prim_func
        def moe_token_permute_grad(
            perm_grad_gm: T.Tensor([E, hidden_size], dtype),
            sorted_idx_gm: T.Tensor([1, padded_E], idx_dtype),
            input_grad_gm: T.Tensor([num_tokens, hidden_size], dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                idx_ub = T.alloc_ub([1, BATCH_T * topK], idx_dtype)

                row_buf = T.alloc_ub([stages * topK, HALF_H], dtype)
                row_tmp = T.alloc_ub([1, HALF_H], dtype)
                row_f32 = T.alloc_ub([1, HALF_H], CAL_DTYPE)
                acc_buf = T.alloc_ub([1, HALF_H], CAL_DTYPE)
                out_buf = T.alloc_ub([1, HALF_H], dtype)

                h_off = vid * HALF_H

                with T.Scope("V"):
                    for batch_id in T.serial(n_batches):
                        batch_base = cid * tokens_per_core + batch_id * BATCH_T

                        T.copy(sorted_idx_gm[0, batch_base * topK], idx_ub)
                        T.set_flag("mte2", "v", 10)
                        T.wait_flag("mte2", "v", 10)

                        T.set_flag("v", "mte2", 0)
                        T.set_flag("v", "mte2", 1)
                        T.set_flag("mte3", "v", 0)

                        if BATCH_T > 0:
                            T.wait_flag("v", "mte2", 0)
                            for lane in T.serial(topK):
                                src_p = idx_ub[0, lane]
                                T.copy(
                                    perm_grad_gm[src_p, h_off],
                                    row_buf[lane, :],
                                )
                            T.set_flag("mte2", "v", 0)

                        for ti in T.serial(BATCH_T):
                            cur_stage = ti % stages
                            nxt_stage = (ti + 1) % stages
                            cur_i_tok = batch_base + ti

                            if ti + 1 < BATCH_T:
                                T.wait_flag("v", "mte2", nxt_stage)
                                nxt_tk_off = (ti + 1) * topK
                                for lane in T.serial(topK):
                                    src_n = idx_ub[0, nxt_tk_off + lane]
                                    T.copy(
                                        perm_grad_gm[src_n, h_off],
                                        row_buf[nxt_stage * topK + lane, :],
                                    )
                                T.set_flag("mte2", "v", nxt_stage)

                            T.wait_flag("mte2", "v", cur_stage)

                            T.tile.fill(acc_buf, 0.0)

                            for lane in T.serial(topK):
                                T.copy(row_buf[cur_stage * topK + lane, :], row_tmp)
                                T.tile.cast(row_f32, row_tmp, CAST_LOW2HIGH, HALF_H)
                                T.tile.add(acc_buf, acc_buf, row_f32)

                            T.set_flag("v", "mte2", cur_stage)

                            T.wait_flag("mte3", "v", 0)
                            T.tile.cast(out_buf, acc_buf, CAST_HIGH2LOW, HALF_H)
                            T.set_flag("v", "mte3", 0)

                            T.wait_flag("v", "mte3", 0)
                            if cur_i_tok < num_tokens:
                                T.copy(out_buf, input_grad_gm[cur_i_tok, h_off])
                            T.set_flag("mte3", "v", 0)

                        T.wait_flag("v", "mte2", 0)
                        T.wait_flag("v", "mte2", 1)
                        T.wait_flag("mte3", "v", 0)

        return moe_token_permute_grad

    return _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        TILE_H,
        HALF_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
        stages,
    )


def _build_gather_reduce_kernel_cast_pipelined(
    num_tokens,
    topK,
    hidden_size,
    E,
    padded_E,
    actual_cores,
    tokens_per_core,
    TILE_H,
    HALF_H,
    BATCH_T,
    n_batches,
    dtype,
    idx_dtype,
):
    stages = 8
    total_iters_per_batch = BATCH_T * topK

    assert topK <= 8
    assert dtype != CAL_DTYPE
    assert hidden_size == TILE_H
    assert HALF_H * 2 == TILE_H

    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS_EXPERT)
    def _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        TILE_H,
        HALF_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
        stages,
        total_iters_per_batch,
    ):
        @T.prim_func
        def moe_token_permute_grad(
            perm_grad_gm: T.Tensor([E, hidden_size], dtype),
            sorted_idx_gm: T.Tensor([1, padded_E], idx_dtype),
            input_grad_gm: T.Tensor([num_tokens, hidden_size], dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                idx_ub = T.alloc_ub([1, BATCH_T * topK], idx_dtype)

                row_buf = T.alloc_ub([stages, HALF_H], dtype)
                row_tmp = T.alloc_ub([1, HALF_H], dtype)
                row_f32 = T.alloc_ub([1, HALF_H], CAL_DTYPE)
                acc_buf = T.alloc_ub([1, HALF_H], CAL_DTYPE)
                out_buf = T.alloc_ub([1, HALF_H], dtype)

                h_off = vid * HALF_H

                with T.Scope("V"):
                    for batch_id in T.serial(n_batches):
                        batch_base = cid * tokens_per_core + batch_id * BATCH_T

                        T.copy(sorted_idx_gm[0, batch_base * topK], idx_ub)
                        T.set_flag("mte2", "v", 10)
                        T.wait_flag("mte2", "v", 10)

                        T.set_flag("v", "mte2", 0)
                        T.set_flag("v", "mte2", 1)
                        T.set_flag("v", "mte2", 2)
                        T.set_flag("v", "mte2", 3)
                        T.set_flag("v", "mte2", 4)
                        T.set_flag("v", "mte2", 5)
                        T.set_flag("v", "mte2", 6)
                        T.set_flag("v", "mte2", 7)
                        T.set_flag("mte3", "v", 0)

                        if total_iters_per_batch > 0:
                            T.wait_flag("v", "mte2", 0)
                            src_p0 = idx_ub[0, 0]
                            T.copy(perm_grad_gm[src_p0, h_off], row_buf[0, :])
                            T.set_flag("mte2", "v", 0)
                        if total_iters_per_batch > 1:
                            T.wait_flag("v", "mte2", 1)
                            src_p1 = idx_ub[0, 1]
                            T.copy(perm_grad_gm[src_p1, h_off], row_buf[1, :])
                            T.set_flag("mte2", "v", 1)
                        if total_iters_per_batch > 2:
                            T.wait_flag("v", "mte2", 2)
                            src_p2 = idx_ub[0, 2]
                            T.copy(perm_grad_gm[src_p2, h_off], row_buf[2, :])
                            T.set_flag("mte2", "v", 2)
                        if total_iters_per_batch > 3:
                            T.wait_flag("v", "mte2", 3)
                            src_p3 = idx_ub[0, 3]
                            T.copy(perm_grad_gm[src_p3, h_off], row_buf[3, :])
                            T.set_flag("mte2", "v", 3)
                        if total_iters_per_batch > 4:
                            T.wait_flag("v", "mte2", 4)
                            src_p4 = idx_ub[0, 4]
                            T.copy(perm_grad_gm[src_p4, h_off], row_buf[4, :])
                            T.set_flag("mte2", "v", 4)
                        if total_iters_per_batch > 5:
                            T.wait_flag("v", "mte2", 5)
                            src_p5 = idx_ub[0, 5]
                            T.copy(perm_grad_gm[src_p5, h_off], row_buf[5, :])
                            T.set_flag("mte2", "v", 5)
                        if total_iters_per_batch > 6:
                            T.wait_flag("v", "mte2", 6)
                            src_p6 = idx_ub[0, 6]
                            T.copy(perm_grad_gm[src_p6, h_off], row_buf[6, :])
                            T.set_flag("mte2", "v", 6)

                        for it in T.serial(total_iters_per_batch):
                            cur_stage = it % stages
                            cur_token = it // topK
                            cur_lane = it % topK
                            cur_i_tok = batch_base + cur_token

                            next_it = it + stages - 1
                            next_stage = next_it % stages

                            if next_it < total_iters_per_batch:
                                T.wait_flag("v", "mte2", next_stage)
                                src_n = idx_ub[0, next_it]
                                T.copy(
                                    perm_grad_gm[src_n, h_off],
                                    row_buf[next_stage, :],
                                )
                                T.set_flag("mte2", "v", next_stage)

                            T.wait_flag("mte2", "v", cur_stage)

                            if cur_lane == 0:
                                T.tile.fill(acc_buf, 0.0)

                            T.copy(row_buf[cur_stage, :], row_tmp)
                            T.tile.cast(row_f32, row_tmp, CAST_LOW2HIGH, HALF_H)
                            T.tile.add(acc_buf, acc_buf, row_f32)

                            T.set_flag("v", "mte2", cur_stage)

                            if cur_lane == topK - 1:
                                T.wait_flag("mte3", "v", 0)
                                T.tile.cast(out_buf, acc_buf, CAST_HIGH2LOW, HALF_H)
                                T.set_flag("v", "mte3", 0)

                                T.wait_flag("v", "mte3", 0)
                                if cur_i_tok < num_tokens:
                                    T.copy(out_buf, input_grad_gm[cur_i_tok, h_off])
                                T.set_flag("mte3", "v", 0)

                        T.wait_flag("v", "mte2", 0)
                        T.wait_flag("v", "mte2", 1)
                        T.wait_flag("v", "mte2", 2)
                        T.wait_flag("v", "mte2", 3)
                        T.wait_flag("v", "mte2", 4)
                        T.wait_flag("v", "mte2", 5)
                        T.wait_flag("v", "mte2", 6)
                        T.wait_flag("v", "mte2", 7)
                        T.wait_flag("mte3", "v", 0)

        return moe_token_permute_grad

    return _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        TILE_H,
        HALF_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
        stages,
        total_iters_per_batch,
    )


def _build_gather_reduce_kernel_cast(
    num_tokens,
    topK,
    hidden_size,
    E,
    padded_E,
    actual_cores,
    tokens_per_core,
    n_htiles,
    TILE_H,
    BATCH_T,
    n_batches,
    dtype,
    idx_dtype,
):
    assert topK >= 1, "cast kernel requires topK >= 1"
    HALF_H = TILE_H // 2

    dtype_bytes = 4 if dtype in ("float32", "float") else 2
    ALIGN_BYTES = 32
    if HALF_H * dtype_bytes >= ALIGN_BYTES:
        LANES_PER_ITER = min(8, topK)
    else:
        LANES_PER_ITER = 1
    n_iters = topK // LANES_PER_ITER
    rem = topK % LANES_PER_ITER

    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS_EXPERT)
    def _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        n_htiles,
        TILE_H,
        HALF_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
        LANES_PER_ITER,
        n_iters,
        rem,
    ):
        @T.prim_func
        def moe_token_permute_grad(
            perm_grad_gm: T.Tensor([E, hidden_size], dtype),
            sorted_idx_gm: T.Tensor([1, padded_E], idx_dtype),
            input_grad_gm: T.Tensor([num_tokens, hidden_size], dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                idx_ub = T.alloc_ub([1, BATCH_T * topK], idx_dtype)

                row_buf = T.alloc_ub([LANES_PER_ITER, HALF_H], dtype)
                row_tmp = T.alloc_ub([1, HALF_H], dtype)
                row_f32 = T.alloc_ub([1, HALF_H], CAL_DTYPE)
                acc_buf = T.alloc_ub([1, HALF_H], CAL_DTYPE)
                out_buf = T.alloc_ub([1, HALF_H], dtype)

                with T.Scope("V"):
                    for batch_id in T.serial(n_batches):
                        batch_base = cid * tokens_per_core + batch_id * BATCH_T

                        T.copy(sorted_idx_gm[0, batch_base * topK], idx_ub)
                        T.barrier_all()

                        for ti in T.serial(BATCH_T):
                            i = batch_base + ti
                            if i < num_tokens:
                                for ht in T.serial(n_htiles):
                                    h_off = ht * TILE_H + vid * HALF_H
                                    tk_off = ti * topK

                                    T.tile.fill(acc_buf, 0.0)

                                    for jj in T.serial(n_iters):
                                        base = jj * LANES_PER_ITER
                                        for lane in T.serial(LANES_PER_ITER):
                                            src = idx_ub[0, tk_off + base + lane]
                                            T.copy(
                                                perm_grad_gm[src, h_off],
                                                row_buf[lane, :],
                                            )
                                        T.barrier_all()
                                        for lane in T.serial(LANES_PER_ITER):
                                            T.copy(row_buf[lane, :], row_tmp)
                                            T.tile.cast(row_f32, row_tmp, CAST_LOW2HIGH, HALF_H)
                                            T.tile.add(acc_buf, acc_buf, row_f32)

                                    if rem > 0:
                                        base = n_iters * LANES_PER_ITER
                                        for lane in T.serial(rem):
                                            src = idx_ub[0, tk_off + base + lane]
                                            T.copy(
                                                perm_grad_gm[src, h_off],
                                                row_buf[lane, :],
                                            )
                                        T.barrier_all()
                                        for lane in T.serial(rem):
                                            T.copy(row_buf[lane, :], row_tmp)
                                            T.tile.cast(row_f32, row_tmp, CAST_LOW2HIGH, HALF_H)
                                            T.tile.add(acc_buf, acc_buf, row_f32)

                                    T.barrier_all()
                                    T.tile.cast(out_buf, acc_buf, CAST_HIGH2LOW, HALF_H)
                                    T.pipe_barrier("v")
                                    T.copy(out_buf, input_grad_gm[i, h_off])
                                    T.pipe_barrier("mte3")

        return moe_token_permute_grad

    return _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        n_htiles,
        TILE_H,
        HALF_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
        LANES_PER_ITER,
        n_iters,
        rem,
    )


def _build_gather_reduce_kernel_nocast(
    num_tokens,
    topK,
    hidden_size,
    E,
    padded_E,
    actual_cores,
    tokens_per_core,
    n_htiles,
    TILE_H,
    BATCH_T,
    n_batches,
    dtype,
    idx_dtype,
):
    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
    def _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        n_htiles,
        TILE_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
    ):
        @T.prim_func
        def moe_token_permute_grad(
            perm_grad_gm: T.Tensor([E, hidden_size], dtype),
            sorted_idx_gm: T.Tensor([1, padded_E], idx_dtype),
            input_grad_gm: T.Tensor([num_tokens, hidden_size], dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                idx_ub = T.alloc_shared([1, BATCH_T * topK], idx_dtype)
                row_buf0 = T.alloc_shared([1, TILE_H], dtype)
                row_buf1 = T.alloc_shared([1, TILE_H], dtype)
                row_buf2 = T.alloc_shared([1, TILE_H], dtype)
                acc_buf = T.alloc_shared([1, TILE_H], dtype)

                for batch_id in T.serial(n_batches):
                    batch_base = cid * tokens_per_core + batch_id * BATCH_T

                    T.copy(sorted_idx_gm[0, batch_base * topK], idx_ub)

                    for ti in T.serial(BATCH_T):
                        i = batch_base + ti
                        if i < num_tokens:
                            for ht in T.serial(n_htiles):
                                h_off = ht * TILE_H
                                tk_off = ti * topK

                                T.tile.fill(acc_buf, 0.0)

                                n_triples = topK // 3
                                remainder = topK % 3

                                for j3 in T.serial(n_triples):
                                    j = j3 * 3
                                    src_a = idx_ub[0, tk_off + j]
                                    src_b = idx_ub[0, tk_off + j + 1]
                                    src_c = idx_ub[0, tk_off + j + 2]
                                    T.copy(perm_grad_gm[src_a, h_off], row_buf0)
                                    T.copy(perm_grad_gm[src_b, h_off], row_buf1)
                                    T.copy(perm_grad_gm[src_c, h_off], row_buf2)
                                    T.tile.add(acc_buf, acc_buf, row_buf0)
                                    T.tile.add(acc_buf, acc_buf, row_buf1)
                                    T.tile.add(acc_buf, acc_buf, row_buf2)

                                if remainder == 2:
                                    base = n_triples * 3
                                    src_a = idx_ub[0, tk_off + base]
                                    src_b = idx_ub[0, tk_off + base + 1]
                                    T.copy(perm_grad_gm[src_a, h_off], row_buf0)
                                    T.copy(perm_grad_gm[src_b, h_off], row_buf1)
                                    T.tile.add(acc_buf, acc_buf, row_buf0)
                                    T.tile.add(acc_buf, acc_buf, row_buf1)

                                if remainder == 1:
                                    src_last = idx_ub[0, tk_off + topK - 1]
                                    T.copy(perm_grad_gm[src_last, h_off], row_buf0)
                                    T.tile.add(acc_buf, acc_buf, row_buf0)

                                T.copy(acc_buf, input_grad_gm[i, h_off])

        return moe_token_permute_grad

    return _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        n_htiles,
        TILE_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
    )


def _compile_gather_reduce(
    num_tokens: int,
    topK: int,
    hidden_size: int,
    E: int,
    NUM_CORES: int = 24,
    TILE_H: int = None,
    dtype: str = "float16",
    idx_dtype: str = "int32",
):
    ALIGN_BYTES = 32
    dtype_bytes = 4 if dtype in ("float32", "float") else 2
    align_elems = ALIGN_BYTES // dtype_bytes

    if TILE_H is None:
        for candidate in [
            hidden_size,
            4096 // dtype_bytes,
            2048 // dtype_bytes,
            1024,
            512,
            align_elems,
        ]:
            if candidate > 0 and candidate >= align_elems and hidden_size % candidate == 0:
                TILE_H = candidate
                break
        else:
            TILE_H = align_elems

    assert TILE_H * dtype_bytes >= ALIGN_BYTES and (TILE_H * dtype_bytes) % ALIGN_BYTES == 0, (
        f"TILE_H={TILE_H} * sizeof({dtype})={dtype_bytes} = {TILE_H * dtype_bytes}B; must be >= 32B and a multiple of 32B"
    )
    assert hidden_size % TILE_H == 0, f"hidden_size ({hidden_size}) must be divisible by TILE_H ({TILE_H})"

    n_htiles = int(hidden_size // TILE_H)

    actual_cores = int(min(NUM_CORES, max(1, num_tokens)))
    tokens_per_core = int(math.ceil(num_tokens / actual_cores))

    padded_E = int(actual_cores * tokens_per_core * topK)

    HALF_H_candidate = hidden_size // 2 if hidden_size % 2 == 0 else 0

    is_cast_path = dtype != CAL_DTYPE
    single_htile = hidden_size == TILE_H
    half_aligned = HALF_H_candidate > 0 and HALF_H_candidate * 2 == TILE_H and HALF_H_candidate * dtype_bytes >= ALIGN_BYTES
    pipelined_eligible = is_cast_path and single_htile and half_aligned

    use_group_pipelined = pipelined_eligible and topK == 8
    use_lane_pipelined = pipelined_eligible and topK <= 8 and not use_group_pipelined

    if use_group_pipelined:
        HALF_H = HALF_H_candidate
        BATCH_T = min(tokens_per_core, max(1, 4096 // (topK * 10)))
        while BATCH_T > 1 and tokens_per_core % BATCH_T != 0:
            BATCH_T -= 1
        n_batches = int(math.ceil(tokens_per_core / BATCH_T))

        kernel = _build_gather_reduce_kernel_cast_group_pipelined(
            num_tokens,
            topK,
            hidden_size,
            E,
            padded_E,
            actual_cores,
            tokens_per_core,
            TILE_H,
            HALF_H,
            BATCH_T,
            n_batches,
            dtype,
            idx_dtype,
        )
    elif use_lane_pipelined:
        HALF_H = HALF_H_candidate
        BATCH_T = min(tokens_per_core, max(1, 4096 // (topK * 10)))
        while BATCH_T > 1 and tokens_per_core % BATCH_T != 0:
            BATCH_T -= 1
        n_batches = int(math.ceil(tokens_per_core / BATCH_T))

        kernel = _build_gather_reduce_kernel_cast_pipelined(
            num_tokens,
            topK,
            hidden_size,
            E,
            padded_E,
            actual_cores,
            tokens_per_core,
            TILE_H,
            HALF_H,
            BATCH_T,
            n_batches,
            dtype,
            idx_dtype,
        )
    else:
        BATCH_T = min(tokens_per_core, max(1, 4096 // (topK * 10)))
        while BATCH_T > 1 and tokens_per_core % BATCH_T != 0:
            BATCH_T -= 1
        n_batches = int(math.ceil(tokens_per_core / BATCH_T))

        if dtype != CAL_DTYPE:
            kernel = _build_gather_reduce_kernel_cast(
                num_tokens,
                topK,
                hidden_size,
                E,
                padded_E,
                actual_cores,
                tokens_per_core,
                n_htiles,
                TILE_H,
                BATCH_T,
                n_batches,
                dtype,
                idx_dtype,
            )
        else:
            kernel = _build_gather_reduce_kernel_nocast(
                num_tokens,
                topK,
                hidden_size,
                E,
                padded_E,
                actual_cores,
                tokens_per_core,
                n_htiles,
                TILE_H,
                BATCH_T,
                n_batches,
                dtype,
                idx_dtype,
            )

    return kernel, padded_E


class MoeTokenPermuteGrad:
    def __init__(
        self,
        num_tokens: int,
        topK: int,
        hidden_size: int,
        num_experts: int = 64,
        num_out_tokens: int = 0,
        padded_mode: bool = False,
        NUM_CORES: int = 24,
        TILE_H: int = None,
        dtype: str = "float16",
    ):
        if padded_mode:
            raise NotImplementedError("padded_mode=True not supported.")

        self.num_tokens = num_tokens
        self.topK = topK
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.E = num_tokens * topK
        self._out_len = num_out_tokens if num_out_tokens > 0 else self.E

        min_compile_h = 64 if _is_fp32(dtype) else 32
        self._compile_hidden_size = max(hidden_size, min_compile_h)
        compile_tile_h = TILE_H if TILE_H is None else max(TILE_H, min_compile_h)

        self._kernel, self._padded_E = _compile_gather_reduce(
            num_tokens,
            topK,
            self._compile_hidden_size,
            self.E,
            NUM_CORES=NUM_CORES,
            TILE_H=compile_tile_h,
            dtype=dtype,
        )

        self._sorted_idx_buf = None
        self._perm_grad_pad_buf = None

    def _get_idx_buf(self, device):
        if self._sorted_idx_buf is None or self._sorted_idx_buf.device != device:
            self._sorted_idx_buf = torch.zeros(
                self._padded_E,
                dtype=torch.int32,
                device=device,
            )
        return self._sorted_idx_buf

    def __call__(self, permuted_output_grad, sorted_indices):
        device = permuted_output_grad.device
        E = self.E
        H = self._compile_hidden_size

        needs_pad = permuted_output_grad.shape[0] < E or permuted_output_grad.shape[1] < H
        if needs_pad:
            target_shape = (E, H)
            if (
                self._perm_grad_pad_buf is None
                or self._perm_grad_pad_buf.device != device
                or self._perm_grad_pad_buf.dtype != permuted_output_grad.dtype
                or tuple(self._perm_grad_pad_buf.shape) != target_shape
            ):
                self._perm_grad_pad_buf = torch.zeros(
                    *target_shape,
                    dtype=permuted_output_grad.dtype,
                    device=device,
                )
            perm_grad_padded = self._perm_grad_pad_buf
            r = permuted_output_grad.shape[0]
            c = permuted_output_grad.shape[1]
            perm_grad_padded[:r, :c].copy_(permuted_output_grad)
        else:
            perm_grad_padded = permuted_output_grad

        sorted_idx_padded = self._get_idx_buf(device)
        si = sorted_indices.view(-1)
        if si.dtype != torch.int32:
            si = si.to(torch.int32)
        sorted_idx_padded[:E].copy_(si)

        input_grad = self._kernel(
            perm_grad_padded,
            sorted_idx_padded.unsqueeze(0),
        )

        if self.hidden_size != H:
            input_grad = input_grad[:, : self.hidden_size].contiguous()

        return input_grad

    def __repr__(self):
        return f"MoeTokenPermuteGrad(T={self.num_tokens}, K={self.topK}, H={self.hidden_size}, experts={self.num_experts})"


def test_permute_grad_parameterized(pt_dtype, tl_dtype_str):
    print(f"\n{'=' * 65}")
    print(f"Testing MoeTokenPermuteGrad, dtype: {tl_dtype_str.upper()}")
    print(f"{'=' * 65}")

    torch.manual_seed(42)

    num_tokens = 16
    hidden_size = 16
    topk = 4
    num_experts = 4

    all_passed = True

    print(">>> Test case 1: Standard Backward gradient alignment test")

    tokens = torch.randn(num_tokens, hidden_size, dtype=pt_dtype, device="npu", requires_grad=True)
    indices = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device="npu")

    npu_permuted, npu_sorted_idx = torch_npu.npu_moe_token_permute(tokens, indices)

    grad_permuted_tokens = torch.randn_like(npu_permuted)

    npu_permuted.backward(grad_permuted_tokens)
    npu_input_grad = tokens.grad.clone()

    tokens.grad.zero_()

    tl_grad_op = MoeTokenPermuteGrad(
        num_tokens=num_tokens,
        topK=topk,
        hidden_size=hidden_size,
        num_experts=num_experts,
        dtype=tl_dtype_str,
    )
    tl_input_grad = tl_grad_op(grad_permuted_tokens, npu_sorted_idx)
    torch.npu.synchronize()

    try:
        torch.testing.assert_close(tl_input_grad, npu_input_grad)
        print(f"    [PASS] {tl_dtype_str.upper()} Standard Backward precision test passed!")
    except AssertionError as e:
        print(
            f"    [FAILED] {tl_dtype_str.upper()} Standard Backward precision test failed!\n",
            e,
        )
        all_passed = False

    print("\n>>> Test case 2: Clip Backward gradient alignment test with truncation")
    num_out_tokens = 10

    tokens_clip = torch.randn(num_tokens, hidden_size, dtype=pt_dtype, device="npu", requires_grad=True)
    indices_clip = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device="npu")

    npu_permuted_clip, npu_sorted_idx_clip = torch_npu.npu_moe_token_permute(tokens_clip, indices_clip, num_out_tokens=num_out_tokens)

    grad_permuted_clip = torch.randn_like(npu_permuted_clip)
    npu_permuted_clip.backward(grad_permuted_clip)
    npu_input_grad_clip = tokens_clip.grad.clone()

    tl_grad_op_clip = MoeTokenPermuteGrad(
        num_tokens=num_tokens,
        topK=topk,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_out_tokens=num_out_tokens,
        dtype=tl_dtype_str,
    )
    tl_input_grad_clip = tl_grad_op_clip(grad_permuted_clip, npu_sorted_idx_clip)
    torch.npu.synchronize()

    try:
        torch.testing.assert_close(tl_input_grad_clip, npu_input_grad_clip)
        print(f"    [PASS] {tl_dtype_str.upper()} Clip truncation Backward precision test passed!")
    except AssertionError as e:
        print(
            f"    [FAILED] {tl_dtype_str.upper()} Clip truncation Backward precision test failed!\n",
            e,
        )
        all_passed = False

    return all_passed


def test_permute_grad():
    dtypes_to_test = [
        (torch.float16, "float16"),
        (torch.bfloat16, "bfloat16"),
        (torch.float32, "float32"),
    ]

    overall_passed = True
    for pt_type, tl_type_str in dtypes_to_test:
        passed = test_permute_grad_parameterized(pt_dtype=pt_type, tl_dtype_str=tl_type_str)
        if not passed:
            overall_passed = False

    print(f"\n{'=' * 65}")
    if overall_passed:
        print("Test passed!")
    else:
        print("Test failed! The precision is not correct!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    test_permute_grad()
