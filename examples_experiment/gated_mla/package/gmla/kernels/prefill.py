"""Isolated C4 persistent Cube prefill/continuation; see design.md.

No NPU code runs at import. This is not wired into the released operator.
QL uses the input dtype; BF16 P uses signed hi/residual PV accumulation.
All GEMM outputs, online state and final Z are FP32.
"""

from functools import lru_cache
import math

import tilelang
from tilelang import language as T

from .._contracts import validated_host_scale, validate_attention_spans, validate_tail_spans


DEFAULT_SCALE = 1.0 / math.sqrt(192)


def _validated_scale(scale):
    """Validate a host scalar without inspecting any tensor values."""
    return validated_host_scale(scale, DEFAULT_SCALE)


EXPERT_PASSES = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}


def probability_precision_mode(dtype):
    """Static dtype policy; there is no runtime opt-out of the BF16 repair."""
    if dtype == "float16":
        return "fp16_p_single"
    if dtype == "bfloat16":
        return "bf16_p_hi_residual"
    raise ValueError("dtype must be 'float16' or 'bfloat16'")


def buffer_plan(block_m=32, block_n=64, *, batch=1, heads=96, query_capacity=1, max_blocks=24, dtype="float16", has_ql_residual=False):
    """Pure host metadata: disjoint, 32-byte-aligned addresses per memory scope."""
    if block_m not in (16, 32) or block_n not in (64, 128):
        raise ValueError("Cube decode supports block_m in (16,32), block_n in (64,128)")
    if type(has_ql_residual) is not bool:
        raise TypeError("has_ql_residual must be bool")
    mode = probability_precision_mode(dtype)
    hm = block_m // 2
    probability_columns = block_n * (2 if dtype == "bfloat16" else 1)
    sizes = {
        "L1": {
            "q_l1": block_m * 512 * 2,
            "qa_l1": block_m * 64 * 2,
            "c_l1": block_n * 512 * 2,
            "e_l1": block_n * 64 * 2,
            "p_l1": block_m * block_n * 2,
        },
        "L0C": {
            "score_l0c": block_m * block_n * 4,
            "partial_l0c": block_m * 512 * 4,
        },
        "UB": {
            "z": hm * 512 * 4,
            "scores": hm * block_n * 4,
            "normalizer": hm * 4,
            "maximum": hm * 4,
            "previous": hm * 4,
            "block_sum": hm * 4,
            "p_half": hm * block_n * 2,
            "partial": hm * 512 * 4,
        },
    }
    if has_ql_residual:
        sizes["L1"]["q_lo_l1"] = block_m * 512 * 2
    if dtype == "bfloat16":
        sizes["UB"]["prob_round_ub"] = hm * block_n * 4
    sizes["UB"].update(
        column_indices=block_n * 4,
        prefix_mask=32,
        select_scratch=block_n * 4,
        mask_columns=hm * block_n * 4,
        block_mask=hm * 32,
        block_select_scratch=hm * block_n * 4,
    )
    result = {}
    for scope, buffers in sizes.items():
        end = 0
        rows = {}
        for name, count in buffers.items():
            start = (end + 31) // 32 * 32
            rows[name] = {"offset": start, "bytes": count, "end": start + count}
            end = start + count
        result[scope] = {"buffers": rows, "total_bytes": end}
    total_tasks = batch * ((query_capacity * heads + block_m - 1) // block_m)
    blocks = min(max_blocks, total_tasks)
    result["logical_tasks"] = total_tasks
    result["physical_blocks"] = blocks
    result["waves"] = (total_tasks + blocks - 1) // blocks
    result["workspace_bytes"] = blocks * (block_m * block_n * 4 + block_m * probability_columns * 2 + block_m * 512 * 4)
    result["has_ql_residual"] = has_ql_residual
    result["query_precision_mode"] = "hi_plus_residual" if has_ql_residual else "single"
    result["precision_mode"] = mode
    result["probability_columns"] = probability_columns
    return result


@lru_cache(maxsize=128, typed=True)
def build_cube_attention(
    *,
    batch=1,
    heads=96,
    query_capacity=1,
    cache_capacity=128,
    block_m=32,
    block_n=64,
    target="pto",
    dtype="float16",
    max_blocks=24,
    scale=None,
    has_ql_residual=False,
):
    """Compile prefill/continuation shapes; runtime metadata VALUES are not factory arguments.

    Tail aliases have original tensor shapes on aligned axes. The production
    entry below launches only necessary NPU tail preparation, then this kernel.
    """
    if target not in ("pto", "ascendc"):
        raise ValueError("target must be 'pto' or 'ascendc'")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError("dtype must be 'float16' or 'bfloat16'")
    scale = _validated_scale(scale)
    if any(type(value) is not int or value < 1 for value in (batch, heads, query_capacity, cache_capacity)):
        raise ValueError("batch, heads, query_capacity and cache_capacity must be positive integers")
    if type(max_blocks) is not int or not 1 <= max_blocks <= 24:
        raise ValueError("max_blocks must be an integer in[1,24]")
    validate_attention_spans(batch, query_capacity, heads, cache_capacity, block_m=block_m, block_n=block_n, max_blocks=max_blocks)
    rows = query_capacity * heads
    if rows >= 2**31 or cache_capacity >= 2**31:
        raise ValueError("row and cache extents must fit signed int32")
    plan = buffer_plan(
        block_m,
        block_n,
        batch=batch,
        heads=heads,
        query_capacity=query_capacity,
        max_blocks=max_blocks,
        dtype=dtype,
        has_ql_residual=has_ql_residual,
    )
    l1 = {name: row["offset"] for name, row in plan["L1"]["buffers"].items()}
    q_lo_offset = l1.get("q_lo_l1", 0)
    l0c = {name: row["offset"] for name, row in plan["L0C"]["buffers"].items()}
    ub = {name: row["offset"] for name, row in plan["UB"]["buffers"].items()}
    use_p_residual = dtype == "bfloat16"
    probability_columns = plan["probability_columns"]
    prob_round_offset = ub.get("prob_round_ub", plan["UB"]["total_bytes"])
    row_tiles = (rows + block_m - 1) // block_m
    total_tasks = batch * row_tiles
    if total_tasks >= 2**31 - 24:
        raise ValueError("task IDs must fit signed int32")
    blocks = min(max_blocks, total_tasks)
    waves = (total_tasks + blocks - 1) // blocks
    head_remainder = rows % block_m
    cache_remainder = cache_capacity % block_n
    head_tail_extent = block_m if head_remainder else rows
    cache_tail_extent = block_n if cache_remainder else cache_capacity
    half_m = block_m // 2
    iterations = (cache_capacity + block_n - 1) // block_n
    single_block_uniform = iterations == 1 and heads % block_m == 0
    pv_l0_k = 64 if target == "pto" else 32

    def allocate_query_residual(q_hi_buffer):
        # Returning the existing Buffer allocates nothing for the static None path.
        if has_ql_residual:
            return T.alloc_L1((block_m, 512), dtype)
        return q_hi_buffer

    @T.prim_func
    def main(
        QL: T.Tensor((batch, rows, 512), dtype),
        QA: T.Tensor((batch, rows, 64), dtype),
        C: T.Tensor((batch, cache_capacity, 512), dtype),
        E: T.Tensor((batch, cache_capacity, 64), dtype),
        Metadata: T.Tensor((batch, 8), "int32"),
        QTail: T.Tensor((batch, head_tail_extent, 512), dtype),
        QATail: T.Tensor((batch, head_tail_extent, 64), dtype),
        CTail: T.Tensor((batch, cache_tail_extent, 512), dtype),
        ETail: T.Tensor((batch, cache_tail_extent, 64), dtype),
        QLo: T.Tensor((batch, rows, 512), dtype),
        QLoTail: T.Tensor((batch, head_tail_extent, 512), dtype),
        Z: T.Tensor((batch, rows, 512), "float32"),
        score_workspace: T.Tensor((blocks, block_m, block_n), "float32"),
        probability_workspace: T.Tensor((blocks, block_m, probability_columns), dtype),
        partial_workspace: T.Tensor((blocks, block_m, 512), "float32"),
    ):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            row0 = vid * half_m
            c_live_rows = T.alloc_var("int32", init=0)
            c_last_row = T.alloc_var("int32", init=0)
            c_max_visible = T.alloc_var("int32", init=0)
            c_cached_bid = T.alloc_var("int32")
            c_prefetched_task = T.alloc_var("int32")
            c_next_task = T.alloc_var("int32")
            c_next_bid = T.alloc_var("int32")
            c_next_row = T.alloc_var("int32")
            c_next_live_rows = T.alloc_var("int32")
            v_live_rows = T.alloc_var("int32", init=0)
            v_last_row = T.alloc_var("int32", init=0)
            v_max_visible = T.alloc_var("int32", init=0)
            v_start = T.alloc_var("int32", init=0)
            v_kvlen = T.alloc_var("int32", init=0)
            valid_rows = T.alloc_var("int32", init=0)
            mask_limit = T.alloc_var("int32", init=0)
            mask_remaining = T.alloc_var("int32", init=0)
            q_l1 = T.alloc_L1((block_m, 512), dtype)
            q_lo_l1 = allocate_query_residual(q_l1)
            qa_l1 = T.alloc_L1((block_m, 64), dtype)
            c_l1 = T.alloc_L1((block_n, 512), dtype)
            e_l1 = T.alloc_L1((block_n, 64), dtype)
            p_l1 = T.alloc_L1((block_m, block_n), dtype)
            score_l0c = T.alloc_L0C((block_m, block_n), "float32")
            partial_l0c = T.alloc_L0C((block_m, 512), "float32")
            z = T.alloc_ub((half_m, 512), "float32")
            scores = T.alloc_ub((half_m, block_n), "float32")
            normalizer = T.alloc_ub((half_m,), "float32")
            maximum = T.alloc_ub((half_m,), "float32")
            previous = T.alloc_ub((half_m,), "float32")
            block_sum = T.alloc_ub((half_m,), "float32")
            p_half = T.alloc_ub((half_m, block_n), dtype)
            partial = T.alloc_ub((half_m, 512), "float32")
            column_indices = T.alloc_ub((block_n,), "float32")
            prefix_mask = T.alloc_ub((32,), "uint8")
            select_scratch = T.alloc_ub((block_n * 4,), "uint8")
            mask_columns = T.alloc_ub((half_m, block_n), "float32")
            block_mask = T.alloc_ub((half_m, block_n // 8), "uint8")
            block_select_scratch = T.alloc_ub((half_m * block_n * 4,), "uint8")
            T.annotate_address(
                {
                    q_l1: l1["q_l1"],
                    qa_l1: l1["qa_l1"],
                    c_l1: l1["c_l1"],
                    e_l1: l1["e_l1"],
                    p_l1: l1["p_l1"],
                    score_l0c: l0c["score_l0c"],
                    partial_l0c: l0c["partial_l0c"],
                    z: ub["z"],
                    scores: ub["scores"],
                    normalizer: ub["normalizer"],
                    maximum: ub["maximum"],
                    previous: ub["previous"],
                    block_sum: ub["block_sum"],
                    p_half: ub["p_half"],
                    partial: ub["partial"],
                    column_indices: ub["column_indices"],
                    prefix_mask: ub["prefix_mask"],
                    select_scratch: ub["select_scratch"],
                    mask_columns: ub["mask_columns"],
                    block_mask: ub["block_mask"],
                    block_select_scratch: ub["block_select_scratch"],
                }
            )

            if has_ql_residual:
                T.annotate_address({q_lo_l1: q_lo_offset})

            # Same-group, single-buffer four-flag protocol from manual FA.
            with T.Scope("C"):
                # Reset within the Cube scope on every kernel launch/replay.
                c_cached_bid = -1
                c_prefetched_task = -1
                for c_wave in T.serial(waves):
                    c_task = c_wave * blocks + cid
                    if c_task < total_tasks:
                        c_bid = c_task // row_tiles
                        c_row_start = (c_task % row_tiles) * block_m
                        c_live_rows = Metadata[c_bid, 0] * heads
                        if c_row_start < c_live_rows:
                            c_last_row = c_row_start + block_m - 1
                            if c_last_row >= c_live_rows:
                                c_last_row = c_live_rows - 1
                            c_max_visible = Metadata[c_bid, 2] + c_last_row // heads + 1
                            if c_max_visible > Metadata[c_bid, 1]:
                                c_max_visible = Metadata[c_bid, 1]
                            if iterations == 1:
                                if c_prefetched_task != c_task:
                                    if head_remainder != 0 and c_row_start + block_m > rows:
                                        T.copy(QTail[c_bid, :, :], q_l1)
                                        T.copy(QATail[c_bid, :, :], qa_l1)
                                        if has_ql_residual:
                                            T.copy(QLoTail[c_bid, :, :], q_lo_l1)
                                    else:
                                        T.copy(QL[c_bid, c_row_start : c_row_start + block_m, :], q_l1)
                                        T.copy(QA[c_bid, c_row_start : c_row_start + block_m, :], qa_l1)
                                        if has_ql_residual:
                                            T.copy(QLo[c_bid, c_row_start : c_row_start + block_m, :], q_lo_l1)
                                c_prefetched_task = -1
                            else:
                                if head_remainder != 0 and c_row_start + block_m > rows:
                                    T.copy(QTail[c_bid, :, :], q_l1)
                                    T.copy(QATail[c_bid, :, :], qa_l1)
                                    if has_ql_residual:
                                        T.copy(QLoTail[c_bid, :, :], q_lo_l1)
                                else:
                                    T.copy(QL[c_bid, c_row_start : c_row_start + block_m, :], q_l1)
                                    T.copy(QA[c_bid, c_row_start : c_row_start + block_m, :], qa_l1)
                                    if has_ql_residual:
                                        T.copy(QLo[c_bid, c_row_start : c_row_start + block_m, :], q_lo_l1)
                            T.barrier_all()
                            for k in T.serial(iterations):
                                if k * block_n < c_max_visible:
                                    if iterations == 1:
                                        if c_cached_bid != c_bid:
                                            if cache_remainder != 0 and k == iterations - 1:
                                                T.copy(CTail[c_bid, :, :], c_l1)
                                                T.copy(ETail[c_bid, :, :], e_l1)
                                            else:
                                                T.copy(C[c_bid, k * block_n : (k + 1) * block_n, :], c_l1)
                                                T.copy(E[c_bid, k * block_n : (k + 1) * block_n, :], e_l1)
                                        T.barrier_all()
                                        c_cached_bid = c_bid
                                    else:
                                        if cache_remainder != 0 and k == iterations - 1:
                                            T.copy(CTail[c_bid, :, :], c_l1)
                                            T.copy(ETail[c_bid, :, :], e_l1)
                                        else:
                                            T.copy(C[c_bid, k * block_n : (k + 1) * block_n, :], c_l1)
                                            T.copy(E[c_bid, k * block_n : (k + 1) * block_n, :], e_l1)
                                        T.barrier_all()
                                    T.gemm_v0(q_l1, c_l1, score_l0c, transpose_B=True, init=True, kL0Size=128)
                                    T.barrier_all()
                                    if has_ql_residual:
                                        T.gemm_v0(q_lo_l1, c_l1, score_l0c, transpose_B=True, init=False, kL0Size=128)
                                        T.barrier_all()
                                    T.gemm_v0(qa_l1, e_l1, score_l0c, transpose_B=True, init=False, kL0Size=64)
                                    T.barrier_all()
                                    T.copy(score_l0c, score_workspace[cid, :, :])
                                    T.barrier_all()
                                    T.set_cross_flag("FIX", 0)

                                    if iterations == 1:
                                        # Current Q/QA/Lo are dead; PV only consumes P and C.
                                        c_next_task = c_task + blocks
                                        if c_next_task < total_tasks:
                                            c_next_bid = c_next_task // row_tiles
                                            c_next_row = (c_next_task % row_tiles) * block_m
                                            if c_next_bid == c_bid:
                                                c_next_live_rows = c_live_rows
                                            else:
                                                c_next_live_rows = Metadata[c_next_bid, 0] * heads
                                            if c_next_row < c_next_live_rows:
                                                if head_remainder != 0 and c_next_row + block_m > rows:
                                                    T.copy(QTail[c_next_bid, :, :], q_l1)
                                                    T.copy(QATail[c_next_bid, :, :], qa_l1)
                                                    if has_ql_residual:
                                                        T.copy(QLoTail[c_next_bid, :, :], q_lo_l1)
                                                else:
                                                    T.copy(QL[c_next_bid, c_next_row : c_next_row + block_m, :], q_l1)
                                                    T.copy(QA[c_next_bid, c_next_row : c_next_row + block_m, :], qa_l1)
                                                    if has_ql_residual:
                                                        T.copy(QLo[c_next_bid, c_next_row : c_next_row + block_m, :], q_lo_l1)
                                                c_prefetched_task = c_next_task

                                    T.wait_cross_flag(1)
                                    T.barrier_all()
                                    if use_p_residual:
                                        T.copy(probability_workspace[cid, :, 0:block_n], p_l1)
                                    else:
                                        T.copy(probability_workspace[cid, :, :], p_l1)
                                    T.barrier_all()
                                    # C stays resident from QK through PV; no second GM load.
                                    T.gemm_v0(p_l1, c_l1, partial_l0c, init=True, kL0Size=pv_l0_k)
                                    T.barrier_all()
                                    if use_p_residual:
                                        T.copy(probability_workspace[cid, :, block_n : 2 * block_n], p_l1)
                                        T.barrier_all()
                                        T.gemm_v0(p_l1, c_l1, partial_l0c, init=False, kL0Size=pv_l0_k)
                                        T.barrier_all()
                                    T.copy(partial_l0c, partial_workspace[cid, :, :])
                                    T.barrier_all()
                                    T.set_cross_flag("FIX", 2)
                                    T.wait_cross_flag(3)
                                    T.barrier_all()

            with T.Scope("V"):
                T.tile.createvecindex(column_indices, 0.0)
                T.barrier_all()
                if heads % block_m == 0:
                    T.tile.broadcast(mask_columns, column_indices, axis=0)
                    T.barrier_all()
                for v_wave in T.serial(waves):
                    v_task = v_wave * blocks + cid
                    if v_task < total_tasks:
                        v_bid = v_task // row_tiles
                        v_row_start = (v_task % row_tiles) * block_m
                        v_live_rows = Metadata[v_bid, 0] * heads
                        v_start = Metadata[v_bid, 2]
                        v_kvlen = Metadata[v_bid, 1]
                        valid_rows = 0
                        if rows > v_row_start + row0:
                            valid_rows = rows - v_row_start - row0
                            if valid_rows > half_m:
                                valid_rows = half_m
                        T.tile.fill(z, 0.0)
                        if not single_block_uniform:
                            for i in T.Parallel(half_m):
                                normalizer[i] = T.if_then_else(v_row_start + row0 + i < v_live_rows, 0.0, 1.0)
                                maximum[i] = T.if_then_else(v_row_start + row0 + i < v_live_rows, -T.infinity("float32"), 0.0)
                        T.barrier_all()
                        if v_row_start < v_live_rows:
                            v_last_row = v_row_start + block_m - 1
                            if v_last_row >= v_live_rows:
                                v_last_row = v_live_rows - 1
                            v_max_visible = v_start + v_last_row // heads + 1
                            if v_max_visible > v_kvlen:
                                v_max_visible = v_kvlen
                            for _k in T.serial(iterations):
                                if _k * block_n < v_max_visible:
                                    if not single_block_uniform:
                                        T.copy(maximum, previous)
                                        T.barrier_all()
                                    T.wait_cross_flag(0)
                                    T.copy(score_workspace[cid, row0 : row0 + half_m, :], scores)
                                    T.barrier_all()
                                    T.tile.mul(scores, scores, scale)
                                    T.barrier_all()
                                    if heads % block_m == 0:
                                        # A head-aligned tile belongs to exactly one query.
                                        mask_limit = v_start + v_row_start // heads + 1
                                        if mask_limit > v_kvlen:
                                            mask_limit = v_kvlen
                                        mask_remaining = mask_limit - _k * block_n
                                        if mask_remaining <= 0:
                                            T.tile.fill(scores, -T.infinity("float32"))
                                            T.barrier_all()
                                        elif mask_remaining < block_n:
                                            T.tile.compare(block_mask, mask_columns, T.cast(mask_remaining, "float32"), "LT")
                                            T.barrier_all()
                                            T.tile.select(
                                                scores,
                                                block_mask,
                                                scores,
                                                -T.infinity("float32"),
                                                "VSEL_TENSOR_SCALAR_MODE",
                                                tmp=block_select_scratch,
                                            )
                                            T.barrier_all()
                                    else:
                                        for mask_row in T.serial(half_m):
                                            if v_row_start + row0 + mask_row < v_live_rows:
                                                mask_limit = v_start + (v_row_start + row0 + mask_row) // heads + 1
                                                if mask_limit > v_kvlen:
                                                    mask_limit = v_kvlen
                                                mask_remaining = mask_limit - _k * block_n
                                                if mask_remaining <= 0:
                                                    T.tile.fill(scores[mask_row, :], -T.infinity("float32"))
                                                    T.barrier_all()
                                                elif mask_remaining < block_n:
                                                    T.tile.compare(prefix_mask, column_indices, T.cast(mask_remaining, "float32"), "LT")
                                                    T.barrier_all()
                                                    T.tile.select(
                                                        scores[mask_row, :],
                                                        prefix_mask,
                                                        scores[mask_row, :],
                                                        -T.infinity("float32"),
                                                        "VSEL_TENSOR_SCALAR_MODE",
                                                        tmp=select_scratch,
                                                    )
                                                    T.barrier_all()
                                            else:
                                                T.tile.fill(scores[mask_row, :], 0.0)
                                                T.barrier_all()
                                    T.barrier_all()
                                    T.reduce_max(scores, maximum, dim=-1)
                                    T.barrier_all()
                                    if not single_block_uniform:
                                        T.tile.max(maximum, maximum, previous)
                                        T.barrier_all()
                                        T.tile.sub(previous, previous, maximum)
                                        T.barrier_all()
                                        T.tile.exp(previous, previous)
                                        T.barrier_all()
                                    for row in T.serial(half_m):
                                        T.tile.sub(scores[row, :], scores[row, :], maximum[row])
                                        T.barrier_all()
                                    T.tile.exp(scores, scores)
                                    T.barrier_all()
                                    # Invalid query rows must contribute no mass to l or Cube PV.
                                    if v_row_start + row0 + half_m > v_live_rows:
                                        for mask_row in T.serial(half_m):
                                            if v_row_start + row0 + mask_row >= v_live_rows:
                                                T.tile.fill(scores[mask_row, :], 0.0)
                                                T.barrier_all()
                                    T.barrier_all()
                                    if single_block_uniform:
                                        # First and only block: l0=0 and z0=0, so no online rescale is needed.
                                        T.reduce_sum(scores, normalizer, dim=-1)
                                        T.barrier_all()
                                    else:
                                        T.reduce_sum(scores, block_sum, dim=-1)
                                        T.barrier_all()
                                        T.tile.mul(normalizer, normalizer, previous)
                                        T.barrier_all()
                                        T.tile.add(normalizer, normalizer, block_sum)
                                        T.barrier_all()
                                        for row in T.serial(half_m):
                                            T.tile.mul(z[row, :], z[row, :], previous[row])
                                            T.barrier_all()

                                    # Explicitly quantized P uses dtype; l retains FP32 p sum.
                                    T.copy(scores, p_half)
                                    T.barrier_all()
                                    if use_p_residual:
                                        T.copy(p_half, probability_workspace[cid, row0 : row0 + half_m, 0:block_n])
                                    else:
                                        T.copy(p_half, probability_workspace[cid, row0 : row0 + half_m, :])
                                    T.barrier_all()
                                    if use_p_residual:
                                        # Independent FP32 storage; never alias scores, z, or partial.
                                        prob_round_ub = T.alloc_ub((half_m, block_n), "float32")
                                        T.annotate_address({prob_round_ub: prob_round_offset})
                                        T.copy(p_half, prob_round_ub)
                                        T.barrier_all()
                                        # FP32 block_sum/l already used the ORIGINAL P above.
                                        T.tile.sub(scores, scores, prob_round_ub)
                                        T.barrier_all()
                                        T.copy(scores, p_half)
                                        T.barrier_all()
                                        T.copy(p_half, probability_workspace[cid, row0 : row0 + half_m, block_n : 2 * block_n])
                                        T.barrier_all()
                                    T.set_cross_flag("MTE3", 1)
                                    T.wait_cross_flag(2)
                                    T.barrier_all()
                                    T.copy(partial_workspace[cid, row0 : row0 + half_m, :], partial)
                                    T.barrier_all()
                                    T.tile.add(z, z, partial)
                                    T.barrier_all()
                                    T.set_cross_flag("V", 3)
                                    T.barrier_all()

                            for row in T.serial(half_m):
                                T.tile.div(z[row, :], z[row, :], normalizer[row])
                                T.barrier_all()
                        if valid_rows > 0:
                            T.copy(z[0:valid_rows, :], Z[v_bid, v_row_start + row0 : v_row_start + row0 + valid_rows, :])
                            T.barrier_all()

    return tilelang.compile(
        main,
        out_idx=[11],
        workspace_idx=[12, 13, 14],
        target=target,
        pass_configs=EXPERT_PASSES,
    )


@lru_cache(maxsize=128)
def build_tail_pair(*, batch, extent, tile, dtype="float16", target="pto"):
    """Prepare only the last physical row tile of a 512/64-width tensor pair.

    Used for QL/QA head tails and C/E cache tails. Every output row is written,
    including explicit zero rows beyond the original physical tensor extent.
    """
    if target not in ("pto", "ascendc") or dtype not in ("float16", "bfloat16"):
        raise ValueError("unsupported target or dtype")
    if any(type(value) is not int or value < 1 for value in (batch, extent, tile)):
        raise ValueError("batch, extent and tile must be positive integers")
    if tile not in (16, 32, 64, 128) or extent % tile == 0:
        raise ValueError("tail preparation requires a supported tile and a nonzero remainder")
    validate_tail_spans(batch, extent, tile)
    half = tile // 2
    remainder = extent % tile
    last_start = extent - remainder

    @T.prim_func
    def main(
        A: T.Tensor((batch, extent, 512), dtype),
        Aux: T.Tensor((batch, extent, 64), dtype),
        APadded: T.Tensor((batch, tile, 512), dtype),
        AuxPadded: T.Tensor((batch, tile, 64), dtype),
    ):
        with T.Kernel(batch, is_npu=True) as (bid, vid):
            row0 = vid * half
            valid = T.alloc_var("int32", init=0)
            if remainder > row0:
                valid = remainder - row0
                if valid > half:
                    valid = half
            a = T.alloc_ub((half, 512), dtype)
            aux = T.alloc_ub((half, 64), dtype)
            T.annotate_address({a: 0, aux: half * 512 * 2})
            with T.Scope("V"):
                T.tile.fill(a, 0.0)
                T.tile.fill(aux, 0.0)
                T.barrier_all()
                if valid > 0:
                    T.copy(A[bid, last_start + row0 : last_start + row0 + valid, :], a[0:valid, :])
                    T.copy(Aux[bid, last_start + row0 : last_start + row0 + valid, :], aux[0:valid, :])
                    T.barrier_all()
                T.copy(a, APadded[bid, row0 : row0 + half, :])
                T.copy(aux, AuxPadded[bid, row0 : row0 + half, :])
                T.barrier_all()

    return tilelang.compile(main, out_idx=[2, 3], target=target, pass_configs=EXPERT_PASSES)


@lru_cache(maxsize=128)
def build_tail_triple(*, batch, extent, tile, dtype="float16", target="pto"):
    """Prepare one final Hi/Lo/Aux row tile without duplicate Aux transfers.

    Used for residual query/head tails only. Every output row is written,
    including explicit zero rows beyond the original physical tensor extent.
    """
    if target not in ("pto", "ascendc") or dtype not in ("float16", "bfloat16"):
        raise ValueError("unsupported target or dtype")
    if any(type(value) is not int or value < 1 for value in (batch, extent, tile)):
        raise ValueError("batch, extent and tile must be positive integers")
    if tile not in (16, 32, 64, 128) or extent % tile == 0:
        raise ValueError("tail preparation requires a supported tile and a nonzero remainder")
    validate_tail_spans(batch, extent, tile)
    half = tile // 2
    remainder = extent % tile
    last_start = extent - remainder

    @T.prim_func
    def main(
        A: T.Tensor((batch, extent, 512), dtype),
        Lo: T.Tensor((batch, extent, 512), dtype),
        Aux: T.Tensor((batch, extent, 64), dtype),
        APadded: T.Tensor((batch, tile, 512), dtype),
        LoPadded: T.Tensor((batch, tile, 512), dtype),
        AuxPadded: T.Tensor((batch, tile, 64), dtype),
    ):
        with T.Kernel(batch, is_npu=True) as (bid, vid):
            row0 = vid * half
            valid = T.alloc_var("int32", init=0)
            if remainder > row0:
                valid = remainder - row0
                if valid > half:
                    valid = half
            a = T.alloc_ub((half, 512), dtype)
            lo = T.alloc_ub((half, 512), dtype)
            aux = T.alloc_ub((half, 64), dtype)
            T.annotate_address({a: 0, lo: half * 512 * 2, aux: half * 1024 * 2})
            with T.Scope("V"):
                T.tile.fill(a, 0.0)
                T.tile.fill(lo, 0.0)
                T.tile.fill(aux, 0.0)
                T.barrier_all()
                if valid > 0:
                    T.copy(A[bid, last_start + row0 : last_start + row0 + valid, :], a[0:valid, :])
                    T.copy(Lo[bid, last_start + row0 : last_start + row0 + valid, :], lo[0:valid, :])
                    T.copy(Aux[bid, last_start + row0 : last_start + row0 + valid, :], aux[0:valid, :])
                    T.barrier_all()
                T.copy(a, APadded[bid, row0 : row0 + half, :])
                T.copy(lo, LoPadded[bid, row0 : row0 + half, :])
                T.copy(aux, AuxPadded[bid, row0 : row0 + half, :])
                T.barrier_all()

    return tilelang.compile(main, out_idx=[3, 4, 5], target=target, pass_configs=EXPERT_PASSES)


def cube_prefill(
    ql,
    qa,
    cache,
    auxiliary_cache,
    metadata,
    *,
    block_m=32,
    block_n=64,
    target="pto",
    max_blocks=24,
    scale=None,
    ql_residual=None,
    return_kernels=False,
):
    """One persistent Cube attention launch for all query/head rows.

    The only host tensor transformations are contiguous, storage-preserving views.
    CPU owner validates metadata values; this entry checks shape/dtype/device only.
    """
    import torch

    scale = _validated_scale(scale)
    if target not in ("pto", "ascendc"):
        raise ValueError("target must be 'pto' or 'ascendc'")
    inputs = (ql, qa, cache, auxiliary_cache)
    if any(not isinstance(value, torch.Tensor) for value in inputs):
        raise TypeError("QL/QA/C/E must be tensors")
    if ql.ndim != 4 or qa.ndim != 4 or cache.ndim != 3 or auxiliary_cache.ndim != 3:
        raise ValueError("QL/QA must be token-major rank4; C/E must be rank3")
    batch, query_capacity, heads, width = ql.shape
    capacity = cache.shape[1]
    if min(batch, query_capacity, heads, capacity) < 1 or width != 512:
        raise ValueError("C4 requires positive B/Tq/H/Tk and latent width512")
    shapes = (
        (batch, query_capacity, heads, 512),
        (batch, query_capacity, heads, 64),
        (batch, capacity, 512),
        (batch, capacity, 64),
    )
    if ql.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("C4 inputs must be float16 or bfloat16")
    if ql.device.type not in ("npu", "privateuseone"):
        raise ValueError("C4 requires an NPU")
    for name, value, shape in zip(("QL", "QA", "C", "E"), inputs, shapes, strict=True):
        if tuple(value.shape) != shape or value.dtype != ql.dtype or value.device != ql.device or not value.is_contiguous():
            raise ValueError(f"{name} shape/dtype/device/contiguity violates the C4 contract")
    if not isinstance(metadata, torch.Tensor) or tuple(metadata.shape) != (batch, 8):
        raise ValueError("metadata v1 must have shape[B,8]")
    if metadata.dtype != torch.int32 or metadata.device != ql.device or not metadata.is_contiguous():
        raise ValueError("metadata v1 must be contiguous int32 on the input device")
    rows = query_capacity * heads
    if ql_residual is not None:
        if not isinstance(ql_residual, torch.Tensor):
            raise TypeError("ql_residual must be a Tensor or None")
        if (
            tuple(ql_residual.shape) != tuple(ql.shape)
            or ql_residual.dtype != ql.dtype
            or ql_residual.device != ql.device
            or not ql_residual.is_contiguous()
        ):
            raise ValueError("ql_residual must match QL shape/dtype/device and be contiguous")
    has_ql_residual = ql_residual is not None
    dtype = "float16" if ql.dtype == torch.float16 else "bfloat16"
    kernel = build_cube_attention(
        batch=batch,
        heads=heads,
        query_capacity=query_capacity,
        cache_capacity=capacity,
        block_m=block_m,
        block_n=block_n,
        dtype=dtype,
        target=target,
        max_blocks=max_blocks,
        scale=scale,
        has_ql_residual=has_ql_residual,
    )
    kernels = {"attention": kernel}
    q_flat = ql.view(batch, rows, 512)
    qa_flat = qa.view(batch, rows, 64)
    q_low_flat = q_flat if ql_residual is None else ql_residual.view(batch, rows, 512)
    q_tail, q_low_tail, qa_tail = q_flat, q_low_flat, qa_flat
    c_tail, e_tail = cache, auxiliary_cache
    if rows % block_m:
        if has_ql_residual:
            prepare = build_tail_triple(batch=batch, extent=rows, tile=block_m, dtype=dtype, target=target)
            q_tail, q_low_tail, qa_tail = prepare(q_flat, q_low_flat, qa_flat)
            kernels["query_tail_triple"] = prepare
        else:
            prepare = build_tail_pair(batch=batch, extent=rows, tile=block_m, dtype=dtype, target=target)
            q_tail, qa_tail = prepare(q_flat, qa_flat)
            q_low_tail = q_tail
            kernels["query_tail"] = prepare
    if capacity % block_n:
        prepare = build_tail_pair(batch=batch, extent=capacity, tile=block_n, dtype=dtype, target=target)
        c_tail, e_tail = prepare(cache, auxiliary_cache)
        kernels["cache_tail"] = prepare
    flat_result = kernel(q_flat, qa_flat, cache, auxiliary_cache, metadata, q_tail, qa_tail, c_tail, e_tail, q_low_flat, q_low_tail)
    result = flat_result.view(batch, query_capacity, heads, 512)
    return (result, kernels) if return_kernels else result
