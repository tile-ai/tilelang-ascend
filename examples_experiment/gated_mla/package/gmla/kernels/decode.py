"""Isolated C3 split-KV decode with FP32 partial states and merge; see design.md.

No NPU code runs at import. This is not wired into the released operator.
QL uses the input dtype; BF16 P uses signed hi/residual PV accumulation.
All GEMM outputs, online state and final Z are FP32.
"""

from functools import lru_cache
import math

import tilelang
from tilelang import language as T

from .._contracts import validate_attention_spans, validate_tail_spans, validate_split_spans


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


def buffer_plan(block_m=32, block_n=64, *, batch=1, heads=96, dtype="float16", has_ql_residual=False, split_count=6, max_blocks=20):
    """Pure host metadata: disjoint, 32-byte-aligned addresses per memory scope."""
    if block_m not in (16, 32) or block_n not in (64, 128):
        raise ValueError("Cube decode supports block_m in (16,32), block_n in (64,128)")
    if type(has_ql_residual) is not bool:
        raise TypeError("has_ql_residual must be bool")
    if type(split_count) is not int or split_count not in (1, 2, 3, 4, 6, 8):
        raise ValueError("split_count must be one of1/2/3/4/6/8")
    if type(max_blocks) is not int or not 1 <= max_blocks <= 24:
        raise ValueError("max_blocks must be in[1,24]")
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
    result = {}
    for scope, buffers in sizes.items():
        end = 0
        rows = {}
        for name, count in buffers.items():
            start = (end + 31) // 32 * 32
            rows[name] = {"offset": start, "bytes": count, "end": start + count}
            end = start + count
        result[scope] = {"buffers": rows, "total_bytes": end}
    tasks = batch * ((heads + block_m - 1) // block_m) * split_count
    blocks = min(max_blocks, tasks)
    result["logical_tasks"] = tasks
    result["physical_blocks"] = blocks
    result["waves"] = (tasks + blocks - 1) // blocks
    result["split_count"] = split_count
    result["partial_output_bytes"] = tasks * block_m * (512 + 2) * 4
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
    cache_capacity=128,
    block_m=32,
    block_n=64,
    target="pto",
    dtype="float16",
    has_ql_residual=False,
    split_count=6,
    max_blocks=20,
):
    """Compile decode shapes; runtime metadata VALUES are not factory arguments.

    Tail aliases have original tensor shapes on aligned axes. The production
    entry below launches only necessary NPU tail preparation, then this kernel.
    """
    if target not in ("pto", "ascendc"):
        raise ValueError("target must be 'pto' or 'ascendc'")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError("dtype must be 'float16' or 'bfloat16'")
    if any(type(value) is not int or value < 1 for value in (batch, heads, cache_capacity)):
        raise ValueError("batch, heads and cache_capacity must be positive integers")
    validate_attention_spans(
        batch, 1, heads, cache_capacity, decode=True, block_m=block_m, block_n=block_n, split_count=split_count, max_blocks=max_blocks
    )
    plan = buffer_plan(
        block_m,
        block_n,
        batch=batch,
        heads=heads,
        dtype=dtype,
        has_ql_residual=has_ql_residual,
        split_count=split_count,
        max_blocks=max_blocks,
    )
    l1 = {name: row["offset"] for name, row in plan["L1"]["buffers"].items()}
    q_lo_offset = l1.get("q_lo_l1", 0)
    l0c = {name: row["offset"] for name, row in plan["L0C"]["buffers"].items()}
    ub = {name: row["offset"] for name, row in plan["UB"]["buffers"].items()}
    use_p_residual = dtype == "bfloat16"
    probability_columns = plan["probability_columns"]
    prob_round_offset = ub.get("prob_round_ub", plan["UB"]["total_bytes"])
    head_tiles = (heads + block_m - 1) // block_m
    tasks = batch * head_tiles * split_count
    if tasks >= 2**31 - 24 or cache_capacity >= 2**31 - block_n:
        raise ValueError("task and cache bounds must fit int32")
    blocks = min(max_blocks, tasks)
    waves = (tasks + blocks - 1) // blocks
    head_remainder = heads % block_m
    cache_remainder = cache_capacity % block_n
    head_tail_extent = block_m if head_remainder else heads
    cache_tail_extent = block_n if cache_remainder else cache_capacity
    half_m = block_m // 2
    iterations = (cache_capacity + block_n - 1) // block_n
    tiles_per_split = (iterations + split_count - 1) // split_count
    scale = 1.0 / math.sqrt(192)

    def allocate_query_residual(q_hi_buffer):
        # Returning the existing Buffer allocates nothing for the static None path.
        if has_ql_residual:
            return T.alloc_L1((block_m, 512), dtype)
        return q_hi_buffer

    @T.prim_func
    def main(
        QL: T.Tensor((batch, heads, 512), dtype),
        QA: T.Tensor((batch, heads, 64), dtype),
        C: T.Tensor((batch, cache_capacity, 512), dtype),
        E: T.Tensor((batch, cache_capacity, 64), dtype),
        Metadata: T.Tensor((batch, 8), "int32"),
        QTail: T.Tensor((batch, head_tail_extent, 512), dtype),
        QATail: T.Tensor((batch, head_tail_extent, 64), dtype),
        CTail: T.Tensor((batch, cache_tail_extent, 512), dtype),
        ETail: T.Tensor((batch, cache_tail_extent, 64), dtype),
        QLo: T.Tensor((batch, heads, 512), dtype),
        QLoTail: T.Tensor((batch, head_tail_extent, 512), dtype),
        PartialMax: T.Tensor((tasks, block_m), "float32"),
        PartialSum: T.Tensor((tasks, block_m), "float32"),
        PartialZ: T.Tensor((tasks, block_m, 512), "float32"),
        score_workspace: T.Tensor((blocks, block_m, block_n), "float32"),
        probability_workspace: T.Tensor((blocks, block_m, probability_columns), dtype),
        partial_workspace: T.Tensor((blocks, block_m, 512), "float32"),
    ):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            row0 = vid * half_m
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
                }
            )

            if has_ql_residual:
                T.annotate_address({q_lo_l1: q_lo_offset})

            # Same-group, single-buffer four-flag protocol from manual FA.
            with T.Scope("C"):
                for wave_c in T.serial(waves):
                    c_task = wave_c * blocks + cid
                    if c_task < tasks:
                        c_split = c_task % split_count
                        c_head = (c_task // split_count) % head_tiles
                        c_bid = c_task // (split_count * head_tiles)
                        c_h0 = c_head * block_m
                        c_visible = T.min(Metadata[c_bid, 1], Metadata[c_bid, 2] + 1)
                        if (Metadata[c_bid, 0] == 1) & (c_split * tiles_per_split * block_n < c_visible):
                            if head_remainder != 0 and c_h0 + block_m > heads:
                                T.copy(QTail[c_bid, :, :], q_l1)
                                T.copy(QATail[c_bid, :, :], qa_l1)
                                if has_ql_residual:
                                    T.copy(QLoTail[c_bid, :, :], q_lo_l1)
                            else:
                                T.copy(QL[c_bid, c_h0 : c_h0 + block_m, :], q_l1)
                                T.copy(QA[c_bid, c_h0 : c_h0 + block_m, :], qa_l1)
                                if has_ql_residual:
                                    T.copy(QLo[c_bid, c_h0 : c_h0 + block_m, :], q_lo_l1)
                            T.barrier_all()
                            for local_k in T.serial(tiles_per_split):
                                k = c_split * tiles_per_split + local_k
                                if (k < iterations) & (k * block_n < c_visible):
                                    if cache_remainder != 0 and k == iterations - 1:
                                        T.copy(CTail[c_bid, :, :], c_l1)
                                        T.copy(ETail[c_bid, :, :], e_l1)
                                    else:
                                        T.copy(C[c_bid, k * block_n : (k + 1) * block_n, :], c_l1)
                                        T.copy(E[c_bid, k * block_n : (k + 1) * block_n, :], e_l1)
                                    T.barrier_all()
                                    T.gemm_v0(q_l1, c_l1, score_l0c, transpose_B=True, init=True, kL0Size=64)
                                    T.barrier_all()
                                    if has_ql_residual:
                                        T.gemm_v0(q_lo_l1, c_l1, score_l0c, transpose_B=True, init=False, kL0Size=64)
                                        T.barrier_all()
                                    T.gemm_v0(qa_l1, e_l1, score_l0c, transpose_B=True, init=False, kL0Size=64)
                                    T.barrier_all()
                                    T.copy(score_l0c, score_workspace[cid, :, :])
                                    T.barrier_all()
                                    T.set_cross_flag("FIX", 0)

                                    T.wait_cross_flag(1)
                                    T.barrier_all()
                                    if use_p_residual:
                                        T.copy(probability_workspace[cid, :, 0:block_n], p_l1)
                                    else:
                                        T.copy(probability_workspace[cid, :, :], p_l1)
                                    T.barrier_all()
                                    # C stays resident from QK through PV; no second GM load.
                                    T.gemm_v0(p_l1, c_l1, partial_l0c, init=True, kL0Size=16)
                                    T.barrier_all()
                                    if use_p_residual:
                                        T.copy(probability_workspace[cid, :, block_n : 2 * block_n], p_l1)
                                        T.barrier_all()
                                        T.gemm_v0(p_l1, c_l1, partial_l0c, init=False, kL0Size=16)
                                        T.barrier_all()
                                    T.copy(partial_l0c, partial_workspace[cid, :, :])
                                    T.barrier_all()
                                    T.set_cross_flag("FIX", 2)
                                    T.wait_cross_flag(3)
                                    T.barrier_all()

            with T.Scope("V"):
                for wave_v in T.serial(waves):
                    v_task = wave_v * blocks + cid
                    if v_task < tasks:
                        v_split = v_task % split_count
                        v_head = (v_task // split_count) % head_tiles  # noqa: F841 -- preserve the measured kernel AST
                        v_bid = v_task // (split_count * head_tiles)
                        v_visible = T.min(Metadata[v_bid, 1], Metadata[v_bid, 2] + 1)
                        T.tile.fill(z, 0.0)
                        T.tile.fill(normalizer, 0.0)
                        T.tile.fill(maximum, -T.infinity("float32"))
                        T.barrier_all()
                        if (Metadata[v_bid, 0] == 1) & (v_split * tiles_per_split * block_n < v_visible):
                            for local_k in T.serial(tiles_per_split):
                                _k = v_split * tiles_per_split + local_k
                                if (_k < iterations) & (_k * block_n < v_visible):
                                    T.copy(maximum, previous)
                                    T.barrier_all()
                                    T.wait_cross_flag(0)
                                    T.copy(score_workspace[cid, row0 : row0 + half_m, :], scores)
                                    T.barrier_all()
                                    T.tile.mul(scores, scores, scale)
                                    T.barrier_all()
                                    if (_k + 1) * block_n > v_visible:
                                        for i, j in T.Parallel(half_m, block_n):
                                            scores[i, j] = T.if_then_else(
                                                _k * block_n + j < v_visible, scores[i, j], -T.infinity("float32")
                                            )
                                    T.barrier_all()
                                    T.reduce_max(scores, maximum, dim=-1)
                                    T.barrier_all()
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

                        # Logical-task outputs survive later physical-slot waves.
                        T.copy(maximum, PartialMax[v_task, row0 : row0 + half_m])
                        T.copy(normalizer, PartialSum[v_task, row0 : row0 + half_m])
                        T.copy(z, PartialZ[v_task, row0 : row0 + half_m, :])
                        T.barrier_all()

    return tilelang.compile(
        main,
        out_idx=[11, 12, 13],
        workspace_idx=[14, 15, 16],
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


def cube_decode(
    ql,
    qa,
    cache,
    auxiliary_cache,
    metadata,
    *,
    block_m=32,
    block_n=64,
    target="pto",
    ql_residual=None,
    split_count=6,
    max_blocks=20,
    return_kernels=False,
    return_partials=False,
):
    """Metadata-only host dispatch; values/lengths stay on the device.

    CPU owner must validate metadata v1 values before transfer. Shapes, dtypes,
    devices and contiguity are checked here without reading NPU tensor data.
    All physical input values are finite under the C3 contract.
    """
    import torch

    if target not in ("pto", "ascendc"):
        raise ValueError("target must be 'pto' or 'ascendc'")
    inputs = (ql, qa, cache, auxiliary_cache)
    if any(not isinstance(value, torch.Tensor) or value.ndim != 3 for value in inputs):
        raise ValueError("QL/QA/C/E must be rank-3 tensors")
    batch, heads, width = ql.shape
    capacity = cache.shape[1]
    if batch < 1 or heads < 1 or capacity < 1 or width != 512:
        raise ValueError("C3 requires positive B/H/Tk and latent width512")
    shapes = ((batch, heads, 512), (batch, heads, 64), (batch, capacity, 512), (batch, capacity, 64))
    if ql.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("C3 inputs must be float16 or bfloat16")
    if ql.device.type not in ("npu", "privateuseone"):
        raise ValueError("C3 requires an NPU")
    for name, value, shape in zip(("QL", "QA", "C", "E"), inputs, shapes, strict=True):
        if tuple(value.shape) != shape or value.dtype != ql.dtype or value.device != ql.device or not value.is_contiguous():
            raise ValueError(f"{name} shape/dtype/device/contiguity violates the C3 contract")
    if not isinstance(metadata, torch.Tensor) or tuple(metadata.shape) != (batch, 8):
        raise ValueError("metadata v1 must have shape[B,8]")
    if metadata.dtype != torch.int32 or metadata.device != ql.device or not metadata.is_contiguous():
        raise ValueError("metadata v1 must be contiguous int32 on the input device")
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
        cache_capacity=capacity,
        block_m=block_m,
        block_n=block_n,
        dtype=dtype,
        target=target,
        has_ql_residual=has_ql_residual,
        split_count=split_count,
        max_blocks=max_blocks,
    )
    kernels = {}
    q_low = ql if ql_residual is None else ql_residual
    q_tail, q_low_tail, qa_tail = ql, q_low, qa
    c_tail, e_tail = cache, auxiliary_cache
    if heads % block_m:
        if has_ql_residual:
            prepare = build_tail_triple(batch=batch, extent=heads, tile=block_m, dtype=dtype, target=target)
            q_tail, q_low_tail, qa_tail = prepare(ql, q_low, qa)
            kernels["head_tail_triple"] = prepare
        else:
            prepare = build_tail_pair(batch=batch, extent=heads, tile=block_m, dtype=dtype, target=target)
            q_tail, qa_tail = prepare(ql, qa)
            q_low_tail = q_tail
            kernels["head_tail"] = prepare
    if capacity % block_n:
        prepare = build_tail_pair(batch=batch, extent=capacity, tile=block_n, dtype=dtype, target=target)
        c_tail, e_tail = prepare(cache, auxiliary_cache)
        kernels["cache_tail"] = prepare
    kernels["attention"] = kernel
    partial_max, partial_sum, partial_z = kernel(
        ql, qa, cache, auxiliary_cache, metadata, q_tail, qa_tail, c_tail, e_tail, q_low, q_low_tail
    )
    merge = build_split_merge(
        batch=batch,
        heads=heads,
        cache_capacity=capacity,
        block_m=block_m,
        block_n=block_n,
        split_count=split_count,
        max_blocks=max_blocks,
        target=target,
    )
    kernels["merge"] = merge
    result = merge(partial_max, partial_sum, partial_z, metadata)
    if return_partials:
        return result, {"maximum": partial_max, "normalizer": partial_sum, "weighted_latent": partial_z}, kernels
    return (result, kernels) if return_kernels else result


@lru_cache(maxsize=128, typed=True)
def build_split_merge(*, batch, heads, cache_capacity, block_m=32, block_n=64, split_count=6, max_blocks=20, target="pto"):
    """Merge FP32(m,l,z) partials; no max-to-zero clamp on valid logits."""
    if target not in ("pto", "ascendc"):
        raise ValueError("unsupported target")
    if any(type(value) is not int or value < 1 for value in (batch, heads, cache_capacity)):
        raise ValueError("batch/heads/cache_capacity must be positive integers")
    buffer_plan(block_m, block_n, batch=batch, heads=heads, split_count=split_count, max_blocks=max_blocks)
    validate_split_spans(batch, heads, block_m=block_m, split_count=split_count)
    if cache_capacity >= 2**31 - block_n:
        raise ValueError("cache extent must fit signed int32 split metadata arithmetic")
    head_tiles = (heads + block_m - 1) // block_m
    head_tasks = batch * head_tiles
    tasks = head_tasks * split_count
    blocks = min(max_blocks, head_tasks)
    waves = (head_tasks + blocks - 1) // blocks
    iterations = (cache_capacity + block_n - 1) // block_n
    tiles_per_split = (iterations + split_count - 1) // split_count
    half_m = block_m // 2
    tile_bytes = half_m * 512 * 4
    vector_bytes = half_m * 4

    @T.prim_func
    def main(
        PartialMax: T.Tensor((tasks, block_m), "float32"),
        PartialSum: T.Tensor((tasks, block_m), "float32"),
        PartialZ: T.Tensor((tasks, block_m, 512), "float32"),
        Metadata: T.Tensor((batch, 8), "int32"),
        Z: T.Tensor((batch, heads, 512), "float32"),
    ):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            row0 = vid * half_m
            valid_rows = T.alloc_var("int32", init=0)
            z = T.alloc_ub((half_m, 512), "float32")
            partial = T.alloc_ub((half_m, 512), "float32")
            global_max = T.alloc_ub((half_m,), "float32")
            part_max = T.alloc_ub((half_m,), "float32")
            part_sum = T.alloc_ub((half_m,), "float32")
            alpha = T.alloc_ub((half_m,), "float32")
            normalizer = T.alloc_ub((half_m,), "float32")
            T.annotate_address(
                {
                    z: 0,
                    partial: tile_bytes,
                    global_max: 2 * tile_bytes,
                    part_max: 2 * tile_bytes + vector_bytes,
                    part_sum: 2 * tile_bytes + 2 * vector_bytes,
                    alpha: 2 * tile_bytes + 3 * vector_bytes,
                    normalizer: 2 * tile_bytes + 4 * vector_bytes,
                }
            )
            with T.Scope("V"):
                for wave in T.serial(waves):
                    head_task = wave * blocks + cid
                    if head_task < head_tasks:
                        bid = head_task // head_tiles
                        h0 = (head_task % head_tiles) * block_m
                        visible = T.min(Metadata[bid, 1], Metadata[bid, 2] + 1)
                        valid_rows = 0
                        if heads > h0 + row0:
                            valid_rows = heads - h0 - row0
                            if valid_rows > half_m:
                                valid_rows = half_m
                        T.tile.fill(z, 0.0)
                        T.tile.fill(normalizer, 0.0)
                        T.tile.fill(global_max, -T.infinity("float32"))
                        T.barrier_all()
                        if (Metadata[bid, 0] == 1) & (visible > 0):
                            # Empty splits are skipped before max/exp. For a
                            # valid query split0 owns key0, so global_max is finite.
                            for split in T.serial(split_count):
                                if (split * tiles_per_split < iterations) & (split * tiles_per_split * block_n < visible):
                                    task = head_task * split_count + split
                                    T.copy(PartialMax[task, row0 : row0 + half_m], part_max)
                                    T.barrier_all()
                                    T.tile.max(global_max, global_max, part_max)
                                    T.barrier_all()
                            for split in T.serial(split_count):
                                if (split * tiles_per_split < iterations) & (split * tiles_per_split * block_n < visible):
                                    task = head_task * split_count + split
                                    T.copy(PartialMax[task, row0 : row0 + half_m], part_max)
                                    T.copy(PartialSum[task, row0 : row0 + half_m], part_sum)
                                    T.barrier_all()
                                    T.tile.sub(part_max, part_max, global_max)
                                    T.barrier_all()
                                    T.tile.exp(alpha, part_max)
                                    T.barrier_all()
                                    T.tile.mul(part_sum, part_sum, alpha)
                                    T.barrier_all()
                                    T.tile.add(normalizer, normalizer, part_sum)
                                    T.barrier_all()
                                    T.copy(PartialZ[task, row0 : row0 + half_m, :], partial)
                                    T.barrier_all()
                                    for row in T.serial(half_m):
                                        T.tile.mul(partial[row, :], partial[row, :], alpha[row])
                                        T.barrier_all()
                                    T.tile.add(z, z, partial)
                                    T.barrier_all()
                            for row in T.serial(half_m):
                                T.tile.div(z[row, :], z[row, :], normalizer[row])
                                T.barrier_all()
                        if valid_rows > 0:
                            T.copy(z[0:valid_rows, :], Z[bid, h0 + row0 : h0 + row0 + valid_rows, :])
                            T.barrier_all()

    return tilelang.compile(main, out_idx=[4], target=target, pass_configs=EXPERT_PASSES)
