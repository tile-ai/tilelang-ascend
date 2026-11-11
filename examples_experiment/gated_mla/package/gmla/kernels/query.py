"""Scoped Q2 Cube projection: real-head decode tiles and same-head prefill tiles."""

from functools import lru_cache
from types import ModuleType

import torch
import tilelang
import tilelang.language as T


EXPERT_PASSES = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}


def plan(batch, query_capacity, *, tile_m=None, blocks=20, residual=True, packed=False):
    if type(batch) is not int or batch not in (1, 2, 4) or type(query_capacity) is not int or query_capacity < 1:
        raise ValueError("Q2 supports B1/2/4 and positive Tq, with H96")
    if batch * query_capacity * 96 * 512 > 2**31 - 1:
        raise ValueError("Q2 output element indexing exceeds int32")
    if blocks not in (20, 24) or type(blocks) is not int:
        raise ValueError("blocks must be explicitly20 or24")
    if type(residual) is not bool or type(packed) is not bool:
        raise TypeError("residual and packed must be bool")
    prefill = query_capacity >= 16
    if tile_m is None:
        tile_m = 32 if query_capacity >= 32 else 16
    if tile_m not in (16, 32) or (not prefill and tile_m != 16) or (prefill and tile_m > query_capacity):
        raise ValueError("Use M16 for small queries; prefill M16/M32 must fit real Tq")
    tasks = batch * 96 if prefill else batch * query_capacity * 96
    vector_rows, vector_cols = (tile_m // 2, 512) if prefill else (1, 256)
    q_bytes = tile_m * 128 * 2
    value_bytes = vector_rows * vector_cols * 4
    high_bytes = vector_rows * vector_cols * 2
    qa_bytes = vector_rows * 64 * 2
    return {
        "B": batch,
        "Tq": query_capacity,
        "H": 96,
        "prefill": prefill,
        "tile_m": tile_m,
        "blocks": blocks,
        "tasks": tasks,
        "rounds": (tasks + blocks - 1) // blocks,
        "query_tiles": (query_capacity + tile_m - 1) // tile_m if prefill else 1,
        "vector_rows": vector_rows,
        "vector_cols": vector_cols,
        "query_width": 192 if packed else 128,
        "weight_rows": 256 if packed else 128,
        "L1_bytes": q_bytes + 128 * 512 * 2,
        "L0C_bytes": tile_m * 512 * 4,
        "UB_bytes": 2 * value_bytes + 2 * high_bytes + qa_bytes,
        "UB_offsets": {
            "value": 0,
            "hi": value_bytes,
            "roundtrip": value_bytes + high_bytes,
            "lo": 2 * value_bytes + high_bytes,
            "qa": 2 * value_bytes + 2 * high_bytes,
        },
        "result_workspace_bytes": blocks * tile_m * 512 * 4,
        "unused_private_workspace_bytes": (0 if residual else batch * query_capacity * 96 * 512 * 2)
        + (0 if packed else batch * query_capacity * 96 * 64 * 2),
    }


@lru_cache(maxsize=128)
def _compile(batch, tq, dtype, target, residual, packed, tile_m, blocks):
    p = plan(batch, tq, tile_m=tile_m, blocks=blocks, residual=residual, packed=packed)
    prefill, tasks, rounds, tiles = (p[key] for key in ("prefill", "tasks", "rounds", "query_tiles"))
    vr, vc = p["vector_rows"], p["vector_cols"]
    qw, wr = p["query_width"], p["weight_rows"]
    q_bytes, offsets = tile_m * 128 * 2, p["UB_offsets"]
    task_batch_stride = 96 if prefill else tq * 96
    full_tiles = tq // tile_m if prefill else 1
    tail = tq % tile_m if prefill else 0
    last_load = max(0, tq - tile_m)
    qa_start = 128 if packed else 0  # dead nonpacked branch remains in-bounds for parsing.

    @T.prim_func
    def main(
        Q: T.Tensor((batch, tq, 96, qw), dtype),
        W: T.Tensor((96, wr, 512), dtype),
        Hi: T.Tensor((batch, tq, 96, 512), dtype),
        Lo: T.Tensor((batch, tq, 96, 512), dtype),
        QA: T.Tensor((batch, tq, 96, 64), dtype),
        Workspace: T.Tensor((blocks, tile_m, 512), "float32"),
    ):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1((tile_m, 128), dtype)
            w_l1 = T.alloc_L1((128, 512), dtype)
            acc = T.alloc_L0C((tile_m, 512), "float32")
            value = T.alloc_ub((vr, vc), "float32")
            hi = T.alloc_ub((vr, vc), dtype)
            roundtrip = T.alloc_ub((vr, vc), "float32")
            lo = T.alloc_ub((vr, vc), dtype)
            qa = T.alloc_ub((vr, 64), dtype)
            T.annotate_address(
                {
                    q_l1: 0,
                    w_l1: q_bytes,
                    acc: 0,
                    value: offsets["value"],
                    hi: offsets["hi"],
                    roundtrip: offsets["roundtrip"],
                    lo: offsets["lo"],
                    qa: offsets["qa"],
                }
            )
            with T.Scope("C"):
                for turn in T.serial(rounds):
                    task = cid + turn * blocks
                    if task < tasks:
                        b = task // task_batch_stride
                        head = task % 96
                        token = (task // 96) % tq
                        T.copy(W[head, 0:128, 0:512], w_l1)
                        T.barrier_all()
                        for qtile in T.serial(tiles):
                            load_t = T.if_then_else(qtile < full_tiles, qtile * tile_m, last_load)
                            if prefill:
                                T.copy(Q[b, load_t : load_t + tile_m, head, 0:128], q_l1)
                            else:
                                first_head = (head // tile_m) * tile_m
                                T.copy(Q[b, token, first_head : first_head + tile_m, 0:128], q_l1)
                            T.barrier_all()
                            T.gemm_v0(q_l1, w_l1, acc, init=True, kL0Size=64)
                            T.barrier_all()
                            T.copy(acc, Workspace[cid, 0:tile_m, 0:512])
                            T.barrier_all()
                            T.set_cross_flag("FIX", 0)
                            T.wait_cross_flag(1)
                            T.barrier_all()
            with T.Scope("V"):
                for turn in T.serial(rounds):
                    task = cid + turn * blocks
                    if task < tasks:
                        b = task // task_batch_stride
                        head = task % 96
                        token = (task // 96) % tq
                        for qtile in T.serial(tiles):
                            load_t = T.if_then_else(qtile < full_tiles, qtile * tile_m, last_load)
                            row0 = vid * vr if prefill else head % tile_m
                            column = 0 if prefill else vid * vc
                            T.wait_cross_flag(0)
                            T.barrier_all()
                            T.copy(Workspace[cid, row0 : row0 + vr, column : column + vc], value)
                            T.barrier_all()
                            T.copy(value, hi)
                            T.barrier_all()
                            if residual:
                                T.copy(hi, roundtrip)
                                T.barrier_all()
                                T.tile.sub(roundtrip, value, roundtrip)
                                T.barrier_all()
                                T.copy(roundtrip, lo)
                                T.barrier_all()
                            if prefill:
                                local_start = T.alloc_var("int32", init=0)
                                valid_rows = T.alloc_var("int32", init=vr)
                                local_start = 0
                                valid_rows = vr
                                if tail:  # noqa: SIM102 -- static specialization before symbolic tile predicate
                                    if qtile == full_tiles:
                                        if vid * vr < tile_m - tail:
                                            local_start = tile_m - tail - vid * vr
                                        if local_start > vr:
                                            local_start = vr
                                        valid_rows = vr - local_start
                                if valid_rows > 0:
                                    output_t = load_t + vid * vr + local_start
                                    T.copy(hi[local_start : local_start + valid_rows, :], Hi[b, output_t : output_t + valid_rows, head, :])
                                    T.barrier_all()
                                    if residual:
                                        T.copy(
                                            lo[local_start : local_start + valid_rows, :], Lo[b, output_t : output_t + valid_rows, head, :]
                                        )
                                        T.barrier_all()
                                    if packed:
                                        T.copy(Q[b, output_t : output_t + valid_rows, head, qa_start : qa_start + 64], qa[0:valid_rows, :])
                                        T.barrier_all()
                                        T.copy(qa[0:valid_rows, :], QA[b, output_t : output_t + valid_rows, head, :])
                                        T.barrier_all()
                            else:
                                T.copy(hi[0, :], Hi[b, token, head, column])
                                T.barrier_all()
                                if residual:
                                    T.copy(lo[0, :], Lo[b, token, head, column])
                                    T.barrier_all()
                                if packed:  # noqa: SIM102 -- static packed branch before symbolic V-lane predicate
                                    if vid == 0:
                                        T.copy(Q[b, token, head, qa_start : qa_start + 64], qa[0, :])
                                        T.barrier_all()
                                        T.copy(qa[0, :], QA[b, token, head, 0])
                                        T.barrier_all()
                            # Even a V partner with zero tail output rows must acknowledge.
                            T.set_cross_flag("MTE3", 1)
                            T.barrier_all()

    outputs = [2] + ([3] if residual else []) + ([4] if packed else [])
    workspaces = [index for index in (3, 4, 5) if index not in outputs]
    return tilelang.compile(main, out_idx=outputs, workspace_idx=workspaces, target=target, pass_configs=EXPERT_PASSES)


def build_query_cube(batch, query_capacity, *, dtype="bfloat16", target="pto", residual=True, packed=False, tile_m=None, blocks=20):
    if dtype not in ("float16", "bfloat16") or target not in ("pto", "ascendc"):
        raise ValueError("unsupported target or dtype")
    p = plan(batch, query_capacity, tile_m=tile_m, blocks=blocks, residual=residual, packed=packed)
    return _compile(batch, query_capacity, dtype, target, residual, packed, p["tile_m"], blocks)


def eligible_shape(B, Tq, H, dtype):
    return B in (1, 2, 4) and Tq >= 1 and H == 96 and dtype in ("float16", "bfloat16")


def _eligible(q, w, packed):
    width, rows = (192, 256) if packed else (128, 128)
    return (
        isinstance(q, torch.Tensor)
        and isinstance(w, torch.Tensor)
        and q.ndim == 4
        and q.shape[0] in (1, 2, 4)
        and q.shape[1] >= 1
        and tuple(q.shape[2:]) == (96, width)
        and tuple(w.shape) == (96, rows, 512)
        and q.dtype in (torch.float16, torch.bfloat16)
        and w.dtype == q.dtype
        and q.device.type in ("npu", "privateuseone")
        and w.device == q.device
        and q.is_contiguous()
        and w.is_contiguous()
    )


def make_provider(vector_projections, vector_query_residuals, vector_packed_projections, *, blocks=20, prefill_tile_m=None):
    """Explicit injection; other heads/batches retain the old Vector implementations."""
    provider = ModuleType("query_cube_family_provider")
    provider.__file__ = __file__
    provider.source_files = {
        "cube": __file__,
        "vector": vector_projections.__file__,
        "residual": vector_query_residuals.__file__,
        "packed": vector_packed_projections.__file__,
    }

    def selected(B, Tq, H, dtype, target, residual, packed, fallback):
        if not eligible_shape(B, Tq, H, dtype):
            return fallback(B, Tq, H, dtype, target=target)
        m = prefill_tile_m if Tq >= 16 else 16
        return build_query_cube(B, Tq, dtype=dtype, target=target, residual=residual, packed=packed, tile_m=m, blocks=blocks)

    def q_absorb_kernel(B, Tq, H, dtype="float16", *, target="pto"):
        return selected(B, Tq, H, dtype, target, False, False, vector_projections.q_absorb_kernel)

    def q_absorb_residual_kernel(B, Tq, H, dtype="float16", *, target="pto"):
        return selected(B, Tq, H, dtype, target, True, False, vector_query_residuals.q_absorb_residual_kernel)

    def q_absorb_packed_kernel(B, Tq, H, dtype="float16", *, target="pto"):
        return selected(B, Tq, H, dtype, target, False, True, vector_packed_projections.q_absorb_packed_kernel)

    def q_absorb_packed_residual_kernel(B, Tq, H, dtype="float16", *, target="pto"):
        return selected(B, Tq, H, dtype, target, True, True, vector_query_residuals.q_absorb_packed_residual_kernel)

    def call(q, w, target, residual, packed, fallback):
        if not _eligible(q, w, packed):
            return fallback(q, w, target=target)
        B, Tq, H, _ = map(int, q.shape)
        dtype = "float16" if q.dtype == torch.float16 else "bfloat16"
        m = prefill_tile_m if Tq >= 16 else 16
        return build_query_cube(B, Tq, dtype=dtype, target=target, residual=residual, packed=packed, tile_m=m, blocks=blocks)(q, w)

    def q_absorb(q, w, *, target="pto"):
        return call(q, w, target, False, False, vector_projections.q_absorb)

    def q_absorb_residual(q, w, *, target="pto"):
        hi, lo = call(q, w, target, True, False, vector_query_residuals.q_absorb_residual)
        return hi, lo

    def q_absorb_packed(q, w, *, target="pto"):
        hi, qa = call(q, w, target, False, True, vector_packed_projections.q_absorb_packed)
        return hi, qa

    def q_absorb_packed_residual(q, w, *, target="pto"):
        hi, lo, qa = call(q, w, target, True, True, vector_query_residuals.q_absorb_packed_residual)
        return hi, lo, qa

    provider.q_absorb, provider.q_absorb_kernel = q_absorb, q_absorb_kernel
    provider.q_absorb_residual, provider.q_absorb_residual_kernel = q_absorb_residual, q_absorb_residual_kernel
    provider.q_absorb_packed, provider.q_absorb_packed_kernel = q_absorb_packed, q_absorb_packed_kernel
    provider.q_absorb_packed_residual, provider.q_absorb_packed_residual_kernel = q_absorb_packed_residual, q_absorb_packed_residual_kernel
    provider.value_gate, provider.value_gate_kernel = vector_projections.value_gate, vector_projections.value_gate_kernel
    provider.value_gate_packed, provider.value_gate_packed_kernel = (
        vector_packed_projections.value_gate_packed,
        vector_packed_projections.value_gate_packed_kernel,
    )
    return provider
