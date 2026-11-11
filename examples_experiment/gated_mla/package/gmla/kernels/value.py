"""Independent R3 range candidate: scaled Hi, original-scale Lo, separate GEMMs."""

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


def plan(batch, query_capacity, *, tile_m=None, blocks=20, packed=False):
    if type(batch) is not int or batch not in (1, 2, 4) or type(query_capacity) is not int or query_capacity < 16:
        raise ValueError("CubeValue supports B1/2/4, H96 and Tq>=16")
    if batch * query_capacity * 96 * 512 > 2**31 - 1:
        raise ValueError("CubeValue input indexing exceeds signed int32")
    if type(blocks) is not int or blocks not in (20, 24):
        raise ValueError("blocks must be explicitly 20 or 24")
    if type(packed) is not bool:
        raise TypeError("packed must be bool")
    if tile_m is None:
        tile_m = 32 if query_capacity >= 32 else 16
    if type(tile_m) is not int or tile_m not in (16, 32) or tile_m > query_capacity:
        raise ValueError("M16/M32 must fit real Tq")
    vr = tile_m // 2
    allocations = {
        "z": vr * 512 * 4,
        "hi": vr * 512 * 2,
        "roundtrip": vr * 512 * 4,
        "lo": vr * 512 * 2,
        "value": vr * 128 * 4,
        "value_low": vr * 128 * 4,
        "gh": vr * 128 * 2,
        "gf": vr * 128 * 4,
        "exponent": vr * 128 * 4,
        "denominator": vr * 128 * 4,
        "numerator": vr * 128 * 4,
        "yh": vr * 128 * 2,
        "mask": 32,
        "select_tmp": 128 * 4,
    }
    offsets, ub_bytes = {}, 0
    for name, size in allocations.items():
        offsets[name] = ub_bytes
        ub_bytes += size
    tasks = batch * 96
    return {
        "B": batch,
        "Tq": query_capacity,
        "H": 96,
        "tile_m": tile_m,
        "blocks": blocks,
        "tasks": tasks,
        "rounds": (tasks + blocks - 1) // blocks,
        "query_tiles": (query_capacity + tile_m - 1) // tile_m,
        "vector_rows": vr,
        "weight_rows": 256 if packed else 128,
        "weight_row_start": 128 if packed else 0,
        "L1_bytes": 2 * tile_m * 512 * 2 + 128 * 512 * 2,
        "L0A_bytes_per_slot": tile_m * 64 * 2,
        "L0B_bytes_per_slot": 64 * 128 * 2,
        "L0C_bytes": tile_m * 128 * 4,
        "UB_bytes": ub_bytes,
        "UB_offsets": offsets,
        "UB_allocations": allocations,
        "workspace_bytes": blocks * tile_m * (2 * 512 * 2 + 2 * 128 * 4),
    }


@lru_cache(maxsize=128)
def _compile(batch, tq, dtype, target, packed, tile_m, blocks):
    p = plan(batch, tq, tile_m=tile_m, blocks=blocks, packed=packed)
    tasks, rounds, tiles, vr = (p[key] for key in ("tasks", "rounds", "query_tiles", "vector_rows"))
    wr, w0, offsets = p["weight_rows"], p["weight_row_start"], p["UB_offsets"]
    full_tiles, tail, last_load = tq // tile_m, tq % tile_m, tq - tile_m
    z_l1_bytes = tile_m * 512 * 2
    scale = 4.0 if dtype == "float16" else 2.0
    inverse_scale = 1.0 / scale

    @T.prim_func
    def main(
        Z: T.Tensor((batch, tq, 96, 512), "float32"),
        W: T.Tensor((96, wr, 512), dtype),
        G: T.Tensor((batch, tq, 96, 128), dtype),
        Y: T.Tensor((batch, tq, 96, 128), dtype),
        HighWorkspace: T.Tensor((blocks, tile_m, 512), dtype),
        LowWorkspace: T.Tensor((blocks, tile_m, 512), dtype),
        ValueWorkspace: T.Tensor((blocks, tile_m, 128), "float32"),
        LowValueWorkspace: T.Tensor((blocks, tile_m, 128), "float32"),
    ):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            hi_l1 = T.alloc_L1((tile_m, 512), dtype)
            lo_l1 = T.alloc_L1((tile_m, 512), dtype)
            w_l1 = T.alloc_L1((128, 512), dtype)
            acc = T.alloc_L0C((tile_m, 128), "float32")
            z = T.alloc_ub((vr, 512), "float32")
            hi = T.alloc_ub((vr, 512), dtype)
            roundtrip = T.alloc_ub((vr, 512), "float32")
            lo = T.alloc_ub((vr, 512), dtype)
            value = T.alloc_ub((vr, 128), "float32")
            value_low = T.alloc_ub((vr, 128), "float32")
            gh = T.alloc_ub((vr, 128), dtype)
            gf = T.alloc_ub((vr, 128), "float32")
            exponent = T.alloc_ub((vr, 128), "float32")
            denominator = T.alloc_ub((vr, 128), "float32")
            numerator = T.alloc_ub((vr, 128), "float32")
            yh = T.alloc_ub((vr, 128), dtype)
            mask = T.alloc_ub((32,), "uint8")
            select_tmp = T.alloc_ub((512,), "uint8")
            T.annotate_address(
                {
                    hi_l1: 0,
                    lo_l1: z_l1_bytes,
                    w_l1: 2 * z_l1_bytes,
                    acc: 0,
                    z: offsets["z"],
                    hi: offsets["hi"],
                    roundtrip: offsets["roundtrip"],
                    lo: offsets["lo"],
                    value: offsets["value"],
                    value_low: offsets["value_low"],
                    gh: offsets["gh"],
                    gf: offsets["gf"],
                    exponent: offsets["exponent"],
                    denominator: offsets["denominator"],
                    numerator: offsets["numerator"],
                    yh: offsets["yh"],
                    mask: offsets["mask"],
                    select_tmp: offsets["select_tmp"],
                }
            )
            with T.Scope("C"):
                for turn in T.serial(rounds):
                    task = cid + turn * blocks
                    if task < tasks:
                        head = task % 96
                        T.copy(W[head, w0 : w0 + 128, 0:512], w_l1)
                        T.barrier_all()
                        for _qtile in T.serial(tiles):
                            # Both AIV partners have written disjoint real rows.
                            T.wait_cross_flag(0)
                            T.barrier_all()
                            T.copy(HighWorkspace[cid, 0:tile_m, 0:512], hi_l1)
                            T.copy(LowWorkspace[cid, 0:tile_m, 0:512], lo_l1)
                            T.barrier_all()
                            T.gemm_v0(hi_l1, w_l1, acc, transpose_B=True, init=True, kL0Size=64)
                            T.barrier_all()
                            T.copy(acc, ValueWorkspace[cid, 0:tile_m, 0:128])
                            T.barrier_all()
                            T.gemm_v0(lo_l1, w_l1, acc, transpose_B=True, init=True, kL0Size=64)
                            T.barrier_all()
                            T.copy(acc, LowValueWorkspace[cid, 0:tile_m, 0:128])
                            T.barrier_all()
                            T.set_cross_flag("FIX", 1)
                            T.wait_cross_flag(2)
                            T.barrier_all()
            with T.Scope("V"):
                for turn in T.serial(rounds):
                    task = cid + turn * blocks
                    if task < tasks:
                        b, head = task // 96, task % 96
                        for qtile in T.serial(tiles):
                            load_t = T.if_then_else(qtile < full_tiles, qtile * tile_m, last_load)
                            row0 = vid * vr
                            T.copy(Z[b, load_t + row0 : load_t + row0 + vr, head, 0:512], z)
                            T.barrier_all()
                            T.tile.mul(z, z, inverse_scale)
                            T.barrier_all()
                            T.copy(z, hi)
                            T.barrier_all()
                            T.copy(hi, roundtrip)
                            T.barrier_all()
                            T.tile.sub(roundtrip, z, roundtrip)
                            T.barrier_all()
                            T.tile.mul(roundtrip, roundtrip, scale)
                            T.barrier_all()
                            T.copy(roundtrip, lo)
                            T.barrier_all()
                            T.copy(hi, HighWorkspace[cid, row0 : row0 + vr, 0:512])
                            T.copy(lo, LowWorkspace[cid, row0 : row0 + vr, 0:512])
                            T.barrier_all()
                            T.set_cross_flag("MTE3", 0)
                            T.wait_cross_flag(1)
                            T.barrier_all()
                            T.copy(ValueWorkspace[cid, row0 : row0 + vr, 0:128], value)
                            T.copy(LowValueWorkspace[cid, row0 : row0 + vr, 0:128], value_low)
                            T.copy(G[b, load_t + row0 : load_t + row0 + vr, head, 0:128], gh)
                            T.barrier_all()
                            T.tile.mul(value_low, value_low, inverse_scale)
                            T.barrier_all()
                            T.tile.add(value, value, value_low)
                            T.barrier_all()
                            T.copy(gh, gf)
                            T.barrier_all()
                            T.tile.abs(exponent, gf)
                            T.barrier_all()
                            T.tile.mul(exponent, exponent, -1.0)
                            T.barrier_all()
                            T.tile.exp(exponent, exponent)
                            T.barrier_all()
                            T.tile.add(denominator, exponent, 1.0)
                            T.barrier_all()
                            for row in T.serial(vr):
                                # Real vector compare/select on128 lanes, no scalar gate writes.
                                T.tile.compare(mask, gf[row, :], 0.0, "LT")
                                T.barrier_all()
                                T.tile.select(numerator[row, :], mask, exponent[row, :], 1.0, "VSEL_TENSOR_SCALAR_MODE", tmp=select_tmp)
                                T.barrier_all()
                            T.tile.div(numerator, numerator, denominator)
                            T.barrier_all()
                            T.tile.mul(value, value, numerator)
                            T.barrier_all()
                            T.tile.mul(value, value, scale)
                            T.barrier_all()
                            T.copy(value, yh)
                            T.barrier_all()
                            local_start = T.alloc_var("int32", init=0)
                            valid_rows = T.alloc_var("int32", init=vr)
                            local_start = 0
                            valid_rows = vr
                            if tail:  # noqa: SIM102 -- static specialization before symbolic tile predicate
                                if qtile == full_tiles:
                                    if row0 < tile_m - tail:
                                        local_start = tile_m - tail - row0
                                    if local_start > vr:
                                        local_start = vr
                                    valid_rows = vr - local_start
                            if valid_rows > 0:
                                output_t = load_t + row0 + local_start
                                T.copy(yh[local_start : local_start + valid_rows, :], Y[b, output_t : output_t + valid_rows, head, :])
                                T.barrier_all()
                            # Zero-output tail partners must acknowledge too.
                            T.set_cross_flag("MTE3", 2)
                            T.barrier_all()

    return tilelang.compile(main, out_idx=[3], workspace_idx=[4, 5, 6, 7], target=target, pass_configs=EXPERT_PASSES)


def build_value_cube(batch, query_capacity, *, dtype="bfloat16", target="pto", packed=False, tile_m=None, blocks=20):
    if dtype not in ("float16", "bfloat16") or target not in ("pto", "ascendc"):
        raise ValueError("unsupported dtype or target")
    p = plan(batch, query_capacity, tile_m=tile_m, blocks=blocks, packed=packed)
    return _compile(batch, query_capacity, dtype, target, packed, p["tile_m"], blocks)


def eligible_shape(B, Tq, H, dtype):
    return B in (1, 2, 4) and Tq >= 16 and H == 96 and dtype in ("float16", "bfloat16")


def _eligible(z, w, g, packed):
    return (
        all(isinstance(t, torch.Tensor) for t in (z, w, g))
        and z.ndim == 4
        and z.shape[0] in (1, 2, 4)
        and z.shape[1] >= 16
        and tuple(z.shape[2:]) == (96, 512)
        and tuple(w.shape) == (96, 256 if packed else 128, 512)
        and tuple(g.shape) == (*z.shape[:3], 128)
        and z.dtype == torch.float32
        and w.dtype in (torch.float16, torch.bfloat16)
        and g.dtype == w.dtype
        and z.device.type in ("npu", "privateuseone")
        and w.device == z.device
        and g.device == z.device
        and all(t.is_contiguous() for t in (z, w, g))
    )


def make_provider(vector_projections, vector_packed_projections, *, blocks=20, prefill_tile_m=None):
    """Pair separate/packed APIs; unsupported Cube domains retain original Vector."""
    provider = ModuleType("cube_value_family_provider")
    provider.__file__ = __file__
    provider.source_files = {"cube_value": __file__, "vector": vector_projections.__file__, "packed": vector_packed_projections.__file__}

    def selected(B, Tq, H, dtype, target, packed, fallback):
        if not eligible_shape(B, Tq, H, dtype):
            return fallback(B, Tq, H, dtype, target=target)
        return build_value_cube(B, Tq, dtype=dtype, target=target, packed=packed, tile_m=prefill_tile_m, blocks=blocks)

    def value_gate_kernel(B, Tq, H, dtype="float16", *, target="pto"):
        return selected(B, Tq, H, dtype, target, False, vector_projections.value_gate_kernel)

    def value_gate_packed_kernel(B, Tq, H, dtype="float16", *, target="pto"):
        return selected(B, Tq, H, dtype, target, True, vector_packed_projections.value_gate_packed_kernel)

    def call(z, w, g, target, packed, fallback):
        if not _eligible(z, w, g, packed):
            return fallback(z, w, g, target=target)
        B, Tq = map(int, z.shape[:2])
        dtype = "float16" if w.dtype == torch.float16 else "bfloat16"
        return build_value_cube(B, Tq, dtype=dtype, target=target, packed=packed, tile_m=prefill_tile_m, blocks=blocks)(z, w, g)

    def value_gate(z, w, g, *, target="pto"):
        return call(z, w, g, target, False, vector_projections.value_gate)

    def value_gate_packed(z, w, g, *, target="pto"):
        return call(z, w, g, target, True, vector_packed_projections.value_gate_packed)

    provider.value_gate, provider.value_gate_kernel = value_gate, value_gate_kernel
    provider.value_gate_packed, provider.value_gate_packed_kernel = value_gate_packed, value_gate_packed_kernel
    return provider
