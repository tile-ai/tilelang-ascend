"""Optimized TileLang RMSNorm for Ascend 910B3 (single-file).

Drop-in replacement for rms_norm_vs_builtin.py's TileLang kernel.

Why the original loses to builtin aclnnRmsNorm:
1. too many tiny blocks: vector-pipe util 1.6% vs builtin 74-86%
2. A read from GM twice; 3. redundant bf16->fp32 casts
4. CV-sync pass inserts PipeBarrier between every op (MTE/Vector never overlap)

Design (auto-selected by build_rms_norm):
- whole_row: single-pass whole row, ping-pong staging, MTE(k+1) overlaps Vector(k)
- row_cache: huge N: row cached in UB chunks, GM read once, incremental reduce
- two_pass:  column chunks, x re-read from GM (builtin does the same)
- orig:      original two-pass kernel, fallback

Benchmarking: JIT dispatch adds ~450us Python overhead; use `msprof op`
Task Duration instead:
    msprof op --launch-skip-before-match=10 --launch-count=10 \
        --output=<dir> python thisfile.py <cfg_idx>

Implementation: pipelined stages / row-cache buffers must be separate buffer
variables, so variants are generated into _rms_norm_opt_gen.py at import time
and imported back (safe to delete; regenerates).
"""

import importlib.util
import os
import sys

import torch
import torch_npu
import tilelang
from tilelang import language as T
from tilelang.carver.arch.ascend import Ascend as _AscendArch

if os.environ.get("TL_CLEAR_CACHE", "0") == "1":
    tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# ---------------- machine model (queried at runtime) ----------------
_NPU_PROPS = torch.npu.get_device_properties(torch.npu.current_device())
UB_LIMIT = _AscendArch().ub_cap - 256  # usable UB bytes per AIV sub-block
S_SUBBLOCKS = _NPU_PROPS.vector_core_num  # AIV sub-blocks (GRID = S/2, MIX_AIC_1_2)

# tiling search space: structural candidates (stages / chunk counts) for planners
STAGE_PREF = (4, 3, 2, 1)  # whole_row pipeline depths, tried in order
RC_N_SET = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24)  # row_cache chunk counts
FB_N_SET = (2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 50)  # two_pass chunk counts


def _floor_pow2(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _reduce_tmp_bytes(rows, n):
    # AscendC AR-sum fp32 workspace heuristic (tilelang allocate_tmp_buffer.cc)
    if n > 64:
        power = _floor_pow2(n)
        per_row = power if (rows == 1 or n != power) else power // 2
        return rows * per_row * 4
    return 32


def _divisors_desc(n):
    ds = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
        i += 1
    return sorted(ds, reverse=True)


# ---------------- tier 1: whole_row (single-pass whole row) ----------------
# 容量估算
def _ub_bytes_whole_row(rows, n, dtype, stages, variant):
    cast = dtype in ("bfloat16", "float16")
    xb = 2 if cast else 4
    per_elem = {
        "cast_bcast": stages * xb + 12,  # a_x_s + a_f32 + sq + inv_tile
        "cast_scalar": stages * xb + 4,  # a_x_s + a_f32 (recast)
        "cast_keep_scalar": stages * xb + 8,  # a_x_s + a_f32 + sq
        "fp32_bcast": stages * 4 + 8,
        "fp32_scalar": stages * 4 + 4,
    }[variant]
    return rows * n * per_elem + max(_reduce_tmp_bytes(rows, n), 1024) + rows * 8 + 512


# tier的决策函数
def _plan_whole_row(M, N, dtype, stages_pref=STAGE_PREF):
    """Returns (variant, stages, ROWS, GRID, k, R) or None. Exact ragged
    distribution: every sub-block runs k pipelined iterations of `stages`
    tiles; R leftover tiles go to a once-per-kernel guarded tail."""
    cast = dtype in ("bfloat16", "float16")
    fitting = []
    for rows in _divisors_desc(M):
        for stages in stages_pref:
            if rows == 1:
                # fp32_scalar (Muls by rsqrt) is dropped for accuracy: vector
                # Rsqrt is a ~2e-3 approximation; fp32 goes through the
                # broadcast + Sqrt + Div path instead (matches builtin's
                # ~3e-7). bf16 keeps the fast scalar path: Rsqrt's error is
                # below bf16's own 3.9e-3 output quantization.
                variants = ("cast_keep_scalar", "cast_scalar", "cast_bcast") if cast else ("fp32_bcast",)
            else:
                variants = ("cast_bcast",) if cast else ("fp32_bcast",)
            for v in variants:
                if _ub_bytes_whole_row(rows, N, dtype, stages, v) <= UB_LIMIT:
                    fitting.append((rows, v, stages))
    if not fitting:
        return None
    good = [x for x in fitting if M // x[0] >= S_SUBBLOCKS] or fitting
    rows, variant, stages = good[0]
    tiles = M // rows
    k = tiles // (S_SUBBLOCKS * stages)
    R = tiles - S_SUBBLOCKS * (k * stages)
    return variant, stages, rows, S_SUBBLOCKS // 2, k, R


# 语句模板
def _whole_row_tile_body(variant, st, buf):
    if variant == "cast_bcast":
        return f"""
                row0 = (base + {st}) * ROWS
                T.copy(A[row0 : row0 + ROWS, :], {buf})
                T.tile.cast(a_f32, {buf}, "CAST_NONE", ROWS * N)
                T.tile.mul(sq_f32, a_f32, a_f32)
                T.reduce_sum(sq_f32, sum_row, dim=-1)
                T.tile.mul(sum_row, sum_row, inv_n)
                T.tile.add(sum_row, sum_row, eps_val)
                T.tile.sqrt(inv_rms, sum_row)
                T.tile.broadcast(inv_tile, inv_rms)
                T.tile.div(a_f32, a_f32, inv_tile)
                T.tile.cast({buf}, a_f32, "CAST_RINT", ROWS * N)
                T.copy({buf}, B[row0 : row0 + ROWS, :])
"""
    if variant == "cast_keep_scalar":  # ROWS == 1: separate sq, 5 vec passes
        return f"""
                row0 = (base + {st}) * ROWS
                T.copy(A[row0 : row0 + ROWS, :], {buf})
                T.tile.cast(a_f32, {buf}, "CAST_NONE", ROWS * N)
                T.tile.mul(sq_f32, a_f32, a_f32)
                T.reduce_sum(sq_f32, sum_row, dim=-1)
                T.tile.mul(sum_row, sum_row, inv_n)
                T.tile.add(sum_row, sum_row, eps_val)
                T.tile.sqrt(sum_row, sum_row)
                T.tile.broadcast(sq_f32, sum_row)
                T.tile.div(a_f32, a_f32, sq_f32)
                T.tile.cast({buf}, a_f32, "CAST_RINT", ROWS * N)
                T.copy({buf}, B[row0 : row0 + ROWS, :])
"""
    if variant == "cast_scalar":  # ROWS == 1: in-place square + re-cast
        return f"""
                row0 = (base + {st}) * ROWS
                T.copy(A[row0 : row0 + ROWS, :], {buf})
                T.tile.cast(a_f32, {buf}, "CAST_NONE", ROWS * N)
                T.tile.mul(a_f32, a_f32, a_f32)
                T.reduce_sum(a_f32, sum_row, dim=-1)
                T.tile.mul(sum_row, sum_row, inv_n)
                T.tile.add(sum_row, sum_row, eps_val)
                T.tile.sqrt(sum_row, sum_row)
                T.tile.cast(a_f32, {buf}, "CAST_NONE", ROWS * N)
                T.tile.div(a_f32, a_f32, sum_row[0])
                T.tile.cast({buf}, a_f32, "CAST_RINT", ROWS * N)
                T.copy({buf}, B[row0 : row0 + ROWS, :])
"""
    if variant == "fp32_bcast":
        return f"""
                row0 = (base + {st}) * ROWS
                T.copy(A[row0 : row0 + ROWS, :], {buf})
                T.tile.mul(sq_f32, {buf}, {buf})
                T.reduce_sum(sq_f32, sum_row, dim=-1)
                T.tile.mul(sum_row, sum_row, inv_n)
                T.tile.add(sum_row, sum_row, eps_val)
                T.tile.sqrt(inv_rms, sum_row)
                T.tile.broadcast(inv_tile, inv_rms)
                T.tile.div({buf}, {buf}, inv_tile)
                T.copy({buf}, B[row0 : row0 + ROWS, :])
"""
    # fp32_scalar: ROWS == 1
    return f"""
                row0 = (base + {st}) * ROWS
                T.copy(A[row0 : row0 + ROWS, :], {buf})
                T.tile.mul(sq_f32, {buf}, {buf})
                T.reduce_sum(sq_f32, sum_row, dim=-1)
                T.tile.mul(sum_row, sum_row, inv_n)
                T.tile.add(sum_row, sum_row, eps_val)
                T.tile.rsqrt(inv_rms, sum_row)
                T.tile.mul({buf}, {buf}, inv_rms[0])
                T.copy({buf}, B[row0 : row0 + ROWS, :])
"""


# buffer模板
def _whole_row_buffers(variant, stages, cast):
    bufs = []
    for s in range(stages):
        bufs.append((f"a_x_{s}", "[ROWS, N]", "dtype" if cast else "acc_dtype"))
    if variant == "cast_bcast":
        bufs += [
            ("a_f32", "[ROWS, N]", "acc_dtype"),
            ("sq_f32", "[ROWS, N]", "acc_dtype"),
            ("inv_tile", "[ROWS, N]", "acc_dtype"),
            ("sum_row", "[ROWS]", "acc_dtype"),
            ("inv_rms", "[ROWS]", "acc_dtype"),
        ]
    elif variant == "cast_scalar":
        bufs += [("a_f32", "[ROWS, N]", "acc_dtype"), ("sum_row", "[ROWS]", "acc_dtype")]
    elif variant == "cast_keep_scalar":
        bufs += [("a_f32", "[ROWS, N]", "acc_dtype"), ("sq_f32", "[ROWS, N]", "acc_dtype"), ("sum_row", "[ROWS]", "acc_dtype")]
    elif variant == "fp32_bcast":
        bufs += [
            ("sq_f32", "[ROWS, N]", "acc_dtype"),
            ("inv_tile", "[ROWS, N]", "acc_dtype"),
            ("sum_row", "[ROWS]", "acc_dtype"),
            ("inv_rms", "[ROWS]", "acc_dtype"),
        ]
    else:  # fp32_scalar
        bufs += [("sq_f32", "[ROWS, N]", "acc_dtype"), ("sum_row", "[ROWS]", "acc_dtype"), ("inv_rms", "[ROWS]", "acc_dtype")]
    return bufs


# 组装器
def _make_whole_row_impl(variant, stages):
    cast = variant.startswith("cast")
    bufs = _whole_row_buffers(variant, stages, cast)
    main_body = "".join(_whole_row_tile_body(variant, s, f"a_x_{s}") for s in range(stages))
    tail_parts = []
    for s in range(stages):
        body_lines = _whole_row_tile_body(variant, s, f"a_x_{s}").strip("\n").splitlines()
        inner = "\n".join("    " + ln for ln in body_lines)
        tail_parts.append("                if s * STG + %d < R:\n                    base = S * (k * STG) + s * STG\n%s" % (s, inner))
    tail = "\n".join(tail_parts)
    decl = "\n            ".join(f"{n} = T.alloc_ub({sh}, {dt})" for n, sh, dt in bufs)
    name = f"_rms_whole_row_{variant}_s{stages}"
    src = f"""
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def {name}(M, N, eps=1e-5, dtype="float", ROWS=1, GRID=20, k=1, R=0):
    need_cast = dtype not in ("float", "float32")
    acc_dtype = "float32" if need_cast else dtype
    STG = {stages}
    S = GRID * 2

    @T.prim_func
    def tilelang_rms_norm_opt(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(GRID, is_npu=True) as (cid, vid):
            s = cid * 2 + vid
            {decl}
            eps_val = T.cast(eps, acc_dtype)
            inv_n = T.cast(1.0 / N, acc_dtype)
            for tt in T.serial(k):
                base = s * (k * STG) + tt * STG{main_body}
            # remainder tiles: at most STG per sub-block, guarded once per kernel
            if s * STG < R:
{tail}
    return tilelang_rms_norm_opt
"""
    return name, src


# ---------------- tier 2: row_cache (huge N, single GM read) ----------------


def _ub_bytes_row_cache(rows, n, bn, dtype):
    cast = dtype in ("bfloat16", "float16")
    if cast:
        total = rows * n * 2 + rows * bn * 8  # rc + a_f32 + inv_tile
    else:
        total = rows * n * 4 + rows * bn * 8  # rc + sq + inv_tile
    return total + max(_reduce_tmp_bytes(rows, bn), 1024) + rows * 8 + 512


def _plan_row_cache(M, N, dtype):
    """Returns (ROWS, GRID, k, R, BLOCK_N, n_num) or None. Biggest chunk
    first (measured best), then more rows."""
    rows_cap = max(1, min(M // S_SUBBLOCKS if M >= S_SUBBLOCKS else M, 4))
    cands = []
    for rows in _divisors_desc(M):
        if rows > rows_cap:
            continue
        for bn in _divisors_desc(N):
            if bn < 512 or bn > 16384:
                continue
            n_num = N // bn
            if n_num not in RC_N_SET:
                continue
            if _ub_bytes_row_cache(rows, N, bn, dtype) <= UB_LIMIT:
                cands.append((rows, bn, n_num))
    if not cands:
        return None
    cands.sort(key=lambda c: (-c[1], -c[0]))
    rows, bn, n_num = cands[0]
    tiles = M // rows
    k = tiles // S_SUBBLOCKS
    R = tiles - S_SUBBLOCKS * k
    return rows, S_SUBBLOCKS // 2, k, R, bn, n_num


# 语句模板
def _row_cache_body(cast_variant, n_num):
    lines = ["T.tile.fill(sum_row, 0.0)"]
    for c in range(n_num):
        if cast_variant:
            lines += [
                f"col0 = {c} * BLOCK_N",
                f"T.copy(A[row0 : row0 + ROWS, col0 : col0 + BLOCK_N], rc_{c})",
                f'T.tile.cast(a_f32, rc_{c}, "CAST_NONE", ROWS * BLOCK_N)',
                "T.tile.mul(a_f32, a_f32, a_f32)",
                "T.reduce_sum(a_f32, sum_row, -1, False)",
            ]
        else:
            lines += [
                f"col0 = {c} * BLOCK_N",
                f"T.copy(A[row0 : row0 + ROWS, col0 : col0 + BLOCK_N], rc_{c})",
                f"T.tile.mul(sq_f32, rc_{c}, rc_{c})",
                "T.reduce_sum(sq_f32, sum_row, -1, False)",
            ]
    lines += [
        "T.tile.mul(sum_row, sum_row, inv_n)",
        "T.tile.add(sum_row, sum_row, eps_val)",
        "T.tile.sqrt(sum_row, sum_row)",
        "T.tile.broadcast(inv_tile, sum_row)",
    ]
    for c in range(n_num):
        if cast_variant:
            lines += [
                f"col0 = {c} * BLOCK_N",
                f'T.tile.cast(a_f32, rc_{c}, "CAST_NONE", ROWS * BLOCK_N)',
                "T.tile.div(a_f32, a_f32, inv_tile)",
                f'T.tile.cast(rc_{c}, a_f32, "CAST_RINT", ROWS * BLOCK_N)',
                f"T.copy(rc_{c}, B[row0 : row0 + ROWS, col0 : col0 + BLOCK_N])",
            ]
        else:
            lines += [
                f"col0 = {c} * BLOCK_N",
                f"T.tile.div(rc_{c}, rc_{c}, inv_tile)",
                f"T.copy(rc_{c}, B[row0 : row0 + ROWS, col0 : col0 + BLOCK_N])",
            ]
    return lines


# buffer模板 + 组装器
def _make_row_cache_impl(cast_variant, n_num):
    name = f"_rms_row_cache_{'cast' if cast_variant else 'fp32'}_n{n_num}"
    buf_dt = "dtype" if cast_variant else "acc_dtype"
    decl = [f"rc_{c} = T.alloc_ub([ROWS, BLOCK_N], {buf_dt})" for c in range(n_num)]
    if cast_variant:
        decl.append("a_f32 = T.alloc_ub([ROWS, BLOCK_N], acc_dtype)")
    else:
        decl.append("sq_f32 = T.alloc_ub([ROWS, BLOCK_N], acc_dtype)")
    decl += [
        "inv_tile = T.alloc_ub([ROWS, BLOCK_N], acc_dtype)",
        "sum_row = T.alloc_ub([ROWS], acc_dtype)",
        "eps_val = T.cast(eps, acc_dtype)",
        "inv_n = T.cast(1.0 / N, acc_dtype)",
    ]
    I2 = " " * 12
    I4 = " " * 16
    main_loop = (
        [I2 + "for tt in T.serial(k):"]
        + [I4 + "base = s * k + tt", I4 + "row0 = base * ROWS"]
        + [I4 + ln for ln in _row_cache_body(cast_variant, n_num)]
    )
    tail = [I4 + "base = S * k + s", I4 + "row0 = base * ROWS"] + [I4 + ln for ln in _row_cache_body(cast_variant, n_num)]
    kernel_body = "\n".join(
        ["        with T.Kernel(GRID, is_npu=True) as (cid, vid):"]
        + ["            s = cid * 2 + vid"]
        + [I2 + ln for ln in decl]
        + main_loop
        + [I2 + "# remainder tiles, guarded once per kernel", I2 + "if s < R:"]
        + tail
    )
    src = f"""
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def {name}(M, N, eps=1e-5, dtype="float", ROWS=1, GRID=20, k=1, R=0, BLOCK_N=1024):
    need_cast = dtype not in ("float", "float32")
    acc_dtype = "float32" if need_cast else dtype
    S = GRID * 2

    @T.prim_func
    def tilelang_rms_norm_opt(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
{kernel_body}
    return tilelang_rms_norm_opt
"""
    return name, src


# ---------------- tier 3: two_pass (column chunks, x re-read) ----------------
# 估算器内置 planner
def _plan_two_pass(M, N, dtype):
    """Returns (ROWS, GRID, k, R, BLOCK_N, n_num) or None. Biggest chunk
    first; n_num must be even (ping-pong chunk staging)."""
    rows_cap = max(1, min(M // S_SUBBLOCKS if M >= S_SUBBLOCKS else M, 4))
    cast = dtype in ("bfloat16", "float16")
    xb = 2 if cast else 4
    per_elem = 2 * xb + 12 if cast else 2 * 4 + 8
    cands = []
    for rows in _divisors_desc(M):
        if rows > rows_cap:
            continue
        for bn in _divisors_desc(N):
            if bn < 512 or bn > 8192:
                continue
            n_num = N // bn
            if n_num not in FB_N_SET:
                continue
            total = rows * bn * per_elem + max(_reduce_tmp_bytes(rows, bn), 1024) + rows * 8 + 512
            if total <= UB_LIMIT:
                cands.append((rows, bn, n_num))
    if not cands:
        return None
    cands.sort(key=lambda c: (-c[1], -c[0]))
    rows, bn, n_num = cands[0]
    tiles = M // rows
    k = tiles // S_SUBBLOCKS
    R = tiles - S_SUBBLOCKS * k
    return rows, S_SUBBLOCKS // 2, k, R, bn, n_num


# 语句模板
def _two_pass_chunk_lines(cast_variant, c, buf):
    if cast_variant:
        return [
            f"col0 = {c} * BLOCK_N",
            f"T.copy(A[row0 : row0 + ROWS, col0 : col0 + BLOCK_N], {buf})",
            f'T.tile.cast(a_f32, {buf}, "CAST_NONE", ROWS * BLOCK_N)',
            "T.tile.mul(sq_f32, a_f32, a_f32)",
            "T.reduce_sum(sq_f32, sum_row, -1, False)",
        ]
    return [
        f"col0 = {c} * BLOCK_N",
        f"T.copy(A[row0 : row0 + ROWS, col0 : col0 + BLOCK_N], {buf})",
        f"T.tile.mul(sq_f32, {buf}, {buf})",
        "T.reduce_sum(sq_f32, sum_row, -1, False)",
    ]


def _two_pass_chunk_lines2(cast_variant, c, buf):
    if cast_variant:
        return [
            f"col0 = {c} * BLOCK_N",
            f"T.copy(A[row0 : row0 + ROWS, col0 : col0 + BLOCK_N], {buf})",
            f'T.tile.cast(a_f32, {buf}, "CAST_NONE", ROWS * BLOCK_N)',
            "T.tile.div(a_f32, a_f32, inv_tile)",
            f'T.tile.cast({buf}, a_f32, "CAST_RINT", ROWS * BLOCK_N)',
            f"T.copy({buf}, B[row0 : row0 + ROWS, col0 : col0 + BLOCK_N])",
        ]
    return [
        f"col0 = {c} * BLOCK_N",
        f"T.copy(A[row0 : row0 + ROWS, col0 : col0 + BLOCK_N], {buf})",
        f"T.tile.div({buf}, {buf}, inv_tile)",
        f"T.copy({buf}, B[row0 : row0 + ROWS, col0 : col0 + BLOCK_N])",
    ]


def _make_two_pass_impl(cast_variant, n_num):
    name = f"_rms_two_pass_{'cast' if cast_variant else 'fp32'}_n{n_num}"
    buf_dt = "dtype" if cast_variant else "acc_dtype"
    decl = [
        f"a_x_0 = T.alloc_ub([ROWS, BLOCK_N], {buf_dt})",
        f"a_x_1 = T.alloc_ub([ROWS, BLOCK_N], {buf_dt})",
    ]
    if cast_variant:
        decl += [
            "a_f32 = T.alloc_ub([ROWS, BLOCK_N], acc_dtype)",
            "sq_f32 = T.alloc_ub([ROWS, BLOCK_N], acc_dtype)",
        ]
    else:
        decl += ["sq_f32 = T.alloc_ub([ROWS, BLOCK_N], acc_dtype)"]
    decl += [
        "inv_tile = T.alloc_ub([ROWS, BLOCK_N], acc_dtype)",
        "sum_row = T.alloc_ub([ROWS], acc_dtype)",
        "inv_rms = T.alloc_ub([ROWS], acc_dtype)",
        "eps_val = T.cast(eps, acc_dtype)",
        "inv_n = T.cast(1.0 / N, acc_dtype)",
    ]
    body = ["T.tile.fill(sum_row, 0.0)"]
    for c in range(n_num):
        body += _two_pass_chunk_lines(cast_variant, c, f"a_x_{c % 2}")
    body += [
        "T.tile.mul(sum_row, sum_row, inv_n)",
        "T.tile.add(sum_row, sum_row, eps_val)",
        "T.tile.sqrt(inv_rms, sum_row)",
        "T.tile.broadcast(inv_tile, inv_rms)",
    ]
    for c in range(n_num):
        body += _two_pass_chunk_lines2(cast_variant, c, f"a_x_{c % 2}")
    I2 = " " * 12
    I4 = " " * 16
    main_loop = [I2 + "for tt in T.serial(k):"] + [I4 + "base = s * k + tt", I4 + "row0 = base * ROWS"] + [I4 + ln for ln in body]
    tail = [I4 + "base = S * k + s", I4 + "row0 = base * ROWS"] + [I4 + ln for ln in body]
    kernel_body = "\n".join(
        ["        with T.Kernel(GRID, is_npu=True) as (cid, vid):"]
        + ["            s = cid * 2 + vid"]
        + [I2 + ln for ln in decl]
        + main_loop
        + [I2 + "# remainder tiles, guarded once per kernel", I2 + "if s < R:"]
        + tail
    )
    src = f"""
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def {name}(M, N, eps=1e-5, dtype="float", ROWS=1, GRID=20, k=1, R=0, BLOCK_N=1024):
    need_cast = dtype not in ("float", "float32")
    acc_dtype = "float32" if need_cast else dtype
    S = GRID * 2

    @T.prim_func
    def tilelang_rms_norm_opt(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
{kernel_body}
    return tilelang_rms_norm_opt
"""
    return name, src


# ---------------- generate + import all variants ----------------

_GEN_NAMES = {}


def _generate_variants():
    here = os.path.dirname(os.path.abspath(__file__))
    gen_path = os.path.join(here, "_rms_norm_opt_gen.py")
    with open(gen_path, "w") as f:
        f.write("from tilelang import language as T\nimport tilelang\n\n")
        f.write("pass_configs = {\n")
        for kk, v in pass_configs.items():
            f.write(f"    tilelang.PassConfigKey.{kk.name}: {v},\n")
        f.write("}\n\n")
        names = {}
        for variant in ("cast_bcast", "cast_scalar", "cast_keep_scalar", "fp32_bcast", "fp32_scalar"):
            for stages in STAGE_PREF:
                nm, src = _make_whole_row_impl(variant, stages)
                f.write(src + "\n\n")
                names[("whole_row", variant, stages)] = nm
        for cast_variant in (True, False):
            for n_num in RC_N_SET:
                nm, src = _make_row_cache_impl(cast_variant, n_num)
                f.write(src + "\n\n")
                names[("row_cache", cast_variant, n_num)] = nm
            for n_num in FB_N_SET:
                nm, src = _make_two_pass_impl(cast_variant, n_num)
                f.write(src + "\n\n")
                names[("two_pass", cast_variant, n_num)] = nm
        return gen_path, names


_GEN_PATH, _GEN_NAMES = _generate_variants()
_spec = importlib.util.spec_from_file_location("_rms_norm_opt_gen", _GEN_PATH)
_gen_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen_mod)
_IMPLS = {key: getattr(_gen_mod, nm) for key, nm in _GEN_NAMES.items()}


def _build_whole_row(M, N, dtype, eps):
    p = _plan_whole_row(M, N, dtype)
    if p is None:
        return None
    variant, stages, rows, grid, k, R = p
    return _IMPLS[("whole_row", variant, stages)](M, N, eps=eps, dtype=dtype, ROWS=rows, GRID=grid, k=k, R=R)


def _build_row_cache(M, N, dtype, eps):
    p = _plan_row_cache(M, N, dtype)
    if p is None:
        return None
    rows, grid, k, R, bn, n_num = p
    cast = dtype in ("bfloat16", "float16")
    return _IMPLS[("row_cache", cast, n_num)](M, N, eps=eps, dtype=dtype, ROWS=rows, GRID=grid, k=k, R=R, BLOCK_N=bn)


def _build_two_pass(M, N, dtype, eps):
    p = _plan_two_pass(M, N, dtype)
    if p is None:
        return None
    rows, grid, k, R, bn, n_num = p
    cast = dtype in ("bfloat16", "float16")
    return _IMPLS[("two_pass", cast, n_num)](M, N, eps=eps, dtype=dtype, ROWS=rows, GRID=grid, k=k, R=R, BLOCK_N=bn)


# ---------------- tier 4: original kernel (last resort) ----------------


def _get_optimized_tiling(M, N, block_M_in, block_N_in, vec_num):
    budget = block_M_in * block_N_in
    ideal_n = budget // 16
    block_N = min(N // 2, ideal_n)
    if block_N < 128:
        block_N = 128 if N >= 256 else N
    while N % block_N != 0:
        block_N -= 1
        if block_N <= 0:
            block_N = 1
            break
    block_M = budget // block_N
    if M % block_M != 0:
        block_M = block_M_in if M % block_M_in == 0 else vec_num
    return block_M, block_N


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def _rms_norm_orig(M, N, block_M_in, block_N_in, eps=1e-5, dtype="float"):
    VEC_NUM = 2
    block_M, block_N = _get_optimized_tiling(M, N, block_M_in, block_N_in, VEC_NUM)
    m_num = M // block_M
    n_num = N // block_N
    ROWS = block_M // VEC_NUM
    tile_elements = ROWS * block_N
    need_cast = dtype not in ("float", "float32")
    acc_dtype = "float32" if need_cast else dtype

    @T.prim_func
    def tilelang_rms_norm_orig(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_ub_0 = T.alloc_ub([ROWS, block_N], dtype)
            a_ub_1 = T.alloc_ub([ROWS, block_N], dtype)
            a_ub_cast_0 = T.alloc_ub([ROWS, block_N], acc_dtype)
            a_ub_cast_1 = T.alloc_ub([ROWS, block_N], acc_dtype)
            sum_sq_acc = T.alloc_ub([ROWS, block_N], acc_dtype)
            sum_sq_row = T.alloc_ub([ROWS, 1], acc_dtype)
            inv_rms_ub = T.alloc_ub([ROWS, 1], acc_dtype)
            inv_rms_tile = T.alloc_ub([ROWS, block_N], acc_dtype)
            row_start = cid * block_M + vid * ROWS
            T.tile.fill(sum_sq_acc, 0.0)
            for by in T.serial(n_num // 2):
                col_off_0 = (by * 2) * block_N
                if need_cast:
                    T.copy(A[row_start : row_start + ROWS, col_off_0 : col_off_0 + block_N], a_ub_0)
                    T.tile.cast(a_ub_cast_0, a_ub_0, "CAST_NONE", tile_elements)
                else:
                    T.copy(A[row_start : row_start + ROWS, col_off_0 : col_off_0 + block_N], a_ub_cast_0)
                T.tile.mul(a_ub_cast_0, a_ub_cast_0, a_ub_cast_0)
                T.tile.add(sum_sq_acc, sum_sq_acc, a_ub_cast_0)
                col_off_1 = (by * 2 + 1) * block_N
                if need_cast:
                    T.copy(A[row_start : row_start + ROWS, col_off_1 : col_off_1 + block_N], a_ub_1)
                    T.tile.cast(a_ub_cast_1, a_ub_1, "CAST_NONE", tile_elements)
                else:
                    T.copy(A[row_start : row_start + ROWS, col_off_1 : col_off_1 + block_N], a_ub_cast_1)
                T.tile.mul(a_ub_cast_1, a_ub_cast_1, a_ub_cast_1)
                T.tile.add(sum_sq_acc, sum_sq_acc, a_ub_cast_1)
            if n_num % 2 != 0:
                col_off_rem = (n_num - 1) * block_N
                if need_cast:
                    T.copy(A[row_start : row_start + ROWS, col_off_rem : col_off_rem + block_N], a_ub_0)
                    T.tile.cast(a_ub_cast_0, a_ub_0, "CAST_NONE", tile_elements)
                else:
                    T.copy(A[row_start : row_start + ROWS, col_off_rem : col_off_rem + block_N], a_ub_cast_0)
                T.tile.mul(a_ub_cast_0, a_ub_cast_0, a_ub_cast_0)
                T.tile.add(sum_sq_acc, sum_sq_acc, a_ub_cast_0)
            T.reduce_sum(sum_sq_acc, sum_sq_row, dim=-1)
            inv_n = T.cast(1.0 / N, acc_dtype)
            eps_val = T.cast(eps, acc_dtype)
            T.tile.mul(sum_sq_row, sum_sq_row, inv_n)
            T.tile.add(sum_sq_row, sum_sq_row, eps_val)
            T.tile.rsqrt(inv_rms_ub, sum_sq_row)
            T.tile.broadcast(inv_rms_tile, inv_rms_ub)
            for by in T.serial(n_num // 2):
                col_off_0 = (by * 2) * block_N
                if need_cast:
                    T.copy(A[row_start : row_start + ROWS, col_off_0 : col_off_0 + block_N], a_ub_0)
                    T.tile.cast(a_ub_cast_0, a_ub_0, "CAST_NONE", tile_elements)
                else:
                    T.copy(A[row_start : row_start + ROWS, col_off_0 : col_off_0 + block_N], a_ub_cast_0)
                T.tile.mul(a_ub_cast_0, a_ub_cast_0, inv_rms_tile)
                if need_cast:
                    T.tile.cast(a_ub_0, a_ub_cast_0, "CAST_RINT", tile_elements)
                    T.copy(a_ub_0, B[row_start : row_start + ROWS, col_off_0 : col_off_0 + block_N])
                else:
                    T.copy(a_ub_cast_0, B[row_start : row_start + ROWS, col_off_0 : col_off_0 + block_N])
                col_off_1 = (by * 2 + 1) * block_N
                if need_cast:
                    T.copy(A[row_start : row_start + ROWS, col_off_1 : col_off_1 + block_N], a_ub_1)
                    T.tile.cast(a_ub_cast_1, a_ub_1, "CAST_NONE", tile_elements)
                else:
                    T.copy(A[row_start : row_start + ROWS, col_off_1 : col_off_1 + block_N], a_ub_cast_1)
                T.tile.mul(a_ub_cast_1, a_ub_cast_1, inv_rms_tile)
                if need_cast:
                    T.tile.cast(a_ub_1, a_ub_cast_1, "CAST_RINT", tile_elements)
                    T.copy(a_ub_1, B[row_start : row_start + ROWS, col_off_1 : col_off_1 + block_N])
                else:
                    T.copy(a_ub_cast_1, B[row_start : row_start + ROWS, col_off_1 : col_off_1 + block_N])
            if n_num % 2 != 0:
                col_off_rem = (n_num - 1) * block_N
                if need_cast:
                    T.copy(A[row_start : row_start + ROWS, col_off_rem : col_off_rem + block_N], a_ub_0)
                    T.tile.cast(a_ub_cast_0, a_ub_0, "CAST_NONE", tile_elements)
                else:
                    T.copy(A[row_start : row_start + ROWS, col_off_rem : col_off_rem + block_N], a_ub_cast_0)
                T.tile.mul(a_ub_cast_0, a_ub_cast_0, inv_rms_tile)
                if need_cast:
                    T.tile.cast(a_ub_0, a_ub_cast_0, "CAST_RINT", tile_elements)
                    T.copy(a_ub_0, B[row_start : row_start + ROWS, col_off_rem : col_off_rem + block_N])
                else:
                    T.copy(a_ub_cast_0, B[row_start : row_start + ROWS, col_off_rem : col_off_rem + block_N])

    return tilelang_rms_norm_orig


# ---------------- unified entry ----------------


def build_rms_norm(M, N, dtype="float", eps=1e-5):
    """Returns (callable, tier). Tiers: whole_row (single-pass) > row_cache
    (GM read once) > two_pass (column chunks) > orig (original kernel)."""
    f = _build_whole_row(M, N, dtype, eps)
    if f is not None:
        return f, "whole_row"
    f = _build_row_cache(M, N, dtype, eps)
    if f is not None:
        return f, "row_cache"
    f = _build_two_pass(M, N, dtype, eps)
    if f is not None:
        return f, "two_pass"
    return _rms_norm_orig(M, N, 64 if dtype != "float" else 128, 128, eps=eps, dtype=dtype), "orig"


# ---------------- test / bench harness (same shapes as rms_norm_vs_builtin.py) ----------------

torch.manual_seed(0)

test_configs = [
    # 小 shape
    (256, 256, 64, 64, "float"),
    (256, 256, 64, 64, "bfloat16"),
    (1024, 1024, 128, 128, "float"),
    (1024, 1024, 64, 128, "bfloat16"),
    # 一大一小: M 大 N 小 / M 小 N 大
    (16384, 1536, 64, 128, "float"),
    (8192, 1536, 64, 128, "float"),
    (1536, 16384, 64, 128, "float"),
    (1536, 8192, 64, 128, "float"),
    (16384, 1536, 64, 128, "bfloat16"),
    (8192, 1536, 64, 128, "bfloat16"),
    (1536, 16384, 64, 128, "bfloat16"),
    (1536, 8192, 64, 128, "bfloat16"),
    # 大 shape: 8k/16k 量级
    (8192, 8192, 128, 128, "float"),
    (8192, 8192, 64, 128, "bfloat16"),
    (16384, 16384, 128, 128, "float"),
    (16384, 16384, 64, 128, "bfloat16"),
    (8192, 16384, 128, 128, "float"),
    (8192, 16384, 64, 128, "bfloat16"),
    (16384, 8192, 128, 128, "float"),
    (16384, 8192, 64, 128, "bfloat16"),
]

WARMUP = 3
ITERS = 5


def _make_input(M, N, dtype):
    if dtype == "bfloat16":
        return torch.randn(M, N, device="npu", dtype=torch.bfloat16)
    return torch.randn(M, N, device="npu", dtype=torch.float32)


def run_config(M, N, block_M, block_N, dtype):
    print(f"\n[Correctness] M={M}, N={N}, dtype={dtype}", flush=True)
    func, tier = build_rms_norm(M, N, dtype=dtype)
    print(f"  tier={tier}", flush=True)

    a = _make_input(M, N, dtype)
    weight = torch.ones(N, device="npu", dtype=a.dtype)

    b = func(a)
    ref_b, _ = torch_npu.npu_rms_norm(a, weight, 1e-5)
    torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
    print("  Match with builtin aclnnRmsNorm!", flush=True)

    # NOTE: wall-clock here is dominated by ~450us/call of Python dispatch;
    # use `msprof op` (Task Duration) for kernel-level performance.
    print(f"[Perf] M={M}, N={N}, dtype={dtype}", flush=True)
    for _ in range(WARMUP):
        func(a)
        torch_npu.npu_rms_norm(a, weight, 1e-5)
    torch.npu.synchronize()
    for _ in range(ITERS):
        func(a)
        torch_npu.npu_rms_norm(a, weight, 1e-5)
    torch.npu.synchronize()
    print(
        "  Perf done (capture with: msprof op --launch-skip-before-match=10 --launch-count=10 --output=<dir> python <thisfile> <idx>)",
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        indices = [int(x) for x in sys.argv[1].split(",")]
    else:
        indices = range(len(test_configs))
    for i in indices:
        run_config(*test_configs[i])
    print("\nAll done. tilelang_rms_norm_opt vs aclnnRmsNorm captured for msprof op.", flush=True)
