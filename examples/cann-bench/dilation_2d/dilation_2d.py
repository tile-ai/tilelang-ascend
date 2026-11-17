# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the CANN Open Software License Agreement Version 2.0 (the
# "License"). You should not use this file except in compliance with the
# License. You may obtain a copy of the License at:
#     https://www.hiascend.com/software/cann/licensing
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# ----------------------------------------------------------------------------

"""Dilation2D TileLang kernel (Developer mode, pure Vector) — perf rev7.

Semantics (NHWC, matches tasks/level3/dilation_2d/golden.py):

    y[b, oy, ox, c] = max_{dy, dx} ( x[b, oy*sh + rh*dy, ox*sw + rw*dx, c]
                                     + filter[dy, dx, c] )

Out-of-range taps contribute ``-inf + filter`` exactly like the golden (it
pads x with -inf and takes ``amax`` over every tap), so NaN inputs propagate
and a ``+inf`` filter tap turns an out-of-range contribution into NaN.

The binding hardware constraint is the copy engine: 2-D GM/UB copies need
32B-multiple row strides on BOTH sides, and vector instructions reject odd
offsets — odd channel counts cannot use 2-D tiled access at all.  The
wrapper dispatches on that single constraint:

* Path AB (C is 16-aligned): one 2-D window load per tap row into a
  ``[WSPAN, C]`` UB tile, fused ``max(add(broadcast))`` on ``[WB, C]``
  tiles, one 2-D store.  Single kernel, no workspace.
  - ``sw == 1`` and ``Wo <= 64``: multi-row RY variant (``_build_ab_ry``) —
    each CTA owns RY output rows over a flat ``[RYROWS*WSPAN, C]`` window
    whose rows are shared between adjacent output rows (r1 windows overlap:
    RY+fh-1 row loads instead of RY*fh) and the filter broadcast runs once
    per tap for all RY rows.
  - ``sw == 1``, wide rows: single-row variant (``_build_ab_s1``).
  - ``sw >= 2``: phase-split variant (``_build_ab_ph``) — the strided
    window is redistributed once into a phase-split UB plane so every tap
    slice is contiguous again.
* Path bridge (odd C, >= 32K elements or sw > 1): pad C up to a 16-multiple
  on the host (x with zeros, filter with -inf) and reuse Path AB unchanged.
  Padded channels contribute ``0 + (-inf) = -inf`` and never win the max,
  so real channels stay bit-exact.
* Path C (odd C, sw == 1, tiny inputs): the tap slice of WB consecutive
  pixels is ONE contiguous flat GM range, so each tap is a single flat 1-D
  load; the per-channel filter is pre-expanded on the host into
  ``frep[K2, WB*C]`` so the flat add needs no broadcast.  A persistent grid
  (<= 44 CTAs, a device-independent cap — each CTA loops over column
  blocks) keeps frep resident in UB.
* Path D: rev1 per-pixel fallback.  Unreachable since rev4 (the bridge
  covers every odd-C shape the dispatcher can produce); kept so the kernel
  bodies stay identical to the evaluated submission.

Tiling search (``_pick_ry_wb`` / ``_pick_wb_ab``): candidates are bounded
by the UB budget — win + acc + tmp/fbp + fb panels must fit ~176KB with
16-aligned widths — and by the grid limit.  Among UB-fitting candidates the
objective is wave quantisation: wall time ~ (fixed + RY*per-row) *
ceil(grid / cores), so the best RY is the smallest one that brings the grid
into a single wave, and when no RY reaches a single wave the largest
UB-fitting RY amortises the fixed cost best.  The small/large split at
``N*Ho >= 180`` and the RY cap 13 come from a per-case sweep over the
20-case benchmark set, not from an analytical model.

Synchronisation (champion model, Path AB kernels): AUTO_SYNC emits ~318
barriers per CTA at RY=13 — measured as ~50% of kernel time.  The
single-pass structure (fill -> load -> taps -> store, no loop-carried
MTE/V alternation) needs only a few pipe flag pairs per CTA:

* RY:  (2) v->mte2 orders fill->load, (1) mte2->v orders load->taps,
  (3) v->mte3 orders taps->store — 3 pairs total.
* s1:  per dy, (6) mte2->v at loop top covers the previous dy's load, (2)
  orders fill->load, (5) orders load->taps; (3) after the loop.
* ph:  same as s1, plus two ``T.barrier_all`` per dy around the phase
  redistribution — ``copy_ub_to_ub`` runs on an MTE pipe that the mte2->v
  flag does not cover, and the GM load -> redistribution pair crosses MTE
  pipes, so only a full barrier orders them reliably.

Interface: dilation_2d(x, filter, strides, rates, padding_mode, pads,
                       ceil_mode, data_format) -> y
"""

import torch
import tilelang
from tilelang import language as T


def torch_dtype_to_tl(dtype):
    """Map a torch dtype to the TileLang dtype string used by these kernels."""
    if dtype == torch.float16:
        return "float16"
    elif dtype == torch.bfloat16:
        return "bfloat16"
    elif dtype == torch.float32:
        return "float"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


_kernel_cache = {}
_inf_cache = {}
_zeros_cache = {}


def _inf_const(device, dtype, n):
    key = (str(device), dtype, n)
    if key not in _inf_cache:
        _inf_cache[key] = torch.full((n,), float("-inf"), dtype=dtype, device=device)
    return _inf_cache[key]


def _zeros_const(device, dtype, n):
    key = (str(device), dtype, n)
    if key not in _zeros_cache:
        _zeros_cache[key] = torch.zeros((n,), dtype=dtype, device=device)
    return _zeros_cache[key]


_GRID_LIMIT = 65535
_UB_BUDGET = 176 * 1024
_PERSISTENT_GRID = 44
# Odd-C tensors below this element count stay on Path C (4-kernel bridge
# launch overhead would dominate); value chosen between case 14 (4.8K) and
# the next smallest odd-C case (119K), verified by measurement in rev4.
_BRIDGE_MIN_ELEMS = 32768


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align(v: int, u: int) -> int:
    return _ceildiv(v, u) * u


def _out_geometry(in_h, in_w, fh, fw, sh, sw, rh, rw, padding_mode, pads):
    """Derive (Ho, Wo, pad_top, pad_left) exactly like the reference golden."""
    eff_fh = (fh - 1) * rh + 1
    eff_fw = (fw - 1) * rw + 1
    if padding_mode == "SAME":
        out_h = _ceildiv(in_h, sh)
        out_w = _ceildiv(in_w, sw)
        pad_h = max((out_h - 1) * sh + eff_fh - in_h, 0)
        pad_w = max((out_w - 1) * sw + eff_fw - in_w, 0)
        return out_h, out_w, pad_h // 2, pad_w // 2
    out_h = (in_h - eff_fh + sh) // sh
    out_w = (in_w - eff_fw + sw) // sw
    return out_h, out_w, int(pads[0]), int(pads[2])


def _pick_wb_ab(N, Ho, Wo, C, fh, fw, sh, sw, rh, rw, dtype_bytes):
    """Largest aligned-C output block that fits UB (win2d [+ phase plane])."""
    K2 = fh * fw
    cap = max(16, _align(Wo, 16))
    for wb in (128, 64, 48, 32, 16):
        if wb > cap:
            continue
        wspan = (wb - 1) * sw + (fw - 1) * rw + 1
        mws = _ceildiv(wspan, sw)
        win_ph = sw * mws * C if sw > 1 else 0
        ub = (wspan * C + win_ph + 3 * wb * C + K2 * C) * dtype_bytes
        grid = N * Ho * _ceildiv(Wo, wb)
        if ub <= _UB_BUDGET and grid <= _GRID_LIMIT:
            return wb
    return 16


def _pick_wb_c(C, K2, fw=3):
    """Flat-path block: frep rows + fw tap rows + acc must fit UB."""
    wb = max(16, min(128, (7800 // max(C, 1)) // 8 * 8))
    while wb > 16 and (K2 + 2) * _align(wb * C, 16) * 2 > _UB_BUDGET:
        wb -= 8
    return max(wb, 16)


# ---------------------------------------------------------------------------
# Path AB: aligned C — single tile kernel
# ---------------------------------------------------------------------------


def _pick_ry_wb(N, H, W, C, fh, fw, sh, rh, rw, pt, pl, Ho, Wo, dtype_bytes):
    """Multi-row params for narrow rows (Wo <= 64): RY output rows per CTA.

    Wall time ~ (fixed + RY*per-row) * ceil(grid/44): the optimal RY is the
    smallest one bringing grid to a single wave (<= 44 CTAs); when no RY
    reaches one wave within UB, take the largest RY that fits (amortizes the
    fixed cost).  Returns (RY, WB); (1, None) falls back to the single-row
    path.  Wide rows (cap > 64) gain nothing: the win fill would grow ~RYx.
    """
    K2 = fh * fw
    cap = max(16, _align(Wo, 16))
    if cap > 64 or Ho < 2:
        return 1, None
    wb = cap

    def fits(ry):
        wspan = wb + (fw - 1) * rw
        ryrows = (ry - 1) * sh + (fh - 1) * rh + 1
        ub = (ryrows * wspan * C + ry * wb * C + 2 * wb * C + K2 * C) * dtype_bytes
        grid = N * _ceildiv(Ho, ry)
        if ub > _UB_BUDGET or grid > _GRID_LIMIT or grid < 8:
            return None
        return grid

    # Empirical RY sweep (rev6): RY=4 wins for small row counts; beyond
    # ~180 total rows amortization dominates and the largest UB-fitting RY
    # (capped at 13) is best.
    if N * Ho >= 180:
        for ry in range(min(Ho, 13), 1, -1):
            grid = fits(ry)
            if grid is not None and grid >= 32:
                return ry, wb
        for ry in range(min(Ho, 13), 1, -1):
            if fits(ry) is not None:
                return ry, wb
    else:
        for ry in (4, 2):
            if fits(ry) is not None:
                return ry, wb
    return 1, None


def _build_ab_ry(N, H, W, C, fh, fw, sh, rh, rw, pt, pl, Ho, Wo, dtype_str, RY, WB):
    """s1 multi-row variant: RY output rows per CTA, flat 2-D UB buffers.

    Window rows for (ry, dy) map to win_flat row rr = ry*sh + rh*dy — r1
    shapes reuse overlapping rows (RY+fh-1 loads instead of RY*fh).  The
    broadcast runs once per tap for all RY rows.  T.tile.add only accepts
    2-D regions, hence flat layouts (verified by micro-benchmark).
    """
    K2 = fh * fw
    Ca = C
    WSPAN = WB + (fw - 1) * rw
    RYROWS = (RY - 1) * sh + (fh - 1) * rh + 1
    wblocks = _ceildiv(Wo, WB)
    hblocks = _ceildiv(Ho, RY)
    grid = N * hblocks * wblocks

    # Manual sync (Champion model): AUTO_SYNC would emit ~318 barriers per
    # CTA at RY=13 (3/row load loop, 2/(tap,row) tap loop, 3/row store loop)
    # — measured as ~50% of kernel time.  The single-pass structure (loads ->
    # compute -> store, no loop-carried MTE/V alternation) needs only 3 flag
    # pairs.  Event ids 1/2/3 within [0,8).
    pc = dict(PASS_CONFIGS)
    pc[tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC] = False

    @tilelang.jit(out_idx=[2], pass_configs=pc)
    def _dilate(dtype=dtype_str):
        @T.prim_func
        def main(
            X: T.Tensor((N, H, W, C), dtype),
            Ff: T.Tensor((K2 * C,), dtype),
            Y: T.Tensor((N, Ho, Wo, C), dtype),
        ):
            with T.Kernel(grid, threads=1, is_npu=True) as (cid):
                wb_i = cid % wblocks
                hb = (cid // wblocks) % hblocks
                b = cid // (wblocks * hblocks)
                ox0 = wb_i * WB
                oy0 = hb * RY

                fb = T.alloc_ub((K2 * Ca,), dtype)
                acc = T.alloc_ub((RY * WB, Ca), dtype)
                tmp = T.alloc_ub((WB, Ca), dtype)
                fbp = T.alloc_ub((WB, Ca), dtype)
                win = T.alloc_ub((RYROWS * WSPAN, Ca), dtype)

                T.tile.fill(acc, -T.infinity(dtype))
                T.tile.fill(win, -T.infinity(dtype))
                T.set_flag("v", "mte2", 2)
                T.wait_flag("v", "mte2", 2)
                T.copy(Ff[0 : K2 * C], fb[0 : K2 * Ca])
                wlo = ox0 - pl
                lg = T.Select(wlo < 0, -wlo, 0)
                rg = T.Select(wlo + WSPAN > W, wlo + WSPAN - W, 0)
                nload = WSPAN - lg - rg
                lo = wlo + lg
                for r in range(RYROWS):
                    iy = oy0 * sh - pt + r
                    if (iy >= 0) and (iy < H) and (nload > 0):
                        T.copy(X[b, iy, lo : lo + nload, 0:C], win[r * WSPAN + lg : r * WSPAN + lg + nload, 0:C], pad_value=float("-inf"))
                T.set_flag("mte2", "v", 1)
                T.wait_flag("mte2", "v", 1)
                for dy in range(fh):
                    for dx in range(fw):
                        t = dy * fw + dx
                        off = rw * dx
                        T.tile.broadcast(fbp, fb[t * Ca : t * Ca + Ca])
                        for ry in range(RY):
                            rr = (ry * sh + rh * dy) * WSPAN + off
                            T.tile.add(tmp, win[rr : rr + WB, 0:Ca], fbp)
                            T.tile.max(acc[ry * WB : ry * WB + WB, 0:Ca], acc[ry * WB : ry * WB + WB, 0:Ca], tmp)
                T.set_flag("v", "mte3", 3)
                T.wait_flag("v", "mte3", 3)
                vw = T.Select(ox0 + WB <= Wo, WB, Wo - ox0)
                for ry in range(RY):
                    if oy0 + ry < Ho:
                        T.copy(acc[ry * WB : ry * WB + vw, 0:C], Y[b, oy0 + ry, ox0 : ox0 + vw, 0:C])

        return main

    return _dilate(dtype=dtype_str)


def _build_ab(N, H, W, C, fh, fw, sh, sw, rh, rw, pt, pl, Ho, Wo, dtype_str):
    if sw == 1:
        return _build_ab_s1(N, H, W, C, fh, fw, sh, rh, rw, pt, pl, Ho, Wo, dtype_str)
    return _build_ab_ph(N, H, W, C, fh, fw, sh, sw, rh, rw, pt, pl, Ho, Wo, dtype_str)


def _build_ab_s1(N, H, W, C, fh, fw, sh, rh, rw, pt, pl, Ho, Wo, dtype_str):
    K2 = fh * fw
    Ca = C
    WB = _pick_wb_ab(N, Ho, Wo, C, fh, fw, sh, 1, rh, 1, 2 if dtype_str != "float" else 4)
    WSPAN = WB + (fw - 1) * rw
    wblocks = _ceildiv(Wo, WB)
    grid = N * Ho * wblocks

    # Manual sync (Champion model): per-dy flag pairs replace AUTO_SYNC's
    # per-op barriers.  (6) at loop top covers the previous dy's load (and
    # the fb load on dy 0); (2) orders fill -> load; (5) orders load -> taps;
    # (3) orders taps -> store.  (6) measures as free (~20ns/pair) — keep.
    pc = dict(PASS_CONFIGS)
    pc[tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC] = False

    @tilelang.jit(out_idx=[2], pass_configs=pc)
    def _dilate(dtype=dtype_str):
        @T.prim_func
        def main(
            X: T.Tensor((N, H, W, C), dtype),
            Ff: T.Tensor((K2 * C,), dtype),
            Y: T.Tensor((N, Ho, Wo, C), dtype),
        ):
            with T.Kernel(grid, threads=1, is_npu=True) as (cid):
                wb_i = cid % wblocks
                oy = (cid // wblocks) % Ho
                b = cid // (wblocks * Ho)
                ox0 = wb_i * WB

                fb = T.alloc_ub((K2 * Ca,), dtype)
                acc = T.alloc_ub((WB, Ca), dtype)
                tmp = T.alloc_ub((WB, Ca), dtype)
                fbp = T.alloc_ub((WB, Ca), dtype)
                win2d = T.alloc_ub((WSPAN, Ca), dtype)

                T.copy(Ff[0 : K2 * C], fb[0 : K2 * Ca])

                T.tile.fill(acc, -T.infinity(dtype))

                for dy in range(fh):
                    T.set_flag("mte2", "v", 6)
                    T.wait_flag("mte2", "v", 6)
                    iy = oy * sh - pt + rh * dy
                    if (iy >= 0) and (iy < H):
                        wlo = ox0 - pl
                        lg = T.Select(wlo < 0, -wlo, 0)
                        rg = T.Select(wlo + WSPAN > W, wlo + WSPAN - W, 0)
                        nload = WSPAN - lg - rg
                        lo = wlo + lg
                        T.tile.fill(win2d, -T.infinity(dtype))
                        T.set_flag("v", "mte2", 2)
                        T.wait_flag("v", "mte2", 2)
                        if nload > 0:
                            T.copy(X[b, iy, lo : lo + nload, 0:C], win2d[lg : lg + nload, 0:C], pad_value=float("-inf"))
                    else:
                        T.tile.fill(win2d, -T.infinity(dtype))
                    T.set_flag("mte2", "v", 5)
                    T.wait_flag("mte2", "v", 5)
                    for dx in range(fw):
                        t = dy * fw + dx
                        off = rw * dx
                        T.tile.broadcast(fbp, fb[t * Ca : t * Ca + Ca])
                        T.tile.add(tmp, win2d[off : off + WB, 0:Ca], fbp)
                        T.tile.max(acc, acc, tmp)

                T.set_flag("v", "mte3", 3)
                T.wait_flag("v", "mte3", 3)
                vw = T.Select(ox0 + WB <= Wo, WB, Wo - ox0)
                T.copy(acc[0:vw, 0:C], Y[b, oy, ox0 : ox0 + vw, 0:C])

        return main

    return _dilate(dtype=dtype_str)


def _build_ab_ph(N, H, W, C, fh, fw, sh, sw, rh, rw, pt, pl, Ho, Wo, dtype_str):
    K2 = fh * fw
    Ca = C
    WB = _pick_wb_ab(N, Ho, Wo, C, fh, fw, sh, sw, rh, rw, 2 if dtype_str != "float" else 4)
    WSPAN = (WB - 1) * sw + (fw - 1) * rw + 1
    MW = _ceildiv(WSPAN, sw)
    PSTRIDE = MW
    wblocks = _ceildiv(Wo, WB)
    grid = N * Ho * wblocks

    # Manual sync (Champion model), same protocol as _build_ab_s1 plus the
    # phase redistribution: (6) at loop top covers the previous dy's
    # load+redistribute (and the fb load on dy 0); (2) orders fill -> load;
    # (5) is a full barrier covering load+redistribute -> taps; (3) orders
    # taps -> store.
    pc = dict(PASS_CONFIGS)
    pc[tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC] = False

    @tilelang.jit(out_idx=[2], pass_configs=pc)
    def _dilate(dtype=dtype_str):
        @T.prim_func
        def main(
            X: T.Tensor((N, H, W, C), dtype),
            Ff: T.Tensor((K2 * C,), dtype),
            Y: T.Tensor((N, Ho, Wo, C), dtype),
        ):
            with T.Kernel(grid, threads=1, is_npu=True) as (cid):
                wb_i = cid % wblocks
                oy = (cid // wblocks) % Ho
                b = cid // (wblocks * Ho)
                ox0 = wb_i * WB

                fb = T.alloc_ub((K2 * Ca,), dtype)
                acc = T.alloc_ub((WB, Ca), dtype)
                tmp = T.alloc_ub((WB, Ca), dtype)
                fbp = T.alloc_ub((WB, Ca), dtype)
                win2d = T.alloc_ub((WSPAN, Ca), dtype)
                win_ph = T.alloc_ub((sw * PSTRIDE, Ca), dtype)

                T.copy(Ff[0 : K2 * C], fb[0 : K2 * Ca])

                T.tile.fill(acc, -T.infinity(dtype))

                for dy in range(fh):
                    T.set_flag("mte2", "v", 6)
                    T.wait_flag("mte2", "v", 6)
                    iy = oy * sh - pt + rh * dy
                    if (iy >= 0) and (iy < H):
                        wlo = ox0 * sw - pl
                        lg = T.Select(wlo < 0, -wlo, 0)
                        rg = T.Select(wlo + WSPAN > W, wlo + WSPAN - W, 0)
                        nload = WSPAN - lg - rg
                        lo = wlo + lg
                        T.tile.fill(win2d, -T.infinity(dtype))
                        T.set_flag("v", "mte2", 2)
                        T.wait_flag("v", "mte2", 2)
                        if nload > 0:
                            T.copy(X[b, iy, lo : lo + nload, 0:C], win2d[lg : lg + nload, 0:C], pad_value=float("-inf"))
                        # The redistribution's copy_ub_to_ub runs on a
                        # different MTE pipe than the GM load: without this
                        # barrier it reads the -inf fill instead of the
                        # loaded data (observed as all -inf output).
                        T.barrier_all()
                        for r in range(WSPAN):
                            T.copy(win2d[r, 0:Ca], win_ph[(r % sw) * PSTRIDE + r // sw, 0:Ca])
                    else:
                        T.tile.fill(win_ph, -T.infinity(dtype))
                    # copy_ub_to_ub is MTE but NOT mte2 (an mte2->v flag
                    # leaves the taps racing it, observed as all -inf output)
                    # — a full barrier is the only reliable ordering here.
                    T.barrier_all()
                    for dx in range(fw):
                        t = dy * fw + dx
                        off = rw * dx
                        p = off % sw
                        m = off // sw
                        T.tile.broadcast(fbp, fb[t * Ca : t * Ca + Ca])
                        T.tile.add(tmp, win_ph[p * PSTRIDE + m : p * PSTRIDE + m + WB, 0:Ca], fbp)
                        T.tile.max(acc, acc, tmp)

                T.set_flag("v", "mte3", 3)
                T.wait_flag("v", "mte3", 3)
                vw = T.Select(ox0 + WB <= Wo, WB, Wo - ox0)
                T.copy(acc[0:vw, 0:C], Y[b, oy, ox0 : ox0 + vw, 0:C])

        return main

    return _dilate(dtype=dtype_str)


# ---------------------------------------------------------------------------
# Path C: odd C, sw == 1 — flat tiles + host-expanded filter + persistent grid
# ---------------------------------------------------------------------------


def _build_c(N, H, W, C, fh, fw, sh, sw, rh, rw, pt, pl, Ho, Wo, dtype_str):
    K2 = fh * fw
    WB = _pick_wb_c(C, K2)
    WBC = WB * C
    WBCP = _align(WBC, 16)
    wblocks = _ceildiv(Wo, WB)
    nblocks = N * Ho * wblocks
    grid = min(nblocks, _PERSISTENT_GRID)
    nit = _ceildiv(nblocks, grid)
    LGMAX = max(1, pl * C)

    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
    def _dilate(dtype=dtype_str):
        @T.prim_func
        def main(
            Xf: T.Tensor((N * H * W * C,), dtype),
            frep: T.Tensor((K2, WBC), dtype),
            Yf: T.Tensor((N * Ho * Wo * C,), dtype),
        ):
            with T.Kernel(grid, threads=1, is_npu=True) as (cid):
                frep_ub = T.alloc_ub((K2, WBCP), dtype)
                buf16 = T.alloc_ub((16,), dtype)
                tmp = T.alloc_ub((WBCP,), dtype)
                acc = T.alloc_ub((WBCP,), dtype)

                for t in range(K2):
                    T.copy(frep[t, 0:WBC], frep_ub[t, 0:WBC])

                for it in range(nit):
                    blk = cid + it * grid
                    if blk < nblocks:
                        wb_i = blk % wblocks
                        oy = (blk // wblocks) % Ho
                        b = blk // (wblocks * Ho)
                        ox0 = wb_i * WB

                        T.tile.fill(acc, -T.infinity(dtype))
                        for dy in range(fh):
                            iy = oy * sh - pt + rh * dy
                            if (iy >= 0) and (iy < H):
                                for dx in range(fw):
                                    t = dy * fw + dx
                                    ix0 = ox0 - pl + rw * dx
                                    lg = T.Select(ix0 < 0, -ix0, 0)
                                    rg = T.Select(ix0 + WB > W, ix0 + WB - W, 0)
                                    nload = WB - lg - rg
                                    # The per-tap -inf fill is load-bearing
                                    # twice: it restores the boundary gaps
                                    # AND its V->MTE2 ordering covers the
                                    # cross-iteration WAR (compute of tap k
                                    # reads tmp while tap k+1's load
                                    # overwrites it), which AUTO_SYNC cannot
                                    # see across loop iterations.
                                    T.tile.fill(tmp, -T.infinity(dtype))
                                    if nload > 0:
                                        gbase = ((b * H + iy) * W + ix0) * C
                                        if gbase >= 0:
                                            T.copy(Xf[gbase : gbase + (lg + nload) * C], tmp[0 : (lg + nload) * C], pad_value=float("-inf"))
                                            if ix0 < 0:
                                                for q in range(LGMAX):
                                                    if q < (-ix0) * C:
                                                        tmp[q] = -T.infinity(dtype)
                                        else:
                                            # Corner (first row of the first
                                            # batch): the window starts
                                            # before the tensor, so the
                                            # unclamped read would touch
                                            # unmapped GM. Land the in-row
                                            # data on the next 32B boundary
                                            # and patch the sub-16-element
                                            # head via scalar writes.
                                            R = (16 - ((-ix0) * C) % 16) % 16
                                            T.copy(Xf[0:16], buf16[0:16])
                                            for q in range(16):
                                                if q < R:
                                                    tmp[(-ix0) * C + q] = buf16[q]
                                            T.copy(Xf[R : nload * C], tmp[lg * C + R : (lg + nload) * C], pad_value=float("-inf"))
                                            if ix0 < 0:
                                                for q in range(LGMAX):
                                                    if q < (-ix0) * C:
                                                        tmp[q] = -T.infinity(dtype)
                                    for p in T.Parallel(WBC):
                                        acc[p] = T.max(acc[p], tmp[p] + frep_ub[t, p])
                            else:
                                for dx in range(fw):
                                    t = dy * fw + dx
                                    T.tile.fill(tmp, -T.infinity(dtype))
                                    for p in T.Parallel(WBC):
                                        acc[p] = T.max(acc[p], tmp[p] + frep_ub[t, p])

                        vw = T.Select(ox0 + WB <= Wo, WB, Wo - ox0)
                        obase = ((b * Ho + oy) * Wo + ox0) * C
                        T.copy(acc[0 : vw * C], Yf[obase : obase + vw * C])

        return main

    return _dilate(dtype=dtype_str)


# ---------------------------------------------------------------------------
# Path D: odd C, sw >= 2 — rev1 per-pixel fallback
# ---------------------------------------------------------------------------


def _pick_pixels_per_block(N, Ho, Wo):
    pb = 1
    while pb < Wo and N * Ho * _ceildiv(Wo, pb) > _GRID_LIMIT:
        pb *= 2
    return pb


def _build_d(N, H, W, C, fh, fw, sh, sw, rh, rw, pt, pl, Ho, Wo, dtype_str):
    K2 = fh * fw
    Ca = _align(C, 16)
    pb = _pick_pixels_per_block(N, Ho, Wo)

    @tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
    def _dilate(dtype=dtype_str):
        @T.prim_func
        def main(
            X: T.Tensor((N, H, W, C), dtype),
            F: T.Tensor((fh, fw, C), dtype),
            Y: T.Tensor((N, Ho, Wo, C), dtype),
        ):
            with T.Kernel(N * Ho * T.ceildiv(Wo, pb), threads=1, is_npu=True) as (cid):
                grp = cid % T.ceildiv(Wo, pb)
                oy = (cid // T.ceildiv(Wo, pb)) % Ho
                b = cid // (T.ceildiv(Wo, pb) * Ho)
                ox0 = grp * pb

                row = T.alloc_shared((Ca,), dtype)
                tmp = T.alloc_shared((Ca,), dtype)
                acc = T.alloc_shared((pb, Ca), dtype)
                fb = T.alloc_shared((K2 * Ca,), dtype)

                T.tile.fill(fb, 0.0)
                for t in range(K2):
                    T.copy(F[t // fw, t % fw, 0:C], fb[t * Ca : t * Ca + C])

                T.tile.fill(acc, -T.infinity(dtype))
                for p in range(pb):
                    ox = ox0 + p
                    if ox < Wo:
                        for dy in range(fh):
                            iy = oy * sh - pt + rh * dy
                            for dx in range(fw):
                                ix = ox * sw - pl + rw * dx
                                t = dy * fw + dx
                                T.tile.fill(row, -T.infinity(dtype))
                                if (iy >= 0) and (iy < H) and (ix >= 0) and (ix < W):
                                    T.copy(X[b, iy, ix, 0:C], row[0:C])
                                T.tile.add(tmp, row, fb[t * Ca : t * Ca + Ca])
                                T.tile.max(acc[p, 0:Ca], acc[p, 0:Ca], tmp)
                        T.copy(acc[p, 0:C], Y[b, oy, ox, 0:C])

        return main

    return _dilate(dtype=dtype_str)


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


def _get_kernel(kind, N, H, W, C, fh, fw, sh, sw, rh, rw, pt, pl, Ho, Wo, dtype_str):
    key = (kind, N, H, W, C, fh, fw, sh, sw, rh, rw, pt, pl, Ho, Wo, dtype_str)
    if key not in _kernel_cache:
        if kind == "ab_ry":
            ry, wb = _pick_ry_wb(N, H, W, C, fh, fw, sh, rh, rw, pt, pl, Ho, Wo, 2 if dtype_str != "float" else 4)
            _kernel_cache[key] = _build_ab_ry(N, H, W, C, fh, fw, sh, rh, rw, pt, pl, Ho, Wo, dtype_str, ry, wb)
        else:
            builder = {"ab": _build_ab, "c": _build_c, "d": _build_d}[kind]
            _kernel_cache[key] = builder(N, H, W, C, fh, fw, sh, sw, rh, rw, pt, pl, Ho, Wo, dtype_str)
    return _kernel_cache[key]


def dilation_2d(
    x: torch.Tensor,
    filter: torch.Tensor,
    strides: list,
    rates: list,
    padding_mode: str = "SAME",
    pads: list = None,
    ceil_mode: bool = False,
    data_format: str = "NHWC",
) -> torch.Tensor:
    """2D morphological dilation with a per-channel structuring element.

    ``y[b, oy, ox, c] = max_{dy,dx} (x[b, oy*sh + rh*dy, ox*sw + rw*dx, c]
                                     + filter[dy, dx, c])``
    """
    if pads is None:
        pads = [0, 0, 0, 0]
    if data_format != "NHWC":
        raise ValueError(f"only NHWC is supported, got {data_format!r}")
    if padding_mode not in ("SAME", "VALID"):
        raise ValueError(f"unsupported padding_mode {padding_mode!r}")
    if x.dim() != 4:
        raise ValueError(f"x must be 4D [N,H,W,C], got {tuple(x.shape)}")
    if filter.dim() != 3:
        raise ValueError(f"filter must be 3D [fh,fw,C], got {tuple(filter.shape)}")
    if x.dtype != filter.dtype:
        raise ValueError("x and filter dtypes must match")
    # bfloat16 is rejected: it is not in the cann-bench proto and the
    # kernels cannot compile for it (the tilelang frontend cannot lower
    # T.infinity("bfloat16") for the -inf boundary fills).
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"unsupported dtype {x.dtype}")
    if not x.is_contiguous():
        x = x.contiguous()
    if not filter.is_contiguous():
        filter = filter.contiguous()

    N, H, W, C = x.shape
    fh, fw, fc = filter.shape
    if fc != C:
        raise ValueError(f"filter channel dim ({fc}) must equal x channel dim ({C})")

    sh, sw = int(strides[1]), int(strides[2])
    rh, rw = int(rates[1]), int(rates[2])
    if sh < 1 or sw < 1 or rh < 1 or rw < 1:
        raise ValueError(f"strides/rates must be >= 1, got {strides} {rates}")

    Ho, Wo, pt, pl = _out_geometry(H, W, fh, fw, sh, sw, rh, rw, padding_mode, pads)
    if Ho < 1 or Wo < 1:
        raise ValueError(f"invalid output size Ho={Ho} Wo={Wo}")

    dtype_str = torch_dtype_to_tl(x.dtype)
    align_unit = 16 if x.dtype != torch.float32 else 8
    K2 = fh * fw
    if C % align_unit == 0:
        kind = "ab"
        kern_c = C
    elif sw == 1 and N * H * W * C < _BRIDGE_MIN_ELEMS:
        kind = "c"
        kern_c = C
    else:
        # Pad-bridge: pad C to a 16-multiple, reuse Path AB machinery.
        # Padding channels contribute 0 + (-inf) = -inf: real channels exact.
        kind = "bridge"
        kern_c = _align(C, align_unit)

    kern_kind = "ab" if kind == "bridge" else kind
    if sw == 1 and kind in ("ab", "bridge"):
        ry, _ = _pick_ry_wb(N, H, W, kern_c, fh, fw, sh, rh, rw, pt, pl, Ho, Wo, 2 if x.dtype != torch.float32 else 4)
        if ry > 1:
            kern_kind = "ab_ry"

    kernel = _get_kernel(kern_kind, N, H, W, kern_c, fh, fw, sh, sw, rh, rw, pt, pl, Ho, Wo, dtype_str)

    if kind == "ab":
        return kernel(x, filter.reshape(-1)).view(N, Ho, Wo, C)
    if kind == "bridge":
        Cp = kern_c
        pad_n = Cp - C
        # cat-based pads: PadV3 runs ~250GB/s and costs 2 kernels (MemSet +
        # PadV3) even for tiny tensors; ConcatD is 9x faster on large tensors
        # and the cached -inf row makes the filter pad a single cheap cat.
        x_pad = torch.cat([x, _zeros_const(x.device, x.dtype, N * H * W * pad_n).view(N, H, W, pad_n)], dim=-1)
        f_pad = torch.cat([filter.reshape(K2, C), _inf_const(filter.device, filter.dtype, K2 * pad_n).view(K2, pad_n)], dim=-1).reshape(-1)
        # tilelang launches on raw stream 0 while torch ops use the torch
        # stream: without a device sync the kernel can read x_pad/f_pad
        # before the cat kernels finish (observed as first-call corruption).
        torch.npu.synchronize()
        y_pad = kernel(x_pad, f_pad)
        torch.npu.synchronize()
        return y_pad[..., :C].contiguous()
    if kind == "c":
        K2 = fh * fw
        WB = _pick_wb_c(C, K2)
        frep = filter.reshape(K2, 1, C).expand(K2, WB, C).reshape(K2, WB * C).contiguous()
        torch.npu.synchronize()
        yf = kernel(x.reshape(-1), frep)
        torch.npu.synchronize()
        return yf.view(N, Ho, Wo, C)
    return kernel(x, filter)


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------


def _golden_dilation(x, flt, sh, sw, rh, rw, pt, pl, Ho, Wo):
    """fp64 CPU reference: per-tap windows, out-of-range taps = -inf + f."""
    x64, f64 = x.double(), flt.double()
    N, H, W, C = x.shape
    fh, fw, _ = flt.shape
    y = None
    for dy in range(fh):
        for dx in range(fw):
            oys = torch.arange(Ho) * sh - pt + rh * dy
            oxs = torch.arange(Wo) * sw - pl + rw * dx
            vy = (oys >= 0) & (oys < H)
            vx = (oxs >= 0) & (oxs < W)
            tap = x64[:, oys.clamp(0, H - 1)][:, :, oxs.clamp(0, W - 1)] + f64[dy, dx].view(1, 1, 1, C)
            oor = torch.full((C,), float("-inf")) + f64[dy, dx]
            mask = vy.view(1, Ho, 1, 1) & vx.view(1, 1, Wo, 1)
            tap = torch.where(mask, tap, oor.view(1, 1, 1, C))
            y = tap if y is None else torch.maximum(y, tap)
    return y


if __name__ == "__main__":
    # One representative shape per dispatch path (ab_ry / s1 / ph / bridge /
    # flat Path C) against the fp64 golden.
    torch.manual_seed(0)
    bounds = {torch.float16: 2e-3, torch.float32: 1e-5}
    for shape, dtype, k, s, d, mode in (
        ((2, 64, 64, 64), torch.float16, (3, 3), (1, 1), (1, 1), "SAME"),  # ab_ry
        ((1, 64, 128, 128), torch.float16, (3, 3), (1, 1), (1, 1), "SAME"),  # ab_s1 (wide rows)
        ((1, 16, 128, 128), torch.float32, (3, 3), (2, 2), (1, 1), "SAME"),  # ab_ph (strided)
        ((1, 64, 64, 63), torch.float16, (3, 3), (1, 1), (1, 1), "SAME"),  # bridge -> ab_ry
        ((1, 11, 13, 17), torch.float32, (3, 3), (1, 1), (1, 1), "SAME"),  # path C (flat)
    ):
        N, H, W, C = shape
        fh, fw = k
        x = (torch.rand(N, H, W, C) * 2 - 1).to(dtype)
        flt = (torch.rand(fh, fw, C) * 0.2 - 0.1).to(dtype)
        Ho = (H + s[0] - 1) // s[0]
        Wo = (W + s[1] - 1) // s[1]
        pt = max((Ho - 1) * s[0] + (fh - 1) * d[0] + 1 - H, 0) // 2
        pl = max((Wo - 1) * s[1] + (fw - 1) * d[1] + 1 - W, 0) // 2
        y = dilation_2d(x.npu(), flt.npu(), [1, s[0], s[1], 1], [1, d[0], d[1], 1], mode, [0, 0, 0, 0], False, "NHWC").cpu()
        ref = _golden_dilation(x, flt, s[0], s[1], d[0], d[1], pt, pl, Ho, Wo).to(dtype)
        diff = (y.float() - ref.float()).abs().max().item()
        print(f"[SMOKE] {shape} {str(dtype).split('.')[-1]} k={k} s={s} d={d} max_abs={diff:.3e}")
        assert diff <= bounds[dtype], f"smoke test failed: {diff}"
    # CI pass criterion (examples/bench_test.sh) requires the output to end
    # with this phrase (or "TEST PASSED!"), same as the other examples.
    print("Kernel Output Match!")
