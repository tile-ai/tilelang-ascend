"""DepthwiseConv2D TileLang kernel (Expert mode, pure Vector).

Design (rev2, perf-tuned):
  Kernel 1 (pad): zero-pad x into GM workspace xp[NC, Hp, Wp].
  Kernel 2 (conv): MULTI-ROW tile.  Each block owns RB consecutive output rows,
    BW output columns and _BLOCK_C channels (split over the 2 vector cores).
    For each kernel column kw a SINGLE 2-D strided load brings in all
    NR = (RB-1)*SH + (KH-1)*DH + 1 source rows with the per-kw column shift
    absorbed into the (unalignment-tolerant) GM base.  Every loaded row is then
    reused for each of the RB x KH (output row, kh) combinations -- for unit
    stride a whole kernel row is one [RB, BW] region op.

  Why: rev1 loaded one row window per (kh, kw) tap -- 4 pipeline
  synchronisations per tap for only BW outputs.  msprof showed 85% of the AIV
  time in `scalar_wait` with vector utilisation 0.46%.  Loading a [NR, WIN_W]
  tile once per (channel, kw) and accumulating whole regions cuts
  synchronisation points per output by roughly two orders of magnitude while
  keeping every UB slice 32B-aligned (WIN_W is a multiple of 16*SW).

  fp32 accumulation with Knuth-2Sum compensation (the Ascend vector unit has no
  fused multiply-add, so plain accumulate gives ~2x the rounding error of
  native FMA and fails the evaluator's cancellation fallback), inf/NaN guarded.

Interface: depthwise_conv_2d(x, weight, bias, kernelSize, stride, padding,
                             dilation, groups) -> y
"""

# ruff: noqa: SIM102
# The nested guard chains in the conv kernel (dtype gate / channel bound /
# validity prefix) are intentionally kept nested: TVMScript folds each
# Python-level condition statically only in this form — merging them with
# `and` lowers to combined runtime branches and changes the generated code
# away from the evaluated binary (verified by lowering diff).  The kernel
# body stays equivalent to the CANN-BENCH evaluated submission.

import torch
import tilelang
import tilelang.ascend.a2a3.language as T


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

# conv runs without AUTO_SYNC: intra-core syncs are issued manually
# (champion model); pad/1x1 keep the automatic pass.
CHAMPION_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_kernel_cache = {}

_BLOCK_C = 8  # channels per block (vid halves -> 4 per vector core)
_RB = 8  # output rows per block (min; grown adaptively)
_BW_MAX = 128  # max output columns per block


def _ceildiv(a, b):
    return (a + b - 1) // b


def _pad16(v):
    return _ceildiv(v, 16) * 16


def _pad_to(v, a):
    return _ceildiv(v, a) * a


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def _dwconv_pad(H, W, Hp, Wp, Wc, NC, ph, pw, dtype="float16"):
    """xp[nc, i, j] = x[nc, i-ph, j-pw] with zeros outside the image.

    Hp is padded to a multiple of RB so each vector core's row count is static,
    and the two cores of a mix block take different row halves instead of
    duplicating the same copy.  Only the first `pw+W+pw` columns are written:
    the conv only ever reads beyond them for output columns that are never
    stored, so the wider tail of Wp may stay uninitialised.
    """
    # pad time tracks the issued statement count (~35ns each), not bytes:
    # one 256-row band per channel and 2 channels per block minimise the
    # zero-fill + copy statements while keeping both UB buffers < 70KB.
    RB = 256
    HRB = RB // 2  # rows per vector core (each block runs on both vids)
    CPB = 2  # channel planes per block
    nrb = (Hp + RB - 1) // RB
    ncb = (NC + CPB - 1) // CPB

    @T.prim_func
    def main(x: T.Tensor((NC, H, W), dtype), xp: T.Tensor((NC, Hp, Wp), dtype)):
        with T.Kernel(ncb * nrb, is_npu=True) as (cid, vid):
            rb = cid % nrb
            cb = cid // nrb
            i0 = rb * RB + vid * HRB
            # Hp is not rounded up to the band size: clamp the last band
            # (and a fully-out-of-range upper vid half) instead of writing
            # tail rows the conv never reads.
            zr = T.Select(i0 + HRB <= Hp, HRB, Hp - i0)
            sv0 = T.Select(i0 - ph > 0, T.Select(i0 - ph < H, i0 - ph, H), 0)
            sv1 = T.Select(i0 - ph + HRB > 0, T.Select(i0 - ph + HRB < H, i0 - ph + HRB, H), 0)
            nrows = sv1 - sv0

            zero_buf = T.alloc_ub((HRB, Wp), dtype)
            # Wc keeps the staging pitch 32B-aligned; the data load's tail
            # masking zero-fills the pad columns (see the copy below)
            data_buf = T.alloc_ub((HRB, Wc), dtype)
            if zr > 0:
                T.tile.fill(zero_buf, 0.0)
            for j in range(CPB):
                nc = cb * CPB + j
                if nc < NC:
                    if zr > 0:
                        T.copy(zero_buf[0:zr, 0:Wp], xp[nc, i0 : i0 + zr, 0:Wp])
                    if nrows > 0:
                        # W-wide GM source into a Wc-wide UB view: the load's
                        # tail masking pads the (Wc - W) pad columns with
                        # zeros, so both the load and the store see the same
                        # 32B-aligned row pitch
                        T.copy(x[nc, sv0:sv1, 0:W], data_buf[0:nrows, 0:Wc])
                        T.copy(data_buf[0:nrows, 0:Wc], xp[nc, sv0 + ph : sv0 + ph + nrows, pw : pw + Wc])

    return main


_1X1_BR = 64  # channel planes per block
_1X1_BCW = 256  # columns (H*W) per block


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def _dwconv_1x1(NC, C, HW, dtype="float16"):
    """y[nc, j] = bias[c] + x[nc, j] * weight[c] for K=1, S=1, P=0, D=1.

    x is viewed as [N*C, H*W] rows.  C % BR == 0 keeps every block's plane
    band inside one image, so weight/bias slices stay contiguous; HW % 16
    keeps the 2-D strided load's row stride legal.
    """
    BR, BCW = _1X1_BR, _1X1_BCW
    HR = BR // 2  # rows per vector core (each block runs on both vids)
    col_bands = _ceildiv(HW, BCW)
    grid = _ceildiv(NC, BR) * col_bands

    @T.prim_func
    def main(x: T.Tensor((NC, HW), dtype), weight: T.Tensor((C,), dtype), bias: T.Tensor((C,), dtype), y: T.Tensor((NC, HW), dtype)):
        with T.Kernel(grid, is_npu=True) as (cid, vid):
            cb = cid % col_bands
            rb = cid // col_bands
            r0 = rb * BR + vid * HR
            c0 = cb * BCW
            ncw = T.Select(c0 + BCW <= HW, BCW, HW - c0)
            x_buf = T.alloc_ub((HR, BCW), dtype)
            x32 = T.alloc_ub((HR, BCW), "float")
            acc = T.alloc_ub((HR, BCW), "float")
            y_buf = T.alloc_ub((HR, BCW), dtype)
            w_vec = T.alloc_ub((HR,), dtype)
            w32 = T.alloc_ub((HR,), "float")
            b_vec = T.alloc_ub((HR,), dtype)
            b32 = T.alloc_ub((HR,), "float")
            T.copy(x[r0 : r0 + HR, c0 : c0 + ncw], x_buf[0:HR, 0:ncw])
            T.tile.cast(x32, x_buf, "CAST_NONE", HR * BCW)
            T.copy(weight[r0 % C : r0 % C + HR], w_vec[0:HR])
            T.tile.cast(w32, w_vec, "CAST_NONE", HR)
            T.copy(bias[r0 % C : r0 % C + HR], b_vec[0:HR])
            T.tile.cast(b32, b_vec, "CAST_NONE", HR)
            for r in range(HR):
                T.tile.fill(acc[r, 0:BCW], b32[r])
                T.tile.axpy(acc[r, 0:BCW], x32[r, 0:BCW], w32[r])
            T.tile.cast(y_buf, acc, "CAST_RINT", HR * BCW)
            if c0 + BCW <= HW:
                # full band: slice width equals the buffer pitch, a legal
                # multi-row DMA source
                T.copy(y_buf[0:HR, 0:ncw], y[r0 : r0 + HR, c0 : c0 + ncw])
            else:
                # tail band: store row by row (see the conv epilogue note)
                for r in range(HR):
                    T.copy(y_buf[r, 0:ncw], y[r0 + r, c0 : c0 + ncw])

    return main


def _conv_geometry(N, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, Ho, Wo, dtype_str):
    """Return the geometry shared by host and both kernels."""
    # T.tile.compare needs 256B-aligned operands: fp32 -> multiples of 64.
    is_fp32 = dtype_str == "float"
    bw_align = 64 if is_fp32 else 16
    BW = min(_BW_MAX, max(bw_align, _pad_to(Wo, bw_align)))
    w_tiles = max(1, _ceildiv(Wo, BW))
    # Wider row tiles halve block count and region-op count, but only pay off
    # when enough blocks remain to fill the vector cores, and never for the
    # fp32 per-row chain (its 1-D ops do not widen with RB).
    blocks8 = N * _ceildiv(C, _BLOCK_C) * max(1, _ceildiv(Ho, 8)) * w_tiles
    RB = 16 if (not is_fp32 and blocks8 >= 64) else 8
    h_tiles = max(1, _ceildiv(Ho, RB))

    pow2_sw = SW in (1, 2, 4)
    lv = {1: 0, 2: 1, 4: 2}[SW] if pow2_sw else 0
    # 32B UB alignment expressed in elements of a 2-byte dtype
    a = 16 * SW if pow2_sw else 16
    m = 16 * SW if pow2_sw else 16
    rho_of = [(kw * DW) % m for kw in range(KW)]
    q_span = max([kw * DW - rho_of[kw] for kw in range(KW)]) if KW > 0 else 0
    # WIN_W sized so the phase-extracted source row is exactly BW wide, which is
    # what lets a whole kernel row be one [RB, BW] region op.
    win_w = _pad_to(q_span + BW * SW, a)

    nr = (RB - 1) * SH + (KH - 1) * DH + 1

    hp = H + 2 * PH
    hp = max(hp, (h_tiles - 1) * RB * SH + nr)
    # pad's staging column pitch: multi-row UB->GM row copies need a
    # 32B-multiple source pitch, so data rows are staged at the 32B-rounded
    # width and the store carries the pad columns (zeros) into xp's padding.
    wc_align = 16 if not is_fp32 else 8
    Wc = _pad_to(W, wc_align)
    wp = PW + W + PW
    wp = max(wp, PW + Wc)
    if KW > 0:
        wp = max(wp, (w_tiles - 1) * BW * SW + max(rho_of) + win_w)
    # xp row stride must stay a multiple of 16: the conv's 2-D strided load
    # silently produces garbage for other strides.
    wp = _pad16(wp)
    unit = SH == 1 and DH == 1 and (win_w >> lv) == BW
    return dict(
        BW=BW,
        RB=RB,
        w_tiles=w_tiles,
        h_tiles=h_tiles,
        lv=lv,
        win_w=win_w,
        nr=nr,
        Hp=hp,
        Wp=wp,
        Wc=Wc,
        pow2_sw=pow2_sw,
        region=(not is_fp32 and unit),
        region32=(is_fp32 and unit),
    )


def _build_conv(N, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, Ho, Wo, dtype_str):
    """Return (pad_kernel, conv_kernel) with all geometry baked in."""
    NC = N * C
    K2 = KH * KW
    K2_pad = _pad16(K2)
    g = _conv_geometry(N, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, Ho, Wo, dtype_str)
    BW, RB = g["BW"], g["RB"]
    w_tiles, h_tiles = g["w_tiles"], g["h_tiles"]
    lv, win_w, nr = g["lv"], g["win_w"], g["nr"]
    Hp, Wp = g["Hp"], g["Wp"]
    pow2_sw, region = g["pow2_sw"], g["region"]
    region32 = g["region32"]

    block_C = _BLOCK_C
    c_tiles = _ceildiv(C, block_C)
    half_C = block_C // 2
    half_pad = _pad16(half_C)
    grid = N * c_tiles * h_tiles * w_tiles
    is_fp32 = dtype_str == "float"
    sw = SW
    # Modulus that makes the per-kw column shift land on an M-aligned slice
    # start, so the shift can be absorbed into the (alignment-free) GM address.
    m_mod = 16 * SW if pow2_sw else 16

    conv_cfg = CHAMPION_CONFIGS if pow2_sw else PASS_CONFIGS

    @tilelang.jit(out_idx=[-1], pass_configs=conv_cfg)
    def _dwconv_conv(dtype=dtype_str):
        @T.prim_func
        def main(
            xp: T.Tensor((NC, Hp, Wp), dtype),
            weight2: T.Tensor((C, K2), dtype),
            bias: T.Tensor((C,), dtype),
            y: T.Tensor((NC, Ho, Wo), dtype),
        ):
            with T.Kernel(grid, is_npu=True) as (cid, vid):
                wt_i = cid % w_tiles
                ht_i = (cid // w_tiles) % h_tiles
                nct = cid // (w_tiles * h_tiles)
                ct = nct % c_tiles
                n = nct // c_tiles
                nc0 = n * C + ct * block_C  # first plane index (x/xp/y side)
                ch0 = ct * block_C  # first channel index (weight/bias)
                ow = wt_i * BW
                oh0 = ht_i * RB
                r0 = oh0 * SH
                vw = T.Select(ow + BW <= Wo, BW, Wo - ow)
                rows_v = T.Select(oh0 + RB <= Ho, RB, Ho - oh0)
                c_lo = vid * half_C

                acc = T.alloc_ub((RB, BW), "float")
                comp = T.alloc_ub((RB, BW), "float")
                tileA = T.alloc_ub((nr, win_w), dtype)
                tileB = T.alloc_ub((nr, win_w), dtype)
                tile32 = T.alloc_ub((nr, win_w), "float")
                gsrc = T.alloc_ub((nr, win_w >> lv), "float")
                gsrc1 = T.alloc_ub((nr, win_w >> 1), "float")
                w_row = T.alloc_ub((K2_pad,), dtype)
                w_row32 = T.alloc_ub((K2_pad,), "float")
                b_vec = T.alloc_ub((half_pad,), dtype)
                b_vec32 = T.alloc_ub((half_pad,), "float")
                y_allA = T.alloc_ub((RB, BW), dtype)
                y_allB = T.alloc_ub((RB, BW), dtype)
                # fp32 region-form compensation scratch: one [RB, BW] tile
                # each; every op is whole-buffer or a proven region form.
                wb2d = T.alloc_ub((RB, BW), "float")
                prod = T.alloc_ub((RB, BW), "float")
                tb2 = T.alloc_ub((RB, BW), "float")
                zb2 = T.alloc_ub((RB, BW), "float")
                e21 = T.alloc_ub((RB, BW), "float")
                e22 = T.alloc_ub((RB, BW), "float")
                e23 = T.alloc_ub((RB, BW), "float")
                cc2 = T.alloc_ub((RB, BW), "float")
                # fp32 uses 1-D per-row scratch: whole-buffer operands keep the
                # dataflow unambiguous (partial-region chains are not reliable
                # for the fp32 compensated path on this backend).
                prod1 = T.alloc_ub((BW,), "float")
                w_b = T.alloc_ub((BW,), "float")
                tb1 = T.alloc_ub((BW,), "float")
                zb1 = T.alloc_ub((BW,), "float")
                ea1 = T.alloc_ub((BW,), "float")
                ea2 = T.alloc_ub((BW,), "float")
                ea3 = T.alloc_ub((BW,), "float")
                cn1 = T.alloc_ub((BW,), "float")
                ac1 = T.alloc_ub((BW,), "float")
                cm1 = T.alloc_ub((BW // 8,), "uint8")
                rs1 = T.alloc_ub((BW,), "float")

                vc_b = T.Select(ch0 + c_lo + half_C <= C, half_C, C - (ch0 + c_lo))
                if vc_b > 0:
                    T.copy(bias[ch0 + c_lo : ch0 + c_lo + vc_b], b_vec[0:vc_b])
                if is_fp32:
                    T.copy(bias[ch0 + c_lo : ch0 + c_lo + vc_b], b_vec32[0:vc_b])
                T.set_flag("mte2", "v", 0)
                # prefetch channel 0's first window + weights: their load
                # latency hides behind the bias wait and the tap compute
                if ch0 + c_lo < C:
                    T.copy(xp[nc0 + c_lo, r0 : r0 + nr, (ow * sw) : (ow * sw) + win_w], tileA[0:nr, 0:win_w])
                    T.set_flag("mte2", "v", 2)
                    if is_fp32:
                        T.copy(weight2[ch0 + c_lo, 0:K2], w_row32[0:K2])
                    else:
                        T.copy(weight2[ch0 + c_lo, 0:K2], w_row[0:K2])
                    T.set_flag("mte2", "v", 1)
                T.wait_flag("mte2", "v", 0)
                if not is_fp32:
                    T.tile.cast(b_vec32, b_vec, "CAST_NONE", half_pad)

                for c_local in range(half_C):
                    c_abs = nc0 + c_lo + c_local  # plane index
                    c_ch = ch0 + c_lo + c_local  # channel index
                    if c_ch < C:
                        # wait for the PREVIOUS channel's MTE3 store.  The
                        # set was issued at its channel end (adjacent across
                        # the loop edge), and MTE3 executes in order, so this
                        # depth-1 wait transitively guarantees the store two
                        # channels back landed before its ping-pong buffer is
                        # recast.  fp32 skips this: its per-row barrier_all
                        # both resets armed event ids and already covers rs1.
                        if not is_fp32:
                            if c_local >= 1:
                                if c_local % 2 == 0:
                                    T.wait_flag("mte3", "mte2", 6)
                                else:
                                    T.wait_flag("mte3", "mte2", 4)
                        # tileA / weights were prefetched at the previous
                        # channel's end; only their flags need consuming
                        T.wait_flag("mte2", "v", 1)
                        if is_fp32:
                            T.tile.fill(acc, b_vec32[c_local])
                            T.tile.fill(comp, 0.0)
                        else:
                            T.tile.cast(w_row32, w_row, "CAST_NONE", K2_pad)
                            T.tile.fill(acc, b_vec32[c_local])

                        # software-pipelined kernel-column loop: the next
                        # window load is issued into the other tile buffer
                        # before the current one is consumed, so the ~1us GM
                        # latency overlaps with the tap compute
                        for kw in T.unroll(KW):
                            rho = (kw * DW) % m_mod
                            q0 = kw * DW - rho
                            if kw < KW - 1:
                                base_n = ow * sw + (((kw + 1) * DW) % m_mod)
                                # drain the previous kw's compute before its
                                # tile buffer is overwritten by the prefetch
                                T.pipe_barrier("v")
                                if kw % 2 == 0:
                                    T.copy(xp[c_abs, r0 : r0 + nr, base_n : base_n + win_w], tileB[0:nr, 0:win_w])
                                    T.set_flag("mte2", "v", 5)
                                else:
                                    T.copy(xp[c_abs, r0 : r0 + nr, base_n : base_n + win_w], tileA[0:nr, 0:win_w])
                                    T.set_flag("mte2", "v", 2)
                            # CAST_NONE is dead-code-eliminated for
                            # float->float, so fp32 imports via copy instead
                            if kw % 2 == 0:
                                T.wait_flag("mte2", "v", 2)
                                if lv == 0:
                                    if is_fp32:
                                        T.copy(tileA[0:nr, 0:win_w], gsrc[0:nr, 0:win_w])
                                    else:
                                        T.tile.cast(gsrc, tileA, "CAST_NONE", nr * win_w)
                                else:
                                    if is_fp32:
                                        T.copy(tileA[0:nr, 0:win_w], tile32[0:nr, 0:win_w])
                                    else:
                                        T.tile.cast(tile32, tileA, "CAST_NONE", nr * win_w)
                            else:
                                T.wait_flag("mte2", "v", 5)
                                if lv == 0:
                                    if is_fp32:
                                        T.copy(tileB[0:nr, 0:win_w], gsrc[0:nr, 0:win_w])
                                    else:
                                        T.tile.cast(gsrc, tileB, "CAST_NONE", nr * win_w)
                                else:
                                    if is_fp32:
                                        T.copy(tileB[0:nr, 0:win_w], tile32[0:nr, 0:win_w])
                                    else:
                                        T.tile.cast(tile32, tileB, "CAST_NONE", nr * win_w)
                            if lv == 2:
                                T.tile.gather_mask(gsrc1, tile32, "P0101")
                                T.tile.gather_mask(gsrc, gsrc1, "P0101")
                            if lv == 1:
                                T.tile.gather_mask(gsrc, tile32, "P0101")

                            if region and not is_fp32:
                                # unit stride: output rows 0..RB-1 for kernel
                                # row kh all read gsrc rows kh..kh+RB-1, so the
                                # whole [RB, BW] tile is one contiguous operand
                                # pair -> 1 op per kernel row instead of RB.
                                for kh in range(KH):
                                    T.tile.axpy(acc[0:RB, 0:BW], gsrc[kh : kh + RB, 0:BW], w_row32[kh * KW + kw])
                            elif region32:
                                # same whole-band reuse for the compensated
                                # chain: 11 statements per tap instead of
                                # 10 per (tap, row)
                                for kh in range(KH):
                                    T.tile.fill(wb2d, w_row32[kh * KW + kw])
                                    T.tile.mul(prod[0:RB, 0:BW], gsrc[kh : kh + RB, 0:BW], wb2d[0:RB, 0:BW])
                                    T.tile.add(tb2, acc, prod)
                                    T.tile.sub(zb2, tb2, acc)
                                    T.tile.sub(e21, tb2, zb2)
                                    T.tile.sub(e22, acc, e21)
                                    T.tile.sub(e21, prod, zb2)
                                    T.tile.add(e23, e22, e21)
                                    T.tile.add(cc2, comp, e23)
                                    T.copy(cc2[0:RB, 0:BW], comp[0:RB, 0:BW])
                                    T.copy(tb2[0:RB, 0:BW], acc[0:RB, 0:BW])
                            else:
                                for r in range(RB):
                                    for kh in range(KH):
                                        if is_fp32:
                                            T.tile.fill(w_b, w_row32[kh * KW + kw])
                                            T.tile.mul(prod1, gsrc[r * SH + kh * DH, q0 // (1 << lv) : q0 // (1 << lv) + BW], w_b)
                                            T.tile.add(tb1, acc[r, 0:BW], prod1)
                                            T.tile.sub(zb1, tb1, acc[r, 0:BW])
                                            T.tile.sub(ea1, tb1, zb1)
                                            T.tile.sub(ea2, acc[r, 0:BW], ea1)
                                            T.tile.sub(ea1, prod1, zb1)
                                            T.tile.add(ea3, ea2, ea1)
                                            T.tile.add(cn1, comp[r, 0:BW], ea3)
                                            T.copy(cn1, comp[r, 0:BW])
                                            T.copy(tb1, acc[r, 0:BW])
                                        elif pow2_sw:
                                            T.tile.axpy(
                                                acc[r, 0:BW],
                                                gsrc[r * SH + kh * DH, q0 // (1 << lv) : q0 // (1 << lv) + BW],
                                                w_row32[kh * KW + kw],
                                            )
                                        else:
                                            for w in T.serial(BW):
                                                acc[r, w] = (
                                                    acc[r, w] + gsrc[r * SH + kh * DH, q0 // (1 << lv) + w * SW] * w_row32[kh * KW + kw]
                                                )

                        # drain this channel's compute so the prefetch can
                        # reuse tileA / w_row.  For non-fp32 the prefetch is
                        # issued right here so its latency overlaps with the
                        # epilogue; the fp32 epilogue still carries per-row
                        # barrier_all (which reset armed event ids), so its
                        # prefetch must wait until after them.
                        T.pipe_barrier("v")
                        if not is_fp32:
                            if c_local < half_C - 1:
                                if ch0 + c_lo + c_local + 1 < C:
                                    T.copy(xp[nc0 + c_lo + c_local + 1, r0 : r0 + nr, (ow * sw) : (ow * sw) + win_w], tileA[0:nr, 0:win_w])
                                    T.set_flag("mte2", "v", 2)
                                    T.copy(weight2[ch0 + c_lo + c_local + 1, 0:K2], w_row[0:K2])
                                    T.set_flag("mte2", "v", 1)
                        if is_fp32:
                            # inf/NaN inputs poison the compensation term, so
                            # fall back to the plain sum where the result is
                            # not finite, preserving inf/NaN bit patterns.
                            for r in range(RB):
                                oh = oh0 + r
                                if oh < Ho:
                                    T.tile.add(cn1, acc[r, 0:BW], comp[r, 0:BW])
                                    T.copy(acc[r, 0:BW], ac1)
                                    T.tile.compare(cm1, cn1, cn1, "EQ")
                                    T.tile.select(rs1, cm1, cn1, ac1, "VSEL_CMPMASK_SPR")
                                    T.set_flag("v", "mte3", 3)
                                    T.wait_flag("v", "mte3", 3)
                                    T.copy(rs1[0:vw], y[c_abs, oh, ow : ow + vw])
                                    # rs1 is rewritten by the next row's
                                    # select: wait for the store to land
                                    T.barrier_all()
                        else:
                            if c_local % 2 == 0:
                                T.tile.cast(y_allA[0:RB, 0:BW], acc[0:RB, 0:BW], "CAST_RINT", RB * BW)
                            else:
                                T.tile.cast(y_allB[0:RB, 0:BW], acc[0:RB, 0:BW], "CAST_RINT", RB * BW)
                            T.set_flag("v", "mte3", 3)
                            T.wait_flag("v", "mte3", 3)
                            if Wo % BW == 0:
                                # every column tile is full-width: the store
                                # slice width equals the buffer pitch, which
                                # keeps the multi-row DMA source representable
                                if rows_v > 0:
                                    if c_local % 2 == 0:
                                        T.copy(y_allA[0:rows_v, 0:vw], y[c_abs, oh0 : oh0 + rows_v, ow : ow + vw])
                                    else:
                                        T.copy(y_allB[0:rows_v, 0:vw], y[c_abs, oh0 : oh0 + rows_v, ow : ow + vw])
                            else:
                                # partial-width tiles: a 2-D slice whose width
                                # is below the buffer pitch is not a legal
                                # multi-row DMA source (the slice view carries
                                # a sub-32B row pitch), so store row by row
                                for r in range(RB):
                                    oh = oh0 + r
                                    if oh < Ho:
                                        if c_local % 2 == 0:
                                            T.copy(y_allA[r, 0:vw], y[c_abs, oh, ow : ow + vw])
                                        else:
                                            T.copy(y_allB[r, 0:vw], y[c_abs, oh, ow : ow + vw])
                        if is_fp32:
                            if c_local < half_C - 1:
                                if ch0 + c_lo + c_local + 1 < C:
                                    T.copy(xp[nc0 + c_lo + c_local + 1, r0 : r0 + nr, (ow * sw) : (ow * sw) + win_w], tileA[0:nr, 0:win_w])
                                    T.set_flag("mte2", "v", 2)
                                    T.copy(weight2[ch0 + c_lo + c_local + 1, 0:K2], w_row32[0:K2])
                                    T.set_flag("mte2", "v", 1)
                        # release for the next channel's top-of-loop wait
                        # (adjacent set/wait; only when it will actually
                        # wait; fp32 rows run their own id-4 protocol)
                        if not is_fp32:
                            if c_local < half_C - 1:
                                if ch0 + c_lo + c_local + 1 < C:
                                    if c_local % 2 == 0:
                                        T.set_flag("mte3", "mte2", 4)
                                    else:
                                        T.set_flag("mte3", "mte2", 6)

        return main

    conv = _dwconv_conv(dtype=dtype_str)
    pad = _dwconv_pad(H, W, Hp, Wp, g["Wc"], NC, PH, PW, dtype=dtype_str)
    return pad, conv


def _get_kernels(N, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, Ho, Wo, dtype_str):
    key = (N, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, dtype_str)
    if key not in _kernel_cache:
        _kernel_cache[key] = _build_conv(N, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, Ho, Wo, dtype_str)
    return _kernel_cache[key]


def depthwise_conv_2d(x, weight, bias, kernelSize, stride, padding, dilation, groups):
    """Depthwise 2D convolution.

    y[n,c,h,w] = bias[c] + sum_{kh,kw} x[n,c,h*sh+kh*dh-ph, w*sw+kw*dw-pw]
                                 * weight[c,kh,kw]
    """
    if x.dim() != 4:
        raise ValueError(f"x must be 4D [N,C,H,W], got {tuple(x.shape)}")
    if weight.dim() != 3:
        raise ValueError(f"weight must be 3D [C,Kh,Kw], got {tuple(weight.shape)}")
    if bias.dim() != 1:
        raise ValueError(f"bias must be 1D [C], got {tuple(bias.shape)}")
    N, C, H, W = x.shape
    Kh, Kw = int(weight.shape[1]), int(weight.shape[2])
    if int(groups) != C:
        raise ValueError(f"depthwise requires groups == C ({C}), got {groups}")
    if [int(v) for v in kernelSize] != [Kh, Kw]:
        raise ValueError(f"kernelSize {kernelSize} != weight shape [{Kh},{Kw}]")
    if int(weight.shape[0]) != C or int(bias.shape[0]) != C:
        raise ValueError("weight/bias channel dim must equal C")
    if not (x.dtype == weight.dtype == bias.dtype):
        raise ValueError("x/weight/bias dtypes must be identical")
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise ValueError(f"unsupported dtype {x.dtype}")
    if not (x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()):
        raise ValueError("x/weight/bias must be contiguous")

    sh, sw = int(stride[0]), int(stride[1])
    ph, pw = int(padding[0]), int(padding[1])
    dh, dw = int(dilation[0]), int(dilation[1])
    Ho = (H + 2 * ph - dh * (Kh - 1) - 1) // sh + 1
    Wo = (W + 2 * pw - dw * (Kw - 1) - 1) // sw + 1
    if Ho < 1 or Wo < 1:
        raise ValueError(f"invalid output size Ho={Ho} Wo={Wo}")

    dtype_str = torch_dtype_to_tl(x.dtype)

    if (
        Kh == 1
        and Kw == 1
        and sh == 1
        and sw == 1
        and ph == 0
        and pw == 0
        and dh == 1
        and dw == 1
        and (H * W) % 16 == 0
        and C % _1X1_BR == 0
    ):
        key = ("1x1", N, C, H, W, dtype_str)
        if key not in _kernel_cache:
            _kernel_cache[key] = _dwconv_1x1(N * C, C, H * W, dtype=dtype_str)
        k1 = _kernel_cache[key]
        y2 = k1(x.reshape(N * C, H * W), weight.reshape(C), bias)
        return y2.reshape(N, C, Ho, Wo)

    pad_k, conv_k = _get_kernels(N, C, H, W, Kh, Kw, sh, sw, ph, pw, dh, dw, Ho, Wo, dtype_str)
    x3 = x.reshape(N * C, H, W)
    w2 = weight.reshape(C, Kh * Kw)
    xp = pad_k(x3)
    y3 = conv_k(xp, w2, bias)
    return y3.reshape(N, C, Ho, Wo)


if __name__ == "__main__":
    # Standalone smoke test: one representative shape per dispatch path
    # (pad+conv fp16, pad+conv fp32 dilated, 1x1 fast path bf16) against a
    # torch fp64 golden.
    import torch.nn.functional as F

    torch.manual_seed(0)
    bounds = {torch.float16: 5e-2, torch.float32: 5e-3, torch.bfloat16: 2e-1}
    for shape, dtype, k, s, p, d in (
        ((2, 64, 32, 32), torch.float16, (3, 3), (1, 1), (1, 1), (1, 1)),
        ((1, 16, 64, 64), torch.float32, (3, 3), (1, 1), (2, 2), (2, 2)),
        ((2, 64, 16, 16), torch.bfloat16, (1, 1), (1, 1), (0, 0), (1, 1)),
    ):
        x = (torch.rand(*shape) * 2 - 1).to(dtype)
        w = (torch.rand(shape[1], *k) * 0.6 - 0.3).to(dtype)
        b = (torch.rand(shape[1]) * 0.2 - 0.1).to(dtype)
        y = depthwise_conv_2d(x.npu(), w.npu(), b.npu(), list(k), list(s), list(p), list(d), shape[1]).cpu()
        ref = F.conv2d(x.double(), w.double().unsqueeze(1), b.double(), stride=s, padding=p, dilation=d, groups=shape[1]).to(dtype)
        diff = (y.float() - ref.float()).abs().max().item()
        print(f"[SMOKE] {tuple(shape)} {str(dtype).split('.')[-1]} k={k} max_abs={diff:.3e}")
        assert diff <= bounds[dtype], f"smoke test failed: {diff}"
    # CI pass criterion (examples/ascend_a2a3/bench_test.sh) requires the output to end
    # with this phrase (or "TEST PASSED!"), same as lstm/gru smoke tests.
    print("Kernel Output Match!")
