# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Standalone Conv2D operator for the CANN-Bench benchmark suite.

Entry point: conv_2d(x, filter, bias, strides, pads, dilations=None)
Layout: NCHW input, OIHW filter, returns NCHW output.
Supports float16, bfloat16, and float32 (fp16 hi/lo split).
"""
"""Standalone single-file Conv2D implementation (distilled from the 4-file
cann_bench package).  Public entry: conv_2d(x, filter, bias, strides, pads,
dilations=None).  Layout: NCHW input, OIHW filter, returns NCHW output.
fp16/bf16 (direct im2col GEMM) and fp32 (fp16 hi/lo split, fused 3-GEMM).
"""

import ctypes
import os

import torch
import tilelang
import tilelang.language as T

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

_GATHER_OFFSET_CACHE = {}

try:
    from torch_npu._C import _npu_getCurrentRawStream as _raw_stream
except ImportError:  # pragma: no cover - very old torch_npu
    _raw_stream = None


def _stream_of(dev_idx):
    """Raw current-stream handle for a device index (fast C path)."""
    if _raw_stream is not None:
        try:
            return _raw_stream(dev_idx)
        except Exception:
            globals()["_raw_stream"] = None  # permanent fallback
    return torch.npu.current_stream().npu_stream


_DISABLED = bool(os.environ.get("CONV2D_NO_FAST", ""))

# id(kernel) -> (kernel, bound_or_None).  JITKernel instances live for the
# process lifetime (tilelang's decorator caches them per args tuple), and
# the stored strong reference makes id() stability a guarantee rather than
# an assumption.
_BINDS = {}


def _bind(kernel):
    """Build a direct-dispatch callable for a compiled JITKernel.

    Returns None (-> stock path) for anything but fully-static,
    all-tensor-parameter kernels.
    """
    try:
        ad = kernel.adapter
    except Exception:
        return None
    try:
        if dict(ad.dynamic_symbolic_map):
            return None
        if list(ad.auto_gm_idx):
            return None
        lib_call = getattr(ad.lib, "call", None)
        if lib_call is None:
            return None
        params = ad.params
        n = len(params)
        result_idx = list(ad.result_idx)
        ws_idx = list(ad.workspace_idx)
        alloc = set(result_idx) | set(ws_idx)
        # every param must be a buffer (tensor): no scalar TIR args
        buf_idxs = {i for (i, _) in ad.buffer_dtype_map.values()}
        if buf_idxs != set(range(n)):
            return None
        shapes = []
        dtypes = []
        for p in params:
            dt = getattr(p, "dtype", None)
            if not isinstance(dt, torch.dtype):
                return None
            shp = []
            for d in p.shape:
                if isinstance(d, int):
                    shp.append(d)
                else:
                    v = getattr(d, "value", None)  # tir.IntImm
                    if v is None:
                        return None
                    shp.append(int(v))
            shapes.append(tuple(shp))
            dtypes.append(dt)
    except Exception:
        return None

    # sanity: raw stream getter must agree with the stock stream query
    # (checked once at bind; eval runs on the default stream throughout)
    if _raw_stream is not None:
        try:
            dev = torch.npu.current_device()
            if _raw_stream(dev) != torch.npu.current_stream().npu_stream:
                return None
        except Exception:
            return None

    in_order = [i for i in range(n) if i not in alloc]
    n_in = len(in_order)
    single_out = len(result_idx) == 1
    c_void_p = ctypes.c_void_p
    empty = torch.empty

    def bound(*inputs):
        if len(inputs) != n_in:
            raise TypeError("fast: expected %d inputs, got %d" % (n_in, len(inputs)))
        t0 = inputs[0]
        dev = t0.device
        tensors = [None] * n
        for i, inp in zip(in_order, inputs):
            tensors[i] = inp
        for i in alloc:  # fresh output/workspace each call (stock semantics)
            tensors[i] = empty(shapes[i], dtype=dtypes[i], device=dev)
        args = [c_void_p(t.data_ptr()) for t in tensors]
        args.append(c_void_p(_stream_of(t0.get_device())))
        lib_call(*args)
        if single_out:
            return tensors[result_idx[0]]
        if result_idx:
            return [tensors[i] for i in result_idx]
        return None

    return bound


def fast(kernel, *args):
    """Launch ``kernel`` bypassing the Cython wrapper.

    Drop-in replacement for ``kernel(*args)`` on fully-static kernels;
    transparently falls back to the stock path otherwise.
    """
    if _DISABLED:
        return kernel(*args)
    entry = _BINDS.get(id(kernel))
    if entry is None:
        bound = _bind(kernel)
        _BINDS[id(kernel)] = (kernel, bound)
        if bound is None:
            return kernel(*args)
        return bound(*args)
    bound = entry[1]
    if bound is None:
        return kernel(*args)
    return bound(*args)


def _ceil16(v):
    return (v + 15) // 16 * 16


@tilelang.jit(out_idx=[], pass_configs=PASS_CONFIGS)
def _pre_pad_kernel(N, Cin, Cin_pad, H, W, HP1, Wp, pt, in_dtype, RPC):
    """Whole-plane pad + cast (fp16/bf16): one 2D fill + one 2D image copy +
    (bf16: two whole-plane casts) + one 2D store per plane."""
    total = N * Cin_pad
    blocks = (total + RPC - 1) // RPC
    bf16 = in_dtype == "bfloat16"

    @T.prim_func
    def main(
            X: T.Tensor((N * Cin, H, W), in_dtype),
            Y: T.Tensor((N * Cin_pad * HP1, Wp), "float16"),
    ):
        with T.Kernel(blocks, is_npu=True) as (bid, vid):
            ub_in = T.alloc_ub((HP1, Wp), in_dtype)
            ub16 = T.alloc_ub((HP1, Wp), "float16")
            with T.Scope("V"):
                for rr in T.serial(RPC):
                    pid = bid * RPC + rr
                    if pid < total:
                        T.tile.fill(ub_in, 0.0)
                        n = pid // Cin_pad
                        c = pid % Cin_pad
                        if c < Cin:
                            T.copy(
                                X[n * Cin + c, 0:H, 0:Wp],
                                ub_in[pt + 1:pt + 1 + H, 0:Wp],
                                pad_value=0.0,
                            )
                        if bf16 == 1:
                            ub32 = T.alloc_ub((HP1, Wp), "float32")
                            T.copy(ub_in, ub32)
                            T.copy(ub32, ub16)
                        else:
                            T.copy(ub_in, ub16)
                        T.copy(ub16, Y[pid * HP1:pid * HP1 + HP1, 0:Wp])

    return main


@tilelang.jit(out_idx=[], pass_configs=PASS_CONFIGS)
def _pre_pad_chunk_kernel(N, Cin, Cin_pad, H, W, HP1, Wp, pt, in_dtype, RPC, BK):
    """Row-chunked pad + cast (fp16/bf16) for LARGE planes: the whole-plane
    _pre_pad_kernel allocates (HP1, Wp) UB buffers, which faults when
    HP1*Wp*elem exceeds the UB limit (256x256 fp16 = 282KB > ~196KB; H,W up
    to 256). Structure mirrors the production-proven _pre_pad_hilo_kernel:
    BK-row chunks with T.min bounds, explicit zero stores for the top
    [0, pt+1) and bottom [pt+1+H, HP1) pad rows (the whole-plane kernel's
    fill covers them in one shot instead)."""
    total = N * Cin_pad
    blocks = (total + RPC - 1) // RPC
    n_chunks = (H + BK - 1) // BK
    bot = HP1 - pt - 1 - H
    bf16 = in_dtype == "bfloat16"

    @T.prim_func
    def main(
            X: T.Tensor((N * Cin, H, W), in_dtype),
            Y: T.Tensor((N * Cin_pad * HP1, Wp), "float16"),
    ):
        with T.Kernel(blocks, is_npu=True) as (bid, vid):
            ub_in = T.alloc_ub((BK, Wp), in_dtype)
            ub16 = T.alloc_ub((BK, Wp), "float16")
            with T.Scope("V"):
                for rr in T.serial(RPC):
                    pid = bid * RPC + rr
                    if pid < total:
                        n = pid // Cin_pad
                        c = pid % Cin_pad
                        for ch in T.serial(n_chunks):
                            r0 = ch * BK
                            r1 = T.min(H, r0 + BK)
                            T.tile.fill(ub_in, 0.0)
                            if c < Cin:
                                T.copy(
                                    X[n * Cin + c, r0:r1, 0:Wp],
                                    ub_in[0:r1 - r0, 0:Wp],
                                    pad_value=0.0,
                                )
                            if bf16 == 1:
                                ub32 = T.alloc_ub((BK, Wp), "float32")
                                T.copy(ub_in, ub32)
                                T.copy(ub32, ub16)
                            else:
                                T.copy(ub_in, ub16)
                            T.copy(
                                ub16,
                                Y[pid * HP1 + pt + 1 + r0:pid * HP1 + pt + 1 + r1, 0:Wp],
                            )
                        T.tile.fill(ub16, 0.0)
                        T.copy(ub16[0:pt + 1, 0:Wp], Y[pid * HP1:pid * HP1 + pt + 1, 0:Wp])
                        if bot > 0:
                            T.copy(
                                ub16[0:bot, 0:Wp],
                                Y[pid * HP1 + pt + 1 + H:pid * HP1 + HP1, 0:Wp],
                            )

    return main


@tilelang.jit(out_idx=[], pass_configs=PASS_CONFIGS)
def _pre_pad_hilo_kernel(N, Cin, Cin_pad, H, W, HP1, Wp, pt, RPC, BK):
    """fp32 hi/lo split in ONE launch (dual 2D stores to distinct tensors).

    x is read from GM ONCE per chunk and both planes are derived in UB:
      hi = fp16(x)                    (NO clamp: inf must propagate)
      lo = fp16(clamp(x) - fp32(hi))  (clamped: inf/huge inputs yield lo=0)
    Halves the x GM read vs separate hi/lo kernels and saves one launch.
    """
    total = N * Cin_pad
    blocks = (total + RPC - 1) // RPC
    n_chunks = (H + BK - 1) // BK
    bot = HP1 - pt - 1 - H

    @T.prim_func
    def main(
            X: T.Tensor((N * Cin, H, W), "float32"),
            YH: T.Tensor((N * Cin_pad * HP1, Wp), "float16"),
            YL: T.Tensor((N * Cin_pad * HP1, Wp), "float16"),
    ):
        with T.Kernel(blocks, is_npu=True) as (bid, vid):
            ub32 = T.alloc_ub((BK, Wp), "float32")
            cl = T.alloc_ub((BK, Wp), "float32")
            hi16 = T.alloc_ub((BK, Wp), "float16")
            lo16 = T.alloc_ub((BK, Wp), "float16")
            hic16 = T.alloc_ub((BK, Wp), "float16")
            hic32 = T.alloc_ub((BK, Wp), "float32")
            with T.Scope("V"):
                for rr in T.serial(RPC):
                    pid = bid * RPC + rr
                    if pid < total:
                        n = pid // Cin_pad
                        c = pid % Cin_pad
                        for ch in T.serial(n_chunks):
                            r0 = ch * BK
                            r1 = T.min(H, r0 + BK)
                            T.tile.fill(ub32, 0.0)
                            if c < Cin:
                                T.copy(
                                    X[n * Cin + c, r0:r1, 0:Wp],
                                    ub32[0:r1 - r0, 0:Wp],
                                    pad_value=0.0,
                                )
                            # hi: unclamped cast (inf propagates)
                            T.copy(ub32, hi16)
                            # lo: clamp so inf/huge -> lo = 0 (no NaN).
                            # NOTE: the lo chain uses fp16(clamped) as its hi;
                            # using the unclamped hi16 would give 65504 - inf.
                            T.tile.max(cl, ub32, -65504.0)
                            T.tile.min(cl, cl, 65504.0)
                            T.copy(cl, hic16)
                            T.copy(hic16, hic32)
                            T.tile.sub(cl, cl, hic32)
                            T.copy(cl, lo16)
                            T.copy(
                                hi16,
                                YH[pid * HP1 + pt + 1 + r0:pid * HP1 + pt + 1 + r1, 0:Wp],
                            )
                            T.copy(
                                lo16,
                                YL[pid * HP1 + pt + 1 + r0:pid * HP1 + pt + 1 + r1, 0:Wp],
                            )
                        T.tile.fill(hi16, 0.0)
                        T.tile.fill(lo16, 0.0)
                        T.copy(hi16[0:pt + 1, 0:Wp], YH[pid * HP1:pid * HP1 + pt + 1, 0:Wp])
                        T.copy(lo16[0:pt + 1, 0:Wp], YL[pid * HP1:pid * HP1 + pt + 1, 0:Wp])
                        if bot > 0:
                            T.copy(
                                hi16[0:bot, 0:Wp],
                                YH[pid * HP1 + pt + 1 + H:pid * HP1 + HP1, 0:Wp],
                            )
                            T.copy(
                                lo16[0:bot, 0:Wp],
                                YL[pid * HP1 + pt + 1 + H:pid * HP1 + HP1, 0:Wp],
                            )

    return main


@tilelang.jit(out_idx=[1, 3], pass_configs=PASS_CONFIGS)
def _pre_weight_kernel(Cout, Cin, TAPS, Cin_pad, TAPS_pad, Kpad, in_dtype, need_bias_cast=0, COB=1):
    if TAPS == 1:
        # W passed as [Cout, Cin] (host reshape [Cout, Cin, 1] -> [Cout, Cin]).
        # COB rows per block: tiny single-row blocks pay ~110ns fixed cost
        # each; batch to ~128 blocks.
        cblocks = (Cout + COB - 1) // COB
        # Buffers are Cin_pad wide (16-multiple): a raw [0:Cin) copy with
        # Cin % 16 != 0 (e.g. 63 -> 126B) violates the UB 16-element row
        # alignment iron rule and corrupts the trailing segment. Read
        # the WIDE region [0:Cin_pad] with pad_value=0 instead -- the copy
        # clamps to the real source extent and zero-fills the pad channels.
        buf_w = Cin_pad

        @T.prim_func
        def main1(
                W: T.Tensor((Cout, Cin), in_dtype),
                WT: T.Tensor((Cout, Kpad), "float16"),
                Bias: T.Tensor((Cout,), in_dtype),
                BIAS32: T.Tensor((Cout,), "float"),
        ):
            with T.Kernel(cblocks, is_npu=True) as (cid, vid):
                ub_in = T.alloc_ub((buf_w,), in_dtype)
                ub32 = T.alloc_ub((buf_w,), "float32")
                ub16 = T.alloc_ub((buf_w,), "float16")
                wt = T.alloc_ub((Kpad,), "float16")
                b_ub = T.alloc_ub((1,), in_dtype)
                b32 = T.alloc_ub((1,), "float")
                with T.Scope("V"):
                    for ci in T.serial(COB):
                        co = cid * COB + ci
                        if co < Cout:
                            if need_bias_cast:
                                T.copy(Bias[co], b_ub)
                                T.copy(b_ub, b32)
                                T.copy(b32, BIAS32[co])
                            T.tile.fill(wt, 0.0)
                            T.copy(W[co, 0:Cin_pad], ub_in[0:Cin_pad], pad_value=0.0)
                            if in_dtype == "bfloat16":
                                T.copy(ub_in, ub32)
                                T.copy(ub32, ub16)
                            else:
                                T.copy(ub_in, ub16)
                            T.copy(ub16[0:Cin_pad], wt[0:Cin_pad])
                            T.copy(wt, WT[co, 0:Kpad])

        return main1

    # chunked variant for large Cin (UB budget): R-row bands over Cin.
    # UB budget: worst case (bf16) a_in(2B)+a32(4B)+a16(2B)+t_ub(2B)=10B/elem
    # plus the wt row (Kpad*2B, allocated for the whole launch).  Keep the
    # total under ~180KB (memory planner needs headroom).
    per_row = TAPS_pad * 10
    budget = 180 * 1024 - Kpad * 2
    R = min(2048, Cin_pad, max(16, budget // per_row))
    if R % 16 != 0:
        R = max(16, R // 16 * 16)
    # Chunk divisibility: R must DIVIDE Cin_pad. The kernel bodies do
    # constant-R chunk copies (literal-only bounds -- the P-series rule;
    # a runtime T.min length lands in the copy template parameter and
    # fails compilation). A non-dividing R (e.g. Cin_pad 2048 with R 912
    # -> chunks 3, last chunk c0+R overruns) reads W out of bounds and
    # corrupts the tap-major layout. Step R down to a 16-multiple factor.
    while R > 16 and Cin_pad % R != 0:
        R -= 16
    chunks = Cin_pad // R

    @T.prim_func
    def main(
            W: T.Tensor((Cout, Cin, TAPS), in_dtype),
            WT: T.Tensor((Cout, Kpad), "float16"),
            Bias: T.Tensor((Cout,), in_dtype),
            BIAS32: T.Tensor((Cout,), "float"),
    ):
        with T.Kernel(Cout, is_npu=True) as (cid, vid):
            a_in = T.alloc_ub((R, TAPS_pad), in_dtype)
            a16 = T.alloc_ub((R, TAPS_pad), "float16")
            t_ub = T.alloc_ub((TAPS_pad, R), "float16")
            wt = T.alloc_ub((Kpad,), "float16")
            b_ub = T.alloc_ub((1,), in_dtype)
            b32 = T.alloc_ub((1,), "float")
            with T.Scope("V"):
                if need_bias_cast:
                    T.copy(Bias[cid], b_ub)
                    T.copy(b_ub, b32)
                    T.copy(b32, BIAS32[cid])
                T.tile.fill(wt, 0.0)
                for ck in T.serial(chunks):
                    c0 = ck * R
                    T.tile.fill(a_in, 0.0)
                    T.copy(W[cid, c0:c0 + R, 0:TAPS], a_in[0:R, 0:TAPS], pad_value=0.0)
                    if in_dtype == "bfloat16":
                        a32 = T.alloc_ub((R, TAPS_pad), "float32")
                        T.copy(a_in, a32)
                        T.copy(a32, a16)
                    else:
                        T.copy(a_in, a16)
                    T.tile.transpose(t_ub, a16)
                    for tap in T.serial(TAPS):
                        T.copy(
                            t_ub[tap, 0:R],
                            wt[tap * Cin_pad + c0:tap * Cin_pad + c0 + R],
                        )
                T.copy(wt, WT[cid, 0:Kpad])

    return main


@tilelang.jit(out_idx=[], pass_configs=PASS_CONFIGS)
def _pre_pad_weight_kernel(
    N,
    Cin,
    Cin_pad,
    H,
    W,
    HP1,
    Wp,
    pt,
    in_dtype,
    RPC,
    Cout,
    TAPS,
    TAPS_pad,
    Kpad,
    R,
    chunks,
):
    bf16 = in_dtype == "bfloat16"
    total_planes = N * Cin_pad
    grid_pad = (total_planes + RPC - 1) // RPC
    total = grid_pad + Cout

    @T.prim_func
    def main(
            X: T.Tensor((N * Cin, H, W), in_dtype),
            W3: T.Tensor((Cout, Cin, TAPS), in_dtype),
            Bias: T.Tensor((Cout,), in_dtype),
            Y: T.Tensor((N * Cin_pad * HP1, Wp), "float16"),
            WT: T.Tensor((Cout, Kpad), "float16"),
            BIAS32: T.Tensor((Cout,), "float"),
    ):
        with T.Kernel(total, is_npu=True) as (bid, vid):
            ub_in = T.alloc_ub((HP1, Wp), in_dtype)
            ub16 = T.alloc_ub((HP1, Wp), "float16")
            a_in = T.alloc_ub((R, TAPS_pad), in_dtype)
            a16 = T.alloc_ub((R, TAPS_pad), "float16")
            t_ub = T.alloc_ub((TAPS_pad, R), "float16")
            wt = T.alloc_ub((Kpad,), "float16")
            b_ub = T.alloc_ub((1,), in_dtype)
            b32 = T.alloc_ub((1,), "float")
            if bid < grid_pad:
                # ---- pad body (verbatim _pre_pad_kernel) ----
                with T.Scope("V"):
                    for rr in T.serial(RPC):
                        pid = bid * RPC + rr
                        if pid < total_planes:
                            T.tile.fill(ub_in, 0.0)
                            n = pid // Cin_pad
                            c = pid % Cin_pad
                            if c < Cin:
                                T.copy(
                                    X[n * Cin + c, 0:H, 0:Wp],
                                    ub_in[pt + 1:pt + 1 + H, 0:Wp],
                                    pad_value=0.0,
                                )
                            if bf16 == 1:
                                ub32 = T.alloc_ub((HP1, Wp), "float32")
                                T.copy(ub_in, ub32)
                                T.copy(ub32, ub16)
                            else:
                                T.copy(ub_in, ub16)
                            T.copy(ub16, Y[pid * HP1:pid * HP1 + HP1, 0:Wp])
            else:
                # ---- weight body (verbatim _pre_weight_kernel) ----
                cid = bid - grid_pad
                with T.Scope("V"):
                    T.copy(Bias[cid], b_ub)
                    T.copy(b_ub, b32)
                    T.copy(b32, BIAS32[cid])
                    T.tile.fill(wt, 0.0)
                    for ck in T.serial(chunks):
                        c0 = ck * R
                        T.tile.fill(a_in, 0.0)
                        T.copy(W3[cid, c0:c0 + R, 0:TAPS], a_in[0:R, 0:TAPS], pad_value=0.0)
                        if bf16 == 1:
                            a32 = T.alloc_ub((R, TAPS_pad), "float32")
                            T.copy(a_in, a32)
                            T.copy(a32, a16)
                        else:
                            T.copy(a_in, a16)
                        T.tile.transpose(t_ub, a16)
                        for tap in T.serial(TAPS):
                            T.copy(
                                t_ub[tap, 0:R],
                                wt[tap * Cin_pad + c0:tap * Cin_pad + c0 + R],
                            )
                    T.copy(wt, WT[cid, 0:Kpad])

    return main


@tilelang.jit(out_idx=[1, 2], pass_configs=PASS_CONFIGS)
def _pre_weight_split_kernel(Cout, Cin, TAPS, Cin_pad, TAPS_pad, Kpad, R, chunks):
    """fp32 weight -> tap-major fp16 hi/lo in ONE launch (halves the W loads).

    Cin is processed in R-row chunks (R a 16-multiple for the transpose):
    per chunk, a WIDE source region load W[cid, c0:c0+R, 0:TAPS_pad] clamps
    to the real [Cin, TAPS] extent and zero-pads the rest (no tile fill),
    then hi = fp16(w), lo = fp16(w - fp32(hi)) -- hi recomputed locally --
    both transposed (hardware TransDataTo5HD) and scattered into the wt /
    wt_lo rows at [tap*Cin_pad + c0, +R).  UB working set ~ (R*TAPS_pad)*16B
    + 2*Kpad*2B (64KB-class at R=256 vs 242KB for the un-chunked pair).
    Two 1D row stores per launch is verified-safe.
    """

    @T.prim_func
    def main(
            W: T.Tensor((Cout, Cin, TAPS), "float32"),
            WT: T.Tensor((Cout, Kpad), "float16"),
            WTLO: T.Tensor((Cout, Kpad), "float16"),
    ):
        with T.Kernel(Cout, is_npu=True) as (cid, vid):
            w32 = T.alloc_ub((R, TAPS_pad), "float32")
            hi16 = T.alloc_ub((R, TAPS_pad), "float16")
            hi32 = T.alloc_ub((R, TAPS_pad), "float32")
            lo16 = T.alloc_ub((R, TAPS_pad), "float16")
            t_hi = T.alloc_ub((TAPS_pad, R), "float16")
            t_lo = T.alloc_ub((TAPS_pad, R), "float16")
            wt = T.alloc_ub((Kpad,), "float16")
            wt_lo = T.alloc_ub((Kpad,), "float16")
            with T.Scope("V"):
                T.tile.fill(wt, 0.0)
                T.tile.fill(wt_lo, 0.0)
                for ck in T.serial(chunks):
                    c0 = ck * R
                    T.copy(
                        W[cid, c0:c0 + R, 0:TAPS_pad],
                        w32[0:R, 0:TAPS_pad],
                        pad_value=0.0,
                    )
                    T.copy(w32, hi16)
                    T.copy(hi16, hi32)
                    T.tile.sub(w32, w32, hi32)
                    T.copy(w32, lo16)
                    T.tile.transpose(t_hi, hi16)
                    T.tile.transpose(t_lo, lo16)
                    for tap in T.serial(TAPS):
                        T.copy(
                            t_hi[tap, 0:R],
                            wt[tap * Cin_pad + c0:tap * Cin_pad + c0 + R],
                        )
                        T.copy(
                            t_lo[tap, 0:R],
                            wt_lo[tap * Cin_pad + c0:tap * Cin_pad + c0 + R],
                        )
                T.copy(wt, WT[cid, 0:Kpad])
                T.copy(wt_lo, WTLO[cid, 0:Kpad])

    return main


@tilelang.jit(out_idx=[1, 2], pass_configs=PASS_CONFIGS)
def _pre_weight_split_cob_kernel(Cout, Cin, TAPS, Cin_pad, TAPS_pad, Kpad, R, chunks, COB):
    """COB-batched weight split: COB cos per block (same body as
    _pre_weight_split_kernel; cid = bid*COB + ci is multiply-add only, no
    div/mod on bid -- the MIX_AIC-safe pattern from _pre_weight_kernel
    main1).
    """
    cblocks = (Cout + COB - 1) // COB

    @T.prim_func
    def main(
            W: T.Tensor((Cout, Cin, TAPS), "float32"),
            WT: T.Tensor((Cout, Kpad), "float16"),
            WTLO: T.Tensor((Cout, Kpad), "float16"),
    ):
        with T.Kernel(cblocks, is_npu=True) as (bid, vid):
            w32 = T.alloc_ub((R, TAPS_pad), "float32")
            hi16 = T.alloc_ub((R, TAPS_pad), "float16")
            hi32 = T.alloc_ub((R, TAPS_pad), "float32")
            lo16 = T.alloc_ub((R, TAPS_pad), "float16")
            t_hi = T.alloc_ub((TAPS_pad, R), "float16")
            t_lo = T.alloc_ub((TAPS_pad, R), "float16")
            wt = T.alloc_ub((Kpad,), "float16")
            wt_lo = T.alloc_ub((Kpad,), "float16")
            with T.Scope("V"):
                for ci in T.serial(COB):
                    cid = bid * COB + ci
                    if cid < Cout:
                        T.tile.fill(wt, 0.0)
                        T.tile.fill(wt_lo, 0.0)
                        for ck in T.serial(chunks):
                            c0 = ck * R
                            T.copy(
                                W[cid, c0:c0 + R, 0:TAPS_pad],
                                w32[0:R, 0:TAPS_pad],
                                pad_value=0.0,
                            )
                            T.copy(w32, hi16)
                            T.copy(hi16, hi32)
                            T.tile.sub(w32, w32, hi32)
                            T.copy(w32, lo16)
                            T.tile.transpose(t_hi, hi16)
                            T.tile.transpose(t_lo, lo16)
                            for tap in T.serial(TAPS):
                                T.copy(
                                    t_hi[tap, 0:R],
                                    wt[tap * Cin_pad + c0:tap * Cin_pad + c0 + R],
                                )
                                T.copy(
                                    t_lo[tap, 0:R],
                                    wt_lo[tap * Cin_pad + c0:tap * Cin_pad + c0 + R],
                                )
                        T.copy(wt, WT[cid, 0:Kpad])
                        T.copy(wt_lo, WTLO[cid, 0:Kpad])

    return main


def pad_input_weight(
    x3d,
    w3d,
    bias,
    N,
    Cin,
    Cin_pad,
    H,
    W,
    HP1,
    Wp,
    pt,
    in_dtype,
    Cout,
    TAPS,
    Kpad,
):
    """Fused pad + weight prep (one launch). Returns (Y, WT, BIAS32) or None
    when the combined UB budget exceeds the safe envelope (caller falls back
    to the separate kernels)."""
    if TAPS == 1:
        # the fused weight body is the chunked TAPS>1 variant (16x inflated
        # buffers for a 1-tap weight); the separate prep_weight handles
        # TAPS==1 with a lean cast-only body
        return None
    total = N * Cin_pad
    rpc_grid = max(1, (total + 32767) // 32768)  # grid <= 2^15
    per = HP1 * Wp * (8 if in_dtype == "bfloat16" else 4)
    prpc = max(rpc_grid, min(4, (160 * 1024) // max(per, 1)))
    TAPS_pad = _ceil16(max(TAPS, 2))
    per_row = TAPS_pad * 10
    budget = 180 * 1024 - Kpad * 2
    R = min(2048, Cin_pad, max(16, budget // per_row))
    if R % 16 != 0:
        R = max(16, R // 16 * 16)
    # Chunk divisibility: R must DIVIDE Cin_pad. The kernel bodies do
    # constant-R chunk copies (literal-only bounds -- the P-series rule;
    # a runtime T.min length lands in the copy template parameter and
    # fails compilation). A non-dividing R (e.g. Cin_pad 2048 with R 912
    # -> chunks 3, last chunk c0+R overruns) reads W out of bounds and
    # corrupts the tap-major layout. Step R down to a 16-multiple factor.
    while R > 16 and Cin_pad % R != 0:
        R -= 16
    chunks = Cin_pad // R
    # UB envelope: pad buffers (in + 16 + optional 32) + weight buffers
    pad_ub = HP1 * Wp * (8 if in_dtype == "bfloat16" else 4)
    wt_ub = R * TAPS_pad * 6 + Kpad * 2
    if pad_ub + wt_ub > 150 * 1024:
        return None
    Y = torch.empty((N * Cin_pad * HP1, Wp), dtype=torch.float16, device=x3d.device)
    WT = torch.empty((Cout, Kpad), dtype=torch.float16, device=x3d.device)
    BIAS32 = torch.empty((Cout,), dtype=torch.float32, device=x3d.device)
    fast(
        _pre_pad_weight_kernel(
            N,
            Cin,
            Cin_pad,
            H,
            W,
            HP1,
            Wp,
            pt,
            in_dtype,
            prpc,
            Cout,
            TAPS,
            TAPS_pad,
            Kpad,
            R,
            chunks,
        ),
        x3d,
        w3d,
        bias.contiguous(),
        Y,
        WT,
        BIAS32,
    )
    return Y, WT, BIAS32


def pad_input(x3d, N, Cin, Cin_pad, H, W, HP1, Wp, pt, in_dtype, split):
    """x3d: [N*Cin, H, W] contiguous view. Returns (Y,) or (Y_hi, Y_lo),
    each [N*Cin_pad*HP1, Wp] fp16 (circular-shift plane, whole-plane design).
    The whole-plane kernels fill zero rows themselves, so no extra launch."""
    total = N * Cin_pad
    rpc_grid = max(1, (total + 32767) // 32768)  # grid <= 2^15
    if split == 0:
        Y = torch.empty((N * Cin_pad * HP1, Wp), dtype=torch.float16, device=x3d.device)
        per = HP1 * Wp * (8 if in_dtype == "bfloat16" else 4)
        if per > 150 * 1024:
            # Large plane (e.g. 256x256 -> 282KB): the whole-plane kernel's
            # (HP1, Wp) UB buffers would fault. Route to the row-chunked
            # variant (H,W up to 256).
            BK = 16
            per_chunk = BK * Wp * (8 if in_dtype == "bfloat16" else 4)
            prpc = max(rpc_grid, min(8, (160 * 1024) // max(per_chunk, 1)))
            fast(
                _pre_pad_chunk_kernel(N, Cin, Cin_pad, H, W, HP1, Wp, pt, in_dtype, prpc, BK),
                x3d,
                Y,
            )
            return Y, None
        prpc = max(rpc_grid, min(4, (160 * 1024) // max(per, 1)))
        fast(
            _pre_pad_kernel(N, Cin, Cin_pad, H, W, HP1, Wp, pt, in_dtype, prpc),
            x3d,
            Y,
        )
        return Y, None
    BK = 16
    per_chunk = BK * Wp * 16
    prpc = max(rpc_grid, min(8, (160 * 1024) // max(per_chunk, 1)))
    Yh = torch.empty((N * Cin_pad * HP1, Wp), dtype=torch.float16, device=x3d.device)
    Yl = torch.empty((N * Cin_pad * HP1, Wp), dtype=torch.float16, device=x3d.device)
    # hi/lo in ONE launch: x read once (halves GM read), dual 2D stores
    fast(
        _pre_pad_hilo_kernel(N, Cin, Cin_pad, H, W, HP1, Wp, pt, prpc, BK),
        x3d,
        Yh,
        Yl,
    )
    return Yh, Yl


def prep_weight(w3d, bias, Cout, Cin, TAPS, Cin_pad, Kpad, in_dtype, split):
    """w3d: [Cout, Cin, TAPS] contiguous view; bias [Cout].
    Returns (WT, WTLO|None, BIAS32).  Out tensors are allocated by the jit
    wrapper (out_idx) and returned by the call -- only inputs are passed."""
    if split == 0:
        if TAPS == 1:
            w_arg = w3d.reshape(Cout, Cin)
            cob = max(1, min(64, (Cout + 127) // 128))
        else:
            w_arg = w3d
            cob = 1
        WT, BIAS32 = fast(
            _pre_weight_kernel(
                Cout,
                Cin,
                TAPS,
                Cin_pad,
                _ceil16(max(TAPS, 2)),
                Kpad,
                in_dtype,
                1,
                cob,
            ),
            w_arg,
            bias,
        )
        return WT, None, BIAS32
    TAPS_pad = _ceil16(max(TAPS, 2))
    # 16-multiple chunk for the hardware transpose; R=512 (when TAPS_pad is
    # small) halves the per-core chunk loop.
    R = min(512, (160 * 1024) // (TAPS_pad * 16), Cin_pad)
    R = max(16, R // 16 * 16)
    # Chunk divisibility: R must DIVIDE Cin_pad. The kernel bodies do
    # constant-R chunk copies (literal-only bounds -- the P-series rule;
    # a runtime T.min length lands in the copy template parameter and
    # fails compilation). A non-dividing R (e.g. Cin_pad 2048 with R 912
    # -> chunks 3, last chunk c0+R overruns) reads W out of bounds and
    # corrupts the tap-major layout. Step R down to a 16-multiple factor.
    while R > 16 and Cin_pad % R != 0:
        R -= 16
    chunks = Cin_pad // R
    # COB batching of the weight split (heuristic):
    #   Cout >= 256  -> cob=4
    #   Cout >= 64, Kpad <= 2048 -> cob=2
    #   else cob=1 (large-Kpad shapes regress at any cob; small Cout neutral)
    if Cout >= 256:
        cob = 4
    elif Cout >= 64 and Kpad <= 2048:
        cob = 2
    else:
        cob = 1
    if cob > 1:
        WT, WTLO = fast(
            _pre_weight_split_cob_kernel(Cout, Cin, TAPS, Cin_pad, TAPS_pad, Kpad, R, chunks, cob),
            w3d,
        )
    else:
        WT, WTLO = fast(
            _pre_weight_split_kernel(Cout, Cin, TAPS, Cin_pad, TAPS_pad, Kpad, R, chunks),
            w3d,
        )
    BIAS32 = bias.contiguous()
    return WT, WTLO, BIAS32


@tilelang.jit(out_idx=[-1], workspace_idx=[3], pass_configs=PASS_CONFIGS)
def _conv2d_1x1_kernel(
    out_channels: int,
    in_channels: int,
    flat_n: int,
    bias: bool,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float16",
    out_dtype: str = "float16",
):
    """Pure GEMM for 1x1 conv: y[co, m] = sum_ci W[co, ci] * X[ci, m] + b[co].

    m = N*H*W (flat output spatial). Cout/Cin/flat_n are host-padded to
    GEMM-friendly multiples; only the leading region carries real data.
    """
    accum_dtype = "float"
    m, k, n = out_channels, in_channels, flat_n
    m_blocks = (m + BLOCK_M - 1) // BLOCK_M
    n_blocks = (n + BLOCK_N - 1) // BLOCK_N
    k_blocks = (k + BLOCK_K - 1) // BLOCK_K

    @T.prim_func
    def main(
            Weight: T.Tensor((m, k), dtype),
            Input: T.Tensor((k, n), dtype),
            Bias: T.Tensor((m,), accum_dtype),
            epilog_workspace: T.Tensor((m_blocks * n_blocks, BLOCK_M, BLOCK_N), accum_dtype),
            Output: T.Tensor((m, n), out_dtype),
    ):
        with T.Kernel(m_blocks * n_blocks, is_npu=True) as (cid, _):
            bm = cid // n_blocks
            bn = cid % n_blocks
            m_start = bm * BLOCK_M
            n_start = bn * BLOCK_N
            valid_m = T.min(BLOCK_M, m - m_start)
            valid_n = T.min(BLOCK_N, n - n_start)

            w_l1 = T.alloc_L1((BLOCK_M, BLOCK_K), dtype)
            x_l1 = T.alloc_L1((BLOCK_K, BLOCK_N), dtype)
            c_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            o_ub = T.alloc_shared((BLOCK_M, BLOCK_N), accum_dtype)
            o_ub16 = T.alloc_shared((BLOCK_M, BLOCK_N), out_dtype)

            for kb in T.serial(k_blocks):
                k_start = kb * BLOCK_K
                valid_k = T.min(BLOCK_K, k - k_start)
                T.copy(
                    Weight[m_start:m_start + valid_m, k_start:k_start + valid_k],
                    w_l1[0:valid_m, 0:valid_k],
                )
                T.copy(
                    Input[k_start:k_start + valid_k, n_start:n_start + valid_n],
                    x_l1[0:valid_k, 0:valid_n],
                    pad_value=0.0,
                )
                T.gemm_v0(w_l1, x_l1, c_frag, init=(kb == 0))

            T.copy(c_frag, epilog_workspace[cid, :, :])
            T.copy(epilog_workspace[cid, :, :], o_ub)
            if bias:
                for mm in T.serial(BLOCK_M):
                    if mm < valid_m:
                        T.tile.add(o_ub[mm, 0:valid_n], o_ub[mm, 0:valid_n], Bias[m_start + mm])
            if out_dtype == "float":
                T.copy(o_ub[0:valid_m, 0:valid_n], Output[m_start:m_start + valid_m,
                                                          n_start:n_start + valid_n])
            else:
                T.copy(o_ub, o_ub16)
                T.copy(o_ub16[0:valid_m, 0:valid_n], Output[m_start:m_start + valid_m,
                                                            n_start:n_start + valid_n])

    return main


@tilelang.jit(out_idx=[-1], workspace_idx=[3], pass_configs=PASS_CONFIGS)
def _conv2d_direct_im2col_kernel(
    in_batch: int,
    in_channels: int,
    in_height: int,
    in_width: int,
    out_channels: int,
    k_h: int,
    k_w: int,
    stride: int,
    pt: int,
    pl: int,
    dh: int,
    dw: int,
    h_grid: int,
    w_grid: int,
    h_plane: int,
    bias: bool,
    win_pad: int,
    n_pad: int,
    k_pad: int,
    needs_x_zero: int,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
    compact_rows: bool = False,
    dtype: str = "float16",
    out_dtype: str = "float16",
    bias_dtype: str = "float",
):
    """Vectorized direct-im2col + GEMM (stride 1 or 2, host-padded shapes).

    Input is host-padded so every output run falls inside one output row and
    every GM->L1 copy stays aligned; one DMA per (tap, run) into x_l1.
    stride2=1: the OUTPUT grid is (h1+1)//2 x (w1+1)//2 over the same padded
    plane; tap reads sample the input plane at 2*oh / 2*ow offsets, so the
    GEMM computes only the kept outputs (4x less work than the dense grid).
    """
    accum_dtype = "float"
    batch, c_in = in_batch, in_channels
    s = stride
    c_out = out_channels
    # DENSE padded grid, both dims HOST-DRIVEN (the kernel must not
    # recompute them: asymmetric pads make pt != pl/pr and dh != dw, so any
    # internal formula diverges from the host allocation -- st4 probe).
    h_out = h_grid
    w_out_k = w_grid
    m, k, n = c_out, k_pad, batch * h_out * w_out_k
    assert w_out_k % 16 == 0
    m_blocks = (m + BLOCK_M - 1) // BLOCK_M
    n_blocks = (n + BLOCK_N - 1) // BLOCK_N
    k_blocks = k // BLOCK_K
    taps = k_h * k_w
    # True K before BLOCK_K padding.  The tap copies below cover x_l1 rows
    # [0, k_real) only; rows [k_real, k_pad) of the LAST kb tile are never
    # written and MUST be zeroed explicitly -- L1 persists across launches,
    # so a previous huge/inf-valued launch leaves inf there and the
    # accumulator folds 0 * inf = NaN (huge->rect repro, diag_nan_leak).
    # tail0 is a python literal per the P-series miscompile rules.
    k_real = c_in * taps
    tail0 = k_real - (k_blocks - 1) * BLOCK_K
    total = m_blocks * n_blocks
    assert k % BLOCK_K == 0, "direct im2col requires K padded to a BLOCK_K multiple"
    # Row-aligned im2col runs: when w_out divides BLOCK_N (or is a multiple of
    # it), each run is one full output row, so the block is filled with the
    # fewest, widest GM->L1 DMAs (e.g. w_out=128 -> one 128-wide DMA per tap).
    if w_out_k % BLOCK_N == 0:
        block_w = BLOCK_N
    elif BLOCK_N % w_out_k == 0 and w_out_k % 16 == 0:
        block_w = w_out_k
    else:
        block_w = 32 if w_out_k % 32 == 0 else 16
    runs_per_block = BLOCK_N // block_w
    # Epilogue segments: keeps o_ub/o_ub16 128-wide even when BLOCK_N=256
    # (halves per-kb weight re-reads without blowing the UB budget).
    EPI_N = 128
    n_seg = max(1, BLOCK_N // EPI_N)

    # The per-kb x_l1 full-tile zero is only needed when a kb leaves rows
    # unwritten -- i.e. a K tail (real K not a BLOCK_K multiple). Otherwise the
    # taps partition [0, K) so every kb writes its full BLOCK_K rows.

    @T.prim_func
    def main(
            Weight: T.Tensor((c_out, k), dtype),
            Input: T.Tensor((batch * c_in, h_plane * win_pad), dtype),
            Bias: T.Tensor((c_out,), bias_dtype),
            epilog_workspace: T.Tensor((total, BLOCK_M, BLOCK_N), accum_dtype),
            Output: T.Tensor((m, n_pad), out_dtype),
    ):
        with T.Kernel(total, is_npu=True) as (cid, _):
            bm = cid // n_blocks
            bn = cid % n_blocks
            m_start = bm * BLOCK_M
            n_start = bn * BLOCK_N
            valid_m = T.min(BLOCK_M, m - m_start)

            w_l1 = T.alloc_L1((BLOCK_M, BLOCK_K), dtype)
            x_l1 = T.alloc_L1((BLOCK_K, BLOCK_N), dtype)
            c_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            o_ub = T.alloc_shared((BLOCK_M, EPI_N), accum_dtype)
            o_ub16 = T.alloc_shared((BLOCK_M, EPI_N), out_dtype)

            for kb in T.serial(k_blocks):
                k_start = kb * BLOCK_K
                valid_k = T.min(BLOCK_K, k - k_start)
                T.copy(
                    Weight[m_start:m_start + valid_m, k_start:k_start + valid_k],
                    w_l1[0:valid_m, 0:valid_k],
                )
                # K-tail zero: rows [tail0, BLOCK_K) of the LAST kb tile are
                # never written by the tap copies below.  L1 persists across
                # launches, so stale inf/NaN from a previous launch folds
                # 0*inf = NaN into the accumulator (huge->rect repro).
                # NOTE: the zero MUST target the tile base (dst offset 0) --
                # copy_gm_to_l1 only zero-inits (need_clear) at offset 0; a
                # partial-slice copy silently degrades to a 1x1 copy and the
                # tail rows keep stale L1 garbage.  The tap copies below
                # overwrite rows [0, tail0) afterwards.
                if needs_x_zero:  # noqa: SIM102
                    if kb == k_blocks - 1 and tail0 < BLOCK_K:
                        T.copy(Input[0:1, 0:1], x_l1[0:BLOCK_K, 0:BLOCK_N], pad_value=0.0)
                for tap in T.serial(taps):
                    kh = tap // k_w
                    kw = tap % k_w
                    c0 = T.max(k_start - tap * c_in, 0)
                    c1 = T.min(c_in, k_start + BLOCK_K - tap * c_in)
                    if c0 < c1:
                        for run in T.serial(runs_per_block):
                            base = n_start + run * block_w
                            oh = base % (h_out * w_out_k) // w_out_k
                            batch_idx = base // (h_out * w_out_k)
                            ih_p = (oh * s + kh * dh if compact_rows else oh + kh * dh)
                            iw_p0 = base % w_out_k + kw * dw
                            if batch_idx < batch and oh < h_out and base + block_w <= n and base % w_out_k + block_w <= w_out_k and (
                                    compact_rows or oh % s == 0):
                                T.copy(
                                    Input[
                                        batch_idx * c_in + c0:batch_idx * c_in + c1,
                                        ih_p * win_pad + iw_p0 + win_pad - pl:ih_p * win_pad +
                                        iw_p0 + win_pad - pl + block_w,
                                    ],
                                    x_l1[
                                        tap * c_in + c0 - k_start:tap * c_in + c1 - k_start,
                                        run * block_w:run * block_w + block_w,
                                    ],
                                )
                T.gemm_v0(w_l1, x_l1, c_frag, init=(kb == 0))

            T.copy(c_frag, epilog_workspace[cid, :, :])
            for seg in T.serial(n_seg):
                T.copy(
                    epilog_workspace[cid, :, seg * EPI_N:(seg + 1) * EPI_N],
                    o_ub,
                )
                if bias:
                    for mm in T.serial(BLOCK_M):
                        if mm < valid_m:
                            T.tile.add(
                                o_ub[mm, 0:EPI_N],
                                o_ub[mm, 0:EPI_N],
                                Bias[m_start + mm],
                            )
                if out_dtype == "float":
                    # Full-width segment write: a partial-width UB->GM copy
                    # mis-steps the source row stride and corrupts rows >= 1.
                    # n_pad is a BLOCK_N multiple, so writing the whole
                    # segment is in bounds; the host slices back to the real n.
                    T.copy(
                        o_ub[0:valid_m, 0:EPI_N],
                        Output[
                            m_start:m_start + valid_m,
                            n_start + seg * EPI_N:n_start + (seg + 1) * EPI_N,
                        ],
                    )
                else:
                    T.copy(o_ub, o_ub16)
                    T.copy(
                        o_ub16[0:valid_m, 0:EPI_N],
                        Output[
                            m_start:m_start + valid_m,
                            n_start + seg * EPI_N:n_start + (seg + 1) * EPI_N,
                        ],
                    )

    return main


@tilelang.jit(out_idx=[-1], workspace_idx=[5], pass_configs=PASS_CONFIGS)
def _conv2d_direct_im2col_fp32_kernel(
    in_batch: int,
    in_channels: int,
    in_height: int,
    in_width: int,
    out_channels: int,
    k_h: int,
    k_w: int,
    stride: int,
    pt: int,
    pl: int,
    dh: int,
    dw: int,
    h_grid: int,
    w_grid: int,
    h_plane: int,
    win_pad: int,
    n_pad: int,
    k_pad: int,
    needs_x_zero: int,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
    compact_rows: bool = False,
    dtype: str = "float16",
    out_dtype: str = "float16",
    bias_dtype: str = "float",
):
    """Fused fp16 hi/lo split conv2d (general stride/pads, dense grid)."""
    accum_dtype = "float"
    batch, c_in = in_batch, in_channels
    s = stride
    c_out = out_channels
    h_out = h_grid
    w_out_k = w_grid
    m, k, n = c_out, k_pad, batch * h_out * w_out_k
    assert w_out_k % 16 == 0
    m_blocks = (m + BLOCK_M - 1) // BLOCK_M
    n_blocks = (n + BLOCK_N - 1) // BLOCK_N
    k_blocks = k // BLOCK_K
    taps = k_h * k_w
    # True K before BLOCK_K padding (see the fp16 kernel's K-tail comment):
    # rows [k_real, k_pad) of the LAST kb tile are never written by the tap
    # copies and MUST be zeroed, else stale L1 inf from a previous launch
    # folds 0*inf = NaN into the accumulator (huge->rect repro).
    k_real = c_in * taps
    tail0 = k_real - (k_blocks - 1) * BLOCK_K
    total = m_blocks * n_blocks
    assert k % BLOCK_K == 0, "fused fp32 direct im2col requires K padded to a BLOCK_K multiple"
    if w_out_k % BLOCK_N == 0:
        block_w = BLOCK_N
    elif BLOCK_N % w_out_k == 0 and w_out_k % 16 == 0:
        block_w = w_out_k
    else:
        block_w = 32 if w_out_k % 32 == 0 else 16
    runs_per_block = BLOCK_N // block_w
    # Epilogue segments: keeps o_ub/o_ub16 128-wide even when BLOCK_N=256
    # (halves per-kb weight re-reads without blowing the UB budget).
    EPI_N = 128
    n_seg = max(1, BLOCK_N // EPI_N)

    @T.prim_func
    def main(
            WeightHi: T.Tensor((c_out, k), dtype),
            WeightLo: T.Tensor((c_out, k), dtype),
            InputHi: T.Tensor((batch * c_in, h_plane * win_pad), dtype),
            InputLo: T.Tensor((batch * c_in, h_plane * win_pad), dtype),
            Bias: T.Tensor((c_out,), bias_dtype),
            epilog_workspace: T.Tensor((total, BLOCK_M, BLOCK_N), accum_dtype),
            Output: T.Tensor((m, n_pad), out_dtype),
    ):
        with T.Kernel(total, is_npu=True) as (cid, _):
            bm = cid // n_blocks
            bn = cid % n_blocks
            m_start = bm * BLOCK_M
            n_start = bn * BLOCK_N
            valid_m = T.min(BLOCK_M, m - m_start)

            w_hi = T.alloc_L1((BLOCK_M, BLOCK_K), dtype)
            w_lo = T.alloc_L1((BLOCK_M, BLOCK_K), dtype)
            x_hi = T.alloc_L1((BLOCK_K, BLOCK_N), dtype)
            x_lo = T.alloc_L1((BLOCK_K, BLOCK_N), dtype)
            c_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            o_ub = T.alloc_shared((BLOCK_M, EPI_N), accum_dtype)
            o_ub16 = T.alloc_shared((BLOCK_M, EPI_N), out_dtype)

            for kb in T.serial(k_blocks):
                k_start = kb * BLOCK_K
                valid_k = T.min(BLOCK_K, k - k_start)
                T.copy(
                    WeightHi[m_start:m_start + valid_m, k_start:k_start + valid_k],
                    w_hi[0:valid_m, 0:valid_k],
                )
                T.copy(
                    WeightLo[m_start:m_start + valid_m, k_start:k_start + valid_k],
                    w_lo[0:valid_m, 0:valid_k],
                )
                if needs_x_zero:  # noqa: SIM102
                    if kb == k_blocks - 1 and tail0 < BLOCK_K:
                        T.copy(InputHi[0:1, 0:1], x_hi[0:BLOCK_K, 0:BLOCK_N], pad_value=0.0)
                for tap in T.serial(taps):
                    kh = tap // k_w
                    kw = tap % k_w
                    c0 = T.max(k_start - tap * c_in, 0)
                    c1 = T.min(c_in, k_start + BLOCK_K - tap * c_in)
                    if c0 < c1:
                        for run in T.serial(runs_per_block):
                            base = n_start + run * block_w
                            oh = base % (h_out * w_out_k) // w_out_k
                            batch_idx = base // (h_out * w_out_k)
                            ih_p = (oh * s + kh * dh if compact_rows else oh + kh * dh)
                            iw_p0 = base % w_out_k + kw * dw
                            if batch_idx < batch and oh < h_out and base + block_w <= n and base % w_out_k + block_w <= w_out_k and (
                                    compact_rows or oh % s == 0):
                                T.copy(
                                    InputHi[
                                        batch_idx * c_in + c0:batch_idx * c_in + c1,
                                        ih_p * win_pad + iw_p0 + win_pad - pl:ih_p * win_pad +
                                        iw_p0 + win_pad - pl + block_w,
                                    ],
                                    x_hi[
                                        tap * c_in + c0 - k_start:tap * c_in + c1 - k_start,
                                        run * block_w:run * block_w + block_w,
                                    ],
                                )
                if needs_x_zero:  # noqa: SIM102
                    if kb == k_blocks - 1 and tail0 < BLOCK_K:
                        T.copy(InputLo[0:1, 0:1], x_lo[0:BLOCK_K, 0:BLOCK_N], pad_value=0.0)
                for tap in T.serial(taps):
                    kh = tap // k_w
                    kw = tap % k_w
                    c0 = T.max(k_start - tap * c_in, 0)
                    c1 = T.min(c_in, k_start + BLOCK_K - tap * c_in)
                    if c0 < c1:
                        for run in T.serial(runs_per_block):
                            base = n_start + run * block_w
                            oh = base % (h_out * w_out_k) // w_out_k
                            batch_idx = base // (h_out * w_out_k)
                            ih_p = (oh * s + kh * dh if compact_rows else oh + kh * dh)
                            iw_p0 = base % w_out_k + kw * dw
                            if batch_idx < batch and oh < h_out and base + block_w <= n and base % w_out_k + block_w <= w_out_k and (
                                    compact_rows or oh % s == 0):
                                T.copy(
                                    InputLo[
                                        batch_idx * c_in + c0:batch_idx * c_in + c1,
                                        ih_p * win_pad + iw_p0 + win_pad - pl:ih_p * win_pad +
                                        iw_p0 + win_pad - pl + block_w,
                                    ],
                                    x_lo[
                                        tap * c_in + c0 - k_start:tap * c_in + c1 - k_start,
                                        run * block_w:run * block_w + block_w,
                                    ],
                                )
                T.gemm_v0(w_hi, x_hi, c_frag, init=(kb == 0))
                T.gemm_v0(w_hi, x_lo, c_frag, init=False)
                T.gemm_v0(w_lo, x_hi, c_frag, init=False)

            T.copy(c_frag, epilog_workspace[cid, :, :])
            for seg in T.serial(n_seg):
                T.copy(
                    epilog_workspace[cid, :, seg * EPI_N:(seg + 1) * EPI_N],
                    o_ub,
                )
                for mm in T.serial(BLOCK_M):
                    if mm < valid_m:
                        T.tile.add(
                            o_ub[mm, 0:EPI_N],
                            o_ub[mm, 0:EPI_N],
                            Bias[m_start + mm],
                        )
                if out_dtype == "float":
                    T.copy(
                        o_ub[0:valid_m, 0:EPI_N],
                        Output[
                            m_start:m_start + valid_m,
                            n_start + seg * EPI_N:n_start + (seg + 1) * EPI_N,
                        ],
                    )
                else:
                    T.copy(o_ub, o_ub16)
                    T.copy(
                        o_ub16[0:valid_m, 0:EPI_N],
                        Output[
                            m_start:m_start + valid_m,
                            n_start + seg * EPI_N:n_start + (seg + 1) * EPI_N,
                        ],
                    )

    return main


@tilelang.jit(out_idx=[], pass_configs=PASS_CONFIGS)
def _post_nchw_kernel(N, Cout, h_out, w_out, w_k, grid_h, y_stride, st, ro, cst, out_dtype, RPC,
                      COB):
    """Dense-grid repack with subsample.
      row (n, co, oh) reads DENSE grid row gr = oh * st + ro (st: row stride,
      ro: row offset); cols via cst (column stride: 1 = contiguous band,
      >1 = per-row vgather with BYTE offsets cst*ow*elem).
      COB: cos per block -- merges the grid from N*Cout*bands to
      N*ceil(Cout/COB)*bands when per-block work is tiny (Cout=2048 x
      h_out=16 x w_k=16 -> 512B/block; per-block fixed cost dominated).
    """
    bands = (h_out + RPC - 1) // RPC
    cblocks = (Cout + COB - 1) // COB
    blocks = N * cblocks * bands

    @T.prim_func
    def main(
            Y: T.Tensor((Cout, y_stride), out_dtype),
            OFF: T.Tensor((w_out,), "uint32"),
            Out: T.Tensor((N * Cout * h_out, w_out), out_dtype),
    ):
        with T.Kernel(blocks, is_npu=True) as (bid, vid):
            n = bid // (cblocks * bands)
            rest = bid % (cblocks * bands)
            cb = rest // bands
            band = rest % bands
            oh0 = band * RPC
            rows = T.min(RPC, h_out - oh0)
            ybase = n * grid_h * w_k + (oh0 * st + ro) * w_k
            ub = T.alloc_ub((RPC, w_k), out_dtype)
            with T.Scope("V"):
                if cst == 1:
                    # contiguous band: 2D load + masked 2D store per co
                    for ci in T.serial(COB):
                        co = cb * COB + ci
                        if co < Cout:
                            obase = (n * Cout + co) * h_out + oh0
                            T.copy(Y[co, ybase:ybase + rows * w_k], ub[0:rows, 0:w_k])
                            T.copy(ub[0:rows, 0:w_k], Out[obase:obase + rows, 0:w_k])
                else:
                    # hardware vgather with BYTE offsets (probed): gather
                    # takes WHOLE buffers only, so run it row-by-row
                    row_in = T.alloc_ub((w_k,), out_dtype)
                    row_out = T.alloc_ub((w_out,), out_dtype)
                    off = T.alloc_ub((w_out,), "uint32")
                    T.copy(OFF, off)
                    for ci in T.serial(COB):
                        co = cb * COB + ci
                        if co < Cout:
                            for rr in T.serial(rows):
                                src = ybase + rr * st * w_k
                                T.copy(Y[co, src:src + w_k], row_in[0:w_k])
                                T.tile.gather(row_out, row_in, off, 0)
                                T.copy(
                                    row_out[0:w_k],
                                    Out[(n * Cout + co) * h_out + oh0 + rr, 0:w_k],
                                )

    return main


@tilelang.jit(out_idx=[], pass_configs=PASS_CONFIGS)
def _post_nchw_band_kernel(N, Cout, h_out, w_out, w_k, grid_h, y_stride, st, ro, cst, out_dtype,
                           RPC, COB):
    """Band vgather repack (cst > 1, st == 1): ONE 2D GM load [RPC x w_k] +
    per-row (UB->UB copy + hardware vgather) + ONE 2D GM store [RPC x w_out]
    per (co, band). Replaces the row-wise path's 3 GM ops per row with 2 GM
    ops per band.

    Component patterns are production-proven: 2D load from a flat 1D source
    slice (cst==1 path verbatim), runtime row index into 2D UB, full-width
    2D store (cst==1 path). Guarded by w_out % 16 == 0 (band rows stay 32B
    aligned -- w_out=12 faults per the UB alignment iron rule) and st == 1
    (compact s2 rows make the source band contiguous).
    """
    bands = (h_out + RPC - 1) // RPC
    cblocks = (Cout + COB - 1) // COB
    blocks = N * cblocks * bands

    @T.prim_func
    def main(
            Y: T.Tensor((Cout, y_stride), out_dtype),
            OFF: T.Tensor((w_out,), "uint32"),
            Out: T.Tensor((N * Cout * h_out, w_out), out_dtype),
    ):
        with T.Kernel(blocks, is_npu=True) as (bid, vid):
            n = bid // (cblocks * bands)
            rest = bid % (cblocks * bands)
            cb = rest // bands
            band = rest % bands
            oh0 = band * RPC
            rows = T.min(RPC, h_out - oh0)
            ybase = n * grid_h * w_k + (oh0 * st + ro) * w_k
            band_in = T.alloc_ub((RPC, w_k), out_dtype)
            band_out = T.alloc_ub((RPC, w_out), out_dtype)
            row_in = T.alloc_ub((w_k,), out_dtype)
            row_out = T.alloc_ub((w_out,), out_dtype)
            off = T.alloc_ub((w_out,), "uint32")
            with T.Scope("V"):
                T.copy(OFF, off)
                for ci in T.serial(COB):
                    co = cb * COB + ci
                    if co < Cout:
                        obase = (n * Cout + co) * h_out + oh0
                        T.copy(Y[co, ybase:ybase + rows * w_k], band_in[0:rows, 0:w_k])
                        for rr in T.serial(rows):
                            T.copy(band_in[rr, 0:w_k], row_in[0:w_k])
                            T.tile.gather(row_out, row_in, off, 0)
                            T.copy(row_out[0:w_out], band_out[rr, 0:w_out])
                        T.copy(band_out[0:rows, 0:w_out], Out[obase:obase + rows, 0:w_out])

    return main


def materialize_nchw(y, N, Cout, h_out, w_out, w_k, grid_h, st, ro, cst, out_dtype):
    """Output is host-allocated as a FLAT tensor then viewed NCHW: certain
    NCHW shapes (e.g. 128x128 bf16) make the wrapper's torch.empty pick a
    non-linear internal format whose storage the kernel's linear writes
    corrupt (AICore fault). A 1D allocation is always linear."""
    y_stride = y.shape[1]
    out_flat = torch.empty((N * Cout * h_out * w_out,), dtype=y.dtype, device=y.device)
    total = N * Cout * h_out
    # UB budget: RPC * w_k * elem <= 128KB. The old formula hardcoded 2B
    # elements (65536 // w_k); an fp32 y (out_dtype=float32 path) with
    # w_k=256 then allocates 256KB and faults.
    elem = y.dtype.itemsize
    rpc = max(1, min(131072 // (max(w_k, 1) * elem), h_out))
    # keep the block grid <= 32768 (2^16 grid dim overflow faults the launch)
    bands = (h_out + rpc - 1) // rpc
    while N * Cout * bands > 32768 and rpc > 1:
        rpc -= 1
        bands = (h_out + rpc - 1) // rpc
    # block-granularity merge: many tiny blocks (e.g. Cout=2048, 16x16 rows
    # -> 512B/block) pay ~110ns fixed cost each; batch cos per block down to
    # a target block count. Unchanged (COB=1) when the grid is already
    # coarse. Targets: w_k >= 128 -> 256 blocks, w_k == 64 -> 64 blocks,
    # w_k <= 32 -> 32 blocks. The vgather path (cst > 1) keeps the 256
    # target: its bottleneck is the per-row gather.
    if cst == 1:
        target = 256 if w_k >= 128 else (64 if w_k >= 64 else 32)
    else:
        target = 256
    cob = 1
    while N * Cout * bands // cob > target and cob < 64:
        cob *= 2
    # vgather BYTE-offset table (cst*ow*elem_size), cached per (w_out, cst, dtype)
    import numpy as _np
    key = (w_out, int(cst), y.dtype)
    off = _GATHER_OFFSET_CACHE.get(key)
    if off is None:
        off = torch.from_numpy(
            (_np.arange(w_out, dtype=_np.uint32) * int(cst) * y.dtype.itemsize)).to(y.device)
        _GATHER_OFFSET_CACHE[key] = off
    # Band vgather path (cst > 1): ONE 2D load + per-row UB gather + ONE 2D
    # store per (co, band) replaces the row-wise path's 3 GM ops per row.
    # Guards: st == 1 and ro == 0 (source band contiguous -- compact s2 rows),
    # w_out % 16 == 0 (band rows stay 32B aligned; w_out=12 faults per the UB
    # alignment iron rule -> row-wise fallback).
    if cst > 1 and st == 1 and ro == 0 and w_out % 16 == 0:
        # The band kernel's per-block work (2D load + per-row gather + 2D
        # store) is larger than the row-wise path's, so it prefers fewer,
        # bigger blocks (target ~96-128 blocks for small grids, 256 for
        # large). UB budget: band_in (RPC, w_k) + band_out (RPC, w_out) both
        # live -- cap RPC so the SUM stays under ~140KB (an fp32 y with
        # w_k=256 and the shared rpc=128 would need 192KB and fault).
        band_rpc = max(1, min(140 * 1024 // ((w_k + w_out) * elem), h_out))
        band_bands = (h_out + band_rpc - 1) // band_rpc
        grid = N * Cout * band_bands
        target_band = 128 if grid <= 512 else 256
        band_cob = 1
        while grid // band_cob > target_band and band_cob < 64:
            band_cob *= 2
        fast(
            _post_nchw_band_kernel(
                N,
                Cout,
                h_out,
                w_out,
                w_k,
                grid_h,
                int(y_stride),
                int(st),
                int(ro),
                int(cst),
                out_dtype,
                band_rpc,
                band_cob,
            ),
            y,
            off,
            out_flat.view(total, w_out),
        )
        return out_flat.view(N, Cout, h_out, w_out)
    fast(
        _post_nchw_kernel(
            N,
            Cout,
            h_out,
            w_out,
            w_k,
            grid_h,
            int(y_stride),
            int(st),
            int(ro),
            int(cst),
            out_dtype,
            rpc,
            cob,
        ),
        y,
        off,
        out_flat.view(total, w_out),
    )
    return out_flat.view(N, Cout, h_out, w_out)


@tilelang.jit(out_idx=[], workspace_idx=[4], pass_configs=PASS_CONFIGS)
def _conv2d_fused_nchw_kernel(
    in_batch: int,
    cin_pad: int,
    out_channels: int,
    k_h: int,
    k_w: int,
    sh: int,
    pt: int,
    pl: int,
    dh: int,
    dw: int,
    h_out: int,
    w_out: int,
    w_out_k: int,
    h_plane: int,
    win_pad: int,
    k_pad: int,
    needs_x_zero: int,
    BLOCK_M: int,
    BLOCK_N: int,
    RB: int,
    BLOCK_K: int,
    dtype: str = "float16",
    out_dtype: str = "float16",
):
    TAPS = k_h * k_w
    k = k_pad
    k_real = cin_pad * TAPS
    k_blocks = k // BLOCK_K
    tail0 = k_real - (k_blocks - 1) * BLOCK_K
    rows = in_batch * h_out
    row_blocks = (rows + RB - 1) // RB
    m_blocks = (out_channels + BLOCK_M - 1) // BLOCK_M
    total = m_blocks * row_blocks

    @T.prim_func
    def main(
            Weight: T.Tensor((out_channels, k), dtype),
            Input: T.Tensor((in_batch * cin_pad, h_plane * win_pad), dtype),
            Bias: T.Tensor((out_channels,), "float"),
            Out: T.Tensor((in_batch * out_channels * h_out, w_out), out_dtype),
            epilog_workspace: T.Tensor((total, BLOCK_M, BLOCK_N), "float"),
    ):
        with T.Kernel(total, is_npu=True) as (cid, vid):
            bm = cid // row_blocks
            rb = cid % row_blocks
            row0 = rb * RB
            valid_rows = T.min(RB, rows - row0)
            m_start = bm * BLOCK_M
            valid_m = T.min(BLOCK_M, out_channels - m_start)

            w_l1 = T.alloc_L1((BLOCK_M, BLOCK_K), dtype)
            x_l1 = T.alloc_L1((BLOCK_K, BLOCK_N), dtype)
            c_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float")
            o_ub = T.alloc_shared((BLOCK_M, BLOCK_N), "float")
            o_ub16 = T.alloc_shared((BLOCK_M, BLOCK_N), out_dtype)

            for kb in T.serial(k_blocks):
                k_start = kb * BLOCK_K
                T.copy(
                    Weight[m_start:m_start + valid_m, k_start:k_start + BLOCK_K],
                    w_l1[0:valid_m, 0:BLOCK_K],
                )
                # K-tail zero (proven): rows [tail0, BLOCK_K) of the last kb
                if needs_x_zero:  # noqa: SIM102
                    if kb == k_blocks - 1 and tail0 < BLOCK_K:
                        T.copy(Input[0:1, 0:1], x_l1[0:BLOCK_K, 0:BLOCK_N], pad_value=0.0)
                for tap in T.serial(TAPS):
                    khi = tap // k_w
                    kwi = tap % k_w
                    c0 = T.max(k_start - tap * cin_pad, 0)
                    c1 = T.min(cin_pad, k_start + BLOCK_K - tap * cin_pad)
                    if c0 < c1:
                        for r in T.serial(RB):
                            if r < valid_rows:
                                row = row0 + r
                                n_img = row // h_out
                                oh = row % h_out
                                ih_p = oh * sh + khi * dh
                                iw_p0 = kwi * dw
                                col0 = ih_p * win_pad + iw_p0 + win_pad - pl
                                T.copy(
                                    Input[
                                        n_img * cin_pad + c0:n_img * cin_pad + c1,
                                        col0:col0 + w_out_k,
                                    ],
                                    x_l1[
                                        tap * cin_pad + c0 - k_start:tap * cin_pad + c1 - k_start,
                                        r * w_out_k:r * w_out_k + w_out_k,
                                    ],
                                )
                T.gemm_v0(w_l1, x_l1, c_frag, init=(kb == 0))

            T.copy(c_frag, epilog_workspace[cid, :, :])
            T.copy(epilog_workspace[cid, :, :], o_ub)
            for mm in T.serial(BLOCK_M):
                if mm < valid_m:
                    T.tile.add(o_ub[mm, 0:BLOCK_N], o_ub[mm, 0:BLOCK_N], Bias[m_start + mm])
            T.copy(o_ub, o_ub16)
            # NCHW-direct store: one 1D store per (co, output row). A
            # partial-width 2D UB->GM store mis-steps the source row stride
            # (known footgun -- corrupts rows >= 1), so 1D it is; the store
            # count is capped by the dispatch heuristic (wave quantization).
            for mm in T.serial(BLOCK_M):
                if mm < valid_m:
                    for r in T.serial(RB):
                        if r < valid_rows:
                            row = row0 + r
                            n_img = row // h_out
                            oh = row % h_out
                            T.copy(
                                o_ub16[mm, r * w_out_k:r * w_out_k + w_out],
                                Out[
                                    (n_img * out_channels + m_start + mm) * h_out + oh,
                                    0:w_out,
                                ],
                            )

    return main


def conv2d_fused_nchw(
    xp,
    wt,
    bf,
    cin_pad,
    N,
    Cout,
    kh,
    kw,
    sh,
    pt,
    pl,
    dh,
    dw,
    h_out,
    w_out,
    w_out_k,
    HP1p,
    Wp,
    k_pad,
    needs_zero,
    in_dtype,
    out_dtype,
):
    """Host glue: fused GEMM + NCHW store. Returns the NCHW tensor or None
    when the shape exceeds this path's budget (caller falls back)."""
    import os as _os

    if _os.environ.get("CONV2D_NO_FUSED", ""):
        return None
    BLOCK_K = 256
    # rows per block: keep BLOCK_N near 128 (weight amortization of the old
    # tiling) while covering WHOLE rows (epilogue alignment)
    RB = max(1, 128 // w_out_k)
    BLOCK_N = RB * w_out_k
    if BLOCK_N > 256:
        return None
    BLOCK_M = 64 if Cout <= 64 else 128
    rows = N * h_out
    row_blocks = (rows + RB - 1) // RB
    m_blocks = (Cout + BLOCK_M - 1) // BLOCK_M
    if m_blocks * row_blocks > 32768:  # grid dim cap
        return None
    # Wave-quantization guard: per-(co,row) 1D epilogue stores cost
    # ~160ns each; blocks beyond one device wave (~20 cores) double
    # that cost.  Keep the fused path for small-K shapes.
    if m_blocks * row_blocks > 20 and k_pad > 512:
        return None

    kernel = _conv2d_fused_nchw_kernel(
        N,
        cin_pad,
        Cout,
        kh,
        kw,
        sh,
        pt,
        pl,
        dh,
        dw,
        h_out,
        w_out,
        w_out_k,
        HP1p,
        Wp,
        k_pad,
        needs_zero,
        BLOCK_M,
        BLOCK_N,
        RB,
        BLOCK_K,
        dtype="float16",
        out_dtype=out_dtype,
    )
    out_t = torch.bfloat16 if out_dtype == "bfloat16" else torch.float16
    out_flat = torch.empty((N * Cout * h_out * w_out,), dtype=out_t, device=xp.device)
    fast(
        kernel,
        wt,
        xp,
        bf,
        out_flat.view(N * Cout * h_out, w_out),
    )
    return out_flat.view(N, Cout, h_out, w_out)


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def _pre_transpose_kernel(N, Cin, Cin_pad, XT_rows, H, W, Wpad, in_dtype, CB):
    # CB channels per block: tiny per-block work (e.g. Cin_pad=2048 with
    # [16,16] rows -> 1KB/block), fixed cost dominated; batching to ~128
    # blocks recovers it. CB=1 reproduces the old grid.
    # XT_rows: allocated XT row count -- the 1x1 GEMM reads K=k_pad rows, so
    # the caller passes k_pad; rows [Cin_pad, XT_rows) are zero-filled (a
    # Cin_pad < k_pad GEMM otherwise reads XT out of bounds and folds
    # 0 x garbage-from-adjacent-GM into the accumulator -> NaN roulette).
    cblocks = (XT_rows + CB - 1) // CB
    fill = Wpad > W  # tail zero-fill only needed when the width pads

    @T.prim_func
    def main(
            X: T.Tensor((N * Cin * H, W), in_dtype),
            XT: T.Tensor((XT_rows, N * H * Wpad), "float16"),
    ):
        with T.Kernel(cblocks, is_npu=True) as (cid, vid):
            ub_in = T.alloc_ub((H, Wpad), in_dtype)
            ub16 = T.alloc_ub((H, Wpad), "float16")
            ub32 = T.alloc_ub((H, Wpad), "float32")
            with T.Scope("V"):
                for ci in T.serial(CB):
                    c = cid * CB + ci
                    if c < Cin:
                        for nn in T.serial(N):
                            if fill:
                                T.tile.fill(ub_in, 0.0)
                            T.copy(
                                X[(nn * Cin + c) * H:(nn * Cin + c) * H + H, 0:Wpad],
                                ub_in[0:H, 0:Wpad],
                                pad_value=0.0,
                            )
                            if in_dtype == "bfloat16":
                                T.copy(ub_in, ub32)
                                T.copy(ub32, ub16)
                            else:
                                T.copy(ub_in, ub16)
                            T.copy(
                                ub16[0:H, 0:Wpad],
                                XT[c, nn * H * Wpad:nn * H * Wpad + H * Wpad],
                            )
                    else:
                        if c < XT_rows:
                            T.tile.fill(ub16, 0.0)
                            for nn in T.serial(N):
                                T.copy(
                                    ub16[0:H, 0:Wpad],
                                    XT[c, nn * H * Wpad:nn * H * Wpad + H * Wpad],
                                )

    return main


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def _pre_transpose_chunk_kernel(N, Cin, Cin_pad, XT_rows, H, W, Wpad, in_dtype, CB, BK):
    """Row-chunked variant of _pre_transpose_kernel for LARGE images: the
    whole-plane kernel allocates (H, Wpad) UB buffers, which faults when
    H*Wpad*elem exceeds the UB limit (256x256 fp16 = 262KB, bf16 524KB;
    H,W up to 256). Same grid/casting semantics; the per-(c, n) H-row copy
    runs in BK-row chunks. XT_rows: see _pre_transpose_kernel (zero rows up
    to the GEMM's K extent)."""
    # CB channels per block (see _pre_transpose_kernel note)
    cblocks = (XT_rows + CB - 1) // CB
    fill = Wpad > W
    n_chunks = (H + BK - 1) // BK

    @T.prim_func
    def main(
            X: T.Tensor((N * Cin * H, W), in_dtype),
            XT: T.Tensor((XT_rows, N * H * Wpad), "float16"),
    ):
        with T.Kernel(cblocks, is_npu=True) as (cid, vid):
            ub_in = T.alloc_ub((BK, Wpad), in_dtype)
            ub16 = T.alloc_ub((BK, Wpad), "float16")
            ub32 = T.alloc_ub((BK, Wpad), "float32")
            with T.Scope("V"):
                for ci in T.serial(CB):
                    c = cid * CB + ci
                    if c < Cin:
                        for nn in T.serial(N):
                            for ch in T.serial(n_chunks):
                                r0 = ch * BK
                                r1 = T.min(H, r0 + BK)
                                if fill:
                                    T.tile.fill(ub_in, 0.0)
                                T.copy(
                                    X[(nn * Cin + c) * H + r0:(nn * Cin + c) * H + r1, 0:Wpad],
                                    ub_in[0:r1 - r0, 0:Wpad],
                                    pad_value=0.0,
                                )
                                if in_dtype == "bfloat16":
                                    T.copy(ub_in, ub32)
                                    T.copy(ub32, ub16)
                                else:
                                    T.copy(ub_in, ub16)
                                T.copy(
                                    ub16[0:r1 - r0, 0:Wpad],
                                    XT[
                                        c,
                                        nn * H * Wpad + r0 * Wpad:nn * H * Wpad + r1 * Wpad,
                                    ],
                                )
                    else:
                        if c < XT_rows:
                            T.tile.fill(ub16, 0.0)
                            for nn in T.serial(N):
                                for ch in T.serial(n_chunks):
                                    r0 = ch * BK
                                    r1 = T.min(H, r0 + BK)
                                    T.copy(
                                        ub16[0:r1 - r0, 0:Wpad],
                                        XT[
                                            c,
                                            nn * H * Wpad + r0 * Wpad:nn * H * Wpad + r1 * Wpad,
                                        ],
                                    )

    return main


def _conv2d_1x1(x, filter, bias, N, C, H, W, Cout, in_dtype, out_dtype):
    """Self-contained 1x1 conv: transpose kernel -> pure GEMM -> repack."""
    Cin_pad = _ceil16(C)
    Wpad = _ceil16(W)
    flat_n = N * H * Wpad
    n_pad = (flat_n + 127) // 128 * 128
    k_pad = (Cin_pad + 255) // 256 * 256

    x16 = x.reshape(N * C * H, W).contiguous()
    cb = max(1, min(64, (Cin_pad + 127) // 128))
    # XT_rows=k_pad: the 1x1 GEMM reads K=k_pad rows; rows [Cin_pad, k_pad)
    # are zero-filled by the transpose kernels (an unpadded XT with
    # Cin_pad < k_pad feeds the GEMM out-of-bounds garbage -> NaN).
    # Large images (e.g. 256x256: fp16 262KB / bf16 524KB) overflow the
    # whole-plane (H, Wpad) UB buffers -> row-chunked variant (H,W up to
    # 256).
    if H * Wpad * (8 if in_dtype == "bfloat16" else 4) > 150 * 1024:
        xt = fast(
            _pre_transpose_chunk_kernel(N, C, Cin_pad, k_pad, H, W, Wpad, in_dtype, cb, 16), x16)
    else:
        xt = fast(_pre_transpose_kernel(N, C, Cin_pad, k_pad, H, W, Wpad, in_dtype, cb), x16)

    wt, _, bf = prep_weight(
        filter.reshape(Cout, C, 1).contiguous(),
        bias.contiguous(),
        Cout,
        C,
        1,
        Cin_pad,
        k_pad,
        in_dtype,
        0,
    )

    # BLOCK_N=256 halves the input re-reads for small-n GEMMs (e.g.
    # M=2048 K=2048 N=512); n_pad is re-derived for the 256 tiling so no
    # partial n-block appears.
    if n_pad % 256 == 0 and n_pad // 256 >= 1:
        bn = 256
    else:
        bn = 128
    kern = _conv2d_1x1_kernel(
        Cout,
        k_pad,
        n_pad,
        True,
        64,
        bn,
        256,
        dtype="float16",
        out_dtype=out_dtype,
    )
    y = fast(kern, wt, xt, bf)
    return materialize_nchw(y, N, Cout, H, W, Wpad, H, 1, 0, 1, out_dtype)


def conv_2d(x, filter, bias, strides, pads, dilations=None):
    """Conv2D schema entry -- fully self-contained (kernel-only transforms)."""
    if dilations is None:
        dilations = [1, 1]
    sh, sw = int(strides[0]), int(strides[1])
    pt, pb, pl, pr = int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])
    dh, dw = int(dilations[0]), int(dilations[1])
    N, C, H, W = x.shape
    Cout, cin, kh, kw = filter.shape
    if bias is None:
        bias = torch.zeros(Cout, dtype=x.dtype, device=x.device)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("fp16/bf16/fp32 only (got %s)" % x.dtype)

    in_dtype = str(x.dtype).replace("torch.", "")
    split = 1 if in_dtype == "float32" else 0
    out_dtype = in_dtype

    # ---- 1x1 fast path (trivial geometry only) ----
    # Routing large-Cin 1x1 through the general path is ~10x slower (narrow
    # w_out_k=16 window copies in the direct kernel are pathological). The
    # shortcut stays for ALL 1x1; its transpose/weight/repack stages carry
    # the COB/CB block-merge fixes.
    if (kh == 1 and kw == 1 and sh == 1 and sw == 1 and pt == 0 and pb == 0 and pl == 0 and
            pr == 0 and dh == 1 and dw == 1 and split == 0):
        return _conv2d_1x1(x, filter, bias, N, C, H, W, Cout, in_dtype, out_dtype)

    # ---- output geometry (golden semantics) ----
    h_out = (H + pt + pb - dh * (kh - 1) - 1) // sh + 1
    w_out = (W + pl + pr - dw * (kw - 1) - 1) // sw + 1
    if h_out <= 0 or w_out <= 0:
        raise NotImplementedError("degenerate output geometry")

    # ---- dense padded grid + plane geometry ----
    h_dense = H + pt + pb - dh * (kh - 1)  # dense grid rows
    w_dense = W + pl + pr - dw * (kw - 1)  # dense grid cols
    w_out_k = _ceil16(w_dense)  # 16-padded dense cols
    # 48-lane grids force 16-wide im2col DMA runs in the GEMM: block_w
    # degenerates (48 % {128,256} != 0 and {128,256} % 48 != 0), quadrupling
    # the MTE2 op count at 1/4 width vs 64-wide runs. Widening to a
    # 64-multiple costs +33% garbage lanes -- the repack's dst-clamp
    # discards them, and the lanes read the plane's zero-pad region (no inf
    # risk) -- but buys 64-wide runs. Only 48-lane grids are widened;
    # 16/32/64+ lane grids already map to full-row or 32/64-wide runs
    # (16-lane grids run full-row 16-wide, which is fine).
    if w_out_k % 64 == 48:
        w_out_k += 16
    HP1 = pt + H + pb + 1  # plane rows (+ guard row)
    # Plane width: tap reads at dense col ow+kw*dw reach (w_out_k-1)+dw(kw-1)
    # and the circular-shift compensation -pl adds one row stride; the
    # verified formula is Wp = ceil16(w_out_k + dw*(kw-1)).
    Wp = _ceil16(w_out_k + dw * (kw - 1))

    cin_pad = _ceil16(cin)
    taps = kh * kw
    k = cin_pad * taps
    BLOCK_K = 256
    k_pad = (k + BLOCK_K - 1) // BLOCK_K * BLOCK_K
    # Stride-2 convolutions can compact the GEMM rows directly. Columns remain
    # dense lanes because consecutive output columns read every other input
    # element; the existing materializer gathers those lanes cheaply.
    compact_s2_rows = sh == 2 and sw == 2
    gemm_h = h_out if compact_s2_rows else h_dense
    n_real = N * gemm_h * w_out_k
    needs_zero = int(k % BLOCK_K != 0)

    # ---- preprocessing: pad + weight in ONE launch when the UB budget fits
    # (saves ~25-40us of per-launch fixed cost); else the separate kernels.
    fused_pre = None
    if split == 0:
        fused_pre = pad_input_weight(
            x.contiguous().view(N * C, H, W),
            filter.reshape(Cout, C, taps).contiguous(),
            bias.contiguous(),
            N,
            C,
            cin_pad,
            H,
            W,
            HP1,
            Wp,
            pt,
            in_dtype,
            Cout,
            taps,
            k_pad,
        )
    if fused_pre is not None:
        xp, wt, bf = fused_pre
        xp_lo = None
        wt_lo = None
    else:
        xp, xp_lo = pad_input(
            x.contiguous().view(N * C, H, W),
            N,
            C,
            cin_pad,
            H,
            W,
            HP1,
            Wp,
            pt,
            in_dtype,
            split,
        )
        wt, wt_lo, bf = prep_weight(
            filter.reshape(Cout, C, taps).contiguous(),
            bias.contiguous(),
            Cout,
            C,
            taps,
            cin_pad,
            k_pad,
            in_dtype,
            split,
        )
    HP1p = xp.shape[0] // (N * cin_pad)  # == HP1 (direct-row design)

    # ---- fused GEMM + NCHW-direct store ----
    # Fused kernel writes GEMM output directly to NCHW, eliminating the
    # separate repack launch.  Measured faster on small outputs (case1
    # +16.5%) but slower on large outputs (per-(co,row) 1D stores dominate
    # multi-wave grids) -- hence the n_real < 8192 gate.  Falls back to the
    # non-fused path otherwise.  Set CONV2D_NO_FUSED=1 to disable.
    if split == 0 and sw == 1 and n_real < 8192:
        y = conv2d_fused_nchw(
            xp.view(N * cin_pad, HP1p * Wp),
            wt,
            bf,
            cin_pad,
            N,
            Cout,
            kh,
            kw,
            sh,
            pt,
            pl,
            dh,
            dw,
            h_out,
            w_out,
            w_out_k,
            HP1p,
            Wp,
            k_pad,
            needs_zero,
            in_dtype,
            out_dtype,
        )
        if y is not None:
            return y

    xp2 = xp.view(N * cin_pad, HP1p * Wp)
    block_m = 64 if Cout <= 64 else 128
    # Big outputs: BLOCK_N=256 halves the per-kb weight re-reads; the
    # epilogue runs in 128-wide segments so the fp16/bf16 cast stays inside
    # the kernel (no host aclnn cast). fp32 fused keeps BLOCK_N=128 (L1
    # budget). Mid-size outputs (2560 < n_real < 8192): BN=128 already
    # exceeds one device wave (~20 cores), so BN=256's halved parallelism
    # still covers a wave while halving the weight re-reads. EXCLUDED:
    # n_real <= 2560 (16 blocks are already single-wave; halving blocks
    # halves parallelism) and fp32 fused (L1 budget, no win).
    small_n_256 = 2560 < n_real < 8192
    BLOCK_N = 256 if (split == 0 and (n_real >= 8192 or small_n_256)) else 128
    n_pad = (n_real + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    if split == 1:
        kernel = _conv2d_direct_im2col_fp32_kernel(
            N,
            cin_pad,
            H,
            W,
            Cout,
            kh,
            kw,
            sh,
            pt,
            pl,
            dh,
            dw,
            gemm_h,
            w_out_k,
            HP1p,
            Wp,
            n_pad,
            k_pad,
            needs_zero,
            block_m,
            BLOCK_N,
            BLOCK_K,
            compact_rows=compact_s2_rows,
            dtype="float16",
            out_dtype=out_dtype,
        )
        y = fast(
            kernel,
            wt,
            wt_lo,
            xp2,
            xp_lo.view(N * cin_pad, HP1p * Wp),
            bf,
        )
    else:
        kernel = _conv2d_direct_im2col_kernel(
            N,
            cin_pad,
            H,
            W,
            Cout,
            kh,
            kw,
            sh,
            pt,
            pl,
            dh,
            dw,
            gemm_h,
            w_out_k,
            HP1p,
            True,
            Wp,
            n_pad,
            k_pad,
            needs_zero,
            block_m,
            BLOCK_N,
            BLOCK_K,
            compact_rows=compact_s2_rows,
            dtype="float16",
            out_dtype=out_dtype,
        )
        y = fast(kernel, wt, xp2, bf)

    # NCHW materialization.  For compact_s2_rows, GEMM rows are already
    # output rows; columns still use dense lanes and are gathered with cst=2.
    mat_grid_h = h_out if compact_s2_rows else h_dense
    mat_row_stride = 1 if compact_s2_rows else sh
    return materialize_nchw(
        y,
        N,
        Cout,
        h_out,
        w_out,
        w_out_k,
        mat_grid_h,
        mat_row_stride,
        0,
        sw,
        out_dtype,
    )


# =============================================================================
# Golden reference: PyTorch functional conv2d (for verification)
# =============================================================================


def _conv_2d_golden(
    x: torch.Tensor,
    filter: torch.Tensor,
    bias: torch.Tensor,
    strides: list,
    pads: list,
    dilations: list = None,
) -> torch.Tensor:
    """PyTorch golden reference for conv2d."""
    if dilations is None:
        dilations = [1, 1]
    pt, pb, pl, pr = pads
    import torch.nn.functional as F
    return F.conv2d(
        F.pad(x, (pl, pr, pt, pb)),
        filter,
        bias,
        stride=tuple(strides),
        padding=0,
        dilation=tuple(dilations),
    )


# =============================================================================
# Minimal __main__ test
# =============================================================================

if __name__ == "__main__":
    import torch
    torch.manual_seed(42)

    N, C, H, W = 1, 16, 8, 8
    Cout, Kh, Kw = 16, 3, 3

    x = (torch.rand(N, C, H, W, dtype=torch.float32) - 0.5).to(torch.float16).npu()
    w = (torch.rand(Cout, C, Kh, Kw, dtype=torch.float32) - 0.5).to(torch.float16).npu()
    b = (torch.rand(Cout, dtype=torch.float32) - 0.5).to(torch.float16).npu()

    strides = [1, 1]
    pads = [1, 1, 1, 1]
    dilations = [1, 1]

    y = conv_2d(x, w, b, strides, pads, dilations)
    torch.npu.synchronize()

    ref = _conv_2d_golden(x.cpu(), w.cpu(), b.cpu(), strides, pads, dilations)

    torch.testing.assert_close(
        y.cpu(),
        ref,
        rtol=1e-2,
        atol=1e-2,
    )
    print("Test Passed!")
