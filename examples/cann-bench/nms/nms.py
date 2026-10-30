# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms of the CANN Open Software License Agreement Version 2.0 (the
# "License"). You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED.
# ----------------------------------------------------------------------------

"""NMS (Non-Maximum Suppression) TileLang kernel — greedy NMS (Expert mode).

Sort-bitmap NMS pipeline for Ascend NPU. Sort scores descending, reorder
boxes, pack pairwise IoU threshold decisions into a uint16 bitmap, then run
a blocked greedy bit-scan. Output keep indices match the torch golden
exactly (index list equality, not tolerance).

Algorithm:
    keep = NMS(boxes, scores, iou_threshold)
    IoU(i, j) = inter / (area_i + area_j - inter + 1e-6)
    suppress when ~(iou <= thr)  (NaN IoU suppresses, matching golden)

Reference: pure-PyTorch greedy NMS (golden_nms in test_nms.py)

Key design points:
- 4-kernel pipeline: sort -> vectorized gather -> IoU bitmap -> greedy bit-scan
- IoU decisions packed 1 bit/pair into a uint16 bitmap (2MB @N=4096 vs 64MB
  for an fp32 matrix): the O(N^2) GM round trip collapses 26x
- Scalar-row x vector-column IoU chain: j-vectors resident in UB across all
  rows of a block, 13 tensor-scalar ops per row, bit-exact vs torch golden
  (incl. +/-inf / NaN propagation, no clamps in the IoU chain)
- Strictly-lower-triangle block skipping (garbage bits provably harmless)
- Word-major greedy fast path: a zero keepmask word skips 16 bit tests
  (bitwise_and only clears bits, so a zero word never resurrects); live bits
  re-read per box so in-word suppression chains stay exact
- Double-buffered gather emit with v<->mte3 flags (manually unrolled so slot
  indices stay compile-time constants; flag tokens closed by a teardown wait)
- Tail blocks handled by T.copy pad_value + an affine loop bound (no manual
  host-side padding)

Interface: nms(boxes, scores, iou_threshold) -> keep_indices
"""

import torch

import tilelang
from tilelang import language as T

tilelang.cache.clear_cache()

_MAX_N = 4096  # greedy chunk [B, NW16] uint16 must stay <= 128KB (UB budget)

# PC_P2: manual barrier skeleton (Expert mode), validated on 910B3.
_PC_P2 = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


# ========== Tiling (pure host-side Python integer arithmetic) ==========
def _tiling(N: int) -> dict:
    """Derive all block/bitmap geometry from N (JIT compile-time constants).

    UB budget (guide 2.11), empirically anchored on 910B3: K3 peak
    = 5*BN*4 + 5*BM*4 + 2*BM*4 + 4*BN*4 + BM*BN/8 bytes (j vecs, i vecs, ti,
    w/h/u/tmp, cmpb; cmp16 aliases cmpb via reinterpretcast). The N>=3584
    tier BM=32 was selected by measured load balance (BM=64 -> -7%, BM=128
    -> +34%: busiest-core serial pairs decide, not GM bandwidth).
    """
    N_aligned = max(((N + 31) // 32) * 32, 1024)  # sort UB alignment
    if N >= 3584:
        BM, BN = 32, 2048  # load-balance optimum (see docstring)
    elif N >= 2048:
        BM, BN = 64, 1024
    elif N >= 1538:
        BM, BN = 64, 512
    elif N >= 512:
        BM, BN = 32, 512
    else:
        BM, BN = 16, 512
    NCG = (N + BN - 1) // BN  # K3 column groups
    NRB = (N + BM - 1) // BM  # K3 row blocks
    B = 256  # K4 batch (256 bits = 32B window)
    NB = (N + B - 1) // B  # K4 batches
    N_pad_rows = max(NRB * BM, NB * B)
    NW16 = max(NCG * BN, NB * B) // 16  # uint16 words per bitmap row
    return {
        "N_aligned": N_aligned,
        "BM": BM,
        "BN": BN,
        "NCG": NCG,
        "NRB": NRB,
        "B": B,
        "NB": NB,
        "N_pad_rows": N_pad_rows,
        "NW16": NW16,
    }


# ========== K1: sort scores descending -> sort_idx (original indices) ==========
_sort_cache = {}


@tilelang.jit(out_idx=[1], pass_configs=_PC_P2)
def _sort_kernel(N, N_aligned, dtype="float"):

    @T.prim_func
    def main(
        scores: T.Tensor((N,), dtype),
        sort_idx_out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            scores_ub = T.alloc_ub((N_aligned,), dtype)
            sort_pairs_ub = T.alloc_ub((N_aligned * 2,), dtype)
            idx_ub = T.alloc_ub((N_aligned,), dtype)
            T.copy(scores[0:N], scores_ub[0:N])
            T.barrier_all()  # MTE2 -> V
            T.tile.sort(sort_pairs_ub, scores_ub, N)  # desc (val, idx) pairs
            T.barrier_all()  # V -> V (gather_mask consumer)
            T.tile.gather_mask(idx_ub, sort_pairs_ub, "P1010")  # odd = indices
            T.barrier_all()  # V -> MTE3
            T.copy(idx_ub[0:N], sort_idx_out[0:N])

    return main


# ========== K2: reorder boxes by sort_idx -> coords [4, N] (descending SoA) ==========
_gather_cache = {}


@tilelang.jit(out_idx=[2], pass_configs=_PC_P2)
def _gather_kernel(N, dtype="float"):
    # Row pitch is padded to 8 elements: slot 1's base offset N_pad*4B must
    # be 32B-aligned — a raw [2, N] layout with N % 8 != 0 misaligns slot 1
    # and raises an aicore exception.
    N_pad = ((N + 7) // 8) * 8

    @T.prim_func
    def main(
        # boxes is passed as the original [N, 4] torch tensor; the kernel
        # declares a FLAT [N*4] view over the same GM bytes (the gm2ub
        # lowering of a 2D->1D flatten copy computes a non-zero row stride
        # and lands OOB; a 1D->1D full copy is the validated pattern).
        boxes: T.Tensor((N * 4,), dtype),
        sort_idx: T.Tensor((N,), dtype),
        coords: T.Tensor((4, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            # gather source must be a FLAT 1D buffer — a multi-row 2D src
            # silently corrupts AscendC::Gather addressing.
            tab = T.alloc_ub((N * 4,), dtype)
            idx_ub = T.alloc_ub((N,), dtype)
            offi = T.alloc_ub((N,), "int32")  # byte offsets = idx*16 + c*4
            # per-segment uint32 views (unrolled bodies share one C++ scope:
            # a single reused reinterpretcast name redefines the view var)
            offv0 = T.alloc_ub((N,), "uint32")
            offv1 = T.alloc_ub((N,), "uint32")
            offv2 = T.alloc_ub((N,), "uint32")
            offv3 = T.alloc_ub((N,), "uint32")
            # Double-buffered emit: iteration c+1's V work overlaps
            # iteration c's MTE3 GM write via per-slot v<->mte3 flags.
            # Manually unrolled x4 so slot indices stay compile-time consts.
            outw2 = T.alloc_ub((2, N_pad), dtype)
            T.copy(boxes[0 : N * 4], tab[0 : N * 4])
            T.copy(sort_idx[0:N], idx_ub[0:N])
            T.barrier_all()  # MTE2 -> V
            T.set_flag("mte3", "v", 0)  # both slots writable
            T.set_flag("mte3", "v", 1)
            # --- c = 0 (slot 0) ---
            T.wait_flag("mte3", "v", 0)
            T.tile.cast(offi, idx_ub, "CAST_RINT", N)
            T.tile.mul(offi, offi, 16)
            T.tile.add(offi, offi, 0)
            T.reinterpretcast(offv0, offi, "uint32_t")
            T.tile.gather(outw2[0, :N], tab, offv0, 0)
            T.set_flag("v", "mte3", 0)
            T.wait_flag("v", "mte3", 0)
            T.copy(outw2[0, 0:N], coords[0, 0:N])
            T.set_flag("mte3", "v", 0)
            # --- c = 1 (slot 1; V overlaps c=0's MTE3 write) ---
            T.wait_flag("mte3", "v", 1)
            T.tile.cast(offi, idx_ub, "CAST_RINT", N)
            T.tile.mul(offi, offi, 16)
            T.tile.add(offi, offi, 4)
            T.reinterpretcast(offv1, offi, "uint32_t")
            T.tile.gather(outw2[1, :N], tab, offv1, 0)
            T.set_flag("v", "mte3", 1)
            T.wait_flag("v", "mte3", 1)
            T.copy(outw2[1, 0:N], coords[1, 0:N])
            T.set_flag("mte3", "v", 1)
            # --- c = 2 (slot 0) ---
            T.wait_flag("mte3", "v", 0)
            T.tile.cast(offi, idx_ub, "CAST_RINT", N)
            T.tile.mul(offi, offi, 16)
            T.tile.add(offi, offi, 8)
            T.reinterpretcast(offv2, offi, "uint32_t")
            T.tile.gather(outw2[0, :N], tab, offv2, 0)
            T.set_flag("v", "mte3", 0)
            T.wait_flag("v", "mte3", 0)
            T.copy(outw2[0, 0:N], coords[2, 0:N])
            T.set_flag("mte3", "v", 0)
            # --- c = 3 (slot 1) ---
            T.wait_flag("mte3", "v", 1)
            T.tile.cast(offi, idx_ub, "CAST_RINT", N)
            T.tile.mul(offi, offi, 16)
            T.tile.add(offi, offi, 12)
            T.reinterpretcast(offv3, offi, "uint32_t")
            T.tile.gather(outw2[1, :N], tab, offv3, 0)
            T.set_flag("v", "mte3", 1)
            T.wait_flag("v", "mte3", 1)
            T.copy(outw2[1, 0:N], coords[3, 0:N])
            T.set_flag("mte3", "v", 1)
            # teardown: consume the last slot-return sets so the hardware
            # events leave the kernel CLEAN — a leftover set makes the next
            # kernel's waits pass immediately (sync lost, data race).
            T.wait_flag("mte3", "v", 0)
            T.wait_flag("mte3", "v", 1)

    return main


# ========== K3: IoU bitmap — scalar row x vector column ==========
_iou_cache = {}


@tilelang.jit(out_idx=[1], pass_configs=_PC_P2)
def _iou_bitmap_kernel(N, BM, BN, NCG, NRB, N_pad_rows, NW16, thr, dtype="float"):
    CMPB = BN // 8  # packed uint8 bytes per block row chunk
    CMP16 = BN // 16  # packed uint16 words per block row chunk

    @T.prim_func
    def main(
        coords: T.Tensor((4, N), dtype),
        bitmap: T.Tensor((N_pad_rows, NW16), "uint16"),
    ):
        with T.Kernel(NRB * NCG, is_npu=True, threads=1) as cid:
            bx = cid // NCG
            by = cid % NCG
            x1_j = T.alloc_ub((BN,), dtype)
            y1_j = T.alloc_ub((BN,), dtype)
            x2_j = T.alloc_ub((BN,), dtype)
            y2_j = T.alloc_ub((BN,), dtype)
            area_j = T.alloc_ub((BN,), dtype)
            x1_i = T.alloc_ub((BM,), dtype)
            y1_i = T.alloc_ub((BM,), dtype)
            x2_i = T.alloc_ub((BM,), dtype)
            y2_i = T.alloc_ub((BM,), dtype)
            area_i = T.alloc_ub((BM,), dtype)
            ti0 = T.alloc_ub((BM,), dtype)
            ti1 = T.alloc_ub((BM,), dtype)
            w = T.alloc_ub((BN,), dtype)
            h = T.alloc_ub((BN,), dtype)
            u = T.alloc_ub((BN,), dtype)
            tmp = T.alloc_ub((BN,), dtype)
            cmpb = T.alloc_ub((BM, CMPB), "uint8")
            cmp16 = T.alloc_ub((BM, CMP16), "uint16")

            # skip strictly-lower-triangle blocks (garbage is provably
            # harmless: greedy only consumes entries with j > k)
            if bx * BM < (by + 1) * BN:
                T.copy(coords[0, by * BN : by * BN + BN], x1_j, pad_value=0.0)
                T.copy(coords[1, by * BN : by * BN + BN], y1_j, pad_value=0.0)
                T.copy(coords[2, by * BN : by * BN + BN], x2_j, pad_value=0.0)
                T.copy(coords[3, by * BN : by * BN + BN], y2_j, pad_value=0.0)
                T.copy(coords[0, bx * BM : bx * BM + BM], x1_i, pad_value=0.0)
                T.copy(coords[1, bx * BM : bx * BM + BM], y1_i, pad_value=0.0)
                T.copy(coords[2, bx * BM : bx * BM + BM], x2_i, pad_value=0.0)
                T.copy(coords[3, bx * BM : bx * BM + BM], y2_i, pad_value=0.0)
                T.barrier_all()  # MTE2 -> V
                # areas, bit-exact order (x2-x1)*(y2-y1)
                T.tile.sub(w, x2_j, x1_j)
                T.tile.sub(h, y2_j, y1_j)
                T.tile.mul(area_j, w, h)
                T.tile.sub(ti0, x2_i, x1_i)
                T.tile.sub(ti1, y2_i, y1_i)
                T.tile.mul(area_i, ti0, ti1)
                T.barrier_all()  # V -> S (scalar reads of area_i[k] below)
                for k in T.serial(BM):
                    # 13 tensor-scalar ops, frozen order == golden formula:
                    # w = clamp0(min(x2i,x2j) - max(x1i,x1j)), h likewise,
                    # inter = w*h, union = ((ai+aj) - inter) + 1e-6,
                    # iou = inter/union  — NaN/inf propagate identically.
                    T.tile.min(w, x2_j, x2_i[k])
                    T.tile.max(u, x1_j, x1_i[k])
                    T.tile.sub(w, w, u)
                    T.tile.max(w, w, 0.0)
                    T.tile.min(h, y2_j, y2_i[k])
                    T.tile.max(u, y1_j, y1_i[k])
                    T.tile.sub(h, h, u)
                    T.tile.max(h, h, 0.0)
                    T.tile.mul(tmp, w, h)
                    T.tile.add(u, area_j, area_i[k])
                    T.tile.sub(u, u, tmp)
                    T.tile.add(u, u, 1e-6)
                    T.tile.div(tmp, tmp, u)
                    # pack keep-bits (iou <= thr) directly into the row region
                    T.tile.compare(cmpb[k, :CMPB], tmp, thr, "LE")
                T.barrier_all()  # V -> MTE3
                T.reinterpretcast(cmp16, cmpb, "uint16_t")
                T.copy(
                    cmp16,
                    bitmap[
                        bx * BM : bx * BM + BM,
                        by * CMP16 : by * CMP16 + CMP16,
                    ],
                )

    return main


# ========== K4: greedy — blocked bit-scan over the uint16 bitmap ==========
# NOTE (documented dead end): a "word-cached decisions" variant (one keepmask
# word read per 16 boxes, cached bits tested per box, cache updated
# arithmetically after each keep) is NOT expressible — the TVM-script parser
# silently drops loop-carried Python-variable rebinding inside T.serial /
# T.unroll bodies (verified with a minimal repro). Per-box UB word reads are
# therefore kept.
_greedy_cache = {}


@tilelang.jit(out_idx=[2], pass_configs=_PC_P2)
def _greedy_kernel(N, N_pad_rows, B, NB, NW16, dtype="float"):
    WPB = B // 16  # words per batch (B is a multiple of 16)

    @T.prim_func
    def main(
        bitmap: T.Tensor((N_pad_rows, NW16), "uint16"),
        idx: T.Tensor((N,), dtype),
        out: T.Tensor((N + 1,), "int64"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            keepmask = T.alloc_ub((NW16,), "uint16")
            chunk = T.alloc_ub((B, NW16), "uint16")
            kept = T.alloc_ub((N,), "int64")
            cnt = T.alloc_ub((1,), "int64")
            idx_ub = T.alloc_ub((N,), dtype)

            T.copy(idx[0:N], idx_ub[0:N])
            T.tile.fill(keepmask, -1)  # all keepable (uint16 Duplicate)
            cnt[0] = 0
            T.barrier_all()  # MTE2/V -> S (decisions read keepmask bits)
            for c in T.serial(NB):
                T.copy(bitmap[c * B : (c + 1) * B, 0:NW16], chunk)
                T.barrier_all()  # MTE2 -> V/S
                # word-major fast path: a zero keepmask word means all 16
                # boxes are already suppressed — bitwise_and only CLEARS
                # bits, so a zero word can never resurrect; skip its 16 bit
                # tests entirely. Inside a live word each box still re-reads
                # the LIVE keepmask bit (not the snapshot above) so in-word
                # suppression chains stay exact.
                for w in T.serial(WPB):
                    gw = c * WPB + w  # global word index (B % 16 == 0)
                    wword = T.cast(keepmask[gw], "int32")
                    if wword != 0:
                        for b in T.serial(16):
                            k = w * 16 + b  # batch-local row
                            kk = c * B + k  # global box index
                            if k < T.min(B, N - c * B):
                                # live bit re-read (bit == b: B%16==0)
                                bit = (T.cast(keepmask[gw], "int32") >> b) & 1
                                if bit == 1:
                                    # cross/in-batch keepmask AND (full row;
                                    # garbage bits only land on decided cols)
                                    T.tile.bitwise_and(keepmask, keepmask, chunk[k, :NW16])
                                    T.barrier_all()  # V -> S (next decision)
                                    kc = T.cast(cnt[0], "int32")
                                    kept[kc] = T.cast(idx_ub[kk], "int64")
                                    cnt[0] = cnt[0] + 1
                T.barrier_all()  # V -> MTE2 (next batch chunk DMA)
            T.barrier_all()  # S -> MTE3
            T.copy(kept[0:N], out[0:N])
            T.copy(cnt, out[N : N + 1])

    return main


# ========== Host Wrapper ==========
def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
    """Non-Maximum Suppression — 4 TileLang kernels, all-NPU, no aclnn ops.

    keep_indices (int64, matching the cann-bench operator spec) ordered by descending
    score. Input contract: boxes [N, 4] fp32 contiguous, scores [N] fp32
    contiguous, iou_threshold in (0, 1), N <= 4096 (greedy chunk UB budget).
    """
    if boxes.dim() != 2 or boxes.shape[1] != 4:
        raise ValueError(f"nms: boxes must be [N, 4], got shape {tuple(boxes.shape)}")
    if scores.dim() != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError(f"nms: scores must be [N] with N=boxes.shape[0], got shape {tuple(scores.shape)} vs boxes {tuple(boxes.shape)}")
    if boxes.dtype != torch.float32 or scores.dtype != torch.float32:
        raise TypeError(f"nms: boxes/scores must be float32, got boxes={boxes.dtype}, scores={scores.dtype}")
    if not boxes.is_contiguous() or not scores.is_contiguous():
        raise ValueError("nms: boxes/scores must be contiguous tensors")
    if not isinstance(iou_threshold, (int, float)):
        raise ValueError(f"nms: iou_threshold must be a Python number, got {type(iou_threshold).__name__}")
    if not (0.0 < iou_threshold < 1.0):
        raise ValueError(f"nms: iou_threshold must be in the open interval (0, 1), got {iou_threshold}")

    N = boxes.shape[0]
    if N == 0:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)
    if N > _MAX_N:
        raise ValueError(f"nms: N={N} exceeds supported maximum {_MAX_N} (greedy bitmap chunk would exceed the 192KB UB budget)")

    tp = _tiling(N)

    key = (N,)
    if key not in _sort_cache:
        _sort_cache[key] = _sort_kernel(N, tp["N_aligned"])
    sort_idx = _sort_cache[key](scores)

    if key not in _gather_cache:
        _gather_cache[key] = _gather_kernel(N)
    coords = _gather_cache[key](boxes, sort_idx)

    key3 = (N, iou_threshold)
    if key3 not in _iou_cache:
        _iou_cache[key3] = _iou_bitmap_kernel(N, tp["BM"], tp["BN"], tp["NCG"], tp["NRB"], tp["N_pad_rows"], tp["NW16"], iou_threshold)
    bitmap = _iou_cache[key3](coords)

    if key not in _greedy_cache:
        _greedy_cache[key] = _greedy_kernel(N, tp["N_pad_rows"], tp["B"], tp["NB"], tp["NW16"])
    out = _greedy_cache[key](bitmap, sort_idx)

    M = int(out[N].item())  # 8B scalar D2H (whitelisted)
    out.resize_(M)  # metadata shrink only
    return out


# ----------------------------------------------------------------------------
# CI smoke test (examples/bench_test.sh runs this file directly and requires
# "Kernel Output Match!" or "TEST PASSED!" in the output, same as the
# lstm/gru/depthwise_conv_2d smoke tests).  The full layered suite
# (L0/L1/L2/Boundary over the 20 cann-bench cases) lives in test_nms.py.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    def _smoke_golden(boxes, scores, thr):
        """Pure-torch greedy NMS with the kernel's frozen IoU formula."""
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        x1 = torch.maximum(boxes[:, None, 0], boxes[None, :, 0])
        y1 = torch.maximum(boxes[:, None, 1], boxes[None, :, 1])
        x2 = torch.minimum(boxes[:, None, 2], boxes[None, :, 2])
        y2 = torch.minimum(boxes[:, None, 3], boxes[None, :, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        iou = inter / (areas[:, None] + areas[None, :] - inter + 1e-6)
        _, order = scores.sort(descending=True)
        suppressed = torch.zeros(scores.shape[0], dtype=torch.bool)
        keep = []
        for idx in order.tolist():
            if suppressed[idx]:
                continue
            keep.append(idx)
            suppressed |= ~(iou[idx] <= thr)
        return keep

    for n, thr in ((1024, 0.5), (4096, 0.3)):
        boxes = torch.empty(n, 4)
        boxes[:, :2].uniform_(-1.0, 1.0)
        boxes[:, 2:] = boxes[:, :2] + torch.rand(n, 2)  # legal: x2 > x1, y2 > y1
        # k/n scores are exactly distinct in fp32 (k * 2**-12), so the greedy
        # order is deterministic without tied-score regularization.
        scores = (torch.randperm(n).double() / n).float()
        out = nms(boxes.npu(), scores.npu(), thr).cpu().long().tolist()
        ref = _smoke_golden(boxes, scores, thr)
        assert out == ref, f"smoke failed: N={n} kept {len(out)} vs golden {len(ref)}"
        print(f"[SMOKE] N={n} thr={thr} kept={len(out)}/{n} exact match")
    print("Kernel Output Match!")
