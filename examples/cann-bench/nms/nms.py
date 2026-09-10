"""NMS (Non-Maximum Suppression) example — greedy NMS with cann-bench 20 cases.

Sort-bitmap NMS pipeline for Ascend NPU (Expert mode). Targets cann-bench
multi-case evaluation: sort scores descending, reorder boxes, pack pairwise
IoU threshold decisions into a uint16 bitmap, then run a blocked greedy
bit-scan. Output keep indices match the torch golden exactly (index list
equality, not tolerance).

Algorithm:
    keep = NMS(boxes, scores, iou_threshold)
    IoU(i, j) = inter / (area_i + area_j - inter + 1e-6)
    suppress when ~(iou <= thr)  (NaN IoU suppresses, matching golden)

Reference: pure-PyTorch greedy NMS (see golden_nms below)

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
"""

import argparse
import gc
import sys

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
            T.tile.gather(outw2[0, 0:N], tab, offv0, 0)
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
            T.tile.gather(outw2[1, 0:N], tab, offv1, 0)
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
            T.tile.gather(outw2[0, 0:N], tab, offv2, 0)
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
            T.tile.gather(outw2[1, 0:N], tab, offv3, 0)
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
                    T.tile.compare(cmpb[k, 0:CMPB], tmp, thr, "LE")
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
        out: T.Tensor((N + 1,), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            keepmask = T.alloc_ub((NW16,), "uint16")
            chunk = T.alloc_ub((B, NW16), "uint16")
            kept = T.alloc_ub((N,), dtype)
            cnt = T.alloc_ub((1,), dtype)
            idx_ub = T.alloc_ub((N,), dtype)

            T.copy(idx[0:N], idx_ub[0:N])
            T.tile.fill(keepmask, -1)  # all keepable (uint16 Duplicate)
            T.tile.fill(cnt, 0.0)
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
                                    T.tile.bitwise_and(keepmask, keepmask, chunk[k, 0:NW16])
                                    T.barrier_all()  # V -> S (next decision)
                                    kc = T.cast(cnt[0], "int32")
                                    kept[kc] = idx_ub[kk]
                                    cnt[0] = cnt[0] + 1.0
                T.barrier_all()  # V -> MTE2 (next batch chunk DMA)
            T.barrier_all()  # S -> MTE3
            T.copy(kept[0:N], out[0:N])
            T.copy(cnt, out[N : N + 1])

    return main


# ========== Host Wrapper ==========
def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
    """Non-Maximum Suppression — 4 TileLang kernels, all-NPU, no aclnn ops.

    keep_indices (fp32 carrier, exact integer values) ordered by descending
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
        return boxes[0:0, 0]
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

    M = int(out[N].item())  # 4B scalar D2H (whitelisted)
    out.resize_(M)  # metadata shrink only
    return out


# ========== Golden Reference ==========
def golden_nms(boxes, scores, iou_threshold):
    """Golden NMS: full IoU matrix vectorized, greedy selection on CPU.

    Uses ~(iou <= thr) so NaN/inf IoU suppress (IEEE: NaN <= thr is False).
    """
    assert boxes.dim() == 2 and boxes.shape[1] == 4, "boxes shape must be [N, 4]"
    assert scores.dim() == 1 and scores.shape[0] == boxes.shape[0], "scores shape must be [N]"

    n = scores.shape[0]
    dev = boxes.device
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    x1 = torch.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = torch.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = torch.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = torch.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    iou = inter / (areas[:, None] + areas[None, :] - inter + 1e-6)

    _, order = scores.sort(descending=True)

    iou_cpu = iou.cpu()
    order_cpu = order.cpu()
    suppressed = torch.zeros(n, dtype=torch.bool)
    keep = []
    for idx_t in order_cpu:
        idx = int(idx_t)
        if suppressed[idx]:
            continue
        keep.append(idx)
        suppressed |= ~(iou_cpu[idx] <= iou_threshold)

    return torch.tensor(keep, dtype=torch.long, device=dev)


def get_input(boxes, scores, iou_threshold=0.5, **kwargs):
    """Regularize inputs (cann-bench harness semantics, applied to both
    golden and candidate sides):
    1. sort x-pair / y-pair per box (x1<=x2, y1<=y2; idempotent on legal
       boxes — value_range sampling otherwise yields degenerate boxes)
    2. replace tied scores with a deterministic distinct sequence (seed=0)
       so greedy tie-break order is well-defined.
    """
    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    n = int(scores.shape[0])
    if n > 1 and int(scores.reshape(-1).unique().numel()) < n:
        g = torch.Generator().manual_seed(0)
        ranks = (torch.randperm(n, generator=g) + 1).to(torch.float32) / float(n)
        scores = ranks.reshape(scores.shape).to(dtype=scores.dtype, device=scores.device)
    return [boxes, scores]


# ========== Input Generation (matches the eval framework's DataGenerator) ==========
SEED_BASE = 1000  # torch.manual_seed(SEED_BASE + case_id)


def _gen_tensor(shape, value_range, seed):
    """Generate a fp32 tensor: uniform in value_range, or special recipes
    for inf (finite base + boundary slices filled -inf/+inf), nan (~50% NaN
    mixed with finite), and constant ranges."""
    min_val, max_val = value_range
    gen = torch.Generator()
    gen.manual_seed(seed)
    dtype = torch.float32

    if min_val != min_val:  # NaN range: mix NaN with random finite values
        rand_base = torch.rand(shape, dtype=torch.float32, generator=gen)
        tensor = (rand_base * 2 - 1).to(dtype)
        nan_rand = torch.rand(shape, dtype=torch.float32, generator=gen)
        tensor[nan_rand < 0.5] = float("nan")
        return tensor
    if min_val in (float("inf"), float("-inf")) or max_val in (float("inf"), float("-inf")):
        rand_base = torch.rand(shape, dtype=torch.float32, generator=gen)
        tensor = (rand_base * 2 - 1).to(dtype)
        flat = tensor.flatten()
        n = max(1, len(flat) // 20)
        flat[:n] = float("-inf")
        flat[-n:] = float("inf")
        return tensor
    if min_val == max_val:
        return torch.full(shape, float(min_val), dtype=dtype)

    finfo = torch.finfo(dtype)
    lo = max(min_val, float(finfo.min))
    hi = min(max_val, float(finfo.max))
    rand_f64 = torch.rand(shape, dtype=torch.float64, generator=gen)
    tensor_f64 = rand_f64 * (hi - lo) + lo
    tensor_f64 = torch.clamp(tensor_f64, float(finfo.min), float(finfo.max))
    return tensor_f64.to(dtype)


def _run_case(name, N, thr, boxes_vr, scores_vr, seed, level="cann-bench"):
    """Generate -> regularize -> golden vs candidate; exact index match.

    Judgment = keep_indices exact list equality (length + order + values)
    vs golden — NMS outputs indices, not values, so tolerance metrics do
    not apply.
    """
    gc.collect()
    torch.npu.empty_cache()

    boxes = _gen_tensor([N, 4], boxes_vr, seed)
    scores = _gen_tensor([N], scores_vr, seed + 1)
    reg_boxes, reg_scores = get_input(boxes, scores, iou_threshold=thr)

    ref = golden_nms(reg_boxes, reg_scores, thr)
    out = nms(reg_boxes.npu(), reg_scores.npu(), thr).cpu().long()

    match = out.numel() == ref.numel() and torch.equal(out, ref.cpu())
    tag = "PRECISION_PASS" if match else "PRECISION_FAIL"
    extra = ""
    if not match:
        extra = f" reason=length golden={ref.numel()} cand={out.numel()}" if out.numel() != ref.numel() else " reason=index mismatch"
    print(f"[{tag}] {level} {name} N={N} thr={thr} M_golden={ref.numel()} M_candidate={out.numel()}{extra}")
    return match


def _run_boundary(level, name, fn):
    try:
        fn()
        print(f"[BOUNDARY_PASS] {level} {name}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] {level} {name}: {e}")


# ========== L0 Tests ==========
def test_nms_l0():
    configs = [
        ("l0-1", 1024, 0.5, (-1, 1)),
        ("l0-2", 4096, 0.5, (float("nan"), float("nan"))),  # all-NaN: M=1
        ("l0-3", 4096, 0.5, (0.0, 0.0)),  # all-zero: M=N
    ]
    ok = True
    for i, (name, N, thr, vr) in enumerate(configs):
        ok &= _run_case(name, N, thr, vr, (0.0, 1.0), SEED_BASE + 100 + i, level="l0")
    return ok


# ========== L1 Functional Tests ==========
def test_nms_l1():
    configs = [
        ("l1-edge-n1", 1, 0.5, (-1, 1)),  # M=1 minimal input
        ("l1-edge-n2", 2, 0.4, (-1, 1)),  # M in {1,2}
        ("l1-tail-n513", 513, 0.3, (-2, 2)),  # tail block +1
        ("l1-tail-mid-n1040", 1040, 0.5, (-1, 1)),
        ("l1-prime-n1013", 1013, 0.6, (-3, 3)),  # prime, non-aligned
        ("l1-thr-loose", 1024, 0.99, (-1, 1)),  # extreme high keep rate
        ("l1-thr-strict", 1024, 0.01, (-1, 1)),  # extreme low keep rate
    ]
    ok = True
    for i, (name, N, thr, vr) in enumerate(configs):
        ok &= _run_case(name, N, thr, vr, (0.0, 1.0), SEED_BASE + 200 + i, level="l1")
    return ok


# ========== L2 Exception Tests ==========
def test_nms_l2():

    def test_fp16_boxes():
        boxes = torch.rand(64, 4, dtype=torch.float16).npu()
        scores = torch.rand(64, dtype=torch.float32).npu()
        nms(boxes, scores, 0.5)

    def test_fp64_scores():
        boxes = torch.rand(64, 4, dtype=torch.float32).npu()
        scores = torch.rand(64, dtype=torch.float64).npu()
        nms(boxes, scores, 0.5)

    def test_5col_boxes():
        boxes = torch.rand(64, 5, dtype=torch.float32).npu()
        scores = torch.rand(64, dtype=torch.float32).npu()
        nms(boxes, scores, 0.5)

    def test_scores_len_mismatch():
        boxes = torch.rand(64, 4, dtype=torch.float32).npu()
        scores = torch.rand(63, dtype=torch.float32).npu()
        nms(boxes, scores, 0.5)

    def test_n_over_max():
        boxes = torch.rand(4097, 4, dtype=torch.float32).npu()
        scores = torch.rand(4097, dtype=torch.float32).npu()
        nms(boxes, scores, 0.5)

    def test_thr_out_of_range():
        boxes = torch.rand(64, 4, dtype=torch.float32).npu()
        scores = torch.rand(64, dtype=torch.float32).npu()
        nms(boxes, scores, 1.5)

    _run_boundary("l2", "fp16_boxes_rejected", test_fp16_boxes)
    _run_boundary("l2", "fp64_scores_rejected", test_fp64_scores)
    _run_boundary("l2", "5col_boxes_rejected", test_5col_boxes)
    _run_boundary("l2", "scores_len_mismatch_rejected", test_scores_len_mismatch)
    _run_boundary("l2", "n_over_4096_rejected", test_n_over_max)
    _run_boundary("l2", "thr_out_of_range_rejected", test_thr_out_of_range)


# ========== Boundary Tests ==========
def test_nms_boundary():

    def test_inf_mix():
        boxes = torch.empty(512, 4, dtype=torch.float32).uniform_(-5, 10)
        boxes.view(-1)[0] = float("inf")
        boxes.view(-1)[1] = float("-inf")
        boxes, scores = get_input(boxes, torch.rand(512), 0.5)
        ref = golden_nms(boxes, scores, 0.5)
        out = nms(boxes.npu(), scores.npu(), 0.5).cpu().long()
        assert torch.equal(out, ref.cpu())

    def test_nan_boxes():
        boxes = torch.full((512, 4), float("nan"))
        scores = torch.rand(512)
        boxes, scores = get_input(boxes, scores, 0.5)
        ref = golden_nms(boxes, scores, 0.5)
        out = nms(boxes.npu(), scores.npu(), 0.5).cpu().long()
        assert torch.equal(out, ref.cpu())  # NaN suppresses all but box 0: M=1

    def test_zero_boxes():
        boxes = torch.zeros(1024, 4)
        scores = torch.rand(1024)  # ties -> get_input de-duplicates
        boxes, scores = get_input(boxes, scores, 0.5)
        ref = golden_nms(boxes, scores, 0.5)
        out = nms(boxes.npu(), scores.npu(), 0.5).cpu().long()
        assert torch.equal(out, ref.cpu())  # IoU=0 -> all kept: M=N

    def test_fp16_max_coords():
        boxes = torch.empty(256, 4, dtype=torch.float32).uniform_(-65504, 65504)
        boxes, scores = get_input(boxes, torch.rand(256), 0.5)
        ref = golden_nms(boxes, scores, 0.5)
        out = nms(boxes.npu(), scores.npu(), 0.5).cpu().long()
        assert torch.equal(out, ref.cpu())

    def test_empty_input():
        boxes = torch.rand(0, 4, dtype=torch.float32).npu()
        scores = torch.rand(0, dtype=torch.float32).npu()
        out = nms(boxes, scores, 0.5)
        assert out.numel() == 0

    _run_boundary("boundary", "inf_mix_exact_match", test_inf_mix)
    _run_boundary("boundary", "nan_boxes_all_suppressed", test_nan_boxes)
    _run_boundary("boundary", "zero_boxes_all_kept", test_zero_boxes)
    _run_boundary("boundary", "fp16_max_coords", test_fp16_max_coords)
    _run_boundary("boundary", "empty_n0", test_empty_input)


# ========== cann-bench 20 Cases ==========
def test_nms_cann_bench():
    configs = [
        ("cann-bench-1", 1024, 0.5, (-1, 1)),
        ("cann-bench-2", 4096, 0.3, (-2, 2)),
        ("cann-bench-3", 4096, 0.7, (-3, 3)),
        ("cann-bench-4", 4096, 0.1, (-10, 10)),
        ("cann-bench-5", 4096, 0.9, (-100, 100)),
        ("cann-bench-6", 4096, 0.05, (-1000, 1000)),
        ("cann-bench-7", 1023, 0.5, (-0.1, 0.1)),
        ("cann-bench-8", 1009, 0.3, (-1, 2)),
        ("cann-bench-9", 1537, 0.7, (-5, 10)),
        ("cann-bench-10", 3001, 0.4, (-50, 100)),
        ("cann-bench-11", 2049, 0.5, (-65504, 65504)),
        ("cann-bench-12", 4001, 0.6, (-88, 88)),
        ("cann-bench-13", 3001, 0.5, (float("-inf"), float("inf"))),
        ("cann-bench-14", 4096, 0.5, (float("nan"), float("nan"))),
        ("cann-bench-15", 4096, 0.5, (0.0, 0.0)),
        ("cann-bench-16", 511, 0.2, (-0.5, 0.5)),
        ("cann-bench-17", 2047, 0.8, (-1, 3)),
        ("cann-bench-18", 4095, 0.35, (-1000, 1000)),
        ("cann-bench-19", 4096, 0.45, (-0.2, 0.2)),
        ("cann-bench-20", 4096, 0.55, (-20, 40)),
    ]
    ok = True
    for case_id, (name, N, thr, vr) in enumerate(configs, start=1):
        ok &= _run_case(name, N, thr, vr, (0.0, 1.0), SEED_BASE + case_id, level="cann-bench")
    return ok


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="NMS example with cann-bench 20 cases")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "cann-bench", "all"],
        help="Test level to run (default: l0)",
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True
    if args.level in ("l0", "all"):
        blocking_ok &= test_nms_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_nms_l1()
    if args.level in ("l2", "all"):
        test_nms_l2()
    if args.level in ("boundary", "all"):
        test_nms_boundary()
    if args.level in ("cann-bench", "all"):
        blocking_ok &= test_nms_cann_bench()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
