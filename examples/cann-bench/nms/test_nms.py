# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms of the CANN Open Software License Agreement Version 2.0 (the
# "License"). You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED.
# ----------------------------------------------------------------------------

"""Layered test suite for nms.

Levels are built from the 20 cann-bench nms cases (deterministic inputs
keyed by case id), redistributed by case nature (dispatchable via --level;
all levels also run under pytest where a failure raises AssertionError):

  L0        gate: baseline / unaligned tail N / all-zero boxes (M=N)
  L1        functional: thr sweep, value-range sweep, prime and mid-size N
  L2        negative: invalid inputs must be rejected by the wrapper
  Boundary  special values: +/-65504 extremes, +/-inf mix, NaN mix, all-zero
            boxes, plus an empty-input (N=0) check

Judgement: keep-index list exact equality (length + order + values) against
the torch golden — NMS outputs indices, not values, so tolerance metrics do
not apply.  Inputs are generated deterministically (torch generator seeded
with SEED_BASE + case_id) and regularized by get_input (per-box x/y pair
sorting + tied-score de-duplication) on both the golden and candidate sides,
mirroring the cann-bench harness semantics.
"""

import argparse
import gc
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import tilelang  # noqa: E402
from nms import nms  # noqa: E402


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


def _run_case(name, N, thr, boxes_vr, scores_vr, seed, level="l0"):
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


def _run_cb(case_id, N, thr, vr, level):
    """Run one cann-bench case with its original id-keyed seed."""
    return _run_case(f"cb{case_id}", N, thr, vr, (0.0, 1.0), SEED_BASE + case_id, level=level)


# ========== cann-bench case tables (20 cases, redistributed by nature) ==========
# (cann-bench case id, N, iou_threshold, boxes value range)
L0_CASES = [
    (1, 1024, 0.5, (-1, 1)),  # baseline
    (7, 1023, 0.5, (-0.1, 0.1)),  # unaligned tail N
    (15, 4096, 0.5, (0.0, 0.0)),  # all-zero boxes -> IoU 0 -> all kept (M=N)
]

L1_CASES = [
    (2, 4096, 0.3, (-2, 2)),
    (3, 4096, 0.7, (-3, 3)),
    (4, 4096, 0.1, (-10, 10)),
    (5, 4096, 0.9, (-100, 100)),
    (6, 4096, 0.05, (-1000, 1000)),
    (8, 1009, 0.3, (-1, 2)),  # prime N
    (9, 1537, 0.7, (-5, 10)),
    (10, 3001, 0.4, (-50, 100)),
    (12, 4001, 0.6, (-88, 88)),
    (16, 511, 0.2, (-0.5, 0.5)),
    (17, 2047, 0.8, (-1, 3)),
    (18, 4095, 0.35, (-1000, 1000)),
    (19, 4096, 0.45, (-0.2, 0.2)),
    (20, 4096, 0.55, (-20, 40)),
]

BOUNDARY_CASES = [
    (11, 2049, 0.5, (-65504, 65504)),  # fp16 dynamic-range bounds
    (13, 3001, 0.5, (float("-inf"), float("inf"))),  # boundary slices +/-inf
    (14, 4096, 0.5, (float("nan"), float("nan"))),  # ~50% NaN mix
]


# ========== L0 (gate) ==========
def test_nms_l0():
    failures = [cid for cid, N, thr, vr in L0_CASES if not _run_cb(cid, N, thr, vr, "l0")]
    assert not failures, f"L0 gate failed for cann-bench cases: {failures}"


# ========== L1 (functional) ==========
def test_nms_l1():
    failures = [cid for cid, N, thr, vr in L1_CASES if not _run_cb(cid, N, thr, vr, "l1")]
    assert not failures, f"L1 functional failed for cann-bench cases: {failures}"


# ========== L2 (negative: invalid inputs must be rejected) ==========
def test_nms_l2():
    def _expect_reject(desc, fn):
        try:
            fn()
        except Exception:  # noqa: BLE001
            print(f"[L2_PASS] {desc}: correctly rejected")
            return True
        print(f"[L2_FAIL] {desc}: invalid input silently accepted")
        return False

    cases = [
        ("fp16_boxes", lambda: nms(torch.rand(64, 4, dtype=torch.float16).npu(), torch.rand(64).npu(), 0.5)),
        ("fp64_scores", lambda: nms(torch.rand(64, 4).npu(), torch.rand(64, dtype=torch.float64).npu(), 0.5)),
        ("5col_boxes", lambda: nms(torch.rand(64, 5).npu(), torch.rand(64).npu(), 0.5)),
        ("scores_len_mismatch", lambda: nms(torch.rand(64, 4).npu(), torch.rand(63).npu(), 0.5)),
        ("n_over_4096", lambda: nms(torch.rand(4097, 4).npu(), torch.rand(4097).npu(), 0.5)),
        ("thr_out_of_range", lambda: nms(torch.rand(64, 4).npu(), torch.rand(64).npu(), 1.5)),
    ]
    failures = [desc for desc, fn in cases if not _expect_reject(desc, fn)]
    assert not failures, f"L2 negative tests failed for: {failures}"


# ========== Boundary (special values) ==========
def test_nms_boundary():
    failures = [cid for cid, N, thr, vr in BOUNDARY_CASES if not _run_cb(cid, N, thr, vr, "boundary")]

    out = nms(torch.rand(0, 4).npu(), torch.rand(0).npu(), 0.5)
    if out.numel() != 0:
        print(f"[BOUNDARY_FAIL] empty_n0: expected 0 elements, got {out.numel()}")
        failures.append("empty_n0")
    else:
        print("[BOUNDARY_PASS] boundary empty_n0: empty output")

    assert not failures, f"boundary tests failed for: {failures}"


def main():
    parser = argparse.ArgumentParser(description="NMS layered test suite (cann-bench cases)")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all"],
        help="Test level to run (default: l0)",
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    if args.level in ("l0", "all"):
        test_nms_l0()
    if args.level in ("l1", "all"):
        test_nms_l1()
    if args.level in ("l2", "all"):
        test_nms_l2()
    if args.level in ("boundary", "all"):
        test_nms_boundary()

    print("Test Passed!")


if __name__ == "__main__":
    main()
