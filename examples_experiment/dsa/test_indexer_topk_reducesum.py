"""Test for the indexer_topk_reducesum example (aligned with tile-ai upstream).

Aligned with the upstream test
(examples/dsa_sparse_finetune/indexer_topk_reducesum.py in tile-ai/tilelang),
with the minimal Ascend adaptations:
- device placement adapted to Ascend NPU (.cuda() -> .npu())
- einops einsum(q, k, "s1 h k, s2 k -> s1 h s2") -> torch.einsum("ahk,bk->ahb")
  (torch requires single-letter axes; a=s1, b=s2, same contraction)
- get_abs_err / get_err_ratio are inlined from the upstream examples/utils.py
- the upstream per-token prints (2048 rows of intersection ratio + err/ratio)
  are aggregated into summary statistics (global abs err / err ratio, total
  and min per-token index-intersection ratio)
- the TopkIndices check gates on the aggregated set-intersection ratio over
  the ref_score > 0 domain (>= 0.99): near-tie rank swaps between adjacent
  topk positions are absorbed by the set-intersection metric
"""

import math
import sys

import pytest
import torch
import torch.nn.functional as F

from indexer_topk_reducesum import (
    _fast_gate_ok,
    _generic_gate_ok,
    indexer_topk_reducesum_interface,
)

ERR_RATIO_MAX = 1e-2
ABS_ERR_MAX = 1e-3
IDX_MATCH_RATIO_MIN = 0.99


def get_abs_err(y, x):
    # Inlined from tile-ai/tilelang examples/utils.py (get_abs_err).
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    return (x - y).flatten().abs().max().item()


def get_err_ratio(y, x):
    # Inlined from tile-ai/tilelang examples/utils.py (get_err_ratio).
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    err = (x - y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def check_tolerance(abs_err, err_ratio, idx_match_ratio, scores_finite=True):
    """NaN-safe golden tolerance gate.

    A NaN in the kernel output degrades both error metrics to NaN and
    ``NaN > threshold`` is False, so a bare threshold compare would print
    PASS whenever the index sets happen to match — non-finite scores (or
    metrics) fail explicitly.
    """
    if not scores_finite:
        return False
    if not (math.isfinite(abs_err) and math.isfinite(err_ratio)):
        return False
    return err_ratio <= ERR_RATIO_MAX and abs_err <= ABS_ERR_MAX and idx_match_ratio >= IDX_MATCH_RATIO_MIN


def ref_index_score(Q: torch.Tensor, Weights: torch.Tensor, K: torch.Tensor, topk: int, offsets: torch.Tensor):
    # q: [seq_len, heads, dim]; k: [seq_len, dim]; varlen segments per offsets
    all_topk_indices = []
    all_topk_score = []
    for i in range(offsets.shape[0] - 1):
        assert (offsets[i + 1] - offsets[i]).item() >= topk
        q = Q[offsets[i] : offsets[i + 1]]
        weights = Weights[offsets[i] : offsets[i + 1]]
        k = K[offsets[i] : offsets[i + 1]]
        softmax_scale = q.shape[-1] ** -0.5
        s = q.shape[0]
        mask = (torch.arange(s)[:, None] >= torch.arange(s)[None, :]).to(q.device)
        logits = torch.einsum("ahk,bk->ahb", q, k)
        logits = F.relu(logits)
        logits = (logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32) * softmax_scale
        logits = torch.where(mask, logits, float("-inf"))
        topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)
        topk_score = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
        all_topk_indices.append(topk_indices)
        all_topk_score.append(topk_score)
    topk_indices = torch.cat(all_topk_indices, dim=0)
    topk_score = torch.cat(all_topk_score, dim=0)
    return topk_indices, topk_score


def run_golden_kernel(
    B=1,
    S=2048,
    H=64,
    D=128,
    topk=64,
):
    torch.manual_seed(42)

    q = torch.randn((S, H, D)).npu().bfloat16()
    weights = torch.randn((S, H)).npu().bfloat16()
    k = torch.randn((S, D)).npu().bfloat16()
    offsets = torch.tensor([0, S], dtype=torch.int32).npu()

    ref_topk_indices, ref_topk_score = ref_index_score(q, weights, k, topk, offsets)

    topk_indices, topk_score = indexer_topk_reducesum_interface(q, weights, k, topk, offsets)

    # finite-value gate: NaN/inf in the kernel output must fail outright —
    # both error metrics degrade to NaN and NaN > threshold is False, so a
    # bare threshold compare would pass on index match alone
    scores_finite = bool(torch.isfinite(topk_score).all().item())

    # ReduceSum (fp32 softmax scores): golden tolerance check
    abs_err = get_abs_err(ref_topk_score, topk_score)
    err_ratio = get_err_ratio(ref_topk_score, topk_score)
    print(f"ReduceSum err: {abs_err:.6f} ratio: {err_ratio:.6f} finite: {scores_finite}")

    # TopkIndices: set-intersection match over the ref_score > 0 domain,
    # aggregated over all tokens (near-tie rank swaps are absorbed)
    total_ref = 0
    total_matched = 0
    min_ratio = 1.0
    for j in range(S):
        mask = ref_topk_score[j] > 0
        set_ref = set(ref_topk_indices[j][mask].tolist())
        set_trt = set(topk_indices[j][mask].tolist())
        matched = len(set_ref & set_trt)
        total_ref += len(set_ref)
        total_matched += matched
        if len(set_ref) > 0:
            min_ratio = min(min_ratio, matched / len(set_ref))
    idx_match_ratio = total_matched / total_ref if total_ref else 1.0
    print(f"TopkIndices intersection: {total_matched}/{total_ref} = {idx_match_ratio:.6f} (min per-token {min_ratio:.6f})")

    if not check_tolerance(abs_err, err_ratio, idx_match_ratio, scores_finite):
        print(
            f"FAIL: exceeds tolerance (err_ratio <= {ERR_RATIO_MAX:g}, abs_err <= {ABS_ERR_MAX:g}, "
            f"idx_match_ratio >= {IDX_MATCH_RATIO_MIN:g}, finite output required)"
        )
        return False
    print(
        f"PASS: within tolerance (err_ratio <= {ERR_RATIO_MAX:g}, abs_err <= {ABS_ERR_MAX:g}, "
        f"idx_match_ratio >= {IDX_MATCH_RATIO_MIN:g}, finite output required)"
    )
    return True


def test_indexer_topk_reducesum_accuracy() -> None:
    """Pytest entry: gate on the golden tolerance check.

    run_golden_kernel returns a bool (kept for script mode); pytest ignores
    return values, so the gate must be an assert.
    """
    assert run_golden_kernel(), "output exceeds tolerance (see printed err/ratio)"


def test_l1_budget_gate_model() -> None:
    """Host-side L1 budget model (no kernel build).

    heads=64 x dim=512 needs 544KiB of Q/K L1 staging at QG=8 — over the
    524032B L1 pool — so both paths must demote the group tier; dim=1024
    leaves no fast tier, and generic's QG=2 rescue tier fits.
    """
    # reference shape: the 8-token tier stays eligible
    assert _fast_gate_ok(64, 128, 64, 2048, 8)
    assert _generic_gate_ok(64, 128, 64, 1024, 8)
    # heads=64 x dim=512: QG=8 L1 staging = 557184B > 524032B; QG=4 fits
    assert not _fast_gate_ok(64, 512, 64, 2048, 8)
    assert _fast_gate_ok(64, 512, 64, 2048, 4)
    assert not _generic_gate_ok(64, 512, 64, 1024, 8)
    assert _generic_gate_ok(64, 512, 64, 1024, 4)
    # heads=64 x dim=1024: no fast tier fits L1; generic rescues at QG=2
    assert not _fast_gate_ok(64, 1024, 64, 2048, 4)
    assert not _generic_gate_ok(64, 1024, 64, 1024, 4)
    assert _generic_gate_ok(64, 1024, 64, 1024, 2)


def test_indexer_topk_reducesum_l1_dim512_fast() -> None:
    """heads=64 x dim=512 (S=2048): the 8-token group tier overflows the L1
    pool (544KiB); the host demotes to QG=4 and the fast path still meets
    the golden tolerance."""
    assert run_golden_kernel(S=2048, H=64, D=512, topk=64), "output exceeds tolerance on the L1-demoted fast path (QG=4)"


def test_indexer_topk_reducesum_l1_dim512_generic() -> None:
    """heads=64 x dim=512 on the generic path (S=4097 > 4096): the same L1
    demotion to QG=4 applies."""
    assert run_golden_kernel(S=4097, H=64, D=512, topk=64), "output exceeds tolerance on the L1-demoted generic path (QG=4)"


def test_fast_path_rejects_l1_infeasible_dim() -> None:
    """heads=64 x dim=1024 leaves no fast QG tier inside the L1 pool: a
    clear ValueError replaces the silent L1 overflow (the raise happens
    before any kernel build)."""
    torch.manual_seed(42)
    S, H, D, topk = 2048, 64, 1024, 64
    q = torch.randn((S, H, D)).npu().bfloat16()
    weights = torch.randn((S, H)).npu().bfloat16()
    k = torch.randn((S, D)).npu().bfloat16()
    offsets = torch.tensor([0, S], dtype=torch.int32).npu()
    with pytest.raises(ValueError, match="L1"):
        indexer_topk_reducesum_interface(q, weights, k, topk, offsets, dispatch="fast")


def test_tolerance_gate_rejects_nan() -> None:
    """NaN outputs must fail the tolerance gate: the error metrics degrade
    to NaN and NaN > threshold is False, so a bare threshold compare would
    print PASS whenever the index sets happen to match."""
    # the review repro: NaN metrics + full index match -> must FAIL
    assert not check_tolerance(float("nan"), float("nan"), 1.0)
    # NaN/inf inside the output tensor itself -> must FAIL
    assert not check_tolerance(1e-6, 1e-7, 1.0, scores_finite=False)
    assert not check_tolerance(float("inf"), float("inf"), 1.0)
    # sanity: a genuinely passing configuration still passes
    assert check_tolerance(1e-6, 1e-7, 1.0)
    # genuine threshold violations still fail
    assert not check_tolerance(1e-2, 1e-7, 1.0)  # abs_err over
    assert not check_tolerance(1e-6, 2e-2, 1.0)  # err_ratio over
    assert not check_tolerance(1e-6, 1e-7, 0.9)  # idx_match_ratio under


def test_rejects_non_contiguous_inputs() -> None:
    """Non-contiguous q must be rejected up front: the cached launch path
    marshals raw pointers after a zero-copy view spec, so an implicit
    reshape copy could not be retained across the (cached) launch — its
    storage may be reused before the kernel reads it."""
    torch.manual_seed(42)
    S, H, D, topk = 256, 64, 128, 64
    q = torch.randn((H, S, D)).npu().bfloat16().transpose(0, 1)  # [S, H, D] view
    assert not q.is_contiguous()
    weights = torch.randn((S, H)).npu().bfloat16()
    k = torch.randn((S, D)).npu().bfloat16()
    offsets = torch.tensor([0, S], dtype=torch.int32).npu()
    with pytest.raises(ValueError, match="contiguous"):
        indexer_topk_reducesum_interface(q, weights, k, topk, offsets)
    # the explicit .contiguous() materialization is accepted
    topk_indices, topk_score = indexer_topk_reducesum_interface(q.contiguous(), weights, k, topk, offsets)
    assert topk_indices.shape == (S, topk) and topk_score.shape == (S, topk)


if __name__ == "__main__":
    success = run_golden_kernel()
    print("Test Passed!" if success else "Test Failed!")
    sys.exit(0 if success else 1)
