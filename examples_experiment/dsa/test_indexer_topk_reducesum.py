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

import sys

import torch
import torch.nn.functional as F

from indexer_topk_reducesum import indexer_topk_reducesum_interface

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

    # ReduceSum (fp32 softmax scores): golden tolerance check
    abs_err = get_abs_err(ref_topk_score, topk_score)
    err_ratio = get_err_ratio(ref_topk_score, topk_score)
    print(f"ReduceSum err: {abs_err:.6f} ratio: {err_ratio:.6f}")

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

    if err_ratio > ERR_RATIO_MAX or abs_err > ABS_ERR_MAX or idx_match_ratio < IDX_MATCH_RATIO_MIN:
        print(
            f"FAIL: exceeds tolerance (err_ratio <= {ERR_RATIO_MAX:g}, abs_err <= {ABS_ERR_MAX:g}, "
            f"idx_match_ratio >= {IDX_MATCH_RATIO_MIN:g})"
        )
        return False
    print(
        f"PASS: within tolerance (err_ratio <= {ERR_RATIO_MAX:g}, abs_err <= {ABS_ERR_MAX:g}, idx_match_ratio >= {IDX_MATCH_RATIO_MIN:g})"
    )
    return True


def test_indexer_topk_reducesum_accuracy() -> None:
    """Pytest entry: gate on the golden tolerance check.

    run_golden_kernel returns a bool (kept for script mode); pytest ignores
    return values, so the gate must be an assert.
    """
    assert run_golden_kernel(), "output exceeds tolerance (see printed err/ratio)"


if __name__ == "__main__":
    success = run_golden_kernel()
    print("Test Passed!" if success else "Test Failed!")
    sys.exit(0 if success else 1)
