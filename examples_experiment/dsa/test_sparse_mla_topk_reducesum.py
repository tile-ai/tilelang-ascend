"""Test for the sparse_mla_topk_reducesum example (aligned with tile-ai upstream).

Aligned with the upstream test
(examples/dsa_sparse_finetune/sparse_mla_topk_reducesum.py in tile-ai/tilelang),
with the minimal Ascend adaptations:
- device placement adapted to Ascend NPU (.npu())
- einops repeat -> torch .unsqueeze(0).repeat(S, 1) (no einops dependency)
- einsum "s1 h d, s2 d -> s1 h s2" -> "ahd,bd->ahb" (torch requires single-letter
  axes; a=s1, b=s2, same contraction)
- the upstream interface applies the ReduceSum merge over REPLICATE_H and
  the final renormalization inside the interface; this repo's host entry ends
  at ReduceSum, so the test applies those two steps itself (mathematically
  equivalent)
- get_abs_err / get_err_ratio are inlined from the upstream examples/utils.py
"""

import sys

import torch
import torch.nn.functional as F

from sparse_mla_topk_reducesum import sparse_mla_topk_reducesum

ERR_RATIO_MAX = 1e-2
ABS_ERR_MAX = 1e-3


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


def ref_mla_topk_softmax(Q: torch.Tensor, K: torch.Tensor, TopkIndices: torch.Tensor, offsets: torch.Tensor):
    # q: [batch, seq_len, heads, dim]
    # k: [batch, seq_len, dim]
    sm_scale = Q.shape[-1] ** -0.5
    all_lse = []
    all_topk_score = []
    for i in range(offsets.shape[0] - 1):
        q = Q[offsets[i] : offsets[i + 1]]
        k = K[offsets[i] : offsets[i + 1]]
        topk_indices = TopkIndices[offsets[i] : offsets[i + 1]]
        seq_len = q.shape[0]
        mask = (torch.arange(seq_len)[:, None] >= torch.arange(seq_len)[None, :]).unsqueeze(-2).npu()
        logits = torch.einsum("ahd,bd->ahb", q, k) * sm_scale
        logits = torch.where(mask, logits, float("-inf"))
        score = F.softmax(logits, dim=-1, dtype=torch.float32)
        score_sum = score.sum(dim=-2)
        topk_score = torch.gather(score_sum, dim=-1, index=topk_indices.to(torch.int64))
        topk_score = topk_score / topk_score.sum(dim=-1, keepdim=True)
        max_logits = logits.amax(dim=-1).to(torch.float32)
        shifted = (logits - max_logits.unsqueeze(-1).to(torch.float32)).exp().sum(dim=-1)
        lse = torch.log(shifted) + max_logits
        all_lse.append(lse)
        all_topk_score.append(topk_score)
    lse = torch.cat(all_lse, dim=0)
    topk_score = torch.cat(all_topk_score, dim=0)
    return lse, topk_score


def run_golden_kernel(
    B=1,
    S=2048,
    H=16,
    D=512,
    tail_D=64,
    topk=128,
):
    torch.manual_seed(42)

    q = torch.randn((S, H, D + tail_D)).npu().bfloat16()
    kv = torch.randn((S, D + tail_D)).npu().bfloat16()
    offsets = torch.tensor([0, 1023, S], dtype=torch.int32).npu()

    # Equivalent to the upstream einops repeat: repeat(arange(topk), "k -> s k", s=S)
    topk_indices = torch.arange(topk, dtype=torch.int32).npu().unsqueeze(0).repeat(S, 1).contiguous()

    lse, ref_attn_score = ref_mla_topk_softmax(q, kv, topk_indices, offsets)

    kv = kv.unsqueeze(-2)
    topk_indices = topk_indices.unsqueeze(-2)

    # The upstream interface returns the normalized attention scores
    # directly; this repo's host entry ends at ReduceSum, so the interface-level
    # merge + renormalization are applied here (mathematically equivalent):
    #   ReduceSum [S, G, RH, topk] --sum over RH--> [S, G, topk]
    #   --renorm over topk--> attn_score --squeeze G=1--> [S, topk]
    rs = sparse_mla_topk_reducesum(q, kv, topk_indices, lse, offsets, dim_v=D)
    reducesum = rs.sum(dim=-2)
    attn_score = (reducesum / reducesum.sum(dim=-1, keepdim=True)).squeeze(-2)

    abs_err = get_abs_err(attn_score, ref_attn_score)
    err_ratio = get_err_ratio(attn_score, ref_attn_score)
    print(f"attn_score err: {abs_err:.6f} ratio: {err_ratio:.6f}")

    if err_ratio > ERR_RATIO_MAX or abs_err > ABS_ERR_MAX:
        print(f"FAIL: attn_score exceeds tolerance (err_ratio <= {ERR_RATIO_MAX:g}, abs_err <= {ABS_ERR_MAX:g})")
        return False
    print(f"PASS: attn_score within tolerance (err_ratio <= {ERR_RATIO_MAX:g}, abs_err <= {ABS_ERR_MAX:g})")
    return True


def test_sparse_mla_topk_reducesum_accuracy() -> None:
    """Pytest entry: gate on the golden tolerance check.

    run_golden_kernel returns a bool (kept for script mode); pytest ignores
    return values, so the gate must be an assert.
    """
    assert run_golden_kernel(), "attn_score exceeds tolerance (see printed err/ratio)"


if __name__ == "__main__":
    success = run_golden_kernel()
    print("Test Passed!" if success else "Test Failed!")
    sys.exit(0 if success else 1)
