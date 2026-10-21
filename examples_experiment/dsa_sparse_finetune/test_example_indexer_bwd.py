import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_indexer_bwd import indexer_bwd_interface  # noqa: E402


def ref_indexer_bwd(
    Q: torch.Tensor,
    Weights: torch.Tensor,
    K: torch.Tensor,
    TopkIndices: torch.Tensor,
    AttnScore: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    Q.requires_grad_(True)
    Weights.requires_grad_(True)
    K.requires_grad_(True)
    softmax_scale = Q.shape[-1] ** -0.5
    all_loss = []
    all_log_topk_prob = []
    for i in range(offsets.shape[0] - 1):
        assert (offsets[i + 1] - offsets[i]).item() >= TopkIndices.shape[-1]
        q = Q[offsets[i] : offsets[i + 1]]
        weights = Weights[offsets[i] : offsets[i + 1]]
        k = K[offsets[i] : offsets[i + 1]]
        topk_indices = TopkIndices[offsets[i] : offsets[i + 1]]
        attn_score = AttnScore[offsets[i] : offsets[i + 1]]
        s = q.shape[0]
        mask = (torch.arange(s)[:, None] >= torch.arange(s)[None, :]).to(q.device)
        logits = torch.einsum("shd,nd->shn", q, k) * softmax_scale
        logits = F.relu(logits)
        score = (logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32)
        score = torch.where(mask, score, float("-inf"))
        topk_value = torch.gather(score, dim=-1, index=topk_indices.to(torch.int64))
        log_topk_prob = F.log_softmax(topk_value, dim=-1, dtype=torch.float32)
        loss = F.kl_div(
            log_topk_prob.clip(-100, 0),
            attn_score.log().clip(-100, 0),
            log_target=True,
            reduction="sum",
        )
        all_loss.append(loss)
        all_log_topk_prob.append(log_topk_prob)
    loss = torch.stack(all_loss).sum()
    loss.backward()
    log_topk_prob = torch.cat(all_log_topk_prob, dim=0)
    return log_topk_prob.exp(), Q.grad, Weights.grad, K.grad


def get_abs_err(y, x):
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    return (x - y).flatten().abs().max().item()


def get_err_ratio(y, x):
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    err = (x - y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def check_precision(actual, golden, atol=2**-10, rtol=2**-6, max_abs_limit=1e0, required_ratio=0.99):
    """Official bf16 dual-gate: per-element |a-g| <= atol + rtol*|g|; a case
    passes iff matched_ratio >= 0.99 AND max_abs <= max_abs_limit."""
    a = actual.detach().cpu().float()
    g = golden.detach().cpu().float()
    abs_err = (a - g).abs()
    matched_ratio = (abs_err <= atol + rtol * g.abs()).float().mean().item()
    return matched_ratio >= required_ratio and abs_err.max().item() <= max_abs_limit


def test_kernel(
    B=1,
    S=2048,
    H=16,
    D=128,
    topk=64,
):
    torch.manual_seed(42)
    q = torch.randn((S, H, D)).npu().bfloat16()
    w = torch.randn((S, H)).npu().bfloat16()
    k = torch.randn((S, D)).npu().bfloat16()
    offsets = torch.tensor([0, 1023, S], dtype=torch.int32).npu()

    all_attn_score = []
    for i in range(offsets.shape[0] - 1):
        seq_len = (offsets[i + 1] - offsets[i]).item()
        mask = (torch.arange(seq_len)[:, None] >= torch.arange(topk)[None, :]).to(q.device)
        logits = torch.ones(seq_len, topk).npu()
        logits = torch.where(mask, logits, float("-inf"))
        attn_score = F.softmax(logits, dim=-1, dtype=torch.float32)
        all_attn_score.append(attn_score)
    attn_score = torch.cat(all_attn_score, dim=0)

    topk_indices = torch.arange(topk, dtype=torch.int32).npu().unsqueeze(0).repeat(S, 1).contiguous()
    index_score, ref_dq, ref_dw, ref_dk = ref_indexer_bwd(q, w, k, topk_indices, attn_score, offsets)

    dq, dw, dk = indexer_bwd_interface(q, w, k, attn_score, index_score, topk_indices, offsets)

    print(f"dq err: {get_abs_err(dq, ref_dq):.6f} ratio: {get_err_ratio(dq, ref_dq):.6f}")
    print(f"dw err: {get_abs_err(dw, ref_dw):.6f} ratio: {get_err_ratio(dw, ref_dw):.6f}")
    print(f"dk err: {get_abs_err(dk, ref_dk):.6f} ratio: {get_err_ratio(dk, ref_dk):.6f}")

    if not all(check_precision(a, g) for a, g in ((dq, ref_dq), (dw, ref_dw), (dk, ref_dk))):
        raise AssertionError("indexer_bwd precision gate failed (bf16 dual-gate)")


if __name__ == "__main__":
    test_kernel()
    print("Test Passed!")
