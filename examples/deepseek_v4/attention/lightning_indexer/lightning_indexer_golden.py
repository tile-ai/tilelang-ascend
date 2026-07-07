"""CPU golden reference for LightningIndexer.

Pure-torch CPU implementation of the same scoring + top-K algorithm the kernel
runs on NPU, used as the precision reference.

Algorithm (per batch b, kv-group n2, query row s1):
  score[s2] = sum_g relu(Q[s1, n2*G+g, :] · K[s2, n2, :]) * W[s1, n2*G+g]      (float32)
  mask: s2 >= s2_valid -> -inf
        s2_valid = actual_k[b]; if sparse_mode == 3 (causal): s2_valid = actual_k - actual_q + s1 + 1 (when > 0)
  output: top-K s2 positions by score desc; slots beyond valid count filled with -1

Inputs use per-batch actual_seq_lengths (same convention as the kernel wrapper),
so no TND prefix-sum conversion is needed here.
"""

import torch
from typing import Optional, Tuple

_NEG_INF = float("-inf")


def cpu_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    *,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 0,
    block_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    is_tnd = layout_query == "TND"
    is_pa = layout_key == "PA_BSND"
    is_tnd_key = layout_key == "TND"

    q = query.detach().cpu()
    k = key.detach().cpu()
    w = weights.detach().cpu()
    bt = block_table.detach().cpu() if block_table is not None else None
    asq = [int(x) for x in actual_seq_lengths_query.detach().cpu().tolist()]
    ask = [int(x) for x in actual_seq_lengths_key.detach().cpu().tolist()]

    if is_tnd:
        q_tot, N1, D = q.shape
        B = len(asq)
        S1 = max(asq) if asq else 0
    else:
        B, S1, N1, D = q.shape
        q_tot = B * S1
    assert D == 128, f"head dim must be 128, got {D}"
    if is_pa:
        N2 = k.shape[2]
        bs = int(block_size) if block_size else k.shape[1]
    elif is_tnd_key:
        N2 = k.shape[1]
        bs = 128
    else:
        N2 = k.shape[2]
        bs = 128
    G = N1 // N2
    K = sparse_count
    dtype = q.dtype
    # pin to CPU explicitly: caller may have set torch.set_default_device("npu")
    _dev = q.device

    if is_tnd:
        indices = torch.full((q_tot, N2, K), -1, dtype=torch.int32, device=_dev)
        values = torch.full((q_tot, N2, K), _NEG_INF, dtype=dtype, device=_dev)
    else:
        indices = torch.full((B, S1, N2, K), -1, dtype=torch.int32, device=_dev)
        values = torch.full((B, S1, N2, K), _NEG_INF, dtype=dtype, device=_dev)

    if (not asq) or (not ask) or max(asq) == 0 or max(ask) == 0:
        return indices, values

    q_off = [0]
    for x in asq:
        q_off.append(q_off[-1] + x)
    k_off = [0]
    for x in ask:
        k_off.append(k_off[-1] + x)

    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    wf = w.to(torch.float32)
    s1_arange = torch.arange(S1, device=_dev)

    for b in range(B):
        aq, ak = asq[b], ask[b]
        if aq == 0 or ak == 0:
            continue
        qo, ko = q_off[b], k_off[b]
        for n2 in range(N2):
            # gather Q [aq, G, D] and W [aq, G]
            if is_tnd:
                qb = qf[qo : qo + aq, n2 * G : (n2 + 1) * G, :]
                wb = wf[qo : qo + aq, n2 * G : (n2 + 1) * G]
            else:
                qb = qf[b, :aq, n2 * G : (n2 + 1) * G, :]
                wb = wf[b, :aq, n2 * G : (n2 + 1) * G]
            # gather K [ak, D]
            if is_pa:
                assert bt is not None, "block_table required for PA_BSND layout"
                s2_idx = torch.arange(ak, device=_dev)
                bids = bt[b, s2_idx // bs]
                kb = kf[bids, s2_idx % bs, n2, :]
            elif is_tnd_key:
                kb = kf[ko : ko + ak, n2, :]
            else:
                kb = kf[b, :ak, n2, :]

            # score[aq, ak] = sum_g relu(qb @ kb^T) * wb
            qk = torch.relu(torch.einsum("qgd,kd->qgk", qb, kb))
            score = (qk * wb.unsqueeze(2)).sum(dim=1)

            # mask invalid s2 (causal when sparse_mode == 3, else actual_k boundary)
            if sparse_mode == 3:
                cl = ak - aq + s1_arange[:aq] + 1
                s2_valid = torch.where(cl > 0, cl, torch.full_like(cl, ak))
            else:
                s2_valid = torch.full((aq,), ak, device=_dev)
            mask = torch.arange(ak, device=_dev)[None, :] < s2_valid[:, None]
            score = score.masked_fill(~mask, _NEG_INF)

            kk = min(K, ak)
            topv, topi = torch.topk(score, kk, dim=1)
            invalid = topv <= -1e30
            topi = topi.to(torch.int32)
            topi = torch.where(invalid, torch.full_like(topi, -1), topi)
            topv = topv.to(dtype)

            if is_tnd:
                indices[qo : qo + aq, n2, :kk] = topi
                values[qo : qo + aq, n2, :kk] = topv
            else:
                indices[b, :aq, n2, :kk] = topi
                values[b, :aq, n2, :kk] = topv

    return indices, values
