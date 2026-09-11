# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch


GOLDEN_S2_TILE = 512


def _gather_cmp_kv_tokens(
    cmp_k_bnsd: torch.Tensor,
    sparse_indices: torch.Tensor,
    b: int,
    n2: int,
    s_local: int,
    act_kv: int,
    act_q: int,
    cmp_ratio: int,
    cmp_mask_mode: int = 3,
    sparse_block_size: int = 1,
) -> torch.Tensor:
    K = sparse_indices.shape[-1]
    cmp_act_kv = math.floor(act_kv / cmp_ratio)
    if cmp_mask_mode == 3:
        threshold = math.floor((act_kv - act_q + s_local + 1) / cmp_ratio)
    else:
        raise ValueError(f"cmp_mask_mode {cmp_mask_mode} not supported")

    valid_count = min(K, math.ceil(threshold / sparse_block_size))
    s2 = []
    for i in range(valid_count):
        topk_id = sparse_indices[b, s_local, n2, i].item()
        if topk_id == -1:
            break
        begin = topk_id * sparse_block_size
        end = begin + sparse_block_size
        if end > cmp_act_kv:
            end = cmp_act_kv
        if begin >= threshold:
            continue
        if end > threshold:
            end = threshold
        s2.extend(range(begin, end))

    if not s2:
        return cmp_k_bnsd.new_zeros((0, cmp_k_bnsd.shape[-1]))
    return cmp_k_bnsd[b, n2, s2, :]


def sparse_flash_mla_golden_bnsd(
    q_bnsd: torch.Tensor,
    ori_k_bnsd: torch.Tensor,
    sinks: torch.Tensor,
    *,
    act_q_lens: Sequence[int],
    act_kv_lens: Sequence[int],
    softmax_scale: float,
    cmp_k_bnsd: Optional[torch.Tensor] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    cmp_ratio: Optional[int] = None,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    s2_tile: int = GOLDEN_S2_TILE,
    return_lse: bool = False,
):
    assert ori_win_right == 0, "only ori_win_right=0 (causal) is supported"
    assert ori_mask_mode == 4, "only ori_mask_mode=4 supported"

    B, N1, S1, D = q_bnsd.shape
    N2 = ori_k_bnsd.shape[1]
    G = N1 // N2
    dtype = q_bnsd.dtype
    out = torch.zeros_like(q_bnsd, dtype=dtype)
    lse = torch.zeros((B, N1, S1), dtype=torch.float32, device=q_bnsd.device) if return_lse else None

    has_cmp = cmp_k_bnsd is not None and cmp_sparse_indices is not None
    is_dense_cmp = cmp_k_bnsd is not None and cmp_sparse_indices is None

    for b in range(B):
        act_q = int(act_q_lens[b])
        act_kv = int(act_kv_lens[b])
        for n2 in range(N2):
            head_lo = n2 * G
            head_hi = (n2 + 1) * G
            sink_group = sinks[head_lo:head_hi].to(torch.float32)
            for s in range(act_q):
                s_global = act_kv - act_q + s
                q = q_bnsd[b, head_lo:head_hi, s, :].to(torch.float32)

                ori_right = s_global + ori_win_right + 1
                ori_left = max(s_global - ori_win_left, 0)
                ori_k = ori_k_bnsd[b, n2, ori_left:ori_right, :].to(torch.float32)

                if has_cmp:
                    cmp_tokens = _gather_cmp_kv_tokens(
                        cmp_k_bnsd,
                        cmp_sparse_indices,
                        b=b,
                        n2=n2,
                        s_local=s,
                        act_kv=act_kv,
                        act_q=act_q,
                        cmp_ratio=cmp_ratio,
                        cmp_mask_mode=cmp_mask_mode,
                    ).to(torch.float32)
                elif is_dense_cmp:
                    threshold = math.floor((act_kv - act_q + s + 1) / cmp_ratio)
                    cmp_tokens = cmp_k_bnsd[b, n2, : max(threshold, 0), :].to(torch.float32)
                else:
                    cmp_tokens = torch.empty((0, D), dtype=torch.float32, device=q_bnsd.device)

                row_max = sink_group.clone()
                row_sum = torch.ones(G, dtype=torch.float32, device=q_bnsd.device)
                acc_o = torch.zeros((G, D), dtype=torch.float32, device=q_bnsd.device)

                ori_tiles = max(1, math.ceil(ori_k.size(0) / s2_tile)) if ori_k.size(0) > 0 else 0
                cmp_tiles = math.ceil(cmp_tokens.size(0) / s2_tile) if cmp_tokens.size(0) > 0 else 0

                for t in range(ori_tiles + cmp_tiles):
                    if t < ori_tiles:
                        k_tile = ori_k[t * s2_tile : (t + 1) * s2_tile, :]
                    else:
                        ct = t - ori_tiles
                        k_tile = cmp_tokens[ct * s2_tile : (ct + 1) * s2_tile, :]
                    if k_tile.size(0) == 0:
                        continue

                    score = torch.matmul(q, k_tile.T) * softmax_scale
                    row_max_old = row_max.clone()
                    row_max = torch.max(row_max, score.amax(dim=1))
                    alpha = torch.exp(row_max_old - row_max)
                    p = torch.exp(score - row_max.unsqueeze(1))
                    row_sum = alpha * row_sum + p.sum(dim=1)

                    p_low = p.to(dtype).to(torch.float32)
                    pv = torch.matmul(p_low, k_tile)
                    acc_o = acc_o * alpha.unsqueeze(1) + pv

                out[b, head_lo:head_hi, s, :] = (acc_o / row_sum.unsqueeze(1)).to(dtype)
                if return_lse:
                    lse[b, head_lo:head_hi, s] = row_max + torch.log(row_sum)

    if return_lse:
        return out, lse
    return out


def sinks_softmax_reference(
    q_bnsd: torch.Tensor,
    k_concat_bnsd: torch.Tensor,
    *,
    sinks: torch.Tensor,
    softmax_scale: float,
    return_lse: bool = False,
):
    q = q_bnsd.to(torch.float32)
    k = k_concat_bnsd.to(torch.float32)
    score = torch.matmul(q, k.transpose(-1, -2)) * softmax_scale
    sinks_b = sinks.to(torch.float32).view(-1, 1)
    x_concat = torch.cat([score, sinks_b.expand(score.shape[:-1] + (1,))], dim=-1)
    x_max = x_concat.amax(dim=-1, keepdim=True)
    y = torch.exp(score - x_max)
    denom = y.sum(dim=-1, keepdim=True) + torch.exp(sinks_b - x_max)
    p = y / denom
    out = torch.matmul(p.to(q_bnsd.dtype).to(torch.float32), k).to(q_bnsd.dtype)
    if return_lse:
        lse = (x_max + torch.log(denom)).squeeze(-1)
        return out, lse
    return out


def unpack_paged_kv(
    kv_pa: torch.Tensor,
    block_table: torch.Tensor,
    max_logical_len: int,
) -> torch.Tensor:
    block_num, block_size, N2, D = kv_pa.shape
    B = block_table.shape[0]
    out = kv_pa.new_zeros((B, N2, max_logical_len, D))
    for b in range(B):
        for blk_i in range(block_table.shape[1]):
            phys = int(block_table[b, blk_i].item())
            if phys < 0:
                continue
            start = blk_i * block_size
            end = start + block_size
            if start >= max_logical_len:
                break
            end = min(end, max_logical_len)
            take = end - start
            out[b, :, start:end, :] = kv_pa[phys, :take, :, :].permute(1, 0, 2)
    return out


def tnd_to_bnsd_q(q_tnd: torch.Tensor, cu_seqlens_q: torch.Tensor) -> torch.Tensor:
    T, N, D = q_tnd.shape
    seq_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    B = len(seq_lens)
    S_max = max(seq_lens) if seq_lens else 0
    out = q_tnd.new_zeros((B, N, S_max, D))
    for b in range(B):
        start = int(cu_seqlens_q[b].item())
        end = int(cu_seqlens_q[b + 1].item())
        L = end - start
        out[b, :, :L, :] = q_tnd[start:end, :, :].permute(1, 0, 2)
    return out


def bnsd_to_tnd_out(out_bnsd: torch.Tensor, cu_seqlens_q: torch.Tensor) -> torch.Tensor:
    B, N, _, D = out_bnsd.shape
    seq_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    T_total = sum(seq_lens)
    out = out_bnsd.new_zeros((T_total, N, D))
    for b in range(B):
        start = int(cu_seqlens_q[b].item())
        end = int(cu_seqlens_q[b + 1].item())
        L = end - start
        out[start:end, :, :] = out_bnsd[b, :, :L, :].permute(1, 0, 2)
    return out


def bnsd_to_tnd_lse(lse_bnsd: torch.Tensor, cu_seqlens_q: torch.Tensor) -> torch.Tensor:
    B, N, _ = lse_bnsd.shape
    seq_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    T_total = sum(seq_lens)
    out = lse_bnsd.new_zeros((T_total, N))
    for b in range(B):
        start = int(cu_seqlens_q[b].item())
        end = int(cu_seqlens_q[b + 1].item())
        L = end - start

        out[start:end, :] = lse_bnsd[b, :, :L].permute(1, 0)
    return out


def gen_ori_kv_paged(
    *,
    B: int,
    N2: int,
    D: int,
    block_num: int,
    block_size: int,
    seqused_kv: Sequence[int],
    dtype: torch.dtype,
    data_range: tuple = (-10, 10),
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    rng = rng or np.random.default_rng(0)
    max_s = max(seqused_kv)
    max_blk = math.ceil(max_s / block_size)
    lo, hi = data_range

    ori_k_bnsd = (torch.rand((B, N2, max_s, D)) * (hi - lo) + lo).to(dtype)

    per_batch_blocks = [math.ceil(s / block_size) for s in seqused_kv]
    total_blocks = sum(per_batch_blocks)
    if block_num < total_blocks:
        raise ValueError(f"ori block_num too small: {block_num} < {total_blocks}")

    phys_ids = rng.permutation(block_num).astype(np.int32)
    table = np.full((B, max_blk), -1, dtype=np.int32)
    cursor = 0
    for b, nb in enumerate(per_batch_blocks):
        for i in range(nb):
            table[b, i] = phys_ids[cursor]
            cursor += 1

    pa = torch.zeros((block_num, block_size, N2, D), dtype=dtype)
    for b in range(B):
        for i in range(per_batch_blocks[b]):
            phys = int(table[b, i])
            s0 = i * block_size
            s1 = min(s0 + block_size, max_s)
            take = s1 - s0
            pa[phys, :take, :, :] = ori_k_bnsd[b, :, s0:s1, :].permute(1, 0, 2)

    return pa, torch.tensor(table, dtype=torch.int32), ori_k_bnsd


def gen_cmp_kv_paged(
    *,
    B: int,
    N2: int,
    D: int,
    block_num: int,
    block_size: int,
    seqused_kv: Sequence[int],
    cmp_ratio: int,
    dtype: torch.dtype,
    data_range: tuple = (-5, 10),
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    rng = rng or np.random.default_rng(1)
    cmp_seqs = [math.floor(s / cmp_ratio) for s in seqused_kv]
    max_cmp = max(cmp_seqs) if cmp_seqs else 0
    max_blk = math.ceil(max_cmp / block_size) if max_cmp > 0 else 1
    lo, hi = data_range

    cmp_k_bnsd = (torch.rand((B, N2, max(max_cmp, 1), D)) * (hi - lo) + lo).to(dtype)

    per_batch_blocks = [math.ceil(s / block_size) if s > 0 else 0 for s in cmp_seqs]
    total_blocks = sum(per_batch_blocks)
    if block_num < max(total_blocks, 1):
        raise ValueError(f"cmp block_num too small: {block_num} < {total_blocks}")

    phys_ids = rng.permutation(block_num).astype(np.int32)
    table = np.full((B, max(max_blk, 1)), -1, dtype=np.int32)
    cursor = 0
    for b, nb in enumerate(per_batch_blocks):
        for i in range(nb):
            table[b, i] = phys_ids[cursor]
            cursor += 1

    pa = torch.zeros((block_num, block_size, N2, D), dtype=dtype)
    for b in range(B):
        for i in range(per_batch_blocks[b]):
            phys = int(table[b, i])
            s0 = i * block_size
            s1 = min(s0 + block_size, cmp_seqs[b])
            take = s1 - s0
            if take > 0:
                pa[phys, :take, :, :] = cmp_k_bnsd[b, :, s0:s1, :].permute(1, 0, 2)

    return pa, torch.tensor(table, dtype=torch.int32), cmp_k_bnsd, cmp_seqs


def gen_cmp_sparse_indices_bsnd(
    *,
    B: int,
    S1: int,
    N2: int,
    K: int,
    seqused_kv: Sequence[int],
    cmp_ratio: int,
    cmp_mask_mode: int = 3,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if cmp_mask_mode != 3:
        raise ValueError("only cmp_mask_mode=3 supported")
    cmp_sparse_indices = torch.full((B, S1, N2, K), fill_value=-1, dtype=torch.int32)
    for b in range(B):
        act_kv = int(seqused_kv[b])
        for n2 in range(N2):
            for s in range(S1):
                cur_max = math.floor((act_kv - S1 + s + 1) / cmp_ratio)
                cur_max = max(0, cur_max)
                perm = torch.randperm(cur_max, generator=rng).to(torch.int32) if cur_max > 0 else torch.empty((0,), dtype=torch.int32)
                take = min(cur_max, K)
                cmp_sparse_indices[b, s, n2, :take] = perm[:take]
    return cmp_sparse_indices


def gen_cmp_sparse_indices_tnd(
    *,
    B: int,
    T1: int,
    N2: int,
    K: int,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: Sequence[int],
    cmp_ratio: int,
    cmp_mask_mode: int = 3,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if cmp_mask_mode != 3:
        raise ValueError("only cmp_mask_mode=3 supported")
    cmp_sparse_indices = torch.full((T1, N2, K), fill_value=-1, dtype=torch.int32)
    for b in range(B):
        s_start = int(cu_seqlens_q[b].item())
        s_end = int(cu_seqlens_q[b + 1].item())
        act_q = s_end - s_start
        act_kv = int(seqused_kv[b])
        for n2 in range(N2):
            for s in range(act_q):
                cur_max = math.floor((act_kv - act_q + s + 1) / cmp_ratio)
                cur_max = max(0, cur_max)
                perm = torch.randperm(cur_max, generator=rng).to(torch.int32) if cur_max > 0 else torch.empty((0,), dtype=torch.int32)
                take = min(cur_max, K)
                cmp_sparse_indices[s_start + s, n2, :take] = perm[:take]
    return cmp_sparse_indices


FAST_SCENARIOS = {
    "swa": dict(
        scenario=1,
        layout_q="TND",
        B=1,
        S1=1024,
        T1=1024,
        N1=64,
        N2=1,
        D=512,
        K=0,
        block_num1=10,
        block_num2=1,
        block_size1=128,
        block_size2=1,
        cu_seqlens_q=[0, 1024],
        seqused_kv=[1024],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
    "hca": dict(
        scenario=2,
        layout_q="TND",
        B=1,
        S1=1024,
        T1=1024,
        N1=64,
        N2=1,
        D=512,
        K=512,
        block_num1=10,
        block_num2=2,
        block_size1=128,
        block_size2=128,
        cu_seqlens_q=[0, 1024],
        seqused_kv=[1024],
        softmax_scale=0.04419417,
        cmp_ratio=128,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
    "csa": dict(
        scenario=3,
        layout_q="TND",
        B=1,
        S1=1024,
        T1=1024,
        N1=64,
        N2=1,
        D=512,
        K=512,
        block_num1=10,
        block_num2=3,
        block_size1=128,
        block_size2=128,
        cu_seqlens_q=[0, 1024],
        seqused_kv=[1024],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
}


def build_case(cfg, dtype, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    layout_q = cfg["layout_q"]
    B = cfg["B"]
    S1 = cfg["S1"]
    T1 = cfg.get("T1", S1 * B)
    N1, N2, D = cfg["N1"], cfg["N2"], cfg["D"]
    K = cfg.get("K", 0)
    cmp_ratio = cfg["cmp_ratio"]
    block_size1, block_num1 = cfg["block_size1"], cfg["block_num1"]
    block_size2, block_num2 = cfg["block_size2"], cfg["block_num2"]
    seqused_kv = cfg["seqused_kv"]
    cu_seqlens_q = torch.tensor(cfg["cu_seqlens_q"], dtype=torch.int32)
    softmax_scale = cfg["softmax_scale"]
    scenario = cfg["scenario"]

    if layout_q == "TND":
        q = (torch.rand((T1, N1, D)) * 20 - 10).to(dtype)
    else:
        q = (torch.rand((B, S1, N1, D)) * 20 - 10).to(dtype)

    ori_pa, ori_bt, ori_k_bnsd = gen_ori_kv_paged(
        B=B,
        N2=N2,
        D=D,
        block_num=block_num1,
        block_size=block_size1,
        seqused_kv=seqused_kv,
        dtype=dtype,
        data_range=(-10, 10),
        rng=np.random.default_rng(seed),
    )

    if scenario >= 2:
        cmp_pa, cmp_bt, cmp_k_bnsd, cmp_seqs = gen_cmp_kv_paged(
            B=B,
            N2=N2,
            D=D,
            block_num=block_num2,
            block_size=block_size2,
            seqused_kv=seqused_kv,
            cmp_ratio=cmp_ratio,
            dtype=dtype,
            data_range=(-5, 10),
            rng=np.random.default_rng(seed + 1),
        )
    else:
        cmp_pa, cmp_bt, cmp_k_bnsd = None, None, None

    if scenario == 3:
        rng = torch.Generator()
        rng.manual_seed(seed + 7)
        if layout_q == "TND":
            cmp_idx = gen_cmp_sparse_indices_tnd(
                B=B,
                T1=T1,
                N2=N2,
                K=K,
                cu_seqlens_q=cu_seqlens_q,
                seqused_kv=seqused_kv,
                cmp_ratio=cmp_ratio,
                cmp_mask_mode=3,
                rng=rng,
            )
        else:
            cmp_idx = gen_cmp_sparse_indices_bsnd(
                B=B,
                S1=S1,
                N2=N2,
                K=K,
                seqused_kv=seqused_kv,
                cmp_ratio=cmp_ratio,
                cmp_mask_mode=3,
                rng=rng,
            )
    else:
        cmp_idx = None

    sinks = (torch.rand(N1) * 2 - 1).to(torch.float32)

    if layout_q == "TND":
        q_bnsd_ref = tnd_to_bnsd_q(q, cu_seqlens_q)
        act_q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    else:
        q_bnsd_ref = q.permute(0, 2, 1, 3).contiguous()
        act_q_lens = [S1] * B

    if cmp_idx is not None:
        if layout_q == "TND":
            S_max = max(act_q_lens)
            cmp_idx_bsnd = torch.full((B, S_max, N2, K), -1, dtype=torch.int32)
            for b in range(B):
                s_start = int(cu_seqlens_q[b].item())
                L = int(act_q_lens[b])
                cmp_idx_bsnd[b, :L, :, :] = cmp_idx[s_start : s_start + L, :, :]
        else:
            cmp_idx_bsnd = cmp_idx
    else:
        cmp_idx_bsnd = None

    cpu_ref, cpu_ref_lse = sparse_flash_mla_golden_bnsd(
        q_bnsd_ref,
        ori_k_bnsd,
        sinks,
        act_q_lens=act_q_lens,
        act_kv_lens=seqused_kv,
        softmax_scale=softmax_scale,
        cmp_k_bnsd=cmp_k_bnsd,
        cmp_sparse_indices=cmp_idx_bsnd if scenario == 3 else None,
        cmp_ratio=cmp_ratio if scenario >= 2 else None,
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
        ori_mask_mode=cfg["ori_mask_mode"],
        cmp_mask_mode=cfg["cmp_mask_mode"],
        return_lse=True,
    )

    if layout_q == "TND":
        cpu_ref = bnsd_to_tnd_out(cpu_ref, cu_seqlens_q)
        cpu_ref_lse = bnsd_to_tnd_lse(cpu_ref_lse, cu_seqlens_q)
    else:
        cpu_ref = cpu_ref.permute(0, 2, 1, 3).contiguous()
        cpu_ref_lse = cpu_ref_lse.permute(0, 2, 1).contiguous()

    return dict(
        cfg=cfg,
        q=q,
        ori_pa=ori_pa,
        ori_bt=ori_bt,
        cmp_pa=cmp_pa,
        cmp_bt=cmp_bt,
        cmp_idx=cmp_idx,
        sinks=sinks,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=torch.tensor(seqused_kv, dtype=torch.int32),
        cpu_ref=cpu_ref,
        cpu_ref_lse=cpu_ref_lse,
    )


def check_result(npu_out, expect):
    if npu_out.dtype == torch.bfloat16:
        rtol, atol = 0.0078125, 0.0001
    else:
        rtol, atol = 0.005, 0.000025

    real = npu_out.detach().cpu().to(torch.float32).numpy().flatten()
    expt = expect.detach().cpu().to(torch.float32).numpy().flatten()
    assert real.size == expt.size, f"size mismatch: {real.size} vs {expt.size}"

    ok = np.isclose(real, expt, rtol=rtol, atol=atol, equal_nan=True)
    n_err = int((~ok).sum())
    fulfill_pct = (real.size - n_err) / real.size * 100.0

    diff_thd = 0.005
    norm_floor = (1.0 / (1 << 14)) / diff_thd
    b = np.maximum(np.maximum(np.abs(real), np.abs(expt)), norm_floor) + 1e-9
    rel_err = np.abs(real - expt) / b
    max_rel = float(rel_err[~ok].max()) if n_err > 0 else 0.0

    assert fulfill_pct >= 99.5, (
        f"only {fulfill_pct:.4f}% of elements within tol "
        f"(rtol={rtol}, atol={atol}); 99.5% required; "
        f"{n_err}/{real.size} failing, max rel err {max_rel:.4f}"
    )
    assert max_rel < 10.0, (
        f"max normalized relative error {max_rel:.4f} exceeds cap 10.0 (fulfill {fulfill_pct:.4f}%, {n_err}/{real.size} failing)"
    )


def check_lse(npu_lse, expect_lse, q_dtype):
    if q_dtype == torch.bfloat16:
        rtol, atol = 0.015, 0.005
    else:
        rtol, atol = 0.01, 0.001

    real = npu_lse.detach().cpu().to(torch.float32).numpy().flatten()
    expt = expect_lse.detach().cpu().to(torch.float32).numpy().flatten()
    assert real.size == expt.size, f"lse size mismatch: {real.size} vs {expt.size}"

    ok = np.isclose(real, expt, rtol=rtol, atol=atol, equal_nan=True)
    n_err = int((~ok).sum())
    fulfill_pct = (real.size - n_err) / real.size * 100.0

    diff_thd = 0.005
    norm_floor = (1.0 / (1 << 14)) / diff_thd
    b = np.maximum(np.maximum(np.abs(real), np.abs(expt)), norm_floor) + 1e-9
    rel_err = np.abs(real - expt) / b
    max_rel = float(rel_err[~ok].max()) if n_err > 0 else 0.0

    assert fulfill_pct >= 99.5, (
        f"lse: only {fulfill_pct:.4f}% within tol "
        f"(rtol={rtol}, atol={atol}); 99.5% required; "
        f"{n_err}/{real.size} failing, max rel err {max_rel:.4f}"
    )
    assert max_rel < 10.0, (
        f"lse: max normalized relative error {max_rel:.4f} exceeds cap 10.0 (fulfill {fulfill_pct:.4f}%, {n_err}/{real.size} failing)"
    )


def run_case(case, cfg, sparse_flash_mla_fn):
    layout_q = cfg["layout_q"]
    scenario = cfg["scenario"]
    K = cfg.get("K", 0)

    def _dev(t):
        return t.npu().contiguous() if t is not None and hasattr(t, "npu") else t

    cu_seqlens_q_dev = _dev(case["cu_seqlens_q"]) if layout_q == "TND" else None
    seqused_kv_dev = _dev(case["seqused_kv"])

    with torch.device("npu"):
        out, lse = sparse_flash_mla_fn(
            _dev(case["q"]),
            ori_kv=_dev(case["ori_pa"]),
            cmp_kv=_dev(case["cmp_pa"]),
            cmp_sparse_indices=_dev(case["cmp_idx"]),
            ori_block_table=_dev(case["ori_bt"]),
            cmp_block_table=_dev(case["cmp_bt"]),
            cu_seqlens_q=cu_seqlens_q_dev,
            seqused_kv=seqused_kv_dev,
            sinks=_dev(case["sinks"]),
            softmax_scale=cfg["softmax_scale"],
            cmp_ratio=cfg["cmp_ratio"] if scenario >= 2 else None,
            ori_mask_mode=cfg["ori_mask_mode"],
            cmp_mask_mode=cfg["cmp_mask_mode"],
            ori_win_left=cfg["ori_win_left"],
            ori_win_right=cfg["ori_win_right"],
            layout_q=layout_q,
            layout_kv="PA_ND",
            return_softmax_lse=True,
            topk_cmp=K,
        )
        torch.npu.synchronize()
    return out, lse
