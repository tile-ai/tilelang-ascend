# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from sparse_flash_mla_kernel import build_sparse_flash_mla, DEFAULT_BLOCK_I, DEFAULT_CORE_NUM


_KERNEL_CACHE: dict = {}


def _torch_to_tilelang_dtype(t: torch.dtype) -> str:
    if t == torch.bfloat16:
        return "bfloat16"
    if t == torch.float16:
        return "float16"
    raise ValueError(f"unsupported torch dtype {t}")


def _get_kernel(
    *,
    batch: int,
    max_seq: int,
    total_tokens: int,
    ori_block_num: int,
    ori_block_size: int,
    ori_table_len: int,
    cmp_block_num: int,
    cmp_block_size: int,
    cmp_table_len: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    topk_cmp: int,
    cmp_ratio: int,
    scenario: int,
    ori_win_left: int,
    softmax_scale: float,
    dtype: str,
    core_num: int,
):
    key = (
        batch,
        max_seq,
        total_tokens,
        ori_block_num,
        ori_block_size,
        ori_table_len,
        cmp_block_num,
        cmp_block_size,
        cmp_table_len,
        n_heads,
        n_kv_heads,
        head_dim,
        topk_cmp,
        cmp_ratio,
        scenario,
        ori_win_left,
        round(softmax_scale, 8),
        dtype,
        core_num,
    )
    func = _KERNEL_CACHE.get(key)
    if func is None:
        func = build_sparse_flash_mla(
            batch=batch,
            max_seq=max_seq,
            total_tokens=total_tokens,
            ori_block_num=ori_block_num,
            ori_block_size=ori_block_size,
            ori_table_len=ori_table_len,
            cmp_block_num=cmp_block_num,
            cmp_block_size=cmp_block_size,
            cmp_table_len=cmp_table_len,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            topk_cmp=topk_cmp,
            cmp_ratio=cmp_ratio,
            scenario=scenario,
            ori_win_left=ori_win_left,
            softmax_scale=softmax_scale,
            dtype=dtype,
            core_num=core_num,
        )
        _KERNEL_CACHE[key] = func
    return func


def sparse_flash_mla(
    q: torch.Tensor,
    *,
    ori_kv: torch.Tensor,
    cmp_kv: Optional[torch.Tensor] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    ori_block_table: torch.Tensor,
    cmp_block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    softmax_scale: float,
    cmp_ratio: Optional[int] = None,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "TND",
    layout_kv: str = "PA_ND",
    return_softmax_lse: bool = False,
    core_num: int = DEFAULT_CORE_NUM,
    topk_cmp: Optional[int] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    del seqused_q
    assert ori_mask_mode == 4, "only ori_mask_mode=4 supported"
    assert cmp_mask_mode == 3 or cmp_kv is None, "only cmp_mask_mode=3 supported"
    assert ori_win_right == 0, "only ori_win_right=0 supported"
    assert layout_kv == "PA_ND", "only layout_kv=PA_ND supported"
    assert layout_q in ("TND", "BSND"), f"unsupported layout_q={layout_q!r}"

    dtype = q.dtype
    tl_dtype = _torch_to_tilelang_dtype(dtype)

    if cmp_kv is None:
        scenario = 1
    elif cmp_sparse_indices is None:
        scenario = 2
    else:
        scenario = 3

    if layout_q == "TND":
        assert cu_seqlens_q is not None, "cu_seqlens_q is required for TND"
        cu = cu_seqlens_q.to(torch.int32).cpu()
        B = cu.numel() - 1
        seq_lens = (cu[1:] - cu[:-1]).tolist()
        S_max = max(seq_lens) if seq_lens else 0
        T1, N1, D = q.shape
        total_tokens = T1
        q_flat = q
        q_prefix = cu[:-1].to(q.device)
        act_q_lens = torch.tensor(seq_lens, dtype=torch.int32, device=q.device)
        if scenario == 3:
            cmp_indices_flat = cmp_sparse_indices.to(torch.int32).to(q.device)
        else:
            cmp_indices_flat = None
    else:
        B, S_max, N1, D = q.shape
        total_tokens = B * S_max
        q_flat = q.reshape(total_tokens, N1, D)
        q_prefix = torch.arange(B, dtype=torch.int32, device=q.device) * S_max
        seq_lens = [S_max] * B
        act_q_lens = torch.full((B,), S_max, dtype=torch.int32, device=q.device)
        if cu_seqlens_q is not None:
            cu = cu_seqlens_q.to(torch.int32).cpu()
            seq_lens = (cu[1:] - cu[:-1]).tolist()
            act_q_lens = torch.tensor(seq_lens, dtype=torch.int32, device=q.device)
        if scenario == 3:
            ci = cmp_sparse_indices.to(torch.int32)
            cmp_indices_flat = ci.reshape(total_tokens, ci.shape[2], ci.shape[3]).to(q.device)
        else:
            cmp_indices_flat = None

    seqused_kv_dev = seqused_kv.to(torch.int32).to(q.device)

    if scenario == 3:
        N2 = cmp_indices_flat.shape[1]
        K = topk_cmp if topk_cmp is not None else cmp_indices_flat.shape[2]
        assert cmp_indices_flat.shape[2] == K, "topk_cmp does not match cmp_sparse_indices last dim"
        cmp_indices_dev = cmp_indices_flat
    elif scenario == 2:
        N2 = cmp_kv.shape[2]

        max_cmp = int(seqused_kv.max().item()) // cmp_ratio

        _bi = DEFAULT_BLOCK_I
        K = max(_bi, ((max_cmp + _bi - 1) // _bi) * _bi)
        cmp_indices_dev = torch.zeros((total_tokens, N2, K), dtype=torch.int32, device=q.device)
    else:
        N2 = ori_kv.shape[2]
        K = 0

        cmp_indices_dev = torch.zeros((total_tokens, N2, DEFAULT_BLOCK_I), dtype=torch.int32, device=q.device)

    if N1 != 64 or N2 != 1 or D != 512:
        raise ValueError(f"only N1=64, N2=1, D=512 supported (got N1={N1}, N2={N2}, D={D})")

    cmp_ratio_eff = cmp_ratio if cmp_ratio is not None else 4

    ori_kv_dev = ori_kv.to(q.device)
    ori_bt_dev = ori_block_table.to(torch.int32).to(q.device)
    ori_block_num, ori_block_size = ori_kv_dev.shape[0], ori_kv_dev.shape[1]
    ori_table_len = ori_bt_dev.shape[1]

    if cmp_kv is not None:
        cmp_kv_dev = cmp_kv.to(q.device)
        cmp_bt_dev = cmp_block_table.to(torch.int32).to(q.device)
    else:
        cmp_kv_dev = torch.zeros((1, 1, N2, D), dtype=dtype, device=q.device)
        cmp_bt_dev = torch.zeros((B, 1), dtype=torch.int32, device=q.device)
    cmp_block_num, cmp_block_size = cmp_kv_dev.shape[0], cmp_kv_dev.shape[1]
    cmp_table_len = cmp_bt_dev.shape[1]

    func = _get_kernel(
        batch=int(B),
        max_seq=int(S_max),
        total_tokens=int(total_tokens),
        ori_block_num=int(ori_block_num),
        ori_block_size=int(ori_block_size),
        ori_table_len=int(ori_table_len),
        cmp_block_num=int(cmp_block_num),
        cmp_block_size=int(cmp_block_size),
        cmp_table_len=int(cmp_table_len),
        n_heads=N1,
        n_kv_heads=N2,
        head_dim=D,
        topk_cmp=K,
        cmp_ratio=cmp_ratio_eff,
        scenario=scenario,
        ori_win_left=ori_win_left,
        softmax_scale=float(softmax_scale),
        dtype=tl_dtype,
        core_num=core_num,
    )

    sinks_dev = sinks.to(torch.float32).to(q.device)

    out_flat, lse_flat = func(
        q_flat.contiguous(),
        ori_kv_dev.contiguous(),
        ori_bt_dev.contiguous(),
        cmp_kv_dev.contiguous(),
        cmp_bt_dev.contiguous(),
        cmp_indices_dev.contiguous(),
        q_prefix.contiguous(),
        act_q_lens.contiguous(),
        seqused_kv_dev.contiguous(),
        sinks_dev.contiguous(),
    )

    if layout_q == "TND":
        out = out_flat
        lse = lse_flat
    else:
        out = out_flat.reshape(B, S_max, N1, D)
        lse = lse_flat.reshape(B, S_max, N1)

    if return_softmax_lse:
        return out, lse
    return out


if __name__ == "__main__":
    from sparse_flash_mla_golden import (
        FAST_SCENARIOS,
        build_case,
        run_case,
        check_lse,
        check_result,
    )

    dtype = torch.bfloat16
    for name, cfg in FAST_SCENARIOS.items():
        case = build_case(cfg, dtype)
        out, lse = run_case(case, cfg, sparse_flash_mla)
        check_lse(lse.cpu(), case["cpu_ref_lse"], dtype)
        check_result(out.cpu(), case["cpu_ref"])
        print(f"[{name}] passed")
    print("Kernel Output Match!")
