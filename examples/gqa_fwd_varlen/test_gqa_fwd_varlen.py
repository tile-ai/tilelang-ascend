import argparse
import os
import sys

import torch

import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gqa_fwd_varlen import (  # noqa: E402
    flashattn,
    generate_random_padding_mask,
    mask_to_cu_seqlens,
    build_attention_mask,
)


# ===========================================================================
# Golden functions (padded layout)
# 1. ref_gqa_varlen_fwd_padded: self-written PyTorch golden (no flash_attn dep)
# 2. ref_sdpa_padded: F.scaled_dot_product_attention golden — the NPU
#    equivalent of the GPU main-repo golden flash_attn.flash_attn_varlen_func.
#    flash_attn is CUDA-only and unavailable on NPU; SDPA is PyTorch's native
#    attention and runs on NPU, providing an independent cross-validation path.
# ===========================================================================


def ref_gqa_varlen_fwd_padded(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    heads,
    groups,
    dim,
    is_causal,
):
    """PyTorch reference for padded GQA forward. Padding positions output 0.

    Args:
        q: [batch, heads, q_seqlen, dim] float16 (padded)
        k: [batch, head_kv, k_seqlen, dim] float16 (padded)
        v: [batch, head_kv, k_seqlen, dim] float16 (padded)
        cu_seqlens_q: [batch+1] int32 (actual Q lengths prefix sum)
        cu_seqlens_k: [batch+1] int32 (actual K lengths prefix sum)

    Returns:
        output: [batch, heads, q_seqlen, dim] float16 (padded, padding rows = 0)
    """
    scale = (1.0 / dim) ** 0.5
    output = torch.zeros_like(q)
    batch = q.shape[0]

    for b in range(batch):
        sq = int(cu_seqlens_q[b + 1].item()) - int(cu_seqlens_q[b].item())
        skv = int(cu_seqlens_k[b + 1].item()) - int(cu_seqlens_k[b].item())
        if sq == 0:
            continue

        q_b = q[b, :, :sq, :].float()  # [heads, sq, dim]
        k_b = k[b, :, :skv, :].float()  # [head_kv, skv, dim]
        v_b = v[b, :, :skv, :].float()  # [head_kv, skv, dim]

        # GQA: repeat KV heads to match Q heads
        k_b_rep = k_b.repeat_interleave(groups, dim=0)  # [heads, skv, dim]
        v_b_rep = v_b.repeat_interleave(groups, dim=0)  # [heads, skv, dim]

        # [heads, sq, skv] = [heads, sq, dim] @ [heads, dim, skv]
        scores = torch.einsum("hqd,hdk->hqk", q_b, k_b_rep.transpose(1, 2)) * scale

        if is_causal:
            q_idx = torch.arange(sq, device=q.device)
            k_idx = torch.arange(skv, device=q.device)
            offset = skv - sq
            mask = q_idx[:, None] + offset < k_idx[None, :]  # [sq, skv]
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # [heads, sq, skv]
        # Invisible Q rows (causal, q_pos+offset<0) have all -inf scores ->
        # softmax returns NaN. Replace with 0 so the golden output is clean
        # (these rows are also excluded from comparison via visible_q_mask).
        attn = torch.nan_to_num(attn, nan=0.0)
        out_b = torch.einsum("hqk,hkd->hqd", attn, v_b_rep)  # [heads, sq, dim]
        output[b, :, :sq, :] = out_b.to(q.dtype)

    return output


def ref_sdpa_padded(q, k, v, cu_seqlens_q, cu_seqlens_k, heads, groups, dim, is_causal):
    """PyTorch SDPA golden (mathematically equivalent to flash_attn_varlen_func).

    Uses F.scaled_dot_product_attention as the golden. This is the NPU
    equivalent of the GPU main-repo golden ``flash_attn.flash_attn_varlen_func``,
    which is CUDA-only and unavailable on NPU. SDPA is PyTorch's native scaled
    dot-product attention implementation and runs on NPU, providing an
    independent cross-validation path against the self-written golden.

    Note: ``flash_attn_varlen_func`` uses **right-aligned** causal masking
    (offset = skv - sq): the last query token attends to the last key token.
    SDPA's ``is_causal=True`` uses **left-aligned** masking (standard lower
    triangular), which only matches when sq == skv. For sq != skv + causal,
    we construct the right-aligned mask manually and pass it as ``attn_mask``
    to SDPA, ensuring mathematical equivalence with
    ``flash_attn_varlen_func`` for all shape combinations.

    Args:
        q: [batch, heads, q_seqlen, dim] float16 (padded)
        k: [batch, head_kv, k_seqlen, dim] float16 (padded)
        v: [batch, head_kv, k_seqlen, dim] float16 (padded)
        cu_seqlens_q: [batch+1] int32 (actual Q lengths prefix sum)
        cu_seqlens_k: [batch+1] int32 (actual K lengths prefix sum)
        heads: number of Q heads.
        groups: GQA group size (heads // head_kv).
        dim: head dimension.
        is_causal: whether to apply causal mask.

    Returns:
        output: [batch, heads, q_seqlen, dim] float16 (padded, padding rows = 0)
    """
    out = torch.zeros_like(q)
    batch = q.shape[0]
    for b in range(batch):
        sq = int(cu_seqlens_q[b + 1].item()) - int(cu_seqlens_q[b].item())
        skv = int(cu_seqlens_k[b + 1].item()) - int(cu_seqlens_k[b].item())
        if sq == 0:
            continue
        q_b = q[b, :, :sq, :].float()  # [heads, sq, dim]
        k_b = k[b, :, :skv, :].float()  # [head_kv, skv, dim]
        v_b = v[b, :, :skv, :].float()  # [head_kv, skv, dim]

        # GQA: repeat KV heads to match Q heads
        k_b_rep = k_b.repeat_interleave(groups, dim=0)  # [heads, skv, dim]
        v_b_rep = v_b.repeat_interleave(groups, dim=0)  # [heads, skv, dim]

        # SDPA expects [batch=1, heads, sq, dim]
        q_b_4d = q_b.unsqueeze(0)
        k_b_4d = k_b_rep.unsqueeze(0)
        v_b_4d = v_b_rep.unsqueeze(0)

        if is_causal and sq == skv:
            # Standard causal mask == right-aligned when offset = 0
            out_b = torch.nn.functional.scaled_dot_product_attention(
                q_b_4d,
                k_b_4d,
                v_b_4d,
                is_causal=True,
            )  # [1, heads, sq, dim]
        elif is_causal:
            # Right-aligned causal: query i attends to key j iff j <= i + offset.
            # flash_attn_varlen_func uses this convention; SDPA's is_causal=True
            # would give left-aligned (j <= i), which differs when sq != skv.
            offset = skv - sq
            q_idx = torch.arange(sq, device=q.device)
            k_idx = torch.arange(skv, device=q.device)
            visible = k_idx[None, :] <= q_idx[:, None] + offset  # [sq, skv]
            attn_mask = torch.zeros(sq, skv, device=q.device, dtype=torch.float32)
            attn_mask[~visible] = float("-inf")
            out_b = torch.nn.functional.scaled_dot_product_attention(
                q_b_4d,
                k_b_4d,
                v_b_4d,
                attn_mask=attn_mask,
            )  # [1, heads, sq, dim]
        else:
            out_b = torch.nn.functional.scaled_dot_product_attention(
                q_b_4d,
                k_b_4d,
                v_b_4d,
            )  # [1, heads, sq, dim]

        out[b, :, :sq, :] = out_b[0].to(q.dtype)

    return out


# ===========================================================================
# Test helpers
# ===========================================================================


def _prepare_and_run(
    batch,
    heads,
    groups,
    q_seqlen,
    k_seqlen,
    dim,
    is_causal,
    padding_mode,
    block_M,
    block_N,
    device,
    dtype,
    atol,
    rtol,
):
    """Prepare padded inputs + mask, run kernel + dual golden, return (max_diff, golden_diff, passed).

    Runs both the self-written golden (ref_gqa_varlen_fwd_padded) and the SDPA
    golden (ref_sdpa_padded, equivalent to the main-repo flash_attn_varlen_func).
    The test passes only if the kernel matches BOTH goldens within tolerance.
    golden_diff reports the max difference between the two goldens (should be
    small, confirming their mathematical equivalence).
    """
    torch.manual_seed(0)
    head_kv = heads // groups

    # Pad seqlens to block_M/block_N multiples to avoid GM OOB reads.
    # Padding rows/cols are zero-filled; mask=0 handles them in kernel.
    padded_sq = ((q_seqlen + block_M - 1) // block_M) * block_M
    padded_skv = ((k_seqlen + block_N - 1) // block_N) * block_N

    # Padded 4D layout: [batch, heads, padded_seqlen, dim]
    q = torch.zeros(batch, heads, padded_sq, dim, dtype=dtype, device=device)
    q[:, :, :q_seqlen, :] = torch.randn(batch, heads, q_seqlen, dim, dtype=dtype, device=device)
    k = torch.zeros(batch, head_kv, padded_skv, dim, dtype=dtype, device=device)
    k[:, :, :k_seqlen, :] = torch.randn(batch, head_kv, k_seqlen, dim, dtype=dtype, device=device)
    v = torch.zeros(batch, head_kv, padded_skv, dim, dtype=dtype, device=device)
    v[:, :, :k_seqlen, :] = torch.randn(batch, head_kv, k_seqlen, dim, dtype=dtype, device=device)

    # Padding masks (original seqlen) -> cu_seqlens -> attention mask tensor (padded seqlens)
    q_mask = generate_random_padding_mask(q_seqlen, batch, device, mode=padding_mode)
    k_mask = generate_random_padding_mask(k_seqlen, batch, device, mode=padding_mode)
    cu_seqlens_q = mask_to_cu_seqlens(q_mask)
    cu_seqlens_k = mask_to_cu_seqlens(k_mask)
    attn_mask = build_attention_mask(
        cu_seqlens_q,
        cu_seqlens_k,
        padded_sq,
        padded_skv,
        is_causal,
        device,
    )

    # Compile kernel with padded seqlens
    # Skip mask only when non-causal + full padding + no block padding needed
    # (mask is all 1.0 only when seq lens are exact multiples of block sizes)
    has_block_padding = (q_seqlen % block_M != 0) or (k_seqlen % block_N != 0)
    apply_mask = is_causal or padding_mode != "full" or has_block_padding
    kernel = flashattn(
        batch,
        groups,
        heads,
        dim,
        padded_sq,
        padded_skv,
        is_causal,
        block_M=block_M,
        block_N=block_N,
        apply_mask=apply_mask,
    )

    # Run kernel
    out = kernel(q, k, v, attn_mask)
    torch.npu.synchronize()

    # Golden 1: self-written PyTorch (padded layout)
    ref_out = ref_gqa_varlen_fwd_padded(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        heads,
        groups,
        dim,
        is_causal,
    )
    torch.npu.synchronize()

    # Golden 2: SDPA (F.scaled_dot_product_attention) — NPU equivalent of the
    # GPU main-repo golden flash_attn.flash_attn_varlen_func.
    ref_sdpa_out = ref_sdpa_padded(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        heads,
        groups,
        dim,
        is_causal,
    )
    torch.npu.synchronize()

    # Build the comparison mask: valid Q rows that are ALSO visible under
    # causal (q_pos + offset >= 0). Invisible rows (causal, q longer than k
    # gives offset<0 -> early Q rows see no KV) produce NaN/0 in both kernel
    # and golden and are excluded.
    visible_q_mask = q_mask.clone()
    if is_causal:
        for b in range(batch):
            q_len_b = int(cu_seqlens_q[b + 1].item()) - int(cu_seqlens_q[b].item())
            kv_len_b = int(cu_seqlens_k[b + 1].item()) - int(cu_seqlens_k[b].item())
            offset_b = kv_len_b - q_len_b
            if offset_b < 0:
                invisible_count = -offset_b
                if invisible_count > 0:
                    visible_q_mask[b, :invisible_count] = False

    # Extend visible_q_mask to padded_sq (False for block-padding rows)
    if padded_sq > q_seqlen:
        vqm_padded = torch.zeros(batch, padded_sq, dtype=torch.bool, device=device)
        vqm_padded[:, :q_seqlen] = visible_q_mask
        visible_q_mask = vqm_padded

    # Compare only visible Q positions.
    # out / ref_out / ref_sdpa_out: [batch, heads, padded_sq, dim].
    out_perm = out.permute(0, 2, 1, 3).contiguous()  # [batch, padded_sq, heads, dim]
    ref_perm = ref_out.permute(0, 2, 1, 3).contiguous()
    ref_sdpa_perm = ref_sdpa_out.permute(0, 2, 1, 3).contiguous()
    out_valid = out_perm[visible_q_mask].cpu()  # [num_visible, heads, dim]
    ref_valid = ref_perm[visible_q_mask].cpu()
    ref_sdpa_valid = ref_sdpa_perm[visible_q_mask].cpu()

    # Guard against NaN leaking into valid rows (should not happen).
    if torch.isnan(out_valid).any():
        return float("nan"), float("nan"), False

    # max_diff: kernel vs self-written golden
    max_diff = (out_valid.float() - ref_valid.float()).abs().max().item()
    # golden_diff: self-written golden vs SDPA golden (cross-validation)
    golden_diff = (ref_valid.float() - ref_sdpa_valid.float()).abs().max().item()

    # Test passes only if kernel matches BOTH goldens within tolerance.
    try:
        torch.testing.assert_close(out_valid, ref_valid, rtol=rtol, atol=atol)
        torch.testing.assert_close(out_valid, ref_sdpa_valid, rtol=rtol, atol=atol)
        passed = True
    except AssertionError:
        passed = False

    return max_diff, golden_diff, passed


# ===========================================================================
# L0 gate tests (DESIGN.md §11.2)
# ===========================================================================


def test_gqa_fwd_varlen_l0():
    """L0 gate tests: regular shapes (block-aligned), for precision convergence."""
    device = "npu"
    dtype = torch.float16
    atol, rtol = 1e-2, 1e-2
    block_M, block_N = 128, 128

    # (name, batch, heads, groups, q_seqlen, k_seqlen, dim, is_causal, padding_mode)
    configs = [
        ("l0_min_full_nc", 1, 4, 2, 128, 128, 128, False, "full"),
        ("l0_min_full_c", 1, 4, 2, 128, 128, 128, True, "full"),
        ("l0_small_rand_nc", 2, 8, 4, 128, 128, 128, False, "random"),
        ("l0_default_full_nc", 8, 64, 16, 2048, 2048, 128, False, "full"),
        ("l0_default_full_c", 8, 64, 16, 2048, 2048, 128, True, "full"),
        # Main-repo-aligned case: same shape + padding mode as the GPU main-repo
        # example_gqa_fwd_varlen.py main() defaults (batch=8, heads=64, groups=16,
        # q_seqlen=2048, k_seqlen=2048, dim=128, is_causal=False, random padding).
        ("l0_main_repo_match", 8, 64, 16, 2048, 2048, 128, False, "random"),
    ]

    ok = True
    for name, b, h, g, sq, skv, d, causal, pmode in configs:
        try:
            max_diff, golden_diff, passed = _prepare_and_run(
                b,
                h,
                g,
                sq,
                skv,
                d,
                causal,
                pmode,
                block_M,
                block_N,
                device,
                dtype,
                atol,
                rtol,
            )
            if passed:
                print(
                    f"[PRECISION_PASS] l0 {name} batch={b} heads={h} groups={g} "
                    f"sq={sq} skv={skv} dim={d} causal={causal} pad={pmode} "
                    f"max_diff={max_diff:.6e} golden_diff={golden_diff:.6e}"
                )
            else:
                print(
                    f"[PRECISION_FAIL] l0 {name} batch={b} heads={h} groups={g} "
                    f"sq={sq} skv={skv} dim={d} causal={causal} pad={pmode} "
                    f"max_diff={max_diff:.6e} golden_diff={golden_diff:.6e}"
                )
                ok = False
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(
                f"[PRECISION_FAIL] l0 {name} batch={b} heads={h} groups={g} sq={sq} skv={skv} dim={d} causal={causal} pad={pmode} error={e}"
            )
            ok = False
    return ok


# ===========================================================================
# L1 / L2 / Boundary (expanded by tilelang-op-test-design scenario B)
# ===========================================================================


def test_gqa_fwd_varlen_l1():
    """L1 functional tests: irregular shapes, tail blocks, q!=k, GQA variants.

    Returns True iff all cases pass (blocking).
    """
    device = "npu"
    dtype = torch.float16
    atol, rtol = 1e-2, 1e-2
    block_M, block_N = 128, 128

    # (name, batch, heads, groups, q_seqlen, k_seqlen, dim, is_causal, padding_mode)
    # Avoid sq > skv + causal (would create invisible Q rows -> NaN in both
    # kernel and golden, which cannot be compared).
    configs = [
        ("l1_irregular_nc", 1, 4, 2, 100, 100, 128, False, "random"),  # tail block + varlen
        ("l1_irregular_c", 1, 4, 2, 100, 100, 128, True, "random"),  # tail + causal + varlen
        ("l1_q_short_k_c", 2, 4, 2, 64, 128, 128, True, "full"),  # q<k + causal (offset>0)
        ("l1_gqa1_c", 1, 4, 4, 128, 128, 128, True, "full"),  # head_kv=1 (full share)
        ("l1_multi_rand_c", 3, 8, 4, 256, 256, 128, True, "random"),  # multi-batch + rand + causal
        ("l1_tail_nc", 1, 4, 2, 33, 65, 128, False, "full"),  # extreme tail (rem 1)
    ]

    ok = True
    for name, b, h, g, sq, skv, d, causal, pmode in configs:
        try:
            max_diff, golden_diff, passed = _prepare_and_run(
                b,
                h,
                g,
                sq,
                skv,
                d,
                causal,
                pmode,
                block_M,
                block_N,
                device,
                dtype,
                atol,
                rtol,
            )
            if passed:
                print(
                    f"[PRECISION_PASS] l1 {name} batch={b} heads={h} groups={g} "
                    f"sq={sq} skv={skv} dim={d} causal={causal} pad={pmode} "
                    f"max_diff={max_diff:.6e} golden_diff={golden_diff:.6e}"
                )
            else:
                print(
                    f"[PRECISION_FAIL] l1 {name} batch={b} heads={h} groups={g} "
                    f"sq={sq} skv={skv} dim={d} causal={causal} pad={pmode} "
                    f"max_diff={max_diff:.6e} golden_diff={golden_diff:.6e}"
                )
                ok = False
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(
                f"[PRECISION_FAIL] l1 {name} batch={b} heads={h} groups={g} sq={sq} skv={skv} dim={d} causal={causal} pad={pmode} error={e}"
            )
            ok = False
    return ok


def _run_boundary_case(name, batch, heads, groups, q_seqlen, k_seqlen, dim, is_causal, padding_mode, input_scale, block_M, block_N):
    """Run one L2/Boundary case. Non-blocking: prints [BOUNDARY_PASS/WARN]."""
    device = "npu"
    dtype = torch.float16
    atol, rtol = 1e-2, 1e-2
    head_kv = heads // groups
    try:
        torch.manual_seed(0)
        # Pad seqlens to block_M/block_N multiples to avoid GM OOB reads
        padded_sq = ((q_seqlen + block_M - 1) // block_M) * block_M
        padded_skv = ((k_seqlen + block_N - 1) // block_N) * block_N
        q = torch.zeros(batch, heads, padded_sq, dim, dtype=dtype, device=device)
        q[:, :, :q_seqlen, :] = torch.randn(batch, heads, q_seqlen, dim, dtype=dtype, device=device) * input_scale
        k = torch.zeros(batch, head_kv, padded_skv, dim, dtype=dtype, device=device)
        k[:, :, :k_seqlen, :] = torch.randn(batch, head_kv, k_seqlen, dim, dtype=dtype, device=device) * input_scale
        v = torch.zeros(batch, head_kv, padded_skv, dim, dtype=dtype, device=device)
        v[:, :, :k_seqlen, :] = torch.randn(batch, head_kv, k_seqlen, dim, dtype=dtype, device=device) * input_scale
        q_mask = generate_random_padding_mask(q_seqlen, batch, device, mode=padding_mode)
        k_mask = generate_random_padding_mask(k_seqlen, batch, device, mode=padding_mode)
        cu_seqlens_q = mask_to_cu_seqlens(q_mask)
        cu_seqlens_k = mask_to_cu_seqlens(k_mask)
        attn_mask = build_attention_mask(
            cu_seqlens_q,
            cu_seqlens_k,
            padded_sq,
            padded_skv,
            is_causal,
            device,
        )
        kernel = flashattn(
            batch,
            groups,
            heads,
            dim,
            padded_sq,
            padded_skv,
            is_causal,
            block_M=block_M,
            block_N=block_N,
            apply_mask=(is_causal or padding_mode != "full" or (q_seqlen % block_M != 0) or (k_seqlen % block_N != 0)),
        )
        out = kernel(q, k, v, attn_mask)
        torch.npu.synchronize()
        ref_out = ref_gqa_varlen_fwd_padded(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            heads,
            groups,
            dim,
            is_causal,
        )
        torch.npu.synchronize()
        # SDPA golden (cross-validation, equivalent to main-repo flash_attn)
        ref_sdpa_out = ref_sdpa_padded(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            heads,
            groups,
            dim,
            is_causal,
        )
        torch.npu.synchronize()
        # Extend q_mask to padded_sq for comparison
        if padded_sq > q_seqlen:
            qm_padded = torch.zeros(batch, padded_sq, dtype=torch.bool, device=device)
            qm_padded[:, :q_seqlen] = q_mask
            q_mask = qm_padded
        out_perm = out.permute(0, 2, 1, 3).contiguous()[q_mask].cpu()
        ref_perm = ref_out.permute(0, 2, 1, 3).contiguous()[q_mask].cpu()
        ref_sdpa_perm = ref_sdpa_out.permute(0, 2, 1, 3).contiguous()[q_mask].cpu()
        if torch.isnan(out_perm).any():
            print(f"[BOUNDARY_WARN] boundary {name}: NaN in valid output")
            return
        max_diff = (out_perm.float() - ref_perm.float()).abs().max().item()
        golden_diff = (ref_perm.float() - ref_sdpa_perm.float()).abs().max().item()
        max_diff_sdpa = (out_perm.float() - ref_sdpa_perm.float()).abs().max().item()
        torch.testing.assert_close(out_perm, ref_perm, rtol=rtol, atol=atol)
        print(f"[BOUNDARY_PASS] boundary {name} max_diff={max_diff:.6e} golden_diff={golden_diff:.6e} max_diff_sdpa={max_diff_sdpa:.6e}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary {name}: {e}")


def test_gqa_fwd_varlen_l2():
    """L2 abnormal input tests. Non-blocking: prints [BOUNDARY_PASS/WARN]."""
    block_M, block_N = 128, 128
    # (name, batch, heads, groups, sq, skv, dim, causal, pad, scale)
    cases = [
        ("l2_single_token", 1, 4, 2, 1, 1, 128, False, "full", 1.0),
        ("l2_min_seqlen", 1, 4, 2, 32, 32, 128, False, "full", 1.0),
        ("l2_batch1_head1", 1, 1, 1, 64, 64, 128, False, "full", 1.0),
    ]
    for name, b, h, g, sq, skv, d, causal, pmode, scale in cases:
        _run_boundary_case(name, b, h, g, sq, skv, d, causal, pmode, scale, block_M, block_N)


def test_gqa_fwd_varlen_boundary():
    """Boundary / special value tests. Non-blocking: prints [BOUNDARY_PASS/WARN]."""
    block_M, block_N = 128, 128
    # (name, batch, heads, groups, sq, skv, dim, causal, pad, scale)
    cases = [
        ("zero_input", 1, 4, 2, 128, 128, 128, False, "full", 0.0),  # all-zero Q/K/V
        ("large_input", 1, 4, 2, 128, 128, 128, False, "full", 10.0),  # large values (stability)
    ]
    for name, b, h, g, sq, skv, d, causal, pmode, scale in cases:
        _run_boundary_case(name, b, h, g, sq, skv, d, causal, pmode, scale, block_M, block_N)


def main():
    parser = argparse.ArgumentParser(description="GQA varlen Flash Attention (Ascend Expert)")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all"],
        help="Test level to run",
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(0)

    blocking_ok = True  # Only L0/L1 count toward blocking

    if args.level in ("l0", "all"):
        blocking_ok &= test_gqa_fwd_varlen_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_gqa_fwd_varlen_l1()
    if args.level in ("l2", "all"):
        test_gqa_fwd_varlen_l2()
    if args.level in ("boundary", "all"):
        test_gqa_fwd_varlen_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
