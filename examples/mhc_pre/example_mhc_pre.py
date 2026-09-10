"""MHC Pre operator for Ascend NPU.

Implements the full mHC pre block:
  1. out = x @ fn.T, sqrsum = x^2.sum(-1)   (Kernel A1 + fused A2)
  2. mixes = out * rsqrt(sqrsum / (hc * hidden) + rms_eps)  (fused A2+B1: RMSNorm)
  3. pre/post/comb = split(mixes) + Sinkhorn  (fused B2+B3: split + sinkhorn)
  4. layer_input = sum(residual * pre_mix)    (fused B2+B3: apply pre_mix)

Reference: tilelang main repo CUDA version examples/deepseek_mhc/example_mhc_pre.py

Architecture (3-kernel pipeline; A1 uses Cube, A2+B1 / B2+B3 use dual-V-core):
  Kernel A1 (Cube):       out = x @ fn.T  (K-tiled GEMM, token_block=128, h_blk=512)
  Kernel A2+B1 (Vector):  sqrsum + RMSNorm (fused, sqrsum stays in UB)
  Kernel B2+B3 (Vector):  split + Sinkhorn + apply pre_mix (fused, pre_mix stays in shared/L1)

  Each kernel launched separately from host. Dual-V-core: bid = cid * 2 + vid.
  fn prepack/cache supported for inference (prepare_fn + fn_packed param).
  hc 1-8 (JIT parameter). In-kernel tail via pad_value + TL_ASCEND_TAIL_MASK.
"""

import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
}

MIN_BLOCK = 16
H_BLK = 512
SQRSUM_H_BLK = 4096
TOKEN_BLOCK = 128


def calc_pad(dim, block):
    """Round dim up to the next multiple of block."""
    return max(block, ((dim + block - 1) // block) * block)


# ============================================================
# Kernel A1: GEMM (Cube) + T.Pipelined
# ============================================================


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def mhc_pre_gemm(pad_hc_hidden, pad_hc_mult3, h_blk=H_BLK, token_block=TOKEN_BLOCK, dtype="bfloat16", accum_dtype="float"):
    """Kernel A1: out = x @ fn.T (K-tiled GEMM with accumulation).

    M = token_block (128 in tuned config; Cube alignment >= 16)
    K = pad_hc_hidden (tiled over h_blk)
    N = pad_hc_mult3 (hc_mult3 padded to 32)
    """
    n = T.symbolic("n")
    k_num = T.ceildiv(pad_hc_hidden, h_blk)

    @T.prim_func
    def main(
        x: T.Tensor((n, pad_hc_hidden), dtype),
        fn_t: T.Tensor((pad_hc_hidden, pad_hc_mult3), dtype),
        out: T.Tensor((n, pad_hc_mult3), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(n, token_block), is_npu=True) as (bid, _):
            a_l1 = T.alloc_L1((token_block, h_blk), dtype)
            b_l1 = T.alloc_L1((h_blk, pad_hc_mult3), dtype)
            c_l0 = T.alloc_L0C((token_block, pad_hc_mult3), accum_dtype)

            with T.Scope("C"):
                for i_k in T.Pipelined(k_num, num_stages=2):
                    T.copy(x[bid * token_block, i_k * h_blk], a_l1)
                    T.copy(fn_t[i_k * h_blk, 0], b_l1)
                    if i_k == 0:
                        T.gemm_v0(a_l1, b_l1, c_l0, init=True)
                    else:
                        T.gemm_v0(a_l1, b_l1, c_l0)

                T.copy(c_l0, out[bid * token_block, 0])

    return main


# ============================================================
# Fused Kernel A2+B1: SqrSum + RMSNorm (Vector)
# ============================================================


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def mhc_pre_sqrsum_rmsnorm(
    hc_hidden,
    pad_hc_mult3,
    hc_mult,
    hidden_size,
    rms_eps,
    sqr_h_blk=SQRSUM_H_BLK,
    dtype="bfloat16",
    accum_dtype="float",
):
    """Fused Kernel A2+B1: sqrsum + RMSNorm in one kernel.

    Step 1 (A2): sqrsum = sum(x^2) over hc_hidden, tiled with T.Pipelined.
    Step 2 (B1): mixes = gemm_out * rsqrt(sqrsum / (hc*hidden) + rms_eps).

    Saves 1 kernel launch + sqrsum GM round-trip vs separate A2+B1.
    """
    n = T.symbolic("n")
    sqr_total_tiles = (hc_hidden + sqr_h_blk - 1) // sqr_h_blk
    VEC_NUM = 2

    @T.prim_func
    def main(
        x: T.Tensor((n, hc_hidden), dtype),
        gemm_out: T.Tensor((n, pad_hc_mult3), accum_dtype),
        mixes: T.Tensor((n, pad_hc_mult3), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(n, VEC_NUM), is_npu=True) as (cid, vid):
            bid = cid * VEC_NUM + vid

            if bid < n:
                with T.Scope("V"):
                    acc_ub = T.alloc_ub((sqr_h_blk,), accum_dtype)
                    x_ub = T.alloc_ub((sqr_h_blk,), dtype)
                    x_fp32 = T.alloc_ub((sqr_h_blk,), accum_dtype)
                    x_sq = T.alloc_ub((sqr_h_blk,), accum_dtype)
                    T.tile.fill(acc_ub, 0.0)

                    for i_k in T.Pipelined(sqr_total_tiles, num_stages=2):
                        T.copy(x[bid, i_k * sqr_h_blk], x_ub, pad_value=0.0)
                        T.tile.cast(x_fp32, x_ub, "CAST_NONE", sqr_h_blk)
                        T.tile.mul(x_sq, x_fp32, x_fp32)
                        T.tile.add(acc_ub, acc_ub, x_sq)

                    result_ub = T.alloc_ub(1, accum_dtype)
                    T.reduce_sum(acc_ub, result_ub, dim=-1)

                    rms_ub = T.alloc_ub(1, accum_dtype)
                    inv_sqrt_ub = T.alloc_ub(1, accum_dtype)

                    T.tile.fill(rms_ub, 0.0)
                    T.tile.add(rms_ub, rms_ub, result_ub[0])
                    T.tile.mul(rms_ub, rms_ub, 1.0 / (hc_mult * hidden_size))
                    T.tile.add(rms_ub, rms_ub, rms_eps)
                    T.tile.rsqrt(inv_sqrt_ub, rms_ub)

                    out_ub = T.alloc_ub(pad_hc_mult3, accum_dtype)
                    mixes_ub = T.alloc_ub(pad_hc_mult3, accum_dtype)
                    T.copy(gemm_out[bid, 0], out_ub)
                    T.tile.mul(mixes_ub, out_ub, inv_sqrt_ub[0])
                    T.copy(mixes_ub, mixes[bid, 0])

    return main


# ============================================================
# Fused Kernel B2+B3: Split + Sinkhorn + Apply (Vector)
# ============================================================


@tilelang.jit(out_idx=[5, 6, 7], workspace_idx=[3], pass_configs=pass_configs)
def mhc_pre_split_sinkhorn_apply(
    hc,
    hidden,
    sinkhorn_iters,
    pre_eps,
    sinkhorn_eps,
    hc_post_mult_value=2.0,
    apply_h_blk=2048,
    dtype="float",
    apply_dtype="bfloat16",
    accum_dtype="float",
):
    """Fused Kernel B2+B3: split + sinkhorn + apply pre_mix.

    B2 produces pre_mix in shared, B3 reads it and applies to residual.
    Saves 1 kernel launch + pre_mix GM round-trip.
    """
    n = T.symbolic("n")
    mix_hc = hc * (2 + hc)
    apply_total_tiles = (hidden + apply_h_blk - 1) // apply_h_blk
    apply_pad_h = apply_total_tiles * apply_h_blk

    hc_pad = hc
    if hc * 4 % 32 != 0:
        hc_pad = tilelang.cdiv(hc * 4, 32) * 32 // 4

    @T.prim_func
    def main(
        mixes: T.Tensor([n, mix_hc], dtype),
        hc_scale: T.Tensor([3], dtype),
        hc_base: T.Tensor([mix_hc], dtype),
        workspace: T.Tensor([n, mix_hc], dtype),
        residual: T.Tensor([n, hc, hidden], apply_dtype),
        post: T.Tensor([n, hc], dtype),
        comb: T.Tensor([n, hc, hc], dtype),
        layer_input: T.Tensor([n, apply_pad_h], apply_dtype),
    ):
        with T.Kernel(T.ceildiv(n, 2), is_npu=True) as (cid, vid):
            bid = cid * 2 + vid
            if bid < n:
                mixes_shared = T.alloc_shared(mix_hc, dtype)
                hc_base_shared = T.alloc_shared(mix_hc, dtype)
                hc_scale_ub = T.alloc_ub(mix_hc, dtype)

                comb_shared = T.alloc_shared((hc, hc_pad), dtype)
                pre_shared = T.alloc_shared(hc_pad, dtype)
                post_shared = T.alloc_shared(hc_pad, dtype)

                tmp_shared = T.alloc_shared(hc_pad, dtype)

                row_sum = T.alloc_shared(hc_pad, dtype)
                col_sum = T.alloc_shared((1, hc_pad), dtype)
                row_max = T.alloc_shared(hc_pad, dtype)

                col_broadcast = T.alloc_shared((hc, hc_pad), dtype)
                row_div = T.alloc_shared((hc, hc_pad), dtype)

                alpha_0 = hc_scale[0]
                alpha_1 = hc_scale[1]
                alpha_2 = hc_scale[2]

                for i in T.unroll(hc):
                    hc_scale_ub[i] = alpha_0
                for i in T.unroll(hc):
                    hc_scale_ub[hc + i] = alpha_1
                for i in T.unroll(hc * hc):
                    hc_scale_ub[2 * hc + i] = alpha_2
                T.copy(hc_base, hc_base_shared)
                T.copy(mixes[bid, :], mixes_shared)

                T.tile.mul(mixes_shared, mixes_shared, hc_scale_ub)
                T.tile.add(mixes_shared, mixes_shared, hc_base_shared)
                T.copy(mixes_shared, workspace[bid, :])

                # pre
                T.copy(workspace[bid, :hc], tmp_shared)
                T.tile.sigmoid(pre_shared, tmp_shared)
                T.tile.add(pre_shared, pre_shared, pre_eps)

                # post (read only hc elements, not hc_pad, to avoid OOB when hc_pad > mix_hc - hc)
                T.copy(workspace[bid, hc : 2 * hc], tmp_shared)
                T.tile.sigmoid(post_shared, tmp_shared)
                T.tile.mul(post_shared, post_shared, hc_post_mult_value)
                T.copy(post_shared[:hc], post[bid, :hc])

                # comb
                T.tile.fill(comb_shared, 0.0)
                for i in T.unroll(hc):
                    start = 2 * hc + i * hc
                    end = 2 * hc + i * hc + hc
                    T.copy(workspace[bid, start:end], tmp_shared)
                    T.copy(tmp_shared, comb_shared[i, :])

                # comb = comb.softmax(-1) + eps
                T.reduce_max(comb_shared, row_max, dim=-1, real_shape=[hc, hc])
                for i in T.unroll(hc):
                    T.tile.fill(row_div[i, :], row_max[i])
                T.tile.sub(comb_shared, comb_shared, row_div)
                T.tile.exp(comb_shared, comb_shared)
                T.reduce_sum(comb_shared, row_sum, dim=-1, real_shape=[hc, hc])
                for i in T.unroll(hc):
                    T.tile.fill(row_div[i, :], row_sum[i])
                T.tile.div(comb_shared, comb_shared, row_div)
                T.tile.add(comb_shared, comb_shared, sinkhorn_eps)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_shared, col_sum, dim=0, real_shape=[hc, hc_pad])
                T.tile.add(col_sum, col_sum, sinkhorn_eps)
                T.tile.broadcast(col_broadcast, col_sum)
                T.tile.div(comb_shared, comb_shared, col_broadcast)

                for _ in T.serial(sinkhorn_iters - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(comb_shared, row_sum, dim=-1, real_shape=[hc, hc])
                    T.tile.add(row_sum, row_sum, sinkhorn_eps)
                    for i in T.unroll(hc):
                        T.tile.fill(row_div[i, :], row_sum[i])
                    T.tile.div(comb_shared, comb_shared, row_div)
                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(comb_shared, col_sum, dim=0, real_shape=[hc, hc_pad])
                    T.tile.add(col_sum, col_sum, sinkhorn_eps)
                    T.tile.broadcast(col_broadcast, col_sum)
                    T.tile.div(comb_shared, comb_shared, col_broadcast)

                for i in T.unroll(hc):
                    T.copy(comb_shared[i, :hc], comb[bid, i, :])

                # ---- B3: apply pre_mix (fused, no GM round-trip for pre) ----
                # pre_shared already in shared, copy to UB for AXPY
                pre_ub_apply = T.alloc_ub(hc, accum_dtype)
                for i in T.unroll(hc):
                    pre_ub_apply[i] = pre_shared[i]

                res_ub_apply = T.alloc_ub((hc, apply_h_blk), apply_dtype)
                res_fp32_apply = T.alloc_ub((hc, apply_h_blk), accum_dtype)
                out_fp32_apply = T.alloc_ub(apply_h_blk, accum_dtype)
                out_bf16_apply = T.alloc_ub(apply_h_blk, apply_dtype)

                for i_h in T.Pipelined(apply_total_tiles, num_stages=2):
                    h_start = i_h * apply_h_blk

                    T.copy(residual[bid, 0:hc, h_start : h_start + apply_h_blk], res_ub_apply, pad_value=0.0)
                    T.tile.cast(res_fp32_apply, res_ub_apply, "CAST_NONE", apply_h_blk * hc)

                    T.tile.fill(out_fp32_apply, 0.0)
                    for res_idx in T.unroll(hc):
                        T.tile.axpy(out_fp32_apply, res_fp32_apply[res_idx, :], pre_ub_apply[res_idx])

                    T.tile.cast(out_bf16_apply, out_fp32_apply, "CAST_RINT", apply_h_blk)
                    T.copy(out_bf16_apply, layer_input[bid, h_start : h_start + apply_h_blk])

    return main


# ============================================================
# Host-side adapter
# ============================================================


def _pad_2d(t, target_rows, target_cols):
    """[rows, cols] -> [target_rows, target_cols], padded with zeros."""
    result = torch.zeros(target_rows, target_cols, dtype=t.dtype, device=t.device)
    result[: t.shape[0], : t.shape[1]] = t
    return result


_kernel_cache = {}
# Note: cache key includes float params (rms_eps, pre_eps, etc.).
# Different values trigger separate JIT compilations; acceptable for examples.


def _get_kernel(name, *args):
    """Cache compiled kernel to avoid repeated JIT lookup."""
    key = (name, *args)
    if key not in _kernel_cache:
        _kernel_cache[key] = _KERNEL_BUILDERS[name](*args)
    return _kernel_cache[key]


_KERNEL_BUILDERS = {
    "gemm": mhc_pre_gemm,
    "sqrsum_rmsnorm": mhc_pre_sqrsum_rmsnorm,
    "sinkhorn_apply": mhc_pre_split_sinkhorn_apply,
}


def prepare_fn(fn, hc_mult):
    """Prepack fn for mhc_pre: fp32 -> bf16 -> transpose -> pad.

    Call once at model init, reuse across inference.
    """
    hc_mult3 = hc_mult * (2 + hc_mult)
    pad_hc_mult3 = calc_pad(hc_mult3, MIN_BLOCK)
    pad_hc_hidden = calc_pad(fn.shape[1], H_BLK)

    fn_t = fn.bfloat16().T.contiguous()
    if pad_hc_hidden == fn.shape[1] and pad_hc_mult3 == hc_mult3:
        return fn_t
    else:
        return _pad_2d(fn_t, pad_hc_hidden, pad_hc_mult3)


def mhc_pre_gemm_sqrsum(x, fn, hc_mult, fn_packed=None):
    """Kernel A1 host adapter: GEMM only (sqrsum fused into A2+B1).

    Args:
        x:  [n, hc*hidden] bf16
        fn: [hc_mult3, hc*hidden] fp32
        hc_mult: int

    Returns:
        out:    [n, hc_mult3] fp32 (padded)
    """
    n = x.shape[0]
    hc_hidden = x.shape[1]
    hc_mult3 = hc_mult * (2 + hc_mult)

    pad_hc_mult3 = calc_pad(hc_mult3, MIN_BLOCK)
    pad_hc_hidden = calc_pad(hc_hidden, H_BLK)
    pad_n = calc_pad(n, TOKEN_BLOCK)

    if pad_n == n and pad_hc_hidden == hc_hidden:
        x_padded = x
    else:
        x_padded = _pad_2d(x, pad_n, pad_hc_hidden)

    if fn_packed is not None:
        fn_t_padded = fn_packed
    else:
        fn_t = fn.bfloat16().T.contiguous()
        if pad_hc_hidden == hc_hidden and pad_hc_mult3 == hc_mult3:
            fn_t_padded = fn_t
        else:
            fn_t_padded = _pad_2d(fn_t, pad_hc_hidden, pad_hc_mult3)

    gemm_kernel = _get_kernel("gemm", pad_hc_hidden, pad_hc_mult3)
    out_padded = gemm_kernel(x_padded, fn_t_padded)

    return out_padded


def mhc_pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, fn_packed=None):
    """Full mHC pre block host adapter.

    Args:
        residual:  [n, hc, hidden] bf16
        fn:        [hc_mult3, hc*hidden] fp32
        hc_scale:  [3] fp32
        hc_base:   [hc_mult3] fp32
        rms_eps, hc_pre_eps, hc_sinkhorn_eps: float
        hc_post_mult_value: float (post_mix multiplier, passed to kernel)
        sinkhorn_repeat: int

    Returns:
        post_mix:     [n, hc, 1]   fp32
        comb_mix:     [n, hc, hc]  fp32
        layer_input:  [n, hidden]  bf16

    Note:
        Kernel B3 (apply_mix) is fused into B2+B3, uses AXPY with hc as
        JIT parameter (1-8). Passing hc outside [1, 8] will raise an assertion error.
    """
    hc_mult = residual.shape[1]
    assert 1 <= hc_mult <= 8, f"hc must be in [1, 8] (tested range), got hc={hc_mult}"
    hidden = residual.shape[2]
    n = residual.shape[0]
    hc_mult3 = hc_mult * (2 + hc_mult)

    x_flat = residual.view(n, hc_mult * hidden)

    # Kernel A1: GEMM only
    gemm_out_padded = mhc_pre_gemm_sqrsum(x_flat, fn, hc_mult, fn_packed)

    # Fused Kernel A2+B1: sqrsum + RMSNorm
    pad_hc_mult3 = calc_pad(hc_mult3, MIN_BLOCK)
    gemm_out = gemm_out_padded[:n, :pad_hc_mult3].contiguous()
    sqrsum_rmsnorm_kernel = _get_kernel("sqrsum_rmsnorm", hc_mult * hidden, pad_hc_mult3, hc_mult, hidden, rms_eps)
    mixes_padded = sqrsum_rmsnorm_kernel(x_flat, gemm_out)
    mixes = mixes_padded[:n, :hc_mult3].contiguous()

    # Fused Kernel B2+B3: sinkhorn + apply pre_mix
    sinkhorn_apply_kernel = _get_kernel("sinkhorn_apply", hc_mult, hidden, sinkhorn_repeat, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value)
    post_mix, comb_mix, layer_input_padded = sinkhorn_apply_kernel(mixes, hc_scale, hc_base, residual)

    # B2 outputs padded shapes (hc_pad), trim to actual hc
    post_mix = post_mix[:, :hc_mult].contiguous()
    comb_mix = comb_mix[:, :, :hc_mult].contiguous()

    layer_input = layer_input_padded[:n, :hidden]

    post_mix_out = post_mix.unsqueeze(-1)
    return post_mix_out, comb_mix, layer_input


# ============================================================
# Golden reference
# ============================================================


def sinkhorn_normalize_ref(x, repeat, eps):
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_ref(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat):
    """PyTorch golden for full mhc_pre, following the same bf16 path as kernel."""
    hc_mult = residual.shape[1]
    hidden = residual.shape[2]
    n = residual.shape[0]

    residual_flat = residual.view(n, hc_mult * hidden).float()
    fn_bf16 = fn.bfloat16()
    out = residual_flat @ fn_bf16.float().T
    sqrsum = residual_flat.square().sum(-1)

    rms = (sqrsum / (hc_mult * hidden) + rms_eps).rsqrt()
    mixes = out * rms.unsqueeze(-1)

    hc_scale_exp = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ]
    )
    mixes = mixes * hc_scale_exp + hc_base

    pre_mix = mixes[:, :hc_mult].sigmoid() + hc_pre_eps
    post_mix = mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)

    res_mix = sinkhorn_normalize_ref(res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps)

    layer_input = (residual.float() * pre_mix.unsqueeze(-1)).sum(-2).bfloat16()

    return post_mix.unsqueeze(-1), res_mix, layer_input


# ============================================================
# Tests
# ============================================================


def generate_full_test_data(
    n, h, hc_mult, device="npu", rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult_value=1.0, sinkhorn_repeat=10
):
    torch.random.manual_seed(42)
    hc_mult3 = hc_mult * (2 + hc_mult)

    residual = (
        torch.randn((n, hc_mult, h), dtype=torch.float, device=device)
        .mul(1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
        .bfloat16()
    )
    fn = (
        torch.randn((hc_mult3, hc_mult, h), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1

    return {
        "residual": residual,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "rms_eps": rms_eps,
        "hc_pre_eps": hc_pre_eps,
        "hc_sinkhorn_eps": hc_sinkhorn_eps,
        "hc_post_mult_value": hc_post_mult_value,
        "sinkhorn_repeat": sinkhorn_repeat,
    }


def _check_output(name, tl, ref, all_passed):
    try:
        torch.testing.assert_close(tl.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)
        diff = (tl.cpu().float() - ref.cpu().float()).abs()
        print(f"  {name} PASSED (max_diff={diff.max().item():.6f})")
    except AssertionError:
        diff = (tl.cpu().float() - ref.cpu().float()).abs()
        print(f"  {name} FAILED (max_diff={diff.max().item():.6f})")
        all_passed[0] = False


def test_full():
    print("=" * 60)
    print("MHC Pre full pipeline test (Ascend NPU)")
    print("=" * 60)

    test_cases = [
        (4, 128, 4),
        (16, 256, 4),
        (4, 1280, 4),
        (512, 2560, 4),
        (4096, 2560, 4),
        (4, 100, 4),
        (4, 128, 1),
        (4, 128, 2),
        (4, 128, 3),
        (4, 128, 8),
        (4, 100, 8),
    ]

    all_passed = [True]
    for n, h, hc_mult in test_cases:
        print(f"\n--- n={n}, h={h}, hc_mult={hc_mult} ---")
        data = generate_full_test_data(n, h, hc_mult)
        post_tl, comb_tl, layer_tl = mhc_pre(**data)
        post_ref, comb_ref, layer_ref = mhc_pre_ref(**data)
        print(f"  post_mix={post_tl.shape}, comb_mix={comb_tl.shape}, layer_input={layer_tl.shape}")
        _check_output("post_mix", post_tl, post_ref, all_passed)
        _check_output("comb_mix", comb_tl, comb_ref, all_passed)
        _check_output("layer_input", layer_tl, layer_ref, all_passed)

    # Distinct parameter routing test (pre_eps != sinkhorn_eps, post_mult != 1.0/2.0)
    print("\n--- distinct params: hc_pre_eps=1e-4, hc_sinkhorn_eps=3e-3, hc_post_mult_value=1.7 ---")
    data = generate_full_test_data(4, 128, 4, hc_pre_eps=1e-4, hc_sinkhorn_eps=3e-3, hc_post_mult_value=1.7, sinkhorn_repeat=3)
    post_tl, comb_tl, layer_tl = mhc_pre(**data)
    post_ref, comb_ref, layer_ref = mhc_pre_ref(**data)
    _check_output("post_mix", post_tl, post_ref, all_passed)
    _check_output("comb_mix", comb_tl, comb_ref, all_passed)
    _check_output("layer_input", layer_tl, layer_ref, all_passed)

    print("\n" + "=" * 60)
    if all_passed[0]:
        print("Kernel Output Match!")
    else:
        print("Some tests failed.")
    print("=" * 60)


if __name__ == "__main__":
    tilelang.disable_cache()
    test_full()
