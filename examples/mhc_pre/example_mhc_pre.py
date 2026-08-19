"""MHC Pre operator for Ascend NPU.

Implements the full mHC pre block:
  1. out = x @ fn.T, sqrsum = x^2.sum(-1)   (Kernel A)
  2. mixes = out * rsqrt(sqrsum/h + eps)      (Kernel B1: RMSNorm)
  3. pre/post/comb = split(mixes) + Sinkhorn  (Kernel B2: split + sinkhorn)
  4. layer_input = sum(residual * pre_mix)    (Kernel B3: apply pre_mix)

Reference: tilelang main repo CUDA version examples/deepseek_mhc/example_mhc_pre.py

Architecture (multi-kernel, same pattern as mhc_post):
  Kernel A1 (Cube):   out = x @ fn.T  (K-tiled GEMM with accumulation)
  Kernel A2 (Vector): sqrsum = sum(x^2)  (tiled reduction)
  Kernel B1 (Vector): mixes = out * rsqrt(sqrsum/(hc*h) + rms_eps)  (RMSNorm)
  Kernel B2 (Vector): mixes -> pre/post/comb + Sinkhorn normalization
                      (adapted from examples/deepseek_v4/hc_split_sinkhorn.py)
  Kernel B3 (Vector): layer_input = sum over hc of (residual * pre_mix)  (apply pre_mix)

  Each kernel launched separately from host. Avoids CV synchronization issues.

Migration from CUDA:
  1. pass_configs: TL_ASCEND_AUTO_SYNC / MEMORY_PLANNING / AUTO_CV_COMBINE
  2. T.gemm_v0: bf16 input + fp32 accumulate (CUDA used TF32 T.gemm)
  3. fn cast to bf16 on host (CUDA used tfloat32)
  4. token_block=16 (Cube minimum, CUDA used 32)
  5. T.clear -> T.tile.fill(buf, 0.0) or gemm_v0 init=(k==0) (T.clear not on Ascend)
  6. sqrsum: separate Vector kernel (CUDA fused sqrsum into GEMM kernel)
  7. T.Pipelined for K-loop (num_stages=2)
  8. CUDA thread-binding warp split -> separate kernels on Ascend
  9. Sinkhorn adapted from examples/deepseek_v4/hc_split_sinkhorn.py (Ascend-verified)
"""

import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
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

    M = token_block (16, Cube minimum)
    K = pad_hc_hidden (tiled over h_blk)
    N = pad_hc_mult3 (hc_mult3 padded to 32)
    """
    n = T.symbolic("n")
    k_num = T.ceildiv(pad_hc_hidden, h_blk)
    k_num_int = (pad_hc_hidden + h_blk - 1) // h_blk

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
                if k_num_int > 1:
                    for i_k in T.Pipelined(k_num, num_stages=2):
                        T.copy(x[bid * token_block, i_k * h_blk], a_l1)
                        T.copy(fn_t[i_k * h_blk, 0], b_l1)
                        if i_k == 0:
                            T.gemm_v0(a_l1, b_l1, c_l0, init=True)
                        else:
                            T.gemm_v0(a_l1, b_l1, c_l0)
                else:
                    for i_k in T.serial(k_num):
                        T.copy(x[bid * token_block, i_k * h_blk], a_l1)
                        T.copy(fn_t[i_k * h_blk, 0], b_l1)
                        if i_k == 0:
                            T.gemm_v0(a_l1, b_l1, c_l0, init=True)
                        else:
                            T.gemm_v0(a_l1, b_l1, c_l0)

                T.copy(c_l0, out[bid * token_block, 0])

    return main


# ============================================================
# Kernel A2: SqrSum (Vector) + T.Pipelined
# ============================================================


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def mhc_pre_sqrsum(pad_hc_hidden, h_blk=SQRSUM_H_BLK, dtype="bfloat16", accum_dtype="float"):
    """Kernel A2: sqrsum = sum(x^2) over hidden dim.

    Dual-V-core, large h_blk, T.Pipelined.
    """
    n = T.symbolic("n")
    k_num = T.ceildiv(pad_hc_hidden, h_blk)
    VEC_NUM = 2

    @T.prim_func
    def main(
        x: T.Tensor((n, pad_hc_hidden), dtype),
        sqrsum: T.Tensor((n,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(n, VEC_NUM), is_npu=True) as (cid, vid):
            bid = cid * VEC_NUM + vid

            if bid < n:
                with T.Scope("V"):
                    acc_ub = T.alloc_ub((h_blk,), accum_dtype)
                    x_ub = T.alloc_ub((h_blk,), dtype)
                    x_fp32 = T.alloc_ub((h_blk,), accum_dtype)
                    x_sq = T.alloc_ub((h_blk,), accum_dtype)
                    T.tile.fill(acc_ub, 0.0)

                    for i_k in T.Pipelined(k_num, num_stages=2):
                        T.copy(x[bid, i_k * h_blk], x_ub)
                        T.tile.cast(x_fp32, x_ub, "CAST_NONE", h_blk)
                        T.tile.mul(x_sq, x_fp32, x_fp32)
                        T.tile.add(acc_ub, acc_ub, x_sq)

                    result_ub = T.alloc_ub(1, accum_dtype)
                    T.reduce_sum(acc_ub, result_ub, dim=-1)
                    T.copy(result_ub, sqrsum[bid])

    return main


# ============================================================
# Kernel B1: RMSNorm (Vector)
# ============================================================


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def mhc_pre_rmsnorm(pad_hc_mult3, hc_mult, hidden_size, rms_eps, dtype="float"):
    """Kernel B1: mixes = out * rsqrt(sqrsum / (hc * hidden) + rms_eps).

    One token per block. Reads gemm_out and sqrsum, writes normalized mixes.
    """
    n = T.symbolic("n")

    @T.prim_func
    def main(
        gemm_out: T.Tensor((n, pad_hc_mult3), dtype),
        sqrsum: T.Tensor((n,), dtype),
        mixes: T.Tensor((n, pad_hc_mult3), dtype),
    ):
        with T.Kernel(T.ceildiv(n, 2), is_npu=True) as (cid, vid):
            bid = cid * 2 + vid
            if bid < n:
                rms_ub = T.alloc_ub(1, dtype)
                inv_sqrt_ub = T.alloc_ub(1, dtype)

                T.copy(sqrsum[bid], rms_ub)
                T.tile.mul(rms_ub, rms_ub, 1.0 / (hc_mult * hidden_size))
                T.tile.add(rms_ub, rms_ub, rms_eps)
                T.tile.rsqrt(inv_sqrt_ub, rms_ub)

                out_ub = T.alloc_ub(pad_hc_mult3, dtype)
                mixes_ub = T.alloc_ub(pad_hc_mult3, dtype)
                T.copy(gemm_out[bid, 0], out_ub)
                T.tile.mul(mixes_ub, out_ub, inv_sqrt_ub[0])
                T.copy(mixes_ub, mixes[bid, 0])

    return main


# ============================================================
# Kernel B2: Split + Sinkhorn (Vector)
# ============================================================


@tilelang.jit(out_idx=[4, 5, 6], workspace_idx=[3], pass_configs=pass_configs)
def mhc_pre_split_sinkhorn(hc, sinkhorn_iters, eps, hc_post_mult_value=2.0, dtype="float"):
    """Kernel B2: mixes -> pre/post/comb + Sinkhorn normalization.

    Adapted from examples/deepseek_v4/hc_split_sinkhorn.py.
    One token per block.

    Args:
        mixes:  [n, mix_hc]  fp32  (mix_hc = hc*(2+hc))
        hc_scale: [3]  fp32
        hc_base:  [mix_hc]  fp32
    Returns:
        pre:  [n, hc]  fp32
        post: [n, hc]  fp32
        comb: [n, hc, hc]  fp32
    """
    n = T.symbolic("n")
    mix_hc = hc * (2 + hc)

    hc_pad = hc
    if hc * 4 % 32 != 0:
        hc_pad = tilelang.cdiv(hc * 4, 32) * 32 // 4

    @T.prim_func
    def main(
        mixes: T.Tensor([n, mix_hc], dtype),
        hc_scale: T.Tensor([3], dtype),
        hc_base: T.Tensor([mix_hc], dtype),
        workspace: T.Tensor([n, mix_hc], dtype),
        pre: T.Tensor([n, hc], dtype),
        post: T.Tensor([n, hc], dtype),
        comb: T.Tensor([n, hc, hc], dtype),
    ):
        with T.Kernel(T.ceildiv(n, 2), is_npu=True) as (cid, vid):
            bid = cid * 2 + vid
            if bid < n:
                mixes_shared = T.alloc_shared(mix_hc, dtype)
                hc_base_shared = T.alloc_shared(mix_hc, dtype)
                hc_scale_shared = T.alloc_ub(mix_hc, dtype)

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

                for i in T.serial(hc):
                    hc_scale_shared[i] = alpha_0
                for i in T.serial(hc):
                    hc_scale_shared[hc + i] = alpha_1
                for i in T.serial(hc * hc):
                    hc_scale_shared[2 * hc + i] = alpha_2
                T.copy(hc_base, hc_base_shared)
                T.copy(mixes[bid, :], mixes_shared)

                T.tile.mul(mixes_shared, mixes_shared, hc_scale_shared)
                T.tile.add(mixes_shared, mixes_shared, hc_base_shared)
                T.copy(mixes_shared, workspace[bid, :])

                # pre
                T.copy(workspace[bid, :hc], tmp_shared)
                T.tile.sigmoid(pre_shared, tmp_shared)
                T.tile.add(pre_shared, pre_shared, eps)
                T.copy(pre_shared[:hc], pre[bid, :hc])

                # post
                T.copy(workspace[bid, hc : hc + hc_pad], tmp_shared)
                T.tile.sigmoid(post_shared, tmp_shared)
                T.tile.mul(post_shared, post_shared, hc_post_mult_value)
                T.copy(post_shared[:hc], post[bid, :hc])

                # comb
                for i in T.serial(hc):
                    start = 2 * hc + i * hc
                    end = 2 * hc + i * hc + hc
                    T.copy(workspace[bid, start:end], tmp_shared)
                    T.copy(tmp_shared, comb_shared[i, :])

                # comb = comb.softmax(-1) + eps
                T.reduce_max(comb_shared, row_max, dim=-1, real_shape=[hc, hc])
                for i in T.serial(hc):
                    T.tile.fill(row_div[i, :], row_max[i])
                T.tile.sub(comb_shared, comb_shared, row_div)
                T.tile.exp(comb_shared, comb_shared)
                T.reduce_sum(comb_shared, row_sum, dim=-1, real_shape=[hc, hc])
                for i in T.serial(hc):
                    T.tile.fill(row_div[i, :], row_sum[i])
                T.tile.div(comb_shared, comb_shared, row_div)
                T.tile.add(comb_shared, comb_shared, eps)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_shared, col_sum, dim=0, real_shape=[hc, hc_pad])
                T.tile.add(col_sum, col_sum, eps)
                T.tile.broadcast(col_broadcast, col_sum)
                T.tile.div(comb_shared, comb_shared, col_broadcast)

                for _ in T.serial(sinkhorn_iters - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(comb_shared, row_sum, dim=-1, real_shape=[hc, hc])
                    T.tile.add(row_sum, row_sum, eps)
                    for i in T.serial(hc):
                        T.tile.fill(row_div[i, :], row_sum[i])
                    T.tile.div(comb_shared, comb_shared, row_div)
                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(comb_shared, col_sum, dim=0, real_shape=[hc, hc_pad])
                    T.tile.add(col_sum, col_sum, eps)
                    T.tile.broadcast(col_broadcast, col_sum)
                    T.tile.div(comb_shared, comb_shared, col_broadcast)

                for i in T.serial(hc):
                    T.copy(comb_shared[i, :hc], comb[bid, i, :])

    return main


# ============================================================
# Kernel B3: Apply pre_mix (Vector)
# ============================================================


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def mhc_pre_apply_mix(pad_hidden, h_blk=2048, hc=4, dtype="bfloat16", accum_dtype="float"):
    """Kernel B3: layer_input = sum over hc of (residual * pre_mix).

    AXPY linear combination for hc=4:
      out = pre0*res0 + pre1*res1 + pre2*res2 + pre3*res3

    Dual-V-core, T.Pipelined.
    """
    n = T.symbolic("n")
    h_num = T.ceildiv(pad_hidden, h_blk)
    VEC_NUM = 2
    HC = hc

    @T.prim_func
    def main(
        residual: T.Tensor((n, HC, pad_hidden), dtype),
        pre_mix: T.Tensor((n, HC), accum_dtype),
        layer_input: T.Tensor((n, pad_hidden), dtype),
    ):
        with T.Kernel(T.ceildiv(n, VEC_NUM), is_npu=True) as (cid, vid):
            bid = cid * VEC_NUM + vid

            if bid < n:
                with T.Scope("V"):
                    pre_ub = T.alloc_ub(HC, accum_dtype)
                    T.copy(pre_mix[bid, 0], pre_ub)

                    res0_ub = T.alloc_ub(h_blk, dtype)
                    res1_ub = T.alloc_ub(h_blk, dtype)
                    res2_ub = T.alloc_ub(h_blk, dtype)
                    res3_ub = T.alloc_ub(h_blk, dtype)
                    res0_fp32 = T.alloc_ub(h_blk, accum_dtype)
                    res1_fp32 = T.alloc_ub(h_blk, accum_dtype)
                    res2_fp32 = T.alloc_ub(h_blk, accum_dtype)
                    res3_fp32 = T.alloc_ub(h_blk, accum_dtype)
                    out_ub = T.alloc_ub(h_blk, accum_dtype)
                    out_bf16 = T.alloc_ub(h_blk, dtype)

                    for i_h in T.Pipelined(h_num, num_stages=2):
                        T.copy(residual[bid, 0, i_h * h_blk], res0_ub)
                        T.copy(residual[bid, 1, i_h * h_blk], res1_ub)
                        T.copy(residual[bid, 2, i_h * h_blk], res2_ub)
                        T.copy(residual[bid, 3, i_h * h_blk], res3_ub)
                        T.tile.cast(res0_fp32, res0_ub, "CAST_NONE", h_blk)
                        T.tile.cast(res1_fp32, res1_ub, "CAST_NONE", h_blk)
                        T.tile.cast(res2_fp32, res2_ub, "CAST_NONE", h_blk)
                        T.tile.cast(res3_fp32, res3_ub, "CAST_NONE", h_blk)

                        T.tile.mul(out_ub, res0_fp32, pre_ub[0])
                        T.tile.axpy(out_ub, res1_fp32, pre_ub[1])
                        T.tile.axpy(out_ub, res2_fp32, pre_ub[2])
                        T.tile.axpy(out_ub, res3_fp32, pre_ub[3])

                        T.tile.cast(out_bf16, out_ub, "CAST_RINT", h_blk)
                        T.copy(out_bf16, layer_input[bid, i_h * h_blk])

    return main


# ============================================================
# Host-side adapter
# ============================================================


def _pad_2d(t, target_rows, target_cols):
    """[rows, cols] -> [target_rows, target_cols], padded with zeros."""
    result = torch.zeros(target_rows, target_cols, dtype=t.dtype, device=t.device)
    result[: t.shape[0], : t.shape[1]] = t
    return result


def _pad_1d(t, target_size):
    """[rows] -> [target], padded with zeros."""
    result = torch.zeros(target_size, dtype=t.dtype, device=t.device)
    result[: t.shape[0]] = t
    return result


def _pad_3d(t, target_dim1, target_dim2):
    """[n, d1, d2] -> [n, target1, target2], padded with zeros."""
    n = t.shape[0]
    result = torch.zeros(n, target_dim1, target_dim2, dtype=t.dtype, device=t.device)
    result[:, : t.shape[1], : t.shape[2]] = t
    return result


_kernel_cache = {}


def _get_kernel(name, *args):
    """Cache compiled kernel to avoid repeated JIT lookup."""
    key = (name, *args)
    if key not in _kernel_cache:
        _kernel_cache[key] = _KERNEL_BUILDERS[name](*args)
    return _kernel_cache[key]


_KERNEL_BUILDERS = {
    "gemm": mhc_pre_gemm,
    "sqrsum": mhc_pre_sqrsum,
    "rmsnorm": mhc_pre_rmsnorm,
    "sinkhorn": mhc_pre_split_sinkhorn,
    "apply": mhc_pre_apply_mix,
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
    """Kernel A host adapter.

    Args:
        x:  [n, hc*hidden] bf16
        fn: [hc_mult3, hc*hidden] fp32
        hc_mult: int

    Returns:
        out:    [n, hc_mult3] fp32
        sqrsum: [n] fp32
    """
    n = x.shape[0]
    hc_hidden = x.shape[1]
    hc_mult3 = hc_mult * (2 + hc_mult)

    pad_hc_mult3 = calc_pad(hc_mult3, MIN_BLOCK)
    pad_hc_hidden = calc_pad(hc_hidden, H_BLK)
    pad_hc_hidden_sqr = calc_pad(hc_hidden, SQRSUM_H_BLK)
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

    if pad_hc_hidden_sqr == hc_hidden:
        sqrsum_kernel = _get_kernel("sqrsum", pad_hc_hidden_sqr)
        sqrsum = sqrsum_kernel(x)
    else:
        x_padded_sqr = _pad_2d(x, n, pad_hc_hidden_sqr)
        sqrsum_kernel = _get_kernel("sqrsum", pad_hc_hidden_sqr)
        sqrsum = sqrsum_kernel(x_padded_sqr[:n])

    return out_padded, sqrsum


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
    """
    hc_mult = residual.shape[1]
    hidden = residual.shape[2]
    n = residual.shape[0]
    hc_mult3 = hc_mult * (2 + hc_mult)

    x_flat = residual.view(n, hc_mult * hidden)

    # Kernel A: gemm + sqrsum (returns padded [n, pad_hc_mult3])
    gemm_out_padded, sqrsum = mhc_pre_gemm_sqrsum(x_flat, fn, hc_mult, fn_packed)

    # Kernel B1: RMSNorm (directly uses padded gemm output)
    pad_hc_mult3 = calc_pad(hc_mult3, MIN_BLOCK)
    rmsnorm_kernel = _get_kernel("rmsnorm", pad_hc_mult3, hc_mult, hidden, rms_eps)
    mixes_padded = rmsnorm_kernel(gemm_out_padded, sqrsum)
    mixes = mixes_padded[:n, :hc_mult3].contiguous()

    # Kernel B2: split + sinkhorn
    sinkhorn_kernel = _get_kernel("sinkhorn", hc_mult, sinkhorn_repeat, hc_pre_eps, hc_post_mult_value)
    pre_mix, post_mix, comb_mix = sinkhorn_kernel(mixes, hc_scale, hc_base)

    # Kernel B3: apply pre_mix
    pad_hidden = calc_pad(hidden, 2048)
    if pad_hidden == hidden:
        residual_padded = residual
    else:
        residual_padded = _pad_3d(residual, hc_mult, pad_hidden)
    apply_kernel = _get_kernel("apply", pad_hidden)
    layer_input_padded = apply_kernel(residual_padded, pre_mix)

    if pad_hidden == hidden:
        layer_input = layer_input_padded
    else:
        layer_input = layer_input_padded[:n, :hidden]

    post_mix_out = post_mix.unsqueeze(-1)
    return post_mix_out, comb_mix, layer_input


# ============================================================
# Golden reference
# ============================================================


def mhc_pre_gemm_sqrsum_ref(x, fn, hc_mult):
    """PyTorch golden, following the same bf16 computation path as the kernel."""
    fn_bf16 = fn.bfloat16()
    out = x.float() @ fn_bf16.float().T
    sqrsum = x.float().square().sum(-1)
    return out, sqrsum


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
    out, sqrsum = mhc_pre_gemm_sqrsum_ref(residual_flat, fn, hc_mult)

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


def generate_test_data(n, h, hc_mult, device="npu"):
    torch.random.manual_seed(42)
    hc_hidden = hc_mult * h
    hc_mult3 = hc_mult * (2 + hc_mult)
    x = torch.randn((n, hc_hidden), dtype=torch.bfloat16, device=device)
    fn = torch.randn((hc_mult3, hc_hidden), dtype=torch.float32, device=device)
    return {"x": x, "fn": fn, "hc_mult": hc_mult}


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


def test_kernel_a():
    print("=" * 60)
    print("MHC Pre Kernel A (gemm_sqrsum) test (Ascend NPU)")
    print("=" * 60)

    test_cases = [
        (4, 128, 4),
        (16, 256, 4),
        (4, 1280, 4),
        (512, 2560, 4),
        (4, 100, 4),
    ]

    all_passed = True
    for n, h, hc_mult in test_cases:
        print(f"\n--- n={n}, h={h}, hc_mult={hc_mult} ---")
        data = generate_test_data(n, h, hc_mult)

        out_tl, sqrsum_tl = mhc_pre_gemm_sqrsum(**data)
        out_ref, sqrsum_ref = mhc_pre_gemm_sqrsum_ref(**data)

        print(f"  out shape={out_tl.shape}, sqrsum shape={sqrsum_tl.shape}")

        try:
            torch.testing.assert_close(out_tl.cpu(), out_ref.cpu(), rtol=1e-2, atol=1e-2)
            out_diff = (out_tl.cpu().float() - out_ref.cpu().float()).abs()
            print(f"  out PASSED (max_diff={out_diff.max().item():.6f})")
        except AssertionError:
            out_diff = (out_tl.cpu().float() - out_ref.cpu().float()).abs()
            print(f"  out FAILED (max_diff={out_diff.max().item():.6f})")
            all_passed = False

        try:
            torch.testing.assert_close(sqrsum_tl.cpu(), sqrsum_ref.cpu(), rtol=1e-2, atol=1e-2)
            sqrsum_diff = (sqrsum_tl.cpu().float() - sqrsum_ref.cpu().float()).abs()
            print(f"  sqrsum PASSED (max_diff={sqrsum_diff.max().item():.6f})")
        except AssertionError:
            sqrsum_diff = (sqrsum_tl.cpu().float() - sqrsum_ref.cpu().float()).abs()
            print(f"  sqrsum FAILED (max_diff={sqrsum_diff.max().item():.6f})")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("Kernel Output Match!")
    else:
        print("Some tests failed.")
    print("=" * 60)


def test_full():
    print("=" * 60)
    print("MHC Pre full pipeline test (Ascend NPU)")
    print("=" * 60)

    test_cases = [
        (4, 128, 4),
        (16, 256, 4),
        (4, 1280, 4),
        (512, 2560, 4),
        (4, 100, 4),
    ]

    all_passed = True
    for n, h, hc_mult in test_cases:
        print(f"\n--- n={n}, h={h}, hc_mult={hc_mult} ---")
        data = generate_full_test_data(n, h, hc_mult)

        post_tl, comb_tl, layer_tl = mhc_pre(**data)
        post_ref, comb_ref, layer_ref = mhc_pre_ref(**data)

        print(f"  post_mix={post_tl.shape}, comb_mix={comb_tl.shape}, layer_input={layer_tl.shape}")

        try:
            torch.testing.assert_close(post_tl.cpu(), post_ref.cpu(), rtol=1e-2, atol=1e-2)
            diff = (post_tl.cpu().float() - post_ref.cpu().float()).abs()
            print(f"  post_mix PASSED (max_diff={diff.max().item():.6f})")
        except AssertionError:
            diff = (post_tl.cpu().float() - post_ref.cpu().float()).abs()
            print(f"  post_mix FAILED (max_diff={diff.max().item():.6f})")
            all_passed = False

        try:
            torch.testing.assert_close(comb_tl.cpu(), comb_ref.cpu(), rtol=1e-2, atol=1e-2)
            diff = (comb_tl.cpu().float() - comb_ref.cpu().float()).abs()
            print(f"  comb_mix PASSED (max_diff={diff.max().item():.6f})")
        except AssertionError:
            diff = (comb_tl.cpu().float() - comb_ref.cpu().float()).abs()
            print(f"  comb_mix FAILED (max_diff={diff.max().item():.6f})")
            all_passed = False

        try:
            torch.testing.assert_close(layer_tl.cpu(), layer_ref.cpu(), rtol=1e-2, atol=1e-2)
            diff = (layer_tl.cpu().float() - layer_ref.cpu().float()).abs()
            print(f"  layer_input PASSED (max_diff={diff.max().item():.6f})")
        except AssertionError:
            diff = (layer_tl.cpu().float() - layer_ref.cpu().float()).abs()
            print(f"  layer_input FAILED (max_diff={diff.max().item():.6f})")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("Kernel Output Match!")
    else:
        print("Some tests failed.")
    print("=" * 60)


if __name__ == "__main__":
    tilelang.disable_cache()
    test_full()
