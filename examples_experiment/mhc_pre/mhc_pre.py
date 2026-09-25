# ruff: noqa
import math

import torch
import tilelang
from tilelang import language as T


tilelang.disable_cache()


# ===========================================================================
# fn_bf16 / hc_scale_base / residual-view caches (iter12/13 Direction D: fn &
# scale are model weights that don't change between calls; cache keyed by
# data_ptr). R3-iter1: merged hc_scale_exp+hc_base into one [2, 24] tensor to
# cut one kernel launch arg (launch tax ~30-60us/arg, measured R3-iter1);
# residual views are pure aliases (saved ~2 view constructions per call).
# ===========================================================================
_fn_bf16_cache: dict[tuple[int, int], torch.Tensor] = {}
_scale_base_cache: dict[tuple[int, int, int], torch.Tensor] = {}
_residual_view_cache: dict[tuple[int, tuple, int], tuple] = {}
_hidden_block_cache: dict[int, int] = {}


# ===========================================================================
# pass_configs (rev3: single-kernel fused, all 4 Developer passes on)
# ===========================================================================

PASS_CONFIGS_CUBE = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

PASS_CONFIGS_DEV = PASS_CONFIGS_CUBE  # same 4 flags for the fused kernel


# ===========================================================================
# Legacy K1 (mhc_pre_gemm_sqrsum) + K2 (mhc_pre_big_fuse) — kept verbatim from
# v2 for example_mhc_pre.py / perf tooling imports (NOT used by mhc_pre host
# entry anymore, which now routes to the rev3 single fused kernel).
# ===========================================================================


@tilelang.jit(out_idx=[2, 3], pass_configs=PASS_CONFIGS_CUBE)
def mhc_pre_gemm_sqrsum(
    num_tokens,
    hc_hidden_size,
    block_M=64,
    block_N_pad=32,
    block_K=256,
    block_K_sqr=512,
):
    """CV fused (legacy v2 K1): Cube=bf16 GEMM (x@fn^T->fp32 L0C), Vector=bf16 sqrsum."""
    dtype = "bfloat16"
    accum_dtype = "float"

    m_num = (num_tokens + block_M - 1) // block_M
    k_num_gemm = T.ceildiv(hc_hidden_size, block_K)
    k_num_sqr = T.ceildiv(hc_hidden_size, block_K_sqr)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        x_bf16: T.Tensor([num_tokens, hc_hidden_size], dtype),  # type: ignore
        fn_bf16: T.Tensor([block_N_pad, hc_hidden_size], dtype),  # type: ignore
        gemm_out: T.Tensor([num_tokens, block_N_pad], accum_dtype),  # type: ignore
        sqrsum: T.Tensor([num_tokens], accum_dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            x_l1 = T.alloc_shared((block_M, block_K), dtype)
            fn_l1 = T.alloc_shared((block_N_pad, block_K), dtype)
            out_l0c = T.alloc_fragment((block_M, block_N_pad), accum_dtype)

            for k in T.serial(k_num_gemm):
                T.copy(x_bf16[cid * block_M, k * block_K], x_l1)
                T.copy(fn_bf16[0, k * block_K], fn_l1)
                if k == 0:
                    T.gemm_v0(x_l1, fn_l1, out_l0c, transpose_B=True, init=True)
                else:
                    T.gemm_v0(x_l1, fn_l1, out_l0c, transpose_B=True)
            T.copy(out_l0c, gemm_out[cid * block_M, 0])

            x_ub = T.alloc_ub((sub_block_M, block_K_sqr), dtype)
            x_fp32_ub = T.alloc_ub((sub_block_M, block_K_sqr), accum_dtype)
            sq_ub = T.alloc_ub((sub_block_M, block_K_sqr), accum_dtype)
            partial_ub = T.alloc_ub((sub_block_M), accum_dtype)
            acc_ub = T.alloc_ub((sub_block_M), accum_dtype)

            T.tile.fill(acc_ub, 0.0)
            row_base = cid * block_M + vid * sub_block_M
            for k in T.serial(k_num_sqr):
                T.copy(x_bf16[row_base, k * block_K_sqr], x_ub)
                T.tile.cast(x_fp32_ub, x_ub, "CAST_NONE", sub_block_M * block_K_sqr)
                T.tile.mul(sq_ub, x_fp32_ub, x_fp32_ub)
                T.reduce_sum(sq_ub, partial_ub, dim=-1)
                T.tile.add(acc_ub, acc_ub, partial_ub)
            T.copy(acc_ub, sqrsum[row_base : row_base + sub_block_M])

    return main


@tilelang.jit(out_idx=[5, 6, 7])
def mhc_pre_big_fuse(
    num_tokens,
    hidden_size,
    hc_mult,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    block_M2=8,
    hidden_block=256,
):
    """big_fuse (legacy v2 K2, Expert): rms + mixes + pre/post/comb + sinkhorn + layer_input."""
    dtype = "float"
    accum_dtype = "float"
    hc_mult3 = hc_mult * (2 + hc_mult)
    hc_mult2 = hc_mult * hc_mult
    block_N_pad = 32
    VEC_NUM = 2
    sub_block_M2 = block_M2 // VEC_NUM
    D = hc_mult * hidden_size
    hc_mult_pad = hc_mult
    if hc_mult * 4 % 32 != 0:
        hc_mult_pad = ((hc_mult * 4 + 31) // 32) * 32 // 4

    m_num = (num_tokens + block_M2 - 1) // block_M2
    h_num = (hidden_size + hidden_block - 1) // hidden_block

    @T.prim_func
    def main(
        gemm_out: T.Tensor([num_tokens, block_N_pad], accum_dtype),  # type: ignore
        sqrsum: T.Tensor([num_tokens], accum_dtype),  # type: ignore
        residual_bf16: T.Tensor([num_tokens, hc_mult, hidden_size], "bfloat16"),  # type: ignore
        hc_scale: T.Tensor([3], accum_dtype),  # type: ignore
        hc_base: T.Tensor([hc_mult3], accum_dtype),  # type: ignore
        post_mix: T.Tensor([num_tokens, hc_mult], accum_dtype),  # type: ignore
        comb_mix: T.Tensor([num_tokens, hc_mult2], accum_dtype),  # type: ignore
        layer_input: T.Tensor([num_tokens, hidden_size], "bfloat16"),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M2 + vid * sub_block_M2

            sqrsum_ub = T.alloc_ub(sub_block_M2, accum_dtype)
            rms_ub = T.alloc_ub(sub_block_M2, accum_dtype)
            rms_tmp_ub = T.alloc_ub(sub_block_M2, accum_dtype)
            rms_2d_ub = T.alloc_ub((sub_block_M2, 1), accum_dtype)
            rms_bcast_ub = T.alloc_ub((sub_block_M2, hc_mult3), accum_dtype)

            mixes_ub = T.alloc_ub((sub_block_M2, hc_mult3), accum_dtype)
            scale_exp_ub = T.alloc_ub((1, hc_mult3), accum_dtype)
            scale_bcast_ub = T.alloc_ub((sub_block_M2, hc_mult3), accum_dtype)
            base_ub = T.alloc_ub((1, hc_mult3), accum_dtype)
            base_bcast_ub = T.alloc_ub((sub_block_M2, hc_mult3), accum_dtype)

            seg_1d_ub = T.alloc_ub(hc_mult_pad, accum_dtype)
            pre_1d_ub = T.alloc_ub(hc_mult_pad, accum_dtype)
            post_1d_ub = T.alloc_ub(hc_mult_pad, accum_dtype)
            pre_ub = T.alloc_ub((sub_block_M2, hc_mult), accum_dtype)

            comb_ub = T.alloc_ub((hc_mult, hc_mult_pad), accum_dtype)
            row_max_ub = T.alloc_ub(hc_mult, accum_dtype)
            row_sum_ub = T.alloc_ub(hc_mult, accum_dtype)
            col_sum_ub = T.alloc_ub((1, hc_mult_pad), accum_dtype)
            row_div_ub = T.alloc_ub((hc_mult, hc_mult_pad), accum_dtype)
            col_bcast_ub = T.alloc_ub((hc_mult, hc_mult_pad), accum_dtype)
            comb_flat_ub = T.alloc_ub(hc_mult2, accum_dtype)

            res_2d_bf16_ub = T.alloc_ub((hc_mult, hidden_block), "bfloat16")
            res_slice_ub = T.alloc_ub(hidden_block, accum_dtype)
            ol_ub = T.alloc_ub(hidden_block, accum_dtype)
            ol_bf16_ub = T.alloc_ub(hidden_block, "bfloat16")
            tmp_weighted_ub = T.alloc_ub(hidden_block, accum_dtype)
            pre_bcast_ub = T.alloc_ub(hidden_block, accum_dtype)

            with T.Scope("V"):
                if row_base < num_tokens:
                    T.copy(sqrsum[row_base : row_base + sub_block_M2], sqrsum_ub)
                    T.barrier_all()
                    T.tile.div(rms_tmp_ub, sqrsum_ub, float(D))
                    T.tile.add(rms_tmp_ub, rms_tmp_ub, rms_eps)
                    T.tile.rsqrt(rms_ub, rms_tmp_ub)
                    T.tile.broadcast(rms_2d_ub, rms_ub)
                    T.tile.broadcast(rms_bcast_ub, rms_2d_ub)

                    T.copy(gemm_out[row_base, 0], mixes_ub)
                    T.barrier_all()
                    T.tile.mul(mixes_ub, mixes_ub, rms_bcast_ub)

                    alpha_0 = hc_scale[0]
                    alpha_1 = hc_scale[1]
                    alpha_2 = hc_scale[2]
                    for i in T.serial(hc_mult):
                        scale_exp_ub[0, i] = alpha_0
                    for i in T.serial(hc_mult):
                        scale_exp_ub[0, hc_mult + i] = alpha_1
                    for i in T.serial(hc_mult * hc_mult):
                        scale_exp_ub[0, 2 * hc_mult + i] = alpha_2
                    T.tile.broadcast(scale_bcast_ub, scale_exp_ub)

                    T.copy(hc_base, base_ub[0, 0])
                    T.barrier_all()
                    T.tile.broadcast(base_bcast_ub, base_ub)
                    T.tile.mul(mixes_ub, mixes_ub, scale_bcast_ub)
                    T.tile.add(mixes_ub, mixes_ub, base_bcast_ub)

                    for tok in T.serial(sub_block_M2):
                        if row_base + tok < num_tokens:
                            for j in T.serial(hc_mult):
                                seg_1d_ub[j] = mixes_ub[tok, j]
                            T.tile.sigmoid(pre_1d_ub, seg_1d_ub)
                            for j in T.serial(hc_mult):
                                pre_ub[tok, j] = pre_1d_ub[j] + hc_pre_eps

                            for j in T.serial(hc_mult):
                                seg_1d_ub[j] = mixes_ub[tok, hc_mult + j]
                            T.tile.sigmoid(post_1d_ub, seg_1d_ub)
                            for j in T.serial(hc_mult):
                                post_mix[row_base + tok, j] = post_1d_ub[j] * hc_post_mult_value
                            T.barrier_all()

                    for tok in T.serial(sub_block_M2):
                        if row_base + tok < num_tokens:
                            for i in T.serial(hc_mult):
                                for j in T.serial(hc_mult):
                                    comb_ub[i, j] = mixes_ub[tok, 2 * hc_mult + i * hc_mult + j]

                            T.reduce_max(
                                comb_ub,
                                row_max_ub,
                                dim=-1,
                                real_shape=[hc_mult, hc_mult],
                            )
                            for i in T.serial(hc_mult):
                                T.tile.fill(row_div_ub[i, :], row_max_ub[i])
                            T.tile.sub(comb_ub, comb_ub, row_div_ub)
                            T.tile.exp(comb_ub, comb_ub)
                            T.reduce_sum(
                                comb_ub,
                                row_sum_ub,
                                dim=-1,
                                real_shape=[hc_mult, hc_mult],
                            )
                            for i in T.serial(hc_mult):
                                T.tile.fill(row_div_ub[i, :], row_sum_ub[i])
                            T.tile.div(comb_ub, comb_ub, row_div_ub)
                            T.tile.add(comb_ub, comb_ub, hc_sinkhorn_eps)

                            for j in T.serial(hc_mult):
                                col_sum_ub[0, j] = 0.0
                                for i in T.serial(hc_mult):
                                    col_sum_ub[0, j] = col_sum_ub[0, j] + comb_ub[i, j]
                            T.tile.add(col_sum_ub, col_sum_ub, hc_sinkhorn_eps)
                            T.tile.broadcast(col_bcast_ub, col_sum_ub)
                            T.tile.div(comb_ub, comb_ub, col_bcast_ub)

                            for _ in T.serial(sinkhorn_repeat - 1):
                                T.reduce_sum(
                                    comb_ub,
                                    row_sum_ub,
                                    dim=-1,
                                    real_shape=[hc_mult, hc_mult],
                                )
                                T.tile.add(row_sum_ub, row_sum_ub, hc_sinkhorn_eps)
                                for i in T.serial(hc_mult):
                                    T.tile.fill(row_div_ub[i, :], row_sum_ub[i])
                                T.tile.div(comb_ub, comb_ub, row_div_ub)

                                for j in T.serial(hc_mult):
                                    col_sum_ub[0, j] = 0.0
                                    for i in T.serial(hc_mult):
                                        col_sum_ub[0, j] = col_sum_ub[0, j] + comb_ub[i, j]
                                T.tile.add(col_sum_ub, col_sum_ub, hc_sinkhorn_eps)
                                T.tile.broadcast(col_bcast_ub, col_sum_ub)
                                T.tile.div(comb_ub, comb_ub, col_bcast_ub)

                            for i in T.serial(hc_mult):
                                for j in T.serial(hc_mult):
                                    comb_mix[row_base + tok, i * hc_mult + j] = comb_ub[i, j]
                            T.barrier_all()

                    for tok in T.serial(sub_block_M2):
                        if row_base + tok < num_tokens:
                            for hb in T.serial(h_num):
                                T.copy(
                                    residual_bf16[
                                        row_base + tok,
                                        0:hc_mult,
                                        hb * hidden_block : (hb + 1) * hidden_block,
                                    ],
                                    res_2d_bf16_ub,
                                )
                                T.barrier_all()
                                T.tile.fill(ol_ub, 0.0)
                                for i_hc in T.serial(hc_mult):
                                    T.copy(res_2d_bf16_ub[i_hc, :], res_slice_ub)
                                    T.tile.fill(pre_bcast_ub, pre_ub[tok, i_hc])
                                    T.tile.mul(tmp_weighted_ub, res_slice_ub, pre_bcast_ub)
                                    T.tile.add(ol_ub, ol_ub, tmp_weighted_ub)
                                T.copy(ol_ub, ol_bf16_ub)
                                T.barrier_all()
                                T.copy(ol_bf16_ub, layer_input[row_base + tok, hb * hidden_block])
                                T.barrier_all()

    return main


# ===========================================================================
# rev3 Primary: single-kernel fused mhc_pre (full Developer mode)
# ===========================================================================


@tilelang.jit(out_idx=[5, 6, 7], pass_configs=PASS_CONFIGS_DEV)
def mhc_pre_fused(
    num_tokens,
    hidden_size,
    hc_mult,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    block_M=64,
    block_N_pad=32,
    block_K=256,
    hidden_block=512,
    use_pipelined_gemm=True,
):
    """rev3 single-kernel fused (DESIGN.md rev3 Primary):
    Cube GEMM (L0C->GM) + Vector sqrsum (parallel) + Vector tail
    (rms/mixes/pre/post/comb sinkhorn/layer_input), all in one kernel.

    Developer mode: AUTO_CV_COMBINE splits Cube loop vs Vector loops; the
    C->V cross-core point is L0C->GM write (Cube) then GM->UB read (Vector),
    mediated by GM with AUTO_CV_SYNC. No manual T.Scope / T.barrier_all.

    NOTE: The direct L0C->UB on-chip copy (T.copy(out_l0c, gemm_out_ub)) is
    broken in newer tilelang (commit >= 386ae6ef) for certain shapes
    (e.g. hidden_size=1280, D=5120). The GM round-trip (L0C->GM->UB) is a
    workaround that trades one GM write+read for correctness. See debug_log.md.

    use_pipelined_gemm=True  -> K-loop T.Pipelined(num_stages=2) (V1, rev3)
    use_pipelined_gemm=False -> K-loop T.serial (v2-proven fallback, DESIGN §10)

    R3-iter1: hc_scale_exp [24] and hc_base [24] merged into hc_scale_base
    [2, 24] (row0=scale_exp, row1=base) -> 7 kernel args instead of 8 (launch
    path per-arg cost ~30-60us, measured in A/B tests).
    """
    dtype = "bfloat16"
    accum_dtype = "float"
    hc_mult3 = hc_mult * (2 + hc_mult)  # 24
    hc_mult2 = hc_mult * hc_mult  # 16
    D = hc_mult * hidden_size
    m_num = (num_tokens + block_M - 1) // block_M
    k_num = T.ceildiv(D, block_K)
    h_num = (hidden_size + hidden_block - 1) // hidden_block
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        x_bf16: T.Tensor([num_tokens, D], dtype),  # type: ignore
        residual_bf16: T.Tensor([num_tokens, hc_mult, hidden_size], dtype),  # type: ignore
        fn_bf16: T.Tensor([block_N_pad, D], dtype),  # type: ignore
        hc_scale_base: T.Tensor([2, hc_mult3], accum_dtype),  # type: ignore
        gemm_out_gm: T.Tensor([num_tokens, block_N_pad], accum_dtype),  # type: ignore
        post_mix: T.Tensor([num_tokens, hc_mult], accum_dtype),  # type: ignore
        comb_mix: T.Tensor([num_tokens, hc_mult2], accum_dtype),  # type: ignore
        layer_input: T.Tensor([num_tokens, hidden_size], dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M

            # ============ Cube (AIC): bf16 GEMM over block_M rows ============
            x_l1 = T.alloc_shared((block_M, block_K), dtype)
            fn_l1 = T.alloc_shared((block_N_pad, block_K), dtype)
            out_l0c = T.alloc_fragment((block_M, block_N_pad), accum_dtype)
            if use_pipelined_gemm:
                # V1: pure-Cube pipelined body (no cast inside; iter8 rule)
                for k in T.Pipelined(k_num, num_stages=2):
                    T.copy(x_bf16[cid * block_M, k * block_K], x_l1)
                    T.copy(fn_bf16[0, k * block_K], fn_l1)
                    if k == 0:
                        T.gemm_v0(x_l1, fn_l1, out_l0c, transpose_B=True, init=True)
                    else:
                        T.gemm_v0(x_l1, fn_l1, out_l0c, transpose_B=True)
            else:
                for k in T.serial(k_num):
                    T.copy(x_bf16[cid * block_M, k * block_K], x_l1)
                    T.copy(fn_bf16[0, k * block_K], fn_l1)
                    if k == 0:
                        T.gemm_v0(x_l1, fn_l1, out_l0c, transpose_B=True, init=True)
                    else:
                        T.gemm_v0(x_l1, fn_l1, out_l0c, transpose_B=True)
            # L0C -> GM (Cube GM write; AUTO_CV_SYNC handles C->V sync via GM)
            T.copy(out_l0c, gemm_out_gm[cid * block_M, 0])

            # ============ Vector (AIV): bf16 sqrsum (serial, parallel to GEMM) ============
            x_ub = T.alloc_ub((sub_block_M, block_K), dtype)
            x_fp32_ub = T.alloc_ub((sub_block_M, block_K), accum_dtype)
            sq_ub = T.alloc_ub((sub_block_M, block_K), accum_dtype)
            partial_ub = T.alloc_ub((sub_block_M), accum_dtype)
            acc_ub = T.alloc_ub((sub_block_M), accum_dtype)

            T.tile.fill(acc_ub, 0.0)
            for k in T.serial(k_num):
                T.copy(x_bf16[row_base, k * block_K], x_ub)
                # cast stays OUTSIDE the pipelined Cube body (iter8 rule: no
                # T.tile.cast in any Pipelined body)
                T.tile.cast(x_fp32_ub, x_ub, "CAST_NONE", sub_block_M * block_K)
                T.tile.mul(sq_ub, x_fp32_ub, x_fp32_ub)
                T.reduce_sum(sq_ub, partial_ub, dim=-1)
                T.tile.add(acc_ub, acc_ub, partial_ub)

            # ============ Vector tail (per-vid rows; all GM writes via T.copy) ============
            if row_base < num_tokens:
                # ---- Step 1: rms = rsqrt(sqrsum / D + rms_eps) ----
                rms_tmp_ub = T.alloc_ub(sub_block_M, accum_dtype)
                rms_ub = T.alloc_ub(sub_block_M, accum_dtype)
                rms_2d_ub = T.alloc_ub((sub_block_M, 1), accum_dtype)
                rms_bcast_ub = T.alloc_ub((sub_block_M, block_N_pad), accum_dtype)
                T.tile.div(rms_tmp_ub, acc_ub, float(D))
                T.tile.add(rms_tmp_ub, rms_tmp_ub, rms_eps)
                T.tile.rsqrt(rms_ub, rms_tmp_ub)
                T.tile.broadcast(rms_2d_ub, rms_ub)  # [sub] -> [sub,1]
                T.tile.broadcast(rms_bcast_ub, rms_2d_ub)  # [sub,1] -> [sub,32]

                # ---- Step 2: mixes = gemm_out * rms * scale + base ----
                # Read gemm rows from GM (L0C->GM->UB round-trip, avoids broken
                # direct L0C->UB cross-core copy in newer tilelang).
                mixes_ub = T.alloc_ub((sub_block_M, block_N_pad), accum_dtype)
                for i in T.serial(sub_block_M):
                    T.copy(gemm_out_gm[row_base + i, 0:block_N_pad], mixes_ub[i, :])
                T.tile.mul(mixes_ub, mixes_ub, rms_bcast_ub)

                scale_ub = T.alloc_ub((1, block_N_pad), accum_dtype)
                scale_bcast_ub = T.alloc_ub((sub_block_M, block_N_pad), accum_dtype)
                base_ub = T.alloc_ub((1, block_N_pad), accum_dtype)
                base_bcast_ub = T.alloc_ub((sub_block_M, block_N_pad), accum_dtype)
                # V8: host pre-expands hc_scale -> hc_scale_exp [24]; R3-iter1
                # merged with hc_base into hc_scale_base [2,24]: single GM->UB
                # copies + 2D broadcast (no 24-scalar per-block build)
                T.copy(hc_scale_base[0, :], scale_ub[0, 0])
                T.copy(hc_scale_base[1, :], base_ub[0, 0])
                T.tile.broadcast(scale_bcast_ub, scale_ub)  # [1,32] -> [sub,32]
                T.tile.broadcast(base_bcast_ub, base_ub)
                T.tile.mul(mixes_ub, mixes_ub, scale_bcast_ub)
                T.tile.add(mixes_ub, mixes_ub, base_bcast_ub)

                # ---- Step 3-4: pre/post per token (8-lane 32B extract + sigmoid) ----
                seg8_ub = T.alloc_ub(8, accum_dtype)
                sig8_ub = T.alloc_ub(8, accum_dtype)
                pre8_ub = T.alloc_ub(8, accum_dtype)
                post8_ub = T.alloc_ub(8, accum_dtype)
                post1d_ub = T.alloc_ub(hc_mult, accum_dtype)
                pre4_1d_ub = T.alloc_ub(hc_mult, accum_dtype)

                # ---- Step 5: comb sinkhorn per token ----
                # comb_ub is [4,8] (32B rows, v2-proven): 2D reduce/broadcast on
                # [4,4]-padded rows via real_shape; scalar extraction/write avoid
                # 16B UB->UB copies (V3 rule keeps copies >= 32B).
                comb_ub = T.alloc_ub((hc_mult, hc_mult * 2), accum_dtype)
                row_max_ub = T.alloc_ub(hc_mult, accum_dtype)
                row_sum_ub = T.alloc_ub(hc_mult, accum_dtype)
                col_sum_ub = T.alloc_ub((1, hc_mult * 2), accum_dtype)
                row_div_ub = T.alloc_ub((hc_mult, hc_mult * 2), accum_dtype)
                col_bcast_ub = T.alloc_ub((hc_mult, hc_mult * 2), accum_dtype)
                comb_flat_ub = T.alloc_ub(hc_mult2, accum_dtype)

                # ---- Step 6: layer_input batched [hc, hidden_block] ----
                res_2d_ub = T.alloc_ub((hc_mult, hidden_block), dtype)
                res_fp32_ub = T.alloc_ub((hc_mult, hidden_block), accum_dtype)
                pre4_bcast_ub = T.alloc_ub((hc_mult, hidden_block), accum_dtype)
                ol_ub = T.alloc_ub(hidden_block, accum_dtype)
                ol_bf16_ub = T.alloc_ub(hidden_block, dtype)

                for tok in T.serial(sub_block_M):
                    if row_base + tok < num_tokens:
                        # ---- pre / post (V3: 32B 1D UB->UB extract) ----
                        T.copy(mixes_ub[tok, 0:8], seg8_ub)
                        T.tile.sigmoid(sig8_ub, seg8_ub)
                        T.tile.add(pre8_ub, sig8_ub, hc_pre_eps)
                        T.tile.mul(post8_ub, sig8_ub, hc_post_mult_value)
                        for i in T.serial(hc_mult):
                            pre4_1d_ub[i] = pre8_ub[i]
                        for i in T.serial(hc_mult):
                            post1d_ub[i] = post8_ub[hc_mult + i]
                        # post_mix GM write (16B DataCopyPad, v2-proven)
                        T.copy(post1d_ub[0:hc_mult], post_mix[row_base + tok, 0:hc_mult])

                        # ---- comb = sinkhorn(mixes[:, 8:24].reshape(4,4)) ----
                        for i in T.serial(hc_mult):
                            for j in T.serial(hc_mult):
                                comb_ub[i, j] = mixes_ub[tok, 2 * hc_mult + i * hc_mult + j]
                        # iter0: row softmax + eps, then col-norm
                        T.reduce_max(comb_ub, row_max_ub, dim=-1, real_shape=[hc_mult, hc_mult])
                        T.tile.broadcast(row_div_ub, row_max_ub)  # [4] -> [4,8] axis=1
                        T.tile.sub(comb_ub, comb_ub, row_div_ub)
                        T.tile.exp(comb_ub, comb_ub)
                        T.reduce_sum(comb_ub, row_sum_ub, dim=-1, real_shape=[hc_mult, hc_mult])
                        T.tile.broadcast(row_div_ub, row_sum_ub)
                        T.tile.div(comb_ub, comb_ub, row_div_ub)
                        T.tile.add(comb_ub, comb_ub, hc_sinkhorn_eps)
                        for j in T.serial(hc_mult):
                            col_sum_ub[0, j] = 0.0
                            for i in T.serial(hc_mult):
                                col_sum_ub[0, j] = col_sum_ub[0, j] + comb_ub[i, j]
                        T.tile.add(col_sum_ub, col_sum_ub, hc_sinkhorn_eps)
                        T.tile.broadcast(col_bcast_ub, col_sum_ub)  # [1,8] -> [4,8]
                        T.tile.div(comb_ub, comb_ub, col_bcast_ub)
                        for _ in T.serial(sinkhorn_repeat - 1):
                            T.reduce_sum(comb_ub, row_sum_ub, dim=-1, real_shape=[hc_mult, hc_mult])
                            T.tile.add(row_sum_ub, row_sum_ub, hc_sinkhorn_eps)
                            T.tile.broadcast(row_div_ub, row_sum_ub)
                            T.tile.div(comb_ub, comb_ub, row_div_ub)
                            for j in T.serial(hc_mult):
                                col_sum_ub[0, j] = 0.0
                                for i in T.serial(hc_mult):
                                    col_sum_ub[0, j] = col_sum_ub[0, j] + comb_ub[i, j]
                            T.tile.add(col_sum_ub, col_sum_ub, hc_sinkhorn_eps)
                            T.tile.broadcast(col_bcast_ub, col_sum_ub)
                            T.tile.div(comb_ub, comb_ub, col_bcast_ub)
                        for i in T.serial(hc_mult):
                            for j in T.serial(hc_mult):
                                comb_flat_ub[i * hc_mult + j] = comb_ub[i, j]
                        T.copy(comb_flat_ub, comb_mix[row_base + tok, 0:hc_mult2])

                        # ---- layer_input = bf16(sum_i pre[i] * residual[i]) ----
                        # V5: batched [4, hb]: broadcast pre4 -> mul -> reduce(dim=0)
                        T.tile.broadcast(pre4_bcast_ub, pre4_1d_ub)  # [4] -> [4,hb] axis=1
                        for hb in T.Pipelined(h_num, num_stages=2):
                            h_off = hb * hidden_block
                            T.copy(
                                residual_bf16[
                                    row_base + tok,
                                    0:hc_mult,
                                    h_off : h_off + hidden_block,
                                ],
                                res_2d_ub,
                            )
                            # P0: 2D cross-dtype copy bf16->fp32 (replaces 4 scalar copies)
                            T.copy(res_2d_ub, res_fp32_ub)
                            T.tile.mul(res_fp32_ub, res_fp32_ub, pre4_bcast_ub)  # in-place
                            T.reduce_sum(res_fp32_ub, ol_ub, dim=0)  # [4,hb] -> [hb]
                            T.copy(ol_ub, ol_bf16_ub)  # fp32 -> bf16 (1D cross)
                            T.copy(ol_bf16_ub, layer_input[row_base + tok, h_off])

    return main


# ===========================================================================
# Host-side entry: mhc_pre (interface identical to v2 — example/test reuse)
# ===========================================================================


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass for mHC pre block (Ascend NPU, rev3 single fused kernel)."""
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32
    assert n_splits == 1, "Ascend version does not support split-k"

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size

    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]
    # R3-iter1: residual_flat / x_bf16 are pure views (alias storage); cache the
    # view objects keyed by (data_ptr, shape) to avoid 2 tensor-view
    # constructions per call. Contents semantics unchanged (views never copy).
    view_key = (residual.data_ptr(), residual.shape, hc_mult, hidden_size)
    if view_key in _residual_view_cache:
        residual_flat, x_bf16, num_tokens = _residual_view_cache[view_key]
    else:
        residual_flat = residual.view(-1, hc_mult, hidden_size)
        num_tokens = residual_flat.shape[0]
        x_bf16 = residual_flat.view(num_tokens, hc_hidden_size)
        _residual_view_cache[view_key] = (residual_flat, x_bf16, num_tokens)
    device = residual.device

    # ---- Host-side preparations (free views + caches, rev3) ----
    # x_bf16: 2D flat view [N, D] for the GEMM (Cube) / sqrsum reads; aliases
    # residual_flat's memory — same pattern as v2 (K1 got x_bf16, K2 got 3D view).

    # fn_bf16: pad fn from [24, D] to [32, D] (last 8 rows = 0) + fp32 -> bf16 cast.
    # Cached keyed by fn.data_ptr() (iter12/13 Direction D, fn is a model weight).
    cache_key = (fn.data_ptr(), hc_hidden_size)
    if cache_key in _fn_bf16_cache:
        fn_bf16 = _fn_bf16_cache[cache_key]
    else:
        fn_bf16 = torch.zeros(32, hc_hidden_size, dtype=torch.bfloat16, device=device)
        fn_bf16[:hc_mult3, :] = fn.to(torch.bfloat16)
        _fn_bf16_cache[cache_key] = fn_bf16

    # V8 (R3-iter1): host builds hc_scale_base [2, 24] (row0 = expanded
    # hc_scale, row1 = hc_base) once per weight. Kernel does TWO GM->UB row
    # copies of [24] instead of [24] GM->UB + 24 scalar fills per block (v2).
    scale_key = (hc_scale.data_ptr(), hc_base.data_ptr(), hc_mult)
    if scale_key in _scale_base_cache:
        hc_scale_base = _scale_base_cache[scale_key]
    else:
        hc_scale_exp = torch.cat(
            [
                hc_scale[0].expand(hc_mult),
                hc_scale[1].expand(hc_mult),
                hc_scale[2].expand(hc_mult * hc_mult),
            ]
        ).contiguous()
        hc_scale_base = torch.stack([hc_scale_exp, hc_base]).contiguous()
        _scale_base_cache[scale_key] = hc_scale_base

    # hidden_block selection (v2-proven): 512 default; gcd-adapt when H not
    # divisible; prime H falls back to 512 with ceildiv tail (v2 behavior).
    hidden_block = _hidden_block_cache.get(hidden_size)
    if hidden_block is None:
        hidden_block = 512
        if hidden_size % hidden_block != 0:
            hidden_block = math.gcd(hidden_block, hidden_size)
        if hidden_block < 32:
            hidden_block = 512
        _hidden_block_cache[hidden_size] = hidden_block

    # ---- Run rev3 single fused kernel (all in one launch) ----
    block_N_pad = 32
    gemm_out_gm = torch.zeros(num_tokens, block_N_pad, dtype=torch.float, device=device)
    kernel_fused = mhc_pre_fused(
        num_tokens,
        hidden_size,
        hc_mult,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        hidden_block=hidden_block,
    )
    post_mix, comb_mix, layer_input = kernel_fused(x_bf16, residual_flat, fn_bf16, hc_scale_base, gemm_out_gm)
    # NOTE: no final torch.npu.synchronize() — NPU stream guarantees in-order
    # execution; callers needing results on CPU trigger implicit sync. (iter11/14c)

    # ---- Reshape outputs (identical to v2) ----
    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


if __name__ == "__main__":
    torch.set_default_device("npu")
    torch.manual_seed(42)

    n, hc_mult, hidden_size = 128, 4, 1024
    hc_mult3 = hc_mult * (2 + hc_mult)

    residual = torch.randn(n, hc_mult, hidden_size, dtype=torch.float).bfloat16()
    fn = torch.randn(hc_mult3, hc_mult * hidden_size, dtype=torch.float) * 1e-4
    hc_scale = torch.randn(3, dtype=torch.float) * 0.1
    hc_base = torch.randn(hc_mult3, dtype=torch.float) * 0.1

    post_mix, comb_mix, layer_input = mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0,
        sinkhorn_repeat=10,
    )

    assert post_mix.shape == (n, hc_mult, 1)
    assert comb_mix.shape == (n, hc_mult, hc_mult)
    assert layer_input.shape == (n, hidden_size)
    assert torch.isfinite(post_mix).all()
    assert torch.isfinite(comb_mix).all()
    assert torch.isfinite(layer_input).all()

    # ---- Small precision check against golden reference ----
    residual_flat = residual.view(-1, hc_mult, hidden_size).float()
    D = hc_mult * hidden_size
    sqrsum = residual_flat.view(n, D).square().sum(-1)
    mixes = residual_flat.view(n, D) @ fn.T.float() * (sqrsum.unsqueeze(-1) / D + 1e-6).rsqrt()
    scale_exp = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ]
    )
    mixes = mixes * scale_exp + hc_base
    post_ref = torch.sigmoid(mixes[:, hc_mult : 2 * hc_mult])
    post_err = (post_mix.view(n, hc_mult).float() - post_ref).abs().max().item()
    assert post_err < 1e-3, f"post_mix precision check failed: err={post_err:.4e}"
    print(f"  post_mix max err: {post_err:.4e}")
    print("Test Passed!")
