# ruff: noqa
import math

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()


# Developer pass_configs
PASS_CONFIGS_DEV = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[4], pass_configs=PASS_CONFIGS_DEV)
def mhc_post_kernel(num_tokens, hidden, hc, h_blk):
    """mhc_post: outer product + A^T B expansion, fp32 accum, bf16 out (Developer)."""
    dtype_bf16 = "bfloat16"
    dtype_fp32 = "float"
    VEC_NUM = 2
    sub_h_blk = h_blk // VEC_NUM  # per-vid column width
    h_num = (hidden + h_blk - 1) // h_blk  # compile-time constant
    TOK_BLK = 16  # tokens per block (Iter2: reduce block count 4096 -> 512)
    m_num = (num_tokens + TOK_BLK - 1) // TOK_BLK

    @T.prim_func
    def main(
        comb_res_mix: T.Tensor([num_tokens, hc, hc], dtype_fp32),  # type: ignore
        residual: T.Tensor([num_tokens, hc, hidden], dtype_bf16),  # type: ignore
        post_layer_mix: T.Tensor([num_tokens, hc], dtype_fp32),  # type: ignore
        x: T.Tensor([num_tokens, hidden], dtype_bf16),  # type: ignore
        out: T.Tensor([num_tokens, hc, hidden], dtype_bf16),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            # ---- per-token coefficient buffers: A rows (1D) + c (reused across tok_i) ----
            a0_ub = T.alloc_ub(hc, dtype_fp32)  # comb_res_mix[tok] row 0
            a1_ub = T.alloc_ub(hc, dtype_fp32)
            a2_ub = T.alloc_ub(hc, dtype_fp32)
            a3_ub = T.alloc_ub(hc, dtype_fp32)
            c_ub = T.alloc_ub(hc, dtype_fp32)  # post_layer_mix[tok]

            # ---- h-block work buffers (per vid, reused across tok_i) ----
            b_bf16 = T.alloc_ub((hc, sub_h_blk), dtype_bf16)
            d_bf16 = T.alloc_ub(sub_h_blk, dtype_bf16)
            b_fp32 = T.alloc_ub((hc, sub_h_blk), dtype_fp32)
            d_fp32 = T.alloc_ub(sub_h_blk, dtype_fp32)
            x_row_ub = T.alloc_ub(sub_h_blk, dtype_fp32)
            tmp_ub = T.alloc_ub(sub_h_blk, dtype_fp32)
            out_bf16_2d = T.alloc_ub((hc, sub_h_blk), dtype_bf16)

            for tok_i in T.serial(TOK_BLK):
                tok = cid * TOK_BLK + tok_i
                if tok < num_tokens:
                    T.copy(comb_res_mix[tok, 0, 0], a0_ub)
                    T.copy(comb_res_mix[tok, 1, 0], a1_ub)
                    T.copy(comb_res_mix[tok, 2, 0], a2_ub)
                    T.copy(comb_res_mix[tok, 3, 0], a3_ub)
                    T.copy(post_layer_mix[tok, 0:hc], c_ub)

                    for hb in T.serial(h_num):
                        h_off = hb * h_blk + vid * sub_h_blk  # this vid's column start
                        T.copy(residual[tok, 0:hc, h_off : h_off + sub_h_blk], b_bf16)
                        T.copy(x[tok, h_off : h_off + sub_h_blk], d_bf16)
                        # bf16 -> fp32 lossless upcast (1D cross-dtype UB->UB)
                        for i in T.serial(hc):
                            T.copy(b_bf16[i, :], b_fp32[i, :])
                        T.copy(d_bf16, d_fp32)

                        # per output row i: x[j] = c[i]*d[j] + sum_k a[k,i]*b[k,j]
                        for i_hco in T.serial(hc):
                            T.tile.mul(x_row_ub, b_fp32[0, :], a0_ub[i_hco])
                            T.tile.mul(tmp_ub, b_fp32[1, :], a1_ub[i_hco])
                            T.tile.add(x_row_ub, x_row_ub, tmp_ub)
                            T.tile.mul(tmp_ub, b_fp32[2, :], a2_ub[i_hco])
                            T.tile.add(x_row_ub, x_row_ub, tmp_ub)
                            T.tile.mul(tmp_ub, b_fp32[3, :], a3_ub[i_hco])
                            T.tile.add(x_row_ub, x_row_ub, tmp_ub)
                            T.tile.mul(tmp_ub, d_fp32, c_ub[i_hco])
                            T.tile.add(x_row_ub, x_row_ub, tmp_ub)

                            T.copy(x_row_ub, out_bf16_2d[i_hco, :])

                        # 4 rows cast complete -> single 2D GM write (10KB burst,
                        # replacing 4x2.5KB small MTE3 writes; iter10 optimization)
                        T.copy(out_bf16_2d, out[tok, 0:hc, h_off : h_off + sub_h_blk])

    return main


# ===========================================================================
# Kernel compile helper
# ===========================================================================


def compute_h_blk(h):
    """h_blk = largest even divisor of h with h_blk <= 2048 (Iter4: minimize
    h_num to 1-2, maximize per-block MTE/vector width); fallback gcd(h,1024).

    Examples: 2560->1280 (h_num=2), 1280->1280 (h_num=1), 7168->1024 (h_num=7),
    1344->672 (h_num=2), 2176->1088 (h_num=2), 3328->1664 (h_num=2), 512->512.
    """
    if h <= 0:
        return 1024
    best = 32
    # iterate candidate blocks (even, >=32, <=2048) in decreasing order
    cand = 2560
    while cand >= 32:
        if cand % 2 == 0 and h % cand == 0:
            return cand
        cand -= 2
    # fallback: gcd with 1024 (guarantees evenness via 2^10 factor)
    h_blk = math.gcd(h, 1024)
    if h_blk < 32 or h_blk % 2 != 0:
        h_blk = 1024
    return h_blk


def _golden_mhc_post(x, residual, post_layer_mix, comb_res_mix):
    """PyTorch reference, numerically identical to GPU reference mhc_post_ref.

    On Ascend, aclnnBatchMatMul rejects M=4 (hc=4) with aicore 507015, so the
    mathematically identical expansion (sum_k A[k,i]*B[k,j], fp32) is used.
    """
    term2 = (comb_res_mix.permute(0, 2, 1).unsqueeze(-1) * residual.float().unsqueeze(1)).sum(-2)
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


if __name__ == "__main__":
    n, h, hc = 32, 2560, 4
    h_blk = compute_h_blk(h)
    kernel = mhc_post_kernel(n, h, hc, h_blk)

    torch.manual_seed(0)
    torch.set_default_device("npu")
    x = torch.randn((n, h), dtype=torch.bfloat16)
    residual = torch.randn((n, hc, h), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((n, hc, 1), dtype=torch.float32)
    comb_res_mix = torch.randn((n, hc, hc), dtype=torch.float32)

    print("init successful!")

    out = kernel(comb_res_mix, residual, post_layer_mix.squeeze(-1), x)
    ref = _golden_mhc_post(x, residual, post_layer_mix, comb_res_mix)

    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=1e-2, rtol=5e-3)
    print("Test Passed!")
