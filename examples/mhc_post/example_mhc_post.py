"""MHC Post operator for Ascend NPU.

Implements: output = x * post_layer_mix + comb_res_mix^T @ residual

Reference: tilelang main repo CUDA version examples/deepseek_mhc/example_mhc_post.py

Architecture (pure Vector, no Cube):
  Single AIV kernel with dual-V-core partitioning.
  - hc=4 specialized (hard constraint, assert enforced)
  - AXPY linear combination for small matrix multiply (comb^T @ residual)
  - Unified UB layout: 2D res (merged copy, 1 T.copy instead of 4) + 1D out
    (streaming write-back, smaller UB footprint). Aligned comb [4, 8] enables
    correct 2D scalar read. Single kernel covers all h_blk including 3584.
  - FP32 inputs for post/comb used directly (no BF16 quantization)
  - out_fp32 reuse across the 4 output rows (smaller UB footprint)

Performance: see examples/mhc_post/benchmark.md.
"""

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

VEC_NUM = 2
HC = 4
H_BLK = 2048

_H_BLK_CANDIDATES = [3584, 3072, 2560, 2048, 1024, 512]

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


# ============================================================
# Kernel: unified 2D res (merged copy) + 1D out (streaming)
# ============================================================


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def mhc_post_kernel(pad_h, h_blk=H_BLK, dtype="bfloat16", accum_dtype="float"):
    """Unified kernel: 2D res merged copy + 1D out streaming write-back.

    2D res keeps 1 T.copy (vs 4) for MTE efficiency; 1D out keeps UB footprint
    low so h_blk=3584 fits (a 2D out would overflow). Host F.pad for
    non-dividing h.
    """
    n = T.symbolic("n")
    h_num = T.ceildiv(pad_h, h_blk)

    @T.prim_func
    def main(
        x: T.Tensor((n, pad_h), dtype),
        post: T.Tensor((n, HC), accum_dtype),
        comb: T.Tensor((n, HC, HC), accum_dtype),
        residual: T.Tensor((n, HC, pad_h), dtype),
        output: T.Tensor((n, HC, pad_h), dtype),
    ):
        with T.Kernel(T.ceildiv(n, VEC_NUM), is_npu=True) as (cid, vid):
            bid = cid * VEC_NUM + vid

            if bid < n:
                with T.Scope("V"):
                    post_fp32 = T.alloc_ub(HC, accum_dtype)
                    T.copy(post[bid, 0:HC], post_fp32)

                    comb_fp32 = T.alloc_ub((HC, (HC + 7) // 8 * 8), accum_dtype)
                    T.copy(comb[bid, 0:HC, 0:HC], comb_fp32[0:HC, 0:HC])

                    res_ub = T.alloc_ub((HC, h_blk), dtype)
                    res_fp32 = T.alloc_ub((HC, h_blk), accum_dtype)
                    x_ub = T.alloc_ub(h_blk, dtype)
                    x_fp32 = T.alloc_ub(h_blk, accum_dtype)
                    out_fp32 = T.alloc_ub(h_blk, accum_dtype)
                    out_bf16 = T.alloc_ub(h_blk, dtype)

                    for i_h in T.Pipelined(h_num, num_stages=2):
                        h_start = i_h * h_blk

                        T.copy(residual[bid, 0:HC, h_start : h_start + h_blk], res_ub)
                        T.tile.cast(res_fp32, res_ub, "CAST_NONE", h_blk * HC)

                        T.copy(x[bid, h_start : h_start + h_blk], x_ub)
                        T.tile.cast(x_fp32, x_ub, "CAST_NONE", h_blk)

                        for out_idx in T.unroll(HC):
                            T.tile.mul(out_fp32, x_fp32, post_fp32[out_idx])
                            for res_idx in T.unroll(HC):
                                T.tile.axpy(out_fp32, res_fp32[res_idx, :], comb_fp32[res_idx, out_idx])
                            T.tile.cast(out_bf16, out_fp32, "CAST_RINT", h_blk)
                            T.copy(out_bf16, output[bid, out_idx, h_start])

    return main


# ============================================================
# Host-side adapter
# ============================================================

_kernel_cache = {}


def _select_h_blk(h):
    """Largest candidate that divides h; fall back to H_BLK (host pad)."""
    for blk in _H_BLK_CANDIDATES:
        if h % blk == 0:
            return blk
    return H_BLK


def _get_kernel(pad_h, h_blk):
    key = (pad_h, h_blk)
    if key not in _kernel_cache:
        _kernel_cache[key] = mhc_post_kernel(pad_h, h_blk=h_blk)
    return _kernel_cache[key]


def mhc_post(x, residual, post_layer_mix, comb_res_mix):
    """MHC Post operator entry point.

    Args:
        x:              [n, h]      bf16
        residual:       [n, hc, h]  bf16
        post_layer_mix: [n, hc, 1]  fp32
        comb_res_mix:   [n, hc, hc] fp32

    Returns:
        output: [n, hc, h] bf16

    Note: hc=4 hard constraint (AXPY specialized).
    """
    h = x.shape[1]
    hc = residual.shape[1]
    assert hc == 4, f"This kernel requires hc=4, got residual hc={hc}"
    assert post_layer_mix.shape[1] == 4, f"post_layer_mix requires hc=4, got {post_layer_mix.shape[1]}"
    assert comb_res_mix.shape[1] == 4 and comb_res_mix.shape[2] == 4, f"comb_res_mix requires [hc, hc]=[4, 4], got {comb_res_mix.shape[1:]}"

    h_blk = _select_h_blk(h)
    pad_h = ((h + h_blk - 1) // h_blk) * h_blk

    post_sq = post_layer_mix.squeeze(-1)
    comb_c = comb_res_mix.contiguous()

    kernel = _get_kernel(pad_h, h_blk)

    if pad_h != h:
        x = F.pad(x, (0, pad_h - h))
        residual = F.pad(residual, (0, pad_h - h))

    output = kernel(x, post_sq, comb_c, residual)

    if pad_h != h:
        output = output[:, :hc, :h]
    return output


# ============================================================
# Golden reference
# ============================================================


def mhc_post_ref(x, residual, post_layer_mix, comb_res_mix):
    """PyTorch golden, FP32 path (same as kernel, no BF16 quantization)."""
    h = x.shape[1]
    hc = residual.shape[1]
    comb_t = comb_res_mix.mT.contiguous()
    term2 = torch.bmm(comb_t.float(), residual.float())
    post_fp32 = post_layer_mix.squeeze(-1)
    term1 = post_fp32.unsqueeze(-1) * x.unsqueeze(-2)
    output = (term1 + term2).bfloat16()[:, :hc, :h]
    return output


# ============================================================
# Tests
# ============================================================


def generate_test_data(n, h, hc, device="npu"):
    torch.random.manual_seed(42)
    x = torch.randn((n, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((n, hc, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n, hc, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((n, hc, hc), dtype=torch.float32, device=device)
    return {"x": x, "residual": residual, "post_layer_mix": post_layer_mix, "comb_res_mix": comb_res_mix}


def test():
    print("=" * 60)
    print("MHC Post test (Ascend NPU - AIV dual-V-core, unified 2D res + 1D out UB)")
    print("=" * 60)

    test_cases = [
        (4, 128, 4),
        (8, 256, 4),
        (16, 512, 4),
        (4, 1024, 4),
        (4, 1280, 4),
        (4, 2048, 4),
        (4, 2560, 4),
        (4, 3072, 4),
        (4, 4096, 4),
        (4, 7168, 4),
        (4096, 2560, 4),
        (4, 100, 4),
        (4, 200, 4),
        (4, 300, 4),
        (8, 500, 4),
    ]

    all_passed = True
    for n, h, hc in test_cases:
        print(f"\n--- n={n}, h={h}, hc={hc} ---")
        data = generate_test_data(n, h, hc)
        output = mhc_post(**data)
        ref = mhc_post_ref(**data)

        print(f"  output shape={output.shape}")

        try:
            torch.testing.assert_close(output.cpu(), ref.cpu(), rtol=1e-2, atol=0.2)
            diff = (output.cpu().float() - ref.cpu().float()).abs()
            print(f"  PASSED (max_diff={diff.max().item():.4f}, mean_diff={diff.mean().item():.4f})")
        except AssertionError:
            diff = (output.cpu().float() - ref.cpu().float()).abs()
            print(f"  FAILED (max_diff={diff.max().item():.4f}, mean_diff={diff.mean().item():.4f})")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("Kernel Output Match!")
    else:
        print("Some tests failed.")
    print("=" * 60)


def mhc_post_pytorch_baseline(x, residual, post_layer_mix, comb_res_mix):
    """Pure PyTorch baseline without padding (fair perf comparison)."""
    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    output = (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()
    return output


def bench_vs_pytorch():
    """Benchmark tilelang kernel vs PyTorch baseline."""
    from tilelang.profiler import do_bench

    print("\n" + "=" * 60)
    print("TileLang vs PyTorch benchmark")
    print("=" * 60)

    test_cases = [
        (512, 2560, 4),
        (4096, 2560, 4),
        (4096, 7168, 4),
    ]

    for n, h, hc in test_cases:
        print(f"\n--- n={n}, h={h}, hc={hc} ---")
        data = generate_test_data(n, h, hc)

        t_tl = do_bench(lambda d=data: mhc_post(**d), warmup=20, rep=100)
        t_pt = do_bench(lambda d=data: mhc_post_pytorch_baseline(**d), warmup=20, rep=100)

        print(f"  TileLang (AIV):  {t_tl:.4f} ms")
        print(f"  PyTorch (CANN):  {t_pt:.4f} ms")
        print(f"  Speedup:         {t_pt / t_tl:.2f}x")


if __name__ == "__main__":
    tilelang.disable_cache()
    test()
