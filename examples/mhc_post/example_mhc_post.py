"""MHC Post operator for Ascend NPU.

Implements: output = x * post_layer_mix + comb_res_mix^T @ residual

Reference: tilelang main repo CUDA version examples/deepseek_mhc/example_mhc_post.py

Architecture (pure Vector, no Cube):
  Single AIV kernel with dual-V-core partitioning.
  - hc=4 specialized (hard constraint, assert enforced)
  - AXPY linear combination for small matrix multiply (comb^T @ residual)
  - h_blk=2048, h padded to multiple of 2048 via F.pad
  - FP32 inputs for post/comb used directly (no BF16 quantization)

Performance (n=4096, h=7168, hc=4):
  Kernel-only:  1.18 ms
  TileLang E2E: 1.96 ms
  PyTorch CANN: 6.08 ms
  E2E speedup:  3.10x
"""

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

H_BLK = 2048
HC = 4


def calc_pad_h(h, h_blk=H_BLK):
    """Round h up to the next multiple of h_blk."""
    return max(h_blk, ((h + h_blk - 1) // h_blk) * h_blk)


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def mhc_post_kernel(pad_h, h_blk=H_BLK, dtype="bfloat16", accum_dtype="float"):
    """Pure AIV kernel: dual-V-core + AXPY, hc=4 specialized."""
    n = T.symbolic("n")
    h_num = T.ceildiv(pad_h, h_blk)
    VEC_NUM = 2

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
                    T.copy(post[bid, 0], post_fp32)

                    comb0_fp32 = T.alloc_ub(HC, accum_dtype)
                    comb1_fp32 = T.alloc_ub(HC, accum_dtype)
                    comb2_fp32 = T.alloc_ub(HC, accum_dtype)
                    comb3_fp32 = T.alloc_ub(HC, accum_dtype)
                    T.copy(comb[bid, 0, 0], comb0_fp32)
                    T.copy(comb[bid, 1, 0], comb1_fp32)
                    T.copy(comb[bid, 2, 0], comb2_fp32)
                    T.copy(comb[bid, 3, 0], comb3_fp32)

                    res0_ub = T.alloc_ub(h_blk, dtype)
                    res1_ub = T.alloc_ub(h_blk, dtype)
                    res2_ub = T.alloc_ub(h_blk, dtype)
                    res3_ub = T.alloc_ub(h_blk, dtype)
                    res0_fp32 = T.alloc_ub(h_blk, accum_dtype)
                    res1_fp32 = T.alloc_ub(h_blk, accum_dtype)
                    res2_fp32 = T.alloc_ub(h_blk, accum_dtype)
                    res3_fp32 = T.alloc_ub(h_blk, accum_dtype)
                    x_ub = T.alloc_ub(h_blk, dtype)
                    x_fp32 = T.alloc_ub(h_blk, accum_dtype)
                    out0 = T.alloc_ub(h_blk, accum_dtype)
                    out1 = T.alloc_ub(h_blk, accum_dtype)
                    out2 = T.alloc_ub(h_blk, accum_dtype)
                    out3 = T.alloc_ub(h_blk, accum_dtype)
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

                        T.copy(x[bid, i_h * h_blk], x_ub)
                        T.tile.cast(x_fp32, x_ub, "CAST_NONE", h_blk)

                        T.tile.mul(out0, x_fp32, post_fp32[0])
                        T.tile.axpy(out0, res0_fp32, comb0_fp32[0])
                        T.tile.axpy(out0, res1_fp32, comb0_fp32[1])
                        T.tile.axpy(out0, res2_fp32, comb0_fp32[2])
                        T.tile.axpy(out0, res3_fp32, comb0_fp32[3])

                        T.tile.mul(out1, x_fp32, post_fp32[1])
                        T.tile.axpy(out1, res0_fp32, comb1_fp32[0])
                        T.tile.axpy(out1, res1_fp32, comb1_fp32[1])
                        T.tile.axpy(out1, res2_fp32, comb1_fp32[2])
                        T.tile.axpy(out1, res3_fp32, comb1_fp32[3])

                        T.tile.mul(out2, x_fp32, post_fp32[2])
                        T.tile.axpy(out2, res0_fp32, comb2_fp32[0])
                        T.tile.axpy(out2, res1_fp32, comb2_fp32[1])
                        T.tile.axpy(out2, res2_fp32, comb2_fp32[2])
                        T.tile.axpy(out2, res3_fp32, comb2_fp32[3])

                        T.tile.mul(out3, x_fp32, post_fp32[3])
                        T.tile.axpy(out3, res0_fp32, comb3_fp32[0])
                        T.tile.axpy(out3, res1_fp32, comb3_fp32[1])
                        T.tile.axpy(out3, res2_fp32, comb3_fp32[2])
                        T.tile.axpy(out3, res3_fp32, comb3_fp32[3])

                        T.tile.cast(out_bf16, out0, "CAST_RINT", h_blk)
                        T.copy(out_bf16, output[bid, 0, i_h * h_blk])
                        T.tile.cast(out_bf16, out1, "CAST_RINT", h_blk)
                        T.copy(out_bf16, output[bid, 1, i_h * h_blk])
                        T.tile.cast(out_bf16, out2, "CAST_RINT", h_blk)
                        T.copy(out_bf16, output[bid, 2, i_h * h_blk])
                        T.tile.cast(out_bf16, out3, "CAST_RINT", h_blk)
                        T.copy(out_bf16, output[bid, 3, i_h * h_blk])

    return main


# ============================================================
# Host-side adapter
# ============================================================

_kernel_cache = {}


def _get_kernel(pad_h):
    if pad_h not in _kernel_cache:
        _kernel_cache[pad_h] = mhc_post_kernel(pad_h)
    return _kernel_cache[pad_h]


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
    assert hc == 4 and post_layer_mix.shape[1] == 4 and comb_res_mix.shape[1] == 4
    pad_h = calc_pad_h(h)

    post_sq = post_layer_mix.squeeze(-1)
    comb_t = comb_res_mix.mT.contiguous()

    if pad_h != h:
        x = F.pad(x, (0, pad_h - h))
        residual = F.pad(residual, (0, pad_h - h))

    kernel = _get_kernel(pad_h)
    output = kernel(x, post_sq, comb_t, residual)

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
    pad_h = calc_pad_h(h)

    comb_t = comb_res_mix.mT.contiguous()
    residual_padded = F.pad(residual, (0, pad_h - h))
    term2 = torch.bmm(comb_t.float(), residual_padded.float())

    post_fp32 = post_layer_mix.squeeze(-1)
    x_padded = F.pad(x.float(), (0, pad_h - h))
    term1 = post_fp32.unsqueeze(-1) * x_padded.unsqueeze(-2)
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
    print("MHC Post test (Ascend NPU - AIV dual-V-core)")
    print("=" * 60)

    test_cases = [
        (4, 128, 4),
        (8, 256, 4),
        (16, 512, 4),
        (4, 1280, 4),
        (4, 2560, 4),
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
