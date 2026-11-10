"""MHC Post operator for Ascend NPU.

Implements: output = x * post_layer_mix + comb_res_mix^T @ residual

Reference: tilelang main repo CUDA version examples/deepseek_mhc/example_mhc_post.py

Architecture (pure Vector, no Cube):
  Single AIV kernel with dual-V-core partitioning.
  - hc 1-8 (JIT parameter, tested range); AXPY linear combination for the
    [hc, hc] @ [hc, h] matrix multiply (comb^T @ residual)
  - Unified UB layout: 2D res (merged copy, 1 T.copy instead of hc copies) +
    2D bf16 out (merged store, 1 MTE3 instead of hc), 1D fp32 out reused per
    row. comb aligned to 32B/row for correct 2D scalar read.
  - h_blk upper-bounded from UB budget per hc (see _max_h_blk).
  - Non-dividing h handled in-kernel via pad_value + TL_ASCEND_TAIL_MASK
    (no host-side padding).
  - FP32 inputs for post/comb used directly (no BF16 quantization)

Performance: see examples/mhc_post/benchmark.md.
"""

import tilelang
import tilelang.language as T
import torch

VEC_NUM = 2
H_BLK = 2048

_H_BLK_CANDIDATES = [3584, 3072, 2560, 2048, 1024, 512]

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
}


# ============================================================
# Kernel: 2D res merged load + 2D bf16 merged store
# ============================================================


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def mhc_post_kernel(hc, h, h_blk=H_BLK, dtype="bfloat16", accum_dtype="float"):
    """Unified kernel: 2D res merged load + 2D bf16 merged store, any h, hc 1-8.

    2D res keeps 1 T.copy (vs hc copies) for MTE efficiency; 2D bf16 out keeps
    1 merged MTE3 store (vs hc stores) while the 1D fp32 out is reused per row.
    comb aligned to comb_row_stride (32B/row) for correct 2D scalar read.
    Non-dividing h is handled in-kernel via pad_value on every tile copy
    (no-op for full tiles, zero-fills the gap on the tail tile) plus
    TL_ASCEND_TAIL_MASK which rewrites vector ops to compute only the valid
    region. No host-side padding is needed.
    """
    n = T.symbolic("n")
    comb_row_stride = (hc + 7) // 8 * 8
    total_tiles = (h + h_blk - 1) // h_blk
    pad_h = total_tiles * h_blk

    @T.prim_func
    def main(
        x: T.Tensor((n, h), dtype),
        post: T.Tensor((n, hc), accum_dtype),
        comb: T.Tensor((n, hc, hc), accum_dtype),
        residual: T.Tensor((n, hc, h), dtype),
        output: T.Tensor((n, hc, pad_h), dtype),
    ):
        with T.Kernel(T.ceildiv(n, VEC_NUM), is_npu=True) as (cid, vid):
            bid = cid * VEC_NUM + vid

            if bid < n:
                with T.Scope("V"):
                    post_fp32 = T.alloc_ub(hc, accum_dtype)
                    T.copy(post[bid, 0:hc], post_fp32)

                    comb_fp32 = T.alloc_ub((hc, comb_row_stride), accum_dtype)
                    T.copy(comb[bid, 0:hc, 0:hc], comb_fp32[0:hc, 0:hc])

                    res_ub = T.alloc_ub((hc, h_blk), dtype)
                    res_fp32 = T.alloc_ub((hc, h_blk), accum_dtype)
                    x_ub = T.alloc_ub(h_blk, dtype)
                    x_fp32 = T.alloc_ub(h_blk, accum_dtype)
                    out_fp32 = T.alloc_ub(h_blk, accum_dtype)
                    out_bf16 = T.alloc_ub((hc, h_blk), dtype)

                    for i_h in T.Pipelined(total_tiles, num_stages=2):
                        h_start = i_h * h_blk

                        T.copy(residual[bid, 0:hc, h_start : h_start + h_blk], res_ub, pad_value=0.0)
                        T.tile.cast(res_fp32, res_ub, "CAST_NONE", h_blk * hc)

                        T.copy(x[bid, h_start : h_start + h_blk], x_ub, pad_value=0.0)
                        T.tile.cast(x_fp32, x_ub, "CAST_NONE", h_blk)

                        for out_idx in T.unroll(hc):
                            T.tile.mul(out_fp32, x_fp32, post_fp32[out_idx])
                            for res_idx in T.unroll(hc):
                                T.tile.axpy(out_fp32, res_fp32[res_idx, :], comb_fp32[res_idx, out_idx])
                            T.tile.cast(out_bf16[out_idx, :], out_fp32, "CAST_RINT", h_blk)

                        T.copy(out_bf16, output[bid, 0:hc, h_start : h_start + h_blk])

    return main


# ============================================================
# Host-side adapter
# ============================================================

_kernel_cache = {}


def _max_h_blk(hc):
    """Conservative upper-bound h_blk from UB budget (192KB) for a given hc.

    Dominant buffers scale as hc*h_blk: res_ub (2*hc), res_fp32 (4*hc),
    out_bf16 (2*hc) bytes per element, plus ~10 bytes/elem for x/out 1D.
    This is a selection heuristic, not a hard guarantee: T.Pipelined
    double-buffering and compiler layout decisions affect the real footprint.
    Validated for hc 1-8 via compile + accuracy tests.
    """
    per_elem = 8 * hc + 10
    return (192 * 1024 - 8192) // per_elem


def _select_h_blk(h, hc):
    """Largest tuned candidate that divides h and fits UB for hc."""
    candidates = [b for b in _H_BLK_CANDIDATES if b <= _max_h_blk(hc)]
    for blk in candidates:
        if h % blk == 0:
            return blk
    if not candidates:
        raise ValueError(f"No h_blk candidate fits UB budget for hc={hc}")
    return candidates[-1]


def _get_kernel(hc, h, h_blk):
    key = (hc, h, h_blk)
    if key not in _kernel_cache:
        _kernel_cache[key] = mhc_post_kernel(hc, h, h_blk=h_blk)
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
    """
    h = x.shape[1]
    hc = residual.shape[1]
    assert 1 <= hc <= 8, f"hc must be in [1, 8] (tested range), got hc={hc}"
    assert post_layer_mix.shape[1] == hc, f"post_layer_mix requires hc={hc}, got {post_layer_mix.shape[1]}"
    assert comb_res_mix.shape[1] == hc and comb_res_mix.shape[2] == hc, (
        f"comb_res_mix requires [hc, hc]=[{hc}, {hc}], got {comb_res_mix.shape[1:]}"
    )
    assert residual.shape[0] == x.shape[0] and residual.shape[2] == x.shape[1]
    assert post_layer_mix.shape[0] == x.shape[0] and comb_res_mix.shape[0] == x.shape[0]

    h_blk = _select_h_blk(h, hc)

    post_sq = post_layer_mix.squeeze(-1)
    comb_c = comb_res_mix.contiguous()

    kernel = _get_kernel(hc, h, h_blk)
    output = kernel(x, post_sq, comb_c, residual)
    return output[:, :hc, :h]


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
    print("MHC Post test (Ascend NPU - AIV dual-V-core)")
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
        (4, 128, 1),
        (4, 128, 2),
        (4, 128, 3),
        (4, 128, 8),
        (4, 100, 8),
        (4, 1280, 8),
        (16, 512, 8),
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
