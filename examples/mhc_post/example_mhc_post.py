"""MHC Post operator for Ascend NPU.

Implements: output = x * post_layer_mix + comb_res_mix^T @ residual

Reference: tilelang main repo CUDA version examples/deepseek_mhc/example_mhc_post.py

Architecture (pure Vector, no Cube):
  Single AIV kernel with dual-V-core partitioning.
  - hc=4 specialized (hard constraint, assert enforced)
  - AXPY linear combination for small matrix multiply (comb^T @ residual)
  - Hybrid UB layout dispatch (selects per-shape at JIT time):
    * 2D UB path: merged res/out copy (1 T.copy instead of 4), aligned comb
      ([4, 8] for correct 2D scalar read). No host-side pad. Used when h
      divides a candidate in _H_BLK_CANDIDATES_2D (<= 3072). Faster due to
      fewer MTE2/MTE3 launch overheads.
    * 1D UB path: separate per-row res/out buffers (smaller UB footprint,
      supports h_blk=3584). Host-side F.pad for non-dividing h. Used when h
      requires 3584 (e.g. h=7168) or h does not divide any 2D candidate.
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

_H_BLK_CANDIDATES_2D = [3072, 2560, 2048, 1024, 512]
_H_BLK_CANDIDATES_1D = [3584, 3072, 2560, 2048, 1024, 512]

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


# ============================================================
# Kernel 1: 2D UB path (merged copy, no host pad)
# ============================================================


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def mhc_post_kernel_2d(h, h_blk=H_BLK, dtype="bfloat16", accum_dtype="float"):
    """2D UB kernel: merged res/out copy, aligned comb ([4,8] for 2D scalar read).

    Requires h % h_blk == 0 (no tail tile, no host-side pad).
    """
    n = T.symbolic("n")
    h_num = T.ceildiv(h, h_blk)

    @T.prim_func
    def main(
        x: T.Tensor((n, h), dtype),
        post: T.Tensor((n, HC), accum_dtype),
        comb: T.Tensor((n, HC, HC), accum_dtype),
        residual: T.Tensor((n, HC, h), dtype),
        output: T.Tensor((n, HC, h), dtype),
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
                    out_fp32 = T.alloc_ub((HC, h_blk), accum_dtype)
                    out_bf16 = T.alloc_ub((HC, h_blk), dtype)

                    for i_h in T.Pipelined(h_num, num_stages=2):
                        h_start = i_h * h_blk

                        T.copy(residual[bid, 0:HC, h_start : h_start + h_blk], res_ub)
                        T.tile.cast(res_fp32, res_ub, "CAST_NONE", h_blk * HC)
                        T.copy(x[bid, h_start : h_start + h_blk], x_ub)
                        T.tile.cast(x_fp32, x_ub, "CAST_NONE", h_blk)

                        for out_idx in T.unroll(HC):
                            T.tile.mul(out_fp32[out_idx, :], x_fp32, post_fp32[out_idx])
                            for res_idx in T.unroll(HC):
                                T.tile.axpy(out_fp32[out_idx, :], res_fp32[res_idx, :], comb_fp32[res_idx, out_idx])

                        T.tile.cast(out_bf16, out_fp32, "CAST_RINT", HC * h_blk)
                        T.copy(out_bf16, output[bid, 0:HC, h_start : h_start + h_blk])

    return main


# ============================================================
# Kernel 2: 1D UB path (separate buffers, host pad for tail)
# ============================================================


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def mhc_post_kernel_1d(pad_h, h_blk=H_BLK, dtype="bfloat16", accum_dtype="float"):
    """1D UB kernel: separate per-row res/out buffers, smaller UB footprint.

    Supports h_blk=3584 (2D UB overflows at this size). Host-side F.pad
    handles non-dividing h.
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
                    out_fp32 = T.alloc_ub(h_blk, accum_dtype)
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

                        for out_idx in T.unroll(HC):
                            T.tile.mul(out_fp32, x_fp32, post_fp32[out_idx])
                            T.tile.axpy(out_fp32, res0_fp32, comb0_fp32[out_idx])
                            T.tile.axpy(out_fp32, res1_fp32, comb1_fp32[out_idx])
                            T.tile.axpy(out_fp32, res2_fp32, comb2_fp32[out_idx])
                            T.tile.axpy(out_fp32, res3_fp32, comb3_fp32[out_idx])
                            T.tile.cast(out_bf16, out_fp32, "CAST_RINT", h_blk)
                            T.copy(out_bf16, output[bid, out_idx, i_h * h_blk])

    return main


# ============================================================
# Host-side adapter
# ============================================================

_kernel_cache = {}


_ALL_CANDIDATES = sorted(set(_H_BLK_CANDIDATES_2D + _H_BLK_CANDIDATES_1D), reverse=True)


def _select_path(h):
    """Select kernel path and h_blk for the given h.

    Returns (path, h_blk, pad_h):
      - path='2d': 2D UB kernel, pad_h == h (no host pad needed)
      - path='1d': 1D UB kernel, pad_h >= h (host F.pad applied when pad_h > h)

    Dispatch rule:
      1. Find the largest candidate that divides h (from both lists combined)
      2. If that candidate is in the 2D list -> 2D path (merged copy, faster)
      3. If that candidate is only in the 1D list (i.e. 3584) -> 1D path
      4. If no candidate divides h -> 1D path with H_BLK and host pad (safe)
    """
    for blk in _ALL_CANDIDATES:
        if h % blk == 0:
            path = "2d" if blk in _H_BLK_CANDIDATES_2D else "1d"
            return path, blk, h
    pad_h = ((h + H_BLK - 1) // H_BLK) * H_BLK
    return "1d", H_BLK, pad_h


def _get_kernel(path, pad_h, h_blk):
    key = (path, pad_h, h_blk)
    if key not in _kernel_cache:
        if path == "2d":
            _kernel_cache[key] = mhc_post_kernel_2d(pad_h, h_blk=h_blk)
        else:
            _kernel_cache[key] = mhc_post_kernel_1d(pad_h, h_blk=h_blk)
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

    path, h_blk, pad_h = _select_path(h)

    post_sq = post_layer_mix.squeeze(-1)
    comb_c = comb_res_mix.contiguous()

    kernel = _get_kernel(path, pad_h, h_blk)

    if path == "1d" and pad_h != h:
        x = F.pad(x, (0, pad_h - h))
        residual = F.pad(residual, (0, pad_h - h))

    output = kernel(x, post_sq, comb_c, residual)

    if path == "1d" and pad_h != h:
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
    print("MHC Post test (Ascend NPU - AIV dual-V-core, hybrid 2D/1D UB)")
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
