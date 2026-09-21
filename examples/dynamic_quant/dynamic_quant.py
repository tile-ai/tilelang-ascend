"""Dynamic quantization: per-token symmetric quantization to int8.

Two-pass algorithm:
  Pass 1 (Max): compute per-row abs-max across N, derive scale = max / 127.
  Pass 2 (Quantize): divide input by scale, round, cast to int8.

Adaptive block dispatch based on N dimension:
  - Tiny  N (≤128):  block_M=128, block_N=128  → minimize GM padding waste
  - Small N (≤512):  block_M=64,  block_N=512  → fewer M-blocks for large M
  - Large N (>512):  block_M=16,  block_N=1024 → fewer N-chunks for large N

Full parallel strategy: core_num = ceil(M / block_M).
"""

import argparse

import tilelang
import torch
from tilelang import language as T

# ── Algorithm constants ──────────────────────────────────────────────────────
DTYPE_MAX = 127.0
VEC_NUM = 2
CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# ── Block dispatch table ─────────────────────────────────────────────────────
# (N_threshold, block_M, block_N)
_DISPATCH = [
    (128, 128, 128),  # Tiny:  N ≤ 128
    (512, 64, 512),  # Small: 128 < N ≤ 512
]
_DEFAULT_BLOCK = (16, 1024)  # Large: N > 512


# ── Single unified kernel (specialized per block_M / block_N at JIT time) ─────
@tilelang.jit(out_idx=[1, 2], pass_configs=pass_configs)
def _dynamic_quant_kernel(M, N, block_M, block_N, core_num, dtype="float16"):
    cal_dtype = "float32"
    sub_block_M = block_M // VEC_NUM
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(m_num, core_num)

    @T.prim_func
    def main(
        x: T.Tensor([M, N], dtype),  # type: ignore
        y_out: T.Tensor([M, N], "int8"),  # type: ignore
        scale_out: T.Tensor([M], "float32"),  # type: ignore
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub([sub_block_M, block_N], dtype)
            a_cal = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            row_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            chunk_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            scale = T.alloc_ub([sub_block_M, 1], cal_dtype)
            y_int8 = T.alloc_ub([sub_block_M, block_N], "int8")

            for bx_idx in T.serial(single_core_load):
                bx = cid * single_core_load + bx_idx
                if bx < m_num:
                    # ════════════════════════════════════════════════════════
                    # Pass 1: Max — find per-row abs-max over all N chunks
                    # ════════════════════════════════════════════════════════
                    T.tile.fill(row_max, 0.0)
                    for n_chunk in T.serial(n_num):
                        T.copy(
                            x[
                                bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                                n_chunk * block_N : (n_chunk + 1) * block_N,
                            ],
                            a_ub,
                            pad_value=0.0,
                        )
                        T.tile.cast(a_cal, a_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        T.tile.abs(a_cal, a_cal)
                        T.reduce_max(a_cal, chunk_max, dim=-1)
                        T.tile.max(row_max, row_max, chunk_max)

                    # Zero-row guard + scale computation
                    T.tile.max(row_max, row_max, 1e-12)
                    T.tile.div(scale, row_max, DTYPE_MAX)

                    # ════════════════════════════════════════════════════════
                    # Pass 2: Quantize — divide, round, cast to int8
                    # ════════════════════════════════════════════════════════
                    for n_chunk in T.serial(n_num):
                        T.copy(
                            x[
                                bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                                n_chunk * block_N : (n_chunk + 1) * block_N,
                            ],
                            a_ub,
                            pad_value=0.0,
                        )
                        T.tile.cast(a_cal, a_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)

                        # Implicit broadcast: scale[sub_block_M, 1] / a_cal[sub_block_M, block_N]
                        for i, j in T.Parallel(sub_block_M, block_N):
                            a_cal[i, j] = a_cal[i, j] / scale[i, 0]

                        T.tile.round(a_cal, a_cal, sub_block_M * block_N)
                        a_fp16 = T.alloc_ub([sub_block_M, block_N], "float16")
                        T.tile.cast(a_fp16, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                        T.tile.cast(y_int8, a_fp16, CAST_MODE_LOW2HIGH, sub_block_M * block_N)

                        T.copy(
                            y_int8,
                            y_out[
                                bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                                n_chunk * block_N : (n_chunk + 1) * block_N,
                            ],
                        )

                    # Write scale to GM (float32)
                    T.copy(
                        scale,
                        scale_out[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M],
                    )

    return main


# ── Public API ────────────────────────────────────────────────────────────────


def dynamic_quant(M: int, N: int, dtype: str = "float16"):
    """Return a JIT-compiled dynamic-quantization kernel for [M, N] inputs.

    Block configuration is selected automatically based on N:
        N ≤ 128  → (block_M, block_N) = (128, 128)
        N ≤ 512  → (block_M, block_N) = ( 64, 512)
        N > 512  → (block_M, block_N) = ( 16, 1024)

    Args:
        M: row count (number of tokens / batch*size dimension).
        N: feature dimension.
        dtype: input element type ('float16' or 'bfloat16').

    Returns:
        A callable kernel: kernel(x_npu) -> (y_int8, scale_float32).
    """
    block_M, block_N = _DEFAULT_BLOCK
    for n_thresh, bm, bn in _DISPATCH:
        if n_thresh >= N:
            block_M, block_N = bm, bn
            break

    m_num = (M + block_M - 1) // block_M
    core_num = m_num  # Full-parallel: one core per M-block

    return _dynamic_quant_kernel(M, N, block_M, block_N, core_num, dtype=dtype)


# ── Golden reference (for __main__ quick-verify; authoritative version is in
#    test_dynamic_quant.py::ref_dynamic_quant) ──────────────────────────────────


def _golden_ref(x: torch.Tensor):
    """Per-token symmetric quantization reference (pure PyTorch)."""
    xf = x.float()
    row_max = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = row_max / DTYPE_MAX
    y = torch.round(xf / scale).clamp(-128, 127).to(torch.int8)
    return y, scale.squeeze(-1).float()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tilelang.cache.clear_cache()

    parser = argparse.ArgumentParser(description="Dynamic Quantization Kernel")
    parser.add_argument("--m", type=int, default=1024, help="Row count M (number of tokens)")
    parser.add_argument("--n", type=int, default=512, help="Feature dimension N")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Input element dtype",
    )
    args = parser.parse_args()

    M, N, dtype_str = args.m, args.n, args.dtype
    x_torch_dtype = getattr(torch, dtype_str)

    # JIT-compile
    func = dynamic_quant(M, N, dtype=dtype_str)

    # Create input
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=x_torch_dtype).npu()
    torch.npu.synchronize()
    print("Kernel compiled. Running...")

    # Run kernel
    y_out, scale_out = func(x)
    torch.npu.synchronize()

    # Compute golden reference (CPU)
    ref_y, ref_scale = _golden_ref(x.cpu())

    # Verify scale (float32): strict fp32 tolerance
    torch.testing.assert_close(
        scale_out.cpu(),
        ref_scale,
        rtol=2**-10,
        atol=2**-16,
    )
    print(f"  scale  PASS  shape={ref_scale.shape}")

    # Verify y (int8): allow ±1 due to rounding path difference
    diff = (y_out.cpu().to(torch.int16) - ref_y.to(torch.int16)).abs()
    max_diff = int(diff.max().item())
    matched = (diff == 0).float().mean().item()
    OVERFLOW_LIMIT = 1
    MATCHED_REQUIRED = 0.99
    if max_diff <= OVERFLOW_LIMIT and matched >= MATCHED_REQUIRED:
        print(f"  y      PASS  max_diff={max_diff}  matched={matched:.4f} (>={MATCHED_REQUIRED})")
    else:
        raise AssertionError(
            f"y precision FAIL: max_diff={max_diff} (limit {OVERFLOW_LIMIT}), matched={matched:.4f} (required >={MATCHED_REQUIRED})"
        )

    print("Kernel Output Match!")
