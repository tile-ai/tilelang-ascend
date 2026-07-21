"""
Fused Add + RMS Norm + Dynamic Quantization operator for Ascend NPU.

2-pass architecture with hybrid kernel dispatch:
  - M < 1024: single kernel (block_M=16, no dispatch overhead)
  - M >= 1024: dual kernel (block_M=32, readback/recompute based on input range)

Pass 1: sum_sq + abs_max(|h*gamma|) + write x_out
  Math: max(|h*inv_rms*gamma|) = inv_rms * max(|h*gamma|)
Pass 2: quantize (readback or recompute h)

All kernels use Double Buffer with 3-stage pipeline (prefetch/main/epilogue).
Adaptive block_N: if H < 256 and H % 16 == 0, block_N = H (eliminates tail block).

Outputs: INT8 quantized data, FP32 scale per token, FP16/BF16 residual add result.

Programming mode: Hybrid (Expert alloc_ub + Developer pass_configs).
Reference: examples/normalization/rms_norm.py, examples/deepseek_v4/act_quant.py.
"""

import argparse
import sys

import torch

import tilelang
from tilelang import language as T


# ============================================================================
# Constants
# ============================================================================

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

HYBRID_THRESHOLD = 1024

_kernel_cache = {}


def _torch_dtype_to_tl(dtype):
    if dtype == torch.float16:
        return "float16"
    elif dtype == torch.bfloat16:
        return "bfloat16"
    elif dtype == torch.float32:
        return "float"
    raise ValueError(f"Unsupported dtype: {dtype}")


# ============================================================================
# Dual Kernel: block_M=32, readback
# Pass 2 reads x_out (fp16) instead of recomputing h from x1+x2
# Used for M >= 1024 when max(|x1|, |x2|) < 100
# ============================================================================


@tilelang.jit(out_idx=[-3, -2, -1], pass_configs=PASS_CONFIGS)
def _kernel_readback(M, H, block_M, block_N, eps, dtype="float16"):
    VEC_NUM = 2
    ROWS = block_M // VEC_NUM
    tile_elements = ROWS * block_N
    stages = 2
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(H, block_N)

    @T.macro
    def init_flags():
        T.set_flag("mte3", "mte2", 0)
        T.set_flag("mte3", "mte2", 1)

    @T.macro
    def drain_flags():
        T.wait_flag("mte3", "mte2", 0)
        T.wait_flag("mte3", "mte2", 1)

    @T.prim_func
    def main(
        x1: T.Tensor((M, H), dtype),
        x2: T.Tensor((M, H), dtype),
        gamma: T.Tensor((H,), dtype),
        output: T.Tensor((M, H), "int8"),
        x_out: T.Tensor((M, H), dtype),
        scale_out: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_start = cid * block_M + vid * ROWS

            x1_ub = T.alloc_ub([stages, ROWS, block_N], dtype)
            x2_ub = T.alloc_ub([stages, ROWS, block_N], dtype)
            gamma_ub = T.alloc_ub([stages, block_N], dtype)
            x1_fp32 = T.alloc_ub([ROWS, block_N], "float32")
            x2_fp32 = T.alloc_ub([ROWS, block_N], "float32")
            h_fp32 = T.alloc_ub([ROWS, block_N], "float32")
            hw_fp32 = T.alloc_ub([ROWS, block_N], "float32")
            out_dtype_ub = T.alloc_ub([ROWS, block_N], dtype)
            out_fp16 = T.alloc_ub([ROWS, block_N], "float16")
            out_int8 = T.alloc_ub([stages, ROWS, block_N], "int8")
            sum_sq_acc = T.alloc_ub([ROWS, block_N], "float32")
            sum_sq_row = T.alloc_ub([ROWS, 1], "float32")
            inv_rms_ub = T.alloc_ub([ROWS, 1], "float32")
            inv_rms_tile = T.alloc_ub([ROWS, block_N], "float32")
            nr_temp = T.alloc_ub([ROWS, 1], "float32")
            gamma_fp32 = T.alloc_ub([block_N], "float32")
            gamma_tile = T.alloc_ub([ROWS, block_N], "float32")
            abs_ub = T.alloc_ub([ROWS, block_N], "float32")
            tile_max = T.alloc_ub([ROWS, 1], "float32")
            abs_max = T.alloc_ub([ROWS, 1], "float32")
            scale_ub = T.alloc_ub([ROWS, 1], "float32")
            scale_tile = T.alloc_ub([ROWS, block_N], "float32")
            min_val = T.alloc_ub([ROWS, 1], "float32")
            xout_ub = T.alloc_ub([stages, ROWS, block_N], dtype)
            gamma_p2_ub = T.alloc_ub([stages, block_N], dtype)

            T.tile.fill(sum_sq_acc, 0.0)
            T.tile.fill(abs_max, 0.0)
            init_flags()

            # --- Pass 1: sum_sq + abs_max(|h*gamma|) + write x_out ---
            if n_num > 0:
                T.wait_flag("mte3", "mte2", 0)
                T.tile.fill(x1_ub[0, :, :], 0.0)
                T.tile.fill(x2_ub[0, :, :], 0.0)
                T.tile.fill(gamma_ub[0, :], 0.0)
                T.copy(x1[row_start:row_start + ROWS, 0:block_N], x1_ub[0, :, :])
                T.copy(x2[row_start:row_start + ROWS, 0:block_N], x2_ub[0, :, :])
                T.copy(gamma[0:block_N], gamma_ub[0, :])
                T.set_flag("mte2", "v", 0)

            for by in T.serial(0, n_num - 1):
                cur = by % stages
                nxt = (by + 1) % stages
                col_off_cur = by * block_N
                col_off_nxt = (by + 1) * block_N

                T.wait_flag("mte3", "mte2", nxt)
                T.tile.fill(x1_ub[nxt, :, :], 0.0)
                T.tile.fill(x2_ub[nxt, :, :], 0.0)
                T.tile.fill(gamma_ub[nxt, :], 0.0)
                T.copy(
                    x1[row_start:row_start + ROWS, col_off_nxt:col_off_nxt + block_N],
                    x1_ub[nxt, :, :],
                )
                T.copy(
                    x2[row_start:row_start + ROWS, col_off_nxt:col_off_nxt + block_N],
                    x2_ub[nxt, :, :],
                )
                T.copy(gamma[col_off_nxt:col_off_nxt + block_N], gamma_ub[nxt, :])
                T.set_flag("mte2", "v", nxt)

                T.wait_flag("mte2", "v", cur)
                T.tile.cast(x1_fp32, x1_ub[cur, :, :], "CAST_NONE", tile_elements)
                T.tile.cast(x2_fp32, x2_ub[cur, :, :], "CAST_NONE", tile_elements)
                T.tile.add(h_fp32, x1_fp32, x2_fp32)
                T.tile.cast(out_dtype_ub, h_fp32, "CAST_RINT", tile_elements)
                T.tile.mul_add_dst(sum_sq_acc, h_fp32, h_fp32)
                T.tile.cast(gamma_fp32, gamma_ub[cur, :], "CAST_NONE", block_N)
                T.tile.broadcast(gamma_tile, gamma_fp32)
                T.tile.mul(hw_fp32, h_fp32, gamma_tile)
                T.tile.abs(abs_ub, hw_fp32)
                T.reduce_max(abs_ub, tile_max, dim=-1)
                T.tile.max(abs_max, abs_max, tile_max)
                T.set_flag("v", "mte3", cur)

                T.wait_flag("v", "mte3", cur)
                T.copy(
                    out_dtype_ub,
                    x_out[row_start:row_start + ROWS, col_off_cur:col_off_cur + block_N],
                )
                T.set_flag("mte3", "mte2", cur)

            if n_num > 0:
                last = (n_num - 1) % stages
                col_off_last = (n_num - 1) * block_N
                T.wait_flag("mte2", "v", last)
                T.tile.cast(x1_fp32, x1_ub[last, :, :], "CAST_NONE", tile_elements)
                T.tile.cast(x2_fp32, x2_ub[last, :, :], "CAST_NONE", tile_elements)
                T.tile.add(h_fp32, x1_fp32, x2_fp32)
                T.tile.cast(out_dtype_ub, h_fp32, "CAST_RINT", tile_elements)
                T.tile.mul_add_dst(sum_sq_acc, h_fp32, h_fp32)
                T.tile.cast(gamma_fp32, gamma_ub[last, :], "CAST_NONE", block_N)
                T.tile.broadcast(gamma_tile, gamma_fp32)
                T.tile.mul(hw_fp32, h_fp32, gamma_tile)
                T.tile.abs(abs_ub, hw_fp32)
                T.reduce_max(abs_ub, tile_max, dim=-1)
                T.tile.max(abs_max, abs_max, tile_max)
                T.set_flag("v", "mte3", last)

                T.wait_flag("v", "mte3", last)
                T.copy(
                    out_dtype_ub,
                    x_out[
                        row_start:row_start + ROWS,
                        col_off_last:col_off_last + block_N,
                    ],
                )
                T.set_flag("mte3", "mte2", last)

            drain_flags()

            # --- Reduction: inv_rms + Newton-Raphson + scale ---
            T.reduce_sum(sum_sq_acc, sum_sq_row, dim=-1)
            inv_H = T.cast(1.0 / H, "float32")
            eps_val = T.cast(eps, "float32")
            T.tile.mul(sum_sq_row, sum_sq_row, inv_H)
            T.tile.add(sum_sq_row, sum_sq_row, eps_val)
            T.tile.rsqrt(inv_rms_ub, sum_sq_row)
            T.tile.mul(nr_temp, inv_rms_ub, inv_rms_ub)
            T.tile.mul(nr_temp, nr_temp, sum_sq_row)
            T.tile.mul(nr_temp, nr_temp, -0.5)
            T.tile.add(nr_temp, nr_temp, 1.5)
            T.tile.mul(inv_rms_ub, inv_rms_ub, nr_temp)
            T.tile.mul(nr_temp, inv_rms_ub, inv_rms_ub)
            T.tile.mul(nr_temp, nr_temp, sum_sq_row)
            T.tile.mul(nr_temp, nr_temp, -0.5)
            T.tile.add(nr_temp, nr_temp, 1.5)
            T.tile.mul(inv_rms_ub, inv_rms_ub, nr_temp)
            T.tile.broadcast(inv_rms_tile, inv_rms_ub)

            T.tile.fill(min_val, 1e-12)
            T.tile.max(abs_max, abs_max, min_val)
            T.tile.mul(scale_ub, abs_max, inv_rms_ub)
            T.tile.div(scale_ub, scale_ub, 127.0)

            # --- Pass 2: read x_out + gamma, quantize ---
            init_flags()

            if n_num > 0:
                T.wait_flag("mte3", "mte2", 0)
                T.tile.fill(xout_ub[0, :, :], 0.0)
                T.tile.fill(gamma_p2_ub[0, :], 0.0)
                T.copy(x_out[row_start:row_start + ROWS, 0:block_N], xout_ub[0, :, :])
                T.copy(gamma[0:block_N], gamma_p2_ub[0, :])
                T.set_flag("mte2", "v", 0)

            for by in T.serial(0, n_num - 1):
                cur = by % stages
                nxt = (by + 1) % stages
                col_off_cur = by * block_N
                col_off_nxt = (by + 1) * block_N

                T.wait_flag("mte3", "mte2", nxt)
                T.tile.fill(xout_ub[nxt, :, :], 0.0)
                T.tile.fill(gamma_p2_ub[nxt, :], 0.0)
                T.copy(
                    x_out[
                        row_start:row_start + ROWS,
                        col_off_nxt:col_off_nxt + block_N,
                    ],
                    xout_ub[nxt, :, :],
                )
                T.copy(gamma[col_off_nxt:col_off_nxt + block_N], gamma_p2_ub[nxt, :])
                T.set_flag("mte2", "v", nxt)

                T.wait_flag("mte2", "v", cur)
                T.tile.cast(h_fp32, xout_ub[cur, :, :], "CAST_NONE", tile_elements)
                T.tile.mul(hw_fp32, h_fp32, inv_rms_tile)
                T.tile.cast(gamma_fp32, gamma_p2_ub[cur, :], "CAST_NONE", block_N)
                T.tile.broadcast(gamma_tile, gamma_fp32)
                T.tile.mul(hw_fp32, hw_fp32, gamma_tile)
                T.tile.broadcast(scale_tile, scale_ub)
                T.tile.div(hw_fp32, hw_fp32, scale_tile)
                T.tile.round(hw_fp32, hw_fp32, tile_elements)
                T.tile.clamp(hw_fp32, hw_fp32, -128.0, 127.0, tile_elements)
                T.tile.cast(out_fp16, hw_fp32, "CAST_RINT", tile_elements)
                T.tile.cast(out_int8[cur, :, :], out_fp16, "CAST_NONE", tile_elements)
                T.set_flag("v", "mte3", cur)

                T.wait_flag("v", "mte3", cur)
                T.copy(
                    out_int8[cur, :, :],
                    output[
                        row_start:row_start + ROWS,
                        col_off_cur:col_off_cur + block_N,
                    ],
                )
                T.set_flag("mte3", "mte2", cur)

            if n_num > 0:
                last = (n_num - 1) % stages
                col_off_last = (n_num - 1) * block_N
                T.wait_flag("mte2", "v", last)
                T.tile.cast(h_fp32, xout_ub[last, :, :], "CAST_NONE", tile_elements)
                T.tile.mul(hw_fp32, h_fp32, inv_rms_tile)
                T.tile.cast(gamma_fp32, gamma_p2_ub[last, :], "CAST_NONE", block_N)
                T.tile.broadcast(gamma_tile, gamma_fp32)
                T.tile.mul(hw_fp32, hw_fp32, gamma_tile)
                T.tile.broadcast(scale_tile, scale_ub)
                T.tile.div(hw_fp32, hw_fp32, scale_tile)
                T.tile.round(hw_fp32, hw_fp32, tile_elements)
                T.tile.clamp(hw_fp32, hw_fp32, -128.0, 127.0, tile_elements)
                T.tile.cast(out_fp16, hw_fp32, "CAST_RINT", tile_elements)
                T.tile.cast(
                    out_int8[last, :, :], out_fp16, "CAST_NONE", tile_elements
                )
                T.set_flag("v", "mte3", last)

                T.wait_flag("v", "mte3", last)
                T.copy(
                    out_int8[last, :, :],
                    output[
                        row_start:row_start + ROWS,
                        col_off_last:col_off_last + block_N,
                    ],
                )
                T.set_flag("mte3", "mte2", last)

            drain_flags()
            T.copy(scale_ub, scale_out[row_start:row_start + ROWS])

    return main


# ============================================================================
# Dual Kernel: block_M=32, recompute
# Pass 2 recomputes h from x1+x2 (fp32 precision)
# Used for M >= 1024 when max(|x1|, |x2|) >= 100
# ============================================================================


@tilelang.jit(out_idx=[-3, -2, -1], pass_configs=PASS_CONFIGS)
def _kernel_recompute(M, H, block_M, block_N, eps, dtype="float16"):
    VEC_NUM = 2
    ROWS = block_M // VEC_NUM
    tile_elements = ROWS * block_N
    stages = 2
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(H, block_N)

    @T.macro
    def init_flags():
        T.set_flag("mte3", "mte2", 0)
        T.set_flag("mte3", "mte2", 1)

    @T.macro
    def drain_flags():
        T.wait_flag("mte3", "mte2", 0)
        T.wait_flag("mte3", "mte2", 1)

    @T.prim_func
    def main(
        x1: T.Tensor((M, H), dtype),
        x2: T.Tensor((M, H), dtype),
        gamma: T.Tensor((H,), dtype),
        output: T.Tensor((M, H), "int8"),
        x_out: T.Tensor((M, H), dtype),
        scale_out: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_start = cid * block_M + vid * ROWS

            x1_ub = T.alloc_ub([stages, ROWS, block_N], dtype)
            x2_ub = T.alloc_ub([stages, ROWS, block_N], dtype)
            gamma_ub = T.alloc_ub([stages, block_N], dtype)
            x1_fp32 = T.alloc_ub([ROWS, block_N], "float32")
            x2_fp32 = T.alloc_ub([ROWS, block_N], "float32")
            h_fp32 = T.alloc_ub([ROWS, block_N], "float32")
            hw_fp32 = T.alloc_ub([ROWS, block_N], "float32")
            out_dtype_ub = T.alloc_ub([ROWS, block_N], dtype)
            out_fp16 = T.alloc_ub([ROWS, block_N], "float16")
            out_int8 = T.alloc_ub([stages, ROWS, block_N], "int8")
            sum_sq_acc = T.alloc_ub([ROWS, block_N], "float32")
            sum_sq_row = T.alloc_ub([ROWS, 1], "float32")
            inv_rms_ub = T.alloc_ub([ROWS, 1], "float32")
            inv_rms_tile = T.alloc_ub([ROWS, block_N], "float32")
            nr_temp = T.alloc_ub([ROWS, 1], "float32")
            gamma_fp32 = T.alloc_ub([block_N], "float32")
            gamma_tile = T.alloc_ub([ROWS, block_N], "float32")
            abs_ub = T.alloc_ub([ROWS, block_N], "float32")
            tile_max = T.alloc_ub([ROWS, 1], "float32")
            abs_max = T.alloc_ub([ROWS, 1], "float32")
            scale_ub = T.alloc_ub([ROWS, 1], "float32")
            scale_tile = T.alloc_ub([ROWS, block_N], "float32")
            min_val = T.alloc_ub([ROWS, 1], "float32")

            T.tile.fill(sum_sq_acc, 0.0)
            T.tile.fill(abs_max, 0.0)
            init_flags()

            # --- Pass 1: sum_sq + abs_max(|h*gamma|) + write x_out ---
            if n_num > 0:
                T.wait_flag("mte3", "mte2", 0)
                T.tile.fill(x1_ub[0, :, :], 0.0)
                T.tile.fill(x2_ub[0, :, :], 0.0)
                T.tile.fill(gamma_ub[0, :], 0.0)
                T.copy(x1[row_start:row_start + ROWS, 0:block_N], x1_ub[0, :, :])
                T.copy(x2[row_start:row_start + ROWS, 0:block_N], x2_ub[0, :, :])
                T.copy(gamma[0:block_N], gamma_ub[0, :])
                T.set_flag("mte2", "v", 0)

            for by in T.serial(0, n_num - 1):
                cur = by % stages
                nxt = (by + 1) % stages
                col_off_cur = by * block_N
                col_off_nxt = (by + 1) * block_N

                T.wait_flag("mte3", "mte2", nxt)
                T.tile.fill(x1_ub[nxt, :, :], 0.0)
                T.tile.fill(x2_ub[nxt, :, :], 0.0)
                T.tile.fill(gamma_ub[nxt, :], 0.0)
                T.copy(
                    x1[row_start:row_start + ROWS, col_off_nxt:col_off_nxt + block_N],
                    x1_ub[nxt, :, :],
                )
                T.copy(
                    x2[row_start:row_start + ROWS, col_off_nxt:col_off_nxt + block_N],
                    x2_ub[nxt, :, :],
                )
                T.copy(gamma[col_off_nxt:col_off_nxt + block_N], gamma_ub[nxt, :])
                T.set_flag("mte2", "v", nxt)

                T.wait_flag("mte2", "v", cur)
                T.tile.cast(x1_fp32, x1_ub[cur, :, :], "CAST_NONE", tile_elements)
                T.tile.cast(x2_fp32, x2_ub[cur, :, :], "CAST_NONE", tile_elements)
                T.tile.add(h_fp32, x1_fp32, x2_fp32)
                T.tile.cast(out_dtype_ub, h_fp32, "CAST_RINT", tile_elements)
                T.tile.mul_add_dst(sum_sq_acc, h_fp32, h_fp32)
                T.tile.cast(gamma_fp32, gamma_ub[cur, :], "CAST_NONE", block_N)
                T.tile.broadcast(gamma_tile, gamma_fp32)
                T.tile.mul(hw_fp32, h_fp32, gamma_tile)
                T.tile.abs(abs_ub, hw_fp32)
                T.reduce_max(abs_ub, tile_max, dim=-1)
                T.tile.max(abs_max, abs_max, tile_max)
                T.set_flag("v", "mte3", cur)

                T.wait_flag("v", "mte3", cur)
                T.copy(
                    out_dtype_ub,
                    x_out[row_start:row_start + ROWS, col_off_cur:col_off_cur + block_N],
                )
                T.set_flag("mte3", "mte2", cur)

            if n_num > 0:
                last = (n_num - 1) % stages
                col_off_last = (n_num - 1) * block_N
                T.wait_flag("mte2", "v", last)
                T.tile.cast(x1_fp32, x1_ub[last, :, :], "CAST_NONE", tile_elements)
                T.tile.cast(x2_fp32, x2_ub[last, :, :], "CAST_NONE", tile_elements)
                T.tile.add(h_fp32, x1_fp32, x2_fp32)
                T.tile.cast(out_dtype_ub, h_fp32, "CAST_RINT", tile_elements)
                T.tile.mul_add_dst(sum_sq_acc, h_fp32, h_fp32)
                T.tile.cast(gamma_fp32, gamma_ub[last, :], "CAST_NONE", block_N)
                T.tile.broadcast(gamma_tile, gamma_fp32)
                T.tile.mul(hw_fp32, h_fp32, gamma_tile)
                T.tile.abs(abs_ub, hw_fp32)
                T.reduce_max(abs_ub, tile_max, dim=-1)
                T.tile.max(abs_max, abs_max, tile_max)
                T.set_flag("v", "mte3", last)

                T.wait_flag("v", "mte3", last)
                T.copy(
                    out_dtype_ub,
                    x_out[
                        row_start:row_start + ROWS,
                        col_off_last:col_off_last + block_N,
                    ],
                )
                T.set_flag("mte3", "mte2", last)

            drain_flags()

            # --- Reduction: inv_rms + Newton-Raphson + scale ---
            T.reduce_sum(sum_sq_acc, sum_sq_row, dim=-1)
            inv_H = T.cast(1.0 / H, "float32")
            eps_val = T.cast(eps, "float32")
            T.tile.mul(sum_sq_row, sum_sq_row, inv_H)
            T.tile.add(sum_sq_row, sum_sq_row, eps_val)
            T.tile.rsqrt(inv_rms_ub, sum_sq_row)
            T.tile.mul(nr_temp, inv_rms_ub, inv_rms_ub)
            T.tile.mul(nr_temp, nr_temp, sum_sq_row)
            T.tile.mul(nr_temp, nr_temp, -0.5)
            T.tile.add(nr_temp, nr_temp, 1.5)
            T.tile.mul(inv_rms_ub, inv_rms_ub, nr_temp)
            T.tile.mul(nr_temp, inv_rms_ub, inv_rms_ub)
            T.tile.mul(nr_temp, nr_temp, sum_sq_row)
            T.tile.mul(nr_temp, nr_temp, -0.5)
            T.tile.add(nr_temp, nr_temp, 1.5)
            T.tile.mul(inv_rms_ub, inv_rms_ub, nr_temp)
            T.tile.broadcast(inv_rms_tile, inv_rms_ub)

            T.tile.fill(min_val, 1e-12)
            T.tile.max(abs_max, abs_max, min_val)
            T.tile.mul(scale_ub, abs_max, inv_rms_ub)
            T.tile.div(scale_ub, scale_ub, 127.0)

            # --- Pass 2: recompute h from x1+x2, quantize ---
            init_flags()

            if n_num > 0:
                T.wait_flag("mte3", "mte2", 0)
                T.tile.fill(x1_ub[0, :, :], 0.0)
                T.tile.fill(x2_ub[0, :, :], 0.0)
                T.tile.fill(gamma_ub[0, :], 0.0)
                T.copy(x1[row_start:row_start + ROWS, 0:block_N], x1_ub[0, :, :])
                T.copy(x2[row_start:row_start + ROWS, 0:block_N], x2_ub[0, :, :])
                T.copy(gamma[0:block_N], gamma_ub[0, :])
                T.set_flag("mte2", "v", 0)

            for by in T.serial(0, n_num - 1):
                cur = by % stages
                nxt = (by + 1) % stages
                col_off_cur = by * block_N
                col_off_nxt = (by + 1) * block_N

                T.wait_flag("mte3", "mte2", nxt)
                T.tile.fill(x1_ub[nxt, :, :], 0.0)
                T.tile.fill(x2_ub[nxt, :, :], 0.0)
                T.tile.fill(gamma_ub[nxt, :], 0.0)
                T.copy(
                    x1[row_start:row_start + ROWS, col_off_nxt:col_off_nxt + block_N],
                    x1_ub[nxt, :, :],
                )
                T.copy(
                    x2[row_start:row_start + ROWS, col_off_nxt:col_off_nxt + block_N],
                    x2_ub[nxt, :, :],
                )
                T.copy(gamma[col_off_nxt:col_off_nxt + block_N], gamma_ub[nxt, :])
                T.set_flag("mte2", "v", nxt)

                T.wait_flag("mte2", "v", cur)
                T.tile.cast(x1_fp32, x1_ub[cur, :, :], "CAST_NONE", tile_elements)
                T.tile.cast(x2_fp32, x2_ub[cur, :, :], "CAST_NONE", tile_elements)
                T.tile.add(h_fp32, x1_fp32, x2_fp32)
                T.tile.mul(hw_fp32, h_fp32, inv_rms_tile)
                T.tile.cast(gamma_fp32, gamma_ub[cur, :], "CAST_NONE", block_N)
                T.tile.broadcast(gamma_tile, gamma_fp32)
                T.tile.mul(hw_fp32, hw_fp32, gamma_tile)
                T.tile.broadcast(scale_tile, scale_ub)
                T.tile.div(hw_fp32, hw_fp32, scale_tile)
                T.tile.round(hw_fp32, hw_fp32, tile_elements)
                T.tile.clamp(hw_fp32, hw_fp32, -128.0, 127.0, tile_elements)
                T.tile.cast(out_fp16, hw_fp32, "CAST_RINT", tile_elements)
                T.tile.cast(out_int8[cur, :, :], out_fp16, "CAST_NONE", tile_elements)
                T.set_flag("v", "mte3", cur)

                T.wait_flag("v", "mte3", cur)
                T.copy(
                    out_int8[cur, :, :],
                    output[
                        row_start:row_start + ROWS,
                        col_off_cur:col_off_cur + block_N,
                    ],
                )
                T.set_flag("mte3", "mte2", cur)

            if n_num > 0:
                last = (n_num - 1) % stages
                col_off_last = (n_num - 1) * block_N
                T.wait_flag("mte2", "v", last)
                T.tile.cast(x1_fp32, x1_ub[last, :, :], "CAST_NONE", tile_elements)
                T.tile.cast(x2_fp32, x2_ub[last, :, :], "CAST_NONE", tile_elements)
                T.tile.add(h_fp32, x1_fp32, x2_fp32)
                T.tile.mul(hw_fp32, h_fp32, inv_rms_tile)
                T.tile.cast(gamma_fp32, gamma_ub[last, :], "CAST_NONE", block_N)
                T.tile.broadcast(gamma_tile, gamma_fp32)
                T.tile.mul(hw_fp32, hw_fp32, gamma_tile)
                T.tile.broadcast(scale_tile, scale_ub)
                T.tile.div(hw_fp32, hw_fp32, scale_tile)
                T.tile.round(hw_fp32, hw_fp32, tile_elements)
                T.tile.clamp(hw_fp32, hw_fp32, -128.0, 127.0, tile_elements)
                T.tile.cast(out_fp16, hw_fp32, "CAST_RINT", tile_elements)
                T.tile.cast(
                    out_int8[last, :, :], out_fp16, "CAST_NONE", tile_elements
                )
                T.set_flag("v", "mte3", last)

                T.wait_flag("v", "mte3", last)
                T.copy(
                    out_int8[last, :, :],
                    output[
                        row_start:row_start + ROWS,
                        col_off_last:col_off_last + block_N,
                    ],
                )
                T.set_flag("mte3", "mte2", last)

            drain_flags()
            T.copy(scale_ub, scale_out[row_start:row_start + ROWS])

    return main


# ============================================================================
# Single Kernel: block_M=16, alternating buffers
# Used for M < 1024 (small shapes, no dispatch overhead)
# ============================================================================


@tilelang.jit(out_idx=[-3, -2, -1], pass_configs=PASS_CONFIGS)
def _kernel_single(M, H, block_M, block_N, eps, dtype="float16"):
    VEC_NUM = 2
    ROWS = block_M // VEC_NUM
    tile_elements = ROWS * block_N
    stages = 2
    m_num = (M + block_M - 1) // block_M
    n_num = (H + block_N - 1) // block_N

    @T.prim_func
    def main(
        x1: T.Tensor((M, H), dtype),
        x2: T.Tensor((M, H), dtype),
        gamma: T.Tensor((H,), dtype),
        output: T.Tensor((M, H), "int8"),
        x_out: T.Tensor((M, H), dtype),
        scale_out: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_start = cid * block_M + vid * ROWS

            x1_ub = T.alloc_ub([stages, ROWS, block_N], dtype)
            x2_ub = T.alloc_ub([stages, ROWS, block_N], dtype)
            gamma_ub = T.alloc_ub([stages, block_N], dtype)
            x1_fp32 = T.alloc_ub([ROWS, block_N], "float32")
            x2_fp32 = T.alloc_ub([ROWS, block_N], "float32")
            h_fp32 = T.alloc_ub([ROWS, block_N], "float32")
            hw_a = T.alloc_ub([ROWS, block_N], "float32")
            hw_b = T.alloc_ub([ROWS, block_N], "float32")
            out_dtype_ub = T.alloc_ub([ROWS, block_N], dtype)
            out_fp16 = T.alloc_ub([ROWS, block_N], "float16")
            out_int8 = T.alloc_ub([stages, ROWS, block_N], "int8")
            sum_sq_acc = T.alloc_ub([ROWS, block_N], "float32")
            sum_sq_row = T.alloc_ub([ROWS, 1], "float32")
            inv_rms_ub = T.alloc_ub([ROWS, 1], "float32")
            inv_rms_tile = T.alloc_ub([ROWS, block_N], "float32")
            nr_temp = T.alloc_ub([ROWS, 1], "float32")
            gamma_fp32 = T.alloc_ub([block_N], "float32")
            gamma_tile = T.alloc_ub([ROWS, block_N], "float32")
            abs_ub = T.alloc_ub([ROWS, block_N], "float32")
            tile_max = T.alloc_ub([ROWS, 1], "float32")
            abs_max = T.alloc_ub([ROWS, 1], "float32")
            scale_ub = T.alloc_ub([ROWS, 1], "float32")
            scale_tile = T.alloc_ub([ROWS, block_N], "float32")
            min_val = T.alloc_ub([ROWS, 1], "float32")

            T.tile.fill(sum_sq_acc, 0.0)
            T.tile.fill(abs_max, 0.0)
            T.set_flag("mte3", "mte2", 0)
            T.set_flag("mte3", "mte2", 1)

            # --- Pass 1: sum_sq + abs_max(|h*gamma|) + write x_out ---
            if n_num > 0:
                T.wait_flag("mte3", "mte2", 0)
                T.tile.fill(x1_ub[0, :, :], 0.0)
                T.tile.fill(x2_ub[0, :, :], 0.0)
                T.tile.fill(gamma_ub[0, :], 0.0)
                T.copy(x1[row_start:row_start + ROWS, 0:block_N], x1_ub[0, :, :])
                T.copy(x2[row_start:row_start + ROWS, 0:block_N], x2_ub[0, :, :])
                T.copy(gamma[0:block_N], gamma_ub[0, :])
                T.set_flag("mte2", "v", 0)

            for by in T.serial(0, n_num - 1):
                cur = by % stages
                nxt = (by + 1) % stages
                col_off_cur = by * block_N
                col_off_nxt = (by + 1) * block_N

                T.wait_flag("mte3", "mte2", nxt)
                T.tile.fill(x1_ub[nxt, :, :], 0.0)
                T.tile.fill(x2_ub[nxt, :, :], 0.0)
                T.tile.fill(gamma_ub[nxt, :], 0.0)
                T.copy(
                    x1[row_start:row_start + ROWS, col_off_nxt:col_off_nxt + block_N],
                    x1_ub[nxt, :, :],
                )
                T.copy(
                    x2[row_start:row_start + ROWS, col_off_nxt:col_off_nxt + block_N],
                    x2_ub[nxt, :, :],
                )
                T.copy(gamma[col_off_nxt:col_off_nxt + block_N], gamma_ub[nxt, :])
                T.set_flag("mte2", "v", nxt)

                T.wait_flag("mte2", "v", cur)
                T.tile.cast(x1_fp32, x1_ub[cur, :, :], "CAST_NONE", tile_elements)
                T.tile.cast(x2_fp32, x2_ub[cur, :, :], "CAST_NONE", tile_elements)
                T.tile.add(h_fp32, x1_fp32, x2_fp32)
                T.tile.cast(out_dtype_ub, h_fp32, "CAST_RINT", tile_elements)
                T.tile.mul_add_dst(sum_sq_acc, h_fp32, h_fp32)
                T.tile.cast(gamma_fp32, gamma_ub[cur, :], "CAST_NONE", block_N)
                T.tile.broadcast(gamma_tile, gamma_fp32)
                T.tile.mul(hw_a, h_fp32, gamma_tile)
                T.tile.abs(hw_b, hw_a)
                T.reduce_max(hw_b, tile_max, dim=-1)
                T.tile.max(abs_max, abs_max, tile_max)
                T.set_flag("v", "mte3", cur)

                T.wait_flag("v", "mte3", cur)
                T.copy(
                    out_dtype_ub,
                    x_out[row_start:row_start + ROWS, col_off_cur:col_off_cur + block_N],
                )
                T.set_flag("mte3", "mte2", cur)

            if n_num > 0:
                last = (n_num - 1) % stages
                col_off_last = (n_num - 1) * block_N
                T.wait_flag("mte2", "v", last)
                T.tile.cast(x1_fp32, x1_ub[last, :, :], "CAST_NONE", tile_elements)
                T.tile.cast(x2_fp32, x2_ub[last, :, :], "CAST_NONE", tile_elements)
                T.tile.add(h_fp32, x1_fp32, x2_fp32)
                T.tile.cast(out_dtype_ub, h_fp32, "CAST_RINT", tile_elements)
                T.tile.mul_add_dst(sum_sq_acc, h_fp32, h_fp32)
                T.tile.cast(gamma_fp32, gamma_ub[last, :], "CAST_NONE", block_N)
                T.tile.broadcast(gamma_tile, gamma_fp32)
                T.tile.mul(hw_a, h_fp32, gamma_tile)
                T.tile.abs(hw_b, hw_a)
                T.reduce_max(hw_b, tile_max, dim=-1)
                T.tile.max(abs_max, abs_max, tile_max)
                T.set_flag("v", "mte3", last)

                T.wait_flag("v", "mte3", last)
                T.copy(
                    out_dtype_ub,
                    x_out[
                        row_start:row_start + ROWS,
                        col_off_last:col_off_last + block_N,
                    ],
                )
                T.set_flag("mte3", "mte2", last)

            T.wait_flag("mte3", "mte2", 0)
            T.wait_flag("mte3", "mte2", 1)

            # --- Reduction: inv_rms + Newton-Raphson + scale ---
            T.reduce_sum(sum_sq_acc, sum_sq_row, dim=-1)
            inv_H = T.cast(1.0 / H, "float32")
            eps_val = T.cast(eps, "float32")
            T.tile.mul(sum_sq_row, sum_sq_row, inv_H)
            T.tile.add(sum_sq_row, sum_sq_row, eps_val)
            T.tile.rsqrt(inv_rms_ub, sum_sq_row)
            T.tile.mul(nr_temp, inv_rms_ub, inv_rms_ub)
            T.tile.mul(nr_temp, nr_temp, sum_sq_row)
            T.tile.mul(nr_temp, nr_temp, -0.5)
            T.tile.add(nr_temp, nr_temp, 1.5)
            T.tile.mul(inv_rms_ub, inv_rms_ub, nr_temp)
            T.tile.mul(nr_temp, inv_rms_ub, inv_rms_ub)
            T.tile.mul(nr_temp, nr_temp, sum_sq_row)
            T.tile.mul(nr_temp, nr_temp, -0.5)
            T.tile.add(nr_temp, nr_temp, 1.5)
            T.tile.mul(inv_rms_ub, inv_rms_ub, nr_temp)
            T.tile.broadcast(inv_rms_tile, inv_rms_ub)

            T.tile.fill(min_val, 1e-12)
            T.tile.max(abs_max, abs_max, min_val)
            T.tile.mul(scale_ub, abs_max, inv_rms_ub)
            T.tile.div(scale_ub, scale_ub, 127.0)

            # --- Pass 2: recompute h from x1+x2, quantize ---
            T.set_flag("mte3", "mte2", 0)
            T.set_flag("mte3", "mte2", 1)

            if n_num > 0:
                T.wait_flag("mte3", "mte2", 0)
                T.tile.fill(x1_ub[0, :, :], 0.0)
                T.tile.fill(x2_ub[0, :, :], 0.0)
                T.tile.fill(gamma_ub[0, :], 0.0)
                T.copy(x1[row_start:row_start + ROWS, 0:block_N], x1_ub[0, :, :])
                T.copy(x2[row_start:row_start + ROWS, 0:block_N], x2_ub[0, :, :])
                T.copy(gamma[0:block_N], gamma_ub[0, :])
                T.set_flag("mte2", "v", 0)

            for by in T.serial(0, n_num - 1):
                cur = by % stages
                nxt = (by + 1) % stages
                col_off_cur = by * block_N
                col_off_nxt = (by + 1) * block_N

                T.wait_flag("mte3", "mte2", nxt)
                T.tile.fill(x1_ub[nxt, :, :], 0.0)
                T.tile.fill(x2_ub[nxt, :, :], 0.0)
                T.tile.fill(gamma_ub[nxt, :], 0.0)
                T.copy(
                    x1[row_start:row_start + ROWS, col_off_nxt:col_off_nxt + block_N],
                    x1_ub[nxt, :, :],
                )
                T.copy(
                    x2[row_start:row_start + ROWS, col_off_nxt:col_off_nxt + block_N],
                    x2_ub[nxt, :, :],
                )
                T.copy(gamma[col_off_nxt:col_off_nxt + block_N], gamma_ub[nxt, :])
                T.set_flag("mte2", "v", nxt)

                T.wait_flag("mte2", "v", cur)
                T.tile.cast(x1_fp32, x1_ub[cur, :, :], "CAST_NONE", tile_elements)
                T.tile.cast(x2_fp32, x2_ub[cur, :, :], "CAST_NONE", tile_elements)
                T.tile.add(h_fp32, x1_fp32, x2_fp32)
                T.tile.mul(hw_a, h_fp32, inv_rms_tile)
                T.tile.cast(gamma_fp32, gamma_ub[cur, :], "CAST_NONE", block_N)
                T.tile.broadcast(gamma_tile, gamma_fp32)
                T.tile.mul(hw_b, hw_a, gamma_tile)
                T.tile.broadcast(scale_tile, scale_ub)
                T.tile.div(hw_a, hw_b, scale_tile)
                T.tile.round(hw_b, hw_a, tile_elements)
                T.tile.clamp(hw_a, hw_b, -128.0, 127.0, tile_elements)
                T.tile.cast(out_fp16, hw_a, "CAST_RINT", tile_elements)
                T.tile.cast(out_int8[cur, :, :], out_fp16, "CAST_NONE", tile_elements)
                T.set_flag("v", "mte3", cur)

                T.wait_flag("v", "mte3", cur)
                T.copy(
                    out_int8[cur, :, :],
                    output[
                        row_start:row_start + ROWS,
                        col_off_cur:col_off_cur + block_N,
                    ],
                )
                T.set_flag("mte3", "mte2", cur)

            if n_num > 0:
                last = (n_num - 1) % stages
                col_off_last = (n_num - 1) * block_N
                T.wait_flag("mte2", "v", last)
                T.tile.cast(x1_fp32, x1_ub[last, :, :], "CAST_NONE", tile_elements)
                T.tile.cast(x2_fp32, x2_ub[last, :, :], "CAST_NONE", tile_elements)
                T.tile.add(h_fp32, x1_fp32, x2_fp32)
                T.tile.mul(hw_a, h_fp32, inv_rms_tile)
                T.tile.cast(gamma_fp32, gamma_ub[last, :], "CAST_NONE", block_N)
                T.tile.broadcast(gamma_tile, gamma_fp32)
                T.tile.mul(hw_b, hw_a, gamma_tile)
                T.tile.broadcast(scale_tile, scale_ub)
                T.tile.div(hw_a, hw_b, scale_tile)
                T.tile.round(hw_b, hw_a, tile_elements)
                T.tile.clamp(hw_a, hw_b, -128.0, 127.0, tile_elements)
                T.tile.cast(out_fp16, hw_a, "CAST_RINT", tile_elements)
                T.tile.cast(
                    out_int8[last, :, :], out_fp16, "CAST_NONE", tile_elements
                )
                T.set_flag("v", "mte3", last)

                T.wait_flag("v", "mte3", last)
                T.copy(
                    out_int8[last, :, :],
                    output[
                        row_start:row_start + ROWS,
                        col_off_last:col_off_last + block_N,
                    ],
                )
                T.set_flag("mte3", "mte2", last)

            T.wait_flag("mte3", "mte2", 0)
            T.wait_flag("mte3", "mte2", 1)
            T.copy(scale_ub, scale_out[row_start:row_start + ROWS])

    return main


# ============================================================================
# Tiling and Hybrid Dispatch
# ============================================================================


def _get_tiling_large(M, H):
    block_M = 32
    block_N = 256
    if H < block_N and H % 16 == 0:
        block_N = H
    max_blocks = 65535
    m_num = (M + block_M - 1) // block_M
    if m_num > max_blocks:
        block_M = ((M + max_blocks - 1) // max_blocks + 1) // 2 * 2
        if block_M < 8:
            block_M = 8
    return block_M, block_N


def _get_tiling_small(M, H):
    block_M = 16
    block_N = 256
    if H < block_N and H % 16 == 0:
        block_N = H
    max_blocks = 65535
    m_num = (M + block_M - 1) // block_M
    if m_num > max_blocks:
        block_M = ((M + max_blocks - 1) // max_blocks + 1) // 2 * 2
        if block_M < 8:
            block_M = 8
    return block_M, block_N


def _get_kernel(M, H, tl_dtype, eps, use_readback):
    if M < HYBRID_THRESHOLD:
        block_M, block_N = _get_tiling_small(M, H)
        key = ("single", M, H, tl_dtype, eps, block_M, block_N)
        if key not in _kernel_cache:
            _kernel_cache[key] = _kernel_single(M, H, block_M, block_N, eps, dtype=tl_dtype)
        return _kernel_cache[key]
    else:
        block_M, block_N = _get_tiling_large(M, H)
        key = ("dual", M, H, tl_dtype, eps, block_M, block_N, use_readback)
        if key not in _kernel_cache:
            if use_readback:
                _kernel_cache[key] = _kernel_readback(
                    M, H, block_M, block_N, eps, dtype=tl_dtype
                )
            else:
                _kernel_cache[key] = _kernel_recompute(
                    M, H, block_M, block_N, eps, dtype=tl_dtype
                )
        return _kernel_cache[key]


def add_rms_norm_dynamic_quant(x1, x2, gamma, eps=1e-6):
    """Fused Add + RMSNorm + Dynamic Quantization.

    Args:
        x1: [*, H] fp16/bf16 input tensor.
        x2: [*, H] fp16/bf16 residual tensor.
        gamma: [H] fp16/bf16 RMSNorm weight.
        eps: RMSNorm epsilon.

    Returns:
        output: [*, H] int8 quantized output.
        x_out: [*, H] fp16/bf16 residual add result.
        scale_out: [*] fp32 per-token dequantization scale.
    """
    original_dtype = x1.dtype
    original_shape = x1.shape
    H = x1.size(-1)
    M = x1.numel() // H

    x1_flat = x1.reshape(M, H).contiguous()
    x2_flat = x2.reshape(M, H).contiguous()
    gamma_flat = gamma.reshape(H).contiguous()

    kernel_dtype = _torch_dtype_to_tl(original_dtype)

    use_readback = False
    if M >= HYBRID_THRESHOLD:
        max_val = max(x1_flat.abs().max().item(), x2_flat.abs().max().item())
        use_readback = max_val < 100.0

    kernel = _get_kernel(M, H, kernel_dtype, eps, use_readback)
    output, x_out, scale_out = kernel(x1_flat, x2_flat, gamma_flat)

    output = output.reshape(original_shape)
    x_out = x_out.reshape(original_shape)

    return output, scale_out, x_out


# ============================================================================
# Golden Function (PyTorch reference implementation)
# ============================================================================


def golden_add_rms_norm_dynamic_quant(x1, x2, gamma, eps=1e-6):
    """PyTorch reference: fp32 computation throughout, no fp16 round-trip."""
    out_dtype = x1.dtype
    x1_f = x1.float()
    x2_f = x2.float()
    gamma_f = gamma.float()

    h_fp32 = x1_f + x2_f
    x_out = h_fp32.to(out_dtype)

    rms = torch.sqrt(h_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    inv_rms = 1.0 / rms
    normed = h_fp32 * inv_rms * gamma_f

    abs_max = normed.abs().amax(dim=-1, keepdim=True)
    abs_max = torch.clamp(abs_max, min=1e-12)
    scale = (abs_max / 127.0).float()
    quantized = torch.clamp(torch.round(normed / scale), -128, 127)
    output = quantized.to(torch.float16).to(torch.int8)

    scale_out = scale.squeeze(-1).reshape(-1).float()
    return output, scale_out, x_out


# ============================================================================
# L0 Threshold Tests
# ============================================================================

L0_CONFIGS = [
    (1, 4, 256),
    (2, 8, 512),
    (1, 32, 1024),
    (2, 16, 256),
]


def test_add_rms_norm_dynamic_quant_l0():
    ok = True
    for B, S, H in L0_CONFIGS:
        try:
            M = B * S
            x1 = torch.randn(B, S, H, device="npu", dtype=torch.float16)
            x2 = torch.randn(B, S, H, device="npu", dtype=torch.float16)
            gamma = torch.randn(H, device="npu", dtype=torch.float16)

            output, scale_out, x_out_k = add_rms_norm_dynamic_quant(x1, x2, gamma)
            ref_output, ref_scale, ref_x_out = golden_add_rms_norm_dynamic_quant(
                x1.cpu(), x2.cpu(), gamma.cpu()
            )

            output_3d = output.cpu().reshape(B, S, H).float()
            scale_1d = scale_out.cpu().float()
            x_out_3d = x_out_k.cpu().reshape(B, S, H).float()

            torch.testing.assert_close(output_3d, ref_output.float(), atol=1, rtol=0)
            torch.testing.assert_close(scale_1d, ref_scale.float(), atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(x_out_3d, ref_x_out.float(), atol=1e-2, rtol=1e-2)
            print(f"[PRECISION_PASS] l0 shape=({B},{S},{H}) dtype=float16")
        except Exception as e:
            print(f"[PRECISION_FAIL] l0 shape=({B},{S},{H}) dtype=float16: {e}")
            ok = False
    return ok


# ============================================================================
# Shared Test Helpers
# ============================================================================


def _run_single_test(level, B, S, H, dtype_str="float16"):
    M = B * S
    torch_dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16

    x1 = torch.randn(B, S, H, device="npu", dtype=torch_dtype)
    x2 = torch.randn(B, S, H, device="npu", dtype=torch_dtype)
    gamma = torch.randn(H, device="npu", dtype=torch_dtype)

    output, scale_out, x_out_k = add_rms_norm_dynamic_quant(x1, x2, gamma)
    ref_output, ref_scale, ref_x_out = golden_add_rms_norm_dynamic_quant(
        x1.cpu(), x2.cpu(), gamma.cpu()
    )

    output_3d = output.cpu().reshape(B, S, H).float()
    scale_1d = scale_out.cpu().float()
    x_out_3d = x_out_k.cpu().reshape(B, S, H).float()

    torch.testing.assert_close(output_3d, ref_output.float(), atol=1, rtol=0)
    torch.testing.assert_close(scale_1d, ref_scale.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(x_out_3d, ref_x_out.float(), atol=1e-2, rtol=1e-2)


def _run_precision(level, B, S, H, dtype_str="float16"):
    try:
        _run_single_test(level, B, S, H, dtype_str)
        print(f"[PRECISION_PASS] {level} shape=({B},{S},{H}) dtype={dtype_str}")
        return True
    except Exception as e:
        print(f"[PRECISION_FAIL] {level} shape=({B},{S},{H}) dtype={dtype_str}: {e}")
        return False


def _run_boundary(level, name, fn):
    try:
        fn()
        print(f"[BOUNDARY_PASS] {level} {name}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] {level} {name}: {e}")


# ============================================================================
# L1 Functional Tests
# ============================================================================

L1_CONFIGS = [
    (4, 4, 256),
    (1, 8, 512),
    (2, 4, 1024),
    (1, 5, 256),
    (1, 7, 256),
    (1, 3, 512),
    (1, 4, 128),
    (2, 8, 200),
    (1, 1024, 256),
    (2, 512, 512),
]


def test_add_rms_norm_dynamic_quant_l1():
    ok = True
    for B, S, H in L1_CONFIGS:
        ok &= _run_precision("l1", B, S, H)
    for B, S, H in [(1, 4, 256), (2, 8, 512)]:
        ok &= _run_precision("l1", B, S, H, dtype_str="bfloat16")
    return ok


# ============================================================================
# L2 Exception Tests (non-blocking)
# ============================================================================


def test_add_rms_norm_dynamic_quant_l2():

    def _test_zero_inputs():
        B, S, H = 1, 4, 256
        x1 = torch.zeros(B, S, H, device="npu", dtype=torch.float16)
        x2 = torch.zeros(B, S, H, device="npu", dtype=torch.float16)
        gamma = torch.randn(H, device="npu", dtype=torch.float16)
        output, _, _ = add_rms_norm_dynamic_quant(x1, x2, gamma)
        assert output.cpu().abs().max().item() <= 1

    def _test_large_magnitude():
        B, S, H = 1, 4, 256
        x1 = torch.randn(B, S, H, device="npu", dtype=torch.float16) * 100
        x2 = torch.randn(B, S, H, device="npu", dtype=torch.float16) * 100
        gamma = torch.ones(H, device="npu", dtype=torch.float16)
        output, _, _ = add_rms_norm_dynamic_quant(x1, x2, gamma)
        assert output.cpu().min().item() >= -128
        assert output.cpu().max().item() <= 127

    def _test_unit_gamma():
        B, S, H = 1, 4, 256
        x1 = torch.randn(B, S, H, device="npu", dtype=torch.float16)
        x2 = torch.randn(B, S, H, device="npu", dtype=torch.float16)
        gamma = torch.ones(H, device="npu", dtype=torch.float16)
        output, _, _ = add_rms_norm_dynamic_quant(x1, x2, gamma)
        M = B * S
        assert output.shape == (M, H)

    _run_boundary("l2", "zero_inputs", _test_zero_inputs)
    _run_boundary("l2", "large_magnitude", _test_large_magnitude)
    _run_boundary("l2", "unit_gamma", _test_unit_gamma)


# ============================================================================
# Boundary Tests (non-blocking)
# ============================================================================


def test_add_rms_norm_dynamic_quant_boundary():

    def _test_inf_input():
        B, S, H = 1, 4, 256
        x1 = torch.randn(B, S, H, device="npu", dtype=torch.float16)
        x1[0, 0, 0] = float("inf")
        x2 = torch.randn(B, S, H, device="npu", dtype=torch.float16)
        gamma = torch.randn(H, device="npu", dtype=torch.float16)
        add_rms_norm_dynamic_quant(x1, x2, gamma)

    def _test_nan_input():
        B, S, H = 1, 4, 256
        x1 = torch.randn(B, S, H, device="npu", dtype=torch.float16)
        x1[0, 0, 0] = float("nan")
        x2 = torch.randn(B, S, H, device="npu", dtype=torch.float16)
        gamma = torch.randn(H, device="npu", dtype=torch.float16)
        add_rms_norm_dynamic_quant(x1, x2, gamma)

    def _test_extreme_residual():
        B, S, H = 1, 4, 256
        x1 = torch.full((B, S, H), 30000.0, device="npu", dtype=torch.float16)
        x2 = torch.full((B, S, H), 30000.0, device="npu", dtype=torch.float16)
        gamma = torch.randn(H, device="npu", dtype=torch.float16)
        output, _, _ = add_rms_norm_dynamic_quant(x1, x2, gamma)
        assert output.cpu().min().item() >= -128
        assert output.cpu().max().item() <= 127

    _run_boundary("boundary", "inf_input", _test_inf_input)
    _run_boundary("boundary", "nan_input", _test_nan_input)
    _run_boundary("boundary", "extreme_residual", _test_extreme_residual)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Test add_rms_norm_dynamic_quant operator")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all"],
        help="Test level to run (default: l0)",
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True
    if args.level in ("l0", "all"):
        blocking_ok &= test_add_rms_norm_dynamic_quant_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_add_rms_norm_dynamic_quant_l1()
    if args.level in ("l2", "all"):
        test_add_rms_norm_dynamic_quant_l2()
    if args.level in ("boundary", "all"):
        test_add_rms_norm_dynamic_quant_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
