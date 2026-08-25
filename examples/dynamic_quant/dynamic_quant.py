"""Dynamic quantization: per-token symmetric quantization to int8.

Optimizations:
  B2: abs_buf eliminated (in-place T.tile.abs)
  #6: scale_2d broadcast eliminated (implicit broadcast via T.Parallel)
  #3: Fixed Core mode (core_num = min(m_num, 24))
  #5: Adaptive block_M + block_N dispatch:
      - Tiny N (≤128): block_M=64, block_N=128 → minimize GM padding waste
      - Small N (128,512]: block_M=64, block_N=512 → fewer serial blocks for large M
      - Large N (>512): block_M=32, block_N=1024 → fewer N-chunks for large N

Three separate JIT functions to avoid mixed buffer size segfault.
"""

import tilelang
from tilelang import language as T

DTYPE_MAX = 127.0
VEC_NUM = 2
NUM_CORES = 24
CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# ====================================================================
# Tiny N kernel: block_M=128, block_N=128
# sub_block_M=64, UB ≈ 73KB (38%)
# For N ≤ 128: minimizes GM padding waste + reduces M-serial bottleneck
# block_M=128 halves m_num vs block_M=64, critical for large M cases (e.g. Case 14: M=163K)
# ====================================================================
@tilelang.jit(out_idx=[1, 2], pass_configs=pass_configs)
def _dynamic_quant_tiny(M, N, block_M, block_N, core_num, dtype="float16"):
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
                    # === Pass 1: Max Pass ===
                    T.tile.fill(row_max, 0.0)
                    for n_chunk in T.serial(n_num):
                        T.copy(
                            x[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                              n_chunk * block_N : (n_chunk + 1) * block_N],
                            a_ub, pad_value=0.0)
                        T.tile.cast(a_cal, a_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        T.tile.abs(a_cal, a_cal)
                        T.reduce_max(a_cal, chunk_max, dim=-1)
                        T.tile.max(row_max, row_max, chunk_max)
                    T.tile.max(row_max, row_max, 1e-12)
                    T.tile.div(scale, row_max, DTYPE_MAX)
                    # === Pass 2: Quantize Pass ===
                    for n_chunk in T.serial(n_num):
                        T.copy(
                            x[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                              n_chunk * block_N : (n_chunk + 1) * block_N],
                            a_ub, pad_value=0.0)
                        T.tile.cast(a_cal, a_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        for i, j in T.Parallel(sub_block_M, block_N):
                            a_cal[i, j] = a_cal[i, j] / scale[i, 0]
                        T.tile.round(a_cal, a_cal, sub_block_M * block_N)
                        a_fp16 = T.alloc_ub([sub_block_M, block_N], "float16")
                        T.tile.cast(a_fp16, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                        T.tile.cast(y_int8, a_fp16, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        T.copy(y_int8,
                            y_out[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                                  n_chunk * block_N : (n_chunk + 1) * block_N])
                    T.copy(scale,
                        scale_out[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M])
    return main


# ====================================================================
# Small N kernel: block_M=64, block_N=512
# sub_block_M=32, UB ≈ 144KB (75%)
# For 128 < N ≤ 512: larger block_M → fewer M-blocks
# ====================================================================
@tilelang.jit(out_idx=[1, 2], pass_configs=pass_configs)
def _dynamic_quant_small(M, N, block_M, block_N, core_num, dtype="float16"):
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
                    # === Pass 1: Max Pass ===
                    T.tile.fill(row_max, 0.0)
                    for n_chunk in T.serial(n_num):
                        T.copy(
                            x[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                              n_chunk * block_N : (n_chunk + 1) * block_N],
                            a_ub, pad_value=0.0)
                        T.tile.cast(a_cal, a_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        T.tile.abs(a_cal, a_cal)
                        T.reduce_max(a_cal, chunk_max, dim=-1)
                        T.tile.max(row_max, row_max, chunk_max)
                    T.tile.max(row_max, row_max, 1e-12)
                    T.tile.div(scale, row_max, DTYPE_MAX)
                    # === Pass 2: Quantize Pass ===
                    for n_chunk in T.serial(n_num):
                        T.copy(
                            x[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                              n_chunk * block_N : (n_chunk + 1) * block_N],
                            a_ub, pad_value=0.0)
                        T.tile.cast(a_cal, a_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        for i, j in T.Parallel(sub_block_M, block_N):
                            a_cal[i, j] = a_cal[i, j] / scale[i, 0]
                        T.tile.round(a_cal, a_cal, sub_block_M * block_N)
                        a_fp16 = T.alloc_ub([sub_block_M, block_N], "float16")
                        T.tile.cast(a_fp16, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                        T.tile.cast(y_int8, a_fp16, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        T.copy(y_int8,
                            y_out[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                                  n_chunk * block_N : (n_chunk + 1) * block_N])
                    T.copy(scale,
                        scale_out[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M])
    return main


# ====================================================================
# Large N kernel: block_M=16, block_N=1024
# sub_block_M=8, UB ≈ 96KB (50%)
# For N > 512: larger block_N → fewer N-chunks → less per-block overhead
# Reduced block_M from 32 to 16 to fit block_N=1024 in UB budget
# ====================================================================
@tilelang.jit(out_idx=[1, 2], pass_configs=pass_configs)
def _dynamic_quant_large(M, N, block_M, block_N, core_num, dtype="float16"):
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
                    # === Pass 1: Max Pass ===
                    T.tile.fill(row_max, 0.0)
                    for n_chunk in T.serial(n_num):
                        T.copy(
                            x[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                              n_chunk * block_N : (n_chunk + 1) * block_N],
                            a_ub, pad_value=0.0)
                        T.tile.cast(a_cal, a_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        T.tile.abs(a_cal, a_cal)
                        T.reduce_max(a_cal, chunk_max, dim=-1)
                        T.tile.max(row_max, row_max, chunk_max)
                    T.tile.max(row_max, row_max, 1e-12)
                    T.tile.div(scale, row_max, DTYPE_MAX)
                    # === Pass 2: Quantize Pass ===
                    for n_chunk in T.serial(n_num):
                        T.copy(
                            x[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                              n_chunk * block_N : (n_chunk + 1) * block_N],
                            a_ub, pad_value=0.0)
                        T.tile.cast(a_cal, a_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        for i, j in T.Parallel(sub_block_M, block_N):
                            a_cal[i, j] = a_cal[i, j] / scale[i, 0]
                        T.tile.round(a_cal, a_cal, sub_block_M * block_N)
                        a_fp16 = T.alloc_ub([sub_block_M, block_N], "float16")
                        T.tile.cast(a_fp16, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                        T.tile.cast(y_int8, a_fp16, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                        T.copy(y_int8,
                            y_out[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                                  n_chunk * block_N : (n_chunk + 1) * block_N])
                    T.copy(scale,
                        scale_out[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M])
    return main


# ====================================================================
# Public API: dispatch based on N
# ====================================================================
BLOCK_M_TINY = 128  # Increased from 64 to reduce M-serial bottleneck for large M
BLOCK_N_TINY = 128
N_THRESHOLD_TINY = 128

BLOCK_M_SMALL = 64
BLOCK_N_SMALL = 512
N_THRESHOLD_SMALL = 512

BLOCK_M_LARGE = 16   # Reduced from 32 to fit block_N=1024 in UB budget
BLOCK_N_LARGE = 1024 # Increased from 768 to reduce N-chunks overhead


def dynamic_quant(M, N, block_M, block_N, core_num, dtype="float16", has_smooth_scales=False):
    """Dispatch to tiny/small/large-N kernel based on N.
    
    Parallel strategy:
    - m_num ≤ 24: Fixed Core (core_num = m_num)
    - m_num > 24: Full parallel (core_num = m_num) to eliminate serial bottleneck
    """
    if N <= N_THRESHOLD_TINY:
        m_num = (M + BLOCK_M_TINY - 1) // BLOCK_M_TINY
        cn = m_num  # Full parallel when m_num > 24
        return _dynamic_quant_tiny(M, N, BLOCK_M_TINY, BLOCK_N_TINY, cn, dtype=dtype)
    elif N <= N_THRESHOLD_SMALL:
        m_num = (M + BLOCK_M_SMALL - 1) // BLOCK_M_SMALL
        cn = m_num  # Full parallel when m_num > 24
        return _dynamic_quant_small(M, N, BLOCK_M_SMALL, BLOCK_N_SMALL, cn, dtype=dtype)
    else:
        m_num = (M + BLOCK_M_LARGE - 1) // BLOCK_M_LARGE
        cn = m_num  # Full parallel when m_num > 24
        return _dynamic_quant_large(M, N, BLOCK_M_LARGE, BLOCK_N_LARGE, cn, dtype=dtype)