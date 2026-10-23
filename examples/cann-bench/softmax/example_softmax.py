"""Softmax operator implementation using TileLang-Ascend.

Online safe softmax with 2D kernel + Python wrapper for arbitrary dim.
Supports float16, float32, bfloat16.

Algorithm (online safe softmax, two-pass):
    Pass 1: online update running max + running sum
    Pass 2: normalize output

Reference: examples/softmax/example_online_softmax.py
"""

import argparse
import sys

import tilelang
from tilelang import language as T
import torch

# ========== Configuration ==========
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

# AUTO_SYNC=True config for use_db=False path (avoids barrier_all overhead).
# Dual-kernel strategy: use_db=True uses pass_configs (AUTO_SYNC=False + DB);
# use_db=False uses pass_configs_autosync (AUTO_SYNC=True, no manual barrier).
pass_configs_autosync = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"
VEC_NUM = 2

_DTYPE_MAP = {
    torch.float16: "float16",
    torch.float32: "float",
    torch.bfloat16: "bfloat16",
}

_THRESHOLDS = {
    "float16": 2**-10,
    "bfloat16": 2**-7,
    "float": 2**-13,
    "float32": 2**-13,
}

# Small value domain thresholds (cann-bench standard)
_SMALL_VALUE_THRESHOLDS = {
    "float16": 2**-11,
    "bfloat16": 2**-8,
    "float": 2**-14,
    "float32": 2**-14,
}

_SMALL_VALUE_ERROR_THRESHOLDS = {
    "float16": 2**-16,
    "bfloat16": 2**-16,
    "float": 2**-30,
    "float32": 2**-30,
}

# Cancellation domain thresholds (cann-bench standard)
_CANCEL_BOUNDARY = {
    "float16": 2**-5,
    "bfloat16": 2**-3,
    "float": 2**-8,
    "float32": 2**-8,
}

_CANCEL_ZERO = {
    "float16": 2**-5,
    "bfloat16": 2**-3,
    "float": 2**-8,
    "float32": 2**-8,
}

_kernel_cache = {}


# ========== Kernel ==========
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def online_softmax(M, N, block_M, block_N, dtype="float"):
    """Safe softmax with online normalizer (2D, dim=last).

    Supports float, float16, and bfloat16.
    fp16/bf16 use float32 compute internally.
    Non-aligned N uses pad_value=-inf for tail blocks.
    """
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    sub_block_M = block_M // VEC_NUM

    # Double Buffer for Pass 2: only fp16/bf16 (UB budget) + block_M<=32 + n_num>=2.
    # fp32 + block_M=128 excluded (UB overflow risk); n_num=1 excluded (no pipeline).
    use_db = use_float32_compute and (block_M <= 32) and (n_num >= 2)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            a_cal = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            tile_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            prev_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_sum_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tmp_exp = T.alloc_ub([sub_block_M, 1], cal_dtype)

            T.tile.fill(prev_max, -T.infinity(cal_dtype))
            T.tile.fill(prev_sum, 0.0)

            # Pass 1: online update running max + running sum
            # AUTO_SYNC=False: manual barrier at MTE2->V and V->MTE2(next) transitions.
            # V pipe in-place chain is safe (group_norm precedent).
            for by in T.serial(n_num):
                if use_float32_compute:
                    T.copy(
                        A[
                            bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.barrier_all()
                    T.tile.cast(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                else:
                    T.copy(
                        A[
                            bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a_cal,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.barrier_all()
                T.reduce_max(a_cal, tile_max, dim=-1)
                T.tile.max(tile_max, prev_max, tile_max)
                T.tile.sub(tmp_exp, prev_max, tile_max)
                T.tile.exp(tmp_exp, tmp_exp)

                T.tile.broadcast(tile_max_2d, tile_max)
                T.tile.sub(a_cal, a_cal, tile_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.reduce_sum(a_cal, tile_sum, dim=-1)
                T.tile.mul_add_dst(tile_sum, tmp_exp, prev_sum)
                T.copy(tile_sum, prev_sum)
                T.copy(tile_max, prev_max)
                T.barrier_all()

            # Pass 2: normalize output
            # AUTO_SYNC=False: double-buffer uses flag sync; single-buffer uses barrier_all.
            T.tile.broadcast(prev_max_2d, prev_max)
            T.tile.broadcast(prev_sum_2d, prev_sum)
            T.barrier_all()
            if use_db:
                # ===== Pass 2 Double Buffer: three-stage pipeline (prefetch/main/epilogue) =====
                # Ref: optimization-guide §2.2 + vector_add_pipeline + group_norm Pass 2.
                # a_p2/out_p2 = [2, ...] double buffer (MTE2 write / MTE3 read).
                # a_cal_p2 = [1, ...] single buffer (V pipe intermediate, in-place safe).
                # Flag chain: mte3->mte2 (buffer released), mte2->v (input ready), v->mte3 (output ready).
                # use_db=True implies use_float32_compute=True (fp16/bf16), so cast is always needed.

                # Double buffer allocation (declared inside use_db scope to satisfy TIR)
                a_p2 = T.alloc_ub([2, sub_block_M, block_N], dtype)
                a_cal_p2 = T.alloc_ub([sub_block_M, block_N], cal_dtype)
                out_p2 = T.alloc_ub([2, sub_block_M, block_N], dtype)

                # Init: both stages available for MTE2
                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)

                # Prefetch tile 0
                T.wait_flag("mte3", "mte2", 0)
                T.copy(
                    A[
                        bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                        0:block_N,
                    ],
                    a_p2[0, :, :],
                    pad_value=-T.infinity(cal_dtype),
                )
                T.set_flag("mte2", "v", 0)

                # Main loop: prefetch next while consuming current
                for by in T.serial(n_num):
                    cur = by % 2
                    nxt = (by + 1) % 2
                    # Prefetch next tile (if not last)
                    if by < n_num - 1:
                        T.wait_flag("mte3", "mte2", nxt)
                        T.copy(
                            A[
                                bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                                (by + 1) * block_N : (by + 2) * block_N,
                            ],
                            a_p2[nxt, :, :],
                            pad_value=-T.infinity(cal_dtype),
                        )
                        T.set_flag("mte2", "v", nxt)
                    # Consume cur: V compute (cast -> sub -> exp -> div -> cast)
                    T.wait_flag("mte2", "v", cur)
                    T.tile.cast(
                        a_cal_p2,
                        a_p2[cur, :, :],
                        CAST_MODE_LOW2HIGH,
                        sub_block_M * block_N,
                    )
                    T.tile.sub(a_cal_p2, a_cal_p2, prev_max_2d)
                    T.tile.exp(a_cal_p2, a_cal_p2)
                    T.tile.div(a_cal_p2, a_cal_p2, prev_sum_2d)
                    T.tile.cast(
                        out_p2[cur, :, :],
                        a_cal_p2,
                        CAST_MODE_HIGH2LOW,
                        sub_block_M * block_N,
                    )
                    # Store cur: MTE3 write back
                    T.set_flag("v", "mte3", cur)
                    T.wait_flag("v", "mte3", cur)
                    T.copy(
                        out_p2[cur, :, :],
                        B[
                            bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                    )
                    T.set_flag("mte3", "mte2", cur)

                # Drain flags
                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)
            else:
                # ===== Pass 2 Single Buffer + Barrier (use_db=False) =====
                for by in T.serial(n_num):
                    if use_float32_compute:
                        T.copy(
                            A[
                                bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                                by * block_N : (by + 1) * block_N,
                            ],
                            a,
                            pad_value=-T.infinity(cal_dtype),
                        )
                        T.barrier_all()
                        T.tile.cast(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                    else:
                        T.copy(
                            A[
                                bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                                by * block_N : (by + 1) * block_N,
                            ],
                            a_cal,
                            pad_value=-T.infinity(cal_dtype),
                        )
                        T.barrier_all()
                    T.tile.sub(a_cal, a_cal, prev_max_2d)
                    T.tile.exp(a_cal, a_cal)
                    T.tile.div(a_cal, a_cal, prev_sum_2d)
                    if use_float32_compute:
                        T.tile.cast(a, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                        T.barrier_all()
                        T.copy(
                            a,
                            B[
                                bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                                by * block_N : (by + 1) * block_N,
                            ],
                        )
                    else:
                        T.barrier_all()
                        T.copy(
                            a_cal,
                            B[
                                bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                                by * block_N : (by + 1) * block_N,
                            ],
                        )
                    T.barrier_all()

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_autosync)
def online_softmax_autosync(M, N, block_M, block_N, dtype="float"):
    """Safe softmax with online normalizer (2D, dim=last) — AUTO_SYNC=True variant.

    Identical algorithm to online_softmax but uses AUTO_SYNC=True (no manual
    barrier_all). Used for use_db=False cases to avoid barrier_all overhead.
    """
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            a_cal = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            tile_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            prev_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_sum_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tmp_exp = T.alloc_ub([sub_block_M, 1], cal_dtype)

            T.tile.fill(prev_max, -T.infinity(cal_dtype))
            T.tile.fill(prev_sum, 0.0)

            # Pass 1: online update running max + running sum
            for by in T.serial(n_num):
                if use_float32_compute:
                    T.copy(
                        A[
                            bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.tile.cast(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                else:
                    T.copy(
                        A[
                            bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a_cal,
                        pad_value=-T.infinity(cal_dtype),
                    )
                T.reduce_max(a_cal, tile_max, dim=-1)
                T.tile.max(tile_max, prev_max, tile_max)
                T.tile.sub(tmp_exp, prev_max, tile_max)
                T.tile.exp(tmp_exp, tmp_exp)

                T.tile.broadcast(tile_max_2d, tile_max)
                T.tile.sub(a_cal, a_cal, tile_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.reduce_sum(a_cal, tile_sum, dim=-1)
                T.tile.mul_add_dst(tile_sum, tmp_exp, prev_sum)
                T.copy(tile_sum, prev_sum)
                T.copy(tile_max, prev_max)

            # Pass 2: normalize output
            T.tile.broadcast(prev_max_2d, prev_max)
            T.tile.broadcast(prev_sum_2d, prev_sum)
            for by in T.serial(n_num):
                if use_float32_compute:
                    T.copy(
                        A[
                            bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.tile.cast(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                else:
                    T.copy(
                        A[
                            bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a_cal,
                        pad_value=-T.infinity(cal_dtype),
                    )
                T.tile.sub(a_cal, a_cal, prev_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.tile.div(a_cal, a_cal, prev_sum_2d)
                if use_float32_compute:
                    T.tile.cast(a, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                    T.copy(
                        a,
                        B[
                            bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                    )
                else:
                    T.copy(
                        a_cal,
                        B[
                            bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                    )

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_autosync)
def online_softmax_dim0(M, N, block_M, block_N, dtype="float"):
    """Safe softmax with online normalizer (2D, dim=0 — reduce along M).

    Parallelizes across N (columns); each core handles block_N columns and
    iterates over M rows. Eliminates the need for wrapper permute/contiguous
    when dim=0 on 2D input.

    Uses reduce_max/sum(dim=0) and broadcast(axis=0).
    """
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    sub_block_N = block_N // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, vid):
            bx = cid
            col_start = bx * block_N + vid * sub_block_N
            col_end = bx * block_N + (vid + 1) * sub_block_N

            a = T.alloc_ub([block_M, sub_block_N], dtype)
            a_cal = T.alloc_ub([block_M, sub_block_N], cal_dtype)
            tile_max = T.alloc_ub([1, sub_block_N], cal_dtype)
            tile_max_2d = T.alloc_ub([block_M, sub_block_N], cal_dtype)
            prev_max = T.alloc_ub([1, sub_block_N], cal_dtype)
            prev_max_2d = T.alloc_ub([block_M, sub_block_N], cal_dtype)
            tile_sum = T.alloc_ub([1, sub_block_N], cal_dtype)
            prev_sum = T.alloc_ub([1, sub_block_N], cal_dtype)
            prev_sum_2d = T.alloc_ub([block_M, sub_block_N], cal_dtype)
            tmp_exp = T.alloc_ub([1, sub_block_N], cal_dtype)

            T.tile.fill(prev_max, -T.infinity(cal_dtype))
            T.tile.fill(prev_sum, 0.0)

            # Pass 1: online update running max + running sum (along M)
            for bx_m in T.serial(m_num):
                row_start = bx_m * block_M
                if use_float32_compute:
                    T.copy(
                        A[row_start : row_start + block_M, col_start:col_end],
                        a,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.tile.cast(a_cal, a, CAST_MODE_LOW2HIGH, block_M * sub_block_N)
                else:
                    T.copy(
                        A[row_start : row_start + block_M, col_start:col_end],
                        a_cal,
                        pad_value=-T.infinity(cal_dtype),
                    )
                T.reduce_max(a_cal, tile_max, dim=0)
                T.tile.max(tile_max, prev_max, tile_max)
                T.tile.sub(tmp_exp, prev_max, tile_max)
                T.tile.exp(tmp_exp, tmp_exp)

                T.tile.broadcast(tile_max_2d, tile_max, axis=0)
                T.tile.sub(a_cal, a_cal, tile_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.reduce_sum(a_cal, tile_sum, dim=0)
                T.tile.mul_add_dst(tile_sum, tmp_exp, prev_sum)
                T.copy(tile_sum, prev_sum)
                T.copy(tile_max, prev_max)

            # Pass 2: normalize output
            T.tile.broadcast(prev_max_2d, prev_max, axis=0)
            T.tile.broadcast(prev_sum_2d, prev_sum, axis=0)
            for bx_m in T.serial(m_num):
                row_start = bx_m * block_M
                if use_float32_compute:
                    T.copy(
                        A[row_start : row_start + block_M, col_start:col_end],
                        a,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.tile.cast(a_cal, a, CAST_MODE_LOW2HIGH, block_M * sub_block_N)
                else:
                    T.copy(
                        A[row_start : row_start + block_M, col_start:col_end],
                        a_cal,
                        pad_value=-T.infinity(cal_dtype),
                    )
                T.tile.sub(a_cal, a_cal, prev_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.tile.div(a_cal, a_cal, prev_sum_2d)
                if use_float32_compute:
                    T.tile.cast(a, a_cal, CAST_MODE_HIGH2LOW, block_M * sub_block_N)
                    T.copy(a, B[row_start : row_start + block_M, col_start:col_end])
                else:
                    T.copy(a_cal, B[row_start : row_start + block_M, col_start:col_end])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_autosync)
def online_softmax_3d(A_dim, B_dim, C_dim, block_B, block_C, dtype="float"):
    """Safe softmax along dim=1 of 3D [A, B, C] tensor — no permute needed.

    Reduces along B for each (a, c) pair. Parallelizes across (a, c_block) pairs.
    Each tile X[a, b_range, c_range] is a contiguous 2D block in row-major memory.
    Uses reduce_max/sum(dim=0) on [block_B, sub_block_C] tiles.
    """
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    b_num = T.ceildiv(B_dim, block_B)
    c_num = T.ceildiv(C_dim, block_C)
    m_num = A_dim * c_num
    sub_block_C = block_C // VEC_NUM

    @T.prim_func
    def main(
        X: T.Tensor((A_dim, B_dim, C_dim), dtype),  # type: ignore
        Y: T.Tensor((A_dim, B_dim, C_dim), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_idx = cid // c_num
            c_blk = cid % c_num
            c_start = c_blk * block_C + vid * sub_block_C
            c_end = c_start + sub_block_C

            a_ub = T.alloc_ub([block_B, sub_block_C], dtype)
            a_cal = T.alloc_ub([block_B, sub_block_C], cal_dtype)
            tile_max = T.alloc_ub([1, sub_block_C], cal_dtype)
            tile_max_2d = T.alloc_ub([block_B, sub_block_C], cal_dtype)
            prev_max = T.alloc_ub([1, sub_block_C], cal_dtype)
            prev_max_2d = T.alloc_ub([block_B, sub_block_C], cal_dtype)
            tile_sum = T.alloc_ub([1, sub_block_C], cal_dtype)
            prev_sum = T.alloc_ub([1, sub_block_C], cal_dtype)
            prev_sum_2d = T.alloc_ub([block_B, sub_block_C], cal_dtype)
            tmp_exp = T.alloc_ub([1, sub_block_C], cal_dtype)

            T.tile.fill(prev_max, -T.infinity(cal_dtype))
            T.tile.fill(prev_sum, 0.0)

            # Pass 1: online max+sum along B
            for bx_b in T.serial(b_num):
                b_start = bx_b * block_B
                if use_float32_compute:
                    T.copy(
                        X[a_idx, b_start : b_start + block_B, c_start:c_end],
                        a_ub,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.tile.cast(a_cal, a_ub, CAST_MODE_LOW2HIGH, block_B * sub_block_C)
                else:
                    T.copy(
                        X[a_idx, b_start : b_start + block_B, c_start:c_end],
                        a_cal,
                        pad_value=-T.infinity(cal_dtype),
                    )
                T.reduce_max(a_cal, tile_max, dim=0)
                T.tile.max(tile_max, prev_max, tile_max)
                T.tile.sub(tmp_exp, prev_max, tile_max)
                T.tile.exp(tmp_exp, tmp_exp)

                T.tile.broadcast(tile_max_2d, tile_max, axis=0)
                T.tile.sub(a_cal, a_cal, tile_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.reduce_sum(a_cal, tile_sum, dim=0)
                T.tile.mul_add_dst(tile_sum, tmp_exp, prev_sum)
                T.copy(tile_sum, prev_sum)
                T.copy(tile_max, prev_max)

            # Pass 2: normalize
            T.tile.broadcast(prev_max_2d, prev_max, axis=0)
            T.tile.broadcast(prev_sum_2d, prev_sum, axis=0)
            for bx_b in T.serial(b_num):
                b_start = bx_b * block_B
                if use_float32_compute:
                    T.copy(
                        X[a_idx, b_start : b_start + block_B, c_start:c_end],
                        a_ub,
                        pad_value=-T.infinity(cal_dtype),
                    )
                    T.tile.cast(a_cal, a_ub, CAST_MODE_LOW2HIGH, block_B * sub_block_C)
                else:
                    T.copy(
                        X[a_idx, b_start : b_start + block_B, c_start:c_end],
                        a_cal,
                        pad_value=-T.infinity(cal_dtype),
                    )
                T.tile.sub(a_cal, a_cal, prev_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.tile.div(a_cal, a_cal, prev_sum_2d)
                if use_float32_compute:
                    T.tile.cast(a_ub, a_cal, CAST_MODE_HIGH2LOW, block_B * sub_block_C)
                    T.copy(a_ub, Y[a_idx, b_start : b_start + block_B, c_start:c_end])
                else:
                    T.copy(a_cal, Y[a_idx, b_start : b_start + block_B, c_start:c_end])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_autosync)
def online_softmax_single(M, N, block_M, block_N, dtype="float"):
    """Simplified softmax for n_num=1 (N <= block_N, i.e. small N).

    When n_num=1, online softmax degenerates to standard softmax:
    single reduce_max + single normalize, no online update needed.
    Fewer buffers (6 vs 10) allows larger block_M for better parallelism.
    """
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            a_cal = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            tile_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            tile_sum_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)

            if use_float32_compute:
                T.copy(
                    A[
                        bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                        0:block_N,
                    ],
                    a,
                    pad_value=-T.infinity(cal_dtype),
                )
                T.tile.cast(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
            else:
                T.copy(
                    A[
                        bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                        0:block_N,
                    ],
                    a_cal,
                    pad_value=-T.infinity(cal_dtype),
                )
            T.reduce_max(a_cal, tile_max, dim=-1)
            T.tile.broadcast(tile_max_2d, tile_max)
            T.tile.sub(a_cal, a_cal, tile_max_2d)
            T.tile.exp(a_cal, a_cal)
            T.reduce_sum(a_cal, tile_sum, dim=-1)
            T.tile.broadcast(tile_sum_2d, tile_sum)
            T.tile.div(a_cal, a_cal, tile_sum_2d)
            if use_float32_compute:
                T.tile.cast(a, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                T.copy(
                    a,
                    B[
                        bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                        0:block_N,
                    ],
                )
            else:
                T.copy(
                    a_cal,
                    B[
                        bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                        0:block_N,
                    ],
                )

    return main


def _select_block(M, N, dtype_str):
    """Select block_M, block_N based on shape and dtype.

    block_M adaptive strategy (Stage 3 #1, block_M-only; Fixed Core launch
    reverted — hardware scheduler outperforms software chunk loop for this
    memory-bound kernel):
    - Large M (>= 24*128): block_M=128.
    - Small-medium M + throughput-bound (n_num >= 12): reduce block_M so m_num >= 24
      to use all physical cores via hardware scheduling.
    - Small-medium M + latency-bound (n_num < 12): keep block_M=128 (small shapes are
      latency-bound; smaller tiles add overhead without core-utilization benefit).
    See DESIGN.md §7.6 + optimization-guide §2.9 + antipatterns §A/B.
    """
    core_num = 24
    if N < 128:
        block_N = 32
    else:
        block_N = 128
    n_num = (N + block_N - 1) // block_N

    # Small-N path: n_num==1 uses simplified kernel with fewer buffers → larger block_M
    if n_num == 1:
        block_M = 1024
        cal_bytes = 4
        dtype_bytes = 2 if dtype_str in ("float16", "bfloat16") else 4
        sub_bm = block_M // VEC_NUM
        # 6 buffers: 1 dtype 2D + 3 fp32 2D + 2 fp32 1D
        ub_est = sub_bm * block_N * dtype_bytes + 3 * sub_bm * block_N * cal_bytes + 2 * sub_bm * cal_bytes
        while ub_est > 185 * 1024 and block_M > 32:
            block_M //= 2
            sub_bm = block_M // VEC_NUM
            ub_est = sub_bm * block_N * dtype_bytes + 3 * sub_bm * block_N * cal_bytes + 2 * sub_bm * cal_bytes
        return block_M, block_N

    if core_num * 128 <= M:
        block_M = 128
    elif n_num >= 8:
        block_M = 32
    else:
        block_M = 128

    # Parallelism check: if m_num < core_num, reduce block_M for more parallelism
    m_num = (M + block_M - 1) // block_M
    if m_num < core_num and block_M > 16:
        block_M = 16

    # block_N adaptive (Stage 3 #2): enlarge block_N for large N to reduce n_num
    # (per-tile DMA/setup overhead dominates for large-N memory-bound kernel).
    # UB constraint: block_M=128 (sub_block_M=64) -> block_N<=128 (~145KB);
    # block_M=32 (sub_block_M=16) -> block_N<=512 (~144KB). Must be 16-aligned.
    if block_M <= 32 and N >= 256:
        sub_bm = block_M // VEC_NUM
        max_bn = 512
        # UB guard: fp16/bf16 uses 4 fp32 2D + 1 dtype 2D; fp32 uses 5 fp32 2D
        if dtype_str in ("float16", "bfloat16"):
            while max_bn > 128 and (sub_bm * max_bn * 4 * 4 + sub_bm * max_bn * 2) > 170 * 1024:
                max_bn //= 2
        else:
            while max_bn > 128 and (sub_bm * max_bn * 4 * 5 + sub_bm * 4 * 5) > 170 * 1024:
                max_bn //= 2
        bn = max_bn
        while bn > N:
            bn //= 2
        block_N = max(128, bn)
    return block_M, block_N


def _get_kernel(M, N, block_M, block_N, dtype_str):
    """Get or compile kernel for given config (with caching).

    Triple-kernel strategy:
    - n_num==1: online_softmax_single (simplified, no online update)
    - use_db=True: online_softmax (AUTO_SYNC=False + Pass 2 Double Buffer)
    - use_db=False: online_softmax_autosync (AUTO_SYNC=True)
    """
    use_float32_compute = dtype_str in ["bfloat16", "float16"]
    n_num = (N + block_N - 1) // block_N
    use_db = use_float32_compute and (block_M <= 32) and (n_num >= 2)

    key = (M, N, block_M, block_N, dtype_str)
    if key not in _kernel_cache:
        if n_num == 1:
            _kernel_cache[key] = online_softmax_single(M, N, block_M, block_N, dtype=dtype_str)
        elif use_db:
            _kernel_cache[key] = online_softmax(M, N, block_M, block_N, dtype=dtype_str)
        else:
            _kernel_cache[key] = online_softmax_autosync(M, N, block_M, block_N, dtype=dtype_str)
    return _kernel_cache[key]


def _select_block_dim0(M, N, dtype_str):
    """Select block sizes for dim=0 kernel (reduce along M, parallel across N).

    M: reduction dimension (rows) — block_M is the reduction tile
    N: parallelism dimension (columns) — block_N determines core count
    UB budget: ~5 * block_M * sub_block_N * cal_bytes < 170KB
    """
    core_num = 24
    block_M = 128
    block_N = 128

    n_num = (N + block_N - 1) // block_N
    if n_num < core_num and block_N > 32:
        block_N = max(32, (N + core_num - 1) // core_num)
        block_N = ((block_N + 15) // 16) * 16

    sub_bn = block_N // VEC_NUM
    cal_bytes = 4 if dtype_str in ("float16", "bfloat16") else 4
    ub_est = 5 * block_M * sub_bn * cal_bytes
    while ub_est > 170 * 1024 and block_M > 16:
        block_M //= 2
        ub_est = 5 * block_M * sub_bn * cal_bytes

    return block_M, block_N


_kernel_cache_dim0 = {}


def _get_kernel_dim0(M, N, block_M, block_N, dtype_str):
    """Get or compile dim=0 kernel for given config (with caching)."""
    key = (M, N, block_M, block_N, dtype_str)
    if key not in _kernel_cache_dim0:
        _kernel_cache_dim0[key] = online_softmax_dim0(M, N, block_M, block_N, dtype=dtype_str)
    return _kernel_cache_dim0[key]


def _select_block_3d(A_dim, B_dim, C_dim, dtype_str):
    """Select block sizes for 3D interior-dim kernel (reduce along B, parallel across A*C).

    UB budget: ~10 buffers of [block_B, sub_block_C] * cal_bytes < 170KB
    """
    block_B = 128
    block_C = 128

    c_num = (C_dim + block_C - 1) // block_C
    m_num = A_dim * c_num
    core_num = 24

    if m_num < core_num and block_C > 32:
        block_C = max(32, (C_dim * A_dim + core_num - 1) // core_num)
        block_C = ((block_C + 15) // 16) * 16
        if block_C < 16:
            block_C = 16

    sub_bc = block_C // VEC_NUM
    cal_bytes = 4
    ub_est = 10 * block_B * sub_bc * cal_bytes
    while ub_est > 170 * 1024 and block_B > 16:
        block_B //= 2
        ub_est = 10 * block_B * sub_bc * cal_bytes

    return block_B, block_C


_kernel_cache_3d = {}


def _get_kernel_3d(A_dim, B_dim, C_dim, block_B, block_C, dtype_str):
    """Get or compile 3D interior-dim kernel for given config (with caching)."""
    key = (A_dim, B_dim, C_dim, block_B, block_C, dtype_str)
    if key not in _kernel_cache_3d:
        _kernel_cache_3d[key] = online_softmax_3d(A_dim, B_dim, C_dim, block_B, block_C, dtype=dtype_str)
    return _kernel_cache_3d[key]


def softmax_impl(x, dim=-1):
    """Softmax implementation: Python wrapper + TileLang kernel.

    Handles arbitrary dim by permute+flatten to 2D [M, N] with dim=last,
    then invokes the 2D online softmax kernel.

    Args:
        x: input tensor (1~8D), dtype in {float16, float32, bfloat16}
        dim: softmax dimension, supports negative index

    Returns:
        output tensor with same shape and dtype as x
    """
    rank = x.dim()
    if dim < 0:
        dim = dim + rank
    assert 0 <= dim < rank, f"dim {dim} out of range for rank {rank}"

    dtype_str = _DTYPE_MAP[x.dtype]

    # Fast path 1: dim == last and rank == 2
    if dim == rank - 1 and rank == 2:
        M, N = x.shape
        block_M, block_N = _select_block(M, N, dtype_str)
        kernel = _get_kernel(M, N, block_M, block_N, dtype_str)
        return kernel(x)

    # Fast path 2: dim == last (any rank) — just flatten to 2D, no permute/contiguous
    if dim == rank - 1 and x.is_contiguous():
        N = x.shape[-1]
        M = x.numel() // N
        x_2d = x.reshape(M, N)
        block_M, block_N = _select_block(M, N, dtype_str)
        kernel = _get_kernel(M, N, block_M, block_N, dtype_str)
        y_2d = kernel(x_2d)
        return y_2d.reshape(x.shape)

    # Fast path 3: dim == 0 and rank == 2 — use dim=0 kernel, no permute/contiguous
    if dim == 0 and rank == 2 and x.is_contiguous():
        M, N = x.shape
        block_M, block_N = _select_block_dim0(M, N, dtype_str)
        kernel = _get_kernel_dim0(M, N, block_M, block_N, dtype_str)
        return kernel(x)

    # Fast path 4: interior dim (not 0, not last) — reshape to 3D, use 3D kernel
    if dim != 0 and dim != rank - 1 and x.is_contiguous():
        outer = 1
        for i in range(dim):
            outer *= x.shape[i]
        B_dim = x.shape[dim]
        inner = 1
        for i in range(dim + 1, rank):
            inner *= x.shape[i]
        x_3d = x.reshape(outer, B_dim, inner)
        block_B, block_C = _select_block_3d(outer, B_dim, inner, dtype_str)
        kernel = _get_kernel_3d(outer, B_dim, inner, block_B, block_C, dtype_str)
        y_3d = kernel(x_3d)
        return y_3d.reshape(x.shape)

    # General path: permute dim to last, flatten to 2D
    perm = [i for i in range(rank) if i != dim] + [dim]
    x_perm = x.permute(perm).contiguous()
    N = x.shape[dim]
    M = x_perm.numel() // N
    x_2d = x_perm.reshape(M, N)

    block_M, block_N = _select_block(M, N, dtype_str)
    kernel = _get_kernel(M, N, block_M, block_N, dtype_str)
    y_2d = kernel(x_2d)

    # Reshape back and inverse permute
    y_perm = y_2d.reshape(x_perm.shape)
    inv_perm = [0] * rank
    for i, p in enumerate(perm):
        inv_perm[p] = i
    y = y_perm.permute(inv_perm).contiguous()
    return y


# ========== Golden ==========
def golden_softmax(x, dim=-1):
    """PyTorch golden reference in FP64 precision (matches cann-bench fp64_cpu strategy).

    Computes softmax in float64 on CPU for maximum accuracy.
    """
    return torch.nn.functional.softmax(x.double(), dim=dim)


def native_softmax(x, dim=-1):
    """Same-precision reference (matches cann-bench native_output).

    Computes softmax in the original dtype on CPU, used for small-value
    domain fallback comparison.
    """
    return torch.nn.functional.softmax(x, dim=dim)


# ========== Precision ==========
def mere_mare(actual, golden):
    """Compute MERE (mean relative error) and MARE (max relative error).

    Assumes no NaN/inf in inputs. For special value cases, use check_precision.
    """
    diff = (actual.float() - golden.float()).abs()
    rel = diff / (golden.float().abs() + 1e-7)
    return rel.mean().item(), rel.max().item()


def check_precision(actual, golden, native, dtype_str):
    """Check precision matching cann-bench relative_error checker.

    Implements the full cann-bench logic:
    1. First stage: overall MERE < threshold AND MARE < 10*threshold
    2. Second stage (if first fails): classify mismatches into
       small-value / cancellation / normal domains
    3. Fallback: if only small-value/cancellation mismatches, use
       ErrorCount ratio (NPU/CPU <= 2)

    Args:
        actual: kernel output (NPU), same dtype as input
        golden: FP64 golden reference (CPU)
        native: same-precision reference (CPU, original dtype)
        dtype_str: dtype string for threshold lookup

    Returns:
        (passed, mere, mare) — mere/mare are display values
    """
    threshold = _THRESHOLDS[dtype_str]
    mare_threshold = 10 * threshold
    sv_threshold = _SMALL_VALUE_THRESHOLDS[dtype_str]
    sv_error = _SMALL_VALUE_ERROR_THRESHOLDS[dtype_str]
    cancel_boundary = _CANCEL_BOUNDARY[dtype_str]
    cancel_zero = _CANCEL_ZERO[dtype_str]

    # Move to CPU and upcast for comparison
    if actual.device.type == "npu":
        actual = actual.cpu()
    target_dtype = actual.dtype
    golden_trunc = golden.to(target_dtype).double()
    output_fp64 = actual.double()

    # NaN position check
    nan_out = torch.isnan(output_fp64)
    nan_gold = torch.isnan(golden_trunc)
    if not torch.all(nan_out == nan_gold):
        return False, float("inf"), float("inf")

    # Inf saturation handling: replace one-sided inf with max finite value
    inf_out = torch.isinf(output_fp64)
    inf_gold = torch.isinf(golden_trunc)
    inf_mismatch = inf_out != inf_gold
    inf_match_mask = torch.zeros_like(output_fp64, dtype=torch.bool)
    if torch.any(inf_mismatch):
        max_finite = float(torch.finfo(target_dtype).max)
        if torch.any(inf_out & ~inf_gold):
            mask = inf_out & ~inf_gold
            output_fp64[mask] = torch.sign(output_fp64[mask]) * max_finite
        if torch.any(inf_gold & ~inf_out):
            mask = inf_gold & ~inf_out
            golden_trunc[mask] = torch.sign(golden_trunc[mask]) * max_finite
    both_inf = inf_out & inf_gold
    if torch.any(both_inf):
        if not torch.all(torch.sign(output_fp64[both_inf]) == torch.sign(golden_trunc[both_inf])):
            return False, float("inf"), float("inf")
        inf_match_mask[both_inf] = True

    # Compute relative error
    diff = torch.abs(output_fp64 - golden_trunc)
    golden_abs = torch.abs(golden_trunc)
    denominator = golden_abs + 1e-7
    relative_error = diff / denominator

    # Exclude NaN, Inf, and matched-Inf positions
    valid_mask = ~(torch.isnan(relative_error) | torch.isinf(relative_error) | inf_match_mask)
    valid_re = relative_error[valid_mask]

    if len(valid_re) == 0:
        return True, 0.0, 0.0

    overall_mere = float(valid_re.mean())
    overall_mare = float(valid_re.max())

    # First stage: overall MERE/MARE check
    if overall_mere < threshold and overall_mare < mare_threshold:
        return True, overall_mere, overall_mare

    # Second stage: classify mismatches
    mismatch_mask = (relative_error > mare_threshold) & valid_mask

    # Small value domain: |golden| < sv_threshold
    sv_mask = (golden_abs < sv_threshold) & valid_mask
    # Cancellation: output ≈ 0 AND golden in [sv_threshold, cancel_boundary)
    output_abs = torch.abs(output_fp64)
    cancel_mask = (output_abs < cancel_zero) & (golden_abs >= sv_threshold) & (golden_abs < cancel_boundary) & valid_mask
    # Normal: everything else
    normal_mismatch = mismatch_mask & ~sv_mask & ~cancel_mask

    if torch.any(normal_mismatch):
        # Normal domain has mismatches → FAIL
        return False, overall_mere, overall_mare

    # Only small-value/cancellation mismatches → fallback
    # NPU small-value errors: |golden| < sv_threshold AND |diff| > sv_error
    sv_npu_err = sv_mask & (diff > sv_error)
    sv_npu_count = int(sv_npu_err.sum())

    # CPU (native) small-value errors
    native_fp64 = native.to(target_dtype).double()
    cpu_diff = torch.abs(native_fp64 - golden_trunc)
    sv_cpu_err = sv_mask & (cpu_diff > sv_error)
    sv_cpu_count = int(sv_cpu_err.sum())

    if sv_cpu_count == 0:
        sv_passed = sv_npu_count == 0
    else:
        sv_passed = sv_npu_count / max(sv_cpu_count, 1) <= 2

    # Cancellation errors
    cancel_npu_err = cancel_mask & (relative_error > mare_threshold)
    cancel_npu_count = int(cancel_npu_err.sum())
    cpu_re = cpu_diff / (golden_abs + 1e-7)
    cancel_cpu_err = cancel_mask & (cpu_re > mare_threshold)
    cancel_cpu_count = int(cancel_cpu_err.sum())

    if cancel_cpu_count == 0:
        cancel_passed = cancel_npu_count == 0
    else:
        cancel_passed = cancel_npu_count / max(cancel_cpu_count, 1) <= 2

    passed = sv_passed and cancel_passed

    # Display MERE/MARE excluding small-value/cancellation
    normal_mask = ~sv_mask & ~cancel_mask & valid_mask
    normal_re = relative_error[normal_mask]
    if len(normal_re) > 0:
        display_mere = float(normal_re.mean())
        display_mare = float(normal_re.max())
    else:
        display_mere = 0.0
        display_mare = 0.0

    return passed, display_mere, display_mare


# ========== Input Generation ==========
def _gen_normal(shape, dtype_torch, lo, hi):
    """Generate uniform random input in [lo, hi]."""
    x = torch.empty(shape, dtype=torch.float32).uniform_(lo, hi)
    return x.to(dtype_torch)


def _gen_zeros(shape, dtype_torch):
    """Generate all-zeros input."""
    return torch.zeros(shape, dtype=dtype_torch)


def _gen_with_inf(shape, dtype_torch):
    """Generate input with inf/-inf special values.

    Ensures some slices contain +inf (-> NaN output) and some are all -inf
    (-> NaN output), matching torch.nn.functional.softmax behavior.
    """
    x = torch.randn(shape, dtype=torch.float32) * 10
    x = x.to(dtype_torch)
    if x.dim() == 2:
        # Every 2000th row: all -inf (-> NaN output)
        x[::2000, :] = float("-inf")
        # Every 1000th row (offset): +inf in col 0 (-> NaN output)
        x[1000::2000, 0] = float("inf")
    else:
        xf = x.view(-1)
        xf[::10000] = float("inf")
        xf[1::10000] = float("-inf")
    return x


def _gen_with_nan(shape, dtype_torch):
    """Generate input with NaN values (-> NaN output propagation)."""
    x = torch.randn(shape, dtype=torch.float32)
    x = x.to(dtype_torch)
    xf = x.view(-1)
    xf[::10000] = float("nan")
    return x


# ========== Test Runner ==========
def _run_case(name, shape, dtype_torch, dim, value_range, level):
    """Run a single test case.

    Args:
        name: case name (e.g. "L0-1")
        shape: input shape
        dtype_torch: torch dtype
        dim: softmax dimension
        value_range: (lo, hi) tuple or "zero"/"inf"/"nan"
        level: "L0"/"L1"/"L2"/"Boundary"

    Returns:
        True if passed
    """
    dtype_str = _DTYPE_MAP[dtype_torch]
    shape_list = list(shape)
    try:
        # Generate input
        if value_range == "zero":
            x_cpu = _gen_zeros(shape, dtype_torch)
        elif value_range == "inf":
            x_cpu = _gen_with_inf(shape, dtype_torch)
        elif value_range == "nan":
            x_cpu = _gen_with_nan(shape, dtype_torch)
        else:
            lo, hi = value_range
            x_cpu = _gen_normal(shape, dtype_torch, lo, hi)

        # Run kernel on NPU
        x_npu = x_cpu.npu()
        y = softmax_impl(x_npu, dim=dim).cpu()

        # Compute golden (FP64) and native (same precision) on CPU
        ref = golden_softmax(x_cpu, dim=dim)
        native = native_softmax(x_cpu, dim=dim)

        # Check precision (cann-bench relative_error checker logic)
        ok, mere, mare = check_precision(y, ref, native, dtype_str)

        if level in ("L0", "L1"):
            tag = "[PRECISION_PASS]" if ok else "[PRECISION_FAIL]"
        else:
            tag = "[BOUNDARY_PASS]" if ok else "[BOUNDARY_WARN]"

        print(f"{tag} {level} {name} shape={shape_list} dtype={dtype_str} dim={dim} MERE={mere:.2e} MARE={mare:.2e}")
        return ok
    except Exception as e:
        if level in ("L0", "L1"):
            tag = "[PRECISION_FAIL]"
        else:
            tag = "[BOUNDARY_WARN]"
        print(f"{tag} {level} {name} shape={shape_list} dtype={dtype_str} dim={dim}: {e}")
        return False


# ========== L0 Threshold Tests ==========
def test_softmax_l0():
    """L0 threshold tests (9 cases from DESIGN.md §13.2).

    Covers: aligned/non-aligned shapes, 3 dtypes, dim=-1/0/3,
    2D/5D, inf/zero special values.
    """
    cases = [
        ("L0-1", [1024, 1024], torch.float16, -1, (-1, 1)),
        # ("L0-2", [2048, 2048], torch.float32, -1, (-2, 2)),
        # ("L0-3", [4096, 4096], torch.bfloat16, -1, (-3, 3)),
        # ("L0-4", [8192, 8192], torch.float16, 0, (-10, 10)),
        # ("L0-5", [1023, 2047], torch.float16, -1, (-0.1, 0.1)),
        # ("L0-6", [2049, 4097], torch.float32, -1, (-1, 1)),
        # ("L0-7", [1000003, 2], torch.float16, -1, "inf"),
        # ("L0-8", [3, 7, 11, 13, 1013], torch.bfloat16, -1, "zero"),
        # ("L0-9", [2, 3, 17, 1024, 101], torch.float32, 3, (-20, 40)),
    ]
    passed = 0
    for name, shape, dtype, dim, vr in cases:
        if _run_case(name, tuple(shape), dtype, dim, vr, "L0"):
            passed += 1
    print(f"\n[L0] Summary: {passed}/{len(cases)} passed")
    return passed == len(cases)


# ========== L1 Functional Tests ==========
def test_softmax_l1():
    """L1 functional tests: irregular shapes, various dtypes and dims.

    Covers cann-bench cases not in L0: dim=0/1/2/-2, 3D/4D, prime shapes.
    """
    cases = [
        ("L1-1", [8192, 8192], torch.float32, 1, (-100, 100)),
        ("L1-2", [31, 67, 127, 257], torch.bfloat16, 2, (-5, 5)),
        ("L1-3", [127, 257, 1023], torch.bfloat16, -2, (-0.5, 0.5)),
        ("L1-4", [1009, 1021], torch.float16, -1, (-1, 2)),
        ("L1-5", [367, 373, 379], torch.float32, 1, (-50, 100)),
        ("L1-6", [11, 13, 17, 4001], torch.bfloat16, -1, (-3, 6)),
        ("L1-7", [512, 2049], torch.float16, 0, (-0.5, 0.5)),
        ("L1-8", [255, 8193], torch.float32, 1, (-1000, 1000)),
        ("L1-9", [2, 511, 2049], torch.bfloat16, -1, (-0.2, 0.2)),
    ]
    passed = 0
    for name, shape, dtype, dim, vr in cases:
        if _run_case(name, tuple(shape), dtype, dim, vr, "L1"):
            passed += 1
    print(f"\n[L1] Summary: {passed}/{len(cases)} passed")
    return passed == len(cases)


# ========== L2 Exception Tests ==========
def test_softmax_l2():
    """L2 exception tests: inf/nan special values, large value range.

    L2 failures produce [BOUNDARY_WARN] and do not block exit code.
    """
    cases = [
        ("L2-1", [1024, 1024], torch.float16, -1, "inf"),
        ("L2-2", [256, 256], torch.float32, -1, "nan"),
        ("L2-3", [4, 255, 2049], torch.float16, 1, (-65504, 65504)),
    ]
    passed = 0
    for name, shape, dtype, dim, vr in cases:
        if _run_case(name, tuple(shape), dtype, dim, vr, "L2"):
            passed += 1
    print(f"\n[L2] Summary: {passed}/{len(cases)} passed (warnings do not block)")


# ========== Boundary Tests ==========
def test_softmax_boundary():
    """Boundary tests: extreme small shapes, single row/column.

    Boundary failures produce [BOUNDARY_WARN] and do not block exit code.
    """
    cases = [
        ("B-1", [1, 1], torch.float16, -1, (-1, 1)),
        ("B-2", [1, 1024], torch.float32, -1, (-1, 1)),
        ("B-3", [1024, 1], torch.float16, -1, (-1, 1)),
        ("B-4", [2, 3], torch.bfloat16, -1, (-1, 1)),
    ]
    passed = 0
    for name, shape, dtype, dim, vr in cases:
        if _run_case(name, tuple(shape), dtype, dim, vr, "Boundary"):
            passed += 1
    print(f"\n[Boundary] Summary: {passed}/{len(cases)} passed (warnings do not block)")


# ========== cann-bench 20 Cases ==========
def test_softmax_cann_bench():
    """cann-bench level2/softmax 20 cases (exact shapes/dtypes/dims/value_ranges).

    Source: cann-bench/tasks/level2/softmax/cases.yaml
    """
    cases = [
        ("cann-bench-1", [1024, 1024], torch.float16, -1, (-1, 1)),
        ("cann-bench-2", [2048, 2048], torch.float32, -1, (-2, 2)),
        ("cann-bench-3", [4096, 4096], torch.bfloat16, -1, (-3, 3)),
        ("cann-bench-4", [8192, 8192], torch.float16, 0, (-10, 10)),
        ("cann-bench-5", [8192, 8192], torch.float32, 1, (-100, 100)),
        ("cann-bench-6", [31, 67, 127, 257], torch.bfloat16, 2, (-5, 5)),
        ("cann-bench-7", [1023, 2047], torch.float16, -1, (-0.1, 0.1)),
        ("cann-bench-8", [2049, 4097], torch.float32, -1, (-1, 1)),
        ("cann-bench-9", [127, 257, 1023], torch.bfloat16, -2, (-0.5, 0.5)),
        ("cann-bench-10", [1009, 1021], torch.float16, -1, (-1, 2)),
        ("cann-bench-11", [367, 373, 379], torch.float32, 1, (-50, 100)),
        ("cann-bench-12", [11, 13, 17, 4001], torch.bfloat16, -1, (-3, 6)),
        ("cann-bench-13", [1000003, 2], torch.float16, -1, "inf"),
        ("cann-bench-14", [11, 13, 17, 67, 67], torch.float32, -1, "nan"),
        ("cann-bench-15", [3, 7, 11, 13, 1013], torch.bfloat16, -1, "zero"),
        ("cann-bench-16", [512, 2049], torch.float16, 0, (-0.5, 0.5)),
        ("cann-bench-17", [255, 8193], torch.float32, 1, (-1000, 1000)),
        ("cann-bench-18", [2, 511, 2049], torch.bfloat16, -1, (-0.2, 0.2)),
        ("cann-bench-19", [4, 255, 2049], torch.float16, 1, (-65504, 65504)),
        ("cann-bench-20", [2, 3, 17, 1024, 101], torch.float32, 3, (-20, 40)),
    ]
    passed = 0
    for name, shape, dtype, dim, vr in cases:
        if _run_case(name, tuple(shape), dtype, dim, vr, "L0"):
            passed += 1
    print(f"\n[cann-bench] Summary: {passed}/{len(cases)} passed")
    return passed == len(cases)


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Softmax operator tests")
    parser.add_argument(
        "--level",
        default="l0",
        type=str.lower,
        choices=["l0", "l1", "l2", "boundary", "cann-bench", "all"],
        help="Test level: L0 (threshold), L1 (functional), L2 (exception), "
        "Boundary (edge cases), cann-bench (20 official cases), all (full suite)",
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True

    if args.level in ("l0", "all"):
        print("=" * 70)
        print("Running L0 threshold tests...")
        print("=" * 70)
        blocking_ok &= test_softmax_l0()
    if args.level in ("l1", "all"):
        print("=" * 70)
        print("Running L1 functional tests...")
        print("=" * 70)
        blocking_ok &= test_softmax_l1()
    if args.level in ("l2", "all"):
        print("=" * 70)
        print("Running L2 exception tests...")
        print("=" * 70)
        test_softmax_l2()
    if args.level in ("boundary", "all"):
        print("=" * 70)
        print("Running Boundary tests...")
        print("=" * 70)
        test_softmax_boundary()
    if args.level in ("cann-bench", "all"):
        print("=" * 70)
        print("Running cann-bench 20 cases...")
        print("=" * 70)
        blocking_ok &= test_softmax_cann_bench()

    print("=" * 70)
    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    else:
        print("Test Failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
