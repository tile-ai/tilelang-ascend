"""Mish activation: y = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x)).

Numerically stable implementation using log-sum-exp trick for softplus and
sigmoid-equivalent for tanh, with float32 intermediate computation to handle
fp16 precision loss and bf16 CANN intrinsic gaps.

Developer mode: T.alloc_shared (auto-mapped to UB) + T.tile.xxx SIMD + auto sync.

Host adapter (mish_forward) uses smart-flatten + dynamic tiling to minimize
num_blocks across diverse cann-bench shapes (1D/ND, aligned/non-aligned).
See perf_tuning/optimization_log.md for the design rationale.
"""

import math

import tilelang
import torch
from tilelang import language as T

# ========== JIT Configuration ==========
# AUTO_CV_COMBINE not set: pure Vector op (12 element-wise steps, all on AIV),
# enabling it would spawn an idle AIC core paying launch + buffer init cost.
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_ACC_DTYPE = "float32"
_VEC_NUM = 2

_TORCH_DTYPE_TO_STR = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
}

# ========== Host-side tiling selection (perf optimization) ==========
# UB budget: 196352B (Ascend A2/A3 UB size, see performance-antipatterns.md).
# Kernel allocates 5 fp32 buffers (a_ub, t0_ub, t1_ub, one_ub, b_ub) = 20B/elem
# + 1 orig buffer (tmp_orig) = dtype_bytes. MEMORY_PLANNING reuses dead buffers
# but we use worst-case for safety.
_UB_BUDGET = 196352
_BYTES_PER_ELEM = {
    "float16": 22,    # cast path: 5 fp32 (20B) + 1 fp16 orig (2B), tmp_orig live
    "bfloat16": 22,   # cast path: 5 fp32 (20B) + 1 bf16 orig (2B), tmp_orig live
    "float32": 20,    # direct path: 5 fp32 (20B), tmp_orig dead (need_cast=False),
                      # MEMORY_PLANNING reuses it. Conservative: still counts if planning disabled.
}


def _select_tiling(dtype_str, M, N):
    """Select optimal (block_M, block_N) minimizing num_blocks under UB budget.

    Strategy (verified by perf measurement):
    - M >= 128: use Vector sweet spot block_M=128, block_N=128. Non-128-multiple
      block_M (e.g. 138, 152) hurts Vector instruction efficiency (~10-15% slower).
    - M < 128 (1D or small-M after smart-flatten): search large block_N (128~8192)
      with small block_M from UB budget. Small M means Vector utilization is less
      critical (few rows), so minimizing num_blocks dominates.

    block_N is always 32B-aligned (DataCopyNd requirement):
    - fp32 (4B): align 8 elems; fp16/bf16 (2B): align 16 elems.
    Non-aligned block_N causes data corruption (verified: case 13/20 fail).

    Args:
        dtype_str: "float16", "float32", "bfloat16"
        M, N: 2D shape after smart-flatten

    Returns:
        (block_M, block_N) — block_M is always a multiple of _VEC_NUM.
    """
    bpe = _BYTES_PER_ELEM[dtype_str]
    dtype_bytes = 4 if dtype_str == "float32" else 2
    align = max(1, 32 // dtype_bytes)  # fp32→8, fp16/bf16→16

    # --- M >= 128: Vector sweet spot (block_M=128, block_N=128) ---
    if M >= 128:
        block_M = 128
        # block_N: 128 when N >= 128 (UB allows: 64*128*bpe <= 196352 for all dtypes)
        # When N < 128: use N aligned down to align (or align if N < align)
        if N >= 128:
            block_N = 128
        elif N >= align:
            block_N = (N // align) * align
            if block_N < align:
                block_N = align
        else:
            block_N = align  # T.copy handles N < align via boundary protection
        return block_M, block_N

    # --- M < 128: search large block_N (128~8192), small block_M from UB ---
    max_bn = min(N, 8192)
    candidates_bn = set()
    if N < 128:
        if N < align:
            candidates_bn.add(align)
        else:
            aligned_n = (N // align) * align
            if aligned_n >= align:
                candidates_bn.add(aligned_n)
    bn = max(align, 128)
    while bn <= max_bn:
        candidates_bn.add(bn)
        bn *= 2
    candidates_bn = sorted(candidates_bn)

    best = None  # (sort_key, block_M, block_N)
    for bn in candidates_bn:
        # UB constraint (hard): rows_per_vec * block_N * bpe <= _UB_BUDGET
        # rows_per_vec = block_M / _VEC_NUM
        # => block_M <= _UB_BUDGET * _VEC_NUM / (block_N * bpe)
        max_block_M = (_UB_BUDGET * _VEC_NUM) // (bn * bpe)
        block_M = (max_block_M // _VEC_NUM) * _VEC_NUM
        block_M = max(_VEC_NUM, min(block_M, 1024))
        # Cap block_M to M rounded up to _VEC_NUM
        if M < block_M:
            block_M = max(_VEC_NUM, ((M + _VEC_NUM - 1) // _VEC_NUM) * _VEC_NUM)

        m_num = math.ceil(M / block_M)
        n_num = math.ceil(N / bn)
        num_blocks = m_num * n_num
        # Tie-breaking: fewer num_blocks, then larger block_N (fewer n_num iters)
        sort_key = (num_blocks, -bn)
        if best is None or sort_key < best[0]:
            best = (sort_key, block_M, bn)

    return best[1], best[2]


def _smart_flatten(shape, dtype_str):
    """Search all split_idx to find (M, N) minimizing num_blocks.

    For ND shape, try splitting at each axis: M = prod(shape[:split_idx+1]),
    N = prod(shape[split_idx+1:]). Pick the split with min num_blocks.
    On tie, prefer larger split_idx (closer to original last-dim logic) to
    avoid regression on already-well-tiled shapes.

    For 1D shape, return (1, N) — kernel handles M=1 with boundary protection.

    Args:
        shape: tuple of dimension sizes
        dtype_str: for tiling selection

    Returns:
        (M, N) — 2D shape for kernel
    """
    if len(shape) <= 1:
        total = shape[0] if len(shape) == 1 else 1
        return 1, total

    total = 1
    for d in shape:
        total *= d

    best = None  # (num_blocks, -split_idx, M, N)
    M_acc = 1
    for split_idx in range(len(shape) - 1):
        M_acc *= shape[split_idx]
        M = M_acc
        N = total // M
        block_M, block_N = _select_tiling(dtype_str, M, N)
        m_num = math.ceil(M / block_M)
        n_num = math.ceil(N / block_N)
        num_blocks = m_num * n_num
        # On tie, prefer larger split_idx (closer to original last-dim logic)
        cand = (num_blocks, -split_idx, M, N)
        if best is None or cand < best:
            best = cand

    return best[2], best[3]


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def mish(M, N, block_M, block_N, dtype="float16"):
    """Mish activation kernel.

    Computes y = x * tanh(softplus(x)) via 12 T.tile.xxx steps in float32.
    Non-fp32 inputs are cast at GM<->UB boundary via T.tile.cast.

    Args:
        M: Number of rows (2D input).
        N: Number of columns (2D input).
        block_M: Row block size (recommend 128).
        block_N: Column block size (recommend 128).
        dtype: Input/output dtype string ("float16", "float32", "bfloat16").

    Returns:
        Compiled prim_func: main(A[M,N], B[M,N]) -> B (out_idx=[1]).
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    rows_per_vec = block_M // _VEC_NUM
    elem_num = rows_per_vec * block_N
    need_cast = dtype not in ("float", "float32")

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):  # type: ignore
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            # UB buffers: all float32 for intermediate computation
            a_ub = T.alloc_shared((rows_per_vec, block_N), _ACC_DTYPE)
            t0_ub = T.alloc_shared((rows_per_vec, block_N), _ACC_DTYPE)
            t1_ub = T.alloc_shared((rows_per_vec, block_N), _ACC_DTYPE)
            one_ub = T.alloc_shared((rows_per_vec, block_N), _ACC_DTYPE)
            b_ub = T.alloc_shared((rows_per_vec, block_N), _ACC_DTYPE)
            tmp_orig = T.alloc_shared((rows_per_vec, block_N), dtype)

            # --- Data load: GM -> UB (with cast for non-fp32) ---
            if need_cast:
                T.copy(A[bx * block_M + vid * rows_per_vec, by * block_N], tmp_orig)
                T.tile.cast(a_ub, tmp_orig, "CAST_NONE", elem_num)
            else:
                T.copy(A[bx * block_M + vid * rows_per_vec, by * block_N], a_ub)

            # --- Compute: y = x * tanh(softplus(x)) -- all fp32, 12 steps ---
            # Numerically stable softplus: max(x,0) + ln(1 + exp(-|x|))
            # exp argument is -|x| <= 0, result in [0,1], never overflows.
            T.tile.fill(one_ub, 1.0)  # one = 1.0
            T.tile.abs(t0_ub, a_ub)  # t0 = |x|
            T.tile.mul(t0_ub, t0_ub, -1.0)  # t0 = -|x|
            T.tile.exp(t0_ub, t0_ub)  # t0 = exp(-|x|) in [0,1]
            T.tile.add(t0_ub, t0_ub, one_ub)  # t0 = 1 + exp(-|x|)
            T.tile.ln(t0_ub, t0_ub)  # t0 = ln(1+exp(-|x|))
            T.tile.max(t1_ub, a_ub, 0.0)  # t1 = max(x, 0)
            T.tile.add(t0_ub, t0_ub, t1_ub)  # t0 = softplus

            # Numerically stable tanh: 2*sigmoid(2s) - 1
            # s = softplus >= 0, so 2s >= 0, exp(-2s) in (0,1], never overflows.
            T.tile.mul(t0_ub, t0_ub, 2.0)  # t0 = 2*softplus
            T.tile.sigmoid(t0_ub, t0_ub)  # t0 = sigmoid(2*softplus)
            T.tile.mul(t0_ub, t0_ub, 2.0)  # t0 = 2*sigmoid
            # T.tile.sub src1 does NOT accept scalar PrimExpr; use one_ub buffer
            T.tile.sub(t0_ub, t0_ub, one_ub)  # t0 = tanh = 2*sigmoid - 1

            # Final: y = x * tanh(softplus(x))
            T.tile.mul(b_ub, a_ub, t0_ub)  # b = x * tanh(softplus(x))

            # --- Data store: UB -> GM (with cast for non-fp32) ---
            if need_cast:
                T.tile.cast(tmp_orig, b_ub, "CAST_RINT", elem_num)
                T.copy(tmp_orig, B[bx * block_M + vid * rows_per_vec, by * block_N])
            else:
                T.copy(b_ub, B[bx * block_M + vid * rows_per_vec, by * block_N])

    return main


def mish_forward(x, block_M=None, block_N=None):
    """Host adapter for Mish activation with smart-flatten + dynamic tiling.

    Handles high-dimensional input by:
    1. Smart-flatten: search all split_idx to find (M, N) minimizing num_blocks
       (zero-copy reshape/view for contiguous tensors).
    2. Dynamic tiling: auto-select (block_M, block_N) based on dtype + UB budget
       when block_M/block_N are not explicitly provided.

    Args:
        x: Input tensor (1D-8D, contiguous, on NPU).
        block_M: Row block size. If None, auto-selected for min num_blocks.
        block_N: Column block size. If None, auto-selected for min num_blocks.

    Returns:
        Output tensor with same shape and dtype as input.
    """
    orig_shape = x.shape
    if x.ndim == 0:
        raise ValueError("Mish requires at least 1D input, got 0D scalar")
    dtype_str = _TORCH_DTYPE_TO_STR[x.dtype]

    # Smart-flatten: pick (M, N) split that minimizes num_blocks
    M, N = _smart_flatten(orig_shape, dtype_str)
    x_2d = x.reshape(M, N)

    # Dynamic tiling: auto-select block sizes if not specified
    if block_M is None or block_N is None:
        block_M, block_N = _select_tiling(dtype_str, M, N)

    kernel = mish(M, N, block_M, block_N, dtype=dtype_str)
    y_2d = kernel(x_2d)
    return y_2d.view(orig_shape)
