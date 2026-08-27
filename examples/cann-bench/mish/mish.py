"""Mish activation kernel: y = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x)).

Numerically stable element-wise activation using log-sum-exp trick for softplus
and sigmoid-equivalent for tanh, with float32 intermediate computation to handle
fp16 precision loss and bf16 CANN intrinsic gaps.

Developer mode, dynamic tiling (block_M/block_N by dtype + UB budget) +
smart-flatten (minimize num_blocks for high-dim shapes).
"""

import math

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

# ========== JIT Configuration ==========
# AUTO_CV_COMBINE not set: pure Vector op (12 element-wise steps, all on AIV),
# enabling it would spawn an idle AIC core paying launch + buffer init cost.
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_ACC_DTYPE = "float32"
_VEC_NUM = 2

_TORCH_TO_TL_DTYPE = {
    "float16": "float16",
    "float32": "float32",
    "bfloat16": "bfloat16",
}

# UB budget: Ascend A2/A3 UB = 196352B. Kernel allocates 5 fp32 compute buffers
# + 1 orig-dtype cast-bridge buffer. See _select_tiling for budget calculation.
_UB_BUDGET = 196352
_BYTES_PER_ELEM = {
    "float16": 22,  # cast path: 5 fp32 (20B) + 1 fp16 orig (2B)
    "bfloat16": 22,  # cast path: 5 fp32 (20B) + 1 bf16 orig (2B)
    "float32": 20,  # direct path: 5 fp32 (20B), tmp_orig elided
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def mish(M, N, block_M, block_N, dtype="float16"):
    """Mish kernel: y = x * tanh(softplus(x)).

    Developer mode + fp32 intermediate + cast bridge. Single path for all dtypes.

    Args:
        M, N: 2D tensor shape (rows, cols).
        block_M, block_N: tile size per block. block_M must be a multiple of
            _VEC_NUM (2).
        dtype: "float16" / "float32" / "bfloat16".

    Returns:
        prim_func mapping A (M, N) -> B (M, N), same dtype.
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
            # Stable softplus: max(x,0) + ln(1 + exp(-|x|))  -- exp arg <= 0
            T.tile.fill(one_ub, 1.0)
            T.tile.abs(t0_ub, a_ub)
            T.tile.mul(t0_ub, t0_ub, -1.0)
            T.tile.exp(t0_ub, t0_ub)
            T.tile.add(t0_ub, t0_ub, one_ub)
            T.tile.ln(t0_ub, t0_ub)
            T.tile.max(t1_ub, a_ub, 0.0)
            T.tile.add(t0_ub, t0_ub, t1_ub)

            # Stable tanh: 2*sigmoid(2s) - 1  -- s=softplus >= 0
            T.tile.mul(t0_ub, t0_ub, 2.0)
            T.tile.sigmoid(t0_ub, t0_ub)
            T.tile.mul(t0_ub, t0_ub, 2.0)
            # T.tile.sub src1 does NOT accept scalar; use one_ub buffer
            T.tile.sub(t0_ub, t0_ub, one_ub)

            # Final: y = x * tanh(softplus(x))
            T.tile.mul(b_ub, a_ub, t0_ub)

            # --- Data store: UB -> GM (with cast for non-fp32) ---
            if need_cast:
                T.tile.cast(tmp_orig, b_ub, "CAST_RINT", elem_num)
                T.copy(tmp_orig, B[bx * block_M + vid * rows_per_vec, by * block_N])
            else:
                T.copy(b_ub, B[bx * block_M + vid * rows_per_vec, by * block_N])

    return main


# ========== Host-side adapter (cann-bench interface: mish(input)) ==========

_kernel_cache = {}
NUM_CORES = 24


def _select_tiling(dtype_str, M, N):
    """Select optimal (block_M, block_N) minimizing num_blocks under UB budget.

    - M >= 128: Vector sweet spot block_M=128, block_N=128.
    - M < 128: search large block_N (128~8192) with small block_M from UB budget.
    block_N is always 32B-aligned (DataCopyNd requirement).
    """
    bpe = _BYTES_PER_ELEM[dtype_str]
    dtype_bytes = 4 if dtype_str == "float32" else 2
    align = max(1, 32 // dtype_bytes)  # fp32->8, fp16/bf16->16

    if M >= 128:
        block_M = 128
        if N >= 128:
            block_N = 128
        elif align <= N:
            block_N = (N // align) * align
            if block_N < align:
                block_N = align
        else:
            block_N = align
        return block_M, block_N

    max_bn = min(N, 8192)
    candidates_bn = set()
    if N < 128:
        if align > N:
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
        max_block_m = (_UB_BUDGET * _VEC_NUM) // (bn * bpe)
        block_m = (max_block_m // _VEC_NUM) * _VEC_NUM
        block_m = max(_VEC_NUM, min(block_m, 1024))
        if block_m > M:
            block_m = max(_VEC_NUM, ((M + _VEC_NUM - 1) // _VEC_NUM) * _VEC_NUM)
        m_num = math.ceil(M / block_m)
        n_num = math.ceil(N / bn)
        num_blocks = m_num * n_num
        sort_key = (num_blocks, -bn)
        if best is None or sort_key < best[0]:
            best = (sort_key, block_m, bn)

    return best[1], best[2]


def _smart_flatten(shape):
    """Search all split_idx to find (M, N) minimizing num_blocks.

    Uses fixed 128x128 evaluation (Vector sweet spot) to prefer M >= 128 splits.
    For 1D shape, return (1, N).
    """
    if len(shape) <= 1:
        total = shape[0] if len(shape) == 1 else 1
        return 1, total

    total = 1
    for d in shape:
        total *= d

    best = None  # (num_blocks_128x128, -split_idx, M, N)
    m_acc = 1
    for split_idx in range(len(shape) - 1):
        m_acc *= shape[split_idx]
        m = m_acc
        n = total // m
        m_num = math.ceil(m / 128)
        n_num = math.ceil(n / 128)
        num_blocks = m_num * n_num
        cand = (num_blocks, -split_idx, m, n)
        if best is None or cand < best:
            best = cand

    return best[2], best[3]


def _get_kernel(tl_dtype, M, N, block_M, block_N):
    """Get or compile a cached kernel for (dtype, M, N, block)."""
    key = (tl_dtype, M, N, block_M, block_N)
    if key not in _kernel_cache:
        _kernel_cache[key] = mish(M, N, block_M, block_N, dtype=tl_dtype)
    return _kernel_cache[key]


def mish_forward(x):
    """Mish activation: y = x * tanh(softplus(x)).

    Adapter for cann-bench interface. Accepts any shape tensor, smart-flattens
    to 2D, dispatches to the kernel by (dtype, M, N), and restores output shape.

    Args:
        x: input tensor (float16/float32/bfloat16).

    Returns:
        output tensor with same shape/dtype as input.
    """
    torch_dtype_str = str(x.dtype).replace("torch.", "")
    if torch_dtype_str not in _TORCH_TO_TL_DTYPE:
        raise ValueError(f"mish unsupported dtype: {torch_dtype_str}. Supported: {list(_TORCH_TO_TL_DTYPE.keys())}")
    tl_dtype = _TORCH_TO_TL_DTYPE[torch_dtype_str]
    orig_shape = x.shape

    if x.ndim == 0:
        raise ValueError("Mish requires at least 1D input, got 0D scalar")

    M, N = _smart_flatten(orig_shape)
    if not x.is_contiguous():
        x = x.contiguous()
    input_2d = x.reshape(M, N)

    block_M, block_N = _select_tiling(tl_dtype, M, N)
    kernel = _get_kernel(tl_dtype, M, N, block_M, block_N)
    output_2d = kernel(input_2d)

    output = output_2d.reshape(orig_shape)
    return output
