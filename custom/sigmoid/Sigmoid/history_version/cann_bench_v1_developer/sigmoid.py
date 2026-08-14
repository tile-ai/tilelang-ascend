"""Sigmoid adapter for cann-bench interface: sigmoid(input) -> output.

Normalizes arbitrary-rank input to 2D, dispatches to the kernel by
(dtype, M, N), and restores output rank/shape. Compiled kernels are cached
in-process so each unique shape compiles only once.
"""

import torch

from ._common import torch_dtype_to_tl
from ._sigmoid_kernel import _sigmoid_kernel


_TORCH_TO_TL_DTYPE = {
    "float16": "float16",
    "float32": "float",
    "bfloat16": "bfloat16",
}

_kernel_cache = {}

# Per-buffer byte budget. The Ascend A2/A3 UB is 196352 B, but the tilelang
# compiler stalls when a *single* buffer reaches 65536 B (64KB). Sigmoid uses
# 2 buffers (a_ub + b_ub), each of size:
#   per_buf = (block_M // VEC_NUM) * block_N * dtype_bytes
# With VEC_NUM=2: per_buf = (block_M // 2) * block_N * dtype_bytes.
# Cap per_buf at 61440 B (60KB, safely below the 64KB stall threshold):
#   (block_M // 2) * block_N * dtype_bytes <= 61440
# => block_M <= 2 * 61440 // (block_N * dtype_bytes) = 122880 // (block_N * db)
_PER_BUF_BUDGET = 61440
NUM_CORES = 24


def _select_tiling(tl_dtype, M, N):
    """Pick (block_M, block_N) minimizing num_iters = ceil(num_blocks / NUM_CORES).

    Searches block_N in {128, 256, 512} (32B-aligned, UB-safe) and computes
    block_M directly from the per-buffer UB budget:
        block_M = (2 * _PER_BUF_BUDGET) // (block_N * dtype_bytes)
    Larger block_N reduces n-direction block count for wide shapes; the search
    picks the (block_M, block_N) combo yielding fewest total iters.

    Tilelang compiler constraint: each buffer must be < 65536 B (64KB).
    Verified safe up to 61440 B (60KB) for all tested block_M/block_N combos.

    Tail-block safety: T.copy with non-symbolic M/N handles tail blocks via
    dynamic shape slicing (DataCopyNd clamps to remaining valid elements).
    """
    dtype_bytes = 2 if tl_dtype in ("bfloat16", "float16") else 4
    align = max(1, 32 // dtype_bytes)  # fp16/bf16->16, fp32->8

    best = None  # (sort_key, block_M, block_N)
    for bn_cap in (128, 256, 512):
        block_N = min(N, bn_cap)
        block_N = max(align, ((block_N + align - 1) // align) * align)
        block_N = min(block_N, bn_cap)
        if block_N <= 0:
            continue
        n_num = (N + block_N - 1) // block_N

        # Direct block_M from per-buffer UB budget.
        block_M = (2 * _PER_BUF_BUDGET) // (block_N * dtype_bytes)
        block_M = (block_M // 32) * 32
        block_M = max(64, min(1024, block_M))
        # Clamp block_M to M (non-symbolic kernel: block_M > M causes compiler
        # stall on out-of-bounds buffer allocation).
        block_M = min(block_M, M)
        # Ensure block_M >= 1 (kernel adapts VEC_NUM for block_M < 2).
        block_M = max(1, block_M)
        # Safety: shrink if per-buffer exceeds budget. VEC_NUM adapts to block_M.
        vec_num = 2 if block_M >= 2 else 1
        rows_per_vec = block_M // vec_num
        while rows_per_vec > 0 and rows_per_vec * block_N * dtype_bytes > _PER_BUF_BUDGET and block_M > 1:
            block_M -= 1
            vec_num = 2 if block_M >= 2 else 1
            rows_per_vec = block_M // vec_num
        if block_M < 1:
            continue  # config does not fit UB, skip
        m_num = (M + block_M - 1) // block_M
        num_blocks = m_num * n_num
        num_iters = (num_blocks + NUM_CORES - 1) // NUM_CORES
        # Prefer: fewer iters > larger block_M.
        sort_key = (num_iters, -block_M)
        if best is None or sort_key < best[0]:
            best = (sort_key, block_M, block_N)

    return best[1], best[2]


def _get_kernel(tl_dtype, M, N):
    """Get or compile a cached kernel for (dtype, M, N)."""
    block_M, block_N = _select_tiling(tl_dtype, M, N)
    key = (tl_dtype, M, N, block_M, block_N)
    if key not in _kernel_cache:
        _kernel_cache[key] = _sigmoid_kernel(M, N, block_M, block_N, dtype=tl_dtype)
    return _kernel_cache[key]


def sigmoid(input):
    """Sigmoid activation: y = 1 / (1 + exp(-x)).

    Adapter for cann-bench interface. Accepts any shape tensor, dispatches
    to the kernel by (dtype, M, N), and restores output rank/shape.

    Args:
        input: input tensor (float16/float32/bfloat16).

    Returns:
        output tensor with same shape/dtype as input.
    """
    torch_dtype_str = str(input.dtype).replace("torch.", "")

    # Validate dtype
    if torch_dtype_str not in _TORCH_TO_TL_DTYPE:
        raise ValueError(
            f"sigmoid unsupported dtype: {torch_dtype_str}. "
            f"Supported: {list(_TORCH_TO_TL_DTYPE.keys())}"
        )
    tl_dtype = _TORCH_TO_TL_DTYPE[torch_dtype_str]

    # Flatten arbitrary-rank input to 2D (M, N) for the kernel.
    # Sigmoid is element-wise so flattening does not change the result.
    original_shape = input.shape
    if input.ndim <= 1:
        M = 1
        N = original_shape[0] if input.ndim == 1 else 1
        input_2d = input.reshape(M, N)
    else:
        M = 1
        for s in original_shape[:-1]:
            M *= s
        N = original_shape[-1]
        input_2d = input.reshape(M, N)

    # Ensure contiguous (kernel expects row-major layout)
    if not input_2d.is_contiguous():
        input_2d = input_2d.contiguous()

    # Dispatch to kernel
    kernel = _get_kernel(tl_dtype, M, N)
    output_2d = kernel(input_2d)

    output = output_2d.reshape(original_shape)
    return output
