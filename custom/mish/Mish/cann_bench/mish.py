"""Mish adapter for cann-bench interface: mish(input) -> output.

Normalizes arbitrary-rank input to 2D, dispatches to the kernel by
(dtype, M, N), and restores output rank/shape. Compiled kernels are cached
in-process so each unique shape compiles only once.

Host-side optimizations (Stage 3, see custom/mish/perf_tuning/perf_report.md):
  - Smart-flatten: search all split_idx to find (M, N) minimizing num_blocks.
    Uses fixed 128x128 evaluation (Vector sweet spot) to prefer M >= 128 splits.
    Critical for high-dim small-N shapes (e.g. case 13/20) where the default
    "merge all but last dim into M" blows up block count.
  - Dynamic tiling: auto-select (block_M, block_N) by dtype + UB budget.
    M >= 128 uses Vector sweet spot 128x128; M < 128 searches large block_N
    (128~8192) with small block_M for 1D/small-M shapes (e.g. case 12).
  - 32B alignment: block_N aligned to 32B (DataCopyNd requirement).

Single path (no dtype dispatch):
  mish's fp32 intermediate (ACC_DTYPE) is required for ALL dtypes. The kernel
  handles dtype conversion internally via T.tile.cast (UB<->UB).
"""

import math


from ._common import BYTES_PER_ELEM, UB_BUDGET, VEC_NUM, torch_dtype_to_tl
from ._mish_kernel import _mish_kernel


NUM_CORES = 24

_kernel_cache = {}


def _select_tiling(dtype_str, M, N):
    """Select optimal (block_M, block_N) minimizing num_blocks under UB budget.

    Strategy (verified by perf measurement):
    - M >= 128: use Vector sweet spot block_M=128, block_N=128. Non-128-multiple
      block_M hurts Vector instruction efficiency (~10-15% slower).
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
        (block_M, block_N) -- block_M is always a multiple of VEC_NUM.
    """
    bpe = BYTES_PER_ELEM[dtype_str]
    dtype_bytes = 4 if dtype_str == "float32" else 2
    align = max(1, 32 // dtype_bytes)  # fp32->8, fp16/bf16->16

    # --- M >= 128: Vector sweet spot (block_M=128, block_N=128) ---
    if M >= 128:
        block_M = 128
        if N >= 128:
            block_N = 128
        elif align <= N:
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
        # UB constraint (hard): rows_per_vec * block_N * bpe <= UB_BUDGET
        # rows_per_vec = block_M / VEC_NUM
        # => block_M <= UB_BUDGET * VEC_NUM / (block_N * bpe)
        max_block_m = (UB_BUDGET * VEC_NUM) // (bn * bpe)
        block_m = (max_block_m // VEC_NUM) * VEC_NUM
        block_m = max(VEC_NUM, min(block_m, 1024))
        if block_m > M:
            block_m = max(VEC_NUM, ((M + VEC_NUM - 1) // VEC_NUM) * VEC_NUM)

        m_num = math.ceil(M / block_m)
        n_num = math.ceil(N / bn)
        num_blocks = m_num * n_num
        sort_key = (num_blocks, -bn)
        if best is None or sort_key < best[0]:
            best = (sort_key, block_m, bn)

    return best[1], best[2]


def _smart_flatten(shape, dtype_str):
    """Search all split_idx to find (M, N) minimizing num_blocks.

    Evaluation uses fixed 128x128 tiling (Vector sweet spot) to compute
    num_blocks, which prefers splits where M >= 128 (Vector-efficient).
    Actual tiling is selected by _select_tiling at execution time, which
    may use large block_N for small M (e.g. 1D shape) -- the evaluation
    num_blocks is a proxy, not the actual value.

    For 1D shape, return (1, N) -- kernel handles M=1 with boundary protection.

    Args:
        shape: tuple of dimension sizes
        dtype_str: for tiling selection (unused in evaluation, kept for API compat)

    Returns:
        (M, N) -- 2D shape for kernel
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
        # Evaluate with fixed 128x128 (Vector sweet spot).
        m_num = math.ceil(m / 128)
        n_num = math.ceil(n / 128)
        num_blocks = m_num * n_num
        # On tie, prefer larger split_idx (closer to original last-dim logic)
        cand = (num_blocks, -split_idx, m, n)
        if best is None or cand < best:
            best = cand

    return best[2], best[3]


def _get_kernel(tl_dtype, M, N, block_M, block_N):
    """Get or compile a cached kernel for (dtype, M, N, block)."""
    key = (tl_dtype, M, N, block_M, block_N)
    if key not in _kernel_cache:
        _kernel_cache[key] = _mish_kernel(M, N, block_M, block_N, dtype=tl_dtype)
    return _kernel_cache[key]


def mish(x):
    """Mish activation: y = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x)).

    Adapter for cann-bench interface. Accepts any shape tensor, dispatches
    to the kernel by (dtype, M, N), and restores output rank/shape.

    Args:
        x: input tensor (float16/float32/bfloat16).

    Returns:
        output tensor with same shape/dtype as input.
    """
    tl_dtype = torch_dtype_to_tl(x.dtype)
    orig_shape = x.shape

    if x.ndim == 0:
        raise ValueError("Mish requires at least 1D input, got 0D scalar")

    # Smart-flatten: pick (M, N) split that minimizes num_blocks
    M, N = _smart_flatten(orig_shape, tl_dtype)

    # Ensure contiguous (kernel expects row-major layout)
    if not x.is_contiguous():
        x = x.contiguous()
    input_2d = x.reshape(M, N)

    # Dynamic tiling: auto-select block sizes
    block_M, block_N = _select_tiling(tl_dtype, M, N)

    # Dispatch to kernel (dtype cast handled inside kernel via T.tile.cast)
    kernel = _get_kernel(tl_dtype, M, N, block_M, block_N)
    output_2d = kernel(input_2d)

    output = output_2d.reshape(orig_shape)
    return output
