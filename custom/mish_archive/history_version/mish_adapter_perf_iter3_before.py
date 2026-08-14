"""Mish adapter for cann-bench interface: mish(input) -> output.

Normalizes arbitrary-rank input to 2D, dispatches to the kernel by
(dtype, M, N), and restores output rank/shape. Compiled kernels are cached
in-process so each unique shape compiles only once.

Single path (no dtype dispatch):
  mish's fp32 intermediate (ACC_DTYPE) is required for ALL dtypes — both
  fp16 (precision: 12-step accumulated error) and bfloat16 (CANN Muls/Maxs/
  Exp/Adds/Div intrinsics do not support __bf16). The kernel handles dtype
  conversion internally via T.tile.cast (UB<->UB), so the adapter passes
  tensors directly without any host-side cast.

  Expert double buffer was rejected — mish's 12-step compute + 6 fp32 buffers
  exceed UB budget under stages=2. Fixed Core mode was also rejected — large
  shapes regressed +25-36% (T.serial loop overhead > launch reduction for
  heavy 12-step compute). See custom/mish/perf_tuning/perf_report.md §2/§5.
"""

import torch

from ._common import torch_dtype_to_tl
from ._mish_kernel import _mish_kernel


_TORCH_TO_TL_DTYPE = {
    "float16": "float16",
    "float32": "float",
    "bfloat16": "bfloat16",
}

_kernel_cache = {}

# Ascend A2/A3 UB = 196352 B. mish allocates 6 UB buffers per tile (Developer
# mode, 2D T.alloc_shared):
#   5 compute buffers in fp32 (a_ub, t0_ub, t1_ub, one_ub, b_ub)
#   1 cast-bridge buffer (tmp_orig) in the original dtype
#
# For float32 input, need_cast=False so tmp_orig is allocated but never
# referenced — MEMORY_PLANNING elides it. Effective UB usage:
#   fp32:   5 buffers * 4 B = 20 B/elem   => rpv*bn <= 196352/20 = 9817
#   fp16:   5*4 + 1*2 = 22 B/elem         => rpv*bn <= 196352/22 = 8925
#   bf16:   5*4 + 1*2 = 22 B/elem         => rpv*bn <= 196352/22 = 8925
# (rpv = rows_per_vec = block_M // VEC_NUM = block_M // 2)
#
# Per-buffer < 64KB (65536 B) tilelang compiler threshold for 2D alloc_shared:
#   fp32 buffer: rpv*bn*4 < 65536 => rpv*bn < 16384  (UB total is tighter)
#   fp16 buffer: rpv*bn*2 < 65536 => rpv*bn < 32768  (UB total is tighter)
# So UB total governs. Safety margin applied below.
NUM_CORES = 24

# Effective per-(rows_per_vec * block_N) element budget, by dtype path.
# Conservative safety margin (~8%) below the theoretical UB limit to absorb
# MEMORY_PLANNING alignment padding and compiler-emitted scratch.
_UB_BUDGET_FP32 = 9000       # elems: 196352/20 = 9817, rounded down w/ margin
_UB_BUDGET_CAST = 8500       # elems: 196352/22 = 8925, rounded down w/ margin


def _select_tiling(tl_dtype, M, N):
    """Pick (block_M, block_N) minimizing num_iters = ceil(num_blocks / NUM_CORES).

    Searches block_N in {128, 256, 512} (32B-aligned, UB-safe) and computes
    block_M directly from the per-buffer UB budget:
        block_M = (2 * effective_budget) // block_N
    (factor 2 because rows_per_vec = block_M // 2 under VEC_NUM=2).

    Constraints:
    - block_N <= N (prevents out-of-bounds T.copy reads for small N)
    - block_M <= M (prevents out-of-bounds buffer allocation)
    - block_M even when >= 2 (VEC_NUM=2: vid=0 handles rows [0, bm//2),
      vid=1 handles [bm//2, bm) — odd bm skips row bm//2)
    - Each buffer < 64KB (tilelang 2D alloc_shared threshold)
    - Total UB < 196352 B (6 buffers, see module docstring)

    Tail-block safety: T.copy with non-symbolic M/N handles tail blocks via
    dynamic shape slicing (DataCopyNd clamps to remaining valid elements).
    """
    # Effective element budget (rows_per_vec * block_N) by dtype path.
    # fp32 path: 5 fp32 buffers (tmp_orig elided).
    # fp16/bf16 path: 5 fp32 + 1 orig-dtype buffer (cast bridge live).
    if tl_dtype in ("float", "float32"):
        effective_budget = _UB_BUDGET_FP32
        dtype_bytes = 4
    else:
        effective_budget = _UB_BUDGET_CAST
        dtype_bytes = 2  # fp16 / bfloat16

    align = max(1, 32 // dtype_bytes)  # fp16/bf16->16, fp32->8

    best = None  # (sort_key, block_M, block_N)
    # Cap block_N at 8192 for small M (M<=2, i.e. 1D shapes where rows_per_vec
    # = 1), allowing the UB budget to be used for wider tiles instead of being
    # capped at 512.  This reduces block count dramatically for 1D cases
    # (e.g. case 12: 1000003 elements → 123 blocks / 6 iters vs 1954 / 82).
    # For M > 2 the cap stays at 512 (2D tilings are already well-optimised).
    if M <= 2:
        max_bn = min(N, 8192)
    else:
        max_bn = min(N, 512)
    bn_caps = []
    b = 128
    while b <= max_bn:
        bn_caps.append(b)
        b *= 2
    if not bn_caps:
        bn_caps = [min(N, max(align, 128))]

    for bn_cap in bn_caps:
        block_N = min(N, bn_cap)
        # Align DOWN to 32B boundary (required for DMA alignment).
        block_N = (block_N // align) * align
        # If N < align, fall back to N (T.copy handles sub-aligned tail).
        block_N = max(min(N, align), block_N) if N < align else block_N
        block_N = min(block_N, N, bn_cap)
        if block_N <= 0:
            continue
        n_num = (N + block_N - 1) // block_N

        # Direct block_M from per-buffer UB budget.
        # rows_per_vec = block_M // 2 (VEC_NUM=2), so:
        #   rows_per_vec * block_N <= effective_budget
        #   block_M <= 2 * effective_budget // block_N
        block_M = (2 * effective_budget) // block_N
        # Round to 32-multiple for alignment, clamp to [1, M].
        block_M = (block_M // 32) * 32
        block_M = max(32, min(128, block_M))
        block_M = min(block_M, M)
        block_M = max(1, block_M)
        # Ensure block_M even when >= 2 (VEC_NUM=2 gap-avoidance).
        if block_M >= 2:
            block_M = (block_M // 2) * 2

        # Safety: shrink if per-buffer exceeds budget.
        vec_num = 2 if block_M >= 2 else 1
        rows_per_vec = block_M // vec_num
        while rows_per_vec > 0 and rows_per_vec * block_N > effective_budget and block_M > 1:
            block_M -= 1
            if block_M >= 2:
                block_M = (block_M // 2) * 2
            vec_num = 2 if block_M >= 2 else 1
            rows_per_vec = block_M // vec_num
        if block_M < 1:
            continue  # config does not fit UB, skip

        m_num = (M + block_M - 1) // block_M
        num_blocks = m_num * n_num
        num_iters = (num_blocks + NUM_CORES - 1) // NUM_CORES
        # Prefer: fewer iters > larger block_N (wider DMA → better MTE2 bw).
        sort_key = (num_iters, -block_N)
        if best is None or sort_key < best[0]:
            best = (sort_key, block_M, block_N)

    return best[1], best[2]


def _get_kernel(tl_dtype, M, N):
    """Get or compile a cached kernel for (dtype, M, N)."""
    block_M, block_N = _select_tiling(tl_dtype, M, N)
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
    torch_dtype_str = str(x.dtype).replace("torch.", "")

    # Validate dtype
    if torch_dtype_str not in _TORCH_TO_TL_DTYPE:
        raise ValueError(
            f"mish unsupported dtype: {torch_dtype_str}. "
            f"Supported: {list(_TORCH_TO_TL_DTYPE.keys())}"
        )
    tl_dtype = _TORCH_TO_TL_DTYPE[torch_dtype_str]

    # Flatten arbitrary-rank input to 2D (M, N) for the kernel.
    # Mish is element-wise so flattening does not change the result.
    # Smart reshape: prefer keeping 2D shape with M > 1 to enable VEC_NUM=2
    # (dual vector sub-core), which gives kernel speedup vs M=1.
    # For 1D tensors, reshape to near-square 2D if possible.
    # For ND tensors, merge all dims except last into M.
    original_shape = x.shape
    if x.ndim <= 1:
        # 1D: try to reshape to near-square 2D for VEC_NUM=2
        total = x.numel()
        if total <= 1:
            M, N = 1, max(total, 1)
        else:
            import math
            sqrt_n = int(math.isqrt(total))
            M = 1
            while M * 2 <= sqrt_n:
                M *= 2
            M = max(2, min(M, 8192))  # ensure M >= 2 for VEC_NUM=2
            while total % M != 0 and M > 1:
                M //= 2
            N = total // M
            if M < 2:
                M, N = 1, total
        input_2d = x.reshape(M, N)
    else:
        # ND: merge all dims except last into M
        M = 1
        for s in original_shape[:-1]:
            M *= s
        N = original_shape[-1]
        input_2d = x.reshape(M, N)

    # Ensure contiguous (kernel expects row-major layout)
    if not input_2d.is_contiguous():
        input_2d = input_2d.contiguous()

    # Dispatch to kernel (dtype cast is handled inside the kernel via
    # T.tile.cast fp16/bf16<->fp32, no host-side cast needed).
    kernel = _get_kernel(tl_dtype, M, N)
    output_2d = kernel(input_2d)

    output = output_2d.reshape(original_shape)
    return output
