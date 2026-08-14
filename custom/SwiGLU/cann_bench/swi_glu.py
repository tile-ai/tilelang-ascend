"""SwiGLU adapter for cann-bench interface: swi_glu(input, dim=-1) -> output.

Normalizes dim, transposes middle dims to last, reshapes to 2D, dispatches to
the single-input kernel by (split_dim, dtype), and restores output rank/shape.
"""

from ._swiglu_kernel import _swiglu_kernel


_TORCH_TO_TL_DTYPE = {
    "float16": "float16",
    "float32": "float",
    "bfloat16": "bfloat16",
}

_kernel_cache = {}


# UB budget (A2/A3 = 196352 B). Worst-case byte/element by stages.
# Verified against generated Ascend C code (get_kernel_source):
#   - fp16/bf16 cast path: a0/a1/b (fp32) + tmp_in0/in1/out (dtype), all live
#     during the cast pipeline. With staged tmp (V4): 3*s*4 + 3*s*2 B/elem.
#   - fp32 direct path: tmp_in0/in1/out are dead (never read/written) and are
#     eliminated by AscendMemoryPlanning, so only a0/a1/b (fp32) count =>
#     3*s*4 B/elem. (Confirmed: fp32 kernel allocates only 3 UB buffers.)
#
# NOTE on stages=2: codegen shows the loop body is bounded by
# PipeBarrier<PIPE_ALL>() at each iteration end, so cur=i%2 buffer rotation
# gives NO cross-iteration MTE2/V overlap. stages=2 is nonetheless RETAINED
# because its higher bpe => smaller block_M => more blocks => higher core
# utilization for SMALL cases (iters=1), where parallelism beats per-block
# compute. Measured: dropping stages=2 regressed average 0.78x -> 0.73x
# (case11 1.51x -> 0.81x). So stages=2 helps via parallelism, not pipelining.
_UB_BUDGET = 196352
NUM_CORES = 48
_STAGE_OPTS = (
    # (stages, cast_bpe, direct_bpe)
    (1, 18, 12),
    (2, 36, 24),
)


def _select_tiling(k_out_cols, tl_dtype, m_out):
    """Pick (block_M, block_N, stages) by searching block_N in {128, 256} and
    stages in {1, 2}, minimizing num_iters = ceil(num_blocks / NUM_CORES).

    V4 optimizations vs V3:
      - block_N searched over {128, 256} (32B-aligned, UB-safe) instead of a
        single 128 cap. Larger block_N reduces N-direction block count for
        large-K / wide-short shapes.
      - block_M computed directly from the UB budget instead of the V3
        target_area=16384 heuristic + while-loop rewind:
            rows_per_vec * block_N * bpe <= _UB_BUDGET,  rows_per_vec = block_M // 2
            => block_M = (2 * _UB_BUDGET) // (block_N * bpe)
        This fills UB more aggressively, yielding fewer blocks / fewer iters
        for large (iters>1) cases.
      - cast_bpe for stages=2 is 36 (was 30) to account for the staged tmp
        buffers (see _swiglu_kernel.py [P0]).

    stages tie-breaker prefers 2: although stages=2 gives no pipeline overlap
      (loop-end PipeBarrier<PIPE_ALL>), its smaller block_M yields more blocks
      and thus higher core utilization for small (iters=1) cases. For large
      cases stages=1 wins on fewer iters anyway.

    Tail-block safety: TileLang lowering of T.copy with symbolic M/N auto-
    generates DataCopyNd with dynamic (row_count, col_count) clamped to the
    remaining valid elements + pad_value=0, so non-aligned shapes (M=2039,
    N=1023, k_out=1, etc.) are handled correctly without explicit masking.
    """
    dtype_bytes = 2 if tl_dtype in ("bfloat16", "float16") else 4
    align = max(1, 32 // dtype_bytes)  # fp16/bf16->16, fp32->8
    need_cast = tl_dtype in ("bfloat16", "float16")

    best = None  # (sort_key, stages, block_M, block_N)
    for bn_cap in (128, 256):
        block_N = min(k_out_cols, bn_cap)
        block_N = max(align, ((block_N + align - 1) // align) * align)
        block_N = min(block_N, bn_cap)
        if block_N <= 0:
            continue
        n_num = (k_out_cols + block_N - 1) // block_N

        for stages, cast_bpe, direct_bpe in _STAGE_OPTS:
            bpe = cast_bpe if need_cast else direct_bpe
            # Direct block_M from UB budget (factor 2 for VEC_NUM).
            block_M = (2 * _UB_BUDGET) // (block_N * bpe)
            block_M = (block_M // 32) * 32
            block_M = max(64, min(1024, block_M))
            # Safety: shrink if the 64 minimum exceeds UB budget.
            rows_per_vec = block_M // 2
            while rows_per_vec * block_N * bpe > _UB_BUDGET and block_M >= 64:
                block_M -= 32
                rows_per_vec = block_M // 2
            if block_M < 64:
                continue  # config does not fit UB, skip
            m_num = (m_out + block_M - 1) // block_M
            num_blocks = m_num * n_num
            num_iters = (num_blocks + NUM_CORES - 1) // NUM_CORES
            # Prefer: fewer iters > more stages (smaller block_M => more
            # parallel for small cases) > larger block_M.
            sort_key = (num_iters, -stages, -block_M)
            if best is None or sort_key < best[0]:
                best = (sort_key, stages, block_M, block_N)

    return best[2], best[3], best[1]


def _get_kernel(split_dim, tl_dtype, n_cols, m_out):
    """Get or compile a cached kernel for (split_dim, dtype, n_cols, m_out)."""
    k_out_cols = n_cols // 2 if split_dim == 1 else n_cols
    block_M, block_N, stages = _select_tiling(k_out_cols, tl_dtype, m_out)
    key = (split_dim, tl_dtype, block_M, block_N, stages)
    if key not in _kernel_cache:
        _kernel_cache[key] = _swiglu_kernel(block_M, block_N, stages, split_dim, dtype=tl_dtype)
    return _kernel_cache[key]


def swi_glu(input, dim=-1):
    """SwiGLU activation: output = silu(x0) * x1.

    Adapter for cann-bench interface. Reshapes to 2D, dispatches to the
    single-input kernel by (split_dim, dtype), and restores output rank/shape.

    For middle-dim split (0 < dim < ndim-1), uses a transpose-free reshape:
    since silu and mul are element-wise, we merge dims before `dim` into M and
    dims from `dim` onward into N (contiguous view, zero copy). This avoids
    permute+contiguous which triggers aclnnInplaceCopy (fails on CANN 9.0.0
    with error 561103 for some shapes).

    Args:
        input: input tensor; size of `dim` axis must be even.
        dim: split dimension (supports negative index).

    Returns:
        output tensor with same rank/dtype as input, `dim` axis halved.
    """
    ndim = input.ndim
    dim = dim % ndim
    torch_dtype_str = str(input.dtype).replace("torch.", "")

    # Validate dtype (only fp16/fp32/bf16 supported)
    if torch_dtype_str not in _TORCH_TO_TL_DTYPE:
        raise ValueError(f"SwiGLU unsupported dtype: {torch_dtype_str}. Supported: {list(_TORCH_TO_TL_DTYPE.keys())}")
    tl_dtype = _TORCH_TO_TL_DTYPE[torch_dtype_str]

    # Validate split dim is even (required for equal x0/x1 split)
    if input.shape[dim] % 2 != 0:
        raise ValueError(f"SwiGLU requires even size on dim={dim}, got {input.shape[dim]}")

    original_shape = list(input.shape)

    if dim == 0:
        # Row split: keep 2D reshape, kernel halves M.
        split_dim = 0
        M = 1
        for s in original_shape[:-1]:
            M *= s
        N = original_shape[-1]
        input_2d = input.reshape(M, N)
        out_shape = list(original_shape)
        out_shape[0] = out_shape[0] // 2
    else:
        # Col split (dim == last OR middle dim): reshape so that all dims
        # before `dim` become M and dims from `dim` onward become N. This is
        # a contiguous view (no copy) because input is row-major contiguous.
        # silu/mul are element-wise so flattening dims after `dim` into N
        # does not change the result.
        split_dim = 1
        M = 1
        for s in original_shape[:dim]:
            M *= s
        N = 1
        for s in original_shape[dim:]:
            N *= s
        input_2d = input.reshape(M, N)
        out_shape = list(original_shape)
        out_shape[dim] = out_shape[dim] // 2

    # Dispatch to kernel
    m_out = M if split_dim == 1 else M // 2
    kernel = _get_kernel(split_dim, tl_dtype, N, m_out)
    output_2d = kernel(input_2d)

    output = output_2d.reshape(out_shape)
    return output
