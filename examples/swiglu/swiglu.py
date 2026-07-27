"""SwiGLU activation kernel: output = silu(x0) * x1.

Single-input kernel with offset access to x0/x1 halves (no host chunk).
Developer mode, persistent + double buffering, T.tile.silu, fp16/bf16 upcast to fp32.
Dynamic tiling (block_M/block_N by output cols) + per-case stages selection.
"""

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def swiglu(block_M, block_N, stages, split_dim, dtype="float16"):
    """SwiGLU kernel: single input A(M,N), output B(M//m_div, N//n_div).

    Args:
        block_M: row block size (UB budget checked by caller; no internal clamp).
        block_N: col block size (must be 32B-aligned for DataCopyNd).
        stages: double-buffer stages (1 = single buffer, 2 = double buffer).
            Caller picks per-case via _select_tiling (stages=1 enlarges block_M
            for large cases reducing iters; stages=2 keeps double-buffer for
            small cases where iters don't decrease).
        split_dim: 1 = split last dim (n_div=2); 0/-2 = split row axis (m_div=2).
        dtype: "float16" / "float" / "float32" / "bfloat16".
    """
    M = T.symbolic("M")
    N = T.symbolic("N")

    need_cast = dtype not in ("float", "float32")
    ACC_DTYPE = "float32"

    # NOTE: use ternary (single assignment) instead of if-block reassignment.
    # TileLang prim_func internal Python if-block variable reassignment does not
    # propagate to codegen (compiler limitation). Ternary expressions are single
    # assignments and propagate correctly.
    row_split = split_dim == 0 or split_dim == -2
    m_div = 2 if row_split else 1
    n_div = 1 if row_split else 2

    m_num = T.ceildiv(M // m_div, block_M)
    n_num = T.ceildiv(N // n_div, block_N)
    num_blocks = m_num * n_num

    VEC_NUM = 2
    rows_per_vec = block_M // VEC_NUM
    elem_num = rows_per_vec * block_N

    NUM_CORES = 48
    num_iters = T.ceildiv(num_blocks, NUM_CORES)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M // m_div, N // n_div), dtype),  # type: ignore
    ):
        # Offset via ternary (see note above on if-block limitation).
        m_offset = M // 2 if row_split else 0
        n_offset = 0 if row_split else N // 2

        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            # UB buffers (double-buffered stages dim)
            a0_ub = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)
            a1_ub = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)
            b_ub = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)

            # dtype transit buffers (only allocated; used when need_cast=True)
            tmp_in0 = T.alloc_ub((rows_per_vec, block_N), dtype)
            tmp_in1 = T.alloc_ub((rows_per_vec, block_N), dtype)
            tmp_out = T.alloc_ub((rows_per_vec, block_N), dtype)

            for i in T.serial(num_iters):
                cur = i % stages

                block_id = cid + i * NUM_CORES
                if block_id < num_blocks:
                    bx = block_id // n_num
                    by = block_id % n_num
                    row = bx * block_M + vid * rows_per_vec
                    col = by * block_N
                    row2 = row + m_offset
                    col2 = col + n_offset

                    # Load + upcast to fp32
                    if need_cast:
                        T.copy(A[row, col], tmp_in0)
                        T.copy(A[row2, col2], tmp_in1)
                        T.tile.cast(a0_ub[cur, :, :], tmp_in0, "CAST_NONE", elem_num)
                        T.tile.cast(a1_ub[cur, :, :], tmp_in1, "CAST_NONE", elem_num)
                    else:
                        T.copy(A[row, col], a0_ub[cur, :, :])
                        T.copy(A[row2, col2], a1_ub[cur, :, :])

                    # SiLU(x0) then gating mul: out = silu(x0) * x1
                    T.tile.silu(b_ub[cur, :, :], a0_ub[cur, :, :])
                    T.tile.mul(b_ub[cur, :, :], b_ub[cur, :, :], a1_ub[cur, :, :])

                    # Downcast back to original dtype + store
                    out_row = bx * block_M + vid * rows_per_vec
                    out_col = by * block_N

                    if need_cast:
                        T.tile.cast(tmp_out, b_ub[cur, :, :], "CAST_RINT", elem_num)
                        T.copy(tmp_out, B[out_row, out_col])
                    else:
                        T.copy(b_ub[cur, :, :], B[out_row, out_col])

    return main


# ========== Host-side adapter (interface: swi_glu(input, dim)) ==========

_TORCH_TO_TL_DTYPE = {
    "float16": "float16",
    "float32": "float",
    "bfloat16": "bfloat16",
}

_kernel_cache = {}

# UB budget (A2/A3 = 196352 B). Worst-case byte/element by stages:
#   stages=1 (single buffer):
#     cast path (fp16/bf16: a0/a1/b fp32 x1 + tmp_in0/in1/out): 3*1*4 + 3*2 = 18
#     direct path (fp32: a0/a1/b fp32 x1):                       3*1*4      = 12
#   stages=2 (double buffer):
#     cast path: 3*2*4 + 3*2 = 30
#     direct path: 3*2*4     = 24
_UB_BUDGET = 196352
NUM_CORES = 48
_STAGE_OPTS = (
    # (stages, cast_bpe, direct_bpe)
    (1, 18, 12),
    (2, 30, 24),
)


def _select_tiling(k_out_cols, tl_dtype, m_out):
    """Pick (block_M, block_N, stages) by output cols, dtype, and output rows.

    block_N: aligned to 32B (DataCopyNd granularity), capped at 128.
    stages: chosen per-case by minimizing num_iters = ceil(num_blocks / NUM_CORES).
      - stages=1 enlarges block_M (lower UB/elem) -> fewer blocks -> fewer iters
        (wins for large cases where iters drop materially).
      - stages=2 keeps double-buffer; when iters are equal to stages=1, prefer
        stages=2 (double-buffer hides GM->UB latency for small/medium cases).
      Ties (equal num_iters) go to stages=2.
    """
    dtype_bytes = 2 if tl_dtype in ("bfloat16", "float16") else 4
    align = max(1, 32 // dtype_bytes)  # fp16/bf16->16, fp32->8

    block_N = min(k_out_cols, 128)
    block_N = max(align, ((block_N + align - 1) // align) * align)
    block_N = min(block_N, 128)

    need_cast = tl_dtype in ("bfloat16", "float16")
    n_num = (k_out_cols + block_N - 1) // block_N

    target_area = 16384
    best = None  # (num_iters, stages, block_M)
    for stages, cast_bpe, direct_bpe in _STAGE_OPTS:
        bpe = cast_bpe if need_cast else direct_bpe
        block_M = target_area // block_N
        block_M = (block_M // 32) * 32
        block_M = max(64, min(1024, block_M))
        rows_per_vec = block_M // 2
        while rows_per_vec * block_N * bpe > _UB_BUDGET and block_M > 64:
            block_M -= 32
            rows_per_vec = block_M // 2
        m_num = (m_out + block_M - 1) // block_M
        num_blocks = m_num * n_num
        num_iters = (num_blocks + NUM_CORES - 1) // NUM_CORES
        # Prefer fewer iters; tie -> larger stages (double-buffer)
        if best is None or num_iters < best[0] or (num_iters == best[0] and stages > best[1]):
            best = (num_iters, stages, block_M)

    return best[2], block_N, best[1]


def _get_kernel(split_dim, tl_dtype, n_cols, m_out):
    """Get or compile a cached kernel for (split_dim, dtype, n_cols, m_out).

    block_M/block_N/stages chosen by _select_tiling based on output column count
    and output row count:
      split_dim=1 -> k_out_cols = n_cols // 2, m_out = M
      split_dim=0 -> k_out_cols = n_cols, m_out = M // 2
    T.symbolic M/N means one compile per (split_dim, dtype, block_M, block_N,
    stages) supports any M/N, so cache key uses the chosen tiling.
    """
    k_out_cols = n_cols // 2 if split_dim == 1 else n_cols
    block_M, block_N, stages = _select_tiling(k_out_cols, tl_dtype, m_out)
    key = (split_dim, tl_dtype, block_M, block_N, stages)
    if key not in _kernel_cache:
        _kernel_cache[key] = swiglu(block_M, block_N, stages, split_dim, dtype=tl_dtype)
    return _kernel_cache[key]


def swi_glu(input, dim=-1):
    """SwiGLU activation: output = silu(x0) * x1.

    Adapter for cann-bench interface. Normalizes dim, transposes middle dims
    to last, reshapes to 2D, dispatches to the single-input kernel by
    (split_dim, dtype), and restores the output rank/shape.

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

    # Strategy: middle dims transposed to last; dim=0 keeps split_dim=0 (row split);
    # dim==last keeps split_dim=1 (col split).
    need_transpose = 0 < dim < ndim - 1
    perm = None
    if need_transpose:
        perm = list(range(ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        input = input.permute(perm).contiguous()
        split_dim = 1
    elif dim == 0:
        split_dim = 0
    else:  # dim == ndim - 1
        split_dim = 1

    # Reshape to 2D (M, N)
    original_shape = list(input.shape)
    M = 1
    for s in original_shape[:-1]:
        M *= s
    N = original_shape[-1]
    input_2d = input.reshape(M, N)

    # Dispatch to kernel
    # m_out = output rows: split_dim=1 -> M; split_dim=0 -> M//2 (row axis halved)
    m_out = M if split_dim == 1 else M // 2
    kernel = _get_kernel(split_dim, tl_dtype, N, m_out)
    output_2d = kernel(input_2d)

    # Reshape back to original rank (split axis halved)
    out_shape = list(original_shape)
    if split_dim == 1:
        out_shape[-1] = out_shape[-1] // 2
    else:  # split_dim == 0: row axis halved (M halved -> first axis halved)
        out_shape[0] = out_shape[0] // 2
    output = output_2d.reshape(out_shape)

    # Inverse transpose
    if need_transpose:
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        output = output.permute(inv_perm).contiguous()

    return output
