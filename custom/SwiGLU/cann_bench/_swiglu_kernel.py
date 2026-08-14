"""SwiGLU kernel for cann-bench (optimized: staged cast-path tmp buffers).

output = silu(x0) * x1, where x0/x1 are the two halves of input along split_dim.

Optimized vs V3:
- [P0] cast-path tmp buffers (tmp_in0/in1/out) now carry the ``stages`` dim so
  the full cast pipeline (GM -> tmp_in -> a0_ub -> b_ub -> tmp_out -> GM) can
  overlap without a single-buffer stall. V3's single-buffer tmp stalled the
  pipeline every iteration, halving the benefit of stages=2.

Key design (unchanged from V3):
- T.symbolic M/N: compile once per (split_dim, dtype, block_M, block_N, stages),
  supports any shape.
- Persistent NUM_CORES=48 + manual double buffering (stages) to overlap GM->UB
  transfer with compute.
- T.tile.silu single instruction + T.tile.mul gating.
- FP16/BF16 upcast to FP32 via T.tile.cast (matches ACLNN / golden).
- Single input A(M,N); kernel accesses x0 / x1 via row/col offset.
"""

import tilelang
import tilelang.language as T

from ._common import PASS_CONFIGS, CAST_MODE_LOW2HIGH, CAST_MODE_HIGH2LOW


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def _swiglu_kernel(block_M, block_N, stages, split_dim, dtype="float16"):
    """SwiGLU kernel: single input A(M,N), output B(M//m_div, N//n_div).

    Args:
        block_M: row block size (UB budget checked by caller; no internal clamp).
        block_N: col block size for DMA (must be 32B-aligned for DataCopyNd).
        stages: double-buffer stages (1 = single, 2 = double). Caller picks
            per-case via _select_tiling.
        split_dim: 1 = split last dim (n_div=2); 0/-2 = split row axis (m_div=2).
        dtype: "float16" / "float" / "float32" / "bfloat16".
    """
    M = T.symbolic("M")
    N = T.symbolic("N")

    need_cast = dtype not in ("float", "float32")
    ACC_DTYPE = "float32"

    row_split = split_dim == 0 or split_dim == -2
    m_div = 2 if row_split else 1
    n_div = 1 if row_split else 2

    m_num = T.ceildiv(M // m_div, block_M)
    n_num = T.ceildiv(N // n_div, block_N)
    num_blocks = m_num * n_num

    VEC_NUM = 2
    rows_per_vec = block_M // VEC_NUM
    elem_num = rows_per_vec * block_N

    # Launch 48 blocks (A2/A3: each block split into VEC_NUM=2 vector sub-cores
    # via vid). Persistent kernel: each core iterates over its assigned blocks.
    NUM_CORES = 48
    num_iters = T.ceildiv(num_blocks, NUM_CORES)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M // m_div, N // n_div), dtype),  # type: ignore
    ):
        # Offset via ternary (TileLang prim_func if-block reassignment does not
        # propagate to codegen; ternary single-assignment does).
        m_offset = M // 2 if row_split else 0
        n_offset = 0 if row_split else N // 2

        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            # UB buffers (double-buffered stages dim)
            a0_ub = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)
            a1_ub = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)
            b_ub = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)

            # [P0] dtype transit buffers now also carry the stages dim so the
            # full cast pipeline (GM -> tmp_in -> a0_ub -> b_ub -> tmp_out -> GM)
            # can overlap without a single-buffer stall. When need_cast is False
            # these are allocated but unused (MEMORY_PLANNING reuses dead buffers).
            tmp_in0 = T.alloc_ub((stages, rows_per_vec, block_N), dtype)
            tmp_in1 = T.alloc_ub((stages, rows_per_vec, block_N), dtype)
            tmp_out = T.alloc_ub((stages, rows_per_vec, block_N), dtype)

            # NOTE: T.Pipelined was tested and REJECTED — it regressed average
            # speedup from 0.78x to 0.70x. Although codegen shows correct cross-
            # iteration MTE2/V overlap structure (prologue + steady + epilogue
            # with if-guarded prefetch), Developer-mode AUTO_SYNC inserts
            # PipeBarrier<PIPE_ALL> around every operation, negating the pipeline
            # benefit. Additionally, T.Pipelined + fp32 + stages=2 triggers a
            # TileLang compiler segfault (memory planning on dead tmp buffers
            # under pipelined loop). T.serial + cur=i%stages is retained.
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
                        T.copy(A[row, col], tmp_in0[cur, :, :])
                        T.copy(A[row2, col2], tmp_in1[cur, :, :])
                        T.tile.cast(a0_ub[cur, :, :], tmp_in0[cur, :, :], CAST_MODE_LOW2HIGH, elem_num)
                        T.tile.cast(a1_ub[cur, :, :], tmp_in1[cur, :, :], CAST_MODE_LOW2HIGH, elem_num)
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
                        T.tile.cast(tmp_out[cur, :, :], b_ub[cur, :, :], CAST_MODE_HIGH2LOW, elem_num)
                        T.copy(tmp_out[cur, :, :], B[out_row, out_col])
                    else:
                        T.copy(b_ub[cur, :, :], B[out_row, out_col])

    return main
