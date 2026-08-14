"""Sigmoid kernel for cann-bench (optimized: Fixed Core + T.tile.sigmoid).

output = 1 / (1 + exp(-x)), element-wise.

Key design:
- Non-symbolic M/N: compile per (M, N, block_M, block_N, dtype). The adapter
  caches compiled kernels in-process so each unique shape compiles only once.
  (T.symbolic mode triggers a tilelang compiler segfault with T.tile.sigmoid.)
- Fixed Core NUM_CORES=24: launch 24 blocks (Ascend A2/A3 physical AI Core
  count), each core iterates over its assigned tiles via T.serial. Eliminates
  per-block launch overhead vs launching one block per tile.
- T.tile.sigmoid single instruction (one-step primitive) instead of the 5-step
  decomposition (fill/sub/exp/add/reciprocal) because the latter's T.tile.exp
  and T.tile.reciprocal internally compute in float16 regardless of buffer
  dtype, causing precision failures for float32. T.tile.sigmoid correctly
  preserves the buffer dtype throughout.
- VEC_NUM=2: each block split into 2 vector sub-cores via vid (falls back to 1
  when block_M < 2 for 1D / single-row inputs).
- AUTO_CV_COMBINE OFF: sigmoid is pure Vector (element-wise); the pass was
  adding an idle AIC core, leaving it idle but still paying launch + buffer
  init cost.
- Buffer allocated inside T.serial loop (hoisting outside triggers a tilelang
  compiler stall with T.tile.sigmoid under Developer-mode AUTO_SYNC).
"""

import tilelang
from tilelang import language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CORE_NUM = 24


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def _sigmoid_kernel(M, N, block_M, block_N, dtype="float16"):
    """Sigmoid kernel: input A(M,N) -> output B(M,N), element-wise.

    Args:
        M, N: tensor shape (rows, cols) — concrete ints, compiled per shape.
        block_M: row block size (UB budget checked by caller).
        block_N: col block size for DMA (must be 32B-aligned).
        dtype: "float16" / "float" / "float32" / "bfloat16".
    """
    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N
    block_num = m_num * n_num
    launch_cores = min(block_num, CORE_NUM)
    single_core_load = (block_num + launch_cores - 1) // launch_cores

    # VEC_NUM adapts to block_M: use 2 (dual vector sub-core) when possible,
    # fall back to 1 for very small block_M (e.g. M=1 → block_M=1).
    VEC_NUM = 2 if block_M >= 2 else 1

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):  # type: ignore
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            for block_idx in T.serial(single_core_load):
                logical_cid = block_idx * launch_cores + cid
                bx = logical_cid // n_num
                by = logical_cid % n_num

                a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
                b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.tile.sigmoid(b_ub, a_ub)
                T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
