"""Sigmoid activation kernel: y = 1 / (1 + exp(-x)).

Element-wise activation using Developer mode with T.alloc_shared (auto-mapped to UB)
and T.tile.sigmoid buffer-level SIMD primitive.

Uses T.tile.sigmoid (one-step primitive) instead of the 5-step decomposition
(fill/sub/exp/add/reciprocal) because the latter's T.tile.exp and T.tile.reciprocal
internally compute in float16 regardless of buffer dtype, causing precision failures
for float32. T.tile.sigmoid correctly preserves the buffer dtype throughout.

Reference: examples/activation/sigmoidv2.py (T.tile.sigmoid usage).
Design: custom/sigmoid/DESIGN.md §3.2 alternative path.
"""

import tilelang
from tilelang import language as T

# ========== Operator implementation ==========
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def sigmoid(M, N, block_M, block_N, dtype="float16"):
    """Sigmoid kernel: y = 1 / (1 + exp(-x)).

    Args:
        M, N: tensor shape (rows, cols)
        block_M, block_N: tile size per block
        dtype: "float16" or "float32"

    Returns:
        prim_func mapping A (M, N) -> B (M, N)
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):  # type: ignore
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.sigmoid(b_ub, a_ub)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
