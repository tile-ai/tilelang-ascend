"""Mish activation kernel: y = x * tanh(softplus(x)).

Element-wise activation using Developer mode with T.alloc_shared (auto-mapped to UB)
and T.tile.xxx buffer-level SIMD primitives.

Numerical stability:
  - softplus(x) = max(x, 0) + ln(1 + exp(-|x|))   (avoids exp(x) overflow for large x)
  - tanh(s)     = 2 * sigmoid(2s) - 1              (T.tile.tanh does not exist in this project)

Key API constraints (DESIGN.md §3.2/§3.4):
  - T.tile.sub(dst, src0, src1) does NOT accept scalar PrimExpr for src1; use one_ub
    buffer (pre-filled with 1.0 via T.tile.fill) instead of scalar 1.0.
  - T.tile.add/mul/max accept scalar PrimExpr directly.
  - T.tile.ln (natural log) is named `ln`, not `log`.

Reference: examples/activation/sigmoid.py (Developer mode + pass_configs triplet)
           examples/activation/tanh.py (VEC_NUM=2 + T.ceildiv + fill/sub pattern)
           examples/xllm_kernels/fused_gdn_gating.py:229-233 (same stable softplus)

Design: custom/mish/DESIGN.md §3.2 API mapping + §3.3 pseudocode.
"""

import tilelang
from tilelang import language as T

# ========== Operator implementation ==========
# Developer mode: T.alloc_shared auto-maps to UB; auto-sync; auto memory planning.
# AUTO_CV_COMBINE kept on (per DESIGN.md §7.3); for pure Vector it degrades to no-op.
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

VEC_NUM = 2


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def mish(M, N, block_M, block_N, dtype="float16"):
    """Mish kernel: y = x * tanh(softplus(x)).

    Args:
        M, N: tensor shape (rows, cols). High-dim inputs are flattened to 2D by host.
        block_M, block_N: tile size per block.
        dtype: "float16" / "float32" / "bfloat16".

    Returns:
        prim_func mapping A (M, N) -> B (M, N), same dtype.
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):  # type: ignore
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            # 1. Allocate UB buffers (Developer: alloc_shared auto-maps to UB).
            #    vid splits rows: each V core handles block_M // VEC_NUM rows.
            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            t0_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            t1_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            one_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            # 2. Data copy-in: GM -> UB
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            # 3. Compute: y = x * tanh(softplus(x))
            #    Stable softplus: max(x,0) + ln(1+exp(-|x|))   -- 7 steps
            #    Stable tanh:     2*sigmoid(2s) - 1            -- 4 steps
            #    Final mul:       x * tanh(softplus(x))        -- 1 step
            T.tile.fill(one_ub, 1.0)  # one = 1.0 (constant buffer for add/sub)
            T.tile.abs(t0_ub, a_ub)  # t0 = |x|
            T.tile.mul(t0_ub, t0_ub, -1.0)  # t0 = -|x|
            T.tile.exp(t0_ub, t0_ub)  # t0 = exp(-|x|)  in [0,1]
            T.tile.add(t0_ub, t0_ub, one_ub)  # t0 = 1 + exp(-|x|)
            T.tile.ln(t0_ub, t0_ub)  # t0 = ln(1 + exp(-|x|))
            T.tile.max(t1_ub, a_ub, 0.0)  # t1 = max(x, 0)
            T.tile.add(t0_ub, t0_ub, t1_ub)  # t0 = softplus = max(x,0) + ln(1+exp(-|x|))
            T.tile.mul(t0_ub, t0_ub, 2.0)  # t0 = 2*softplus
            T.tile.sigmoid(t0_ub, t0_ub)  # t0 = sigmoid(2*softplus)
            T.tile.mul(t0_ub, t0_ub, 2.0)  # t0 = 2*sigmoid
            T.tile.sub(t0_ub, t0_ub, one_ub)  # t0 = tanh = 2*sigmoid - 1
            T.tile.mul(b_ub, a_ub, t0_ub)  # b  = x * tanh(softplus(x))

            # 4. Data copy-out: UB -> GM
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
