"""Mish kernel for cann-bench (Developer mode + fp32 intermediate + cast bridge).

output = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x)), element-wise.

Single path (no dispatch needed):
  Developer mode with T.alloc_shared (auto-mapped to UB) + T.tile.xxx 12-step
  decomposition in float32 (ACC_DTYPE) for precision + bf16 compatibility.
  For non-float32 input/output, T.tile.cast bridges UB<->UB at copy-in/copy-out.

Numerical stability (unchanged from DESIGN.md S3.2):
  - softplus(x) = max(x, 0) + ln(1 + exp(-|x|))   (avoids exp(x) overflow)
  - tanh(s)     = 2 * sigmoid(2s) - 1              (T.tile.tanh does not exist)

Reference: custom/mish/mish.py (Stage 3 optimized version)
           examples/activation/sigmoid.py (Developer mode + T.tile.xxx pattern)
"""

import tilelang
from tilelang import language as T

from ._common import (
    ACC_DTYPE,
    CAST_MODE_HIGH2LOW,
    CAST_MODE_LOW2HIGH,
    PASS_CONFIGS,
    VEC_NUM,
)


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def _mish_kernel(M, N, block_M, block_N, dtype="float16"):
    """Mish kernel: y = x * tanh(softplus(x)).

    Developer mode + fp32 intermediate + cast bridge. Single path for all dtypes.

    Args:
        M, N: 2D tensor shape (rows, cols).
        block_M, block_N: tile size per block. block_M must be a multiple of
            VEC_NUM (2), except block_M=1 which uses VEC_NUM=1.
        dtype: "float16" / "float32" / "bfloat16".

    Returns:
        prim_func mapping A (M, N) -> B (M, N), same dtype.
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    rows_per_vec = block_M // VEC_NUM
    elem_num = rows_per_vec * block_N
    need_cast = dtype not in ("float", "float32")

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):  # type: ignore
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            # 1. Allocate UB buffers (compute in float32, tmp_orig bridges cast).
            a_ub = T.alloc_shared((rows_per_vec, block_N), ACC_DTYPE)
            t0_ub = T.alloc_shared((rows_per_vec, block_N), ACC_DTYPE)
            t1_ub = T.alloc_shared((rows_per_vec, block_N), ACC_DTYPE)
            one_ub = T.alloc_shared((rows_per_vec, block_N), ACC_DTYPE)
            b_ub = T.alloc_shared((rows_per_vec, block_N), ACC_DTYPE)
            tmp_orig = T.alloc_shared((rows_per_vec, block_N), dtype)

            # 2. Data copy-in: GM -> UB
            if need_cast:
                T.copy(A[bx * block_M + vid * rows_per_vec, by * block_N], tmp_orig)
                T.tile.cast(a_ub, tmp_orig, CAST_MODE_LOW2HIGH, elem_num)
            else:
                T.copy(A[bx * block_M + vid * rows_per_vec, by * block_N], a_ub)

            # 3. Compute: y = x * tanh(softplus(x))  -- all in float32
            #    Stable softplus: max(x,0) + ln(1+exp(-|x|))   -- 7 steps
            #    Stable tanh:     2*sigmoid(2s) - 1            -- 4 steps
            #    Final mul:       x * tanh(softplus(x))        -- 1 step
            T.tile.fill(one_ub, 1.0)
            T.tile.abs(t0_ub, a_ub)
            T.tile.mul(t0_ub, t0_ub, -1.0)
            T.tile.exp(t0_ub, t0_ub)
            T.tile.add(t0_ub, t0_ub, one_ub)
            T.tile.ln(t0_ub, t0_ub)
            T.tile.max(t1_ub, a_ub, 0.0)
            T.tile.add(t0_ub, t0_ub, t1_ub)
            T.tile.mul(t0_ub, t0_ub, 2.0)
            T.tile.sigmoid(t0_ub, t0_ub)
            T.tile.mul(t0_ub, t0_ub, 2.0)
            # T.tile.sub src1 does NOT accept scalar PrimExpr; use one_ub buffer.
            T.tile.sub(t0_ub, t0_ub, one_ub)
            T.tile.mul(b_ub, a_ub, t0_ub)

            # 4. Data copy-out: UB -> GM
            if need_cast:
                T.tile.cast(tmp_orig, b_ub, CAST_MODE_HIGH2LOW, elem_num)
                T.copy(tmp_orig, B[bx * block_M + vid * rows_per_vec, by * block_N])
            else:
                T.copy(b_ub, B[bx * block_M + vid * rows_per_vec, by * block_N])

    return main


def mish_kernel(M, N, block_M, block_N, dtype="float16"):
    """Public entry: compile and return mish kernel for given (dtype, M, N, block).

    Single path (no dispatch) — mish's fp32 intermediate is required for all
    non-float32 dtypes (fp16 precision + bf16 CANN intrinsic gap).
    """
    return _mish_kernel(M, N, block_M, block_N, dtype=dtype)
