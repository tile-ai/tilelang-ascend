"""Mish kernel for cann-bench (Developer mode + fp32 intermediate + cast bridge).

output = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x)), element-wise.

Single path (no dispatch needed):
  Developer mode with T.alloc_shared (auto-mapped to UB) + T.tile.xxx 12-step
  decomposition in float32 (ACC_DTYPE) for precision + bf16 compatibility.
  For non-float32 input/output, T.tile.cast bridges UB<->UB at copy-in/copy-out.

Why no Expert double buffer / Fixed Core / multi-path dispatch (unlike sigmoid):
  - mish has 12-step compute per tile (vs sigmoid 1-step), making NPU kernel time
    the dominant factor for large shapes (bench 0.92-0.96x of torch at 8192x8192).
  - Expert double buffer requires stages=2 * 6 buffers * fp32 = 12 * rpv * bn * 4B,
    exceeding 192KB UB budget for any reasonable tile size.
  - Fixed Core (launch min(block_num,24) + T.serial) was tested and rejected:
    large shapes regressed +25-36% because T.serial loop overhead (171 tiles/core
    for 8192x8192) outweighs launch-count reduction for heavy 12-step compute.
  - bf16 cast is not a separate path: mish's fp32 intermediate is required for ALL
    non-float32 dtypes (both fp16 and bf16), unlike sigmoid where bf16 is special.

Numerical stability (unchanged from DESIGN.md §3.2):
  - softplus(x) = max(x, 0) + ln(1 + exp(-|x|))   (avoids exp(x) overflow)
  - tanh(s)     = 2 * sigmoid(2s) - 1              (T.tile.tanh does not exist)

Reference: custom/mish/mish.py (Stage 3 optimized version)
           examples/activation/swi_glu_v2.py (need_cast + ACC_DTYPE + T.tile.cast)
           examples/xllm_kernels/fused_gdn_gating.py:229-233 (same stable softplus)
"""

import tilelang
from tilelang import language as T

from ._common import CAST_MODE_LOW2HIGH, CAST_MODE_HIGH2LOW, PASS_CONFIGS


VEC_NUM = 2
ACC_DTYPE = "float32"  # intermediate compute dtype for precision + bf16 compatibility


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def _mish_kernel(M, N, block_M, block_N, dtype="float16"):
    """Mish kernel: y = x * tanh(softplus(x)).

    Developer mode + fp32 intermediate + cast bridge. Single path for all dtypes.

    Args:
        M, N: 2D tensor shape (rows, cols).
        block_M, block_N: tile size per block. block_M must be even for VEC_NUM=2
            (except block_M=1 which uses VEC_NUM=1).
        dtype: "float16" / "float32" / "bfloat16".

    Returns:
        prim_func mapping A (M, N) -> B (M, N), same dtype.
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    vec_num = VEC_NUM if block_M >= 2 else 1
    rows_per_vec = block_M // vec_num
    elem_num = rows_per_vec * block_N
    # float32 input computes directly; fp16/bf16 needs cast at GM boundary.
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
    non-float32 dtypes, and Expert/Fixed-Core optimizations were rejected
    (see module docstring).
    """
    return _mish_kernel(M, N, block_M, block_N, dtype=dtype)
