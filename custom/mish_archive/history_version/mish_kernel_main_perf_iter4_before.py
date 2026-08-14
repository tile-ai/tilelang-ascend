"""Mish activation kernel: y = x * tanh(softplus(x)).

Element-wise activation using Developer mode with T.alloc_shared (auto-mapped to UB)
and T.tile.xxx buffer-level SIMD primitives.

Precision strategy (Stage 2 attempt 2 fix, unchanged in Stage 3):
  - All intermediate compute buffers are allocated in float32 (ACC_DTYPE) to:
    (a) fix bf16 compile failure (CANN Muls/Maxs/Exp/Adds/Div don't support __bf16),
    (b) fix fp16 precision loss from 12-step accumulated error.
  - For non-float32 input/output, a single reusable temp UB buffer (tmp_orig) in the
    original dtype bridges GM↔UB; T.tile.cast handles UB↔UB dtype conversion:
      cast-in : T.copy(GM -> tmp_orig) -> T.tile.cast(a_ub_fp32, tmp_orig, "CAST_NONE", N)
      cast-out: T.tile.cast(tmp_orig, b_ub_fp32, "CAST_RINT", N) -> T.copy(tmp_orig -> GM)
  - For float32 input/output, no cast needed; T.copy goes directly GM↔UB(fp32).

Numerical stability (unchanged from DESIGN.md §3.2):
  - softplus(x) = max(x, 0) + ln(1 + exp(-|x|))   (avoids exp(x) overflow for large x)
  - tanh(s)     = 2 * sigmoid(2s) - 1              (T.tile.tanh does not exist in this project)

Key API constraints (DESIGN.md §3.2/§3.4):
  - T.tile.sub(dst, src0, src1) does NOT accept scalar PrimExpr for src1; use one_ub
    buffer (pre-filled with 1.0 via T.tile.fill) instead of scalar 1.0.
  - T.tile.add/mul/max accept scalar PrimExpr directly.
  - T.tile.ln (natural log) is named `ln`, not `log`.
  - T.copy does NOT support cross-dtype (src/dst must match); use T.tile.cast for conversion.

Perf tuning iter1 (Stage 3):
  - [#1] Closed TL_ASCEND_AUTO_CV_COMBINE: mish is pure Vector (12-step element-wise),
    the pass was emitting MIX_AIC_1_2 with all compute inside `if ASCEND_IS_AIV`,
    leaving the AIC core idle but still paying its launch + L0A/L0B/L1/L0C buffer
    init cost. Same finding as custom/sigmoid/sigmoid.py iter1.
  - NOTE: [#3] Fixed Core (launch min(block_num,24) + T.serial) was tested and
    REJECTED — mish's 12-step compute per tile is much heavier than sigmoid's
    1-step, so T.serial's serial loop overhead (171 tiles/core for (8192,8192))
    outweighs the launch-count reduction benefit. Large shapes regressed
    +25-36%. Kept the original T.Kernel(m_num*n_num) launch.

Reference: examples/activation/swi_glu_v2.py (need_cast + ACC_DTYPE + T.tile.cast pattern)
           examples/activation/sigmoid.py (Developer mode + pass_configs triplet)
           examples/activation/tanh.py (VEC_NUM=2 + T.ceildiv + fill/sub pattern)
           examples/xllm_kernels/fused_gdn_gating.py:229-233 (same stable softplus)
           custom/sigmoid/sigmoid.py (Stage 3 [#1] close CV pattern)

Design: custom/mish/DESIGN.md §3.2 API mapping + §3.3 pseudocode + §12 perf target.
"""

import tilelang
from tilelang import language as T

# ========== Operator implementation ==========
# [#1] AUTO_CV_COMBINE intentionally OFF: mish is pure Vector (12-step element-wise),
# the pass was adding an idle AIC core. AUTO_SYNC + MEMORY_PLANNING kept on
# (Developer mode auto-sync + UB memory planning).
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

VEC_NUM = 2
ACC_DTYPE = "float32"  # intermediate compute dtype for precision + bf16 compatibility


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def mish(M, N, block_M, block_N, dtype="float16"):
    """Mish kernel: y = x * tanh(softplus(x)).

    Args:
        M, N: tensor shape (rows, cols). High-dim inputs are flattened to 2D by host.
        block_M, block_N: tile size per block.
        dtype: "float16" / "float32" / "bfloat16".

    Returns:
        prim_func mapping A (M, N) -> B (M, N), same dtype.

    Precision:
        All 12-step intermediate compute runs in float32 (ACC_DTYPE) to avoid
        (a) bf16 CANN intrinsic unsupported errors and (b) fp16 accumulated error.
        For non-float32 dtype, T.tile.cast bridges UB↔UB at copy-in/copy-out.
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    rows_per_vec = block_M // VEC_NUM
    elem_num = rows_per_vec * block_N
    # float32 input computes directly; fp16/bf16 needs cast at GM boundary.
    need_cast = dtype not in ("float", "float32")

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):  # type: ignore
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            # 1. Allocate UB buffers.
            #    Compute buffers in float32 (precision + bf16 compatibility).
            #    tmp_orig (original dtype) bridges GM<->UB when need_cast.
            #    Allocated unconditionally (TileLang DSL parser requires the symbol
            #    to exist even when the float32 path never references it); for float32
            #    input MEMORY_PLANNING will elide the unused buffer.
            a_ub = T.alloc_shared((rows_per_vec, block_N), ACC_DTYPE)
            t0_ub = T.alloc_shared((rows_per_vec, block_N), ACC_DTYPE)
            t1_ub = T.alloc_shared((rows_per_vec, block_N), ACC_DTYPE)
            one_ub = T.alloc_shared((rows_per_vec, block_N), ACC_DTYPE)
            b_ub = T.alloc_shared((rows_per_vec, block_N), ACC_DTYPE)
            tmp_orig = T.alloc_shared((rows_per_vec, block_N), dtype)

            # 2. Data copy-in: GM -> UB
            if need_cast:
                T.copy(A[bx * block_M + vid * rows_per_vec, by * block_N], tmp_orig)
                T.tile.cast(a_ub, tmp_orig, "CAST_NONE", elem_num)  # fp16/bf16 -> fp32 (lossless)
            else:
                T.copy(A[bx * block_M + vid * rows_per_vec, by * block_N], a_ub)

            # 3. Compute: y = x * tanh(softplus(x))  -- all in float32
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
            if need_cast:
                T.tile.cast(tmp_orig, b_ub, "CAST_RINT", elem_num)  # fp32 -> fp16/bf16 (round)
                T.copy(tmp_orig, B[bx * block_M + vid * rows_per_vec, by * block_N])
            else:
                T.copy(b_ub, B[bx * block_M + vid * rows_per_vec, by * block_N])

    return main
