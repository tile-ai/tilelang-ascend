"""Mish activation: y = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x)).

Numerically stable implementation using log-sum-exp trick for softplus and
sigmoid-equivalent for tanh, with float32 intermediate computation to handle
fp16 precision loss and bf16 CANN intrinsic gaps.

Developer mode: T.alloc_shared (auto-mapped to UB) + T.tile.xxx SIMD + auto sync.
"""

import tilelang
import torch
from tilelang import language as T

# ========== JIT Configuration ==========
# AUTO_CV_COMBINE not set: pure Vector op (12 element-wise steps, all on AIV),
# enabling it would spawn an idle AIC core paying launch + buffer init cost.
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_ACC_DTYPE = "float32"
_VEC_NUM = 2

_TORCH_DTYPE_TO_STR = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def mish(M, N, block_M, block_N, dtype="float16"):
    """Mish activation kernel.

    Computes y = x * tanh(softplus(x)) via 12 T.tile.xxx steps in float32.
    Non-fp32 inputs are cast at GM<->UB boundary via T.tile.cast.

    Args:
        M: Number of rows (2D input).
        N: Number of columns (2D input).
        block_M: Row block size (recommend 128).
        block_N: Column block size (recommend 128).
        dtype: Input/output dtype string ("float16", "float32", "bfloat16").

    Returns:
        Compiled prim_func: main(A[M,N], B[M,N]) -> B (out_idx=[1]).
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    rows_per_vec = block_M // _VEC_NUM
    elem_num = rows_per_vec * block_N
    need_cast = dtype not in ("float", "float32")

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):  # type: ignore
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            # UB buffers: all float32 for intermediate computation
            a_ub = T.alloc_shared((rows_per_vec, block_N), _ACC_DTYPE)
            t0_ub = T.alloc_shared((rows_per_vec, block_N), _ACC_DTYPE)
            t1_ub = T.alloc_shared((rows_per_vec, block_N), _ACC_DTYPE)
            one_ub = T.alloc_shared((rows_per_vec, block_N), _ACC_DTYPE)
            b_ub = T.alloc_shared((rows_per_vec, block_N), _ACC_DTYPE)
            tmp_orig = T.alloc_shared((rows_per_vec, block_N), dtype)

            # --- Data load: GM -> UB (with cast for non-fp32) ---
            if need_cast:
                T.copy(A[bx * block_M + vid * rows_per_vec, by * block_N], tmp_orig)
                T.tile.cast(a_ub, tmp_orig, "CAST_NONE", elem_num)
            else:
                T.copy(A[bx * block_M + vid * rows_per_vec, by * block_N], a_ub)

            # --- Compute: y = x * tanh(softplus(x)) -- all fp32, 12 steps ---
            # Numerically stable softplus: max(x,0) + ln(1 + exp(-|x|))
            # exp argument is -|x| <= 0, result in [0,1], never overflows.
            T.tile.fill(one_ub, 1.0)  # one = 1.0
            T.tile.abs(t0_ub, a_ub)  # t0 = |x|
            T.tile.mul(t0_ub, t0_ub, -1.0)  # t0 = -|x|
            T.tile.exp(t0_ub, t0_ub)  # t0 = exp(-|x|) in [0,1]
            T.tile.add(t0_ub, t0_ub, one_ub)  # t0 = 1 + exp(-|x|)
            T.tile.ln(t0_ub, t0_ub)  # t0 = ln(1+exp(-|x|))
            T.tile.max(t1_ub, a_ub, 0.0)  # t1 = max(x, 0)
            T.tile.add(t0_ub, t0_ub, t1_ub)  # t0 = softplus

            # Numerically stable tanh: 2*sigmoid(2s) - 1
            # s = softplus >= 0, so 2s >= 0, exp(-2s) in (0,1], never overflows.
            T.tile.mul(t0_ub, t0_ub, 2.0)  # t0 = 2*softplus
            T.tile.sigmoid(t0_ub, t0_ub)  # t0 = sigmoid(2*softplus)
            T.tile.mul(t0_ub, t0_ub, 2.0)  # t0 = 2*sigmoid
            # T.tile.sub src1 does NOT accept scalar PrimExpr; use one_ub buffer
            T.tile.sub(t0_ub, t0_ub, one_ub)  # t0 = tanh = 2*sigmoid - 1

            # Final: y = x * tanh(softplus(x))
            T.tile.mul(b_ub, a_ub, t0_ub)  # b = x * tanh(softplus(x))

            # --- Data store: UB -> GM (with cast for non-fp32) ---
            if need_cast:
                T.tile.cast(tmp_orig, b_ub, "CAST_RINT", elem_num)
                T.copy(tmp_orig, B[bx * block_M + vid * rows_per_vec, by * block_N])
            else:
                T.copy(b_ub, B[bx * block_M + vid * rows_per_vec, by * block_N])

    return main


def mish_forward(x, block_M=128, block_N=128):
    """Host adapter for Mish activation.

    Handles high-dimensional input by reshaping to 2D (zero-copy view for
    contiguous tensors) and restoring the original shape on output.

    Args:
        x: Input tensor (1D-8D, contiguous, on NPU).
        block_M: Row block size.
        block_N: Column block size.

    Returns:
        Output tensor with same shape and dtype as input.
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape
    dtype_str = _TORCH_DTYPE_TO_STR[x.dtype]
    kernel = mish(M, N, block_M, block_N, dtype=dtype_str)
    y_2d = kernel(x_2d)
    return y_2d.view(orig_shape)
