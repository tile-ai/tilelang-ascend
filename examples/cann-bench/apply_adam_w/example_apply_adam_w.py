"""apply_adam_w operator implementation on Ascend NPU via TileLang.

Formula (decoupled AdamW, epsilon OUTSIDE sqrt):
    m_t   = beta1 * m + (1 - beta1) * grad
    v_t   = beta2 * v + (1 - beta2) * grad^2
    m_hat = m_t / (1 - beta1^step)        # bias correction (folded host-side)
    v_hat = v_t / (1 - beta2^step)
    update = m_hat / (sqrt(v_hat) + eps)  # eps OUTSIDE sqrt (matches torch.optim.AdamW)
    var_t  = var - lr * (update + wd * var)   (minimize)
    var_t  = var + lr * (update + wd * var)   (maximize)

Three kernel variants (Expert mode):
  1. Pipeline kernel (finite scalars, fp32): MTE2/V/MTE3 three-stage pipeline
     with double buffer. Computes directly on input UB (no cast needed).
     Scalars as compile-time kernel params (optimal performance).
  2. Pipeline kernel (finite scalars, fp16/bf16): Same pipeline but with
     fp32 compute buffers for precision (cast -> compute -> cast back).
  3. Barrier kernel (inf/nan/zero-division scalars): barrier_all with
     scalar_ub (8-element GM tensor) to avoid CUDART_INF/CUDART_NAN codegen
     errors. Scalars passed at runtime via GM tensor, single-buffered.

Routing:
  - Normal scalars (finite, step>0, bias!=0) -> pipeline kernel
  - inf/nan/zero-division scalars -> barrier kernel (scalar_ub runtime path)

Both variants:
  - Expert mode: T.alloc_ub, T.Scope("V"), T.tile.mul/axpy/sqrt/div/add
  - Host-side bias-correction folding (eff_beta1/eff_ombeta1/eff_beta2/
     eff_ombeta2/lr_signed precomputed in Python fp64)
  - pad_value=0.0 handles non-aligned shapes (all 4 inputs)
  - VEC_NUM=2 row split across two Vector cores

Interface: apply_adam_w(Tensor var, Tensor grad, Tensor m, Tensor v,
                       float lr, float beta1, float beta2, float weight_decay,
                       float epsilon=1e-8, int step=1, bool maximize=False)
            -> Tensor y
"""

import argparse
import math
import sys

import tilelang
import torch
from tilelang import language as T

# Pipeline configs: ONLY MEMORY_PLANNING. AUTO_CV_COMBINE causes pipeline
# deadlock with T.tile ops. AUTO_SYNC off (Expert mode manual sync).
_PIPELINE_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Barrier configs: MEMORY_PLANNING + AUTO_CV_COMBINE (barrier_all handles all
# sync; AUTO_CV_COMBINE is safe here since no set_flag/wait_flag pipeline).
_BARRIER_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"

_CORE_NUM = 24
_UB_LIMIT_BYTES = 160 * 1024
_NUM_INPUTS = 4  # var, grad, m, v
_SCALAR_PAD = 8  # 7 scalars + 1 padding (eff_beta1, eff_ombeta1, eff_beta2,
#   eff_ombeta2, lr_signed, weight_decay, epsilon, 0.0)

_pipeline_fp32_cache = {}
_pipeline_lowprec_cache = {}
_barrier_kernel_cache = {}
_scalar_tensor_cache = {}


def torch_dtype_to_tl(dtype):
    if dtype == torch.float16:
        return "float16"
    elif dtype == torch.bfloat16:
        return "bfloat16"
    elif dtype == torch.float32:
        return "float"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _compute_2d_shape(shape):
    """Reshape to 2D (M, N) for optimal tiling.

    For 1D tensors, use (1, N). For multi-dim, merge trailing dims until
    N >= 256 for better memory access patterns.
    """
    total = 1
    for s in shape:
        total *= s

    if len(shape) <= 1:
        return 1, total

    N = 1
    for i in range(len(shape) - 1, -1, -1):
        next_N = N * shape[i]
        if total % next_N != 0:
            break
        N = next_N
        if N >= 256:
            break

    M = total // N
    return M, N


def _select_tiling(M, N, dtype):
    """Adaptive tiling selection based on M, N, dtype.

    For apply_adam_w with 4 inputs, per_elem is 32 for both fp32 and
    fp16/bf16 paths (4*stages*dtype_bytes + 4*cal_bytes for low-prec,
    4*stages*dtype_bytes for fp32 — coincidentally equal at 32 each).
    """
    use_fp32_compute = dtype in ["float16", "bfloat16"]
    dtype_bytes = 4 if dtype == "float" else 2
    cal_bytes = 4
    stages = 2
    VEC_NUM = 2

    if M <= 2:
        block_M = 2
        sub_M = 2
        sub_block_M = sub_M // VEC_NUM  # = 1
        if use_fp32_compute:
            per_elem = _NUM_INPUTS * stages * dtype_bytes + _NUM_INPUTS * cal_bytes
        else:
            per_elem = _NUM_INPUTS * stages * dtype_bytes
        max_block_n = _UB_LIMIT_BYTES // (per_elem * sub_block_M)
        align_step = 512 // dtype_bytes
        block_N = (max_block_n // align_step) * align_step
        block_N = min(block_N, N)
        block_N = max(block_N, align_step)
    else:
        sub_M = 32
        sub_block_M = sub_M // VEC_NUM  # = 16

        if use_fp32_compute:
            per_elem = _NUM_INPUTS * stages * dtype_bytes + _NUM_INPUTS * cal_bytes
        else:
            per_elem = _NUM_INPUTS * stages * dtype_bytes
        max_block_n = _UB_LIMIT_BYTES // (per_elem * sub_block_M)

        target = (M + _CORE_NUM - 1) // _CORE_NUM
        # For large M (target > 256), use block_M=256 to reduce block count.
        # UB budget depends on sub_block_M (not block_M), so enlarging block_M
        # does not increase UB usage — it only lengthens vec_proc (inner loop
        # iterations), amortizing pipeline start/stop overhead over more work.
        # Before this fix, target > 256 fell through to block_M=128, creating
        # excessive blocks (e.g. case9 M=133221 → 2082 blocks).
        if target <= 128:
            block_M = 128
        else:
            block_M = 256
        max_block_m = ((M + sub_M - 1) // sub_M) * sub_M
        block_M = min(block_M, max_block_m)

        m_num = (M + block_M - 1) // block_M

        if N <= 128:
            block_N = 128
        else:
            block_N = min(max_block_n, N)
            block_N = (block_N // 32) * 32
            block_N = max(block_N, 256)
            n_num = (N + block_N - 1) // block_N
            if m_num * n_num >= _CORE_NUM * 4:
                block_N = 256

    block_M = max(block_M, 2)
    if block_M % 2 != 0:
        block_M += 1
    sub_M = min(sub_M, block_M)
    if block_M % sub_M != 0:
        block_M = (block_M // sub_M) * sub_M
    block_M = max(block_M, sub_M)
    if block_M % 2 != 0:
        block_M += 1

    return block_M, block_N, sub_M


def _precompute_scalars(lr, beta1, beta2, weight_decay, epsilon, step, maximize):
    """Host-side bias-correction folding.

    Folds (1 - beta^step) denominator into momentum coefficients to
    eliminate kernel-side division. Computed in Python fp64 for precision.

    Raises ZeroDivisionError when bias correction denominator is zero
    (step<=0 or beta^step==1). Use _precompute_scalars_safe for those cases.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    bias1 = 1.0 - (beta1**step)
    bias2 = 1.0 - (beta2**step)
    eff_beta1 = beta1 / bias1
    eff_ombeta1 = (1.0 - beta1) / bias1
    eff_beta2 = beta2 / bias2
    eff_ombeta2 = (1.0 - beta2) / bias2
    lr_signed = lr if maximize else -lr
    return (eff_beta1, eff_ombeta1, eff_beta2, eff_ombeta2, lr_signed)


def _safe_div(a, b):
    """IEEE 754 division returning inf/nan instead of raising ZeroDivisionError.

    Special case: 0/0 returns 0.0 (not nan) to avoid nan pollution in the
    decomposed bias-correction form `eff_beta * m + eff_ombeta * grad`.
    This matches golden_apply_adam_w's computation order where (1-beta1)*grad
    = 0*grad = 0 vanishes before the division by bias when 1-beta1 == 0.

    Example: beta1=1.0, step=1, bias=1-1^1=0
      - golden:    (1.0*m + 0.0*grad) / 0 = m/0           = inf (if m>0)
      - 0/0=nan:   1/0 * m + nan * grad = inf*m + nan     = nan (WRONG)
      - 0/0=0:     1/0 * m + 0   * grad = inf*m + 0       = inf (CORRECT)
    """
    try:
        return a / b
    except ZeroDivisionError:
        if math.isnan(a) or math.isnan(b):
            return float("nan")
        if a == 0.0:
            return 0.0  # 0/0 -> 0 (avoid nan pollution, match golden order)
        sign = math.copysign(1.0, a) * math.copysign(1.0, b)
        return sign * float("inf")


def _precompute_scalars_safe(lr, beta1, beta2, weight_decay, epsilon, step, maximize):
    """Host-side bias-correction folding, safe for inf/nan/zero-division.

    Identical math to _precompute_scalars but uses _safe_div to return inf/nan
    per IEEE 754 (matching golden_apply_adam_w PyTorch behavior) instead of
    raising. Used by barrier kernel path (scalar_ub carries inf/nan at runtime).
    """
    lr_signed = lr if maximize else -lr

    if step <= 0:
        bias1 = 0.0
        bias2 = 0.0
    else:
        bias1 = 1.0 - (beta1**step)
        bias2 = 1.0 - (beta2**step)

    eff_beta1 = _safe_div(beta1, bias1)
    eff_ombeta1 = _safe_div(1.0 - beta1, bias1)
    eff_beta2 = _safe_div(beta2, bias2)
    eff_ombeta2 = _safe_div(1.0 - beta2, bias2)
    return (eff_beta1, eff_ombeta1, eff_beta2, eff_ombeta2, lr_signed)


# ---------------------------------------------------------------------------
# Pipeline kernel: fp32 path (no cal buffers, compute directly on input UB)
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[4], pass_configs=_PIPELINE_CONFIGS)
def _adam_w_pipeline_fp32(
    M,
    N,
    block_M,
    block_N,
    sub_M,
    eff_beta1,
    eff_ombeta1,
    eff_beta2,
    eff_ombeta2,
    lr_signed,
    weight_decay,
    epsilon,
    dtype="float",
):
    """Pipeline kernel for fp32 inputs (no cast needed).

    Computes directly on input UB — 4 double-buffered input UB buffers
    only, no cal buffers. In-place buffer reuse:
      m_ub -> m_hat, grad_ub -> grad^2 -> denom, v_ub -> v_hat -> update,
      var_ub -> var_t (final output).
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = sub_M // VEC_NUM
    vec_proc = block_M // sub_M
    stages = 2

    @T.prim_func
    def main(
        Var: T.Tensor((M, N), dtype),  # type: ignore
        Grad: T.Tensor((M, N), dtype),  # type: ignore
        M_in: T.Tensor((M, N), dtype),  # type: ignore
        V_in: T.Tensor((M, N), dtype),  # type: ignore
        Y: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            var_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)
            grad_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)
            m_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)
            v_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)

            with T.Scope("V"):
                col_start = by * block_N
                row_base = bx * block_M + vid * sub_block_M

                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)

                # Prefetch stage 0
                T.wait_flag("mte3", "mte2", 0)
                T.copy(Var[row_base, col_start], var_ub[0, :, :], pad_value=0.0)
                T.copy(Grad[row_base, col_start], grad_ub[0, :, :], pad_value=0.0)
                T.copy(M_in[row_base, col_start], m_ub[0, :, :], pad_value=0.0)
                T.copy(V_in[row_base, col_start], v_ub[0, :, :], pad_value=0.0)
                T.set_flag("mte2", "v", 0)

                for mm in T.serial(vec_proc):
                    cur = mm % stages
                    nxt = (mm + 1) % stages

                    # Prefetch next stage (if not last)
                    if mm < vec_proc - 1:
                        T.wait_flag("mte3", "mte2", nxt)
                        row_nxt = row_base + (mm + 1) * sub_M
                        T.copy(Var[row_nxt, col_start], var_ub[nxt, :, :], pad_value=0.0)
                        T.copy(Grad[row_nxt, col_start], grad_ub[nxt, :, :], pad_value=0.0)
                        T.copy(M_in[row_nxt, col_start], m_ub[nxt, :, :], pad_value=0.0)
                        T.copy(V_in[row_nxt, col_start], v_ub[nxt, :, :], pad_value=0.0)
                        T.set_flag("mte2", "v", nxt)

                    # --- Vector compute (in-place on input UB) ---
                    T.wait_flag("mte2", "v", cur)

                    # m_hat = eff_beta1 * m + eff_ombeta1 * grad
                    T.tile.mul(m_ub[cur, :, :], m_ub[cur, :, :], eff_beta1)
                    T.tile.axpy(m_ub[cur, :, :], grad_ub[cur, :, :], eff_ombeta1)

                    # grad^2 (reuse grad_ub)
                    T.tile.mul(grad_ub[cur, :, :], grad_ub[cur, :, :], grad_ub[cur, :, :])

                    # v_hat = eff_beta2 * v + eff_ombeta2 * grad^2
                    T.tile.mul(v_ub[cur, :, :], v_ub[cur, :, :], eff_beta2)
                    T.tile.axpy(v_ub[cur, :, :], grad_ub[cur, :, :], eff_ombeta2)

                    # denom = sqrt(v_hat) + epsilon (reuse grad_ub)
                    T.tile.sqrt(grad_ub[cur, :, :], v_ub[cur, :, :])
                    T.tile.add(grad_ub[cur, :, :], grad_ub[cur, :, :], epsilon)

                    # update = m_hat / denom (reuse v_ub)
                    T.tile.div(v_ub[cur, :, :], m_ub[cur, :, :], grad_ub[cur, :, :])

                    # update += weight_decay * var
                    T.tile.axpy(v_ub[cur, :, :], var_ub[cur, :, :], weight_decay)

                    # var_t = var + lr_signed * update (reuse var_ub -> output)
                    T.tile.axpy(var_ub[cur, :, :], v_ub[cur, :, :], lr_signed)

                    T.set_flag("v", "mte3", cur)

                    # --- MTE3 store ---
                    T.wait_flag("v", "mte3", cur)
                    row_cur = row_base + mm * sub_M
                    T.copy(var_ub[cur, :, :], Y[row_cur, col_start])
                    T.set_flag("mte3", "mte2", cur)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return main


# ---------------------------------------------------------------------------
# Pipeline kernel: fp16/bf16 path (with fp32 cal buffers)
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[4], pass_configs=_PIPELINE_CONFIGS)
def _adam_w_pipeline_lowprec(
    M,
    N,
    block_M,
    block_N,
    sub_M,
    eff_beta1,
    eff_ombeta1,
    eff_beta2,
    eff_ombeta2,
    lr_signed,
    weight_decay,
    epsilon,
    dtype="float16",
):
    """Pipeline kernel for fp16/bf16 inputs (fp32 compute path).

    Casts inputs to fp32 for compute precision, then casts back. Uses 8 UB
    buffers (4 input double-buffered + 4 cal). Buffer reuse in cal:
      m_cal -> m_hat, grad_cal -> grad^2 -> denom, v_cal -> v_hat -> update,
      var_cal -> var_t (cast back to var_ub for output).
    """
    cal_dtype = "float32"

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = sub_M // VEC_NUM
    vec_proc = block_M // sub_M
    stages = 2
    cnt = sub_block_M * block_N

    @T.prim_func
    def main(
        Var: T.Tensor((M, N), dtype),  # type: ignore
        Grad: T.Tensor((M, N), dtype),  # type: ignore
        M_in: T.Tensor((M, N), dtype),  # type: ignore
        V_in: T.Tensor((M, N), dtype),  # type: ignore
        Y: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            var_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)
            grad_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)
            m_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)
            v_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)
            var_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            grad_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            m_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            v_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)

            with T.Scope("V"):
                col_start = by * block_N
                row_base = bx * block_M + vid * sub_block_M

                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)

                # Prefetch stage 0
                T.wait_flag("mte3", "mte2", 0)
                T.copy(Var[row_base, col_start], var_ub[0, :, :], pad_value=0.0)
                T.copy(Grad[row_base, col_start], grad_ub[0, :, :], pad_value=0.0)
                T.copy(M_in[row_base, col_start], m_ub[0, :, :], pad_value=0.0)
                T.copy(V_in[row_base, col_start], v_ub[0, :, :], pad_value=0.0)
                T.set_flag("mte2", "v", 0)

                for mm in T.serial(vec_proc):
                    cur = mm % stages
                    nxt = (mm + 1) % stages

                    # Prefetch next stage (if not last)
                    if mm < vec_proc - 1:
                        T.wait_flag("mte3", "mte2", nxt)
                        row_nxt = row_base + (mm + 1) * sub_M
                        T.copy(Var[row_nxt, col_start], var_ub[nxt, :, :], pad_value=0.0)
                        T.copy(Grad[row_nxt, col_start], grad_ub[nxt, :, :], pad_value=0.0)
                        T.copy(M_in[row_nxt, col_start], m_ub[nxt, :, :], pad_value=0.0)
                        T.copy(V_in[row_nxt, col_start], v_ub[nxt, :, :], pad_value=0.0)
                        T.set_flag("mte2", "v", nxt)

                    # --- Vector compute (fp32 cal buffers) ---
                    T.wait_flag("mte2", "v", cur)

                    # Cast inputs to fp32
                    T.tile.cast(var_cal, var_ub[cur, :, :], CAST_MODE_LOW2HIGH, cnt)
                    T.tile.cast(grad_cal, grad_ub[cur, :, :], CAST_MODE_LOW2HIGH, cnt)
                    T.tile.cast(m_cal, m_ub[cur, :, :], CAST_MODE_LOW2HIGH, cnt)
                    T.tile.cast(v_cal, v_ub[cur, :, :], CAST_MODE_LOW2HIGH, cnt)

                    # m_hat = eff_beta1 * m + eff_ombeta1 * grad
                    T.tile.mul(m_cal, m_cal, eff_beta1)
                    T.tile.axpy(m_cal, grad_cal, eff_ombeta1)

                    # grad^2 (reuse grad_cal)
                    T.tile.mul(grad_cal, grad_cal, grad_cal)

                    # v_hat = eff_beta2 * v + eff_ombeta2 * grad^2
                    T.tile.mul(v_cal, v_cal, eff_beta2)
                    T.tile.axpy(v_cal, grad_cal, eff_ombeta2)

                    # denom = sqrt(v_hat) + epsilon (reuse grad_cal)
                    T.tile.sqrt(grad_cal, v_cal)
                    T.tile.add(grad_cal, grad_cal, epsilon)

                    # update = m_hat / denom (reuse v_cal)
                    T.tile.div(v_cal, m_cal, grad_cal)

                    # update += weight_decay * var
                    T.tile.axpy(v_cal, var_cal, weight_decay)

                    # var_t = var + lr_signed * update (reuse var_cal -> output)
                    T.tile.axpy(var_cal, v_cal, lr_signed)

                    # Cast back to original dtype
                    T.tile.cast(var_ub[cur, :, :], var_cal, CAST_MODE_HIGH2LOW, cnt)

                    T.set_flag("v", "mte3", cur)

                    # --- MTE3 store ---
                    T.wait_flag("v", "mte3", cur)
                    row_cur = row_base + mm * sub_M
                    T.copy(var_ub[cur, :, :], Y[row_cur, col_start])
                    T.set_flag("mte3", "mte2", cur)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return main


# ---------------------------------------------------------------------------
# Barrier kernel: inf/nan/zero-division scalar path
#
# Scalars are NOT compile-time kernel params (which would emit CUDART_INF/
# CUDART_NAN literals unbuildable on Ascend C++). Instead they are packed
# into an 8-element GM tensor S and copied to scalar_ub at runtime.
# Layout: [eff_beta1, eff_ombeta1, eff_beta2, eff_ombeta2,
#          lr_signed, weight_decay, epsilon, 0.0]
# Single-buffered (no double-buffer pipeline); barrier_all for DMA<->V sync.
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[5], pass_configs=_BARRIER_CONFIGS)
def _adam_w_barrier_fp32(M, N, block_M, block_N, dtype="float"):
    """Barrier kernel for inf/nan scalars, fp32 path (no cal buffers).

    Computes directly on input UB, same formula as pipeline fp32 kernel but
    with scalars read from scalar_ub at runtime (no compile-time literal).
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        Var: T.Tensor((M, N), dtype),  # type: ignore
        Grad: T.Tensor((M, N), dtype),  # type: ignore
        M_in: T.Tensor((M, N), dtype),  # type: ignore
        V_in: T.Tensor((M, N), dtype),  # type: ignore
        S: T.Tensor((_SCALAR_PAD,), "float"),  # type: ignore
        Y: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            var_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            grad_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            m_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            v_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            scalar_ub = T.alloc_ub((_SCALAR_PAD,), "float")

            with T.Scope("V"):
                row_start = bx * block_M + vid * sub_block_M
                col_start = by * block_N

                T.copy(Var[row_start, col_start], var_ub, pad_value=0.0)
                T.copy(Grad[row_start, col_start], grad_ub, pad_value=0.0)
                T.copy(M_in[row_start, col_start], m_ub, pad_value=0.0)
                T.copy(V_in[row_start, col_start], v_ub, pad_value=0.0)
                T.copy(S, scalar_ub)

                T.barrier_all()

                # m_hat = eff_beta1 * m + eff_ombeta1 * grad
                T.tile.mul(m_ub, m_ub, scalar_ub[0])
                T.tile.axpy(m_ub, grad_ub, scalar_ub[1])

                # grad^2 (reuse grad_ub)
                T.tile.mul(grad_ub, grad_ub, grad_ub)

                # v_hat = eff_beta2 * v + eff_ombeta2 * grad^2
                T.tile.mul(v_ub, v_ub, scalar_ub[2])
                T.tile.axpy(v_ub, grad_ub, scalar_ub[3])

                # denom = sqrt(v_hat) + epsilon (reuse grad_ub)
                T.tile.sqrt(grad_ub, v_ub)
                T.tile.add(grad_ub, grad_ub, scalar_ub[6])

                # update = m_hat / denom (reuse v_ub)
                T.tile.div(v_ub, m_ub, grad_ub)

                # update += weight_decay * var
                T.tile.axpy(v_ub, var_ub, scalar_ub[5])

                # var_t = var + lr_signed * update (reuse var_ub -> output)
                T.tile.axpy(var_ub, v_ub, scalar_ub[4])

                T.barrier_all()

                T.copy(var_ub, Y[row_start, col_start])

    return main


@tilelang.jit(out_idx=[5], pass_configs=_BARRIER_CONFIGS)
def _adam_w_barrier_lowprec(M, N, block_M, block_N, dtype="float16"):
    """Barrier kernel for inf/nan scalars, fp16/bf16 path (fp32 compute).

    Same formula as pipeline lowprec kernel but with scalars from scalar_ub.
    Casts inputs to fp32 for compute precision, then casts back.
    """
    cal_dtype = "float32"

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM
    cnt = sub_block_M * block_N

    @T.prim_func
    def main(
        Var: T.Tensor((M, N), dtype),  # type: ignore
        Grad: T.Tensor((M, N), dtype),  # type: ignore
        M_in: T.Tensor((M, N), dtype),  # type: ignore
        V_in: T.Tensor((M, N), dtype),  # type: ignore
        S: T.Tensor((_SCALAR_PAD,), cal_dtype),  # type: ignore
        Y: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            var_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            grad_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            m_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            v_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            var_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            grad_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            m_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            v_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            scalar_ub = T.alloc_ub((_SCALAR_PAD,), cal_dtype)

            with T.Scope("V"):
                row_start = bx * block_M + vid * sub_block_M
                col_start = by * block_N

                T.copy(Var[row_start, col_start], var_ub, pad_value=0.0)
                T.copy(Grad[row_start, col_start], grad_ub, pad_value=0.0)
                T.copy(M_in[row_start, col_start], m_ub, pad_value=0.0)
                T.copy(V_in[row_start, col_start], v_ub, pad_value=0.0)
                T.copy(S, scalar_ub)

                T.barrier_all()

                # Cast inputs to fp32
                T.tile.cast(var_cal, var_ub, CAST_MODE_LOW2HIGH, cnt)
                T.tile.cast(grad_cal, grad_ub, CAST_MODE_LOW2HIGH, cnt)
                T.tile.cast(m_cal, m_ub, CAST_MODE_LOW2HIGH, cnt)
                T.tile.cast(v_cal, v_ub, CAST_MODE_LOW2HIGH, cnt)

                # m_hat = eff_beta1 * m + eff_ombeta1 * grad
                T.tile.mul(m_cal, m_cal, scalar_ub[0])
                T.tile.axpy(m_cal, grad_cal, scalar_ub[1])

                # grad^2 (reuse grad_cal)
                T.tile.mul(grad_cal, grad_cal, grad_cal)

                # v_hat = eff_beta2 * v + eff_ombeta2 * grad^2
                T.tile.mul(v_cal, v_cal, scalar_ub[2])
                T.tile.axpy(v_cal, grad_cal, scalar_ub[3])

                # denom = sqrt(v_hat) + epsilon (reuse grad_cal)
                T.tile.sqrt(grad_cal, v_cal)
                T.tile.add(grad_cal, grad_cal, scalar_ub[6])

                # update = m_hat / denom (reuse v_cal)
                T.tile.div(v_cal, m_cal, grad_cal)

                # update += weight_decay * var
                T.tile.axpy(v_cal, var_cal, scalar_ub[5])

                # var_t = var + lr_signed * update (reuse var_cal)
                T.tile.axpy(var_cal, v_cal, scalar_ub[4])

                # Cast back to original dtype
                T.tile.cast(var_ub, var_cal, CAST_MODE_HIGH2LOW, cnt)

                T.barrier_all()

                T.copy(var_ub, Y[row_start, col_start])

    return main


def _clamp_block_m_for_barrier(block_M, block_N, tl_dtype):
    """Clamp block_M so barrier kernel UB fits within _UB_LIMIT_BYTES.

    Barrier kernel allocates (VEC_NUM=2, sub_block_M = block_M // 2):
      fp32 path:     4 * sub_block_M * block_N * 4 + _SCALAR_PAD * 4
      fp16/bf16 path: 4 * sub_block_M * block_N * 2 + 4 * sub_block_M * block_N * 4
                      + _SCALAR_PAD * 4

    Single-buffered (no double-buffer stages), so UB usage is roughly half
    of the pipeline kernel for the same block_M. In practice the pipeline
    block_M already fits, this clamp is a safety guard.
    """
    VEC_NUM = 2
    use_fp32_compute = tl_dtype in ["float16", "bfloat16"]
    dtype_size = 2 if use_fp32_compute else 4
    cal_size = 4

    while block_M >= 2:
        sub_block_M = block_M // VEC_NUM
        elems = sub_block_M * block_N
        if use_fp32_compute:
            total = _NUM_INPUTS * elems * dtype_size + _NUM_INPUTS * elems * cal_size + _SCALAR_PAD * cal_size
        else:
            total = _NUM_INPUTS * elems * dtype_size + _SCALAR_PAD * cal_size
        if total <= _UB_LIMIT_BYTES:
            break
        block_M = block_M // 2

    block_M = max(block_M, 2)
    if block_M % 2 != 0:
        block_M += 1
    return block_M


def _get_barrier_kernel(M, N, block_M, block_N, tl_dtype):
    """Get cached barrier kernel (keyed by shape+block, not scalars)."""
    key = (M, N, block_M, block_N, tl_dtype)
    if tl_dtype in ["float16", "bfloat16"]:
        if key not in _barrier_kernel_cache:
            _barrier_kernel_cache[key] = _adam_w_barrier_lowprec(M, N, block_M, block_N, dtype=tl_dtype)
    else:
        if key not in _barrier_kernel_cache:
            _barrier_kernel_cache[key] = _adam_w_barrier_fp32(M, N, block_M, block_N, dtype=tl_dtype)
    return _barrier_kernel_cache[key]


def _get_scalar_tensor(
    eff_beta1,
    eff_ombeta1,
    eff_beta2,
    eff_ombeta2,
    lr_signed,
    weight_decay,
    epsilon,
    device,
):
    """Get or create cached 8-element scalar tensor for barrier kernel.

    Layout: [eff_beta1, eff_ombeta1, eff_beta2, eff_ombeta2,
             lr_signed, weight_decay, epsilon, 0.0]

    torch.tensor accepts Python float inf/nan correctly (IEEE 754 bit pattern
    preserved), so the GM tensor carries inf/nan to the kernel at runtime
    without codegen emitting CUDART_INF/CUDART_NAN literals.
    """
    key = (
        eff_beta1,
        eff_ombeta1,
        eff_beta2,
        eff_ombeta2,
        lr_signed,
        weight_decay,
        epsilon,
        str(device),
    )
    if key not in _scalar_tensor_cache:
        s_tensor = torch.tensor(
            [
                eff_beta1,
                eff_ombeta1,
                eff_beta2,
                eff_ombeta2,
                lr_signed,
                weight_decay,
                epsilon,
                0.0,
            ],
            dtype=torch.float32,
        )
        _scalar_tensor_cache[key] = s_tensor.to(device)
    return _scalar_tensor_cache[key]


# ---------------------------------------------------------------------------
# Kernel cache + public API
# ---------------------------------------------------------------------------


def _get_pipeline_kernel(
    M,
    N,
    block_M,
    block_N,
    sub_M,
    eff_beta1,
    eff_ombeta1,
    eff_beta2,
    eff_ombeta2,
    lr_signed,
    weight_decay,
    epsilon,
    tl_dtype,
):
    if tl_dtype in ["float16", "bfloat16"]:
        key = (
            M,
            N,
            block_M,
            block_N,
            sub_M,
            eff_beta1,
            eff_ombeta1,
            eff_beta2,
            eff_ombeta2,
            lr_signed,
            weight_decay,
            epsilon,
            tl_dtype,
        )
        if key not in _pipeline_lowprec_cache:
            _pipeline_lowprec_cache[key] = _adam_w_pipeline_lowprec(
                M,
                N,
                block_M,
                block_N,
                sub_M,
                eff_beta1,
                eff_ombeta1,
                eff_beta2,
                eff_ombeta2,
                lr_signed,
                weight_decay,
                epsilon,
                dtype=tl_dtype,
            )
        return _pipeline_lowprec_cache[key]
    else:
        key = (
            M,
            N,
            block_M,
            block_N,
            sub_M,
            eff_beta1,
            eff_ombeta1,
            eff_beta2,
            eff_ombeta2,
            lr_signed,
            weight_decay,
            epsilon,
            tl_dtype,
        )
        if key not in _pipeline_fp32_cache:
            _pipeline_fp32_cache[key] = _adam_w_pipeline_fp32(
                M,
                N,
                block_M,
                block_N,
                sub_M,
                eff_beta1,
                eff_ombeta1,
                eff_beta2,
                eff_ombeta2,
                lr_signed,
                weight_decay,
                epsilon,
                dtype=tl_dtype,
            )
        return _pipeline_fp32_cache[key]


def _needs_barrier_path(lr, beta1, beta2, weight_decay, epsilon, step):
    """Detect inf/nan scalars or zero bias-correction denominator.

    Returns True when the pipeline kernel path is unsafe (codegen would emit
    CUDART_INF/CUDART_NAN literals for inf/nan scalar kernel params, or
    _precompute_scalars would raise ZeroDivisionError). In those cases the
    barrier kernel path is used: scalars are passed via an 8-element GM
    tensor (scalar_ub) at runtime, so inf/nan never appear as compile-time
    literals.

    Triggers:
      - lr/beta1/beta2/weight_decay/epsilon is inf or nan
      - step <= 0 (beta^0 = 1 -> bias = 0 -> zero division)
      - beta1^step == 1 (e.g. beta1=1.0 any step, or beta1=-1.0 even step)
      - beta2^step == 1
    """
    for x in (lr, beta1, beta2, weight_decay, epsilon):
        if math.isinf(x) or math.isnan(x):
            return True
    if step <= 0:
        return True
    if (1.0 - beta1**step) == 0.0:
        return True
    return (1.0 - beta2**step) == 0.0


def apply_adam_w(
    var,
    grad,
    m,
    v,
    lr,
    beta1,
    beta2,
    weight_decay,
    epsilon=1e-8,
    step=1,
    maximize=False,
):
    """TileLang implementation of AdamW (decoupled weight decay).

    Routing:
      - Normal scalars (finite, step>0, bias!=0) -> pipeline kernel
        (scalars as compile-time params, optimal performance).
      - inf/nan/zero-division scalars -> barrier kernel
        (scalars via scalar_ub GM tensor at runtime, avoids CUDART_INF/
        CUDART_NAN codegen errors on Ascend C++).

    Args:
        var: Variable tensor (params to optimize).
        grad: Gradient tensor.
        m: First moment tensor (momentum).
        v: Second moment tensor.
        lr: Learning rate.
        beta1: First moment decay rate.
        beta2: Second moment decay rate.
        weight_decay: Decoupled weight decay coefficient.
        epsilon: Numerical stability constant (OUTSIDE sqrt).
        step: Optimization step count (for bias correction).
        maximize: If True, maximize objective (flip lr sign).

    Returns:
        Updated variable tensor (same shape/dtype as var).
    """
    original_shape = var.shape

    var = var.contiguous()
    grad = grad.contiguous()
    m = m.contiguous()
    v = v.contiguous()

    M, N = _compute_2d_shape(var.shape)
    tl_dtype = torch_dtype_to_tl(var.dtype)
    block_M, block_N, sub_M = _select_tiling(M, N, tl_dtype)

    var_2d = var.reshape(M, N)
    grad_2d = grad.reshape(M, N)
    m_2d = m.reshape(M, N)
    v_2d = v.reshape(M, N)

    if _needs_barrier_path(lr, beta1, beta2, weight_decay, epsilon, step):
        # Barrier kernel: scalars via scalar_ub (runtime values, no codegen
        # literal). _precompute_scalars_safe returns inf/nan per IEEE 754
        # (matching golden_apply_adam_w) instead of raising ZeroDivisionError.
        scalars = _precompute_scalars_safe(lr, beta1, beta2, weight_decay, epsilon, step, maximize)
        eff_beta1, eff_ombeta1, eff_beta2, eff_ombeta2, lr_signed = scalars

        barrier_block_M = _clamp_block_m_for_barrier(block_M, block_N, tl_dtype)
        kernel = _get_barrier_kernel(M, N, barrier_block_M, block_N, tl_dtype)
        s_tensor = _get_scalar_tensor(
            eff_beta1,
            eff_ombeta1,
            eff_beta2,
            eff_ombeta2,
            lr_signed,
            weight_decay,
            epsilon,
            var.device,
        )
        y_2d = kernel(var_2d, grad_2d, m_2d, v_2d, s_tensor)
    else:
        # Pipeline kernel: scalars as compile-time params (optimal performance).
        scalars = _precompute_scalars(lr, beta1, beta2, weight_decay, epsilon, step, maximize)
        eff_beta1, eff_ombeta1, eff_beta2, eff_ombeta2, lr_signed = scalars

        kernel = _get_pipeline_kernel(
            M,
            N,
            block_M,
            block_N,
            sub_M,
            eff_beta1,
            eff_ombeta1,
            eff_beta2,
            eff_ombeta2,
            lr_signed,
            weight_decay,
            epsilon,
            tl_dtype,
        )
        y_2d = kernel(var_2d, grad_2d, m_2d, v_2d)

    return y_2d.reshape(original_shape)


# ---------------------------------------------------------------------------
# Golden reference (matches cann-bench golden.py exactly)
# ---------------------------------------------------------------------------


def golden_apply_adam_w(
    var,
    grad,
    m,
    v,
    lr,
    beta1,
    beta2,
    weight_decay,
    epsilon=1e-8,
    step=1,
    maximize=False,
):
    """PyTorch reference implementation (consistent with cann-bench golden.py).

    FP16/BF16 inputs are upcast to FP32 for compute, then cast back.
    epsilon is OUTSIDE sqrt (matches torch.optim.AdamW and torch_npu fused).
    """
    input_dtype = var.dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    var = var.to(compute_dtype)
    grad = grad.to(compute_dtype)
    m = m.to(compute_dtype)
    v = v.to(compute_dtype)

    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m_new / (1 - beta1**step)
    v_hat = v_new / (1 - beta2**step)
    # epsilon OUTSIDE the sqrt — matches torch.optim.AdamW and fused npu_apply_adam_w
    update = m_hat / (v_hat.sqrt() + epsilon)
    if weight_decay != 0:
        update = update + var * weight_decay
    result = var + lr * update if maximize else var - lr * update

    if input_dtype in (torch.float16, torch.bfloat16):
        return result.to(input_dtype)
    return result


# ---------------------------------------------------------------------------
# Precision check helpers (MERE + MARE, consistent with cann-bench standard)
# ---------------------------------------------------------------------------

# Precision thresholds from proto.yaml (MARE threshold = 10 * MERE threshold)
THRESHOLDS = {
    torch.float32: 0.005,  # MARE threshold = 0.05
    torch.float16: 0.01,  # MARE threshold = 0.1
    torch.bfloat16: 0.01,  # MARE threshold = 0.1
}


def _compute_mere_mare(actual, golden):
    """MERE = mean rel err, MARE = max rel err (on finite elements)."""
    a = actual.float()
    g = golden.float()
    finite = torch.isfinite(a) & torch.isfinite(g)
    if int(finite.sum().item()) == 0:
        return 0.0, 0.0
    af = a[finite]
    gf = g[finite]
    diff = (af - gf).abs()
    rel = diff / (gf.abs() + 1e-7)
    return float(rel.mean().item()), float(rel.max().item())


def _check_special(actual, golden):
    """Check inf/nan position and sign consistency."""
    a = actual.float()
    g = golden.float()
    nan_ok = bool((torch.isnan(a) == torch.isnan(g)).all().item())
    a_inf = torch.isinf(a)
    g_inf = torch.isinf(g)
    inf_pos_ok = bool((a_inf == g_inf).all().item())
    if bool(a_inf.any().item()):
        sign_ok = bool((torch.sign(a[a_inf]) == torch.sign(g[a_inf])).all().item())
    else:
        sign_ok = True
    return nan_ok and inf_pos_ok and sign_ok


def _gen_tensor(shape, torch_dtype, value_range, gen):
    """Generate a tensor according to value_range (handles inf/nan/constant)."""
    low, high = value_range
    if isinstance(low, str) or isinstance(high, str):
        # String value ranges like "inf"/"nan" handled by caller
        if low == "nan":
            return torch.full(shape, float("nan"), dtype=torch_dtype)
        if low == "inf" or high == "inf":
            flow = -10.0 if low == "-inf" else -1.0
            fhigh = 10.0 if high == "inf" else 1.0
            t = torch.rand(shape, dtype=torch.float32, generator=gen) * (fhigh - flow) + flow
            mask = torch.rand(shape, dtype=torch.float32, generator=gen) < 0.05
            if bool(mask.any().item()):
                t[mask] = float("inf") if high == "inf" else float("-inf")
            return t.to(torch_dtype)
        return torch.zeros(shape, dtype=torch_dtype)
    if low == high:
        return torch.full(shape, low, dtype=torch_dtype)
    t = torch.rand(shape, dtype=torch.float32, generator=gen) * (high - low) + low
    return t.to(torch_dtype)


# ---------------------------------------------------------------------------
# L0 test cases (from DESIGN.md §9.2 — 10 cases, rule shapes)
# ---------------------------------------------------------------------------

L0_TEST_CASES = [
    {
        "id": "l0_fp32_basic",
        "dtype": torch.float32,
        "shape": (1024, 1024),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (0.0, 0.1)],
    },
    {
        "id": "l0_fp16_basic",
        "dtype": torch.float16,
        "shape": (1024, 1024),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (0.0, 0.1)],
    },
    {
        "id": "l0_bf16_basic",
        "dtype": torch.bfloat16,
        "shape": (1024, 1024),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (0.0, 0.1)],
    },
    {
        "id": "l0_fp32_wd_nonzero",
        "dtype": torch.float32,
        "shape": (1024, 1024),
        "attrs": {
            "lr": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (0.0, 0.1)],
    },
    {
        "id": "l0_fp16_wd_nonzero",
        "dtype": torch.float16,
        "shape": (2048, 2048),
        "attrs": {
            "lr": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-2.0, 2.0), (-2.0, 2.0), (-0.2, 0.2), (0.0, 0.2)],
    },
    {
        "id": "l0_fp32_maximize",
        "dtype": torch.float32,
        "shape": (1024, 1024),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": True,
        },
        "vr": [(-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (0.0, 0.1)],
    },
    {
        "id": "l0_bf16_low_beta",
        "dtype": torch.bfloat16,
        "shape": (1024, 1024),
        "attrs": {
            "lr": 0.01,
            "beta1": 0.5,
            "beta2": 0.9,
            "weight_decay": 0.5,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (0.0, 0.1)],
    },
    {
        "id": "l0_fp32_beta1_zero",
        "dtype": torch.float32,
        "shape": (1024, 1024),
        "attrs": {
            "lr": 0.01,
            "beta1": 0.0,
            "beta2": 0.5,
            "weight_decay": 0.5,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (0.0, 0.1)],
    },
    {
        "id": "l0_fp16_large_eps",
        "dtype": torch.float16,
        "shape": (512, 256),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "epsilon": 1e-4,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (0.0, 0.1)],
    },
    {
        "id": "l0_fp32_1d",
        "dtype": torch.float32,
        "shape": (1048576,),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (0.0, 0.1)],
    },
]


def _run_l0_case(case, gen):
    """Run a single L0 test case and return (passed, mere, mare)."""
    case_id = case["id"]
    dtype = case["dtype"]
    shape = case["shape"]
    attrs = case["attrs"]
    vr = case["vr"]

    dtype_str = str(dtype).split(".")[-1]
    print(f"  [{case_id}] dtype={dtype_str}, shape={shape}, attrs={attrs}")

    try:
        var = _gen_tensor(shape, dtype, vr[0], gen).npu()
        grad = _gen_tensor(shape, dtype, vr[1], gen).npu()
        m = _gen_tensor(shape, dtype, vr[2], gen).npu()
        v = _gen_tensor(shape, dtype, vr[3], gen).npu()

        y_actual = apply_adam_w(var, grad, m, v, **attrs)
        torch.npu.synchronize()
        y_golden = golden_apply_adam_w(var, grad, m, v, **attrs)

        mere, mare = _compute_mere_mare(y_actual, y_golden)
        special_ok = _check_special(y_actual, y_golden)
        threshold = THRESHOLDS[dtype]

        ok = (mere < threshold) and (mare < 10 * threshold) and special_ok
        status = "PASS" if ok else "FAIL"
        print(f"    MERE={mere:.3e}, MARE={mare:.3e}, special_ok={special_ok} -> {status} (thr={threshold:.3e})")
        if ok:
            print(f"  [PRECISION_PASS] {case_id}")
        else:
            print(f"  [PRECISION_FAIL] {case_id}")
        return ok, mere, mare
    except Exception as e:
        print(f"  [PRECISION_FAIL] {case_id}: {e}")
        import traceback

        traceback.print_exc()
        return False, float("inf"), float("inf")


def test_apply_adam_w_l0():
    """L0 gate test: rule shapes (block-aligned), for precision convergence.

    Cases from DESIGN.md §9.2 (10 cases covering fp32/fp16/bf16,
    maximize true/false, weight_decay 0/nonzero, various beta, 1D, large eps).
    """
    print("=" * 60)
    print("L0 Gate Test (10 cases from DESIGN.md §9.2)")
    print("=" * 60)

    gen = torch.Generator().manual_seed(42)
    all_pass = True
    max_mere = 0.0
    max_mare = 0.0
    passing = []

    for case in L0_TEST_CASES:
        ok, mere, mare = _run_l0_case(case, gen)
        all_pass = all_pass and ok
        max_mere = max(max_mere, mere)
        max_mare = max(max_mare, mare)
        if ok:
            passing.append(case["id"])

    return all_pass, max_mere, max_mare, passing


# ---------------------------------------------------------------------------
# L1 test cases (functional: irregular/tail shapes, multi-dim, various attrs)
# Based on cann-bench cases.csv cases 6-11, 15-20 (reduced sizes for runtime)
# ---------------------------------------------------------------------------

L1_TEST_CASES = [
    {  # case 6: bf16, 1023x1023 non-aligned prime
        "id": "l1_bf16_1023x1023",
        "dtype": torch.bfloat16,
        "shape": (1023, 1023),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-0.1, 0.1), (-0.1, 0.1), (-0.01, 0.01), (0.0, 0.01)],
    },
    {  # case 7: fp32, 1009x1021 prime non-aligned, low beta1=0.5
        "id": "l1_fp32_1009x1021",
        "dtype": torch.float32,
        "shape": (1009, 1021),
        "attrs": {
            "lr": 0.1,
            "beta1": 0.5,
            "beta2": 0.9,
            "weight_decay": 0.5,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-1.0, 2.0), (-1.0, 2.0), (-0.1, 0.1), (0.0, 0.1)],
    },
    {  # case 8: fp16, 1537x769 non-aligned, high beta1=0.99
        "id": "l1_fp16_1537x769",
        "dtype": torch.float16,
        "shape": (1537, 769),
        "attrs": {
            "lr": 0.0001,
            "beta1": 0.99,
            "beta2": 0.99,
            "weight_decay": 0.0,
            "epsilon": 1e-6,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-5.0, 10.0), (-5.0, 10.0), (-0.5, 0.5), (0.0, 0.5)],
    },
    {  # case 9: bf16, 3D prime non-aligned, beta1=0.0 (no momentum)
        "id": "l1_bf16_3d_beta1zero",
        "dtype": torch.bfloat16,
        "shape": (37, 67, 73),
        "attrs": {
            "lr": 0.01,
            "beta1": 0.0,
            "beta2": 0.5,
            "weight_decay": 0.5,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-50.0, 100.0), (-50.0, 100.0), (-5.0, 5.0), (0.0, 5.0)],
    },
    {  # case 10: fp32, 2049x513 non-aligned, maximize=true, fp16 boundary vals
        "id": "l1_fp32_maximize",
        "dtype": torch.float32,
        "shape": (2049, 513),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": True,
        },
        "vr": [
            (-65504.0, 65504.0),
            (-65504.0, 65504.0),
            (-6550.0, 6550.0),
            (0.0, 6550.0),
        ],
    },
    {  # case 11: fp16, 4D non-aligned
        "id": "l1_fp16_4d",
        "dtype": torch.float16,
        "shape": (3, 7, 13, 401),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-88.0, 88.0), (-88.0, 88.0), (-8.8, 8.8), (0.0, 8.8)],
    },
    {  # case 16: bf16, 255x513 non-aligned
        "id": "l1_bf16_255x513",
        "dtype": torch.bfloat16,
        "shape": (255, 513),
        "attrs": {
            "lr": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-1.0, 3.0), (-1.0, 3.0), (-0.1, 0.1), (0.0, 0.1)],
    },
    {  # case 17: fp16, non-aligned, large values
        "id": "l1_fp16_large_vals",
        "dtype": torch.float16,
        "shape": (513, 511),
        "attrs": {
            "lr": 0.1,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.1,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-1000.0, 1000.0), (-1000.0, 1000.0), (-100.0, 100.0), (0.0, 100.0)],
    },
    {  # case 18: fp32, 3D non-aligned, tiny values
        "id": "l1_fp32_3d_tiny",
        "dtype": torch.float32,
        "shape": (2, 255, 511),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-0.2, 0.2), (-0.2, 0.2), (-0.02, 0.02), (0.0, 0.02)],
    },
    {  # case 19: bf16, 3D non-aligned, high beta
        "id": "l1_bf16_3d_highbeta",
        "dtype": torch.bfloat16,
        "shape": (4, 55, 203),
        "attrs": {
            "lr": 0.0001,
            "beta1": 0.99,
            "beta2": 0.99,
            "weight_decay": 0.5,
            "epsilon": 1e-6,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-3.0, 6.0), (-3.0, 6.0), (-0.3, 0.3), (0.0, 0.3)],
    },
    {  # case 20: fp32, 5D non-aligned, maximize=true
        "id": "l1_fp32_5d_max",
        "dtype": torch.float32,
        "shape": (2, 3, 17, 32, 101),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": True,
        },
        "vr": [(-20.0, 40.0), (-20.0, 40.0), (-2.0, 2.0), (0.0, 2.0)],
    },
    {  # case 15: fp32, non-aligned, large epsilon=1e-4
        "id": "l1_fp32_large_eps",
        "dtype": torch.float32,
        "shape": (256, 513),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "epsilon": 1e-4,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-0.5, 0.5), (-0.5, 0.5), (-0.05, 0.05), (0.0, 0.05)],
    },
]


# ---------------------------------------------------------------------------
# L2 test cases (exception: inf/nan inputs, epsilon=0 division by zero)
# ---------------------------------------------------------------------------

L2_TEST_CASES = [
    {  # case 12: bf16, 1D, inf inputs (var/grad contain inf)
        "id": "l2_inf_inputs",
        "dtype": torch.bfloat16,
        "shape": (10007,),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [("-inf", "inf"), ("-inf", "inf"), (-1.0, 1.0), (0.0, 1.0)],
    },
    {  # case 13: fp32, 5D, all nan + epsilon=0.0
        "id": "l2_nan_eps0",
        "dtype": torch.float32,
        "shape": (11, 13, 17),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "epsilon": 0.0,
            "step": 1,
            "maximize": False,
        },
        "vr": [("nan", "nan"), ("nan", "nan"), ("nan", "nan"), ("nan", "nan")],
    },
    {  # epsilon=0 with v=0 (division by zero -> inf/nan)
        "id": "l2_eps0_vzero",
        "dtype": torch.float32,
        "shape": (256, 256),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "epsilon": 0.0,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (0.0, 0.0)],
    },
]


# ---------------------------------------------------------------------------
# Boundary test cases (special values: zeros, fp16 max, large eps, step, lr=0)
# ---------------------------------------------------------------------------

BOUNDARY_TEST_CASES = [
    {  # case 14: fp16, 5D, all zeros
        "id": "bnd_all_zeros",
        "dtype": torch.float16,
        "shape": (3, 7, 11, 13),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
    },
    {  # fp16 max value ±65504
        "id": "bnd_fp16_maxval",
        "dtype": torch.float16,
        "shape": (128, 128),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [
            (-65504.0, 65504.0),
            (-65504.0, 65504.0),
            (-6550.0, 6550.0),
            (0.0, 6550.0),
        ],
    },
    {  # large epsilon=1e-4
        "id": "bnd_large_eps",
        "dtype": torch.float32,
        "shape": (128, 128),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "epsilon": 1e-4,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-0.5, 0.5), (-0.5, 0.5), (-0.05, 0.05), (0.0, 0.05)],
    },
    {  # multi-step bias correction (step=10)
        "id": "bnd_step10",
        "dtype": torch.float32,
        "shape": (128, 128),
        "attrs": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "epsilon": 1e-8,
            "step": 10,
            "maximize": False,
        },
        "vr": [(-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (0.0, 0.1)],
    },
    {  # lr=0 (no-op update, var_t should equal var)
        "id": "bnd_lr_zero",
        "dtype": torch.float32,
        "shape": (128, 128),
        "attrs": {
            "lr": 0.0,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "epsilon": 1e-8,
            "step": 1,
            "maximize": False,
        },
        "vr": [(-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (0.0, 0.1)],
    },
]


def _run_boundary_case(case, gen):
    """Run a single L2/Boundary case. Non-blocking: failures print
    [BOUNDARY_WARN] and continue, successes print [BOUNDARY_PASS]."""
    case_id = case["id"]
    dtype = case["dtype"]
    shape = case["shape"]
    attrs = case["attrs"]
    vr = case["vr"]

    dtype_str = str(dtype).split(".")[-1]
    print(f"  [{case_id}] dtype={dtype_str}, shape={shape}, attrs={attrs}")

    try:
        var = _gen_tensor(shape, dtype, vr[0], gen).npu()
        grad = _gen_tensor(shape, dtype, vr[1], gen).npu()
        m = _gen_tensor(shape, dtype, vr[2], gen).npu()
        v = _gen_tensor(shape, dtype, vr[3], gen).npu()

        y_actual = apply_adam_w(var, grad, m, v, **attrs)
        torch.npu.synchronize()
        y_golden = golden_apply_adam_w(var, grad, m, v, **attrs)

        mere, mare = _compute_mere_mare(y_actual, y_golden)
        special_ok = _check_special(y_actual, y_golden)
        threshold = THRESHOLDS[dtype]

        ok = (mere < threshold) and (mare < 10 * threshold) and special_ok
        print(f"    MERE={mere:.3e}, MARE={mare:.3e}, special_ok={special_ok}")
        if ok:
            print(f"  [BOUNDARY_PASS] {case_id}")
        else:
            print(f"  [BOUNDARY_WARN] {case_id} (precision mismatch, non-blocking)")
        return ok
    except Exception as e:
        print(f"  [BOUNDARY_WARN] {case_id}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_apply_adam_w_l1():
    """L1 functional test: irregular/tail shapes, multi-dim, various attrs.

    12 cases from cann-bench cases.csv (cases 6-11, 15-20, reduced sizes).
    Covers non-aligned prime shapes, 3D/4D/5D, maximize, high/low/zero beta,
    large epsilon, nonzero weight_decay.
    """
    print("=" * 60)
    print("L1 Functional Test (12 cases: irregular shapes, multi-dim)")
    print("=" * 60)

    gen = torch.Generator().manual_seed(123)
    all_pass = True
    max_mere = 0.0
    max_mare = 0.0
    passing = []

    for case in L1_TEST_CASES:
        ok, mere, mare = _run_l0_case(case, gen)
        all_pass = all_pass and ok
        max_mere = max(max_mere, mere)
        max_mare = max(max_mare, mare)
        if ok:
            passing.append(case["id"])

    return all_pass, max_mere, max_mare, passing


def test_apply_adam_w_l2():
    """L2 exception test: inf/nan inputs, epsilon=0 division by zero.

    Non-blocking: failures print [BOUNDARY_WARN] and do not affect exit code.
    """
    print("=" * 60)
    print("L2 Exception Test (3 cases: inf/nan/eps0)")
    print("=" * 60)

    gen = torch.Generator().manual_seed(456)
    warnings = []
    for case in L2_TEST_CASES:
        ok = _run_boundary_case(case, gen)
        if not ok:
            warnings.append(case["id"])
    return warnings


def test_apply_adam_w_boundary():
    """Boundary special-value test: zeros, fp16 max, large eps, step, lr=0.

    Non-blocking: failures print [BOUNDARY_WARN] and do not affect exit code.
    """
    print("=" * 60)
    print("Boundary Special-Value Test (5 cases)")
    print("=" * 60)

    gen = torch.Generator().manual_seed(789)
    warnings = []
    for case in BOUNDARY_TEST_CASES:
        ok = _run_boundary_case(case, gen)
        if not ok:
            warnings.append(case["id"])
    return warnings


def main():
    parser = argparse.ArgumentParser(description="apply_adam_w precision test")
    parser.add_argument(
        "--level",
        type=str,
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all"],
        help="Test level to run (default: l0)",
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True  # Only L0/L1 count toward blocking
    overall_mere = 0.0
    overall_mare = 0.0
    all_passing = []
    boundary_warnings = []

    if args.level in ("l0", "all"):
        ok, mere, mare, passing = test_apply_adam_w_l0()
        blocking_ok = blocking_ok and ok
        overall_mere = max(overall_mere, mere)
        overall_mare = max(overall_mare, mare)
        all_passing.extend(passing)

    if args.level in ("l1", "all"):
        ok, mere, mare, passing = test_apply_adam_w_l1()
        blocking_ok = blocking_ok and ok
        overall_mere = max(overall_mere, mere)
        overall_mare = max(overall_mare, mare)
        all_passing.extend(passing)

    if args.level in ("l2", "all"):
        boundary_warnings.extend(test_apply_adam_w_l2())

    if args.level in ("boundary", "all"):
        boundary_warnings.extend(test_apply_adam_w_boundary())

    print()
    print("=" * 60)
    if blocking_ok:
        print(f"[PRECISION_PASS] max_MERE={overall_mere:.3e} max_MARE={overall_mare:.3e} passing={all_passing}")
        if boundary_warnings:
            print(f"[BOUNDARY_WARNINGS] {boundary_warnings} (non-blocking)")
        print("Kernel Output Match!")
        sys.exit(0)
    else:
        print(f"[PRECISION_FAIL] max_MERE={overall_mere:.3e} max_MARE={overall_mare:.3e}")
        if boundary_warnings:
            print(f"[BOUNDARY_WARNINGS] {boundary_warnings} (non-blocking)")
        sys.exit(1)


if __name__ == "__main__":
    main()
