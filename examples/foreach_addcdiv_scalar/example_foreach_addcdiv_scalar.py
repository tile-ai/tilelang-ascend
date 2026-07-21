"""foreach_addcdiv_scalar operator implementation on Ascend NPU via TileLang.

Formula: y_i = x1_i + (x2_i / x3_i) * scalar

Two kernel variants (Stage 3 iter 1):
  1. Pipeline kernel (finite scalar): MTE2/V/MTE3 three-stage pipeline with
     set_flag/wait_flag and double buffer (stages=2). scalar is a compile-time
     kernel parameter to avoid scalar_ub MTE2->V PipeBarrier deadlock.
  2. Barrier kernel (inf/nan scalar): barrier_all with scalar_ub (8-element
     GM tensor) to avoid CUDART_INF/CUDART_NAN codegen errors.

Both variants:
  - Expert mode: T.alloc_ub, T.Scope("V"), T.tile.div/mul/add
  - FP16/BF16 -> FP32 compute -> cast back (cast_or_copy pattern)
  - pad_value handles non-aligned shapes (x1/x2 pad=0.0, x3 pad=1.0)
  - VEC_NUM=2 row split across two Vector cores

Interface: foreach_addcdiv_scalar(Tensor[] x1, Tensor[] x2, Tensor[] x3, float scalar)
           -> Tensor[] y
"""

import argparse
import math
import sys
from typing import List

import tilelang
import torch
from tilelang import language as T

# Pipeline configs: ONLY MEMORY_PLANNING. AUTO_CV_COMBINE causes pipeline
# deadlock with T.tile ops. AUTO_SYNC default is fine (no scalar_ub in
# pipeline kernel).
_PIPELINE_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Barrier configs: standard Expert mode (AUTO_CV_COMBINE is fine here,
# barrier_all handles all synchronization).
_BARRIER_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"

_CORE_NUM = 24
_SCALAR_PAD = 8

_pipeline_kernel_cache = {}
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
    """Reshape to 2D for optimal tiling.

    For 1D tensors, split into rows of 256 to get M > 1 (enables pipeline
    benefit and better core utilization).
    For multi-dim tensors, merge trailing dims until N >= 256 for better
    memory access patterns (larger contiguous chunks, less padding waste).

    Returns (M, N) where M*N >= total (may exceed for 1D padding).
    """
    total = 1
    for s in shape:
        total *= s

    if len(shape) <= 1:
        # 1D: split into rows of 256 for M > 1
        target_n = 256
        M = (total + target_n - 1) // target_n
        return M, target_n

    # Multi-dim: merge trailing dims until N >= 256
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

    Returns (block_M, block_N, sub_M) where sub_M is the pipeline slice size.
    block_M is always a multiple of sub_M; vec_proc = block_M // sub_M >= 1.
    Pipeline benefits grow with vec_proc (>=2 enables overlap).
    """
    if N <= 128:
        block_N = 128
    else:
        block_N = 256
    block_N = max(block_N, 32)

    if M <= 2:
        block_M = 2
        sub_M = 2
    elif M <= 4:
        block_M = 4
        sub_M = 4
    elif M <= 8:
        block_M = 8
        sub_M = 8
    elif M <= 16:
        block_M = 16
        sub_M = 16
    elif M <= 32:
        block_M = 32
        sub_M = 32
    else:
        target = (M + _CORE_NUM - 1) // _CORE_NUM
        sub_M = 32
        block_M = 128
        for candidate in (128, 256):
            if target <= candidate:
                block_M = candidate
                break
        # Cap block_M at M (rounded up to sub_M) to avoid excessive padding
        max_block_m = ((M + sub_M - 1) // sub_M) * sub_M
        block_M = min(block_M, max_block_m)

    block_M = max(block_M, 2)
    if block_M % 2 != 0:
        block_M += 1

    # Ensure sub_M divides block_M
    sub_M = min(sub_M, block_M)
    if block_M % sub_M != 0:
        block_M = (block_M // sub_M) * sub_M
    block_M = max(block_M, sub_M)
    if block_M % 2 != 0:
        block_M += 1

    return block_M, block_N, sub_M


# Scalar padding for barrier kernel (inf/nan path): 8-element GM tensor
# avoids CUDART_INF/CUDART_NAN codegen issues.


@tilelang.jit(out_idx=[3], pass_configs=_PIPELINE_CONFIGS)
def _addcdiv_kernel_pipeline(M, N, block_M, block_N, sub_M, scalar, dtype="float16"):
    """Pipeline kernel for finite scalar. scalar is a compile-time constant."""
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = sub_M // VEC_NUM
    vec_proc = block_M // sub_M
    stages = 2

    def cast_or_copy(dst, src, mode, count):
        if use_float32_compute:
            return T.tile.cast(dst, src, mode, count)
        else:
            return T.copy(src, dst)

    @T.prim_func
    def main(
        X1: T.Tensor((M, N), dtype),  # type: ignore
        X2: T.Tensor((M, N), dtype),  # type: ignore
        X3: T.Tensor((M, N), dtype),  # type: ignore
        Y: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            x1_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)
            x2_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)
            x3_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)
            x1_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            x2_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            x3_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)

            with T.Scope("V"):
                col_start = by * block_N
                row_base = bx * block_M + vid * sub_block_M
                cnt = sub_block_M * block_N

                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)

                T.wait_flag("mte3", "mte2", 0)
                T.copy(X1[row_base, col_start], x1_ub[0, :, :], pad_value=0.0)
                T.copy(X2[row_base, col_start], x2_ub[0, :, :], pad_value=0.0)
                T.copy(X3[row_base, col_start], x3_ub[0, :, :], pad_value=1.0)
                T.set_flag("mte2", "v", 0)

                for mm in T.serial(vec_proc):
                    cur = mm % stages
                    nxt = (mm + 1) % stages

                    if mm < vec_proc - 1:
                        T.wait_flag("mte3", "mte2", nxt)
                        row_nxt = row_base + (mm + 1) * sub_M
                        T.copy(X1[row_nxt, col_start], x1_ub[nxt, :, :], pad_value=0.0)
                        T.copy(X2[row_nxt, col_start], x2_ub[nxt, :, :], pad_value=0.0)
                        T.copy(X3[row_nxt, col_start], x3_ub[nxt, :, :], pad_value=1.0)
                        T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    if use_float32_compute:
                        # FP16/BF16: cast to FP32, compute, cast back
                        cast_or_copy(x1_cal, x1_ub[cur, :, :], CAST_MODE_LOW2HIGH, cnt)
                        cast_or_copy(x2_cal, x2_ub[cur, :, :], CAST_MODE_LOW2HIGH, cnt)
                        cast_or_copy(x3_cal, x3_ub[cur, :, :], CAST_MODE_LOW2HIGH, cnt)
                        T.tile.div(x2_cal, x2_cal, x3_cal)
                        T.tile.axpy(x1_cal, x2_cal, scalar)
                        cast_or_copy(x1_ub[cur, :, :], x1_cal, CAST_MODE_HIGH2LOW, cnt)
                    else:
                        # FP32: compute directly on staged buffer (no cast needed)
                        T.tile.div(x2_ub[cur, :, :], x2_ub[cur, :, :], x3_ub[cur, :, :])
                        T.tile.axpy(x1_ub[cur, :, :], x2_ub[cur, :, :], scalar)
                    T.set_flag("v", "mte3", cur)

                    T.wait_flag("v", "mte3", cur)
                    row_cur = row_base + mm * sub_M
                    T.copy(x1_ub[cur, :, :], Y[row_cur, col_start])
                    T.set_flag("mte3", "mte2", cur)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return main


@tilelang.jit(out_idx=[4], pass_configs=_BARRIER_CONFIGS)
def _addcdiv_kernel_barrier(M, N, block_M, block_N, dtype="float16"):
    """Barrier kernel for inf/nan scalar. Uses scalar_ub with barrier_all."""
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    def cast_or_copy(dst, src, mode, count):
        if use_float32_compute:
            return T.tile.cast(dst, src, mode, count)
        else:
            return T.copy(src, dst)

    @T.prim_func
    def main(
        X1: T.Tensor((M, N), dtype),  # type: ignore
        X2: T.Tensor((M, N), dtype),  # type: ignore
        X3: T.Tensor((M, N), dtype),  # type: ignore
        S: T.Tensor((_SCALAR_PAD,), cal_dtype),  # type: ignore
        Y: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            x1_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            x2_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            x3_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            x1_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            x2_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            x3_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            scalar_ub = T.alloc_ub((_SCALAR_PAD,), cal_dtype)

            with T.Scope("V"):
                row_start = bx * block_M + vid * sub_block_M
                col_start = by * block_N

                T.copy(X1[row_start, col_start], x1_ub, pad_value=0.0)
                T.copy(X2[row_start, col_start], x2_ub, pad_value=0.0)
                T.copy(X3[row_start, col_start], x3_ub, pad_value=1.0)
                T.copy(S, scalar_ub)

                T.barrier_all()

                if use_float32_compute:
                    # FP16/BF16: cast to FP32, compute, cast back
                    cast_or_copy(x1_cal, x1_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                    cast_or_copy(x2_cal, x2_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                    cast_or_copy(x3_cal, x3_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                    T.tile.div(x2_cal, x2_cal, x3_cal)
                    T.tile.axpy(x1_cal, x2_cal, scalar_ub[0])
                    cast_or_copy(x1_ub, x1_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                else:
                    # FP32: compute directly on staged buffer (no cast needed)
                    T.tile.div(x2_ub, x2_ub, x3_ub)
                    T.tile.axpy(x1_ub, x2_ub, scalar_ub[0])

                T.barrier_all()

                T.copy(x1_ub, Y[row_start, col_start])

    return main


_UB_LIMIT_BYTES = 192 * 1024


def _clamp_block_m_for_barrier(block_M, block_N, tl_dtype):
    """Clamp block_M so barrier kernel UB fits within 192 KB.

    Barrier kernel allocates (VEC_NUM=2, sub_block_M = block_M // 2):
      fp32 path: 3 * sub_block_M * block_N * 4 + 8 * 4
      fp16/bf16 path: 3 * sub_block_M * block_N * 2 + 3 * sub_block_M * block_N * 4 + 8 * 4
    """
    VEC_NUM = 2
    use_fp32_compute = tl_dtype in ["float16", "bfloat16"]
    dtype_size = 2 if use_fp32_compute else 4
    cal_size = 4

    while block_M >= 2:
        sub_block_M = block_M // VEC_NUM
        elems = sub_block_M * block_N
        if use_fp32_compute:
            total = 3 * elems * dtype_size + 3 * elems * cal_size + _SCALAR_PAD * cal_size
        else:
            total = 3 * elems * dtype_size + _SCALAR_PAD * cal_size
        if total <= _UB_LIMIT_BYTES:
            break
        block_M = block_M // 2

    block_M = max(block_M, 2)
    if block_M % 2 != 0:
        block_M += 1
    return block_M


def _get_pipeline_kernel(M, N, block_M, block_N, sub_M, scalar, tl_dtype):
    key = (M, N, block_M, block_N, sub_M, scalar, tl_dtype)
    if key not in _pipeline_kernel_cache:
        _pipeline_kernel_cache[key] = _addcdiv_kernel_pipeline(
            M, N, block_M, block_N, sub_M, scalar, dtype=tl_dtype
        )
    return _pipeline_kernel_cache[key]


def _get_barrier_kernel(M, N, block_M, block_N, tl_dtype):
    key = (M, N, block_M, block_N, tl_dtype)
    if key not in _barrier_kernel_cache:
        _barrier_kernel_cache[key] = _addcdiv_kernel_barrier(
            M, N, block_M, block_N, dtype=tl_dtype
        )
    return _barrier_kernel_cache[key]


def _get_scalar_tensor(scalar, device):
    """Get or create cached scalar tensor for barrier kernel."""
    key = (float(scalar), str(device))
    if key not in _scalar_tensor_cache:
        s_tensor = torch.zeros(_SCALAR_PAD, dtype=torch.float32, device=device)
        s_tensor[0] = float(scalar)
        _scalar_tensor_cache[key] = s_tensor
    return _scalar_tensor_cache[key]


def _process_single_tensor(x1_i, x2_i, x3_i, scalar):
    original_shape = x1_i.shape
    original_numel = x1_i.numel()

    x1_i = x1_i.contiguous()
    x2_i = x2_i.contiguous()
    x3_i = x3_i.contiguous()

    M, N = _compute_2d_shape(x1_i.shape)
    padded_numel = M * N

    tl_dtype = torch_dtype_to_tl(x1_i.dtype)
    block_M, block_N, sub_M = _select_tiling(M, N, tl_dtype)

    if padded_numel > original_numel:
        # 1D tensor needs padding to fit (M, N) reshape
        pad_size = padded_numel - original_numel
        x1_2d = torch.zeros(M, N, dtype=x1_i.dtype, device=x1_i.device)
        x1_2d.view(-1)[:original_numel] = x1_i.view(-1)
        x2_2d = torch.zeros(M, N, dtype=x2_i.dtype, device=x2_i.device)
        x2_2d.view(-1)[:original_numel] = x2_i.view(-1)
        x3_2d = torch.ones(M, N, dtype=x3_i.dtype, device=x3_i.device)
        x3_2d.view(-1)[:original_numel] = x3_i.view(-1)
    else:
        x1_2d = x1_i.reshape(M, N)
        x2_2d = x2_i.reshape(M, N)
        x3_2d = x3_i.reshape(M, N)

    if math.isfinite(scalar):
        # Pipeline kernel: scalar as compile-time parameter
        kernel = _get_pipeline_kernel(
            M, N, block_M, block_N, sub_M, scalar, tl_dtype
        )
        y_2d = kernel(x1_2d, x2_2d, x3_2d)
    else:
        # Barrier kernel: scalar_ub for inf/nan (clamp block_M to fit UB)
        barrier_block_M = _clamp_block_m_for_barrier(block_M, block_N, tl_dtype)
        kernel = _get_barrier_kernel(M, N, barrier_block_M, block_N, tl_dtype)
        s_tensor = _get_scalar_tensor(scalar, x1_i.device)
        y_2d = kernel(x1_2d, x2_2d, x3_2d, s_tensor)

    if padded_numel > original_numel:
        y_i = y_2d.view(-1)[:original_numel].reshape(original_shape)
    else:
        y_i = y_2d.reshape(original_shape)
    return y_i


def foreach_addcdiv_scalar(
    x1: List[torch.Tensor],
    x2: List[torch.Tensor],
    x3: List[torch.Tensor],
    scalar: float,
) -> List[torch.Tensor]:
    """TileLang implementation: y_i = x1_i + (x2_i / x3_i) * scalar."""
    results = []
    for x1_i, x2_i, x3_i in zip(x1, x2, x3):
        y_i = _process_single_tensor(x1_i, x2_i, x3_i, scalar)
        results.append(y_i)
    return results


def golden_foreach_addcdiv_scalar(
    x1: List[torch.Tensor],
    x2: List[torch.Tensor],
    x3: List[torch.Tensor],
    scalar: float,
) -> List[torch.Tensor]:
    """PyTorch reference implementation (consistent with cann-bench golden.py).

    FP16/BF16 inputs are upcast to FP32 for compute, then cast back.
    """
    input_dtype = x1[0].dtype if x1 else torch.float32
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    x1_c = [t.to(compute_dtype) for t in x1]
    x2_c = [t.to(compute_dtype) for t in x2]
    x3_c = [t.to(compute_dtype) for t in x3]

    y = [a + (b / c) * scalar for a, b, c in zip(x1_c, x2_c, x3_c)]

    if input_dtype in (torch.float16, torch.bfloat16):
        return [t.to(input_dtype) for t in y]
    return y


# ---------------------------------------------------------------------------
# Precision check helpers (MERE + MARE, consistent with cann-bench standard)
# ---------------------------------------------------------------------------

THRESHOLDS = {
    torch.float16: 9.77e-4,  # 2^-10
    torch.bfloat16: 7.81e-3,  # 2^-7
    torch.float32: 1.22e-4,  # 2^-13
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
    if math.isnan(low):
        return torch.full(shape, float("nan"), dtype=torch_dtype)
    if low == high:
        return torch.full(shape, low, dtype=torch_dtype)
    flow = low if math.isfinite(low) else -10.0
    fhigh = high if math.isfinite(high) else 10.0
    t = torch.rand(shape, dtype=torch.float32, generator=gen) * (fhigh - flow) + flow
    if not math.isfinite(high) or not math.isfinite(low):
        mask = torch.rand(shape, dtype=torch.float32, generator=gen) < 0.05
        if bool(mask.any().item()):
            t[mask] = float("inf")
    return t.to(torch_dtype)


# ---------------------------------------------------------------------------
# Test cases (from DESIGN §9.2 / cann-bench cases.yaml; large shapes reduced
# to keep runtime manageable while preserving dtype/ndim/alignment/scalar
# coverage)
# ---------------------------------------------------------------------------

INF = float("inf")
NAN = float("nan")

TEST_CASES = [
    # case 1: fp32, 2D aligned, list_len=2, scalar=1.0
    {
        "id": 1,
        "dtype": torch.float32,
        "shapes": [(1024, 1024)] * 2,
        "scalar": 1.0,
        "vr": [(-1.0, 1.0), (-1.0, 1.0), (0.5, 1.0)],
    },
    # case 2: fp16, 2D aligned, list_len=3, scalar=1.0
    {
        "id": 2,
        "dtype": torch.float16,
        "shapes": [(2048, 2048)] * 3,
        "scalar": 1.0,
        "vr": [(-2.0, 2.0), (-2.0, 2.0), (1.0, 2.0)],
    },
    # case 3: bf16, 2D aligned, list_len=1, scalar=1.0 (reduced from 4096)
    {
        "id": 3,
        "dtype": torch.bfloat16,
        "shapes": [(1024, 1024)],
        "scalar": 1.0,
        "vr": [(-3.0, 3.0), (-3.0, 3.0), (0.3, 3.0)],
    },
    # case 6: bf16, 2D non-aligned (1023x1023), scalar=-1.0
    {
        "id": 6,
        "dtype": torch.bfloat16,
        "shapes": [(1023, 1023)],
        "scalar": -1.0,
        "vr": [(-0.1, 0.1), (-0.1, 0.1), (0.1, 0.1)],
    },
    # case 7: fp32, 2D prime non-aligned, scalar=1.5
    {
        "id": 7,
        "dtype": torch.float32,
        "shapes": [(1009, 1021)],
        "scalar": 1.5,
        "vr": [(-1.0, 2.0), (-1.0, 2.0), (1.0, 2.0)],
    },
    # case 8: fp16, 2D non-aligned (1537x769), scalar=1.0
    {
        "id": 8,
        "dtype": torch.float16,
        "shapes": [(1537, 769)],
        "scalar": 1.0,
        "vr": [(-5.0, 10.0), (-5.0, 10.0), (0.1, 10.0)],
    },
    # case 9: bf16, 3D prime non-aligned, list_len=2, scalar=1.0 (reduced)
    {
        "id": 9,
        "dtype": torch.bfloat16,
        "shapes": [(37, 67, 73)] * 2,
        "scalar": 1.0,
        "vr": [(-50.0, 100.0), (-50.0, 100.0), (0.1, 100.0)],
    },
    # case 11: fp16, 4D, list_len=2, scalar=1.0 (reduced last dim)
    {
        "id": 11,
        "dtype": torch.float16,
        "shapes": [(3, 7, 13, 401)] * 2,
        "scalar": 1.0,
        "vr": [(-88.0, 88.0), (-88.0, 88.0), (0.1, 88.0)],
    },
    # case 12: fp32, 1D prime, scalar=inf (reduced from 1000003)
    {
        "id": 12,
        "dtype": torch.float32,
        "shapes": [(10007)],
        "scalar": INF,
        "vr": [(-10.0, 10.0), (-10.0, 10.0), (0.1, 10.0)],
    },
    # case 13: bf16, 5D, scalar=nan (reduced)
    {
        "id": 13,
        "dtype": torch.bfloat16,
        "shapes": [(3, 5, 7, 11, 13)],
        "scalar": NAN,
        "vr": [(-3.0, 3.0), (-3.0, 3.0), (0.3, 3.0)],
    },
    # case 14: fp32, 5D, zero values, scalar=1.0 (reduced last dim)
    {
        "id": 14,
        "dtype": torch.float32,
        "shapes": [(3, 7, 11, 13, 101)],
        "scalar": 1.0,
        "vr": [(0.0, 0.0), (0.0, 0.0), (0.001, 0.001)],
    },
    # case 16: bf16, 2D non-aligned, list_len=4, scalar=1.0 (reduced)
    {
        "id": 16,
        "dtype": torch.bfloat16,
        "shapes": [(255, 513)] * 4,
        "scalar": 1.0,
        "vr": [(-1.0, 3.0), (-1.0, 3.0), (0.1, 3.0)],
    },
    # case 17: fp16, 2D non-aligned, scalar=0.0
    {
        "id": 17,
        "dtype": torch.float16,
        "shapes": [(4097, 511)],
        "scalar": 0.0,
        "vr": [(-1000.0, 1000.0), (-1000.0, 1000.0), (0.1, 1000.0)],
    },
    # case 19: bf16, 3D non-aligned, scalar=-0.5 (reduced)
    {
        "id": 19,
        "dtype": torch.bfloat16,
        "shapes": [(4, 55, 203)],
        "scalar": -0.5,
        "vr": [(-3.0, 6.0), (-3.0, 6.0), (0.1, 6.0)],
    },
    # case 20: fp32, 5D, scalar=1.5 (reduced)
    {
        "id": 20,
        "dtype": torch.float32,
        "shapes": [(2, 3, 17, 64, 101)],
        "scalar": 1.5,
        "vr": [(-20.0, 40.0), (-20.0, 40.0), (20.0, 40.0)],
    },
]


def _run_case(case, gen):
    case_id = case["id"]
    dtype = case["dtype"]
    shapes = case["shapes"]
    scalar = case["scalar"]
    vr = case["vr"]
    list_len = len(shapes)

    scalar_str = (
        "inf" if scalar == INF else ("nan" if math.isnan(scalar) else str(scalar))
    )
    print(
        f"Case {case_id}: dtype={str(dtype).split('.')[-1]}, "
        f"shape={shapes[0]}, list_len={list_len}, scalar={scalar_str}"
    )

    x1 = [_gen_tensor(s, dtype, vr[0], gen).npu() for s in shapes]
    x2 = [_gen_tensor(s, dtype, vr[1], gen).npu() for s in shapes]
    x3 = [_gen_tensor(s, dtype, vr[2], gen).npu() for s in shapes]

    y_actual = foreach_addcdiv_scalar(x1, x2, x3, scalar)
    torch.npu.synchronize()
    y_golden = golden_foreach_addcdiv_scalar(x1, x2, x3, scalar)

    threshold = THRESHOLDS[dtype]
    case_pass = True
    max_mere = 0.0
    max_mare = 0.0
    for i, (a, g) in enumerate(zip(y_actual, y_golden)):
        mere, mare = _compute_mere_mare(a, g)
        special_ok = _check_special(a, g)
        max_mere = max(max_mere, mere)
        max_mare = max(max_mare, mare)
        ok = (mere < threshold) and (mare < 10 * threshold) and special_ok
        if not ok:
            case_pass = False
            print(
                f"  tensor {i}: MERE={mere:.3e}, MARE={mare:.3e}, "
                f"special_ok={special_ok} -> FAIL (thr={threshold:.3e})"
            )
        else:
            print(
                f"  tensor {i}: MERE={mere:.3e}, MARE={mare:.3e}, "
                f"special_ok={special_ok} -> PASS"
            )
    return case_pass, max_mere, max_mare


def main():
    tilelang.disable_cache()
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(42)

    parser = argparse.ArgumentParser(
        description="foreach_addcdiv_scalar precision test"
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Comma-separated case IDs to run (default: all)",
    )
    args = parser.parse_args()

    if args.cases is not None:
        selected_ids = {int(x) for x in args.cases.split(",")}
        cases = [c for c in TEST_CASES if c["id"] in selected_ids]
    else:
        cases = TEST_CASES

    results = []
    for case in cases:
        ok, mere, mare = _run_case(case, gen)
        results.append((case["id"], ok, mere, mare))

    all_pass = all(ok for _, ok, _, _ in results)
    overall_mere = max((m for _, _, m, _ in results), default=0.0)
    overall_mare = max((m for _, _, _, m in results), default=0.0)

    if all_pass:
        passing = [cid for cid, ok, _, _ in results if ok]
        print(
            f"[PRECISION_PASS] max_MERE={overall_mere:.3e} "
            f"max_MARE={overall_mare:.3e} passing_cases={passing}"
        )
        # print("Test Passed!")
        print("\nKernel Output Match!")
    else:
        failing = [(cid, mere) for cid, ok, mere, _ in results if not ok]
        failing_str = [f"case_{c}(MERE={m:.3e})" for c, m in failing]
        print(
            f"[PRECISION_FAIL] max_MERE={overall_mere:.3e} "
            f"max_MARE={overall_mare:.3e} failing={failing_str}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
