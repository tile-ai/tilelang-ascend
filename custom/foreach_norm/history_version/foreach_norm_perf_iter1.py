"""ForeachNorm: per-tensor p-norm over a TensorList (Developer mode, multi-core).

Host dispatch routes each tensor to a specialized JIT kernel based on the
scalar (p) value, then combines per-core partial results on host:
  - p == 0      -> L0-count (compare + select + reduce_sum)
  - p == 1      -> L1       (abs + reduce_sum)
  - p == 2      -> L2       (mul + reduce_sum + sqrt)
  - p == +inf   -> Linf     (abs + reduce_max)
  - p == -inf   -> Lneg-inf (abs + reduce_min)
  - p > 0 (other) -> general positive p (exp(p*ln|x|) + reduce_sum)
  - p < 0 (other) -> general negative p (same formula, pad with +inf)

Optimization (Stage 3 iter 1): multi-core parallel partial reduction.
  - launch_cores = min(n_num, CORE_NUM) tiles distributed across physical
    AI Cores (up to 24), each core serially processes its strided tile subset.
  - Each core writes its FP32 partial (sum / max / min of its tiles) to
    Partial[cid] in GM.
  - Host combines partials (sum / max / min) and applies finalize
    (sqrt / exp(ln/p) / identity) + cast back to input dtype.
  - This replaces the single-block T.Kernel(1) + T.serial(n_num) serial
    loop that left 23/24 physical cores idle.
  - Reference: custom/sigmoid/Sigmoid/cann_bench/_sigmoid_kernel.py
    (launch_cores + single_core_load strided assignment pattern).
"""

import tilelang
from tilelang import language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CAST_LOW2HIGH = "CAST_NONE"
CAST_HIGH2LOW = "CAST_RINT"

DEFAULT_BLOCK_N = 8192
CORE_NUM = 24  # Ascend910B3 physical AI Core count


# ============================================================================
# Specialized multi-core partial-reduction kernels
# Each kernel outputs Partial: (launch_cores,) FP32 — per-core partial result.
# Host combines + finalizes.
# ============================================================================


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def l2_norm_kernel(N, block_N, launch_cores, dtype="float16"):
    """L2 partial: per-core sum(x_i^2). Host finalizes with sqrt(sum)."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0  # |0|^2 = 0, no contribution

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores,), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            pow_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            T.tile.fill(acc_ub, 0.0)

            for k in T.serial(single_core_load):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N], x_ub, pad_value=pad_val)
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.mul(pow_ub, x_cal, x_cal)
                    T.reduce_sum(pow_ub, tile_sum_ub, dim=-1)
                    T.tile.add(acc_ub, acc_ub, tile_sum_ub)

            T.copy(acc_ub, Partial[cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def l1_norm_kernel(N, block_N, launch_cores, dtype="float16"):
    """L1 partial: per-core sum(|x_i|). Host finalizes with sum (identity)."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores,), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            abs_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            T.tile.fill(acc_ub, 0.0)

            for k in T.serial(single_core_load):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N], x_ub, pad_value=pad_val)
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.abs(abs_ub, x_cal)
                    T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                    T.tile.add(acc_ub, acc_ub, tile_sum_ub)

            T.copy(acc_ub, Partial[cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def linf_norm_kernel(N, block_N, launch_cores, dtype="float16"):
    """Linf partial: per-core max(|x_i|). Host finalizes with max (identity)."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0  # max(|x|, 0) = max(|x|) since |x| >= 0

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores,), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            abs_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_max_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            T.tile.fill(acc_ub, -T.infinity(cal_dtype))

            for k in T.serial(single_core_load):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N], x_ub, pad_value=pad_val)
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.abs(abs_ub, x_cal)
                    T.reduce_max(abs_ub, tile_max_ub, dim=-1)
                    T.tile.max(acc_ub, acc_ub, tile_max_ub)

            T.copy(acc_ub, Partial[cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def lneg_inf_norm_kernel(N, block_N, launch_cores, dtype="float16"):
    """L-neg-inf partial: per-core min(|x_i|). Host finalizes with min (identity)."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    # Pad with +inf so padded elements are ignored by min
    pad_val = T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores,), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            abs_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_min_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            T.tile.fill(acc_ub, T.infinity(cal_dtype))

            for k in T.serial(single_core_load):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N], x_ub, pad_value=pad_val)
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.abs(abs_ub, x_cal)
                    T.reduce_min(abs_ub, tile_min_ub, dim=-1)
                    T.tile.min(acc_ub, acc_ub, tile_min_ub)

            T.copy(acc_ub, Partial[cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def l0_count_kernel(N, block_N, launch_cores, dtype="float16"):
    """L0 partial: per-core count of non-zero elements. Host finalizes with sum."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0  # padded zeros are not counted

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores,), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            one_ub = T.alloc_shared((block_N,), cal_dtype)
            mask_ub = T.alloc_shared((block_N // 8,), "uint8")
            tile_count_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            T.tile.fill(acc_ub, 0.0)

            for k in T.serial(single_core_load):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N], x_ub, pad_value=pad_val)
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.fill(one_ub, 1.0)
                    T.tile.compare(mask_ub, x_cal, 0.0, "NE")
                    T.tile.select(one_ub, mask_ub, one_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                    T.reduce_sum(one_ub, tile_count_ub, dim=-1)
                    T.tile.add(acc_ub, acc_ub, tile_count_ub)

            T.copy(acc_ub, Partial[cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def lp_norm_kernel(N, block_N, scalar, launch_cores, dtype="float16"):
    """General p partial: per-core sum(|x_i|^p) = sum(exp(p*ln|x|)).

    Host finalizes with exp(ln(sum)/p).
    Handles both positive and negative p.
    """
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    # For p > 0: pad with 0 (|0|^p = 0, no contribution)
    # For p < 0: pad with +inf (|inf|^p = 0 for p < 0, no contribution)
    if scalar > 0:
        pad_val = 0.0
    else:
        pad_val = T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores,), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            abs_ub = T.alloc_shared((block_N,), cal_dtype)
            pow_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            T.tile.fill(acc_ub, 0.0)

            for k in T.serial(single_core_load):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N], x_ub, pad_value=pad_val)
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.abs(abs_ub, x_cal)
                    T.tile.ln(abs_ub, abs_ub)
                    T.tile.mul(abs_ub, abs_ub, scalar)
                    T.tile.exp(pow_ub, abs_ub)
                    T.reduce_sum(pow_ub, tile_sum_ub, dim=-1)
                    T.tile.add(acc_ub, acc_ub, tile_sum_ub)

            T.copy(acc_ub, Partial[cid])

    return main


# ============================================================================
# Host dispatch: multi-core partial reduction + host-side combine + finalize
# ============================================================================


def _choose_block_n(n: int) -> int:
    """Pick block_N adaptively based on element count.

    For n >= 8192 use 8192 (balanced UB usage and tile count).
    For smaller n, round up to a power of two (min 32 for UB alignment).
    """
    if n >= DEFAULT_BLOCK_N:
        return DEFAULT_BLOCK_N
    bn = max(32, 1 << max(0, (n - 1).bit_length()))
    return min(bn, DEFAULT_BLOCK_N)


SUPPORTED_DTYPES = {"float16", "float32", "bfloat16"}


def _dtype_str(x: torch.Tensor) -> str:
    return str(x.dtype).replace("torch.", "")


# In-process kernel cache: (kernel_name, N, block_N, launch_cores, dtype[, scalar])
# Each unique config compiles once; subsequent calls reuse the compiled kernel.
_kernel_cache = {}


def _get_partial_kernel(scalar: float, n: int, block_n: int, launch_cores: int, dt: str):
    """Get or compile a cached partial-reduction kernel for the given config."""
    if scalar == 0.0:
        key = ("l0", n, block_n, launch_cores, dt)
        if key not in _kernel_cache:
            _kernel_cache[key] = l0_count_kernel(n, block_n, launch_cores, dt)
    elif scalar == 1.0:
        key = ("l1", n, block_n, launch_cores, dt)
        if key not in _kernel_cache:
            _kernel_cache[key] = l1_norm_kernel(n, block_n, launch_cores, dt)
    elif scalar == 2.0:
        key = ("l2", n, block_n, launch_cores, dt)
        if key not in _kernel_cache:
            _kernel_cache[key] = l2_norm_kernel(n, block_n, launch_cores, dt)
    elif scalar == float("inf"):
        key = ("linf", n, block_n, launch_cores, dt)
        if key not in _kernel_cache:
            _kernel_cache[key] = linf_norm_kernel(n, block_n, launch_cores, dt)
    elif scalar == float("-inf"):
        key = ("lneg_inf", n, block_n, launch_cores, dt)
        if key not in _kernel_cache:
            _kernel_cache[key] = lneg_inf_norm_kernel(n, block_n, launch_cores, dt)
    else:
        key = ("lp", n, block_n, scalar, launch_cores, dt)
        if key not in _kernel_cache:
            _kernel_cache[key] = lp_norm_kernel(n, block_n, scalar, launch_cores, dt)
    return _kernel_cache[key]


def _finalize(partial: torch.Tensor, scalar: float, out_dtype: torch.dtype) -> torch.Tensor:
    """Combine per-core FP32 partials + apply finalize + cast to out_dtype.

    Args:
        partial: (launch_cores,) FP32 tensor on NPU — per-core partials.
        scalar: norm order p.
        out_dtype: target output dtype (matches input dtype).

    Returns:
        0-dim scalar tensor on NPU in out_dtype.
    """
    if scalar == float("inf"):
        # max of partials, no root
        result = partial.max()
    elif scalar == float("-inf"):
        # min of partials, no root
        result = partial.min()
    elif scalar == 0.0 or scalar == 1.0:
        # sum of partials, no root
        result = partial.sum()
    elif scalar == 2.0:
        # sum then sqrt
        result = partial.sum().sqrt()
    else:
        # general p: exp(ln(sum) / p)
        s = partial.sum()
        result = (s.log() / scalar).exp()
    return result.to(out_dtype)


def foreach_norm(x_list: list[torch.Tensor], scalar: float) -> list[torch.Tensor]:
    """Compute p-norm of each tensor in the TensorList (multi-core parallel).

    Args:
        x_list: List of input tensors (each ND, same dtype).
        scalar: Norm order p (0, 1, 2, +/-inf, or any real p).

    Returns:
        List of 0-dim scalar tensors (one per input tensor).

    Raises:
        ValueError: If dtypes are unsupported or inconsistent across the list.
    """
    if not x_list:
        return []

    # Validate dtype consistency and support
    first_dt = _dtype_str(x_list[0])
    if first_dt not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype: {first_dt}. Supported: {sorted(SUPPORTED_DTYPES)}")
    for i, x in enumerate(x_list[1:], 1):
        dt_i = _dtype_str(x)
        if dt_i != first_dt:
            raise ValueError(f"All tensors must share the same dtype: tensor 0 is {first_dt}, tensor {i} is {dt_i}")

    torch_dt = x_list[0].dtype
    results: list[torch.Tensor] = []
    for x in x_list:
        # Flatten ND to 1D (view — zero-copy for contiguous tensors)
        x_flat = x.view(-1)
        n = x_flat.shape[0]
        if n == 0:
            results.append(torch.zeros((), dtype=torch_dt, device=x.device))
            continue
        block_n = _choose_block_n(n)
        n_num = (n + block_n - 1) // block_n
        # Multi-core: distribute tiles across up to CORE_NUM physical cores.
        # For n_num <= CORE_NUM, one tile per core (launch_cores = n_num).
        # For n_num > CORE_NUM, strided assignment (each core handles
        # ceildiv(n_num, CORE_NUM) tiles).
        launch_cores = min(n_num, CORE_NUM)

        kernel = _get_partial_kernel(scalar, n, block_n, launch_cores, first_dt)
        partial = kernel(x_flat)  # (launch_cores,) FP32 on NPU
        result = _finalize(partial, scalar, torch_dt)
        results.append(result.view(()))

    return results
