"""ForeachNorm: per-tensor p-norm over a TensorList (Developer mode, multi-core).

Host dispatch routes each tensor to a specialized JIT kernel based on the
scalar (p) value:
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

Optimization (Stage 3 iter 4): batch same-shape tensors into 1 kernel launch.
  - Tensors with the same flattened N are stacked into (batch, N) and processed
    in a single kernel launch (outer T.serial(batch) loop, inner strided tile
    loop). This reduces TileLang launch overhead (185us/launch) from list_len
    to 1, and enables batched host-side finalize (2~5 CANN ops total instead
    of list_len × 2~5).
  - Measured TileLang launch overhead: ~185us (vs CANN op ~44us). Reducing
    launch count is the highest-ROI optimization for multi-tensor cases.
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
# Specialized multi-core partial-reduction kernels (batched).
# Each kernel processes `batch` tensors of the same flattened N in one launch.
# Outer loop: T.serial(batch) — per-tensor accumulator reset.
# Inner loop: T.serial(single_core_load) — strided tile assignment across cores.
# Output Partial: (batch, launch_cores) FP32 — per-core partial per tensor.
# Host combines + finalizes.
# ============================================================================


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def l2_norm_kernel(batch, N, block_N, launch_cores, dtype="float16"):
    """L2 partial: per-core sum(x_i^2). Host finalizes with sqrt(sum)."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            pow_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N], x_ub, pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.mul(pow_ub, x_cal, x_cal)
                        T.reduce_sum(pow_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[t, cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def l1_norm_kernel(batch, N, block_N, launch_cores, dtype="float16"):
    """L1 partial: per-core sum(|x_i|). Host finalizes with sum (identity)."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            abs_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N], x_ub, pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[t, cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def linf_norm_kernel(batch, N, block_N, launch_cores, dtype="float16"):
    """Linf partial: per-core max(|x_i|). Host finalizes with max (identity)."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            abs_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_max_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_ub, -T.infinity(cal_dtype))
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N], x_ub, pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.reduce_max(abs_ub, tile_max_ub, dim=-1)
                        T.tile.max(acc_ub, acc_ub, tile_max_ub)
                T.copy(acc_ub, Partial[t, cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def lneg_inf_norm_kernel(batch, N, block_N, launch_cores, dtype="float16"):
    """L-neg-inf partial: per-core min(|x_i|). Host finalizes with min."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            abs_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_min_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_ub, T.infinity(cal_dtype))
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N], x_ub, pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.reduce_min(abs_ub, tile_min_ub, dim=-1)
                        T.tile.min(acc_ub, acc_ub, tile_min_ub)
                T.copy(acc_ub, Partial[t, cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def l0_count_kernel(batch, N, block_N, launch_cores, dtype="float16"):
    """L0 partial: per-core count of non-zero elements. Host finalizes sum."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            one_ub = T.alloc_shared((block_N,), cal_dtype)
            mask_ub = T.alloc_shared((block_N // 8,), "uint8")
            tile_count_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N], x_ub, pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.fill(one_ub, 1.0)
                        T.tile.compare(mask_ub, x_cal, 0.0, "NE")
                        T.tile.select(one_ub, mask_ub, one_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.reduce_sum(one_ub, tile_count_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_count_ub)
                T.copy(acc_ub, Partial[t, cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def lp_norm_kernel(batch, N, block_N, scalar, launch_cores, dtype="float16"):
    """General p partial: per-core sum(|x_i|^p) = sum(exp(p*ln|x|)).

    Host finalizes with exp(ln(sum)/p). Handles both positive and negative p.
    """
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    if scalar > 0:
        pad_val = 0.0
    else:
        pad_val = T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            abs_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N], x_ub, pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.tile.ln(abs_ub, abs_ub)
                        T.tile.mul(abs_ub, abs_ub, scalar)
                        T.tile.exp(abs_ub, abs_ub)
                        T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[t, cid])

    return main


# ============================================================================
# 1D kernels (batch=1 fast path — avoids 2D T.copy overhead)
# Used when batch=1 or when torch.stack cost exceeds launch saving (large N).
# ============================================================================


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def l2_norm_kernel_1d(N, block_N, launch_cores, dtype="float16"):
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
def l1_norm_kernel_1d(N, block_N, launch_cores, dtype="float16"):
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
def linf_norm_kernel_1d(N, block_N, launch_cores, dtype="float16"):
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
def lneg_inf_norm_kernel_1d(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
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
def l0_count_kernel_1d(N, block_N, launch_cores, dtype="float16"):
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
def lp_norm_kernel_1d(N, block_N, scalar, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
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
                    T.tile.exp(abs_ub, abs_ub)
                    T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                    T.tile.add(acc_ub, acc_ub, tile_sum_ub)
            T.copy(acc_ub, Partial[cid])

    return main


# ============================================================================
# Pipelined kernels (Stage 3 Round 2 Direction 1: conditional T.Pipelined)
#
# Double buffer for large single_core_load >= PIPELINE_THRESHOLD to overlap
# MTE2 load (GM→UB) with V compute (cast/abs/mul/reduce/add).
#
# Parity-split accumulators (acc_a, acc_b) prevent WAW/WAR hazards on the
# cross-iteration accumulator when pipeline stages overlap. After the loop,
# merge: acc_ub = acc_a ⊕ acc_b  (⊕ = add for sum types, max for Linf,
# min for Lneg-inf).
#
# Uses same pass_configs as serial (AUTO_SYNC=True). The T.Pipelined compiler
# pass handles double-buffering of input/work buffers automatically.
# ============================================================================

PIPELINE_THRESHOLD = 20

# Pipelined kernels use AUTO_SYNC=False for manual pipeline synchronization.
# T.barrier_all() after MTE2 load (sync MTE2→V) and before MTE3 store (sync V→MTE3).
# V-queue operations (cast/mul/reduce/add) are serial within V, no explicit sync needed.
pass_configs_pipelined = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# --- 2D pipelined kernels (batched) ---


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def l2_norm_kernel_pipelined(batch, N, block_N, launch_cores, dtype="float16"):
    """L2 pipelined: T.Pipelined(num_stages=2) for single_core_load >= 20."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            pow_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_a, 0.0)
                T.tile.fill(acc_b, 0.0)
                for k in T.Pipelined(single_core_load, num_stages=2):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N], x_ub, pad_value=pad_val)
                        T.barrier_all()
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.mul(pow_ub, x_cal, x_cal)
                        T.reduce_sum(pow_ub, tile_sum_ub, dim=-1)
                        if k % 2 == 0:
                            T.tile.add(acc_a, acc_a, tile_sum_ub)
                        else:
                            T.tile.add(acc_b, acc_b, tile_sum_ub)
                T.tile.add(acc_ub, acc_a, acc_b)
                T.barrier_all()
                T.copy(acc_ub, Partial[t, cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def l1_norm_kernel_pipelined(batch, N, block_N, launch_cores, dtype="float16"):
    """L1 pipelined: T.Pipelined(num_stages=2) for single_core_load >= 20."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            abs_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_a, 0.0)
                T.tile.fill(acc_b, 0.0)
                for k in T.Pipelined(single_core_load, num_stages=2):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N], x_ub, pad_value=pad_val)
                        T.barrier_all()
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                        if k % 2 == 0:
                            T.tile.add(acc_a, acc_a, tile_sum_ub)
                        else:
                            T.tile.add(acc_b, acc_b, tile_sum_ub)
                T.tile.add(acc_ub, acc_a, acc_b)
                T.barrier_all()
                T.copy(acc_ub, Partial[t, cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def linf_norm_kernel_pipelined(batch, N, block_N, launch_cores, dtype="float16"):
    """Linf pipelined: T.Pipelined(num_stages=2) for single_core_load >= 20."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            abs_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_max_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_a, -T.infinity(cal_dtype))
                T.tile.fill(acc_b, -T.infinity(cal_dtype))
                for k in T.Pipelined(single_core_load, num_stages=2):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N], x_ub, pad_value=pad_val)
                        T.barrier_all()
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.reduce_max(abs_ub, tile_max_ub, dim=-1)
                        if k % 2 == 0:
                            T.tile.max(acc_a, acc_a, tile_max_ub)
                        else:
                            T.tile.max(acc_b, acc_b, tile_max_ub)
                T.tile.max(acc_ub, acc_a, acc_b)
                T.barrier_all()
                T.copy(acc_ub, Partial[t, cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def lneg_inf_norm_kernel_pipelined(batch, N, block_N, launch_cores, dtype="float16"):
    """Lneg-inf pipelined: T.Pipelined(num_stages=2) for single_core_load >= 20."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            abs_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_min_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_a, T.infinity(cal_dtype))
                T.tile.fill(acc_b, T.infinity(cal_dtype))
                for k in T.Pipelined(single_core_load, num_stages=2):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N], x_ub, pad_value=pad_val)
                        T.barrier_all()
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.reduce_min(abs_ub, tile_min_ub, dim=-1)
                        if k % 2 == 0:
                            T.tile.min(acc_a, acc_a, tile_min_ub)
                        else:
                            T.tile.min(acc_b, acc_b, tile_min_ub)
                T.tile.min(acc_ub, acc_a, acc_b)
                T.barrier_all()
                T.copy(acc_ub, Partial[t, cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def l0_count_kernel_pipelined(batch, N, block_N, launch_cores, dtype="float16"):
    """L0-count pipelined: T.Pipelined(num_stages=2) for single_core_load >= 20."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            one_ub = T.alloc_shared((block_N,), cal_dtype)
            mask_ub = T.alloc_shared((block_N // 8,), "uint8")
            tile_count_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_a, 0.0)
                T.tile.fill(acc_b, 0.0)
                for k in T.Pipelined(single_core_load, num_stages=2):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N], x_ub, pad_value=pad_val)
                        T.barrier_all()
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.fill(one_ub, 1.0)
                        T.tile.compare(mask_ub, x_cal, 0.0, "NE")
                        T.tile.select(one_ub, mask_ub, one_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.reduce_sum(one_ub, tile_count_ub, dim=-1)
                        if k % 2 == 0:
                            T.tile.add(acc_a, acc_a, tile_count_ub)
                        else:
                            T.tile.add(acc_b, acc_b, tile_count_ub)
                T.tile.add(acc_ub, acc_a, acc_b)
                T.barrier_all()
                T.copy(acc_ub, Partial[t, cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def lp_norm_kernel_pipelined(batch, N, block_N, scalar, launch_cores, dtype="float16"):
    """General p pipelined: T.Pipelined(num_stages=2) for single_core_load >= 20."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    if scalar > 0:
        pad_val = 0.0
    else:
        pad_val = T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            abs_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_a, 0.0)
                T.tile.fill(acc_b, 0.0)
                for k in T.Pipelined(single_core_load, num_stages=2):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N], x_ub, pad_value=pad_val)
                        T.barrier_all()
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.tile.ln(abs_ub, abs_ub)
                        T.tile.mul(abs_ub, abs_ub, scalar)
                        T.tile.exp(abs_ub, abs_ub)
                        T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                        if k % 2 == 0:
                            T.tile.add(acc_a, acc_a, tile_sum_ub)
                        else:
                            T.tile.add(acc_b, acc_b, tile_sum_ub)
                T.tile.add(acc_ub, acc_a, acc_b)
                T.barrier_all()
                T.copy(acc_ub, Partial[t, cid])

    return main


# --- 1D pipelined kernels (batch=1 fast path) ---


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def l2_norm_kernel_1d_pipelined(N, block_N, launch_cores, dtype="float16"):
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
            pow_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_a, 0.0)
            T.tile.fill(acc_b, 0.0)
            for k in T.Pipelined(single_core_load, num_stages=2):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N], x_ub, pad_value=pad_val)
                    T.barrier_all()
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.mul(pow_ub, x_cal, x_cal)
                    T.reduce_sum(pow_ub, tile_sum_ub, dim=-1)
                    if k % 2 == 0:
                        T.tile.add(acc_a, acc_a, tile_sum_ub)
                    else:
                        T.tile.add(acc_b, acc_b, tile_sum_ub)
            T.tile.add(acc_ub, acc_a, acc_b)
            T.barrier_all()
            T.copy(acc_ub, Partial[cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def l1_norm_kernel_1d_pipelined(N, block_N, launch_cores, dtype="float16"):
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
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_a, 0.0)
            T.tile.fill(acc_b, 0.0)
            for k in T.Pipelined(single_core_load, num_stages=2):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N], x_ub, pad_value=pad_val)
                    T.barrier_all()
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.abs(abs_ub, x_cal)
                    T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                    if k % 2 == 0:
                        T.tile.add(acc_a, acc_a, tile_sum_ub)
                    else:
                        T.tile.add(acc_b, acc_b, tile_sum_ub)
            T.tile.add(acc_ub, acc_a, acc_b)
            T.barrier_all()
            T.copy(acc_ub, Partial[cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def linf_norm_kernel_1d_pipelined(N, block_N, launch_cores, dtype="float16"):
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
            tile_max_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_a, -T.infinity(cal_dtype))
            T.tile.fill(acc_b, -T.infinity(cal_dtype))
            for k in T.Pipelined(single_core_load, num_stages=2):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N], x_ub, pad_value=pad_val)
                    T.barrier_all()
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.abs(abs_ub, x_cal)
                    T.reduce_max(abs_ub, tile_max_ub, dim=-1)
                    if k % 2 == 0:
                        T.tile.max(acc_a, acc_a, tile_max_ub)
                    else:
                        T.tile.max(acc_b, acc_b, tile_max_ub)
            T.tile.max(acc_ub, acc_a, acc_b)
            T.barrier_all()
            T.copy(acc_ub, Partial[cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def lneg_inf_norm_kernel_1d_pipelined(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
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
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_a, T.infinity(cal_dtype))
            T.tile.fill(acc_b, T.infinity(cal_dtype))
            for k in T.Pipelined(single_core_load, num_stages=2):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N], x_ub, pad_value=pad_val)
                    T.barrier_all()
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.abs(abs_ub, x_cal)
                    T.reduce_min(abs_ub, tile_min_ub, dim=-1)
                    if k % 2 == 0:
                        T.tile.min(acc_a, acc_a, tile_min_ub)
                    else:
                        T.tile.min(acc_b, acc_b, tile_min_ub)
            T.tile.min(acc_ub, acc_a, acc_b)
            T.barrier_all()
            T.copy(acc_ub, Partial[cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def l0_count_kernel_1d_pipelined(N, block_N, launch_cores, dtype="float16"):
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
            one_ub = T.alloc_shared((block_N,), cal_dtype)
            mask_ub = T.alloc_shared((block_N // 8,), "uint8")
            tile_count_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_a, 0.0)
            T.tile.fill(acc_b, 0.0)
            for k in T.Pipelined(single_core_load, num_stages=2):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N], x_ub, pad_value=pad_val)
                    T.barrier_all()
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.fill(one_ub, 1.0)
                    T.tile.compare(mask_ub, x_cal, 0.0, "NE")
                    T.tile.select(one_ub, mask_ub, one_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                    T.reduce_sum(one_ub, tile_count_ub, dim=-1)
                    if k % 2 == 0:
                        T.tile.add(acc_a, acc_a, tile_count_ub)
                    else:
                        T.tile.add(acc_b, acc_b, tile_count_ub)
            T.tile.add(acc_ub, acc_a, acc_b)
            T.barrier_all()
            T.copy(acc_ub, Partial[cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def lp_norm_kernel_1d_pipelined(N, block_N, scalar, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
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
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_a, 0.0)
            T.tile.fill(acc_b, 0.0)
            for k in T.Pipelined(single_core_load, num_stages=2):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N], x_ub, pad_value=pad_val)
                    T.barrier_all()
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.abs(abs_ub, x_cal)
                    T.tile.ln(abs_ub, abs_ub)
                    T.tile.mul(abs_ub, abs_ub, scalar)
                    T.tile.exp(abs_ub, abs_ub)
                    T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                    if k % 2 == 0:
                        T.tile.add(acc_a, acc_a, tile_sum_ub)
                    else:
                        T.tile.add(acc_b, acc_b, tile_sum_ub)
            T.tile.add(acc_ub, acc_a, acc_b)
            T.barrier_all()
            T.copy(acc_ub, Partial[cid])

    return main


# ============================================================================
# Host dispatch: batched multi-core partial reduction + batched host finalize
# ============================================================================


def _choose_block_n(n: int) -> int:
    """Pick block_N adaptively based on element count."""
    if n >= DEFAULT_BLOCK_N:
        return DEFAULT_BLOCK_N
    bn = max(32, 1 << max(0, (n - 1).bit_length()))
    return min(bn, DEFAULT_BLOCK_N)


SUPPORTED_DTYPES = {"float16", "float32", "bfloat16"}


def _dtype_str(x: torch.Tensor) -> str:
    return str(x.dtype).replace("torch.", "")


_kernel_cache = {}
_kernel_cache_1d = {}
_kernel_cache_pipelined = {}
_kernel_cache_1d_pipelined = {}


def _get_kernel_pipelined(scalar: float, batch: int, n: int, block_n: int, launch_cores: int, dt: str):
    """Get or compile a cached pipelined (2D) kernel for large single_core_load."""
    if scalar == 0.0:
        key = ("l0", batch, n, block_n, launch_cores, dt)
        if key not in _kernel_cache_pipelined:
            _kernel_cache_pipelined[key] = l0_count_kernel_pipelined(batch, n, block_n, launch_cores, dt)
    elif scalar == 1.0:
        key = ("l1", batch, n, block_n, launch_cores, dt)
        if key not in _kernel_cache_pipelined:
            _kernel_cache_pipelined[key] = l1_norm_kernel_pipelined(batch, n, block_n, launch_cores, dt)
    elif scalar == 2.0:
        key = ("l2", batch, n, block_n, launch_cores, dt)
        if key not in _kernel_cache_pipelined:
            _kernel_cache_pipelined[key] = l2_norm_kernel_pipelined(batch, n, block_n, launch_cores, dt)
    elif scalar == float("inf"):
        key = ("linf", batch, n, block_n, launch_cores, dt)
        if key not in _kernel_cache_pipelined:
            _kernel_cache_pipelined[key] = linf_norm_kernel_pipelined(batch, n, block_n, launch_cores, dt)
    elif scalar == float("-inf"):
        key = ("lneg_inf", batch, n, block_n, launch_cores, dt)
        if key not in _kernel_cache_pipelined:
            _kernel_cache_pipelined[key] = lneg_inf_norm_kernel_pipelined(batch, n, block_n, launch_cores, dt)
    else:
        key = ("lp", batch, n, block_n, scalar, launch_cores, dt)
        if key not in _kernel_cache_pipelined:
            _kernel_cache_pipelined[key] = lp_norm_kernel_pipelined(batch, n, block_n, scalar, launch_cores, dt)
    return _kernel_cache_pipelined[key]


def _get_kernel_1d_pipelined(scalar: float, n: int, block_n: int, launch_cores: int, dt: str):
    """Get or compile a cached pipelined 1D kernel for large single_core_load."""
    if scalar == 0.0:
        key = ("l0", n, block_n, launch_cores, dt)
        if key not in _kernel_cache_1d_pipelined:
            _kernel_cache_1d_pipelined[key] = l0_count_kernel_1d_pipelined(n, block_n, launch_cores, dt)
    elif scalar == 1.0:
        key = ("l1", n, block_n, launch_cores, dt)
        if key not in _kernel_cache_1d_pipelined:
            _kernel_cache_1d_pipelined[key] = l1_norm_kernel_1d_pipelined(n, block_n, launch_cores, dt)
    elif scalar == 2.0:
        key = ("l2", n, block_n, launch_cores, dt)
        if key not in _kernel_cache_1d_pipelined:
            _kernel_cache_1d_pipelined[key] = l2_norm_kernel_1d_pipelined(n, block_n, launch_cores, dt)
    elif scalar == float("inf"):
        key = ("linf", n, block_n, launch_cores, dt)
        if key not in _kernel_cache_1d_pipelined:
            _kernel_cache_1d_pipelined[key] = linf_norm_kernel_1d_pipelined(n, block_n, launch_cores, dt)
    elif scalar == float("-inf"):
        key = ("lneg_inf", n, block_n, launch_cores, dt)
        if key not in _kernel_cache_1d_pipelined:
            _kernel_cache_1d_pipelined[key] = lneg_inf_norm_kernel_1d_pipelined(n, block_n, launch_cores, dt)
    else:
        key = ("lp", n, block_n, scalar, launch_cores, dt)
        if key not in _kernel_cache_1d_pipelined:
            _kernel_cache_1d_pipelined[key] = lp_norm_kernel_1d_pipelined(n, block_n, scalar, launch_cores, dt)
    return _kernel_cache_1d_pipelined[key]


def _get_kernel(scalar: float, batch: int, n: int, block_n: int, launch_cores: int, dt: str):
    """Get or compile a cached batched (2D) kernel for the given config.

    Routes to pipelined kernel when single_core_load >= PIPELINE_THRESHOLD
    (large shapes where pipeline steady-state overlap dominates fill/drain).
    """
    n_num = (n + block_n - 1) // block_n
    single_core_load = (n_num + launch_cores - 1) // launch_cores
    if single_core_load >= PIPELINE_THRESHOLD:
        return _get_kernel_pipelined(scalar, batch, n, block_n, launch_cores, dt)
    if scalar == 0.0:
        key = ("l0", batch, n, block_n, launch_cores, dt)
        if key not in _kernel_cache:
            _kernel_cache[key] = l0_count_kernel(batch, n, block_n, launch_cores, dt)
    elif scalar == 1.0:
        key = ("l1", batch, n, block_n, launch_cores, dt)
        if key not in _kernel_cache:
            _kernel_cache[key] = l1_norm_kernel(batch, n, block_n, launch_cores, dt)
    elif scalar == 2.0:
        key = ("l2", batch, n, block_n, launch_cores, dt)
        if key not in _kernel_cache:
            _kernel_cache[key] = l2_norm_kernel(batch, n, block_n, launch_cores, dt)
    elif scalar == float("inf"):
        key = ("linf", batch, n, block_n, launch_cores, dt)
        if key not in _kernel_cache:
            _kernel_cache[key] = linf_norm_kernel(batch, n, block_n, launch_cores, dt)
    elif scalar == float("-inf"):
        key = ("lneg_inf", batch, n, block_n, launch_cores, dt)
        if key not in _kernel_cache:
            _kernel_cache[key] = lneg_inf_norm_kernel(batch, n, block_n, launch_cores, dt)
    else:
        key = ("lp", batch, n, block_n, scalar, launch_cores, dt)
        if key not in _kernel_cache:
            _kernel_cache[key] = lp_norm_kernel(batch, n, block_n, scalar, launch_cores, dt)
    return _kernel_cache[key]


def _get_kernel_1d(scalar: float, n: int, block_n: int, launch_cores: int, dt: str):
    """Get or compile a cached 1D kernel (batch=1 fast path).

    Routes to pipelined kernel when single_core_load >= PIPELINE_THRESHOLD.
    """
    n_num = (n + block_n - 1) // block_n
    single_core_load = (n_num + launch_cores - 1) // launch_cores
    if single_core_load >= PIPELINE_THRESHOLD:
        return _get_kernel_1d_pipelined(scalar, n, block_n, launch_cores, dt)
    if scalar == 0.0:
        key = ("l0", n, block_n, launch_cores, dt)
        if key not in _kernel_cache_1d:
            _kernel_cache_1d[key] = l0_count_kernel_1d(n, block_n, launch_cores, dt)
    elif scalar == 1.0:
        key = ("l1", n, block_n, launch_cores, dt)
        if key not in _kernel_cache_1d:
            _kernel_cache_1d[key] = l1_norm_kernel_1d(n, block_n, launch_cores, dt)
    elif scalar == 2.0:
        key = ("l2", n, block_n, launch_cores, dt)
        if key not in _kernel_cache_1d:
            _kernel_cache_1d[key] = l2_norm_kernel_1d(n, block_n, launch_cores, dt)
    elif scalar == float("inf"):
        key = ("linf", n, block_n, launch_cores, dt)
        if key not in _kernel_cache_1d:
            _kernel_cache_1d[key] = linf_norm_kernel_1d(n, block_n, launch_cores, dt)
    elif scalar == float("-inf"):
        key = ("lneg_inf", n, block_n, launch_cores, dt)
        if key not in _kernel_cache_1d:
            _kernel_cache_1d[key] = lneg_inf_norm_kernel_1d(n, block_n, launch_cores, dt)
    else:
        key = ("lp", n, block_n, scalar, launch_cores, dt)
        if key not in _kernel_cache_1d:
            _kernel_cache_1d[key] = lp_norm_kernel_1d(n, block_n, scalar, launch_cores, dt)
    return _kernel_cache_1d[key]


def _finalize_single(partial: torch.Tensor, scalar: float, out_dtype: torch.dtype) -> torch.Tensor:
    """Combine per-core partials + finalize + cast for a single tensor."""
    if scalar == float("inf"):
        result = partial.max()
    elif scalar == float("-inf"):
        result = partial.min()
    elif scalar == 0.0 or scalar == 1.0:
        result = partial.sum()
    elif scalar == 2.0:
        result = partial.sum().sqrt()
    else:
        s = partial.sum()
        result = torch.pow(s, 1.0 / scalar)
    return result.to(out_dtype)


# TileLang launch overhead measured at ~185us; CANN op ~44us.
# torch.stack cost: 2 * batch * N * dtype_bytes / GM_BANDWIDTH.
# Batching is beneficial when launch saving > stack cost.
_TL_LAUNCH_OVERHEAD_US = 185.0
_GM_BANDWIDTH_BPS = 1.2e12
_DTYPE_BYTES = {"float16": 2, "float32": 4, "bfloat16": 2}


def _should_batch(n: int, batch: int, dt: str) -> bool:
    """Return True if batching saves more than it costs (torch.stack overhead)."""
    if batch <= 1:
        return False
    launch_saving_us = (batch - 1) * _TL_LAUNCH_OVERHEAD_US
    dtype_bytes = _DTYPE_BYTES.get(dt, 4)
    stack_cost_us = 2.0 * batch * n * dtype_bytes / _GM_BANDWIDTH_BPS * 1e6
    return launch_saving_us > stack_cost_us


def _finalize_batched(partial: torch.Tensor, scalar: float, out_dtype: torch.dtype) -> torch.Tensor:
    """Combine per-core FP32 partials + apply finalize + cast.

    Args:
        partial: (batch, launch_cores) FP32 tensor on NPU.
        scalar: norm order p.
        out_dtype: target output dtype (matches input dtype).

    Returns:
        (batch,) tensor on NPU in out_dtype.
    """
    if scalar == float("inf"):
        result = partial.max(dim=1).values
    elif scalar == float("-inf"):
        result = partial.min(dim=1).values
    elif scalar == 0.0 or scalar == 1.0:
        result = partial.sum(dim=1)
    elif scalar == 2.0:
        result = partial.sum(dim=1).sqrt()
    else:
        s = partial.sum(dim=1)
        result = torch.pow(s, 1.0 / scalar)
    return result.to(out_dtype)


def foreach_norm(x: list[torch.Tensor], scalar: float) -> list[torch.Tensor]:
    """Compute p-norm of each tensor in the TensorList (multi-core, batched).

    Tensors with the same flattened N are batched into a single kernel launch
    to reduce TileLang launch overhead (185us/launch).

    Args:
        x: List of input tensors (each ND, same dtype).
        scalar: Norm order p (0, 1, 2, +/-inf, or any real p).

    Returns:
        List of 0-dim scalar tensors (one per input tensor, in input order).

    Raises:
        ValueError: If dtypes are unsupported or inconsistent across the list.
    """
    if not x:
        return []

    first_dt = _dtype_str(x[0])
    if first_dt not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype: {first_dt}. Supported: {sorted(SUPPORTED_DTYPES)}")
    for i, t in enumerate(x[1:], 1):
        dt_i = _dtype_str(t)
        if dt_i != first_dt:
            raise ValueError(f"All tensors must share the same dtype: tensor 0 is {first_dt}, tensor {i} is {dt_i}")

    torch_dt = x[0].dtype

    # Group tensors by flattened N to enable batching
    groups: dict = {}
    for idx, t in enumerate(x):
        n = t.view(-1).shape[0]
        groups.setdefault(n, []).append(idx)

    results: list[torch.Tensor] = [None] * len(x)  # type: ignore

    for n, indices in groups.items():
        batch = len(indices)
        if n == 0:
            for idx in indices:
                results[idx] = torch.zeros((), dtype=torch_dt, device=x[idx].device)
            continue

        block_n = _choose_block_n(n)
        n_num = (n + block_n - 1) // block_n
        launch_cores = min(n_num, CORE_NUM)

        use_batch = _should_batch(n, batch, first_dt)

        if not use_batch:
            # 1D fast path: per-tensor 1D kernels (no 2D overhead, no stack)
            for idx in indices:
                x_flat = x[idx].view(-1)
                kernel = _get_kernel_1d(scalar, n, block_n, launch_cores, first_dt)
                partial = kernel(x_flat)  # (launch_cores,) FP32
                result = _finalize_single(partial, scalar, torch_dt)
                results[idx] = result.view(())
        else:
            # 2D batched: 1 kernel launch for all same-N tensors
            x_batched = torch.stack([x[idx].view(-1) for idx in indices])
            kernel = _get_kernel(scalar, batch, n, block_n, launch_cores, first_dt)
            partial = kernel(x_batched)  # (batch, launch_cores)
            result = _finalize_batched(partial, scalar, torch_dt)  # (batch,)
            for i, idx in enumerate(indices):
                results[idx] = result[i].view(())

    return results
