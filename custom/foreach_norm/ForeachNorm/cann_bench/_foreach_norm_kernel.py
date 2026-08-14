"""ForeachNorm JIT kernels for cann-bench (Developer mode, multi-core).

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
    of list_len x 2~5).
  - Measured TileLang launch overhead: ~185us (vs CANN op ~44us). Reducing
    launch count is the highest-ROI optimization for multi-tensor cases.

This module contains:
  - 6 batched 2D kernels (batch, N): l2/l1/linf/lneg_inf/l0/lp_norm_kernel
  - 6 1D fast-path kernels (N,):       l2/l1/linf/lneg_inf/l0/lp_norm_kernel_1d

Output Partial tensors are FP32 (per-core partial per tensor); the host
adapter combines + finalizes.
"""

import tilelang
from tilelang import language as T

from ._common import PASS_CONFIGS, CAST_LOW2HIGH, CAST_HIGH2LOW


# ============================================================================
# Specialized multi-core partial-reduction kernels (batched).
# Each kernel processes `batch` tensors of the same flattened N in one launch.
# Outer loop: T.serial(batch) -- per-tensor accumulator reset.
# Inner loop: T.serial(single_core_load) -- strided tile assignment across cores.
# Output Partial: (batch, launch_cores) FP32 -- per-core partial per tensor.
# Host combines + finalizes.
# ============================================================================

@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
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
                        T.copy(X[t, logical_tile * block_N], x_ub,
                               pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.mul(pow_ub, x_cal, x_cal)
                        T.reduce_sum(pow_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[t, cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
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
                        T.copy(X[t, logical_tile * block_N], x_ub,
                               pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[t, cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
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
                        T.copy(X[t, logical_tile * block_N], x_ub,
                               pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.reduce_max(abs_ub, tile_max_ub, dim=-1)
                        T.tile.max(acc_ub, acc_ub, tile_max_ub)
                T.copy(acc_ub, Partial[t, cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
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
                        T.copy(X[t, logical_tile * block_N], x_ub,
                               pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.reduce_min(abs_ub, tile_min_ub, dim=-1)
                        T.tile.min(acc_ub, acc_ub, tile_min_ub)
                T.copy(acc_ub, Partial[t, cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
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
                        T.copy(X[t, logical_tile * block_N], x_ub,
                               pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, block_N)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.fill(one_ub, 1.0)
                        T.tile.compare(mask_ub, x_cal, 0.0, "NE")
                        T.tile.select(one_ub, mask_ub, one_ub, 0.0,
                                      "VSEL_TENSOR_SCALAR_MODE")
                        T.reduce_sum(one_ub, tile_count_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_count_ub)
                T.copy(acc_ub, Partial[t, cid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
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
                        T.copy(X[t, logical_tile * block_N], x_ub,
                               pad_value=pad_val)
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
# 1D kernels (batch=1 fast path -- avoids 2D T.copy overhead)
# Used when batch=1 or when torch.stack cost exceeds launch saving (large N).
# ============================================================================

@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
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


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
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


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
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


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
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


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
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
                    T.tile.select(one_ub, mask_ub, one_ub, 0.0,
                                  "VSEL_TENSOR_SCALAR_MODE")
                    T.reduce_sum(one_ub, tile_count_ub, dim=-1)
                    T.tile.add(acc_ub, acc_ub, tile_count_ub)
            T.copy(acc_ub, Partial[cid])
    return main


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
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
