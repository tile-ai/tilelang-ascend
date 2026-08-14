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

Optimization (Round 2 Direction 1): conditional T.Pipelined double buffer.
  - For single_core_load >= 20, inner tile loop uses T.Pipelined(num_stages=2)
    with AUTO_SYNC=False + manual T.barrier_all() to overlap MTE2 load (GM→UB)
    with V compute (cast/abs/mul/reduce/add). Parity-split accumulators
    (acc_a/acc_b) prevent WAW/WAR hazards on cross-iteration accumulator.

Optimization (Round 2 Direction 2): VEC_NUM=2 dual vector sub-core.
  - Ascend910B3 default T.Kernel(N, is_npu=True) hardcodes vid extent=2
    (src/ir.cc L259). Previously both vids ran identical code (vid=1 wasted).
    Now each vid processes half_block = block_N // 2 elements in parallel:
    halved buffer sizes, GM read offset by vid*half_block, per-vid Partial
    output (batch, launch_cores, VEC_NUM), host merges via dim=[1,2].
  - Large-shape cases (scl>=20) benefit most: case 9 -23%, case 20 -19%.

Optimization (Best+List): generalized list kernels for L2/Lp.
  - Extended the L1 list kernel pattern (multi-input single-launch, no stack)
    to L2 (l2_norm_kernel_list2/3/4) and general Lp (lp_norm_kernel_list2/3/4).
  - _use_list_kernel replaces _use_l1_list_kernel: routes batch=2/3/4 + scl<20
    to specialized list kernels for L1/L2/Lp, eliminating torch.stack overhead
    on the 5 remaining multi-tensor cases (cann-bench case 11/15/18/19).
  - Linf/Lneg-inf still use _direct_norm (CANN native amax/amin); L0 excluded.
"""

from typing import List

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
VEC_NUM = 2  # Ascend910B3: each AIV core has 2 vector sub-cores (vid=0,1).


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
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            pow_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.mul(pow_ub, x_cal, x_cal)
                        T.reduce_sum(pow_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[t, cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def l1_norm_kernel(batch, N, block_N, launch_cores, dtype="float16"):
    """L1 partial: per-core sum(|x_i|). Host finalizes with sum (identity)."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[t, cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def linf_norm_kernel(batch, N, block_N, launch_cores, dtype="float16"):
    """Linf partial: per-core max(|x_i|). Host finalizes with max (identity)."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_max_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_ub, -T.infinity(cal_dtype))
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.reduce_max(abs_ub, tile_max_ub, dim=-1)
                        T.tile.max(acc_ub, acc_ub, tile_max_ub)
                T.copy(acc_ub, Partial[t, cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def lneg_inf_norm_kernel(batch, N, block_N, launch_cores, dtype="float16"):
    """L-neg-inf partial: per-core min(|x_i|). Host finalizes with min."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_min_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_ub, T.infinity(cal_dtype))
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.reduce_min(abs_ub, tile_min_ub, dim=-1)
                        T.tile.min(acc_ub, acc_ub, tile_min_ub)
                T.copy(acc_ub, Partial[t, cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def l0_count_kernel(batch, N, block_N, launch_cores, dtype="float16"):
    """L0 partial: per-core count of non-zero elements. Host finalizes sum."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            one_ub = T.alloc_shared((half_block,), cal_dtype)
            mask_ub = T.alloc_shared((half_block // 8,), "uint8")
            tile_count_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.fill(one_ub, 1.0)
                        T.tile.compare(mask_ub, x_cal, 0.0, "NE")
                        T.tile.select(one_ub, mask_ub, one_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.reduce_sum(one_ub, tile_count_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_count_ub)
                T.copy(acc_ub, Partial[t, cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def lp_norm_kernel(batch, N, block_N, scalar, launch_cores, dtype="float16"):
    """General p partial: per-core sum(|x_i|^p) = sum(exp(p*ln|x|)).

    Host finalizes with exp(ln(sum)/p). Handles both positive and negative p.
    """
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    if scalar > 0:
        pad_val = 0.0
    else:
        pad_val = T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for t in T.serial(batch):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        T.copy(X[t, logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        if scalar == 3.0:
                            T.tile.mul(x_cal, abs_ub, abs_ub)
                            T.tile.mul(abs_ub, x_cal, abs_ub)
                        elif scalar == 4.0:
                            T.tile.mul(abs_ub, abs_ub, abs_ub)
                            T.tile.mul(abs_ub, abs_ub, abs_ub)
                        elif scalar == 5.0:
                            T.tile.mul(x_cal, abs_ub, abs_ub)
                            T.tile.mul(abs_ub, x_cal, x_cal)
                            T.tile.mul(abs_ub, abs_ub, x_cal)
                        else:
                            T.tile.ln(abs_ub, abs_ub)
                            T.tile.mul(abs_ub, abs_ub, scalar)
                            T.tile.exp(abs_ub, abs_ub)
                        T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[t, cid, vid])

    return main


# ============================================================================
# 1D kernels (batch=1 fast path — avoids 2D T.copy overhead)
# Used when batch=1 or when torch.stack cost exceeds launch saving (large N).
# ============================================================================


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def l2_norm_kernel_1d(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            pow_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_ub, 0.0)
            for k in T.serial(single_core_load):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.mul(pow_ub, x_cal, x_cal)
                    T.reduce_sum(pow_ub, tile_sum_ub, dim=-1)
                    T.tile.add(acc_ub, acc_ub, tile_sum_ub)
            T.copy(acc_ub, Partial[cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def l1_norm_kernel_1d(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_ub, 0.0)
            for k in T.serial(single_core_load):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.abs(abs_ub, x_cal)
                    T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                    T.tile.add(acc_ub, acc_ub, tile_sum_ub)
            T.copy(acc_ub, Partial[cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def linf_norm_kernel_1d(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_max_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_ub, -T.infinity(cal_dtype))
            for k in T.serial(single_core_load):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.abs(abs_ub, x_cal)
                    T.reduce_max(abs_ub, tile_max_ub, dim=-1)
                    T.tile.max(acc_ub, acc_ub, tile_max_ub)
            T.copy(acc_ub, Partial[cid, vid])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def l1_norm_kernel_list2(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype

    @T.prim_func
    def main(
        X0: T.Tensor((N,), dtype),  # type: ignore
        X1: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((2, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for tensor_id in T.serial(2):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        if tensor_id == 0:
                            T.copy(X0[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        else:
                            T.copy(X1[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[tensor_id, cid, vid])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def l1_norm_kernel_list3(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype

    @T.prim_func
    def main(
        X0: T.Tensor((N,), dtype),  # type: ignore
        X1: T.Tensor((N,), dtype),  # type: ignore
        X2: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((3, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for tensor_id in T.serial(3):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        if tensor_id == 0:
                            T.copy(X0[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        elif tensor_id == 1:
                            T.copy(X1[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        else:
                            T.copy(X2[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[tensor_id, cid, vid])

    return main


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def l1_norm_kernel_list4(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype

    @T.prim_func
    def main(
        X0: T.Tensor((N,), dtype),  # type: ignore
        X1: T.Tensor((N,), dtype),  # type: ignore
        X2: T.Tensor((N,), dtype),  # type: ignore
        X3: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((4, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for tensor_id in T.serial(4):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        if tensor_id == 0:
                            T.copy(X0[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        elif tensor_id == 1:
                            T.copy(X1[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        elif tensor_id == 2:
                            T.copy(X2[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        else:
                            T.copy(X3[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[tensor_id, cid, vid])

    return main


# ============================================================================
# L2 list kernels (batch=2/3/4, multi-input single-launch, no torch.stack).
# Same structure as l1_norm_kernel_listN but compute x² (mul) instead of |x|.
# ============================================================================


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def l2_norm_kernel_list2(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype

    @T.prim_func
    def main(
        X0: T.Tensor((N,), dtype),  # type: ignore
        X1: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((2, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            pow_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for tensor_id in T.serial(2):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        if tensor_id == 0:
                            T.copy(X0[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        else:
                            T.copy(X1[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.mul(pow_ub, x_cal, x_cal)
                        T.reduce_sum(pow_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[tensor_id, cid, vid])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def l2_norm_kernel_list3(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype

    @T.prim_func
    def main(
        X0: T.Tensor((N,), dtype),  # type: ignore
        X1: T.Tensor((N,), dtype),  # type: ignore
        X2: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((3, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            pow_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for tensor_id in T.serial(3):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        if tensor_id == 0:
                            T.copy(X0[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        elif tensor_id == 1:
                            T.copy(X1[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        else:
                            T.copy(X2[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.mul(pow_ub, x_cal, x_cal)
                        T.reduce_sum(pow_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[tensor_id, cid, vid])

    return main


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def l2_norm_kernel_list4(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype

    @T.prim_func
    def main(
        X0: T.Tensor((N,), dtype),  # type: ignore
        X1: T.Tensor((N,), dtype),  # type: ignore
        X2: T.Tensor((N,), dtype),  # type: ignore
        X3: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((4, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            pow_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for tensor_id in T.serial(4):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        if tensor_id == 0:
                            T.copy(X0[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        elif tensor_id == 1:
                            T.copy(X1[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        elif tensor_id == 2:
                            T.copy(X2[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        else:
                            T.copy(X3[logical_tile * block_N + vid * half_block], x_ub, pad_value=0.0)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.mul(pow_ub, x_cal, x_cal)
                        T.reduce_sum(pow_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[tensor_id, cid, vid])

    return main


# ============================================================================
# Lp list kernels (batch=2/3/4, general p > 0).
# Same structure as l1_norm_kernel_listN but compute |x|^p via abs + ln + mul + exp
# (or special-cased integer powers for p=3/4/5).
# ============================================================================


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def lp_norm_kernel_list2(N, block_N, scalar, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0 if scalar > 0 else T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X0: T.Tensor((N,), dtype),  # type: ignore
        X1: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((2, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for tensor_id in T.serial(2):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        if tensor_id == 0:
                            T.copy(X0[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        else:
                            T.copy(X1[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        if scalar == 3.0:
                            T.tile.mul(x_cal, abs_ub, abs_ub)
                            T.tile.mul(abs_ub, x_cal, abs_ub)
                        elif scalar == 4.0:
                            T.tile.mul(abs_ub, abs_ub, abs_ub)
                            T.tile.mul(abs_ub, abs_ub, abs_ub)
                        elif scalar == 5.0:
                            T.tile.mul(x_cal, abs_ub, abs_ub)
                            T.tile.mul(abs_ub, x_cal, x_cal)
                            T.tile.mul(abs_ub, abs_ub, x_cal)
                        else:
                            T.tile.ln(abs_ub, abs_ub)
                            T.tile.mul(abs_ub, abs_ub, scalar)
                            T.tile.exp(abs_ub, abs_ub)
                        T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[tensor_id, cid, vid])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def lp_norm_kernel_list3(N, block_N, scalar, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0 if scalar > 0 else T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X0: T.Tensor((N,), dtype),  # type: ignore
        X1: T.Tensor((N,), dtype),  # type: ignore
        X2: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((3, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for tensor_id in T.serial(3):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        if tensor_id == 0:
                            T.copy(X0[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        elif tensor_id == 1:
                            T.copy(X1[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        else:
                            T.copy(X2[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        if scalar == 3.0:
                            T.tile.mul(x_cal, abs_ub, abs_ub)
                            T.tile.mul(abs_ub, x_cal, abs_ub)
                        elif scalar == 4.0:
                            T.tile.mul(abs_ub, abs_ub, abs_ub)
                            T.tile.mul(abs_ub, abs_ub, abs_ub)
                        elif scalar == 5.0:
                            T.tile.mul(x_cal, abs_ub, abs_ub)
                            T.tile.mul(abs_ub, x_cal, x_cal)
                            T.tile.mul(abs_ub, abs_ub, x_cal)
                        else:
                            T.tile.ln(abs_ub, abs_ub)
                            T.tile.mul(abs_ub, abs_ub, scalar)
                            T.tile.exp(abs_ub, abs_ub)
                        T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[tensor_id, cid, vid])

    return main


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def lp_norm_kernel_list4(N, block_N, scalar, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0 if scalar > 0 else T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X0: T.Tensor((N,), dtype),  # type: ignore
        X1: T.Tensor((N,), dtype),  # type: ignore
        X2: T.Tensor((N,), dtype),  # type: ignore
        X3: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((4, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            for tensor_id in T.serial(4):
                T.tile.fill(acc_ub, 0.0)
                for k in T.serial(single_core_load):
                    logical_tile = k * launch_cores + cid
                    if logical_tile < n_num:
                        if tensor_id == 0:
                            T.copy(X0[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        elif tensor_id == 1:
                            T.copy(X1[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        elif tensor_id == 2:
                            T.copy(X2[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        else:
                            T.copy(X3[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        if scalar == 3.0:
                            T.tile.mul(x_cal, abs_ub, abs_ub)
                            T.tile.mul(abs_ub, x_cal, abs_ub)
                        elif scalar == 4.0:
                            T.tile.mul(abs_ub, abs_ub, abs_ub)
                            T.tile.mul(abs_ub, abs_ub, abs_ub)
                        elif scalar == 5.0:
                            T.tile.mul(x_cal, abs_ub, abs_ub)
                            T.tile.mul(abs_ub, x_cal, x_cal)
                            T.tile.mul(abs_ub, abs_ub, x_cal)
                        else:
                            T.tile.ln(abs_ub, abs_ub)
                            T.tile.mul(abs_ub, abs_ub, scalar)
                            T.tile.exp(abs_ub, abs_ub)
                        T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                        T.tile.add(acc_ub, acc_ub, tile_sum_ub)
                T.copy(acc_ub, Partial[tensor_id, cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def lneg_inf_norm_kernel_1d(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_min_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_ub, T.infinity(cal_dtype))
            for k in T.serial(single_core_load):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.abs(abs_ub, x_cal)
                    T.reduce_min(abs_ub, tile_min_ub, dim=-1)
                    T.tile.min(acc_ub, acc_ub, tile_min_ub)
            T.copy(acc_ub, Partial[cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def l0_count_kernel_1d(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            one_ub = T.alloc_shared((half_block,), cal_dtype)
            mask_ub = T.alloc_shared((half_block // 8,), "uint8")
            tile_count_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_ub, 0.0)
            for k in T.serial(single_core_load):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.fill(one_ub, 1.0)
                    T.tile.compare(mask_ub, x_cal, 0.0, "NE")
                    T.tile.select(one_ub, mask_ub, one_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                    T.reduce_sum(one_ub, tile_count_ub, dim=-1)
                    T.tile.add(acc_ub, acc_ub, tile_count_ub)
            T.copy(acc_ub, Partial[cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def lp_norm_kernel_1d(N, block_N, scalar, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    if scalar > 0:
        pad_val = 0.0
    else:
        pad_val = T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_ub, 0.0)
            for k in T.serial(single_core_load):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.abs(abs_ub, x_cal)
                    if scalar == 3.0:
                        T.tile.mul(x_cal, abs_ub, abs_ub)
                        T.tile.mul(abs_ub, x_cal, abs_ub)
                    elif scalar == 4.0:
                        T.tile.mul(abs_ub, abs_ub, abs_ub)
                        T.tile.mul(abs_ub, abs_ub, abs_ub)
                    elif scalar == 5.0:
                        T.tile.mul(x_cal, abs_ub, abs_ub)
                        T.tile.mul(abs_ub, x_cal, x_cal)
                        T.tile.mul(abs_ub, abs_ub, x_cal)
                    else:
                        T.tile.ln(abs_ub, abs_ub)
                        T.tile.mul(abs_ub, abs_ub, scalar)
                        T.tile.exp(abs_ub, abs_ub)
                    T.reduce_sum(abs_ub, tile_sum_ub, dim=-1)
                    T.tile.add(acc_ub, acc_ub, tile_sum_ub)
            T.copy(acc_ub, Partial[cid, vid])

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
# Pipelined kernels use AUTO_SYNC=False with explicit barriers around the
# load/compute/store boundaries.
# ============================================================================

PIPELINE_THRESHOLD = 24

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
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            pow_ub = T.alloc_shared((half_block,), cal_dtype)
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
                        T.copy(X[t, logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        T.barrier_all()
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
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
                T.copy(acc_ub, Partial[t, cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def l1_norm_kernel_pipelined(batch, N, block_N, launch_cores, dtype="float16"):
    """L1 pipelined: T.Pipelined(num_stages=2) for single_core_load >= 20."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
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
                        T.copy(X[t, logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        T.barrier_all()
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
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
                T.copy(acc_ub, Partial[t, cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def linf_norm_kernel_pipelined(batch, N, block_N, launch_cores, dtype="float16"):
    """Linf pipelined: T.Pipelined(num_stages=2) for single_core_load >= 20."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
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
                        T.copy(X[t, logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        T.barrier_all()
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
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
                T.copy(acc_ub, Partial[t, cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def lneg_inf_norm_kernel_pipelined(batch, N, block_N, launch_cores, dtype="float16"):
    """Lneg-inf pipelined: T.Pipelined(num_stages=2) for single_core_load >= 20."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
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
                        T.copy(X[t, logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        T.barrier_all()
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
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
                T.copy(acc_ub, Partial[t, cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def l0_count_kernel_pipelined(batch, N, block_N, launch_cores, dtype="float16"):
    """L0-count pipelined: T.Pipelined(num_stages=2) for single_core_load >= 20."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            one_ub = T.alloc_shared((half_block,), cal_dtype)
            mask_ub = T.alloc_shared((half_block // 8,), "uint8")
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
                        T.copy(X[t, logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        T.barrier_all()
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
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
                T.copy(acc_ub, Partial[t, cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def lp_norm_kernel_pipelined(batch, N, block_N, scalar, launch_cores, dtype="float16"):
    """General p pipelined: T.Pipelined(num_stages=2) for single_core_load >= 20."""
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    if scalar > 0:
        pad_val = 0.0
    else:
        pad_val = T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X: T.Tensor((batch, N), dtype),  # type: ignore
        Partial: T.Tensor((batch, launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
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
                        T.copy(X[t, logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                        T.barrier_all()
                        if use_upcast:
                            T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                        else:
                            T.copy(x_ub, x_cal)
                        T.tile.abs(abs_ub, x_cal)
                        if scalar == 3.0:
                            T.tile.mul(x_cal, abs_ub, abs_ub)
                            T.tile.mul(abs_ub, x_cal, abs_ub)
                        elif scalar == 4.0:
                            T.tile.mul(abs_ub, abs_ub, abs_ub)
                            T.tile.mul(abs_ub, abs_ub, abs_ub)
                        elif scalar == 5.0:
                            T.tile.mul(x_cal, abs_ub, abs_ub)
                            T.tile.mul(abs_ub, x_cal, x_cal)
                            T.tile.mul(abs_ub, abs_ub, x_cal)
                        else:
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
                T.copy(acc_ub, Partial[t, cid, vid])

    return main


# --- 1D pipelined kernels (batch=1 fast path) ---


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def l2_norm_kernel_1d_pipelined(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            pow_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_a, 0.0)
            T.tile.fill(acc_b, 0.0)
            for k in T.Pipelined(single_core_load, num_stages=2):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                    T.barrier_all()
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
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
            T.copy(acc_ub, Partial[cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def l1_norm_kernel_1d_pipelined(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_a, 0.0)
            T.tile.fill(acc_b, 0.0)
            for k in T.Pipelined(single_core_load, num_stages=2):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                    T.barrier_all()
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
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
            T.copy(acc_ub, Partial[cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def linf_norm_kernel_1d_pipelined(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_max_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_a, -T.infinity(cal_dtype))
            T.tile.fill(acc_b, -T.infinity(cal_dtype))
            for k in T.Pipelined(single_core_load, num_stages=2):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                    T.barrier_all()
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
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
            T.copy(acc_ub, Partial[cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def lneg_inf_norm_kernel_1d_pipelined(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_min_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_a, T.infinity(cal_dtype))
            T.tile.fill(acc_b, T.infinity(cal_dtype))
            for k in T.Pipelined(single_core_load, num_stages=2):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                    T.barrier_all()
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
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
            T.copy(acc_ub, Partial[cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def l0_count_kernel_1d_pipelined(N, block_N, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    pad_val = 0.0

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            one_ub = T.alloc_shared((half_block,), cal_dtype)
            mask_ub = T.alloc_shared((half_block // 8,), "uint8")
            tile_count_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_a, 0.0)
            T.tile.fill(acc_b, 0.0)
            for k in T.Pipelined(single_core_load, num_stages=2):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                    T.barrier_all()
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
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
            T.copy(acc_ub, Partial[cid, vid])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs_pipelined)
def lp_norm_kernel_1d_pipelined(N, block_N, scalar, launch_cores, dtype="float16"):
    n_num = T.ceildiv(N, block_N)
    single_core_load = T.ceildiv(n_num, launch_cores)
    half_block = block_N // VEC_NUM
    use_upcast = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_upcast else dtype
    if scalar > 0:
        pad_val = 0.0
    else:
        pad_val = T.infinity(cal_dtype)

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),  # type: ignore
        Partial: T.Tensor((launch_cores, VEC_NUM), cal_dtype),  # type: ignore
    ):
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            x_ub = T.alloc_shared((half_block,), dtype)
            x_cal = T.alloc_shared((half_block,), cal_dtype)
            abs_ub = T.alloc_shared((half_block,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_a = T.alloc_shared((1,), cal_dtype)
            acc_b = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)
            T.tile.fill(acc_a, 0.0)
            T.tile.fill(acc_b, 0.0)
            for k in T.Pipelined(single_core_load, num_stages=2):
                logical_tile = k * launch_cores + cid
                if logical_tile < n_num:
                    T.copy(X[logical_tile * block_N + vid * half_block], x_ub, pad_value=pad_val)
                    T.barrier_all()
                    if use_upcast:
                        T.tile.cast(x_cal, x_ub, CAST_LOW2HIGH, half_block)
                    else:
                        T.copy(x_ub, x_cal)
                    T.tile.abs(abs_ub, x_cal)
                    if scalar == 3.0:
                        T.tile.mul(x_cal, abs_ub, abs_ub)
                        T.tile.mul(abs_ub, x_cal, abs_ub)
                    elif scalar == 4.0:
                        T.tile.mul(abs_ub, abs_ub, abs_ub)
                        T.tile.mul(abs_ub, abs_ub, abs_ub)
                    elif scalar == 5.0:
                        T.tile.mul(x_cal, abs_ub, abs_ub)
                        T.tile.mul(abs_ub, x_cal, x_cal)
                        T.tile.mul(abs_ub, abs_ub, x_cal)
                    else:
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
            T.copy(acc_ub, Partial[cid, vid])

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


def _direct_norm(t: torch.Tensor, scalar: float, out_dtype: torch.dtype) -> torch.Tensor:
    """Use simple CANN reductions for norm orders with cheaper native ops."""
    x_abs = torch.abs(t.view(-1).to(torch.float32))
    if scalar == float("inf"):
        return torch.amax(x_abs, dim=0).to(out_dtype).view(())
    if scalar == float("-inf"):
        return torch.amin(x_abs, dim=0).to(out_dtype).view(())
    raise ValueError(f"Unsupported direct norm scalar: {scalar}")


def _use_direct_norm(scalar: float, n: int, dt: str) -> bool:
    """Return True when a scalar norm maps to one cheap built-in reduction."""
    return scalar == float("inf") or scalar == float("-inf")


_kernel_cache = {}
_kernel_cache_1d = {}
_kernel_cache_pipelined = {}
_kernel_cache_1d_pipelined = {}
_kernel_cache_l1_list = {}
_kernel_cache_l2_list = {}
_kernel_cache_lp_list = {}


def _get_l1_list_kernel(batch: int, n: int, block_n: int, launch_cores: int, dt: str):
    key = ("l1_list", batch, n, block_n, launch_cores, dt)
    if key not in _kernel_cache_l1_list:
        if batch == 2:
            _kernel_cache_l1_list[key] = l1_norm_kernel_list2(n, block_n, launch_cores, dt)
        elif batch == 3:
            _kernel_cache_l1_list[key] = l1_norm_kernel_list3(n, block_n, launch_cores, dt)
        elif batch == 4:
            _kernel_cache_l1_list[key] = l1_norm_kernel_list4(n, block_n, launch_cores, dt)
        else:
            raise ValueError(f"Unsupported L1 list batch: {batch}")
    return _kernel_cache_l1_list[key]


def _get_l2_list_kernel(batch: int, n: int, block_n: int, launch_cores: int, dt: str):
    key = ("l2_list", batch, n, block_n, launch_cores, dt)
    if key not in _kernel_cache_l2_list:
        if batch == 2:
            _kernel_cache_l2_list[key] = l2_norm_kernel_list2(n, block_n, launch_cores, dt)
        elif batch == 3:
            _kernel_cache_l2_list[key] = l2_norm_kernel_list3(n, block_n, launch_cores, dt)
        elif batch == 4:
            _kernel_cache_l2_list[key] = l2_norm_kernel_list4(n, block_n, launch_cores, dt)
        else:
            raise ValueError(f"Unsupported L2 list batch: {batch}")
    return _kernel_cache_l2_list[key]


def _get_lp_list_kernel(scalar: float, batch: int, n: int, block_n: int, launch_cores: int, dt: str):
    key = ("lp_list", scalar, batch, n, block_n, launch_cores, dt)
    if key not in _kernel_cache_lp_list:
        if batch == 2:
            _kernel_cache_lp_list[key] = lp_norm_kernel_list2(n, block_n, scalar, launch_cores, dt)
        elif batch == 3:
            _kernel_cache_lp_list[key] = lp_norm_kernel_list3(n, block_n, scalar, launch_cores, dt)
        elif batch == 4:
            _kernel_cache_lp_list[key] = lp_norm_kernel_list4(n, block_n, scalar, launch_cores, dt)
        else:
            raise ValueError(f"Unsupported Lp list batch: {batch}")
    return _kernel_cache_lp_list[key]


def _get_list_kernel(scalar: float, batch: int, n: int, block_n: int, launch_cores: int, dt: str):
    """Generalized list-kernel dispatcher (L1/L2/Lp)."""
    if scalar == 1.0:
        return _get_l1_list_kernel(batch, n, block_n, launch_cores, dt)
    elif scalar == 2.0:
        return _get_l2_list_kernel(batch, n, block_n, launch_cores, dt)
    else:
        return _get_lp_list_kernel(scalar, batch, n, block_n, launch_cores, dt)


def _use_list_kernel(scalar: float, batch: int, single_core_load: int) -> bool:
    """Whether to use a list kernel (eliminates torch.stack overhead).

    Applies to L1/L2/Lp norms with batch in {2,3,4} and small single_core_load
    (where stack overhead dominates kernel time). Linf/Lneg-inf use _direct_norm
    instead; L0 is rare and excluded.
    """
    if batch not in (2, 3, 4):
        return False
    if single_core_load >= PIPELINE_THRESHOLD:
        return False
    # L0/Linf/Lneg-inf don't need list kernels
    # L1 (scalar==1.0), L2 (scalar==2.0), and general Lp (scalar>0 or scalar<0)
    return scalar not in (0.0, float("inf"), float("-inf"))


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


def _should_batch(n: int, batch: int, dt: str) -> bool:
    """Return True when same-shape inputs should be stacked into a 2D batch.

    In CANN-Bench the host-side torch.stack materializes as aclnnStack_Pack,
    which is often slower than the TileLang launch it saves for ForeachNorm's
    official TensorList cases. Prefer the 1D fast path unless a future case
    proves stack is cheap enough on the target harness.
    """
    return False


def _finalize_batched(partial: torch.Tensor, scalar: float, out_dtype: torch.dtype) -> torch.Tensor:
    """Combine per-core FP32 partials + apply finalize + cast.

    Args:
        partial: (batch, launch_cores, VEC_NUM) FP32 tensor on NPU.
        scalar: norm order p.
        out_dtype: target output dtype (matches input dtype).

    Returns:
        (batch,) tensor on NPU in out_dtype.
    """
    # Combine both launch_cores and VEC_NUM dims (vid partials).
    reduce_dims = (1, 2)
    if scalar == float("inf"):
        result = torch.amax(partial, dim=reduce_dims)
    elif scalar == float("-inf"):
        result = torch.amin(partial, dim=reduce_dims)
    elif scalar == 0.0 or scalar == 1.0:
        result = partial.sum(dim=reduce_dims)
    elif scalar == 2.0:
        result = partial.sum(dim=reduce_dims).sqrt()
    else:
        s = partial.sum(dim=reduce_dims)
        result = torch.pow(s, 1.0 / scalar)
    return result.to(out_dtype)


def foreach_norm(x: List[torch.Tensor], scalar: float) -> List[torch.Tensor]:
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

    results: List[torch.Tensor] = [None] * len(x)  # type: ignore

    for n, indices in groups.items():
        batch = len(indices)
        if n == 0:
            for idx in indices:
                results[idx] = torch.zeros((), dtype=torch_dt, device=x[idx].device)
            continue

        if _use_direct_norm(scalar, n, first_dt):
            for idx in indices:
                results[idx] = _direct_norm(x[idx], scalar, torch_dt)
            continue

        block_n = _choose_block_n(n)
        n_num = (n + block_n - 1) // block_n
        launch_cores = min(n_num, CORE_NUM)
        single_core_load = (n_num + launch_cores - 1) // launch_cores

        if _use_list_kernel(scalar, batch, single_core_load):
            flats = [x[idx].view(-1) for idx in indices]
            kernel = _get_list_kernel(scalar, batch, n, block_n, launch_cores, first_dt)
            partial = kernel(*flats)
            result = _finalize_batched(partial, scalar, torch_dt)
            for i, idx in enumerate(indices):
                results[idx] = result[i].view(())
            continue

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


def main():
    """Simple self-test for foreach_norm operator."""
    import torch

    # Test case 1: L2 norm (batch=1, single tensor)
    x1 = torch.randn(1024, dtype=torch.float16, device="cpu").npu()
    result1 = foreach_norm([x1], scalar=2.0)
    expected1 = torch.norm(x1.cpu(), p=2.0).npu()
    assert len(result1) == 1, f"Expected 1 result, got {len(result1)}"
    assert torch.allclose(result1[0], expected1, rtol=1e-2, atol=1e-3), f"L2 norm mismatch: {result1[0].item()} vs {expected1.item()}"

    # Test case 2: L1 norm (batch=2, different tensors)
    x2 = [torch.randn(512, dtype=torch.float16, device="cpu").npu() for _ in range(2)]
    result2 = foreach_norm(x2, scalar=1.0)
    expected2 = [torch.norm(t.cpu(), p=1.0).npu() for t in x2]
    assert len(result2) == 2, f"Expected 2 results, got {len(result2)}"
    for r, e in zip(result2, expected2):
        assert torch.allclose(r, e, rtol=1e-2, atol=1e-3), f"L1 norm mismatch: {r.item()} vs {e.item()}"

    # Test case 3: Inf norm (batch=1)
    x3 = torch.randn(2048, dtype=torch.float32, device="cpu").npu()
    result3 = foreach_norm([x3], scalar=float("inf"))
    expected3 = torch.norm(x3.cpu(), p=float("inf")).npu()
    assert len(result3) == 1, f"Expected 1 result, got {len(result3)}"
    assert torch.allclose(result3[0], expected3, rtol=1e-3, atol=1e-4), f"Inf norm mismatch: {result3[0].item()} vs {expected3.item()}"

    print("KERNEL OUTPUT MATCH")
    print("TEST PASSED!")


if __name__ == "__main__":
    main()
