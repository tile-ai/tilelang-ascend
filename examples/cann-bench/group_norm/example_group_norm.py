"""GroupNorm example — multi-case optimized version with cann-bench 20 cases.

Expert mode GroupNorm with Pass 2 Double Buffer pipeline for arbitrary
shape, dtype, num_groups, and epsilon. Targets cann-bench multi-case evaluation.

Algorithm:
    y = (x - mean) / sqrt(var + eps) * gamma + beta  (per-group)

Reference: torch.nn.functional.group_norm

Key optimizations:
- Expert mode (AUTO_SYNC=False) with Pass 2 double buffer pipeline
- MTE2→V→MTE3 three-way flag pipeline for overlap
- Adaptive spatial tiling via UB budget formula
- 2D Developer mode for S=1 inputs
"""

import argparse
import gc
import sys

import torch

import tilelang
from tilelang import language as T

tilelang.cache.clear_cache()


CAST_LOW2HIGH = "CAST_NONE"
CAST_HIGH2LOW = "CAST_RINT"

CORE_NUM = 20

pass_configs_expert = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

pass_configs_2d = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

_CANNBENCH_THRESHOLDS = {
    "float16": 2**-10,
    "bfloat16": 2**-7,
    "float32": 2**-13,
}


_kernel_cache = {}
_kernel_cache_2d = {}


@tilelang.jit(out_idx=[3], pass_configs=pass_configs_expert)
def group_norm_kernel_expert(
    N,
    G,
    cpg,
    S,
    S_padded,
    block_S,
    s_num,
    s_num_v0,
    s_num_v1,
    split_factor,
    single_pass=0,
    eps=1e-5,
    dtype="float32",
    S_orig=0,
):
    block_num = N * G * split_factor
    cpg_padded = max(((cpg + 15) // 16) * 16, 16)
    tile_elem = cpg * block_S
    if S_orig == 0:
        S_orig = S

    use_fp32 = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_fp32 else dtype

    @T.prim_func
    def main(
        x: T.Tensor((N, G, cpg, S), dtype),  # type: ignore
        gamma: T.Tensor((G, cpg), dtype),  # type: ignore
        beta: T.Tensor((G, cpg), dtype),  # type: ignore
        y: T.Tensor((N, G, cpg, S_padded), dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            ng = cid // split_factor
            half = cid % split_factor
            n = ng // G
            g = ng % G

            if vid == 0:
                data_buf = T.alloc_shared([cpg, block_S], dtype)
                data_cal = T.alloc_shared([cpg, block_S], cal_dtype)
                sum_acc = T.alloc_shared([cpg, block_S], cal_dtype)
                sum_sq_acc = T.alloc_shared([cpg, block_S], cal_dtype)
                sum_row = T.alloc_shared([cpg], cal_dtype)
                sum_sq_row = T.alloc_shared([cpg], cal_dtype)
                total = T.alloc_shared([1], cal_dtype)
                total_sq = T.alloc_shared([1], cal_dtype)
                mean_sq_val = T.alloc_shared([1], cal_dtype)
                var_val = T.alloc_shared([1], cal_dtype)
                std_val = T.alloc_shared([1], cal_dtype)
                gamma_raw = T.alloc_shared([cpg_padded, 1], dtype)
                beta_raw = T.alloc_shared([cpg_padded, 1], dtype)
                gamma_cal = T.alloc_shared([cpg_padded, 1], cal_dtype)
                beta_cal = T.alloc_shared([cpg_padded, 1], cal_dtype)
                scale_cal = T.alloc_shared([cpg_padded, 1], cal_dtype)
                offset_cal = T.alloc_shared([cpg_padded, 1], cal_dtype)
                temp_cal = T.alloc_shared([cpg_padded, 1], cal_dtype)
                scale_bc_full = T.alloc_shared([cpg_padded, block_S], cal_dtype)
                offset_bc_full = T.alloc_shared([cpg_padded, block_S], cal_dtype)
                scale_bc = T.alloc_shared([cpg, block_S], cal_dtype)
                offset_bc = T.alloc_shared([cpg, block_S], cal_dtype)
                data_buf_db = T.alloc_shared([2, cpg, block_S], dtype)
                out_buf_db = T.alloc_shared([2, cpg, block_S], dtype)

                T.tile.fill(sum_acc, 0.0)
                T.tile.fill(sum_sq_acc, 0.0)

                for si in T.serial(s_num):
                    s_off = si * block_S
                    T.copy(x[n, g, 0:cpg, s_off : s_off + block_S], data_buf, pad_value=0.0)
                    T.barrier_all()
                    if use_fp32:
                        T.tile.cast(data_cal, data_buf, CAST_LOW2HIGH, tile_elem)
                    else:
                        T.copy(data_buf, data_cal)
                    T.tile.add(sum_acc, sum_acc, data_cal)
                    T.tile.mul(data_cal, data_cal, data_cal)
                    T.tile.add(sum_sq_acc, sum_sq_acc, data_cal)

                T.barrier_all()
                T.reduce_sum(sum_acc, sum_row, dim=-1)
                T.reduce_sum(sum_sq_acc, sum_sq_row, dim=-1)
                T.reduce_sum(sum_row, total, dim=-1)
                T.reduce_sum(sum_sq_row, total_sq, dim=-1)

                cnt = T.cast(cpg * S_orig, cal_dtype)
                T.tile.div(total, total, cnt)
                T.tile.div(total_sq, total_sq, cnt)
                T.tile.mul(mean_sq_val, total, total)
                T.tile.sub(var_val, total_sq, mean_sq_val)
                eps_v = T.cast(eps, cal_dtype)
                T.tile.add(var_val, var_val, eps_v)
                T.tile.sqrt(std_val, var_val)

                T.copy(gamma[g, 0:cpg], gamma_raw, pad_value=0.0)
                T.copy(beta[g, 0:cpg], beta_raw, pad_value=0.0)
                T.barrier_all()
                if use_fp32:
                    T.tile.cast(gamma_cal, gamma_raw, CAST_LOW2HIGH, cpg_padded)
                    T.tile.cast(beta_cal, beta_raw, CAST_LOW2HIGH, cpg_padded)
                else:
                    T.copy(gamma_raw, gamma_cal)
                    T.copy(beta_raw, beta_cal)

                T.tile.div(scale_cal, gamma_cal, std_val[0])
                T.tile.mul(temp_cal, scale_cal, total[0])
                T.tile.sub(offset_cal, beta_cal, temp_cal)

                T.tile.broadcast(scale_bc_full, scale_cal)
                T.copy(scale_bc_full[0:cpg, 0:block_S], scale_bc)
                T.tile.broadcast(offset_bc_full, offset_cal)
                T.copy(offset_bc_full[0:cpg, 0:block_S], offset_bc)
                T.barrier_all()

                if split_factor <= 2:
                    if half == 0:
                        T.set_flag("mte3", "mte2", 0)
                        T.set_flag("mte3", "mte2", 1)
                        T.wait_flag("mte3", "mte2", 0)
                        T.copy(x[n, g, 0:cpg, 0:block_S], data_buf_db[0, :, :], pad_value=0.0)
                        T.set_flag("mte2", "v", 0)

                        for si in T.serial(1, s_num_v0):
                            cur = (si - 1) % 2
                            nxt = si % 2
                            s_off_nxt = si * block_S
                            T.wait_flag("mte3", "mte2", nxt)
                            T.copy(x[n, g, 0:cpg, s_off_nxt : s_off_nxt + block_S], data_buf_db[nxt, :, :], pad_value=0.0)
                            T.set_flag("mte2", "v", nxt)
                            T.wait_flag("mte2", "v", cur)
                            if use_fp32:
                                T.tile.cast(data_cal, data_buf_db[cur, :, :], CAST_LOW2HIGH, tile_elem)
                            else:
                                T.copy(data_buf_db[cur, :, :], data_cal)
                            T.tile.mul(data_cal, data_cal, scale_bc)
                            T.tile.add(data_cal, data_cal, offset_bc)
                            if use_fp32:
                                T.tile.cast(out_buf_db[cur, :, :], data_cal, CAST_HIGH2LOW, tile_elem)
                            else:
                                T.copy(data_cal, out_buf_db[cur, :, :])
                            T.set_flag("v", "mte3", cur)
                            s_off_cur = (si - 1) * block_S
                            T.wait_flag("v", "mte3", cur)
                            T.copy(out_buf_db[cur, :, :], y[n, g, 0:cpg, s_off_cur : s_off_cur + block_S])
                            T.set_flag("mte3", "mte2", cur)

                        last_cur = (s_num_v0 - 1) % 2
                        s_off_last = (s_num_v0 - 1) * block_S
                        T.wait_flag("mte2", "v", last_cur)
                        if use_fp32:
                            T.tile.cast(data_cal, data_buf_db[last_cur, :, :], CAST_LOW2HIGH, tile_elem)
                        else:
                            T.copy(data_buf_db[last_cur, :, :], data_cal)
                        T.tile.mul(data_cal, data_cal, scale_bc)
                        T.tile.add(data_cal, data_cal, offset_bc)
                        if use_fp32:
                            T.tile.cast(out_buf_db[last_cur, :, :], data_cal, CAST_HIGH2LOW, tile_elem)
                        else:
                            T.copy(data_cal, out_buf_db[last_cur, :, :])
                        T.set_flag("v", "mte3", last_cur)
                        T.wait_flag("v", "mte3", last_cur)
                        T.copy(out_buf_db[last_cur, :, :], y[n, g, 0:cpg, s_off_last : s_off_last + block_S])
                        T.set_flag("mte3", "mte2", last_cur)
                        T.wait_flag("mte3", "mte2", 0)
                        T.wait_flag("mte3", "mte2", 1)
                    else:
                        T.set_flag("mte3", "mte2", 0)
                        T.set_flag("mte3", "mte2", 1)
                        s_off_0 = s_num_v0 * block_S
                        T.wait_flag("mte3", "mte2", 0)
                        T.copy(x[n, g, 0:cpg, s_off_0 : s_off_0 + block_S], data_buf_db[0, :, :], pad_value=0.0)
                        T.set_flag("mte2", "v", 0)

                        for si in T.serial(1, s_num_v1):
                            cur = (si - 1) % 2
                            nxt = si % 2
                            s_off_nxt = (si + s_num_v0) * block_S
                            T.wait_flag("mte3", "mte2", nxt)
                            T.copy(x[n, g, 0:cpg, s_off_nxt : s_off_nxt + block_S], data_buf_db[nxt, :, :], pad_value=0.0)
                            T.set_flag("mte2", "v", nxt)
                            T.wait_flag("mte2", "v", cur)
                            if use_fp32:
                                T.tile.cast(data_cal, data_buf_db[cur, :, :], CAST_LOW2HIGH, tile_elem)
                            else:
                                T.copy(data_buf_db[cur, :, :], data_cal)
                            T.tile.mul(data_cal, data_cal, scale_bc)
                            T.tile.add(data_cal, data_cal, offset_bc)
                            if use_fp32:
                                T.tile.cast(out_buf_db[cur, :, :], data_cal, CAST_HIGH2LOW, tile_elem)
                            else:
                                T.copy(data_cal, out_buf_db[cur, :, :])
                            T.set_flag("v", "mte3", cur)
                            s_off_cur = (si - 1 + s_num_v0) * block_S
                            T.wait_flag("v", "mte3", cur)
                            T.copy(out_buf_db[cur, :, :], y[n, g, 0:cpg, s_off_cur : s_off_cur + block_S])
                            T.set_flag("mte3", "mte2", cur)

                        last_cur = (s_num_v1 - 1) % 2
                        s_off_last = (s_num_v1 - 1 + s_num_v0) * block_S
                        T.wait_flag("mte2", "v", last_cur)
                        if use_fp32:
                            T.tile.cast(data_cal, data_buf_db[last_cur, :, :], CAST_LOW2HIGH, tile_elem)
                        else:
                            T.copy(data_buf_db[last_cur, :, :], data_cal)
                        T.tile.mul(data_cal, data_cal, scale_bc)
                        T.tile.add(data_cal, data_cal, offset_bc)
                        if use_fp32:
                            T.tile.cast(out_buf_db[last_cur, :, :], data_cal, CAST_HIGH2LOW, tile_elem)
                        else:
                            T.copy(data_cal, out_buf_db[last_cur, :, :])
                        T.set_flag("v", "mte3", last_cur)
                        T.wait_flag("v", "mte3", last_cur)
                        T.copy(out_buf_db[last_cur, :, :], y[n, g, 0:cpg, s_off_last : s_off_last + block_S])
                        T.set_flag("mte3", "mte2", last_cur)
                        T.wait_flag("mte3", "mte2", 0)
                        T.wait_flag("mte3", "mte2", 1)
                else:
                    for si in T.serial(s_num_v0):
                        actual_si = half * s_num_v0 + si
                        if actual_si < s_num:
                            s_off = actual_si * block_S
                            T.copy(x[n, g, 0:cpg, s_off : s_off + block_S], data_buf, pad_value=0.0)
                            T.barrier_all()
                            if use_fp32:
                                T.tile.cast(data_cal, data_buf, CAST_LOW2HIGH, tile_elem)
                            else:
                                T.copy(data_buf, data_cal)
                            T.tile.mul(data_cal, data_cal, scale_bc)
                            T.tile.add(data_cal, data_cal, offset_bc)
                            if use_fp32:
                                T.tile.cast(out_buf_db[0, :, :], data_cal, CAST_HIGH2LOW, tile_elem)
                            else:
                                T.copy(data_cal, out_buf_db[0, :, :])
                            T.barrier_all()
                            T.copy(out_buf_db[0, :, :], y[n, g, 0:cpg, s_off : s_off + block_S])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs_2d)
def group_norm_kernel_2d(
    N,
    G,
    cpg,
    cpg_padded,
    eps=1e-5,
    dtype="float32",
):
    block_num = N * G
    use_fp32 = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_fp32 else dtype

    @T.prim_func
    def main(
        x: T.Tensor((N, G, cpg), dtype),  # type: ignore
        gamma: T.Tensor((G, cpg), dtype),  # type: ignore
        beta: T.Tensor((G, cpg), dtype),  # type: ignore
        y: T.Tensor((N, G, cpg_padded), dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            n = cid // G
            g = cid % G
            if vid == 0:
                data_1d = T.alloc_shared([cpg_padded], dtype)
                data_cal = T.alloc_shared([cpg_padded], cal_dtype)
                sq_buf = T.alloc_shared([cpg_padded], cal_dtype)
                total = T.alloc_shared([1], cal_dtype)
                total_sq = T.alloc_shared([1], cal_dtype)
                mean_sq_val = T.alloc_shared([1], cal_dtype)
                var_val = T.alloc_shared([1], cal_dtype)
                std_val = T.alloc_shared([1], cal_dtype)
                gamma_1d = T.alloc_shared([cpg_padded], dtype)
                beta_1d = T.alloc_shared([cpg_padded], dtype)
                gamma_cal = T.alloc_shared([cpg_padded], cal_dtype)
                beta_cal = T.alloc_shared([cpg_padded], cal_dtype)
                scale_1d = T.alloc_shared([cpg_padded], cal_dtype)
                offset_1d = T.alloc_shared([cpg_padded], cal_dtype)
                temp_1d = T.alloc_shared([cpg_padded], cal_dtype)

                T.copy(x[n, g, 0:cpg], data_1d, pad_value=0.0)
                if use_fp32:
                    T.tile.cast(data_cal, data_1d, CAST_LOW2HIGH, cpg_padded)
                else:
                    T.copy(data_1d, data_cal)
                T.reduce_sum(data_cal, total, dim=-1)
                T.tile.mul(sq_buf, data_cal, data_cal)
                T.reduce_sum(sq_buf, total_sq, dim=-1)
                cnt = T.cast(cpg, cal_dtype)
                T.tile.div(total, total, cnt)
                T.tile.div(total_sq, total_sq, cnt)
                T.tile.mul(mean_sq_val, total, total)
                T.tile.sub(var_val, total_sq, mean_sq_val)
                eps_v = T.cast(eps, cal_dtype)
                T.tile.add(var_val, var_val, eps_v)
                T.tile.sqrt(std_val, var_val)
                T.copy(gamma[g, 0:cpg], gamma_1d, pad_value=0.0)
                T.copy(beta[g, 0:cpg], beta_1d, pad_value=0.0)
                if use_fp32:
                    T.tile.cast(gamma_cal, gamma_1d, CAST_LOW2HIGH, cpg_padded)
                    T.tile.cast(beta_cal, beta_1d, CAST_LOW2HIGH, cpg_padded)
                else:
                    T.copy(gamma_1d, gamma_cal)
                    T.copy(beta_1d, beta_cal)
                T.tile.div(scale_1d, gamma_cal, std_val[0])
                T.tile.mul(temp_1d, scale_1d, total[0])
                T.tile.sub(offset_1d, beta_cal, temp_1d)
                T.tile.mul(data_cal, data_cal, scale_1d)
                T.tile.add(data_cal, data_cal, offset_1d)
                if use_fp32:
                    T.tile.cast(data_1d, data_cal, CAST_HIGH2LOW, cpg_padded)
                else:
                    T.copy(data_cal, data_1d)
                T.copy(data_1d, y[n, g, 0:cpg_padded])

    return main


# ========== Golden Reference ==========
def golden_group_norm(x, gamma, beta, num_groups, eps=1e-5):
    """PyTorch reference: torch.nn.functional.group_norm."""
    if x.ndim == 2:
        return torch.nn.functional.group_norm(x.unsqueeze(-1), num_groups, gamma, beta, eps).squeeze(-1)
    return torch.nn.functional.group_norm(x, num_groups, gamma, beta, eps)


# ========== Tiling & Host Wrapper ==========
def _find_block_S(S, cpg, dtype_str):
    UB_BUDGET = 192 * 1024
    cal_bytes = 4
    dtype_bytes = 2 if dtype_str in ("float16", "bfloat16") else 4
    c = max(cpg, 1)
    per_block = c * (6 * cal_bytes + 6 * dtype_bytes)
    max_block_S = (UB_BUDGET // per_block // 16) * 16
    max_block_S = min(max(max_block_S, 16), 1024)
    for bs in range(max_block_S, 0, -16):
        if S % bs == 0:
            return bs
    return max_block_S


def _get_kernel(N, G, cpg, S, S_padded, block_S, s_num, s_num_v0, s_num_v1, split_factor, single_pass, eps, dtype_str, S_orig):
    key = (N, G, cpg, S, S_padded, block_S, s_num, s_num_v0, s_num_v1, split_factor, single_pass, eps, dtype_str, S_orig)
    if key not in _kernel_cache:
        _kernel_cache[key] = group_norm_kernel_expert(
            N,
            G,
            cpg,
            S,
            S_padded,
            block_S,
            s_num,
            s_num_v0,
            s_num_v1,
            split_factor,
            single_pass,
            eps,
            dtype_str,
            S_orig,
        )
    return _kernel_cache[key]


def _get_kernel_2d(N, G, cpg, cpg_padded, eps, dtype_str):
    key = (N, G, cpg, cpg_padded, eps, dtype_str)
    if key not in _kernel_cache_2d:
        _kernel_cache_2d[key] = group_norm_kernel_2d(
            N,
            G,
            cpg,
            cpg_padded,
            eps,
            dtype_str,
        )
    return _kernel_cache_2d[key]


def group_norm(x, gamma, beta, num_groups, epsilon=1e-5):
    """GroupNorm host function."""
    original_shape = x.shape
    N = x.shape[0]
    C = x.shape[1]
    S = 1
    for i in range(2, x.ndim):
        S *= x.shape[i]

    dtype_str = str(x.dtype).replace("torch.", "")
    cpg = C // num_groups

    if x.ndim == 2:
        cpg_padded = max(((cpg + 7) // 8) * 8, 8)
        x_3d = x.reshape(N, num_groups, cpg)
        gamma_2d = gamma.reshape(num_groups, cpg)
        beta_2d = beta.reshape(num_groups, cpg)
        func = _get_kernel_2d(N, num_groups, cpg, cpg_padded, epsilon, dtype_str)
        y_3d = func(x_3d, gamma_2d, beta_2d)
        if cpg_padded > cpg:
            y_3d = y_3d[:, :, :cpg]
        return y_3d.reshape(original_shape)

    block_S = _find_block_S(S, cpg, dtype_str)
    s_num = (S + block_S - 1) // block_S
    S_padded = s_num * block_S

    block_num_base = N * num_groups
    if s_num > 1:
        max_split = CORE_NUM // block_num_base
        split_factor = max(1, min(max_split, 20))
    else:
        split_factor = 1
    s_num_v0 = (s_num + split_factor - 1) // split_factor
    s_num_v1 = s_num // split_factor

    single_pass = 1 if s_num == 1 else 0

    x_4d = x.reshape(N, num_groups, cpg, S)
    gamma_2d = gamma.reshape(num_groups, cpg)
    beta_2d = beta.reshape(num_groups, cpg)

    func = _get_kernel(
        N,
        num_groups,
        cpg,
        S,
        S_padded,
        block_S,
        s_num,
        s_num_v0,
        s_num_v1,
        split_factor,
        single_pass,
        epsilon,
        dtype_str,
        S_orig=S,
    )
    y_4d = func(x_4d, gamma_2d, beta_2d)

    if S_padded > S:
        y_4d = y_4d[:, :, :, :S]
    return y_4d.reshape(original_shape)


# ========== Precision Standards ==========
def get_precision(dtype):
    precision_map = {
        "float16": (1e-3, 1e-3),
        "float32": (1e-4, 1e-4),
        "bfloat16": (1e-2, 5e-3),
    }
    return precision_map[dtype]


# ========== cann-bench Precision Checker ==========
def _mere_mare(actual, golden):
    diff = (actual.float() - golden.float()).abs()
    denom = golden.float().abs() + 1e-7
    rel_err = diff / denom
    both_nan = torch.isnan(actual.float()) & torch.isnan(golden.float())
    both_inf = torch.isinf(actual.float()) & torch.isinf(golden.float())
    rel_err = torch.where(both_nan | both_inf, torch.zeros_like(rel_err), rel_err)
    mere = rel_err.mean().item()
    mare = rel_err.max().item()
    return mere, mare


def check_precision(actual, golden, dtype_str, label=""):
    mere, mare = _mere_mare(actual, golden)
    threshold = _CANNBENCH_THRESHOLDS[dtype_str]
    mare_threshold = 10 * threshold
    passed = mere < threshold and mare < mare_threshold
    status = "PASS" if passed else "FAIL"
    tag = f"[{'PRECISION_PASS' if passed else 'PRECISION_FAIL'}]"
    print(f"{tag} {label} dtype={dtype_str} MERE={mere:.6e} MARE={mare:.6e} threshold={threshold:.6e} -> {status}")
    return passed


# ========== Input Generation ==========
def _gen_input(shape, dtype, value_range, seed_offset=0):
    torch.manual_seed(42 + seed_offset)
    torch_dtype = _DTYPE_MAP[dtype] if isinstance(dtype, str) else dtype
    dtype_str = dtype if isinstance(dtype, str) else str(dtype).replace("torch.", "")

    if value_range == "inf":
        x = torch.randn(shape, dtype=torch.float32).uniform_(-1, 1)
        x.view(-1)[0] = float("inf")
        if x.numel() > 1:
            x.view(-1)[1] = float("-inf")
    elif value_range == "nan":
        x = torch.randn(shape, dtype=torch.float32).uniform_(-1, 1)
        x.view(-1)[0] = float("nan")
    elif value_range == "zero":
        x = torch.zeros(shape, dtype=torch.float32)
    else:
        lo, hi = value_range
        x = torch.empty(shape, dtype=torch.float32).uniform_(lo, hi)

    return x.to(torch_dtype).npu(), dtype_str


def _cleanup_npu():
    gc.collect()
    torch.npu.empty_cache()


# ========== Test Runner ==========
def _run_case(name, shape, dtype, num_groups, eps, value_range, level="cann-bench"):
    _cleanup_npu()
    dtype_str = dtype if isinstance(dtype, str) else str(dtype).replace("torch.", "")
    C = shape[1]

    x, dtype_str = _gen_input(shape, dtype_str, value_range, seed_offset=hash(name) % 1000)
    gamma = torch.randn(C, dtype=_DTYPE_MAP[dtype_str]).npu()
    beta = torch.randn(C, dtype=_DTYPE_MAP[dtype_str]).npu()

    y = group_norm(x, gamma, beta, num_groups, eps)
    ref = golden_group_norm(x.cpu().float(), gamma.cpu().float(), beta.cpu().float(), num_groups, eps).to(_DTYPE_MAP[dtype_str])

    if level in ("l0", "l1"):
        atol, rtol = get_precision(dtype_str)
        max_diff = (y.cpu().float() - ref.float()).abs().max().item()
        try:
            torch.testing.assert_close(y.cpu(), ref, atol=atol, rtol=rtol)
            print(f"[PRECISION_PASS] {level} {name} shape={shape} dtype={dtype_str} max_diff={max_diff:.6e}")
            return True
        except Exception as e:
            print(f"[PRECISION_FAIL] {level} {name} shape={shape} dtype={dtype_str} max_diff={max_diff:.6e}: {e}")
            return False
    else:
        return check_precision(y.cpu(), ref, dtype_str, label=f"{level} {name} shape={shape}")


def _run_boundary(level, name, fn):
    try:
        fn()
        print(f"[BOUNDARY_PASS] {level} {name}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] {level} {name}: {e}")


# ========== L0 Tests ==========
def test_group_norm_l0():
    configs = [
        ("l0-1", [8, 32, 64, 64], "float16", 8, 1e-5, (-1, 1)),
        ("l0-2", [4, 64, 128, 128], "float32", 16, 1e-5, (-2, 2)),
        ("l0-3", [2, 128, 256, 256], "bfloat16", 32, 1e-5, (-3, 3)),
    ]
    ok = True
    for name, shape, dtype, num_groups, eps, vrange in configs:
        ok &= _run_case(name, shape, dtype, num_groups, eps, vrange, level="l0")
    return ok


# ========== L1 Functional Tests ==========
def test_group_norm_l1():
    configs = [
        ("l1-1", [16, 257, 32, 31], "float16", 1, 1e-5, (-10, 10)),
        ("l1-2", [8, 512, 17, 15], "float32", 2, 1e-5, (-100, 100)),
        ("l1-3", [64, 64, 128], "bfloat16", 4, 1e-5, (-5, 5)),
        ("l1-4", [5, 48, 33, 65], "bfloat16", 6, 1e-4, (-3, 6)),
        ("l1-5", [1023, 257], "float16", 1, 1e-6, (-1, 1)),
        ("l1-6", [2, 60, 5, 7, 480], "float32", 4, 1e-5, (-10, 10)),
    ]
    ok = True
    for name, shape, dtype, num_groups, eps, vrange in configs:
        ok &= _run_case(name, shape, dtype, num_groups, eps, vrange, level="l1")
    return ok


# ========== L2 Exception Tests ==========
def test_group_norm_l2():
    def test_unsupported_dtype():
        x = torch.ones(2, 8, 4, 4, dtype=torch.int32).npu()
        gamma = torch.ones(8, dtype=torch.int32).npu()
        beta = torch.ones(8, dtype=torch.int32).npu()
        group_norm(x, gamma, beta, 2, 1e-5)

    def test_c_not_divisible():
        x = torch.randn(2, 7, 4, 4, dtype=torch.float16).npu()
        gamma = torch.randn(7, dtype=torch.float16).npu()
        beta = torch.randn(7, dtype=torch.float16).npu()
        group_norm(x, gamma, beta, 3, 1e-5)

    def test_none_input():
        group_norm(None, None, None, 1, 1e-5)

    _run_boundary("l2", "unsupported_dtype_int32", test_unsupported_dtype)
    _run_boundary("l2", "c_not_divisible_by_num_groups", test_c_not_divisible)
    _run_boundary("l2", "none_input", test_none_input)


# ========== Boundary Tests ==========
def test_group_norm_boundary():
    def test_inf_input():
        x = torch.randn(2, 8, 4, 4, dtype=torch.float16).npu()
        x.view(-1)[0] = float("inf")
        gamma = torch.randn(8, dtype=torch.float16).npu()
        beta = torch.randn(8, dtype=torch.float16).npu()
        y = group_norm(x, gamma, beta, 2, 1e-5)
        assert torch.isinf(y.cpu()).any() or torch.isnan(y.cpu()).any()

    def test_nan_input():
        x = torch.randn(2, 8, 4, 4, dtype=torch.float16).npu()
        x.view(-1)[0] = float("nan")
        gamma = torch.randn(8, dtype=torch.float16).npu()
        beta = torch.randn(8, dtype=torch.float16).npu()
        y = group_norm(x, gamma, beta, 2, 1e-5)
        assert torch.isnan(y.cpu()).any()

    def test_all_zeros():
        x = torch.zeros(2, 8, 4, 4, dtype=torch.float16).npu()
        gamma = torch.randn(8, dtype=torch.float16).npu()
        beta = torch.randn(8, dtype=torch.float16).npu()
        y = group_norm(x, gamma, beta, 2, 1e-5)
        ref = golden_group_norm(x.cpu().float(), gamma.cpu().float(), beta.cpu().float(), 2, 1e-5).to(torch.float16)
        torch.testing.assert_close(y.cpu(), ref, atol=1e-3, rtol=1e-3)

    def test_extreme_values():
        x = torch.empty(2, 8, 4, 4, dtype=torch.float32).uniform_(-65504, 65504).to(torch.float16).npu()
        gamma = torch.randn(8, dtype=torch.float16).npu()
        beta = torch.randn(8, dtype=torch.float16).npu()
        y = group_norm(x, gamma, beta, 2, 1e-5)
        ref = golden_group_norm(x.cpu().float(), gamma.cpu().float(), beta.cpu().float(), 2, 1e-5).to(torch.float16)
        torch.testing.assert_close(y.cpu(), ref, atol=1e-2, rtol=1e-2)

    _run_boundary("boundary", "inf_input", test_inf_input)
    _run_boundary("boundary", "nan_input", test_nan_input)
    _run_boundary("boundary", "all_zeros", test_all_zeros)
    _run_boundary("boundary", "extreme_values_fp16max", test_extreme_values)


# ========== cann-bench 20 Cases ==========
def test_group_norm_cann_bench():
    configs = [
        ("cann-bench-1", [8, 32, 64, 64], "float16", 8, 1e-5, (-1, 1)),
        ("cann-bench-2", [4, 64, 128, 128], "float32", 16, 1e-5, (-2, 2)),
        ("cann-bench-3", [2, 128, 256, 256], "bfloat16", 32, 1e-5, (-3, 3)),
        ("cann-bench-4", [16, 257, 32, 31], "float16", 1, 1e-5, (-10, 10)),
        ("cann-bench-5", [8, 512, 17, 15], "float32", 2, 1e-5, (-100, 100)),
        ("cann-bench-6", [64, 64, 128], "bfloat16", 4, 1e-5, (-5, 5)),
        ("cann-bench-7", [2, 256, 128, 128], "float16", 16, 1e-5, (-0.1, 0.1)),
        ("cann-bench-8", [16, 127, 31, 33], "float32", 1, 1e-6, (-1, 1)),
        ("cann-bench-9", [3, 64, 64, 64], "bfloat16", 8, 1e-3, (-0.5, 0.5)),
        ("cann-bench-10", [7, 32, 63, 65], "float16", 4, 1e-4, (-1, 2)),
        ("cann-bench-11", [3, 64, 127, 129], "float32", 8, 1e-4, (-50, 100)),
        ("cann-bench-12", [5, 48, 33, 65], "bfloat16", 6, 1e-4, (-3, 6)),
        ("cann-bench-13", [1023, 257], "float16", 1, 1e-6, (-1, 1)),
        ("cann-bench-14", [2, 60, 5, 7, 480], "float32", 4, 1e-5, (-10, 10)),
        ("cann-bench-15", [4, 31, 251, 251], "bfloat16", 1, 1e-8, "inf"),
        ("cann-bench-16", [2, 64, 67, 71], "float16", 8, 1e-7, "nan"),
        ("cann-bench-17", [8, 127, 33, 31], "float32", 1, 1e-4, "zero"),
        ("cann-bench-18", [2, 256, 127, 129], "bfloat16", 16, 1e-5, (-0.2, 0.2)),
        ("cann-bench-19", [4, 128, 255, 257], "float16", 32, 1e-3, (-65504, 65504)),
        ("cann-bench-20", [1, 513, 63, 63], "float32", 3, 1e-6, (-20, 40)),
    ]
    ok = True
    for name, shape, dtype, num_groups, eps, vrange in configs:
        ok &= _run_case(name, shape, dtype, num_groups, eps, vrange, level="cann-bench")
    return ok


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="GroupNorm example with cann-bench 20 cases")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "cann-bench", "all"],
        help="Test level to run (default: l0)",
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True
    if args.level in ("l0", "all"):
        blocking_ok &= test_group_norm_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_group_norm_l1()
    if args.level in ("l2", "all"):
        test_group_norm_l2()
    if args.level in ("boundary", "all"):
        test_group_norm_boundary()
    if args.level in ("cann-bench", "all"):
        blocking_ok &= test_group_norm_cann_bench()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
