# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms of the CANN Open Software License Agreement Version 2.0 (the
# "License"). You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED.
# ----------------------------------------------------------------------------

"""GroupNorm TileLang kernel — Expert mode, double-buffer pipeline.

Expert mode GroupNorm with a Pass 2 double buffer pipeline for arbitrary
shape, dtype, num_groups, and epsilon.

Algorithm:
    y = (x - mean) / sqrt(var + eps) * gamma + beta  (per-group)

Reference: torch.nn.functional.group_norm (golden_group_norm in test_group_norm.py)

Key optimizations:
- Expert mode (AUTO_SYNC=False) with Pass 2 double buffer pipeline
- MTE2→V→MTE3 three-way flag pipeline for overlap
- Adaptive spatial tiling via UB budget formula
- 2D Developer mode path for S=1 inputs

Interface: group_norm(x, gamma, beta, num_groups, epsilon) -> y
"""

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


# ----------------------------------------------------------------------------
# CI smoke test (examples/bench_test.sh runs this file directly and requires
# "Kernel Output Match!" or "TEST PASSED!" in the output, same as the
# lstm/gru/depthwise_conv_2d/nms smoke tests).  The full layered suite over
# the 20 cann-bench cases lives in test_group_norm.py.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn.functional as F

    torch.manual_seed(0)

    def _ref(x, gamma, beta, g, eps):
        if x.ndim == 2:
            return F.group_norm(x.unsqueeze(-1), g, gamma, beta, eps).squeeze(-1)
        return F.group_norm(x, g, gamma, beta, eps)

    # one case per kernel path and dtype: 4D expert / 4D expert / 2D developer
    for shape, dtype, g, eps in (
        ((8, 32, 64, 64), torch.float16, 8, 1e-5),
        ((4, 64, 128, 128), torch.float32, 16, 1e-5),
        ((1023, 257), torch.bfloat16, 1, 1e-6),
    ):
        x = (torch.rand(*shape) * 2 - 1).to(dtype)
        gamma = (torch.rand(shape[1]) * 2 - 1).to(dtype)
        beta = (torch.rand(shape[1]) * 2 - 1).to(dtype)
        y = group_norm(x.npu(), gamma.npu(), beta.npu(), g, eps).cpu()
        ref = _ref(x.float(), gamma.float(), beta.float(), g, eps).to(dtype)
        atol, rtol = {
            torch.float16: (1e-3, 1e-3),
            torch.float32: (1e-4, 1e-4),
            torch.bfloat16: (1e-2, 5e-3),
        }[dtype]
        torch.testing.assert_close(y, ref, atol=atol, rtol=rtol)
        diff = (y.float() - ref.float()).abs().max().item()
        print(f"[SMOKE] {tuple(shape)} {str(dtype).split('.')[-1]} G={g} max_abs={diff:.3e}")
    print("Kernel Output Match!")
