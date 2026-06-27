"""GroupNorm operator implementation for Ascend NPU using TileLang (Expert mode).

Group Normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
Computed per group across all channels and spatial dimensions within each group.

Two kernels, selected at host level:
- Serial kernel (s_num>=2): T.serial + double-buffer + MTE2/V/MTE3 3-stage pipeline
- Cpipeline kernel (s_num==1, cpg tiles>=2): T.Pipelined + parity-split accumulation
  + cpg-dim pipeline for Pass 2
- T.alloc_ub + T.Scope("V") + manual flag sync throughout
- TL_ASCEND_MEMORY_PLANNING enabled for buffer reuse
"""

import tilelang
from tilelang import language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CAST_LOW2HIGH = "CAST_NONE"
CAST_HIGH2LOW = "CAST_RINT"


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def group_norm_kernel_serial(
    N, G, cpg_padded, S_padded, block_S, s_num, eps=1e-5, cpg=0, S_orig=0, dtype="float32"
):
    """Original serial kernel: T.serial + double-buffer + MTE2/V/MTE3 3-stage."""
    block_num = N * G
    tile_elem = cpg * block_S

    use_fp32 = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_fp32 else dtype

    @T.prim_func
    def main(
        x: T.Tensor((N, G, cpg, S_orig), dtype),  # type: ignore
        gamma: T.Tensor((G, cpg), dtype),  # type: ignore
        beta: T.Tensor((G, cpg), dtype),  # type: ignore
        y: T.Tensor((N, G, cpg, S_padded), dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            n = cid // G
            g = cid % G
            if vid == 0:
                sum_a = T.alloc_ub([cpg, block_S], cal_dtype)
                sum_sq_a = T.alloc_ub([cpg, block_S], cal_dtype)
                sum_row = T.alloc_ub([cpg], cal_dtype)
                sum_sq_row = T.alloc_ub([cpg], cal_dtype)
                data_buf_p1 = T.alloc_ub([2, cpg, block_S], dtype)
                total = T.alloc_ub([1], cal_dtype)
                total_sq = T.alloc_ub([1], cal_dtype)
                mean_sq_val = T.alloc_ub([1], cal_dtype)
                var_val = T.alloc_ub([1], cal_dtype)
                std_val = T.alloc_ub([1], cal_dtype)
                mean_col = T.alloc_ub([cpg, 1], cal_dtype)
                std_col = T.alloc_ub([cpg, 1], cal_dtype)
                data_cal = T.alloc_ub([cpg, block_S], cal_dtype)

                with T.Scope("V"):
                    T.tile.fill(sum_a, 0.0)
                    T.tile.fill(sum_sq_a, 0.0)

                    T.copy(x[n, g, 0:cpg, 0:block_S], data_buf_p1[0, :, :])
                    T.barrier_all()

                    for si in T.serial(s_num):
                        cur = si % 2
                        nxt = (si + 1) % 2
                        if si < s_num - 1:
                            s_off_nxt = (si + 1) * block_S
                            T.copy(
                                x[n, g, 0:cpg, s_off_nxt : s_off_nxt + block_S],
                                data_buf_p1[nxt, :, :],
                            )
                        if use_fp32:
                            T.tile.cast(data_cal, data_buf_p1[cur, :, :], CAST_LOW2HIGH, tile_elem)
                        else:
                            T.copy(data_buf_p1[cur, :, :], data_cal)
                        T.tile.add(sum_a, sum_a, data_cal)
                        T.tile.mul(data_cal, data_cal, data_cal)
                        T.tile.add(sum_sq_a, sum_sq_a, data_cal)
                        T.barrier_all()

                    T.reduce_sum(sum_a, sum_row, dim=-1)
                    T.reduce_sum(sum_sq_a, sum_sq_row, dim=-1)
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

                mean_bc = T.alloc_ub([cpg, block_S], cal_dtype)
                std_bc = T.alloc_ub([cpg, block_S], cal_dtype)
                gamma_raw = T.alloc_ub([cpg_padded, 1], dtype)
                beta_raw = T.alloc_ub([cpg_padded, 1], dtype)
                gamma_cal = T.alloc_ub([cpg_padded, 1], cal_dtype)
                beta_cal = T.alloc_ub([cpg_padded, 1], cal_dtype)
                gamma_bc_full = T.alloc_ub([cpg_padded, block_S], cal_dtype)
                beta_bc_full = T.alloc_ub([cpg_padded, block_S], cal_dtype)
                gamma_bc = T.alloc_ub([cpg, block_S], cal_dtype)
                beta_bc = T.alloc_ub([cpg, block_S], cal_dtype)
                data_buf_p2 = T.alloc_ub([2, cpg, block_S], dtype)
                data_cal_p2 = T.alloc_ub([cpg, block_S], cal_dtype)
                out_buf_p2 = T.alloc_ub([2, cpg, block_S], dtype)

                with T.Scope("V"):
                    T.tile.fill(mean_col, total)
                    T.tile.broadcast(mean_bc, mean_col)
                    T.tile.fill(std_col, std_val)
                    T.tile.broadcast(std_bc, std_col)

                    T.copy(gamma[g, 0:cpg], gamma_raw, pad_value=0.0)
                    T.copy(beta[g, 0:cpg], beta_raw, pad_value=0.0)
                    T.barrier_all()
                    if use_fp32:
                        T.tile.cast(gamma_cal, gamma_raw, CAST_LOW2HIGH, cpg_padded)
                        T.tile.cast(beta_cal, beta_raw, CAST_LOW2HIGH, cpg_padded)
                    else:
                        T.copy(gamma_raw, gamma_cal)
                        T.copy(beta_raw, beta_cal)
                    T.tile.broadcast(gamma_bc_full, gamma_cal)
                    T.tile.broadcast(beta_bc_full, beta_cal)
                    T.copy(gamma_bc_full[0:cpg, 0:block_S], gamma_bc)
                    T.copy(beta_bc_full[0:cpg, 0:block_S], beta_bc)
                    T.barrier_all()

                    T.set_flag("mte3", "mte2", 0)
                    T.set_flag("mte3", "mte2", 1)
                    T.wait_flag("mte3", "mte2", 0)
                    T.copy(x[n, g, 0:cpg, 0:block_S], data_buf_p2[0, :, :])
                    T.set_flag("mte2", "v", 0)

                    for si in T.serial(s_num):
                        cur = si % 2
                        nxt = (si + 1) % 2
                        if si < s_num - 1:
                            s_off_nxt = (si + 1) * block_S
                            T.wait_flag("mte3", "mte2", nxt)
                            T.copy(
                                x[n, g, 0:cpg, s_off_nxt : s_off_nxt + block_S],
                                data_buf_p2[nxt, :, :],
                            )
                            T.set_flag("mte2", "v", nxt)
                        T.wait_flag("mte2", "v", cur)
                        if use_fp32:
                            T.tile.cast(
                                data_cal_p2, data_buf_p2[cur, :, :],
                                CAST_LOW2HIGH, tile_elem,
                            )
                        else:
                            T.copy(data_buf_p2[cur, :, :], data_cal_p2)
                        T.tile.sub(data_cal_p2, data_cal_p2, mean_bc)
                        T.tile.div(data_cal_p2, data_cal_p2, std_bc)
                        T.tile.mul(data_cal_p2, data_cal_p2, gamma_bc)
                        T.tile.add(data_cal_p2, data_cal_p2, beta_bc)
                        if use_fp32:
                            T.tile.cast(
                                out_buf_p2[cur, :, :], data_cal_p2,
                                CAST_HIGH2LOW, tile_elem,
                            )
                        else:
                            T.copy(data_cal_p2, out_buf_p2[cur, :, :])
                        T.set_flag("v", "mte3", cur)
                        T.wait_flag("v", "mte3", cur)
                        s_off_cur = si * block_S
                        T.copy(
                            out_buf_p2[cur, :, :],
                            y[n, g, 0:cpg, s_off_cur : s_off_cur + block_S],
                        )
                        T.set_flag("mte3", "mte2", cur)

                    T.wait_flag("mte3", "mte2", 0)
                    T.wait_flag("mte3", "mte2", 1)

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def group_norm_kernel_cpipeline(
    N, G, cpg_padded, S_padded, block_S, eps=1e-5, cpg=0, S_orig=0, dtype="float32",
):
    """Cpipeline kernel: s_num==1 + T.Pipelined + parity-split + cpg-dim pipeline."""
    # Force block_C=128 max to ensure 2-tile pipeline even when block_S=1
    block_C = max(16, min(cpg, 128))
    block_C = (block_C // 16) * 16
    cpg_full_tiles = cpg // block_C
    cpg_rem = cpg % block_C
    tile_elem_c = block_C * block_S
    block_num = N * G
    tile_elem = cpg * block_S
    s_num = S_padded // block_S

    use_fp32 = dtype in ["float16", "bfloat16"]
    cal_dtype = "float32" if use_fp32 else dtype

    @T.prim_func
    def main(
        x: T.Tensor((N, G, cpg, S_padded), dtype),  # type: ignore
        gamma: T.Tensor((G, cpg), dtype),  # type: ignore
        beta: T.Tensor((G, cpg), dtype),  # type: ignore
        y: T.Tensor((N, G, cpg, S_padded), dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            n = cid // G
            g = cid % G
            if vid == 0:
                data_buf = T.alloc_ub([cpg, block_S], dtype)
                data_cal = T.alloc_ub([cpg, block_S], cal_dtype)
                sum_a = T.alloc_ub([cpg, block_S], cal_dtype)
                sum_b = T.alloc_ub([cpg, block_S], cal_dtype)
                sum_sq_a = T.alloc_ub([cpg, block_S], cal_dtype)
                sum_sq_b = T.alloc_ub([cpg, block_S], cal_dtype)
                sum_row = T.alloc_ub([cpg], cal_dtype)
                sum_sq_row = T.alloc_ub([cpg], cal_dtype)
                total = T.alloc_ub([1], cal_dtype)
                total_sq = T.alloc_ub([1], cal_dtype)
                mean_sq_val = T.alloc_ub([1], cal_dtype)
                var_val = T.alloc_ub([1], cal_dtype)
                std_val = T.alloc_ub([1], cal_dtype)
                mean_bc = T.alloc_ub([cpg, block_S], cal_dtype)
                std_bc = T.alloc_ub([cpg, block_S], cal_dtype)
                gamma_raw = T.alloc_ub([cpg_padded, 1], dtype)
                beta_raw = T.alloc_ub([cpg_padded, 1], dtype)
                gamma_cal = T.alloc_ub([cpg_padded, 1], cal_dtype)
                beta_cal = T.alloc_ub([cpg_padded, 1], cal_dtype)
                gamma_bc_full = T.alloc_ub([cpg_padded, block_S], cal_dtype)
                beta_bc_full = T.alloc_ub([cpg_padded, block_S], cal_dtype)
                gamma_bc = T.alloc_ub([cpg, block_S], cal_dtype)
                beta_bc = T.alloc_ub([cpg, block_S], cal_dtype)
                data_buf_p2 = T.alloc_ub([2, cpg, block_S], dtype)
                data_cal_p2 = T.alloc_ub([cpg, block_S], cal_dtype)
                out_buf_p2 = T.alloc_ub([2, cpg, block_S], dtype)

                with T.Scope("V"):
                    T.tile.fill(sum_a, 0.0)
                    T.tile.fill(sum_b, 0.0)
                    T.tile.fill(sum_sq_a, 0.0)
                    T.tile.fill(sum_sq_b, 0.0)

                    for si in T.Pipelined(s_num, num_stages=2):
                        s_off = si * block_S
                        T.copy(x[n, g, 0:cpg, s_off : s_off + block_S], data_buf)
                        T.barrier_all()
                        if use_fp32:
                            T.tile.cast(data_cal, data_buf, CAST_LOW2HIGH, tile_elem)
                        else:
                            T.copy(data_buf, data_cal)
                        if si % 2 == 0:
                            T.tile.add(sum_a, sum_a, data_cal)
                        else:
                            T.tile.add(sum_b, sum_b, data_cal)
                        T.tile.mul(data_cal, data_cal, data_cal)
                        if si % 2 == 0:
                            T.tile.add(sum_sq_a, sum_sq_a, data_cal)
                        else:
                            T.tile.add(sum_sq_b, sum_sq_b, data_cal)

                    T.tile.add(sum_a, sum_a, sum_b)
                    T.tile.add(sum_sq_a, sum_sq_a, sum_sq_b)

                    T.reduce_sum(sum_a, sum_row, dim=-1)
                    T.reduce_sum(sum_sq_a, sum_sq_row, dim=-1)
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

                    T.tile.fill(mean_bc, total)
                    T.tile.fill(std_bc, std_val)

                    T.copy(gamma[g, 0:cpg], gamma_raw, pad_value=0.0)
                    T.copy(beta[g, 0:cpg], beta_raw, pad_value=0.0)
                    T.barrier_all()
                    if use_fp32:
                        T.tile.cast(gamma_cal, gamma_raw, CAST_LOW2HIGH, cpg_padded)
                        T.tile.cast(beta_cal, beta_raw, CAST_LOW2HIGH, cpg_padded)
                    else:
                        T.copy(gamma_raw, gamma_cal)
                        T.copy(beta_raw, beta_cal)
                    T.tile.broadcast(gamma_bc_full, gamma_cal)
                    T.tile.broadcast(beta_bc_full, beta_cal)
                    T.copy(gamma_bc_full[0:cpg, 0:block_S], gamma_bc)
                    T.copy(beta_bc_full[0:cpg, 0:block_S], beta_bc)
                    T.barrier_all()

                    # === Pass 2: cpg-dim 3-stage pipeline (MTE2/V/MTE3) ===
                    # Use compile-time constant indices for flag-based pipeline
                    # (T.serial with runtime ci causes memory planner issues)
                    data_buf_c0 = T.alloc_ub([block_C, block_S], dtype)
                    data_buf_c1 = T.alloc_ub([block_C, block_S], dtype)
                    data_cal_c = T.alloc_ub([block_C, block_S], cal_dtype)
                    out_buf_c0 = T.alloc_ub([block_C, block_S], dtype)
                    out_buf_c1 = T.alloc_ub([block_C, block_S], dtype)
                    param_tile = T.alloc_ub([block_C, block_S], cal_dtype)

                    T.set_flag("mte3", "mte2", 0)
                    T.set_flag("mte3", "mte2", 1)
                    T.wait_flag("mte3", "mte2", 0)
                    T.copy(
                        x[n, g, 0:block_C, 0:block_S],
                        data_buf_c0,
                    )
                    T.set_flag("mte2", "v", 0)

                    # Iteration 0: process tile 0, prefetch tile 1
                    if cpg_full_tiles >= 2:
                        T.wait_flag("mte3", "mte2", 1)
                        T.copy(
                            x[n, g, block_C : 2 * block_C, 0:block_S],
                            data_buf_c1,
                        )
                        T.set_flag("mte2", "v", 1)

                    T.wait_flag("mte2", "v", 0)
                    if use_fp32:
                        T.tile.cast(data_cal_c, data_buf_c0, CAST_LOW2HIGH, tile_elem_c)
                    else:
                        T.copy(data_buf_c0, data_cal_c)
                    T.copy(mean_bc[0:block_C, :], param_tile)
                    T.tile.sub(data_cal_c, data_cal_c, param_tile)
                    T.copy(std_bc[0:block_C, :], param_tile)
                    T.tile.div(data_cal_c, data_cal_c, param_tile)
                    T.copy(gamma_bc[0:block_C, :], param_tile)
                    T.tile.mul(data_cal_c, data_cal_c, param_tile)
                    T.copy(beta_bc[0:block_C, :], param_tile)
                    T.tile.add(data_cal_c, data_cal_c, param_tile)
                    if use_fp32:
                        T.tile.cast(out_buf_c0, data_cal_c, CAST_HIGH2LOW, tile_elem_c)
                    else:
                        T.copy(data_cal_c, out_buf_c0)
                    T.set_flag("v", "mte3", 0)
                    T.wait_flag("v", "mte3", 0)
                    T.copy(out_buf_c0, y[n, g, 0:block_C, 0:block_S])
                    T.set_flag("mte3", "mte2", 0)

                    # Iteration 1: process tile 1, prefetch tile 2 (if exists)
                    if cpg_full_tiles >= 2:
                        if cpg_full_tiles >= 3:
                            T.wait_flag("mte3", "mte2", 0)
                            T.copy(
                                x[n, g, 2 * block_C : 3 * block_C, 0:block_S],
                                data_buf_c0,
                            )
                            T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 1)
                        if use_fp32:
                            T.tile.cast(data_cal_c, data_buf_c1, CAST_LOW2HIGH, tile_elem_c)
                        else:
                            T.copy(data_buf_c1, data_cal_c)
                        T.copy(mean_bc[block_C : 2 * block_C, :], param_tile)
                        T.tile.sub(data_cal_c, data_cal_c, param_tile)
                        T.copy(std_bc[block_C : 2 * block_C, :], param_tile)
                        T.tile.div(data_cal_c, data_cal_c, param_tile)
                        T.copy(gamma_bc[block_C : 2 * block_C, :], param_tile)
                        T.tile.mul(data_cal_c, data_cal_c, param_tile)
                        T.copy(beta_bc[block_C : 2 * block_C, :], param_tile)
                        T.tile.add(data_cal_c, data_cal_c, param_tile)
                        if use_fp32:
                            T.tile.cast(out_buf_c1, data_cal_c, CAST_HIGH2LOW, tile_elem_c)
                        else:
                            T.copy(data_cal_c, out_buf_c1)
                        T.set_flag("v", "mte3", 1)
                        T.wait_flag("v", "mte3", 1)
                        T.copy(out_buf_c1, y[n, g, block_C : 2 * block_C, 0:block_S])
                        T.set_flag("mte3", "mte2", 1)

                    T.wait_flag("mte3", "mte2", 0)
                    T.wait_flag("mte3", "mte2", 1)

                    # Remainder handling (cpg_rem > 0)
                    if cpg_rem > 0:
                        off = cpg_full_tiles * block_C
                        rem_elem = cpg_rem * block_S
                        data_rem = T.alloc_ub([cpg_rem, block_S], dtype)
                        data_cal_rem = T.alloc_ub([cpg_rem, block_S], cal_dtype)
                        out_rem = T.alloc_ub([cpg_rem, block_S], dtype)
                        param_rem = T.alloc_ub([cpg_rem, block_S], cal_dtype)
                        T.copy(x[n, g, off:cpg, 0:block_S], data_rem)
                        T.barrier_all()
                        if use_fp32:
                            T.tile.cast(data_cal_rem, data_rem, CAST_LOW2HIGH, rem_elem)
                        else:
                            T.copy(data_rem, data_cal_rem)
                        T.copy(mean_bc[off:cpg, :], param_rem)
                        T.tile.sub(data_cal_rem, data_cal_rem, param_rem)
                        T.copy(std_bc[off:cpg, :], param_rem)
                        T.tile.div(data_cal_rem, data_cal_rem, param_rem)
                        T.copy(gamma_bc[off:cpg, :], param_rem)
                        T.tile.mul(data_cal_rem, data_cal_rem, param_rem)
                        T.copy(beta_bc[off:cpg, :], param_rem)
                        T.tile.add(data_cal_rem, data_cal_rem, param_rem)
                        if use_fp32:
                            T.tile.cast(out_rem, data_cal_rem, CAST_HIGH2LOW, rem_elem)
                        else:
                            T.copy(data_cal_rem, out_rem)
                        T.barrier_all()
                        T.copy(out_rem, y[n, g, off:cpg, 0:block_S])

    return main


def _find_block_S(S, cpg, dtype_str):
    UB_BUDGET = 192 * 1024
    cal_bytes = 4 if dtype_str in ("float16", "bfloat16") else int(dtype_str[-2:]) // 8
    dtype_bytes = cal_bytes if dtype_str == "float32" else 2
    c = max(cpg, 1)
    per_block = c * (6 * cal_bytes + 6 * dtype_bytes)
    max_block_S = (UB_BUDGET // per_block // 16) * 16
    max_block_S = min(max_block_S, 512)
    for bs in range(max_block_S, 0, -16):
        if S % bs == 0:
            return bs
    return max(16, max_block_S)


def group_norm(x, gamma, beta, num_groups, eps=1e-5):
    """GroupNorm host function: dispatches to serial or cpipeline kernel."""
    original_shape = x.shape
    N = x.shape[0]
    C = x.shape[1]
    S = 1
    for i in range(2, x.ndim):
        S *= x.shape[i]

    dtype_str = str(x.dtype).replace("torch.", "")
    cpg = C // num_groups

    cpg_padded = max(((cpg + 15) // 16) * 16, 16)
    block_S = _find_block_S(S, cpg, dtype_str)
    s_num = (S + block_S - 1) // block_S
    S_padded = s_num * block_S

    block_C = max(16, 2048 // max(block_S, 1)) if s_num == 1 else 0
    block_C = ((block_C // 16) * 16) if s_num == 1 else 0
    block_C = max(16, min(cpg, block_C)) if s_num == 1 else 0
    cpg_full_tiles = (cpg // block_C) if s_num == 1 else 0
    use_cpipeline = (s_num == 1 and cpg_full_tiles >= 2)

    x_4d = x.reshape(N, num_groups, cpg, S)
    gamma_2d = gamma.reshape(num_groups, cpg)
    beta_2d = beta.reshape(num_groups, cpg)

    if use_cpipeline:
        # Only pad when S < block_S to match kernel's memory access pattern
        if S < block_S:
            x_4d = torch.nn.functional.pad(x_4d, (0, block_S - S))
        func = group_norm_kernel_cpipeline(
            N, num_groups, cpg_padded, S_padded, block_S, eps, cpg, S, dtype_str
        )
    else:
        func = group_norm_kernel_serial(
            N, num_groups, cpg_padded, S_padded, block_S, s_num, eps, cpg, S, dtype_str
        )
    y_4d = func(x_4d, gamma_2d, beta_2d)

    y_4d = y_4d[:, :, :, :S]
    return y_4d.reshape(original_shape)


def golden_group_norm(x, gamma, beta, num_groups, eps=1e-5):
    """PyTorch reference implementation."""
    if x.ndim == 2:
        return torch.nn.functional.group_norm(
            x.unsqueeze(-1), num_groups, gamma, beta, eps
        ).squeeze(-1)
    return torch.nn.functional.group_norm(x, num_groups, gamma, beta, eps)


if __name__ == "__main__":
    tilelang.disable_cache()
    torch.manual_seed(42)

    thresholds = {
        "float16": (2**-10, 10 * 2**-10),
        "bfloat16": (2**-7, 10 * 2**-7),
        "float32": (2**-13, 10 * 2**-13),
    }

    test_cases = [
        # ("basic_4d_fp16", [8, 32, 64, 64], "float16", 8, 1e-5),
        # ("basic_4d_fp32", [4, 64, 128, 128], "float32", 16, 1e-5),
        # ("basic_4d_bf16", [2, 128, 256, 256], "bfloat16", 32, 1e-5),
        # ("group1_layernorm", [16, 257, 32, 31], "float16", 1, 1e-5),
        # ("group2_fp32", [8, 512, 17, 15], "float32", 2, 1e-5),
        # ("basic_3d_bf16", [64, 64, 128], "bfloat16", 4, 1e-5),
        # ("large_4d_fp16", [2, 256, 128, 128], "float16", 16, 1e-5),
        # ("group1_fp32", [16, 127, 31, 33], "float32", 1, 1e-6),
        # ("small_eps_bf16", [3, 64, 64, 64], "bfloat16", 8, 1e-3),
        # ("non_aligned", [7, 32, 63, 65], "float16", 4, 1e-4),
        # ("large_range_fp32", [3, 64, 127, 129], "float32", 8, 1e-4),
        # ("group6_bf16", [5, 48, 33, 65], "bfloat16", 6, 1e-4),
        ("basic_2d_fp16", [1023, 257], "float16", 1, 1e-6),
        # ("basic_5d_fp32", [2, 60, 5, 7, 480], "float32", 4, 1e-5),
        # ("inf_special_bf16", [4, 31, 251, 251], "bfloat16", 1, 1e-8),
        # ("nan_special_fp16", [2, 64, 67, 71], "float16", 8, 1e-7),
        # ("all_zeros_fp32", [8, 127, 33, 31], "float32", 1, 1e-4),
        # ("large_4d_bf16", [2, 256, 127, 129], "bfloat16", 16, 1e-5),
        # ("fp16_boundary", [4, 128, 255, 257], "float16", 32, 1e-3),
        # ("group3_fp32", [1, 513, 63, 63], "float32", 3, 1e-6),
    ]

    all_passed = True
    results = []

    for name, shape, dtype_str, num_groups, eps in test_cases:
        torch_dtype = getattr(torch, dtype_str)
        mere_thresh, mare_thresh = thresholds[dtype_str]

        if name == "all_zeros_fp32":
            x = torch.zeros(shape, dtype=torch_dtype, device="npu")
        elif name == "inf_special_bf16":
            x = torch.randn(shape, dtype=torch_dtype, device="npu")
            x.view(-1)[:100] = float("inf")
            x.view(-1)[100:200] = float("-inf")
        elif name == "nan_special_fp16":
            x = torch.randn(shape, dtype=torch_dtype, device="npu")
            x.view(-1)[:100] = float("nan")
        else:
            x = torch.randn(shape, dtype=torch_dtype, device="npu")

        C = shape[1]
        gamma = torch.randn(C, dtype=torch_dtype, device="npu")
        beta = torch.randn(C, dtype=torch_dtype, device="npu")

        try:
            y_kernel = group_norm(x, gamma, beta, num_groups, eps)
            torch.npu.synchronize()
        except Exception as e:
            print(f"[PRECISION_FAIL] {name}: Runtime error - {str(e)[:200]}")
            all_passed = False
            results.append((name, False))
            continue

        y_golden = golden_group_norm(x, gamma, beta, num_groups, eps)
        try:
            y_k = y_kernel.float().cpu()
        except Exception as e:
            print(f"[PRECISION_FAIL] {name}: CPU transfer error - {str(e)[:200]}")
            all_passed = False
            results.append((name, False))
            continue
        y_g = y_golden.float().cpu()

        k_nan = y_k.isnan()
        g_nan = y_g.isnan()
        k_inf = y_k.isinf()
        g_inf = y_g.isinf()

        if name in ("inf_special_bf16", "nan_special_fp16"):
            nan_match = (k_nan == g_nan).float().mean().item()
            inf_match = (k_inf == g_inf).float().mean().item()
            valid = ~(k_nan | g_nan | k_inf | g_inf)
            if valid.sum() > 0:
                abs_diff = (y_k[valid] - y_g[valid]).abs()
                mere = (abs_diff / y_g[valid].abs().clamp(min=1e-7)).mean().item()
                mare = abs_diff.max().item()
            else:
                mere, mare = 0.0, 0.0
            passed = nan_match > 0.99 and inf_match > 0.99 and mere < mere_thresh and mare < mare_thresh
        else:
            abs_diff = (y_k - y_g).abs()
            mere = (abs_diff / y_g.abs().clamp(min=1e-7)).mean().item()
            mare = abs_diff.max().item()
            passed = mere < mere_thresh and mare < mare_thresh
        status = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"
        if not passed:
            all_passed = False

        print(
            f"{status} {name}: shape={shape}, dtype={dtype_str}, groups={num_groups}, "
            f"MERE={mere:.6e} (thresh={mere_thresh:.6e}), "
            f"MARE={mare:.6e} (thresh={mare_thresh:.6e})"
        )
        results.append((name, passed))

    print()
    if all_passed:
        print("Test Passed!")
    else:
        failed = [r[0] for r in results if not r[1]]
        print(f"[PRECISION_FAIL] {len(failed)} test(s) failed: {failed}")
        exit(1)
