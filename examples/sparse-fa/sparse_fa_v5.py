# msprof op --kernel-name="main_kernel" python examples/sparse-fa/sparse_fa_v5.py

import math
import torch
import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

# 初始化环境
torch.set_default_device("npu")
torch.manual_seed(0)
tilelang.disable_cache()

# ---------------------------------------------------------------------------
# 常量与优化 Pass 配置
# ---------------------------------------------------------------------------
NEG_INF = -(2.0**30)

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,  # 使用手动规划与双缓冲
}


@tilelang.jit(
    pass_configs=PASS_CONFIGS,
)
def high_perf_mtgr_sparse_attn_kernel(
    heads,
    dim,
    kv_group=1,
    sm_scale=None,
    block_M=128,
    block_N=128,
    core_num=24,
    num_stages=14,
    cross_interval=2,
    max_splits=32,
    max_segs=4,
):
    sm_scale = (1.0 / dim) ** 0.5 if sm_scale is None else sm_scale
    dtype = "bfloat16"  # HBM 搬运动力最佳格式
    accum_dtype = "float32"  # 关键中间累加采用 FP32 防溢出

    batch = T.symbolic("batch")
    total_q = T.symbolic("total_q")
    total_seq_tiles = T.symbolic("total_seq_tiles")

    kv_heads = heads // kv_group
    half_M = block_M // 2

    # 跨核信号与单核内信号定义
    SEM_WS1_C2V = 0
    SEM_WS1_V2C = 1
    SEM_WS2_V2C = 2
    SEM_WS2_C2V = 3
    SEM_WS3_C2V = 4
    SEM_WS3_V2C = 5

    SIG_K_L1 = 0
    SIG_P_L1 = 1
    SIG_V_L1 = 2
    SIG_L0AB = 3
    SIG_L0C = 5
    SIG_IO_UB = 0
    SIG_S_HALF = 1

    @T.prim_func
    def main(
        Q: T.Tensor([total_q, heads, dim], dtype),
        K: T.Tensor([total_q, kv_heads, dim], dtype),
        V: T.Tensor([total_q, kv_heads, dim], dtype),
        Output: T.Tensor([total_q, heads, dim], dtype),
        q_seq_starts: T.Tensor([batch + 1], "int32"),
        split_points: T.Tensor([batch, max_splits], "int32"),
        tiles_prefix_sum: T.Tensor([batch + 1], "int32"),
        segment_offsets: T.Tensor([batch, max_segs + 1], "int32"),
        segment_rules: T.Tensor([max_segs], "int32"),
        # 使用动态流水线槽位
        workspace_1: T.Tensor([core_num, num_stages, block_M, block_N], dtype),
        workspace_2: T.Tensor([core_num, num_stages, block_M, block_N], dtype),
        workspace_3: T.Tensor([core_num, num_stages, block_M, dim], dtype),
        _dummy: T.Tensor([total_seq_tiles], "int32"),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)

            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1: make_zn_layout(v_l1),
                }
            )

            l0a = T.alloc_L0A([2, block_M, dim], dtype)
            l0b = T.alloc_L0B([2, dim, block_N], dtype)
            l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)

            acc_o = T.alloc_ub([half_M, dim], accum_dtype)
            r_factors = T.alloc_ub([num_stages, half_M, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, half_M, 1], accum_dtype)
            sumexp = T.alloc_ub([half_M, 1], accum_dtype)
            neg_sm = T.alloc_ub([2, half_M, 1], accum_dtype)

            io_buf = T.alloc_ub([half_M, block_N], dtype)
            acc_s_half = T.alloc_ub([half_M, block_N], dtype)
            work_ub = T.alloc_ub([half_M, block_N], accum_dtype)
            buf_2d = T.alloc_ub([half_M, block_N], accum_dtype) # mask_ub
            # buf_2d = T.alloc_ub([half_M, block_N], accum_dtype)

            # 用于存储平铺流水线的真实有效 k 索引
            valid_k_indices = T.alloc_ub([max_splits], "int32")
            b_i = T.alloc_var("int32", init=0)
            seg_id = T.alloc_var("int32", init=0)
            next_seg_first_tile = T.alloc_var("int32", init=0)
            valid_k_total = T.alloc_var("int32", init=0)
            scan_count = T.alloc_var("int32", init=0)
            scan_k_idx = T.alloc_var("int32", init=-1)

            total_tasks = total_seq_tiles * heads
            my_iters = T.if_then_else(
                cid < total_tasks,
                T.ceildiv(total_tasks - cid, core_num),
                0,
            )

            # =========================================================================
            # Scope C: Cube 核心 (负责张量搬运与矩阵乘)
            # =========================================================================
            with T.Scope("C"):
                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("M", "MTE1", SIG_L0AB)
                T.set_flag("M", "MTE1", SIG_L0AB + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)

                for core_index in T.serial(my_iters):
                    pid = cid + core_index * core_num
                    tile_id = pid // heads
                    h_i = pid % heads
                    h_kv = h_i // kv_group

                    b_i = 0
                    for _b in T.serial(batch):
                        b_i = T.if_then_else(tile_id >= tiles_prefix_sum[_b + 1], _b + 1, b_i)
                    s_local = tile_id - tiles_prefix_sum[b_i]
                    q_start = split_points[b_i, s_local]
                    q_end = split_points[b_i, s_local + 1]
                    q_tile_size = q_end - q_start

                    seg_id = 0
                    for _seg in T.serial(max_segs - 1):
                        seg_id = T.if_then_else(q_start >= segment_offsets[b_i, _seg + 1], _seg + 1, seg_id)

                    rule = segment_rules[seg_id]
                    q_packed_start = q_seq_starts[b_i] + q_start
                    seg_end_offset = segment_offsets[b_i, seg_id + 1]

                    next_seg_first_tile = s_local
                    for _k in T.serial(max_splits):
                        next_seg_first_tile = T.if_then_else(
                            (_k > s_local) & (split_points[b_i, _k] < seg_end_offset), _k, next_seg_first_tile
                        )
                    tiles_this_batch = tiles_prefix_sum[b_i + 1] - tiles_prefix_sum[b_i]
                    next_seg_first_tile = T.if_then_else(next_seg_first_tile == s_local, tiles_this_batch, next_seg_first_tile)
                    kv_iter_end = T.if_then_else(rule == 1, next_seg_first_tile, s_local + 1)

                    # 计算有效 K 切片数量（Cube 无法读写 UB，仅计数）
                    seg_start = segment_offsets[b_i, seg_id]
                    seg_end = segment_offsets[b_i, seg_id + 1]
                    max_row_pos = q_start + q_tile_size - 1
                    valid_k_total = 0
                    for k_i in T.serial(kv_iter_end):
                        kv_start_k = split_points[b_i, k_i]
                        process_cond = (
                            ((rule == 0) & (kv_start_k <= max_row_pos))
                            | ((rule == 1) & (kv_start_k < seg_end))
                            | ((rule == 2) & ((kv_start_k <= seg_start) | (kv_start_k <= max_row_pos)))
                        )
                        if process_cond:
                            valid_k_total += 1

                    # 载入 Q
                    T.copy(Q[q_packed_start : q_packed_start + q_tile_size, h_i, :], q_l1[:, :])
                    num_outer = T.ceildiv(valid_k_total, num_stages)

                    for k_outer in T.serial(num_outer):
                        _remaining = valid_k_total - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # ---------------------------------
                        # GEMM1: S = Q * K^T (写入 workspace_1)
                        # ---------------------------------
                        T.wait_cross_flag(SEM_WS1_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            scan_count = 0
                            scan_k_idx = -1
                            target = k_outer * num_stages + i
                            for k_scan in T.serial(kv_iter_end):
                                kv_scan_start = split_points[b_i, k_scan]
                                process_cond_scan = (
                                    ((rule == 0) & (kv_scan_start <= max_row_pos))
                                    | ((rule == 1) & (kv_scan_start < seg_end))
                                    | ((rule == 2) & ((kv_scan_start <= seg_start) | (kv_scan_start <= max_row_pos)))
                                )
                                scan_k_idx = T.if_then_else(process_cond_scan & (scan_count == target), k_scan, scan_k_idx)
                                scan_count = T.if_then_else(process_cond_scan, scan_count + 1, scan_count)
                            kv_start = split_points[b_i, scan_k_idx]
                            kv_size = split_points[b_i, scan_k_idx + 1] - kv_start
                            kv_packed_start = q_seq_starts[b_i] + kv_start

                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            T.copy(K[kv_packed_start : kv_packed_start + kv_size, h_kv, :], k_l1[:, :])
                            T.set_flag("MTE2", "MTE1", SIG_K_L1)

                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i < 2:
                                T.copy(q_l1, l0a[side, :, :])

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            T.copy(k_l1, l0b[side, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], workspace_1[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)

                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)

                        # ---------------------------------
                        # GEMM2: O = P * V (写入 workspace_3)
                        # ---------------------------------
                        T.wait_cross_flag(SEM_WS3_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            scan_count = 0
                            scan_k_idx = -1
                            target = k_outer * num_stages + i
                            for k_scan in T.serial(kv_iter_end):
                                kv_scan_start = split_points[b_i, k_scan]
                                process_cond_scan = (
                                    ((rule == 0) & (kv_scan_start <= max_row_pos))
                                    | ((rule == 1) & (kv_scan_start < seg_end))
                                    | ((rule == 2) & ((kv_scan_start <= seg_start) | (kv_scan_start <= max_row_pos)))
                                )
                                scan_k_idx = T.if_then_else(process_cond_scan & (scan_count == target), k_scan, scan_k_idx)
                                scan_count = T.if_then_else(process_cond_scan, scan_count + 1, scan_count)
                            kv_start = split_points[b_i, scan_k_idx]
                            kv_size = split_points[b_i, scan_k_idx + 1] - kv_start
                            kv_packed_start = q_seq_starts[b_i] + kv_start

                            T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                            T.copy(V[kv_packed_start : kv_packed_start + kv_size, h_kv, :], v_l1[:, :])
                            T.set_flag("MTE2", "MTE1", SIG_V_L1)

                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(workspace_2[cid, i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            T.copy(v_l1, l0b[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_V_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], workspace_3[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)

                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS3_C2V)
                        T.set_cross_flag("MTE2", SEM_WS2_C2V)

                # 回收初始化的 Signal
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB)
                T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                T.wait_flag("FIX", "M", SIG_L0C)
                T.wait_flag("FIX", "M", SIG_L0C + 1)

            # =========================================================================
            # Scope V: Vector 核心 (负责生成 Mask、Softmax 和累加降级)
            # =========================================================================
            with T.Scope("V"):
                T.set_cross_flag("MTE2", SEM_WS1_V2C)
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("MTE3", "V", SIG_S_HALF)

                for core_index in T.serial(my_iters):
                    pid = cid + core_index * core_num
                    tile_id = pid // heads
                    h_i = pid % heads
                    h_kv = h_i // kv_group

                    # 重新解析位置（必须与 Scope C 完全对齐）
                    b_i = 0
                    for _b in T.serial(batch):
                        b_i = T.if_then_else(tile_id >= tiles_prefix_sum[_b + 1], _b + 1, b_i)
                    s_local = tile_id - tiles_prefix_sum[b_i]
                    q_start = split_points[b_i, s_local]
                    q_end = split_points[b_i, s_local + 1]
                    q_tile_size = q_end - q_start

                    seg_id = 0
                    for _seg in T.serial(max_segs - 1):
                        seg_id = T.if_then_else(q_start >= segment_offsets[b_i, _seg + 1], _seg + 1, seg_id)

                    rule = segment_rules[seg_id]
                    q_packed_start = q_seq_starts[b_i] + q_start
                    seg_end_offset = segment_offsets[b_i, seg_id + 1]

                    next_seg_first_tile = s_local
                    for _k in T.serial(max_splits):
                        next_seg_first_tile = T.if_then_else(
                            (_k > s_local) & (split_points[b_i, _k] < seg_end_offset), _k, next_seg_first_tile
                        )
                    tiles_this_batch = tiles_prefix_sum[b_i + 1] - tiles_prefix_sum[b_i]
                    next_seg_first_tile = T.if_then_else(next_seg_first_tile == s_local, tiles_this_batch, next_seg_first_tile)
                    kv_iter_end = T.if_then_else(rule == 1, next_seg_first_tile, s_local + 1)

                    valid_k_total = 0
                    for k_i in T.serial(kv_iter_end):
                        kv_start = split_points[b_i, k_i]
                        seg_start = segment_offsets[b_i, seg_id]
                        seg_end = segment_offsets[b_i, seg_id + 1]
                        max_row_pos = q_start + q_tile_size - 1

                        process_cond = (
                            ((rule == 0) & (kv_start <= max_row_pos))
                            | ((rule == 1) & (kv_start < seg_end))
                            | ((rule == 2) & ((kv_start <= seg_start) | (kv_start <= max_row_pos)))
                        )
                        if process_cond:
                            valid_k_indices[valid_k_total] = k_i
                            valid_k_total += 1

                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(neg_sm, 2**30)

                    num_outer = T.ceildiv(valid_k_total, num_stages)
                    for k_outer in T.serial(num_outer):
                        _remaining = valid_k_total - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- Softmax Batch 处理 ---
                        T.wait_cross_flag(SEM_WS2_C2V)
                        for i in T.serial(batch_iters):
                            cur = i % 2
                            prv = 1 - cur

                            k_idx = valid_k_indices[k_outer * num_stages + i]
                            kv_start = split_points[b_i, k_idx]
                            kv_size = split_points[b_i, k_idx + 1] - kv_start
                            seg_start = segment_offsets[b_i, seg_id]
                            seg_end = segment_offsets[b_i, seg_id + 1]

                            # 【核心优化】计算掩盖 (Hiding Computation)：在等 Cube 前利用算力资源提前生成 MASK
                            T.tile.fill(buf_2d, NEG_INF)
                            for row in T.serial(half_M):
                                row_abs_pos = q_start + vid * half_M + row
                                if rule == 0:
                                    raw_len = row_abs_pos - kv_start + 1
                                    fill_len = T.if_then_else(raw_len < kv_size, raw_len, kv_size)
                                    fill_len = T.if_then_else(fill_len > 0, fill_len, 0)
                                    if fill_len > 0:
                                        T.tile.fill(buf_2d[row, 0:fill_len], 0.0)
                                elif rule == 1:
                                    raw_len = seg_end - kv_start
                                    fill_len = T.if_then_else(raw_len < kv_size, raw_len, kv_size)
                                    fill_len = T.if_then_else(fill_len > 0, fill_len, 0)
                                    if fill_len > 0:
                                        T.tile.fill(buf_2d[row, 0:fill_len], 0.0)
                                elif rule == 2:
                                    raw_len = seg_start - kv_start
                                    fill_len = T.if_then_else(raw_len < kv_size, raw_len, kv_size)
                                    fill_len = T.if_then_else(fill_len > 0, fill_len, 0)
                                    if fill_len > 0:
                                        T.tile.fill(buf_2d[row, 0:fill_len], 0.0)
                                    diag_col = row_abs_pos - kv_start
                                    if (diag_col >= 0) & (diag_col < kv_size):
                                        buf_2d[row, diag_col] = 0.0

                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS1_C2V)

                            T.copy(workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :], io_buf)
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.copy(io_buf, work_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB)

                            # 施加 Mask 到矩阵 S
                            T.tile.add(work_ub, work_ub, buf_2d)

                            T.reduce_max(work_ub, neg_sm[cur, :, :], dim=-1)
                            T.tile.mul(neg_sm[cur, :, :], neg_sm[cur, :, :], -sm_scale)
                            T.tile.min(neg_sm[cur, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])
                            T.tile.broadcast(buf_2d, neg_sm[cur, :, :])
                            T.tile.axpy(buf_2d, work_ub, sm_scale)
                            T.tile.exp(work_ub, buf_2d)

                            T.wait_flag("MTE3", "V", SIG_S_HALF)
                            T.copy(work_ub, acc_s_half)
                            T.set_flag("V", "MTE3", SIG_S_HALF)

                            T.wait_flag("V", "MTE3", SIG_S_HALF)
                            T.copy(acc_s_half, workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :])
                            T.set_flag("MTE3", "V", SIG_S_HALF)

                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("MTE3", SEM_WS2_V2C)

                            T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)
                            T.tile.sub(r_factors[i, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])
                        T.set_cross_flag("MTE2", SEM_WS1_V2C)

                        # --- O 累加 Batch ---
                        for i in T.serial(batch_iters):
                            T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])
                            T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                            T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                            T.tile.broadcast(buf_2d, r_factors[i, :, :])
                            T.tile.mul(acc_o, acc_o, buf_2d)

                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS3_C2V)
                            T.copy(workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :], io_buf[:, 0:dim])
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.copy(io_buf[:, 0:dim], work_ub[:, 0:dim])
                            T.set_flag("V", "MTE2", SIG_IO_UB)

                            T.tile.add(acc_o, acc_o, work_ub[:, 0:dim])

                        T.set_cross_flag("MTE2", SEM_WS3_V2C)

                    # 最终输出标准化
                    T.tile.max(sumexp, sumexp, 1.0)
                    T.tile.broadcast(buf_2d[:, 0:dim], sumexp)
                    T.tile.div(acc_o, acc_o, buf_2d[:, 0:dim])

                    T.copy(acc_o, acc_s_half[:, 0:dim])
                    T.barrier_all()

                    # 收尾与拷贝回 GM (支持不规则最后一块裁减)
                    valid_rows = T.if_then_else(
                        q_tile_size >= (vid + 1) * half_M,
                        half_M,
                        T.if_then_else(q_tile_size > vid * half_M, q_tile_size - vid * half_M, 0),
                    )

                    h_i_out = (cid + core_index * core_num) % heads
                    output_packed_start = q_packed_start + vid * half_M

                    T.copy(acc_s_half[0:valid_rows, 0:dim], Output[output_packed_start : output_packed_start + valid_rows, h_i_out, :])

                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("MTE3", "V", SIG_S_HALF)

    return main


# ---------------------------------------------------------------------------
# Host 端 Wrapper（集成动态长度推导与 Workspace 分配）
# ---------------------------------------------------------------------------
def compute_split_points(seg_lengths, block_M):
    splits = [0]
    for seg_id, length in enumerate(seg_lengths):
        seg_start = splits[-1]
        seg_end = seg_start + length
        pos = seg_start
        while pos < seg_end:
            next_pos = min(pos + block_M, seg_end)
            if next_pos not in splits:
                splits.append(next_pos)
            pos = next_pos
    return splits


def high_perf_sparse_attn_wrapper(
    query,
    key,
    value,
    segment_offsets_i32,
    segment_rules_i32,
    q_seq_starts_i32,
    sm_scale,
    block_M=128,
    block_N=128,
    core_num=24,
    num_stages=4,
    kv_group=1,
    cross_interval=2,
):
    B = segment_offsets_i32.size(0)
    H = query.size(1)
    D = query.size(2)

    seg_lengths_list = []
    for b in range(B):
        offsets = segment_offsets_i32[b].cpu().tolist()
        lengths = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
        seg_lengths_list.append(lengths)

    max_splits = 0
    max_segs = 0
    all_split_points = []

    for b in range(B):
        num_segs_b = len(seg_lengths_list[b])
        max_segs = max(max_segs, num_segs_b)
        splits = compute_split_points(seg_lengths_list[b], block_M)
        all_split_points.append(splits)
        max_splits = max(max_splits, len(splits))

    split_points_padded = []
    for sp in all_split_points:
        split_points_padded.append(sp + [0] * (max_splits - len(sp)))
    split_points_i32 = torch.tensor(split_points_padded, dtype=torch.int32, device=query.device)

    seg_offsets_padded = []
    for b in range(B):
        offsets = segment_offsets_i32[b].cpu().tolist()
        seg_offsets_padded.append(offsets + [0] * (max_segs + 1 - len(offsets)))
    segment_offsets_padded_i32 = torch.tensor(seg_offsets_padded, dtype=torch.int32, device=query.device)

    rules_padded = segment_rules_i32.cpu().tolist() + [0] * (max_segs - len(segment_rules_i32))
    segment_rules_padded_i32 = torch.tensor(rules_padded, dtype=torch.int32, device=query.device)

    tiles_per_batch = [len(sp) - 1 for sp in all_split_points]
    total_seq_tiles = sum(tiles_per_batch)

    tiles_prefix_sum_list = [0]
    for t in tiles_per_batch:
        tiles_prefix_sum_list.append(tiles_prefix_sum_list[-1] + t)
    tiles_prefix_sum_i32 = torch.tensor(tiles_prefix_sum_list, dtype=torch.int32, device=query.device)

    # Workspace 申请 （支持 num_stages 大小）
    # HBM 大约消耗 = 24 * 4 * 128 * 128 * 2(float16) = 3MB，这在显存中完全微不足道。
    ws1 = torch.empty((core_num, num_stages, block_M, block_N), dtype=torch.bfloat16, device=query.device)
    ws2 = torch.empty((core_num, num_stages, block_M, block_N), dtype=torch.bfloat16, device=query.device)
    ws3 = torch.empty((core_num, num_stages, block_M, D), dtype=torch.bfloat16, device=query.device)
    output = torch.empty_like(query)
    dummy_tensor = torch.empty(total_seq_tiles, dtype=torch.int32, device=query.device)

    print("开始编译kernel")
    # 获得编译好的 JIT Kernel
    func = high_perf_mtgr_sparse_attn_kernel(
        heads=H,
        dim=D,
        kv_group=kv_group,
        sm_scale=sm_scale,
        block_M=block_M,
        block_N=block_N,
        core_num=core_num,
        num_stages=num_stages,
        max_splits=max_splits,
        max_segs=max_segs,
        cross_interval=cross_interval
    )

    print(func.get_kernel_source()) # 可以解除注释打印算子源码验证

    func(
        query,
        key,
        value,
        output,
        q_seq_starts_i32,
        split_points_i32,
        tiles_prefix_sum_i32,
        segment_offsets_padded_i32,
        segment_rules_padded_i32,
        ws1,
        ws2,
        ws3,
        dummy_tensor,
    )

    torch.npu.synchronize()
    return output

def golden_attention(
    query_snd,
    key_snd,
    value_snd,
    segment_offsets_i32,
    segment_rules_i32,
    q_seq_starts_i32,
    matched_prefix_lens_i32,
    key_cache,
    value_cache,
    block_table_i32,
    block_size,
    sm_scale,
):
    B = segment_offsets_i32.size(0)
    D = query_snd.size(2)
    H = query_snd.size(1)
    kv_heads = key_snd.size(1)
    total_live_q = query_snd.size(0)

    q_starts = q_seq_starts_i32.cpu().tolist()
    matched_prefix_list = matched_prefix_lens_i32.cpu().tolist()

    q_lens = [q_starts[b + 1] - q_starts[b] for b in range(B)]

    ref_outputs = []
    for b in range(B):
        q_start = q_starts[b]
        q_len = q_lens[b]
        q_b = query_snd[q_start : q_start + q_len].float()

        prefix_len = matched_prefix_list[b]

        if prefix_len > 0:
            num_prefix_blocks = (prefix_len + block_size - 1) // block_size
            phys_blocks = block_table_i32[b, :num_prefix_blocks]
            k_prefix_b = key_cache[phys_blocks].reshape(-1, kv_heads, D)[:prefix_len].float()
            v_prefix_b = value_cache[phys_blocks].reshape(-1, kv_heads, D)[:prefix_len].float()

        k_live_b = key_snd[q_start : q_start + q_len].float()
        v_live_b = value_snd[q_start : q_start + q_len].float()

        if prefix_len > 0:
            k_full_b = torch.cat([k_prefix_b, k_live_b], dim=0)
            v_full_b = torch.cat([v_prefix_b, v_live_b], dim=0)
        else:
            k_full_b = k_live_b
            v_full_b = v_live_b

        total_kv_len_b = prefix_len + q_len

        offsets = segment_offsets_i32[b].cpu().tolist()
        rules = segment_rules_i32.cpu().tolist()

        logical_positions = prefix_len + torch.arange(q_len)
        offsets_tensor = torch.tensor(offsets, dtype=torch.int32)
        seg_ids = torch.searchsorted(offsets_tensor[1:], logical_positions, right=True)

        mask_b = torch.zeros(q_len, total_kv_len_b, dtype=torch.float32)
        for seg_id_val in range(len(rules)):
            rule = rules[seg_id_val]
            q_indices = (seg_ids == seg_id_val).nonzero().squeeze(-1)
            if q_indices.numel() == 0:
                continue
            if rule == 0:
                k_range = torch.arange(total_kv_len_b)
                causal_mask = k_range.unsqueeze(0) <= logical_positions[q_indices].unsqueeze(1)
                mask_b[q_indices] = causal_mask.float()
            elif rule == 1:
                end = offsets[seg_id_val + 1]
                mask_b[q_indices, :end] = 1.0
            elif rule == 2:
                start = offsets[seg_id_val]
                mask_b[q_indices, :start] = 1.0
                mask_b[q_indices, logical_positions[q_indices]] = 1.0

        scores = torch.einsum("qhd,khd->hqk", q_b, k_full_b) * sm_scale
        scores = scores.masked_fill(
            mask_b.unsqueeze(0) == 0.0,
            float("-inf"),
        )
        probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        ref_b = torch.einsum("hqk,khd->qhd", probs, v_full_b).to(torch.bfloat16)
        ref_outputs.append(ref_b)

    ref_output = torch.cat(ref_outputs, dim=0)
    return ref_output

def test(
    H,
    D,
    seg_lengths,
    rules,
    matched_prefix_arr,
    kv_group=1,
    block_M=128,
    block_N=128,
    block_size=128,
    core_num=24,
    num_stages=14,
    cross_interval=2,
):
    B = len(seg_lengths)
    kv_heads = H // kv_group

    S_logical_list = [sum(sl) for sl in seg_lengths]

    offsets_list = []
    for sl in seg_lengths:
        off = [0]
        for s in sl:
            off.append(off[-1] + s)
        offsets_list.append(off)

    actual_q_len_arr = [S_logical_list[b] - matched_prefix_arr[b] for b in range(B)]

    q_seq_starts_arr = [0]
    for b in range(1, B):
        q_seq_starts_arr.append(q_seq_starts_arr[b - 1] + actual_q_len_arr[b - 1])
    q_seq_starts_arr.append(q_seq_starts_arr[B - 1] + actual_q_len_arr[B - 1])

    q_list = []
    k_list_full = []
    v_list_full = []
    k_list_live = []
    v_list_live = []
    for b in range(B):
        q_b = torch.randn(actual_q_len_arr[b], H, D, dtype=torch.bfloat16)
        k_b_full = torch.randn(S_logical_list[b], kv_heads, D, dtype=torch.bfloat16)
        v_b_full = torch.randn(S_logical_list[b], kv_heads, D, dtype=torch.bfloat16)
        q_list.append(q_b)
        k_list_full.append(k_b_full)
        v_list_full.append(v_b_full)

        prefix_len = matched_prefix_arr[b]
        k_list_live.append(k_b_full[prefix_len:])
        v_list_live.append(v_b_full[prefix_len:])

    query_snd = torch.cat(q_list, dim=0)
    key_snd = torch.cat(k_list_live, dim=0)
    value_snd = torch.cat(v_list_live, dim=0)

    num_cache_blocks = sum((matched_prefix_arr[b] + block_size - 1) // block_size for b in range(B))
    key_cache = torch.zeros(num_cache_blocks, block_size, kv_heads, D, dtype=torch.bfloat16)
    value_cache = torch.zeros(num_cache_blocks, block_size, kv_heads, D, dtype=torch.bfloat16)

    block_table_arr = []
    physical_block_offset = 0
    for b in range(B):
        prefix_len = matched_prefix_arr[b]
        num_logical_blocks = (S_logical_list[b] + block_size - 1) // block_size
        bt_row = []
        for lb in range(num_logical_blocks):
            if lb < (prefix_len + block_size - 1) // block_size:
                bt_row.append(physical_block_offset + lb)
            else:
                bt_row.append(0)
        block_table_arr.append(bt_row)
        physical_block_offset += (prefix_len + block_size - 1) // block_size

    for b in range(B):
        prefix_len = matched_prefix_arr[b]
        if prefix_len > 0:
            for p in range(prefix_len):
                block_idx = p // block_size
                block_offset = p % block_size
                physical_block = block_table_arr[b][block_idx]
                key_cache[physical_block, block_offset, :, :] = k_list_full[b][p, :, :]
                value_cache[physical_block, block_offset, :, :] = v_list_full[b][p, :, :]

    max_blocks_per_request = max(len(bt) for bt in block_table_arr)
    block_table_tensor = torch.zeros(B, max_blocks_per_request, dtype=torch.int32)
    for b in range(B):
        for lb in range(len(block_table_arr[b])):
            block_table_tensor[b, lb] = block_table_arr[b][lb]

    segment_offsets_i32 = torch.tensor(offsets_list, dtype=torch.int32)
    segment_rules_i32 = torch.tensor(rules, dtype=torch.int32)
    q_seq_starts_i32 = torch.tensor(q_seq_starts_arr, dtype=torch.int32)
    matched_prefix_lens_i32 = torch.tensor(matched_prefix_arr, dtype=torch.int32)

    sm_scale = 1.0 / math.sqrt(D)

    torch.npu.synchronize()
    print("init successful!")

    output_snd = high_perf_sparse_attn_wrapper(
        query_snd,
        key_snd,
        value_snd,
        segment_offsets_i32,
        segment_rules_i32,
        q_seq_starts_i32,
        sm_scale,
        block_M=block_M,
        block_N=block_N,
        core_num=core_num,
        kv_group=kv_group,
        num_stages=num_stages,
        cross_interval=cross_interval,
    )

    ref_output = golden_attention(
        query_snd,
        key_snd,
        value_snd,
        segment_offsets_i32,
        segment_rules_i32,
        q_seq_starts_i32,
        matched_prefix_lens_i32,
        key_cache,
        value_cache,
        block_table_tensor,
        block_size,
        sm_scale,
    )

    torch.npu.synchronize()
    torch.testing.assert_close(ref_output, output_snd, rtol=1e-2, atol=1e-2)
    print("Test Passed!")


if __name__ == "__main__":
    test_configs = [
        {
            "H": 8,
            "D": 128,
            "seg_lengths": [
                [2200, 8, 200, 1024],
                [1700, 8, 300, 1100],
                [2440, 8, 200, 2048],
                [1600, 8, 600, 1800],
                [3300, 8, 200, 1300],
                [1700, 8, 300, 2048],
                [1780, 8, 300, 1024],
                [2048, 8, 500, 1800],
            ],
            "rules": [0, 1, 0, 2],
            "matched_prefix_arr": [0, 0, 0, 0, 0, 0, 0, 0],
        },
    ]

    for config in test_configs:
        test(**config)
