import argparse
from collections import Counter

import torch
import tilelang
import tilelang.language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}


def _get_cube_core_num() -> int:
    """检测当前设备 Cube 核心数（与 DeepSeek 同模式）。"""
    try:
        import torch_npu as _tnpu

        return _tnpu.npu.get_device_properties(0).cube_core_num
    except Exception:
        return 20


def auto_s2_splits(batch, s1, s2, block_n, max_cores=None):
    """Host 侧自动选择 S2 切分数：最小化 Phase 1 关键路径 + 并行 Phase 2 总成本。

    成本模型（单位≈1 trunk 或 1 merge 的时间，实测约等价）：
      cost(sp) = items(sp) × tps(sp)               # Phase 1 关键路径
               + rows_per_aiv(sp) × merge_calls     # Phase 2（并行均分，所有 AIV）
    其中 merge_calls 与 sp 无关（每行仍归并全部 trunk），rows_per_aiv 随 grid 增大而减小。
    splits==1 时 Phase 2 由 owner（=每 block）独立执行且无 barrier，Phase 2 成本不同。
    """
    if max_cores is None:
        max_cores = _get_cube_core_num()
    base = batch * (s1 // 64)
    trunks_total = s2 // block_n
    merge_calls = (trunks_total - 1 + 2) // 3

    def _items(sp):
        total = base * sp
        grid = min(total, max_cores)
        return (total + grid - 1) // grid

    def _crit(sp):
        return _items(sp) * (trunks_total // sp)

    def _p2_rows(sp):
        if sp > 1:
            grid = min(base * sp, max_cores)
            return (base * 64 + grid * 2 - 1) // (grid * 2)
        # splits==1：owner 即全部 block，每 AIV 处理 VID_ROWS=32 行。
        return 32

    def _cost(sp):
        barrier = 1.0 if sp > 1 else 0.0
        return _crit(sp) + _p2_rows(sp) * merge_calls + barrier

    # 基线：不切分。切分需同时满足（实测校准）：
    #   ① Phase 1 关键路径降幅 ≥ 2 单位（barrier+不均衡+流水线排空约 1.5-2 单位，
    #      降幅不足时净退化：B2/S2=1024 crit 不变 +71us、S2=2048 降 1 单位 +56us）
    #   ② 含 barrier 的总成本下降
    #   ③ 每 block 工作项 ≤ 4
    base_crit = _crit(1)
    base_cost = _cost(1)
    best_splits, best_cost = 1, base_cost
    for sp in range(2, trunks_total + 1):
        if s2 % (sp * block_n) != 0:
            continue
        if _items(sp) > 4:
            continue
        if _crit(sp) > base_crit - 2:
            continue
        cost = _cost(sp)
        if cost < best_cost:
            best_splits, best_cost = sp, cost
    return best_splits


@tilelang.jit(out_idx=[-1], workspace_idx=[-4, -3], pass_configs=pass_configs, target="ascendc")
def indexer(
    N2,
    G,
    D,
    TOP_K,
    VECTOR_BASEN,
    VECTOR_BASEG,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    MAX_S2=4096,
    input_dtype="float16",
    calc_dtype="float",
    s2_splits=1,
    max_cores=None,
):
    # BLOCK_N 约束：≤256 走单 GEMM；=512 走双 256 列子 GEMM（gemm_v0 单次 L0B 64KB 限制）。
    if BLOCK_N not in (64, 128, 256, 512):
        raise ValueError(f"BLOCK_N 仅支持 64/128/256/512，当前 {BLOCK_N}")
    DUAL_GEMM = BLOCK_N > 256
    GEMM_COLS = min(BLOCK_N, 256)
    SECOND_COLS = max(BLOCK_N - 256, 1)
    if BLOCK_M % 2:
        raise ValueError("双 AIV 模式要求 BLOCK_M 为偶数")
    if calc_dtype != "float":
        raise ValueError("row_expand_mul 仅支持 float32 列块（256B/行）")
    if G % 8 or not 8 <= G <= 248:
        raise ValueError(f"row_expand_mul 要求 G 为 8..248 且整除 8，当前 {G}")
    if VECTOR_BASEN % 64:
        raise ValueError(f"row_expand_mul 要求 VECTOR_BASEN 整除 64（fp32 256B 列块），当前 {VECTOR_BASEN}")
    VID_ROWS = BLOCK_M // 2
    # 每 trunk 只需保留 top min(TOP_K, BLOCK_N) 对：全局 top-K 的候选必在各自 trunk
    # 的 top-K 内。截断同时修复 BLOCK_N > TOP_K 时 Phase 2 越界写 history_ub 的隐患。
    TRUNK_KEEP = min(TOP_K, BLOCK_N)
    B = T.symbolic("B")
    S1 = T.symbolic("S1")
    S2 = T.symbolic("S2")
    S1_TILES = S1 // BLOCK_M
    TASK_COUNT = B * N2 * S1_TILES
    TRUNKS_MAX = MAX_S2 // BLOCK_N
    if MAX_S2 % BLOCK_N:
        raise ValueError(f"MAX_S2={MAX_S2} 必须能被 BLOCK_N={BLOCK_N} 整除")
    # 持久化 kernel：固定 grid ≤ 核心数，每 block 串行处理多个工作项。
    if max_cores is None:
        max_cores = _get_cube_core_num()
    S2_SPLITS = s2_splits
    TOTAL_WORK = TASK_COUNT * S2_SPLITS
    GRID_SIZE = T.min(TOTAL_WORK, max_cores)
    ITEMS_PER_BLOCK = (TOTAL_WORK + GRID_SIZE - 1) // GRID_SIZE
    READY_FLAG = 0
    FREE_FLAG = 2
    # 本地事件 ID 按有向管线对独立分配，910B/A3 的合法范围为 0 到 7。
    L0C_FLAG = 1  # 避免与 GEMM 内部使用的事件 ID 0 混淆。
    K_L1_READY_FLAG = 0
    Q_L1_READY_FLAG = 1
    OUTPUT_READY_FLAG = 1
    HISTORY_LOAD_FLAG = 1
    HISTORY_STORE_FLAG = 0
    K_L1_FREE_FLAG = 0
    Q_L1_FREE_FLAG = 1
    G_REDUCE_FLAG = 0
    SLOT_RELEASE_FLAG = 2

    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N2, G * D), input_dtype),
        KEY: T.Tensor((B, S2, N2, D), input_dtype),
        QK_SLOT: T.Tensor((GRID_SIZE, 2, BLOCK_M, G, BLOCK_N), calc_dtype),
        SORTED_WORKSPACE: T.Tensor((TASK_COUNT, BLOCK_M, TRUNKS_MAX, 2 * TRUNK_KEEP), calc_dtype),
        WEIGHTS: T.Tensor((B, S1, N2, G), calc_dtype),
        OUT: T.Tensor((B, N2, S1, TOP_K), "int32"),
    ):
        with T.Kernel(GRID_SIZE, is_npu=True) as (cid, vid):
            real_start = cid * ITEMS_PER_BLOCK

            with T.Scope("C"):
                q_l1 = T.alloc_L1((BLOCK_M, BLOCK_K), input_dtype)
                # 双 GEMM 分裂：k_l1_b 在单 GEMM 模式下为哑元，不参与计算。
                k_l1_a = T.alloc_L1((GEMM_COLS, BLOCK_K), input_dtype)
                k_l1_b = T.alloc_L1((SECOND_COLS, BLOCK_K), input_dtype)
                c_l0 = T.alloc_L0C((BLOCK_M, GEMM_COLS), calc_dtype)
                T.set_flag("FIX", "M", L0C_FLAG)
                T.set_flag("M", "MTE2", K_L1_FREE_FLAG)
                T.set_flag("M", "MTE2", Q_L1_FREE_FLAG)

                # ===== Phase 1：串行工作项（group × S2 split），每项完整 C/V 流水线 =====
                for wi in T.serial(ITEMS_PER_BLOCK):
                    gwi = real_start + wi
                    if gwi < TOTAL_WORK:
                        group = gwi // S2_SPLITS
                        split = gwi % S2_SPLITS
                        gb = group // (N2 * S1_TILES)
                        gn2 = (group // S1_TILES) % N2
                        gm = group % S1_TILES
                        s2_start = split * (S2 // S2_SPLITS)

                        for n in T.serial(S2 // S2_SPLITS // BLOCK_N):
                            slot = n % 2
                            T.wait_cross_flag(FREE_FLAG + slot)
                            T.wait_flag("M", "MTE2", K_L1_FREE_FLAG)
                            # K 装载：第一段始终执行；双 GEMM 时第二段跟随（MTE2 FIFO，一个 ready 覆盖）。
                            T.copy(
                                KEY[gb, s2_start + n * BLOCK_N : s2_start + n * BLOCK_N + GEMM_COLS, gn2, 0:D],
                                k_l1_a,
                            )
                            if DUAL_GEMM:
                                T.copy(
                                    KEY[gb, s2_start + n * BLOCK_N + GEMM_COLS : s2_start + (n + 1) * BLOCK_N, gn2, 0:D],
                                    k_l1_b,
                                )
                            T.set_flag("MTE2", "M", K_L1_READY_FLAG)
                            T.wait_flag("MTE2", "M", K_L1_READY_FLAG)
                            for g in T.serial(G):
                                T.wait_flag("M", "MTE2", Q_L1_FREE_FLAG)
                                T.copy(
                                    Query[
                                        gb,
                                        gm * BLOCK_M : (gm + 1) * BLOCK_M,
                                        gn2,
                                        g * D : (g + 1) * D,
                                    ],
                                    q_l1,
                                )
                                T.set_flag("MTE2", "M", Q_L1_READY_FLAG)
                                T.wait_flag("MTE2", "M", Q_L1_READY_FLAG)
                                # 子 GEMM A（列 0 至 GEMM_COLS-1）：始终执行。
                                T.wait_flag("FIX", "M", L0C_FLAG)
                                T.gemm_v0(q_l1, k_l1_a, c_l0, transpose_B=True, init=True)
                                T.set_flag("M", "FIX", L0C_FLAG)
                                T.wait_flag("M", "FIX", L0C_FLAG)
                                T.copy(c_l0, QK_SLOT[cid, slot, :, g, 0:GEMM_COLS], enable_relu=True)
                                T.set_flag("FIX", "M", L0C_FLAG)
                                if DUAL_GEMM:
                                    # 子 GEMM B（列 GEMM_COLS 至末尾）：Q 在此 GEMM 后才释放。
                                    T.wait_flag("FIX", "M", L0C_FLAG)
                                    T.gemm_v0(q_l1, k_l1_b, c_l0, transpose_B=True, init=True)
                                    T.set_flag("M", "FIX", L0C_FLAG)
                                    T.wait_flag("M", "FIX", L0C_FLAG)
                                    T.set_flag("M", "MTE2", Q_L1_FREE_FLAG)
                                    T.copy(c_l0, QK_SLOT[cid, slot, :, g, GEMM_COLS:BLOCK_N], enable_relu=True)
                                    T.set_flag("FIX", "M", L0C_FLAG)
                                else:
                                    T.set_flag("M", "MTE2", Q_L1_FREE_FLAG)
                            T.set_flag("M", "MTE2", K_L1_FREE_FLAG)
                            T.set_cross_flag("FIX", READY_FLAG + slot)

                T.wait_flag("FIX", "M", L0C_FLAG)
                T.wait_flag("M", "MTE2", K_L1_FREE_FLAG)
                T.wait_flag("M", "MTE2", Q_L1_FREE_FLAG)
                if S2_SPLITS > 1:
                    # 全核屏障：仅 splits>1 时需要（owner 读其他 block 的排序结果）。
                    T.sync_all()

            with T.Scope("V"):
                # Phase 1/Phase 2 拆分：trunk 主序仅算+排序+存 GM；行主序 history 常驻 UB 批量 4-way 归并。
                mm_res_ub = T.alloc_ub((G, VECTOR_BASEN), calc_dtype)
                weight_ub = T.alloc_ub(G, calc_dtype)
                expand_tmp_ub = T.alloc_ub(G * 8, calc_dtype)
                reduce_g_ub = T.alloc_ub(VECTOR_BASEN, calc_dtype)
                sorted_block_ub = T.alloc_ub((1, 2 * BLOCK_N), calc_dtype)
                history_ub = T.alloc_ub(2 * TOP_K, calc_dtype)
                sort_src0_ub = T.alloc_ub(2 * TOP_K, calc_dtype)
                sort_src1_ub = T.alloc_ub(2 * TOP_K, calc_dtype)
                sort_src2_ub = T.alloc_ub(2 * TOP_K, calc_dtype)
                merged_ub = T.alloc_ub(8 * TOP_K, calc_dtype)
                index_lane_ub = T.alloc_ub(2 * BLOCK_N, calc_dtype)
                topk_index_ub = T.alloc_ub(TOP_K, calc_dtype)
                output_ub = T.alloc_ub(TOP_K, "int32")

                # 两个槽在首个 Cube 写入前均可复用。
                T.set_cross_flag("MTE2", FREE_FLAG)
                T.set_cross_flag("MTE2", FREE_FLAG + 1)
                T.tile.fill(index_lane_ub, 0)
                T.pipe_barrier("V")
                for index_offset in range(BLOCK_N):
                    index_lane_ub[index_offset * 2 + 1] = T.cast(1, calc_dtype)

                # ===== Phase 1：串行工作项，trunk 主序（算+排序+存 GM）=====
                for wi in T.serial(ITEMS_PER_BLOCK):
                    gwi = real_start + wi
                    if gwi < TOTAL_WORK:
                        group = gwi // S2_SPLITS
                        split = gwi % S2_SPLITS
                        s2_start = split * (S2 // S2_SPLITS)
                        trunk_base = split * (S2 // S2_SPLITS // BLOCK_N)
                        gb = group // (N2 * S1_TILES)
                        gn2 = (group // S1_TILES) % N2
                        gm = group % S1_TILES

                        for n in T.serial(S2 // S2_SPLITS // BLOCK_N):
                            slot = n % 2
                            T.wait_cross_flag(READY_FLAG + slot)
                            # 行 0 预取：MTE2 立即开始搬运（mm_res 此刻必然空闲），
                            # 搬运与循环前的 flag 开销重叠。
                            T.copy(
                                QK_SLOT[
                                    cid,
                                    slot,
                                    vid * VID_ROWS,
                                    0:G,
                                    0:VECTOR_BASEN,
                                ],
                                mm_res_ub,
                            )
                            T.copy(
                                WEIGHTS[gb, gm * BLOCK_M + vid * VID_ROWS, gn2, 0:G],
                                weight_ub,
                            )
                            T.set_flag("MTE2", "V", G_REDUCE_FLAG)
                            # 双 AIV：每个 AIV 只处理自己的一半行（vid 分工）。
                            for s1_local in T.serial(VID_ROWS):
                                s1_offset = vid * VID_ROWS + s1_local
                                s1_id = gm * BLOCK_M + s1_offset
                                T.wait_flag("MTE2", "V", G_REDUCE_FLAG)
                                # 64 元素（256B）列块 fused 广播乘：brcb 权重 + 单条
                                # mul_mask 覆盖全部 G 行，消除原 G 循环每迭代的
                                # PipeBarrier + GetValue 标量同步 + MOVEMASK 掩码开销
                                # （simulator 指令 trace：MOVEMASK 占 VECTOR 管线 56%），
                                # 且额外 UB 仅 G*8 元素（1KB），远小于 (G,BASEN) 广播
                                # 缓冲（64KB，BN=512 时会超 A2/A3 196KB UB 上限）。
                                for c in T.serial(VECTOR_BASEN // 64):
                                    T.tile.row_expand_mul_experiment(
                                        mm_res_ub[:, c * 64 : (c + 1) * 64],
                                        mm_res_ub[:, c * 64 : (c + 1) * 64],
                                        weight_ub,
                                        tmp=expand_tmp_ub,
                                    )
                                T.reduce_sum(mm_res_ub, reduce_g_ub, 0)
                                T.set_flag("V", "MTE2", G_REDUCE_FLAG)
                                T.tile.sort(sorted_block_ub, reduce_g_ub, BLOCK_N)
                                T.tile.axpy(
                                    sorted_block_ub,
                                    index_lane_ub,
                                    T.cast(s2_start + n * BLOCK_N, calc_dtype),
                                )
                                # 跨行预取：sort/axpy 已发射（V 管线在 flag 唤醒期间
                                # 继续执行），此处发射下一行 64KB 拷贝，MTE2 与本行
                                # sort 执行 + MTE3 store 重叠，消除逐行拷贝的暴露等待。
                                if s1_local + 1 < VID_ROWS:
                                    T.wait_flag("V", "MTE2", G_REDUCE_FLAG)
                                    T.copy(
                                        QK_SLOT[
                                            cid,
                                            slot,
                                            s1_offset + 1,
                                            0:G,
                                            0:VECTOR_BASEN,
                                        ],
                                        mm_res_ub,
                                    )
                                    T.copy(
                                        WEIGHTS[gb, s1_id + 1, gn2, 0:G],
                                        weight_ub,
                                    )
                                    T.set_flag("MTE2", "V", G_REDUCE_FLAG)
                                # 排序结果（有效前缀）存 GM（全局 trunk 索引），Phase 2 负责填充 -inf 尾部。
                                T.set_flag("V", "MTE3", HISTORY_STORE_FLAG)
                                T.wait_flag("V", "MTE3", HISTORY_STORE_FLAG)
                                T.copy(
                                    sorted_block_ub[0, 0 : 2 * TRUNK_KEEP],
                                    SORTED_WORKSPACE[group, s1_offset, trunk_base + n, 0 : 2 * TRUNK_KEEP],
                                )
                                T.set_flag("MTE3", "V", HISTORY_STORE_FLAG)
                                T.wait_flag("MTE3", "V", HISTORY_STORE_FLAG)
                            # trunk 末行 compute 释放的收尾等待：保持 (V→MTE2) 事件
                            # set/wait 逐 trunk 配平，避免跨 trunk 的陈旧 set 触发
                            # 下一 trunk 预取提前发射（数据竞争）。
                            T.wait_flag("V", "MTE2", G_REDUCE_FLAG)
                            T.set_flag("V", "MTE2", SLOT_RELEASE_FLAG + slot)
                            T.wait_flag("V", "MTE2", SLOT_RELEASE_FLAG + slot)
                            T.set_cross_flag("MTE2", FREE_FLAG + slot)

                if S2_SPLITS > 1:
                    # 全核屏障：Phase 1 全部完成后 owner 才能读其他 block 的排序结果。
                    T.sync_all()

                # ===== Phase 2 =====
                num_trunks = S2 // BLOCK_N
                num_batches = (num_trunks - 1 + 2) // 3
                if S2_SPLITS > 1:
                    # 并行 Phase 2：barrier 后全部 block×AIV 均分所有 group 的行归并，
                    # 替代仅 owner 归并（消除 #3 owner 集中：B=1 时 16→48 个 AIV 参与）。
                    total_rows = TASK_COUNT * BLOCK_M
                    total_aivs = GRID_SIZE * 2
                    rows_per_aiv = (total_rows + total_aivs - 1) // total_aivs
                    aiv_flat = cid * 2 + vid

                    for row_local in T.serial(rows_per_aiv):
                        global_row = aiv_flat * rows_per_aiv + row_local
                        if global_row < total_rows:
                            group = global_row // BLOCK_M
                            s1_offset = global_row % BLOCK_M
                            gb = group // (N2 * S1_TILES)
                            gn2 = (group // S1_TILES) % N2
                            gm = group % S1_TILES
                            s1_id = gm * BLOCK_M + s1_offset
                            # trunk 0 直接载入 history（-inf 填充 + 有效前缀）。
                            T.tile.fill(history_ub, -T.infinity(calc_dtype))
                            T.set_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                            T.wait_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                            T.copy(SORTED_WORKSPACE[group, s1_offset, 0, 0 : 2 * TRUNK_KEEP], history_ub[0 : 2 * TRUNK_KEEP])
                            T.set_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                            T.wait_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                            # 剩余 trunk 按 3 个一批做 4-way 归并（history + 3 源等长 TOP_K 对）。
                            for batch in T.serial(num_batches):
                                base = 1 + batch * 3
                                # 源 0：本批首个 trunk，始终存在。
                                T.tile.fill(sort_src0_ub, -T.infinity(calc_dtype))
                                T.set_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                T.wait_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                T.copy(
                                    SORTED_WORKSPACE[group, s1_offset, base, 0 : 2 * TRUNK_KEEP],
                                    sort_src0_ub[0 : 2 * TRUNK_KEEP],
                                )
                                T.set_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                T.wait_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                # 源 1：存在才装载。
                                if base + 1 < num_trunks:
                                    T.tile.fill(sort_src1_ub, -T.infinity(calc_dtype))
                                    T.set_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                    T.wait_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                    T.copy(
                                        SORTED_WORKSPACE[group, s1_offset, base + 1, 0 : 2 * TRUNK_KEEP],
                                        sort_src1_ub[0 : 2 * TRUNK_KEEP],
                                    )
                                    T.set_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                    T.wait_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                # 源 2：存在才装载。
                                if base + 2 < num_trunks:
                                    T.tile.fill(sort_src2_ub, -T.infinity(calc_dtype))
                                    T.set_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                    T.wait_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                    T.copy(
                                        SORTED_WORKSPACE[group, s1_offset, base + 2, 0 : 2 * TRUNK_KEEP],
                                        sort_src2_ub[0 : 2 * TRUNK_KEEP],
                                    )
                                    T.set_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                    T.wait_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                # 按实际源数选择归并路数（AIV MrgSort 要求等长源）。
                                if base + 2 < num_trunks:
                                    T.tile.merge_sort(merged_ub, history_ub, sort_src0_ub, sort_src1_ub, sort_src2_ub)
                                elif base + 1 < num_trunks:
                                    T.tile.merge_sort(merged_ub, history_ub, sort_src0_ub, sort_src1_ub)
                                else:
                                    T.tile.merge_sort(merged_ub, history_ub, sort_src0_ub)
                                T.copy(merged_ub[0 : 2 * TOP_K], history_ub)
                            # 输出：直接从 history_ub 抽取索引。
                            T.tile.gather_mask(topk_index_ub, history_ub, "P1010")
                            T.tile.cast(output_ub, topk_index_ub, "CAST_ROUND", TOP_K)
                            T.set_flag("V", "MTE3", OUTPUT_READY_FLAG)
                            T.wait_flag("V", "MTE3", OUTPUT_READY_FLAG)
                            T.copy(output_ub, OUT[gb, gn2, s1_id, 0:TOP_K])
                            T.set_flag("MTE3", "V", OUTPUT_READY_FLAG)
                            T.wait_flag("MTE3", "V", OUTPUT_READY_FLAG)
                else:
                    # splits==1：per-block 独立归并（无 barrier，只读本 block 写入的数据）。
                    for wi in T.serial(ITEMS_PER_BLOCK):
                        gwi = real_start + wi
                        if gwi < TOTAL_WORK:
                            group = gwi // S2_SPLITS
                            gb = group // (N2 * S1_TILES)
                            gn2 = (group // S1_TILES) % N2
                            gm = group % S1_TILES

                            for s1_local in T.serial(VID_ROWS):
                                s1_offset = vid * VID_ROWS + s1_local
                                s1_id = gm * BLOCK_M + s1_offset
                                # trunk 0 直接载入 history（-inf 填充 + 有效前缀）。
                                T.tile.fill(history_ub, -T.infinity(calc_dtype))
                                T.set_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                T.wait_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                T.copy(SORTED_WORKSPACE[group, s1_offset, 0, 0 : 2 * TRUNK_KEEP], history_ub[0 : 2 * TRUNK_KEEP])
                                T.set_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                T.wait_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                # 剩余 trunk 按 3 个一批做 4-way 归并。
                                for batch in T.serial(num_batches):
                                    base = 1 + batch * 3
                                    T.tile.fill(sort_src0_ub, -T.infinity(calc_dtype))
                                    T.set_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                    T.wait_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                    T.copy(
                                        SORTED_WORKSPACE[group, s1_offset, base, 0 : 2 * TRUNK_KEEP],
                                        sort_src0_ub[0 : 2 * TRUNK_KEEP],
                                    )
                                    T.set_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                    T.wait_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                    if base + 1 < num_trunks:
                                        T.tile.fill(sort_src1_ub, -T.infinity(calc_dtype))
                                        T.set_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                        T.wait_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                        T.copy(
                                            SORTED_WORKSPACE[group, s1_offset, base + 1, 0 : 2 * TRUNK_KEEP],
                                            sort_src1_ub[0 : 2 * TRUNK_KEEP],
                                        )
                                        T.set_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                        T.wait_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                    if base + 2 < num_trunks:
                                        T.tile.fill(sort_src2_ub, -T.infinity(calc_dtype))
                                        T.set_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                        T.wait_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                        T.copy(
                                            SORTED_WORKSPACE[group, s1_offset, base + 2, 0 : 2 * TRUNK_KEEP],
                                            sort_src2_ub[0 : 2 * TRUNK_KEEP],
                                        )
                                        T.set_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                        T.wait_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                    if base + 2 < num_trunks:
                                        T.tile.merge_sort(merged_ub, history_ub, sort_src0_ub, sort_src1_ub, sort_src2_ub)
                                    elif base + 1 < num_trunks:
                                        T.tile.merge_sort(merged_ub, history_ub, sort_src0_ub, sort_src1_ub)
                                    else:
                                        T.tile.merge_sort(merged_ub, history_ub, sort_src0_ub)
                                    T.copy(merged_ub[0 : 2 * TOP_K], history_ub)
                                # 输出。
                                T.tile.gather_mask(topk_index_ub, history_ub, "P1010")
                                T.tile.cast(output_ub, topk_index_ub, "CAST_ROUND", TOP_K)
                                T.set_flag("V", "MTE3", OUTPUT_READY_FLAG)
                                T.wait_flag("V", "MTE3", OUTPUT_READY_FLAG)
                                T.copy(output_ub, OUT[gb, gn2, s1_id, 0:TOP_K])
                                T.set_flag("MTE3", "V", OUTPUT_READY_FLAG)
                                T.wait_flag("MTE3", "V", OUTPUT_READY_FLAG)

    return main


def index_golden(q, k, weights, top_k):
    score_1 = torch.einsum("bsmgd, btmd->bmsgt", q, k).relu()
    score = score_1.permute(0, 2, 1, 3, 4)
    reduce_res = torch.sum(score * weights[..., None], dim=3)
    golden_out = torch.topk(reduce_res, top_k, dim=3, largest=True, sorted=True)
    return golden_out.indices.to(torch.int32).permute(0, 2, 1, 3)


def count_index_multiset_mismatches(expected, actual):
    if expected.shape != actual.shape:
        raise ValueError("输出索引张量形状不一致")
    total_mismatches = 0
    expected_rows = expected.reshape(-1, expected.shape[-1])
    actual_rows = actual.reshape(-1, actual.shape[-1])
    for expected_row, actual_row in zip(expected_rows, actual_rows):
        expected_counter = Counter(expected_row.tolist())
        actual_counter = Counter(actual_row.tolist())
        difference = (expected_counter - actual_counter) + (actual_counter - expected_counter)
        total_mismatches += sum(difference.values())
    return total_mismatches


def test_indexer(
    s1,
    s2,
    top_k,
    block_n=256,
    vector_basen=256,
    batch=1,
    dimension=64,
    block_k=None,
    groups=32,
    n2=1,
    s2_splits=None,
):
    # BLOCK_K 默认等于 D：KEY 拷贝按 D 宽度装载到 (BLOCK_N, BLOCK_K) 缓冲。
    if block_k is None:
        block_k = dimension
    block_m = 64
    vector_baseg = 16
    if s1 % block_m or s2 % block_n or groups % vector_baseg:
        raise ValueError("当前实现仅支持 S1、S2 和 G 分别整除 BLOCK_M、BLOCK_N、VECTOR_BASEG")
    if top_k > s2:
        raise ValueError("TOP_K 不能大于 S2")

    if s2_splits is None:
        s2_splits = auto_s2_splits(batch, s1, s2, block_n)
    if s2 % (s2_splits * block_n):
        raise ValueError(f"S2={s2} 不能被 S2_SPLITS×BLOCK_N={s2_splits * block_n} 整除")

    torch.manual_seed(2)
    func = indexer(
        n2,
        groups,
        dimension,
        top_k,
        vector_basen,
        vector_baseg,
        block_m,
        block_n,
        block_k,
        MAX_S2=s2,
        s2_splits=s2_splits,
    )
    query = torch.randn(batch, s1, n2, groups, dimension).half()
    key = torch.randn(batch, s2, n2, dimension).half()
    weights = torch.randn(batch, s1, n2, groups).float()
    golden_out = index_golden(query, key, weights, top_k)

    torch.npu.synchronize()
    actual_out = func(query.view(batch, s1, n2, -1).npu(), key.npu(), weights.npu()).cpu()
    torch.npu.synchronize()
    mismatches = count_index_multiset_mismatches(golden_out, actual_out)
    total_indices = batch * s1 * n2 * top_k
    matched_ratio = 1 - mismatches / total_indices
    print(f"索引多重集合匹配率: {matched_ratio:.6f}，不匹配索引数: {mismatches}")
    if matched_ratio > 0.99:
        print("[PRECISION_PASS] 在线 TopK 索引多重集合匹配率超过 0.99")
        return
    print("[PRECISION_FAIL] 在线 TopK 索引多重集合匹配率未超过 0.99")
    raise AssertionError("在线 TopK 索引多重集合校验失败")


# 全量精度套件：覆盖 B/S1/S2/TOP_K/BLOCK_N/D 维度、边界（K==S2、单 trunk、
# 单批满 4-way、大 K UB 边界）与非常规整除值。
PRECISION_SUITE = [
    # (batch, s1, s2, top_k, block_n, dimension) — block_k 自动等于 dimension
    (1, 64, 512, 256, 512, 64),  # 最小 S2：单 trunk
    (1, 64, 512, 512, 512, 64),  # K == S2 边界
    (1, 64, 1024, 256, 64, 64),  # quick 既有：BN=64
    (1, 128, 1024, 512, 128, 64),  # BN=128
    (1, 128, 512, 512, 256, 64),  # S2=2 trunk
    (1, 128, 1536, 512, 512, 64),  # 3 trunk：单批 4-way 满
    (1, 256, 2048, 1024, 256, 64),  # BN=256 常规
    (1, 256, 4096, 256, 64, 64),  # 小 K 长序列
    (1, 512, 4096, 1024, 512, 64),  # 主路径 D=64
    (1, 512, 4096, 1024, 512, 128),  # 主路径 D=128
    (1, 1024, 8192, 1024, 512, 64),  # full 既有
    (1, 1024, 8192, 1024, 512, 128),  # full D=128
    (2, 512, 4096, 1024, 512, 128),  # benchmark 形状
    (4, 128, 2048, 512, 512, 64),  # 多 batch
    (2, 64, 8192, 2048, 128, 64),  # K=2048 上界：需 BN=128（UB 容量 72K B/对依赖）
    (1, 512, 4096, 2048, 128, 64),  # K=2048 大 K
    (2, 256, 1024, 256, 128, 128),  # D=128 小形状
    (3, 192, 3072, 768, 512, 64),  # 非常规整除值
]


def run_precision_suite():
    passed = 0
    failed = 0
    for case_id, (batch, s1, s2, top_k, block_n, dimension) in enumerate(PRECISION_SUITE, 1):
        label = f"B{batch}_S1_{s1}_S2_{s2}_K{top_k}_BN{block_n}_D{dimension}"
        print(f"\n[Case {case_id}/{len(PRECISION_SUITE)}] {label}")
        try:
            test_indexer(s1=s1, s2=s2, top_k=top_k, block_n=block_n, vector_basen=block_n, batch=batch, dimension=dimension)
            passed += 1
        except Exception as error:  # noqa: BLE001 — 套件需汇总失败而非中断
            failed += 1
            print(f"[CASE_FAIL] {label}: {error}")
    print(f"\n{'=' * 60}")
    print(f"精度套件汇总: {passed} 通过 / {failed} 失败 / {len(PRECISION_SUITE)} 总计")
    if failed == 0:
        print("Kernel Output Match!")
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证 Lightning Indexer 在线 TopK")
    parser.add_argument("--quick", action="store_true", help="运行较小的整除形状用例")
    parser.add_argument("--suite", action="store_true", help="运行全量精度测试套件")
    parser.add_argument("--block-n", type=int, default=256, help="BLOCK_N（64/128/256/512）")
    parser.add_argument("--s2-splits", type=int, default=None, help="S2 切分数（默认自动）")
    arguments = parser.parse_args()
    tilelang.disable_cache()
    if arguments.suite:
        raise SystemExit(0 if run_precision_suite() else 1)
    if arguments.quick:
        test_indexer(s1=64, s2=1024, top_k=256, block_n=64, vector_basen=64, s2_splits=arguments.s2_splits)
        print("Kernel Output Match!")
    else:
        test_indexer(
            s1=1024,
            s2=8192,
            top_k=1024,
            block_n=arguments.block_n,
            vector_basen=arguments.block_n,
            s2_splits=arguments.s2_splits,
        )
        print("Kernel Output Match!")
