import torch
import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout
from golden import golden_attention_simulated_kernel
from testcase import prepare_data


# ---------------------------------------------------------------------------
# 常量与优化 Pass 配置
# ---------------------------------------------------------------------------
NEG_INF = -(2.0**30)

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
}


@tilelang.jit(
    pass_configs=PASS_CONFIGS,
)
def mtgr_ragged_segment_attention_kernel(
    heads,
    dim,
    kv_group=1,
    sm_scale=None,
    block_M=128,
    block_N=128,
    core_num=24,
    num_stages=14,
    cross_interval=2,
    rule1_seg_idx=1,
):
    sm_scale = (1.0 / dim) ** 0.5 if sm_scale is None else sm_scale
    dtype = "bfloat16"
    accum_dtype = "float32"

    batch = T.symbolic("batch")
    total_q = T.symbolic("total_q")
    num_blocks = T.symbolic("num_blocks")
    max_blocks = T.symbolic("max_blocks")
    max_segs = T.symbolic("max_segs")

    kv_heads = heads // kv_group
    half_M = block_M // 2
    sub_M = half_M // 2

    # 跨核信号与单核内信号定义
    SEM_WS1_C2V = 0
    SEM_WS1_V2C = 1
    SEM_WS2_V2C = 2
    SEM_WS2_C2V = 3
    SEM_WS3_C2V = 4
    SEM_WS3_V2C = 5

    # Local event IDs are allocated per directed pipe pair. Different meanings keep
    # distinct names even when their directed pairs allow the same numeric ID.
    # MTE2 <-> MTE1
    SIG_K_L1 = 0
    SIG_P_L1 = 1
    SIG_V_L1 = 2
    SIG_Q_L1 = 3

    # MTE1 <-> M
    SIG_L0AB = 0  # double-buffer slots 0 and 1

    # M <-> FIX
    SIG_L0C = 0  # double-buffer slots 0 and 1

    # MTE2 <-> V
    SIG_IO_UB = 0

    # V <-> MTE3
    # acc_s_half and o_acc_half alias one physical UB region, so every access
    # shares one ownership token even when the O store is issued in row slices.
    SIG_STORE_UB = 0

    @T.prim_func
    def main(
        Q: T.Tensor([total_q, heads, dim], dtype),
        K: T.Tensor([total_q, kv_heads, dim], dtype),
        V: T.Tensor([total_q, kv_heads, dim], dtype),
        Output: T.Tensor([total_q, heads, dim], dtype),
        q_seq_starts: T.Tensor([batch + 1], "int32"),
        segment_offsets: T.Tensor([batch, max_segs + 1], "int32"),
        segment_rules: T.Tensor([max_segs], "int32"),
        workspace_1: T.Tensor([core_num, num_stages, block_M, block_N], dtype),
        workspace_2: T.Tensor([core_num, num_stages, block_M, block_N], dtype),
        workspace_3: T.Tensor([core_num, num_stages, block_M, dim], dtype),
        key_cache: T.Tensor([num_blocks, block_N, kv_heads, dim], dtype),
        value_cache: T.Tensor([num_blocks, block_N, kv_heads, dim], dtype),
        block_table: T.Tensor([batch, max_blocks], "int32"),
        prefix_lens: T.Tensor([batch], "int32"),
        bin_iters: T.int32,
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

            l0a_s = T.alloc_L0A([2, block_M, dim], dtype)
            l0b_s = T.alloc_L0B([2, dim, block_N], dtype)
            l0c_s = T.alloc_L0C([2, block_M, block_N], accum_dtype)

            l0a_o = T.alloc_L0A([2, block_M, block_N], dtype)
            l0b_o = T.alloc_L0B([2, block_N, dim], dtype)
            l0c_o_0 = T.alloc_L0C([block_M, dim], accum_dtype)
            l0c_o_1 = T.alloc_L0C([block_M, dim], accum_dtype)

            acc_o = T.alloc_ub([half_M, dim], accum_dtype)
            r_factors = T.alloc_ub([num_stages, half_M, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, half_M, 1], accum_dtype)
            sumexp = T.alloc_ub([half_M, 1], accum_dtype)
            neg_sm = T.alloc_ub([2, half_M, 1], accum_dtype)

            io_buf = T.alloc_ub([half_M, block_N], dtype)
            acc_s_half = T.alloc_ub([half_M, block_N], dtype)
            work_ub = T.alloc_ub([half_M, block_N], accum_dtype)
            buf_2d = T.alloc_ub([half_M, block_N], accum_dtype)

            bcast_buf = T.alloc_ub([half_M, dim], accum_dtype)
            o_io_buf = T.alloc_ub([half_M, dim], dtype)
            o_work_buf = T.alloc_ub([half_M, dim], accum_dtype)
            o_acc_half = T.alloc_ub([half_M, dim], dtype)

            row_rule_buf = T.alloc_ub([half_M], "int32")
            row_seg_start_buf = T.alloc_ub([half_M], "int32")
            row_seg_end_buf = T.alloc_ub([half_M], "int32")

            T.annotate_address(
                {
                    l0a_s: 0,
                    l0a_o: 0,
                    l0b_s: 0,
                    l0b_o: 0,
                    l0c_s: 0,
                    l0c_o_0: 0,
                    l0c_o_1: block_M * block_N * 4,
                    acc_o: 0,
                    r_factors: half_M * dim * 4,
                    sumexp_is: half_M * dim * 4 + num_stages * half_M * 4,
                    sumexp: half_M * dim * 4 + num_stages * half_M * 4 * 2,
                    neg_sm: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 4,
                    io_buf: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12,
                    o_io_buf: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12,
                    acc_s_half: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12 + half_M * block_N * 2,
                    o_acc_half: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12 + half_M * block_N * 2,
                    work_ub: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12 + half_M * block_N * 4,
                    o_work_buf: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12 + half_M * block_N * 4,
                    buf_2d: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12 + half_M * block_N * 8,
                    bcast_buf: half_M * dim * 4 + num_stages * half_M * 4 * 2 + half_M * 12 + half_M * block_N * 8,
                    row_rule_buf: half_M * dim * 4
                    + num_stages * half_M * 4 * 2
                    + half_M * 12
                    + half_M * block_N * 8
                    + half_M * block_N * 4,
                    row_seg_start_buf: half_M * dim * 4
                    + num_stages * half_M * 4 * 2
                    + half_M * 12
                    + half_M * block_N * 8
                    + half_M * block_N * 4
                    + half_M * 4,
                    row_seg_end_buf: half_M * dim * 4
                    + num_stages * half_M * 4 * 2
                    + half_M * 12
                    + half_M * block_N * 8
                    + half_M * block_N * 4
                    + half_M * 4 * 2,
                }
            )

            b_i = T.alloc_var("int32", init=0)
            s_local = T.alloc_var("int32", init=0)
            kv_start = T.alloc_var("int32", init=0)
            kv_size = T.alloc_var("int32", init=0)
            cum_tiles = T.alloc_var("int32", init=0)
            num_tiles_b = T.alloc_var("int32", init=0)
            total_seq_tiles = T.alloc_var("int32", init=0)
            total_seq_len_b = T.alloc_var("int32", init=0)
            k_upper_bound = T.alloc_var("int32", init=0)
            num_k_tiles = T.alloc_var("int32", init=0)
            seg_id = T.alloc_var("int32", init=0)
            last_seg_start = T.alloc_var("int32", init=0)
            gap_k_start = T.alloc_var("int32", init=0)
            gap_k_end = T.alloc_var("int32", init=0)
            gap_size = T.alloc_var("int32", init=0)
            num_effective_k = T.alloc_var("int32", init=0)

            _lo = T.alloc_var("int32", init=0)
            _hi = T.alloc_var("int32", init=0)
            _mid = T.alloc_var("int32", init=0)

            total_seq_tiles = 0
            for _b in T.serial(batch):
                _total_seq_len = segment_offsets[_b, max_segs]
                total_seq_tiles = total_seq_tiles + T.ceildiv(_total_seq_len, block_M)
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
                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                T.set_flag("M", "MTE1", SIG_L0AB)
                T.set_flag("M", "MTE1", SIG_L0AB + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)

                for core_index in T.serial(my_iters):
                    pid = cid + core_index * core_num
                    tile_id = pid // heads
                    h_i = pid % heads
                    h_kv = h_i // kv_group

                    cum_tiles = 0
                    b_i = 0
                    s_local = 0
                    for _b in T.serial(batch):
                        _total_seq_len = segment_offsets[_b, max_segs]
                        num_tiles_b = T.ceildiv(_total_seq_len, block_M)
                        _is_this = (tile_id >= cum_tiles) & (tile_id < cum_tiles + num_tiles_b)
                        b_i = T.if_then_else(_is_this, _b, b_i)
                        s_local = T.if_then_else(_is_this, tile_id - cum_tiles, s_local)
                        cum_tiles = cum_tiles + num_tiles_b

                    total_seq_len_b = segment_offsets[b_i, max_segs]
                    q_start = s_local * block_M
                    q_end = T.if_then_else(q_start + block_M < total_seq_len_b, q_start + block_M, total_seq_len_b)

                    prefix_len_b = prefix_lens[b_i]
                    q_start_live = T.if_then_else(q_start >= prefix_len_b, q_start, prefix_len_b)
                    q_tile_size_live = q_end - q_start_live
                    q_tile_size_live = T.if_then_else(q_tile_size_live > 0, q_tile_size_live, 0)
                    q_packed_start = q_seq_starts[b_i] + q_start_live - prefix_len_b

                    _rule1_start = segment_offsets[b_i, rule1_seg_idx]
                    _rule1_end = segment_offsets[b_i, rule1_seg_idx + 1]
                    _overlaps = (_rule1_end > q_start) & (_rule1_start < q_end)
                    k_upper_bound = T.if_then_else(_overlaps & (_rule1_end > q_end), _rule1_end, q_end)
                    num_k_tiles = T.ceildiv(k_upper_bound, block_M)

                    last_seg_start = segment_offsets[b_i, max_segs - 1]
                    gap_k_start = T.ceildiv(last_seg_start, block_M)
                    gap_k_end = q_start // block_M
                    gap_size = T.if_then_else((q_start >= last_seg_start) & (gap_k_end > gap_k_start), gap_k_end - gap_k_start, 0)
                    num_effective_k = num_k_tiles - gap_size

                    # 载入 Q，并在所有 KV batch 完成前保持 MTE1 ownership。
                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.copy(Q[q_packed_start : q_packed_start + q_tile_size_live, h_i, :], q_l1[:, :])
                    T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                    T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                    num_outer = T.ceildiv(num_effective_k, num_stages)

                    for k_outer in T.serial(num_outer):
                        _remaining = num_effective_k - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # ---------------------------------
                        # GEMM1: S = Q * K^T (写入 workspace_1)
                        # ---------------------------------
                        T.wait_cross_flag(SEM_WS1_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            compact_k_idx = k_outer * num_stages + i
                            k_idx = T.if_then_else(
                                (q_start >= last_seg_start) & (compact_k_idx >= gap_k_start) & (gap_size > 0),
                                compact_k_idx + gap_size,
                                compact_k_idx,
                            )
                            kv_start = k_idx * block_M
                            kv_size = T.if_then_else(kv_start + block_M < k_upper_bound, block_M, k_upper_bound - kv_start)

                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            if kv_start < prefix_len_b:
                                cache_block_idx = kv_start // block_N
                                physical_block = block_table[b_i, cache_block_idx]
                                block_offset_start = kv_start % block_N
                                T.copy(
                                    key_cache[physical_block, block_offset_start : block_offset_start + kv_size, h_kv, :],
                                    k_l1[:, :],
                                )
                            else:
                                kv_packed_start = q_seq_starts[b_i] + kv_start - prefix_len_b
                                T.copy(
                                    K[kv_packed_start : kv_packed_start + kv_size, h_kv, :],
                                    k_l1[:, :],
                                )
                            T.set_flag("MTE2", "MTE1", SIG_K_L1)

                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i < 2:
                                T.copy(q_l1, l0a_s[side, :, :])

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            T.copy(k_l1, l0b_s[side, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a_s[side, :, :], l0b_s[side, :, :], l0c_s[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c_s[side, :, :], workspace_1[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)

                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)

                        # ---------------------------------
                        # GEMM2: O = P * V (写入 workspace_3)
                        # ---------------------------------
                        T.wait_cross_flag(SEM_WS3_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            compact_k_idx = k_outer * num_stages + i
                            k_idx = T.if_then_else(
                                (q_start >= last_seg_start) & (compact_k_idx >= gap_k_start) & (gap_size > 0),
                                compact_k_idx + gap_size,
                                compact_k_idx,
                            )
                            kv_start = k_idx * block_M
                            kv_size = T.if_then_else(kv_start + block_M < k_upper_bound, block_M, k_upper_bound - kv_start)

                            T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                            if kv_start < prefix_len_b:
                                cache_block_idx = kv_start // block_N
                                physical_block = block_table[b_i, cache_block_idx]
                                block_offset_start = kv_start % block_N
                                T.copy(
                                    value_cache[physical_block, block_offset_start : block_offset_start + kv_size, h_kv, :],
                                    v_l1[:, :],
                                )
                            else:
                                kv_packed_start = q_seq_starts[b_i] + kv_start - prefix_len_b
                                T.copy(
                                    V[kv_packed_start : kv_packed_start + kv_size, h_kv, :],
                                    v_l1[:, :],
                                )
                            T.set_flag("MTE2", "MTE1", SIG_V_L1)

                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(workspace_2[cid, i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            T.copy(v_l1, l0b_o[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_V_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a_o[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            if side == 0:
                                T.mma(l0a_o[side, :, :], l0b_o[side, :, :], l0c_o_0[:, :], init=True)
                            else:
                                T.mma(l0a_o[side, :, :], l0b_o[side, :, :], l0c_o_1[:, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            if side == 0:
                                T.copy(l0c_o_0[:, :], workspace_3[cid, i, :, :])
                            else:
                                T.copy(l0c_o_1[:, :], workspace_3[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)

                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS3_C2V)
                        T.set_cross_flag("MTE2", SEM_WS2_C2V)

                    # MTE1 不再读取 q_l1；归还 ownership 后，下一个 task 才能重载 Q。
                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)

                # 回收初始化的 Signal
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
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
                T.set_flag("MTE3", "V", SIG_STORE_UB)

                for core_index in T.serial(my_iters):
                    pid = cid + core_index * core_num
                    tile_id = pid // heads
                    h_i = pid % heads
                    h_kv = h_i // kv_group

                    cum_tiles = 0
                    b_i = 0
                    s_local = 0
                    for _b in T.serial(batch):
                        _total_seq_len = segment_offsets[_b, max_segs]
                        num_tiles_b = T.ceildiv(_total_seq_len, block_M)
                        _is_this = (tile_id >= cum_tiles) & (tile_id < cum_tiles + num_tiles_b)
                        b_i = T.if_then_else(_is_this, _b, b_i)
                        s_local = T.if_then_else(_is_this, tile_id - cum_tiles, s_local)
                        cum_tiles = cum_tiles + num_tiles_b

                    total_seq_len_b = segment_offsets[b_i, max_segs]
                    q_start = s_local * block_M
                    q_end = T.if_then_else(q_start + block_M < total_seq_len_b, q_start + block_M, total_seq_len_b)

                    prefix_len_b = prefix_lens[b_i]
                    q_start_live = T.if_then_else(q_start >= prefix_len_b, q_start, prefix_len_b)
                    q_tile_size_live = q_end - q_start_live
                    q_tile_size_live = T.if_then_else(q_tile_size_live > 0, q_tile_size_live, 0)
                    q_packed_start = q_seq_starts[b_i] + q_start_live - prefix_len_b

                    _rule1_start = segment_offsets[b_i, rule1_seg_idx]
                    _rule1_end = segment_offsets[b_i, rule1_seg_idx + 1]
                    _overlaps = (_rule1_end > q_start) & (_rule1_start < q_end)
                    k_upper_bound = T.if_then_else(_overlaps & (_rule1_end > q_end), _rule1_end, q_end)
                    num_k_tiles = T.ceildiv(k_upper_bound, block_M)

                    last_seg_start = segment_offsets[b_i, max_segs - 1]
                    gap_k_start = T.ceildiv(last_seg_start, block_M)
                    gap_k_end = q_start // block_M
                    gap_size = T.if_then_else((q_start >= last_seg_start) & (gap_k_end > gap_k_start), gap_k_end - gap_k_start, 0)
                    num_effective_k = num_k_tiles - gap_size

                    for row in T.serial(half_M):
                        row_abs_pos = q_start + vid * half_M + row

                        _lo = 0
                        _hi = max_segs
                        for _bi in T.serial(bin_iters):
                            _mid = (_lo + _hi) // 2
                            _active = _lo < _hi
                            _le = segment_offsets[b_i, _mid] <= row_abs_pos
                            _gt = segment_offsets[b_i, _mid] > row_abs_pos
                            _lo = T.if_then_else(_active & _le, _mid + 1, _lo)
                            _hi = T.if_then_else(_active & _gt, _mid, _hi)
                        seg_id = _lo - 1

                        row_rule_buf[row] = segment_rules[seg_id]
                        row_seg_start_buf[row] = segment_offsets[b_i, seg_id]
                        row_seg_end_buf[row] = segment_offsets[b_i, seg_id + 1]

                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(neg_sm, 2**30)

                    num_outer = T.ceildiv(num_effective_k, num_stages)
                    for k_outer in T.serial(num_outer):
                        _remaining = num_effective_k - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- Softmax Batch 处理 ---
                        T.wait_cross_flag(SEM_WS2_C2V)
                        for i in T.serial(batch_iters):
                            cur = i % 2
                            prv = 1 - cur

                            compact_k_idx = k_outer * num_stages + i
                            k_idx = T.if_then_else(
                                (q_start >= last_seg_start) & (compact_k_idx >= gap_k_start) & (gap_size > 0),
                                compact_k_idx + gap_size,
                                compact_k_idx,
                            )
                            kv_start = k_idx * block_M
                            kv_size = T.if_then_else(kv_start + block_M < k_upper_bound, block_M, k_upper_bound - kv_start)

                            # 【核心优化】计算掩盖 (Hiding Computation)：在等 Cube 前利用算力资源提前生成 MASK
                            _vid_first_rule = row_rule_buf[0]
                            _vid_first_seg_start = row_seg_start_buf[0]
                            _vid_first_seg_end = row_seg_end_buf[0]
                            _first_row_pos = q_start + vid * half_M

                            _is_full_kv = kv_size == block_M

                            # 基于“可见性单调递增”原理：首行完全可见，则全块必然完全可见
                            _r0_valid = (_vid_first_rule == 0) & (_first_row_pos + 1 >= kv_start + kv_size)
                            _r1_valid = (_vid_first_rule == 1) & (kv_start + kv_size <= _vid_first_seg_end)
                            _r2_valid = (_vid_first_rule == 2) & (kv_start + kv_size <= _vid_first_seg_start)

                            _all_valid = T.if_then_else(
                                _is_full_kv & _r0_valid,
                                1,
                                T.if_then_else(_is_full_kv & _r1_valid, 1, T.if_then_else(_is_full_kv & _r2_valid, 1, 0)),
                            )
                            if _all_valid == 1:
                                T.tile.fill(buf_2d, 0.0)
                            else:
                                T.tile.fill(buf_2d, NEG_INF)
                                for row in T.serial(half_M):
                                    row_abs_pos = q_start + vid * half_M + row

                                    _row_rule = row_rule_buf[row]
                                    _row_seg_start = row_seg_start_buf[row]
                                    _row_seg_end = row_seg_end_buf[row]

                                    if _row_rule == 0:
                                        raw_len = row_abs_pos - kv_start + 1
                                        fill_len = T.if_then_else(raw_len < kv_size, raw_len, kv_size)
                                        if fill_len > 0:
                                            T.tile.fill(buf_2d[row, 0:fill_len], 0.0)
                                    elif _row_rule == 1:
                                        raw_len = _row_seg_end - kv_start
                                        fill_len = T.if_then_else(raw_len < kv_size, raw_len, kv_size)
                                        if fill_len > 0:
                                            T.tile.fill(buf_2d[row, 0:fill_len], 0.0)
                                    elif _row_rule == 2:
                                        raw_len = _row_seg_start - kv_start
                                        fill_len = T.if_then_else(raw_len < kv_size, raw_len, kv_size)
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

                            T.wait_flag("MTE3", "V", SIG_STORE_UB)
                            T.copy(work_ub, acc_s_half)
                            T.set_flag("V", "MTE3", SIG_STORE_UB)

                            T.wait_flag("V", "MTE3", SIG_STORE_UB)
                            T.copy(acc_s_half, workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :])
                            T.set_flag("MTE3", "V", SIG_STORE_UB)

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
                            T.tile.broadcast(bcast_buf, r_factors[i, :, :])
                            T.tile.mul(acc_o, acc_o, bcast_buf)

                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS3_C2V)
                            T.copy(workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :], o_io_buf)
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.copy(o_io_buf, o_work_buf)
                            T.set_flag("V", "MTE2", SIG_IO_UB)
                            T.tile.add(acc_o, acc_o, o_work_buf)

                        T.set_cross_flag("MTE2", SEM_WS3_V2C)

                    for sub in T.serial(2):
                        row_start = sub * sub_M
                        T.tile.max(sumexp[row_start : row_start + sub_M, :], sumexp[row_start : row_start + sub_M, :], 1.0)
                        T.tile.broadcast(bcast_buf[row_start : row_start + sub_M, :], sumexp[row_start : row_start + sub_M, :])
                        T.tile.div(
                            acc_o[row_start : row_start + sub_M, :],
                            acc_o[row_start : row_start + sub_M, :],
                            bcast_buf[row_start : row_start + sub_M, :],
                        )
                        T.wait_flag("MTE3", "V", SIG_STORE_UB)
                        T.copy(
                            acc_o[row_start : row_start + sub_M, :],
                            o_acc_half[row_start : row_start + sub_M, :],
                        )
                        T.set_flag("V", "MTE3", SIG_STORE_UB)

                        T.wait_flag("V", "MTE3", SIG_STORE_UB)
                        valid_rows_sub = T.if_then_else(
                            q_tile_size_live >= vid * half_M + row_start + sub_M,
                            sub_M,
                            T.if_then_else(
                                q_tile_size_live > vid * half_M + row_start,
                                q_tile_size_live - vid * half_M - row_start,
                                0,
                            ),
                        )
                        h_i_out = (cid + core_index * core_num) % heads
                        output_packed_start = q_packed_start + vid * half_M + row_start

                        if valid_rows_sub > 0:
                            T.copy(
                                o_acc_half[row_start : row_start + valid_rows_sub, :],
                                Output[
                                    output_packed_start : output_packed_start + valid_rows_sub,
                                    h_i_out,
                                    :,
                                ],
                            )
                        T.set_flag("MTE3", "V", SIG_STORE_UB)

                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("MTE3", "V", SIG_STORE_UB)

    return main


# ---------------------------------------------------------------------------
# Host 端 Wrapper（集成动态长度推导与 Workspace 分配）
# ---------------------------------------------------------------------------
def mtgr_ragged_segment_attention(
    query,
    key,
    value,
    segment_offsets_i32,
    segment_rules_i32,
    q_seq_starts_i32,
    matched_prefix_lens_i32,
    match_mode,
    key_cache,
    value_cache,
    block_table_i32,
    block_size,
    max_request_len,
    sm_scale,
    output_snd,
    block_M=128,
    core_num=24,
    num_stages=14,
    kv_group=1,
    cross_interval=2,
):
    assert block_size == 128, f"block_size must be 128, got {block_size}"
    block_N = block_size

    B = segment_offsets_i32.size(0)
    H = query.size(1)
    D = query.size(2)
    max_segs = segment_offsets_i32.size(1) - 1

    assert segment_offsets_i32.size(1) == max_segs + 1, (
        f"segment_offsets_i32 shape {segment_offsets_i32.shape} != [batch={B}, max_segs+1={max_segs + 1}]"
    )
    assert segment_rules_i32.size(0) == max_segs, f"segment_rules_i32 shape {segment_rules_i32.shape} != [max_segs={max_segs}]"

    ws1 = torch.empty((core_num, num_stages, block_M, block_N), dtype=torch.bfloat16, device=query.device)
    ws2 = torch.empty((core_num, num_stages, block_M, block_N), dtype=torch.bfloat16, device=query.device)
    ws3 = torch.zeros((core_num, num_stages, block_M, D), dtype=torch.bfloat16, device=query.device)

    bin_iters = max_segs.bit_length()

    func = mtgr_ragged_segment_attention_kernel(
        heads=H,
        dim=D,
        kv_group=kv_group,
        sm_scale=sm_scale,
        block_M=block_M,
        block_N=block_N,
        core_num=core_num,
        num_stages=num_stages,
        cross_interval=cross_interval,
    )

    func(
        query,
        key,
        value,
        output_snd,
        q_seq_starts_i32,
        segment_offsets_i32,
        segment_rules_i32,
        ws1,
        ws2,
        ws3,
        key_cache,
        value_cache,
        block_table_i32,
        matched_prefix_lens_i32,
        bin_iters,
    )

    torch.npu.synchronize()
    return output_snd


def test(config, block_M=128, core_num=24, num_stages=14, cross_interval=2):
    tilelang.disable_cache()
    tilelang.cache.clear_cache()
    data = prepare_data(config)

    torch.npu.synchronize()
    print("init successful!")

    q_seq_starts = data["q_seq_starts_i32"].tolist()
    max_request_len = max(q_seq_starts[b + 1] - q_seq_starts[b] for b in range(len(q_seq_starts) - 1))
    prefix_lens = data["matched_prefix_lens_i32"].tolist()
    if all(p == 0 for p in prefix_lens):
        match_mode = 0
    elif all(p > 0 for p in prefix_lens):
        match_mode = 1
    else:
        match_mode = 2

    output_snd = torch.empty_like(data["query_snd"].npu())
    output_snd = mtgr_ragged_segment_attention(
        data["query_snd"].npu(),
        data["key_snd"].npu(),
        data["value_snd"].npu(),
        data["segment_offsets_i32"].npu(),
        data["segment_rules_i32"].npu(),
        data["q_seq_starts_i32"].npu(),
        data["matched_prefix_lens_i32"].npu(),
        match_mode,
        data["key_cache"].npu(),
        data["value_cache"].npu(),
        data["block_table_tensor"].npu(),
        data["block_size"],
        max_request_len,
        data["sm_scale"],
        output_snd,
        block_M=block_M,
        core_num=core_num,
        kv_group=data["kv_group"],
        num_stages=num_stages,
        cross_interval=cross_interval,
    )

    torch.npu.synchronize()

    ref_output = golden_attention_simulated_kernel(
        data["query_snd"],
        data["key_snd"],
        data["value_snd"],
        data["segment_offsets_i32"],
        data["segment_rules_i32"],
        data["q_seq_starts_i32"],
        data["matched_prefix_lens_i32"],
        data["key_cache"],
        data["value_cache"],
        data["block_table_tensor"],
        data["block_size"],
        data["sm_scale"],
    ).to(torch.bfloat16)

    torch.npu.synchronize()
    torch.testing.assert_close(ref_output.cpu(), output_snd.cpu(), rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")


if __name__ == "__main__":
    test_configs = [
        {
            "H": 8,
            "D": 128,
            "seg_lengths": [[1600, 8] + [5] * 1 + [1200]],
            "rules": [0, 1] + [2] * 1 + [2],
            "matched_prefix_arr": [0],
        },
    ]

    for config in test_configs:
        test(config)
