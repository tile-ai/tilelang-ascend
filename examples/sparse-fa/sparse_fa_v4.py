import tilelang
from tilelang import language as T
import torch
import math

torch.set_default_device("npu")
torch.manual_seed(0)

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

NEG_INF = -(2.0**30)


@tilelang.jit(
    out_idx=[3],
    workspace_idx=[9, 10, 11],
    pass_configs=PASS_CONFIGS,
    debug_root_path="/tmp/sparse_fa_v4_debug",
)
def mtgr_sparse_attn_kernel(
    heads,
    dim,
    kv_group=1,
    sm_scale=None,
    block_M=128,
    block_N=128,
    core_num=24,
    max_splits=32,
    max_segs=4,
):
    sm_scale = (1.0 / dim) ** 0.5 if sm_scale is None else sm_scale
    dtype = "bfloat16"
    accum_dtype = "float32"

    batch = T.symbolic("batch")
    total_q = T.symbolic("total_q")
    total_seq_tiles = T.symbolic("total_seq_tiles")

    kv_heads = heads // kv_group
    v_block = block_M // 2

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
        workspace_s: T.Tensor([core_num, block_M, block_N], accum_dtype),
        workspace_p: T.Tensor([core_num, block_M, block_N], dtype),
        workspace_o: T.Tensor([core_num, block_M, dim], accum_dtype),
        _dummy: T.Tensor([total_seq_tiles], "int32"),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)
            acc_s_l1 = T.alloc_L1([block_M, block_N], dtype)

            acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_M, dim], accum_dtype)

            acc_o = T.alloc_ub([v_block, dim], accum_dtype)
            sumexp = T.alloc_ub([v_block], accum_dtype)
            m_i = T.alloc_ub([v_block], accum_dtype)
            m_i_broadcast = T.alloc_ub([v_block, block_N], accum_dtype)
            m_i_prev = T.alloc_ub([v_block], accum_dtype)
            m_i_prev_broadcast = T.alloc_ub([v_block, dim], accum_dtype)

            acc_s_ub = T.alloc_ub([v_block, block_N], accum_dtype)
            acc_s_ub_ = T.alloc_ub([v_block, block_N], accum_dtype)
            mask_ub = T.alloc_ub([v_block, block_N], accum_dtype)
            sumexp_i_ub = T.alloc_ub([v_block], accum_dtype)
            acc_s_half = T.alloc_ub([v_block, block_N], dtype)
            acc_o_ub = T.alloc_ub([v_block, dim], accum_dtype)
            acc_o_half = T.alloc_ub([v_block, dim], dtype)

            sumexp_broadcast = T.alloc_ub([v_block, dim], accum_dtype)

            b_i = T.alloc_var("int32", init=0)
            seg_id = T.alloc_var("int32", init=0)
            next_seg_first_tile = T.alloc_var("int32", init=0)

            total_tasks = total_seq_tiles * heads
            my_iters = T.if_then_else(
                cid < total_tasks,
                T.ceildiv(total_tasks - cid, core_num),
                0,
            )

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
                    seg_id = T.if_then_else(
                        q_start >= segment_offsets[b_i, _seg + 1],
                        _seg + 1,
                        seg_id,
                    )

                rule = segment_rules[seg_id]
                q_packed_start = q_seq_starts[b_i] + q_start

                seg_end_offset = segment_offsets[b_i, seg_id + 1]
                next_seg_first_tile = s_local
                for _k in T.serial(max_splits):
                    next_seg_first_tile = T.if_then_else(
                        (_k > s_local) & (split_points[b_i, _k] < seg_end_offset),
                        _k,
                        next_seg_first_tile,
                    )
                tiles_this_batch = tiles_prefix_sum[b_i + 1] - tiles_prefix_sum[b_i]
                next_seg_first_tile = T.if_then_else(
                    next_seg_first_tile == s_local,
                    tiles_this_batch,
                    next_seg_first_tile,
                )

                kv_iter_end = T.if_then_else(
                    rule == 1,
                    next_seg_first_tile,
                    s_local + 1,
                )

                with T.Scope("C"):
                    T.copy(
                        Q[q_packed_start : q_packed_start + q_tile_size, h_i, :],
                        q_l1[0:q_tile_size, :],
                    )

                    for k_i in T.serial(kv_iter_end):
                        kv_start = split_points[b_i, k_i]
                        kv_end_pos = split_points[b_i, k_i + 1]
                        kv_size = kv_end_pos - kv_start

                        seg_start = segment_offsets[b_i, seg_id]
                        seg_end = segment_offsets[b_i, seg_id + 1]
                        max_row_pos = q_start + q_tile_size - 1

                        process_rule0 = (rule == 0) & (kv_start <= max_row_pos)
                        process_rule1 = (rule == 1) & (kv_start < seg_end)
                        process_rule2 = (rule == 2) & (
                            (kv_start <= seg_start) | (kv_start <= max_row_pos)
                        )
                        process_cond = process_rule0 | process_rule1 | process_rule2

                        if process_cond:
                            kv_packed_start = q_seq_starts[b_i] + kv_start
                            T.copy(
                                K[kv_packed_start : kv_packed_start + kv_size, h_kv, :],
                                k_l1[0:kv_size, :],
                            )

                            T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                            T.copy(acc_s_l0c, workspace_s[cid, :, :])
                            T.set_cross_flag("FIX", 0)

                            T.wait_cross_flag(1)

                            T.copy(workspace_p[cid, :, :], acc_s_l1)

                            kv_packed_start_v = q_seq_starts[b_i] + kv_start
                            T.copy(
                                V[kv_packed_start_v : kv_packed_start_v + kv_size, h_kv, :],
                                v_l1[0:kv_size, :],
                            )

                            T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                            T.copy(acc_o_l0c, workspace_o[cid, :, :])
                            T.set_cross_flag("FIX", 2)

                            T.wait_cross_flag(3)

                with T.Scope("V"):
                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(m_i, NEG_INF)

                    for k_i in T.serial(kv_iter_end):
                        kv_start = split_points[b_i, k_i]
                        kv_end_pos = split_points[b_i, k_i + 1]
                        kv_size = kv_end_pos - kv_start

                        seg_start = segment_offsets[b_i, seg_id]
                        seg_end = segment_offsets[b_i, seg_id + 1]
                        max_row_pos = q_start + q_tile_size - 1

                        process_rule0 = (rule == 0) & (kv_start <= max_row_pos)
                        process_rule1 = (rule == 1) & (kv_start < seg_end)
                        process_rule2 = (rule == 2) & (
                            (kv_start <= seg_start) | (kv_start <= max_row_pos)
                        )
                        process_cond = process_rule0 | process_rule1 | process_rule2

                        if process_cond:
                            T.wait_cross_flag(0)

                            T.copy(
                                workspace_s[
                                    cid,
                                    vid * v_block : vid * v_block + v_block,
                                    :,
                                ],
                                acc_s_ub_,
                            )
                            T.tile.mul(acc_s_ub_, acc_s_ub_, sm_scale)

                            T.tile.fill(mask_ub, NEG_INF)

                            for row in T.serial(v_block):
                                row_abs_pos = q_start + vid * v_block + row

                                if rule == 0:
                                    raw_len = row_abs_pos - kv_start + 1
                                    fill_len = T.if_then_else(
                                        raw_len < kv_size, raw_len, kv_size
                                    )
                                    fill_len = T.if_then_else(fill_len > 0, fill_len, 0)
                                    if fill_len > 0:
                                        T.tile.fill(mask_ub[row, 0:fill_len], 0.0)

                                elif rule == 1:
                                    raw_len = seg_end - kv_start
                                    fill_len = T.if_then_else(
                                        raw_len < kv_size, raw_len, kv_size
                                    )
                                    fill_len = T.if_then_else(fill_len > 0, fill_len, 0)
                                    if fill_len > 0:
                                        T.tile.fill(mask_ub[row, 0:fill_len], 0.0)

                                elif rule == 2:
                                    raw_len = seg_start - kv_start
                                    fill_len = T.if_then_else(
                                        raw_len < kv_size, raw_len, kv_size
                                    )
                                    fill_len = T.if_then_else(fill_len > 0, fill_len, 0)
                                    if fill_len > 0:
                                        T.tile.fill(mask_ub[row, 0:fill_len], 0.0)
                                    diag_col = row_abs_pos - kv_start
                                    if (diag_col >= 0) & (diag_col < kv_size):
                                        mask_ub[row, diag_col] = 0.0

                            T.tile.add(acc_s_ub, acc_s_ub_, mask_ub)

                            T.copy(m_i, m_i_prev)
                            T.reduce_max(acc_s_ub, m_i, dim=-1)
                            T.tile.max(m_i, m_i, m_i_prev)
                            T.tile.sub(m_i_prev, m_i_prev, m_i)
                            T.tile.exp(m_i_prev, m_i_prev)

                            T.tile.broadcast(m_i_broadcast, m_i, axis=1)
                            T.tile.sub(acc_s_ub, acc_s_ub, m_i_broadcast)
                            T.tile.exp(acc_s_ub, acc_s_ub)
                            T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                            T.tile.mul(sumexp, sumexp, m_i_prev)
                            T.tile.add(sumexp, sumexp, sumexp_i_ub)

                            T.tile.broadcast(m_i_prev_broadcast, m_i_prev, axis=1)
                            T.tile.mul(acc_o, acc_o, m_i_prev_broadcast)

                            T.copy(acc_s_ub, acc_s_half)
                            T.copy(
                                acc_s_half,
                                workspace_p[
                                    cid,
                                    vid * v_block : vid * v_block + v_block,
                                    :,
                                ],
                            )
                            T.set_cross_flag("MTE3", 1)

                            T.wait_cross_flag(2)

                            T.copy(
                                workspace_o[
                                    cid,
                                    vid * v_block : vid * v_block + v_block,
                                    :,
                                ],
                                acc_o_ub,
                            )
                            T.tile.add(acc_o, acc_o, acc_o_ub)
                            T.set_cross_flag("MTE3", 3)

                    valid_rows = T.if_then_else(
                        q_tile_size >= (vid + 1) * v_block,
                        v_block,
                        T.if_then_else(
                            q_tile_size > vid * v_block, q_tile_size - vid * v_block, 0
                        ),
                    )

                    T.tile.max(sumexp, sumexp, 1.0)
                    T.tile.broadcast(sumexp_broadcast, sumexp, axis=1)
                    T.tile.div(acc_o, acc_o, sumexp_broadcast)

                    T.copy(acc_o, acc_o_half)
                    output_packed_start = q_packed_start + vid * v_block
                    T.copy(
                        acc_o_half[0:valid_rows, :],
                        Output[
                            output_packed_start : output_packed_start + valid_rows,
                            h_i,
                            :,
                        ],
                    )

    return main


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


def mtgr_sparse_attn_wrapper(
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
    kv_group=1,
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
    split_points_i32 = torch.tensor(split_points_padded, dtype=torch.int32)

    seg_offsets_padded = []
    for b in range(B):
        offsets = segment_offsets_i32[b].cpu().tolist()
        seg_offsets_padded.append(offsets + [0] * (max_segs + 1 - len(offsets)))
    segment_offsets_padded_i32 = torch.tensor(seg_offsets_padded, dtype=torch.int32)

    rules_padded = segment_rules_i32.cpu().tolist() + [0] * (max_segs - len(segment_rules_i32))
    segment_rules_padded_i32 = torch.tensor(rules_padded, dtype=torch.int32)

    tiles_per_batch = [len(sp) - 1 for sp in all_split_points]
    total_seq_tiles = sum(tiles_per_batch)

    tiles_prefix_sum_list = [0]
    for t in tiles_per_batch:
        tiles_prefix_sum_list.append(tiles_prefix_sum_list[-1] + t)
    tiles_prefix_sum_i32 = torch.tensor(tiles_prefix_sum_list, dtype=torch.int32)

    func = mtgr_sparse_attn_kernel(
        heads=H,
        dim=D,
        kv_group=kv_group,
        sm_scale=sm_scale,
        block_M=block_M,
        block_N=block_N,
        core_num=core_num,
        max_splits=max_splits,
        max_segs=max_segs,
    )

    # print(func.get_kernel_source())

    dummy_tensor = torch.empty(total_seq_tiles, dtype=torch.int32, device=query.device)
    output = func(
        query,
        key,
        value,
        q_seq_starts_i32,
        split_points_i32,
        tiles_prefix_sum_i32,
        segment_offsets_padded_i32,
        segment_rules_padded_i32,
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

    output_snd = mtgr_sparse_attn_wrapper(
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
        # {
        #     "H": 1,
        #     "D": 8,
        #     "seg_lengths": [[4, 4, 4, 4]],
        #     "rules": [0, 1, 2, 2],
        #     "matched_prefix_arr": [0],
        # },
        # {
        #     "H": 8,
        #     "D": 64,
        #     "seg_lengths": [[1600, 8, 200, 1200]],
        #     "rules": [0, 1, 2, 2],
        #     "matched_prefix_arr": [0],
        # },
        # {
        #     "H": 8,
        #     "D": 64,
        #     "seg_lengths": [[1600, 8, 200, 1200]],
        #     "rules": [0, 1, 0, 2],
        #     "matched_prefix_arr": [0],
        # },
        # {
        #     "H": 8,
        #     "D": 64,
        #     "seg_lengths": [[1600, 8, 200, 1200], [1700, 8, 300, 1024]],
        #     "rules": [0, 1, 2, 2],
        #     "matched_prefix_arr": [0, 0],
        # },
        # {
        #     "H": 8,
        #     "D": 64,
        #     "seg_lengths": [[1800, 8, 100, 1500], [1500, 8, 300, 1200]],
        #     "rules": [0, 1, 0, 2],
        #     "matched_prefix_arr": [0, 0],
        # },
        # {
        #     "H": 8,
        #     "D": 64,
        #     "seg_lengths": [[1600, 8, 200, 1200], [1700, 8, 300, 1024], [1680, 8, 200, 1280], [2000, 8, 700, 2048]],
        #     "rules": [0, 1, 2, 2],
        #     "matched_prefix_arr": [0, 0, 0, 0],
        # },
        # {
        #     "H": 8,
        #     "D": 64,
        #     "seg_lengths": [[3200, 8, 200, 1200], [2300, 8, 400, 1800], [2080, 8, 200, 1800], [1700, 8, 100, 1024]],
        #     "rules": [0, 1, 0, 2],
        #     "matched_prefix_arr": [0, 0, 0, 0],
        # },
        # {
        #     "H": 8,
        #     "D": 64,
        #     "seg_lengths": [
        #         [2200, 8, 200, 1024],
        #         [1700, 8, 100, 1100],
        #         [2440, 8, 200, 2048],
        #         [1600, 8, 600, 1900],
        #         [3300, 8, 200, 1300],
        #         [1700, 8, 300, 2100],
        #         [1780, 8, 700, 1200],
        #         [2048, 8, 500, 1800],
        #     ],
        #     "rules": [0, 1, 2, 2],
        #     "matched_prefix_arr": [0, 0, 0, 0, 0, 0, 0, 0],
        # },
        {
            "H": 8,
            "D": 64,
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
            "rules": [0, 1, 2, 2],
            "matched_prefix_arr": [0, 0, 0, 0, 0, 0, 0, 0],
        },
    ]

    for config in test_configs:
        test(**config)
