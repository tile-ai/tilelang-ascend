import tilelang
from tilelang import language as T
import torch
import math

torch.set_default_device("npu")
torch.manual_seed(0)

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(
    out_idx=[3],
    workspace_idx=[16, 17, 18],
    pass_configs=PASS_CONFIGS,
)
def mtgr_ragged_segment_attention_nokv(
    heads,
    dim,
    kv_group=1,
    sm_scale=None,
    block_M=128,
    block_N=128,
    core_num=24,
    max_splits=16,
    max_segments=4,
):
    sm_scale = (1.0 / dim) ** 0.5 if sm_scale is None else sm_scale
    dtype = "bfloat16"
    accum_dtype = "float32"

    batch = T.symbolic("batch")
    total_q = T.symbolic("total_q")
    num_groups = T.symbolic("num_groups")
    total_seq_tiles = T.symbolic("total_seq_tiles")
    max_q_tiles = T.symbolic("max_q_tiles")

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
        split_count: T.Tensor([batch], "int32"),
        tile_seg_id: T.Tensor([batch, max_q_tiles], "int32"),
        seg_first_tile: T.Tensor([batch, max_segments + 1], "int32"),
        mask_causal: T.Tensor([block_M, block_N], accum_dtype),
        mask_diag: T.Tensor([block_M, block_N], accum_dtype),
        group_batch_ids: T.Tensor([num_groups], "int32"),
        group_rules: T.Tensor([num_groups], "int32"),
        group_tile_offsets: T.Tensor([num_groups], "int32"),
        group_tile_counts: T.Tensor([num_groups], "int32"),
        sorted_s_local: T.Tensor([total_seq_tiles], "int32"),
        workspace_s: T.Tensor([core_num, block_M, block_N], accum_dtype),
        workspace_p: T.Tensor([core_num, block_M, block_N], dtype),
        workspace_o: T.Tensor([core_num, block_M, dim], accum_dtype),
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
            sumexp_i_ub = T.alloc_ub([v_block], accum_dtype)
            acc_s_half = T.alloc_ub([v_block, block_N], dtype)
            acc_o_ub = T.alloc_ub([v_block, dim], accum_dtype)
            acc_o_half = T.alloc_ub([v_block, dim], dtype)

            mask_ub = T.alloc_ub([v_block, block_N], accum_dtype)

            total_tasks = num_groups * heads
            my_iters = T.if_then_else(cid < total_tasks, T.ceildiv(total_tasks - cid, core_num), 0)
            for core_index in T.serial(my_iters):
                pid = cid + core_index * core_num
                group_id = pid // heads
                h_i = pid % heads
                h_kv = h_i // kv_group

                b_i = group_batch_ids[group_id]
                rule = group_rules[group_id]
                tile_offset = group_tile_offsets[group_id]
                tile_count = group_tile_counts[group_id]

                for tile_idx in T.serial(tile_count):
                    s_local = sorted_s_local[tile_offset + tile_idx]

                    q_start = split_points[b_i, s_local]
                    q_end = split_points[b_i, s_local + 1]
                    q_tile_size = q_end - q_start

                    q_packed_start = q_seq_starts[b_i] + q_start
                    T.copy(
                        Q[q_packed_start : q_packed_start + q_tile_size, h_i, :],
                        q_l1[0:q_tile_size, :],
                    )

                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(m_i, -(2.0**30))

                    seg_id = tile_seg_id[b_i, s_local]

                    kv_iter_start = 0
                    if rule == 0:
                        kv_iter_end = s_local + 1
                    elif rule == 1:
                        kv_iter_end = seg_first_tile[b_i, seg_id + 1]
                    elif rule == 2:
                        kv_iter_end = seg_first_tile[b_i, seg_id + 1]

                    for k_i in T.serial(kv_iter_start, kv_iter_end):
                        kv_start = split_points[b_i, k_i]
                        kv_end = split_points[b_i, k_i + 1]
                        kv_size = kv_end - kv_start

                        is_overlap = T.if_then_else(kv_start == q_start, 1, 0)

                        kv_packed_start = q_seq_starts[b_i] + kv_start
                        T.copy(
                            K[kv_packed_start : kv_packed_start + kv_size, h_kv, :],
                            k_l1[0:kv_size, :],
                        )

                        T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                        T.copy(acc_s_l0c, workspace_s[cid, :, :])

                        T.tile.fill(acc_s_ub, 0.0)
                        T.copy(
                            workspace_s[
                                cid,
                                vid * v_block : vid * v_block + v_block,
                                :,
                            ],
                            acc_s_ub_,
                        )
                        T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                        T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)

                        if rule == 0:
                            if is_overlap == 1:
                                T.copy(
                                    mask_causal[
                                        vid * v_block : vid * v_block + v_block,
                                        0:block_N,
                                    ],
                                    mask_ub,
                                )
                                T.tile.add(acc_s_ub, acc_s_ub, mask_ub)
                        elif rule == 2:
                            is_in_segment = T.if_then_else(
                                k_i >= seg_first_tile[b_i, seg_id], 1, 0
                            )
                            if is_in_segment == 1:
                                if is_overlap == 1:
                                    T.copy(
                                        mask_diag[
                                            vid * v_block : vid * v_block + v_block,
                                            0:block_N,
                                        ],
                                        mask_ub,
                                    )
                                    T.tile.add(acc_s_ub, acc_s_ub, mask_ub)
                                else:
                                    T.tile.fill(mask_ub, -(2.0**30))
                                    T.tile.add(acc_s_ub, acc_s_ub, mask_ub)

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

                        T.copy(workspace_p[cid, :, :], acc_s_l1)

                        kv_packed_start_v = q_seq_starts[b_i] + kv_start
                        T.copy(
                            V[kv_packed_start_v : kv_packed_start_v + kv_size, h_kv, :],
                            v_l1[0:kv_size, :],
                        )

                        T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                        T.copy(acc_o_l0c, workspace_o[cid, :, :])

                        T.copy(
                            workspace_o[
                                cid,
                                vid * v_block : vid * v_block + v_block,
                                :,
                            ],
                            acc_o_ub,
                        )
                        T.tile.add(acc_o, acc_o, acc_o_ub)

                    for row in T.serial(v_block):
                        row_i = vid * v_block + row
                        if row_i < q_tile_size:
                            T.tile.div(
                                acc_o[row, :],
                                acc_o[row, :],
                                sumexp[row],
                            )

                    T.copy(acc_o, acc_o_half)
                    output_packed_start = q_packed_start + vid * v_block
                    T.copy(
                        acc_o_half,
                        Output[
                            output_packed_start : output_packed_start + v_block,
                            h_i,
                            :,
                        ],
                    )

    return main


def compute_split_points(seg_lengths, rules, block_M):
    splits = [0]
    tile_seg_ids = []
    for seg_id, (length, rule) in enumerate(zip(seg_lengths, rules)):
        seg_start = splits[-1]
        seg_end = seg_start + length
        pos = seg_start
        while pos < seg_end:
            next_pos = min(pos + block_M, seg_end)
            if next_pos not in splits:
                splits.append(next_pos)
                tile_seg_ids.append(seg_id)
            pos = next_pos
    q_seg_rule = [rules[sid] for sid in tile_seg_ids]
    return splits, q_seg_rule, tile_seg_ids


def compute_seg_first_tile(tile_seg_ids, num_segments):
    first_tile = [0] * (num_segments + 1)
    for seg_id in range(num_segments):
        found = False
        for i, sid in enumerate(tile_seg_ids):
            if sid == seg_id and not found:
                first_tile[seg_id] = i
                found = True
    first_tile[num_segments] = len(tile_seg_ids)
    for seg_id in range(1, num_segments):
        if first_tile[seg_id] == 0:
            first_tile[seg_id] = first_tile[seg_id - 1]
    return first_tile


def mtgr_ragged_segment_attention_wrapper(
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
    kv_heads = H // kv_group

    total_q = query.size(0)
    q_starts = q_seq_starts_i32.cpu().tolist()
    seg_lengths_list = []
    for b in range(B):
        offsets = segment_offsets_i32[b].cpu().tolist()
        lengths = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
        seg_lengths_list.append(lengths)

    max_splits = 0
    max_num_segments = 0
    all_split_points = []
    all_q_seg_rule = []
    all_tile_seg_id = []
    all_seg_first_tile = []

    for b in range(B):
        num_segments_b = len(seg_lengths_list[b])
        max_num_segments = max(max_num_segments, num_segments_b)
        splits, q_rule, t_seg_id = compute_split_points(
            seg_lengths_list[b], segment_rules_i32.cpu().tolist(), block_M
        )
        all_split_points.append(splits)
        all_q_seg_rule.append(q_rule)
        all_tile_seg_id.append(t_seg_id)
        all_seg_first_tile.append(compute_seg_first_tile(t_seg_id, num_segments_b))
        max_splits = max(max_splits, len(splits))

    max_q_tiles = max(len(r) for r in all_q_seg_rule)

    split_points_padded = []
    for splits in all_split_points:
        padded = splits + [0] * (max_splits - len(splits))
        split_points_padded.append(padded)
    split_points_i32 = torch.tensor(split_points_padded, dtype=torch.int32)

    split_count_i32 = torch.tensor([len(s) for s in all_split_points], dtype=torch.int32)

    tile_seg_id_padded = []
    for seg_ids in all_tile_seg_id:
        padded = seg_ids + [-1] * (max_q_tiles - len(seg_ids))
        tile_seg_id_padded.append(padded)
    tile_seg_id_i32 = torch.tensor(tile_seg_id_padded, dtype=torch.int32)

    seg_first_tile_padded = []
    for sft in all_seg_first_tile:
        padded = sft + [0] * (max_num_segments + 1 - len(sft))
        seg_first_tile_padded.append(padded)
    seg_first_tile_i32 = torch.tensor(seg_first_tile_padded, dtype=torch.int32)

    mask_causal = torch.full((block_M, block_N), float("-inf"), dtype=torch.float32)
    for i in range(block_M):
        for j in range(block_N):
            if j <= i:
                mask_causal[i, j] = 0.0

    mask_diag = torch.full((block_M, block_N), float("-inf"), dtype=torch.float32)
    for i in range(block_M):
        for j in range(block_N):
            if j == i:
                mask_diag[i, j] = 0.0

    group_batch_ids_list = []
    group_rules_list = []
    group_tile_offsets_list = []
    group_tile_counts_list = []
    sorted_s_local_list = []

    tile_offset_accum = 0
    for b in range(B):
        rules_in_batch = set(all_q_seg_rule[b])
        for rule_val in sorted(rules_in_batch):
            group_batch_ids_list.append(b)
            group_rules_list.append(rule_val)
            group_tile_offsets_list.append(tile_offset_accum)
            tiles_for_rule = [
                i for i, r in enumerate(all_q_seg_rule[b]) if r == rule_val
            ]
            group_tile_counts_list.append(len(tiles_for_rule))
            for s_local in tiles_for_rule:
                sorted_s_local_list.append(s_local)
            tile_offset_accum += len(tiles_for_rule)

    num_groups = len(group_batch_ids_list)
    total_seq_tiles = len(sorted_s_local_list)

    group_batch_ids_i32 = torch.tensor(group_batch_ids_list, dtype=torch.int32)
    group_rules_i32 = torch.tensor(group_rules_list, dtype=torch.int32)
    group_tile_offsets_i32 = torch.tensor(group_tile_offsets_list, dtype=torch.int32)
    group_tile_counts_i32 = torch.tensor(group_tile_counts_list, dtype=torch.int32)
    sorted_s_local_i32 = torch.tensor(sorted_s_local_list, dtype=torch.int32)

    func = mtgr_ragged_segment_attention_nokv(
        heads=H,
        dim=D,
        kv_group=kv_group,
        sm_scale=sm_scale,
        block_M=block_M,
        block_N=block_N,
        core_num=core_num,
        max_splits=max_splits,
        max_segments=max_num_segments,
    )
    print(func.get_kernel_source())

    output = func(
        query,
        key,
        value,
        q_seq_starts_i32,
        split_points_i32,
        split_count_i32,
        tile_seg_id_i32,
        seg_first_tile_i32,
        mask_causal,
        mask_diag,
        group_batch_ids_i32,
        group_rules_i32,
        group_tile_offsets_i32,
        group_tile_counts_i32,
        sorted_s_local_i32,
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
        seg_ids = torch.searchsorted(offsets_tensor[1:], logical_positions)

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
    B,
    H,
    D,
    kv_group,
    block_M,
    block_N,
    block_size,
    core_num,
    seg_lengths,
    rules,
    matched_prefix_arr,
):
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

    max_request_len = max(actual_q_len_arr)

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

    match_mode = 0
    sm_scale = 1.0 / math.sqrt(D)

    torch.npu.synchronize()
    print("init successful!")

    output_snd = mtgr_ragged_segment_attention_wrapper(
        query_snd,
        key_snd,
        value_snd,
        segment_offsets_i32,
        segment_rules_i32,
        q_seq_starts_i32,
        matched_prefix_lens_i32,
        match_mode,
        key_cache,
        value_cache,
        block_table_tensor,
        block_size,
        max_request_len,
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
        {
            "B": 1,
            "H": 8,
            "D": 64,
            "kv_group": 1,
            "block_M": 128,
            "block_N": 128,
            "block_size": 128,
            "core_num": 24,
            "seg_lengths": [[1600, 8, 200, 1200]],
            "rules": [0, 1, 2, 2],
            "matched_prefix_arr": [0],
        },
    ]

    for config in test_configs:
        test(**config)
