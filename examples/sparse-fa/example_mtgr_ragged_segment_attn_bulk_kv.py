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
    workspace_idx=[16, 17, 18, 19, 20],
    pass_configs=PASS_CONFIGS,
)
def mtgr_ragged_segment_attention_fwd_pa(
    heads,
    dim,
    kv_group=1,
    sm_scale=None,
    block_M=64,
    block_N=128,
    block_size=128,
    core_num=24,
):
    sm_scale = (1.0 / dim) ** 0.5 if sm_scale is None else sm_scale
    dtype = "bfloat16"
    accum_dtype = "float32"

    batch = T.symbolic("batch")
    total_live_q = T.symbolic("total_live_q")
    total_live_kv = T.symbolic("total_live_kv")
    max_request_len = T.symbolic("max_request_len")
    total_tasks = T.symbolic("total_tasks")
    num_blocks = T.symbolic("num_blocks")
    max_blocks = T.symbolic("max_blocks")

    kv_heads = heads // kv_group
    v_block = block_M // 2

    @T.prim_func
    def main(
        Q: T.Tensor([total_live_q, heads, dim], dtype),
        K: T.Tensor([total_live_kv, kv_heads, dim], dtype),
        V: T.Tensor([total_live_kv, kv_heads, dim], dtype),
        Output: T.Tensor([total_live_q, heads, dim], dtype),
        q_seq_starts: T.Tensor([batch], "int32"),
        kv_seq_starts: T.Tensor([batch], "int32"),
        actual_q_len: T.Tensor([batch], "int32"),
        actual_kv_len: T.Tensor([batch], "int32"),
        cum_seq_tiles_per_request: T.Tensor([batch], "int32"),
        visible_end: T.Tensor([batch, max_request_len], "float32"),
        diag_col: T.Tensor([batch, max_request_len], "float32"),
        matched_prefix_lens: T.Tensor([batch], "int32"),
        block_table: T.Tensor([batch, max_blocks], "int32"),
        key_cache: T.Tensor([num_blocks, block_size, kv_heads, dim], dtype),
        value_cache: T.Tensor([num_blocks, block_size, kv_heads, dim], dtype),
        task_meta: T.Tensor([total_tasks], "int32"),
        workspace_kv: T.Tensor([core_num, block_N, dim], dtype),
        workspace_kv_v: T.Tensor([core_num, block_N, dim], dtype),
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
            m_i_prev = T.alloc_ub([v_block], accum_dtype)
            acc_s_ub = T.alloc_ub([v_block, block_N], accum_dtype)
            acc_s_ub_ = T.alloc_ub([v_block, block_N], accum_dtype)
            sumexp_i_ub = T.alloc_ub([v_block], accum_dtype)
            acc_s_half = T.alloc_ub([v_block, block_N], dtype)
            acc_o_ub = T.alloc_ub([v_block, dim], accum_dtype)
            acc_o_half = T.alloc_ub([v_block, dim], dtype)
            kv_ub = T.alloc_ub([dim], dtype)

            visible_end_ub = T.alloc_ub([block_M], accum_dtype)
            diag_col_ub = T.alloc_ub([block_M], accum_dtype)
            kv_col_base_ub = T.alloc_ub([block_N], accum_dtype)
            kv_col_float_ub = T.alloc_ub([block_N], accum_dtype)
            mask_vis_ub = T.alloc_ub([block_N // 8], "uint8")
            mask_diag_ub = T.alloc_ub([block_N // 8], "uint8")
            mask_valid_ub = T.alloc_ub([block_N // 8], "uint8")
            mask_combined_ub = T.alloc_ub([block_N // 8], "uint8")

            max_iters = T.ceildiv(total_tasks, core_num)
            for core_index in T.serial(max_iters):
                pid = core_index * core_num + cid
                if pid < total_tasks:
                    flat_seq_tile = pid // heads
                    h_i = pid % heads
                    h_kv = h_i // kv_group

                    b_i = batch - 1
                    for _b in T.serial(batch):
                        cum_b = cum_seq_tiles_per_request[_b]
                        prev_cum = T.if_then_else(_b == 0, 0, cum_seq_tiles_per_request[_b - 1])
                        if prev_cum <= flat_seq_tile and flat_seq_tile < cum_b:
                            b_i = _b

                    cum_before = T.if_then_else(b_i == 0, 0, cum_seq_tiles_per_request[b_i - 1])
                    s_local = flat_seq_tile - cum_before

                    q_packed_start = q_seq_starts[b_i] + s_local * block_M
                    T.copy(
                        Q[
                            q_packed_start : q_packed_start + block_M,
                            h_i,
                            :,
                        ],
                        q_l1,
                    )

                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(m_i, -(2.0**30))

                    prefix_len_b = matched_prefix_lens[b_i]
                    kv_len_b = prefix_len_b + actual_kv_len[b_i]
                    kv_tiles_b = T.ceildiv(kv_len_b, block_N)

                    ve_start = s_local * block_M
                    T.copy(
                        visible_end[b_i, ve_start : ve_start + block_M],
                        visible_end_ub,
                    )
                    T.copy(
                        diag_col[b_i, ve_start : ve_start + block_M],
                        diag_col_ub,
                    )

                    for k_i in T.serial(kv_tiles_b):
                        kv_local_start = k_i * block_N

                        valid_cols = T.if_then_else(
                            kv_local_start + block_N > kv_len_b,
                            kv_len_b - kv_local_start,
                            block_N,
                        )

                        # === Optimized KV loading with 3-case branching ===
                        # block_N == block_size, so kv_local_start is aligned to block boundaries.
                        # Each vid handles half_N = block_N // 2 rows independently.
                        # Three cases per vid range:
                        #   Case 1: all cache  -> 1 bulk copy (half_N rows from key_cache)
                        #   Case 2: all live   -> 1 bulk copy (half_N rows from K)
                        #   Case 3: mixed      -> per-row fallback (same as original)

                        half_N = block_N // 2
                        vid_row_start = vid * half_N
                        vid_logical_start = kv_local_start + vid_row_start
                        vid_logical_end = vid_logical_start + half_N

                        # Case 1: entire vid range is cache
                        if vid_logical_end <= prefix_len_b:
                            cache_block_idx = vid_logical_start // block_size
                            physical_block = block_table[b_i, cache_block_idx]
                            block_offset_start = vid_logical_start % block_size
                            T.copy(
                                key_cache[physical_block, block_offset_start:block_offset_start + half_N, h_kv, :],
                                workspace_kv[cid, vid_row_start : vid_row_start + half_N, :],
                            )
                        # Case 2: entire vid range is live (no padding)
                        elif vid_logical_start >= prefix_len_b and vid_logical_end <= kv_len_b:
                            kv_packed_start = kv_seq_starts[b_i] + vid_logical_start - prefix_len_b
                            T.copy(
                                K[kv_packed_start : kv_packed_start + half_N, h_kv, :],
                                workspace_kv[cid, vid_row_start : vid_row_start + half_N, :],
                            )
                        # Case 3: mixed cache/live/pad — per-row fallback
                        else:
                            for row in range(half_N):
                                k_row = vid * half_N + row
                                kv_logical_pos = kv_local_start + k_row

                                if kv_logical_pos < prefix_len_b:
                                    block_idx = kv_logical_pos // block_size
                                    physical_block = block_table[b_i, block_idx]
                                    block_offset = kv_logical_pos % block_size
                                    T.copy(
                                        key_cache[physical_block, block_offset, h_kv, :],
                                        kv_ub,
                                    )
                                elif kv_logical_pos < kv_len_b:
                                    live_pos = kv_logical_pos - prefix_len_b
                                    kv_packed_pos = kv_seq_starts[b_i] + live_pos
                                    T.copy(
                                        K[kv_packed_pos, h_kv, :],
                                        kv_ub,
                                    )
                                else:
                                    T.tile.fill(kv_ub, 0.0)

                                T.copy(kv_ub, workspace_kv[cid, k_row, :])

                        # === End optimized KV loading ===

                        T.copy(workspace_kv[cid, :, :], k_l1)

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

                        T.tile.arith_progression(kv_col_base_ub, 0.0, 1.0, block_N)
                        T.tile.add(
                            kv_col_float_ub,
                            kv_col_base_ub,
                            T.float32(kv_local_start),
                        )

                        T.tile.compare(
                            mask_valid_ub,
                            kv_col_base_ub,
                            T.float32(valid_cols),
                            "LT",
                        )

                        for row in T.serial(v_block):
                            row_i = vid * v_block + row
                            T.tile.compare(
                                mask_vis_ub,
                                kv_col_float_ub,
                                visible_end_ub[row_i],
                                "LT",
                            )
                            T.tile.compare(
                                mask_diag_ub,
                                kv_col_float_ub,
                                diag_col_ub[row_i],
                                "EQ",
                            )
                            T.tile.bitwise_or(mask_combined_ub, mask_vis_ub, mask_diag_ub)
                            T.tile.bitwise_and(
                                mask_combined_ub,
                                mask_combined_ub,
                                mask_valid_ub,
                            )
                            T.tile.select(
                                acc_s_ub[row, :],
                                mask_combined_ub,
                                acc_s_ub[row, :],
                                -T.infinity(accum_dtype),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s_ub, m_i, dim=-1)
                        T.tile.max(m_i, m_i, m_i_prev)
                        T.tile.sub(m_i_prev, m_i_prev, m_i)
                        T.tile.exp(m_i_prev, m_i_prev)

                        for row in T.serial(v_block):
                            T.tile.sub(
                                acc_s_ub[row, :],
                                acc_s_ub[row, :],
                                m_i[row],
                            )
                        T.tile.exp(acc_s_ub, acc_s_ub)
                        T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                        T.tile.mul(sumexp, sumexp, m_i_prev)
                        T.tile.add(sumexp, sumexp, sumexp_i_ub)

                        for row in T.serial(v_block):
                            T.tile.mul(
                                acc_o[row, :],
                                acc_o[row, :],
                                m_i_prev[row],
                            )

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

                        # === Optimized V loading with same 3-case branching ===
                        if vid_logical_end <= prefix_len_b:
                            cache_block_idx = vid_logical_start // block_size
                            physical_block = block_table[b_i, cache_block_idx]
                            block_offset_start = vid_logical_start % block_size
                            T.copy(
                                value_cache[physical_block, block_offset_start:block_offset_start + half_N, h_kv, :],
                                workspace_kv_v[cid, vid_row_start : vid_row_start + half_N, :],
                            )
                        elif vid_logical_start >= prefix_len_b and vid_logical_end <= kv_len_b:
                            kv_packed_start = kv_seq_starts[b_i] + vid_logical_start - prefix_len_b
                            T.copy(
                                V[kv_packed_start : kv_packed_start + half_N, h_kv, :],
                                workspace_kv_v[cid, vid_row_start : vid_row_start + half_N, :],
                            )
                        else:
                            for row in range(half_N):
                                v_row = vid * half_N + row
                                kv_logical_pos = kv_local_start + v_row

                                if kv_logical_pos < prefix_len_b:
                                    block_idx = kv_logical_pos // block_size
                                    physical_block = block_table[b_i, block_idx]
                                    block_offset = kv_logical_pos % block_size
                                    T.copy(
                                        value_cache[physical_block, block_offset, h_kv, :],
                                        kv_ub,
                                    )
                                elif kv_logical_pos < kv_len_b:
                                    live_pos = kv_logical_pos - prefix_len_b
                                    kv_packed_pos = kv_seq_starts[b_i] + live_pos
                                    T.copy(
                                        V[kv_packed_pos, h_kv, :],
                                        kv_ub,
                                    )
                                else:
                                    T.tile.fill(kv_ub, 0.0)

                                T.copy(kv_ub, workspace_kv_v[cid, v_row, :])

                        # === End optimized V loading ===

                        T.copy(workspace_kv_v[cid, :, :], v_l1)

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
                        if visible_end_ub[row_i] >= 0.0:
                            T.tile.div(
                                acc_o[row, :],
                                acc_o[row, :],
                                sumexp[row_i],
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


def compute_visible_end_diag_col(offsets, rules, seq_len, matched_prefix=0):
    visible_end = torch.full((seq_len,), -1.0, dtype=torch.float32)
    diag_col = torch.full((seq_len,), -1.0, dtype=torch.float32)
    for s in range(seq_len):
        logical_pos = matched_prefix + s
        for seg_id in range(len(rules)):
            if logical_pos < offsets[seg_id + 1]:
                r = rules[seg_id]
                if r == 0:
                    visible_end[s] = float(logical_pos + 1)
                elif r == 1:
                    visible_end[s] = float(offsets[seg_id + 1])
                elif r == 2:
                    visible_end[s] = float(offsets[seg_id])
                    diag_col[s] = float(logical_pos)
                break
    return visible_end, diag_col


def build_attention_mask(visible_end, diag_col, S_kv):
    mask_val = torch.zeros(S_kv, dtype=torch.float32)
    for k in range(S_kv):
        ve = visible_end.item()
        dc = diag_col.item()
        if k < ve or k == dc:
            mask_val[k] = 1.0
    return mask_val


def mtgr_ragged_segment_attention_wrapper(
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
    block_table_i32,
    block_size,
    max_request_len,
    sm_scale,
    block_M=64,
    block_N=128,
    core_num=24,
    kv_group=1,
):

    B = segment_offsets_i32.size(0)
    num_segments = segment_rules_i32.size(0)
    H = query_snd.size(1)
    D = query_snd.size(2)
    kv_heads = H // kv_group

    print("start compile kernel")
    func = mtgr_ragged_segment_attention_fwd_pa(
            heads=H,
            dim=D,
            kv_group=kv_group,
            sm_scale=sm_scale,
            block_M=block_M,
            block_N=block_N,
            block_size=block_size,
            core_num=core_num,
        )
    print(func.get_kernel_source())

    total_live_q = query_snd.size(0)
    total_live_kv = key_snd.size(0)

    q_starts = q_seq_starts_i32.cpu().tolist()
    kv_starts = q_seq_starts_i32.cpu().tolist()
    matched_prefix_list = matched_prefix_lens_i32.cpu().tolist()

    actual_q_len_arr = []
    actual_kv_len_arr = []
    for b in range(B):
        next_start = q_starts[b + 1] if b + 1 < B else total_live_q
        q_len = next_start - q_starts[b]
        actual_q_len_arr.append(q_len)
        kv_len = q_len
        actual_kv_len_arr.append(kv_len)

    max_req_len_padded = ((max(actual_q_len_arr) + block_M - 1) // block_M) * block_M

    visible_end_list = []
    diag_col_list = []
    for b in range(B):
        offset_list = segment_offsets_i32[b].cpu().tolist()
        rule_list = segment_rules_i32.cpu().tolist()
        ve, dc = compute_visible_end_diag_col(offset_list, rule_list, actual_q_len_arr[b], matched_prefix_list[b])
        ve_padded = torch.full((max_req_len_padded,), -1.0, dtype=torch.float32)
        dc_padded = torch.full((max_req_len_padded,), -1.0, dtype=torch.float32)
        ve_padded[: actual_q_len_arr[b]] = ve
        dc_padded[: actual_q_len_arr[b]] = dc
        visible_end_list.append(ve_padded)
        diag_col_list.append(dc_padded)

    visible_end_tensor = torch.stack(visible_end_list)
    diag_col_tensor = torch.stack(diag_col_list)

    num_seq_tiles = [(ql + block_M - 1) // block_M for ql in actual_q_len_arr]
    cum_seq_tiles_arr = []
    cum = 0
    for nt in num_seq_tiles:
        cum += nt
        cum_seq_tiles_arr.append(cum)
    total_tasks = cum_seq_tiles_arr[-1] * H

    actual_q_len_t = torch.tensor(actual_q_len_arr, dtype=torch.int32)
    actual_kv_len_t = torch.tensor(actual_kv_len_arr, dtype=torch.int32)
    kv_seq_starts_t = torch.tensor(kv_starts, dtype=torch.int32)
    cum_seq_tiles_t = torch.tensor(cum_seq_tiles_arr, dtype=torch.int32)

    task_meta_t = torch.zeros(total_tasks, dtype=torch.int32)
    wk = torch.zeros(core_num, block_N, D, dtype=torch.bfloat16)
    wk_v = torch.zeros(core_num, block_N, D, dtype=torch.bfloat16)
    ws = torch.zeros(core_num, block_M, block_N, dtype=torch.float32)
    wp = torch.zeros(core_num, block_M, block_N, dtype=torch.bfloat16)
    wo = torch.zeros(core_num, block_M, D, dtype=torch.float32)

    output = func(
        query_snd,
        key_snd,
        value_snd,
        q_seq_starts_i32,
        kv_seq_starts_t,
        actual_q_len_t,
        actual_kv_len_t,
        cum_seq_tiles_t,
        visible_end_tensor,
        diag_col_tensor,
        matched_prefix_lens_i32,
        block_table_i32,
        key_cache,
        value_cache,
        task_meta_t,
        wk,
        wk_v,
        ws,
        wp,
        wo,
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

    actual_q_len_arr = []
    for b in range(B):
        next_start = q_starts[b + 1] if b + 1 < B else total_live_q
        actual_q_len_arr.append(next_start - q_starts[b])

    ref_outputs = []
    for b in range(B):
        q_start = q_starts[b]
        q_len = actual_q_len_arr[b]
        q_b = query_snd[q_start : q_start + q_len].float()

        prefix_len = matched_prefix_list[b]

        k_prefix_b_list = []
        v_prefix_b_list = []
        for p in range(prefix_len):
            block_idx = p // block_size
            block_offset = p % block_size
            phys_block = block_table_i32[b, block_idx].item()
            k_prefix_b_list.append(key_cache[phys_block, block_offset].float().unsqueeze(0))
            v_prefix_b_list.append(value_cache[phys_block, block_offset].float().unsqueeze(0))

        kv_start = q_starts[b]
        kv_len = q_len
        k_live_b = key_snd[kv_start : kv_start + kv_len].float()
        v_live_b = value_snd[kv_start : kv_start + kv_len].float()

        if prefix_len > 0:
            k_prefix_b = torch.cat(k_prefix_b_list, dim=0)
            v_prefix_b = torch.cat(v_prefix_b_list, dim=0)
            k_full_b = torch.cat([k_prefix_b, k_live_b], dim=0)
            v_full_b = torch.cat([v_prefix_b, v_live_b], dim=0)
        else:
            k_full_b = k_live_b
            v_full_b = v_live_b

        total_kv_len_b = prefix_len + kv_len

        offset_list = segment_offsets_i32[b].cpu().tolist()
        rule_list = segment_rules_i32.cpu().tolist()
        ve_b, dc_b = compute_visible_end_diag_col(offset_list, rule_list, q_len, prefix_len)
        if isinstance(ve_b, torch.Tensor):
            ve_b = ve_b.cpu()
            dc_b = dc_b.cpu()

        masks_b = []
        for s in range(q_len):
            mask_row = build_attention_mask(ve_b[s], dc_b[s], total_kv_len_b)
            masks_b.append(mask_row)
        mask_b = torch.stack(masks_b)

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
    actual_kv_len_arr = actual_q_len_arr[:]

    q_seq_starts_arr = [0]
    kv_seq_starts_arr = [0]
    for b in range(1, B):
        q_seq_starts_arr.append(q_seq_starts_arr[b - 1] + actual_q_len_arr[b - 1])
        kv_seq_starts_arr.append(kv_seq_starts_arr[b - 1] + actual_kv_len_arr[b - 1])

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
    B = 2
    H = 4
    D = 128
    kv_group = 1
    block_M = 64
    block_N = 128
    block_size = 128
    core_num = 24

    seg_lengths = [[128, 8, 32, 24], [256, 8, 32, 64]]
    rules = [0, 1, 0, 2]
    matched_prefix_arr = [0, 140]

    test(
        B=B,
        H=H,
        D=D,
        kv_group=kv_group,
        block_M=block_M,
        block_N=block_N,
        block_size=block_size,
        core_num=core_num,
        seg_lengths=seg_lengths,
        rules=rules,
        matched_prefix_arr=matched_prefix_arr,
    )