import torch


def _prepare_kv_and_mask(
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
):
    query_snd = query_snd.cpu()
    key_snd = key_snd.cpu()
    value_snd = value_snd.cpu()
    segment_offsets_i32 = segment_offsets_i32.cpu()
    segment_rules_i32 = segment_rules_i32.cpu()
    q_seq_starts_i32 = q_seq_starts_i32.cpu()
    matched_prefix_lens_i32 = matched_prefix_lens_i32.cpu()
    key_cache = key_cache.cpu()
    value_cache = value_cache.cpu()
    block_table_i32 = block_table_i32.cpu()

    B = segment_offsets_i32.size(0)
    H = query_snd.size(1)
    D = query_snd.size(2)
    kv_heads = key_snd.size(1)
    kv_group = H // kv_heads

    q_starts = q_seq_starts_i32.tolist()
    matched_prefix_list = matched_prefix_lens_i32.tolist()
    q_lens = [q_starts[b + 1] - q_starts[b] for b in range(B)]

    per_batch = []
    for b in range(B):
        q_start = q_starts[b]
        q_len = q_lens[b]
        q_b = query_snd[q_start : q_start + q_len]

        prefix_len = matched_prefix_list[b]

        if prefix_len > 0:
            num_prefix_blocks = (prefix_len + block_size - 1) // block_size
            phys_blocks = block_table_i32[b, :num_prefix_blocks]
            k_prefix_b = key_cache[phys_blocks].reshape(-1, kv_heads, D)[:prefix_len]
            v_prefix_b = value_cache[phys_blocks].reshape(-1, kv_heads, D)[:prefix_len]

        k_live_b = key_snd[q_start : q_start + q_len]
        v_live_b = value_snd[q_start : q_start + q_len]

        if prefix_len > 0:
            k_full_b = torch.cat([k_prefix_b, k_live_b], dim=0)
            v_full_b = torch.cat([v_prefix_b, v_live_b], dim=0)
        else:
            k_full_b = k_live_b
            v_full_b = v_live_b

        if kv_group > 1:
            k_full_b = k_full_b.repeat_interleave(kv_group, dim=1)
            v_full_b = v_full_b.repeat_interleave(kv_group, dim=1)

        total_kv_len_b = prefix_len + q_len
        offsets = segment_offsets_i32[b].tolist()
        rules = segment_rules_i32.tolist()

        q_abs_positions = torch.arange(prefix_len, total_kv_len_b)
        k_abs_positions = torch.arange(total_kv_len_b)

        offsets_tensor = torch.tensor(offsets, dtype=torch.int32)
        seg_ids = torch.searchsorted(offsets_tensor[1:], q_abs_positions, right=True)

        full_mask = torch.zeros(q_len, total_kv_len_b, dtype=torch.float32)

        for seg_id_val in range(len(rules)):
            rule = rules[seg_id_val]
            q_indices = (seg_ids == seg_id_val).nonzero().squeeze(-1)
            if q_indices.numel() == 0:
                continue

            q_abs = q_abs_positions[q_indices]

            if rule == 0:
                causal_mask = k_abs_positions.unsqueeze(0) <= q_abs.unsqueeze(1)
                full_mask[q_indices] = causal_mask.float()
            elif rule == 1:
                end = min(offsets[seg_id_val + 1], total_kv_len_b)
                full_mask[q_indices, :end] = 1.0
            elif rule == 2:
                start = min(offsets[seg_id_val], total_kv_len_b)
                full_mask[q_indices, :start] = 1.0
                full_mask[q_indices, q_abs] = 1.0

        per_batch.append(
            {
                "q_b": q_b,
                "k_full_b": k_full_b,
                "v_full_b": v_full_b,
                "full_mask": full_mask,
            }
        )

    return per_batch, kv_group


def golden_attention_float64(
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
    per_batch, kv_group = _prepare_kv_and_mask(
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
    )

    ref_outputs = []
    for item in per_batch:
        q = item["q_b"].double()
        k = item["k_full_b"].double()
        v = item["v_full_b"].double()
        full_mask = item["full_mask"]

        scores = torch.einsum("qhd,khd->hqk", q, k) * sm_scale
        scores = scores.masked_fill(full_mask.unsqueeze(0) == 0.0, float("-inf"))
        probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        ref_b = torch.einsum("hqk,khd->qhd", probs, v)
        ref_outputs.append(ref_b)

    return torch.cat(ref_outputs, dim=0)


def golden_attention_simulated_kernel(
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
    per_batch, kv_group = _prepare_kv_and_mask(
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
    )

    ref_outputs = []
    for item in per_batch:
        q = item["q_b"].to(torch.bfloat16)
        k = item["k_full_b"].to(torch.bfloat16)
        v = item["v_full_b"].to(torch.bfloat16)
        full_mask = item["full_mask"]

        S = torch.einsum("qhd,khd->hqk", q, k)
        S = (S * sm_scale).to(torch.bfloat16)
        S = S.masked_fill(full_mask.unsqueeze(0) == 0.0, float("-inf"))

        m = torch.max(S, dim=-1, keepdim=True).values
        P = torch.exp((S - m).float()).to(torch.bfloat16).nan_to_num(0.0)

        O = torch.einsum("hqk,khd->qhd", P, v).to(torch.bfloat16)

        sumexp = torch.sum(P.float(), dim=-1, keepdim=True).permute(1, 0, 2)
        out = (O.float() / sumexp.clamp(min=1.0)).to(torch.bfloat16)
        ref_outputs.append(out)

    return torch.cat(ref_outputs, dim=0)
