import math
import random
import torch


def prepare_data(config, seed=0):
    H = config["H"]
    D = config["D"]
    seg_lengths = config["seg_lengths"]
    rules = config["rules"]
    matched_prefix_arr = config["matched_prefix_arr"]
    kv_group = config.get("kv_group", 1)
    block_N = config.get("block_N", 128)
    block_size = block_N

    B = len(seg_lengths)
    kv_heads = H // kv_group

    for b_idx, plen in enumerate(matched_prefix_arr):
        if plen % block_N != 0:
            raise ValueError(f"matched_prefix_arr[{b_idx}] = {plen} is not a multiple of block_N={block_N}")

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

    torch.manual_seed(seed)

    q_list, k_list_full, v_list_full, k_list_live, v_list_live = [], [], [], [], []
    for b in range(B):
        q_b = torch.randn(actual_q_len_arr[b], H, D, dtype=torch.float32, device="cpu").to(torch.bfloat16)
        k_b_full = torch.randn(S_logical_list[b], kv_heads, D, dtype=torch.float32, device="cpu").to(torch.bfloat16)
        v_b_full = torch.randn(S_logical_list[b], kv_heads, D, dtype=torch.float32, device="cpu").to(torch.bfloat16)
        q_list.append(q_b)
        k_list_full.append(k_b_full)
        v_list_full.append(v_b_full)
        prefix_len = matched_prefix_arr[b]
        k_list_live.append(k_b_full[prefix_len:])
        v_list_live.append(v_b_full[prefix_len:])

    query_snd = torch.cat(q_list, dim=0)
    key_snd = torch.cat(k_list_live, dim=0)
    value_snd = torch.cat(v_list_live, dim=0)

    num_cache_blocks = max(1, sum((matched_prefix_arr[b] + block_size - 1) // block_size for b in range(B)))
    key_cache = torch.zeros(num_cache_blocks, block_size, kv_heads, D, dtype=torch.bfloat16, device="cpu")
    value_cache = torch.zeros(num_cache_blocks, block_size, kv_heads, D, dtype=torch.bfloat16, device="cpu")

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

    max_blocks_per_request = max(1, max(len(bt) for bt in block_table_arr))
    block_table_tensor = torch.zeros(B, max_blocks_per_request, dtype=torch.int32, device="cpu")
    for b in range(B):
        for lb in range(len(block_table_arr[b])):
            block_table_tensor[b, lb] = block_table_arr[b][lb]

    segment_offsets_i32 = torch.tensor(offsets_list, dtype=torch.int32, device="cpu")
    segment_rules_i32 = torch.tensor(rules, dtype=torch.int32, device="cpu")
    q_seq_starts_i32 = torch.tensor(q_seq_starts_arr, dtype=torch.int32, device="cpu")
    matched_prefix_lens_i32 = torch.tensor(matched_prefix_arr, dtype=torch.int32, device="cpu")

    sm_scale = 1.0 / math.sqrt(D)

    return dict(
        query_snd=query_snd,
        key_snd=key_snd,
        value_snd=value_snd,
        segment_offsets_i32=segment_offsets_i32,
        segment_rules_i32=segment_rules_i32,
        q_seq_starts_i32=q_seq_starts_i32,
        matched_prefix_lens_i32=matched_prefix_lens_i32,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table_tensor=block_table_tensor,
        num_cache_blocks=num_cache_blocks,
        sm_scale=sm_scale,
        H=H,
        D=D,
        kv_group=kv_group,
        block_size=block_size,
    )


# ---------------------------------------------------------------------------
# Pattern helpers
# ---------------------------------------------------------------------------


def _make_pattern(seg_lengths, rules=None, matched_prefix_arr=None):
    n = len(seg_lengths[0])
    if rules is None:
        rules = [0, 1] + [2] * (n - 2)
    if matched_prefix_arr is None:
        matched_prefix_arr = [0] * len(seg_lengths)
    return {
        "seg_lengths": seg_lengths,
        "rules": rules,
        "matched_prefix_arr": matched_prefix_arr,
    }


def _gen_uniform(num_seg, seg_len, B, prefix=1600, suffix=1200, matched_prefix=0):
    sl = [prefix, 8] + [seg_len] * num_seg + [suffix]
    return _make_pattern([sl] * B, matched_prefix_arr=[matched_prefix] * B)


_PREFIX_POOL = [1600, 2000, 2400, 3200]
_SUFFIX_POOL = [1024, 1200, 1800, 2048]


def _gen_random_ps(num_seg, seg_len, B, rng):
    return _gen_uniform(
        num_seg,
        seg_len,
        B,
        prefix=rng.choice(_PREFIX_POOL),
        suffix=rng.choice(_SUFFIX_POOL),
    )


_SEG_NUM_MU = math.log(50)
_SEG_NUM_SIGMA = 1.4
_MAX_CORE_LEN = 32768


def _sample_seg_num(rng):
    x = rng.lognormvariate(_SEG_NUM_MU, _SEG_NUM_SIGMA)
    n = min(max(int(round(x)), 1), 1024)
    if n * 50 > _MAX_CORE_LEN:
        n = _MAX_CORE_LEN // 50
    return n


def _sample_seg_len(rng):
    r = rng.random()
    if r < 0.90:
        return rng.randint(5, 15)
    elif r < 0.95:
        return rng.randint(2, 4)
    else:
        return rng.randint(16, 50)


# ---------------------------------------------------------------------------
# Pattern generators
# ---------------------------------------------------------------------------

_base_patterns = [
    # 4-seg, rules [0, 1, 2, 2]
    _make_pattern([[1600, 8, 200, 1200]], [0, 1, 2, 2]),
    _make_pattern([[1600, 8, 200, 1200], [1700, 8, 300, 1024]], [0, 1, 2, 2]),
    _make_pattern(
        [
            [1600, 8, 200, 1200],
            [1700, 8, 300, 1024],
            [1680, 8, 200, 1280],
            [2000, 8, 700, 2048],
        ],
        [0, 1, 2, 2],
    ),
    _make_pattern(
        [
            [2200, 8, 200, 1024],
            [1700, 8, 100, 1100],
            [2440, 8, 200, 2048],
            [1600, 8, 600, 1900],
            [3300, 8, 200, 1300],
            [1700, 8, 300, 2100],
            [1780, 8, 700, 1200],
            [2048, 8, 500, 1800],
        ],
        [0, 1, 2, 2],
    ),
    # various seg counts, rules auto-derived as [0, 1, 2, ..., 2]
    _make_pattern([[1600, 8, 10, 1200]]),
    _make_pattern([[1600, 8, 8, 12, 1200], [1700, 8, 6, 15, 1024]]),
    _make_pattern(
        [
            [1600, 8, 5, 6, 7, 8, 1200],
            [1700, 8, 10, 12, 13, 15, 1024],
            [1680, 8, 10, 12, 13, 15, 1280],
            [2000, 8, 13, 14, 15, 15, 2048],
        ]
    ),
    _make_pattern(
        [
            [2200, 8, 5, 5, 5, 5, 5, 5, 5, 5, 1024],
            [1700, 8, 5, 5, 5, 5, 5, 5, 5, 10, 1100],
            [2440, 8, 5, 5, 5, 5, 5, 5, 5, 12, 2048],
            [1600, 8, 5, 15, 5, 5, 5, 5, 5, 5, 1900],
            [3300, 8, 5, 10, 10, 5, 5, 5, 5, 5, 1300],
            [1700, 8, 15, 15, 5, 5, 5, 5, 5, 5, 2100],
            [1780, 8, 15, 15, 15, 5, 5, 5, 5, 5, 1200],
            [2048, 8, 15, 15, 15, 15, 5, 5, 5, 5, 1800],
        ]
    ),
]


def _generate_multi_seg():
    cases = [1, 2, 4, 7, 8, 15, 16, 25, 32, 43, 63, 64, 85, 119, 128, 143, 256, 512, 1024]
    return [_gen_uniform(ns, 5, 1) for ns in cases]


def _generate_big_seg():
    cases = [4, 8, 16, 32, 64, 128, 256, 512]
    return [_gen_uniform(1, sl, 1) for sl in cases]


def _generate_multi_seg_patterns():
    rng = random.Random(42)
    cases = [
        (64, 16, 4),
        (64, 16, 2),
        (64, 16, 1),
        (128, 16, 2),
        (128, 16, 1),
        (256, 16, 1),
        (512, 16, 1),
        (64, 32, 2),
        (64, 64, 2),
        (64, 64, 1),
        (64, 128, 1),
        (128, 32, 2),
        (128, 32, 1),
        (128, 64, 1),
        (256, 32, 1),
    ]
    return [_gen_random_ps(ns, sl, b, rng) for ns, sl, b in cases]


def _generate_constraint_patterns():
    rng = random.Random(123)
    cases = [
        (32, 5, 4),
        (32, 5, 2),
        (32, 5, 1),
        (32, 7, 2),
        (32, 7, 1),
        (32, 10, 2),
        (32, 10, 1),
        (32, 12, 1),
        (32, 15, 1),
        (50, 5, 2),
        (50, 5, 1),
        (50, 7, 1),
        (50, 10, 1),
        (50, 12, 1),
        (50, 15, 1),
        (64, 5, 1),
        (64, 7, 1),
        (64, 10, 1),
        (64, 12, 1),
        (64, 15, 1),
    ]
    return [_gen_random_ps(ns, sl, b, rng) for ns, sl, b in cases]


def _generate_prefix_patterns():
    cases = [(ns, 5, 1, pl) for ns in [1, 4, 8, 16, 32] for pl in [128, 256, 512, 1024]] + [
        (4, 5, 2, 128),
        (4, 5, 2, 512),
        (8, 5, 4, 256),
        (8, 5, 4, 1024),
    ]
    return [_gen_uniform(ns, sl, b, matched_prefix=pl) for ns, sl, b, pl in cases]


def _generate_sampled_patterns():
    rng = random.Random(777)
    patterns = []
    B_choices = [1, 2, 4, 8]
    for _ in range(50):
        B = rng.choice(B_choices)
        num_seg = _sample_seg_num(rng)
        seg_lengths_all = []
        for _ in range(B):
            prefix = rng.choice(_PREFIX_POOL)
            suffix = rng.choice(_SUFFIX_POOL)
            seg_lens = [_sample_seg_len(rng) for _ in range(num_seg)]
            seg_lengths_all.append([prefix, 8] + seg_lens + [suffix])
        rules = [0, 1] + [2] * (len(seg_lengths_all[0]) - 2)
        patterns.append(_make_pattern(seg_lengths_all, rules=rules))
    return patterns


# ---------------------------------------------------------------------------
# Assemble test configs
# ---------------------------------------------------------------------------

case = []
case.extend(_base_patterns)
case.extend(_generate_multi_seg())
case.extend(_generate_big_seg())
case.extend(_generate_multi_seg_patterns())
case.extend(_generate_constraint_patterns())
case.extend(_generate_prefix_patterns())
case.extend(_generate_sampled_patterns())

_D_values = [128, 64, 32]

test_configs = [{"H": 8, "D": d, **pattern} for d in _D_values for pattern in case]

# ---------------------------------------------------------------------------
# GQA test configs (kv_group > 1)
# ---------------------------------------------------------------------------

_gqa_patterns = [
    # P0: 基础单 batch 4-seg
    _make_pattern([[1600, 8, 200, 1200]], [0, 1, 2, 2]),
    # P1: 多 batch(4) 4-seg
    _make_pattern(
        [
            [1600, 8, 200, 1200],
            [1700, 8, 300, 1024],
            [1680, 8, 200, 1280],
            [2000, 8, 700, 2048],
        ],
        [0, 1, 2, 2],
    ),
    # P2: prefix + paged cache
    _make_pattern([[1600, 8, 200, 1200]], [0, 1, 2, 2], matched_prefix_arr=[512]),
    # P3: 多 segment(16 段)
    _make_pattern([[1600, 8] + [5] * 16 + [1200]]),
    # P4: 多 batch 多 seg
    _make_pattern([[1600, 8, 10, 12, 1200], [1700, 8, 6, 15, 1024]]),
]

_gqa_kv_groups = [2, 4, 8]
_gqa_D_values = [128, 64]

test_configs += [{"H": 8, "D": d, "kv_group": kg, **p} for d in _gqa_D_values for kg in _gqa_kv_groups for p in _gqa_patterns]

# ---------------------------------------------------------------------------
# Per-case test flags
# ---------------------------------------------------------------------------
# multi_seed: accuracy 模式下是否跑多种子（CPU golden 开销大，每类仅 1-2 条）
# perf_test:  compare 模式下是否跑性能测试
#
# 索引布局（共 462 条）：
#   D=128:  idx 0-143   (144 条 MHA) + idx 432-446 (15 条 GQA)
#   D=64:   idx 144-287 (144 条 MHA) + idx 447-461 (15 条 GQA)
#   D=32:   idx 288-431 (144 条 MHA, 无 GQA)
#
# 各类别边界（在每个 D 内）：
#   base:                 0-7
#   multi_seg:            8-26
#   big_seg:              27-34
#   multi_seg_patterns:   35-49
#   constraint:           50-69
#   prefix:               70-93
#   sampled:              94-143

_multi_seed_indices = {
    # D=128
    0,
    3,  # base: 单batch基础 + 多batch(8)
    14,  # multi_seg: segs=19
    30,  # big_seg: seg_len=64
    37,  # multi_seg_patterns: B=1, segs=67
    52,  # constraint: B=1, segs=35
    72,  # prefix: pf=512, segs=4
    98,  # sampled: B=1, segs=51
    437,
    442,  # GQA4 + GQA8(MQA)
    # D=64
    144,  # base: 单batch基础
    216,  # prefix: pf=512, segs=4
    # D=32
    288,  # base: 单batch基础
    360,  # prefix: pf=512, segs=4
}

_perf_test_indices = {
    # === D=128 (33 条) ===
    # base: B=1/2/4/8
    0,
    1,
    2,
    3,
    # multi_seg: segs=4/19/66/131/1027
    8,
    14,
    18,
    22,
    26,
    # big_seg: seg_len=4/64/512
    27,
    30,
    34,
    # multi_seg_patterns: B=4/B=1/B=2
    35,
    37,
    42,
    # constraint: 不同约束组合
    50,
    52,
    58,
    # prefix: pf=128/512/1024 + B=2
    70,
    72,
    73,
    90,
    # sampled: 随机场景采样
    94,
    96,
    98,
    103,
    114,
    118,
    # GQA: GQA2(B=1/B=4) + GQA4 + GQA8(B=1/pf=512)
    432,
    433,
    437,
    442,
    444,
    # === D=64 (15 条) ===
    144,
    147,
    152,
    162,
    170,
    171,
    178,
    181,
    196,
    216,
    217,
    240,
    247,
    262,
    452,
    # === D=32 (15 条) ===
    288,
    291,
    296,
    306,
    314,
    315,
    322,
    325,
    340,
    360,
    361,
    384,
    391,
    402,
    406,
}

for _idx in range(len(test_configs)):
    test_configs[_idx]["multi_seed"] = _idx in _multi_seed_indices
    test_configs[_idx]["perf_test"] = _idx in _perf_test_indices
