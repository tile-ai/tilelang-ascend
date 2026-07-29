import tilelang
import tilelang.language as T
from tilelang import jit
from tilelang.intrinsics import make_zn_layout, make_nz_layout
import torch
from typing import Optional, Tuple

try:
    import torch_npu
except ImportError:
    torch_npu = None


def _get_cube_core_num() -> int:
    try:
        import torch_npu as _tnpu

        props = _tnpu.npu.get_device_properties(0)
        return props.cube_core_num
    except Exception:
        return 20


# ============================================================
# Host-side layout conversion helpers
# ============================================================


def _to_per_batch_tensor(seq_len, B: int, device) -> torch.Tensor:
    """PA_BSND case: seq_len is required, only do device/dtype conversion.

    Returns: per-batch int32 tensor of shape [B].
    """
    assert seq_len is not None, "PA_BSND case requires actual_seq_lengths"
    return seq_len.to(device=device, dtype=torch.int32)


def _to_per_batch_tensor_or_default(seq_len, B: int, default_S: int, device) -> torch.Tensor:
    """BSND case: seq_len may be None, fall back to default_S for all batches.

    Returns: per-batch int32 tensor of shape [B].
    """
    if seq_len is None:
        return torch.full((B,), default_S, dtype=torch.int32, device=device)
    return seq_len.to(device=device, dtype=torch.int32)


def _tnd_to_bsnd_query(
    query: torch.Tensor,
    asq: torch.Tensor,
) -> tuple:
    """Convert TND query [T, N1, D] to BSND padded [B, S, N1, D].

    Args:
        query: TND query tensor of shape [T, N1, D].
        asq: actual_seq_lengths_query in prefix-sum format, shape [B], int32.

    Returns:
        query_bsnd: BSND padded query [B, S, N1, D], padding rows are 0.
        actual_q_len: per-batch lengths [B], int32.
        S1: max(per-batch lengths).
        B: number of batches.
        N1: head number.
        D: head dim.
        q_tot: original T (for output reverse conversion).
    """
    q_tot, N1, D = query.shape
    asq = asq.to(device=query.device, dtype=torch.int32)
    B = asq.shape[0]
    # prefix-sum -> per-batch lengths
    q_lens = torch.zeros(B, dtype=torch.int32, device=query.device)
    q_lens[0] = asq[0]
    q_lens[1:] = asq[1:] - asq[:-1]
    S1 = int(q_lens.max().item())
    # Build BSND padded (padding 0)
    s_arange = torch.arange(S1, device=query.device).unsqueeze(0)  # [1, S1]
    mask = s_arange < q_lens.unsqueeze(1)  # [B, S1] bool
    # Build TND row index for each (b, s): asq_prefix[b] + s (if valid else 0)
    asq_prefix = torch.zeros(B, dtype=torch.int32, device=query.device)
    asq_prefix[1:] = asq[:-1]
    t_row_idx = asq_prefix.unsqueeze(1) + s_arange  # [B, S1]
    t_row_idx = torch.where(mask, t_row_idx, torch.zeros_like(t_row_idx))  # invalid -> 0
    # Gather: query_bsnd[b, s] = query[t_row_idx[b, s]]
    query_bsnd = query[t_row_idx]  # [B, S1, N1, D]
    # Zero out padding rows (in case query[0] is not 0, though t_row_idx=0 for invalid)
    query_bsnd = query_bsnd * mask.unsqueeze(-1).unsqueeze(-1).to(query.dtype)
    return query_bsnd, q_lens, S1, B, N1, D, q_tot


def _tnd_to_bsnd_weights(
    weights: torch.Tensor,
    actual_q_len: torch.Tensor,
    B: int,
    S1: int,
    N1: int,
) -> torch.Tensor:
    """Convert TND weights [T, N1] to BSND padded [B, S, N1].

    Args:
        weights: TND weights [T, N1].
        actual_q_len: per-batch lengths [B], int32.
        B, S1, N1: target BSND shape.

    Returns:
        weights_bsnd: BSND padded weights [B, S1, N1], padding 0.
    """
    T, N1_w = weights.shape
    assert N1_w == N1
    device = weights.device
    # Build prefix-sum from actual_q_len
    asq = torch.zeros(B, dtype=torch.int32, device=device)
    asq[0] = actual_q_len[0]
    for i in range(1, B):
        asq[i] = asq[i - 1] + actual_q_len[i]
    weights_bsnd, _, _, _, _, _, _ = _tnd_to_bsnd_query(
        weights.unsqueeze(-1),  # [T, N1, 1]
        asq,
    )
    return weights_bsnd.squeeze(-1)  # [B, S1, N1]


def _tnd_to_bsnd_key(
    key: torch.Tensor,
    ask: torch.Tensor,
    B: int,
    device,
) -> tuple:
    """Convert TND key [T, N2, D] to BSND padded [B, S, N2, D].

    Args:
        key: TND key tensor of shape [T, N2, D].
        ask: actual_seq_lengths_key in prefix-sum format, shape [B], int32.
        B: number of batches.
        device: target device.

    Returns:
        key_bsnd: BSND padded key [B, S, N2, D], padding 0.
        actual_k_len: per-batch lengths [B], int32.
        S2: max(per-batch lengths, 257).
        max_block_num: 1 (TND key -> BSND, no PA).
        block_size: 128 (default for BSND).
    """
    k_tot, N2, D = key.shape
    ask = ask.to(device=device, dtype=torch.int32)
    # prefix-sum -> per-batch lengths
    k_lens = torch.zeros(B, dtype=torch.int32, device=device)
    k_lens[0] = ask[0]
    k_lens[1:] = ask[1:] - ask[:-1]
    S2 = max(int(k_lens.max().item()), 257)
    # Build BSND padded
    s_arange = torch.arange(S2, device=device).unsqueeze(0)  # [1, S2]
    mask = s_arange < k_lens.unsqueeze(1)  # [B, S2]
    ask_prefix = torch.zeros(B, dtype=torch.int32, device=device)
    ask_prefix[1:] = ask[:-1]
    t_row_idx = ask_prefix.unsqueeze(1) + s_arange  # [B, S2]
    t_row_idx = torch.where(mask, t_row_idx, torch.zeros_like(t_row_idx))
    key_bsnd = key[t_row_idx]
    key_bsnd = key_bsnd * mask.unsqueeze(-1).unsqueeze(-1).to(key.dtype)
    return key_bsnd, k_lens, S2, 1, 128


def _bsnd_to_tnd_output(
    out_bsnd: torch.Tensor,
    actual_q_len: torch.Tensor,
    q_tot: int,
) -> torch.Tensor:
    """Convert BSND output [B, S, N2, K] to TND [q_tot, N2, K] (drop padding rows).

    Args:
        out_bsnd: BSND padded output [B, S, N2, K].
        actual_q_len: per-batch lengths [B], int32.
        q_tot: sum(actual_q_len), target T dim.

    Returns:
        out_tnd: TND compact output [q_tot, N2, K].
    """
    B, S, N2, K = out_bsnd.shape
    device = out_bsnd.device
    actual_q_len = actual_q_len.to(device=device, dtype=torch.int32)
    # Build flat indices: for each (b, s) where s < actual_q_len[b],
    # the TND row is asq_prefix[b] + s
    s_arange = torch.arange(S, device=device).unsqueeze(0)  # [1, S]
    mask = s_arange < actual_q_len.unsqueeze(1)  # [B, S]
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, S)  # [B, S]
    s_idx = s_arange.expand(B, S)  # [B, S]
    flat_b = b_idx[mask]  # [q_tot]
    flat_s = s_idx[mask]  # [q_tot]
    # Gather: out_tnd = out_bsnd[flat_b, flat_s]
    out_tnd = out_bsnd[flat_b, flat_s]  # [q_tot, N2, K]
    return out_tnd


# ============================================================
# CPU Golden Reference
# ============================================================
_NEG_INF = float("-inf")


def cpu_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    *,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 0,
    block_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_tnd = layout_query == "TND"
    is_pa = layout_key == "PA_BSND"
    is_tnd_key = layout_key == "TND"

    q = query.detach().cpu()
    k = key.detach().cpu()
    w = weights.detach().cpu()
    bt = block_table.detach().cpu() if block_table is not None else None
    asq = [int(x) for x in actual_seq_lengths_query.detach().cpu().tolist()]
    ask = [int(x) for x in actual_seq_lengths_key.detach().cpu().tolist()]

    if is_tnd:
        q_tot, N1, D = q.shape
        B = len(asq)
        S1 = max(asq) if asq else 0
    else:
        B, S1, N1, D = q.shape
        q_tot = B * S1
    assert D == 128, f"head dim must be 128, got {D}"
    if is_pa:
        N2 = k.shape[2]
        bs = int(block_size) if block_size else k.shape[1]
    elif is_tnd_key:
        N2 = k.shape[1]
        bs = 128
    else:
        N2 = k.shape[2]
        bs = 128
    G = N1 // N2
    K = sparse_count
    dtype = q.dtype
    # pin to CPU explicitly: caller may have set torch.set_default_device("npu")
    _dev = q.device

    if is_tnd:
        indices = torch.full((q_tot, N2, K), -1, dtype=torch.int32, device=_dev)
        values = torch.full((q_tot, N2, K), _NEG_INF, dtype=dtype, device=_dev)
    else:
        indices = torch.full((B, S1, N2, K), -1, dtype=torch.int32, device=_dev)
        values = torch.full((B, S1, N2, K), _NEG_INF, dtype=dtype, device=_dev)

    max_ak = max(ask) if ask else 0
    full_scores = torch.full((B, N2, S1, max_ak), _NEG_INF, dtype=torch.float32, device=_dev)

    if (not asq) or (not ask) or max(asq) == 0 or max(ask) == 0:
        return indices, full_scores, values

    # TND asq/ask are prefix-sum (official spec); BSND/PA are per-batch.
    if is_tnd:
        q_off = [0] + list(asq)
    else:
        q_off = [0]
        for x in asq:
            q_off.append(q_off[-1] + x)
    if is_tnd_key:
        k_off = [0] + list(ask)
    else:
        k_off = [0]
        for x in ask:
            k_off.append(k_off[-1] + x)

    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    wf = w.to(torch.float32)
    s1_arange = torch.arange(S1, device=_dev)

    for b in range(B):
        aq, ak = q_off[b + 1] - q_off[b], k_off[b + 1] - k_off[b]
        if aq == 0 or ak == 0:
            continue
        qo, ko = q_off[b], k_off[b]
        for n2 in range(N2):
            # gather Q [aq, G, D] and W [aq, G]
            if is_tnd:
                qb = qf[qo : qo + aq, n2 * G : (n2 + 1) * G, :]
                wb = wf[qo : qo + aq, n2 * G : (n2 + 1) * G]
            else:
                qb = qf[b, :aq, n2 * G : (n2 + 1) * G, :]
                wb = wf[b, :aq, n2 * G : (n2 + 1) * G]
            # gather K [ak, D]
            if is_pa:
                assert bt is not None, "block_table required for PA_BSND layout"
                s2_idx = torch.arange(ak, device=_dev)
                bids = bt[b, s2_idx // bs]
                kb = kf[bids, s2_idx % bs, n2, :]
            elif is_tnd_key:
                kb = kf[ko : ko + ak, n2, :]
            else:
                kb = kf[b, :ak, n2, :]

            # score[aq, ak] = sum_g relu(qb @ kb^T) * wb
            qk = torch.relu(torch.einsum("qgd,kd->qgk", qb, kb))
            score = (qk * wb.unsqueeze(2)).sum(dim=1)

            # mask invalid s2 (causal when sparse_mode == 3, else actual_k boundary)
            if sparse_mode == 3:
                cl = ak - aq + s1_arange[:aq] + 1
                s2_valid = torch.where(cl > 0, cl, torch.full_like(cl, ak))
            else:
                s2_valid = torch.full((aq,), ak, device=_dev)
            mask = torch.arange(ak, device=_dev)[None, :] < s2_valid[:, None]
            score = score.masked_fill(~mask, _NEG_INF)
            full_scores[b, n2, :aq, :ak] = score

            kk = min(K, ak)
            # Stable sort: argsort(-score, stable=True) gives descending order,
            # ties broken by ascending original index (matches AscendC stable sort).
            sorted_order = torch.argsort(-score, dim=1, stable=True)
            topi = sorted_order[:, :kk]
            topv = score.gather(1, topi)
            invalid = topv <= -1e30
            topi = topi.to(torch.int32)
            topi = torch.where(invalid, torch.full_like(topi, -1), topi)
            topv = topv.to(dtype)

            if is_tnd:
                indices[qo : qo + aq, n2, :kk] = topi
                values[qo : qo + aq, n2, :kk] = topv
            else:
                indices[b, :aq, n2, :kk] = topi
                values[b, :aq, n2, :kk] = topv

    return indices, full_scores, values


def make_lightning_indexer_kernel(
    B: int,
    S1: int,
    S2: int,
    N1: int,
    D: int = 128,
    N2: int = 1,
    TOP_K: int = 2048,
    sparse_mode: int = 0,
    input_dtype: str = "float16",
    block_n: Optional[int] = None,
    max_cores: Optional[int] = None,
    layout_key: str = "BSND",
    block_size: int = 128,
    max_block_num: int = 1,
    pp_slots: int = 2,
    return_value: bool = False,
):
    G = N1 // N2
    is_pa = layout_key == "PA_BSND"
    calc_dtype = "float"

    BLOCK_N = block_n if block_n is not None else 128
    BLOCK_K = D
    _q_bufs = 2  # Q: 2 L1 buffers (ping-pong)
    _k_bufs = 3  # K: 3 L1 buffers (3-slot pipeline)

    S1_BLOCK = 4 if TOP_K <= 2048 else 2

    # VID_S1: each AIV processes half the S1_BLOCK rows.
    VID_S1 = (S1_BLOCK + 1) // 2 if S1_BLOCK >= 2 else 1
    USE_SORT_CACHE = (S1 <= 4) and (TOP_K <= 2048)

    M_L1 = S1_BLOCK * G
    BLOCK_M_L0 = 128
    _M_L1_padded = ((M_L1 + BLOCK_M_L0 - 1) // BLOCK_M_L0) * BLOCK_M_L0
    _num_full_iters = _M_L1_padded // BLOCK_M_L0
    # _tail_m/_has_tail removed: _M_L1_padded is always BLOCK_M_L0-aligned,
    # so there is never a tail block.

    if block_n is None:
        # Q L1 = 2 bufs × _M_L1_padded × BLOCK_K × sizeof(fp16)
        _q_l1_kb = _q_bufs * _M_L1_padded * BLOCK_K * 2 / 1024
        _pa_small_bs = is_pa and block_size < 64
        _candidates = [256, 128] if _pa_small_bs else ([512, 256, 128] if is_pa else [256, 128])
        for _test_n in _candidates:
            _n_split_test = max(1, (_test_n + 127) // 128)
            _l0b_n_test = _test_n // _n_split_test
            _l0c_kb = 2 * BLOCK_M_L0 * _l0b_n_test * 4 / 1024
            _k_l1_kb = 3 * _test_n * 128 * 2 / 1024
            if _l0c_kb <= 128 and (_q_l1_kb + _k_l1_kb) <= 500:
                BLOCK_N = _test_n
                break
    # PA gather: physical blocks (block_size each) per BLOCK_N K tile. 1 if block_size>=128.
    _BLOCKS_PER_TILE = (BLOCK_N // block_size) if is_pa else 1

    # L0B N-split: L0B 2-buffer must fit 64KB → N_SPLIT = ceil(BLOCK_N/128).
    _N_SPLIT = max(1, (BLOCK_N + 127) // 128)
    _L0B_N = BLOCK_N // _N_SPLIT  # per-split N in L0B/L0C

    _s2_vec_cap = 256 if _pa_small_bs else 512
    S2_VEC_BLOCK = min(_s2_vec_cap, ((S2 + _s2_vec_cap - 1) // _s2_vec_cap) * _s2_vec_cap) if S2 >= 256 else S2
    _S2_SUB_COUNT = S2_VEC_BLOCK // BLOCK_N  # Cube sub-iterations per s2_block

    S2_padded = ((S2 + S2_VEC_BLOCK - 1) // S2_VEC_BLOCK) * S2_VEC_BLOCK
    num_s2_blocks = S2_padded // S2_VEC_BLOCK

    if G % 8 == 0:
        VECTOR_BASEG = 8
    elif G % 4 == 0:
        VECTOR_BASEG = 4
    elif G % 2 == 0:
        VECTOR_BASEG = 2
    else:
        VECTOR_BASEG = G

    TOP_K_ALIGNED = ((TOP_K + 63) // 64) * 64
    _K_PER_BLOCK = min(TOP_K, S2_VEC_BLOCK)

    # PA with block_size < 64: pad Vector-scope buffers for 256-byte alignment
    # (T.tile.compare requires BLOCK_N*sizeof(calc_dtype) ≥ 256, i.e. BLOCK_N ≥ 64).
    _BLOCK_N_VEC = max(S2_VEC_BLOCK, 64) if is_pa else S2_VEC_BLOCK
    _NEED_VEC_PAD = _BLOCK_N_VEC > S2_VEC_BLOCK

    s1_blocks = (S1 + S1_BLOCK - 1) // S1_BLOCK
    num_bsns = B * s1_blocks * N2
    block_num = B * s1_blocks * N2 * num_s2_blocks
    _tasks_per_bN2 = s1_blocks * num_s2_blocks

    if max_cores is None or max_cores <= 0:
        max_cores = _get_cube_core_num()

    core_num = min(block_num, max_cores)
    tasks_per_core = (block_num + core_num - 1) // core_num

    if tasks_per_core == num_s2_blocks and core_num > 1:
        for _spg in [s for s in (2, 4, 8, 16, 32) if num_s2_blocks % s == 0]:
            _cn = core_num
            while _cn > 1:
                _cn -= 1
                _tpc = (block_num + _cn - 1) // _cn
                if _tpc > num_s2_blocks and _tpc % _spg == 0:
                    core_num = _cn
                    tasks_per_core = _tpc
                    break
            if tasks_per_core > num_s2_blocks:
                break

    # Per-core local-BSN workspace (recomputed after core_num reduction).
    # Per-core local-BSN workspace: 2 positional slots per core.
    # Slot 0: BSN that ends NATURALLY on this core (started on a previous core).
    # Slot 1: BSN that CONTINUES to the next core (core boundary mid-BSN).
    # A core can save at most 2 cross-core BSNs (one incoming, one outgoing),
    # so positional slots avoid hash collisions that _bsn_v % N could cause
    # when runtime actual_k_len varies across batches.
    _max_bsns_per_core = 2

    BSN_TASK_SPAN = num_s2_blocks  # tasks per BSN (correct granularity)

    # NEED_CROSS_CORE must check whether any BSN spans multiple cores,
    # not just whether num_s2_blocks > tasks_per_core. A BSN can span cores
    # even when BSN_TASK_SPAN <= tasks_per_core, if BSN boundaries don't
    # align with core task boundaries.
    _max_cores_per_bsn = 1
    for _b in range(B * s1_blocks * N2):
        _start = _b * BSN_TASK_SPAN
        _end = (_b + 1) * BSN_TASK_SPAN - 1
        _first_c = _start // tasks_per_core
        _last_c = _end // tasks_per_core
        _n_cores = _last_c - _first_c + 1
        if _n_cores > _max_cores_per_bsn:
            _max_cores_per_bsn = _n_cores
    NEED_CROSS_CORE = _max_cores_per_bsn > 1

    qk_ws_shape = (core_num, pp_slots, _M_L1_padded, S2_VEC_BLOCK)  # padded to 128 multiple
    # Per-core local-BSN workspace.
    topk_ws_shape = (core_num, _max_bsns_per_core, S1_BLOCK, 2 * TOP_K_ALIGNED)

    num_g_groups = G // VECTOR_BASEG
    SYNC_C1V1 = 0  # Cube→Vector counting semaphore
    SYNC_V1C1 = 1  # Vector→Cube counting semaphore

    # L1 address planning (codegen requires T.annotate_address for L1/L0C).
    _q_l1_bytes = _q_bufs * _M_L1_padded * BLOCK_K * 2
    _q_l1_addr = 0
    _acc_l0c_addr = 0

    # =====================================================================
    # Buffer allocation helpers
    # =====================================================================
    _TA2 = 2 * TOP_K_ALIGNED
    _KP2 = 2 * _K_PER_BLOCK

    _UB_LIMIT = 196352
    _w_raw_slot = 16  # 16 half = 32B aligned
    _ub_w_raw = _UB_LIMIT - 2 * _w_raw_slot * 2  # 196288
    _ub_mm_res = _ub_w_raw - 2 * VECTOR_BASEG * S2_VEC_BLOCK * 4  # 163520

    def _alloc_topk_bufs():
        topk_a = T.alloc_ub((VID_S1, _TA2), calc_dtype)  # per-row global topk accumulator
        # 2D buffer so cache_tmp_ub[0, :] creates BufferRegion (not BufferLoad).
        cache_tmp = T.alloc_ub((1, _KP2), calc_dtype)
        return topk_a, cache_tmp

    def _alloc_sort_cache():
        if USE_SORT_CACHE:
            return T.alloc_ub((VID_S1 * 4, _KP2), calc_dtype)
        return T.alloc_ub((1, 1), calc_dtype)

    def _alloc_g_reduce_bufs():
        w_raw = T.alloc_ub((2, _w_raw_slot), input_dtype)
        mm_res = T.alloc_ub((2, VECTOR_BASEG, S2_VEC_BLOCK), calc_dtype)
        weight = T.alloc_ub(VECTOR_BASEG, calc_dtype)
        weight_2d = T.alloc_ub((VECTOR_BASEG, S2_VEC_BLOCK), calc_dtype)
        reduce_tmp = T.alloc_ub((VECTOR_BASEG, S2_VEC_BLOCK), calc_dtype)
        return w_raw, weight, weight_2d, mm_res, reduce_tmp

    _key_shape = (max_block_num, block_size, N2, D) if is_pa else (B, S2, N2, D)
    _bt_shape = (B, max_block_num) if is_pa else (1, 1)

    _q_shape_q = (B, S1, N1, D)
    _q_shape_w = (B, S1, N1)
    _q_shape_o = (B, S1, N2, TOP_K)

    # Q L1 shape: padded to _M_L1_padded so copy_l1_to_l0a always reads
    # BLOCK_M_L0 rows without OOB (AC does the same with M_BASIC_BLOCK=256).
    _q_l1_shape = (_q_bufs, 1, _M_L1_padded, BLOCK_K)

    _g_id_list = list(range(num_g_groups))

    @jit(
        out_idx=[6, 7],
        workspace_idx=[3, 8],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
        },
    )
    def kernel_func():

        # Cube: L0A/L0B 2-buffer pingpong (m<->mte1; mma->fixpipe L0C handled by unit_flag=0b11)
        EVT_L0AB = 30  # m<->mte1: L0A/L0B[side] free/ready (30 + side)
        # Cube: K L1 3-buffer (KEY_BUF_NUM=3, mte1<->mte2)
        EVT_K_L1 = 20  # mte1->mte2: K L1[k_slot] freed (20 + k_slot)
        EVT_K_L1_READY = 10  # mte2->mte1: K L1[k_slot] ready (10 + k_slot)
        # Cube: Q L1 2-buffer (QUERY_BUF_NUM=2, mte1<->mte2)
        EVT_Q_L1 = 40  # mte1->mte2: Q L1[q_slot] freed (40 + q_slot)
        EVT_Q_L1_READY = 0  # mte2->mte1: Q L1[q_slot] ready (bare q_slot)
        # Vector: g-reduce 2-buffer pingpong (MTE2<->V, prefetch g_id+1)
        EVT_GREDUCE_PP = 13  # 13/14 for the 2 buffers
        # Vector: Phase2 cross-core merge
        EVT_P2_MERGE_RAW = 6  # V->MTE2: cross-iter RAW (merge_sort reads topk_a)
        EVT_P2_WS = 3  # MTE2->V: workspace read (3/4)
        # Vector: output copy (V<->MTE3)
        EVT_OUT_IDX = 8  # V<->MTE3: output index copy
        EVT_OUT_VAL = 5  # V->MTE3: output value copy
        EVT_OUT_VAL_DONE = 15  # MTE3->V: output value done
        EVT_OUT_ROW = 2  # V->MTE3: output row copy (Phase1 Opt2 + Phase2)

        @T.prim_func
        def main(
            Query: T.Tensor(_q_shape_q, input_dtype),
            Key: T.Tensor(_key_shape, input_dtype),
            Weights: T.Tensor(_q_shape_w, input_dtype),
            QK_Workspace: T.Tensor(qk_ws_shape, calc_dtype),
            actual_q_len: T.Tensor((B,), "int32"),
            actual_k_len: T.Tensor((B,), "int32"),
            Out: T.Tensor(_q_shape_o, "int32"),
            OutVal: T.Tensor(_q_shape_o, input_dtype),
            TopK_Workspace: T.Tensor(topk_ws_shape, calc_dtype),
            BlockTable: T.Tensor(_bt_shape, "int32"),
            # P2-3: BSN real task range (precomputed by host, aligns AscendC vec1ParamGm)
            BsnRf: T.Tensor((num_bsns,), "int32"),
            BsnRl: T.Tensor((num_bsns,), "int32"),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # Cube buffers: Q L1 (2 bufs), K L1 (3 bufs), L0A/L0B/L0C (2 bufs each).
                q_l1 = T.alloc_L1(_q_l1_shape, input_dtype)
                k_l1 = T.alloc_L1((_k_bufs, BLOCK_N, BLOCK_K), input_dtype)
                a_l0 = T.alloc_L0A((2, BLOCK_M_L0, BLOCK_K), input_dtype)
                b_l0 = T.alloc_L0B((2, BLOCK_K, _L0B_N), input_dtype)
                acc_l0c = T.alloc_L0C((2, BLOCK_M_L0, _L0B_N), calc_dtype)

                T.annotate_layout(
                    {
                        q_l1: make_zn_layout(q_l1),
                        k_l1: make_nz_layout(k_l1),
                    }
                )

                reduce_g_ub = T.alloc_ub(_BLOCK_N_VEC, calc_dtype)
                (w_raw_ub, weight_ub, weight_2d_ub, mm_res_ub, reduce_tmp_ub) = _alloc_g_reduce_bufs()

                # ===== Vector: Per-S1-row buffers =====
                (topk_a_ub, cache_tmp_ub) = _alloc_topk_bufs()

                merged_ub = T.alloc_ub(2 * _TA2, calc_dtype)
                cache_buf = _alloc_sort_cache()
                p2_acc_ub = T.alloc_ub(_TA2, calc_dtype)
                stride2_blk_ub = T.alloc_ub(_KP2, calc_dtype)
                reduce_sum_tmp_ub = T.alloc_ub(S2_VEC_BLOCK if _NEED_VEC_PAD else 1, calc_dtype)

                index_blk_ub = T.alloc_ub(_BLOCK_N_VEC, calc_dtype)
                mask_blk_ub = T.alloc_ub(_BLOCK_N_VEC // 8, "uint8")

                topk_index_ub = T.alloc_ub(TOP_K_ALIGNED, calc_dtype)
                output_ub = T.alloc_ub(TOP_K_ALIGNED, "int32")
                output_val_ub = T.alloc_ub(TOP_K_ALIGNED, input_dtype)
                score_topk_ub = T.alloc_ub(TOP_K_ALIGNED, calc_dtype)
                mask_topk_ub = T.alloc_ub(TOP_K_ALIGNED // 8, "uint8")

                prev_bsn_ub = T.alloc_ub(1, "int32")

                b_idx = T.alloc_var("int32")
                n2_idx = T.alloc_var("int32")
                s1_blk_idx = T.alloc_var("int32")
                s2_blk = T.alloc_var("int32")
                q_slot = T.alloc_var("int32")
                _gloop = T.alloc_var("int32")
                _k_slot = T.alloc_var("int32")
                _pp = T.alloc_var("int32")
                _gpp = T.alloc_var("int32")
                _prev_bsn_cube = T.alloc_var("int32")

                if USE_SORT_CACHE:
                    T.annotate_address(
                        {
                            q_l1: _q_l1_addr,
                            acc_l0c: _acc_l0c_addr,
                            w_raw_ub: _ub_w_raw,
                            mm_res_ub: _ub_mm_res,
                            cache_buf: _ub_mm_res - VID_S1 * 4 * _KP2 * 4,
                        }
                    )
                else:
                    T.annotate_address(
                        {
                            q_l1: _q_l1_addr,
                            acc_l0c: _acc_l0c_addr,
                            w_raw_ub: _ub_w_raw,
                            mm_res_ub: _ub_mm_res,
                        }
                    )

                _total_real_v = T.alloc_var("int32")
                _iter_v = T.alloc_var("int32")
                _total_real_v = T.cast(0, "int32")
                _iter_v = T.cast(0, "int32")
                while _iter_v < T.cast(B * N2, "int32"):
                    _b_h = _iter_v // N2
                    _s1_r = (actual_q_len[_b_h] + S1_BLOCK - 1) // S1_BLOCK
                    _s2_r = (actual_k_len[_b_h] + S2_VEC_BLOCK - 1) // S2_VEC_BLOCK
                    _total_real_v = _total_real_v + _s1_r * _s2_r
                    _iter_v = _iter_v + T.cast(1, "int32")

                _tpc_real = (_total_real_v + core_num - 1) // core_num
                _real_start = cid * _tpc_real
                _real_end = T.if_then_else(_real_start + _tpc_real > _total_real_v, _total_real_v, _real_start + _tpc_real)
                _real_count = _real_end - _real_start

                # =================================================================
                # C scope: Cube GEMM
                # =================================================================
                with T.Scope("C"):
                    # Q.K GEMM buffer scheme:
                    # KEY_BUF_NUM=3, QUERY_BUF_NUM=2, L0_BUF_NUM=2. K-load at iter top
                    # (3-deep K pipeline), Q 2-buffer w/ BSN-end free. L0A/B/C 2-buffer pingpong.
                    # Init: L0A/B[0/1] free, K[0/1/2] free, Q[0/1] free. L0C[0/1] are
                    # managed by the hardware mma->fixpipe pipeline (mma unit_flag=0b11
                    # + copy L0C->GM unit_flag=0b11), so no software M_FIX/FIX_M pre-set.
                    T.set_flag("m", "mte1", EVT_L0AB)
                    T.set_flag("m", "mte1", EVT_L0AB + 1)
                    T.set_flag("mte1", "mte2", EVT_K_L1)
                    T.set_flag("mte1", "mte2", EVT_K_L1 + 1)
                    T.set_flag("mte1", "mte2", EVT_K_L1 + 2)
                    T.set_flag("mte1", "mte2", EVT_Q_L1)
                    T.set_flag("mte1", "mte2", EVT_Q_L1 + 1)

                    # SplitCore — real task range computed outside scope
                    _k_slot = 0
                    _pp = 0
                    _prev_bsn_cube = T.cast(-1, "int32")

                    _cur = T.alloc_var("int32")
                    _cur = T.cast(0, "int32")
                    for _bN2 in range(B * N2):
                        _b_h = _bN2 // N2
                        _aq = actual_q_len[_b_h]
                        _ak = actual_k_len[_b_h]
                        _s1_r = (_aq + S1_BLOCK - 1) // S1_BLOCK
                        _s2_r = (_ak + S2_VEC_BLOCK - 1) // S2_VEC_BLOCK
                        _s2_real_safe = T.max(_s2_r, T.cast(1, "int32"))
                        for _gS1 in T.serial(_s1_r):
                            _bsn_v = _b_h * (N2 * s1_blocks) + _gS1 * N2 + (_bN2 % N2)
                            for _s2 in T.serial(_s2_real_safe):
                                _gloop_real = _cur
                                if (_cur >= _real_start) and (_cur < _real_end):
                                    b_idx = _b_h
                                    n2_idx = _bN2 % N2
                                    s1_blk_idx = _gS1
                                    s2_blk = _s2
                                    q_slot = _bsn_v % 2
                                    k_slot = _k_slot
                                    pp = _pp
                                    s1_start = s1_blk_idx * S1_BLOCK
                                    _q_len = _aq
                                    s2_start = s2_blk * S2_VEC_BLOCK
                                    _q_need = (_gloop_real == _real_start) | (_bsn_v != _prev_bsn_cube)
                                    _prev_bsn_cube = _bsn_v

                                    if _q_need:
                                        T.wait_flag("mte1", "mte2", EVT_Q_L1 + q_slot)
                                        for s1_local in range(S1_BLOCK):
                                            s1_idx = T.if_then_else(s1_start + s1_local < _q_len, s1_start + s1_local, T.max(_q_len - 1, 0))
                                            _l1_off = s1_local * G
                                            T.copy(
                                                Query[b_idx, s1_idx, n2_idx * G : n2_idx * G + G, 0:D],
                                                q_l1[q_slot, 0, _l1_off : _l1_off + G, :],
                                            )
                                        T.set_flag("mte2", "mte1", EVT_Q_L1_READY + q_slot)

                                    # Cross-core sync + Q wait (once per gloop, before s2_sub loop)
                                    T.wait_cross_flag(SYNC_V1C1)
                                    if _q_need:
                                        T.wait_flag("mte2", "mte1", EVT_Q_L1_READY + q_slot)

                                    for s2_sub in range(_S2_SUB_COUNT):
                                        _s2_sub_offset = s2_sub * BLOCK_N
                                        _s2_sub_start = s2_start + _s2_sub_offset

                                        # K load (BLOCK_N columns per sub-block)
                                        T.wait_flag("mte1", "mte2", EVT_K_L1 + k_slot)
                                        if is_pa:
                                            for sub in range(_BLOCKS_PER_TILE):
                                                _block_table_idx = (
                                                    s2_blk * _BLOCKS_PER_TILE * _S2_SUB_COUNT + s2_sub * _BLOCKS_PER_TILE + sub
                                                )
                                                _safe_block_table_idx = T.min(_block_table_idx, T.cast(max_block_num - 1, "int32"))
                                                T.copy(
                                                    Key[BlockTable[b_idx, _safe_block_table_idx], 0:block_size, n2_idx, 0:D],
                                                    k_l1[k_slot, sub * block_size : (sub + 1) * block_size, :],
                                                )
                                        else:
                                            T.copy(Key[b_idx, _s2_sub_start : _s2_sub_start + BLOCK_N, n2_idx, 0:D], k_l1[k_slot, :, :])
                                        T.set_flag("mte2", "mte1", EVT_K_L1_READY + k_slot)
                                        T.wait_flag("mte2", "mte1", EVT_K_L1_READY + k_slot)

                                        for m_iter in range(_num_full_iters):
                                            for n_l0 in range(_N_SPLIT):
                                                side = (m_iter * _N_SPLIT + n_l0) % 2
                                                _nlo = n_l0 * _L0B_N
                                                _nhi = _nlo + _L0B_N
                                                _m_off = m_iter * BLOCK_M_L0
                                                T.wait_flag("m", "mte1", EVT_L0AB + side)
                                                T.copy(q_l1[q_slot, 0, _m_off : _m_off + BLOCK_M_L0, :], a_l0[side, :, :])
                                                T.copy(k_l1[k_slot, _nlo:_nhi, :], b_l0[side, :, :], transpose=True)
                                                T.set_flag("mte1", "m", EVT_L0AB + side)
                                                T.wait_flag("mte1", "m", EVT_L0AB + side)
                                                T.mma(a_l0[side, :, :], b_l0[side, :, :], acc_l0c[side, :, :], init=True, unit_flag=0b11)
                                                T.set_flag("m", "mte1", EVT_L0AB + side)
                                                T.copy(
                                                    acc_l0c[side, :, :],
                                                    QK_Workspace[
                                                        cid, pp, _m_off : _m_off + BLOCK_M_L0, _s2_sub_offset + _nlo : _s2_sub_offset + _nhi
                                                    ],
                                                    enable_relu=True,
                                                    unit_flag=0b11,
                                                )

                                        # Free K buffer for next s2_sub or next gloop
                                        T.set_flag("mte1", "mte2", EVT_K_L1 + k_slot)

                                    _bsn_end = (s2_blk == _s2_real_safe - 1) | (_gloop_real == _real_end - 1)
                                    if _bsn_end:
                                        T.set_flag("mte1", "mte2", EVT_Q_L1 + q_slot)

                                    # K buffer already freed inside s2_sub loop (last iteration).
                                    T.pipe_barrier("FIX")
                                    T.set_cross_flag("FIX", SYNC_C1V1)

                                    _k_slot = _k_slot + 1
                                    if _k_slot >= 3:
                                        _k_slot = 0
                                    _pp = _pp + 1
                                    if _pp >= pp_slots:
                                        _pp = 0
                                _cur = _cur + 1

                    T.wait_flag("m", "mte1", EVT_L0AB)
                    T.wait_flag("m", "mte1", EVT_L0AB + 1)
                    T.wait_flag("mte1", "mte2", EVT_K_L1)
                    T.wait_flag("mte1", "mte2", EVT_K_L1 + 1)
                    T.wait_flag("mte1", "mte2", EVT_K_L1 + 2)
                    T.wait_flag("mte1", "mte2", EVT_Q_L1)
                    T.wait_flag("mte1", "mte2", EVT_Q_L1 + 1)
                    T.sync_all()

                # =================================================================
                # V scope: Phase 1-V + Phase 2 (manual sync matching auto_sync)
                # =================================================================
                with T.Scope("V"):
                    # Counting semaphore init — pre-set SYNC_V1C1 pp_slots times.
                    T.set_cross_flag("MTE2", SYNC_V1C1)
                    T.set_cross_flag("MTE2", SYNC_V1C1)

                    T.set_flag("V", "MTE2", EVT_GREDUCE_PP)
                    T.set_flag("V", "MTE2", EVT_GREDUCE_PP + 1)

                    T.tile.fill(stride2_blk_ub, 0)
                    T.pipe_barrier("V")
                    for _i in range(_K_PER_BLOCK):
                        stride2_blk_ub[_i * 2 + 1] = T.cast(1, calc_dtype)

                    T.tile.arith_progression(index_blk_ub, T.cast(0, calc_dtype), T.cast(1, calc_dtype), _BLOCK_N_VEC)

                    prev_bsn_ub[0] = -1
                    T.tile.fill(topk_a_ub, -T.infinity(calc_dtype))

                    # SplitCore — real task range computed outside scope (V scope)
                    _pp = 0

                    _cur = T.alloc_var("int32")
                    _cur = T.cast(0, "int32")
                    for _bN2 in range(B * N2):
                        _b_h = _bN2 // N2
                        _aq = actual_q_len[_b_h]
                        _ak = actual_k_len[_b_h]
                        _s1_r = (_aq + S1_BLOCK - 1) // S1_BLOCK
                        _s2_r = (_ak + S2_VEC_BLOCK - 1) // S2_VEC_BLOCK
                        _s2_real_safe = T.max(_s2_r, T.cast(1, "int32"))
                        for _gS1 in T.serial(_s1_r):
                            _bsn_v = _b_h * (N2 * s1_blocks) + _gS1 * N2 + (_bN2 % N2)
                            for _s2 in T.serial(_s2_real_safe):
                                _gloop_real = _cur
                                if (_cur >= _real_start) and (_cur < _real_end):
                                    b_idx = _b_h
                                    n2_idx = _bN2 % N2
                                    s1_blk_idx = _gS1
                                    s2_blk = _s2
                                    _s2_idx = s2_blk
                                    _s2_real_b_v = _s2_real_safe
                                    pp = _pp
                                    s1_start = s1_blk_idx * S1_BLOCK
                                    _q_len_b = _aq
                                    _k_len_b = _ak
                                    s2_start = s2_blk * S2_VEC_BLOCK

                                    _s1blk_c = T.cast(S1_BLOCK, "int32")
                                    _valid_s1_v = T.min(_s1blk_c, T.max(T.cast(0, "int32"), _q_len_b - s1_start))
                                    _first_half_v = (_valid_s1_v + 1) // 2
                                    _second_half_v = _valid_s1_v // 2
                                    _s1_base_v = vid * _first_half_v
                                    _my_half = _first_half_v - vid * (_first_half_v - _second_half_v)

                                    if prev_bsn_ub[0] >= 0 and _bsn_v != prev_bsn_ub[0]:
                                        T.tile.fill(topk_a_ub, -T.infinity(calc_dtype))
                                    prev_bsn_ub[0] = _bsn_v

                                    T.wait_cross_flag(SYNC_C1V1)

                                    for s1_local in range(VID_S1):
                                        actual_s1 = _s1_base_v + T.cast(s1_local, "int32")

                                        s1_idx = s1_start + actual_s1
                                        s2_valid = _k_len_b
                                        causal_limit = _k_len_b - _q_len_b + s1_idx + 1
                                        _mode_val = T.cast(sparse_mode, "int32")
                                        _is_causal = _mode_val == T.cast(3, "int32")
                                        s2_valid = T.if_then_else(_is_causal & (causal_limit > 0), causal_limit, s2_valid)

                                        # ===== Phase A: g-reduce + reduce_sum + mask (MTE2 + V) =====
                                        if T.cast(s1_local, "int32") < _my_half and s2_start < s2_valid:
                                            T.tile.fill(reduce_tmp_ub, 0)
                                            # MTE2/V software pipeline: prefetch g_id+1's QK/Weights
                                            # (MTE2) while V computes g_id, hiding GM->UB latency behind V.
                                            # Prologue: MTE2 for g_id 0 into buffer 0.
                                            T.wait_flag("V", "MTE2", EVT_GREDUCE_PP)
                                            T.copy(
                                                Weights[b_idx, s1_idx, n2_idx * G : n2_idx * G + VECTOR_BASEG],
                                                w_raw_ub[0, :],
                                            )
                                            T.copy(
                                                QK_Workspace[cid, pp, actual_s1 * G : actual_s1 * G + VECTOR_BASEG, 0:S2_VEC_BLOCK],
                                                mm_res_ub[0, :, :],
                                            )
                                            T.set_flag("MTE2", "V", EVT_GREDUCE_PP)
                                            for g_id in T.serial(num_g_groups):
                                                _gpp = g_id % 2
                                                _gpp_next = (g_id + 1) % 2
                                                _flag_id = 13 + _gpp
                                                _flag_next = 13 + _gpp_next
                                                # Prefetch MTE2 for g_id+1 (overlaps V of g_id below)
                                                if g_id + 1 < num_g_groups:
                                                    _wg_next = (g_id + 1) * VECTOR_BASEG
                                                    _qk_next = actual_s1 * G + _wg_next
                                                    T.wait_flag("V", "MTE2", _flag_next)
                                                    T.copy(
                                                        Weights[
                                                            b_idx, s1_idx, n2_idx * G + _wg_next : n2_idx * G + _wg_next + VECTOR_BASEG
                                                        ],
                                                        w_raw_ub[_gpp_next, :],
                                                    )
                                                    T.copy(
                                                        QK_Workspace[cid, pp, _qk_next : _qk_next + VECTOR_BASEG, 0:S2_VEC_BLOCK],
                                                        mm_res_ub[_gpp_next, :, :],
                                                    )
                                                    T.set_flag("MTE2", "V", _flag_next)
                                                # V compute for g_id (consume buffer _gpp)
                                                T.wait_flag("MTE2", "V", _flag_id)
                                                T.tile.cast(weight_ub, w_raw_ub[_gpp, :], "CAST_NONE", VECTOR_BASEG)
                                                T.tile.broadcast(weight_2d_ub, weight_ub)

                                                if num_g_groups <= 3:
                                                    if g_id == 0:
                                                        T.tile.mul(reduce_tmp_ub, mm_res_ub[_gpp, :, :], weight_2d_ub)
                                                    else:
                                                        T.tile.mul_add_dst(reduce_tmp_ub, mm_res_ub[_gpp, :, :], weight_2d_ub)
                                                else:
                                                    T.tile.mul_add_dst(reduce_tmp_ub, mm_res_ub[_gpp, :, :], weight_2d_ub)
                                                T.set_flag("V", "MTE2", _flag_id)

                                            # Tree-reduce VECTOR_BASEG rows to 2 rows.
                                            if VECTOR_BASEG == 8:
                                                T.tile.add(reduce_tmp_ub[0:4, :], reduce_tmp_ub[0:4, :], reduce_tmp_ub[4:8, :])
                                                T.tile.add(reduce_tmp_ub[0:2, :], reduce_tmp_ub[0:2, :], reduce_tmp_ub[2:4, :])
                                            elif VECTOR_BASEG == 4:
                                                T.tile.add(reduce_tmp_ub[0:2, :], reduce_tmp_ub[0:2, :], reduce_tmp_ub[2:4, :])

                                            if _NEED_VEC_PAD:
                                                T.tile.add(reduce_sum_tmp_ub, reduce_tmp_ub[0, :], reduce_tmp_ub[1, :])
                                                T.pipe_barrier("V")
                                                T.copy(reduce_sum_tmp_ub, reduce_g_ub[0:BLOCK_N])
                                                T.pipe_barrier("V")
                                                for _ti in range(BLOCK_N, _BLOCK_N_VEC):
                                                    reduce_g_ub[_ti] = -T.infinity(calc_dtype)
                                            else:
                                                T.tile.add(reduce_g_ub, reduce_tmp_ub[0, :], reduce_tmp_ub[1, :])

                                            limit = s2_valid - s2_start
                                            T.tile.compare(mask_blk_ub, index_blk_ub, limit, "LT")
                                            T.tile.select(
                                                reduce_g_ub, mask_blk_ub, reduce_g_ub, -T.infinity(calc_dtype), "VSEL_TENSOR_SCALAR_MODE"
                                            )

                                        if s1_local == VID_S1 - 1:
                                            T.set_cross_flag("MTE2", SYNC_V1C1)

                                        # ===== Phase B: sort + merge (V only, no MTE2) =====
                                        if T.cast(s1_local, "int32") < _my_half and s2_start < s2_valid:
                                            if USE_SORT_CACHE:
                                                _bsn_first_real = _gloop_real - s2_blk
                                                _core_bsn_start = T.max(_real_start, _bsn_first_real)
                                                _seg_off = _gloop_real - _core_bsn_start
                                                _cache_slot = _seg_off % 4
                                                T.tile.sort(cache_tmp_ub, reduce_g_ub, _BLOCK_N_VEC)
                                                T.tile.axpy(cache_tmp_ub, stride2_blk_ub, T.cast(s2_start, calc_dtype))
                                                T.pipe_barrier("V")
                                                T.copy(cache_tmp_ub[0, :], cache_buf[s1_local * 4 + _cache_slot, :])
                                                _is_trigger = (
                                                    (_cache_slot == 3) | (_s2_idx == _s2_real_safe - 1) | (_gloop_real == _real_end - 1)
                                                )
                                                if _is_trigger:
                                                    T.pipe_barrier("V")
                                                    _num_cached = _cache_slot + 1
                                                    _is_first_grp = _seg_off < 4
                                                    if _is_first_grp:
                                                        # MrgBasicBlock: merge cache -> merged_ub -> topk_a (init).
                                                        T.tile.fill(merged_ub, -T.infinity(calc_dtype))
                                                        T.pipe_barrier("V")
                                                        if _num_cached == 4:
                                                            T.tile.merge_sort(
                                                                merged_ub,
                                                                cache_buf[s1_local * 4 + 0, :],
                                                                cache_buf[s1_local * 4 + 1, :],
                                                                cache_buf[s1_local * 4 + 2, :],
                                                                cache_buf[s1_local * 4 + 3, :],
                                                            )
                                                        elif _num_cached == 3:
                                                            T.tile.merge_sort(
                                                                merged_ub,
                                                                cache_buf[s1_local * 4 + 0, :],
                                                                cache_buf[s1_local * 4 + 1, :],
                                                                cache_buf[s1_local * 4 + 2, :],
                                                            )
                                                        elif _num_cached == 2:
                                                            T.tile.merge_sort(
                                                                merged_ub, cache_buf[s1_local * 4 + 0, :], cache_buf[s1_local * 4 + 1, :]
                                                            )
                                                        if _num_cached >= 2:
                                                            T.pipe_barrier("V")
                                                            T.copy(merged_ub[0:_TA2], topk_a_ub[s1_local, :])
                                                        else:
                                                            T.pipe_barrier("V")
                                                            T.copy(cache_buf[s1_local * 4 + 0, :], topk_a_ub[s1_local, 0:_KP2])
                                                    else:
                                                        # 精排 cache -> merged_ub -> p2_acc, then SparseTopK
                                                        T.tile.fill(merged_ub, -T.infinity(calc_dtype))
                                                        T.pipe_barrier("V")
                                                        if _num_cached == 4:
                                                            T.tile.merge_sort(
                                                                merged_ub,
                                                                cache_buf[s1_local * 4 + 0, :],
                                                                cache_buf[s1_local * 4 + 1, :],
                                                                cache_buf[s1_local * 4 + 2, :],
                                                                cache_buf[s1_local * 4 + 3, :],
                                                            )
                                                        elif _num_cached == 3:
                                                            T.tile.merge_sort(
                                                                merged_ub,
                                                                cache_buf[s1_local * 4 + 0, :],
                                                                cache_buf[s1_local * 4 + 1, :],
                                                                cache_buf[s1_local * 4 + 2, :],
                                                            )
                                                        elif _num_cached == 2:
                                                            T.tile.merge_sort(
                                                                merged_ub, cache_buf[s1_local * 4 + 0, :], cache_buf[s1_local * 4 + 1, :]
                                                            )
                                                        if _num_cached >= 2:
                                                            T.pipe_barrier("V")
                                                            T.copy(merged_ub[0:_TA2], p2_acc_ub)
                                                        else:
                                                            T.tile.fill(p2_acc_ub, -T.infinity(calc_dtype))
                                                            T.pipe_barrier("V")
                                                            T.copy(cache_buf[s1_local * 4 + 0, :], p2_acc_ub[0:_KP2])
                                                        # SparseTopK: merge topk_a + p2_acc -> merged_ub -> topk_a
                                                        T.pipe_barrier("V")
                                                        T.tile.merge_sort(merged_ub, topk_a_ub[s1_local, :], p2_acc_ub)
                                                        T.pipe_barrier("V")
                                                        T.copy(merged_ub[0:_TA2], topk_a_ub[s1_local, :])
                                            else:
                                                # no cache (S1>4, AscendC condition actS1Size>4): sort + 2-way merge each block
                                                T.tile.fill(cache_tmp_ub, -T.infinity(calc_dtype))
                                                T.tile.sort(cache_tmp_ub, reduce_g_ub, _BLOCK_N_VEC)
                                                T.tile.axpy(cache_tmp_ub, stride2_blk_ub, T.cast(s2_start, calc_dtype))
                                                T.tile.merge_sort(merged_ub, topk_a_ub[s1_local, :], cache_tmp_ub[0, :])
                                                T.copy(merged_ub[0:_TA2], topk_a_ub[s1_local, :])

                                    _is_natural_bsn_end = _s2_idx == _s2_real_b_v - 1
                                    _is_core_boundary = _gloop_real == _real_end - 1
                                    _is_bsn_end = _is_natural_bsn_end | _is_core_boundary
                                    if _is_bsn_end:
                                        _bsn_first_real = _gloop_real - _s2_idx
                                        _tpc_safe = T.max(_tpc_real, T.cast(1, "int32"))
                                        _bsn_first_core = _bsn_first_real // _tpc_safe
                                        _bsn_is_split = T.if_then_else(_is_natural_bsn_end, _bsn_first_core != cid, _is_core_boundary)
                                        if _bsn_is_split:
                                            _is_first_core = cid == _bsn_first_core
                                            _save_bsn = T.if_then_else(_is_core_boundary & _is_first_core, 1, 0)
                                            T.set_flag("V", "MTE3", EVT_OUT_ROW)
                                            T.wait_flag("V", "MTE3", EVT_OUT_ROW)
                                            for _si in range(VID_S1):
                                                if T.cast(_si, "int32") < _my_half:
                                                    T.copy(
                                                        topk_a_ub[_si, :],
                                                        TopK_Workspace[cid, _save_bsn, _s1_base_v + T.cast(_si, "int32"), 0:_TA2],
                                                    )
                                            T.pipe_barrier("MTE3")
                                        else:
                                            # Opt2: Non-cross-core BSN — direct Extract+Cast+CopyOut
                                            for _si in range(VID_S1):
                                                actual_s1_d = _s1_base_v + T.cast(_si, "int32")
                                                s1_idx_d = s1_start + actual_s1_d
                                                if T.cast(_si, "int32") < _my_half:
                                                    T.copy(topk_a_ub[_si, :], p2_acc_ub)
                                                    T.tile.gather_mask(topk_index_ub, p2_acc_ub, "P1010")
                                                    T.tile.fill(score_topk_ub, 0)
                                                    T.tile.gather_mask(score_topk_ub, p2_acc_ub, "P0101")
                                                    T.tile.compare(mask_topk_ub, score_topk_ub, T.cast(-1e30, calc_dtype), "GT")
                                                    if return_value:
                                                        T.tile.cast(output_val_ub, score_topk_ub, "CAST_RINT", TOP_K)
                                                    T.tile.select(
                                                        topk_index_ub,
                                                        mask_topk_ub,
                                                        topk_index_ub,
                                                        T.cast(-1.0, calc_dtype),
                                                        "VSEL_TENSOR_SCALAR_MODE",
                                                    )
                                                    T.tile.cast(output_ub, topk_index_ub, "CAST_ROUND", TOP_K)
                                                    T.set_flag("V", "MTE3", EVT_OUT_IDX)
                                                    T.wait_flag("V", "MTE3", EVT_OUT_IDX)
                                                    # li_v2: BSND-only Out write (TND handled by host conversion)
                                                    T.copy(output_ub, Out[b_idx, s1_idx_d, n2_idx, 0:TOP_K])
                                                    T.set_flag("MTE3", "V", EVT_OUT_IDX)
                                                    T.wait_flag("MTE3", "V", EVT_OUT_IDX)
                                                    if return_value:
                                                        T.set_flag("V", "MTE3", EVT_OUT_VAL)
                                                        T.wait_flag("V", "MTE3", EVT_OUT_VAL)
                                                        # li_v2: BSND-only OutVal write
                                                        T.copy(output_val_ub, OutVal[b_idx, s1_idx_d, n2_idx, 0:TOP_K])
                                                        T.set_flag("MTE3", "V", EVT_OUT_VAL_DONE)
                                                        T.wait_flag("MTE3", "V", EVT_OUT_VAL_DONE)

                                    _pp = _pp + 1
                                    if _pp >= pp_slots:
                                        _pp = 0
                                _cur = _cur + 1

                    T.wait_flag("V", "MTE2", EVT_GREDUCE_PP)
                    T.wait_flag("V", "MTE2", EVT_GREDUCE_PP + 1)

                    T.barrier_all()
                    T.sync_all()

                _num_output_rows_p15 = B * S1 * N2
                _rows_per_core_p15 = (_num_output_rows_p15 + core_num - 1) // core_num
                for _row_off_p15 in T.serial(_rows_per_core_p15):
                    _out_row_p15 = cid * _rows_per_core_p15 + _row_off_p15
                    if _out_row_p15 < _num_output_rows_p15:
                        _b_p15 = _out_row_p15 // (S1 * N2)
                        _s1_p15 = (_out_row_p15 // N2) % S1
                        _n2_p15 = _out_row_p15 % N2
                        _qlen_p15 = actual_q_len[_b_p15]
                        if _s1_p15 >= _qlen_p15:
                            T.tile.fill(output_ub, -1)
                            T.pipe_barrier("V")
                            T.copy(output_ub, Out[_b_p15, _s1_p15, _n2_p15, 0:TOP_K])

                # =========================================================
                # Phase 2: Dispersed cross-core merge + output
                # Each core handles ceil(num_bsns*S1_BLOCK/core_num) rows.
                # workspace dim is S1_BLOCK (not S1), iterate per-core rows.
                # =========================================================
                num_output_rows = num_bsns * S1_BLOCK
                rows_per_core_p2 = (num_output_rows + core_num - 1) // core_num
                for row_off in T.serial(rows_per_core_p2):
                    out_row = cid * rows_per_core_p2 + row_off
                    if out_row < num_output_rows:
                        out_bsn = out_row // S1_BLOCK
                        out_s1_local = out_row % S1_BLOCK
                        b_idx_p2 = out_bsn // (s1_blocks * N2)
                        s1_blk_idx_p2 = (out_bsn // N2) % s1_blocks
                        n2_idx_p2 = out_bsn % N2
                        s1_start_p2 = s1_blk_idx_p2 * S1_BLOCK
                        s1_idx_p2 = s1_start_p2 + out_s1_local
                        _q_len_p2 = actual_q_len[b_idx_p2]
                        # li_v2: removed _q_off_p2 = QOffset[b_idx_p2] (BSND-only kernel uses [b, s, ...] indexing)
                        if s1_idx_p2 < _q_len_p2 and out_s1_local < S1_BLOCK:
                            # P2-3: Read precomputed BSN task range from GM (eliminates while-loop)
                            _bsn_rf_v = BsnRf[out_bsn]
                            _bsn_rl_v = BsnRl[out_bsn]
                            _tpc_safe_p2 = T.max(_tpc_real, T.cast(1, "int32"))
                            _rc_first = _bsn_rf_v // _tpc_safe_p2
                            _rc_last = _bsn_rl_v // _tpc_safe_p2
                            if _rc_first != _rc_last:
                                # Cross-core BSN: workspace read + merge + extract + cast + copyout
                                # P4-2: Align AscendC head/tail — first core saved at slot 1 (tail),
                                # middle/last cores saved at slot 0 (head). Read slot 0 uniformly.
                                T.copy(TopK_Workspace[_rc_first, 1, out_s1_local, 0:_TA2], p2_acc_ub)
                                T.set_flag("MTE2", "V", EVT_P2_WS)
                                T.wait_flag("MTE2", "V", EVT_P2_WS)
                                if NEED_CROSS_CORE:
                                    num_merge_p2 = _rc_last - _rc_first
                                    for m in T.serial(num_merge_p2):
                                        other_cid = _rc_first + 1 + m
                                        # V->MTE2 flag 6 for the topk_a_ub cross-iter RAW (merge_sort
                                        # reads topk_a; next iter's copy overwrites it). pipe_barrier("V")
                                        # covers the in-pipe p2_acc/merged RAW. Workspace is read-only
                                        # (sync_all after Phase 1), so no cross-core sync in the merge loop.
                                        if m > 0:
                                            T.wait_flag("V", "MTE2", EVT_P2_MERGE_RAW)
                                        T.copy(TopK_Workspace[other_cid, 0, out_s1_local, 0:_TA2], topk_a_ub[0, :])
                                        T.set_flag("MTE2", "V", EVT_P2_WS + 1)
                                        T.wait_flag("MTE2", "V", EVT_P2_WS + 1)
                                        T.tile.merge_sort(merged_ub, p2_acc_ub, topk_a_ub[0, :])
                                        T.pipe_barrier("V")
                                        T.set_flag("V", "MTE2", EVT_P2_MERGE_RAW)
                                        T.copy(merged_ub[0:_TA2], p2_acc_ub)
                                        T.pipe_barrier("V")
                                    # Balance flag 6 (consume last set) so per-row leftovers don't race
                                    # with later cross-core rows' merge loops.
                                    T.wait_flag("V", "MTE2", EVT_P2_MERGE_RAW)
                                # Output: extract indices
                                # V-pipe in-order execution guarantees ordering here
                                # internal RAW (gather_mask→fill→gather_mask→compare→select→cast),
                                # V-pipe in-order execution already guarantees ordering.
                                T.tile.gather_mask(topk_index_ub, p2_acc_ub, "P1010")
                                T.tile.fill(score_topk_ub, 0)
                                T.tile.gather_mask(score_topk_ub, p2_acc_ub, "P0101")
                                T.tile.compare(mask_topk_ub, score_topk_ub, T.cast(-1e30, calc_dtype), "GT")
                                if return_value:
                                    T.tile.cast(output_val_ub, score_topk_ub, "CAST_RINT", TOP_K)
                                T.tile.select(
                                    topk_index_ub, mask_topk_ub, topk_index_ub, T.cast(-1.0, calc_dtype), "VSEL_TENSOR_SCALAR_MODE"
                                )
                                T.tile.cast(output_ub, topk_index_ub, "CAST_ROUND", TOP_K)

                                T.set_flag("V", "MTE3", EVT_OUT_ROW)
                                T.wait_flag("V", "MTE3", EVT_OUT_ROW)
                                T.copy(output_ub, Out[b_idx_p2, s1_idx_p2, n2_idx_p2, 0:TOP_K])
                                if return_value:
                                    T.set_flag("MTE3", "V", EVT_OUT_VAL_DONE)
                                    T.wait_flag("MTE3", "V", EVT_OUT_VAL_DONE)
                                    T.set_flag("V", "MTE3", EVT_OUT_VAL)
                                    T.wait_flag("V", "MTE3", EVT_OUT_VAL)
                                    T.copy(output_val_ub, OutVal[b_idx_p2, s1_idx_p2, n2_idx_p2, 0:TOP_K])

        return main

    return kernel_func()


# ============================================================
# Wrapper
# ============================================================
def lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    *,
    actual_seq_lengths_query: Optional[torch.Tensor] = None,
    actual_seq_lengths_key: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    pre_tokens: int = (1 << 63) - 1,
    next_tokens: int = (1 << 63) - 1,
    return_value: bool = False,
    block_n: Optional[int] = None,
    pp_slots: int = 2,
    max_cores: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """TileLang lightning_indexer (li_v2) -- BSND-only kernel.

    TND case is converted to BSND padded on host side.
    PA_BSND case is preserved (block_table indirect addressing).
    Compatible with torch_npu.npu_lightning_indexer interface.

    Key: actual_seq_lengths in TND mode are prefix-sum format (per official spec).
    """
    assert layout_query in ("BSND", "TND"), f"Unsupported query layout: {layout_query}"
    assert layout_key in ("BSND", "PA_BSND", "TND"), f"Unsupported key layout: {layout_key}"
    is_pa = layout_key == "PA_BSND"
    is_tnd = layout_query == "TND"
    is_tnd_key = layout_key == "TND"

    if is_pa and return_value:
        raise ValueError("PA_BSND layout does not support return_value")

    if not is_pa and layout_query != layout_key:
        raise ValueError(f"layout_query({layout_query}) must equal layout_key({layout_key}) when key is non-PA_BSND")
    if not (1 <= sparse_count <= 2048 or (sparse_count % 1024 == 0 and sparse_count <= 8192)):
        raise ValueError(f"sparse_count must be in [1,2048] or a multiple of 1024 up to 8192, got {sparse_count}")
    if sparse_mode not in (0, 3):
        raise ValueError(f"sparse_mode must be 0 (defaultMask) or 3 (rightDownCausal), got {sparse_mode}")
    _INT64_MAX = (1 << 63) - 1
    if pre_tokens != _INT64_MAX:
        raise ValueError(f"pre_tokens must be INT64_MAX, got {pre_tokens}")
    if next_tokens != _INT64_MAX:
        raise ValueError(f"next_tokens must be INT64_MAX, got {next_tokens}")
    if query.dtype != key.dtype:
        raise ValueError(f"query dtype {query.dtype} must equal key dtype {key.dtype}")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"query/key dtype must be float16 or bfloat16, got {query.dtype}")
    if weights.dtype != torch.float32 and weights.dtype != query.dtype:
        raise ValueError(f"weights dtype {weights.dtype} must be float32 or same as query {query.dtype}")

    # ===== Step 1: 统一 query 为 BSND padded =====
    if is_tnd:
        query, actual_q_len, S1, B, N1, D, q_tot = _tnd_to_bsnd_query(query, actual_seq_lengths_query)
        weights = _tnd_to_bsnd_weights(weights, actual_q_len, B, S1, N1)
    else:
        B, S1, N1, D = query.shape
        q_tot = B * S1
        actual_q_len = _to_per_batch_tensor_or_default(actual_seq_lengths_query, B, S1, query.device)

    # ===== Step 2: 统一 key 为 BSND padded 或保留 PA_BSND =====
    if is_tnd_key:
        key, actual_k_len, S2, max_block_num, _block_size = _tnd_to_bsnd_key(key, actual_seq_lengths_key, B, query.device)
        layout_key = "BSND"  # TND key -> BSND, kernel only sees "BSND"
        N2_k = key.shape[2]  # BSND key [B, S2, N2, D] -> N2 is dim 2
    elif is_pa:
        max_block_num, _block_size, N2_k, _ = key.shape
        actual_k_len = _to_per_batch_tensor(actual_seq_lengths_key, B, query.device)
        S2 = max(int(actual_k_len.max().item()), 257)
    else:  # BSND key
        _, S2, N2_k, _ = key.shape
        max_block_num, _block_size = 1, 128
        actual_k_len = _to_per_batch_tensor_or_default(actual_seq_lengths_key, B, S2, query.device)

    N2 = N2_k
    assert D == 128, f"head dim must be 128, got {D}"
    assert N2 == N2_k == 1, f"N2 must be 1, got N2={N2}, N2_k={N2_k}"
    if N1 > 64:
        raise ValueError(f"query head_num (N1) must be <= 64, got {N1}")
    if is_pa and (_block_size % 16 != 0 or _block_size > 1024 or _block_size <= 0):
        raise ValueError(f"PA block_size must be a positive multiple of 16 and <= 1024, got {_block_size}")
    assert query.device == key.device == weights.device, "tensors must be on same device"
    input_dtype = "float16" if query.dtype == torch.float16 else "bfloat16"

    # ===== Step 3: Early exit (output shape by original is_tnd) =====
    if int(actual_q_len.max().item()) == 0 or int(actual_k_len.max().item()) == 0:
        _out_t = q_tot if is_tnd else (B * S1)
        indices = torch.full((_out_t, N2, sparse_count), -1, dtype=torch.int32, device=query.device)
        if return_value:
            values = torch.full((_out_t, N2, sparse_count), float("-inf"), dtype=query.dtype, device=query.device)
        else:
            values = torch.empty((0,), dtype=query.dtype, device=query.device)
        return indices, values

    # ===== Step 4: PA block_table padding (unchanged) =====
    if block_table is None:
        block_table = torch.zeros((B, max(max_block_num, 1)), dtype=torch.int32, device=query.device)
    elif is_pa and block_table.shape[1] < max_block_num:
        padded = torch.full((B, max_block_num), -1, dtype=torch.int32, device=query.device)
        padded[:, : block_table.shape[1]] = block_table
        block_table = padded

    # ===== Step 5: kernel 调用（统一 BSND query 路径）=====
    max_cores = max_cores if max_cores is not None else _get_cube_core_num()

    _S1_BLOCK_h = 4 if sparse_count <= 2048 else 2
    _s1_blocks_h = (S1 + _S1_BLOCK_h - 1) // _S1_BLOCK_h
    _pa_small_bs_h = (layout_key == "PA_BSND") and (_block_size < 64)
    _s2_vec_cap_h = 256 if _pa_small_bs_h else 512
    _S2_VEC_BLOCK_h = min(_s2_vec_cap_h, ((S2 + _s2_vec_cap_h - 1) // _s2_vec_cap_h) * _s2_vec_cap_h) if S2 >= 256 else S2
    _total_bsns_h = B * _s1_blocks_h * N2
    bsn_rf = torch.zeros(_total_bsns_h, dtype=torch.int32, device=query.device)
    bsn_rl = torch.zeros(_total_bsns_h, dtype=torch.int32, device=query.device)
    _real_idx = 0
    for _b in range(B):
        _s1_r_h = (int(actual_q_len[_b]) + _S1_BLOCK_h - 1) // _S1_BLOCK_h
        _s2_r_h = (int(actual_k_len[_b]) + _S2_VEC_BLOCK_h - 1) // _S2_VEC_BLOCK_h
        for _n2 in range(N2):
            for _s1_blk in range(_s1_blocks_h):
                _bsn_idx = _b * _s1_blocks_h * N2 + _s1_blk * N2 + _n2
                if _s1_blk < _s1_r_h:
                    bsn_rf[_bsn_idx] = _real_idx
                    bsn_rl[_bsn_idx] = _real_idx + _s2_r_h - 1
                    _real_idx += _s2_r_h
                else:
                    bsn_rf[_bsn_idx] = _real_idx
                    bsn_rl[_bsn_idx] = _real_idx - 1  # empty BSN marker

    func = make_lightning_indexer_kernel(
        B=B,
        S1=S1,
        S2=S2,
        N1=N1,
        D=D,
        N2=N2,
        TOP_K=sparse_count,
        sparse_mode=sparse_mode,
        input_dtype=input_dtype,
        block_n=block_n,
        max_cores=max_cores,
        layout_key=layout_key,  # li_v2: "BSND" 或 "PA_BSND"
        block_size=_block_size,
        max_block_num=max_block_num,
        pp_slots=pp_slots,
        return_value=return_value,
    )
    func_out = func(
        query,
        key,
        weights,
        actual_q_len,
        actual_k_len,
        block_table,
        bsn_rf,
        bsn_rl,
    )
    indices, values = func_out[0], func_out[1]

    # ===== Step 6: 输出反转置（仅 TND case）=====
    if is_tnd:
        indices = _bsnd_to_tnd_output(indices, actual_q_len, q_tot)
        if return_value:
            values = _bsnd_to_tnd_output(values, actual_q_len, q_tot)
    if not return_value:
        values = torch.empty((0,), dtype=query.dtype, device=query.device)
    return indices, values


# ============================================================
# Examples (4 scenarios, one per layout) — kernel vs CPU golden
# Usage: python lightning_indexer.py [bsnd_bsnd|bsnd_pa|tnd_tnd|tnd_pa|all]
# ============================================================
if __name__ == "__main__":
    import math
    import sys
    import numpy as np
    import torch_npu

    _DEV = 0
    torch_npu.npu.set_device(_DEV)
    torch.set_default_device(f"npu:{_DEV}")
    print(f"Using device {_DEV}: {torch.npu.get_device_name(_DEV)}\n")

    def _make_rand(shape, dtype=torch.float16):
        return torch.tensor(np.random.uniform(-1, 1, shape), dtype=dtype)

    def _make_rand_w(shape, dtype=torch.float16):
        return _make_rand(shape, dtype).abs()

    def _check_result(g_idx, g_score_matrix, g_val, tl_idx, tl_val, asq, layout_query, is_pa):
        """AscendC-style 3-step verification (strict per-row).

        Step 1: Per-row index set comparison (sort + set equality).
        Step 2: For rows with differing sets, check if diff indices' scores
                have relative error < 0.0001 vs the K-th boundary score.
        Step 3: For ALL rows, compare sorted top-K values
                (rtol=0.005, atol=0.000025, after fp16 conversion, >= 95% pass).

        Returns (ok: bool, detail: str).
        """
        K = g_idx.shape[-1]
        if layout_query == "TND":
            N2 = g_idx.shape[1]
            B = g_score_matrix.shape[0]
            S1 = g_score_matrix.shape[2]
        else:  # BSND
            B = g_idx.shape[0]
            S1 = g_idx.shape[1]
            N2 = g_idx.shape[2]

        g_flat = g_idx.cpu().reshape(-1, K)
        t_flat = tl_idx.cpu().reshape(-1, K)
        total_rows = g_flat.shape[0]

        # Build q_off prefix sum for TND row -> (b, n2, s1) mapping
        asq_l = asq.tolist() if isinstance(asq, torch.Tensor) else list(asq)
        q_off = [0]
        for x in asq_l:
            q_off.append(q_off[-1] + x)

        def _row_to_bsn(r):
            if layout_query == "TND":
                q_row = r // N2
                n2 = r % N2
                for b in range(B):
                    if q_off[b] <= q_row < q_off[b + 1]:
                        return b, n2, q_row - q_off[b]
                return 0, n2, 0
            else:
                b = r // (S1 * N2)
                s1 = (r // N2) % S1
                n2 = r % N2
                return b, n2, s1

        # Precompute per-row index sets
        g_sets = []
        t_sets = []
        for r in range(total_rows):
            g_sets.append(set(int(x) for x in g_flat[r].tolist() if int(x) >= 0))
            t_sets.append(set(int(x) for x in t_flat[r].tolist() if int(x) >= 0))

        diff_rows = sum(1 for r in range(total_rows) if g_sets[r] != t_sets[r])

        # Step 2: score tolerance for differing rows (matches AscendC thres=0.0001)
        thres = 0.0001
        step2_fail = 0
        for r in range(total_rows):
            if g_sets[r] == t_sets[r]:
                continue
            b, n2, s1 = _row_to_bsn(r)
            row_scores = g_score_matrix[b, n2, s1]
            # Boundary: smallest valid score in golden top-K
            g_valid_scores = sorted([row_scores[int(i)].item() for i in g_sets[r]], reverse=True)
            value_bm = g_valid_scores[-1] if g_valid_scores else float("-inf")
            all_diff = list(g_sets[r] - t_sets[r]) + list(t_sets[r] - g_sets[r])
            for idx in all_diff:
                sc = row_scores[int(idx)].item()
                if value_bm == 0:
                    re = 0.0 if sc == 0 else float("inf")
                elif value_bm == float("-inf"):
                    re = 0.0 if sc <= -1e30 else float("inf")
                else:
                    re = abs(sc - value_bm) / abs(value_bm)
                if re > thres:
                    step2_fail += 1

        # Step 3: sorted value comparison for ALL rows (matches AscendC check_result)
        step3_fail = 0
        step3_total = 0
        if not is_pa and tl_val is not None:
            g_val_flat = g_val.cpu().to(torch.float32).reshape(-1, K)
            t_val_flat = tl_val.cpu().to(torch.float32).reshape(-1, K)
            for r in range(total_rows):
                g_sorted, _ = torch.sort(g_val_flat[r], descending=True)
                t_sorted, _ = torch.sort(t_val_flat[r], descending=True)
                g_fp16 = g_sorted.to(torch.float16).float()
                t_fp16 = t_sorted.to(torch.float16).float()
                step3_total += K
                isclose = torch.isclose(
                    g_fp16,
                    t_fp16,
                    rtol=0.005,
                    atol=0.000025,
                    equal_nan=True,
                )
                step3_fail += (~isclose).sum().item()

        # Pass logic: Step 2 all diff indices must be within tolerance (no row-level
        # exemption), Step 3 >= 95% values must pass (matches AscendC pct_thd=0.05).
        idx_pass = step2_fail == 0
        val_pass = True if is_pa else (step3_total == 0 or step3_fail / step3_total < 0.05)
        ok = idx_pass and val_pass

        detail = f"diff_rows={diff_rows}/{total_rows}, idx_fail={step2_fail}"
        if not is_pa and step3_total > 0:
            val_pct = (1 - step3_fail / step3_total) * 100
            detail += f", val_pass={val_pct:.2f}%"

        return ok, detail

    def _build_pa(k_bsnd, act_k, block_size, block_num):
        """Scatter BSND key [B, S2_max, N2, D] -> PA key [block_num, block_size, N2, D]
        and a padded block_table [B, block_num]."""
        B, S2_max, N2, D = k_bsnd.shape
        bt_cols = math.ceil(S2_max / block_size)
        np.random.seed(42)
        perm = np.random.permutation(block_num).astype(np.int32)
        bt = np.full((B, bt_cols), -1, dtype=np.int32)
        c = 0
        for b in range(B):
            for i in range(math.ceil(act_k[b] / block_size)):
                bt[b, i] = perm[c]
                c += 1
        bt_full = np.full((B, block_num), -1, dtype=np.int32)
        bt_full[:, : bt.shape[1]] = bt
        bt_t = torch.from_numpy(bt_full).to(torch.int32).to(f"npu:{_DEV}")
        k_pa = torch.zeros((block_num, block_size, N2, D), dtype=k_bsnd.dtype, device=f"npu:{_DEV}")
        for b in range(B):
            ak = act_k[b]
            for ib, bid in enumerate(bt[b]):
                if bid < 0:
                    break
                s0 = ib * block_size
                s1 = min(s0 + block_size, ak)
                if s1 <= s0:
                    break
                for n2 in range(N2):
                    k_pa[int(bid), : s1 - s0, n2, :] = k_bsnd[b, s0:s1, n2, :]
        return k_pa, bt_t

    def _run(name, lq, lk, q, k, w, asq, ask, bt, top_k, smode, block_size=None):
        is_pa = lk == "PA_BSND"
        g_idx, g_score_matrix, g_val = cpu_lightning_indexer(
            q,
            k,
            w,
            actual_seq_lengths_query=asq,
            actual_seq_lengths_key=ask,
            block_table=bt,
            layout_query=lq,
            layout_key=lk,
            sparse_count=top_k,
            sparse_mode=smode,
            block_size=block_size,
        )
        tl_idx, tl_val = lightning_indexer(
            q,
            k,
            w,
            actual_seq_lengths_query=asq,
            actual_seq_lengths_key=ask,
            block_table=bt,
            layout_query=lq,
            layout_key=lk,
            sparse_count=top_k,
            sparse_mode=smode,
            return_value=not is_pa,
        )

        torch.npu.synchronize()

        ok, detail = _check_result(g_idx, g_score_matrix, g_val, tl_idx, tl_val, asq, lq, is_pa)
        print(f"[{name}]")
        if ok:
            print("  Kernel Output Match!")
        else:
            print(f"  MISMATCH ({detail})")

    # Example 1: BSND + BSND
    def example_bsnd_bsnd():
        B, S1, S2, N1, N2, D, TOP_K, s_mode = 2, 3, 32768, 64, 1, 128, 2048, 3
        q = _make_rand((B, S1, N1, D))
        k = _make_rand((B, S2, N2, D))
        w = _make_rand_w((B, S1, N1))
        asq = torch.full((B,), S1, dtype=torch.int32)
        ask = torch.full((B,), S2, dtype=torch.int32)
        _run(
            "Example 1: BSND+BSND  (B=2, G=64, S1=3, S2=32768, TOP_K=2048, mode=3)", "BSND", "BSND", q, k, w, asq, ask, None, TOP_K, s_mode
        )

    # Example 2: BSND + PA_BSND
    def example_bsnd_pa():
        B, S1, N1, N2, D, TOP_K, s_mode = 2, 1, 8, 1, 128, 2048, 0
        block_size, block_num = 128, 32
        act_k = [1024, 2048]
        q = _make_rand((B, S1, N1, D))
        w = _make_rand_w((B, S1, N1))
        k_bsnd = _make_rand((B, max(act_k), N2, D))
        k_pa, bt_t = _build_pa(k_bsnd, act_k, block_size, block_num)
        asq = torch.full((B,), S1, dtype=torch.int32)
        ask = torch.tensor(act_k, dtype=torch.int32)
        _run(
            "Example 2: BSND+PA_BSND (B=2, G=8, S1=1, block_size=128, block_num=32, TOP_K=2048)",
            "BSND",
            "PA_BSND",
            q,
            k_pa,
            w,
            asq,
            ask,
            bt_t,
            TOP_K,
            s_mode,
            block_size,
        )

    # Example 3: TND + TND
    def example_tnd_tnd():
        B, S1, S2, N1, N2, D, TOP_K, s_mode = 8, 64, 3072, 24, 1, 128, 2048, 0
        q_t, k_t = B * S1, B * S2
        q = _make_rand((q_t, N1, D))
        k = _make_rand((k_t, N2, D))
        w = _make_rand_w((q_t, N1))
        asq = torch.cumsum(torch.full((B,), S1, dtype=torch.int32), dim=0)  # TND prefix-sum
        ask = torch.cumsum(torch.full((B,), S2, dtype=torch.int32), dim=0)  # TND prefix-sum
        _run("Example 3: TND+TND   (B=8, G=24, S1=64, S2=3072, TOP_K=2048, mode=0)", "TND", "TND", q, k, w, asq, ask, None, TOP_K, s_mode)

    # Example 4: TND + PA_BSND
    def example_tnd_pa():
        B, N1, N2, D, TOP_K, s_mode = 2, 8, 1, 128, 2048, 0
        block_size, block_num = 128, 128
        act_q, act_k = [1024, 2048], [4096, 8192]
        q_t, S2_max = sum(act_q), max(act_k)
        q = _make_rand((q_t, N1, D))
        w = _make_rand_w((q_t, N1))
        k_bsnd = _make_rand((B, S2_max, N2, D))
        k_pa, bt_t = _build_pa(k_bsnd, act_k, block_size, block_num)
        asq = torch.cumsum(torch.tensor(act_q, dtype=torch.int32), dim=0)  # TND query prefix-sum
        ask = torch.tensor(act_k, dtype=torch.int32)  # PA key per-batch
        _run(
            "Example 4: TND+PA_BSND (B=2, G=8, S1=2048, block_size=128, block_num=128, TOP_K=2048)",
            "TND",
            "PA_BSND",
            q,
            k_pa,
            w,
            asq,
            ask,
            bt_t,
            TOP_K,
            s_mode,
            block_size,
        )

    examples = {
        "bsnd_bsnd": example_bsnd_bsnd,
        "bsnd_pa": example_bsnd_pa,
        "tnd_tnd": example_tnd_tnd,
        "tnd_pa": example_tnd_pa,
    }
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if arg == "all":
        for fn in examples.values():
            fn()
    elif arg in examples:
        examples[arg]()
    else:
        print(f"Usage: python lightning_indexer.py [{('|').join(examples.keys())}|all]")
