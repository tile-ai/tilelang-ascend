import tilelang
import tilelang.language as T
from tilelang import jit
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

    q_off = [0]
    for x in asq:
        q_off.append(q_off[-1] + x)
    k_off = [0]
    for x in ask:
        k_off.append(k_off[-1] + x)

    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    wf = w.to(torch.float32)
    s1_arange = torch.arange(S1, device=_dev)

    for b in range(B):
        aq, ak = asq[b], ask[b]
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
    seg_size: int = 4096,
    block_n: Optional[int] = None,
    max_cores: Optional[int] = None,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    block_size: int = 128,
    max_block_num: int = 1,
    q_t_size: Optional[int] = None,
    k_t_size: Optional[int] = None,
    pp_slots: int = 2,  # AC arch22 uses 2 (saves 50% QK_Workspace GM)
    return_value: bool = False,
):
    G = N1 // N2
    is_tnd = layout_query == "TND"
    is_pa = layout_key == "PA_BSND"
    is_tnd_key = layout_key == "TND"
    calc_dtype = "float"

    BLOCK_N = block_n if block_n is not None else 128
    BLOCK_K = D
    _q_bufs = 2  # Q: 2 L1 buffers (ping-pong)
    _k_bufs = 3  # K: 3 L1 buffers (3-slot pipeline)

    S1_BLOCK = 8 if S1 >= 8 else (4 if TOP_K <= 2048 else 2)

    # VID_S1: each AIV processes half the S1_BLOCK rows.
    VID_S1 = (S1_BLOCK + 1) // 2 if S1_BLOCK >= 2 else 1

    M_L1 = S1_BLOCK * G
    BLOCK_M_L0 = 128
    _M_L1_padded = ((M_L1 + BLOCK_M_L0 - 1) // BLOCK_M_L0) * BLOCK_M_L0
    _num_full_iters = _M_L1_padded // BLOCK_M_L0
    _tail_m = 0
    _has_tail = False

    if block_n is None:
        # Q L1 = 2 bufs × _M_L1_padded × BLOCK_K × sizeof(fp16)
        _q_l1_kb = _q_bufs * _M_L1_padded * BLOCK_K * 2 / 1024
        # BLOCK_N selection: pick largest N that fits L0C (≤128KB) and L1 (≤500KB).
        # PA allows BLOCK_N=512 (4 K-blocks per tile); non-PA caps at 256.
        _candidates = [512, 256, 128] if is_pa else [256, 128]
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

    # Separate Cube L1 N tile (BLOCK_N) from Vector processing block (S2_VEC_BLOCK).
    # AC: S2_BASIC_BLOCK=256 (L1), S2_BASE_SIZE=512 (Vector).
    S2_VEC_BLOCK = min(512, ((S2 + 511) // 512) * 512) if S2 >= 256 else S2
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

    TARGET_S2_PG = max(512 // S2_VEC_BLOCK, 1)

    if tasks_per_core >= TARGET_S2_PG and tasks_per_core % TARGET_S2_PG == 0:
        s2_per_group = TARGET_S2_PG
        tasks_per_core // TARGET_S2_PG
    elif tasks_per_core >= num_s2_blocks and tasks_per_core % num_s2_blocks == 0:
        s2_per_group = num_s2_blocks
        tasks_per_core // num_s2_blocks
    else:
        s2_per_group = 1

    if s2_per_group > 1 and num_s2_blocks % s2_per_group != 0:
        s2_per_group = 1

    if tasks_per_core == num_s2_blocks and core_num > 1:
        for _spg in [s for s in (2, 4, 8, 16, 32) if num_s2_blocks % s == 0]:
            _cn = core_num
            while _cn > 1:
                _cn -= 1
                _tpc = (block_num + _cn - 1) // _cn
                if _tpc > num_s2_blocks and _tpc % _spg == 0:
                    core_num = _cn
                    tasks_per_core = _tpc
                    s2_per_group = _spg
                    _tpc // _spg
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
    _max_merge_slots = _max_cores_per_bsn - 1

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

    def _alloc_g_reduce_bufs():
        w_raw = T.alloc_ub((2, _w_raw_slot), input_dtype)
        mm_res = T.alloc_ub((2, VECTOR_BASEG, S2_VEC_BLOCK), calc_dtype)
        weight = T.alloc_ub(VECTOR_BASEG, calc_dtype)
        weight_2d = T.alloc_ub((VECTOR_BASEG, S2_VEC_BLOCK), calc_dtype)
        reduce_tmp = T.alloc_ub((VECTOR_BASEG, S2_VEC_BLOCK), calc_dtype)
        return w_raw, weight, weight_2d, mm_res, reduce_tmp

    _q_tot = q_t_size if q_t_size is not None else (B * S1)
    _k_tot = k_t_size if k_t_size is not None else (B * S2)
    # TND key: use 4D [1, k_tot, N2, D] to work around 3D indexing compiler issue.
    # Host wrapper unsqueezes the 3D flat key to 4D. Indexing: Key[0, offset, ...].
    _key_shape = (max_block_num, block_size, N2, D) if is_pa else (1, _k_tot, N2, D) if is_tnd_key else (B, S2, N2, D)
    _bt_shape = (B, max_block_num) if is_pa else (1, 1)

    _q_shape_q = (_q_tot * N1, D) if is_tnd else (B, S1, N1, D)
    _q_shape_w = (_q_tot, N1) if is_tnd else (B, S1, N1)
    _q_shape_o = (_q_tot, N2, TOP_K) if is_tnd else (B, S1, N2, TOP_K)

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
            QOffset: T.Tensor((B,), "int32"),
            KOffset: T.Tensor((B,), "int32"),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # Cube buffers: Q L1 (2 bufs), K L1 (3 bufs), L0A/L0B/L0C (2 bufs each).
                q_l1 = T.alloc_L1(_q_l1_shape, input_dtype)
                k_l1 = T.alloc_L1((_k_bufs, BLOCK_N, BLOCK_K), input_dtype)
                a_l0 = T.alloc_L0A((2, BLOCK_M_L0, BLOCK_K), input_dtype)
                b_l0 = T.alloc_L0B((2, BLOCK_K, _L0B_N), input_dtype)
                acc_l0c = T.alloc_L0C((2, BLOCK_M_L0, _L0B_N), calc_dtype)

                reduce_g_ub = T.alloc_ub(_BLOCK_N_VEC, calc_dtype)
                (w_raw_ub, weight_ub, weight_2d_ub, mm_res_ub, reduce_tmp_ub) = _alloc_g_reduce_bufs()

                # ===== Vector: Per-S1-row buffers =====
                (topk_a_ub, cache_tmp_ub) = _alloc_topk_bufs()

                merged_ub = T.alloc_ub(2 * _TA2, calc_dtype)
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
                    # Init: L0C[0/1] free, L0A/B[0/1] free, K[0/1/2] free, Q[0/1] free.
                    T.set_flag("fix", "m", 2)
                    T.set_flag("fix", "m", 3)
                    T.set_flag("m", "mte1", 30)
                    T.set_flag("m", "mte1", 31)
                    T.set_flag("mte1", "mte2", 20)
                    T.set_flag("mte1", "mte2", 21)
                    T.set_flag("mte1", "mte2", 22)
                    T.set_flag("mte1", "mte2", 40)
                    T.set_flag("mte1", "mte2", 41)

                    # SplitCore — real task range computed outside scope
                    _k_slot = 0
                    _pp = 0
                    _prev_bsn_cube = T.cast(-1, "int32")

                    _bN2_py_v = T.alloc_var("int32")
                    _bsn_cs_v = T.alloc_var("int32")
                    _s2_real_b_v = T.alloc_var("int32")
                    _cum_l_v = T.alloc_var("int32")

                    for _real_off in T.serial(_real_count):
                        _gloop_real = _real_start + _real_off

                        # Find bN2 by scanning prefix sums (B*N2 is compile-time, unrolled)
                        _bN2_py_v = T.cast(0, "int32")
                        _cum_l_v = T.cast(0, "int32")
                        for _bN2_h in range(B * N2):
                            _b_h = _bN2_h // N2
                            _s1_r = (actual_q_len[_b_h] + S1_BLOCK - 1) // S1_BLOCK
                            _s2_r = (actual_k_len[_b_h] + S2_VEC_BLOCK - 1) // S2_VEC_BLOCK
                            _cnt = _s1_r * _s2_r
                            _bN2_py_v = T.if_then_else(_gloop_real >= _cum_l_v, T.cast(_bN2_h, "int32"), _bN2_py_v)
                            _cum_l_v = _cum_l_v + _cnt

                        # Select bsn_cs and s2_real_b for the found bN2
                        _bsn_cs_v = T.cast(0, "int32")
                        _s2_real_b_v = T.cast(1, "int32")
                        _cum_l_v = T.cast(0, "int32")
                        for _bN2_h in range(B * N2):
                            _b_h = _bN2_h // N2
                            _s1_r = (actual_q_len[_b_h] + S1_BLOCK - 1) // S1_BLOCK
                            _s2_r = (actual_k_len[_b_h] + S2_VEC_BLOCK - 1) // S2_VEC_BLOCK
                            _cnt = _s1_r * _s2_r
                            _is_this = _bN2_py_v == T.cast(_bN2_h, "int32")
                            _bsn_cs_v = T.if_then_else(_is_this, _cum_l_v, _bsn_cs_v)
                            _s2_real_b_v = T.if_then_else(_is_this, _s2_r, _s2_real_b_v)
                            _cum_l_v = _cum_l_v + _cnt

                        _s2_real_safe = T.max(_s2_real_b_v, T.cast(1, "int32"))
                        _local_idx = _gloop_real - _bsn_cs_v
                        _b_py = _bN2_py_v // N2
                        _n2_py = _bN2_py_v % N2
                        _gS1_idx = _local_idx // _s2_real_safe
                        _s2_idx = _local_idx % _s2_real_safe

                        b_idx = _b_py
                        n2_idx = _n2_py
                        s1_blk_idx = _gS1_idx
                        s2_blk = _s2_idx
                        q_slot = (_b_py * (N2 * s1_blocks) + _gS1_idx * N2 + _n2_py) % 2
                        _bsn_v = _b_py * (N2 * s1_blocks) + _gS1_idx * N2 + _n2_py

                        k_slot = _k_slot
                        pp = _pp
                        s1_start = s1_blk_idx * S1_BLOCK
                        _q_len = actual_q_len[b_idx]
                        s2_start = s2_blk * S2_VEC_BLOCK
                        _q_need = (_gloop_real == _real_start) | (_bsn_v != _prev_bsn_cube)
                        _prev_bsn_cube = _bsn_v

                        if _q_need:
                            T.wait_flag("mte1", "mte2", 40 + q_slot)

                            if is_tnd:
                                # 2D Q: load S1_BLOCK*G rows = M_L1 rows in one T.copy
                                # QOffset[b] is in s1 units; 2D row offset = QOffset[b]*N1
                                _q_2d_start = (QOffset[b_idx] + s1_start) * N1 + n2_idx * G
                                T.copy(
                                    Query[_q_2d_start : _q_2d_start + M_L1, 0:D],
                                    q_l1[q_slot, 0, 0:M_L1, :],
                                )
                            else:
                                # BSND: per-s1_local loop with s1_idx clamping for invalid rows
                                for s1_local in range(S1_BLOCK):
                                    s1_idx = T.if_then_else(s1_start + s1_local < _q_len, s1_start + s1_local, T.max(_q_len - 1, 0))
                                    _l1_off = s1_local * G
                                    T.copy(
                                        Query[b_idx, s1_idx, n2_idx * G : n2_idx * G + G, 0:D],
                                        q_l1[q_slot, 0, _l1_off : _l1_off + G, :],
                                    )
                            T.set_flag("mte2", "mte1", q_slot)

                        # Cross-core sync + Q wait (once per gloop, before s2_sub loop)
                        T.wait_cross_flag(SYNC_V1C1)
                        if _q_need:
                            T.wait_flag("mte2", "mte1", q_slot)

                        for s2_sub in range(_S2_SUB_COUNT):
                            _s2_sub_offset = s2_sub * BLOCK_N
                            _s2_sub_start = s2_start + _s2_sub_offset

                            # K load (BLOCK_N columns per sub-block)
                            T.wait_flag("mte1", "mte2", 20 + k_slot)
                            if is_pa:
                                for sub in range(_BLOCKS_PER_TILE):
                                    _block_table_idx = s2_blk * _BLOCKS_PER_TILE * _S2_SUB_COUNT + s2_sub * _BLOCKS_PER_TILE + sub
                                    _safe_block_table_idx = T.min(_block_table_idx, T.cast(max_block_num - 1, "int32"))
                                    T.copy(
                                        Key[BlockTable[b_idx, _safe_block_table_idx], 0:block_size, n2_idx, 0:D],
                                        k_l1[k_slot, sub * block_size : (sub + 1) * block_size, :],
                                    )
                            elif is_tnd_key:
                                T.copy(
                                    Key[0, KOffset[b_idx] + _s2_sub_start : KOffset[b_idx] + _s2_sub_start + BLOCK_N, n2_idx, 0:D],
                                    k_l1[k_slot, :, :],
                                )
                            else:
                                T.copy(Key[b_idx, _s2_sub_start : _s2_sub_start + BLOCK_N, n2_idx, 0:D], k_l1[k_slot, :, :])
                            T.set_flag("mte2", "mte1", 10 + k_slot)
                            T.wait_flag("mte2", "mte1", 10 + k_slot)

                            for m_iter in range(_num_full_iters):
                                for n_l0 in range(_N_SPLIT):
                                    side = (m_iter * _N_SPLIT + n_l0) % 2
                                    _nlo = n_l0 * _L0B_N
                                    _nhi = _nlo + _L0B_N
                                    _m_off = m_iter * BLOCK_M_L0
                                    T.wait_flag("fix", "m", 2 + side)
                                    T.wait_flag("m", "mte1", 30 + side)
                                    T.copy(q_l1[q_slot, 0, _m_off : _m_off + BLOCK_M_L0, :], a_l0[side, :, :])
                                    T.copy(k_l1[k_slot, _nlo:_nhi, :], b_l0[side, :, :], transpose=True)
                                    T.set_flag("mte1", "m", 30 + side)
                                    T.wait_flag("mte1", "m", 30 + side)
                                    T.mma(a_l0[side, :, :], b_l0[side, :, :], acc_l0c[side, :, :], init=True)
                                    T.set_flag("m", "mte1", 30 + side)
                                    T.set_flag("m", "fix", 2 + side)
                                    T.wait_flag("m", "fix", 2 + side)
                                    T.copy(
                                        acc_l0c[side, :, :],
                                        QK_Workspace[cid, pp, _m_off : _m_off + BLOCK_M_L0, _s2_sub_offset + _nlo : _s2_sub_offset + _nhi],
                                        enable_relu=True,
                                    )
                                    T.set_flag("fix", "m", 2 + side)

                            if _has_tail:
                                _m_off = _num_full_iters * BLOCK_M_L0
                                for n_l0 in range(_N_SPLIT):
                                    side = (_num_full_iters * _N_SPLIT + n_l0) % 2
                                    _nlo = n_l0 * _L0B_N
                                    _nhi = _nlo + _L0B_N
                                    T.wait_flag("fix", "m", 2 + side)
                                    T.wait_flag("m", "mte1", 30 + side)
                                    T.copy(q_l1[q_slot, 0, _m_off : _m_off + BLOCK_M_L0, :], a_l0[side, :, :])
                                    T.copy(k_l1[k_slot, _nlo:_nhi, :], b_l0[side, :, :], transpose=True)
                                    T.set_flag("mte1", "m", 30 + side)
                                    T.wait_flag("mte1", "m", 30 + side)
                                    T.mma(a_l0[side, :, :], b_l0[side, :, :], acc_l0c[side, :_tail_m, :], init=True)
                                    T.set_flag("m", "mte1", 30 + side)
                                    T.set_flag("m", "fix", 2 + side)
                                    T.wait_flag("m", "fix", 2 + side)
                                    T.copy(
                                        acc_l0c[side, :_tail_m, :],
                                        QK_Workspace[cid, pp, _m_off : _m_off + _tail_m, _s2_sub_offset + _nlo : _s2_sub_offset + _nhi],
                                        enable_relu=True,
                                    )
                                    T.set_flag("fix", "m", 2 + side)

                            # Free K buffer for next s2_sub or next gloop
                            T.set_flag("mte1", "mte2", 20 + k_slot)

                        _bsn_end = (_s2_idx == _s2_real_b_v - 1) | (_gloop_real == _real_end - 1)
                        if _bsn_end:
                            T.set_flag("mte1", "mte2", 40 + q_slot)

                        # K buffer already freed inside s2_sub loop (last iteration).
                        T.pipe_barrier("FIX")
                        T.set_cross_flag("FIX", SYNC_C1V1)

                        _k_slot = _k_slot + 1
                        if _k_slot >= 3:
                            _k_slot = 0
                        _pp = _pp + 1
                        if _pp >= pp_slots:
                            _pp = 0

                    T.wait_flag("fix", "m", 2)
                    T.wait_flag("fix", "m", 3)
                    T.wait_flag("m", "mte1", 30)
                    T.wait_flag("m", "mte1", 31)
                    T.wait_flag("mte1", "mte2", 20)
                    T.wait_flag("mte1", "mte2", 21)
                    T.wait_flag("mte1", "mte2", 22)
                    T.wait_flag("mte1", "mte2", 40)
                    T.wait_flag("mte1", "mte2", 41)
                    T.sync_all()

                # =================================================================
                # V scope: Phase 1-V + Phase 2 (manual sync matching auto_sync)
                # =================================================================
                with T.Scope("V"):
                    # Counting semaphore init — pre-set SYNC_V1C1 pp_slots times.
                    T.set_cross_flag("MTE2", SYNC_V1C1)
                    T.set_cross_flag("MTE2", SYNC_V1C1)

                    T.set_flag("V", "MTE2", 13)
                    T.set_flag("V", "MTE2", 14)

                    T.tile.fill(stride2_blk_ub, 0)
                    T.pipe_barrier("V")
                    for _i in range(_K_PER_BLOCK):
                        stride2_blk_ub[_i * 2 + 1] = T.cast(1, calc_dtype)

                    T.tile.arith_progression(index_blk_ub, T.cast(0, calc_dtype), T.cast(1, calc_dtype), _BLOCK_N_VEC)

                    prev_bsn_ub[0] = -1
                    T.tile.fill(topk_a_ub, -T.infinity(calc_dtype))

                    # SplitCore — real task range computed outside scope (V scope)
                    _pp = 0

                    _bN2_py_v = T.alloc_var("int32")
                    _bsn_cs_v = T.alloc_var("int32")
                    _s2_real_b_v = T.alloc_var("int32")
                    _cum_l_v = T.alloc_var("int32")

                    for _real_off in T.serial(_real_count):
                        _gloop_real = _real_start + _real_off

                        _bN2_py_v = T.cast(0, "int32")
                        _cum_l_v = T.cast(0, "int32")
                        for _bN2_h in range(B * N2):
                            _b_h = _bN2_h // N2
                            _s1_r = (actual_q_len[_b_h] + S1_BLOCK - 1) // S1_BLOCK
                            _s2_r = (actual_k_len[_b_h] + S2_VEC_BLOCK - 1) // S2_VEC_BLOCK
                            _cnt = _s1_r * _s2_r
                            _bN2_py_v = T.if_then_else(_gloop_real >= _cum_l_v, T.cast(_bN2_h, "int32"), _bN2_py_v)
                            _cum_l_v = _cum_l_v + _cnt

                        _bsn_cs_v = T.cast(0, "int32")
                        _s2_real_b_v = T.cast(1, "int32")
                        _cum_l_v = T.cast(0, "int32")
                        for _bN2_h in range(B * N2):
                            _b_h = _bN2_h // N2
                            _s1_r = (actual_q_len[_b_h] + S1_BLOCK - 1) // S1_BLOCK
                            _s2_r = (actual_k_len[_b_h] + S2_VEC_BLOCK - 1) // S2_VEC_BLOCK
                            _cnt = _s1_r * _s2_r
                            _is_this = _bN2_py_v == T.cast(_bN2_h, "int32")
                            _bsn_cs_v = T.if_then_else(_is_this, _cum_l_v, _bsn_cs_v)
                            _s2_real_b_v = T.if_then_else(_is_this, _s2_r, _s2_real_b_v)
                            _cum_l_v = _cum_l_v + _cnt

                        _s2_real_safe = T.max(_s2_real_b_v, T.cast(1, "int32"))
                        _local_idx = _gloop_real - _bsn_cs_v
                        _b_py = _bN2_py_v // N2
                        _n2_py = _bN2_py_v % N2
                        _gS1_idx = _local_idx // _s2_real_safe
                        _s2_idx = _local_idx % _s2_real_safe

                        b_idx = _b_py
                        n2_idx = _n2_py
                        s1_blk_idx = _gS1_idx
                        s2_blk = _s2_idx

                        pp = _pp
                        s1_start = s1_blk_idx * S1_BLOCK
                        _q_len_b = actual_q_len[b_idx]
                        _k_len_b = actual_k_len[b_idx]
                        s2_start = s2_blk * S2_VEC_BLOCK
                        _bsn_v = _b_py * (N2 * s1_blocks) + _gS1_idx * N2 + _n2_py

                        if prev_bsn_ub[0] >= 0 and _bsn_v != prev_bsn_ub[0]:
                            T.tile.fill(topk_a_ub, -T.infinity(calc_dtype))
                        prev_bsn_ub[0] = _bsn_v

                        T.wait_cross_flag(SYNC_C1V1)

                        for s1_local in range(VID_S1):
                            actual_s1 = vid * VID_S1 + s1_local

                            s1_idx = s1_start + actual_s1
                            s2_valid = _k_len_b
                            causal_limit = _k_len_b - _q_len_b + s1_idx + 1
                            _mode_val = T.cast(sparse_mode, "int32")
                            _is_causal = _mode_val == T.cast(3, "int32")
                            s2_valid = T.if_then_else(_is_causal & (causal_limit > 0), causal_limit, s2_valid)

                            # ===== Phase A: g-reduce + reduce_sum + mask (MTE2 + V) =====
                            if s1_start + actual_s1 < _q_len_b and s2_start < s2_valid:
                                T.tile.fill(reduce_tmp_ub, 0)
                                for g_id in T.unroll(num_g_groups):
                                    qk_g_start = actual_s1 * G + g_id * VECTOR_BASEG
                                    _wg = g_id * VECTOR_BASEG
                                    _gpp = g_id % 2
                                    _flag_id = 13 + _gpp
                                    T.wait_flag("V", "MTE2", _flag_id)
                                    if is_tnd:
                                        T.copy(
                                            Weights[QOffset[b_idx] + s1_idx, n2_idx * G + _wg : n2_idx * G + _wg + VECTOR_BASEG],
                                            w_raw_ub[_gpp, :],
                                        )
                                    else:
                                        T.copy(
                                            Weights[b_idx, s1_idx, n2_idx * G + _wg : n2_idx * G + _wg + VECTOR_BASEG],
                                            w_raw_ub[_gpp, :],
                                        )
                                    T.copy(
                                        QK_Workspace[cid, pp, qk_g_start : qk_g_start + VECTOR_BASEG, 0:S2_VEC_BLOCK], mm_res_ub[_gpp, :, :]
                                    )
                                    T.set_flag("MTE2", "V", _flag_id)
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
                                T.tile.select(reduce_g_ub, mask_blk_ub, reduce_g_ub, -T.infinity(calc_dtype), "VSEL_TENSOR_SCALAR_MODE")

                            if s1_local == VID_S1 - 1:
                                T.set_cross_flag("MTE2", SYNC_V1C1)

                            # ===== Phase B: sort + merge (V only, no MTE2) =====
                            if s1_start + actual_s1 < _q_len_b and s2_start < s2_valid:
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
                                # Cross-core BSN: save to TopK_Workspace
                                # Positional slot: 0=natural BSN end (incoming), 1=core boundary (outgoing)
                                _save_bsn = T.if_then_else(_is_natural_bsn_end, 0, 1)
                                T.set_flag("V", "MTE3", 2)
                                T.wait_flag("V", "MTE3", 2)
                                for _si in range(VID_S1):
                                    T.copy(
                                        topk_a_ub[_si, :],
                                        TopK_Workspace[cid, _save_bsn, vid * VID_S1 + _si, 0:_TA2],
                                    )
                                T.pipe_barrier("MTE3")
                            else:
                                # Opt2: Non-cross-core BSN — direct Extract+Cast+CopyOut
                                for _si in range(VID_S1):
                                    actual_s1_d = vid * VID_S1 + _si
                                    s1_idx_d = s1_start + actual_s1_d
                                    if s1_start + actual_s1_d < _q_len_b:
                                        T.copy(topk_a_ub[_si, :], p2_acc_ub)
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
                                        T.set_flag("V", "MTE3", 8)
                                        T.wait_flag("V", "MTE3", 8)
                                        if is_tnd:
                                            _o_row_d = QOffset[b_idx] + s1_idx_d
                                            T.copy(output_ub, Out[_o_row_d, n2_idx, 0:TOP_K])
                                        else:
                                            T.copy(output_ub, Out[b_idx, s1_idx_d, n2_idx, 0:TOP_K])
                                        T.set_flag("MTE3", "V", 8)
                                        T.wait_flag("MTE3", "V", 8)
                                        if return_value:
                                            T.set_flag("V", "MTE3", 5)
                                            T.wait_flag("V", "MTE3", 5)
                                            if is_tnd:
                                                _o_row_d = QOffset[b_idx] + s1_idx_d
                                                T.copy(output_val_ub, OutVal[_o_row_d, n2_idx, 0:TOP_K])
                                            else:
                                                T.copy(output_val_ub, OutVal[b_idx, s1_idx_d, n2_idx, 0:TOP_K])
                                            T.set_flag("MTE3", "V", 15)
                                            T.wait_flag("MTE3", "V", 15)

                        _pp = _pp + 1
                        if _pp >= pp_slots:
                            _pp = 0

                    T.wait_flag("V", "MTE2", 13)
                    T.wait_flag("V", "MTE2", 14)

                    T.barrier_all()
                    T.sync_all()

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
                        _q_off_p2 = QOffset[b_idx_p2]
                        if s1_idx_p2 < _q_len_p2 and out_s1_local < S1:
                            # Real-space cross-core detection using prefix sums
                            _s2_r_p2 = (actual_k_len[b_idx_p2] + S2_VEC_BLOCK - 1) // S2_VEC_BLOCK
                            _s2_r_p2_safe = T.max(_s2_r_p2, T.cast(1, "int32"))
                            # Compute bsn_real_first via while loop (avoids LetStmt chain)
                            _bsn_rf_v = T.alloc_var("int32")
                            _bi_v = T.alloc_var("int32")
                            _bsn_rf_v = T.cast(0, "int32")
                            _bi_v = T.cast(0, "int32")
                            while _bi_v < b_idx_p2:
                                _s1_r_h = (actual_q_len[_bi_v] + S1_BLOCK - 1) // S1_BLOCK
                                _s2_r_h = (actual_k_len[_bi_v] + S2_VEC_BLOCK - 1) // S2_VEC_BLOCK
                                _bsn_rf_v = _bsn_rf_v + _s1_r_h * _s2_r_h * N2
                                _bi_v = _bi_v + T.cast(1, "int32")
                            _bsn_rf_v = _bsn_rf_v + s1_blk_idx_p2 * _s2_r_p2_safe * N2 + n2_idx_p2 * _s2_r_p2_safe
                            _bsn_rl_v = _bsn_rf_v + _s2_r_p2_safe - 1
                            _tpc_safe_p2 = T.max(_tpc_real, T.cast(1, "int32"))
                            _rc_first = _bsn_rf_v // _tpc_safe_p2
                            _rc_last = _bsn_rl_v // _tpc_safe_p2
                            if _rc_first != _rc_last:
                                # Cross-core BSN: workspace read + merge + extract + cast + copyout
                                # Positional slots: _rc_first saved at core boundary (slot 1),
                                # intermediate cores saved at core boundary (slot 1),
                                # _rc_last saved at natural BSN end (slot 0).
                                T.copy(TopK_Workspace[_rc_first, 1, out_s1_local, 0:_TA2], p2_acc_ub)
                                T.set_flag("MTE2", "V", 3)
                                T.wait_flag("MTE2", "V", 3)
                                if NEED_CROSS_CORE:
                                    num_merge_p2 = _rc_last - _rc_first
                                    for m in T.serial(num_merge_p2):
                                        other_cid = _rc_first + 1 + m
                                        _slot_p2 = T.if_then_else(other_cid == _rc_last, 0, 1)
                                        T.barrier_all()
                                        T.copy(TopK_Workspace[other_cid, _slot_p2, out_s1_local, 0:_TA2], topk_a_ub[0, :])
                                        T.set_flag("MTE2", "V", 4)
                                        T.wait_flag("MTE2", "V", 4)
                                        T.tile.merge_sort(merged_ub, p2_acc_ub, topk_a_ub[0, :])
                                        T.pipe_barrier("V")
                                        T.copy(merged_ub[0:_TA2], p2_acc_ub)
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

                                T.set_flag("V", "MTE3", 2)
                                T.wait_flag("V", "MTE3", 2)
                                if is_tnd:
                                    _o_row_p2 = _q_off_p2 + s1_idx_p2
                                    T.copy(output_ub, Out[_o_row_p2, n2_idx_p2, 0:TOP_K])
                                else:
                                    T.copy(output_ub, Out[b_idx_p2, s1_idx_p2, n2_idx_p2, 0:TOP_K])
                                if return_value:
                                    T.set_flag("MTE3", "V", 15)
                                    T.wait_flag("MTE3", "V", 15)
                                    T.set_flag("V", "MTE3", 5)
                                    T.wait_flag("V", "MTE3", 5)
                                    if is_tnd:
                                        _o_row_p2 = _q_off_p2 + s1_idx_p2
                                        T.copy(output_val_ub, OutVal[_o_row_p2, n2_idx_p2, 0:TOP_K])
                                    else:
                                        T.copy(output_val_ub, OutVal[b_idx_p2, s1_idx_p2, n2_idx_p2, 0:TOP_K])
                                T.barrier_all()
                            else:
                                T.barrier_all()
                        else:
                            if not is_tnd:
                                T.tile.fill(output_ub, -1)
                                T.pipe_barrier("V")
                                T.copy(output_ub, Out[b_idx_p2, s1_idx_p2, n2_idx_p2, 0:TOP_K])
                                T.barrier_all()

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
    sparse_mode: int = 0,
    pre_tokens: int = (1 << 63) - 1,
    next_tokens: int = (1 << 63) - 1,
    return_value: bool = False,
    seg_size: int = 4096,
    block_n: Optional[int] = None,
    pp_slots: int = 2,  # AC arch22 uses 2 (saves 50% QK_Workspace GM)
    max_cores: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert layout_query in ("BSND", "TND"), f"Unsupported query layout: {layout_query}"
    assert layout_key in ("BSND", "PA_BSND", "TND"), f"Unsupported key layout: {layout_key}"
    is_pa = layout_key == "PA_BSND"
    is_tnd = layout_query == "TND"
    is_tnd_key = layout_key == "TND"

    if is_pa and return_value:
        raise ValueError("PA_BSND layout does not support return_value; use return_value=False")

    if is_tnd:
        q_tot, N1, D = query.shape
        _, N2 = weights.shape[1], key.shape[1] if is_tnd_key else key.shape[2]
        B = actual_seq_lengths_query.shape[0] if actual_seq_lengths_query is not None else 1
        S1 = max(actual_seq_lengths_query.max().item() if actual_seq_lengths_query is not None else q_tot, 1)
    else:
        B, S1, N1, D = query.shape
        N2 = key.shape[2]
        q_tot = B * S1

    if is_pa:
        assert block_table is not None
        max_block_num, _block_size, N2_k, _ = key.shape
        if is_tnd:
            # TND query: S2 from actual key length per batch (not physical block count)
            S2 = max(actual_seq_lengths_key.max().item() if actual_seq_lengths_key is not None else 0, 1)
            # num_s2_blocks >= 2 (same as TND+TND safeguard)
            S2 = max(S2, 257)
        else:
            S2 = max_block_num * _block_size
    elif is_tnd_key:
        _k_tot, N2_k, _ = key.shape
        S2 = max(actual_seq_lengths_key.max().item() if actual_seq_lengths_key is not None else _k_tot, 1)
        S2 = max(S2, 257)
        _block_size, max_block_num = 128, 1
    else:
        _, S2, N2_k, _ = key.shape
        _block_size, max_block_num = 128, 1

    assert D == 128
    assert query.device == key.device == weights.device, "query, key, and weights must be on the same device"
    assert N2 > 0, f"N2 must be greater than 0, got {N2}"
    assert N1 % N2 == 0, f"N1 must be divisible by N2, got N1={N1}, N2={N2}"
    input_dtype = "float16" if query.dtype == torch.float16 else "bfloat16"

    if actual_seq_lengths_query is None:
        actual_seq_lengths_query = torch.full((B,), S1, dtype=torch.int32, device=query.device)
    else:
        actual_seq_lengths_query = actual_seq_lengths_query.to(device=query.device, dtype=torch.int32)
    if actual_seq_lengths_key is None:
        actual_seq_lengths_key = torch.full((B,), S2, dtype=torch.int32, device=query.device)
    else:
        actual_seq_lengths_key = actual_seq_lengths_key.to(device=query.device, dtype=torch.int32)

    _max_q = int(actual_seq_lengths_query.max().item()) if actual_seq_lengths_query.numel() > 0 else 0
    _max_k = int(actual_seq_lengths_key.max().item()) if actual_seq_lengths_key.numel() > 0 else 0
    if _max_q == 0 or _max_k == 0:
        _out_t = q_tot if is_tnd else (B * S1)
        indices = torch.full((_out_t, N2, sparse_count), -1, dtype=torch.int32, device=query.device)
        if return_value:
            values = torch.full((_out_t, N2, sparse_count), float("-inf"), dtype=query.dtype, device=query.device)
        else:
            values = torch.empty((0,), dtype=query.dtype, device=query.device)
        return indices, values

    # TND: compute prefix-sum offsets on host
    if is_tnd:
        q_offset = torch.zeros(B, dtype=torch.int32, device=query.device)
        q_offset[1:] = actual_seq_lengths_query.cumsum(0)[:-1]
    else:
        q_offset = torch.zeros(B, dtype=torch.int32, device=query.device)
    if is_tnd_key:
        k_offset = torch.zeros(B, dtype=torch.int32, device=query.device)
        k_offset[1:] = actual_seq_lengths_key.cumsum(0)[:-1]
    else:
        k_offset = torch.zeros(B, dtype=torch.int32, device=query.device)

    if is_tnd_key:
        _s2_eff = max(int(actual_seq_lengths_key.max().item()), S2)
        _pad_need = _k_tot + _s2_eff + 256
        if _k_tot < _pad_need:
            _pad = _pad_need - _k_tot
            key = torch.cat([key, torch.zeros((_pad,) + tuple(key.shape[1:]), dtype=key.dtype, device=key.device)], dim=0)
            _k_tot = key.shape[0]

    if block_table is None:
        block_table = torch.zeros((B, max(max_block_num, 1)), dtype=torch.int32, device=query.device)

    max_cores = max_cores if max_cores is not None else _get_cube_core_num()

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
        seg_size=seg_size,
        block_n=block_n,
        max_cores=max_cores,
        layout_query=layout_query,
        layout_key=layout_key,
        block_size=_block_size,
        max_block_num=max_block_num,
        q_t_size=q_tot if is_tnd else None,
        k_t_size=_k_tot if is_tnd_key else None,
        pp_slots=pp_slots,
        return_value=return_value,
    )
    if is_tnd_key:
        key = key.unsqueeze(0)

    if is_tnd:
        query = query.reshape(-1, D)
        # Pad TND query so the last batch's last s1_block doesn't read past the
        # end. The kernel reads S1_BLOCK*G rows per s1_block in a single Nd2Nz;
        # if actual_q_len[B-1] < S1, the tail block reads extra rows beyond q_tot.
        _s1_block = 8 if S1 >= 8 else (4 if sparse_count <= 2048 else 2)
        _q_pad = _s1_block * N1
        query = torch.cat([query, torch.zeros(_q_pad, D, dtype=query.dtype, device=query.device)], dim=0)

    func_out = func(
        query,
        key,
        weights,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        q_offset,
        k_offset,
    )
    indices, values = func_out[0], func_out[1]
    if is_tnd:
        indices = indices[:q_tot]
        values = values[:q_tot]
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
        asq = torch.full((B,), S1, dtype=torch.int32)
        ask = torch.full((B,), S2, dtype=torch.int32)
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
        asq = torch.tensor(act_q, dtype=torch.int32)
        ask = torch.tensor(act_k, dtype=torch.int32)
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
