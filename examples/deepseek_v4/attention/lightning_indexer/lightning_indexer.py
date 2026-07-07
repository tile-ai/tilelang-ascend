"""
LightningIndexer — sparse attention index selector for Ascend NPU (TileLang-Ascend DSL).

For each query row, scores every key position by
    score[s2] = sum_g relu(Q[s1, n2*G+g, :] . K[s2, n2, :]) * W[s1, n2*G+g]
(G = N1 // N2 group-query factor), applies a causal/boundary mask, then emits the
top-K key indices (and scores). Output indices use -1 for slots beyond the valid
key count.

Layouts (layout_query x layout_key):
  BSND + BSND    : Q/W [B,S1,N1,D]/[B,S1,N1], K [B,S2,N2,D]
  BSND + PA_BSND : K scattered into [block_num, block_size, N2, D] via block_table
  TND  + TND     : Q/W flat [T,N1,D]/[T,N1], K flat [T,N2,D]; actual_seq per batch
  TND  + PA_BSND : Q flat, K paged

actual_seq_lengths are per-batch for the wrapper; TND offsets are prefix-sums
computed on the host.

Pipeline: Cube scope runs the Q.K GEMM (mma, L1/L0 ping-pong, 3-slot K pipeline);
Vector scope does the G-reduce, mask, per-block top-K sort, deferred 3-slot merge,
and a dispersed cross-core Phase-2 merge that writes the final top-K indices/scores.
C/V overlap via counting-semaphore cross-core flags; auto_sync off (manual sync).
"""

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
    pp_slots: int = 2,
    return_value: bool = False,
):
    G = N1 // N2
    is_tnd = layout_query == "TND"
    is_pa = layout_key == "PA_BSND"
    is_tnd_key = layout_key == "TND"
    calc_dtype = "float"

    CUBE_ALIGN = 16
    BLOCK_N = block_n if block_n is not None else 128
    BLOCK_K = D

    S1_BLOCK = 4 if TOP_K <= 2048 else 2
    if S1 < S1_BLOCK:
        S1_BLOCK = max(S1, 1)

    # Per-vid S1 rows: each AIV processes half (S1_BLOCK >= 2)
    # S1_BLOCK=1: VID_S1=1 and vid 0 only (no splitting benefit, no regression)
    # ceil division ensures odd S1_BLOCK covers all rows (e.g. 3→2, not 3→1)
    VID_S1 = (S1_BLOCK + 1) // 2 if S1_BLOCK >= 2 else 1

    G_padded = ((G + CUBE_ALIGN - 1) // CUBE_ALIGN) * CUBE_ALIGN

    if block_n is None:
        _bm = min(G_padded, 128)
        # Q uses 2 L1 buffers, K uses 3 (3-slot K pipeline)
        _q_l1_kb = 2 * S1_BLOCK * max(G_padded // _bm, 1) * _bm * 128 * 2 / 1024
        # PA: BLOCK_N >= 128 (gather across physical blocks if block_size < 128).
        _max_n = min(256, max(block_size, 128)) if is_pa else 256
        for _test_n in [_max_n, 128]:
            _l0c_kb = 2 * _bm * _test_n * 4 / 1024
            _k_l1_kb = 3 * _test_n * 128 * 2 / 1024
            if _l0c_kb <= 128 and (_q_l1_kb + _k_l1_kb) <= 500:
                BLOCK_N = _test_n
                break
    # PA gather: physical blocks (block_size each) per BLOCK_N K tile. 1 if block_size>=128.
    _BLOCKS_PER_TILE = (BLOCK_N // block_size) if is_pa else 1

    if G_padded <= 128:
        BLOCK_M_CUBE = G_padded
        M_SUB_TILES = 1
    else:
        BLOCK_M_CUBE = 128
        M_SUB_TILES = G_padded // BLOCK_M_CUBE

    S2_padded = ((S2 + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    num_s2_blocks = S2_padded // BLOCK_N

    # Prefer larger VECTOR_BASEG to cut G-reduce g_id loop count.
    # VG=32 → halves iterations vs VG=16 (-8.7%). VG=64 overflows UB (mm_res+weight_2d=128KB).
    if G % 32 == 0:
        VECTOR_BASEG = 32
    elif G % 16 == 0:
        VECTOR_BASEG = 16
    elif G % 8 == 0:
        VECTOR_BASEG = 8
    else:
        VECTOR_BASEG = G

    TOP_K_ALIGNED = ((TOP_K + 63) // 64) * 64
    _K_PER_BLOCK = min(TOP_K, BLOCK_N)

    # PA with block_size < 64: pad Vector-scope buffers to satisfy
    # T.tile.compare 256-byte alignment (CompareScalarCodegen requires
    # BLOCK_N*sizeof(calc_dtype) ≥ 256, i.e. BLOCK_N ≥ 64).
    # Cube scope stays at original BLOCK_N to read within one PA block.
    _BLOCK_N_VEC = max(BLOCK_N, 64) if is_pa else BLOCK_N
    _NEED_VEC_PAD = _BLOCK_N_VEC > BLOCK_N

    s1_blocks = (S1 + S1_BLOCK - 1) // S1_BLOCK
    num_bsns = B * s1_blocks * N2
    block_num = B * s1_blocks * N2 * num_s2_blocks

    if max_cores is None or max_cores <= 0:
        max_cores = _get_cube_core_num()

    core_num = min(block_num, max_cores)
    tasks_per_core = (block_num + core_num - 1) // core_num

    TARGET_S2_PG = max(512 // BLOCK_N, 1)

    if tasks_per_core >= TARGET_S2_PG and tasks_per_core % TARGET_S2_PG == 0:
        s2_per_group = TARGET_S2_PG
        bsn_groups = tasks_per_core // TARGET_S2_PG
    elif tasks_per_core >= num_s2_blocks and tasks_per_core % num_s2_blocks == 0:
        s2_per_group = num_s2_blocks
        bsn_groups = tasks_per_core // num_s2_blocks
    else:
        s2_per_group = 1
        bsn_groups = tasks_per_core

    # When s2_per_group > 1 and BSN boundaries don't align with
    # group boundaries (num_s2_blocks % s2_per_group != 0), tasks from
    # different BSNs can fall into the same bsn_off group. The Cube scope's
    # Q/K preload and Vector scope's b_idx/n2_idx use group_task_id, which
    # belongs to the first task's BSN — wrong for the second task's BSN.
    # Force s2_per_group=1 when alignment isn't guaranteed.
    if s2_per_group > 1 and num_s2_blocks % s2_per_group != 0:
        s2_per_group = 1
        bsn_groups = tasks_per_core

    # Compile-fix: when tasks_per_core == num_s2_blocks (structural when
    # B*s1_blocks*N2 == core_num), tir.transform.Simplify hangs on the flag IR's
    # (X*tasks_per_core)//num_s2_blocks indexing. Break the equality with the smallest
    # core_num reduction keeping spg>1 and num_s2_blocks % spg == 0. ONLY triggers on
    # this rare equality (case 3/10: B=20 S1=3); common cases (large B*S1) never hit it.
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
                    bsn_groups = _tpc // _spg
                    break
            if tasks_per_core > num_s2_blocks:
                break

    # Per-core local-BSN workspace (recomputed after core_num reduction).
    _max_bsns_per_core = 0
    for _c in range(core_num):
        _first = (_c * tasks_per_core) // num_s2_blocks
        _last = min(((_c + 1) * tasks_per_core - 1) // num_s2_blocks, num_bsns - 1)
        if _last >= _first:
            _max_bsns_per_core = max(_max_bsns_per_core, _last - _first + 1)
    _max_bsns_per_core = max(_max_bsns_per_core, 1)

    # BSN task range is exactly num_s2_blocks tasks per BSN.
    # Previously BSN_TASK_SPAN = num_s2_blocks * s1_blocks * N2 was used,
    # which incorrectly spanned an entire batch's tasks, causing:
    #   1) BSN change NOT detected between S1 blocks within same batch
    #   2) Workspace writes to wrong slot (batch index vs. BSN index)
    #   3) Cross-core merge reading from wrong cores
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

    qk_ws_shape = (core_num, pp_slots, S1_BLOCK * G_padded, BLOCK_N)
    # Per-core local-BSN workspace — [core_num, max_bsns_per_core, S1_BLOCK, _TA2]
    # Each core accesses its local BSN slot, keeping per-BSN stride small (~65KB).
    topk_ws_shape = (core_num, _max_bsns_per_core, S1_BLOCK, 2 * TOP_K_ALIGNED)

    num_g_groups = G // VECTOR_BASEG
    # C/V sync via counting semaphores (single flag ID per direction).
    # SYNC_C1V1: cube->vector (pp ready), SYNC_V1C1: vector->cube (pp free, init count=pp_slots)
    SYNC_C1V1 = 0
    SYNC_V1C1 = 1

    def _align32(x):
        return ((x + 31) // 32) * 32

    # Q uses 2 L1 buffers, K uses 3 (3-slot K pipeline, 5 L1 buffers total)
    _q_bufs = 2
    _k_bufs = 3
    _q_l1_bytes = _q_bufs * S1_BLOCK * M_SUB_TILES * BLOCK_M_CUBE * BLOCK_K * 2
    _k_l1_bytes = _k_bufs * BLOCK_N * BLOCK_K * 2
    _q_l1_addr = 0
    _k_l1_addr = _align32(_q_l1_bytes)
    _l1_total = _k_l1_addr + _k_l1_bytes
    if _l1_total > 524032:
        raise RuntimeError(f"L1 overflow: {_l1_total} > 524032")
    _acc_l0c_addr = 0

    # =====================================================================
    # Buffer allocation helpers
    # =====================================================================
    _TA2 = 2 * TOP_K_ALIGNED
    _KP2 = 2 * _K_PER_BLOCK

    def _alloc_topk_bufs():
        """Deferred merge: 3 cache slots + per-row topk accumulator.
        UB counter tracks filled slots. Every 3 s2 blocks → 4-way merge(topk_a+3 slots).
        BSN change or end: tail-merge remaining slots (1-2) into topk_a before flush.
        """
        topk_a = T.alloc_ub((VID_S1, _TA2), calc_dtype)  # per-row accumulator
        cache_tmp = T.alloc_ub(_KP2, calc_dtype)  # topk temp buffer
        cache_slot0 = T.alloc_ub((VID_S1, _KP2), calc_dtype)  # deferred merge slot 0
        cache_slot1 = T.alloc_ub((VID_S1, _KP2), calc_dtype)  # deferred merge slot 1
        cache_slot2 = T.alloc_ub((VID_S1, _KP2), calc_dtype)  # deferred merge slot 2
        return topk_a, cache_tmp, cache_slot0, cache_slot1, cache_slot2

    def _alloc_g_reduce_bufs():
        """G-dim reduce buffers.
        v16: All paths use 1× buffers (same as v15).
             T.Pipelined handles DMA/Compute sync without needing ping-pong doubling.
             For num_g_groups==1: reduce_tmp is dummy (1 element).
        """
        w_raw = T.alloc_ub(VECTOR_BASEG, input_dtype)
        weight = T.alloc_ub(VECTOR_BASEG, calc_dtype)
        weight_2d = T.alloc_ub((VECTOR_BASEG, BLOCK_N), calc_dtype)
        mm_res = T.alloc_ub((VECTOR_BASEG, BLOCK_N), calc_dtype)
        if num_g_groups == 1:
            reduce_tmp = T.alloc_ub(1, calc_dtype)  # dummy for fused path
        else:
            reduce_tmp = T.alloc_ub((VECTOR_BASEG, BLOCK_N), calc_dtype)
        return w_raw, weight, weight_2d, mm_res, reduce_tmp

    _q_tot = q_t_size if q_t_size is not None else (B * S1)
    _k_tot = k_t_size if k_t_size is not None else (B * S2)
    # TND key: use 4D [1, k_tot, N2, D] to work around 3D indexing compiler issue.
    # Host wrapper unsqueezes the 3D flat key to 4D. Indexing: Key[0, offset, ...].
    _key_shape = (max_block_num, block_size, N2, D) if is_pa else (1, _k_tot, N2, D) if is_tnd_key else (B, S2, N2, D)
    _bt_shape = (B, max_block_num) if is_pa else (1, 1)
    # TND Q/W/O: 3D flat [t, ...]. N1=64 时复合索引 codegen crash, wrapper fallback BSND.
    _q_shape_q = (_q_tot, N1, D) if is_tnd else (B, S1, N1, D)
    _q_shape_w = (_q_tot, N1) if is_tnd else (B, S1, N1)
    _q_shape_o = (_q_tot, N2, TOP_K) if is_tnd else (B, S1, N2, TOP_K)

    @jit(
        out_idx=[6, 7],
        workspace_idx=[3, 8],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,  # Expert mode: manual sync
            # AUTO_SYNC=False: all barriers are manually written
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
                # ===== Cube buffers (Q: 2 L1 buffers, K: 3 L1 buffers) =====
                q_l1 = T.alloc_L1((_q_bufs, S1_BLOCK, M_SUB_TILES, BLOCK_M_CUBE, BLOCK_K), input_dtype)
                k_l1 = T.alloc_L1((_k_bufs, BLOCK_N, BLOCK_K), input_dtype)
                a_l0 = T.alloc_L0A((2, BLOCK_M_CUBE, BLOCK_K), input_dtype)
                # L0B must be (K, N) for T.mma (ref: example_gemm_transpose_l1).
                # Was (BLOCK_N//2, BLOCK_K)=(N,K) — silent when BLOCK_N//2==BLOCK_K (non-PA),
                # but PA block_size<256 makes BLOCK_N//2 != BLOCK_K → T.gemm K mismatch crash.
                b_l0 = T.alloc_L0B((2, BLOCK_K, BLOCK_N // 2), input_dtype)
                acc_l0c = T.alloc_L0C((2, BLOCK_M_CUBE, BLOCK_N // 2), calc_dtype)

                # ===== Vector: G-reduce buffers (conditional 1× vs 2× ping-pong) =====
                reduce_g_ub = T.alloc_ub(_BLOCK_N_VEC, calc_dtype)
                (w_raw_ub, weight_ub, weight_2d_ub, mm_res_ub, reduce_tmp_ub) = _alloc_g_reduce_bufs()

                # ===== Vector: Per-S1-row buffers =====
                (topk_a_ub, cache_tmp_ub, cache_slot0_ub, cache_slot1_ub, cache_slot2_ub) = _alloc_topk_bufs()

                # merged_ub size (2*_TA2) matches the copy stride
                merged_ub = T.alloc_ub(2 * _TA2, calc_dtype)
                p2_acc_ub = T.alloc_ub(_TA2, calc_dtype)
                stride2_blk_ub = T.alloc_ub(_KP2, calc_dtype)
                # pad-mode: temp buffer for reduce_sum output before copy to padded reduce_g_ub
                reduce_sum_tmp_ub = T.alloc_ub(BLOCK_N if _NEED_VEC_PAD else 1, calc_dtype)

                index_blk_ub = T.alloc_ub(_BLOCK_N_VEC, calc_dtype)
                mask_blk_ub = T.alloc_ub(_BLOCK_N_VEC // 8, "uint8")

                topk_index_ub = T.alloc_ub(TOP_K_ALIGNED, calc_dtype)
                output_ub = T.alloc_ub(TOP_K_ALIGNED, "int32")
                output_val_ub = T.alloc_ub(TOP_K_ALIGNED, input_dtype)
                score_topk_ub = T.alloc_ub(TOP_K_ALIGNED, calc_dtype)
                mask_topk_ub = T.alloc_ub(TOP_K_ALIGNED // 8, "uint8")

                prev_bsn_ub = T.alloc_ub(1, "int32")

                T.annotate_address(
                    {
                        q_l1: _q_l1_addr,
                        k_l1: _k_l1_addr,
                        acc_l0c: _acc_l0c_addr,
                    }
                )

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

                    for bsn_off in range(bsn_groups):
                        group_task_id = cid * tasks_per_core + bsn_off * s2_per_group
                        safe_gid = T.if_then_else(group_task_id < block_num, group_task_id, 0)
                        n2_idx = (safe_gid // num_s2_blocks) % N2
                        s1_blk_idx = (safe_gid // (num_s2_blocks * N2)) % s1_blocks
                        b_idx = safe_gid // (num_s2_blocks * N2 * s1_blocks)
                        s1_start = s1_blk_idx * S1_BLOCK
                        _q_len = actual_q_len[b_idx]
                        bsn_idx = group_task_id // num_s2_blocks
                        q_slot = bsn_idx % 2

                        # Q reload (2-buffer, q_slot = bsn_idx%2): first group of core
                        # (bsn_off==0) OR BSN boundary. Q shared within a BSN.
                        _q_need = (bsn_off == 0) | (group_task_id % num_s2_blocks == 0)
                        if _q_need:
                            T.wait_flag("mte1", "mte2", 40 + q_slot)
                            for s1_local in range(S1_BLOCK):
                                s1_idx = T.if_then_else(s1_start + s1_local < _q_len, s1_start + s1_local, _q_len - 1)
                                for g_sub in range(M_SUB_TILES):
                                    g_start = g_sub * BLOCK_M_CUBE
                                    g_end = g_start + BLOCK_M_CUBE
                                    if g_end > G:
                                        g_end = G
                                    if is_tnd:
                                        T.copy(
                                            Query[QOffset[b_idx] + s1_idx, n2_idx * G + g_start : n2_idx * G + g_end, 0:D],
                                            q_l1[q_slot, s1_local, g_sub, : (g_end - g_start), :],
                                        )
                                    else:
                                        T.copy(
                                            Query[b_idx, s1_idx, n2_idx * G + g_start : n2_idx * G + g_end, 0:D],
                                            q_l1[q_slot, s1_local, g_sub, : (g_end - g_start), :],
                                        )
                            T.set_flag("mte2", "mte1", q_slot)

                        for s2_local in T.serial(s2_per_group):
                            seq = bsn_off * s2_per_group + s2_local
                            k_slot = seq % 3
                            pp = seq % pp_slots
                            cur_task = group_task_id + s2_local
                            s2_blk = (safe_gid + s2_local) % num_s2_blocks
                            s2_start = s2_blk * BLOCK_N

                            # 3-slot K pipeline: K[seq] load waits
                            # slot freed by mma[seq-3], so K-load ‖ prev 2 iters' mma.
                            T.wait_flag("mte1", "mte2", 20 + k_slot)
                            if is_pa:
                                for sub in range(_BLOCKS_PER_TILE):
                                    T.copy(
                                        Key[BlockTable[b_idx, s2_blk * _BLOCKS_PER_TILE + sub], 0:block_size, n2_idx, 0:D],
                                        k_l1[k_slot, sub * block_size : (sub + 1) * block_size, :],
                                    )
                            elif is_tnd_key:
                                T.copy(
                                    Key[0, KOffset[b_idx] + s2_start : KOffset[b_idx] + s2_start + BLOCK_N, n2_idx, 0:D], k_l1[k_slot, :, :]
                                )
                            else:
                                T.copy(Key[b_idx, s2_start : s2_start + BLOCK_N, n2_idx, 0:D], k_l1[k_slot, :, :])
                            T.set_flag("mte2", "mte1", 10 + k_slot)

                            # K load issued before wait_cross -> overlaps V's prev read
                            # Counting semaphore (syncV1C1): wait for vector to free a pp slot
                            T.wait_cross_flag(SYNC_V1C1)
                            if _q_need and s2_local == 0:
                                T.wait_flag("mte2", "mte1", q_slot)
                            T.wait_flag("mte2", "mte1", 10 + k_slot)

                            if cur_task < block_num:
                                for s1_local in range(S1_BLOCK):
                                    for n_l0 in range(2):
                                        side = (s1_local * 2 + n_l0) % 2
                                        _nlo = n_l0 * (BLOCK_N // 2)
                                        _nhi = _nlo + (BLOCK_N // 2)
                                        T.wait_flag("fix", "m", 2 + side)
                                        T.wait_flag("m", "mte1", 30 + side)
                                        T.copy(q_l1[q_slot, s1_local, 0, :, :], a_l0[side, :, :])
                                        T.copy(k_l1[k_slot, _nlo:_nhi, :], b_l0[side, :, :], transpose=True)
                                        T.set_flag("mte1", "m", 30 + side)
                                        T.wait_flag("mte1", "m", 30 + side)
                                        T.mma(a_l0[side, :, :], b_l0[side, :, :], acc_l0c[side, :, :], init=True)
                                        T.set_flag("m", "mte1", 30 + side)
                                        T.set_flag("m", "fix", 2 + side)
                                        T.wait_flag("m", "fix", 2 + side)
                                        T.copy(
                                            acc_l0c[side, :, :],
                                            QK_Workspace[cid, pp, s1_local * BLOCK_M_CUBE : (s1_local + 1) * BLOCK_M_CUBE, _nlo:_nhi],
                                            enable_relu=True,
                                        )
                                        T.set_flag("fix", "m", 2 + side)
                            # release K slot after all L0B copies done (outside `if` for pad tasks)
                            T.set_flag("mte1", "mte2", 20 + k_slot)

                            T.set_cross_flag("FIX", SYNC_C1V1)

                        # Q-free at BSN end: free current BSN's Q slot after its last mma.
                        # _bsn_end = last bsn_off of core OR next bsn_off crosses BSN boundary.
                        _next_gid = cid * tasks_per_core + (bsn_off + 1) * s2_per_group
                        _bsn_end = (bsn_off == bsn_groups - 1) | (_next_gid % num_s2_blocks == 0)
                        if _bsn_end:
                            T.set_flag("mte1", "mte2", 40 + q_slot)

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
                    # Counting semaphore init: pre-set SYNC_V1C1 pp_slots times
                    # (syncV1C1 depth=pp_slots, cube can be pp_slots ahead before blocking)
                    for _ in range(pp_slots):
                        T.set_cross_flag("V", SYNC_V1C1)

                    T.tile.fill(stride2_blk_ub, 0)
                    T.pipe_barrier("V")
                    for _i in range(_K_PER_BLOCK):
                        stride2_blk_ub[_i * 2 + 1] = T.cast(1, calc_dtype)

                    T.tile.arith_progression(index_blk_ub, T.cast(0, calc_dtype), T.cast(1, calc_dtype), _BLOCK_N_VEC)

                    prev_bsn_ub[0] = -1
                    T.tile.fill(topk_a_ub, -T.infinity(calc_dtype))

                    # =========================================================
                    # Phase 1-V Main Loop (deferred merge: _sc-based, no counter)
                    # =========================================================
                    for bsn_off in range(bsn_groups):
                        group_task_id = cid * tasks_per_core + bsn_off * s2_per_group
                        safe_gid = T.if_then_else(group_task_id < block_num, group_task_id, 0)
                        n2_idx = (safe_gid // num_s2_blocks) % N2
                        s1_blk_idx = (safe_gid // (num_s2_blocks * N2)) % s1_blocks
                        b_idx = safe_gid // (num_s2_blocks * N2 * s1_blocks)
                        s1_start = s1_blk_idx * S1_BLOCK

                        # Per-batch lengths (actual_q/actual_k are per-batch len)
                        _q_len_b = actual_q_len[b_idx]
                        _k_len_b = actual_k_len[b_idx]

                        # ---- P1a: BSN tracking hoisted to bsn_off level ----
                        # Within a single bsn_off group, BSN does not change
                        # (s2_per_group < num_s2_blocks, or forced to 1 at
                        # BSN boundaries by the alignment guard above).
                        blk_bsn = group_task_id // num_s2_blocks
                        if prev_bsn_ub[0] >= 0 and blk_bsn != prev_bsn_ub[0]:
                            # BSN change: cached slots already merged at prev loop end → flush topk_a
                            T.pipe_barrier("V")
                            for _si in range(VID_S1):
                                T.copy(
                                    topk_a_ub[_si, :],
                                    TopK_Workspace[
                                        cid, prev_bsn_ub[0] - (cid * tasks_per_core) // num_s2_blocks, vid * VID_S1 + _si, 0:_TA2
                                    ],
                                )
                            T.set_flag("MTE3", "V", 5)
                            T.wait_flag("MTE3", "V", 5)
                            T.tile.fill(topk_a_ub, -T.infinity(calc_dtype))
                            # Clear cache_slots on BSN change so prev BSN's
                            # valid-score slots don't leak into next BSN's 4-way merge
                            # (large-S1 multi-BSN: tl had extra s2 indices from prev BSN).
                            for _si in range(VID_S1):
                                T.tile.fill(cache_slot0_ub[_si, :], -T.infinity(calc_dtype))
                                T.tile.fill(cache_slot1_ub[_si, :], -T.infinity(calc_dtype))
                                T.tile.fill(cache_slot2_ub[_si, :], -T.infinity(calc_dtype))
                        T.pipe_barrier("V")
                        prev_bsn_ub[0] = blk_bsn

                        for s2_local in T.serial(s2_per_group):
                            pp = (bsn_off * s2_per_group + s2_local) % pp_slots
                            T.wait_cross_flag(SYNC_C1V1)

                            s2_blk = (safe_gid + s2_local) % num_s2_blocks
                            s2_start = s2_blk * BLOCK_N

                            if group_task_id + s2_local < block_num:
                                # Slot idx (global form — stable, no race)
                                _sc = (bsn_off * s2_per_group + s2_local) % 3

                                # ---- Process each S1 row (vid-aware) ----
                                for s1_local in range(VID_S1):
                                    actual_s1 = vid * VID_S1 + s1_local
                                    if s1_start + actual_s1 < _q_len_b:
                                        s1_idx = s1_start + actual_s1

                                        s2_valid = _k_len_b
                                        causal_limit = _k_len_b - _q_len_b + s1_idx + 1
                                        _mode_val = T.cast(sparse_mode, "int32")
                                        _is_causal = _mode_val == T.cast(3, "int32")
                                        s2_valid = T.if_then_else(_is_causal & (causal_limit > 0), causal_limit, s2_valid)

                                        if s2_start < s2_valid:
                                            # ============================================
                                            # G-dim Reduce (v16: Pipelined for num_g_groups>1)
                                            # ============================================
                                            if num_g_groups == 1:
                                                # Batch both DMAs for MTE2 overlap
                                                qk_g_start = actual_s1 * G_padded
                                                if is_tnd:
                                                    _w_row = QOffset[b_idx] + s1_idx
                                                    T.copy(Weights[_w_row, n2_idx * G : n2_idx * G + VECTOR_BASEG], w_raw_ub)
                                                else:
                                                    T.copy(Weights[b_idx, s1_idx, n2_idx * G : n2_idx * G + VECTOR_BASEG], w_raw_ub)
                                                T.copy(QK_Workspace[cid, pp, qk_g_start : qk_g_start + VECTOR_BASEG, 0:BLOCK_N], mm_res_ub)
                                                # Manual sync: MTE2→V
                                                T.set_flag("MTE2", "V", 6)
                                                T.wait_flag("MTE2", "V", 6)
                                                T.tile.cast(weight_ub, w_raw_ub, "CAST_NONE", VECTOR_BASEG)
                                                T.pipe_barrier("V")  # cast → broadcast (RAW on weight_ub)
                                                T.tile.broadcast(weight_2d_ub, weight_ub)
                                                T.pipe_barrier("V")  # broadcast → mul (RAW on weight_2d_ub)
                                                T.tile.mul(mm_res_ub, mm_res_ub, weight_2d_ub)
                                                T.pipe_barrier("V")  # mul → reduce_sum
                                                if _NEED_VEC_PAD:
                                                    T.reduce_sum(mm_res_ub, reduce_sum_tmp_ub, 0)
                                                    T.pipe_barrier("V")
                                                    T.copy(reduce_sum_tmp_ub, reduce_g_ub[0:BLOCK_N])
                                                    T.pipe_barrier("V")
                                                    for _ti in range(BLOCK_N, _BLOCK_N_VEC):
                                                        reduce_g_ub[_ti] = -T.infinity(calc_dtype)
                                                else:
                                                    T.reduce_sum(mm_res_ub, reduce_g_ub, 0)
                                            else:
                                                # Unfused path with DMA batch per g_id
                                                T.tile.fill(reduce_tmp_ub, 0)
                                                for g_id in range(num_g_groups):
                                                    qk_g_start = actual_s1 * G_padded + g_id * VECTOR_BASEG
                                                    _wg = g_id * VECTOR_BASEG
                                                    if is_tnd:
                                                        _w_row = QOffset[b_idx] + s1_idx
                                                        T.copy(
                                                            Weights[_w_row, n2_idx * G + _wg : n2_idx * G + _wg + VECTOR_BASEG], w_raw_ub
                                                        )
                                                    else:
                                                        T.copy(
                                                            Weights[b_idx, s1_idx, n2_idx * G + _wg : n2_idx * G + _wg + VECTOR_BASEG],
                                                            w_raw_ub,
                                                        )
                                                    T.copy(
                                                        QK_Workspace[cid, pp, qk_g_start : qk_g_start + VECTOR_BASEG, 0:BLOCK_N], mm_res_ub
                                                    )
                                                    # Manual sync: MTE2→V
                                                    T.set_flag("MTE2", "V", 6)
                                                    T.wait_flag("MTE2", "V", 6)
                                                    T.tile.cast(weight_ub, w_raw_ub, "CAST_NONE", VECTOR_BASEG)
                                                    T.pipe_barrier("V")  # cast → broadcast
                                                    T.tile.broadcast(weight_2d_ub, weight_ub)
                                                    T.pipe_barrier("V")  # broadcast → mul_add_dst
                                                    T.tile.mul_add_dst(reduce_tmp_ub, mm_res_ub, weight_2d_ub)
                                                    # Manual sync: V→MTE2 releases buffer for next g_id DMA
                                                    T.set_flag("V", "MTE2", 7)
                                                    T.wait_flag("V", "MTE2", 7)
                                                T.pipe_barrier("V")  # end of g_id loop → reduce_sum
                                                if _NEED_VEC_PAD:
                                                    T.reduce_sum(reduce_tmp_ub, reduce_sum_tmp_ub, 0)
                                                    T.pipe_barrier("V")
                                                    T.copy(reduce_sum_tmp_ub, reduce_g_ub[0:BLOCK_N])
                                                    T.pipe_barrier("V")
                                                    for _ti in range(BLOCK_N, _BLOCK_N_VEC):
                                                        reduce_g_ub[_ti] = -T.infinity(calc_dtype)
                                                else:
                                                    T.reduce_sum(reduce_tmp_ub, reduce_g_ub, 0)

                                            # --- Mask invalid positions ---
                                            # Stage2: removed reduce_sum→compare barrier (no RAW: reduce_sum
                                            # writes reduce_g, compare reads index_blk). V-pipe in-order issues
                                            # compare after reduce_sum; compare→select barrier below still
                                            # ensures reduce_sum done before select reads reduce_g.
                                            limit = s2_valid - s2_start
                                            T.tile.compare(mask_blk_ub, index_blk_ub, limit, "LT")
                                            T.pipe_barrier("V")  # compare → select (RAW on mask_blk_ub)
                                            T.tile.select(
                                                reduce_g_ub, mask_blk_ub, reduce_g_ub, -T.infinity(calc_dtype), "VSEL_TENSOR_SCALAR_MODE"
                                            )

                                            # --- Deferred merge: topk+axpy → cache slot (_sc approach) ---
                                            T.pipe_barrier("V")  # select → sort (sort vs topk: +2% precision, K=N equivalent)
                                            T.tile.sort(cache_tmp_ub, reduce_g_ub, _BLOCK_N_VEC)
                                            T.pipe_barrier("V")  # sort → axpy
                                            T.tile.axpy(cache_tmp_ub, stride2_blk_ub, T.cast(s2_start, calc_dtype))
                                            T.pipe_barrier("V")  # axpy → copy to slot
                                            if _sc == 0:
                                                T.copy(cache_tmp_ub, cache_slot0_ub[s1_local, :])
                                            elif _sc == 1:
                                                T.copy(cache_tmp_ub, cache_slot1_ub[s1_local, :])
                                            else:
                                                T.copy(cache_tmp_ub, cache_slot2_ub[s1_local, :])
                                            # 4-way merge when 3 slots full (per-row, inside s1_local loop)
                                            if _sc == 2:
                                                T.pipe_barrier("V")
                                                T.tile.merge_sort(
                                                    merged_ub,
                                                    topk_a_ub[s1_local, :],
                                                    cache_slot0_ub[s1_local, :],
                                                    cache_slot1_ub[s1_local, :],
                                                    cache_slot2_ub[s1_local, :],
                                                )
                                                T.pipe_barrier("V")
                                                T.copy(merged_ub[0:_TA2], topk_a_ub[s1_local, :])
                                            # Tail merge at last s2_local of last group (inside if s2_start<s2_valid)
                                            # BSN-local _sc fixes slot overwrite within a BSN.
                                            # Note: cross-BSN last-block merge not done (TileLang TIR-or crashes);
                                            #       ~3% precision loss on multi-BSN cross-core cases only.
                                            if s2_local == s2_per_group - 1 and bsn_off == bsn_groups - 1:
                                                if _sc == 1:
                                                    T.pipe_barrier("V")
                                                    T.tile.merge_sort(merged_ub, topk_a_ub[s1_local, :], cache_slot0_ub[s1_local, :])
                                                    T.pipe_barrier("V")
                                                    T.copy(merged_ub[0:_TA2], topk_a_ub[s1_local, :])
                                                    T.tile.merge_sort(merged_ub, topk_a_ub[s1_local, :], cache_slot1_ub[s1_local, :])
                                                    T.pipe_barrier("V")
                                                    T.copy(merged_ub[0:_TA2], topk_a_ub[s1_local, :])
                                                elif _sc == 0:
                                                    T.pipe_barrier("V")
                                                    T.tile.merge_sort(merged_ub, topk_a_ub[s1_local, :], cache_slot0_ub[s1_local, :])
                                                    T.pipe_barrier("V")
                                                    T.copy(merged_ub[0:_TA2], topk_a_ub[s1_local, :])
                                        else:
                                            # Skip block (s2_start>=s2_valid=act_k), fill this slot
                                            # -inf so 4-way merge doesn't pick UB garbage.
                                            # Also trigger 4-way merge and tail merge for skip blocks,
                                            # otherwise valid data in earlier slots is never merged into topk_a
                                            # (act_k tiny → most blocks skip → merge never fires → topk_a stays -inf).
                                            if _sc == 0:
                                                T.tile.fill(cache_slot0_ub[s1_local, :], -T.infinity(calc_dtype))
                                            elif _sc == 1:
                                                T.tile.fill(cache_slot1_ub[s1_local, :], -T.infinity(calc_dtype))
                                            else:
                                                T.tile.fill(cache_slot2_ub[s1_local, :], -T.infinity(calc_dtype))
                                            # 4-way merge when 3 slots full (same as valid block path)
                                            if _sc == 2:
                                                T.pipe_barrier("V")
                                                T.tile.merge_sort(
                                                    merged_ub,
                                                    topk_a_ub[s1_local, :],
                                                    cache_slot0_ub[s1_local, :],
                                                    cache_slot1_ub[s1_local, :],
                                                    cache_slot2_ub[s1_local, :],
                                                )
                                                T.pipe_barrier("V")
                                                T.copy(merged_ub[0:_TA2], topk_a_ub[s1_local, :])
                                            # Tail merge for skip blocks at last iteration
                                            if s2_local == s2_per_group - 1 and bsn_off == bsn_groups - 1:
                                                if _sc == 1:
                                                    T.pipe_barrier("V")
                                                    T.tile.merge_sort(merged_ub, topk_a_ub[s1_local, :], cache_slot0_ub[s1_local, :])
                                                    T.pipe_barrier("V")
                                                    T.copy(merged_ub[0:_TA2], topk_a_ub[s1_local, :])
                                                    T.tile.merge_sort(merged_ub, topk_a_ub[s1_local, :], cache_slot1_ub[s1_local, :])
                                                    T.pipe_barrier("V")
                                                    T.copy(merged_ub[0:_TA2], topk_a_ub[s1_local, :])
                                                elif _sc == 0:
                                                    T.pipe_barrier("V")
                                                    T.tile.merge_sort(merged_ub, topk_a_ub[s1_local, :], cache_slot0_ub[s1_local, :])
                                                    T.pipe_barrier("V")
                                                    T.copy(merged_ub[0:_TA2], topk_a_ub[s1_local, :])
                            T.pipe_barrier("V")
                            T.set_cross_flag("V", SYNC_V1C1)

                    # Final flush: topk_a → workspace (tail merge already done per-group in loop)
                    T.pipe_barrier("V")
                    if prev_bsn_ub[0] >= 0:
                        T.pipe_barrier("V")
                        for s1_local in range(VID_S1):
                            T.copy(
                                topk_a_ub[s1_local, :],
                                TopK_Workspace[
                                    cid, prev_bsn_ub[0] - (cid * tasks_per_core) // num_s2_blocks, vid * VID_S1 + s1_local, 0:_TA2
                                ],
                            )

                    T.pipe_barrier("V")
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
                            bsn_first_task_p2 = out_bsn * BSN_TASK_SPAN
                            bsn_last_task_p2 = (out_bsn + 1) * BSN_TASK_SPAN - 1
                            bsn_first_cid_p2 = bsn_first_task_p2 // tasks_per_core
                            bsn_last_cid_p2 = bsn_last_task_p2 // tasks_per_core
                            if bsn_last_cid_p2 >= core_num:
                                bsn_last_cid_p2 = core_num - 1
                            bsn_local_first = out_bsn - (bsn_first_cid_p2 * tasks_per_core) // num_s2_blocks
                            T.copy(TopK_Workspace[bsn_first_cid_p2, bsn_local_first, out_s1_local, 0:_TA2], p2_acc_ub)
                            # MTE2 copy → V read needs explicit sync (auto_sync=False).
                            # Without this, V reads p2_acc before MTE2 done → garbage idx (233643).
                            T.set_flag("MTE2", "V", 3)
                            T.wait_flag("MTE2", "V", 3)
                            if NEED_CROSS_CORE:
                                num_merge_p2 = bsn_last_cid_p2 - bsn_first_cid_p2
                                for m in T.serial(num_merge_p2):
                                    other_cid = bsn_first_cid_p2 + 1 + m
                                    bsn_local_other = out_bsn - (other_cid * tasks_per_core) // num_s2_blocks
                                    T.barrier_all()
                                    T.copy(TopK_Workspace[other_cid, bsn_local_other, out_s1_local, 0:_TA2], topk_a_ub[0, :])
                                    T.set_flag("MTE2", "V", 4)
                                    T.wait_flag("MTE2", "V", 4)
                                    T.tile.merge_sort(merged_ub, p2_acc_ub, topk_a_ub[0, :])
                                    T.pipe_barrier("V")
                                    T.copy(merged_ub[0:_TA2], p2_acc_ub)
                            # Output: extract indices
                            T.tile.gather_mask(topk_index_ub, p2_acc_ub, "P1010")
                            T.tile.fill(score_topk_ub, 0)
                            T.pipe_barrier("V")
                            T.tile.gather_mask(score_topk_ub, p2_acc_ub, "P0101")
                            T.pipe_barrier("V")
                            T.tile.compare(mask_topk_ub, score_topk_ub, T.cast(-1e30, calc_dtype), "GT")
                            T.pipe_barrier("V")
                            T.tile.select(topk_index_ub, mask_topk_ub, topk_index_ub, T.cast(-1.0, calc_dtype), "VSEL_TENSOR_SCALAR_MODE")
                            T.pipe_barrier("V")
                            T.tile.cast(output_ub, topk_index_ub, "CAST_ROUND", TOP_K)
                            T.barrier_all()
                            if is_tnd:
                                _o_row_p2 = _q_off_p2 + s1_idx_p2
                                T.copy(output_ub, Out[_o_row_p2, n2_idx_p2, 0:TOP_K])
                            else:
                                T.copy(output_ub, Out[b_idx_p2, s1_idx_p2, n2_idx_p2, 0:TOP_K])
                            if return_value:
                                T.set_flag("MTE3", "V", 15)
                                T.wait_flag("MTE3", "V", 15)
                                T.tile.gather_mask(topk_index_ub, p2_acc_ub, "P0101")
                                T.pipe_barrier("V")
                                T.tile.cast(output_val_ub, topk_index_ub, "CAST_RINT", TOP_K)
                                T.set_flag("V", "MTE3", 5)
                                T.wait_flag("V", "MTE3", 5)
                                if is_tnd:
                                    _o_row_p2 = _q_off_p2 + s1_idx_p2
                                    T.copy(output_val_ub, OutVal[_o_row_p2, n2_idx_p2, 0:TOP_K])
                                else:
                                    T.copy(output_val_ub, OutVal[b_idx_p2, s1_idx_p2, n2_idx_p2, 0:TOP_K])
                        else:
                            # s1 >= act_q or out_s1_local >= S1 — fill -1
                            # Without this, Out tensor retains uninitialized memory for
                            # skipped rows (e.g. b=0 s1=2 when act_q[0]=2).
                            T.tile.fill(output_ub, -1)
                            T.pipe_barrier("V")
                            if is_tnd:
                                _o_row_p2 = _q_off_p2 + s1_idx_p2
                                T.copy(output_ub, Out[_o_row_p2, n2_idx_p2, 0:TOP_K])
                            else:
                                T.copy(output_ub, Out[b_idx_p2, s1_idx_p2, n2_idx_p2, 0:TOP_K])

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
    pp_slots: int = 2,
    max_cores: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert layout_query in ("BSND", "TND"), f"Unsupported query layout: {layout_query}"
    is_pa = layout_key == "PA_BSND"
    is_tnd = layout_query == "TND"
    is_tnd_key = layout_key == "TND"

    # PA_BSND does not expose top-K scores (return_value=True is rejected).
    # The op always returns a 2-tuple (indices, values); for PA or return_value=False the
    # values tensor is an empty placeholder.
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
        # 保证 num_s2_blocks >= 2 (单 s2 块触发 codegen 执行期 crash, S2=256→1块crash, S2=257→2块OK)
        S2 = max(S2, 257)
        _block_size, max_block_num = 128, 1
    else:
        _, S2, N2_k, _ = key.shape
        _block_size, max_block_num = 128, 1

    assert D == 128
    input_dtype = "float16" if query.dtype == torch.float16 else "bfloat16"

    if actual_seq_lengths_query is None:
        actual_seq_lengths_query = torch.full((B,), S1, dtype=torch.int32, device=query.device)
    if actual_seq_lengths_key is None:
        actual_seq_lengths_key = torch.full((B,), S2, dtype=torch.int32, device=query.device)

    # 对齐 CPU golden (cal_atten_bnsd): max(act_q)==0 或 max(act_k)==0 时,
    # 所有 batch 无有效输出 (curr_q=0 pass / actual_selected_count=min(0,K)=0),
    # 直接返回全 -1 indices + -inf values, 避免 S2=0 退化参数触发 codegen crash.
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

    # TND key: pad tail so Cube DMA [KOffset[b]:KOffset[b]+BLOCK_N] never reads
    # past the flat buffer.  _s2_eff = max(act_k, S2) because S2 may be bumped to >=257
    # to avoid single-s2-block codegen issue.  V-scope masks s2>=actual_k_len.
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
        # Reshape flat [k_tot, N2, D] to 4D [1, k_tot, N2, D] for kernel compatibility
        key = key.unsqueeze(0)
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
    from lightning_indexer_golden import cpu_lightning_indexer

    _DEV = 2
    for d in [2, 6, 7, 3, 1, 5]:
        try:
            torch_npu.npu.set_device(d)
            torch.zeros(1, device=f"npu:{d}")
            torch.npu.synchronize()
            _DEV = d
            break
        except Exception:
            pass
    torch_npu.npu.set_device(_DEV)
    torch.set_default_device(f"npu:{_DEV}")
    print(f"Using device {_DEV}: {torch.npu.get_device_name(_DEV)}\n")
    tilelang.disable_cache()

    def _make_rand(shape, dtype=torch.float16):
        return torch.tensor(np.random.uniform(-1, 1, shape), dtype=dtype)

    def _make_rand_w(shape, dtype=torch.float16):
        return _make_rand(shape, dtype).abs()

    def _validate(ref_idx, tl_idx):
        r = set(int(x) for x in ref_idx.cpu().flatten()[ref_idx.cpu().flatten() >= 0].tolist())
        t = set(int(x) for x in tl_idx.cpu().flatten()[tl_idx.cpu().flatten() >= 0].tolist())
        v = max(len(r), len(t))
        return (len(r & t) / v * 100) if v > 0 else 100.0

    def _validate_values(g_idx, g_val, tl_idx, tl_val):
        """Per-row idx->score match. Returns (match%, max_rel_diff).
        Scores are fp16 on both sides; compare as fp32. A score agrees when
        |g-t| <= 2% of |g| (fp16 rounding + float32 accum-order ULPs)."""
        gi = g_idx.cpu().reshape(-1, g_idx.shape[-1])
        gv = g_val.cpu().to(torch.float32).reshape(-1, g_val.shape[-1])
        ti = tl_idx.cpu().reshape(-1, tl_idx.shape[-1])
        tv = tl_val.cpu().to(torch.float32).reshape(-1, tl_val.shape[-1])
        matched = total = 0
        max_rel = 0.0
        for r in range(gi.shape[0]):
            gmap = {int(x): float(y) for x, y in zip(gi[r].tolist(), gv[r].tolist()) if int(x) >= 0}
            tmap = {int(x): float(y) for x, y in zip(ti[r].tolist(), tv[r].tolist()) if int(x) >= 0}
            for idx, sc in gmap.items():
                if idx in tmap:
                    total += 1
                    rel = abs(sc - tmap[idx]) / abs(sc) if abs(sc) > 1e-6 else abs(sc - tmap[idx])
                    max_rel = max(max_rel, rel)
                    if rel <= 2e-2:
                        matched += 1
        pct = (matched / total * 100) if total > 0 else 100.0
        return pct, max_rel, total

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
        g_idx, g_val = cpu_lightning_indexer(
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
        if is_pa:
            # PA does not expose return_value — second output is an empty placeholder
            tl_idx, _ = lightning_indexer(
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
                return_value=False,
            )
        else:
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
                return_value=True,
            )
        torch.npu.synchronize()
        idx_pct = _validate(g_idx, tl_idx)
        ok = idx_pct >= 95.0
        detail = f"index={idx_pct:.2f}%"
        if not is_pa:
            vpct, _, _ = _validate_values(g_idx, g_val, tl_idx, tl_val)
            ok = ok and vpct >= 95.0
            detail += f", value={vpct:.2f}%"
        print(f"[{name}]")
        print("  Kernel Output Match!" if ok else f"  MISMATCH ({detail})")

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
