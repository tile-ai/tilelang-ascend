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

SUPPORTED_BLOCK_N = (64, 128, 256, 512)
SUPPORTED_DIMENSIONS = (64, 128)
TOP_K_LIMIT_BY_BLOCK_N = {64: 2048, 128: 2048, 256: 1920, 512: 1536}
MAX_EXACT_FP32_INDEX = 2**24


def _require_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def validate_indexer_config(
    n2,
    groups,
    dimension,
    top_k,
    vector_basen,
    vector_baseg,
    block_m,
    block_n,
    block_k,
    max_s2,
    input_dtype="float16",
    calc_dtype="float",
    s2_splits=1,
    max_cores=None,
    *,
    batch=None,
    s1=None,
    s2=None,
):
    """Validate compile-time parameters and optional runtime shape values."""
    integer_parameters = {
        "N2": n2,
        "G": groups,
        "D": dimension,
        "TOP_K": top_k,
        "VECTOR_BASEN": vector_basen,
        "VECTOR_BASEG": vector_baseg,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "MAX_S2": max_s2,
        "S2_SPLITS": s2_splits,
    }
    if max_cores is not None:
        integer_parameters["max_cores"] = max_cores
    if batch is not None:
        integer_parameters["B"] = batch
    if s1 is not None:
        integer_parameters["S1"] = s1
    if s2 is not None:
        integer_parameters["S2"] = s2
    for name, value in integer_parameters.items():
        _require_positive_int(name, value)

    if input_dtype != "float16":
        raise ValueError(f"input_dtype must be float16, got {input_dtype!r}")
    if calc_dtype != "float":
        raise ValueError(f"calc_dtype must be float, got {calc_dtype!r}")
    if block_m != 64:
        raise ValueError(f"BLOCK_M must be 64, got {block_m}")
    if block_n not in SUPPORTED_BLOCK_N:
        raise ValueError(f"BLOCK_N must be one of {SUPPORTED_BLOCK_N}, got {block_n}")
    if dimension not in SUPPORTED_DIMENSIONS:
        raise ValueError(f"D must be one of {SUPPORTED_DIMENSIONS}, got {dimension}")
    if block_k != dimension:
        raise ValueError(f"BLOCK_K must equal D, got BLOCK_K={block_k}, D={dimension}")
    if groups % 8 or not 8 <= groups <= 248:
        raise ValueError(f"G must be a multiple of 8 in [8, 248], got {groups}")
    if groups % vector_baseg:
        raise ValueError(f"G={groups} must be divisible by VECTOR_BASEG={vector_baseg}")
    if vector_basen != block_n:
        raise ValueError(f"VECTOR_BASEN must equal BLOCK_N, got {vector_basen} != {block_n}")
    if max_s2 % block_n:
        raise ValueError(f"MAX_S2={max_s2} must be divisible by BLOCK_N={block_n}")
    if max_s2 > MAX_EXACT_FP32_INDEX:
        raise ValueError(f"MAX_S2 must not exceed the exact fp32 integer limit {MAX_EXACT_FP32_INDEX}, got {max_s2}")
    if top_k > max_s2:
        raise ValueError(f"TOP_K={top_k} must not exceed MAX_S2={max_s2}")
    top_k_limit = TOP_K_LIMIT_BY_BLOCK_N[block_n]
    if top_k > top_k_limit:
        raise ValueError(f"TOP_K must not exceed {top_k_limit} when BLOCK_N={block_n}, got {top_k}")

    if s1 is not None and s1 % block_m:
        raise ValueError(f"S1={s1} must be divisible by BLOCK_M={block_m}")
    if s2 is not None:
        if s2 > max_s2:
            raise ValueError(f"S2={s2} must not exceed MAX_S2={max_s2}")
        if s2 % (s2_splits * block_n):
            raise ValueError(f"S2={s2} must be divisible by S2_SPLITS*BLOCK_N={s2_splits * block_n}")
        if top_k > s2:
            raise ValueError(f"TOP_K={top_k} must not exceed S2={s2}")


def _get_cube_core_num() -> int:
    """Detect the Cube core count of the current device."""
    try:
        import torch_npu as _tnpu

        return _tnpu.npu.get_device_properties(0).cube_core_num
    except Exception:
        return 20


def auto_s2_splits(batch, s1, s2, block_n, max_cores=None):
    """Select the S2 split count using a host-side cost model.

    The cost approximates one trunk or one merge operation:
      cost(sp) = items(sp) * tps(sp) + rows_per_aiv(sp) * merge_calls
    The first term models the Phase 1 critical path. The second models Phase 2
    with rows distributed across all AIVs. With one split, each owning block
    performs Phase 2 independently without a global barrier.
    """
    for name, value in {"B": batch, "S1": s1, "S2": s2, "BLOCK_N": block_n}.items():
        _require_positive_int(name, value)
    if s1 % 64:
        raise ValueError(f"S1={s1} must be divisible by 64")
    if block_n not in SUPPORTED_BLOCK_N:
        raise ValueError(f"BLOCK_N must be one of {SUPPORTED_BLOCK_N}, got {block_n}")
    if s2 % block_n:
        raise ValueError(f"S2={s2} must be divisible by BLOCK_N={block_n}")
    if max_cores is None:
        max_cores = _get_cube_core_num()
    _require_positive_int("max_cores", max_cores)
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
        # With one split, every block is an owner and each AIV handles 32 rows.
        return 32

    def _cost(sp):
        barrier = 1.0 if sp > 1 else 0.0
        return _crit(sp) + _p2_rows(sp) * merge_calls + barrier

    # Use one split as the baseline. A candidate split must reduce the Phase 1
    # critical path by at least two units, reduce total cost including the
    # barrier, and keep the number of work items per block at four or fewer.
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
    if max_cores is None:
        max_cores = _get_cube_core_num()
    validate_indexer_config(
        N2,
        G,
        D,
        TOP_K,
        VECTOR_BASEN,
        VECTOR_BASEG,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        MAX_S2,
        input_dtype,
        calc_dtype,
        s2_splits,
        max_cores,
    )
    # BLOCK_N <= 256 uses one GEMM; BLOCK_N = 512 uses two 256-column GEMMs.
    DUAL_GEMM = BLOCK_N > 256
    GEMM_COLS = min(BLOCK_N, 256)
    SECOND_COLS = max(BLOCK_N - 256, 1)
    VID_ROWS = BLOCK_M // 2
    # Each trunk keeps only min(TOP_K, BLOCK_N) pairs because every global
    # top-K candidate must belong to the local top-K of its source trunk.
    TRUNK_KEEP = min(TOP_K, BLOCK_N)
    B = T.symbolic("B")
    S1 = T.symbolic("S1")
    S2 = T.symbolic("S2")
    S1_TILES = S1 // BLOCK_M
    TASK_COUNT = B * N2 * S1_TILES
    TRUNKS_MAX = MAX_S2 // BLOCK_N
    # Persistent kernel: cap the grid at the core count and process multiple
    # work items serially in each block.
    S2_SPLITS = s2_splits
    TOTAL_WORK = TASK_COUNT * S2_SPLITS
    GRID_SIZE = T.min(TOTAL_WORK, max_cores)
    ITEMS_PER_BLOCK = (TOTAL_WORK + GRID_SIZE - 1) // GRID_SIZE
    READY_FLAG = 0
    FREE_FLAG = 2
    L0C_FLAG = 1  # Avoid event ID 0, which is used internally by GEMM.
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
                # k_l1_b is an unused placeholder in single-GEMM mode.
                k_l1_a = T.alloc_L1((GEMM_COLS, BLOCK_K), input_dtype)
                k_l1_b = T.alloc_L1((SECOND_COLS, BLOCK_K), input_dtype)
                c_l0 = T.alloc_L0C((BLOCK_M, GEMM_COLS), calc_dtype)
                T.set_flag("FIX", "M", L0C_FLAG)
                T.set_flag("M", "MTE2", K_L1_FREE_FLAG)
                T.set_flag("M", "MTE2", Q_L1_FREE_FLAG)

                # Phase 1: process group-by-S2-split work items serially.
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
                            # Always load the first K segment. In dual-GEMM mode,
                            # the second load follows in the same MTE2 FIFO.
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
                                # GEMM A always covers columns [0, GEMM_COLS).
                                T.wait_flag("FIX", "M", L0C_FLAG)
                                T.gemm_v0(q_l1, k_l1_a, c_l0, transpose_B=True, init=True)
                                T.set_flag("M", "FIX", L0C_FLAG)
                                T.wait_flag("M", "FIX", L0C_FLAG)
                                T.copy(c_l0, QK_SLOT[cid, slot, :, g, 0:GEMM_COLS], enable_relu=True)
                                T.set_flag("FIX", "M", L0C_FLAG)
                                if DUAL_GEMM:
                                    # GEMM B covers the remaining columns and
                                    # releases Q only after it completes.
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
                    # Required before owners read results written by other blocks.
                    T.sync_all()

            with T.Scope("V"):
                # Phase 1 computes, sorts, and stores in trunk-major order.
                # Phase 2 keeps row-major history in UB for batched 4-way merges.
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

                # Both slots are free before the first Cube write.
                T.set_cross_flag("MTE2", FREE_FLAG)
                T.set_cross_flag("MTE2", FREE_FLAG + 1)
                T.tile.fill(index_lane_ub, 0)
                T.pipe_barrier("V")
                for index_offset in range(BLOCK_N):
                    index_lane_ub[index_offset * 2 + 1] = T.cast(1, calc_dtype)

                # Phase 1: serial work items in trunk-major order.
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
                            # Prefetch row 0 immediately while mm_res_ub is free,
                            # overlapping the copy with loop setup and flag work.
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
                            # Each of the two AIVs handles half of the rows.
                            for s1_local in T.serial(VID_ROWS):
                                s1_offset = vid * VID_ROWS + s1_local
                                s1_id = gm * BLOCK_M + s1_offset
                                T.wait_flag("MTE2", "V", G_REDUCE_FLAG)
                                # Apply a fused broadcast multiply to each
                                # 64-element fp32 column block. The temporary
                                # buffer uses only G * 8 elements instead of a
                                # full (G, VECTOR_BASEN) broadcast buffer.
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
                                # Prefetch the next row after issuing sort and
                                # axpy so MTE2 overlaps with vector execution
                                # and the current row's MTE3 store.
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
                                # Store only the valid sorted prefix. Phase 2
                                # fills the unused history tail with -inf.
                                T.set_flag("V", "MTE3", HISTORY_STORE_FLAG)
                                T.wait_flag("V", "MTE3", HISTORY_STORE_FLAG)
                                T.copy(
                                    sorted_block_ub[0, 0 : 2 * TRUNK_KEEP],
                                    SORTED_WORKSPACE[group, s1_offset, trunk_base + n, 0 : 2 * TRUNK_KEEP],
                                )
                                T.set_flag("MTE3", "V", HISTORY_STORE_FLAG)
                                T.wait_flag("MTE3", "V", HISTORY_STORE_FLAG)
                            # Drain the final row event so every trunk has
                            # balanced V-to-MTE2 set/wait operations and stale
                            # events cannot trigger the next trunk prematurely.
                            T.wait_flag("V", "MTE2", G_REDUCE_FLAG)
                            T.set_flag("V", "MTE2", SLOT_RELEASE_FLAG + slot)
                            T.wait_flag("V", "MTE2", SLOT_RELEASE_FLAG + slot)
                            T.set_cross_flag("MTE2", FREE_FLAG + slot)

                if S2_SPLITS > 1:
                    # Wait for all Phase 1 writes before cross-block reads.
                    T.sync_all()

                # ===== Phase 2 =====
                num_trunks = S2 // BLOCK_N
                num_batches = (num_trunks - 1 + 2) // 3
                if S2_SPLITS > 1:
                    # Distribute all group rows across every block and AIV
                    # after the barrier.
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
                            # Initialize history with -inf and load trunk 0.
                            T.tile.fill(history_ub, -T.infinity(calc_dtype))
                            T.set_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                            T.wait_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                            T.copy(SORTED_WORKSPACE[group, s1_offset, 0, 0 : 2 * TRUNK_KEEP], history_ub[0 : 2 * TRUNK_KEEP])
                            T.set_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                            T.wait_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                            # Merge history with up to three additional trunks.
                            for batch in T.serial(num_batches):
                                base = 1 + batch * 3
                                # Source 0 always exists for this batch.
                                T.tile.fill(sort_src0_ub, -T.infinity(calc_dtype))
                                T.set_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                T.wait_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                T.copy(
                                    SORTED_WORKSPACE[group, s1_offset, base, 0 : 2 * TRUNK_KEEP],
                                    sort_src0_ub[0 : 2 * TRUNK_KEEP],
                                )
                                T.set_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                T.wait_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                # Load source 1 only when present.
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
                                # Load source 2 only when present.
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
                                # Select the merge arity from the available sources.
                                if base + 2 < num_trunks:
                                    T.tile.merge_sort(merged_ub, history_ub, sort_src0_ub, sort_src1_ub, sort_src2_ub)
                                elif base + 1 < num_trunks:
                                    T.tile.merge_sort(merged_ub, history_ub, sort_src0_ub, sort_src1_ub)
                                else:
                                    T.tile.merge_sort(merged_ub, history_ub, sort_src0_ub)
                                T.copy(merged_ub[0 : 2 * TOP_K], history_ub)
                            # Extract indices directly from history_ub.
                            T.tile.gather_mask(topk_index_ub, history_ub, "P1010")
                            T.tile.cast(output_ub, topk_index_ub, "CAST_ROUND", TOP_K)
                            T.set_flag("V", "MTE3", OUTPUT_READY_FLAG)
                            T.wait_flag("V", "MTE3", OUTPUT_READY_FLAG)
                            T.copy(output_ub, OUT[gb, gn2, s1_id, 0:TOP_K])
                            T.set_flag("MTE3", "V", OUTPUT_READY_FLAG)
                            T.wait_flag("MTE3", "V", OUTPUT_READY_FLAG)
                else:
                    # With one split, each block merges only its own data.
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
                                # Initialize history with -inf and load trunk 0.
                                T.tile.fill(history_ub, -T.infinity(calc_dtype))
                                T.set_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                T.wait_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                                T.copy(SORTED_WORKSPACE[group, s1_offset, 0, 0 : 2 * TRUNK_KEEP], history_ub[0 : 2 * TRUNK_KEEP])
                                T.set_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                T.wait_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                                # Merge the remaining trunks in batches of three.
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
                                # Write the output indices.
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
        raise ValueError("Output index tensor shapes do not match")
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
    # BLOCK_K defaults to D because KEY is loaded at width D.
    if block_k is None:
        block_k = dimension
    block_m = 64
    vector_baseg = 16
    if s2_splits is None:
        s2_splits = auto_s2_splits(batch, s1, s2, block_n)
    validate_indexer_config(
        n2,
        groups,
        dimension,
        top_k,
        vector_basen,
        vector_baseg,
        block_m,
        block_n,
        block_k,
        s2,
        s2_splits=s2_splits,
        batch=batch,
        s1=s1,
        s2=s2,
    )

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
    print(f"Index multiset match ratio: {matched_ratio:.6f}; mismatched indices: {mismatches}")
    if matched_ratio > 0.99:
        print("[PRECISION_PASS] Online TopK index multiset match ratio exceeds 0.99")
        return
    print("[PRECISION_FAIL] Online TopK index multiset match ratio does not exceed 0.99")
    raise AssertionError("Online TopK index multiset validation failed")


def main():
    """Run the small correctness case used by the example entry point."""
    tilelang.disable_cache()
    test_indexer(s1=64, s2=1024, top_k=256, block_n=64, vector_basen=64)
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
