"""Expert Flash Attention v2 with three-stage composed online softmax.

See ``README.md`` for the optimization journey.
"""

import argparse

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout
import torch

tilelang.disable_cache()

# ===========================================================================
# Lock declarations (preprocessor expands to T.set_flag / T.wait_flag)
# ===========================================================================

# Cube scope

# Vector scope: shared load/store buffers across softmax and O_acc

# ===========================================================================
# Constants
# ===========================================================================
NUM_CORES = 24  # 910B AI Cores
ROW_EXPAND_VECTOR_ELEMS = 64
DIM = 128  # head dimension (fixed)

TILE_Q_L2 = 256  # query tile cached in L2
TILE_KV_L1 = 1024  # key/value tile staged in L1
TILE_KV_L2 = 1024  # key/value tile at L2 granularity

TILE_M = 128  # L0 tile on M (query) axis
TILE_K = 128  # L0 tile on contraction / output-split axis

TILE_Q_UB = 8  # narrow strip for softmax (transcendentals)
TILE_Q_ACC = 64  # wide strip for O_acc update (MACs)

NUM_STAGES = 3  # pipeline depth / token-ring size
NUM_L1_CHUNKS = TILE_KV_L2 // TILE_KV_L1  # currently 1

# L0 iteration counts (derived, used in GEMM macros)
M_ITERS = TILE_Q_L2 // TILE_M  # 2  — m-tiles per GEMM invocation
N_ITERS = TILE_KV_L1 // TILE_K  # 8  — output-split tiles (GEMM1)
K_ITERS = TILE_KV_L1 // TILE_K  # 8  — contraction tiles  (GEMM2)

# Cross-core semaphore IDs
SEM_CUBE = 0  # Vector → Cube: "P ready" / "slot free"
SEM_VEC = 1  # Cube → Vector: "S ready" / "O ready"

# composed softmax block alignment: float32 → 8 elements per 32-byte block
ELEM_PER_BLK = 8
SFM_WORKSPACE_BYTES = 8384  # PR #1852 fp32 [8,1024], including clear=False

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False,
}


# ===========================================================================
# Layer 2: Building-block macros
# ===========================================================================


# ---------------------------------------------------------------------------
# gemm_output_split: Q[TILE_Q_L2, DIM] @ K[TILE_KV_L1, DIM]^T → S[TILE_Q_L2, TILE_KV_L1]
# ---------------------------------------------------------------------------
@T.macro(hygienic=False)
def gemm_output_split(
    a_src,
    a_i0,
    a_offset,
    b_src,
    b_i0,
    b_offset,
    out,
    out_i0,
    out_i1,
    a_l1,
    kv_l1,
    l0a,
    l0b,
    l0c,
):
    # Phase 1: bulk-load all K tiles to L1
    T.wait_flag("MTE1", "MTE2", 2)
    for ni in T.serial(N_ITERS):
        T.copy(b_src[b_i0, b_offset + ni * TILE_K : b_offset + (ni + 1) * TILE_K, :], kv_l1[ni, :, :])
    T.set_flag("MTE2", "MTE1", 2)

    # Phase 2: tiled MMA (Q loaded per m-tile, a_l1 double-buffered on mi)
    T.wait_flag("MTE2", "MTE1", 2)
    for mi in T.serial(M_ITERS):
        mi_side = mi % 2
        T.wait_flag("MTE1", "MTE2", 0 + mi_side)
        T.copy(a_src[a_i0, a_offset + mi * TILE_M : a_offset + (mi + 1) * TILE_M, :], a_l1[mi_side, :, :])
        T.set_flag("MTE2", "MTE1", 0 + mi_side)

        T.wait_flag("MTE2", "MTE1", 0 + mi_side)
        for ni in T.serial(N_ITERS):
            side = ni % 2
            T.wait_flag("M", "MTE1", 3 + side)
            if ni < 2:
                T.copy(a_l1[mi_side, :, :], l0a[side, :, :])
            T.copy(kv_l1[ni, :, :], l0b[side, :, :], transpose=True)
            T.set_flag("MTE1", "M", 3 + side)

            T.wait_flag("MTE1", "M", 3 + side)
            T.wait_flag("FIX", "M", 5 + side)
            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
            T.set_flag("M", "MTE1", 3 + side)
            T.set_flag("M", "FIX", 5 + side)
            T.wait_flag("M", "FIX", 5 + side)
            T.copy(l0c[side, :, :], out[out_i0, out_i1, mi * TILE_M : (mi + 1) * TILE_M, ni * TILE_K : (ni + 1) * TILE_K])
            T.set_flag("FIX", "M", 5 + side)
        T.set_flag("MTE1", "MTE2", 0 + mi_side)
    T.set_flag("MTE1", "MTE2", 2)


# ---------------------------------------------------------------------------
# gemm_contraction_split: P[TILE_Q_L2, TILE_KV_L1] @ V[TILE_KV_L1, DIM] → O[TILE_Q_L2, DIM]
# ---------------------------------------------------------------------------
@T.macro(hygienic=False)
def gemm_contraction_split(
    a_src,
    a_i0,
    a_i1,
    b_src,
    b_i0,
    b_offset,
    out,
    out_i0,
    out_i1,
    a_l1,
    kv_l1,
    l0a,
    l0b,
    l0c,
):
    # Phase 1: load all V tiles to L1
    T.wait_flag("MTE1", "MTE2", 2)
    for ki in T.serial(K_ITERS):
        T.copy(b_src[b_i0, b_offset + ki * TILE_K : b_offset + (ki + 1) * TILE_K, :], kv_l1[ki, :, :])
    T.set_flag("MTE2", "MTE1", 2)

    # Phase 2: contraction-split tiled MMA (a_l1 double-buffered on ki)
    T.wait_flag("MTE2", "MTE1", 2)
    for mi in T.serial(M_ITERS):
        c_side = mi % 2
        T.wait_flag("FIX", "M", 5 + c_side)
        for ki in T.serial(K_ITERS):
            side = ki % 2
            T.wait_flag("MTE1", "MTE2", 0 + side)
            T.copy(a_src[a_i0, a_i1, mi * TILE_M : (mi + 1) * TILE_M, ki * TILE_K : (ki + 1) * TILE_K], a_l1[side, :, :])
            T.set_flag("MTE2", "MTE1", 0 + side)

            T.wait_flag("M", "MTE1", 3 + side)
            T.wait_flag("MTE2", "MTE1", 0 + side)
            T.copy(a_l1[side, :, :], l0a[side, :, :])
            T.set_flag("MTE1", "MTE2", 0 + side)
            T.copy(kv_l1[ki, :, :], l0b[side, :, :])
            T.set_flag("MTE1", "M", 3 + side)

            T.wait_flag("MTE1", "M", 3 + side)
            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[c_side, :, :], init=(ki == 0))
            T.set_flag("M", "MTE1", 3 + side)
        T.set_flag("M", "FIX", 5 + c_side)
        T.wait_flag("M", "FIX", 5 + c_side)
        T.copy(l0c[c_side, :, :], out[out_i0, out_i1, mi * TILE_M : (mi + 1) * TILE_M, :])
        T.set_flag("FIX", "M", 5 + c_side)
    T.set_flag("MTE1", "MTE2", 2)


# ===========================================================================
# Layer 3: Task macros (Vector scope)
# ===========================================================================

HALF_Q = TILE_Q_L2 // 2
SOFTMAX_STRIPS = HALF_Q // TILE_Q_UB  # 16
O_ACC_STRIPS = HALF_Q // TILE_Q_ACC  # 2


# ---------------------------------------------------------------------------
# vec_softmax: TASK 2 — S → P + stats + early release
#
# Phase 2A: 16 narrow strips [TILE_Q_UB=8, TILE_KV_L2=1024]
#   load S → composed softmax → store P
# Early release: signal Cube after all P stored
# Phase 2B: scale_ring for O_acc rescaling — no DMA
# ---------------------------------------------------------------------------
@T.macro(hygienic=False)
def softmax_composed_update(r, score_local, m_stats, l_panel, sfm_brcb, sfm_tmp):
    # Scaled scores -> reduce -> BRCB -> independent Sub batch -> Exp -> sum.
    # Explicit boundaries also protect reuse of the shared reduction arena.
    T.pipe_barrier("v")
    T.reduce_max(
        score_local,
        m_stats[r * TILE_Q_UB : (r + 1) * TILE_Q_UB, :],
        dim=-1,
        clear=False,
        tmp=sfm_tmp,
    )
    T.pipe_barrier("v")
    T.tile.brcb_experiment(
        sfm_brcb,
        m_stats[r * TILE_Q_UB : (r + 1) * TILE_Q_UB, :],
        TILE_Q_UB // 8,
        1,
        8,
    )
    T.pipe_barrier("v")
    for sub_chunk in T.unroll(TILE_KV_L2 // ROW_EXPAND_VECTOR_ELEMS):
        sub_col = sub_chunk * ROW_EXPAND_VECTOR_ELEMS
        T.tile.row_expand_sub_experiment(
            score_local[:, sub_col : sub_col + ROW_EXPAND_VECTOR_ELEMS],
            score_local[:, sub_col : sub_col + ROW_EXPAND_VECTOR_ELEMS],
            sfm_brcb,
        )
    T.pipe_barrier("v")
    T.tile.exp(score_local, score_local)
    T.pipe_barrier("v")
    T.reduce_sum(
        score_local,
        l_panel[r * TILE_Q_UB : (r + 1) * TILE_Q_UB, :],
        dim=-1,
        tmp=sfm_tmp,
    )
    # Probability casting only reads score_local too. The next strip's entry
    # barrier protects score/scratch reuse; Phase 2B orders the l_panel consumer.


@T.macro(hygienic=False)
def finish_softmax_stats(stage, q_ring, scale_ring, m_prev, sum_brcb, l_panel, l_stats_ring):
    T.tile.brcb_experiment(scale_ring[stage, :, :], m_prev, HALF_Q // 8, 1, 8)
    T.tile.brcb_experiment(sum_brcb, l_panel, HALF_Q // 8, 1, 8)
    T.pipe_barrier("v")
    T.tile.mul(l_stats_ring[q_ring, :, :], l_stats_ring[q_ring, :, :], scale_ring[stage, :, :])
    T.pipe_barrier("v")
    T.tile.add(l_stats_ring[q_ring, :, :], l_stats_ring[q_ring, :, :], sum_brcb)
    T.pipe_barrier("v")


# ---------------------------------------------------------------------------
# vec_softmax: TASK 2 — S → P + stats + early release
#
# Phase 2A: 16 narrow strips [TILE_Q_UB=8, TILE_KV_L2=1024]
#   load S → reduce/broadcast/sub/exp/reduce → store P
# Early release: signal Cube after all P stored
# Phase 2B: scale banks for O_acc rescaling — no DMA
# ---------------------------------------------------------------------------
@T.macro(hygienic=False)
def vec_softmax(
    step,
    my_start,
    num_kv_blocks,
    num_q_stages,
    sm_scale,
    cid,
    vid,
    ws_sp,
    ld_score,
    score_local,
    st_prob,
    m_stats,
    m_prev,
    l_panel,
    sfm_brcb,
    sum_brcb,
    l_stats_ring,
    scale_ring,
    sfm_tmp,
):
    stage = step % NUM_STAGES
    q_task = step // num_kv_blocks
    global_q = my_start + q_task
    kv_idx = step % num_kv_blocks
    q_ring = global_q % num_q_stages

    if kv_idx == 0:
        T.tile.fill(m_stats, -(2**30))
        T.tile.fill(l_stats_ring[q_ring, :, :], 0.0)
        T.pipe_barrier("v")

    # Save the previous max in bulk, before any strip replaces its values.
    T.copy(m_stats, m_prev)

    # --- Phase 2A: narrow strip softmax via composed online softmax ---
    for r in T.serial(SOFTMAX_STRIPS):
        row = vid * HALF_Q + r * TILE_Q_UB

        # MTE2: load S strip from workspace
        T.wait_flag("V", "MTE2", 0)
        T.copy(ws_sp[cid, stage, row : row + TILE_Q_UB, :], ld_score)
        T.set_flag("MTE2", "V", 0)

        # V: fp16 → fp32
        T.wait_flag("MTE2", "V", 0)
        # Previous strip's probability cast reads score_local; next cast overwrites it.
        T.pipe_barrier("v")
        T.copy(ld_score, score_local)
        T.set_flag("V", "MTE2", 0)
        T.pipe_barrier("v")

        # Pre-scale scores for online softmax
        T.tile.mul(score_local, score_local, sm_scale)

        softmax_composed_update(r, score_local, m_stats, l_panel, sfm_brcb, sfm_tmp)

        # V → MTE3: store P strip
        T.wait_flag("MTE3", "V", 1)
        T.copy(score_local, st_prob)
        T.set_flag("V", "MTE3", 1)
        T.wait_flag("V", "MTE3", 1)
        T.copy(st_prob, ws_sp[cid, stage, row : row + TILE_Q_UB, :])
        T.set_flag("MTE3", "V", 1)

    # --- Early release: P is complete, let Cube start P@V ---
    T.set_cross_flag("MTE3", SEM_CUBE)

    # No Cube consumer uses statistics. Batch their update after releasing P.
    T.tile.sub(m_prev, m_prev, m_stats)
    T.pipe_barrier("v")
    T.tile.exp(m_prev, m_prev)
    T.pipe_barrier("v")
    finish_softmax_stats(stage, q_ring, scale_ring, m_prev, sum_brcb, l_panel, l_stats_ring)


@T.macro(hygienic=False)
def vec_o_acc(
    step,
    my_start,
    num_q_blocks,
    num_kv_blocks,
    num_q_stages,
    cid,
    vid,
    ws_op,
    ws_oa,
    O_bh,
    ld_o_partial,
    ld_o_acc,
    st_o_acc,
    st_o_norm,
    row_expand_scalars,
    scale_ring,
    l_stats_ring,
):
    old_step = step - NUM_STAGES
    stage = step % NUM_STAGES
    old_q_task = old_step // num_kv_blocks
    old_global_q = my_start + old_q_task
    bh = old_global_q // num_q_blocks
    local_q = old_global_q % num_q_blocks
    kv_idx = old_step % num_kv_blocks
    q_ring = old_global_q % num_q_stages

    for r in T.serial(O_ACC_STRIPS):
        row = vid * HALF_Q + r * TILE_Q_ACC

        if kv_idx == 0:
            T.wait_flag("V", "MTE2", 0)
            T.copy(ws_op[cid, stage, row : row + TILE_Q_ACC, :], ld_o_partial)
            T.set_flag("MTE2", "V", 0)

            T.wait_flag("MTE3", "V", 1)
            T.wait_flag("MTE2", "V", 0)
            T.copy(ld_o_partial, st_o_acc)
            T.set_flag("V", "MTE2", 0)
        else:
            T.wait_flag("V", "MTE2", 0)
            T.copy(ws_oa[cid, row : row + TILE_Q_ACC, :], ld_o_acc)
            T.copy(ws_op[cid, stage, row : row + TILE_Q_ACC, :], ld_o_partial)
            T.set_flag("MTE2", "V", 0)

            T.wait_flag("MTE2", "V", 0)
            T.wait_flag("MTE3", "V", 1)
            T.copy(
                scale_ring[stage, r * TILE_Q_ACC : (r + 1) * TILE_Q_ACC, :],
                row_expand_scalars,
            )
            T.pipe_barrier("v")
            for chunk in T.unroll(DIM // ROW_EXPAND_VECTOR_ELEMS):
                col = chunk * ROW_EXPAND_VECTOR_ELEMS
                T.tile.row_expand_mul_experiment(
                    st_o_acc[:, col : col + ROW_EXPAND_VECTOR_ELEMS],
                    ld_o_acc[:, col : col + ROW_EXPAND_VECTOR_ELEMS],
                    row_expand_scalars,
                )
            T.pipe_barrier("v")  # Row-wise rescaling -> partial-O addition.
            T.tile.add(st_o_acc, st_o_acc, ld_o_partial)
            T.set_flag("V", "MTE2", 0)

        if kv_idx == num_kv_blocks - 1:
            T.copy(
                l_stats_ring[q_ring, r * TILE_Q_ACC : (r + 1) * TILE_Q_ACC, :],
                row_expand_scalars,
            )
            # O copy/add and the denominator broadcast-copy feed normalization.
            T.pipe_barrier("v")
            for chunk in T.serial(DIM // ROW_EXPAND_VECTOR_ELEMS):
                col = chunk * ROW_EXPAND_VECTOR_ELEMS
                T.tile.row_expand_div_experiment(
                    st_o_acc[:, col : col + ROW_EXPAND_VECTOR_ELEMS],
                    st_o_acc[:, col : col + ROW_EXPAND_VECTOR_ELEMS],
                    row_expand_scalars,
                )
            T.pipe_barrier("v")  # Normalization -> fp16 cast.

            q_row = local_q * TILE_Q_L2 + row
            T.wait_flag("MTE3", "V", 2)
            T.copy(st_o_acc, st_o_norm)
            T.set_flag("V", "MTE3", 2)
            T.wait_flag("V", "MTE3", 2)
            T.copy(st_o_norm, O_bh[bh, q_row : q_row + TILE_Q_ACC, :])
            T.set_flag("MTE3", "V", 2)

        T.set_flag("V", "MTE3", 1)
        T.wait_flag("V", "MTE3", 1)
        if kv_idx != num_kv_blocks - 1:
            T.copy(st_o_acc, ws_oa[cid, row : row + TILE_Q_ACC, :])
        T.set_flag("MTE3", "V", 1)


# ===========================================================================


# ===========================================================================
# Main kernel
# ===========================================================================
@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs, compile_flags=["--cce-auto-sync=off", "-O3"])
def flash_attention_fwd(
    batch,
    seq_len,
    heads_q,
    heads_kv,
    dim,
):
    assert heads_q % heads_kv == 0
    assert dim == DIM
    assert seq_len % TILE_KV_L2 == 0
    assert seq_len % TILE_Q_L2 == 0

    dtype = "float16"
    accum_dtype = "float"

    sm_scale = (1.0 / dim) ** 0.5

    shape_q = [batch, heads_q, seq_len, dim]
    shape_kv = [batch, heads_kv, seq_len, dim]

    # --- GQA → MHA fold (identical to tl_expert) ---
    q_per_kv = heads_q // heads_kv
    num_bh = batch * heads_kv
    q_seq = seq_len * q_per_kv
    kv_seq = seq_len

    # --- Task geometry ---
    num_q_blocks = q_seq // TILE_Q_L2
    num_kv_blocks = kv_seq // TILE_KV_L2
    global_q_tasks = num_bh * num_q_blocks

    num_q_stages = 1 + (NUM_STAGES + num_kv_blocks - 1) // num_kv_blocks

    q_tasks_per_core = global_q_tasks // NUM_CORES
    r_tasks = global_q_tasks % NUM_CORES

    @T.prim_func
    def main(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_kv, dtype),
        V: T.Tensor(shape_kv, dtype),
        Output: T.Tensor(shape_q, dtype),
        ws_sp: T.Tensor([NUM_CORES, NUM_STAGES, TILE_Q_L2, TILE_KV_L2], dtype),
        ws_op: T.Tensor([NUM_CORES, NUM_STAGES, TILE_Q_L2, DIM], accum_dtype),
        ws_oa: T.Tensor([NUM_CORES, TILE_Q_L2, DIM], accum_dtype),
    ):
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            # --- GQA reshape aliases (no copy) ---
            Q_bh = T.decl_buffer([num_bh, q_seq, dim], dtype, data=Q.data, scope="global")
            K_bh = T.decl_buffer([num_bh, kv_seq, dim], dtype, data=K.data, scope="global")
            V_bh = T.decl_buffer([num_bh, kv_seq, dim], dtype, data=V.data, scope="global")
            O_bh = T.decl_buffer([num_bh, q_seq, dim], dtype, data=Output.data, scope="global")

            # --- L1 buffers (Cube data path) ---
            a_l1 = T.alloc_L1([2, TILE_M, TILE_K], dtype)
            kv_l1 = T.alloc_L1([N_ITERS, TILE_K, DIM], dtype)

            T.annotate_layout(
                {
                    a_l1: make_zn_layout(a_l1),
                    kv_l1: make_nz_layout(kv_l1),
                }
            )

            # --- L0 buffers (Cube compute, double-buffered) ---
            l0a = T.alloc_L0A([2, TILE_M, TILE_K], dtype)
            l0b = T.alloc_L0B([2, TILE_K, DIM], dtype)
            l0c = T.alloc_L0C([2, TILE_M, DIM], accum_dtype)

            # --- UB buffers (Vector scope, per sub-core: 192KB each) ---
            #
            # [0K,   32K)  work  — score_local: composed softmax in-place src/dst
            # [32K,  48K)  store — st_o_norm: final O output (fp16)
            # [48K,  80K)  store — st_prob / st_o_acc (alias, fp16/fp32)
            # [80K,  96K)  load  — ld_score (fp16, alias ld_o_acc)
            # [80K, 112K)  load  — ld_o_acc (fp32, alias ld_score)
            # [112K,144K)  load  — ld_o_partial (fp32)
            # [144K,~160K) auto  — stats + composed softmax temporaries

            # --- Work [0K, 32K) — V-pipe only, no Lock ---
            score_local = T.alloc_ub([TILE_Q_UB, TILE_KV_L2], accum_dtype)  # 32KB

            # --- Store [32K, 80K) — ST_ON + ST_BUF Lock ---
            st_o_norm = T.alloc_ub([TILE_Q_ACC, DIM], dtype)  # 16KB
            st_prob = T.alloc_ub([TILE_Q_UB, TILE_KV_L2], dtype)  # 16KB
            st_o_acc = T.alloc_ub([TILE_Q_ACC, DIM], accum_dtype)  # 32KB (alias st_prob)

            # --- Load [80K, 144K) — LD_BUF Lock ---
            ld_score = T.alloc_ub([TILE_Q_UB, TILE_KV_L2], dtype)  # 16KB
            ld_o_acc = T.alloc_ub([TILE_Q_ACC, DIM], accum_dtype)  # 32KB (alias ld_score)
            ld_o_partial = T.alloc_ub([TILE_Q_ACC, DIM], accum_dtype)  # 32KB

            row_expand_scalars = T.alloc_ub([TILE_Q_ACC, ELEM_PER_BLK], accum_dtype)
            T.annotate_address(
                {
                    # work [0K, 32K)
                    score_local: 0,
                    row_expand_scalars: 0,
                    # store [32K, 80K)
                    st_o_norm: 32768,
                    st_prob: 49152,
                    st_o_acc: 49152,
                    # load [80K, 144K)
                    ld_score: 81920,
                    ld_o_acc: 81920,
                    ld_o_partial: 114688,
                }
            )

            # --- Auto [144K, ~158K) — stats + composed softmax workspace ---
            # Compact statistics are expanded only for row-wise consumers.
            m_stats = T.alloc_ub([HALF_Q, 1], accum_dtype)
            m_prev = T.alloc_ub([HALF_Q, 1], accum_dtype)
            l_panel = T.alloc_ub([HALF_Q, 1], accum_dtype)
            sfm_brcb = T.alloc_ub([TILE_Q_UB, ELEM_PER_BLK], accum_dtype)
            sum_brcb = T.alloc_ub([HALF_Q, ELEM_PER_BLK], accum_dtype)
            l_stats_ring = T.alloc_ub([num_q_stages, HALF_Q, ELEM_PER_BLK], accum_dtype)
            scale_ring = T.alloc_ub([NUM_STAGES, HALF_Q, ELEM_PER_BLK], accum_dtype)
            sfm_tmp = T.alloc_ub([SFM_WORKSPACE_BYTES], "uint8")

            my_start = cid * q_tasks_per_core + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks_per_core + T.if_then_else(cid < r_tasks, 1, 0)
            my_total_steps = my_count * num_kv_blocks

            # ===============================================================
            # Cube scope
            # ===============================================================
            with T.Scope("C"):
                T.set_flag("MTE1", "MTE2", 0)
                T.set_flag("MTE1", "MTE2", 1)
                T.set_flag("MTE1", "MTE2", 2)
                T.set_flag("M", "MTE1", 3)
                T.set_flag("M", "MTE1", 4)
                T.set_flag("FIX", "M", 5)
                T.set_flag("FIX", "M", 6)

                for step in T.serial(my_total_steps + NUM_STAGES):
                    T.wait_cross_flag(SEM_CUBE)
                    stage = step % NUM_STAGES

                    if step >= NUM_STAGES:
                        old_step = step - NUM_STAGES
                        bh_a = (my_start + old_step // num_kv_blocks) // num_q_blocks
                        kv_off_a = (old_step % num_kv_blocks) * TILE_KV_L2
                        gemm_contraction_split(
                            ws_sp,
                            cid,
                            stage,
                            V_bh,
                            bh_a,
                            kv_off_a,
                            ws_op,
                            cid,
                            stage,
                            a_l1,
                            kv_l1,
                            l0a,
                            l0b,
                            l0c,
                        )

                    if step < my_total_steps:
                        global_q = my_start + step // num_kv_blocks
                        bh_b = global_q // num_q_blocks
                        q_off = (global_q % num_q_blocks) * TILE_Q_L2
                        kv_off_b = (step % num_kv_blocks) * TILE_KV_L2
                        gemm_output_split(
                            Q_bh,
                            bh_b,
                            q_off,
                            K_bh,
                            bh_b,
                            kv_off_b,
                            ws_sp,
                            cid,
                            stage,
                            a_l1,
                            kv_l1,
                            l0a,
                            l0b,
                            l0c,
                        )

                    T.set_cross_flag("FIX", SEM_VEC)

                T.wait_flag("MTE1", "MTE2", 0)
                T.wait_flag("MTE1", "MTE2", 1)
                T.wait_flag("MTE1", "MTE2", 2)
                T.wait_flag("M", "MTE1", 3)
                T.wait_flag("M", "MTE1", 4)
                T.wait_flag("FIX", "M", 5)
                T.wait_flag("FIX", "M", 6)

            # ===============================================================
            # Vector scope
            # ===============================================================
            with T.Scope("V"):
                for _ in range(NUM_STAGES):
                    T.set_cross_flag("MTE2", SEM_CUBE)
                T.set_flag("V", "MTE2", 0)
                T.set_flag("MTE3", "V", 1)
                T.set_flag("MTE3", "V", 2)
                T.set_flag("MTE3", "MTE2", 3)

                for step in T.serial(my_total_steps + NUM_STAGES):
                    T.wait_cross_flag(SEM_VEC)

                    if step >= NUM_STAGES:
                        T.wait_flag("MTE3", "MTE2", 3)
                        vec_o_acc(
                            step,
                            my_start,
                            num_q_blocks,
                            num_kv_blocks,
                            num_q_stages,
                            cid,
                            vid,
                            ws_op,
                            ws_oa,
                            O_bh,
                            ld_o_partial,
                            ld_o_acc,
                            st_o_acc,
                            st_o_norm,
                            row_expand_scalars,
                            scale_ring,
                            l_stats_ring,
                        )
                        T.set_flag("MTE3", "MTE2", 3)

                    if step < my_total_steps:
                        vec_softmax(
                            step,
                            my_start,
                            num_kv_blocks,
                            num_q_stages,
                            sm_scale,
                            cid,
                            vid,
                            ws_sp,
                            ld_score,
                            score_local,
                            st_prob,
                            m_stats,
                            m_prev,
                            l_panel,
                            sfm_brcb,
                            sum_brcb,
                            l_stats_ring,
                            scale_ring,
                            sfm_tmp,
                        )

                T.wait_flag("V", "MTE2", 0)
                T.wait_flag("MTE3", "V", 1)
                T.wait_flag("MTE3", "V", 2)
                T.wait_flag("MTE3", "MTE2", 3)

    return main


def ref_flash_attn(q, k, v):
    """Compute the non-causal GQA reference with PyTorch SDPA."""
    if k.shape[1] != q.shape[1]:
        repeats = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    return torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float(),
        v.float(),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    ).to(torch.float16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1, help="batch size")
    parser.add_argument("--S", type=int, default=4096, help="sequence length")
    parser.add_argument("--H", type=int, default=12, help="query heads when --q-heads is omitted")
    parser.add_argument("--q-heads", type=int, default=None, help="query head count")
    parser.add_argument("--kv-heads", type=int, default=1, help="key/value head count")
    parser.add_argument("--D", type=int, default=128, help="head dimension")
    parser.add_argument("--no-check", action="store_true", help="skip the PyTorch reference check")
    args = parser.parse_args()

    batch, seq_len, dim = args.B, args.S, args.D
    heads_q = args.q_heads or args.H
    heads_kv = args.kv_heads

    torch.set_default_device("npu")
    torch.manual_seed(0)

    func = flash_attention_fwd(
        batch=batch,
        seq_len=seq_len,
        heads_q=heads_q,
        heads_kv=heads_kv,
        dim=dim,
    )
    print("Init successful!")

    q = torch.randn((batch, heads_q, seq_len, dim), dtype=torch.float16)
    k = torch.randn((batch, heads_kv, seq_len, dim), dtype=torch.float16)
    v = torch.randn((batch, heads_kv, seq_len, dim), dtype=torch.float16)

    output = func(q, k, v)
    torch.npu.synchronize()

    if not args.no_check:
        reference = ref_flash_attn(q, k, v)
        torch.npu.synchronize()
        torch.testing.assert_close(reference, output, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
        print("Kernel Output Match!")
    else:
        print("Reference check skipped.")
