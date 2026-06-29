import os

import tilelang
from tilelang import language as T
import torch


def _is_simulator():
    return "OPPROF" in os.environ.get("CAMODEL_CONFIG_PATH", "")


NUM_HEADS = 32
KV_HEADS = 8
GROUP_SIZE = NUM_HEADS // KV_HEADS
D = 128
BEAM_SIZE = 128
KV_LEN = 1024
BLOCK_N = 128
STACK_NUM = 4
HALF_STACK = STACK_NUM // 2
EFFECTIVE_BLOCK_N = BLOCK_N * STACK_NUM
HALF_BLOCK_N = BLOCK_N * HALF_STACK
BM = 128
NUM_CORES = 20
SHARED_CORES = 12
UNSHARED_CORES = NUM_CORES - SHARED_CORES

UNSHARED_BEAM = BEAM_SIZE
DECODE_STEP = 4
BATCH_SIZE = 32
UNSHARED_GROUPS_PER_TASK = BATCH_SIZE
UNSHARED_BEAMS_PER_TASK = UNSHARED_GROUPS_PER_TASK // KV_HEADS
Q_BATCH = UNSHARED_GROUPS_PER_TASK * GROUP_SIZE
KV_BATCH = UNSHARED_GROUPS_PER_TASK * DECODE_STEP
TOTAL_UNSHARED_GROUPS = UNSHARED_BEAM * KV_HEADS
TOTAL_UNSHARED_TASKS = TOTAL_UNSHARED_GROUPS // UNSHARED_GROUPS_PER_TASK
TASKS_PER_CORE = (TOTAL_UNSHARED_TASKS + UNSHARED_CORES - 1) // UNSHARED_CORES
UB_ALIGN_ELEMS = 8
PRE_LAUNCH = 2
TASKQUE_SLOTS = PRE_LAUNCH + 1
BUILD_TAG = "target_qhead_parallel_stack4_direct_unshared_o_v37_light_spo_slots"
WARMUP_RUNS = 3
NUM_RUNS = 10


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


tilelang.disable_cache()
VEC_NUM = 2
v_p = BM // VEC_NUM
SHARED_S_UB_ELEMS = 8192
SHARED_S_ROWS = SHARED_S_UB_ELEMS // EFFECTIVE_BLOCK_N
SHARED_S_ROW_TILES = v_p // SHARED_S_ROWS
MERGE_ROW_TILE = 64
MERGE_BEAMS_PER_TILE = MERGE_ROW_TILE // NUM_HEADS
MERGE_WORKERS = NUM_CORES * VEC_NUM
MERGE_TOTAL_TILES = BEAM_SIZE // MERGE_BEAMS_PER_TILE
MERGE_TILES_PER_WORKER = (MERGE_TOTAL_TILES + MERGE_WORKERS - 1) // MERGE_WORKERS
SHARED_SOFTMAX_DB = 2
SHARED_RESCALE_EVENT = 2
MERGE_BRCB_REPEAT = MERGE_ROW_TILE // UB_ALIGN_ELEMS
MERGE_FLOAT_VECTOR_ELEMS = 64
MERGE_FLOAT_MASK = 0xFFFFFFFFFFFFFFFF
SHARED_S_BRCB_REPEAT = SHARED_S_ROWS // UB_ALIGN_ELEMS
SHARED_S_VECTOR_ELEMS = 64
SHARED_O_VECTOR_ELEMS = 64
SHARED_GMGL_BRCB_REPEAT = v_p // UB_ALIGN_ELEMS
UNSHARED_VECTOR_ELEMS = 64
UNSHARED_V_ROWS = Q_BATCH // VEC_NUM
UNSHARED_BRCB_REPEAT = UNSHARED_V_ROWS // UB_ALIGN_ELEMS
UNSHARED_GROUPS_PER_VEC = UNSHARED_GROUPS_PER_TASK // VEC_NUM
UNSHARED_BEAMS_PER_VEC = UNSHARED_GROUPS_PER_VEC // KV_HEADS
UNSHARED_KV_COPY_CHUNKS = UNSHARED_GROUPS_PER_TASK
UNSHARED_KV_ROWS_PER_COPY = KV_BATCH // UNSHARED_KV_COPY_CHUNKS
UNSHARED_QK_READY_BASE = 4
UNSHARED_SOFTMAX_READY_BASE = 7
UNSHARED_STAGE_RELEASE_BASE = 10
SHARED_QK_READY_BASE = 0
SHARED_V_READY_BASE = 1
SHARED_PV_READY_BASE = 2
SHARED_S_SLOTS = _env_int("XA_SHARED_S_SLOTS", TASKQUE_SLOTS)
SHARED_P_SLOTS = _env_int("XA_SHARED_P_SLOTS", TASKQUE_SLOTS)
SHARED_O_SLOTS = _env_int("XA_SHARED_O_SLOTS", TASKQUE_SLOTS)
if min(SHARED_S_SLOTS, SHARED_P_SLOTS, SHARED_O_SLOTS) < TASKQUE_SLOTS:
    raise ValueError("XA_SHARED_{S,P,O}_SLOTS must be >= TASKQUE_SLOTS")
SHARED_KV_TILES = KV_LEN // EFFECTIVE_BLOCK_N
SHARED_BEAM_CHUNKS = (BEAM_SIZE + BM - 1) // BM
SHARED_Q_HEADS_PER_TILE = 1
SHARED_Q_TILES_PER_KV_HEAD = GROUP_SIZE // SHARED_Q_HEADS_PER_TILE
SHARED_BEAM_KV_PAIRS = SHARED_BEAM_CHUNKS * KV_HEADS
SHARED_TOTAL_PAIRS = SHARED_BEAM_KV_PAIRS * SHARED_Q_TILES_PER_KV_HEAD
SHARED_RESIDENT_GROUPS = SHARED_BEAM_KV_PAIRS
SHARED_RESIDENT_GROUPS_PER_CORE = (
    SHARED_RESIDENT_GROUPS + SHARED_CORES - 1
) // SHARED_CORES
SHARED_ACTIVE_RESIDENT_CORES = min(SHARED_RESIDENT_GROUPS, SHARED_CORES)
SHARED_CURRENT_KV_LOAD_UNITS = SHARED_TOTAL_PAIRS * SHARED_KV_TILES * STACK_NUM
SHARED_QGROUP_KV_LOAD_UNITS = SHARED_RESIDENT_GROUPS * SHARED_KV_TILES * STACK_NUM
SHARED_PAIRS_PER_CORE = (SHARED_TOTAL_PAIRS + SHARED_CORES - 1) // SHARED_CORES
SHARED_TASKS_PER_CORE = SHARED_PAIRS_PER_CORE * SHARED_KV_TILES
SHARED_PIPE_STEPS = SHARED_TASKS_PER_CORE + PRE_LAUNCH


dtype = "bfloat16"
accum_dtype = "float"
sm_scale = (1.0 / D) ** 0.5

q_shape = [BEAM_SIZE, NUM_HEADS, D]
q_unshared_shape = [BEAM_SIZE * NUM_HEADS, D]
k_shared_shape = [KV_LEN, KV_HEADS, D]
v_shared_shape = [KV_LEN, KV_HEADS, D]
MERGE_ROWS = BEAM_SIZE * NUM_HEADS
o_shape = [MERGE_ROWS, D]
k_unshared_shape = [TOTAL_UNSHARED_GROUPS * DECODE_STEP, D]
v_unshared_shape = [TOTAL_UNSHARED_GROUPS * DECODE_STEP, D]
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


@tilelang.jit(
    out_idx=[3],
    workspace_idx=[],
    pass_configs=pass_configs,
)
def x_attention_decode_v_parallel_p0_stack4_3stage():
    block_num = NUM_CORES

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K_shared: T.Tensor(k_shared_shape, dtype),
        V_shared: T.Tensor(v_shared_shape, dtype),
        Output: T.Tensor(o_shape, dtype),
        s_buf: T.Tensor(
            [SHARED_S_SLOTS, block_num, BM, EFFECTIVE_BLOCK_N], accum_dtype
        ),
        p_buf: T.Tensor([SHARED_P_SLOTS, block_num, BM, EFFECTIVE_BLOCK_N], dtype),
        o_buf: T.Tensor([SHARED_O_SLOTS, block_num, BM, D], accum_dtype),
        K_unshared: T.Tensor(k_unshared_shape, dtype),
        V_unshared: T.Tensor(v_unshared_shape, dtype),
        Q_unshared_flat: T.Tensor(q_unshared_shape, dtype),
        shared_O_merge_ws: T.Tensor([BEAM_SIZE, NUM_HEADS, D], accum_dtype),
        shared_gm_merge_ws: T.Tensor(
            [BEAM_SIZE, NUM_HEADS, UB_ALIGN_ELEMS], accum_dtype
        ),
        shared_gl_merge_ws: T.Tensor(
            [BEAM_SIZE, NUM_HEADS, UB_ALIGN_ELEMS], accum_dtype
        ),
        unshared_O_ws: T.Tensor([UNSHARED_BEAM, NUM_HEADS, D], accum_dtype),
        unshared_gm_ws: T.Tensor([UNSHARED_BEAM, NUM_HEADS], accum_dtype),
        unshared_gl_ws: T.Tensor([UNSHARED_BEAM, NUM_HEADS], accum_dtype),
        shared_O_merge_flat: T.Tensor([MERGE_ROWS, D], accum_dtype),
        shared_gm_merge_flat: T.Tensor([MERGE_ROWS, UB_ALIGN_ELEMS], accum_dtype),
        shared_gl_merge_flat: T.Tensor([MERGE_ROWS, UB_ALIGN_ELEMS], accum_dtype),
        unshared_O_flat: T.Tensor([MERGE_ROWS, D], accum_dtype),
        unshared_gm_flat: T.Tensor([MERGE_ROWS], accum_dtype),
        unshared_gl_flat: T.Tensor([MERGE_ROWS], accum_dtype),
        block_mask_ws: T.Tensor([Q_BATCH, KV_BATCH], accum_dtype),
    ):
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([BM, D], dtype)
            k_qk_l1_0 = T.alloc_L1([BLOCK_N, D], dtype)
            k_qk_l1_1 = T.alloc_L1([BLOCK_N, D], dtype)
            shared_v_stack_l1 = T.alloc_L1([STACK_NUM, BLOCK_N, D], dtype)
            acc_s_l1 = T.alloc_L1([2, BM, BLOCK_N], dtype)
            mma_l0a = T.alloc_L0A([2, BM, BLOCK_N], dtype)
            mma_l0b = T.alloc_L0B([2, BLOCK_N, D], dtype)
            mma_l0c = T.alloc_L0C([2, BM, BLOCK_N], accum_dtype)

            p_l1_u = T.alloc_L1([BM, BLOCK_N], dtype)
            v_l1_u = T.alloc_L1([BLOCK_N, D], dtype)
            acc_s_ub = T.alloc_ub([2, SHARED_S_ROWS, EFFECTIVE_BLOCK_N], accum_dtype)
            m_i = T.alloc_ub([v_p, 1], accum_dtype)
            m_i_tiles = T.alloc_ub([SHARED_S_ROW_TILES, SHARED_S_ROWS, 1], accum_dtype)
            sumexp = T.alloc_ub([v_p, 1], accum_dtype)
            sumexp_tiles = T.alloc_ub(
                [SHARED_S_ROW_TILES, SHARED_S_ROWS, 1], accum_dtype
            )
            m_i_prev = T.alloc_ub([v_p, 1], accum_dtype)
            m_i_prev_chunk = T.alloc_ub([v_p, 1], accum_dtype)
            m_i_prev_chunk_tiles = T.alloc_ub(
                [SHARED_S_ROW_TILES, SHARED_S_ROWS, 1], accum_dtype
            )
            sumexp_i = T.alloc_ub([1, SHARED_S_ROWS, 1], accum_dtype)
            m_i_next = T.alloc_ub([1, SHARED_S_ROWS, 1], accum_dtype)
            shared_s_brcb_buf = T.alloc_ub([SHARED_S_ROWS, UB_ALIGN_ELEMS], accum_dtype)
            acc_s_half = T.alloc_ub([2, SHARED_S_ROWS, EFFECTIVE_BLOCK_N], dtype)
            acc_o_ub = T.alloc_ub([1, v_p, D], accum_dtype)
            shared_O_ub = T.alloc_ub([1, v_p, D], accum_dtype)
            shared_o_scale_slots = T.alloc_ub([SHARED_O_SLOTS, v_p, 1], accum_dtype)
            shared_m_broadcast_buf = T.alloc_ub([v_p, UB_ALIGN_ELEMS], accum_dtype)
            shared_gm_store_buf = T.alloc_ub([v_p, UB_ALIGN_ELEMS], accum_dtype)
            shared_gl_store_buf = T.alloc_ub([v_p, UB_ALIGN_ELEMS], accum_dtype)
            shared_c_pair_slot0 = T.alloc_var("int32", init=SHARED_TOTAL_PAIRS)
            shared_c_pair_slot1 = T.alloc_var("int32", init=SHARED_TOTAL_PAIRS)
            shared_c_pair_slot2 = T.alloc_var("int32", init=SHARED_TOTAL_PAIRS)
            shared_c_k_slot0 = T.alloc_var("int32", init=0)
            shared_c_k_slot1 = T.alloc_var("int32", init=0)
            shared_c_k_slot2 = T.alloc_var("int32", init=0)
            shared_c_kv_head_slot0 = T.alloc_var("int32", init=0)
            shared_c_kv_head_slot1 = T.alloc_var("int32", init=0)
            shared_c_kv_head_slot2 = T.alloc_var("int32", init=0)
            shared_pv_pair_cur = T.alloc_var("int32", init=SHARED_TOTAL_PAIRS)
            shared_pv_k_cur = T.alloc_var("int32", init=0)
            shared_pv_kv_head_cur = T.alloc_var("int32", init=0)

            shared_v_pair_slot0 = T.alloc_var("int32", init=SHARED_TOTAL_PAIRS)
            shared_v_pair_slot1 = T.alloc_var("int32", init=SHARED_TOTAL_PAIRS)
            shared_v_pair_slot2 = T.alloc_var("int32", init=SHARED_TOTAL_PAIRS)
            shared_v_k_slot0 = T.alloc_var("int32", init=0)
            shared_v_k_slot1 = T.alloc_var("int32", init=0)
            shared_v_k_slot2 = T.alloc_var("int32", init=0)
            shared_v_beam_chunk_slot0 = T.alloc_var("int32", init=0)
            shared_v_beam_chunk_slot1 = T.alloc_var("int32", init=0)
            shared_v_beam_chunk_slot2 = T.alloc_var("int32", init=0)
            shared_v_head_slot0 = T.alloc_var("int32", init=0)
            shared_v_head_slot1 = T.alloc_var("int32", init=0)
            shared_v_head_slot2 = T.alloc_var("int32", init=0)
            shared_add_pair_cur = T.alloc_var("int32", init=SHARED_TOTAL_PAIRS)
            shared_add_k_cur = T.alloc_var("int32", init=0)
            shared_add_beam_chunk_cur = T.alloc_var("int32", init=0)
            shared_add_head_cur = T.alloc_var("int32", init=0)

            unshared_s = T.alloc_ub([UNSHARED_V_ROWS, KV_BATCH], accum_dtype)
            unshared_s_half = T.alloc_ub([UNSHARED_V_ROWS, KV_BATCH], dtype)
            unshared_m = T.alloc_ub([UNSHARED_V_ROWS, 1], accum_dtype)
            unshared_m_pack = T.alloc_ub(
                [UNSHARED_BEAMS_PER_VEC, NUM_HEADS], accum_dtype
            )
            unshared_m_broadcast = T.alloc_ub(
                [UNSHARED_V_ROWS, UB_ALIGN_ELEMS], accum_dtype
            )
            unshared_sum = T.alloc_ub([UNSHARED_V_ROWS, 1], accum_dtype)
            unshared_sum_pack = T.alloc_ub(
                [UNSHARED_BEAMS_PER_VEC, NUM_HEADS], accum_dtype
            )
            unshared_mask = T.alloc_ub([UNSHARED_V_ROWS, KV_BATCH], accum_dtype)
            precision_dump_f32 = T.alloc_ub([1, D], accum_dtype)
            precision_dump_bf16 = T.alloc_ub([1, D], dtype)
            precision_dump_scalar_f32 = T.alloc_ub([UB_ALIGN_ELEMS, 1], accum_dtype)
            merge_s_O = T.alloc_ub([2, MERGE_ROW_TILE, D], accum_dtype)
            merge_u_O = T.alloc_ub([2, MERGE_ROW_TILE, D], accum_dtype)
            merge_s_gm_lane = T.alloc_ub([MERGE_ROW_TILE, UB_ALIGN_ELEMS], accum_dtype)
            merge_s_gl_lane = T.alloc_ub([MERGE_ROW_TILE, UB_ALIGN_ELEMS], accum_dtype)
            merge_s_gm_load_lane = T.alloc_ub(
                [2, MERGE_ROW_TILE, UB_ALIGN_ELEMS], accum_dtype
            )
            merge_s_gl_load_lane = T.alloc_ub(
                [2, MERGE_ROW_TILE, UB_ALIGN_ELEMS], accum_dtype
            )
            merge_s_gm = T.alloc_ub([MERGE_ROW_TILE, 1], accum_dtype)
            merge_u_gm = T.alloc_ub([2, MERGE_ROW_TILE, 1], accum_dtype)
            merge_s_gl = T.alloc_ub([MERGE_ROW_TILE, 1], accum_dtype)
            merge_u_gl = T.alloc_ub([2, MERGE_ROW_TILE, 1], accum_dtype)
            merge_gm = T.alloc_ub([MERGE_ROW_TILE, 1], accum_dtype)
            merge_cor_s = T.alloc_ub([MERGE_ROW_TILE, 1], accum_dtype)
            merge_cor_u = T.alloc_ub([MERGE_ROW_TILE, 1], accum_dtype)
            merge_gl = T.alloc_ub([MERGE_ROW_TILE, 1], accum_dtype)
            merge_out = T.alloc_ub([2, MERGE_ROW_TILE, D], dtype)

            T.annotate_address(
                {
                    q_l1: 0,
                    k_qk_l1_0: 32768,
                    k_qk_l1_1: 65536,
                    acc_s_l1: 32768,
                    shared_v_stack_l1: 98304,
                    p_l1_u: 131072,
                    v_l1_u: 163840,
                    mma_l0a: 0,
                    mma_l0b: 0,
                    mma_l0c: 0,
                    acc_s_ub: 0,
                    acc_s_half: 65536,
                    acc_o_ub: 98304,
                    shared_O_ub: 131072,
                    shared_m_broadcast_buf: 163840,
                    shared_gm_store_buf: 165888,
                    shared_gl_store_buf: 168448,
                    shared_s_brcb_buf: 167936,
                    m_i: 172032,
                    m_i_tiles: 172032,
                    sumexp: 172288,
                    sumexp_tiles: 172288,
                    m_i_prev: 172544,
                    m_i_prev_chunk: 172800,
                    m_i_prev_chunk_tiles: 172800,
                    sumexp_i: 173056,
                    m_i_next: 173120,
                    shared_o_scale_slots: 173184,
                    unshared_s: 0,
                    unshared_s_half: 32768,
                    unshared_m_broadcast: 49152,
                    unshared_mask: 81920,
                    unshared_m: 147456,
                    unshared_m_pack: 147456,
                    unshared_sum: 147712,
                    unshared_sum_pack: 147712,
                    precision_dump_f32: 174080,
                    precision_dump_bf16: 174592,
                    precision_dump_scalar_f32: 174848,
                    merge_s_O: 0,
                    merge_u_O: 65536,
                    merge_out: 131072,
                    merge_s_gm_lane: 163840,
                    merge_s_gl_lane: 165888,
                    merge_s_gm_load_lane: 167936,
                    merge_s_gl_load_lane: 172032,
                    merge_s_gm: 176128,
                    merge_u_gm: 176384,
                    merge_s_gl: 176896,
                    merge_u_gl: 177152,
                    merge_gm: 177664,
                    merge_cor_s: 177920,
                    merge_cor_u: 178176,
                    merge_gl: 178432,
                }
            )

            if cid < SHARED_CORES:
                shared_local_pair_count = (
                    SHARED_TOTAL_PAIRS + SHARED_CORES - 1 - cid
                ) // SHARED_CORES
                shared_local_real_steps = shared_local_pair_count * SHARED_KV_TILES
                shared_local_total_steps = shared_local_real_steps + PRE_LAUNCH
                for _shared_launch in range(1):
                    with T.Scope("C"):
                        for init_mte_event in T.serial(4):
                            T.set_flag("MTE1", "MTE2", init_mte_event)
                        for init_m_event in T.serial(2):
                            T.set_flag("M", "MTE1", 3 + init_m_event)
                            T.set_flag("FIX", "M", 5 + init_m_event)

                        for pipe_step in T.serial(SHARED_PIPE_STEPS):
                            current_stage = pipe_step % TASKQUE_SLOTS
                            delayed_stage = (current_stage + 1) % TASKQUE_SLOTS
                            if pipe_step < shared_local_real_steps:
                                task_idx = pipe_step
                                task_pair_loop = task_idx // SHARED_KV_TILES
                                task_k = task_idx % SHARED_KV_TILES
                                task_pair_idx = cid + task_pair_loop * SHARED_CORES
                                side = current_stage
                                s_slot = task_idx % SHARED_S_SLOTS
                                if side == 0:
                                    shared_c_pair_slot0 = SHARED_TOTAL_PAIRS
                                    shared_c_k_slot0 = task_k
                                if side == 1:
                                    shared_c_pair_slot1 = SHARED_TOTAL_PAIRS
                                    shared_c_k_slot1 = task_k
                                if side == 2:
                                    shared_c_pair_slot2 = SHARED_TOTAL_PAIRS
                                    shared_c_k_slot2 = task_k
                                if task_pair_idx < SHARED_TOTAL_PAIRS:
                                    task_beam_kv_idx = (
                                        task_pair_idx // SHARED_Q_TILES_PER_KV_HEAD
                                    )
                                    task_q_tile = (
                                        task_pair_idx
                                        - task_beam_kv_idx * SHARED_Q_TILES_PER_KV_HEAD
                                    )
                                    task_beam_chunk = task_beam_kv_idx // KV_HEADS
                                    task_kv_head_idx = (
                                        task_beam_kv_idx - task_beam_chunk * KV_HEADS
                                    )
                                    task_head_idx = (
                                        task_kv_head_idx * GROUP_SIZE
                                        + task_q_tile * SHARED_Q_HEADS_PER_TILE
                                    )
                                    if side == 0:
                                        shared_c_pair_slot0 = task_pair_idx
                                        shared_c_kv_head_slot0 = task_kv_head_idx
                                    if side == 1:
                                        shared_c_pair_slot1 = task_pair_idx
                                        shared_c_kv_head_slot1 = task_kv_head_idx
                                    if side == 2:
                                        shared_c_pair_slot2 = task_pair_idx
                                        shared_c_kv_head_slot2 = task_kv_head_idx

                                    if task_k == 0:
                                        T.wait_flag("MTE1", "MTE2", 2)
                                        T.copy(
                                            Q[
                                                task_beam_chunk * BM : (
                                                    task_beam_chunk + 1
                                                )
                                                * BM,
                                                task_head_idx,
                                                :D,
                                            ],
                                            q_l1,
                                        )

                                    T.wait_flag("MTE1", "MTE2", 0)
                                    T.copy(
                                        K_shared[
                                            task_k * STACK_NUM * BLOCK_N : (
                                                task_k * STACK_NUM + 1
                                            )
                                            * BLOCK_N,
                                            task_kv_head_idx,
                                            :D,
                                        ],
                                        k_qk_l1_0,
                                    )
                                    T.set_flag("MTE2", "MTE1", 0)

                                    T.wait_flag("MTE1", "MTE2", 1)
                                    T.copy(
                                        K_shared[
                                            (task_k * STACK_NUM + 1) * BLOCK_N : (
                                                task_k * STACK_NUM + 2
                                            )
                                            * BLOCK_N,
                                            task_kv_head_idx,
                                            :D,
                                        ],
                                        k_qk_l1_1,
                                    )
                                    T.set_flag("MTE2", "MTE1", 1)
                                    T.wait_flag("MTE2", "MTE1", 0)
                                    T.wait_flag("M", "MTE1", 3)
                                    T.wait_flag("FIX", "M", 5)
                                    T.copy(q_l1, mma_l0a[0, :, :])
                                    T.copy(k_qk_l1_0, mma_l0b[0, :, :], transpose=True)
                                    T.set_flag("MTE1", "MTE2", 0)
                                    T.set_flag("MTE1", "M", 3)
                                    T.wait_flag("MTE1", "M", 3)
                                    T.mma(
                                        mma_l0a[0, :, :],
                                        mma_l0b[0, :, :],
                                        mma_l0c[0, :, :],
                                        init=True,
                                    )
                                    T.set_flag("M", "MTE1", 3)
                                    T.set_flag("M", "FIX", 5)

                                    T.wait_flag("MTE1", "MTE2", 0)
                                    T.copy(
                                        K_shared[
                                            (task_k * STACK_NUM + 2) * BLOCK_N : (
                                                task_k * STACK_NUM + 3
                                            )
                                            * BLOCK_N,
                                            task_kv_head_idx,
                                            :D,
                                        ],
                                        k_qk_l1_0,
                                    )
                                    T.set_flag("MTE2", "MTE1", 0)
                                    T.wait_flag("MTE2", "MTE1", 1)
                                    T.wait_flag("M", "MTE1", 4)
                                    T.wait_flag("FIX", "M", 6)
                                    T.copy(q_l1, mma_l0a[1, :, :])
                                    T.copy(k_qk_l1_1, mma_l0b[1, :, :], transpose=True)
                                    T.set_flag("MTE1", "MTE2", 1)
                                    T.set_flag("MTE1", "M", 4)
                                    T.wait_flag("MTE1", "M", 4)
                                    T.mma(
                                        mma_l0a[1, :, :],
                                        mma_l0b[1, :, :],
                                        mma_l0c[1, :, :],
                                        init=True,
                                    )
                                    T.set_flag("M", "MTE1", 4)
                                    T.set_flag("M", "FIX", 6)

                                    T.wait_flag("MTE1", "MTE2", 1)
                                    T.copy(
                                        K_shared[
                                            (task_k * STACK_NUM + 3) * BLOCK_N : (
                                                task_k * STACK_NUM + 4
                                            )
                                            * BLOCK_N,
                                            task_kv_head_idx,
                                            :D,
                                        ],
                                        k_qk_l1_1,
                                    )
                                    T.set_flag("MTE2", "MTE1", 1)
                                    T.wait_flag("M", "FIX", 5)
                                    T.copy(
                                        mma_l0c[0, :, :],
                                        s_buf[s_slot, cid, :, 0:BLOCK_N],
                                    )
                                    T.set_flag("FIX", "M", 5)
                                    T.wait_flag("MTE2", "MTE1", 0)
                                    T.wait_flag("M", "MTE1", 3)
                                    T.wait_flag("FIX", "M", 5)
                                    T.copy(q_l1, mma_l0a[0, :, :])
                                    T.copy(k_qk_l1_0, mma_l0b[0, :, :], transpose=True)
                                    T.set_flag("MTE1", "MTE2", 0)
                                    T.set_flag("MTE1", "M", 3)
                                    T.wait_flag("MTE1", "M", 3)
                                    T.mma(
                                        mma_l0a[0, :, :],
                                        mma_l0b[0, :, :],
                                        mma_l0c[0, :, :],
                                        init=True,
                                    )
                                    T.set_flag("M", "MTE1", 3)
                                    T.set_flag("M", "FIX", 5)

                                    T.wait_flag("M", "FIX", 6)
                                    T.copy(
                                        mma_l0c[1, :, :],
                                        s_buf[s_slot, cid, :, BLOCK_N : 2 * BLOCK_N],
                                    )
                                    T.set_flag("FIX", "M", 6)
                                    T.wait_flag("MTE2", "MTE1", 1)
                                    T.wait_flag("M", "MTE1", 4)
                                    T.wait_flag("FIX", "M", 6)
                                    T.copy(q_l1, mma_l0a[1, :, :])
                                    T.copy(k_qk_l1_1, mma_l0b[1, :, :], transpose=True)
                                    T.set_flag("MTE1", "MTE2", 1)
                                    T.set_flag("MTE1", "M", 4)
                                    T.wait_flag("MTE1", "M", 4)
                                    T.mma(
                                        mma_l0a[1, :, :],
                                        mma_l0b[1, :, :],
                                        mma_l0c[1, :, :],
                                        init=True,
                                    )
                                    T.set_flag("M", "MTE1", 4)
                                    T.set_flag("M", "FIX", 6)
                                    T.wait_flag("M", "FIX", 5)
                                    T.copy(
                                        mma_l0c[0, :, :],
                                        s_buf[
                                            s_slot, cid, :, 2 * BLOCK_N : 3 * BLOCK_N
                                        ],
                                    )
                                    T.set_flag("FIX", "M", 5)
                                    T.wait_flag("M", "FIX", 6)
                                    T.copy(
                                        mma_l0c[1, :, :],
                                        s_buf[
                                            s_slot, cid, :, 3 * BLOCK_N : 4 * BLOCK_N
                                        ],
                                    )
                                    T.set_flag("FIX", "M", 6)
                                    if task_k == SHARED_KV_TILES - 1:
                                        T.set_flag("MTE1", "MTE2", 2)
                                    T.set_cross_flag("FIX", SHARED_QK_READY_BASE, 2)

                            if pipe_step >= PRE_LAUNCH and pipe_step < shared_local_total_steps:
                                    pv_task_idx = pipe_step - PRE_LAUNCH
                                    p_slot = pv_task_idx % SHARED_P_SLOTS
                                    o_slot = pv_task_idx % SHARED_O_SLOTS
                                    pv_side = delayed_stage
                                    shared_pv_pair_cur = SHARED_TOTAL_PAIRS
                                    if pv_side == 0:
                                        shared_pv_pair_cur = shared_c_pair_slot0
                                        shared_pv_k_cur = shared_c_k_slot0
                                        shared_pv_kv_head_cur = shared_c_kv_head_slot0
                                    if pv_side == 1:
                                        shared_pv_pair_cur = shared_c_pair_slot1
                                        shared_pv_k_cur = shared_c_k_slot1
                                        shared_pv_kv_head_cur = shared_c_kv_head_slot1
                                    if pv_side == 2:
                                        shared_pv_pair_cur = shared_c_pair_slot2
                                        shared_pv_k_cur = shared_c_k_slot2
                                        shared_pv_kv_head_cur = shared_c_kv_head_slot2
                                    if shared_pv_pair_cur < SHARED_TOTAL_PAIRS:
                                        T.wait_flag("MTE1", "MTE2", 3)
                                        for chunk in T.serial(STACK_NUM):
                                            T.copy(
                                                V_shared[
                                                    (
                                                        shared_pv_k_cur * STACK_NUM
                                                        + chunk
                                                    )
                                                    * BLOCK_N : (
                                                        shared_pv_k_cur * STACK_NUM
                                                        + chunk
                                                        + 1
                                                    )
                                                    * BLOCK_N,
                                                    shared_pv_kv_head_cur,
                                                    :D,
                                                ],
                                                shared_v_stack_l1[chunk, :, :],
                                            )
                                        T.set_flag("MTE2", "MTE1", 3)

                                        T.wait_cross_flag(SHARED_V_READY_BASE)
                                        T.wait_flag("FIX", "M", 5)
                                        T.wait_flag("MTE2", "MTE1", 3)
                                        for chunk in T.serial(STACK_NUM):
                                            mma_slot = chunk % 2
                                            T.wait_flag("MTE1", "MTE2", mma_slot)
                                            T.copy(
                                                p_buf[
                                                    p_slot,
                                                    cid,
                                                    :,
                                                    chunk * BLOCK_N : (chunk + 1)
                                                    * BLOCK_N,
                                                ],
                                                acc_s_l1[mma_slot, :, :],
                                            )
                                            T.set_flag("MTE2", "MTE1", mma_slot)
                                            T.wait_flag("MTE2", "MTE1", mma_slot)
                                            T.wait_flag("M", "MTE1", 3 + mma_slot)
                                            T.copy(
                                                acc_s_l1[mma_slot, :, :],
                                                mma_l0a[mma_slot, :, :],
                                            )
                                            T.copy(
                                                shared_v_stack_l1[chunk, :, :],
                                                mma_l0b[mma_slot, :, :],
                                            )
                                            T.set_flag("MTE1", "MTE2", mma_slot)
                                            T.set_flag("MTE1", "M", 3 + mma_slot)
                                            T.wait_flag("MTE1", "M", 3 + mma_slot)
                                            T.mma(
                                                mma_l0a[mma_slot, :, :],
                                                mma_l0b[mma_slot, :, :],
                                                mma_l0c[0, :, :],
                                                init=(chunk == 0),
                                            )
                                            T.set_flag("M", "MTE1", 3 + mma_slot)

                                        T.set_flag("MTE1", "MTE2", 3)
                                        T.set_flag("M", "FIX", 5)
                                        T.wait_flag("M", "FIX", 5)
                                        T.copy(
                                            mma_l0c[0, :, :],
                                            o_buf[
                                                o_slot,
                                                cid,
                                                :,
                                                :,
                                            ],
                                        )
                                        T.set_flag("FIX", "M", 5)
                                        T.set_cross_flag("FIX", SHARED_PV_READY_BASE, 2)

                        for final_m_event in T.serial(2):
                            T.wait_flag("M", "MTE1", 3 + final_m_event)
                            T.wait_flag("FIX", "M", 5 + final_m_event)
                        for final_mte_event in T.serial(4):
                            T.wait_flag("MTE1", "MTE2", final_mte_event)

                    with T.Scope("V"):
                        for init_v_event in T.serial(3):
                            T.set_flag("V", "MTE2", init_v_event)
                        for init_store_event in T.serial(4):
                            T.set_flag("MTE3", "V", init_store_event)
                        for loop_idx in T.serial(SHARED_PIPE_STEPS):
                            current_stage = loop_idx % TASKQUE_SLOTS
                            delayed_stage = (current_stage + 1) % TASKQUE_SLOTS
                            if loop_idx < shared_local_real_steps:
                                task_idx = loop_idx
                                task_pair_loop = task_idx // SHARED_KV_TILES
                                task_k = task_idx % SHARED_KV_TILES
                                task_pair_idx = cid + task_pair_loop * SHARED_CORES
                                side = current_stage
                                s_slot = task_idx % SHARED_S_SLOTS
                                p_slot = task_idx % SHARED_P_SLOTS
                                o_slot = task_idx % SHARED_O_SLOTS
                                if side == 0:
                                    shared_v_pair_slot0 = SHARED_TOTAL_PAIRS
                                    shared_v_k_slot0 = task_k
                                if side == 1:
                                    shared_v_pair_slot1 = SHARED_TOTAL_PAIRS
                                    shared_v_k_slot1 = task_k
                                if side == 2:
                                    shared_v_pair_slot2 = SHARED_TOTAL_PAIRS
                                    shared_v_k_slot2 = task_k
                                if task_pair_idx < SHARED_TOTAL_PAIRS:
                                    task_beam_kv_idx = (
                                        task_pair_idx // SHARED_Q_TILES_PER_KV_HEAD
                                    )
                                    task_q_tile = (
                                        task_pair_idx
                                        - task_beam_kv_idx * SHARED_Q_TILES_PER_KV_HEAD
                                    )
                                    task_beam_chunk = task_beam_kv_idx // KV_HEADS
                                    task_kv_head_idx = (
                                        task_beam_kv_idx - task_beam_chunk * KV_HEADS
                                    )
                                    task_head_idx = (
                                        task_kv_head_idx * GROUP_SIZE
                                        + task_q_tile * SHARED_Q_HEADS_PER_TILE
                                    )
                                    if side == 0:
                                        shared_v_pair_slot0 = task_pair_idx
                                        shared_v_beam_chunk_slot0 = task_beam_chunk
                                        shared_v_head_slot0 = task_head_idx
                                    if side == 1:
                                        shared_v_pair_slot1 = task_pair_idx
                                        shared_v_beam_chunk_slot1 = task_beam_chunk
                                        shared_v_head_slot1 = task_head_idx
                                    if side == 2:
                                        shared_v_pair_slot2 = task_pair_idx
                                        shared_v_beam_chunk_slot2 = task_beam_chunk
                                        shared_v_head_slot2 = task_head_idx

                                    if task_k == 0:
                                        T.wait_flag("mte3", "v", 3)
                                        T.tile.fill(sumexp, 0.0)
                                        T.tile.fill(sumexp_tiles, 0.0)
                                        T.tile.fill(m_i, -(2.0**30))
                                        T.tile.fill(m_i_tiles, -(2.0**30))
                                    T.pipe_barrier("v")

                                    T.copy(m_i, m_i_prev)
                                    T.copy(m_i, m_i_prev_chunk)
                                    T.pipe_barrier("v")
                                    T.wait_cross_flag(SHARED_QK_READY_BASE)

                                    for row_pipe in T.serial(SHARED_S_ROW_TILES + 1):
                                        if row_pipe < SHARED_S_ROW_TILES:
                                            load_side = row_pipe % SHARED_SOFTMAX_DB
                                            load_row_offset = (
                                                vid * v_p + row_pipe * SHARED_S_ROWS
                                            )
                                            T.wait_flag("v", "mte2", load_side)
                                            T.copy(
                                                s_buf[
                                                    s_slot,
                                                    cid,
                                                    load_row_offset : load_row_offset
                                                    + SHARED_S_ROWS,
                                                    :EFFECTIVE_BLOCK_N,
                                                ],
                                                acc_s_ub[load_side, :, :],
                                            )
                                            T.set_flag("mte2", "v", load_side)

                                        if row_pipe >= 1:
                                            row_tile = row_pipe - 1
                                            compute_side = row_tile % SHARED_SOFTMAX_DB
                                            row_offset = (
                                                vid * v_p + row_tile * SHARED_S_ROWS
                                            )
                                            T.wait_flag("mte2", "v", compute_side)
                                            T.tile.mul(
                                                acc_s_ub[compute_side, :, :],
                                                acc_s_ub[compute_side, :, :],
                                                sm_scale,
                                            )
                                            T.reduce_max(
                                                acc_s_ub[compute_side, :, :],
                                                sumexp_i[0, :, :],
                                                dim=-1,
                                            )
                                            T.pipe_barrier("v")
                                            T.tile.max(
                                                m_i_next[0, :, :],
                                                m_i_tiles[row_tile, :, :],
                                                sumexp_i[0, :, :],
                                            )
                                            T.pipe_barrier("v")
                                            T.copy(
                                                m_i_next[0, :, :],
                                                m_i_tiles[row_tile, :, :],
                                            )
                                            T.pipe_barrier("v")
                                            if task_k > 0:
                                                T.tile.sub(
                                                    m_i_prev_chunk_tiles[
                                                        row_tile, :, :
                                                    ],
                                                    m_i_prev_chunk_tiles[
                                                        row_tile, :, :
                                                    ],
                                                    m_i_tiles[row_tile, :, :],
                                                )
                                                T.tile.exp(
                                                    m_i_prev_chunk_tiles[
                                                        row_tile, :, :
                                                    ],
                                                    m_i_prev_chunk_tiles[
                                                        row_tile, :, :
                                                    ],
                                                )
                                                T.pipe_barrier("v")
                                                T.tile.mul(
                                                    sumexp_tiles[row_tile, :, :],
                                                    sumexp_tiles[row_tile, :, :],
                                                    m_i_prev_chunk_tiles[
                                                        row_tile, :, :
                                                    ],
                                                )
                                                T.pipe_barrier("v")
                                            T.tile.brcb_experiment(
                                                shared_s_brcb_buf,
                                                m_i_next,
                                                SHARED_S_BRCB_REPEAT,
                                                1,
                                                8,
                                            )
                                            T.pipe_barrier("v")
                                            for sub_chunk in T.serial(
                                                EFFECTIVE_BLOCK_N
                                                // SHARED_S_VECTOR_ELEMS
                                            ):
                                                sub_col = (
                                                    sub_chunk * SHARED_S_VECTOR_ELEMS
                                                )
                                                T.tile.row_expand_sub_experiment(
                                                    acc_s_ub[
                                                        compute_side,
                                                        :,
                                                        sub_col : sub_col
                                                        + SHARED_S_VECTOR_ELEMS,
                                                    ],
                                                    acc_s_ub[
                                                        compute_side,
                                                        :,
                                                        sub_col : sub_col
                                                        + SHARED_S_VECTOR_ELEMS,
                                                    ],
                                                    shared_s_brcb_buf,
                                                )
                                            T.tile.exp(
                                                acc_s_ub[compute_side, :, :],
                                                acc_s_ub[compute_side, :, :],
                                            )
                                            T.pipe_barrier("v")
                                            T.reduce_sum(
                                                acc_s_ub[compute_side, :, :],
                                                sumexp_i[0, :, :],
                                                dim=-1,
                                            )
                                            T.pipe_barrier("v")
                                            T.tile.add(
                                                sumexp_tiles[row_tile, :, :],
                                                sumexp_tiles[row_tile, :, :],
                                                sumexp_i[0, :, :],
                                            )
                                            T.pipe_barrier("v")

                                            T.wait_flag("mte3", "v", compute_side)
                                            T.copy(
                                                acc_s_ub[compute_side, :, :],
                                                acc_s_half[compute_side, :, :],
                                            )
                                            T.set_flag("v", "mte2", compute_side)
                                            T.set_flag("v", "mte3", compute_side)
                                            T.wait_flag("v", "mte3", compute_side)
                                            T.copy(
                                                acc_s_half[compute_side, :, :],
                                                p_buf[
                                                    p_slot,
                                                    cid,
                                                    row_offset : row_offset
                                                    + SHARED_S_ROWS,
                                                    :EFFECTIVE_BLOCK_N,
                                                ],
                                            )
                                            T.set_flag("mte3", "v", compute_side)

                                    T.tile.sub(m_i_prev, m_i_prev, m_i)
                                    T.tile.exp(m_i_prev, m_i_prev)
                                    T.pipe_barrier("v")
                                    T.copy(m_i_prev, shared_o_scale_slots[o_slot, :, :])
                                    T.pipe_barrier("v")
                                    if task_k == SHARED_KV_TILES - 1:
                                        T.tile.brcb_experiment(
                                            shared_gm_store_buf,
                                            m_i,
                                            SHARED_GMGL_BRCB_REPEAT,
                                            1,
                                            8,
                                        )
                                        T.tile.brcb_experiment(
                                            shared_gl_store_buf,
                                            sumexp,
                                            SHARED_GMGL_BRCB_REPEAT,
                                            1,
                                            8,
                                        )
                                        T.pipe_barrier("v")
                                        T.set_flag("v", "mte3", 3)
                                        T.wait_flag("v", "mte3", 3)
                                        T.copy(
                                            shared_gm_store_buf[:, :],
                                            shared_gm_merge_ws[
                                                task_beam_chunk * BM
                                                + vid * v_p : task_beam_chunk * BM
                                                + (vid + 1) * v_p,
                                                task_head_idx,
                                                :,
                                            ],
                                        )
                                        T.copy(
                                            shared_gl_store_buf[:, :],
                                            shared_gl_merge_ws[
                                                task_beam_chunk * BM
                                                + vid * v_p : task_beam_chunk * BM
                                                + (vid + 1) * v_p,
                                                task_head_idx,
                                                :,
                                            ],
                                        )
                                        T.set_flag("mte3", "v", 3)
                                    T.set_cross_flag("MTE3", SHARED_V_READY_BASE, 2)

                            if loop_idx >= PRE_LAUNCH and loop_idx < shared_local_total_steps:
                                    rescale_task_idx = loop_idx - PRE_LAUNCH
                                    o_slot = rescale_task_idx % SHARED_O_SLOTS
                                    add_side = delayed_stage
                                    shared_add_pair_cur = SHARED_TOTAL_PAIRS
                                    if add_side == 0:
                                        shared_add_pair_cur = shared_v_pair_slot0
                                        shared_add_k_cur = shared_v_k_slot0
                                        shared_add_beam_chunk_cur = (
                                            shared_v_beam_chunk_slot0
                                        )
                                        shared_add_head_cur = shared_v_head_slot0
                                    if add_side == 1:
                                        shared_add_pair_cur = shared_v_pair_slot1
                                        shared_add_k_cur = shared_v_k_slot1
                                        shared_add_beam_chunk_cur = (
                                            shared_v_beam_chunk_slot1
                                        )
                                        shared_add_head_cur = shared_v_head_slot1
                                    if add_side == 2:
                                        shared_add_pair_cur = shared_v_pair_slot2
                                        shared_add_k_cur = shared_v_k_slot2
                                        shared_add_beam_chunk_cur = (
                                            shared_v_beam_chunk_slot2
                                        )
                                        shared_add_head_cur = shared_v_head_slot2
                                    if shared_add_pair_cur < SHARED_TOTAL_PAIRS:
                                        add_k = shared_add_k_cur
                                        add_beam_chunk = shared_add_beam_chunk_cur
                                        add_head_idx = shared_add_head_cur
                                        T.wait_cross_flag(SHARED_PV_READY_BASE)
                                        T.wait_flag("v", "mte2", SHARED_RESCALE_EVENT)
                                        T.copy(
                                            o_buf[
                                                o_slot,
                                                cid,
                                                vid * v_p : (vid + 1) * v_p,
                                                :,
                                            ],
                                            acc_o_ub[0, :, :],
                                        )
                                        T.set_flag("mte2", "v", SHARED_RESCALE_EVENT)
                                        T.wait_flag("mte2", "v", SHARED_RESCALE_EVENT)
                                        if add_k == 0:
                                            T.wait_flag("mte3", "v", 2)
                                            T.tile.fill(shared_O_ub[0, :, :], 0.0)
                                        else:
                                            T.copy(
                                                shared_o_scale_slots[o_slot, :, :],
                                                m_i_prev,
                                            )
                                            T.pipe_barrier("v")
                                            T.tile.brcb_experiment(
                                                shared_m_broadcast_buf,
                                                m_i_prev,
                                                SHARED_GMGL_BRCB_REPEAT,
                                                1,
                                                8,
                                            )
                                            T.pipe_barrier("v")
                                            for o_chunk in T.serial(
                                                D // SHARED_O_VECTOR_ELEMS
                                            ):
                                                o_col = o_chunk * SHARED_O_VECTOR_ELEMS
                                                T.tile.row_expand_mul_experiment(
                                                    shared_O_ub[
                                                        0,
                                                        :,
                                                        o_col : o_col
                                                        + SHARED_O_VECTOR_ELEMS,
                                                    ],
                                                    shared_O_ub[
                                                        0,
                                                        :,
                                                        o_col : o_col
                                                        + SHARED_O_VECTOR_ELEMS,
                                                    ],
                                                    shared_m_broadcast_buf,
                                                )
                                            T.pipe_barrier("v")
                                        T.tile.add(
                                            shared_O_ub[0, :, :],
                                            shared_O_ub[0, :, :],
                                            acc_o_ub[0, :, :],
                                        )
                                        T.pipe_barrier("v")
                                        T.set_flag("v", "mte2", SHARED_RESCALE_EVENT)
                                        if add_k == SHARED_KV_TILES - 1:
                                            T.set_flag("v", "mte3", 2)
                                            T.wait_flag("v", "mte3", 2)
                                            T.copy(
                                                shared_O_ub[0, :, :D],
                                                shared_O_merge_ws[
                                                    add_beam_chunk * BM
                                                    + vid * v_p : add_beam_chunk * BM
                                                    + (vid + 1) * v_p,
                                                    add_head_idx,
                                                    :D,
                                                ],
                                            )
                                            T.set_flag("mte3", "v", 2)
                                        T.pipe_barrier("v")

                        for final_mte3_event in T.serial(4):
                            T.wait_flag("mte3", "v", final_mte3_event)
                        for final_mte2_event in T.serial(3):
                            T.wait_flag("v", "mte2", final_mte2_event)

            else:
                unshared_cid = cid - SHARED_CORES

                with T.Scope("V"):
                    T.copy(
                        block_mask_ws[
                            vid * UNSHARED_V_ROWS : (vid + 1) * UNSHARED_V_ROWS, :
                        ],
                        unshared_mask,
                    )
                    T.set_flag("mte2", "v", 1)
                    T.wait_flag("mte2", "v", 1)
                    T.set_flag("v", "mte2", 0)

                with T.Scope("C"):
                    for init_mte_event in T.serial(2):
                        T.set_flag("MTE1", "MTE2", init_mte_event)
                    for init_m_event in T.serial(2):
                        T.set_flag("M", "MTE1", 3 + init_m_event)
                        T.set_flag("FIX", "M", 5 + init_m_event)
                    for i in T.serial(TASKS_PER_CORE + PRE_LAUNCH):
                        side = i % TASKQUE_SLOTS
                        prev_side = (i - PRE_LAUNCH) % TASKQUE_SLOTS

                        task_idx = unshared_cid + i * UNSHARED_CORES
                        group_base = task_idx * UNSHARED_GROUPS_PER_TASK
                        beam_base = group_base // KV_HEADS
                        kv_row_base = group_base * DECODE_STEP

                        prev_i = i - PRE_LAUNCH
                        prev_task_idx = unshared_cid + prev_i * UNSHARED_CORES
                        prev_group_base = prev_task_idx * UNSHARED_GROUPS_PER_TASK

                        if i < TASKS_PER_CORE:
                            T.wait_flag("MTE1", "MTE2", 0)
                            q_row_base = beam_base * NUM_HEADS
                            T.copy(
                                Q_unshared_flat[q_row_base : q_row_base + Q_BATCH, :D],
                                q_l1,
                            )
                            mma_slot = side % 2
                            T.set_flag("MTE2", "MTE1", 0)
                            T.wait_flag("MTE2", "MTE1", 0)
                            T.wait_flag("M", "MTE1", 3 + mma_slot)
                            T.copy(q_l1, mma_l0a[mma_slot, :, :])
                            T.copy(
                                K_unshared[kv_row_base : kv_row_base + KV_BATCH, :D],
                                k_qk_l1_0[0:KV_BATCH, :D],
                            )
                            T.set_flag("MTE2", "MTE1", 0)
                            T.wait_flag("MTE2", "MTE1", 0)
                            T.copy(k_qk_l1_0, mma_l0b[mma_slot, :, :], transpose=True)
                            T.wait_flag("FIX", "M", 5 + mma_slot)
                            T.set_flag("MTE1", "MTE2", 0)
                            T.set_flag("MTE1", "M", 3 + mma_slot)
                            T.wait_flag("MTE1", "M", 3 + mma_slot)
                            T.pipe_barrier("m")
                            T.mma(
                                mma_l0a[mma_slot, :, :],
                                mma_l0b[mma_slot, :, :],
                                mma_l0c[mma_slot, :, :],
                                init=True,
                            )
                            T.set_flag("M", "MTE1", 3 + mma_slot)
                            T.set_flag("M", "FIX", 5 + mma_slot)
                            T.wait_flag("M", "FIX", 5 + mma_slot)
                            if i >= TASKQUE_SLOTS:
                                T.wait_cross_flag(UNSHARED_STAGE_RELEASE_BASE + side)
                            T.copy(
                                mma_l0c[mma_slot, :, :], s_buf[side, cid, :, :BLOCK_N]
                            )
                            T.set_flag("FIX", "M", 5 + mma_slot)
                            T.set_flag("FIX", "M", 2)
                            T.wait_flag("FIX", "M", 2)
                            T.set_flag("M", "FIX", 2)
                            T.wait_flag("M", "FIX", 2)
                            T.set_cross_flag("FIX", UNSHARED_QK_READY_BASE + side)

                        if i >= PRE_LAUNCH:
                            T.wait_flag("MTE1", "MTE2", 1)
                            prev_kv_row_base = prev_group_base * DECODE_STEP
                            T.copy(
                                V_unshared[
                                    prev_kv_row_base : prev_kv_row_base + KV_BATCH, :D
                                ],
                                v_l1_u[0:KV_BATCH, :D],
                            )
                            T.wait_cross_flag(UNSHARED_SOFTMAX_READY_BASE + prev_side)
                            T.copy(
                                p_buf[prev_side, cid, 0:Q_BATCH, 0:BLOCK_N],
                                p_l1_u[0:Q_BATCH, 0:BLOCK_N],
                            )
                            mma_slot = 1
                            pv_l0c_slot = 1
                            pv_fix_event = 6
                            T.set_flag("MTE2", "MTE1", 1)
                            T.wait_flag("MTE2", "MTE1", 1)
                            T.wait_flag("M", "MTE1", 3 + mma_slot)
                            T.wait_flag("FIX", "M", pv_fix_event)
                            T.copy(p_l1_u, mma_l0a[mma_slot, :, :])
                            T.copy(v_l1_u, mma_l0b[mma_slot, :, :])
                            T.set_flag("MTE1", "MTE2", 1)
                            T.set_flag("MTE1", "M", 3 + mma_slot)
                            T.wait_flag("MTE1", "M", 3 + mma_slot)
                            T.pipe_barrier("m")
                            T.mma(
                                mma_l0a[mma_slot, :, :],
                                mma_l0b[mma_slot, :, :],
                                mma_l0c[pv_l0c_slot, :, :],
                                init=True,
                            )
                            T.set_flag("M", "MTE1", 3 + mma_slot)
                            T.set_flag("M", "FIX", pv_fix_event)
                            T.wait_flag("M", "FIX", pv_fix_event)

                            if prev_task_idx < TOTAL_UNSHARED_TASKS:
                                prev_beam_base = prev_group_base // KV_HEADS
                                prev_o_row_base = prev_beam_base * NUM_HEADS
                                T.copy(
                                    mma_l0c[pv_l0c_slot, :, :],
                                    unshared_O_flat[
                                        prev_o_row_base : prev_o_row_base + Q_BATCH, :D
                                    ],
                                )
                            T.set_flag("FIX", "M", pv_fix_event)
                            T.set_flag("FIX", "M", 2)
                            T.wait_flag("FIX", "M", 2)
                            T.set_flag("M", "FIX", 2)
                            T.wait_flag("M", "FIX", 2)

                    for final_m_event in T.serial(2):
                        T.wait_flag("M", "MTE1", 3 + final_m_event)
                        T.wait_flag("FIX", "M", 5 + final_m_event)
                    for final_mte_event in T.serial(2):
                        T.wait_flag("MTE1", "MTE2", final_mte_event)

                with T.Scope("V"):
                    for i in T.serial(TASKS_PER_CORE):
                        side = i % TASKQUE_SLOTS
                        task_idx = unshared_cid + i * UNSHARED_CORES
                        group_base = task_idx * UNSHARED_GROUPS_PER_TASK
                        vec_group_base = group_base + vid * UNSHARED_GROUPS_PER_VEC
                        is_valid_task = task_idx < TOTAL_UNSHARED_TASKS

                        T.wait_cross_flag(UNSHARED_QK_READY_BASE + side)

                        T.wait_flag("v", "mte2", 0)
                        T.copy(
                            s_buf[
                                side,
                                cid,
                                vid * UNSHARED_V_ROWS : (vid + 1) * UNSHARED_V_ROWS,
                                0:KV_BATCH,
                            ],
                            unshared_s,
                        )
                        T.set_flag("mte2", "v", 1)
                        T.wait_flag("mte2", "v", 1)
                        if i + TASKQUE_SLOTS < TASKS_PER_CORE:
                            T.set_cross_flag("V", UNSHARED_STAGE_RELEASE_BASE + side)

                        T.tile.add(
                            unshared_s,
                            unshared_s,
                            unshared_mask,
                        )
                        T.tile.mul(unshared_s, unshared_s, sm_scale)
                        T.reduce_max(unshared_s, unshared_m, dim=-1)
                        T.pipe_barrier("v")
                        T.tile.brcb_experiment(
                            unshared_m_broadcast,
                            unshared_m,
                            UNSHARED_BRCB_REPEAT,
                            1,
                            8,
                        )
                        T.pipe_barrier("v")
                        for unshared_sub_chunk in T.serial(
                            KV_BATCH // UNSHARED_VECTOR_ELEMS
                        ):
                            unshared_col = unshared_sub_chunk * UNSHARED_VECTOR_ELEMS
                            T.tile.row_expand_sub_experiment(
                                unshared_s[
                                    :,
                                    unshared_col : unshared_col + UNSHARED_VECTOR_ELEMS,
                                ],
                                unshared_s[
                                    :,
                                    unshared_col : unshared_col + UNSHARED_VECTOR_ELEMS,
                                ],
                                unshared_m_broadcast,
                            )
                        T.tile.exp(unshared_s, unshared_s)
                        T.pipe_barrier("v")
                        T.reduce_sum(unshared_s, unshared_sum, dim=-1)
                        T.pipe_barrier("v")
                        T.copy(unshared_s, unshared_s_half)
                        if i + 1 < TASKS_PER_CORE:
                            T.set_flag("v", "mte2", 0)
                        T.set_flag("v", "mte3", 1)
                        T.wait_flag("v", "mte3", 1)
                        T.copy(
                            unshared_s_half,
                            p_buf[
                                side,
                                cid,
                                vid * UNSHARED_V_ROWS : (vid + 1) * UNSHARED_V_ROWS,
                                0:KV_BATCH,
                            ],
                        )
                        T.set_flag("mte3", "v", 0)
                        T.wait_flag("mte3", "v", 0)
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.set_cross_flag("MTE3", UNSHARED_SOFTMAX_READY_BASE + side)

                        if is_valid_task:
                            T.set_flag("v", "mte3", 1)
                            T.wait_flag("v", "mte3", 1)
                            vec_beam_base = vec_group_base // KV_HEADS
                            T.copy(
                                unshared_m_pack,
                                unshared_gm_ws[
                                    vec_beam_base : vec_beam_base
                                    + UNSHARED_BEAMS_PER_VEC,
                                    0:NUM_HEADS,
                                ],
                            )
                            T.copy(
                                unshared_sum_pack,
                                unshared_gl_ws[
                                    vec_beam_base : vec_beam_base
                                    + UNSHARED_BEAMS_PER_VEC,
                                    0:NUM_HEADS,
                                ],
                            )
                            T.set_flag("mte3", "v", 1)
                            T.wait_flag("mte3", "v", 1)

            T.sync_all()

            merge_worker_idx = cid * VEC_NUM + vid

            with T.Scope("V"):
                for merge_load_event in T.serial(2):
                    T.set_flag("V", "MTE2", merge_load_event)
                for merge_store_event in T.serial(2):
                    T.set_flag("MTE3", "V", 2 + merge_store_event)

                merge_worker_tile_count = (
                    MERGE_TOTAL_TILES + MERGE_WORKERS - 1 - merge_worker_idx
                ) // MERGE_WORKERS
                merge_worker_first_tile = merge_worker_idx

                if merge_worker_tile_count > 0:
                    first_merge_row = merge_worker_first_tile * MERGE_ROW_TILE
                    T.wait_flag("v", "mte2", 0)
                    T.copy(
                        shared_O_merge_flat[
                            first_merge_row : first_merge_row + MERGE_ROW_TILE, :D
                        ],
                        merge_s_O[0, :, :],
                    )
                    T.copy(
                        unshared_O_flat[
                            first_merge_row : first_merge_row + MERGE_ROW_TILE, :D
                        ],
                        merge_u_O[0, :, :],
                    )
                    T.copy(
                        shared_gm_merge_flat[
                            first_merge_row : first_merge_row + MERGE_ROW_TILE,
                            :UB_ALIGN_ELEMS,
                        ],
                        merge_s_gm_load_lane[0, :, :],
                    )
                    T.copy(
                        shared_gl_merge_flat[
                            first_merge_row : first_merge_row + MERGE_ROW_TILE,
                            :UB_ALIGN_ELEMS,
                        ],
                        merge_s_gl_load_lane[0, :, :],
                    )
                    T.copy(
                        unshared_gm_flat[
                            first_merge_row : first_merge_row + MERGE_ROW_TILE
                        ],
                        merge_u_gm[0, :, 0],
                    )
                    T.copy(
                        unshared_gl_flat[
                            first_merge_row : first_merge_row + MERGE_ROW_TILE
                        ],
                        merge_u_gl[0, :, 0],
                    )
                    T.set_flag("mte2", "v", 0)

                for merge_slot in T.serial(MERGE_TILES_PER_WORKER):
                    if merge_slot < merge_worker_tile_count:
                        merge_tile = (
                            merge_worker_first_tile + merge_slot * MERGE_WORKERS
                        )
                        merge_row = merge_tile * MERGE_ROW_TILE
                        merge_buf = merge_slot % 2
                        merge_next_buf = (merge_slot + 1) % 2

                        T.wait_flag("mte2", "v", merge_buf)
                        if merge_slot + 1 < merge_worker_tile_count:
                            next_merge_row = merge_row + MERGE_WORKERS * MERGE_ROW_TILE
                            T.wait_flag("v", "mte2", merge_next_buf)
                            T.copy(
                                shared_O_merge_flat[
                                    next_merge_row : next_merge_row + MERGE_ROW_TILE, :D
                                ],
                                merge_s_O[merge_next_buf, :, :],
                            )
                            T.copy(
                                unshared_O_flat[
                                    next_merge_row : next_merge_row + MERGE_ROW_TILE, :D
                                ],
                                merge_u_O[merge_next_buf, :, :],
                            )
                            T.copy(
                                shared_gm_merge_flat[
                                    next_merge_row : next_merge_row + MERGE_ROW_TILE,
                                    :UB_ALIGN_ELEMS,
                                ],
                                merge_s_gm_load_lane[merge_next_buf, :, :],
                            )
                            T.copy(
                                shared_gl_merge_flat[
                                    next_merge_row : next_merge_row + MERGE_ROW_TILE,
                                    :UB_ALIGN_ELEMS,
                                ],
                                merge_s_gl_load_lane[merge_next_buf, :, :],
                            )
                            T.copy(
                                unshared_gm_flat[
                                    next_merge_row : next_merge_row + MERGE_ROW_TILE
                                ],
                                merge_u_gm[merge_next_buf, :, 0],
                            )
                            T.copy(
                                unshared_gl_flat[
                                    next_merge_row : next_merge_row + MERGE_ROW_TILE
                                ],
                                merge_u_gl[merge_next_buf, :, 0],
                            )
                            T.set_flag("mte2", "v", merge_next_buf)

                        T.reduce_max(
                            merge_s_gm_load_lane[merge_buf, :, :],
                            merge_s_gm,
                            dim=-1,
                        )
                        T.pipe_barrier("v")
                        T.reduce_max(
                            merge_s_gl_load_lane[merge_buf, :, :],
                            merge_s_gl,
                            dim=-1,
                        )
                        T.pipe_barrier("v")

                        T.tile.max(merge_gm, merge_s_gm, merge_u_gm[merge_buf, :, :])
                        T.pipe_barrier("v")
                        T.tile.sub(merge_cor_s, merge_s_gm, merge_gm)
                        T.tile.sub(merge_cor_u, merge_u_gm[merge_buf, :, :], merge_gm)
                        T.pipe_barrier("v")
                        T.tile.exp(merge_cor_s, merge_cor_s)
                        T.tile.exp(merge_cor_u, merge_cor_u)
                        T.pipe_barrier("v")
                        T.tile.mul(merge_s_gl, merge_s_gl, merge_cor_s)
                        T.tile.mul(
                            merge_u_gl[merge_buf, :, :],
                            merge_u_gl[merge_buf, :, :],
                            merge_cor_u,
                        )
                        T.pipe_barrier("v")
                        T.tile.add(merge_gl, merge_s_gl, merge_u_gl[merge_buf, :, :])
                        T.pipe_barrier("v")
                        T.tile.brcb_experiment(
                            merge_s_gm_lane,
                            merge_cor_s,
                            MERGE_BRCB_REPEAT,
                            1,
                            8,
                        )
                        T.pipe_barrier("v")
                        for merge_vec_chunk in T.serial(D // MERGE_FLOAT_VECTOR_ELEMS):
                            merge_vec_col = merge_vec_chunk * MERGE_FLOAT_VECTOR_ELEMS
                            T.tile.row_expand_mul_experiment(
                                merge_s_O[
                                    merge_buf,
                                    :,
                                    merge_vec_col : merge_vec_col
                                    + MERGE_FLOAT_VECTOR_ELEMS,
                                ],
                                merge_s_O[
                                    merge_buf,
                                    :,
                                    merge_vec_col : merge_vec_col
                                    + MERGE_FLOAT_VECTOR_ELEMS,
                                ],
                                merge_s_gm_lane,
                            )
                        T.pipe_barrier("v")
                        T.tile.brcb_experiment(
                            merge_s_gl_lane,
                            merge_cor_u,
                            MERGE_BRCB_REPEAT,
                            1,
                            8,
                        )
                        T.pipe_barrier("v")
                        for merge_vec_chunk in T.serial(D // MERGE_FLOAT_VECTOR_ELEMS):
                            merge_vec_col = merge_vec_chunk * MERGE_FLOAT_VECTOR_ELEMS
                            T.tile.row_expand_mul_experiment(
                                merge_u_O[
                                    merge_buf,
                                    :,
                                    merge_vec_col : merge_vec_col
                                    + MERGE_FLOAT_VECTOR_ELEMS,
                                ],
                                merge_u_O[
                                    merge_buf,
                                    :,
                                    merge_vec_col : merge_vec_col
                                    + MERGE_FLOAT_VECTOR_ELEMS,
                                ],
                                merge_s_gl_lane,
                            )
                        T.pipe_barrier("v")
                        T.tile.add(
                            merge_s_O[merge_buf, :, :],
                            merge_s_O[merge_buf, :, :],
                            merge_u_O[merge_buf, :, :],
                        )
                        T.pipe_barrier("v")
                        T.tile.brcb_experiment(
                            merge_s_gm_lane,
                            merge_gl,
                            MERGE_BRCB_REPEAT,
                            1,
                            8,
                        )
                        T.pipe_barrier("v")
                        for merge_vec_chunk in T.serial(D // MERGE_FLOAT_VECTOR_ELEMS):
                            merge_vec_col = merge_vec_chunk * MERGE_FLOAT_VECTOR_ELEMS
                            T.tile.row_expand_div_experiment(
                                merge_s_O[
                                    merge_buf,
                                    :,
                                    merge_vec_col : merge_vec_col
                                    + MERGE_FLOAT_VECTOR_ELEMS,
                                ],
                                merge_s_O[
                                    merge_buf,
                                    :,
                                    merge_vec_col : merge_vec_col
                                    + MERGE_FLOAT_VECTOR_ELEMS,
                                ],
                                merge_s_gm_lane,
                            )
                        T.pipe_barrier("v")

                        T.wait_flag("mte3", "v", 2 + merge_buf)
                        T.copy(
                            merge_s_O[merge_buf, :, :],
                            merge_out[merge_buf, :, :],
                        )
                        T.pipe_barrier("v")
                        T.set_flag("v", "mte2", merge_buf)

                        T.set_flag("v", "mte3", 2 + merge_buf)
                        T.wait_flag("v", "mte3", 2 + merge_buf)
                        T.copy(
                            merge_out[merge_buf, :, :],
                            Output[merge_row : merge_row + MERGE_ROW_TILE, :D],
                        )
                        T.set_flag("mte3", "v", 2 + merge_buf)

                for merge_store_event in T.serial(2):
                    T.wait_flag("mte3", "v", 2 + merge_store_event)
                for merge_load_event in T.serial(2):
                    T.wait_flag("v", "mte2", merge_load_event)

    return main


def ref_x_attention_decode(
    Q,
    K_shared,
    V_shared,
    K_unshared,
    V_unshared,
    scale,
    num_heads,
    kv_heads,
    decode_step,
    return_parts=False,
):
    Q_f = Q.float()
    K_sf = K_shared.float()
    V_sf = V_shared.float()
    K_uf = K_unshared.float()
    V_uf = V_unshared.float()
    beam_size = Q_f.shape[0]
    group_size = num_heads // kv_heads

    s_O = torch.zeros(beam_size, num_heads, D, dtype=torch.float32, device=Q.device)
    s_O_kernel = torch.zeros_like(s_O)
    s_gm = torch.zeros(beam_size, num_heads, 1, dtype=torch.float32, device=Q.device)
    s_gl = torch.zeros(beam_size, num_heads, 1, dtype=torch.float32, device=Q.device)
    for h in range(num_heads):
        kv_h = h // group_size
        scores = Q_f[:, h, :] @ K_sf[:, kv_h, :].T * scale
        row_max = scores.max(dim=-1, keepdim=True).values
        scores_exp = torch.exp(scores - row_max)
        row_sum = scores_exp.sum(dim=-1, keepdim=True)
        s_O[:, h, :] = scores_exp @ V_sf[:, kv_h, :].float()
        s_gm[:, h, :] = row_max
        s_gl[:, h, :] = row_sum

        running_m = torch.full_like(row_max, -float("inf"))
        running_l = torch.zeros_like(row_sum)
        running_o = torch.zeros(beam_size, D, dtype=torch.float32, device=Q.device)
        for tile_start in range(0, K_sf.shape[0], EFFECTIVE_BLOCK_N):
            tile_end = min(tile_start + EFFECTIVE_BLOCK_N, K_sf.shape[0])
            tile_scores = Q_f[:, h, :] @ K_sf[tile_start:tile_end, kv_h, :].T * scale
            tile_max = tile_scores.max(dim=-1, keepdim=True).values
            next_m = torch.maximum(running_m, tile_max)
            old_scale = torch.exp(running_m - next_m)
            tile_exp = torch.exp(tile_scores - next_m)
            running_o = (
                running_o * old_scale
                + tile_exp.to(Q.dtype).float()
                @ V_sf[tile_start:tile_end, kv_h, :].float()
            )
            running_l = running_l * old_scale + tile_exp.sum(dim=-1, keepdim=True)
            running_m = next_m
        s_O_kernel[:, h, :] = running_o

    u_O = torch.zeros(beam_size, num_heads, D, dtype=torch.float32, device=Q.device)
    u_O_kernel = torch.zeros_like(u_O)
    u_gm = torch.zeros(beam_size, num_heads, 1, dtype=torch.float32, device=Q.device)
    u_gl = torch.zeros(beam_size, num_heads, 1, dtype=torch.float32, device=Q.device)
    for b in range(beam_size):
        for h in range(num_heads):
            kv_h = h // group_size
            scores = Q_f[b : b + 1, h, :] @ K_uf[b, kv_h, :decode_step, :].T * scale
            row_max = scores.max(dim=-1, keepdim=True).values
            scores_exp = torch.exp(scores - row_max)
            row_sum = scores_exp.sum(dim=-1, keepdim=True)
            u_O[b, h, :] = scores_exp @ V_uf[b, kv_h, :decode_step, :].float()
            u_O_kernel[b, h, :] = (
                scores_exp.to(Q.dtype).float() @ V_uf[b, kv_h, :decode_step, :].float()
            )
            u_gm[b, h, :] = row_max
            u_gl[b, h, :] = row_sum

    gm = torch.maximum(s_gm, u_gm)
    cor_s = torch.exp(s_gm - gm)
    cor_u = torch.exp(u_gm - gm)
    gl = s_gl * cor_s + u_gl * cor_u
    final_O = (s_O * cor_s + u_O * cor_u) / gl
    final_kernel = (s_O_kernel * cor_s + u_O_kernel * cor_u) / gl
    if return_parts:
        return final_O.to(torch.bfloat16), {
            "shared_O_high": s_O,
            "shared_O_kernel": s_O_kernel,
            "shared_gm": s_gm,
            "shared_gl": s_gl,
            "unshared_O_high": u_O,
            "unshared_O_kernel": u_O_kernel,
            "unshared_gm": u_gm,
            "unshared_gl": u_gl,
            "final_high": final_O,
            "final_kernel": final_kernel,
        }
    return final_O.to(torch.bfloat16)


if __name__ == "__main__":
    torch.manual_seed(42)
    tilelang.disable_cache()

    if _is_simulator():
        Q = torch.full(
            (BEAM_SIZE, NUM_HEADS, D), 0.1, dtype=torch.bfloat16, device="cpu"
        ).npu()
        K_shared = torch.full(
            (KV_LEN, KV_HEADS, D), 0.1, dtype=torch.bfloat16, device="cpu"
        ).npu()
        V_shared = torch.full(
            (KV_LEN, KV_HEADS, D), 0.1, dtype=torch.bfloat16, device="cpu"
        ).npu()
        K_unshared = torch.full(
            (UNSHARED_BEAM, KV_HEADS, DECODE_STEP, D),
            0.1,
            dtype=torch.bfloat16,
            device="cpu",
        ).npu()
        V_unshared = torch.full(
            (UNSHARED_BEAM, KV_HEADS, DECODE_STEP, D),
            0.1,
            dtype=torch.bfloat16,
            device="cpu",
        ).npu()
    else:
        torch.set_default_device("npu")
        Q = torch.randn(BEAM_SIZE, NUM_HEADS, D, dtype=torch.bfloat16).uniform_(-1, 1)
        K_shared = torch.randn(KV_LEN, KV_HEADS, D, dtype=torch.bfloat16).uniform_(
            -1, 1
        )
        V_shared = torch.randn(KV_LEN, KV_HEADS, D, dtype=torch.bfloat16).uniform_(
            -1, 1
        )
        K_unshared = torch.randn(
            UNSHARED_BEAM, KV_HEADS, DECODE_STEP, D, dtype=torch.bfloat16
        ).uniform_(-1, 1)
        V_unshared = torch.randn(
            UNSHARED_BEAM, KV_HEADS, DECODE_STEP, D, dtype=torch.bfloat16
        ).uniform_(-1, 1)

    Q_unshared_kernel = Q.reshape(MERGE_ROWS, D)
    K_unshared_kernel = K_unshared.reshape(TOTAL_UNSHARED_GROUPS * DECODE_STEP, D)
    V_unshared_kernel = V_unshared.reshape(TOTAL_UNSHARED_GROUPS * DECODE_STEP, D)

    block_mask_ws = torch.full(
        (Q_BATCH, KV_BATCH), -1e10, dtype=torch.float32, device="npu"
    )
    for b in range(UNSHARED_GROUPS_PER_TASK):
        for q_row in range(GROUP_SIZE):
            for kv_col in range(DECODE_STEP):
                block_mask_ws[b * GROUP_SIZE + q_row, b * DECODE_STEP + kv_col] = 0.0

    s_buf = torch.zeros(
        SHARED_S_SLOTS,
        NUM_CORES,
        BM,
        EFFECTIVE_BLOCK_N,
        dtype=torch.float32,
        device="npu",
    )
    p_buf = torch.zeros(
        SHARED_P_SLOTS,
        NUM_CORES,
        BM,
        EFFECTIVE_BLOCK_N,
        dtype=torch.bfloat16,
        device="npu",
    )
    o_buf = torch.zeros(
        SHARED_O_SLOTS,
        NUM_CORES,
        BM,
        D,
        dtype=torch.float32,
        device="npu",
    )
    shared_O_merge_ws = torch.zeros(
        BEAM_SIZE, NUM_HEADS, D, dtype=torch.float32, device="npu"
    )
    shared_gm_merge_ws = torch.zeros(
        BEAM_SIZE, NUM_HEADS, UB_ALIGN_ELEMS, dtype=torch.float32, device="npu"
    )
    shared_gl_merge_ws = torch.zeros(
        BEAM_SIZE, NUM_HEADS, UB_ALIGN_ELEMS, dtype=torch.float32, device="npu"
    )
    unshared_O_ws = torch.zeros(
        UNSHARED_BEAM, NUM_HEADS, D, dtype=torch.float32, device="npu"
    )
    unshared_gm_ws = torch.zeros(
        UNSHARED_BEAM, NUM_HEADS, dtype=torch.float32, device="npu"
    )
    unshared_gl_ws = torch.zeros(
        UNSHARED_BEAM, NUM_HEADS, dtype=torch.float32, device="npu"
    )
    shared_O_merge_flat = shared_O_merge_ws.reshape(MERGE_ROWS, D)
    shared_gm_merge_flat = shared_gm_merge_ws.reshape(MERGE_ROWS, UB_ALIGN_ELEMS)
    shared_gl_merge_flat = shared_gl_merge_ws.reshape(MERGE_ROWS, UB_ALIGN_ELEMS)
    unshared_O_flat = unshared_O_ws.reshape(MERGE_ROWS, D)
    unshared_gm_flat = unshared_gm_ws.reshape(MERGE_ROWS)
    unshared_gl_flat = unshared_gl_ws.reshape(MERGE_ROWS)
    func = x_attention_decode_v_parallel_p0_stack4_3stage()

    print("init successful!")

    for _ in range(WARMUP_RUNS):
        Output_flat = func(
            Q,
            K_shared,
            V_shared,
            s_buf,
            p_buf,
            o_buf,
            K_unshared_kernel,
            V_unshared_kernel,
            Q_unshared_kernel,
            shared_O_merge_ws,
            shared_gm_merge_ws,
            shared_gl_merge_ws,
            unshared_O_ws,
            unshared_gm_ws,
            unshared_gl_ws,
            shared_O_merge_flat,
            shared_gm_merge_flat,
            shared_gl_merge_flat,
            unshared_O_flat,
            unshared_gm_flat,
            unshared_gl_flat,
            block_mask_ws,
        )
    torch.npu.synchronize()

    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    for _ in range(NUM_RUNS):
        Output_flat = func(
            Q,
            K_shared,
            V_shared,
            s_buf,
            p_buf,
            o_buf,
            K_unshared_kernel,
            V_unshared_kernel,
            Q_unshared_kernel,
            shared_O_merge_ws,
            shared_gm_merge_ws,
            shared_gl_merge_ws,
            unshared_O_ws,
            unshared_gm_ws,
            unshared_gl_ws,
            shared_O_merge_flat,
            shared_gm_merge_flat,
            shared_gl_merge_flat,
            unshared_O_flat,
            unshared_gm_flat,
            unshared_gl_flat,
            block_mask_ws,
        )
    end_event.record()
    torch.npu.synchronize()
    Output = Output_flat.reshape(BEAM_SIZE, NUM_HEADS, D)

    ref_O = ref_x_attention_decode(
        Q,
        K_shared,
        V_shared,
        K_unshared,
        V_unshared,
        sm_scale,
        NUM_HEADS,
        KV_HEADS,
        DECODE_STEP,
        return_parts=False,
    )
    torch.testing.assert_close(Output, ref_O, rtol=1e-3, atol=1e-3)
    print("Kernel Output Match!")
