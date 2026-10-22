"""TileLang FP32 attention with resident output and useful-work scheduling.

Edit kernel.tl, then regenerate kernel.py using tools/preprocess.py.
No BAR.V, raw source injection, or external softmax helper.
"""

import argparse

import tilelang
import torch
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout

NUM_CORES = 24
Q_L1 = WS_Q = 256
WS_K = 512
DIM = 128
NUM_STAGES = 3
TRACE_ELEMS = 66560
S_READY, P_READY, O_READY, VECTOR_DONE = 0, 1, 2, 3
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False,
}
COMPILE_FLAGS = ["--cce-auto-sync=off", "-O3"]


@T.macro(hygienic=False)
def vec_capture(src, dst):
    # Diagnostic only. V owns this completion token between observations.
    T.set_flag("V", "MTE3", 7)
    T.wait_flag("V", "MTE3", 7)
    T.copy(src, dst)
    T.set_flag("MTE3", "V", 7)
    T.wait_flag("MTE3", "V", 7)


# Each token protects one complete GM S/P stage. Initially FIX owns FREE.


@T.macro(hygienic=False)
def cube_init():
    T.set_flag("MTE1", "MTE2", 0)
    T.set_flag("MTE1", "MTE2", 1)
    T.set_flag("MTE1", "MTE2", 2)
    T.set_flag("MTE1", "MTE2", 3)
    T.set_flag("MTE1", "MTE2", 4)
    T.set_flag("MTE1", "MTE2", 5)
    T.set_flag("MTE1", "MTE2", 6)
    T.set_flag("M", "MTE1", 0)
    T.set_flag("M", "MTE1", 1)
    T.set_flag("M", "MTE1", 2)
    T.set_flag("M", "MTE1", 3)
    T.set_flag("MTE2", "FIX", 0)
    T.set_flag("MTE2", "FIX", 1)
    T.set_flag("MTE2", "FIX", 2)


@T.macro(hygienic=False)
def cube_begin_q(q_src, q_head, q_begin, q_l1):
    T.wait_flag("MTE1", "MTE2", 0)
    T.copy(q_src[q_head, q_begin : q_begin + 256, :], q_l1)
    T.set_flag("MTE2", "MTE1", 0)
    # Keep the resident query until the final QK packet's final L1 reader.
    T.wait_flag("MTE2", "MTE1", 0)


@T.macro(hygienic=False)
def cube_qk(
    k_src,
    kv_head,
    kv_begin,
    sp,
    core,
    stage,
    last_score,
    q_l1,
    k_l1,
    a_l0,
    b_l0,
    c_half,
    c_wide,
):
    # Previous PV must return this GM stage before any FIX can overwrite P.
    T.wait_flag("MTE2", "FIX", 0 + stage)
    for cube_kp in T.unroll(2):
        cube_k0 = kv_begin + cube_kp * 256
        T.wait_flag("MTE1", "MTE2", 1 + 0)
        T.copy(k_src[kv_head, cube_k0 : cube_k0 + 128, :], k_l1[0, :, :])
        T.set_flag("MTE2", "MTE1", 1 + 0)
        T.wait_flag("M", "MTE1", 2 + 0)
        T.wait_flag("MTE2", "MTE1", 1 + 0)
        T.copy(k_l1[0, :, :], b_l0[0, :, :], transpose=True)
        T.set_flag("MTE1", "MTE2", 1 + 0)
        T.set_flag("MTE1", "M", 2 + 0)
        T.wait_flag("MTE1", "M", 2 + 0)

        for cube_qh in T.unroll(2):
            # Both Q halves survive the transition to the second K pair.
            if cube_kp == 0:
                T.wait_flag("M", "MTE1", 0 + cube_qh)
                T.copy(q_l1[cube_qh * 128 : (cube_qh + 1) * 128, :], a_l0[cube_qh, :, :])
                T.set_flag("MTE1", "M", 0 + cube_qh)
                if T.And(cube_qh == 1, last_score):
                    T.set_flag("MTE1", "MTE2", 0)
                T.wait_flag("MTE1", "M", 0 + cube_qh)

            T.mma(
                a_l0[cube_qh, :, :],
                b_l0[0, :, :],
                c_half[0, :, :],
                init=True,
                unit_flag=3,
            )
            if cube_qh == 1:
                T.set_flag("M", "MTE1", 2 + 0)

            if cube_qh == 0:
                T.wait_flag("MTE1", "MTE2", 1 + 1)
                T.copy(k_src[kv_head, cube_k0 + 128 : cube_k0 + 256, :], k_l1[1, :, :])
                T.set_flag("MTE2", "MTE1", 1 + 1)
                T.wait_flag("M", "MTE1", 2 + 1)
                T.wait_flag("MTE2", "MTE1", 1 + 1)
                T.copy(k_l1[1, :, :], b_l0[1, :, :], transpose=True)
                T.set_flag("MTE1", "MTE2", 1 + 1)
                T.set_flag("MTE1", "M", 2 + 1)
                T.wait_flag("MTE1", "M", 2 + 1)

            T.mma(
                a_l0[cube_qh, :, :],
                b_l0[1, :, :],
                c_half[1, :, :],
                init=True,
                unit_flag=3,
            )
            if cube_qh == 1:
                T.set_flag("M", "MTE1", 2 + 1)
            if cube_kp == 1:
                T.set_flag("M", "MTE1", 0 + cube_qh)
            # These disjoint N halves form one physical [128,256] L0C image.
            T.copy(
                c_wide,
                sp[core, stage, cube_qh * 128 : (cube_qh + 1) * 128, cube_kp * 256 : (cube_kp + 1) * 256],
                unit_flag=3,
            )
    T.set_flag("FIX", "MTE2", 0 + stage)
    # Caller publishes S_READY from FIX after this macro.


@T.macro(hygienic=False)
def cube_prepare_v(v_src, kv_head, kv_begin, v_l1):
    # Call before the Scalar P_READY wait: this immutable input is independent.
    T.wait_flag("MTE1", "MTE2", 5 + 0)
    T.copy(v_src[kv_head, kv_begin : kv_begin + 128, :], v_l1[0, :, :])
    T.set_flag("MTE2", "MTE1", 5 + 0)


@T.macro(hygienic=False)
def cube_pv(
    sp,
    v_src,
    kv_head,
    kv_begin,
    op,
    core,
    stage,
    p_l1,
    v_l1,
    a_l0,
    b_l0,
    c_half,
):
    # Caller prepared V panel 0, then waited for P_READY including old O return.
    T.wait_flag("FIX", "MTE2", 0 + stage)
    for cube_ki in T.unroll(4):
        cube_vs = cube_ki % 2
        cube_k = cube_ki * 128
        for cube_ph in T.unroll(2):
            T.wait_flag("MTE1", "MTE2", 3 + cube_ph)
            T.copy(sp[core, stage, cube_ph * 128 : (cube_ph + 1) * 128, cube_k : cube_k + 128], p_l1[cube_ph, :, :])
            T.set_flag("MTE2", "MTE1", 3 + cube_ph)
        if cube_ki == 3:
            # The release completes after the final GM read, before QK reuses S/P.
            T.set_flag("MTE2", "FIX", 0 + stage)

        T.wait_flag("M", "MTE1", 2 + cube_vs)
        T.wait_flag("MTE2", "MTE1", 5 + cube_vs)
        T.copy(v_l1[cube_vs, :, :], b_l0[cube_vs, :, :])
        T.set_flag("MTE1", "MTE2", 5 + cube_vs)
        T.set_flag("MTE1", "M", 2 + cube_vs)
        if cube_ki < 3:
            # The next V load is independent of the current pair of MADs.
            cube_next_vs = (cube_ki + 1) % 2
            T.wait_flag("MTE1", "MTE2", 5 + cube_next_vs)
            T.copy(v_src[kv_head, kv_begin + cube_k + 128 : kv_begin + cube_k + 256, :], v_l1[cube_next_vs, :, :])
            T.set_flag("MTE2", "MTE1", 5 + cube_next_vs)
        T.wait_flag("MTE1", "M", 2 + cube_vs)
        for cube_ph in T.unroll(2):
            T.wait_flag("M", "MTE1", 0 + cube_ph)
            T.wait_flag("MTE2", "MTE1", 3 + cube_ph)
            T.copy(p_l1[cube_ph, :, :], a_l0[cube_ph, :, :])
            T.set_flag("MTE1", "MTE2", 3 + cube_ph)
            T.set_flag("MTE1", "M", 0 + cube_ph)
            T.wait_flag("MTE1", "M", 0 + cube_ph)
            T.mma(
                a_l0[cube_ph, :, :],
                b_l0[cube_vs, :, :],
                c_half[cube_ph, :, :],
                init=(cube_ki == 0),
                unit_flag=T.if_then_else(cube_ki == 3, 3, 2),
            )
            T.set_flag("M", "MTE1", 0 + cube_ph)
        T.set_flag("M", "MTE1", 2 + cube_vs)

    # Independent M halves require independent FIX geometry, never a wide reshape.
    for cube_ph in T.unroll(2):
        T.copy(
            c_half[cube_ph, :, :],
            op[core, stage, cube_ph * 128 : (cube_ph + 1) * 128, :],
            unit_flag=3,
        )
    # Caller publishes O_READY from FIX after this macro.


@T.macro(hygienic=False)
def cube_finish():
    # Every QK packet must have a PV consumer; every query has last_score=True once.
    T.wait_flag("MTE1", "MTE2", 0)
    T.wait_flag("MTE1", "MTE2", 1)
    T.wait_flag("MTE1", "MTE2", 2)
    T.wait_flag("MTE1", "MTE2", 3)
    T.wait_flag("MTE1", "MTE2", 4)
    T.wait_flag("MTE1", "MTE2", 5)
    T.wait_flag("MTE1", "MTE2", 6)
    T.wait_flag("M", "MTE1", 0)
    T.wait_flag("M", "MTE1", 1)
    T.wait_flag("M", "MTE1", 2)
    T.wait_flag("M", "MTE1", 3)
    T.wait_flag("MTE2", "FIX", 0)
    T.wait_flag("MTE2", "FIX", 1)
    T.wait_flag("MTE2", "FIX", 2)
    # Terminal completion only; no per-packet all-pipe barrier.
    T.barrier_all()


# DMA queues use independent IDs for each directed pipe pair.

# Retained O-coefficient spacing for this schedule, not a generic latency.
VECTOR_RAW_GAP = 32


@T.macro(hygienic=False)
def vec_raw_gap(aux):
    for _raw_gap_i in T.unroll(VECTOR_RAW_GAP):
        T.tile.block_reduce_max(aux, aux, 0, 64, 1, 1, 8)


@T.macro(hygienic=False)
def vec_load_packet(src_packet, input_ub, input_side):
    T.wait_flag("V", "MTE2", 0 + input_side)
    T.copy(src_packet, input_ub[input_side, :])
    T.set_flag("MTE2", "V", 0 + input_side)


@T.macro(hygienic=False)
def vec_store_packet(output_ub, output_side, dst_packet):
    T.wait_flag("V", "MTE3", 0 + output_side)
    T.copy(output_ub[output_side, :], dst_packet)
    T.set_flag("MTE3", "V", 0 + output_side)


@T.macro(hygienic=False)
def vec_softmax_strip(
    kv,
    stage,
    row,
    input_side,
    output_side,
    sm_scale,
    input_ub,
    output_ub,
    score,
    affine_tmp,
    reduce_tmp,
    m,
    m_tile,
    alpha,
    sums,
    coeff_lo,
    coeff_hi,
    aux,
    capture,
    Trace,
    trace_packet,
    query_row,
):
    # row is local to this AIV: 0, 16, ..., 112; stage is the owned ring slot.
    T.wait_flag("MTE2", "V", 0 + input_side)
    T.tile.cast(score, input_ub[input_side, :], "CAST_NONE", 8192)
    T.set_flag("V", "MTE2", 0 + input_side)
    if kv != 0:
        T.copy(m[row : row + 16], alpha[stage, row : row + 16])
    # Positive scaling commutes with max for the finite-input contract. Keep
    # the full FP32 reduction; its output is private until the scaled m update.
    T.reduce_max(score, m_tile, dim=-1, clear=True, tmp=reduce_tmp)
    # Four required 32-repeat scaling chunks replace the short-edge fillers.
    T.tile.mul(score[0:4, :], score[0:4, :], sm_scale)
    T.tile.mul(m_tile, m_tile, sm_scale)
    T.tile.mul(score[4:8, :], score[4:8, :], sm_scale)
    if kv == 0:
        T.copy(m_tile, m[row : row + 16])
    else:
        T.tile.max(m[row : row + 16], m[row : row + 16], m_tile)
    T.tile.mul(score[8:12, :], score[8:12, :], sm_scale)

    if kv == 0:
        # The first O/L update initializes its destination and ignores alpha.
        T.tile.fill(alpha[stage, row : row + 16], 0.0)
    else:
        T.tile.sub(
            alpha[stage, row : row + 16],
            alpha[stage, row : row + 16],
            m[row : row + 16],
        )
    # Explicit BRCB plus the no-tmp row-expand form avoids its built-in BAR.V.
    T.tile.brcb_experiment(coeff_lo, m[row : row + 16], 2, 1, 8)
    T.tile.brcb_experiment(coeff_hi, m[row : row + 16], 2, 1, 8)
    # Final useful chunk separates both coefficient and alpha producers from
    # their consumers; the experiment binds the validated complete schedule.
    T.tile.mul(score[12:16, :], score[12:16, :], sm_scale)
    if kv != 0:
        T.tile.exp(alpha[stage, row : row + 16], alpha[stage, row : row + 16])
    for softmax_half in T.unroll(2):
        for score_window in T.unroll(8):
            # Data windows alternate group halves; the next eight rows shift
            # each coefficient view by eight groups. The condition is static.
            if (softmax_half + score_window) % 2 == 0:
                T.tile.row_expand_sub_experiment(
                    affine_tmp[:, score_window * 64 : (score_window + 1) * 64],
                    score[softmax_half * 8 : (softmax_half + 1) * 8, score_window * 64 : (score_window + 1) * 64],
                    coeff_hi[softmax_half * 8 : (softmax_half + 1) * 8, :],
                )
            else:
                T.tile.row_expand_sub_experiment(
                    affine_tmp[:, score_window * 64 : (score_window + 1) * 64],
                    score[softmax_half * 8 : (softmax_half + 1) * 8, score_window * 64 : (score_window + 1) * 64],
                    coeff_lo[softmax_half * 8 : (softmax_half + 1) * 8, :],
                )
        # The eight SUB windows already separate each element from its EXP read.
        T.tile.exp(score[softmax_half * 8 : (softmax_half + 1) * 8, :], affine_tmp)
        # The complete EXP traversal precedes the next half's tmp reuse.

    T.reduce_sum(
        score,
        sums[stage, row : row + 16],
        dim=-1,
        clear=True,
        tmp=reduce_tmp,
    )
    T.wait_flag("MTE3", "V", 0 + output_side)
    # Required useful conversion follows the sum before anyone consumes z.
    T.tile.cast(output_ub[output_side, :], score, "CAST_NONE", 8192)
    T.set_flag("V", "MTE3", 0 + output_side)
    if capture:
        vec_capture(m[row : row + 16], Trace[trace_packet, query_row : query_row + 16])
        vec_capture(alpha[stage, row : row + 16], Trace[trace_packet, 256 + query_row : 272 + query_row])
        vec_capture(sums[stage, row : row + 16], Trace[trace_packet, 512 + query_row : 528 + query_row])


@T.macro(hygienic=False)
def vec_merge_o_tile(
    kv,
    num_kv,
    stage,
    row,
    input_side,
    output_side,
    input_ub,
    output_ub,
    o,
    l,
    alpha,
    sums,
    coeff,
    aux,
    capture,
    Trace,
    trace_packet,
    query_row,
    score,
    score_flat,
    o_flat,
):
    # o is the separately allocated [64,128] O0 or O1; row is 0 or 64.
    # Caller owns this alpha/sums generation until BOTH O tiles are consumed.
    if kv == 0:
        T.copy(sums[stage, row : row + 64], l[row : row + 64])
    else:
        T.tile.brcb_experiment(coeff, alpha[stage, row : row + 64], 8, 1, 8)
        # sums becomes the new L for these rows. Keep old L until this reads it.
        T.tile.mul_add_dst(
            sums[stage, row : row + 64],
            alpha[stage, row : row + 64],
            l[row : row + 64],
        )
        vec_raw_gap(aux)
        for o_window in T.unroll(2):
            T.tile.row_expand_mul_experiment(
                o[:, o_window * 64 : (o_window + 1) * 64],
                o[:, o_window * 64 : (o_window + 1) * 64],
                coeff,
            )
        # The two row multiplications supply useful work before reading new L.
        T.copy(sums[stage, row : row + 64], l[row : row + 64])

    T.wait_flag("MTE2", "V", 0 + input_side)
    if capture:
        T.tile.cast(score, input_ub[input_side, :], "CAST_NONE", 8192)
        vec_capture(score_flat, Trace[trace_packet, 1024 + query_row * 128 : 1024 + (query_row + 64) * 128])
    if kv == 0:
        T.tile.cast(o, input_ub[input_side, :], "CAST_NONE", 8192)
    else:
        T.tile.axpy(o, input_ub[input_side, :], 1.0)
    T.set_flag("V", "MTE2", 0 + input_side)
    if capture:
        vec_capture(l[row : row + 64], Trace[trace_packet, 768 + query_row : 832 + query_row])
        vec_capture(o_flat, Trace[trace_packet, 33792 + query_row * 128 : 33792 + (query_row + 64) * 128])

    if kv + 1 == num_kv:
        T.tile.brcb_experiment(coeff, l[row : row + 64], 8, 1, 8)
        vec_raw_gap(aux)
        for final_window in T.unroll(2):
            T.tile.row_expand_div_experiment(
                o[:, final_window * 64 : (final_window + 1) * 64],
                o[:, final_window * 64 : (final_window + 1) * 64],
                coeff,
            )
        T.wait_flag("MTE3", "V", 0 + output_side)
        T.tile.cast(output_ub[output_side, :], o, "CAST_NONE", 8192)
        T.set_flag("V", "MTE3", 0 + output_side)


def make_primfunc(batch, seq_len, heads_q, heads_kv, dim, *, kernel_name, key_len=None, capture=False):
    """Build the same dataflow for production and local intermediate checks."""
    nq = seq_len
    nk = seq_len if key_len is None else key_len
    assert batch > 0 and heads_kv > 0 and heads_q % heads_kv == 0
    assert dim == DIM and nq > 0 and nk > 0
    assert nq % Q_L1 == 0 and nk % WS_K == 0
    query_tasks = nq // Q_L1
    kv_blocks = nk // WS_K
    all_tasks = batch * heads_q * query_tasks
    task_base, task_remainder = divmod(all_tasks, NUM_CORES)
    trace_packets = all_tasks * kv_blocks if capture else 1
    trace_elements = TRACE_ELEMS if capture else 1
    scale = DIM**-0.5

    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads_q, nq, DIM), "float16"),
        K: T.Tensor((batch, heads_kv, nk, DIM), "float16"),
        V: T.Tensor((batch, heads_kv, nk, DIM), "float16"),
        Output: T.Tensor((batch, heads_q, nq, DIM), "float16"),
        SP: T.Tensor((NUM_CORES, NUM_STAGES, WS_Q, WS_K), "float16"),
        OP: T.Tensor((NUM_CORES, NUM_STAGES, WS_Q, DIM), "float16"),
        Trace: T.Tensor((trace_packets, trace_elements), "float32"),
    ):
        T.func_attr({"global_symbol": kernel_name})
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            q_view = T.decl_buffer((batch * heads_q, nq, DIM), "float16", data=Q.data)
            k_view = T.decl_buffer((batch * heads_kv, nk, DIM), "float16", data=K.data)
            v_view = T.decl_buffer((batch * heads_kv, nk, DIM), "float16", data=V.data)
            out_flat = T.decl_buffer((batch * heads_q * nq * DIM,), "float16", data=Output.data)
            sp_flat = T.decl_buffer((NUM_CORES * NUM_STAGES * WS_Q * WS_K,), "float16", data=SP.data)
            op_flat = T.decl_buffer((NUM_CORES * NUM_STAGES * WS_Q * DIM,), "float16", data=OP.data)
            q_l1 = T.alloc_L1((256, 128), "float16")
            k_l1 = T.alloc_L1((2, 128, 128), "float16")
            p_l1 = T.alloc_L1((2, 128, 128), "float16")
            v_l1 = T.alloc_L1((2, 128, 128), "float16")
            a_l0 = T.alloc_L0A((2, 128, 128), "float16")
            b_l0 = T.alloc_L0B((2, 128, 128), "float16")
            c_half = T.alloc_L0C((2, 128, 128), "float32")
            c_wide = T.decl_buffer((128, 256), "float32", data=c_half.data, scope="wmma.accumulator")
            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_zn_layout(k_l1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1: make_zn_layout(v_l1),
                }
            )
            T.annotate_address(
                {
                    q_l1: 0,
                    k_l1: 65536,
                    p_l1: 131072,
                    v_l1: 196608,
                    a_l0: 0,
                    b_l0: 0,
                    c_half: 0,
                }
            )
            score = T.alloc_ub((16, 512), "float32")
            o0 = T.alloc_ub((64, 128), "float32")
            input_ub = T.alloc_ub((2, 8192), "float16")
            output_ub = T.alloc_ub((2, 8192), "float16")
            o1 = T.alloc_ub((64, 128), "float32")
            score_flat = T.decl_buffer((8192,), "float32", data=score.data, scope="shared.ub")
            o0_flat = T.decl_buffer((8192,), "float32", data=o0.data, scope="shared.ub")
            o1_flat = T.decl_buffer((8192,), "float32", data=o1.data, scope="shared.ub")
            affine_tmp = T.alloc_ub((8, 512), "float32")
            reduce_tmp = T.alloc_ub((5248,), "uint8")
            m = T.alloc_ub((128,), "float32")
            l = T.alloc_ub((128,), "float32")
            alpha = T.alloc_ub((3, 128), "float32")
            sums = T.alloc_ub((3, 128), "float32")
            coeff = T.alloc_ub((64, 8), "float32")
            # Phase views share the V-only coefficient region with O merge.
            coeff_lo = T.alloc_ub((16, 8), "float32")
            coeff_hi = T.alloc_ub((16, 8), "float32")
            aux = T.alloc_ub((128,), "float32")
            m_tile = T.alloc_ub((16,), "float32")
            T.annotate_address(
                {
                    score: 0,
                    o0: 32768,
                    input_ub: 65536,
                    output_ub: 98304,
                    o1: 131072,
                    affine_tmp: 163840,
                    reduce_tmp: 180224,
                    m: 185472,
                    l: 185984,
                    alpha: 186496,
                    sums: 188032,
                    coeff: 189568,
                    coeff_lo: 189952,
                    coeff_hi: 190720,
                    aux: 191616,
                    m_tile: 192128,
                }
            )
            my_start = cid * task_base + T.min(cid, task_remainder)
            my_tasks = task_base + T.if_then_else(cid < task_remainder, 1, 0)
            packets = my_tasks * kv_blocks

            with T.Scope("C"):
                cube_init()
                for step in T.serial(packets + NUM_STAGES):
                    stage = step % NUM_STAGES
                    if step >= NUM_STAGES:
                        old = step - NUM_STAGES
                        old_task = my_start + old // kv_blocks
                        old_head = old_task // query_tasks
                        old_kv_head = (old_head // heads_q) * heads_kv + (old_head % heads_q) // (heads_q // heads_kv)
                        old_key = (old % kv_blocks) * WS_K
                        cube_prepare_v(v_view, old_kv_head, old_key, v_l1)
                        T.wait_cross_flag(P_READY)
                        cube_pv(SP, v_view, old_kv_head, old_key, OP, cid, stage, p_l1, v_l1, a_l0, b_l0, c_half)
                        T.set_cross_flag("FIX", O_READY)
                    if step < packets:
                        task = my_start + step // kv_blocks
                        head = task // query_tasks
                        q_begin = (task % query_tasks) * Q_L1
                        kv = step % kv_blocks
                        kv_head = (head // heads_q) * heads_kv + (head % heads_q) // (heads_q // heads_kv)
                        if kv == 0:
                            cube_begin_q(q_view, head, q_begin, q_l1)
                        cube_qk(k_view, kv_head, kv * WS_K, SP, cid, stage, kv + 1 == kv_blocks, q_l1, k_l1, a_l0, b_l0, c_half, c_wide)
                        T.set_cross_flag("FIX", S_READY)
                cube_finish()
                T.wait_cross_flag(VECTOR_DONE)

            with T.Scope("V"):
                T.set_flag("V", "MTE2", 0)
                T.set_flag("V", "MTE2", 1)
                T.set_flag("MTE3", "V", 0)
                T.set_flag("MTE3", "V", 1)
                if capture:
                    T.set_flag("MTE3", "V", 7)
                    T.wait_flag("MTE3", "V", 7)
                for step in T.serial(packets + NUM_STAGES):
                    stage = step % NUM_STAGES
                    if step >= NUM_STAGES:
                        old = step - NUM_STAGES
                        old_task = my_start + old // kv_blocks
                        old_kv = old % kv_blocks
                        old_trace = old_task * kv_blocks + old_kv
                        T.wait_cross_flag(O_READY)
                        for half in T.unroll(2):
                            qr = vid * 128 + half * 64
                            op_begin = ((cid * NUM_STAGES + stage) * WS_Q + qr) * DIM
                            vec_load_packet(op_flat[op_begin : op_begin + 8192], input_ub, half)
                            if half == 0:
                                vec_merge_o_tile(
                                    old_kv,
                                    kv_blocks,
                                    stage,
                                    0,
                                    half,
                                    half,
                                    input_ub,
                                    output_ub,
                                    o0,
                                    l,
                                    alpha,
                                    sums,
                                    coeff,
                                    aux,
                                    capture,
                                    Trace,
                                    old_trace,
                                    qr,
                                    score,
                                    score_flat,
                                    o0_flat,
                                )
                            else:
                                vec_merge_o_tile(
                                    old_kv,
                                    kv_blocks,
                                    stage,
                                    64,
                                    half,
                                    half,
                                    input_ub,
                                    output_ub,
                                    o1,
                                    l,
                                    alpha,
                                    sums,
                                    coeff,
                                    aux,
                                    capture,
                                    Trace,
                                    old_trace,
                                    qr,
                                    score,
                                    score_flat,
                                    o1_flat,
                                )
                            if old_kv + 1 == kv_blocks:
                                out_begin = (old_task * WS_Q + qr) * DIM
                                vec_store_packet(output_ub, half, out_flat[out_begin : out_begin + 8192])
                    if step < packets:
                        task = my_start + step // kv_blocks
                        kv = step % kv_blocks
                        trace_packet = task * kv_blocks + kv
                        T.wait_cross_flag(S_READY)
                        for strip in T.serial(8):
                            row = strip * 16
                            qr = vid * 128 + row
                            side = strip % 2
                            sp_begin = ((cid * NUM_STAGES + stage) * WS_Q + qr) * WS_K
                            vec_load_packet(sp_flat[sp_begin : sp_begin + 8192], input_ub, side)
                            vec_softmax_strip(
                                kv,
                                stage,
                                row,
                                side,
                                side,
                                scale,
                                input_ub,
                                output_ub,
                                score,
                                affine_tmp,
                                reduce_tmp,
                                m,
                                m_tile,
                                alpha,
                                sums,
                                coeff_lo,
                                coeff_hi,
                                aux,
                                capture,
                                Trace,
                                trace_packet,
                                qr,
                            )
                            vec_store_packet(output_ub, side, sp_flat[sp_begin : sp_begin + 8192])
                        T.set_cross_flag("MTE3", P_READY)
                if capture:
                    T.set_flag("V", "MTE3", 7)
                    T.wait_flag("V", "MTE3", 7)
                    T.set_flag("MTE3", "V", 7)
                    T.wait_flag("MTE3", "V", 7)
                T.wait_flag("V", "MTE2", 0)
                T.wait_flag("V", "MTE2", 1)
                T.wait_flag("MTE3", "V", 0)
                T.wait_flag("MTE3", "V", 1)
                T.set_cross_flag("MTE3", VECTOR_DONE)

    return main


def flash_attention_fwd(batch, seq_len, heads_q, heads_kv, dim, *, kernel_name="main_kernel", key_len=None, capture=False):
    func = make_primfunc(batch, seq_len, heads_q, heads_kv, dim, kernel_name=kernel_name, key_len=key_len, capture=capture)
    return tilelang.compile(
        func,
        out_idx=[3, 4, 5, 6] if capture else [3],
        workspace_idx=[] if capture else [4, 5, 6],
        pass_configs=PASS_CONFIGS,
        compile_flags=COMPILE_FLAGS,
    )


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
