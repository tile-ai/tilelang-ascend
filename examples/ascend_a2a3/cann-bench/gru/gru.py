"""GRU forward operator (cann-bench level4): fused z/r/n gates + state update.

Implements the recurrent forward pass of torch.nn.GRU in inference mode for the
cann-bench tasks/level4/gru specification: optional stacked layers,
bidirectional processing, batchFirst layout and optional bias. Weights arrive
as per-layer, per-direction TensorLists and the operator returns (y, hn).

Kernel layout:
  pad kernel   : x normalization (batchFirst transpose + batch pad to 16) and
                 F-side padding
  h0pad kernel : h0 -> h0 workspace + fp32 copy (only when h0 is given)
  gpad/wpad    : weight and bias padding to 16-aligned widths; gate-grouped
                 row placement for the weight M-side pad
  xgemm kernel : all-timestep x-side gate contributions, precomputed in
                 parallel (independent of the recurrence)
  seq kernel   : the recurrence itself: one launch per (layer, direction) runs
                 the whole t loop; Cube side does a batched h-side GEMM per
                 timestep, Vector side does the gate math, state update and
                 writeback, synchronized by hand-placed cross-core flags
  scatter/     : y compaction (pad-row trim + optional transpose) and hn trim
  gather

Precision: GEMM accumulates in fp32; the recurrent state h persists through
an fp32 GM ping-pong buffer (no per-step quantization of the carried state);
the GEMM A-side multiplicand is decomposed into a hi/lo dtype pair whose
effective precision is far better than a single dtype (fp16: 22 bits, bf16
multi-layer y relay adds a mid term: 24 bits). sigmoid runs as e^-x/(1+e^-x)
with Newton refinement of the reciprocal, tanh clamps to +/-30 and refines
the reciprocal the same way.

The architecture inherits the measured pitfalls documented for the LSTM
operator in this directory: the auto-CV pipeline must not run inside the
recurrence loop (the seq kernel therefore disables AUTO_SYNC and places its
own flags), L0C may not be written to GM directly, same-dtype casts zero
through T.copy, K-side padding keeps 16-multiple widths, scatter copy widths
stay 16-aligned to avoid tail truncation, and the t=0 lo multiplicand comes
from a dedicated zero buffer to avoid reading stale data.
"""

import torch
import tilelang
import tilelang.ascend.a2a3.language as T


def torch_dtype_to_tl(dtype):
    """Map a torch dtype to the TileLang dtype string used by these kernels."""
    if dtype == torch.float16:
        return "float16"
    elif dtype == torch.bfloat16:
        return "bfloat16"
    elif dtype == torch.float32:
        return "float"
    raise ValueError(f"Unsupported dtype: {dtype}")


M_TILE = 16
VSPLIT = M_TILE // 2

# Fused recurrence: manual synchronization (AUTO_SYNC removed -- the ~58
# PipeBarriers it inserts per timestep dominated the serial-chain cost; every
# in-kernel conversion point is hand-synchronized, see the sync table in the
# _gru_seq_kernel header).
FUSED_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
}

# Helper kernels and xgemm (parallel work, off the serial chain): fully
# automatic (the cross-gate acc WAR hazard is covered by automatic barriers).
PASS_CONFIGS_GRU = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _ceil_div(a, b):
    return (a + b - 1) // b


def _pick_kb(k, h, elem_size):
    """Choose the K block size: it must divide K evenly and keep the
    B tile [H, Kb] within the L0B ping-pong budget.

    The bound comes from the static assert in `gemm_v0`
    (tl_templates/ascend/common.h): on the transposed-B path nTile == N == H
    (the width of a single gate), and each L0 step loads a `(kL0Size x nTile)`
    B sub-block with kL0Size fixed at 128, so `H * 128 * sizeof(elem) <= 64KB`
    (a single step when K <= 128, using the full 64KB). The upper bound on Kb
    is therefore min(128, 65536/(elem*H)).
    """
    kp = ((k + 15) // 16) * 16
    cap = min(128, max(16, (65536 // (elem_size * h) // 16) * 16))
    if kp <= cap:
        return kp
    for d in range(cap, 15, -16):
        if kp % d == 0:
            return d
    return kp


# ----------------------------------------------------------------------------
# Kernel 1: pad / normalization -- x (batchFirst transpose + B/F pad),
# grid = row count in parallel. Pad rows/columns rely on the zero-initialized
# workspace buffers staying zero (nothing ever writes them).
# ----------------------------------------------------------------------------
@tilelang.jit(pass_configs=PASS_CONFIGS_GRU)
def _gru_pad_kernel(S, B, B_pad, F, F_pad, batch_first, dtype="float"):
    ROWS = 16
    nblocks = (S * B + ROWS - 1) // ROWS

    @T.prim_func
    def main(
        X: T.Tensor([S * B * F], dtype),
        X_ws: T.Tensor([S * B_pad * F_pad], dtype),
    ):
        with T.Kernel(nblocks, is_npu=True) as (cid, vid):
            row = T.alloc_ub([F], dtype)
            for i in T.serial(ROWS):
                idx = cid * ROWS + i
                if idx < S * B:
                    s = idx // B
                    r = idx % B
                    if batch_first == 1:
                        T.copy(X[(r * S + s) * F], row)
                    else:
                        T.copy(X[(s * B + r) * F], row)
                    T.copy(row, X_ws[(s * B_pad + r) * F_pad])

    return main


@tilelang.jit(pass_configs=PASS_CONFIGS_GRU)
def _gru_h0pad_kernel(LD, B, B_pad, H, H_pad, dtype="float"):
    is_fp32 = 1 if dtype == "float" else 0

    @T.prim_func
    def main(
        H0: T.Tensor([LD * B * H], dtype),
        H0_ws: T.Tensor([LD * B_pad * H_pad], dtype),
        H0_32: T.Tensor([LD * B_pad * H_pad], "float"),
    ):
        with T.Kernel(LD * B, is_npu=True) as (cid, vid):
            relay_h = T.alloc_ub([H], dtype)
            relay_h32 = T.alloc_ub([H], "float")
            li = cid // B
            r = cid % B
            T.copy(H0[(li * B + r) * H], relay_h)
            # Explicit MTE2->V handshake: AUTO_SYNC inserts no barrier at
            # this conversion point (GM->UB followed by a cast), and the V
            # pipe would read relay_h before it lands (measured: h0_32 came
            # back as garbage without the flag pair).
            T.set_flag("mte2", "v", 0)
            T.wait_flag("mte2", "v", 0)
            T.copy(relay_h, H0_ws[(li * B_pad + r) * H_pad])
            if is_fp32 == 1:
                T.copy(relay_h, relay_h32)
            else:
                T.tile.cast(relay_h32, relay_h, "CAST_NONE", H)
            T.copy(relay_h32, H0_32[(li * B_pad + r) * H_pad])

    return main


# ----------------------------------------------------------------------------
# Kernel 2: weight / bias K-side pad (R rows x C cols -> C_pad cols, grid =
# row count in parallel; row-padding scenarios reuse the same kernel: the
# destination buffer is larger and pad rows stay zero).
# ----------------------------------------------------------------------------
@tilelang.jit(pass_configs=PASS_CONFIGS_GRU)
def _gru_wpad_kernel(R, C, C_pad, dtype="float"):
    ROWS = 8
    nblocks = (R + ROWS - 1) // ROWS

    @T.prim_func
    def main(
        Src: T.Tensor([R * C], dtype),
        Dst: T.Tensor([R * C_pad], dtype),
    ):
        with T.Kernel(nblocks, is_npu=True) as (cid, vid):
            row = T.alloc_ub([C], dtype)
            for i in T.serial(ROWS):
                idx = cid * ROWS + i
                if idx < R:
                    T.copy(Src[idx * C], row)
                    T.copy(row, Dst[idx * C_pad])

    return main


# ----------------------------------------------------------------------------
# Kernel 2b: 3-gate weight M+K double pad (gate-aware) -- src [3H, C] ->
# dst [3H_pad, C_pad]. Source rows [g*H, g*H+H) of gate g must land in
# destination rows [g*H_pad, g*H_pad+H); a 1:1 row copy would let the first
# row of the z gate slip into the r-gate group (the H_pad group boundary and
# the 3H row boundary are offset).
# ----------------------------------------------------------------------------
@tilelang.jit(pass_configs=PASS_CONFIGS_GRU)
def _gru_gpad_kernel(H, C, C_pad, dtype="float"):
    H_pad = ((H + 15) // 16) * 16

    ROWS = 8
    nblocks = (3 * H + ROWS - 1) // ROWS

    @T.prim_func
    def main(
        Src: T.Tensor([3 * H * C], dtype),
        Dst: T.Tensor([3 * H_pad * C_pad], dtype),
    ):
        with T.Kernel(nblocks, is_npu=True) as (cid, vid):
            row = T.alloc_ub([C], dtype)
            for i in T.serial(ROWS):
                idx0 = cid * ROWS + i
                if idx0 < 3 * H:
                    g = idx0 // H
                    dst_r = g * H_pad + idx0 % H
                    T.copy(Src[idx0 * C], row)
                    T.copy(row, Dst[dst_r * C_pad])

    return main


# ----------------------------------------------------------------------------
# Kernel 4a: all-timestep x-side gate precompute (independent of the
#   recurrence -> fully parallel):
#   GX[g, t*B_pad+b, :] = sum_k X[t*B_pad+b, k] * W_ih[g*H+k, :]
# ----------------------------------------------------------------------------
@tilelang.jit(pass_configs=PASS_CONFIGS_GRU)
def _gru_xgemm_kernel(S, B_pad, XW, H, Kb_x, split_x, dtype="float"):
    H3 = 3 * H
    kbx_num = XW // Kb_x
    btiles = S * B_pad // M_TILE

    @T.prim_func
    def xgmain(
        X_hi: T.Tensor([S * B_pad, XW], dtype),
        X_lo: T.Tensor([S * B_pad, XW], dtype),
        X_mid: T.Tensor([S * B_pad, XW], dtype),
        W_ih: T.Tensor([H3, XW], dtype),
        GX: T.Tensor([3, S * B_pad, H], "float"),
    ):
        with T.Kernel(btiles, is_npu=True) as (cid, vid):
            A_L1 = T.alloc_L1([M_TILE, Kb_x], dtype)
            B_L1 = T.alloc_L1([H, Kb_x], dtype)
            acc = T.alloc_L0C([M_TILE, H], "float")
            b0 = cid * M_TILE
            with T.Scope("C"):
                for g in range(3):
                    wr = g * H
                    for kk in T.serial(kbx_num):
                        T.copy(X_hi[b0, kk * Kb_x], A_L1)
                        T.copy(W_ih[wr, kk * Kb_x], B_L1)
                        T.gemm_v0(A_L1, B_L1, acc, transpose_B=True, init=(kk == 0))
                    if split_x >= 2:
                        for kk in T.serial(kbx_num):
                            T.copy(X_lo[b0, kk * Kb_x], A_L1)
                            T.copy(W_ih[wr, kk * Kb_x], B_L1)
                            T.gemm_v0(A_L1, B_L1, acc, transpose_B=True)
                    if split_x >= 3:
                        for kk in T.serial(kbx_num):
                            T.copy(X_mid[b0, kk * Kb_x], A_L1)
                            T.copy(W_ih[wr, kk * Kb_x], B_L1)
                            T.gemm_v0(A_L1, B_L1, acc, transpose_B=True)
                    T.copy(acc, GX[g, b0, 0])

    return xgmain


# Kernel 4b: the recurrence kernel -- the whole t loop inside one launch
# (manual synchronization, see FUSED_PASS_CONFIGS).
#   C scope: one batched GEMM per t: A=[16,P](h) x B=[3H,P](the whole W_hh
#            panel) -> acc[16,3H] (fp16/bf16 append a lo pass to the same
#            accumulator); acc -> Gates[b0, 0:3H] (per-gate column layout)
#   V scope: load batch (MTE2) -> flag 4 -> gate math (V order) -> flag 5 ->
#            store batch (MTE3)
#   Sync table (AUTO_SYNC removed, everything manual):
#     cross-core: C sets FIX(2+t%2) / V waits; V sets MTE3(t%2) / C waits at
#                 the next round
#     C in-core : the gemm_v0 template carries its own MTE2->MTE1 and
#                 MTE1->MTE2(WAR) and M->FIX barriers; before acc is
#                 overwritten by the next t's gemm, FIX->M flag 6 lands the
#                 L0C->GM copy
#     V in-core : after the load batch, mte2->v flag 4 (RAW); before the
#                 store batch, pipe_barrier(v) + v->mte3 flag 5 (load targets
#                 and store sources are disjoint buffers, so the cross-t WAR
#                 is covered by issue order + the flag waits)
#     event ids: cross-core 0-3, local 4/5/6 (leaving 0/1 of the L0AB_EVENT
#                 pair used inside the gemm template free)
#   Performance note: the number of gemm_v0 calls per t dominates the serial
#   chain (each call pays an L0 load + sync latency), so merging the three
#   gates into one GEMM plus the adaptive kL0Size keeps it at 1 call (fp32)
#   or 2 calls (fp16/bf16) per timestep.
@tilelang.jit(pass_configs=FUSED_PASS_CONFIGS)
def _gru_seq_kernel(
    S,
    B,
    B_pad,
    H,
    P,
    DPO,
    DPO_OUT,
    kL0,
    need_ylo,
    need_yt,
    col_off,
    direction,
    direct_y,
    dyq,
    batch_first,
    loskip,
    kconcat,
    hn_rows,
    hn_base,
    hoist_w=1,
    dtype="float",
):
    HN_ROWS = hn_rows
    btiles = B_pad // M_TILE
    is_fp32 = 1 if dtype == "float" else 0

    @T.prim_func
    def main(
        W_hh: T.Tensor([3 * H, P], dtype),
        B_ih: T.Tensor([3 * H], dtype),
        B_hh: T.Tensor([3 * H], dtype),
        H0: T.Tensor([B_pad, P], dtype),
        H_zero: T.Tensor([B_pad, P], dtype),
        H_hi: T.Tensor([2 * B_pad, P], dtype),
        H_lo: T.Tensor([2 * B_pad, P], dtype),
        H_hilo: T.Tensor([2 * B_pad, 2 * P], dtype),
        Hn: T.Tensor([HN_ROWS, H], dtype),
        H0_32: T.Tensor([B_pad, H], "float"),
        H32_pp: T.Tensor([2 * B_pad, H], "float"),
        Gates: T.Tensor([B_pad, 3 * H], "float"),
        GX: T.Tensor([3, S * B_pad, H], "float"),
        Y_t: T.Tensor([S * B_pad, DPO], "float"),
        Y_hi: T.Tensor([S * B_pad, DPO], dtype),
        Y_lo: T.Tensor([S * B_pad, DPO], dtype),
        Y_mid: T.Tensor([S * B_pad, DPO], dtype),
        Y_out: T.Tensor([S * B, DPO_OUT], dtype),
        Y_fin: T.Tensor([S * B_pad, DPO], dtype),
    ):
        with T.Kernel(btiles, is_npu=True) as (cid, vid):
            A_hi_l1 = T.alloc_L1([M_TILE, P], dtype)
            A_lo_l1 = T.alloc_L1([M_TILE, P], dtype)
            A_l1 = T.alloc_L1([M_TILE, 2 * P], dtype)
            B_L1h = T.alloc_L1([3 * H, P], dtype)
            W_panel = T.alloc_L1([3 * H, P], dtype)
            W2_panel = T.alloc_L1([3 * H, 2 * P], dtype)
            acc = T.alloc_L0C([M_TILE, 3 * H], "float")
            b0 = cid * M_TILE

            with T.Scope("C"):
                # W panel hoisted to L1 once (outside the loop); with
                # kconcat the W|W concatenated panel is built directly in L1
                # (two static slice copies, saving a host-side wdup kernel).
                if hoist_w == 1:
                    if kconcat == 1:
                        T.copy(W_hh[0, 0], W2_panel[0 : 3 * H, 0:P])
                        T.copy(W_hh[0, 0], W2_panel[0 : 3 * H, P : 2 * P])
                    else:
                        T.copy(W_hh[0, 0], W_panel)
                for t in T.serial(S):
                    if t > 0:
                        T.wait_cross_flag((t - 1) % 2)
                    hrow = ((t - 1) % 2) * B_pad + b0
                    if kconcat == 1:
                        # hi|lo concatenated along K into a single gemm
                        # (saves one template-round overhead)
                        if t > 0:
                            T.copy(H_hilo[hrow, 0], A_l1)
                        else:
                            T.copy(H0[b0, 0], A_l1[0:M_TILE, 0:P])
                            T.copy(H_zero[b0, 0], A_l1[0:M_TILE, P : 2 * P])
                        T.gemm_v0(
                            A_l1,
                            W2_panel,
                            acc,
                            transpose_B=True,
                            init=True,
                            kL0Size=kL0,
                        )
                    else:
                        if t > 0:
                            T.copy(H_hi[hrow, 0], A_hi_l1)
                            if is_fp32 == 0 and loskip == 0:
                                T.copy(H_lo[hrow, 0], A_lo_l1)
                        else:
                            T.copy(H0[b0, 0], A_hi_l1)
                            if is_fp32 == 0 and loskip == 0:
                                T.copy(H_zero[b0, 0], A_lo_l1)
                        if hoist_w == 1:
                            T.gemm_v0(
                                A_hi_l1,
                                W_panel,
                                acc,
                                transpose_B=True,
                                init=True,
                                kL0Size=kL0,
                            )
                            if is_fp32 == 0 and loskip == 0:
                                T.gemm_v0(A_lo_l1, W_panel, acc, transpose_B=True, kL0Size=kL0)
                        else:
                            T.copy(W_hh[0, 0], B_L1h)
                            T.gemm_v0(
                                A_hi_l1,
                                B_L1h,
                                acc,
                                transpose_B=True,
                                init=True,
                                kL0Size=kL0,
                            )
                            if is_fp32 == 0 and loskip == 0:
                                T.gemm_v0(A_lo_l1, B_L1h, acc, transpose_B=True, kL0Size=kL0)
                    T.copy(acc, Gates[b0, 0])
                    T.set_flag("fix", "m", 6)
                    T.wait_flag("fix", "m", 6)
                    T.set_cross_flag("FIX", 2 + (t % 2))

            with T.Scope("V"):
                z2 = T.alloc_ub([VSPLIT, H], "float")
                r2 = T.alloc_ub([VSPLIT, H], "float")
                n2 = T.alloc_ub([VSPLIT, H], "float")
                hp2 = T.alloc_ub([VSPLIT, H], "float")
                hnew2 = T.alloc_ub([VSPLIT, H], "float")
                sc1 = T.alloc_ub([VSPLIT, H], "float")
                sc2 = T.alloc_ub([VSPLIT, H], "float")
                sc3 = T.alloc_ub([VSPLIT, H], "float")
                sc4 = T.alloc_ub([VSPLIT, H], "float")
                sc5 = T.alloc_ub([VSPLIT, H], "float")
                bi_z = T.alloc_ub([VSPLIT, H], "float")
                bi_r = T.alloc_ub([VSPLIT, H], "float")
                bi_n = T.alloc_ub([VSPLIT, H], "float")
                bh_n = T.alloc_ub([VSPLIT, H], "float")
                gx_z = T.alloc_ub([VSPLIT, H], "float")
                gx_r = T.alloc_ub([VSPLIT, H], "float")
                gx_n = T.alloc_ub([VSPLIT, H], "float")
                h2 = T.alloc_ub([VSPLIT, H], dtype)
                h32 = T.alloc_ub([VSPLIT, H], "float")
                lo32 = T.alloc_ub([VSPLIT, H], "float")
                lo2 = T.alloc_ub([VSPLIT, H], dtype)
                mid2 = T.alloc_ub([VSPLIT, H], dtype)
                mid32 = T.alloc_ub([VSPLIT, H], "float")
                b_z_ih = T.alloc_ub([H], dtype)
                b_z_hh = T.alloc_ub([H], dtype)
                b_r_ih = T.alloc_ub([H], dtype)
                b_r_hh = T.alloc_ub([H], dtype)
                b_n_ih = T.alloc_ub([H], dtype)
                b_n_hh = T.alloc_ub([H], dtype)
                bh1 = T.alloc_ub([H], "float")
                bh2 = T.alloc_ub([H], "float")
                brow = T.alloc_ub([H], "float")

                # ---- bias precompute (one load batch -> one flag -> V math)
                T.copy(B_ih[1 * H], b_z_ih)
                T.copy(B_hh[1 * H], b_z_hh)
                T.copy(B_ih[0 * H], b_r_ih)
                T.copy(B_hh[0 * H], b_r_hh)
                T.copy(B_ih[2 * H], b_n_ih)
                T.copy(B_hh[2 * H], b_n_hh)
                T.set_flag("mte2", "v", 4)
                T.wait_flag("mte2", "v", 4)
                if is_fp32 == 1:
                    T.copy(b_z_ih, bh1)
                    T.copy(b_z_hh, bh2)
                else:
                    T.tile.cast(bh1, b_z_ih, "CAST_NONE", H)
                    T.tile.cast(bh2, b_z_hh, "CAST_NONE", H)
                T.tile.add(brow, bh1, bh2)
                T.tile.broadcast(bi_z, brow)
                if is_fp32 == 1:
                    T.copy(b_r_ih, bh1)
                    T.copy(b_r_hh, bh2)
                else:
                    T.tile.cast(bh1, b_r_ih, "CAST_NONE", H)
                    T.tile.cast(bh2, b_r_hh, "CAST_NONE", H)
                T.tile.add(brow, bh1, bh2)
                T.tile.broadcast(bi_r, brow)
                if is_fp32 == 1:
                    T.copy(b_n_ih, bh1)
                else:
                    T.tile.cast(bh1, b_n_ih, "CAST_NONE", H)
                T.tile.broadcast(bi_n, bh1)
                if is_fp32 == 1:
                    T.copy(b_n_hh, bh1)
                else:
                    T.tile.cast(bh1, b_n_hh, "CAST_NONE", H)
                T.tile.broadcast(bh_n, bh1)

                for t in T.serial(S):
                    T.wait_cross_flag(2 + (t % 2))
                    t_row = t if direction == 0 else (S - 1 - t)
                    rbase = b0 + vid * VSPLIT
                    # ---- load batch (MTE2) ----
                    T.copy(Gates[rbase : rbase + VSPLIT, H : 2 * H], z2)
                    T.copy(
                        GX[
                            1,
                            t_row * B_pad + rbase : t_row * B_pad + rbase + VSPLIT,
                            0:H,
                        ],
                        gx_z,
                    )
                    T.copy(Gates[rbase : rbase + VSPLIT, 0:H], r2)
                    T.copy(
                        GX[
                            0,
                            t_row * B_pad + rbase : t_row * B_pad + rbase + VSPLIT,
                            0:H,
                        ],
                        gx_r,
                    )
                    T.copy(Gates[rbase : rbase + VSPLIT, 2 * H : 3 * H], n2)
                    T.copy(
                        GX[
                            2,
                            t_row * B_pad + rbase : t_row * B_pad + rbase + VSPLIT,
                            0:H,
                        ],
                        gx_n,
                    )
                    if t > 0:
                        T.copy(
                            H32_pp[
                                ((t - 1) % 2) * B_pad + rbase : ((t - 1) % 2) * B_pad + rbase + VSPLIT,
                                0:H,
                            ],
                            hp2,
                        )
                    else:
                        T.copy(H0_32[rbase : rbase + VSPLIT, 0:H], hp2)
                    T.set_flag("mte2", "v", 4)
                    T.wait_flag("mte2", "v", 4)
                    # ---- z gate: sigmoid(Gates[z] + GX[1] + bz); torch
                    # row order is [r; z; n], so z lives in slab 1.
                    # Newton x1 (vrec 2e-3 -> ~4e-6; invisible under the fp16
                    # quantization of 4.9e-4, 8x margin under the fp32
                    # threshold of 1.22e-4).
                    T.tile.add(z2, z2, gx_z)
                    T.tile.add(z2, z2, bi_z)
                    T.tile.fill(sc3, 0.0)
                    T.tile.sub(z2, sc3, z2)
                    T.tile.exp(sc4, z2)
                    T.tile.add(sc5, sc4, 1.0)
                    T.tile.reciprocal(sc4, sc5)
                    T.tile.mul(sc2, sc5, sc4)
                    T.tile.fill(sc1, 2.0)
                    T.tile.sub(sc1, sc1, sc2)
                    T.tile.mul(z2, sc4, sc1)
                    # ---- r gate: sigmoid(Gates[r] + GX[0] + br) ----
                    T.tile.add(r2, r2, gx_r)
                    T.tile.add(r2, r2, bi_r)
                    T.tile.fill(sc3, 0.0)
                    T.tile.sub(r2, sc3, r2)
                    T.tile.exp(sc4, r2)
                    T.tile.add(sc5, sc4, 1.0)
                    T.tile.reciprocal(sc4, sc5)
                    T.tile.mul(sc2, sc5, sc4)
                    T.tile.fill(sc1, 2.0)
                    T.tile.sub(sc1, sc1, sc2)
                    T.tile.mul(r2, sc4, sc1)
                    # ---- n gate: tanh(GX[2] + b_ih_n + r * (Gates[n]
                    #      + b_hh_n)) ----
                    T.tile.add(n2, n2, bh_n)
                    T.tile.mul(n2, r2, n2)
                    T.tile.add(n2, n2, gx_n)
                    T.tile.add(n2, n2, bi_n)
                    T.tile.max(n2, n2, -30.0)
                    T.tile.min(n2, n2, 30.0)
                    # tanh = (e^x - e^-x) * recip(e^x + e^-x), reciprocal
                    # refined by 2 Newton steps (one step leaves a 4e-6
                    # relative error, which is the per-step epsilon
                    # bottleneck; two steps reach 1.6e-11)
                    T.tile.fill(sc1, 0.0)
                    T.tile.sub(sc2, sc1, n2)
                    T.tile.exp(n2, n2)
                    T.tile.exp(sc2, sc2)
                    T.tile.sub(sc1, n2, sc2)
                    T.tile.add(sc4, n2, sc2)
                    T.tile.reciprocal(sc3, sc4)
                    T.tile.mul(sc2, sc4, sc3)
                    T.tile.fill(sc5, 2.0)
                    T.tile.sub(sc5, sc5, sc2)
                    T.tile.mul(sc3, sc3, sc5)
                    T.tile.mul(n2, sc1, sc3)
                    # ---- state update: h_new = n + z * (h_prev32 - n) ----
                    T.tile.sub(sc1, hp2, n2)
                    T.tile.mul(sc1, z2, sc1)
                    T.tile.add(hnew2, n2, sc1)
                    # ---- relay quantization (V order; with loskip the A-side
                    # single-precision split is skipped) ----
                    if is_fp32 == 1:
                        T.copy(hnew2, h2)
                        T.copy(hnew2, h32)
                        T.tile.fill(lo2, 0.0)
                    else:
                        T.tile.cast(h2, hnew2, "CAST_RINT", VSPLIT * H)
                        if loskip == 1:
                            T.tile.fill(lo2, 0.0)
                        else:
                            T.tile.cast(h32, h2, "CAST_NONE", VSPLIT * H)
                            T.tile.sub(lo32, hnew2, h32)
                            T.tile.cast(lo2, lo32, "CAST_RINT", VSPLIT * H)
                        if need_ylo == 2:
                            T.tile.cast(mid32, lo2, "CAST_NONE", VSPLIT * H)
                            T.tile.sub(lo32, lo32, mid32)
                            T.tile.cast(mid2, lo32, "CAST_RINT", VSPLIT * H)
                    # ---- store batch (MTE3) ----
                    T.pipe_barrier("v")
                    T.set_flag("v", "mte3", 5)
                    T.wait_flag("v", "mte3", 5)
                    T.copy(
                        hnew2,
                        H32_pp[
                            (t % 2) * B_pad + rbase : (t % 2) * B_pad + rbase + VSPLIT,
                            0:H,
                        ],
                    )
                    if kconcat == 1:
                        # hi|lo land in the concatenated buffer (the A side
                        # of the next t's single gemm)
                        T.copy(
                            h2,
                            H_hilo[
                                (t % 2) * B_pad + rbase : (t % 2) * B_pad + rbase + VSPLIT,
                                0:P,
                            ],
                        )
                        T.copy(
                            lo2,
                            H_hilo[
                                (t % 2) * B_pad + rbase : (t % 2) * B_pad + rbase + VSPLIT,
                                P : 2 * P,
                            ],
                        )
                    else:
                        T.copy(
                            h2,
                            H_hi[
                                (t % 2) * B_pad + rbase : (t % 2) * B_pad + rbase + VSPLIT,
                                0:P,
                            ],
                        )
                        T.copy(
                            lo2,
                            H_lo[
                                (t % 2) * B_pad + rbase : (t % 2) * B_pad + rbase + VSPLIT,
                                0:P,
                            ],
                        )
                    if t == S - 1:
                        # Final-step state h written straight into hn_ws
                        # (no separate hncopy kernel). The destination must
                        # carry an explicit row range -- a bare anchor lets
                        # the copy shape degrade to a whole-buffer
                        # descriptor (measured: a (512,1,32) shape wrote
                        # only 1 row).
                        T.copy(h2, Hn[hn_base + rbase : hn_base + rbase + VSPLIT, 0:H])
                    if need_yt == 1:
                        T.copy(
                            hnew2,
                            Y_t[
                                t_row * B_pad + rbase : t_row * B_pad + rbase + VSPLIT,
                                col_off : col_off + H,
                            ],
                        )
                    if direct_y == 1:
                        # Final layer writes the output directly (no scatter
                        # kernel). The block span [rbase, rbase+VSPLIT) must
                        # not cross the B row boundary (it would corrupt the
                        # neighbouring timestep row), so this is enabled
                        # only when !batchFirst and B%16==0 (in batchFirst
                        # the block maps onto rows strided by S, so a
                        # direct block write is impossible).
                        T.copy(
                            h2,
                            Y_out[
                                t_row * B + rbase : t_row * B + rbase + VSPLIT,
                                col_off : col_off + H,
                            ],
                        )
                    if dyq == 1:
                        # Final-layer quantized state lands in the dedicated
                        # Y_fin buffer (Y_hi cannot be reused -- in a
                        # bidirectional multi-layer run the d=0 store would
                        # clobber the relay that the d=1 xgemm has not read
                        # yet). Uniform S-major row order; the host-side
                        # compaction trims the pad rows (and transposes).
                        T.copy(
                            h2,
                            Y_fin[
                                t_row * B_pad + rbase : t_row * B_pad + rbase + VSPLIT,
                                col_off : col_off + H,
                            ],
                        )
                    if need_ylo >= 1:
                        T.copy(
                            h2,
                            Y_hi[
                                t_row * B_pad + rbase : t_row * B_pad + rbase + VSPLIT,
                                col_off : col_off + H,
                            ],
                        )
                        T.copy(
                            lo2,
                            Y_lo[
                                t_row * B_pad + rbase : t_row * B_pad + rbase + VSPLIT,
                                col_off : col_off + H,
                            ],
                        )
                    if need_ylo == 2:
                        T.copy(
                            mid2,
                            Y_mid[
                                t_row * B_pad + rbase : t_row * B_pad + rbase + VSPLIT,
                                col_off : col_off + H,
                            ],
                        )
                    T.set_cross_flag("MTE3", t % 2)

    return main


# ----------------------------------------------------------------------------
# Kernel 6: y scatter -- y_ws[S*B_pad, DPO_SRC] (fp32) -> y_out (optional
#   batchFirst, pad-row trim, dtype cast), grid = row count in parallel.
#   direct=1: with DPO a multiple of 16 the kernel writes the final
#   [S*B, DPO] layout directly (no slice + contiguous); direct=0: padded
#   output fallback (copy width stays DPO_SRC, a multiple of 16, to avoid
#   tail truncation).
# ----------------------------------------------------------------------------
@tilelang.jit(pass_configs=PASS_CONFIGS_GRU)
def _gru_scatter_kernel(S, B, B_pad, DPO, DPO_SRC, batch_first, direct, dtype="float"):
    is_fp32 = 1 if dtype == "float" else 0
    W_OUT = DPO if direct == 1 else DPO_SRC
    CPY = DPO if direct == 1 else DPO_SRC

    ROWS = 8
    nblocks = (S * B + ROWS - 1) // ROWS

    @T.prim_func
    def main(
        Y_ws: T.Tensor([S * B_pad * DPO_SRC], "float"),
        Y_out: T.Tensor([S * B * W_OUT], dtype),
    ):
        with T.Kernel(nblocks, is_npu=True) as (cid, vid):
            row32 = T.alloc_ub([CPY], "float")
            row = T.alloc_ub([CPY], dtype)
            for i in T.serial(ROWS):
                idx = cid * ROWS + i
                if idx < S * B:
                    s = idx // B
                    r = idx % B
                    T.copy(Y_ws[(s * B_pad + r) * DPO_SRC], row32)
                    if is_fp32 == 1:
                        T.copy(row32, row)
                    else:
                        T.tile.cast(row, row32, "CAST_RINT", CPY)
                    if batch_first == 0:
                        T.copy(row, Y_out[(s * B + r) * W_OUT])
                    else:
                        T.copy(row, Y_out[(r * S + s) * W_OUT])

    return main


# ----------------------------------------------------------------------------
# Kernel 7: hn final trim -- padded workspace -> output (B rows,
#   W_OUT -> W_DST compaction), grid = row count in parallel.
# ----------------------------------------------------------------------------
@tilelang.jit(pass_configs=PASS_CONFIGS_GRU)
def _gru_gather_kernel(LD, B, B_pad, W_OUT, W_DST, dtype="float"):
    ROWS = 8
    nblocks = (LD * B + ROWS - 1) // ROWS

    @T.prim_func
    def main(
        Src: T.Tensor([LD * B_pad * W_OUT], dtype),
        Dst: T.Tensor([LD * B * W_DST], dtype),
    ):
        with T.Kernel(nblocks, is_npu=True) as (cid, vid):
            relay = T.alloc_ub([W_DST], dtype)
            for i in T.serial(ROWS):
                idx = cid * ROWS + i
                if idx < LD * B:
                    li = idx // B
                    r = idx % B
                    T.copy(Src[(li * B_pad + r) * W_OUT], relay)
                    T.copy(relay, Dst[(li * B + r) * W_DST])

    return main


# ----------------------------------------------------------------------------
# Python interface (signature matches the tasks/level4/gru proto schema; the
# harness builds the params after the golden signature and calls
# func(**params)).
# ----------------------------------------------------------------------------
# Workspace shape-keyed cache: zero-filled once, reused across calls (the
# harness timing metric is the sum of kernel durations, and every
# torch.zeros would emit a ZerosLike fill kernel, 17-35us per call of pure
# overhead). Reuse is safe because every buffer's live region is written
# before it is read on each call (GEMM row isolation + ping-pong
# write-before-read), and pad rows/columns only rely on the zero
# initialization and are never written. Output tensors are not cached (the
# harness may hold a reference to an output across calls); they come from
# torch.empty (no fill kernel).
_WS_CACHE = {}
_WS_CACHE_ORDER = []


def _ws(key, numel, dtype, dev):
    k = (key, numel, dtype, str(dev))
    buf = _WS_CACHE.get(k)
    if buf is None:
        buf = torch.zeros(numel, dtype=dtype, device=dev)
        _WS_CACHE[k] = buf
        _WS_CACHE_ORDER.append(k)
        if len(_WS_CACHE_ORDER) > 128:
            _WS_CACHE.pop(_WS_CACHE_ORDER.pop(0), None)
    return buf


def gru(
    x: torch.Tensor,
    weight_ih,
    weight_hh,
    inputSize: int,
    hiddenSize: int,
    numLayers: int = 1,
    bias: bool = True,
    batchFirst: bool = False,
    dropout: float = 0.0,
    bidirectional: bool = False,
    bias_ih=None,
    bias_hh=None,
    h0: torch.Tensor = None,
):
    """GRU forward pass (torch.nn.GRU eval-mode semantics; inter-layer
    dropout is a no-op in inference)."""
    del dropout  # golden eval() semantics -> no-op
    D = 2 if bidirectional else 1
    L = int(numLayers)
    H = int(hiddenSize)
    F = int(inputSize)
    H3 = 3 * H
    tl_dtype = torch_dtype_to_tl(x.dtype)
    dev = x.device

    if batchFirst:
        B, S = int(x.shape[0]), int(x.shape[1])
    else:
        S, B = int(x.shape[0]), int(x.shape[1])

    B_pad = _ceil_div(B, M_TILE) * M_TILE
    elem = 4 if tl_dtype == "float" else 2
    F_pad = _ceil_div(F, 16) * 16
    H_pad = _ceil_div(H, 16) * 16
    DPO = D * H
    DPO_pad = _ceil_div(DPO, 16) * 16
    # The N side is uniformly H_pad (GEMM B tile / acc alignment; an
    # unaligned N combined with the dual acc fragment was measured to trip
    # an L0C-layout aicore crash).
    Kb_h = _pick_kb(H_pad, H_pad, elem)
    LD = L * D
    assert B_pad >= M_TILE and H >= 16 and Kb_h >= 8, f"GEMM fract violation: B_pad={B_pad} H={H} Kb_h={Kb_h}"
    if tl_dtype != "float":
        assert Kb_h >= 16, "fp16/bf16 K >= 16"
    if D == 2:
        assert H % 16 == 0, (
            "bidirectional requires hiddenSize % 16 == 0 "
            "(col_off = d*H needs 16-alignment; every bidirectional case in "
            "the cann-bench gru suite satisfies this)"
        )
    has_h0 = 1 if h0 is not None else 0
    ck = (S, B, F, H, L, D, tl_dtype, has_h0)

    def _as_list(v, fill_shape):
        if v is None:
            return [torch.zeros(fill_shape, dtype=x.dtype, device=dev) for _ in range(LD)]
        out = list(v)
        assert len(out) == LD, f"TensorList length {len(out)} != L*D {LD}"
        return out

    w_ih = _as_list(weight_ih, (H3, F))
    w_hh = _as_list(weight_hh, (H3, H))
    have_bias = bias and bias_ih is not None and bias_hh is not None
    if have_bias:
        b_ih = _as_list(bias_ih, (H3,))
        b_hh = _as_list(bias_hh, (H3,))

    # ---- weight / bias padding (zero-copy reshape when aligned; otherwise a
    # cached buffer whose content is refreshed each call). The weight M-side
    # pad must place rows by gate group (gpad): a 1:1 row copy would offset
    # the gate groups.
    w_ih_p = [None] * LD
    w_hh_p = [None] * LD
    b_ih_p = [None] * LD
    b_hh_p = [None] * LD
    b_zero = None
    for li in range(LD):
        in_w = F if li < D else DPO
        in_w_pad = F_pad if li < D else DPO_pad
        if H_pad == H and in_w == in_w_pad:
            w_ih_p[li] = w_ih[li].reshape(H3, in_w)
        else:
            buf = _ws(("wih", li) + ck, 3 * H_pad * in_w_pad, x.dtype, dev)  # codespell:ignore wih
            k_wx = _gru_gpad_kernel(H, in_w, in_w_pad, tl_dtype)
            k_wx(w_ih[li].reshape(-1), buf.reshape(-1))
            w_ih_p[li] = buf.view(3 * H_pad, in_w_pad)
        if H_pad == H:
            w_hh_p[li] = w_hh[li].reshape(H3, H)
        else:
            buf = _ws(("whh", li) + ck, 3 * H_pad * H_pad, x.dtype, dev)
            k_wh = _gru_gpad_kernel(H, H, H_pad, tl_dtype)
            k_wh(w_hh[li].reshape(-1), buf.reshape(-1))
            w_hh_p[li] = buf.view(3 * H_pad, H_pad)
        if H_pad == H:
            if have_bias:
                b_ih_p[li] = b_ih[li].reshape(-1)
                b_hh_p[li] = b_hh[li].reshape(-1)
            else:
                if b_zero is None:
                    b_zero = _ws(("bzero",) + ck, 3 * H_pad, x.dtype, dev)
                b_ih_p[li] = b_zero
                b_hh_p[li] = b_zero
        else:
            b_ih_p[li] = _ws(("bih", li) + ck, 3 * H_pad, x.dtype, dev)
            b_hh_p[li] = _ws(("bhh", li) + ck, 3 * H_pad, x.dtype, dev)
            if have_bias:
                k_bpad = _gru_wpad_kernel(3, H, H_pad, tl_dtype)
                k_bpad(b_ih[li].reshape(-1), b_ih_p[li].reshape(-1))
                k_bpad(b_hh[li].reshape(-1), b_hh_p[li].reshape(-1))

    # ---- workspaces (cache-reused) ----
    y_a = _ws(("ya",) + ck, S * B_pad * DPO_pad, torch.float32, dev)
    y_b = _ws(("yb",) + ck, S * B_pad * DPO_pad, torch.float32, dev)
    y_bufs = [y_a, y_b]
    h0_ws = _ws(("h0ws",) + ck, LD * B_pad * H_pad, x.dtype, dev)
    h0_32_ws = _ws(("h032",) + ck, LD * B_pad * H_pad, torch.float32, dev)
    h_hi_ws = _ws(("hhi",) + ck, D * 2 * B_pad * H_pad, x.dtype, dev)
    h_lo_ws = _ws(("hlo",) + ck, D * 2 * B_pad * H_pad, x.dtype, dev)
    h32_ws = _ws(("h32",) + ck, D * 2 * B_pad * H_pad, torch.float32, dev)
    hn_ws = _ws(("hnws",) + ck, LD * B_pad * H_pad, x.dtype, dev)
    fused_gates = _ws(("fgates",) + ck, 3 * B_pad * H_pad, torch.float32, dev)
    gx_all = _ws(("gx",) + ck, LD * 3 * S * B_pad * H_pad, torch.float32, dev)
    gx_view = gx_all.view(LD, 3, S * B_pad, H_pad)
    y_hi_ws = _ws(("yhi",) + ck, S * B_pad * DPO_pad, x.dtype, dev)
    y_lo_ws = _ws(("ylo",) + ck, S * B_pad * DPO_pad, x.dtype, dev)
    y_mid_ws = _ws(("ymid",) + ck, S * B_pad * DPO_pad, x.dtype, dev)
    h_zero_ws = _ws(("hzero",) + ck, B_pad * H_pad, x.dtype, dev)
    hn_out = torch.empty(LD * B * H, dtype=x.dtype, device=dev)
    y_out = torch.empty(S * B * DPO, dtype=x.dtype, device=dev)
    y_fin_ws = _ws(("yfin",) + ck, S * B_pad * DPO_pad, x.dtype, dev)

    # ---- 1) x pad (zero-copy view when B/F are aligned and not batchFirst)
    needs_x_pad = (B_pad != B) or batchFirst or (F_pad != F)
    if needs_x_pad:
        x_ws = _ws(("xws",) + ck, S * B_pad * F_pad, x.dtype, dev)
        k_pad = _gru_pad_kernel(S, B, B_pad, F, F_pad, 1 if batchFirst else 0, tl_dtype)
        k_pad(x.reshape(-1), x_ws)
        x_src = x_ws
    else:
        x_src = x.reshape(-1)
    if has_h0:
        k_h0p = _gru_h0pad_kernel(LD, B, B_pad, H, H_pad, tl_dtype)
        k_h0p(h0.reshape(-1), h0_ws, h0_32_ws)

    # ---- 2) per layer x direction recurrence --------------------------------
    # Routing strategy (all shapes go through the fused seq kernel):
    # unaligned H is first padded to H_pad; the weight M-pad rows/columns are
    # zero, so the pad-gate outputs stay bounded and are isolated by the zero
    # weight columns, and the scatter/gather trims never read pad columns.
    # The bidirectional case is guaranteed H%16==0 by the assert above.
    # knobs:
    #   loskip   fp16 A-side single pass -- measured to leave an insufficient
    #            precision margin (L=2 hn MERE 1.07e-3 over the 9.76e-4
    #            threshold), so it is off; the parameter is kept for tuning.
    #   ylo_mode inter-layer y relay precision (hi/lo, plus mid for bf16
    #            multi-layer); 0 on the last layer (the output is cast once).
    #   x_split  same split for the relayed x-side GEMM A operand.
    loskip = 0
    ylo_mode = 0 if (L <= 1 or tl_dtype == "float") else (2 if tl_dtype == "bfloat16" else 1)
    x_split_mode = 0 if tl_dtype == "float" else (3 if tl_dtype == "bfloat16" else 2)
    for layer in range(L):
        XW = F_pad if layer == 0 else DPO_pad
        x_base = x_src if layer == 0 else y_bufs[(layer - 1) % 2]
        y_dst = y_bufs[layer % 2]
        Kb_x_layer = _pick_kb(XW, H_pad, elem)
        assert Kb_x_layer >= 8 and (tl_dtype == "float" or Kb_x_layer >= 16), f"Kb_x violation: {Kb_x_layer}"
        split_x = x_split_mode if layer > 0 else 0
        vec_ylo = ylo_mode if layer < L - 1 else 0
        for d in range(D):
            li = layer * D + d
            col_off = d * H
            h_hi_in0 = h0_ws.view(LD * B_pad, H_pad).narrow(0, li * B_pad, B_pad)
            h32_in0 = h0_32_ws.view(LD * B_pad, H_pad).narrow(0, li * B_pad, B_pad)
            h_hi_pp = h_hi_ws.view(D * 2 * B_pad, H_pad).narrow(0, d * 2 * B_pad, 2 * B_pad)
            h_lo_pp = h_lo_ws.view(D * 2 * B_pad, H_pad).narrow(0, d * 2 * B_pad, 2 * B_pad)
            h32_pp = h32_ws.view(D * 2 * B_pad, H_pad).narrow(0, d * 2 * B_pad, 2 * B_pad)
            wi = w_ih_p[li]
            wh = w_hh_p[li]
            ba = b_ih_p[li]
            bb = b_hh_p[li]
            if layer == 0 or split_x == 0:
                x_hi_all = x_base.view(S * B_pad, XW)
                x_lo_all = x_hi_all
                x_mid_all = x_hi_all
            else:
                x_hi_all = y_hi_ws.view(S * B_pad, DPO_pad)
                x_lo_all = y_lo_ws.view(S * B_pad, DPO_pad)
                x_mid_all = y_mid_ws.view(S * B_pad, DPO_pad)
            # The x-side gate contributions are all precomputed by the
            # parallel xgemm (moved off the serial recurrence chain). With
            # loskip the relayed-layer xgemm A side also runs a single pass
            # (split_x drops to 1).
            sp_x = 1 if (split_x >= 1 and loskip == 1) else split_x
            k_xg = _gru_xgemm_kernel(S, B_pad, XW, H_pad, Kb_x_layer, sp_x, tl_dtype)
            k_xg(x_hi_all, x_lo_all, x_mid_all, wi, gx_view[li])
            # kL0Size: the L0B ping-pong dual-slot budget is
            # 2*(3H * kL0 * elem) <= 64KB, rounded to a multiple of 16.
            kL0 = max(16, (32768 // (3 * H_pad * elem) // 16) * 16)
            kL0 = min(kL0, H_pad)
            # K-concat (hi|lo in one gemm): the W|W panel must fit the L1
            # budget 2*3H*P.
            kconcat = 1 if (tl_dtype != "float" and loskip == 0 and 6 * H_pad * H_pad * elem <= 400 * 1024) else 0
            # kconcat => the W2 panel stays resident in L1 (6H^2*elem
            # <= 400KB, inside the 450KB hoist budget).
            hoist_w = 1 if ((6 if kconcat == 1 else 3) * H_pad * H_pad * elem <= 450 * 1024) else 0
            # Last-layer scatter avoidance: with B and DPO both 16-aligned
            # the V side block-writes y_out directly; with only DPO aligned
            # (B has pad rows) the V side stores the quantized rows to Y_fin
            # and the host does a light compaction.
            last_l = layer == L - 1
            direct_y = 1 if (last_l and DPO % 16 == 0 and B % 16 == 0 and not batchFirst) else 0
            dyq = 1 if (last_l and DPO % 16 == 0 and (B % 16 != 0 or batchFirst)) else 0
            need_yt = 1 if (layer < L - 1 or (direct_y == 0 and dyq == 0)) else 0
            k_seq = _gru_seq_kernel(
                S,
                B,
                B_pad,
                H_pad,
                H_pad,
                DPO_pad,
                DPO,
                kL0,
                vec_ylo,
                need_yt,
                col_off,
                d,
                direct_y,
                dyq,
                1 if batchFirst else 0,
                loskip,
                kconcat,
                LD * B_pad,
                li * B_pad,
                hoist_w,
                tl_dtype,
            )
            h_hilo = _ws(("hhilo", d) + ck, 2 * B_pad * 2 * H_pad, x.dtype, dev)
            k_seq(
                wh,
                ba.reshape(-1),
                bb.reshape(-1),
                h_hi_in0,
                h_zero_ws.view(B_pad, H_pad),
                h_hi_pp,
                h_lo_pp,
                h_hilo.view(2 * B_pad, 2 * H_pad),
                hn_ws.view(LD * B_pad, H_pad),
                h32_in0,
                h32_pp,
                fused_gates.view(B_pad, 3 * H_pad),
                gx_view[li],
                y_dst.view(S * B_pad, DPO_pad),
                y_hi_ws.view(S * B_pad, DPO_pad),
                y_lo_ws.view(S * B_pad, DPO_pad),
                y_mid_ws.view(S * B_pad, DPO_pad),
                y_out.view(S * B, DPO),
                y_fin_ws.view(S * B_pad, DPO_pad),
            )

            # The final state h was written straight into hn_ws by the seq
            # V side at the last timestep (no hncopy kernel).

    # ---- 3) y output + 4) hn trim ----
    y_final = y_bufs[(L - 1) % 2]
    if DPO % 16 == 0 and B % 16 == 0:
        # The seq V side already block-wrote y_out (no scatter kernel).
        if batchFirst:
            y = y_out.view(B, S, DPO)
        else:
            y = y_out.view(S, B, DPO)
    elif DPO % 16 == 0:
        # The seq V side already stored the quantized rows into y_fin_ws
        # (S-major dedicated buffer); the host does a light compaction that
        # trims the pad rows (batchFirst also transposes here).
        if batchFirst:
            y = y_fin_ws.view(S, B_pad, DPO)[:, :B, :].transpose(0, 1).contiguous()
        else:
            y = y_fin_ws.view(S, B_pad, DPO)[:, :B, :].contiguous()
    else:
        y_flat = _ws(("yflat",) + ck, S * B * DPO_pad, x.dtype, dev)
        k_sca = _gru_scatter_kernel(S, B, B_pad, DPO, DPO_pad, 1 if batchFirst else 0, 0, tl_dtype)
        k_sca(y_final.reshape(-1), y_flat)
        if batchFirst:
            y = y_flat.view(B, S, DPO_pad)[:, :, :DPO].contiguous()
        else:
            y = y_flat.view(S, B, DPO_pad)[:, :, :DPO].contiguous()
    k_g1 = _gru_gather_kernel(LD, B, B_pad, H_pad, H, tl_dtype)
    k_g1(hn_ws.reshape(-1), hn_out.reshape(-1))
    return y, hn_out.view(LD, B, H)


# ---------------------------------------------------------------------------
# Standalone run entry (smoke test mirroring an L0 case of test_gru.py).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    S, B, F, H = 7, 6, 64, 32
    x = torch.randn(S, B, F, dtype=torch.float32).npu()
    w_ih = [(torch.randn(3 * H, F, dtype=torch.float32) * 0.1).npu()]
    w_hh = [(torch.randn(3 * H, H, dtype=torch.float32) * 0.1).npu()]
    b_ih = [(torch.randn(3 * H, dtype=torch.float32) * 0.01).npu()]
    b_hh = [(torch.randn(3 * H, dtype=torch.float32) * 0.01).npu()]

    y, hn = gru(
        x,
        w_ih,
        w_hh,
        inputSize=F,
        hiddenSize=H,
        bias=True,
        bias_ih=b_ih,
        bias_hh=b_hh,
    )
    torch.npu.synchronize()

    ref = torch.nn.GRU(input_size=F, hidden_size=H, bias=True)
    with torch.no_grad():
        for name, src in (
            ("weight_ih_l0", w_ih[0]),
            ("weight_hh_l0", w_hh[0]),
            ("bias_ih_l0", b_ih[0]),
            ("bias_hh_l0", b_hh[0]),
        ):
            getattr(ref, name).copy_(src.cpu())
    ref.eval()
    gy, ghn = ref(x.cpu())

    torch.testing.assert_close(y.cpu(), gy, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(hn.cpu(), ghn, rtol=1e-3, atol=1e-3)
    print("Kernel Output Match!")
