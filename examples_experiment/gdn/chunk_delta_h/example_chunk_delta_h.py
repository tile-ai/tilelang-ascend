"""chunk_delta_h operator (Ascend NPU, Developer mode).

Two compile-time structures selected by use_g (Python-level branch;
only the taken arm is traced):

  use_g=True — "M-form": h_new = M @ h + C where
      M = exp(g_last)*I - K_gated^T@W  (host precompute)
      C = K_gated^T@U                   (kernel GEMM per chunk)
  The h recurrence lives entirely on the Cube core; cross-core handoffs
  drop from 4/chunk to 1/chunk (V_wh relay only).

  use_g=False — 2-GEMM structure:
      GEMM1: V_wh = W @ h_prev        (bf16, fp32 L0C)
      Vector: V_new = U - V_wh        (UB fp32)
      GEMM2: kv = K_gated^T @ V_new   (bf16 transpose_A, fp32 L0C)
      Vector: h_new = h * exp(g_last) + kv

Key design:
  - bf16 GEMM route (fp32 GEMM abandoned)
  - M-form: Cube-side h state, 1 cross-core handoff/chunk
  - Developer mode: full CV pass_configs (auto_sync + combineCV)
  - Transit buffers MUST be named "workspace*" (cross-core flag pairing
    matches by name pattern; non-matching names deadlock at BS>=16)
  - T.serial(BS) chunk loop (T.Pipelined fails with loop-carried h state)
  - threads=2 on mform path (GM-read-port decongestion); 1 on non-gate
  - out_idx=[12, 13, 14] (positive indices for h/final_state/V_new)
  - block_DV=128 default (fits L0C 64KB + UB 144KB budgets)

Precision: all outputs judged by bf16 thresholds (atol=2^-10, rtol=2^-6).
Golden stays high-precision fp32 GEMM; kernel bf16 quantization gap is the
precision metric.
"""

import os

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

# Developer mode: full CV pass_configs. Transit buffers named "workspace*"
# (cross-core flag pairing matches by name pattern; see module docstring).
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[12, 13, 14], pass_configs=pass_configs)
def _chunk_delta_h_jit(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    use_g,
    use_initial_state,
    store_final_state,
    save_new_value,
    input_dtype,
    output_dtype,
    accum_dtype,
    state_dtype,
    block_DV=128,
):
    """Chunk delta h forward kernel (Developer mode, bf16 GEMM fusion route).

    Two compile-time structures (Python-level branch; only taken arm traced):

    use_g=True — "M-form": h_new = M @ h + C. h recurrence is entirely
    Cube-side (L0C fp32 -> fixpipe bf16 -> GM -> L1 ping-pong -> next GEMM).
    Vector core is a decoupled lagging consumer (V_new = U - W@h).

    use_g=False — 2-GEMM structure. No-gate configs are all S<=256 (perf
    irrelevant), keeping the numerically safer non-gate path.

    Args:
        M (input 5): host-precomputed g*I - K_gated^T@W, bf16 (B,BS,H,DK,DK).
            Unused (dummy zeros) when use_g=False.
        h_state_gm (input 6): Cube-side state carrier GM scratch, bf16.
        state_zero (input 7): host-zeroed bf16 — non-init zero source for h_l1.
    """
    block_S = chunk_size
    BS = S // block_S
    bv_num = T.ceildiv(DV, block_DV)
    total_blocks = bv_num * B * H
    # U staging slots: 1 (init configs) or 2 (ping-pong, non-init)
    u_slots = 1 if use_initial_state else 2
    # M-form structure switch (Python-level branch, only taken arm is traced)
    mform = use_g
    # W/K_gated L1 ping-pong slots
    l1_slots = 1 if (mform or use_initial_state) else u_slots
    # U UB slots: 1 for mform (lagging consumer), ping-pong for non-gate
    ub_slots = 1 if mform else u_slots
    # threads=2 for mform (separate cores); 1 for non-gate
    threads = 2 if mform else 1
    # S-split at C>128: GEMM1 output exceeds L0C budget, split into (128, block_DV) segments
    ssplit = mform and block_S > 128
    _n1_c64 = mform and not ssplit and block_S in (64, 128)
    half_S = block_S // 2 if ssplit else block_S
    # S-split segments: nseg = block_S // 128 (2 @C=256, 4 @C=512, 8 @C=1024)
    nseg = (block_S // 128) if ssplit else 1
    seg = 128 if ssplit else block_S
    # 4 slots for mform ssplit (C>128); 2 otherwise
    relay_slots = 4 if (mform and ssplit) else 2
    # bf16 relay for mform ssplit calibers (halves relay traffic)
    relay_bf16 = mform and ssplit
    relay_dt = output_dtype if relay_bf16 else accum_dtype
    # Per-segment streaming at nseg>=8 (C=1024): K/U/W as (2, seg, x) ping-pong slabs
    streaming = mform and nseg >= 8
    # Per-program count for relay_bf16 or nseg>=8 (vid-split under threads=2)
    cast_ct_seg = (seg * block_DV // 2) if (relay_bf16 or nseg >= 8) else seg * block_DV

    # ---- Host-side asserts: memory-budget overflows crash the backend ----
    # with opaque diagnostics. These give clean, early errors for the
    # structurally-certain constraint classes.
    assert block_S >= 16, f"block_S={block_S} < 16: below the GEMM fractal M minimum (chunk_size too small for the bf16 GEMM route)"
    assert DK >= 16 and block_DV >= 16, f"DK={DK}, block_DV={block_DV}: GEMM dims below the fractal minimum 16"
    # (2) Alignment contracts
    assert DV % block_DV == 0, (
        f"DV={DV} not a multiple of block_DV={block_DV}: silent partial-tile acceptance — repack or choose block_DV | DV"
    )
    assert S % block_S == 0, (
        f"S={S} not a multiple of block_S={block_S}: the kernel only supports full chunks — truncate to BS*chunk_size host-side"
    )
    # Hardware budget constants (bytes)
    _L0C_BUDGET = 128 * 1024  # 128KB L0C hard limit
    _L1_BUDGET = 524032  # ~512KB L1
    _UB_PER_PROGRAM_BUDGET = 196352  # ~192KB UB per program (threads=2)
    _L0A_BUDGET = 64 * 1024  # 64KB L0A

    # (3) L0C budget (kv_l0c + V_wh tile)
    _vwh_rows = (seg if ssplit else block_S) if mform else block_S
    _l0c_bytes = (DK * block_DV + _vwh_rows * block_DV) * 4
    assert _l0c_bytes <= _L0C_BUDGET, (
        f"L0C budget exceeded: kv_l0c + V_wh = {_l0c_bytes}B > {_L0C_BUDGET}B "
        f"(DK={DK}, block_DV={block_DV}, V_wh rows={_vwh_rows}) — split the "
        "GEMM1 output into 128-row segments (the ssplit structure) or reduce "
        "block_DV"
    )
    # (4) L1 budget (bf16 operands)
    if mform:
        if streaming:
            # Streamed operands (h 2-slot + M + K/U/W ping-pong slabs)
            _l1_elems = 2 * DK * block_DV + DK * DK + 2 * seg * (DK + block_DV + DK)
        else:
            _l1_elems = 2 * DK * block_DV + DK * DK + block_S * (DK + block_DV) + ((nseg * seg * DK) if ssplit else block_S * DK)
    else:
        _l1_elems = DK * block_DV + 2 * l1_slots * block_S * DK + 2 * block_S * block_DV
    _l1_bytes = _l1_elems * 2
    assert _l1_bytes <= _L1_BUDGET, (
        f"L1 budget exceeded: {_l1_bytes}B > {_L1_BUDGET}B at chunk_size={block_S} "
        f"(mform={mform}) — reduce chunk_size or stream operands per segment"
    )
    # (5) mform AIV UB cluster (per program, threads=2; skipped for streaming)
    if mform and not streaming:
        _r = seg if ssplit else block_S  # phase tile rows
        _pf_rows = half_S if ssplit else block_S
        _ub_pp = (
            (_r // 2) * block_DV * (4 + 4 + 2)  # V_wh+U fp32, U bf16
            + ((2 * (_r // 2) * block_DV * 2) if relay_bf16 else 0)
            + (DK // 4) * block_DV * 2  # h0_ub ((DK//2)//2 rows)
            + (DK // 2) * DK * 2  # pf_m_ub
            + (_pf_rows // 2) * DK * 2
        )  # pf_kuw_ub
        assert _ub_pp <= _UB_PER_PROGRAM_BUDGET, (
            f"per-program UB exceeded: {_ub_pp}B > {_UB_PER_PROGRAM_BUDGET}B (mform AIV cluster at chunk_size={block_S}) — reduce prefetch granularity"
        )
    # (6) Supported nseg values (literal-unroll: {1, 2, 4, 8} only)
    if mform:
        assert nseg in (1, 2, 4, 8), (
            f"chunk_size={block_S} -> nseg={nseg}: the kernel's literal-"
            "unroll structures support nseg in {{1, 2, 4, 8}} only "
            "(C in {64..512, 1024}). nseg=16 (C=2048) is not supported."
        )

    @T.prim_func
    def kernel(
        # --- Inputs (host preprocessed, chunk-major layout) ---
        # K_gated: (B, BS, H, block_S, DK) — K * exp(g_last - g_i), bf16
        K_gated: T.Tensor((B, BS, H, block_S, DK), input_dtype),  # type: ignore
        # W: (B, BS, H, block_S, DK) — chunk-major repacked W, bf16
        W: T.Tensor((B, BS, H, block_S, DK), input_dtype),  # type: ignore
        # U: (B, BS, H, block_S, DV) — chunk-major repacked U, bf16
        U: T.Tensor((B, BS, H, block_S, DV), input_dtype),  # type: ignore
        # gate_scale: (B, BS, H, block_DV) — exp(g_last) per chunk, fp32
        gate_scale: T.Tensor((B, BS, H, block_DV), accum_dtype),  # type: ignore
        # initial_state: (B, H, DK, DV) — bf16
        initial_state: T.Tensor((B, H, DK, DV), input_dtype),  # type: ignore
        # M: exp(g_last)*I - K_gated^T@W, bf16 (B,BS,H,DK,DK). Dummy zeros when use_g=False.
        M: T.Tensor((B, BS, H, DK, DK), input_dtype),  # type: ignore
        # h_state_gm: Cube-side state carrier scratch (NOT workspace-named: cube-only carrier)
        h_state_gm: T.Tensor((total_blocks, DK, block_DV), output_dtype),  # type: ignore
        # state_zero: Host-zeroed; non-init zero source for h_l1
        state_zero: T.Tensor((total_blocks, DK, block_DV), output_dtype),  # type: ignore
        # V->C transit buffers (MUST be named "workspace*": cross-core flag pairing requires it)
        workspace_h: T.Tensor((total_blocks, DK, block_DV), output_dtype),  # type: ignore
        workspace_v: T.Tensor((total_blocks, block_S, block_DV), output_dtype),  # type: ignore
        # C->V relay workspace (L0C->GM->UB). Slot count and dtype via relay_slot_count()/relay_dtype().
        workspace_vwh: T.Tensor((total_blocks, relay_slots, block_S, block_DV), relay_dt),  # type: ignore
        workspace_kv: T.Tensor((total_blocks, DK, block_DV), accum_dtype),  # type: ignore
        # --- Outputs ---
        h: T.Tensor((B, BS, H, DK, DV), output_dtype),  # type: ignore
        final_state: T.Tensor((B, H, DK, DV), state_dtype),  # type: ignore
        # V_new is chunk-major (B, BS, H, block_S, DV) — required by mform threads=2;
        # host permutes back: V_new.permute(0, 1, 3, 2, 4).reshape(B, S, H, DV)
        V_new: T.Tensor((B, BS, H, block_S, DV), output_dtype),  # type: ignore
    ):
        with T.Kernel(total_blocks, threads=threads, is_npu=True) as (cid):
            bv = cid % bv_num
            bbh = cid // bv_num
            bb = bbh // H
            bh = bbh % H
            dv_start = bv * block_DV

            # --- L1 buffers (Cube, GEMM operands, bf16) ---
            # h_l1: 2 ping-pong slots (mform) or plain (DK, block_DV) operand (non-gate)
            h_l1 = T.alloc_shared((2, DK, block_DV) if mform else (DK, block_DV), input_dtype)
            # Double-buffered GEMM operands (slot p = i_s % 2, prefetch one chunk ahead)
            W_l1 = T.alloc_shared(
                (2, seg, DK) if streaming else ((nseg, seg, DK) if ssplit else (l1_slots, block_S, DK)),
                input_dtype,
            )
            K_gated_l1 = T.alloc_shared(
                (2, seg, DK) if streaming else (l1_slots, block_S, DK),
                input_dtype,
            )
            # M-form-only L1 buffers (dummy (1,1) in non-gate trace)
            M_l1 = T.alloc_shared((DK, DK) if mform else (1, 1), input_dtype)
            U_l1 = T.alloc_shared(  # GEMM_b B operand (bf16, Cube side)
                (2, seg, block_DV) if streaming else ((block_S, block_DV) if mform else (1, 1)),
                output_dtype,
            )
            V_new_l1 = T.alloc_shared((1, 1) if mform else (block_S, block_DV), output_dtype)

            # --- L0C buffers (Cube, GEMM output, fp32) ---
            # GEMM1 output (S-split: (seg, block_DV) fragment reused serially)
            V_wh_l0c = T.alloc_L0C(
                (seg, block_DV) if ssplit else (block_S, block_DV),
                accum_dtype,
            )
            # kv_l0c doubles as M-form hnew accumulator (C + M@h, fp32)
            kv_l0c = T.alloc_L0C((DK, block_DV), accum_dtype)
            # L0A/L0B staging for T.mma (C=64/128 M-form only)
            K_l0a = T.alloc_L0A((DK, block_S) if _n1_c64 else (1, 1), input_dtype)
            M_l0a = T.alloc_L0A((DK, DK) if _n1_c64 else (1, 1), input_dtype)
            W_l0a = T.alloc_L0A((block_S, DK) if _n1_c64 else (1, 1), input_dtype)
            h_l0b = T.alloc_L0B((DK, block_DV) if _n1_c64 else (1, 1), input_dtype)
            U_l0b = T.alloc_L0B((block_S, block_DV) if _n1_c64 else (1, 1), input_dtype)

            # --- UB buffers (Vector, element-wise) ---
            # Non-gate-only tiles become (1,1) dummies under mform
            h_ub = T.alloc_shared((DK, block_DV) if not mform else (1, 1), accum_dtype)  # fp32 recursion state (non-gate form only)
            h_bf16_ub = T.alloc_shared((DK, block_DV) if not mform else (1, 1), output_dtype)
            # S-split: (seg, block_DV) phase tiles
            V_wh_ub = T.alloc_shared(
                (seg, block_DV) if ssplit else (block_S, block_DV),
                accum_dtype,
            )
            # bf16 relay landing tiles (per-parity pair: phase k uses _ub for k%2==0, _ub2 for k%2==1)
            V_wh_bf16_ub = T.alloc_shared((seg, block_DV) if relay_bf16 else (1, 1), output_dtype)
            V_wh_bf16_ub2 = T.alloc_shared((seg, block_DV) if relay_bf16 else (1, 1), output_dtype)
            # U staging: 2 slots (non-init) or 1 slot (init configs). V_new bf16 reuses dead slot.
            U_bf16_ub = T.alloc_shared(
                (seg, block_DV) if ssplit else (block_S, block_DV) if mform else (ub_slots, block_S, block_DV),
                input_dtype,
            )
            U_ub = T.alloc_shared(
                (seg, block_DV) if ssplit else (block_S, block_DV),
                accum_dtype,
            )
            kv_ub = T.alloc_shared((DK, block_DV) if not mform else (1, 1), accum_dtype)
            # Pre-loop h[bb,0] staging (halved: (DK//2, block_DV))
            h0_ub = T.alloc_shared((DK // 2, block_DV) if mform else (1, 1), output_dtype)
            # AIV L2-prefetch scratch (requires threads=2: separate cores -> separate GM-read ports)
            pf_m_ub = T.alloc_shared((DK, DK) if mform else (1, 1), input_dtype)
            # K/W/U prefetch scratch (halved under ssplit; M stays whole)
            pf_kuw_ub = T.alloc_shared(
                ((half_S, DK) if ssplit else (block_S, DK)) if mform else (1, 1),
                input_dtype,
            )
            # Scalar gate scale staging (exp(g_last) as single scalar; (4,) for 16B alignment)
            scale_scalar_ub = T.alloc_shared((4,), accum_dtype)
            h_init_ub = T.alloc_shared(
                (DK, block_DV) if (not mform and use_initial_state) else (1, 1),
                input_dtype,
            )

            if mform:
                # ================= M-form (use_g=True) =================
                # h recurrence Cube-side; Vector is decoupled lagging consumer (V_wh relay only).
                # --- pre-loop: state init + h[bb, 0] output write ---
                if use_initial_state:
                    T.copy(
                        initial_state[bb, bh, 0:DK, dv_start : dv_start + block_DV],
                        h_l1[0, :, :],
                    )
                    # h[bb,0] staged through halved h0_ub (two row halves)
                    T.copy(
                        initial_state[bb, bh, 0 : DK // 2, dv_start : dv_start + block_DV],
                        h0_ub,
                    )
                    T.copy(
                        h0_ub,
                        h[bb, 0, bh, 0 : DK // 2, dv_start : dv_start + block_DV],
                    )
                    T.copy(
                        initial_state[bb, bh, DK // 2 : DK, dv_start : dv_start + block_DV],
                        h0_ub,
                    )
                    T.copy(
                        h0_ub,
                        h[
                            bb,
                            0,
                            bh,
                            DK // 2 : DK,
                            dv_start : dv_start + block_DV,
                        ],
                    )
                else:
                    # state_zero is host-zeroed (Cube-only read, once)
                    T.copy(state_zero[cid, 0, 0], h_l1[0, :, :])
                    T.tile.fill(h0_ub, 0.0)
                    T.copy(
                        h0_ub,
                        h[bb, 0, bh, 0 : DK // 2, dv_start : dv_start + block_DV],
                    )
                    T.copy(
                        h0_ub,
                        h[
                            bb,
                            0,
                            bh,
                            DK // 2 : DK,
                            dv_start : dv_start + block_DV,
                        ],
                    )

                if ssplit:
                    # ===== S-split loop (chunk_size>128) =====
                    # GEMM1 split into (seg, block_DV) segments serially reusing one L0C fragment.
                    # AIV consumer S-PHASED with manually unrolled phases (sync-point count balance).
                    for i_s in T.serial(BS):
                        p = i_s % 2
                        # Relay slot index (p for 2-slot, i_s % relay_slots for 4-slot)
                        slot = p if relay_slots == 2 else i_s % relay_slots
                        if BS > 1:
                            T.copy(
                                kv_l0c,
                                h[
                                    bb,
                                    T.max(i_s, 1),
                                    bh,
                                    0:DK,
                                    dv_start : dv_start + block_DV,
                                ],
                            )
                        # [AIC] M load
                        T.copy(M[bb, i_s, bh, 0:DK, 0:DK], M_l1)
                        if streaming:
                            # ===== per-segment streaming (nseg>=8) =====
                            # GEMM_a first (init=True), then nseg GEMM_b segments accumulate.
                            # Each segment: prefetch seg s+1 into other slot, then GEMM_b + GEMM1 + relay write.
                            T.copy(
                                K_gated[bb, i_s, bh, 0:seg, 0:DK],
                                K_gated_l1[0, :, :],
                            )
                            T.copy(
                                U[
                                    bb,
                                    i_s,
                                    bh,
                                    0:seg,
                                    dv_start : dv_start + block_DV,
                                ],
                                U_l1[0, :, :],
                            )
                            T.copy(W[bb, i_s, bh, 0:seg, 0:DK], W_l1[0, :, :])
                            T.gemm_v0(
                                M_l1,
                                h_l1[p, :, :],
                                kv_l0c,
                                init=True,
                                kL0Size=32,
                            )
                            # segment 0: prefetch seg 1 -> slot 1
                            T.copy(
                                K_gated[bb, i_s, bh, seg : 2 * seg, 0:DK],
                                K_gated_l1[1, :, :],
                            )
                            T.copy(
                                U[
                                    bb,
                                    i_s,
                                    bh,
                                    seg : 2 * seg,
                                    dv_start : dv_start + block_DV,
                                ],
                                U_l1[1, :, :],
                            )
                            T.copy(
                                W[bb, i_s, bh, seg : 2 * seg, 0:DK],
                                W_l1[1, :, :],
                            )
                            T.gemm_v0(
                                K_gated_l1[0, :, :],
                                U_l1[0, :, :],
                                kv_l0c,
                                transpose_A=True,
                                init=False,
                                kL0Size=32,
                            )
                            T.gemm_v0(
                                W_l1[0, :, :],
                                h_l1[p, :, :],
                                V_wh_l0c,
                                init=True,
                                kL0Size=64,
                            )
                            T.copy(
                                V_wh_l0c,
                                workspace_vwh[cid, slot, 0:seg, 0:block_DV],
                            )
                            # segment 1: prefetch seg 2 -> slot 0
                            T.copy(
                                K_gated[bb, i_s, bh, 2 * seg : 3 * seg, 0:DK],
                                K_gated_l1[0, :, :],
                            )
                            T.copy(
                                U[
                                    bb,
                                    i_s,
                                    bh,
                                    2 * seg : 3 * seg,
                                    dv_start : dv_start + block_DV,
                                ],
                                U_l1[0, :, :],
                            )
                            T.copy(
                                W[bb, i_s, bh, 2 * seg : 3 * seg, 0:DK],
                                W_l1[0, :, :],
                            )
                            T.gemm_v0(
                                K_gated_l1[1, :, :],
                                U_l1[1, :, :],
                                kv_l0c,
                                transpose_A=True,
                                init=False,
                                kL0Size=32,
                            )
                            T.gemm_v0(
                                W_l1[1, :, :],
                                h_l1[p, :, :],
                                V_wh_l0c,
                                init=True,
                                kL0Size=64,
                            )
                            T.copy(
                                V_wh_l0c,
                                workspace_vwh[cid, slot, seg : 2 * seg, 0:block_DV],
                            )
                            # segment 2: prefetch seg 3 -> slot 1
                            T.copy(
                                K_gated[bb, i_s, bh, 3 * seg : 4 * seg, 0:DK],
                                K_gated_l1[1, :, :],
                            )
                            T.copy(
                                U[
                                    bb,
                                    i_s,
                                    bh,
                                    3 * seg : 4 * seg,
                                    dv_start : dv_start + block_DV,
                                ],
                                U_l1[1, :, :],
                            )
                            T.copy(
                                W[bb, i_s, bh, 3 * seg : 4 * seg, 0:DK],
                                W_l1[1, :, :],
                            )
                            T.gemm_v0(
                                K_gated_l1[0, :, :],
                                U_l1[0, :, :],
                                kv_l0c,
                                transpose_A=True,
                                init=False,
                                kL0Size=32,
                            )
                            T.gemm_v0(
                                W_l1[0, :, :],
                                h_l1[p, :, :],
                                V_wh_l0c,
                                init=True,
                                kL0Size=64,
                            )
                            T.copy(
                                V_wh_l0c,
                                workspace_vwh[cid, slot, 2 * seg : 3 * seg, 0:block_DV],
                            )
                            # segment 3: prefetch seg 4 -> slot 0
                            T.copy(
                                K_gated[bb, i_s, bh, 4 * seg : 5 * seg, 0:DK],
                                K_gated_l1[0, :, :],
                            )
                            T.copy(
                                U[
                                    bb,
                                    i_s,
                                    bh,
                                    4 * seg : 5 * seg,
                                    dv_start : dv_start + block_DV,
                                ],
                                U_l1[0, :, :],
                            )
                            T.copy(
                                W[bb, i_s, bh, 4 * seg : 5 * seg, 0:DK],
                                W_l1[0, :, :],
                            )
                            T.gemm_v0(
                                K_gated_l1[1, :, :],
                                U_l1[1, :, :],
                                kv_l0c,
                                transpose_A=True,
                                init=False,
                                kL0Size=32,
                            )
                            T.gemm_v0(
                                W_l1[1, :, :],
                                h_l1[p, :, :],
                                V_wh_l0c,
                                init=True,
                                kL0Size=64,
                            )
                            T.copy(
                                V_wh_l0c,
                                workspace_vwh[cid, slot, 3 * seg : 4 * seg, 0:block_DV],
                            )
                            # segment 4: prefetch seg 5 -> slot 1
                            T.copy(
                                K_gated[bb, i_s, bh, 5 * seg : 6 * seg, 0:DK],
                                K_gated_l1[1, :, :],
                            )
                            T.copy(
                                U[
                                    bb,
                                    i_s,
                                    bh,
                                    5 * seg : 6 * seg,
                                    dv_start : dv_start + block_DV,
                                ],
                                U_l1[1, :, :],
                            )
                            T.copy(
                                W[bb, i_s, bh, 5 * seg : 6 * seg, 0:DK],
                                W_l1[1, :, :],
                            )
                            T.gemm_v0(
                                K_gated_l1[0, :, :],
                                U_l1[0, :, :],
                                kv_l0c,
                                transpose_A=True,
                                init=False,
                                kL0Size=32,
                            )
                            T.gemm_v0(
                                W_l1[0, :, :],
                                h_l1[p, :, :],
                                V_wh_l0c,
                                init=True,
                                kL0Size=64,
                            )
                            T.copy(
                                V_wh_l0c,
                                workspace_vwh[cid, slot, 4 * seg : 5 * seg, 0:block_DV],
                            )
                            # segment 5: prefetch seg 6 -> slot 0
                            T.copy(
                                K_gated[bb, i_s, bh, 6 * seg : 7 * seg, 0:DK],
                                K_gated_l1[0, :, :],
                            )
                            T.copy(
                                U[
                                    bb,
                                    i_s,
                                    bh,
                                    6 * seg : 7 * seg,
                                    dv_start : dv_start + block_DV,
                                ],
                                U_l1[0, :, :],
                            )
                            T.copy(
                                W[bb, i_s, bh, 6 * seg : 7 * seg, 0:DK],
                                W_l1[0, :, :],
                            )
                            T.gemm_v0(
                                K_gated_l1[1, :, :],
                                U_l1[1, :, :],
                                kv_l0c,
                                transpose_A=True,
                                init=False,
                                kL0Size=32,
                            )
                            T.gemm_v0(
                                W_l1[1, :, :],
                                h_l1[p, :, :],
                                V_wh_l0c,
                                init=True,
                                kL0Size=64,
                            )
                            T.copy(
                                V_wh_l0c,
                                workspace_vwh[cid, slot, 5 * seg : 6 * seg, 0:block_DV],
                            )
                            # segment 6: prefetch seg 7 -> slot 1
                            T.copy(
                                K_gated[bb, i_s, bh, 7 * seg : 8 * seg, 0:DK],
                                K_gated_l1[1, :, :],
                            )
                            T.copy(
                                U[
                                    bb,
                                    i_s,
                                    bh,
                                    7 * seg : 8 * seg,
                                    dv_start : dv_start + block_DV,
                                ],
                                U_l1[1, :, :],
                            )
                            T.copy(
                                W[bb, i_s, bh, 7 * seg : 8 * seg, 0:DK],
                                W_l1[1, :, :],
                            )
                            T.gemm_v0(
                                K_gated_l1[0, :, :],
                                U_l1[0, :, :],
                                kv_l0c,
                                transpose_A=True,
                                init=False,
                                kL0Size=32,
                            )
                            T.gemm_v0(
                                W_l1[0, :, :],
                                h_l1[p, :, :],
                                V_wh_l0c,
                                init=True,
                                kL0Size=64,
                            )
                            T.copy(
                                V_wh_l0c,
                                workspace_vwh[cid, slot, 6 * seg : 7 * seg, 0:block_DV],
                            )
                            # segment 7 (last: no prefetch)
                            T.gemm_v0(
                                K_gated_l1[1, :, :],
                                U_l1[1, :, :],
                                kv_l0c,
                                transpose_A=True,
                                init=False,
                                kL0Size=32,
                            )
                            T.gemm_v0(
                                W_l1[1, :, :],
                                h_l1[p, :, :],
                                V_wh_l0c,
                                init=True,
                                kL0Size=64,
                            )
                            T.copy(
                                V_wh_l0c,
                                workspace_vwh[cid, slot, 7 * seg : 8 * seg, 0:block_DV],
                            )
                        else:
                            # [AIC] operand loads (W in nseg segment slots)
                            T.copy(
                                K_gated[bb, i_s, bh, 0:block_S, 0:DK],
                                K_gated_l1[0, :, :],
                            )
                            T.copy(
                                U[
                                    bb,
                                    i_s,
                                    bh,
                                    0:block_S,
                                    dv_start : dv_start + block_DV,
                                ],
                                U_l1,
                            )
                            T.copy(W[bb, i_s, bh, 0:seg, 0:DK], W_l1[0, :, :])
                            T.copy(
                                W[bb, i_s, bh, seg : 2 * seg, 0:DK],
                                W_l1[1, :, :],
                            )
                            if nseg >= 4:
                                T.copy(
                                    W[bb, i_s, bh, 2 * seg : 3 * seg, 0:DK],
                                    W_l1[2, :, :],
                                )
                                # explicit upper bound (cross-segment boundary correctness)
                                T.copy(
                                    W[bb, i_s, bh, 3 * seg : 4 * seg, 0:DK],
                                    W_l1[3, :, :],
                                )
                            # [AIC] h chain: C = K^T@U (init) + M@h (accumulate)
                            T.gemm_v0(
                                K_gated_l1[0, :, :],
                                U_l1,
                                kv_l0c,
                                transpose_A=True,
                                init=True,
                                kL0Size=32,
                            )
                            T.gemm_v0(
                                M_l1,
                                h_l1[p, :, :],
                                kv_l0c,
                                init=False,
                                kL0Size=32,
                            )
                            # [AIC] GEMM1 segment 0: V_wh[0:seg] = W0 @ h
                            T.gemm_v0(
                                W_l1[0, :, :],
                                h_l1[p, :, :],
                                V_wh_l0c,
                                init=True,
                                kL0Size=64,
                            )
                            T.copy(
                                V_wh_l0c,
                                workspace_vwh[cid, slot, 0:seg, 0:block_DV],
                            )
                            # [AIC] GEMM1 segment 1: V_wh[seg:2*seg] = W1 @ h (serial L0C reuse)
                            T.gemm_v0(
                                W_l1[1, :, :],
                                h_l1[p, :, :],
                                V_wh_l0c,
                                init=True,
                                kL0Size=64,
                            )
                            T.copy(
                                V_wh_l0c,
                                workspace_vwh[cid, slot, seg : 2 * seg, 0:block_DV],
                            )
                            # [AIC] GEMM1 segments 2/3 (trace-time Python bool; nseg=2 skips)
                            if nseg >= 4:
                                T.gemm_v0(
                                    W_l1[2, :, :],
                                    h_l1[p, :, :],
                                    V_wh_l0c,
                                    init=True,
                                    kL0Size=64,
                                )
                                T.copy(
                                    V_wh_l0c,
                                    workspace_vwh[cid, slot, 2 * seg : 3 * seg, 0:block_DV],
                                )
                                T.gemm_v0(
                                    W_l1[3, :, :],
                                    h_l1[p, :, :],
                                    V_wh_l0c,
                                    init=True,
                                    kL0Size=64,
                                )
                                T.copy(
                                    V_wh_l0c,
                                    workspace_vwh[cid, slot, 3 * seg : 4 * seg, 0:block_DV],
                                )
                        # [AIC] state carrier ping-pong
                        T.copy(kv_l0c, h_state_gm[cid, 0, 0])
                        T.copy(h_state_gm[cid, 0, 0], h_l1[1 - p, :, :])
                        # [AIV] L2 PREFETCH la=2: M whole + K/W/U in halves (half_S-granular)
                        nxt = T.min(i_s + 2, BS - 1)
                        T.copy(M[bb, nxt, bh, 0:DK, 0:DK], pf_m_ub)
                        T.copy(
                            K_gated[bb, nxt, bh, 0:half_S, 0:DK],
                            pf_kuw_ub,
                        )
                        T.copy(
                            K_gated[bb, nxt, bh, half_S:block_S, 0:DK],
                            pf_kuw_ub,
                        )
                        T.copy(W[bb, nxt, bh, 0:half_S, 0:DK], pf_kuw_ub)
                        T.copy(W[bb, nxt, bh, half_S:block_S, 0:DK], pf_kuw_ub)
                        T.copy(
                            U[
                                bb,
                                nxt,
                                bh,
                                0:half_S,
                                dv_start : dv_start + block_DV,
                            ],
                            pf_kuw_ub,
                        )
                        T.copy(
                            U[
                                bb,
                                nxt,
                                bh,
                                half_S:block_S,
                                dv_start : dv_start + block_DV,
                            ],
                            pf_kuw_ub,
                        )
                        # [AIV] lagging consumer S-PHASE 0: rows [0, seg) — relay read every phase (sync-point balance)
                        if relay_bf16:
                            T.copy(
                                workspace_vwh[cid, slot, 0:seg, 0:block_DV],
                                V_wh_bf16_ub2,
                            )
                            T.copy(V_wh_bf16_ub2, V_wh_ub)
                        else:
                            T.copy(
                                workspace_vwh[cid, slot, 0:seg, 0:block_DV],
                                V_wh_ub,
                            )
                        T.copy(
                            U[
                                bb,
                                i_s,
                                bh,
                                0:seg,
                                dv_start : dv_start + block_DV,
                            ],
                            U_bf16_ub,
                        )
                        T.tile.cast(U_ub, U_bf16_ub, "CAST_NONE", cast_ct_seg)
                        T.tile.sub(U_ub, U_ub, V_wh_ub)
                        T.tile.cast(U_bf16_ub, U_ub, "CAST_RINT", cast_ct_seg)
                        if save_new_value:
                            T.copy(
                                U_bf16_ub,
                                V_new[
                                    bb,
                                    i_s,
                                    bh,
                                    0:seg,
                                    dv_start : dv_start + block_DV,
                                ],
                            )
                        # [AIV] S-PHASE 1: rows [seg, 2*seg) — manually unrolled (parity: phase 1 -> _ub)
                        if relay_bf16:
                            T.copy(
                                workspace_vwh[cid, slot, seg : 2 * seg, 0:block_DV],
                                V_wh_bf16_ub,
                            )
                            T.copy(V_wh_bf16_ub, V_wh_ub)
                        else:
                            T.copy(
                                workspace_vwh[cid, slot, seg : 2 * seg, 0:block_DV],
                                V_wh_ub,
                            )
                        T.copy(
                            U[
                                bb,
                                i_s,
                                bh,
                                seg : 2 * seg,
                                dv_start : dv_start + block_DV,
                            ],
                            U_bf16_ub,
                        )
                        T.tile.cast(U_ub, U_bf16_ub, "CAST_NONE", cast_ct_seg)
                        T.tile.sub(U_ub, U_ub, V_wh_ub)
                        T.tile.cast(U_bf16_ub, U_ub, "CAST_RINT", cast_ct_seg)
                        if save_new_value:
                            T.copy(
                                U_bf16_ub,
                                V_new[
                                    bb,
                                    i_s,
                                    bh,
                                    seg : 2 * seg,
                                    dv_start : dv_start + block_DV,
                                ],
                            )
                        # [AIV] S-PHASES 2/3 (trace-time bool; nseg=2 skips) — parity: 2 -> _ub, 3 -> _ub2
                        if nseg >= 4:
                            if relay_bf16:
                                T.copy(
                                    workspace_vwh[cid, slot, 2 * seg : 3 * seg, 0:block_DV],
                                    V_wh_bf16_ub,
                                )
                                T.copy(V_wh_bf16_ub, V_wh_ub)
                            else:
                                T.copy(
                                    workspace_vwh[cid, slot, 2 * seg : 3 * seg, 0:block_DV],
                                    V_wh_ub,
                                )
                            T.copy(
                                U[
                                    bb,
                                    i_s,
                                    bh,
                                    2 * seg : 3 * seg,
                                    dv_start : dv_start + block_DV,
                                ],
                                U_bf16_ub,
                            )
                            T.tile.cast(U_ub, U_bf16_ub, "CAST_NONE", cast_ct_seg)
                            T.tile.sub(U_ub, U_ub, V_wh_ub)
                            T.tile.cast(U_bf16_ub, U_ub, "CAST_RINT", cast_ct_seg)
                            if save_new_value:
                                T.copy(
                                    U_bf16_ub,
                                    V_new[
                                        bb,
                                        i_s,
                                        bh,
                                        2 * seg : 3 * seg,
                                        dv_start : dv_start + block_DV,
                                    ],
                                )
                            if relay_bf16:
                                T.copy(
                                    workspace_vwh[cid, slot, 3 * seg : 4 * seg, 0:block_DV],
                                    V_wh_bf16_ub2,
                                )
                                T.copy(V_wh_bf16_ub2, V_wh_ub)
                            else:
                                T.copy(
                                    workspace_vwh[cid, slot, 3 * seg : 4 * seg, 0:block_DV],
                                    V_wh_ub,
                                )
                            T.copy(
                                U[
                                    bb,
                                    i_s,
                                    bh,
                                    3 * seg : 4 * seg,
                                    dv_start : dv_start + block_DV,
                                ],
                                U_bf16_ub,
                            )
                            T.tile.cast(U_ub, U_bf16_ub, "CAST_NONE", cast_ct_seg)
                            T.tile.sub(U_ub, U_ub, V_wh_ub)
                            T.tile.cast(U_bf16_ub, U_ub, "CAST_RINT", cast_ct_seg)
                            if save_new_value:
                                T.copy(
                                    U_bf16_ub,
                                    V_new[
                                        bb,
                                        i_s,
                                        bh,
                                        3 * seg : 4 * seg,
                                        dv_start : dv_start + block_DV,
                                    ],
                                )
                        # [AIV] S-PHASES 4-7 (C=1024; trace-time bool, nseg<=4 skips) — parity: 4,6 -> _ub; 5,7 -> _ub2
                        if nseg >= 8:
                            if relay_bf16:
                                T.copy(
                                    workspace_vwh[cid, slot, 4 * seg : 5 * seg, 0:block_DV],
                                    V_wh_bf16_ub,
                                )
                                T.copy(V_wh_bf16_ub, V_wh_ub)
                            else:
                                T.copy(
                                    workspace_vwh[cid, slot, 4 * seg : 5 * seg, 0:block_DV],
                                    V_wh_ub,
                                )
                            T.copy(
                                U[
                                    bb,
                                    i_s,
                                    bh,
                                    4 * seg : 5 * seg,
                                    dv_start : dv_start + block_DV,
                                ],
                                U_bf16_ub,
                            )
                            T.tile.cast(U_ub, U_bf16_ub, "CAST_NONE", cast_ct_seg)
                            T.tile.sub(U_ub, U_ub, V_wh_ub)
                            T.tile.cast(U_bf16_ub, U_ub, "CAST_RINT", cast_ct_seg)
                            if save_new_value:
                                T.copy(
                                    U_bf16_ub,
                                    V_new[
                                        bb,
                                        i_s,
                                        bh,
                                        4 * seg : 5 * seg,
                                        dv_start : dv_start + block_DV,
                                    ],
                                )
                            if relay_bf16:
                                T.copy(
                                    workspace_vwh[cid, slot, 5 * seg : 6 * seg, 0:block_DV],
                                    V_wh_bf16_ub2,
                                )
                                T.copy(V_wh_bf16_ub2, V_wh_ub)
                            else:
                                T.copy(
                                    workspace_vwh[cid, slot, 5 * seg : 6 * seg, 0:block_DV],
                                    V_wh_ub,
                                )
                            T.copy(
                                U[
                                    bb,
                                    i_s,
                                    bh,
                                    5 * seg : 6 * seg,
                                    dv_start : dv_start + block_DV,
                                ],
                                U_bf16_ub,
                            )
                            T.tile.cast(U_ub, U_bf16_ub, "CAST_NONE", cast_ct_seg)
                            T.tile.sub(U_ub, U_ub, V_wh_ub)
                            T.tile.cast(U_bf16_ub, U_ub, "CAST_RINT", cast_ct_seg)
                            if save_new_value:
                                T.copy(
                                    U_bf16_ub,
                                    V_new[
                                        bb,
                                        i_s,
                                        bh,
                                        5 * seg : 6 * seg,
                                        dv_start : dv_start + block_DV,
                                    ],
                                )
                            if relay_bf16:
                                T.copy(
                                    workspace_vwh[cid, slot, 6 * seg : 7 * seg, 0:block_DV],
                                    V_wh_bf16_ub,
                                )
                                T.copy(V_wh_bf16_ub, V_wh_ub)
                            else:
                                T.copy(
                                    workspace_vwh[cid, slot, 6 * seg : 7 * seg, 0:block_DV],
                                    V_wh_ub,
                                )
                            T.copy(
                                U[
                                    bb,
                                    i_s,
                                    bh,
                                    6 * seg : 7 * seg,
                                    dv_start : dv_start + block_DV,
                                ],
                                U_bf16_ub,
                            )
                            T.tile.cast(U_ub, U_bf16_ub, "CAST_NONE", cast_ct_seg)
                            T.tile.sub(U_ub, U_ub, V_wh_ub)
                            T.tile.cast(U_bf16_ub, U_ub, "CAST_RINT", cast_ct_seg)
                            if save_new_value:
                                T.copy(
                                    U_bf16_ub,
                                    V_new[
                                        bb,
                                        i_s,
                                        bh,
                                        6 * seg : 7 * seg,
                                        dv_start : dv_start + block_DV,
                                    ],
                                )
                            if relay_bf16:
                                T.copy(
                                    workspace_vwh[cid, slot, 7 * seg : 8 * seg, 0:block_DV],
                                    V_wh_bf16_ub2,
                                )
                                T.copy(V_wh_bf16_ub2, V_wh_ub)
                            else:
                                T.copy(
                                    workspace_vwh[cid, slot, 7 * seg : 8 * seg, 0:block_DV],
                                    V_wh_ub,
                                )
                            T.copy(
                                U[
                                    bb,
                                    i_s,
                                    bh,
                                    7 * seg : 8 * seg,
                                    dv_start : dv_start + block_DV,
                                ],
                                U_bf16_ub,
                            )
                            T.tile.cast(U_ub, U_bf16_ub, "CAST_NONE", cast_ct_seg)
                            T.tile.sub(U_ub, U_ub, V_wh_ub)
                            T.tile.cast(U_bf16_ub, U_ub, "CAST_RINT", cast_ct_seg)
                            if save_new_value:
                                T.copy(
                                    U_bf16_ub,
                                    V_new[
                                        bb,
                                        i_s,
                                        bh,
                                        7 * seg : 8 * seg,
                                        dv_start : dv_start + block_DV,
                                    ],
                                )
                else:
                    for i_s in T.serial(BS):
                        p = i_s % 2
                        # [AIC] h output: fixpipe cast L0C fp32 -> GM bf16.
                        # BS>1 guard: at BS=1 the garbage write would land past h's end.
                        if BS > 1:
                            T.copy(
                                kv_l0c,
                                h[
                                    bb,
                                    T.max(i_s, 1),
                                    bh,
                                    0:DK,
                                    dv_start : dv_start + block_DV,
                                ],
                            )
                        # [AIC] operand loads (fresh each chunk)
                        T.copy(M[bb, i_s, bh, 0:DK, 0:DK], M_l1)
                        T.copy(
                            K_gated[bb, i_s, bh, 0:block_S, 0:DK],
                            K_gated_l1[0, :, :],
                        )
                        T.copy(
                            U[
                                bb,
                                i_s,
                                bh,
                                0:block_S,
                                dv_start : dv_start + block_DV,
                            ],
                            U_l1,
                        )
                        T.copy(W[bb, i_s, bh, 0:block_S, 0:DK], W_l1[0, :, :])
                        if block_S in (64, 128):
                            # T.mma batch: explicit L0A/L0B staging, h_l0b shared by GEMM_a + GEMM1
                            # GEMM_b: kv_l0c = K^T @ U (init=True)
                            T.copy(K_gated_l1[0, :, :], K_l0a, transpose=True)
                            T.copy(U_l1, U_l0b)
                            T.mma(K_l0a, U_l0b, kv_l0c, init=True)
                            # GEMM_a: kv_l0c += M @ h (shared h_l0b)
                            T.copy(M_l1, M_l0a)
                            T.copy(h_l1[p, :, :], h_l0b)
                            T.mma(M_l0a, h_l0b, kv_l0c, init=False)
                            # GEMM1: V_wh = W @ h (reuse h_l0b)
                            T.copy(W_l1[0, :, :], W_l0a)
                            T.mma(W_l0a, h_l0b, V_wh_l0c, init=True)
                        else:
                            # [AIC] dual-GEMM accumulate: C = K^T@U (init=True) + M@h (accumulate) -> h_new in one fp32 L0C
                            # kL0Size=32 enables internal L0A/L0B ping-pong (numerically neutral K-split)
                            T.gemm_v0(
                                K_gated_l1[0, :, :],
                                U_l1,
                                kv_l0c,
                                transpose_A=True,
                                init=True,
                                kL0Size=32,
                            )
                            T.gemm_v0(M_l1, h_l1[p, :, :], kv_l0c, init=False, kL0Size=32)
                            # [AIC] V_wh = W @ h (V_new output path)
                            T.gemm_v0(
                                W_l1[0, :, :],
                                h_l1[p, :, :],
                                V_wh_l0c,
                                init=True,
                                kL0Size=64,
                            )
                        # [AIC] state carrier: fixpipe fp32 -> bf16 GM, reload into other h_l1 slot (ping-pong)
                        T.copy(kv_l0c, h_state_gm[cid, 0, 0])
                        T.copy(h_state_gm[cid, 0, 0], h_l1[1 - p, :, :])
                        # [AIC] C->V relay (only cross-core handoff; 2-slot ping-pong)
                        T.copy(V_wh_l0c, workspace_vwh[cid, p, 0, 0])
                        # [AIV] L2 PREFETCH (lookahead=2): read-only GM->UB warm for next-but-one chunk.
                        # Hidden under CrossCoreWaitFlag; numerics neutral (read-only side effect).
                        nxt = T.min(i_s + 2, BS - 1)
                        T.copy(M[bb, nxt, bh, 0:DK, 0:DK], pf_m_ub)
                        T.copy(
                            K_gated[bb, nxt, bh, 0:block_S, 0:DK],
                            pf_kuw_ub,
                        )
                        T.copy(W[bb, nxt, bh, 0:block_S, 0:DK], pf_kuw_ub)
                        T.copy(
                            U[
                                bb,
                                nxt,
                                bh,
                                0:block_S,
                                dv_start : dv_start + block_DV,
                            ],
                            pf_kuw_ub,
                        )
                        # [AIV] lagging consumer: V_new = U - V_wh. Relay read every iteration (sync-point balance).
                        T.copy(workspace_vwh[cid, p, 0, 0], V_wh_ub)
                        # [AIV] U staging (2D, vid-split compatible under threads=2)
                        T.copy(
                            U[
                                bb,
                                i_s,
                                bh,
                                0:block_S,
                                dv_start : dv_start + block_DV,
                            ],
                            U_bf16_ub,
                        )
                        T.tile.cast(
                            U_ub,
                            U_bf16_ub,
                            "CAST_NONE",
                            block_S * block_DV,
                        )
                        T.tile.sub(U_ub, U_ub, V_wh_ub)
                        T.tile.cast(
                            U_bf16_ub,
                            U_ub,
                            "CAST_RINT",
                            block_S * block_DV,
                        )
                        if save_new_value:
                            # chunk-major contiguous store
                            T.copy(
                                U_bf16_ub,
                                V_new[
                                    bb,
                                    i_s,
                                    bh,
                                    0:block_S,
                                    dv_start : dv_start + block_DV,
                                ],
                            )

                # --- post-loop: final_state = last h_new (L0C fp32 direct) ---
                if store_final_state:
                    T.copy(
                        kv_l0c,
                        final_state[bb, bh, 0:DK, dv_start : dv_start + block_DV],
                    )
            else:
                # ===== 2-GEMM structure (use_g=False, verbatim) =====
                # --- Initialize h state (UB fp32) ---
                if use_initial_state:
                    T.copy(
                        initial_state[bb, bh, 0:DK, dv_start : dv_start + block_DV],
                        h_init_ub,
                    )
                    T.tile.cast(h_ub, h_init_ub, "CAST_NONE", DK * block_DV)
                else:
                    # Tile-level fill (T.Parallel zeroing would emit per-element barriers)
                    T.tile.fill(h_ub, 0.0)

                # --- Preload chunk 0's operands (non-init layout only) ---
                if not use_initial_state:
                    T.copy(W[bb, 0, bh, 0:block_S, 0:DK], W_l1[0, :, :])
                    T.copy(K_gated[bb, 0, bh, 0:block_S, 0:DK], K_gated_l1[0, :, :])
                    T.copy(
                        U[bb, 0, bh, 0:block_S, dv_start : dv_start + block_DV],
                        U_bf16_ub[0, :, :],
                    )

                # --- Chunk loop (sequential recurrence) ---
                for i_s in T.serial(BS):
                    # Ping-pong slot index (pinned to 0 for u_slots=1)
                    p = (i_s % 2) if u_slots == 2 else 0
                    # 1. Store h_prev: cast fp32->bf16, UB->GM
                    T.tile.cast(h_bf16_ub, h_ub, "CAST_RINT", DK * block_DV)
                    T.copy(
                        h_bf16_ub,
                        h[bb, i_s, bh, 0:DK, dv_start : dv_start + block_DV],
                    )

                    # 2. h V->C transit: UB->GM->L1 (bf16)
                    T.copy(h_bf16_ub, workspace_h[cid, 0, 0])
                    T.copy(workspace_h[cid, 0, 0], h_l1)

                    # 3. Load W (init layout: fresh; non-init: from ping-pong slot)
                    if use_initial_state:
                        T.copy(W[bb, i_s, bh, 0:block_S, 0:DK], W_l1[0, :, :])

                    # 4. GEMM1: V_wh = W @ h (bf16, fp32 L0C)
                    T.gemm_v0(W_l1[p, :, :], h_l1, V_wh_l0c, init=True)

                    # 5. V_wh C->V relay (2-slot ping-pong)
                    T.copy(V_wh_l0c, workspace_vwh[cid, i_s % 2, 0, 0])
                    T.copy(workspace_vwh[cid, i_s % 2, 0, 0], V_wh_ub)

                    # 6. Vector: V_new = U - V_wh (UB fp32, in-place sub)
                    if use_initial_state:
                        T.copy(
                            U[
                                bb,
                                i_s,
                                bh,
                                0:block_S,
                                dv_start : dv_start + block_DV,
                            ],
                            U_bf16_ub[0, :, :],
                        )
                    T.tile.cast(
                        U_ub,
                        U_bf16_ub[p, :, :],
                        "CAST_NONE",
                        block_S * block_DV,
                    )
                    # In-place sub: V_new lives in U_ub from here on
                    T.tile.sub(U_ub, U_ub, V_wh_ub)

                    # 7. Cast V_new fp32->bf16; store to GM if save_new_value
                    T.tile.cast(
                        U_bf16_ub[p, :, :],
                        U_ub,
                        "CAST_RINT",
                        block_S * block_DV,
                    )
                    if save_new_value:
                        T.copy(
                            U_bf16_ub[p, :, :],
                            V_new[
                                bb,
                                i_s,
                                bh,
                                0:block_S,
                                dv_start : dv_start + block_DV,
                            ],
                        )

                    # 8. V_new V->C transit: UB->GM->L1 (bf16)
                    T.copy(U_bf16_ub[p, :, :], workspace_v[cid, 0, 0])
                    T.copy(workspace_v[cid, 0, 0], V_new_l1)

                    # 9. Load K_gated (init: fresh; non-init: ping-pong slot)
                    if use_initial_state:
                        T.copy(
                            K_gated[bb, i_s, bh, 0:block_S, 0:DK],
                            K_gated_l1[0, :, :],
                        )

                    # 10. GEMM2: kv = K_gated^T @ V_new (transpose_A, bf16)
                    T.gemm_v0(
                        K_gated_l1[p, :, :],
                        V_new_l1,
                        kv_l0c,
                        transpose_A=True,
                        init=True,
                    )

                    # 11. kv C->V relay (L0C fp32 -> GM -> UB fp32)
                    T.copy(kv_l0c, workspace_kv[cid, 0, 0])
                    T.copy(workspace_kv[cid, 0, 0], kv_ub)

                    # 12. Vector: h = h * exp(g_last) + kv (two separate loops, NOT merged)
                    if use_g:
                        # Loop a: gate scale (single scalar mul)
                        T.copy(gate_scale[bb, i_s, bh, 0:4], scale_scalar_ub)
                        T.tile.mul(h_ub, h_ub, scale_scalar_ub[0])
                    # Loop b: kv accumulate (separate from loop a — do NOT merge)
                    T.tile.add(h_ub, h_ub, kv_ub)

                    # 13. Prefetch next chunk's operands (non-init layout only; clamps to BS-1)
                    if not use_initial_state:
                        nxt = T.min(i_s + 1, BS - 1)
                        q = (i_s + 1) % 2
                        T.copy(W[bb, nxt, bh, 0:block_S, 0:DK], W_l1[q, :, :])
                        T.copy(
                            K_gated[bb, nxt, bh, 0:block_S, 0:DK],
                            K_gated_l1[q, :, :],
                        )
                        # Vector-side: next chunk's U into UB slot
                        T.copy(
                            U[bb, nxt, bh, 0:block_S, dv_start : dv_start + block_DV],
                            U_bf16_ub[q, :, :],
                        )

                # --- Store final state (UB fp32 -> GM fp32, same dtype) ---
                if store_final_state:
                    T.copy(
                        h_ub,
                        final_state[bb, bh, 0:DK, dv_start : dv_start + block_DV],
                    )

    return kernel


# ============================================================================
# Compile retry (transient bisheng failures)
# ============================================================================


def _cleanup_stale_tmp_so():
    """Remove stale tilelang temp .so artifacts (/tmp/tmptl_*.so only)."""
    import glob as _glob

    removed = 0
    for p in _glob.glob("/tmp/tmptl_*.so"):
        try:
            os.remove(p)
            removed += 1
        except OSError:
            pass
    return removed


def _run_with_retry(fn, retries=2):
    """Run fn(), retrying only transient bisheng "Compilation Failed" errors.
    stderr captured into exception; other exceptions propagate unchanged."""
    import contextlib
    import io

    last_err = None
    last_tail = ""
    for attempt in range(retries + 1):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf):
                return fn()
        except Exception as e:  # noqa: BLE001 — re-raised with context below
            msg = str(e)
            stderr_tail = buf.getvalue().strip()[-2000:]
            if "Compilation Failed" not in msg:
                if stderr_tail:
                    raise type(e)(f"{msg}\n[bisheng stderr tail]\n{stderr_tail}") from e
                raise
            last_err = e
            last_tail = stderr_tail
            removed = _cleanup_stale_tmp_so()
            print(
                f"[chunk_delta_h] transient 'Compilation Failed' (attempt "
                f"{attempt + 1}/{retries + 1}); cleaned {removed} stale "
                f"tmptl_*.so, retrying ..."
            )
    # retries exhausted: attach the bisheng stderr tail for diagnosis
    if last_tail:
        raise type(last_err)(f"{last_err}\n[bisheng stderr tail]\n{last_tail}") from last_err
    raise last_err


def chunk_delta_h(**kwargs):
    """Public kernel factory = _chunk_delta_h_jit + compile retry (2 retries).
    Host-side asserts fail fast on budget/alignment violations before compile."""
    _DK = kwargs["DK"]
    _DV = kwargs["DV"]
    _block_DV = kwargs.get("block_DV", 128)
    _chunk_size = kwargs["chunk_size"]
    _S = kwargs["S"]
    _input_dtype = kwargs.get("input_dtype", "bfloat16")
    _output_dtype = kwargs.get("output_dtype", "bfloat16")
    _accum_dtype = kwargs.get("accum_dtype", "float32")
    _state_dtype = kwargs.get("state_dtype", "float32")

    # Supported dtypes: bf16 input/output, fp32 accum/state
    _SUPPORTED_INPUT = {"bfloat16"}
    _SUPPORTED_OUTPUT = {"bfloat16"}
    _SUPPORTED_ACCUM = {"float32"}
    _SUPPORTED_STATE = {"float32"}
    assert _input_dtype in _SUPPORTED_INPUT, (
        f"unsupported input_dtype={_input_dtype!r}: supported={_SUPPORTED_INPUT} "
        "(fp32 GEMM route was abandoned — use bfloat16; bf16 route is the performance path)"
    )
    assert _output_dtype in _SUPPORTED_OUTPUT, f"unsupported output_dtype={_output_dtype!r}: supported={_SUPPORTED_OUTPUT}"
    assert _accum_dtype in _SUPPORTED_ACCUM, (
        f"unsupported accum_dtype={_accum_dtype!r}: supported={_SUPPORTED_ACCUM} (L0C accumulation must be fp32 for the bf16 GEMM route)"
    )
    assert _state_dtype in _SUPPORTED_STATE, (
        f"unsupported state_dtype={_state_dtype!r}: supported={_SUPPORTED_STATE} (final_state is output directly from L0C fp32)"
    )

    assert _DK % 16 == 0 and _DV % 16 == 0 and _block_DV % 16 == 0 and _chunk_size % 16 == 0, (
        f"fractal 16 alignment: DK={_DK}, DV={_DV}, block_DV={_block_DV}, chunk_size={_chunk_size} — all must be multiples of 16"
    )
    assert _S % _chunk_size == 0, (
        f"S={_S} not a multiple of chunk_size={_chunk_size}: the kernel only supports full chunks — truncate to BS*chunk_size host-side"
    )
    _L0C_BUDGET_HOST = 128 * 1024  # 128KB L0C hard limit
    assert _DK * _block_DV * 4 <= _L0C_BUDGET_HOST, (
        f"L0C budget: DK*block_DV*4 = {_DK * _block_DV * 4}B > {_L0C_BUDGET_HOST}B (DK={_DK}, block_DV={_block_DV}) — reduce block_DV"
    )
    _block_S = _chunk_size
    _use_g = kwargs.get("use_g", True)
    _mform = _use_g
    _ssplit = _mform and _block_S > 128
    _n1 = _mform and not _ssplit and _block_S in (64, 128)
    _L0A_BUDGET = 64 * 1024  # 64KB L0A ping-pong budget
    if _n1:
        _l0a_peak = _DK * _block_S * 2 + _DK * _DK * 2
        assert _l0a_peak <= _L0A_BUDGET, (
            f"L0A budget (M-form T.mma): peak(K_l0a,W_l0a)+M_l0a "
            f"= {_l0a_peak}B > {_L0A_BUDGET}B (DK={_DK}, block_S={_block_S}) "
            f"— reduce chunk_size or DK"
        )

    return _run_with_retry(lambda: _chunk_delta_h_jit(**kwargs))


# ============================================================================
# Host preprocessing helpers (CPU operations, then H2D)
# ============================================================================


def chunk_local_cumsum(g, chunk_size):
    """Compute chunk-local cumulative sum (CPU, fla-compatible).

    Args:
        g: (B, S, H) float32
        chunk_size: C

    Returns:
        (B, S, H) float32, cumsum within each chunk
    """
    B, S, H = g.shape
    BS = S // chunk_size
    g = g.view(B, BS, chunk_size, H)
    g_sum = torch.cumsum(g, dim=2)
    return g_sum.view(B, S, H)


def prepare_k_gated(K, G, chunk_size, use_g=True):
    """K_gated = K * exp(g_last - g_i), host precompute (CPU).

    Args:
        K: (B, S, H, DK) bfloat16
        G: (B, S, H) float32 (already chunk_local_cumsum'd)
        chunk_size: C
        use_g: whether to apply gate

    Returns:
        (B, BS, H, chunk_size, DK) bfloat16 — chunk-major, gate absorbed
    """
    B, S, H, DK = K.shape
    BS = S // chunk_size
    K_f = K.float().view(B, BS, chunk_size, H, DK)  # (B, BS, C, H, DK)
    G_f = G.view(B, BS, chunk_size, H)  # (B, BS, C, H)
    if use_g:
        g_last = G_f[:, :, -1, :].unsqueeze(2)  # (B, BS, 1, H)
        coeff = torch.exp(g_last - G_f)  # (B, BS, C, H)
        K_gated = K_f * coeff.unsqueeze(-1)  # (B, BS, C, H, DK)
    else:
        K_gated = K_f
    # (B, BS, C, H, DK) -> (B, BS, H, C, DK)
    return K_gated.permute(0, 1, 3, 2, 4).contiguous().to(torch.bfloat16)


def prepare_gate_scale(G, chunk_size, block_DV, use_g=True):
    """gate_scale = exp(g_last) broadcast over block_DV (CPU, fp32).

    Args:
        G: (B, S, H) float32 (already chunk_local_cumsum'd)
        chunk_size: C
        block_DV: DV block size
        use_g: whether to apply gate (False -> all ones, unused by kernel)

    Returns:
        (B, BS, H, block_DV) float32
    """
    B, S, H = G.shape
    BS = S // chunk_size
    G_f = G.view(B, BS, chunk_size, H)
    if use_g:
        scale = torch.exp(G_f[:, :, -1, :])  # (B, BS, H)
    else:
        scale = torch.ones(B, BS, H, dtype=torch.float32)
    return scale.unsqueeze(-1).expand(B, BS, H, block_DV).contiguous()


def relay_slot_count(chunk_size, use_g=True):
    """Host-side workspace_vwh slot count (must match kernel's relay_slots).
    4 for mform ssplit (C>128); 2 otherwise."""
    return 4 if (use_g and chunk_size > 128) else 2


def relay_dtype(chunk_size, use_g=True):
    """Host-side workspace_vwh dtype (must match kernel's relay_dt).
    bf16 for mform ssplit (C>128); fp32 otherwise."""
    return "bfloat16" if (use_g and chunk_size > 128) else "float32"


def prepare_m(K_gated, W_cm, gate_scale, DK, use_g=True):
    """M = exp(g_last)*I - K_gated^T @ W, host precompute (CPU, M-form).

    Args:
        K_gated: (B, BS, H, C, DK) bfloat16 (chunk-major)
        W_cm: (B, BS, H, C, DK) bfloat16 (chunk-major)
        gate_scale: (B, BS, H, block_DV) float32 (exp(g_last))
        DK: head key dim
        use_g: when False, returns dummy zeros (kernel never reads M)

    Returns:
        (B, BS, H, DK, DK) bfloat16, chunk-major
    """
    B, BS, H = K_gated.shape[0], K_gated.shape[1], K_gated.shape[2]
    if not use_g:
        return torch.zeros(B, BS, H, DK, DK, dtype=torch.bfloat16)
    Kg = K_gated.float()
    Wf = W_cm.float()
    g = gate_scale[:, :, :, 0].float()  # (B, BS, H) — exp(g_last)
    Mi = Kg.transpose(-2, -1) @ Wf  # (B, BS, H, DK, DK) fp32
    eye = torch.eye(DK).expand(Mi.shape)
    Mi = g.unsqueeze(-1).unsqueeze(-1) * eye - Mi
    return Mi.to(torch.bfloat16).contiguous()


def to_chunk_major(t, chunk_size):
    """token-major (B, S, H, D) -> chunk-major (B, BS, H, C, D) (CPU)."""
    B, S, H, D = t.shape
    BS = S // chunk_size
    return t.view(B, BS, chunk_size, H, D).permute(0, 1, 3, 2, 4).contiguous()


def prepare_input(B, S, H, DK, DV, chunk_size, block_DV=128, use_g=True):
    """Prepare normalized inputs on CPU (gen + preprocess, then H2D).

    K/W/U are bf16 F.normalize'd; G is fp32 logsigmoid + chunk_local_cumsum.
    Kernel inputs precomputed: K_gated, W_cm, U_cm (bf16 chunk-major), gate_scale (fp32).

    Returns:
        (K, W, U, G, initial_state): original CPU tensors (golden)
        (K_gated, W_cm, U_cm, gate_scale, initial_state): kernel CPU inputs
    """
    K = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    K = F.normalize(K, dim=-1, p=2)
    W = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    W = F.normalize(W, dim=-1, p=2)
    U = torch.randn(B, S, H, DV, dtype=torch.bfloat16)
    U = F.normalize(U, dim=-1, p=2)
    G = torch.randn(B, S, H, dtype=torch.float32)
    G = F.logsigmoid(G)
    G = chunk_local_cumsum(G, chunk_size)
    initial_state = torch.randn(B, H, DK, DV, dtype=torch.bfloat16)

    K_gated = prepare_k_gated(K, G, chunk_size, use_g=use_g)  # bf16 chunk-major
    W_cm = to_chunk_major(W, chunk_size)  # bf16 chunk-major
    U_cm = to_chunk_major(U, chunk_size)  # bf16 chunk-major
    gate_scale = prepare_gate_scale(G, chunk_size, block_DV, use_g=use_g)  # fp32
    return (
        K,
        W,
        U,
        G,
        initial_state,
        K_gated,
        W_cm,
        U_cm,
        gate_scale,
        initial_state,
    )


# ============================================================================
# Smoke Test
# ============================================================================


def smoke_test():
    """Quick smoke test: small shape (S=4096, H=8, use_g=True)."""
    torch.manual_seed(0)
    B, S, H, DK, DV, chunk_size = 1, 4096, 8, 128, 128, 64
    block_DV = 128
    use_g = True
    use_initial_state = False
    store_final_state = True
    save_new_value = True

    print("[smoke] preparing input (CPU) ...")
    (
        K,
        W,
        U,
        G,
        initial_state,
        K_gated,
        W_cm,
        U_cm,
        gate_scale,
        _init,
    ) = prepare_input(B, S, H, DK, DV, chunk_size, block_DV=block_DV, use_g=use_g)

    print("[smoke] compiling kernel (bf16 GEMM route) ...")
    kernel = chunk_delta_h(
        B=B,
        S=S,
        H=H,
        DK=DK,
        DV=DV,
        chunk_size=chunk_size,
        use_g=use_g,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        save_new_value=save_new_value,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        state_dtype="float32",
        block_DV=block_DV,
    )

    print("[smoke] computing M (host, M-form) ...")
    M = prepare_m(K_gated, W_cm, gate_scale, DK, use_g=use_g)

    print("[smoke] moving inputs to NPU ...")
    bv_num = (DV + block_DV - 1) // block_DV
    total_blocks = bv_num * B * H
    h_state_gm = torch.zeros(total_blocks, DK, block_DV, dtype=torch.bfloat16, device="npu")
    state_zero = torch.zeros(total_blocks, DK, block_DV, dtype=torch.bfloat16, device="npu")
    workspace_h = torch.zeros(total_blocks, DK, block_DV, dtype=torch.bfloat16, device="npu")
    workspace_v = torch.zeros(total_blocks, chunk_size, block_DV, dtype=torch.bfloat16, device="npu")
    # Relay dtype is caliber-dependent (see relay_dtype())
    workspace_vwh = torch.zeros(
        total_blocks,
        relay_slot_count(chunk_size, use_g=use_g),
        chunk_size,
        block_DV,
        dtype=getattr(torch, relay_dtype(chunk_size, use_g=use_g)),
        device="npu",
    )
    workspace_kv = torch.zeros(total_blocks, DK, block_DV, dtype=torch.float32, device="npu")

    print("[smoke] running kernel ...")
    h_out, final_state, V_new = kernel(
        K_gated.to("npu"),
        W_cm.to("npu"),
        U_cm.to("npu"),
        gate_scale.to("npu"),
        initial_state.to("npu"),
        M.to("npu"),
        h_state_gm,
        state_zero,
        workspace_h,
        workspace_v,
        workspace_vwh,
        workspace_kv,
    )
    torch.npu.synchronize()

    # Verify output shapes (golden comparison is in test_chunk_delta_h.py)
    assert h_out.shape == (B, S // chunk_size, H, DK, DV), f"h shape mismatch: {h_out.shape} vs {(B, S // chunk_size, H, DK, DV)}"
    assert final_state.shape == (B, H, DK, DV), f"final_state shape mismatch: {final_state.shape} vs {(B, H, DK, DV)}"
    assert V_new.shape == (B, S // chunk_size, H, chunk_size, DV), (
        f"V_new shape mismatch: {V_new.shape} vs {(B, S // chunk_size, H, chunk_size, DV)}"
    )

    assert not torch.isnan(h_out).any(), "h_out contains NaN"
    assert not torch.isnan(final_state).any(), "final_state contains NaN"
    assert not torch.isnan(V_new).any(), "V_new contains NaN"

    print("Test Passed!")


if __name__ == "__main__":
    tilelang.disable_cache()
    smoke_test()
