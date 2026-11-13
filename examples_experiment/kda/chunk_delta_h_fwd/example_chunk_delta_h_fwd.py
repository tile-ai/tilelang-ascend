"""chunk_delta_h_fwd operator — KDA (Kimi Delta Attention) variant
(Ascend NPU, Developer mode).

Reference: fla/ops/common/chunk_delta_h.py (GPU chunked delta-rule state
recurrence); KDA semantics per Kimi Delta Attention — per-DK-dim vector
gate, ungated K and V_new.

KDA gated delta rule forward state recurrence (gk path, use_exp2=True):
per (b, h) and DV block, state h[DK, DV] (fp32 semantics) is scanned forward
chunk by chunk (BS = S // chunk_size, C = chunk_size):

    h_snap[:, i] = bf16(h)                    # snapshot BEFORE the update
    V_new[:, i]  = U_i - W_i @ h              # (C,DK)@(DK,DV), UNGATED
    h            = diag(exp2(gk_last_i)) @ h  # per-DK-dim VECTOR gate (exp2 domain)
    h           += K_i^T @ V_new[:, i]        # K UNGATED (KDA vs GDN key difference)
    final_state  = h (fp32 direct store)

where gk_last_i = GK[b, (i+1)*C-1, h, :] — GK is (B,S,H,DK) fp32 carrying the
chunk-local CUMULATIVE log2-domain gate (host: logsigmoid + per-dim chunk
cumsum). Unlike GDN (Gated Delta Networks, the scalar-gate delta-rule form)
there is NO per-token gating of K (the K slot holds the PLAIN chunk-major
K) and V_new is ungated.

Two compile-time structures selected by use_gk (Python-level branch;
only the taken arm is traced):

  use_gk=True — "M-form": h_new = M @ h + C where
      M = diag(exp2(gk_last)) - K^T @ W   (host precompute, KDA per-dim diag)
      C = K^T @ U                          (kernel GEMM per chunk)
  The h recurrence lives entirely on the Cube core; cross-core handoffs
  drop to 1/chunk (V_wh relay only). The kernel NEVER reads GK — the gate is
  fully absorbed into the host-precomputed M (structural M-form benefit:
  zero transcendental functions on the kernel side).

  use_gk=False — 2-GEMM structure (this arm carries no gate step, which is
  exactly the KDA ungated recurrence h += K^T@(U - W@h)):
      GEMM1: V_wh = W @ h_prev        (bf16, fp32 L0C)
      Vector: V_new = U - V_wh        (UB fp32)
      GEMM2: kv = K^T @ V_new         (bf16 transpose_A, fp32 L0C)
      Vector: h_new = h + kv

Key design (the kernel body is deliberately algorithm-agnostic — ALL
KDA-specific logic lives in the host preprocessing):
  - bf16 GEMM route (fp32 GEMM abandoned)
  - M-form: Cube-side h state, 1 cross-core handoff/chunk
  - Developer mode: full CV pass_configs (auto_sync + combineCV)
  - Transit buffers MUST be named "workspace*" (cross-core flag pairing
    matches by name pattern; non-matching names deadlock at BS>=16)
  - T.serial(BS) chunk loop (T.Pipelined fails with loop-carried h state)
  - threads=2 on mform path (GM-read-port decongestion); 1 on non-gate
  - out_idx=[11, 12, 13] (positive indices for h/final_state/V_new)
  - block_DV=128 default (fits L0C/L1/UB budgets)
  - Statement-group order is an implicit auto_sync contract — kernel statements are never reordered

Precision: all three outputs judged by bfloat16 thresholds (atol=2^-10,
rtol=2^-6, max_abs=1e0, ratio>=0.99) against the high-precision fp32 golden;
test data uses the contractive prep (see prepare_input).

Contract: kernel inputs must be FINITE. Non-finite inputs (inf/nan) are
rejected host-side (validate_inputs; prepare_m_kda additionally asserts GK
finite) BEFORE the launch — the kernel body performs no in-kernel scan
(zero overhead on the performance path; IEEE inf/nan propagation semantics
are not defined for this operator).
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


@tilelang.jit(out_idx=[11, 12, 13], pass_configs=pass_configs)
def _chunk_delta_h_fwd_jit(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    use_gk,
    use_initial_state,
    store_final_state,
    save_new_value,
    input_dtype,
    output_dtype,
    accum_dtype,
    state_dtype,
    block_DV=128,
):
    """chunk_delta_h_fwd kernel (KDA variant, Developer mode, bf16 GEMM route).

    Two compile-time structures (Python-level branch; only taken arm traced):

    use_gk=True — "M-form": h_new = M @ h + C. h recurrence is entirely
    Cube-side (L0C fp32 -> fixpipe bf16 -> GM -> L1 ping-pong -> next GEMM).
    Vector core is a decoupled lagging consumer (V_new = U - W@h).

    use_gk=False — 2-GEMM structure (no gate step exists in this arm, so it
    implements the KDA ungated recurrence h += K^T@(U - W@h) directly).

    Args:
        K_cm (input 0): plain chunk-major K (B,BS,H,C,DK), bf16 — KDA puts
            the PLAIN K in the K slot (no per-token gating).
        initial_state (input 3): (B, H, DK, DV) — bf16.
        M (input 4): host-precomputed diag(exp2(gk_last)) - K^T@W, bf16
            (B,BS,H,DK,DK), KDA per-DK-dim diagonal. Unused by the 2-GEMM arm.
        h_state_gm (input 5): Cube-side state carrier GM scratch, bf16.
        state_zero (input 6): host-zeroed bf16 — non-init zero source for h_l1.

    Note: there is no gate_scale operand — the gate is fully absorbed into
    M (mform arm) and no gate step exists on the 2-GEMM arm, so a
    gate-scale input would be dead on both arms.

    Finite-input contract: ALL value-carrying inputs must be finite.
    Non-finite inputs are rejected host-side (see validate_inputs;
    prepare_m_kda additionally asserts GK finite) — the kernel body
    performs no scan (zero overhead on the performance path; IEEE inf/nan
    propagation semantics are not defined for this operator).
    """
    block_S = chunk_size
    BS = S // block_S
    bv_num = T.ceildiv(DV, block_DV)
    total_blocks = bv_num * B * H
    # U staging slots: 1 (init configs) or 2 (ping-pong, non-init)
    u_slots = 1 if use_initial_state else 2
    # M-form structure switch (Python-level branch, only taken arm is traced)
    mform = use_gk
    # W/K_cm L1 ping-pong slots
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
    # Relay slots: 4 for the mform relay (widened WAR margin); 2 for the 2-GEMM arm
    relay_slots = 4 if mform else 2
    # bf16 relay for mform ssplit calibers (halves relay traffic)
    relay_bf16 = mform and ssplit
    relay_dt = output_dtype if relay_bf16 else accum_dtype
    # Per-segment streaming at nseg>=8 (C=1024): K/U/W as (2, seg, x) ping-pong slabs
    streaming = mform and nseg >= 8
    # Phase-tile cast count for the ssplit consumer phases: the FULL
    # (seg, block_DV) tile count. The framework halves the count per vector
    # sub-block under threads=2 — pre-halving here would leave the second
    # half of each sub-block's partition un-cast (stale V_new rows).
    cast_ct_seg = seg * block_DV

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
    # (5) ssplit segment divisibility: nseg*seg covers only nseg*128 rows —
    # a non-divisible block_S would silently drop the tail (and index past
    # the segment slabs). Supported chunk_size: {64, 128, 256, 512, 1024}
    # (64/128 run the non-ssplit route; 256/512/1024 are nseg*128 multiples).
    assert (not ssplit) or block_S % 128 == 0, (
        f"ssplit chunk_size={block_S} is not a multiple of 128: the ssplit "
        "structure covers only nseg*128 rows and would silently drop the "
        "tail. Supported chunk_size: {64, 128, 256, 512, 1024} (64/128 run "
        "the non-ssplit route; 256/512/1024 are nseg*128 multiples)."
    )
    # (6) Supported nseg values (literal-unroll: {1, 2, 4, 8} only)
    if mform:
        assert nseg in (1, 2, 4, 8), (
            f"chunk_size={block_S} -> nseg={nseg}: the kernel's literal-"
            "unroll structures support nseg in {{1, 2, 4, 8}} only "
            "(chunk_size in {64, 128, 256, 512, 1024}; chunk_size=2048 -> "
            "nseg=16 is not supported)."
        )
    # (7) mform AIV UB cluster (per program, threads=2). Ordered AFTER the
    # structural domain asserts (5)/(6) so unsupported-nseg calibers (e.g.
    # chunk_size=2048 -> nseg=16) keep their specific rejection message.
    if mform:
        _r = seg if ssplit else block_S  # phase tile rows
        _pf_rows = half_S if ssplit else block_S
        _relay_tiles = 2 if relay_bf16 else 0
        if streaming:
            # Streaming UB accounting, calibrated against the GENERATED C
            # (C=1024/DK=128/block_DV=128): the memory planner liveness-
            # aliases the second relay tile (V_wh_bf16_ub2) onto the dead
            # h0_ub/pf_m_ub region — measured high-water 188416B <= 196352B
            # budget (the no-alias sum, 204800B, would overflow; the
            # non-streaming formula below reproduces the no-alias layout
            # exactly, verified against the C=256/512 dumps: 155648B /
            # 172032B). The aliasing is available only while the relay
            # tile fits the dead region; otherwise the planner must give
            # it a fresh slot.
            _dead_region = (DK // 4) * block_DV * 2 + (DK // 2) * DK * 2
            _relay_tile = (_r // 2) * block_DV * 2
            if relay_bf16 and _relay_tile <= _dead_region:
                _relay_tiles = 1
        _ub_pp = (
            (_r // 2) * block_DV * (4 + 4 + 2)  # V_wh+U fp32, U bf16
            + _relay_tiles * (_r // 2) * block_DV * 2
            + (DK // 4) * block_DV * 2  # h0_ub ((DK//2)//2 rows)
            + (DK // 2) * DK * 2  # pf_m_ub
            + (_pf_rows // 2) * DK * 2
        )  # pf_kuw_ub
        assert _ub_pp <= _UB_PER_PROGRAM_BUDGET, (
            f"per-program UB exceeded: {_ub_pp}B > {_UB_PER_PROGRAM_BUDGET}B (mform AIV cluster at chunk_size={block_S}) — reduce prefetch granularity"
        )
    # (8) mform chunk_size domain (use_gk=True): the mform arm's traced
    # structures cover C in {64, 128} (T.mma _n1_c64 route) and C >= 256
    # multiples of 128 (ssplit nseg in {2, 4, 8} via assert (6)). Other
    # C >= 16 values (16/32/48/80/96/112 ...) pass the generic alignment
    # asserts but land in an unverified gemm_v0 route with (1,1) dummy
    # L0A/L0B staging — reject them explicitly so the value-range check
    # does not fake a verified domain (the 2-GEMM arm keeps the
    # family-tested C >= 16 domain). Placed AFTER (5)/(6) so non-divisible
    # or bad-nseg ssplit calibers keep their more specific messages.
    assert (not mform) or block_S in (64, 128) or (block_S % 128 == 0 and block_S >= 256), (
        f"mform (use_gk=True) chunk_size={block_S} is outside the supported "
        "domain {{64, 128, 256, 512, 1024}}: 64/128 run the T.mma route; "
        "256/512/1024 run the ssplit route (nseg*128 row coverage) — "
        "use_gk=False (2-GEMM arm) keeps the family-tested C >= 16 domain"
    )

    @T.prim_func
    def kernel(
        # --- Inputs (host preprocessed, chunk-major layout) ---
        # K_cm: (B, BS, H, block_S, DK) — PLAIN chunk-major K (KDA: NO per-token gating), bf16
        K_cm: T.Tensor((B, BS, H, block_S, DK), input_dtype),  # type: ignore
        # W: (B, BS, H, block_S, DK) — chunk-major repacked W, bf16
        W: T.Tensor((B, BS, H, block_S, DK), input_dtype),  # type: ignore
        # U: (B, BS, H, block_S, DV) — chunk-major repacked U, bf16
        U: T.Tensor((B, BS, H, block_S, DV), input_dtype),  # type: ignore
        # initial_state: (B, H, DK, DV) — bf16
        initial_state: T.Tensor((B, H, DK, DV), input_dtype),  # type: ignore
        # M: diag(exp2(gk_last)) - K^T@W (KDA per-DK-dim diagonal), bf16 (B,BS,H,DK,DK)
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
            h_init_ub = T.alloc_shared(
                (DK, block_DV) if (not mform and use_initial_state) else (1, 1),
                input_dtype,
            )

            if mform:
                # ================= M-form (use_gk=True) =================
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
                                K_cm[bb, i_s, bh, 0:seg, 0:DK],
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
                                K_cm[bb, i_s, bh, seg : 2 * seg, 0:DK],
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
                                K_cm[bb, i_s, bh, 2 * seg : 3 * seg, 0:DK],
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
                                K_cm[bb, i_s, bh, 3 * seg : 4 * seg, 0:DK],
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
                                K_cm[bb, i_s, bh, 4 * seg : 5 * seg, 0:DK],
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
                                K_cm[bb, i_s, bh, 5 * seg : 6 * seg, 0:DK],
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
                                K_cm[bb, i_s, bh, 6 * seg : 7 * seg, 0:DK],
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
                                K_cm[bb, i_s, bh, 7 * seg : 8 * seg, 0:DK],
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
                                K_cm[bb, i_s, bh, 0:block_S, 0:DK],
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
                            K_cm[bb, nxt, bh, 0:half_S, 0:DK],
                            pf_kuw_ub,
                        )
                        T.copy(
                            K_cm[bb, nxt, bh, half_S:block_S, 0:DK],
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
                                V_wh_bf16_ub,
                            )
                            T.copy(V_wh_bf16_ub, V_wh_ub)
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
                        # [AIV] S-PHASE 1: rows [seg, 2*seg) — manually unrolled (parity: phase 1 -> _ub2)
                        if relay_bf16:
                            T.copy(
                                workspace_vwh[cid, slot, seg : 2 * seg, 0:block_DV],
                                V_wh_bf16_ub2,
                            )
                            T.copy(V_wh_bf16_ub2, V_wh_ub)
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
                        # Relay slot index (mform relay is 4-slot;
                        # p for the 2-slot arm, i_s % relay_slots for 4-slot
                        # mform — same indexing pattern as the ssplit branch)
                        slot = p if relay_slots == 2 else i_s % relay_slots
                        # [AIC] operand loads (fresh each chunk)
                        T.copy(M[bb, i_s, bh, 0:DK, 0:DK], M_l1)
                        T.copy(
                            K_cm[bb, i_s, bh, 0:block_S, 0:DK],
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
                        # [AIC] h output + state carrier MERGED: the old
                        # head-of-loop h write and the h_state_gm write were a
                        # pure double fixpipe of the same kv_l0c data. Now a
                        # single fixpipe into h[bb, i_s+1] (the snapshot slot),
                        # then reload h_l1 from h — the h_state_gm round-trip
                        # drops out. Last chunk (i_s == BS-1): no h slot left
                        # (its h_new goes to final_state directly) and
                        # h_l1[1-p] is never read again — skip both.
                        if i_s < BS - 1:
                            T.copy(
                                kv_l0c,
                                h[
                                    bb,
                                    i_s + 1,
                                    bh,
                                    0:DK,
                                    dv_start : dv_start + block_DV,
                                ],
                            )
                            T.copy(
                                h[
                                    bb,
                                    i_s + 1,
                                    bh,
                                    0:DK,
                                    dv_start : dv_start + block_DV,
                                ],
                                h_l1[1 - p, :, :],
                            )
                        # [AIC] C->V relay (only cross-core handoff; relay_slots
                        # ping-pong — 4 for mform, 2 for the 2-GEMM arm)
                        T.copy(V_wh_l0c, workspace_vwh[cid, slot, 0, 0])
                        # [AIV] L2 PREFETCH (lookahead=2): read-only GM->UB warm for next-but-one chunk.
                        # Hidden under CrossCoreWaitFlag; numerics neutral (read-only side effect).
                        nxt = T.min(i_s + 2, BS - 1)
                        T.copy(M[bb, nxt, bh, 0:DK, 0:DK], pf_m_ub)
                        T.copy(
                            K_cm[bb, nxt, bh, 0:block_S, 0:DK],
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
                        # Slot follows the producer's i_s % relay_slots (4-slot mform).
                        T.copy(workspace_vwh[cid, slot, 0, 0], V_wh_ub)
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
                        # count=full assumes this statement group is not vid-split under the current framework version
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
                # ===== 2-GEMM structure (use_gk=False) =====
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
                    T.copy(K_cm[bb, 0, bh, 0:block_S, 0:DK], K_gated_l1[0, :, :])
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

                    # 9. Load K_cm (init: fresh; non-init: ping-pong slot)
                    if use_initial_state:
                        T.copy(
                            K_cm[bb, i_s, bh, 0:block_S, 0:DK],
                            K_gated_l1[0, :, :],
                        )

                    # 10. GEMM2: kv = K^T @ V_new (transpose_A, bf16) — plain K (KDA: ungated)
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

                    # 12. Vector: h = h + kv (KDA use_gk=False: NO gate step — the
                    # ungated recurrence is exactly h += K^T@(U - W@h)).
                    T.tile.add(h_ub, h_ub, kv_ub)

                    # 13. Prefetch next chunk's operands (non-init layout only; clamps to BS-1)
                    if not use_initial_state:
                        nxt = T.min(i_s + 1, BS - 1)
                        q = (i_s + 1) % 2
                        T.copy(W[bb, nxt, bh, 0:block_S, 0:DK], W_l1[q, :, :])
                        T.copy(
                            K_cm[bb, nxt, bh, 0:block_S, 0:DK],
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
                f"[chunk_delta_h_fwd] transient 'Compilation Failed' (attempt "
                f"{attempt + 1}/{retries + 1}); cleaned {removed} stale "
                f"tmptl_*.so, retrying ..."
            )
    # retries exhausted: attach the bisheng stderr tail for diagnosis
    if last_tail:
        raise type(last_err)(f"{last_err}\n[bisheng stderr tail]\n{last_tail}") from last_err
    raise last_err


def chunk_delta_h_fwd(**kwargs):
    """Public kernel factory = _chunk_delta_h_fwd_jit + compile retry (2 retries).
    Host-side asserts fail fast on budget/alignment violations before compile.

    Finite-input contract: inputs must be finite — validate host-side via
    validate_inputs(K_cm, W_cm, U_cm, M, initial_state) BEFORE the launch;
    the kernel body performs no scan (zero overhead on the perf path).
    """
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
    _B = kwargs["B"]
    _H = kwargs["H"]
    assert _B >= 1 and _H >= 1 and _chunk_size <= _S, (
        f"degenerate shape: B={_B}, H={_H}, S={_S}, chunk_size={_chunk_size} — "
        "need B >= 1, H >= 1, S >= chunk_size (at least one full chunk)"
    )
    assert _S % _chunk_size == 0, (
        f"S={_S} not a multiple of chunk_size={_chunk_size}: the kernel only supports full chunks — truncate to BS*chunk_size host-side"
    )
    _L0C_BUDGET_HOST = 128 * 1024  # 128KB L0C hard limit
    assert _DK * _block_DV * 4 <= _L0C_BUDGET_HOST, (
        f"L0C budget: DK*block_DV*4 = {_DK * _block_DV * 4}B > {_L0C_BUDGET_HOST}B (DK={_DK}, block_DV={_block_DV}) — reduce block_DV"
    )
    _block_S = _chunk_size
    _use_gk = kwargs.get("use_gk", True)
    _mform = _use_gk
    _ssplit = _mform and _block_S > 128
    _n1 = _mform and not _ssplit and _block_S in (64, 128)
    # ssplit segment divisibility (same contract as the kernel-side assert)
    assert (not _ssplit) or _chunk_size % 128 == 0, (
        f"ssplit chunk_size={_chunk_size} is not a multiple of 128: the ssplit "
        "structure covers only nseg*128 rows and would silently drop the "
        "tail. Supported chunk_size: {64, 128, 256, 512, 1024} (64/128 run "
        "the non-ssplit route; 256/512/1024 are nseg*128 multiples)."
    )
    _L0A_BUDGET = 64 * 1024  # 64KB L0A ping-pong budget
    if _n1:
        _l0a_peak = _DK * _block_S * 2 + _DK * _DK * 2
        assert _l0a_peak <= _L0A_BUDGET, (
            f"L0A budget (M-form T.mma): peak(K_l0a,W_l0a)+M_l0a "
            f"= {_l0a_peak}B > {_L0A_BUDGET}B (DK={_DK}, block_S={_block_S}) "
            f"— reduce chunk_size or DK"
        )
    # mform chunk_size domain (same contract as the kernel-side assert (8);
    # placed AFTER the ssplit-divisibility assert above so non-divisible
    # ssplit calibers keep their specific "not a multiple of 128" message):
    # the mform arm's traced structures cover {64, 128} (T.mma route) +
    # C >= 256 multiples of 128 (ssplit route); other C >= 16 values would
    # land in an unverified route — the generic alignment asserts alone
    # must not fake a verified domain.
    assert (not _mform) or _chunk_size in (64, 128) or (_chunk_size % 128 == 0 and _chunk_size >= 256), (
        f"mform (use_gk=True) chunk_size={_chunk_size} is outside the supported "
        "domain {{64, 128, 256, 512, 1024}}: 64/128 run the T.mma route; "
        "256/512/1024 run the ssplit route (nseg*128 row coverage) — "
        "use_gk=False (2-GEMM arm) keeps the family-tested C >= 16 domain"
    )

    return _run_with_retry(lambda: _chunk_delta_h_fwd_jit(**kwargs))


# ============================================================================
# Host preprocessing helpers (CPU operations, then H2D) — KDA variants
# ============================================================================


def chunk_local_cumsum_gk(gk, chunk_size):
    """Per-dim chunk-local cumulative sum (CPU, fla-compatible, exp2 domain).

    KDA counterpart of FLA's chunk_local_cumsum for the per-head-DIM gate:
    the cumulative gate is accumulated over S within each chunk, per (B,H,DK).

    Args:
        gk: (B, S, H, DK) float32 — raw per-dim log2-domain gate increments
        chunk_size: C

    Returns:
        (B, S, H, DK) float32, cumsum along S within each chunk
    """
    B, S, H, DK = gk.shape
    BS = S // chunk_size
    gk = gk.view(B, BS, chunk_size, H, DK)
    gk_sum = torch.cumsum(gk, dim=2)
    return gk_sum.view(B, S, H, DK)


def relay_slot_count(chunk_size, use_gk=True):
    """Host-side workspace_vwh slot count (must match kernel's relay_slots).
    4 for mform (widened relay WAR window; the ssplit branch already used
    4); 2 for the 2-GEMM arm."""
    return 4 if use_gk else 2


def relay_dtype(chunk_size, use_gk=True):
    """Host-side workspace_vwh dtype (must match kernel's relay_dt).
    bf16 for mform ssplit (C>128); fp32 otherwise."""
    return "bfloat16" if (use_gk and chunk_size > 128) else "float32"


def prepare_m_kda(K_cm, W_cm, GK, chunk_size, DK, use_gk=True):
    """M = diag(exp2(gk_last)) - K^T @ W, host precompute (CPU, KDA M-form).

    THE KDA-specific change vs the scalar-gate (GDN-style) delta rule: the
    diagonal is a per-DK-dim VECTOR diag(exp2(gk_last)) instead of the
    scalar broadcast exp(g_last)*I. gk_last is the chunk-end cumulative
    gate in the exp2 domain: gk_last_i = GK[b, (i+1)*C-1, h, :].

    fp32 matmul then ONE bf16 quantization (single-rounding contract).

    GK finiteness is asserted here (parallel to the shape assert): the
    per-dim gate is consumed host-side only — exp2 of a non-finite
    log2-domain gate would poison M with no in-kernel chance of detection.

    Args:
        K_cm: (B, BS, H, C, DK) bfloat16 chunk-major — PLAIN K (no gating)
        W_cm: (B, BS, H, C, DK) bfloat16 chunk-major
        GK: (B, S, H, DK) float32, token-major, already chunk_local_cumsum_gk'd
        chunk_size: C
        DK: head key dim
        use_gk: when False the diagonal becomes ones (M = I - K^T@W, the
            ungated recurrence; the 2-GEMM arm never reads M — value kept
            semantically correct anyway)

    Returns:
        (B, BS, H, DK, DK) bfloat16, chunk-major
    """
    B, BS, H = K_cm.shape[0], K_cm.shape[1], K_cm.shape[2]
    assert GK.shape == (B, BS * chunk_size, H, DK), (
        f"GK shape mismatch: {tuple(GK.shape)} vs {(B, BS * chunk_size, H, DK)} "
        "(GK is the per-dim cumulative gate, consumed ONLY host-side)"
    )
    _gk_nonfinite = int((~torch.isfinite(GK)).sum().item())
    assert _gk_nonfinite == 0, (
        f"GK contains {_gk_nonfinite} non-finite element(s): the per-dim gate is "
        "consumed host-side only (M = diag(exp2(gk_last)) - K^T@W; exp2 of a "
        "non-finite log2-domain gate is undefined) — inputs must be finite"
    )
    Kf = K_cm.float()
    Wf = W_cm.float()
    KWt = Kf.transpose(-2, -1) @ Wf  # (B, BS, H, DK, DK) fp32
    GK_c = GK.view(B, BS, chunk_size, H, DK)
    if use_gk:
        gk_last = GK_c[:, :, -1, :, :]  # (B, BS, H, DK) — chunk-end cumulative gate
        diag = torch.exp2(gk_last)  # fp32, per-DK-dim
    else:
        diag = torch.ones(B, BS, H, DK, dtype=torch.float32)
    Mi = torch.diag_embed(diag) - KWt  # (B, BS, H, DK, DK) fp32
    return Mi.to(torch.bfloat16).contiguous()


def to_chunk_major(t, chunk_size):
    """token-major (B, S, H, D) -> chunk-major (B, BS, H, C, D) (CPU)."""
    B, S, H, D = t.shape
    BS = S // chunk_size
    return t.view(B, BS, chunk_size, H, D).permute(0, 1, 3, 2, 4).contiguous()


def prepare_input(B, S, H, DK, DV, chunk_size, block_DV=128, use_gk=True, init_scale=0.05):
    """Prepare KDA inputs on CPU (gen + preprocess, then H2D).

    K/W/U are bf16 F.normalize'd (official distribution); GK is fp32
    logsigmoid + per-dim chunk-local cumsum (exp2 domain, official).
    initial_state uses the CONTRACTIVE prep (randn * init_scale, bf16) with
    default init_scale=0.05: full-magnitude randn pushes the KDA ungated-K
    recurrence past the bf16 thresholds as an intrinsic bf16-GEMM error
    property; x0.05 keeps it well inside. Pass init_scale=1.0 for the
    official-magnitude boundary case.

    Kernel inputs precomputed: K_cm (PLAIN chunk-major K — KDA has NO
    per-token gating), W_cm, U_cm (bf16 chunk-major), initial_state (bf16).

    Returns:
        (K, W, U, GK, initial_state): original CPU tensors (golden inputs;
            initial_state doubles as the kernel input)
        (K_cm, W_cm, U_cm): kernel CPU inputs (chunk-major)
    """
    K = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    K = F.normalize(K, dim=-1, p=2)
    W = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    W = F.normalize(W, dim=-1, p=2)
    U = torch.randn(B, S, H, DV, dtype=torch.bfloat16)
    U = F.normalize(U, dim=-1, p=2)
    GK = torch.randn(B, S, H, DK, dtype=torch.float32)
    GK = F.logsigmoid(GK)
    GK = chunk_local_cumsum_gk(GK, chunk_size)
    initial_state = (torch.randn(B, H, DK, DV, dtype=torch.float32) * init_scale).to(torch.bfloat16)

    K_cm = to_chunk_major(K, chunk_size)  # bf16 chunk-major, PLAIN (KDA: ungated)
    W_cm = to_chunk_major(W, chunk_size)  # bf16 chunk-major
    U_cm = to_chunk_major(U, chunk_size)  # bf16 chunk-major
    return (
        K,
        W,
        U,
        GK,
        initial_state,
        K_cm,
        W_cm,
        U_cm,
    )


def validate_inputs(K_cm, W_cm, U_cm, M, initial_state):
    """Host-side finiteness validation of the kernel inputs (public).

    Contract: ALL value-carrying kernel inputs must be finite. Non-finite
    inputs (inf/nan) are rejected HERE, host-side BEFORE the launch — the
    kernel body performs no scan (zero overhead on the performance path;
    IEEE inf/nan propagation semantics are not defined for this operator).
    The host-zeroed workspace/scratch tensors are not value-carrying and
    are excluded. GK finiteness is asserted separately inside
    prepare_m_kda (the gate never reaches the kernel).

    Args:
        K_cm: (B, BS, H, C, DK) bfloat16 chunk-major K
        W_cm: (B, BS, H, C, DK) bfloat16 chunk-major W
        U_cm: (B, BS, H, C, DV) bfloat16 chunk-major U
        M: (B, BS, H, DK, DK) bfloat16 mform precompute (the 2-GEMM arm
            never reads it, but the value is kept semantically correct —
            validate it all the same)
        initial_state: (B, H, DK, DV) bfloat16

    Raises:
        AssertionError: names the offending tensor and its inf/nan element
        count (actionable: sanitize or reject the batch host-side).
    """
    for name, t in (
        ("K_cm", K_cm),
        ("W_cm", W_cm),
        ("U_cm", U_cm),
        ("M", M),
        ("initial_state", initial_state),
    ):
        n_bad = int((~torch.isfinite(t)).sum().item())
        assert n_bad == 0, (
            f"non-finite input rejected: {name} contains {n_bad} inf/nan "
            "element(s) — inputs must be finite (the kernel performs no "
            "in-kernel scan; sanitize or reject the batch host-side)"
        )


# ============================================================================
# Smoke Test
# ============================================================================


def smoke_test():
    """Quick smoke test: small shape (S=1024, H=4), all flags True.

    Compile + run + shape/NaN verification (golden comparison lives in
    test_chunk_delta_h_fwd.py). Uses the contractive prep (init x0.05).
    """
    torch.manual_seed(0)
    B, S, H, DK, DV, chunk_size = 1, 1024, 4, 128, 128, 64
    block_DV = 128
    use_gk = True
    use_initial_state = True
    store_final_state = True
    save_new_value = True

    print("[smoke] preparing input (CPU, contractive prep init x0.05) ...")
    (
        K,
        W,
        U,
        GK,
        initial_state,
        K_cm,
        W_cm,
        U_cm,
    ) = prepare_input(B, S, H, DK, DV, chunk_size, block_DV=block_DV, use_gk=use_gk)

    print("[smoke] compiling kernel (bf16 GEMM route) ...")
    kernel = chunk_delta_h_fwd(
        B=B,
        S=S,
        H=H,
        DK=DK,
        DV=DV,
        chunk_size=chunk_size,
        use_gk=use_gk,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        save_new_value=save_new_value,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        state_dtype="float32",
        block_DV=block_DV,
    )

    print("[smoke] computing M (host, KDA M-form per-dim diag) ...")
    M = prepare_m_kda(K_cm, W_cm, GK, chunk_size, DK, use_gk=use_gk)

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
        relay_slot_count(chunk_size, use_gk=use_gk),
        chunk_size,
        block_DV,
        dtype=getattr(torch, relay_dtype(chunk_size, use_gk=use_gk)),
        device="npu",
    )
    workspace_kv = torch.zeros(total_blocks, DK, block_DV, dtype=torch.float32, device="npu")

    print("[smoke] running kernel ...")
    h_out, final_state, V_new = kernel(
        K_cm.to("npu"),
        W_cm.to("npu"),
        U_cm.to("npu"),
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

    # Verify output shapes (golden comparison is in test_chunk_delta_h_fwd.py)
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
