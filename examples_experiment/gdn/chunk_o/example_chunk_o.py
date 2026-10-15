"""chunk_o (chunk-based linear-attention O forward) for Ascend NPU.

Computes O = (Q @ HIDDEN) * exp(G) + A @ V for each chunk, where
A = causal_mask((Q @ K^T) * exp(G_i - G_j)) is the intra-chunk gated
attention (GDN chunk-O forward). V and HIDDEN must be pre-scaled by
scale = DK ** -0.5 on the host (see _prepare_inputs); the kernel does
not apply scale.
"""

import math
import sys

import tilelang
import torch
from tilelang import language as T

_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[5], pass_configs=_pass_configs)
def chunk_o(
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    chunk_size,
    use_g=True,
    block_DK=128,
    block_DV=128,
    core_num=20,
    chunks_per_tile=1,
):
    """chunk_o forward kernel (Fixed Core + grid-stride loop).

    Per chunk of size `chunk_size`:
        A = causal_lower_tri((Q @ K^T) * exp(G_i - G_j))   [gated, if use_g]
        O = (Q @ HIDDEN) * exp(G) + A @ V

    Contracts (guarded at entry):
      - chunk_size / block_DK / block_DV: positive multiples of 16
        (T.tile.compare 256B alignment + packed uint8 masks + L0 fractal layout).
      - DK % block_DK == 0 and DV % block_DV == 0: partial tiles are rejected.
        DK is the Cube reduction dim, where a partial tile would make every
        dot product depend on framework bounds-clamping zero-fill; rejecting
        keeps that silent-corruption surface out of the contract.
      - accum_dtype / gate_dtype: must be "float32" (AscendC cast path limit).
      - input_dtype: bfloat16 / float16; output_dtype: bfloat16 / float32
        (both covered by the test suite).
      - core_num: positive int, default 20 (910B3 AI cores).
      - chunks_per_tile: positive int, default 1.
      - UB capacity: the peak simultaneously-live UB footprint (a function of
        chunk_size / block_DV / use_g / input_dtype) must fit the A2/A3 UB
        limit of 196,352 bytes (ASCEND_A2A3_UB_SIZE). E.g. chunk_size=128 with
        use_g=True, block_DV=128 needs >= 330,240 bytes and is rejected;
        reduce chunk_size / block_DV or set use_g=False.
      - S is expected to be a multiple of chunk_size; `_prepare_inputs`
        zero-pads S. Direct calls with non-divisible S produce correct results
        via framework bounds clamping + tril causality (out-of-range tail rows
        only feed clipped output rows or tril-masked columns); covered by
        suite case l1_s_nondiv_direct.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive multiple of 16, got {chunk_size}.")
    if chunk_size % 16 != 0:
        raise ValueError(
            f"chunk_size must be a multiple of 16, got {chunk_size}. Hardware constraints: "
            "(1) T.tile.compare requires 256B-aligned fp32 sources (4*chunk_size^2 % 256 == 0, "
            "i.e. chunk_size % 8 == 0); (2) packed uint8 masks are allocated as "
            "(chunk_size, chunk_size // 8) — floor division loses mask bits when chunk_size % 8 != 0; "
            "(3) Cube mma M dimension (L0A/L0C fractal layout) requires chunk_size % 16 == 0 — "
            "non-multiples crash the AICore at runtime (error 507015)."
        )
    if block_DK % 16 != 0 or block_DV % 16 != 0:
        raise ValueError(
            f"block_DK and block_DV must be multiples of 16, got block_DK={block_DK}, "
            f"block_DV={block_DV}. L0A/L0B/L0C fractal layouts require 16-alignment: "
            "non-multiple block_DK crashes the AICore at runtime, and non-multiple block_DV "
            "silently produces garbage results (no error raised)."
        )
    if DK % block_DK != 0 or DV % block_DV != 0:
        raise ValueError(
            f"DK must be divisible by block_DK and DV by block_DV, got DK={DK}, "
            f"block_DK={block_DK}, DV={DV}, block_DV={block_DV}. DK is the Cube "
            "reduction dim: a partial DK tile would make every dot product "
            "depend on framework bounds-clamping zero-fill, so partial tiles "
            "are rejected instead of relying on clamping behavior."
        )
    if accum_dtype != "float32":
        raise ValueError(
            f"accum_dtype must be 'float32', got {accum_dtype!r}: the AscendC cast path for "
            "non-fp32 accumulation buffers fails at C++ codegen with an obscure "
            "CastIntrinsicsImpl template error."
        )
    if gate_dtype != "float32":
        raise ValueError(
            f"gate_dtype must be 'float32', got {gate_dtype!r}: non-fp32 gate buffers fail "
            "at C++ codegen (same unsupported cast path as accum_dtype)."
        )
    if not isinstance(core_num, int) or core_num < 1:
        raise ValueError(f"core_num must be a positive integer, got {core_num!r}.")
    if not isinstance(chunks_per_tile, int) or chunks_per_tile < 1:
        raise ValueError(f"chunks_per_tile must be a positive integer, got {chunks_per_tile!r}.")
    # UB capacity guard. Memory planning (TL_ASCEND_MEMORY_PLANNING) can only
    # reuse buffers whose lifetimes do not overlap, so the peak simultaneously-
    # live UB footprint is a hard lower bound no pass can optimize away.
    # Limit: A2/A3 usable UB = 196,352 B (ASCEND_A2A3_UB_SIZE in
    # src/target/codegen_ascend.cc); A5 has 262,144 B, so the A2/A3 limit is
    # the conservative one. Per-phase peaks (S = chunk_size, DVb = block_DV;
    # accum/gate are fp32-enforced above; tril mask is packed uint8, S*(S//8)):
    #   init : row_1d + col_1d + row_2d + col_2d + tril
    #          = 8*S^2 + S*(S//8) + 8*S
    #   gate : O_ub + A_ub + G_2d + G_col + G_row + G_1d + tril   (use_g only)
    #          = 8*S*DVb + 12*S^2 + S*(S//8) + 4*S
    #   comp : A_ub + A_cast + A_delta = (8 + in_bytes)*S^2 (+ tril)
    #   out  : O_ub + AV_ub = 8*S*DVb (+ tril; covers O_ub + output cast too)
    # E.g. S=128, DVb=128, use_g=True -> gate peak 330,240 B > 196,352 B:
    # deterministically overflows UB and must be rejected at entry.
    _dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}
    in_bytes = _dtype_bytes.get(input_dtype, 2)
    ub_capacity = 196352
    tril_bytes = chunk_size * (chunk_size // 8)
    peak_init = 8 * chunk_size * chunk_size + tril_bytes + 8 * chunk_size
    peak_comp = (8 + in_bytes) * chunk_size * chunk_size + tril_bytes
    peak_out = 8 * chunk_size * block_DV + tril_bytes
    peak_gate = 8 * chunk_size * block_DV + 12 * chunk_size * chunk_size + tril_bytes + 4 * chunk_size if use_g else 0
    peak_ub = max(peak_init, peak_comp, peak_out, peak_gate)
    if peak_ub > ub_capacity:
        raise ValueError(
            f"UB capacity exceeded: chunk_size={chunk_size}, block_DV={block_DV}, "
            f"use_g={use_g}, input_dtype={input_dtype!r} requires >= {peak_ub} "
            f"simultaneously-live UB bytes, over the {ub_capacity}-byte A2/A3 UB "
            "limit (ASCEND_A2A3_UB_SIZE). Memory planning cannot reuse "
            "simultaneously-live buffers; with use_g the gate phase alone needs "
            "8*chunk_size*block_DV + 12*chunk_size^2 bytes (O_ub + A_ub + G_2d + "
            "G_col + G_row). Reduce chunk_size or block_DV, or set use_g=False."
        )
    block_S = chunk_size
    BS = (S + block_S - 1) // block_S
    n_chunk_groups = (BS + chunks_per_tile - 1) // chunks_per_tile
    DV_blocks = (DV + block_DV - 1) // block_DV
    total_tiles = DV_blocks * n_chunk_groups * B * H
    waves = (total_tiles + core_num - 1) // core_num

    @T.prim_func
    def kernel(
        Q: T.Tensor((B, H, S, DK), input_dtype),
        K: T.Tensor((B, H, S, DK), input_dtype),
        V: T.Tensor((B, H, S, DV), input_dtype),
        HIDDEN: T.Tensor((B, H, BS, DK, DV), input_dtype),
        G_T: T.Tensor((B, H, S), gate_dtype),
        O: T.Tensor((B, S, H, DV), output_dtype),
    ):
        with T.Kernel(core_num, threads=1, is_npu=True) as (cid):
            Q_l1 = T.alloc_L1((block_S, block_DK), input_dtype)
            K_l1 = T.alloc_L1((block_S, block_DK), input_dtype)
            H_l1 = T.alloc_L1((block_DK, block_DV), input_dtype)
            V_l1 = T.alloc_L1((block_S, block_DV), input_dtype)
            A_l1 = T.alloc_L1((block_S, block_S), input_dtype)
            A_delta_l1 = T.alloc_L1((block_S, block_S), input_dtype)

            Q_l0a = T.alloc_L0A((block_S, block_DK), input_dtype)
            a_l0a = T.alloc_L0A((block_S, block_S), input_dtype)
            a_delta_l0a = T.alloc_L0A((block_S, block_S), input_dtype)

            h_l0b = T.alloc_L0B((block_DK, block_DV), input_dtype)
            k_l0b = T.alloc_L0B((block_DK, block_S), input_dtype)
            v_l0b = T.alloc_L0B((block_S, block_DV), input_dtype)

            O_l0c = T.alloc_L0C((block_S, block_DV), accum_dtype)
            A_l0c = T.alloc_L0C((block_S, block_S), accum_dtype)
            AV_l0c = T.alloc_L0C((block_S, block_DV), accum_dtype)

            O_ub = T.alloc_ub((block_S, block_DV), accum_dtype)
            A_ub = T.alloc_ub((block_S, block_S), accum_dtype)
            AV_ub = T.alloc_ub((block_S, block_DV), accum_dtype)
            A_bf16_ub = T.alloc_ub((block_S, block_S), input_dtype)
            A_delta_ub = T.alloc_ub((block_S, block_S), accum_dtype)
            A_delta_bf16_ub = T.alloc_ub((block_S, block_S), input_dtype)
            O_bf16_ub = T.alloc_ub((block_S, block_DV), output_dtype)

            G_1d_ub = T.alloc_ub((block_S,), gate_dtype)
            G_2d_ub = T.alloc_ub((block_S, block_DV), gate_dtype)
            G_col_2d = T.alloc_ub((block_S, block_S), gate_dtype)
            G_row_2d = T.alloc_ub((block_S, block_S), gate_dtype)

            row_1d = T.alloc_ub((block_S,), accum_dtype)
            col_1d = T.alloc_ub((block_S,), accum_dtype)
            row_2d = T.alloc_ub((block_S, block_S), accum_dtype)
            col_2d = T.alloc_ub((block_S, block_S), accum_dtype)
            tril_mask = T.alloc_ub((block_S, block_S // 8), "uint8")

            T.tile.arith_progression(row_1d, 0, 1, block_S)
            T.tile.arith_progression(col_1d, 0, 1, block_S)
            T.tile.broadcast(row_2d, row_1d, axis=1)
            T.tile.broadcast(col_2d, col_1d, axis=0)
            T.tile.compare(tril_mask, row_2d, col_2d, "GE")

            for wave in T.serial(waves):
                pid = wave * core_num + cid
                if pid < total_tiles:
                    dv_ncg_bh = n_chunk_groups * B * H
                    bv = pid // dv_ncg_bh
                    remainder = pid % dv_ncg_bh
                    bs_base = remainder // (B * H)
                    bbh = remainder % (B * H)
                    bb = bbh // H
                    bh = bbh % H
                    v0 = bv * block_DV

                    for ci in T.serial(chunks_per_tile):
                        bs = bs_base * chunks_per_tile + ci
                        if bs < BS:
                            s0 = bs * block_S

                            if use_g:
                                T.copy(G_T[bb, bh, s0 : s0 + block_S], G_1d_ub)

                            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=0):
                                k0 = i_k * block_DK
                                T.copy(Q[bb, bh, s0 : s0 + block_S, k0 : k0 + block_DK], Q_l1)
                                T.copy(K[bb, bh, s0 : s0 + block_S, k0 : k0 + block_DK], K_l1)
                                T.copy(
                                    HIDDEN[bb, bh, bs, k0 : k0 + block_DK, v0 : v0 + block_DV],
                                    H_l1,
                                )
                                T.copy(Q_l1, Q_l0a)
                                T.copy(H_l1, h_l0b)
                                T.mma(Q_l0a, h_l0b, O_l0c, init=(i_k == 0))
                                T.copy(K_l1, k_l0b, transpose=True)
                                T.mma(Q_l0a, k_l0b, A_l0c, init=(i_k == 0))

                            T.copy(O_l0c, O_ub)
                            T.copy(A_l0c, A_ub)

                            if use_g:
                                T.tile.broadcast(G_2d_ub, G_1d_ub, axis=1)
                                T.tile.broadcast(G_col_2d, G_1d_ub, axis=1)
                                T.tile.broadcast(G_row_2d, G_1d_ub, axis=0)

                                T.tile.exp(G_2d_ub, G_2d_ub)
                                T.tile.mul(O_ub, O_ub, G_2d_ub)

                                # A = tril((Q @ K^T) * exp(G_i - G_j)): the causal
                                # mask is purely lower-triangular (i >= j), same
                                # semantics as the repo GDN reference
                                # (examples/linear_attention_and_rnn/gdn/gdn_chunk_o.py).
                                # Real GDN gates are cumulative log-decay (non-
                                # increasing), so exp(G_i - G_j) <= 1 inside the
                                # kept triangle; arbitrary gates may overflow fp32
                                # exp only when G_i - G_j > ~88 (the golden
                                # reference overflows identically).
                                T.tile.sub(G_col_2d, G_col_2d, G_row_2d)
                                T.tile.exp(G_col_2d, G_col_2d)
                                T.tile.mul(A_ub, A_ub, G_col_2d)
                                T.tile.select(A_ub, tril_mask, A_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                            else:
                                T.tile.select(A_ub, tril_mask, A_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")

                            T.copy(A_ub, A_bf16_ub)
                            T.copy(A_bf16_ub, A_l1)

                            # Compensated GEMM: delta = A - cast(A_bf16)
                            T.copy(A_bf16_ub, A_delta_ub)
                            T.tile.sub(A_delta_ub, A_ub, A_delta_ub)
                            T.copy(A_delta_ub, A_delta_bf16_ub)
                            T.copy(A_delta_bf16_ub, A_delta_l1)

                            T.copy(V[bb, bh, s0 : s0 + block_S, v0 : v0 + block_DV], V_l1)

                            T.copy(A_l1, a_l0a)
                            T.copy(V_l1, v_l0b)
                            T.mma(a_l0a, v_l0b, AV_l0c, init=True)
                            T.copy(A_delta_l1, a_delta_l0a)
                            T.mma(a_delta_l0a, v_l0b, AV_l0c, init=False)

                            T.copy(AV_l0c, AV_ub)
                            T.tile.add(O_ub, O_ub, AV_ub)

                            T.copy(O_ub, O_bf16_ub)
                            T.copy(O_bf16_ub, O[bb, s0 : s0 + block_S, bh, v0 : v0 + block_DV])

    return kernel


def _prepare_inputs(B, S, H, DK, DV, chunk_size, scale=None, device="npu", dtype=torch.bfloat16):
    """Prepare test inputs on CPU, then H2D.

    Q/K/V/HIDDEN are permuted to BHSD layout for contiguous S-dim DMA.
    O output stays BSHD. Pads S to S_padded = ceildiv(S, chunk_size) * chunk_size
    (zero-pad tail) so the kernel always sees a divisible S. Returns S_padded.

    Scale folding: V and HIDDEN are pre-scaled by `scale` on the host (fp32
    multiply then cast to `dtype`), so the kernel does not apply scale.
    If `scale` is None, defaults to DK**-0.5 (standard attention scale).
    `dtype` sets the Q/K/V/HIDDEN element type (bfloat16 or float16); G/G_T
    are always float32 (gate_dtype contract).
    """
    BS = chunk_size
    n_c = math.ceil(S / BS)
    S_padded = n_c * BS
    if scale is None:
        scale = DK**-0.5
    Q = torch.randn(B, S_padded, H, DK, dtype=dtype, device="cpu")
    K = torch.randn(B, S_padded, H, DK, dtype=dtype, device="cpu")
    V = (torch.randn(B, S_padded, H, DV, dtype=torch.float32, device="cpu") * scale).to(dtype)
    HIDDEN = (torch.randn(B, n_c, H, DK, DV, dtype=torch.float32, device="cpu") * scale).to(dtype)
    G_cpu = torch.randn(B, S_padded, H, dtype=torch.float32, device="cpu")
    if S_padded > S:
        Q[:, S:, :, :] = 0
        K[:, S:, :, :] = 0
        V[:, S:, :, :] = 0
        G_cpu[:, S:, :] = 0
    Q = Q.permute(0, 2, 1, 3).contiguous().to(device)  # (B,H,S,DK)
    K = K.permute(0, 2, 1, 3).contiguous().to(device)  # (B,H,S,DK)
    V = V.permute(0, 2, 1, 3).contiguous().to(device)  # (B,H,S,DV)
    HIDDEN = HIDDEN.permute(0, 2, 1, 3, 4).contiguous().to(device)  # (B,H,BS,DK,DV)
    G = G_cpu.to(device)  # (B,S,H)
    G_T = G_cpu.permute(0, 2, 1).contiguous().to(device)  # (B,H,S)
    return Q, K, V, HIDDEN, G, G_T, S_padded


if __name__ == "__main__":
    tilelang.disable_cache()
    torch.set_default_device("npu")

    B, S, H, DK, DV, cs = 1, 32768, 32, 128, 128, 64

    torch.manual_seed(0)
    Q, K, V, HIDDEN, G, G_T, S_act = _prepare_inputs(B, S, H, DK, DV, cs)
    kernel = chunk_o(B, S_act, H, DK, DV, "bfloat16", "bfloat16", "float32", "float32", cs, True, 128, 128)
    O_actual = kernel(Q, K, V, HIDDEN, G_T)
    torch.npu.synchronize()

    ok = True
    if O_actual.shape != (B, S_act, H, DV):
        print(f"Shape mismatch: got {O_actual.shape}, expected {(B, S_act, H, DV)}")
        ok = False
    if O_actual.dtype != torch.bfloat16:
        print(f"Dtype mismatch: got {O_actual.dtype}, expected bfloat16")
        ok = False
    if not torch.isfinite(O_actual).all():
        print("Output contains inf/nan values")
        ok = False

    if ok:
        print(f"Output: shape={tuple(O_actual.shape)}, dtype={O_actual.dtype}, all_finite=True")
        print("Smoke Test Passed!")
    else:
        print("Smoke Test FAILED!")
        sys.exit(1)
