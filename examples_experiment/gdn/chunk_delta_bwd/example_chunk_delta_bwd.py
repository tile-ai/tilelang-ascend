"""Chunk Delta Backward operator (Ascend NPU, Developer mode, fp32 GEMM + transpose_B=True).

Supports chunk_size up to 1024 via M-axis segment streaming (seg_M=128) and
per-segment K-split (seg_K=128). BS loop uses T.Pipelined(num_stages=2) to overlap GM->L1 loads with GEMM compute.

Math (per chunk, reverse scan BS-1 -> 0):
  dh[:, i_s]      = dh_T (store, L0C fp32 -> GM fp32, transposed layout [DV,DK])
  dv              = Kg @ dh + dv_in @ I_DV            (GEMM 1 + 1', M-segmented)
  dv2[:, chunk]   = dv                             (store, M-segmented [seg_M, block_DV])
  dv_T            = I_DV @ dv^T                    (identity GEMM, N-segmented for 2c)
  dh_T            = dh^T @ Dmat^T + dO^T @ Qg + dv_T @ Wn_T^T
                                                      (GEMM 2a/2b/2c, transposed dh output)

Key design decisions:
  - All GEMMs use transpose_B=True (required for fp32 GEMM on NPU).
  - All L1/L0C buffers are fp32 (NOT bf16).
  - State relay via fp32 transit buffers (L0C fp32 -> GM fp32 -> L1 fp32).
  - threads=1 + is_npu=True.
  - 1 transpose GEMM per chunk (dv only; dh uses transposed L0C layout).
"""

import ctypes

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Constant identity matrices cached across calls.
_EYE_CACHE = {}


def _eye(n, dtype, device):
    key = (n, dtype, str(device))
    if key not in _EYE_CACHE:
        _EYE_CACHE[key] = torch.eye(n, dtype=dtype, device=device)
    return _EYE_CACHE[key]


class _FastLaunch:
    """Direct .so launch path — bypasses JITKernel.__call__ overhead.

    Reuses compiled artifacts (lib, prim_func, result_idx) and calls lib.call
    directly with ctypes pointer args. Falls back to normal JITKernel call
    path if construction fails.
    """

    def __init__(self, jit_kernel):
        ad = jit_kernel.adapter
        self._lib = ad.lib
        pf = ad.prim_func
        bmap = pf.buffer_map
        from tilelang.utils.tensor import map_torch_type

        specs = []
        for p in pf.params:
            buf = bmap.get(p)
            if buf is None:
                specs.append(None)
                continue
            shape = tuple(int(s.value) if hasattr(s, "value") else int(s) for s in buf.shape)
            specs.append((shape, map_torch_type(buf.dtype)))
        self._specs = specs
        self._result_idx = set(ad.result_idx)
        if getattr(ad, "workspace_idx", None) or getattr(ad, "auto_gm_idx", None):
            raise NotImplementedError("fast path: workspace params unsupported")
        if getattr(ad, "dynamic_symbolic_map", None):
            raise NotImplementedError("fast path: dynamic shapes unsupported")

    def __call__(self, *inputs, stream_ptr):
        args = []
        outputs = []
        ins = 0
        for i, spec in enumerate(self._specs):
            if spec is None:
                args.append(int(inputs[ins]))
                ins += 1
                continue
            shape, dtype = spec
            if i in self._result_idx:
                out = torch.empty(shape, dtype=dtype, device=inputs[0].device)
                outputs.append(out)
                args.append(ctypes.c_void_p(out.data_ptr()))
            else:
                t = inputs[ins]
                ins += 1
                args.append(ctypes.c_void_p(t.data_ptr()))
        args.append(ctypes.c_void_p(stream_ptr))
        self._lib.call(*args)
        return tuple(outputs)


_FAST_CACHE = {}


def _fast(key, jit_kernel):
    """Cached _FastLaunch with permanent fallback to the normal path."""
    fl = _FAST_CACHE.get(key)
    if fl is None:
        try:
            fl = _FastLaunch(jit_kernel)
        except Exception:
            fl = False  # permanent fallback marker
        _FAST_CACHE[key] = fl
    return fl


@tilelang.jit(out_idx=[12, 13, 14], pass_configs=pass_configs)
def chunk_delta_bwd_kernel(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    scale,
    use_g,
    use_initial_state,
    use_final_state_gradient,
    input_dtype,
    output_dtype,
    accum_dtype,
    state_dtype,
    block_DV=64,
):
    """Chunk delta backward kernel (Developer mode, fp32 GEMM, transpose_B=True).

    BS loop uses T.Pipelined(num_stages=2) to overlap next chunk GM->L1
    loads with current chunk GEMM compute. MEMORY_PLANNING reuses segment
    buffers across pipeline stages (compiled without L1 overflow).

    M-axis segment streaming: seg_M=min(block_S,128), nsegs_M=block_S//seg_M.
    GEMM 1/1' and dv transpose loop over M-segments. Segment-sized L1 buffers.

    Per-segment K-split for GEMM 2b/2c: K=block_S split into seg_K=128
    segments.     GEMM 1' uses dv_in @ I_DV^T (K=block_DV).

    GEMMs per chunk: 5 compute + 1 dv transpose + 1 init = 7 total.
    GEMM 1/1' and dv transpose use M-segment split for L1 capacity.
    GEMM 2b/2c use per-segment K-split for L1 capacity (nsegs_K segments).

    Args:
        B, S, H, DK, DV: tensor dimensions.
        chunk_size: C, chunk length (block_S).
        scale: not used in kernel (precomputed into Qg_T by host).
        use_g: not used in kernel (precomputed into gates by host).
        use_initial_state: whether to store dh0 output.
        use_final_state_gradient: not used in kernel (host passes dht_T=zeros when False).
        input_dtype: GEMM input dtype string (float32).
        output_dtype: output dtype string (float32).
        accum_dtype: L0C accum dtype string (float32).
        state_dtype: state/transit dtype string (float32).
        block_DV: DV tiling block size.

    Returns:
        Compiled kernel callable.
    """
    block_S = chunk_size
    BS = S // block_S
    bv_num = T.ceildiv(DV, block_DV)
    total_blocks = bv_num * B * H

    # Per-segment streaming: cap K-dim segment at 128 for GEMM 2b/2c
    # (K=block_S, split into nsegs_K segments of seg_K each)
    # For C<=128: nsegs_K=1 (no split).
    # For C=256: nsegs_K=2 (seg_K=128). For C=512: nsegs_K=4.
    seg_K = min(block_S, 128)
    nsegs_K = block_S // seg_K

    # M-axis segmentation: cap M-dim segment at 128 for GEMM 1/1' and dv transpose
    # (M=block_S for GEMM 1/1', N=block_S for dv transpose)
    # For C<=128: nsegs_M=1 (no split).
    # For C=256: nsegs_M=2. For C=512: nsegs_M=4.
    seg_M = min(block_S, 128)
    nsegs_M = block_S // seg_M

    # kL0Size for GEMMs with M=seg_M (GEMM 1: Kg @ dh, K=DK=128):
    # L0A ping-pong slot budget = 32KB (half of 64KB L0A when kL0Size < K).
    # M * kL0Size * 4 <= 32768 → kL0Size <= 32768 / (M * 4)
    # 32768 = 32KB / 4 bytes (fp32). seg_M=128: kL0Size=32 (4 K-splits).
    kL0Size_bigM = min(32, 32768 // (seg_M * 4))

    @T.prim_func
    def kernel(
        # --- Inputs (host preprocessed, ALL fp32) ---
        Kg: T.Tensor((B, BS, H, block_S, DK), input_dtype),  # type: ignore
        Qg_T: T.Tensor((B, BS, H, DK, block_S), input_dtype),  # type: ignore
        Wn_T: T.Tensor((B, BS, H, DK, block_S), input_dtype),  # type: ignore
        Dmat: T.Tensor((B, BS, H, DK, DK), input_dtype),  # type: ignore
        dO_T: T.Tensor((B, BS, H, DV, block_S), input_dtype),  # type: ignore  # chunk-major layout
        dv_in: T.Tensor((B, H, S, DV), input_dtype),  # type: ignore
        dht_T: T.Tensor((B, H, DV, DK), input_dtype),  # type: ignore
        I_DK: T.Tensor((DK, DK), input_dtype),  # type: ignore
        I_DV: T.Tensor((block_DV, block_DV), input_dtype),  # type: ignore
        # --- Transit buffers (GM, fp32, C->C relay) ---
        transit_dh_T: T.Tensor((total_blocks, block_DV, DK), state_dtype),  # type: ignore
        transit_dv: T.Tensor((total_blocks, block_S, block_DV), state_dtype),  # type: ignore
        transit_dv_T: T.Tensor((total_blocks, block_DV, block_S), state_dtype),  # type: ignore
        # --- Outputs ---
        dh: T.Tensor((B, BS, H, DV, DK), output_dtype),  # type: ignore
        dh0: T.Tensor((B, H, DV, DK), state_dtype),  # type: ignore
        dv2: T.Tensor((B, S, H, DV), output_dtype),  # type: ignore
    ):
        with T.Kernel(total_blocks, threads=1, is_npu=True) as (cid):
            bv = cid % bv_num
            bbh = cid // bv_num
            bb = bbh // H
            bh = bbh % H
            dv_start = bv * block_DV

            # --- L1 (Cube) buffers, ALL fp32 ---
            # Always-live constants
            I_DK_l1 = T.alloc_shared((DK, DK), input_dtype)
            I_DV_l1 = T.alloc_shared((block_DV, block_DV), input_dtype)
            # dh_l1: transposed dh state [block_DV, DK] (GEMM B operand)
            dh_l1 = T.alloc_shared((block_DV, DK), input_dtype)
            # Full-size L1 buffers (K=DK or block_DV, not affected by C)
            Dmat_l1 = T.alloc_shared((DK, DK), input_dtype)
            # M-segment-sized L1 buffers (M=seg_M, replacing full block_S)
            # For C<=128: seg_M=block_S. For C=256/512: seg_M=128.
            Kg_seg = T.alloc_shared((seg_M, DK), input_dtype)
            dv_in_seg = T.alloc_shared((seg_M, block_DV), input_dtype)
            dv_l1_orig_seg = T.alloc_shared((seg_M, block_DV), input_dtype)
            # Segment-sized L1 buffers (K=block_S, split into seg_K)
            Qg_T_seg = T.alloc_shared((DK, seg_K), input_dtype)
            Wn_T_seg = T.alloc_shared((DK, seg_K), input_dtype)
            dO_T_seg = T.alloc_shared((block_DV, seg_K), input_dtype)
            dv_T_seg = T.alloc_shared((block_DV, seg_K), input_dtype)

            # --- L0C (Cube output) buffers, fp32 ---
            # dh_T_l0c: transposed dh [block_DV, DK] (output of GEMM 2a/2b/2c)
            dh_T_l0c = T.alloc_fragment((block_DV, DK), accum_dtype)
            # dv_l0c_seg [seg_M, block_DV] (M-segment-sized)
            dv_l0c_seg = T.alloc_fragment((seg_M, block_DV), accum_dtype)
            # dv_T_l0c_seg [block_DV, seg_M] (N-segment-sized)
            dv_T_l0c_seg = T.alloc_fragment((block_DV, seg_M), accum_dtype)

            # --- Load constants (GM -> L1, once) ---
            T.copy(I_DK, I_DK_l1)
            T.copy(I_DV, I_DV_l1)

            # --- Initialize dh_l1 from dht_T (transposed layout) ---
            T.copy(
                dht_T[bb, bh, dv_start : dv_start + block_DV, 0:DK],
                dh_l1,
            )

            # Init dh_T_l0c = dh @ I_DK (L0C requires GEMM write)
            T.gemm_v0(dh_l1, I_DK_l1, dh_T_l0c, transpose_B=True, init=True, kL0Size=32)

            # --- Chunk loop (reverse scan, software pipelined) ---
            # T.Pipelined(num_stages=2) overlaps next chunk's GM->L1 loads
            # with current chunk's GEMM compute. Safe on pure Cube kernel.
            # MEMORY_PLANNING reuses segment buffers across pipeline stages.
            for i_s in T.Pipelined(BS, num_stages=2):
                i_s_inv = BS - i_s - 1
                cs = i_s_inv * block_S

                # 1. Store dh (L0C fp32 -> GM fp32, transposed layout [DV, DK])
                T.copy(
                    dh_T_l0c,
                    dh[bb, i_s_inv, bh, dv_start : dv_start + block_DV, 0:DK],
                )

                # 2-5. GEMM 1 + 1' + store dv2 + relay dv (M-segmented)
                # For C<=128: nsegs_M=1. C=256: nsegs_M=2. C=512: nsegs_M=4.
                for m_seg in T.unroll(nsegs_M):
                    ms = m_seg * seg_M
                    # Load Kg segment (GM fp32 -> L1 fp32)
                    T.copy(
                        Kg[bb, i_s_inv, bh, ms : ms + seg_M, 0:DK],
                        Kg_seg,
                    )
                    # Load dv_in segment (GM fp32 -> L1 fp32)
                    T.copy(
                        dv_in[
                            bb,
                            bh,
                            cs + ms : cs + ms + seg_M,
                            dv_start : dv_start + block_DV,
                        ],
                        dv_in_seg,
                    )
                    # GEMM 1: dv = Kg @ dh (init)
                    T.gemm_v0(
                        Kg_seg,
                        dh_l1,
                        dv_l0c_seg,
                        transpose_B=True,
                        init=True,
                        kL0Size=kL0Size_bigM,
                    )
                    # GEMM 1': dv += dv_in @ I_DV (accumulate)
                    T.gemm_v0(
                        dv_in_seg,
                        I_DV_l1,
                        dv_l0c_seg,
                        transpose_B=True,
                        init=False,
                        kL0Size=32,
                    )
                    # Store dv2 segment (L0C fp32 -> GM fp32)
                    T.copy(
                        dv_l0c_seg,
                        dv2[
                            bb,
                            cs + ms : cs + ms + seg_M,
                            bh,
                            dv_start : dv_start + block_DV,
                        ],
                    )
                    # Relay dv segment to transit_dv (L0C -> GM, for transpose GEMM)
                    T.copy(
                        dv_l0c_seg,
                        transit_dv[cid, ms : ms + seg_M, 0:block_DV],
                    )

                # 6. Load Dmat for GEMM 2a
                T.copy(Dmat[bb, i_s_inv, bh, 0:DK, 0:DK], Dmat_l1)

                # 7. GEMM 2a: dh = Dmat @ dh (init)
                T.gemm_v0(
                    dh_l1,
                    Dmat_l1,
                    dh_T_l0c,
                    transpose_B=True,
                    init=True,
                    kL0Size=32,
                )
                # 8. GEMM 2b: dh += dO^T @ Qg (K-split, accumulate on 2a)
                for seg in T.unroll(nsegs_K):
                    ks = seg * seg_K
                    T.copy(
                        dO_T[
                            bb,
                            i_s_inv,
                            bh,
                            dv_start : dv_start + block_DV,
                            ks : ks + seg_K,
                        ],
                        dO_T_seg,
                    )
                    T.copy(
                        Qg_T[bb, i_s_inv, bh, 0:DK, ks : ks + seg_K],
                        Qg_T_seg,
                    )
                    T.gemm_v0(
                        dO_T_seg,
                        Qg_T_seg,
                        dh_T_l0c,
                        transpose_B=True,
                        init=False,
                        kL0Size=32,
                    )

                # 9. Transpose dv for GEMM 2c (N-segmented)
                for n_seg in T.unroll(nsegs_M):
                    ns = n_seg * seg_M
                    # Load dv segment from transit_dv (GM -> L1)
                    T.copy(
                        transit_dv[cid, ns : ns + seg_M, 0:block_DV],
                        dv_l1_orig_seg,
                    )
                    # dv_T = I_DV @ dv^T (transpose via identity GEMM)
                    T.gemm_v0(
                        I_DV_l1,
                        dv_l1_orig_seg,
                        dv_T_l0c_seg,
                        transpose_B=True,
                        init=True,
                        kL0Size=32,
                    )
                    # Relay dv_T segment to GM (transit_dv_T [block_DV, block_S])
                    T.copy(
                        dv_T_l0c_seg,
                        transit_dv_T[cid, 0:block_DV, ns : ns + seg_M],
                    )

                # 10. GEMM 2c: dh += dv^T @ Wn_T (K-split, accumulate on 2b)
                for seg in T.unroll(nsegs_K):
                    ks = seg * seg_K
                    T.copy(
                        transit_dv_T[cid, 0:block_DV, ks : ks + seg_K],
                        dv_T_seg,
                    )
                    T.copy(
                        Wn_T[bb, i_s_inv, bh, 0:DK, ks : ks + seg_K],
                        Wn_T_seg,
                    )
                    T.gemm_v0(
                        dv_T_seg,
                        Wn_T_seg,
                        dh_T_l0c,
                        transpose_B=True,
                        init=False,
                        kL0Size=32,
                    )

                # 11. Relay dh_T for next iteration (L0C -> GM -> L1)
                T.copy(dh_T_l0c, transit_dh_T[cid, 0:block_DV, 0:DK])
                T.copy(transit_dh_T[cid, 0:block_DV, 0:DK], dh_l1)

            # 12. Store dh0 (L0C fp32 -> GM fp32, transposed layout [DV, DK])
            if use_initial_state:
                T.copy(
                    dh_T_l0c,
                    dh0[bb, bh, dv_start : dv_start + block_DV, 0:DK],
                )

    return kernel


# ============================================================================
# Prep kernel: fuse gate compute + permute + fp32 conversion into a single
# NPU Vector kernel launch.
# ============================================================================


@tilelang.jit(out_idx=[8, 9, 10, 11, 12, 13, 14], pass_configs=pass_configs)
def chunk_delta_bwd_prep(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size=64,
    scale=0.0884,
    input_dtype="bfloat16",
    gate_dtype="float32",
    output_dtype="float32",
    prep_block_S=64,
):
    """fp32 prep kernel: Kg + Qg_T + Wn_T + Dmat + dv_in + dO_T + dht_T in ONE launch.

    Each block handles one (batch, chunk, head) work item, loading K/Q/W
    tiles in bf16, converting to fp32, applying gates, and storing results
    in chunk-major layout. Qg_T and Wn_T are stored transposed [DK, block_S]
    to match the scan kernel's transpose_B=True GEMM layout.

    Also converts dv [B,S,H,DV] bf16 → dv_in [B,H,S,DV] fp32 in the same kernel.

    Also converts dO [B,S,H,DV] bf16 → dO_T [B,BS,H,DV,block_S] fp32 and dht [B,H,DK,DV] bf16 → dht_T
    [B,H,DV,DK] fp32 in the same kernel.
    dO_T is computed per sub-block. dht_T is computed only when cc==0 (one block per (batch,head)),
    in two DK//2 halves.

    Sub-blocked data processing at prep_block_S=64 to fit C=128 in UB.
    Gate computation (1D) stays at full chunk_size; 2D data buffers loop over
    nsegs = chunk_size // prep_block_S sub-blocks. For C=64, nsegs=1.

    Store strategy: direct GM store after each computation. Qg_T/Wn_T use
    a shared transpose buffer (one live at a time).

    Outputs (all fp32):
      Kg      = K * gate_dv                          [B, BS, H, block_S, DK]
      Qg_T    = (Q * exp(G) * scale)^T               [B, BS, H, DK, block_S]
      Wn_T    = (-W)^T                               [B, BS, H, DK, block_S]
      Dmat    = diag(exp(G_last)) * I                [B, BS, H, DK, DK]
      dv_in   = dv.float() permuted to [B, H, S, DV]  [B, H, S, DV]
      dO_T    = dO.float()^T chunk-major [B, BS, H, DV, block_S]
      dht_T   = dht.float()^T            [B, H, DV, DK]
    """
    block_S = chunk_size
    BS = S // block_S
    nsegs = chunk_size // prep_block_S
    max_dim = max(DK, DV)  # accommodate both DK and DV in shared buffers

    @T.prim_func
    def prep(
        K: T.Tensor((B, S, H, DK), input_dtype),  # type: ignore
        Q: T.Tensor((B, S, H, DK), input_dtype),  # type: ignore
        W: T.Tensor((B, S, H, DK), input_dtype),  # type: ignore
        Gc: T.Tensor((B, BS, H, block_S), gate_dtype),  # type: ignore
        eye_f32_gm: T.Tensor((DK, DK), gate_dtype),  # type: ignore
        dv: T.Tensor((B, S, H, DV), input_dtype),  # type: ignore
        # dO + dht inputs (bf16, for in-kernel permute+float)
        dO: T.Tensor((B, S, H, DV), input_dtype),  # type: ignore
        dht: T.Tensor((B, H, DK, DV), input_dtype),  # type: ignore
        # --- Outputs (all fp32) ---
        Kg_out: T.Tensor((B, BS, H, block_S, DK), output_dtype),  # type: ignore
        Qg_T_out: T.Tensor((B, BS, H, DK, block_S), output_dtype),  # type: ignore
        Wn_T_out: T.Tensor((B, BS, H, DK, block_S), output_dtype),  # type: ignore
        Dmat_out: T.Tensor((B, BS, H, DK, DK), output_dtype),  # type: ignore
        dv_in_out: T.Tensor((B, H, S, DV), output_dtype),  # type: ignore
        # dO_T + dht_T outputs (fp32, permuted layout for scan kernel)
        # dO_T uses chunk-major [B, BS, H, DV, block_S] layout (same as Qg_T/Wn_T)
        # for efficient prep-kernel writes (stride=block_S vs stride=S).
        dO_T_out: T.Tensor((B, BS, H, DV, block_S), output_dtype),  # type: ignore
        dht_T_out: T.Tensor((B, H, DV, DK), output_dtype),  # type: ignore
    ):
        with T.Kernel(B * BS * H, threads=1, is_npu=True) as (cid):
            bb = cid // (BS * H)
            cc = (cid // H) % BS
            hh = cid % H
            cs = cc * block_S

            # --- Shared gate section (1D, full chunk_size) ---
            G_ub = T.alloc_shared((block_S,), dtype=gate_dtype)
            glast_ub = T.alloc_shared((block_S,), dtype=gate_dtype)
            d_ub = T.alloc_shared((block_S,), dtype=gate_dtype)
            e_ub = T.alloc_shared((block_S,), dtype=gate_dtype)
            gmask_ub = T.alloc_shared((block_S // 8,), dtype="uint8")
            gate_dv_ub = T.alloc_shared((block_S,), dtype=gate_dtype)
            gexp_ub = T.alloc_shared((block_S,), dtype=gate_dtype)
            gate_q_ub = T.alloc_shared((block_S,), dtype=gate_dtype)
            gqs_ub = T.alloc_shared((block_S,), dtype=gate_dtype)

            T.copy(Gc[bb, cc, hh, 0:block_S], G_ub)
            g_last = G_ub[block_S - 1]
            # gate_dv = exp(g_last - G) if (g_last - G) <= 0 else 0
            T.tile.fill(glast_ub, g_last)
            T.tile.sub(d_ub, glast_ub, G_ub)
            T.tile.compare(gmask_ub, d_ub, T.float32(0.0), "LE")
            T.tile.exp(e_ub, d_ub)
            T.tile.select(
                gate_dv_ub,
                gmask_ub,
                e_ub,
                T.float32(0.0),
                "VSEL_TENSOR_SCALAR_MODE",
            )
            # g_last_exp = exp(g_last)
            T.tile.fill(gexp_ub, g_last)
            T.tile.exp(gexp_ub, gexp_ub)
            g_last_exp = gexp_ub[0]
            # gate_q_scaled = exp(G) * scale
            T.tile.exp(gate_q_ub, G_ub)
            T.tile.mul(gqs_ub, gate_q_ub, scale)

            # --- Work buffers (2D, sub-blocked at prep_block_S) ---
            # Single shared src buffer (reused for K/Q/W/dv, one at a time)
            # sized to max(DK, DV) to accommodate dv conversion
            src_ub = T.alloc_shared((prep_block_S, max_dim), dtype=input_dtype)
            # gate2d sized to max_dim to match src_f32 when DV != DK.
            gate2d = T.alloc_shared((prep_block_S, max_dim), dtype=gate_dtype)
            src_f32 = T.alloc_shared((prep_block_S, max_dim), dtype=gate_dtype)
            # Transpose buffer (reused for Qg_T then Wn_T then dht_T, one live at a time).
            # Sized to max_dim rows: T.tile.transpose writes SRC-width rows to dst.
            trans_ub = T.alloc_shared((max_dim, prep_block_S), dtype=output_dtype)
            # Dedicated transpose buffer for dO_T (non-overlapping lifetime with trans_ub).
            trans_dO_ub = T.alloc_shared((max_dim, prep_block_S), dtype=output_dtype)
            # Gate slice for current sub-block (small, prep_block_S elements)
            gate_slice = T.alloc_shared((prep_block_S,), dtype=gate_dtype)

            # --- Sub-block loop: process chunk in prep_block_S segments ---
            for seg in T.serial(nsegs):
                s = cs + seg * prep_block_S  # S-dim offset in source tensor
                sg = seg * prep_block_S  # offset within chunk output

                # --- Kg = K * gate_dv (fp32, [prep_block_S, DK]) → direct GM store ---
                T.copy(K[bb, s : s + prep_block_S, hh, 0:DK], src_ub)
                T.copy(src_ub, src_f32)
                T.copy(gate_dv_ub[sg : sg + prep_block_S], gate_slice)
                # Broadcast along DK dim (same gate per row).
                T.tile.broadcast(gate2d, gate_slice, axis=1)
                T.tile.mul(src_f32, src_f32, gate2d)
                T.copy(
                    src_f32,
                    Kg_out[bb, cc, hh, sg : sg + prep_block_S, 0:DK],
                )

                # --- Qg_T = (Q * gate_q_scaled)^T (fp32, [DK, prep_block_S]) ---
                T.copy(Q[bb, s : s + prep_block_S, hh, 0:DK], src_ub)
                T.copy(src_ub, src_f32)
                T.copy(gqs_ub[sg : sg + prep_block_S], gate_slice)
                T.tile.broadcast(gate2d, gate_slice, axis=1)
                T.tile.mul(src_f32, src_f32, gate2d)
                T.tile.transpose(trans_ub, src_f32)
                T.copy(
                    trans_ub,
                    Qg_T_out[bb, cc, hh, 0:DK, sg : sg + prep_block_S],
                )

                # --- Wn_T = (-W)^T (fp32, [DK, prep_block_S]) ---
                # Reuse trans_ub (Qg_T already stored to GM)
                T.copy(W[bb, s : s + prep_block_S, hh, 0:DK], src_ub)
                T.copy(src_ub, src_f32)
                T.tile.mul(src_f32, src_f32, T.float32(-1.0))
                T.tile.transpose(trans_ub, src_f32)
                T.copy(
                    trans_ub,
                    Wn_T_out[bb, cc, hh, 0:DK, sg : sg + prep_block_S],
                )

                # --- dv_in = dv.float() (fp32, [prep_block_S, DV]) ---
                # Layout change: [B,S,H,DV] → [B,H,S,DV] (permute S and H dims)
                # No gate application, just bf16→fp32 conversion + layout change.
                # Reuses src_ub/src_f32 (Wn_T already stored to GM).
                # Note: full-buffer copy (src_ub is [prep_block_S, max_dim], dv slice
                # is [prep_block_S, DV]; when DV=max_dim they match exactly).
                T.copy(dv[bb, s : s + prep_block_S, hh, 0:DV], src_ub)
                T.copy(src_ub, src_f32)
                T.copy(
                    src_f32,
                    dv_in_out[bb, hh, s : s + prep_block_S, 0:DV],
                )

                # --- dO_T = dO.float()^T (fp32, [DV, prep_block_S]) ---
                # Layout change: dO [B,S,H,DV] → dO_T [B,BS,H,DV,block_S]
                # (permute 0,2,3,1 + chunk-major, same as Qg_T/Wn_T).
                # bf16→fp32 conversion + transpose (same pattern as Qg_T/Wn_T).
                # Chunk-major layout gives stride=block_S between DV rows (vs stride=S
                # for [B,H,DV,S] layout).
                # dO_T transpose uses dedicated trans_dO_ub.
                # Reuses src_ub/src_f32 (dv_in already stored to GM).
                T.copy(dO[bb, s : s + prep_block_S, hh, 0:DV], src_ub)
                T.copy(src_ub, src_f32)
                T.tile.transpose(trans_dO_ub, src_f32)
                T.copy(
                    trans_dO_ub,
                    dO_T_out[bb, cc, hh, 0:DV, sg : sg + prep_block_S],
                )

            # --- Dmat = exp(G_last) * I (fp32, [DK, DK]) → direct GM store ---
            # Load eye halves into work buffer, multiply in-place, store to GM
            Dmat_h = T.alloc_shared((DK // 2, DK), dtype=gate_dtype)
            T.copy(eye_f32_gm[0 : DK // 2, 0:DK], Dmat_h)
            T.tile.mul(Dmat_h, Dmat_h, g_last_exp)
            T.copy(Dmat_h, Dmat_out[bb, cc, hh, 0 : DK // 2, 0:DK])
            T.copy(eye_f32_gm[DK // 2 : DK, 0:DK], Dmat_h)
            T.tile.mul(Dmat_h, Dmat_h, g_last_exp)
            T.copy(Dmat_h, Dmat_out[bb, cc, hh, DK // 2 : DK, 0:DK])

            # --- dht_T = dht.float()^T (fp32, [DV, DK]) ---
            # Only when cc==0 (one block per (batch,head) computes dht_T).
            # Layout change: dht [B,H,DK,DV] → dht_T [B,H,DV,DK] (permute 0,1,3,2)
            # bf16→fp32 conversion + transpose, in two DK//2 halves.
            # Reuses src_ub/src_f32/trans_ub (Dmat already stored to GM).
            # Note: host passes zeros for dht when use_final_state_gradient=False,
            # so dht_T_out will be zeros (transpose of zeros = zeros).
            if cc == 0:
                # Half 1: dht[bb,hh,0:DK//2,0:DV] → transpose → dht_T_out[bb,hh,0:DV,0:DK//2]
                T.copy(dht[bb, hh, 0 : DK // 2, 0:DV], src_ub)
                T.copy(src_ub, src_f32)
                T.tile.transpose(trans_ub, src_f32)
                T.copy(
                    trans_ub,
                    dht_T_out[bb, hh, 0:DV, 0 : DK // 2],
                )
                # Half 2: dht[bb,hh,DK//2:DK,0:DV] → transpose → dht_T_out[bb,hh,0:DV,DK//2:DK]
                T.copy(dht[bb, hh, DK // 2 : DK, 0:DV], src_ub)
                T.copy(src_ub, src_f32)
                T.tile.transpose(trans_ub, src_f32)
                T.copy(
                    trans_ub,
                    dht_T_out[bb, hh, 0:DV, DK // 2 : DK],
                )

    return prep


# ============================================================================
# Host preprocessing helpers
# ============================================================================


def chunk_local_cumsum(g, chunk_size):
    """Compute chunk-local cumulative sum."""
    B, S, H = g.shape
    chunk_num = S // chunk_size
    g = g.view(B, chunk_num, chunk_size, H)
    g_sum = torch.cumsum(g, dim=2)
    return g_sum.view(B, S, H)


def prepare_bwd_gates(K, W, Q, G, scale, chunk_size, use_g=True):
    """Host precompute: Kg, Qg_T, Wn_T, Dmat (all fp32).

    Permute before .float() so dtype conversion also does layout
    rearrangement, eliminating separate .contiguous() calls.

    Args:
        K: (B, S, H, DK) bfloat16
        W: (B, S, H, DK) bfloat16
        Q: (B, S, H, DK) bfloat16
        G: (B, S, H) float32 (already chunk_local_cumsum'd)
        scale: float
        chunk_size: C
        use_g: whether to apply gate

    Returns:
        Kg: (B, BS, H, block_S, DK) float32
        Qg_T: (B, BS, H, DK, block_S) float32
        Wn_T: (B, BS, H, DK, block_S) float32
        Dmat: (B, BS, H, DK, DK) float32
    """
    B, S, H, DK = K.shape
    BS = S // chunk_size
    block_S = chunk_size

    G_f = G.view(B, BS, block_S, H)

    if use_g:
        G_last = G_f[:, :, -1, :]  # [B, BS, H]
        G_diff = G_last.unsqueeze(2) - G_f  # [B, BS, block_S, H]
        mask = G_diff <= 0
        gate_dv = torch.where(mask, torch.exp(G_diff), torch.zeros_like(G_diff))  # [B, BS, block_S, H]
        gate_dv_perm = gate_dv.permute(0, 1, 3, 2).unsqueeze(-1)  # [B, BS, H, block_S, 1]

        # Qg = Q * exp(G) * scale
        gate_q = torch.exp(G_f)  # [B, BS, block_S, H]
        gate_q_perm = gate_q.permute(0, 1, 3, 2).unsqueeze(-2)  # [B, BS, H, 1, block_S]

        # Dmat = diag(exp(G_last))
        g_last_exp = torch.exp(G_last)  # [B, BS, H]
        eye_DK = torch.eye(DK, dtype=torch.float32, device=K.device)
        Dmat = g_last_exp.unsqueeze(-1).unsqueeze(-1) * eye_DK  # [B, BS, H, DK, DK]
    else:
        gate_dv_perm = torch.ones(B, BS, H, block_S, 1, dtype=torch.float32, device=K.device)
        gate_q_perm = torch.ones(B, BS, H, 1, block_S, dtype=torch.float32, device=K.device)
        eye_DK = torch.eye(DK, dtype=torch.float32, device=K.device)
        Dmat = eye_DK.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, BS, H, DK, DK).contiguous()

    # Permute bf16 view first, then fuse upcast+multiply into one op.
    # Kg: K [B,S,H,DK] -> view [B,BS,block_S,H,DK] -> permute [B,BS,H,block_S,DK] -> * gate
    K_view = K.view(B, BS, block_S, H, DK).permute(0, 1, 3, 2, 4)  # [B,BS,H,block_S,DK] bf16
    Kg = K_view * gate_dv_perm  # bf16 × fp32 → fp32, contiguous

    # Qg_T: fuse upcast + multiply (precompute gate_q * scale as scalar)
    gate_q_scaled = gate_q_perm * scale  # [B,BS,H,1,block_S] fp32, small
    Q_view = Q.view(B, BS, block_S, H, DK).permute(0, 1, 3, 4, 2)  # [B,BS,H,DK,block_S] bf16
    Qg_T = Q_view * gate_q_scaled  # bf16 × fp32 → fp32, contiguous

    # Wn_T: negate in bf16, then float
    W_view = W.view(B, BS, block_S, H, DK).permute(0, 1, 3, 4, 2)  # [B,BS,H,DK,block_S] bf16
    Wn_T = (-W_view).float()

    return Kg, Qg_T, Wn_T, Dmat


# ============================================================================
# Host wrapper
# ============================================================================


def chunk_delta_bwd(
    Q,
    K,
    W,
    G,
    h0,
    dht,
    dO,
    dv,
    scale,
    chunk_size=1024,
    use_g=True,
    use_initial_state=True,
    use_final_state_gradient=True,
    block_DV=64,
):
    """Host wrapper for chunk_delta_bwd kernel.

    Precomputes all gates in fp32 on NPU, pre-transposes B operands for
    transpose_B GEMM, allocates transit buffers, and calls the kernel.

    Default chunk_size=1024 with T.Pipelined(num_stages=2) on BS loop.
    M-segment streaming (seg_M=128, nsegs_M=block_S//seg_M).
    Transposed dh L0C layout eliminates dh identity GEMM (1 fewer GEMM/chunk).
    dv transpose retained (T.copy cannot transpose [64,64]->[64,64]).
    No host overhead.

    Args:
        Q: (B, S, H, DK) bfloat16
        K: (B, S, H, DK) bfloat16
        W: (B, S, H, DK) bfloat16
        G: (B, S, H) float32 (already chunk_local_cumsum'd)
        h0: (B, H, DK, DV) bfloat16 (unused, for API compat)
        dht: (B, H, DK, DV) bfloat16
        dO: (B, S, H, DV) bfloat16
        dv: (B, S, H, DV) bfloat16
        scale: float
        chunk_size: int
        use_g: bool
        use_initial_state: bool
        use_final_state_gradient: bool
        block_DV: int

    Returns:
        dh: (B, BS, H, DV, DK) float32 (transposed layout)
        dh0: (B, H, DV, DK) float32 (transposed layout)
        dv2: (B, S, H, DV) float32
    """
    B, S, H, DK = Q.shape
    DV = dv.shape[-1]
    BS = S // chunk_size
    block_S = chunk_size
    device = Q.device

    # Gate precompute + permute + fp32 conversion via prep kernel
    # (falls back to torch prepare_bwd_gates when use_g=False).
    stream_ptr = torch.npu.current_stream().npu_stream
    if use_g:
        # Gc: permute G [B,S,H] → [B,BS,H,block_S]
        Gc = G.view(B, BS, block_S, H).permute(0, 1, 3, 2).contiguous()  # [B, BS, H, block_S] fp32
        eye_f32 = _eye(DK, torch.float32, device)

        # When use_final_state_gradient=False, pass zeros for dht so the
        # prep kernel produces dht_T = zeros (transpose of zeros = zeros).
        # Sync after zeros: torch.zeros_like on NPU is async, and without
        # sync the prep kernel may read stale non-zero values, causing dh
        # recursion to diverge to NaN.
        dht_input = dht if use_final_state_gradient else torch.zeros_like(dht)
        if not use_final_state_gradient:
            torch.npu.synchronize()

        prep = chunk_delta_bwd_prep(
            B=B,
            S=S,
            H=H,
            DK=DK,
            DV=DV,
            chunk_size=chunk_size,
            scale=scale,
            input_dtype="bfloat16",
            gate_dtype="float32",
            output_dtype="float32",
            prep_block_S=min(chunk_size, 64),  # sub-block at 64 to fit UB
        )
        fl_prep = _fast(
            ("prep_fp32_dOt_dhtT", B, S, H, DK, DV, chunk_size, scale, min(chunk_size, 64)),
            prep,
        )
        if fl_prep:
            Kg, Qg_T, Wn_T, Dmat, dv_in, dO_T, dht_T = fl_prep(
                K,
                Q,
                W,
                Gc,
                eye_f32,
                dv,
                dO,
                dht_input,
                stream_ptr=stream_ptr,
            )
        else:
            Kg, Qg_T, Wn_T, Dmat, dv_in, dO_T, dht_T = prep(K, Q, W, Gc, eye_f32, dv, dO, dht_input)
    else:
        # Torch fallback (use_g=False): gates are all ones
        Kg, Qg_T, Wn_T, Dmat = prepare_bwd_gates(K, W, Q, G, scale, chunk_size, use_g)
        dv_in = dv.permute(0, 2, 1, 3).float()  # [B, H, S, DV] fp32, contiguous
        # dO_T in chunk-major [B, BS, H, DV, block_S] layout to match
        # the scan kernel (which expects chunk-major dO_T from prep kernel).
        dO_T = dO.view(B, BS, block_S, H, DV).permute(0, 1, 3, 4, 2).contiguous().float()  # [B, BS, H, DV, block_S] fp32
        if use_final_state_gradient:
            dht_T = dht.permute(0, 1, 3, 2).float()  # [B, H, DV, DK] fp32
        else:
            dht_T = torch.zeros(B, H, DV, DK, dtype=torch.float32, device=device)

    # Identity matrices (constants)
    I_DK = _eye(DK, torch.float32, device)
    I_DV = _eye(block_DV, torch.float32, device)

    # Transit buffers (workspace, fp32)
    bv_num = (DV + block_DV - 1) // block_DV
    total_blocks = bv_num * B * H
    transit_dh_T = torch.zeros(total_blocks, block_DV, DK, dtype=torch.float32, device=device)
    transit_dv = torch.zeros(total_blocks, block_S, block_DV, dtype=torch.float32, device=device)
    transit_dv_T = torch.zeros(total_blocks, block_DV, block_S, dtype=torch.float32, device=device)

    # Compile kernel
    kernel = chunk_delta_bwd_kernel(
        B=B,
        S=S,
        H=H,
        DK=DK,
        DV=DV,
        chunk_size=chunk_size,
        scale=scale,
        use_g=use_g,
        use_initial_state=use_initial_state,
        use_final_state_gradient=use_final_state_gradient,
        input_dtype="float32",
        output_dtype="float32",
        accum_dtype="float32",
        state_dtype="float32",
        block_DV=block_DV,
    )

    # Call kernel via fast-launch path. Falls back to normal path if
    # construction fails.
    fl = _fast(
        ("scan_fp32_gemreorder", B, S, H, DK, DV, chunk_size, block_DV, use_initial_state, use_final_state_gradient),
        kernel,
    )
    if fl:
        dh, dh0, dv2 = fl(
            Kg,
            Qg_T,
            Wn_T,
            Dmat,
            dO_T,
            dv_in,
            dht_T,
            I_DK,
            I_DV,
            transit_dh_T,
            transit_dv,
            transit_dv_T,
            stream_ptr=stream_ptr,
        )
    else:
        dh, dh0, dv2 = kernel(
            Kg,
            Qg_T,
            Wn_T,
            Dmat,
            dO_T,
            dv_in,
            dht_T,
            I_DK,
            I_DV,
            transit_dh_T,
            transit_dv,
            transit_dv_T,
        )

    # When use_initial_state=False, the kernel skips the dh0 store.
    # _FastLaunch allocates dh0 with torch.empty (uninitialized), so
    # explicit zeroing ensures correct semantics (golden returns zeros).
    if not use_initial_state:
        dh0.zero_()

    return dh, dh0, dv2


# ============================================================================
# Smoke test (verify kernel runs + output shapes correct)
# ============================================================================


def smoke_test():
    """Quick smoke test: verify kernel runs and output shapes are correct."""
    torch.manual_seed(0)
    B, S, H, DK, DV, chunk_size = 1, 128, 1, 128, 128, 64
    block_DV = 64
    scale = DK**-0.5

    print("[smoke] preparing input (NPU) ...")
    Q = torch.randn(B, S, H, DK, dtype=torch.bfloat16, device="npu")
    K = F.normalize(
        torch.randn(B, S, H, DK, dtype=torch.bfloat16, device="npu"),
        dim=-1,
        p=2,
    )
    W = torch.randn(B, S, H, DK, dtype=torch.bfloat16, device="npu")
    G = F.logsigmoid(torch.randn(B, S, H, dtype=torch.float32, device="npu"))
    G = chunk_local_cumsum(G, chunk_size)
    h0 = torch.randn(B, H, DK, DV, dtype=torch.bfloat16, device="npu")
    dht = torch.randn(B, H, DK, DV, dtype=torch.bfloat16, device="npu")
    dO = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device="npu")
    dv = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device="npu")

    print("[smoke] compiling kernel (fp32 GEMM, transpose_B=True) ...")
    dh_out, dh0_out, dv2_out = chunk_delta_bwd(
        Q,
        K,
        W,
        G,
        h0,
        dht,
        dO,
        dv,
        scale,
        chunk_size,
        use_g=True,
        use_initial_state=True,
        use_final_state_gradient=True,
        block_DV=block_DV,
    )

    # Verify output shapes only (golden comparison in test file)
    assert dh_out.shape == (B, S // chunk_size, H, DV, DK), f"dh shape mismatch: {dh_out.shape}"
    assert dh0_out.shape == (B, H, DV, DK), f"dh0 shape mismatch: {dh0_out.shape}"
    assert dv2_out.shape == (B, S, H, DV), f"dv2 shape mismatch: {dv2_out.shape}"

    # Verify no NaN
    assert not torch.isnan(dh_out).any(), "dh NaN"
    assert not torch.isnan(dh0_out).any(), "dh0 NaN"
    assert not torch.isnan(dv2_out).any(), "dv2 NaN"

    print("Test Passed!")


if __name__ == "__main__":
    tilelang.disable_cache()
    smoke_test()
