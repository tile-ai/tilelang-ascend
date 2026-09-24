"""wy_fast operator (Ascend NPU, Developer mode + combineCV).

Computes W = A @ (K * Beta * exp(G)) and U = A @ (V * Beta) per chunk,
where each chunk is BS = chunk_size rows. Chunks are fully independent
(no cross-chunk dependency).

Math (per chunk c):
  W_c = A_c @ (K_c * Beta_c * exp(G_c))   # [BS, BS] @ [BS, DK] -> [BS, DK]
  U_c = A_c @ (V_c * Beta_c)              # [BS, BS] @ [BS, DV] -> [BS, DV]

Design highlights:
  - Merged GEMM: host concatenates V_Beta and K_Beta_G column-wise into KVG;
    the kernel computes both W and U with a single T.mma over N = DV + DK.
  - bh-major [B, H, S, D] layout for contiguous per-chunk DMA.
  - Fixed Core + grid-stride loop; multi-chunk per tile for L2 locality.
  - 8-segment manually-unrolled software pipeline with cross-tile prefetch.
  - Fallback path (gemm_v0 per-tile) for DK/DV > block size.
  - Host pre-computes V*Beta (bf16 direct mul) and K*Beta*exp(G) (fp32 promote).
  - 4 pass_configs ON (AUTO_CV_COMBINE, AUTO_CV_SYNC, AUTO_SYNC,
    MEMORY_PLANNING). No T.Scope / set_flag / wait_flag / barrier_all.
"""

import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def wy_fast_kernel(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    block_DK=128,
    block_DV=128,
    num_stages=2,
    input_dtype="bfloat16",
    output_dtype="bfloat16",
    accum_dtype="float32",
    core_num=20,
    chunks_per_tile=8,
):
    """WY-fast forward kernel (merged GEMM + cross-tile prefetch pipeline).

    Args:
        B, S, H, DK, DV: tensor dims (JIT compile-time params).
        chunk_size: chunk block size (BS), must be in [16, 128]. Algorithm
            requires full BS×BS local matrices per chunk, so the last partial
            chunk is dropped. BS >= 16 is the mma fractal minimum (A is a
            [BS, BS] operand with M=K=BS); BS <= 128 keeps the merged-path
            L0B/L0C buffers within capacity.
        block_DK, block_DV: K/V block sizes for the fallback path.
        num_stages: pipeline stages for the fallback (non-unrolled) path.
        input_dtype, output_dtype, accum_dtype: dtypes (bf16 in/out, fp32 accum).
        core_num: number of physical cube cores to launch.
        chunks_per_tile: chunks processed per tile (8 for L2 locality).

    Returns:
        kernel prim_func taking (KVG, A) and returning concatenated WU.
        Callers take zero-copy views: U = WU[..., :DV], W = WU[..., DV:].
    """
    BS = chunk_size
    # Floor division: tail chunk (S % BS != 0) is dropped. The host golden
    # zero-pads the tail; the test compares only the valid region.
    num_chunks = S // BS
    bh_total = B * H
    num_dv_tiles = (DV + block_DV - 1) // block_DV
    num_dk_tiles = (DK + block_DK - 1) // block_DK
    # Merged GEMM: host concatenates V_Beta and K_Beta_G into KVG [B,H,S,DV+DK];
    # kernel computes both U and W with one T.mma over N=DV+DK per chunk.
    D_total = DV + DK

    # Unrolled-path L0 budget: the 8-segment pipeline multi-buffers its L0
    # slots (3x A_l0a, 2x kvg_l0b / WU_l0c), sized for BS=64 (L0B/L0C
    # exactly at capacity). BS > 64 exceeds L0A/L0B/L0C capacity, so larger
    # chunk sizes take the single-buffered fallback path (fits up to 128).
    use_unrolled = num_dv_tiles == 1 and num_dk_tiles == 1 and BS <= 64 and chunks_per_tile in (4, 8) and num_chunks % chunks_per_tile == 0

    @T.macro(hygienic=False)
    def wy_pf(A, KVG, A_sh, KVG_sh, bb, bh, s):
        # GM -> L1 shared slot for chunk s (2 non-blocking DMA issues)
        T.copy(A[bb, bh, s * BS : (s + 1) * BS, :], A_sh)
        T.copy(KVG[bb, bh, s * BS : (s + 1) * BS, :], KVG_sh)

    @T.macro(hygienic=False)
    def wy_st(A_sh, KVG_sh, A_l0a, kvg_l0b):
        # L1 -> L0A/L0B staging of the prefetched chunk (overlaps in-flight mma)
        T.copy(A_sh, A_l0a)
        T.copy(KVG_sh, kvg_l0b)

    @T.macro(hygienic=False)
    def wy_cp(A_l0a, kvg_l0b, WU_l0c):
        # Merged GEMM for the current chunk from the given L0 slot
        T.mma(A_l0a, kvg_l0b, WU_l0c, init=True)

    @T.macro(hygienic=False)
    def wy_wb(WU, WU_l0c, bb, bh, s):
        # L0C -> GM writeback for chunk s
        T.copy(WU_l0c, WU[bb, bh, s * BS : (s + 1) * BS, :])

    # Persistent grid + multi-chunk per tile:
    # - Fixed Core launch (core_num): buffers allocated once per core,
    #   reused across waves.
    # - Multi-chunk per tile (chunks_per_tile=8): grid reduced 8x,
    #   each tile processes 8 consecutive chunks for L2 locality.
    n_chunk_groups = (num_chunks + chunks_per_tile - 1) // chunks_per_tile
    total_tiles = n_chunk_groups * bh_total
    waves = (total_tiles + core_num - 1) // core_num

    # GM layout: bh-major [B, H, S, D] so each [BS, D] tile is ONE contiguous
    # DMA instead of BS strided rows. Host permutes inputs in
    # prepare_wy_fast_inputs. Outputs are ALSO bh-major for contiguous
    # write-back per tile. Merged interface: single concatenated input KVG
    # and output WU, both [B, H, S, DV+DK].
    KVG_shape = (B, H, S, D_total)
    A_shape = (B, H, S, BS)
    WU_shape = (B, H, S, D_total)

    @T.prim_func
    def kernel(
        KVG: T.Tensor(KVG_shape, dtype=input_dtype),
        A: T.Tensor(A_shape, dtype=output_dtype),
        WU: T.Tensor(WU_shape, dtype=output_dtype),
    ):
        # Fixed Core: core_num physical cores, grid-stride loop over waves.
        # Buffers are allocated once per core and reused across waves.
        # All buffers are allocated at kernel scope; unused ones (depending
        # on the active code path) are eliminated by the compiler.
        with T.Kernel(core_num, threads=1, is_npu=True) as (cid):
            # ---- shared by both paths ----
            A_shared = T.alloc_shared((BS, BS), dtype=output_dtype)

            # ---- merged fast-path single-slot buffers ----
            KVG_shared = T.alloc_shared((BS, D_total), dtype=input_dtype)
            A_l0a = T.alloc_L0A((BS, BS), dtype=input_dtype)  # A [M,K]
            kvg_l0b = T.alloc_L0B((BS, D_total), dtype=input_dtype)  # [V|K] [K,N]
            WU_l0c = T.alloc_L0C((BS, D_total), dtype=accum_dtype)  # [U|W] [M,N]

            # ---- triple shared slots + dual L0B/L0C for the manually
            # unrolled software pipeline. ----
            A_sh_0 = T.alloc_shared((BS, BS), dtype=output_dtype)
            A_sh_1 = T.alloc_shared((BS, BS), dtype=output_dtype)
            A_sh_2 = T.alloc_shared((BS, BS), dtype=output_dtype)
            KVG_sh_0 = T.alloc_shared((BS, D_total), dtype=input_dtype)
            KVG_sh_1 = T.alloc_shared((BS, D_total), dtype=input_dtype)
            KVG_sh_2 = T.alloc_shared((BS, D_total), dtype=input_dtype)
            A_l0a_0 = T.alloc_L0A((BS, BS), dtype=input_dtype)
            A_l0a_1 = T.alloc_L0A((BS, BS), dtype=input_dtype)
            A_l0a_2 = T.alloc_L0A((BS, BS), dtype=input_dtype)
            kvg_l0b_0 = T.alloc_L0B((BS, D_total), dtype=input_dtype)
            kvg_l0b_1 = T.alloc_L0B((BS, D_total), dtype=input_dtype)
            WU_l0c_0 = T.alloc_L0C((BS, D_total), dtype=accum_dtype)
            WU_l0c_1 = T.alloc_L0C((BS, D_total), dtype=accum_dtype)

            # ---- cross-tile prefetch slots: next tile's chunks 0/1 are
            # DMA'd here so the MTE2 pipe stays back-to-back across tiles. ----
            A_sh_x0 = T.alloc_shared((BS, BS), dtype=output_dtype)
            A_sh_x1 = T.alloc_shared((BS, BS), dtype=output_dtype)
            KVG_sh_x0 = T.alloc_shared((BS, D_total), dtype=input_dtype)
            KVG_sh_x1 = T.alloc_shared((BS, D_total), dtype=input_dtype)

            # ---- fallback-path buffers for DK/DV > block. ----
            K_Beta_G_shared = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            V_Beta_shared = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            W_fragment = T.alloc_fragment((BS, block_DK), dtype=accum_dtype)
            U_fragment = T.alloc_fragment((BS, block_DV), dtype=accum_dtype)

            # Wave-0 prologue: prefetch the first tile's chunks 0/1 so the
            # wave loop can start with staging. The pid clamp keeps OOB
            # cores on a legal address (their loop body is skipped by the
            # guard); the chunk-index clamp keeps the DMA in-bounds when
            # the last chunk group holds a single chunk.
            pid_pro = T.min(cid, total_tiles - 1)
            bs_base_pro = (pid_pro // bh_total) * chunks_per_tile
            bbh_pro = pid_pro % bh_total
            bb_pro = bbh_pro // H
            bh_pro = bbh_pro % H
            s_pro_0 = T.min(bs_base_pro + 0, num_chunks - 1)
            s_pro_1 = T.min(bs_base_pro + 1, num_chunks - 1)
            wy_pf(A, KVG, A_sh_x0, KVG_sh_x0, bb_pro, bh_pro, s_pro_0)
            wy_pf(A, KVG, A_sh_x1, KVG_sh_x1, bb_pro, bh_pro, s_pro_1)

            # grid-stride loop: each core processes multiple tiles
            for wave in T.serial(waves):
                pid = wave * core_num + cid
                if pid < total_tiles:
                    # Decode 1D pid -> (bs_base, bb, bh)
                    bs_base = (pid // bh_total) * chunks_per_tile
                    bbh = pid % bh_total
                    bb = bbh // H
                    bh = bbh % H
                    # Next tile's pid (clamped to the last tile; the final
                    # wave's cores re-prefetch the last tile harmlessly)
                    pid_next = T.min(pid + core_num, total_tiles - 1)
                    bs_base_next = (pid_next // bh_total) * chunks_per_tile
                    bbh_next = pid_next % bh_total
                    bb_next = bbh_next // H
                    bh_next = bbh_next % H

                    if use_unrolled and chunks_per_tile == 4:
                        # 4-segment unroll (cpt=4). Cross-tile prefetch at
                        # seg2/3 (next tile chunks 0/1 -> x0/x1).
                        wy_st(A_sh_x0, KVG_sh_x0, A_l0a_0, kvg_l0b_0)
                        wy_st(A_sh_x1, KVG_sh_x1, A_l0a_1, kvg_l0b_1)
                        # seg0
                        wy_pf(A, KVG, A_sh_2, KVG_sh_2, bb, bh, bs_base + 2)
                        wy_cp(A_l0a_0, kvg_l0b_0, WU_l0c_0)
                        wy_wb(WU, WU_l0c_0, bb, bh, bs_base + 0)
                        # seg1
                        wy_pf(A, KVG, A_sh_0, KVG_sh_0, bb, bh, bs_base + 3)
                        wy_cp(A_l0a_1, kvg_l0b_1, WU_l0c_1)
                        wy_st(A_sh_2, KVG_sh_2, A_l0a_2, kvg_l0b_0)
                        wy_wb(WU, WU_l0c_1, bb, bh, bs_base + 1)
                        # seg2 (prefetch next tile)
                        wy_pf(A, KVG, A_sh_x0, KVG_sh_x0, bb_next, bh_next, bs_base_next + 0)
                        wy_cp(A_l0a_2, kvg_l0b_0, WU_l0c_0)
                        wy_st(A_sh_0, KVG_sh_0, A_l0a_0, kvg_l0b_1)
                        wy_wb(WU, WU_l0c_0, bb, bh, bs_base + 2)
                        # seg3 (prefetch next tile)
                        wy_pf(A, KVG, A_sh_x1, KVG_sh_x1, bb_next, bh_next, bs_base_next + 1)
                        wy_cp(A_l0a_0, kvg_l0b_1, WU_l0c_1)
                        wy_wb(WU, WU_l0c_1, bb, bh, bs_base + 3)
                    elif use_unrolled:
                        # 8-segment unroll (cpt=8). Each segment: prefetch(i+2),
                        # compute(i), stage(i+1), writeback(i). Segments 6/7
                        # prefetch the next tile's chunks 0/1.
                        wy_st(A_sh_x0, KVG_sh_x0, A_l0a_0, kvg_l0b_0)
                        wy_st(A_sh_x1, KVG_sh_x1, A_l0a_1, kvg_l0b_1)
                        # seg0
                        wy_pf(A, KVG, A_sh_2, KVG_sh_2, bb, bh, bs_base + 2)
                        wy_cp(A_l0a_0, kvg_l0b_0, WU_l0c_0)
                        wy_wb(WU, WU_l0c_0, bb, bh, bs_base + 0)
                        # seg1
                        wy_pf(A, KVG, A_sh_0, KVG_sh_0, bb, bh, bs_base + 3)
                        wy_cp(A_l0a_1, kvg_l0b_1, WU_l0c_1)
                        wy_st(A_sh_2, KVG_sh_2, A_l0a_2, kvg_l0b_0)
                        wy_wb(WU, WU_l0c_1, bb, bh, bs_base + 1)
                        # seg2
                        wy_pf(A, KVG, A_sh_1, KVG_sh_1, bb, bh, bs_base + 4)
                        wy_cp(A_l0a_2, kvg_l0b_0, WU_l0c_0)
                        wy_st(A_sh_0, KVG_sh_0, A_l0a_0, kvg_l0b_1)
                        wy_wb(WU, WU_l0c_0, bb, bh, bs_base + 2)
                        # seg3
                        wy_pf(A, KVG, A_sh_2, KVG_sh_2, bb, bh, bs_base + 5)
                        wy_cp(A_l0a_0, kvg_l0b_1, WU_l0c_1)
                        wy_st(A_sh_1, KVG_sh_1, A_l0a_1, kvg_l0b_0)
                        wy_wb(WU, WU_l0c_1, bb, bh, bs_base + 3)
                        # seg4
                        wy_pf(A, KVG, A_sh_0, KVG_sh_0, bb, bh, bs_base + 6)
                        wy_cp(A_l0a_1, kvg_l0b_0, WU_l0c_0)
                        wy_st(A_sh_2, KVG_sh_2, A_l0a_2, kvg_l0b_1)
                        wy_wb(WU, WU_l0c_0, bb, bh, bs_base + 4)
                        # seg5
                        wy_pf(A, KVG, A_sh_1, KVG_sh_1, bb, bh, bs_base + 7)
                        wy_cp(A_l0a_2, kvg_l0b_1, WU_l0c_1)
                        wy_st(A_sh_0, KVG_sh_0, A_l0a_0, kvg_l0b_0)
                        wy_wb(WU, WU_l0c_1, bb, bh, bs_base + 5)
                        # seg6 (prefetch next tile)
                        wy_pf(A, KVG, A_sh_x0, KVG_sh_x0, bb_next, bh_next, bs_base_next + 0)
                        wy_cp(A_l0a_0, kvg_l0b_0, WU_l0c_0)
                        wy_st(A_sh_1, KVG_sh_1, A_l0a_1, kvg_l0b_1)
                        wy_wb(WU, WU_l0c_0, bb, bh, bs_base + 6)
                        # seg7 (prefetch next tile)
                        wy_pf(A, KVG, A_sh_x1, KVG_sh_x1, bb_next, bh_next, bs_base_next + 1)
                        wy_cp(A_l0a_1, kvg_l0b_1, WU_l0c_1)
                        wy_wb(WU, WU_l0c_1, bb, bh, bs_base + 7)
                    else:
                        # Fallback: process chunks_per_tile consecutive chunks
                        for ci in T.Pipelined(chunks_per_tile, num_stages=num_stages):
                            bs = bs_base + ci
                            if bs < num_chunks:
                                T.copy(A[bb, bh, bs * BS : (bs + 1) * BS, :], A_shared)

                                if num_dv_tiles == 1 and num_dk_tiles == 1:
                                    # Merged fast path (single chunk per tile)
                                    T.copy(KVG[bb, bh, bs * BS : (bs + 1) * BS, :], KVG_shared)
                                    T.copy(A_shared, A_l0a)
                                    T.copy(KVG_shared, kvg_l0b)
                                    T.mma(A_l0a, kvg_l0b, WU_l0c, init=True)
                                    T.copy(WU_l0c, WU[bb, bh, bs * BS : (bs + 1) * BS, :])
                                else:
                                    # Fallback: DK or DV > block size,
                                    # iterate per-DV/DK tile with gemm_v0.
                                    for i_v in range(num_dv_tiles):
                                        kv_off = i_v * block_DV
                                        valid_dv = T.min(block_DV, DV - kv_off)
                                        T.copy(
                                            KVG[
                                                bb,
                                                bh,
                                                bs * BS : (bs + 1) * BS,
                                                kv_off : kv_off + valid_dv,
                                            ],
                                            V_Beta_shared,
                                            pad_value=0,
                                        )
                                        T.gemm_v0(A_shared, V_Beta_shared, U_fragment, init=True, kL0Size=64)
                                        T.copy(
                                            U_fragment[:, :valid_dv],
                                            WU[
                                                bb,
                                                bh,
                                                bs * BS : (bs + 1) * BS,
                                                kv_off : kv_off + valid_dv,
                                            ],
                                        )

                                    for i_k in range(num_dk_tiles):
                                        kk_off = i_k * block_DK
                                        valid_dk = T.min(block_DK, DK - kk_off)
                                        T.copy(
                                            KVG[
                                                bb,
                                                bh,
                                                bs * BS : (bs + 1) * BS,
                                                DV + kk_off : DV + kk_off + valid_dk,
                                            ],
                                            K_Beta_G_shared,
                                            pad_value=0,
                                        )
                                        T.gemm_v0(A_shared, K_Beta_G_shared, W_fragment, init=True, kL0Size=64)
                                        T.copy(
                                            W_fragment[:, :valid_dk],
                                            WU[
                                                bb,
                                                bh,
                                                bs * BS : (bs + 1) * BS,
                                                DV + kk_off : DV + kk_off + valid_dk,
                                            ],
                                        )

    return kernel


def to_bh_major(t):
    """Permute a [B, S, H, D] tensor to bh-major [B, H, S, D] contiguous layout.

    The kernel requires bh-major inputs so each [BS, D] tile is one contiguous
    GM DMA instead of BS strided rows.
    """
    return t.permute(0, 2, 1, 3).contiguous()


def _get_core_num():
    """Query the device's cube core count at runtime.

    Queries the device's cube core count so multi-core devices are not
    silently capped. Falls back to 20 (a common cube core count) when the
    property query fails, keeping behavior consistent on any device where
    the API is unavailable.
    """
    try:
        return torch.npu.get_device_properties(0).cube_core_num
    except Exception:
        return 20  # fallback when property query is unavailable


def get_wy_fast_kernel(B, S, H, DK, DV, chunk_size):
    """Construct the JIT kernel with the device's actual cube core count.

    All wrapper entry points (example main / test / bench) route through
    this. The kernel takes the concatenated bh-major KVG [B, H, S, DV+DK]
    plus A [B, H, S, BS] and returns the concatenated WU [B, H, S, DV+DK];
    callers take zero-copy views U = WU[..., :DV], W = WU[..., DV:].
    """
    # Also enforce the chunk_size contract here for callers that bypass
    # prepare_wy_fast_inputs.
    _validate_chunk_size(chunk_size)
    return wy_fast_kernel(
        B,
        S,
        H,
        DK,
        DV,
        chunk_size,
        core_num=_get_core_num(),
    )


def _validate_chunk_size(chunk_size):
    """Validate the chunk_size (BS) hardware contract: 16 <= BS <= 128.

    BS >= 16 is the mma fractal minimum (A is a [BS, BS] operand with
    M=K=BS); BS <= 128 keeps the merged-path L0B/L0C buffers within
    capacity. See the kernel docstring for details.
    """
    assert chunk_size >= 16, f"chunk_size must be >= 16 (mma fractal minimum, A is [BS, BS] with M=K=BS): chunk_size={chunk_size}"
    assert chunk_size <= 128, f"chunk_size must be <= 128 (merged-path L0B/L0C capacity for [BS, DV+DK]): chunk_size={chunk_size}"


def _validate_wy_fast_inputs(K, V, Beta, G, chunk_size):
    """Host-side input validation.

    Catches silently-wrong configurations at the wrapper boundary instead of
    producing non-deterministic corruption downstream:
    - fractal 16-alignment (DK/DV): unaligned D reads/writes OOB in the
      single-tile fast path.
    - chunk_size in [16, 128]: mma fractal minimum / L0B-L0C capacity
      (see _validate_chunk_size).
    - Beta bfloat16 / G float32 dtype contract.
    - S >= chunk_size: num_chunks=0 would silently output all zeros.
    - dtype/device/ndim/shape consistency + contiguity.
    """
    assert K.dim() == 4 and V.dim() == 4, f"K/V must be 4D [B,S,H,D]: K.dim={K.dim()}, V.dim={V.dim()}"
    B, S, H, DK = K.shape
    DV = V.shape[3]
    assert K.shape[:3] == V.shape[:3], f"K/V batch/seq/head dims must match: K={tuple(K.shape)}, V={tuple(V.shape)}"
    assert Beta.shape == (B, S, H) and G.shape == (B, S, H), (
        f"Beta/G must be [B,S,H]=({B},{S},{H}): Beta={tuple(Beta.shape)}, G={tuple(G.shape)}"
    )
    assert Beta.dtype == torch.bfloat16, f"Beta must be bfloat16 (bf16 multiply feeds the KVG concat): Beta.dtype={Beta.dtype}"
    assert G.dtype == torch.float32, f"G must be float32 (exp(G) precision contract): G.dtype={G.dtype}"
    assert DK % 16 == 0 and DV % 16 == 0, f"fractal 16 alignment required: DK={DK}, DV={DV}"
    _validate_chunk_size(chunk_size)
    assert chunk_size <= S, f"S={S} must be >= chunk_size={chunk_size} (num_chunks=0 would silently output zeros)"
    assert K.dtype == V.dtype == torch.bfloat16, f"K/V must be bfloat16 (kernel input_dtype): K={K.dtype}, V={V.dtype}"
    assert K.device.type == "npu" and V.device.type == "npu", f"K/V must be on npu: K={K.device}, V={V.device}"
    for t, name in ((K, "K"), (V, "V")):
        assert t.is_contiguous(), f"{name} must be contiguous"


def prepare_wy_fast_inputs(K, V, Beta, G, chunk_size, merged=True):
    """Pre-compute V*Beta and K*Beta*exp(G) on host (bf16), in bh-major layout.

    Outputs are permuted to [B, H, S, D] contiguous (bh-major) so the
    kernel's per-chunk tile loads are contiguous DMAs instead of strided
    rows. Full input validation is performed — see _validate_wy_fast_inputs.

    merged=True (default) returns the column-concatenated KVG
    [B, H, S, DV+DK] ([0:DV] = V_Beta, [DV:] = K_Beta_G) required by the
    merged-GEMM kernel interface.

    Arithmetic choices:
    - V side: bf16 direct multiply (V * Beta) — bit-exact vs the fp32 chain
      on NPU (bf16 mul is correctly rounded: the 16-bit exact product fits
      fp32's 24-bit mantissa, so fp32-mul-then-round == bf16 mul).
    - K side: fp32-promote single multiply — the tiny [B,S,H] factor
      (Beta_f32 * exp(G)_f32) is computed on the small tensor, then ONE
      big fp32 multiply K * factor, then the bf16 cast.
    - Copy-out: contiguous bh-major outputs + torch.cat.
    """
    _validate_wy_fast_inputs(K, V, Beta, G, chunk_size)
    # bh-major muls on permuted views: contiguous [B,H,S,D] outputs
    V_bh = V.permute(0, 2, 1, 3) * Beta.permute(0, 2, 1).unsqueeze(-1)
    factor_bh = (Beta.float() * torch.exp(G)).unsqueeze(-1).permute(0, 2, 1, 3)
    K_bh = (K.permute(0, 2, 1, 3) * factor_bh).to(K.dtype)
    if merged:
        return torch.cat([V_bh, K_bh], dim=-1)  # [B, H, S, DV+DK]
    return K_bh, V_bh


def golden_wy_fast(K, V, Beta, G, A, chunk_size):
    """PyTorch reference: W = A @ (K * Beta * exp(G)), U = A @ (V * Beta)

    Uses bf16 matmul (same precision path as kernel: bf16 input + fp32 accum
    via NPU Cube GEMM). Pre-computes V*Beta and K*Beta*exp(G) in fp32 then
    casts to bf16 (matching the kernel's host-side pre-computation). The bf16
    matmul ensures the golden uses the same NPU GEMM hardware as the kernel.

    For non-aligned S (S % chunk_size != 0), the WY-fast algorithm requires
    full BS×BS local matrices per chunk, so the tail rows are dropped
    (matching the kernel's floor-division num_chunks). The output tail is
    zero-padded so the test can compare the full tensor or slice to the
    valid region.
    """
    B, S, H, DK = K.shape
    _, _, _, DV = V.shape
    BS = chunk_size
    nc = S // BS  # floor: only full chunks
    valid_S = nc * BS

    # Slice to the valid (full-chunk) region for reshape compatibility
    K_v = K[:, :valid_S, :, :]
    V_v = V[:, :valid_S, :, :]
    Beta_v = Beta[:, :valid_S, :]
    G_v = G[:, :valid_S, :]
    A_v = A[:, :valid_S, :, :]

    # Golden uses the SAME arithmetic as prepare_wy_fast_inputs
    # (V bf16 direct mul; K fp32-promote single mul). Golden and prepare
    # must use IDENTICAL formulas so the kernel-vs-golden comparison
    # stays bit-exact.
    V_Beta = V_v * Beta_v.unsqueeze(-1)
    factor = (Beta_v.float() * torch.exp(G_v)).unsqueeze(-1)
    K_Beta_G = (K_v * factor).to(K.dtype)

    K_Beta_G_c = K_Beta_G.reshape(B, nc, BS, H, DK)
    V_Beta_c = V_Beta.reshape(B, nc, BS, H, DV)
    A_c = A_v.reshape(B, nc, BS, H, BS)

    A_perm = A_c.permute(0, 1, 3, 2, 4)
    KBG_perm = K_Beta_G_c.permute(0, 1, 3, 2, 4)
    VB_perm = V_Beta_c.permute(0, 1, 3, 2, 4)

    W_perm = A_perm @ KBG_perm
    U_perm = A_perm @ VB_perm

    W_valid = W_perm.permute(0, 1, 3, 2, 4).reshape(B, valid_S, H, DK)
    U_valid = U_perm.permute(0, 1, 3, 2, 4).reshape(B, valid_S, H, DV)

    # Zero-pad to full S so output shape matches kernel's output tensor
    W = torch.zeros(B, S, H, DK, dtype=K.dtype, device=K.device)
    U = torch.zeros(B, S, H, DV, dtype=V.dtype, device=V.device)
    W[:, :valid_S, :, :] = W_valid
    U[:, :valid_S, :, :] = U_valid

    return W, U


def get_precision(dtype):
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    int_types = {"int8", "int16", "int32", "int64", "uint8"}
    if dtype in int_types:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    special = ~torch.isfinite(g)
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        return False, 0.0, float("inf")
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    matched_ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error


if __name__ == "__main__":
    torch.manual_seed(1)
    B, S, H, DK, DV, chunk_size = 1, 256, 4, 128, 128, 64
    K = torch.randn(B, S, H, DK, dtype=torch.bfloat16, device="npu")
    V = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device="npu")
    Beta = torch.randn(B, S, H, dtype=torch.bfloat16, device="npu")
    G = torch.randn(B, S, H, dtype=torch.float32, device="npu")
    A = torch.randn(B, S, H, chunk_size, dtype=torch.bfloat16, device="npu")

    KVG = prepare_wy_fast_inputs(K, V, Beta, G, chunk_size, merged=True)
    A_bh = to_bh_major(A)  # [B, H, S, BS] bh-major kernel input

    kernel = get_wy_fast_kernel(B, S, H, DK, DV, chunk_size)
    WU = kernel(KVG, A_bh)
    torch.npu.synchronize()

    # Kernel output is the concatenated bh-major WU [B, H, S, DV+DK]
    # ([0:DV]=U, [DV:]=W) — zero-copy views, permuted back to [B, S, H, D]
    # for comparison.
    W = WU[..., DV:].permute(0, 2, 1, 3)
    U = WU[..., :DV].permute(0, 2, 1, 3)

    W_ref, U_ref = golden_wy_fast(K, V, Beta, G, A, chunk_size)

    # The kernel writes only the valid (full-chunk) region; tail rows are
    # left uninitialized. Compare valid_S only (same contract as the tests).
    valid_S = (S // chunk_size) * chunk_size
    W_cmp, U_cmp = W[:, :valid_S], U[:, :valid_S]
    W_ref_cmp, U_ref_cmp = W_ref[:, :valid_S], U_ref[:, :valid_S]

    w_passed, w_ratio, w_max = check_precision(W_cmp, W_ref_cmp, "bfloat16")
    u_passed, u_ratio, u_max = check_precision(U_cmp, U_ref_cmp, "bfloat16")

    print(f"W: [PRECISION_{'PASS' if w_passed else 'FAIL'}] ratio={w_ratio:.4f} max_abs={w_max:.3e}")
    print(f"U: [PRECISION_{'PASS' if u_passed else 'FAIL'}] ratio={u_ratio:.4f} max_abs={u_max:.3e}")

    assert w_passed and u_passed, "Smoke test failed"
    print("Test Passed!")
