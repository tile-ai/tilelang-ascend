"""Chunk Scaled Dot KKT operator (Ascend NPU, Developer mode + combineCV).

Computes A = (K @ K^T) * Beta with optional gating, chunked over sequence S.

Math:
  A[i,j] = (sum_k K[i,k] * K[j,k]) * Beta[i] * exp(G[i] - G[j])
           if G[i] <= G[j] and i > j, else 0.

Algorithm (per chunk, BS = chunk_size = 64, along S axis):
  1. Load K[BS, DK] from GM to L1 (block_DK=128; DK < block_DK zero-pads)
  2. GEMM: A_frag = K @ K^T (transpose_B=True, init=True)
  3. L0C -> UB via combineCV (automatic C->V transfer)
  4. Load Beta[BS] from GM to UB
  5. Beta row scaling: A_ub *= Beta[i]
  6. Lower-triangular mask: tril = (i > j)
  7. Gating (use_g=True): apply exp(G_diff) * select(mask); else select(tril)
  8. Cast fp32 -> bf16, store UB -> GM

Design highlights:
- Multi-chunk merge: each block processes 4 consecutive chunks via static
  unroll, amortizing per-block fixed overhead 4x.
- In-place exp: G_diff_2d doubles as the exp result (compare consumes raw
  G_diff before the in-place exp), saving a UB buffer.
- Tail-group guards: chunks 1/2/3 of the last partial group are runtime-
  guarded off; chunk 0 is always valid. Works for any S (divisible or not).
- K layout (B,H,S,DK) and A output (B,H,S,BS) for contiguous kernel access.
- Beta/G use (B,H,S) layout + Beta fp32 cast (host-side data prep, works
  around codegen strided 3D copy limitation).

Developer mode: 4 pass_configs ON (AUTO_CV_COMBINE, AUTO_CV_SYNC,
AUTO_SYNC, MEMORY_PLANNING). No T.Scope / set_flag / wait_flag / barrier_all.
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


@tilelang.jit(pass_configs=pass_configs)
def chunk_scaled_dot_kkt_fwd(
    B,
    S,
    H,
    DK,
    chunk_size=64,
    use_g=True,
    input_dtype="bfloat16",
    output_dtype="bfloat16",
    accum_dtype="float32",
    block_DK=128,
    chunks_per_block=4,
):
    """Chunk Scaled Dot KKT forward kernel (4-chunk static unroll).

    Args:
        B, S, H, DK: tensor dims (JIT compile-time params).
        chunk_size: chunk block size, fixed 64 by algorithm semantics.
            Must be 64 (asserted at entry): T.tile ops on (BS,BS) fp32
            tiles additionally require 256B alignment (BS%8==0).
        use_g: gating mode (JIT compile-time param; different kernels per value).
        input_dtype: dtype of K (bfloat16).
        output_dtype: dtype of A (bfloat16).
        accum_dtype: accumulation dtype (float32). Also Beta dtype.
        block_DK: K-dim block size (default 128). When DK < block_DK, the
            GM->L1 copy zero-pads cols [DK, block_DK).
        chunks_per_block: number of consecutive chunks processed per block.
            MUST be 4 (the per-chunk body is statically unrolled). Chunks
            1/2/3 of the last partial group are runtime-guarded off, so
            any S works.

    Returns:
        kernel prim_func. Caller pre-allocates A and calls kernel(K, Beta, G, A).

    Note:
        Beta is loaded as accum_dtype (float32) because the codegen does
        not support strided bf16 T.copy with pad value. Host-side casts
        Beta to float32 before .npu(). This is data preparation, not
        gate/tri_mask pre-computation.
    """
    block_S = chunk_size
    N = chunks_per_block
    assert N == 4, "static unroll handles exactly 4 chunks per block"
    # chunk_size constraint (fail-fast, review hardening): the algorithm
    # semantics fix the chunk size at 64, and T.tile.compare/broadcast/cast
    # operate on (BS,BS) fp32 tiles requiring 256B alignment (BS%8==0).
    # Non-conforming values previously only failed at AscendC lowering.
    assert chunk_size == 64, (
        f"chunk_size={chunk_size} must be 64: the algorithm semantics fix "
        f"the chunk size at 64; additionally T.tile.compare/broadcast/cast "
        f"operate on (BS,BS) fp32 tiles requiring 256B alignment (BS%8==0), "
        f"and non-conforming values previously only failed at AscendC "
        f"lowering."
    )
    num_chunks = S // block_S
    num_chunk_groups = (num_chunks + N - 1) // N
    total_blocks = num_chunk_groups * B * H
    # K layout (B,H,S,DK) for contiguous kernel read access. Host permutes
    # K to (B,H,S,DK) before .npu() (data prep, not gate/tri_mask
    # pre-computation).
    K_shape = (B, H, S, DK)
    # Beta/G use (B,H,S) layout to enable contiguous 1D T.copy in kernel.
    # (B,S,H) would make the S-dim slice strided; (B,H,S) makes it contiguous.
    Beta_shape = (B, H, S)
    G_shape = (B, H, S)
    # A output layout (B,H,S,block_S) for contiguous kernel write access.
    # After kernel, host permutes A back to (B,S,H,block_S).
    output_shape = (B, H, S, block_S)

    @T.prim_func
    def kernel(
        K: T.Tensor(K_shape, dtype=input_dtype),
        Beta: T.Tensor(Beta_shape, dtype=accum_dtype),
        G: T.Tensor(G_shape, dtype=accum_dtype),
        A: T.Tensor(output_shape, dtype=output_dtype),
    ):
        with T.Kernel(total_blocks, threads=1, is_npu=True) as (cid):
            bg = cid // (B * H)
            bbh = cid % (B * H)
            bb = bbh // H
            bh = bbh % H
            s_base = bg * block_S * N

            # --- L1 (Cube): K per chunk (4 separate 2D buffers) ---
            K_shared_0 = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            K_shared_1 = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            K_shared_2 = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            K_shared_3 = T.alloc_shared((block_S, block_DK), dtype=input_dtype)

            # --- L0C (Cube): GEMM accumulators (4 distinct; combineCV routes by name) ---
            A_frag_0 = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            A_frag_1 = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            A_frag_2 = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            A_frag_3 = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

            # --- UB (Vector): C->V destinations + shared cast/store ---
            A_ub_0 = T.alloc_shared((block_S, block_S), dtype=accum_dtype)
            A_ub_1 = T.alloc_shared((block_S, block_S), dtype=accum_dtype)
            A_ub_2 = T.alloc_shared((block_S, block_S), dtype=accum_dtype)
            A_ub_3 = T.alloc_shared((block_S, block_S), dtype=accum_dtype)
            A_out = T.alloc_shared((block_S, block_S), dtype=output_dtype)

            # Per-chunk Beta/G staging (unconditional; MEMORY_PLANNING prunes dead G under use_g=False)
            Beta_ub_0 = T.alloc_shared((block_S,), dtype=accum_dtype)
            Beta_ub_1 = T.alloc_shared((block_S,), dtype=accum_dtype)
            Beta_ub_2 = T.alloc_shared((block_S,), dtype=accum_dtype)
            Beta_ub_3 = T.alloc_shared((block_S,), dtype=accum_dtype)
            G_ub_0 = T.alloc_shared((block_S,), dtype=accum_dtype)
            G_ub_1 = T.alloc_shared((block_S,), dtype=accum_dtype)
            G_ub_2 = T.alloc_shared((block_S,), dtype=accum_dtype)
            G_ub_3 = T.alloc_shared((block_S,), dtype=accum_dtype)

            # Lower-triangle mask (computed once, shared by all chunks)
            idx_1d = T.alloc_shared((block_S,), dtype=accum_dtype)
            row_2d = T.alloc_shared((block_S, block_S), dtype=accum_dtype)  # reused as G_row
            col_2d = T.alloc_shared((block_S, block_S), dtype=accum_dtype)  # reused as G_col
            tril_mask = T.alloc_shared((block_S, block_S), dtype=accum_dtype)

            # Gating work buffers (shared across chunks; G_diff_2d doubles as exp result)
            Beta_2d = T.alloc_shared((block_S, block_S), dtype=accum_dtype)
            G_diff_2d = T.alloc_shared((block_S, block_S), dtype=accum_dtype)
            g_mask = T.alloc_shared((block_S, block_S), dtype=accum_dtype)

            # --- Phase 1: K loads (tail chunks guarded off) ---
            if bg * N + 0 < num_chunks:
                T.copy(K[bb, bh, s_base : s_base + block_S, :], K_shared_0)
            if bg * N + 1 < num_chunks:
                T.copy(
                    K[bb, bh, s_base + block_S : s_base + 2 * block_S, :],
                    K_shared_1,
                )
            if bg * N + 2 < num_chunks:
                T.copy(
                    K[bb, bh, s_base + 2 * block_S : s_base + 3 * block_S, :],
                    K_shared_2,
                )
            if bg * N + 3 < num_chunks:
                T.copy(
                    K[bb, bh, s_base + 3 * block_S : s_base + 4 * block_S, :],
                    K_shared_3,
                )

            # --- Phase 2: Beta/G loads ---
            if bg * N + 0 < num_chunks:
                T.copy(Beta[bb, bh, s_base : s_base + block_S], Beta_ub_0)
                if use_g:
                    T.copy(G[bb, bh, s_base : s_base + block_S], G_ub_0)
            if bg * N + 1 < num_chunks:
                T.copy(
                    Beta[bb, bh, s_base + block_S : s_base + 2 * block_S],
                    Beta_ub_1,
                )
                if use_g:
                    T.copy(
                        G[bb, bh, s_base + block_S : s_base + 2 * block_S],
                        G_ub_1,
                    )
            if bg * N + 2 < num_chunks:
                T.copy(
                    Beta[bb, bh, s_base + 2 * block_S : s_base + 3 * block_S],
                    Beta_ub_2,
                )
                if use_g:
                    T.copy(
                        G[bb, bh, s_base + 2 * block_S : s_base + 3 * block_S],
                        G_ub_2,
                    )
            if bg * N + 3 < num_chunks:
                T.copy(
                    Beta[bb, bh, s_base + 3 * block_S : s_base + 4 * block_S],
                    Beta_ub_3,
                )
                if use_g:
                    T.copy(
                        G[bb, bh, s_base + 3 * block_S : s_base + 4 * block_S],
                        G_ub_3,
                    )

            # --- Phase 3: tril_mask = (i > j) ---
            T.tile.arith_progression(idx_1d, 0, 1, block_S)
            T.tile.broadcast(row_2d, idx_1d, axis=1)
            T.tile.broadcast(col_2d, idx_1d, axis=0)
            T.tile.compare(tril_mask, row_2d, col_2d, "GT")

            # --- Phase 4: per-chunk [prep | gemm | C->V | gating | store] x4 ---
            # In-place exp order: sub -> compare(g_mask) -> exp -> bitwise_and

            # chunk 0 (always valid)
            T.tile.broadcast(Beta_2d, Beta_ub_0, axis=1)
            if use_g:
                T.tile.broadcast(row_2d, G_ub_0, axis=1)
                T.tile.broadcast(col_2d, G_ub_0, axis=0)
                T.tile.sub(G_diff_2d, row_2d, col_2d)
                T.tile.compare(g_mask, G_diff_2d, 0.0, "LE")
                T.tile.exp(G_diff_2d, G_diff_2d)
                T.tile.bitwise_and(g_mask, g_mask, tril_mask)
            T.gemm_v0(
                K_shared_0,
                K_shared_0,
                A_frag_0,
                transpose_B=True,
                init=True,
            )
            T.copy(A_frag_0, A_ub_0)
            T.tile.mul(A_ub_0, A_ub_0, Beta_2d)
            if use_g:
                T.tile.mul(A_ub_0, A_ub_0, G_diff_2d)
                T.tile.select(A_ub_0, g_mask, A_ub_0, 0.0, "VSEL_TENSOR_SCALAR_MODE")
            else:
                T.tile.select(A_ub_0, tril_mask, A_ub_0, 0.0, "VSEL_TENSOR_SCALAR_MODE")
            T.tile.cast(A_out, A_ub_0, "CAST_RINT", block_S * block_S)
            T.copy(A_out, A[bb, bh, s_base : s_base + block_S, :])

            # chunk 1
            if bg * N + 1 < num_chunks:
                T.tile.broadcast(Beta_2d, Beta_ub_1, axis=1)
                if use_g:
                    T.tile.broadcast(row_2d, G_ub_1, axis=1)
                    T.tile.broadcast(col_2d, G_ub_1, axis=0)
                    T.tile.sub(G_diff_2d, row_2d, col_2d)
                    T.tile.compare(g_mask, G_diff_2d, 0.0, "LE")
                    T.tile.exp(G_diff_2d, G_diff_2d)
                    T.tile.bitwise_and(g_mask, g_mask, tril_mask)
                T.gemm_v0(
                    K_shared_1,
                    K_shared_1,
                    A_frag_1,
                    transpose_B=True,
                    init=True,
                )
                T.copy(A_frag_1, A_ub_1)
                T.tile.mul(A_ub_1, A_ub_1, Beta_2d)
                if use_g:
                    T.tile.mul(A_ub_1, A_ub_1, G_diff_2d)
                    T.tile.select(A_ub_1, g_mask, A_ub_1, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                else:
                    T.tile.select(A_ub_1, tril_mask, A_ub_1, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.tile.cast(A_out, A_ub_1, "CAST_RINT", block_S * block_S)
                T.copy(A_out, A[bb, bh, s_base + block_S : s_base + 2 * block_S, :])

            # chunk 2
            if bg * N + 2 < num_chunks:
                T.tile.broadcast(Beta_2d, Beta_ub_2, axis=1)
                if use_g:
                    T.tile.broadcast(row_2d, G_ub_2, axis=1)
                    T.tile.broadcast(col_2d, G_ub_2, axis=0)
                    T.tile.sub(G_diff_2d, row_2d, col_2d)
                    T.tile.compare(g_mask, G_diff_2d, 0.0, "LE")
                    T.tile.exp(G_diff_2d, G_diff_2d)
                    T.tile.bitwise_and(g_mask, g_mask, tril_mask)
                T.gemm_v0(
                    K_shared_2,
                    K_shared_2,
                    A_frag_2,
                    transpose_B=True,
                    init=True,
                )
                T.copy(A_frag_2, A_ub_2)
                T.tile.mul(A_ub_2, A_ub_2, Beta_2d)
                if use_g:
                    T.tile.mul(A_ub_2, A_ub_2, G_diff_2d)
                    T.tile.select(A_ub_2, g_mask, A_ub_2, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                else:
                    T.tile.select(A_ub_2, tril_mask, A_ub_2, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.tile.cast(A_out, A_ub_2, "CAST_RINT", block_S * block_S)
                T.copy(A_out, A[bb, bh, s_base + 2 * block_S : s_base + 3 * block_S, :])

            # chunk 3
            if bg * N + 3 < num_chunks:
                T.tile.broadcast(Beta_2d, Beta_ub_3, axis=1)
                if use_g:
                    T.tile.broadcast(row_2d, G_ub_3, axis=1)
                    T.tile.broadcast(col_2d, G_ub_3, axis=0)
                    T.tile.sub(G_diff_2d, row_2d, col_2d)
                    T.tile.compare(g_mask, G_diff_2d, 0.0, "LE")
                    T.tile.exp(G_diff_2d, G_diff_2d)
                    T.tile.bitwise_and(g_mask, g_mask, tril_mask)
                T.gemm_v0(
                    K_shared_3,
                    K_shared_3,
                    A_frag_3,
                    transpose_B=True,
                    init=True,
                )
                T.copy(A_frag_3, A_ub_3)
                T.tile.mul(A_ub_3, A_ub_3, Beta_2d)
                if use_g:
                    T.tile.mul(A_ub_3, A_ub_3, G_diff_2d)
                    T.tile.select(A_ub_3, g_mask, A_ub_3, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                else:
                    T.tile.select(A_ub_3, tril_mask, A_ub_3, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.tile.cast(A_out, A_ub_3, "CAST_RINT", block_S * block_S)
                T.copy(A_out, A[bb, bh, s_base + 3 * block_S : s_base + 4 * block_S, :])

    return kernel


# =============================================================================
# Smoke test (CI entry; golden comparison is in test_chunk_scaled_dot_kkt.py)
# =============================================================================
if __name__ == "__main__":
    tilelang.disable_cache()
    torch.manual_seed(0)
    B, S, H, DK, chunk_size = 1, 32768, 32, 128, 64
    use_g = True
    # Data prep: permute K to (B,H,S,DK), Beta/G to (B,H,S), cast Beta to float32
    K_cpu = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    Beta_cpu = torch.randn(B, S, H, dtype=torch.bfloat16).permute(0, 2, 1).contiguous().to(torch.float32)
    G_cpu = torch.randn(B, S, H, dtype=torch.float32).permute(0, 2, 1).contiguous()
    K = K_cpu.permute(0, 2, 1, 3).contiguous().npu()
    Beta = Beta_cpu.npu()
    G = G_cpu.npu()

    print("[smoke] compiling kernel ...")
    kernel = chunk_scaled_dot_kkt_fwd(
        B=B,
        S=S,
        H=H,
        DK=DK,
        chunk_size=chunk_size,
        use_g=use_g,
        chunks_per_block=4,
    )
    print("[smoke] running kernel ...")
    A = torch.empty(B, H, S, chunk_size, dtype=torch.bfloat16, device="npu")
    kernel(K, Beta, G, A)
    torch.npu.synchronize()
    # Verify output shape (golden comparison is in test_chunk_scaled_dot_kkt.py)
    expected_shape = (B, H, S, chunk_size)
    assert A.shape == expected_shape, f"Shape mismatch: {A.shape} vs {expected_shape}"
    assert A.dtype == torch.bfloat16, f"Dtype mismatch: {A.dtype}"
    assert not torch.isnan(A).any(), "Output contains NaN"
    print(f"[smoke] output shape: {A.shape}, dtype: {A.dtype}")
    print("Test Passed!")
