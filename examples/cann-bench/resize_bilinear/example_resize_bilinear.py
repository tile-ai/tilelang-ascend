"""ResizeBilinear (bilinear interpolation) for Ascend NPU using TileLang-Ascend.

Bilinear resize of a 4D tensor (N, C, H, W) via the separable two-pass
formulation, dispatched across the Cube (GEMM) and Vector cores:

    Y[n, c, i, j] = wh0[i] * (ww0[j] * X[n, c, h0[i], w0[j]]
                             + ww1[j] * X[n, c, h0[i], w1[j]])
                  + wh1[i] * (ww0[j] * X[n, c, h1[i], w0[j]]
                             + ww1[j] * X[n, c, h1[i], w1[j]])

  width pass  : Y_h = X_flat @ W_w_T   -- Cube core (T.gemm_v0, fp32 accum)
  height pass : Y    = W_h-rows of Y_h -- Vector core (contiguous row-tile
                                          load + vectorized mul/axpy)

Key techniques demonstrated:

1. Heterogeneous dispatch (Cube + Vector):
   - The width pass is reformulated as a dense GEMM against a sparse
     (2 non-zeros per column) weight matrix. The Cube core's Mmad
     throughput beats a Vector gather by 10-50x.
   - The height pass stays on the Vector core: one contiguous 2D
     GM->UB tile load covering all source rows of the output block,
     then per-row vectorized `T.tile.mul` / `T.tile.axpy` weighted sums.

2. L0 ping-pong via `kL0Size`: with K_L1=128 and kL0Size=64,
   kL0split = 2 enables L0-level double buffering inside T.gemm_v0
   (measured +15.9% over the default kL0Size=128).

3. Hardware Gather via `T.tile.gather`: element-wise T.Parallel with a
   runtime UB index does NOT vectorize (measured 14x slower). The width
   fallback kernel uses uint32 byte offsets with the hardware Gather
   instruction instead.

4. K-tail handling: framework `compute_valid_extent` ZERO-fILLS out-of-
   bounds A_L1 reads, so W_in not divisible by K_L1 needs no host-side
   padding (the B matrix is padded with zero rows on the host).

Run:
    python example_resize_bilinear.py                       # defaults
    python example_resize_bilinear.py --n 2 --c 8 --h 512 --w 512 \\
        --oh 256 --ow 256 --dtype float16 --align-corners
"""

import argparse

import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CAST_MODE_HIGH2LOW = "CAST_RINT"  # round to nearest, fp32 -> fp16/bf16


# ============================ Width pass (Cube) ============================
# C = A @ B, A = X_flat [M, K_a] (fp16/bf16), B = W_w_T [K_b, N] (fp16/bf16,
# 2 non-zero entries per column), C = Y_h [M, N] fp32. The fp32 accumulator
# keeps full precision; the only rounding is the host-side fp32->fp16 cast
# of the weights (bit-exact when the weights are dyadic, e.g. exact-2x).
@tilelang.jit(out_idx=[-1])
def gemm_width_kernel(M, K_a, K_b, N, block_M, block_N, K_L1, dtype, c_dtype="float32", accum_dtype="float", kL0Size=64):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    has_m_tail = (M % block_M) != 0
    has_n_tail = (N % block_N) != 0

    @T.prim_func
    def main(
        A: T.Tensor((M, K_a), dtype),
        B: T.Tensor((K_b, N), dtype),
        C: T.Tensor((M, N), c_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K_b, K_L1)
                # T.Pipelined (L1 software pipeline, 2 stages) + kL0Size=64
                # (L0 ping-pong: kL0split = ceil(K_L1 / kL0Size) = 2).
                for k in T.Pipelined(loop_k, num_stages=2):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.gemm_v0(
                        A_L1,
                        B_L1,
                        C_L0,
                        init=(k == 0),
                        kL0Size=kL0Size,
                    )

                if has_m_tail or has_n_tail:
                    valid_m = T.min(block_M, M - bx * block_M)
                    valid_n = T.min(block_N, N - by * block_N)
                    T.copy(
                        C_L0[:valid_m, :valid_n],
                        C[
                            bx * block_M : bx * block_M + valid_m,
                            by * block_N : by * block_N + valid_n,
                        ],
                    )
                else:
                    T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


# =========================== Height pass (Vector) ===========================
# Y[n, c, i, j] = wh0[i] * Y_h[n, c, h0[i], j] + wh1[i] * Y_h[n, c, h1[i], j]
# Each block covers [block_H, block_W] of the output. Instead of 2*block_H
# per-row indirect GM->UB DMAs (high fixed launch overhead), ONE contiguous
# 2D GM->UB T.copy loads all source rows the block needs ([range_h, block_W]);
# the two source rows of each output row are then gathered from the UB tile
# by runtime row index (on-chip UB reads, much cheaper than indirect GM DMA).
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def height_kernel(N, C, H_in, H_out, W_out, block_H, block_W, range_h, dtype):
    nc_num = N * C
    h_num = T.ceildiv(H_out, block_H)
    w_num = T.ceildiv(W_out, block_W)
    total_blocks = nc_num * h_num * w_num

    @T.prim_func
    def main(
        Y_h: T.Tensor((N, C, H_in, W_out), "float32"),
        block_row_starts: T.Tensor((h_num,), "int32"),
        h0_off: T.Tensor((H_out,), "int32"),
        h1_off: T.Tensor((H_out,), "int32"),
        wh0: T.Tensor((H_out,), "float32"),
        wh1: T.Tensor((H_out,), "float32"),
        Y: T.Tensor((N, C, H_out, W_out), dtype),
    ):
        with T.Kernel(total_blocks, is_npu=True) as (cid, _):
            w_tile = cid % w_num
            h_tile = (cid // w_num) % h_num
            nc = cid // (w_num * h_num)
            n = nc // C
            c = nc % C

            h_start = h_tile * block_H
            w_start = w_tile * block_W

            row_start_ub = T.alloc_ub((1,), "int32")
            ih0_off_ub = T.alloc_ub((block_H,), "int32")
            ih1_off_ub = T.alloc_ub((block_H,), "int32")
            wh0_ub = T.alloc_ub((block_H,), "float32")
            wh1_ub = T.alloc_ub((block_H,), "float32")
            row_tile_ub = T.alloc_ub((range_h, block_W), "float32")
            y_row_ub = T.alloc_ub((block_W,), "float32")
            y_out_ub = T.alloc_ub((block_H, block_W), dtype)

            with T.Scope("V"):
                T.copy(block_row_starts[h_tile : h_tile + 1], row_start_ub)
                T.copy(h0_off[h_start : h_start + block_H], ih0_off_ub)
                T.copy(h1_off[h_start : h_start + block_H], ih1_off_ub)
                T.copy(wh0[h_start : h_start + block_H], wh0_ub)
                T.copy(wh1[h_start : h_start + block_H], wh1_ub)
                T.barrier_all()

                # ONE contiguous 2D GM->UB tile load. row_start is the
                # block's minimum source row (floor indices are monotonic);
                # range_h is the compile-time max row span of this shape.
                # OOB rows of tail blocks are zero-filled by the framework.
                T.copy(
                    Y_h[
                        n,
                        c,
                        row_start_ub[0] : row_start_ub[0] + range_h,
                        w_start : w_start + block_W,
                    ],
                    row_tile_ub,
                )
                T.barrier_all()

                # Per output row: gather the two source rows from the UB
                # tile (runtime row index, on-chip) + vectorized weighted
                # sum. AUTO_SYNC inserts the fine-grained PipeBarrier<PIPE_V>
                # at the V->V and V->MTE3 transitions (manual
                # T.barrier_all() would compile to the coarser PIPE_ALL).
                for i in T.serial(block_H):
                    h0o = ih0_off_ub[i]
                    h1o = ih1_off_ub[i]
                    # y = wh0[i] * row[h0o, :]   (mul overwrites y_row)
                    T.tile.mul(y_row_ub, row_tile_ub[h0o, :], wh0_ub[i])
                    # y += wh1[i] * row[h1o, :]
                    T.tile.axpy(y_row_ub, row_tile_ub[h1o, :], wh1_ub[i])
                    # cast fused into the row loop (fp32 -> dtype)
                    T.tile.cast(y_out_ub[i, :], y_row_ub, CAST_MODE_HIGH2LOW, block_W)

                T.barrier_all()
                T.copy(
                    y_out_ub,
                    Y[n, c, h_start : h_start + block_H, w_start : w_start + block_W],
                )

    return main


# ========================= Host-side weight building =========================
def compute_indices_weights(in_size, out_size, align_corners):
    """Interpolation indices/weights on CPU (no NPU aclnn dispatch).

    Follows PyTorch's area_pixel_compute_source_index semantics:
    negative source coords clamp to 0; positives may exceed in_size-1
    (floor is clamped instead). Returns CPU int64/fp32 tensors.
    """
    if align_corners:
        s = (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
        coords = torch.arange(out_size, dtype=torch.float64) * s
    else:
        s = float(in_size) / float(out_size)
        coords = (torch.arange(out_size, dtype=torch.float64) + 0.5) * s - 0.5
    coords = coords.clamp_min(0)
    floor_idx = coords.floor().long().clamp_max(in_size - 1)
    ceil_idx = floor_idx + (floor_idx < (in_size - 1)).long()
    delta = (coords - floor_idx.double()).clamp(0, 1)
    wf, wc = 1.0 - delta, delta
    # Upper-edge clamp replication (ceil == floor, e.g. upsampling's last
    # rows/cols): force weights (1, 0). For finite values this is the same
    # value (even more exact — no 1-ulp rounding); for Inf inputs it
    # reproduces the lerp-form 1*Inf + 0*Inf = Inf + NaN = NaN behavior of
    # the reference upsample kernel.
    replicated = ceil_idx == floor_idx
    wf = torch.where(replicated, torch.ones_like(delta), wf)
    wc = torch.where(replicated, torch.zeros_like(delta), wc)
    return floor_idx, ceil_idx, wf, wc


def build_weight_matrix(W_in, W_out, align_corners, dtype_str, K_L1):
    """Dense [K_padded, W_out] width-interpolation weight matrix (CPU).

    Column j has exactly 2 non-zero entries: W[fw[j], j] = wf[j] and
    W[cw[j], j] += wc[j]. K is padded up to a multiple of K_L1 with zero
    rows so the GEMM K-loop runs full iterations; the framework zero-fills
    the A-side OOB reads (compute_valid_extent), so X needs no padding.
    """
    fw, cw, wf, wc = compute_indices_weights(W_in, W_out, align_corners)
    K_padded = ((W_in + K_L1 - 1) // K_L1) * K_L1
    W = torch.zeros(K_padded, W_out, dtype=torch.float32)
    col_idx = torch.arange(W_out, dtype=torch.long)
    W[fw, col_idx] = wf.float()
    W[cw, col_idx] += wc.float()  # += handles the fw == cw overlap
    return W.to(getattr(torch, dtype_str)).npu()  # CPU cast + H2D (allowed)


def build_height_block_info(H_in, H_out, block_H, align_corners):
    """Per-block row spans + per-row offsets/weights for the height pass.

    block_row_starts[t] = floor_idx[t * block_H] (the block's min source
    row — floor indices are monotonic); h0_off/h1_off are row offsets
    within the loaded [range_h, block_W] tile; range_h is the compile-time
    max source-row span of any block.
    """
    fh, ch, hf, hc = compute_indices_weights(H_in, H_out, align_corners)
    fh_i32, ch_i32 = fh.to(torch.int32), ch.to(torch.int32)
    h_num = (H_out + block_H - 1) // block_H
    h_starts = torch.arange(h_num, dtype=torch.long) * block_H
    block_row_starts = fh_i32[h_starts].contiguous()
    block_id = torch.arange(H_out, dtype=torch.long) // block_H
    block_id = block_id.clamp(max=h_num - 1)
    rs_per_row = block_row_starts[block_id]
    h0_off = (fh_i32 - rs_per_row).to(torch.int32).contiguous()
    h1_off = (ch_i32 - rs_per_row).to(torch.int32).contiguous()
    h_ends = (h_starts + block_H - 1).clamp(max=H_out - 1)
    range_h = int((ch[h_ends] - fh[h_starts]).max().item()) + 2  # +2 covers the ceil tap
    range_h = min(range_h, H_in)
    return (
        block_row_starts.to(torch.int32).npu(),
        h0_off.npu(),
        h1_off.npu(),
        hf.float().npu(),
        hc.float().npu(),
        range_h,
    )


# ================================ Driver ================================
def resize_bilinear_gemm_height(x, H_out, W_out, align_corners=False, k_l1=128):
    """Run the two-pass resize: Cube GEMM width + Vector height.

    Args:
        x: input tensor (N, C, H_in, W_in) on NPU, fp16/bf16
        H_out, W_out: output spatial size
        align_corners: bilinear alignment mode
        k_l1: K tile of the width GEMM (64 or 128)

    Returns:
        Output tensor (N, C, H_out, W_out), same dtype as x.
    """
    N, C, H_in, W_in = x.shape
    dtype_str = str(x.dtype).replace("torch.", "")

    # GEMM-path applicability (weight-precision analysis):
    #   - fp16 weights: ~2^-11 relative rounding — safe for typical ranges.
    #   - bf16 weights: ~2^-8 rounding — precise enough EXCEPT for
    #     align_corners upsampling, where near-zero outputs (cancellation of
    #     opposite-sign neighbors) amplify the weight error to mare > 1.0.
    #     Route bf16 + ac + upsampling to a Vector kernel instead (see the
    #     full cann-bench submission's dispatch).
    if dtype_str == "bfloat16" and align_corners and H_out > H_in:
        raise ValueError(
            "bf16 + align_corners + upsampling: the 2^-8 bf16 weight "
            "rounding is amplified at near-zero outputs (mare > 1.0). "
            "Use fp16, align_corners=False, or the Vector fallback path."
        )
    M1 = N * C * H_in
    block_M, block_N = 128, 256
    # The 2-tile-wide GEMM grid (m_num >= 2) needs enough rows to fill the
    # Mmad pipeline; tiny inputs should use the Vector gather formulation
    # instead (see the fused kernel in the cann-bench submission).
    if 2 * block_M > M1:
        raise ValueError(f"N*C*H_in = {M1} rows is too small for the GEMM width pass (needs >= {2 * block_M}); increase N/C/H_in")

    # ---- Width pass (Cube GEMM): Y_h = X_flat @ W_w_T ----
    K_a, N1 = W_in, W_out
    W_w_T = build_weight_matrix(W_in, W_out, align_corners, dtype_str, k_l1)
    X_flat = x.view(M1, K_a)
    gemm = gemm_width_kernel(
        M1,
        K_a,
        W_w_T.shape[0],
        N1,
        block_M,
        block_N,
        k_l1,
        dtype_str,
        c_dtype="float32",
        kL0Size=64,
    )
    Y_h = gemm(X_flat, W_w_T).view(N, C, H_in, W_out)  # fp32

    # ---- Height pass (Vector): Y_h -> Y ----
    # block_W must be a multiple of 16 (UB 32-byte alignment); a
    # non-divisible W_out leaves a tail block that the framework's
    # compute_valid_extent handles (OOB reads zero-filled, OOB writes
    # clamped).
    block_H = 16
    block_W = min(256, ((W_out + 15) // 16) * 16)
    brs, h0_off, h1_off, hf, hc, range_h = build_height_block_info(H_in, H_out, block_H, align_corners)
    height = height_kernel(N, C, H_in, H_out, W_out, block_H, block_W, range_h, dtype_str)
    return height(Y_h, brs, h0_off, h1_off, hf, hc)


def main():
    parser = argparse.ArgumentParser(description="ResizeBilinear NPU Kernel")
    parser.add_argument("--n", type=int, default=2, help="Batch N")
    parser.add_argument("--c", type=int, default=8, help="Channels C")
    parser.add_argument("--h", type=int, default=512, help="Input height H_in")
    parser.add_argument("--w", type=int, default=512, help="Input width W_in")
    parser.add_argument("--oh", type=int, default=256, help="Output height H_out")
    parser.add_argument("--ow", type=int, default=256, help="Output width W_out")
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "bfloat16"], help="Input/output dtype (GEMM path requires fp16/bf16)"
    )
    parser.add_argument("--align-corners", action="store_true", help="align_corners mode")
    parser.add_argument("--k-l1", type=int, default=128, help="K tile of the width GEMM (64 or 128)")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch_dtype = getattr(torch, args.dtype)

    x = (torch.rand(args.n, args.c, args.h, args.w) * 2 - 1).to(torch_dtype).npu()
    print(f"init successful! x={tuple(x.shape)} {args.dtype}, {args.h}x{args.w} -> {args.oh}x{args.ow}, align_corners={args.align_corners}")

    y = resize_bilinear_gemm_height(x, args.oh, args.ow, align_corners=args.align_corners, k_l1=args.k_l1)
    torch.npu.synchronize()
    print("kernel executed!")

    # ---- Verify against the fp64 reference ----
    ref = torch.nn.functional.interpolate(
        x.double().cpu(),
        size=[args.oh, args.ow],
        mode="bilinear",
        align_corners=args.align_corners,
    ).to(torch_dtype)
    rtol = 1e-2 if args.dtype != "float32" else 1e-4
    atol = 1e-3 if args.dtype != "float32" else 1e-4
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol, atol=atol)
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
