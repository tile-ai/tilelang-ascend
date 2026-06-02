"""MXFP8 GEMM example using T.mma_mx (pto backend).

This example demonstrates the low-level MXFP GEMM API `T.mma_mx` that wraps
pto-isa `pto::TMATMUL_MX`. The example operates on L0A/L0B tiles directly;
users are expected to manage:
    * GM -> L1 data loading (T.copy from global tensors to L1 buffers).
    * L1 -> L0A/L0B extraction (via T.copy or the underlying TEXTRACT in
      the generated pto-isa code).
    * Scale tile binding (E8M0 block scales accompanying each L0 slice; one
      scale value per 32 elements along the K dimension).

Scale storage: stored as `uint8` in Python/DSL; the pto-isa C++ runtime
treats the underlying bits as `float8_e8m0_t`.

This file targets the PTO backend (`target="pto"`) on Ascend A5 (910C).
The hardware TMATMUL_MX instruction requires K to be a multiple of 64.
"""

import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

FP8_E4M3 = "e4m3_float8"
SCALE_DTYPE = "uint8"


# ---------------------------------------------------------------------------
# Developer-mode wrapper using `T.mma_mx`.
#
# Since `mma_mx` operates at L0 level, the kernel below allocates the
# required L0A / L0B / L0C / scale buffers at the Cube (matrix) scope and
# relies on `pass_configs` to let the compiler insert synchronization and
# memory planning automatically.
# ---------------------------------------------------------------------------


def mxfp8_gemm_kernel(M: int, N: int, K: int, block_M: int = 64, block_N: int = 64, block_K: int = 64):
    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }

    m_num = M // block_M
    n_num = N // block_N
    assert M % block_M == 0 and N % block_N == 0
    assert K % 64 == 0, "MXFP8 requires K % 64 == 0"

    scale_block = 32
    sa_cols = K // scale_block  # (M, K/32)
    sb_rows = K // scale_block  # (K/32, N)
    local_sa_cols = block_K // scale_block  # 2 for block_K=64
    local_sb_rows = block_K // scale_block

    @tilelang.jit(out_idx=[-1], target="pto", pass_configs=pass_configs)
    def build():
        @T.prim_func
        def main(
            A: T.Tensor((M, K), FP8_E4M3),
            B: T.Tensor((K, N), FP8_E4M3),
            scale_A: T.Tensor((M, sa_cols), SCALE_DTYPE),
            scale_B: T.Tensor((sb_rows, N), SCALE_DTYPE),
            C: T.Tensor((M, N), "float32"),
        ):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
                bm = cid // n_num
                bn = cid % n_num

                with T.Scope("C"):
                    # Allocate L0A / L0B / L0C + scale tiles for one L0 slice
                    A_l0a = T.alloc_L0A((block_M, block_K), FP8_E4M3)
                    B_l0b = T.alloc_L0B((block_K, block_N), FP8_E4M3)
                    C_l0c = T.alloc_L0C((block_M, block_N), "float32")

                    # Slice of scale_A: (block_M, block_K/32)
                    Sa_buf = T.alloc_shared((block_M, local_sa_cols), SCALE_DTYPE)
                    # Slice of scale_B: (block_K/32, block_N)
                    Sb_buf = T.alloc_shared((local_sb_rows, block_N), SCALE_DTYPE)

                    # Single K-tile pass: block_K must equal total K here for
                    # a single slice. For K > block_K the user should wrap the
                    # following in `for k_idx in T.serial(K // block_K)` and
                    # update the GM indices accordingly.
                    k_idx = 0
                    T.copy(A[bm * block_M, k_idx * block_K], A_l0a)
                    T.copy(B[k_idx * block_K, bn * block_N], B_l0b)
                    T.copy(scale_A[bm * block_M, k_idx * local_sa_cols], Sa_buf)
                    T.copy(scale_B[k_idx * local_sb_rows, bn * block_N], Sb_buf)

                    T.mma_mx(
                        A_l0a,
                        B_l0b,
                        C_l0c,
                        Sa_buf,
                        Sb_buf,
                        init=True,
                        scale_dtype=SCALE_DTYPE,
                    )

                    T.copy(C_l0c, C[bm * block_M, bn * block_N])

        return main

    return build()


# ---------------------------------------------------------------------------
# Host-side reference: decode the E4M3 bits and the E8M0 block scales, then
# run plain float matmul. This function is CPU-only and is used only to
# verify the kernel's numerical output.
# ---------------------------------------------------------------------------


def _decode_e4m3(x_byte: torch.Tensor) -> torch.Tensor:
    """Decode raw E4M3 byte (unsigned storage) to float32.

    We use a simple sign-magnitude decode that matches the MX convention:
      sign  = bit 7
      exp   = bits 3..6  (4-bit biased exponent, bias=7)
      mantissa = bits 0..2 (3-bit explicit leading 1 assumed)
    """
    x = x_byte.to(torch.int32) & 0xFF
    sign = ((x >> 7) & 0x1).float()
    exp = ((x >> 3) & 0xF).float()
    mant = (x & 0x7).float()
    val = torch.where(exp == 0, (1.0 + mant / 8.0) * (2.0 ** (1 - 7)), (1.0 + mant / 8.0) * (2.0 ** (exp - 7)))
    return torch.where(sign.bool(), -val, val)


def _decode_e8m0_scale(s_byte: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 block-scale byte to 2^(scale_int - 127)."""
    s = s_byte.to(torch.int32) & 0xFF
    return 2.0 ** (s.float() - 127.0)


def mxfp8_golden(A_bytes, B_bytes, scale_A, scale_B):
    """Host-side reference implementation (float matmul after dequant)."""
    M, K = A_bytes.shape
    _, N = B_bytes.shape
    scale_block = 32
    assert K % scale_block == 0
    k_blocks = K // scale_block

    A_fp = _decode_e4m3(A_bytes).to(torch.float32)
    B_fp = _decode_e4m3(B_bytes).to(torch.float32)

    # Apply block scales
    scale_A_decoded = _decode_e8m0_scale(scale_A)  # (M, k_blocks)
    scale_B_decoded = _decode_e8m0_scale(scale_B)  # (k_blocks, N)

    A_scaled = torch.zeros_like(A_fp)
    B_scaled = torch.zeros_like(B_fp)
    for b in range(k_blocks):
        k0, k1 = b * scale_block, (b + 1) * scale_block
        A_scaled[:, k0:k1] = A_fp[:, k0:k1] * scale_A_decoded[:, b : b + 1]
        B_scaled[k0:k1, :] = B_fp[k0:k1, :] * scale_B_decoded[b : b + 1, :]

    return A_scaled @ B_scaled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--block_M", type=int, default=64)
    parser.add_argument("--block_N", type=int, default=64)
    parser.add_argument("--block_K", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(0)

    # Build random E4M3 data + E8M0 block scales (stored as uint8 bytes)
    A_bytes = torch.randint(0, 256, (args.M, args.K), dtype=torch.uint8)
    B_bytes = torch.randint(0, 256, (args.K, args.N), dtype=torch.uint8)
    sa_cols = args.K // 32
    sb_rows = args.K // 32
    scale_A = torch.randint(80, 115, (args.M, sa_cols), dtype=torch.uint8)
    scale_B = torch.randint(80, 115, (sb_rows, args.N), dtype=torch.uint8)

    # Host-side golden reference
    ref = mxfp8_golden(A_bytes, B_bytes, scale_A, scale_B)
    print(f"Host reference shape: {tuple(ref.shape)}, dtype: {ref.dtype}")

    # Compile the kernel (requires NPU / PTO backend environment).
    # On a real NPU the kernel would be invoked as:
    #     kernel = mxfp8_gemm_kernel(args.M, args.N, args.K,
    #                                 args.block_M, args.block_N, args.block_K)
    #     C = kernel(A.npu(), B.npu(), scale_A.npu(), scale_B.npu())
    # The example intentionally stops at compile-time verification to keep it
    # safe to run on CPU-only hosts: the kernel function is built and its
    # source is printed if available.
    try:
        kernel = mxfp8_gemm_kernel(
            args.M,
            args.N,
            args.K,
            args.block_M,
            args.block_N,
            args.block_K,
        )
        print("Kernel compiled successfully.")
        try:
            src = kernel.get_kernel_source() if hasattr(kernel, "get_kernel_source") else None
            if src:
                print("----- Generated pto C++ source (excerpt) -----")
                print(src[:4000])
        except Exception:  # pragma: no cover - best-effort
            pass
    except RuntimeError as exc:
        print(f"Kernel compilation skipped (PTO backend unavailable): {exc}")


if __name__ == "__main__":
    main()
