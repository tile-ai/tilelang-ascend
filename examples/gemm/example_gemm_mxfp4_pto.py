import argparse
import sys

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

# MXFP4 GEMM (OCP Microscaling with e8m0 block scale + TMATMUL_MX on float4x2
# data) requires the A5 Cube core. A2/A3 devices (C220) don't provide
# TMATMUL_MX at all.
from tilelang.utils.target import determine_platform

if determine_platform() != "A5":
    print(f"[SKIP] MXFP4 GEMM requires A5 platform; detected: {determine_platform()}")
    sys.exit(0)

parser = argparse.ArgumentParser(description="NPU MXFP4 GEMM Kernel (A5 PTO)")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument(
    "--k",
    type=int,
    default=1024,
    help="Matrix K dimension (must be multiple of 64 and even for FP4 packing)",
)
parser.add_argument(
    "--fp4",
    type=str,
    default="e2m1x2",
    choices=["e2m1x2", "e1m2x2"],
    help="MXFP4 format: e2m1x2 or e1m2x2 (twin/packed)",
)
args = parser.parse_args()

M = args.m
N = args.n
K = args.k
MX_SCALE_BLOCK = 32

assert M % 128 == 0 and N % 128 == 0, "M and N must be multiples of 128"
assert K % 64 == 0 and K % 2 == 0, "K must be multiple of 64 and even (FP4 packing)"


def pack_fp4_x2(values: torch.Tensor) -> torch.Tensor:
    """
    Pack 4-bit float values (already quantized to [-6,6] range) into uint8
    bytes with two FP4 values per byte (low nibble first, then high nibble).
    The FP4 bit pattern for e2m1 format is: 1 sign + 2 exp + 1 mantissa.
    """
    flat = values.contiguous().view(-1)
    assert flat.numel() % 2 == 0, "FP4 packing requires even number of elements"
    lo = (flat[0::2].to(torch.int8) & 0x0F).to(torch.uint8)
    hi = (flat[1::2].to(torch.int8) & 0x0F).to(torch.uint8) << 4
    return (lo | hi).reshape(values.shape[0], values.shape[1] // 2).to(torch.uint8)


def quantize_mxfp4_host(x_fp16: torch.Tensor, block: int = MX_SCALE_BLOCK, fmt: str = "e2m1x2"):
    """
    Quantize a float16 tensor to MXFP4 (twin/packed) with e8m0 per-block exponents.
    x_fp16 : (rows, K) float16 tensor
    fmt    : "e2m1x2" (E2M1, max absolute value 6.0)
             "e1m2x2" (E1M2, max absolute value 28.0, rarely used)
    Returns (data_uint8_packed, scales_uint8_e8m0)
      data_uint8_packed : (rows, K // 2) bytes of packed MXFP4
      scales_uint8_e8m0 : (rows, K // 32) bytes of E8M0 block exponents
    """
    rows, cols = x_fp16.shape
    assert cols % block == 0, f"K must be a multiple of {block}"
    n_blocks = cols // block

    max_abs = 6.0 if fmt == "e2m1x2" else 28.0

    x_blocks = x_fp16.reshape(rows, n_blocks, block)
    block_max = x_blocks.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float16).tiny)

    exp = torch.floor(torch.log2(block_max.float())).to(torch.int32)
    exp = exp.clamp(min=-127, max=127)
    e8m0_scale = (exp + 127).to(torch.uint8)

    scale_factor = (2.0 ** (-exp.float())).unsqueeze(-1)
    normalized = (x_blocks.float() * scale_factor).to(torch.float16)
    normalized = normalized.clamp(-max_abs, max_abs)

    # Round-to-nearest FP4 (simplified): cast to int8 via bit ops. For a
    # production-quality encoder the exact FP4 bit pattern must match the
    # hardware's rounding mode; this helper is illustrative.
    fp4_rounded = torch.round(normalized).to(torch.int8)
    packed = pack_fp4_x2(fp4_rounded.reshape(rows, cols))

    # Pad data to the declared (M, K_logical) shape for L1 buffer alignment.
    # Only the first K_logical/2 bytes of each row carry real data — the
    # tail K_logical/2 bytes are zero-fill. This matches the shape
    # convention used by T.gemm_mx for the MXFP4 format: we declare the L1
    # buffer as (M, K_logical) uint8 even though only half is populated, so
    # that template-tile addressing is consistent across MXFP8 and MXFP4.
    padded = torch.zeros(rows, cols, dtype=torch.uint8, device=x_fp16.device)
    padded[:, : packed.shape[1]] = packed

    return padded, e8m0_scale


def fake_mxfp4_matmul_ref(a_q, b_q, a_scale, b_scale, block: int = MX_SCALE_BLOCK):
    """
    Reference: dequantize a/b back to float using e8m0 scales, then matmul.
    Used for accuracy checking.
    """
    rows, K_phys = a_q.shape
    cols = b_q.shape[1]
    K_logical = K_phys
    assert K_phys % 128 == 0, "expect K_logical multiple of 128 here"
    # unpack
    a_unpacked = torch.zeros(rows, K_logical, dtype=torch.float32, device=a_q.device)
    for i in range(K_logical // 2):
        byte_val = a_q[:, i].to(torch.int8)
        a_unpacked[:, 2 * i] = byte_val & 0x0F
        a_unpacked[:, 2 * i + 1] = (byte_val >> 4) & 0x0F

    a_scale_expanded = a_scale.repeat_interleave(block, dim=1)
    a_dequant = a_unpacked * (2.0 ** (a_scale_expanded.float() - 127))

    b_unpacked = torch.zeros(K_logical, cols, dtype=torch.float32, device=b_q.device)
    for i in range(K_logical // 2):
        byte_val = b_q[i, :].to(torch.int8)
        b_unpacked[2 * i, :] = byte_val & 0x0F
        b_unpacked[2 * i + 1, :] = (byte_val >> 4) & 0x0F

    b_scale_expanded = b_scale.repeat_interleave(block, dim=0)
    b_dequant = b_unpacked * (2.0 ** (b_scale_expanded.float() - 127))

    return a_dequant @ b_dequant


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], target="pto", pass_configs=pass_configs)
def mxfp4_matmul(M, N, K, block_M, block_N, K_L1, fmt: str):
    m_num = M // block_M
    n_num = N // block_N
    K_SLABS_PER_CHUNK = K_L1 // MX_SCALE_BLOCK

    @T.prim_func
    def main(
        # A, B declared as uint8 with shape (M, K) — K is *logical* K.
        # Each row actually stores K/2 packed MXFP4 bytes followed by
        # K/2 zero-padding (see quantize_mxfp4_host).
        A: T.Tensor((M, K), "uint8"),
        B: T.Tensor((K, N), "uint8"),
        sA: T.Tensor((M, K // MX_SCALE_BLOCK), "uint8"),
        sB: T.Tensor((K // MX_SCALE_BLOCK, N), "uint8"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            # Data L1 buffers use uint8 and full logical K shape.
            A_L1 = T.alloc_L1((block_M, K_L1), "uint8")
            B_L1 = T.alloc_L1((K_L1, block_N), "uint8")
            sA_L1 = T.alloc_L1((block_M, K_SLABS_PER_CHUNK), "uint8")
            sB_L1 = T.alloc_L1((K_SLABS_PER_CHUNK, block_N // MX_SCALE_BLOCK), "uint8")
            C_L0 = T.alloc_L0C((block_M, block_N), "float32")

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.copy(sA[bx * block_M, k * K_SLABS_PER_CHUNK], sA_L1)
                    T.copy(
                        sB[k * K_SLABS_PER_CHUNK, by * block_N // MX_SCALE_BLOCK],
                        sB_L1,
                    )

                    T.gemm_mx(
                        A_L1,
                        B_L1,
                        C_L0,
                        sA_L1,
                        sB_L1,
                        init=(k == 0),
                        format=fmt,
                    )

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


func = mxfp4_matmul(M, N, K, 128, 128, 128, args.fp4)

torch.manual_seed(0)

a_fp16 = torch.randn(M, K, dtype=torch.float16).npu()
b_fp16 = torch.randn(K, N, dtype=torch.float16).npu()

a_fp4_packed, a_scale = quantize_mxfp4_host(a_fp16, block=MX_SCALE_BLOCK, fmt=args.fp4)
b_fp4_packed, b_scale = quantize_mxfp4_host(b_fp16, block=MX_SCALE_BLOCK, fmt=args.fp4)

a_for_kernel = a_fp4_packed.contiguous().npu()
b_for_kernel = b_fp4_packed.contiguous().npu()
a_scale = a_scale.contiguous().npu()
b_scale = b_scale.contiguous().npu()

print(f"Running MXFP4 GEMM ({args.fp4}): M={M}, N={N}, K={K}")
print("init successful!")

c_fp32 = func(a_for_kernel, b_for_kernel, a_scale, b_scale)

ref_c = fake_mxfp4_matmul_ref(a_for_kernel, b_for_kernel, a_scale, b_scale)

torch.testing.assert_close(c_fp32, ref_c, rtol=1e-1, atol=1e-1)
print("Kernel Output Match!")
