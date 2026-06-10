import argparse
import sys

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

# MXFP8 GEMM (OCP Microscaling with e8m0 block scale + TMATMUL_MX) requires
# the A5 Cube core. A2/A3 devices (C220) don't provide TMATMUL_MX at all.
from tilelang.utils.target import determine_platform
if determine_platform() != "A5":
    print(f"[SKIP] MXFP8 GEMM requires A5 platform; detected: {determine_platform()}")
    sys.exit(0)

parser = argparse.ArgumentParser(description="NPU MXFP8 GEMM Kernel (A5 PTO)")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension (must be multiple of 64)")
parser.add_argument(
    "--fp8",
    type=str,
    default="e5m2",
    choices=["e4m3", "e5m2"],
    help="MXFP8 dtype variant: e4m3 or e5m2",
)
args = parser.parse_args()

M = args.m
N = args.n
K = args.k
MX_SCALE_BLOCK = 32

assert M % 128 == 0 and N % 128 == 0 and K % 64 == 0, (
    "M, N must be multiples of 128; K must be a multiple of 64"
)

fp8_dtype = T.e4m3_float8 if args.fp8 == "e4m3" else T.e5m2_float8
torch_fp8_dtype = torch.float8_e4m3fn if args.fp8 == "e4m3" else torch.float8_e5m2
input_dtype_str = "e4m3_float8" if args.fp8 == "e4m3" else "e5m2_float8"

torch_fp8_max = torch.finfo(torch_fp8_dtype).max


def quantize_mxfp8_host(x_fp16: torch.Tensor, block: int = MX_SCALE_BLOCK):
    """
    Quantize a float16 tensor to MXFP8 with e8m0 per-block exponents.
    x_fp16 : (rows, K) float16
    returns (data_uint8_e8m0_shape_same_as_x, scales_uint8)
    """
    rows, cols = x_fp16.shape
    assert cols % block == 0, f"K must be a multiple of {block}"
    n_blocks = cols // block

    x_blocks = x_fp16.reshape(rows, n_blocks, block)
    block_max = x_blocks.abs().amax(dim=-1)
    block_max = block_max.clamp(min=torch.finfo(torch.float16).tiny)

    exp = torch.floor(torch.log2(block_max.float())).to(torch.int32)
    exp = exp.clamp(min=-127, max=127)
    e8m0_scale = (exp + 127).to(torch.uint8)

    scale_factor = (2.0 ** (-exp.float())).unsqueeze(-1)
    normalized = (x_blocks.float() * scale_factor).to(torch.float16)
    normalized = normalized.clamp(-torch_fp8_max, torch_fp8_max)
    data_fp8 = normalized.to(torch_fp8_dtype)

    return data_fp8.reshape(rows, cols), e8m0_scale


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(
    out_idx=[-1],
    target="pto",
    pass_configs=pass_configs,
)
def mxfp8_matmul(M, N, K, block_M, block_N, K_L1):
    m_num = M // block_M
    n_num = N // block_N
    K_SLABS_PER_CHUNK = K_L1 // MX_SCALE_BLOCK

    @T.prim_func
    def main(
            A: T.Tensor((M, K), input_dtype_str),
            B: T.Tensor((K, N), input_dtype_str),
            sA: T.Tensor((M, K // MX_SCALE_BLOCK), "uint8"),
            sB: T.Tensor((K // MX_SCALE_BLOCK, N), "uint8"),
            C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), input_dtype_str)
            B_L1 = T.alloc_L1((K_L1, block_N), input_dtype_str)
            sA_L1 = T.alloc_L1(
                (block_M, K_SLABS_PER_CHUNK), "uint8"
            )
            sB_L1 = T.alloc_L1(
                (K_SLABS_PER_CHUNK, block_N // MX_SCALE_BLOCK), "uint8"
            )
            C_L0 = T.alloc_L0C((block_M, block_N), "float32")

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.copy(
                        sA[bx * block_M, k * K_SLABS_PER_CHUNK],
                        sA_L1,
                    )
                    T.copy(
                        sB[k * K_SLABS_PER_CHUNK, by * block_N // MX_SCALE_BLOCK],
                        sB_L1,
                    )

                    T.gemm_mx(A_L1, B_L1, C_L0, sA_L1, sB_L1, init=(k == 0))

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


func = mxfp8_matmul(M, N, K, 128, 128, 128)

torch.manual_seed(0)

a_fp16 = torch.randn(M, K, dtype=torch.float16).npu()
b_fp16 = torch.randn(K, N, dtype=torch.float16).npu()

a_fp8, a_scale = quantize_mxfp8_host(a_fp16)
b_fp8, b_scale = quantize_mxfp8_host(b_fp16)
a_fp8 = a_fp8.contiguous().npu()
b_fp8 = b_fp8.contiguous().npu()
a_scale = a_scale.contiguous().npu()
b_scale = b_scale.contiguous().npu()

print(f"Running MXFP8 GEMM ({args.fp8}): M={M}, N={N}, K={K}")
print("init successful!")

c_fp32 = func(a_fp8, b_fp8, a_scale, b_scale)

a_dequant = a_fp8.float()
b_dequant = b_fp8.float()
ref_c = a_dequant @ b_dequant

torch.testing.assert_close(c_fp32, ref_c, rtol=1e-1, atol=1e-1)
print("Kernel Output Match!")
