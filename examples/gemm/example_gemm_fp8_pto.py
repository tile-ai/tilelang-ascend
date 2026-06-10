import argparse
import sys

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

# FP8 GEMM (TMATMUL accepting float8_e4m3 / float8_e5m2) requires the A5
# Cube core. A2/A3 devices (C220) don't provide the FP8 TMATMUL path, so we
# skip gracefully on non-A5 hardware rather than failing at CCE compile time.
from tilelang.utils.target import determine_platform
if determine_platform() != "A5":
    print(f"[SKIP] FP8 GEMM requires A5 platform; detected: {determine_platform()}")
    sys.exit(0)

parser = argparse.ArgumentParser(description="NPU FP8 GEMM Kernel (A5 PTO)")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
parser.add_argument(
    "--fp8",
    type=str,
    default="e4m3",
    choices=["e4m3", "e5m2"],
    help="FP8 dtype variant: e4m3 or e5m2",
)
args = parser.parse_args()

M = args.m
N = args.n
K = args.k
fp8_dtype = T.e4m3_float8 if args.fp8 == "e4m3" else T.e5m2_float8
torch_fp8_dtype = (
    torch.float8_e4m3fn if args.fp8 == "e4m3" else torch.float8_e5m2
)
input_dtype_str = "e4m3_float8" if args.fp8 == "e4m3" else "e5m2_float8"

assert M % 128 == 0 and N % 128 == 0 and K % 128 == 0, (
    "M, N, K must be multiples of 128 for FP8 GEMM tiling"
)

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], target="pto", pass_configs=pass_configs)
def fp8_matmul(M, N, K, block_M, block_N, K_L1):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, K), input_dtype_str),
            B: T.Tensor((K, N), input_dtype_str),
            C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), input_dtype_str)
            B_L1 = T.alloc_L1((K_L1, block_N), input_dtype_str)

            C_L0 = T.alloc_L0C((block_M, block_N), "float32")

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


func = fp8_matmul(M, N, K, 128, 128, 128)

torch.manual_seed(0)

a_fp16 = torch.randn(M, K, dtype=torch.float16).npu()
b_fp16 = torch.randn(K, N, dtype=torch.float16).npu()

a_fp8 = a_fp16.to(torch_fp8_dtype)
b_fp8 = b_fp16.to(torch_fp8_dtype)

print(f"Running FP8 GEMM ({args.fp8}): M={M}, N={N}, K={K}")
print("init successful!")

c_fp32 = func(a_fp8, b_fp8)

ref_c = a_fp8.float() @ b_fp8.float()

torch.testing.assert_close(c_fp32, ref_c, rtol=5e-2, atol=5e-2)
print("Kernel Output Match!")
