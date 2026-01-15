import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation with Automatic UB to GM Copy")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    },
)
def vec_add_auto_copy(M, N, block_M, block_N, dtype="float"):
    """Vector addition kernel demonstrating automatic UB to GM copy using T.Parallel.

    This example shows how the system automatically handles copying data from
    UB (Unified Buffer) to GM (Global Memory) when writing directly to GM tensors
    within T.Parallel loops, without requiring explicit T.copy() calls.

    Key features:
    - Explicit T.copy() for input: GM -> UB (required)
    - Computation and direct GM write using T.Parallel
    - Automatic copy for output: UB -> GM (handled by system)
    """
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            # Allocate UB buffers for input data
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):
                # Step 1: Copy inputs from GM to UB (explicit copy required)
                # Data must be explicitly loaded from global memory to UB for computation
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                # Step 2: Compute and write directly to GM using T.Parallel
                # The parallel loop will be vectorized by ascend_lower_parallel_to_vector pass
                # When writing to GM tensor C directly, the system automatically handles
                # copying the computed result from UB to GM without explicit T.copy()
                for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                    C[bx * block_M + vid * block_M // VEC_NUM + i, by * block_N + j] = a_ub[i, j] + b_ub[i, j]

    return main


func = vec_add_auto_copy(M, N, 128, 128)

torch.manual_seed(0)

a = torch.randn(M, N).npu()
b = torch.randn(M, N).npu()

torch.npu.synchronize()
print("init successful!")

c = func(a, b)

ref_c = a + b

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
