"""
Dynamic Shape GEMM Kernel Definition

Supports arbitrary M, N, K dimensions for matrix multiplication
"""

import os
import torch
import tilelang
import tilelang.language as T


def compile_gemm_kernel():
    """Compile dynamic shape GEMM kernel"""
    print("=" * 60)
    print("Compiling dynamic shape GEMM kernel")
    print("=" * 60)
    
    os.environ['TILELANG_ASCEND_MODE'] = 'Expert'
    
    @tilelang.jit(target="npuir")
    def matmul(block_M=128, block_N=256, K_L1=16, dtype="float16", accum_dtype="float32"):
        M = T.symbolic("M")
        N = T.symbolic("N")
        K = T.symbolic("K")

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype)
        ):
            with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, _):
                with T.Scope("Cube"):
                    bx = cid // T.ceildiv(N, block_N) * block_M
                    by = cid % T.ceildiv(N, block_N) * block_N
                    A_BUF = T.alloc_L1([block_M, K_L1], dtype)
                    B_BUF = T.alloc_L1([K_L1, block_N], dtype)
                    C_BUF = T.alloc_L0C([block_M, block_N], accum_dtype)

                    remain_M = T.min(M - bx, block_M)
                    remain_N = T.min(N - by, block_N)

                    for i in T.serial(T.ceildiv(K, K_L1)):
                        remain_K = T.min(K - i * K_L1, K_L1)
                        T.load_nd2nz(A[bx, i * K_L1], A_BUF, [remain_M, remain_K])
                        T.load_nd2nz(B[i * K_L1, by], B_BUF, [remain_K, remain_N])

                        if i == 0:
                            T.gemm(A_BUF, B_BUF, C_BUF, initC=True, b_transpose=False,
                                size=[remain_M, remain_K, remain_N])
                        else:
                            T.gemm(A_BUF, B_BUF, C_BUF, initC=False, b_transpose=False,
                                size=[remain_M, remain_K, remain_N])

                        T.store_fixpipe(C_BUF, C[bx, by],
                            size=[remain_M, remain_N], enable_nz2nd=True)

        return main
    
    print("Compiling...")
    kernel = matmul()
    
    print(f"Kernel compilation completed")
    print(f"  - symbolic: {kernel.symbolic}")
    print(f"  - param_info: {kernel.param_info}")
    print(f"  - out_idx: {kernel.out_idx}")
    
    print("\nRunning tests...")
    test_cases = [
        (1024, 512, 2048),
        (512, 1024, 512),
    ]
    
    for M, N, K in test_cases:
        print(f"  Testing shape: M={M}, N={N}, K={K}")
        a = torch.randn(M, K, dtype=torch.float16, device="npu")
        b = torch.randn(K, N, dtype=torch.float16, device="npu")
        c = torch.randn(M, N, dtype=torch.float16, device="npu")
        
        kernel(a, b, c)
        
        ref = a @ b
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
        print(f"    ✓ Verification passed")
    
    print("✓ Dynamic GEMM kernel tests passed")
    return kernel
