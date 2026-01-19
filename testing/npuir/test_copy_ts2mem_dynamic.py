import os
import torch
import tilelang
import tilelang.language as T

# Set Environment
os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
dtype = "float32"

# ==========================================
# Kernel 1: 1D Dynamic Shape + Rank Change
# ==========================================
@tilelang.jit(out_idx=[-1], target="npuir")
def kernel_1d(M, N, K_split):
    """
    Simulate GEMM Epilogue: 
    1. Frag(Reg) -> Cast -> Frag(Reg) (Generates Tensor SSA Value)
    2. Frag(2D) -> Global(3D) (Triggers Rank Change 2D->3D)
    3. Slice with `real_m` (Triggers Dynamic Shape)
    """
    assert N % K_split == 0
    N_dim1 = N // K_split
    block_M = 64
    
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype), 
        B: T.Tensor((M, N_dim1, K_split), dtype)
    ):
        with T.Kernel(T.ceildiv(M, block_M), is_npu=True) as (bid, _):

            frag_raw = T.alloc_fragment((block_M, N), dtype)
            frag_cast = T.alloc_fragment((block_M, N), dtype) 
            
            offset_m = bid * block_M
            real_m = T.min(block_M, M - offset_m)
            
            T.copy(A[offset_m:offset_m+real_m, 0:N], frag_raw[0:real_m, 0:N])
            
            T.npuir_cast(frag_raw[0:block_M, 0:N], frag_cast[0:block_M, 0:N], "rint")

            T.copy(
                frag_cast[0:real_m, 0:N], 
                B[offset_m:offset_m+real_m, 0:N_dim1, 0:K_split]
            )
    return main

def test_1d():
    print("\n" + "="*20 + " Running Test 1D: GEMM Epilogue (Dynamic M) " + "="*20)
    
    # M=100 (64+36), N=128, K=16
    M, N, K = 100, 128, 16

    # Execution
    kernel = kernel_1d(M, N, K)
    
    torch.npu.set_device(0)
    a_cpu = torch.randn(M, N)
    b_cpu = torch.zeros(M, N // K, K)
    
    a_npu = a_cpu.npu()
    b_npu = b_cpu.npu()
    
    kernel(a_npu, b_npu)
    
    # Verification
    # Simulate Cast (rint) + Reshape
    a_ref = torch.round(a_npu)
    b_flat = b_npu.reshape(M, N)
    
    try:
        torch.testing.assert_close(a_ref, b_flat, rtol=1e-3, atol=1e-3)
        print("\033[92m[PASS] Test 1D (Dynamic M + Reshape) Verified!\033[0m")
    except Exception as e:
        print("\033[91m[FAIL] Test 1D Result Mismatch!\033[0m")
        print(e)

# ==========================================
# Kernel 2: 2D Double Dynamic Dimensions
# ==========================================
@tilelang.jit(out_idx=[-1], target="npuir")
def kernel_2d(M, N):
    """
    Scenario: Input [M, N] -> Output [M, N, 1]
    1. Grid X handles M -> generates dynamic variable `real_m`
    2. Grid Y handles N -> generates dynamic variable `real_n`
    3. Copy operation reshapes (real_m, real_n) to (real_m, real_n, 1)
    """
    block_M = 64
    block_N = 64 
    
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype), 
        B: T.Tensor((M, N, 1), dtype)
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (bid, _):
            idx_m = bid // T.ceildiv(N, block_N)
            idx_n = bid % T.ceildiv(N, block_N)
            
            offset_m = idx_m * block_M
            offset_n = idx_n * block_N
            
            real_m = T.min(block_M, M - offset_m)
            real_n = T.min(block_N, N - offset_n)
            
            frag_raw = T.alloc_fragment((block_M, block_N), dtype)
            frag_cast = T.alloc_fragment((block_M, block_N), dtype)
            
            T.copy(A[offset_m:offset_m+real_m, offset_n:offset_n+real_n], 
                   frag_raw[0:real_m, 0:real_n])
            
            T.npuir_cast(frag_raw[0:block_M, 0:block_N], frag_cast[0:block_M, 0:block_N], "rint")

            T.copy(
                frag_cast[0:real_m, 0:real_n],
                B[offset_m:offset_m+real_m, offset_n:offset_n+real_n, 0:1]
            )
    return main

def test_2d():
    print("\n" + "="*20 + " Running Test 2D: Double Dynamic Dims " + "="*20)
    
    # M=100 (64+36), N=100 (64+36)
    M, N = 100, 100

    # Execution
    kernel = kernel_2d(M, N)
    
    torch.npu.set_device(0)
    a_cpu = torch.randn(M, N)
    b_cpu = torch.zeros(M, N, 1)
    
    a_npu = a_cpu.npu()
    b_npu = b_cpu.npu()
    
    kernel(a_npu, b_npu)
    
    # Verification
    a_ref = torch.round(a_npu).unsqueeze(-1) 
    
    try:
        torch.testing.assert_close(a_ref, b_npu, rtol=1e-3, atol=1e-3)
        print("\033[92m[PASS] Test 2D (Double Dynamic Dims) Verified!\033[0m")
    except Exception as e:
        print("\033[91m[FAIL] Test 2D Result Mismatch!\033[0m")
        print(e)

if __name__ == "__main__":
    test_1d()
    test_2d()