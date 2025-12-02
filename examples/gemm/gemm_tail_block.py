import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul_with_tail(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    # Calculate main blocks and tail sizes
    m_main_blocks = M // block_M
    n_main_blocks = N // block_N
    
    m_tail = M % block_M
    n_tail = N % block_N
    k_tail = K % K_L1
    
    has_m_tail = m_tail > 0
    has_n_tail = n_tail > 0
    
    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        # Process main M x main N blocks
        with T.Kernel(m_main_blocks * n_main_blocks, is_npu=True) as (cid, _):
            bx = cid // n_main_blocks
            by = cid % n_main_blocks

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    is_k_tail = (k == loop_k - 1) and (k_tail > 0)
                    curr_k = k_tail if is_k_tail else K_L1
                    
                    T.copy(A[bx * block_M : bx * block_M + block_M, 
                            k * K_L1 : k * K_L1 + curr_k], 
                          A_L1[:block_M, :curr_k])
                    
                    T.copy(B[k * K_L1 : k * K_L1 + curr_k, 
                            by * block_N : by * block_N + block_N], 
                          B_L1[:curr_k, :block_N])

                    T.gemm_v0(A_L1[:block_M, :curr_k], 
                             B_L1[:curr_k, :block_N], 
                             C_L0[:block_M, :block_N], 
                             init=(k == 0))

                T.copy(C_L0[:block_M, :block_N], 
                      C[bx * block_M : bx * block_M + block_M, 
                        by * block_N : by * block_N + block_N])

        # Process M tail x main N blocks
        if has_m_tail:
            with T.Kernel(n_main_blocks, is_npu=True) as (by, _):
                bx = m_main_blocks  # Last row block
                
                A_L1 = T.alloc_L1((block_M, K_L1), dtype)
                B_L1 = T.alloc_L1((K_L1, block_N), dtype)
                C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

                with T.Scope("C"):
                    loop_k = T.ceildiv(K, K_L1)
                    for k in T.serial(loop_k):
                        is_k_tail = (k == loop_k - 1) and (k_tail > 0)
                        curr_k = k_tail if is_k_tail else K_L1
                        
                        T.copy(A[bx * block_M : bx * block_M + m_tail, 
                                k * K_L1 : k * K_L1 + curr_k], 
                              A_L1[:m_tail, :curr_k])
                        
                        T.copy(B[k * K_L1 : k * K_L1 + curr_k, 
                                by * block_N : by * block_N + block_N], 
                              B_L1[:curr_k, :block_N])

                        T.gemm_v0(A_L1[:m_tail, :curr_k], 
                                 B_L1[:curr_k, :block_N], 
                                 C_L0[:m_tail, :block_N], 
                                 init=(k == 0))

                    T.copy(C_L0[:m_tail, :block_N], 
                          C[bx * block_M : bx * block_M + m_tail, 
                            by * block_N : by * block_N + block_N])

        # Process main M x N tail blocks
        if has_n_tail:
            with T.Kernel(m_main_blocks, is_npu=True) as (bx, _):
                by = n_main_blocks  # Last column block
                
                A_L1 = T.alloc_L1((block_M, K_L1), dtype)
                B_L1 = T.alloc_L1((K_L1, block_N), dtype)
                C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

                with T.Scope("C"):
                    loop_k = T.ceildiv(K, K_L1)
                    for k in T.serial(loop_k):
                        is_k_tail = (k == loop_k - 1) and (k_tail > 0)
                        curr_k = k_tail if is_k_tail else K_L1
                        
                        T.copy(A[bx * block_M : bx * block_M + block_M, 
                                k * K_L1 : k * K_L1 + curr_k], 
                              A_L1[:block_M, :curr_k])
                        
                        T.copy(B[k * K_L1 : k * K_L1 + curr_k, 
                                by * block_N : by * block_N + n_tail], 
                              B_L1[:curr_k, :n_tail])


                        T.gemm_v0(A_L1[:block_M, :curr_k], 
                                 B_L1[:curr_k, :n_tail], 
                                 C_L0[:block_M, :n_tail], 
                                 init=(k == 0))

                    T.copy(C_L0[:block_M, :n_tail], 
                          C[bx * block_M : bx * block_M + block_M, 
                            by * block_N : by * block_N + n_tail])

        # Process M tail x N tail block (corner)
        if has_m_tail and has_n_tail:
            with T.Kernel(1, is_npu=True) as (_, _):
                bx = m_main_blocks
                by = n_main_blocks
                
                A_L1 = T.alloc_L1((block_M, K_L1), dtype)
                B_L1 = T.alloc_L1((K_L1, block_N), dtype)
                C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

                with T.Scope("C"):
                    loop_k = T.ceildiv(K, K_L1)
                    for k in T.serial(loop_k):
                        is_k_tail = (k == loop_k - 1) and (k_tail > 0)
                        curr_k = k_tail if is_k_tail else K_L1
                        
                        T.copy(A[bx * block_M : bx * block_M + m_tail, 
                                k * K_L1 : k * K_L1 + curr_k], 
                              A_L1[:m_tail, :curr_k])
                        
                        T.copy(B[k * K_L1 : k * K_L1 + curr_k, 
                                by * block_N : by * block_N + n_tail], 
                              B_L1[:curr_k, :n_tail])

                        T.gemm_v0(A_L1[:m_tail, :curr_k], 
                                 B_L1[:curr_k, :n_tail], 
                                 C_L0[:m_tail, :n_tail], 
                                 init=(k == 0))

                    T.copy(C_L0[:m_tail, :n_tail], 
                          C[bx * block_M : bx * block_M + m_tail, 
                            by * block_N : by * block_N + n_tail])

    return main


torch.manual_seed(0)
test_configs = [
    (1024, 1024, 1024, 128, 256, 64),  # No tail
    (1000, 1000, 1000, 128, 256, 64),  # All dimensions have tails
    (1024, 1000, 1024, 128, 256, 64),  # Only N has tail
    (1000, 1024, 1024, 128, 256, 64),  # Only M has tail
    (1024, 1024, 1000, 128, 256, 64),  # Only K has tail
]

for M, N, K, block_M, block_N, K_L1 in test_configs:
    func = matmul_with_tail(M, N, K, block_M, block_N, K_L1, dtype="float16")
    print(f"Testing M={M}, N={N}, K={K}...")
    
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()
    c = func(a, b)
    
    ref_c = a @ b
    
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print(f"  Tail sizes: M_tail={M % block_M}, N_tail={N % block_N}, K_tail={K % K_L1} - Passed!")

print("All tests passed!")