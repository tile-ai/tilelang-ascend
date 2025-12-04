import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def gemm_tail_block(M, N, K, block_M, block_N, block_K, dtype="float16", accum_type="float32"):
    """
    Able to handle tail block.
    """
    m_num = M // block_M
    n_num = N // block_N
    k_num = K // block_K
    
    m_tail = M - m_num * block_M
    n_tail = N - n_num * block_N
    k_tail = K - k_num * block_K

    total_m_blocks = m_num + (1 if m_tail > 0 else 0)
    total_n_blocks = n_num + (1 if n_tail > 0 else 0)
    total_blocks = total_m_blocks * total_n_blocks

    @T.prim_func
    def main(
        A: T.Tensor([M, K], dtype),
        B: T.Tensor([K, N], dtype),
        C: T.Tensor([M, N], dtype),
    ):
         T.func_attr({"enable_auto_sync": True})
         with T.Kernel(total_blocks, is_npu=True) as (cid, _):
            bx = cid // total_n_blocks
            by = cid % total_n_blocks
            with T.Scope("C"):
                # Case 1: Regular block (no tails)
                if bx < m_num and by < n_num:
                    a = T.alloc_L1([block_M, block_K], dtype)
                    b = T.alloc_L1([block_K, block_N], dtype)
                    c = T.alloc_L0C([block_M, block_N], accum_type)

                    for k in T.serial(k_num):
                        T.copy(A[bx * block_M, k * block_K], a)
                        T.copy(B[k * block_K, by * block_N], b)
                        T.gemm_v1(a, b, c, init=(k == 0))
                    
                    if k_tail > 0:
                        a_k = T.alloc_L1([block_M, k_tail], dtype)
                        b_k = T.alloc_L1([k_tail, block_N], dtype)
                        T.copy(A[bx * block_M, k_num * block_K], a_k)
                        T.copy(B[k_num * block_K, by * block_N], b_k)
                        T.gemm_v1(a_k, b_k, c, init=False)
                    
                    T.copy(c, C[bx * block_M, by * block_N])
                
                # Case 2: M tail block (bottom edge)
                elif bx == m_num and by < n_num and m_tail > 0:
                    a_m = T.alloc_L1([m_tail, block_K], dtype)
                    b = T.alloc_L1([block_K, block_N], dtype)
                    c_m = T.alloc_L0C([m_tail, block_N], accum_type)

                    for k in T.serial(k_num):
                        T.copy(A[bx * block_M, k * block_K], a_m)
                        T.copy(B[k * block_K, by * block_N], b)
                        T.gemm_v1(a_m, b, c_m, init=(k == 0))
                    
                    if k_tail > 0:
                        a_mk = T.alloc_L1([m_tail, k_tail], dtype)
                        b_k = T.alloc_L1([k_tail, block_N], dtype)
                        T.copy(A[bx * block_M, k_num * block_K], a_mk)
                        T.copy(B[k_num * block_K, by * block_N], b_k)
                        T.gemm_v1(a_mk, b_k, c_m, init=False)
                    
                    T.copy(c_m, C[bx * block_M, by * block_N])
                
                # Case 3: N tail block (right edge)
                elif bx < m_num and by == n_num and n_tail > 0:
                    a = T.alloc_L1([block_M, block_K], dtype)
                    b_n = T.alloc_L1([block_K, n_tail], dtype)
                    c_n = T.alloc_L0C([block_M, n_tail], accum_type)

                    for k in T.serial(k_num):
                        T.copy(A[bx * block_M, k * block_K], a)
                        T.copy(B[k * block_K, by * block_N], b_n)
                        T.gemm_v1(a, b_n, c_n, init=(k == 0))
                    
                    if k_tail > 0:
                        a_k = T.alloc_L1([block_M, k_tail], dtype)
                        b_kn = T.alloc_L1([k_tail, n_tail], dtype)
                        T.copy(A[bx * block_M, k_num * block_K], a_k)
                        T.copy(B[k_num * block_K, by * block_N], b_kn)
                        T.gemm_v1(a_k, b_kn, c_n, init=False)
                    
                    T.copy(c_n, C[bx * block_M, by * block_N])
                
                # Case 4: M and N tail block (bottom-right corner)
                elif bx == m_num and by == n_num and m_tail > 0 and n_tail > 0:
                    a_m = T.alloc_L1([m_tail, block_K], dtype)
                    b_n = T.alloc_L1([block_K, n_tail], dtype)
                    c_mn = T.alloc_L0C([m_tail, n_tail], accum_type)

                    for k in T.serial(k_num):
                        T.copy(A[bx * block_M, k * block_K], a_m)
                        T.copy(B[k * block_K, by * block_N], b_n)
                        T.gemm_v1(a_m, b_n, c_mn, init=(k == 0))
                    
                    if k_tail > 0:
                        a_mk = T.alloc_L1([m_tail, k_tail], dtype)
                        b_kn = T.alloc_L1([k_tail, n_tail], dtype)
                        T.copy(A[bx * block_M, k_num * block_K], a_mk)
                        T.copy(B[k_num * block_K, by * block_N], b_kn)
                        T.gemm_v1(a_mk, b_kn, c_mn, init=False)
                    
                    T.copy(c_mn, C[bx * block_M, by * block_N])

    return main

torch.manual_seed(0)
test_configs = [
    (512, 512, 512, 128, 128, 128),
    (512 + 64, 512, 512, 128, 128, 128),
    (512, 512 + 64, 512, 128, 128, 128),
    (512, 512, 512 + 64, 128, 128, 128),
    (512 + 64, 512 + 64, 512, 128, 128, 128),
    (512 + 64, 512, 512 + 64, 128, 128, 128),
    (512, 512 + 64, 512 + 64, 128, 128, 128),
    (512 + 64, 512 + 64, 512 + 64, 128, 128, 128),
]

for M, N, K, block_M, block_N, block_K in test_configs:
    func = gemm_tail_block(M, N, K, block_M, block_N, block_K, dtype="float16", accum_type="float32")
    print("Init successful!")
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()
    c = func(a, b)
    ref_c = torch.matmul(a, b)
    torch.testing.assert_close(c, ref_c, rtol=1e-3, atol=1e-3)
    print("Test passed!")
print("Kernel Output Match!")