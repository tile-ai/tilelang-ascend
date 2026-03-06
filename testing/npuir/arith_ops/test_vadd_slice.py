# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[-1], target="npuir")
def simple_copy_1d(L, block_L, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        In: T.Tensor((L,), dtype),
        A: T.Tensor((L,), dtype),
        B: T.Tensor((L,), dtype),
        C: T.Tensor((L,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(L, block_L), is_npu=True) as (cid, _):
            start_idx = cid * block_L
            
            A_frag = T.alloc_fragment((block_L,), dtype)
            B_frag = T.alloc_fragment((block_L,), dtype)
            C_frag = T.alloc_fragment((block_L,), accum_dtype)
            
            T.copy(In[start_idx], A_frag)

            T.copy(A_frag, B_frag)  # ub to ub
            T.npuir_add(A_frag[0:32], A_frag[0:32], B_frag[0:32])
            T.copy(B_frag, C_frag)  # ub to ub with cast

            T.copy(A_frag, A[start_idx])
            T.copy(B_frag, B[start_idx])
            T.copy(C_frag, C[start_idx])

    return main

@tilelang.jit(out_idx=[-1], target="npuir")
def simple_copy_2d(M, N, block_M, block_N, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        In: T.Tensor((M, N), dtype),
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            A_frag = T.alloc_fragment((block_M, block_N), dtype)
            B_frag = T.alloc_fragment((block_M, block_N), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            tmp = T.alloc_fragment((3, block_N), dtype)

            T.copy(In[by * block_M, bx * block_N], A_frag)
            T.copy(A_frag, B_frag)
            T.npuir_add(A_frag[2:4,:], A_frag[2:4,:], B_frag[2:4,:])
            T.copy(B_frag, C_frag)  # ub to ub with cast

            T.copy(A_frag, A[by * block_M, bx * block_N])
            T.copy(B_frag, B[by * block_M, bx * block_N])
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return main

@tilelang.jit(out_idx=[-1], target="npuir")
def simple_copy_3d(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        In: T.Tensor((M, N, K), dtype),
        A: T.Tensor((M, N, K), dtype),
        B: T.Tensor((M, N, K), dtype),
        C: T.Tensor((M, N, K), accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(M, block_M) * T.ceildiv(N, block_N) * T.ceildiv(K, block_K), 
            is_npu=True
        ) as (cid, _):
            
            num_blocks_N = T.ceildiv(N, block_N)
            num_blocks_K = T.ceildiv(K, block_K)

            # cid = bm * (num_blocks_N * num_blocks_K) + bn * num_blocks_K + bk
            bk = cid % num_blocks_K
            bn = (cid // num_blocks_K) % num_blocks_N
            bm = cid // (num_blocks_N * num_blocks_K)
            
            start_m = bm * block_M
            start_n = bn * block_N
            start_k = bk * block_K
            
            A_frag = T.alloc_fragment((block_M, block_N, block_K), dtype)
            B_frag = T.alloc_fragment((block_M, block_N, block_K), dtype)
            C_frag = T.alloc_fragment((block_M, block_N, block_K), accum_dtype)
            
            T.copy(In[start_m, start_n, start_k], A_frag)
            T.copy(A_frag, B_frag)  # ub to ub
            T.npuir_add(A_frag[2:4,:,:], B_frag[2:4,:,:], B_frag[2:4,:,:])
            
            T.copy(B_frag, C_frag)  # ub to ub with cast
            
            T.copy(A_frag, A[start_m, start_n, start_k])
            T.copy(B_frag, B[start_m, start_n, start_k])
            T.copy(C_frag, C[start_m, start_n, start_k])
    
    return main

def test_1d():
    print("Testing 1d copy...")
    kernel = simple_copy_1d(64, 64)
    
    input = torch.ones(64).npu().half()
    a = torch.zeros(64).npu().half()
    b = torch.zeros(64).npu().half()
    c = torch.zeros(64).npu()
    
    kernel(input, a, b, c)
    print("Input:\n", input)
    print("a:\n", a)
    print("b:\n", b)
    print("c:\n", c)
    torch.testing.assert_close(a, input, rtol=1e-5, atol=1e-5)
    b_ref = input.clone()
    b_ref[0:32] = a[0:32] * 2
    torch.testing.assert_close(b, b_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(c, b.to(torch.float32), rtol=1e-5, atol=1e-5)
    print("1D Test success!")

def test_2d():
    print("Testing 2d copy...")
    kernel = simple_copy_2d(32, 16, 32, 16)

    input = torch.randn(32, 16).npu().half()
    a = torch.zeros(32, 16).npu().half()
    b = torch.zeros(32, 16).npu().half()
    c = torch.zeros(32, 16).npu()

    kernel(input, a, b, c)
    print("input:\n", input)
    print("a:\n", a)
    print("b:\n", b)
    print("c:\n", c)
    print("a, input")
    torch.testing.assert_close(a, input, rtol=1e-5, atol=1e-5)
    b_ref = input.clone()
    b_ref[2:4, :] = b_ref[2:4, :] * 2

    print("b, b_ref")
    torch.testing.assert_close(b, b_ref, rtol=1e-5, atol=1e-5)
    # torch.testing.assert_close(
    #     c,
    #     b_ref.to(torch.float32),
    #     rtol=1e-5,
    #     atol=1e-5,
    # )
    print("2D Test success!")

def reference_simple_copy_3d(In, M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    """
    Reference implementation mimicking simple_copy_3d behavior
    including fragment copy, ub-to-ub copy, slice add, and type cast.
    """
    # Output tensors
    A = torch.zeros((M, N, K), dtype=getattr(torch, dtype))
    B = torch.zeros((M, N, K), dtype=getattr(torch, dtype))
    C = torch.zeros((M, N, K), dtype=getattr(torch, accum_dtype))
    
    num_blocks_N = (N + block_N - 1) // block_N
    num_blocks_K = (K + block_K - 1) // block_K

    # Iterate over all blocks (simulate cid loop)
    for bm in range((M + block_M - 1) // block_M):
        for bn in range(num_blocks_N):
            for bk in range(num_blocks_K):
                start_m = bm * block_M
                start_n = bn * block_N
                start_k = bk * block_K

                # Slice the block
                m_slice = slice(start_m, start_m + block_M)
                n_slice = slice(start_n, start_n + block_N)
                k_slice = slice(start_k, start_k + block_K)

                # Allocate fragments (ub)
                A_frag = In[m_slice, n_slice, k_slice].clone()
                B_frag = A_frag.clone()
                C_frag = torch.zeros_like(A_frag, dtype=getattr(torch, accum_dtype))

                # ub-to-ub add on slice [2:4,:,:]
                # Ensure we don't exceed fragment boundary
                frag_m = A_frag.shape[0]
                m_subslice = slice(2, min(4, frag_m))
                B_frag[m_subslice, :, :] += A_frag[m_subslice, :, :]

                # Copy B_frag to C_frag with type cast
                C_frag = B_frag.to(getattr(torch, accum_dtype))

                # Write back to global memory
                A[m_slice, n_slice, k_slice] = A_frag
                B[m_slice, n_slice, k_slice] = B_frag
                C[m_slice, n_slice, k_slice] = C_frag

    return A, B, C

def test_3d():
    print("Testing 3d copy...")
    
    M, N, K = 64, 128, 256
    block_M, block_N, block_K = 16, 32, 32
    
    assert M % block_M == 0, f"M({M}) must be divisible by block_M({block_M})"
    assert N % block_N == 0, f"N({N}) must be divisible by block_N({block_N})"
    assert K % block_K == 0, f"K({K}) must be divisible by block_K({block_K})"
    
    kernel = simple_copy_3d(M, N, K, block_M, block_N, block_K)

    input = torch.randn(M, N, K).npu().half()
    a = torch.zeros(M, N, K).npu().half()
    b = torch.zeros(M, N, K).npu().half()
    c = torch.zeros(M, N, K).npu()
    
    kernel(input, a, b, c)
    print("input:\n", input)
    print("a:\n", a)
    print("b:\n", b)
    print("c:\n", c)
    A_ref, B_ref, C_ref = reference_simple_copy_3d(input, M, N, K, block_M, block_N, block_K)
    torch.testing.assert_close(a.cpu(), A_ref.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(b.cpu(), B_ref.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(c.cpu(), C_ref.cpu(), rtol=1e-5, atol=1e-5)
    
    print("3D test success!")

def main():
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'

    test_1d()
    test_2d()
    test_3d()

if __name__ == "__main__":
    main()
