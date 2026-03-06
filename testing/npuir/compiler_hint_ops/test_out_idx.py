# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[-1], target="npuir")
def matmul(block_M, block_N, K_L1, dtype="float16", accum_dtype="float32"):
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
                    T.npuir_load_nd2nz(A[bx, i * K_L1], A_BUF, [remain_M, remain_K])
                    T.npuir_load_nd2nz(B[i * K_L1, by], B_BUF, [remain_K, remain_N])

                    if i == 0:
                        T.npuir_dot(A_BUF, B_BUF, C_BUF, initC=True, b_transpose=False,
                            size=[remain_M, remain_K, remain_N])
                    else:
                        T.npuir_dot(A_BUF, B_BUF, C_BUF, initC=False, b_transpose=False,
                            size=[remain_M, remain_K, remain_N])

                    T.npuir_store_fixpipe(C_BUF, C[bx, by],
                        size=[remain_M, remain_N], enable_nz2nd=True)

    return main


def test_mat_mul():
    
    func = matmul(128, 256, 16)
    # shape 1
    M, N, K = 1024, 512, 2048
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()
    
    c = func(a, b)
    print(c)
    ref_c = a @ b
    print(ref_c)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    # shape 2
    M, N, K = 512, 1024, 512
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()
    
    c = func(a, b)
    print(c)
    ref_c = a @ b
    print(ref_c)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

@tilelang.jit(out_idx=[-2, -1], target="npuir")
def minicv(M, N, K, block_M, block_N, dtype="float16", inner_dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def minicv(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), inner_dtype),
            D: T.Tensor((M, N), inner_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            blockx = cid // n_num
            bx = blockx * block_M
            blocky = cid % n_num
            by = blocky * block_N

            A_BUF = T.alloc_shared((block_M, K), dtype)
            B_BUF = T.alloc_shared((K, block_N), dtype)
            C_BUF = T.alloc_fragment((block_M, block_N), inner_dtype)
            D_BUF = T.alloc_fragment((block_M, block_N), inner_dtype)

            T.copy(A[bx, 0], A_BUF, [block_M, K])
            T.copy(B[0, by], B_BUF, [K, block_N])

            T.gemm(A_BUF, B_BUF, C_BUF, [block_M, K, block_N], initC = True)
            T.vexp(C_BUF, D_BUF)

            T.copy(C_BUF, C[bx, by], [block_M, block_N])
            T.copy(D_BUF, D[bx, by], [block_M, block_N])

    return minicv

def test_minicv():
    M = 1024
    N = 1024
    K = 512
    dtype="float16"
    inner_dtype="float32"

    os.environ['TILELANG_ASCEND_WORKSPACE_SIZE'] = str(M * N)
    func = minicv(M, N, K, 128, 256)

    v1 = torch.randn(size=[M, K], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[K, N], dtype=eval("torch." + dtype)).npu()

    y_ref = torch.exp(v1.to(torch.float32) @ v2.to(torch.float32))
    v3, v4 = func(v1, v2)

    print(y_ref)
    print(v4)
    torch.testing.assert_close(v4, y_ref, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":
    # test dynamic shape
    os.environ['TILELANG_ASCEND_MODE'] = 'Expert'
    test_mat_mul()
    # test multi out_idxs
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    test_minicv()
 