# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import torch
import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

@tilelang.jit(target="npuir")
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
    c = torch.randn(M, N).half().npu()
    func(a, b, c)
    print(c)
    ref_c = a @ b
    print(ref_c)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    # shape 2
    M, N, K = 512, 1024, 512
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()
    c = torch.randn(M, N).half().npu()
    func(a, b, c)
    print(c)
    ref_c = a @ b
    print(ref_c)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    # shape 3
    M, N, K = 512, 512, 1264
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()
    c = torch.randn(M, N).half().npu()
    func(a, b, c)
    print(c)
    ref_c = a @ b
    print(ref_c)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    # print result
    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":
    test_mat_mul()
 