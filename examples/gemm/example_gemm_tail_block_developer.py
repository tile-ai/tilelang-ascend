import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)

            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main

torch.manual_seed(0)
test_configs = [
    (32 * 3 + 30, 32 * 2 + 16, 32 * 4 + 31, 32, 32, 32),
    (512 + 64, 512, 512, 128, 128, 128),
    (512, 512 + 64, 512, 128, 128, 128),
    (512, 512, 512 + 64, 128, 128, 128),
    (512 + 64, 512 + 64, 512, 128, 128, 128),
    (512 + 64, 512, 512 + 64, 128, 128, 128),
    (512, 512 + 64, 512 + 64, 128, 128, 128),
    (512 + 64, 512 + 64, 512 + 64, 128, 128, 128),
]

for M, N, K, block_M, block_N, block_K in test_configs:
    func = matmul(M, N, K, block_M, block_N, block_K)
    print("init successful!")
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()
    c = torch.empty(M, N).half().npu()
    c = func(a, b)
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Test passed!")
print("Kernel Output Match!")

