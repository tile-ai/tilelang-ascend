# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], target="npuir")
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, K), dtype),
        D: T.Tensor((K, N), dtype),
        Output: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            AB_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            C_shared = T.alloc_shared((block_M, block_K), dtype)
            D_shared = T.alloc_shared((block_K, block_N), dtype)
            CD_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            Output_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, AB_local, initC=(k == 0))

                T.copy(C[by * block_M, k * block_K], C_shared)
                T.copy(D[k * block_K, bx * block_N], D_shared)
                T.gemm(C_shared, D_shared, CD_local, initC=(k == 0))

                T.npuir_add(AB_local, CD_local, Output_local)

            T.copy(Output_local, Output[by * block_M, bx * block_N])

    return gemm


def main():
    # In the futrue, Developer mode and Expert Mode will transition smoothly without
    # requiring explicit declarations.
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)

    a = torch.randn(1024, 1024).npu().half()
    b = torch.randn(1024, 1024).npu().half()
    c = torch.randn(1024, 1024).npu().half()
    d = torch.randn(1024, 1024).npu().half()
    output = torch.randn(1024, 1024).npu().half()

    kernel(a, b, c, d, output)

    ref_output = a @ b + c @ d

    print("c:")
    print(output)
    print("ref_c:")
    print(ref_output)

    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print("All check passed.")


if __name__ == "__main__":
    main()