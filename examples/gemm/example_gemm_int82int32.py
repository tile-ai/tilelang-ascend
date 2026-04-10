# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], target="npuir")
def matmul(M, N, K, block_M, block_N, block_K, dtype="int8", accum_dtype="int32"):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (
            cid,
            _,
        ):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, initC=(k == 0))

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    # In the future, Developer mode and Expert Mode will transition smoothly without
    # requiring explicit declarations.
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)

    a = torch.randint(-128, 127, (1024, 1024), dtype=torch.int8).npu()
    b = torch.randint(-128, 127, (1024, 1024), dtype=torch.int8).npu()

    c = kernel(a, b)

    ref_c = (a.to(torch.float32) @ b.to(torch.float32)).to(torch.int32)

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")


if __name__ == "__main__":
    main()
