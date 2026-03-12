# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import argparse
import os
import torch
import torch_npu
import tilelang as tl
import tilelang.language as T


@tl.jit(target="npuir")
def naive_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float32"
):
    @T.prim_func
    def main(
        A: T.Tensor((K,), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), is_npu=True) as (bn, _):
            A_shared = T.alloc_shared((BLOCK_K,), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_reg = T.alloc_shared((1,), accum_dtype)
            for tn in T.serial(BLOCK_N):
                T.clear(C_reg)
                for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                    for tk in T.serial(BLOCK_K):
                        A_shared[tk] = A[bk * BLOCK_K + tk]
                        B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                    for tk in T.serial(BLOCK_K):
                        C_reg[0] += A_shared[tk].astype(accum_dtype) * B_shared[tn, tk].astype(accum_dtype)
                C[bn * BLOCK_N + tn] = C_reg[0]

    return main


@tl.jit(target="npuir")
def naive_gemv_high_perf(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float32"
):
    @T.prim_func
    def naive_gemv_hp(
        A: T.Tensor((K,), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), is_npu=True) as (cid, _):
            A_shared = T.alloc_shared((BLOCK_K, 1), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_shared = T.alloc_shared((BLOCK_N, 1), accum_dtype)
            T.clear(C_shared)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                T.copy(A[bk * BLOCK_K:(bk + 1)*BLOCK_K], A_shared[:, 0])
                T.copy(B[cid*BLOCK_N, bk*BLOCK_K], B_shared)
                T.gemm(B_shared, A_shared, C_shared, initC=False)
            T.copy(C_shared[:,0], C[cid*BLOCK_N:(cid+1)*BLOCK_N])

    return naive_gemv_hp

@tl.jit(target="npuir")
def naive_splitk_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float32"
):
    @T.prim_func
    def main(
        A: T.Tensor((K,), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), is_npu=True) as (cid, _):
            A_local = T.alloc_local((1,), dtype)
            B_local = T.alloc_local((1,), dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            for tn in T.serial(BLOCK_N):
                T.clear(C_accum)
                for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                    for tk in T.serial(BLOCK_K):
                        A_local[0] = A[bk*BLOCK_K + tk]
                        B_local[0] = B[cid*BLOCK_N + tn, bk*BLOCK_K + tk]
                        C_accum[0] += (A_local[0].astype(accum_dtype) * B_local[0].astype(accum_dtype))
                C[cid*BLOCK_N + tn] = C_accum[0]
    return main

@tl.jit(target="npuir")
def naive_splitk_gemv_high_perf(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float32"
):
    @T.prim_func
    def naive_splitk_gemv_hp(
        A: T.Tensor((K,), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), is_npu=True) as (cid, _):
            A_shared = T.alloc_shared((BLOCK_K,1), dtype)
            B_shared = T.alloc_shared((1,BLOCK_K), dtype)
            C_accum = T.alloc_shared((1,1), accum_dtype)
            for tn in T.serial(BLOCK_N):
                T.clear(C_accum)
                for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                    T.copy(A[bk * BLOCK_K:(bk + 1)*BLOCK_K], A_shared[:, 0])
                    T.copy(B[cid*BLOCK_N + tn, bk*BLOCK_K:(bk+1)*BLOCK_K], B_shared)
                    T.gemm(B_shared, A_shared, C_accum, initC=False)
                T.copy(C_accum[0,0], C[cid*BLOCK_N + tn:(cid)*BLOCK_N + tn + 1])
    return naive_splitk_gemv_hp

def main():
    parser = argparse.ArgumentParser(description="GEMV Example")
    parser.add_argument("--n", type=int, default=1024, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=1024, help="Matrix dimension K")
    args, _ = parser.parse_known_args()
    N, K = args.n, args.k

    # case 1
    # kernel = naive_gemv(N, K, 128, 128)
    kernel1 = naive_gemv_high_perf(N, K, 128, 128)
    A = torch.randn((K,), dtype=torch.float16).npu()
    B = torch.randn((N, K), dtype=torch.float16).npu()
    C = torch.randn((N,), dtype=torch.float16).npu()
    kernel1(A, B, C)
    print(C)
    res = torch.matmul(B,A)
    print(res)
    torch.testing.assert_close(C, res, rtol=1e-2, atol=1e-2)

    # case 2
    # kernel = naive_splitk_gemv(N, K, 128, 128)
    kernel2 = naive_splitk_gemv_high_perf(N, K, 128, 128)
    A = torch.randn((K,), dtype=torch.float16).npu()
    B = torch.randn((N, K), dtype=torch.float16).npu()
    C = torch.randn((N,), dtype=torch.float16).npu()
    kernel2(A, B, C)
    print(C)
    res = torch.matmul(B,A)
    print(res)
    torch.testing.assert_close(C, res, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    main()
