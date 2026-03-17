import argparse
import os
import torch
import tilelang as tl
import tilelang.language as T


@tl.jit(target="npuir")
def rms_norm(M: int, N: int, BLOCK_M: int, dtype: str = "float32"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, BLOCK_M), is_npu=True) as (cid, _):
            A_shared = T.alloc_shared((BLOCK_M, N), dtype)
            A_pow = T.alloc_shared((BLOCK_M, N), dtype)
            A_pow_sum = T.alloc_shared((BLOCK_M, 1), dtype)

            T.copy(A[cid * BLOCK_M : (cid + 1) * BLOCK_M, :], A_shared)
            for i, j in T.Parallel(BLOCK_M, N):
                A_pow[i, j] = A_shared[i, j] * A_shared[i, j]
            T.reduce_sum(A_pow, A_pow_sum, dim=1)
            for i in T.Parallel(BLOCK_M):
                A_pow_sum[i, 0] = A_pow_sum[i, 0] / N + 1e-12
            T.npuir_rsqrt(A_pow_sum, A_pow_sum)
            for i, j in T.Parallel(BLOCK_M, N):
                A_shared[i, j] *= A_pow_sum[i, 0]
            T.copy(A_shared, B[cid * BLOCK_M : (cid + 1) * BLOCK_M, :])

    return main


@tl.jit(target="npuir")
def rms_norm_high_perf(M: int, N: int, BLOCK_M: int, dtype: str = "float32"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, BLOCK_M), is_npu=True) as (cid, _):
            A_shared = T.alloc_shared((BLOCK_M, N), dtype)
            A_pow = T.alloc_shared((BLOCK_M, N), dtype)
            A_pow_sum = T.alloc_shared((BLOCK_M, 1), dtype)

            T.copy(A[cid * BLOCK_M : (cid + 1) * BLOCK_M, :], A_shared)
            T.npuir_mul(A_shared, A_shared, A_pow)
            T.reduce_sum(A_pow, A_pow_sum, dim=1)
            T.npuir_div(A_pow_sum, N, A_pow_sum)
            T.npuir_add(A_pow_sum, 1e-12, A_pow_sum)
            T.npuir_rsqrt(A_pow_sum, A_pow_sum)
            T.npuir_mul(A_shared, A_pow_sum, A_shared)
            T.copy(A_shared, B[cid * BLOCK_M : (cid + 1) * BLOCK_M, :])

    return main


@tl.jit(target="npuir")
def rms_norm_splitk(M: int, N: int, BLOCK_M: int, BLOCK_K: int, dtype: str = "float32"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, BLOCK_M), is_npu=True) as (cid, _):
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            A_accm = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            A_pow_sum = T.alloc_shared((BLOCK_M, 1), dtype)
            num_k_step = T.ceildiv(N, BLOCK_K)
            T.clear(A_accm)
            for k in range(num_k_step):
                T.copy(A[cid * BLOCK_M, k * BLOCK_K], A_shared)
                for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                    A_accm[i, j] += A_shared[i, j] * A_shared[i, j]
            T.reduce_sum(A_accm, A_pow_sum, dim=1)
            for i in T.Parallel(BLOCK_M):
                A_pow_sum[i, 0] = A_pow_sum[i, 0] / N + 1e-12
            T.npuir_rsqrt(A_pow_sum, A_pow_sum)

            for k in range(num_k_step):
                T.copy(A[cid * BLOCK_M, (num_k_step - 1 - k) * BLOCK_K], A_shared)
                for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                    A_shared[i, j] *= A_pow_sum[i, 0]
                T.copy(A_shared, B[cid * BLOCK_M, (num_k_step - 1 - k) * BLOCK_K])

    return main


@tl.jit(target="npuir")
def rms_norm_splitk_high_perf(
    M: int, N: int, BLOCK_M: int, BLOCK_K: int, dtype: str = "float32"
):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, BLOCK_M), is_npu=True) as (cid, _):
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            A_accm = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            A_pow_sum = T.alloc_shared((BLOCK_M, 1), dtype)

            num_k_step = T.ceildiv(N, BLOCK_K)
            T.clear(A_accm)
            for k in T.serial(num_k_step):
                T.copy(A[cid * BLOCK_M, k * BLOCK_K], A_shared)
                T.npuir_mul(A_shared, A_shared, A_shared)
                T.npuir_add(A_accm, A_shared, A_accm)
            T.reduce_sum(A_accm, A_pow_sum, dim=1)
            T.npuir_div(A_pow_sum, N, A_pow_sum)
            T.npuir_add(A_pow_sum, 1e-12, A_pow_sum)
            T.npuir_rsqrt(A_pow_sum, A_pow_sum)

            for k in T.serial(num_k_step):
                T.copy(A[cid * BLOCK_M, (num_k_step - 1 - k) * BLOCK_K], A_shared)
                T.npuir_mul(A_shared, A_pow_sum, A_shared)
                T.copy(A_shared, B[cid * BLOCK_M, (num_k_step - 1 - k) * BLOCK_K])

    return main


def main():
    parser = argparse.ArgumentParser(description="Rms_Norm Example")
    parser.add_argument("--m", type=int, default=1024, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=1024, help="Matrix dimension N")
    args, _ = parser.parse_known_args()
    M, N = args.m, args.n

    kernel = rms_norm_splitk_high_perf(M, N, 32, 32)
    A = torch.randn((M, N), dtype=torch.float32).npu()
    B = torch.randn((M, N), dtype=torch.float32).npu()
    kernel(A, B)
    print(B)
    res = A * torch.rsqrt(A.pow(2).mean(-1, keepdim=True) + 1e-12)
    print(res)
    torch.testing.assert_close(B, res, rtol=1e-2, atol=1e-2)

    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    main()
