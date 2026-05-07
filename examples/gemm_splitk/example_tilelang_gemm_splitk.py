import argparse

import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def gemm_splitk(M, N, K, block_M, block_N, block_K, split_k, dtype="float16", accum_dtype="float"):
    n_num = (N + block_N - 1) // block_N
    m_num = (M + block_M - 1) // block_M
    splitK = K // split_k
    m_split_blocks = m_num * split_k

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), accum_dtype),  # type: ignore
    ):
        with T.Kernel(n_num * m_split_blocks, is_npu=True) as (cid, _):
            bx = cid // m_split_blocks
            rem = cid % m_split_blocks
            by = rem // split_k
            bz = rem % split_k

            A_L1 = T.alloc_shared((block_M, block_K), dtype)
            B_L1 = T.alloc_shared((block_K, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            k_start = bz * splitK
            loop_k = T.ceildiv(splitK, block_K)

            for k in T.serial(loop_k):
                T.copy(A[by * block_M, k_start + k * block_K], A_L1)
                T.copy(B[k_start + k * block_K, bx * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.tile.atomic_add(C[by * block_M, bx * block_N], C_L0)

    return main


def test(M, N, K, split_k):
    block_M = 128
    block_N = 128
    block_K = 32

    kernel = gemm_splitk(M, N, K, block_M, block_N, block_K, split_k)

    torch.manual_seed(42)

    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()

    c = kernel(a, b)

    ref_c = a @ b

    torch.testing.assert_close(c, ref_c.to(c.dtype), rtol=1e-2, atol=1e-2)

    print(f"Test passed: M={M}, N={N}, K={K}, split_k={split_k}")


def main():
    parser = argparse.ArgumentParser(description="TileLang-Ascend Split-K GEMM")
    parser.add_argument("--M", type=int, default=0)
    parser.add_argument("--N", type=int, default=0)
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--split-k", type=int, default=0)
    args = parser.parse_args()

    tilelang.disable_cache()

    if args.M > 0 and args.N > 0 and args.K > 0 and args.split_k > 0:
        test(args.M, args.N, args.K, args.split_k)
    else:
        test(128, 128, 128, 2)
        test(1024, 1024, 1024, 4)

    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
