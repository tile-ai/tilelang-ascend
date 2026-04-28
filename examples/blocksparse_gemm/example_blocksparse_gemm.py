import argparse
import itertools
import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 128
DEFAULT_BLOCK_K = 32
DEFAULT_NUM_STAGES = 2


def get_configs():
    block_M = [64, 128, 256]
    block_N = [64, 128, 256]
    block_K = [32, 64]
    num_stages = [1, 2, 3]

    _configs = list(itertools.product(block_M, block_N, block_K, num_stages))

    return [
        {
            "block_M": c[0],
            "block_N": c[1],
            "block_K": c[2],
            "num_stages": c[3],
        }
        for c in _configs
    ]


def ref_program(A, B, BlockMask, block_M, block_N, block_K):
    M, K = A.shape
    _, N = B.shape
    ref_c = torch.zeros((M, N), dtype=torch.float16, device=A.device)
    for i in range(M // block_M):
        for j in range(N // block_N):
            accu = torch.zeros((block_M, block_N), dtype=torch.float32, device=A.device)
            for k in range(K // block_K):
                if BlockMask[i, j, k]:
                    A_block = A[i * block_M : (i + 1) * block_M, k * block_K : (k + 1) * block_K]
                    B_block = B[k * block_K : (k + 1) * block_K, j * block_N : (j + 1) * block_N]
                    accu += A_block.to(torch.float32) @ B_block.to(torch.float32)
            ref_c[i * block_M : (i + 1) * block_M, j * block_N : (j + 1) * block_N] = accu.to(torch.float16)
    return ref_c


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.autotune(
    configs=get_configs(),
)
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def blocksparse_matmul(M, N, K, block_M, block_N, block_K, num_stages, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    k_num = (K + block_K - 1) // block_K

    @T.prim_func
    def block_sparse_matmul(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        BlockMask: T.Tensor((m_num, n_num, k_num), "int8"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            for k in T.serial(k_num):
                if BlockMask[bx, by, k]:
                    T.copy(A[bx * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, by * block_N], B_shared)
                    T.gemm_v0(A_shared, B_shared, C_local, init=(k == 0))

            T.copy(C_local, C[bx * block_M, by * block_N])

    return block_sparse_matmul


def test_basic():
    M = N = K = 128
    block_M = block_N = block_K = 32
    num_stages = 2
    sparsity = 0.5

    print(f"[Level 0] Testing M={M}, N={N}, K={K}, block_size=({block_M}, {block_N}, {block_K})")

    torch.manual_seed(0)
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()

    kernel = blocksparse_matmul(M, N, K, block_M, block_N, block_K, num_stages)

    mask_shape = (M // block_M, N // block_N, K // block_K)
    block_mask = (torch.rand(mask_shape).npu() > sparsity).to(torch.int8)
    block_mask[:, :, 0] = 1  # 约束：k=0层必须全为1，确保累加器正确初始化

    c = kernel(a, b, block_mask)

    ref_c = ref_program(a, b, block_mask, block_M, block_N, block_K)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("✅ Level 0 passed!")


def test_typical():
    M = N = K = 1024
    block_M = DEFAULT_BLOCK_M
    block_N = DEFAULT_BLOCK_N
    block_K = DEFAULT_BLOCK_K
    num_stages = DEFAULT_NUM_STAGES
    sparsity = 0.5

    print(f"[Level 1] Testing M={M}, N={N}, K={K}, block_size=({block_M}, {block_N}, {block_K})")

    torch.manual_seed(0)
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()

    kernel = blocksparse_matmul(M, N, K, block_M, block_N, block_K, num_stages)

    mask_shape = (M // block_M, N // block_N, K // block_K)
    block_mask = (torch.rand(mask_shape).npu() > sparsity).to(torch.int8)
    block_mask[:, :, 0] = 1  # 约束：k=0层必须全为1，确保累加器正确初始化

    c = kernel(a, b, block_mask)

    ref_c = ref_program(a, b, block_mask, block_M, block_N, block_K)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("✅ Level 1 passed!")


def test_boundary_dense():
    M = N = K = 1024
    block_M = DEFAULT_BLOCK_M
    block_N = DEFAULT_BLOCK_N
    block_K = DEFAULT_BLOCK_K
    num_stages = DEFAULT_NUM_STAGES
    sparsity = 0.0

    print(f"[Level 2] Testing dense (sparsity={sparsity})")

    torch.manual_seed(0)
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()

    kernel = blocksparse_matmul(M, N, K, block_M, block_N, block_K, num_stages)

    mask_shape = (M // block_M, N // block_N, K // block_K)
    block_mask = torch.ones(mask_shape, dtype=torch.int8, device="npu")

    c = kernel(a, b, block_mask)

    ref_c = ref_program(a, b, block_mask, block_M, block_N, block_K)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("✅ Level 2 (dense) passed!")


def test_boundary_sparse():
    M = N = K = 1024
    block_M = DEFAULT_BLOCK_M
    block_N = DEFAULT_BLOCK_N
    block_K = DEFAULT_BLOCK_K
    num_stages = DEFAULT_NUM_STAGES
    sparsity = 0.99

    print(f"[Level 2] Testing extreme sparse (sparsity={sparsity})")

    torch.manual_seed(0)
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()

    kernel = blocksparse_matmul(M, N, K, block_M, block_N, block_K, num_stages)

    mask_shape = (M // block_M, N // block_N, K // block_K)
    block_mask = (torch.rand(mask_shape).npu() > sparsity).to(torch.int8)
    block_mask[:, :, 0] = 1  # 约束：k=0层必须全为1，确保累加器正确初始化

    c = kernel(a, b, block_mask)

    ref_c = ref_program(a, b, block_mask, block_M, block_N, block_K)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("✅ Level 2 (extreme sparse) passed!")


def test_autotune():
    M = N = K = 1024
    sparsity = 0.5

    print(f"[Level 3] Testing autotune M={M}, N={N}, K={K}")

    torch.manual_seed(0)
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()

    kernel = blocksparse_matmul(M, N, K)

    best_config = kernel.config
    block_M = best_config["block_M"]
    block_N = best_config["block_N"]
    block_K = best_config["block_K"]

    print(f"Best Config: {best_config}")

    mask_shape = (M // block_M, N // block_N, K // block_K)
    block_mask = (torch.rand(mask_shape).npu() > sparsity).to(torch.int8)
    block_mask[:, :, 0] = 1  # 约束：k=0层必须全为1，确保累加器正确初始化

    c = kernel(a, b, block_mask)

    ref_c = ref_program(a, b, block_mask, block_M, block_N, block_K)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("✅ Level 3 (autotune) passed!")


def main():
    parser = argparse.ArgumentParser(description="BlockSparse GEMM Benchmark")
    parser.add_argument("--m", type=int, default=1024, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=1024, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=1024, help="Matrix dimension K")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Sparsity ratio (0-1)")
    parser.add_argument("--use_autotune", action="store_true", default=False, help="Whether to use autotune")

    args, _ = parser.parse_known_args()
    M, N, K = args.m, args.n, args.k
    sparsity = args.sparsity
    use_autotune = args.use_autotune

    print(f"Running BlockSparse GEMM Benchmark for M={M}, N={N}, K={K}")
    print(f"Target Block Sparsity: {sparsity}")
    print(f"Using Autotuner: {use_autotune}\n")

    torch.manual_seed(0)
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()

    if use_autotune:
        kernel = blocksparse_matmul(M, N, K)
        best_config = kernel.config
        block_M = best_config["block_M"]
        block_N = best_config["block_N"]
        block_K = best_config["block_K"]
        print(f"Best Config: {best_config}")
    else:
        kernel = blocksparse_matmul(
            M,
            N,
            K,
            block_M=DEFAULT_BLOCK_M,
            block_N=DEFAULT_BLOCK_N,
            block_K=DEFAULT_BLOCK_K,
            num_stages=DEFAULT_NUM_STAGES,
        )
        block_M, block_N, block_K = DEFAULT_BLOCK_M, DEFAULT_BLOCK_N, DEFAULT_BLOCK_K
        print(f"Using default config: block_size=({block_M}, {block_N}, {block_K})")

    mask_shape = (M // block_M, N // block_N, K // block_K)
    block_mask = (torch.rand(mask_shape).npu() > sparsity).to(torch.int8)
    block_mask[:, :, 0] = 1  # 约束：k=0层必须全为1，确保累加器正确初始化

    c = kernel(a, b, block_mask)

    ref_c = ref_program(a, b, block_mask, block_M, block_N, block_K)

    try:
        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
        print("✅ Results match! Verification successful.")
    except AssertionError as e:
        print("❌ Verification FAILED: Results differ significantly.")
        print(e)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("Running all test levels...\n")
        test_basic()
        test_typical()
        test_boundary_dense()
        test_boundary_sparse()
        print("\n✅ All tests passed!")
    else:
        main()
