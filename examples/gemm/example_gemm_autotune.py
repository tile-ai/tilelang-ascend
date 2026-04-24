import argparse

import tilelang
import tilelang.language as T
import torch
from tilelang.profiler import do_bench

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="GEMM Autotune Advanced for Ascend NPU")
parser.add_argument("--m", type=int, default=4096, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=4096, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=4096, help="Matrix K dimension")
parser.add_argument("--use_autotune", action="store_true", default=False, help="Whether to use autotune")
parser.add_argument("--use_pipeline", action="store_true", default=True, help="Whether to use pipeline optimization")
parser.add_argument("--use_swizzle", action="store_true", default=False, help="Whether to use swizzle optimization")
parser.add_argument("--profile_backend", type=str, default="event", help="Profiler backend")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k
USE_PIPELINE = args.use_pipeline
USE_SWIZZLE = args.use_swizzle

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def ref_program(A, B):
    """
    Compute the matrix product of A and the transpose of B.
    A: (M, K), B: (N, K) -> C: (M, N) = A @ B.T
    """
    return A @ B.T


def supply_prog(params):
    torch.manual_seed(0)
    return [
        torch.randn(M, K).half().npu(),
        torch.randn(N, K).half().npu(),
    ]


def get_configs():
    """
    Generate search space configurations for autotuning with advanced optimizations.
    """
    configs = []
    for block_M in [64, 128, 256]:
        for block_N in [64, 128, 256]:
            for block_K in [32, 64, 128]:
                for num_stages in [0, 2, 3]:
                    if block_M * block_N <= 256 * 256:
                        configs.append(
                            {
                                "block_M": block_M,
                                "block_N": block_N,
                                "block_K": block_K,
                                "num_stages": num_stages,
                            }
                        )
    return configs


def get_heuristic_config():
    """
    Get heuristic config based on problem size with pipeline optimization.
    """
    if M >= 4096 and N >= 4096 and K >= 4096:
        return {"block_M": 128, "block_N": 256, "block_K": 64, "num_stages": 3}
    elif M >= 2048 and N >= 2048:
        return {"block_M": 128, "block_N": 128, "block_K": 64, "num_stages": 2}
    else:
        return {"block_M": 64, "block_N": 64, "block_K": 32, "num_stages": 0}


@tilelang.autotune(
    configs=get_configs(),
    ref_prog=ref_program,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul(M, N, K, block_M, block_N, block_K, num_stages=0, dtype="float16", accum_dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            if USE_SWIZZLE:
                cid = T.use_swizzle(cid, M, N, K, block_M, block_N, off=1)

            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((block_M, block_K), dtype)
            B_L1 = T.alloc_shared((block_N, block_K), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, block_K)

            if USE_PIPELINE and num_stages > 0:
                for k in T.Pipelined(loop_k, num_stages=num_stages):
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    T.copy(B[by * block_N, k * block_K], B_L1)
                    T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k == 0))
            else:
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    T.copy(B[by * block_N, k * block_K], B_L1)
                    T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def main_func():
    if args.use_autotune:
        func = matmul(M, N, K)
        print("Best Config:", func.get_tuner_result())
    else:
        config = get_heuristic_config()
        func = matmul(M, N, K, **config)
        print("Using heuristic config:", config)
        print(f"Pipeline enabled: {USE_PIPELINE}, num_stages={config['num_stages']}")
        print(f"Swizzle enabled: {USE_SWIZZLE}")

    torch.manual_seed(0)
    a = torch.randn(M, K).half().npu()
    b = torch.randn(N, K).half().npu()
    c = torch.empty(M, N).half().npu()
    print("Input tensors initialized!")

    c = func(a, b)
    ref_c = ref_program(a, b)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")

    torch.npu.synchronize()

    tilelang_latency = do_bench(lambda: func(a, b))
    ref_latency = do_bench(lambda: ref_program(a, b))

    print(f"TileLang latency: {tilelang_latency:.4f} ms")
    print(f"Ref latency: {ref_latency:.4f} ms")
    print(f"TileLang TFlops: {2 * M * N * K / tilelang_latency * 1e-9:.2f}")
    print(f"Ref TFlops: {2 * M * N * K / ref_latency * 1e-9:.2f}")


if __name__ == "__main__":
    main_func()
