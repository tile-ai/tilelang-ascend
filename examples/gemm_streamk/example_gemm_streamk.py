"""
StreamK GEMM on TileLang-Ascend

设计思路：
- 使用T.Persistent进行tile级负载均衡（替代GPU的while动态调度）
- 使用T.Pipelined进行K维度流水线优化
- Ascend不支持atomic_add，简化为每个tile由单一core完整计算

参考：
- tilelang/examples/gemm_streamk/example_tilelang_gemm_streamk.py（GPU StreamK）
- examples/gemm/example_gemm_persistent.py（Ascend Persistent模式）
- examples/pipeline/gemm_v0_pipeline.py（Ascend Pipeline模式）

Ascend限制：
1. 不支持T.atomic_add
2. 不支持K维度分割（partial tiles）
3. 使用T.Persistent替代动态while循环
"""

import argparse
import math
import torch
import tilelang as tl
import tilelang.language as T

tl.cache.clear_cache()


def cdiv(a, b):
    return math.ceil(a / b)


@tl.jit(out_idx=[-1])
def gemm_streamk(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    core_num: int,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),   # type: ignore
        B: T.Tensor((K, N), dtype),   # type: ignore
        C: T.Tensor((M, N), dtype),   # type: ignore
    ):
        # Kernel使用tiles总数，Persistent内部做负载均衡
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, _):
            # L1 buffer for A and B tiles
            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)

            # L0C buffer for accumulator (Cube output)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            # Persistent循环：每个core处理多个tiles实现负载均衡
            with T.Scope("C"):
                for bx, by in T.Persistent([T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
                    core_num, cid):
                    # K维度迭代
                    loop_k = T.ceildiv(K, block_K)
                    for k in T.serial(loop_k):
                        T.copy(A[bx * block_M, k * block_K], A_L1)
                        T.copy(B[k * block_K, by * block_N], B_L1)

                        T.barrier_all()
                        T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                        T.barrier_all()

                    # 写回结果到Global Memory
                    T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def ref_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现"""
    return torch.matmul(A, B)


def check_case(M: int, N: int, K: int, block_M: int, block_N: int, block_K: int, core_num: int):
    """验证测试"""
    torch.manual_seed(42)

    A = torch.randn(M, K, dtype=torch.float16).npu()
    B = torch.randn(K, N, dtype=torch.float16).npu()

    kernel = gemm_streamk(M, N, K, block_M, block_N, block_K, core_num)

    C_npu = kernel(A, B)
    C_ref = ref_matmul(A.cpu(), B.cpu()).npu()

    torch.testing.assert_close(C_npu.cpu(), C_ref.cpu(), rtol=1e-2, atol=1e-2)


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="StreamK GEMM Example")
    parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
    parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
    parser.add_argument("--block_m", type=int, default=128, help="Block M size")
    parser.add_argument("--block_n", type=int, default=256, help="Block N size")
    parser.add_argument("--block_k", type=int, default=64, help="Block K size")
    parser.add_argument("--core_num", type=int, default=24, help="Number of cores")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}]", "Unknown args:", remains)

    torch.manual_seed(0)
    tl.cache.clear_cache()

    check_case(args.m, args.n, args.k, args.block_m, args.block_n, args.block_k, args.core_num)
    check_case(512, 512, 512, 64, 128, 32, 8)
    check_case(2048, 2048, 2048, 128, 256, 64, 24)

    print("StreamK GEMM example passed!")
    print("Kernel Output Match!")

    return True


if __name__ == "__main__":
    main()