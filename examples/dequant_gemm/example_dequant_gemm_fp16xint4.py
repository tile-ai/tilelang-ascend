"""
INT4 Dequantize GEMV on TileLang-Ascend

设计思路：
- Host端：使用PyTorch完成INT4 → FP16 unpack（避开Ascend不支持的操作）
- NPU端：运行标准FP16×FP16 GEMV（使用已验证的gemv_c模式）

参考：
- tilelang/examples/gemv/example_gemv_c.py（Ascend GEMV基础模式）
- examples/quant_batch_matmul/example_quant_batch_matmul.py（Ascend量化matmul模式）
"""

import argparse
import torch
import tilelang as tl
import tilelang.language as T

tl.cache.clear_cache()


def unpack_int4_to_fp16(B_packed: torch.Tensor) -> torch.Tensor:
    """
    将INT4 packed权重unpack为FP16。

    Args:
        B_packed: (N, K//2) int8 tensor，每个byte存储2个INT4

    Returns:
        B: (N, K) float16 tensor
    """
    N, K_compressed = B_packed.shape
    K = K_compressed * 2

    B = torch.zeros(N, K, dtype=torch.float16, device=B_packed.device)
    for j in range(K):
        shift = 4 * (j % 2)
        B[:, j] = ((B_packed[:, j // 2].int() >> shift) & 0xF).half()

    return B


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    }
)
def gemv_fp16(N: int, K: int, block_N: int, block_K: int, dtype="float16", accum_dtype="float32"):
    """
    Ascend标准FP16 GEMV kernel。
    参考 examples/gemv/example_gemv_c.py
    """
    FRACTAL_SIZE = 16

    n_num = T.ceildiv(N, block_N)
    k_num = T.ceildiv(K, block_K)

    @T.prim_func
    def main(
        A: T.Tensor((1, K), dtype),    # type: ignore
        B: T.Tensor((N, K), dtype),    # type: ignore
        C: T.Tensor((1, N), dtype),    # type: ignore
    ):
        with T.Kernel(n_num, is_npu=True) as (bn_idx, _):
            A_L1 = T.alloc_L1((FRACTAL_SIZE, block_K), dtype)
            B_L1 = T.alloc_L1((block_N, block_K), dtype)
            C_L0 = T.alloc_L0C((FRACTAL_SIZE, block_N), accum_dtype)

            for bk in T.serial(k_num):
                T.copy(A[0, bk * block_K], A_L1)
                T.copy(B[bn_idx * block_N, bk * block_K], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(bk == 0))

            T.copy(C_L0, C[0, bn_idx * block_N])

    return main


def dequant_gemv_fp16xint4(A: torch.Tensor, B_packed: torch.Tensor) -> torch.Tensor:
    """
    INT4 Dequantize GEMV完整流程。

    Args:
        A: (1, K) float16 输入向量
        B_packed: (N, K//2) int8 packed权重

    Returns:
        C: (1, N) float16 输出向量
    """
    N, K_compressed = B_packed.shape
    K = K_compressed * 2

    # Step 1: Host端INT4 → FP16 unpack
    B_fp16 = unpack_int4_to_fp16(B_packed).npu()

    # Step 2: NPU端标准GEMV
    block_N = 128
    block_K = 128
    kernel = gemv_fp16(N, K, block_N, block_K)

    C = kernel(A.npu(), B_fp16)

    return C


def ref_dequant_gemv(A: torch.Tensor, B_packed: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现"""
    B_fp16 = unpack_int4_to_fp16(B_packed)
    return torch.matmul(A, B_fp16.T).half()


def check_case(N: int, K: int):
    """验证测试"""
    K_compressed = K // 2

    torch.manual_seed(42)

    A = torch.randn(1, K, dtype=torch.float16)
    B_packed = torch.randint(0, 127, (N, K_compressed), dtype=torch.int8)

    C_npu = dequant_gemv_fp16xint4(A, B_packed).cpu()
    C_ref = ref_dequant_gemv(A, B_packed)

    torch.testing.assert_close(C_npu, C_ref, atol=1e-3, rtol=1e-3)


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="FP16×INT4 Dequantize GEMV Example")
    parser.add_argument("--n", type=int, default=1024, help="Output dimension N")
    parser.add_argument("--k", type=int, default=1024, help="Input dimension K")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}]", "Unknown args:", remains)

    torch.manual_seed(0)
    tl.cache.clear_cache()

    check_case(args.n, args.k)
    check_case(512, 512)
    check_case(4096, 4096)

    print("FP16×INT4 Dequantize GEMV example passed!")
    print("Kernel Output Match!")

    return True


if __name__ == "__main__":
    main()