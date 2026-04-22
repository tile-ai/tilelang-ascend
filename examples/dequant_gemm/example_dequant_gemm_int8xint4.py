"""
INT8×INT4 Dequantize GEMM on TileLang-Ascend

设计思路：
- Host端：使用PyTorch完成INT4 → INT8 unpack（带符号扩展）
- NPU端：运行INT8×INT8→INT32 matmul（完全参考quant_matmul模式）

参考：
- examples/quant_batch_matmul/example_quant_matmul.py（已验证的Ascend INT8 matmul）
"""

import argparse
import torch
import tilelang as tl
import tilelang.language as T

tl.cache.clear_cache()


def unpack_int4_to_int8(B_packed: torch.Tensor) -> torch.Tensor:
    """
    将INT4 packed权重unpack为INT8（带符号扩展）。
    """
    N, K_compressed = B_packed.shape
    K = K_compressed * 2

    B = torch.zeros(N, K, dtype=torch.int8, device=B_packed.device)
    for j in range(K):
        shift = 4 * (j % 2)
        i4 = (B_packed[:, j // 2].to(torch.int32) >> shift) & 0xF
        # 符号扩展
        i4_signed = ((i4 << 28) >> 28)
        B[:, j] = i4_signed.to(torch.int8)

    return B


@tl.jit(out_idx=[-1])
def gemm_int8(M: int, N: int, K: int, block_M: int, block_N: int, block_K: int):
    """
    Ascend INT8 GEMM kernel。
    完全参考 examples/gemm/example_gemm.py 模式
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "int8"),      # type: ignore
        B: T.Tensor((K, N), "int8"),      # type: ignore
        C: T.Tensor((M, N), "int32"),     # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, block_K), "int8")
            B_L1 = T.alloc_L1((block_K, block_N), "int8")
            C_L0 = T.alloc_L0C((block_M, block_N), "int32")

            with T.Scope("C"):
                k_num = T.ceildiv(K, block_K)
                for k in T.serial(k_num):
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    T.copy(B[k * block_K, by * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def dequant_gemm_int8xint4(A: torch.Tensor, B_packed: torch.Tensor) -> torch.Tensor:
    """
    INT8×INT4 GEMM完整流程。
    """
    M, K = A.shape
    N, K_compressed = B_packed.shape
    K_check = K_compressed * 2

    assert K == K_check, f"K mismatch: A has K={K}, but B_packed implies K={K_check}"

    # Host端INT4 → INT8 unpack
    B_int8 = unpack_int4_to_int8(B_packed)

    # NPU端INT8×INT8 matmul
    # 确保维度能被block整除
    block_M = 128
    block_N = 256
    block_K = 64

    kernel = gemm_int8(M, N, K, block_M, block_N, block_K)

    # A: (M, K), B需要是(K, N)格式，B_int8是(N, K)，所以要传B_int8.T
    C = kernel(A.npu(), B_int8.T.contiguous().npu())
    return C


def ref_gemm_int8xint4(A: torch.Tensor, B_packed: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现"""
    B_int8 = unpack_int4_to_int8(B_packed)
    return torch.matmul(A.to(torch.float32), B_int8.T.to(torch.float32)).to(torch.int32)


def check_case(M: int, N: int, K: int):
    """验证测试"""
    K_compressed = K // 2

    torch.manual_seed(42)

    A = torch.randint(-128, 127, (M, K), dtype=torch.int8)
    B_packed = torch.randint(-8, 7, (N, K_compressed), dtype=torch.int8)

    C_npu = dequant_gemm_int8xint4(A, B_packed).cpu()
    C_ref = ref_gemm_int8xint4(A, B_packed)

    torch.testing.assert_close(C_npu, C_ref, atol=0, rtol=0)


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="INT8×INT4 Dequantize GEMM Example")
    parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
    parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}]", "Unknown args:", remains)

    torch.manual_seed(0)
    tl.cache.clear_cache()

    check_case(args.m, args.n, args.k)
    check_case(1024, 1024, 1024)
    check_case(2048, 2048, 2048)

    print("INT8×INT4 Dequantize GEMM example passed!")
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()