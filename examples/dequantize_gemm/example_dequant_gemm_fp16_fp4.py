import argparse

import numpy as np
import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()


def fp4_to_fp16_bits(val: int, pos: int) -> int:
    """
    FP4 to FP16 bit conversion

    FP4 format: s1e2m1 (sign bit + 2-bit exponent + 1-bit mantissa)
    FP16 format: s1e5m10 (sign bit + 5-bit exponent + 10-bit mantissa)

    Args:
        val: uint8 value containing 2 packed FP4 values
        pos: position index (0 or 1)

    Returns:
        uint16 value representing FP16 bit pattern
    """
    f4 = (val >> (pos * 4)) & 0xF

    s = f4 >> 3  # sign bit (bit 3)
    e_f4 = (f4 >> 1) & 0x3  # exponent bits (bits 1-2)
    m_f4 = f4 & 1  # mantissa bit (bit 0)

    # FP16 exponent: FP4 bias is 1, FP16 bias is 15
    # e_fp4 - 1 = e_fp16 - 15, so e_fp16 = e_fp4 + 14
    e_fp16 = e_f4 + 14

    # FP16 mantissa: expand FP4's 1-bit mantissa to 10-bit
    m_fp16 = m_f4 << 9

    fp16_bits = (s << 15) | (e_fp16 << 10) | m_fp16
    return fp16_bits


def torch_unpack_fp4_to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation: unpack UINT8 packed FP4 data to FP16

    Args:
        tensor: torch.Tensor, shape (N, K//2), dtype uint8

    Returns:
        torch.Tensor, shape (N, K), dtype float16
    """
    assert tensor.dtype == torch.uint8
    N, K_packed = tensor.shape
    K = K_packed * 2

    # First convert to numpy uint16 bits
    result_bits = np.empty((N, K), dtype=np.uint16)

    tensor_np = tensor.numpy()
    for i in range(N):
        for j in range(K):
            val = tensor_np[i, j // 2]
            pos = j % 2
            result_bits[i, j] = fp4_to_fp16_bits(val, pos)

    # Convert uint16 bits to float16 values
    result = torch.from_numpy(result_bits.view(np.float16)).to(torch.float16)
    return result


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def fp16_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
):
    """
    FP16 GEMM operator (Developer mode)

    Function: Execute matrix multiplication C = A @ B
    Computation: A: (M, K) FP16, B: (K, N) FP16, C: (M, N) FP16
    Accumulation: FP32

    Args:
        M, N, K: matrix dimensions
        block_M, block_N, block_K: block sizes
    """
    m_num = M // block_M
    n_num = N // block_N

    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((block_M, block_K), dtype)
            B_L1 = T.alloc_shared((block_K, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            for k in T.serial(K // block_K):
                T.copy(A[bx * block_M, k * block_K], A_L1)
                T.copy(B[k * block_K, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def dequant_gemm_fp16_fp4_wrapper(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    input_dtype: str = "float16",
    output_dtype: str = "float16",
):
    """
    Dequantize GEMM operator wrapper (FP16-FP4)

    Function: Dequantize FP4 packed matrix to FP16, then execute matrix multiplication
    Computation: C = A @ B.T, where A: (M, K), B: (K, N), C: (M, N)

    Implementation: Python-side FP4->FP16 + NPU-side FP16xFP16 GEMM + FP32 computation precision

    Args:
        M, N, K: matrix dimensions
        block_M, block_N, block_K: block sizes
        input_dtype: input data type (float16)
        output_dtype: output data type (float16)
    """
    kernel = fp16_gemm(M, N, K, block_M, block_N, block_K)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    def wrapper(A: torch.Tensor, B_packed: torch.Tensor):
        # 1. FP4 -> FP16 dequantization
        B_fp16 = torch_unpack_fp4_to_fp16(B_packed)

        # 2. Convert input A to FP16
        if input_dtype != "float16":
            A_fp16 = A.to(torch.float16)
        else:
            A_fp16 = A

        # 3. Transpose B to (K, N)
        B_fp16_T = B_fp16.T.contiguous()

        # 4. NPU-side FP16xFP16 matrix multiplication, output FP16 (internal FP32 precision)
        C_fp16 = kernel(A_fp16.npu(), B_fp16_T.npu())

        # 5. FP16 -> output dtype
        C_out = C_fp16.cpu().to(dtype_map[output_dtype])
        return C_out

    return wrapper


def ref_program(A: torch.Tensor, B_packed: torch.Tensor, output_dtype: str):
    """
    PyTorch reference implementation

    Args:
        A: torch.Tensor, shape (M, K), dtype float16
        B_packed: torch.Tensor, shape (N, K//2), dtype uint8
        output_dtype: str, output type

    Returns:
        torch.Tensor, shape (M, N), dtype output_dtype
    """
    # FP4 -> FP16
    B_fp16 = torch_unpack_fp4_to_fp16(B_packed)

    # Matrix multiplication (using FP32 computation precision)
    A_fp32 = A.to(torch.float32)
    B_fp32 = B_fp16.to(torch.float32)
    C_fp32 = torch.matmul(A_fp32, B_fp32.T)

    # FP32 -> output type
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    C_out = C_fp32.to(dtype_map[output_dtype])
    return C_out


def check_case(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    input_dtype: str = "float16",
    output_dtype: str = "float16",
):
    """
    Test case execution function
    """
    assert K % 2 == 0, "K must be divisible by 2 (FP4 packed format)"

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    # Generate test data
    A = torch.randn(M, K, dtype=dtype_map[input_dtype])
    B_packed = torch.randint(0, 255, [N, K // 2], dtype=torch.uint8)

    # Execute kernel
    wrapper = dequant_gemm_fp16_fp4_wrapper(M, N, K, block_M, block_N, block_K, input_dtype, output_dtype)
    C = wrapper(A, B_packed)

    # Reference implementation
    ref_C = ref_program(A, B_packed, output_dtype)

    # Precision check
    rtol = 1e-2 if output_dtype in ["float16", "bfloat16"] else 1e-4
    atol = 1e-2 if output_dtype in ["float16", "bfloat16"] else 1e-4

    torch.testing.assert_close(C, ref_C, rtol=rtol, atol=atol)
    print(f"Test passed: M={M}, N={N}, K={K}, input={input_dtype}, output={output_dtype}")


def main(custom_args=None):
    """
    Main function: execute multiple test groups
    """
    parser = argparse.ArgumentParser(description="Dequantize GEMM Example (FP16-FP4)")
    parser.add_argument("--m", type=int, default=256, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=256, help="Matrix N dimension")
    parser.add_argument("--k", type=int, default=256, help="Matrix K dimension")
    parser.add_argument(
        "--input-dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Input data type",
    )
    parser.add_argument(
        "--output-dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Output data type",
    )
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}] Unknown args:", remains)

    M, N, K = args.m, args.n, args.k

    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    print("=" * 60)
    print("Dequantize GEMM Tests (FP16-FP4)")
    print("=" * 60)

    print("\n--- Level 0: Basic Small ---")
    check_case(128, 128, 128, block_M=128, block_N=128, block_K=128, input_dtype=args.input_dtype, output_dtype=args.output_dtype)

    print("\n--- Level 1: Typical 256 ---")
    check_case(256, 256, 256, block_M=128, block_N=128, block_K=128, input_dtype=args.input_dtype, output_dtype=args.output_dtype)

    print("\n--- Level 1: Typical 512 ---")
    check_case(512, 512, 512, block_M=128, block_N=128, block_K=128, input_dtype=args.input_dtype, output_dtype=args.output_dtype)

    print("\n--- Level 3: Large ---")
    check_case(M, N, K, block_M=128, block_N=128, block_K=128, input_dtype=args.input_dtype, output_dtype=args.output_dtype)

    print("=" * 60)
    print("All tests passed!")
    print("Kernel Output Match!")
    print("=" * 60)


if __name__ == "__main__":
    main()
