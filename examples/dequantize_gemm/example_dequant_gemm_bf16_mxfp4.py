import argparse
import numpy as np
import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()


def mxfp4_to_fp16_bits(val: int, pos: int) -> int:
    """
    MXFP4 to FP16 bit conversion

    MXFP4 format: s1e2m1 (sign bit + 2-bit exponent + 1-bit mantissa)
    FP16 format: s1e5m10 (sign bit + 5-bit exponent + 10-bit mantissa)

    Args:
        val: uint8 value containing 2 packed MXFP4 values
        pos: position index (0 or 1)

    Returns:
        uint16 value representing FP16 bit pattern
    """
    f4 = (val >> (pos * 4)) & 0xF

    s = f4 >> 3  # Sign bit (bit 3)
    e_f4 = (f4 >> 1) & 0x3  # Exponent bits (bits 1-2)
    m_f4 = f4 & 1  # Mantissa bit (bit 0)

    # FP16 exponent: e_f4 bias is 1, FP16 bias is 15
    # e_f4 - 1 = e_fp16 - 15, so e_fp16 = e_f4 + 14
    e_fp16 = e_f4 + 14

    # FP16 mantissa: extend MXFP4's 1-bit mantissa to 10 bits
    m_fp16 = m_f4 << 9

    fp16_bits = (s << 15) | (e_fp16 << 10) | m_fp16
    return fp16_bits


def torch_unpack_mxfp4_to_fp16(B_packed: torch.Tensor, Scale: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation: MXFP4 to FP16 dequantization + scale application

    Args:
        B_packed: (N, K // 2) uint8 (packed MXFP4)
        Scale: (N, K // 32) uint8 (per-block scale factor)

    Returns:
        (N, K) float16
    """
    assert B_packed.dtype == torch.uint8
    assert Scale.dtype == torch.uint8

    N, K_packed = B_packed.shape
    K = K_packed * 2
    scale_size = 32

    result_bits = np.empty((N, K), dtype=np.uint16)
    B_packed_np = B_packed.numpy()
    Scale_np = Scale.numpy()

    for i in range(N):
        for j in range(K):
            val = B_packed_np[i, j // 2]
            pos = j % 2
            fp16_bits = mxfp4_to_fp16_bits(val, pos)

            # Apply scale
            scale_idx = j // scale_size
            scale_val = Scale_np[i, scale_idx]

            # FP16 value
            fp16_view = np.frombuffer(fp16_bits.astype(np.uint16).tobytes(), dtype=np.float16)
            fp16_value = fp16_view[0]

            # Scale: 2^(scale_val - 127), MXFP4 specification
            scale_factor = 2.0 ** (scale_val - 127)
            result_value = fp16_value * scale_factor

            # Convert back to uint16 bits
            result_bits[i, j] = np.frombuffer(result_value.astype(np.float16).tobytes(), dtype=np.uint16)[0]

    result = torch.from_numpy(result_bits.view(np.float16)).to(torch.float16)
    return result


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def fp16_gemm_no_bias(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    FP16 GEMM without Bias (Developer mode)

    Function: Perform matrix multiplication C = A @ B
    Computation: A: (M, K) FP16, B: (K, N) FP16, C: (M, N) FP16
    Accumulation: FP32

    Args:
        M, N, K: Matrix dimensions
        block_M, block_N, block_K: Block sizes
    """
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((block_M, block_K), dtype)
            B_L1 = T.alloc_shared((block_K, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, block_K)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * block_K], A_L1)
                T.copy(B[k * block_K, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


@tilelang.jit(out_idx=[2], workspace_idx=4, pass_configs=pass_configs)
def fp16_gemm_with_bias(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    FP16 GEMM with Bias (Developer mode)

    Function: Perform matrix multiplication C = A @ B + Bias
    Computation: A: (M, K) FP16, B: (K, N) FP16, C: (M, N) FP16, Bias: (M, N) FP16
    Accumulation: FP32

    Args:
        M, N, K: Matrix dimensions
        block_M, block_N, block_K: Block sizes
    """
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
        Bias: T.Tensor((M, N), dtype),
        workspace: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((block_M, block_K), dtype)
            B_L1 = T.alloc_shared((block_K, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            bias_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            loop_k = T.ceildiv(K, block_K)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * block_K], A_L1)
                T.copy(B[k * block_K, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, workspace[bx * block_M, by * block_N])

            T.copy(workspace[bx * block_M + vid * block_M // VEC_NUM, by * block_N], c_ub)
            T.copy(Bias[bx * block_M + vid * block_M // VEC_NUM, by * block_N], bias_ub)

            T.tile.add(c_ub, c_ub, bias_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def dequant_gemm_mxfp4_wrapper(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    input_dtype: str = "bfloat16",
    output_dtype: str = "bfloat16",
    with_bias: bool = False,
):
    """
    Dequantize GEMM (MXFP4) operator wrapper

    Function: Dequantize MXFP4 packed matrix to FP16, then perform matrix multiplication
    Computation: C = A @ B.T + Bias (optional)

    Implementation:
        1. Python-side MXFP4 to FP16 dequantization + scale application
        2. BF16 to FP16 conversion (input A)
        3. NPU-side FP16 x FP16 GEMM
        4. FP16 to BF16 output

    Args:
        M, N, K: Matrix dimensions
        block_M, block_N, block_K: Block sizes
        input_dtype: Input data type (bfloat16)
        output_dtype: Output data type (bfloat16)
        with_bias: Whether to include Bias
    """
    if with_bias:
        kernel = fp16_gemm_with_bias(M, N, K, block_M, block_N, block_K)
    else:
        kernel = fp16_gemm_no_bias(M, N, K, block_M, block_N, block_K)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    def wrapper(A: torch.Tensor, B_packed: torch.Tensor, Scale: torch.Tensor, Bias: torch.Tensor = None):
        # 1. MXFP4 to FP16 dequantization + scale application
        B_fp16 = torch_unpack_mxfp4_to_fp16(B_packed, Scale)  # (N, K)

        # 2. Convert input A to FP16
        A_fp16 = A.to(torch.float16)  # (M, K)

        # 3. Transpose B to (K, N)
        B_fp16_T = B_fp16.T.contiguous()  # (K, N)

        # 4. Call different kernel based on with_bias
        if with_bias and Bias is not None:
            Bias_fp16 = Bias.to(torch.float16)  # (M, N)
            workspace_fp16 = torch.zeros(M, N, dtype=torch.float16).npu()
            C_fp16 = kernel(A_fp16.npu(), B_fp16_T.npu(), Bias_fp16.npu(), workspace_fp16)
        else:
            C_fp16 = kernel(A_fp16.npu(), B_fp16_T.npu())

        # 5. FP16 to output type
        C_out = C_fp16.cpu().to(dtype_map[output_dtype])
        return C_out

    return wrapper


def ref_program(A: torch.Tensor, B_packed: torch.Tensor, Scale: torch.Tensor, Bias: torch.Tensor = None, output_dtype: str = "bfloat16"):
    """
    PyTorch reference implementation

    Args:
        A: torch.Tensor, shape (M, K), dtype bfloat16
        B_packed: torch.Tensor, shape (N, K//2), dtype uint8
        Scale: torch.Tensor, shape (N, K//32), dtype uint8
        Bias: torch.Tensor, shape (M, N), dtype bfloat16 (optional)
        output_dtype: str, output data type

    Returns:
        torch.Tensor, shape (M, N), dtype output_dtype
    """
    # MXFP4 to FP16 + Scale
    B_fp16 = torch_unpack_mxfp4_to_fp16(B_packed, Scale)

    # Matrix multiplication (using FP32 computation precision)
    A_fp32 = A.to(torch.float32)
    B_fp32 = B_fp16.to(torch.float32)
    C_fp32 = torch.matmul(A_fp32, B_fp32.T)

    # Bias addition (optional)
    if Bias is not None:
        C_fp32 = C_fp32 + Bias.to(torch.float32)

    # FP32 to output type
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    C_out = C_fp32.to(dtype_map[output_dtype])
    return C_out


def check_case(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    input_dtype: str = "bfloat16",
    output_dtype: str = "bfloat16",
    with_bias: bool = False,
):
    """
    Test case execution function
    """
    assert K % 2 == 0, "K must be divisible by 2 (MXFP4 packed format)"
    assert K % 32 == 0, "K must be divisible by 32 (scale_size=32)"
    assert N % 32 == 0, "N must be divisible by 32 (scale_size=32)"

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "uint8": torch.uint8,
    }

    # Generate test data
    A = torch.randn(M, K, dtype=dtype_map[input_dtype])
    B_packed = torch.randint(0, 255, [N, K // 2], dtype=torch.uint8)
    # Scale value range limited to 120-134 (corresponding scale factor: 2^-7 ~ 2^7) to avoid FP16 overflow
    Scale = torch.randint(120, 135, [N, K // 32], dtype=torch.uint8)

    if with_bias:
        Bias = torch.randn(M, N, dtype=dtype_map[input_dtype])
    else:
        Bias = None

    # Execute kernel
    wrapper = dequant_gemm_mxfp4_wrapper(M, N, K, block_M, block_N, block_K, input_dtype, output_dtype, with_bias)
    C = wrapper(A, B_packed, Scale, Bias)

    # Reference implementation
    ref_C = ref_program(A, B_packed, Scale, Bias, output_dtype)

    # Accuracy check
    rtol = 1e-2 if output_dtype in ["float16", "bfloat16"] else 1e-4
    atol = 1e-2 if output_dtype in ["float16", "bfloat16"] else 1e-4

    torch.testing.assert_close(C, ref_C, rtol=rtol, atol=atol)

    bias_str = "with_bias" if with_bias else "no_bias"
    print(f"✅ Test passed: M={M}, N={N}, K={K}, input={input_dtype}, output={output_dtype}, {bias_str}")


def main(custom_args=None):
    """
    Main function: Execute multiple test cases
    """
    parser = argparse.ArgumentParser(description="Dequantize GEMM Example (MXFP4 to FP16)")
    parser.add_argument("--m", type=int, default=256, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=256, help="Matrix N dimension")
    parser.add_argument("--k", type=int, default=256, help="Matrix K dimension")
    parser.add_argument("--input-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="Input data type")
    parser.add_argument("--output-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="Output data type")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}] Unknown args:", remains)

    M, N, K = args.m, args.n, args.k

    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    print("=" * 80)
    print("Dequantize GEMM Tests (MXFP4 to FP16)")
    print("=" * 80)

    # Level 0: Small scale test (without Bias)
    print("\n--- Level 0: Basic Small ---")
    check_case(
        128, 128, 128, block_M=128, block_N=128, block_K=128, input_dtype=args.input_dtype, output_dtype=args.output_dtype, with_bias=False
    )

    # Level 0: Small scale test (with Bias)
    print("\n--- Level 0: Basic with Bias ---")
    check_case(
        128, 128, 128, block_M=128, block_N=128, block_K=128, input_dtype=args.input_dtype, output_dtype=args.output_dtype, with_bias=True
    )

    # Level 1: Typical scale test (without Bias)
    print("\n--- Level 1: Typical 256 ---")
    check_case(
        256, 256, 256, block_M=128, block_N=128, block_K=128, input_dtype=args.input_dtype, output_dtype=args.output_dtype, with_bias=False
    )

    # Level 1: Typical scale test (with Bias)
    print("\n--- Level 1: Typical 256 with Bias ---")
    check_case(
        256, 256, 256, block_M=128, block_N=128, block_K=128, input_dtype=args.input_dtype, output_dtype=args.output_dtype, with_bias=True
    )

    # Level 1: Medium scale test (with Bias)
    print("\n--- Level 1: Typical 512 with Bias ---")
    check_case(
        512, 512, 512, block_M=128, block_N=128, block_K=128, input_dtype=args.input_dtype, output_dtype=args.output_dtype, with_bias=True
    )

    # Level 3: Large scale test (without Bias)
    print("\n--- Level 3: Large 1024 ---")
    check_case(
        M, N, K, block_M=128, block_N=128, block_K=128, input_dtype=args.input_dtype, output_dtype=args.output_dtype, with_bias=False
    )

    print("=" * 80)
    print("✅ All tests passed!")
    print("✅ Kernel Output Match!")
    print("=" * 80)


if __name__ == "__main__":
    main()
