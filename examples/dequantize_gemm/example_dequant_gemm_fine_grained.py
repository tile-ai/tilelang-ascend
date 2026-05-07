import argparse

import tilelang as tl
import tilelang.language as T
import torch

tl.cache.clear_cache()


def torch_unpack_uint4_to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation: unpack UINT8 packed Unsigned INT4 data to FP16

    Args:
        tensor: torch.Tensor, shape (N, K//2), dtype uint8

    Returns:
        torch.Tensor, shape (N, K), dtype float16
    """
    assert tensor.dtype == torch.uint8
    N, K_packed = tensor.shape
    K = K_packed * 2

    result = torch.empty(N, K, dtype=torch.float16)
    for i in range(N):
        for j in range(K):
            val = tensor[i, j // 2].item()
            pos = j % 2
            u4 = (val >> (pos * 4)) & 0xF
            result[i, j] = float(u4)

    return result


def torch_unpack_uint4_to_int8(tensor: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation: unpack UINT8 packed Unsigned INT4 data to INT8

    Args:
        tensor: torch.Tensor, shape (N, K//2), dtype uint8

    Returns:
        torch.Tensor, shape (N, K), dtype int8
    """
    assert tensor.dtype == torch.uint8
    N, K_packed = tensor.shape
    K = K_packed * 2

    result = torch.empty(N, K, dtype=torch.int8)
    for i in range(N):
        for j in range(K):
            val = tensor[i, j // 2].item()
            pos = j % 2
            u4 = (val >> (pos * 4)) & 0xF
            result[i, j] = u4

    return result


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    },
)
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
    Accumulation: FP32 (must use FP32 accumulation for precision)

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


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    },
)
def int8_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
):
    """
    INT8 GEMM operator (Developer mode)

    Function: Execute matrix multiplication C = A @ B
    Computation: A: (M, K) INT8, B: (K, N) INT8, C: (M, N) INT32
    Accumulation: INT32

    Args:
        M, N, K: matrix dimensions
        block_M, block_N, block_K: block sizes
    """
    m_num = M // block_M
    n_num = N // block_N

    in_dtype = "int8"
    accum_dtype = "int32"

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((block_M, block_K), in_dtype)
            B_L1 = T.alloc_shared((block_K, block_N), in_dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            for k in T.serial(K // block_K):
                T.copy(A[bx * block_M, k * block_K], A_L1)
                T.copy(B[k * block_K, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def dequant_gemm_fp16_wrapper(
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
    Dequantize GEMM operator wrapper (FP16 mode)

    Function: Dequantize INT4 packed matrix to FP16, then execute matrix multiplication
    Computation: C = A @ B.T, where A: (M, K), B: (K, N), C: (M, N)

    Implementation: Python-side INT4->FP16 + NPU-side FP16xFP16 GEMM + FP32 computation precision

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
        B_fp16 = torch_unpack_uint4_to_fp16(B_packed)

        if input_dtype == "bfloat16":
            A_fp16 = A.to(torch.float16)
        else:
            A_fp16 = A

        B_fp16_T = B_fp16.T.contiguous()

        C_fp16 = kernel(A_fp16.npu(), B_fp16_T.npu())

        C_out = C_fp16.cpu().to(dtype_map[output_dtype])
        return C_out

    return wrapper


def dequant_gemm_int8_wrapper(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    output_dtype: str = "int32",
):
    """
    Dequantize GEMM operator wrapper (INT8 mode)

    Function: Dequantize INT4 packed matrix to INT8, then execute matrix multiplication
    Computation: C = A @ B.T, where A: (M, K), B: (K, N), C: (M, N)

    Implementation: Python-side INT4->INT8 + NPU-side INT8xINT8 GEMM + INT32 accumulation

    Args:
        M, N, K: matrix dimensions
        block_M, block_N, block_K: block sizes
        output_dtype: output data type (int32/float16)
    """
    kernel = int8_gemm(M, N, K, block_M, block_N, block_K)

    dtype_map = {
        "int32": torch.int32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def wrapper(A: torch.Tensor, B_packed: torch.Tensor):
        B_int8 = torch_unpack_uint4_to_int8(B_packed)

        B_int8_T = B_int8.T.contiguous()

        C_int32 = kernel(A.npu(), B_int8_T.npu())

        C_out = C_int32.cpu().to(dtype_map[output_dtype])
        return C_out

    return wrapper


def ref_program_fp16(A: torch.Tensor, B_packed: torch.Tensor, output_dtype: str):
    """
    PyTorch reference implementation (FP16 mode)

    Args:
        A: torch.Tensor, shape (M, K), dtype float16/bfloat16
        B_packed: torch.Tensor, shape (N, K//2), dtype uint8
        output_dtype: str, output type

    Returns:
        torch.Tensor, shape (M, N), dtype output_dtype
    """
    B_fp16 = torch_unpack_uint4_to_fp16(B_packed)

    A_fp32 = A.to(torch.float32)
    B_fp32 = B_fp16.to(torch.float32)
    C_fp32 = torch.matmul(A_fp32, B_fp32.T)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    C_out = C_fp32.to(dtype_map[output_dtype])
    return C_out


def ref_program_int8(A: torch.Tensor, B_packed: torch.Tensor, output_dtype: str):
    """
    PyTorch reference implementation (INT8 mode)

    Args:
        A: torch.Tensor, shape (M, K), dtype int8
        B_packed: torch.Tensor, shape (N, K//2), dtype uint8
        output_dtype: str, output type

    Returns:
        torch.Tensor, shape (M, N), dtype output_dtype
    """
    B_int8 = torch_unpack_uint4_to_int8(B_packed)

    A_int32 = A.to(torch.int32)
    B_int32 = B_int8.to(torch.int32)
    C_int32 = torch.matmul(A_int32, B_int32.T)

    dtype_map = {"int32": torch.int32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    C_out = C_int32.to(dtype_map[output_dtype])
    return C_out


def check_case_fp16(
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
    Test case execution function (FP16 mode)
    """
    assert K % 2 == 0, "K must be divisible by 2 (INT4 packed format)"

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    A = torch.randn(M, K, dtype=dtype_map[input_dtype])
    B_packed = torch.randint(0, 255, [N, K // 2], dtype=torch.uint8)

    wrapper = dequant_gemm_fp16_wrapper(M, N, K, block_M, block_N, block_K, input_dtype, output_dtype)
    C = wrapper(A, B_packed)

    ref_C = ref_program_fp16(A, B_packed, output_dtype)

    rtol = 1e-2 if output_dtype in ["float16", "bfloat16"] else 1e-4
    atol = 1e-2 if output_dtype in ["float16", "bfloat16"] else 1e-4

    torch.testing.assert_close(C, ref_C, rtol=rtol, atol=atol)
    print(f"FP16 Test passed: M={M}, N={N}, K={K}, input={input_dtype}, output={output_dtype}")


def check_case_int8(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    output_dtype: str = "int32",
):
    """
    Test case execution function (INT8 mode)
    """
    assert K % 2 == 0, "K must be divisible by 2 (INT4 packed format)"

    A = torch.randint(-128, 127, [M, K], dtype=torch.int8)
    B_packed = torch.randint(0, 255, [N, K // 2], dtype=torch.uint8)

    wrapper = dequant_gemm_int8_wrapper(M, N, K, block_M, block_N, block_K, output_dtype)
    C = wrapper(A, B_packed)

    ref_C = ref_program_int8(A, B_packed, output_dtype)

    if output_dtype == "int32":
        torch.testing.assert_close(C, ref_C, rtol=0, atol=0)
    else:
        torch.testing.assert_close(C, ref_C, rtol=1e-2, atol=1e-2)
    print(f"INT8 Test passed: M={M}, N={N}, K={K}, output={output_dtype}")


def check_unpack_functions():
    """
    Test INT4 unpack functions correctness
    """
    B_packed = torch.tensor([[0x12, 0x34]], dtype=torch.uint8)

    B_fp16 = torch_unpack_uint4_to_fp16(B_packed)
    assert B_fp16.shape == (1, 4), f"FP16 shape mismatch: {B_fp16.shape}"
    assert B_fp16.dtype == torch.float16, f"FP16 dtype mismatch: {B_fp16.dtype}"
    assert abs(B_fp16[0, 0].item() - 2.0) < 0.01, "FP16 value[0] mismatch"
    assert abs(B_fp16[0, 1].item() - 1.0) < 0.01, "FP16 value[1] mismatch"
    assert abs(B_fp16[0, 2].item() - 4.0) < 0.01, "FP16 value[2] mismatch"
    assert abs(B_fp16[0, 3].item() - 3.0) < 0.01, "FP16 value[3] mismatch"
    print("INT4 -> FP16 unpack test passed!")

    B_int8 = torch_unpack_uint4_to_int8(B_packed)
    assert B_int8.shape == (1, 4), f"INT8 shape mismatch: {B_int8.shape}"
    assert B_int8.dtype == torch.int8, f"INT8 dtype mismatch: {B_int8.dtype}"
    assert B_int8[0, 0].item() == 2, "INT8 value[0] mismatch"
    assert B_int8[0, 1].item() == 1, "INT8 value[1] mismatch"
    assert B_int8[0, 2].item() == 4, "INT8 value[2] mismatch"
    assert B_int8[0, 3].item() == 3, "INT8 value[3] mismatch"
    print("INT4 -> INT8 unpack test passed!")


def main(custom_args=None):
    """
    Main function: execute multiple test groups
    """
    parser = argparse.ArgumentParser(description="Dequantize GEMM Example (INT4 -> FP16/INT8)")
    parser.add_argument("--m", type=int, default=256, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=256, help="Matrix N dimension")
    parser.add_argument("--k", type=int, default=256, help="Matrix K dimension")
    parser.add_argument(
        "--input-dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "int8"],
        help="Input data type",
    )
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}] Unknown args:", remains)

    M, N, K = args.m, args.n, args.k

    tl.cache.clear_cache()
    torch.manual_seed(0)

    print("=" * 60)
    print("Dequantize GEMM Tests (Unsigned INT4)")
    print("=" * 60)

    print("\n[Unpack Function Tests]")
    check_unpack_functions()

    print("\n[FP16 Mode Tests - Level 0: Small Scale]")
    check_case_fp16(256, 256, 256, block_M=128, block_N=256, block_K=64, input_dtype="float16", output_dtype="float16")

    print("\n[FP16 Mode Tests - Level 1: Typical Scale]")
    check_case_fp16(512, 512, 512, block_M=128, block_N=256, block_K=64, input_dtype="float16", output_dtype="float16")
    check_case_fp16(1024, 1024, 1024, block_M=128, block_N=256, block_K=64, input_dtype="float16", output_dtype="float16")

    print("\n[FP16 Mode Tests - Level 1: BFloat16 Input]")
    check_case_fp16(512, 512, 512, block_M=128, block_N=256, block_K=64, input_dtype="bfloat16", output_dtype="bfloat16")

    # print("\n[FP16 Mode Tests - Level 3: Large Scale]")
    # check_case_fp16(4096, 4096, 4096, block_M=128, block_N=256, block_K=64, input_dtype="float16", output_dtype="float16")

    print("\n[FP16 Mode Tests - Custom Dimensions]")
    check_case_fp16(
        M,
        N,
        K,
        block_M=128,
        block_N=256,
        block_K=64,
        input_dtype=args.input_dtype if args.input_dtype != "int8" else "float16",
        output_dtype="float16",
    )

    print("\n[INT8 Mode Tests - Level 0: Small Scale]")
    check_case_int8(256, 256, 256, block_M=128, block_N=128, block_K=64, output_dtype="int32")

    print("\n[INT8 Mode Tests - Level 1: Typical Scale]")
    check_case_int8(512, 512, 512, block_M=128, block_N=128, block_K=64, output_dtype="int32")
    check_case_int8(1024, 1024, 1024, block_M=128, block_N=128, block_K=64, output_dtype="int32")

    print("\n[INT8 Mode Tests - Level 1: Output to Float16]")
    check_case_int8(512, 512, 512, block_M=128, block_N=128, block_K=64, output_dtype="float16")

    print("\n[INT8 Mode Tests - Custom Dimensions]")
    check_case_int8(M, N, K, block_M=128, block_N=128, block_K=64, output_dtype="int32")

    print("=" * 60)
    print("All tests passed!")
    print("Kernel Output Match!")
    print("=" * 60)


if __name__ == "__main__":
    main()
