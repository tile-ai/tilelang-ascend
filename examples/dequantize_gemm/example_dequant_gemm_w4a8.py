import argparse

import tilelang as tl
import tilelang.language as T
import torch

tl.cache.clear_cache()


def torch_unpack_int4(tensor: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """
    PyTorch reference implementation: Unpack UINT8 packed INT4 data to INT8 (sign extension)

    Args:
        tensor: torch.Tensor, shape (N, K//num_elems_per_byte), dtype uint8
        num_bits: int, number of bits per element (default 4)

    Returns:
        torch.Tensor, shape (N, K), dtype int8

    INT4 encoding range: [-8, 7]
    Sign extension method:
        nibble >= 8: nibble - 16 (negative)
        nibble < 8:  nibble (positive)
    """
    assert tensor.dtype == torch.uint8
    num_elems_per_byte = 8 // num_bits
    N, K_packed = tensor.shape
    K = K_packed * num_elems_per_byte

    result = torch.empty(N, K, dtype=torch.int8)
    for i in range(N):
        for j in range(K):
            val = tensor[i, j // num_elems_per_byte].item()
            pos = j % num_elems_per_byte
            nibble = (val >> (pos * num_bits)) & 0xF
            if nibble >= 8:
                nibble -= 16
            result[i, j] = nibble

    return result


pass_configs = {
    tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


@tl.jit(out_idx=[2], pass_configs=pass_configs)
def dequant_gemm_w4a8(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    in_dtype: str = "int8",
    accum_dtype: str = "int32",
):
    """
    Dequantize GEMM (W4A8) NPU Kernel

    Function: Performs matrix multiplication C = B @ A.T
    Computation: B: (N, K) INT8, A: (M, K) INT8, C: (N, M) INT32
    Accumulation: INT32

    Note: INT4 unpacking is done on Python side, NPU only performs INT8 GEMM

    Args:
        M, N, K: Matrix dimensions
        block_M, block_N, block_K: Block sizes
        in_dtype: Input data type (int8)
        accum_dtype: Accumulation data type (int32)
    """
    assert M % block_M == 0, f"M ({M}) must be divisible by block_M ({block_M})"
    assert N % block_N == 0, f"N ({N}) must be divisible by block_N ({block_N})"
    assert K % block_K == 0, f"K ({K}) must be divisible by block_K ({block_K})"

    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        C: T.Tensor((N, M), accum_dtype),
    ):
        with T.Kernel(n_num * m_num, is_npu=True) as (cid, _):
            bx = cid // m_num
            by = cid % m_num

            A_L1 = T.alloc_L1((block_M, block_K), in_dtype)
            B_L1 = T.alloc_L1((block_N, block_K), in_dtype)
            C_L0 = T.alloc_L0C((block_N, block_M), accum_dtype)

            with T.Scope("C"):
                for k in T.serial(K // block_K):
                    T.copy(A[by * block_M, k * block_K], A_L1)
                    T.copy(B[bx * block_N, k * block_K], B_L1)

                    T.barrier_all()
                    T.gemm_v0(B_L1, A_L1, C_L0, transpose_B=True, init=(k == 0))
                    T.barrier_all()

                T.copy(C_L0, C[bx * block_N, by * block_M])

    return main


def ref_program(A: torch.Tensor, B_packed: torch.Tensor, out_dtype: str = "float16"):
    """
    PyTorch reference implementation

    Args:
        A: torch.Tensor, shape (M, K), dtype int8
        B_packed: torch.Tensor, shape (N, K//2), dtype uint8 (packed INT4)
        out_dtype: str, output type

    Returns:
        torch.Tensor, shape (N, M), dtype out_dtype
    """
    B = torch_unpack_int4(B_packed, num_bits=4)
    C_int = torch.matmul(B.to(torch.int32), A.T.to(torch.int32))
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    C_out = C_int.to(dtype_map[out_dtype])
    return C_out


def check_case(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    in_dtype: str = "int8",
    accum_dtype: str = "int32",
    out_dtype: str = "float16",
):
    """
    Test case execution function
    """
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int32": torch.int32,
        "uint8": torch.uint8,
    }

    assert K % 2 == 0, "K must be divisible by 2 (INT4 packed format)"
    assert M % block_M == 0, f"M ({M}) must be divisible by block_M ({block_M})"
    assert N % block_N == 0, f"N ({N}) must be divisible by block_N ({block_N})"
    assert K % block_K == 0, f"K ({K}) must be divisible by block_K ({block_K})"

    A = torch.randint(-128, 127, [M, K], dtype=dtype_map[in_dtype])
    B_packed = torch.randint(0, 255, [N, K // 2], dtype=torch.uint8)

    B = torch_unpack_int4(B_packed, num_bits=4)

    kernel = dequant_gemm_w4a8(M, N, K, block_M, block_N, block_K, in_dtype, accum_dtype)

    C_int = kernel(A.npu(), B.npu())
    ref_C = ref_program(A, B_packed, out_dtype)

    rtol = 1e-2
    atol = 1e-2
    torch.testing.assert_close(C_int.cpu().to(dtype_map[out_dtype]), ref_C, rtol=rtol, atol=atol)
    print(f"Test passed: M={M}, N={N}, K={K}, dtype={in_dtype}->{out_dtype}")


def main(custom_args=None):
    """
    Main function: Execute multiple test cases
    """
    parser = argparse.ArgumentParser(description="Dequantize GEMM (W4A8) Example")
    parser.add_argument("--m", type=int, default=128, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=256, help="Matrix N dimension")
    parser.add_argument("--k", type=int, default=256, help="Matrix K dimension")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}] Unknown args:", remains)

    M, N, K = args.m, args.n, args.k

    tl.cache.clear_cache()
    torch.manual_seed(0)

    print("=" * 60)
    print("Dequantize GEMM (W4A8) Tests")
    print("=" * 60)

    print("\n--- Level 0: Basic Small ---")
    check_case(128, 128, 128, block_M=64, block_N=64, block_K=64, out_dtype="float16")

    print("\n--- Level 1: Typical 256 ---")
    check_case(256, 256, 256, block_M=128, block_N=128, block_K=128, out_dtype="float16")

    print("\n--- Level 1: Typical 512 ---")
    check_case(512, 512, 512, block_M=128, block_N=128, block_K=128, out_dtype="float16")

    print("\n--- Level 3: Large (user specified) ---")
    check_case(M, N, K, block_M=128, block_N=128, block_K=128, out_dtype="float16")

    print("\n--- Level 3: Large bfloat16 ---")
    check_case(M, N, K, block_M=128, block_N=128, block_K=128, out_dtype="bfloat16")

    print("=" * 60)
    print("All tests passed!")
    print("Kernel Output Match!")
    print("=" * 60)


if __name__ == "__main__":
    main()
