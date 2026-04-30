import argparse
import tilelang as tl
import tilelang.language as T
import torch

tl.cache.clear_cache()


def int4_to_fp16(packed_val: int, pos: int) -> float:
    """
    INT4 -> FP16 dequantization (unsigned INT4: 0-15)

    Args:
        packed_val: uint8 value containing 2 packed INT4 values
        pos: position index (0 or 1)

    Returns:
        float16 value
    """
    i4 = (packed_val >> (pos * 4)) & 0xF
    return float(i4)


def torch_unpack_int4_to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation: unpack UINT8 packed INT4 data to FP16

    Args:
        tensor: torch.Tensor, shape (N, K//2), dtype uint8

    Returns:
        torch.Tensor, shape (N, K), dtype float16
    """
    assert tensor.dtype == torch.uint8
    N, K_packed = tensor.shape
    K = K_packed * 2

    result = torch.zeros(N, K, dtype=torch.float16)
    tensor_np = tensor.numpy()

    for i in range(N):
        for j in range(K):
            val = tensor_np[i, j // 2]
            pos = j % 2
            result[i, j] = int4_to_fp16(val, pos)

    return result


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    },
)
def fp16_gemv(
    N: int,
    K: int,
    block_N: int,
    block_K: int,
):
    """
    FP16 GEMV operator (Developer mode)

    Function: Performs vector-matrix multiplication C = x @ A.T
    Computation: x: (K,) FP16, A: (N, K) FP16, C: (N,) FP16
    Accumulation: FP32

    Args:
        N, K: vector/matrix dimensions
        block_N, block_K: block sizes
    """
    n_num = N // block_N

    dtype = "float16"
    accum_dtype = "float"
    CAST_MODE = "CAST_NONE"

    @T.prim_func
    def main(
        x: T.Tensor((K,), dtype),
        A: T.Tensor((N, K), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            bn = cid

            x_ub = T.alloc_ub((1, block_K), dtype)
            x_32_ub = T.alloc_ub((1, block_K), accum_dtype)
            A_ub = T.alloc_ub((block_N, block_K), dtype)
            A_32_ub = T.alloc_ub((block_N, block_K), accum_dtype)
            y_single_32_ub = T.alloc_ub((block_N,), accum_dtype)
            y_total_32_ub = T.alloc_ub((block_N,), accum_dtype)
            C_ub = T.alloc_ub((block_N,), dtype)

            T.tile.fill(y_total_32_ub, 0.0)

            k_num = K // block_K
            for bk in T.serial(k_num):
                T.copy(x[bk * block_K], x_ub)
                T.copy(A[bn * block_N, bk * block_K], A_ub)
                T.tile.cast(x_32_ub, x_ub, CAST_MODE, block_K)
                T.tile.cast(A_32_ub, A_ub, CAST_MODE, block_N * block_K)
                for i in T.serial(block_N):
                    T.tile.mul(A_32_ub[i, :], A_32_ub[i, :], x_32_ub)
                T.reduce_sum(A_32_ub, y_single_32_ub, dim=-1)
                T.tile.add(y_total_32_ub, y_total_32_ub, y_single_32_ub)

            T.tile.cast(C_ub, y_total_32_ub, CAST_MODE, block_N)
            T.copy(C_ub, C[bn * block_N])

    return main


def dequant_gemv_wrapper(
    N: int,
    K: int,
    block_N: int,
    block_K: int,
    input_dtype: str = "float16",
    output_dtype: str = "float16",
):
    """
    Dequantize GEMV operator wrapper

    Function: Dequantizes INT4 packed matrix to FP16, then performs vector-matrix multiplication
    Computation: C = x @ B.T, where x: (K,), B: (N, K), C: (N,)

    Implementation: Python-side INT4->FP16 + NPU-side FP16 GEMV + FP32 computation precision

    Args:
        N, K: vector/matrix dimensions
        block_N, block_K: block sizes
        input_dtype: input data type
        output_dtype: output data type
    """
    kernel = fp16_gemv(N, K, block_N, block_K)

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
    }

    def wrapper(x: torch.Tensor, B_packed: torch.Tensor):
        B_fp16 = torch_unpack_int4_to_fp16(B_packed)

        if input_dtype != "float16":
            x_fp16 = x.to(torch.float16)
        else:
            x_fp16 = x

        C_fp16 = kernel(x_fp16.npu(), B_fp16.npu())

        C_out = C_fp16.cpu().to(dtype_map[output_dtype])
        return C_out

    return wrapper


def ref_program(x: torch.Tensor, B_packed: torch.Tensor, output_dtype: str):
    """
    PyTorch reference implementation

    Args:
        x: torch.Tensor, shape (K,), dtype float16
        B_packed: torch.Tensor, shape (N, K//2), dtype uint8
        output_dtype: str, output type

    Returns:
        torch.Tensor, shape (N,), dtype output_dtype
    """
    B_fp16 = torch_unpack_int4_to_fp16(B_packed)

    x_fp32 = x.to(torch.float32)
    B_fp32 = B_fp16.to(torch.float32)
    C_fp32 = torch.matmul(x_fp32, B_fp32.T)

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    C_out = C_fp32.to(dtype_map[output_dtype])
    return C_out


def check_case(
    N: int,
    K: int,
    block_N: int,
    block_K: int,
    input_dtype: str = "float16",
    output_dtype: str = "float16",
):
    assert K % 2 == 0, "K must be divisible by 2 (INT4 packed format)"
    assert N % block_N == 0, "N must be divisible by block_N"
    assert K % block_K == 0, "K must be divisible by block_K"

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
    }

    x = torch.randn(K, dtype=dtype_map[input_dtype])
    B_packed = torch.randint(0, 255, [N, K // 2], dtype=torch.uint8)

    wrapper = dequant_gemv_wrapper(N, K, block_N, block_K, input_dtype, output_dtype)
    C = wrapper(x, B_packed)

    ref_C = ref_program(x, B_packed, output_dtype)

    rtol = 1e-2 if output_dtype == "float16" else 1e-4
    atol = 1e-2 if output_dtype == "float16" else 1e-4

    torch.testing.assert_close(C, ref_C, rtol=rtol, atol=atol)
    print(f"Test passed: N={N}, K={K}, input={input_dtype}, output={output_dtype}")


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="Dequantize GEMV Example (INT4 -> FP16)")
    parser.add_argument("--n", type=int, default=1024, help="Vector N dimension")
    parser.add_argument("--k", type=int, default=1024, help="Vector K dimension")
    parser.add_argument("--input-dtype", type=str, default="float16", choices=["float16", "float32"], help="Input data type")
    parser.add_argument("--output-dtype", type=str, default="float16", choices=["float16", "float32"], help="Output data type")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}] Unknown args:", remains)

    N, K = args.n, args.k

    tl.cache.clear_cache()
    torch.manual_seed(0)

    print("=" * 60)
    print("Dequantize GEMV Tests (INT4 -> FP16)")
    print("=" * 60)

    print("\n--- Level 0: Basic Small ---")
    check_case(128, 128, block_N=128, block_K=128, input_dtype=args.input_dtype, output_dtype=args.output_dtype)

    print("\n--- Level 1: Typical 256 ---")
    check_case(256, 256, block_N=128, block_K=128, input_dtype=args.input_dtype, output_dtype=args.output_dtype)

    print("\n--- Level 1: Typical 512 ---")
    check_case(512, 512, block_N=128, block_K=128, input_dtype=args.input_dtype, output_dtype=args.output_dtype)

    print("\n--- Level 1: Typical 1024 ---")
    check_case(1024, 1024, block_N=128, block_K=128, input_dtype=args.input_dtype, output_dtype=args.output_dtype)

    print("\n--- Level 3: Large Scale ---")
    check_case(N, K, block_N=128, block_K=128, input_dtype=args.input_dtype, output_dtype=args.output_dtype)

    print("=" * 60)
    print("All tests passed!")
    print("Kernel Output Match!")
    print("=" * 60)


if __name__ == "__main__":
    main()
