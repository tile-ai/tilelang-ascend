# Copyright (c) Huawei Technologies Co., Ltd. 2026.
import os
import torch
import tilelang as tl
import tilelang.language as T


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _gen_fp8_e4m3_like_tensor(shape, device: torch.device) -> torch.Tensor:
    # E4M3 finite range is approximately [-448, 448].
    x = torch.randn(shape, dtype=torch.float32) * 96.0
    x = torch.clamp(x, -448.0, 448.0)
    if hasattr(torch, "float8_e4m3fn"):
        x = x.to(torch.float8_e4m3fn).to(torch.float16)
    else:
        x = x.to(torch.float16)
    return x.to(device).contiguous()


@tl.jit(target="npuir")
def fp8_gemm_kernel(
    N, K, out_dtype="float16", in_dtype="float16", accum_dtype="float32"
):
    assert out_dtype in ["float16", "float32", "bfloat16"]

    M = T.symbolic("M")
    group_size = 128
    block_M = 32
    block_N = 128
    block_K = 128

    assert in_dtype in ["float16"]
    assert group_size == block_N, "This kernel expects group_size == block_N"
    assert N % block_N == 0, "N must be divisible by block_N"
    assert K % block_K == 0, "K must be divisible by block_K"

    @T.prim_func
    def fp8_gemm_kernel_(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        C: T.Tensor((M, N), out_dtype),
        scales_a: T.Tensor((M, T.ceildiv(K, group_size)), "float32"),
        scales_b: T.Tensor(
            (T.ceildiv(N, group_size), T.ceildiv(K, group_size)), "float32"
        ),
    ):
        # GPU 2D launch (by, bx) -> NPU 1D launch (cid) mapping.
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (
            cid,
            _,
        ):
            bx = cid % T.ceildiv(N, block_N)
            by = cid // T.ceildiv(N, block_N)

            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            Scale_C_shared = T.alloc_shared((block_M), "float32")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local_accum)
            k_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(k_iters, num_stages=2):
                k_start = k * block_K

                T.copy(
                    A[
                        by * block_M : (by + 1) * block_M,
                        k_start : k_start + block_K,
                    ],
                    A_shared,
                )
                T.copy(
                    B[
                        bx * block_N : (bx + 1) * block_N,
                        k_start : k_start + block_K,
                    ],
                    B_shared,
                )

                scale_b = scales_b[bx * block_N // group_size, k]
                for i in T.Parallel(block_M):
                    Scale_C_shared[i] = scales_a[by * block_M + i, k] * scale_b

                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    initC=True,
                    b_transpose=True,
                    size=[block_M, block_K, block_N],
                )

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]

            T.copy(
                C_local_accum,
                C[
                    by * block_M : (by + 1) * block_M,
                    bx * block_N : (bx + 1) * block_N,
                ],
            )

    return fp8_gemm_kernel_


def fp8_gemm_torch_ref(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    group_size: int,
    out_dtype: str,
) -> torch.Tensor:
    m, k = a.shape
    n, _ = b.shape
    k_groups = a_s.shape[1]
    out = torch.zeros((m, n), dtype=torch.float32, device=a.device)

    for kg in range(k_groups):
        k0 = kg * group_size
        k1 = min(k, k0 + group_size)

        part = a[:, k0:k1].float() @ b[:, k0:k1].float().transpose(0, 1)
        part = part * a_s[:, kg].float().unsqueeze(1)

        for ng in range(b_s.shape[0]):
            n0 = ng * group_size
            n1 = min(n, n0 + group_size)
            part[:, n0:n1] = part[:, n0:n1] * b_s[ng, kg].float()

        out += part

    if out_dtype == "float16":
        return out.to(torch.float16)
    if out_dtype == "bfloat16":
        return out.to(torch.bfloat16)
    return out


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    out_dtype: str = "float16",
    group_size: int = 128,
) -> torch.Tensor:
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), (
        "Scale tensors must be contiguous"
    )

    m, k = a.shape
    n = b.shape[0]

    kernel = fp8_gemm_kernel(
        N=n,
        K=k,
        out_dtype=out_dtype,
        in_dtype="float16",
        accum_dtype="float32",
    )

    c = torch.empty((m, n), dtype=getattr(torch, out_dtype), device=a.device)
    out = kernel(a, b, c, a_s, b_s)
    if out is None:
        return c
    return out


def run_test_case(m: int, n: int, k: int, out_dtype: str = "float16"):
    group_size = 128
    assert m % 32 == 0
    assert n % 128 == 0
    assert k % 128 == 0
    k_groups = _ceildiv(k, group_size)
    n_groups = _ceildiv(n, group_size)

    # Use fp8_e4m3 range and quantization, then run kernel with fp16 carriers.
    npu_device = torch.device("npu")
    a = _gen_fp8_e4m3_like_tensor((m, k), npu_device)
    b = _gen_fp8_e4m3_like_tensor((n, k), npu_device)

    a_s = (
        torch.rand((m, k_groups), dtype=torch.float32).npu() * 0.2 + 0.9
    ).contiguous()
    b_s = (
        torch.rand((n_groups, k_groups), dtype=torch.float32).npu() * 0.2 + 0.9
    ).contiguous()

    out = fp8_gemm(a, a_s, b, b_s, out_dtype=out_dtype, group_size=group_size)
    ref = fp8_gemm_torch_ref(a, a_s, b, b_s, group_size=group_size, out_dtype=out_dtype)

    atol = 2e-2 if out_dtype in ("float16", "bfloat16") else 1e-2
    rtol = 2e-2 if out_dtype in ("float16", "bfloat16") else 1e-2
    torch.testing.assert_close(out.float(), ref.float(), rtol=rtol, atol=atol)


def run_test():
    run_test_case(m=128, n=256, k=256, out_dtype="float16")
    run_test_case(m=96, n=128, k=256, out_dtype="float32")
    run_test_case(m=128, n=256, k=256, out_dtype="bfloat16")
    print("\033[92mFP8 GEMM NPU test passed.\033[0m")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    torch.npu.set_device(0)
    tl.cache.clear_cache()
    run_test()