import tilelang
import tilelang.language as T
import torch


def _check_precision(actual, golden, dtype):
    configs = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1.0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1.0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        assert torch.equal(actual.detach().cpu(), golden.detach().cpu()), "integer output mismatch"
        return
    atol, rtol, max_limit, ratio_limit = configs[dtype]
    actual, golden = actual.detach().cpu().float(), golden.detach().cpu().float()
    assert actual.shape == golden.shape, f"shape mismatch: {actual.shape} != {golden.shape}"
    assert torch.equal(torch.isnan(actual), torch.isnan(golden)), "NaN positions differ"
    assert torch.equal(torch.isinf(actual), torch.isinf(golden)), "Inf positions differ"
    finite = torch.isfinite(golden)
    if not finite.any():
        return
    errors = (actual[finite] - golden[finite]).abs()
    ratio, maximum = (errors <= atol + rtol * golden[finite].abs()).float().mean().item(), errors.max().item()
    assert ratio >= ratio_limit and maximum <= max_limit, f"matched_ratio={ratio:.4f}, max_abs_error={maximum:.3e}"


tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)

            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


torch.manual_seed(0)
test_configs = [
    (32 * 3 + 30, 32 * 2 + 16, 32 * 4 + 31, 32, 32, 32),
    (64 * 8 + 45, 64 * 8, 64 * 8 + 27, 64, 64, 64),
    (128 * 4, 128 * 4 + 99, 128 * 4, 128, 128, 128),
    (1024 + 118, 1024 + 206, 1024 + 55, 128, 256, 64),
]

for idx, (M, N, K, block_M, block_N, block_K) in enumerate(test_configs, 1):
    try:
        func = matmul(M, N, K, block_M, block_N, block_K)
        a = torch.randn(M, K).half().npu()
        b = torch.randn(K, N).half().npu()
        c = torch.empty(M, N).half().npu()
        c = func(a, b)
        ref_c = a @ b
        _check_precision(c, ref_c, "float16")
        print(f"Passed test case {idx}/{len(test_configs)}: M={M}, N={N}, K={K}")
    except Exception as e:
        print("error message:", e)
        print(f"Failed test case {idx}/{len(test_configs)}: M={M}, N={N}, K={K}")
        raise

print("All test cases passed!")
print("Kernel Output Match!")
