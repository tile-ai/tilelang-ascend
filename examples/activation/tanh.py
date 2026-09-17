import tilelang
import tilelang.language as T
import torch


def _get_precision(dtype):
    dtype_name = str(dtype).replace("torch.", "")
    if dtype_name.startswith("float8_e4m3"):
        dtype_name = "float8_e4m3"
    elif dtype_name.startswith("float8_e5m2"):
        dtype_name = "float8_e5m2"
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1.0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1.0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),
    }
    if dtype_name in {"int8", "int16", "int32", "int64", "uint8"}:
        return 0.0, 0.0, 0.0, 1.0
    return fp_table.get(dtype_name, fp_table["float16"])


def _check_precision(actual, golden, dtype):
    if actual.shape != golden.shape:
        return False, 0.0, float("inf")
    atol, rtol, max_abs_limit, required_ratio = _get_precision(dtype)
    actual_cpu = actual.detach().cpu()
    golden_cpu = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:
        mismatches = (actual_cpu != golden_cpu).sum().item()
        total = max(actual_cpu.numel(), 1)
        return mismatches == 0, 1.0 - mismatches / total, 0.0 if mismatches == 0 else float("inf")
    actual_fp32 = actual_cpu.float()
    golden_fp32 = golden_cpu.float()
    special = ~torch.isfinite(golden_fp32)
    if special.any() and (
        not torch.equal(torch.isnan(actual_fp32[special]), torch.isnan(golden_fp32[special]))
        or not torch.equal(torch.isinf(actual_fp32[special]), torch.isinf(golden_fp32[special]))
        or not torch.equal(actual_fp32[special][torch.isinf(golden_fp32[special])], golden_fp32[special][torch.isinf(golden_fp32[special])])
    ):
        return False, 0.0, float("inf")
    finite = torch.isfinite(golden_fp32)
    if finite.sum().item() == 0:
        return True, 1.0, 0.0
    abs_error = (actual_fp32[finite] - golden_fp32[finite]).abs()
    abs_error = torch.where(torch.isfinite(abs_error), abs_error, torch.full_like(abs_error, float("inf")))
    matched_ratio = (abs_error <= (atol + rtol * golden_fp32[finite].abs())).float().mean().item()
    max_abs_error = abs_error.max().item()
    return matched_ratio >= required_ratio and max_abs_error <= max_abs_limit, matched_ratio, max_abs_error


tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs, target="pto")
def tanh(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            nega_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            denom_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            zero_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.fill(zero_ub, 0.0)
            T.tile.sub(nega_ub, zero_ub, a_ub)
            T.tile.exp(a_ub, a_ub)
            T.tile.exp(nega_ub, nega_ub)
            T.tile.sub(b_ub, a_ub, nega_ub)
            T.tile.add(denom_ub, a_ub, nega_ub)
            T.tile.div(b_ub, b_ub, denom_ub)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


torch.manual_seed(0)
# Tests
test_configs = [
    (256, 256, 64, 64),
    (300, 300, 64, 64),
    (1100, 50000, 128, 128),
]

for M, N, block_M, block_N in test_configs:
    print(f"Testing tanh with M={M}, N={N}, block_M={block_M}, block_N={block_N}")
    func = tanh(M, N, block_M, block_N)
    print("Init successful!")
    a = torch.randn(M, N).npu()
    b = func(a)
    ref_b = torch.tanh(a)
    passed, ratio, max_abs = _check_precision(b, ref_b, b.dtype)
    assert passed, f"dtype={b.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
    print("Test passed!")

print("Kernel Output Match!")
