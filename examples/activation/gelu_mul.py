import tilelang
import tilelang.language as T
import torch


def _check_precision(actual, golden, dtype):
    name = str(dtype).replace("torch.", "")
    if name.startswith("float8_e4m3"):
        name = "float8_e4m3"
    elif name.startswith("float8_e5m2"):
        name = "float8_e5m2"
    table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1.0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1.0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),
    }
    actual_cpu, golden_cpu = actual.detach().cpu(), golden.detach().cpu()
    if actual_cpu.shape != golden_cpu.shape:
        return False, 0.0, float("inf")
    if name in {"int8", "int16", "int32", "int64", "uint8"}:
        mismatches, total = (actual_cpu != golden_cpu).sum().item(), max(actual_cpu.numel(), 1)
        return mismatches == 0, 1.0 - mismatches / total, 0.0 if mismatches == 0 else float("inf")
    atol, rtol, max_limit, required_ratio = table.get(name, table["float16"])
    actual_fp32, golden_fp32 = actual_cpu.float(), golden_cpu.float()
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
    ratio = (abs_error <= atol + rtol * golden_fp32[finite].abs()).float().mean().item()
    max_abs = abs_error.max().item()
    return ratio >= required_ratio and max_abs <= max_limit, ratio, max_abs


import torch.nn as nn

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def gelu_mul(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    # The `gelu_mul` operator splits the input tensor into two tensors, x1 and x2, based on the last dimension.
    # It performs a GELU operation on x1 and multiplies the result by x2. Therefore, the kernel splitting is only relative to the dimension of x1.
    n_num = T.ceildiv(N // 2, block_N)

    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N // 2), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a1_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            a2_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            temp_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            ## [In vector]
            # The left half is cached using a1_ub
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a1_ub)
            # The right half is cached using a2_ub
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N + N // 2], a2_ub)
            # Calculation formula:x^2
            T.tile.mul(temp_ub, a1_ub, a1_ub)
            # Calculation formula:x^3
            T.tile.mul(temp_ub, a1_ub, temp_ub)
            # Calculation formula:0.044715 * x^3
            T.tile.mul(temp_ub, temp_ub, 0.044715)
            # Calculation formula:x + 0.044715 * x^3
            T.tile.add(temp_ub, a1_ub, temp_ub)
            # Calculation formula:-sqrt(8/pi)(x + 0.044715 * x^3)
            T.tile.mul(temp_ub, temp_ub, -1.5957691)
            # Calculation formula:exp(-sqrt(8/pi)(x + 0.044715 * x^3))
            T.tile.exp(temp_ub, temp_ub)
            # Calculation formula:1 + exp(-sqrt(8/pi)(x + 0.044715 * x^3))
            T.tile.add(temp_ub, temp_ub, 1.0)
            # Calculation formula:x / (1 + exp(-sqrt(8/pi)(x + 0.044715 * x^3)))
            T.tile.div(temp_ub, a1_ub, temp_ub)
            # Multiply the result of the left half by the right half
            T.tile.mul(b_ub, temp_ub, a2_ub)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


torch.manual_seed(0)
# Tests
test_configs = [
    (256, 256, 64, 64),
    (1024, 1024, 128, 128),
]

for M, N, block_M, block_N in test_configs:
    print(f"Testing gelu_mul with M={M}, N={N}, block_M={block_M}, block_N={block_N}")
    func = gelu_mul(M, N, block_M, block_N)
    print("Init successful!")
    a = torch.randn(M, N, dtype=torch.float).npu()
    b = func(a)
    gelu = nn.GELU(approximate="tanh")
    a1, a2 = torch.split(a, N // 2, dim=1)
    ref_b = gelu(a1) * a2
    passed, ratio, max_abs = _check_precision(b, ref_b, b.dtype)
    assert passed, f"dtype={b.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
    print("Test passed!")

print("Kernel Output Match!")
