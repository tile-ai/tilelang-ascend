import tilelang
import tilelang.language as T
import torch


def _check_precision(actual, golden, dtype):
    name = str(dtype).replace("torch.", "")
    table = {
        "float16": (2**-14, 2**-9, 0.1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1.0, 0.99),
        "float32": (2**-16, 2**-10, 0.01, 0.99),
        "hifloat32": (2**-16, 2**-10, 0.01, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1.0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 0.1, 0.99),
    }
    actual_cpu, golden_cpu = actual.detach().cpu(), golden.detach().cpu()
    if actual_cpu.shape != golden_cpu.shape:
        return False, 0.0, float("inf")
    if name in {"int8", "int16", "int32", "int64", "uint8"}:
        mismatches = (actual_cpu != golden_cpu).sum().item()
        return mismatches == 0, 1.0 - mismatches / max(actual_cpu.numel(), 1), 0.0 if mismatches == 0 else float("inf")
    atol, rtol, limit, required = table.get(name, table["float16"])
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
    error = (actual_fp32[finite] - golden_fp32[finite]).abs()
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))
    ratio, max_abs = (error <= atol + rtol * golden_fp32[finite].abs()).float().mean().item(), error.max().item()
    return ratio >= required and max_abs <= limit, ratio, max_abs


import torch.nn as nn

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def swi_glu(M, N, block_M, block_N, split_dim, dtype="float"):
    # The `swi_glu` operator splits the input tensor into two tensors, x1 and x2, based on the split dimension.
    # It performs a Swish operation on x1 and multiplies the result by x2.
    m_div = 1
    n_div = 2
    m_offset = 0
    n_offset = N // 2
    if split_dim == split_dim == 0 or split_dim == -2:
        m_div = 2
        n_div = 1
        m_offset = M // 2
        n_offset = 0
    m_num = T.ceildiv(M // m_div, block_M)
    n_num = T.ceildiv(N // n_div, block_N)

    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M // m_div, N // n_div), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a0_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            a1_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            zero_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            temp_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            ## [In vector]
            # The first half is cached using a0_ub
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a0_ub)
            # The second half is cached using a1_ub
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM + m_offset, by * block_N + n_offset], a1_ub)
            T.tile.fill(zero_ub, 0.0)
            # Calculation formula:-x
            T.tile.sub(temp_ub, zero_ub, a0_ub)
            # Calculation formula:exp(-x)
            T.tile.exp(temp_ub, temp_ub)
            # Calculation formula:1 + exp(-x)
            T.tile.add(temp_ub, temp_ub, 1.0)
            # Calculation formula:x / (1 + exp(-x))
            T.tile.div(temp_ub, a0_ub, temp_ub)
            # Multiply the result of the first half by the second half
            T.tile.mul(b_ub, temp_ub, a1_ub)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


torch.manual_seed(0)
# Tests
test_configs = [
    (256, 256, 64, 64, 0),
    (256, 256, 64, 64, 1),
    (1024, 1024, 128, 128, -2),
    (1024, 1024, 128, 128, -1),
]

for M, N, block_M, block_N, split_dim in test_configs:
    print(f"Testing swi_gul with M={M}, N={N}, block_M={block_M}, block_N={block_N}, split_dim={split_dim}")
    func = swi_glu(M, N, block_M, block_N, split_dim)
    print("Init successful!")
    a = torch.randn(M, N, dtype=torch.float).npu()
    b = func(a)
    print(func.get_kernel_source())
    split_size = N // 2
    if split_dim == 0 or split_dim == -2:
        split_size = M // 2
    a1, a2 = torch.split(a, split_size, dim=split_dim)
    silu = nn.SiLU()
    ref_b = silu(a1) * a2
    passed, ratio, max_abs = _check_precision(b, ref_b, b.dtype)
    assert passed, f"dtype={b.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
    print("Test passed!")

print("Kernel Output Match!")
