import argparse

import tilelang
import tilelang.language as T
import torch


def _check_precision(actual, golden, dtype):
    thresholds = {
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
    dtype_name = str(dtype).replace("torch.", "")
    if dtype_name in {"int8", "int16", "int32", "int64", "uint8"}:
        mismatches = (actual_cpu != golden_cpu).sum().item()
        return mismatches == 0, 1.0 - mismatches / max(actual_cpu.numel(), 1), 0.0 if mismatches == 0 else float("inf")
    if dtype_name.startswith("float8_e4m3"):
        dtype_name = "float8_e4m3"
    elif dtype_name.startswith("float8_e5m2"):
        dtype_name = "float8_e5m2"
    atol, rtol, max_limit, required_ratio = thresholds.get(dtype_name, thresholds["float16"])
    actual_fp32, golden_fp32 = actual_cpu.float(), golden_cpu.float()
    special = ~torch.isfinite(golden_fp32)
    if special.any() and (
        not torch.equal(torch.isnan(actual_fp32[special]), torch.isnan(golden_fp32[special]))
        or not torch.equal(torch.isinf(actual_fp32[special]), torch.isinf(golden_fp32[special]))
    ):
        return False, 0.0, float("inf")
    finite = torch.isfinite(golden_fp32)
    if not finite.any():
        return True, 1.0, 0.0
    abs_error = (actual_fp32[finite] - golden_fp32[finite]).abs()
    abs_error = torch.where(torch.isfinite(abs_error), abs_error, torch.full_like(abs_error, float("inf")))
    matched_ratio = (abs_error <= atol + rtol * golden_fp32[finite].abs()).float().mean().item()
    max_abs_error = abs_error.max().item()
    return matched_ratio >= required_ratio and max_abs_error <= max_limit, matched_ratio, max_abs_error


tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((block_M, K_L1), dtype)
            B_L1 = T.alloc_shared((K_L1, block_N), dtype)

            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, K_L1)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * K_L1], A_L1)
                T.copy(B[k * K_L1, by * block_N], B_L1)

                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


func = matmul(M, N, K, 128, 256, 64)

torch.manual_seed(0)

a = torch.randn(M, K).half().npu()
b = torch.randn(K, N).half().npu()
c = torch.empty(M, N).half().npu()
print("init successful!")

c = func(a, b)

ref_c = a @ b

passed, ratio, max_abs = _check_precision(c, ref_c, c.dtype)
assert passed, f"dtype={c.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
print("Kernel Output Match!")
