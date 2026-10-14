import argparse

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

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n


@tilelang.jit(out_idx=[-1])
def vec_add(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                T.barrier_all()
                T.tile.add(c_ub, a_ub, b_ub)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


func = vec_add(M, N, 128, 256)

torch.manual_seed(0)

a = torch.randn(M, N).npu()
b = torch.randn(M, N).npu()

torch.npu.synchronize()
print("init successful!")

c = func(a, b)

ref_c = a + b

passed, ratio, max_abs = _check_precision(c, ref_c, c.dtype)
assert passed, f"dtype={c.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
print("Kernel Output Match!")
