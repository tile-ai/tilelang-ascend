import argparse

import tilelang
from tilelang import language as T
import torch


def _check_precision(actual, golden, dtype):
    table = {
        "float16": (2**-14, 2**-9, 0.1),
        "bfloat16": (2**-10, 2**-6, 1.0),
        "float32": (2**-16, 2**-10, 0.01),
        "hifloat32": (2**-16, 2**-10, 0.01),
        "float8_e4m3": (2**-4, 2**-2, 1.0),
        "float8_e5m2": (2**-3, 2**-1, 0.1),
    }
    actual, golden = actual.detach().cpu().float(), golden.detach().cpu().float()
    if actual.shape != golden.shape:
        return False, 0.0, float("inf")
    dtype_name = str(dtype).replace("torch.", "")
    dtype_name = "float8_e4m3" if "float8_e4m3" in dtype_name else "float8_e5m2" if "float8_e5m2" in dtype_name else dtype_name
    atol, rtol, limit = table.get(dtype_name, table["float16"])
    special = ~torch.isfinite(golden)
    if special.any() and (
        not torch.equal(torch.isnan(actual[special]), torch.isnan(golden[special]))
        or not torch.equal(torch.isinf(actual[special]), torch.isinf(golden[special]))
    ):
        return False, 0.0, float("inf")
    finite = torch.isfinite(golden)
    if not finite.any():
        return True, 1.0, 0.0
    error = (actual[finite] - golden[finite]).abs()
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))
    ratio, maximum = (error <= atol + rtol * golden[finite].abs()).float().mean().item(), error.max().item()
    return ratio >= 0.99 and maximum <= limit, ratio, maximum


@tilelang.jit(out_idx=[1])
def reduce_min(M, N, block_M, dtype="float"):
    m_num = M // block_M
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),
        B: T.Tensor([M], dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((sub_block_M, N), dtype)
            b_ub = T.alloc_ub((sub_block_M), dtype)

            row_base = cid * block_M + vid * sub_block_M
            with T.Scope("V"):
                T.copy(A[row_base : row_base + sub_block_M, :], a_ub)

                T.barrier_all()
                T.reduce_min(a_ub, b_ub, dim=-1)
                T.barrier_all()

                T.copy(b_ub, B[row_base : row_base + sub_block_M])

    return main


if __name__ == "__main__":
    tilelang.cache.clear_cache()

    parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
    parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=256, help="Matrix N dimension")
    args = parser.parse_args()

    M = args.m
    N = args.n

    func = reduce_min(M, N, 128)

    torch.manual_seed(0)

    a = torch.randn(M, N).npu()

    torch.npu.synchronize()
    print("init successful!")

    c = func(a)

    ref_c = torch.min(a, dim=-1).values

    passed, ratio, max_abs = _check_precision(c, ref_c, c.dtype)
    assert passed, f"dtype={c.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
    print("Kernel Output Match!")
