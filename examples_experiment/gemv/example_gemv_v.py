import argparse

import tilelang as tl
import tilelang.language as T

import torch


def _check_precision(actual, golden, dtype):
    values = {
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
    atol, rtol, limit = values.get(dtype_name, values["float16"])
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


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    },
)
def simple_gemv(N: int, K: int, block_N: int, block_K: int, dtype: str = "float16", accum_dtype: str = "float32"):
    """Vector core GEMV implementation"""
    VEC_NUM = 2
    TEMP_DTYPE = "uint8"
    CAST_MODE = "CAST_NONE"

    def current_jit_target(jit_func):
        """Get current jit target of jit_func. e.g. 'auto', 'pto'"""
        return jit_func.__jit_impl__.target

    n_num = T.ceildiv(N, block_N)
    k_num = T.ceildiv(K, block_K)

    kernel_num = T.ceildiv(n_num, VEC_NUM)

    not_same_dtype = dtype != accum_dtype

    def cast_or_copy(dst, src, mode, count):
        if not_same_dtype:
            return T.tile.cast(dst, src, mode, count)
        else:
            return T.copy(src, dst)

    is_pto = current_jit_target(simple_gemv) == "pto"

    def alloc_temp():
        if is_pto:
            return T.alloc_ub((block_N, block_K // 2), accum_dtype)
        else:
            return T.alloc_ub((block_N, block_K), TEMP_DTYPE)

    @T.prim_func
    def main(
        x: T.Tensor((K,), dtype),  # type: ignore
        A: T.Tensor((N, K), dtype),  # type: ignore
        y: T.Tensor((N,), dtype),  # type: ignore
    ):
        with T.Kernel(kernel_num, is_npu=True) as (cid, vid):
            bn = (cid * VEC_NUM + vid) % n_num

            x_ub = T.alloc_ub((1, block_K), dtype)
            x_32_ub = T.alloc_ub((1, block_K), accum_dtype)
            A_ub = T.alloc_ub((block_N, block_K), dtype)
            A_32_ub = T.alloc_ub((block_N, block_K), accum_dtype)
            y_single_32_ub = T.alloc_ub((block_N,), accum_dtype)
            y_total_32_ub = T.alloc_ub((block_N,), accum_dtype)
            y_ub = T.alloc_ub((block_N,), dtype)

            T.tile.fill(y_total_32_ub, 0.0)

            # block_N * K  per vector core
            for bk in T.serial(k_num):
                T.copy(x[bk * block_K], x_ub)
                T.copy(A[bn * block_N, bk * block_K], A_ub)
                cast_or_copy(x_32_ub, x_ub, CAST_MODE, block_K)  # cast to float for reduce_sum
                cast_or_copy(A_32_ub, A_ub, CAST_MODE, block_N * block_K)
                for i in T.serial(block_N):
                    T.tile.mul(A_32_ub[i, :], A_32_ub[i, :], x_32_ub)
                T.reduce_sum(A_32_ub, y_single_32_ub, dim=-1)
                T.tile.add(y_total_32_ub, y_total_32_ub, y_single_32_ub)

            cast_or_copy(y_ub, y_total_32_ub, CAST_MODE, block_N)  # cast back
            T.copy(y_ub, y[bn * block_N])

    return main


def ref_program(x, A):
    return x @ A.T


def check_case(N: int, K: int, block_N: int = 64, block_K: int = 128, dtype="float16"):
    torch_dtype_map = {"float16": torch.half, "float32": torch.float32, "float": torch.float32}
    x = torch.randn(K).to(torch_dtype_map[dtype]).npu()
    A = torch.randn(N, K).to(torch_dtype_map[dtype]).npu()

    kernel = simple_gemv(N, K, block_N, block_K, dtype=dtype)

    y = kernel(x, A)
    ref_y = ref_program(x, A)

    passed, ratio, max_abs = _check_precision(y, ref_y, y.dtype)
    assert passed, f"dtype={y.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="GEMV Example")
    parser.add_argument("--n", type=int, default=1024, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=1024, help="Matrix dimension K")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}]", "Unknown args:", remains)
    N, K = args.n, args.k

    torch.manual_seed(0)

    check_case(N, K, 128, 128)
    check_case(N, K, 64, 64, dtype="float32")
    check_case(64, 64, 16, 16)

    print("GEMV example passed!")
    print("Kernel Output Match!")


if __name__ == "__main__":
    tl.disable_cache()
    main()
