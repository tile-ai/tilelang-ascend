import argparse

import tilelang
import tilelang.language as T
import torch


def _check_precision(actual, golden, dtype):
    table = {
        "float16": (2**-14, 2**-9, 1e-1),
        "bfloat16": (2**-10, 2**-6, 1e0),
        "float32": (2**-16, 2**-10, 1e-2),
        "hifloat32": (2**-16, 2**-10, 1e-2),
        "float8_e4m3": (2**-4, 2**-2, 1e0),
        "float8_e5m2": (2**-3, 2**-1, 1e-1),
    }
    a, g = actual.detach().cpu(), golden.detach().cpu()
    if a.shape != g.shape:
        return False, 0.0, float("inf")
    name = str(dtype).replace("torch.", "")
    if name in {"int8", "int16", "int32", "int64", "uint8"} or not a.dtype.is_floating_point:
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, 0.0 if mism == 0 else float("inf")
    if name.startswith("float8_e4m3"):
        name = "float8_e4m3"
    if name.startswith("float8_e5m2"):
        name = "float8_e5m2"
    atol, rtol, limit = table.get(name, table["float16"])
    a, g = a.float(), g.float()
    if not (
        torch.equal(torch.isnan(a), torch.isnan(g))
        and torch.equal(torch.isposinf(a), torch.isposinf(g))
        and torch.equal(torch.isneginf(a), torch.isneginf(g))
    ):
        return False, 0.0, float("inf")
    finite = torch.isfinite(g)
    if not finite.any():
        return True, 1.0, 0.0
    error = torch.where(
        torch.isfinite((a[finite] - g[finite]).abs()), (a[finite] - g[finite]).abs(), torch.full_like(g[finite], float("inf"))
    )
    ratio = (error <= atol + rtol * g[finite].abs()).float().mean().item()
    maximum = error.max().item()
    return ratio >= 0.99 and maximum <= limit, ratio, maximum


tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=8192, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=8192, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)

            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.Pipelined(loop_k, num_stages=3):
                    T.barrier_all()
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    T.copy(B[k * block_K, by * block_N], B_L1)

                    if k == 0:
                        T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                    else:
                        T.gemm_v0(A_L1, B_L1, C_L0)

                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


func = matmul(M, N, K, 128, 256, 64)

torch.manual_seed(0)

a = torch.randn(M, K).half().npu()
b = torch.randn(K, N).half().npu()
d = torch.randn(M, N).half().npu()
print("init successful!")

c = func(a, b)

ref_c = a @ b

passed, ratio, max_abs = _check_precision(c, ref_c, c.dtype)
assert passed, f"dtype={c.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
print("Kernel Output Match!")
