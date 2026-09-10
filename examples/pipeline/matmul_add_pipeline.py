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
    actual, golden = actual.detach().cpu().float(), golden.detach().cpu().float()
    if actual.shape != golden.shape:
        return False, 0.0, float("inf")
    atol, rtol, limit = table.get(str(dtype).replace("torch.", ""), table["float16"])
    ~torch.isfinite(golden)
    if not (
        torch.equal(torch.isnan(actual), torch.isnan(golden))
        and torch.equal(torch.isposinf(actual), torch.isposinf(golden))
        and torch.equal(torch.isneginf(actual), torch.isneginf(golden))
    ):
        return False, 0.0, float("inf")
    finite = torch.isfinite(golden)
    if not finite.any():
        return True, 1.0, 0.0
    error = (actual[finite] - golden[finite]).abs()
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))
    ratio, maximum = (error <= atol + rtol * golden[finite].abs()).float().mean().item(), error.max().item()
    return ratio >= 0.99 and maximum <= limit, ratio, maximum


@tilelang.jit(
    out_idx=[2],
    workspace_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    },
)
def matmul_add(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2
    vec_proc = 4

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
        D: T.Tensor((M, N), dtype),  # type: ignore
        workspace_1: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            A_L1 = T.alloc_shared((block_M, block_K), dtype)
            B_L1 = T.alloc_shared((block_K, block_N), dtype)

            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            c_ub = T.alloc_shared((block_M // VEC_NUM, block_N // vec_proc), dtype)
            d_ub = T.alloc_shared((block_M // VEC_NUM, block_N // vec_proc), dtype)
            e_ub = T.alloc_shared((block_M // VEC_NUM, block_N // vec_proc), dtype)

            loop_k = T.ceildiv(K, block_K)
            for k in T.Pipelined(loop_k, num_stages=3):
                T.copy(A[bx * block_M, k * block_K], A_L1)
                T.copy(B[k * block_K, by * block_N], B_L1)

                if k == 0:
                    T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                else:
                    T.gemm_v0(A_L1, B_L1, C_L0)

            T.copy(C_L0, workspace_1[bx * block_M, by * block_N])

            for i in T.Pipelined(vec_proc, num_stages=2):
                T.copy(workspace_1[bx * block_M + vid * block_M // VEC_NUM, by * block_N + i * block_N // vec_proc], c_ub)
                T.copy(D[bx * block_M + vid * block_M // VEC_NUM, by * block_N + i * block_N // vec_proc], d_ub)

                for j, k in T.Parallel(block_M // VEC_NUM, block_N // vec_proc):
                    e_ub[j, k] = c_ub[j, k] + d_ub[j, k]

                T.copy(e_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N + i * block_N // vec_proc])

    return main


if __name__ == "__main__":
    tilelang.disable_cache()

    parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
    parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
    parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
    args = parser.parse_args()

    M = args.m
    N = args.n
    K = args.k

    func = matmul_add(M, N, K, 128, 256, 64)

    torch.manual_seed(0)

    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()
    d = torch.randn(M, N).half().npu()
    print("init successful!")

    c = func(a, b, d)

    ref_c = a @ b + d

    passed, ratio, max_abs = _check_precision(c, ref_c, c.dtype)
    assert passed, f"dtype={c.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
    print("Kernel Output Match!")
