import argparse

import tilelang
import tilelang.language as T
import torch
from tilelang.profiler import do_bench


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


@tilelang.jit(out_idx=[-1], target="ascendc")
def matmul(M, N, K, block_M, block_N, block_K, K_L1, S1, S2, dtype="float16", accum_dtype="float"):
    core_num = 20

    @T.macro
    def init_flag():
        T.set_flag("mte1", "mte2", 0)
        T.set_flag("mte1", "mte2", 1)
        T.set_flag("m", "mte1", 0)
        T.set_flag("m", "mte1", 1)
        T.set_flag("fix", "m", 0)

    @T.macro
    def clear_flag():
        T.wait_flag("mte1", "mte2", 0)
        T.wait_flag("mte1", "mte2", 1)
        T.wait_flag("m", "mte1", 0)
        T.wait_flag("m", "mte1", 1)
        T.wait_flag("fix", "m", 0)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((S1, block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((S1, K_L1, block_N), dtype)

            A_L0 = T.alloc_L0A((S2, block_M, block_K), dtype)
            B_L0 = T.alloc_L0B((S2, block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                init_flag()

                for bx, by in T.Persistent([T.ceildiv(M, block_M), T.ceildiv(N, block_N)], core_num, cid):
                    loop_k = T.ceildiv(K, K_L1)

                    T.wait_flag("mte1", "mte2", 0)
                    T.copy(A[bx * block_M, 0], A_L1[0, :, :])
                    T.copy(B[0, by * block_N], B_L1[0, :, :])
                    T.set_flag("mte2", "mte1", 0)
                    T.wait_flag("fix", "m", 0)
                    for k in T.serial(loop_k):
                        if k < loop_k - 1:
                            T.wait_flag("mte1", "mte2", (k + 1) % S1)
                            T.copy(A[bx * block_M, (k + 1) * K_L1], A_L1[(k + 1) % S1, :, :])
                            T.copy(B[(k + 1) * K_L1, by * block_N], B_L1[(k + 1) % S1, :, :])
                            T.set_flag("mte2", "mte1", (k + 1) % S1)

                        loop_kk = T.ceildiv(K_L1, block_K)

                        for kk in T.serial(loop_kk):
                            if kk == 0:
                                T.wait_flag("mte2", "mte1", k % S1)
                            T.wait_flag("m", "mte1", kk % S2)
                            T.copy(A_L1[k % S1, 0, kk * block_K], A_L0[kk % S2, :, :])
                            T.copy(B_L1[k % S1, kk * block_K, 0], B_L0[kk % S2, :, :])
                            if kk == 3:
                                T.set_flag("mte1", "mte2", k % S1)
                            T.set_flag("mte1", "m", kk % S2)
                            T.wait_flag("mte1", "m", kk % S2)

                            T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :], C_L0, init=T.And(k == 0, kk == 0))

                            T.set_flag("m", "mte1", kk % S2)

                    T.set_flag("m", "fix", 0)
                    T.wait_flag("m", "fix", 0)
                    T.copy(C_L0, C[bx * block_M, by * block_N])
                    T.set_flag("fix", "m", 0)

                clear_flag()

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
    parser.add_argument("--m", type=int, default=8192, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
    parser.add_argument("--k", type=int, default=8192, help="Matrix K dimension")
    args = parser.parse_args()

    M = args.m
    N = args.n
    K = args.k

    func = matmul(M, N, K, 128, 256, 64, 256, 2, 2)

    print(func.get_kernel_source())

    torch.manual_seed(0)

    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()

    print("init successful!")

    c = func(a, b)
    ref_c = a @ b

    torch.npu.synchronize()

    print(c, ref_c)

    tilelang_time = do_bench(lambda: func(a, b))
    torch_time = do_bench(lambda: a @ b)

    print(f"tilelang time: {tilelang_time} ms")
    print(f"torch time: {torch_time} ms")

    _check_precision(c, ref_c, "float16")
    print("Kernel Output Match!")
