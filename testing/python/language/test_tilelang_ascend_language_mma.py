import pytest
import tilelang
import tilelang.language as T
import torch
import argparse
from tilelang.intrinsics import make_zn_layout


def matmul(M, N, K, block_M, block_N, block_K, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)

            T.annotate_layout(
                {
                    A_L1: make_zn_layout(A_L1),
                    B_L1: make_zn_layout(B_L1),
                }
            )

            A_L0 = T.alloc_L0A((block_M, block_K), dtype)
            B_L0 = T.alloc_L0B((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                bx = cid // n_num
                by = cid % n_num

                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.barrier_all()
                    T.copy(A[bx * block_M, k * block_K], A_L1[:, :])
                    T.copy(B[k * block_K, by * block_N], B_L1[:, :])
                    T.barrier_all()

                    T.copy(A_L1[:, :], A_L0[:, :])
                    T.copy(B_L1[:, :], B_L0[:, :])

                    T.barrier_all()

                    T.mma(A_L0[:, :], B_L0[:, :], C_L0, init=(k == 0))

                    T.barrier_all()

                T.barrier_all()
                T.copy(C_L0, C[bx * block_M, by * block_N])
                T.barrier_all()

    return main


def run_test(M, N, K, block_M, block_N, block_k, K_L1, target):
    device = "npu"
    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    # 1. Compile the operator
    func_def = matmul(M, N, K, block_M, block_N, block_k, K_L1)
    func = tilelang.compile(func_def, out_idx=[-1], target=target)
    print(func.get_kernel_source())

    # 2. Prepare data
    a = torch.randn(M, K).to(device).half()
    b = torch.randn(K, N).to(device).half()

    # 3. Run the operator
    torch.npu.synchronize()
    c = func(a, b)

    # 4. Verify accuracy
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-3, atol=1e-3)
    print("Test Passed!")


# -----------------------------------------------------------------------------
# Pytest entry point
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(8192, 1024, 8192)])
def test_select_op(target, shape):
    M, N, K = shape
    run_test(M, N, K, 128, 256, 64, 64, target=target)


# -----------------------------------------------------------------------------
# Standalone command-line entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8192, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
    parser.add_argument("--k", type=int, default=8192, help="Matrix K dimension")
    parser.add_argument("--target", type=str, choices=["ascendc", "pto"], default="ascendc")
    args = parser.parse_args()

    run_test(args.m, args.n, args.k, 128, 256, 64, 64, target=args.target)
