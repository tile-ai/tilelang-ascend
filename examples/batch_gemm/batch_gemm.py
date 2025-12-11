import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Batch Kernel Compilation")
parser.add_argument("--b", type=int, default=8, help="Batch size")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
args = parser.parse_args()

B = args.b
M = args.m
N = args.n
K = args.k


@tilelang.jit(out_idx=[-1])
def batch_matmul(B, M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A_mat: T.Tensor((B, M, K), dtype),
            B_mat: T.Tensor((B, K, N), dtype),
            C_mat: T.Tensor((B, M, N), dtype),
    ):
        total = B * m_num * n_num
        with T.Kernel(total, is_npu=True) as (cid, _):
            bid = cid // (m_num * n_num)
            rem = cid % (m_num * n_num)
            bx = rem // n_num
            by = rem % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)

            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A_mat[bid, bx * block_M, k * K_L1], A_L1)
                    T.copy(B_mat[bid, k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()

                T.copy(C_L0, C_mat[bid, bx * block_M, by * block_N])

    return main


if __name__ == "__main__":
    func = batch_matmul(B, M, N, K, 128, 256, 64)

    torch.manual_seed(0)

    a = torch.randn(B, M, K).half().npu()
    b = torch.randn(B, K, N).half().npu()

    print("init successful!")

    c = func(a, b)

    ref_c = torch.matmul(a, b)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Batch Kernel Output Match!")