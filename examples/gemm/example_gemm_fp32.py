import argparse

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="fp32 GEMM on Ascend NPU")
parser.add_argument("--m", type=int, default=1024)
parser.add_argument("--n", type=int, default=1024)
parser.add_argument("--k", type=int, default=1024)
args = parser.parse_args()

M, N, K = args.m, args.n, args.k
block_M, block_N, K_L1 = 128, 128, 64

# fp32 elements are 4 bytes: (block_M x kL0Size) must fit the L0A ping-pong
# slot, so kL0Size=64 (not the fp16-typical 128) for a 128-row A tile.
kL0Size = 64


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float32", accum_dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0), kL0Size=kL0Size)
                    T.barrier_all()
                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float32).npu()
    b = torch.randn(K, N, dtype=torch.float32).npu()
    func = matmul(M, N, K, block_M, block_N, K_L1)
    c = func(a, b)
    torch.npu.synchronize()
    ref = a @ b
    diff = (c - ref).abs().max().item()
    print(f"fp32 GEMM {M}x{N}x{K}, max_abs_diff = {diff}")
    print("Kernel Output Match!" if torch.allclose(c, ref, rtol=1e-3, atol=1e-3) else "MISMATCH!")
