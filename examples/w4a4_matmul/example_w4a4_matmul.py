import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu

# W4A4 GEMM: int4 (weights) x int4 (activations) -> int32 accumulation.
#
# int4 support notes (issue #438):
#   - int4 kernels require the default `ascendc` target; the `pto` target does
#     not support int4 yet (pto-isa lacks the s4 instructions, see
#     https://gitcode.com/cann/pto-isa/issues/115).
#   - transpose_A / transpose_B are not supported for int4 (the s4 L1->L0 load
#     instructions have no transpose mode for A, and the B transpose load
#     requires a zN source layout).
#   - GM int4 tensors are nibble-packed: two elements share one byte (low
#     nibble = even element, high nibble = odd element); row r of an (M, N)
#     tensor starts at byte r * N / 2. The quant kernel below produces this
#     packing directly on device, so the whole chain stays on NPU.

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def float16_to_int4_quant(M, N, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "int4"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), "float16")
            q_ub = T.alloc_ub((block_M, block_N), "int4")

            with T.Scope("V"):
                T.copy(A[bx * block_M, by * block_N], a_ub)
                T.tile.cast(q_ub, a_ub, mode="CAST_RINT", count=block_M * block_N)
                T.copy(q_ub, B[bx * block_M, by * block_N])

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def w4a4_matmul(M, N, K, block_M, block_N, block_K):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "int4"),
        B: T.Tensor((K, N), "int4"),
        C: T.Tensor((M, N), "int32"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, block_K), "int4")
            B_L1 = T.alloc_L1((block_K, block_N), "int4")
            C_L0 = T.alloc_L0C((block_M, block_N), "int32")

            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    T.barrier_all()
                    T.copy(B[k * block_K, by * block_N], B_L1)
                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def main():
    parser = argparse.ArgumentParser(description="W4A4 int4xint4 Matmul Example")
    parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
    parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
    parser.add_argument("--block-m", type=int, default=128, help="Tile M dimension")
    parser.add_argument("--block-n", type=int, default=256, help="Tile N dimension")
    parser.add_argument("--block-k", type=int, default=64, help="Tile K dimension")
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k
    block_M, block_N, block_K = args.block_m, args.block_n, args.block_k
    assert M % block_M == 0 and N % block_N == 0 and K % block_K == 0, "M/N/K must be divisible by the block sizes"

    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    # Integer-valued fp16 in [-8, 7] (the full int4 range) so that CAST_RINT
    # quantization is lossless and the int4 matmul is exact.
    a = torch.randint(-8, 8, (M, K), dtype=torch.float16).npu()
    b = torch.randint(-8, 8, (K, N), dtype=torch.float16).npu()

    quant_a = float16_to_int4_quant(M, K, block_M, block_K)
    quant_b = float16_to_int4_quant(K, N, block_K, block_N)
    matmul = w4a4_matmul(M, N, K, block_M, block_N, block_K)

    a_q = quant_a(a)
    torch_npu.npu.synchronize()
    b_q = quant_b(b)
    torch_npu.npu.synchronize()
    c = matmul(a_q, b_q)
    torch_npu.npu.synchronize()

    # int32 reference computed from the original fp16 tensors.
    ref_c = a.cpu().to(torch.int32) @ b.cpu().to(torch.int32)

    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
