# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
#
# W4A8 GEMM CV Fusion Kernel - Chunked Processing
#
# Strategy: Vector core processes B matrix chunk-by-chunk, writes to workspace,
#           Cube core performs GEMM

import argparse
import tilelang
import tilelang.language as T
import torch

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

BLOCK_K_HALF = 128
VEC_NUM = 2


@tilelang.jit(out_idx=[-1], target="pto", pass_configs=PASS_CONFIGS)
def w4a8_gemm_cv(M, N, K):
    """W4A8 GEMM CV Fusion - Developer mode with dual V-core parallelization

    With AUTO_CV_COMBINE enabled, compiler automatically separates Cube/Vector cores.
    Use vid to distribute work across two V cores to avoid redundant computation.
    """
    K_half = K // 2
    block_M = 64
    block_N = 16
    block_K_chunk = BLOCK_K_HALF * 2
    block_N_2 = block_N // VEC_NUM

    k_num = T.ceildiv(K_half, BLOCK_K_HALF)
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "int8"),  # type: ignore
        B_packed: T.Tensor((N, K_half), "uint8"),  # type: ignore
        workspace: T.Tensor((N, K), "int8"),  # type: ignore
        C: T.Tensor((M, N), "int32"),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num

            packed_ub = T.alloc_shared((BLOCK_K_HALF,), "uint8")
            half_ub = T.alloc_shared((BLOCK_K_HALF,), "float16")
            int16_ub = T.alloc_shared((BLOCK_K_HALF,), "int16")
            mask_ub = T.alloc_shared((BLOCK_K_HALF,), "int16")
            low_ub = T.alloc_shared((BLOCK_K_HALF,), "int16")
            high_ub = T.alloc_shared((BLOCK_K_HALF,), "int16")
            low_half_ub = T.alloc_shared((BLOCK_K_HALF,), "float16")
            high_half_ub = T.alloc_shared((BLOCK_K_HALF,), "float16")
            cmp_ub = T.alloc_shared((BLOCK_K_HALF,), "uint8")
            neg16_ub = T.alloc_shared((BLOCK_K_HALF,), "float16")
            zero_ub = T.alloc_shared((BLOCK_K_HALF,), "float16")
            adj_ub = T.alloc_shared((BLOCK_K_HALF,), "float16")
            low_int8_ub = T.alloc_shared((BLOCK_K_HALF,), "int8")
            high_int8_ub = T.alloc_shared((BLOCK_K_HALF,), "int8")
            output_ub = T.alloc_shared((BLOCK_K_HALF * 2,), "int8")

            for row in T.serial(block_N_2):
                actual_row = bn * block_N + vid * block_N_2 + row

                for k_chunk in T.serial(k_num):
                    chunk_offset = k_chunk * BLOCK_K_HALF

                    T.copy(B_packed[actual_row, chunk_offset], packed_ub)

                    T.tile.cast(half_ub, packed_ub, "CAST_NONE", BLOCK_K_HALF)
                    T.tile.cast(int16_ub, half_ub, "CAST_RINT", BLOCK_K_HALF)

                    T.tile.fill(mask_ub, 15)
                    T.tile.bitwise_and(low_ub, int16_ub, mask_ub)
                    T.tile.bitwise_rshift(high_ub, int16_ub, 4)
                    T.tile.bitwise_and(high_ub, high_ub, mask_ub)

                    T.tile.cast(low_half_ub, low_ub, "CAST_NONE", BLOCK_K_HALF)
                    T.tile.cast(high_half_ub, high_ub, "CAST_NONE", BLOCK_K_HALF)

                    T.tile.fill(neg16_ub, -16.0)
                    T.tile.fill(zero_ub, 0.0)

                    T.tile.compare(cmp_ub, low_half_ub, T.float16(8.0), "GE")
                    T.tile.select(adj_ub, cmp_ub, neg16_ub, zero_ub, "VSEL_CMPMASK_SPR")
                    T.tile.add(low_half_ub, low_half_ub, adj_ub)

                    T.tile.compare(cmp_ub, high_half_ub, T.float16(8.0), "GE")
                    T.tile.select(adj_ub, cmp_ub, neg16_ub, zero_ub, "VSEL_CMPMASK_SPR")
                    T.tile.add(high_half_ub, high_half_ub, adj_ub)

                    T.tile.cast(low_int8_ub, low_half_ub, "CAST_RINT", BLOCK_K_HALF)
                    T.tile.cast(high_int8_ub, high_half_ub, "CAST_RINT", BLOCK_K_HALF)

                    for j in T.serial(BLOCK_K_HALF):
                        output_ub[j * 2] = low_int8_ub[j]
                        output_ub[j * 2 + 1] = high_int8_ub[j]

                    T.copy(output_ub, workspace[actual_row, chunk_offset * 2])

            A_L1 = T.alloc_shared((block_M, block_K_chunk), "int8")
            B_L1 = T.alloc_shared((block_N, block_K_chunk), "int8")
            C_L0 = T.alloc_fragment((block_M, block_N), "int32")

            for k_chunk in T.serial(k_num):
                k_offset = k_chunk * BLOCK_K_HALF * 2

                T.copy(A[bm * block_M, k_offset], A_L1)
                T.copy(workspace[bn * block_N, k_offset], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k_chunk == 0))

            T.copy(C_L0, C[bm * block_M, bn * block_N])

    return main


def torch_convert(tensor):
    """Convert packed uint8 tensor to int8 tensor (from original CUDA implementation)

    Each uint8 contains 2 int4 values: low nibble at position 0, high nibble at position 1.
    """

    def _convert(val, pos):
        assert val.dtype == torch.uint8
        val = val.view(torch.int8)
        mask = (1 << 4) - 1
        i4_shifted = (val >> (pos * 4)) & mask
        i4 = (i4_shifted << 4) >> 4

        return i4.view(torch.int8)

    N = tensor.shape[0]
    K = tensor.shape[1]
    new_tensor = torch.empty(N, K * 2, dtype=torch.int8, device=tensor.device)
    for i in range(new_tensor.shape[0]):
        for j in range(new_tensor.shape[1]):
            new_tensor[i][j] = _convert(tensor[i][j // 2], j % 2)
    return new_tensor


def ref_program(A, qB):
    """Reference implementation from original CUDA example"""
    B = torch_convert(qB)
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.int32)
    return C.transpose(0, 1)


def generate_quantized_weight(N, K):
    K_half = K // 2
    B_int8 = torch.randint(-8, 8, (N, K), dtype=torch.int8)
    qB = torch.empty(N, K_half, dtype=torch.uint8)
    for i in range(N):
        for j in range(K_half):
            low = B_int8[i, j * 2].item() & 0x0F
            high = B_int8[i, j * 2 + 1].item() & 0x0F
            qB[i, j] = (high << 4) | low
    return qB, B_int8


def test(M, N, K):
    kernel = w4a8_gemm_cv(M, N, K)
    print(kernel.get_kernel_source())

    A_int8 = torch.randint(-8, 8, (M, K), dtype=torch.int8).npu()
    qB, B_int8_ref = generate_quantized_weight(N, K)
    qB = qB.npu()
    workspace = torch.zeros(N, K, dtype=torch.int8).npu()
    C_output = torch.zeros(M, N, dtype=torch.int32).npu()

    result = kernel(A_int8, qB, workspace, C_output)
    torch.npu.synchronize()

    expected = ref_program(A_int8.cpu(), qB.cpu())

    torch.testing.assert_close(result.cpu().transpose(0, 1), expected, rtol=0, atol=0)
    print(f"Test passed: M={M}, N={N}, K={K}")


def main():
    tilelang.disable_cache()  # Disable cache for testing correctness
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=0)
    parser.add_argument("--N", type=int, default=0)
    parser.add_argument("--K", type=int, default=0)
    args = parser.parse_args()

    # If user specified custom dimensions, run single test
    if args.M > 0 and args.N > 0 and args.K > 0:
        test(args.M, args.N, args.K)
    else:
        # Default: run three test levels
        # Level 0: K=256
        test(64, 64, 256)

        # Level 1: K=512
        # test(128, 128, 512)

        # Level 2: K=1024
        # test(256, 256, 1024)

    print("Test Passed!")


if __name__ == "__main__":
    main()
