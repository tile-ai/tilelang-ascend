import torch
import argparse
import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()


def torch_grouped_gemm_bwd(A, B, block_metadata):
    """
    Golden function: compute dB_i = A_i^T @ B_i for each group
    """
    batch_count = A.shape[1]  # M dimension
    N = B.shape[1]
    M = A.shape[1]

    output = torch.empty((batch_count, M, N), device=A.device, dtype=A.dtype)

    metadata_cpu = block_metadata.cpu().tolist()
    for row in metadata_cpu:
        batch_idx = int(row[0])
        m_block_idx = int(row[1])
        n_block_idx = int(row[2])
        batch_offset = int(row[3])
        k_iters = int(row[4])

        # Skip invalid blocks
        if k_iters == 0:
            continue

        m_start = m_block_idx * 64
        n_start = n_block_idx * 64

        # Compute: C[batch_idx, m_start:m_start+64, n_start:n_start+64] = A[batch_offset:batch_offset+k_iters*64, m_start:m_start+64]^T @ B[batch_offset:batch_offset+k_iters*64, n_start:n_start+64]
        A_i = A[batch_offset : batch_offset + k_iters * 64, m_start : m_start + 64]
        B_i = B[batch_offset : batch_offset + k_iters * 64, n_start : n_start + 64]
        output[batch_idx, m_start : m_start + 64, n_start : n_start + 64] = torch.mm(A_i.T, B_i)

    return output


@tilelang.jit(out_idx=[2], target="pto")
def grouped_gemm_bwd(batch_sum, batch_count, M, N, max_k_iters, block_M, block_N, block_K, total_blocks, dtype="float16"):
    """
    Grouped GEMM Backward with static loop bounds
    Using block_metadata mechanism similar to grouped_gemm_fwd
    """
    accum_dtype = "float32"

    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, M], dtype),
        B: T.Tensor([batch_sum, N], dtype),
        C: T.Tensor([batch_count, M, N], dtype),
        block_metadata: T.Tensor([total_blocks, 5], "int32"),
    ):
        # block_metadata columns: [batch_idx, m_block_idx, n_block_idx, batch_offset, k_iters]
        with T.Kernel(total_blocks, is_npu=True) as (cid, _):
            cur_batch_idx = block_metadata[cid, 0]
            m_block_idx = block_metadata[cid, 1]
            n_block_idx = block_metadata[cid, 2]
            batch_offset = block_metadata[cid, 3]
            k_iters = block_metadata[cid, 4]

            A_L1 = T.alloc_L1([block_K, block_M], dtype)
            B_L1 = T.alloc_L1([block_K, block_N], dtype)
            C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)

            with T.Scope("C"):
                for i, j in T.Parallel(block_M, block_N):
                    C_L0[i, j] = 0

                for k in T.serial(max_k_iters):
                    # Only compute if k < k_iters for this block
                    if k < k_iters:
                        T.copy(
                            A[batch_offset + k * block_K, m_block_idx * block_M],
                            A_L1,
                        )
                        T.copy(
                            B[batch_offset + k * block_K, n_block_idx * block_N],
                            B_L1,
                        )
                        T.barrier_all()
                        T.gemm_v0(A_L1, B_L1, C_L0, transpose_A=True, init=(k == 0))
                        T.barrier_all()

                T.copy(C_L0, C[cur_batch_idx, m_block_idx * block_M, n_block_idx * block_N])

    return kernel


def construct_metadata(batch_sizes_list, M, N, block_M, block_N, block_K, device):
    """
    Construct block metadata for grouped_gemm_bwd
    Returns: metadata tensor and total_blocks count
    """
    m_num = M // block_M
    n_num = N // block_N

    metadata_list = []
    batch_offset = 0

    for batch_idx, batch_size in enumerate(batch_sizes_list):
        k_iters = batch_size // block_K
        for m_block in range(m_num):
            for n_block in range(n_num):
                metadata_list.append(
                    [
                        batch_idx,
                        m_block,
                        n_block,
                        batch_offset,
                        k_iters,
                    ]
                )
        batch_offset += batch_size

    total_blocks = len(metadata_list)
    block_metadata = torch.tensor(metadata_list, device=device, dtype=torch.int32)
    return block_metadata, total_blocks


def run_tilelang_grouped_gemm_bwd(batch_sizes_list, M, N, block_M, block_N, block_K):
    device = torch.device("npu")
    dtype = torch.float16

    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)

    # Ensure divisibility
    assert M % block_M == 0, f"M={M} must be divisible by block_M={block_M}"
    assert N % block_N == 0, f"N={N} must be divisible by block_N={block_N}"
    for size in batch_sizes_list:
        assert size % block_K == 0, f"batch_size={size} must be divisible by block_K={block_K}"

    block_metadata, total_blocks = construct_metadata(batch_sizes_list, M, N, block_M, block_N, block_K, device)

    max_k_iters = max(size // block_K for size in batch_sizes_list)

    A = torch.randn(batch_sum, M, device=device, dtype=dtype)
    B = torch.randn(batch_sum, N, device=device, dtype=dtype)

    kernel = grouped_gemm_bwd(batch_sum, batch_count, M, N, max_k_iters, block_M, block_N, block_K, total_blocks)

    out = kernel(A, B, block_metadata)

    # Reference implementation
    ref_output = torch.empty((batch_count, M, N), device=device, dtype=dtype)
    start = 0
    for i, size in enumerate(batch_sizes_list):
        end = start + size
        A_i = A[start:end]
        B_i = B[start:end]
        ref_output[i] = torch.mm(A_i.T, B_i)
        start = end

    if torch.allclose(out, ref_output, rtol=1e-2, atol=1e-2):
        print(f"✅ TileLang and Torch match (batch_sizes={batch_sizes_list}, M={M}, N={N})")
    else:
        print(f"❌ TileLang and Torch mismatch (batch_sizes={batch_sizes_list}, M={M}, N={N})")
        max_diff = torch.max(torch.abs(out - ref_output)).item()
        print(f"   Max difference: {max_diff}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", type=str, default="64, 128", help="comma-separated batch sizes")
    parser.add_argument("--M", type=int, default=512, help="activation dimension (M)")
    parser.add_argument("--N", type=int, default=512, help="output dimension (N)")
    parser.add_argument("--block_M", type=int, default=64, help="M direction block size")
    parser.add_argument("--block_N", type=int, default=64, help="N direction block size")
    parser.add_argument("--block_K", type=int, default=64, help="batch direction block size")
    args = parser.parse_args()

    batch_sizes_list = [int(x) for x in args.batch_sizes.split(",")]
    M, N = args.M, args.N
    block_M, block_N, block_K = args.block_M, args.block_N, args.block_K

    run_tilelang_grouped_gemm_bwd(batch_sizes_list, M, N, block_M, block_N, block_K)
