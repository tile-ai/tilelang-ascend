import torch
import argparse
import tilelang
import tilelang.language as T


def torch_gmm(a, b, block_metadata, trans_b=False):
    """
    Perform grouped matrix multiplication using PyTorch.

    Args:
        a (torch.Tensor): Input tensor of shape (N, K).
        b (torch.Tensor): Input tensor of shape (G, K, M).
        block_metadata (torch.Tensor): A metadata tensor of shape (Total_Blocks, 3) with dtype int32.

    Returns:
        torch.Tensor: The output tensor of shape (Total_M, N).
    """
    total_m = a.shape[0]

    n_dim = b.shape[2] if not trans_b else b.shape[1]

    output = torch.empty((total_m, n_dim), device=a.device, dtype=a.dtype)

    metadata_cpu = block_metadata.cpu().tolist()
    for row in metadata_cpu:
        batch_idx = int(row[0])
        m_start = int(row[1])
        valid_m = int(row[2])

        part_a = a[m_start : m_start + valid_m, :]

        part_b = b[batch_idx]
        if trans_b:
            part_b = part_b.transpose(0, 1)

        part_out = torch.mm(part_a, part_b)

        output[m_start : m_start + valid_m, :] = part_out

    return output


@tilelang.jit(out_idx=[2])
def grouped_gemm(batch_sizes_list, K, N, block_M, block_N, block_K, dtype="float16"):
    """
    Kernel function

    args:
        a (torch.Tensor): Input tensor of shape (M, K).
        b (torch.Tensor): Input tensor of shape (G, K, N).
        block_metadata (torch.Tensor): A metadata tensor of shape (Total_Blocks, 3) with dtype int32.
    """
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    accum_dtype = "float32"
    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)
    n_num = (N + block_N - 1) // block_N

    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, K], dtype),  # type: ignore
        B: T.Tensor([batch_count, K, N], dtype),  # type: ignore
        C: T.Tensor([batch_sum, N], dtype),  # type: ignore
        # Metadata table: [batch_idx, m_start_offset, valid_rows]
        block_metadata: T.Tensor([total_m_blocks, 3], "int32"),  # type: ignore
    ):
        with T.Kernel(total_m_blocks * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            cur_batch_idx = block_metadata[bx, 0]
            m_start = block_metadata[bx, 1]
            # Partial memory movement (tail handling) is not supported yet;
            # this variable is currently unused.
            _actual_rows = block_metadata[bx, 2]

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    # Copyin
                    T.copy(
                        A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K],
                        A_L1,
                    )
                    T.copy(
                        B[
                            cur_batch_idx,
                            k * block_K : (k + 1) * block_K,
                            by * block_N : (by + 1) * block_N,
                        ],
                        B_L1,
                    )
                    T.barrier_all()

                    # Compute
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()

                # Copyout
                T.copy(
                    C_L0,
                    C[
                        m_start : m_start + block_M,
                        by * block_N : by * block_N + block_N,
                    ],
                )

    return kernel


def construct_inputs(batch_sizes_list, K, M, trans_b, block_M, device, dtype):
    """
    Constructs input tensors and block metadata for Grouped GEMM.

    Args:
        batch_sizes_list (list[int]): A list containing the number of rows (M-dimension) for each batch.
        K (int): The size of the reduction dimension.
        M (int): The size of the output dimension (often denoted as N in standard GEMM).
        trans_b (bool): Whether the weight tensor B is transposed.
        block_M (int): The tile size for the M-dimension used in the kernel.
        device (torch.device): The device to allocate tensors on (e.g., "cuda", "npu").
        dtype (torch.dtype): The data type for input tensors (e.g., torch.float16).

    Returns:
        tuple:
            - A (torch.Tensor): Stacked input tensor of shape (Sum(batch_sizes), K).
            - B (torch.Tensor): Batched weight tensor of shape (Batch_Count, K, M).
            - block_metadata (torch.Tensor): Metadata table of shape (Total_Blocks, 3), int32.
                - Column 0: batch_idx (The index of the weight matrix in `b` to use).
                - Column 1: m_start (The global starting row index in `a`).
                - Column 2: valid_m (The number of valid rows in this block).
    """
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)

    A = torch.randn(batch_sum, K, device=device, dtype=dtype)
    B = torch.randn(batch_count, K, M, device=device, dtype=dtype)

    metadata_list = []
    current_global_offset = 0

    for batch_idx, size in enumerate(batch_sizes_list):
        num_blocks = (size + block_M - 1) // block_M

        for i in range(num_blocks):
            local_start = i * block_M

            m_start_global = current_global_offset + local_start

            valid_m = min(block_M, size - local_start)

            metadata_list.append([batch_idx, m_start_global, valid_m])

        current_global_offset += size

    block_metadata = torch.tensor(metadata_list, device=device, dtype=torch.int32)

    return A, B, block_metadata


def run_tilelang_grouped_gemm(
    batch_sizes_list,
    K,
    M,
    block_M,
    block_N,
    block_K,
    trans_b,
):
    kernel = grouped_gemm(tuple(batch_sizes_list), K, M, block_M, block_N, block_K)
    # print(kernel.get_kernel_source())

    device = torch.device("npu")
    dtype = torch.float16

    A, B, block_metadata = construct_inputs(
        batch_sizes_list, K, M, trans_b, block_M, device, dtype
    )
    out = kernel(A, B, block_metadata)
    ref_output = torch_gmm(A, B, block_metadata, trans_b)
    # print(out)
    # print(ref_output)
    if torch.allclose(out, ref_output, rtol=0.01, atol=0.01):
        print("Kernel Output Match!")
    else:
        print("Kernel Output Mismatch!")


def test_grouped_gemm():
    run_tilelang_grouped_gemm([64], 64, 64, 64, 64, 64, False)
    run_tilelang_grouped_gemm([64], 8192, 8192, 64, 64, 64, False)
    run_tilelang_grouped_gemm([64, 128, 256], 8192, 8192, 64, 64, 64, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_sizes", type=str, default="64, 128", help="comma-separated batch sizes"
    )
    parser.add_argument("--K", type=int, default=8192, help="reduce dim")
    parser.add_argument("--M", type=int, default=8192, help="output dim")
    parser.add_argument("--trans_b", action="store_true", help="transpose B")
    args = parser.parse_args()

    batch_sizes_list = [int(x) for x in args.batch_sizes.split(",")]
    K, M, trans_b = args.K, args.M, args.trans_b

    block_M = 64
    block_N = 128
    block_K = 64

    run_tilelang_grouped_gemm(
        batch_sizes_list,
        K,
        M,
        block_M,
        block_N,
        block_K,
        trans_b,
    )

    # test_grouped_gemm()