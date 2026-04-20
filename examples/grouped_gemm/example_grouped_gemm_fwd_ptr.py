import argparse
import math

import torch

import tilelang
import tilelang.language as T


def torch_grouped_gemm(a_list, b_list):
    """PyTorch reference: grouped GEMM with separate tensors per group."""
    assert len(a_list) == len(b_list), "A/B group count mismatch"
    outputs = []
    for a, b in zip(a_list, b_list):
        assert a.shape[1] == b.shape[0], "incompatible GEMM shapes"
        outputs.append(torch.matmul(a, b))
        # outputs.append(torch.zeros_like(torch.matmul(a, b)))
    return outputs


@tilelang.jit(out_idx=[2])
def grouped_gemm_fwd_ptr(batch_sizes_list, K, N, block_M, block_N, block_K, dtype="float16"):
    padded_sizes = [math.ceil(s / block_M) * block_M for s in batch_sizes_list]
    batch_sum_padded = sum(padded_sizes)
    batch_count = len(batch_sizes_list)
    total_m_blocks = sum(math.ceil(s / block_M) for s in batch_sizes_list)
    n_num = math.ceil(N / block_N)
    accum_dtype = "float32"

    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum_padded, K], dtype),
        B: T.Tensor([batch_count, K, N], dtype),
        C: T.Tensor([batch_sum_padded, N], dtype),
        block_metadata: T.Tensor([total_m_blocks, 3], "int32"),
    ):
        with T.Kernel(total_m_blocks * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            cur_batch_idx = block_metadata[bx, 0]
            m_start = block_metadata[bx, 1]

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], A_L1)
                    T.copy(B[cur_batch_idx, k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()

                T.copy(C_L0, C[m_start : m_start + block_M, by * block_N : (by + 1) * block_N])

    return kernel


def construct_inputs(batch_sizes_list, K, N, block_M, device, dtype):
    padded_sizes = [math.ceil(s / block_M) * block_M for s in batch_sizes_list]
    batch_sum_padded = sum(padded_sizes)

    padded_offsets = [0]
    for ps in padded_sizes[:-1]:
        padded_offsets.append(padded_offsets[-1] + ps)

    A = torch.zeros(batch_sum_padded, K, device=device, dtype=dtype)
    for i, size in enumerate(batch_sizes_list):
        A[padded_offsets[i] : padded_offsets[i] + size].copy_(torch.randn(size, K, device=device, dtype=dtype))

    b_list = [torch.randn(K, N, device=device, dtype=dtype) for _ in batch_sizes_list]
    B = torch.stack(b_list, dim=0)
    C = torch.zeros(batch_sum_padded, N, device=device, dtype=dtype)

    metadata_list = []
    for batch_idx, size in enumerate(batch_sizes_list):
        for i in range(math.ceil(size / block_M)):
            metadata_list.append(
                [
                    batch_idx,
                    padded_offsets[batch_idx] + i * block_M,
                    min(block_M, size - i * block_M),
                ]
            )
    block_metadata = torch.tensor(metadata_list, device=device, dtype=torch.int32)

    return A, B, C, block_metadata, b_list, padded_offsets


def verify_outputs(C, refs, batch_sizes_list, padded_offsets, atol=1e-3, rtol=1e-3):
    for idx, (ref, size) in enumerate(zip(refs, batch_sizes_list)):
        out_slice = C[padded_offsets[idx] : padded_offsets[idx] + size]
        try:
            torch.testing.assert_close(out_slice, ref, atol=atol, rtol=rtol)
        except AssertionError as err:
            raise AssertionError(f"group {idx}: {err}") from err


def benchmark(kernel, inputs, warmup=10, rep=30):
    for _ in range(warmup):
        kernel(*inputs)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        kernel(*inputs)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / rep


def run_tilelang_grouped_gemm_fwd_ptr(
    batch_sizes_list,
    K,
    N,
    block_M,
    block_N,
    block_K,
    profile=False,
):
    device = torch.device("npu")
    dtype = torch.float16

    kernel = grouped_gemm_fwd_ptr(tuple(batch_sizes_list), K, N, block_M, block_N, block_K)

    A, B, C, block_metadata, b_list, padded_offsets = construct_inputs(batch_sizes_list, K, N, block_M, device, dtype)

    a_list_valid = [A[padded_offsets[i] : padded_offsets[i] + s] for i, s in enumerate(batch_sizes_list)]
    refs = torch_grouped_gemm(a_list_valid, b_list)

    out = kernel(A, B, block_metadata)
    verify_outputs(out, refs, batch_sizes_list, padded_offsets)
    print("Kernel Output Match!")

    if profile:
        latency = benchmark(kernel, (A, B, block_metadata))
        total_flops = sum(s * K * N * 2 for s in batch_sizes_list)
        print(f"Latency: {latency:.4f} ms")
        print(f"TFlops: {total_flops / (latency * 1e9):.4f}")


def test_grouped_gemm_fwd_ptr():
    run_tilelang_grouped_gemm_fwd_ptr([16, 33, 96], 128, 96, 32, 32, 32)
    run_tilelang_grouped_gemm_fwd_ptr([16, 64, 128], 128, 96, 32, 32, 32)
    run_tilelang_grouped_gemm_fwd_ptr([29, 57, 101], 128, 96, 32, 32, 32)
    run_tilelang_grouped_gemm_fwd_ptr([100, 200, 300], 128, 96, 32, 32, 32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", type=str, default="64,128,256", help="comma-separated per-group M sizes")
    parser.add_argument("--K", type=int, default=4096, help="reduce dim")
    parser.add_argument("--N", type=int, default=4096, help="output dim")
    parser.add_argument("--profile", action="store_true", help="benchmark the kernel")
    args = parser.parse_args()

    batch_sizes_list = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    block_M = 64
    block_N = 128
    block_K = 64

    run_tilelang_grouped_gemm_fwd_ptr(batch_sizes_list, args.K, args.N, block_M, block_N, block_K, profile=args.profile)

    # test_grouped_gemm_fwd_ptr()
