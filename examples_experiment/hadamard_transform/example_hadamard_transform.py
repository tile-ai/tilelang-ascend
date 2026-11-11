import argparse
import math

import tilelang
from tilelang import language as T
import torch

tilelang.cache.clear_cache()


def is_pow_of_2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def hadamard_block_intra(b, n, block_size, dtype="float"):
    """
    Execute in-block butterfly operations (first log2(block_size) stages)
    Each block processes one chunk

    FP16/BF16 compute in FP32 (T.tile.cast in/out) to avoid half-precision
    scalar errors and BF16 vector add/sub hardware limitation.
    Ping-pong 2D UB avoids in-place overwrite during butterfly.
    """
    assert is_pow_of_2(n), "n must be a power of 2"
    assert is_pow_of_2(block_size), "block_size must be a power of 2"
    assert n % block_size == 0, "n must be divisible by block_size"

    log_block = int(math.log2(block_size))
    num_blocks_per_batch = n // block_size
    total_blocks = b * num_blocks_per_batch

    need_cast = dtype in ("float16", "bfloat16")
    compute_dtype = "float32" if need_cast else dtype

    @T.prim_func
    def main(A: T.Tensor((b, n), dtype), B: T.Tensor((b, n), dtype)):
        with T.Kernel(total_blocks, is_npu=True) as (cid, vid):
            if vid == 0:
                batch_id = cid // num_blocks_per_batch
                block_id_in_batch = cid % num_blocks_per_batch
                offset = block_id_in_batch * block_size

                data_ub = T.alloc_ub((block_size,), dtype)
                work_ub = T.alloc_ub((2, block_size), compute_dtype)

                T.copy(A[batch_id, offset : offset + block_size], data_ub)

                if need_cast:
                    T.tile.cast(work_ub[0, 0:block_size], data_ub, "CAST_NONE", block_size)
                else:
                    T.copy(data_ub, work_ub[0, 0:block_size])

                for stage in T.serial(log_block):
                    src_idx = stage % 2
                    dst_idx = 1 - src_idx
                    chunk_size = 1 << (stage + 1)
                    chunk_num = block_size // chunk_size
                    half = chunk_size // 2

                    for chunk_idx in T.serial(chunk_num):
                        base = chunk_idx * chunk_size
                        for k in T.serial(half):
                            a_val = work_ub[src_idx, base + k]
                            b_val = work_ub[src_idx, base + k + half]
                            work_ub[dst_idx, base + k] = a_val + b_val
                            work_ub[dst_idx, base + k + half] = a_val - b_val

                final_idx = log_block % 2
                if need_cast:
                    T.tile.cast(data_ub, work_ub[final_idx, 0:block_size], "CAST_RINT", block_size)
                else:
                    T.copy(work_ub[final_idx, 0:block_size], data_ub)

                T.copy(data_ub, B[batch_id, offset : offset + block_size])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def hadamard_cross_block_pair(b, n, block_size, cross_stage, dtype="float"):
    """
    Execute one level of cross-block butterfly operation (stage cross_stage)
    Handles butterfly with chunk_size = 2^(cross_stage+1)

    Cross-block butterfly is needed when cross_stage >= log2(block_size)

    Vectorised using T.tile.add/sub. FP16/BF16 compute in FP32.
    """
    assert is_pow_of_2(n), "n must be a power of 2"
    assert is_pow_of_2(block_size), "block_size must be a power of 2"
    assert n % block_size == 0, "n must be divisible by block_size"

    log_block = int(math.log2(block_size))
    assert cross_stage >= log_block, "cross_stage must >= log_block"

    chunk_size = 1 << (cross_stage + 1)
    half = chunk_size // 2

    num_chunks_per_batch = n // chunk_size
    total_chunks = b * num_chunks_per_batch

    need_cast = dtype in ("float16", "bfloat16")
    compute_dtype = "float32" if need_cast else dtype

    @T.prim_func
    def main(A: T.Tensor((b, n), dtype), B: T.Tensor((b, n), dtype)):
        with T.Kernel(total_chunks, is_npu=True) as (cid, vid):
            batch_id = cid // num_chunks_per_batch
            chunk_id_in_batch = cid % num_chunks_per_batch

            src_offset = chunk_id_in_batch * chunk_size
            dst_offset = src_offset + half

            if need_cast:
                data_ub = T.alloc_ub((half,), dtype)
                data2_ub = T.alloc_ub((half,), dtype)
                data_fp32 = T.alloc_ub((half,), compute_dtype)
                data2_fp32 = T.alloc_ub((half,), compute_dtype)
                add_fp32 = T.alloc_ub((half,), compute_dtype)
                sub_fp32 = T.alloc_ub((half,), compute_dtype)
                tmp_ub = T.alloc_ub((half,), dtype)

                T.copy(A[batch_id, src_offset : src_offset + half], data_ub)
                T.copy(A[batch_id, dst_offset : dst_offset + half], data2_ub)

                T.tile.cast(data_fp32, data_ub, "CAST_NONE", half)
                T.tile.cast(data2_fp32, data2_ub, "CAST_NONE", half)

                T.tile.add(add_fp32, data_fp32, data2_fp32)
                T.tile.sub(sub_fp32, data_fp32, data2_fp32)

                T.tile.cast(tmp_ub, add_fp32, "CAST_RINT", half)
                T.copy(tmp_ub, B[batch_id, src_offset : src_offset + half])

                T.tile.cast(tmp_ub, sub_fp32, "CAST_RINT", half)
                T.copy(tmp_ub, B[batch_id, dst_offset : dst_offset + half])
            else:
                data_ub = T.alloc_ub((half,), compute_dtype)
                data2_ub = T.alloc_ub((half,), compute_dtype)
                add_ub = T.alloc_ub((half,), compute_dtype)
                sub_ub = T.alloc_ub((half,), compute_dtype)

                T.copy(A[batch_id, src_offset : src_offset + half], data_ub)
                T.copy(A[batch_id, dst_offset : dst_offset + half], data2_ub)

                T.tile.add(add_ub, data_ub, data2_ub)
                T.tile.sub(sub_ub, data_ub, data2_ub)

                T.copy(add_ub, B[batch_id, src_offset : src_offset + half])
                T.copy(sub_ub, B[batch_id, dst_offset : dst_offset + half])

    return main


def hadamard_transform_complete(b, n, dtype="float", block_size=1024):
    """
    Complete Hadamard transform
    - n <= block_size: single kernel completes all butterfly operations
    - n > block_size: in-block butterfly + cross-block butterfly (host coordinated)
    """
    if not is_pow_of_2(n):
        raise ValueError(f"n={n} must be a power of 2")

    if n <= block_size:
        kernel = hadamard_block_intra(b, n, n, dtype)
        return lambda x: kernel(x)

    log_n = int(math.log2(n))
    log_block = int(math.log2(block_size))

    kernel_intra = hadamard_block_intra(b, n, block_size, dtype)

    cross_kernels = []
    for cross_stage in range(log_block, log_n):
        kernel_cross = hadamard_cross_block_pair(b, n, block_size, cross_stage, dtype)
        cross_kernels.append(kernel_cross)

    def full_transform(x):
        y = kernel_intra(x)

        for kernel_cross in cross_kernels:
            y = kernel_cross(y)

        return y

    return full_transform


def ref_hadamard(x: torch.Tensor):
    import scipy.linalg

    assert x.ndim == 2
    dim = x.shape[-1]
    assert is_pow_of_2(dim)
    H = torch.tensor(scipy.linalg.hadamard(dim, dtype=float), dtype=torch.float32, device=x.device)
    x_fp32 = x.to(torch.float32)
    return torch.nn.functional.linear(x_fp32, H).to(x.dtype)


def main():
    parser = argparse.ArgumentParser(description="Hadamard Transform on Ascend NPU (Complete)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--dim", type=int, default=2048, help="Dimension (must be power of 2)")
    parser.add_argument("--dtype", type=str, default="float", choices=["float", "float16", "bfloat16"], help="Data type")
    parser.add_argument("--block_size", type=int, default=2048, help="Block size for multi-block mode")
    args = parser.parse_args()

    B, N = args.batch, args.dim
    dtype = args.dtype
    block_size = args.block_size

    if not is_pow_of_2(N):
        print(f"Error: dim={N} must be a power of 2")
        return

    if not is_pow_of_2(block_size):
        print(f"Error: block_size={block_size} must be a power of 2")
        return

    if N % block_size != 0:
        print(f"Error: dim={N} must be divisible by block_size={block_size}")
        return

    torch_dtype = getattr(torch, dtype) if dtype != "float" else torch.float32

    print(f"Testing Hadamard transform with batch={B}, dim={N}, dtype={dtype}, block_size={block_size}")

    if block_size < N:
        log_n = int(math.log2(N))
        log_block = int(math.log2(block_size))
        print(f"Using multi-block mode: {N // block_size} blocks, {log_n} stages")
        print(f"  - Intra-block stages: {log_block}")
        print(f"  - Cross-block stages: {log_n - log_block}")

    transform = hadamard_transform_complete(B, N, dtype, block_size)

    x = torch.randn(B, N, dtype=torch_dtype).npu()
    torch.npu.synchronize()
    print("Input data initialized")

    y = transform(x)

    y_ref = ref_hadamard(x.cpu()).npu()
    if dtype == "bfloat16":
        rtol = 1e-1
        atol = 2.0
    elif dtype == "float16":
        rtol = 1e-2
        atol = 1e-1
    else:
        rtol = 1e-3
        atol = 1e-3
    torch.testing.assert_close(y.cpu(), y_ref.cpu(), rtol=rtol, atol=atol)
    print(f"Test passed! (rtol={rtol}, atol={atol})")


if __name__ == "__main__":
    main()
