import argparse
import os
import torch
import tilelang
import tilelang.language as T

# Constants for data types
INT32_DTYPE = "int32"
INT64_DTYPE = "int64"

# Clear cache to force recompilation or load latest kernel
tilelang.cache.clear_cache()

# Compilation optimization configs for Ascend NPU
pass_configs = {
    tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tilelang.jit(pass_configs=pass_configs)
def get_mask_indices_by_tp_kernel(num_tokens: int, top_k: int):
    """Generate Ascend operator kernel for given shape."""
    # Align top_k to 32 for better Vector instruction handling
    num_aligned_top_k = (top_k + 31) // 32 * 32

    @T.prim_func
    def mask_indices_by_tp_kernel(
        indices: T.Tensor[(num_tokens, top_k), INT64_DTYPE],
        masked_indices: T.Tensor[(num_tokens, top_k), INT64_DTYPE],
        per_npu: T.int32,
        per_dp: T.int32,
        num_tp_ranks: T.int32,
        tp_rank: T.int32,
    ):
        # is_npu=True indicates this kernel runs on Ascend NPU
        with T.Kernel(num_tokens, is_npu=True) as (cid, _):
            # Allocate Unified Buffer (UB)
            indices_ub = T.alloc_ub((num_aligned_top_k,), INT64_DTYPE)
            masked_ub = T.alloc_ub((num_aligned_top_k,), INT64_DTYPE)

            # Copy data from global memory to UB
            T.copy(indices[cid, 0:top_k], indices_ub[0:top_k])

            for j in T.serial(top_k):
                value = indices_ub[j]
                value_int32 = T.cast(value, INT32_DTYPE)

                # Mask experts not belonging to current TP rank
                if value_int32 < 0:
                    masked_ub[j] = T.cast(-1, INT64_DTYPE)
                else:
                    tp_group = T.truncdiv(value_int32, per_npu)
                    rank_in_tp = T.truncmod(tp_group, num_tp_ranks)
                    if rank_in_tp != tp_rank:
                        masked_ub[j] = T.cast(-1, INT64_DTYPE)
                    else:
                        local_value = value_int32 - tp_rank * per_npu
                        dp_rank = T.truncdiv(local_value, per_dp)
                        remapped = local_value - dp_rank * (per_dp - per_npu)
                        # Use T.Select to avoid nested if
                        masked_ub[j] = T.Select(
                            remapped < 0,
                            T.cast(-1, INT64_DTYPE),
                            T.cast(remapped, INT64_DTYPE),
                        )

            # Write back results from UB to global memory
            T.copy(masked_ub[0:top_k], masked_indices[cid, 0:top_k])

    return mask_indices_by_tp_kernel


def mask_indices_by_tp(
    indices: torch.Tensor,
    n: int,
    num_ep_ranks: int,
    num_tp_ranks: int,
    tp_rank: int,
) -> torch.Tensor:
    """Forward wrapper to invoke TileLang kernel."""
    num_tokens, top_k = indices.shape
    per_npu = n // num_ep_ranks
    per_dp = num_tp_ranks * per_npu

    if not indices.is_contiguous():
        indices = indices.contiguous()

    kernel = get_mask_indices_by_tp_kernel(num_tokens=num_tokens, top_k=top_k)

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    masked_indices = torch.empty((num_tokens, top_k), dtype=indices.dtype, device=indices.device)
    if num_tokens > 0:
        kernel(indices, masked_indices, per_npu, per_dp, num_tp_ranks, tp_rank)

    return masked_indices


def ref_mask_indices_by_tp(
    indices: torch.Tensor,
    n: int,
    num_ep_ranks: int,
    tp_rank: int,
    num_tp_ranks: int,
) -> torch.Tensor:
    """CPU reference implementation to avoid NPU internal format warnings."""
    orig_device = indices.device
    value_cpu = indices.cpu()

    per_npu = n // num_ep_ranks
    per_dp = num_tp_ranks * per_npu

    invalid = (value_cpu < 0) | ((value_cpu // per_npu) % num_tp_ranks != tp_rank)

    value_cpu = value_cpu - tp_rank * per_npu
    dp_rank = value_cpu // per_dp
    value_cpu = value_cpu - dp_rank * (per_dp - per_npu)

    value_cpu[invalid | (value_cpu < 0)] = -1

    return value_cpu.to(orig_device)


def check_case(
    num_tokens: int,
    top_k: int,
    n: int,
    num_ep_ranks: int,
    num_tp_ranks: int,
    tp_rank: int,
):
    """Test and compare custom kernel against PyTorch reference."""
    print(f"\n Testing Case: tokens={num_tokens}, top_k={top_k}, n={n}, ep={num_ep_ranks}, tp={num_tp_ranks}, tp_rank={tp_rank} ")

    indices = torch.randint(-1, n, (num_tokens, top_k), dtype=torch.int64).npu()

    out_indices = mask_indices_by_tp(indices, n, num_ep_ranks, num_tp_ranks, tp_rank)
    ref_indices = ref_mask_indices_by_tp(indices, n, num_ep_ranks, tp_rank, num_tp_ranks)

    out_indices_cpu = out_indices.cpu()
    ref_indices_cpu = ref_indices.cpu()

    total = num_tokens * top_k
    matches = (out_indices_cpu == ref_indices_cpu).sum().item()
    accuracy = matches / total

    print(f"[Forward] Total matches: {matches} / {total}")
    print(f"[Forward] Accuracy: {accuracy}")

    assert matches == total, "TileLang kernel output does not match PyTorch reference!"
    print("Kernel Check Passed!")


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="mask_indices_by_tp Test Example")
    parser.add_argument("--tokens", type=int, default=1024, help="Number of tokens")
    parser.add_argument("--topk", type=int, default=8, help="Top-K experts")
    parser.add_argument("--n", type=int, default=16, help="Total number of experts")
    parser.add_argument("--ep", type=int, default=2, help="Expert Parallel size")
    parser.add_argument("--tp", type=int, default=4, help="Tensor Parallel size")
    parser.add_argument("--rank", type=int, default=1, help="Current TP rank")

    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}] Unknown args:", remains)

    torch.manual_seed(0)

    check_case(args.tokens, args.topk, args.n, args.ep, args.tp, args.rank)

    check_case(4096, 4, 32, 4, 4, 0)
    check_case(8192, 2, 8, 1, 8, 3)
    check_case(128, 16, 64, 8, 2, 1)

    print("\nmask_indices_by_tp example passed!")
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
