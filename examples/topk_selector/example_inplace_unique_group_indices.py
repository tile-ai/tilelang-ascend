import argparse
import os
import torch
import tilelang
from tilelang import language as T

INDEX_DTYPE_INT64 = "int64"
INDEX_DTYPE_INT32 = "int32"

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tilelang.jit(pass_configs=pass_configs)
def get_inplace_unique_group_indices_kernel(num_topk: int, num_groups_aligned: int, num_sms: int):
    num_tokens = T.symbolic("num_tokens")
    num_aligned_topk = (num_topk + 31) // 32 * 32

    @T.prim_func
    def inplace_unique_group_indices_kernel(
        group_indices: T.Tensor[(num_tokens, num_topk), INDEX_DTYPE_INT64],
    ):
        with T.Kernel(num_tokens, is_npu=True) as (cid, _):
            group_indices_ub = T.alloc_ub((num_aligned_topk,), INDEX_DTYPE_INT64)
            seen_ub = T.alloc_ub((num_groups_aligned,), INDEX_DTYPE_INT32)

            for g in T.serial(num_groups_aligned):
                seen_ub[g] = 0

            T.copy(group_indices[cid, 0:num_topk], group_indices_ub[0:num_topk])

            for j in T.serial(num_topk):
                group_idx_int32 = T.cast(group_indices_ub[j], INDEX_DTYPE_INT32)
                if group_idx_int32 >= 0 and seen_ub[group_idx_int32] == 1:
                    group_indices_ub[j] = T.cast(-1, INDEX_DTYPE_INT64)
                elif group_idx_int32 >= 0:
                    seen_ub[group_idx_int32] = 1

            T.copy(group_indices_ub[0:num_topk], group_indices[cid, 0:num_topk])

    return inplace_unique_group_indices_kernel


def inplace_unique_group_indices(
    indices: torch.Tensor,
    n: int,
    num_ep_ranks: int,
    num_tp_ranks: int,
    tp_rank: int,
) -> torch.Tensor:
    num_groups = n

    assert indices.dim() == 2
    assert num_groups <= 128

    num_topk = indices.shape[1]
    num_groups_aligned = (num_groups + 63) // 64 * 64
    kernel = get_inplace_unique_group_indices_kernel(num_topk, num_groups_aligned, num_sms=1)

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    if indices.shape[0] > 0:
        kernel(indices)

    return indices


def ref_inplace_unique_group_indices(group_indices: torch.Tensor, num_groups: int) -> torch.Tensor:
    ref_indices = group_indices.clone()
    num_tokens, num_topk = ref_indices.shape

    vals, idx = torch.sort(ref_indices, dim=1, stable=True)

    first_in_sorted = torch.ones((num_tokens, num_topk), dtype=torch.bool, device=ref_indices.device)
    first_in_sorted[:, 1:] = vals[:, 1:] != vals[:, :-1]
    dup_in_sorted = ~first_in_sorted

    dup_in_orig = torch.zeros((num_tokens, num_topk), dtype=torch.bool, device=ref_indices.device)
    dup_in_orig.scatter_(1, idx, dup_in_sorted)

    ref_indices[dup_in_orig] = -1

    return ref_indices


def check_case(
    num_tokens: int,
    top_k: int,
    n: int,
    num_ep_ranks: int,
    num_tp_ranks: int,
    tp_rank: int,
):
    print(f"\n Testing Case: tokens={num_tokens}, top_k={top_k}, n={n}, ep={num_ep_ranks}, tp={num_tp_ranks}, tp_rank={tp_rank} ")

    indices = torch.randint(-1, n, (num_tokens, top_k), dtype=torch.int64).npu()

    indices_for_kernel = indices.clone()

    out_indices = inplace_unique_group_indices(indices_for_kernel, n, num_ep_ranks, num_tp_ranks, tp_rank)

    ref_indices = ref_inplace_unique_group_indices(indices, n)

    total = num_tokens * top_k
    matches = (out_indices == ref_indices).sum().item()
    accuracy = matches / total

    print(f"Total matches: {matches} / {total}")
    print(f"Accuracy: {accuracy}")

    assert matches == total, "TileLang kernel output does not match PyTorch NPU reference!"
    print("Kernel Check Passed!")


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="inplace_unique_group_indices Test Example")
    parser.add_argument("--tokens", type=int, default=1024, help="Number of tokens")
    parser.add_argument("--topk", type=int, default=8, help="Top-K experts")
    parser.add_argument("--n", type=int, default=32, help="Total number of groups (max 128)")
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

    print("\ninplace_unique_group_indices example passed!")
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
