import argparse
from typing import Literal
from collections import Counter

import tilelang as tl
import tilelang.language as T
import torch


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    },
)
def simple_topk_selector(B: int, N: int, top_k: int, block_N: int, dtype: Literal["float32"] = "float32"):

    VEC_NUM = 2
    INDEX_DTYPE = "int32"

    b_num = T.ceildiv(B, VEC_NUM)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        x: T.Tensor([B, N], dtype),  # type: ignore
        indices: T.Tensor([B, top_k], INDEX_DTYPE),  # type: ignore
    ):
        with T.Kernel(b_num, is_npu=True) as (cid, vid):
            row_id = (cid * VEC_NUM + vid) % B  # one v-core for one row

            x_ub = T.alloc_ub([block_N], dtype)
            score_accum = T.alloc_ub([N], dtype)
            topk_dst = T.alloc_ub([2 * top_k], dtype)
            topk_index = T.alloc_ub([top_k], dtype)
            output_index = T.alloc_ub([top_k], INDEX_DTYPE)

            # Accumulate all N scores into flat buffer
            for bn in T.serial(n_num):
                T.copy(x[row_id, bn * block_N], x_ub)
                for i in range(block_N):
                    score_accum[bn * block_N + i] = x_ub[i]

            # Sort all N elements and extract top_k
            T.tile.topk(topk_dst, score_accum, top_k, N)
            T.tile.gather_mask(topk_index, topk_dst, "P1010")  # [value, idx] => [idx]
            T.tile.cast(output_index, topk_index, "CAST_ROUND", top_k)

            T.copy(output_index, indices[row_id, :top_k])

    return main


def ref_program(x, top_k):
    return torch.topk(x, top_k, dim=-1)[1]


def count_per_row_mismatches(indices: torch.Tensor, ref_indices: torch.Tensor):
    row_mismatches = 0

    for i in range(indices.shape[0]):
        ref_indices_np = ref_indices[i].to(torch.int32).numpy()
        indices_np = indices[i].to(torch.int32).numpy()

        ref_indices_set = set(ref_indices_np)
        indices_set = set(indices_np)
        intersection = ref_indices_set & indices_set
        intersection_len = len(intersection)
        indices_len = len(indices_set)
        ref_indices_len = len(ref_indices_set)

        if intersection_len != ref_indices_len:
            row_mismatches += 1

        indices_ratio = intersection_len / indices_len
        ref_indices_ratio = intersection_len / ref_indices_len
        assert indices_ratio == ref_indices_ratio and indices_ratio > 0.99, (
            f"Row {i} check failed: {indices_ratio = }, {ref_indices_ratio = }; {row_mismatches = }"
        )
    return row_mismatches


def count_total_mismatches(indices: torch.Tensor, ref_indices: torch.Tensor):
    assert indices.shape[-1] == ref_indices.shape[-1], "the last dimension of two tensors must be the same"

    total_mismatches = 0

    for i in range(indices.shape[0]):
        indices_row = indices[i].tolist()
        ref_indices_row = ref_indices[i].tolist()

        indices_counter = Counter(indices_row)
        ref_indices_counter = Counter(ref_indices_row)

        diff = (indices_counter - ref_indices_counter) + (ref_indices_counter - indices_counter)
        total_mismatches += sum(diff.values())

    return total_mismatches


def check_case(B: int, N: int, top_k: int):
    x = torch.randn(B, N).to(torch.float32).npu()

    kernel = simple_topk_selector(B, N, top_k, top_k // 4)

    indices = kernel(x)
    ref_indices = ref_program(x, top_k)
    indices = indices.cpu()
    ref_indices = ref_indices.cpu()

    row_mismatches = count_per_row_mismatches(indices, ref_indices)
    print(f"Row matches: {B - row_mismatches} / {B}")
    total = B * top_k
    total_mismatches = count_total_mismatches(indices, ref_indices)
    accuracy = 1 - total_mismatches / total
    print(f"Total matches: {total - total_mismatches} / {total}")
    print(f"Accuracy: {accuracy}")
    assert accuracy > 0.99


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="topk_selector Simple Example")
    parser.add_argument("--b", type=int, default=64, help="Matrix dimension B")
    parser.add_argument("--n", type=int, default=4 * 1024, help="Matrix dimension N (max ~8192 for 192KB UB)")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}]", "Unknown args:", remains)
    B, N = args.b, args.n
    top_k = 2048

    torch.manual_seed(0)
    tl.cache.clear_cache()

    check_case(B, N, top_k)
    check_case(1024, 1024, 128)

    print("topk_selector example passed!")
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
