import argparse
from typing import Literal

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
    }
)
def simple_topk_selector(B:int, N:int, top_k:int, block_N:int, dtype: Literal["float32"] = "float32"):
    """Simple TopK implementation"""
    
    VEC_NUM = 2
    INDEX_DTYPE = "int32"
    SORT_INDEX_DTYPE = "uint32"

    n_num = T.ceildiv(N, block_N)
    merge_num = T.ceildiv(top_k, block_N)
    repeat_time = T.ceildiv(block_N, 32)

    assert merge_num == 4

    @T.prim_func
    def main(
        x: T.Tensor([B, N], dtype),                     # type: ignore
        indices: T.Tensor([B, top_k], INDEX_DTYPE),     # type: ignore
    ):
        with T.Kernel(B, is_npu=True) as (cid, vid):
            # bb = (cid * VEC_NUM + vid) % b_2_num

            # sort_temp = T.alloc_ub([block_B, block_N], dtype)
            # sort_temp = T.alloc_ub([merge_num, block_N * 2], dtype)
            sort_temp = T.alloc_ub([32, block_N], dtype)
            x_ub = T.alloc_ub([block_N], dtype)

            sort_indices = T.alloc_ub([block_N], INDEX_DTYPE)
            sort_indices_u = T.alloc_ub([block_N], SORT_INDEX_DTYPE)

            sort_result = T.alloc_ub([merge_num, block_N * 2], dtype)
            sort_result_index = T.alloc_ub([top_k], INDEX_DTYPE)

            topk_global = T.alloc_ub([top_k * 2], dtype)

            T.annotate_address({
                # ub address
                sort_temp: 0,
                x_ub: 65536,
                sort_indices: 67584,
                sort_indices_u: 67584,
                sort_result: 69632,
                sort_result_index: 69632,
                topk_global: 86016,
            })

            T.tile.arith_progression(sort_indices, 0, 1, block_N)
            
            T.tile.init_sort_buf(topk_global, top_k * 2, rsv=0)  # rsv is always 0

            for bn in T.serial(n_num):
                T.copy(x[cid, bn * block_N], x_ub)

                T.tile.sort(sort_result[(bn % merge_num), :], x_ub, sort_indices_u, sort_temp, repeat_time)
                if bn % merge_num == merge_num - 1:
                    if bn == merge_num - 1:  # first time merge
                        T.tile.merge_sort(topk_global, sort_result, block_N, merge_num, is_copy=0)
                    else:  # later merges
                        T.tile.merge_sort(sort_temp, sort_result, block_N, merge_num, is_copy=1)
                        T.tile.topk(topk_global, sort_result, sort_temp, top_k)
                
                T.tile.add(sort_indices, sort_indices, T.int32(block_N))

            T.tile.gather_mask(sort_result, topk_global, top_k)
            # T.tile.cast_tl(sort_result_index, sort_result, "CAST_RINT", top_k)
            T.copy(sort_result_index, indices[cid, :top_k])
    
    return main

def ref_program(x, top_k):
    return torch.topk(x, top_k, dim=-1)[1]

def check_case(B:int, N:int, top_k:int, block_N:int, dtype="float32"):
    torch_dtype_map = {"float32": torch.float32, "float": torch.float32}
    x = torch.randn(B, N).to(torch_dtype_map[dtype]).npu()

    kernel = simple_topk_selector(B, N, top_k, block_N, dtype=dtype)

    indices = kernel(x)
    ref_indices = ref_program(x, top_k)
    
    for i in range(B):
        ref_indices_np = ref_indices[i].cpu().to(torch.int32).numpy()
        indices_np = indices[i].cpu().to(torch.int32).numpy()

        ref_indices_set = set(ref_indices_np)
        indices_set = set(indices_np)
        intersection = ref_indices_set & indices_set
        intersection_len = len(intersection)
        indices_len = len(indices_set)
        ref_indices_len = len(ref_indices_set)
        assert (intersection_len / indices_len == intersection_len / ref_indices_len) and (intersection_len / indices_len > 0.99)

def main(custom_args=None):
    parser = argparse.ArgumentParser(description="topk_selector Example")
    parser.add_argument("--b", type=int, default=64, help="Matrix dimension B")
    parser.add_argument("--n", type=int, default=32 * 1024, help="Matrix dimension N")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}]", "Unknown args:", remains)
    B, N = args.b, args.n
    top_k = 2048

    torch.manual_seed(0)
    tl.cache.clear_cache()

    check_case(B, N, top_k, top_k // 4)

    print("topk_selector example passed!")
    print("Kernel Output Match!")

if __name__ == "__main__":
    main()