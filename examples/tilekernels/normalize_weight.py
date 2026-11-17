import os
from importlib.util import find_spec

import torch
import tilelang
import tilelang.language as T


tilelang.cache.clear_cache()

os.environ["TILELANG_PRINT_ON_COMPILATION"] = "0"

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(pass_configs=pass_configs)
def get_normalize_weight_kernel(
    num_topk: int,
):
    num_tokens = T.symbolic("num_tokens")
    vec_num = 2
    rows_per_vec = 128
    tokens_per_block = rows_per_vec * vec_num
    num_token_blocks = T.ceildiv(num_tokens, tokens_per_block)
    num_cores = 24
    num_iters = T.ceildiv(num_token_blocks, num_cores)
    stages = 2
    mte3_store_flag = 2
    aligned_topk = ((num_topk + 7) // 8) * 8

    @T.prim_func
    def normalize_weight_kernel(
        topk_weights: T.Tensor((num_tokens, num_topk), "float"),
        denominator: T.Tensor((num_tokens,), "float"),
        normalized_weights: T.Tensor((num_tokens, num_topk), "float"),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid), T.Scope("V"):
            weights_ub = T.alloc_ub((stages, rows_per_vec, aligned_topk), "float")
            denom_ub = T.alloc_ub((rows_per_vec, 1), "float")
            denom_broadcast_ub = T.alloc_ub((rows_per_vec, aligned_topk), "float")
            normalized_ub = T.alloc_ub((rows_per_vec, aligned_topk), "float")
            block_id = T.alloc_var("int", init=0)
            next_block_id = T.alloc_var("int", init=0)
            token_start = T.alloc_var("int", init=0)
            next_token_start = T.alloc_var("int", init=0)

            for stage in T.serial(stages):
                T.set_flag("v", "mte2", stage)
            T.set_flag("mte3", "v", mte3_store_flag)

            if cid < num_token_blocks:
                token_start = cid * tokens_per_block + vid * rows_per_vec
                T.wait_flag("v", "mte2", 0)
                T.copy(
                    topk_weights[token_start : token_start + rows_per_vec, 0:num_topk],
                    weights_ub[0, :, :],
                    pad_value=0.0,
                )
                T.set_flag("mte2", "v", 0)

            for i in T.serial(num_iters):
                cur = i % stages
                nxt = (i + 1) % stages
                block_id = cid + i * num_cores
                token_start = block_id * tokens_per_block + vid * rows_per_vec
                if block_id < num_token_blocks:
                    next_block_id = cid + (i + 1) * num_cores
                    if next_block_id < num_token_blocks:
                        next_token_start = next_block_id * tokens_per_block + vid * rows_per_vec
                        T.wait_flag("v", "mte2", nxt)
                        T.copy(
                            topk_weights[next_token_start : next_token_start + rows_per_vec, 0:num_topk],
                            weights_ub[nxt, :, :],
                            pad_value=0.0,
                        )
                        T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    T.wait_flag("mte3", "v", mte3_store_flag)

                    T.reduce_sum(weights_ub[cur, :, :], denom_ub, dim=-1)
                    T.tile.add(denom_ub, denom_ub, 1.0e-20)

                    T.tile.broadcast(denom_broadcast_ub, denom_ub, axis=1)
                    T.tile.div(normalized_ub, weights_ub[cur, :, :], denom_broadcast_ub)

                    T.set_flag("v", "mte3", cur)
                    T.wait_flag("v", "mte3", cur)
                    T.copy(
                        normalized_ub[:, :num_topk],
                        normalized_weights[token_start : token_start + rows_per_vec, :],
                    )
                    T.copy(denom_ub[:, 0], denominator[token_start : token_start + rows_per_vec])
                    T.set_flag("mte3", "v", mte3_store_flag)
                    T.set_flag("v", "mte2", cur)

            T.wait_flag("v", "mte2", 0)
            T.wait_flag("v", "mte2", 1)
            T.wait_flag("mte3", "v", mte3_store_flag)

    return normalize_weight_kernel


def ascend_normalize_weight(topk_weights):
    assert topk_weights.dim() == 2 and topk_weights.is_contiguous()
    assert topk_weights.dtype == torch.float32

    num_tokens, num_topk = topk_weights.shape
    denominator = torch.empty((num_tokens,), dtype=torch.float32, device=topk_weights.device)
    normalized_weights = torch.empty((num_tokens, num_topk), dtype=torch.float32, device=topk_weights.device)

    if num_tokens > 0:
        kernel = get_normalize_weight_kernel(num_topk)
        if int(os.getenv("TK_PRINT_KERNEL_SOURCE", "0")):
            print(kernel.get_kernel_source())
        kernel(topk_weights, denominator, normalized_weights)

    return denominator, normalized_weights


def torch_normalize_weight(topk_weights):
    denominator = topk_weights.sum(dim=1) + 1.0e-20
    normalized_weights = topk_weights / denominator.unsqueeze(1)
    return denominator, normalized_weights


HAS_NPU = find_spec("torch_npu") is not None


def get_device() -> str:
    if HAS_NPU and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_test_configs():
    return [
        (8451, 2),
        (15013, 2),
        (25268, 2),
        (18676, 6),
        (19443, 6),
        (26903, 6),
        (21977, 8),
        (22451, 8),
        (34415, 8),
        (23527, 9),
        (23688, 9),
        (37876, 9),
    ]


def main():
    device = get_device()
    for num_tokens, num_topk in get_test_configs():
        print(f"\nTesting normalize_weight with num_tokens={num_tokens}, num_topk={num_topk}")
        topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32, device=device)
        denominator, normalized_weights = ascend_normalize_weight(topk_weights)

        ref_denom, ref_norm = torch_normalize_weight(topk_weights)
        denom_diff = (denominator - ref_denom).abs()
        norm_diff = (normalized_weights - ref_norm).abs()

        print(f"  denominator  max error: {denom_diff.max().item():.6e}")
        print(f"  normalized   max error: {norm_diff.max().item():.6e}")

        torch.testing.assert_close(denominator, ref_denom, rtol=1e-5, atol=1e-6, check_dtype=True)
        torch.testing.assert_close(normalized_weights, ref_norm, rtol=1e-5, atol=1e-6, check_dtype=True)

        print(f"  Test passed for num_tokens={num_tokens}, num_topk={num_topk}")

    print("\n" + "=" * 50)
    print("All normalize_weight cases passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
