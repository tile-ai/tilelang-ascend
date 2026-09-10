import os
from importlib.util import find_spec

import torch
import tilelang
import tilelang.language as T


tilelang.cache.clear_cache()

os.environ["TILELANG_PRINT_ON_COMPILATION"] = "0"

pass_configs = {tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True}


@tilelang.jit(pass_configs=pass_configs)
def get_normalize_weight_bwd_kernel(num_topk: int):
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
    def normalize_weight_bwd_kernel(
        d_norm: T.Tensor((num_tokens, num_topk), "float"),
        weights: T.Tensor((num_tokens, num_topk), "float"),
        denom: T.Tensor((num_tokens,), "float"),
        result: T.Tensor((num_tokens, num_topk), "float"),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid), T.Scope("V"):
            d_norm_ub = T.alloc_ub((stages, rows_per_vec, aligned_topk), "float")
            weights_ub = T.alloc_ub((stages, rows_per_vec, aligned_topk), "float")
            result_ub = T.alloc_ub((rows_per_vec, aligned_topk), "float")
            dot_ub = T.alloc_ub((rows_per_vec, 1), "float")
            denom_ub = T.alloc_ub((rows_per_vec, 1), "float")
            denom_sq_ub = T.alloc_ub((rows_per_vec, 1), "float")
            dot_broadcast_ub = T.alloc_ub((rows_per_vec, aligned_topk), "float")
            denom_broadcast_ub = T.alloc_ub((rows_per_vec, aligned_topk), "float")
            denom_sq_broadcast_ub = T.alloc_ub((rows_per_vec, aligned_topk), "float")
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
                T.copy(d_norm[token_start : token_start + rows_per_vec, 0:num_topk], d_norm_ub[0, :, :], pad_value=0.0)
                T.copy(weights[token_start : token_start + rows_per_vec, 0:num_topk], weights_ub[0, :, :], pad_value=0.0)
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
                        T.copy(d_norm[next_token_start : next_token_start + rows_per_vec, 0:num_topk], d_norm_ub[nxt, :, :], pad_value=0.0)
                        T.copy(
                            weights[next_token_start : next_token_start + rows_per_vec, 0:num_topk],
                            weights_ub[nxt, :, :],
                            pad_value=0.0,
                        )
                        T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    T.wait_flag("mte3", "v", mte3_store_flag)

                    T.tile.mul(result_ub, d_norm_ub[cur, :, :], weights_ub[cur, :, :])
                    T.reduce_sum(result_ub, dot_ub, dim=-1)
                    T.reduce_sum(weights_ub[cur, :, :], denom_ub, dim=-1)
                    T.tile.add(denom_ub, denom_ub, 1.0e-20)

                    T.tile.mul(denom_sq_ub, denom_ub, denom_ub)
                    T.tile.broadcast(dot_broadcast_ub, dot_ub, axis=1)
                    T.tile.broadcast(denom_broadcast_ub, denom_ub, axis=1)
                    T.tile.broadcast(denom_sq_broadcast_ub, denom_sq_ub, axis=1)

                    T.tile.mul(result_ub, d_norm_ub[cur, :, :], denom_broadcast_ub)
                    T.tile.sub(result_ub, result_ub, dot_broadcast_ub)
                    T.tile.div(result_ub, result_ub, denom_sq_broadcast_ub)

                    T.set_flag("v", "mte3", cur)
                    T.wait_flag("v", "mte3", cur)
                    T.pipe_barrier("mte3")
                    T.copy(result_ub[:, :num_topk], result[token_start : token_start + rows_per_vec, :])
                    T.pipe_barrier("mte3")
                    T.set_flag("mte3", "v", mte3_store_flag)
                    T.set_flag("v", "mte2", cur)

            T.wait_flag("v", "mte2", 0)
            T.wait_flag("v", "mte2", 1)
            T.wait_flag("mte3", "v", mte3_store_flag)

    return normalize_weight_bwd_kernel


def ascend_normalize_weight_backward(d_norm, topk_weights, denominator):
    assert d_norm.dim() == 2 and d_norm.is_contiguous()
    assert d_norm.dtype == torch.float32
    assert topk_weights.dim() == 2 and topk_weights.is_contiguous()
    assert topk_weights.dtype == torch.float32
    assert denominator.dim() == 1 and denominator.is_contiguous()
    assert denominator.dtype == torch.float32

    num_tokens, num_topk = topk_weights.shape
    result = torch.empty((num_tokens, num_topk), dtype=torch.float32, device=d_norm.device)

    if num_tokens > 0:
        kernel = get_normalize_weight_bwd_kernel(num_topk)
        kernel(d_norm, topk_weights, denominator, result)

    return result


def torch_normalize_weight_bwd(d_norm, weights, denom):
    dot = (d_norm * weights).sum(dim=1)
    return (d_norm * denom.unsqueeze(1) - dot.unsqueeze(1)) / (denom * denom).unsqueeze(1)


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


def generate_test_data(num_tokens, num_topk, device):
    topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32, device=device)
    denominator = topk_weights.sum(dim=1) + 1.0e-20
    d_norm = torch.rand((num_tokens, num_topk), dtype=torch.float32, device=device)
    return d_norm, topk_weights, denominator


def synchronize(device: str):
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def main():
    device = get_device()
    test_configs = get_test_configs()

    for num_tokens, num_topk in test_configs:
        d_norm, topk_weights, denominator = generate_test_data(num_tokens, num_topk, device)

        result = ascend_normalize_weight_backward(d_norm, topk_weights, denominator)
        expected = torch_normalize_weight_bwd(d_norm, topk_weights, denominator)

        torch.testing.assert_close(result, expected, rtol=1.0e-5, atol=1.0e-6, check_dtype=True)
        case = f"normalize_weight_bwd T={num_tokens} K={num_topk}"
        print(f"pass {case}")

    print("TEST PASSED!")


if __name__ == "__main__":
    main()
