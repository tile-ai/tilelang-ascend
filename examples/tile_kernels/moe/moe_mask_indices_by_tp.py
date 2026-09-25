import os
import sys
import torch
import tilelang
from tilelang import language as T
import pytest

tilelang.cache.clear_cache()

INT64_DTYPE = "int64"
INT32_DTYPE = "int32"

pass_configs = {
    tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tilelang.jit(pass_configs=pass_configs)
def get_mask_indices_by_tp_kernel(num_topk: int, num_aligned_topk: int):
    num_tokens = T.symbolic("num_tokens")

    @T.prim_func
    def mask_indices_by_tp_kernel(
        indices: T.Tensor[(num_tokens, num_topk), INT64_DTYPE],
        masked_indices: T.Tensor[(num_tokens, num_aligned_topk), INT64_DTYPE],
        per_npu: T.int32,
        per_dp: T.int32,
        num_tp_ranks: T.int32,
        tp_rank: T.int32,
    ):
        VEC_NUM = 2
        num_blocks = 64
        chunk_size = num_blocks * VEC_NUM
        num_batches = (num_tokens + chunk_size - 1) // chunk_size

        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            indices_ub_ping = T.alloc_ub((num_aligned_topk,), INT64_DTYPE)
            indices_ub_pong = T.alloc_ub((num_aligned_topk,), INT64_DTYPE)
            masked_ub_ping = T.alloc_ub((num_aligned_topk,), INT64_DTYPE)
            masked_ub_pong = T.alloc_ub((num_aligned_topk,), INT64_DTYPE)

            token_idx_0 = (0 * num_blocks + cid) * VEC_NUM + vid
            if token_idx_0 < num_tokens:
                T.copy(
                    indices[token_idx_0, 0:num_topk],
                    indices_ub_ping,
                    pad_value=T.cast(-1, INT64_DTYPE),
                )

            for batch_idx in T.serial(num_batches):
                if batch_idx % 2 == 0:
                    T.barrier_all()

                    token_idx = (batch_idx * num_blocks + cid) * VEC_NUM + vid
                    if token_idx < num_tokens:
                        next_batch_idx = batch_idx + 1
                        if next_batch_idx < num_batches:
                            next_token_idx = (next_batch_idx * num_blocks + cid) * VEC_NUM + vid
                            if next_token_idx < num_tokens:
                                T.copy(
                                    indices[next_token_idx, 0:num_topk],
                                    indices_ub_pong,
                                    pad_value=T.cast(-1, INT64_DTYPE),
                                )

                        for j in T.serial(num_aligned_topk):
                            value = indices_ub_ping[j]
                            value_int32 = T.cast(value, INT32_DTYPE)

                            local_value = value_int32 - tp_rank * per_npu
                            dp_rank = T.truncdiv(local_value, per_dp)
                            remapped = local_value - dp_rank * (per_dp - per_npu)

                            ans = T.Select(
                                remapped < 0,
                                T.cast(-1, INT64_DTYPE),
                                T.cast(remapped, INT64_DTYPE),
                            )
                            ans = T.Select(
                                T.truncmod(T.truncdiv(value_int32, per_npu), num_tp_ranks) != tp_rank,
                                T.cast(-1, INT64_DTYPE),
                                ans,
                            )
                            ans = T.Select(value_int32 < 0, T.cast(-1, INT64_DTYPE), ans)

                            masked_ub_ping[j] = ans

                        T.copy(
                            masked_ub_ping[0:num_aligned_topk],
                            masked_indices[token_idx, 0:num_aligned_topk],
                        )
                else:
                    T.barrier_all()

                    token_idx = (batch_idx * num_blocks + cid) * VEC_NUM + vid
                    if token_idx < num_tokens:
                        next_batch_idx = batch_idx + 1
                        if next_batch_idx < num_batches:
                            next_token_idx = (next_batch_idx * num_blocks + cid) * VEC_NUM + vid
                            if next_token_idx < num_tokens:
                                T.copy(
                                    indices[next_token_idx, 0:num_topk],
                                    indices_ub_ping,
                                    pad_value=T.cast(-1, INT64_DTYPE),
                                )

                        for j in T.serial(num_aligned_topk):
                            value = indices_ub_pong[j]
                            value_int32 = T.cast(value, INT32_DTYPE)

                            local_value = value_int32 - tp_rank * per_npu
                            dp_rank = T.truncdiv(local_value, per_dp)
                            remapped = local_value - dp_rank * (per_dp - per_npu)

                            ans = T.Select(
                                remapped < 0,
                                T.cast(-1, INT64_DTYPE),
                                T.cast(remapped, INT64_DTYPE),
                            )
                            ans = T.Select(
                                T.truncmod(T.truncdiv(value_int32, per_npu), num_tp_ranks) != tp_rank,
                                T.cast(-1, INT64_DTYPE),
                                ans,
                            )
                            ans = T.Select(value_int32 < 0, T.cast(-1, INT64_DTYPE), ans)

                            masked_ub_pong[j] = ans

                        T.copy(
                            masked_ub_pong[0:num_aligned_topk],
                            masked_indices[token_idx, 0:num_aligned_topk],
                        )

    return mask_indices_by_tp_kernel


def mask_indices_by_tp(indices: torch.Tensor, n: int, num_ep_ranks: int, tp_rank: int, num_tp_ranks: int) -> torch.Tensor:
    """Mask expert indices to keep only those belonging to the given TP rank."""
    num_topk = indices.shape[1]
    per_npu = n // num_ep_ranks
    per_dp = num_tp_ranks * per_npu

    num_aligned_topk = (num_topk + 7) // 8 * 8

    kernel = get_mask_indices_by_tp_kernel(num_topk, num_aligned_topk)

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    if indices.shape[0] > 0:
        masked_indices_padded = torch.empty((indices.shape[0], num_aligned_topk), dtype=torch.int64, device=indices.device)
        kernel(indices, masked_indices_padded, per_npu, per_dp, num_tp_ranks, tp_rank)
        return masked_indices_padded[:, :num_topk].contiguous()
    else:
        return torch.empty_like(indices)


def mask_indices_by_tp_ref(
    indices: torch.Tensor,
    n: int,
    num_ep_ranks: int,
    tp_rank: int,
    num_tp_ranks: int,
) -> torch.Tensor:
    """Reference implementation of expert index masking for TP ranks."""
    per_gpu = n // num_ep_ranks
    per_dp = num_tp_ranks * per_gpu

    value = indices.clone()
    invalid = (value < 0) | ((value // per_gpu) % num_tp_ranks != tp_rank)

    value = value - tp_rank * per_gpu
    dp_rank = value // per_dp
    value = value - dp_rank * (per_dp - per_gpu)

    value[invalid | (value < 0)] = -1
    return value


def get_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate_moe_params():
    """Generate coverage combinations for MoE parameters."""
    for ep, experts_list in [(8, [9, 32]), (64, [4])]:
        for experts in experts_list:
            for topk in [2, 6, 8, 9]:
                yield {
                    "num_experts": experts,
                    "num_ep_ranks": ep,
                    "num_topk": topk,
                }


def generate_test_params() -> list[dict]:
    """Augment base configs with different tensor-parallel sizes."""
    params = []
    for moe in generate_moe_params():
        for tp in [2, 4, 8]:
            params.append({**moe, "num_tp_ranks": tp})
    return params


def generate_test_data(params: dict):
    """Create random test data: indices, token count, tp_rank, total experts."""
    num_experts = params["num_experts"]
    num_ep_ranks = params["num_ep_ranks"]
    num_topk = params["num_topk"]
    num_tp_ranks = params["num_tp_ranks"]
    total_experts = num_experts * num_ep_ranks

    num_tokens = torch.randint(1000, 40000, (1,)).item()

    indices = torch.randint(0, total_experts, (num_tokens, num_topk), dtype=torch.int64)
    mask = torch.rand(indices.shape) < 0.1
    indices[mask] = -1

    tp_rank = torch.randint(0, num_tp_ranks, (1,)).item()
    n = total_experts

    return indices, num_tokens, tp_rank, n


def make_param_id(params: dict) -> str:
    return f"ep={params['num_ep_ranks']}_experts={params['num_experts']}_topk={params['num_topk']}_tp={params['num_tp_ranks']}"


@pytest.mark.parametrize("params", generate_test_params(), ids=make_param_id)
def test_mask_indices_by_tp(params):
    device = get_device()
    torch.manual_seed(42)

    indices, num_tokens, tp_rank, n = generate_test_data(params)
    indices = indices.to(device)

    out_ref = mask_indices_by_tp_ref(indices.cpu(), n, params["num_ep_ranks"], tp_rank, params["num_tp_ranks"]).to(device)
    out_tl = mask_indices_by_tp(indices, n, params["num_ep_ranks"], tp_rank, params["num_tp_ranks"])

    assert out_tl.shape == out_ref.shape, f"Shape mismatch: {out_tl.shape} vs {out_ref.shape}"
    assert torch.equal(out_tl, out_ref), f"Result mismatch for {make_param_id(params)}"
    print(f"Forward Test passed for params: {make_param_id(params)}")


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v"])
    if exit_code == pytest.ExitCode.OK:
        print("All mask_indices_by_tp tests passed! Kernel Output Match!")
    sys.exit(exit_code)
