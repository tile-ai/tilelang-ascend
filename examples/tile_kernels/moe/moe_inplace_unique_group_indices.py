import os
import sys
import torch
import tilelang
from tilelang import language as T
import pytest

tilelang.cache.clear_cache()

INT64 = "int64"
INT32 = "int32"

pass_configs = {
    tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tilelang.jit(pass_configs=pass_configs)
def get_inplace_unique_group_indices_kernel(num_topk: int, num_groups_aligned: int, num_sms: int):
    num_tokens = T.symbolic("num_tokens")
    num_aligned_topk = (num_topk + 31) // 32 * 32

    VEC_NUM = 2
    chunk_size = num_sms * VEC_NUM
    num_batches = (num_tokens + chunk_size - 1) // chunk_size

    @T.prim_func
    def inplace_unique_group_indices_kernel(
        group_indices: T.Tensor[(num_tokens, num_topk), INT64],
    ):
        with T.Kernel(num_sms, is_npu=True) as (cid, vid):
            # Ping-pong UB buffers
            group_indices_ub_ping = T.alloc_ub((num_aligned_topk,), INT64)
            group_indices_ub_pong = T.alloc_ub((num_aligned_topk,), INT64)
            seen_ub_ping = T.alloc_ub((num_groups_aligned,), INT32)
            seen_ub_pong = T.alloc_ub((num_groups_aligned,), INT32)

            # Prologue: prefetch batch 0 to ping buffer
            token_idx_init = cid * VEC_NUM + vid
            if token_idx_init < num_tokens:
                T.copy(group_indices[token_idx_init, 0:num_topk], group_indices_ub_ping[0:num_topk])

            # Steady state: double buffering pipeline
            for b in T.serial(num_batches):
                token_idx_curr = (b * num_sms + cid) * VEC_NUM + vid

                if b % 2 == 0:
                    T.barrier_all()
                    if b + 1 < num_batches:
                        token_idx_next = ((b + 1) * num_sms + cid) * VEC_NUM + vid
                        if token_idx_next < num_tokens:
                            T.copy(group_indices[token_idx_next, 0:num_topk], group_indices_ub_pong[0:num_topk])

                    if token_idx_curr < num_tokens:
                        for g in T.serial(num_groups_aligned):
                            seen_ub_ping[g] = 0
                        for j in T.serial(num_topk):
                            group_idx_int32 = T.cast(group_indices_ub_ping[j], INT32)
                            if group_idx_int32 >= 0 and seen_ub_ping[group_idx_int32] == 1:
                                group_indices_ub_ping[j] = T.cast(-1, INT64)
                            elif group_idx_int32 >= 0:
                                seen_ub_ping[group_idx_int32] = 1
                        T.copy(group_indices_ub_ping[0:num_topk], group_indices[token_idx_curr, 0:num_topk])
                else:
                    T.barrier_all()
                    if b + 1 < num_batches:
                        token_idx_next = ((b + 1) * num_sms + cid) * VEC_NUM + vid
                        if token_idx_next < num_tokens:
                            T.copy(group_indices[token_idx_next, 0:num_topk], group_indices_ub_ping[0:num_topk])

                    if token_idx_curr < num_tokens:
                        for g in T.serial(num_groups_aligned):
                            seen_ub_pong[g] = 0
                        for j in T.serial(num_topk):
                            group_idx_int32 = T.cast(group_indices_ub_pong[j], INT32)
                            if group_idx_int32 >= 0 and seen_ub_pong[group_idx_int32] == 1:
                                group_indices_ub_pong[j] = T.cast(-1, INT64)
                            elif group_idx_int32 >= 0:
                                seen_ub_pong[group_idx_int32] = 1
                        T.copy(group_indices_ub_pong[0:num_topk], group_indices[token_idx_curr, 0:num_topk])

    return inplace_unique_group_indices_kernel


def inplace_unique_group_indices(group_indices: torch.Tensor, num_groups: int) -> None:
    """Deduplicate group indices per token, marking duplicates as -1 in-place.

    Args:
        group_indices: Int64 tensor of shape (num_tokens, num_topk) with group ids.
        num_groups: Total number of groups (must be <= 128).
    """
    assert group_indices.dim() == 2
    assert num_groups <= 128

    num_topk = group_indices.shape[1]
    num_groups_aligned = (num_groups + 63) // 64 * 64

    # Use 32 SMs to fill NPU AI Core
    num_sms = 32
    kernel = get_inplace_unique_group_indices_kernel(num_topk, num_groups_aligned, num_sms=num_sms)

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    if group_indices.shape[0] > 0:
        kernel(group_indices)


def inplace_unique_group_indices_ref(group_indices: torch.Tensor) -> None:
    """Reference implementation: deduplicate group indices per token in-place."""
    num_tokens, num_topk = group_indices.shape
    for i in range(num_tokens):
        seen = set()
        for j in range(num_topk):
            val = group_indices[i, j].item()
            if val >= 0:
                if val in seen:
                    group_indices[i, j] = -1
                else:
                    seen.add(val)


def generate_moe_params(is_benchmark: bool = False):
    """Generate MoE configurations for parameter coverage."""
    for ep, experts_list in [(8, [9, 32]), (64, [4])]:
        for experts in experts_list:
            for topk in [2, 6, 8, 9]:
                yield {
                    "num_experts": experts,
                    "num_ep_ranks": ep,
                    "num_topk": topk,
                }


def generate_test_params(is_benchmark: bool = False) -> list[dict]:
    """Create test parameters where num_groups divides total experts."""
    params = []
    for moe in generate_moe_params(is_benchmark=is_benchmark):
        total_experts = moe["num_experts"] * moe["num_ep_ranks"]
        for num_groups in (8, 16, 72):
            if total_experts % num_groups == 0:
                params.append({**moe, "num_groups": num_groups})
    return params


def generate_topk_idx(params: dict):
    """Generate random topk expert indices with some -1 values."""
    num_experts = params["num_experts"]
    num_ep_ranks = params["num_ep_ranks"]
    num_topk = params["num_topk"]
    total_experts = num_experts * num_ep_ranks
    num_tokens = torch.randint(1000, 40000, (1,)).item()
    indices = torch.randint(0, total_experts, (num_tokens, num_topk), dtype=torch.int64)
    mask = torch.rand(indices.shape) < 0.1
    indices[mask] = -1
    return indices


def generate_test_data(params: dict):
    """Derive group indices from topk expert indices."""
    num_experts = params["num_experts"]
    num_ep_ranks = params["num_ep_ranks"]
    num_groups = params["num_groups"]

    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]
    experts_per_group = (num_experts * num_ep_ranks) // num_groups
    _group_indices = topk_idx // experts_per_group
    _group_indices[topk_idx == -1] = -1

    return _group_indices, num_tokens


def get_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_param_id(params: dict) -> str:
    return f"ep={params['num_ep_ranks']}_experts={params['num_experts']}_topk={params['num_topk']}_groups={params['num_groups']}"


@pytest.mark.parametrize("params", generate_test_params(), ids=make_param_id)
def test_inplace_unique_group_indices(params):
    device = get_device()
    torch.manual_seed(42)

    group_indices, num_tokens = generate_test_data(params)
    group_indices = group_indices.to(device)

    # Clone for reference
    group_indices_ref = group_indices.clone()
    inplace_unique_group_indices_ref(group_indices_ref)

    # Run tiling kernel (in-place)
    inplace_unique_group_indices(group_indices, params["num_groups"])

    assert torch.equal(group_indices, group_indices_ref), f"Result mismatch for {make_param_id(params)}"
    print(f"Test passed for params: {make_param_id(params)}")


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v"])
    if exit_code == pytest.ExitCode.OK:
        print("All inplace_unique_group_indices tests passed! Kernel Output Match!")
    sys.exit(exit_code)
