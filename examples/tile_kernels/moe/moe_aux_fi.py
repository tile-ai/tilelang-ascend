import os
import pytest
import numpy as np
import sys
import torch
import tilelang
import tilelang.language as T
from typing import Iterable

# For compatibility, keep data types consistent with those inside the operator.
INDEX_DTYPE = "int64"
FLOAT_DTYPE = "float32"
INT32_DTYPE = "int32"

tilelang.cache.clear_cache()


def get_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


@tilelang.jit()
def get_aux_fi_kernel(num_topk: int, num_experts: int, num_sms: int):
    num_tokens = T.symbolic("num_tokens")

    # Core fix: align inner dimension to 32 bytes.
    num_aligned_topk = (num_topk + 31) // 32 * 32

    # DMA transfers 256 tokens per burst, maximizing MTE bandwidth.
    BLOCK_TOKENS = 256

    @T.prim_func
    def aux_fi_kernel(
        topk_idx: T.Tensor[(num_tokens, num_topk), INDEX_DTYPE],
        out: T.Tensor[(num_experts,), FLOAT_DTYPE],
        num_aux_topk: T.int32,
    ):
        with T.Kernel(num_experts, is_npu=True) as (cid, vid):
            # UB allocation must use aligned shape to prevent DMA overwrite.
            ub_ping = T.alloc_ub((BLOCK_TOKENS, num_aligned_topk), INDEX_DTYPE)
            ub_pong = T.alloc_ub((BLOCK_TOKENS, num_aligned_topk), INDEX_DTYPE)
            count_ub = T.alloc_ub((1,), FLOAT_DTYPE)

            count_ub[0] = 0.0

            num_full_chunks = num_tokens // BLOCK_TOKENS
            tail_tokens = num_tokens % BLOCK_TOKENS

            # Prologue: preload first block.
            # Note: all T.copy must use pad_value=-1 to safely pad unaligned edges.
            if num_full_chunks > 0:
                T.copy(
                    topk_idx[0:BLOCK_TOKENS, 0:num_topk],
                    ub_ping[0:BLOCK_TOKENS, 0:num_aligned_topk],
                    pad_value=-1,
                )
                T.set_flag("MTE2", "S", 0)
                T.wait_flag("MTE2", "S", 0)

            # Steady State (pipeline loop).
            for c in T.serial(num_full_chunks):
                if c % 2 == 0:
                    if c + 1 < num_full_chunks:
                        T.copy(
                            topk_idx[(c + 1) * BLOCK_TOKENS : (c + 2) * BLOCK_TOKENS, 0:num_topk],
                            ub_pong[0:BLOCK_TOKENS, 0:num_aligned_topk],
                            pad_value=-1,
                        )
                        T.set_flag("MTE2", "S", 0)
                        T.wait_flag("MTE2", "S", 0)

                    for i in T.serial(BLOCK_TOKENS):
                        # Only unroll real num_topk (excluding padding) for max performance.
                        for j in T.unroll(num_topk):
                            idx_int32 = T.cast(ub_ping[i, j], INT32_DTYPE)
                            match_val = T.Select(idx_int32 == cid, T.float32(1.0), T.float32(0.0))
                            count_ub[0] = count_ub[0] + match_val

                else:
                    if c + 1 < num_full_chunks:
                        T.copy(
                            topk_idx[(c + 1) * BLOCK_TOKENS : (c + 2) * BLOCK_TOKENS, 0:num_topk],
                            ub_ping[0:BLOCK_TOKENS, 0:num_aligned_topk],
                            pad_value=-1,
                        )
                        T.set_flag("MTE2", "S", 0)
                        T.wait_flag("MTE2", "S", 0)

                    for i in T.serial(BLOCK_TOKENS):
                        for j in T.unroll(num_topk):
                            idx_int32 = T.cast(ub_pong[i, j], INT32_DTYPE)
                            match_val = T.Select(idx_int32 == cid, T.float32(1.0), T.float32(0.0))
                            count_ub[0] = count_ub[0] + match_val

            # Tail Processing.
            if tail_tokens > 0:
                offset = num_full_chunks * BLOCK_TOKENS
                T.copy(
                    topk_idx[offset : offset + tail_tokens, 0:num_topk],
                    ub_ping[0:tail_tokens, 0:num_aligned_topk],
                    pad_value=-1,
                )
                T.set_flag("MTE2", "S", 0)
                T.wait_flag("MTE2", "S", 0)

                for i in T.serial(tail_tokens):
                    for j in T.unroll(num_topk):
                        idx_int32 = T.cast(ub_ping[i, j], INT32_DTYPE)
                        match_val = T.Select(idx_int32 == cid, T.float32(1.0), T.float32(0.0))
                        count_ub[0] = count_ub[0] + match_val

            # Post-processing: normalization.
            denom = T.cast(num_tokens, FLOAT_DTYPE) * T.cast(num_aux_topk, FLOAT_DTYPE)
            count_ub[0] = count_ub[0] * T.cast(num_experts, FLOAT_DTYPE) / denom

            T.copy(count_ub[0:1], out[cid : cid + 1])

    return aux_fi_kernel


def aux_fi_tl(topk_idx: torch.Tensor, num_experts: int, num_aux_topk: int) -> torch.Tensor:
    """Compute per-expert frequency indicator (f_i) for the auxiliary loss via TileLang kernel."""
    assert topk_idx.dim() == 2 and topk_idx.is_contiguous()

    num_tokens = topk_idx.shape[0]
    num_topk = topk_idx.shape[1]

    out = torch.zeros(num_experts, dtype=torch.float32, device=topk_idx.device)
    if num_tokens == 0:
        return out

    kernel = get_aux_fi_kernel(num_topk, num_experts, num_sms=1)

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    kernel(topk_idx, out, num_aux_topk)
    return out


def aux_fi_ref(topk_idx: torch.Tensor, num_experts: int, num_aux_topk: int) -> torch.Tensor:
    """Compute auxiliary load-balancing frequency indicator f_i for each expert (PyTorch Reference)."""
    num_tokens, num_topk = topk_idx.shape
    if num_tokens == 0:
        return torch.zeros(num_experts, dtype=torch.float32, device=topk_idx.device)
    valid_idx = topk_idx[topk_idx >= 0]
    counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_idx.device)
    counts.scatter_add_(0, valid_idx, torch.ones_like(valid_idx))
    return counts.float() * num_experts / (num_tokens * num_aux_topk)


def generate_moe_params(is_benchmark: bool = False) -> Iterable[dict]:
    do_full_test = os.getenv("TK_FULL_TEST") in ["1", "true", "True"]
    extra_num_topk_list = (1, 7) if do_full_test else ()
    extra_num_experts_list = (288, 384) if do_full_test else ()
    extra_num_ep_ranks_list = (1, 72, 256) if do_full_test else ()

    if do_full_test and not is_benchmark:
        yield {"num_send_tokens": 0, "num_topk": 1, "num_experts": 1, "num_ep_ranks": 1}

    for num_tokens in (4001,):
        for num_topk in (2, 6, 8, 9) + extra_num_topk_list:
            for num_experts in (72, 256) + extra_num_experts_list:
                for num_ep_ranks in (8, 64) + extra_num_ep_ranks_list:
                    if num_experts % num_ep_ranks == 0:
                        yield {
                            "num_send_tokens": num_tokens,
                            "num_topk": num_topk,
                            "num_experts": num_experts // num_ep_ranks,
                            "num_ep_ranks": num_ep_ranks,
                        }


def generate_topk_idx(params: dict) -> torch.Tensor:
    num_send_tokens = params["num_send_tokens"]
    num_experts = params["num_experts"]
    num_topk = params["num_topk"]
    num_ep_ranks = params["num_ep_ranks"]

    if num_send_tokens == 0:
        return torch.empty((0, num_topk), dtype=torch.int64, device=get_device())
    scores = torch.rand(
        (num_send_tokens * num_ep_ranks, num_experts * num_ep_ranks),
        dtype=torch.bfloat16,
        device=get_device(),
    )
    _, topk_idx = torch.topk(scores, k=num_topk, dim=-1, sorted=False)
    mask = topk_idx >= num_experts
    topk_idx[mask] = -1
    mask = mask.all(dim=1)
    topk_idx = topk_idx[~mask]
    return topk_idx


def generate_test_data(params):
    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]
    return (topk_idx, num_tokens)


def generate_test_params(is_benchmark: bool = False) -> list[dict]:
    return [
        {**moe, "num_aux_topk": num_aux_topk}
        for moe in generate_moe_params(is_benchmark=is_benchmark)
        for num_aux_topk in (1, moe["num_topk"])
    ]


def make_param_id(params: dict) -> str:
    nt = params.get("num_send_tokens", 0)
    ne = params["num_experts"]
    nk = params["num_topk"]
    na = params["num_aux_topk"]
    return f"tokens={nt}_experts={ne}_topk={nk}_aux={na}"


@pytest.mark.parametrize("params", generate_test_params(is_benchmark=False), ids=make_param_id)
def test_aux_fi(params):
    num_experts = params["num_experts"]
    num_aux_topk = params["num_aux_topk"]

    torch.manual_seed(42)
    topk_idx, num_tokens = generate_test_data(params)

    # Get PyTorch reference result.
    out_ref = aux_fi_ref(topk_idx, num_experts, num_aux_topk)

    # Get TileLang operator result.
    out_tl = aux_fi_tl(topk_idx, num_experts, num_aux_topk)

    np.testing.assert_allclose(
        out_tl.cpu().numpy(),
        out_ref.cpu().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    print(f"Forward Test passed for params: {make_param_id(params)}")


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v"])
    if exit_code == pytest.ExitCode.OK:
        print("All aux_fi tests passed! Kernel Output Match!")
    sys.exit(exit_code)
