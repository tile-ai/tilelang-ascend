import os
import pytest
import numpy as np
import torch
import tilelang
import tilelang.language as T


FLOAT_DTYPE = "float32"
INDEX_DTYPE = "int64"
INT32_DTYPE = "int32"

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tilelang.jit(pass_configs=pass_configs)
def get_topk_gate_kernel(num_experts: int, num_topk: int):
    num_tokens = T.symbolic("num_tokens")
    num_threads = 32
    num_aligned_experts = (num_experts + num_threads - 1) // num_threads * num_threads

    VEC_NUM = 2
    num_batches = (num_tokens + VEC_NUM - 1) // VEC_NUM
    num_blocks = 128

    aligned_num_topk_f32 = (num_topk + 7) // 8 * 8
    aligned_num_topk_i64 = (num_topk + 3) // 4 * 4

    @T.prim_func
    def topk_gate_kernel(
        scores: T.Tensor[(num_tokens, num_experts), FLOAT_DTYPE],
        topk_idx: T.Tensor[(num_tokens, num_topk), INDEX_DTYPE],
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            scores_ub_ping = T.alloc_ub((num_aligned_experts,), FLOAT_DTYPE)
            scores_ub_pong = T.alloc_ub((num_aligned_experts,), FLOAT_DTYPE)

            topk_idx_out_ub_ping = T.alloc_ub((aligned_num_topk_i64,), INDEX_DTYPE)
            topk_idx_out_ub_pong = T.alloc_ub((aligned_num_topk_i64,), INDEX_DTYPE)

            topk_dst_ub_ping = T.alloc_ub((2 * aligned_num_topk_f32,), FLOAT_DTYPE)
            topk_dst_ub_pong = T.alloc_ub((2 * aligned_num_topk_f32,), FLOAT_DTYPE)
            topk_index_f32_ping = T.alloc_ub((aligned_num_topk_f32,), FLOAT_DTYPE)
            topk_index_f32_pong = T.alloc_ub((aligned_num_topk_f32,), FLOAT_DTYPE)

            my_iters = (num_batches - cid + num_blocks - 1) // num_blocks

            # Prologue
            if my_iters > 0:
                batch_idx_0 = cid
                token_idx_0 = T.min(batch_idx_0 * VEC_NUM + vid, num_tokens - 1)
                T.copy(scores[token_idx_0, 0:num_experts], scores_ub_ping, pad_value=-T.infinity(FLOAT_DTYPE))
                T.set_flag("MTE2", "V", 0)

            if my_iters > 1:
                batch_idx_1 = cid + num_blocks
                token_idx_1 = T.min(batch_idx_1 * VEC_NUM + vid, num_tokens - 1)
                T.copy(scores[token_idx_1, 0:num_experts], scores_ub_pong, pad_value=-T.infinity(FLOAT_DTYPE))

            if my_iters > 0:
                T.wait_flag("MTE2", "V", 0)
                T.tile.topk(topk_dst_ub_ping, scores_ub_ping, num_topk, num_aligned_experts)
                T.pipe_barrier("V")
                T.tile.gather_mask(topk_index_f32_ping, topk_dst_ub_ping, "P1010")
                T.tile.cast(topk_idx_out_ub_ping, topk_index_f32_ping, "CAST_ROUND", num_topk)

            # Main loop (unrolled by step 2)
            loop_iters = my_iters - 2
            if loop_iters > 0:
                for i_step in T.serial(loop_iters // 2):
                    i_ping = i_step * 2 + 1
                    batch_idx_prev_ping = cid + (i_ping - 1) * num_blocks
                    batch_idx_prev_pong = cid + i_ping * num_blocks
                    batch_idx_next_ping = cid + (i_ping + 1) * num_blocks
                    batch_idx_next_pong = cid + (i_ping + 2) * num_blocks

                    token_idx_prev_ping = batch_idx_prev_ping * VEC_NUM + vid
                    token_idx_next_ping = T.min(batch_idx_next_ping * VEC_NUM + vid, num_tokens - 1)
                    token_idx_prev_pong = batch_idx_prev_pong * VEC_NUM + vid
                    token_idx_next_pong = T.min(batch_idx_next_pong * VEC_NUM + vid, num_tokens - 1)

                    # Ping phase
                    T.barrier_all()
                    T.copy(topk_idx_out_ub_ping[0:num_topk], topk_idx[token_idx_prev_ping, 0:num_topk])
                    T.copy(scores[token_idx_next_ping, 0:num_experts], scores_ub_ping, pad_value=-T.infinity(FLOAT_DTYPE))

                    T.tile.topk(topk_dst_ub_pong, scores_ub_pong, num_topk, num_aligned_experts)
                    T.pipe_barrier("V")
                    T.tile.gather_mask(topk_index_f32_pong, topk_dst_ub_pong, "P1010")
                    T.tile.cast(topk_idx_out_ub_pong, topk_index_f32_pong, "CAST_ROUND", num_topk)

                    # Pong phase
                    T.barrier_all()
                    T.copy(topk_idx_out_ub_pong[0:num_topk], topk_idx[token_idx_prev_pong, 0:num_topk])
                    T.copy(scores[token_idx_next_pong, 0:num_experts], scores_ub_pong, pad_value=-T.infinity(FLOAT_DTYPE))

                    T.tile.topk(topk_dst_ub_ping, scores_ub_ping, num_topk, num_aligned_experts)
                    T.pipe_barrier("V")
                    T.tile.gather_mask(topk_index_f32_ping, topk_dst_ub_ping, "P1010")
                    T.tile.cast(topk_idx_out_ub_ping, topk_index_f32_ping, "CAST_ROUND", num_topk)

            if loop_iters > 0 and loop_iters % 2 == 1:
                i_last = loop_iters
                batch_idx_prev_last = cid + (i_last - 1) * num_blocks
                batch_idx_next_last = cid + (i_last + 1) * num_blocks

                token_idx_prev_last = batch_idx_prev_last * VEC_NUM + vid
                token_idx_next_last = T.min(batch_idx_next_last * VEC_NUM + vid, num_tokens - 1)

                T.barrier_all()
                T.copy(topk_idx_out_ub_ping[0:num_topk], topk_idx[token_idx_prev_last, 0:num_topk])
                T.copy(scores[token_idx_next_last, 0:num_experts], scores_ub_ping, pad_value=-T.infinity(FLOAT_DTYPE))

                T.tile.topk(topk_dst_ub_pong, scores_ub_pong, num_topk, num_aligned_experts)
                T.pipe_barrier("V")
                T.tile.gather_mask(topk_index_f32_pong, topk_dst_ub_pong, "P1010")
                T.tile.cast(topk_idx_out_ub_pong, topk_index_f32_pong, "CAST_ROUND", num_topk)

            # Epilogue
            T.barrier_all()
            i_epi2 = my_iters - 1
            if i_epi2 > 0 and i_epi2 % 2 == 0:
                T.tile.topk(topk_dst_ub_ping, scores_ub_ping, num_topk, num_aligned_experts)
                T.pipe_barrier("V")
                T.tile.gather_mask(topk_index_f32_ping, topk_dst_ub_ping, "P1010")
                T.tile.cast(topk_idx_out_ub_ping, topk_index_f32_ping, "CAST_ROUND", num_topk)
            elif i_epi2 > 0:
                T.tile.topk(topk_dst_ub_pong, scores_ub_pong, num_topk, num_aligned_experts)
                T.pipe_barrier("V")
                T.tile.gather_mask(topk_index_f32_pong, topk_dst_ub_pong, "P1010")
                T.tile.cast(topk_idx_out_ub_pong, topk_index_f32_pong, "CAST_ROUND", num_topk)

            i_epi1 = my_iters - 2
            if i_epi1 >= 0 and i_epi1 % 2 == 0:
                batch_idx_epi1 = cid + i_epi1 * num_blocks
                token_idx_epi1 = batch_idx_epi1 * VEC_NUM + vid
                T.copy(topk_idx_out_ub_ping[0:num_topk], topk_idx[token_idx_epi1, 0:num_topk])
            elif i_epi1 >= 0:
                batch_idx_epi1 = cid + i_epi1 * num_blocks
                token_idx_epi1 = batch_idx_epi1 * VEC_NUM + vid
                T.copy(topk_idx_out_ub_pong[0:num_topk], topk_idx[token_idx_epi1, 0:num_topk])

            if i_epi2 >= 0 and i_epi2 % 2 == 0:
                batch_idx_epi2 = cid + i_epi2 * num_blocks
                token_idx_epi2 = T.min(batch_idx_epi2 * VEC_NUM + vid, num_tokens - 1)
                T.copy(topk_idx_out_ub_ping[0:num_topk], topk_idx[token_idx_epi2, 0:num_topk])
            elif i_epi2 >= 0:
                batch_idx_epi2 = cid + i_epi2 * num_blocks
                token_idx_epi2 = T.min(batch_idx_epi2 * VEC_NUM + vid, num_tokens - 1)
                T.copy(topk_idx_out_ub_pong[0:num_topk], topk_idx[token_idx_epi2, 0:num_topk])

    return topk_gate_kernel


def topk_gate(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    assert scores.dim() == 2 and scores.is_contiguous() and scores.dtype == torch.float32
    num_tokens, num_experts = scores.shape
    assert num_topk <= num_experts
    topk_idx = torch.empty((num_tokens, num_topk), dtype=torch.int64, device=scores.device)
    if num_tokens == 0:
        return topk_idx

    kernel = get_topk_gate_kernel(num_experts, num_topk)

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    kernel(scores, topk_idx)
    return topk_idx


@tilelang.jit()
def get_topk_gate_backward_kernel(num_experts: int, num_topk: int):
    num_tokens = T.symbolic("num_tokens")
    num_threads = 32
    num_aligned_experts = (num_experts + num_threads - 1) // num_threads * num_threads
    VEC_NUM = 2
    num_batches = (num_tokens + VEC_NUM - 1) // VEC_NUM
    num_blocks = 128

    aligned_num_topk = (num_topk + 7) // 8 * 8

    @T.prim_func
    def topk_gate_backward_kernel(
        grad_out: T.Tensor[(num_tokens, num_topk), FLOAT_DTYPE],
        topk_idx: T.Tensor[(num_tokens, num_topk), INDEX_DTYPE],
        grad_scores: T.Tensor[(num_tokens, num_experts), FLOAT_DTYPE],
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            grad_out_ub_ping = T.alloc_ub((aligned_num_topk,), FLOAT_DTYPE)
            grad_out_ub_pong = T.alloc_ub((aligned_num_topk,), FLOAT_DTYPE)
            topk_idx_ub_ping = T.alloc_ub((aligned_num_topk,), INDEX_DTYPE)
            topk_idx_ub_pong = T.alloc_ub((aligned_num_topk,), INDEX_DTYPE)
            grad_scores_ub_ping = T.alloc_ub((num_aligned_experts,), FLOAT_DTYPE)
            grad_scores_ub_pong = T.alloc_ub((num_aligned_experts,), FLOAT_DTYPE)

            my_iters = (num_batches - cid + num_blocks - 1) // num_blocks

            # Prologue
            if my_iters > 0:
                batch_idx_0 = cid
                token_idx_0 = T.min(batch_idx_0 * VEC_NUM + vid, num_tokens - 1)
                T.copy(grad_out[token_idx_0, 0:num_topk], grad_out_ub_ping[0:num_topk])
                T.copy(topk_idx[token_idx_0, 0:num_topk], topk_idx_ub_ping[0:num_topk])

            if my_iters > 1:
                batch_idx_1 = cid + num_blocks
                token_idx_1 = T.min(batch_idx_1 * VEC_NUM + vid, num_tokens - 1)
                T.copy(grad_out[token_idx_1, 0:num_topk], grad_out_ub_pong[0:num_topk])
                T.copy(topk_idx[token_idx_1, 0:num_topk], topk_idx_ub_pong[0:num_topk])

            if my_iters > 0:
                T.barrier_all()
                T.tile.fill(grad_scores_ub_ping, 0.0)
                T.pipe_barrier("V")
                for k in range(num_topk):
                    idx_int32 = T.cast(topk_idx_ub_ping[k], INT32_DTYPE)
                    grad_scores_ub_ping[idx_int32] = grad_out_ub_ping[k]
                T.pipe_barrier("V")

            # Main loop (unrolled by step 2)
            loop_iters = my_iters - 2
            if loop_iters > 0:
                for i_step in T.serial(loop_iters // 2):
                    i_ping = i_step * 2 + 1
                    batch_idx_prev_ping = cid + (i_ping - 1) * num_blocks
                    batch_idx_prev_pong = cid + i_ping * num_blocks
                    batch_idx_next_ping = cid + (i_ping + 1) * num_blocks
                    batch_idx_next_pong = cid + (i_ping + 2) * num_blocks

                    token_idx_prev_ping = batch_idx_prev_ping * VEC_NUM + vid
                    token_idx_next_ping = T.min(batch_idx_next_ping * VEC_NUM + vid, num_tokens - 1)
                    token_idx_prev_pong = batch_idx_prev_pong * VEC_NUM + vid
                    token_idx_next_pong = T.min(batch_idx_next_pong * VEC_NUM + vid, num_tokens - 1)

                    # Ping phase
                    T.barrier_all()
                    T.copy(grad_out[token_idx_next_ping, 0:num_topk], grad_out_ub_ping[0:num_topk])
                    T.copy(topk_idx[token_idx_next_ping, 0:num_topk], topk_idx_ub_ping[0:num_topk])

                    T.copy(grad_scores_ub_ping[0:num_experts], grad_scores[token_idx_prev_ping, 0:num_experts])

                    T.tile.fill(grad_scores_ub_pong, 0.0)
                    T.pipe_barrier("V")
                    for k in range(num_topk):
                        idx_int32 = T.cast(topk_idx_ub_pong[k], INT32_DTYPE)
                        grad_scores_ub_pong[idx_int32] = grad_out_ub_pong[k]
                    T.pipe_barrier("V")

                    # Pong phase
                    T.barrier_all()
                    T.copy(grad_out[token_idx_next_pong, 0:num_topk], grad_out_ub_pong[0:num_topk])
                    T.copy(topk_idx[token_idx_next_pong, 0:num_topk], topk_idx_ub_pong[0:num_topk])

                    T.copy(grad_scores_ub_pong[0:num_experts], grad_scores[token_idx_prev_pong, 0:num_experts])

                    T.tile.fill(grad_scores_ub_ping, 0.0)
                    T.pipe_barrier("V")
                    for k in range(num_topk):
                        idx_int32 = T.cast(topk_idx_ub_ping[k], INT32_DTYPE)
                        grad_scores_ub_ping[idx_int32] = grad_out_ub_ping[k]
                    T.pipe_barrier("V")

            if loop_iters > 0 and loop_iters % 2 == 1:
                i_last = loop_iters
                batch_idx_prev_last = cid + (i_last - 1) * num_blocks
                batch_idx_next_last = cid + (i_last + 1) * num_blocks

                token_idx_prev_last = batch_idx_prev_last * VEC_NUM + vid
                token_idx_next_last = T.min(batch_idx_next_last * VEC_NUM + vid, num_tokens - 1)

                T.barrier_all()
                T.copy(grad_out[token_idx_next_last, 0:num_topk], grad_out_ub_ping[0:num_topk])
                T.copy(topk_idx[token_idx_next_last, 0:num_topk], topk_idx_ub_ping[0:num_topk])

                T.copy(grad_scores_ub_ping[0:num_experts], grad_scores[token_idx_prev_last, 0:num_experts])

                T.tile.fill(grad_scores_ub_pong, 0.0)
                T.pipe_barrier("V")
                for k in range(num_topk):
                    idx_int32 = T.cast(topk_idx_ub_pong[k], INT32_DTYPE)
                    grad_scores_ub_pong[idx_int32] = grad_out_ub_pong[k]
                T.pipe_barrier("V")

            # Epilogue
            i_epi1 = my_iters - 2
            if i_epi1 >= 0 and i_epi1 % 2 == 0:
                batch_idx_epi1 = cid + i_epi1 * num_blocks
                token_idx_epi1 = batch_idx_epi1 * VEC_NUM + vid
                T.copy(grad_scores_ub_ping[0:num_experts], grad_scores[token_idx_epi1, 0:num_experts])
                T.barrier_all()
            elif i_epi1 >= 0:
                batch_idx_epi1 = cid + i_epi1 * num_blocks
                token_idx_epi1 = batch_idx_epi1 * VEC_NUM + vid
                T.copy(grad_scores_ub_pong[0:num_experts], grad_scores[token_idx_epi1, 0:num_experts])
                T.barrier_all()

            i_epi2 = my_iters - 1
            if i_epi2 > 0 and i_epi2 % 2 == 0:
                T.tile.fill(grad_scores_ub_ping, 0.0)
                T.pipe_barrier("V")
                for k in range(num_topk):
                    idx_int32 = T.cast(topk_idx_ub_ping[k], INT32_DTYPE)
                    grad_scores_ub_ping[idx_int32] = grad_out_ub_ping[k]
                T.pipe_barrier("V")
            elif i_epi2 > 0:
                T.tile.fill(grad_scores_ub_pong, 0.0)
                T.pipe_barrier("V")
                for k in range(num_topk):
                    idx_int32 = T.cast(topk_idx_ub_pong[k], INT32_DTYPE)
                    grad_scores_ub_pong[idx_int32] = grad_out_ub_pong[k]
                T.pipe_barrier("V")

            if i_epi2 >= 0 and i_epi2 % 2 == 0:
                batch_idx_epi2 = cid + i_epi2 * num_blocks
                token_idx_epi2 = T.min(batch_idx_epi2 * VEC_NUM + vid, num_tokens - 1)
                if batch_idx_epi2 * VEC_NUM + vid < num_tokens:
                    T.copy(grad_scores_ub_ping[0:num_experts], grad_scores[token_idx_epi2, 0:num_experts])
                T.barrier_all()
            elif i_epi2 >= 0:
                batch_idx_epi2 = cid + i_epi2 * num_blocks
                token_idx_epi2 = T.min(batch_idx_epi2 * VEC_NUM + vid, num_tokens - 1)
                if batch_idx_epi2 * VEC_NUM + vid < num_tokens:
                    T.copy(grad_scores_ub_pong[0:num_experts], grad_scores[token_idx_epi2, 0:num_experts])
                T.barrier_all()

    return topk_gate_backward_kernel


def topk_gate_backward(grad_out: torch.Tensor, topk_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    assert grad_out.dim() == 2 and grad_out.is_contiguous() and grad_out.dtype == torch.float32
    assert topk_idx.dim() == 2 and topk_idx.is_contiguous() and topk_idx.dtype == torch.int64
    num_tokens, num_topk = grad_out.shape
    assert topk_idx.shape == (num_tokens, num_topk)
    assert num_topk <= num_experts

    grad_scores = torch.empty((num_tokens, num_experts), dtype=torch.float32, device=grad_out.device)
    if num_tokens == 0:
        return grad_scores

    kernel = get_topk_gate_backward_kernel(num_experts, num_topk)
    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    kernel(grad_out, topk_idx, grad_scores)
    return grad_scores


def stable_topk(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    _, sorted_indices = torch.sort(scores, dim=1, descending=True, stable=True)
    return sorted_indices[:, :num_topk].contiguous()


def stable_topk_backward(grad_out: torch.Tensor, topk_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    grad_scores = torch.zeros((grad_out.shape[0], num_experts), dtype=grad_out.dtype, device=grad_out.device)
    grad_scores.scatter_(1, topk_idx, grad_out)
    return grad_scores


_EXPERT_CONFIGS = [
    (72, 6),
    (32, 6),
    (64, 6),
    (96, 6),
    (16, 6),
    (36, 6),
    (108, 6),
    (128, 6),
    (144, 6),
    (256, 8),
]


def generate_test_params() -> list[dict]:
    return [
        {
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "num_topk": num_topk,
        }
        for num_tokens in [4001, 8001]
        for num_experts, num_topk in _EXPERT_CONFIGS
    ]


def make_param_id(params: dict) -> str:
    nt = params["num_tokens"]
    ne = params["num_experts"]
    nk = params["num_topk"]
    return f"tokens={nt}_experts={ne}_topk={nk}"


@pytest.mark.parametrize("params", generate_test_params(), ids=make_param_id)
def test_topk_gate(params):
    num_tokens = params["num_tokens"]
    num_experts = params["num_experts"]
    num_topk = params["num_topk"]

    torch.manual_seed(42)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float)
    topk_idx_ref = stable_topk(scores, num_topk)

    if hasattr(torch, "npu") and torch.npu.is_available():
        scores = scores.to("npu").contiguous()
    elif torch.cuda.is_available():
        scores = scores.to("cuda").contiguous()

    topk_idx = topk_gate(scores, num_topk)

    np.testing.assert_equal(topk_idx.cpu().numpy(), topk_idx_ref.cpu().numpy())
    print(f"Forward Test passed for params: {make_param_id(params)}")


@pytest.mark.parametrize("params", generate_test_params(), ids=make_param_id)
def test_topk_gate_backward(params):
    num_tokens = params["num_tokens"]
    num_experts = params["num_experts"]
    num_topk = params["num_topk"]

    torch.manual_seed(42)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float)
    grad_out = torch.randn((num_tokens, num_topk), dtype=torch.float)
    topk_idx = stable_topk(scores, num_topk)

    if hasattr(torch, "npu") and torch.npu.is_available():
        grad_out = grad_out.to("npu").contiguous()
        topk_idx = topk_idx.to("npu").contiguous()
    elif torch.cuda.is_available():
        grad_out = grad_out.to("cuda").contiguous()
        topk_idx = topk_idx.to("cuda").contiguous()

    grad_scores_ref = stable_topk_backward(grad_out, topk_idx, num_experts)

    grad_scores_tl = topk_gate_backward(grad_out, topk_idx, num_experts)

    np.testing.assert_allclose(
        grad_scores_tl.cpu().numpy(),
        grad_scores_ref.cpu().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    print(f"Backward Test passed for params: {make_param_id(params)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
