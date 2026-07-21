import os
from importlib.util import find_spec
from typing import Optional

import torch
import tilelang
from tilelang import language as T

tilelang.cache.clear_cache()


def align(x, alignment):
    return (x + alignment - 1) // alignment * alignment


def ceil_div(x, divisor):
    return (x + divisor - 1) // divisor


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

HAS_NPU = find_spec("torch_npu") is not None


def get_device() -> str:
    if HAS_NPU and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def npu_sync_if_needed(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def dtype_to_tilelang_str(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


@tilelang.jit(pass_configs=pass_configs)
def get_expand_to_fused_backward_kernel(
    hidden: int,
    num_topk: int,
    num_per_channels: Optional[int],
    use_tma_aligned_col_major_sf: Optional[bool],
    use_packed_ue8m0: Optional[bool],
    x_dtype: str,
    sf_dtype: str,
):

    num_tokens = T.symbolic("num_tokens")
    num_expanded_tokens = T.symbolic("num_expanded_tokens")
    num_cores = 24
    rows_per_vec = 16 if hidden <= 1024 else 8 if hidden <= 2048 else 2 if hidden <= 4096 else 1
    tokens_per_block = rows_per_vec * 2
    num_token_blocks = T.ceildiv(num_tokens, tokens_per_block)
    num_iters = T.ceildiv(num_token_blocks, num_cores)
    aligned_topk = ((num_topk + 7) // 8) * 8

    need_cast = x_dtype not in ("float", "float32")
    ACC_DTYPE = "float32"

    if num_per_channels is not None:
        hidden_sf = ceil_div(hidden, num_per_channels)
        if use_packed_ue8m0:
            hidden_sf = ceil_div(hidden_sf, 4)
        hidden_sf_aligned = hidden_sf
    else:
        hidden_sf, hidden_sf_aligned = 1, 1

    sf_shape = (hidden_sf, num_expanded_tokens) if use_tma_aligned_col_major_sf else (num_expanded_tokens, hidden_sf)

    @T.prim_func
    def expand_to_fused_backward_kernel(
        grad_output: T.Tensor((num_expanded_tokens, hidden), x_dtype),
        token_topk_to_pos: T.Tensor((num_tokens, num_topk), "int32"),
        grad_x: T.Tensor((num_tokens, hidden), x_dtype),
        grad_x_sf: T.Tensor((num_tokens, hidden_sf), sf_dtype),
        grad_expanded_x_sf: T.Tensor(sf_shape, sf_dtype),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid), T.Scope("V"):
            pos_ub = T.alloc_ub((rows_per_vec, aligned_topk), "int32")
            grad_x_sf_fragment_ub = T.alloc_ub((hidden_sf_aligned,), sf_dtype)
            temp_grad_bf16_ub = T.alloc_ub((hidden,), x_dtype)
            temp_grad_f32_ub = T.alloc_ub((hidden,), ACC_DTYPE)
            temp_acc_ub = T.alloc_ub((hidden,), ACC_DTYPE)
            temp_output_bf16_vec_ub = T.alloc_ub((hidden,), x_dtype)
            temp_grad_sf_ub = T.alloc_ub((hidden_sf,), sf_dtype)
            temp_float_sf_ub = T.alloc_ub((hidden_sf,), ACC_DTYPE)
            temp_acc_sf_ub = T.alloc_ub((hidden_sf,), ACC_DTYPE)
            temp_output_sf_vec_ub = T.alloc_ub((hidden_sf,), sf_dtype)

            for iter_idx in T.serial(num_iters):
                if cid + iter_idx * num_cores < num_token_blocks:
                    for row in T.serial(rows_per_vec):
                        if (cid + iter_idx * num_cores) * tokens_per_block + vid * rows_per_vec + row < num_tokens:
                            T.tile.fill(temp_acc_ub, 0.0)
                            if num_per_channels is not None:
                                T.tile.fill(temp_acc_sf_ub, 0.0)

                            T.copy(
                                token_topk_to_pos[
                                    (cid + iter_idx * num_cores) * tokens_per_block + vid * rows_per_vec + row,
                                    :,
                                ],
                                pos_ub[row, :],
                            )

                            for k in T.unroll(num_topk):
                                if pos_ub[row, k] >= 0:
                                    T.copy(grad_output[pos_ub[row, k], :], temp_grad_bf16_ub)

                                    if need_cast:
                                        T.tile.cast(temp_grad_f32_ub, temp_grad_bf16_ub, "CAST_NONE", hidden)
                                        T.tile.add(temp_acc_ub, temp_acc_ub, temp_grad_f32_ub)
                                    else:
                                        T.tile.add(temp_acc_ub, temp_acc_ub, temp_grad_bf16_ub)

                                    if num_per_channels is not None:
                                        if use_tma_aligned_col_major_sf:
                                            T.copy(grad_expanded_x_sf[:, pos_ub[row, k]], temp_grad_sf_ub)
                                        else:
                                            T.copy(grad_expanded_x_sf[pos_ub[row, k], :], temp_grad_sf_ub)

                                        if need_cast:
                                            T.tile.cast(temp_float_sf_ub, temp_grad_sf_ub, "CAST_NONE", hidden_sf)
                                            T.tile.add(temp_acc_sf_ub, temp_acc_sf_ub, temp_float_sf_ub)
                                        else:
                                            if use_packed_ue8m0:
                                                for i in T.serial(hidden_sf):
                                                    temp_acc_sf_ub[i] += temp_grad_sf_ub[i]
                                            else:
                                                T.tile.add(temp_acc_sf_ub, temp_acc_sf_ub, temp_grad_sf_ub)

                            if need_cast:
                                T.tile.cast(temp_output_bf16_vec_ub, temp_acc_ub, "CAST_RINT", hidden)
                                T.copy(
                                    temp_output_bf16_vec_ub,
                                    grad_x[
                                        (cid + iter_idx * num_cores) * tokens_per_block + vid * rows_per_vec + row,
                                        :,
                                    ],
                                )
                            else:
                                T.copy(
                                    temp_acc_ub,
                                    grad_x[
                                        (cid + iter_idx * num_cores) * tokens_per_block + vid * rows_per_vec + row,
                                        :,
                                    ],
                                )

                            if num_per_channels is not None:
                                if need_cast:
                                    T.tile.cast(temp_output_sf_vec_ub, temp_acc_sf_ub, "CAST_RINT", hidden_sf)
                                    T.copy(
                                        temp_output_sf_vec_ub,
                                        grad_x_sf[
                                            (cid + iter_idx * num_cores) * tokens_per_block + vid * rows_per_vec + row,
                                            :,
                                        ],
                                    )
                                else:
                                    if use_packed_ue8m0:
                                        for i in T.serial(hidden_sf):
                                            grad_x_sf_fragment_ub[i] = temp_acc_sf_ub[i]
                                        T.copy(
                                            grad_x_sf_fragment_ub[0:hidden_sf],
                                            grad_x_sf[
                                                (cid + iter_idx * num_cores) * tokens_per_block + vid * rows_per_vec + row,
                                                :,
                                            ],
                                        )
                                    else:
                                        T.copy(
                                            temp_acc_sf_ub,
                                            grad_x_sf[
                                                (cid + iter_idx * num_cores) * tokens_per_block + vid * rows_per_vec + row,
                                                :,
                                            ],
                                        )

    return expand_to_fused_backward_kernel


@tilelang.jit(pass_configs=pass_configs)
def get_expand_to_fused_backward_full_h_init_acc_kernel(
    hidden: int,
    num_topk: int,
    x_dtype: str,
):
    num_tokens = T.symbolic("num_tokens")
    num_expanded_tokens = T.symbolic("num_expanded_tokens")

    assert num_topk > 0

    num_cores = 24
    rows_per_vec = 16 if hidden <= 1024 else 8 if hidden <= 2048 else 2 if hidden <= 4096 else 1
    tokens_per_block = rows_per_vec * 2
    num_token_blocks = T.ceildiv(num_tokens, tokens_per_block)
    num_iters = T.ceildiv(num_token_blocks, num_cores)
    aligned_topk = ((num_topk + 7) // 8) * 8

    need_cast = x_dtype not in ("float", "float32")
    ACC_DTYPE = "float32"

    @T.prim_func
    def expand_to_fused_backward_full_h_init_acc_kernel(
        grad_output: T.Tensor((num_expanded_tokens, hidden), x_dtype),
        token_topk_to_pos: T.Tensor((num_tokens, num_topk), "int32"),
        grad_x: T.Tensor((num_tokens, hidden), x_dtype),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid), T.Scope("V"):
            pos_ub = T.alloc_ub((rows_per_vec, aligned_topk), "int32")
            temp_grad_x_ub = T.alloc_ub((hidden,), x_dtype)
            temp_grad_f32_ub = T.alloc_ub((hidden,), ACC_DTYPE)
            temp_acc_ub = T.alloc_ub((hidden,), ACC_DTYPE)
            temp_output_x_ub = T.alloc_ub((hidden,), x_dtype)

            for iter_idx in T.serial(num_iters):
                if cid + iter_idx * num_cores < num_token_blocks:
                    for row in T.serial(rows_per_vec):
                        if (cid + iter_idx * num_cores) * tokens_per_block + vid * rows_per_vec + row < num_tokens:
                            T.copy(
                                token_topk_to_pos[
                                    (cid + iter_idx * num_cores) * tokens_per_block + vid * rows_per_vec + row,
                                    :,
                                ],
                                pos_ub[row, :],
                            )

                            if pos_ub[row, 0] >= 0:
                                if need_cast:
                                    T.copy(
                                        grad_output[pos_ub[row, 0], :],
                                        temp_grad_x_ub,
                                    )
                                    T.tile.cast(
                                        temp_acc_ub,
                                        temp_grad_x_ub,
                                        "CAST_NONE",
                                        hidden,
                                    )
                                else:
                                    T.copy(
                                        grad_output[pos_ub[row, 0], :],
                                        temp_acc_ub,
                                    )
                            else:
                                T.tile.fill(temp_acc_ub, 0.0)

                            for k_offset in T.unroll(num_topk - 1):
                                if pos_ub[row, k_offset + 1] >= 0:
                                    T.copy(
                                        grad_output[pos_ub[row, k_offset + 1], :],
                                        temp_grad_x_ub,
                                    )

                                    if need_cast:
                                        T.tile.cast(
                                            temp_grad_f32_ub,
                                            temp_grad_x_ub,
                                            "CAST_NONE",
                                            hidden,
                                        )
                                        T.tile.add(
                                            temp_acc_ub,
                                            temp_acc_ub,
                                            temp_grad_f32_ub,
                                        )
                                    else:
                                        T.tile.add(
                                            temp_acc_ub,
                                            temp_acc_ub,
                                            temp_grad_x_ub,
                                        )

                            if need_cast:
                                T.tile.cast(
                                    temp_output_x_ub,
                                    temp_acc_ub,
                                    "CAST_RINT",
                                    hidden,
                                )
                                T.copy(
                                    temp_output_x_ub,
                                    grad_x[
                                        (cid + iter_idx * num_cores) * tokens_per_block + vid * rows_per_vec + row,
                                        :,
                                    ],
                                )
                            else:
                                T.copy(
                                    temp_acc_ub,
                                    grad_x[
                                        (cid + iter_idx * num_cores) * tokens_per_block + vid * rows_per_vec + row,
                                        :,
                                    ],
                                )

    return expand_to_fused_backward_full_h_init_acc_kernel


def expand_to_fused_backward(
    grad_output: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    assert grad_output.is_contiguous() and token_topk_to_pos.is_contiguous()
    assert grad_output.dim() == 2 and token_topk_to_pos.dim() == 2

    num_expanded_tokens, hidden = grad_output.shape
    num_tokens_, num_topk = token_topk_to_pos.shape
    assert num_tokens == num_tokens_

    x_dtype_str = dtype_to_tilelang_str(grad_output.dtype)

    kernel = get_expand_to_fused_backward_full_h_init_acc_kernel(
        hidden,
        num_topk,
        x_dtype_str,
    )

    grad_x = torch.empty((num_tokens, hidden), dtype=grad_output.dtype, device=grad_output.device)
    if num_tokens > 0:
        kernel(grad_output, token_topk_to_pos, grad_x)

    return grad_x


def expand_to_fused_backward_profile_only(
    grad_output: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    assert grad_output.is_contiguous() and token_topk_to_pos.is_contiguous()
    assert grad_output.dim() == 2 and token_topk_to_pos.dim() == 2

    num_expanded_tokens, hidden = grad_output.shape
    num_tokens_, num_topk = token_topk_to_pos.shape
    assert num_tokens == num_tokens_

    x_dtype_str = dtype_to_tilelang_str(grad_output.dtype)

    kernel = get_expand_to_fused_backward_full_h_init_acc_kernel(
        hidden,
        num_topk,
        x_dtype_str,
    )

    grad_x = torch.empty((num_tokens, hidden), dtype=grad_output.dtype, device=grad_output.device)
    if num_tokens > 0:
        kernel(grad_output, token_topk_to_pos, grad_x)

    return grad_x


def expand_to_fused_backward_with_sf(
    grad_output: tuple,
    num_per_channels: int,
    token_topk_to_pos: torch.Tensor,
    num_tokens: int,
    use_tma_aligned_col_major_sf: bool = False,
) -> tuple:
    grad_data, grad_sf = grad_output
    assert grad_data.is_contiguous() and grad_sf.is_contiguous() and token_topk_to_pos.is_contiguous()
    assert grad_data.dim() == 2 and token_topk_to_pos.dim() == 2
    assert grad_sf.dtype in (torch.float32, torch.int32)
    assert num_per_channels in [32, 128]

    _num_expanded_tokens, hidden = grad_data.shape
    num_topk = token_topk_to_pos.shape[1]
    assert num_tokens == token_topk_to_pos.shape[0]

    hidden_sf = ceil_div(hidden, num_per_channels)

    use_packed_ue8m0 = False
    if grad_sf.dtype == torch.int32:
        use_packed_ue8m0 = True
        hidden_sf = ceil_div(hidden_sf, 4)
        assert use_tma_aligned_col_major_sf

    assert hidden_sf == grad_sf.shape[1] if not use_tma_aligned_col_major_sf else hidden_sf == grad_sf.shape[0]

    x_dtype_str = dtype_to_tilelang_str(grad_data.dtype)
    sf_dtype_str = dtype_to_tilelang_str(grad_sf.dtype)

    kernel = get_expand_to_fused_backward_kernel(
        hidden,
        num_topk,
        num_per_channels,
        use_tma_aligned_col_major_sf,
        use_packed_ue8m0,
        x_dtype_str,
        sf_dtype_str,
    )

    grad_x = torch.zeros((num_tokens, hidden), dtype=grad_data.dtype, device=grad_data.device)
    grad_x_sf = torch.zeros((num_tokens, hidden_sf), dtype=grad_sf.dtype, device=grad_sf.device)

    if num_tokens > 0:
        grad_sf_input = grad_sf.T if use_tma_aligned_col_major_sf else grad_sf
        kernel(grad_data, token_topk_to_pos, grad_x, grad_x_sf, grad_sf_input)

    return grad_x, grad_x_sf


def torch_expand_to_fused_backward_cpu(
    grad_output_cpu: torch.Tensor,
    token_topk_to_pos_cpu: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    assert grad_output_cpu.device.type == "cpu"
    assert token_topk_to_pos_cpu.device.type == "cpu"

    num_topk = token_topk_to_pos_cpu.shape[1]
    hidden = grad_output_cpu.shape[1]
    grad_x_float = torch.zeros((num_tokens, hidden), dtype=torch.float32, device="cpu")

    for token in range(num_tokens):
        for k in range(num_topk):
            pos = int(token_topk_to_pos_cpu[token, k])
            if pos >= 0:
                grad_x_float[token] += grad_output_cpu[pos].float()

    return grad_x_float.to(grad_output_cpu.dtype)


def make_topk_idx_cpu(num_tokens: int, num_topk: int, num_experts: int) -> torch.Tensor:
    topk_idx_cpu = torch.randint(
        0,
        num_experts,
        (num_tokens, num_topk),
        dtype=torch.int32,
        device="cpu",
    )

    if num_topk > 1:
        topk_idx_cpu[::7, -1] = -1

    return topk_idx_cpu.contiguous()


def make_fused_mapping_cpu(
    topk_idx_cpu: torch.Tensor,
    num_experts: int,
    alignment: int = 16,
):
    topk_idx_cpu = topk_idx_cpu.contiguous().cpu()
    num_tokens, num_topk = topk_idx_cpu.shape

    token_topk_to_pos = torch.full(
        (num_tokens, num_topk),
        -1,
        dtype=torch.int32,
        device="cpu",
    )

    pos_to_expert_parts = []
    next_pos = 0

    for expert in range(num_experts):
        token_ids, topk_ids = torch.where(topk_idx_cpu == expert)
        num_routed = token_ids.numel()

        if num_routed > 0:
            token_topk_to_pos[token_ids, topk_ids] = torch.arange(
                next_pos,
                next_pos + num_routed,
                dtype=torch.int32,
                device="cpu",
            )
            pos_to_expert_parts.append(torch.full((num_routed,), expert, dtype=torch.int32, device="cpu"))
            next_pos += num_routed

        num_padding = align(num_routed, alignment) - num_routed
        if num_padding > 0:
            pos_to_expert_parts.append(torch.full((num_padding,), -1, dtype=torch.int32, device="cpu"))
            next_pos += num_padding

    pos_to_expert = (
        torch.cat(pos_to_expert_parts).contiguous() if pos_to_expert_parts else torch.empty((0,), dtype=torch.int32, device="cpu")
    )

    return pos_to_expert, token_topk_to_pos.contiguous()


def get_test_configs():
    return [
        {"num_tokens": 500, "hidden": 128, "num_topk": 2, "num_experts": 4},
        {"num_tokens": 4001, "hidden": 576, "num_topk": 8, "num_experts": 4},
        {"num_tokens": 4001, "hidden": 2048, "num_topk": 8, "num_experts": 4},
        {"num_tokens": 4001, "hidden": 4096, "num_topk": 8, "num_experts": 4},
        {"num_tokens": 4001, "hidden": 7168, "num_topk": 8, "num_experts": 4},
        {"num_tokens": 4001, "hidden": 576, "num_topk": 9, "num_experts": 4},
        {"num_tokens": 4001, "hidden": 2048, "num_topk": 9, "num_experts": 4},
        {"num_tokens": 4001, "hidden": 4096, "num_topk": 9, "num_experts": 4},
        {"num_tokens": 4001, "hidden": 7168, "num_topk": 9, "num_experts": 4},
    ]


def prepare_case_on_cpu(config):
    num_tokens = config["num_tokens"]
    hidden = config["hidden"]
    num_topk = config["num_topk"]
    num_experts = config["num_experts"]

    topk_idx_cpu = make_topk_idx_cpu(num_tokens, num_topk, num_experts)
    pos_to_expert_cpu, token_topk_to_pos_cpu = make_fused_mapping_cpu(
        topk_idx_cpu,
        num_experts,
        alignment=16,
    )
    grad_output_cpu = torch.randn(
        (pos_to_expert_cpu.numel(), hidden),
        dtype=torch.bfloat16,
        device="cpu",
    )
    return grad_output_cpu, pos_to_expert_cpu, token_topk_to_pos_cpu


def run_profile_only_case(config):
    device = get_device()
    if device == "cpu":
        raise RuntimeError("Profile-only path needs NPU/CUDA device, but got CPU.")

    num_tokens = config["num_tokens"]
    hidden = config["hidden"]
    num_topk = config["num_topk"]
    num_experts = config["num_experts"]

    grad_output_cpu, pos_to_expert_cpu, token_topk_to_pos_cpu = prepare_case_on_cpu(config)

    grad_output = grad_output_cpu.contiguous().to(device)
    token_topk_to_pos = token_topk_to_pos_cpu.contiguous().to(device)
    npu_sync_if_needed(device)

    expand_to_fused_backward_profile_only(grad_output, token_topk_to_pos, num_tokens)
    npu_sync_if_needed(device)

    case = f"expand_to_fused_backward T={num_tokens} H={hidden} K={num_topk} E={num_experts} profile_only=True"
    print(f"pass {case}")


def run_correctness_case(config):
    device = get_device()

    num_tokens = config["num_tokens"]
    hidden = config["hidden"]
    num_topk = config["num_topk"]
    num_experts = config["num_experts"]

    grad_output_cpu, pos_to_expert_cpu, token_topk_to_pos_cpu = prepare_case_on_cpu(config)
    grad_output = grad_output_cpu.contiguous().to(device)
    token_topk_to_pos = token_topk_to_pos_cpu.contiguous().to(device)
    npu_sync_if_needed(device)

    grad_x = expand_to_fused_backward(grad_output, token_topk_to_pos, num_tokens)
    npu_sync_if_needed(device)

    grad_x_cpu = grad_x.cpu()
    ref_cpu = torch_expand_to_fused_backward_cpu(
        grad_output_cpu,
        token_topk_to_pos_cpu,
        num_tokens,
    )
    torch.testing.assert_close(grad_x_cpu, ref_cpu, rtol=1e-5, atol=1e-6, check_dtype=True)

    case = f"expand_to_fused_backward T={num_tokens} H={hidden} K={num_topk} E={num_experts} profile_only=False"
    print(f"pass {case}")


def main():
    torch.manual_seed(0)
    os.environ.setdefault("TILELANG_PRINT_ON_COMPILATION", "0")

    run_correctness = int(os.getenv("RUN_CORRECTNESS", "1")) == 1

    for config in get_test_configs():
        if run_correctness:
            run_correctness_case(config)
        else:
            run_profile_only_case(config)

    print("TEST PASSED!")


if __name__ == "__main__":
    main()
