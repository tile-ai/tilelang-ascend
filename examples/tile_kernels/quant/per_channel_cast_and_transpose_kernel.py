import os
import sys

import pytest
import tilelang
import tilelang.language as T
import torch

try:
    from .utils import *
except ImportError:
    from utils import *

pytest.importorskip("torch_npu")
os.environ["TILELANG_PRINT_ON_COMPILATION"] = "0"
tilelang.cache.clear_cache()

pass_configs = {tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True}


def _select_compute_k(hidden: int) -> int:
    for candidate_k in (96, 64):
        if hidden % candidate_k == 0:
            return candidate_k
    return 64


@tilelang.jit(out_idx=[-5, -4], pass_configs=pass_configs)
def get_per_channel_cast_and_transpose_kernel(hidden: int, num_per_tokens: int, round_sf: bool, tile_m: int):
    block_m = tile_m // 2
    sf_groups = tile_m // num_per_tokens
    groups_per_half = block_m // num_per_tokens
    reduce_m = min(block_m, num_per_tokens)
    group_scratch_m = reduce_m if num_per_tokens == 32 else 1
    abs_block_m = block_m if tile_m == 256 and num_per_tokens == 32 else 1

    base_compute_k = _select_compute_k(hidden)
    compute_k = base_compute_k // 2 if tile_m == 256 else base_compute_k
    fp8_max_value = 448.0

    assert num_per_tokens in (32, 128)
    assert tile_m in (128, 256)

    num_tokens = T.symbolic("num_tokens")
    num_tokens_out = T.symbolic("num_tokens_out")
    m_num = T.ceildiv(num_tokens_out, tile_m)
    k_num = T.ceildiv(hidden, compute_k)
    logical_aiv_tasks = m_num * k_num
    tasks_per_aiv = 4
    tasks_per_block = 2 * tasks_per_aiv
    kernel_blocks = T.ceildiv(logical_aiv_tasks, tasks_per_block)

    @T.prim_func
    def per_channel_cast_and_transpose_kernel(
        x: T.Tensor((num_tokens, hidden), "bfloat16"),
        out: T.Tensor((hidden, num_tokens_out), "float32"),
        out_sf: T.Tensor((T.ceildiv(num_tokens_out, num_per_tokens), hidden), "float32"),
        _x_sf_invs: T.Tensor((num_tokens, T.ceildiv(hidden, 128)), "float32"),
        _pos_to_token: T.Tensor((num_tokens_out,), "int32"),
        _tok_dim_ref: T.Tensor((num_tokens_out,), "int32"),
    ):
        with T.Kernel(kernel_blocks, is_npu=True) as (cid, vid):
            base_task = (cid * 2 + vid) * tasks_per_aiv
            x_ub0 = T.alloc_ub((block_m, compute_k), "bfloat16")
            x_ub1 = T.alloc_ub((block_m, compute_k), "bfloat16")
            x_fp32_ub0 = T.alloc_ub((block_m, compute_k), "float32")
            x_fp32_ub1 = T.alloc_ub((block_m, compute_k), "float32")
            x_abs_ub = T.alloc_ub((reduce_m, compute_k), "float32")
            x_abs_block_ub = T.alloc_ub((abs_block_m, compute_k), "float32")
            x_group_ub = T.alloc_ub((group_scratch_m, compute_k), "float32")
            amax_ub = T.alloc_ub((sf_groups, compute_k), "float32")
            local_amax_ub = T.alloc_ub((1, compute_k), "float32")
            sf_ub = T.alloc_ub((sf_groups, compute_k), "float32")
            amax_bits_ub = T.alloc_ub((sf_groups, compute_k), "int32")
            sf_bits_ub = T.alloc_ub((sf_groups, compute_k), "int32")
            transpose_stage_ub0 = T.alloc_ub((compute_k, block_m), "float32")
            transpose_stage_ub1 = T.alloc_ub((compute_k, block_m), "float32")
            transpose_offset_i32_ub = T.alloc_ub((block_m,), "int32")
            transpose_offset_u32_ub = T.alloc_ub((block_m,), "uint32")
            T.tile.arith_progression(transpose_offset_i32_ub, 0, compute_k * 4, block_m)
            T.pipe_barrier("v")

            if base_task < logical_aiv_tasks:
                first_pid_token = base_task // k_num
                first_pid_hidden = base_task % k_num
                first_row_offset = first_pid_token * tile_m
                first_col_offset = first_pid_hidden * compute_k
                T.copy(x[first_row_offset : first_row_offset + block_m, first_col_offset : first_col_offset + compute_k], x_ub0)
                T.copy(x[first_row_offset + block_m : first_row_offset + tile_m, first_col_offset : first_col_offset + compute_k], x_ub1)
                T.set_flag("mte2", "v", 3)

            for task_stage in T.unroll(tasks_per_aiv):
                task_id = base_task + task_stage
                if task_id < logical_aiv_tasks:
                    pid_token = task_id // k_num
                    pid_hidden = task_id % k_num
                    row_offset = pid_token * tile_m
                    local_col_offset = pid_hidden * compute_k

                    T.wait_flag("mte2", "v", 3)
                    T.tile.cast(x_fp32_ub0, x_ub0, mode="CAST_NONE", count=block_m * compute_k)
                    T.tile.cast(x_fp32_ub1, x_ub1, mode="CAST_NONE", count=block_m * compute_k)
                    T.set_flag("v", "mte2", 4)
                    next_task_id = task_id + 1
                    T.wait_flag("v", "mte2", 4)
                    if task_stage + 1 < tasks_per_aiv:  # noqa: SIM102
                        if next_task_id < logical_aiv_tasks:
                            next_pid_token = next_task_id // k_num
                            next_pid_hidden = next_task_id % k_num
                            next_row_offset = next_pid_token * tile_m
                            next_col_offset = next_pid_hidden * compute_k
                            T.copy(x[next_row_offset : next_row_offset + block_m, next_col_offset : next_col_offset + compute_k], x_ub0)
                            T.copy(
                                x[next_row_offset + block_m : next_row_offset + tile_m, next_col_offset : next_col_offset + compute_k],
                                x_ub1,
                            )
                            T.set_flag("mte2", "v", 3)
                    T.pipe_barrier("v")
                    if num_per_tokens == 128:
                        T.tile.abs(x_abs_ub, x_fp32_ub0)
                        T.pipe_barrier("v")
                        T.reduce_max(x_abs_ub, amax_ub[0, :], dim=0, clear=True)
                        T.pipe_barrier("v")
                        T.tile.abs(x_abs_ub, x_fp32_ub1)
                        T.pipe_barrier("v")
                        if groups_per_half == 1:
                            T.reduce_max(x_abs_ub, amax_ub[1, :], dim=0, clear=True)
                        else:
                            T.reduce_max(x_abs_ub, local_amax_ub, dim=0, clear=True)
                            T.pipe_barrier("v")
                            T.tile.max(amax_ub[0, :], amax_ub[0, :], local_amax_ub)
                    elif tile_m == 256:
                        T.tile.abs(x_abs_block_ub, x_fp32_ub0)
                        T.pipe_barrier("v")
                        for group_in_half in T.unroll(groups_per_half):
                            group_row = group_in_half * num_per_tokens
                            T.copy(x_abs_block_ub[group_row : group_row + num_per_tokens, :], x_group_ub)
                            T.pipe_barrier("v")
                            T.reduce_max(x_group_ub, amax_ub[group_in_half, :], dim=0, clear=True)
                            T.pipe_barrier("v")

                        T.tile.abs(x_abs_block_ub, x_fp32_ub1)
                        T.pipe_barrier("v")
                        for group_in_half in T.unroll(groups_per_half):
                            group_row = group_in_half * num_per_tokens
                            sf_group = groups_per_half + group_in_half
                            T.copy(x_abs_block_ub[group_row : group_row + num_per_tokens, :], x_group_ub)
                            T.pipe_barrier("v")
                            T.reduce_max(x_group_ub, amax_ub[sf_group, :], dim=0, clear=True)
                            T.pipe_barrier("v")
                    else:
                        for sf_group in T.unroll(sf_groups):
                            group_in_half = sf_group % groups_per_half
                            group_row = group_in_half * num_per_tokens
                            if sf_group < groups_per_half:
                                T.copy(x_fp32_ub0[group_row : group_row + num_per_tokens, :], x_group_ub)
                            else:
                                T.copy(x_fp32_ub1[group_row : group_row + num_per_tokens, :], x_group_ub)
                            T.pipe_barrier("v")
                            T.tile.abs(x_abs_ub, x_group_ub)
                            T.pipe_barrier("v")
                            T.reduce_max(x_abs_ub, amax_ub[sf_group, :], dim=0, clear=True)
                            T.pipe_barrier("v")

                    T.tile.max(amax_ub, amax_ub, 1e-4)
                    T.pipe_barrier("v")

                    if task_stage > 0:
                        T.wait_flag("mte3", "v", 0)
                    if round_sf:
                        with T.Scope("V"):
                            T.reinterpretcast(amax_bits_ub, amax_ub, "int32_t")
                            T.reinterpretcast(sf_bits_ub, sf_ub, "int32_t")
                            T.tile.add(sf_bits_ub, amax_bits_ub, 0x1FFFFF)
                            T.tile.bitwise_rshift(sf_bits_ub, sf_bits_ub, 23)
                            T.tile.fill(amax_bits_ub, 0xFF)
                            T.tile.bitwise_and(sf_bits_ub, sf_bits_ub, amax_bits_ub)
                            T.tile.add(sf_bits_ub, sf_bits_ub, -135)
                            T.tile.mul(amax_bits_ub, sf_bits_ub, -1)
                            T.tile.add(amax_bits_ub, amax_bits_ub, 127)
                            T.tile.max(amax_bits_ub, amax_bits_ub, 0)
                            T.tile.bitwise_lshift(amax_bits_ub, amax_bits_ub, 23)
                            T.tile.add(sf_bits_ub, sf_bits_ub, 127)
                            T.tile.bitwise_lshift(sf_bits_ub, sf_bits_ub, 23)
                    else:
                        T.tile.div(sf_ub, amax_ub, fp8_max_value)

                    T.pipe_barrier("v")
                    if num_per_tokens == 128:
                        T.tile.broadcast(x_abs_ub, amax_ub[0, :], axis=0)
                        T.pipe_barrier("v")
                        if round_sf:
                            T.tile.mul(x_fp32_ub0, x_fp32_ub0, x_abs_ub)
                        else:
                            T.tile.div(x_fp32_ub0, x_fp32_ub0, x_abs_ub)
                            T.pipe_barrier("v")
                            T.tile.mul(x_fp32_ub0, x_fp32_ub0, fp8_max_value)
                        T.pipe_barrier("v")
                        if groups_per_half == 1:
                            T.tile.broadcast(x_abs_ub, amax_ub[1, :], axis=0)
                            T.pipe_barrier("v")
                        if round_sf:
                            T.tile.mul(x_fp32_ub1, x_fp32_ub1, x_abs_ub)
                        else:
                            T.tile.div(x_fp32_ub1, x_fp32_ub1, x_abs_ub)
                            T.pipe_barrier("v")
                            T.tile.mul(x_fp32_ub1, x_fp32_ub1, fp8_max_value)
                    else:
                        for sf_group in T.unroll(sf_groups):
                            group_in_half = sf_group % groups_per_half
                            group_row = group_in_half * num_per_tokens
                            if sf_group < groups_per_half:
                                T.copy(x_fp32_ub0[group_row : group_row + num_per_tokens, :], x_group_ub)
                            else:
                                T.copy(x_fp32_ub1[group_row : group_row + num_per_tokens, :], x_group_ub)
                            T.pipe_barrier("v")
                            T.tile.broadcast(x_abs_ub, amax_ub[sf_group, :], axis=0)
                            T.pipe_barrier("v")
                            if round_sf:
                                T.tile.mul(x_group_ub, x_group_ub, x_abs_ub)
                            else:
                                T.tile.div(x_group_ub, x_group_ub, x_abs_ub)
                                T.pipe_barrier("v")
                                T.tile.mul(x_group_ub, x_group_ub, fp8_max_value)
                            T.pipe_barrier("v")
                            if sf_group < groups_per_half:
                                T.copy(x_group_ub, x_fp32_ub0[group_row : group_row + num_per_tokens, :])
                            else:
                                T.copy(x_group_ub, x_fp32_ub1[group_row : group_row + num_per_tokens, :])
                            T.pipe_barrier("v")

                    T.set_flag("v", "mte3", 0)
                    T.wait_flag("v", "mte3", 0)
                    T.copy(
                        sf_ub, out_sf[pid_token * sf_groups : (pid_token + 1) * sf_groups, local_col_offset : local_col_offset + compute_k]
                    )
                    T.set_flag("mte3", "v", 0)

                    if task_stage > 0:
                        T.wait_flag("mte3", "v", 1)
                    with T.Scope("V"):
                        T.reinterpretcast(transpose_offset_u32_ub, transpose_offset_i32_ub, "uint32_t")
                        for k_row in T.unroll(compute_k):
                            T.tile.gather(transpose_stage_ub0[k_row, :], x_fp32_ub0, transpose_offset_u32_ub, k_row * 4)
                    T.set_flag("v", "mte3", 1)
                    T.wait_flag("v", "mte3", 1)
                    T.copy(transpose_stage_ub0, out[local_col_offset : local_col_offset + compute_k, row_offset : row_offset + block_m])
                    T.set_flag("mte3", "v", 1)

                    if task_stage > 0:
                        T.wait_flag("mte3", "v", 2)
                    with T.Scope("V"):
                        T.reinterpretcast(transpose_offset_u32_ub, transpose_offset_i32_ub, "uint32_t")
                        for k_row in T.unroll(compute_k):
                            T.tile.gather(transpose_stage_ub1[k_row, :], x_fp32_ub1, transpose_offset_u32_ub, k_row * 4)
                    T.set_flag("v", "mte3", 2)
                    T.wait_flag("v", "mte3", 2)
                    T.copy(
                        transpose_stage_ub1,
                        out[local_col_offset : local_col_offset + compute_k, row_offset + block_m : row_offset + tile_m],
                    )
                    T.set_flag("mte3", "v", 2)

            if base_task < logical_aiv_tasks:
                T.wait_flag("mte3", "v", 0)
                T.wait_flag("mte3", "v", 1)
                T.wait_flag("mte3", "v", 2)

    return per_channel_cast_and_transpose_kernel


def per_channel_cast_and_transpose(
    x: torch.Tensor, fmt: str, num_per_tokens: int, round_sf: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    assert fmt in ("fp32", "float32")
    assert num_per_tokens in (32, 128)
    assert x.is_contiguous() and x.dim() == 2

    x_data, x_sf, in_config = get_cast_input_and_config(x, (1, 1))
    assert x_sf is None and not in_config.with_sf
    assert x_data.dim() == 2 and x_data.is_contiguous()
    assert x_data.dtype == torch.bfloat16
    assert x_data.device.type == "npu"

    num_tokens, hidden = x_data.shape
    assert num_tokens % 128 == 0 and hidden % 64 == 0
    tile_m = 256 if num_tokens % 256 == 0 else 128

    out_config = get_cast_output_config("fp32", (num_per_tokens, 1), round_sf=round_sf, custom_clamp_min_value=1e-4)

    if num_tokens == 0:
        return (
            torch.empty((hidden, 0), dtype=torch.float32, device=x_data.device),
            torch.empty((0, hidden), dtype=out_config.sf_torch_dtype, device=x_data.device),
        )

    kernel = get_per_channel_cast_and_transpose_kernel(hidden, num_per_tokens, round_sf, tile_m)
    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    x_sf_invs = torch.empty((num_tokens, (hidden + 127) // 128), dtype=torch.float32, device=x_data.device)
    pos_to_token = torch.empty((num_tokens,), dtype=torch.int32, device=x_data.device)
    tok_dim_ref = torch.empty((num_tokens,), dtype=torch.int32, device=x_data.device)
    return kernel(x_data, x_sf_invs, pos_to_token, tok_dim_ref)


def _round_sf_to_power_of_two_ref(sf: torch.Tensor) -> torch.Tensor:
    bits = sf.view(torch.int32)
    exp = ((bits - 1) >> 23) + 1 - 127
    exp = torch.clamp(127 + exp, 0, 255)
    return (exp.to(torch.int32) << 23).view(torch.float32)


def _round_sf_inv_to_power_of_two_ref(sf: torch.Tensor) -> torch.Tensor:
    bits = sf.view(torch.int32)
    exp = ((bits - 1) >> 23) + 1 - 127
    exp_inv = torch.clamp(127 - exp, 0, 255)
    return (exp_inv.to(torch.int32) << 23).view(torch.float32)


def per_channel_cast_and_transpose_ref(x: torch.Tensor, num_per_tokens: int, round_sf: bool) -> tuple[torch.Tensor, torch.Tensor]:
    x_fp32 = x.detach().cpu().float()
    num_tokens, hidden = x_fp32.shape
    num_blocks_m = (num_tokens + num_per_tokens - 1) // num_per_tokens
    out = torch.empty((num_tokens, hidden), dtype=torch.float32, device="cpu")
    out_sf = torch.empty((num_blocks_m, hidden), dtype=torch.float32, device="cpu")

    for block_idx in range(num_blocks_m):
        row_start = block_idx * num_per_tokens
        row_end = min(row_start + num_per_tokens, num_tokens)
        block = x_fp32[row_start:row_end]
        amax = block.abs().amax(dim=0).clamp(min=1e-4)
        sf_value = amax / 448.0

        if round_sf:
            sf = _round_sf_to_power_of_two_ref(sf_value)
            sf_inv = _round_sf_inv_to_power_of_two_ref(sf_value)
        else:
            sf = sf_value
            sf_inv = 448.0 / amax

        out_sf[block_idx] = sf
        out[row_start:row_end] = block * sf_inv

    return out.T.contiguous(), out_sf


def _generate_num_tokens() -> list[int]:
    num_tokens = [4001, 8001]
    if os.getenv("TK_FULL_TEST", "0").lower() in ("1", "true"):
        num_tokens.insert(0, 0)
    return [((value + 127) // 128) * 128 for value in num_tokens]


def _generate_hidden_sizes() -> list[int]:
    return [576, 2048, 2560, 3072, 4096, 6144, 7168]


def generate_test_params() -> list[dict]:
    return [
        {"num_tokens": num_tokens, "hidden": hidden_size, "round_sf": round_sf, "dtype": dtype, "num_per_tokens": num_per_tokens}
        for num_tokens in _generate_num_tokens()
        for hidden_size in _generate_hidden_sizes()
        for round_sf in (True, False)
        for dtype in (torch.bfloat16,)
        for num_per_tokens in (32, 128)
    ]


def _make_param_id(params: dict) -> str:
    return "-".join(f"{key}={value}" for key, value in params.items())


@pytest.mark.parametrize("params", generate_test_params(), ids=_make_param_id)
def test_per_channel_cast_and_transpose(params: dict) -> None:
    x = torch.randn((params["num_tokens"], params["hidden"]), dtype=params["dtype"], device="npu")

    x_fp8, x_sf = per_channel_cast_and_transpose(x, "fp32", params["num_per_tokens"], params["round_sf"])
    torch.npu.synchronize()

    out_ref, out_sf_ref = per_channel_cast_and_transpose_ref(x.detach().cpu(), params["num_per_tokens"], params["round_sf"])

    assert x_fp8.dtype == torch.float32
    assert x_sf.dtype == torch.float32
    assert x_fp8.shape == (params["hidden"], params["num_tokens"])
    assert x_sf.shape == out_sf_ref.shape
    torch.testing.assert_close(x_fp8.cpu(), out_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x_sf.cpu(), out_sf_ref, rtol=1e-3, atol=1e-3)

    out_fp8 = x_fp8.detach().cpu().to(torch.float8_e4m3fn)
    out_ref_fp8 = out_ref.to(torch.float8_e4m3fn)
    fp8_mismatches = torch.count_nonzero(out_fp8.float() != out_ref_fp8.float()).item()
    fp8_mismatch_rate = fp8_mismatches / out_fp8.numel()
    assert fp8_mismatch_rate <= 1e-3, f"FP8 bucket mismatch rate {fp8_mismatch_rate:.6e} ({fp8_mismatches}/{out_fp8.numel()}) exceeds 1e-3"


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v"])
    if exit_code == pytest.ExitCode.OK:
        print("All per_channel_cast_and_transpose tests passed! Kernel Output Match!")
    sys.exit(exit_code)
