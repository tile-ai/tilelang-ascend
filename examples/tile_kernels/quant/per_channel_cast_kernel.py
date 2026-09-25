import os
import sys
from collections.abc import Callable

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

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-5, -4], pass_configs=pass_configs)
def get_per_channel_cast_kernel(hidden: int, round_sf: bool):
    tile_m = 128
    block_m = 64
    compute_k = 128 if hidden % 128 == 0 else 64
    fp8_max_value = 448.0

    num_tokens = T.symbolic("num_tokens")
    num_tokens_out = T.symbolic("num_tokens_out")
    m_num = T.ceildiv(num_tokens_out, tile_m)
    k_num = T.ceildiv(hidden, compute_k)
    logical_aiv_tasks = m_num * k_num
    tasks_per_aiv = 2
    tasks_per_block = 2 * tasks_per_aiv
    kernel_blocks = T.ceildiv(logical_aiv_tasks, tasks_per_block)

    @T.prim_func
    def per_channel_cast_kernel(
        x: T.Tensor((num_tokens, hidden), "bfloat16"),
        out: T.Tensor((num_tokens_out, hidden), "float32"),
        out_sf: T.Tensor((T.ceildiv(num_tokens_out, tile_m), hidden), "float32"),
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
            x_abs_ub = T.alloc_ub((block_m, compute_k), "float32")
            amax_ub = T.alloc_ub((1, compute_k), "float32")
            local_amax_ub = T.alloc_ub((1, compute_k), "float32")
            sf_ub = T.alloc_ub((1, compute_k), "float32")
            amax_bits_ub = T.alloc_ub((1, compute_k), "int32")
            sf_bits_ub = T.alloc_ub((1, compute_k), "int32")

            if base_task < logical_aiv_tasks:
                first_pid_token = base_task // k_num
                first_pid_hidden = base_task % k_num
                first_row_offset = first_pid_token * tile_m
                first_col_offset = first_pid_hidden * compute_k
                T.copy(
                    x[
                        first_row_offset : first_row_offset + block_m,
                        first_col_offset : first_col_offset + compute_k,
                    ],
                    x_ub0,
                )
                T.copy(
                    x[
                        first_row_offset + block_m : first_row_offset + tile_m,
                        first_col_offset : first_col_offset + compute_k,
                    ],
                    x_ub1,
                )
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
                    T.pipe_barrier("v")
                    T.tile.abs(x_abs_ub, x_fp32_ub0)
                    T.pipe_barrier("v")
                    T.reduce_max(x_abs_ub, amax_ub, dim=0, clear=True)
                    T.pipe_barrier("v")
                    T.tile.abs(x_abs_ub, x_fp32_ub1)
                    T.pipe_barrier("v")
                    T.reduce_max(x_abs_ub, local_amax_ub, dim=0, clear=True)
                    T.pipe_barrier("v")
                    T.tile.max(amax_ub, amax_ub, local_amax_ub)
                    T.pipe_barrier("v")
                    T.tile.max(amax_ub, amax_ub, 1e-4)
                    T.pipe_barrier("v")

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

                    T.tile.broadcast(x_abs_ub, amax_ub, axis=0)
                    T.pipe_barrier("v")
                    if round_sf:
                        T.tile.mul(x_fp32_ub0, x_fp32_ub0, x_abs_ub)
                    else:
                        T.tile.div(x_fp32_ub0, x_fp32_ub0, x_abs_ub)
                        T.pipe_barrier("v")
                        T.tile.mul(x_fp32_ub0, x_fp32_ub0, fp8_max_value)

                    next_task_id = task_id + 1
                    T.wait_flag("v", "mte2", 4)
                    if task_stage + 1 < tasks_per_aiv and next_task_id < logical_aiv_tasks:
                        next_pid_token = next_task_id // k_num
                        next_pid_hidden = next_task_id % k_num
                        next_row_offset = next_pid_token * tile_m
                        next_col_offset = next_pid_hidden * compute_k
                        T.copy(
                            x[
                                next_row_offset : next_row_offset + block_m,
                                next_col_offset : next_col_offset + compute_k,
                            ],
                            x_ub0,
                        )
                        T.copy(
                            x[
                                next_row_offset + block_m : next_row_offset + tile_m,
                                next_col_offset : next_col_offset + compute_k,
                            ],
                            x_ub1,
                        )
                        T.set_flag("mte2", "v", 3)

                    T.set_flag("v", "mte3", 0)
                    T.wait_flag("v", "mte3", 0)
                    T.copy(sf_ub, out_sf[pid_token, local_col_offset : local_col_offset + compute_k])
                    T.copy(
                        x_fp32_ub0,
                        out[
                            row_offset : row_offset + block_m,
                            local_col_offset : local_col_offset + compute_k,
                        ],
                    )
                    T.set_flag("mte3", "v", 0)

                    if round_sf:
                        T.tile.mul(x_fp32_ub1, x_fp32_ub1, x_abs_ub)
                    else:
                        T.tile.div(x_fp32_ub1, x_fp32_ub1, x_abs_ub)
                        T.pipe_barrier("v")
                        T.tile.mul(x_fp32_ub1, x_fp32_ub1, fp8_max_value)
                    T.set_flag("v", "mte3", 1)
                    T.wait_flag("v", "mte3", 1)
                    T.copy(
                        x_fp32_ub1,
                        out[
                            row_offset + block_m : row_offset + tile_m,
                            local_col_offset : local_col_offset + compute_k,
                        ],
                    )
                    T.set_flag("mte3", "v", 1)

                    T.wait_flag("mte3", "v", 0)
                    T.wait_flag("mte3", "v", 1)

    return per_channel_cast_kernel


def per_channel_cast(
    x: torch.Tensor,
    fmt: str,
    num_per_tokens: int,
    round_sf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert fmt in ("fp32", "float32")
    assert num_per_tokens == 128
    assert x.is_contiguous() and x.dim() == 2

    x_data, x_sf, in_config = get_cast_input_and_config(x, (1, 1))
    assert x_sf is None and not in_config.with_sf
    assert x_data.dim() == 2 and x_data.is_contiguous()
    assert x_data.dtype == torch.bfloat16
    assert x_data.device.type == "npu"
    num_tokens, hidden = x_data.shape
    assert num_tokens % 128 == 0 and hidden % 64 == 0
    out_config = get_cast_output_config(fmt, (num_per_tokens, 1), round_sf=round_sf, custom_clamp_min_value=1e-4)

    if num_tokens == 0:
        return (
            torch.empty((0, hidden), dtype=torch.float32, device=x_data.device),
            torch.empty((0, hidden), dtype=out_config.sf_torch_dtype, device=x_data.device),
        )

    kernel = get_per_channel_cast_kernel(hidden, round_sf)
    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    x_sf_invs = torch.empty(
        (num_tokens, (hidden + 127) // 128),
        dtype=torch.float32,
        device=x_data.device,
    )
    pos_to_token = torch.zeros((num_tokens,), dtype=torch.int32, device=x_data.device)
    tok_dim_ref = torch.zeros((num_tokens,), dtype=torch.int32, device=x_data.device)
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


def per_channel_cast_ref(
    x: torch.Tensor,
    num_per_tokens: int,
    round_sf: bool,
    max_value: float = 448.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_fp32 = x.detach().to(device="cpu", dtype=torch.float32)
    num_tokens, hidden = x_fp32.shape
    num_blocks_m = (num_tokens + num_per_tokens - 1) // num_per_tokens
    out = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=x_fp32.device)
    out_sf = torch.zeros((num_blocks_m, hidden), dtype=torch.float32, device=x_fp32.device)

    for block_idx in range(num_blocks_m):
        row_start = block_idx * num_per_tokens
        row_end = min(row_start + num_per_tokens, num_tokens)
        block = x_fp32[row_start:row_end]
        amax = block.abs().amax(dim=0).clamp(min=1e-4)

        if round_sf:
            sf_value = amax / max_value
            sf = _round_sf_to_power_of_two_ref(sf_value)
            sf_inv = _round_sf_inv_to_power_of_two_ref(sf_value)
        else:
            sf = amax / max_value
            sf_inv = max_value / amax

        out_sf[block_idx] = sf
        out[row_start:row_end] = block * sf_inv

    return out, out_sf


def generate_num_tokens(alignment: int = 1, is_benchmark: bool = False) -> list[int]:
    base_list = [4001, 8001]
    full_list = [0, *base_list] if os.getenv("TK_FULL_TEST") in ("1", "true", "True") and not is_benchmark else base_list
    return [align_up(num_tokens, alignment) for num_tokens in full_list]


def generate_hidden_sizes(align: int = 64) -> list[int]:
    return [hidden for hidden in (576, 2048, 2560, 3072, 4096, 6144, 7168) if hidden % align == 0]


def generate_test_data(
    params: dict,
) -> tuple[torch.Tensor, Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    x = torch.randn(
        (params["num_tokens"], params["hidden"]),
        dtype=params["dtype"],
        device="npu",
    )

    def cast_func() -> tuple[torch.Tensor, torch.Tensor]:
        return per_channel_cast(
            x,
            "fp32",
            params["num_per_tokens"],
            params["round_sf"],
        )

    return x, cast_func


def generate_test_params() -> list[dict]:
    return [
        {
            "num_per_tokens": num_per_tokens,
            "num_tokens": num_tokens,
            "hidden": hidden_size,
            "round_sf": round_sf,
            "dtype": dtype,
        }
        for num_per_tokens in (128,)
        for num_tokens in generate_num_tokens(128)
        for hidden_size in generate_hidden_sizes()
        for round_sf in (False, True)
        for dtype in (torch.bfloat16,)
    ]


def make_param_id(params: dict) -> str:
    dtype_names = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float8_e4m3fn: "e4m3", torch.int8: "e2m1"}
    return "-".join(f"{key}={dtype_names.get(value, value)}" for key, value in params.items()) or "default"


@pytest.mark.parametrize("params", generate_test_params(), ids=make_param_id)
def test_per_channel_cast(params: dict) -> None:
    x, cast_func = generate_test_data(params)
    out, out_sf = cast_func()
    torch.npu.synchronize()
    out_ref, out_sf_ref = per_channel_cast_ref(
        x.detach().cpu(),
        params["num_per_tokens"],
        params["round_sf"],
    )

    assert out.dtype == torch.float32
    assert out_sf.dtype == torch.float32
    assert out.shape == out_ref.shape
    assert out_sf.shape == out_sf_ref.shape
    torch.testing.assert_close(out.cpu(), out_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_sf.cpu(), out_sf_ref, rtol=1e-3, atol=1e-3)

    out_fp8 = out.detach().cpu().to(torch.float8_e4m3fn)
    out_ref_fp8 = out_ref.to(torch.float8_e4m3fn)
    fp8_mismatches = torch.count_nonzero(out_fp8.float() != out_ref_fp8.float()).item()
    fp8_mismatch_rate = fp8_mismatches / out_fp8.numel()
    assert fp8_mismatch_rate <= 1e-3, f"FP8 bucket mismatch rate {fp8_mismatch_rate:.6e} ({fp8_mismatches}/{out_fp8.numel()}) exceeds 1e-3"


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v"])
    if exit_code == pytest.ExitCode.OK:
        print("All per_channel_cast tests passed! Kernel Output Match!")
    sys.exit(exit_code)
