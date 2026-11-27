import itertools
import os
import sys
from typing import Iterable, Optional, Union

import pytest
import tilelang
import tilelang.language as T
import torch

try:
    from .utils import *
except ImportError:
    from utils import *

pytest.importorskip("torch_npu")
tilelang.cache.clear_cache()

pass_configs = {tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True}


@tilelang.jit(out_idx=[-5, -4], pass_configs=pass_configs)
def get_per_channel_cast_fused_kernel(hidden: int, with_expand: bool, in_config: CastInputConfig, out_config: CastOutputConfig):
    TILE_M = 128
    BLOCK_M = 64
    FULL_M_TILES = TILE_M // BLOCK_M
    M_TAIL = TILE_M % BLOCK_M
    HAS_M_TAIL = M_TAIL != 0
    TAIL_BLOCK_M = M_TAIL if HAS_M_TAIL else 1
    M_TILES = FULL_M_TILES + int(HAS_M_TAIL)
    num_per_channels = 128

    with_sf = in_config.with_sf
    round_sf = out_config.round_sf
    in_dtype = in_config.dtype
    out_dtype = out_config.dtype
    in_sf_dtype = in_config.sf_dtype
    out_sf_dtype = out_config.sf_dtype
    FP8_MAX_VALUE = 448.0
    COMPUTE_K = 128 if hidden % 128 == 0 else 64
    K_GROUP = 2 if COMPUTE_K == 128 and hidden % 256 == 0 else 1
    LOCAL_K_GROUP = 1
    TASK_K = COMPUTE_K * K_GROUP
    SF_CHUNK = 8
    num_sf_total = ceil_div(hidden, num_per_channels)

    num_tokens = T.symbolic("num_tokens")
    num_tokens_out = T.symbolic("num_tokens_out")

    m_num = T.ceildiv(num_tokens_out, TILE_M)
    n_num = T.ceildiv(hidden, TASK_K)

    @T.prim_func
    def per_channel_cast_fused_kernel(
        x: T.Tensor((num_tokens, hidden), in_dtype),
        out: T.Tensor((num_tokens_out, hidden), out_dtype),
        out_sf: T.Tensor((T.ceildiv(num_tokens_out, TILE_M), hidden), out_sf_dtype),
        x_sf_invs: T.Tensor((num_tokens, num_sf_total), in_sf_dtype),
        pos_to_token: T.Tensor((num_tokens_out,), "int32"),
        _tok_dim_ref: T.Tensor((num_tokens_out,), "int32"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            pid_token = cid // n_num
            pid_hidden = cid % n_num
            row_offset = pid_token * TILE_M
            col_offset = pid_hidden * TASK_K
            local_k_sub = vid * (K_GROUP - 1)
            local_col_offset = col_offset + local_k_sub * COMPUTE_K
            x_ub = T.alloc_ub((TILE_M, COMPUTE_K), in_dtype)
            x_block_ub = T.alloc_ub((BLOCK_M, COMPUTE_K), in_dtype)
            tail_x_block_ub = T.alloc_ub((TAIL_BLOCK_M, COMPUTE_K), in_dtype)
            x_fp32_ub = T.alloc_ub((BLOCK_M, COMPUTE_K), out_dtype)
            scale_tile_ub = T.alloc_ub((BLOCK_M, COMPUTE_K), out_dtype)
            tail_x_fp32_ub = T.alloc_ub((TAIL_BLOCK_M, COMPUTE_K), out_dtype)
            tail_scale_tile_ub = T.alloc_ub((TAIL_BLOCK_M, COMPUTE_K), out_dtype)
            amax_ub = T.alloc_ub((1, COMPUTE_K), out_dtype)
            local_amax_ub = T.alloc_ub((1, COMPUTE_K), out_dtype)
            tail_local_amax_ub = T.alloc_ub((1, COMPUTE_K), out_dtype)
            sf_ub = T.alloc_ub((1, COMPUTE_K), out_dtype)
            reciprocal_ub = T.alloc_ub((1, COMPUTE_K), out_dtype)
            newton_ub = T.alloc_ub((1, COMPUTE_K), out_dtype)
            sf_chunk_ub = T.alloc_ub((TILE_M, SF_CHUNK), in_sf_dtype)
            sf_invs_ub = T.alloc_ub((LOCAL_K_GROUP, TILE_M, 1), in_sf_dtype)
            sf_pair_ub = T.alloc_ub((TILE_M * 2,), in_sf_dtype)
            sf_selected_ub = T.alloc_ub((TILE_M,), in_sf_dtype)
            sf_block_ub = T.alloc_ub((BLOCK_M, 1), in_sf_dtype)
            tail_sf_block_ub = T.alloc_ub((TAIL_BLOCK_M, 1), in_sf_dtype)
            sf_col_base = local_col_offset // num_per_channels
            sf_col_start = (sf_col_base // SF_CHUNK) * SF_CHUNK
            sf_col_count = SF_CHUNK
            if sf_col_start + SF_CHUNK > num_sf_total:
                sf_col_count = num_sf_total - sf_col_start
            sf_col_in_chunk = sf_col_base - sf_col_start

            if with_expand:
                pt_ub = T.alloc_ub((TILE_M,), "int32")
                T.tile.fill(x_ub, 0.0)
                if with_sf:
                    T.tile.fill(sf_chunk_ub, 0.0)
                T.copy(pos_to_token[row_offset : row_offset + TILE_M], pt_ub)
                T.set_flag("mte2", "s", 0)
                T.wait_flag("mte2", "s", 0)
                T.set_flag("v", "mte2", 0)
                T.wait_flag("v", "mte2", 0)
                for row_tile in T.serial(M_TILES):
                    for i in T.serial(BLOCK_M):
                        tile_row = row_tile * BLOCK_M + i
                        if tile_row < TILE_M and row_offset + tile_row < num_tokens_out:
                            pos = T.alloc_var("int32", init=pt_ub[tile_row])
                            if pos >= 0 and pos < num_tokens:
                                if with_sf:
                                    T.copy(
                                        x_sf_invs[pos : pos + 1, sf_col_start : sf_col_start + sf_col_count],
                                        sf_chunk_ub[tile_row, 0:sf_col_count],
                                    )
                                T.copy(x[pos, local_col_offset : local_col_offset + COMPUTE_K], x_ub[tile_row, 0:COMPUTE_K])
            else:
                T.copy(x[row_offset : row_offset + TILE_M, local_col_offset : local_col_offset + COMPUTE_K], x_ub)
                if with_sf:
                    T.copy(
                        x_sf_invs[row_offset : row_offset + TILE_M, sf_col_start : sf_col_start + sf_col_count],
                        sf_chunk_ub[0:TILE_M, 0:sf_col_count],
                    )

            if with_sf:
                T.set_flag("mte2", "v", 1)
                T.wait_flag("mte2", "v", 1)
                for k_sub in T.unroll(LOCAL_K_GROUP):
                    selected_col = sf_col_in_chunk + k_sub
                    selected_mod4 = selected_col % 4
                    if selected_mod4 == 0:
                        T.tile.gather_mask(sf_pair_ub, sf_chunk_ub, "P0001")
                    elif selected_mod4 == 1:
                        T.tile.gather_mask(sf_pair_ub, sf_chunk_ub, "P0010")
                    elif selected_mod4 == 2:
                        T.tile.gather_mask(sf_pair_ub, sf_chunk_ub, "P0100")
                    else:
                        T.tile.gather_mask(sf_pair_ub, sf_chunk_ub, "P1000")
                    if selected_col < 4:
                        T.tile.gather_mask(sf_selected_ub, sf_pair_ub, "P0101")
                    else:
                        T.tile.gather_mask(sf_selected_ub, sf_pair_ub, "P1010")
                    T.copy(sf_selected_ub, sf_invs_ub[k_sub, 0:TILE_M, 0:1])
            else:
                T.set_flag("mte2", "v", 0)
                T.wait_flag("mte2", "v", 0)

            for k_sub in T.unroll(LOCAL_K_GROUP):
                sub_col_offset = local_col_offset + k_sub * COMPUTE_K

                T.tile.fill(amax_ub, 0.0)
                for row_tile in T.serial(FULL_M_TILES):
                    T.copy(x_ub[row_tile * BLOCK_M : (row_tile + 1) * BLOCK_M, k_sub * COMPUTE_K : (k_sub + 1) * COMPUTE_K], x_block_ub)
                    T.tile.cast(x_fp32_ub, x_block_ub, mode="CAST_NONE", count=BLOCK_M * COMPUTE_K)
                    if with_sf:
                        T.copy(sf_invs_ub[k_sub, row_tile * BLOCK_M : (row_tile + 1) * BLOCK_M, 0:1], sf_block_ub)
                        T.tile.broadcast(scale_tile_ub, sf_block_ub, axis=1)
                        T.pipe_barrier("v")
                        T.tile.mul(x_fp32_ub, x_fp32_ub, scale_tile_ub)
                    T.pipe_barrier("v")
                    T.tile.abs(x_fp32_ub, x_fp32_ub)
                    T.pipe_barrier("v")
                    T.reduce_max(x_fp32_ub, local_amax_ub, dim=0, clear=True)
                    T.pipe_barrier("v")
                    T.tile.max(amax_ub, amax_ub, local_amax_ub)
                if HAS_M_TAIL:
                    T.copy(x_ub[FULL_M_TILES * BLOCK_M : TILE_M, k_sub * COMPUTE_K : (k_sub + 1) * COMPUTE_K], tail_x_block_ub)
                    T.tile.cast(tail_x_fp32_ub, tail_x_block_ub, mode="CAST_NONE", count=M_TAIL * COMPUTE_K)
                    if with_sf:
                        T.copy(sf_invs_ub[k_sub, FULL_M_TILES * BLOCK_M : TILE_M, 0:1], tail_sf_block_ub)
                        T.tile.broadcast(tail_scale_tile_ub, tail_sf_block_ub, axis=1)
                        T.pipe_barrier("v")
                        T.tile.mul(tail_x_fp32_ub, tail_x_fp32_ub, tail_scale_tile_ub)
                    T.pipe_barrier("v")
                    T.tile.abs(tail_x_fp32_ub, tail_x_fp32_ub)
                    T.pipe_barrier("v")
                    T.reduce_max(tail_x_fp32_ub, tail_local_amax_ub, dim=0, clear=True)
                    T.pipe_barrier("v")
                    T.tile.max(amax_ub, amax_ub, tail_local_amax_ub)

                if round_sf:
                    T.set_flag("v", "s", 0)
                    T.wait_flag("v", "s", 0)
                    for j in T.serial(COMPUTE_K):
                        clamped_amax = T.max(amax_ub[0, j], 1e-4)
                        amax_bits = T.reinterpret("int32", clamped_amax)
                        candidate_exp = ((amax_bits >> 23) & 0xFF) - 135
                        exp_sf = T.alloc_var("int32", init=candidate_exp)
                        if (amax_bits & 0x7FFFFF) > 0x600000:
                            exp_sf = candidate_exp + 1
                        sf_inv_exp = T.max(127 - exp_sf, 0)
                        sf_inv_bits = T.alloc_var("int32", init=sf_inv_exp << 23)
                        sf_bits = T.alloc_var("int32", init=(127 + exp_sf) << 23)
                        amax_ub[0, j] = T.reinterpret("float32", sf_inv_bits)
                        sf_ub[0, j] = T.reinterpret("float32", sf_bits)
                    T.set_flag("s", "mte3", 0)
                    T.wait_flag("s", "mte3", 0)
                    T.set_flag("s", "v", 0)
                    T.wait_flag("s", "v", 0)
                else:
                    T.tile.max(amax_ub, amax_ub, 1e-4)
                    T.pipe_barrier("v")
                    T.tile.div(sf_ub, amax_ub, FP8_MAX_VALUE)
                    if with_sf:
                        T.tile.reciprocal(reciprocal_ub, amax_ub)
                        T.pipe_barrier("v")
                        T.tile.mul(newton_ub, amax_ub, reciprocal_ub)
                        T.pipe_barrier("v")
                        T.tile.mul(newton_ub, newton_ub, -1.0)
                        T.tile.add(newton_ub, newton_ub, 2.0)
                        T.pipe_barrier("v")
                        T.tile.mul(reciprocal_ub, reciprocal_ub, newton_ub)
                        T.pipe_barrier("v")
                        T.tile.mul(amax_ub, reciprocal_ub, FP8_MAX_VALUE)
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                    else:
                        T.set_flag("v", "s", 0)
                        T.wait_flag("v", "s", 0)
                        for j in T.serial(COMPUTE_K):
                            amax_ub[0, j] = FP8_MAX_VALUE / amax_ub[0, j]
                        T.set_flag("s", "mte3", 0)
                        T.wait_flag("s", "mte3", 0)
                        T.set_flag("s", "v", 0)
                        T.wait_flag("s", "v", 0)

                if vid < K_GROUP:
                    T.copy(sf_ub, out_sf[pid_token, sub_col_offset : sub_col_offset + COMPUTE_K])
                    T.set_flag("mte3", "v", 2)

                for row_tile in T.unroll(FULL_M_TILES):
                    tile_row_offset = row_offset + row_tile * BLOCK_M
                    T.copy(x_ub[row_tile * BLOCK_M : (row_tile + 1) * BLOCK_M, k_sub * COMPUTE_K : (k_sub + 1) * COMPUTE_K], x_block_ub)
                    if with_sf:
                        T.copy(sf_invs_ub[k_sub, row_tile * BLOCK_M : (row_tile + 1) * BLOCK_M, 0:1], sf_block_ub)
                    if vid < K_GROUP and row_tile > 0:
                        T.wait_flag("mte3", "v", 0)
                    T.tile.cast(x_fp32_ub, x_block_ub, mode="CAST_NONE", count=BLOCK_M * COMPUTE_K)
                    if with_sf:
                        T.tile.broadcast(scale_tile_ub, sf_block_ub, axis=1)
                        T.pipe_barrier("v")
                        T.tile.mul(x_fp32_ub, x_fp32_ub, scale_tile_ub)
                        T.pipe_barrier("v")
                    T.tile.broadcast(scale_tile_ub, amax_ub, axis=0)
                    T.pipe_barrier("v")
                    T.tile.mul(x_fp32_ub, x_fp32_ub, scale_tile_ub)

                    if vid < K_GROUP:
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.copy(x_fp32_ub, out[tile_row_offset : tile_row_offset + BLOCK_M, sub_col_offset : sub_col_offset + COMPUTE_K])
                        T.set_flag("mte3", "v", 0)
                if HAS_M_TAIL:
                    tail_row_offset = row_offset + FULL_M_TILES * BLOCK_M
                    T.copy(x_ub[FULL_M_TILES * BLOCK_M : TILE_M, k_sub * COMPUTE_K : (k_sub + 1) * COMPUTE_K], tail_x_block_ub)
                    T.tile.cast(tail_x_fp32_ub, tail_x_block_ub, mode="CAST_NONE", count=M_TAIL * COMPUTE_K)
                    if with_sf:
                        T.copy(sf_invs_ub[k_sub, FULL_M_TILES * BLOCK_M : TILE_M, 0:1], tail_sf_block_ub)
                        T.tile.broadcast(tail_scale_tile_ub, tail_sf_block_ub, axis=1)
                        T.pipe_barrier("v")
                        T.tile.mul(tail_x_fp32_ub, tail_x_fp32_ub, tail_scale_tile_ub)
                        T.pipe_barrier("v")
                    T.tile.broadcast(tail_scale_tile_ub, amax_ub, axis=0)
                    T.pipe_barrier("v")
                    T.tile.mul(tail_x_fp32_ub, tail_x_fp32_ub, tail_scale_tile_ub)
                    if vid < K_GROUP:
                        T.set_flag("v", "mte3", 1)
                        T.wait_flag("v", "mte3", 1)
                        T.copy(tail_x_fp32_ub, out[tail_row_offset : tail_row_offset + M_TAIL, sub_col_offset : sub_col_offset + COMPUTE_K])
                        T.set_flag("mte3", "v", 1)
                        T.wait_flag("mte3", "v", 1)
                if vid < K_GROUP:
                    T.wait_flag("mte3", "v", 0)
                    T.wait_flag("mte3", "v", 2)

    return per_channel_cast_fused_kernel


def per_channel_cast_fused(
    x, num_per_tokens: int, round_sf: bool = False, num_per_channels: Optional[int] = None, pos_to_token: Optional[torch.Tensor] = None
) -> tuple:
    x_data, x_sf_invs, in_config = get_cast_input_and_config(x, (1, num_per_channels) if num_per_channels is not None else (1, 1))
    assert x_data.dim() == 2 and x_data.is_contiguous()
    num_tokens, hidden = x_data.shape
    num_tokens_out = num_tokens

    if pos_to_token is not None:
        assert pos_to_token.dim() == 1 and pos_to_token.is_contiguous()
        assert pos_to_token.dtype == torch.int32
        assert pos_to_token.device == x_data.device
        num_tokens_out = pos_to_token.size(0)

    assert num_tokens_out % 128 == 0
    assert num_per_tokens == 128
    tile_k = 128 if hidden % 128 == 0 else 64
    assert hidden % tile_k == 0
    if x_sf_invs is not None:
        assert num_per_channels == 128
        assert x_sf_invs.dim() == 2 and x_sf_invs.is_contiguous()
        assert x_sf_invs.dtype == in_config.sf_torch_dtype
        assert x_sf_invs.device == x_data.device
        assert x_sf_invs.size(0) == num_tokens
        assert x_sf_invs.size(1) * 128 == hidden

    out_config = get_cast_output_config("fp32", (num_per_tokens, 1), round_sf=round_sf)
    kernel = get_per_channel_cast_fused_kernel(hidden, with_expand=(pos_to_token is not None), in_config=in_config, out_config=out_config)

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    if num_tokens_out > 0:
        _x_sf_invs = (
            x_sf_invs
            if x_sf_invs is not None
            else torch.empty((num_tokens, ceil_div(hidden, 128)), dtype=torch.float32, device=x_data.device)
        )
        _pos_to_token = (
            pos_to_token if pos_to_token is not None else torch.zeros((num_tokens_out,), dtype=torch.int32, device=x_data.device)
        )
        _tok_dim_ref = torch.zeros((num_tokens_out,), dtype=torch.int32, device=x_data.device)
        out, out_sf = kernel(x_data, _x_sf_invs, _pos_to_token, _tok_dim_ref)
    else:
        out = torch.empty((0, hidden), dtype=torch.float32, device=x_data.device)
        out_sf = torch.empty((0, hidden), dtype=torch.float32, device=x_data.device)

    return out, out_sf


def _cast_fp32_to_fp8_cpu(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float32
    return x.detach().cpu().to(torch.float8_e4m3fn)


def generate_hidden_sizes(align: int = 64) -> list[int]:
    base_list = [576, 2048, 2560, 3072, 4096, 6144, 7168]
    return [h for h in base_list if h % align == 0]


def generate_moe_params() -> Iterable[dict]:
    do_full_test = os.getenv("TK_FULL_TEST") in ["1", "true", "True"]
    extra_num_topk_list = (1, 7) if do_full_test else ()
    extra_num_experts_list = (288, 384) if do_full_test else ()
    extra_num_ep_ranks_list = (1, 72, 256) if do_full_test else ()

    if do_full_test:
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
    num_topk = params["num_topk"]
    num_experts = params["num_experts"]
    num_ep_ranks = params["num_ep_ranks"]

    if num_send_tokens == 0:
        return torch.empty((0, num_topk), dtype=torch.int64)
    topk_idx = torch.randint(
        0,
        num_experts * num_ep_ranks,
        (num_send_tokens * num_ep_ranks, num_topk),
        dtype=torch.int64,
    )
    mask = topk_idx >= num_experts
    topk_idx[mask] = -1
    mask = mask.all(dim=1)
    topk_idx = topk_idx[~mask]
    return topk_idx


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


def per_channel_cast_fused_ref(
    x: Union[torch.Tensor, tuple],
    num_per_tokens: int,
    num_per_channels: Optional[int],
    round_sf: bool,
    pos_to_token: Optional[torch.Tensor],
    max_value: float = 448.0,
) -> tuple:
    is_fused_cast_back = isinstance(x, tuple)
    num_per_channels = num_per_channels if is_fused_cast_back else None

    if pos_to_token is not None:
        x_data = x[0] if is_fused_cast_back else x
        x_gathered = x_data[pos_to_token.clamp(min=0)]
        valid_mask = (pos_to_token >= 0).unsqueeze(1)
        x_gathered = torch.where(valid_mask, x_gathered.to(torch.float32), torch.zeros_like(x_gathered, dtype=torch.float32)).to(
            x_data.dtype
        )
        if is_fused_cast_back:
            x_sf = x[1]
            x_sf_gathered = x_sf[pos_to_token.clamp(min=0)]
            x_sf_gathered = torch.where(valid_mask, x_sf_gathered, torch.zeros_like(x_sf_gathered))
            x = (x_gathered, x_sf_gathered)
        else:
            x = x_gathered

    if is_fused_cast_back:
        x_data, x_sf_invs = x
        x_fp32 = x_data.float()
        x_sf_invs = x_sf_invs.float()
        assert num_per_channels is not None
    else:
        x_data = x
        x_fp32 = x_data.float()
        x_sf_invs = None
        num_per_channels = None

    num_tokens, hidden = x_fp32.shape

    out = torch.zeros(num_tokens, hidden, dtype=torch.float32)
    out_sf = torch.zeros(ceil_div(num_tokens, num_per_tokens), hidden, dtype=torch.float32)

    num_blocks_m = ceil_div(num_tokens, num_per_tokens)

    for bm in range(num_blocks_m):
        r0 = bm * num_per_tokens
        r1 = min(r0 + num_per_tokens, num_tokens)

        block = x_fp32[r0:r1, :].clone()

        if x_sf_invs is not None:
            for c0 in range(0, hidden, num_per_channels):
                c1 = min(c0 + num_per_channels, hidden)
                k_idx = c0 // num_per_channels
                sf_col = x_sf_invs[r0:r1, k_idx].unsqueeze(1)
                block[:, c0:c1] = block[:, c0:c1] * sf_col

        amax = block.abs().amax(dim=0, keepdim=False).clamp(min=1e-4)

        if round_sf:
            sf_val = amax / max_value
            sf = _round_sf_to_power_of_two_ref(sf_val)
            sf_inv = _round_sf_inv_to_power_of_two_ref(sf_val)
        else:
            sf = amax / max_value
            sf_inv = max_value / amax

        out_sf[bm, :] = sf

        if x_sf_invs is not None:
            for c0 in range(0, hidden, num_per_channels):
                c1 = min(c0 + num_per_channels, hidden)
                k_idx = c0 // num_per_channels
                sf_col = x_sf_invs[r0:r1, k_idx].unsqueeze(1)
                out[r0:r1, c0:c1] = x_fp32[r0:r1, c0:c1] * sf_col * sf_inv[c0:c1]
        else:
            out[r0:r1, :] = x_fp32[r0:r1, :] * sf_inv

    return out, out_sf


def _generate_random_pos_to_token(num_tokens: int, num_topk: int, topk_idx: torch.Tensor, device: str) -> torch.Tensor:
    valid_count = (topk_idx >= 0).sum().item()
    padded = ((valid_count + 127) // 128) * 128
    if padded == 0:
        padded = 128
    pt = torch.randint(0, num_tokens, (padded,), dtype=torch.int32, device=device)
    pt[valid_count:] = -1
    return pt


def generate_test_data(params: dict):
    num_send_tokens = params["num_send_tokens"]
    num_topk = params["num_topk"]
    hidden = params["hidden"]
    num_per_tokens = params["num_per_tokens"]
    num_per_channels = params["num_per_channels"]
    is_fused_cast_back = params["is_fused_cast_back"]
    round_sf = params["round_sf"]
    device = "npu"

    pos_to_token = None
    if num_topk > 0:
        topk_idx = generate_topk_idx(params)
        num_tokens = topk_idx.shape[0]
        pos_to_token = _generate_random_pos_to_token(num_tokens, num_topk, topk_idx, device)
        x = torch.randn((pos_to_token.shape[0], hidden), dtype=torch.bfloat16, device=device)
    else:
        num_tokens = num_send_tokens
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device)

    if is_fused_cast_back:
        sf_shape = (x.shape[0], ceil_div(hidden, num_per_channels))
        x_sf_invs_val = torch.rand(sf_shape, dtype=torch.float32, device=device) * 0.5 + 0.5
        x = (x, x_sf_invs_val)

    def func():
        return per_channel_cast_fused(
            x,
            num_per_tokens=num_per_tokens,
            round_sf=round_sf,
            num_per_channels=num_per_channels if is_fused_cast_back else None,
            pos_to_token=pos_to_token,
        )

    def func_ref():
        ch = num_per_channels if is_fused_cast_back else None
        x_cpu = (x[0].cpu(), x[1].cpu()) if isinstance(x, tuple) else x.cpu()
        pt_cpu = pos_to_token.cpu() if pos_to_token is not None else None
        return per_channel_cast_fused_ref(x_cpu, num_per_tokens, ch, round_sf, pt_cpu, max_value=448.0)

    return x, num_tokens, pos_to_token, func, func_ref


def generate_test_params() -> list[dict]:
    return [
        {
            **moe,
            "hidden": hidden_size,
            "num_per_tokens": num_per_tokens,
            "num_per_channels": num_per_channels,
            "is_fused_cast_back": is_fused_cast_back,
            "round_sf": round_sf,
        }
        for moe in itertools.chain(
            iter([{"num_send_tokens": 4096, "num_topk": 0, "num_experts": 0, "num_ep_ranks": 0}]), generate_moe_params()
        )
        for hidden_size in generate_hidden_sizes(128)
        for num_per_tokens, num_per_channels in [(128, 128)]
        for is_fused_cast_back in (False, True)
        for round_sf in (False, True)
    ]


def _check_case(params: dict) -> None:
    _, _, _, func, func_ref = generate_test_data(params)

    out, out_sf = func()
    torch.npu.synchronize()

    ref_out, ref_sf = func_ref()

    assert out.dtype == torch.float32
    assert out_sf.dtype == torch.float32
    assert out.shape == ref_out.shape
    assert out_sf.shape == ref_sf.shape
    out_cpu = out.cpu()
    ref_out_cpu = ref_out.cpu()
    out_sf_cpu = out_sf.cpu()
    ref_sf_cpu = ref_sf.cpu()

    torch.testing.assert_close(out_cpu, ref_out_cpu, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_sf_cpu, ref_sf_cpu, rtol=1e-3, atol=1e-3)

    out_fp8 = _cast_fp32_to_fp8_cpu(out)
    ref_out_fp8 = _cast_fp32_to_fp8_cpu(ref_out)
    fp8_mismatches = torch.count_nonzero(out_fp8.float() != ref_out_fp8.float()).item()
    fp8_mismatch_rate = fp8_mismatches / out_fp8.numel()
    assert fp8_mismatch_rate <= 1e-3, f"FP8 bucket mismatch rate {fp8_mismatch_rate:.6e} ({fp8_mismatches}/{out_fp8.numel()}) exceeds 1e-3"


def _make_param_id(params: dict) -> str:
    return "-".join(
        f"{key}={params[key]}"
        for key in ("num_send_tokens", "num_topk", "num_experts", "num_ep_ranks", "hidden", "is_fused_cast_back", "round_sf")
    )


@pytest.mark.parametrize("params", generate_test_params(), ids=_make_param_id)
def test_per_channel_cast_fused_npu(params: dict) -> None:
    _check_case(params)


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v"])
    if exit_code == pytest.ExitCode.OK:
        print("All per_channel_cast_fused tests passed! Kernel Output Match!")
    sys.exit(exit_code)
