import math
import os
import sys
from dataclasses import replace
from typing import Optional

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
import pytest

try:
    from .utils import *
except ImportError:
    from utils import *

pytest.importorskip("torch_npu")
os.environ["TILELANG_PRINT_ON_COMPILATION"] = "0"
tilelang.cache.clear_cache()

pass_configs = {tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True}

_PRECOMPUTED_NUM_ELEMS_PER_BLOCK = 128 * 32
_MAIN_NUM_ELEMS_PER_BLOCK = 128 * 64


@T.macro
def load_sf(tensor, m_idx: int, k_idx: int, config: BaseCastConfig):
    if config.use_packed_ue8m0:
        return tensor[k_idx // 4, m_idx * 4 + k_idx % 4]
    if config.use_tma_aligned_col_major_sf:
        return tensor[k_idx, m_idx]
    return tensor[m_idx, k_idx]


@T.macro
def store_sf(tensor, sf, m_idx: int, k_idx: int, config: BaseCastConfig) -> None:
    if config.use_packed_ue8m0:
        tensor[k_idx // 4, m_idx * 4 + k_idx % 4] = sf
    elif config.use_tma_aligned_col_major_sf:
        tensor[k_idx, m_idx] = sf
    else:
        tensor[m_idx, k_idx] = sf


@T.macro
def transform_sf(sf, config: BaseCastConfig):
    if config.use_packed_ue8m0:
        return T.reinterpret("float32", T.Cast("int32", sf) << 23)
    return sf


def transform_sf_for_ref(
    sf: torch.Tensor, config: BaseCastConfig, data_shape: tuple[int, int], already_internal_layout: bool = False
) -> torch.Tensor:
    num_sf_m = ceil_div(data_shape[0], config.sf_block[0])
    num_sf_k = ceil_div(data_shape[1], config.sf_block[1])
    if config.use_packed_ue8m0:
        num_sf_k_packed = ceil_div(num_sf_k, 4)
        sf_uint8 = sf.detach().cpu().contiguous().view(torch.uint8)
        if already_internal_layout:
            sf_uint8 = sf_uint8[:num_sf_k_packed, : num_sf_m * 4]
            sf_uint8 = sf_uint8.reshape(num_sf_k_packed, num_sf_m, 4).permute(1, 0, 2)
        else:
            sf_uint8 = sf_uint8.reshape(num_sf_m, num_sf_k_packed, 4)
        sf_uint8 = sf_uint8.reshape(num_sf_m, num_sf_k_packed * 4)[:, :num_sf_k]
        return (sf_uint8.to(torch.int32) << 23).view(torch.float32).to(device=sf.device)
    if config.use_tma_aligned_col_major_sf and already_internal_layout:
        sf = sf.T
    return sf[:num_sf_m, :num_sf_k].to(torch.float32)


def generate_input_scaling_factors(shape: tuple[int, int], config: CastInputConfig, device: torch.device) -> torch.Tensor:
    num_sf_m = ceil_div(shape[0], config.sf_block[0])
    num_sf_k = ceil_div(shape[1], config.sf_block[1])
    if config.use_packed_ue8m0:
        assert config.use_tma_aligned_col_major_sf
        num_sf_k_packed = ceil_div(num_sf_k, 4)
        logical_u8 = torch.randint(125, 130, (num_sf_m, num_sf_k_packed * 4), dtype=torch.uint8, device="cpu")
        logical_u8[:, num_sf_k:] = 127
        physical_u8 = logical_u8.reshape(num_sf_m, num_sf_k_packed, 4).permute(1, 0, 2).contiguous().reshape(num_sf_k_packed, num_sf_m * 4)
        return physical_u8.view(torch.int32).to(device).T
    sf_exp = torch.randint(-2, 3, (num_sf_m, num_sf_k), dtype=torch.int32, device=device)
    sf = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=device), sf_exp.to(torch.float32))
    return sf.T.contiguous().T if config.use_tma_aligned_col_major_sf else sf


def _max_quant_value_for_config(out_config: CastOutputConfig) -> float:
    return 448.0 if out_config.clamp_min_value == 1e-4 else 6.0


def _get_best_vectorize_size(dtype: str) -> int:
    bytes_by_dtype = {"float32": 4, "bfloat16": 2, "float16": 2, "int8": 1, "uint8": 1}
    return 16 // bytes_by_dtype.get(dtype, 4)


def _get_logical_block_m(num_elems_per_block: int, block_k: int) -> int:
    block_m = max(1, num_elems_per_block // block_k)
    return block_m if block_m == 1 else block_m - block_m % 2


def _get_kernel_tile_shape(
    hidden: int, num_per_channels: int, fmt: str, use_packed_ue8m0: bool = False, num_elems_per_block: int = _MAIN_NUM_ELEMS_PER_BLOCK
) -> tuple[int, int, int]:
    num_threads = 128
    kernel_num_per_channels = num_per_channels
    if hidden == num_per_channels:
        kernel_num_per_channels = align_up(hidden, num_threads * (2 if fmt == "fp4" else 1))
        block_k = kernel_num_per_channels
    elif use_packed_ue8m0:
        block_k = 4 * num_per_channels
        kernel_num_per_channels = block_k
    else:
        block_k = num_per_channels
    block_m = _get_logical_block_m(num_elems_per_block, block_k)
    return block_m, block_k, kernel_num_per_channels


def _unpack_int32_bytes(packed: torch.Tensor) -> torch.Tensor:
    packed_i32 = packed.to(torch.int32)
    bytes_i32 = [torch.remainder(torch.div(packed_i32, 256**idx, rounding_mode="floor"), 256) for idx in range(4)]
    return torch.stack(bytes_i32, dim=-1).to(torch.uint8)


def _output_to_fp32(quant_tensor: torch.Tensor, fmt: str) -> torch.Tensor:
    _ = fmt
    return quant_tensor.to(torch.float32)


def _pad_internal_input_sf(
    x_sf: torch.Tensor | None, in_config: CastInputConfig, padded_shape: tuple[int, int], row_major: bool = False
) -> torch.Tensor | None:
    if x_sf is None:
        return None
    sf_m = math.ceil(padded_shape[0] / in_config.sf_block[0])
    sf_k = math.ceil(padded_shape[1] / in_config.sf_block[1])

    if in_config.use_packed_ue8m0:
        packed_sf_k = x_sf.shape[0]
        source_sf_m = x_sf.shape[1] // 4
        logical_u8 = x_sf.contiguous().reshape(packed_sf_k, source_sf_m, 4).permute(1, 0, 2).reshape(source_sf_m, packed_sf_k * 4)
        exponent = logical_u8.to(torch.float32) - 127.0
        logical_sf = torch.pow(torch.full_like(exponent, 2.0), exponent)
    elif in_config.use_tma_aligned_col_major_sf:
        logical_sf = x_sf.T.contiguous().to(torch.float32)
    else:
        logical_sf = x_sf.to(torch.float32)

    padded = torch.ones((sf_m, sf_k), dtype=torch.float32, device=x_sf.device)
    copy_m = min(logical_sf.shape[0], sf_m)
    copy_k = min(logical_sf.shape[1], sf_k)
    padded[:copy_m, :copy_k] = logical_sf[:copy_m, :copy_k]
    return padded.contiguous() if row_major else padded.T.contiguous()


def _pad_precomputed_sf_for_kernel(
    sf: torch.Tensor, out_config: CastOutputConfig, padded_shape: tuple[int, int], device: torch.device
) -> torch.Tensor:
    sf_m = math.ceil(padded_shape[0] / out_config.sf_block[0])
    sf_k = math.ceil(padded_shape[1] / out_config.sf_block[1])
    sf_device = sf.to(device=device)

    if out_config.use_packed_ue8m0:
        packed_u8 = _unpack_int32_bytes(sf_device.contiguous())
        packed_u8 = packed_u8.reshape(sf_device.shape[0], sf_device.shape[1] * 4)
        exponent = packed_u8.to(torch.float32) - 127.0
        logical_sf = torch.pow(torch.full_like(exponent, 2.0), exponent)
    else:
        logical_sf = sf_device.to(torch.float32)

    padded = torch.ones((sf_m, sf_k), dtype=torch.float32, device=device)
    copy_m = min(logical_sf.shape[0], sf_m)
    copy_k = min(logical_sf.shape[1], sf_k)
    padded[:copy_m, :copy_k] = logical_sf[:copy_m, :copy_k]
    return padded.T.contiguous()


@tilelang.jit(out_idx=[2, 3], pass_configs=pass_configs)
def get_per_token_cast_kernel(
    hidden: int,
    token_stride: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    sf_only: bool = False,
    cast_only: bool = False,
    input_sf_row_major: bool = False,
):
    _ = token_stride
    num_threads = 128
    num_elems_per_block = _PRECOMPUTED_NUM_ELEMS_PER_BLOCK if cast_only else _MAIN_NUM_ELEMS_PER_BLOCK
    num_per_channels = out_config.sf_block[1]
    input_sf_block = (1, 1) if in_config.sf_block is None else in_config.sf_block
    input_sf_block_m = input_sf_block[0]
    input_sf_block_k = input_sf_block[1]

    if hidden == num_per_channels and not cast_only:
        assert not in_config.with_sf and not sf_only
        block_k = align_up(hidden, num_threads * (2 if _max_quant_value_for_config(out_config) == 6.0 else 1))
        num_per_channels = block_k
    elif out_config.use_packed_ue8m0:
        block_k = 4 * num_per_channels
    else:
        block_k = num_per_channels
    assert block_k % num_per_channels == 0

    logical_block_m = _get_logical_block_m(num_elems_per_block, block_k)
    if in_config.with_sf and math.ceil(block_k / input_sf_block_k) == 1:
        logical_block_m = min(logical_block_m, 2 * input_sf_block_m)
    use_vid_split = logical_block_m % 2 == 0
    block_m = logical_block_m // 2 if use_vid_split else logical_block_m
    num_groups = block_k // num_per_channels
    use_direct_fp32_group = num_groups == 1 and in_config.dtype != "float32" and block_m * block_k >= 7168
    fp32_group_ub_shape = (1, 1) if use_direct_fp32_group else (block_m * num_groups, num_per_channels)
    if not out_config.use_packed_ue8m0 and not out_config.use_tma_aligned_col_major_sf:
        assert num_groups == 1
    num_vectorize = min(_get_best_vectorize_size(in_config.dtype), math.gcd(block_m * block_k // num_threads, 32))
    num_sf_rows_per_block = math.ceil(block_m / input_sf_block_m)
    num_sf_cols_per_block = math.ceil(block_k / input_sf_block_k)
    use_four_col_input_sf_fast_path = num_sf_cols_per_block == 4 and block_m % 2 == 0
    use_multi_col_input_sf_fast_path = num_sf_rows_per_block == 1 and (num_sf_cols_per_block >= 8 or use_four_col_input_sf_fast_path)
    use_single_input_sf_copy = in_config.with_sf and (
        input_sf_row_major or (num_sf_cols_per_block == 1 and not in_config.use_packed_ue8m0 and not in_config.use_tma_aligned_col_major_sf)
    )
    input_sf_broadcast_cols = 8 if use_four_col_input_sf_fast_path else num_sf_cols_per_block
    input_sf_broadcast_rows = block_m * num_sf_cols_per_block // input_sf_broadcast_cols
    x_input_ub_shape = (1, 1) if in_config.dtype == "float32" else (block_m, block_k)
    packed_ub_m = block_m if out_config.use_packed_ue8m0 else 1
    input_sf_ub_rows = block_m if cast_only else num_sf_rows_per_block if in_config.with_sf else 1
    input_sf_ub_cols = num_sf_cols_per_block if in_config.with_sf else 1
    use_input_sf_row_group = in_config.with_sf and num_sf_cols_per_block == 1 and num_sf_rows_per_block > 1
    input_sf_row_group_shape = (input_sf_block_m, block_k) if use_input_sf_row_group else (1, 1)
    input_sf_group_m = block_m * num_sf_cols_per_block if in_config.with_sf and use_multi_col_input_sf_fast_path else 1
    input_sf_group_k = input_sf_block_k if in_config.with_sf and use_multi_col_input_sf_fast_path else 1
    input_sf_group_rows_shape = (
        (input_sf_broadcast_rows, input_sf_broadcast_cols) if in_config.with_sf and use_multi_col_input_sf_fast_path else (1, 1)
    )
    input_sf_group_seed_cols = input_sf_broadcast_cols if in_config.with_sf and use_four_col_input_sf_fast_path else 1
    sf_col_ub_m = block_m if not out_config.round_sf else 1
    sf_round_ub_m = block_m * num_groups if out_config.round_sf else 1
    if cast_only:
        assert not in_config.with_sf and not sf_only
    elif in_config.with_sf:
        assert not cast_only and not sf_only
        assert block_k % num_vectorize == 0
        assert num_per_channels >= num_vectorize, f"num_per_channels ({num_per_channels}) must be >= num_vectorize ({num_vectorize})"
        assert block_m % input_sf_block_m == 0 or input_sf_block_m % block_m == 0
        assert block_k % input_sf_block_k == 0 or input_sf_block_k % block_k == 0

    input_is_float32 = in_config.dtype == "float32"
    max_quant_value = _max_quant_value_for_config(out_config)
    sf_rounding_bias = 0x1FFFFF if max_quant_value == 448.0 else 0x3FFFFF
    sf_exponent_offset = 8 if max_quant_value == 448.0 else 2

    num_tokens = T.symbolic("num_tokens")
    in_sf_stride = T.symbolic("in_sf_stride")
    out_sf_stride = T.symbolic("out_sf_stride")
    if cast_only:
        x_sf_shape = (T.ceildiv(hidden, out_config.sf_block[1]), T.ceildiv(num_tokens, out_config.sf_block[0]))
    elif in_config.with_sf:
        if input_sf_row_major:
            x_sf_shape = (T.ceildiv(num_tokens, input_sf_block_m), T.ceildiv(hidden, input_sf_block_k))
        else:
            x_sf_shape = get_sf_shape((num_tokens, hidden), in_config)
        if not input_sf_row_major and not in_config.use_tma_aligned_col_major_sf:
            x_sf_shape = (x_sf_shape[1], x_sf_shape[0])
    else:
        x_sf_shape = (1, 1)
    if cast_only:
        sf_shape = (1, 1)
        out_sf_dtype = "float32"
    elif out_config.use_packed_ue8m0:
        sf_shape = (T.ceildiv(T.ceildiv(hidden, out_config.sf_block[1]), 4), T.ceildiv(num_tokens, out_config.sf_block[0]))
        out_sf_dtype = "int32"
    elif out_config.use_tma_aligned_col_major_sf:
        sf_shape = get_sf_shape((num_tokens, hidden), out_config)
        out_sf_dtype = out_config.sf_dtype
    else:
        sf_shape = (T.ceildiv(hidden, out_config.sf_block[1]), T.ceildiv(num_tokens, out_config.sf_block[0]))
        out_sf_dtype = out_config.sf_dtype
    _ = in_sf_stride, out_sf_stride
    m_num = T.ceildiv(num_tokens, logical_block_m * 2)
    n_num = T.ceildiv(hidden, block_k)
    x_sf_dtype = in_config.sf_dtype if in_config.with_sf and not cast_only else "float32"

    @T.macro
    def load_data_tile(x, x_ub, x_fp32_ub, row_offset, col_offset, event_id):
        T.wait_flag("v", "mte2", event_id)
        if input_is_float32:
            T.copy(x[row_offset : row_offset + block_m, col_offset : col_offset + block_k], x_fp32_ub)
        else:
            T.copy(x[row_offset : row_offset + block_m, col_offset : col_offset + block_k], x_ub)
        T.set_flag("mte2", "v", event_id)

    @T.macro
    def store_data_tile(out, x_fp32_ub, row_offset, col_offset, event_id):
        if not sf_only:
            T.wait_flag("v", "mte3", event_id)
            T.copy(x_fp32_ub, out[row_offset : row_offset + block_m, col_offset : col_offset + block_k])
            T.set_flag("mte3", "v", event_id)

    @T.macro
    def process_data_tile(
        x_ub,
        x_fp32_ub,
        x_fp32_group_ub,
        abs_ub,
        amax_ub,
        sf_inv_ub,
        sf_bits_ub,
        sf_inv_bits_ub,
        sf_packed_i32_ub,
        sf_pack_exp1_ub,
        sf_pack_offset_i32_ub,
        sf_pack_offset_u32_ub,
        sf_col_ub,
        x_sf_ub,
        x_sf_col_ub,
        x_sf_row_group_ub,
        x_fp32_input_group_ub,
        x_sf_group_rows_ub,
        x_sf_group_row_ub,
        x_sf_group_tile_ub,
        x_sf_group_seed_ub,
        x_sf,
        out_sf,
        row_offset,
        col_offset,
        pid_hidden,
    ):
        sf_row_offset = row_offset // input_sf_block_m
        sf_col_offset = pid_hidden * block_k // input_sf_block_k
        if use_single_input_sf_copy:
            T.wait_flag("v", "mte2", 4)
            if input_sf_row_major:
                T.copy(x_sf[sf_row_offset, sf_col_offset : sf_col_offset + num_sf_cols_per_block], x_sf_ub[0, :])
            else:
                T.copy(x_sf[sf_col_offset, sf_row_offset : sf_row_offset + num_sf_rows_per_block], x_sf_ub[:, 0])
            T.set_flag("mte2", "v", 4)
        if not input_is_float32:
            T.tile.cast(x_fp32_ub, x_ub, mode="CAST_NONE", count=block_m * block_k)

        if in_config.with_sf:
            if use_single_input_sf_copy:
                T.wait_flag("mte2", "v", 4)
            elif in_config.use_packed_ue8m0 or in_config.use_tma_aligned_col_major_sf:
                for i in T.serial(num_sf_rows_per_block):
                    for j in T.serial(num_sf_cols_per_block):
                        m_idx = sf_row_offset + i
                        k_idx = sf_col_offset + j
                        x_sf_ub[i, j] = transform_sf(load_sf(x_sf, m_idx, k_idx, in_config), in_config)
            else:
                for j in T.serial(num_sf_cols_per_block):
                    T.copy(x_sf[sf_col_offset + j, sf_row_offset : sf_row_offset + num_sf_rows_per_block], x_sf_col_ub)
                    T.set_flag("mte2", "v", 4)
                    T.wait_flag("mte2", "v", 4)
                    for i in T.serial(num_sf_rows_per_block):
                        x_sf_ub[i, j] = x_sf_col_ub[i]
                    T.set_flag("v", "mte2", 4)
                    T.wait_flag("v", "mte2", 4)

            if num_sf_cols_per_block == 1:
                if num_sf_rows_per_block == 1:
                    T.tile.mul(x_fp32_ub, x_fp32_ub, x_sf_ub[0, 0])
                else:
                    for sf_row in T.unroll(num_sf_rows_per_block):
                        row_begin = sf_row * input_sf_block_m
                        T.copy(x_fp32_ub[row_begin : row_begin + input_sf_block_m, :], x_sf_row_group_ub)
                        T.tile.mul(x_sf_row_group_ub, x_sf_row_group_ub, x_sf_ub[sf_row, 0])
                        T.copy(x_sf_row_group_ub, x_fp32_ub[row_begin : row_begin + input_sf_block_m, :])
            else:
                if use_multi_col_input_sf_fast_path:
                    if use_four_col_input_sf_fast_path:
                        for j in T.unroll(num_sf_cols_per_block):
                            x_sf_group_seed_ub[0, j] = x_sf_ub[0, j]
                            x_sf_group_seed_ub[0, j + num_sf_cols_per_block] = x_sf_ub[0, j]
                        T.tile.broadcast(x_sf_group_rows_ub, x_sf_group_seed_ub[0, :], axis=0)
                    else:
                        T.tile.broadcast(x_sf_group_rows_ub, x_sf_ub[0, :], axis=0)
                    T.tile.broadcast(x_sf_group_tile_ub, x_sf_group_row_ub, axis=1)
                    T.tile.mul(x_fp32_input_group_ub, x_fp32_input_group_ub, x_sf_group_tile_ub)
                else:
                    for i in T.serial(block_m):
                        for j in T.serial(block_k):
                            x_fp32_ub[i, j] = x_fp32_ub[i, j] * x_sf_ub[i // input_sf_block_m, j // input_sf_block_k]

            if use_single_input_sf_copy:
                T.set_flag("v", "mte2", 4)

            if use_direct_fp32_group:
                T.tile.abs(abs_ub, x_fp32_ub)
            else:
                T.tile.abs(abs_ub, x_fp32_group_ub)
            T.reduce_max(abs_ub, amax_ub, dim=-1, clear=True, real_shape=[block_m * num_groups, num_per_channels])

            if out_config.round_sf:
                T.tile.max(amax_ub, amax_ub, out_config.clamp_min_value)
            else:
                for i in T.serial(block_m):
                    for g in T.serial(num_groups):
                        sf_idx = i * num_groups + g
                        clamped = T.max(amax_ub[sf_idx], out_config.clamp_min_value)
                        amax_ub[sf_idx] = clamped / T.float32(max_quant_value)
                        sf_inv_ub[sf_idx] = T.float32(max_quant_value) / clamped
            T.pipe_barrier("v")
            if out_config.round_sf:
                T.tile.add(sf_bits_ub, sf_bits_ub, sf_rounding_bias)
                T.tile.bitwise_rshift(sf_bits_ub, sf_bits_ub, 23)
                T.tile.add(sf_bits_ub, sf_bits_ub, -sf_exponent_offset)
                T.tile.mul(sf_inv_bits_ub, sf_bits_ub, -1)
                T.tile.add(sf_inv_bits_ub, sf_inv_bits_ub, 254)
                T.tile.bitwise_lshift(sf_inv_bits_ub, sf_inv_bits_ub, 23)
                if not out_config.use_packed_ue8m0 and not out_config.use_tma_aligned_col_major_sf:
                    T.tile.bitwise_lshift(sf_bits_ub, sf_bits_ub, 23)
                T.pipe_barrier("v")

            if out_config.use_packed_ue8m0:
                if num_groups == 4:
                    T.tile.arith_progression(sf_pack_offset_i32_ub, 0, num_groups * 4, block_m)
                    T.tile.gather(sf_packed_i32_ub, sf_bits_ub, sf_pack_offset_u32_ub, 0)
                    T.tile.add(sf_pack_offset_i32_ub, sf_pack_offset_i32_ub, 4)
                    T.tile.gather(sf_pack_exp1_ub, sf_bits_ub, sf_pack_offset_u32_ub, 0)
                    T.tile.bitwise_lshift(sf_pack_exp1_ub, sf_pack_exp1_ub, 8)
                    T.tile.add(sf_packed_i32_ub, sf_packed_i32_ub, sf_pack_exp1_ub)
                    T.tile.add(sf_pack_offset_i32_ub, sf_pack_offset_i32_ub, 4)
                    T.tile.gather(sf_pack_exp1_ub, sf_bits_ub, sf_pack_offset_u32_ub, 0)
                    T.tile.bitwise_lshift(sf_pack_exp1_ub, sf_pack_exp1_ub, 16)
                    T.tile.add(sf_packed_i32_ub, sf_packed_i32_ub, sf_pack_exp1_ub)
                    T.tile.add(sf_pack_offset_i32_ub, sf_pack_offset_i32_ub, 4)
                    T.tile.gather(sf_pack_exp1_ub, sf_bits_ub, sf_pack_offset_u32_ub, 0)
                    T.tile.bitwise_lshift(sf_pack_exp1_ub, sf_pack_exp1_ub, 24)
                    T.tile.add(sf_packed_i32_ub, sf_packed_i32_ub, sf_pack_exp1_ub)
                else:
                    for i in T.serial(block_m):
                        sf_packed_i32_ub[i] = sf_bits_ub[i * num_groups]
                T.set_flag("v", "mte3", 4)
                T.wait_flag("v", "mte3", 4)
                T.copy(sf_packed_i32_ub, out_sf[pid_hidden, row_offset : row_offset + block_m])
            elif out_config.use_tma_aligned_col_major_sf:
                for i in T.serial(block_m):
                    for g in T.serial(num_groups):
                        sf_idx = i * num_groups + g
                        store_sf(
                            out_sf,
                            T.float32(1.0) / sf_inv_ub[sf_idx] if out_config.round_sf else amax_ub[sf_idx],
                            row_offset + i,
                            pid_hidden * num_groups + g,
                            out_config,
                        )

            if not out_config.use_packed_ue8m0 and not out_config.use_tma_aligned_col_major_sf:
                if out_config.round_sf:
                    T.set_flag("v", "mte3", 4)
                    T.wait_flag("v", "mte3", 4)
                    T.copy(amax_ub, out_sf[pid_hidden, row_offset : row_offset + block_m])
                else:
                    for g in T.serial(num_groups):
                        for i in T.Parallel(block_m):
                            sf_idx = i * num_groups + g
                            sf_col_ub[i] = amax_ub[sf_idx]
                        T.set_flag("v", "mte3", 4)
                        T.wait_flag("v", "mte3", 4)
                        T.copy(sf_col_ub, out_sf[pid_hidden * num_groups + g, row_offset : row_offset + block_m])
                        if g + 1 < num_groups:
                            T.set_flag("mte3", "v", 4)
                            T.wait_flag("mte3", "v", 4)

            if not sf_only:
                T.tile.broadcast(abs_ub, sf_inv_ub, axis=1)
                if use_direct_fp32_group:
                    T.tile.mul(x_fp32_ub, x_fp32_ub, abs_ub)
                else:
                    T.tile.mul(x_fp32_group_ub, x_fp32_group_ub, abs_ub)
        else:
            if cast_only:
                for g in T.serial(num_groups):
                    T.wait_flag("v", "mte2", 4)
                    T.copy(x_sf[pid_hidden * num_groups + g, row_offset : row_offset + block_m], x_sf_col_ub)
                    T.set_flag("mte2", "v", 4)
                    T.wait_flag("mte2", "v", 4)
                    for i in T.serial(block_m):
                        sf_idx = i * num_groups + g
                        amax_ub[sf_idx] = x_sf_col_ub[i]
                        sf_inv_ub[sf_idx] = T.float32(1.0) / amax_ub[sf_idx]
                    T.set_flag("v", "mte2", 4)
            else:
                if use_direct_fp32_group:
                    T.tile.abs(abs_ub, x_fp32_ub)
                else:
                    T.tile.abs(abs_ub, x_fp32_group_ub)
                T.reduce_max(abs_ub, amax_ub, dim=-1, clear=True, real_shape=[block_m * num_groups, num_per_channels])

                if out_config.round_sf:
                    T.tile.max(amax_ub, amax_ub, out_config.clamp_min_value)
                else:
                    for i in T.serial(block_m):
                        for g in T.serial(num_groups):
                            sf_idx = i * num_groups + g
                            clamped = T.max(amax_ub[sf_idx], out_config.clamp_min_value)
                            amax_ub[sf_idx] = clamped / T.float32(max_quant_value)
                            sf_inv_ub[sf_idx] = T.float32(max_quant_value) / clamped
                T.pipe_barrier("v")
                if out_config.round_sf:
                    T.tile.add(sf_bits_ub, sf_bits_ub, sf_rounding_bias)
                    T.tile.bitwise_rshift(sf_bits_ub, sf_bits_ub, 23)
                    T.tile.add(sf_bits_ub, sf_bits_ub, -sf_exponent_offset)
                    T.tile.mul(sf_inv_bits_ub, sf_bits_ub, -1)
                    T.tile.add(sf_inv_bits_ub, sf_inv_bits_ub, 254)
                    T.tile.bitwise_lshift(sf_inv_bits_ub, sf_inv_bits_ub, 23)
                    if not out_config.use_packed_ue8m0 and not out_config.use_tma_aligned_col_major_sf:
                        T.tile.bitwise_lshift(sf_bits_ub, sf_bits_ub, 23)
                    T.pipe_barrier("v")

                if out_config.use_packed_ue8m0:
                    if num_groups == 4:
                        T.tile.arith_progression(sf_pack_offset_i32_ub, 0, num_groups * 4, block_m)
                        T.tile.gather(sf_packed_i32_ub, sf_bits_ub, sf_pack_offset_u32_ub, 0)
                        T.tile.add(sf_pack_offset_i32_ub, sf_pack_offset_i32_ub, 4)
                        T.tile.gather(sf_pack_exp1_ub, sf_bits_ub, sf_pack_offset_u32_ub, 0)
                        T.tile.bitwise_lshift(sf_pack_exp1_ub, sf_pack_exp1_ub, 8)
                        T.tile.add(sf_packed_i32_ub, sf_packed_i32_ub, sf_pack_exp1_ub)
                        T.tile.add(sf_pack_offset_i32_ub, sf_pack_offset_i32_ub, 4)
                        T.tile.gather(sf_pack_exp1_ub, sf_bits_ub, sf_pack_offset_u32_ub, 0)
                        T.tile.bitwise_lshift(sf_pack_exp1_ub, sf_pack_exp1_ub, 16)
                        T.tile.add(sf_packed_i32_ub, sf_packed_i32_ub, sf_pack_exp1_ub)
                        T.tile.add(sf_pack_offset_i32_ub, sf_pack_offset_i32_ub, 4)
                        T.tile.gather(sf_pack_exp1_ub, sf_bits_ub, sf_pack_offset_u32_ub, 0)
                        T.tile.bitwise_lshift(sf_pack_exp1_ub, sf_pack_exp1_ub, 24)
                        T.tile.add(sf_packed_i32_ub, sf_packed_i32_ub, sf_pack_exp1_ub)
                    else:
                        for i in T.serial(block_m):
                            sf_packed_i32_ub[i] = sf_bits_ub[i * num_groups]
                    T.set_flag("v", "mte3", 4)
                    T.wait_flag("v", "mte3", 4)
                    T.copy(sf_packed_i32_ub, out_sf[pid_hidden, row_offset : row_offset + block_m])
                elif out_config.use_tma_aligned_col_major_sf:
                    for i in T.serial(block_m):
                        for g in T.serial(num_groups):
                            sf_idx = i * num_groups + g
                            store_sf(
                                out_sf,
                                T.float32(1.0) / sf_inv_ub[sf_idx] if out_config.round_sf else amax_ub[sf_idx],
                                row_offset + i,
                                pid_hidden * num_groups + g,
                                out_config,
                            )

                if not out_config.use_packed_ue8m0 and not out_config.use_tma_aligned_col_major_sf:
                    if out_config.round_sf:
                        T.set_flag("v", "mte3", 4)
                        T.wait_flag("v", "mte3", 4)
                        T.copy(amax_ub, out_sf[pid_hidden, row_offset : row_offset + block_m])
                    else:
                        for g in T.serial(num_groups):
                            for i in T.Parallel(block_m):
                                sf_idx = i * num_groups + g
                                sf_col_ub[i] = amax_ub[sf_idx]
                            T.set_flag("v", "mte3", 4)
                            T.wait_flag("v", "mte3", 4)
                            T.copy(sf_col_ub, out_sf[pid_hidden * num_groups + g, row_offset : row_offset + block_m])
                            if g + 1 < num_groups:
                                T.set_flag("mte3", "v", 4)
                                T.wait_flag("mte3", "v", 4)

            if not sf_only:
                T.tile.broadcast(abs_ub, sf_inv_ub, axis=1)
                if use_direct_fp32_group:
                    T.tile.mul(x_fp32_ub, x_fp32_ub, abs_ub)
                else:
                    T.tile.mul(x_fp32_group_ub, x_fp32_group_ub, abs_ub)

    @T.prim_func
    def per_token_cast_kernel(
        x: T.Tensor((num_tokens, hidden), in_config.dtype),
        x_sf: T.Tensor(x_sf_shape, x_sf_dtype),
        out: T.Tensor((num_tokens, hidden), "float32"),
        out_sf: T.Tensor(sf_shape, out_sf_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            pid_token = cid // n_num
            pid_hidden = cid % n_num
            vid_row_offset = vid * block_m if use_vid_split else 0
            row_offset_0 = pid_token * logical_block_m * 2 + vid_row_offset
            row_offset_1 = row_offset_0 + logical_block_m
            col_offset = pid_hidden * block_k

            x_ub_0 = T.alloc_ub(x_input_ub_shape, in_config.dtype)
            x_ub_1 = T.alloc_ub(x_input_ub_shape, in_config.dtype)
            x_fp32_ub_0 = T.alloc_ub((block_m, block_k), "float32")
            x_fp32_ub_1 = T.alloc_ub((block_m, block_k), "float32")
            x_fp32_group_ub_0 = T.alloc_ub(fp32_group_ub_shape, "float32")
            x_fp32_group_ub_1 = T.alloc_ub(fp32_group_ub_shape, "float32")
            abs_ub = T.alloc_ub((block_m * num_groups, num_per_channels), "float32")
            amax_ub = T.alloc_ub((block_m * num_groups,), "float32")
            sf_inv_ub = T.alloc_ub((block_m * num_groups,), "float32")
            sf_bits_ub = T.alloc_ub((block_m * num_groups,), "int32")
            sf_inv_bits_ub = T.alloc_ub((sf_round_ub_m,), "int32")
            sf_packed_i32_ub = T.alloc_ub((packed_ub_m,), "int32")
            sf_pack_exp1_ub = T.alloc_ub((packed_ub_m,), "int32")
            sf_pack_offset_i32_ub = T.alloc_ub((packed_ub_m,), "int32")
            sf_pack_offset_u32_ub = T.alloc_ub((packed_ub_m,), "uint32")
            sf_col_ub = T.alloc_ub((sf_col_ub_m,), out_config.sf_dtype)
            x_sf_ub = T.alloc_ub((input_sf_ub_rows, input_sf_ub_cols), "float32")
            x_sf_col_ub = T.alloc_ub((input_sf_ub_rows,), "float32")
            x_sf_row_group_ub = T.alloc_ub(input_sf_row_group_shape, "float32")
            x_fp32_input_group_ub_0 = T.alloc_ub((input_sf_group_m, input_sf_group_k), "float32")
            x_fp32_input_group_ub_1 = T.alloc_ub((input_sf_group_m, input_sf_group_k), "float32")
            x_sf_group_rows_ub = T.alloc_ub(input_sf_group_rows_shape, "float32")
            x_sf_group_row_ub = T.alloc_ub((input_sf_group_m,), "float32")
            x_sf_group_tile_ub = T.alloc_ub((input_sf_group_m, input_sf_group_k), "float32")
            x_sf_group_seed_ub = T.alloc_ub((1, input_sf_group_seed_cols), "float32")

            with T.Scope("V"):
                if not use_direct_fp32_group:
                    T.reinterpretcast(x_fp32_group_ub_0, x_fp32_ub_0, "float")
                    T.reinterpretcast(x_fp32_group_ub_1, x_fp32_ub_1, "float")
                T.reinterpretcast(sf_pack_offset_u32_ub, sf_pack_offset_i32_ub, "uint32_t")
                if out_config.round_sf:
                    T.reinterpretcast(sf_bits_ub, amax_ub, "int32_t")
                    T.reinterpretcast(sf_inv_bits_ub, sf_inv_ub, "int32_t")
                if in_config.with_sf and use_multi_col_input_sf_fast_path:
                    T.reinterpretcast(x_fp32_input_group_ub_0, x_fp32_ub_0, "float")
                    T.reinterpretcast(x_fp32_input_group_ub_1, x_fp32_ub_1, "float")
                    T.reinterpretcast(x_sf_group_row_ub, x_sf_group_rows_ub, "float")
                if use_single_input_sf_copy or cast_only:
                    T.set_flag("v", "mte2", 4)
                for slot in T.unroll(2):
                    T.set_flag("v", "mte2", slot)
                    if not sf_only:
                        T.set_flag("mte3", "v", slot)
                if not cast_only and (out_config.use_packed_ue8m0 or not out_config.use_tma_aligned_col_major_sf):
                    T.set_flag("mte3", "v", 4)

                load_data_tile(x, x_ub_0, x_fp32_ub_0, row_offset_0, col_offset, 0)
                load_data_tile(x, x_ub_1, x_fp32_ub_1, row_offset_1, col_offset, 1)

                T.wait_flag("mte2", "v", 0)
                if not sf_only:
                    T.wait_flag("mte3", "v", 0)
                if not cast_only and (out_config.use_packed_ue8m0 or not out_config.use_tma_aligned_col_major_sf):
                    T.wait_flag("mte3", "v", 4)
                process_data_tile(
                    x_ub_0,
                    x_fp32_ub_0,
                    x_fp32_group_ub_0,
                    abs_ub,
                    amax_ub,
                    sf_inv_ub,
                    sf_bits_ub,
                    sf_inv_bits_ub,
                    sf_packed_i32_ub,
                    sf_pack_exp1_ub,
                    sf_pack_offset_i32_ub,
                    sf_pack_offset_u32_ub,
                    sf_col_ub,
                    x_sf_ub,
                    x_sf_col_ub,
                    x_sf_row_group_ub,
                    x_fp32_input_group_ub_0,
                    x_sf_group_rows_ub,
                    x_sf_group_row_ub,
                    x_sf_group_tile_ub,
                    x_sf_group_seed_ub,
                    x_sf,
                    out_sf,
                    row_offset_0,
                    col_offset,
                    pid_hidden,
                )
                T.set_flag("v", "mte2", 0)
                if not cast_only and (out_config.use_packed_ue8m0 or not out_config.use_tma_aligned_col_major_sf):
                    T.set_flag("mte3", "v", 4)
                if not sf_only:
                    T.set_flag("v", "mte3", 0)
                store_data_tile(out, x_fp32_ub_0, row_offset_0, col_offset, 0)

                T.wait_flag("mte2", "v", 1)
                if not sf_only:
                    T.wait_flag("mte3", "v", 1)
                if not cast_only and (out_config.use_packed_ue8m0 or not out_config.use_tma_aligned_col_major_sf):
                    T.wait_flag("mte3", "v", 4)
                process_data_tile(
                    x_ub_1,
                    x_fp32_ub_1,
                    x_fp32_group_ub_1,
                    abs_ub,
                    amax_ub,
                    sf_inv_ub,
                    sf_bits_ub,
                    sf_inv_bits_ub,
                    sf_packed_i32_ub,
                    sf_pack_exp1_ub,
                    sf_pack_offset_i32_ub,
                    sf_pack_offset_u32_ub,
                    sf_col_ub,
                    x_sf_ub,
                    x_sf_col_ub,
                    x_sf_row_group_ub,
                    x_fp32_input_group_ub_1,
                    x_sf_group_rows_ub,
                    x_sf_group_row_ub,
                    x_sf_group_tile_ub,
                    x_sf_group_seed_ub,
                    x_sf,
                    out_sf,
                    row_offset_1,
                    col_offset,
                    pid_hidden,
                )
                T.set_flag("v", "mte2", 1)
                if not cast_only and (out_config.use_packed_ue8m0 or not out_config.use_tma_aligned_col_major_sf):
                    T.set_flag("mte3", "v", 4)
                if not sf_only:
                    T.set_flag("v", "mte3", 1)
                store_data_tile(out, x_fp32_ub_1, row_offset_1, col_offset, 1)

                T.wait_flag("v", "mte2", 0)
                T.wait_flag("v", "mte2", 1)
                if not sf_only:
                    T.wait_flag("mte3", "v", 0)
                    T.wait_flag("mte3", "v", 1)
                if not cast_only and (out_config.use_packed_ue8m0 or not out_config.use_tma_aligned_col_major_sf):
                    T.wait_flag("mte3", "v", 4)
                if use_single_input_sf_copy or cast_only:
                    T.wait_flag("v", "mte2", 4)

    return per_token_cast_kernel


def _per_token_cast_impl(
    x_device: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    out_config: CastOutputConfig,
    x_sf: Optional[torch.Tensor] = None,
    in_config: Optional[CastInputConfig] = None,
    sf: Optional[torch.Tensor] = None,
    sf_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    if in_config is None:
        in_config = CastInputConfig(torch_dtype=x_device.dtype, sf_block=(1, 1), with_sf=False)
    if in_config.sf_block is None:
        in_config = replace(in_config, sf_block=(1, 1))
    in_config = replace(in_config, with_sf=(x_sf is not None), torch_dtype=x_device.dtype)

    assert not (x_sf is not None and (sf is not None or sf_only))
    assert x_device.device.type == "npu"
    assert x_device.dim() == 2 and x_device.is_contiguous()

    num_tokens, hidden = x_device.shape
    orig_num_tokens = num_tokens
    orig_hidden = hidden
    if sf is not None and hidden == num_per_channels and not out_config.use_packed_ue8m0:
        block_k = num_per_channels
        kernel_num_per_channels = num_per_channels
        block_m = _get_logical_block_m(_PRECOMPUTED_NUM_ELEMS_PER_BLOCK, block_k)
    else:
        block_m, block_k, kernel_num_per_channels = _get_kernel_tile_shape(
            hidden,
            num_per_channels,
            fmt,
            out_config.use_packed_ue8m0,
            _PRECOMPUTED_NUM_ELEMS_PER_BLOCK if sf is not None else _MAIN_NUM_ELEMS_PER_BLOCK,
        )
    kernel_out_config = out_config
    if hidden == num_per_channels and x_sf is None and sf is None and not sf_only:
        kernel_out_config = replace(kernel_out_config, sf_block=(1, kernel_num_per_channels))

    token_pad_multiple = block_m * 2
    pad_tokens = (token_pad_multiple - num_tokens % token_pad_multiple) % token_pad_multiple
    pad_hidden = (kernel_num_per_channels - hidden % kernel_num_per_channels) % kernel_num_per_channels
    if pad_tokens or pad_hidden:
        padded = torch.empty((num_tokens + pad_tokens, hidden + pad_hidden), dtype=x_device.dtype, device=x_device.device)
        padded[:num_tokens, :hidden] = x_device
        if pad_hidden:
            padded[:num_tokens, hidden:] = 0
        if pad_tokens:
            padded[num_tokens:, :] = 0
        x_device = padded
        num_tokens, hidden = x_device.shape

    vector_block_m = block_m // 2 if block_m % 2 == 0 else block_m
    input_sf_block_m, input_sf_block_k = in_config.sf_block
    input_sf_row_major = x_sf is not None and (
        math.ceil(vector_block_m / input_sf_block_m) == 1 and math.ceil(block_k / input_sf_block_k) > 1
    )
    x_sf = _pad_internal_input_sf(x_sf, in_config, (num_tokens, hidden), row_major=input_sf_row_major)
    in_config = replace(in_config, torch_dtype=x_device.dtype, use_tma_aligned_col_major_sf=False, use_packed_ue8m0=False)

    cast_only = sf is not None
    if cast_only:
        x_sf = _pad_precomputed_sf_for_kernel(sf, kernel_out_config, (num_tokens, hidden), x_device.device)
    kernel = get_per_token_cast_kernel(
        hidden=hidden,
        token_stride=x_device.stride(0),
        in_config=in_config,
        out_config=kernel_out_config,
        sf_only=sf_only,
        cast_only=cast_only,
        input_sf_row_major=input_sf_row_major,
    )
    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    out_fp32 = torch.empty((num_tokens, hidden), dtype=torch.float32, device=x_device.device)
    if cast_only:
        sf_shape = (1, 1)
    elif kernel_out_config.use_packed_ue8m0:
        sf_shape = (math.ceil(math.ceil(hidden / kernel_out_config.sf_block[1]) / 4), math.ceil(num_tokens / kernel_out_config.sf_block[0]))
    elif kernel_out_config.use_tma_aligned_col_major_sf:
        sf_shape = get_sf_shape((num_tokens, hidden), kernel_out_config)
    else:
        sf_shape = (math.ceil(hidden / kernel_out_config.sf_block[1]), math.ceil(num_tokens / kernel_out_config.sf_block[0]))
    if cast_only:
        out_sf_kernel = torch.empty(sf_shape, dtype=torch.float32, device=x_device.device)
    elif kernel_out_config.use_packed_ue8m0:
        out_sf_kernel = torch.zeros(sf_shape, dtype=torch.int32, device=x_device.device)
    else:
        out_sf_kernel = torch.empty(sf_shape, dtype=kernel_out_config.sf_torch_dtype, device=x_device.device)

    if x_sf is None:
        x_sf = torch.empty((1, 1), dtype=torch.float32, device=x_device.device)
    if num_tokens > 0:
        out_fp32, out_sf_kernel = kernel(x_device, x_sf, out_fp32, out_sf_kernel)

    quant_tensor = out_fp32[:orig_num_tokens, :orig_hidden]

    if cast_only:
        return _output_to_fp32(quant_tensor, fmt)

    if kernel_out_config.use_packed_ue8m0:
        out_sf = out_sf_kernel.T.contiguous()[
            : math.ceil(orig_num_tokens / kernel_out_config.sf_block[0]),
            : math.ceil(math.ceil(orig_hidden / kernel_out_config.sf_block[1]) / 4),
        ]
    elif kernel_out_config.use_tma_aligned_col_major_sf:
        out_sf = cast_epilogue(out_sf_kernel, orig_num_tokens, orig_hidden, kernel_out_config)
    else:
        out_sf = out_sf_kernel[
            : math.ceil(orig_hidden / kernel_out_config.sf_block[1]), : math.ceil(orig_num_tokens / kernel_out_config.sf_block[0])
        ].T.contiguous()
    if sf_only:
        return out_sf
    return _output_to_fp32(quant_tensor, fmt), out_sf


def per_token_cast(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    fmt: str,
    num_per_channels: int,
    x_block_size: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert fmt in ("fp8", "fp4")
    if isinstance(x, tuple):
        assert x_block_size is not None
        output_device = x[0].device
    else:
        assert x_block_size is None
        output_device = x.device

    x_data, x_sf, in_config = get_cast_input_and_config(x, x_block_size)
    assert x_data.dim() == 2
    assert x_data.dtype in (torch.bfloat16, torch.float32)
    assert x_data.device.type == "npu"
    x_data = x_data.contiguous()

    _, hidden = x_data.shape
    assert num_per_channels in (16, 32, 64, 128) or (num_per_channels == hidden and hidden % 64 == 0)
    out_config = get_cast_output_config(fmt, (1, num_per_channels), use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)
    out, out_sf = _per_token_cast_impl(x_data, fmt, num_per_channels, out_config, x_sf=x_sf, in_config=in_config)
    assert out.device == output_device and out_sf.device == output_device
    return out, out_sf


def per_token_cast_with_sf_only(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    fmt: str,
    num_per_channels: int,
    x_block_size: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> torch.Tensor:
    assert not isinstance(x, tuple)
    assert x_block_size is None
    x_data, x_sf, in_config = get_cast_input_and_config(x, x_block_size)
    assert x_sf is None
    x_data = x_data.contiguous()
    out_config = get_cast_output_config(fmt, (1, num_per_channels), use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)
    out_sf = _per_token_cast_impl(x_data, fmt, num_per_channels, out_config, in_config=in_config, sf_only=True)
    return out_sf


def per_token_cast_with_precomputed_sf(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    fmt: str,
    num_per_channels: int,
    sf: torch.Tensor,
    x_block_size: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> torch.Tensor:
    assert not isinstance(x, tuple)
    assert x_block_size is None
    x_data, x_sf, in_config = get_cast_input_and_config(x, x_block_size)
    assert x_sf is None
    x_data = x_data.contiguous()
    out_config = get_cast_output_config(fmt, (1, num_per_channels), use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)
    out = _per_token_cast_impl(x_data, fmt, num_per_channels, out_config, in_config=in_config, sf=sf)
    return out


def expand_input_with_sf_ref(
    x: tuple[torch.Tensor, torch.Tensor], block_size: tuple[int, int], use_tma_aligned_col_major_sf: bool, use_packed_ue8m0: bool
) -> torch.Tensor:
    x_data, x_sf = x
    config = CastInputConfig(
        torch_dtype=x_data.dtype,
        sf_block=block_size,
        with_sf=True,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    x_sf = transform_sf_for_ref(x_sf, config, tuple(x_data.shape), already_internal_layout=False)
    x_sf = x_sf.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=1)
    x_sf = x_sf[: x_data.shape[0], : x_data.shape[1]]
    return x_data.to(torch.float32) * x_sf.to(device=x_data.device, dtype=torch.float32)


def cast_back_quantized(x: tuple[torch.Tensor, torch.Tensor], block_size: tuple[int, int], use_packed_ue8m0: bool) -> torch.Tensor:
    input_tensor, input_sf = x
    config = CastInputConfig(
        torch_dtype=input_tensor.dtype,
        sf_block=block_size,
        with_sf=True,
        use_tma_aligned_col_major_sf=use_packed_ue8m0,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    input_sf = transform_sf_for_ref(input_sf, config, tuple(input_tensor.shape), already_internal_layout=False)
    input_sf = input_sf.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=1)
    input_sf = input_sf[: input_tensor.shape[0], : input_tensor.shape[1]]
    return input_tensor * input_sf


def clear_unused_sf(sf: torch.Tensor, hidden: int, num_per_channels: int) -> torch.Tensor:
    num_channel_blocks = ceil_div(hidden, num_per_channels)
    aligned_num_channel_blocks = align_up(num_channel_blocks, 4)
    sf_flattened = sf.contiguous().flatten().view(torch.uint8).view(-1, aligned_num_channel_blocks)
    sf_flattened[:, num_channel_blocks:] = 0
    return sf_flattened


def check_bias(x: torch.Tensor, ref_x: torch.Tensor) -> None:
    count = x.numel()
    if count == 0:
        return
    less_count = (x < ref_x).sum()
    equal_count = (x == ref_x).sum()
    less_ratio = (less_count + equal_count / 2) / count
    allowed_diff_ratio = 10 / math.sqrt(count)
    assert abs(less_ratio - 0.5) < allowed_diff_ratio, f"Less than ratio not close to 0.5 (size = {count}): {less_ratio=:.4f}"


def format_sf(ds_int_rounded: torch.Tensor, use_tma_aligned_col_major_sf: bool, use_packed_ue8m0: bool) -> torch.Tensor:
    if use_tma_aligned_col_major_sf:
        pad_h = align_up(ds_int_rounded.shape[0], 4) - ds_int_rounded.shape[0]
        pad_w = align_up(ds_int_rounded.shape[1], 4 if use_packed_ue8m0 else 1) - ds_int_rounded.shape[1]
        ds_int_rounded = F.pad(ds_int_rounded, (0, pad_w, 0, pad_h))
        if use_packed_ue8m0:
            dq_sf = (ds_int_rounded >> 23).to(torch.int8).view(torch.int32)
        else:
            dq_sf = ds_int_rounded.view(torch.float32)
        return dq_sf.T.contiguous().T[: ds_int_rounded.shape[0] - pad_h, :]

    return ds_int_rounded.view(torch.float32)


def cast_ref(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    fmt: str,
    block_size: tuple[int, int],
    sf: torch.Tensor | None = None,
    x_block_size: tuple[int, int] | None = None,
    round_sf: bool = False,
    use_tma_aligned_col_major_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if isinstance(x, tuple):
        assert x_block_size is not None
        x = expand_input_with_sf_ref(x, x_block_size, use_tma_aligned_col_major_sf, use_packed_ue8m0)
    else:
        assert x_block_size is None
    assert x.dtype in (torch.bfloat16, torch.float32)
    assert x.ndim == 2

    h, w = x.shape
    bh, bw = block_size
    device = x.device
    out_config = get_cast_output_config(fmt, block_size, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)
    max_quant_val = _max_quant_value_for_config(out_config)

    if h == 0:
        out = torch.empty((0, w), dtype=torch.float32, device=device)
        if sf is not None:
            return out
        sf_h = 0
        sf_w = ceil_div(w, bw)
        if use_packed_ue8m0:
            dq_sf = torch.empty((sf_h, ceil_div(sf_w, 4)), dtype=torch.int32, device=device)
        else:
            dq_sf = torch.empty((sf_h, sf_w), dtype=torch.float32, device=device)
        return out, dq_sf

    pad_h = (bh - h % bh) % bh
    pad_w = (bw - w % bw) % bw
    padded_src = F.pad(x.to(torch.float32), (0, pad_w, 0, pad_h))
    ph, pw = padded_src.shape
    valid_mask = torch.zeros((ph, pw), dtype=torch.bool, device=device)
    valid_mask[:h, :w] = True

    if sf is None:
        reshaped_for_max = padded_src.view(ph // bh, bh, pw // bw, bw).permute(0, 2, 1, 3).reshape(ph // bh, pw // bw, -1)
        reshaped_mask = valid_mask.view(ph // bh, bh, pw // bw, bw).permute(0, 2, 1, 3).reshape(ph // bh, pw // bw, -1)
        abs_f = torch.where(reshaped_mask, reshaped_for_max.abs(), torch.tensor(-1.0, device=device, dtype=torch.float32))
        max_val = abs_f.max(dim=-1, keepdim=True)[0].clamp(min=out_config.clamp_min_value)
        max_quant_val_expanded = max_val.new_full(max_val.shape, max_quant_val, dtype=torch.float32)
        dequant_sf = max_val / max_quant_val_expanded
        ds_int = dequant_sf.view(torch.int32)
        if round_sf:
            ds_int_rounded = (ds_int + 0x007FFFFF) & 0x7F800000
            dequant_sf_rounded = ds_int_rounded.view(torch.float32)
            quant_sf = torch.where(dequant_sf_rounded == 0, torch.tensor(0.0, device=device), 1.0 / dequant_sf_rounded)
        else:
            ds_int_rounded = ds_int
            quant_sf = torch.where(ds_int_rounded == 0, torch.tensor(0.0, device=device), max_quant_val_expanded / max_val)
    else:
        expected_sf_shape = (ph // bh, pw // bw)
        sf_config = CastInputConfig(
            torch_dtype=torch.float32,
            sf_block=block_size,
            with_sf=True,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            use_packed_ue8m0=use_packed_ue8m0,
        )
        sf = transform_sf_for_ref(sf, sf_config, (h, w), already_internal_layout=False)
        assert tuple(sf.shape) == expected_sf_shape, (tuple(sf.shape), expected_sf_shape)
        quant_sf = sf.to(device=device, dtype=torch.float32).reciprocal().unsqueeze(-1)

    padded_src_view = padded_src.view(ph // bh, bh, pw // bw, bw)
    quant_sf_view = quant_sf.view(ph // bh, 1, pw // bw, 1)
    quant_tensor = (padded_src_view * quant_sf_view).reshape(ph, pw)[:h, :w]

    out = quant_tensor.to(torch.float32)

    if sf is not None:
        return out

    dq_sf = format_sf(ds_int_rounded.squeeze(-1).detach().cpu(), use_tma_aligned_col_major_sf, use_packed_ue8m0)
    return out, dq_sf.to(device)


def generate_num_tokens(alignment: int = 1, is_benchmark: bool = False) -> list[int]:
    do_full_test = os.getenv("TK_FULL_TEST") in ("1", "true", "True")
    values = [4001]
    if do_full_test and not is_benchmark:
        values.insert(0, 0)
    return [((value + alignment - 1) // alignment) * alignment for value in values]


def generate_hidden_sizes(alignment: int = 64) -> list[int]:
    return [value for value in (576, 7168) if value % alignment == 0]


def make_param_id(params: dict) -> str:
    return "-".join(f"{key}={value}" for key, value in params.items())


def generate_test_params() -> list[dict]:
    return [
        {
            "num_tokens": num_tokens,
            "hidden": hidden,
            "in_dtype": in_dtype,
            "input_with_sf": input_with_sf,
            "fmt": fmt,
            "num_per_channels": num_per_channels,
            "x_block_size": x_block_size,
            "use_tma_aligned_col_major_sf": use_tma_aligned_col_major_sf,
            "round_sf": round_sf,
            "use_packed_ue8m0": use_packed_ue8m0,
        }
        for num_tokens in generate_num_tokens(is_benchmark=False)
        for hidden in generate_hidden_sizes()
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in ((False, True, False), (True, True, True))
        for input_with_sf in (False, True)
        for in_dtype in (torch.float32, torch.bfloat16)
        for num_per_channels in ((32, 128) if input_with_sf else (32, 64, 128, hidden))
        for x_block_size in (((128, 128), (32, 32)) if input_with_sf else (None,))
        for fmt in ("fp8", "fp4")
    ]


def assert_bit_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_cpu = actual.cpu().contiguous()
    expected_cpu = expected.cpu().contiguous()
    assert actual_cpu.shape == expected_cpu.shape, (actual_cpu.shape, expected_cpu.shape)
    assert actual_cpu.dtype == expected_cpu.dtype, (actual_cpu.dtype, expected_cpu.dtype)
    torch.testing.assert_close(actual_cpu.view(torch.uint8), expected_cpu.view(torch.uint8), rtol=1e-3, atol=1e-3)


def assert_fp32_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=1e-3, atol=1e-3)


def _check_case(params: dict) -> None:
    num_tokens = params["num_tokens"]
    hidden = params["hidden"]
    in_dtype = params["in_dtype"]
    input_with_sf = params["input_with_sf"]
    fmt = params["fmt"]
    num_per_channels = params["num_per_channels"]
    x_block_size = params.get("x_block_size")
    use_tma_aligned_col_major_sf = params["use_tma_aligned_col_major_sf"]
    round_sf = params["round_sf"]
    use_packed_ue8m0 = params["use_packed_ue8m0"]
    assert (x_block_size is not None) == input_with_sf
    print(
        f"Testing per_token_cast: tokens={num_tokens}, hidden={hidden}, "
        f"in_dtype={in_dtype}, fmt={fmt}, "
        f"block={num_per_channels}, x_block={x_block_size}, "
        f"tma={use_tma_aligned_col_major_sf}, "
        f"round_sf={round_sf}, ue8={use_packed_ue8m0}"
    )

    x_data = torch.randn((num_tokens, hidden), dtype=in_dtype, device="npu")
    if input_with_sf:
        input_config = CastInputConfig(
            torch_dtype=in_dtype,
            sf_block=x_block_size,
            with_sf=True,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            use_packed_ue8m0=use_packed_ue8m0,
        )
        x_sf = generate_input_scaling_factors((num_tokens, hidden), input_config, x_data.device)
        x = (x_data, x_sf)
        original_x = expand_input_with_sf_ref(x, x_block_size, use_tma_aligned_col_major_sf, use_packed_ue8m0)
    else:
        x = x_data
        original_x = x_data

    out, out_sf = per_token_cast(
        x,
        fmt,
        num_per_channels=num_per_channels,
        x_block_size=x_block_size,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    torch.npu.synchronize()
    assert out.device == x_data.device
    assert out_sf.device == x_data.device
    assert out.dtype == torch.float32
    assert out.shape == x_data.shape

    ref_out, ref_sf = cast_ref(
        x,
        fmt,
        (1, num_per_channels),
        x_block_size=x_block_size,
        round_sf=round_sf,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    assert_fp32_close(out, ref_out)
    if use_packed_ue8m0:
        assert_bit_equal(clear_unused_sf(out_sf, hidden, num_per_channels), clear_unused_sf(ref_sf, hidden, num_per_channels))
    else:
        assert_bit_equal(out_sf, ref_sf)

    out_back = cast_back_quantized((out, out_sf), (1, num_per_channels), use_packed_ue8m0)
    check_bias(out_back, original_x)

    if not input_with_sf and fmt == "fp8" and not use_packed_ue8m0 and num_tokens > 0:
        x_non_contiguous = torch.randn((num_tokens, hidden * 2), dtype=in_dtype, device="npu")[:, :hidden]
        x_non_contiguous.copy_(original_x)
        non_contiguous_out, non_contiguous_sf = per_token_cast(
            x_non_contiguous,
            fmt,
            num_per_channels=num_per_channels,
            x_block_size=x_block_size,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            round_sf=round_sf,
            use_packed_ue8m0=use_packed_ue8m0,
        )
        assert non_contiguous_out.device == x_data.device
        assert non_contiguous_sf.device == x_data.device
        assert non_contiguous_out.dtype == torch.float32
        assert_fp32_close(non_contiguous_out, out)
        assert_bit_equal(non_contiguous_sf, out_sf)

    if not input_with_sf and num_per_channels != hidden:
        if not use_tma_aligned_col_major_sf:
            out_with_sf = per_token_cast_with_precomputed_sf(
                x,
                fmt,
                num_per_channels=num_per_channels,
                sf=out_sf,
                x_block_size=x_block_size,
                use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
                round_sf=round_sf,
                use_packed_ue8m0=use_packed_ue8m0,
            )
            assert out_with_sf.device == x_data.device
            assert out_with_sf.dtype == torch.float32
            ref_with_sf = cast_ref(
                x,
                fmt,
                (1, num_per_channels),
                sf=out_sf,
                x_block_size=x_block_size,
                round_sf=round_sf,
                use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
                use_packed_ue8m0=use_packed_ue8m0,
            )
            assert_fp32_close(out_with_sf, ref_with_sf)

        sf_only = per_token_cast_with_sf_only(
            x,
            fmt,
            num_per_channels=num_per_channels,
            x_block_size=x_block_size,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            round_sf=round_sf,
            use_packed_ue8m0=use_packed_ue8m0,
        )
        assert sf_only.device == x_data.device
        if use_packed_ue8m0:
            assert_bit_equal(clear_unused_sf(sf_only, hidden, num_per_channels), clear_unused_sf(ref_sf, hidden, num_per_channels))
        else:
            assert_bit_equal(sf_only, ref_sf)

    print("  PASS")


@pytest.mark.parametrize("params", generate_test_params(), ids=make_param_id)
def test_per_token_cast_npu(params: dict) -> None:
    _check_case(params)


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v"])
    if exit_code == pytest.ExitCode.OK:
        print("All per_token_cast tests passed! Kernel Output Match!")
    sys.exit(exit_code)
