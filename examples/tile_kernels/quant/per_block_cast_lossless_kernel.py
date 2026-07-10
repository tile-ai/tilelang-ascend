import os
import sys
from dataclasses import replace
from typing import Callable
import pytest
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

try:
    from .utils import *
except ImportError:
    from utils import *  # type: ignore[no-redef]
tilelang.cache.clear_cache()
os.environ.setdefault("TILELANG_PRINT_ON_COMPILATION", "0")


DEFAULT_IN_SF_BLOCK = (1, 32)
DEFAULT_OUT_SF_BLOCK = (1, 128)
VEC_NUM = 2
NUM_ELEMENTS_PER_BLOCK = 8192
INPUT_DTYPE = "bfloat16"
OUTPUT_DTYPE = "float32"
INPUT_MAX_QUANT_VAL = 6.0
INPUT_MIN_CLAMP_VAL = 6.0 * 2.0 ** (-126)
pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False, tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True}


def _derive_cast_layout(hidden: int, in_config: CastInputConfig, out_config: CastOutputConfig) -> tuple[int, int]:
    assert in_config.dtype == INPUT_DTYPE and out_config.dtype == OUTPUT_DTYPE, (
        "lossless mode only supports bf16 -> fp32 conversion currently"
    )
    assert in_config.with_sf, "lossless mode requires both input and output scaling factors"
    assert is_power_of_two(in_config.sf_block[1]) and is_power_of_two(out_config.sf_block[1]), (
        "block_k must be power of 2 for lossless mode"
    )
    assert out_config.sf_block[0] % in_config.sf_block[0] == 0 and out_config.sf_block[1] % in_config.sf_block[1] == 0, (
        "Output block size must be multiple of input block size"
    )
    if in_config.sf_block == DEFAULT_IN_SF_BLOCK and out_config.sf_block == (32, 32):
        block_m = 64
        block_k = 128
    elif in_config.sf_block == DEFAULT_IN_SF_BLOCK and out_config.sf_block == DEFAULT_OUT_SF_BLOCK:
        block_m = 128
        block_k_candidates = (256, 512)
        block_k = min(
            (
                candidate
                for candidate in block_k_candidates
                if candidate % in_config.sf_block[1] == 0 and candidate % out_config.sf_block[1] == 0
            ),
            key=lambda candidate: (align_up(hidden, candidate), -candidate),
        )
    else:
        block_m = max(out_config.sf_block[0], 32)
        block_k = max(out_config.sf_block[1], NUM_ELEMENTS_PER_BLOCK // block_m)
    assert block_m % out_config.sf_block[0] == 0
    assert block_k % out_config.sf_block[1] == 0
    assert hidden > 0
    return block_m, block_k


def _pack_plain_sf_for_lossless_kernel(
    x_sf_padded: torch.Tensor,
    padded_tokens: int,
    padded_hidden: int,
    block_m: int,
    block_k: int,
    in_sf_block: tuple[int, int],
) -> torch.Tensor:
    """Pack plain fp32 input sf into contiguous per-kernel tiles."""
    in_sf_block_m, in_sf_block_k = in_sf_block
    data_tile_m = min(block_m, 32, NUM_ELEMENTS_PER_BLOCK // block_k)
    num_data_tiles_m = block_m // data_tile_m
    num_in_sf_per_block_m = block_m // in_sf_block_m
    num_in_sf_per_block_k = block_k // in_sf_block_k
    num_in_sf_per_data_tile_m = data_tile_m // in_sf_block_m
    m_num = padded_tokens // block_m
    n_num = padded_hidden // block_k
    x_sf_tile = x_sf_padded.view(m_num, num_in_sf_per_block_m, n_num, num_in_sf_per_block_k)
    x_sf_tile = x_sf_tile.view(m_num, num_data_tiles_m, num_in_sf_per_data_tile_m, n_num, num_in_sf_per_block_k)
    x_sf_tile = x_sf_tile.permute(0, 3, 1, 2, 4).contiguous()
    return x_sf_tile.view(m_num, n_num, num_data_tiles_m, num_in_sf_per_data_tile_m * num_in_sf_per_block_k)


def _needs_tile_packed_input_sf(in_config: CastInputConfig, out_config: CastOutputConfig, num_data_tiles_m: int) -> bool:
    """Use tile-packed plain SF for layouts where strided SF copy is unstable.

    128x128 plain also uses 32-row data tiles. Keep it on the stable
    generic vid==0 path, but feed x_sf through the same packed layout so
    each data_m sees a contiguous (32, 4) SF tile.
    """
    return (
        not in_config.use_packed_ue8m0
        and (not in_config.use_tma_aligned_col_major_sf)
        and out_config.sf_block in ((32, 32), (128, 128))
        and num_data_tiles_m > 1
    )


def _pad_2d_tensor(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape) == shape:
        return x if x.is_contiguous() else x.contiguous()
    padded = torch.empty(shape, dtype=x.dtype, device=x.device)
    padded.zero_()
    copy_m = min(x.shape[0], shape[0])
    copy_k = min(x.shape[1], shape[1])
    padded[:copy_m, :copy_k].copy_(x[:copy_m, :copy_k])
    return padded


def _get_input_sf_load_spec(
    input_sf_is_tile_packed: bool,
    in_config: CastInputConfig,
    num_in_sf_per_block_k: int,
    num_in_sf_per_data_tile_m: int,
    packed_sf_tile_elems: int,
) -> tuple[tuple[int, ...], str]:
    """Return the UB shape and dtype required by the selected SF layout."""
    if input_sf_is_tile_packed:
        return ((packed_sf_tile_elems,), "float32")
    if in_config.use_packed_ue8m0:
        assert num_in_sf_per_block_k % 4 == 0
        return ((num_in_sf_per_block_k // 4, num_in_sf_per_data_tile_m * 4), "uint8")
    if in_config.use_tma_aligned_col_major_sf:
        return ((num_in_sf_per_block_k, num_in_sf_per_data_tile_m), "float32")
    return ((num_in_sf_per_data_tile_m, num_in_sf_per_block_k), "float32")


@tilelang.jit(out_idx=[2, 3], pass_configs=pass_configs)
def get_per_block_cast_lossless_kernel(
    hidden: int,
    block_m: int,
    block_k: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    in_sf_block_m: int = DEFAULT_IN_SF_BLOCK[0],
    in_sf_block_k: int = DEFAULT_IN_SF_BLOCK[1],
    out_sf_block_m: int = DEFAULT_OUT_SF_BLOCK[0],
    out_sf_block_k: int = DEFAULT_OUT_SF_BLOCK[1],
    input_sf_is_tile_packed: bool = False,
):
    assert block_m > 0 and block_k > 0
    assert block_m % out_sf_block_m == 0
    assert block_k % out_sf_block_k == 0
    assert out_sf_block_m % in_sf_block_m == 0
    assert out_sf_block_k % in_sf_block_k == 0
    num_tokens = T.symbolic("num_tokens")
    m_num = (num_tokens + block_m - 1) // block_m
    n_num = hidden // block_k
    num_in_sf_per_block_m = block_m // in_sf_block_m
    num_in_sf_per_block_k = block_k // in_sf_block_k
    num_out_sf_per_block_m = block_m // out_sf_block_m
    num_out_sf_per_block_k = block_k // out_sf_block_k
    num_in_sf_per_out_sf_m = out_sf_block_m // in_sf_block_m
    num_in_sf_per_out_sf_k = out_sf_block_k // in_sf_block_k
    out_sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    packed_data_tile_m = min(block_m, 32, NUM_ELEMENTS_PER_BLOCK // block_k)
    assert packed_data_tile_m > 0 and block_m % packed_data_tile_m == 0
    assert packed_data_tile_m % in_sf_block_m == 0
    packed_num_data_tiles_m = block_m // packed_data_tile_m
    packed_num_in_sf_per_data_tile_m = packed_data_tile_m // in_sf_block_m
    use_32_row_data_tiles = input_sf_is_tile_packed or (out_sf_block_m == 128 and out_sf_block_k == 128)
    data_tile_m = packed_data_tile_m if use_32_row_data_tiles else block_m
    num_data_tiles_m = packed_num_data_tiles_m if use_32_row_data_tiles else 1
    num_in_sf_per_data_tile_m = packed_num_in_sf_per_data_tile_m if use_32_row_data_tiles else num_in_sf_per_block_m
    packed_sf_tile_elems = num_in_sf_per_data_tile_m * num_in_sf_per_block_k
    x_sf_shape = (
        (m_num, n_num, num_data_tiles_m, packed_sf_tile_elems) if input_sf_is_tile_packed else get_sf_shape((num_tokens, hidden), in_config)
    )
    x_sf_load_shape, x_sf_load_dtype = _get_input_sf_load_spec(
        input_sf_is_tile_packed,
        in_config,
        num_in_sf_per_block_k,
        num_in_sf_per_data_tile_m,
        packed_sf_tile_elems,
    )
    fast_data_tile_m = block_m // VEC_NUM
    fast_num_in_sf_per_data_tile_m = fast_data_tile_m // in_sf_block_m
    fast_packed_sf_tile_elems = fast_num_in_sf_per_data_tile_m * num_in_sf_per_block_k
    fast_x_sf_load_shape, fast_x_sf_load_dtype = _get_input_sf_load_spec(
        False,
        in_config,
        num_in_sf_per_block_k,
        fast_num_in_sf_per_data_tile_m,
        fast_packed_sf_tile_elems,
    )
    fast_num_pipeline_pairs = num_out_sf_per_block_k * 2
    fast_num_initial_prefetch_stages = min(4, fast_num_pipeline_pairs)
    use_max4_fast_path = (
        in_sf_block_m == 1
        and out_sf_block_m == 1
        and num_in_sf_per_out_sf_m == 1
        and num_in_sf_per_out_sf_k == 4
        and (not input_sf_is_tile_packed)
        and block_m % VEC_NUM == 0
    )
    use_32x32_packed_fast_path = (
        in_sf_block_m == 1
        and in_sf_block_k == 32
        and out_sf_block_m == 32
        and out_sf_block_k == 32
        and num_in_sf_per_out_sf_m == 32
        and num_in_sf_per_out_sf_k == 1
        and num_out_sf_per_block_m == VEC_NUM
        and block_m == 64
        and block_k == 128
        and in_config.use_packed_ue8m0
        and (not input_sf_is_tile_packed)
    )
    use_32x32_tile_packed_fast_path = (
        input_sf_is_tile_packed
        and in_sf_block_m == 1
        and in_sf_block_k == 32
        and out_sf_block_m == 32
        and out_sf_block_k == 32
        and num_in_sf_per_out_sf_m == 32
        and num_in_sf_per_out_sf_k == 1
        and num_out_sf_per_block_m == VEC_NUM
        and block_m == 64
        and block_k == 128
    )
    use_128x128_generic_fast_path = (
        in_sf_block_m == 1
        and in_sf_block_k == 32
        and out_sf_block_m == 128
        and out_sf_block_k == 128
        and num_in_sf_per_out_sf_m == 128
        and num_in_sf_per_out_sf_k == 4
        and num_out_sf_per_block_m == 1
        and num_out_sf_per_block_k == 1
        and block_m == 128
        and block_k == 128
    )
    use_128x128_plain_fast_path = (
        use_128x128_generic_fast_path
        and (not in_config.use_packed_ue8m0)
        and (not in_config.use_tma_aligned_col_major_sf)
        and (not input_sf_is_tile_packed)
    )
    use_128x128_tile_packed_plain_fast_path = use_128x128_generic_fast_path and input_sf_is_tile_packed and (not in_config.use_packed_ue8m0)

    @T.macro
    def load_input_sf_block(dst, x_sf, sf_m, sf_k, data_m, pid_token, pid_hidden):
        if input_sf_is_tile_packed:
            T.copy(x_sf[pid_token, pid_hidden, data_m, 0:packed_sf_tile_elems], dst)
        elif in_config.use_packed_ue8m0:
            T.copy(x_sf[sf_k // 4 : sf_k // 4 + num_in_sf_per_block_k // 4, sf_m * 4 : sf_m * 4 + num_in_sf_per_data_tile_m * 4], dst)
        elif in_config.use_tma_aligned_col_major_sf:
            T.copy(x_sf[sf_k : sf_k + num_in_sf_per_block_k, sf_m : sf_m + num_in_sf_per_data_tile_m], dst)
        else:
            T.copy(x_sf[sf_m : sf_m + num_in_sf_per_data_tile_m, sf_k : sf_k + num_in_sf_per_block_k], dst)

    @T.macro
    def decode_input_sf_exp(dst, src):
        if (not in_config.use_packed_ue8m0) and (not in_config.use_tma_aligned_col_major_sf) and (not input_sf_is_tile_packed):
            src_bits_ub = T.alloc_ub((num_in_sf_per_data_tile_m, num_in_sf_per_block_k), "int32")
            T.reinterpretcast(src_bits_ub, src, "int32_t")
            T.tile.bitwise_rshift(dst, src_bits_ub, 23)
            T.pipe_barrier("v")
            T.tile.bitwise_and(dst, dst, 255)
        else:
            if in_config.use_packed_ue8m0:
                for i in T.serial(num_in_sf_per_data_tile_m):
                    for j in T.serial(num_in_sf_per_block_k):
                        dst[i, j] = T.Cast("int32", src[j // 4, i * 4 + j % 4])
            elif input_sf_is_tile_packed:
                sf_value = T.alloc_var("float32", init=0.0)
                sf_bits = T.alloc_var("int32", init=0)
                for idx in T.unroll(packed_sf_tile_elems):
                    sf_value = src[idx]
                    sf_bits = T.reinterpret("int32", sf_value)
                    dst[idx // num_in_sf_per_block_k, idx % num_in_sf_per_block_k] = (sf_bits >> 23) & 255
            else:
                for i in T.serial(num_in_sf_per_data_tile_m):
                    for j in T.serial(num_in_sf_per_block_k):
                        sf_value = T.alloc_var("float32", init=0.0)
                        sf_bits = T.alloc_var("int32", init=0)
                        if in_config.use_tma_aligned_col_major_sf:
                            sf_value = src[j, i]
                        else:
                            sf_value = src[i, j]
                        sf_bits = T.reinterpret("int32", sf_value)
                        dst[i, j] = sf_bits >> 23 & 255

    @T.macro
    def store_output_sf(out_sf, out_sf_fp32_ub, out_sf_exp_ub, i, j, sf_m, sf_k):
        if out_config.use_packed_ue8m0:
            out_sf[sf_k // 4, sf_m * 4 + sf_k % 4] = T.Cast("uint8", out_sf_exp_ub[i, j])
        elif out_config.use_tma_aligned_col_major_sf:
            out_sf[sf_k, sf_m] = out_sf_fp32_ub[i, j]
        else:
            out_sf[sf_m, sf_k] = out_sf_fp32_ub[i, j]

    @T.macro
    def store_output_sf_scalar(out_sf, out_sf_exp, sf_m, sf_k):
        if out_config.use_packed_ue8m0:
            out_sf[sf_k // 4, sf_m * 4 + sf_k % 4] = T.Cast("uint8", out_sf_exp)
        else:
            out_sf_bits = T.alloc_var("int32", init=0)
            out_sf_value = T.alloc_var("float32", init=0.0)
            out_sf_bits = out_sf_exp << 23
            out_sf_value = T.reinterpret("float32", out_sf_bits)
            if out_config.use_tma_aligned_col_major_sf:
                out_sf[sf_k, sf_m] = out_sf_value
            else:
                out_sf[sf_m, sf_k] = out_sf_value

    @T.macro
    def load_and_decode_input_sf(x_sf_exp_ub, x_sf_load_ub, x_sf, sf_m, sf_k, data_m, pid_token, pid_hidden):
        load_input_sf_block(x_sf_load_ub, x_sf, sf_m, sf_k, data_m, pid_token, pid_hidden)
        T.pipe_barrier("all")
        decode_input_sf_exp(x_sf_exp_ub, x_sf_load_ub)

    @T.macro
    def load_input_sf_block_fast(dst, x_sf, sf_m, sf_k):
        if in_config.use_packed_ue8m0:
            T.copy(x_sf[sf_k // 4 : sf_k // 4 + num_in_sf_per_block_k // 4, sf_m * 4 : sf_m * 4 + fast_num_in_sf_per_data_tile_m * 4], dst)
        elif in_config.use_tma_aligned_col_major_sf:
            T.copy(x_sf[sf_k : sf_k + num_in_sf_per_block_k, sf_m : sf_m + fast_num_in_sf_per_data_tile_m], dst)
        else:
            T.copy(x_sf[sf_m : sf_m + fast_num_in_sf_per_data_tile_m, sf_k : sf_k + num_in_sf_per_block_k], dst)

    @T.macro
    def decode_input_sf_exp_fast(dst, src):
        for i in T.serial(fast_num_in_sf_per_data_tile_m):
            for j in T.serial(num_in_sf_per_block_k):
                if in_config.use_packed_ue8m0:
                    dst[i, j] = T.Cast("int32", src[j // 4, i * 4 + j % 4])
                else:
                    sf_value = T.alloc_var("float32", init=0.0)
                    sf_bits = T.alloc_var("int32", init=0)
                    if in_config.use_tma_aligned_col_major_sf:
                        sf_value = src[j, i]
                    else:
                        sf_value = src[i, j]
                    sf_bits = T.reinterpret("int32", sf_value)
                    dst[i, j] = sf_bits >> 23 & 255

    @T.macro
    def load_and_decode_input_sf_fast(x_sf_exp_ub, x_sf_load_ub, x_sf, sf_m, sf_k):
        load_input_sf_block_fast(x_sf_load_ub, x_sf, sf_m, sf_k)
        T.pipe_barrier("all")
        decode_input_sf_exp_fast(x_sf_exp_ub, x_sf_load_ub)

    @T.macro
    def reduce_output_sf_exp(out_sf_exp_ub, x_sf_exp_ub, data_m):
        for i in T.serial(num_in_sf_per_data_tile_m):
            for j in T.serial(num_in_sf_per_block_k):
                out_sf_exp_ub[
                    (data_m * num_in_sf_per_data_tile_m + i) // num_in_sf_per_out_sf_m,
                    j // num_in_sf_per_out_sf_k,
                ] = T.max(
                    out_sf_exp_ub[
                        (data_m * num_in_sf_per_data_tile_m + i) // num_in_sf_per_out_sf_m,
                        j // num_in_sf_per_out_sf_k,
                    ],
                    x_sf_exp_ub[i, j],
                )

    @T.macro
    def reduce_128x128_exp_rows(
        x_sf_exp_ub,
        row_offset_u32_ub,
        e0_ub,
        e1_ub,
        e2_ub,
        e3_ub,
        max01_ub,
        max23_ub,
        tile_max_ub,
        block_max_ub,
        base_offset,
    ):
        T.tile.gather(e0_ub, x_sf_exp_ub, row_offset_u32_ub, base_offset)
        T.tile.gather(e1_ub, x_sf_exp_ub, row_offset_u32_ub, base_offset + 4)
        T.tile.gather(e2_ub, x_sf_exp_ub, row_offset_u32_ub, base_offset + 8)
        T.tile.gather(e3_ub, x_sf_exp_ub, row_offset_u32_ub, base_offset + 12)
        T.pipe_barrier("v")
        T.tile.max(max01_ub, e0_ub, e1_ub)
        T.tile.max(max23_ub, e2_ub, e3_ub)
        T.pipe_barrier("v")
        T.tile.max(tile_max_ub, max01_ub, max23_ub)
        T.pipe_barrier("v")
        T.tile.max(block_max_ub, block_max_ub, tile_max_ub)

    @T.macro
    def load_reduce_128_cache(
        cache_ub,
        x_sf_load_ub,
        x_sf,
        sf_m,
        sf_k,
        data_m,
        pid_token,
        pid_hidden,
        offset_u32_ub,
        e0_ub,
        e1_ub,
        e2_ub,
        e3_ub,
        max01_ub,
        max23_ub,
        tile_max_ub,
        block_max_ub,
    ):
        if input_sf_is_tile_packed:
            load_input_sf_block(x_sf_load_ub, x_sf, sf_m, sf_k, data_m, pid_token, pid_hidden)
            T.set_flag("mte2", "v", 2)
            T.wait_flag("mte2", "v", 2)
            decode_input_sf_exp(cache_ub, x_sf_load_ub)
        else:
            load_and_decode_input_sf(
                cache_ub,
                x_sf_load_ub,
                x_sf,
                sf_m,
                sf_k,
                data_m,
                pid_token,
                pid_hidden,
            )
        reduce_128x128_exp_rows(
            cache_ub,
            offset_u32_ub,
            e0_ub,
            e1_ub,
            e2_ub,
            e3_ub,
            max01_ub,
            max23_ub,
            tile_max_ub,
            block_max_ub,
            0,
        )

    @T.macro
    def decode_reduce_128_plain_pair(
        cache64_ub,
        load64_ub,
        bits64_ub,
        offset_u32_ub,
        e0_ub,
        e1_ub,
        e2_ub,
        e3_ub,
        max01_ub,
        max23_ub,
        tile_max_ub,
        block_max_ub,
    ):
        T.reinterpretcast(bits64_ub, load64_ub, "int32_t")
        T.tile.bitwise_rshift(cache64_ub, bits64_ub, 23)
        T.pipe_barrier("v")
        reduce_128x128_exp_rows(
            cache64_ub,
            offset_u32_ub,
            e0_ub,
            e1_ub,
            e2_ub,
            e3_ub,
            max01_ub,
            max23_ub,
            tile_max_ub,
            block_max_ub,
            0,
        )
        reduce_128x128_exp_rows(
            cache64_ub,
            offset_u32_ub,
            e0_ub,
            e1_ub,
            e2_ub,
            e3_ub,
            max01_ub,
            max23_ub,
            tile_max_ub,
            block_max_ub,
            512,
        )

    @T.macro
    def store_output_sf_block(out_sf, out_sf_fp32_ub, out_sf_exp_ub, sf_m, sf_k):
        if num_out_sf_per_block_m == 1 and num_out_sf_per_block_k == 1:
            out_sf_bits = T.alloc_var("int32", init=0)
            out_sf_bits = out_sf_exp_ub[0, 0] << 23
            out_sf_fp32_ub[0, 0] = T.reinterpret("float32", out_sf_bits)
            store_output_sf(out_sf, out_sf_fp32_ub, out_sf_exp_ub, 0, 0, sf_m, sf_k)
        else:
            for i in T.serial(num_out_sf_per_block_m):
                for j in T.serial(num_out_sf_per_block_k):
                    out_sf_bits = T.alloc_var("int32", init=0)
                    out_sf_bits = out_sf_exp_ub[i, j] << 23
                    out_sf_fp32_ub[i, j] = T.reinterpret("float32", out_sf_bits)
                    store_output_sf(out_sf, out_sf_fp32_ub, out_sf_exp_ub, i, j, sf_m + i, sf_k + j)

    @T.macro
    def update_relative_sf_exp(x_sf_exp_ub, out_sf_exp_ub, data_m):
        for i in T.serial(num_in_sf_per_data_tile_m):
            for j in T.serial(num_in_sf_per_block_k):
                x_sf_exp_ub[i, j] = (
                    x_sf_exp_ub[i, j]
                    - out_sf_exp_ub[
                        (data_m * num_in_sf_per_data_tile_m + i) // num_in_sf_per_out_sf_m,
                        j // num_in_sf_per_out_sf_k,
                    ]
                    + 127
                )

    @T.macro
    def expand_relative_sf(x_relative_sf_ub, x_sf_exp_ub):
        for i in T.serial(data_tile_m):
            for j in T.serial(block_k):
                relative_sf_bits = T.alloc_var("int32", init=0)
                relative_sf_bits = x_sf_exp_ub[i // in_sf_block_m, j // in_sf_block_k] << 23
                x_relative_sf_ub[i, j] = T.reinterpret("float32", relative_sf_bits)

    @T.macro
    def cast_data_tile(x, out, x_in_ub, x_out_ub, x_relative_sf_ub, row_offset, col_offset):
        T.copy(x[row_offset : row_offset + data_tile_m, col_offset : col_offset + block_k], x_in_ub)
        T.pipe_barrier("all")
        T.tile.cast(x_out_ub, x_in_ub, mode="CAST_NONE", count=data_tile_m * block_k)
        T.tile.mul(x_out_ub, x_out_ub, x_relative_sf_ub)
        T.pipe_barrier("all")
        T.copy(x_out_ub, out[row_offset : row_offset + data_tile_m, col_offset : col_offset + block_k])

    @T.macro
    def apply_32x32_packed_fast_path(x_sf, x, out, out_sf, sf_m, sf_k, out_sf_m, out_sf_k, row_offset, col_offset):
        tile_elem_count = 32 * in_sf_block_k
        x_sf_load_ub = T.alloc_ub(fast_x_sf_load_shape, fast_x_sf_load_dtype)
        x_sf_word_ub = T.alloc_ub((32,), "int32")
        e0_ub = T.alloc_ub((32,), "int32")
        e1_ub = T.alloc_ub((32,), "int32")
        e2_ub = T.alloc_ub((32,), "int32")
        e3_ub = T.alloc_ub((32,), "int32")
        out_sf_exp_ub = T.alloc_ub((1, 4), "int32")
        out_sf_fp32_ub = T.alloc_ub((1, 4), "float32")
        relative_exp0_ub = T.alloc_ub((32,), "int32")
        relative_exp1_ub = T.alloc_ub((32,), "int32")
        relative_exp2_ub = T.alloc_ub((32,), "int32")
        relative_exp3_ub = T.alloc_ub((32,), "int32")
        relative_bits0_ub = T.alloc_ub((32,), "int32")
        relative_bits1_ub = T.alloc_ub((32,), "int32")
        relative_bits2_ub = T.alloc_ub((32,), "int32")
        relative_bits3_ub = T.alloc_ub((32,), "int32")
        relative_sf0_view_ub = T.alloc_ub((32, 1), "float32")
        relative_sf1_view_ub = T.alloc_ub((32, 1), "float32")
        relative_sf2_view_ub = T.alloc_ub((32, 1), "float32")
        relative_sf3_view_ub = T.alloc_ub((32, 1), "float32")
        relative_sf_tile0_ub = T.alloc_ub((32, in_sf_block_k), "float32")
        relative_sf_tile1_ub = T.alloc_ub((32, in_sf_block_k), "float32")
        relative_sf_tile2_ub = T.alloc_ub((32, in_sf_block_k), "float32")
        relative_sf_tile3_ub = T.alloc_ub((32, in_sf_block_k), "float32")
        x_in0_ub = T.alloc_ub((32, in_sf_block_k), INPUT_DTYPE)
        x_in1_ub = T.alloc_ub((32, in_sf_block_k), INPUT_DTYPE)
        x_out0_ub = T.alloc_ub((32, in_sf_block_k), OUTPUT_DTYPE)
        x_out1_ub = T.alloc_ub((32, in_sf_block_k), OUTPUT_DTYPE)
        T.reinterpretcast(x_sf_word_ub, x_sf_load_ub, "int32_t")
        T.reinterpretcast(relative_sf0_view_ub, relative_bits0_ub, "float")
        T.reinterpretcast(relative_sf1_view_ub, relative_bits1_ub, "float")
        T.reinterpretcast(relative_sf2_view_ub, relative_bits2_ub, "float")
        T.reinterpretcast(relative_sf3_view_ub, relative_bits3_ub, "float")
        load_input_sf_block_fast(x_sf_load_ub, x_sf, sf_m, sf_k)
        T.set_flag("mte2", "v", 2)
        T.wait_flag("mte2", "v", 2)
        T.copy(x[row_offset : row_offset + 32, col_offset : col_offset + in_sf_block_k], x_in0_ub)
        T.copy(x[row_offset : row_offset + 32, col_offset + in_sf_block_k : col_offset + 2 * in_sf_block_k], x_in1_ub)
        T.set_flag("mte2", "v", 0)
        for word_group in T.unroll(num_in_sf_per_block_k // 4):
            max_exp0 = T.alloc_var("int32", init=0)
            max_exp1 = T.alloc_var("int32", init=0)
            max_exp2 = T.alloc_var("int32", init=0)
            max_exp3 = T.alloc_var("int32", init=0)
            for i in T.serial(32):
                packed_word = T.alloc_var("int32", init=0)
                exp0 = T.alloc_var("int32", init=0)
                exp1 = T.alloc_var("int32", init=0)
                exp2 = T.alloc_var("int32", init=0)
                exp3 = T.alloc_var("int32", init=0)
                packed_word = x_sf_word_ub[i]
                exp0 = T.Cast("int32", T.Cast("uint8", packed_word))
                exp1 = T.Cast("int32", T.Cast("uint8", packed_word >> 8))
                exp2 = T.Cast("int32", T.Cast("uint8", packed_word >> 16))
                exp3 = T.Cast("int32", T.Cast("uint8", packed_word >> 24))
                e0_ub[i] = exp0
                e1_ub[i] = exp1
                e2_ub[i] = exp2
                e3_ub[i] = exp3
                max_exp0 = T.max(max_exp0, exp0)
                max_exp1 = T.max(max_exp1, exp1)
                max_exp2 = T.max(max_exp2, exp2)
                max_exp3 = T.max(max_exp3, exp3)
            out_sf_exp_ub[0, 0] = T.max(max_exp0 - 6, 0)
            out_sf_exp_ub[0, 1] = T.max(max_exp1 - 6, 0)
            out_sf_exp_ub[0, 2] = T.max(max_exp2 - 6, 0)
            out_sf_exp_ub[0, 3] = T.max(max_exp3 - 6, 0)
            for line_idx in T.unroll(4):
                out_sf_bits = T.alloc_var("int32", init=0)
                out_sf_bits = out_sf_exp_ub[0, line_idx] << 23
                out_sf_fp32_ub[0, line_idx] = T.reinterpret("float32", out_sf_bits)
                store_output_sf(out_sf, out_sf_fp32_ub, out_sf_exp_ub, 0, line_idx, out_sf_m, out_sf_k + word_group * 4 + line_idx)
            T.tile.add(relative_exp0_ub, e0_ub, 127 - out_sf_exp_ub[0, 0])
            T.tile.add(relative_exp1_ub, e1_ub, 127 - out_sf_exp_ub[0, 1])
            T.tile.add(relative_exp2_ub, e2_ub, 127 - out_sf_exp_ub[0, 2])
            T.tile.add(relative_exp3_ub, e3_ub, 127 - out_sf_exp_ub[0, 3])
            T.pipe_barrier("v")
            T.tile.bitwise_lshift(relative_bits0_ub, relative_exp0_ub, 23)
            T.tile.bitwise_lshift(relative_bits1_ub, relative_exp1_ub, 23)
            T.tile.bitwise_lshift(relative_bits2_ub, relative_exp2_ub, 23)
            T.tile.bitwise_lshift(relative_bits3_ub, relative_exp3_ub, 23)
            T.pipe_barrier("v")
            T.tile.broadcast(relative_sf_tile0_ub, relative_sf0_view_ub, axis=1)
            T.tile.broadcast(relative_sf_tile1_ub, relative_sf1_view_ub, axis=1)
            T.tile.broadcast(relative_sf_tile2_ub, relative_sf2_view_ub, axis=1)
            T.tile.broadcast(relative_sf_tile3_ub, relative_sf3_view_ub, axis=1)
            T.set_flag("mte3", "v", 1)
            for pair_idx in T.unroll(2):
                line0 = pair_idx * 2
                line1 = line0 + 1
                col_base0 = col_offset + line0 * in_sf_block_k
                col_base1 = col_offset + line1 * in_sf_block_k
                if pair_idx == 0:
                    T.wait_flag("mte2", "v", 0)
                else:
                    T.wait_flag("v", "mte2", 0)
                    T.copy(x[row_offset : row_offset + 32, col_base0 : col_base0 + in_sf_block_k], x_in0_ub)
                    T.copy(x[row_offset : row_offset + 32, col_base1 : col_base1 + in_sf_block_k], x_in1_ub)
                    T.set_flag("mte2", "v", 0)
                    T.wait_flag("mte2", "v", 0)
                T.wait_flag("mte3", "v", 1)
                T.tile.cast(x_out0_ub, x_in0_ub, mode="CAST_NONE", count=tile_elem_count)
                T.tile.cast(x_out1_ub, x_in1_ub, mode="CAST_NONE", count=tile_elem_count)
                if pair_idx == 0:
                    T.tile.mul(x_out0_ub, x_out0_ub, relative_sf_tile0_ub)
                    T.tile.mul(x_out1_ub, x_out1_ub, relative_sf_tile1_ub)
                else:
                    T.tile.mul(x_out0_ub, x_out0_ub, relative_sf_tile2_ub)
                    T.tile.mul(x_out1_ub, x_out1_ub, relative_sf_tile3_ub)
                T.set_flag("v", "mte2", 0)
                T.set_flag("v", "mte3", 1)
                T.wait_flag("v", "mte3", 1)
                T.copy(x_out0_ub, out[row_offset : row_offset + 32, col_base0 : col_base0 + in_sf_block_k])
                T.copy(x_out1_ub, out[row_offset : row_offset + 32, col_base1 : col_base1 + in_sf_block_k])
                T.set_flag("mte3", "v", 1)
            T.wait_flag("v", "mte2", 0)
            T.wait_flag("mte3", "v", 1)

    @T.macro
    def apply_max4_fast_path(x_sf, x, out, out_sf, x_sf_load_ub, x_sf_exp_ub, sf_m, sf_k, out_sf_m, out_sf_k, row_offset, col_offset):
        tile_elem_count = fast_data_tile_m * in_sf_block_k
        x_in0_slot0_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in1_slot0_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in0_slot1_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in1_slot1_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in0_slot2_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in1_slot2_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in0_slot3_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_in1_slot3_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), INPUT_DTYPE)
        x_out00_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out01_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out10_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out11_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out20_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out21_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out30_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        x_out31_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), OUTPUT_DTYPE)
        xsf_row_offset_i32_ub = T.alloc_ub((fast_data_tile_m,), "int32")
        xsf_row_offset_u32_ub = T.alloc_ub((fast_data_tile_m,), "uint32")
        e0_ub = T.alloc_ub((fast_data_tile_m,), "int32")
        e1_ub = T.alloc_ub((fast_data_tile_m,), "int32")
        e2_ub = T.alloc_ub((fast_data_tile_m,), "int32")
        e3_ub = T.alloc_ub((fast_data_tile_m,), "int32")
        packed_e_word_ub = T.alloc_ub((fast_data_tile_m,), "int32")
        x_sf_word_ub = T.alloc_ub((num_in_sf_per_block_k // 4, fast_num_in_sf_per_data_tile_m), "int32")
        max01_ub = T.alloc_ub((fast_data_tile_m,), "int32")
        max23_ub = T.alloc_ub((fast_data_tile_m,), "int32")
        out_exp_ub = T.alloc_ub((fast_data_tile_m,), "int32")
        relative_exp0_ub = T.alloc_ub((fast_data_tile_m,), "int32")
        relative_exp1_ub = T.alloc_ub((fast_data_tile_m,), "int32")
        relative_bits0_ub = T.alloc_ub((fast_data_tile_m,), "int32")
        relative_bits1_ub = T.alloc_ub((fast_data_tile_m,), "int32")
        relative_sf0_view_ub = T.alloc_ub((fast_data_tile_m, 1), "float32")
        relative_sf1_view_ub = T.alloc_ub((fast_data_tile_m, 1), "float32")
        relative_sf_tile0_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), "float32")
        relative_sf_tile1_ub = T.alloc_ub((fast_data_tile_m, in_sf_block_k), "float32")
        T.reinterpretcast(xsf_row_offset_u32_ub, xsf_row_offset_i32_ub, "uint32_t")
        T.reinterpretcast(relative_sf0_view_ub, relative_bits0_ub, "float")
        T.reinterpretcast(relative_sf1_view_ub, relative_bits1_ub, "float")
        if in_config.use_packed_ue8m0:
            T.reinterpretcast(x_sf_word_ub, x_sf_load_ub, "int32_t")
            load_input_sf_block_fast(x_sf_load_ub, x_sf, sf_m, sf_k)
            T.pipe_barrier("all")
        else:
            load_and_decode_input_sf_fast(x_sf_exp_ub, x_sf_load_ub, x_sf, sf_m, sf_k)
        for input_slot in T.unroll(4):
            T.set_flag("v", "mte2", input_slot)
        for preload_stage in T.unroll(fast_num_initial_prefetch_stages):
            preload_slot = preload_stage
            preload_block_idx = preload_stage // 2
            preload_pair_idx = preload_stage % 2
            preload_line0_idx = preload_pair_idx * 2
            preload_line1_idx = preload_line0_idx + 1
            preload_col_base = col_offset + preload_block_idx * out_sf_block_k
            T.wait_flag("v", "mte2", preload_slot)
            if preload_slot == 0:
                T.copy(
                    x[
                        row_offset : row_offset + fast_data_tile_m,
                        preload_col_base + preload_line0_idx * in_sf_block_k : preload_col_base + (preload_line0_idx + 1) * in_sf_block_k,
                    ],
                    x_in0_slot0_ub,
                )
                T.copy(
                    x[
                        row_offset : row_offset + fast_data_tile_m,
                        preload_col_base + preload_line1_idx * in_sf_block_k : preload_col_base + (preload_line1_idx + 1) * in_sf_block_k,
                    ],
                    x_in1_slot0_ub,
                )
            elif preload_slot == 1:
                T.copy(
                    x[
                        row_offset : row_offset + fast_data_tile_m,
                        preload_col_base + preload_line0_idx * in_sf_block_k : preload_col_base + (preload_line0_idx + 1) * in_sf_block_k,
                    ],
                    x_in0_slot1_ub,
                )
                T.copy(
                    x[
                        row_offset : row_offset + fast_data_tile_m,
                        preload_col_base + preload_line1_idx * in_sf_block_k : preload_col_base + (preload_line1_idx + 1) * in_sf_block_k,
                    ],
                    x_in1_slot1_ub,
                )
            elif preload_slot == 2:
                T.copy(
                    x[
                        row_offset : row_offset + fast_data_tile_m,
                        preload_col_base + preload_line0_idx * in_sf_block_k : preload_col_base + (preload_line0_idx + 1) * in_sf_block_k,
                    ],
                    x_in0_slot2_ub,
                )
                T.copy(
                    x[
                        row_offset : row_offset + fast_data_tile_m,
                        preload_col_base + preload_line1_idx * in_sf_block_k : preload_col_base + (preload_line1_idx + 1) * in_sf_block_k,
                    ],
                    x_in1_slot2_ub,
                )
            else:
                T.copy(
                    x[
                        row_offset : row_offset + fast_data_tile_m,
                        preload_col_base + preload_line0_idx * in_sf_block_k : preload_col_base + (preload_line0_idx + 1) * in_sf_block_k,
                    ],
                    x_in0_slot3_ub,
                )
                T.copy(
                    x[
                        row_offset : row_offset + fast_data_tile_m,
                        preload_col_base + preload_line1_idx * in_sf_block_k : preload_col_base + (preload_line1_idx + 1) * in_sf_block_k,
                    ],
                    x_in1_slot3_ub,
                )
            T.set_flag("mte2", "v", preload_slot)
        T.pipe_barrier("v")
        for block_idx in T.unroll(num_out_sf_per_block_k):
            sf_k_base = block_idx * num_in_sf_per_out_sf_k
            if in_config.use_packed_ue8m0:
                T.tile.arith_progression(xsf_row_offset_i32_ub, block_idx * fast_num_in_sf_per_data_tile_m * 4, 4, fast_data_tile_m)
                T.tile.gather(packed_e_word_ub, x_sf_word_ub, xsf_row_offset_u32_ub, 0)
                for i in T.serial(fast_num_in_sf_per_data_tile_m):
                    packed_word = T.alloc_var("int32", init=0)
                    packed_word = packed_e_word_ub[i]
                    e0_ub[i] = T.Cast("int32", T.Cast("uint8", packed_word))
                    e1_ub[i] = T.Cast("int32", T.Cast("uint8", packed_word >> 8))
                    e2_ub[i] = T.Cast("int32", T.Cast("uint8", packed_word >> 16))
                    e3_ub[i] = T.Cast("int32", T.Cast("uint8", packed_word >> 24))
            else:
                T.tile.arith_progression(xsf_row_offset_i32_ub, (sf_k_base + 0) * 4, num_in_sf_per_block_k * 4, fast_data_tile_m)
                T.tile.gather(e0_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 0)
                T.tile.arith_progression(xsf_row_offset_i32_ub, (sf_k_base + 1) * 4, num_in_sf_per_block_k * 4, fast_data_tile_m)
                T.tile.gather(e1_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 0)
                T.tile.arith_progression(xsf_row_offset_i32_ub, (sf_k_base + 2) * 4, num_in_sf_per_block_k * 4, fast_data_tile_m)
                T.tile.gather(e2_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 0)
                T.tile.arith_progression(xsf_row_offset_i32_ub, (sf_k_base + 3) * 4, num_in_sf_per_block_k * 4, fast_data_tile_m)
                T.tile.gather(e3_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 0)
            T.pipe_barrier("v")
            T.tile.max(max01_ub, e0_ub, e1_ub)
            T.tile.max(max23_ub, e2_ub, e3_ub)
            T.tile.max(out_exp_ub, max01_ub, max23_ub)
            T.tile.add(out_exp_ub, out_exp_ub, -6)
            T.tile.max(out_exp_ub, out_exp_ub, 0)
            T.pipe_barrier("v")
            for pair_idx in T.unroll(2):
                stage = block_idx * 2 + pair_idx
                input_slot = stage % 4
                output_slot = stage % 4
                line0_idx = pair_idx * 2
                line1_idx = line0_idx + 1
                if stage + 4 < fast_num_pipeline_pairs:
                    future_stage = stage + 4
                    future_block_idx = future_stage // 2
                    future_pair_idx = future_stage % 2
                    future_line0_idx = future_pair_idx * 2
                    future_line1_idx = future_line0_idx + 1
                    future_col_base = col_offset + future_block_idx * out_sf_block_k
                    T.wait_flag("v", "mte2", input_slot)
                    if input_slot == 0:
                        T.copy(
                            x[
                                row_offset : row_offset + fast_data_tile_m,
                                future_col_base + future_line0_idx * in_sf_block_k : future_col_base
                                + (future_line0_idx + 1) * in_sf_block_k,
                            ],
                            x_in0_slot0_ub,
                        )
                        T.copy(
                            x[
                                row_offset : row_offset + fast_data_tile_m,
                                future_col_base + future_line1_idx * in_sf_block_k : future_col_base
                                + (future_line1_idx + 1) * in_sf_block_k,
                            ],
                            x_in1_slot0_ub,
                        )
                    elif input_slot == 1:
                        T.copy(
                            x[
                                row_offset : row_offset + fast_data_tile_m,
                                future_col_base + future_line0_idx * in_sf_block_k : future_col_base
                                + (future_line0_idx + 1) * in_sf_block_k,
                            ],
                            x_in0_slot1_ub,
                        )
                        T.copy(
                            x[
                                row_offset : row_offset + fast_data_tile_m,
                                future_col_base + future_line1_idx * in_sf_block_k : future_col_base
                                + (future_line1_idx + 1) * in_sf_block_k,
                            ],
                            x_in1_slot1_ub,
                        )
                    elif input_slot == 2:
                        T.copy(
                            x[
                                row_offset : row_offset + fast_data_tile_m,
                                future_col_base + future_line0_idx * in_sf_block_k : future_col_base
                                + (future_line0_idx + 1) * in_sf_block_k,
                            ],
                            x_in0_slot2_ub,
                        )
                        T.copy(
                            x[
                                row_offset : row_offset + fast_data_tile_m,
                                future_col_base + future_line1_idx * in_sf_block_k : future_col_base
                                + (future_line1_idx + 1) * in_sf_block_k,
                            ],
                            x_in1_slot2_ub,
                        )
                    else:
                        T.copy(
                            x[
                                row_offset : row_offset + fast_data_tile_m,
                                future_col_base + future_line0_idx * in_sf_block_k : future_col_base
                                + (future_line0_idx + 1) * in_sf_block_k,
                            ],
                            x_in0_slot3_ub,
                        )
                        T.copy(
                            x[
                                row_offset : row_offset + fast_data_tile_m,
                                future_col_base + future_line1_idx * in_sf_block_k : future_col_base
                                + (future_line1_idx + 1) * in_sf_block_k,
                            ],
                            x_in1_slot3_ub,
                        )
                    T.set_flag("mte2", "v", input_slot)
                if stage >= 4:
                    T.wait_flag("mte3", "v", output_slot)
                if pair_idx == 0:
                    T.tile.sub(relative_exp0_ub, e0_ub, out_exp_ub)
                    T.tile.sub(relative_exp1_ub, e1_ub, out_exp_ub)
                else:
                    T.tile.sub(relative_exp0_ub, e2_ub, out_exp_ub)
                    T.tile.sub(relative_exp1_ub, e3_ub, out_exp_ub)
                T.tile.add(relative_exp0_ub, relative_exp0_ub, 127)
                T.tile.add(relative_exp1_ub, relative_exp1_ub, 127)
                T.tile.bitwise_lshift(relative_bits0_ub, relative_exp0_ub, 23)
                T.tile.bitwise_lshift(relative_bits1_ub, relative_exp1_ub, 23)
                T.pipe_barrier("v")
                T.tile.broadcast(relative_sf_tile0_ub, relative_sf0_view_ub, axis=1)
                T.tile.broadcast(relative_sf_tile1_ub, relative_sf1_view_ub, axis=1)
                T.wait_flag("mte2", "v", input_slot)
                if output_slot == 0:
                    if input_slot == 0:
                        T.tile.cast(x_out00_ub, x_in0_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out01_ub, x_in1_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 1:
                        T.tile.cast(x_out00_ub, x_in0_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out01_ub, x_in1_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 2:
                        T.tile.cast(x_out00_ub, x_in0_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out01_ub, x_in1_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                    else:
                        T.tile.cast(x_out00_ub, x_in0_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out01_ub, x_in1_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                    if stage + 4 < fast_num_pipeline_pairs:
                        T.set_flag("v", "mte2", input_slot)
                    T.tile.mul(x_out00_ub, x_out00_ub, relative_sf_tile0_ub)
                    T.tile.mul(x_out01_ub, x_out01_ub, relative_sf_tile1_ub)
                elif output_slot == 1:
                    if input_slot == 0:
                        T.tile.cast(x_out10_ub, x_in0_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out11_ub, x_in1_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 1:
                        T.tile.cast(x_out10_ub, x_in0_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out11_ub, x_in1_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 2:
                        T.tile.cast(x_out10_ub, x_in0_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out11_ub, x_in1_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                    else:
                        T.tile.cast(x_out10_ub, x_in0_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out11_ub, x_in1_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                    if stage + 4 < fast_num_pipeline_pairs:
                        T.set_flag("v", "mte2", input_slot)
                    T.tile.mul(x_out10_ub, x_out10_ub, relative_sf_tile0_ub)
                    T.tile.mul(x_out11_ub, x_out11_ub, relative_sf_tile1_ub)
                elif output_slot == 2:
                    if input_slot == 0:
                        T.tile.cast(x_out20_ub, x_in0_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out21_ub, x_in1_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 1:
                        T.tile.cast(x_out20_ub, x_in0_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out21_ub, x_in1_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 2:
                        T.tile.cast(x_out20_ub, x_in0_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out21_ub, x_in1_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                    else:
                        T.tile.cast(x_out20_ub, x_in0_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out21_ub, x_in1_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                    if stage + 4 < fast_num_pipeline_pairs:
                        T.set_flag("v", "mte2", input_slot)
                    T.tile.mul(x_out20_ub, x_out20_ub, relative_sf_tile0_ub)
                    T.tile.mul(x_out21_ub, x_out21_ub, relative_sf_tile1_ub)
                else:
                    if input_slot == 0:
                        T.tile.cast(x_out30_ub, x_in0_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out31_ub, x_in1_slot0_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 1:
                        T.tile.cast(x_out30_ub, x_in0_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out31_ub, x_in1_slot1_ub, mode="CAST_NONE", count=tile_elem_count)
                    elif input_slot == 2:
                        T.tile.cast(x_out30_ub, x_in0_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out31_ub, x_in1_slot2_ub, mode="CAST_NONE", count=tile_elem_count)
                    else:
                        T.tile.cast(x_out30_ub, x_in0_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                        T.tile.cast(x_out31_ub, x_in1_slot3_ub, mode="CAST_NONE", count=tile_elem_count)
                    if stage + 4 < fast_num_pipeline_pairs:
                        T.set_flag("v", "mte2", input_slot)
                    T.tile.mul(x_out30_ub, x_out30_ub, relative_sf_tile0_ub)
                    T.tile.mul(x_out31_ub, x_out31_ub, relative_sf_tile1_ub)
                T.set_flag("v", "mte3", output_slot)
                T.wait_flag("v", "mte3", output_slot)
                current_col_base = col_offset + block_idx * out_sf_block_k
                if output_slot == 0:
                    T.copy(
                        x_out00_ub,
                        out[
                            row_offset : row_offset + fast_data_tile_m,
                            current_col_base + line0_idx * in_sf_block_k : current_col_base + (line0_idx + 1) * in_sf_block_k,
                        ],
                    )
                    T.copy(
                        x_out01_ub,
                        out[
                            row_offset : row_offset + fast_data_tile_m,
                            current_col_base + line1_idx * in_sf_block_k : current_col_base + (line1_idx + 1) * in_sf_block_k,
                        ],
                    )
                elif output_slot == 1:
                    T.copy(
                        x_out10_ub,
                        out[
                            row_offset : row_offset + fast_data_tile_m,
                            current_col_base + line0_idx * in_sf_block_k : current_col_base + (line0_idx + 1) * in_sf_block_k,
                        ],
                    )
                    T.copy(
                        x_out11_ub,
                        out[
                            row_offset : row_offset + fast_data_tile_m,
                            current_col_base + line1_idx * in_sf_block_k : current_col_base + (line1_idx + 1) * in_sf_block_k,
                        ],
                    )
                elif output_slot == 2:
                    T.copy(
                        x_out20_ub,
                        out[
                            row_offset : row_offset + fast_data_tile_m,
                            current_col_base + line0_idx * in_sf_block_k : current_col_base + (line0_idx + 1) * in_sf_block_k,
                        ],
                    )
                    T.copy(
                        x_out21_ub,
                        out[
                            row_offset : row_offset + fast_data_tile_m,
                            current_col_base + line1_idx * in_sf_block_k : current_col_base + (line1_idx + 1) * in_sf_block_k,
                        ],
                    )
                else:
                    T.copy(
                        x_out30_ub,
                        out[
                            row_offset : row_offset + fast_data_tile_m,
                            current_col_base + line0_idx * in_sf_block_k : current_col_base + (line0_idx + 1) * in_sf_block_k,
                        ],
                    )
                    T.copy(
                        x_out31_ub,
                        out[
                            row_offset : row_offset + fast_data_tile_m,
                            current_col_base + line1_idx * in_sf_block_k : current_col_base + (line1_idx + 1) * in_sf_block_k,
                        ],
                    )
                if stage + 4 < fast_num_pipeline_pairs:
                    T.set_flag("mte3", "v", output_slot)
        T.pipe_barrier("mte3")

    @T.macro
    def apply_32x32_tile_packed_fast_path(x_sf, x, out, out_sf, pid_token, pid_hidden, data_m, out_sf_m, out_sf_k, row_offset, col_offset):
        tile_elem_count = 32 * in_sf_block_k
        x_sf_load_ub = T.alloc_ub((packed_sf_tile_elems,), "float32")
        x_sf_bits_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
        x_sf_exp_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
        xsf_row_offset_i32_ub = T.alloc_ub((32,), "int32")
        xsf_row_offset_u32_ub = T.alloc_ub((32,), "uint32")
        e0_ub = T.alloc_ub((1, 32), "int32")
        e1_ub = T.alloc_ub((1, 32), "int32")
        e2_ub = T.alloc_ub((1, 32), "int32")
        e3_ub = T.alloc_ub((1, 32), "int32")
        e0_fp32_ub = T.alloc_ub((1, 32), "float32")
        e1_fp32_ub = T.alloc_ub((1, 32), "float32")
        e2_fp32_ub = T.alloc_ub((1, 32), "float32")
        e3_fp32_ub = T.alloc_ub((1, 32), "float32")
        max_exp0_fp32_ub = T.alloc_ub((1, 1), "float32")
        max_exp1_fp32_ub = T.alloc_ub((1, 1), "float32")
        max_exp2_fp32_ub = T.alloc_ub((1, 1), "float32")
        max_exp3_fp32_ub = T.alloc_ub((1, 1), "float32")
        max_exp0_ub = T.alloc_ub((1, 1), "int32")
        max_exp1_ub = T.alloc_ub((1, 1), "int32")
        max_exp2_ub = T.alloc_ub((1, 1), "int32")
        max_exp3_ub = T.alloc_ub((1, 1), "int32")
        out_sf_exp_ub = T.alloc_ub((1, num_out_sf_per_block_k), "int32")
        out_sf_fp32_ub = T.alloc_ub((1, num_out_sf_per_block_k), "float32")
        relative_exp0_ub = T.alloc_ub((32,), "int32")
        relative_exp1_ub = T.alloc_ub((32,), "int32")
        relative_exp2_ub = T.alloc_ub((32,), "int32")
        relative_exp3_ub = T.alloc_ub((32,), "int32")
        relative_bits0_ub = T.alloc_ub((32,), "int32")
        relative_bits1_ub = T.alloc_ub((32,), "int32")
        relative_bits2_ub = T.alloc_ub((32,), "int32")
        relative_bits3_ub = T.alloc_ub((32,), "int32")
        relative_sf0_view_ub = T.alloc_ub((32, 1), "float32")
        relative_sf1_view_ub = T.alloc_ub((32, 1), "float32")
        relative_sf2_view_ub = T.alloc_ub((32, 1), "float32")
        relative_sf3_view_ub = T.alloc_ub((32, 1), "float32")
        relative_sf_tile0_ub = T.alloc_ub((32, in_sf_block_k), "float32")
        relative_sf_tile1_ub = T.alloc_ub((32, in_sf_block_k), "float32")
        relative_sf_tile2_ub = T.alloc_ub((32, in_sf_block_k), "float32")
        relative_sf_tile3_ub = T.alloc_ub((32, in_sf_block_k), "float32")
        x_in0_ub = T.alloc_ub((32, in_sf_block_k), INPUT_DTYPE)
        x_in1_ub = T.alloc_ub((32, in_sf_block_k), INPUT_DTYPE)
        x_out0_ub = T.alloc_ub((32, in_sf_block_k), OUTPUT_DTYPE)
        x_out1_ub = T.alloc_ub((32, in_sf_block_k), OUTPUT_DTYPE)
        T.reinterpretcast(x_sf_bits_ub, x_sf_load_ub, "int32_t")
        T.reinterpretcast(xsf_row_offset_u32_ub, xsf_row_offset_i32_ub, "uint32_t")
        T.reinterpretcast(relative_sf0_view_ub, relative_bits0_ub, "float")
        T.reinterpretcast(relative_sf1_view_ub, relative_bits1_ub, "float")
        T.reinterpretcast(relative_sf2_view_ub, relative_bits2_ub, "float")
        T.reinterpretcast(relative_sf3_view_ub, relative_bits3_ub, "float")
        T.copy(x_sf[pid_token, pid_hidden, data_m, 0:packed_sf_tile_elems], x_sf_load_ub)
        T.set_flag("mte2", "v", 2)
        T.wait_flag("mte2", "v", 2)
        T.copy(x[row_offset : row_offset + 32, col_offset : col_offset + in_sf_block_k], x_in0_ub)
        T.copy(x[row_offset : row_offset + 32, col_offset + in_sf_block_k : col_offset + 2 * in_sf_block_k], x_in1_ub)
        T.set_flag("mte2", "v", 0)
        T.tile.bitwise_rshift(x_sf_exp_ub, x_sf_bits_ub, 23)
        T.pipe_barrier("v")
        T.tile.arith_progression(xsf_row_offset_i32_ub, 0, num_in_sf_per_block_k * 4, 32)
        T.pipe_barrier("v")
        T.tile.gather(e0_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 0)
        T.tile.gather(e1_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 4)
        T.tile.gather(e2_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 8)
        T.tile.gather(e3_ub, x_sf_exp_ub, xsf_row_offset_u32_ub, 12)
        T.pipe_barrier("v")
        T.tile.cast(e0_fp32_ub, e0_ub, mode="CAST_NONE", count=32)
        T.tile.cast(e1_fp32_ub, e1_ub, mode="CAST_NONE", count=32)
        T.tile.cast(e2_fp32_ub, e2_ub, mode="CAST_NONE", count=32)
        T.tile.cast(e3_fp32_ub, e3_ub, mode="CAST_NONE", count=32)
        T.pipe_barrier("v")
        T.reduce_max(e0_fp32_ub, max_exp0_fp32_ub, dim=-1, clear=True, real_shape=[1, 32])
        T.reduce_max(e1_fp32_ub, max_exp1_fp32_ub, dim=-1, clear=True, real_shape=[1, 32])
        T.reduce_max(e2_fp32_ub, max_exp2_fp32_ub, dim=-1, clear=True, real_shape=[1, 32])
        T.reduce_max(e3_fp32_ub, max_exp3_fp32_ub, dim=-1, clear=True, real_shape=[1, 32])
        T.tile.add(max_exp0_fp32_ub, max_exp0_fp32_ub, -6.0)
        T.tile.add(max_exp1_fp32_ub, max_exp1_fp32_ub, -6.0)
        T.tile.add(max_exp2_fp32_ub, max_exp2_fp32_ub, -6.0)
        T.tile.add(max_exp3_fp32_ub, max_exp3_fp32_ub, -6.0)
        T.tile.max(max_exp0_fp32_ub, max_exp0_fp32_ub, 0.0)
        T.tile.max(max_exp1_fp32_ub, max_exp1_fp32_ub, 0.0)
        T.tile.max(max_exp2_fp32_ub, max_exp2_fp32_ub, 0.0)
        T.tile.max(max_exp3_fp32_ub, max_exp3_fp32_ub, 0.0)
        T.tile.cast(max_exp0_ub, max_exp0_fp32_ub, mode="CAST_ROUND", count=1)
        T.tile.cast(max_exp1_ub, max_exp1_fp32_ub, mode="CAST_ROUND", count=1)
        T.tile.cast(max_exp2_ub, max_exp2_fp32_ub, mode="CAST_ROUND", count=1)
        T.tile.cast(max_exp3_ub, max_exp3_fp32_ub, mode="CAST_ROUND", count=1)
        T.pipe_barrier("v")
        out_sf_exp_ub[0, 0] = max_exp0_ub[0, 0]
        out_sf_exp_ub[0, 1] = max_exp1_ub[0, 0]
        out_sf_exp_ub[0, 2] = max_exp2_ub[0, 0]
        out_sf_exp_ub[0, 3] = max_exp3_ub[0, 0]
        for block_idx in T.unroll(num_out_sf_per_block_k):
            out_sf_bits = T.alloc_var("int32", init=0)
            out_sf_bits = out_sf_exp_ub[0, block_idx] << 23
            out_sf_fp32_ub[0, block_idx] = T.reinterpret("float32", out_sf_bits)
            store_output_sf(out_sf, out_sf_fp32_ub, out_sf_exp_ub, 0, block_idx, out_sf_m, out_sf_k + block_idx)
        T.tile.add(relative_exp0_ub, e0_ub, 127 - max_exp0_ub[0, 0])
        T.tile.add(relative_exp1_ub, e1_ub, 127 - max_exp1_ub[0, 0])
        T.tile.add(relative_exp2_ub, e2_ub, 127 - max_exp2_ub[0, 0])
        T.tile.add(relative_exp3_ub, e3_ub, 127 - max_exp3_ub[0, 0])
        T.pipe_barrier("v")
        T.tile.bitwise_lshift(relative_bits0_ub, relative_exp0_ub, 23)
        T.tile.bitwise_lshift(relative_bits1_ub, relative_exp1_ub, 23)
        T.tile.bitwise_lshift(relative_bits2_ub, relative_exp2_ub, 23)
        T.tile.bitwise_lshift(relative_bits3_ub, relative_exp3_ub, 23)
        T.pipe_barrier("v")
        T.tile.broadcast(relative_sf_tile0_ub, relative_sf0_view_ub, axis=1)
        T.tile.broadcast(relative_sf_tile1_ub, relative_sf1_view_ub, axis=1)
        T.tile.broadcast(relative_sf_tile2_ub, relative_sf2_view_ub, axis=1)
        T.tile.broadcast(relative_sf_tile3_ub, relative_sf3_view_ub, axis=1)
        T.set_flag("mte3", "v", 1)
        for pair_idx in T.unroll(2):
            line0 = pair_idx * 2
            line1 = line0 + 1
            col_base0 = col_offset + line0 * in_sf_block_k
            col_base1 = col_offset + line1 * in_sf_block_k
            if pair_idx == 0:
                T.wait_flag("mte2", "v", 0)
            else:
                T.wait_flag("v", "mte2", 0)
                T.copy(x[row_offset : row_offset + 32, col_base0 : col_base0 + in_sf_block_k], x_in0_ub)
                T.copy(x[row_offset : row_offset + 32, col_base1 : col_base1 + in_sf_block_k], x_in1_ub)
                T.set_flag("mte2", "v", 0)
                T.wait_flag("mte2", "v", 0)
            T.wait_flag("mte3", "v", 1)
            T.tile.cast(x_out0_ub, x_in0_ub, mode="CAST_NONE", count=tile_elem_count)
            T.tile.cast(x_out1_ub, x_in1_ub, mode="CAST_NONE", count=tile_elem_count)
            if pair_idx == 0:
                T.tile.mul(x_out0_ub, x_out0_ub, relative_sf_tile0_ub)
                T.tile.mul(x_out1_ub, x_out1_ub, relative_sf_tile1_ub)
            else:
                T.tile.mul(x_out0_ub, x_out0_ub, relative_sf_tile2_ub)
                T.tile.mul(x_out1_ub, x_out1_ub, relative_sf_tile3_ub)
            T.set_flag("v", "mte2", 0)
            T.set_flag("v", "mte3", 1)
            T.wait_flag("v", "mte3", 1)
            T.copy(x_out0_ub, out[row_offset : row_offset + 32, col_base0 : col_base0 + in_sf_block_k])
            T.copy(x_out1_ub, out[row_offset : row_offset + 32, col_base1 : col_base1 + in_sf_block_k])
            T.set_flag("mte3", "v", 1)
        T.wait_flag("v", "mte2", 0)
        T.wait_flag("mte3", "v", 1)

    @T.macro
    def apply_128x128_plain_fast_path(
        x_sf,
        x,
        out,
        out_sf,
        pid_token,
        pid_hidden,
        sf_row_offset,
        sf_col_offset,
        out_sf_row_offset,
        out_sf_col_offset,
        row_offset,
        col_offset,
        vid,
    ):
        row_offset = pid_token * block_m
        reduce_offset_i32_ub = T.alloc_ub((32,), "int32")
        reduce_offset_u32_ub = T.alloc_ub((32,), "uint32")
        reduce_e0_ub = T.alloc_ub((32,), "int32")
        reduce_e1_ub = T.alloc_ub((32,), "int32")
        reduce_e2_ub = T.alloc_ub((32,), "int32")
        reduce_e3_ub = T.alloc_ub((32,), "int32")
        reduce_max01_ub = T.alloc_ub((32,), "int32")
        reduce_max23_ub = T.alloc_ub((32,), "int32")
        reduce_tile_max_ub = T.alloc_ub((32,), "int32")
        reduce_block_max_ub = T.alloc_ub((32,), "int32")
        T.reinterpretcast(reduce_offset_u32_ub, reduce_offset_i32_ub, "uint32_t")
        T.tile.arith_progression(reduce_offset_i32_ub, 0, 16, 32)
        T.tile.fill(reduce_block_max_ub, 0)
        if input_sf_is_tile_packed:
            packed_cached_exp0_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
            packed_cached_exp1_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
            packed_cached_exp2_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
            packed_cached_exp3_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
            packed_load0_ub = T.alloc_ub((packed_sf_tile_elems,), "float32")
            packed_load1_ub = T.alloc_ub((packed_sf_tile_elems,), "float32")
            packed_load2_ub = T.alloc_ub((packed_sf_tile_elems,), "float32")
            packed_load3_ub = T.alloc_ub((packed_sf_tile_elems,), "float32")
            packed_bits0_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
            packed_bits1_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
            packed_bits2_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
            packed_bits3_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
            T.copy(x_sf[pid_token, pid_hidden, 0, 0:packed_sf_tile_elems], packed_load0_ub)
            T.set_flag("mte2", "v", 2)
            T.copy(x_sf[pid_token, pid_hidden, 1, 0:packed_sf_tile_elems], packed_load1_ub)
            T.set_flag("mte2", "v", 3)
            T.wait_flag("mte2", "v", 2)
            T.reinterpretcast(packed_bits0_ub, packed_load0_ub, "int32_t")
            T.tile.bitwise_rshift(packed_cached_exp0_ub, packed_bits0_ub, 23)
            T.pipe_barrier("v")
            reduce_128x128_exp_rows(
                packed_cached_exp0_ub,
                reduce_offset_u32_ub,
                reduce_e0_ub,
                reduce_e1_ub,
                reduce_e2_ub,
                reduce_e3_ub,
                reduce_max01_ub,
                reduce_max23_ub,
                reduce_tile_max_ub,
                reduce_block_max_ub,
                0,
            )
            T.wait_flag("mte2", "v", 3)
            T.reinterpretcast(packed_bits1_ub, packed_load1_ub, "int32_t")
            T.tile.bitwise_rshift(packed_cached_exp1_ub, packed_bits1_ub, 23)
            T.pipe_barrier("v")
            reduce_128x128_exp_rows(
                packed_cached_exp1_ub,
                reduce_offset_u32_ub,
                reduce_e0_ub,
                reduce_e1_ub,
                reduce_e2_ub,
                reduce_e3_ub,
                reduce_max01_ub,
                reduce_max23_ub,
                reduce_tile_max_ub,
                reduce_block_max_ub,
                0,
            )
            T.copy(x_sf[pid_token, pid_hidden, 2, 0:packed_sf_tile_elems], packed_load2_ub)
            T.set_flag("mte2", "v", 2)
            T.copy(x_sf[pid_token, pid_hidden, 3, 0:packed_sf_tile_elems], packed_load3_ub)
            T.set_flag("mte2", "v", 3)
            T.wait_flag("mte2", "v", 2)
            T.reinterpretcast(packed_bits2_ub, packed_load2_ub, "int32_t")
            T.tile.bitwise_rshift(packed_cached_exp2_ub, packed_bits2_ub, 23)
            T.pipe_barrier("v")
            reduce_128x128_exp_rows(
                packed_cached_exp2_ub,
                reduce_offset_u32_ub,
                reduce_e0_ub,
                reduce_e1_ub,
                reduce_e2_ub,
                reduce_e3_ub,
                reduce_max01_ub,
                reduce_max23_ub,
                reduce_tile_max_ub,
                reduce_block_max_ub,
                0,
            )
            T.wait_flag("mte2", "v", 3)
            T.reinterpretcast(packed_bits3_ub, packed_load3_ub, "int32_t")
            T.tile.bitwise_rshift(packed_cached_exp3_ub, packed_bits3_ub, 23)
            T.pipe_barrier("v")
            reduce_128x128_exp_rows(
                packed_cached_exp3_ub,
                reduce_offset_u32_ub,
                reduce_e0_ub,
                reduce_e1_ub,
                reduce_e2_ub,
                reduce_e3_ub,
                reduce_max01_ub,
                reduce_max23_ub,
                reduce_tile_max_ub,
                reduce_block_max_ub,
                0,
            )
        else:
            plain_cached_exp01_ub = T.alloc_ub((64, 4), "int32")
            plain_cached_exp23_ub = T.alloc_ub((64, 4), "int32")
            plain_load64_ub0 = T.alloc_ub((64, 4), "float32")
            plain_load64_ub1 = T.alloc_ub((64, 4), "float32")
            plain_bits64_ub0 = T.alloc_ub((64, 4), "int32")
            plain_bits64_ub1 = T.alloc_ub((64, 4), "int32")
            T.copy(
                x_sf[sf_row_offset : sf_row_offset + 64, sf_col_offset : sf_col_offset + 4],
                plain_load64_ub0,
            )
            T.set_flag("mte2", "v", 2)
            T.copy(
                x_sf[sf_row_offset + 64 : sf_row_offset + 128, sf_col_offset : sf_col_offset + 4],
                plain_load64_ub1,
            )
            T.set_flag("mte2", "v", 3)
            T.wait_flag("mte2", "v", 2)
            decode_reduce_128_plain_pair(
                plain_cached_exp01_ub,
                plain_load64_ub0,
                plain_bits64_ub0,
                reduce_offset_u32_ub,
                reduce_e0_ub,
                reduce_e1_ub,
                reduce_e2_ub,
                reduce_e3_ub,
                reduce_max01_ub,
                reduce_max23_ub,
                reduce_tile_max_ub,
                reduce_block_max_ub,
            )
            T.wait_flag("mte2", "v", 3)
            decode_reduce_128_plain_pair(
                plain_cached_exp23_ub,
                plain_load64_ub1,
                plain_bits64_ub1,
                reduce_offset_u32_ub,
                reduce_e0_ub,
                reduce_e1_ub,
                reduce_e2_ub,
                reduce_e3_ub,
                reduce_max01_ub,
                reduce_max23_ub,
                reduce_tile_max_ub,
                reduce_block_max_ub,
            )
        T.pipe_barrier("v")
        max_exp = T.alloc_var("int32", init=0)
        for i in T.unroll(32):
            max_exp = T.max(max_exp, reduce_block_max_ub[i])
        out_sf_exp = T.alloc_var("int32", init=0)
        out_sf_exp = T.max(max_exp - 6, 0)
        if vid == 0:
            store_output_sf_scalar(out_sf, out_sf_exp, out_sf_row_offset, out_sf_col_offset)

        fast_relative_exp_a_ub = T.alloc_ub((32,), "int32")
        fast_relative_exp_b_ub = T.alloc_ub((32,), "int32")
        fast_relative_bits_a_ub = T.alloc_ub((32,), "int32")
        fast_relative_bits_b_ub = T.alloc_ub((32,), "int32")
        fast_relative_sf_a_view_ub = T.alloc_ub((32, 1), "float32")
        fast_relative_sf_b_view_ub = T.alloc_ub((32, 1), "float32")
        fast_relative_sf_tile_a_ub = T.alloc_ub((32, 32), "float32")
        fast_relative_sf_tile_b_ub = T.alloc_ub((32, 32), "float32")
        fast_x_in0_ub = T.alloc_ub((32, 32), INPUT_DTYPE)
        fast_x_in1_ub = T.alloc_ub((32, 32), INPUT_DTYPE)
        fast_x_out0_ub = T.alloc_ub((32, 32), OUTPUT_DTYPE)
        fast_x_out1_ub = T.alloc_ub((32, 32), OUTPUT_DTYPE)
        T.reinterpretcast(fast_relative_sf_a_view_ub, fast_relative_bits_a_ub, "float")
        T.reinterpretcast(fast_relative_sf_b_view_ub, fast_relative_bits_b_ub, "float")
        for data_m in T.unroll(num_data_tiles_m):
            fast_row_offset = row_offset + data_m * data_tile_m
            if vid == 0:
                T.copy(x[fast_row_offset : fast_row_offset + 32, col_offset : col_offset + 32], fast_x_in0_ub)
                T.copy(x[fast_row_offset : fast_row_offset + 32, col_offset + 32 : col_offset + 64], fast_x_in1_ub)
            else:
                T.copy(x[fast_row_offset : fast_row_offset + 32, col_offset + 64 : col_offset + 96], fast_x_in0_ub)
                T.copy(x[fast_row_offset : fast_row_offset + 32, col_offset + 96 : col_offset + 128], fast_x_in1_ub)
            T.set_flag("mte2", "v", 0)
            if vid == 0:
                if input_sf_is_tile_packed:
                    if data_m == 0:
                        T.tile.gather(reduce_e0_ub, packed_cached_exp0_ub, reduce_offset_u32_ub, 0)
                        T.tile.gather(reduce_e1_ub, packed_cached_exp0_ub, reduce_offset_u32_ub, 4)
                    elif data_m == 1:
                        T.tile.gather(reduce_e0_ub, packed_cached_exp1_ub, reduce_offset_u32_ub, 0)
                        T.tile.gather(reduce_e1_ub, packed_cached_exp1_ub, reduce_offset_u32_ub, 4)
                    elif data_m == 2:
                        T.tile.gather(reduce_e0_ub, packed_cached_exp2_ub, reduce_offset_u32_ub, 0)
                        T.tile.gather(reduce_e1_ub, packed_cached_exp2_ub, reduce_offset_u32_ub, 4)
                    else:
                        T.tile.gather(reduce_e0_ub, packed_cached_exp3_ub, reduce_offset_u32_ub, 0)
                        T.tile.gather(reduce_e1_ub, packed_cached_exp3_ub, reduce_offset_u32_ub, 4)
                else:
                    if data_m == 0:
                        T.tile.gather(reduce_e0_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 0)
                        T.tile.gather(reduce_e1_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 4)
                    elif data_m == 1:
                        T.tile.gather(reduce_e0_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 512)
                        T.tile.gather(reduce_e1_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 516)
                    elif data_m == 2:
                        T.tile.gather(reduce_e0_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 0)
                        T.tile.gather(reduce_e1_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 4)
                    else:
                        T.tile.gather(reduce_e0_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 512)
                        T.tile.gather(reduce_e1_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 516)
                T.pipe_barrier("v")
                T.tile.add(fast_relative_exp_a_ub, reduce_e0_ub, 127 - out_sf_exp)
                T.tile.add(fast_relative_exp_b_ub, reduce_e1_ub, 127 - out_sf_exp)
                T.pipe_barrier("v")
                T.tile.bitwise_lshift(fast_relative_bits_a_ub, fast_relative_exp_a_ub, 23)
                T.tile.bitwise_lshift(fast_relative_bits_b_ub, fast_relative_exp_b_ub, 23)
                T.pipe_barrier("v")
                T.tile.broadcast(fast_relative_sf_tile_a_ub, fast_relative_sf_a_view_ub, axis=1)
                T.tile.broadcast(fast_relative_sf_tile_b_ub, fast_relative_sf_b_view_ub, axis=1)
            else:
                if input_sf_is_tile_packed:
                    if data_m == 0:
                        T.tile.gather(reduce_e2_ub, packed_cached_exp0_ub, reduce_offset_u32_ub, 8)
                        T.tile.gather(reduce_e3_ub, packed_cached_exp0_ub, reduce_offset_u32_ub, 12)
                    elif data_m == 1:
                        T.tile.gather(reduce_e2_ub, packed_cached_exp1_ub, reduce_offset_u32_ub, 8)
                        T.tile.gather(reduce_e3_ub, packed_cached_exp1_ub, reduce_offset_u32_ub, 12)
                    elif data_m == 2:
                        T.tile.gather(reduce_e2_ub, packed_cached_exp2_ub, reduce_offset_u32_ub, 8)
                        T.tile.gather(reduce_e3_ub, packed_cached_exp2_ub, reduce_offset_u32_ub, 12)
                    else:
                        T.tile.gather(reduce_e2_ub, packed_cached_exp3_ub, reduce_offset_u32_ub, 8)
                        T.tile.gather(reduce_e3_ub, packed_cached_exp3_ub, reduce_offset_u32_ub, 12)
                else:
                    if data_m == 0:
                        T.tile.gather(reduce_e2_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 8)
                        T.tile.gather(reduce_e3_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 12)
                    elif data_m == 1:
                        T.tile.gather(reduce_e2_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 520)
                        T.tile.gather(reduce_e3_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 524)
                    elif data_m == 2:
                        T.tile.gather(reduce_e2_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 8)
                        T.tile.gather(reduce_e3_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 12)
                    else:
                        T.tile.gather(reduce_e2_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 520)
                        T.tile.gather(reduce_e3_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 524)
                T.pipe_barrier("v")
                T.tile.add(fast_relative_exp_a_ub, reduce_e2_ub, 127 - out_sf_exp)
                T.tile.add(fast_relative_exp_b_ub, reduce_e3_ub, 127 - out_sf_exp)
                T.pipe_barrier("v")
                T.tile.bitwise_lshift(fast_relative_bits_a_ub, fast_relative_exp_a_ub, 23)
                T.tile.bitwise_lshift(fast_relative_bits_b_ub, fast_relative_exp_b_ub, 23)
                T.pipe_barrier("v")
                T.tile.broadcast(fast_relative_sf_tile_a_ub, fast_relative_sf_a_view_ub, axis=1)
                T.tile.broadcast(fast_relative_sf_tile_b_ub, fast_relative_sf_b_view_ub, axis=1)
            if vid == 0:
                T.wait_flag("mte2", "v", 0)
                T.tile.cast(fast_x_out0_ub, fast_x_in0_ub, mode="CAST_NONE", count=32 * 32)
                T.tile.cast(fast_x_out1_ub, fast_x_in1_ub, mode="CAST_NONE", count=32 * 32)
                T.tile.mul(fast_x_out0_ub, fast_x_out0_ub, fast_relative_sf_tile_a_ub)
                T.tile.mul(fast_x_out1_ub, fast_x_out1_ub, fast_relative_sf_tile_b_ub)
                T.set_flag("v", "mte2", 0)
                T.set_flag("v", "mte3", 1)
                T.wait_flag("v", "mte3", 1)
                T.copy(fast_x_out0_ub, out[fast_row_offset : fast_row_offset + 32, col_offset : col_offset + 32])
                T.copy(fast_x_out1_ub, out[fast_row_offset : fast_row_offset + 32, col_offset + 32 : col_offset + 64])
                T.set_flag("mte3", "v", 1)
            else:
                T.wait_flag("mte2", "v", 0)
                T.tile.cast(fast_x_out0_ub, fast_x_in0_ub, mode="CAST_NONE", count=32 * 32)
                T.tile.cast(fast_x_out1_ub, fast_x_in1_ub, mode="CAST_NONE", count=32 * 32)
                T.tile.mul(fast_x_out0_ub, fast_x_out0_ub, fast_relative_sf_tile_a_ub)
                T.tile.mul(fast_x_out1_ub, fast_x_out1_ub, fast_relative_sf_tile_b_ub)
                T.set_flag("v", "mte2", 0)
                T.set_flag("v", "mte3", 1)
                T.wait_flag("v", "mte3", 1)
                T.copy(fast_x_out0_ub, out[fast_row_offset : fast_row_offset + 32, col_offset + 64 : col_offset + 96])
                T.copy(fast_x_out1_ub, out[fast_row_offset : fast_row_offset + 32, col_offset + 96 : col_offset + 128])
                T.set_flag("mte3", "v", 1)
            T.wait_flag("v", "mte2", 0)
            T.wait_flag("mte3", "v", 1)

    @T.macro
    def apply_128x128_tile_packed_plain_fast_path(
        x_sf,
        x,
        out,
        out_sf,
        pid_token,
        pid_hidden,
        out_sf_row_offset,
        out_sf_col_offset,
        row_offset,
        col_offset,
        vid,
    ):
        row_offset = pid_token * block_m
        tp128_cached_exp0_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
        tp128_cached_exp1_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
        tp128_cached_exp2_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
        tp128_cached_exp3_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
        tp128_load0_ub = T.alloc_ub((packed_sf_tile_elems,), "float32")
        tp128_load1_ub = T.alloc_ub((packed_sf_tile_elems,), "float32")
        tp128_load2_ub = T.alloc_ub((packed_sf_tile_elems,), "float32")
        tp128_load3_ub = T.alloc_ub((packed_sf_tile_elems,), "float32")
        tp128_bits0_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
        tp128_bits1_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
        tp128_bits2_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
        tp128_bits3_ub = T.alloc_ub((packed_sf_tile_elems,), "int32")
        tp128_offset_i32_ub = T.alloc_ub((32,), "int32")
        tp128_offset_u32_ub = T.alloc_ub((32,), "uint32")
        tp128_e0_ub = T.alloc_ub((32,), "int32")
        tp128_e1_ub = T.alloc_ub((32,), "int32")
        tp128_e2_ub = T.alloc_ub((32,), "int32")
        tp128_e3_ub = T.alloc_ub((32,), "int32")
        tp128_max01_ub = T.alloc_ub((32,), "int32")
        tp128_max23_ub = T.alloc_ub((32,), "int32")
        tp128_tile_max_ub = T.alloc_ub((32,), "int32")
        tp128_block_max_ub = T.alloc_ub((32,), "int32")
        T.reinterpretcast(tp128_offset_u32_ub, tp128_offset_i32_ub, "uint32_t")
        T.tile.arith_progression(tp128_offset_i32_ub, 0, num_in_sf_per_block_k * 4, 32)
        T.tile.fill(tp128_block_max_ub, 0)

        T.copy(x_sf[pid_token, pid_hidden, 0, 0:packed_sf_tile_elems], tp128_load0_ub)
        T.set_flag("mte2", "v", 2)
        T.copy(x_sf[pid_token, pid_hidden, 1, 0:packed_sf_tile_elems], tp128_load1_ub)
        T.set_flag("mte2", "v", 3)
        T.wait_flag("mte2", "v", 2)
        T.reinterpretcast(tp128_bits0_ub, tp128_load0_ub, "int32_t")
        T.tile.bitwise_rshift(tp128_cached_exp0_ub, tp128_bits0_ub, 23)
        T.pipe_barrier("v")
        reduce_128x128_exp_rows(
            tp128_cached_exp0_ub,
            tp128_offset_u32_ub,
            tp128_e0_ub,
            tp128_e1_ub,
            tp128_e2_ub,
            tp128_e3_ub,
            tp128_max01_ub,
            tp128_max23_ub,
            tp128_tile_max_ub,
            tp128_block_max_ub,
            0,
        )
        T.copy(x_sf[pid_token, pid_hidden, 2, 0:packed_sf_tile_elems], tp128_load2_ub)
        T.set_flag("mte2", "v", 2)
        T.wait_flag("mte2", "v", 3)
        T.reinterpretcast(tp128_bits1_ub, tp128_load1_ub, "int32_t")
        T.tile.bitwise_rshift(tp128_cached_exp1_ub, tp128_bits1_ub, 23)
        T.pipe_barrier("v")
        reduce_128x128_exp_rows(
            tp128_cached_exp1_ub,
            tp128_offset_u32_ub,
            tp128_e0_ub,
            tp128_e1_ub,
            tp128_e2_ub,
            tp128_e3_ub,
            tp128_max01_ub,
            tp128_max23_ub,
            tp128_tile_max_ub,
            tp128_block_max_ub,
            0,
        )
        T.copy(x_sf[pid_token, pid_hidden, 3, 0:packed_sf_tile_elems], tp128_load3_ub)
        T.set_flag("mte2", "v", 3)
        T.wait_flag("mte2", "v", 2)
        T.reinterpretcast(tp128_bits2_ub, tp128_load2_ub, "int32_t")
        T.tile.bitwise_rshift(tp128_cached_exp2_ub, tp128_bits2_ub, 23)
        T.pipe_barrier("v")
        reduce_128x128_exp_rows(
            tp128_cached_exp2_ub,
            tp128_offset_u32_ub,
            tp128_e0_ub,
            tp128_e1_ub,
            tp128_e2_ub,
            tp128_e3_ub,
            tp128_max01_ub,
            tp128_max23_ub,
            tp128_tile_max_ub,
            tp128_block_max_ub,
            0,
        )
        T.wait_flag("mte2", "v", 3)
        T.reinterpretcast(tp128_bits3_ub, tp128_load3_ub, "int32_t")
        T.tile.bitwise_rshift(tp128_cached_exp3_ub, tp128_bits3_ub, 23)
        T.pipe_barrier("v")
        reduce_128x128_exp_rows(
            tp128_cached_exp3_ub,
            tp128_offset_u32_ub,
            tp128_e0_ub,
            tp128_e1_ub,
            tp128_e2_ub,
            tp128_e3_ub,
            tp128_max01_ub,
            tp128_max23_ub,
            tp128_tile_max_ub,
            tp128_block_max_ub,
            0,
        )

        T.pipe_barrier("v")
        tp128_max_exp = T.alloc_var("int32", init=0)
        for i in T.unroll(32):
            tp128_max_exp = T.max(tp128_max_exp, tp128_block_max_ub[i])
        tp128_out_sf_exp = T.alloc_var("int32", init=0)
        tp128_out_sf_exp = T.max(tp128_max_exp - 6, 0)
        if vid == 0:
            store_output_sf_scalar(out_sf, tp128_out_sf_exp, out_sf_row_offset, out_sf_col_offset)

        tp128_relative_exp_a_ub = T.alloc_ub((32,), "int32")
        tp128_relative_exp_b_ub = T.alloc_ub((32,), "int32")
        tp128_relative_bits_a_ub = T.alloc_ub((32,), "int32")
        tp128_relative_bits_b_ub = T.alloc_ub((32,), "int32")
        tp128_relative_sf_a_view_ub = T.alloc_ub((32, 1), "float32")
        tp128_relative_sf_b_view_ub = T.alloc_ub((32, 1), "float32")
        tp128_relative_sf_tile_a_ub = T.alloc_ub((32, 32), "float32")
        tp128_relative_sf_tile_b_ub = T.alloc_ub((32, 32), "float32")
        tp128_x_in0_ub = T.alloc_ub((32, 32), INPUT_DTYPE)
        tp128_x_in1_ub = T.alloc_ub((32, 32), INPUT_DTYPE)
        tp128_x_out0_ub = T.alloc_ub((32, 32), OUTPUT_DTYPE)
        tp128_x_out1_ub = T.alloc_ub((32, 32), OUTPUT_DTYPE)
        T.reinterpretcast(tp128_relative_sf_a_view_ub, tp128_relative_bits_a_ub, "float")
        T.reinterpretcast(tp128_relative_sf_b_view_ub, tp128_relative_bits_b_ub, "float")
        for data_m in T.unroll(num_data_tiles_m):
            tp128_row_offset = row_offset + data_m * data_tile_m
            if vid == 0:
                T.copy(x[tp128_row_offset : tp128_row_offset + 32, col_offset : col_offset + 32], tp128_x_in0_ub)
                T.copy(x[tp128_row_offset : tp128_row_offset + 32, col_offset + 32 : col_offset + 64], tp128_x_in1_ub)
            else:
                T.copy(x[tp128_row_offset : tp128_row_offset + 32, col_offset + 64 : col_offset + 96], tp128_x_in0_ub)
                T.copy(x[tp128_row_offset : tp128_row_offset + 32, col_offset + 96 : col_offset + 128], tp128_x_in1_ub)
            T.set_flag("mte2", "v", 0)
            if data_m == 0:
                if vid == 0:
                    T.tile.gather(tp128_e0_ub, tp128_cached_exp0_ub, tp128_offset_u32_ub, 0)
                    T.tile.gather(tp128_e1_ub, tp128_cached_exp0_ub, tp128_offset_u32_ub, 4)
                else:
                    T.tile.gather(tp128_e0_ub, tp128_cached_exp0_ub, tp128_offset_u32_ub, 8)
                    T.tile.gather(tp128_e1_ub, tp128_cached_exp0_ub, tp128_offset_u32_ub, 12)
            elif data_m == 1:
                if vid == 0:
                    T.tile.gather(tp128_e0_ub, tp128_cached_exp1_ub, tp128_offset_u32_ub, 0)
                    T.tile.gather(tp128_e1_ub, tp128_cached_exp1_ub, tp128_offset_u32_ub, 4)
                else:
                    T.tile.gather(tp128_e0_ub, tp128_cached_exp1_ub, tp128_offset_u32_ub, 8)
                    T.tile.gather(tp128_e1_ub, tp128_cached_exp1_ub, tp128_offset_u32_ub, 12)
            elif data_m == 2:
                if vid == 0:
                    T.tile.gather(tp128_e0_ub, tp128_cached_exp2_ub, tp128_offset_u32_ub, 0)
                    T.tile.gather(tp128_e1_ub, tp128_cached_exp2_ub, tp128_offset_u32_ub, 4)
                else:
                    T.tile.gather(tp128_e0_ub, tp128_cached_exp2_ub, tp128_offset_u32_ub, 8)
                    T.tile.gather(tp128_e1_ub, tp128_cached_exp2_ub, tp128_offset_u32_ub, 12)
            else:
                if vid == 0:
                    T.tile.gather(tp128_e0_ub, tp128_cached_exp3_ub, tp128_offset_u32_ub, 0)
                    T.tile.gather(tp128_e1_ub, tp128_cached_exp3_ub, tp128_offset_u32_ub, 4)
                else:
                    T.tile.gather(tp128_e0_ub, tp128_cached_exp3_ub, tp128_offset_u32_ub, 8)
                    T.tile.gather(tp128_e1_ub, tp128_cached_exp3_ub, tp128_offset_u32_ub, 12)
            T.pipe_barrier("v")
            T.tile.add(tp128_relative_exp_a_ub, tp128_e0_ub, 127 - tp128_out_sf_exp)
            T.tile.add(tp128_relative_exp_b_ub, tp128_e1_ub, 127 - tp128_out_sf_exp)
            T.pipe_barrier("v")
            T.tile.bitwise_lshift(tp128_relative_bits_a_ub, tp128_relative_exp_a_ub, 23)
            T.tile.bitwise_lshift(tp128_relative_bits_b_ub, tp128_relative_exp_b_ub, 23)
            T.pipe_barrier("v")
            T.tile.broadcast(tp128_relative_sf_tile_a_ub, tp128_relative_sf_a_view_ub, axis=1)
            T.tile.broadcast(tp128_relative_sf_tile_b_ub, tp128_relative_sf_b_view_ub, axis=1)
            T.wait_flag("mte2", "v", 0)
            T.tile.cast(tp128_x_out0_ub, tp128_x_in0_ub, mode="CAST_NONE", count=32 * 32)
            T.tile.cast(tp128_x_out1_ub, tp128_x_in1_ub, mode="CAST_NONE", count=32 * 32)
            T.tile.mul(tp128_x_out0_ub, tp128_x_out0_ub, tp128_relative_sf_tile_a_ub)
            T.tile.mul(tp128_x_out1_ub, tp128_x_out1_ub, tp128_relative_sf_tile_b_ub)
            T.set_flag("v", "mte2", 0)
            T.set_flag("v", "mte3", 1)
            T.wait_flag("v", "mte3", 1)
            if vid == 0:
                T.copy(tp128_x_out0_ub, out[tp128_row_offset : tp128_row_offset + 32, col_offset : col_offset + 32])
                T.copy(tp128_x_out1_ub, out[tp128_row_offset : tp128_row_offset + 32, col_offset + 32 : col_offset + 64])
            else:
                T.copy(tp128_x_out0_ub, out[tp128_row_offset : tp128_row_offset + 32, col_offset + 64 : col_offset + 96])
                T.copy(tp128_x_out1_ub, out[tp128_row_offset : tp128_row_offset + 32, col_offset + 96 : col_offset + 128])
            T.set_flag("mte3", "v", 1)
            T.wait_flag("v", "mte2", 0)
            T.wait_flag("mte3", "v", 1)

    @T.macro
    def apply_128x128_generic_fast_path(
        x_sf,
        x,
        out,
        out_sf,
        pid_token,
        pid_hidden,
        sf_row_offset,
        sf_col_offset,
        out_sf_row_offset,
        out_sf_col_offset,
        row_offset,
        col_offset,
    ):
        row_offset = pid_token * block_m
        out_sf_exp_ub = T.alloc_ub((num_out_sf_per_block_m, num_out_sf_per_block_k), "int32")
        out_sf_fp32_ub = T.alloc_ub((num_out_sf_per_block_m, num_out_sf_per_block_k), "float32")
        cached_exp0_ub = T.alloc_ub((32, 4), "int32")
        cached_exp1_ub = T.alloc_ub((32, 4), "int32")
        cached_exp2_ub = T.alloc_ub((32, 4), "int32")
        cached_exp3_ub = T.alloc_ub((32, 4), "int32")
        plain_cached_exp01_ub = T.alloc_ub((64, 4), "int32")
        plain_cached_exp23_ub = T.alloc_ub((64, 4), "int32")
        plain_load64_ub0 = T.alloc_ub((64, 4), "float32")
        plain_load64_ub1 = T.alloc_ub((64, 4), "float32")
        plain_bits64_ub0 = T.alloc_ub((64, 4), "int32")
        plain_bits64_ub1 = T.alloc_ub((64, 4), "int32")
        reduce_offset_i32_ub = T.alloc_ub((32,), "int32")
        reduce_offset_u32_ub = T.alloc_ub((32,), "uint32")
        reduce_e0_ub = T.alloc_ub((32,), "int32")
        reduce_e1_ub = T.alloc_ub((32,), "int32")
        reduce_e2_ub = T.alloc_ub((32,), "int32")
        reduce_e3_ub = T.alloc_ub((32,), "int32")
        reduce_max01_ub = T.alloc_ub((32,), "int32")
        reduce_max23_ub = T.alloc_ub((32,), "int32")
        reduce_tile_max_ub = T.alloc_ub((32,), "int32")
        reduce_block_max_ub = T.alloc_ub((32,), "int32")
        T.reinterpretcast(reduce_offset_u32_ub, reduce_offset_i32_ub, "uint32_t")
        T.tile.arith_progression(reduce_offset_i32_ub, 0, 16, 32)
        T.tile.fill(reduce_block_max_ub, 0)
        if (not in_config.use_packed_ue8m0) and (not in_config.use_tma_aligned_col_major_sf) and (not input_sf_is_tile_packed):
            T.copy(
                x_sf[sf_row_offset : sf_row_offset + 64, sf_col_offset : sf_col_offset + 4],
                plain_load64_ub0,
            )
            T.set_flag("mte2", "v", 2)
            T.copy(
                x_sf[sf_row_offset + 64 : sf_row_offset + 128, sf_col_offset : sf_col_offset + 4],
                plain_load64_ub1,
            )
            T.set_flag("mte2", "v", 3)
            T.wait_flag("mte2", "v", 2)
            decode_reduce_128_plain_pair(
                plain_cached_exp01_ub,
                plain_load64_ub0,
                plain_bits64_ub0,
                reduce_offset_u32_ub,
                reduce_e0_ub,
                reduce_e1_ub,
                reduce_e2_ub,
                reduce_e3_ub,
                reduce_max01_ub,
                reduce_max23_ub,
                reduce_tile_max_ub,
                reduce_block_max_ub,
            )
            T.wait_flag("mte2", "v", 3)
            decode_reduce_128_plain_pair(
                plain_cached_exp23_ub,
                plain_load64_ub1,
                plain_bits64_ub1,
                reduce_offset_u32_ub,
                reduce_e0_ub,
                reduce_e1_ub,
                reduce_e2_ub,
                reduce_e3_ub,
                reduce_max01_ub,
                reduce_max23_ub,
                reduce_tile_max_ub,
                reduce_block_max_ub,
            )
        else:
            x_sf_load_ub = T.alloc_ub(x_sf_load_shape, x_sf_load_dtype)
            load_reduce_128_cache(
                cached_exp0_ub,
                x_sf_load_ub,
                x_sf,
                sf_row_offset,
                sf_col_offset,
                0,
                pid_token,
                pid_hidden,
                reduce_offset_u32_ub,
                reduce_e0_ub,
                reduce_e1_ub,
                reduce_e2_ub,
                reduce_e3_ub,
                reduce_max01_ub,
                reduce_max23_ub,
                reduce_tile_max_ub,
                reduce_block_max_ub,
            )
            load_reduce_128_cache(
                cached_exp1_ub,
                x_sf_load_ub,
                x_sf,
                sf_row_offset + 32,
                sf_col_offset,
                1,
                pid_token,
                pid_hidden,
                reduce_offset_u32_ub,
                reduce_e0_ub,
                reduce_e1_ub,
                reduce_e2_ub,
                reduce_e3_ub,
                reduce_max01_ub,
                reduce_max23_ub,
                reduce_tile_max_ub,
                reduce_block_max_ub,
            )
            load_reduce_128_cache(
                cached_exp2_ub,
                x_sf_load_ub,
                x_sf,
                sf_row_offset + 64,
                sf_col_offset,
                2,
                pid_token,
                pid_hidden,
                reduce_offset_u32_ub,
                reduce_e0_ub,
                reduce_e1_ub,
                reduce_e2_ub,
                reduce_e3_ub,
                reduce_max01_ub,
                reduce_max23_ub,
                reduce_tile_max_ub,
                reduce_block_max_ub,
            )
            load_reduce_128_cache(
                cached_exp3_ub,
                x_sf_load_ub,
                x_sf,
                sf_row_offset + 96,
                sf_col_offset,
                3,
                pid_token,
                pid_hidden,
                reduce_offset_u32_ub,
                reduce_e0_ub,
                reduce_e1_ub,
                reduce_e2_ub,
                reduce_e3_ub,
                reduce_max01_ub,
                reduce_max23_ub,
                reduce_tile_max_ub,
                reduce_block_max_ub,
            )
        T.pipe_barrier("v")
        max_exp = T.alloc_var("int32", init=0)
        for i in T.unroll(32):
            max_exp = T.max(max_exp, reduce_block_max_ub[i])
        out_sf_exp_ub[0, 0] = T.max(max_exp - 6, 0)
        store_output_sf_block(out_sf, out_sf_fp32_ub, out_sf_exp_ub, out_sf_row_offset, out_sf_col_offset)
        if use_128x128_generic_fast_path:
            fast_relative_exp0_ub = T.alloc_ub((32,), "int32")
            fast_relative_exp1_ub = T.alloc_ub((32,), "int32")
            fast_relative_exp2_ub = T.alloc_ub((32,), "int32")
            fast_relative_exp3_ub = T.alloc_ub((32,), "int32")
            fast_relative_bits0_ub = T.alloc_ub((32,), "int32")
            fast_relative_bits1_ub = T.alloc_ub((32,), "int32")
            fast_relative_bits2_ub = T.alloc_ub((32,), "int32")
            fast_relative_bits3_ub = T.alloc_ub((32,), "int32")
            fast_relative_sf0_view_ub = T.alloc_ub((32, 1), "float32")
            fast_relative_sf1_view_ub = T.alloc_ub((32, 1), "float32")
            fast_relative_sf2_view_ub = T.alloc_ub((32, 1), "float32")
            fast_relative_sf3_view_ub = T.alloc_ub((32, 1), "float32")
            fast_relative_sf_tile0_ub = T.alloc_ub((32, 32), "float32")
            fast_relative_sf_tile1_ub = T.alloc_ub((32, 32), "float32")
            fast_relative_sf_tile2_ub = T.alloc_ub((32, 32), "float32")
            fast_relative_sf_tile3_ub = T.alloc_ub((32, 32), "float32")
            fast_x_in0_ub = T.alloc_ub((32, 32), INPUT_DTYPE)
            fast_x_in1_ub = T.alloc_ub((32, 32), INPUT_DTYPE)
            fast_x_out0_ub = T.alloc_ub((32, 32), OUTPUT_DTYPE)
            fast_x_out1_ub = T.alloc_ub((32, 32), OUTPUT_DTYPE)
            T.reinterpretcast(fast_relative_sf0_view_ub, fast_relative_bits0_ub, "float")
            T.reinterpretcast(fast_relative_sf1_view_ub, fast_relative_bits1_ub, "float")
            T.reinterpretcast(fast_relative_sf2_view_ub, fast_relative_bits2_ub, "float")
            T.reinterpretcast(fast_relative_sf3_view_ub, fast_relative_bits3_ub, "float")
            for data_m in T.unroll(num_data_tiles_m):
                if (not in_config.use_packed_ue8m0) and (not in_config.use_tma_aligned_col_major_sf) and (not input_sf_is_tile_packed):
                    if data_m == 0:
                        T.tile.gather(reduce_e0_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 0)
                        T.tile.gather(reduce_e1_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 4)
                        T.tile.gather(reduce_e2_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 8)
                        T.tile.gather(reduce_e3_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 12)
                    elif data_m == 1:
                        T.tile.gather(reduce_e0_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 512)
                        T.tile.gather(reduce_e1_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 516)
                        T.tile.gather(reduce_e2_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 520)
                        T.tile.gather(reduce_e3_ub, plain_cached_exp01_ub, reduce_offset_u32_ub, 524)
                    elif data_m == 2:
                        T.tile.gather(reduce_e0_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 0)
                        T.tile.gather(reduce_e1_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 4)
                        T.tile.gather(reduce_e2_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 8)
                        T.tile.gather(reduce_e3_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 12)
                    else:
                        T.tile.gather(reduce_e0_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 512)
                        T.tile.gather(reduce_e1_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 516)
                        T.tile.gather(reduce_e2_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 520)
                        T.tile.gather(reduce_e3_ub, plain_cached_exp23_ub, reduce_offset_u32_ub, 524)
                else:
                    if data_m == 0:
                        T.tile.gather(reduce_e0_ub, cached_exp0_ub, reduce_offset_u32_ub, 0)
                        T.tile.gather(reduce_e1_ub, cached_exp0_ub, reduce_offset_u32_ub, 4)
                        T.tile.gather(reduce_e2_ub, cached_exp0_ub, reduce_offset_u32_ub, 8)
                        T.tile.gather(reduce_e3_ub, cached_exp0_ub, reduce_offset_u32_ub, 12)
                    elif data_m == 1:
                        T.tile.gather(reduce_e0_ub, cached_exp1_ub, reduce_offset_u32_ub, 0)
                        T.tile.gather(reduce_e1_ub, cached_exp1_ub, reduce_offset_u32_ub, 4)
                        T.tile.gather(reduce_e2_ub, cached_exp1_ub, reduce_offset_u32_ub, 8)
                        T.tile.gather(reduce_e3_ub, cached_exp1_ub, reduce_offset_u32_ub, 12)
                    elif data_m == 2:
                        T.tile.gather(reduce_e0_ub, cached_exp2_ub, reduce_offset_u32_ub, 0)
                        T.tile.gather(reduce_e1_ub, cached_exp2_ub, reduce_offset_u32_ub, 4)
                        T.tile.gather(reduce_e2_ub, cached_exp2_ub, reduce_offset_u32_ub, 8)
                        T.tile.gather(reduce_e3_ub, cached_exp2_ub, reduce_offset_u32_ub, 12)
                    else:
                        T.tile.gather(reduce_e0_ub, cached_exp3_ub, reduce_offset_u32_ub, 0)
                        T.tile.gather(reduce_e1_ub, cached_exp3_ub, reduce_offset_u32_ub, 4)
                        T.tile.gather(reduce_e2_ub, cached_exp3_ub, reduce_offset_u32_ub, 8)
                        T.tile.gather(reduce_e3_ub, cached_exp3_ub, reduce_offset_u32_ub, 12)
                T.pipe_barrier("v")
                T.tile.add(fast_relative_exp0_ub, reduce_e0_ub, 127 - out_sf_exp_ub[0, 0])
                T.tile.add(fast_relative_exp1_ub, reduce_e1_ub, 127 - out_sf_exp_ub[0, 0])
                T.tile.add(fast_relative_exp2_ub, reduce_e2_ub, 127 - out_sf_exp_ub[0, 0])
                T.tile.add(fast_relative_exp3_ub, reduce_e3_ub, 127 - out_sf_exp_ub[0, 0])
                T.pipe_barrier("v")
                T.tile.bitwise_lshift(fast_relative_bits0_ub, fast_relative_exp0_ub, 23)
                T.tile.bitwise_lshift(fast_relative_bits1_ub, fast_relative_exp1_ub, 23)
                T.tile.bitwise_lshift(fast_relative_bits2_ub, fast_relative_exp2_ub, 23)
                T.tile.bitwise_lshift(fast_relative_bits3_ub, fast_relative_exp3_ub, 23)
                T.pipe_barrier("v")
                T.tile.broadcast(fast_relative_sf_tile0_ub, fast_relative_sf0_view_ub, axis=1)
                T.tile.broadcast(fast_relative_sf_tile1_ub, fast_relative_sf1_view_ub, axis=1)
                T.tile.broadcast(fast_relative_sf_tile2_ub, fast_relative_sf2_view_ub, axis=1)
                T.tile.broadcast(fast_relative_sf_tile3_ub, fast_relative_sf3_view_ub, axis=1)
                fast_row_offset = row_offset + data_m * data_tile_m
                T.copy(x[fast_row_offset : fast_row_offset + 32, col_offset : col_offset + 32], fast_x_in0_ub)
                T.copy(x[fast_row_offset : fast_row_offset + 32, col_offset + 32 : col_offset + 64], fast_x_in1_ub)
                T.set_flag("mte2", "v", 0)
                T.set_flag("mte3", "v", 1)
                for pair_idx in T.unroll(2):
                    fast_col_offset0 = col_offset + pair_idx * 64
                    fast_col_offset1 = fast_col_offset0 + 32
                    if pair_idx == 0:
                        T.wait_flag("mte2", "v", 0)
                    else:
                        T.wait_flag("v", "mte2", 0)
                        T.copy(
                            x[fast_row_offset : fast_row_offset + 32, fast_col_offset0 : fast_col_offset0 + 32],
                            fast_x_in0_ub,
                        )
                        T.copy(
                            x[fast_row_offset : fast_row_offset + 32, fast_col_offset1 : fast_col_offset1 + 32],
                            fast_x_in1_ub,
                        )
                        T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 0)
                    T.wait_flag("mte3", "v", 1)
                    T.tile.cast(fast_x_out0_ub, fast_x_in0_ub, mode="CAST_NONE", count=32 * 32)
                    T.tile.cast(fast_x_out1_ub, fast_x_in1_ub, mode="CAST_NONE", count=32 * 32)
                    if pair_idx == 0:
                        T.tile.mul(fast_x_out0_ub, fast_x_out0_ub, fast_relative_sf_tile0_ub)
                        T.tile.mul(fast_x_out1_ub, fast_x_out1_ub, fast_relative_sf_tile1_ub)
                    else:
                        T.tile.mul(fast_x_out0_ub, fast_x_out0_ub, fast_relative_sf_tile2_ub)
                        T.tile.mul(fast_x_out1_ub, fast_x_out1_ub, fast_relative_sf_tile3_ub)
                    T.set_flag("v", "mte2", 0)
                    T.set_flag("v", "mte3", 1)
                    T.wait_flag("v", "mte3", 1)
                    T.copy(
                        fast_x_out0_ub,
                        out[fast_row_offset : fast_row_offset + 32, fast_col_offset0 : fast_col_offset0 + 32],
                    )
                    T.copy(
                        fast_x_out1_ub,
                        out[fast_row_offset : fast_row_offset + 32, fast_col_offset1 : fast_col_offset1 + 32],
                    )
                    T.set_flag("mte3", "v", 1)
                T.wait_flag("v", "mte2", 0)
                T.wait_flag("mte3", "v", 1)

    @T.prim_func
    def per_block_cast_lossless_kernel(
        x: T.Tensor([num_tokens, hidden], INPUT_DTYPE),
        x_sf: T.Tensor(x_sf_shape, in_config.sf_dtype),
        out: T.Tensor([num_tokens, hidden], OUTPUT_DTYPE),
        out_sf: T.Tensor(out_sf_shape, out_config.sf_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid), T.Scope("V"):
            pid_token = cid // n_num
            pid_hidden = cid % n_num
            row_offset = pid_token * block_m + vid * fast_data_tile_m
            col_offset = pid_hidden * block_k
            sf_row_offset = pid_token * num_in_sf_per_block_m
            sf_col_offset = pid_hidden * num_in_sf_per_block_k
            out_sf_row_offset = pid_token * num_out_sf_per_block_m
            out_sf_col_offset = pid_hidden * num_out_sf_per_block_k
            if use_32x32_tile_packed_fast_path:
                apply_32x32_tile_packed_fast_path(
                    x_sf,
                    x,
                    out,
                    out_sf,
                    pid_token,
                    pid_hidden,
                    vid,
                    out_sf_row_offset + vid,
                    out_sf_col_offset,
                    row_offset,
                    col_offset,
                )
            elif use_32x32_packed_fast_path:
                apply_32x32_packed_fast_path(
                    x_sf,
                    x,
                    out,
                    out_sf,
                    sf_row_offset + vid * 32,
                    sf_col_offset,
                    out_sf_row_offset + vid,
                    out_sf_col_offset,
                    row_offset,
                    col_offset,
                )
            elif use_max4_fast_path:
                x_sf_load_ub = T.alloc_ub(fast_x_sf_load_shape, fast_x_sf_load_dtype)
                x_sf_exp_ub = T.alloc_ub((fast_num_in_sf_per_data_tile_m, num_in_sf_per_block_k), "int32")
                apply_max4_fast_path(
                    x_sf,
                    x,
                    out,
                    out_sf,
                    x_sf_load_ub,
                    x_sf_exp_ub,
                    sf_row_offset + vid * fast_num_in_sf_per_data_tile_m,
                    sf_col_offset,
                    out_sf_row_offset + vid * fast_data_tile_m,
                    out_sf_col_offset,
                    row_offset,
                    col_offset,
                )
            elif use_128x128_tile_packed_plain_fast_path:
                apply_128x128_tile_packed_plain_fast_path(
                    x_sf,
                    x,
                    out,
                    out_sf,
                    pid_token,
                    pid_hidden,
                    out_sf_row_offset,
                    out_sf_col_offset,
                    row_offset,
                    col_offset,
                    vid,
                )
            elif use_128x128_plain_fast_path:
                apply_128x128_plain_fast_path(
                    x_sf,
                    x,
                    out,
                    out_sf,
                    pid_token,
                    pid_hidden,
                    sf_row_offset,
                    sf_col_offset,
                    out_sf_row_offset,
                    out_sf_col_offset,
                    row_offset,
                    col_offset,
                    vid,
                )
            elif use_128x128_generic_fast_path:
                if vid == 0:
                    apply_128x128_generic_fast_path(
                        x_sf,
                        x,
                        out,
                        out_sf,
                        pid_token,
                        pid_hidden,
                        sf_row_offset,
                        sf_col_offset,
                        out_sf_row_offset,
                        out_sf_col_offset,
                        row_offset,
                        col_offset,
                    )
            else:
                if vid == 0:
                    row_offset = pid_token * block_m
                    x_sf_load_ub = T.alloc_ub(x_sf_load_shape, x_sf_load_dtype)
                    x_sf_exp_ub = T.alloc_ub((num_in_sf_per_data_tile_m, num_in_sf_per_block_k), "int32")
                    x_in_ub = T.alloc_ub((data_tile_m, block_k), INPUT_DTYPE)
                    x_out_ub = T.alloc_ub((data_tile_m, block_k), OUTPUT_DTYPE)
                    x_relative_sf_ub = T.alloc_ub((data_tile_m, block_k), "float32")
                    out_sf_exp_ub = T.alloc_ub((num_out_sf_per_block_m, num_out_sf_per_block_k), "int32")
                    out_sf_fp32_ub = T.alloc_ub((num_out_sf_per_block_m, num_out_sf_per_block_k), "float32")
                    for i in T.serial(num_out_sf_per_block_m):
                        for j in T.serial(num_out_sf_per_block_k):
                            out_sf_exp_ub[i, j] = 0
                    for data_m in T.serial(num_data_tiles_m):
                        load_and_decode_input_sf(
                            x_sf_exp_ub,
                            x_sf_load_ub,
                            x_sf,
                            sf_row_offset + data_m * num_in_sf_per_data_tile_m,
                            sf_col_offset,
                            data_m,
                            pid_token,
                            pid_hidden,
                        )
                        reduce_output_sf_exp(out_sf_exp_ub, x_sf_exp_ub, data_m)
                    for i in T.serial(num_out_sf_per_block_m):
                        for j in T.serial(num_out_sf_per_block_k):
                            out_sf_exp_ub[i, j] = T.max(out_sf_exp_ub[i, j] - 6, 0)
                    store_output_sf_block(
                        out_sf,
                        out_sf_fp32_ub,
                        out_sf_exp_ub,
                        out_sf_row_offset,
                        out_sf_col_offset,
                    )
                    for data_m in T.serial(num_data_tiles_m):
                        if input_sf_is_tile_packed:
                            load_and_decode_input_sf(
                                x_sf_exp_ub,
                                x_sf_load_ub,
                                x_sf,
                                sf_row_offset + data_m * num_in_sf_per_data_tile_m,
                                sf_col_offset,
                                data_m,
                                pid_token,
                                pid_hidden,
                            )
                        update_relative_sf_exp(x_sf_exp_ub, out_sf_exp_ub, data_m)
                        expand_relative_sf(x_relative_sf_ub, x_sf_exp_ub)
                        cast_data_tile(
                            x,
                            out,
                            x_in_ub,
                            x_out_ub,
                            x_relative_sf_ub,
                            row_offset + data_m * data_tile_m,
                            col_offset,
                        )

    return per_block_cast_lossless_kernel


def per_block_cast_lossless(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    fmt: str = "fp32",
    x_block_size: tuple[int, int] = DEFAULT_IN_SF_BLOCK,
    out_block_size: tuple[int, int] = DEFAULT_OUT_SF_BLOCK,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    in_use_tma_aligned_col_major_sf: bool | None = None,
    in_round_sf: bool | None = None,
    in_use_packed_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_data, x_sf, in_config = get_cast_input_and_config(
        x,
        x_block_size,
        use_tma_aligned_col_major_sf=in_use_tma_aligned_col_major_sf,
        round_sf=in_round_sf,
        use_packed_ue8m0=in_use_packed_ue8m0,
    )
    if not in_config.with_sf:
        in_config = replace(in_config, with_sf=True, sf_block=x_block_size)
    assert fmt in ("fp32", "float32", "e4m3")
    assert x_data.dim() == 2 and x_data.is_contiguous()
    assert x_data.device.type == "npu"
    assert x_data.dtype == torch.bfloat16
    num_tokens, hidden = x_data.shape
    out_config = get_cast_output_config("fp32", out_block_size, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)
    if num_tokens == 0 or hidden == 0:
        out = torch.empty((num_tokens, hidden), dtype=out_config.torch_dtype, device=x_data.device)
        out_sf = alloc_scaling_factors((num_tokens, hidden), out_config, x_data.device)
        return (out, cast_epilogue(out_sf, num_tokens, hidden, out_config))
    block_m, block_k = _derive_cast_layout(hidden, in_config, out_config)
    padded_tokens = align_up(num_tokens, block_m)
    padded_hidden = align_up(hidden, block_k)
    x_padded = _pad_2d_tensor(x_data, (padded_tokens, padded_hidden))

    x_sf_shape = get_sf_shape((padded_tokens, padded_hidden), in_config)
    x_sf_padded = _pad_2d_tensor(x_sf, x_sf_shape)

    out = torch.empty((padded_tokens, padded_hidden), dtype=out_config.torch_dtype, device=x_data.device)
    out_sf = alloc_scaling_factors((padded_tokens, padded_hidden), out_config, x_data.device)
    data_tile_m = min(block_m, 32, NUM_ELEMENTS_PER_BLOCK // block_k)
    num_data_tiles_m = block_m // data_tile_m
    input_sf_is_tile_packed = _needs_tile_packed_input_sf(in_config, out_config, num_data_tiles_m)
    if input_sf_is_tile_packed:
        kernel_input_sf = _pack_plain_sf_for_lossless_kernel(
            x_sf_padded.contiguous(),
            padded_tokens,
            padded_hidden,
            block_m,
            block_k,
            in_config.sf_block,
        )
    else:
        kernel_input_sf = x_sf_padded
    kernel = get_per_block_cast_lossless_kernel(
        hidden=padded_hidden,
        block_m=block_m,
        block_k=block_k,
        in_config=in_config,
        out_config=out_config,
        in_sf_block_m=in_config.sf_block[0],
        in_sf_block_k=in_config.sf_block[1],
        out_sf_block_m=out_config.sf_block[0],
        out_sf_block_k=out_config.sf_block[1],
        input_sf_is_tile_packed=input_sf_is_tile_packed,
    )
    print_kernel_source = int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0))
    if print_kernel_source:
        print(kernel.get_kernel_source())
    out, out_sf = kernel(x_padded, kernel_input_sf, out, out_sf)
    if out.shape[0] != num_tokens or out.shape[1] != hidden:
        out = out[:num_tokens, :hidden].contiguous()
    elif not out.is_contiguous():
        out = out.contiguous()
    out_sf = cast_epilogue(out_sf, num_tokens, hidden, out_config)
    return (out, out_sf)


def generate_num_tokens(is_benchmark: bool = False) -> list[int]:
    return [4001, 8001]


def generate_hidden_sizes() -> list[int]:
    return [576, 2048, 2560, 3072, 4096, 6144, 7168]


def generate_rand_float(shape: tuple[int, int]) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float32, device="npu")


def make_param_id(params: dict) -> str:
    parts = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, tuple):
            value = "x".join(str(item) for item in value)
        parts.append(f"{key}={value}")
    return "_".join(parts)


def assert_equal(actual: torch.Tensor, expected: torch.Tensor, check_stride: bool = True) -> None:
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)
    if check_stride:
        assert actual.stride() == expected.stride()


def clamp_abs_ratio(t: torch.Tensor, max_ratio: float = 2**9) -> torch.Tensor:
    if t.numel() == 0:
        return t
    floor_val = t.abs().max() / max_ratio
    return torch.sign(t) * torch.maximum(t.abs(), floor_val)


def cast(
    x: torch.Tensor,
    block_size: tuple[int, int],
    round_sf: bool = False,
    use_tma_aligned_col_major_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 2 and x.dtype == torch.float32
    h, w = x.shape
    block_m, block_k = block_size
    pad_m = (block_m - h % block_m) % block_m
    pad_k = (block_k - w % block_k) % block_k
    padded = F.pad(x, (0, pad_k, 0, pad_m))
    valid = F.pad(torch.ones_like(x, dtype=torch.bool), (0, pad_k, 0, pad_m))
    padded_m, padded_k = padded.shape

    values = (
        padded.view(padded_m // block_m, block_m, padded_k // block_k, block_k)
        .permute(0, 2, 1, 3)
        .reshape(padded_m // block_m, padded_k // block_k, -1)
    )
    mask = (
        valid.view(padded_m // block_m, block_m, padded_k // block_k, block_k)
        .permute(0, 2, 1, 3)
        .reshape(padded_m // block_m, padded_k // block_k, -1)
    )
    values = torch.where(mask, values.abs(), torch.tensor(-1.0, device=x.device))
    max_value = values.max(dim=-1, keepdim=True).values.clamp(min=INPUT_MIN_CLAMP_VAL)
    sf_bits = (max_value / INPUT_MAX_QUANT_VAL).contiguous().view(torch.int32)
    if round_sf:
        sf_bits = (sf_bits + 0x007FFFFF) & 0x7F800000

    sf = sf_bits.view(torch.float32)
    quant_sf = torch.ones_like(sf) / sf
    x_bf16 = (
        (
            padded.view(padded_m // block_m, block_m, padded_k // block_k, block_k)
            * quant_sf.view(padded_m // block_m, 1, padded_k // block_k, 1)
        )
        .reshape(padded_m, padded_k)[:h, :w]
        .to(torch.bfloat16)
        .contiguous()
    )

    sf_bits = sf_bits.squeeze(-1)
    if use_packed_ue8m0:
        sf_bits = sf_bits.detach().cpu()
        pad_m = align_up(sf_bits.shape[0], 4) - sf_bits.shape[0]
        pad_k = align_up(sf_bits.shape[1], 4) - sf_bits.shape[1]
        sf_bits = F.pad(sf_bits, (0, pad_k, 0, pad_m))
        sf = (sf_bits >> 23).to(torch.int8).view(torch.int32).to(device=x.device)
        sf = sf.T.contiguous().T[: max_value.shape[0], :]
    else:
        if use_tma_aligned_col_major_sf:
            pad_m = align_up(sf_bits.shape[0], 4) - sf_bits.shape[0]
            sf_bits = F.pad(sf_bits, (0, 0, 0, pad_m))
        sf = sf_bits.view(torch.float32)
        if use_tma_aligned_col_major_sf:
            sf = sf.T.contiguous().T[: max_value.shape[0], :]
        else:
            sf = sf.contiguous()
    return x_bf16, sf


def compute_reference_out(x: tuple[torch.Tensor, torch.Tensor], params: dict) -> torch.Tensor:
    x_bf16, x_sf = x
    output_device = x_bf16.device
    x_bf16 = x_bf16.detach().cpu()
    x_sf = x_sf.detach().cpu()
    in_block_m, in_block_k = params["in_sf_block"]
    out_block_m, out_block_k = params["out_sf_block"]
    if params["in_use_packed_ue8m0"]:
        sf_m = (x_bf16.shape[0] + in_block_m - 1) // in_block_m
        sf_k = (x_bf16.shape[1] + in_block_k - 1) // in_block_k
        x_sf = x_sf.contiguous().view(torch.uint8).reshape(x_sf.shape[0], -1)
        x_sf = (x_sf[:sf_m, :sf_k].to(torch.int32) << 23).contiguous().view(torch.float32)
    input_sf = (x_sf.repeat_interleave(in_block_m, dim=0).repeat_interleave(in_block_k, dim=1))[: x_bf16.shape[0], : x_bf16.shape[1]]
    input_dequant = x_bf16.to(torch.float32) * input_sf

    group_m = out_block_m // in_block_m
    group_k = out_block_k // in_block_k
    out_sf_m = (x_sf.shape[0] + group_m - 1) // group_m
    out_sf_k = (x_sf.shape[1] + group_k - 1) // group_k
    padded_exp = torch.zeros((out_sf_m * group_m, out_sf_k * group_k), dtype=torch.int32, device=x_sf.device)
    input_exp = (x_sf.contiguous().view(torch.int32) >> 23) & 0xFF
    padded_exp[: input_exp.shape[0], : input_exp.shape[1]] = input_exp
    output_exp = padded_exp.view(out_sf_m, group_m, out_sf_k, group_k).amax(dim=(1, 3)).sub(6).clamp(min=0)
    output_sf = (output_exp << 23).contiguous().view(torch.float32)
    output_sf = (output_sf.repeat_interleave(out_block_m, dim=0).repeat_interleave(out_block_k, dim=1))[
        : x_bf16.shape[0], : x_bf16.shape[1]
    ]
    return (input_dequant / output_sf).to(device=output_device)


def generate_test_data(
    params: dict,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    num_tokens = params["num_tokens"]
    hidden = params["hidden"]
    in_sf_block_m, in_sf_block_k = params["in_sf_block"]
    out_sf_block_m, out_sf_block_k = params["out_sf_block"]
    x = clamp_abs_ratio(generate_rand_float((num_tokens, hidden)))
    x_bf16 = cast(
        x,
        (in_sf_block_m, in_sf_block_k),
        use_tma_aligned_col_major_sf=params["in_use_tma_aligned_col_major_sf"],
        round_sf=params["in_round_sf"],
        use_packed_ue8m0=params["in_use_packed_ue8m0"],
    )

    def cast_func():
        return per_block_cast_lossless(
            x_bf16,
            "fp32",
            x_block_size=(in_sf_block_m, in_sf_block_k),
            out_block_size=(out_sf_block_m, out_sf_block_k),
            use_tma_aligned_col_major_sf=params["out_use_tma_aligned_col_major_sf"],
            round_sf=params["out_round_sf"],
            use_packed_ue8m0=params["out_use_packed_ue8m0"],
        )

    return (x, x_bf16, cast_func)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {
            "num_tokens": num_tokens,
            "hidden": hidden_size,
            "in_use_tma_aligned_col_major_sf": in_use_tma_aligned_col_major_sf,
            "in_round_sf": in_round_sf,
            "in_use_packed_ue8m0": in_use_packed_ue8m0,
            "out_use_tma_aligned_col_major_sf": out_use_tma_aligned_col_major_sf,
            "out_round_sf": out_round_sf,
            "out_use_packed_ue8m0": out_use_packed_ue8m0,
            "out_sf_block": (out_sf_block_m, out_sf_block_k),
            "in_sf_block": (in_sf_block_m, in_sf_block_k),
        }
        for num_tokens in generate_num_tokens(is_benchmark=is_benchmark)
        for hidden_size in generate_hidden_sizes()
        for in_use_tma_aligned_col_major_sf, in_round_sf, in_use_packed_ue8m0 in [(False, True, False), (True, True, True)]
        for out_use_tma_aligned_col_major_sf, out_round_sf, out_use_packed_ue8m0 in [(False, True, False), (True, True, True)]
        for out_sf_block_m, out_sf_block_k in (
            (1, 128),
            (32, 32),
            (128, 128),
        )
        for in_sf_block_m, in_sf_block_k in ((1, 32),)
        if out_sf_block_m % in_sf_block_m == 0 and out_sf_block_k % in_sf_block_k == 0
    ]


@pytest.mark.parametrize("params", generate_test_params(is_benchmark=False), ids=make_param_id)
def test_per_block_cast_lossless(params: dict) -> None:
    _, x_bf16, cast_func = generate_test_data(params)
    out, _ = cast_func()
    if hasattr(torch, "npu"):
        torch.npu.synchronize()

    out_ref = compute_reference_out(x_bf16, params)
    assert_equal(out, out_ref, check_stride=False)


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v"])
    if exit_code == pytest.ExitCode.OK:
        print("All per_block_cast_lossless tests passed! Kernel Output Match!")
    sys.exit(exit_code)
