import os
import sys
from collections.abc import Iterable
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
FAST_PATH_HIDDEN = 576
GENERIC_BLOCK_M = 12
PERSISTENT_CORE_NUM = 24


def _get_block_m(hidden: int) -> int:
    return 8 if hidden == FAST_PATH_HIDDEN else GENERIC_BLOCK_M


@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def get_per_token_cast_to_e5m6_kernel(hidden: int, in_config: CastInputConfig, out_config: CastOutputConfig):
    assert not in_config.with_sf
    assert out_config.sf_block == (1, hidden)
    block_m = _get_block_m(hidden)
    if hidden == FAST_PATH_HIDDEN or hidden <= 256:
        block_k = hidden
    else:
        block_k = 512
        while hidden % block_k != 0:
            block_k //= 2
    VEC_NUM = 2
    block_m_per_vec = block_m // VEC_NUM
    num_tokens = T.symbolic("num_tokens")
    m_num = T.ceildiv(num_tokens, block_m)
    n_num = T.ceildiv(hidden, block_k)
    single_chunk = hidden == block_k
    round_sf = bool(out_config.round_sf)
    use_packed_ue8m0 = bool(out_config.use_packed_ue8m0)
    kernel_sf_dtype = "int32" if use_packed_ue8m0 else out_config.sf_dtype
    packed_block_k = block_k // 8 * 3
    num_pack_groups = block_k // 8
    fast_num_pack_groups = block_m_per_vec * num_pack_groups
    interleave_groups = block_k // 8
    interleave_words_per_row = interleave_groups * 3
    interleave_words = block_m_per_vec * interleave_words_per_row
    use_tile_pack_fast_path = round_sf
    use_persistent_fast_path = round_sf
    use_cross_wave_pipeline = hidden == FAST_PATH_HIDDEN and round_sf
    persistent_core_num = 24
    kernel_blocks = persistent_core_num if use_persistent_fast_path else m_num

    @T.macro
    def pack_e5m6_group_scalar(x_bits_ub, packed_ub, row, group):
        w0 = T.alloc_var("int32", init=0)
        w1 = T.alloc_var("int32", init=0)
        w2 = T.alloc_var("int32", init=0)
        for lane in T.unroll(8):
            col = group * 8 + lane
            x_bits = x_bits_ub[row, col]
            exp = (x_bits >> 23) & 0xFF
            e5m6 = T.alloc_var("int32", init=0)
            if exp >= 113 and exp <= 142:
                e5m6 = ((x_bits >> 20) & 0x800) | ((exp - 112) << 6) | ((x_bits >> 17) & 0x3F)
                if (e5m6 & 1) + (x_bits & 0x1FFFF) > 0x10000:
                    e5m6 = e5m6 + 1
            else:
                sign = (x_bits >> 16) & 0x8000
                mant = x_bits & 0x7FFFFF
                fp16_bits = T.alloc_var("int32", init=sign)
                if exp >= 113:
                    if exp < 255 or mant == 0:
                        fp16_bits = sign | 0x7C00
                    else:
                        fp16_bits = sign | 0x7FFF
                elif exp >= 103:
                    fp16_bits = sign | ((0x800000 | mant) >> (126 - exp))
                e5m6 = fp16_bits >> 4
                if (e5m6 & 1) + (x_bits & 0x1FFFF) > 0x10000:
                    e5m6 = e5m6 + 1
            h = e5m6 & 0xFFF
            if lane == 0:
                w0 = h << 20
            elif lane == 1:
                w0 = w0 | (h << 8)
            elif lane == 2:
                w0 = w0 | (h >> 4)
                w1 = h << 28
            elif lane == 3:
                w1 = w1 | (h << 16)
            elif lane == 4:
                w1 = w1 | (h << 4)
            elif lane == 5:
                w1 = w1 | (h >> 8)
                w2 = h << 24
            elif lane == 6:
                w2 = w2 | (h << 12)
            else:
                w2 = w2 | h
        packed_ub[row, group * 3] = w0
        packed_ub[row, group * 3 + 1] = w1
        packed_ub[row, group * 3 + 2] = w2

    @T.macro
    def pack_e5m6_tile_scalar(x_bits_ub, packed_ub):
        for i in T.serial(block_m_per_vec):
            for group_quad in T.serial(block_k // 32):
                for group_in_quad in T.unroll(4):
                    group = group_quad * 4 + group_in_quad
                    pack_e5m6_group_scalar(x_bits_ub, packed_ub, i, group)

    @T.macro
    def pack_e5m6_tile_fast(
        x_bits_ub,
        x_fp16_bits_ub,
        packed_ub,
        pack_offset_u32_ub,
        lane_bits_ub,
        lane_exp_ub,
        lane_e5m6_ub,
        pack_scratch0_ub,
        pack_scratch1_ub,
        pack_mask_exp_pair_ub,
        pack_const_exp30_pair_ub,
        pack_round_pair_ub,
        pack_mask_e5m6_pair_ub,
        exception_group_ub,
        exception_fp32_ub,
        exception_max_ub,
        packed_w0_ub,
        packed_w1_ub,
        packed_w2_ub,
        packed_words_i32_ub,
        interleave_out_i32_ub,
        interleave_offset_u32_ub,
    ):
        for _ in T.unroll(1):
            for pair in T.unroll(4):
                T.tile.gather(lane_bits_ub, x_fp16_bits_ub, pack_offset_u32_ub, pair * 4)
                T.tile.bitwise_and(lane_exp_ub, lane_bits_ub, pack_mask_exp_pair_ub)
                T.tile.add(pack_scratch0_ub, lane_exp_ub, -67109888)
                T.tile.sub(pack_scratch1_ub, pack_const_exp30_pair_ub, lane_exp_ub)
                if pair == 0:
                    T.tile.bitwise_or(exception_group_ub, pack_scratch0_ub, pack_scratch1_ub)
                else:
                    T.tile.bitwise_or(pack_scratch0_ub, pack_scratch0_ub, pack_scratch1_ub)
                    T.tile.bitwise_or(exception_group_ub, exception_group_ub, pack_scratch0_ub)
                T.tile.bitwise_rshift(pack_scratch0_ub, lane_bits_ub, 4)
                T.tile.bitwise_and(pack_scratch0_ub, pack_scratch0_ub, pack_round_pair_ub)
                T.tile.add(lane_exp_ub, lane_bits_ub, 458759)
                T.tile.add(lane_exp_ub, lane_exp_ub, pack_scratch0_ub)
                T.tile.bitwise_rshift(lane_exp_ub, lane_exp_ub, 4)
                T.tile.bitwise_and(lane_exp_ub, lane_exp_ub, pack_mask_e5m6_pair_ub)
                T.tile.bitwise_lshift(lane_e5m6_ub, lane_exp_ub, 16)
                T.tile.bitwise_rshift(lane_e5m6_ub, lane_e5m6_ub, 4)
                T.tile.bitwise_rshift(pack_scratch0_ub, lane_exp_ub, 16)
                T.tile.bitwise_or(lane_e5m6_ub, lane_e5m6_ub, pack_scratch0_ub)
                if pair == 0:
                    T.tile.bitwise_lshift(packed_w0_ub, lane_e5m6_ub, 8)
                elif pair == 1:
                    T.tile.bitwise_rshift(pack_scratch0_ub, lane_e5m6_ub, 16)
                    T.tile.bitwise_or(packed_w0_ub, packed_w0_ub, pack_scratch0_ub)
                    T.tile.bitwise_lshift(packed_w1_ub, lane_e5m6_ub, 16)
                elif pair == 2:
                    T.tile.bitwise_rshift(pack_scratch0_ub, lane_e5m6_ub, 8)
                    T.tile.bitwise_or(packed_w1_ub, packed_w1_ub, pack_scratch0_ub)
                    T.tile.bitwise_lshift(packed_w2_ub, lane_e5m6_ub, 24)
                else:
                    T.tile.bitwise_or(packed_w2_ub, packed_w2_ub, lane_e5m6_ub)
            T.tile.bitwise_rshift(exception_group_ub, exception_group_ub, 15)
            T.tile.bitwise_and(exception_group_ub, exception_group_ub, pack_round_pair_ub)
            T.tile.cast(exception_fp32_ub, exception_group_ub, mode="CAST_NONE", count=2 * fast_num_pack_groups)
            T.reduce_max(exception_fp32_ub, exception_max_ub, dim=-1, clear=True)
            T.set_flag("v", "s", 4)
            T.wait_flag("v", "s", 4)
            if exception_max_ub[0] == 0.0:
                T.copy(packed_w0_ub, packed_words_i32_ub[0:2, :])
                T.copy(packed_w1_ub, packed_words_i32_ub[2:4, :])
                T.copy(packed_w2_ub, packed_words_i32_ub[4:6, :])
                T.pipe_barrier("v")
                T.tile.gather(interleave_out_i32_ub, packed_words_i32_ub, interleave_offset_u32_ub, 0)
                T.pipe_barrier("v")
                T.copy(interleave_out_i32_ub[0, 0:block_m_per_vec, 0:interleave_words_per_row], packed_ub)
            else:
                for row in T.serial(block_m_per_vec):
                    for group_quad in T.serial(num_pack_groups // 4):
                        for group_in_quad in T.unroll(4):
                            group = group_quad * 4 + group_in_quad
                            flat_group = row * num_pack_groups + group
                            if exception_group_ub[0, flat_group] == 0:
                                packed_ub[row, group * 3] = packed_w0_ub[0, flat_group]
                                packed_ub[row, group * 3 + 1] = packed_w1_ub[0, flat_group]
                                packed_ub[row, group * 3 + 2] = packed_w2_ub[0, flat_group]
                            else:
                                pack_e5m6_group_scalar(x_bits_ub, packed_ub, row, group)
            T.set_flag("v", "s", 7)
            T.wait_flag("v", "s", 7)

    @T.prim_func(private=False)
    def per_token_cast_to_e5m6_kernel(
        x: T.Tensor([num_tokens, hidden], in_config.dtype),
        out: T.Tensor([num_tokens, hidden // 8 * 3], "int32"),
        out_sf: T.Tensor([num_tokens], kernel_sf_dtype),
    ):
        with T.Kernel(kernel_blocks, is_npu=True) as (cid, vid):
            row_offset = T.alloc_var("int32", init=0)
            next_row_offset = T.alloc_var("int32", init=0)
            scale = block_m_per_vec * block_k
            x_ub = T.alloc_ub((block_m_per_vec, block_k), in_config.dtype)
            x_ub_next = T.alloc_ub((block_m_per_vec, block_k), in_config.dtype)
            x_fp32_ub = T.alloc_ub((block_m_per_vec, block_k), "float32")
            x_fp16_ub = T.alloc_ub((block_m_per_vec, block_k), "float16")
            x_fp16_bits_ub = T.alloc_ub((block_m_per_vec, block_k // 2), "int32")
            x_abs_ub = T.alloc_ub((block_m_per_vec, block_k), "float32")
            local_max_ub = T.alloc_ub((block_m_per_vec,), "float32")
            amax_ub = T.alloc_ub((block_m_per_vec,), "float32")
            sf_inv_ub = T.alloc_ub((block_m_per_vec, 1), "float32")
            sf_inv_tile_ub = T.alloc_ub((block_m_per_vec, block_k), "float32")
            sf_fp32_ub = T.alloc_ub((block_m_per_vec,), "float32")
            sf_exp_i32_ub = T.alloc_ub((block_m_per_vec,), "int32")
            x_bits_ub = T.alloc_ub((block_m_per_vec, block_k), "int32")
            packed_ub = T.alloc_ub((block_m_per_vec, packed_block_k), "int32")
            packed_ub_next = T.alloc_ub((block_m_per_vec, packed_block_k), "int32")
            pack_offset_i32_ub = T.alloc_ub((fast_num_pack_groups,), "int32")
            pack_offset_u32_ub = T.alloc_ub((fast_num_pack_groups,), "uint32")
            lane_bits_ub = T.alloc_ub((2, fast_num_pack_groups), "int32")
            lane_exp_ub = T.alloc_ub((2, fast_num_pack_groups), "int32")
            lane_e5m6_ub = T.alloc_ub((2, fast_num_pack_groups), "int32")
            pack_scratch0_ub = T.alloc_ub((2, fast_num_pack_groups), "int32")
            pack_scratch1_ub = T.alloc_ub((2, fast_num_pack_groups), "int32")
            pack_mask_exp_pair_ub = T.alloc_ub((2, fast_num_pack_groups), "int32")
            pack_const_exp30_pair_ub = T.alloc_ub((2, fast_num_pack_groups), "int32")
            pack_round_pair_ub = T.alloc_ub((2, fast_num_pack_groups), "int32")
            pack_mask_e5m6_pair_ub = T.alloc_ub((2, fast_num_pack_groups), "int32")
            exception_group_ub = T.alloc_ub((2, fast_num_pack_groups), "int32")
            exception_fp32_ub = T.alloc_ub((2, fast_num_pack_groups), "float32")
            exception_max_ub = T.alloc_ub((2,), "float32")
            packed_w0_ub = T.alloc_ub((2, fast_num_pack_groups), "int32")
            packed_w1_ub = T.alloc_ub((2, fast_num_pack_groups), "int32")
            packed_w2_ub = T.alloc_ub((2, fast_num_pack_groups), "int32")
            packed_words_i32_ub = T.alloc_ub((6, fast_num_pack_groups), "int32")
            interleave_out_i32_ub = T.alloc_ub((2, block_m_per_vec, interleave_words_per_row), "int32")
            interleave_offset_u32_ub = T.alloc_ub((interleave_words,), "uint32")
            with T.Scope("V"):
                T.reinterpretcast(pack_offset_u32_ub, pack_offset_i32_ub, "uint32_t")
                T.reinterpretcast(x_fp16_bits_ub, x_fp16_ub, "int32_t")
                if use_tile_pack_fast_path:
                    T.tile.arith_progression(pack_offset_i32_ub, 0, 8 * 2, fast_num_pack_groups)
                    T.tile.fill(pack_mask_exp_pair_ub, 0x7C007C00)
                    T.tile.fill(pack_const_exp30_pair_ub, 0x78007800)
                    T.tile.fill(pack_round_pair_ub, 0x00010001)
                    T.tile.fill(pack_mask_e5m6_pair_ub, 0x0FFF0FFF)
                    for row in T.unroll(block_m_per_vec):
                        for group_in_row in T.unroll(interleave_groups):
                            for word in T.unroll(3):
                                interleave_offset_u32_ub[row * interleave_words_per_row + group_in_row * 3 + word] = (
                                    word * 2 * fast_num_pack_groups * 4 + (row * interleave_groups + group_in_row) * 4
                                )
                    T.set_flag("s", "v", 7)
                    T.wait_flag("s", "v", 7)
                    T.set_flag("v", "mte2", 0)
                    T.set_flag("v", "mte2", 1)
                    T.set_flag("mte3", "v", 0)
                    T.set_flag("mte3", "v", 1)
                    if use_cross_wave_pipeline:
                        row_offset = cid * block_m
                        if vid != 0:
                            row_offset = cid * block_m + block_m_per_vec
                        T.wait_flag("v", "mte2", 0)
                        T.copy(x[row_offset : row_offset + block_m_per_vec, 0:block_k], x_ub)
                        T.set_flag("mte2", "v", 0)
                for persistent_wave in T.serial(T.ceildiv(m_num, kernel_blocks)):
                    pid_token = persistent_wave * kernel_blocks + cid
                    row_offset = pid_token * block_m
                    if vid == 0:
                        row_offset = pid_token * block_m
                    else:
                        row_offset = pid_token * block_m + block_m_per_vec
                    T.tile.fill(amax_ub, 0.0)
                    for col_start in T.serial(n_num):
                        if use_cross_wave_pipeline:
                            if persistent_wave + 1 < T.ceildiv(m_num, kernel_blocks):
                                next_pid_token = (persistent_wave + 1) * kernel_blocks + cid
                                next_row_offset = next_pid_token * block_m
                                if vid != 0:
                                    next_row_offset = next_pid_token * block_m + block_m_per_vec
                                if persistent_wave % 2 == 0:
                                    T.wait_flag("v", "mte2", 1)
                                    T.copy(x[next_row_offset : next_row_offset + block_m_per_vec, 0:block_k], x_ub_next)
                                    T.set_flag("mte2", "v", 1)
                                else:
                                    T.wait_flag("v", "mte2", 0)
                                    T.copy(x[next_row_offset : next_row_offset + block_m_per_vec, 0:block_k], x_ub)
                                    T.set_flag("mte2", "v", 0)
                            if persistent_wave % 2 == 0:
                                T.wait_flag("mte2", "v", 0)
                                T.tile.cast(x_fp32_ub, x_ub, mode="CAST_NONE", count=scale)
                                T.set_flag("v", "mte2", 0)
                            else:
                                T.wait_flag("mte2", "v", 1)
                                T.tile.cast(x_fp32_ub, x_ub_next, mode="CAST_NONE", count=scale)
                                T.set_flag("v", "mte2", 1)
                        else:
                            if use_tile_pack_fast_path:
                                if col_start == 0:
                                    T.wait_flag("v", "mte2", 0)
                                    T.copy(x[row_offset : row_offset + block_m_per_vec, 0:block_k], x_ub)
                                    T.set_flag("mte2", "v", 0)
                                if col_start + 1 < n_num:
                                    if col_start % 2 == 0:
                                        T.wait_flag("v", "mte2", 1)
                                        T.copy(
                                            x[
                                                row_offset : row_offset + block_m_per_vec,
                                                (col_start + 1) * block_k : (col_start + 2) * block_k,
                                            ],
                                            x_ub_next,
                                        )
                                        T.set_flag("mte2", "v", 1)
                                    else:
                                        T.wait_flag("v", "mte2", 0)
                                        T.copy(
                                            x[
                                                row_offset : row_offset + block_m_per_vec,
                                                (col_start + 1) * block_k : (col_start + 2) * block_k,
                                            ],
                                            x_ub,
                                        )
                                        T.set_flag("mte2", "v", 0)
                                if col_start % 2 == 0:
                                    T.wait_flag("mte2", "v", 0)
                                    T.tile.cast(x_fp32_ub, x_ub, mode="CAST_NONE", count=scale)
                                    T.set_flag("v", "mte2", 0)
                                else:
                                    T.wait_flag("mte2", "v", 1)
                                    T.tile.cast(x_fp32_ub, x_ub_next, mode="CAST_NONE", count=scale)
                                    T.set_flag("v", "mte2", 1)
                            else:
                                T.copy(x[row_offset : row_offset + block_m_per_vec, col_start * block_k : (col_start + 1) * block_k], x_ub)
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                T.tile.cast(x_fp32_ub, x_ub, mode="CAST_NONE", count=scale)
                        T.tile.abs(x_abs_ub, x_fp32_ub)
                        T.pipe_barrier("v")
                        T.reduce_max(x_abs_ub, local_max_ub, dim=-1, clear=True)
                        T.pipe_barrier("v")
                        T.tile.max(amax_ub, amax_ub, local_max_ub)
                    T.set_flag("v", "s", 1)
                    T.wait_flag("v", "s", 1)
                    for i in T.unroll(block_m_per_vec):
                        clamped = T.max(amax_ub[i], out_config.clamp_min_value)
                        if round_sf:
                            amax_bits = T.reinterpret("int32", clamped)
                            candidate_exp = ((amax_bits >> 23) & 0xFF) - 142
                            exp_sf = T.alloc_var("int32", init=candidate_exp)
                            if (amax_bits & 0x7F800000) != 0x7F800000 and (amax_bits & 0x7F0000) == 0x7F0000:
                                exp_sf = candidate_exp + 1
                            sf_inv_exp = T.max(127 - exp_sf, 0)
                            sf_inv_bits = T.alloc_var("int32", init=sf_inv_exp << 23)
                            sf_inv_ub[i, 0] = T.reinterpret("float32", sf_inv_bits)
                            if use_packed_ue8m0:
                                sf_exp_i32_ub[i] = exp_sf + 127
                            else:
                                sf_bits = T.alloc_var("int32", init=(127 + exp_sf) << 23)
                                sf_fp32_ub[i] = T.reinterpret("float32", sf_bits)
                        else:
                            sf_val = clamped / T.float32(65024.0)
                            sf_inv_ub[i, 0] = T.float32(65024.0) / clamped
                            sf_fp32_ub[i] = sf_val
                    T.set_flag("s", "mte3", 2)
                    T.wait_flag("s", "mte3", 2)
                    if use_packed_ue8m0:
                        T.copy(sf_exp_i32_ub, out_sf[row_offset : row_offset + block_m_per_vec])
                    else:
                        T.copy(sf_fp32_ub, out_sf[row_offset : row_offset + block_m_per_vec])
                    T.set_flag("s", "v", 3)
                    T.wait_flag("s", "v", 3)
                    T.tile.broadcast(sf_inv_tile_ub, sf_inv_ub, axis=1)
                    T.pipe_barrier("v")
                    if single_chunk:
                        T.tile.mul(x_fp32_ub, x_fp32_ub, sf_inv_tile_ub)
                        T.tile.cast(x_fp16_ub, x_fp32_ub, mode="CAST_NONE", count=scale)
                        T.reinterpretcast(x_bits_ub, x_fp32_ub, "int32_t")
                        T.set_flag("v", "s", 4)
                        T.wait_flag("v", "s", 4)
                        if use_cross_wave_pipeline:
                            if persistent_wave % 2 == 0:
                                T.wait_flag("mte3", "v", 0)
                                pack_e5m6_tile_fast(
                                    x_bits_ub,
                                    x_fp16_bits_ub,
                                    packed_ub,
                                    pack_offset_u32_ub,
                                    lane_bits_ub,
                                    lane_exp_ub,
                                    lane_e5m6_ub,
                                    pack_scratch0_ub,
                                    pack_scratch1_ub,
                                    pack_mask_exp_pair_ub,
                                    pack_const_exp30_pair_ub,
                                    pack_round_pair_ub,
                                    pack_mask_e5m6_pair_ub,
                                    exception_group_ub,
                                    exception_fp32_ub,
                                    exception_max_ub,
                                    packed_w0_ub,
                                    packed_w1_ub,
                                    packed_w2_ub,
                                    packed_words_i32_ub,
                                    interleave_out_i32_ub,
                                    interleave_offset_u32_ub,
                                )
                            else:
                                T.wait_flag("mte3", "v", 1)
                                pack_e5m6_tile_fast(
                                    x_bits_ub,
                                    x_fp16_bits_ub,
                                    packed_ub_next,
                                    pack_offset_u32_ub,
                                    lane_bits_ub,
                                    lane_exp_ub,
                                    lane_e5m6_ub,
                                    pack_scratch0_ub,
                                    pack_scratch1_ub,
                                    pack_mask_exp_pair_ub,
                                    pack_const_exp30_pair_ub,
                                    pack_round_pair_ub,
                                    pack_mask_e5m6_pair_ub,
                                    exception_group_ub,
                                    exception_fp32_ub,
                                    exception_max_ub,
                                    packed_w0_ub,
                                    packed_w1_ub,
                                    packed_w2_ub,
                                    packed_words_i32_ub,
                                    interleave_out_i32_ub,
                                    interleave_offset_u32_ub,
                                )
                        else:
                            pack_e5m6_tile_scalar(x_bits_ub, packed_ub)
                        T.set_flag("s", "mte3", 5)
                        T.wait_flag("s", "mte3", 5)
                        if use_cross_wave_pipeline:
                            if persistent_wave % 2 == 0:
                                T.copy(packed_ub, out[row_offset : row_offset + block_m_per_vec, 0:packed_block_k])
                                T.set_flag("mte3", "v", 0)
                            else:
                                T.copy(packed_ub_next, out[row_offset : row_offset + block_m_per_vec, 0:packed_block_k])
                                T.set_flag("mte3", "v", 1)
                        else:
                            T.copy(packed_ub, out[row_offset : row_offset + block_m_per_vec, 0:packed_block_k])
                            T.set_flag("mte3", "s", 6)
                            T.wait_flag("mte3", "s", 6)
                    else:
                        for col_start in T.serial(n_num):
                            col_offset = col_start * block_k
                            if use_tile_pack_fast_path:
                                if col_start == 0:
                                    T.wait_flag("v", "mte2", 0)
                                    T.copy(x[row_offset : row_offset + block_m_per_vec, 0:block_k], x_ub)
                                    T.set_flag("mte2", "v", 0)
                                if col_start + 1 < n_num:
                                    if col_start % 2 == 0:
                                        T.wait_flag("v", "mte2", 1)
                                        T.copy(
                                            x[row_offset : row_offset + block_m_per_vec, col_offset + block_k : col_offset + 2 * block_k],
                                            x_ub_next,
                                        )
                                        T.set_flag("mte2", "v", 1)
                                    else:
                                        T.wait_flag("v", "mte2", 0)
                                        T.copy(
                                            x[row_offset : row_offset + block_m_per_vec, col_offset + block_k : col_offset + 2 * block_k],
                                            x_ub,
                                        )
                                        T.set_flag("mte2", "v", 0)
                                if col_start % 2 == 0:
                                    T.wait_flag("mte2", "v", 0)
                                    T.tile.cast(x_fp32_ub, x_ub, mode="CAST_NONE", count=scale)
                                    T.set_flag("v", "mte2", 0)
                                else:
                                    T.wait_flag("mte2", "v", 1)
                                    T.tile.cast(x_fp32_ub, x_ub_next, mode="CAST_NONE", count=scale)
                                    T.set_flag("v", "mte2", 1)
                            else:
                                T.copy(x[row_offset : row_offset + block_m_per_vec, col_offset : col_offset + block_k], x_ub)
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                T.tile.cast(x_fp32_ub, x_ub, mode="CAST_NONE", count=scale)
                            T.tile.mul(x_fp32_ub, x_fp32_ub, sf_inv_tile_ub)
                            T.tile.cast(x_fp16_ub, x_fp32_ub, mode="CAST_NONE", count=scale)
                            T.reinterpretcast(x_bits_ub, x_fp32_ub, "int32_t")
                            if use_tile_pack_fast_path:
                                if col_start % 2 == 0:
                                    T.wait_flag("mte3", "v", 0)
                                    pack_e5m6_tile_fast(
                                        x_bits_ub,
                                        x_fp16_bits_ub,
                                        packed_ub,
                                        pack_offset_u32_ub,
                                        lane_bits_ub,
                                        lane_exp_ub,
                                        lane_e5m6_ub,
                                        pack_scratch0_ub,
                                        pack_scratch1_ub,
                                        pack_mask_exp_pair_ub,
                                        pack_const_exp30_pair_ub,
                                        pack_round_pair_ub,
                                        pack_mask_e5m6_pair_ub,
                                        exception_group_ub,
                                        exception_fp32_ub,
                                        exception_max_ub,
                                        packed_w0_ub,
                                        packed_w1_ub,
                                        packed_w2_ub,
                                        packed_words_i32_ub,
                                        interleave_out_i32_ub,
                                        interleave_offset_u32_ub,
                                    )
                                else:
                                    T.wait_flag("mte3", "v", 1)
                                    pack_e5m6_tile_fast(
                                        x_bits_ub,
                                        x_fp16_bits_ub,
                                        packed_ub_next,
                                        pack_offset_u32_ub,
                                        lane_bits_ub,
                                        lane_exp_ub,
                                        lane_e5m6_ub,
                                        pack_scratch0_ub,
                                        pack_scratch1_ub,
                                        pack_mask_exp_pair_ub,
                                        pack_const_exp30_pair_ub,
                                        pack_round_pair_ub,
                                        pack_mask_e5m6_pair_ub,
                                        exception_group_ub,
                                        exception_fp32_ub,
                                        exception_max_ub,
                                        packed_w0_ub,
                                        packed_w1_ub,
                                        packed_w2_ub,
                                        packed_words_i32_ub,
                                        interleave_out_i32_ub,
                                        interleave_offset_u32_ub,
                                    )
                            else:
                                T.set_flag("v", "s", 4)
                                T.wait_flag("v", "s", 4)
                                pack_e5m6_tile_scalar(x_bits_ub, packed_ub)
                            packed_col_offset = col_start * packed_block_k
                            T.set_flag("s", "mte3", 5)
                            T.wait_flag("s", "mte3", 5)
                            if use_tile_pack_fast_path:
                                if col_start % 2 == 0:
                                    T.copy(
                                        packed_ub,
                                        out[
                                            row_offset : row_offset + block_m_per_vec,
                                            packed_col_offset : packed_col_offset + packed_block_k,
                                        ],
                                    )
                                    T.set_flag("mte3", "v", 0)
                                else:
                                    T.copy(
                                        packed_ub_next,
                                        out[
                                            row_offset : row_offset + block_m_per_vec,
                                            packed_col_offset : packed_col_offset + packed_block_k,
                                        ],
                                    )
                                    T.set_flag("mte3", "v", 1)
                            else:
                                T.copy(
                                    packed_ub,
                                    out[row_offset : row_offset + block_m_per_vec, packed_col_offset : packed_col_offset + packed_block_k],
                                )
                                T.set_flag("mte3", "s", 6)
                                T.wait_flag("mte3", "s", 6)
                if use_tile_pack_fast_path:
                    T.wait_flag("v", "mte2", 0)
                    T.wait_flag("v", "mte2", 1)
                if use_tile_pack_fast_path:
                    T.wait_flag("mte3", "v", 0)
                    T.wait_flag("mte3", "v", 1)

    return per_token_cast_to_e5m6_kernel


def per_token_cast_to_e5m6(
    x: torch.Tensor,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> tuple:
    assert x.dtype == torch.bfloat16
    assert x.dim() == 2 and x.is_contiguous()
    num_tokens, hidden = x.shape
    assert num_per_channels == hidden
    assert hidden % 8 == 0
    orig_num_tokens = num_tokens
    orig_device = x.device
    if orig_device.type != "npu":
        npu_device = torch.device("npu")
        x = x.to(npu_device)
    else:
        npu_device = orig_device
    x_data, _, in_config = get_cast_input_and_config(x, None)
    out_config = get_cast_output_config("e5m6", (1, num_per_channels), use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0, 1e-4)
    block_m = _get_block_m(hidden)
    padded_num_tokens = align_up(num_tokens, block_m * (PERSISTENT_CORE_NUM if round_sf else 1))
    if padded_num_tokens != num_tokens:
        x_padded = torch.empty((padded_num_tokens, hidden), dtype=x_data.dtype, device=npu_device)
        x_padded[:num_tokens, :] = x_data
        x_padded[num_tokens:, :] = 0
        x_data = x_padded
        num_tokens = padded_num_tokens
    kernel = get_per_token_cast_to_e5m6_kernel(hidden=hidden, in_config=in_config, out_config=out_config)
    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())
    out = torch.empty((num_tokens, hidden // 8 * 3), dtype=torch.int32, device=npu_device)
    kernel_sf_torch_dtype = torch.int32 if use_packed_ue8m0 else out_config.sf_torch_dtype
    out_sf = torch.zeros((num_tokens,), dtype=kernel_sf_torch_dtype, device=npu_device)
    if num_tokens > 0:
        kernel_result = kernel(x_data, out, out_sf)
        if isinstance(kernel_result, (tuple, list)):
            out, out_sf = kernel_result
        else:
            out = kernel_result
    out = out[:orig_num_tokens, :].view(torch.uint8)
    if use_tma_aligned_col_major_sf:
        sf_token_stride = align_up(orig_num_tokens, 4)
        if use_packed_ue8m0:
            sf_storage = out_sf[:sf_token_stride].view(torch.uint8).reshape(1, -1)
        else:
            sf_storage = out_sf[:sf_token_stride].reshape(1, sf_token_stride)
    else:
        sf_storage = out_sf[:orig_num_tokens].reshape(orig_num_tokens, 1)
    out_sf = cast_epilogue(sf_storage, orig_num_tokens, hidden, out_config)
    return out.to(orig_device), out_sf.to(orig_device)


def _right_shift_unsigned(x: torch.Tensor, shift: int | torch.Tensor) -> torch.Tensor:
    return (x >> shift) & ((1 << (32 - shift)) - 1)


def _float32_to_fp16_rtz_bits(x: torch.Tensor) -> torch.Tensor:
    x_bits = x.contiguous().view(torch.int32)
    sign = (x_bits >> 16) & 0x8000
    exp = (x_bits >> 23) & 0xFF
    mant = x_bits & 0x7FFFFF
    normal = (exp >= 113) & (exp <= 142)
    subnormal = (exp >= 103) & (exp <= 112)
    overflow = (exp > 142) & (exp < 255)
    underflow = exp < 103
    is_nan = exp == 255
    exp_f16 = (exp - 112).to(torch.int32)
    mant_f16 = (mant >> 13).to(torch.int32)
    shift = (113 - exp).to(torch.int32)
    mant_sub = _right_shift_unsigned(0x800000 | mant, shift + 13)
    result = sign.to(torch.int32)
    result = torch.where(normal, result | (exp_f16 << 10) | mant_f16, result)
    result = torch.where(subnormal, result | mant_sub, result)
    result = torch.where(overflow | (is_nan & (mant == 0)), result | 0x7C00, result)
    result = torch.where(is_nan & (mant != 0), result | 0x7FFF, result)
    result = torch.where(underflow, sign.to(torch.int32), result)
    return result.to(torch.uint16)


def _cast_to_e5m6_ref(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.dtype in (torch.float32, torch.bfloat16)
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float32)
    num_tokens, hidden = x.shape
    assert hidden % 8 == 0
    x_bits = x.contiguous().view(torch.int32)
    fp16_bits = _float32_to_fp16_rtz_bits(x)
    remain_bits = x_bits & 0x1FFFF
    e5m6_bits = _right_shift_unsigned(fp16_bits.to(torch.int32), 4)
    lsb = e5m6_bits & 1
    should_round = (lsb.to(torch.int64) + remain_bits.to(torch.int64)) > 0x10000
    e5m6 = ((e5m6_bits + should_round.to(torch.int32)) & 0xFFF).to(torch.int64)
    e5m6 = e5m6.view(num_tokens, hidden // 8, 8)
    lanes = [e5m6[..., lane] for lane in range(8)]
    word0 = (lanes[0] << 20) | (lanes[1] << 8) | (lanes[2] >> 4)
    word1 = (lanes[2] << 28) | (lanes[3] << 16) | (lanes[4] << 4) | (lanes[5] >> 8)
    word2 = (lanes[5] << 24) | (lanes[6] << 12) | lanes[7]
    packed = torch.stack([word0, word1, word2], dim=-1).to(torch.uint32)
    return packed.view(num_tokens, hidden // 8 * 3).view(torch.uint8)


def _make_col_major(sf: torch.Tensor, tma_alignment: int) -> torch.Tensor:
    num_tokens, num_groups = sf.shape
    padded_tokens = align_up(num_tokens, tma_alignment)
    storage = torch.zeros(num_groups, padded_tokens, dtype=sf.dtype, device=sf.device)
    storage[:, :num_tokens] = sf.T
    return storage.T[:num_tokens, :]


def cast_to_e5m6_ref(
    x: torch.Tensor,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 2
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float32)
    num_tokens, hidden = x.shape
    assert hidden % num_per_channels == 0
    assert hidden % 8 == 0
    num_groups = hidden // num_per_channels
    max_value = torch.tensor(65024.0, dtype=torch.float32, device=x.device)
    x_view = x.view(num_tokens, num_groups, num_per_channels)
    amax = x_view.abs().amax(dim=-1).clamp(min=1e-4)
    dequant_sf = amax / max_value
    dequant_sf_int = dequant_sf.view(torch.int32)
    if round_sf:
        exp_sf = ((dequant_sf_int - 1) >> 23) + 1 - 127
        sf_inv_bits = (127 - exp_sf).clamp(min=0) << 23
        sf_inv = sf_inv_bits.view(torch.float32)
        sf_inv = torch.where(dequant_sf_int == 0, torch.zeros_like(sf_inv), sf_inv)
    else:
        exp_sf = None
        sf_inv = torch.where(dequant_sf_int == 0, torch.zeros_like(amax), max_value / amax)
    x_scaled = x * sf_inv.unsqueeze(-1).expand(num_tokens, num_groups, num_per_channels).reshape(num_tokens, hidden)
    packed = _cast_to_e5m6_ref(x_scaled)
    if use_packed_ue8m0:
        if round_sf:
            sf_raw = (exp_sf + 127).to(torch.uint8)
        else:
            sf_raw = ((dequant_sf_int >> 23) & 0xFF).to(torch.uint8)
        padded_groups = align_up(num_groups, 4)
        if padded_groups != num_groups:
            sf_padded = torch.zeros(num_tokens, padded_groups, dtype=sf_raw.dtype, device=sf_raw.device)
            sf_padded[:, :num_groups] = sf_raw
            sf_raw = sf_padded
        sf_out = sf_raw.view(torch.int32)
    elif round_sf:
        sf_out = ((exp_sf + 127) << 23).view(torch.float32)
    else:
        sf_out = dequant_sf
    if use_tma_aligned_col_major_sf:
        sf_out = _make_col_major(sf_out, 4)
    return packed, sf_out


def _generate_num_tokens(is_benchmark: bool = False) -> list[int]:
    full_test = os.getenv("TK_FULL_TEST") in ("1", "true", "True")
    if full_test and not is_benchmark:
        return [0, 4001, 8001]
    return [4001, 8001]


def _generate_hidden_sizes(align: int = 64) -> list[int]:
    return [hidden for hidden in (576, 2048, 2560, 3072, 4096, 6144, 7168) if hidden % align == 0]


def _generate_e5m6_inputs(num_tokens: int, hidden: int, dtype: torch.dtype) -> Iterable[tuple[torch.Tensor, bool]]:
    yield torch.randn((num_tokens, hidden), dtype=dtype, device="npu"), False
    for value in (2**-20, 2**-14 * 63 / 64, 2**-14):
        x = torch.full((num_tokens, hidden), value, dtype=dtype, device="npu")
        if num_tokens > 0:
            x[:, -1] = 65024.0
        yield x, True


def _clear_unused_sf(sf: torch.Tensor, hidden: int, num_per_channels: int) -> torch.Tensor:
    num_channel_blocks = ceil_div(hidden, num_per_channels)
    aligned_num_channel_blocks = align_up(num_channel_blocks, 4)
    sf_flattened = sf.contiguous().flatten().view(torch.uint8)
    sf_flattened = sf_flattened.view(-1, aligned_num_channel_blocks)
    sf_flattened[:, num_channel_blocks:] = 0
    return sf_flattened


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.numel() == 0 or actual.stride() == expected.stride()
    assert torch.equal(actual.contiguous().flatten().view(torch.uint8), expected.contiguous().flatten().view(torch.uint8))


def generate_test_params(is_benchmark: bool = False) -> list[dict]:
    return [
        {
            "num_tokens": num_tokens,
            "hidden": hidden,
            "use_tma_aligned_col_major_sf": use_tma,
            "round_sf": round_sf,
            "use_packed_ue8m0": use_packed,
            "in_dtype": torch.bfloat16,
        }
        for num_tokens in _generate_num_tokens(is_benchmark=is_benchmark)
        for hidden in _generate_hidden_sizes()
        for use_tma, round_sf, use_packed in ((False, True, False), (True, True, True))
    ]


def _make_param_id(params: dict) -> str:
    return "-".join(f"{key}={value}" for key, value in params.items())


@pytest.mark.parametrize("params", generate_test_params(is_benchmark=False), ids=_make_param_id)
def test_per_token_cast_to_e5m6_npu(params: dict) -> None:
    num_tokens = params["num_tokens"]
    hidden = params["hidden"]
    use_tma = params["use_tma_aligned_col_major_sf"]
    round_sf = params["round_sf"]
    use_packed = params["use_packed_ue8m0"]
    num_per_channels = hidden
    for x, _ in _generate_e5m6_inputs(num_tokens, hidden, params["in_dtype"]):
        actual, actual_sf = per_token_cast_to_e5m6(
            x, num_per_channels=num_per_channels, use_tma_aligned_col_major_sf=use_tma, round_sf=round_sf, use_packed_ue8m0=use_packed
        )
        torch.npu.synchronize()
        expected, expected_sf = cast_to_e5m6_ref(
            x.cpu(), num_per_channels=num_per_channels, use_tma_aligned_col_major_sf=use_tma, round_sf=round_sf, use_packed_ue8m0=use_packed
        )
        actual = actual.cpu()
        actual_sf = actual_sf.cpu()
        if use_packed:
            actual_sf = _clear_unused_sf(actual_sf, hidden, num_per_channels)
            expected_sf = _clear_unused_sf(expected_sf, hidden, num_per_channels)
        _assert_equal(actual, expected)
        _assert_equal(actual_sf, expected_sf)


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v"])
    if exit_code == pytest.ExitCode.OK:
        print("All per_token_cast_to_e5m6 tests passed! Kernel Output Match!")
    sys.exit(exit_code)
