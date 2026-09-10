# ruff: noqa: SIM102
# fmt: off

import os
from typing import Callable, Optional, Union

import tilelang
import tilelang.language as T
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

tilelang.cache.clear_cache()

auto_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True
    }

# DMA stage reuse is explicit. Dynamic route addresses still need automatic
# scalar/Vector dependency insertion to keep the event ring live.
unweighted_unscaled_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
h576_fp32_weighted_unscaled_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True}


@T.macro
def process_scaled_route(
    mapping_ub,
    weights_ub,
    route_base,
    route,
    dout_f32_buf,
    x_input_buf,
    x_f32_buf,
    dx_f32_buf,
    dx_output_buf,
    dot_buf,
    dot_sum_buf,
    dtopk_buf,
    dx_gm,
    dx_sf_gm,
    sf_gm,
    x_sf_gm,
    dsf_ref,
    hidden,
    need_input_cast,
    need_dx_cast,
    with_weights,
    with_sf,
    with_x_sf,
):
    pos = mapping_ub[route_base + route]
    if pos >= 0:
        dx_scale = 1.0
        dweight_scale = 1.0
        dx_sf_scale = 1.0
        dsf_scale = 1.0
        if with_weights:
            route_weight = weights_ub[route_base + route]
            dx_scale = dx_scale * route_weight
            dx_sf_scale = dx_sf_scale * route_weight
            dsf_scale = dsf_scale * route_weight
        if with_x_sf:
            route_x_sf = x_sf_gm[pos]
            dx_scale = dx_scale * route_x_sf
            dweight_scale = dweight_scale * route_x_sf
            dsf_scale = dsf_scale * route_x_sf
        if with_sf:
            route_sf = sf_gm[0]
            dx_scale = dx_scale * route_sf
            dweight_scale = dweight_scale * route_sf
            dx_sf_scale = dx_sf_scale * route_sf
        if with_weights and not with_sf and not with_x_sf:
            T.tile.mul(dx_f32_buf, dout_f32_buf, weights_ub[route_base + route])
        else:
            T.tile.mul(dx_f32_buf, dout_f32_buf, dx_scale)
        if need_dx_cast:
            T.tile.cast(dx_output_buf, dx_f32_buf, "CAST_RINT", hidden)
            T.copy(dx_output_buf, dx_gm[pos, 0:hidden])
        else:
            T.copy(dx_f32_buf, dx_gm[pos, 0:hidden])
        if need_input_cast:
            T.tile.cast(x_f32_buf, x_input_buf, "CAST_NONE", hidden)
            T.tile.mul(dot_buf, dout_f32_buf, x_f32_buf)
        else:
            T.tile.mul(dot_buf, dout_f32_buf, x_input_buf)
        T.reduce_sum(dot_buf, dot_sum_buf, dim=-1)
        if with_weights:
            dtopk_buf[route_base + route] = dot_sum_buf[0] * dweight_scale
        if with_x_sf:
            dx_sf_gm[pos] = dot_sum_buf[0] * dx_sf_scale
        if with_sf:
            dsf_ref = dsf_ref + dot_sum_buf[0] * dsf_scale


@tilelang.jit(pass_configs=auto_pass_configs)
def get_reduce_fused_backward_other_k_fallback_kernel(hidden: int, num_topk: int, in_dtype: torch.dtype, out_dtype: torch.dtype, with_sf: bool, with_weights: bool, with_x_sf: bool):
    num_tokens = T.symbolic("num_tokens")
    num_expanded_tokens = T.symbolic("num_expanded_tokens")

    dtype_map = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}
    tl_in_dtype = dtype_map.get(in_dtype, "float32")
    tl_out_dtype = dtype_map.get(out_dtype, "float32")
    need_input_cast = in_dtype != torch.float32
    need_output_cast = out_dtype != torch.float32
    need_dx_cast = in_dtype != torch.float32

    @T.prim_func
    def reduce_fused_backward_other_k_fallback(
        x: T.Tensor((num_expanded_tokens, hidden), tl_in_dtype),
        topk_weights: T.Tensor((num_tokens, num_topk), "float"),
        token_topk_to_pos: T.Tensor((num_tokens, num_topk), "int32"),
        sf: T.Tensor((1,), "float"),
        x_sf: T.Tensor((num_expanded_tokens,), "float"),
        dout: T.Tensor((num_tokens, hidden), tl_out_dtype),
        dx: T.Tensor((num_expanded_tokens, hidden), tl_in_dtype),
        dtopk_weights: T.Tensor((num_tokens, num_topk), "float"),
        dx_sf_route: T.Tensor((num_tokens, num_topk), "float"),
        dsf: T.Tensor((1,), "float"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _), T.Scope("V"):
            dout_ub = T.alloc_ub((hidden,), "float")
            dout_input_ub = T.alloc_ub((hidden,), tl_out_dtype)
            x_ub = T.alloc_ub((hidden,), "float")
            x_input_ub = T.alloc_ub((hidden,), tl_in_dtype)
            topk_weights_ub = T.alloc_ub((num_topk,), "float")
            topk_to_pos_ub = T.alloc_ub((num_topk,), "int32")
            dx_ub = T.alloc_ub((hidden,), "float")
            dx_output_ub = T.alloc_ub((hidden,), tl_in_dtype)
            dot_ub = T.alloc_ub((hidden,), "float")
            dot_sum_ub = T.alloc_ub((1,), "float")
            scale_ub = T.alloc_ub((hidden,), "float")
            dtopk_weights_out_ub = T.alloc_ub((num_topk,), "float")
            dx_sf_route_out_ub = T.alloc_ub((num_topk,), "float")

            pos = T.alloc_var("int32", init=-1)
            dot_val = T.alloc_var("float", init=0.0)
            scale_val = T.alloc_var("float", init=1.0)
            dx_sf_val = T.alloc_var("float", init=0.0)
            dsf_val = T.alloc_var("float", init=0.0)

            dsf_val = 0.0
            for token in T.serial(num_tokens):
                for k in T.serial(num_topk):
                    dtopk_weights_out_ub[k] = 0.0
                    dx_sf_route_out_ub[k] = 0.0

                if need_output_cast:
                    T.copy(dout[token, :], dout_input_ub)
                    T.tile.cast(dout_ub, dout_input_ub, "CAST_NONE", hidden)
                else:
                    T.copy(dout[token, :], dout_ub)

                if with_weights:
                    T.copy(topk_weights[token, :], topk_weights_ub)
                T.copy(token_topk_to_pos[token, :], topk_to_pos_ub)

                for k in T.serial(num_topk):
                    pos = topk_to_pos_ub[k]
                    if pos >= 0:
                        if need_input_cast:
                            T.copy(x[pos, 0:hidden], x_input_ub)
                            T.tile.cast(x_ub, x_input_ub, "CAST_NONE", hidden)
                        else:
                            T.copy(x[pos, 0:hidden], x_ub)

                        T.tile.mul(dot_ub, dout_ub, x_ub)
                        T.reduce_sum(dot_ub, dot_sum_ub, dim=-1)
                        dot_val = dot_sum_ub[0]

                        T.tile.fill(scale_ub, 1.0)
                        if with_weights:
                            T.tile.mul(scale_ub, scale_ub, topk_weights_ub[k])
                        if with_x_sf:
                            T.tile.mul(scale_ub, scale_ub, x_sf[pos])
                        if with_sf:
                            T.tile.mul(scale_ub, scale_ub, sf[0])
                        T.tile.mul(dx_ub, dout_ub, scale_ub)

                        if need_dx_cast:
                            T.tile.cast(dx_output_ub, dx_ub, "CAST_RINT", hidden)
                            T.copy(dx_output_ub, dx[pos, 0:hidden])
                        else:
                            T.copy(dx_ub, dx[pos, 0:hidden])

                        if with_weights:
                            scale_val = 1.0
                            if with_x_sf:
                                scale_val = scale_val * x_sf[pos]
                            if with_sf:
                                scale_val = scale_val * sf[0]
                            dtopk_weights_out_ub[k] = dot_val * scale_val

                        if with_sf:
                            scale_val = 1.0
                            if with_weights:
                                scale_val = scale_val * topk_weights_ub[k]
                            if with_x_sf:
                                scale_val = scale_val * x_sf[pos]
                            dsf_val = dsf_val + dot_val * scale_val

                        if with_x_sf:
                            scale_val = 1.0
                            if with_weights:
                                scale_val = scale_val * topk_weights_ub[k]
                            if with_sf:
                                scale_val = scale_val * sf[0]
                            dx_sf_val = dot_val * scale_val
                            dx_sf_route_out_ub[k] = dx_sf_val

                T.copy(dtopk_weights_out_ub, dtopk_weights[token, :])
                T.copy(dx_sf_route_out_ub, dx_sf_route[token, :])
            dsf[0] = dsf_val

    return reduce_fused_backward_other_k_fallback


@tilelang.jit(pass_configs=unweighted_unscaled_pass_configs)
def get_reduce_fused_backward_k2_k6_k8_k9_unweighted_unscaled_kernel(hidden: int, num_topk: int, in_dtype: torch.dtype, out_dtype: torch.dtype):
    """Pipelined backward for the unweighted, unscaled reduction."""
    num_tokens = T.symbolic("num_tokens")
    num_expanded_tokens = T.symbolic("num_expanded_tokens")
    num_cores = 24
    num_owners = num_cores * 2
    rows_per_block = 4 if hidden <= 2048 else 2
    tokens_per_owner = T.ceildiv(num_tokens, num_owners)
    blocks_per_owner = T.ceildiv(tokens_per_owner, rows_per_block)
    metadata_count = rows_per_block * num_topk
    metadata_aligned = ((metadata_count + 7) // 8) * 8
    extra_row_extent = hidden if rows_per_block == 4 else 1
    STAGE0 = 0
    STAGE1 = 1

    dtype_map = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}
    tl_in_dtype = dtype_map.get(in_dtype, "float32")
    tl_out_dtype = dtype_map.get(out_dtype, "float32")
    if tl_in_dtype != tl_out_dtype:
        raise ValueError("unweighted/unscaled path requires matching dout/dx dtypes")

    @T.macro
    def load_stage(mapping_gm, dout_gm, mapping_ub, dout_buf0, dout_buf1, dout_buf2, dout_buf3, token_base, token_limit, stage):
        for row in T.unroll(rows_per_block):
            for route in T.unroll(num_topk):
                mapping_ub[row * num_topk + route] = -1
        T.wait_flag("mte3", "mte2", stage)
        if token_base + rows_per_block <= token_limit:
            T.copy(mapping_gm[token_base * num_topk : token_base * num_topk + metadata_count], mapping_ub[0:metadata_count])
        else:
            T.copy(mapping_gm[token_base * num_topk : token_limit * num_topk], mapping_ub[0:metadata_count], pad_value=-1)
        if token_base < token_limit:
            T.copy(dout_gm[token_base, 0:hidden], dout_buf0)
        if token_base + 1 < token_limit:
            T.copy(dout_gm[token_base + 1, 0:hidden], dout_buf1)
        if rows_per_block == 4:
            if token_base + 2 < token_limit:
                T.copy(dout_gm[token_base + 2, 0:hidden], dout_buf2)
            if token_base + 3 < token_limit:
                T.copy(dout_gm[token_base + 3, 0:hidden], dout_buf3)
        T.set_flag("mte2", "v", stage)

    @T.macro
    def scatter_token(dout_ub, mapping_ub, mapping_row, dx_gm):
        for route in T.unroll(num_topk):
            pos = mapping_ub[mapping_row * num_topk + route]
            if pos >= 0:
                T.copy(dout_ub, dx_gm[pos, 0:hidden])

    @T.macro
    def begin_store_stage(dout_buf0, mapping_ub, dx_gm, stage):
        T.wait_flag("mte2", "v", stage)
        T.set_flag("v", "mte3", stage)
        T.wait_flag("v", "mte3", stage)
        pos = mapping_ub[0]
        if pos >= 0:
            T.copy(dout_buf0, dx_gm[pos, 0:hidden])

    @T.macro
    def finish_store_stage(dout_buf0, dout_buf1, dout_buf2, dout_buf3, mapping_ub, dx_gm, stage):
        for route in T.unroll(1, num_topk):
            pos = mapping_ub[route]
            if pos >= 0:
                T.copy(dout_buf0, dx_gm[pos, 0:hidden])
        scatter_token(dout_buf1, mapping_ub, 1, dx_gm)
        if rows_per_block == 4:
            scatter_token(dout_buf2, mapping_ub, 2, dx_gm)
            scatter_token(dout_buf3, mapping_ub, 3, dx_gm)
        T.pipe_barrier("mte3")
        T.set_flag("mte3", "mte2", stage)

    @T.prim_func
    def reduce_fused_backward_k2_k6_k8_k9_unweighted_unscaled(
        x: T.Tensor((num_expanded_tokens, hidden), tl_in_dtype),
        topk_weights: T.Tensor((num_tokens * num_topk,), "float"),
        token_topk_to_pos: T.Tensor((num_tokens * num_topk,), "int32"),
        sf: T.Tensor((1,), "float"),
        x_sf: T.Tensor((num_expanded_tokens,), "float"),
        dout: T.Tensor((num_tokens, hidden), tl_out_dtype),
        dx: T.Tensor((num_expanded_tokens, hidden), tl_in_dtype),
        dtopk_weights: T.Tensor((num_tokens * num_topk,), "float"),
        dx_sf: T.Tensor((num_expanded_tokens,), "float"),
        dsf: T.Tensor((1,), "float"),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid), T.Scope("V"):
            mapping_ub0 = T.alloc_ub((metadata_aligned,), "int32")
            mapping_ub1 = T.alloc_ub((metadata_aligned,), "int32")
            dout_ub00 = T.alloc_ub((hidden,), tl_out_dtype)
            dout_ub01 = T.alloc_ub((hidden,), tl_out_dtype)
            dout_ub02 = T.alloc_ub((extra_row_extent,), tl_out_dtype)
            dout_ub03 = T.alloc_ub((extra_row_extent,), tl_out_dtype)
            dout_ub10 = T.alloc_ub((hidden,), tl_out_dtype)
            dout_ub11 = T.alloc_ub((hidden,), tl_out_dtype)
            dout_ub12 = T.alloc_ub((extra_row_extent,), tl_out_dtype)
            dout_ub13 = T.alloc_ub((extra_row_extent,), tl_out_dtype)

            owner = cid * 2 + vid
            owner_token_base = owner * tokens_per_owner
            owner_token_end = T.min(owner_token_base + tokens_per_owner, num_tokens)

            T.set_flag("mte3", "mte2", STAGE0)
            T.set_flag("mte3", "mte2", STAGE1)

            if owner_token_base < owner_token_end:
                load_stage(token_topk_to_pos, dout, mapping_ub0, dout_ub00, dout_ub01, dout_ub02, dout_ub03, owner_token_base, owner_token_end, STAGE0)

            for block in T.serial(blocks_per_owner):
                token_base = owner_token_base + block * rows_per_block
                if token_base < owner_token_end:
                    next_token_base = token_base + rows_per_block
                    if block % 2 == 0:
                        begin_store_stage(dout_ub00, mapping_ub0, dx, STAGE0)
                        if next_token_base < owner_token_end:
                            load_stage(token_topk_to_pos, dout, mapping_ub1, dout_ub10, dout_ub11, dout_ub12, dout_ub13, next_token_base, owner_token_end, STAGE1)
                        finish_store_stage(dout_ub00, dout_ub01, dout_ub02, dout_ub03, mapping_ub0, dx, STAGE0)
                    else:
                        begin_store_stage(dout_ub10, mapping_ub1, dx, STAGE1)
                        if next_token_base < owner_token_end:
                            load_stage(token_topk_to_pos, dout, mapping_ub0, dout_ub00, dout_ub01, dout_ub02, dout_ub03, next_token_base, owner_token_end, STAGE0)
                        finish_store_stage(dout_ub10, dout_ub11, dout_ub12, dout_ub13, mapping_ub1, dx, STAGE1)

            T.wait_flag("mte3", "mte2", STAGE0)
            T.wait_flag("mte3", "mte2", STAGE1)
            T.pipe_barrier("ALL")

    return reduce_fused_backward_k2_k6_k8_k9_unweighted_unscaled


@tilelang.jit(pass_configs=auto_pass_configs)
def get_reduce_fused_backward_k2_h_le_3072_kernel(hidden: int, in_dtype: torch.dtype, out_dtype: torch.dtype, with_sf: bool, with_weights: bool, with_x_sf: bool):
    """Compact two-token pipeline for weighted/scaled K=2 backward."""
    num_tokens = T.symbolic("num_tokens")
    num_expanded_tokens = T.symbolic("num_expanded_tokens")
    num_topk = 2
    num_cores = 24
    num_owners = num_cores * 2
    rows_per_block = 2
    tokens_per_owner = T.ceildiv(num_tokens, num_owners)
    blocks_per_owner = T.ceildiv(tokens_per_owner, rows_per_block)
    metadata_count = rows_per_block * num_topk
    metadata_aligned = 8

    dtype_map = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}
    tl_in_dtype = dtype_map.get(in_dtype, "float32")
    tl_out_dtype = dtype_map.get(out_dtype, "float32")
    need_input_cast = in_dtype != torch.float32
    need_dout_cast = out_dtype != torch.float32
    need_dx_cast = in_dtype != torch.float32
    x_cast_extent = hidden if need_input_cast else 1
    dout_cast_extent = hidden if need_dout_cast else 1
    dx_cast_extent = hidden if need_dx_cast else 1

    @T.macro
    def load_dout(dout_gm, token, dout_input_buf, dout_f32_buf):
        if need_dout_cast:
            T.copy(dout_gm[token, 0:hidden], dout_input_buf)
            T.tile.cast(dout_f32_buf, dout_input_buf, "CAST_NONE", hidden)
        else:
            T.copy(dout_gm[token, 0:hidden], dout_f32_buf)

    @T.prim_func
    def reduce_fused_backward_k2_h_le_3072(
        x: T.Tensor((num_expanded_tokens, hidden), tl_in_dtype),
        topk_weights: T.Tensor((num_tokens * num_topk,), "float"),
        token_topk_to_pos: T.Tensor((num_tokens * num_topk,), "int32"),
        sf: T.Tensor((1,), "float"),
        x_sf: T.Tensor((num_expanded_tokens,), "float"),
        dout: T.Tensor((num_tokens, hidden), tl_out_dtype),
        dx: T.Tensor((num_expanded_tokens, hidden), tl_in_dtype),
        dtopk_weights: T.Tensor((num_tokens * num_topk,), "float"),
        dx_sf: T.Tensor((num_expanded_tokens,), "float"),
        dsf: T.Tensor((1,), "float"),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid), T.Scope("V"):
            mapping_ub = T.alloc_ub((metadata_aligned,), "int32")
            weights_ub = T.alloc_ub((metadata_aligned,), "float")
            dtopk_ub = T.alloc_ub((metadata_aligned,), "float")
            dout_input_ub0 = T.alloc_ub((dout_cast_extent,), tl_out_dtype)
            dout_input_ub1 = T.alloc_ub((dout_cast_extent,), tl_out_dtype)
            dout_f32_ub0 = T.alloc_ub((hidden,), "float")
            dout_f32_ub1 = T.alloc_ub((hidden,), "float")
            x_input_ub0 = T.alloc_ub((hidden,), tl_in_dtype)
            x_input_ub1 = T.alloc_ub((hidden,), tl_in_dtype)
            x_input_ub2 = T.alloc_ub((hidden,), tl_in_dtype)
            x_input_ub3 = T.alloc_ub((hidden,), tl_in_dtype)
            x_f32_ub = T.alloc_ub((x_cast_extent,), "float")
            dx_f32_ub0 = T.alloc_ub((hidden,), "float")
            dx_f32_ub1 = T.alloc_ub((hidden,), "float")
            dx_f32_ub2 = T.alloc_ub((hidden,), "float")
            dx_f32_ub3 = T.alloc_ub((hidden,), "float")
            dx_output_ub0 = T.alloc_ub((dx_cast_extent,), tl_in_dtype)
            dx_output_ub1 = T.alloc_ub((dx_cast_extent,), tl_in_dtype)
            dx_output_ub2 = T.alloc_ub((dx_cast_extent,), tl_in_dtype)
            dx_output_ub3 = T.alloc_ub((dx_cast_extent,), tl_in_dtype)
            dot_ub = T.alloc_ub((hidden,), "float")
            dot_sum_ub = T.alloc_ub((1,), "float")
            dsf_local_ub = T.alloc_ub((1,), "float")

            owner = cid * 2 + vid
            owner_token_base = owner * tokens_per_owner
            owner_token_end = T.min(owner_token_base + tokens_per_owner, num_tokens)
            dsf_local = T.alloc_var("float32", init=0.0)
            dsf_local = 0.0

            for block in T.serial(blocks_per_owner):
                token0 = owner_token_base + block * rows_per_block
                if token0 < owner_token_end:
                    token1 = token0 + 1
                    metadata_offset = token0 * num_topk
                    T.copy(token_topk_to_pos[metadata_offset : metadata_offset + metadata_count], mapping_ub[0:metadata_count], pad_value=-1)
                    if with_weights:
                        T.copy(topk_weights[metadata_offset : metadata_offset + metadata_count], weights_ub[0:metadata_count], pad_value=0.0)
                        T.tile.fill(dtopk_ub, 0.0)

                    load_dout(dout, token0, dout_input_ub0, dout_f32_ub0)
                    if token1 < owner_token_end:
                        load_dout(dout, token1, dout_input_ub1, dout_f32_ub1)

                    pos0 = mapping_ub[0]
                    if pos0 >= 0:
                        T.copy(x[pos0, 0:hidden], x_input_ub0)
                    pos1 = mapping_ub[1]
                    if pos1 >= 0:
                        T.copy(x[pos1, 0:hidden], x_input_ub1)

                    process_scaled_route(mapping_ub, weights_ub, 0, 0, dout_f32_ub0, x_input_ub0, x_f32_ub, dx_f32_ub0, dx_output_ub0, dot_ub, dot_sum_ub, dtopk_ub, dx, dx_sf, sf, x_sf, dsf_local, hidden, need_input_cast, need_dx_cast, with_weights, with_sf, with_x_sf)

                    if token1 < owner_token_end:
                        pos2 = mapping_ub[2]
                        if pos2 >= 0:
                            T.copy(x[pos2, 0:hidden], x_input_ub2)

                    process_scaled_route(mapping_ub, weights_ub, 0, 1, dout_f32_ub0, x_input_ub1, x_f32_ub, dx_f32_ub1, dx_output_ub1, dot_ub, dot_sum_ub, dtopk_ub, dx, dx_sf, sf, x_sf, dsf_local, hidden, need_input_cast, need_dx_cast, with_weights, with_sf, with_x_sf)

                    if token1 < owner_token_end:
                        pos3 = mapping_ub[3]
                        if pos3 >= 0:
                            T.copy(x[pos3, 0:hidden], x_input_ub3)
                        process_scaled_route(mapping_ub, weights_ub, 0, 2, dout_f32_ub1, x_input_ub2, x_f32_ub, dx_f32_ub2, dx_output_ub2, dot_ub, dot_sum_ub, dtopk_ub, dx, dx_sf, sf, x_sf, dsf_local, hidden, need_input_cast, need_dx_cast, with_weights, with_sf, with_x_sf)
                        process_scaled_route(mapping_ub, weights_ub, 0, 3, dout_f32_ub1, x_input_ub3, x_f32_ub, dx_f32_ub3, dx_output_ub3, dot_ub, dot_sum_ub, dtopk_ub, dx, dx_sf, sf, x_sf, dsf_local, hidden, need_input_cast, need_dx_cast, with_weights, with_sf, with_x_sf)

                    if with_weights:
                        if token1 < owner_token_end:
                            T.copy(dtopk_ub[0:metadata_count], dtopk_weights[metadata_offset : metadata_offset + metadata_count])
                        else:
                            T.copy(dtopk_ub[0:num_topk], dtopk_weights[metadata_offset : metadata_offset + num_topk])

            if with_sf:
                dsf_local_ub[0] = dsf_local
                T.tile.atomic_add(dsf[0], dsf_local_ub)

    return reduce_fused_backward_k2_h_le_3072


@tilelang.jit(pass_configs=h576_fp32_weighted_unscaled_pass_configs)
def get_reduce_fused_backward_h576_fp32_weighted_unscaled_kernel(num_topk: int):
    """Two-token, four-route batched pipeline for weighted FP32 H=576."""
    if num_topk not in (2, 6, 8, 9):
        raise ValueError(f"Unsupported H=576 TopK: {num_topk}")

    hidden = 576
    num_tokens = T.symbolic("num_tokens")
    num_expanded_tokens = T.symbolic("num_expanded_tokens")
    num_cores = 24
    num_owners = num_cores * 2
    rows_per_block = 2
    routes_per_block = rows_per_block * num_topk
    routes_per_batch = 4
    num_route_batches = (routes_per_block + routes_per_batch - 1) // routes_per_batch
    tokens_per_owner = T.ceildiv(num_tokens, num_owners)
    blocks_per_owner = T.ceildiv(tokens_per_owner, rows_per_block)
    metadata_aligned = ((routes_per_block + 7) // 8) * 8

    X_EVENT0 = 0
    X_EVENT1 = 1
    STORE_EVENT0 = 2
    STORE_EVENT1 = 3
    DTOP_EVENT = 4
    META_READY = 5
    DOUT_READY = 6
    DOT_READY = 7

    @T.macro
    def queue_batch(x_gm, mapping_ub, route_base, x_buf, event):
        T.wait_flag("v", "mte2", event)
        for lane in T.unroll(routes_per_batch):
            route = route_base + lane
            if route < routes_per_block:
                pos = mapping_ub[route]
                if pos >= 0:
                    T.copy(x_gm[pos, 0:hidden], x_buf[lane, 0:hidden])
        T.set_flag("mte2", "v", event)

    @T.macro
    def process_batch(mapping_ub, weights_ub, route_base, dout_buf0, dout_buf1, x_buf, dx_buf, x_lane_buf, dx_lane_buf, dot_buf, dot_sum_buf, dtopk_buf, dx_gm, x_event, store_event):
        T.wait_flag("mte2", "v", x_event)
        T.wait_flag("mte3", "v", store_event)

        for lane in T.unroll(routes_per_batch):
            route = route_base + lane
            if route < routes_per_block:
                pos = mapping_ub[route]
                route_weight = weights_ub[route]
                if pos >= 0:
                    # Tile intrinsics require a 1-D Buffer here. A row slice
                    # of a macro argument lowers to BufferLoad in this
                    # TileLang version, so stage it through a stable buffer.
                    T.copy(x_buf[lane, 0:hidden], x_lane_buf)
                    if route < num_topk:
                        T.tile.mul(dx_lane_buf, dout_buf0, route_weight)
                        T.tile.mul(dot_buf, dout_buf0, x_lane_buf)
                    else:
                        T.tile.mul(dx_lane_buf, dout_buf1, route_weight)
                        T.tile.mul(dot_buf, dout_buf1, x_lane_buf)
                    T.reduce_sum(dot_buf, dot_sum_buf, dim=-1)
                    T.copy(dx_lane_buf, dx_buf[lane, 0:hidden])

                    T.set_flag("v", "s", DOT_READY)
                    T.wait_flag("v", "s", DOT_READY)
                    dtopk_buf[route] = dot_sum_buf[0]
                    T.set_flag("s", "v", DOT_READY)
                    T.wait_flag("s", "v", DOT_READY)

        # MTE2 can refill this entire group while MTE3 drains its dx rows.
        T.set_flag("v", "mte2", x_event)
        T.set_flag("v", "mte3", store_event)
        T.wait_flag("v", "mte3", store_event)
        for lane in T.unroll(routes_per_batch):
            route = route_base + lane
            if route < routes_per_block:
                pos = mapping_ub[route]
                if pos >= 0:
                    T.copy(dx_buf[lane, 0:hidden], dx_gm[pos, 0:hidden])
        T.set_flag("mte3", "v", store_event)

    @T.prim_func
    def reduce_fused_backward_h576_fp32_weighted_unscaled(
        x: T.Tensor((num_expanded_tokens, hidden), "float32"),
        topk_weights: T.Tensor((num_tokens * num_topk,), "float32"),
        token_topk_to_pos: T.Tensor((num_tokens * num_topk,), "int32"),
        sf: T.Tensor((1,), "float32"),
        x_sf: T.Tensor((num_expanded_tokens,), "float32"),
        dout: T.Tensor((num_tokens, hidden), "float32"),
        dx: T.Tensor((num_expanded_tokens, hidden), "float32"),
        dtopk_weights: T.Tensor((num_tokens * num_topk,), "float32"),
        dx_sf: T.Tensor((num_expanded_tokens,), "float32"),
        dsf: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid), T.Scope("V"):
            mapping_ub = T.alloc_ub((metadata_aligned,), "int32")
            weights_ub = T.alloc_ub((metadata_aligned,), "float32")
            dtopk_ub = T.alloc_ub((metadata_aligned,), "float32")
            dout_ub0 = T.alloc_ub((hidden,), "float32")
            dout_ub1 = T.alloc_ub((hidden,), "float32")
            x_ub0 = T.alloc_ub((routes_per_batch, hidden), "float32")
            x_ub1 = T.alloc_ub((routes_per_batch, hidden), "float32")
            dx_ub0 = T.alloc_ub((routes_per_batch, hidden), "float32")
            dx_ub1 = T.alloc_ub((routes_per_batch, hidden), "float32")
            x_lane_ub = T.alloc_ub((hidden,), "float32")
            dx_lane_ub = T.alloc_ub((hidden,), "float32")
            dot_ub = T.alloc_ub((hidden,), "float32")
            dot_sum_ub = T.alloc_ub((1,), "float32")

            owner = cid * 2 + vid
            owner_token_base = owner * tokens_per_owner
            owner_token_end = T.min(owner_token_base + tokens_per_owner, num_tokens)

            T.set_flag("v", "mte2", X_EVENT0)
            T.set_flag("v", "mte2", X_EVENT1)
            T.set_flag("mte3", "v", STORE_EVENT0)
            T.set_flag("mte3", "v", STORE_EVENT1)
            T.set_flag("mte3", "v", DTOP_EVENT)

            for block in T.serial(blocks_per_owner):
                token0 = owner_token_base + block * rows_per_block
                if token0 < owner_token_end:
                    token1 = token0 + 1
                    metadata_offset = token0 * num_topk

                    if token1 < owner_token_end:
                        T.copy(token_topk_to_pos[metadata_offset : metadata_offset + routes_per_block], mapping_ub[0:routes_per_block])
                        T.copy(topk_weights[metadata_offset : metadata_offset + routes_per_block], weights_ub[0:routes_per_block])
                    else:
                        # Do not issue a 2*K DMA for the final odd token. With
                        # safe-memory lowering disabled, pad_value does not
                        # protect the valid first row from that OOB transfer.
                        T.copy(token_topk_to_pos[metadata_offset : metadata_offset + num_topk], mapping_ub[0:num_topk])
                        T.copy(topk_weights[metadata_offset : metadata_offset + num_topk], weights_ub[0:num_topk])
                    T.set_flag("mte2", "s", META_READY)

                    T.copy(dout[token0, 0:hidden], dout_ub0)
                    if token1 < owner_token_end:
                        T.copy(dout[token1, 0:hidden], dout_ub1)
                    T.set_flag("mte2", "v", DOUT_READY)

                    T.wait_flag("mte3", "v", DTOP_EVENT)
                    T.tile.fill(dtopk_ub, 0.0)
                    T.wait_flag("mte2", "s", META_READY)
                    if token1 >= owner_token_end:
                        for tail_route in T.unroll(num_topk):
                            mapping_ub[num_topk + tail_route] = -1

                    queue_batch(x, mapping_ub, 0, x_ub0, X_EVENT0)
                    T.wait_flag("mte2", "v", DOUT_READY)

                    for batch in T.unroll(num_route_batches):
                        if batch + 1 < num_route_batches:
                            if batch % 2 == 0:
                                queue_batch(x, mapping_ub, (batch + 1) * routes_per_batch, x_ub1, X_EVENT1)
                            else:
                                queue_batch(x, mapping_ub, (batch + 1) * routes_per_batch, x_ub0, X_EVENT0)

                        if batch % 2 == 0:
                            process_batch(mapping_ub, weights_ub, batch * routes_per_batch, dout_ub0, dout_ub1, x_ub0, dx_ub0, x_lane_ub, dx_lane_ub, dot_ub, dot_sum_ub, dtopk_ub, dx, X_EVENT0, STORE_EVENT0)
                        else:
                            process_batch(mapping_ub, weights_ub, batch * routes_per_batch, dout_ub0, dout_ub1, x_ub1, dx_ub1, x_lane_ub, dx_lane_ub, dot_ub, dot_sum_ub, dtopk_ub, dx, X_EVENT1, STORE_EVENT1)

                    T.set_flag("s", "mte3", DTOP_EVENT)
                    T.wait_flag("s", "mte3", DTOP_EVENT)
                    if token1 < owner_token_end:
                        T.copy(dtopk_ub[0:routes_per_block], dtopk_weights[metadata_offset : metadata_offset + routes_per_block])
                    else:
                        T.copy(dtopk_ub[0:num_topk], dtopk_weights[metadata_offset : metadata_offset + num_topk])
                    T.set_flag("mte3", "v", DTOP_EVENT)

            # Every initialized/released event must be consumed before the
            # AIV exits. In particular K=2 uses only one route batch, leaving
            # both V->MTE2 slot-release flags pending without this epilogue.
            T.wait_flag("v", "mte2", X_EVENT0)
            T.wait_flag("v", "mte2", X_EVENT1)
            T.wait_flag("mte3", "v", STORE_EVENT0)
            T.wait_flag("mte3", "v", STORE_EVENT1)
            T.wait_flag("mte3", "v", DTOP_EVENT)
            T.pipe_barrier("ALL")

    return reduce_fused_backward_h576_fp32_weighted_unscaled


@tilelang.jit(pass_configs=auto_pass_configs)
def get_reduce_fused_backward_k6_k8_k9_weighted_unscaled_kernel(hidden: int, num_topk: int, in_dtype: torch.dtype, out_dtype: torch.dtype):
    """Compact four-route pipeline for standard weighted K=6/8/9."""
    if num_topk not in (6, 8, 9):
        raise ValueError(f"Weighted/unscaled path does not support K={num_topk}")

    num_tokens = T.symbolic("num_tokens")
    num_expanded_tokens = T.symbolic("num_expanded_tokens")
    num_cores = 24
    num_owners = num_cores * 2
    tokens_per_owner = T.ceildiv(num_tokens, num_owners)
    x_stages = 4 if hidden <= 3072 else 2
    metadata_aligned = ((num_topk + 7) // 8) * 8

    dtype_map = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}
    tl_in_dtype = dtype_map.get(in_dtype, "float32")
    tl_out_dtype = dtype_map.get(out_dtype, "float32")
    need_input_cast = in_dtype != torch.float32
    need_dout_cast = out_dtype != torch.float32
    need_dx_cast = in_dtype != torch.float32
    x_cast_extent = hidden if need_input_cast else 1
    dout_cast_extent = hidden if need_dout_cast else 1
    dx_cast_extent = hidden if need_dx_cast else 1
    extra_x_extent = hidden if x_stages == 4 else 1
    extra_dx_cast_extent = dx_cast_extent if x_stages == 4 else 1

    @T.macro
    def process_route(mapping_ub, weights_ub, route, dout_f32_buf, x_input_buf, x_f32_buf, dx_f32_buf, dx_output_buf, dot_buf, dot_sum_buf, dtopk_buf, dx_gm):
        pos = mapping_ub[route]
        if pos >= 0:
            T.tile.mul(dx_f32_buf, dout_f32_buf, weights_ub[route])
            if need_dx_cast:
                T.tile.cast(dx_output_buf, dx_f32_buf, "CAST_RINT", hidden)
                T.copy(dx_output_buf, dx_gm[pos, 0:hidden])
            else:
                T.copy(dx_f32_buf, dx_gm[pos, 0:hidden])

            if need_input_cast:
                T.tile.cast(x_f32_buf, x_input_buf, "CAST_NONE", hidden)
                T.tile.mul(dot_buf, dout_f32_buf, x_f32_buf)
            else:
                T.tile.mul(dot_buf, dout_f32_buf, x_input_buf)
            T.reduce_sum(dot_buf, dot_sum_buf, dim=-1)
            dtopk_buf[route] = dot_sum_buf[0]

    @T.macro
    def run_routes(
        x_gm,
        mapping_ub,
        weights_ub,
        dout_f32_buf,
        x_input_buf0,
        x_input_buf1,
        x_input_buf2,
        x_input_buf3,
        x_f32_buf,
        dx_f32_buf0,
        dx_f32_buf1,
        dx_f32_buf2,
        dx_f32_buf3,
        dx_output_buf0,
        dx_output_buf1,
        dx_output_buf2,
        dx_output_buf3,
        dot_buf,
        dot_sum_buf,
        dtopk_buf,
        dx_gm,
    ):
        pos0 = mapping_ub[0]
        if pos0 >= 0:
            T.copy(x_gm[pos0, 0:hidden], x_input_buf0)
        pos1 = mapping_ub[1]
        if pos1 >= 0:
            T.copy(x_gm[pos1, 0:hidden], x_input_buf1)
        if x_stages == 4:
            pos2 = mapping_ub[2]
            if pos2 >= 0:
                T.copy(x_gm[pos2, 0:hidden], x_input_buf2)
            pos3 = mapping_ub[3]
            if pos3 >= 0:
                T.copy(x_gm[pos3, 0:hidden], x_input_buf3)

        for route in T.unroll(num_topk):
            if route % x_stages == 0:
                process_route(mapping_ub, weights_ub, route, dout_f32_buf, x_input_buf0, x_f32_buf, dx_f32_buf0, dx_output_buf0, dot_buf, dot_sum_buf, dtopk_buf, dx_gm)
                if route + x_stages < num_topk:
                    next_pos = mapping_ub[route + x_stages]
                    if next_pos >= 0:
                        T.copy(x_gm[next_pos, 0:hidden], x_input_buf0)
            if route % x_stages == 1:
                process_route(mapping_ub, weights_ub, route, dout_f32_buf, x_input_buf1, x_f32_buf, dx_f32_buf1, dx_output_buf1, dot_buf, dot_sum_buf, dtopk_buf, dx_gm)
                if route + x_stages < num_topk:
                    next_pos = mapping_ub[route + x_stages]
                    if next_pos >= 0:
                        T.copy(x_gm[next_pos, 0:hidden], x_input_buf1)
            if x_stages == 4:
                if route % x_stages == 2:
                    process_route(mapping_ub, weights_ub, route, dout_f32_buf, x_input_buf2, x_f32_buf, dx_f32_buf2, dx_output_buf2, dot_buf, dot_sum_buf, dtopk_buf, dx_gm)
                    if route + x_stages < num_topk:
                        next_pos = mapping_ub[route + x_stages]
                        if next_pos >= 0:
                            T.copy(x_gm[next_pos, 0:hidden], x_input_buf2)
                if route % x_stages == 3:
                    process_route(mapping_ub, weights_ub, route, dout_f32_buf, x_input_buf3, x_f32_buf, dx_f32_buf3, dx_output_buf3, dot_buf, dot_sum_buf, dtopk_buf, dx_gm)
                    if route + x_stages < num_topk:
                        next_pos = mapping_ub[route + x_stages]
                        if next_pos >= 0:
                            T.copy(x_gm[next_pos, 0:hidden], x_input_buf3)

    @T.prim_func
    def reduce_fused_backward_k6_k8_k9_weighted_unscaled(
        x: T.Tensor((num_expanded_tokens, hidden), tl_in_dtype),
        topk_weights: T.Tensor((num_tokens * num_topk,), "float"),
        token_topk_to_pos: T.Tensor((num_tokens * num_topk,), "int32"),
        sf: T.Tensor((1,), "float"),
        x_sf: T.Tensor((num_expanded_tokens,), "float"),
        dout: T.Tensor((num_tokens, hidden), tl_out_dtype),
        dx: T.Tensor((num_expanded_tokens, hidden), tl_in_dtype),
        dtopk_weights: T.Tensor((num_tokens * num_topk,), "float"),
        dx_sf: T.Tensor((num_expanded_tokens,), "float"),
        dsf: T.Tensor((1,), "float"),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid), T.Scope("V"):
            mapping_ub = T.alloc_ub((metadata_aligned,), "int32")
            weights_ub = T.alloc_ub((metadata_aligned,), "float")
            dtopk_ub = T.alloc_ub((metadata_aligned,), "float")
            dout_input_ub = T.alloc_ub((dout_cast_extent,), tl_out_dtype)
            dout_f32_ub = T.alloc_ub((hidden,), "float")
            x_input_ub0 = T.alloc_ub((hidden,), tl_in_dtype)
            x_input_ub1 = T.alloc_ub((hidden,), tl_in_dtype)
            x_input_ub2 = T.alloc_ub((extra_x_extent,), tl_in_dtype)
            x_input_ub3 = T.alloc_ub((extra_x_extent,), tl_in_dtype)
            x_f32_ub = T.alloc_ub((x_cast_extent,), "float")
            dx_f32_ub0 = T.alloc_ub((hidden,), "float")
            dx_f32_ub1 = T.alloc_ub((hidden,), "float")
            dx_f32_ub2 = T.alloc_ub((extra_x_extent,), "float")
            dx_f32_ub3 = T.alloc_ub((extra_x_extent,), "float")
            dx_output_ub0 = T.alloc_ub((dx_cast_extent,), tl_in_dtype)
            dx_output_ub1 = T.alloc_ub((dx_cast_extent,), tl_in_dtype)
            dx_output_ub2 = T.alloc_ub((extra_dx_cast_extent,), tl_in_dtype)
            dx_output_ub3 = T.alloc_ub((extra_dx_cast_extent,), tl_in_dtype)
            dot_ub = T.alloc_ub((hidden,), "float")
            dot_sum_ub = T.alloc_ub((1,), "float")

            owner = cid * 2 + vid
            owner_token_base = owner * tokens_per_owner
            owner_token_end = T.min(owner_token_base + tokens_per_owner, num_tokens)

            for local_token in T.serial(tokens_per_owner):
                token = owner_token_base + local_token
                if token < owner_token_end:
                    metadata_offset = token * num_topk
                    T.copy(token_topk_to_pos[metadata_offset : metadata_offset + num_topk], mapping_ub[0:num_topk], pad_value=-1)
                    T.copy(topk_weights[metadata_offset : metadata_offset + num_topk], weights_ub[0:num_topk], pad_value=0.0)
                    T.tile.fill(dtopk_ub, 0.0)

                    if need_dout_cast:
                        T.copy(dout[token, 0:hidden], dout_input_ub)
                        T.tile.cast(dout_f32_ub, dout_input_ub, "CAST_NONE", hidden)
                    else:
                        T.copy(dout[token, 0:hidden], dout_f32_ub)

                    run_routes(x, mapping_ub, weights_ub, dout_f32_ub, x_input_ub0, x_input_ub1, x_input_ub2, x_input_ub3, x_f32_ub, dx_f32_ub0, dx_f32_ub1, dx_f32_ub2, dx_f32_ub3, dx_output_ub0, dx_output_ub1, dx_output_ub2, dx_output_ub3, dot_ub, dot_sum_ub, dtopk_ub, dx)
                    T.copy(dtopk_ub[0:num_topk], dtopk_weights[metadata_offset : metadata_offset + num_topk])

    return reduce_fused_backward_k6_k8_k9_weighted_unscaled


@tilelang.jit(pass_configs=auto_pass_configs)
def get_reduce_fused_backward_k2_k6_k8_k9_general_kernel(hidden: int, num_topk: int, in_dtype: torch.dtype, out_dtype: torch.dtype, with_sf: bool, with_weights: bool, with_x_sf: bool):
    if num_topk not in (2, 6, 8, 9):
        raise ValueError(f"Supported-TopK general path does not support K={num_topk}")

    num_tokens = T.symbolic("num_tokens")
    num_expanded_tokens = T.symbolic("num_expanded_tokens")
    num_cores = 24
    num_owners = num_cores * 2
    rows_per_block = 4 if hidden <= 2048 else 2
    x_stages = 2 if num_topk == 2 or hidden > 3072 else 4
    tokens_per_owner = T.ceildiv(num_tokens, num_owners)
    blocks_per_owner = T.ceildiv(tokens_per_owner, rows_per_block)
    metadata_count = rows_per_block * num_topk
    metadata_aligned = ((metadata_count + 7) // 8) * 8

    dtype_map = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}
    tl_in_dtype = dtype_map.get(in_dtype, "float32")
    tl_out_dtype = dtype_map.get(out_dtype, "float32")
    need_input_cast = in_dtype != torch.float32
    need_dout_cast = out_dtype != torch.float32
    need_dx_cast = in_dtype != torch.float32
    x_cast_extent = hidden if need_input_cast else 1
    dout_cast_extent = hidden if need_dout_cast else 1
    dx_cast_extent = hidden if need_dx_cast else 1
    extra_x_extent = hidden if x_stages == 4 else 1
    extra_dout_extent = hidden if rows_per_block == 4 else 1
    extra_dout_cast_extent = dout_cast_extent if rows_per_block == 4 else 1
    extra_dx_cast_extent = dx_cast_extent if x_stages == 4 else 1

    @T.macro
    def load_metadata(mapping_gm, weights_gm, mapping_ub, weights_ub, token_base):
        metadata_offset = token_base * num_topk
        T.copy(mapping_gm[metadata_offset : metadata_offset + metadata_count], mapping_ub[0:metadata_count], pad_value=-1)
        if with_weights:
            T.copy(weights_gm[metadata_offset : metadata_offset + metadata_count], weights_ub[0:metadata_count], pad_value=0.0)

    @T.macro
    def load_dout(dout_gm, token, dout_input_buf, dout_f32_buf):
        if need_dout_cast:
            T.copy(dout_gm[token, 0:hidden], dout_input_buf)
            T.tile.cast(dout_f32_buf, dout_input_buf, "CAST_NONE", hidden)
        else:
            T.copy(dout_gm[token, 0:hidden], dout_f32_buf)

    @T.macro
    def store_contiguous_routes(output_ub, output_gm, token_base, token_limit):
        output_offset = token_base * num_topk
        if token_base + rows_per_block <= token_limit:
            T.copy(output_ub[0:metadata_count], output_gm[output_offset : output_offset + metadata_count])
        else:
            for row in T.serial(rows_per_block):
                token = token_base + row
                if token < token_limit:
                    row_offset = row * num_topk
                    gm_offset = token * num_topk
                    T.copy(output_ub[row_offset : row_offset + num_topk], output_gm[gm_offset : gm_offset + num_topk])

    @T.macro
    def process_token(
        mapping_ub,
        weights_ub,
        metadata_row,
        dout_f32_buf,
        x_input_buf0,
        x_input_buf1,
        x_input_buf2,
        x_input_buf3,
        x_f32_buf,
        dx_f32_buf0,
        dx_f32_buf1,
        dx_f32_buf2,
        dx_f32_buf3,
        dx_output_buf0,
        dx_output_buf1,
        dx_output_buf2,
        dx_output_buf3,
        dot_buf,
        dot_sum_buf,
        dtopk_buf,
        x_gm,
        dx_gm,
        dx_sf_gm,
        sf_gm,
        x_sf_gm,
        dsf_ref,
    ):
        metadata_row_offset = metadata_row * num_topk

        pos0 = mapping_ub[metadata_row_offset]
        if pos0 >= 0:
            T.copy(x_gm[pos0, 0:hidden], x_input_buf0)
        pos1 = mapping_ub[metadata_row_offset + 1]
        if pos1 >= 0:
            T.copy(x_gm[pos1, 0:hidden], x_input_buf1)
        if x_stages == 4:
            if num_topk > 2:
                pos2 = mapping_ub[metadata_row_offset + 2]
                if pos2 >= 0:
                    T.copy(x_gm[pos2, 0:hidden], x_input_buf2)
            if num_topk > 3:
                pos3 = mapping_ub[metadata_row_offset + 3]
                if pos3 >= 0:
                    T.copy(x_gm[pos3, 0:hidden], x_input_buf3)

        for route in T.unroll(num_topk):
            if route % x_stages == 0:
                process_scaled_route(mapping_ub, weights_ub, metadata_row_offset, route, dout_f32_buf, x_input_buf0, x_f32_buf, dx_f32_buf0, dx_output_buf0, dot_buf, dot_sum_buf, dtopk_buf, dx_gm, dx_sf_gm, sf_gm, x_sf_gm, dsf_ref, hidden, need_input_cast, need_dx_cast, with_weights, with_sf, with_x_sf)
                if route + x_stages < num_topk:
                    next_pos = mapping_ub[metadata_row_offset + route + x_stages]
                    if next_pos >= 0:
                        T.copy(x_gm[next_pos, 0:hidden], x_input_buf0)
            if route % x_stages == 1:
                process_scaled_route(mapping_ub, weights_ub, metadata_row_offset, route, dout_f32_buf, x_input_buf1, x_f32_buf, dx_f32_buf1, dx_output_buf1, dot_buf, dot_sum_buf, dtopk_buf, dx_gm, dx_sf_gm, sf_gm, x_sf_gm, dsf_ref, hidden, need_input_cast, need_dx_cast, with_weights, with_sf, with_x_sf)
                if route + x_stages < num_topk:
                    next_pos = mapping_ub[metadata_row_offset + route + x_stages]
                    if next_pos >= 0:
                        T.copy(x_gm[next_pos, 0:hidden], x_input_buf1)
            if x_stages == 4:
                if route % x_stages == 2:
                    process_scaled_route(mapping_ub, weights_ub, metadata_row_offset, route, dout_f32_buf, x_input_buf2, x_f32_buf, dx_f32_buf2, dx_output_buf2, dot_buf, dot_sum_buf, dtopk_buf, dx_gm, dx_sf_gm, sf_gm, x_sf_gm, dsf_ref, hidden, need_input_cast, need_dx_cast, with_weights, with_sf, with_x_sf)
                    if route + x_stages < num_topk:
                        next_pos = mapping_ub[metadata_row_offset + route + x_stages]
                        if next_pos >= 0:
                            T.copy(x_gm[next_pos, 0:hidden], x_input_buf2)
                if route % x_stages == 3:
                    process_scaled_route(mapping_ub, weights_ub, metadata_row_offset, route, dout_f32_buf, x_input_buf3, x_f32_buf, dx_f32_buf3, dx_output_buf3, dot_buf, dot_sum_buf, dtopk_buf, dx_gm, dx_sf_gm, sf_gm, x_sf_gm, dsf_ref, hidden, need_input_cast, need_dx_cast, with_weights, with_sf, with_x_sf)
                    if route + x_stages < num_topk:
                        next_pos = mapping_ub[metadata_row_offset + route + x_stages]
                        if next_pos >= 0:
                            T.copy(x_gm[next_pos, 0:hidden], x_input_buf3)

    @T.prim_func
    def reduce_fused_backward_k2_k6_k8_k9_general(
        x: T.Tensor((num_expanded_tokens, hidden), tl_in_dtype),
        topk_weights: T.Tensor((num_tokens * num_topk,), "float"),
        token_topk_to_pos: T.Tensor((num_tokens * num_topk,), "int32"),
        sf: T.Tensor((1,), "float"),
        x_sf: T.Tensor((num_expanded_tokens,), "float"),
        dout: T.Tensor((num_tokens, hidden), tl_out_dtype),
        dx: T.Tensor((num_expanded_tokens, hidden), tl_in_dtype),
        dtopk_weights: T.Tensor((num_tokens * num_topk,), "float"),
        dx_sf: T.Tensor((num_expanded_tokens,), "float"),
        dsf: T.Tensor((1,), "float"),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid), T.Scope("V"):
            dout_input_ub0 = T.alloc_ub((dout_cast_extent,), tl_out_dtype)
            dout_input_ub1 = T.alloc_ub((dout_cast_extent,), tl_out_dtype)
            dout_input_ub2 = T.alloc_ub((extra_dout_cast_extent,), tl_out_dtype)
            dout_input_ub3 = T.alloc_ub((extra_dout_cast_extent,), tl_out_dtype)
            dout_f32_ub0 = T.alloc_ub((hidden,), "float")
            dout_f32_ub1 = T.alloc_ub((hidden,), "float")
            dout_f32_ub2 = T.alloc_ub((extra_dout_extent,), "float")
            dout_f32_ub3 = T.alloc_ub((extra_dout_extent,), "float")
            x_input_ub0 = T.alloc_ub((hidden,), tl_in_dtype)
            x_input_ub1 = T.alloc_ub((hidden,), tl_in_dtype)
            x_input_ub2 = T.alloc_ub((extra_x_extent,), tl_in_dtype)
            x_input_ub3 = T.alloc_ub((extra_x_extent,), tl_in_dtype)
            x_f32_ub = T.alloc_ub((x_cast_extent,), "float")
            dx_f32_ub0 = T.alloc_ub((hidden,), "float")
            dx_f32_ub1 = T.alloc_ub((hidden,), "float")
            dx_f32_ub2 = T.alloc_ub((extra_x_extent,), "float")
            dx_f32_ub3 = T.alloc_ub((extra_x_extent,), "float")
            dx_output_ub0 = T.alloc_ub((dx_cast_extent,), tl_in_dtype)
            dx_output_ub1 = T.alloc_ub((dx_cast_extent,), tl_in_dtype)
            dx_output_ub2 = T.alloc_ub((extra_dx_cast_extent,), tl_in_dtype)
            dx_output_ub3 = T.alloc_ub((extra_dx_cast_extent,), tl_in_dtype)
            dot_ub = T.alloc_ub((hidden,), "float")
            dot_sum_ub = T.alloc_ub((1,), "float")
            topk_weights_ub0 = T.alloc_ub((metadata_aligned,), "float")
            topk_weights_ub1 = T.alloc_ub((metadata_aligned,), "float")
            topk_to_pos_ub0 = T.alloc_ub((metadata_aligned,), "int32")
            topk_to_pos_ub1 = T.alloc_ub((metadata_aligned,), "int32")
            dtopk_weights_ub0 = T.alloc_ub((metadata_aligned,), "float")
            dtopk_weights_ub1 = T.alloc_ub((metadata_aligned,), "float")
            dsf_local_ub = T.alloc_ub((1,), "float")

            dsf_local = T.alloc_var("float32", init=0.0)
            pending_output = T.alloc_var("int32", init=0)
            pending_stage = T.alloc_var("int32", init=0)
            pending_token_base = T.alloc_var("int32", init=0)

            owner = cid * 2 + vid
            owner_token_base = owner * tokens_per_owner
            owner_token_end = T.min(owner_token_base + tokens_per_owner, num_tokens)
            dsf_local = 0.0
            pending_output = 0

            if owner_token_base < owner_token_end:
                load_metadata(token_topk_to_pos, topk_weights, topk_to_pos_ub0, topk_weights_ub0, owner_token_base)

            for block in T.serial(blocks_per_owner):
                token_base = owner_token_base + block * rows_per_block
                if token_base < owner_token_end:
                    if with_weights:
                        if pending_output != 0:
                            if pending_stage == 0:
                                store_contiguous_routes(dtopk_weights_ub0, dtopk_weights, pending_token_base, owner_token_end)
                            else:
                                store_contiguous_routes(dtopk_weights_ub1, dtopk_weights, pending_token_base, owner_token_end)
                            pending_output = 0

                    if block + 1 < blocks_per_owner:
                        next_token_base = token_base + rows_per_block
                        if next_token_base < owner_token_end:
                            if block % 2 == 0:
                                load_metadata(token_topk_to_pos, topk_weights, topk_to_pos_ub1, topk_weights_ub1, next_token_base)
                            else:
                                load_metadata(token_topk_to_pos, topk_weights, topk_to_pos_ub0, topk_weights_ub0, next_token_base)

                    if with_weights:
                        if block % 2 == 0:
                            T.tile.fill(dtopk_weights_ub0, 0.0)
                        else:
                            T.tile.fill(dtopk_weights_ub1, 0.0)

                    load_dout(dout, token_base, dout_input_ub0, dout_f32_ub0)
                    for row in T.serial(rows_per_block):
                        token = token_base + row
                        if token < owner_token_end:
                            if row == 0:
                                if token + 1 < owner_token_end:
                                    load_dout(dout, token + 1, dout_input_ub1, dout_f32_ub1)
                            if rows_per_block == 4:
                                if row == 1:
                                    if token + 1 < owner_token_end:
                                        load_dout(dout, token + 1, dout_input_ub2, dout_f32_ub2)
                                if row == 2:
                                    if token + 1 < owner_token_end:
                                        load_dout(dout, token + 1, dout_input_ub3, dout_f32_ub3)

                            if row == 0:
                                if block % 2 == 0:
                                    process_token(topk_to_pos_ub0, topk_weights_ub0, row, dout_f32_ub0, x_input_ub0, x_input_ub1, x_input_ub2, x_input_ub3, x_f32_ub, dx_f32_ub0, dx_f32_ub1, dx_f32_ub2, dx_f32_ub3, dx_output_ub0, dx_output_ub1, dx_output_ub2, dx_output_ub3, dot_ub, dot_sum_ub, dtopk_weights_ub0, x, dx, dx_sf, sf, x_sf, dsf_local)
                                else:
                                    process_token(topk_to_pos_ub1, topk_weights_ub1, row, dout_f32_ub0, x_input_ub0, x_input_ub1, x_input_ub2, x_input_ub3, x_f32_ub, dx_f32_ub0, dx_f32_ub1, dx_f32_ub2, dx_f32_ub3, dx_output_ub0, dx_output_ub1, dx_output_ub2, dx_output_ub3, dot_ub, dot_sum_ub, dtopk_weights_ub1, x, dx, dx_sf, sf, x_sf, dsf_local)
                            if row == 1:
                                if block % 2 == 0:
                                    process_token(topk_to_pos_ub0, topk_weights_ub0, row, dout_f32_ub1, x_input_ub0, x_input_ub1, x_input_ub2, x_input_ub3, x_f32_ub, dx_f32_ub0, dx_f32_ub1, dx_f32_ub2, dx_f32_ub3, dx_output_ub0, dx_output_ub1, dx_output_ub2, dx_output_ub3, dot_ub, dot_sum_ub, dtopk_weights_ub0, x, dx, dx_sf, sf, x_sf, dsf_local)
                                else:
                                    process_token(topk_to_pos_ub1, topk_weights_ub1, row, dout_f32_ub1, x_input_ub0, x_input_ub1, x_input_ub2, x_input_ub3, x_f32_ub, dx_f32_ub0, dx_f32_ub1, dx_f32_ub2, dx_f32_ub3, dx_output_ub0, dx_output_ub1, dx_output_ub2, dx_output_ub3, dot_ub, dot_sum_ub, dtopk_weights_ub1, x, dx, dx_sf, sf, x_sf, dsf_local)
                            if rows_per_block == 4:
                                if row == 2:
                                    if block % 2 == 0:
                                        process_token(topk_to_pos_ub0, topk_weights_ub0, row, dout_f32_ub2, x_input_ub0, x_input_ub1, x_input_ub2, x_input_ub3, x_f32_ub, dx_f32_ub0, dx_f32_ub1, dx_f32_ub2, dx_f32_ub3, dx_output_ub0, dx_output_ub1, dx_output_ub2, dx_output_ub3, dot_ub, dot_sum_ub, dtopk_weights_ub0, x, dx, dx_sf, sf, x_sf, dsf_local)
                                    else:
                                        process_token(topk_to_pos_ub1, topk_weights_ub1, row, dout_f32_ub2, x_input_ub0, x_input_ub1, x_input_ub2, x_input_ub3, x_f32_ub, dx_f32_ub0, dx_f32_ub1, dx_f32_ub2, dx_f32_ub3, dx_output_ub0, dx_output_ub1, dx_output_ub2, dx_output_ub3, dot_ub, dot_sum_ub, dtopk_weights_ub1, x, dx, dx_sf, sf, x_sf, dsf_local)
                                if row == 3:
                                    if block % 2 == 0:
                                        process_token(topk_to_pos_ub0, topk_weights_ub0, row, dout_f32_ub3, x_input_ub0, x_input_ub1, x_input_ub2, x_input_ub3, x_f32_ub, dx_f32_ub0, dx_f32_ub1, dx_f32_ub2, dx_f32_ub3, dx_output_ub0, dx_output_ub1, dx_output_ub2, dx_output_ub3, dot_ub, dot_sum_ub, dtopk_weights_ub0, x, dx, dx_sf, sf, x_sf, dsf_local)
                                    else:
                                        process_token(topk_to_pos_ub1, topk_weights_ub1, row, dout_f32_ub3, x_input_ub0, x_input_ub1, x_input_ub2, x_input_ub3, x_f32_ub, dx_f32_ub0, dx_f32_ub1, dx_f32_ub2, dx_f32_ub3, dx_output_ub0, dx_output_ub1, dx_output_ub2, dx_output_ub3, dot_ub, dot_sum_ub, dtopk_weights_ub1, x, dx, dx_sf, sf, x_sf, dsf_local)

                    if with_weights:
                        pending_output = 1
                        pending_stage = block % 2
                        pending_token_base = token_base

            if with_weights:
                if pending_output != 0:
                    if pending_stage == 0:
                        store_contiguous_routes(dtopk_weights_ub0, dtopk_weights, pending_token_base, owner_token_end)
                    else:
                        store_contiguous_routes(dtopk_weights_ub1, dtopk_weights, pending_token_base, owner_token_end)

            if with_sf:
                dsf_local_ub[0] = dsf_local
                T.tile.atomic_add(dsf[0], dsf_local_ub)

    return reduce_fused_backward_k2_k6_k8_k9_general


def get_reduce_fused_backward_kernel(hidden: int, num_topk: int, in_dtype: torch.dtype, out_dtype: torch.dtype, with_sf: bool, with_weights: bool, with_x_sf: bool):
    """Select one execution model; only the selected JIT factory is compiled."""
    supported_topk = num_topk in (2, 6, 8, 9)
    unweighted_unscaled = supported_topk and in_dtype == out_dtype and not with_weights and not with_sf and not with_x_sf
    k6_k8_k9_weighted_unscaled = num_topk in (6, 8, 9) and with_weights and not with_sf and not with_x_sf
    h576_fp32_weighted_unscaled = hidden == 576 and in_dtype == torch.float32 and out_dtype == torch.float32 and supported_topk and with_weights and not with_sf and not with_x_sf
    if h576_fp32_weighted_unscaled:
        return get_reduce_fused_backward_h576_fp32_weighted_unscaled_kernel(num_topk)

    args = (hidden, num_topk, in_dtype, out_dtype, with_sf, with_weights, with_x_sf)
    if unweighted_unscaled:
        return get_reduce_fused_backward_k2_k6_k8_k9_unweighted_unscaled_kernel(hidden, num_topk, in_dtype, out_dtype)
    if num_topk == 2 and hidden <= 3072:
        return get_reduce_fused_backward_k2_h_le_3072_kernel(hidden, in_dtype, out_dtype, with_sf, with_weights, with_x_sf)
    if k6_k8_k9_weighted_unscaled:
        return get_reduce_fused_backward_k6_k8_k9_weighted_unscaled_kernel(hidden, num_topk, in_dtype, out_dtype)
    if supported_topk:
        return get_reduce_fused_backward_k2_k6_k8_k9_general_kernel(*args)
    return get_reduce_fused_backward_other_k_fallback_kernel(*args)


def ascend_reduce_fused_backward(
    x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], topk_weights: Optional[torch.Tensor], token_topk_to_pos: torch.Tensor, dout: torch.Tensor, fp8_format: str = "", sf: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if fp8_format != "":
        raise AssertionError("Ascend example reduce_fused_backward supports only non-FP8 output.")
    if isinstance(x, tuple):
        x, x_sf = x
    else:
        x_sf = None

    assert x.dim() == 2 and x.is_contiguous()
    assert token_topk_to_pos.dim() == 2 and token_topk_to_pos.dtype == torch.int32
    num_expanded_tokens, hidden = x.shape
    num_tokens, num_topk = token_topk_to_pos.shape
    assert dout.shape == (num_tokens, hidden)
    device = x.device
    uses_flat_route_layout = num_topk in (2, 6, 8, 9)

    if topk_weights is None:
        topk_weights_arg = torch.empty((num_tokens, num_topk), dtype=torch.float32, device=device)
    else:
        assert topk_weights.shape == (num_tokens, num_topk)
        assert topk_weights.dtype == torch.float32
        topk_weights_arg = topk_weights.contiguous()

    if sf is None:
        sf_arg = torch.empty((1,), dtype=torch.float32, device=device)
    else:
        assert sf.shape == (1,) and sf.dtype == torch.float32
        sf_arg = sf.contiguous()

    if x_sf is None:
        x_sf_arg = torch.empty((num_expanded_tokens,), dtype=torch.float32, device=device)
    else:
        assert x_sf.shape == (num_expanded_tokens,)
        assert x_sf.dtype == torch.float32
        x_sf_arg = x_sf.contiguous()

    if num_tokens == 0:
        dx = torch.empty_like(x)
        dx.zero_()
        dtopk_weights = torch.zeros((num_tokens, num_topk), dtype=torch.float32, device=device)
        dx_sf = torch.zeros((num_expanded_tokens,), dtype=torch.float32, device=device)
        dsf = torch.zeros((1,), dtype=torch.float32, device=device)
    else:
        dx = torch.zeros_like(x)
        dtopk_weights = torch.empty((num_tokens, num_topk), dtype=torch.float32, device=device)
        dx_sf = torch.zeros((num_expanded_tokens,), dtype=torch.float32, device=device) if x_sf is not None else torch.empty((num_expanded_tokens,), dtype=torch.float32, device=device)
        dx_sf_route = torch.empty((num_tokens, num_topk), dtype=torch.float32, device=device) if not uses_flat_route_layout else torch.empty((1,), dtype=torch.float32, device=device)
        dsf = torch.zeros((1,), dtype=torch.float32, device=device) if sf is not None else torch.empty((1,), dtype=torch.float32, device=device)

        kernel = get_reduce_fused_backward_kernel(hidden=hidden, num_topk=num_topk, in_dtype=x.dtype, out_dtype=dout.dtype, with_sf=sf is not None, with_weights=topk_weights is not None, with_x_sf=x_sf is not None)
        if int(os.getenv("TK_PRINT_KERNEL_SOURCE", "0")):
            print(kernel.get_kernel_source())

        topk_weights_kernel_arg = topk_weights_arg.view(-1) if uses_flat_route_layout else topk_weights_arg
        token_topk_to_pos_kernel_arg = token_topk_to_pos.contiguous().view(-1) if uses_flat_route_layout else token_topk_to_pos.contiguous()
        dtopk_weights_kernel_arg = dtopk_weights.view(-1) if uses_flat_route_layout else dtopk_weights
        dx_sf_kernel_arg = dx_sf if uses_flat_route_layout else dx_sf_route

        kernel(x.contiguous(), topk_weights_kernel_arg, token_topk_to_pos_kernel_arg, sf_arg, x_sf_arg, dout.contiguous(), dx, dtopk_weights_kernel_arg, dx_sf_kernel_arg, dsf)
        if x_sf is not None and not uses_flat_route_layout:
            flat_pos = token_topk_to_pos.reshape(-1)
            valid_pos = flat_pos >= 0
            dx_sf.index_copy_(0, flat_pos[valid_pos].to(torch.int64), dx_sf_route.reshape(-1)[valid_pos])

    return (dx, dtopk_weights if topk_weights is not None else None, dx_sf if x_sf is not None else None, dsf if sf is not None else None)


def build_reduce_fused_backward_func(hidden: int, num_topk: int, dtype: torch.dtype = torch.float32, with_weights: bool = True) -> Callable:
    """Compile once and return the standard reduce_fused backward callable."""
    kernel = get_reduce_fused_backward_kernel(hidden=hidden, num_topk=num_topk, in_dtype=dtype, out_dtype=dtype, with_sf=False, with_weights=with_weights, with_x_sf=False)

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", "0")):
        print(kernel.get_kernel_source())

    def run(x: torch.Tensor, topk_weights: Optional[torch.Tensor], token_topk_to_pos: torch.Tensor, dout: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = token_topk_to_pos.shape[0]
        num_expanded_tokens = x.shape[0]

        if with_weights:
            topk_weights_arg = topk_weights.contiguous().view(-1)
        else:
            topk_weights_arg = torch.empty(num_tokens * num_topk, dtype=torch.float32, device=x.device)

        mapping_arg = token_topk_to_pos.contiguous().view(-1)
        sf_arg = torch.empty(1, dtype=torch.float32, device=x.device)
        x_sf_arg = torch.empty(num_expanded_tokens, dtype=torch.float32, device=x.device)
        dx = torch.empty(num_expanded_tokens, hidden, dtype=dtype, device=x.device)
        dtopk_weights = torch.empty(num_tokens * num_topk, dtype=torch.float32, device=x.device)
        dx_sf_arg = torch.empty(num_expanded_tokens, dtype=torch.float32, device=x.device)
        dsf_arg = torch.empty(1, dtype=torch.float32, device=x.device)

        kernel(x.contiguous(), topk_weights_arg, mapping_arg, sf_arg, x_sf_arg, dout.contiguous(), dx, dtopk_weights, dx_sf_arg, dsf_arg)
        return (dx, dtopk_weights.view(num_tokens, num_topk) if with_weights else None)

    run.kernel = kernel
    return run


def get_ascendc_reduce_fused_grad() -> Callable:
    if torch_npu is not None:
        op = getattr(torch_npu, "npu_moe_token_unpermute_grad", None)
        if op is not None:
            return op
    try:
        return torch.ops.npu.npu_moe_token_unpermute_grad
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError("torch_npu/CANN does not expose npu_moe_token_unpermute_grad") from error


def make_dense_mapping(num_tokens: int, num_topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    mapping_flat = torch.randperm(num_tokens * num_topk, dtype=torch.int32, device="cpu")
    return (mapping_flat.view(num_tokens, num_topk).contiguous(), mapping_flat.contiguous())


def verify_result(output: torch.Tensor, golden: torch.Tensor, rtol: float = 1.0e-4, atol: float = 1.0e-4, error_tol: float = 1.0e-4) -> bool:
    output = output.reshape(-1).to(torch.float64)
    golden = golden.reshape(-1).to(torch.float64)
    close = torch.isclose(output, golden, rtol=rtol, atol=atol, equal_nan=True)
    bad = torch.where(~close)[0]
    error_ratio = float(bad.numel()) / golden.numel()
    return error_ratio <= error_tol


def main():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("An available Ascend NPU is required")

    torch.manual_seed(0)
    profile_only = False
    cann_grad_op = None if profile_only else get_ascendc_reduce_fused_grad()
    test_configs = [
        (4001, 576, 2, torch.float32),
        (4001, 576, 6, torch.float32),
        (4001, 576, 8, torch.float32),
        (4001, 576, 9, torch.float32),
        (4001, 2048, 2, torch.float32),
        (4001, 2048, 6, torch.float32),
        (4001, 2048, 8, torch.float32),
        (4001, 2048, 9, torch.float32),
        (4001, 2560, 2, torch.float32),
        (4001, 2560, 6, torch.float32),
        (4001, 2560, 8, torch.float32),
        (4001, 2560, 9, torch.float32),
        (4001, 3072, 2, torch.float32),
        (4001, 3072, 6, torch.float32),
        (4001, 3072, 8, torch.float32),
        (4001, 3072, 9, torch.float32),
    ]
    failures = []

    for num_tokens, hidden, num_topk, dtype in test_configs:
        for with_weights in (False, True):
            case = f"reduce_fused_backward T={num_tokens} H={hidden} K={num_topk} dtype={dtype} with_weights={with_weights}"
            mapping_cpu, sorted_indices_cpu = make_dense_mapping(num_tokens, num_topk)
            num_expanded_tokens = num_tokens * num_topk
            x = torch.randn(num_expanded_tokens, hidden, dtype=dtype, device="cpu").npu()
            dout = torch.randn(num_tokens, hidden, dtype=dtype, device="cpu").npu()
            mapping = mapping_cpu.npu()
            sorted_indices = None if profile_only else sorted_indices_cpu.npu()

            if with_weights:
                weights_cpu = torch.rand(num_tokens, num_topk, dtype=torch.float32, device="cpu")
                weights_cpu = (weights_cpu / weights_cpu.sum(dim=1, keepdim=True)).contiguous()
            else:
                weights_cpu = torch.ones(num_tokens, num_topk, dtype=torch.float32, device="cpu")

            weights_npu = weights_cpu.npu() if with_weights or not profile_only else None
            tile_weights = weights_npu if with_weights else None
            func = build_reduce_fused_backward_func(hidden=hidden, num_topk=num_topk, dtype=dtype, with_weights=with_weights)
            tile_dx, tile_dweights = func(x, tile_weights, mapping, dout)
            torch.npu.synchronize()

            if profile_only:
                print(f"pass {case}")
                del tile_dx, tile_dweights, x, dout, mapping, sorted_indices, weights_npu, tile_weights
                torch.npu.synchronize()
                continue

            tile_dx_cpu = tile_dx.cpu()
            tile_dweights_cpu = tile_dweights.cpu() if tile_dweights is not None else None
            del tile_dx, tile_dweights

            cann_dx, cann_dweights_all = cann_grad_op(x, dout, sorted_indices, weights_npu)
            torch.npu.synchronize()
            cann_dweights = cann_dweights_all if with_weights else None
            cann_dx_cpu = cann_dx.cpu()
            cann_dweights_cpu = cann_dweights.cpu() if cann_dweights is not None else None

            dx_ok = verify_result(tile_dx_cpu, cann_dx_cpu, rtol=1.0e-4, atol=1.0e-4)
            dweights_ok = True
            if with_weights:
                dweights_ok = verify_result(tile_dweights_cpu, cann_dweights_cpu, rtol=1.0e-4, atol=1.0e-3)

            if dx_ok and dweights_ok:
                print(f"pass {case}")
            else:
                failures.append(case)

            del (cann_dx, cann_dweights_all, cann_dweights, tile_dx_cpu, cann_dx_cpu, tile_dweights_cpu, cann_dweights_cpu, x, dout, mapping, sorted_indices, weights_npu, tile_weights)
            torch.npu.synchronize()

    if failures:
        raise AssertionError("reduce_fused_backward validation failed: " + "; ".join(failures))

    print("TEST PASSED!")


if __name__ == "__main__":
    main()
