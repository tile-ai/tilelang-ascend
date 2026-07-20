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
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True
    }

manual_db_pass_configs = {tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True}


@tilelang.jit(pass_configs=auto_pass_configs)
def get_reduce_fused_generic_kernel(hidden: int, num_topk: int, in_dtype: torch.dtype, out_dtype: torch.dtype, with_sf: bool, with_weights: bool, with_x_sf: bool):
    """Build the stable serial fallback with all UB buffers allocated unconditionally."""
    num_tokens = T.symbolic("num_tokens")
    num_expanded_tokens = T.symbolic("num_expanded_tokens")

    num_cores = 24
    rows_per_vec = 4 if hidden <= 2048 else 2 if hidden <= 4096 else 1
    tokens_per_block = rows_per_vec * 2
    num_token_blocks = T.ceildiv(num_tokens, tokens_per_block)
    num_iters = T.ceildiv(num_token_blocks, num_cores)

    dtype_map = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}
    if in_dtype not in dtype_map:
        raise ValueError(f"Unsupported input dtype: {in_dtype}")
    if out_dtype not in dtype_map:
        raise ValueError(f"Unsupported output dtype: {out_dtype}")

    tl_in_dtype = dtype_map[in_dtype]
    tl_out_dtype = dtype_map[out_dtype]
    need_input_cast = in_dtype != torch.float32
    need_output_cast = out_dtype != torch.float32
    metadata_count = rows_per_vec * num_topk
    metadata_aligned_count = ((metadata_count + 7) // 8) * 8
    # Generic fallback also uses a scalar-produced metadata GM address.
    META_ADDR_READY = 4

    @T.prim_func
    def reduce_fused_generic_kernel(
        x: T.Tensor((num_expanded_tokens, hidden), tl_in_dtype),
        topk_weights_flat: T.Tensor((num_tokens * num_topk,), "float32"),
        token_topk_to_pos_flat: T.Tensor((num_tokens * num_topk,), "int32"),
        sf: T.Tensor((1,), "float32"),
        x_sf: T.Tensor((num_expanded_tokens,), "float32"),
        out: T.Tensor((num_tokens, hidden), tl_out_dtype),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid), T.Scope("V"):
            # Unconditional allocations: never define a UB only inside a flag.
            acc_ub = T.alloc_ub((hidden,), "float32")
            x_compute_f32_ub = T.alloc_ub((hidden,), "float32")
            x_input_ub = T.alloc_ub((hidden,), tl_in_dtype)
            scaled_output_ub = T.alloc_ub((hidden,), "float32")
            out_output_ub = T.alloc_ub((hidden,), tl_out_dtype)

            topk_to_pos_rows_ub = T.alloc_ub((metadata_aligned_count,), "int32")
            topk_weights_rows_ub = T.alloc_ub((metadata_aligned_count,), "float32")

            block_id = T.alloc_var("int32", init=0)
            token_base = T.alloc_var("int32", init=0)
            token_id = T.alloc_var("int32", init=0)
            metadata_offset = T.alloc_var("int32", init=0)
            pos = T.alloc_var("int32", init=-1)
            has_acc = T.alloc_var("int32", init=0)
            scale = T.alloc_var("float32", init=1.0)
            output_scale = T.alloc_var("float32", init=1.0)

            for iter_idx in T.serial(num_iters):
                block_id = cid + iter_idx * num_cores
                if block_id < num_token_blocks:
                    token_base = block_id * tokens_per_block + vid * rows_per_vec

                    if token_base < num_tokens:
                        metadata_offset = token_base * num_topk

                        # metadata_offset is produced by Scalar and consumed
                        # by the following dynamic-address MTE2 copy.
                        T.set_flag("s", "mte2", META_ADDR_READY)
                        T.wait_flag("s", "mte2", META_ADDR_READY)

                        T.copy(token_topk_to_pos_flat[metadata_offset : metadata_offset + metadata_count], topk_to_pos_rows_ub[0:metadata_count], pad_value=-1)
                        if with_weights:
                            T.copy(topk_weights_flat[metadata_offset : metadata_offset + metadata_count], topk_weights_rows_ub[0:metadata_count], pad_value=0.0)

                        for row in T.serial(rows_per_vec):
                            token_id = token_base + row
                            if token_id < num_tokens:
                                has_acc = 0

                                for k in T.serial(num_topk):
                                    pos = topk_to_pos_rows_ub[row * num_topk + k]
                                    if pos >= 0:
                                        scale = 1.0
                                        if with_weights:
                                            scale = topk_weights_rows_ub[row * num_topk + k]
                                        if with_x_sf:
                                            scale = scale * x_sf[pos]

                                        if need_input_cast:
                                            T.copy(x[pos, 0:hidden], x_input_ub)
                                            T.tile.cast(x_compute_f32_ub, x_input_ub, "CAST_NONE", hidden)
                                        else:
                                            T.copy(x[pos, 0:hidden], x_compute_f32_ub)

                                        if has_acc == 0:
                                            if with_weights or with_x_sf:
                                                # Avoid T.tile.mul with a
                                                # T.alloc_var scalar: current
                                                # lowering emits invalid
                                                # scalar.GetValue(0).
                                                T.tile.fill(acc_ub, 0.0)
                                                T.tile.axpy(acc_ub, x_compute_f32_ub, scale)
                                            else:
                                                T.tile.mul(acc_ub, x_compute_f32_ub, 1.0)
                                            has_acc = 1
                                        else:
                                            if with_weights or with_x_sf:
                                                T.tile.axpy(acc_ub, x_compute_f32_ub, scale)
                                            else:
                                                T.tile.add(acc_ub, acc_ub, x_compute_f32_ub)

                                if has_acc == 0:
                                    T.tile.fill(acc_ub, 0.0)

                                if with_sf:
                                    output_scale = sf[0]
                                    T.tile.fill(scaled_output_ub, 0.0)
                                    T.tile.axpy(scaled_output_ub, acc_ub, output_scale)
                                    if need_output_cast:
                                        T.tile.cast(out_output_ub, scaled_output_ub, "CAST_RINT", hidden)
                                        T.copy(out_output_ub, out[token_id, 0:hidden])
                                    else:
                                        T.copy(scaled_output_ub, out[token_id, 0:hidden])
                                else:
                                    if need_output_cast:
                                        T.tile.cast(out_output_ub, acc_ub, "CAST_RINT", hidden)
                                        T.copy(out_output_ub, out[token_id, 0:hidden])
                                    else:
                                        T.copy(acc_ub, out[token_id, 0:hidden])

    return reduce_fused_generic_kernel


@tilelang.jit(pass_configs=manual_db_pass_configs)
def get_reduce_fused_specialized_kernel(hidden: int, num_topk: int, in_dtype: torch.dtype, out_dtype: torch.dtype, with_sf: bool, with_weights: bool, with_x_sf: bool):
    """Build the full-H pipelined kernel specialized for K=2/6/8/9."""
    if num_topk not in (2, 6, 8, 9):
        raise ValueError(f"Specialized kernel does not support topk={num_topk}")

    num_tokens = T.symbolic("num_tokens")
    num_expanded_tokens = T.symbolic("num_expanded_tokens")

    num_cores = 24

    # Cross-row prefetch is part of the default specialized schedule.
    enable_cross_token_prefetch = num_topk in (2, 6, 8, 9)
    use_four_input_schedule = num_topk in (8, 9) and (hidden == 576 or (not with_weights and 2048 <= hidden <= 4096))
    # K=6 at H=2048 performs better with two rows despite the extra metadata
    # transactions: four rows extend output-slot lifetimes and increase UB
    # pressure enough to outweigh the coarser MTE3 batch.
    if num_topk == 6 and hidden == 2048:
        rows_per_vec = 2
    else:
        rows_per_vec = 4 if hidden <= 2048 else 2 if hidden <= 4096 else 1
    # Large-H K=2 is dominated by two wide random gathers. Keeping the
    # original interleaved core ownership distributes simultaneous GM reads
    # better on A3; contiguous AIV ownership is retained for the other paths.
    use_interleaved_blocks = num_topk == 2 and hidden >= 3072
    num_owners = num_cores if use_interleaved_blocks else num_cores * 2
    tokens_per_block = rows_per_vec * 2 if use_interleaved_blocks else rows_per_vec
    num_token_blocks = T.ceildiv(num_tokens, tokens_per_block)
    blocks_per_owner = T.ceildiv(num_token_blocks, num_owners)
    num_iters = blocks_per_owner

    dtype_map = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}
    if in_dtype not in dtype_map:
        raise ValueError(f"Unsupported input dtype: {in_dtype}")
    if out_dtype not in dtype_map:
        raise ValueError(f"Unsupported output dtype: {out_dtype}")

    tl_in_dtype = dtype_map[in_dtype]
    tl_out_dtype = dtype_map[out_dtype]
    need_input_cast = in_dtype != torch.float32
    need_output_cast = out_dtype != torch.float32
    metadata_count = rows_per_vec * num_topk
    metadata_aligned_count = ((metadata_count + 7) // 8) * 8
    extra_x_buffer_len = hidden if num_topk == 6 or use_four_input_schedule else 8

    X_BUF0 = 0
    X_BUF1 = 1
    X_BUF2 = 2
    X_BUF3 = 3
    STORE_EVENT0 = 4
    STORE_EVENT1 = 5
    META_READY0 = 6
    META_READY1 = 7
    META_ADDR_READY = 8
    X_ADDR_READY = 9
    OUT_ADDR_READY = 10
    SCALE_READY = 11

    @T.macro
    def consume_input(input_buf, compute_buf, acc_buf, scale, event, has_acc_ref):
        T.wait_flag("mte2", "v", event)

        if need_input_cast:
            T.tile.cast(compute_buf, input_buf, "CAST_NONE", hidden)
            T.pipe_barrier("V")

            if has_acc_ref == 0:
                T.tile.fill(acc_buf, 0.0)
                T.pipe_barrier("V")
                if with_weights or with_x_sf:
                    T.tile.axpy(acc_buf, compute_buf, scale)
                else:
                    T.tile.axpy(acc_buf, compute_buf, 1.0)
                has_acc_ref = 1
            else:
                if with_weights or with_x_sf:
                    T.tile.axpy(acc_buf, compute_buf, scale)
                else:
                    T.tile.axpy(acc_buf, compute_buf, 1.0)
        else:
            if has_acc_ref == 0:
                T.tile.fill(acc_buf, 0.0)
                T.pipe_barrier("V")
                if with_weights or with_x_sf:
                    T.tile.axpy(acc_buf, input_buf, scale)
                else:
                    T.tile.axpy(acc_buf, input_buf, 1.0)
                has_acc_ref = 1
            else:
                if with_weights or with_x_sf:
                    T.tile.axpy(acc_buf, input_buf, scale)
                else:
                    T.tile.axpy(acc_buf, input_buf, 1.0)

        T.pipe_barrier("V")
        T.set_flag("v", "mte2", event)
        T.wait_flag("v", "mte2", event)

    @T.macro
    def consume_input_mul_add(input_buf, compute_buf, acc_buf, scale, event, has_acc_ref):
        T.wait_flag("mte2", "v", event)
        if need_input_cast:
            T.tile.cast(compute_buf, input_buf, "CAST_NONE", hidden)
            T.pipe_barrier("V")
            if has_acc_ref == 0:
                if with_weights or with_x_sf:
                    T.tile.fill(acc_buf, 0.0)
                    T.pipe_barrier("V")
                    T.tile.axpy(acc_buf, compute_buf, scale)
                else:
                    T.tile.mul(acc_buf, compute_buf, 1.0)
                has_acc_ref = 1
            else:
                if with_weights or with_x_sf:
                    T.tile.axpy(acc_buf, compute_buf, scale)
                else:
                    T.tile.add(acc_buf, acc_buf, compute_buf)
        else:
            if has_acc_ref == 0:
                if with_weights or with_x_sf:
                    T.tile.fill(acc_buf, 0.0)
                    T.pipe_barrier("V")
                    T.tile.axpy(acc_buf, input_buf, scale)
                else:
                    T.tile.mul(acc_buf, input_buf, 1.0)
                has_acc_ref = 1
            else:
                if with_weights or with_x_sf:
                    T.tile.axpy(acc_buf, input_buf, scale)
                else:
                    T.tile.add(acc_buf, acc_buf, input_buf)
        T.pipe_barrier("V")
        T.set_flag("v", "mte2", event)
        T.wait_flag("v", "mte2", event)

    @T.macro
    def queue_input(source, input_buf, pos, event):
        if pos >= 0:
            T.copy(source[pos, 0:hidden], input_buf)
            T.set_flag("mte2", "v", event)

    @T.macro
    def load_route_state(mapping_ub, weights_ub, x_sf_gm, metadata_stage, route_offset, pos_ref, scale_ref):
        pos_ref = mapping_ub[metadata_stage, route_offset]
        scale_ref = 1.0
        if pos_ref >= 0:
            if with_weights:
                scale_ref = weights_ub[metadata_stage, route_offset]
            if with_x_sf:
                scale_ref = scale_ref * x_sf_gm[pos_ref]

    @T.macro
    def stage_output(output_buf, acc_buf, scale):
        if need_output_cast:
            T.tile.cast(output_buf, acc_buf, "CAST_RINT", hidden)
        else:
            if with_sf:
                T.tile.fill(output_buf, 0.0)
                T.pipe_barrier("V")
                T.tile.axpy(output_buf, acc_buf, scale)
            else:
                T.tile.mul(output_buf, acc_buf, 1.0)

        T.pipe_barrier("V")

    @T.macro
    def launch_output(output_buf, output_gm, output_token_base, store_event):
        T.set_flag("s", "mte3", OUT_ADDR_READY)
        T.wait_flag("s", "mte3", OUT_ADDR_READY)
        T.set_flag("v", "mte3", store_event)
        T.wait_flag("v", "mte3", store_event)

        if output_token_base + rows_per_vec <= num_tokens:
            T.copy(output_buf, output_gm[output_token_base : output_token_base + rows_per_vec, 0:hidden])
        else:
            if output_token_base < num_tokens:
                T.copy(output_buf[0, 0:hidden], output_gm[output_token_base, 0:hidden])
            if rows_per_vec >= 2:
                if output_token_base + 1 < num_tokens:
                    T.copy(output_buf[1, 0:hidden], output_gm[output_token_base + 1, 0:hidden])
            if rows_per_vec >= 4:
                if output_token_base + 2 < num_tokens:
                    T.copy(output_buf[2, 0:hidden], output_gm[output_token_base + 2, 0:hidden])
                if output_token_base + 3 < num_tokens:
                    T.copy(output_buf[3, 0:hidden], output_gm[output_token_base + 3, 0:hidden])

        T.set_flag("mte3", "v", store_event)

    @T.prim_func
    def reduce_fused_specialized_kernel(
        x: T.Tensor((num_expanded_tokens, hidden), tl_in_dtype),
        topk_weights_flat: T.Tensor((num_tokens * num_topk,), "float32"),
        token_topk_to_pos_flat: T.Tensor((num_tokens * num_topk,), "int32"),
        sf: T.Tensor((1,), "float32"),
        x_sf: T.Tensor((num_expanded_tokens,), "float32"),
        out: T.Tensor((num_tokens, hidden), tl_out_dtype),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid), T.Scope("V"):
            x_ub0 = T.alloc_ub((hidden,), tl_in_dtype)
            x_ub1 = T.alloc_ub((hidden,), tl_in_dtype)
            x_ub2 = T.alloc_ub((extra_x_buffer_len,), tl_in_dtype)
            x_ub3 = T.alloc_ub((extra_x_buffer_len,), tl_in_dtype)
            x_compute_f32_ub = T.alloc_ub((hidden,), "float32")
            acc_ub = T.alloc_ub((hidden,), "float32")
            output_batch_ub0 = T.alloc_ub((rows_per_vec, hidden), tl_out_dtype)
            output_batch_ub1 = T.alloc_ub((rows_per_vec, hidden), tl_out_dtype)
            topk_to_pos_rows_ub = T.alloc_ub((2, metadata_aligned_count), "int32")
            topk_weights_rows_ub = T.alloc_ub((2, metadata_aligned_count), "float32")

            block_id = T.alloc_var("int32", init=0)
            next_block_id = T.alloc_var("int32", init=0)
            token_base = T.alloc_var("int32", init=0)
            next_token_base = T.alloc_var("int32", init=0)
            token_id = T.alloc_var("int32", init=0)
            metadata_offset = T.alloc_var("int32", init=0)
            next_metadata_offset = T.alloc_var("int32", init=0)
            metadata_stage = T.alloc_var("int32", init=0)
            pending_output = T.alloc_var("int32", init=0)
            pending_output_stage = T.alloc_var("int32", init=0)
            pending_output_token_base = T.alloc_var("int32", init=0)
            has_acc = T.alloc_var("int32", init=0)
            output_scale = T.alloc_var("float32", init=1.0)

            # Allocate the maximum K=9 scalar state unconditionally. The
            # compile-time num_topk branch uses only the required prefix.
            p0 = T.alloc_var("int32", init=-1)
            p1 = T.alloc_var("int32", init=-1)
            p2 = T.alloc_var("int32", init=-1)
            p3 = T.alloc_var("int32", init=-1)
            p4 = T.alloc_var("int32", init=-1)
            p5 = T.alloc_var("int32", init=-1)
            p6 = T.alloc_var("int32", init=-1)
            p7 = T.alloc_var("int32", init=-1)
            p8 = T.alloc_var("int32", init=-1)

            # Used by compile-time K=2 and K=6 cross-row prefetch.
            next_p0 = T.alloc_var("int32", init=-1)
            next_p1 = T.alloc_var("int32", init=-1)
            next_p2 = T.alloc_var("int32", init=-1)
            next_p3 = T.alloc_var("int32", init=-1)
            head0_prefetched = T.alloc_var("int32", init=0)
            head1_prefetched = T.alloc_var("int32", init=0)
            head2_prefetched = T.alloc_var("int32", init=0)
            head3_prefetched = T.alloc_var("int32", init=0)
            head_metadata_consumed = T.alloc_var("int32", init=0)

            s0 = T.alloc_var("float32", init=1.0)
            s1 = T.alloc_var("float32", init=1.0)
            s2 = T.alloc_var("float32", init=1.0)
            s3 = T.alloc_var("float32", init=1.0)
            s4 = T.alloc_var("float32", init=1.0)
            s5 = T.alloc_var("float32", init=1.0)
            s6 = T.alloc_var("float32", init=1.0)
            s7 = T.alloc_var("float32", init=1.0)
            s8 = T.alloc_var("float32", init=1.0)

            # sf is invariant across all tokens handled by this AIV.
            if with_sf:
                output_scale = sf[0]
                T.set_flag("s", "v", SCALE_READY)
                T.wait_flag("s", "v", SCALE_READY)

            # Both output slots are initially writable by Vector. After a
            # store, MTE3 returns only the slot it has finished consuming.
            T.set_flag("mte3", "v", STORE_EVENT0)
            T.set_flag("mte3", "v", STORE_EVENT1)

            # Most paths give each AIV a contiguous token interval. Large-H
            # K=2 retains the old core-interleaved interval selected above.
            if use_interleaved_blocks:
                block_id = cid
            else:
                block_id = (cid * 2 + vid) * blocks_per_owner
            if block_id < num_token_blocks:
                if use_interleaved_blocks:
                    token_base = block_id * tokens_per_block + vid * rows_per_vec
                else:
                    token_base = block_id * tokens_per_block
                if token_base < num_tokens:
                    metadata_offset = token_base * num_topk
                    T.set_flag("s", "mte2", META_ADDR_READY)
                    T.wait_flag("s", "mte2", META_ADDR_READY)
                    T.copy(token_topk_to_pos_flat[metadata_offset : metadata_offset + metadata_count], topk_to_pos_rows_ub[0, 0:metadata_count], pad_value=-1)
                    if with_weights:
                        T.copy(topk_weights_flat[metadata_offset : metadata_offset + metadata_count], topk_weights_rows_ub[0, 0:metadata_count], pad_value=0.0)
                    T.set_flag("mte2", "s", META_READY0)

            for iter_idx in T.serial(num_iters):
                # Store block N-1 in the same loop body that starts block N.
                # Some Ascend lowering paths conservatively close an MTE3
                # issued at the end of one serial iteration before admitting
                # the next iteration's MTE2. Delaying the store removes that
                # loop boundary from the intended overlap window.
                if pending_output != 0:
                    if pending_output_stage == 0:
                        launch_output(output_batch_ub0, out, pending_output_token_base, STORE_EVENT0)
                    else:
                        launch_output(output_batch_ub1, out, pending_output_token_base, STORE_EVENT1)
                    pending_output = 0

                if use_interleaved_blocks:
                    block_id = cid + iter_idx * num_cores
                else:
                    block_id = (cid * 2 + vid) * blocks_per_owner + iter_idx
                if block_id < num_token_blocks:
                    if use_interleaved_blocks:
                        token_base = block_id * tokens_per_block + vid * rows_per_vec
                    else:
                        token_base = block_id * tokens_per_block

                    if token_base < num_tokens:
                        metadata_stage = iter_idx % 2

                        # Look one block ahead. This MTE2 transaction is queued
                        # before current-block Vector work, so the next loop can
                        # start its x loads immediately while this block stores.
                        # Do not prefetch across the interval owned by this
                        # AIV on its final iteration.
                        next_token_base = num_tokens
                        if iter_idx + 1 < num_iters:
                            if use_interleaved_blocks:
                                next_block_id = block_id + num_cores
                            else:
                                next_block_id = block_id + 1
                        else:
                            next_block_id = num_token_blocks
                        if next_block_id < num_token_blocks:
                            if use_interleaved_blocks:
                                next_token_base = next_block_id * tokens_per_block + vid * rows_per_vec
                            else:
                                next_token_base = next_block_id * tokens_per_block
                            if next_token_base < num_tokens:
                                next_metadata_offset = next_token_base * num_topk
                                T.set_flag("s", "mte2", META_ADDR_READY)
                                T.wait_flag("s", "mte2", META_ADDR_READY)
                                if metadata_stage == 0:
                                    T.copy(token_topk_to_pos_flat[next_metadata_offset : next_metadata_offset + metadata_count], topk_to_pos_rows_ub[1, 0:metadata_count], pad_value=-1)
                                    if with_weights:
                                        T.copy(topk_weights_flat[next_metadata_offset : next_metadata_offset + metadata_count], topk_weights_rows_ub[1, 0:metadata_count], pad_value=0.0)
                                    T.set_flag("mte2", "s", META_READY1)
                                else:
                                    T.copy(token_topk_to_pos_flat[next_metadata_offset : next_metadata_offset + metadata_count], topk_to_pos_rows_ub[0, 0:metadata_count], pad_value=-1)
                                    if with_weights:
                                        T.copy(topk_weights_flat[next_metadata_offset : next_metadata_offset + metadata_count], topk_weights_rows_ub[0, 0:metadata_count], pad_value=0.0)
                                    T.set_flag("mte2", "s", META_READY0)

                        # A cross-block head prefetch has already consumed this
                        # metadata-ready event in the preceding iteration.
                        if enable_cross_token_prefetch:
                            if head_metadata_consumed == 0:
                                if metadata_stage == 0:
                                    T.wait_flag("mte2", "s", META_READY0)
                                else:
                                    T.wait_flag("mte2", "s", META_READY1)
                            head_metadata_consumed = 0
                        else:
                            if metadata_stage == 0:
                                T.wait_flag("mte2", "s", META_READY0)
                            else:
                                T.wait_flag("mte2", "s", META_READY1)

                        for row in T.serial(rows_per_vec):
                            token_id = token_base + row
                            if token_id < num_tokens:
                                has_acc = 0

                                if num_topk == 2:
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk, p0, s0)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 1, p1, s1)

                                    if enable_cross_token_prefetch:
                                        next_p0 = -1
                                        next_p1 = -1
                                        if row + 1 < rows_per_vec:
                                            if token_id + 1 < num_tokens:
                                                next_p0 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk]
                                                next_p1 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk + 1]
                                        else:
                                            # The following block's metadata
                                            # was queued at this iteration's
                                            # prologue. Consume that event now
                                            # and continue the same k0 chain
                                            # across the block boundary.
                                            if next_block_id < num_token_blocks:
                                                if next_token_base < num_tokens:
                                                    if metadata_stage == 0:
                                                        T.wait_flag("mte2", "s", META_READY1)
                                                        next_p0 = topk_to_pos_rows_ub[1, 0]
                                                        next_p1 = topk_to_pos_rows_ub[1, 1]
                                                    else:
                                                        T.wait_flag("mte2", "s", META_READY0)
                                                        next_p0 = topk_to_pos_rows_ub[0, 0]
                                                        next_p1 = topk_to_pos_rows_ub[0, 1]
                                                    head_metadata_consumed = 1

                                    # All dynamic x addresses are now produced on Scalar.
                                    # One handshake covers the complete static K schedule.
                                    T.set_flag("s", "mte2", X_ADDR_READY)
                                    T.wait_flag("s", "mte2", X_ADDR_READY)

                                    if with_weights or with_x_sf:
                                        T.set_flag("s", "v", SCALE_READY)
                                        T.wait_flag("s", "v", SCALE_READY)

                                    # Seed both ping-pong lanes.
                                    if p0 >= 0:
                                        if enable_cross_token_prefetch:
                                            if head0_prefetched == 0:
                                                T.copy(x[p0, 0:hidden], x_ub0)
                                                T.set_flag("mte2", "v", X_BUF0)
                                        else:
                                            T.copy(x[p0, 0:hidden], x_ub0)
                                            T.set_flag("mte2", "v", X_BUF0)

                                    # A prefetched head now belongs to this row.
                                    if enable_cross_token_prefetch:
                                        head0_prefetched = 0

                                    if p1 >= 0:
                                        if enable_cross_token_prefetch:
                                            if head1_prefetched == 0:
                                                T.copy(x[p1, 0:hidden], x_ub1)
                                                T.set_flag("mte2", "v", X_BUF1)
                                        else:
                                            T.copy(x[p1, 0:hidden], x_ub1)
                                            T.set_flag("mte2", "v", X_BUF1)

                                    if enable_cross_token_prefetch:
                                        head1_prefetched = 0

                                    if p0 >= 0:
                                        consume_input(x_ub0, x_compute_f32_ub, acc_ub, s0, X_BUF0, has_acc)

                                    # Hide next-token k0 under current-token k1.
                                    if enable_cross_token_prefetch:
                                        if next_p0 >= 0:
                                            T.copy(x[next_p0, 0:hidden], x_ub0)
                                            T.set_flag("mte2", "v", X_BUF0)
                                            head0_prefetched = 1

                                    if p1 >= 0:
                                        consume_input(x_ub1, x_compute_f32_ub, acc_ub, s1, X_BUF1, has_acc)

                                    # The second head load overlaps output staging and
                                    # leaves both MTE2 lanes ready for the next token.
                                    if enable_cross_token_prefetch:
                                        if next_p1 >= 0:
                                            T.copy(x[next_p1, 0:hidden], x_ub1)
                                            T.set_flag("mte2", "v", X_BUF1)
                                            head1_prefetched = 1
                                        else:
                                            head1_prefetched = 0
                                elif num_topk == 6:
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 0, p0, s0)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 1, p1, s1)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 2, p2, s2)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 3, p3, s3)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 4, p4, s4)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 5, p5, s5)

                                    if enable_cross_token_prefetch:
                                        # Read the next row's first two positions while
                                        # its metadata is already resident.
                                        next_p0 = -1
                                        next_p1 = -1
                                        next_p2 = -1
                                        next_p3 = -1
                                        if row + 1 < rows_per_vec:
                                            if token_id + 1 < num_tokens:
                                                next_p0 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk]
                                                next_p1 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk + 1]
                                                next_p2 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk + 2]
                                                next_p3 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk + 3]
                                        else:
                                            if next_block_id < num_token_blocks:
                                                if next_token_base < num_tokens:
                                                    if metadata_stage == 0:
                                                        T.wait_flag("mte2", "s", META_READY1)
                                                        next_p0 = topk_to_pos_rows_ub[1, 0]
                                                        next_p1 = topk_to_pos_rows_ub[1, 1]
                                                        next_p2 = topk_to_pos_rows_ub[1, 2]
                                                        next_p3 = topk_to_pos_rows_ub[1, 3]
                                                    else:
                                                        T.wait_flag("mte2", "s", META_READY0)
                                                        next_p0 = topk_to_pos_rows_ub[0, 0]
                                                        next_p1 = topk_to_pos_rows_ub[0, 1]
                                                        next_p2 = topk_to_pos_rows_ub[0, 2]
                                                        next_p3 = topk_to_pos_rows_ub[0, 3]
                                                    head_metadata_consumed = 1

                                    # All current and next-token GM addresses are
                                    # scalar-produced before the rotating schedule starts.
                                    T.set_flag("s", "mte2", X_ADDR_READY)
                                    T.wait_flag("s", "mte2", X_ADDR_READY)

                                    if with_weights or with_x_sf:
                                        T.set_flag("s", "v", SCALE_READY)
                                        T.wait_flag("s", "v", SCALE_READY)

                                    # Fixed four-stage schedule. p0-p3 keep stable
                                    # slots; p4/p5 refill x0/x1, and the next token's
                                    # p2/p3/p0/p1 are queued as those slots retire.
                                    if p0 >= 0:
                                        if head0_prefetched == 0:
                                            T.copy(x[p0, 0:hidden], x_ub0)
                                            T.set_flag("mte2", "v", X_BUF0)
                                    if p1 >= 0:
                                        if head1_prefetched == 0:
                                            T.copy(x[p1, 0:hidden], x_ub1)
                                            T.set_flag("mte2", "v", X_BUF1)
                                    if p2 >= 0:
                                        if head2_prefetched == 0:
                                            T.copy(x[p2, 0:hidden], x_ub2)
                                            T.set_flag("mte2", "v", X_BUF2)
                                    if p3 >= 0:
                                        if head3_prefetched == 0:
                                            T.copy(x[p3, 0:hidden], x_ub3)
                                            T.set_flag("mte2", "v", X_BUF3)

                                    head0_prefetched = 0
                                    head1_prefetched = 0
                                    head2_prefetched = 0
                                    head3_prefetched = 0

                                    if p0 >= 0:
                                        consume_input(x_ub0, x_compute_f32_ub, acc_ub, s0, X_BUF0, has_acc)
                                    if p4 >= 0:
                                        T.copy(x[p4, 0:hidden], x_ub0)
                                        T.set_flag("mte2", "v", X_BUF0)

                                    if p1 >= 0:
                                        consume_input(x_ub1, x_compute_f32_ub, acc_ub, s1, X_BUF1, has_acc)
                                    if p5 >= 0:
                                        T.copy(x[p5, 0:hidden], x_ub1)
                                        T.set_flag("mte2", "v", X_BUF1)

                                    if p2 >= 0:
                                        consume_input(x_ub2, x_compute_f32_ub, acc_ub, s2, X_BUF2, has_acc)
                                    if next_p2 >= 0:
                                        T.copy(x[next_p2, 0:hidden], x_ub2)
                                        T.set_flag("mte2", "v", X_BUF2)
                                        head2_prefetched = 1

                                    if p3 >= 0:
                                        consume_input(x_ub3, x_compute_f32_ub, acc_ub, s3, X_BUF3, has_acc)
                                    if next_p3 >= 0:
                                        T.copy(x[next_p3, 0:hidden], x_ub3)
                                        T.set_flag("mte2", "v", X_BUF3)
                                        head3_prefetched = 1

                                    if p4 >= 0:
                                        consume_input(x_ub0, x_compute_f32_ub, acc_ub, s4, X_BUF0, has_acc)
                                    if next_p0 >= 0:
                                        T.copy(x[next_p0, 0:hidden], x_ub0)
                                        T.set_flag("mte2", "v", X_BUF0)
                                        head0_prefetched = 1

                                    if p5 >= 0:
                                        consume_input(x_ub1, x_compute_f32_ub, acc_ub, s5, X_BUF1, has_acc)
                                    if next_p1 >= 0:
                                        T.copy(x[next_p1, 0:hidden], x_ub1)
                                        T.set_flag("mte2", "v", X_BUF1)
                                        head1_prefetched = 1
                                elif num_topk == 8 and not use_four_input_schedule:
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 0, p0, s0)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 1, p1, s1)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 2, p2, s2)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 3, p3, s3)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 4, p4, s4)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 5, p5, s5)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 6, p6, s6)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 7, p7, s7)

                                    next_p0 = -1
                                    next_p1 = -1
                                    if row + 1 < rows_per_vec:
                                        if token_id + 1 < num_tokens:
                                            next_p0 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk]
                                            next_p1 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk + 1]
                                    else:
                                        if next_block_id < num_token_blocks:
                                            if next_token_base < num_tokens:
                                                if metadata_stage == 0:
                                                    T.wait_flag("mte2", "s", META_READY1)
                                                    next_p0 = topk_to_pos_rows_ub[1, 0]
                                                    next_p1 = topk_to_pos_rows_ub[1, 1]
                                                else:
                                                    T.wait_flag("mte2", "s", META_READY0)
                                                    next_p0 = topk_to_pos_rows_ub[0, 0]
                                                    next_p1 = topk_to_pos_rows_ub[0, 1]
                                                head_metadata_consumed = 1

                                    # All dynamic x addresses are now produced on Scalar.
                                    # One handshake covers the complete static K schedule.
                                    T.set_flag("s", "mte2", X_ADDR_READY)
                                    T.wait_flag("s", "mte2", X_ADDR_READY)

                                    if with_weights or with_x_sf:
                                        T.set_flag("s", "v", SCALE_READY)
                                        T.wait_flag("s", "v", SCALE_READY)

                                    # Seed both ping-pong lanes.
                                    if p0 >= 0:
                                        if head0_prefetched == 0:
                                            T.copy(x[p0, 0:hidden], x_ub0)
                                            T.set_flag("mte2", "v", X_BUF0)

                                    head0_prefetched = 0

                                    if p1 >= 0:
                                        if head1_prefetched == 0:
                                            T.copy(x[p1, 0:hidden], x_ub1)
                                            T.set_flag("mte2", "v", X_BUF1)

                                    head1_prefetched = 0

                                    if p0 >= 0:
                                        consume_input_mul_add(x_ub0, x_compute_f32_ub, acc_ub, s0, X_BUF0, has_acc)

                                    # Prefetch k=2 into the buffer just released.
                                    # It can overlap Vector consumption of k=1.
                                    if p2 >= 0:
                                        T.copy(x[p2, 0:hidden], x_ub0)
                                        T.set_flag("mte2", "v", X_BUF0)

                                    if p1 >= 0:
                                        consume_input_mul_add(x_ub1, x_compute_f32_ub, acc_ub, s1, X_BUF1, has_acc)

                                    # Prefetch k=3 into the buffer just released.
                                    # It can overlap Vector consumption of k=2.
                                    if p3 >= 0:
                                        T.copy(x[p3, 0:hidden], x_ub1)
                                        T.set_flag("mte2", "v", X_BUF1)

                                    if p2 >= 0:
                                        consume_input_mul_add(x_ub0, x_compute_f32_ub, acc_ub, s2, X_BUF0, has_acc)

                                    # Prefetch k=4 into the buffer just released.
                                    # It can overlap Vector consumption of k=3.
                                    if p4 >= 0:
                                        T.copy(x[p4, 0:hidden], x_ub0)
                                        T.set_flag("mte2", "v", X_BUF0)

                                    if p3 >= 0:
                                        consume_input_mul_add(x_ub1, x_compute_f32_ub, acc_ub, s3, X_BUF1, has_acc)

                                    # Prefetch k=5 into the buffer just released.
                                    # It can overlap Vector consumption of k=4.
                                    if p5 >= 0:
                                        T.copy(x[p5, 0:hidden], x_ub1)
                                        T.set_flag("mte2", "v", X_BUF1)

                                    if p4 >= 0:
                                        consume_input_mul_add(x_ub0, x_compute_f32_ub, acc_ub, s4, X_BUF0, has_acc)

                                    # Prefetch k=6 into the buffer just released.
                                    # It can overlap Vector consumption of k=5.
                                    if p6 >= 0:
                                        T.copy(x[p6, 0:hidden], x_ub0)
                                        T.set_flag("mte2", "v", X_BUF0)

                                    if p5 >= 0:
                                        consume_input_mul_add(x_ub1, x_compute_f32_ub, acc_ub, s5, X_BUF1, has_acc)

                                    # Prefetch k=7 into the buffer just released.
                                    # It can overlap Vector consumption of k=6.
                                    if p7 >= 0:
                                        T.copy(x[p7, 0:hidden], x_ub1)
                                        T.set_flag("mte2", "v", X_BUF1)

                                    if p6 >= 0:
                                        consume_input_mul_add(x_ub0, x_compute_f32_ub, acc_ub, s6, X_BUF0, has_acc)

                                    if next_p0 >= 0:
                                        T.copy(x[next_p0, 0:hidden], x_ub0)
                                        T.set_flag("mte2", "v", X_BUF0)
                                        head0_prefetched = 1
                                    else:
                                        head0_prefetched = 0

                                    if p7 >= 0:
                                        consume_input_mul_add(x_ub1, x_compute_f32_ub, acc_ub, s7, X_BUF1, has_acc)

                                    if next_p1 >= 0:
                                        T.copy(x[next_p1, 0:hidden], x_ub1)
                                        T.set_flag("mte2", "v", X_BUF1)
                                        head1_prefetched = 1
                                    else:
                                        head1_prefetched = 0
                                elif num_topk == 8:
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 0, p0, s0)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 1, p1, s1)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 2, p2, s2)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 3, p3, s3)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 4, p4, s4)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 5, p5, s5)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 6, p6, s6)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 7, p7, s7)

                                    next_p0 = -1
                                    next_p1 = -1
                                    next_p2 = -1
                                    next_p3 = -1
                                    if row + 1 < rows_per_vec:
                                        if token_id + 1 < num_tokens:
                                            next_p0 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk + 0]
                                            next_p1 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk + 1]
                                            next_p2 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk + 2]
                                            next_p3 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk + 3]
                                    else:
                                        if next_block_id < num_token_blocks:
                                            if next_token_base < num_tokens:
                                                if metadata_stage == 0:
                                                    T.wait_flag("mte2", "s", META_READY1)
                                                    next_p0 = topk_to_pos_rows_ub[1, 0]
                                                    next_p1 = topk_to_pos_rows_ub[1, 1]
                                                    next_p2 = topk_to_pos_rows_ub[1, 2]
                                                    next_p3 = topk_to_pos_rows_ub[1, 3]
                                                else:
                                                    T.wait_flag("mte2", "s", META_READY0)
                                                    next_p0 = topk_to_pos_rows_ub[0, 0]
                                                    next_p1 = topk_to_pos_rows_ub[0, 1]
                                                    next_p2 = topk_to_pos_rows_ub[0, 2]
                                                    next_p3 = topk_to_pos_rows_ub[0, 3]
                                                head_metadata_consumed = 1

                                    T.set_flag("s", "mte2", X_ADDR_READY)
                                    T.wait_flag("s", "mte2", X_ADDR_READY)
                                    if with_weights or with_x_sf:
                                        T.set_flag("s", "v", SCALE_READY)
                                        T.wait_flag("s", "v", SCALE_READY)

                                    if head0_prefetched == 0:
                                        queue_input(x, x_ub0, p0, X_BUF0)
                                    if head1_prefetched == 0:
                                        queue_input(x, x_ub1, p1, X_BUF1)
                                    if head2_prefetched == 0:
                                        queue_input(x, x_ub2, p2, X_BUF2)
                                    if head3_prefetched == 0:
                                        queue_input(x, x_ub3, p3, X_BUF3)
                                    head0_prefetched = 0
                                    head1_prefetched = 0
                                    head2_prefetched = 0
                                    head3_prefetched = 0

                                    if p0 >= 0:
                                        consume_input(x_ub0, x_compute_f32_ub, acc_ub, s0, X_BUF0, has_acc)
                                    queue_input(x, x_ub0, p4, X_BUF0)

                                    if p1 >= 0:
                                        consume_input(x_ub1, x_compute_f32_ub, acc_ub, s1, X_BUF1, has_acc)
                                    queue_input(x, x_ub1, p5, X_BUF1)

                                    if p2 >= 0:
                                        consume_input(x_ub2, x_compute_f32_ub, acc_ub, s2, X_BUF2, has_acc)
                                    queue_input(x, x_ub2, p6, X_BUF2)

                                    if p3 >= 0:
                                        consume_input(x_ub3, x_compute_f32_ub, acc_ub, s3, X_BUF3, has_acc)
                                    queue_input(x, x_ub3, p7, X_BUF3)

                                    if p4 >= 0:
                                        consume_input(x_ub0, x_compute_f32_ub, acc_ub, s4, X_BUF0, has_acc)
                                    if next_p0 >= 0:
                                        queue_input(x, x_ub0, next_p0, X_BUF0)
                                        head0_prefetched = 1

                                    if p5 >= 0:
                                        consume_input(x_ub1, x_compute_f32_ub, acc_ub, s5, X_BUF1, has_acc)
                                    if next_p1 >= 0:
                                        queue_input(x, x_ub1, next_p1, X_BUF1)
                                        head1_prefetched = 1

                                    if p6 >= 0:
                                        consume_input(x_ub2, x_compute_f32_ub, acc_ub, s6, X_BUF2, has_acc)
                                    if next_p2 >= 0:
                                        queue_input(x, x_ub2, next_p2, X_BUF2)
                                        head2_prefetched = 1

                                    if p7 >= 0:
                                        consume_input(x_ub3, x_compute_f32_ub, acc_ub, s7, X_BUF3, has_acc)
                                    if next_p3 >= 0:
                                        queue_input(x, x_ub3, next_p3, X_BUF3)
                                        head3_prefetched = 1
                                elif num_topk == 9 and not use_four_input_schedule:
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 0, p0, s0)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 1, p1, s1)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 2, p2, s2)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 3, p3, s3)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 4, p4, s4)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 5, p5, s5)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 6, p6, s6)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 7, p7, s7)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 8, p8, s8)

                                    next_p0 = -1
                                    next_p1 = -1
                                    if row + 1 < rows_per_vec:
                                        if token_id + 1 < num_tokens:
                                            next_p0 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk]
                                            next_p1 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk + 1]
                                    else:
                                        if next_block_id < num_token_blocks:
                                            if next_token_base < num_tokens:
                                                if metadata_stage == 0:
                                                    T.wait_flag("mte2", "s", META_READY1)
                                                    next_p0 = topk_to_pos_rows_ub[1, 0]
                                                    next_p1 = topk_to_pos_rows_ub[1, 1]
                                                else:
                                                    T.wait_flag("mte2", "s", META_READY0)
                                                    next_p0 = topk_to_pos_rows_ub[0, 0]
                                                    next_p1 = topk_to_pos_rows_ub[0, 1]
                                                head_metadata_consumed = 1

                                    # All dynamic x addresses are now produced on Scalar.
                                    # One handshake covers the complete static K schedule.
                                    T.set_flag("s", "mte2", X_ADDR_READY)
                                    T.wait_flag("s", "mte2", X_ADDR_READY)

                                    if with_weights or with_x_sf:
                                        T.set_flag("s", "v", SCALE_READY)
                                        T.wait_flag("s", "v", SCALE_READY)

                                    # Seed both ping-pong lanes.
                                    if p0 >= 0:
                                        if head0_prefetched == 0:
                                            T.copy(x[p0, 0:hidden], x_ub0)
                                            T.set_flag("mte2", "v", X_BUF0)

                                    head0_prefetched = 0

                                    if p1 >= 0:
                                        if head1_prefetched == 0:
                                            T.copy(x[p1, 0:hidden], x_ub1)
                                            T.set_flag("mte2", "v", X_BUF1)

                                    head1_prefetched = 0

                                    if p0 >= 0:
                                        consume_input_mul_add(x_ub0, x_compute_f32_ub, acc_ub, s0, X_BUF0, has_acc)

                                    # Prefetch k=2 into the buffer just released.
                                    # It can overlap Vector consumption of k=1.
                                    if p2 >= 0:
                                        T.copy(x[p2, 0:hidden], x_ub0)
                                        T.set_flag("mte2", "v", X_BUF0)

                                    if p1 >= 0:
                                        consume_input_mul_add(x_ub1, x_compute_f32_ub, acc_ub, s1, X_BUF1, has_acc)

                                    # Prefetch k=3 into the buffer just released.
                                    # It can overlap Vector consumption of k=2.
                                    if p3 >= 0:
                                        T.copy(x[p3, 0:hidden], x_ub1)
                                        T.set_flag("mte2", "v", X_BUF1)

                                    if p2 >= 0:
                                        consume_input_mul_add(x_ub0, x_compute_f32_ub, acc_ub, s2, X_BUF0, has_acc)

                                    # Prefetch k=4 into the buffer just released.
                                    # It can overlap Vector consumption of k=3.
                                    if p4 >= 0:
                                        T.copy(x[p4, 0:hidden], x_ub0)
                                        T.set_flag("mte2", "v", X_BUF0)

                                    if p3 >= 0:
                                        consume_input_mul_add(x_ub1, x_compute_f32_ub, acc_ub, s3, X_BUF1, has_acc)

                                    # Prefetch k=5 into the buffer just released.
                                    # It can overlap Vector consumption of k=4.
                                    if p5 >= 0:
                                        T.copy(x[p5, 0:hidden], x_ub1)
                                        T.set_flag("mte2", "v", X_BUF1)

                                    if p4 >= 0:
                                        consume_input_mul_add(x_ub0, x_compute_f32_ub, acc_ub, s4, X_BUF0, has_acc)

                                    # Prefetch k=6 into the buffer just released.
                                    # It can overlap Vector consumption of k=5.
                                    if p6 >= 0:
                                        T.copy(x[p6, 0:hidden], x_ub0)
                                        T.set_flag("mte2", "v", X_BUF0)

                                    if p5 >= 0:
                                        consume_input_mul_add(x_ub1, x_compute_f32_ub, acc_ub, s5, X_BUF1, has_acc)

                                    # Prefetch k=7 into the buffer just released.
                                    # It can overlap Vector consumption of k=6.
                                    if p7 >= 0:
                                        T.copy(x[p7, 0:hidden], x_ub1)
                                        T.set_flag("mte2", "v", X_BUF1)

                                    if p6 >= 0:
                                        consume_input_mul_add(x_ub0, x_compute_f32_ub, acc_ub, s6, X_BUF0, has_acc)

                                    # Prefetch k=8 into the buffer just released.
                                    # It can overlap Vector consumption of k=7.
                                    if p8 >= 0:
                                        T.copy(x[p8, 0:hidden], x_ub0)
                                        T.set_flag("mte2", "v", X_BUF0)

                                    if p7 >= 0:
                                        consume_input_mul_add(x_ub1, x_compute_f32_ub, acc_ub, s7, X_BUF1, has_acc)

                                    if next_p1 >= 0:
                                        T.copy(x[next_p1, 0:hidden], x_ub1)
                                        T.set_flag("mte2", "v", X_BUF1)
                                        head1_prefetched = 1
                                    else:
                                        head1_prefetched = 0

                                    if p8 >= 0:
                                        consume_input_mul_add(x_ub0, x_compute_f32_ub, acc_ub, s8, X_BUF0, has_acc)

                                    if next_p0 >= 0:
                                        T.copy(x[next_p0, 0:hidden], x_ub0)
                                        T.set_flag("mte2", "v", X_BUF0)
                                        head0_prefetched = 1
                                    else:
                                        head0_prefetched = 0
                                elif num_topk == 9:
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 0, p0, s0)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 1, p1, s1)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 2, p2, s2)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 3, p3, s3)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 4, p4, s4)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 5, p5, s5)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 6, p6, s6)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 7, p7, s7)
                                    load_route_state(topk_to_pos_rows_ub, topk_weights_rows_ub, x_sf, metadata_stage, row * num_topk + 8, p8, s8)

                                    next_p0 = -1
                                    next_p1 = -1
                                    next_p2 = -1
                                    next_p3 = -1
                                    if row + 1 < rows_per_vec:
                                        if token_id + 1 < num_tokens:
                                            next_p0 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk + 0]
                                            next_p1 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk + 1]
                                            next_p2 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk + 2]
                                            next_p3 = topk_to_pos_rows_ub[metadata_stage, (row + 1) * num_topk + 3]
                                    else:
                                        if next_block_id < num_token_blocks:
                                            if next_token_base < num_tokens:
                                                if metadata_stage == 0:
                                                    T.wait_flag("mte2", "s", META_READY1)
                                                    next_p0 = topk_to_pos_rows_ub[1, 0]
                                                    next_p1 = topk_to_pos_rows_ub[1, 1]
                                                    next_p2 = topk_to_pos_rows_ub[1, 2]
                                                    next_p3 = topk_to_pos_rows_ub[1, 3]
                                                else:
                                                    T.wait_flag("mte2", "s", META_READY0)
                                                    next_p0 = topk_to_pos_rows_ub[0, 0]
                                                    next_p1 = topk_to_pos_rows_ub[0, 1]
                                                    next_p2 = topk_to_pos_rows_ub[0, 2]
                                                    next_p3 = topk_to_pos_rows_ub[0, 3]
                                                head_metadata_consumed = 1

                                    T.set_flag("s", "mte2", X_ADDR_READY)
                                    T.wait_flag("s", "mte2", X_ADDR_READY)
                                    if with_weights or with_x_sf:
                                        T.set_flag("s", "v", SCALE_READY)
                                        T.wait_flag("s", "v", SCALE_READY)

                                    if head0_prefetched == 0:
                                        queue_input(x, x_ub0, p0, X_BUF0)
                                    if head1_prefetched == 0:
                                        queue_input(x, x_ub1, p1, X_BUF1)
                                    if head2_prefetched == 0:
                                        queue_input(x, x_ub2, p2, X_BUF2)
                                    if head3_prefetched == 0:
                                        queue_input(x, x_ub3, p3, X_BUF3)
                                    head0_prefetched = 0
                                    head1_prefetched = 0
                                    head2_prefetched = 0
                                    head3_prefetched = 0

                                    if p0 >= 0:
                                        consume_input(x_ub0, x_compute_f32_ub, acc_ub, s0, X_BUF0, has_acc)
                                    queue_input(x, x_ub0, p4, X_BUF0)

                                    if p1 >= 0:
                                        consume_input(x_ub1, x_compute_f32_ub, acc_ub, s1, X_BUF1, has_acc)
                                    queue_input(x, x_ub1, p5, X_BUF1)

                                    if p2 >= 0:
                                        consume_input(x_ub2, x_compute_f32_ub, acc_ub, s2, X_BUF2, has_acc)
                                    queue_input(x, x_ub2, p6, X_BUF2)

                                    if p3 >= 0:
                                        consume_input(x_ub3, x_compute_f32_ub, acc_ub, s3, X_BUF3, has_acc)
                                    queue_input(x, x_ub3, p7, X_BUF3)

                                    if p4 >= 0:
                                        consume_input(x_ub0, x_compute_f32_ub, acc_ub, s4, X_BUF0, has_acc)
                                    queue_input(x, x_ub0, p8, X_BUF0)

                                    if p5 >= 0:
                                        consume_input(x_ub1, x_compute_f32_ub, acc_ub, s5, X_BUF1, has_acc)
                                    if next_p1 >= 0:
                                        queue_input(x, x_ub1, next_p1, X_BUF1)
                                        head1_prefetched = 1

                                    if p6 >= 0:
                                        consume_input(x_ub2, x_compute_f32_ub, acc_ub, s6, X_BUF2, has_acc)
                                    if next_p2 >= 0:
                                        queue_input(x, x_ub2, next_p2, X_BUF2)
                                        head2_prefetched = 1

                                    if p7 >= 0:
                                        consume_input(x_ub3, x_compute_f32_ub, acc_ub, s7, X_BUF3, has_acc)
                                    if next_p3 >= 0:
                                        queue_input(x, x_ub3, next_p3, X_BUF3)
                                        head3_prefetched = 1

                                    if p8 >= 0:
                                        consume_input(x_ub0, x_compute_f32_ub, acc_ub, s8, X_BUF0, has_acc)
                                    if next_p0 >= 0:
                                        queue_input(x, x_ub0, next_p0, X_BUF0)
                                        head0_prefetched = 1

                                if has_acc == 0:
                                    T.tile.fill(acc_ub, 0.0)
                                    T.pipe_barrier("V")

                                if with_sf and need_output_cast:
                                    T.tile.mul(acc_ub, acc_ub, output_scale)
                                    T.pipe_barrier("V")
                                # Keep both BufferRegions static. Waiting only
                                # at row 0 gives the previous MTE3 store one
                                # complete intervening batch in which to finish.
                                if iter_idx % 2 == 0:
                                    if row == 0:
                                        T.wait_flag("mte3", "v", STORE_EVENT0)
                                        stage_output(output_batch_ub0[0, 0:hidden], acc_ub, output_scale)
                                    if rows_per_vec >= 2:
                                        if row == 1:
                                            stage_output(output_batch_ub0[1, 0:hidden], acc_ub, output_scale)
                                    if rows_per_vec >= 4:
                                        if row == 2:
                                            stage_output(output_batch_ub0[2, 0:hidden], acc_ub, output_scale)
                                        if row == 3:
                                            stage_output(output_batch_ub0[3, 0:hidden], acc_ub, output_scale)
                                else:
                                    if row == 0:
                                        T.wait_flag("mte3", "v", STORE_EVENT1)
                                        stage_output(output_batch_ub1[0, 0:hidden], acc_ub, output_scale)
                                    if rows_per_vec >= 2:
                                        if row == 1:
                                            stage_output(output_batch_ub1[1, 0:hidden], acc_ub, output_scale)
                                    if rows_per_vec >= 4:
                                        if row == 2:
                                            stage_output(output_batch_ub1[2, 0:hidden], acc_ub, output_scale)
                                        if row == 3:
                                            stage_output(output_batch_ub1[3, 0:hidden], acc_ub, output_scale)

                        # The completed batch becomes the pending store for
                        # the next iteration. That iteration launches MTE3 and
                        # then immediately proceeds into its metadata/x MTE2
                        # and Vector work using the alternate output slot.
                        pending_output = 1
                        pending_output_stage = iter_idx % 2
                        pending_output_token_base = token_base

            # Epilogue: the last block has no following iteration in which to
            # launch its delayed store.
            if pending_output != 0:
                if pending_output_stage == 0:
                    launch_output(output_batch_ub0, out, pending_output_token_base, STORE_EVENT0)
                else:
                    launch_output(output_batch_ub1, out, pending_output_token_base, STORE_EVENT1)

            # Drain both slots. An unused slot still owns its initial event.
            T.wait_flag("mte3", "v", STORE_EVENT0)
            T.wait_flag("mte3", "v", STORE_EVENT1)

    return reduce_fused_specialized_kernel


def get_reduce_fused_kernel(hidden: int, num_topk: int, in_dtype: torch.dtype, out_dtype: torch.dtype, with_sf: bool, with_weights: bool, with_x_sf: bool):
    """Dispatch common TopK values to specialized kernels."""
    args = (hidden, num_topk, in_dtype, out_dtype, with_sf, with_weights, with_x_sf)
    if num_topk in (2, 6, 8, 9):
        return get_reduce_fused_specialized_kernel(*args)
    return get_reduce_fused_generic_kernel(*args)


def reduce_fused(
    x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], topk_weights: Optional[torch.Tensor], token_topk_to_pos: torch.Tensor, fp8_format: str = "", sf: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Reduce expanded expert rows directly into a caller-owned output tensor."""
    if fp8_format != "":
        raise AssertionError("Ascend example reduce_fused currently supports only non-FP8 output.")

    if isinstance(x, tuple):
        x, x_sf = x
    else:
        x_sf = None

    assert x.dim() == 2 and x.is_contiguous()
    assert token_topk_to_pos.dim() == 2
    assert token_topk_to_pos.dtype == torch.int32

    num_expanded_tokens, hidden = x.shape
    num_tokens, num_topk = token_topk_to_pos.shape
    device = x.device

    if topk_weights is None:
        # The compile-time with_weights=False branch never reads this tensor.
        topk_weights_arg = torch.empty((num_tokens * num_topk,), dtype=torch.float32, device=device)
    else:
        assert topk_weights.shape == (num_tokens, num_topk)
        assert topk_weights.dtype == torch.float32
        assert topk_weights.device == device
        topk_weights_arg = topk_weights.contiguous().view(-1)

    if sf is None:
        # The compile-time with_sf=False branch never reads this tensor.
        sf_arg = torch.empty((1,), dtype=torch.float32, device=device)
    else:
        assert sf.shape == (1,)
        assert sf.dtype == torch.float32
        assert sf.device == device
        sf_arg = sf.contiguous()

    if x_sf is None:
        # The compile-time with_x_sf=False branch never reads this tensor.
        x_sf_arg = torch.empty((num_expanded_tokens,), dtype=torch.float32, device=device)
    else:
        assert x_sf.shape == (num_expanded_tokens,)
        assert x_sf.dtype == torch.float32
        assert x_sf.device == device
        x_sf_arg = x_sf.contiguous()

    # A contiguous view is metadata-only: no device data copy is introduced.
    # The kernel receives one flat GM tensor and issues one 1-D metadata DMA.
    token_topk_to_pos_arg = token_topk_to_pos.contiguous().view(-1)

    # P0: allocate only when the caller did not supply an output. In both cases
    # the tensor is passed into the kernel and written directly, so there is no
    # result -> user_out device-to-device copy.
    if out is None:
        out = torch.empty((num_tokens, hidden), dtype=x.dtype, device=device)
    else:
        assert out.shape == (num_tokens, hidden)
        assert out.dtype == x.dtype
        assert out.device == device
        assert out.is_contiguous()

    if num_tokens == 0:
        return out

    kernel = get_reduce_fused_kernel(hidden=hidden, num_topk=num_topk, in_dtype=x.dtype, out_dtype=out.dtype, with_sf=sf is not None, with_weights=topk_weights is not None, with_x_sf=x_sf is not None)

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", "0")):
        print(kernel.get_kernel_source())

    kernel(x, topk_weights_arg, token_topk_to_pos_arg, sf_arg, x_sf_arg, out)
    return out


def build_reduce_fused_func(hidden: int, num_topk: int, dtype: torch.dtype = torch.float32, with_weights: bool = True) -> Callable[[torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor]:
    """Compile the kernel once and return a SwiGLU-style callable."""
    kernel = get_reduce_fused_kernel(hidden=hidden, num_topk=num_topk, in_dtype=dtype, out_dtype=dtype, with_sf=False, with_weights=with_weights, with_x_sf=False)

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", "0")):
        print(kernel.get_kernel_source())

    def run(x: torch.Tensor, topk_weights: Optional[torch.Tensor], token_topk_to_pos: torch.Tensor) -> torch.Tensor:
        num_expanded_tokens = x.shape[0]
        num_tokens = token_topk_to_pos.shape[0]

        if with_weights:
            topk_weights_arg = topk_weights.contiguous().view(-1)
        else:
            # This compile-time path never reads the placeholder.
            topk_weights_arg = torch.empty(num_tokens * num_topk, dtype=torch.float32, device=x.device)

        token_topk_to_pos_arg = token_topk_to_pos.contiguous().view(-1)

        # sf/x_sf are compile-time disabled and are never read.
        sf_arg = torch.empty(1, dtype=torch.float32, device=x.device)
        x_sf_arg = torch.empty(num_expanded_tokens, dtype=torch.float32, device=x.device)

        out = torch.empty(num_tokens, hidden, dtype=dtype, device=x.device)

        kernel(x, topk_weights_arg, token_topk_to_pos_arg, sf_arg, x_sf_arg, out)
        return out

    # Useful when source inspection is needed.
    run.kernel = kernel
    return run


def get_ascendc_reduce_fused() -> Callable:
    """Return the CANN MoE token-unpermute API."""
    # This is the path already verified by the previous CANN comparison.
    if torch_npu is not None:
        op = getattr(torch_npu, "npu_moe_token_unpermute", None)
        if op is not None:
            return op

    # Fall back to the torch.ops registration on environments that expose
    # the operator only through the dispatcher namespace.
    try:
        op = torch.ops.npu.npu_moe_token_unpermute
        if op is not None:
            return op
    except (AttributeError, RuntimeError):
        pass

    raise RuntimeError("The installed torch_npu/CANN does not expose npu_moe_token_unpermute")


def make_dense_mapping(num_tokens: int, num_topk: int, random_permute: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the flattened route mapping shared by TileLang and CANN."""
    num_expanded_tokens = num_tokens * num_topk
    if random_permute:
        mapping_flat = torch.randperm(num_expanded_tokens, dtype=torch.int32, device="cpu")
    else:
        mapping_flat = torch.arange(num_expanded_tokens, dtype=torch.int32, device="cpu")

    token_topk_to_pos = mapping_flat.view(num_tokens, num_topk).contiguous()
    sorted_indices = mapping_flat.contiguous()
    return token_topk_to_pos, sorted_indices


def verify_result(output, golden, rtol=1e-3, atol=1e-3, error_tol=1e-4):
    """Use the same element-ratio validation style as the SwiGLU script."""
    output = output.reshape(-1)
    golden = golden.reshape(-1)
    assert output.dtype == golden.dtype

    if output.dtype in (torch.float16, torch.bfloat16, torch.float):
        output = output.to(torch.float64)
        golden = golden.to(torch.float64)

    different_element_results = torch.isclose(output, golden, rtol=rtol, atol=atol, equal_nan=True)
    different_element_indexes = torch.where(~different_element_results)[0]
    error_ratio = float(different_element_indexes.numel()) / golden.numel()
    return error_ratio <= error_tol


def main():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("An available Ascend NPU is required")

    torch.manual_seed(0)

    # These are the representative paths of the current best kernel.
    # Add/remove cases here without changing the validation code.
    # num_tokens, hidden, topk, dtype
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

    # CANN comparison covers weighted and no-weight paths. sf/x_sf are not
    # included because npu_moe_token_unpermute does not implement those
    # reduce_fused extensions.
    weight_modes = (False, True)

    random_permute = True
    # Validate the TileLang kernel against the CANN reference implementation.
    profile_only = False

    cann_op = None if profile_only else get_ascendc_reduce_fused()

    failures = []

    for num_tokens, hidden, num_topk, dtype in test_configs:
        for with_weights in weight_modes:
            case = f"reduce_fused T={num_tokens} H={hidden} K={num_topk} dtype={dtype} with_weights={with_weights}"
            num_expanded_tokens = num_tokens * num_topk
            token_topk_to_pos_cpu, sorted_indices_cpu = make_dense_mapping(num_tokens=num_tokens, num_topk=num_topk, random_permute=random_permute)
            x = torch.randn(num_expanded_tokens, hidden, dtype=dtype, device="npu")
            token_topk_to_pos = token_topk_to_pos_cpu.npu()
            sorted_indices = None if profile_only else sorted_indices_cpu.npu()

            if with_weights:
                topk_weights = torch.rand(num_tokens, num_topk, dtype=torch.float32, device="cpu")
                topk_weights = (topk_weights / topk_weights.sum(dim=1, keepdim=True)).contiguous().npu()
                cann_probs = None if profile_only else topk_weights
            else:
                topk_weights = None
                # CANN probs=None only unpermutes. Passing ones makes
                # CANN reduce the K dimension like no-weight TileLang.
                # Some profiling environments cannot initialize the
                # dynamic OnesLike kernel (aclnnInplaceOne). Build this
                # reference-only tensor on CPU and copy it to NPU instead.
                cann_probs = None
                if not profile_only:
                    cann_probs = torch.ones(num_tokens, num_topk, dtype=torch.float32, device="cpu").npu()

            # Same structure as the supplied SwiGLU validation:
            # build function -> call TileLang -> call AscendC -> compare.
            func = build_reduce_fused_func(hidden=hidden, num_topk=num_topk, dtype=dtype, with_weights=with_weights)
            tilelang_out = func(x, topk_weights, token_topk_to_pos)
            torch.npu.synchronize()

            if profile_only:
                print(f"pass {case}")
                del tilelang_out, x, token_topk_to_pos
                if topk_weights is not None:
                    del topk_weights
                torch.npu.synchronize()
                continue

            ascendc_out = cann_op(x, sorted_indices, cann_probs)
            torch.npu.synchronize()

            tilelang_cpu = tilelang_out.cpu()
            ascendc_cpu = ascendc_out.cpu()

            result_ok = verify_result(tilelang_cpu, ascendc_cpu, rtol=1e-4, atol=1e-4, error_tol=1e-4)

            try:
                torch.testing.assert_close(tilelang_cpu, ascendc_cpu, rtol=1e-4, atol=1e-4)
                strict_close = True
            except AssertionError:
                strict_close = False

            if result_ok and strict_close:
                print(f"pass {case}")
            else:
                failures.append(case)

            del tilelang_out, ascendc_out, tilelang_cpu, ascendc_cpu, x, token_topk_to_pos, sorted_indices, cann_probs
            if topk_weights is not None:
                del topk_weights
            torch.npu.synchronize()

    if failures:
        raise AssertionError("reduce_fused validation failed: " + "; ".join(failures))

    print("TEST PASSED!")


if __name__ == "__main__":
    main()
