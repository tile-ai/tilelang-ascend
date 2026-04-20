# adapted from vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# SPDX-License-Identifier: Apache-2.0

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

PAD_SLOT_ID = -1

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}

_kernel_cache_fn = {}
_kernel_cache_update = {}


def clear_all_caches():
    global _kernel_cache_fn, _kernel_cache_update
    _kernel_cache_fn = {}
    _kernel_cache_update = {}
    tilelang.cache.clear_cache()


# ============================================================================
# Kernel 1: Prefill (FN mode - process full sequences)
# ============================================================================


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def causal_conv1d_fn_kernel(
    batch_size: int,
    width: int,
    has_bias: bool,
    has_activation: bool,
    has_cache_indices: bool,
    has_initial_state_mode: bool,
    block_M: int = 64,
    block_D: int = 512,
):
    """
    Prefill kernel: FN VARLEN mode

    Reference: causal_conv1d_varlen_with_update.py

    Grid = batch_size * dim_num * seqlen_num
    Each kernel processes one batch, one token block, one dim block

    Input:
    - x: (total_len, dim) packed layout
    - weight: (width, dim)
    - conv_state: (num_cache_lines, state_len, dim)
    - cu_seqlens: (batch_size + 1)
    - cache_indices: (batch_size) optional
    - initial_state_mode: (batch_size) optional

    Output:
    - y: (total_len, dim)
    - conv_state: update last width-1 tokens
    """
    dtype = "float"

    dim = T.symbolic("dim")
    total_len = T.symbolic("total_len")
    num_cache_lines = T.symbolic("num_cache_lines")
    state_len = T.symbolic("state_len")

    dim_num = T.ceildiv(dim, block_D)
    seqlen_num = T.ceildiv(total_len, block_M)
    grid_size = dim_num * seqlen_num * batch_size

    @T.prim_func
    def main(
        x: T.Tensor((total_len, dim), dtype),
        weight: T.Tensor((width, dim), dtype),
        bias: T.Tensor((dim,), dtype) if has_bias else T.Tensor((1,), dtype),
        conv_state: T.Tensor((num_cache_lines, state_len, dim), dtype),
        cu_seqlens: T.Tensor((batch_size + 1,), "int32"),
        cache_indices: T.Tensor((batch_size,), "int32"),
        initial_state_mode: T.Tensor((batch_size,), "int32"),
        y: T.Tensor((total_len, dim), dtype),
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            batch_id = cid // (dim_num * seqlen_num)
            remaining = cid % (dim_num * seqlen_num)
            seq_block = remaining // dim_num
            dim_block = remaining % dim_num

            d_offset = dim_block * block_D

            seq_start = cu_seqlens[batch_id]
            seq_end = cu_seqlens[batch_id + 1]
            seqlen = seq_end - seq_start

            has_init = False
            if has_initial_state_mode:
                init_val = initial_state_mode[batch_id]
                has_init = init_val != 0

            t_block_start = seq_block * block_M
            t_block_end_candidate = t_block_start + block_M
            t_block_end = T.if_then_else(t_block_end_candidate < seqlen, t_block_end_candidate, seqlen)
            num_tokens = t_block_end - t_block_start

            hist0_ub = T.alloc_ub((block_D,), dtype)
            hist1_ub = T.alloc_ub((block_D,), dtype)
            hist2_ub = T.alloc_ub((block_D,), dtype)
            hist3_ub = T.alloc_ub((block_D,), dtype)
            hist4_ub = T.alloc_ub((block_D,), dtype)

            w0_ub = T.alloc_ub((block_D,), dtype)
            w1_ub = T.alloc_ub((block_D,), dtype)
            w2_ub = T.alloc_ub((block_D,), dtype)
            w3_ub = T.alloc_ub((block_D,), dtype)
            w4_ub = T.alloc_ub((block_D,), dtype)
            w5_ub = T.alloc_ub((block_D,), dtype)
            bias_ub = T.alloc_ub((block_D,), dtype)

            x_cur_ub = T.alloc_ub((block_D,), dtype)
            acc_ub = T.alloc_ub((block_D,), dtype)
            tmp_ub = T.alloc_ub((block_D,), dtype)
            out_ub = T.alloc_ub((block_D,), dtype)

            T.copy(weight[0, d_offset], w0_ub)
            T.copy(weight[1, d_offset], w1_ub)
            T.copy(weight[2, d_offset], w2_ub)
            if width >= 4:
                T.copy(weight[3, d_offset], w3_ub)
            if width >= 5:
                T.copy(weight[4, d_offset], w4_ub)
            if width >= 6:
                T.copy(weight[5, d_offset], w5_ub)

            if has_bias:
                T.copy(bias[d_offset], bias_ub)
            else:
                T.tile.fill(bias_ub, 0.0)

            T.tile.fill(hist0_ub, 0.0)
            T.tile.fill(hist1_ub, 0.0)
            T.tile.fill(hist2_ub, 0.0)
            if width >= 4:
                T.tile.fill(hist3_ub, 0.0)
            if width >= 5:
                T.tile.fill(hist4_ub, 0.0)

            hist_len = width - 1

            if has_init and seq_block == 0:
                if has_cache_indices:
                    ci = cache_indices[batch_id]
                    for h in T.serial(hist_len):
                        if h < state_len:
                            if h == 0:
                                T.copy(conv_state[ci, 0, d_offset], hist0_ub)
                            if h == 1:
                                T.copy(conv_state[ci, 1, d_offset], hist1_ub)
                            if h == 2:
                                T.copy(conv_state[ci, 2, d_offset], hist2_ub)
                            if h == 3:
                                T.copy(conv_state[ci, 3, d_offset], hist3_ub)
                            if h == 4:
                                T.copy(conv_state[ci, 4, d_offset], hist4_ub)
                else:
                    for h in T.serial(hist_len):
                        if h < state_len:
                            if h == 0:
                                T.copy(conv_state[batch_id, 0, d_offset], hist0_ub)
                            if h == 1:
                                T.copy(conv_state[batch_id, 1, d_offset], hist1_ub)
                            if h == 2:
                                T.copy(conv_state[batch_id, 2, d_offset], hist2_ub)
                            if h == 3:
                                T.copy(conv_state[batch_id, 3, d_offset], hist3_ub)
                            if h == 4:
                                T.copy(conv_state[batch_id, 4, d_offset], hist4_ub)
            else:
                for h in T.serial(hist_len):
                    hist_token_idx = t_block_start - hist_len + h
                    if hist_token_idx >= 0:
                        hist_global_idx = seq_start + hist_token_idx
                        if h == 0:
                            T.copy(x[hist_global_idx, d_offset], hist0_ub)
                        if h == 1:
                            T.copy(x[hist_global_idx, d_offset], hist1_ub)
                        if h == 2:
                            T.copy(x[hist_global_idx, d_offset], hist2_ub)
                        if h == 3:
                            T.copy(x[hist_global_idx, d_offset], hist3_ub)
                        if h == 4:
                            T.copy(x[hist_global_idx, d_offset], hist4_ub)

            for t_idx in T.serial(num_tokens):
                t = t_block_start + t_idx

                T.copy(x[seq_start + t, d_offset], x_cur_ub)

                T.copy(bias_ub, acc_ub)

                if width == 3:
                    T.tile.mul(tmp_ub, w0_ub, hist0_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                    T.tile.mul(tmp_ub, w1_ub, hist1_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                    T.tile.mul(tmp_ub, w2_ub, x_cur_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                elif width == 4:
                    T.tile.mul(tmp_ub, w0_ub, hist0_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                    T.tile.mul(tmp_ub, w1_ub, hist1_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                    T.tile.mul(tmp_ub, w2_ub, hist2_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                    T.tile.mul(tmp_ub, w3_ub, x_cur_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                elif width == 5:
                    T.tile.mul(tmp_ub, w0_ub, hist0_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                    T.tile.mul(tmp_ub, w1_ub, hist1_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                    T.tile.mul(tmp_ub, w2_ub, hist2_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                    T.tile.mul(tmp_ub, w3_ub, hist3_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                    T.tile.mul(tmp_ub, w4_ub, x_cur_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                elif width == 6:
                    T.tile.mul(tmp_ub, w0_ub, hist0_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                    T.tile.mul(tmp_ub, w1_ub, hist1_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                    T.tile.mul(tmp_ub, w2_ub, hist2_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                    T.tile.mul(tmp_ub, w3_ub, hist3_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                    T.tile.mul(tmp_ub, w4_ub, hist4_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)
                    T.tile.mul(tmp_ub, w5_ub, x_cur_ub)
                    T.tile.add(acc_ub, acc_ub, tmp_ub)

                if has_activation:
                    zero_ub = T.alloc_ub((block_D,), dtype)
                    denom_ub = T.alloc_ub((block_D,), dtype)
                    T.tile.fill(zero_ub, 0.0)
                    T.tile.sub(denom_ub, zero_ub, acc_ub)
                    T.tile.exp(denom_ub, denom_ub)
                    T.tile.add(denom_ub, denom_ub, 1.0)
                    T.tile.div(out_ub, acc_ub, denom_ub)
                else:
                    T.copy(acc_ub, out_ub)

                T.copy(out_ub, y[seq_start + t, d_offset])

                if width == 3:
                    T.copy(hist1_ub, hist0_ub)
                    T.copy(x_cur_ub, hist1_ub)
                elif width == 4:
                    T.copy(hist1_ub, hist0_ub)
                    T.copy(hist2_ub, hist1_ub)
                    T.copy(x_cur_ub, hist2_ub)
                elif width == 5:
                    T.copy(hist1_ub, hist0_ub)
                    T.copy(hist2_ub, hist1_ub)
                    T.copy(hist3_ub, hist2_ub)
                    T.copy(x_cur_ub, hist3_ub)
                elif width == 6:
                    T.copy(hist1_ub, hist0_ub)
                    T.copy(hist2_ub, hist1_ub)
                    T.copy(hist3_ub, hist2_ub)
                    T.copy(hist4_ub, hist3_ub)
                    T.copy(x_cur_ub, hist4_ub)

            if seq_block == seqlen_num - 1:
                state_tmp_ub = T.alloc_ub((block_D,), dtype)

                if has_cache_indices:
                    ci = cache_indices[batch_id]
                    for pos in T.serial(hist_len):
                        if pos < state_len:
                            last_hist_idx = seqlen - hist_len + pos
                            if last_hist_idx >= 0:
                                T.copy(x[seq_start + last_hist_idx, d_offset], state_tmp_ub)
                                T.copy(state_tmp_ub, conv_state[ci, pos, d_offset])
                else:
                    for pos in T.serial(hist_len):
                        if pos < state_len:
                            last_hist_idx = seqlen - hist_len + pos
                            if last_hist_idx >= 0:
                                T.copy(x[seq_start + last_hist_idx, d_offset], state_tmp_ub)
                                T.copy(state_tmp_ub, conv_state[batch_id, pos, d_offset])

    return main


def get_fn_kernel(batch_size, width, has_bias, has_activation, has_cache_indices, has_initial_state_mode, block_M=64, block_D=512):
    key = (batch_size, width, has_bias, has_activation, has_cache_indices, has_initial_state_mode, block_M, block_D)
    if key not in _kernel_cache_fn:
        _kernel_cache_fn[key] = causal_conv1d_fn_kernel(
            batch_size, width, has_bias, has_activation, has_cache_indices, has_initial_state_mode, block_M, block_D
        )
    return _kernel_cache_fn[key]


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    conv_states: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    initial_state_mode: torch.Tensor | None = None,
    weight_format: str = "kernel",
    conv_state_format: str = "kernel",
):
    """
    Prefill mode: Process full sequences (FN VARLEN)

    性能优化：支持 kernel 格式输入，避免 transpose 开销

    Args:
        x: (total_tokens, dim) for varlen mode (packed layout)
        weight: 卷积权重
            - kernel format (推荐): (width, dim) - 无 transpose，性能最优
            - vLLM format: (dim, width) - 会 transpose，有开销
        conv_states: 卷积状态
            - kernel format (推荐): (num_cache_lines, state_len, dim) - 无 transpose
            - vLLM format: (num_cache_lines, dim, state_len) - 会 transpose
        cache_indices: (batch,) int32
        query_start_loc: (batch + 1,) int32
        initial_state_mode: (batch,) int32
        weight_format: "kernel" 或 "vllm"
        conv_state_format: "kernel" 或 "vllm"

    性能对比:
        - kernel format (无transpose): FN kernel = 0.169 ms
        - vLLM format (每次transpose): FN kernel = 0.864 ms (5.1x慢)

    使用建议:
        # 方式1：初始化时 transpose（推荐）
        weight_kernel = weight_vllm.transpose(0, 1).contiguous()  # 一次性
        conv_state_kernel = conv_state_vllm.transpose(1, 2).contiguous()
        out = causal_conv1d_fn(x, weight_kernel, ..., weight_format="kernel")

        # 方式2：直接传 kernel 格式（最优）
        out = causal_conv1d_fn(x, weight_kernel, conv_state_kernel,
                               weight_format="kernel", conv_state_format="kernel")
    """
    original_dtype = x.dtype

    is_varlen = query_start_loc is not None

    if is_varlen:
        batch_size = cache_indices.size(0)
        total_len = x.size(0)
        dim = x.size(1)
    else:
        batch_size, dim, max_seqlen = x.shape
        total_len = batch_size * max_seqlen
        x = x.reshape(total_len, dim).contiguous()
        query_start_loc = torch.arange(0, total_len + 1, max_seqlen, dtype=torch.int32, device=x.device)

    # 格式转换（只在需要时做）
    if weight_format == "vllm":
        weight_kernel = weight.transpose(0, 1).contiguous()
    else:
        weight_kernel = weight

    if conv_state_format == "vllm":
        conv_state_kernel = conv_states.transpose(1, 2).contiguous()
    else:
        conv_state_kernel = conv_states

    width = weight_kernel.shape[0]

    has_bias = bias is not None
    has_activation = activation in ["silu", "swish"]
    has_cache_indices = cache_indices is not None
    has_initial_state_mode = initial_state_mode is not None

    block_M = 64
    block_D = 512 if dim >= 512 else 256

    kernel = get_fn_kernel(batch_size, width, has_bias, has_activation, has_cache_indices, has_initial_state_mode, block_M, block_D)

    if bias is None:
        bias = torch.zeros(dim, dtype=conv_state_kernel.dtype, device=x.device)
    if cache_indices is None:
        cache_indices = torch.zeros(batch_size, dtype=torch.int32, device=x.device)
    if initial_state_mode is None:
        initial_state_mode = torch.zeros(batch_size, dtype=torch.int32, device=x.device)

    conv_state_kernel_f32 = conv_state_kernel.float()
    out = kernel(x.float(), weight_kernel.float(), bias.float(), conv_state_kernel_f32, query_start_loc, cache_indices, initial_state_mode)

    conv_state_kernel.copy_(conv_state_kernel_f32)

    if conv_state_format == "vllm":
        conv_states.copy_(conv_state_kernel.transpose(1, 2))
    elif conv_states.data_ptr() != conv_state_kernel.data_ptr():
        conv_states.copy_(conv_state_kernel)

    return out.to(original_dtype)


# ============================================================================
# Kernel 2: Decode (统一支持单token和投机解码)
# ============================================================================


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def causal_conv1d_decode_kernel(
    batch: int,
    seqlen: int,
    dim: int,
    state_len: int,
    width: int,
    has_bias: bool,
    has_activation: bool,
    has_cache_indices: bool,
    block_D: int = 512,
):
    """
    Decode kernel: 支持 seqlen=1 (单token) 或 seqlen>1 (投机解码)

    Grid = dim_num (按 dim 分块)

    输入：
    - x: (batch, seqlen, dim)
    - weight: (width, dim)
    - bias: (dim,)
    - conv_state: (num_cache_lines, state_len, dim)
    - cache_indices: (batch,)

    输出：
    - y: (batch, seqlen, dim)
    - conv_state: 状态更新

    状态逻辑：
    - seqlen=1: 从 state[0,1,2] 读取，写入 state[2]
    - seqlen>1: 从 state[offset,offset+1,offset+2] 读取，滚动+写入多个token
    """
    dtype = "float"
    dim_num = T.ceildiv(dim, block_D)
    num_cache_lines = T.symbolic("num_cache_lines")

    @T.prim_func
    def main(
        x: T.Tensor((batch, seqlen, dim), dtype),
        weight: T.Tensor((width, dim), dtype),
        bias: T.Tensor((dim,), dtype),
        conv_state: T.Tensor((num_cache_lines, state_len, dim), dtype),
        cache_indices: T.Tensor((batch,), "int32"),
        y: T.Tensor((batch, seqlen, dim), dtype),
    ):
        with T.Kernel(dim_num, is_npu=True) as (cid, vid):
            d_offset = cid * block_D

            w0 = T.alloc_ub((block_D,), dtype)
            w1 = T.alloc_ub((block_D,), dtype)
            w2 = T.alloc_ub((block_D,), dtype)
            w3 = T.alloc_ub((block_D,), dtype)
            bias_ub = T.alloc_ub((block_D,), dtype)

            T.copy(weight[0, d_offset], w0)
            T.copy(weight[1, d_offset], w1)
            T.copy(weight[2, d_offset], w2)
            if width == 4:
                T.copy(weight[3, d_offset], w3)

            if has_bias:
                T.copy(bias[d_offset], bias_ub)
            else:
                T.tile.fill(bias_ub, 0.0)

            for b_idx in T.serial(batch):
                ci = cache_indices[b_idx]

                hist0_ub = T.alloc_ub((block_D,), dtype)
                hist1_ub = T.alloc_ub((block_D,), dtype)
                hist2_ub = T.alloc_ub((block_D,), dtype)

                state_token_offset = seqlen - 1

                if has_cache_indices:
                    T.copy(conv_state[ci, state_token_offset + 0, d_offset], hist0_ub)
                    T.copy(conv_state[ci, state_token_offset + 1, d_offset], hist1_ub)
                    T.copy(conv_state[ci, state_token_offset + 2, d_offset], hist2_ub)
                else:
                    T.copy(conv_state[b_idx, state_token_offset + 0, d_offset], hist0_ub)
                    T.copy(conv_state[b_idx, state_token_offset + 1, d_offset], hist1_ub)
                    T.copy(conv_state[b_idx, state_token_offset + 2, d_offset], hist2_ub)

                for t_idx in T.serial(seqlen):
                    x_cur = T.alloc_ub((block_D,), dtype)
                    acc = T.alloc_ub((block_D,), dtype)
                    tmp = T.alloc_ub((block_D,), dtype)
                    out = T.alloc_ub((block_D,), dtype)

                    T.copy(x[b_idx, t_idx, d_offset], x_cur)

                    T.copy(bias_ub, acc)
                    T.tile.mul(tmp, hist0_ub, w0)
                    T.tile.add(acc, acc, tmp)
                    T.tile.mul(tmp, hist1_ub, w1)
                    T.tile.add(acc, acc, tmp)
                    T.tile.mul(tmp, hist2_ub, w2)
                    T.tile.add(acc, acc, tmp)
                    if width == 4:
                        T.tile.mul(tmp, x_cur, w3)
                        T.tile.add(acc, acc, tmp)

                    if has_activation:
                        zero_ub = T.alloc_ub((block_D,), dtype)
                        denom_ub = T.alloc_ub((block_D,), dtype)
                        T.tile.fill(zero_ub, 0.0)
                        T.tile.sub(denom_ub, zero_ub, acc)
                        T.tile.exp(denom_ub, denom_ub)
                        T.tile.add(denom_ub, denom_ub, 1.0)
                        T.tile.div(out, acc, denom_ub)
                    else:
                        T.copy(acc, out)

                    T.copy(out, y[b_idx, t_idx, d_offset])

                    T.copy(hist1_ub, hist0_ub)
                    T.copy(hist2_ub, hist1_ub)
                    T.copy(x_cur, hist2_ub)

                tmp_state1 = T.alloc_ub((block_D,), dtype)
                tmp_state2 = T.alloc_ub((block_D,), dtype)

                if has_cache_indices:
                    T.copy(conv_state[ci, state_token_offset + 1, d_offset], tmp_state1)
                    T.copy(conv_state[ci, state_token_offset + 2, d_offset], tmp_state2)
                    T.copy(tmp_state1, conv_state[ci, 0, d_offset])
                    T.copy(tmp_state2, conv_state[ci, 1, d_offset])

                    for write_t in T.serial(seqlen):
                        write_x = T.alloc_ub((block_D,), dtype)
                        T.copy(x[b_idx, write_t, d_offset], write_x)
                        write_pos = 2 + write_t
                        if write_pos < state_len:
                            T.copy(write_x, conv_state[ci, write_pos, d_offset])
                else:
                    T.copy(conv_state[b_idx, state_token_offset + 1, d_offset], tmp_state1)
                    T.copy(conv_state[b_idx, state_token_offset + 2, d_offset], tmp_state2)
                    T.copy(tmp_state1, conv_state[b_idx, 0, d_offset])
                    T.copy(tmp_state2, conv_state[b_idx, 1, d_offset])

                    for write_t in T.serial(seqlen):
                        write_x = T.alloc_ub((block_D,), dtype)
                        T.copy(x[b_idx, write_t, d_offset], write_x)
                        write_pos = 2 + write_t
                        if write_pos < state_len:
                            T.copy(write_x, conv_state[b_idx, write_pos, d_offset])

    return main


_kernel_cache_decode = {}


def get_decode_kernel(batch, seqlen, dim, state_len, width, has_bias, has_activation, has_cache_indices, block_D=512):
    key = (batch, seqlen, dim, state_len, width, has_bias, has_activation, has_cache_indices, block_D)
    if key not in _kernel_cache_decode:
        _kernel_cache_decode[key] = causal_conv1d_decode_kernel(
            batch, seqlen, dim, state_len, width, has_bias, has_activation, has_cache_indices, block_D
        )
    return _kernel_cache_decode[key]


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    cache_indices: torch.Tensor | None = None,
    weight_format: str = "kernel",
    conv_state_format: str = "kernel",
):
    """
    Decode mode: 统一支持单token和投机解码

    Args:
        x: 输入 token
            - 单token: (1, dim) 或 (dim,)
            - 投机解码: (batch, seqlen, dim)，seqlen=numAcceptedTokens
        conv_state: 卷积状态
            - kernel format (推荐): (num_cache_lines, state_len, dim)
            - vLLM format: (num_cache_lines, dim, state_len)
        weight: 卷积权重
            - kernel format (推荐): (width, dim)
            - vLLM format: (dim, width)
        cache_indices: (batch,) int32

    状态更新逻辑：
    - seqlen=1: 单token decode，从 state[0,1,2] 读取
    - seqlen>1: 投机解码，从 state[offset,offset+1,offset+2] 读取
    """
    original_dtype = x.dtype

    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(0)  # (dim,) -> (1, 1, dim)
    elif x.dim() == 2:
        x = x.unsqueeze(1)  # (batch, dim) -> (batch, 1, dim)

    batch, seqlen, dim = x.shape

    state_len = conv_state.shape[1] if conv_state_format == "kernel" else conv_state.shape[2]

    if weight_format == "vllm":
        weight_kernel = weight.transpose(0, 1).contiguous()
    else:
        weight_kernel = weight

    if conv_state_format == "vllm":
        conv_state_kernel = conv_state.transpose(1, 2).contiguous()
    else:
        conv_state_kernel = conv_state

    width = weight_kernel.shape[0]

    has_bias = bias is not None
    has_activation = activation in ["silu", "swish"]
    has_cache_indices = cache_indices is not None

    block_D = 512 if dim >= 512 else 256

    kernel = get_decode_kernel(batch, seqlen, dim, state_len, width, has_bias, has_activation, has_cache_indices, block_D)

    if bias is None:
        bias = torch.zeros(dim, dtype=conv_state_kernel.dtype, device=x.device)
    if cache_indices is None:
        cache_indices = torch.zeros(batch, dtype=torch.int32, device=x.device)

    conv_state_kernel_f32 = conv_state_kernel.float()
    out = kernel(x.float(), weight_kernel.float(), bias.float(), conv_state_kernel_f32, cache_indices)

    conv_state_kernel.copy_(conv_state_kernel_f32)

    if conv_state_format == "vllm":
        conv_state.copy_(conv_state_kernel.transpose(1, 2).contiguous())
    elif conv_state.data_ptr() != conv_state_kernel.data_ptr():
        conv_state.copy_(conv_state_kernel)

    if seqlen == 1:
        return out.squeeze(1).to(original_dtype)  # (batch, 1, dim) -> (batch, dim)
    else:
        return out.to(original_dtype)


# ============================================================================
# Reference Implementation (PyTorch)
# ============================================================================


def causal_conv1d_fn_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    conv_states: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    initial_state_mode: torch.Tensor | None = None,
):
    """PyTorch reference implementation for prefill mode."""
    dtype = x.dtype
    x = x.float()
    weight = weight.float()
    if bias is not None:
        bias = bias.float()

    batch_size = cache_indices.size(0)
    total_len, dim = x.shape
    width = weight.shape[0]

    y = torch.zeros_like(x)

    for b in range(batch_size):
        seq_start = query_start_loc[b].item()
        seq_end = query_start_loc[b + 1].item()
        seqlen = seq_end - seq_start

        ci = cache_indices[b].item() if cache_indices is not None else b
        has_init = initial_state_mode[b].item() if initial_state_mode is not None else 0

        hist_len = width - 1
        history = []

        if has_init and conv_states is not None:
            for h in range(hist_len):
                if h < conv_states.shape[1]:
                    history.append(conv_states[ci, h, :].float().clone())
                else:
                    history.append(torch.zeros(dim, dtype=torch.float32, device=x.device))
        else:
            for _h in range(hist_len):
                history.append(torch.zeros(dim, dtype=torch.float32, device=x.device))

        for t in range(seqlen):
            x_t = x[seq_start + t, :]

            acc = bias.clone() if bias is not None else torch.zeros(dim, dtype=torch.float32, device=x.device)
            for w_idx in range(width - 1):
                acc = acc + weight[w_idx, :] * history[w_idx]
            acc = acc + weight[width - 1, :] * x_t

            if activation in ["silu", "swish"]:
                out = acc / (1.0 + torch.exp(-acc))
            else:
                out = acc

            y[seq_start + t, :] = out

            for h in range(hist_len - 1):
                history[h] = history[h + 1]
            history[hist_len - 1] = x_t.clone()

        if conv_states is not None and seqlen > 0:
            for pos in range(hist_len):
                last_idx = seqlen - hist_len + pos
                if last_idx >= 0:
                    conv_states[ci, pos, :] = x[seq_start + last_idx, :].to(conv_states.dtype)

    return y.to(dtype)


def causal_conv1d_update_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    cache_indices: torch.Tensor | None = None,
):
    """PyTorch reference implementation for decode mode."""
    dtype = x.dtype

    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 2:
        x = x.unsqueeze(1)

    batch, seqlen, dim = x.shape
    width = weight.shape[0]
    hist_len = width - 1
    state_len = conv_state.shape[1]

    x = x.float()
    weight = weight.float()
    if bias is not None:
        bias = bias.float()
    conv_state = conv_state.float()

    y = torch.zeros(batch, seqlen, dim, dtype=torch.float32, device=x.device)

    for b in range(batch):
        ci = cache_indices[b].item() if cache_indices is not None else b
        state_token_offset = seqlen - 1

        history = []
        for h in range(hist_len):
            src_idx = state_token_offset + h
            if src_idx < state_len:
                history.append(conv_state[ci, src_idx, :].clone())
            else:
                history.append(torch.zeros(dim, dtype=torch.float32, device=x.device))

        for t in range(seqlen):
            x_t = x[b, t, :]

            acc = bias.clone() if bias is not None else torch.zeros(dim, dtype=torch.float32, device=x.device)
            for w_idx in range(width - 1):
                acc = acc + weight[w_idx, :] * history[w_idx]
            acc = acc + weight[width - 1, :] * x_t

            if activation in ["silu", "swish"]:
                out = acc / (1.0 + torch.exp(-acc))
            else:
                out = acc

            y[b, t, :] = out

            for h in range(hist_len - 1):
                history[h] = history[h + 1]
            history[hist_len - 1] = x_t.clone()

        if state_len >= 2:
            conv_state[ci, 0, :] = conv_state[ci, state_token_offset + 1, :]
            conv_state[ci, 1, :] = conv_state[ci, state_token_offset + 2, :]

        for t in range(seqlen):
            write_pos = 2 + t
            if write_pos < state_len:
                conv_state[ci, write_pos, :] = x[b, t, :]

    if seqlen == 1:
        return y.squeeze(1).to(dtype)
    return y.to(dtype)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    torch.set_default_device("npu")
    torch.manual_seed(42)

    seqlen = 2048
    dim = 2048
    width = 4
    num_cache_lines = 804
    state_len = 3
    batch = 1

    x_fn = torch.randn(seqlen, dim, dtype=torch.float16, device="npu")
    weight_fn = torch.randn(width, dim, dtype=torch.float16, device="npu")
    conv_state_fn = torch.randn(num_cache_lines, state_len, dim, dtype=torch.float16, device="npu")
    cache_indices_fn = torch.tensor([0], dtype=torch.int32, device="npu")
    query_start_loc_fn = torch.tensor([0, seqlen], dtype=torch.int32, device="npu")
    initial_state_mode_fn = torch.tensor([0], dtype=torch.int32, device="npu")

    out_ref_fn = causal_conv1d_fn_ref(
        x_fn,
        weight_fn,
        None,
        activation="silu",
        conv_states=conv_state_fn,
        cache_indices=cache_indices_fn,
        query_start_loc=query_start_loc_fn,
        initial_state_mode=initial_state_mode_fn,
    )

    out_kernel_fn = causal_conv1d_fn(
        x_fn,
        weight_fn,
        None,
        activation="silu",
        conv_states=conv_state_fn,
        cache_indices=cache_indices_fn,
        query_start_loc=query_start_loc_fn,
        initial_state_mode=initial_state_mode_fn,
        weight_format="kernel",
        conv_state_format="kernel",
    )

    torch.testing.assert_close(out_kernel_fn, out_ref_fn, rtol=1e-2, atol=1e-2)
    print("Batch Kernel Output Match!")
