#!/usr/bin/env python
import tilelang
import tilelang.language as T
import torch


# ---- Symbolic dimensions shared across kernel instantiations ----
symbol_cache_lines = T.symbolic("num_cache_lines")
symbol_state_len = T.symbolic("state_len")

BATCH_TOKENS = 4
STAGES = 2
CORE_NUM = 24

pass_configs_config = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_kernel_cache = {}


# ======================== Kernel ========================


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs_config)
def _build_kernel(
    width: int,
    dim_num: int,
    num_batches: int,
    base_dim: int,
    dtype_str: str = "float16",
):
    hist_len = width - 1
    symbol_dim = T.symbolic("dim")
    symbol_total_len = T.symbolic("total_len")

    @T.prim_func
    def kernel_func(
        x: T.Tensor((symbol_total_len, symbol_dim), dtype_str),
        weight: T.Tensor((width, symbol_dim), dtype_str),
        conv_state: T.Tensor((symbol_cache_lines, symbol_state_len, symbol_dim), dtype_str),
        cu_seqlens: T.Tensor((num_batches + 1,), "int32"),
        cache_indices: T.Tensor((num_batches,), "int32"),
        initial_state_mode: T.Tensor((num_batches,), "int32"),
        y: T.Tensor((symbol_total_len, symbol_dim), dtype_str),
    ):
        with T.Kernel(num_batches * dim_num, is_npu=True) as (cid, vid):
            batch_id = cid // dim_num
            block_idx = cid % dim_num
            seq_start = cu_seqlens[batch_id]
            seq_end = cu_seqlens[batch_id + 1]
            seqlen = seq_end - seq_start
            block_size = (seqlen + dim_num - 1) // dim_num
            block_offset = block_idx * block_size
            block_end = T.Select(block_offset + block_size > seqlen, seqlen, block_offset + block_size)
            num_tokens = T.Select(block_end > block_offset, block_end - block_offset, 0)
            global_start = seq_start + block_offset
            is_first_buf = T.alloc_ub((1,), "int32")
            T.tile.fill(is_first_buf, (block_idx == 0))
            is_last_buf = T.alloc_ub((1,), "int32")
            T.tile.fill(is_last_buf, (block_end >= seqlen))
            cache_line = T.Select(batch_id == 0, cache_indices[0], cache_indices[1])
            has_initial = T.Select(batch_id == 0, initial_state_mode[0], initial_state_mode[1])
            hist_base = global_start - hist_len * (block_idx != 0)
            d_offset = 0

            # UB buffers
            x_buf = T.alloc_ub((STAGES, BATCH_TOKENS, base_dim), dtype_str)
            y_buf = T.alloc_ub((STAGES, BATCH_TOKENS, base_dim), dtype_str)
            state0 = T.alloc_ub((base_dim,), dtype_str)
            state1 = T.alloc_ub((base_dim,), dtype_str)
            state2 = T.alloc_ub((base_dim,), dtype_str)
            hist0 = T.alloc_ub((base_dim,), dtype_str)
            hist1 = T.alloc_ub((base_dim,), dtype_str)
            hist2 = T.alloc_ub((base_dim,), dtype_str)
            w0 = T.alloc_ub((base_dim,), dtype_str)
            w1 = T.alloc_ub((base_dim,), dtype_str)
            w2 = T.alloc_ub((base_dim,), dtype_str)
            w3 = T.alloc_ub((base_dim,), dtype_str)
            tmp = T.alloc_ub((base_dim,), dtype_str)
            save0 = T.alloc_ub((base_dim,), dtype_str)
            save1 = T.alloc_ub((base_dim,), dtype_str)
            save2 = T.alloc_ub((base_dim,), dtype_str)

            # weight preload
            T.copy(weight[0, d_offset], w0)
            T.copy(weight[1, d_offset], w1)
            T.copy(weight[2, d_offset], w2)
            T.copy(weight[3, d_offset], w3)
            T.barrier_all()
            T.tile.fill(hist0, 0.0)
            T.tile.fill(hist1, 0.0)
            T.tile.fill(hist2, 0.0)

            # history loading
            if is_first_buf[0] != 0 and has_initial != 0:
                if hist_len >= 1 and symbol_state_len > 0:
                    T.copy(conv_state[cache_line, 0, d_offset], hist0)
                if hist_len >= 2 and symbol_state_len > 1:
                    T.copy(conv_state[cache_line, 1, d_offset], hist1)
                if hist_len >= 3 and symbol_state_len > 2:
                    T.copy(conv_state[cache_line, 2, d_offset], hist2)
            if is_first_buf[0] == 0:
                if hist_len >= 1:
                    T.copy(x[hist_base, d_offset], hist0)
                if hist_len >= 2:
                    T.copy(x[hist_base + 1, d_offset], hist1)
                if hist_len >= 3:
                    T.copy(x[hist_base + 2, d_offset], hist2)
            T.barrier_all()

            # initial state compute
            T.tile.mul(state2, w0, hist2)
            T.tile.mul(state1, w0, hist1)
            T.tile.mul(tmp, w1, hist2)
            T.tile.add(state1, state1, tmp)
            T.tile.mul(state0, w0, hist0)
            T.tile.mul(tmp, w1, hist1)
            T.tile.add(state0, state0, tmp)
            T.tile.mul(tmp, w2, hist2)
            T.tile.add(state0, state0, tmp)

            # pipeline loop
            num_iterations = (num_tokens + 3) // 4
            T.set_flag("mte3", "mte2", 0)
            T.set_flag("mte3", "mte2", 1)
            T.wait_flag("mte3", "mte2", 0)
            if num_tokens > 0:
                T.copy(x[global_start, d_offset], x_buf[0, 0, :])
                T.copy(x[global_start + 1, d_offset], x_buf[0, 1, :])
                T.copy(x[global_start + 2, d_offset], x_buf[0, 2, :])
                T.copy(x[global_start + 3, d_offset], x_buf[0, 3, :])
                T.set_flag("mte2", "v", 0)
            for i in T.serial(num_iterations):
                cur = i % 2
                nxt = (i + 1) % 2
                out_base = global_start + i * 4
                if i < num_iterations - 1:
                    T.wait_flag("mte3", "mte2", nxt)
                    next_base = global_start + (i + 1) * 4
                    T.copy(x[next_base, d_offset], x_buf[nxt, 0, :])
                    T.copy(x[next_base + 1, d_offset], x_buf[nxt, 1, :])
                    T.copy(x[next_base + 2, d_offset], x_buf[nxt, 2, :])
                    T.copy(x[next_base + 3, d_offset], x_buf[nxt, 3, :])
                    T.set_flag("mte2", "v", nxt)
                T.wait_flag("mte2", "v", cur)

                T.tile.mul_add_dst(state0, x_buf[cur, 0, :], w3)
                T.tile.silu(y_buf[cur, 0, :], state0)
                T.tile.mul(tmp, w2, x_buf[cur, 0, :])
                T.tile.add(state0, tmp, state1)
                T.tile.mul(tmp, w1, x_buf[cur, 0, :])
                T.tile.add(state1, tmp, state2)
                T.tile.mul(state2, w0, x_buf[cur, 0, :])
                T.tile.mul_add_dst(state0, x_buf[cur, 1, :], w3)
                T.tile.silu(y_buf[cur, 1, :], state0)
                T.tile.mul(tmp, w2, x_buf[cur, 1, :])
                T.tile.add(state0, tmp, state1)
                T.tile.mul(tmp, w1, x_buf[cur, 1, :])
                T.tile.add(state1, tmp, state2)
                T.tile.mul(state2, w0, x_buf[cur, 1, :])
                T.tile.mul_add_dst(state0, x_buf[cur, 2, :], w3)
                T.tile.silu(y_buf[cur, 2, :], state0)
                T.tile.mul(tmp, w2, x_buf[cur, 2, :])
                T.tile.add(state0, tmp, state1)
                T.tile.mul(tmp, w1, x_buf[cur, 2, :])
                T.tile.add(state1, tmp, state2)
                T.tile.mul(state2, w0, x_buf[cur, 2, :])
                T.tile.mul_add_dst(state0, x_buf[cur, 3, :], w3)
                T.tile.silu(y_buf[cur, 3, :], state0)
                T.tile.mul(tmp, w2, x_buf[cur, 3, :])
                T.tile.add(state0, tmp, state1)
                T.tile.mul(tmp, w1, x_buf[cur, 3, :])
                T.tile.add(state1, tmp, state2)
                T.tile.mul(state2, w0, x_buf[cur, 3, :])

                T.set_flag("v", "mte3", cur)
                T.wait_flag("v", "mte3", cur)
                remain = num_tokens - i * 4
                if remain >= 1:
                    T.copy(y_buf[cur, 0, :], y[out_base, d_offset])
                if remain >= 2:
                    T.copy(y_buf[cur, 1, :], y[out_base + 1, d_offset])
                if remain >= 3:
                    T.copy(y_buf[cur, 2, :], y[out_base + 2, d_offset])
                if remain >= 4:
                    T.copy(y_buf[cur, 3, :], y[out_base + 3, d_offset])
                T.set_flag("mte3", "mte2", cur)
            T.wait_flag("mte3", "mte2", 0)
            T.wait_flag("mte3", "mte2", 1)

            # conv_state writeback (only for last block of each batch)
            if is_last_buf[0] != 0 and seqlen > 0:
                T.tile.fill(save0, 0.0)
                T.tile.fill(save1, 0.0)
                T.tile.fill(save2, 0.0)
                if hist_len >= 1:
                    T.copy(x[seq_end - 1, d_offset], save2)
                if hist_len >= 2:
                    T.copy(x[seq_end - 2, d_offset], save1)
                if hist_len >= 3:
                    T.copy(x[seq_end - 3, d_offset], save0)
                T.barrier_all()
                if hist_len >= 1 and symbol_state_len > 0:
                    T.copy(save0, conv_state[cache_line, 0, d_offset])
                if hist_len >= 2 and symbol_state_len > 1:
                    T.copy(save1, conv_state[cache_line, 1, d_offset])
                if hist_len >= 3 and symbol_state_len > 2:
                    T.copy(save2, conv_state[cache_line, 2, d_offset])

    return kernel_func


# ======================= Wrapper =======================


def get_causal_conv1d_fn_pipeline_v15(weight_width, num_batches, dim, dtype_str="float16"):
    """Return a cached compiled kernel for the given parameters."""
    cache_key = (weight_width, num_batches, dim, dtype_str)
    if cache_key not in _kernel_cache:
        _kernel_cache[cache_key] = _build_kernel(weight_width, CORE_NUM, num_batches, dim, dtype_str)
    return _kernel_cache[cache_key]


def causal_conv1d_fn_pipeline_v15(
    x,
    weight,
    conv_states,
    cache_indices,
    cu_seqlens,
    initial_state_mode,
    dtype=torch.float16,
):
    """FN/Prefill: CUTBS token-block tiling, 24/24 cores."""
    kernel_dtype_str = "float16" if dtype == torch.bfloat16 else "float16"
    if dtype == torch.bfloat16:
        x = x.to(torch.float16)
        weight = weight.to(torch.float16)
        cs_copy = conv_states.to(torch.float16)
    else:
        cs_copy = conv_states
    kernel = get_causal_conv1d_fn_pipeline_v15(weight.shape[0], cache_indices.size(0), x.size(1), kernel_dtype_str)
    output = kernel(x, weight, cs_copy.clone(), cu_seqlens, cache_indices, initial_state_mode)
    if dtype == torch.bfloat16:
        output = output.to(torch.bfloat16)
        conv_states.copy_(cs_copy.to(torch.bfloat16))
    else:
        conv_states.copy_(cs_copy)
    return output


# ======================= Reference =======================


def causal_conv1d_fn_ref(
    x,
    weight,
    conv_states,
    cache_indices,
    cu_seqlens,
    initial_state_mode,
    activation="silu",
):
    """CPU golden reference."""
    dtype = x.dtype
    x = x.float()
    weight = weight.float()
    conv_states_f = conv_states.float().clone()
    num_batches = cache_indices.size(0)
    hist_len = weight.shape[0] - 1
    y = torch.zeros_like(x)
    for b in range(num_batches):
        seq_start = cu_seqlens[b].item()
        seq_end = cu_seqlens[b + 1].item()
        seqlen = seq_end - seq_start
        if seqlen == 0:
            continue
        cache_line = cache_indices[b].item()
        has_initial = initial_state_mode[b].item()
        history = torch.zeros(hist_len, x.size(1), dtype=torch.float32, device=x.device)
        if has_initial:
            for h in range(hist_len):
                if h < conv_states_f.shape[1]:
                    history[h] = conv_states_f[cache_line, h].clone()
        for t in range(seqlen):
            acc = torch.zeros(x.size(1), dtype=torch.float32, device=x.device)
            for w in range(hist_len):
                acc += weight[w] * history[w]
            acc += weight[hist_len] * x[seq_start + t]
            if activation in ("silu", "swish"):
                acc = acc / (1 + torch.exp(-acc))
            y[seq_start + t] = acc
            if hist_len > 1:
                history[:-1] = history[1:].clone()
            history[-1] = x[seq_start + t].clone()
        for p in range(hist_len):
            if p < conv_states_f.shape[1]:
                idx = seqlen - hist_len + p
                if idx >= 0:
                    conv_states_f[cache_line, p] = x[seq_start + idx]
    conv_states.copy_(conv_states_f)
    return y.to(dtype)


# ======================== Test ========================


def _run_test(dtype_str, dtype):

    tilelang.cache.clear_cache()
    _kernel_cache.clear()
    torch.manual_seed(42)

    dim = 2048
    width = 4
    num_batches = 2
    total_tokens = 2048
    num_cache_lines = 804
    state_len = 3
    query_start_loc = [0, 662, total_tokens]
    cache_indices_list = [0, 1]
    initial_state_mode_list = [1, 1]

    grid_size = num_batches * CORE_NUM
    print(f"baseDim={dim}, num_batches={num_batches}, dim_num={CORE_NUM}, Grid={grid_size}")
    print(f"varlen: batch0=662, batch1={total_tokens - 662}")

    cu_seqlens_t = torch.tensor(query_start_loc, dtype=torch.int32)
    cache_indices_t = torch.tensor(cache_indices_list, dtype=torch.int32)
    initial_state_mode_t = torch.tensor(initial_state_mode_list, dtype=torch.int32)

    X = torch.randn(total_tokens, dim, dtype=dtype, device="npu")
    W = torch.randn(width, dim, dtype=dtype, device="npu")
    CS = torch.randn(num_cache_lines, state_len, dim, dtype=dtype, device="npu")
    CS[0] = torch.randn(state_len, dim, dtype=dtype, device="npu")
    CS[1] = torch.randn(state_len, dim, dtype=dtype, device="npu")
    CI = cache_indices_t.npu()
    QL = cu_seqlens_t.npu()
    IM = initial_state_mode_t.npu()

    cs_ref = CS.cpu().clone()
    golden_output = causal_conv1d_fn_ref(X.cpu(), W.cpu(), cs_ref, CI.cpu(), QL.cpu(), IM.cpu())

    OT = causal_conv1d_fn_pipeline_v15(X, W, CS.clone(), CI, QL, IM, dtype=dtype)
    torch.testing.assert_close(OT.cpu(), golden_output, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    print("=" * 60)
    print("Causal Conv1D — Prefill, token-block tiling, fp16/bf16")
    print("=" * 60)
    _run_test("float16", torch.float16)
    _run_test("bfloat16", torch.bfloat16)
    print("Batch Kernel Output Match!")
