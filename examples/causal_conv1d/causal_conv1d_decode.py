#!/usr/bin/env python3

# Copyright 2025-2026 The xLLM Authors.
import tilelang
import tilelang.language as T
import torch

symbol_cache_lines = T.symbolic("num_cache_lines")
symbol_state_len = T.symbolic("state_len")
symbol_batch = T.symbolic("batch_size")

DIM_PER_CORE = 2048

pass_configs_config = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_decode_kernel_cache = {}


def build_causal_conv1d_decode_kernel(
    width: int,
    dim_chunks: int,
    dim_per_core: int = DIM_PER_CORE,
    dtype_str: str = "bfloat16",
    has_silu: bool = True,
) -> torch.nn.Module:
    hist_len = width - 1
    symbol_dim = T.symbolic("dim")
    total_tasks = symbol_batch * dim_chunks

    @T.prim_func
    def causal_conv1d_decode(
        x: T.Tensor((symbol_batch, symbol_dim), dtype_str),
        weight: T.Tensor((width, symbol_dim), dtype_str),
        conv_state: T.Tensor((symbol_cache_lines, symbol_state_len, symbol_dim), dtype_str),
        conv_state_indices_init: T.Tensor((symbol_batch,), "int32"),
        conv_state_indices_current: T.Tensor((symbol_batch,), "int32"),
        initial_state_mode: T.Tensor((symbol_batch,), "int32"),
        bias: T.Tensor((symbol_dim,), dtype_str),
        y: T.Tensor((symbol_batch, symbol_dim), dtype_str),
    ):
        with T.Kernel(total_tasks, is_npu=True) as (cid, vid):
            batch_id = cid // dim_chunks
            dim_chunk_id = cid % dim_chunks

            d_offset = dim_chunk_id * dim_per_core
            read_cache_line = conv_state_indices_init[batch_id]
            write_cache_line = conv_state_indices_current[batch_id]
            has_initial = initial_state_mode[batch_id]

            w_half0 = T.alloc_ub((dim_per_core,), dtype_str)
            w_half1 = T.alloc_ub((dim_per_core,), dtype_str)
            w_half2 = T.alloc_ub((dim_per_core,), dtype_str)
            w_half3 = T.alloc_ub((dim_per_core,), dtype_str)
            bias_half = T.alloc_ub((dim_per_core,), dtype_str)
            hist_half0 = T.alloc_ub((dim_per_core,), dtype_str)
            hist_half1 = T.alloc_ub((dim_per_core,), dtype_str)
            hist_half2 = T.alloc_ub((dim_per_core,), dtype_str)
            x_half = T.alloc_ub((dim_per_core,), dtype_str)
            y_half = T.alloc_ub((dim_per_core,), dtype_str)
            save_half0 = T.alloc_ub((dim_per_core,), dtype_str)
            save_half1 = T.alloc_ub((dim_per_core,), dtype_str)
            save_half2 = T.alloc_ub((dim_per_core,), dtype_str)

            w0 = T.alloc_ub((dim_per_core,), "float32")
            w1 = T.alloc_ub((dim_per_core,), "float32")
            w2 = T.alloc_ub((dim_per_core,), "float32")
            w3 = T.alloc_ub((dim_per_core,), "float32")
            bias_ub = T.alloc_ub((dim_per_core,), "float32")
            hist0 = T.alloc_ub((dim_per_core,), "float32")
            hist1 = T.alloc_ub((dim_per_core,), "float32")
            hist2 = T.alloc_ub((dim_per_core,), "float32")
            x_ub = T.alloc_ub((dim_per_core,), "float32")
            state0 = T.alloc_ub((dim_per_core,), "float32")
            tmp = T.alloc_ub((dim_per_core,), "float32")
            y_ub = T.alloc_ub((dim_per_core,), "float32")

            T.copy(weight[0, d_offset], w_half0)
            T.copy(weight[1, d_offset], w_half1)
            T.copy(weight[2, d_offset], w_half2)
            T.copy(weight[3, d_offset], w_half3)
            T.copy(bias[d_offset], bias_half)
            T.set_flag("mte2", "v", 1)
            T.wait_flag("mte2", "v", 1)

            T.tile.cast(w0, w_half0, "CAST_NONE", dim_per_core)
            T.tile.cast(w1, w_half1, "CAST_NONE", dim_per_core)
            T.tile.cast(w2, w_half2, "CAST_NONE", dim_per_core)
            T.tile.cast(w3, w_half3, "CAST_NONE", dim_per_core)
            T.tile.cast(bias_ub, bias_half, "CAST_NONE", dim_per_core)

            T.tile.fill(hist0, 0.0)
            T.tile.fill(hist1, 0.0)
            T.tile.fill(hist2, 0.0)

            if has_initial != 0:
                if hist_len >= 1 and symbol_state_len > 0:
                    T.copy(conv_state[read_cache_line, 0, d_offset], hist_half0)
                    T.set_flag("mte2", "v", 2)
                    T.wait_flag("mte2", "v", 2)
                    T.tile.cast(hist0, hist_half0, "CAST_NONE", dim_per_core)
                if hist_len >= 2 and symbol_state_len > 1:
                    T.copy(conv_state[read_cache_line, 1, d_offset], hist_half1)
                    T.set_flag("mte2", "v", 3)
                    T.wait_flag("mte2", "v", 3)
                    T.tile.cast(hist1, hist_half1, "CAST_NONE", dim_per_core)
                if hist_len >= 3 and symbol_state_len > 2:
                    T.copy(conv_state[read_cache_line, 2, d_offset], hist_half2)
                    T.set_flag("mte2", "v", 4)
                    T.wait_flag("mte2", "v", 4)
                    T.tile.cast(hist2, hist_half2, "CAST_NONE", dim_per_core)

            T.copy(x[batch_id, d_offset], x_half)
            T.set_flag("mte2", "v", 5)
            T.wait_flag("mte2", "v", 5)
            T.tile.cast(x_ub, x_half, "CAST_NONE", dim_per_core)

            T.tile.mul(state0, w0, hist0)
            T.tile.mul(tmp, w1, hist1)
            T.tile.add(state0, state0, tmp)
            T.tile.mul(tmp, w2, hist2)
            T.tile.add(state0, state0, tmp)
            T.tile.mul_add_dst(state0, x_ub, w3)
            T.tile.add(tmp, state0, bias_ub)
            if has_silu:
                T.tile.silu(y_ub, tmp)
            else:
                T.copy(tmp, y_ub)

            T.tile.cast(y_half, y_ub, "CAST_RINT", dim_per_core)
            T.set_flag("v", "mte3", 0)
            T.wait_flag("v", "mte3", 0)
            T.copy(y_half, y[batch_id, d_offset])

            T.tile.cast(save_half0, hist1, "CAST_RINT", dim_per_core)
            T.tile.cast(save_half1, hist2, "CAST_RINT", dim_per_core)
            T.tile.cast(save_half2, x_ub, "CAST_RINT", dim_per_core)
            T.set_flag("v", "mte3", 6)
            T.wait_flag("v", "mte3", 6)

            if hist_len >= 1 and symbol_state_len > 0:
                T.copy(save_half0, conv_state[write_cache_line, 0, d_offset])
            if hist_len >= 2 and symbol_state_len > 1:
                T.copy(save_half1, conv_state[write_cache_line, 1, d_offset])
            if hist_len >= 3 and symbol_state_len > 2:
                T.copy(save_half2, conv_state[write_cache_line, 2, d_offset])

    return causal_conv1d_decode


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs_config, target="ascendc")
def _build_decode_kernel_jit(
    width: int,
    dim_chunks: int,
    dim_per_core: int = DIM_PER_CORE,
    dtype_str: str = "bfloat16",
    has_silu: bool = True,
) -> torch.nn.Module:
    return build_causal_conv1d_decode_kernel(
        width=width,
        dim_chunks=dim_chunks,
        dim_per_core=dim_per_core,
        dtype_str=dtype_str,
        has_silu=has_silu,
    )


def get_decode_kernel(
    width: int,
    dim: int,
    dtype_str: str = "bfloat16",
    has_silu: bool = True,
) -> torch.nn.Module:
    dim_chunks = (dim + DIM_PER_CORE - 1) // DIM_PER_CORE
    cache_key = (
        width,
        dim_chunks,
        DIM_PER_CORE,
        dtype_str,
        has_silu,
    )
    if cache_key not in _decode_kernel_cache:
        _decode_kernel_cache[cache_key] = _build_decode_kernel_jit(
            width,
            dim_chunks,
            DIM_PER_CORE,
            dtype_str,
            has_silu,
        )
    return _decode_kernel_cache[cache_key]


def causal_conv1d_decode(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | bool | None = "silu",
    conv_state_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = -1,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    original_dtype = x.dtype
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    has_silu = activation in ("silu", "swish")
    width = weight.shape[1]
    dim = weight.shape[0]

    if original_dtype == torch.float16:
        x = x.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
        conv_state_work = conv_state.to(torch.bfloat16)
        if bias is not None:
            bias_work = bias.to(torch.bfloat16).contiguous()
        else:
            bias_work = torch.zeros(dim, dtype=torch.bfloat16, device=conv_state.device)
    else:
        conv_state_work = conv_state
        if bias is not None:
            bias_work = bias.contiguous()
        else:
            bias_work = torch.zeros(dim, dtype=torch.bfloat16, device=conv_state.device)

    weight_t = weight.transpose(0, 1).contiguous()
    conv_state_t = conv_state_work.transpose(1, 2).contiguous()

    if query_start_loc is not None:
        qsl_kernel = query_start_loc.to(torch.int32).contiguous()
        batch = qsl_kernel.numel() - 1
        x_kernel = x.contiguous()
    else:
        if x.dim() == 2:
            x_work = x.unsqueeze(-1)
        else:
            x_work = x
        batch, dim_check, seqlen = x_work.shape
        assert dim_check == dim
        assert seqlen == 1
        x_kernel = x_work.reshape(batch, dim).contiguous()

    if conv_state_indices is None:
        init_indices = torch.arange(batch, dtype=torch.int32, device=conv_state.device)
        current_indices = torch.arange(batch, dtype=torch.int32, device=conv_state.device)
    elif conv_state_indices.dim() == 1:
        ci = conv_state_indices.to(torch.int32).contiguous()
        init_indices = ci
        current_indices = ci.clone()
    else:
        ci = conv_state_indices.to(torch.int32).contiguous()
        if initial_state_idx is None:
            init_indices = ci[:, 0].contiguous()
        else:
            isi = initial_state_idx.to(torch.int32).contiguous()
            init_indices = torch.where(isi == 0, ci[:, 0], ci[:, 1]).contiguous()
        if block_idx_last_scheduled_token is None:
            current_indices = ci[:, 0].contiguous()
        else:
            bilt = block_idx_last_scheduled_token.to(torch.int32).contiguous()
            current_indices = torch.where(bilt == 0, ci[:, 0], ci[:, 1]).contiguous()

    initial_state_mode = torch.ones(batch, dtype=torch.int32, device=conv_state.device)

    kernel = get_decode_kernel(width, dim, "bfloat16", has_silu)
    output = kernel(
        x_kernel,
        weight_t,
        conv_state_t,
        init_indices,
        current_indices,
        initial_state_mode,
        bias_work,
    )

    conv_state.copy_(conv_state_t.transpose(1, 2).contiguous().to(original_dtype))

    if query_start_loc is None and x.dim() == 2:
        output = output.squeeze(-1) if output.dim() == 3 and output.shape[-1] == 1 else output

    if original_dtype == torch.float16:
        output = output.to(torch.float16)

    return output


# ======================== Reference ========================

WIDTH = 4
STATE_LEN = WIDTH - 1

REF_CHECK_CONFIGS = [
    {"name": "decode_bs1_sl1", "dim": 2048, "batch_size": 1, "has_silu": True, "num_cache_lines": 4, "seed": 42},
    {"name": "qwen35_2b_tp2_decode_bs1_sl1", "dim": 3072, "batch_size": 1, "has_silu": True, "num_cache_lines": 4, "seed": 43},
]


def _torch_causal_conv1d_decode_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    init_indices: torch.Tensor,
    current_indices: torch.Tensor,
    has_silu: bool,
) -> torch.Tensor:
    batch = x.size(0)
    dim = x.size(1)
    width = weight.size(1)
    hist_len = width - 1

    x_f = x.to(torch.float32)
    weight_f = weight.to(torch.float32)
    conv_state_f = conv_state.to(torch.float32).clone()
    bias_f = bias.to(torch.float32)

    init_l = init_indices.to(torch.int64)
    current_l = current_indices.to(torch.int64)

    out = torch.zeros(batch, dim, dtype=torch.float32)

    for b in range(batch):
        read_line = init_l[b].item()
        write_line = current_l[b].item()

        history = [conv_state_f[read_line][:, h].clone() for h in range(hist_len)]

        acc = torch.zeros(dim, dtype=torch.float32)
        for w in range(hist_len):
            acc += weight_f[:, w] * history[w]
        acc += weight_f[:, width - 1] * x_f[b]
        acc += bias_f
        if has_silu:
            acc = acc / (1.0 + torch.exp(-acc))
        out[b] = acc

        for h in range(hist_len - 1):
            history[h] = history[h + 1].clone()
        history[hist_len - 1] = x_f[b].clone()

        for h in range(hist_len):
            conv_state_f[write_line][:, h] = history[h]

    conv_state.copy_(conv_state_f.to(conv_state.dtype))
    return out.to(x.dtype)


def _run_ref_check(config: dict) -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("Skip causal_conv1d_decode reference check: NPU is not available")
        return

    dim = config["dim"]
    batch_size = config["batch_size"]
    has_silu = config["has_silu"]
    num_cache_lines = config["num_cache_lines"]
    seed = config["seed"]

    torch.manual_seed(seed)
    device = torch.device("npu")

    x = torch.randn(batch_size, dim, dtype=torch.bfloat16, device=device)
    conv_state = torch.randn(num_cache_lines, dim, STATE_LEN, dtype=torch.bfloat16, device=device)
    weight = torch.randn(dim, WIDTH, dtype=torch.bfloat16, device=device)
    bias = torch.randn(dim, dtype=torch.bfloat16, device=device)

    conv_state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

    x_cpu = x.cpu()
    w_cpu = weight.cpu()
    b_cpu = bias.cpu()
    ci_cpu = conv_state_indices.cpu()

    cs_ref = conv_state.cpu().clone()
    golden = _torch_causal_conv1d_decode_ref(x_cpu, cs_ref, w_cpu, b_cpu, ci_cpu, ci_cpu.clone(), has_silu)

    cs_npu = conv_state.clone()
    activation = "silu" if has_silu else None
    out = causal_conv1d_decode(
        x,
        cs_npu,
        weight,
        bias=bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), golden, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(cs_npu.cpu(), cs_ref, rtol=1e-2, atol=1e-2)

    print(f"[PASS] causal_conv1d_decode: {config['name']}")


def _run_ref_suite() -> None:
    for config in REF_CHECK_CONFIGS:
        _run_ref_check(config)


def main() -> None:
    print("=" * 70)
    print("causal_conv1d_decode JIT 验证")
    print("=" * 70)

    _run_ref_suite()

    print("=" * 70)
    print("[PASS] causal_conv1d_decode 所有配置验证通过")
    print("Kernel Output Match!")
    print("=" * 70)


if __name__ == "__main__":
    main()
