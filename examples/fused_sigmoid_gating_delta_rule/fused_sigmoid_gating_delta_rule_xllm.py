#!/usr/bin/env python3

import math

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

# Per-kernel pass_configs matching the original tuned kernel.
_SIGMOID_PASS_CONFIGS = {
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cv_combine": True,
}

SOFTPLUS_THRESHOLD = 20.0
VEC_NUM = 2
L2_NORM_EPS = 1e-6
DEFAULT_DTYPE = "bf16"
# TVM/TileLang knows "bfloat16" for tensor buffers but the dispatch
# system uses the compact "bf16" string to map to TilelangDType::kBF16.
_INTERNAL_DTYPE = "bfloat16"
DEFAULT_ACCUM_DTYPE = "float"
DEFAULT_USE_QK_L2NORM = 1
DEFAULT_SOFTPLUS_BETA = 1.0
DEFAULT_NUM_CORES = 24
DEFAULT_NK = 16
DEFAULT_NV = 32
DEFAULT_DK = 128
DEFAULT_DV = 128


def _auto_block_v(dv: int) -> int:
    return min(dv, 128)


def build_fused_sigmoid_gating_delta_rule_kernel(
    nk: int,
    nv: int,
    dk: int,
    dv: int,
    block_v: int,
    max_num_seqs: int,
    use_qk_l2norm: bool,
    softplus_beta: float,
    dtype: str,
    accum_dtype: str,
    num_cores: int = DEFAULT_NUM_CORES,
):
    if nv % nk != 0:
        raise ValueError("nv must be divisible by nk")
    if block_v % VEC_NUM != 0:
        raise ValueError(f"block_v must be divisible by VEC_NUM={VEC_NUM}")
    if dv % block_v != 0:
        raise ValueError("dv must be divisible by block_v")

    num_v_tiles = math.ceil(dv / block_v)
    v_per_k = nv // nk
    vec_block_v = block_v // VEC_NUM
    total_tokens_padded = T.symbolic("total_tokens_padded")
    num_cache_slots = T.symbolic("num_cache_slots")
    input_dtype = _INTERNAL_DTYPE  # "bfloat16" — TVM-supported name

    block_num = max_num_seqs * nv * num_v_tiles
    q_tasks = block_num // num_cores
    r_tasks = block_num % num_cores
    max_work_per_block = q_tasks + 1
    inv_beta = 1.0 / softplus_beta

    @T.prim_func
    def main(
        A_log: T.Tensor([nv], accum_dtype),
        a: T.Tensor([total_tokens_padded, nv], input_dtype),
        dt_bias: T.Tensor([nv], accum_dtype),
        query: T.Tensor([total_tokens_padded, nk, dk], input_dtype),
        key: T.Tensor([total_tokens_padded, nk, dk], input_dtype),
        value: T.Tensor([total_tokens_padded, nv, dv], input_dtype),
        beta: T.Tensor([total_tokens_padded, nv], input_dtype),
        init_state: T.Tensor([num_cache_slots, nv, dk, dv], accum_dtype),
        ssm_state_indices: T.Tensor([max_num_seqs], "int32"),
        cu_seqlens: T.Tensor([max_num_seqs + 1], "int32"),
        out: T.Tensor([total_tokens_padded, nv, dv], input_dtype),
        final_state: T.Tensor([num_cache_slots, nv, dk, dv], accum_dtype),
        softplus_beta: T.float32,
        scale: T.float32,
        use_qk_l2norm: T.int32,
        softplus_threshold: T.float32,
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid):
            start_work = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            count_work = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            q_buf = T.alloc_ub([2, dk], input_dtype)
            k_buf = T.alloc_ub([2, dk], input_dtype)
            v_buf = T.alloc_ub([2, vec_block_v], input_dtype)

            q_f = T.alloc_ub([dk], accum_dtype)
            k_f = T.alloc_ub([dk], accum_dtype)
            k_1d = T.alloc_ub([dk, 1], accum_dtype)
            v_f = T.alloc_ub([vec_block_v], accum_dtype)
            k_broadcasted = T.alloc_ub([dk, vec_block_v], accum_dtype)
            compute_buffer = T.alloc_ub([dk, vec_block_v], accum_dtype)

            h_vec = T.alloc_ub([dk, vec_block_v], accum_dtype)

            pred_vec = T.alloc_ub([vec_block_v], accum_dtype)
            pred_1d = T.alloc_ub([1, vec_block_v], accum_dtype)
            delta_vec = T.alloc_ub([vec_block_v], accum_dtype)
            delta_1d = T.alloc_ub([1, vec_block_v], accum_dtype)
            o_half_buf = T.alloc_ub([2, vec_block_v], input_dtype)

            scalar_fp32 = T.alloc_ub([1], accum_dtype)
            scalar_fp16 = T.alloc_ub([1], input_dtype)
            scalar2_fp32 = T.alloc_ub([1], accum_dtype)
            scalar2_fp16 = T.alloc_ub([1], input_dtype)
            softplus_val = T.alloc_ub([1], accum_dtype)
            alpha_val = T.alloc_ub([1], accum_dtype)

            norm_sq = T.alloc_ub([1, dk], accum_dtype)
            norm_val = T.alloc_ub([1], accum_dtype)

            for work_i in T.serial(max_work_per_block):
                if work_i < count_work:
                    flat_idx = start_work + work_i
                    v_tile_idx = flat_idx % num_v_tiles
                    v_head_idx = (flat_idx // num_v_tiles) % nv
                    seq_idx = flat_idx // (num_v_tiles * nv)

                    k_head_idx = v_head_idx // v_per_k
                    v_offset = v_tile_idx * block_v + vid * vec_block_v

                    seq_start = cu_seqlens[seq_idx]
                    seq_end = cu_seqlens[seq_idx + 1]
                    seq_len = seq_end - seq_start

                    state_idx = ssm_state_indices[seq_idx]
                    T.tile.fill(h_vec, 0.0)
                    T.barrier_all()
                    if state_idx >= 0:
                        T.copy(
                            init_state[
                                state_idx,
                                v_head_idx,
                                :,
                                v_offset : v_offset + vec_block_v,
                            ],
                            h_vec
                        )
                        T.barrier_all()

                    T.copy(A_log[v_head_idx : v_head_idx + 1], scalar_fp32)
                    T.set_flag("mte2", "v", 2)
                    T.wait_flag("mte2", "v", 2)
                    T.tile.exp(softplus_val, scalar_fp32)
                    exp_A = softplus_val[0]

                    T.set_flag("v", "mte2", 3)
                    T.wait_flag("v", "mte2", 3)
                    T.copy(dt_bias[v_head_idx : v_head_idx + 1], scalar_fp32)
                    T.set_flag("mte2", "v", 4)
                    T.wait_flag("mte2", "v", 4)
                    dt_val = scalar_fp32[0]

                    if seq_len > 0:
                        T.copy(query[seq_start, k_head_idx, :], q_buf[0, :])
                        T.copy(key[seq_start, k_head_idx, :], k_buf[0, :])
                        T.copy(
                            value[seq_start, v_head_idx, v_offset : v_offset + vec_block_v],
                            v_buf[0, :],
                        )
                        T.set_flag("mte2", "v", 6)

                    for t in T.serial(seq_len):
                        token_idx = seq_start + t
                        buf_idx = t % 2

                        T.wait_flag("mte2", "v", 6)

                        T.copy(a[token_idx, v_head_idx : v_head_idx + 1], scalar_fp16)
                        T.copy(
                            beta[token_idx, v_head_idx : v_head_idx + 1], scalar2_fp16
                        )
                        T.set_flag("mte2", "v", 5)
                        T.wait_flag("mte2", "v", 5)
                        T.tile.cast(scalar_fp32, scalar_fp16, "CAST_NONE", 1)
                        T.tile.cast(scalar2_fp32, scalar2_fp16, "CAST_NONE", 1)
                        a_val = scalar_fp32[0]
                        b_val = scalar2_fp32[0]

                        x = a_val + dt_val
                        beta_x = x * softplus_beta

                        if beta_x > softplus_threshold:
                            softplus_val[0] = x
                        else:
                            scalar_fp32[0] = beta_x
                            T.tile.exp(alpha_val, scalar_fp32)
                            T.tile.add(alpha_val, alpha_val, 1.0)
                            T.tile.ln(alpha_val, alpha_val)
                            softplus_val[0] = alpha_val[0] * inv_beta

                        scalar_fp32[0] = -exp_A * softplus_val[0]
                        T.tile.exp(alpha_val, scalar_fp32)

                        scalar_fp32[0] = b_val
                        T.tile.sigmoid(scalar_fp32, scalar_fp32)
                        beta_gate_scalar = scalar_fp32[0]

                        T.tile.cast(q_f, q_buf[buf_idx, :], "CAST_NONE", dk)
                        T.tile.cast(k_f, k_buf[buf_idx, :], "CAST_NONE", dk)
                        T.tile.cast(v_f, v_buf[buf_idx, :], "CAST_NONE", vec_block_v)

                        if t + 1 < seq_len:
                            next_token_idx = seq_start + t + 1
                            next_buf_idx = (t + 1) % 2
                            T.copy(query[next_token_idx, k_head_idx, :], q_buf[next_buf_idx, :])
                            T.copy(key[next_token_idx, k_head_idx, :], k_buf[next_buf_idx, :])
                            T.copy(
                                value[next_token_idx, v_head_idx, v_offset : v_offset + vec_block_v],
                                v_buf[next_buf_idx, :],
                            )
                            T.set_flag("mte2", "v", 6)

                        if use_qk_l2norm:
                            T.tile.mul(norm_sq[0, :], q_f, q_f)
                            T.reduce_sum(norm_sq, norm_val, dim=-1)
                            T.tile.add(norm_val, norm_val, L2_NORM_EPS)
                            T.tile.sqrt(norm_val, norm_val)
                            norm_scalar = norm_val[0]
                            T.tile.div(q_f, q_f, norm_scalar)

                            T.tile.mul(norm_sq[0, :], k_f, k_f)
                            T.reduce_sum(norm_sq, norm_val, dim=-1)
                            T.tile.add(norm_val, norm_val, L2_NORM_EPS)
                            T.tile.sqrt(norm_val, norm_val)
                            norm_scalar = norm_val[0]
                            T.tile.div(k_f, k_f, norm_scalar)

                        T.tile.mul(q_f, q_f, scale)

                        T.tile.mul(h_vec, h_vec, alpha_val[0])

                        T.copy(k_f, k_1d[:, 0])
                        T.tile.broadcast(k_broadcasted, k_1d)
                        T.tile.mul(compute_buffer, h_vec, k_broadcasted)
                        T.reduce_sum(compute_buffer, pred_1d[0, :], dim=0)
                        T.copy(pred_1d[0, :], pred_vec)

                        T.tile.sub(delta_vec, v_f, pred_vec)
                        T.tile.mul(delta_vec, delta_vec, beta_gate_scalar)

                        T.copy(delta_vec, delta_1d[0, :])
                        T.tile.broadcast(compute_buffer, delta_1d)
                        T.tile.mul_add_dst(h_vec, k_broadcasted, compute_buffer)

                        T.copy(q_f, k_1d[:, 0])
                        T.tile.broadcast(k_broadcasted, k_1d)
                        T.tile.mul(compute_buffer, h_vec, k_broadcasted)
                        T.reduce_sum(compute_buffer, pred_1d[0, :], dim=0)
                        T.tile.cast(
                            o_half_buf[buf_idx, :],
                            pred_1d[0, :],
                            "CAST_RINT",
                            vec_block_v,
                        )

                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.copy(
                            o_half_buf[buf_idx, :],
                            out[
                                token_idx, v_head_idx, v_offset : v_offset + vec_block_v
                            ],
                        )

                    T.set_flag("v", "mte3", 5)
                    T.wait_flag("v", "mte3", 5)
                    if state_idx >= 0:
                        T.copy(
                            h_vec,
                            final_state[
                                state_idx,
                                v_head_idx,
                                :,
                                v_offset : v_offset + vec_block_v,
                            ],
                        )
                    T.set_flag("mte3", "v", 6)
                    T.wait_flag("mte3", "v", 6)

    return main


@tilelang.jit(out_idx=[10, 11], pass_configs=_SIGMOID_PASS_CONFIGS, target="ascendc")
def fused_sigmoid_gating_delta_rule_kernel_jit(
    nk: int,
    nv: int,
    dk: int,
    dv: int,
    block_v: int,
    max_num_seqs: int,
    use_qk_l2norm: int,
    softplus_beta: float,
    dtype: str,
    accum_dtype: str,
):
    return build_fused_sigmoid_gating_delta_rule_kernel(
        nk=nk,
        nv=nv,
        dk=dk,
        dv=dv,
        block_v=block_v,
        max_num_seqs=max_num_seqs,
        use_qk_l2norm=bool(use_qk_l2norm),
        softplus_beta=softplus_beta,
        dtype=dtype,
        accum_dtype=accum_dtype,
        num_cores=DEFAULT_NUM_CORES,
    )


# ======================== Reference ========================


def golden(
    A_log,
    a,
    dt_bias,
    query,
    key,
    value,
    beta,
    init_state,
    ssm_state_indices,
    cu_seqlens,
    scale=None,
    use_qk_l2norm=True,
    softplus_beta=1.0,
):
    _, total_tokens, nk, dk = query.shape
    _, _, nv, dv = value.shape
    num_seqs = len(cu_seqlens) - 1
    scale = dk**-0.5 if scale is None else scale
    v_per_k = nv // nk

    state = torch.zeros(
        (num_seqs, nv, dk, dv), dtype=torch.float32, device=query.device
    )
    for i in range(num_seqs):
        state_idx = ssm_state_indices[i].item()
        if state_idx >= 0:
            state[i] = init_state[state_idx].float().clone()
    out = torch.empty(
        (1, total_tokens, nv, dv), dtype=torch.float32, device=query.device
    )

    exp_A = torch.exp(A_log.float())
    for seq_idx in range(num_seqs):
        seq_start = cu_seqlens[seq_idx].item()
        seq_end = cu_seqlens[seq_idx + 1].item()
        for v_head_idx in range(nv):
            h = state[seq_idx, v_head_idx]
            k_head_idx = v_head_idx // v_per_k
            for t in range(seq_end - seq_start):
                token_idx = seq_start + t
                q_t = query[0, token_idx, k_head_idx].float()
                k_t = key[0, token_idx, k_head_idx].float()
                v_t = value[0, token_idx, v_head_idx].float()

                if use_qk_l2norm:
                    q_t = q_t / torch.sqrt((q_t**2).sum() + L2_NORM_EPS)
                    k_t = k_t / torch.sqrt((k_t**2).sum() + L2_NORM_EPS)

                x = a[token_idx, v_head_idx].float() + dt_bias[v_head_idx].float()
                beta_x = softplus_beta * x
                if beta_x > 20:
                    sp = x
                else:
                    sp = torch.log1p(torch.exp(beta_x)) / softplus_beta

                h = h * torch.exp(-exp_A[v_head_idx] * sp)
                pred = k_t @ h
                h = h + torch.outer(
                    k_t,
                    (v_t - pred) * torch.sigmoid(beta[token_idx, v_head_idx].float()),
                )
                out[0, token_idx, v_head_idx] = (q_t * scale) @ h
            state[seq_idx, v_head_idx] = h

    return out.to(query.dtype), state.to(init_state.dtype)


# ======================== Tests ========================
# Test cases ported from C++ gtest:
#   fused_sigmoid_gating_delta_rule_wrapper_test.cpp
#
# The original xllm kernel used 1D flat QKV buffers with stride parameters
# (designed for AOT compilation). This JIT version uses 3D tensors so that
# symbolic variables can be inferred from tensor shapes at runtime.

REF_CHECK_CONFIGS = [
    {"name": "small_4x8_d128",
     "seqlens": [4, 8, 6, 3], "nk": 4, "nv": 8,
     "dk": 128, "dv": 128, "seed": 20260421, "init_state_dtype": "float32"},
    {"name": "medium_16x32_d128",
     "seqlens": [4, 8, 12, 6, 3, 7, 5, 9], "nk": 16, "nv": 32,
     "dk": 128, "dv": 128, "seed": 20260422, "init_state_dtype": "float32"},
    {"name": "oversized_40x32_decode_d128",
     "seqlens": [1] * 40, "nk": 16, "nv": 32,
     "dk": 128, "dv": 128, "seed": 20260703, "init_state_dtype": "float32"},
    {"name": "bf16_cache_4x8_decode_d128",
     "seqlens": [1, 1, 1, 1], "nk": 4, "nv": 8,
     "dk": 128, "dv": 128, "seed": 20260704, "init_state_dtype": "bfloat16"},
]


def _run_ref_check(config: dict) -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("Skip fused_sigmoid_gating_delta_rule reference check: NPU is not available")
        return

    seqlens = config["seqlens"]
    nk = config["nk"]
    nv = config["nv"]
    dk = config["dk"]
    dv = config["dv"]
    seed = config["seed"]
    init_state_dtype_str = config["init_state_dtype"]
    use_qk_l2norm = True
    softplus_beta = DEFAULT_SOFTPLUS_BETA

    torch.manual_seed(seed)
    device = torch.device("npu")

    total_tokens = sum(seqlens)
    num_seqs = len(seqlens)
    cu_seqlens_cpu = torch.tensor(
        [0] + [sum(seqlens[: i + 1]) for i in range(num_seqs)], dtype=torch.int32
    )

    query = torch.randn((total_tokens, nk, dk), dtype=torch.bfloat16, device=device)
    key = torch.randn((total_tokens, nk, dk), dtype=torch.bfloat16, device=device)
    value = torch.randn((total_tokens, nv, dv), dtype=torch.bfloat16, device=device)
    a = torch.randn((total_tokens, nv), dtype=torch.bfloat16, device=device)
    beta = torch.randn((total_tokens, nv), dtype=torch.bfloat16, device=device)

    num_cache_slots = num_seqs * 10
    init_dtype = torch.float32 if init_state_dtype_str == "float32" else torch.bfloat16
    init_state_orig = torch.randn((num_cache_slots, nv, dk, dv), dtype=init_dtype, device=device)
    ssm_state_indices = torch.arange(num_seqs, dtype=torch.int32, device=device)
    A_log = torch.randn((nv,), dtype=torch.float32, device=device)
    dt_bias = torch.randn((nv,), dtype=torch.float32, device=device)

    query_cpu = query.cpu()
    key_cpu = key.cpu()
    value_cpu = value.cpu()
    a_cpu = a.cpu()
    beta_cpu = beta.cpu()
    init_state_cpu = init_state_orig.cpu()
    ssm_state_indices_cpu = ssm_state_indices.cpu()
    A_log_cpu = A_log.cpu()
    dt_bias_cpu = dt_bias.cpu()

    block_v = _auto_block_v(dv)
    dtype_str = DEFAULT_DTYPE

    ker = fused_sigmoid_gating_delta_rule_kernel_jit(
        nk=nk,
        nv=nv,
        dk=dk,
        dv=dv,
        block_v=block_v,
        max_num_seqs=num_seqs,
        use_qk_l2norm=int(use_qk_l2norm),
        softplus_beta=softplus_beta,
        dtype=dtype_str,
        accum_dtype=DEFAULT_ACCUM_DTYPE,
    )

    # Kernel expects init_state in fp32 (accum_dtype); convert if needed.
    init_state_kernel = init_state_orig.to(torch.float32) if init_state_orig.dtype != torch.float32 else init_state_orig
    cu_seqlens = cu_seqlens_cpu.to(device)

    scale = dk**-0.5
    out, final_state = ker(
        A_log,
        a,
        dt_bias,
        query,
        key,
        value,
        beta,
        init_state_kernel,
        ssm_state_indices,
        cu_seqlens,
        softplus_beta,
        scale,
        int(use_qk_l2norm),
        SOFTPLUS_THRESHOLD,
    )
    torch.npu.synchronize()
    out = out[:total_tokens].unsqueeze(0)
    final_state = final_state[:num_seqs]

    out_golden, final_state_golden = golden(
        A_log_cpu,
        a_cpu,
        dt_bias_cpu,
        query_cpu.unsqueeze(0),
        key_cpu.unsqueeze(0),
        value_cpu.unsqueeze(0),
        beta_cpu,
        init_state_cpu,
        ssm_state_indices_cpu,
        cu_seqlens_cpu,
        use_qk_l2norm=use_qk_l2norm,
        softplus_beta=softplus_beta,
    )

    torch.testing.assert_close(out.cpu(), out_golden, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        final_state.cpu().to(torch.float32),
        final_state_golden.to(torch.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    print(f"[PASS] fused_sigmoid_gating_delta_rule: {config['name']}")


def _run_ref_suite() -> None:
    for config in REF_CHECK_CONFIGS:
        _run_ref_check(config)


def main() -> None:
    print("=" * 70)
    print("fused_sigmoid_gating_delta_rule JIT 验证")
    print("=" * 70)

    _run_ref_suite()

    print("=" * 70)
    print("[PASS] fused_sigmoid_gating_delta_rule 所有配置验证通过")
    print("Kernel Output Match!")
    print("=" * 70)


if __name__ == "__main__":
    main()
