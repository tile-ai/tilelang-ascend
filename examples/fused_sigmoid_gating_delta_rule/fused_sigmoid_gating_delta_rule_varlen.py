import math
import tilelang
from tilelang import language as T
import torch

tilelang.cache.clear_cache()
SOFTPLUS_THRESHOLD = 20.0
VEC_NUM = 2
L2_NORM_EPS = 1e-12
NUM_CORES = 24


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    },
)
def kernel(
    num_seqs,
    num_cache_slots,
    total_tokens_padded,
    max_seq_len,
    nk,
    nv,
    dk,
    dv,
    block_v,
    scale,
    use_qk_l2norm=True,
    softplus_beta=1.0,
    dtype="float16",
    accum_dtype="float",
):
    if nv % nk != 0:
        raise ValueError("nv must be divisible by nk")
    if block_v % VEC_NUM != 0:
        raise ValueError(f"block_v must be divisible by VEC_NUM={VEC_NUM}")

    num_v_tiles = math.ceil(dv / block_v)
    v_per_k = nv // nk
    vec_block_v = block_v // VEC_NUM
    inv_beta = 1.0 / softplus_beta

    block_num = num_seqs * nv * num_v_tiles
    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES
    max_work_per_block = q_tasks + 1

    @T.prim_func
    def main(
        A_log: T.Tensor([nv], dtype),
        a: T.Tensor([total_tokens_padded, nv], dtype),
        dt_bias: T.Tensor([nv], dtype),
        query: T.Tensor([total_tokens_padded, nk, dk], dtype),
        key: T.Tensor([total_tokens_padded, nk, dk], dtype),
        value: T.Tensor([total_tokens_padded, nv, dv], dtype),
        beta: T.Tensor([total_tokens_padded, nv], dtype),
        init_state: T.Tensor([num_cache_slots, nv, dv, dk], dtype),
        ssm_state_indices: T.Tensor([num_seqs], "int32"),
        cu_seqlens: T.Tensor([num_seqs + 1], "int32"),
        out: T.Tensor([total_tokens_padded, nv, dv], dtype),
        final_state: T.Tensor([num_seqs, nv, dv, dk], dtype),
    ):
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            start_work = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            count_work = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            # -- QKV ping-pong (double-buffered for prefetch) --
            q_buf = T.alloc_ub([2, dk], dtype)  # [2, dk]
            k_buf = T.alloc_ub([2, dk], dtype)  # [2, dk]
            v_buf = T.alloc_ub([2, vec_block_v], dtype)  # [2, vec_block_v]

            # -- current-token QKV → fp32 --
            q_f = T.alloc_ub([dk], accum_dtype)  # [dk]  Q
            k_f = T.alloc_ub([dk], accum_dtype)  # [dk]  K
            k_1d = T.alloc_ub([1, dk], accum_dtype)  # [1, dk]  broadcast src
            v_f = T.alloc_ub([vec_block_v], accum_dtype)  # [vec_block_v]  V
            k_broadcasted = T.alloc_ub([vec_block_v, dk], accum_dtype)  # [vec_block_v, dk]
            compute_buffer = T.alloc_ub([vec_block_v, dk], accum_dtype)  # [vec_block_v, dk] workspace

            # -- state matrix H ∈ R^{vec_block_v × dk} --
            h_vec = T.alloc_ub([vec_block_v, dk], accum_dtype)  # [vec_block_v, dk]  fp32 state
            h_load_vec = T.alloc_ub([vec_block_v, dk], dtype)  # [vec_block_v, dk]  load init_state
            h_store_vec = T.alloc_ub([vec_block_v, dk], dtype)  # [vec_block_v, dk]  store final_state

            # -- prediction & delta --
            pred_vec = T.alloc_ub([vec_block_v], accum_dtype)  # [vec_block_v]  H @ K
            pred_1d = T.alloc_ub([vec_block_v, 1], accum_dtype)  # [vec_block_v, 1]  reduce dst
            delta_vec = T.alloc_ub([vec_block_v], accum_dtype)  # [vec_block_v]  (V−pred)·gate
            delta_1d = T.alloc_ub([vec_block_v, 1], accum_dtype)  # [vec_block_v, 1]  broadcast src
            o_half_buf = T.alloc_ub([2, vec_block_v], dtype)  # [2, vec_block_v] output ping-pong

            # -- scalars --
            scalar_fp32 = T.alloc_ub([1], accum_dtype)  # [1] fp32 ws
            scalar_fp16 = T.alloc_ub([1], dtype)  # [1] fp16 load src
            scalar2_fp32 = T.alloc_ub([1], accum_dtype)  # [1] fp32 ws
            scalar2_fp16 = T.alloc_ub([1], dtype)  # [1] fp16 load src
            softplus_val = T.alloc_ub([1], accum_dtype)  # [1] softplus(x)
            alpha_val = T.alloc_ub([1], accum_dtype)  # [1] α = exp(−exp_A·softplus(x))

            # -- L2 norm workspace --
            norm_sq = T.alloc_ub([1, dk], accum_dtype)  # [1, dk]  q² or k²
            norm_val = T.alloc_ub([1], accum_dtype)  # [1]  ||q||²

            # ===== work assignment: flat_idx → (seq, v_head, v_tile) =====
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

                    # -- init state H ← init_state[state_idx],  H = 0 if no prior state --
                    state_idx = ssm_state_indices[seq_idx]
                    T.tile.fill(h_vec, 0.0)  # H = 0  [vec_block_v, dk]
                    if state_idx >= 0:
                        T.copy(init_state[state_idx, v_head_idx, v_offset : v_offset + vec_block_v, :], h_load_vec)
                        T.set_flag("mte2", "v", 1)
                        T.wait_flag("mte2", "v", 1)
                        T.tile.cast(h_vec, h_load_vec, "CAST_NONE", vec_block_v * dk)  # fp16→fp32

                    # -- exp_A = exp(A_log[v_head])  … scalar, used throughout the scan --
                    T.copy(A_log[v_head_idx : v_head_idx + 1], scalar_fp16)
                    T.set_flag("mte2", "v", 2)
                    T.wait_flag("mte2", "v", 2)
                    T.copy(scalar_fp16, scalar_fp32)
                    T.tile.exp(softplus_val, scalar_fp32)
                    exp_A = softplus_val[0]  # exp_A ∈ ℝ

                    # -- dt_bias[v_head]  … per-head time delta bias --
                    T.set_flag("v", "mte2", 3)
                    T.wait_flag("v", "mte2", 3)
                    T.copy(dt_bias[v_head_idx : v_head_idx + 1], scalar_fp16)
                    T.set_flag("mte2", "v", 4)
                    T.wait_flag("mte2", "v", 4)
                    T.copy(scalar_fp16, scalar_fp32)
                    dt_val = scalar_fp32[0]  # dt_bias ∈ ℝ

                    # -- prefetch first token's Q,K,V → ping-pong buf[0] --
                    T.copy(query[seq_start, k_head_idx, :], q_buf[0, :])  # Q[0,k_head,:]  [dk]
                    T.copy(key[seq_start, k_head_idx, :], k_buf[0, :])  # K[0,k_head,:]  [dk]
                    T.copy(value[seq_start, v_head_idx, v_offset : v_offset + vec_block_v], v_buf[0, :])  # V[0,v_head,v_off:v_off+vec_blk]
                    T.set_flag("mte2", "v", 6)

                    # ================================================================
                    #  Mamba SSM scan — iterate over tokens of this sequence
                    # ================================================================
                    for t in T.serial(seq_len):
                        token_idx = seq_start + t
                        buf_idx = t % 2  # 0/1 ping-pong index

                        T.wait_flag("mte2", "v", 6)

                        # -- a_t, β_t: scalar per head  [1] → fp32 --
                        T.copy(a[token_idx, v_head_idx : v_head_idx + 1], scalar_fp16)
                        T.copy(beta[token_idx, v_head_idx : v_head_idx + 1], scalar2_fp16)
                        T.set_flag("mte2", "v", 5)
                        T.wait_flag("mte2", "v", 5)
                        T.copy(scalar_fp16, scalar_fp32)
                        T.copy(scalar2_fp16, scalar2_fp32)
                        a_val = scalar_fp32[0]  # a_{t,v_head} ∈ ℝ
                        b_val = scalar2_fp32[0]  # β_{t,v_head} ∈ ℝ

                        # x = a + dt_bias,  then softplus(x, β)
                        # sp = log(1 + exp(x·β)) / β  (≈ x when x·β > SOFTPLUS_THRESHOLD)
                        x = a_val + dt_val
                        beta_x = x * softplus_beta

                        if beta_x > SOFTPLUS_THRESHOLD:
                            softplus_val[0] = x
                        else:
                            scalar_fp32[0] = beta_x
                            T.tile.exp(alpha_val, scalar_fp32)  # exp(x·β)
                            T.tile.add(alpha_val, alpha_val, 1.0)  # 1 + exp(x·β)
                            T.tile.ln(alpha_val, alpha_val)  # ln(1+exp(x·β))
                            softplus_val[0] = alpha_val[0] * inv_beta  # / β

                        # -- state decay factor:  α = exp(−exp_A · softplus(x))  ∈ ℝ --
                        scalar_fp32[0] = -exp_A * softplus_val[0]
                        T.tile.exp(alpha_val, scalar_fp32)

                        # -- sigmoid gate:  gate = σ(β) = 1 / (1 + exp(−β))  ∈ ℝ --
                        scalar_fp32[0] = -b_val
                        T.tile.exp(scalar_fp32, scalar_fp32)  # exp(−β)
                        T.tile.add(scalar_fp32, scalar_fp32, 1.0)  # 1 + exp(−β)
                        T.tile.reciprocal(scalar_fp32, scalar_fp32)  # 1 / (1+exp(−β))
                        beta_gate_scalar = scalar_fp32[0]

                        # -- q, k, v: fp16 → fp32  [dk] / [vec_block_v] --
                        T.copy(q_buf[buf_idx, :], q_f)  # q_t: [dk] → fp32
                        T.copy(k_buf[buf_idx, :], k_f)  # k_t: [dk] → fp32
                        T.copy(v_buf[buf_idx, :], v_f)  # v_t: [vec_block_v] → fp32

                        # -- prefetch next token's Q,K,V into ping-pong alternate slot --
                        if t + 1 < seq_len:
                            next_token_idx = seq_start + t + 1
                            next_buf_idx = (t + 1) % 2
                            T.copy(query[next_token_idx, k_head_idx, :], q_buf[next_buf_idx, :])
                            T.copy(key[next_token_idx, k_head_idx, :], k_buf[next_buf_idx, :])
                            T.copy(value[next_token_idx, v_head_idx, v_offset : v_offset + vec_block_v], v_buf[next_buf_idx, :])
                            T.set_flag("mte2", "v", 6)

                        # -- (optional) L2 norm:  q̂ = q / √(‖q‖²+ε),  k̂ = k / √(‖k‖²+ε) --
                        if use_qk_l2norm:
                            T.tile.mul(norm_sq[0, :], q_f, q_f)  # q²  [1, dk]
                            T.reduce_sum(norm_sq, norm_val, dim=-1)  # Σq² → [1]
                            T.tile.add(norm_val, norm_val, L2_NORM_EPS)  # Σq²+ε
                            T.tile.rsqrt(norm_val, norm_val)  # 1/√(Σq²+ε)
                            norm_scalar = norm_val[0]
                            T.tile.mul(q_f, q_f, norm_scalar)  # q̂ = q · 1/√(‖q‖²+ε)

                            T.tile.mul(norm_sq[0, :], k_f, k_f)  # k²  [1, dk]
                            T.reduce_sum(norm_sq, norm_val, dim=-1)  # Σk² → [1]
                            T.tile.add(norm_val, norm_val, L2_NORM_EPS)  # Σk²+ε
                            T.tile.rsqrt(norm_val, norm_val)
                            norm_scalar = norm_val[0]
                            T.tile.mul(k_f, k_f, norm_scalar)  # k̂ = k · 1/√(‖k‖²+ε)

                        # -- q *= scale  (e.g. scale = 1/√dk) --
                        T.tile.mul(q_f, q_f, scale)

                        # =====  state decay:  H = H · α   [vec_block_v, dk]  =====
                        T.tile.mul(h_vec, h_vec, alpha_val[0])

                        # =====  pred = H @ k̂   [vec_block_v,dk]·[dk] → [vec_block_v]  =====
                        T.copy(k_f, k_1d[0, :])  # 1D →
                        T.tile.broadcast(k_broadcasted, k_1d)  #    → [vec_block_v, dk]
                        T.tile.mul(compute_buffer, h_vec, k_broadcasted)  # H⊙k
                        T.reduce_sum(compute_buffer, pred_1d[:, 0], dim=-1)  # Σ_j H[v,j]·k[j]
                        T.copy(pred_1d[:, 0], pred_vec)  # pred_v

                        # Δ = (v − pred) · gate   → [vec_block_v]
                        T.tile.sub(delta_vec, v_f, pred_vec)  # v − pred
                        T.tile.mul(delta_vec, delta_vec, beta_gate_scalar)  # (v−pred)·σ(β)

                        # =====  H = H + k̂ ⊗ Δ  (outer-product update)  =====
                        #   H[v,j] += k[j] · Δ[v]
                        T.copy(delta_vec, delta_1d[:, 0])  # → [vec_block_v, 1]
                        T.tile.broadcast(compute_buffer, delta_1d)  # broadcast to [vec_block_v,dk]
                        T.tile.mul_add_dst(h_vec, k_broadcasted, compute_buffer)

                        # =====  o = H @ q̂   [vec_block_v,dk]·[dk] → [vec_block_v]  =====
                        #   (q̂ was already multiplied by scale above)
                        T.copy(q_f, k_1d[0, :])
                        T.tile.broadcast(k_broadcasted, k_1d)  # q̂ → [vec_block_v, dk]
                        T.tile.mul(compute_buffer, h_vec, k_broadcasted)  # H⊙q̂
                        T.reduce_sum(compute_buffer, pred_1d[:, 0], dim=-1)  # Σ_j H[v,j]·q̂[j]
                        T.copy(pred_1d[:, 0], o_half_buf[buf_idx, :])  # fp32 o_t → o_half

                        # -- write output tile → out[token, v_head, v_off:v_off+vec_blk] --
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.copy(o_half_buf[buf_idx, :], out[token_idx, v_head_idx, v_offset : v_offset + vec_block_v])

                    # -- store final state H → final_state[seq_idx] (fp32→fp16, round-to-nearest) --
                    T.tile.cast(h_store_vec, h_vec, "CAST_RINT", vec_block_v * dk)
                    T.set_flag("v", "mte3", 5)
                    T.wait_flag("v", "mte3", 5)
                    T.copy(h_store_vec, final_state[seq_idx, v_head_idx, v_offset : v_offset + vec_block_v, :])

    return main


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

    state = torch.zeros((num_seqs, nv, dk, dv), dtype=torch.float32, device=query.device)
    for i in range(num_seqs):
        state_idx = ssm_state_indices[i].item()
        if state_idx >= 0:
            state[i] = init_state[state_idx].float().clone()
    out = torch.empty((1, total_tokens, nv, dv), dtype=torch.float32, device=query.device)

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
                h = h + torch.outer(k_t, (v_t - pred) * torch.sigmoid(beta[token_idx, v_head_idx].float()))
                out[0, token_idx, v_head_idx] = (q_t * scale) @ h
            state[seq_idx, v_head_idx] = h

    return out.to(query.dtype), state.to(init_state.dtype)


def main(
    seqlens=None,
    batch_size=1,
    seq_len=256,
    nk=4,
    nv=8,
    dk=128,
    dv=128,
    use_qk_l2norm=True,
    softplus_beta=1.0,
    block_v=128,
):
    torch.manual_seed(41)
    device = "npu"

    if seqlens:
        total_tokens = sum(seqlens)
        num_seqs = len(seqlens)
        cu_seqlens_cpu = torch.tensor([0] + [sum(seqlens[: i + 1]) for i in range(num_seqs)], dtype=torch.int32)
        query_cpu = torch.randn((1, total_tokens, nk, dk), dtype=torch.float16)
        key_cpu = torch.randn((1, total_tokens, nk, dk), dtype=torch.float16)
        value_cpu = torch.randn((1, total_tokens, nv, dv), dtype=torch.float16)
        a_cpu = torch.randn((total_tokens, nv), dtype=torch.float16)
        beta_cpu = torch.randn((total_tokens, nv), dtype=torch.float16)
    else:
        total_tokens = batch_size * seq_len
        num_seqs = batch_size
        cu_seqlens_cpu = torch.arange(0, total_tokens + 1, seq_len, dtype=torch.int32)
        query_cpu = torch.randn((batch_size, seq_len, nk, dk), dtype=torch.float16)
        key_cpu = torch.randn((batch_size, seq_len, nk, dk), dtype=torch.float16)
        value_cpu = torch.randn((batch_size, seq_len, nv, dv), dtype=torch.float16)
        a_cpu = torch.randn((batch_size, seq_len, nv), dtype=torch.float16)
        beta_cpu = torch.randn((batch_size, seq_len, nv), dtype=torch.float16)
        query_cpu = query_cpu.reshape(1, total_tokens, nk, dk)
        key_cpu = key_cpu.reshape(1, total_tokens, nk, dk)
        value_cpu = value_cpu.reshape(1, total_tokens, nv, dv)
        a_cpu = a_cpu.reshape(total_tokens, nv)
        beta_cpu = beta_cpu.reshape(total_tokens, nv)

    num_cache_slots = num_seqs * 10
    init_state_cpu = torch.randn((num_cache_slots, nv, dk, dv), dtype=torch.float16)
    ssm_state_indices_cpu = torch.arange(num_seqs, dtype=torch.int32)
    A_log_cpu = torch.randn((nv,), dtype=torch.float16)
    dt_bias_cpu = torch.randn((nv,), dtype=torch.float16)
    scale = dk**-0.5

    block_v = min(block_v, dv)
    if block_v >= 32:
        block_v = (block_v // 32) * 32
    if block_v < VEC_NUM:
        block_v = VEC_NUM if dv >= VEC_NUM else block_v
    if block_v % VEC_NUM != 0:
        block_v = (block_v // VEC_NUM) * VEC_NUM

    while block_v > 0 and dv % block_v != 0:
        block_v -= VEC_NUM
    if block_v <= 0:
        block_v = min(dv, VEC_NUM)
        if block_v % VEC_NUM != 0:
            block_v = (block_v // VEC_NUM) * VEC_NUM
    if block_v <= 0 or block_v > dv or block_v % VEC_NUM != 0 or dv % block_v != 0:
        raise ValueError(
            f"illegal block_v={block_v} for dv={dv}; require 0 < block_v <= dv, block_v % {VEC_NUM} == 0, and dv % block_v == 0"
        )

    cu_seqlens_np = cu_seqlens_cpu.numpy()
    max_seq_len = int(max(cu_seqlens_np[i + 1] - cu_seqlens_np[i] for i in range(num_seqs)))
    padding = 64

    def pad_tensor_cpu(t):
        padding_tensor = torch.zeros((padding,) + t.shape[1:], dtype=t.dtype)
        return torch.cat([t, padding_tensor], dim=0)

    dtype_str = "float16"
    init_state_vk_cpu = init_state_cpu.transpose(-1, -2).contiguous()
    ker = kernel(
        num_seqs,
        num_cache_slots,
        total_tokens + padding,
        max_seq_len,
        nk,
        nv,
        dk,
        dv,
        block_v,
        scale,
        use_qk_l2norm,
        softplus_beta,
        dtype_str,
        "float",
    )
    with open("./ascendc.cpp", "w") as fp:
        fp.write(ker.get_kernel_source())

    A_log = A_log_cpu.to(device)
    a = pad_tensor_cpu(a_cpu).to(device)
    dt_bias = dt_bias_cpu.to(device)
    query = pad_tensor_cpu(query_cpu.squeeze(0)).to(device)
    key = pad_tensor_cpu(key_cpu.squeeze(0)).to(device)
    value = pad_tensor_cpu(value_cpu.squeeze(0)).to(device)
    beta = pad_tensor_cpu(beta_cpu).to(device)
    init_state_vk = init_state_vk_cpu.to(device)
    ssm_state_indices = ssm_state_indices_cpu.to(device)
    cu_seqlens = cu_seqlens_cpu.to(device)

    out, final_state = ker(
        A_log,
        a,
        dt_bias,
        query,
        key,
        value,
        beta,
        init_state_vk,
        ssm_state_indices,
        cu_seqlens,
    )
    out = out[:total_tokens].unsqueeze(0)
    final_state = final_state.transpose(-1, -2).contiguous()

    out_golden, final_state_golden = golden(
        A_log_cpu,
        a_cpu,
        dt_bias_cpu,
        query_cpu,
        key_cpu,
        value_cpu,
        beta_cpu,
        init_state_cpu,
        ssm_state_indices_cpu,
        cu_seqlens_cpu,
        scale,
        use_qk_l2norm,
        softplus_beta,
    )

    torch.testing.assert_close(out.cpu(), out_golden, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(final_state.cpu(), final_state_golden, rtol=2e-2, atol=2e-2)
    print("Kernel Output Match!")


if __name__ == "__main__":
    import threading
    import argparse

    test_cases = [
        {
            "seqlens": [4, 8] * 50,
            "nk": 16,
            "nv": 32,
            "dk": 128,
            "dv": 128,
        },
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    if args.threads > 1:
        threads = []
        for i, case in enumerate(test_cases):
            t = threading.Thread(target=main, kwargs=case, name=f"Test-{i + 1}")
            threads.append(t)
            t.start()
            if len(threads) >= args.threads:
                for t in threads:
                    t.join()
                threads = []

        for t in threads:
            t.join()
    else:
        for i, case in enumerate(test_cases):
            print(f"Running test case {i + 1}: {case}")
            main(**case)

    print("All tests passed!")
