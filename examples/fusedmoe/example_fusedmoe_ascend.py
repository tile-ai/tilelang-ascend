import torch
import torch.nn as nn

import tilelang
import tilelang.language as T

from example_fusedmoe_torch import *

tilelang.cache.clear_cache()

# Expert mode: no auto-CV-combine, manual C/V scopes
pass_configs_expert = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(pass_configs=pass_configs_expert)
def moe_shared_gate_up_ascend(num_tokens, dhidden, dexpert, block_M=128, block_N=128, K_L1=64, dtype="float16", accum_dtype="float"):
    """Shared expert: GEMM gate+up then SiLU fusion using Expert mode C/V scopes."""
    m_num = T.ceildiv(num_tokens, block_M)
    n_num = T.ceildiv(dexpert, block_N)
    VEC_NUM = 2
    BLOCKS = m_num * n_num

    @T.prim_func
    def main(
        input: T.Tensor((num_tokens, dhidden), dtype),
        W_gate: T.Tensor((dexpert, dhidden), dtype),
        W_up: T.Tensor((dexpert, dhidden), dtype),
        ws_gate_logits: T.Tensor((num_tokens, dexpert), dtype),
        ws_up_logits: T.Tensor((num_tokens, dexpert), dtype),
        up_logits: T.Tensor((num_tokens, dexpert), dtype),
    ):
        with T.Kernel(BLOCKS, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            sub_row_start = bx * block_M + vid * block_M // VEC_NUM
            col_start = by * block_N

            with T.Scope("C"):
                A_L1 = T.alloc_L1((block_M, K_L1), dtype)
                B_gate_L1 = T.alloc_L1((block_N, K_L1), dtype)
                B_up_L1 = T.alloc_L1((block_N, K_L1), dtype)
                C_gate_L0C = T.alloc_L0C((block_M, block_N), accum_dtype)
                C_up_L0C = T.alloc_L0C((block_M, block_N), accum_dtype)

                loop_k = T.ceildiv(dhidden, K_L1)
                for k in T.serial(loop_k):
                    T.copy(input[bx * block_M : (bx + 1) * block_M, k * K_L1 : (k + 1) * K_L1], A_L1)
                    T.copy(W_gate[by * block_N : (by + 1) * block_N, k * K_L1 : (k + 1) * K_L1], B_gate_L1)
                    T.copy(W_up[by * block_N : (by + 1) * block_N, k * K_L1 : (k + 1) * K_L1], B_up_L1)
                    T.gemm_v0(A_L1, B_gate_L1, C_gate_L0C, transpose_B=True, init=(k == 0))
                    T.gemm_v0(A_L1, B_up_L1, C_up_L0C, transpose_B=True, init=(k == 0))

                T.copy(C_gate_L0C, ws_gate_logits[bx * block_M : (bx + 1) * block_M, col_start : col_start + block_N])
                T.copy(C_up_L0C, ws_up_logits[bx * block_M : (bx + 1) * block_M, col_start : col_start + block_N])
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                gate_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                up_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                denom_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                zero_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

                T.wait_cross_flag(0)
                T.copy(ws_gate_logits[sub_row_start : sub_row_start + block_M // VEC_NUM, col_start : col_start + block_N], gate_ub)

                T.tile.fill(zero_ub, 0.0)
                T.tile.sub(denom_ub, zero_ub, gate_ub)
                T.tile.exp(denom_ub, denom_ub)
                T.tile.add(denom_ub, denom_ub, 1.0)
                T.tile.div(gate_ub, gate_ub, denom_ub)

                T.copy(ws_up_logits[sub_row_start : sub_row_start + block_M // VEC_NUM, col_start : col_start + block_N], up_ub)
                T.tile.mul(up_ub, up_ub, gate_ub)

                T.copy(up_ub, up_logits[sub_row_start : sub_row_start + block_M // VEC_NUM, col_start : col_start + block_N])

    return main


@tilelang.jit(pass_configs=pass_configs_expert)
def moe_shared_down_ascend(num_tokens, dhidden, dexpert, block_M=128, block_N=128, K_L1=64, dtype="float16", accum_dtype="float"):
    """Shared expert: down projection GEMM (Cube only)."""
    m_num = T.ceildiv(num_tokens, block_M)
    n_num = T.ceildiv(dhidden, block_N)
    BLOCKS = m_num * n_num

    @T.prim_func
    def main(
        up_logits: T.Tensor((num_tokens, dexpert), dtype),
        W_down: T.Tensor((dhidden, dexpert), dtype),
        output: T.Tensor((num_tokens, dhidden), dtype),
    ):
        with T.Kernel(BLOCKS, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            with T.Scope("C"):
                A_L1 = T.alloc_L1((block_M, K_L1), dtype)
                B_L1 = T.alloc_L1((block_N, K_L1), dtype)
                C_L0C = T.alloc_L0C((block_M, block_N), accum_dtype)

                loop_k = T.ceildiv(dexpert, K_L1)
                for k in T.serial(loop_k):
                    T.copy(up_logits[bx * block_M : (bx + 1) * block_M, k * K_L1 : (k + 1) * K_L1], A_L1)
                    T.copy(W_down[by * block_N : (by + 1) * block_N, k * K_L1 : (k + 1) * K_L1], B_L1)
                    T.gemm_v0(A_L1, B_L1, C_L0C, transpose_B=True, init=(k == 0))

                T.copy(C_L0C, output[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return main


class Expert(nn.Module):
    def __init__(self, config, gate, up, down, d_expert=None):
        super().__init__()
        self.config = config
        self.d_hidden = config["d_hidden"]
        self.d_expert = config["d_expert"] if d_expert is None else d_expert
        self.device = torch.device("npu")
        self.W_gate_weight = gate.t().contiguous().to(self.device)
        self.W_up_weight = up.t().contiguous().to(self.device)
        self.W_down_weight = down.t().contiguous().to(self.device)


class MoEGate(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.top_k = config["n_experts_per_token"]
        self.num_experts = config["n_routed_experts"]
        self.d_hidden = config["d_hidden"]
        self.W_g_weight = weights["router.weight"].t()

    def forward(self, x):
        logits = x @ self.W_g_weight
        scores = logits.softmax(dim=-1)
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        return topk_indices, topk_scores


class MoE(nn.Module):
    def __init__(self, config, weights, padding_M=128):
        super().__init__()
        self.config = config
        self.padding_M = padding_M
        self.device = torch.device("npu")
        self.dtype_str = "float16"
        self.accum_str = "float"

        self.experts = nn.ModuleList(
            [
                Expert(
                    config,
                    gate=weights[f"experts.{i}.0.weight"],
                    up=weights[f"experts.{i}.1.weight"],
                    down=weights[f"experts.{i}.2.weight"],
                )
                for i in range(config["n_routed_experts"])
            ]
        )
        self.gating_network = MoEGate(config, weights).to(self.device)
        shared_dim = config["d_expert"] * config["n_shared_experts"]
        self.shared_expert = Expert(
            config,
            gate=weights["shared_experts.0.weight"],
            up=weights["shared_experts.1.weight"],
            down=weights["shared_experts.2.weight"],
            d_expert=shared_dim,
        ).to(self.device)

        nt = config["batch_size"] * config["seq_len"]
        self.dexpert = config["d_expert"]
        self.dhidden = config["d_hidden"]
        self.expert_cache = torch.zeros((nt, self.dhidden), dtype=torch.float16, device=self.device)
        self.stacked_expert_tokens = torch.empty(
            (nt * config["n_experts_per_token"], self.dhidden), dtype=torch.float16, device=self.device
        )
        self.stacked_expert_weights = torch.empty((nt * config["n_experts_per_token"],), dtype=torch.float16, device=self.device)
        self.stacked_expert_tokens_idxs = torch.empty((nt * config["n_experts_per_token"],), dtype=torch.int64, device=self.device)
        self.up_logits_shared = torch.empty((nt, self.dexpert), dtype=torch.float16, device=self.device)
        self.expert_output_shared = torch.empty((nt, self.dhidden), dtype=torch.float16, device=self.device)
        self.ws_gate_shared = torch.empty((nt, self.dexpert), dtype=torch.float16, device=self.device)
        self.ws_up_shared = torch.empty((nt, self.dexpert), dtype=torch.float16, device=self.device)

    @torch.no_grad()
    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        BLOCK_M = 128
        expert_indices, expert_scores = self.gating_network(x)
        flat_expert_indices = expert_indices.view(-1)
        flat_expert_weights = expert_scores.view(-1)

        idxs = flat_expert_indices.argsort()
        counts = flat_expert_indices.bincount()
        n_experts = self.config["n_routed_experts"]
        if len(counts) < n_experts:
            counts = torch.nn.functional.pad(counts, (0, n_experts - len(counts)))
        counts_np = counts.cpu().numpy()
        tokens_per_expert = counts_np.cumsum()
        num_per_tok = self.config["n_experts_per_token"]
        token_idxs = idxs // num_per_tok

        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            if start_idx == end_idx:
                continue
            exp_token_idxs = token_idxs[start_idx:end_idx]
            self.stacked_expert_tokens[start_idx:end_idx] = x_flat[exp_token_idxs]
            self.stacked_expert_tokens_idxs[start_idx:end_idx] = exp_token_idxs
            self.stacked_expert_weights[start_idx:end_idx] = flat_expert_weights[idxs[start_idx:end_idx]]

        # Pre-compile kernels for all needed sizes (workaround for JIT re-entrancy)
        pad_sizes = set()
        pad_sizes.add(x_flat.shape[0])  # shared expert
        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            nt = end_idx - start_idx
            if nt > 0:
                pad_sizes.add(((max(nt, 1) + BLOCK_M - 1) // BLOCK_M) * BLOCK_M)
        _precompile_kernels(sorted(pad_sizes), self.dhidden, self.dexpert)

        # Shared expert: compile and run for full num_tokens
        nt = x_flat.shape[0]
        sgk = _get_gate_up_kernel(nt, self.dhidden, self.dexpert, self.dtype_str, self.accum_str)
        sdk = _get_down_kernel(nt, self.dhidden, self.dexpert, self.dtype_str, self.accum_str)
        sgk(
            x_flat,
            self.shared_expert.W_gate_weight,
            self.shared_expert.W_up_weight,
            self.ws_gate_shared,
            self.ws_up_shared,
            self.up_logits_shared,
        )
        sdk(self.up_logits_shared, self.shared_expert.W_down_weight, self.expert_output_shared)

        # Routed experts: compile per-expert with exact padded n_tokens
        self.expert_cache.zero_()
        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            n_tokens = end_idx - start_idx
            if n_tokens == 0:
                continue
            pad_to = ((max(n_tokens, 1) + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
            exp_tokens = self.stacked_expert_tokens[start_idx:end_idx].contiguous()
            exp_wt = self.stacked_expert_weights[start_idx:end_idx]
            exp_tids = self.stacked_expert_tokens_idxs[start_idx:end_idx]
            exp_gate_w = self.experts[expert_id].W_gate_weight
            exp_up_w = self.experts[expert_id].W_up_weight
            exp_down_w = self.experts[expert_id].W_down_weight

            # Compile kernels for this exact padded token count
            rgk = _get_gate_up_kernel(pad_to, self.dhidden, self.dexpert, self.dtype_str, self.accum_str)
            rdk = _get_down_kernel(pad_to, self.dhidden, self.dexpert, self.dtype_str, self.accum_str)

            if n_tokens != pad_to:
                pad_x = torch.cat(
                    [exp_tokens, torch.zeros(pad_to - n_tokens, exp_tokens.shape[1], dtype=torch.float16, device=self.device)]
                )
                pad_wsg = torch.zeros(pad_to, self.dexpert, dtype=torch.float16, device=self.device)
                pad_wsu = torch.zeros(pad_to, self.dexpert, dtype=torch.float16, device=self.device)
                pad_up = torch.zeros(pad_to, self.dexpert, dtype=torch.float16, device=self.device)
                pad_out = torch.zeros(pad_to, self.dhidden, dtype=torch.float16, device=self.device)
                rgk(pad_x, exp_gate_w, exp_up_w, pad_wsg, pad_wsu, pad_up)
                rdk(pad_up, exp_down_w, pad_out)
                out_exp = pad_out[:n_tokens]
            else:
                ws_gate_exp = torch.zeros(pad_to, self.dexpert, dtype=torch.float16, device=self.device)
                ws_up_exp = torch.zeros(pad_to, self.dexpert, dtype=torch.float16, device=self.device)
                up_exp = torch.zeros(pad_to, self.dexpert, dtype=torch.float16, device=self.device)
                out_exp_all = torch.zeros(pad_to, self.dhidden, dtype=torch.float16, device=self.device)
                rgk(exp_tokens, exp_gate_w, exp_up_w, ws_gate_exp, ws_up_exp, up_exp)
                rdk(up_exp, exp_down_w, out_exp_all)
                out_exp = out_exp_all

            out_exp.mul_(exp_wt.view(-1, 1))
            self.expert_cache.scatter_reduce_(0, exp_tids.view(-1, 1).expand(-1, self.dhidden), out_exp, reduce="sum")

        routed_output = self.expert_cache.view(*orig_shape)
        return self.expert_output_shared.view(*orig_shape) + routed_output


def _get_gate_up_kernel(n_tokens, dhidden, dexpert, dtype="float16", accum_dtype="float"):
    """Get or compile a gate+up kernel for the given n_tokens."""
    BM, BN, KL = 128, 128, 64
    # Use int(n_tokens) to ensure clean Python int
    return moe_shared_gate_up_ascend(int(n_tokens), int(dhidden), int(dexpert), BM, BN, KL, dtype, accum_dtype)


def _get_down_kernel(n_tokens, dhidden, dexpert, dtype="float16", accum_dtype="float"):
    """Get or compile a down kernel for the given n_tokens."""
    BM, BN, KL = 128, 128, 64
    return moe_shared_down_ascend(int(n_tokens), int(dhidden), int(dexpert), BM, BN, KL, dtype, accum_dtype)


def _precompile_kernels(n_tokens_list, dhidden, dexpert):
    """Pre-compile kernels for a list of n_tokens values to avoid JIT re-entrancy issues."""
    for nt in sorted(set(n_tokens_list)):
        try:
            _get_gate_up_kernel(nt, dhidden, dexpert)
            _get_down_kernel(nt, dhidden, dexpert)
        except Exception:
            pass  # Ignore pre-compilation failures, let it fail at runtime


def custom_kernel(data):
    input_tensor, weights, config = data
    moe = MoE(config, weights, padding_M=128)
    return moe(input_tensor)


def main(d_hidden=7168, d_expert=2048, n_routed_experts=8, n_shared_experts=1, n_experts_per_token=4, batch_size=1, seq_len=8192):
    config = {
        "dhidden": d_hidden,
        "dexpert": d_expert,
        "nroutedexperts": n_routed_experts,
        "nsharedexperts": n_shared_experts,
        "nexpertspertoken": n_experts_per_token,
        "bs": batch_size,
        "seqlen": seq_len,
        "seed": 81394,
    }

    data = generate_input(**config)

    def clone(d):
        if isinstance(d, tuple):
            return tuple(clone(v) for v in d)
        if isinstance(d, list):
            return [clone(v) for v in d]
        if isinstance(d, dict):
            return {k: clone(v) for k, v in d.items()}
        if isinstance(d, torch.Tensor):
            return d.clone()
        return d

    ref_output = ref_kernel(clone_data(data)).float()
    # ref_output = ref_kernel_npu(clone(data)).float()

    out = custom_kernel(clone(data)).float()
    torch.npu.synchronize()

    torch.testing.assert_close(ref_output, out, atol=1e-2, rtol=1e-2)
    print("TileLang Ascend and Torch match!")
    print("Test passed!")


if __name__ == "__main__":
    tilelang.disable_cache()
    torch.set_default_device("npu")
    main()
