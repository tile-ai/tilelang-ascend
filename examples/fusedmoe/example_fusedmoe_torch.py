import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


# Reference code in PyTorch
class ExpertTorch(nn.Module):
    def __init__(self, config: Dict, d_expert: Optional[int] = None):
        super().__init__()
        self.config = config
        self.act_fn = nn.SiLU()
        self.d_hidden: int = config["d_hidden"]
        self.d_expert: int = config["d_expert"] if d_expert is None else d_expert

        self.W_gate = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_up = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_down = nn.Linear(self.d_expert, self.d_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.W_gate(x))
        out = self.W_down(gate * self.W_up(x))
        return out


class MoEGateTorch(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.top_k: int = config["n_experts_per_token"]
        self.num_experts: int = config["n_routed_experts"]
        self.d_hidden: int = config["d_hidden"]

        self.W_g = nn.Linear(self.d_hidden, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.W_g(x)
        scores = logits.softmax(dim=-1)
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        return topk_indices, topk_scores


class MoETorch(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([ExpertTorch(config) for _ in range(config["n_routed_experts"])])
        self.gating_network = MoEGateTorch(config)
        shared_expert_dim = config["d_expert"] * config["n_shared_experts"]
        self.shared_expert = ExpertTorch(config=config, d_expert=shared_expert_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_output = self.shared_expert(x)
        expert_indices, expert_scores = self.gating_network(x)
        batch_size, seq_len, hidden_dim = x.shape
        orig_shape = x.shape
        x_flat = x.view(-1, hidden_dim)
        flat_expert_indices = expert_indices.view(-1)
        flat_expert_weights = expert_scores.view(-1, 1)
        routed_output_flat = self.moe_infer(x_flat, flat_expert_indices, flat_expert_weights)

        routed_output = routed_output_flat.view(*orig_shape)
        return routed_output + shared_output

    @torch.no_grad()
    def moe_infer(self, x: torch.Tensor, flat_expert_indices: torch.Tensor, flat_expert_weights: torch.Tensor) -> torch.Tensor:
        expert_cache = torch.zeros_like(x)
        # test_expert_cache = torch.zeros((x.shape[0] * self.config["n_experts_per_token"], self.config["d_hidden"]))
        # test_expert_tokens = torch.zeros((x.shape[0] * self.config["n_experts_per_token"], self.config["d_hidden"]))
        # test_expert_ups = torch.zeros((self.config["n_routed_experts"], self.config["d_hidden"], self.config["d_expert"]))
        # test_expert_tokens_num = torch.zeros((self.config["n_routed_experts"]))

        idxs = flat_expert_indices.argsort()
        counts = flat_expert_indices.bincount().cpu().numpy()
        tokens_per_expert = counts.cumsum()
        num_per_tok = self.config["n_experts_per_token"]
        token_idxs = idxs // num_per_tok
        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            if start_idx == end_idx:
                continue

            expert = self.experts[expert_id]
            exp_token_idxs = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idxs]
            expert_out = expert(expert_tokens)

            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(0, exp_token_idxs.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce="sum")

        return expert_cache


# Reference implementation of the entire MoE module using PyTorch
def ref_kernel(data: Tuple[torch.Tensor, Dict, Dict]) -> torch.Tensor:
    """
    Reference implementation of DeepSeek-style Mixture of Experts using PyTorch.

    Args:
        data: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, hidden_dim]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters

    Returns:
        Tuple containing:
            - output: Processed tensor [batch_size, seq_len, d_model]
    """
    input_tensor, weights, config = data
    num_experts = config["n_routed_experts"]
    moe = MoETorch(config)

    # Fill in the given weights of the model
    moe.gating_network.W_g.weight = nn.Parameter(weights["router.weight"])

    for i in range(num_experts):
        gate_proj_weight = weights[f"experts.{i}.0.weight"]
        up_proj_weight = weights[f"experts.{i}.1.weight"]
        down_proj_weight = weights[f"experts.{i}.2.weight"]

        # Transpose weights to match expected shape for nn.Linear
        moe.experts[i].W_gate.weight = nn.Parameter(gate_proj_weight.t())
        moe.experts[i].W_up.weight = nn.Parameter(up_proj_weight.t())
        moe.experts[i].W_down.weight = nn.Parameter(down_proj_weight.t())

    moe.shared_expert.W_gate.weight = nn.Parameter(weights["shared_experts.0.weight"].t())
    moe.shared_expert.W_up.weight = nn.Parameter(weights["shared_experts.1.weight"].t())
    moe.shared_expert.W_down.weight = nn.Parameter(weights["shared_experts.2.weight"].t())

    output = moe(input_tensor)

    return output


# Reference implementation of the entire MoE module using PyTorch, with more direct tensor operations to mirror the fused kernel logic.
def ref_kernel_npu(data):
    """NPU reference using nn.Linear semantics: weight.T is [d_out, d_in].
    x @ weight = [N, d_in] @ [d_in, d_out] = [N, d_out]
    gemm_v0 with transpose_B=True: x @ W where W loaded as [d_out, K] (K-tiled d_in)."""
    input_tensor, weights, config = data
    ne = config["n_routed_experts"]
    orig = input_tensor.shape
    xf = input_tensor.view(-1, orig[-1])
    Wr = weights["router.weight"].t()
    logits = xf @ Wr
    scores = logits.softmax(-1)
    tk, ti = torch.topk(scores, k=config["n_experts_per_token"], dim=-1, sorted=False)

    sd = config["d_expert"] * config["n_shared_experts"]
    # Weight format in kernel: [dexpert, dhidden]
    # gemm_v0(x [M, dhidden], W [dexpert, dhidden], C, transpose_B=True):
    #   C[M, dexpert] = x @ W^T = x[M, dhidden] @ [dhidden, dexpert]
    # So reference: x @ weight_original = x @ [dhidden, dexpert]
    wsg = weights["shared_experts.0.weight"]  # [dhidden, sd]
    wsu = weights["shared_experts.1.weight"]  # [dhidden, sd]
    wsd = weights["shared_experts.2.weight"]  # [sd, dhidden]
    s_out = (torch.nn.functional.silu(xf @ wsg) * (xf @ wsu)) @ wsd

    ecache = torch.zeros_like(xf)
    fti = ti.view(-1)
    ftw = tk.view(-1, 1)
    idxs = fti.argsort()
    counts = fti.bincount(minlength=ne).cpu().numpy()
    tpe = counts.cumsum()
    npt = config["n_experts_per_token"]
    tids = idxs // npt

    for eid, ei in enumerate(tpe):
        si = 0 if eid == 0 else tpe[eid - 1]
        if si == ei:
            continue
        etids = tids[si:ei]
        etoks = xf[etids]
        wg = weights[f"experts.{eid}.0.weight"]  # [dhidden, dexpert]
        wu = weights[f"experts.{eid}.1.weight"]  # [dhidden, dexpert]
        wd = weights[f"experts.{eid}.2.weight"]  # [dexpert, dhidden]
        eo = (torch.nn.functional.silu(etoks @ wg) * (etoks @ wu)) @ wd
        eo.mul_(ftw[idxs[si:ei]])
        ecache.scatter_reduce_(0, etids.view(-1, 1).expand(-1, xf.shape[-1]), eo, reduce="sum")

    return (s_out + ecache).view(*orig)


# Input generation for the reference code
def generate_input(
    dhidden: int, dexpert: int, nroutedexperts: int, nsharedexperts: int, nexpertspertoken: int, bs: int, seqlen: int, seed: int
) -> Tuple[torch.Tensor, Dict, Dict]:
    # Really dumb but for now _ isn't parsing correctly.
    d_hidden = dhidden
    d_expert = dexpert
    n_routed_experts = nroutedexperts
    n_shared_experts = nsharedexperts
    n_experts_per_token = nexpertspertoken
    batch_size = bs
    seq_len = seqlen

    config = {
        "d_hidden": d_hidden,
        "d_expert": d_expert,
        "n_routed_experts": n_routed_experts,
        "n_shared_experts": n_shared_experts,
        "n_experts_per_token": n_experts_per_token,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }

    gen = torch.Generator(device="npu")
    gen.manual_seed(seed)

    num_experts = n_routed_experts
    expert_dim = d_expert
    weights = {}

    input_tensor = torch.randn((batch_size, seq_len, d_hidden), device="npu", dtype=torch.float16, generator=gen).contiguous()

    # Initialize router weights
    weights["router.weight"] = torch.randn((num_experts, d_hidden), device="npu", dtype=torch.float16, generator=gen) / math.sqrt(d_hidden)

    for i in range(num_experts):
        weights[f"experts.{i}.0.weight"] = torch.randn(
            (d_hidden, expert_dim), device="npu", dtype=torch.float16, generator=gen
        ) / math.sqrt(expert_dim)

        weights[f"experts.{i}.1.weight"] = torch.randn(
            (d_hidden, expert_dim), device="npu", dtype=torch.float16, generator=gen
        ) / math.sqrt(expert_dim)

        weights[f"experts.{i}.2.weight"] = torch.randn(
            (expert_dim, d_hidden), device="npu", dtype=torch.float16, generator=gen
        ) / math.sqrt(d_hidden)

    weights["shared_experts.0.weight"] = torch.randn(
        (d_hidden, expert_dim * n_shared_experts), device="npu", dtype=torch.float16, generator=gen
    ) / math.sqrt(expert_dim * n_shared_experts)
    weights["shared_experts.1.weight"] = torch.randn(
        (d_hidden, expert_dim * n_shared_experts), device="npu", dtype=torch.float16, generator=gen
    ) / math.sqrt(expert_dim * n_shared_experts)
    weights["shared_experts.2.weight"] = torch.randn(
        (expert_dim * n_shared_experts, d_hidden), device="npu", dtype=torch.float16, generator=gen
    ) / math.sqrt(d_hidden)

    return (input_tensor, weights, config)


def clone_data(data):
    """
    Recursively goes through data and clones all tensors.
    """
    if isinstance(data, tuple):
        return tuple(clone_data(x) for x in data)
    elif isinstance(data, list):
        return [clone_data(x) for x in data]
    elif isinstance(data, dict):
        return {k: clone_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return data
