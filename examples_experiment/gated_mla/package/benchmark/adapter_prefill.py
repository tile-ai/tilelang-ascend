"""Pinned vLLM ordinary-prefill operator path, adapted to seven inputs.

Only B1/H96/full-prefill is currently exposed. It preserves one LP KV linear,
the explicit FP16->BF16 FIA conversion, the old FIA API and LP gate rounding.
No two-BMM substitute, quantized route or continuation/LSE-merge fallback.
"""

from dataclasses import dataclass
import math

import torch

from adapter_decode import GATE_URL, SOURCE_COMMIT, SOURCE_URL, torch_lp_gate
from source_unquantized_gemm import unquantized_gemm


@dataclass(frozen=True)
class PrefillPlan:
    tokens: int
    scale: float
    heads: int = 96


def prefill_plan(inputs, metadata):
    q = inputs["q_content"]
    if not isinstance(q, torch.Tensor) or q.ndim != 4:
        raise TypeError("q_content must be [1,Tq,96,128]")
    B, Q, H, D = map(int, q.shape)
    if (B, H, D) != (1, 96, 128) or Q < 1:
        raise NotImplementedError("initial source prefill adapter supports B1/H96 and positive Tq")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("unquantized FP16/BF16 only")
    shapes = dict(
        q_content=(1, Q, 96, 128),
        q_aux=(1, Q, 96, 64),
        latent_kv=(1, Q, 512),
        key_aux=(1, Q, 64),
        w_k=(96, 128, 512),
        w_v=(96, 128, 512),
        gate_logits=(1, Q, 96, 128),
    )
    if set(inputs) != set(shapes):
        raise ValueError("exact original seven inputs required")
    for name, shape in shapes.items():
        t = inputs[name]
        if not isinstance(t, torch.Tensor) or tuple(t.shape) != shape:
            raise NotImplementedError(f"{name} must be {shape}; history/chunked prefill is not substituted")
        if t.dtype != q.dtype or t.device != q.device or not t.is_contiguous():
            raise ValueError(f"{name} must share dtype/device and be contiguous")
    for name, expected in (("q_lens", Q), ("kv_lens", Q), ("query_start", 0)):
        values = metadata[name]
        if not isinstance(values, (tuple, list)) or len(values) != 1 or type(values[0]) is not int:
            raise TypeError("metadata must be CPU-owned Python integer sequences")
        if values[0] != expected:
            raise NotImplementedError("initial adapter requires complete prefill, all active, Tq=Tk, start0; no historical context")
    scale = metadata.get("scale", 1 / math.sqrt(192))
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite positive host number")
    return PrefillPlan(Q, float(scale))


def prepare_prefill_constants(inputs, plan):
    # Restore exactly the loaded kv_b_proj.weight row order [head,K128+V128,R512].
    weight = torch.cat((inputs["w_k"], inputs["w_v"]), dim=1).view(plan.heads * 256, 512)
    # Exact source AttentionMaskBuilder.get_splitfuse_attn_mask dtype/values.
    mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(inputs["q_content"].device)
    return {"packed_w_kv": weight, "causal_mask": mask}


def prefill_call(inputs, plan, constants, api, gate):
    original_dtype = inputs["q_content"].dtype
    q = inputs["q_content"].view(plan.tokens, plan.heads, 128)
    qa = inputs["q_aux"].view(plan.tokens, plan.heads, 64)
    # Actual registered unquantized_gemm function body, preserved AST: one LP linear.
    expanded = unquantized_gemm(inputs["latent_kv"].view(plan.tokens, 512), constants["packed_w_kv"], None)
    if expanded.dtype != original_dtype or tuple(expanded.shape) != (plan.tokens, plan.heads * 256):
        raise RuntimeError("LP fused KV expansion output contract changed")
    k, v = expanded.view(plan.tokens, plan.heads, 256).split([128, 128], dim=-1)
    ka = inputs["key_aux"].view(plan.tokens, 1, 64).expand(plan.tokens, plan.heads, 64)
    # The pinned source explicitly converts all five operands, even for FP16.
    if original_dtype != torch.bfloat16:
        q, qa, k, ka, v = (x.to(torch.bfloat16) for x in (q, qa, k, ka, v))
    # H96 has head_padding>0 in the source, selecting merged192 Q/K here.
    # Unlike decode, ordinary prefill does not pad the head count to128.
    query = torch.cat((q, qa), dim=-1)
    key = torch.cat((k, ka), dim=-1)
    output, _ = api.npu_fused_infer_attention_score(
        query,
        key.contiguous(),
        v.contiguous(),
        num_heads=plan.heads,
        num_key_value_heads=plan.heads,
        input_layout="TND",
        atten_mask=constants["causal_mask"],
        sparse_mode=3,
        scale=plan.scale,
        antiquant_mode=0,
        antiquant_scale=None,
        block_table=None,
        block_size=0,
        softmax_lse_flag=True,
        actual_seq_lengths=[plan.tokens],
        actual_seq_lengths_kv=[plan.tokens],
    )
    if output.dtype != torch.bfloat16 or tuple(output.shape) != (plan.tokens, plan.heads, 128):
        raise RuntimeError("legacy FIA output layout/dtype differs from source prefill; no fallback")
    output = output.reshape(plan.tokens, plan.heads * 128)
    if original_dtype != torch.bfloat16:
        output = output.to(original_dtype)
    output = gate(output, inputs["gate_logits"].view(plan.tokens, plan.heads * 128))
    if output.dtype != original_dtype:
        raise RuntimeError("gate did not preserve original output dtype")
    return output.view(1, plan.tokens, plan.heads, 128)


class VllmSourcePrefill:
    def __init__(self, inputs, metadata, *, api=None, gate_impl=None, gate_name=None):
        self.plan = prefill_plan(inputs, metadata)
        if inputs["q_content"].device.type not in ("npu", "privateuseone"):
            raise ValueError("real adapter requires NPU inputs; CPU tests call prefill_call with explicit fake API")
        if api is None:
            import torch_npu

            api = torch_npu
        if not callable(getattr(api, "npu_fused_infer_attention_score", None)):
            raise RuntimeError("source prefill requires the old FIA API; no v2 replacement")
        if torch.npu.current_device() != inputs["q_content"].device.index:
            raise ValueError("current NPU must match the inputs; no implicit device switch")
        if "910B" not in torch.npu.get_device_name(inputs["q_content"].device).upper():
            raise NotImplementedError("current common comparison domain is910B3/A2; no950-only substitution")
        for name, tensor in inputs.items():
            if api.get_npu_format(tensor) not in (api.Format.ND, api.Format.NCHW):
                raise NotImplementedError(f"{name} requires the selected non-NZ base-format path")
        if gate_impl is not None and not gate_name:
            raise ValueError("an injected gate requires an explicit provenance label")
        self.inputs, self.api = inputs, api
        self.gate = torch_lp_gate if gate_impl is None else gate_impl
        self.gate_mode = "torch_rounding" if gate_impl is None else "injected_callable"
        self.gate_name = "torch_lp_rounding_expression_not_fused" if gate_impl is None else gate_name
        self.constants = prepare_prefill_constants(inputs, self.plan)

    def __call__(self):
        return prefill_call(self.inputs, self.plan, self.constants, self.api, self.gate)

    def describe(self):
        return {
            "name": "vllm_source_prefill_lp_kv_linear_legacy_fia_" + self.gate_name,
            "source_commit": SOURCE_COMMIT,
            "source_url": SOURCE_URL,
            "gate_source_url": GATE_URL,
            "scope": "B1/H96 full prefill at start0, original seven inputs, before W_O; no chunked historical context",
            "source_configuration": "unquantized, TP/CP1, default weight_nz_mode1 for LP, no FlashMLA",
            "input_dtype": str(self.inputs["q_content"].dtype),
            "FIA_dtype": "torch.bfloat16",
            "FIA_layout": "TND",
            "FIA_heads": 96,
            "FIA_value_width": 128,
            "mask_dtype": "torch.int8",
            "scale": self.plan.scale,
            "gate_mode": self.gate_mode,
            "gate": self.gate_name,
            "linear": "exact AST of vllm_ascend.ops.linear.unquantized_gemm: one torch.nn.functional.linear; no two-BMM/FP32 substitute",
            "host_boundary": "call original registered-op body directly, without vLLM custom-op dispatcher/TP wrapper overhead",
            "preparation_excluded": ["static KV weight concatenation once into[H*256,512]", "static2048 int8 causal mask H2D"],
            "per_call_included": [
                "LP KV linear",
                "K/V split views and auxiliary expansion",
                "FP16 five-operand BF16 casts if needed",
                "Q/K concat192 and required contiguity",
                "old FIA with LSE flag",
                "output original-dtype restoration",
                "gate",
            ],
            "rounding": "KV output LP; FP16 operands then BF16 FIA; FIA output BF16; original-dtype output; LP sigmoid/product",
            "runtime": {"torch": str(torch.__version__), "torch_npu": str(getattr(self.api, "__version__", "unknown"))},
            "qualification": "not established by source inspection; original FP64 budget must pass before comparative timing",
        }
