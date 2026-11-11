"""Seven-input adaptation of pinned vLLM-Ascend's ordinary unquantized decode.

Uses real LP torch.bmm, FIA v2 BNSD_NBSD and npu_transpose_batchmatmul.
The default gate is explicitly a Torch LP-rounding expression, not the fused
official Triton gate. No API/shape failure silently falls back to another route.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


SOURCE_COMMIT = "e4d533d33211f0696bcdf65eda47245d52e428c5"
SOURCE_URL = f"https://github.com/vllm-project/vllm-ascend/blob/{SOURCE_COMMIT}/vllm_ascend/attention/mla_v1.py"
GATE_URL = f"https://github.com/vllm-project/vllm-ascend/blob/{SOURCE_COMMIT}/vllm_ascend/ops/triton/flash_attention_output.py"
NAMES = ("q_content", "q_aux", "latent_kv", "key_aux", "w_k", "w_v", "gate_logits")


@dataclass(frozen=True)
class DecodePlan:
    batch: int
    heads: int
    padded_heads: int
    capacity: int
    visible: tuple
    q_lens: tuple
    kv_lens: tuple
    query_start: tuple
    scale: float
    block_size: int = 128


def decode_plan(inputs, metadata):
    if set(inputs) != set(NAMES):
        raise ValueError("exact original seven inputs are required")
    q = inputs["q_content"]
    if not isinstance(q, torch.Tensor) or q.ndim != 4:
        raise TypeError("q_content must be [B,1,H,128]")
    B, Q, H, D = map(int, q.shape)
    c = inputs["latent_kv"]
    if not isinstance(c, torch.Tensor) or c.ndim != 3:
        raise TypeError("latent_kv must be [B,Tk,512]")
    K = int(c.shape[1])
    if B < 1 or H < 1 or Q != 1 or D != 128 or K < 128 or K % 128:
        raise NotImplementedError("first source adapter requires Tq1, positive B/H, and physical Tk a positive multiple of128")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("only unquantized FP16/BF16 are implemented")
    shapes = dict(
        q_content=(B, 1, H, 128),
        q_aux=(B, 1, H, 64),
        latent_kv=(B, K, 512),
        key_aux=(B, K, 64),
        w_k=(H, 128, 512),
        w_v=(H, 128, 512),
        gate_logits=(B, 1, H, 128),
    )
    for name, shape in shapes.items():
        value = inputs[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} must be {shape}")
        if value.dtype != q.dtype or value.device != q.device or not value.is_contiguous():
            raise ValueError(f"{name} must share query dtype/device and be logically contiguous")
    fields = []
    for name in ("q_lens", "kv_lens", "query_start"):
        values = metadata[name]
        if not isinstance(values, (tuple, list)) or len(values) != B or any(type(v) is not int for v in values):
            raise TypeError("lengths/start must be CPU-owned Python integer sequences, not device tensors")
        fields.append(tuple(values))
    ql, kl, start = fields
    if any(q != 1 or not (1 <= k <= K and 0 <= s < k) for q, k, s in zip(ql, kl, start)):
        raise NotImplementedError(
            "this ordinary-decode extraction requires one active query per batch; inactive/ragged packing is not hidden"
        )
    scale = metadata.get("scale", 1 / math.sqrt(192))
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive host number")
    return DecodePlan(B, H, 1 << (H - 1).bit_length(), K, tuple(min(k, s + 1) for k, s in zip(kl, start)), ql, kl, start, float(scale))


def prepare_constants(inputs, plan):
    """Explicit immutable benchmark setup. Wv transpose-copy is weight preparation."""
    pages = plan.capacity // plan.block_size
    table = torch.arange(plan.batch * pages, dtype=torch.int32, device="cpu").view(plan.batch, pages).to(inputs["q_content"].device)
    return {
        "block_table": table,
        "cache": inputs["latent_kv"].view(plan.batch * pages, 1, plan.block_size, 512),
        "cache_aux": inputs["key_aux"].view(plan.batch * pages, 1, plan.block_size, 64),
        "w_uv": inputs["w_v"].transpose(1, 2).contiguous(),
    }


def torch_lp_gate(projected, gate):
    """LP sigmoid result, LP projected operand, in-place LP product; not fused."""
    if projected.dtype != gate.dtype or projected.shape != gate.shape:
        raise ValueError("projected and gate must have identical shape/dtype")
    return projected.mul_(torch.sigmoid(gate))


def decode_call(inputs, plan, constants, api, gate):
    # Source _q_proj_and_k_up_proj, after the externally supplied q_b projection.
    q = inputs["q_content"].view(plan.batch, plan.heads, 128).transpose(0, 1)
    ql = torch.bmm(q, inputs["w_k"]).transpose(0, 1)
    ql = ql.view(plan.batch, plan.heads, 1, 512).contiguous()
    qa = inputs["q_aux"].view(plan.batch, plan.heads, 1, 64)
    padding = plan.padded_heads - plan.heads
    if padding:
        ql = F.pad(ql, (0, 0, 0, 0, 0, padding), "constant", 0)
        qa = F.pad(qa, (0, 0, 0, 0, 0, padding), "constant", 0)
    latent, _ = api.npu_fused_infer_attention_score_v2(
        ql,
        constants["cache"],
        constants["cache"],
        query_rope=qa,
        key_rope=constants["cache_aux"],
        num_query_heads=plan.padded_heads,
        num_key_value_heads=1,
        input_layout="BNSD_NBSD",
        atten_mask=None,
        sparse_mode=0,
        softmax_scale=plan.scale,
        block_table=constants["block_table"],
        block_size=plan.block_size,
        actual_seq_qlen=None,
        actual_seq_kvlen=list(plan.visible),
    )
    if tuple(latent.shape) != (plan.padded_heads, plan.batch, 1, 512) or latent.dtype != inputs["q_content"].dtype:
        raise RuntimeError("FIA returned an unexpected source-path output layout/dtype; no alternate layout fallback")
    latent = latent[: plan.heads].view(plan.heads, plan.batch, 512)
    # Exact source _v_up_proj primitive/permutation; no FP32 replacement.
    projected = api.npu_transpose_batchmatmul(latent, constants["w_uv"], perm_y=(1, 0, 2))
    if projected.dtype != inputs["q_content"].dtype or tuple(projected.shape) != (plan.batch, plan.heads, 128):
        raise RuntimeError("transpose BMM output does not match the unquantized source contract")
    projected = projected.reshape(plan.batch, plan.heads * 128)
    result = gate(projected, inputs["gate_logits"].view(plan.batch, plan.heads * 128))
    if result.dtype != inputs["q_content"].dtype or result.shape != projected.shape:
        raise RuntimeError("gate output contract mismatch")
    return result.view(plan.batch, 1, plan.heads, 128)


class VllmSourceDecode:
    """Bound, source-derived seven-input baseline for coordinator-owned NPU tests."""

    def __init__(self, inputs, metadata, *, api=None, gate_impl=None, gate_name=None):
        self.plan = decode_plan(inputs, metadata)
        if inputs["q_content"].device.type not in ("npu", "privateuseone"):
            raise ValueError("real adapter requires NPU inputs; CPU contract tests call decode_call with an explicit fake API")
        if api is None:
            import torch_npu

            api = torch_npu
        for name in ("npu_fused_infer_attention_score_v2", "npu_transpose_batchmatmul"):
            if not callable(getattr(api, name, None)):
                raise RuntimeError(f"required real official API unavailable: {name}")
        if torch.npu.current_device() != inputs["q_content"].device.index:
            raise ValueError("select the input's device before creating the baseline; this adapter never switches devices")
        self.device_name = torch.npu.get_device_name(inputs["q_content"].device)
        if "910B" not in self.device_name.upper():
            raise NotImplementedError("this source adapter is scoped to the current 910B/A2 benchmark; no950 FlashMLA substitution")
        if gate_impl is not None and not gate_name:
            raise ValueError("an explicit gate callable requires an honest provenance/name")
        for name, tensor in inputs.items():
            fmt = api.get_npu_format(tensor)
            if fmt not in (api.Format.ND, api.Format.NCHW):
                raise NotImplementedError(f"{name} requires the selected non-NZ base-format path; no format conversion fallback")
        self.inputs, self.api = inputs, api
        self.gate = torch_lp_gate if gate_impl is None else gate_impl
        self.gate_mode = "torch_rounding" if gate_impl is None else "injected_callable"
        self.gate_name = "torch_lp_rounding_expression_not_fused" if gate_impl is None else gate_name
        self.constants = prepare_constants(inputs, self.plan)

    def __call__(self):
        return decode_call(self.inputs, self.plan, self.constants, self.api, self.gate)

    def describe(self):
        return {
            "name": "vllm_source_decode_nd_hpad_lp_bmm_transpose_bmm_" + self.gate_name,
            "runtime": {
                "torch": str(torch.__version__),
                "torch_npu": str(getattr(self.api, "__version__", "unknown")),
                "torch_npu_file": str(getattr(self.api, "__file__", "unknown")),
            },
            "source_commit": SOURCE_COMMIT,
            "source_url": SOURCE_URL,
            "gate_source_url": GATE_URL,
            "scope": "seven inputs after q_b projection/latent norm, before W_O; no cache append or external projections",
            "source_configuration": {
                "unquantized": True,
                "TP": 1,
                "CP": 1,
                "speculative": False,
                "enable_kv_nz": False,
                "weight_nz_mode": 1,
                "flash_mla": False,
                "block_size": 128,
            },
            "logical_heads": self.plan.heads,
            "FIA_physical_heads": self.plan.padded_heads,
            "visible": list(self.plan.visible),
            "original_kv_lens": list(self.plan.kv_lens),
            "query_start": list(self.plan.query_start),
            "prefix_adapter_used": any(k != v for k, v in zip(self.plan.kv_lens, self.plan.visible)),
            "rounding": "Q absorption output LP; FIA output LP; Wv output LP; sigmoid output LP; product LP",
            "gate_mode": self.gate_mode,
            "gate": self.gate_name,
            "source_layout_adaptation": "seven-input contiguous Q_content/QA replace upstream packed Q192 views; no hidden repacking",
            "preparation_excluded": [
                "CPU-owned effective-prefix metadata",
                "identity block table H2D",
                "zero-copy PA128 cache views",
                "one-time Wv transpose-contiguous to [H,512,128] (static weights)",
            ],
            "per_call_included": [
                "LP Q bmm",
                "query contiguous/power-of-two head padding",
                "FIA v2",
                "discard padded heads/view",
                "real npu_transpose_batchmatmul",
                "explicit gate callable",
            ],
            "fixed_input_note": "callable binds supplied tensors; static weight preparation must be recreated after weight updates",
            "hardware_correctness_and_performance": "not established by source inspection; validate original FP64 before timing",
        }
