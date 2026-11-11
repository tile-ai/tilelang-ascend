"""Execute the real pinned KimiMLAAttention class for a one-layer fixture.

Exact class/function ASTs are loaded without importing the full transformer stack.
Reference-only dense arithmetic below is never used by the NPU adapter runtime.
"""

import ast
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

SOURCE_REVISION = "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
SOURCE_SHA256 = "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a"


@dataclass(frozen=True)
class FrozenDefinitions:
    attention_class: type
    eager_attention: object
    source_path: str
    source_sha256: str
    ast_sha256: dict


@lru_cache(maxsize=4)
def load_frozen_layer(source_path):
    path = Path(source_path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("frozen Kimi source SHA256 mismatch")
    parsed = ast.parse(data.decode("utf-8"), filename=str(path))
    wanted = {"KimiRMSNorm", "repeat_kv", "eager_attention_forward", "KimiMLAAttention"}
    nodes = [n for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise ValueError("the pinned official layer definitions are missing")
    hashes = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest() for n in nodes}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    selected = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    module = ModuleType("gmla_frozen_kimi_layer_" + SOURCE_REVISION[:8])
    module.__file__ = str(path)
    module.__dict__.update(torch=torch, nn=nn, F=F, ALL_ATTENTION_FUNCTIONS={})
    sys.modules[module.__name__] = module
    exec(compile(selected, str(path), "exec"), module.__dict__)
    return FrozenDefinitions(module.KimiMLAAttention, module.eager_attention_forward, str(path), SOURCE_SHA256, hashes)


def small_config(heads=3, hidden_size=256, q_rank=64):
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        attention_dropout=0.0,
        q_lora_rank=q_rank,
        qk_rope_head_dim=64,
        kv_lora_rank=512,
        v_head_dim=128,
        qk_nope_head_dim=128,
        mla_use_nope=True,
        mla_use_output_gate=True,
        _attn_implementation="eager",
    )


def validate_frozen_layer(layer):
    if layer.training:
        raise ValueError("the adapter is inference-only; the frozen layer must be in eval mode")
    if not layer.use_nope or not layer.use_output_gate:
        raise ValueError("the adapter requires the frozen NoPE and output-gate semantics")
    if (layer.qk_nope_head_dim, layer.qk_rope_head_dim, layer.kv_lora_rank, layer.v_head_dim) != (128, 64, 512, 128):
        raise ValueError("the adapter requires D/E/R/V128/64/512/128")
    if layer.num_key_value_groups != 1:
        raise ValueError("the fixture requires expanded per-head K/V from the original kv_b projection")


def prepare_external_projections(layer, hidden_states, *, guard=None):
    """Original layer modules outside the TileLang core, with explicit cost boundary."""
    validate_frozen_layer(layer)
    if hidden_states.device.type in ("npu", "privateuseone") and guard is None:
        raise ValueError("NPU external preparation requires the physical layout guard")
    if guard is not None:
        guard.require_dense_base(hidden_states, name="hidden_states")
    if layer.q_lora_rank is not None:
        query = layer.q_b_proj(layer.q_a_layernorm(layer.q_a_proj(hidden_states)))
    else:
        query = layer.q_proj(hidden_states)
    raw_kv = layer.kv_a_proj_with_mqa(hidden_states)
    if guard is not None:
        guard.require_dense_base(query, name="q_projection")
        guard.require_dense_base(raw_kv, name="raw_kv_projection")
    c_input = raw_kv[..., :512]
    if c_input.untyped_storage().data_ptr() != raw_kv.untyped_storage().data_ptr():
        raise AssertionError("the raw C slice must remain a view before the original normalization")
    normalized_c = layer.kv_a_layernorm(c_input)
    gate_logits = layer.g_proj(hidden_states)
    if guard is not None:
        guard.require_dense_base(normalized_c, name="normalized_c")
        guard.require_dense_base(gate_logits, name="gate_logits")
    return query, layer.kv_b_proj.weight, normalized_c, raw_kv, gate_logits


def causal_mask(batch, query_tokens, key_tokens, start, *, dtype, device):
    # Reference/fixture only. Runtime Cube uses metadata, not this materialized mask.
    query_positions = torch.arange(query_tokens, device=device) + start
    keys = torch.arange(key_tokens, device=device)
    mask = torch.zeros(batch, 1, query_tokens, key_tokens, dtype=dtype, device=device)
    return mask.masked_fill(keys[None, None, None, :] > query_positions[None, None, :, None], -torch.inf)


class MinimalExpandedKVCache:
    """Test-only cache implementing the exact update hook used by the frozen layer.

    This is not transformers.DynamicCache or the candidate compressed cache.
    It appends already-expanded K/V on their sequence dimension without recomputing
    prior projections. One layer, uniform batch lengths, no eviction/beam reorder.
    """

    def __init__(self):
        self.key = None
        self.value = None
        self.layer_idx = None
        self.update_calls = []

    def get_seq_length(self, layer_idx=0):
        if self.layer_idx is not None and self.layer_idx != layer_idx:
            raise ValueError("this fixture cache belongs to one frozen attention layer")
        return 0 if self.key is None else int(self.key.shape[-2])

    def update(self, key_states, value_states, layer_idx):
        old_length = self.get_seq_length(layer_idx)
        if key_states.ndim != 4 or value_states.ndim != 4 or key_states.shape[:3] != value_states.shape[:3]:
            raise ValueError("official expanded K/V must be [B,H,T,D] with matching B/H/T")
        if key_states.dtype != value_states.dtype or key_states.device != value_states.device:
            raise ValueError("official expanded K/V dtype and device must match")
        if self.key is None:
            self.key, self.value = key_states, value_states
            self.layer_idx = layer_idx
        else:
            if self.key.dtype != key_states.dtype or self.key.device != key_states.device:
                raise ValueError("expanded cache dtype/device changed")
            if self.key.shape[:2] != key_states.shape[:2] or self.key.shape[-1] != key_states.shape[-1]:
                raise ValueError("expanded key cache geometry changed")
            if self.value.shape[-1] != value_states.shape[-1]:
                raise ValueError("expanded value cache geometry changed")
            self.key = torch.cat((self.key, key_states), dim=-2)
            self.value = torch.cat((self.value, value_states), dim=-2)
        self.update_calls.append({"layer_idx": layer_idx, "old_length": old_length, "new_tokens": int(key_states.shape[-2])})
        return self.key, self.value


def actual_dense_layer_forward(layer, hidden_states, *, past_key_values=None, capture_external=None):
    """Run untouched class.forward; optional hooks capture its real external inputs."""
    batch, tokens, _ = hidden_states.shape
    captured = {}

    def capture(_module, args):
        captured["pre_wo"] = args[0].detach()

    handles = [layer.o_proj.register_forward_pre_hook(capture)]
    if capture_external is not None:

        def save_output(name):
            def hook(_module, _args, result):
                capture_external[name] = result.detach()

            return hook

        query_module = layer.q_b_proj if layer.q_lora_rank is not None else layer.q_proj
        for name, module in (
            ("q_projection", query_module),
            ("raw_kv", layer.kv_a_proj_with_mqa),
            ("normalized_c", layer.kv_a_layernorm),
            ("gate_logits", layer.g_proj),
            ("expanded_kv", layer.kv_b_proj),
        ):
            handles.append(module.register_forward_hook(save_output(name)))
    try:
        start = 0 if past_key_values is None else past_key_values.get_seq_length(layer.layer_idx)
        mask = causal_mask(batch, tokens, start + tokens, start, dtype=hidden_states.dtype, device=hidden_states.device)
        output = layer(hidden_states, attention_mask=mask, past_key_values=past_key_values)
    finally:
        for handle in handles:
            handle.remove()
    return output, captured["pre_wo"]


def staged_dense_reference(definitions, layer, query_projection, cache_c, cache_e, gate_logits, *, start):
    """CPU/test reference only, using exact official eager with expanded cached K/V."""
    batch, tq, _ = query_projection.shape
    heads = layer.num_heads
    tk = cache_c.shape[1]
    query = query_projection.view(batch, tq, heads, 192).transpose(1, 2)
    expanded = layer.kv_b_proj(cache_c).view(batch, tk, heads, 256).transpose(1, 2)
    content, value = torch.split(expanded, (128, 128), dim=-1)
    auxiliary = cache_e[:, None, :, :].expand(batch, heads, tk, 64)
    key = torch.cat((content, auxiliary), dim=-1)
    mask = causal_mask(batch, tq, tk, start, dtype=query.dtype, device=query.device)
    ungated, _ = definitions.eager_attention(layer, query, key, value, mask, scaling=layer.scaling, dropout=0.0)
    pre_wo = ungated.reshape(batch, tq, heads * 128) * torch.sigmoid(gate_logits)
    return pre_wo, layer.o_proj(pre_wo)
