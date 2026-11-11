"""Guarded inference-only unified forward; original arithmetic path retained.

This is not an installed package or a production-readiness claim. Runtime
metadata values are owned/validated by the caller before H2D transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import ModuleType

import torch

from .metadata import pack_metadata as pack_metadata, metadata_for_reference as metadata_for_reference
from ._contracts import validated_host_scale, validate_chain_spans
from . import layout as default_layout


DEFAULT_SCALE = 1 / math.sqrt(192)


def validate_scale(scale):
    return validated_host_scale(scale, DEFAULT_SCALE)


def validate_forward_inputs(
    q_content,
    q_aux,
    latent_kv,
    key_aux,
    w_k,
    w_v,
    gate_logits,
    metadata,
    *,
    scale=None,
    layout_module=None,
    block_m=32,
    block_n=64,
    max_blocks=20,
    block_n_selector=None,
):
    """Complete host-side contract before kernel import, views, allocations or launch.

    Tensor attributes/physical format and current device are queried, never data
    or device metadata contents. CPU metadata content validity remains caller-owned.
    """
    if torch.is_grad_enabled() and any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for value in (q_content, q_aux, latent_kv, key_aux, w_k, w_v, gate_logits, metadata)
    ):
        raise RuntimeError("Gated MLA is inference-only; use no_grad/inference_mode or detach inputs before calling")
    scale = validate_scale(scale)
    if not isinstance(q_content, torch.Tensor) or q_content.ndim != 4:
        raise ValueError("q_content must have shape[B,Tq,H,128]")
    batch, tq, heads, width = map(int, q_content.shape)
    if batch < 1 or heads < 1 or tq < 0 or width != 128:
        raise ValueError("forward requires positive B/H, nonnegative Tq and content width128")
    if q_content.device.type not in ("npu", "privateuseone"):
        raise ValueError("decode requires an NPU")
    if q_content.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("decode dtype must be float16 or bfloat16")
    if not isinstance(latent_kv, torch.Tensor) or latent_kv.ndim != 3:
        raise ValueError("latent_kv must have shape[B,Tk,512]")
    capacity = int(latent_kv.shape[1])
    if capacity < 0:
        raise ValueError("physical cache capacity must be nonnegative")
    tensors = {
        "q_content": (q_content, (batch, tq, heads, 128)),
        "q_aux": (q_aux, (batch, tq, heads, 64)),
        "latent_kv": (latent_kv, (batch, capacity, 512)),
        "key_aux": (key_aux, (batch, capacity, 64)),
        "w_k": (w_k, (heads, 128, 512)),
        "w_v": (w_v, (heads, 128, 512)),
        "gate_logits": (gate_logits, (batch, tq, heads, 128)),
    }
    for name, (value, shape) in tensors.items():
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape{shape}")
        if value.dtype != q_content.dtype or value.device != q_content.device or not value.is_contiguous():
            raise ValueError(f"{name} must share query dtype/device and be contiguous")
    if not isinstance(metadata, torch.Tensor) or tuple(metadata.shape) != (batch, 8):
        raise ValueError("metadata must have shape[B,8]")
    if metadata.dtype != torch.int32 or metadata.device != q_content.device or not metadata.is_contiguous():
        raise ValueError("metadata must be contiguous device int32")
    if block_n_selector is not None:
        block_n = block_n_selector(batch, tq, heads, capacity, scale)
    validate_chain_spans(
        batch,
        tq,
        heads,
        capacity,
        decode=tq == 1 and scale == DEFAULT_SCALE,
        block_m=block_m,
        block_n=block_n,
        max_blocks=max_blocks,
    )
    # The cc9 adapter selects current_stream() without a device argument while
    # allocating outputs on inputs[0].device. Reject a mismatch; never switch it.
    if q_content.device.index is None or torch.npu.current_device() != q_content.device.index:
        raise ValueError("select the input NPU device before forward; current device must match all inputs")
    layout_module = default_layout if layout_module is None else layout_module
    for name, (value, shape) in tensors.items():
        layout_module.require_dense_base(value, name=name, shape=shape, dtype=q_content.dtype, device=q_content.device)
    layout_module.require_dense_base(metadata, name="metadata", shape=(batch, 8), dtype=torch.int32, device=q_content.device)
    return batch, tq, heads, capacity, scale


def _view(value, shape, name, audit, layout_guard):
    viewed = layout_guard.guarded_view(value, shape, name=name)
    alias = (
        value.untyped_storage().data_ptr() == viewed.untyped_storage().data_ptr()
        and value.data_ptr() == viewed.data_ptr()
        and value.storage_offset() == viewed.storage_offset()
        and value.numel() == viewed.numel()
    )
    if not alias:
        raise AssertionError(f"{name} must be a zero-copy, same-storage view")
    audit[name] = {
        "from": list(value.shape),
        "to": list(viewed.shape),
        "same_storage": True,
        "dtype": str(value.dtype),
        "strides": list(viewed.stride()),
    }
    return viewed


@dataclass(frozen=True)
class KernelBundle:
    projections: ModuleType
    decode_module: ModuleType
    prefill_module: ModuleType
    query_module: ModuleType
    layout_module: ModuleType
    empty_cache_module: ModuleType
    target: str = "pto"
    block_m: int = 32
    block_n: int = 64
    max_blocks: int = 24

    def __post_init__(self):
        if self.target not in ("pto", "ascendc"):
            raise ValueError("target must be pto or ascendc")
        if self.block_m not in (16, 32) or self.block_n not in (64, 128):
            raise ValueError("unsupported Cube tile configuration")
        if type(self.max_blocks) is not int or self.max_blocks < 1:
            raise ValueError("max_blocks must be a positive integer")
        for module, names in [
            (self.projections, ("q_absorb", "value_gate", "q_absorb_kernel", "value_gate_kernel")),
            (self.decode_module, ("cube_decode", "build_cube_attention")),
            (self.prefill_module, ("cube_prefill", "build_cube_attention")),
            (self.query_module, ("q_absorb_residual", "q_absorb_residual_kernel")),
            (self.layout_module, ("require_dense_base", "guarded_view")),
            (self.empty_cache_module, ("empty_cache_forward", "build_empty_cache_zero")),
        ]:
            for name in names:
                if not callable(getattr(module, name, None)):
                    raise TypeError(f"Injected module lacks callable {name}")

    def forward(self, q_content, q_aux, latent_kv, key_aux, w_k, w_v, gate_logits, metadata, *, scale=None, return_intermediates=False):
        contract = validate_forward_inputs(
            q_content,
            q_aux,
            latent_kv,
            key_aux,
            w_k,
            w_v,
            gate_logits,
            metadata,
            scale=scale,
            layout_module=self.layout_module,
            block_m=self.block_m,
            block_n=self.block_n,
            max_blocks=self.max_blocks,
        )
        return self._forward_validated(
            q_content,
            q_aux,
            latent_kv,
            key_aux,
            w_k,
            w_v,
            gate_logits,
            metadata,
            contract,
            return_intermediates=return_intermediates,
        )

    def _forward_validated(
        self, q_content, q_aux, latent_kv, key_aux, w_k, w_v, gate_logits, metadata, contract, *, return_intermediates=False
    ):
        """Private execution path; callers use public forward for validation."""
        batch, tq, heads, capacity, scale = contract
        if tq == 0:
            output = torch.empty((batch, 0, heads, 128), dtype=q_content.dtype, device=q_content.device)
            self.layout_module.require_dense_base(output, name="empty_query.output")
            if return_intermediates:
                return (
                    output,
                    {"q_latent_hi": None, "q_latent_lo": None, "latent_out": None},
                    {
                        "views": {},
                        "kernels": {},
                        "query_residual_enabled": False,
                        "selected_attention": "empty_query",
                        "scale": scale,
                    },
                )
            return output
        if capacity == 0:
            # CPU metadata owner has guaranteed all rows inactive. The real
            # output buffer is format-checked before the dedicated NPU write.
            output = torch.empty((batch, tq, heads, 128), dtype=q_content.dtype, device=q_content.device)
            self.layout_module.require_dense_base(output, name="empty_cache.output")
            output, zero_kernel = self.empty_cache_module.empty_cache_forward(
                q_content,
                q_aux,
                latent_kv,
                key_aux,
                w_k,
                w_v,
                gate_logits,
                metadata,
                target=self.target,
                scale=scale,
                out=output,
                return_kernel=True,
            )
            if return_intermediates:
                return (
                    output,
                    {"q_latent_hi": None, "q_latent_lo": None, "latent_out": None},
                    {
                        "views": {},
                        "kernels": {"empty_cache_zero": zero_kernel},
                        "query_residual_enabled": False,
                        "selected_attention": "empty_cache_zero",
                        "scale": scale,
                    },
                )
            return output

        views = {}
        ql_low = None
        if q_content.dtype == torch.bfloat16:
            ql, ql_low = self.query_module.q_absorb_residual(q_content, w_k, target=self.target)
        else:
            ql = self.projections.q_absorb(q_content, w_k, target=self.target)
        if tuple(ql.shape) != (batch, tq, heads, 512) or ql.dtype != q_content.dtype or ql.device != q_content.device:
            raise AssertionError("projection QL contract mismatch")
        use_decode = tq == 1 and scale == DEFAULT_SCALE
        qshape = (batch, heads, 512) if use_decode else (batch, tq, heads, 512)
        qashape = (batch, heads, 64) if use_decode else (batch, tq, heads, 64)
        ql_view = _view(ql, qshape, "q_latent_hi", views, self.layout_module)
        low_view = None
        if ql_low is not None:
            if tuple(ql_low.shape) != tuple(ql.shape) or ql_low.dtype != ql.dtype or ql_low.device != ql.device:
                raise AssertionError("projection QL residual contract mismatch")
            if ql_low.untyped_storage().data_ptr() == ql.untyped_storage().data_ptr():
                raise AssertionError("Hi and Lo must be independent output tensors")
            low_view = _view(ql_low, qshape, "q_latent_lo", views, self.layout_module)
        qa_view = _view(q_aux, qashape, "q_aux", views, self.layout_module)
        common = dict(block_m=self.block_m, block_n=self.block_n, target=self.target, ql_residual=low_view, return_kernels=True)
        if use_decode:
            latent, attention_kernels = self.decode_module.cube_decode(ql_view, qa_view, latent_kv, key_aux, metadata, **common)
            route = "c3_decode"
        else:
            latent, attention_kernels = self.prefill_module.cube_prefill(
                ql_view,
                qa_view,
                latent_kv,
                key_aux,
                metadata,
                scale=scale,
                max_blocks=self.max_blocks,
                **common,
            )
            route = "c4_prefill"
        if tuple(latent.shape) != qshape or latent.dtype != torch.float32:
            raise AssertionError("Cube latent contract mismatch")
        latent_view = _view(latent, (batch, tq, heads, 512), "latent_out", views, self.layout_module)
        output = self.projections.value_gate(latent_view, w_v, gate_logits, target=self.target)
        if tuple(output.shape) != (batch, tq, heads, 128) or output.dtype != q_content.dtype:
            raise AssertionError("value/gate output contract mismatch")
        self.layout_module.require_dense_base(
            output, name="output", shape=(batch, tq, heads, 128), dtype=q_content.dtype, device=q_content.device
        )
        if return_intermediates:
            dtype = "float16" if q_content.dtype == torch.float16 else "bfloat16"
            query_name = "q_absorb" if ql_low is None else "q_absorb_residual"
            query_kernel = (
                self.projections.q_absorb_kernel(batch, tq, heads, dtype, target=self.target)
                if ql_low is None
                else self.query_module.q_absorb_residual_kernel(batch, tq, heads, dtype, target=self.target)
            )
            kernels = {
                query_name: query_kernel,
                **attention_kernels,
                "value_gate": self.projections.value_gate_kernel(batch, tq, heads, dtype, target=self.target),
            }
            # No host-side tensor addition: the probe reconstructs the logical
            # Q on CPU, while the Cube kernel consumes the two terms directly.
            return (
                output,
                {"q_latent_hi": ql, "q_latent_lo": ql_low, "latent_out": latent_view},
                {
                    "views": views,
                    "kernels": kernels,
                    "query_residual_enabled": ql_low is not None,
                    "selected_attention": route,
                    "scale": scale,
                },
            )
        return output
