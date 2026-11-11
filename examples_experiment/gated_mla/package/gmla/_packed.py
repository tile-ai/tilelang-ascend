"""Synchronous packed Kimi MLA boundary using injected verified NPU components.

No Torch attention fallback, payload packing/copy, or host head/token arithmetic.
External original-layer projections, norms and W_O remain explicit model operations.
"""

from dataclasses import dataclass
import threading

import torch

from ._contracts import validated_host_scale, validate_chain_spans


DEFAULT_SCALE = 192**-0.5


def require_inference():
    if torch.is_grad_enabled():
        raise RuntimeError("use torch.no_grad or inference_mode; this adapter does not implement backward")


def checked_scale(scale):
    return validated_host_scale(scale, DEFAULT_SCALE)


def attention_route(query_capacity, scale):
    if type(query_capacity) is not int or query_capacity <= 0:
        raise NotImplementedError("this model adapter requires positive physical Tq; logical inactive rows use q_lens=0")
    return "c3_decode" if query_capacity == 1 and checked_scale(scale) == DEFAULT_SCALE else "c4_prefill"


def append_metadata(protocol, lengths, q_lens, *, Qcap, capacity):
    if not isinstance(q_lens, (tuple, list)) or len(q_lens) != len(lengths) or any(type(x) is not int for x in q_lens):
        raise TypeError("q_lens must be CPU-owned Python integer tuples/lists")
    next_lengths = [old + count for old, count in zip(lengths, q_lens)]
    return protocol.build_runtime_metadata(q_lens, next_lengths, list(lengths), B=len(lengths), Tq=Qcap, Tk=capacity)


class CacheCommittedForwardError(RuntimeError):
    def __init__(self, owner):
        self.cache_committed = True
        self.lengths = owner.lengths
        self.generation = owner.generation
        self.retry_policy = "repair_then_read_existing_or_abandon"
        super().__init__(
            f"cache append committed at generation {self.generation}, lengths {self.lengths}, but forward failed; "
            "do not append these tokens again. Repair and explicitly read-existing, or abandon this request."
        )


@dataclass(frozen=True)
class Components:
    packed_projection: object
    query_residual: object
    decode: object
    prefill: object
    layout: object
    protocol: object
    target: str = "pto"
    block_m: int = 32
    block_n: int = 64
    max_blocks: int = 24

    def __post_init__(self):
        if self.target not in ("pto", "ascendc") or self.block_m not in (16, 32) or self.block_n not in (64, 128):
            raise ValueError("unsupported backend or Cube tile")
        if type(self.max_blocks) is not int or not 1 <= self.max_blocks <= 24:
            raise ValueError("max_blocks must be in [1,24]")
        for module, names in (
            (self.packed_projection, ("q_absorb_packed", "value_gate_packed", "q_absorb_packed_kernel", "value_gate_packed_kernel")),
            (self.query_residual, ("q_absorb_packed_residual", "q_absorb_packed_residual_kernel")),
            (self.decode, ("cube_decode",)),
            (self.prefill, ("cube_prefill",)),
            (self.layout, ("require_dense_base", "guarded_view")),
            (self.protocol, ("build_runtime_metadata",)),
        ):
            if any(not callable(getattr(module, name, None)) for name in names):
                raise TypeError(f"injected component lacks {names}")


class PackedModelAdapter:
    def __init__(self, owner, *, heads, components):
        if type(heads) is not int or heads <= 0:
            raise ValueError("heads must be a positive integer")
        if owner.target != components.target:
            raise ValueError("cache owner and core must select the same backend")
        self.owner, self.heads, self.parts = owner, heads, components
        self._check_cache_formats()
        # Public owner entry validates its bound context without private attributes.
        binding = owner.read_existing((0,) * owner.B, owner.lengths, Qcap=0)
        self.binding_control_transfer_bytes = binding.metadata_h2d_bytes
        self._thread = threading.get_ident()
        self._stream = torch.npu.current_stream(owner.device)
        self._cache_storage = (owner.cache_c.untyped_storage().data_ptr(), owner.cache_e.untyped_storage().data_ptr())

    def _check_cache_formats(self):
        o, guard = self.owner, self.parts.layout.require_dense_base
        guard(o.cache_c, name="CacheC", shape=(o.B, o.capacity, 512), dtype=o.dtype, device=o.device)
        guard(o.cache_e, name="CacheE", shape=(o.B, o.capacity, 64), dtype=o.dtype, device=o.device)

    def check_context(self):
        if threading.get_ident() != self._thread or torch.npu.current_device() != self.owner.device.index:
            raise RuntimeError("adapter requires its bound CPU thread and NPU device")
        if torch.npu.current_stream(self.owner.device).npu_stream != self._stream.npu_stream:
            raise RuntimeError("adapter requires its bound NPU stream")
        if (self.owner.cache_c.untyped_storage().data_ptr(), self.owner.cache_e.untyped_storage().data_ptr()) != self._cache_storage:
            raise RuntimeError("cache storage changed; grow/reset/rebind is not implemented")

    def synchronize(self):
        self.check_context()
        self._stream.synchronize()

    def _packed_inputs(self, query, weight, gate, scale, *, append_sources=False):
        self.check_context()
        self._check_cache_formats()
        guard, view = self.parts.layout.require_dense_base, self.parts.layout.guarded_view
        o, H = self.owner, self.heads
        guard(query, name="raw Q projection", dtype=o.dtype, device=o.device)
        guard(weight, name="raw kv_b weight", dtype=o.dtype, device=o.device)
        guard(gate, name="gate logits", dtype=o.dtype, device=o.device)
        flat_q = query.ndim == 3 and query.shape[0] == o.B and query.shape[2] == H * 192
        packed_q = query.ndim == 4 and query.shape[0] == o.B and tuple(query.shape[-2:]) == (H, 192)
        if not (flat_q or packed_q):
            raise ValueError("Q must be [B,Tq,H*192] or packed [B,Tq,H,192]")
        Tq = int(query.shape[1])
        attention_route(Tq, scale)
        if tuple(weight.shape) not in ((H * 256, 512), (H, 256, 512)):
            raise ValueError("KV weight must be original [H*256,512] or packed [H,256,512]")
        if tuple(gate.shape) not in ((o.B, Tq, H * 128), (o.B, Tq, H, 128)):
            raise ValueError("gate logits must be [B,Tq,H*128] or [B,Tq,H,128]")
        validate_chain_spans(
            o.B,
            Tq,
            H,
            o.capacity,
            packed=True,
            decode=attention_route(Tq, scale) == "c3_decode",
            append_sources=append_sources,
            block_m=self.parts.block_m,
            block_n=self.parts.block_n,
            max_blocks=self.parts.max_blocks,
        )
        for tensor in (query, weight, gate):
            if tensor.untyped_storage().data_ptr() in self._cache_storage:
                raise ValueError("query, weight or gate must not alias writable cache storage")
        return (
            view(query, (o.B, Tq, H, 192), name="packed Q"),
            view(weight, (H, 256, 512), name="packed KV weight"),
            view(gate, (o.B, Tq, H, 128), name="packed gate logits"),
            Tq,
        )

    def append_and_forward(self, query, weight, normalized_c, raw_kv, gate_logits, q_lens, *, scale=None, return_intermediates=False):
        require_inference()
        scale = checked_scale(scale)
        q, w, g, Tq = self._packed_inputs(query, weight, gate_logits, scale, append_sources=True)
        guard = self.parts.layout.require_dense_base
        guard(normalized_c, name="Cnorm", shape=(self.owner.B, Tq, 512), dtype=self.owner.dtype, device=self.owner.device)
        guard(raw_kv, name="rawKV", shape=(self.owner.B, Tq, 576), dtype=self.owner.dtype, device=self.owner.device)
        meta = append_metadata(self.parts.protocol, self.owner.lengths, q_lens, Qcap=Tq, capacity=self.owner.capacity)
        read = self.owner.append(normalized_c, raw_kv, q_lens, metadata_cpu=meta)
        try:
            return self._core(q, w, g, read, scale, return_intermediates)
        except Exception as error:
            raise CacheCommittedForwardError(self.owner) from error

    def read_existing_forward(self, query, weight, gate_logits, q_lens, query_start, *, scale=None, return_intermediates=False):
        require_inference()
        scale = checked_scale(scale)
        q, w, g, Tq = self._packed_inputs(query, weight, gate_logits, scale)
        read = self.owner.read_existing(q_lens, query_start, Qcap=Tq)
        return self._core(q, w, g, read, scale, return_intermediates)

    def _core(self, q, w, gate, read, scale, return_intermediates):
        p, o = self.parts, self.owner
        guard, view = p.layout.require_dense_base, p.layout.guarded_view
        B, Tq, H, _ = q.shape
        guard(read.metadata, name="runtime metadata", shape=(B, 8), dtype=torch.int32, device=o.device)
        guard(read.cache_c, name="read C", shape=(B, o.capacity, 512), dtype=o.dtype, device=o.device)
        guard(read.cache_e, name="read E", shape=(B, o.capacity, 64), dtype=o.dtype, device=o.device)
        low = None
        if q.dtype == torch.bfloat16:
            high, low, qa = p.query_residual.q_absorb_packed_residual(q, w, target=p.target)
        else:
            high, qa = p.packed_projection.q_absorb_packed(q, w, target=p.target)
        guard(high, name="QL high", shape=(B, Tq, H, 512), dtype=o.dtype, device=o.device)
        guard(qa, name="QA", shape=(B, Tq, H, 64), dtype=o.dtype, device=o.device)
        if low is not None:
            guard(low, name="QL low", shape=(B, Tq, H, 512), dtype=o.dtype, device=o.device)
            if high.untyped_storage().data_ptr() == low.untyped_storage().data_ptr():
                raise AssertionError("QL high and low must have independent output storage")
        route = attention_route(Tq, scale)
        qshape = (B, H, 512) if route == "c3_decode" else (B, Tq, H, 512)
        ashape = (B, H, 64) if route == "c3_decode" else (B, Tq, H, 64)
        qh = view(high, qshape, name="attention Q high")
        ql = None if low is None else view(low, qshape, name="attention Q low")
        aq = view(qa, ashape, name="attention QA")
        options = dict(block_m=p.block_m, block_n=p.block_n, target=p.target, ql_residual=ql, return_kernels=True)
        if route == "c3_decode":
            latent, kernels = p.decode.cube_decode(qh, aq, read.cache_c, read.cache_e, read.metadata, **options)
        else:
            latent, kernels = p.prefill.cube_prefill(
                qh, aq, read.cache_c, read.cache_e, read.metadata, scale=scale, max_blocks=p.max_blocks, **options
            )
        guard(latent, name="latent output", shape=qshape, dtype=torch.float32, device=o.device)
        latent_4d = view(latent, (B, Tq, H, 512), name="value input")
        output = p.packed_projection.value_gate_packed(latent_4d, w, gate, target=p.target)
        guard(output, name="gated output before W_O", shape=(B, Tq, H, 128), dtype=o.dtype, device=o.device)
        self.synchronize()
        if return_intermediates:
            dtype = "float16" if q.dtype == torch.float16 else "bfloat16"
            query_kernel = (
                p.packed_projection.q_absorb_packed_kernel(B, Tq, H, dtype, target=p.target)
                if low is None
                else p.query_residual.q_absorb_packed_residual_kernel(B, Tq, H, dtype, target=p.target)
            )
            audit = {
                "route": route,
                "scale": scale,
                "cache_generation": read.generation,
                "cache_lengths": read.lengths,
                "metadata_h2d_bytes": read.metadata_h2d_bytes,
                "query_residual_enabled": low is not None,
                "kernels": {
                    "packed_query": query_kernel,
                    **kernels,
                    "packed_value_gate": p.packed_projection.value_gate_packed_kernel(B, Tq, H, dtype, target=p.target),
                },
            }
            return output, {"q_latent_hi": high, "q_latent_lo": low, "qa": qa, "latent_out": latent_4d}, audit
        return output
