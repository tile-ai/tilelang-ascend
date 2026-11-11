"""Independent original seven-input CPU references; not a candidate fallback."""

from __future__ import annotations
from dataclasses import dataclass
import math
from numbers import Real
from collections.abc import Sequence
import torch

REFERENCE_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
FP16_RTOL = 1.0e-2
FP16_ATOL = 2.0e-3
FP16_MAX_NORMALIZED_ERROR = 5.0e-3


@dataclass(frozen=True)
class ShapeSpec:
    B: int
    Tq: int
    Tk: int
    H: int
    D: int
    E: int
    R: int
    V: int
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    query_start: tuple[int, ...]
    scale: float


def _host_lengths(name: str, values: Sequence[int], batch: int) -> tuple[int, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be a host tuple/list of Python integers")
    if len(values) != batch:
        raise ValueError(f"{name} must have {batch} entries, got {len(values)}")
    if any(type(value) is not int for value in values):
        raise TypeError(f"{name} entries must be Python integers (not bool or tensors)")
    return tuple(values)


def validate_inputs(
    q_content: torch.Tensor,
    q_aux: torch.Tensor,
    latent_kv: torch.Tensor,
    key_aux: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    gate_logits: torch.Tensor,
    *,
    q_lens: Sequence[int],
    kv_lens: Sequence[int],
    query_start: Sequence[int],
    scale: float | None = None,
    require_npu: bool = False,
    allowed_dtypes: tuple[torch.dtype, ...] = REFERENCE_DTYPES,
) -> ShapeSpec:
    """Validate the shared contract without hiding device transfers or conversions."""
    tensors = {
        "q_content": (q_content, 4),
        "q_aux": (q_aux, 4),
        "latent_kv": (latent_kv, 3),
        "key_aux": (key_aux, 3),
        "w_k": (w_k, 3),
        "w_v": (w_v, 3),
        "gate_logits": (gate_logits, 4),
    }
    for name, (tensor, rank) in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != rank:
            raise ValueError(f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.dtype not in allowed_dtypes:
            raise TypeError(f"{name} dtype {tensor.dtype} is unsupported")
        if tensor.dtype != q_content.dtype:
            raise TypeError(f"{name} dtype must match q_content ({q_content.dtype})")
        if tensor.device != q_content.device:
            raise ValueError(f"{name} device must match q_content ({q_content.device})")
    if require_npu and q_content.device.type not in ("npu", "privateuseone"):
        raise ValueError("q_content must be on an NPU")

    B, Tq, H, D = map(int, q_content.shape)
    Tk, R = map(int, latent_kv.shape[1:])
    E = int(q_aux.shape[-1])
    V = int(w_v.shape[1])
    if min(B, H, D, E, R, V) <= 0:
        raise ValueError("B, H, D, E, R and V must be positive; only Tq/Tk may be zero")
    expected = {
        "q_aux": (B, Tq, H, E),
        "latent_kv": (B, Tk, R),
        "key_aux": (B, Tk, E),
        "w_k": (H, D, R),
        "w_v": (H, V, R),
        "gate_logits": (B, Tq, H, V),
    }
    for name, shape in expected.items():
        if tuple(tensors[name][0].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensors[name][0].shape)}")

    ql = _host_lengths("q_lens", q_lens, B)
    kl = _host_lengths("kv_lens", kv_lens, B)
    qs = _host_lengths("query_start", query_start, B)
    for b in range(B):
        if not 0 <= ql[b] <= Tq:
            raise ValueError(f"q_lens[{b}] must lie in [0, {Tq}]")
        if not 0 <= kl[b] <= Tk:
            raise ValueError(f"kv_lens[{b}] must lie in [0, {Tk}]")
        if not 0 <= qs[b] <= kl[b]:
            raise ValueError(f"query_start[{b}] must lie in [0, kv_lens[{b}]]")
        if qs[b] + ql[b] > kl[b]:
            raise ValueError(f"query_start[{b}] + q_lens[{b}] exceeds kv_lens[{b}]")
    if scale is None:
        scale = 1.0 / math.sqrt(D + E)
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise TypeError("scale must be a finite positive host number")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return ShapeSpec(B, Tq, Tk, H, D, E, R, V, ql, kl, qs, float(scale))


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Explicit diagnostic helper, separate from the metadata-only operator wrapper."""
    if actual.shape != expected.shape:
        raise AssertionError(f"output shape {actual.shape} != reference {expected.shape}")
    a = actual.detach().to(device="cpu", dtype=torch.float64)
    e = expected.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(a).all() or not torch.isfinite(e).all():
        raise AssertionError("output and reference must contain only finite values")
    if a.numel() == 0:
        return {"max_abs": 0.0, "max_normalized": 0.0, "reference_abs_max": 0.0}
    max_abs = float((a - e).abs().max())
    reference_abs_max = float(e.abs().max())
    return {
        "max_abs": max_abs,
        "max_normalized": max_abs / max(reference_abs_max, 1.0e-9),
        "reference_abs_max": reference_abs_max,
    }


def assert_fp16_close(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    metrics = error_metrics(actual, expected)
    torch.testing.assert_close(
        actual.detach().to(device="cpu", dtype=torch.float64),
        expected.detach().to(device="cpu", dtype=torch.float64),
        rtol=FP16_RTOL,
        atol=FP16_ATOL,
    )
    assert metrics["max_normalized"] <= FP16_MAX_NORMALIZED_ERROR, metrics
    return metrics


def expanded_reference(
    q_content: torch.Tensor,
    q_aux: torch.Tensor,
    latent_kv: torch.Tensor,
    key_aux: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    gate_logits: torch.Tensor,
    *,
    q_lens: Sequence[int],
    kv_lens: Sequence[int],
    query_start: Sequence[int],
    scale: float | None = None,
    compute_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Expand K/V, explicitly slice the causal prefix, then reduce values.

    This route loops over query positions, so it does not share the absorbed route's
    vectorized mask or softmax implementation.
    """
    spec = validate_inputs(
        q_content,
        q_aux,
        latent_kv,
        key_aux,
        w_k,
        w_v,
        gate_logits,
        q_lens=q_lens,
        kv_lens=kv_lens,
        query_start=query_start,
        scale=scale,
    )
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError("reference compute_dtype must be torch.float32 or torch.float64")
    q, qa, c, ka, wk, wv, gate = (
        tensor.to(dtype=compute_dtype) for tensor in (q_content, q_aux, latent_kv, key_aux, w_k, w_v, gate_logits)
    )
    out = torch.zeros(
        (spec.B, spec.Tq, spec.H, spec.V),
        dtype=compute_dtype,
        device=q.device,
    )
    for b in range(spec.B):
        if spec.q_lens[b] == 0:
            continue
        for h in range(spec.H):
            keys = c[b, : spec.kv_lens[b]] @ wk[h].transpose(-1, -2)
            values = c[b, : spec.kv_lens[b]] @ wv[h].transpose(-1, -2)
            for i in range(spec.q_lens[b]):
                visible = spec.query_start[b] + i + 1
                score = (keys[:visible] @ q[b, i, h] + ka[b, :visible] @ qa[b, i, h]) * spec.scale
                exp_score = torch.exp(score - torch.max(score))
                probability = exp_score / torch.sum(exp_score)
                value = torch.sum(probability[:, None] * values[:visible], dim=0)
                out[b, i, h] = value * torch.sigmoid(gate[b, i, h])
    return out


def absorbed_reference(
    q_content: torch.Tensor,
    q_aux: torch.Tensor,
    latent_kv: torch.Tensor,
    key_aux: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    gate_logits: torch.Tensor,
    *,
    q_lens: Sequence[int],
    kv_lens: Sequence[int],
    query_start: Sequence[int],
    scale: float | None = None,
    compute_dtype: torch.dtype = torch.float64,
    round_intermediates: bool = False,
) -> torch.Tensor:
    """Absorb W_K into queries; softmax-read the latent cache; restore values.

    round_intermediates=True is a CPU numerical-budget model: Qlatent, softmax
    probabilities and Z are rounded to the input dtype after FP32/FP64 arithmetic.
    It is deliberately not claimed to emulate any specific Cube accumulation order.
    """
    spec = validate_inputs(
        q_content,
        q_aux,
        latent_kv,
        key_aux,
        w_k,
        w_v,
        gate_logits,
        q_lens=q_lens,
        kv_lens=kv_lens,
        query_start=query_start,
        scale=scale,
    )
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError("reference compute_dtype must be torch.float32 or torch.float64")
    q, qa, c, ka, wk, wv, gate = (
        tensor.to(dtype=compute_dtype) for tensor in (q_content, q_aux, latent_kv, key_aux, w_k, w_v, gate_logits)
    )
    out = torch.zeros(
        (spec.B, spec.Tq, spec.H, spec.V),
        dtype=compute_dtype,
        device=q.device,
    )
    for b in range(spec.B):
        qn, kn = spec.q_lens[b], spec.kv_lens[b]
        if qn == 0:
            continue
        # This implementation uses a broadcast mask, unlike the per-query slicing above.
        query_positions = torch.arange(qn, device=q.device) + spec.query_start[b]
        key_positions = torch.arange(kn, device=q.device)
        visible = key_positions[None, :] <= query_positions[:, None]
        for h in range(spec.H):
            q_latent = q[b, :qn, h] @ wk[h]
            if round_intermediates:
                q_latent = q_latent.to(q_content.dtype).to(compute_dtype)
            scores = (q_latent @ c[b, :kn].transpose(-1, -2) + qa[b, :qn, h] @ ka[b, :kn].transpose(-1, -2)) * spec.scale
            probability = torch.softmax(scores.masked_fill(~visible, -torch.inf), dim=-1)
            if round_intermediates:
                probability = probability.to(q_content.dtype).to(compute_dtype)
            z = probability @ c[b, :kn]
            if round_intermediates:
                z = z.to(q_content.dtype).to(compute_dtype)
            values = z @ wv[h].transpose(-1, -2)
            out[b, :qn, h] = values * torch.sigmoid(gate[b, :qn, h])
    if round_intermediates:
        out = out.to(q_content.dtype)
    return out
