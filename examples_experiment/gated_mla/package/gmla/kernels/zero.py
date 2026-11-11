"""Isolated nonempty-query/empty-cache NPU zero branch; no host zero fallback."""

from functools import lru_cache
import math

import torch
import tilelang
import tilelang.language as T

from .._contracts import validated_host_scale


DEFAULT_SCALE = 1 / math.sqrt(192)
EXPERT_PASSES = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}


def zero_plan(batch, query_capacity, heads):
    if any(type(x) is not int or x < 1 for x in (batch, query_capacity, heads)):
        raise ValueError("empty-cache kernel requires positive B/Tq/H; Tq0 uses the no-kernel path")
    rows = batch * query_capacity * heads
    if rows * 128 > 2**31 - 1:
        raise ValueError("output element indexing exceeds int32 range")
    blocks = min(24, (rows + 1) // 2)
    workers = 2 * blocks
    return {"rows": rows, "blocks": blocks, "logical_aiv_workers": workers, "rounds": (rows + workers - 1) // workers}


def validate_scale(scale):
    return validated_host_scale(scale, DEFAULT_SCALE)


def pack_inactive_metadata(rows, *, batch):
    """CPU owner validates all-inactive Tk0 values before H2D; no GPU reads."""
    if type(batch) is not int or batch < 1 or not isinstance(rows, (tuple, list)) or len(rows) != batch:
        raise ValueError("metadata must contain B CPU-owned rows")
    packed = []
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) not in (3, 8) or any(type(x) is not int for x in row):
            raise TypeError("metadata rows require3 or8 Python integers")
        if any(row):
            raise ValueError("Tk0 requires q_len=kv_len=query_start=0 and reserved columns zero")
        packed.append([*row[:3], 0, 0, 0, 0, 0])
    return torch.tensor(packed, dtype=torch.int32, device="cpu")


@lru_cache(maxsize=128)
def build_empty_cache_zero(*, batch, query_capacity, heads, dtype="float16", target="pto"):
    """Explicit-output kernel: caller supplies Q and Y; only Y is written.

    out_idx=None intentionally disables automatic output allocation, allowing
    the acceptance probe to poison the actual output buffer before launch.
    """
    if target not in ("pto", "ascendc") or dtype not in ("float16", "bfloat16"):
        raise ValueError("unsupported target or dtype")
    plan = zero_plan(batch, query_capacity, heads)
    rows, blocks, workers, rounds = (plan[k] for k in ("rows", "blocks", "logical_aiv_workers", "rounds"))

    @T.prim_func
    def main(
        Q: T.Tensor((batch, query_capacity, heads, 128), dtype),
        Y: T.Tensor((batch, query_capacity, heads, 128), dtype),
    ):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            zeros = T.alloc_ub((128,), dtype)
            T.annotate_address({zeros: 0})
            with T.Scope("V"):
                T.tile.fill(zeros, 0.0)
                T.barrier_all()
                for turn in T.serial(rounds):
                    row = cid * 2 + vid + turn * workers
                    if row < rows:
                        h = row % heads
                        t = (row // heads) % query_capacity
                        b = row // (query_capacity * heads)
                        T.copy(zeros, Y[b, t, h, 0])
                        T.barrier_all()

    return tilelang.compile(main, out_idx=None, target=target, pass_configs=EXPERT_PASSES)


def empty_cache_forward(
    q_content, q_aux, latent_kv, key_aux, w_k, w_v, gate_logits, metadata, *, target="pto", scale=None, out=None, return_kernel=False
):
    """Validate the seven-input Tk0 contract then launch exactly one zero kernel.

    Caller MUST validate CPU-owned metadata values with pack_inactive_metadata
    before transfer. This wrapper reads only tensor attributes, never NPU data.
    All physical input values are finite under the surrounding model contract.
    """
    if target not in ("pto", "ascendc"):
        raise ValueError("target must be pto or ascendc")
    validate_scale(scale)
    if not isinstance(q_content, torch.Tensor) or q_content.ndim != 4:
        raise ValueError("q_content must have shape[B,Tq,H,128]")
    batch, tq, heads, width = map(int, q_content.shape)
    zero_plan(batch, tq, heads)
    if width != 128 or q_content.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q_content requires width128 and FP16/BF16")
    if q_content.device.type not in ("npu", "privateuseone"):
        raise ValueError("empty-cache zero requires an NPU")
    tensors = {
        "q_content": (q_content, (batch, tq, heads, 128)),
        "q_aux": (q_aux, (batch, tq, heads, 64)),
        "latent_kv": (latent_kv, (batch, 0, 512)),
        "key_aux": (key_aux, (batch, 0, 64)),
        "w_k": (w_k, (heads, 128, 512)),
        "w_v": (w_v, (heads, 128, 512)),
        "gate_logits": (gate_logits, (batch, tq, heads, 128)),
    }
    for name, (value, shape) in tensors.items():
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape{shape}")
        if value.dtype != q_content.dtype or value.device != q_content.device or not value.is_contiguous():
            raise ValueError(f"{name} must match query dtype/device and be contiguous")
    if not isinstance(metadata, torch.Tensor) or tuple(metadata.shape) != (batch, 8):
        raise ValueError("metadata must have shape[B,8]")
    if metadata.dtype != torch.int32 or metadata.device != q_content.device or not metadata.is_contiguous():
        raise ValueError("metadata must be contiguous device int32")
    if out is None:
        out = torch.empty((batch, tq, heads, 128), dtype=q_content.dtype, device=q_content.device)
    if not isinstance(out, torch.Tensor) or tuple(out.shape) != (batch, tq, heads, 128):
        raise ValueError("out shape must match the nonempty output")
    if out.dtype != q_content.dtype or out.device != q_content.device or not out.is_contiguous():
        raise ValueError("out must match query dtype/device and be contiguous")
    for name, (value, _) in tensors.items():
        if value.numel() and out.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
            raise ValueError(f"out must not alias read-only input {name}")
    if out.untyped_storage().data_ptr() == metadata.untyped_storage().data_ptr():
        raise ValueError("out must not alias read-only metadata")
    dtype = "float16" if q_content.dtype == torch.float16 else "bfloat16"
    kernel = build_empty_cache_zero(batch=batch, query_capacity=tq, heads=heads, dtype=dtype, target=target)
    kernel(q_content, out)
    return (out, kernel) if return_kernel else out
