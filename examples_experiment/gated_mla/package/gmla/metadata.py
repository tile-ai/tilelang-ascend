"""CPU-owned runtime metadata v1; never infer lengths from device data."""

import torch


def build_runtime_metadata(q_lens, kv_lens, query_start, *, B, Tq, Tk):
    """CPU-owner construction of v1 int32[B,8]; never accepts device tensors."""
    if any(type(x) is not int or x < 0 or x > 2**31 - 1 for x in (B, Tq, Tk)) or B == 0:
        raise ValueError("B must be positive and capacities must be nonnegative int32 host integers")
    fields = (q_lens, kv_lens, query_start)
    for values in fields:
        if not isinstance(values, (tuple, list)) or len(values) != B or any(type(x) is not int for x in values):
            raise TypeError("metadata fields must be B-element host tuples/lists of Python integers")
    rows = []
    for q, k, start in zip(*fields):
        if not (0 <= q <= Tq and 0 <= k <= Tk and start >= 0 and start + q <= k):
            raise ValueError("invalid q_lens, kv_lens or query_start for the declared capacities")
        rows.append([q, k, start, 0, 0, 0, 0, 0])
    return torch.tensor(rows, dtype=torch.int32, device="cpu")


def validate_runtime_metadata_cpu(metadata, *, B, Tq, Tk):
    """Validate only CPU-owned contents, before any H2D transfer is enqueued."""
    if not isinstance(metadata, torch.Tensor) or metadata.device.type != "cpu":
        raise TypeError("metadata content validation requires a CPU tensor; device readback is forbidden")
    if metadata.dtype != torch.int32 or tuple(metadata.shape) != (B, 8) or not metadata.is_contiguous():
        raise ValueError("metadata must be contiguous CPU int32[B,8]")
    if torch.count_nonzero(metadata[:, 3:]).item() != 0:
        raise ValueError("reserved metadata columns 3..7 must be zero")
    expected = build_runtime_metadata(metadata[:, 0].tolist(), metadata[:, 1].tolist(), metadata[:, 2].tolist(), B=B, Tq=Tq, Tk=Tk)
    if not torch.equal(expected, metadata):
        raise ValueError("metadata does not match the v1 schema")


def validate_device_metadata_layout(metadata, *, B, device):
    """Inspect device tensor metadata only; validity of its contents is caller-owned."""
    if not isinstance(metadata, torch.Tensor):
        raise TypeError("device metadata must be a Tensor")
    if metadata.device != torch.device(device) or metadata.dtype != torch.int32:
        raise ValueError("device metadata must have the requested device and int32 dtype")
    if tuple(metadata.shape) != (B, 8) or not metadata.is_contiguous():
        raise ValueError("device metadata must be contiguous [B,8]")


def metadata_for_reference(cpu_metadata):
    """Convert explicitly CPU metadata for the independent mathematical goldens."""
    if not isinstance(cpu_metadata, torch.Tensor) or cpu_metadata.device.type != "cpu":
        raise ValueError("reference metadata must already be on CPU")
    if cpu_metadata.dtype != torch.int32 or cpu_metadata.ndim != 2 or cpu_metadata.shape[1] != 8:
        raise ValueError("reference metadata must have int32 shape[B,8]")
    rows = cpu_metadata.tolist()
    return {"q_lens": tuple(r[0] for r in rows), "kv_lens": tuple(r[1] for r in rows), "query_start": tuple(r[2] for r in rows)}


def prepare_metadata(rows, *, batch, query_capacity, cache_capacity):
    """Return validated CPU int32[B,8]; caller explicitly transfers control data."""
    if not isinstance(rows, (tuple, list)):
        raise TypeError("rows must be CPU-owned Python rows")
    if len(rows) != batch:
        raise ValueError("rows must contain one CPU-owned row per batch")
    q_lens, kv_lens, starts = [], [], []
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) not in (3, 8) or any(type(x) is not int for x in row):
            raise TypeError("rows require3 or8 Python integers")
        if len(row) == 8 and any(row[3:]):
            raise ValueError("reserved metadata columns must be zero")
        q_lens.append(row[0])
        kv_lens.append(row[1])
        starts.append(row[2])
    return build_runtime_metadata(q_lens, kv_lens, starts, B=batch, Tq=query_capacity, Tk=cache_capacity)


def pack_metadata(rows, *, batch, capacity, query_capacity=1):
    """Compatibility spelling used by the retained internal runtime/fixtures."""
    return prepare_metadata(rows, batch=batch, query_capacity=query_capacity, cache_capacity=capacity)
