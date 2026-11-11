"""CPU-authoritative append/read metadata transitions; no device value reads."""

import torch

from .metadata import build_runtime_metadata, validate_runtime_metadata_cpu


def append_transition(lengths, q_lens, *, Qcap, capacity, metadata_cpu=None):
    B = len(lengths)
    if not isinstance(q_lens, (tuple, list)) or len(q_lens) != B or any(type(x) is not int for x in q_lens):
        raise TypeError("q_lens must be B host integers")
    next_lengths = tuple(old + count for old, count in zip(lengths, q_lens))
    expected = build_runtime_metadata(q_lens, list(next_lengths), list(lengths), B=B, Tq=Qcap, Tk=capacity)
    if metadata_cpu is not None:
        validate_runtime_metadata_cpu(metadata_cpu, B=B, Tq=Qcap, Tk=capacity)
        if not torch.equal(metadata_cpu, expected):
            raise ValueError("append metadata must start at owner old lengths and satisfy kv_len=start+q_len")
    return expected, next_lengths


def read_transition(lengths, q_lens, query_start, *, Qcap, capacity):
    return build_runtime_metadata(q_lens, list(lengths), query_start, B=len(lengths), Tq=Qcap, Tk=capacity)


def reject_storage_aliases(cache_c, cache_e, sources=()):
    caches = (cache_c, cache_e)
    identities = [(value.device, value.untyped_storage().data_ptr()) for value in caches]
    if identities[0] == identities[1]:
        raise ValueError("CacheC and CacheE must not alias storage")
    for source in sources:
        if (source.device, source.untyped_storage().data_ptr()) in identities:
            raise ValueError("append source must not alias cache storage")
