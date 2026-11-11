"""Synchronous, single-thread/same-stream owner for contiguous compressed caches.

No host payload copy/fill, no device-length readback, and no asynchronous lifetime
claim. The caller must use returned cache handles read-only on the bound stream.
"""

from dataclasses import dataclass
import threading

import torch

from .kernels.cache import append_cache_kernel, initialize_cache_kernel
from ._cache_state import append_transition, read_transition, reject_storage_aliases
from ._packed import require_inference
from .layout import require_dense_base

DTYPES = {torch.float16: "float16", torch.bfloat16: "bfloat16"}


@dataclass(frozen=True)
class CacheRead:
    cache_c: torch.Tensor
    cache_e: torch.Tensor
    metadata: torch.Tensor
    lengths: tuple[int, ...]
    generation: int
    metadata_h2d_bytes: int


class CacheOwner:
    def __init__(self, B, capacity, dtype=torch.float16, *, device="npu:0", target="pto"):
        if any(type(v) is not int or v <= 0 for v in (B, capacity)):
            raise ValueError("B and capacity must be positive host integers")
        if B * capacity * 576 > 2**31 - 1:
            raise ValueError("cache offsets exceed int32 indexing domain")
        if dtype not in DTYPES:
            raise TypeError("cache dtype must be FP16 or BF16")
        if target not in ("pto", "ascendc"):
            raise ValueError("target must be pto or ascendc")
        self.device = torch.device(device)
        if self.device.type not in ("npu", "privateuseone") or self.device.index is None:
            raise ValueError("an explicit indexed NPU device is required")
        if torch.npu.current_device() != self.device.index:
            raise ValueError("select the intended NPU device before creating the owner")
        self.B, self.capacity, self.dtype, self.target = B, capacity, dtype, target
        self._thread = threading.get_ident()
        self._stream = torch.npu.current_stream(self.device)
        self._lengths = (0,) * B
        self._generation = 0
        self._valid = False
        self.launches = 0
        self._cache_c = torch.empty((B, capacity, 512), dtype=dtype, device=self.device)
        self._cache_e = torch.empty((B, capacity, 64), dtype=dtype, device=self.device)
        self._caches()
        kernel = initialize_cache_kernel(B, capacity, DTYPES[dtype], target=target)
        self._invoke_and_wait(kernel, self._cache_c, self._cache_e)
        self._valid = True

    @property
    def cache_c(self):
        return self._cache_c

    @property
    def cache_e(self):
        return self._cache_e

    @property
    def lengths(self):
        return self._lengths

    @property
    def generation(self):
        return self._generation

    def _context(self):
        if not self._valid:
            raise RuntimeError("cache owner is invalid after a device operation error")
        if threading.get_ident() != self._thread:
            raise RuntimeError("cache owner only supports its original CPU thread")
        if torch.npu.current_device() != self.device.index:
            raise RuntimeError("cache owner requires its original NPU device")
        if torch.npu.current_stream(self.device).npu_stream != self._stream.npu_stream:
            raise RuntimeError("cache owner requires its original NPU stream")

    def _tensor(self, name, value, shape):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if tuple(value.shape) != shape or value.dtype != self.dtype or value.device != self.device:
            raise ValueError(f"{name} must be {shape}, {self.dtype}, {self.device}")
        require_dense_base(value, name=name, shape=shape, dtype=self.dtype, device=self.device)

    def _caches(self):
        self._tensor("CacheC", self._cache_c, (self.B, self.capacity, 512))
        self._tensor("CacheE", self._cache_e, (self.B, self.capacity, 64))
        reject_storage_aliases(self._cache_c, self._cache_e)

    def _invoke_and_wait(self, kernel, *inputs):
        try:
            kernel(*inputs)
            self.launches += 1
            self._stream.synchronize()
        except BaseException:
            self._valid = False
            raise

    def _upload_metadata(self, metadata_cpu):
        # Explicit small H2D control transfer, never a cache payload copy.
        try:
            return metadata_cpu.to(device=self.device, non_blocking=False)
        except BaseException:
            self._valid = False
            raise

    def _read_handle(self, metadata):
        return CacheRead(self._cache_c, self._cache_e, metadata, self._lengths, self._generation, self.B * 8 * 4)

    def append(self, c_norm, raw_kv, q_lens, *, metadata_cpu=None):
        require_inference()
        self._context()
        self._caches()
        if not isinstance(c_norm, torch.Tensor) or c_norm.ndim != 3:
            raise ValueError("Cnorm must be [B,Qcap,512]")
        Qcap = int(c_norm.shape[1])
        if Qcap <= 0 or self.B * Qcap * 576 > 2**31 - 1:
            raise ValueError("Qcap must be positive and within int32 indexing limits")
        self._tensor("Cnorm", c_norm, (self.B, Qcap, 512))
        self._tensor("rawKV", raw_kv, (self.B, Qcap, 576))
        reject_storage_aliases(self._cache_c, self._cache_e, (c_norm, raw_kv))
        cpu_meta, next_lengths = append_transition(self._lengths, q_lens, Qcap=Qcap, capacity=self.capacity, metadata_cpu=metadata_cpu)
        # No data transfer or launch happens before all checks above finish.
        kernel = append_cache_kernel(self.B, Qcap, self.capacity, DTYPES[self.dtype], target=self.target)
        device_meta = self._upload_metadata(cpu_meta)
        self._invoke_and_wait(kernel, c_norm, raw_kv, device_meta, self._cache_c, self._cache_e)
        self._lengths = next_lengths
        self._generation += 1
        return self._read_handle(device_meta)

    def read_existing(self, q_lens, query_start, *, Qcap):
        require_inference()
        self._context()
        self._caches()
        cpu_meta = read_transition(self._lengths, q_lens, query_start, Qcap=Qcap, capacity=self.capacity)
        metadata = self._upload_metadata(cpu_meta)
        try:
            self._stream.synchronize()
        except BaseException:
            self._valid = False
            raise
        return self._read_handle(metadata)

    def reset(self):
        raise NotImplementedError("reset/slot reuse is not implemented; create a new owner after caller-owned consumers finish")

    def grow(self, capacity):
        raise NotImplementedError("cache growth is not implemented")
