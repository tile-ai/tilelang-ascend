# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Official gate source with a direct Triton launcher; not vLLM host timing."""

from typing import Any
import torch
import torch_npu  # noqa: F401 - registers the NPU backend
import triton
import triton.language as tl

HAS_TRITON = True


SMALL_GATE_ELEMENTS = 8192


MEDIUM_GATE_ELEMENTS = 65536


LARGE_GATE_ELEMENTS = 1048576


TRITON_POINTER_ALIGNMENT = 16


@triton.jit(do_not_specialize=["numel"])
def _flash_attention_gate_kernel(
    projected,
    gate,
    numel,
    HIDDEN: tl.constexpr,
    PROJECTED_STRIDE: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    CONTIGUOUS: tl.constexpr = PROJECTED_STRIDE == HIDDEN and GATE_STRIDE == HIDDEN
    if CONTIGUOUS:
        tasks = tl.cdiv(numel, BLOCK)
    else:
        tasks = numel // HIDDEN * tl.cdiv(HIDDEN, BLOCK)
    for block in range(tl.program_id(0), tasks, tl.num_programs(0)):
        if CONTIGUOUS:
            # Cross row boundaries without vector division or partial row tiles.
            offsets = block * BLOCK + tl.arange(0, BLOCK)
            projected_offsets = offsets
            gate_offsets = offsets
            mask = offsets < numel
        else:
            row = block // tl.cdiv(HIDDEN, BLOCK)
            columns = block % tl.cdiv(HIDDEN, BLOCK) * BLOCK + tl.arange(0, BLOCK)
            projected_offsets = row * PROJECTED_STRIDE + columns
            gate_offsets = row * GATE_STRIDE + columns
            mask = columns < HIDDEN
        values = tl.load(projected + projected_offsets, mask, other=0)
        gates = tl.load(gate + gate_offsets, mask, other=0).to(tl.float32)
        # Preserve the intermediate rounding of torch.sigmoid(gate).
        scale = tl.sigmoid(gates).to(gate.dtype.element_ty)
        tl.store(projected + projected_offsets, values * scale, mask)


def _flash_attention_gate_block_size(numel: int) -> int:
    # Small decode tiles limit vector work; larger tiles amortize prefill loops.
    if numel <= SMALL_GATE_ELEMENTS:
        return 512
    if numel < MEDIUM_GATE_ELEMENTS:
        return 2048
    if numel < LARGE_GATE_ELEMENTS:
        return 4096
    return 8192


def flash_attention_gate(projected: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply output * sigmoid(gate) in place, preserving sigmoid rounding."""
    if projected.ndim != 2 or gate.ndim != 2:
        raise ValueError("Projected and gate tensors must be 2-D")
    if projected.shape != gate.shape:
        raise ValueError(f"Gate shape {gate.shape} must match projected shape {projected.shape}")
    if projected.dtype != gate.dtype:
        raise ValueError("Projected and gate dtypes must match")
    if projected.numel() == 0:
        return projected
    if projected.stride(1) != 1 or gate.stride(1) != 1:
        raise ValueError("Projected and gate columns must be contiguous")
    init_device_properties_triton()
    numel = projected.numel()
    block = _flash_attention_gate_block_size(numel)
    if projected.stride(0) == gate.stride(0) == projected.shape[1]:
        tasks = triton.cdiv(numel, block)
    else:
        tasks = projected.shape[0] * triton.cdiv(projected.shape[1], block)
    grid = (min(tasks, get_vectorcore_num()),)
    _FLASH_ATTENTION_GATE_KERNEL(
        projected=projected,
        gate=gate,
        numel=numel,
        HIDDEN=projected.shape[1],
        BLOCK=block,
        grid=grid,
    )
    return projected


_NUM_AICORE = -1


_NUM_VECTORCORE = -1


_UB_SIZE_BYTES = -1


DEFAULT_UB_SIZE_BYTES = 192 * 1024


_MIN_UB_SIZE_BYTES = 4 * 1024


_UB_SIZE_PROPERTY_KEYS = (
    "ub_size",
    "max_ub_size",
    "unified_buffer_size",
    "max_shared_memory",
    "max_shared_mem",
)


def init_device_properties_triton():
    global _NUM_AICORE, _NUM_VECTORCORE, _UB_SIZE_BYTES
    if _NUM_AICORE == -1 and HAS_TRITON:
        device_properties: dict[str, Any] = triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())
        _NUM_AICORE = device_properties.get("num_aicore", -1)
        _NUM_VECTORCORE = device_properties.get("num_vectorcore", -1)
        assert _NUM_AICORE > 0 and _NUM_VECTORCORE > 0, "Failed to detect device properties."

        # Detect UB size: try each candidate key from device properties.
        # Reject values below _MIN_UB_SIZE_BYTES: some driver versions return
        # sentinels (e.g., 1) when UB info is unavailable, which would produce
        # absurdly small tiles and kernel compilation failures.
        for key in _UB_SIZE_PROPERTY_KEYS:
            value = device_properties.get(key)
            if isinstance(value, (int, float)) and value >= _MIN_UB_SIZE_BYTES:
                _UB_SIZE_BYTES = int(value)
                break
        # Fall back to safe default if not found or invalid.
        if _UB_SIZE_BYTES <= 0:
            _UB_SIZE_BYTES = DEFAULT_UB_SIZE_BYTES


def get_vectorcore_num():
    global _NUM_VECTORCORE
    assert _NUM_VECTORCORE > 0, "Device properties not initialized. Please call init_device_properties_triton() first."
    return _NUM_VECTORCORE


class _DirectTritonLauncher:
    """Replace only vLLM warmup/CompileKey dispatch; launch the real original kernel."""

    def __call__(self, *, projected, gate, numel, HIDDEN, BLOCK, grid):
        return _flash_attention_gate_kernel[grid](
            projected=projected,
            gate=gate,
            numel=numel,
            HIDDEN=HIDDEN,
            BLOCK=BLOCK,
            PROJECTED_STRIDE=projected.stride(0),
            GATE_STRIDE=gate.stride(0),
            multibuffer=False,
        )


_FLASH_ATTENTION_GATE_KERNEL = _DirectTritonLauncher()
