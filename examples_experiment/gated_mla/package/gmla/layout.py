"""Read-only physical-format guard for raw-pointer TileLang NPU consumers.

No global option changes, format casts, tensor copies or device-value readbacks.
The base set follows the installed torch_npu source, using public enum names.
"""

import torch


class LayoutContractError(ValueError):
    pass


class InternalFormatError(LayoutContractError):
    pass


def base_formats(backend):
    # Ascend/pytorch@d99d6cda: FormatHelper::IsOpInputBaseFormat, L144-151.
    # Never replace this with a guessed numeric list or all enum values.
    formats = backend.Format
    return frozenset((formats.ND, formats.NCHW, formats.NHWC, formats.NCDHW))


def raw_pointer_formats(backend):
    # This operator's current rank2-4 producer/storage contract is narrower
    # than the framework's generic base-format set. NHWC/NCDHW need separate
    # physical raw-copy validation before admitting them here.
    return frozenset((backend.Format.ND, backend.Format.NCHW))


def require_dense_base(tensor, *, name="tensor", shape=None, dtype=None, device=None):
    """Validate physical base format and logical dense layout without changing data."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if tensor.device.type not in ("npu", "privateuseone"):
        raise LayoutContractError(f"{name} must be an NPU tensor")
    if tensor.layout != torch.strided:
        raise LayoutContractError(f"{name} must have strided Tensor layout")
    import torch_npu

    try:
        physical_format = torch_npu.get_npu_format(tensor)
        allowed = base_formats(torch_npu)
    except (AttributeError, ValueError, RuntimeError) as error:
        raise LayoutContractError(f"{name}: unable to establish a supported physical NPU base format") from error
    if physical_format not in allowed:
        raise InternalFormatError(
            f"{name} has internal/non-base physical format {physical_format}; raw-pointer kernels require a base format. "
            "This guard does not convert or copy the tensor."
        )
    if physical_format not in raw_pointer_formats(torch_npu):
        raise LayoutContractError(f"{name} base format {physical_format} is outside this operator's current ND/NCHW raw-pointer contract")
    if not tensor.is_contiguous():
        raise LayoutContractError(f"{name} must be logically contiguous in row-major order")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise LayoutContractError(f"{name} must have shape {tuple(shape)}")
    if dtype is not None and tensor.dtype != dtype:
        raise LayoutContractError(f"{name} must have dtype {dtype}")
    if device is not None and tensor.device != torch.device(device):
        raise LayoutContractError(f"{name} must be on device {device}")
    offset = tensor.storage_offset()
    required_bytes = (offset + tensor.numel()) * tensor.element_size()
    if offset < 0 or required_bytes > tensor.untyped_storage().nbytes():
        raise LayoutContractError(f"{name} dense logical span exceeds its storage")
    return tensor


def guarded_view(tensor, shape, *, name="tensor"):
    """Base-format check precedes view; verify the result really shares storage.

    require_dense_base itself is metadata-only. This separate convenience helper
    performs a logical view, never a format cast or a contiguous-copy fallback.
    """
    require_dense_base(tensor, name=name)
    before = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tensor.data_ptr(), tensor.dtype, tensor.device)
    result = tensor.view(*shape)
    require_dense_base(result, name=name + " view", shape=shape)
    after = (result.untyped_storage().data_ptr(), result.storage_offset(), result.data_ptr(), result.dtype, result.device)
    if after != before:
        raise LayoutContractError(f"{name} view did not preserve storage/offset/device/dtype; no fallback is permitted")
    return result
