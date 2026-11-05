"""A2/A3 annotations from TA 0deb8b93da11b7f0becbcded732d9633b7ca4e34."""

from numbers import Number

from tvm.runtime import Object, ObjectConvertible, const
from tvm.tirx.script.builder.ir import sblock_attr as block_attr

__all__ = ["annotate_address"]


def _address_value(address):
    if isinstance(address, Number):
        return const(address)
    # Preserve TA's frontend rejection of unknown Python objects. The new Any
    # annotation map can otherwise carry them until a native typed-map cast.
    if address is None or isinstance(address, (Object, ObjectConvertible, str, list, tuple, dict)) or callable(address):
        return address
    raise ValueError(f"don't know how to convert type {type(address)} to object")


def annotate_address(address_map: dict):
    # TA's Object conversion wrapped numeric literals as PrimExpr. TIRX's
    # generic annotation map retains primitives, so preserve that typed ABI here.
    address_map = {buffer.data: _address_value(address) for buffer, address in address_map.items()}
    return block_attr({"address_map": address_map})
