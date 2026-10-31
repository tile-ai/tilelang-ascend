"""Lazy bridge for optional ``tvm.info.mem.<scope>`` metadata callbacks."""

from tilelang import tvm


def make_memory_info(unit_bits, max_num_bits, max_simd_bits, head_address=None):
    """Create the backend-local equivalent of TA's four-field MemoryInfo.

    Use this object as a callback result. Importing this module does not look up
    an Ascend type or native function, including in USE_ASCEND=OFF packages.
    """
    return tvm.ffi.get_global_func("tl.ascend.a2a3.MakeMemoryInfo")(unit_bits, max_num_bits, max_simd_bits, head_address)
