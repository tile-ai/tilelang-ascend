# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import ctypes
import os
import subprocess
from typing import Dict, List, Optional, Tuple

_HOST_SUFFIX_GET_KERNEL_NUM_ARGS = "_get_kernel_num_args"
_HOST_SUFFIX_GET_KERNEL_ARG_TYPE = "_get_kernel_arg_type"
_HOST_SUFFIX_INFER_WORKSPACE_SHAPE = "_infer_workspace_shape_function"

_NM_TIMEOUT_SEC = 2
_DEFAULT_WORKSPACE_SIZE = 32768

# Type encoding (must stay in sync with WrapHostFunction.cpp ElementTypeCode)

_WRAPHOST_POINTER_FLAG = 0x80
_WRAPHOST_TYPE_CODE_MASK = 0x7F
_WRAPHOST_UNKNOWN_TYPE_CODE = 0x7F
_WRAPHOST_TYPE_CODE_TO_SIG: Dict[int, str] = {
    0: "fp32",
    1: "fp16",
    2: "bf16",
    3: "i8",
    4: "i16",
    5: "i32",
    6: "i64",
    7: "u32",
    8: "u64",
    9: "fp64",
    10: "i1",
}


class _NpuSoHostProbe:
    """Facade for probing compiled ``.so`` host callbacks (internal API)."""

    @staticmethod
    def _find_symbols(lib_path: str, suffix: str) -> list:
        """List dynamic symbol names in ``lib_path`` that end with ``suffix`` (``nm -D``)."""
        symbols = []
        if not os.path.exists(lib_path):
            return symbols
        try:
            result = subprocess.run(
                ["nm", "-D", lib_path],
                capture_output=True,
                text=True,
                timeout=_NM_TIMEOUT_SEC,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 3 and parts[2].endswith(suffix):
                        symbols.append(parts[2])
        except (subprocess.SubprocessError, FileNotFoundError, OSError, TimeoutError):
            pass
        return symbols

    @staticmethod
    def _call_so_symbol(lib_path: str, func_name: str, restype=ctypes.c_int):
        """``CDLL`` + call exported ``func_name``."""
        try:
            lib = ctypes.CDLL(lib_path)
            func = getattr(lib, func_name)
            func.restype = restype
            return func()
        except (OSError, AttributeError, TypeError):
            return None

    @staticmethod
    def _pick_num_args_symbol(
        lib_path: str, num_args_symbols: List[str]
    ) -> Optional[str]:
        """Pick a deterministic ``*_get_kernel_num_args`` symbol."""
        if not num_args_symbols:
            return None
        type_symbols = set(
            _NpuSoHostProbe._find_symbols(lib_path, _HOST_SUFFIX_GET_KERNEL_ARG_TYPE)
        )
        for sym_name in sorted(num_args_symbols):
            base_name = sym_name[: -len(_HOST_SUFFIX_GET_KERNEL_NUM_ARGS)]
            if base_name + _HOST_SUFFIX_GET_KERNEL_ARG_TYPE in type_symbols:
                return sym_name
        return sorted(num_args_symbols)[0]

    @staticmethod
    def _get_kernel_signature(
        lib_path: str,
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict[int, str]]]:
        """Extract kernel symbol name, base name and signature from compiled ``.so``."""
        num_args_syms = _NpuSoHostProbe._find_symbols(
            lib_path, _HOST_SUFFIX_GET_KERNEL_NUM_ARGS
        )
        if not num_args_syms:
            return None, None, None

        sym_name = _NpuSoHostProbe._pick_num_args_symbol(lib_path, num_args_syms)
        if sym_name is None:
            return None, None, None
        kernel_name = sym_name[: -len(_HOST_SUFFIX_GET_KERNEL_NUM_ARGS)]
        base_name = kernel_name
        if base_name.endswith("_mix_aic") or base_name.endswith("_mix_aiv"):
            base_name = base_name[:-8]
        type_func_name = kernel_name + _HOST_SUFFIX_GET_KERNEL_ARG_TYPE

        num_args = _NpuSoHostProbe._call_so_symbol(lib_path, sym_name, ctypes.c_int)
        if num_args is None:
            return kernel_name, base_name, None

        try:
            lib = ctypes.CDLL(lib_path)
            type_func = getattr(lib, type_func_name)
            type_func.restype = ctypes.c_int
            type_func.argtypes = [ctypes.c_int]
        except (OSError, AttributeError):
            return kernel_name, base_name, None

        sig: Dict[int, str] = {}
        for i in range(num_args):
            type_code = type_func(i)
            is_ptr = bool(type_code & _WRAPHOST_POINTER_FLAG)
            elem_code = type_code & _WRAPHOST_TYPE_CODE_MASK
            type_str = _WRAPHOST_TYPE_CODE_TO_SIG.get(elem_code)
            if elem_code == _WRAPHOST_UNKNOWN_TYPE_CODE or type_str is None:
                raise RuntimeError(
                    "Unknown kernel arg type code from host symbol: "
                    f"func={type_func_name}, arg_index={i}, type_code={type_code}, "
                    f"elem_code={elem_code}, so_path={lib_path}"
                )
            sig[i] = f"*{type_str}" if is_ptr else type_str
        return kernel_name, base_name, sig

    @staticmethod
    def _get_workspace_size(
        lib_path: str, default: int = _DEFAULT_WORKSPACE_SIZE
    ) -> int:
        """Resolve workspace size via ``*_infer_workspace_shape_function`` exports."""
        symbols = _NpuSoHostProbe._find_symbols(
            lib_path, _HOST_SUFFIX_INFER_WORKSPACE_SHAPE
        )
        if not symbols:
            return default
        for func_name in symbols:
            val = _NpuSoHostProbe._call_so_symbol(lib_path, func_name, ctypes.c_int)
            if val is not None:
                return val
        return default
