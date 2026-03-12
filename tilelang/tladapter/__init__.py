# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import os
import sys
import ctypes

from tilelang import libinfo

_native = None

def _load_tilelangir_passes_so():
    global _native
    lib_path = libinfo.find_lib_path("tilelangir", optional=False)
    path = lib_path[0]
    ctypes.CDLL(path)
    import importlib.util
    from importlib.machinery import ExtensionFileLoader
    mod_name = "tilelangir"
    loader = ExtensionFileLoader(mod_name, path)
    spec = importlib.util.spec_from_loader(mod_name, loader)
    if spec is None:
        raise RuntimeError(
            f"TileLangIR: failed to create module spec for {path!r}. "
            "Pass pipeline will not be available."
        )
    _native_mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = _native_mod
    spec.loader.exec_module(_native_mod)
    _native = _native_mod
    sys.modules[__name__]._native = _native_mod


_load_tilelangir_passes_so()

import importlib
utils = importlib.import_module("tilelang.tladapter.utils")
transforms = importlib.import_module("tilelang.tladapter.transforms")
conversion = importlib.import_module("tilelang.tladapter.conversion")

__all__ = ["utils", "transforms", "conversion"]
