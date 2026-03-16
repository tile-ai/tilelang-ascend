# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
# Copied from bitblas
import functools
import os
import shutil
from pathlib import Path

class NPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance
    
    def __init__(self) -> None:
        # TODO: change to use cache, non-fixed directory (Finish before 330)
        fname = "npu_utils.so"
        import importlib.util
        spec = importlib.util.spec_from_file_location("npu_utils", fname)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.npu_utils_mod = mod
    
    def load_binary(self, name, kernel, shared, device, mix_mode):
        return self.npu_utils_mod.load_kernel_binary(
            name, kernel, shared, device, mix_mode
        )
    
    @functools.lru_cache()
    def get_arch(self):
        # Return Ascend soc version
        return self.npu_utils_mod.get_arch()
    
    @functools.lru_cache()
    def get_aicore_num(self):
        # Return Ascend soc aicore number
        return self.npu_utils_mod.get_aicore_num()
    
    @functools.lru_cache()
    def get_aivector_core_num(self):
        # Return Ascend soc vector core number
        return self.get_aicore_num() * 2
    
    @functools.lru_cache()
    def get_aicube_core_num(self):
        # Return Ascend soc cube core number
        return self.get_aicore_num()
    
    @functools.lru_cache()
    def get_device_num(self):
        # Return Ascend device number
        return self.npu_utils_mod.get_device_num()


@functools.lru_cache()
def get_ascend_path() -> Path:
    """Get CANN root directory"""
    path = os.getenv("ASCEND_HOME_PATH", "")
    if path == "":
        raise EnvironmentError(
            "ASCEND_HOME_PATH is not set, source <ascend-toolkit>/set_env.sh first"
        )
    return path


@functools.lru_cache()
def get_cxx():
    """Get C++ compiler"""
    cxx = os.environ.get("CC", "")
    if cxx == "":
        clangxx = shutil.which("clang++")
        if clangxx is not None:
            return clangxx
        gxx = shutil.which("g++")
        if gxx is not None:
            return gxx
        raise RuntimeError("Failed to find C++ compiler")
    return cxx


@functools.lru_cache()
def get_npucompiler_path():
    """Get bishengir-compile"""
    npu_compiler_path = shutil.which("bishengir-compile")
    if npu_compiler_path is None:
        npu_compiler_root = os.getenv("TILELANG_NPU_COMPILER_PATH", "")
        if npu_compiler_root == "":
            raise EnvironmentError(
                "Couldn't find executable bishengir-compile or TILELANG_NPU_COMPILER_PATH."
            )
        npu_compiler_path = os.path.join(npu_compiler_root, "npuc")
    return npu_compiler_path


@functools.lru_cache()
def get_hivmc_path():
    """Get hivmc (HIVM binary compiler). Used when lower() returns HIVM-optimized IR."""
    hivmc_path = shutil.which("hivmc")
    if hivmc_path is not None:
        return hivmc_path
    npu_compiler_path = get_npucompiler_path()
    if npu_compiler_path is not None:
        install_dir = os.path.dirname(npu_compiler_path)
        candidate = os.path.join(install_dir, "hivmc")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    bisheng_install = os.getenv("BISHENG_INSTALL_PATH", "")
    if bisheng_install:
        candidate = os.path.join(bisheng_install, "hivmc")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise EnvironmentError(
        "Couldn't find executable hivmc (check PATH, bishengir-compile dir, or BISHENG_INSTALL_PATH)."
    )


@functools.lru_cache()
def get_npucompiler_opt_path():
    """Get bishengir-opt"""
    npu_compiler_opt_path = shutil.which("bishengir-opt")
    if npu_compiler_opt_path is None:
        raise EnvironmentError(
            "Couldn't find executable bishengir-opt."
        )
    return npu_compiler_opt_path


@functools.lru_cache()
def get_bisheng_path():
    """Get bisheng"""
    bisheng_path = shutil.which("bisheng")
    if bisheng_path is None:
        npu_compiler_root = os.getenv("TILELANG_NPU_COMPILER_PATH", "")
        if npu_compiler_root == "":
            raise EnvironmentError(
                "Couldn't find executable bisheng or TILELANG_NPU_COMPILER_PATH"
            )
        bisheng_path = os.path.join(npu_compiler_root, "ccec")
    return bisheng_path