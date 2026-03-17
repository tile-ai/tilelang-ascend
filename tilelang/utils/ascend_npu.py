# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
# Copied from bitblas
import functools
import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_VALID_NPU_TARGETS = frozenset({
    "Ascend910B1", "Ascend910B2", "Ascend910B3", "Ascend910B4",
    "Ascend910B4-1", "Ascend910B2C",
    "Ascend910_9362", "Ascend910_9372",
    "Ascend910_9381", "Ascend910_9382",
    "Ascend910_9391", "Ascend910_9392",
    "Ascend910_950z", "Ascend910_9579",
    "Ascend910_957b", "Ascend910_957d",
    "Ascend910_9581", "Ascend910_9589",
    "Ascend910_958a", "Ascend910_958b",
    "Ascend910_9599",
})

_DEFAULT_NPU_TARGET = "Ascend910B1"

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


def _try_detect_npu_target_from_cann_rt() -> "str | None":
    """Detect SOC version via CANN runtime ``rtGetSocVersion`` (ctypes).

    This mirrors triton-ascend's primary detection approach and returns the
    exact SOC variant string (e.g. "Ascend910B3") without depending on
    ``npu_utils.so`` or ``torch_npu``.
    """
    try:
        import ctypes
        ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
        search_names = ["libruntime.so"]
        if ascend_home:
            search_names.insert(0, os.path.join(ascend_home, "lib64", "libruntime.so"))
        librt = None
        for name in search_names:
            try:
                librt = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if librt is None:
            return None
        buf = ctypes.create_string_buffer(64)
        ret = librt.rtGetSocVersion(buf, ctypes.c_uint32(64))
        if ret != 0:
            return None
        soc = buf.value.decode("utf-8").strip()
        if soc and soc in _VALID_NPU_TARGETS:
            return soc
    except Exception as e:
        logger.debug("CANN runtime detection failed: %s", e)
    return None


def _try_detect_npu_target_from_smi() -> "str | None":
    """Detect SOC version via ``npu-smi info`` command."""
    npu_smi = shutil.which("npu-smi")
    if npu_smi is None:
        return None
    try:
        out = subprocess.check_output(
            [npu_smi, "info", "-t", "board", "-i", "0"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if "SOC Version" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    soc = parts[-1].strip()
                    if soc in _VALID_NPU_TARGETS:
                        return soc
    except Exception as e:
        logger.debug("npu-smi detection failed: %s", e)
    return None


@functools.lru_cache()
def detect_npu_target() -> str:
    """Auto-detect the Ascend NPU target device for compilation.

    Detection priority (mirrors triton-ascend's approach):
      1. ``TILELANG_NPU_TARGET`` environment variable (e.g. "Ascend910B3")
      2. CANN runtime ``rtGetSocVersion`` via ctypes — most accurate
      3. ``npu-smi info`` command-line tool
      4. Fallback: ``Ascend910B1``
    """
    env = os.environ.get("TILELANG_NPU_TARGET", "").strip()
    if env:
        if env in _VALID_NPU_TARGETS:
            logger.debug("NPU target from env: %s", env)
            return env
        logger.warning(
            "TILELANG_NPU_TARGET=%r is not a recognized target; "
            "valid values: %s. Falling back to auto-detection.",
            env, ", ".join(sorted(_VALID_NPU_TARGETS)),
        )

    detected = _try_detect_npu_target_from_cann_rt()
    if detected is not None:
        logger.debug("NPU target from CANN rtGetSocVersion: %s", detected)
        return detected

    detected = _try_detect_npu_target_from_smi()
    if detected is not None:
        logger.debug("NPU target from npu-smi: %s", detected)
        return detected

    logger.debug("NPU target detection failed, using default: %s", _DEFAULT_NPU_TARGET)
    return _DEFAULT_NPU_TARGET


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