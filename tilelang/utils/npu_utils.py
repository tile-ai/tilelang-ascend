# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
# Copied from bitblas
import functools
import os
import shutil
import sysconfig
from pathlib import Path
import torch
import torch_npu
import subprocess
import tempfile
import uuid
import logging
from hashlib import sha256
from tilelang import env

import pybind11


class NPUUtils(object):
    """Singleton helper for Ascend NPU utilities.

    The object compiles and loads a small shared library on first use, and
    caches the resulting module in ``self.npu_utils_mod``.  Subsequent calls to
    ``NPUUtils()`` will return the same instance and will not repeat the
    compilation or module import.

    Use ``NPUUtils.get()`` to obtain the singleton, or simply call the
    constructor; both behave identically but ``get`` makes the intent clearer.
    """

    _initialized = False

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        # initialization is relatively expensive (compiling/loading shared object),
        # so skip if we've already run once for this process.  The ``__new__``
        # method already ensures the same instance is returned; ``_initialized``
        # will prevent repeated work on subsequent calls.
        if self._initialized:
            return

        pkg_root = os.path.dirname(os.path.abspath(__file__))
        npu_utils_cpp = os.path.join(pkg_root, "npu_utils.cpp")
        fname_path = "npu_utils.so"
        if os.path.exists(npu_utils_cpp):
            cache_path = get_runtime_file_cache(npu_utils_cpp)
            fname_path = os.path.join(cache_path, "npu_utils.so")
            # protect against stale or empty files; compile only if missing or zero-sized
            if not (os.path.exists(fname_path) and os.path.getsize(fname_path) > 0):
                # compile npu_utils.so
                with tempfile.TemporaryDirectory() as tmpdir:
                    dst_path = os.path.join(tmpdir, "npu_utils.cxx")
                    safe_copy(npu_utils_cpp, dst_path)
                    so = build_npu_ext(
                        "npu_utils", None, dst_path, kernel_launcher="torch"
                    )
                    safe_copy(so, fname_path)
        else:
            raise FileNotFoundError(f"Could not find npu_utils.cpp at {npu_utils_cpp}.")
        import importlib.util

        spec = importlib.util.spec_from_file_location("npu_utils", str(fname_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.npu_utils_mod = mod
        # mark that initialization is complete, so the next time __init__ is
        # invoked we can return immediately.
        self._initialized = True

    @classmethod
    def get(cls):
        """Return the singleton instance.

        This is just a thin wrapper around the constructor that makes the intent
        explicit and hides the fact that ``__init__`` is idempotent.
        """
        return cls()

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


def safe_copy(src: str, dst: str, tmp_dir: str = None):
    """Atomic copy functions resolve write conflicts in multi-process environments.

    Args:
        src: source file path
        dst: target file path
        tmp_dir: Temporary file directory (the system temporary directory is used by default)
    """
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()

    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    temp_filename = f"{os.getpid()}_{uuid.uuid4()}_{os.path.basename(dst)}"
    temp_path = os.path.join(tmp_dir, temp_filename)

    try:
        shutil.copy(src, temp_path)
        # Use atomic POSIX replace, so other processes cannot see a partial write
        os.replace(temp_path, dst)
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                logging.warning(f"Failed to clean temp file: {temp_path}")


def get_runtime_file_cache(source):
    """
    Return the path to cache runtime files, such as npu_utils or precompiled headers.

    Args:
        source: Either a file path (str or Path) or file content (bytes).
                If a file path is provided, the file's content is read and hashed.
                If content is provided directly, it is hashed directly.
                This ensures consistent caching: identical content always produces the same cache path.

    Returns:
        str: The cache directory path.
    """
    if isinstance(source, (str, Path)) and os.path.isfile(str(source)):
        # source is a file path, read its content
        with open(source, "rb") as f:
            content = f.read()
    elif isinstance(source, bytes):
        # source is content directly
        content = source
    else:
        raise ValueError("source must be a file path (str or Path) or content (bytes)")

    hashvalue = sha256(content)
    key = hashvalue.hexdigest()
    cache_dir = os.path.join(env.TILELANG_CACHE_DIR, key)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


@functools.lru_cache()
def get_ascend_path() -> Path:
    """Get ASCEND_HOME_PATH root directory"""
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


def get_cxx_precompiled(header_path):
    cc_cmd = []
    cxx = os.environ.get("CC")
    if cxx is None:
        clangxx = shutil.which("clang++")
        if clangxx is not None:
            cc_cmd += [clangxx, "-include", header_path]
            return cc_cmd
        gxx = shutil.which("g++")
        if gxx is not None:
            cc_cmd += [gxx]
        else:
            raise RuntimeError("Failed to find C++ compiler")
    else:
        cc_cmd += [cxx]
    return cc_cmd


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
def get_npucompiler_opt_path():
    """Get bishengir-opt"""
    npu_compiler_opt_path = shutil.which("bishengir-opt")
    if npu_compiler_opt_path is None:
        raise EnvironmentError("Couldn't find executable bishengir-opt.")
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


@functools.lru_cache()
def get_torch_cxx_abi():
    return 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0


def get_torch_npu_cc_cmd(build_pch):
    torch_path = os.path.dirname(os.path.realpath(torch.__file__))
    torch_npu_path = os.path.dirname(os.path.realpath(torch_npu.__file__))
    cc_cmd = [
        f"-I{os.path.join(torch_path, 'include')}",
        f"-I{os.path.join(torch_npu_path, 'include')}",
        f"-D_GLIBCXX_USE_CXX11_ABI={get_torch_cxx_abi()}",
    ]
    if not build_pch:
        cc_cmd += [f"-L{os.path.join(torch_npu_path, 'lib')}", "-ltorch_npu"]
    return cc_cmd


@functools.lru_cache()
def get_default_scheme():
    """Get sysconfig default scheme"""
    if hasattr(sysconfig, "get_default_scheme"):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use tilelang with system-wide python.
    if scheme == "posix_local":
        scheme = "posix_prefix"
    return scheme


@functools.lru_cache()
def get_npu_launcher_header():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "npu_launcher.h")


def precompile_npu_ext(header_path, gch_path):
    cxx = get_cxx()
    cc_cmd = [cxx, "-x", "c++-header", header_path]
    # disable all warnings
    cc_cmd += ["-w"]
    # find the python library
    scheme = get_default_scheme()
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    cc_cmd += [f"-I{py_include_dir}"]
    # device_print.h
    cc_cmd += [f"-I{os.path.dirname(os.path.realpath(__file__))}"]
    # find the ascend library
    asc_path = get_ascend_path()
    rt_path = os.path.join(asc_path, "include/experiment/runtime/runtime/rt.h")
    if not os.path.exists(rt_path):
        cc_cmd += [
            f"-I{os.path.join(asc_path, 'pkg_inc')}",
            f"-I{os.path.join(asc_path, 'pkg_inc/profiling')}",
        ]
    cc_cmd += [
        f"-I{os.path.join(asc_path, 'include')}",
        f"-I{os.path.join(asc_path, 'include/experiment')}",
        f"-I{os.path.join(asc_path, 'include/experiment/msprof')}",
        f"-I{pybind11.get_include()}",
    ]
    cc_cmd += get_torch_npu_cc_cmd(build_pch=True)
    cc_cmd += ["-std=c++17", "-shared", "-fPIC", "-o", gch_path]
    result = subprocess.run(cc_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return header_path
    else:
        raise RuntimeError(
            f"Failed to compile {gch_path}, error: {result.stderr}, cmd={cc_cmd}"
        )


def build_npu_ext(
    obj_name: str, header_path, src_path, *, kernel_launcher="torch", precompile=False
) -> str:
    # TODO: change to use Cache before 330
    so_path = f"{obj_name}.so"
    if precompile:
        cc_cmd = get_cxx_precompiled(header_path)
        cc_cmd += [src_path]
    else:
        cxx = get_cxx()
        cc_cmd = [cxx, src_path]
    # disable all warnings
    cc_cmd += ["-w"]
    # find python library
    scheme = get_default_scheme()
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    cc_cmd += [f"-I{py_include_dir}"]
    # device_print.h
    cc_cmd += [f"-I{os.path.dirname(os.path.realpath(__file__))}"]
    # find the ascend library
    asc_path = get_ascend_path()
    if header_path is not None:
        cc_cmd += [f"-I{os.path.dirname(header_path)}"]

    rt_path = os.path.join(asc_path, "include/experiment/runtime/runtime/rt.h")
    if not os.path.exists(rt_path):
        cc_cmd += [
            f"-I{os.path.join(asc_path, 'pkg_inc')}",
            f"-I{os.path.join(asc_path, 'pkg_inc/profiling')}",
        ]
    cc_cmd += [
        f"-I{os.path.join(asc_path, 'include')}",
        f"-I{os.path.join(asc_path, 'include/experiment')}",
        f"-I{os.path.join(asc_path, 'include/experiment/msprof')}",
        f"-I{pybind11.get_include()}",
        f"-L{os.path.join(asc_path, 'lib64')}",
        "-lruntime",
        "-lascendcl",
    ]
    if kernel_launcher == "torch":
        cc_cmd += get_torch_npu_cc_cmd(build_pch=False)
    cc_cmd += ["-std=c++17", "-shared", "-fPIC", "-Winvalid-pch", "-o", so_path]
    result = subprocess.run(cc_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        return so_path
    else:
        if "npu_launcher.h.gch" in result.stderr:
            # only for clang++, when precompile invalid, fallback to normal compile
            return build_npu_ext(obj_name, header_path, src_path, precompile=False)
        else:
            raise RuntimeError(
                f"Failed to compile {src_path}, error: {result.stderr}, cmd={cc_cmd}"
            )
