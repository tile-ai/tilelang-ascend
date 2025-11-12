# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
from typing import Optional
from tilelang import tvm as tvm
import ctypes
import os
import tempfile
import subprocess
import logging
from tilelang.env import TILELANG_TEMPLATE_PATH

logger = logging.getLogger(__name__)


class LibraryGenerator(object):
    srcpath: Optional[str] = None
    libpath: Optional[str] = None
    lib_code: Optional[str] = None

    def __init__(self, target: str):
        self.target = target

    def update_lib_code(self, lib_code: str):
        self.lib_code = lib_code

    # Assume currently we only support CUDA compilation
    def load_lib(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = self.libpath
        return ctypes.CDLL(lib_path)

    def compile_lib(self, timeout: float = None):
        src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)
        libpath = src.name.replace(".cpp", ".so")
        ASCEND_HOME_PATH = os.environ["ASCEND_HOME_PATH"]
        TL_ROOT = os.environ["TL_ROOT"]
        command = [
            "bisheng",
            "--npu-arch=dav-2201",
            "-O2",
            "-std=c++17",
            "-xasc",
            f"-I{ASCEND_HOME_PATH}/include",
            f"-I{ASCEND_HOME_PATH}/include/experiment/msprof",
            f"-I{ASCEND_HOME_PATH}/include/experiment/runtime",
            f"-I{TL_ROOT}/3rdparty/catlass/include",
            "-I" + TILELANG_TEMPLATE_PATH,
            f"-L{ASCEND_HOME_PATH}/lib64",
            "-Wno-macro-redefined",
            "-Wno-ignored-attributes",
            "-lruntime",
            "-lascendcl",
            "-lm",
            "-ltiling_api",
            "-lplatform",
            "-lc_sec",
            "-ldl",
            "-fPIC",
            "--shared",
            src.name,
        ]
        command += ["-o", libpath]

        src.write(self.lib_code)
        src.flush()
        try:
            ret = subprocess.run(command, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Compile kernel failed because of {e}") from e

        if ret.returncode != 0:
            raise RuntimeError(f"Compilation Failed! {command}")

        self.srcpath = src.name
        self.libpath = libpath

    def remove_lib(self):
        if self.libpath:
            os.remove(self.libpath)
        self.libpath = None

    def get_source_path(self):
        return self.srcpath

    def get_lib_path(self):
        return self.libpath

    def set_lib_path(self, libpath):
        self.libpath = libpath

    def set_src_path(self, srcpath):
        self.srcpath = srcpath
