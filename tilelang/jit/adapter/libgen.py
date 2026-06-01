# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
from __future__ import annotations
from tilelang import tvm as tvm
import ctypes
import os
import tempfile
import subprocess
import logging
from pathlib import Path
from tilelang.env import TILELANG_TEMPLATE_PATH, TILELANG_PACKAGE_PATH

logger = logging.getLogger(__name__)


def _get_tl_root(third_party_name="3rdparty") -> str:
    """Get TL_ROOT path, fallback to package path if not set."""
    tl_root = os.environ.get("TL_ROOT")
    if tl_root is not None:
        if not (Path(tl_root) / third_party_name).exists():
            raise ValueError(f"3rdparty dependencies not found. TL_ROOT is {tl_root!r}")
        return tl_root

    for path in [TILELANG_PACKAGE_PATH, TILELANG_PACKAGE_PATH.parent]:
        if (path / third_party_name).exists():
            return str(path)
    # In most situations, (path / third_party_name) should exists.
    # Otherwise, "import tvm" will fail before reaching this point.
    raise ValueError(
        f"TL_ROOT is not set and {third_party_name}/ directory not found in "
        f"{TILELANG_PACKAGE_PATH} or {TILELANG_PACKAGE_PATH.parent}. "
        "Please set the TL_ROOT environment variable."
    )


def _get_ascend_home_path() -> str:
    """Get Ascend home path, with fallback options."""
    ascend_home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_HOME")
    if ascend_home is None:
        potential_paths = [
            "/usr/local/Ascend/ascend-toolkit/latest",
        ]
        for path in potential_paths:
            if os.path.exists(path):
                ascend_home = path
                break
    if ascend_home is None:
        raise ValueError("ASCEND_HOME_PATH or ASCEND_HOME is not set. Please set the environment variable.")
    return ascend_home


def _get_simulator_lib_path(ascend_home: str, platform: str) -> str:
    """Get the camodel simulator library path for the given platform.

    Returns the directory containing libruntime_camodel.so.
    """
    sim_base = os.path.join(ascend_home, "tools", "simulator")
    if not os.path.isdir(sim_base):
        sim_base = os.path.join(ascend_home, "simulator")

    if platform == "A5":
        soc_candidates = ["Ascend950PR_9599", "Ascend910_9599"]
    else:
        soc_candidates = ["Ascend910B1", "Ascend910_9599"]

    for soc in soc_candidates:
        soc_dir = os.path.join(sim_base, soc)
        if not os.path.isdir(soc_dir):
            continue
        # The lib/ directory contains required config.json + *.toml files.
        # The camodel/ directory lacks these configs, causing TMultiRing crash.
        candidate = os.path.join(soc_dir, "lib")
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        f"Simulator library not found for platform {platform}. "
        f"Searched: {sim_base}/<soc>/lib and {sim_base}/<soc>/camodel"
    )


class LibraryGenerator:
    srcpath: str | None = None
    libpath: str | None = None
    lib_code: str | None = None

    def __init__(self, target: str, platform: str):
        self.target = target
        self.platform = platform

    def update_lib_code(self, lib_code: str):
        self.lib_code = lib_code

    # Assume currently we only support CUDA compilation
    def load_lib(self, lib_path: str | None = None):
        if lib_path is None:
            lib_path = self.libpath
        run_mode = os.environ.get("TL_RUN_MODE", "npu")
        if run_mode == "sim":
            ascend_home = _get_ascend_home_path()
            sim_lib_path = _get_simulator_lib_path(ascend_home, self.platform)
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if sim_lib_path not in ld_path.split(":"):
                os.environ["LD_LIBRARY_PATH"] = f"{sim_lib_path}:{ld_path}"
        return ctypes.CDLL(lib_path)

    def compile_lib(self, timeout: float = None):
        src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)
        libpath = src.name.replace(".cpp", ".so")
        ASCEND_HOME_PATH = _get_ascend_home_path()
        TL_ROOT = _get_tl_root()
        if self.target == "ascendc" or self.target == "auto":
            command = [
                "bisheng",
                "--npu-arch=dav-2201",
                "-O2",
                "-std=c++17",
                "-xasc",
                f"-I{ASCEND_HOME_PATH}/include",
                f"-I{ASCEND_HOME_PATH}/include/experiment/msprof",
                f"-I{ASCEND_HOME_PATH}/include/experiment/runtime",
                f"-I{ASCEND_HOME_PATH}/pkg_inc",
                f"-I{ASCEND_HOME_PATH}/pkg_inc/runtime",
                f"-I{ASCEND_HOME_PATH}/pkg_inc/profiling",
                f"-I{TL_ROOT}/3rdparty/catlass/include",
                f"-I{TL_ROOT}/3rdparty/shmem/include",
                f"-I{TL_ROOT}/3rdparty/shmem/src/device",
                "-DBACKEND_HYBM",
                "-I" + TILELANG_TEMPLATE_PATH,
                f"-L{ASCEND_HOME_PATH}/lib64",
                "-Wno-macro-redefined",
                "-Wno-ignored-attributes",
                "-Wno-non-c-typedef-for-linkage",
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
        elif self.target == "pto":
            ccec = "dav-c310" if self.platform == "A5" else "dav-c220"
            memory = "REGISTER_BASE" if self.platform == "A5" else "MEMORY_BASE"
            command = [
                "bisheng",
                f"--cce-aicore-arch={ccec}",
                f"-D{memory}",
                "-O2",
                "-std=gnu++17",
                "-xcce",
                "-mllvm",
                "-cce-aicore-stack-size=0x8000",
                "-mllvm",
                "-cce-aicore-function-stack-size=0x8000",
                "-mllvm",
                "-cce-aicore-record-overflow=true",
                "-mllvm",
                "-cce-aicore-addr-transform",
                "-mllvm",
                "-cce-aicore-dcci-insert-for-scalar=false",
                "-DL2_CACHE_HINT",
                "-I../../src/",
                f"-I{TL_ROOT}/3rdparty/pto-isa/include",
                f"-I{ASCEND_HOME_PATH}/include",
                f"-I{ASCEND_HOME_PATH}/include/experiment/msprof",
                f"-I{ASCEND_HOME_PATH}/include/experiment/runtime",
                "-I/usr/local/Ascend/driver/kernel/inc",
                f"-I{ASCEND_HOME_PATH}/pkg_inc",
                f"-I{ASCEND_HOME_PATH}/pkg_inc/runtime",
                f"-I{ASCEND_HOME_PATH}/pkg_inc/profiling",
                f"-L{ASCEND_HOME_PATH}/lib64",
                "-I" + TILELANG_TEMPLATE_PATH,
                "-Wno-macro-redefined",
                "-Wno-ignored-attributes",
                "-lruntime",
                "-lstdc++",
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
            if os.environ.get("TL_PTO_DEBUG") == "1":
                command += ["-D_DEBUG", "--cce-enable-print"]

        # --- camodel (simulator) support ---
        run_mode = os.environ.get("TL_RUN_MODE", "npu")
        if run_mode == "sim":
            sim_lib_path = _get_simulator_lib_path(ASCEND_HOME_PATH, self.platform)
            # Insert simulator library path before ASCEND_HOME_PATH/lib64 so
            # libruntime_camodel.so takes precedence over libruntime.so.
            # --disable-new-dtags makes -rpath set DT_RPATH (transitive) instead
            # of DT_RUNPATH (non-transitive), so that libruntime_camodel.so's
            # own dependencies (e.g. libnpu_drv_camodel.so) are also resolved.
            try:
                ascend_lib_idx = command.index(f"-L{ASCEND_HOME_PATH}/lib64")
                command.insert(ascend_lib_idx, f"-L{sim_lib_path}")
                command.insert(ascend_lib_idx + 1, f"-Wl,-rpath,{sim_lib_path}")
                command.insert(ascend_lib_idx + 2, "-Wl,--disable-new-dtags")
            except ValueError:
                command.insert(1, f"-L{sim_lib_path}")
                command.insert(2, f"-Wl,-rpath,{sim_lib_path}")
                command.insert(3, "-Wl,--disable-new-dtags")
            # Replace '-lruntime' with '-lruntime_camodel'
            try:
                rt_idx = command.index("-lruntime")
                command[rt_idx] = "-lruntime_camodel"
            except ValueError:
                pass
            logger.info("camodel sim mode: using %s", sim_lib_path)

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
