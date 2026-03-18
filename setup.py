# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.

import io
import subprocess
import shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
from setuptools.command.develop import develop
from setuptools.command.bdist_wheel import bdist_wheel
import distutils.dir_util
from typing import List
import re
import tarfile
from io import BytesIO
import os
import sys
import urllib.request
from distutils.version import LooseVersion
import platform
import multiprocessing
from setuptools.command.build_ext import build_ext
import importlib
import logging

# Configure logging with basic settings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Environment variables False/True
PYPI_BUILD = os.environ.get("PYPI_BUILD", "False").lower() == "true"
PACKAGE_NAME = "tilelang"
ROOT_DIR = os.path.dirname(__file__)

# Add LLVM control environment variable
USE_LLVM = os.environ.get("USE_LLVM", "False").lower() == "true"
# Add ROCM control environment variable
USE_ROCM = os.environ.get("USE_ROCM", "False").lower() == "true"
# Add NPUIR control environment variable
USE_NPUIR = os.environ.get("USE_NPUIR", "False").lower() == "true"


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


envs = load_module_from_path("env", os.path.join(ROOT_DIR, PACKAGE_NAME, "env.py"))

CUDA_HOME = envs.CUDA_HOME
ROCM_HOME = envs.ROCM_HOME

# If USE_NPUIR, skip CUDA/ROCM
if USE_NPUIR:
    logger.info("NPUIR support is enabled. CUDA/ROCM detection may be skipped.")
elif USE_ROCM and not ROCM_HOME:
    raise ValueError(
        "ROCM support is enabled (USE_ROCM=True) but ROCM_HOME is not set or detected."
    )

if not USE_ROCM and not CUDA_HOME and not USE_NPUIR:
    raise ValueError(
        "Failed to automatically detect CUDA or ROCM installation and NPUIR is not enabled."
    )

# TileLang only supports Linux platform
assert sys.platform.startswith("linux"), (
    "TileLang only supports Linux platform (including WSL)."
)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_requirements(file_path: str = "requirements.txt") -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path(file_path)) as f:
        requirements = f.read().strip().split("\n")
    requirements = [
        r.strip()
        for r in requirements
        if r.strip()
        and not r.strip().startswith("#")
        and not r.strip().startswith("--")
    ]
    return requirements


def find_version(version_file_path: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    # Read and store the version information from the VERSION file
    # Use 'strip()' to remove any leading/trailing whitespace or newline characters
    if not os.path.exists(version_file_path):
        raise FileNotFoundError(f"Version file not found at {version_file_path}")
    with open(version_file_path, "r") as version_file:
        version = version_file.read().strip()
    return version


def get_nvcc_cuda_version():
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output(["nvcc", "-V"], universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = LooseVersion(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_rocm_version():
    """Get the ROCM version from rocminfo."""
    rocm_output = subprocess.check_output(["rocminfo"], universal_newlines=True)
    # Parse ROCM version from output
    # Example output: ROCM version: x.y.z-...
    match = re.search(r"ROCm Version: (\d+\.\d+\.\d+)", rocm_output)
    if match:
        return LooseVersion(match.group(1))
    else:
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        rocm_version_file = os.path.join(
            rocm_path, "lib", "cmake", "rocm", "rocm-config-version.cmake"
        )
        if os.path.exists(rocm_version_file):
            with open(rocm_version_file, "r") as f:
                content = f.read()
                match = re.search(r'set\(PACKAGE_VERSION "(\d+\.\d+\.\d+)"', content)
                if match:
                    return LooseVersion(match.group(1))
    # return a default
    return LooseVersion("5.0.0")


def get_tilelang_version(
    with_cuda=True, with_system_info=True, with_commit_id=False
) -> str:
    version = find_version(get_path(".", "VERSION"))
    local_version_parts = []
    if with_system_info:
        local_version_parts.append(get_system_info().replace("-", "."))

    if USE_NPUIR:
        local_version_parts.append("npuir")
    else:
        if USE_ROCM:
            if ROCM_HOME:
                rocm_version = str(get_rocm_version())
                rocm_version_str = rocm_version.replace(".", "")[:3]
                local_version_parts.append(f"rocm{rocm_version_str}")
        else:
            if CUDA_HOME:
                cuda_version = str(get_nvcc_cuda_version())
                cuda_version_str = cuda_version.replace(".", "")[:3]
                local_version_parts.append(f"cu{cuda_version_str}")

    if local_version_parts:
        version += f"+{'.'.join(local_version_parts)}"

    if with_commit_id:
        commit_id = None
        try:
            commit_id = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
            ).strip()
        except subprocess.SubprocessError as error:
            raise RuntimeError("Failed to get git commit id") from error
        if commit_id:
            version += f"+{commit_id}"

    return version


def get_system_info():
    system = platform.system().lower()
    if system == "linux":
        try:
            with open("/etc/os-release") as f:
                os_release = f.read()
            version_id_match = re.search(r'VERSION_ID="(\d+\.\d+)"', os_release)
            if version_id_match:
                version_id = version_id_match.group(1)
                distro = "ubuntu"
                return f"{distro}-{version_id}"
        except FileNotFoundError:
            pass
    return system


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


def download_and_extract_llvm(version, is_aarch64=False, extract_path="3rdparty"):
    """
    Downloads and extracts the specified version of LLVM for the given platform.
    Args:
        version (str): The version of LLVM to download.
        is_aarch64 (bool): True if the target platform is aarch64, False otherwise.
        extract_path (str): The directory path where the archive will be extracted.

    Returns:
        str: The path where the LLVM archive was extracted.
    """
    ubuntu_version = "16.04"
    if version >= "16.0.0":
        ubuntu_version = "20.04"
    elif version >= "13.0.0":
        ubuntu_version = "18.04"

    base_url = (
        f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{version}"
    )
    file_name = f"clang+llvm-{version}-{'aarch64-linux-gnu' if is_aarch64 else f'x86_64-linux-gnu-ubuntu-{ubuntu_version}'}.tar.xz"

    download_url = f"{base_url}/{file_name}"

    # Download the file
    logger.info(f"Downloading {file_name} from {download_url}")
    with urllib.request.urlopen(download_url) as response:
        if response.status != 200:
            raise Exception(f"Download failed with status code {response.status}")
        file_content = response.read()
    # Ensure the extract path exists
    os.makedirs(extract_path, exist_ok=True)

    # if the file already exists, remove it
    if os.path.exists(os.path.join(extract_path, file_name)):
        os.remove(os.path.join(extract_path, file_name))

    # Extract the file
    logger.info(f"Extracting {file_name} to {extract_path}")
    with tarfile.open(fileobj=BytesIO(file_content), mode="r:xz") as tar:
        tar.extractall(path=extract_path)

    logger.info("Download and extraction completed successfully.")
    return os.path.abspath(os.path.join(extract_path, file_name.replace(".tar.xz", "")))


package_data = {
    "tilelang": ["py.typed", "*pyx"],
}

LLVM_VERSION = "10.0.1"
IS_AARCH64 = False  # Set to True if on an aarch64 platform
EXTRACT_PATH = "3rdparty"  # Default extraction path


def update_submodules():
    """Updates git submodules if in a git repository."""

    def is_git_repo():
        try:
            # Check if current directory is a git repository
            subprocess.check_output(
                ["git", "rev-parse", "--is-inside-work-tree"], stderr=subprocess.STDOUT
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    if not is_git_repo():
        logger.info("Info: Not a git repository, skipping submodule update.")
        return

    try:
        subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Failed to update submodules") from error


def build_csrc(llvm_config_path):
    """Configures and builds TVM."""

    if not os.path.exists("build"):
        os.makedirs("build")
    os.chdir("build")
    # Copy the config.cmake as a baseline
    if not os.path.exists("config.cmake"):
        shutil.copy("../3rdparty/tvm/cmake/config.cmake", "config.cmake")
    # Set LLVM path and enable CUDA or ROCM in config.cmake
    with open("config.cmake", "a") as config_file:
        config_file.write(f"set(USE_LLVM {llvm_config_path})\n")
        if USE_ROCM:
            config_file.write(f"set(USE_ROCM {ROCM_HOME})\n")
            config_file.write("set(USE_CUDA OFF)\n")
        else:
            config_file.write(f"set(USE_CUDA {CUDA_HOME})\n")
            config_file.write("set(USE_ROCM OFF)\n")
    # Run CMake and make
    try:
        subprocess.check_call(["cmake", ".."])
        num_jobs = max(1, int(multiprocessing.cpu_count() * 0.75))
        subprocess.check_call(["make", f"-j{num_jobs}"])
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Failed to build TileLang C Source") from error


def setup_llvm_for_tvm():
    """Downloads and extracts LLVM, then configures TVM to use it."""
    # Assume the download_and_extract_llvm function and its dependencies are defined elsewhere in this script
    extract_path = download_and_extract_llvm(LLVM_VERSION, IS_AARCH64, EXTRACT_PATH)
    llvm_config_path = os.path.join(extract_path, "bin", "llvm-config")
    return extract_path, llvm_config_path


def patch_libs(libpath):
    """
    tvm and tilelang libs are copied from elsewhere into wheels
    and have a hard-coded rpath.
    Set rpath to the directory of libs so auditwheel works well.
    """
    # check if patchelf is installed
    # find patchelf in the system
    patchelf_path = shutil.which("patchelf")
    if not patchelf_path:
        logger.warning(
            "patchelf is not installed, which is required for auditwheel to work for compatible wheels."
        )
        return
    subprocess.run([patchelf_path, "--set-rpath", "$ORIGIN", libpath])


class TileLangBuilPydCommand(build_py):
    """Customized setuptools install command - builds TVM after setting up LLVM."""

    def run(self):
        build_py.run(self)
        self.run_command("build_ext")
        build_ext_cmd = self.get_finalized_command("build_ext")
        build_temp_dir = build_ext_cmd.build_temp
        ext_modules = build_ext_cmd.extensions
        for ext in ext_modules:
            extdir = build_ext_cmd.get_ext_fullpath(ext.name)
            logger.info(f"Extension {ext.name} output directory: {extdir}")

        ext_output_dir = os.path.dirname(extdir)
        logger.info(f"Extension output directory (parent): {ext_output_dir}")
        logger.info(f"Build temp directory: {build_temp_dir}")

        # copy cython files
        CYTHON_SRC = [
            "tilelang/jit/adapter/cython/cython_wrapper.pyx",
        ]
        for item in CYTHON_SRC:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, item)
            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                distutils.dir_util.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                if not os.path.exists(
                    os.path.join(target_dir, os.path.basename(source_dir))
                ):
                    shutil.copy2(source_dir, target_dir)

        # copy the tl_templates
        TILELANG_SRC = [
            "src/tl_templates",
        ]
        for item in TILELANG_SRC:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, item)
            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                distutils.dir_util.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(source_dir, target_dir)

        # Packing npu_utils.cpp and npu_launcher.h file
        TILELANG_FILE_SRC = [
            "utils/npu_launcher.h",
            "utils/npu_utils.cpp",
        ]
        for item in TILELANG_FILE_SRC:
            source_path = os.path.join(ROOT_DIR, "tilelang", item)
            target_path = os.path.join(self.build_lib, PACKAGE_NAME, item)
            if not os.path.isfile(source_path):
                logger.warning(f"Source file {source_path} does not exist, skipping.")
                continue
            target_dir = os.path.dirname(target_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            shutil.copy2(source_path, target_path)

        TVM_PREBUILD_ITEMS = [
            "libtvm_runtime.so",
            "libtvm.so",
            "libtilelang.so",
            "libtilelang_module.so",
            "libtilelangir.so",
        ]

        potential_dirs = [
            ext_output_dir,
            self.build_lib,
            build_temp_dir,
            os.path.join(ROOT_DIR, "build"),
            os.path.join(ROOT_DIR, "build/tvm"),
            os.path.join(ROOT_DIR, "build", "tilelangir"),
        ]

        tvm_prebuild_path = os.environ.get("TVM_PREBUILD_PATH")
        if tvm_prebuild_path:
            potential_dirs.insert(0, os.path.abspath(tvm_prebuild_path))

        for item in TVM_PREBUILD_ITEMS:
            source_lib_file = None
            for dir in potential_dirs:
                candidate = os.path.join(dir, item)
                if os.path.exists(candidate):
                    source_lib_file = candidate
                    break

            if source_lib_file:
                patch_libs(source_lib_file)
                target_dir_release = os.path.join(self.build_lib, PACKAGE_NAME, "lib")
                target_dir_develop = os.path.join(PACKAGE_NAME, "lib")
                os.makedirs(target_dir_release, exist_ok=True)
                os.makedirs(target_dir_develop, exist_ok=True)
                shutil.copy2(source_lib_file, target_dir_release)
                logger.info(f"Copied {source_lib_file} to {target_dir_release}")
                shutil.copy2(source_lib_file, target_dir_develop)
                logger.info(f"Copied {source_lib_file} to {target_dir_develop}")
                os.remove(source_lib_file)
            else:
                logger.info(f"WARNING: {item} not found in any expected directories!")

        # Bundle MLIR Python (mlir_core + bishengir) for NPUIR when source has both. Required when present.
        npuir_python_base = os.path.join(
            self.build_lib, PACKAGE_NAME, "lib", "npuir_python"
        )
        bundle_src = None
        if os.path.isdir(os.path.join(ROOT_DIR, "build", "tilelangir", "mlir_core")):
            bundle_src = os.path.join(ROOT_DIR, "build", "tilelangir")
        elif os.environ.get("BISHENGIR_ROOT_PATH"):
            pp = os.path.join(os.environ["BISHENGIR_ROOT_PATH"], "python_packages")
            if os.path.isdir(os.path.join(pp, "mlir_core")):
                bundle_src = pp
        if bundle_src:
            for sub in ("mlir_core", "bishengir"):
                src = os.path.join(bundle_src, sub)
                if not os.path.isdir(src):
                    raise SystemExit(
                        f"NPUIR bundle requires both mlir_core and bishengir; missing {sub} under {bundle_src}"
                    )
            for sub in ("mlir_core", "bishengir"):
                src = os.path.join(bundle_src, sub)
                dst = os.path.join(npuir_python_base, sub)
                self.mkpath(dst)
                distutils.dir_util.copy_tree(src, dst)
                logger.info(f"Bundled NPUIR Python: {src} -> {dst}")

        TVM_CONFIG_ITEMS = [
            f"{build_temp_dir}/config.cmake",
        ]
        for item in TVM_CONFIG_ITEMS:
            source_dir = os.path.join(ROOT_DIR, item)
            file_name = os.path.basename(item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, file_name)
            target_dir = os.path.dirname(target_dir)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            if os.path.exists(source_dir):
                shutil.copy2(source_dir, target_dir)
            else:
                logger.info(f"INFO: {source_dir} does not exist.")

        TVM_PACAKGE_ITEMS = [
            "3rdparty/tvm/src",
            "3rdparty/tvm/python",
            "3rdparty/tvm/licenses",
            "3rdparty/tvm/conftest.py",
            "3rdparty/tvm/CONTRIBUTORS.md",
            "3rdparty/tvm/KEYS",
            "3rdparty/tvm/LICENSE",
            "3rdparty/tvm/README.md",
            "3rdparty/tvm/mypy.ini",
            "3rdparty/tvm/pyproject.toml",
            "3rdparty/tvm/version.py",
        ]
        for item in TVM_PACAKGE_ITEMS:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, item)
            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                distutils.dir_util.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(source_dir, target_dir)

        # Copy CUTLASS to the package directory
        if CUDA_HOME:
            CUTLASS_PREBUILD_ITEMS = [
                "3rdparty/cutlass/include",
                "3rdparty/cutlass/tools",
            ]
            for item in CUTLASS_PREBUILD_ITEMS:
                source_dir = os.path.join(ROOT_DIR, item)
                target_dir = os.path.join(self.build_lib, PACKAGE_NAME, item)
                if os.path.isdir(source_dir):
                    self.mkpath(target_dir)
                    distutils.dir_util.copy_tree(source_dir, target_dir)
                else:
                    target_dir = os.path.dirname(target_dir)
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    shutil.copy2(source_dir, target_dir)

        # copy composable kernel to the package directory
        CK_PREBUILD_ITEMS = [
            "3rdparty/composable_kernel/include",
            "3rdparty/composable_kernel/library",
        ]
        for item in CK_PREBUILD_ITEMS:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, item)
            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                distutils.dir_util.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(source_dir, target_dir)

        # copy config files to the package directory
        TL_CONFIG_ITEMS = ["CMakeLists.txt", "VERSION", "README.md", "LICENSE"]
        for item in TL_CONFIG_ITEMS:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, item)
            if not PYPI_BUILD and item == "VERSION":
                version = get_tilelang_version(
                    with_cuda=False, with_system_info=False, with_commit_id=True
                )
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                with open(os.path.join(target_dir, item), "w") as f:
                    print(f"Writing {version} to {os.path.join(target_dir, item)}")
                    f.write(version)
                continue

            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                distutils.dir_util.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(source_dir, target_dir)

        self.remove_unwanted_dirs()
        # ===== Critical fixes: Patch TVM and __init__.py =====
        # Apply patches after all files are copied
        self.patch_tvm_base_py()
        self.patch_init_py()

    def remove_unwanted_dirs(self):
        """Force remove test/unused directories from build_lib"""
        unwanted_dirs = [
            "testing",
            "unittest",
            "examples",
            "benchmark",
            "docs",
        ]

        for dir_name in unwanted_dirs:
            dir_path = os.path.join(self.build_lib, PACKAGE_NAME, dir_name)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logger.info(f"Removed {dir_path}")

    def patch_tvm_base_py(self):
        """Patch TVM's base.py to use the bundled libtvm.so"""
        base_py_path = os.path.join(
            self.build_lib,
            PACKAGE_NAME,
            "3rdparty",
            "tvm",
            "python",
            "tvm",
            "_ffi",
            "base.py",
        )

        if not os.path.exists(base_py_path):
            logger.warning(f"base.py not found at {base_py_path}, skipping patch")
            return

        with open(base_py_path, "r") as f:
            content = f.read()

        if "# --- Patched by TileLang: Force use bundled libtvm.so ---" in content:
            logger.info("base.py already patched, skipping")
            return

        patch = """\
# --- Patched by TileLang: Force use bundled libtvm.so ---
import os, sys, ctypes
_current_dir = os.path.dirname(os.path.abspath(__file__))
_tilelang_root = os.path.abspath(os.path.join(_current_dir, *['..'] * 4))
_lib_path = os.path.join(_tilelang_root, 'lib', 'libtvm.so')
if os.path.exists(_lib_path):
    try:
        _lib = ctypes.CDLL(_lib_path, ctypes.RTLD_GLOBAL)
        os.environ['TVM_LIBRARY_PATH'] = os.path.dirname(_lib_path)
        _LIB = _lib
    except Exception as e:
        print(f"[TileLang] Failed to load bundled TVM library: {e}")
# --------------------------------------------------------
"""

        with open(base_py_path, "w") as f:
            f.write(patch + content)
        logger.info(f" Patched {base_py_path} to use bundled libtvm.so")

    def patch_init_py(self):
        """Patch tilelang/__init__.py to set up TVM paths properly"""
        target_init = os.path.join(self.build_lib, PACKAGE_NAME, "__init__.py")
        if not os.path.exists(target_init):
            logger.warning(f"__init__.py not found at {target_init}, skipping patch")
            return

        with open(target_init, "r") as f:
            content = f.read()

        # check the patch
        if "# --- Built-in TVM support ---" in content:
            logger.info("__init__.py already patched, skipping")
            return

        patch = """\
# --- Built-in TVM support ---
import sys, os
_tvm_python_path = os.path.join(os.path.dirname(__file__), '3rdparty', 'tvm', 'python')
if os.path.exists(_tvm_python_path) and _tvm_python_path not in sys.path:
    sys.path.insert(0, _tvm_python_path)
_lib_path = os.path.join(os.path.dirname(__file__), 'lib')
if os.path.exists(_lib_path):
    os.environ['TVM_LIBRARY_PATH'] = _lib_path
try:
    import tvm
except ImportError as e:
    pass
# -----------------------------
"""

        with open(target_init, "w") as f:
            f.write(patch + content)
        logger.info("Patched __init__.py for built-in TVM")


class TileLangSdistCommand(sdist):
    """Customized setuptools sdist command - includes the pyproject.toml file."""

    def make_distribution(self):
        self.distribution.metadata.name = PACKAGE_NAME
        self.distribution.metadata.version = get_tilelang_version(
            with_cuda=False, with_system_info=False, with_commit_id=False
        )
        super().make_distribution()


# ------------------------------------------------------------------------
# NEW: Add a custom 'develop' command so that `pip install -e .` works.
# ------------------------------------------------------------------------
class TileLangDevelopCommand(develop):
    """
    Customized setuptools 'develop' command for an editable install.
    Ensures the extension is built and all necessary assets are copied.
    """

    def run(self):
        logger.info("Running TileLangDevelopCommand")
        # 1. Build the C/C++ extension modules
        self.run_command("build_ext")

        build_ext_cmd = self.get_finalized_command("build_ext")
        ext_modules = build_ext_cmd.extensions
        for ext in ext_modules:
            extdir = build_ext_cmd.get_ext_fullpath(ext.name)
            logger.info(f"Extension {ext.name} output directory: {extdir}")

        ext_output_dir = os.path.dirname(extdir)
        logger.info(f"Extension output directory (parent): {ext_output_dir}")

        # Copy the built TVM to the package directory
        TVM_PREBUILD_ITEMS = [
            f"{ext_output_dir}/libtvm_runtime.so",
            f"{ext_output_dir}/libtvm.so",
            f"{ext_output_dir}/libtilelang.so",
            f"{ext_output_dir}/libtilelang_module.so",
            f"{ext_output_dir}/libtilelangir.so",
        ]
        for item in TVM_PREBUILD_ITEMS:
            source_lib_file = os.path.join(ROOT_DIR, item)
            # only copy the file
            file_name = os.path.basename(item)
            target_dir = os.path.join(PACKAGE_NAME, file_name)
            target_dir = os.path.dirname(target_dir)
            target_dir = os.path.join(target_dir, "lib")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            if os.path.exists(source_lib_file):
                patch_libs(source_lib_file)
                shutil.copy2(source_lib_file, target_dir)
                # remove the original file (only when under ext_output_dir, not source tree)
                if os.path.abspath(source_lib_file).startswith(
                    os.path.abspath(ext_output_dir)
                ):
                    os.remove(source_lib_file)
            else:
                # Develop: libtilelangir.so may be in build/tilelangir (built by CMake, not setuptools)
                if file_name == "libtilelangir.so":
                    fallback = os.path.join(ROOT_DIR, "build", "tilelangir", file_name)
                    if os.path.isfile(fallback):
                        patch_libs(fallback)
                        shutil.copy2(fallback, target_dir)
                        logger.info(f"Copied {fallback} to {target_dir}")
                else:
                    logger.info(f"INFO: {source_lib_file} does not exist.")


# ------------------------------------------------------------------------
# NEW: Add a custom 'bdist_wheel' command for wheel building.
# ------------------------------------------------------------------------
class TileLangBdistWheel(bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()
        return python, abi, plat

    def run(self):
        self.run_command("build_py")
        self.run_command("build_ext")
        super().run()


class CMakeExtension(Extension):
    """
    A specialized setuptools Extension class for building a CMake project.

    :param name: Name of the extension module.
    :param sourcedir: Directory containing the top-level CMakeLists.txt.
    """

    def __init__(self, name, sourcedir=""):
        # We pass an empty 'sources' list because
        # the actual build is handled by CMake, not setuptools.
        super().__init__(name=name, sources=[])

        # Convert the source directory to an absolute path
        # so that CMake can correctly locate the CMakeLists.txt.
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """
    Custom build_ext command for CMake-based projects.

    This class overrides the 'run' method to ensure that CMake is available,
    and then iterates over all extensions defined as CMakeExtension,
    delegating the actual build logic to 'build_cmake'.
    """

    def run(self):
        # skip cmake if TILELANG_SKIP_BUILD=1
        skip_build = os.environ.get("TILELANG_SKIP_BUILD", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if skip_build:
            print("CMake build skipped, using pre-built libraries from build/")
            return
        # Check if CMake is installed and accessible by attempting to run 'cmake --version'.
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as error:
            # If CMake is not found, raise an error.
            raise RuntimeError(
                "CMake must be installed to build the following extensions"
            ) from error

        update_submodules()

        # Build each extension (of type CMakeExtension) using our custom method.
        for ext in self.extensions:
            self.build_cmake(ext)

        # To make it works with editable install,
        # we need to copy the lib*.so files to the tilelang/lib directory
        import glob

        files = glob.glob("*.so")
        if os.path.exists(PACKAGE_NAME):
            target_lib_dir = os.path.join(PACKAGE_NAME, "lib")
            for file in files:
                if not os.path.exists(target_lib_dir):
                    os.makedirs(target_lib_dir)
                shutil.copy(file, target_lib_dir)
                # remove the original file
                os.remove(file)

    def build_cmake(self, ext):
        """
        Build a single CMake-based extension.

        :param ext: The extension (an instance of CMakeExtension).
        """
        # Only setup LLVM if it's enabled
        llvm_config_path = "OFF"
        if USE_LLVM:
            # Setup LLVM for TVM and retrieve the path to llvm-config
            _, llvm_config_path = setup_llvm_for_tvm()

        # Determine the directory where the final .so or .pyd library should go.
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Prepare arguments for the CMake configuration step.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        tvm_prebuild_path = os.environ.get("TVM_PREBUILD_PATH")
        if tvm_prebuild_path:
            tvm_prebuild_path = os.path.abspath(tvm_prebuild_path)
            cmake_args.append(f"-DTVM_PREBUILD_PATH={tvm_prebuild_path}")
            logger.info(f"Using prebuilt TVM from {tvm_prebuild_path}")

        # Create the temporary build directory (if it doesn't exist).
        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        # Copy the default 'config.cmake' from the source tree into our build directory.
        src_config_cmake = os.path.join(
            ext.sourcedir, "3rdparty", "tvm", "cmake", "config.cmake"
        )
        dst_config_cmake = os.path.join(build_temp, "config.cmake")
        shutil.copy(src_config_cmake, dst_config_cmake)

        # Append some configuration variables to 'config.cmake'
        with open(dst_config_cmake, "a") as config_file:
            config_file.write(f"set(USE_LLVM {llvm_config_path})\n")

            # Add NPUIR
            if USE_NPUIR:
                config_file.write("set(USE_NPUIR ON)\n")
                # Check for BISHENGIR_PATH environment variable
                bishengir_path = os.environ.get("BISHENGIR_PATH")
                if bishengir_path:
                    config_file.write(f"set(BISHENGIR_ROOT_PATH {bishengir_path})\n")
                config_file.write("set(USE_CUDA OFF)\n")
                config_file.write("set(USE_ROCM OFF)\n")
            else:
                if USE_ROCM:
                    config_file.write(f"set(USE_ROCM {ROCM_HOME})\n")
                    config_file.write("set(USE_CUDA OFF)\n")
                else:
                    config_file.write(f"set(USE_CUDA {CUDA_HOME})\n")
                    config_file.write("set(USE_ROCM OFF)\n")

        # Run CMake to configure the project with the given arguments.
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)

        # Build the project in "Release" mode with all available CPU cores ("-j").
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release", "-j"], cwd=build_temp
        )


setup(
    name=PACKAGE_NAME,
    version=(
        get_tilelang_version(with_cuda=False, with_system_info=False)
        if PYPI_BUILD
        else get_tilelang_version()
    ),
    packages=[
        p
        for p in find_packages(where=".")
        if not p.startswith(("testing", "unittest", "examples", "benchmark", "docs"))
    ],
    package_dir={"": "."},
    author="Microsoft Research",
    description="A tile level programming language to generate high performance code.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    platforms=[
        "Operating System :: POSIX :: Linux",
        "Hardware :: Ascend NPU",
    ],
    license="MIT",
    keywords="NPU, ASCENDNPUIR, HIP, Code Generation, TVM",
    url="https://github.com/tile-ai/tilelang-ascend",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    install_requires=get_requirements(),
    python_requires=">=3.8",
    package_data=package_data,
    include_package_data=False,
    ext_modules=[CMakeExtension("TileLangCXX", sourcedir=".")],
    cmdclass={
        "build_py": TileLangBuilPydCommand,
        "sdist": TileLangSdistCommand,
        "build_ext": CMakeBuild,
        "develop": TileLangDevelopCommand,
        "bdist_wheel": TileLangBdistWheel,
    },
)
