"""
TileLang Ascend Operators - Precompile Script

Compiles all operator kernels and saves them to the kernels directory.

Usage:
    python compile/precompile.py              # Compile all kernels
    python compile/precompile.py flash_attention  # Compile only specified kernel
"""

import os
import sys
import shutil
import pickle
from pathlib import Path
from copy import deepcopy

import torch

torch.npu.set_device(0)

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
PACKAGE_DIR = PROJECT_DIR / "src"
KERNELS_DIR = PACKAGE_DIR / "kernels"
UTILS_DIR = PACKAGE_DIR / "utils"

sys.path.insert(0, str(SCRIPT_DIR))
from kernels import KERNEL_REGISTRY

from tilelang.utils.npu_utils import NPUUtils, get_runtime_file_cache


def save_npu_utils():
    """Save npu_utils.so to utils directory (shared by all kernels)"""
    utils_dir = UTILS_DIR
    utils_dir.mkdir(parents=True, exist_ok=True)
    
    npu_utils_so_path = utils_dir / "npu_utils.so"
    
    if npu_utils_so_path.exists() and npu_utils_so_path.stat().st_size > 0:
        print(f"  npu_utils.so already exists at {npu_utils_so_path}")
        return npu_utils_so_path
    
    npu_utils = NPUUtils.get()
    
    import tilelang.utils.npu_utils as npu_utils_module
    pkg_root = os.path.dirname(os.path.abspath(npu_utils_module.__file__))
    npu_utils_cpp = os.path.join(pkg_root, "npu_utils.cpp")
    cache_path = get_runtime_file_cache(npu_utils_cpp)
    source_so_path = Path(cache_path) / "npu_utils.so"
    
    if source_so_path.exists():
        shutil.copy(source_so_path, npu_utils_so_path)
        print(f"  ✓ Saved npu_utils.so to {npu_utils_so_path}")
        return npu_utils_so_path
    else:
        raise FileNotFoundError(f"Cannot find npu_utils.so at {source_so_path}")


def _to_pure_python(obj):
    """
    Recursively convert TVM types to pure Python types
    
    Preserves types that don't involve TVM dependencies (e.g. torch.dtype)
    """
    if obj is None:
        return None
    
    # Return basic types directly
    if isinstance(obj, (bool, int, float, str)):
        return obj
    
    # Return bytes directly
    if isinstance(obj, bytes):
        return obj
    
    # Return torch.dtype directly (no TVM dependency)
    if isinstance(obj, torch.dtype):
        return obj
    
    # TVM IntImm -> int
    if hasattr(obj, 'value') and not isinstance(obj, (bool, int, float, str, torch.dtype)):
        try:
            return int(obj.value)
        except:
            pass
    
    # TVM tir.Var -> str (variable name)
    if hasattr(obj, 'name') and not isinstance(obj, (bool, int, float, str, torch.dtype)):
        try:
            return str(obj.name)
        except:
            pass
    
    # list/tuple -> recursive conversion
    if isinstance(obj, (list, tuple)):
        return [_to_pure_python(item) for item in obj]
    
    # dict -> recursive conversion
    if isinstance(obj, dict):
        return {
            _to_pure_python(k): _to_pure_python(v)
            for k, v in obj.items()
        }
    
    # Try to convert other types to string
    try:
        return str(obj)
    except:
        return None


def save_kernel(kernel, name: str):
    """Save kernel to kernels directory"""
    kernel_dir = KERNELS_DIR / name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving kernel to: {kernel_dir}")
    
    # Deep convert all fields to pure Python types
    metadata = {
        "symbolic": _to_pure_python(kernel.symbolic),
        "out_idx": _to_pure_python(kernel.out_idx),
        "param_info": _to_pure_python(kernel.param_info),
        "signature": _to_pure_python(kernel.signature),
        "shared": _to_pure_python(kernel.utils_shared),
        "kernel_name": _to_pure_python(kernel.kernel_name),
        "gridfunc": _to_pure_python(kernel.gridfunc),
        "mix_mode": _to_pure_python(kernel.mix_mode),
        "name": _to_pure_python(kernel.utils_name),
        "tensor_kinds": _to_pure_python(kernel.tensor_kinds),
        "kernel_src": _to_pure_python(kernel.utils_kernel_src),
    }
    
    # Print converted content for debugging
    print(f"  symbolic: {metadata['symbolic']}")
    print(f"  out_idx: {metadata['out_idx']}")
    print(f"  param_info: {metadata['param_info']}")
    
    metadata_path = kernel_dir / "metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"  ✓ Saved metadata.pkl (using standard pickle)")
    
    # .so files are in current working directory
    cwd = Path(os.getcwd())
    
    # Copy main.so (launcher)
    launcher_so_path = cwd / kernel.so_launcher_path
    
    if launcher_so_path.exists():
        shutil.copy(launcher_so_path, kernel_dir / "main.so")
        print(f"  ✓ Saved main.so (from {launcher_so_path})")
    else:
        print(f"  ✗ Error: Cannot find {launcher_so_path}")
        print(f"    .so files in current directory: {list(cwd.glob('*.so'))}")


def main():
    print("TileLang Ascend Operators - Precompile Script")
    print("=" * 60)
    print(f"Available kernels: {list(KERNEL_REGISTRY.keys())}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Kernel output directory: {KERNELS_DIR}")
    print(f"Utils output directory: {UTILS_DIR}")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        kernels_to_compile = sys.argv[1:]
    else:
        kernels_to_compile = list(KERNEL_REGISTRY.keys())
    
    print(f"Will compile: {kernels_to_compile}")
    
    # Save npu_utils.so to utils directory (shared by all kernels)
    print("\nSaving npu_utils.so...")
    save_npu_utils()
    
    if KERNELS_DIR.exists():
        shutil.rmtree(KERNELS_DIR)
    KERNELS_DIR.mkdir(parents=True, exist_ok=True)
    
    success_kernels = []
    for name in kernels_to_compile:
        if name not in KERNEL_REGISTRY:
            print(f"Warning: Unknown kernel '{name}', skipping")
            continue
        
        try:
            compile_func = KERNEL_REGISTRY[name]
            kernel = compile_func()
            save_kernel(kernel, name)
            success_kernels.append(name)
        except Exception as e:
            print(f"Error: Failed to compile '{name}': {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"✓ Precompilation completed! Successful: {success_kernels}")
    print(f"Kernel directory: {KERNELS_DIR}")
    print(f"Utils directory: {UTILS_DIR}")
    
    # List generated files
    for name in success_kernels:
        kernel_dir = KERNELS_DIR / name
        files = list(kernel_dir.glob("*"))
        print(f"  {name}/: {[f.name for f in files]}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
