import sys
import shutil
import inspect
from pathlib import Path

import tilelang
from tilelang import JITKernel
from tilelang.jit.adapter import CythonKernelAdapter

HERE = Path(__file__).parent
PACKAGE_ROOT = HERE.parent / "src" / "torch_tl_ascend"
PACKAGE_SOURCE_ROOT = PACKAGE_ROOT / "op_source"
EXAMPLE_ROOT = HERE.parent.parent

SO_NAME = "libop.so"

def _ensure_lib_path():
    from compile_tl_op.util import wrap_libgen  # noqa: F401
    # ensure lib_generator.libpath exists

def get_jit_op_func():
    OP_PATH = EXAMPLE_ROOT / "flash_attention"  # examples/...
    sys.path.append(OP_PATH.as_posix())

    from flash_attn_bhsd import flash_attention_fwd as op_func

    if not hasattr(op_func, "__jit_impl__"):
        raise AttributeError(f"{op_func!r} is not decorated by @tilelang.jit")

    return op_func

def get_kernel(op_func, *args, force_compile=False, **kwargs) -> JITKernel:
    if force_compile:
        tilelang.disable_cache()  # compile will be triggered without caching
    kernel = op_func(*args, **kwargs)
    return kernel

def source_path_of(op_func):
    while hasattr(op_func, "__wrapped__"):
        op_func = op_func.__wrapped__
    return Path(inspect.getfile(op_func))  # file of the original function

def so_path_of(kernel: JITKernel):
    adapter: CythonKernelAdapter = kernel.adapter
    if not adapter.lib_generator.libpath:
        raise RuntimeError("Compiled library path not found!")
    return Path(adapter.lib_generator.libpath)

def update_package_files(force_compile=False):
    _ensure_lib_path()
    op_func = get_jit_op_func()

    op_source_path = source_path_of(op_func)
    print(f"Grabbing {op_source_path.suffix} of {op_func!r} ({op_source_path.as_posix()})")
    package_source_path = PACKAGE_SOURCE_ROOT / op_source_path.name
    shutil.copy(op_source_path, package_source_path)

    B, S, H, D = 4, 4096, 16, 128
    print("Grabbing .so of", op_func, f"with B={B}, S={S}, H={H}, D={D}")
    kernel = get_kernel(op_func, B, S, H, D, force_compile=force_compile)

    lib_so_path = so_path_of(kernel)
    package_so_path = PACKAGE_ROOT / SO_NAME

    shutil.copy(lib_so_path, package_so_path)
    print("Got", package_so_path.name)
