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

def update_package_files(force_compile=False):
    sys.path.append(HERE.as_posix())
    from compile_tl_op.util import wrap_libgen  # noqa: F401
    # ensure lib_generator.libpath exists

    OP_PATH = EXAMPLE_ROOT / "flash_attention"  # examples/...
    sys.path.append(OP_PATH.as_posix())

    from flash_attn_bhsd_v2 import flash_attention_fwd as op_func

    if not hasattr(op_func, "__jit_impl__"):
        raise AttributeError(f"{op_func!r} is not decorated by @tilelang.jit")

    op_source_path = Path(inspect.getfile(op_func.__wrapped__))  # the original function file
    print(f"Grabbing {op_source_path.suffix} of {op_func!r} ({op_source_path.as_posix()})")
    package_source_path = PACKAGE_SOURCE_ROOT / op_source_path.name
    shutil.copy(op_source_path, package_source_path)

    if force_compile:
        tilelang.disable_cache()  # compile will be triggered without caching

    B, S, H, D = 4, 4096, 32, 512
    print("Grabbing .so of", op_func, f"with B={B}, S={S}, H={H}, D={D}")
    kernel: JITKernel = op_func(B, S, H, D)

    adapter: CythonKernelAdapter = kernel.adapter

    if not adapter.lib_generator.libpath:
        raise RuntimeError("Compiled library path not found!")

    lib_so_path = Path(adapter.lib_generator.libpath)
    package_so_path = PACKAGE_ROOT / SO_NAME

    shutil.copy(lib_so_path, package_so_path)
    print("Got", package_so_path.name)
