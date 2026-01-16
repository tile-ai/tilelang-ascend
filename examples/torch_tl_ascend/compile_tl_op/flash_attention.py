import sys
import shutil
import inspect
from pathlib import Path

import tilelang
from tilelang import JITKernel
from tilelang.jit.adapter import CythonKernelAdapter

HERE = Path(__file__).parent
PACKAGE_ROOT = HERE.parent / "src" / "torch_tl_ascend"
EXAMPLE_ROOT = HERE.parent.parent

SO_NAME = "libop.so"

def update_package_files(force_compile=False):
    sys.path.append(HERE.as_posix())
    from compile_tl_op.util import wrap_libgen  # ensure lib_generator.libpath exists

    OP_PATH = EXAMPLE_ROOT / "flash_attention"  # examples/...
    sys.path.append(OP_PATH.as_posix())

    from flash_attention.flash_attn_bhsd_cc_sync_auto_pipeline import flash_attention_fwd as op_func

    op_code_path = Path(inspect.getfile(op_func))
    print(f"Grabbing {op_code_path.suffix} of {op_func!r} ({op_code_path.as_posix()})")
    package_code_path = PACKAGE_ROOT / op_code_path.name
    shutil.copy(op_code_path, package_code_path)

    if force_compile:
        tilelang.disable_cache()  # compile will be triggered without caching

    print("Grabbing .so of", op_func)
    B, S, H, D = 4, 4096, 32, 512
    kernel: JITKernel = op_func(B, S, H, D)

    adapter: CythonKernelAdapter = kernel.adapter

    if not adapter.lib_generator.libpath:
        raise RuntimeError("Compiled library path not found!")

    lib_so_path = Path(adapter.lib_generator.libpath)
    package_so_path = PACKAGE_ROOT / SO_NAME

    shutil.copy(lib_so_path, package_so_path)
    print("Got", package_so_path.name)
