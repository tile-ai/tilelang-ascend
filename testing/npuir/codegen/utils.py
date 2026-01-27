# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""
仅编译出 kernel.o 的辅助函数，不依赖 torch_npu。
参考 tilelang/jit/jit_npu.py 中的 NPU IR -> 二进制 流程。
"""
import os
import subprocess
import tempfile
from pathlib import Path
import shutil
from typing import Optional, Tuple

from tvm import tir
from tvm.tir import PrimFunc

from tilelang.engine import lower


def _transform_stmt(stmt, symbolic_var_names):
    if stmt is None:
        return stmt

    if isinstance(stmt, tir.Block):
        new_body = _transform_stmt(stmt.body, symbolic_var_names)
        return tir.Block(
            iter_vars=stmt.iter_vars,
            reads=stmt.reads,
            writes=stmt.writes,
            name_hint=stmt.name_hint,
            body=new_body,
            init=stmt.init,
            alloc_buffers=stmt.alloc_buffers,
            annotations=stmt.annotations,
            span=stmt.span,
        )
    elif isinstance(stmt, tir.SeqStmt):
        new_seqs = []
        for seq_stmt in stmt.seq:
            transformed = _transform_stmt(seq_stmt, symbolic_var_names)
            if transformed is not None:
                new_seqs.append(transformed)
        return tir.SeqStmt(new_seqs, stmt.span) if new_seqs else None
    elif isinstance(stmt, tir.LetStmt):
        if stmt.var.name in symbolic_var_names:
            return _transform_stmt(stmt.body, symbolic_var_names)
        else:
            new_body = _transform_stmt(stmt.body, symbolic_var_names)
            return tir.LetStmt(
                var=stmt.var, value=stmt.value, body=new_body, span=stmt.span
            )
    else:
        return stmt


def _process_dynamic_symbolic(func: PrimFunc):
    params = func.params
    buffer_map = func.buffer_map
    dynamic_symbolic_map = {}
    for i, param in enumerate(params):
        if param not in buffer_map:
            continue
        buffer = buffer_map[param]
        for j, shape in enumerate(buffer.shape):
            if isinstance(shape, tir.Var) and (shape not in dynamic_symbolic_map):
                dynamic_symbolic_map[shape] = (i, j)
    return dynamic_symbolic_map


def _symbolic_var_promoter_pass(func: PrimFunc) -> Tuple[PrimFunc, dict]:
    dynamic_symbolic_map = _process_dynamic_symbolic(func)
    symbolic_vars = list(dynamic_symbolic_map.keys())

    if len(symbolic_vars) == 0:
        return func, {}

    new_params = list(func.params) + symbolic_vars
    new_body = _transform_stmt(func.body, symbolic_vars)

    new_primfunc = tir.PrimFunc(
        params=new_params,
        body=new_body,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
        span=func.span,
    )
    return new_primfunc, dynamic_symbolic_map


def _get_npucompiler_path() -> str:
    npu_compiler_path = shutil.which("bishengir-compile")
    if npu_compiler_path is None:
        npu_compiler_root = os.getenv("TRITON_NPU_COMPILER_PATH", "")
        if not npu_compiler_root:
            raise EnvironmentError(
                "Couldn't find executable bishengir-compile or TRITON_NPU_COMPILER_PATH."
            )
        npu_compiler_path = os.path.join(npu_compiler_root, "npuc")
    return npu_compiler_path


def _get_tilelangir_compile_path() -> str:
    path = shutil.which("tilelangir-compile")
    if path is not None:
        return path
    root = os.getenv("TILELANGIR_COMPILE_PATH", "")
    if root:
        p = os.path.join(root, "tilelangir-compile")
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    # 仓库内默认：从 utils 所在目录向上找到 repo 根（含 CMakeLists.txt 的 tilelang-ascend），再找 build/.../tilelangir-compile
    _this_dir = Path(__file__).resolve().parent
    for _parent in [_this_dir] + list(_this_dir.parents):
        cmake_at = _parent / "CMakeLists.txt"
        if cmake_at.is_file() and (_parent / "tilelangir").is_dir():
            default_exe = _parent / "build" / "tilelangir" / "tools" / "tilelangir-compile" / "tilelangir-compile"
            if default_exe.is_file() and os.access(default_exe, os.X_OK):
                return str(default_exe)
            break
    raise EnvironmentError(
        "Couldn't find executable tilelangir-compile or TILELANGIR_COMPILE_PATH."
    )


def npuir_to_kernel_o(mlir_content: str) -> bytes:
    """
    将 NPU IR 源码编译为 kernel.o 的二进制内容。
    与 jit_npu._npuir_to_bin_enable_npu_compile 中的 npuir -> .o 流程一致，
    不依赖 torch_npu。

    mlir_content 应为已跑过 TileLangIR pipeline 的 MLIR（例如由 lower(..., target="npuir") 返回的字符串）。

    Returns:
        kernel.o 的二进制内容。
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.npuir")
        Path(ttadapter_path).write_text(mlir_content)
        bin_file = os.path.join(tmpdir, "kernel")
        bin_path = os.path.join(tmpdir, "kernel.o")

        npu_compiler_path = _get_npucompiler_path()
        _compile_option_list = [
            "--enable-auto-multi-buffer=true",
            "--enable-triton-kernel-compile=true",
            "--enable-hivm-compile=true",
        ]
        tilelang_ascend_mode = os.environ.get("TILELANG_ASCEND_MODE")
        if tilelang_ascend_mode is None:
            _compile_option_list.append("--disable-hivm-tensor-compile=true")
        elif tilelang_ascend_mode.lower().strip() in ("expert", "exp", "e"):
            _compile_option_list.append("--disable-hivm-tensor-compile=true")

        cmd_list = (
            [npu_compiler_path, ttadapter_path]
            + _compile_option_list
            + ["-o", bin_file]
        )
        subprocess.run(cmd_list, capture_output=True, check=True, text=True)
        return Path(bin_path).read_bytes()


def compile_to_kernel_o(func: PrimFunc) -> bytes:
    """
    将 TileLang PrimFunc 编译为 kernel.o 的二进制内容。
    仅做 lowering + NPU IR 编译，不生成 launcher、不依赖 torch_npu。

    Args:
        func: 带有 is_npu=True 的 T.Kernel 的 PrimFunc（如 flash_attn_npuir 中定义）。

    Returns:
        kernel.o 的二进制内容。
    """
    mod, _ = _symbolic_var_promoter_pass(func)
    out = lower(mod, target="npuir")
    # target="npuir" 时 lower 返回 MLIR 字符串
    if not isinstance(out, str):
        raise TypeError("lower(..., target='npuir') 应返回 str")
    return npuir_to_kernel_o(out)


def is_compile_success(
    o_bytes: Optional[bytes],
) -> bool:
    """
    判断一次编译是否成功。
    成功标准：无 subprocess 报错且最终存在 .o 文件（即得到非空 bytes）。
    """
    return isinstance(o_bytes, bytes) and len(o_bytes) > 0


def compile_to_kernel_o_safe(func: PrimFunc) -> Tuple[bool, Optional[bytes]]:
    """
    执行 compile_to_kernel_o，捕获 subprocess/环境异常，不抛错。
    成功标准：无 subprocess 报错且最终存在 .o 文件。
    Returns:
        (是否成功, kernel.o 的二进制内容，失败时为 None)
    """
    try:
        o_bytes = compile_to_kernel_o(func)
        return (is_compile_success(o_bytes), o_bytes)
    except (subprocess.CalledProcessError, FileNotFoundError, EnvironmentError, OSError):
        return (False, None)


def assert_compile_to_kernel_o_success(func: PrimFunc) -> bytes:
    """
    UT 辅助：对 PrimFunc 做「仅编译出 kernel.o」并完成所有断言逻辑。
    成功标准：无 subprocess 报错且最终存在 .o 文件。
    失败时抛出 AssertionError，并附带子进程的 stderr/stdout 便于排查。
    Returns:
        成功时返回 kernel.o 的二进制内容。
    """
    try:
        o_bytes = compile_to_kernel_o(func)
    except subprocess.CalledProcessError as e:
        stdout = (e.stdout or "").strip()
        stderr = (e.stderr or "").strip()
        msg = (
            "compile_to_kernel_o 失败 (子进程非零退出):\n"
            f"  命令: {e.cmd}\n  返回码: {e.returncode}\n"
        )
        if stdout:
            msg += f"  stdout:\n{stdout}\n"
        if stderr:
            msg += f"  stderr:\n{stderr}\n"
        raise AssertionError(msg) from e
    except (FileNotFoundError, EnvironmentError, OSError) as e:
        raise AssertionError(f"compile_to_kernel_o 失败 (环境/IO): {e}") from e
    if not is_compile_success(o_bytes):
        raise AssertionError(
            "compile_to_kernel_o 应成功：无 subprocess 报错且最终存在 .o 文件"
        )
    return o_bytes
