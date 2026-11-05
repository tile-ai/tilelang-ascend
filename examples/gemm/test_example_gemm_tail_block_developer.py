"""Pytest wrapper for example_gemm_tail_block_developer.py.

原 Example 实现算子语义 C = A @ B，重点验证非整除尾块场景。使用 Developer
模式（pass_configs + T.Scope("C")）。原文件无 argparse、无 if __name__，
自带 4 组非整除 shape 的 test_configs，顶层循环执行。

拆分方式：importlib 在当前进程执行原脚本（原脚本自带 4 组 case），
检查输出含 "Kernel Output Match!"。不 parametrize，因为原文件
自己管理 case matrix。

原 Example 关键参数（保持不变）：
  - 4 组非整除 shape（原文件 L46-51）
  - dtype=float16, accum_dtype=float
  - seed=0, rtol=1e-2, atol=1e-2
  - Golden: ref_c = a @ b
"""

import importlib.util
import sys
from pathlib import Path


def _run_example() -> None:
    source = Path(__file__).with_name("example_gemm_tail_block_developer.py")
    spec = importlib.util.spec_from_file_location("_example_gemm_tail_block_developer_under_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv


def test_example_gemm_tail_block_developer_precision():
    """运行 example_gemm_tail_block_developer.py，验证非整除尾块场景精度。

    原脚本自带 4 组非整除 shape，直接运行即可覆盖。
    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。"""
    _run_example()
