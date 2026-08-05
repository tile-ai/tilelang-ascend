"""Pytest wrapper for gelu_mul_developer.py.

原 Example 实现算子语义 B = GELU(x1) * x2（输入按最后一维 split）。使用
Developer 模式。原文件无 argparse、无 if __name__，自带 2 组 test_configs
顶层循环执行。用 importlib 在当前进程执行原脚本。

原 Example 关键参数（保持不变）：
  - 2 组 test_configs（原文件 L61-64）
  - dtype=float (float32)
  - seed=0, rtol=1e-2, atol=1e-2
  - Golden: ref_b = gelu(a1) * a2
"""

import importlib.util
import sys
from pathlib import Path


def _run_example() -> None:
    source = Path(__file__).with_name("gelu_mul_developer.py")
    spec = importlib.util.spec_from_file_location("_gelu_mul_developer_under_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv


def test_gelu_mul_developer_precision():
    """运行 gelu_mul_developer.py，验证 B = GELU(x1) * x2 精度。

    原脚本自带 2 组 test_configs，直接运行即可覆盖。
    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。"""
    _run_example()
