"""Pytest wrapper for sparse_flash_attn_developer.py.

原 Example 实现稀疏 Flash Attention forward（带 workspace_idx 多输出），
使用 Developer 模式。原文件无 argparse、无 if __name__，参数硬编码，
顶层执行。用 importlib 在当前进程执行原脚本。

原 Example 关键参数（保持不变）：
  - B=1, S=128, SKV=32768, H=128, D=512, topk=2048
  - dtype=float16, accum_dtype=float
  - seed=0, rtol=1e-2, atol=1e-2
  - Golden: ref_sparse_attention_fwd_interface
  - 成功输出: "Test Passed!"
"""

import importlib.util
import sys
from pathlib import Path


def _run_example() -> None:
    source = Path(__file__).with_name("sparse_flash_attn_developer.py")
    spec = importlib.util.spec_from_file_location("_sparse_flash_attn_developer_under_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv


def test_sparse_flash_attn_developer_precision():
    """运行 sparse_flash_attn_developer.py，验证稀疏 Flash Attention 精度。

    原文件参数硬编码，直接运行即可。
    成功判定：退出码 0 且 stdout 包含 "Test Passed!"。"""
    _run_example()
