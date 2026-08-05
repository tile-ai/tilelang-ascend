"""Pytest wrapper for flash_attn_bshd_developer.py.

原 Example 实现 Flash Attention forward（BSHD 布局），使用 Developer 模式。
原文件无 argparse、无 if __name__，参数硬编码（B=1,S=128,H=1,D=512），
顶层执行 + do_bench。用 importlib 在当前进程执行原脚本。

原 Example 关键参数（保持不变）：
  - B=1, S=128, H=1, D=512, block_M=32, block_N=64
  - dtype=float16, accum_dtype=float
  - seed=0, rtol=1e-2, atol=1e-2
  - Golden: ref_flash_attn (einsum + softmax)
  - 成功输出: "Test Passed!"（非 "Kernel Output Match!"）
"""

import importlib.util
import sys
from pathlib import Path


def _run_example() -> None:
    source = Path(__file__).with_name("flash_attn_bshd_developer.py")
    spec = importlib.util.spec_from_file_location("_flash_attn_bshd_developer_under_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv


def test_flash_attn_bshd_developer_precision():
    """运行 flash_attn_bshd_developer.py，验证 Flash Attention 精度。

    原文件参数硬编码，直接运行即可。
    成功判定：退出码 0 且 stdout 包含 "Test Passed!"。
    注意：原文件含 do_bench 性能测试，执行时间较长。"""
    _run_example()
