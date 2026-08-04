"""Pytest wrapper for flash_attn_bshd_developer.py.

原 Example 实现 Flash Attention forward（BSHD 布局），使用 Developer 模式。
原文件无 argparse、无 if __name__，参数硬编码（B=1,S=128,H=1,D=512），
顶层执行 + do_bench。用 subprocess 直接运行原脚本。

原 Example 关键参数（保持不变）：
  - B=1, S=128, H=1, D=512, block_M=32, block_N=64
  - dtype=float16, accum_dtype=float
  - seed=0, rtol=1e-2, atol=1e-2
  - Golden: ref_flash_attn (einsum + softmax)
  - 成功输出: "Test Passed!"（非 "Kernel Output Match!"）
"""

import os
import subprocess
import sys


EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SCRIPT = os.path.join(EXAMPLE_DIR, "flash_attn_bshd_developer.py")


def test_flash_attn_bshd_developer_precision():
    """运行 flash_attn_bshd_developer.py，验证 Flash Attention 精度。

    原文件参数硬编码，直接运行即可。
    成功判定：退出码 0 且 stdout 包含 "Test Passed!"。
    注意：原文件含 do_bench 性能测试，执行时间较长。
    """
    result = subprocess.run(
        [sys.executable, EXAMPLE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=EXAMPLE_DIR,
    )
    assert result.returncode == 0, f"脚本执行失败 (exit={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Test Passed!" in result.stdout, f"精度校验未通过\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
