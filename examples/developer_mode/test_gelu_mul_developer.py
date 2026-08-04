"""Pytest wrapper for gelu_mul_developer.py.

原 Example 实现算子语义 B = GELU(x1) * x2（输入按最后一维 split）。使用
Developer 模式。原文件无 argparse、无 if __name__，自带 2 组 test_configs
顶层循环执行。用 subprocess 直接运行原脚本。

原 Example 关键参数（保持不变）：
  - 2 组 test_configs（原文件 L61-64）
  - dtype=float (float32)
  - seed=0, rtol=1e-2, atol=1e-2
  - Golden: ref_b = gelu(a1) * a2
"""

import os
import subprocess
import sys


EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SCRIPT = os.path.join(EXAMPLE_DIR, "gelu_mul_developer.py")


def test_gelu_mul_developer_precision():
    """运行 gelu_mul_developer.py，验证 B = GELU(x1) * x2 精度。

    原脚本自带 2 组 test_configs，直接运行即可覆盖。
    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。
    """
    result = subprocess.run(
        [sys.executable, EXAMPLE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=EXAMPLE_DIR,
    )
    assert result.returncode == 0, f"脚本执行失败 (exit={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Kernel Output Match!" in result.stdout, f"精度校验未通过\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
