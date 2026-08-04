"""Pytest wrapper for activation/swi_glu_v2.py.

原 Example 实现算子语义 B = SiLU(x1) * x2（SwiGLU forward v2，persistent +
双缓冲 + fp32 中间计算）。原文件无 argparse、无 if __name__，自带 7 组
test_configs 顶层循环执行。用 subprocess 直接运行原脚本。

原 Example 关键参数（保持不变）：
  - 7 组 test_configs（原文件 L111-119）
  - dtype=bfloat16/float
  - seed=0, rtol=1e-2, atol=1e-2
  - Golden: ascendc_b = torch.ops.npu.npu_swiglu(a)
"""

import os
import subprocess
import sys

import pytest

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SCRIPT = os.path.join(EXAMPLE_DIR, "swi_glu_v2.py")


def test_swi_glu_v2_precision():
    """运行 swi_glu_v2.py，验证 SwiGLU v2 forward 精度。

    原脚本自带 7 组 test_configs，直接运行即可覆盖。
    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。
    """
    result = subprocess.run(
        [sys.executable, EXAMPLE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=EXAMPLE_DIR,
    )
    assert result.returncode == 0, (
        f"脚本执行失败 (exit={result.returncode})\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Kernel Output Match!" in result.stdout, (
        f"精度校验未通过\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
