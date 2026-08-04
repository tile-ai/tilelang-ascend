"""Pytest wrapper for sparse_flash_attn_gqa_pipeline_pto.py.

原 Example 实现稀疏 Flash Attention GQA forward（PTO 路线 + 核间流水），
使用 Developer 模式 + T.Pipelined。原文件无 argparse、无 if __name__，
参数硬编码，seed=42，顶层执行。用 subprocess 直接运行原脚本。

原 Example 关键参数（保持不变）：
  - B=2, S=273, SKV=44444, H_Q=64, H_KV=4, DIM=128, topk=2048
  - dtype=float16, accum_dtype=float
  - seed=42, rtol=1e-2, atol=1e-2
  - target="pto"
  - Golden: ref_sparse_attention_fwd_interface_gqa
  - 成功输出: "Test Passed!"
"""

import os
import subprocess
import sys


EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SCRIPT = os.path.join(EXAMPLE_DIR, "sparse_flash_attn_gqa_pipeline_pto.py")


def test_sparse_flash_attn_gqa_pipeline_pto_precision():
    """运行 sparse_flash_attn_gqa_pipeline_pto.py，验证稀疏 GQA Flash Attention 精度（PTO）。

    原文件参数硬编码，直接运行即可。
    成功判定：退出码 0 且 stdout 包含 "Test Passed!"。
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
