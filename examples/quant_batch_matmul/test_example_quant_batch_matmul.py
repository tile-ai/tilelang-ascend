"""Pytest wrapper for example_quant_batch_matmul.py.

原 Example 实现量化 Batch 矩阵乘法：C = (A @ B) * scale（batch 维）。
和 example_quant_matmul.py 结构相同，有 check_case() + main() + if __name__，
可安全 import。

原 Example 关键参数（保持不变）：
  - block_M=128, block_N=256, block_K=64
  - Batch=8, in_dtype=int8, accum_dtype=int32, scale_dtype=float32
  - seed=0, rtol=1e-2, atol=1e-2
  - Golden: ref_program 中 C = A.to(int32) @ B.to(int32); (C.to(float32) * scale).to(out_dtype)
"""

import pytest
import torch
import tilelang as tl

from example_quant_batch_matmul import check_case


@pytest.mark.parametrize(
    ("batch", "m", "n", "k", "scale_size", "out_dtype"),
    [
        (8, 1024, 1024, 1024, "1", "float16"),
        (8, 1024, 1024, 1024, "N", "bfloat16"),
        (4, 512, 512, 512, "1", "float16"),
        (4, 512, 1024, 512, "N", "bfloat16"),
        (8, 1024, 2048, 1024, "1", "float16"),
    ],
    ids=[
        "baseline_8x1024_scale1_fp16",
        "baseline_8x1024_scaleN_bf16",
        "small_4x512_scale1_fp16",
        "rect_4x512x1024x512_scaleN_bf16",
        "rect_8x1024x2048x1024_scale1_fp16",
    ],
)
def test_quant_batch_matmul_precision(batch, m, n, k, scale_size, out_dtype):
    """运行量化 Batch 矩阵乘法，验证 C = (A @ B) * scale 精度。

    直接调用原文件的 check_case()。
    """
    torch.manual_seed(0)
    tl.cache.clear_cache()
    check_case(
        batch,
        m,
        n,
        k,
        scale_size=scale_size,
        block_M=128,
        block_N=256,
        block_K=64,
        out_dtype=out_dtype,
    )
