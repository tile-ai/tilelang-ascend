# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

# T.tile.silu in-place（dst 与 src 同一 buffer）回归测试。
#
# 背景：AscendC::Silu 官方约束"不支持源操作数与目的操作数地址重叠"。此前
# 前端对 silu(acc, acc) 直接发射 Silu(acc, acc, n)，原地计算行为未定义
# （实测输出全 1.0）。修复后前端检测 alias 并自动分配隐藏 UB tmp 中转：
# Silu(tmp, src, n) + DataCopy(dst, tmp, n)。

import pytest
import torch

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def _run_silu(target, dtype, inplace):
    torch_dtype = {"float32": torch.float32, "float16": torch.float16}[dtype]
    N = 256

    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def kernel():
        @T.prim_func
        def main(X: T.Tensor([N], dtype), Y: T.Tensor([N], dtype)):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                acc = T.alloc_ub([N], dtype)
                tmp = T.alloc_ub([N], dtype)
                with T.Scope("V"):
                    T.copy(X, acc)
                    if inplace:
                        T.tile.silu(acc, acc)   # in-place：走自动 tmp 中转
                    else:
                        T.tile.silu(tmp, acc)   # 非 in-place：3 参直发
                        T.copy(tmp, acc)
                    T.copy(acc, Y)

        return main

    torch.manual_seed(0)
    x = torch.randn(N, dtype=torch_dtype)
    got = kernel()(x.npu()).cpu().float()
    ref = (x.float() * torch.sigmoid(x.float()))
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("target", ["ascendc"])
def test_silu_inplace_fp32(target):
    _run_silu(target, "float32", inplace=True)


@pytest.mark.parametrize("target", ["ascendc"])
def test_silu_inplace_fp16(target):
    _run_silu(target, "float16", inplace=True)


@pytest.mark.parametrize("target", ["ascendc"])
def test_silu_outofplace_fp32(target):
    _run_silu(target, "float32", inplace=False)


if __name__ == "__main__":
    test_silu_inplace_fp32("ascendc")
    print("inplace fp32 PASS")
    test_silu_inplace_fp16("ascendc")
    print("inplace fp16 PASS")
    test_silu_outofplace_fp32("ascendc")
    print("out-of-place fp32 PASS")
