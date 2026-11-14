# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

# Regression coverage for a BF16 scalar load followed by T.float32(...).

import pytest
import torch

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def _run_scalar_read(target, dtype):
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    T_len, H = 8, 32

    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def kernel():
        @T.prim_func
        def main(G: T.Tensor([T_len, H], dtype), Y: T.Tensor([T_len], "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                sc = T.alloc_ub([8], "float32")
                with T.Scope("V"):
                    for t in T.serial(T_len):
                        # GM scalar load, scalar conversion, UB fill, and writeback.
                        T.tile.fill(sc, T.float32(G[t, 3]) * 2.0)
                        T.copy(sc[0:1], Y[t : t + 1])

        return main

    torch.manual_seed(0)
    g = torch.randn(T_len, H, dtype=torch_dtype)
    got = kernel()(g.npu()).cpu()
    ref = g[:, 3].float() * 2
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("target", ["ascendc"])
def test_scalar_read_bf16(target):
    _run_scalar_read(target, "bfloat16")


@pytest.mark.parametrize("target", ["ascendc"])
def test_scalar_read_fp16(target):
    _run_scalar_read(target, "float16")


if __name__ == "__main__":
    test_scalar_read_bf16("ascendc")
    print("bf16 PASS")
    test_scalar_read_fp16("ascendc")
    print("fp16 PASS")
