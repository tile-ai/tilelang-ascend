"""BF16 transpose correctness and hardware-path regression."""

import pytest

import torch

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def build_bf16_transpose2d(batch, H, W):
    TH, TW = 64, 64  # UB 砖（16 倍数），逐砖转置
    tasks = batch * (H // TH) * (W // TW)

    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target="ascendc")
    def kernel():
        @T.prim_func
        def main(x: T.Tensor((batch, H, W), "bfloat16"), out: T.Tensor((batch, W, H), "bfloat16")):
            with T.Kernel(tasks, is_npu=True) as (cid, vid):
                ub = T.alloc_ub((TH, TW), "bfloat16")
                ubt = T.alloc_ub((TW, TH), "bfloat16")
                with T.Scope("V"):
                    b = cid // ((H // TH) * (W // TW))
                    rem = cid % ((H // TH) * (W // TW))
                    h0 = (rem // (W // TW)) * TH
                    w0 = (rem % (W // TW)) * TW
                    T.copy(x[b, h0 : h0 + TH, w0 : w0 + TW], ub)
                    T.tile.transpose(ubt, ub)
                    T.copy(ubt, out[b, w0 : w0 + TW, h0 : h0 + TH])

        return main

    return kernel()


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="BF16 transpose correctness requires an Ascend NPU runtime",
)
def test_bf16_transpose_uses_hardware_path():
    torch.manual_seed(0)
    batch, H, W = 1, 64, 64
    x = torch.randn(batch, H, W, dtype=torch.bfloat16)
    kernel = build_bf16_transpose2d(batch, H, W)
    got = kernel(x.npu()).cpu()
    ref = x.permute(0, 2, 1).contiguous()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)

    source = kernel.get_kernel_source()
    assert "TransDataTo5HDImpl<MovT>" in source


if __name__ == "__main__":
    test_bf16_transpose_uses_hardware_path()
