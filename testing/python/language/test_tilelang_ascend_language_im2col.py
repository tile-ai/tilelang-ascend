# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

# T.tile.im2col 回归测试：L1(NC1HWC0) -> L0A 的卷积 im2col tile 提取。
#
# 验证方式：im2col 结果在 L0A 无法直接读出，用 fp16 单位矩阵 mma 把 L0A 内容
# 原样搬到 L0C 再写回 GM（fp16 单位阵 mma 精确），与 torch 手工 im2col 参考比对。
#
# 布局要点：C0=16(fp16) 且 L1 tile 的 N 维 == C0 时，zN 分形退化为连续行主序，
# 与 NC1HWC0 内存完全一致——因此 2D [H*W, 16] 的 T.copy(GM->L1) 得到的就是
# 合法的 NC1HWC0 feature map。

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def _run_im2col(target, H=5, W=5, C0=16, KH=3, KW=3, pad=1):
    HW = H * W
    M_ROUND = ((HW + 15) // 16) * 16  # L0A 分形 16 行对齐
    KDIM = KH * KW * C0  # im2col K 维

    @tilelang.jit(out_idx=[2], pass_configs=pass_configs, target=target)
    def kernel():
        @T.prim_func
        def main(Fmap: T.Tensor([HW, C0], "float16"), Eye: T.Tensor([KDIM, KDIM], "float16"), Out: T.Tensor([M_ROUND, KDIM], "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                f_l1 = T.alloc_L1([HW, C0], "float16")
                w_l1 = T.alloc_L1([KDIM, KDIM], "float16")
                T.annotate_layout(
                    {
                        f_l1: make_zn_layout(f_l1),
                        w_l1: make_zn_layout(w_l1),
                    }
                )
                a_l0 = T.alloc_L0A([M_ROUND, KDIM], "float16")
                w_l0 = T.alloc_L0B([KDIM, KDIM], "float16")
                c_l0c = T.alloc_L0C([M_ROUND, KDIM], "float")
                with T.Scope("C"):
                    T.copy(Fmap, f_l1)
                    T.copy(Eye, w_l1)
                    T.tile.im2col(a_l0, f_l1, (H, W), (KH, KW), (1, 1), (1, 1), (pad, pad, pad, pad), 0, 0, HW, KDIM)
                    T.copy(w_l1, w_l0)
                    T.mma(a_l0, w_l0, c_l0c, init=True)
                    T.copy(c_l0c, Out)

        return main

    torch.manual_seed(0)
    fmap = torch.randn(1, C0, H, W, dtype=torch.float16)  # NCHW
    fmap_nc1hwc0 = fmap.permute(0, 2, 3, 1).contiguous().view(HW, C0).npu()
    eye = torch.eye(KDIM, dtype=torch.float16).npu()

    got = kernel()(fmap_nc1hwc0, eye).cpu()

    # 参考：K 序 = (kh, kw, c)，c 最快（NC1HWC0 的 C0 在 L1 中最内层连续）
    xpad = torch.nn.functional.pad(fmap, (pad, pad, pad, pad))  # [1,C,H+2p,W+2p]
    ref = torch.zeros(HW, KDIM)
    for oh in range(H):
        for ow in range(W):
            m = oh * W + ow
            for kh in range(KH):
                for kw in range(KW):
                    ih, iw = oh + kh, ow + kw  # stride=1, pad 已含
                    ref[m, (kh * KW + kw) * C0 : (kh * KW + kw + 1) * C0] = xpad[0, :, ih, iw].float()
    torch.testing.assert_close(got[:HW, :], ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("target", ["ascendc"])
def test_tile_im2col(target):
    _run_im2col(target)


if __name__ == "__main__":
    test_tile_im2col("ascendc")
    print("ascendc PASS")
