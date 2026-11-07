#!/usr/bin/env python3
"""FloorDiv/FloorMod 负数语义回归(codegen 曾把 floordiv 直降为 C 的 `/`
截断除法——(-3)/4=0 而非 -1,conv strip 闭式 s0 在 pad 边界 band 上错位,
实测首/末 band 输出错,见 conv2d trace 2026-08-30)。

验证: 运行期数据驱动,核内计算 floordiv/floormod 负数组合并与 python
// % (floor 语义)逐点对照。
"""

import torch

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

N = 64


def build():
    @tilelang.jit(out_idx=[1, 2], pass_configs=pass_configs, target="ascendc")
    def kernel():
        @T.prim_func
        def main(X: T.Tensor([N], "int32"), D: T.Tensor([N], "int32"), M: T.Tensor([N], "int32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                xb = T.alloc_ub([N], "int32")
                db = T.alloc_ub([N], "int32")
                mb = T.alloc_ub([N], "int32")
                with T.Scope("V"):
                    T.copy(X, xb)
                    for i in T.serial(N):
                        db[i] = T.floordiv(xb[i] - 5, 4)
                        mb[i] = T.floormod(xb[i] - 5, 4)
                    T.copy(db, D)
                    T.copy(mb, M)

        return main

    return kernel()


def test_negative_floordiv_and_floormod():
    torch.manual_seed(0)
    x = torch.randint(0, 12, (N,), dtype=torch.int32).npu()  # x-5 ∈ [-5, 6]
    k = build()
    d, m = k(x)
    torch.npu.synchronize()
    xd = (x.cpu() - 5).numpy()
    import numpy as np

    ref_d = np.floor_divide(xd, 4)
    ref_m = np.mod(xd, 4)  # python 语义 = floor mod
    ok_d = (d.cpu().numpy() == ref_d).all()
    ok_m = (m.cpu().numpy() == ref_m).all()
    print(f"floordiv: {'PASS' if ok_d else 'FAIL'}, floormod: {'PASS' if ok_m else 'FAIL'}")
    if not ok_d:
        bad = d.cpu().numpy() != ref_d
        i = np.nonzero(bad)[0][0]
        print(f"  样例: floordiv({xd[i]},4) 核={d.cpu().numpy()[i]} 应为 {ref_d[i]}")
    assert ok_d and ok_m


if __name__ == "__main__":
    test_negative_floordiv_and_floormod()
