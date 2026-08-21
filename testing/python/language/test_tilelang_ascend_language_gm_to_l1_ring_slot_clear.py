# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

# 回归测试：双 buffer（ring slot）L1 预取模式下，尾块（partial tile）拷贝进
# 非零偏移 slot 时，copy_gm_to_l1 的 need_clear 必须仍为 true。
#
# 背景（cannbot matmul 样例调试发现，ISSUE-M2）：codegen_ascend.cc 曾用
# `need_clear = (dst_offset == 0)` 判断是否清零——首个 chunk（slot 0，偏移 0）
# 正确清零，但预取 chunk 写入 slot (k+1)%S1（偏移为 slot 步长倍数，非零）被误判为
# "子区域拼接拷贝"而跳过清零，导致 K 尾块混入前序 chunk 的残留数据。
# 现象：K 有尾块且 loop_k >= 2 时结果错误；单 chunk 或无尾块时正常。
# 修复：need_clear 判定改为"dst 偏移为 0 或 tile 元素数整数倍"（slot 基址也是
# 主拷贝；拼接子拷贝的偏移不是 tile 对齐）。

import pytest
import torch

import tilelang
import tilelang.language as T

CORE_NUM = 20


def _run_ring_slot_clear(target, M=64, N=64, K=200, CHUNK=80):
    """intrinsic 风格双 buffer 预取 matmul，K 带尾块（200 = 2*80 + 40）。"""
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 40
    S1, S2 = 2, 2
    m_num = (M + BLOCK_M - 1) // BLOCK_M
    n_num = (N + BLOCK_N - 1) // BLOCK_N
    loop_k = (K + CHUNK - 1) // CHUNK
    loop_kk = CHUNK // BLOCK_K

    @tilelang.jit(out_idx=[2], target=target)
    def kernel():
        @T.macro
        def init_flag():
            T.set_flag("mte1", "mte2", 0)
            T.set_flag("mte1", "mte2", 1)
            T.set_flag("m", "mte1", 0)
            T.set_flag("m", "mte1", 1)
            T.set_flag("fix", "m", 0)

        @T.macro
        def clear_flag():
            T.wait_flag("mte1", "mte2", 0)
            T.wait_flag("mte1", "mte2", 1)
            T.wait_flag("m", "mte1", 0)
            T.wait_flag("m", "mte1", 1)
            T.wait_flag("fix", "m", 0)

        @T.prim_func
        def main(A: T.Tensor([M, K], "float32"), B: T.Tensor([K, N], "float32"), C: T.Tensor([M, N], "float32")):
            with T.Kernel(CORE_NUM, is_npu=True) as (cid, vid):
                A_L1 = T.alloc_L1([S1, BLOCK_M, CHUNK], "float32")
                B_L1 = T.alloc_L1([S1, CHUNK, BLOCK_N], "float32")
                A_L0 = T.alloc_L0A([S2, BLOCK_M, BLOCK_K], "float32")
                B_L0 = T.alloc_L0B([S2, BLOCK_K, BLOCK_N], "float32")
                C_L0 = T.alloc_L0C([BLOCK_M, BLOCK_N], "float")
                with T.Scope("C"):
                    init_flag()
                    for i in T.serial((m_num * n_num + CORE_NUM - 1) // CORE_NUM):
                        task = i * CORE_NUM + cid
                        if task < m_num * n_num:
                            bx = task // n_num
                            by = task % n_num
                            T.wait_flag("mte1", "mte2", 0)
                            T.copy(A[bx * BLOCK_M : (bx + 1) * BLOCK_M, 0:CHUNK], A_L1[0, :, :])
                            T.copy(B[0:CHUNK, by * BLOCK_N : (by + 1) * BLOCK_N], B_L1[0, :, :])
                            T.set_flag("mte2", "mte1", 0)
                            T.wait_flag("fix", "m", 0)
                            for k in T.serial(loop_k):
                                if k < loop_k - 1:
                                    T.wait_flag("mte1", "mte2", (k + 1) % S1)
                                    # 尾块预取进非零偏移 slot：need_clear 必须为 true
                                    T.copy(
                                        A[bx * BLOCK_M : (bx + 1) * BLOCK_M, (k + 1) * CHUNK : (k + 2) * CHUNK], A_L1[(k + 1) % S1, :, :]
                                    )
                                    T.copy(
                                        B[(k + 1) * CHUNK : (k + 2) * CHUNK, by * BLOCK_N : (by + 1) * BLOCK_N], B_L1[(k + 1) % S1, :, :]
                                    )
                                    T.set_flag("mte2", "mte1", (k + 1) % S1)
                                for kk in T.serial(loop_kk):
                                    if kk == 0:
                                        T.wait_flag("mte2", "mte1", k % S1)
                                    T.wait_flag("m", "mte1", kk % S2)
                                    T.copy(A_L1[k % S1, :, kk * BLOCK_K : (kk + 1) * BLOCK_K], A_L0[kk % S2, :, :])
                                    T.copy(B_L1[k % S1, kk * BLOCK_K : (kk + 1) * BLOCK_K, :], B_L0[kk % S2, :, :])
                                    if kk == loop_kk - 1:
                                        T.set_flag("mte1", "mte2", k % S1)
                                    T.set_flag("mte1", "m", kk % S2)
                                    T.wait_flag("mte1", "m", kk % S2)
                                    T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :], C_L0, init=T.And(k == 0, kk == 0))
                                    T.set_flag("m", "mte1", kk % S2)
                            T.set_flag("m", "fix", 0)
                            T.wait_flag("m", "fix", 0)
                            T.copy(C_L0, C[bx * BLOCK_M : (bx + 1) * BLOCK_M, by * BLOCK_N : (by + 1) * BLOCK_N])
                            T.set_flag("fix", "m", 0)
                    clear_flag()

        return main

    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float32).npu()
    b = torch.randn(K, N, dtype=torch.float32).npu()
    got = kernel()(a, b)
    torch.testing.assert_close(got, a @ b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("target", ["ascendc"])
def test_gm_to_l1_ring_slot_need_clear(target):
    _run_ring_slot_clear(target)


if __name__ == "__main__":
    test_gm_to_l1_ring_slot_need_clear("ascendc")
    print("ascendc PASS")
