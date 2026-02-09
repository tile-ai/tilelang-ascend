# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

"""
本文件专注测试 Cube 核上的「切片」GM→L1→L0C→GM 流程：
    GM (某一行切片) --npuir_load_nd2nz--> L1
    L1×L1 (利用 npuir_dot) --> L0C
    L0C --npuir_store_fixpipe--> GM(同一行位置)

思路：构造 A 为 [M, N]，只有第 idx 行非零；B 为 [N, N] 单位阵。
在 Cube 中用：
    T.npuir_load_nd2nz(A[idx, 0], l1_a, [1, N])
    T.npuir_load_nd2nz(B[0, 0],   l1_b, [N, N])
    T.npuir_dot(l1_a, l1_b, l0_c, ..., b_transpose=True, size=[1, N, N])
    T.npuir_store_fixpipe(l0_c, Out[idx, 0], size=[1, N], enable_nz2nd=True)
则 Out 的第 idx 行应等于 A 的第 idx 行，其余行为 0。
"""


@tilelang.jit(out_idx=[-1], target="npuir")
def cube_sliced_copy_2d(M, N, idx, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A_in: T.Tensor((M, N), dtype),
        B_in: T.Tensor((N, N), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, subid):
            # 只需要处理一行切片，因此 L1 / L0C 的形状都是 [1, N] 或 [N, N]
            l1_a = T.alloc_L1([1, N], dtype)
            l1_b = T.alloc_L1([N, N], dtype)
            l0_c = T.alloc_L0C([1, N], accum_dtype)

            with T.Scope("Cube"):
                tail_m = 1
                tail_n = N
                tail_k = N

                # GM -> L1：加载 A 的第 idx 行切片 和 B 的整块（单位阵）
                T.npuir_load_nd2nz(A_in[idx, 0], l1_a, [tail_m, tail_k])
                T.npuir_load_nd2nz(B_in[0, 0], l1_b, [tail_n, tail_k])

                # L1×L1 -> L0C：通过 Cube dot 把数据“走一遍 L0C”
                T.npuir_dot(
                    l1_a,
                    l1_b,
                    l0_c,
                    initC=True,
                    b_transpose=True,
                    size=[tail_m, tail_k, tail_n],
                )

                # L0C -> GM：通过 fixpipe 写回 GM 对应的那一行
                with T.rs("PIPE_FIX"):
                    T.npuir_store_fixpipe(
                        l0_c,
                        Out[idx, 0],
                        size=[tail_m, tail_n],
                        enable_nz2nd=True,
                    )

    return main


def test_cube_sliced_copy_2d():
    print("=" * 30 + " Running Cube Sliced Copy 2D Test " + "=" * 30)

    M, N = 16, 32
    idx = 5

    kernel = cube_sliced_copy_2d(M, N, idx)

    # A：只有第 idx 行是非零数据，其余行为 0
    A = torch.zeros(M, N).npu().half()
    row = torch.arange(1, N + 1, dtype=torch.float16).npu()
    A[idx] = row

    # B：单位阵，使得 A[idx] @ B^T == A[idx]
    B = torch.eye(N, dtype=torch.float16).npu()

    Out = torch.zeros(M, N).npu().half()

    # 执行 kernel：应当只通过 Cube 管线把第 idx 行写回 Out 对应行
    kernel(A, B, Out)

    expected = torch.zeros(M, N).npu().half()
    expected[idx] = row

    torch.testing.assert_close(Out, expected, rtol=1e-5, atol=1e-5)

    print("Cube Sliced Copy 2D Test Passed!")


def main():
    test_cube_sliced_copy_2d()


if __name__ == "__main__":
    main()