# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

dtype = "float32"


# ---------------------------------------------------------------------------
#  1-D atomic add (Expert Mode — alloc_ub, memref path)
# ---------------------------------------------------------------------------
def atomic_add_1d(N, block_size, dtype="float32"):
    n_blocks = N // block_size

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(n_blocks, is_npu=True) as (bid, _):
            A_ub = T.alloc_ub((block_size,), dtype)

            start = bid * block_size
            remaining = N - start
            tail_size = T.min(block_size, remaining)

            T.copy(A[start : start + tail_size], A_ub[0:tail_size])
            T.npuir_atomic_add(B[start], A_ub, [tail_size])

    return main


# ---------------------------------------------------------------------------
#  2-D atomic add (Expert Mode)
# ---------------------------------------------------------------------------
def atomic_add_2d(M, N, block_M, block_N, dtype="float32"):
    m_blocks = M // block_M
    n_blocks = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_blocks * n_blocks, is_npu=True) as (cid, _):
            bx = (cid // n_blocks) * block_M
            by = (cid % n_blocks) * block_N
            A_ub = T.alloc_ub((block_M, block_N), dtype)

            tile_M = T.min(block_M, M - bx)
            tile_N = T.min(block_N, N - by)

            T.copy(
                A[bx : bx + tile_M, by : by + tile_N],
                A_ub[0:tile_M, 0:tile_N],
            )
            T.npuir_atomic_add(B[bx, by], A_ub, [tile_M, tile_N])

    return main


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------
def test_atomic_add_1d():
    torch.npu.set_device(0)
    N = 64

    a = torch.randn(N, dtype=eval("torch." + dtype)).npu()
    b = torch.randn(N, dtype=eval("torch." + dtype)).npu()
    expected = a + b

    func = atomic_add_1d(N, block_size=32)
    kernel = tilelang.compile(func, target="npuir")
    kernel(a, b)

    print("1-D atomic add:")
    print(f"  expected[:8] = {expected[:8]}")
    print(f"  actual  [:8] = {b[:8]}")
    torch.testing.assert_close(b, expected, rtol=1e-5, atol=1e-5)
    print("  \033[92mPASSED\033[0m")


def test_atomic_add_2d():
    torch.npu.set_device(0)
    M, N = 256, 256

    a = torch.randn(M, N, dtype=eval("torch." + dtype)).npu()
    b = torch.randn(M, N, dtype=eval("torch." + dtype)).npu()
    expected = a + b

    func = atomic_add_2d(M, N, block_M=16, block_N=16)
    kernel = tilelang.compile(func, target="npuir")
    kernel(a, b)

    print("2-D atomic add:")
    print(f"  expected[:4,:4] = {expected[:4, :4]}")
    print(f"  actual  [:4,:4] = {b[:4, :4]}")
    torch.testing.assert_close(b, expected, rtol=1e-5, atol=1e-5)
    print("  \033[92mPASSED\033[0m")


if __name__ == "__main__":
    test_atomic_add_1d()
    test_atomic_add_2d()
