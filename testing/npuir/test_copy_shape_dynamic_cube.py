# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""
Cube core dynamic-tail-block copy tests (nd2nz → dot → fixpipe).

For each rank (2D / 3D / 4D):
  1. Tile the M×N region with block_M × block_N tiles.
  2. The last tile in each dimension uses a dynamic tail:
       tile_size_N = min(block_N, shape_N - by)
  3. Each tile goes through: GM → nd2nz → L1 → dot(identity) → L0C → fixpipe → GM
  4. Verify the output matches the input (identity copy).

Edge cases covered via pytest parametrize:
  - Exact division (no tail block)
  - Single tile (block == dim)
  - Oversized block (block > dim)
  - M = 1  (single row)
  - Tail of 1 in M / N / both  (e.g. 7 % 3 = 1)
  - Larger irregular shapes
  - Leading dimensions = 1  (3D: [1,M,N], 4D: [1,1,M,N])
"""
import pytest
import torch
import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

dtype = "float16"

# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------

def cube_copy_shape_1d_2d(M, N, block_M, block_N):
    """Dynamic tail block copy: 2D GM [M, N] → 2D GM [M, N]."""

    @T.prim_func
    def copyShapeCube(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        Id: T.Tensor((N, N), dtype),
        shape_M: T.int32,
        shape_N: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True
        ) as (cid, _):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            l1_a = T.alloc_L1((1, block_N), dtype)
            l1_b = T.alloc_L1((block_N, block_N), dtype)
            l0_c = T.alloc_L0C((1, block_N), "float32")

            with T.Scope("Cube"):
                for i in T.Parallel(block_M):
                    bx = blockx * block_M + i
                    t0 = shape_N - by
                    tile_size_N = T.min(block_N, t0)

                    T.npuir_load_nd2nz(A[bx, by], l1_a, size=[1, tile_size_N])
                    T.npuir_load_nd2nz(Id[0, 0], l1_b, size=[tile_size_N, tile_size_N])

                    T.npuir_dot(
                        l1_a, l1_b, l0_c,
                        initC=True, b_transpose=True,
                        size=[1, tile_size_N, tile_size_N],
                    )

                    with T.rs("PIPE_FIX"):
                        T.npuir_store_fixpipe(
                            l0_c, B[bx, by],
                            size=[1, tile_size_N],
                            enable_nz2nd=True,
                        )

    return copyShapeCube


def cube_copy_shape_2d_3d(M, N, block_M, block_N):
    """Dynamic tail block copy: 3D GM [1, M, N] → 3D GM [1, M, N]."""

    @T.prim_func
    def copyShapeCube2D3D(
        A: T.Tensor((1, M, N), dtype),
        B: T.Tensor((1, M, N), dtype),
        Id: T.Tensor((N, N), dtype),
        shape_M: T.int32,
        shape_N: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True
        ) as (cid, _):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            l1_a = T.alloc_L1((1, block_N), dtype)
            l1_b = T.alloc_L1((block_N, block_N), dtype)
            l0_c = T.alloc_L0C((1, block_N), "float32")

            with T.Scope("Cube"):
                for i in T.Parallel(block_M):
                    bx = blockx * block_M + i
                    t0 = shape_N - by
                    tile_size_N = T.min(block_N, t0)

                    T.npuir_load_nd2nz(A[0, bx, by], l1_a, size=[1, 1, tile_size_N])
                    T.npuir_load_nd2nz(Id[0, 0], l1_b, size=[tile_size_N, tile_size_N])

                    T.npuir_dot(
                        l1_a, l1_b, l0_c,
                        initC=True, b_transpose=True,
                        size=[1, tile_size_N, tile_size_N],
                    )

                    with T.rs("PIPE_FIX"):
                        T.npuir_store_fixpipe(
                            l0_c, B[0, bx, by],
                            size=[1, 1, tile_size_N],
                            enable_nz2nd=True,
                        )

    return copyShapeCube2D3D


def cube_copy_shape_3d_4d(M, N, block_M, block_N):
    """Dynamic tail block copy: 4D GM [1, 1, M, N] → 4D GM [1, 1, M, N]."""

    @T.prim_func
    def copyShapeCube3D4D(
        A: T.Tensor((1, 1, M, N), dtype),
        B: T.Tensor((1, 1, M, N), dtype),
        Id: T.Tensor((N, N), dtype),
        shape_M: T.int32,
        shape_N: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True
        ) as (cid, _):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            l1_a = T.alloc_L1((1, block_N), dtype)
            l1_b = T.alloc_L1((block_N, block_N), dtype)
            l0_c = T.alloc_L0C((1, block_N), "float32")

            with T.Scope("Cube"):
                for i in T.Parallel(block_M):
                    bx = blockx * block_M + i
                    t0 = shape_N - by
                    tile_size_N = T.min(block_N, t0)

                    T.npuir_load_nd2nz(A[0, 0, bx, by], l1_a, size=[1, 1, 1, tile_size_N])
                    T.npuir_load_nd2nz(Id[0, 0], l1_b, size=[tile_size_N, tile_size_N])

                    T.npuir_dot(
                        l1_a, l1_b, l0_c,
                        initC=True, b_transpose=True,
                        size=[1, tile_size_N, tile_size_N],
                    )

                    with T.rs("PIPE_FIX"):
                        T.npuir_store_fixpipe(
                            l0_c, B[0, 0, bx, by],
                            size=[1, 1, 1, tile_size_N],
                            enable_nz2nd=True,
                        )

    return copyShapeCube3D4D


# ---------------------------------------------------------------------------
# Test shape configurations
# ---------------------------------------------------------------------------

DYNAMIC_CASES = [
    # (M,  N,  block_M, block_N)   — description
    (8,  8,  3, 3),                 # base: tail in both dims (8%3=2)
    (6,  6,  3, 3),                 # exact divide, no tail
    (8,  8,  8, 8),                 # single tile (block == dim)
    (4,  4,  8, 8),                 # oversized block (block > dim)
    (1,  8,  1, 3),                 # M=1, single row
    (16, 32, 5, 7),                 # larger, irregular tiling
    (7,  8,  3, 3),                 # M tail = 1  (7%3 = 1)
    (8,  7,  3, 3),                 # N tail = 1  (7%3 = 1)
    (7,  7,  3, 3),                 # both dims tail = 1
    (8,  8,  3, 8),                 # block_N == N, only M has tail
    (8,  8,  8, 3),                 # block_M == M, only N has tail
]


# ---------------------------------------------------------------------------
# 2D tests  —  A: [M, N]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M, N, block_M, block_N", DYNAMIC_CASES)
def test_cube_copy_shape_2d(M, N, block_M, block_N):
    func = cube_copy_shape_1d_2d(M, N, block_M, block_N)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = torch.randn(M, N, dtype=torch.float16).npu()
    v2 = torch.zeros(M, N, dtype=torch.float16).npu()
    v_ref = v1.clone()
    Id = torch.eye(N, dtype=torch.float16).npu()

    compiled_kernel(v1, v2, Id, M, N)
    torch.testing.assert_close(v2, v_ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 3D tests  —  A: [1, M, N]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M, N, block_M, block_N", DYNAMIC_CASES)
def test_cube_copy_shape_3d(M, N, block_M, block_N):
    func = cube_copy_shape_2d_3d(M, N, block_M, block_N)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = torch.randn(1, M, N, dtype=torch.float16).npu()
    v2 = torch.zeros(1, M, N, dtype=torch.float16).npu()
    v_ref = v1.clone()
    Id = torch.eye(N, dtype=torch.float16).npu()

    compiled_kernel(v1, v2, Id, M, N)
    torch.testing.assert_close(v2, v_ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 4D tests  —  A: [1, 1, M, N]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M, N, block_M, block_N", DYNAMIC_CASES)
def test_cube_copy_shape_4d(M, N, block_M, block_N):
    func = cube_copy_shape_3d_4d(M, N, block_M, block_N)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = torch.randn(1, 1, M, N, dtype=torch.float16).npu()
    v2 = torch.zeros(1, 1, M, N, dtype=torch.float16).npu()
    v_ref = v1.clone()
    Id = torch.eye(N, dtype=torch.float16).npu()

    compiled_kernel(v1, v2, Id, M, N)
    torch.testing.assert_close(v2, v_ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
