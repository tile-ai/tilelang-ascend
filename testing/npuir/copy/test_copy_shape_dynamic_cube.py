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
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

from testcommon import assert_close, build_dtype_param_combos, gen_tensor

pytestmark = [pytest.mark.copy, pytest.mark.op("copy")]

IN_DTYPES = ["float16", "float32"]
OUT_DTYPES = ["float16", "float32"]
DTYPE_COMBOS = build_dtype_param_combos(IN_DTYPES, OUT_DTYPES)

# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------

def cube_copy_shape_1d_2d(M, N, block_M, block_N, in_dtype, out_dtype):
    """Dynamic tail block copy: 2D GM [M, N] → 2D GM [M, N]."""

    @T.prim_func
    def copyShapeCube(
        A: T.Tensor((M, N), in_dtype),
        B: T.Tensor((M, N), out_dtype),
        Id: T.Tensor((N, N), in_dtype),
        shape_M: T.int32,
        shape_N: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True
        ) as (cid, _):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            l1_a = T.alloc_L1((1, block_N), in_dtype)
            l1_b = T.alloc_L1((block_N, block_N), in_dtype)
            l0_c = T.alloc_L0C((1, block_N), "float32")

            with T.Scope("Cube"):
                for i in T.serial(block_M):
                    bx = blockx * block_M + i
                    t0 = shape_N - by
                    tile_size_N = T.min(block_N, t0)

                    T.copy(A[bx:bx + 1, by:by + tile_size_N], l1_a[0:1, 0:tile_size_N])
                    T.copy(Id[0:tile_size_N, 0:tile_size_N],
                           l1_b[0:tile_size_N, 0:tile_size_N])

                    T.npuir_dot(
                        l1_a, l1_b, l0_c,
                        initC=True, b_transpose=True,
                        size=[1, tile_size_N, tile_size_N],
                    )

                    with T.rs("PIPE_FIX"):
                        T.copy(l0_c[0:1, 0:tile_size_N], B[bx:bx + 1, by:by + tile_size_N])

    return copyShapeCube


def cube_copy_shape_2d_3d(M, N, block_M, block_N, in_dtype, out_dtype):
    """Dynamic tail block copy: 3D GM [1, M, N] → 3D GM [1, M, N]."""

    @T.prim_func
    def copyShapeCube2D3D(
        A: T.Tensor((1, M, N), in_dtype),
        B: T.Tensor((1, M, N), out_dtype),
        Id: T.Tensor((N, N), in_dtype),
        shape_M: T.int32,
        shape_N: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True
        ) as (cid, _):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            l1_a = T.alloc_L1((1, block_N), in_dtype)
            l1_b = T.alloc_L1((block_N, block_N), in_dtype)
            l0_c = T.alloc_L0C((1, block_N), "float32")

            with T.Scope("Cube"):
                for i in T.serial(block_M):
                    bx = blockx * block_M + i
                    t0 = shape_N - by
                    tile_size_N = T.min(block_N, t0)

                    T.copy(A[0, bx:bx + 1, by:by + tile_size_N], l1_a[0:1, 0:tile_size_N])
                    T.copy(Id[0:tile_size_N, 0:tile_size_N],
                           l1_b[0:tile_size_N, 0:tile_size_N])

                    T.npuir_dot(
                        l1_a, l1_b, l0_c,
                        initC=True, b_transpose=True,
                        size=[1, tile_size_N, tile_size_N],
                    )

                    with T.rs("PIPE_FIX"):
                        T.copy(l0_c[0:1, 0:tile_size_N], B[0, bx:bx + 1, by:by + tile_size_N])

    return copyShapeCube2D3D


def cube_copy_shape_3d_4d(M, N, block_M, block_N, in_dtype, out_dtype):
    """Dynamic tail block copy: 4D GM [1, 1, M, N] → 4D GM [1, 1, M, N]."""

    @T.prim_func
    def copyShapeCube3D4D(
        A: T.Tensor((1, 1, M, N), in_dtype),
        B: T.Tensor((1, 1, M, N), out_dtype),
        Id: T.Tensor((N, N), in_dtype),
        shape_M: T.int32,
        shape_N: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True
        ) as (cid, _):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            l1_a = T.alloc_L1((1, block_N), in_dtype)
            l1_b = T.alloc_L1((block_N, block_N), in_dtype)
            l0_c = T.alloc_L0C((1, block_N), "float32")

            with T.Scope("Cube"):
                for i in T.serial(block_M):
                    bx = blockx * block_M + i
                    t0 = shape_N - by
                    tile_size_N = T.min(block_N, t0)

                    T.copy(A[0, 0, bx:bx + 1, by:by + tile_size_N], l1_a[0:1, 0:tile_size_N])
                    T.copy(Id[0:tile_size_N, 0:tile_size_N],
                           l1_b[0:tile_size_N, 0:tile_size_N])

                    T.npuir_dot(
                        l1_a, l1_b, l0_c,
                        initC=True, b_transpose=True,
                        size=[1, tile_size_N, tile_size_N],
                    )

                    with T.rs("PIPE_FIX"):
                        T.copy(l0_c[0:1, 0:tile_size_N],
                               B[0, 0, bx:bx + 1, by:by + tile_size_N])

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

@pytest.mark.parametrize("in_dtype, out_dtype", DTYPE_COMBOS)
@pytest.mark.parametrize("M, N, block_M, block_N", DYNAMIC_CASES)
def test_cube_copy_shape_2d(M, N, block_M, block_N, in_dtype, out_dtype):
    func = cube_copy_shape_1d_2d(M, N, block_M, block_N, in_dtype, out_dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = gen_tensor((M, N), in_dtype, kind="randn")
    v2 = gen_tensor((M, N), out_dtype, kind="zeros")
    v_ref = v1.to(dtype=v2.dtype)
    Id = torch.eye(N, dtype=v1.dtype).npu()

    compiled_kernel(v1, v2, Id, M, N)
    assert_close(v2.cpu(), v_ref.cpu(), dtype=out_dtype, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 3D tests  —  A: [1, M, N]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_dtype, out_dtype", DTYPE_COMBOS)
@pytest.mark.parametrize("M, N, block_M, block_N", DYNAMIC_CASES)
def test_cube_copy_shape_3d(M, N, block_M, block_N, in_dtype, out_dtype):
    func = cube_copy_shape_2d_3d(M, N, block_M, block_N, in_dtype, out_dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = gen_tensor((1, M, N), in_dtype, kind="randn")
    v2 = gen_tensor((1, M, N), out_dtype, kind="zeros")
    v_ref = v1.to(dtype=v2.dtype)
    Id = torch.eye(N, dtype=v1.dtype).npu()

    compiled_kernel(v1, v2, Id, M, N)
    assert_close(v2.cpu(), v_ref.cpu(), dtype=out_dtype, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 4D tests  —  A: [1, 1, M, N]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_dtype, out_dtype", DTYPE_COMBOS)
@pytest.mark.parametrize("M, N, block_M, block_N", DYNAMIC_CASES)
def test_cube_copy_shape_4d(M, N, block_M, block_N, in_dtype, out_dtype):
    func = cube_copy_shape_3d_4d(M, N, block_M, block_N, in_dtype, out_dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = gen_tensor((1, 1, M, N), in_dtype, kind="randn")
    v2 = gen_tensor((1, 1, M, N), out_dtype, kind="zeros")
    v_ref = v1.to(dtype=v2.dtype)
    Id = torch.eye(N, dtype=v1.dtype).npu()

    compiled_kernel(v1, v2, Id, M, N)
    assert_close(v2.cpu(), v_ref.cpu(), dtype=out_dtype, rtol=1e-2, atol=1e-2)
