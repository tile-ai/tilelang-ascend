import os
import torch
import tilelang
import tilelang.language as T


tilelang.cache.clear_cache()
dtype = "float16"


def cube_copy_shape_1d_2d(M, N, block_M, block_N):
    """
    Dynamic tail block copy on Cube core (2D GM -> 2D GM).

    Layout:
        A: [M, N], B: [M, N], Id: [N, N]
    We copy row-by-row in tiles of size `block_N`, but the last tile in
    each row uses a dynamic `tile_size_N = min(block_N, shape_N - by)`.
    For each tile, we:
        GM (1 × tile_size_N) -> L1 -> L0C -> GM
    using:
        npuir_load_nd2nz + npuir_dot (with identity) + npuir_store_fixpipe.
    """

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

            # L1/L0C buffers for a single row tile.
            l1_a = T.alloc_L1((1, block_N), dtype)
            l1_b = T.alloc_L1((block_N, block_N), dtype)
            l0_c = T.alloc_L0C((1, block_N), "float32")

            with T.Scope("Cube"):
                for i in T.Parallel(block_M):
                    bx = blockx * block_M + i
                    # Guard against out-of-range rows.
                    if bx >= shape_M:
                        continue

                    # Dynamic tail size for N dimension.
                    t0 = shape_N - by
                    tile_size_N = T.min(block_N, t0)
                    if tile_size_N <= 0:
                        continue

                    # GM -> L1: load a 1 × tile_size_N slice.
                    T.npuir_load_nd2nz(
                        A[bx, by],
                        l1_a,
                        size=[1, tile_size_N],
                    )

                    # GM -> L1: load top-left tile_size_N × tile_size_N part of Id.
                    T.npuir_load_nd2nz(
                        Id[0, 0],
                        l1_b,
                        size=[tile_size_N, tile_size_N],
                    )

                    # L1 × L1 -> L0C: multiply by identity, effectively copying.
                    T.npuir_dot(
                        l1_a,
                        l1_b,
                        l0_c,
                        initC=True,
                        b_transpose=True,
                        size=[1, tile_size_N, tile_size_N],
                    )

                    # L0C -> GM: write back to B with dynamic tail size.
                    with T.rs("PIPE_FIX"):
                        T.npuir_store_fixpipe(
                            l0_c,
                            B[bx, by],
                            size=[1, tile_size_N],
                            enable_nz2nd=True,
                        )

    return copyShapeCube


def cube_copy_shape_2d_3d(M, N, block_M, block_N):
    """
    Dynamic tail block copy on Cube core (3D GM -> 3D GM).

    Layout:
        A: [1, M, N], B: [1, M, N], Id: [N, N]
    Similar to `cube_copy_shape_1d_2d`, but with an extra leading batch
    dimension kept at size 1. We still copy 1 × tile_size_N slices via
    the Cube pipeline.
    """

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
                    if bx >= shape_M:
                        continue

                    t0 = shape_N - by
                    tile_size_N = T.min(block_N, t0)
                    if tile_size_N <= 0:
                        continue

                    # GM -> L1: load slice [0, bx, by:by+tile_size_N].
                    T.npuir_load_nd2nz(
                        A[0, bx, by],
                        l1_a,
                        size=[1, tile_size_N],
                    )

                    # GM -> L1: identity tile for N dimension.
                    T.npuir_load_nd2nz(
                        Id[0, 0],
                        l1_b,
                        size=[tile_size_N, tile_size_N],
                    )

                    # L1 × L1 -> L0C.
                    T.npuir_dot(
                        l1_a,
                        l1_b,
                        l0_c,
                        initC=True,
                        b_transpose=True,
                        size=[1, tile_size_N, tile_size_N],
                    )

                    # L0C -> GM.
                    with T.rs("PIPE_FIX"):
                        T.npuir_store_fixpipe(
                            l0_c,
                            B[0, bx, by],
                            size=[1, tile_size_N],
                            enable_nz2nd=True,
                        )

    return copyShapeCube2D3D


def test_cube_copy_shape_1d_2d():
    # In the future, Developer mode and Expert Mode will transition smoothly
    # without requiring explicit declarations.
    torch.npu.set_device(0)

    M = 8
    N = 8
    v1 = torch.randn(size=[M, N], dtype=eval("torch." + dtype)).npu()
    v2 = torch.zeros(size=[M, N], dtype=eval("torch." + dtype)).npu()
    v_ref = v1.clone()

    func = cube_copy_shape_1d_2d(M, N, block_M=3, block_N=3)
    compiled_kernel = tilelang.compile(func, target="npuir")

    # Identity matrix for Cube dot.
    Id = torch.eye(N, dtype=eval("torch." + dtype)).npu()

    compiled_kernel(v1, v2, Id, M, N)
    print(v_ref)
    print(v2)
    torch.testing.assert_close(v2, v_ref, rtol=1e-2, atol=1e-2)
    print("\033[92mAll Cube 2D dynamic-shape checks passed!\033[0m")


def test_cube_copy_shape_2d_3d():
    # In the future, Developer mode and Expert Mode will transition smoothly
    # without requiring explicit declarations.
    torch.npu.set_device(0)

    M = 8
    N = 8
    func = cube_copy_shape_2d_3d(M, N, block_M=3, block_N=3)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = torch.randn(size=[1, M, N], dtype=eval("torch." + dtype)).npu()
    v2 = torch.zeros(size=[1, M, N], dtype=eval("torch." + dtype)).npu()
    v_ref = v1.clone()

    Id = torch.eye(N, dtype=eval("torch." + dtype)).npu()

    compiled_kernel(v1, v2, Id, M, N)

    print(v_ref)
    print(v2)
    torch.testing.assert_close(v2, v_ref, rtol=1e-2, atol=1e-2)
    print("\033[92mAll Cube 3D dynamic-shape checks passed!\033[0m")


if __name__ == "__main__":
    test_cube_copy_shape_1d_2d()
    test_cube_copy_shape_2d_3d()

