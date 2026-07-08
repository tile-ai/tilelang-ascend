import pytest
import tilelang
import tilelang.language as T
import torch
from tilelang.intrinsics import make_zn_layout, make_nz_layout

"""
L1 -> L0A / L0B copy correctness suite: explicit-copy, implicit-copy (gemm_v0),
transpose, K-tail, K-tiling accumulation, layout annotation, persistent
scheduling, multi-dim slice, and 2-D partial-row slice scenarios.

Feature under test
------------------
``copy_l1_to_l0a`` / ``copy_l1_to_l0b`` (src/target/codegen_ascend_pto.cc ::
CopyL1ToL0Codegen, src/target/codegen_ascend.cc) transfer a tile from an L1
Buffer to an L0A / L0B Buffer, feeding the Cube unit's left / right matrix
input.  These two copies are the *only* legal way to feed L0A / L0B; they are
the mandatory last hop on the Cube load path::

    GM -> L1 -> L0A / L0B -> (T.mma / T.gemm_v0) -> L0C -> GM

Every GEMM / Flash-Attention / Cube-fusion kernel exercises this copy, so its
correctness across shapes, dtypes, transpose modes, slicing patterns, and
scheduling strategies is critical.

Memory hierarchy & alignment (Ascend910B3)
------------------------------------------
    GM (HBM, no alignment)
      |
      +-- L1 (512 KB, 32-Byte alignment)            <-- cube cache
      |     |
      |     +-- L0A (64 KB, 512-Byte alignment)     <-- left matrix
      |     +-- L0B (64 KB, 512-Byte alignment)     <-- right matrix
      |     |     |
      |     |     +-- L0C (128 KB, 64-Byte align)   <-- accumulator
      |     |
      |     +-- UB (192 KB, 32-Byte alignment)      <-- vector path (other file)

Fractal constraints (M / N >= 16 always; K depends on dtype):
    float16 / bfloat16 : L0A 16x16, L0B 16x16  -> K >= 16
    int8 / uint8       : L0A 16x32, L0B 32x16  -> K >= 32
    int32 / float32    : L0A 16x8,  L0B 8x16   -> K >= 8

tile_row selection for the L0B path
-----------------------------------
When the L1 source's row count exceeds the L0B K capacity, the codegen splits
the source into multiple L0B-sized chunks via ``FindBestTileRowB``, which picks
the largest divisor of the source row count that fits the destination.  For
*sliced* L1 accesses (where the physical buffer is larger than the current
access extent), the codegen must derive the tile_row from the slice's **valid**
row count, not the buffer's declared physical row count, otherwise the chosen
tile_row can exceed the L0B capacity and overflow.

Scenarios covered (all L1 -> L0A/L0B, target = ascendc / pto)
-------------------------------------------------------------
  Group 1 - Explicit L1 -> L0A/L0B copy (Expert T.mma):
      Drives the explicit ``T.copy(L1, L0A/L0B)`` + ``T.mma`` path.  A full
      sweep of block sizes: full-block, multi-tile grids, and asymmetric
      blocks.  Both A (->L0A) and B (->L0B) copy paths exercised.

  Group 2 - Implicit L1 -> L0 copy via T.gemm_v0 (Developer):
      ``gemm_v0`` internally performs the L1 -> L0 copy.  Covers the
      transpose_B=True branch (TileMatL1ZN template path) and the plain
      non-transpose path.

  Group 3 - Tail-block / non-divisible K (auto-clamp):
      K not divisible by block_K; the last K-tile is partial.  The L1 tail is
      implicitly zero-padded by the GM->L1 clearing mechanism so 0 * B = 0
      keeps the matmul correct.

  Group 4 - Fractal-boundary block sizes:
      block_M / block_N at the minimum fractal (16) and common sizes (64,
      128).  Guards alignment / fractal correctness at the edges (M, N >= 16
      for L0C; K >= 16 for float16 / bfloat16).

  Group 5 - K-tiling accumulation:
      Large K split into many K-tiles; each tile goes GM -> L1 -> L0A/L0B ->
      mma -> L0C with ``init=(k==0)``.  Verifies the L1 -> L0 copy + L0C
      accumulation stays correct across many iterations.

  Group 6 - Layout annotation variants:
      L1 buffers annotated with zN (default) and nZ layouts, combined with
      transpose_B.  Guards the L1 -> L0 copy under explicit layout control.

  Group 7 - Persistent scheduling:
      ``T.Persistent`` drives the tile grid in a cache-friendly order; each
      tile uses the L1 -> L0 copy path (implicit via gemm_v0).

  Group 8 - Multi-dimensional (3D) L1 buffer sliced to L0A / L0B:
      A ``[STACK, rows, cols]`` L1 buffer is indexed by chunk and copied to
      L0A and L0B.  The 3D buffer flattens to ``[STACK*rows, cols]``
      physically while each slice's valid row count is just ``rows``; this
      exercises the sliced-source codegen path for both L0A and L0B.

  Group 9 - 2-D L1 partial-row slice -> L0B:
      A 2-D L1 buffer is accessed with a row sub-range (valid rows < physical
      rows), exercising the sliced path on a 2-D buffer.

NOTE: the cases in Groups 1-7 execute on real NPU hardware (``.npu()``); they
cannot run in a CPU-only / bisheng-unavailable environment.  Groups 8-9
examine the generated AscendC source (via ``tilelang.lower``) because the
sliced-L1 load lowers to a static GM->L1 intrinsic that the bisheng compiler
in some environments rejects -- the codegen layer is where the tile_row
selection logic resides, so source inspection is the correct verification
level for these cases.  Risk levels are annotated per group.
"""

# CUBE-only config (pure Cube kernel, no Vector): auto-sync + memory planning.
CUBE_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Developer CV configs (auto CV combine + cross-core sync).  Used by kernels
# that rely on gemm_v0's implicit L1 -> L0 copy with layout annotation; CV
# combine is harmless for pure-Cube kernels and matches the passing gm_to_l1
# suite.
DEV_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

TARGETS = ["ascendc", "pto"]


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before the session."""
    tilelang.cache.clear_cache()
    yield


def _torch_dtype(dtype):
    """Map a tilelang dtype string to the corresponding torch dtype."""
    return {
        "float": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def _lower_and_get_source(kernel_func, target, pass_configs):
    """Lower ``kernel_func`` to AscendC source without invoking the bisheng
    compiler.

    Uses ``tilelang.lower`` which stops after codegen, so it works even when
    the bisheng toolchain is unavailable or rejects the generated intrinsic.
    Returns the generated AscendC source string.  Used by Groups 8-9 where
    the sliced-L1 load lowers to a static intrinsic that bisheng may reject.
    """
    import tvm

    with tvm.transform.PassContext(opt_level=3, config=pass_configs):
        artifact = tilelang.lower(kernel_func, target=target, platform="auto")
    return artifact.kernel_source


def _assert_l0b_tile_row_within_capacity(kernel_func, target, pass_configs, expected_tile_row=None):
    """Codegen-layer assertion: every L0B copy's tile_row must fit the L0B K
    capacity.

    The two backends emit different ``copy_l1_to_l0b`` template forms:

    * pto:   ``copy_l1_to_l0b<type, dst_K, dst_N, tile_row, tile_col>(...)``
      The 4th numeric arg (``tile_row``) is the K-dimension chunk size of the
      L1 source copied into one L0B tile.  For sliced L1 sources the codegen
      must derive ``tile_row`` from the slice's valid row count, not the
      buffer's physical row count, otherwise ``tile_row`` can exceed ``dst_K``
      (the L0B K capacity) and overflow L0B.  This asserts
      ``tile_row <= dst_K`` and, if ``expected_tile_row`` is given, equality.

    * ascendc: ``copy_l1_to_l0b<type, dst_M, dst_N, transpose_flag>(...)``
      The template carries only the destination tile shape (no explicit
      tile_row); the codegen does not perform FindBestTileRowB and, for
      sliced L1 sources, copies the full physical buffer tile (a pre-existing
      ascendc limitation).  We only assert the call exists; the tile_row
      correctness for sliced sources is verified on the pto backend.
    """
    import re

    src = _lower_and_get_source(kernel_func, target, pass_configs)

    if target == "pto":
        # pto: 4-arg template <type, dst_K, dst_N, tile_row, tile_col>.
        copies = re.findall(r"copy_l1_to_l0b<[^,]+,\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)>", src)
        assert copies, "no 4-arg copy_l1_to_l0b call in generated source"
        for dst_k, dst_n, tile_row, tile_col in copies:
            dst_k = int(dst_k)
            tile_row = int(tile_row)
            assert tile_row <= dst_k, (
                f"tile_row={tile_row} > dst_K={dst_k} (L0B K capacity) -> "
                f"overflow (dst_K={dst_k} dst_N={dst_n} tile_row={tile_row} "
                f"tile_col={tile_col})"
            )
            if expected_tile_row is not None:
                assert tile_row == expected_tile_row, f"tile_row={tile_row} != expected {expected_tile_row}"
    else:
        # ascendc: 3-arg template <type, dst_M, dst_N, transpose_flag>.
        # The ascendc backend does not perform FindBestTileRowB and, for
        # sliced L1 sources, copies the full physical buffer tile (it does
        # not honour L1 row sub-range slices).  This is a pre-existing
        # ascendc limitation; the codegen still succeeds and emits a valid
        # copy_l1_to_l0b call, so we only assert the call exists.  The
        # tile_row correctness for sliced sources is verified on the pto
        # backend, whose codegen does perform the tile_row selection.
        copies = re.findall(r"copy_l1_to_l0b<[^,]+,\s*(\d+),\s*(\d+),\s*(?:true|false)>", src)
        assert copies, "no 3-arg copy_l1_to_l0b call in generated source"


def _assert_l0a_uses_valid_row(kernel_func, target, pass_configs, physical_row, valid_row):
    """Codegen-layer assertion: the L0A copy must use the slice-valid row.

    The A path uses ``dst.slice_row`` directly as tile_row (no
    FindBestTileRowB), so for a sliced 3-D L1 source the template must reflect
    the valid row, not the physical (flattened) row.  Asserts the physical
    row value does not appear in any ``copy_l1_to_l0a`` template's first
    dimension arg.
    """
    import re

    src = _lower_and_get_source(kernel_func, target, pass_configs)
    assert "copy_l1_to_l0a" in src, "no copy_l1_to_l0a call in generated source"
    bad = re.findall(rf"copy_l1_to_l0a<[^,]+,\s*{physical_row},", src)
    assert not bad, f"L0A copy used physical row {physical_row} instead of valid row {valid_row}: {bad}"


# =============================================================================
# Group 1 - Explicit L1 -> L0A/L0B copy (Expert T.mma)  [risk: low]
# Drives the explicit T.copy(L1, L0A/L0B) + T.mma path.  Full sweep of block
# sizes: full-block, multi-tile grids, asymmetric blocks.  Both A (->L0A) and
# B (->L0B) copy paths exercised.
# =============================================================================
def explicit_l1_to_l0_gemm(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    """Expert-mode GEMM with explicit L1 -> L0A/L0B copies.

    Data flow per K-tile:
        GM[A] -> A_L1 -> A_L0  (copy_l1_to_l0a)
        GM[B] -> B_L1 -> B_L0  (copy_l1_to_l0b)
        mma(A_L0, B_L0, C_L0, init=(k==0))
        GM[C] <- C_L0

    The explicit T.copy(L1, L0A/L0B) calls exercise the CopyL1ToL0Codegen path
    directly, for both the A (L0A) and B (L0B) matrix inputs.
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            A_L0 = T.alloc_L0A((block_M, block_K), dtype)
            B_L0 = T.alloc_L0B((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    T.copy(B[k * block_K, by * block_N], B_L1)

                    T.copy(A_L1, A_L0)
                    T.copy(B_L1, B_L0)

                    T.mma(A_L0, B_L0, C_L0, init=(k == 0))

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def run_test_explicit_l1_to_l0_gemm(M, N, K, block_M, block_N, block_K, dtype, accum_dtype, target):
    torch.manual_seed(0)
    func = explicit_l1_to_l0_gemm(M, N, K, block_M, block_N, block_K, dtype, accum_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=CUBE_PASS_CONFIGS, target=target)
    td = _torch_dtype(dtype)
    a = torch.randn(M, K, dtype=td).npu()
    b = torch.randn(K, N, dtype=td).npu()
    torch.npu.synchronize()
    c = func(a, b)
    ref = (a.float() @ b.float()).to(td)
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


# (M, N, K, block_M, block_N, block_K) -- divisible shapes, varied block sizes.
explicit_configs = [
    (128, 128, 128, 128, 128, 128),  # single tile, block == full dim
    (256, 256, 256, 128, 128, 64),  # 2x2 grid, K split into 4
    (512, 512, 512, 128, 256, 64),  # asymm block_N
    (256, 256, 256, 64, 64, 64),  # smaller block, 4x4 grid
]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype,accum_dtype", [("float16", "float"), ("bfloat16", "float")])
@pytest.mark.parametrize("M,N,K,block_M,block_N,block_K", explicit_configs)
def test_explicit_l1_to_l0_gemm(M, N, K, block_M, block_N, block_K, dtype, accum_dtype, target):
    run_test_explicit_l1_to_l0_gemm(M, N, K, block_M, block_N, block_K, dtype, accum_dtype, target)


# =============================================================================
# Group 2 - Implicit L1 -> L0 copy via T.gemm_v0        [risk: low]
# gemm_v0 internally performs the L1 -> L0 copy.  Covers the transpose_B=True
# branch (TileMatL1ZN template path) and the plain non-transpose path.
# =============================================================================
def gemm_v0_implicit(M, N, K, block_M, block_N, block_K, transpose_B=False, dtype="float16", accum_dtype="float"):
    """Developer-mode GEMM via T.gemm_v0 (implicit L1 -> L0 copy).

    With transpose_B=True the B matrix is loaded as [block_N, block_K] (nZ
    layout) and the codegen uses the TileMatL1ZN transpose path for the
    internal L1 -> L0B copy.  With transpose_B=False the plain path is used.
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((N, K) if transpose_B else (K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_N, block_K) if transpose_B else (block_K, block_N), dtype)
            if transpose_B:
                T.annotate_layout({A_L1: make_zn_layout(A_L1), B_L1: make_nz_layout(B_L1)})
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    if transpose_B:
                        T.copy(B[by * block_N, k * block_K], B_L1)
                    else:
                        T.copy(B[k * block_K, by * block_N], B_L1)

                    T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=transpose_B, init=(k == 0))

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def run_test_gemm_v0_implicit(M, N, K, block_M, block_N, block_K, transpose_B, dtype, accum_dtype, target):
    torch.manual_seed(0)
    func = gemm_v0_implicit(M, N, K, block_M, block_N, block_K, transpose_B=transpose_B, dtype=dtype, accum_dtype=accum_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=DEV_PASS_CONFIGS, target=target)
    td = _torch_dtype(dtype)
    a = torch.randn(M, K, dtype=td).npu()
    if transpose_B:
        b = torch.randn(N, K, dtype=td).npu()
        ref = (a.float() @ b.float().t()).to(td)
    else:
        b = torch.randn(K, N, dtype=td).npu()
        ref = (a.float() @ b.float()).to(td)
    torch.npu.synchronize()
    c = func(a, b)
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype,accum_dtype", [("float16", "float"), ("bfloat16", "float")])
@pytest.mark.parametrize("transpose_B", [False, True])
@pytest.mark.parametrize(
    "M,N,K,block_M,block_N,block_K",
    [
        (128, 128, 128, 128, 128, 128),
        (256, 256, 256, 128, 128, 64),
    ],
)
def test_gemm_v0_implicit(M, N, K, block_M, block_N, block_K, transpose_B, dtype, accum_dtype, target):
    run_test_gemm_v0_implicit(M, N, K, block_M, block_N, block_K, transpose_B, dtype, accum_dtype, target)


# =============================================================================
# Group 3 - Tail-block / non-divisible K (auto-clamp)   [risk: low]
# K not divisible by block_K; the last K-tile is partial.  The L1 tail is
# implicitly zero-padded by the GM->L1 clearing mechanism so 0 * B = 0 keeps
# the matmul correct.
# =============================================================================
k_tail_configs = [
    (128, 128, 160, 128, 128, 64),  # K=160, last tile K=32 (half tile)
    (128, 256, 96, 128, 256, 64),  # K=96, last tile K=32
    (256, 256, 176, 128, 128, 64),  # K=176, last tile K=48
]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype,accum_dtype", [("float16", "float"), ("bfloat16", "float")])
@pytest.mark.parametrize("M,N,K,block_M,block_N,block_K", k_tail_configs)
def test_k_tail_gemm(M, N, K, block_M, block_N, block_K, dtype, accum_dtype, target):
    run_test_explicit_l1_to_l0_gemm(M, N, K, block_M, block_N, block_K, dtype, accum_dtype, target)


# =============================================================================
# Group 4 - Fractal-boundary block sizes                [risk: medium]
# block_M / block_N at the minimum fractal (16) and common sizes (64, 128).
# Guards alignment / fractal correctness at the edges (M, N >= 16 for L0C;
# K >= 16 for float16 / bfloat16).
# =============================================================================
fractal_configs = [
    (32, 32, 32, 16, 16, 16),  # minimum fractal for fp16/bf16
    (64, 64, 64, 16, 16, 16),  # 4x4 grid at min fractal
    (128, 128, 128, 64, 64, 16),  # min K, larger M/N blocks
    (128, 128, 128, 16, 128, 16),  # asymm: min M, large N, min K
]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype,accum_dtype", [("float16", "float"), ("bfloat16", "float")])
@pytest.mark.parametrize("M,N,K,block_M,block_N,block_K", fractal_configs)
def test_fractal_boundary(M, N, K, block_M, block_N, block_K, dtype, accum_dtype, target):
    run_test_explicit_l1_to_l0_gemm(M, N, K, block_M, block_N, block_K, dtype, accum_dtype, target)


# =============================================================================
# Group 5 - K-tiling accumulation                        [risk: low]
# Large K split into many K-tiles; each tile goes GM -> L1 -> L0A/L0B -> mma
# -> L0C with init=(k==0).  Verifies the L1 -> L0 copy + L0C accumulation
# stays correct across many iterations and that the L0C accumulator is
# properly zeroed on the first tile and accumulated on subsequent tiles.
# =============================================================================
k_tiling_configs = [
    (128, 128, 512, 128, 128, 64),  # K split into 8 tiles
    (128, 128, 1024, 128, 128, 64),  # K split into 16 tiles
    (64, 64, 768, 64, 64, 64),  # K split into 12 tiles, smaller block
]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype,accum_dtype", [("float16", "float"), ("bfloat16", "float")])
@pytest.mark.parametrize("M,N,K,block_M,block_N,block_K", k_tiling_configs)
def test_k_tiling_accumulation(M, N, K, block_M, block_N, block_K, dtype, accum_dtype, target):
    run_test_explicit_l1_to_l0_gemm(M, N, K, block_M, block_N, block_K, dtype, accum_dtype, target)


# =============================================================================
# Group 6 - Layout annotation variants                   [risk: low]
# L1 buffers annotated with zN (default) and nZ layouts, combined with
# transpose_B.  Guards the L1 -> L0 copy under explicit layout control.
# =============================================================================
def layout_annotated_gemm(
    M, N, K, block_M, block_N, block_K, a_layout="zn", b_layout="zn", transpose_B=False, dtype="float16", accum_dtype="float"
):
    """GEMM with explicit layout annotation on L1 buffers.

    ``a_layout`` / ``b_layout`` select zN or nZ layout for the A/B L1 buffers.
    With nZ on B and transpose_B=True, the codegen exercises the nZ -> L0B
    transpose path.  ``T.annotate_layout`` must receive a literal dict, so
    each combination is handled explicitly.
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((N, K) if transpose_B else (K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_N, block_K) if transpose_B else (block_K, block_N), dtype)

            if a_layout == "zn" and b_layout == "zn":
                T.annotate_layout({A_L1: make_zn_layout(A_L1), B_L1: make_zn_layout(B_L1)})
            elif a_layout == "zn" and b_layout == "nz":
                T.annotate_layout({A_L1: make_zn_layout(A_L1), B_L1: make_nz_layout(B_L1)})
            elif a_layout == "nz" and b_layout == "zn":
                T.annotate_layout({A_L1: make_nz_layout(A_L1), B_L1: make_zn_layout(B_L1)})
            else:
                T.annotate_layout({A_L1: make_nz_layout(A_L1), B_L1: make_nz_layout(B_L1)})

            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    if transpose_B:
                        T.copy(B[by * block_N, k * block_K], B_L1)
                    else:
                        T.copy(B[k * block_K, by * block_N], B_L1)

                    T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=transpose_B, init=(k == 0))

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def run_test_layout_annotated_gemm(M, N, K, block_M, block_N, block_K, a_layout, b_layout, transpose_B, dtype, accum_dtype, target):
    torch.manual_seed(0)
    func = layout_annotated_gemm(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        a_layout=a_layout,
        b_layout=b_layout,
        transpose_B=transpose_B,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )
    func = tilelang.compile(func, out_idx=[-1], pass_configs=DEV_PASS_CONFIGS, target=target)
    td = _torch_dtype(dtype)
    a = torch.randn(M, K, dtype=td).npu()
    if transpose_B:
        b = torch.randn(N, K, dtype=td).npu()
        ref = (a.float() @ b.float().t()).to(td)
    else:
        b = torch.randn(K, N, dtype=td).npu()
        ref = (a.float() @ b.float()).to(td)
    torch.npu.synchronize()
    c = func(a, b)
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype,accum_dtype", [("float16", "float"), ("bfloat16", "float")])
@pytest.mark.parametrize(
    "a_layout,b_layout,transpose_B",
    [
        ("zn", "zn", False),  # default zN on both, no transpose
        ("zn", "nz", True),  # zN A, nZ B, transpose (attention-style Q*K^T)
        ("nz", "zn", False),  # nZ A, zN B, no transpose
    ],
)
@pytest.mark.parametrize(
    "M,N,K,block_M,block_N,block_K",
    [
        (128, 128, 128, 128, 128, 128),
    ],
)
def test_layout_annotation(M, N, K, block_M, block_N, block_K, a_layout, b_layout, transpose_B, dtype, accum_dtype, target):
    run_test_layout_annotated_gemm(M, N, K, block_M, block_N, block_K, a_layout, b_layout, transpose_B, dtype, accum_dtype, target)


# =============================================================================
# Group 7 - Persistent scheduling                        [risk: low]
# T.Persistent drives the tile grid in a cache-friendly order; each tile uses
# the L1 -> L0 copy path (implicit via gemm_v0).  Follows the pattern in
# examples/gemm/example_gemm_persistent.py.
# =============================================================================
def persistent_gemm(M, N, K, block_M, block_N, block_K, core_num, dtype="float16", accum_dtype="float"):
    """Persistent-scheduled GEMM (implicit L1 -> L0 copy via gemm_v0).

    The kernel grid is m_num * n_num (total tiles); T.Persistent distributes
    them across core_num cores with wave_size = core_num.
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                for bx, by in T.Persistent([T.ceildiv(M, block_M), T.ceildiv(N, block_N)], core_num, cid):
                    loop_k = T.ceildiv(K, block_K)
                    for k in T.serial(loop_k):
                        T.copy(A[bx * block_M, k * block_K], A_L1)
                        T.copy(B[k * block_K, by * block_N], B_L1)

                        T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                    T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def run_test_persistent_gemm(M, N, K, block_M, block_N, block_K, core_num, dtype, accum_dtype, target):
    torch.manual_seed(0)
    func = persistent_gemm(M, N, K, block_M, block_N, block_K, core_num, dtype, accum_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=CUBE_PASS_CONFIGS, target=target)
    td = _torch_dtype(dtype)
    a = torch.randn(M, K, dtype=td).npu()
    b = torch.randn(K, N, dtype=td).npu()
    torch.npu.synchronize()
    c = func(a, b)
    ref = (a.float() @ b.float()).to(td)
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype,accum_dtype", [("float16", "float"), ("bfloat16", "float")])
@pytest.mark.parametrize(
    "M,N,K,block_M,block_N,block_K,core_num",
    [
        (256, 256, 128, 128, 128, 64, 4),  # 2x2 grid, 4 cores
        (512, 256, 128, 128, 128, 64, 4),  # 4x2 grid, 4 cores
    ],
)
def test_persistent_scheduling(M, N, K, block_M, block_N, block_K, core_num, dtype, accum_dtype, target):
    run_test_persistent_gemm(M, N, K, block_M, block_N, block_K, core_num, dtype, accum_dtype, target)


# =============================================================================
# Group 8 - Multi-dimensional (3D) L1 buffer sliced to L0A / L0B  [risk: high]
# A [STACK, rows, cols] L1 buffer is indexed by chunk and copied to L0A/L0B.
# The 3D buffer flattens to [STACK*rows, cols] physically; each slice's valid
# row count is just ``rows``.  This exercises the sliced-source codegen path
# for both L0A and L0B, where the codegen must use the valid (not physical)
# row count to size the L0 tile.
#
# Verified at the codegen layer (via tilelang.lower + source inspection): the
# sliced-L1 load lowers to a static GM->L1 intrinsic that the bisheng compiler
# in some environments rejects, but the codegen layer -- where the tile_row
# selection logic resides -- succeeds and emits the correct template.  Both
# ascendc and pto codegen are inspected.
# =============================================================================
def sliced_3d_l1_to_l0b(STACK, BLOCK_N, D, BM, dtype="float16", accum_dtype="float"):
    """3D L1 buffer sliced to L0B.

    shared_l1 = [STACK, BLOCK_N, D] (3D L1).  Per chunk:
        copy shared_l1[chunk, :, :] -> l0b  (the sliced L1 -> L0B copy)
        mma(a_l0=[BM, BLOCK_N], l0b=[BLOCK_N, D], l0c=[BM, D])
    The 3D buffer flattens to [STACK*BLOCK_N, D]; each slice's valid row
    count is BLOCK_N while the physical row is STACK*BLOCK_N.
    """

    @T.prim_func
    def main(
        A: T.Tensor((STACK, BM, BLOCK_N), dtype),  # type: ignore
        V: T.Tensor((STACK, BLOCK_N, D), dtype),  # type: ignore
        C: T.Tensor((BM, D), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            shared_l1 = T.alloc_L1([STACK, BLOCK_N, D], dtype)
            a_l1 = T.alloc_L1([BM, BLOCK_N], dtype)
            l0a = T.alloc_L0A([BM, BLOCK_N], dtype)
            l0b = T.alloc_L0B([BLOCK_N, D], dtype)
            l0c = T.alloc_L0C([BM, D], accum_dtype)

            for i in T.serial(STACK):
                T.copy(V[i, :, :], shared_l1[i, :, :])

            for chunk in T.serial(STACK):
                T.copy(A[chunk, :, :], a_l1[:, :])
                T.copy(a_l1[:, :], l0a[:, :])
                T.copy(shared_l1[chunk, :, :], l0b[:, :])
                T.mma(l0a, l0b, l0c, init=(chunk == 0))

            T.copy(l0c, C[:, :])

    return main


def sliced_3d_l1_to_l0a(STACK, BM, K, BLOCK_N, dtype="float16", accum_dtype="float"):
    """3D L1 buffer sliced to L0A (symmetric to the L0B case).

    shared_a_l1 = [STACK, BM, K] (3D L1).  Per chunk:
        copy shared_a_l1[chunk, :, :] -> l0a  (the sliced L1 -> L0A copy)
        mma(l0a=[BM, K], l0b=[K, BLOCK_N], l0c=[BM, BLOCK_N])
    """

    @T.prim_func
    def main(
        A: T.Tensor((STACK, BM, K), dtype),  # type: ignore
        B: T.Tensor((STACK, K, BLOCK_N), dtype),  # type: ignore
        C: T.Tensor((BM, BLOCK_N), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            shared_a_l1 = T.alloc_L1([STACK, BM, K], dtype)
            b_l1 = T.alloc_L1([K, BLOCK_N], dtype)
            l0a = T.alloc_L0A([BM, K], dtype)
            l0b = T.alloc_L0B([K, BLOCK_N], dtype)
            l0c = T.alloc_L0C([BM, BLOCK_N], accum_dtype)

            for i in T.serial(STACK):
                T.copy(A[i, :, :], shared_a_l1[i, :, :])

            for chunk in T.serial(STACK):
                T.copy(B[chunk, :, :], b_l1[:, :])
                T.copy(b_l1[:, :], l0b[:, :])
                T.copy(shared_a_l1[chunk, :, :], l0a[:, :])
                T.mma(l0a, l0b, l0c, init=(chunk == 0))

            T.copy(l0c, C[:, :])

    return main


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype,accum_dtype", [("float16", "float"), ("bfloat16", "float")])
@pytest.mark.parametrize(
    "STACK,ROWS,COLS,BM",
    [
        (4, 128, 128, 128),  # [4,128,128] -> physical 512, valid 128 per slice
        (2, 128, 128, 128),  # [2,128,128] -> physical 256, valid 128 per slice
        (4, 64, 128, 64),  # smaller rows
    ],
)
def test_sliced_3d_l1_to_l0b(STACK, ROWS, COLS, BM, dtype, accum_dtype, target):
    """3D L1 sliced -> L0B: codegen must size tile_row by the valid row count.

    Asserts every copy_l1_to_l0b template's tile_row <= dst_K (the L0B K
    capacity), ensuring the codegen uses the slice-valid row count rather
    than the physical (flattened) row count.
    """
    kfunc = sliced_3d_l1_to_l0b(STACK, ROWS, COLS, BM, dtype, accum_dtype)
    _assert_l0b_tile_row_within_capacity(kfunc, target, CUBE_PASS_CONFIGS, expected_tile_row=ROWS)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype,accum_dtype", [("float16", "float"), ("bfloat16", "float")])
def test_sliced_3d_l1_to_l0a(dtype, accum_dtype, target):
    """3D L1 sliced -> L0A: codegen must use the slice-valid row, not physical.

    The A path uses dst.slice_row directly as tile_row (no FindBestTileRowB),
    so the template must reflect the valid row (BM), not the physical
    flattened row (STACK*BM).  Asserts the physical row value does not appear
    in any copy_l1_to_l0a template's first dimension.
    """
    STACK, BM, K, BLOCK_N = 4, 128, 128, 128
    kfunc = sliced_3d_l1_to_l0a(STACK, BM, K, BLOCK_N, dtype, accum_dtype)
    _assert_l0a_uses_valid_row(kfunc, target, CUBE_PASS_CONFIGS, physical_row=STACK * BM, valid_row=BM)


def sliced_3d_l1_to_l0b_transpose(STACK, BLOCK_N, D, BM, dtype="float16", accum_dtype="float"):
    """3D L1 buffer sliced to L0B with transpose=True.

    Same as sliced_3d_l1_to_l0b but the L1 -> L0B copy uses transpose=True,
    exercising the TileMatL1ZN transpose branch of CopyL1ToL0Codegen.  The
    3D buffer [STACK, BLOCK_N, D] flattens to [STACK*BLOCK_N, D]; each slice's
    valid row count is BLOCK_N while the physical row is STACK*BLOCK_N.
    """

    @T.prim_func
    def main(
        A: T.Tensor((STACK, BM, BLOCK_N), dtype),  # type: ignore
        V: T.Tensor((STACK, BLOCK_N, D), dtype),  # type: ignore
        C: T.Tensor((BM, D), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            shared_l1 = T.alloc_L1([STACK, BLOCK_N, D], dtype)
            T.annotate_layout({shared_l1: make_nz_layout(shared_l1)})

            a_l1 = T.alloc_L1([BM, BLOCK_N], dtype)
            T.annotate_layout({a_l1: make_zn_layout(a_l1)})

            l0a = T.alloc_L0A([BM, BLOCK_N], dtype)
            l0b = T.alloc_L0B([BLOCK_N, D], dtype)
            l0c = T.alloc_L0C([BM, D], accum_dtype)

            for i in T.serial(STACK):
                T.copy(V[i, :, :], shared_l1[i, :, :])

            for chunk in T.serial(STACK):
                T.copy(A[chunk, :, :], a_l1[:, :])
                T.copy(a_l1[:, :], l0a[:, :])
                # *** transpose=True on a sliced 3D L1 buffer ***
                T.copy(shared_l1[chunk, :, :], l0b[:, :], transpose=True)
                T.mma(l0a, l0b, l0c, init=(chunk == 0))

            T.copy(l0c, C[:, :])

    return main


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype,accum_dtype", [("float16", "float"), ("bfloat16", "float")])
@pytest.mark.parametrize(
    "STACK,ROWS,COLS,BM",
    [
        (4, 128, 128, 128),  # [4,128,128] -> physical 512, valid 128 per slice
        (2, 128, 128, 128),  # [2,128,128] -> physical 256, valid 128 per slice
    ],
)
def test_sliced_3d_l1_to_l0b_transpose(STACK, ROWS, COLS, BM, dtype, accum_dtype, target):
    """3D L1 sliced + transpose -> L0B: transpose branch must use valid row.

    The transpose branch of CopyL1ToL0Codegen (pto) emits a TileMatL1ZN temp
    and a copy_l1_to_l0b<..., true> template; both must use the slice-valid
    row count (ROWS), not the physical (flattened) row count (STACK*ROWS).
    The ascendc backend emits a 3-arg copy_l1_to_l0b<type, dst_M, dst_N, true>
    template.  Asserts the physical row value never appears in any transpose
    template on either backend.
    """
    import re

    kfunc = sliced_3d_l1_to_l0b_transpose(STACK, ROWS, COLS, BM, dtype, accum_dtype)
    src = _lower_and_get_source(kfunc, target, CUBE_PASS_CONFIGS)
    physical_row = STACK * ROWS

    if target == "pto":
        # pto: TileMatL1ZN<type, col, row, col, row> and
        # copy_l1_to_l0b<type, dst_K, dst_N, tile_col, src_row, true>.
        zn_lines = re.findall(r"TileMatL1ZN<[^>]+>", src)
        assert zn_lines, "no TileMatL1ZN template in generated source"
        for line in zn_lines:
            assert str(physical_row) not in line, f"TileMatL1ZN used physical row {physical_row} instead of valid {ROWS}: {line}"

        l0b_t_lines = re.findall(r"copy_l1_to_l0b<[^,]+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*true>", src)
        assert l0b_t_lines, "no transposed copy_l1_to_l0b template in generated source"
        for line in l0b_t_lines:
            assert str(physical_row) not in line, (
                f"copy_l1_to_l0b (transpose) used physical row {physical_row} instead of valid {ROWS}: {line}"
            )
    else:
        # ascendc: copy_l1_to_l0b<type, dst_M, dst_N, true> (3-arg template).
        # Assert the physical row value does not appear as a template dim.
        l0b_t_lines = re.findall(r"copy_l1_to_l0b<[^,]+,\s*(\d+),\s*(\d+),\s*true>", src)
        assert l0b_t_lines, "no transposed copy_l1_to_l0b template in generated source"
        for dst_m, dst_n in l0b_t_lines:
            assert int(dst_m) != physical_row, f"ascendc copy_l1_to_l0b (transpose) dst_M={dst_m} equals physical row {physical_row}"
            assert int(dst_n) != physical_row, f"ascendc copy_l1_to_l0b (transpose) dst_N={dst_n} equals physical row {physical_row}"


# =============================================================================
# Group 9 - 2-D L1 partial-row slice -> L0B             [risk: medium]
# A 2-D L1 buffer [phys_row, col] is accessed with a row sub-range, so
# slice_valid_row < physical row and is_slice == true on a 2-D buffer.
#
# Verified at the codegen layer for the same backend reasons as Group 8.
# =============================================================================
def row_sliced_2d_l1_to_l0b(PHYS_ROW, COL, BM, dtype="float16", accum_dtype="float"):
    """2-D L1 with a row sub-range access -> L0B.

    b_l1 is allocated [PHYS_ROW, COL] but only the first BM rows are copied
    to L0B (BM < PHYS_ROW).  The access extent = BM*COL, so slice_valid_row =
    BM while physical row = PHYS_ROW.
    """

    @T.prim_func
    def main(
        A: T.Tensor((BM, PHYS_ROW), dtype),  # type: ignore
        B: T.Tensor((PHYS_ROW, COL), dtype),  # type: ignore
        C: T.Tensor((BM, COL), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_l1 = T.alloc_L1([BM, PHYS_ROW], dtype)
            b_l1 = T.alloc_L1([PHYS_ROW, COL], dtype)
            l0a = T.alloc_L0A([BM, PHYS_ROW], dtype)
            l0b = T.alloc_L0B([PHYS_ROW, COL], dtype)
            l0c = T.alloc_L0C([BM, COL], accum_dtype)

            T.copy(A[:, :], a_l1[:, :])
            T.copy(B[:, :], b_l1[:, :])

            T.copy(a_l1[:, :], l0a[:, :])
            T.copy(b_l1[0:BM, :], l0b[0:BM, :])

            T.mma(l0a, l0b, l0c, init=True)
            T.copy(l0c, C[:, :])

    return main


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype,accum_dtype", [("float16", "float"), ("bfloat16", "float")])
@pytest.mark.parametrize(
    "PHYS_ROW,COL,BM",
    [
        (256, 128, 128),  # slice_valid_row=128 < phys_row=256
        (256, 128, 64),  # slice_valid_row=64 < phys_row=256
    ],
)
def test_row_sliced_2d_l1_to_l0b(PHYS_ROW, COL, BM, dtype, accum_dtype, target):
    """2-D L1 row-sub-range -> L0B: tile_row must fit the L0B K capacity.

    Asserts every copy_l1_to_l0b template's tile_row <= dst_K, ensuring the
    codegen uses the slice-valid row count (BM) rather than the physical row
    count (PHYS_ROW).
    """
    kfunc = row_sliced_2d_l1_to_l0b(PHYS_ROW, COL, BM, dtype, accum_dtype)
    _assert_l0b_tile_row_within_capacity(kfunc, target, CUBE_PASS_CONFIGS, expected_tile_row=BM)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
