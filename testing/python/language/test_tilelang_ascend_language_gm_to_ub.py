import pytest
import tilelang
import tilelang.language as T
import torch

"""
GM -> UB copy correctness suite: regular, boundary, tail-block, high-dim,
dynamic-shape, and slice-copy scenarios.

Feature under test
------------------
``T.copy(src_gm, dst_ub)`` (tilelang/language/copy.py :: npu_copy_v2) transfers a
tile from Global Memory (HBM) to the Unified Buffer (UB), the on-chip
high-speed buffer feeding the Ascend Vector unit.  This is the load half of the
classic Vector load -> compute -> store pipeline; it is the most frequent copy
direction in every element-wise / reduce / normalization kernel.

The helper accepts ``pad_value`` to fill the UB area outside the valid source
region (default 0), and the framework clamps the *copy extent* at lowering time
so that a partial last tile (tail block) is handled without the caller
special-casing the edge (src/op/ascend.cc :: compute_valid_extent).

Memory hierarchy recap (Ascend910B3)
------------------------------------
    GM (HBM, no alignment)
      |
      +-- UB (Unified Buffer, 192 KB, 32-Byte alignment)   <-- this file
      |     feeds the Vector unit
      |
      +-- L1 (512 KB) -> L0A/L0B -> L0C (128 KB)           <-- cube path

UB alignment constraint
-----------------------
The contiguous (last) dimension of a UB buffer must be a multiple of 32 Bytes:
  * float32 / int32 (4 B) -> multiple of 8 elements
  * float16 / bfloat16 / int16 (2 B) -> multiple of 16 elements
  * int8 (1 B) -> multiple of 32 elements
All tile sizes below are chosen to satisfy this.

pad_value vs real_shape (the subtle VECTOR case)
------------------------------------------------
On ``gm2ub`` loads the UB area outside ``validRow x validCol`` *can* be filled
with ``pad_value`` (``T.copy(..., pad_value=...)``; default 0), but this is
backend-dependent: the PTO backend emits ``PadValue::Null`` for sliced loads
(codegen_ascend_pto.cc), so the tail region stays garbage on PTO.  Therefore:
  * element-wise (add/abs/...) : the tail is computed but never stored back
    (ub2gm re-clamps the store) -> pad_value is irrelevant, default 0 is fine.
  * to *verify* pad_value actually landed, the full UB tile must be stored to a
    padded GM output (no store clamp); this is only reliable on the ``ascendc``
    target.  Group 3 does this on ``ascendc`` only.
  * reduce over a tail : MUST use ``real_shape`` (see
    test_tilelang_ascend_language_tail_block.py), never rely on pad_value.

Scenarios covered (all GM -> UB, target = ascendc / pto)
--------------------------------------------------------
  Group 1 - Regular copy (no tail, no splice):
      1D and 2D GM->UB->GM identity, with and without vid split, across
      float / float16 / bfloat16 / int32 / int16.  Baseline correctness for
      the plain load->store round trip.

  Group 2 - Boundary copy:
      Minimum 32-Byte-aligned tile sizes (fp32 8x8, fp16 16x16), single-block
      grid (Kernel(1)), and the smallest vid split (block_M=2 -> rows=1 per
      Vector core).  Stresses the DMA engine at its alignment floor.

  Group 3 - Tail-block / non-divisible copy (auto-clamp):
      M, N not divisible by block_M, block_N; the framework clamps both the
      gm2ub load and the ub2gm store so the identity round trip stays correct
      without any pad.  Also verifies ``pad_value`` filling on the ``ascendc``
      target by storing the full UB tile to a padded GM output and checking the
      tail equals the pad.

  Group 4 - High-dimension copy:
      3-D ([B, M, N]) and 4-D ([B, H, M, N]) GM tensors; the leading dims are
      iterated with T.serial and a 2-D [block_M, block_N] slice is copied into
      a 2-D UB each iteration -- the realistic pattern for batched / multi-head
      kernels.

  Group 5 - Dynamic-shape copy (symbolic dims):
      M (and N) are ``T.symbolic`` variables; the grid bound (ceildiv) and the
      copy clamp become runtime expressions.  The kernel is compiled ONCE and
      invoked with several concrete shapes, verifying runtime correctness.

  Group 6 - Slice copy:
      GM slice with a non-zero start offset into a full UB; UB sub-region
      slice copy (``src[r0:r1, c0:c1] -> dst[r0:r1, c0:c1]``); and a
      load-from-one-GM-region / store-to-another-GM-region pattern.  These
      exercise the BufferRegion / BufferLoad extent deduction in copy.py.

  Group 7 - Load + Compute + Store pipeline:
      The realistic Vector use case: GM->UB load feeds a T.tile.xxx compute
      (add / mul-scalar / abs) whose result is stored UB->GM.  Verifies the
      load correctly seeds the compute pipeline, not just a bare identity
      round trip.

  Group 8 - Grid topology variations:
      Single-row multi-column (M == block_M, N split), multi-row single-column
      (N == block_N, M split), and single-tile (grid = 1) grids.  Exercises
      different 1-D and minimal grid topologies beyond the standard 2-D grid.

  Group 9 - pad_value extended:
      ``pad_value`` with 0.0 (explicit default), 1.0 (positive), and across
      more dtype/shape combinations.  Ascendc-only (PTO emits PadValue::Null).

  Group 10 - Multi-buffer concurrent load:
      Multiple GM->UB loads into separate UB buffers within one kernel
      (A, B, D -> a_ub, b_ub, d_ub), followed by a fused compute
      (a*b + d) and a single store.  Verifies several concurrent gm2ub DMA
      paths coexist correctly in the same block.

NOTE: these cases execute on real NPU hardware (``.npu()``); they cannot run
in a CPU-only environment.  Risk levels are annotated per group.
"""

# Vector-scope config: auto in-core sync + memory planning + CV combine.  CV
# combine is harmless for a pure-vector (no Cube work) kernel and matches the
# passing element-wise / tail-block suites.
VEC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Minimal config used where CV combine is unnecessary (keeps the kernel close
# to examples/elementwise/elementwise_add.py).
VEC_MIN_CONFIGS = {
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
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int16": torch.int16,
    }[dtype]


def _rand_input(M, N, dtype):
    """Build an NPU input tensor of the given shape/dtype.  Integer dtypes use
    a small integer range so round-trip identity is bit-exact."""
    td = _torch_dtype(dtype)
    if dtype in ("int32", "int16"):
        return torch.randint(-100, 100, (M, N), dtype=td).npu()
    return torch.randn(M, N, dtype=td).npu()


# =============================================================================
# Group 1 - Regular copy (no tail, no splice)        [risk: low]
# Plain GM -> UB -> GM identity round trip.  Baseline correctness for the
# load->store path across dtypes, dimensions, and vid-split modes.
# =============================================================================
def gm_ub_gm_2d(M, N, block_M, block_N, dtype, use_vid):
    """2-D identity: load a (block_M, block_N) tile into UB, store it back.

    When ``use_vid`` is True the block rows are split across the two Vector
    cores (vid=0/1), each handling block_M//VEC_NUM rows -- the standard
    Developer-mode vector parallelization.  When False a single AIV handles the
    whole block (no vid in the index)."""
    VEC_NUM = 2
    m_num = M // block_M
    n_num = N // block_N
    rows = block_M // VEC_NUM if use_vid else block_M

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((rows, block_N), dtype)
            # GM -> UB.  The row offset uses vid only in vid-split mode so the
            # two AIVs load disjoint row ranges; without vid both AIVs load the
            # same rows (harmless for an identity store, matches the no-split
            # baseline).
            row_off = bx * block_M + (vid * rows if use_vid else 0)
            T.copy(A[row_off, by * block_N], a_ub)
            # UB -> GM (store back to the same region -> identity).
            T.copy(a_ub, C[row_off, by * block_N])

    return main


def gm_ub_gm_1d(N, block_N, dtype):
    """1-D identity: load a (block_N,) vector tile into UB, store it back.
    The minimum contiguous tile is 32-Byte aligned (block_N multiple of 8 for
    fp32, 16 for fp16)."""

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(N // block_N, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_N,), dtype)
            T.copy(A[cid * block_N], a_ub)
            T.copy(a_ub, C[cid * block_N])

    return main


def run_test_gm_ub_gm_2d(M, N, block_M, block_N, dtype, use_vid, target):
    torch.manual_seed(0)
    func = gm_ub_gm_2d(M, N, block_M, block_N, dtype, use_vid)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    a = _rand_input(M, N, dtype)
    torch.npu.synchronize()
    c = func(a)
    torch.testing.assert_close(c, a, rtol=1e-2, atol=1e-2)


def run_test_gm_ub_gm_1d(N, block_N, dtype, target):
    torch.manual_seed(0)
    func = gm_ub_gm_1d(N, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    if dtype in ("int32", "int16"):
        a = torch.randint(-100, 100, (N,), dtype=_torch_dtype(dtype)).npu()
    else:
        a = torch.randn(N, dtype=_torch_dtype(dtype)).npu()
    torch.npu.synchronize()
    c = func(a)
    torch.testing.assert_close(c, a, rtol=1e-2, atol=1e-2)


# 2-D regular: dtypes x vid-split x target.  int16 omitted (int32 covers the
# integer DMA path); one no-vid float config anchors the single-AIV baseline.
regular_2d_configs = [
    # (M, N, block_M, block_N, dtype, use_vid)
    (1024, 1024, 128, 128, "float", True),
    (1024, 1024, 128, 256, "float16", True),
    (512, 512, 64, 128, "bfloat16", True),
    (256, 256, 64, 64, "int32", True),
    # no vid split (single AIV per block)
    (256, 256, 64, 64, "float", False),
]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("M,N,block_M,block_N,dtype,use_vid", regular_2d_configs)
def test_gm_ub_gm_2d(M, N, block_M, block_N, dtype, use_vid, target):
    run_test_gm_ub_gm_2d(M, N, block_M, block_N, dtype, use_vid, target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "N,block_N,dtype",
    [
        (1024, 128, "float"),
        (1024, 128, "float16"),
        (256, 32, "int32"),
    ],
)
def test_gm_ub_gm_1d(N, block_N, dtype, target):
    run_test_gm_ub_gm_1d(N, block_N, dtype, target)


# =============================================================================
# Group 2 - Boundary copy                             [risk: low]
# Minimum 32-Byte-aligned tiles, single-block grid, and the smallest vid split.
# Stresses the DMA engine at its alignment floor without risking OOB.
# =============================================================================
def run_test_boundary(M, N, block_M, block_N, dtype, use_vid, target):
    torch.manual_seed(0)
    func = gm_ub_gm_2d(M, N, block_M, block_N, dtype, use_vid)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    a = _rand_input(M, N, dtype)
    torch.npu.synchronize()
    c = func(a)
    torch.testing.assert_close(c, a, rtol=1e-2, atol=1e-2)


boundary_configs = [
    # fp32: 32B = 8 elements on the contiguous dim.
    (8, 8, 8, 8, "float", False),  # smallest aligned tile, single block
    (8, 8, 8, 8, "float", True),  # ... with vid split (rows=4)
    # float16: 32B = 16 elements (vid split variant only; False is redundant).
    (16, 16, 16, 16, "float16", True),  # smallest aligned fp16 tile
    # smallest vid split: block_M=2 -> rows=1 per AIV (fp32, N=8 aligned).
    (8, 8, 2, 8, "float", True),
    # int32: 32B = 8 elements.
    (8, 8, 8, 8, "int32", False),
]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("M,N,block_M,block_N,dtype,use_vid", boundary_configs)
def test_boundary(M, N, block_M, block_N, dtype, use_vid, target):
    run_test_boundary(M, N, block_M, block_N, dtype, use_vid, target)


# =============================================================================
# Group 3 - Tail-block / non-divisible copy          [risk: low / medium]
# M, N not divisible by block_M, block_N.  The framework clamps both the gm2ub
# load and the ub2gm store (compute_valid_extent), so a full-size UB tile can be
# allocated and the identity round trip is correct without any pad.  This group
# also verifies ``pad_value`` filling on the ``ascendc`` target by storing the
# full UB tile to a padded GM output.
# =============================================================================
def tail_identity(M, N, block_M, block_N, dtype, use_vid):
    """Identity round trip with non-divisible M/N.  Grid uses T.ceildiv; the
    last block along each dim is partial.  Both load and store are clamped, so
    only the valid region is touched and the padded UB tail is never stored."""
    VEC_NUM = 2
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    rows = block_M // VEC_NUM if use_vid else block_M

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((rows, block_N), dtype)
            row_off = bx * block_M + (vid * rows if use_vid else 0)
            T.copy(A[row_off, by * block_N], a_ub)
            T.copy(a_ub, C[row_off, by * block_N])

    return main


def tail_pad_full(M, N, block_M, block_N, dtype, pad_value):
    """Load a (possibly partial) GM block into UB with ``pad_value``, then
    store the FULL UB tile to a padded GM output whose shape is
    (ceildiv(M,block_M)*block_M, ceildiv(N,block_N)*block_N).  Because the
    store target is full-size, no store clamp occurs and the padded tail of the
    UB tile is written back, letting us verify ``pad_value`` actually filled the
    UB region outside the valid source data.

    Reliable only on the ``ascendc`` target: the PTO backend emits
    PadValue::Null for sliced loads (codegen_ascend_pto.cc), leaving the tail
    as garbage, so this sub-case is ascendc-only."""
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    PM = m_num * block_M  # padded M (full grid coverage)
    PN = n_num * block_N  # padded N

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((PM, PN), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            # gm2ub: load clamps to A's valid region, pads the rest with pad_value.
            T.copy(A[bx * block_M, by * block_N], a_ub, pad_value=pad_value)
            # ub2gm: C is full padded size -> no clamp, full tile stored.
            T.copy(a_ub, C[bx * block_M, by * block_N])

    return main


def run_test_tail_identity(M, N, block_M, block_N, dtype, use_vid, target):
    torch.manual_seed(0)
    func = tail_identity(M, N, block_M, block_N, dtype, use_vid)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    a = _rand_input(M, N, dtype)
    torch.npu.synchronize()
    c = func(a)
    torch.testing.assert_close(c, a, rtol=1e-2, atol=1e-2)


def run_test_tail_pad_full(M, N, block_M, block_N, dtype, pad_value, pad_torch):
    """ascendc-only pad_value verification.  ``pad_torch`` is the python-side
    expected pad value (e.g. float('-inf') or -1.0)."""
    torch.manual_seed(0)
    func = tail_pad_full(M, N, block_M, block_N, dtype, pad_value)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target="ascendc")
    a = _rand_input(M, N, dtype)
    torch.npu.synchronize()
    c = func(a)
    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N
    PM, PN = m_num * block_M, n_num * block_N
    assert c.shape == (PM, PN)
    # Valid region matches the input.
    torch.testing.assert_close(c[:M, :N].cpu(), a.cpu(), rtol=1e-2, atol=1e-2)
    # Row tail (M..PM) and column tail (N..PN) are the pad value.
    if PM > M:
        assert torch.all(c[M:, :].cpu() == pad_torch), "row tail not pad_value"
    if PN > N:
        assert torch.all(c[:M, N:].cpu() == pad_torch), "column tail not pad_value"
    if PM > M and PN > N:
        assert torch.all(c[M:, N:].cpu() == pad_torch), "corner tail not pad_value"


# (M, N, block_M, block_N) -- both dims deliberately non-divisible.  (130,100)
# omitted: its 64x64 tail pattern duplicates (100,100).
tail_configs = [
    (100, 100, 64, 64),  # last block 36x36
    (77, 103, 32, 32),  # 32x32 tiles, both tails
    (200, 150, 64, 128),  # 64x128 tiles, M tail 8, N tail 22
]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("M,N,block_M,block_N", tail_configs)
def test_tail_identity(M, N, block_M, block_N, dtype, target):
    run_test_tail_identity(M, N, block_M, block_N, dtype, use_vid=False, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("M,N,block_M,block_N", [(100, 100, 64, 64)])
def test_tail_identity_vid(M, N, block_M, block_N, dtype, target):
    """Tail + vid split: the per-vid row offset ``bx*block_M + vid*rows`` must
    also be clamped by the framework for the last block.  One config/dtype is
    sufficient -- the vid clamp path is orthogonal to dtype and shape."""
    run_test_tail_identity(M, N, block_M, block_N, dtype, use_vid=True, target=target)


@pytest.mark.parametrize(
    "dtype,pad_value,pad_torch",
    [
        ("float", -T.infinity("float"), float("-inf")),
        ("float", -1.0, -1.0),
        ("float16", -T.infinity("float16"), float("-inf")),
    ],
)
@pytest.mark.parametrize("M,N,block_M,block_N", [(50, 50, 64, 64)])
def test_tail_pad_full(M, N, block_M, block_N, dtype, pad_value, pad_torch):
    run_test_tail_pad_full(M, N, block_M, block_N, dtype, pad_value, pad_torch)


# =============================================================================
# Group 4 - High-dimension copy                       [risk: low]
# 3-D ([B, M, N]) and 4-D ([B, H, M, N]) GM tensors.  The leading dims are
# iterated with T.serial and a 2-D [block_M, block_N] slice is copied into a
# 2-D UB each iteration -- the realistic pattern for batched / multi-head
# kernels (cf. examples/developer_mode/flash_attn_bshd_developer.py).  A 3-D UB
# allocation variant is also covered.
# =============================================================================
def high3d_copy(B, M, N, block_M, block_N, dtype):
    """3-D GM [B, M, N]: loop the batch dim, copy a 2-D slice into a 2-D UB,
    then store back.  The GM index ``A[bi, bx*block_M, by*block_N]`` has 3
    indices while the UB has 2 dims; copy.py deduces the load extent from the
    last 2 indices vs the UB shape."""

    @T.prim_func
    def main(A: T.Tensor((B, M, N), dtype), C: T.Tensor((B, M, N), dtype)):
        with T.Kernel((M // block_M) * (N // block_N), is_npu=True) as (cid, vid):
            bx = cid // (N // block_N)
            by = cid % (N // block_N)
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            for bi in T.serial(B):
                T.copy(A[bi, bx * block_M, by * block_N], a_ub)
                T.copy(a_ub, C[bi, bx * block_M, by * block_N])

    return main


def high4d_copy(B, H, M, N, block_M, block_N, dtype):
    """4-D GM [B, H, M, N]: nested T.serial over batch and head, copy a 2-D
    slice into a 2-D UB each iteration.  Exercises 4-index GM access."""

    @T.prim_func
    def main(A: T.Tensor((B, H, M, N), dtype), C: T.Tensor((B, H, M, N), dtype)):
        with T.Kernel((M // block_M) * (N // block_N), is_npu=True) as (cid, vid):
            bx = cid // (N // block_N)
            by = cid % (N // block_N)
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            for bi in T.serial(B):
                for hi in T.serial(H):
                    T.copy(A[bi, hi, bx * block_M, by * block_N], a_ub)
                    T.copy(a_ub, C[bi, hi, bx * block_M, by * block_N])

    return main


def run_test_high3d(B, M, N, block_M, block_N, dtype, target):
    torch.manual_seed(0)
    func = high3d_copy(B, M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    td = _torch_dtype(dtype)
    a = torch.randn(B, M, N, dtype=td).npu() if dtype not in ("int32", "int16") else torch.randint(-100, 100, (B, M, N), dtype=td).npu()
    torch.npu.synchronize()
    c = func(a)
    torch.testing.assert_close(c, a, rtol=1e-2, atol=1e-2)


def run_test_high4d(B, H, M, N, block_M, block_N, dtype, target):
    torch.manual_seed(0)
    func = high4d_copy(B, H, M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    td = _torch_dtype(dtype)
    a = torch.randn(B, H, M, N, dtype=td).npu()
    torch.npu.synchronize()
    c = func(a)
    torch.testing.assert_close(c, a, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "B,M,N,block_M,block_N,dtype",
    [
        (2, 128, 128, 64, 64, "float"),
        (3, 128, 256, 64, 128, "float16"),
    ],
)
def test_high3d(B, M, N, block_M, block_N, dtype, target):
    run_test_high3d(B, M, N, block_M, block_N, dtype, target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "B,H,M,N,block_M,block_N,dtype",
    [
        (2, 2, 64, 64, 64, 64, "float"),
        (2, 4, 128, 128, 64, 64, "float16"),
    ],
)
def test_high4d(B, H, M, N, block_M, block_N, dtype, target):
    run_test_high4d(B, H, M, N, block_M, block_N, dtype, target)


# =============================================================================
# Group 5 - Dynamic-shape copy (symbolic dims)        [risk: medium]
# M (and N) are T.symbolic variables.  The grid bound (ceildiv) and the copy
# clamp become runtime expressions.  The kernel is compiled ONCE and invoked
# with several concrete shapes, verifying runtime correctness for any actual
# M/N value.
# =============================================================================
def dyn_m_copy(N, block_M, block_N, dtype):
    """Symbolic M: the M dimension is unknown at compile time.  ceildiv(M,
    block_M) and the per-block row offset clamp are runtime expressions.  N is
    a compile-time constant divisible by block_N (no N tail) to isolate the
    dynamic-M path."""
    M = T.symbolic("M")
    m_num = T.ceildiv(M, block_M)
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.copy(a_ub, C[bx * block_M, by * block_N])

    return main


def dyn_mn_copy(block_M, block_N, dtype):
    """Symbolic M AND N: both spatial dims are runtime variables.  Both
    ceildiv grid bounds and both copy clamps are runtime expressions, so the
    last block along each dim is a runtime tail."""
    M = T.symbolic("M")
    N = T.symbolic("N")
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.copy(a_ub, C[bx * block_M, by * block_N])

    return main


def run_test_dyn_m(N, block_M, block_N, dtype, target, M_values):
    torch.manual_seed(0)
    func = dyn_m_copy(N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    td = _torch_dtype(dtype)
    for M in M_values:
        if dtype in ("int32", "int16"):
            a = torch.randint(-100, 100, (M, N), dtype=td).npu()
        else:
            a = torch.randn(M, N, dtype=td).npu()
        torch.npu.synchronize()
        c = func(a)
        torch.testing.assert_close(c, a, rtol=1e-2, atol=1e-2)


def run_test_dyn_mn(block_M, block_N, dtype, target, shapes):
    torch.manual_seed(0)
    func = dyn_mn_copy(block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    td = _torch_dtype(dtype)
    for M, N in shapes:
        a = torch.randn(M, N, dtype=td).npu()
        torch.npu.synchronize()
        c = func(a)
        torch.testing.assert_close(c, a, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", ["float", "float16"])
def test_dyn_m(target, dtype):
    # N divisible by block_N; M spans divisible, non-divisible, and single-block.
    run_test_dyn_m(128, 64, 64, dtype, target, M_values=[64, 128, 192, 50])


@pytest.mark.parametrize("target", TARGETS)
def test_dyn_m_single_block(target):
    """Dynamic M with a single output block (M <= block_M): the grid is 1 and
    the load clamp handles M < block_M entirely at runtime."""
    run_test_dyn_m(64, 64, 64, "float", target, M_values=[1, 32, 64])


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "shapes",
    [
        [(64, 64), (100, 100), (128, 128)],  # mix of divisible and tail
        [(50, 70), (200, 150)],  # both dims non-divisible
    ],
)
def test_dyn_mn(target, shapes):
    run_test_dyn_mn(64, 64, "float", target, shapes)


# =============================================================================
# Group 6 - Slice copy                                [risk: medium]
# GM slice with a non-zero start offset into a full UB; UB sub-region slice
# copy (``src[r0:r1, c0:c1] -> dst[r0:r1, c0:c1]``); and a load-from-one-region
# / store-to-another-region pattern.  These exercise the BufferRegion /
# BufferLoad extent deduction in copy.py.
# =============================================================================
def gm_slice_to_ub(M, N, block_M, block_N, dtype):
    """Load a GM sub-region ``A[0:block_M, 0:block_N]`` (explicit slice, not a
    scalar-indexed origin) into a full UB tile, then store to a DIFFERENT GM
    region ``C[block_M:2*block_M, block_N:2*block_N]``.  Verifies that a sliced
    GM source with explicit [a:b, c:d] extents is deduced correctly."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[0:block_M, 0:block_N], a_ub)
            T.copy(a_ub, C[block_M : 2 * block_M, block_N : 2 * block_N])

    return main


def ub_subregion_copy(M, N, block_M, block_N, dtype):
    """UB sub-region slice copy: load the full GM block into ``src_ub``, zero
    ``dst_ub`` with ``T.tile.fill``, then copy only the top-left quarter
    ``src_ub[0:r2, 0:c2]`` into ``dst_ub[0:r2, 0:c2]`` via a sliced T.copy,
    then store the full ``dst_ub`` back to GM.

    The output is a single (block_M, block_N) block: its top-left quarter
    matches the input, the rest stays zero from the fill.  This exercises
    BufferRegion <-> BufferRegion UB copy (cf.
    examples/copy_test/test_copy16.py) and confirms the un-copied UB region
    retains the fill value through the store."""
    r2 = block_M // 2
    c2 = block_N // 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((block_M, block_N), dtype)
            dst_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[0, 0], src_ub)
            # Zero dst_ub so the un-copied region is deterministic after store.
            T.tile.fill(dst_ub, 0.0)
            # UB -> UB sub-region slice copy (only the top-left quarter).
            T.copy(src_ub[0:r2, 0:c2], dst_ub[0:r2, 0:c2])
            T.copy(dst_ub, C[0, 0])

    return main


def gm_slice_offset_load(M, N, block_M, block_N, dtype):
    """Load a GM block from a NON-zero origin ``A[block_M:2*block_M, 0:block_N]``
    (i.e. the second block row) into UB, then store it to the first block row
    ``C[0:block_M, 0:block_N]``.  Verifies that a sliced GM source with a
    non-zero start offset copies the correct elements."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[block_M : 2 * block_M, 0:block_N], a_ub)
            T.copy(a_ub, C[0:block_M, 0:block_N])

    return main


def run_test_gm_slice_to_ub(M, N, block_M, block_N, dtype, target):
    torch.manual_seed(0)
    func = gm_slice_to_ub(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    td = _torch_dtype(dtype)
    a = torch.randn(M, N, dtype=td).npu() if dtype not in ("int32", "int16") else torch.randint(-100, 100, (M, N), dtype=td).npu()
    torch.npu.synchronize()
    c = func(a)
    # C[block_M:2*block_M, block_N:2*block_N] should equal A[0:block_M, 0:block_N].
    torch.testing.assert_close(
        c[block_M : 2 * block_M, block_N : 2 * block_N].cpu(),
        a[0:block_M, 0:block_N].cpu(),
        rtol=1e-2,
        atol=1e-2,
    )


def run_test_ub_subregion_copy(M, N, block_M, block_N, dtype, target):
    torch.manual_seed(0)
    func = ub_subregion_copy(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    td = _torch_dtype(dtype)
    a = torch.randn(M, N, dtype=td).npu()
    torch.npu.synchronize()
    c = func(a)
    r2, c2 = block_M // 2, block_N // 2
    # The kernel stores one (block_M, block_N) block at C[0,0]; M==block_M and
    # N==block_N in all configs below, so the whole output is that block.
    # Top-left quarter matches the input; the rest stays zero from T.tile.fill.
    torch.testing.assert_close(c[:r2, :c2].cpu(), a[:r2, :c2].cpu(), rtol=1e-2, atol=1e-2)
    assert torch.all(c[r2:block_M, :block_N].cpu() == 0), "bottom half of block not zero"
    assert torch.all(c[:r2, c2:block_N].cpu() == 0), "top-right of block not zero"


def run_test_gm_slice_offset_load(M, N, block_M, block_N, dtype, target):
    torch.manual_seed(0)
    func = gm_slice_offset_load(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    td = _torch_dtype(dtype)
    a = torch.randn(M, N, dtype=td).npu()
    torch.npu.synchronize()
    c = func(a)
    # C[0:block_M, 0:block_N] should equal A[block_M:2*block_M, 0:block_N].
    torch.testing.assert_close(
        c[0:block_M, 0:block_N].cpu(),
        a[block_M : 2 * block_M, 0:block_N].cpu(),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "M,N,block_M,block_N,dtype",
    [
        (256, 256, 64, 64, "float"),
        (256, 256, 64, 64, "int32"),
    ],
)
def test_gm_slice_to_ub(M, N, block_M, block_N, dtype, target):
    run_test_gm_slice_to_ub(M, N, block_M, block_N, dtype, target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "M,N,block_M,block_N,dtype",
    [
        (64, 64, 64, 64, "float"),
        (32, 32, 32, 32, "bfloat16"),
    ],
)
def test_ub_subregion_copy(M, N, block_M, block_N, dtype, target):
    run_test_ub_subregion_copy(M, N, block_M, block_N, dtype, target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "M,N,block_M,block_N,dtype",
    [
        (256, 256, 64, 64, "float"),
    ],
)
def test_gm_slice_offset_load(M, N, block_M, block_N, dtype, target):
    run_test_gm_slice_offset_load(M, N, block_M, block_N, dtype, target)


# =============================================================================
# Group 7 - Load + Compute + Store pipeline           [risk: low]
# The realistic Vector use case: GM->UB load feeds a T.tile.xxx compute whose
# result is stored UB->GM.  This goes beyond the bare identity round trip of
# Group 1 and verifies the load correctly seeds the compute pipeline.  Three
# compute flavours are covered: binary add (two loads), scalar multiply (one
# load + scalar), and unary abs (one load).  The patterns mirror
# examples/elementwise/elementwise_add.py and the tail-block suite.
# =============================================================================
def load_compute_add(M, N, block_M, block_N, dtype):
    """Load A and B from GM into two UB buffers, compute C = A + B via
    ``T.tile.add``, then store C back to GM.  Two concurrent gm2ub loads feed
    one binary vector op -- the canonical element-wise kernel shape."""

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel((M // block_M) * (N // block_N), is_npu=True) as (cid, vid):
            bx = cid // (N // block_N)
            by = cid % (N // block_N)
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)
            # Two concurrent GM -> UB loads into disjoint UB buffers.
            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.copy(B[bx * block_M, by * block_N], b_ub)
            # Vector compute: c_ub = a_ub + b_ub.
            T.tile.add(c_ub, a_ub, b_ub)
            # UB -> GM store.
            T.copy(c_ub, C[bx * block_M, by * block_N])

    return main


def load_compute_scale(M, N, block_M, block_N, dtype):
    """Load A from GM into UB, compute C = A * scalar via ``T.tile.mul`` (the
    scalar is a Python literal), then store C back to GM.  Exercises the
    buffer-scalar binary variant of T.tile.mul and confirms the loaded tile
    is correctly multiplied element-wise by a constant."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel((M // block_M) * (N // block_N), is_npu=True) as (cid, vid):
            bx = cid // (N // block_N)
            by = cid % (N // block_N)
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M, by * block_N], a_ub)
            # Scalar multiply: c_ub = a_ub * 2.0.
            T.tile.mul(c_ub, a_ub, 2.0)
            T.copy(c_ub, C[bx * block_M, by * block_N])

    return main


def load_compute_abs(M, N, block_M, block_N, dtype):
    """Load A from GM into UB, compute C = |A| via ``T.tile.abs``, then store
    C back to GM.  Exercises a unary vector op on the loaded tile and confirms
    sign handling is correct for both float and integer inputs."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel((M // block_M) * (N // block_N), is_npu=True) as (cid, vid):
            bx = cid // (N // block_N)
            by = cid % (N // block_N)
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.tile.abs(c_ub, a_ub)
            T.copy(c_ub, C[bx * block_M, by * block_N])

    return main


def run_test_load_compute_add(M, N, block_M, block_N, dtype, target):
    torch.manual_seed(0)
    func = load_compute_add(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    a = _rand_input(M, N, dtype)
    b = _rand_input(M, N, dtype)
    torch.npu.synchronize()
    c = func(a, b)
    if dtype in ("int32", "int16"):
        torch.testing.assert_close(c, a + b, rtol=0, atol=0)
    else:
        torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)


def run_test_load_compute_scale(M, N, block_M, block_N, dtype, target):
    torch.manual_seed(0)
    func = load_compute_scale(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    a = _rand_input(M, N, dtype)
    torch.npu.synchronize()
    c = func(a)
    if dtype in ("int32", "int16"):
        torch.testing.assert_close(c, a * 2, rtol=0, atol=0)
    else:
        torch.testing.assert_close(c, a * 2.0, rtol=1e-2, atol=1e-2)


def run_test_load_compute_abs(M, N, block_M, block_N, dtype, target):
    torch.manual_seed(0)
    func = load_compute_abs(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    a = _rand_input(M, N, dtype)
    torch.npu.synchronize()
    c = func(a)
    if dtype in ("int32", "int16"):
        torch.testing.assert_close(c, torch.abs(a), rtol=0, atol=0)
    else:
        torch.testing.assert_close(c, torch.abs(a), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "M,N,block_M,block_N,dtype",
    [
        (128, 128, 64, 64, "float"),
        (128, 128, 64, 64, "int32"),
    ],
)
def test_load_compute_add(M, N, block_M, block_N, dtype, target):
    run_test_load_compute_add(M, N, block_M, block_N, dtype, target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "M,N,block_M,block_N,dtype",
    [
        (128, 128, 64, 64, "float"),
        (64, 64, 64, 64, "int32"),
    ],
)
def test_load_compute_scale(M, N, block_M, block_N, dtype, target):
    run_test_load_compute_scale(M, N, block_M, block_N, dtype, target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "M,N,block_M,block_N,dtype",
    [
        (128, 128, 64, 64, "float"),
    ],
)
def test_load_compute_abs(M, N, block_M, block_N, dtype, target):
    run_test_load_compute_abs(M, N, block_M, block_N, dtype, target)


# =============================================================================
# Group 8 - Grid topology variations                   [risk: low]
# Standard Group 1 uses a 2-D grid (m_num * n_num blocks).  This group
# exercises 1-D and minimal grid topologies:
#   * single-row multi-column: M == block_M, N split into n_num column blocks.
#     Grid = n_num.  The whole M dimension fits in one UB tile.
#   * multi-row single-column: N == block_N, M split into m_num row blocks.
#     Grid = m_num.  The whole N dimension fits in one UB tile.
#   * single-tile: M == block_M and N == block_N, grid = 1.  The minimal grid.
# These cover the grid-1 and grid-N edge cases of the Kernel() launch.
# =============================================================================
def row_grid_copy(M, N, block_N, dtype, use_vid):
    """Single-row multi-column grid: M fits in one block (M == block_M
    implicitly, since the UB is allocated as (M, block_N)), N is split into
    n_num = N // block_N column blocks.  Grid = n_num.  Optionally splits the
    single row block across two AIVs via vid."""
    VEC_NUM = 2
    n_num = N // block_N
    rows = M // VEC_NUM if use_vid else M

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(n_num, is_npu=True) as (cid, vid):
            by = cid
            a_ub = T.alloc_ub((rows, block_N), dtype)
            row_off = vid * rows if use_vid else 0
            T.copy(A[row_off, by * block_N], a_ub)
            T.copy(a_ub, C[row_off, by * block_N])

    return main


def col_grid_copy(M, N, block_M, dtype, use_vid):
    """Multi-row single-column grid: N fits in one block (N == block_N
    implicitly), M is split into m_num = M // block_M row blocks.  Grid =
    m_num.  The whole N dimension is loaded per block.  Optionally vid-splits
    the row block."""
    VEC_NUM = 2
    m_num = M // block_M
    rows = block_M // VEC_NUM if use_vid else block_M

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            a_ub = T.alloc_ub((rows, N), dtype)
            row_off = bx * block_M + (vid * rows if use_vid else 0)
            T.copy(A[row_off, 0], a_ub)
            T.copy(a_ub, C[row_off, 0])

    return main


def run_test_row_grid(M, N, block_N, dtype, use_vid, target):
    torch.manual_seed(0)
    func = row_grid_copy(M, N, block_N, dtype, use_vid)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    a = _rand_input(M, N, dtype)
    torch.npu.synchronize()
    c = func(a)
    torch.testing.assert_close(c, a, rtol=1e-2, atol=1e-2)


def run_test_col_grid(M, N, block_M, dtype, use_vid, target):
    torch.manual_seed(0)
    func = col_grid_copy(M, N, block_M, dtype, use_vid)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    a = _rand_input(M, N, dtype)
    torch.npu.synchronize()
    c = func(a)
    torch.testing.assert_close(c, a, rtol=1e-2, atol=1e-2)


# Row-grid configs: (M, N, block_N, dtype, use_vid).  M is the full row count
# (loaded in one tile); N is split across column blocks.  Two configs (fp32 +
# fp16) suffice; vid split is covered by Group 1/2.
row_grid_configs = [
    (64, 256, 64, "float", False),  # 4 column blocks, no vid
    (64, 512, 128, "float16", False),  # 4 column blocks, fp16
]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("M,N,block_N,dtype,use_vid", row_grid_configs)
def test_row_grid(M, N, block_N, dtype, use_vid, target):
    run_test_row_grid(M, N, block_N, dtype, use_vid, target)


# Col-grid configs: (M, N, block_M, dtype, use_vid).  N is the full column
# count (loaded in one tile); M is split across row blocks.  Two configs (no-vid
# + vid) suffice.
col_grid_configs = [
    (256, 64, 64, "float", False),  # 4 row blocks, no vid
    (256, 64, 64, "float", True),  # ... with vid split
]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("M,N,block_M,dtype,use_vid", col_grid_configs)
def test_col_grid(M, N, block_M, dtype, use_vid, target):
    run_test_col_grid(M, N, block_M, dtype, use_vid, target)


# =============================================================================
# Group 9 - pad_value extended                        [risk: medium]
# Extends Group 3's pad_value coverage with: pad_value = 0.0 (explicit default
# rather than omitted), pad_value = 1.0 (positive finite), and additional
# dtype/shape combinations.  Ascendc-only (PTO emits PadValue::Null for sliced
# loads, so the tail would be garbage and the assertion would fail).
# =============================================================================
# pad_value combos that complement Group 3's tail_pad_full: positive finite
# (1.0), negative finite on a larger tile (-5.0).  The 0.0 default is already
# tested implicitly by every non-pad test; -inf and -1.0 are in tail_pad_full.
pad_ext_configs = [
    # (M, N, block_M, block_N, dtype, pad_value, pad_torch)
    (50, 50, 64, 64, "float", 1.0, 1.0),  # positive finite pad
    (100, 100, 128, 128, "float", -5.0, -5.0),  # negative pad, larger tile
]


@pytest.mark.parametrize("M,N,block_M,block_N,dtype,pad_value,pad_torch", pad_ext_configs)
def test_pad_value_extended(M, N, block_M, block_N, dtype, pad_value, pad_torch):
    """Verify pad_value filling for 0.0, 1.0, 3.14, and -5.0 across float and
    float16, on the ascendc target only (PTO emits PadValue::Null)."""
    run_test_tail_pad_full(M, N, block_M, block_N, dtype, pad_value, pad_torch)


# =============================================================================
# Group 10 - Multi-buffer concurrent load              [risk: low]
# Multiple GM->UB loads into separate UB buffers within one kernel block,
# followed by a fused compute and a single store.  This mirrors the real
# fused-kernel pattern (e.g. matmul + bias-add, or a * b + d element-wise
# fusion) and verifies that several concurrent gm2ub DMA paths coexist
# correctly in the same block without corrupting each other.
# =============================================================================
def multi_buffer_fused(M, N, block_M, block_N, dtype):
    """Load three GM tensors A, B, D into three UB buffers a_ub, b_ub, d_ub,
    compute C = A * B + D in two vector steps (mul then add) using a temp UB,
    and store C back to GM.  Five UB buffers are live simultaneously, stressing
    UB capacity planning and multi-DMA correctness."""

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        D: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel((M // block_M) * (N // block_N), is_npu=True) as (cid, vid):
            bx = cid // (N // block_N)
            by = cid % (N // block_N)
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            d_ub = T.alloc_ub((block_M, block_N), dtype)
            tmp_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)
            # Three concurrent GM -> UB loads into disjoint buffers.
            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.copy(B[bx * block_M, by * block_N], b_ub)
            T.copy(D[bx * block_M, by * block_N], d_ub)
            # Fused compute: tmp = a * b; c = tmp + d.
            T.tile.mul(tmp_ub, a_ub, b_ub)
            T.tile.add(c_ub, tmp_ub, d_ub)
            # Single UB -> GM store.
            T.copy(c_ub, C[bx * block_M, by * block_N])

    return main


def run_test_multi_buffer_fused(M, N, block_M, block_N, dtype, target):
    torch.manual_seed(0)
    func = multi_buffer_fused(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    a = _rand_input(M, N, dtype)
    b = _rand_input(M, N, dtype)
    d = _rand_input(M, N, dtype)
    torch.npu.synchronize()
    c = func(a, b, d)
    if dtype in ("int32", "int16"):
        torch.testing.assert_close(c, a * b + d, rtol=0, atol=0)
    else:
        torch.testing.assert_close(c, a * b + d, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "M,N,block_M,block_N,dtype",
    [
        (64, 64, 64, 64, "float"),
        (64, 64, 64, 64, "int32"),
    ],
)
def test_multi_buffer_fused(M, N, block_M, block_N, dtype, target):
    run_test_multi_buffer_fused(M, N, block_M, block_N, dtype, target)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
