"""Regression tests for unaligned K-tails in Ascend ``gemm_v0`` (issue #1341).

``gemm_v0`` splits K into ``kL0Size``-wide L0 tiles. When the last K-tile is
not 16-aligned (``kL0Tail % 16 != 0``) three separate defects conspired to
corrupt ~10% of the output (fp16, rtol=atol=2e-2):

1. **L1 buffer under-sizing.** The zN L1 layout pads K up to a multiple of 16
   and interleaves the padding holes *inside* the tile (each 16-column N band
   occupies ``roundUp16(K) * 16`` elements). The buffer size was derived from
   the layout's max *valid* offset + 1 (e.g. 26496 for zN(200, 128) instead of
   the padded extent 26624), and the 2D re-flattening truncated the outer dim
   (13312 // 200 = 66). The next L1 allocation therefore overlapped the tail of
   the previous buffer: B's GM->L1 copy physically overwrote A's K-padding.

2. **Lost layout annotations.** ``AscendLowerParallelToVector`` rebuilt Blocks
   without their annotations, dropping the default zN ``layout_map`` injected
   by ``AscendInferBufferScope``. Layout inference then fell back to a
   non-fractal layout and sized L1 buffers by the *logical* shape (128 * 200 =
   25600 elements), re-introducing the overlap for any kernel with a
   ``T.Parallel`` epilogue (the classic blocked-GEMM writer loop). This
   upstream annotation-drop is NOT fixed in this PR (preserving it regressed
   the vid-reduce workspace transform, which relies on linear offsets);
   instead the sizing is corrected downstream in ``Flatten2DBuffer``
   (``ascend_collect_buffer_shape.cc``), which pads every ``shared.l1``
   buffer's element count up to the zN fractal extent once tile ops are
   lowered and offsets are frozen.

3. **Un-zeroed fractal padding.** ``copy_gm_to_l1`` only zero-initialized the
   L1 tile for partial tail copies, and even then with the logical element
   count. Full-tile copies with K % 16 != 0 left the zN padding holes holding
   stale L1 garbage. ``Mmad`` consumes whole C0 fractals (a tail mma with
   k = 72 accumulates 80 K-slots), so the garbage was multiplied into C.
   The zero-init also failed for pipeline multi-versioned L1 buffers: the
   codegen's ``need_clear`` was ``(dst_offset == 0)``, so every non-zero
   pipeline version skipped the clear; it is now granted to any copy whose
   destination offset is provably a whole multiple of the tile element count
   (a version base), while in-tile splice sub-region offsets keep skipping it.

The fixes: size zN/nZ buffers by the fractal-padded extent
(``lower_tile_op.cc``), ceil-div the 2D re-flattening and pad ``shared.l1``
totals to the zN extent (``ascend_collect_buffer_shape.cc``), zero-init the
full fractal extent whenever padding holes exist plus version-base-aware
``need_clear`` (``copy_gm_to_l1`` in ``src/tl_templates/ascend/common.h`` and
its codegen in ``src/target/codegen_ascend.cc``), and compile-time-reject
unservable int8 K-tails and non-C0-aligned ``kL0Size`` when ``kL0split > 1``.

NOTE: executes on real NPU hardware (``.npu()``); ascendc target only. The PTO
backend rejects non-fractal-divisible L1 tiles at compile time (pre-existing
``static_assert`` in pto_tile.hpp), which is a separate limitation.
"""

import pytest
import tilelang
import tilelang.language as T
import torch

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

TARGET = "ascendc"

# (M, N, K) — the issue's reproduce shape plus further unaligned K-tails and
# aligned sanity anchors. kL0Size defaults to 128, so K=200 gives kL0split=2
# with kL0Tail=72 (72 % 16 = 8), K=136 gives kL0Tail=8, K=1000 gives
# kL0split=8 with kL0Tail=104.
SINGLE_TILE_CONFIGS = [
    (64, 128, 128),  # single K-tile (aligned anchor)
    (64, 128, 256),  # two aligned K-tiles (anchor)
    (64, 128, 200),  # issue #1341 reproduce: kL0Tail=72
    (64, 128, 136),  # kL0Tail=8
    (64, 128, 264),  # kL0split=3, kL0Tail=8
    (64, 128, 1000),  # kL0split=8, kL0Tail=104
]


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


def _compile(program):
    return tilelang.compile(program, pass_configs=PASS_CONFIGS, target=TARGET)


def _single_l1_tile_gemm(M, N, K, dtype="float16", kL0Size=128):
    """The issue's kernel: one Expert-mode L1 tile feeding a single gemm_v0,
    so K is split inside gemm_v0 by kL0Size."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((M, K), dtype)
            B_L1 = T.alloc_L1((K, N), dtype)
            C_L0 = T.alloc_L0C((M, N), "float")
            with T.Scope("C"):
                T.copy(A[0, 0], A_L1)
                T.copy(B[0, 0], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=True, kL0Size=kL0Size)
                T.copy(C_L0, C[0, 0])

    return main


def _blocked_gemm(M, N, K, block_M, block_N, K_L1, dtype="float16", kL0Size=128, transpose_A=False, transpose_B=False):
    """Developer-style blocked GEMM. Besides the gemm_v0-internal K split this
    exercises the K_L1 tail copy (K_L1 % 16 != 0) and, through its T.Parallel
    epilogue, the Block-annotation preservation in AscendLowerParallelToVector
    (defect 2: the default zN layout_map used to be dropped there, sizing L1
    buffers by the logical shape and overlapping neighbouring tiles)."""
    m_num = M // block_M
    n_num = N // block_N

    a_gm_shape = (K, M) if transpose_A else (M, K)
    b_gm_shape = (N, K) if transpose_B else (K, N)
    a_l1_shape = (K_L1, block_M) if transpose_A else (block_M, K_L1)
    b_l1_shape = (block_N, K_L1) if transpose_B else (K_L1, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(a_gm_shape, dtype),
        B: T.Tensor(b_gm_shape, dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1(a_l1_shape, dtype)
            B_L1 = T.alloc_L1(b_l1_shape, dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), "float")

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    if transpose_A:
                        T.copy(A[k * K_L1, bx * block_M], A_L1)
                    else:
                        T.copy(A[bx * block_M, k * K_L1], A_L1)
                    if transpose_B:
                        T.copy(B[by * block_N, k * K_L1], B_L1)
                    else:
                        T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(
                        A_L1,
                        B_L1,
                        C_L0,
                        transpose_A=transpose_A,
                        transpose_B=transpose_B,
                        init=(k == 0),
                        kL0Size=kL0Size,
                    )
                    T.barrier_all()

                for i, j in T.Parallel(block_M, block_N):
                    C[bx * block_M + i, by * block_N + j] = C_L0[i, j]

    return main


def _run(kernel_fn, ref_fn, shapes):
    torch.manual_seed(0)
    kernel = _compile(kernel_fn())
    tensors = [torch.randn(*s, dtype=torch.float16, device="npu") for s in shapes[:-1]]
    tensors.append(torch.zeros(*shapes[-1], dtype=torch.float16, device="npu"))
    torch.npu.synchronize()
    kernel(*tensors)
    torch.npu.synchronize()
    ref = ref_fn(*[t.float() for t in tensors[:-1]])
    torch.testing.assert_close(tensors[-1].float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("M,N,K", SINGLE_TILE_CONFIGS)
def test_gemm_v0_ktail_single_l1_tile(M, N, K):
    """Issue #1341: unaligned kL0Tail with kL0split >= 2 on a single L1 tile."""
    _run(
        lambda: _single_l1_tile_gemm(M, N, K),
        lambda a, b: a @ b,
        [(M, K), (K, N), (M, N)],
    )


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("K,K_L1", [(200, 200), (456, 200), (200, 128), (264, 128)])
def test_gemm_v0_ktail_blocked(K, K_L1):
    """Unaligned K with a T.Parallel epilogue (lost-annotation defect) and an
    unaligned K_L1 tail copy (zero-init defect), plus multi-iteration K loops."""
    M = N = 256
    block_M = block_N = 128
    _run(
        lambda: _blocked_gemm(M, N, K, block_M, block_N, K_L1),
        lambda a, b: a @ b,
        [(M, K), (K, N), (M, N)],
    )


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("transpose_A,transpose_B", [(True, False), (False, True), (True, True)])
def test_gemm_v0_ktail_transpose(transpose_A, transpose_B):
    """Unaligned K through the transpose-A / transpose-B (QK-style) paths."""
    M, N, K = 128, 256, 200
    block_M, block_N, K_L1 = 128, 128, 200
    _run(
        lambda: _blocked_gemm(
            M,
            N,
            K,
            block_M,
            block_N,
            K_L1,
            transpose_A=transpose_A,
            transpose_B=transpose_B,
        ),
        lambda a, b: (a.t() if transpose_A else a) @ (b.t() if transpose_B else b),
        [
            (K, M) if transpose_A else (M, K),
            (N, K) if transpose_B else (K, N),
            (M, N),
        ],
    )


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("K", [200, 264, 1000])
def test_gemm_v0_ktail_kl0size_64(K):
    """Same unaligned tails with a finer kL0Size=64 split (kL0split doubles)."""
    M, N = 64, 128
    _run(
        lambda: _single_l1_tile_gemm(M, N, K, kL0Size=64),
        lambda a, b: a @ b,
        [(M, K), (K, N), (M, N)],
    )


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_gemm_v0_ktail_dtypes(dtype):
    """bf16 shares the fp16 fractal geometry (ELE_NUM_PER_C0 == 16)."""
    M, N, K = 64, 128, 200
    torch.manual_seed(0)
    kernel = _compile(_single_l1_tile_gemm(M, N, K, dtype=dtype))
    td = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    a = torch.randn(M, K, dtype=td, device="npu")
    b = torch.randn(K, N, dtype=td, device="npu")
    c = torch.zeros(M, N, dtype=td, device="npu")
    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()
    ref = a.float() @ b.float()
    torch.testing.assert_close(c.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize(
    "K,expected",
    [
        (512, "pass"),  # aligned tail (existing #1395 coverage shape)
        (216, "pass"),  # kL0Tail=88: 88 % 32 = 24 > 16, fractal padding covers it
        (152, "pass"),  # kL0Tail=24: 24 % 32 = 24 > 16
        (200, "reject"),  # kL0Tail=72: 72 % 32 = 8 <= 16 -> L0B C0 gap, rejected
        (264, "reject"),  # kL0Tail=8
    ],
)
def test_gemm_v0_ktail_int8(K, expected):
    """int8 (ELE_NUM_PER_C0 == 32): tails whose C0 rounding exceeds the 16-row
    fractal padding (kL0Tail % 32 in [1, 16]) can not be served from the L1
    layout and are rejected at compile time instead of silently miscomputing;
    the other unaligned tails must stay exact."""
    M = N = 256
    block_M = block_N = 128
    m_num = M // block_M
    n_num = N // block_M

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "int8"),
        B: T.Tensor((K, N), "int8"),
        C: T.Tensor((M, N), "int32"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            A_L1 = T.alloc_L1((block_M, K), "int8")
            B_L1 = T.alloc_L1((K, block_N), "int8")
            C_L0 = T.alloc_L0C((block_M, block_N), "int32")
            with T.Scope("C"):
                T.copy(A[bx * block_M, 0], A_L1)
                T.copy(B[0, by * block_N], B_L1)
                T.barrier_all()
                T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                T.barrier_all()
                # T.copy (not T.Parallel) for the int32 L0C -> GM write: the
                # Parallel path hits an unrelated codegen InternalError on
                # int32 GM stores (see test_tilelang_ascend_language_gemm_v0).
                T.copy(C_L0, C[bx * block_M, by * block_N])

    if expected == "reject":
        with pytest.raises(RuntimeError, match="Compilation Failed"):
            _compile(main)
        return
    torch.manual_seed(0)
    kernel = _compile(main)
    a = torch.randint(-8, 8, (M, K), dtype=torch.int8, device="npu")
    b = torch.randint(-8, 8, (K, N), dtype=torch.int8, device="npu")
    c = torch.zeros(M, N, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()
    # int32 matmul is not implemented on NPU; compare on CPU.
    ref = (a.cpu().int() @ b.cpu().int()).to("npu")
    torch.testing.assert_close(c, ref)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("K,num_stages", [(200, 2), (200, 3), (456, 2), (256, 2)])
def test_gemm_v0_ktail_pipeline(K, num_stages):
    """Pipelined K-loop with an unaligned K-tail: software pipelining
    (``T.Pipelined``) multi-versions the L1 buffers, and every non-zero
    version base used to skip the zero-init (the codegen's ``need_clear`` was
    ``dst_offset == 0``), so a K-tail Mmad accumulated stale data from the
    previous iteration's version (~97% wrong). The fix grants the clear to
    any copy whose dst offset is provably a whole multiple of the tile
    element count (a version base). K=256 is the aligned sanity anchor."""
    M = N = 256
    block_M = block_N = 128
    block_K = 64

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(4, is_npu=True) as (cid, _):
            bx = cid // 2
            by = cid % 2
            A_L1 = T.alloc_shared((block_M, block_K), "float16")
            B_L1 = T.alloc_shared((block_K, block_N), "float16")
            C_L0 = T.alloc_L0C((block_M, block_N), "float")
            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.Pipelined(loop_k, num_stages=num_stages):
                    T.barrier_all()
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    T.copy(B[k * block_K, by * block_N], B_L1)
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()
                T.copy(C_L0, C[bx * block_M, by * block_N])

    torch.manual_seed(0)
    kernel = _compile(main)
    a = torch.randn(M, K, dtype=torch.float16, device="npu")
    b = torch.randn(K, N, dtype=torch.float16, device="npu")
    c = torch.zeros(M, N, dtype=torch.float16, device="npu")
    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()
    ref = a.float() @ b.float()
    torch.testing.assert_close(c.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize(
    "kL0Size,expected",
    [
        (128, "pass"),  # C0-aligned (128 % 32 == 0): fine with kL0split >= 2
        (96, "pass"),  # C0-aligned (96 % 32 == 0)
        (48, "reject"),  # 48 % 32 == 16: every tile consumed with a 64-slot
        # C0 extent while the L1 zN layout provides only 48 rows; the L0
        # ping-pong bases desynchronize. Must be compile-time rejected.
    ],
)
def test_gemm_v0_ktail_int8_kl0size_c0(kL0Size, expected):
    """int8 with ``kL0split > 1`` requires ``kL0Size`` itself to be
    C0-aligned (a multiple of 32): each full tile's mma consumes
    ``ceil(kL0Size/32)*32`` K-slots, so a non-aligned kL0Size (e.g. 48) reads
    unwritten L0B slots on the FIRST tile already, not just the tail."""
    M = N = 256
    block_M = block_N = 128
    K = 256  # aligned overall; only kL0Size varies

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "int8"),
        B: T.Tensor((K, N), "int8"),
        C: T.Tensor((M, N), "int32"),
    ):
        with T.Kernel(4, is_npu=True) as (cid, _):
            bx = cid // 2
            by = cid % 2
            A_L1 = T.alloc_L1((block_M, K), "int8")
            B_L1 = T.alloc_L1((K, block_N), "int8")
            C_L0 = T.alloc_L0C((block_M, block_N), "int32")
            with T.Scope("C"):
                T.copy(A[bx * block_M, 0], A_L1)
                T.copy(B[0, by * block_N], B_L1)
                T.barrier_all()
                T.gemm_v0(A_L1, B_L1, C_L0, init=True, kL0Size=kL0Size)
                T.barrier_all()
                T.copy(C_L0, C[bx * block_M, by * block_N])

    if expected == "reject":
        with pytest.raises(RuntimeError, match="Compilation Failed"):
            _compile(main)
        return
    torch.manual_seed(0)
    kernel = _compile(main)
    a = torch.randint(-8, 8, (M, K), dtype=torch.int8, device="npu")
    b = torch.randint(-8, 8, (K, N), dtype=torch.int8, device="npu")
    c = torch.zeros(M, N, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()
    ref = (a.cpu().int() @ b.cpu().int()).to("npu")
    torch.testing.assert_close(c, ref)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="gemm_v0 correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("K_L1", [200, 152])
def test_gemm_v0_ktail_pipeline_parallel(K_L1):
    """Pipelined K-loop with an unaligned K_L1 AND a T.Parallel epilogue (the
    layout_map-loss path): InjectSoftwarePipeline versions the L1 buffers and
    used to bake the version stride from the LOGICAL slice size (128*200 =
    25600), while each version physically occupies the zN fractal extent
    (26624) -- so version 0's fractal tail overlapped version 1's base and the
    K-tail Mmad read stale data (59% wrong). RewriteAllocBuffer now pads the
    per-version shape to the fractal extent, which fixes both the version
    stride and the flattened Allocate size."""
    M = N = 256
    K = 2 * K_L1  # exactly two K_L1-wide iterations
    block_M = block_N = 128

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(4, is_npu=True) as (cid, _):
            bx = cid // 2
            by = cid % 2
            A_L1 = T.alloc_shared((block_M, K_L1), "float16")
            B_L1 = T.alloc_shared((K_L1, block_N), "float16")
            C_L0 = T.alloc_L0C((block_M, block_N), "float")
            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.Pipelined(loop_k, num_stages=2):
                    T.barrier_all()
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()
                # T.Parallel epilogue: forces the AscendLowerParallelToVector
                # layout_map-loss path (L1 buffers keep logical shapes).
                for i, j in T.Parallel(block_M, block_N):
                    C[bx * block_M + i, by * block_N + j] = C_L0[i, j]

    torch.manual_seed(0)
    kernel = _compile(main)
    a = torch.randn(M, K, dtype=torch.float16, device="npu")
    b = torch.randn(K, N, dtype=torch.float16, device="npu")
    c = torch.zeros(M, N, dtype=torch.float16, device="npu")
    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()
    ref = a.float() @ b.float()
    torch.testing.assert_close(c.float(), ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
