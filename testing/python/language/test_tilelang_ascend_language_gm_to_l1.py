import pytest
import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout
import torch

"""
GM -> L1 copy correctness suite: full-block, K-tail, and L1-splice scenarios.

Feature under test
------------------
``copy_gm_to_l1`` (src/tl_templates/ascend/common.h) transfers a 2-D tile from
Global Memory to L1 Buffer, applying an ND->NZ format conversion along the way.
The helper accepts ``realTailM`` / ``realTailN`` so that a *partial* source tile
(whose valid rows/cols are smaller than the allocated ``dstM x dstN`` L1 tile)
can be copied without the caller having to pre-zero the L1 region.

The clearing mechanism
----------------------
When ``realTailM < dstM`` (or ``realTailN < dstN``) the helper calls
``InitConstValue`` to zero the **entire** ``dstM x dstN`` L1 tile before writing
the partial data.  This is essential for K-tail GEMM: the zero-padded tail of
the last K-tile contributes ``0 * B = 0`` to the accumulator, keeping the matmul
correct.

The splice bug (fixed)
----------------------
When the user issues **multiple** ``T.copy`` calls that write into sub-regions
of the *same* L1 tile (a "vertical splice" / "L1 merge"), the second and
subsequent copies must **not** clear the tile -- otherwise they clobber the data
written by the first copy.  The fix adds a ``need_clear`` flag to the helper and
makes the codegen pass ``need_clear = (dst_offset == 0)`` so that only the
primary copy (tile base, offset 0) performs the zero-init.

Scenarios covered (all GM -> L1, target = ascendc)
--------------------------------------------------
  Group 1 - Full-block copy (no splice, no tail):
      Single T.copy fills the entire L1 tile.  Baseline correctness.
      With and without explicit layout annotation; transpose_B variants.

  Group 2 - K-tail GEMM (clearing regression guard):
      K is not divisible by K_L1; the last K-tile is partial and must be
      zero-padded by the clearing mechanism (need_clear == true, offset 0).

  Group 3 - 2-way vertical splice (nZ layout, transpose_B=True):
      Two disjoint GM tensors are written into the same L1 tile at different
      row offsets, forming one coherent NZ tile for the Cube unit.
      Aligned splits (multiple of 16) and non-aligned splits (1, 127).

  Group 4 - 3-way splice with unfilled tail:
      Three sources cover only part of the tile; the remainder must stay zero
      from the first copy's clear (need_clear == true) while copies 2 and 3
      skip the clear (need_clear == false).

  Group 5 - 2-way vertical splice (zN layout, transpose_B=False):
      Same splice pattern but with zN layout on the spliced buffer (A matrix,
      M-dimension split) and a non-transposed GEMM.
"""

TARGET = "ascendc"

DEV_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CUBE_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before the session."""
    tilelang.cache.clear_cache()
    yield


def _torch_dtype(dtype):
    return {
        "float": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


# =============================================================================
# Group 1 - Full-block copy (no splice, no tail)      [risk: low]
# Single T.copy fills the entire L1 tile.  Baseline correctness for both
# annotated (nZ/zN) and non-annotated (default zN) L1 buffers.
# =============================================================================
def full_copy_annotated(block_M, block_N, dim, dtype, accum_dtype):
    """GEMM with explicit zN (Q) + nZ (K) layout annotation, transpose_B=True.
    Mirrors examples/tests/nz2nd.py::full_copy_gemm."""

    @T.prim_func
    def main(
        Q: T.Tensor([1, block_M, dim], dtype),  # type: ignore
        K: T.Tensor([1, block_N, dim], dtype),  # type: ignore
        S: T.Tensor([1, block_M, block_N], accum_dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                }
            )
            T.copy(Q[0, :, :], q_l1[:, :])
            T.copy(K[0, :, :], k_l1[:, :])
            T.gemm_v0(q_l1, k_l1, l0c, transpose_B=True, init=True)
            T.copy(l0c, S[0, :, :])

    return main


def full_copy_plain(M, N, K, block_M, block_N, K_L1, dtype, accum_dtype):
    """Plain GEMM without layout annotation (default zN), transpose_B=False.
    Mirrors examples/gemm/example_gemm.py."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[0, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, 0], B_L1)
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                T.copy(C_L0, C[0, 0])

    return main


def run_test_full_copy_annotated(block_M, block_N, dim, dtype, accum_dtype):
    torch.manual_seed(0)
    func = full_copy_annotated(block_M, block_N, dim, dtype, accum_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=DEV_CONFIGS, target=TARGET)
    td = _torch_dtype(dtype)
    q = torch.randn(1, block_M, dim, dtype=td).npu()
    k = torch.randn(1, block_N, dim, dtype=td).npu()
    torch.npu.synchronize()
    s = func(q, k)
    ref = torch.einsum("bqd,bkd->bqk", q.float(), k.float()).to(torch.float32)
    torch.testing.assert_close(s, ref, rtol=1e-2, atol=1e-2)


def run_test_full_copy_plain(M, N, K, block_M, block_N, K_L1, dtype, accum_dtype):
    torch.manual_seed(0)
    func = full_copy_plain(M, N, K, block_M, block_N, K_L1, dtype, accum_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=CUBE_CONFIGS, target=TARGET)
    td = _torch_dtype(dtype)
    a = torch.randn(M, K, dtype=td).npu()
    b = torch.randn(K, N, dtype=td).npu()
    torch.npu.synchronize()
    c = func(a, b)
    ref = a @ b
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "dtype,accum_dtype",
    [
        ("bfloat16", "float32"),
        ("float16", "float32"),
    ],
)
def test_full_copy_annotated(dtype, accum_dtype):
    run_test_full_copy_annotated(64, 128, 128, dtype, accum_dtype)


@pytest.mark.parametrize(
    "dtype,accum_dtype",
    [
        ("float16", "float"),
        ("bfloat16", "float"),
    ],
)
def test_full_copy_plain(dtype, accum_dtype):
    run_test_full_copy_plain(128, 128, 128, 128, 128, 128, dtype, accum_dtype)


# =============================================================================
# Group 2 - K-tail GEMM (clearing regression guard)   [risk: low]
# K is not divisible by K_L1; the last K-tile is partial.  The clearing
# mechanism (need_clear == true for offset-0 copies) must zero-pad the tail
# so that 0 * B = 0 keeps the matmul correct.
# =============================================================================
def run_test_k_tail_gemm(M, N, K, block_M, block_N, K_L1, dtype, accum_dtype):
    torch.manual_seed(0)
    func = full_copy_plain(M, N, K, block_M, block_N, K_L1, dtype, accum_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=CUBE_CONFIGS, target=TARGET)
    td = _torch_dtype(dtype)
    a = torch.randn(M, K, dtype=td).npu()
    b = torch.randn(K, N, dtype=td).npu()
    torch.npu.synchronize()
    c = func(a, b)
    ref = a @ b
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


# (M, N, K, block_M, block_N, K_L1) -- K deliberately non-divisible.
k_tail_configs = [
    (128, 128, 160, 128, 128, 64),  # K=160, last tile K=32 (half tile)
    (128, 256, 96, 128, 256, 64),  # K=96,  last tile K=32
    (128, 128, 176, 128, 128, 64),  # K=176, last tile K=48
]


@pytest.mark.parametrize("M,N,K,block_M,block_N,K_L1", k_tail_configs)
def test_k_tail_gemm(M, N, K, block_M, block_N, K_L1):
    run_test_k_tail_gemm(M, N, K, block_M, block_N, K_L1, "float16", "float")


# =============================================================================
# Group 3 - 2-way vertical splice (nZ layout)         [risk: medium]
# Two disjoint GM tensors (K_src0, K_src1) are written into the same k_l1
# tile at row offsets [0:split] and [split:block_N].  The first copy (offset 0)
# clears the full tile and writes its rows; the second copy (non-zero offset)
# must NOT clear, otherwise it clobbers the first copy's data.
#
# Splits include both 16-aligned values (clean NZ fractal boundaries) and
# non-aligned values (1, 127) to stress the offset / DMA engine.
# =============================================================================
def splice_2way_nz(block_M, block_N, dim, split, dtype, accum_dtype):
    """nZ-layout splice with transpose_B=True.  K rows are split into two
    sources that together fill the full [block_N, dim] L1 tile."""

    @T.prim_func
    def main(
        Q: T.Tensor([1, block_M, dim], dtype),  # type: ignore
        K_src0: T.Tensor([1, split, dim], dtype),  # type: ignore
        K_src1: T.Tensor([1, block_N - split, dim], dtype),  # type: ignore
        S: T.Tensor([1, block_M, block_N], accum_dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                }
            )
            T.copy(Q[0, :, :], q_l1[:, :])
            T.copy(K_src0[0, :, :], k_l1[0:split, :])
            T.copy(K_src1[0, :, :], k_l1[split:block_N, :])
            T.gemm_v0(q_l1, k_l1, l0c, transpose_B=True, init=True)
            T.copy(l0c, S[0, :, :])

    return main


def run_test_splice_2way_nz(block_M, block_N, dim, split, dtype, accum_dtype):
    torch.manual_seed(0)
    func = splice_2way_nz(block_M, block_N, dim, split, dtype, accum_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=DEV_CONFIGS, target=TARGET)
    td = _torch_dtype(dtype)
    q = torch.randn(1, block_M, dim, dtype=td).npu()
    k_src0 = torch.randn(1, split, dim, dtype=td).npu()
    k_src1 = torch.randn(1, block_N - split, dim, dtype=td).npu()
    torch.npu.synchronize()
    s = func(q, k_src0, k_src1)
    # Reference: assemble full K then compute Q @ K^T
    k_full = torch.cat([k_src0, k_src1], dim=1).float()
    ref = torch.einsum("bqd,bkd->bqk", q.float(), k_full).to(torch.float32)
    torch.testing.assert_close(s, ref, rtol=1e-2, atol=1e-2)


# Aligned splits (multiple of C0_SIZE=16 for bfloat16/float16)
splice_aligned = [16, 32, 48, 64, 80, 96, 112]
# Non-aligned splits (stress offset / DMA alignment)
splice_unaligned = [1, 127]


@pytest.mark.parametrize("split", splice_aligned)
def test_splice_2way_nz_aligned(split):
    run_test_splice_2way_nz(64, 128, 128, split, "bfloat16", "float32")


@pytest.mark.parametrize("split", splice_unaligned)
def test_splice_2way_nz_unaligned(split):
    run_test_splice_2way_nz(64, 128, 128, split, "bfloat16", "float32")


@pytest.mark.parametrize("split", [32, 64, 96])
def test_splice_2way_nz_float16(split):
    run_test_splice_2way_nz(64, 128, 128, split, "float16", "float32")


# =============================================================================
# Group 4 - 3-way splice with unfilled tail           [risk: medium]
# Three sources cover rows [0:s0], [s0:s0+s1], [s0+s1:s0+s1+s2] where
# s0+s1+s2 < block_N.  The remaining rows must stay zero from the first copy's
# clear (need_clear == true).  Copies 2 and 3 skip the clear (need_clear ==
# false).  The GEMM's zero-padded rows contribute zero columns to the output.
# =============================================================================
def splice_3way_tail(block_M, block_N, dim, s0, s1, s2, dtype, accum_dtype):
    total = s0 + s1 + s2

    @T.prim_func
    def main(
        Q: T.Tensor([1, block_M, dim], dtype),  # type: ignore
        K_src0: T.Tensor([1, s0, dim], dtype),  # type: ignore
        K_src1: T.Tensor([1, s1, dim], dtype),  # type: ignore
        K_src2: T.Tensor([1, s2, dim], dtype),  # type: ignore
        S: T.Tensor([1, block_M, block_N], accum_dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                }
            )
            T.copy(Q[0, :, :], q_l1[:, :])
            T.copy(K_src0[0, :, :], k_l1[0:s0, :])
            T.copy(K_src1[0, :, :], k_l1[s0 : s0 + s1, :])
            T.copy(K_src2[0, :, :], k_l1[s0 + s1 : total, :])
            T.gemm_v0(q_l1, k_l1, l0c, transpose_B=True, init=True)
            T.copy(l0c, S[0, :, :])

    return main


def run_test_splice_3way_tail(block_M, block_N, dim, s0, s1, s2, dtype, accum_dtype):
    torch.manual_seed(0)
    func = splice_3way_tail(block_M, block_N, dim, s0, s1, s2, dtype, accum_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=DEV_CONFIGS, target=TARGET)
    td = _torch_dtype(dtype)
    q = torch.randn(1, block_M, dim, dtype=td).npu()
    k0 = torch.randn(1, s0, dim, dtype=td).npu()
    k1 = torch.randn(1, s1, dim, dtype=td).npu()
    k2 = torch.randn(1, s2, dim, dtype=td).npu()
    torch.npu.synchronize()
    s = func(q, k0, k1, k2)
    # Reference: assemble full K with zero-padded tail
    k_full = torch.zeros(1, block_N, dim, dtype=td).npu().float()
    k_full[0, 0:s0, :] = k0[0].float()
    k_full[0, s0 : s0 + s1, :] = k1[0].float()
    k_full[0, s0 + s1 : s0 + s1 + s2, :] = k2[0].float()
    ref = torch.einsum("bqd,bkd->bqk", q.float(), k_full).to(torch.float32)
    torch.testing.assert_close(s, ref, rtol=1e-2, atol=1e-2)


# (s0, s1, s2, block_N) -- total < block_N, tail must be zero-padded.
splice_3way_configs = [
    (48, 48, 24, 128),  # total=120, tail=8  (aligned groups + sub-group tail)
    (32, 32, 32, 128),  # total=96,  tail=32 (clean group boundary)
    (16, 16, 1, 128),  # total=33,  tail=95 (heavy tail, non-aligned last)
]


@pytest.mark.parametrize("s0,s1,s2,block_N", splice_3way_configs)
def test_splice_3way_tail(s0, s1, s2, block_N):
    run_test_splice_3way_tail(64, block_N, 128, s0, s1, s2, "bfloat16", "float32")


# =============================================================================
# Group 5 - 2-way vertical splice (zN layout)         [risk: medium]
# Same splice concept but with zN layout on the spliced buffer (A matrix,
# M-dimension split) and a non-transposed GEMM (C = A @ B).  This exercises
# the zN variant of the Nd2NzParams offset computation.
# =============================================================================
def splice_2way_zn(block_M, block_N, K, split, dtype, accum_dtype):
    """zN-layout splice with transpose_B=False.  A rows (M dimension) are split
    into two sources that fill the full [block_M, K] L1 tile."""

    @T.prim_func
    def main(
        A_src0: T.Tensor([1, split, K], dtype),  # type: ignore
        A_src1: T.Tensor([1, block_M - split, K], dtype),  # type: ignore
        B: T.Tensor([1, K, block_N], dtype),  # type: ignore
        C: T.Tensor([1, block_M, block_N], dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_l1 = T.alloc_L1([block_M, K], dtype)
            b_l1 = T.alloc_L1([K, block_N], dtype)
            l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            T.annotate_layout(
                {
                    a_l1: make_zn_layout(a_l1),
                    b_l1: make_zn_layout(b_l1),
                }
            )
            T.copy(A_src0[0, :, :], a_l1[0:split, :])
            T.copy(A_src1[0, :, :], a_l1[split:block_M, :])
            T.copy(B[0, :, :], b_l1[:, :])
            T.gemm_v0(a_l1, b_l1, l0c, init=True)
            T.copy(l0c, C[0, :, :])

    return main


def run_test_splice_2way_zn(block_M, block_N, K, split, dtype, accum_dtype):
    torch.manual_seed(0)
    func = splice_2way_zn(block_M, block_N, K, split, dtype, accum_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=DEV_CONFIGS, target=TARGET)
    td = _torch_dtype(dtype)
    a_src0 = torch.randn(1, split, K, dtype=td).npu()
    a_src1 = torch.randn(1, block_M - split, K, dtype=td).npu()
    b = torch.randn(1, K, block_N, dtype=td).npu()
    torch.npu.synchronize()
    c = func(a_src0, a_src1, b)
    a_full = torch.cat([a_src0, a_src1], dim=1)
    ref = torch.einsum("bmk,bkn->bmn", a_full, b)
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("split", [32, 64, 96])
def test_splice_2way_zn(split):
    run_test_splice_2way_zn(128, 128, 128, split, "float16", "float")


# =============================================================================
# Group 6 - Splice + K-tail combination               [risk: medium]
# The spliced L1 tile is also the last K-tile in a K-loop (partial K).  This
# combines the splice clear-suppression with the K-tail zero-padding: the
# first copy clears the full tile (need_clear == true), writes its partial-K
# rows; the second copy skips the clear (need_clear == false), writes its
# partial-K rows; the K-tail columns remain zero.
# =============================================================================
def splice_with_k_tail(block_M, block_N, dim, split, K_L1, dtype, accum_dtype):
    """Splice on N dimension (nZ layout) with K-dimension tail.  dim (=K) is
    not divisible by K_L1, so each source copy also has a K-tail.

    Uses 2-D GM tensors (no batch dim) with scalar indices for the K offset,
    matching the basic GEMM pattern (example_gemm.py) so the codegen correctly
    infers copy extents from the destination L1 tile shape."""

    @T.prim_func
    def main(
        Q: T.Tensor([block_M, dim], dtype),  # type: ignore
        K_src0: T.Tensor([split, dim], dtype),  # type: ignore
        K_src1: T.Tensor([block_N - split, dim], dtype),  # type: ignore
        S: T.Tensor([block_M, block_N], accum_dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block_M, K_L1], dtype)
            k_l1 = T.alloc_L1([block_N, K_L1], dtype)
            l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                }
            )
            loop_k = T.ceildiv(dim, K_L1)
            for k in T.serial(loop_k):
                T.copy(Q[0, k * K_L1], q_l1[:, :])
                T.copy(K_src0[0, k * K_L1], k_l1[0:split, :])
                T.copy(K_src1[0, k * K_L1], k_l1[split:block_N, :])
                T.gemm_v0(q_l1, k_l1, l0c, transpose_B=True, init=(k == 0))
            T.copy(l0c, S[0, 0])

    return main


def run_test_splice_with_k_tail(block_M, block_N, dim, split, K_L1, dtype, accum_dtype):
    torch.manual_seed(0)
    func = splice_with_k_tail(block_M, block_N, dim, split, K_L1, dtype, accum_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=DEV_CONFIGS, target=TARGET)
    td = _torch_dtype(dtype)
    q = torch.randn(block_M, dim, dtype=td).npu()
    k_src0 = torch.randn(split, dim, dtype=td).npu()
    k_src1 = torch.randn(block_N - split, dim, dtype=td).npu()
    torch.npu.synchronize()
    s = func(q, k_src0, k_src1)
    k_full = torch.cat([k_src0, k_src1], dim=0).float()
    ref = torch.einsum("qd,kd->qk", q.float(), k_full).to(torch.float32)
    torch.testing.assert_close(s, ref, rtol=1e-2, atol=1e-2)


# dim (=K) is non-divisible by K_L1; split divides N.
splice_ktail_configs = [
    (64, 128, 160, 64, 64),  # K=160, K_L1=64, last K-tile=32; split=64
    (64, 128, 96, 64, 32),  # K=96,  K_L1=32, last K-tile=32; split=64
]


@pytest.mark.parametrize("block_M,block_N,dim,split,K_L1", splice_ktail_configs)
def test_splice_with_k_tail(block_M, block_N, dim, split, K_L1):
    run_test_splice_with_k_tail(block_M, block_N, dim, split, K_L1, "float16", "float")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
