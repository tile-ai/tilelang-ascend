"""Regression tests for double-buffered L0 address straddle (issue #1664).

A buffer allocated as ``[2, ...]`` in L0A/L0B/L0C follows the hardware
ping-pong convention: its two slots live at ``[base, base + size/2)`` and
``[base + size/2, base + size)``, and ownership is transferred with one token
per side.  When ``T.annotate_address`` gives two such buffers overlapping
ranges whose side boundary points differ, a slot of one buffer straddles both
slots of the other; the per-side token then returns while MMAD may still be
reading the straddled region, and the kernel dies at runtime with::

    507015: L0B read/write conflict in the MTE (same address)

The original bug (``flashattn_fwd_v4`` in the GQA backward example, commit
f55c24f7) placed GEMM1's ``[2, 192, 64]`` L0B (24KB slots) and GEMM2's
``[2, 64, 128]`` L0B (16KB slots) both at base 0.  AscendMemoryPlanning now
rejects that layout at compile time.  These tests reproduce the exact shapes
and verify the accepted alternatives (aligned overlap — the fix proposed in
the issue — plus equal-slot and disjoint layouts).
"""

import re

import pytest

import tilelang
import tilelang.language as T

M, DK, BN, DV = 32, 192, 64, 128

# Expert-style config of the original kernel: linear memory planning takes
# annotate_address ranges as given, which is where the straddle used to pass.
PASS_EXPERT_LINEAR = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}

# Same, but with the auto sync passes enabled: the guard must be independent
# of the planning strategy and of the sync insertion mode.
PASS_AUTO_SYNC = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}

CONFLICT_HEADER = "Double-buffered"

# (g2_l0b base, g2_l0a base) for the issue-shaped kernel. GEMM1's L0A is 24KB
# (12KB slots) at base 0; GEMM2's L0A (8KB, 4KB slots) sits disjoint at 24KB
# exactly like the original kernel, so only the L0B layout varies.
ISSUE_BASE0 = (0, 24576)
ISSUE_FIXED = (8192, 24576)  # 8KB shift: both L0B side boundaries at 24KB


@pytest.fixture(scope="module", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


def gqa_style_kernel(g2_l0b_base=0, g2_l0a_base=0):
    """Two double-buffered GEMM groups with the issue #1664 L0B shapes.

    GEMM1 (QK^T): B operand ``[2, DK, BN]`` -> 24KB slots.
    GEMM2 (PV):   B operand ``[2, BN, DV]`` -> 16KB slots.
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, DK), "float16"),  # type: ignore
        Kq: T.Tensor((BN, DK), "float16"),  # type: ignore
        P: T.Tensor((M, BN), "float16"),  # type: ignore
        Vq: T.Tensor((BN, DV), "float16"),  # type: ignore
        C1: T.Tensor((M, BN), "float"),  # type: ignore
        C2: T.Tensor((M, DV), "float"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_l1 = T.alloc_L1((M, DK), "float16")
            k_l1 = T.alloc_L1((BN, DK), "float16")
            p_l1 = T.alloc_L1((M, BN), "float16")
            v_l1 = T.alloc_L1((BN, DV), "float16")

            g1_l0a = T.alloc_L0A((2, M, DK), "float16")
            g1_l0b = T.alloc_L0B((2, DK, BN), "float16")
            g1_l0c = T.alloc_L0C((M, BN), "float")
            g2_l0a = T.alloc_L0A((2, M, BN), "float16")
            g2_l0b = T.alloc_L0B((2, BN, DV), "float16")
            g2_l0c = T.alloc_L0C((M, DV), "float")

            T.annotate_address({g1_l0a: 0, g1_l0b: 0, g2_l0a: g2_l0a_base, g2_l0b: g2_l0b_base})
            with T.Scope("C"):
                T.barrier_all()
                T.copy(A, a_l1)
                T.copy(Kq, k_l1)
                T.copy(P, p_l1)
                T.copy(Vq, v_l1)
                T.barrier_all()

                # GEMM1: A @ Kq^T (K operand double-buffered, 24KB slots)
                for s in T.serial(2):
                    T.copy(a_l1, g1_l0a[s, :, :])
                    T.copy(k_l1, g1_l0b[s, :, :], transpose=True)
                    T.barrier_all()
                    T.mma(g1_l0a[s, :, :], g1_l0b[s, :, :], g1_l0c[:, :], init=(s == 0))
                    T.barrier_all()
                T.copy(g1_l0c[:, :], C1[:, :])
                T.barrier_all()

                # GEMM2: P @ Vq (V operand double-buffered, 16KB slots)
                for s in T.serial(2):
                    T.copy(p_l1, g2_l0a[s, :, :])
                    T.copy(v_l1, g2_l0b[s, :, :])
                    T.barrier_all()
                    T.mma(g2_l0a[s, :, :], g2_l0b[s, :, :], g2_l0c[:, :], init=(s == 0))
                    T.barrier_all()
                T.copy(g2_l0c[:, :], C2[:, :])
                T.barrier_all()

    return main


def equal_slot_kernel(g2_l0b_base=0):
    """Two double-buffered L0B groups with identical slot sizes (mtgr style).

    Both B operands are ``[2, BN, DV]`` -> 16KB slots, so any same-base overlap
    is aligned and must be accepted.
    """

    @T.prim_func
    def main(
        A1: T.Tensor((M, BN), "float16"),  # type: ignore
        B1: T.Tensor((BN, DV), "float16"),  # type: ignore
        A2: T.Tensor((M, BN), "float16"),  # type: ignore
        B2: T.Tensor((BN, DV), "float16"),  # type: ignore
        C1: T.Tensor((M, DV), "float"),  # type: ignore
        C2: T.Tensor((M, DV), "float"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a1_l1 = T.alloc_L1((M, BN), "float16")
            b1_l1 = T.alloc_L1((BN, DV), "float16")
            a2_l1 = T.alloc_L1((M, BN), "float16")
            b2_l1 = T.alloc_L1((BN, DV), "float16")

            g1_l0a = T.alloc_L0A((2, M, BN), "float16")
            g1_l0b = T.alloc_L0B((2, BN, DV), "float16")
            g1_l0c = T.alloc_L0C((M, DV), "float")
            g2_l0a = T.alloc_L0A((2, M, BN), "float16")
            g2_l0b = T.alloc_L0B((2, BN, DV), "float16")
            g2_l0c = T.alloc_L0C((M, DV), "float")

            T.annotate_address({g1_l0b: 0, g2_l0b: g2_l0b_base})

            with T.Scope("C"):
                T.barrier_all()
                T.copy(A1, a1_l1)
                T.copy(B1, b1_l1)
                T.copy(A2, a2_l1)
                T.copy(B2, b2_l1)
                T.barrier_all()

                for s in T.serial(2):
                    T.copy(a1_l1, g1_l0a[s, :, :])
                    T.copy(b1_l1, g1_l0b[s, :, :])
                    T.barrier_all()
                    T.mma(g1_l0a[s, :, :], g1_l0b[s, :, :], g1_l0c[:, :], init=(s == 0))
                    T.barrier_all()
                T.copy(g1_l0c[:, :], C1[:, :])
                T.barrier_all()

                for s in T.serial(2):
                    T.copy(a2_l1, g2_l0a[s, :, :])
                    T.copy(b2_l1, g2_l0b[s, :, :])
                    T.barrier_all()
                    T.mma(g2_l0a[s, :, :], g2_l0b[s, :, :], g2_l0c[:, :], init=(s == 0))
                    T.barrier_all()
                T.copy(g2_l0c[:, :], C2[:, :])
                T.barrier_all()

    return main


def _get_buffer_offsets(kernel_source: str) -> dict[str, int]:
    """Parse buffer byte offsets from the generated AscendC source."""
    offsets = {}
    for line in kernel_source.split("\n"):
        m = re.search(
            r"auto\s+(\w+)\s*=.*GetWithOffset<[^>]+>\(\s*\d+\s*,\s*(\d+)\s*\)",
            line,
        )
        if m:
            offsets[m.group(1)] = int(m.group(2))
    return offsets


def _compile_and_get_offsets(program, pass_configs):
    kernel = tilelang.compile(program, pass_configs=pass_configs, target="ascendc", out_idx=[])
    return _get_buffer_offsets(kernel.get_kernel_source())


# ---------------------------------------------------------------------------
# Rejected layouts
# ---------------------------------------------------------------------------


def test_issue_straddle_l0b_base0_rejected():
    """The issue #1664 layout: both L0B groups at base 0, 24KB vs 16KB slots.

    GEMM2's slot 1 [16KB, 32KB) covers the tail of GEMM1's slot 0 and the head
    of GEMM1's slot 1, so compilation must fail with the straddle diagnostic.
    """
    with pytest.raises(Exception, match=re.escape(CONFLICT_HEADER)) as exc_info:
        tilelang.compile(
            gqa_style_kernel(*ISSUE_BASE0),
            pass_configs=PASS_EXPERT_LINEAR,
            target="ascendc",
            out_idx=[],
        )
    assert "straddles" in str(exc_info.value)
    assert "g1_l0b" in str(exc_info.value) and "g2_l0b" in str(exc_info.value)


def test_issue_straddle_l0b_base0_rejected_auto_sync():
    """The guard must also fire with the auto sync passes enabled."""
    with pytest.raises(Exception, match=re.escape(CONFLICT_HEADER)):
        tilelang.compile(
            gqa_style_kernel(*ISSUE_BASE0),
            pass_configs=PASS_AUTO_SYNC,
            target="ascendc",
            out_idx=[],
        )


def test_straddle_l0a_rejected():
    """Same hazard in L0A: mismatched slots with a straddling base.

    L0B uses the issue's fixed layout (base 8KB), while GEMM2's L0A
    ([2, M, BN] -> 4KB slots) sits at 4KB inside GEMM1's L0A ([2, M, DK] ->
    12KB slots): GEMM2's slot 1 [8KB, 12KB) straddles GEMM1's side boundary.
    """
    with pytest.raises(Exception, match=re.escape(CONFLICT_HEADER)) as exc_info:
        tilelang.compile(
            gqa_style_kernel(g2_l0b_base=ISSUE_FIXED[0], g2_l0a_base=4096),
            pass_configs=PASS_EXPERT_LINEAR,
            target="ascendc",
            out_idx=[],
        )
    assert "l0a" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Accepted layouts
# ---------------------------------------------------------------------------


def test_issue_fixed_layout_base8k_accepted():
    """The fix proposed in issue #1664: shift GEMM2's L0B base to 8KB.

    Both groups then share the side boundary at 24KB and every GEMM2 slot lies
    inside a single GEMM1 slot, so the original per-side ownership tokens keep
    protecting the whole region.
    """
    offsets = _compile_and_get_offsets(gqa_style_kernel(*ISSUE_FIXED), PASS_EXPERT_LINEAR)
    assert offsets["g1_l0b"] == 0
    assert offsets["g2_l0b"] == 8192


def test_aligned_equal_slots_accepted():
    """Equal slot sizes at the same base (mtgr-style overlap) stay valid."""
    offsets = _compile_and_get_offsets(equal_slot_kernel(0), PASS_EXPERT_LINEAR)
    assert offsets["g1_l0b"] == 0
    assert offsets["g2_l0b"] == 0


def test_disjoint_equal_slots_accepted():
    """Disjoint address ranges are always valid."""
    offsets = _compile_and_get_offsets(equal_slot_kernel(32768), PASS_EXPERT_LINEAR)
    assert offsets["g1_l0b"] == 0
    assert offsets["g2_l0b"] == 32768


def test_single_double_buffered_l0b_accepted():
    """A single pre-allocated double-buffered L0B has no pair and must pass."""

    @T.prim_func
    def main(
        A: T.Tensor((M, BN), "float16"),  # type: ignore
        B: T.Tensor((BN, DV), "float16"),  # type: ignore
        C: T.Tensor((M, DV), "float"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_l1 = T.alloc_L1((M, BN), "float16")
            b_l1 = T.alloc_L1((BN, DV), "float16")
            a_l0a = T.alloc_L0A((2, M, BN), "float16")
            b_l0b = T.alloc_L0B((2, BN, DV), "float16")
            c_l0c = T.alloc_L0C((M, DV), "float")
            T.annotate_address({a_l0a: 0, b_l0b: 0})
            with T.Scope("C"):
                T.barrier_all()
                T.copy(A, a_l1)
                T.copy(B, b_l1)
                T.barrier_all()
                for s in T.serial(2):
                    T.copy(a_l1, a_l0a[s, :, :])
                    T.copy(b_l1, b_l0b[s, :, :])
                    T.barrier_all()
                    T.mma(a_l0a[s, :, :], b_l0b[s, :, :], c_l0c[:, :], init=(s == 0))
                    T.barrier_all()
                T.copy(c_l0c[:, :], C[:, :])
                T.barrier_all()

    offsets = _compile_and_get_offsets(main, PASS_EXPERT_LINEAR)
    assert offsets["b_l0b"] == 0
    assert offsets["a_l0a"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
