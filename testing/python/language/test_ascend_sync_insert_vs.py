"""Comprehensive tests for AscendSyncInsertVS pass.

Covers:
  L0 - Threshold tests (smoke):
    1. Config disabled is a no-op
    2. V->V barrier inserted
    3. No sync for independent buffers

  L1 - Functional tests (full trigger matrix + mechanisms):
    1.  V->V PipeBarrier
    2.  S->V EventPair
    3.  S->MTE2 EventPair
    4.  S->MTE3 EventPair
    5.  No MTE2->V sync (core negative)
    6.  No V->MTE3 sync (core negative)
    7.  No S->S sync (scalar pipeline is in-order)
    8.  No MTE2->MTE2 sync (same non-S, non-V pipeline)
    9.  No MTE3->MTE3 sync (same non-S, non-V pipeline)
    10. No MTE2->MTE3 / MTE3->MTE2 sync
    11. PIPE_M (GEMM) ignored — no M_V / V_M events
    12. S->S write-then-read no sync
    13. Loop back-edge V->V barrier
    14. Loop back-edge V->V preserved with preceding MTE2 (back-edge overwrite guard)
    15. Loop back-edge S->V cross-iteration
    16. If/else history isolation (different buffers per branch)
    17. If/else same buffer same pipeline (no PipeBarrier_ALL)
    18. Dedup multiple V deps in single statement
    19. Consecutive V->V across statements (each needs own barrier)
    20. Event-pair dedup (multiple S->V deps in single statement)
    21. Mixed dedup (barrier + event in one statement)
    22. Event ID rotation (1..7 round-robin)
    23. Event ID wraparound (mod 8, IDs reused)
    24. Set/Wait flag ID pairing
    25. Alias detection via physical address overlap
    26. V->S EventPair (scalar read from V-written UB)
    27. MTE2->S EventPair (scalar read from MTE2-written UB)
    28. MTE3->S EventPair (scalar read from MTE3-read UB)

  L2 - Anomaly tests:
    1. Read-only no sync
    2. Single statement no crash
    3. Resource scope (CV combine) no crash
    4. Nested loop back-edge (inner + outer)
    5. If-without-else history propagation
    6. Nested loop V->S dedup (revisit dedup guard, PR #1384)
    7. Outer back-edge V -> inner S cross-iteration sync

  Boundary - Platform/backend edge cases:
    1. A5 skips PIPE_V barrier (PTO)
    2. A5 skips PIPE_V barrier (AscendC)
    3. A5 still emits event pairs (S->V)
    4. Both backends consistent
    5. PTO auto-enabled by default
    6. AscendC defaults to VS-OFF
    7. Platform "auto" inserts V barrier (non-A5)
    8. Full pipeline integration smoke
    9. Pre-existing barrier recognition (sibling + VS, no duplicate)
    10. VS with CV combine OFF
    11. VS with memory planning OFF

Strategy:
  - VS-only mode (sibling AscendSyncInsert OFF, VS ON) for isolation.
  - CombineCV is enabled by default (TL_ASCEND_AUTO_CV_COMBINE: True) as it is
    required for correct pipeline tracking.  A dedicated test verifies the
    CV-combine-OFF path.
  - Differential testing: compile same kernel with VS ON vs OFF, assert delta.
  - Inspect generated source via kernel.get_kernel_source() + regex assertions.
  - Parametrized over ascendc and pto backends.

"""

import re

import pytest

import tilelang
import tilelang.language as T

# ---------------------------------------------------------------------------
# Pass configuration dicts
# ---------------------------------------------------------------------------

PASS_VS_ONLY = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

PASS_NO_SYNC = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

PASS_FULL = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

PASS_VS_NO_CV = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
}

PASS_VS_NO_PLAN = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


# ---------------------------------------------------------------------------
# Parametrize helper
# ---------------------------------------------------------------------------

TARGETS = ["ascendc", "pto"]
TARGETS_WITH_PTO = pytest.mark.parametrize("target", TARGETS)


# ---------------------------------------------------------------------------
# Sync pattern definitions for each backend
# ---------------------------------------------------------------------------

_SYNC_PATTERNS = {
    "barrier_v": {
        "ascendc": r"AscendC::PipeBarrier<PIPE_V>",
        "pto": r"pipe_barrier\(PIPE_V\)",
    },
    "barrier_all": {
        "ascendc": r"AscendC::PipeBarrier<PIPE_ALL>",
        "pto": r"pipe_barrier\(PIPE_ALL\)",
    },
    "v_v": {
        "ascendc": r"AscendC::SetFlag<AscendC::HardEvent::V_V>",
        "pto": r"set_flag\(PIPE_V,\s*PIPE_V",
    },
    "s_v": {
        "ascendc": r"AscendC::SetFlag<AscendC::HardEvent::S_V>",
        "pto": r"set_flag\(PIPE_S,\s*PIPE_V",
    },
    "v_s": {
        "ascendc": r"AscendC::SetFlag<AscendC::HardEvent::V_S>",
        "pto": r"set_flag\(PIPE_V,\s*PIPE_S",
    },
    "s_mte2": {
        "ascendc": r"AscendC::SetFlag<AscendC::HardEvent::S_MTE2>",
        "pto": r"set_flag\(PIPE_S,\s*PIPE_MTE2",
    },
    "mte2_s": {
        "ascendc": r"AscendC::SetFlag<AscendC::HardEvent::MTE2_S>",
        "pto": r"set_flag\(PIPE_MTE2,\s*PIPE_S",
    },
    "s_mte3": {
        "ascendc": r"AscendC::SetFlag<AscendC::HardEvent::S_MTE3>",
        "pto": r"set_flag\(PIPE_S,\s*PIPE_MTE3",
    },
    "mte3_s": {
        "ascendc": r"AscendC::SetFlag<AscendC::HardEvent::MTE3_S>",
        "pto": r"set_flag\(PIPE_MTE3,\s*PIPE_S",
    },
}

# Generic patterns to detect ANY auto sync inserted by VS
_ANY_BARRIER = {
    "ascendc": r"AscendC::PipeBarrier<PIPE_",
    "pto": r"pipe_barrier\(PIPE_",
}
_ANY_SETFLAG = {
    "ascendc": r"AscendC::SetFlag<AscendC::HardEvent::",
    "pto": r"set_flag\(PIPE_",
}
_ANY_WAITFLAG = {
    "ascendc": r"AscendC::WaitFlag<AscendC::HardEvent::",
    "pto": r"wait_flag\(PIPE_",
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compile_and_get_source(program, pass_configs, target="ascendc", platform="auto", out_idx=None):
    kwargs = {"pass_configs": pass_configs, "target": target, "platform": platform}
    if out_idx is not None:
        kwargs["out_idx"] = out_idx
    kernel = tilelang.compile(program, **kwargs)
    return kernel.get_kernel_source(), kernel


def _count_pattern(src, pattern):
    return len(re.findall(pattern, src))


def _get_sync_pattern(target, sync_kind):
    patterns = _SYNC_PATTERNS.get(sync_kind, {})
    return patterns.get(target, "")


def _count_sync(src, target, sync_kind):
    pattern = _get_sync_pattern(target, sync_kind)
    if not pattern:
        return 0
    return _count_pattern(src, pattern)


def _assert_has_sync(src, target, sync_kind):
    count = _count_sync(src, target, sync_kind)
    assert count > 0, f"Expected sync '{sync_kind}' in {target} source, but not found.\nSource:\n{src}"


def _assert_no_sync(src, target, sync_kind):
    count = _count_sync(src, target, sync_kind)
    assert count == 0, f"Expected NO sync '{sync_kind}' in {target} source, but found {count}.\nSource:\n{src}"


def _count_any_auto_sync(src, target):
    """Count total auto sync operations (barriers + set_flags + wait_flags)."""
    total = 0
    for pat in [_ANY_BARRIER, _ANY_SETFLAG, _ANY_WAITFLAG]:
        total += _count_pattern(src, pat[target])
    return total


def _assert_no_auto_sync(src, target):
    """Assert that NO auto sync operations exist in the source."""
    total = _count_any_auto_sync(src, target)
    assert total == 0, f"Expected NO auto sync in {target} source, but found {total} sync operations.\nSource:\n{src}"


# ---------------------------------------------------------------------------
# L0 - Threshold tests (smoke)
# ---------------------------------------------------------------------------


@TARGETS_WITH_PTO
def test_config_disabled_is_noop(target):
    """VS pass disabled -> no sync operations in output."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.tile.exp(b_ub, b_ub)

    src, _ = _compile_and_get_source(main, PASS_NO_SYNC, target=target, out_idx=[])
    _assert_no_auto_sync(src, target)


@TARGETS_WITH_PTO
def test_v_to_v_barrier_inserted(target):
    """V->V dependency on same buffer -> PipeBarrier<PIPE_V> inserted."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.tile.exp(b_ub, b_ub)

    src_on, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[])
    src_off, _ = _compile_and_get_source(main, PASS_NO_SYNC, target=target, out_idx=[])

    _assert_has_sync(src_on, target, "barrier_v")
    _assert_no_sync(src_off, target, "barrier_v")


@TARGETS_WITH_PTO
def test_no_sync_independent_buffers(target):
    """V ops on different buffers -> no V->V barrier."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.copy(B[:, :], b_ub)
            T.tile.exp(a_ub, a_ub)
            T.tile.exp(b_ub, b_ub)

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[])
    _assert_no_sync(src, target, "barrier_v")


# ---------------------------------------------------------------------------
# L1 - Functional tests
# ---------------------------------------------------------------------------


@TARGETS_WITH_PTO
def test_v_to_v_pipe_barrier(target):
    """V write -> V read+write same buffer -> PipeBarrier<PIPE_V>."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.tile.exp(b_ub, b_ub)
            T.copy(b_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    _assert_has_sync(src, target, "barrier_v")


@TARGETS_WITH_PTO
def test_s_to_v_event_pair(target):
    """S write to UB -> V read same UB -> SetFlag/WaitFlag S_V."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            a_ub[0, 0] = T.cast(1.0, "float32")
            T.tile.exp(b_ub, a_ub)
            T.copy(b_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    _assert_has_sync(src, target, "s_v")


@TARGETS_WITH_PTO
def test_s_to_mte2_event(target):
    """S write to UB -> MTE2 write same UB -> SetFlag/WaitFlag S_MTE2."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            a_ub[0, 0] = T.cast(1.0, "float32")
            T.copy(A[:, :], a_ub)
            T.copy(a_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    _assert_has_sync(src, target, "s_mte2")


@TARGETS_WITH_PTO
def test_s_to_mte3_event(target):
    """S write to UB -> MTE3 read same UB -> SetFlag/WaitFlag S_MTE3."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            a_ub[0, 0] = T.cast(1.0, "float32")
            T.copy(a_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    _assert_has_sync(src, target, "s_mte3")


@TARGETS_WITH_PTO
def test_no_mte2_to_v_sync(target):
    """MTE2 write UB -> V read same UB -> NO sync (VS core negative)."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.copy(b_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    _assert_no_auto_sync(src, target)


@TARGETS_WITH_PTO
def test_no_v_to_mte3_sync(target):
    """V write UB -> MTE3 read same UB -> NO sync (VS core negative)."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(a_ub, a_ub)
            T.copy(a_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    _assert_no_auto_sync(src, target)


@TARGETS_WITH_PTO
def test_no_s_to_s_sync(target):
    """S write -> S write same buffer -> NO S_S sync (scalar in-order).

    Note: MTE2->S and S->MTE3 syncs ARE expected (they involve different
    pipelines). This test only asserts the absence of S_S event pairs.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            a_ub[0, 0] = T.cast(1.0, "float32")
            a_ub[0, 1] = T.cast(2.0, "float32")
            T.copy(a_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    # S->S should NOT produce any sync (ShouldSync returns false for S,S)
    s_s_pattern = {
        "ascendc": r"HardEvent::S_S",
        "pto": r"set_flag\(PIPE_S,\s*PIPE_S",
    }
    assert _count_pattern(src, s_s_pattern[target]) == 0


@TARGETS_WITH_PTO
def test_loop_back_edge_v_to_v(target):
    """Loop body with only V ops -> cross-iteration V->V -> barrier."""

    @T.prim_func
    def main(
        A: T.Tensor((128,), "float32"),  # type: ignore
        B: T.Tensor((128,), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((128,), "float32")
            T.copy(A[:], a_ub)
            for _i in T.serial(4):
                T.tile.exp(a_ub, a_ub)
            T.copy(a_ub, B[:])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    _assert_has_sync(src, target, "barrier_v")


@TARGETS_WITH_PTO
def test_loop_back_edge_v_to_v_with_mte2(target):
    """Loop back-edge V->V preserved when MTE2 precedes V in loop body.

    The MTE2 copy (GM->UB) at the start of each iteration overwrites the
    same buffer that the V op (exp) writes at the end.  Without the
    back-edge preservation fix, the MTE2 access would overwrite the V
    back-edge in the access history, causing the cross-iteration V->V
    dependency to be missed.
    """

    @T.prim_func
    def main(
        A: T.Tensor((128,), "float32"),  # type: ignore
        B: T.Tensor((128,), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((128,), "float32")
            T.copy(A[:], a_ub)
            for _i in T.serial(4):
                T.copy(A[:], a_ub)
                T.tile.exp(a_ub, a_ub)
            T.copy(a_ub, B[:])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    _assert_has_sync(src, target, "barrier_v")


@TARGETS_WITH_PTO
def test_if_else_history_isolation(target):
    """If/else branches each have V->V barrier; no cross-branch sync."""

    @T.prim_func
    def main(
        A: T.Tensor((2, 64), "float32"),  # type: ignore
        B: T.Tensor((2, 64), "float32"),  # type: ignore
        sel: T.Tensor((1,), "int32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s = sel[0]
            if s > 0:
                x_ub = T.alloc_ub((64,), "float32")
                T.copy(A[0, :], x_ub)
                T.tile.exp(x_ub, x_ub)
                T.tile.mul(x_ub, x_ub, x_ub)
                T.copy(x_ub, B[0, :])
            else:
                y_ub = T.alloc_ub((64,), "float32")
                T.copy(A[1, :], y_ub)
                T.tile.exp(y_ub, y_ub)
                T.tile.mul(y_ub, y_ub, y_ub)
                T.copy(y_ub, B[1, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    _assert_has_sync(src, target, "barrier_v")
    _assert_no_sync(src, target, "s_v")
    _assert_no_sync(src, target, "v_s")


@TARGETS_WITH_PTO
def test_dedup_multiple_v_deps(target):
    """Single V op with multiple V->V deps -> deduped to 1 barrier."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
        C: T.Tensor((64, 128), "float32"),  # type: ignore
        D: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub1 = T.alloc_ub((64, 128), "float32")
            ub2 = T.alloc_ub((64, 128), "float32")
            ub3 = T.alloc_ub((64, 128), "float32")
            ub_out = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], ub1)
            T.copy(B[:, :], ub2)
            T.copy(C[:, :], ub3)
            T.tile.exp(ub1, ub1)
            T.tile.exp(ub2, ub2)
            T.tile.exp(ub3, ub3)
            T.tile.add(ub_out, ub1, ub2)
            T.copy(ub_out, D[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[3])

    barrier_count = _count_sync(src, target, "barrier_v")
    assert barrier_count == 1, (
        f"Expected exactly 1 V barrier (3 V->V deps on ub1/ub2/ub3 deduped to 1 in the add statement), got {barrier_count}.\nSource:\n{src}"
    )


@TARGETS_WITH_PTO
def test_consecutive_v_to_v_each_needs_barrier(target):
    """exp(b_ub,b_ub) -> mul(b_ub,b_ub,b_ub): 2 V->V deps across separate
    statements, each needs its own barrier.

    PipeBarrier<PIPE_V> ensures pre-barrier V ops complete before post-barrier
    V ops begin, but does NOT order ops within the post-barrier region. Since
    both exp and mul execute after barrier #1, mul may start before exp's write
    commits — a second barrier is required for the RAW/WAW hazard.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.tile.exp(b_ub, b_ub)
            T.tile.mul(b_ub, b_ub, b_ub)
            T.copy(b_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    barrier_count = _count_sync(src, target, "barrier_v")
    assert barrier_count == 2, (
        f"Expected exactly 2 V barriers (one per V->V dependency across separate statements), got {barrier_count}.\nSource:\n{src}"
    )


@TARGETS_WITH_PTO
def test_event_id_rotation(target):
    """Multiple S->V EventPairs -> event IDs are unique (rotation)."""

    @T.prim_func
    def main(
        A: T.Tensor((3, 64), "float32"),  # type: ignore
        B: T.Tensor((3, 64), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub1 = T.alloc_ub((64,), "float32")
            ub2 = T.alloc_ub((64,), "float32")
            ub3 = T.alloc_ub((64,), "float32")
            T.copy(A[0, :], ub1)
            T.copy(A[1, :], ub2)
            T.copy(A[2, :], ub3)
            ub1[0] = T.cast(1.0, "float32")
            T.tile.exp(ub1, ub1)
            ub2[0] = T.cast(1.0, "float32")
            T.tile.exp(ub2, ub2)
            ub3[0] = T.cast(1.0, "float32")
            T.tile.exp(ub3, ub3)
            T.copy(ub1, B[0, :])
            T.copy(ub2, B[1, :])
            T.copy(ub3, B[2, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    if target == "ascendc":
        ids = re.findall(r"AscendC::SetFlag<AscendC::HardEvent::S_V>\((\d+)\)", src)
    else:
        ids = re.findall(r"set_flag\(PIPE_S,\s*PIPE_V,\s*EVENT_ID(\d+)\)", src)

    assert len(ids) >= 2, f"Expected at least 2 S_V event IDs, got {len(ids)}.\nSource:\n{src}"
    assert len(set(ids)) == len(ids), f"Event IDs should be unique, got {ids}"


@TARGETS_WITH_PTO
def test_alias_detection_via_address(target):
    """Two sequential UB buffers with non-overlapping lifetimes may share
    address. VS should handle aliasing without crash.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
        C: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub1 = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], ub1)
            T.tile.exp(ub1, ub1)
            T.copy(ub1, B[:, :])

            ub2 = T.alloc_ub((64, 128), "float32")
            T.copy(C[:, :], ub2)
            T.tile.exp(ub2, ub2)
            T.copy(ub2, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    assert len(src) > 0, "Expected non-empty source"


@TARGETS_WITH_PTO
def test_mte2_to_mte2_no_sync(target):
    """Two MTE2 writes to same UB -> NO sync (ShouldSync(MTE2,MTE2)=false)."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
        C: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.copy(B[:, :], a_ub)
            T.copy(a_ub, C[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[2])
    _assert_no_auto_sync(src, target)


@TARGETS_WITH_PTO
def test_mte3_to_mte3_no_sync(target):
    """Two MTE3 reads from same UB -> NO sync (ShouldSync(MTE3,MTE3)=false)."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
        C: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.copy(a_ub, B[:, :])
            T.copy(a_ub, C[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1, 2])
    _assert_no_auto_sync(src, target)


@TARGETS_WITH_PTO
def test_mte2_to_mte3_no_sync(target):
    """MTE2 write UB -> MTE3 read same UB -> NO sync (neither is S or V->V)."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.copy(a_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    _assert_no_auto_sync(src, target)


@TARGETS_WITH_PTO
def test_s_to_s_write_then_read_no_sync(target):
    """S write -> S read same buffer -> NO S_S sync (scalar in-order).

    Note: MTE2->S and S->MTE3 syncs ARE expected.  This test only asserts
    the absence of S_S event pairs.
    """

    @T.prim_func
    def main(
        A: T.Tensor((1, 128), "float32"),  # type: ignore
        B: T.Tensor((1, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((1, 128), "float32")
            T.copy(A[:, :], a_ub)
            a_ub[0, 0] = T.cast(1.0, "float32")
            val = a_ub[0, 0]  # noqa: F841  intentional scalar read (S pipeline)
            T.copy(a_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    s_s_pattern = {
        "ascendc": r"HardEvent::S_S",
        "pto": r"set_flag\(PIPE_S,\s*PIPE_S",
    }
    assert _count_pattern(src, s_s_pattern[target]) == 0


def test_pipe_m_ignored():
    """GEMM (PIPE_M) -> V on different buffer -> NO M_V / V_M sync.

    PIPE_M is not in IsSupportedPipeline, so GEMM ops are invisible to VS.
    No M_V or V_M event pairs should ever be inserted.
    """

    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float16"),  # type: ignore
        B: T.Tensor((128, 128), "float16"),  # type: ignore
        C: T.Tensor((128, 128), "float16"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            A_L1 = T.alloc_shared((128, 128), "float16")
            B_L1 = T.alloc_shared((128, 128), "float16")
            C_L0 = T.alloc_fragment((128, 128), "float32")
            c_ub = T.alloc_shared((128, 128), "float16")

            T.copy(A[:, :], A_L1)
            T.copy(B[:, :], B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0, init=True)
            T.copy(C_L0, c_ub)
            T.tile.exp(c_ub, c_ub)
            T.copy(c_ub, C[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target="ascendc", out_idx=[2])

    m_v_pattern = r"HardEvent::M_V"
    v_m_pattern = r"HardEvent::V_M"
    assert _count_pattern(src, m_v_pattern) == 0, f"PIPE_M should be ignored, but M_V event found.\nSource:\n{src}"
    assert _count_pattern(src, v_m_pattern) == 0, f"PIPE_M should be ignored, but V_M event found.\nSource:\n{src}"


@TARGETS_WITH_PTO
def test_if_else_same_buffer_same_pipeline(target):
    """If/else with same buffer, V in both branches -> V->V barrier, no
    PipeBarrier_ALL (same pipeline, no cross-branch conflict).
    """

    @T.prim_func
    def main(
        A: T.Tensor((2, 64), "float32"),  # type: ignore
        B: T.Tensor((64,), "float32"),  # type: ignore
        sel: T.Tensor((1,), "int32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64,), "float32")
            s = sel[0]
            if s > 0:
                T.copy(A[0, :], a_ub)
                T.tile.exp(a_ub, a_ub)
                T.tile.mul(a_ub, a_ub, a_ub)
            else:
                T.copy(A[1, :], a_ub)
                T.tile.exp(a_ub, a_ub)
                T.tile.mul(a_ub, a_ub, a_ub)
            T.copy(a_ub, B[:])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    _assert_has_sync(src, target, "barrier_v")
    _assert_no_sync(src, target, "barrier_all")


@TARGETS_WITH_PTO
def test_event_pair_dedup(target):
    """Multiple S->V deps in single V statement -> deduped to 1 S_V event."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
        C: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub1 = T.alloc_ub((64, 128), "float32")
            ub2 = T.alloc_ub((64, 128), "float32")
            ub_out = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], ub1)
            T.copy(B[:, :], ub2)
            ub1[0, 0] = T.cast(1.0, "float32")
            ub2[0, 0] = T.cast(1.0, "float32")
            T.tile.add(ub_out, ub1, ub2)
            T.copy(ub_out, C[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[2])

    sv_count = _count_sync(src, target, "s_v")
    assert sv_count == 1, f"Expected exactly 1 S_V event pair (deduped from 2 S->V deps), got {sv_count}.\nSource:\n{src}"


@TARGETS_WITH_PTO
def test_mixed_dedup_barrier_and_event(target):
    """Single V statement with V->V (barrier) + S->V (event) -> both inserted.

    The dedup logic should keep both sync types since they are different.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
        C: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub1 = T.alloc_ub((64, 128), "float32")
            ub2 = T.alloc_ub((64, 128), "float32")
            ub_out = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], ub1)
            T.tile.exp(ub1, ub1)
            T.copy(B[:, :], ub2)
            ub2[0, 0] = T.cast(1.0, "float32")
            T.tile.add(ub_out, ub1, ub2)
            T.copy(ub_out, C[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[2])

    _assert_has_sync(src, target, "barrier_v")
    _assert_has_sync(src, target, "s_v")


# ---------------------------------------------------------------------------
# Cross-statement EventPair dedup tests
# ---------------------------------------------------------------------------


@TARGETS_WITH_PTO
def test_cross_stmt_s_v_dedup(target):
    """Two separate V statements reading two S-written UBs -> 1 S_V event.

    Cross-statement dedup: set_flag(S_V) synchronizes ALL prior S writes,
    so the second S->V dependency is covered by the first event pair.
    Without synced_events tracking, each V statement would generate its
    own S_V event pair.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
        C: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub1 = T.alloc_ub((64, 128), "float32")
            ub2 = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], ub1)
            T.copy(B[:, :], ub2)
            ub1[0, 0] = T.cast(1.0, "float32")
            ub2[0, 0] = T.cast(1.0, "float32")
            T.tile.exp(ub1, ub1)
            T.tile.exp(ub2, ub2)
            T.tile.add(ub1, ub1, ub2)
            T.copy(ub1, C[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[2])

    sv_count = _count_sync(src, target, "s_v")
    assert sv_count == 1, f"Expected exactly 1 S_V event pair (cross-stmt dedup), got {sv_count}.\nSource:\n{src}"


@TARGETS_WITH_PTO
def test_cross_stmt_v_s_dedup(target):
    """Two separate S scalar reads from two V-written UBs -> 1 V_S event.

    Cross-statement dedup: set_flag(V_S) synchronizes ALL prior V writes,
    so the second V->S dependency is covered by the first event pair.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub1 = T.alloc_ub((1,), "float32")
            ub2 = T.alloc_ub((1,), "float32")
            b_ub = T.alloc_ub((1,), "float32")
            T.copy(A[0, 0:1], ub1)
            T.copy(A[0, 0:1], ub2)
            T.tile.exp(ub1, ub1)
            T.tile.exp(ub2, ub2)
            val1 = ub1[0]
            b_ub[0] = val1
            val2 = ub2[0]
            b_ub[0] = val2
            T.copy(ub1, B[0, 0:1])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    vs_count = _count_sync(src, target, "v_s")
    assert vs_count == 1, f"Expected exactly 1 V_S event pair (cross-stmt dedup), got {vs_count}.\nSource:\n{src}"


@TARGETS_WITH_PTO
def test_new_s_write_breaks_dedup(target):
    """S write after first S_V event -> second S_V event NOT deduped.

    The new S write creates a fresh access history entry with empty
    synced_events, so the subsequent V read requires a new S_V event.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
        C: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub1 = T.alloc_ub((64, 128), "float32")
            ub2 = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], ub1)
            T.copy(B[:, :], ub2)
            ub1[0, 0] = T.cast(1.0, "float32")
            T.tile.exp(ub1, ub1)
            ub2[0, 0] = T.cast(1.0, "float32")
            T.tile.exp(ub2, ub2)
            T.tile.add(ub1, ub1, ub2)
            T.copy(ub1, C[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[2])

    sv_count = _count_sync(src, target, "s_v")
    assert sv_count == 2, f"Expected exactly 2 S_V event pairs (new S write breaks dedup), got {sv_count}.\nSource:\n{src}"


@TARGETS_WITH_PTO
def test_write_history_event_dedup(target):
    """WAW dep via write_history is deduped by prior S_V event.

    S writes ub1, V reads ub1 (EventPair_S_V inserted), V writes ub1.
    The V write checks write_history (S-write with synced_events) and
    the S->V WAW dependency is deduped.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub1 = T.alloc_ub((64, 128), "float32")
            ub_out = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], ub1)
            ub1[0, 0] = T.cast(1.0, "float32")
            T.tile.exp(ub_out, ub1)
            T.tile.exp(ub1, ub1)
            T.copy(ub1, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    sv_count = _count_sync(src, target, "s_v")
    assert sv_count == 1, f"Expected exactly 1 S_V event pair (write_history WAW deduped), got {sv_count}.\nSource:\n{src}"


@TARGETS_WITH_PTO
def test_loop_back_edge_event_dedup(target):
    """Loop with 2 S writes + 2 V reads -> 1 S_V + 1 V_S per iteration.

    Cross-statement dedup applies in both the first pass (S->V) and the
    revisit pass (back-edge V->S). Without synced_events tracking, each
    pass would generate 2 event pairs instead of 1.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub1 = T.alloc_ub((64, 128), "float32")
            ub2 = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], ub1)
            T.copy(A[:, :], ub2)
            for _i in T.serial(4):
                ub1[0, 0] = T.cast(1.0, "float32")
                ub2[0, 0] = T.cast(1.0, "float32")
                T.tile.exp(ub1, ub1)
                T.tile.exp(ub2, ub2)
            T.copy(ub1, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    sv_count = _count_sync(src, target, "s_v")
    vs_count = _count_sync(src, target, "v_s")
    assert sv_count == 1, f"Expected exactly 1 S_V event pair (loop cross-stmt dedup), got {sv_count}.\nSource:\n{src}"
    assert vs_count == 1, f"Expected exactly 1 V_S event pair (loop back-edge dedup), got {vs_count}.\nSource:\n{src}"


@TARGETS_WITH_PTO
def test_mixed_direction_no_cross_dedup(target):
    """V->S and S->V in sequence -> both event types inserted (no cross-dedup).

    EventPair_V_S and EventPair_S_V are different sync types; dedup is
    exact-string-match only.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub1 = T.alloc_ub((1,), "float32")
            ub2 = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((1,), "float32")
            T.copy(A[0, 0:1], ub1)
            T.copy(A[:, :], ub2)
            T.tile.exp(ub1, ub1)
            val1 = ub1[0]
            b_ub[0] = val1
            ub2[0, 0] = T.cast(1.0, "float32")
            T.tile.exp(ub2, ub2)
            T.copy(ub2, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    vs_count = _count_sync(src, target, "v_s")
    sv_count = _count_sync(src, target, "s_v")
    assert vs_count == 1, f"Expected exactly 1 V_S event pair, got {vs_count}.\nSource:\n{src}"
    assert sv_count == 1, f"Expected exactly 1 S_V event pair, got {sv_count}.\nSource:\n{src}"


@TARGETS_WITH_PTO
def test_event_id_wraparound(target):
    """9+ S->V event pairs -> event IDs wrap around mod 8.

    AllocateEventId does (counter + 1) % 8, cycling 1..7, 0, 1..  After 8
    allocations ID 0 is used, and subsequent IDs repeat.  This test verifies
    the wrap occurs and IDs are reused.
    """

    @T.prim_func
    def main(
        A: T.Tensor((9, 64), "float32"),  # type: ignore
        B: T.Tensor((9, 64), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub0 = T.alloc_ub((64,), "float32")
            ub1 = T.alloc_ub((64,), "float32")
            ub2 = T.alloc_ub((64,), "float32")
            ub3 = T.alloc_ub((64,), "float32")
            ub4 = T.alloc_ub((64,), "float32")
            ub5 = T.alloc_ub((64,), "float32")
            ub6 = T.alloc_ub((64,), "float32")
            ub7 = T.alloc_ub((64,), "float32")
            ub8 = T.alloc_ub((64,), "float32")
            T.copy(A[0, :], ub0)
            ub0[0] = T.cast(1.0, "float32")
            T.tile.exp(ub0, ub0)
            T.copy(ub0, B[0, :])
            T.copy(A[1, :], ub1)
            ub1[0] = T.cast(1.0, "float32")
            T.tile.exp(ub1, ub1)
            T.copy(ub1, B[1, :])
            T.copy(A[2, :], ub2)
            ub2[0] = T.cast(1.0, "float32")
            T.tile.exp(ub2, ub2)
            T.copy(ub2, B[2, :])
            T.copy(A[3, :], ub3)
            ub3[0] = T.cast(1.0, "float32")
            T.tile.exp(ub3, ub3)
            T.copy(ub3, B[3, :])
            T.copy(A[4, :], ub4)
            ub4[0] = T.cast(1.0, "float32")
            T.tile.exp(ub4, ub4)
            T.copy(ub4, B[4, :])
            T.copy(A[5, :], ub5)
            ub5[0] = T.cast(1.0, "float32")
            T.tile.exp(ub5, ub5)
            T.copy(ub5, B[5, :])
            T.copy(A[6, :], ub6)
            ub6[0] = T.cast(1.0, "float32")
            T.tile.exp(ub6, ub6)
            T.copy(ub6, B[6, :])
            T.copy(A[7, :], ub7)
            ub7[0] = T.cast(1.0, "float32")
            T.tile.exp(ub7, ub7)
            T.copy(ub7, B[7, :])
            T.copy(A[8, :], ub8)
            ub8[0] = T.cast(1.0, "float32")
            T.tile.exp(ub8, ub8)
            T.copy(ub8, B[8, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    if target == "ascendc":
        sv_ids = re.findall(r"AscendC::SetFlag<AscendC::HardEvent::S_V>\((\d+)\)", src)
    else:
        sv_ids = re.findall(r"set_flag\(PIPE_S,\s*PIPE_V,\s*EVENT_ID(\d+)\)", src)

    assert len(sv_ids) >= 9, f"Expected at least 9 S_V event IDs, got {len(sv_ids)}.\nSource:\n{src}"
    assert "0" in sv_ids, f"Expected ID 0 (wrap-around after 8 allocations), got IDs {sv_ids}.\nSource:\n{src}"
    assert len(set(sv_ids)) < len(sv_ids), f"Expected some IDs to be reused after wrap, all unique: {sv_ids}"


@TARGETS_WITH_PTO
def test_set_wait_flag_pairing(target):
    """Each set_flag must have a matching wait_flag with the same event ID."""

    @T.prim_func
    def main(
        A: T.Tensor((3, 64), "float32"),  # type: ignore
        B: T.Tensor((3, 64), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub1 = T.alloc_ub((64,), "float32")
            ub2 = T.alloc_ub((64,), "float32")
            ub3 = T.alloc_ub((64,), "float32")
            T.copy(A[0, :], ub1)
            ub1[0] = T.cast(1.0, "float32")
            T.tile.exp(ub1, ub1)
            T.copy(ub1, B[0, :])
            T.copy(A[1, :], ub2)
            ub2[0] = T.cast(1.0, "float32")
            T.tile.exp(ub2, ub2)
            T.copy(ub2, B[1, :])
            T.copy(A[2, :], ub3)
            ub3[0] = T.cast(1.0, "float32")
            T.tile.exp(ub3, ub3)
            T.copy(ub3, B[2, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    if target == "ascendc":
        set_ids = re.findall(r"AscendC::SetFlag<AscendC::HardEvent::(\w+)>\((\d+)\)", src)
        wait_ids = re.findall(r"AscendC::WaitFlag<AscendC::HardEvent::(\w+)>\((\d+)\)", src)
    else:
        set_ids = re.findall(r"set_flag\(PIPE_\w+,\s*PIPE_\w+,\s*EVENT_ID(\d+)\)", src)
        wait_ids = re.findall(r"wait_flag\(PIPE_\w+,\s*PIPE_\w+,\s*EVENT_ID(\d+)\)", src)

    if target == "ascendc":
        set_pairs = [(et, eid) for et, eid in set_ids]
        wait_pairs = [(et, eid) for et, eid in wait_ids]
    else:
        set_pairs = [("any", eid) for eid in set_ids]
        wait_pairs = [("any", eid) for eid in wait_ids]

    assert len(set_pairs) == len(wait_pairs), f"set_flag count ({len(set_pairs)}) != wait_flag count ({len(wait_pairs)}).\nSource:\n{src}"

    for sp in set_pairs:
        assert sp in wait_pairs, f"set_flag {sp} has no matching wait_flag.\nset: {set_pairs}\nwait: {wait_pairs}\nSource:\n{src}"


@TARGETS_WITH_PTO
def test_loop_back_edge_s_to_v(target):
    """Loop body with S write -> V read+write -> cross-iteration V->S sync.

    In the revisit pass, the back-edge (V from prev iteration) triggers a
    V->S event pair before the S write in the next iteration.
    """

    @T.prim_func
    def main(
        A: T.Tensor((128,), "float32"),  # type: ignore
        B: T.Tensor((128,), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((128,), "float32")
            T.copy(A[:], a_ub)
            for _i in T.serial(4):
                a_ub[0] = T.cast(1.0, "float32")
                T.tile.exp(a_ub, a_ub)
            T.copy(a_ub, B[:])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    _assert_has_sync(src, target, "v_s")


@TARGETS_WITH_PTO
def test_v_to_s_scalar_read(target):
    """V write to UB -> S scalar read same UB -> SetFlag/WaitFlag V_S."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((1,), "float32")
            b_ub = T.alloc_ub((1,), "float32")
            T.copy(A[0, 0:1], a_ub)
            T.tile.exp(a_ub, a_ub)
            val = a_ub[0]
            b_ub[0] = val
            T.copy(a_ub, B[0, 0:1])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    _assert_has_sync(src, target, "v_s")


@TARGETS_WITH_PTO
def test_mte2_to_s_scalar_read(target):
    """MTE2 write to UB -> S scalar read same UB -> SetFlag/WaitFlag MTE2_S."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((1,), "float32")
            b_ub = T.alloc_ub((1,), "float32")
            T.copy(A[0, 0:1], a_ub)
            val = a_ub[0]
            b_ub[0] = val
            T.copy(a_ub, B[0, 0:1])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    _assert_has_sync(src, target, "mte2_s")


@TARGETS_WITH_PTO
def test_mte3_to_s_scalar_read(target):
    """MTE2 write UB -> MTE3 read UB -> S scalar read same UB.

    With write-history tracking, the S read is synced directly to the
    MTE2 write (MTE2_S, correct RAW) rather than to the intervening
    MTE3 read (which was an invalid RAR proxy since MTE2->MTE3 is
    unsynced by the VS pass).
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((1,), "float32")
            b_ub = T.alloc_ub((1,), "float32")
            T.copy(A[0, 0:1], a_ub)
            T.copy(a_ub, B[0, 0:1])
            val = a_ub[0]
            b_ub[0] = val
            T.copy(a_ub, B[0, 1:2])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    _assert_has_sync(src, target, "mte2_s")


# ---------------------------------------------------------------------------
# L2 - Anomaly tests
# ---------------------------------------------------------------------------


@TARGETS_WITH_PTO
def test_read_only_no_sync(target):
    """V read-only ops on same buffer -> no V->V barrier."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
        C: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            c_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.copy(B[:, :], b_ub)
            T.tile.add(c_ub, a_ub, b_ub)
            T.copy(c_ub, C[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[2])

    _assert_no_sync(src, target, "barrier_v")
    _assert_no_sync(src, target, "s_v")
    _assert_no_sync(src, target, "v_s")


@TARGETS_WITH_PTO
def test_single_stmt_no_crash(target):
    """Single T.copy pair -> no crash."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.copy(a_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    assert len(src) > 0


def test_resource_scope_no_crash():
    """Kernel with CV combine (resource_scope) -> VS handles without crash.

    Uses a GEMM + Vector fusion kernel that triggers CombineCV to split
    into resource_scope=0 (Cube) and resource_scope=1 (Vector) blocks.
    The VS pass must handle the resource_scope boundary without crash.
    """

    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float16"),  # type: ignore
        B: T.Tensor((128, 128), "float16"),  # type: ignore
        C: T.Tensor((128, 128), "float16"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            A_L1 = T.alloc_shared((128, 128), "float16")
            B_L1 = T.alloc_shared((128, 128), "float16")
            C_L0 = T.alloc_fragment((128, 128), "float32")
            c_ub = T.alloc_shared((128, 128), "float16")

            T.copy(A[:, :], A_L1)
            T.copy(B[:, :], B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0, init=True)
            T.copy(C_L0, c_ub)
            T.tile.exp(c_ub, c_ub)
            T.copy(c_ub, C[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target="ascendc", out_idx=[2])
    assert len(src) > 0


@TARGETS_WITH_PTO
def test_nested_loop_back_edge(target):
    """Nested loops with V ops in inner body -> V->V barriers from back-edges.

    Both the inner loop and the outer loop produce V->V back-edge dependencies
    that should result in PipeBarrier<PIPE_V>.
    """

    @T.prim_func
    def main(
        A: T.Tensor((128,), "float32"),  # type: ignore
        B: T.Tensor((128,), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((128,), "float32")
            T.copy(A[:], a_ub)
            for _i in T.serial(4):
                for _j in T.serial(4):
                    T.tile.exp(a_ub, a_ub)
            T.copy(a_ub, B[:])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    _assert_has_sync(src, target, "barrier_v")


@TARGETS_WITH_PTO
def test_nested_loop_v_to_s_dedup(target):
    """Nested loops with single V->S dep in inner body -> exactly 1 V_S pair.

    The inner loop body has a V write (exp) followed by an S scalar read
    of the same buffer.  Each enclosing loop's revisit pass re-traverses
    the inner body, but the non-back-edge guard in GetRequiredSyncType
    prevents re-inserting the same-iteration V_S pair, so exactly one V_S
    pair is emitted regardless of nesting depth.
    """

    @T.prim_func
    def main(
        A: T.Tensor((128,), "float32"),  # type: ignore
        B: T.Tensor((128,), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((128,), "float32")
            b_ub = T.alloc_ub((1,), "float32")
            T.copy(A[:], a_ub)
            for _i in T.serial(4):
                for _j in T.serial(4):
                    T.tile.exp(a_ub, a_ub)
                    val = a_ub[0]
                    b_ub[0] = val
            T.copy(a_ub, B[:])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    vs_count = _count_sync(src, target, "v_s")
    assert vs_count == 1, f"Expected exactly 1 V_S pair for nested-loop V->S dep, got {vs_count}.\nSource:\n{src}"


@TARGETS_WITH_PTO
def test_outer_back_edge_v_to_inner_s_sync(target):
    """Outer loop back-edge V write -> next-iter inner-body S read needs V_S.

    The outer loop writes V (exp) AFTER the inner loop; the inner loop's
    first statement reads the same buffer (S scalar read).  This creates a
    cross-iteration RAW dependency (outer back-edge V -> inner S) requiring
    a V_S event pair.  The outer-loop revisit pass re-traverses the inner
    body, detecting this cross-iteration dependency.
    """

    @T.prim_func
    def main(
        A: T.Tensor((128,), "float32"),  # type: ignore
        B: T.Tensor((128,), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((128,), "float32")
            b_ub = T.alloc_ub((1,), "float32")
            T.copy(A[:], a_ub)
            for _i in T.serial(4):
                for _j in T.serial(4):
                    val = a_ub[0]
                    b_ub[0] = val
                T.tile.exp(a_ub, a_ub)
            T.copy(a_ub, B[:])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    _assert_has_sync(src, target, "v_s")


@TARGETS_WITH_PTO
def test_if_without_else_propagation(target):
    """If-without-else -> then-branch history propagates, no PipeBarrier_ALL.

    The then-branch has V->V barrier.  No else means no cross-branch conflict
    check, so no PipeBarrier_ALL.  After the if, the then-branch's V history
    is merged into current_access_history_.
    """

    @T.prim_func
    def main(
        A: T.Tensor((2, 64), "float32"),  # type: ignore
        B: T.Tensor((64,), "float32"),  # type: ignore
        sel: T.Tensor((1,), "int32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64,), "float32")
            s = sel[0]
            T.copy(A[0, :], a_ub)
            if s > 0:
                T.tile.exp(a_ub, a_ub)
                T.tile.mul(a_ub, a_ub, a_ub)
            T.copy(a_ub, B[:])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    _assert_has_sync(src, target, "barrier_v")
    _assert_no_sync(src, target, "barrier_all")


# ---------------------------------------------------------------------------
# Boundary - Platform/backend edge cases
# ---------------------------------------------------------------------------


def test_a5_pto_pipe_v_v_barrier():

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.tile.exp(b_ub, b_ub)
            T.copy(b_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target="pto", platform="A5", out_idx=[1])
    _assert_has_sync(src, "pto", "v_v")


@TARGETS_WITH_PTO
def test_both_backends_consistent(target):
    """Same V->V kernel -> all backends have PipeBarrier<PIPE_V>."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.tile.exp(b_ub, b_ub)
            T.copy(b_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])
    _assert_has_sync(src, target, "barrier_v")


def test_pto_auto_enabled_by_default():
    """PTO target without explicit VS config -> VS auto-enabled."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.tile.exp(b_ub, b_ub)
            T.copy(b_ub, B[:, :])

    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    }
    src, _ = _compile_and_get_source(main, pass_configs, target="pto", out_idx=[1])
    _assert_has_sync(src, "pto", "barrier_v")


@TARGETS_WITH_PTO
def test_full_pipeline_integration(target):
    """Full pipeline (sibling + VS + CV + planning) -> no crash, has sync."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.tile.exp(b_ub, b_ub)
            T.tile.mul(b_ub, b_ub, b_ub)
            T.copy(b_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_FULL, target=target, out_idx=[1])
    assert len(src) > 0
    assert _count_pattern(src, _ANY_BARRIER[target]) > 0 or _count_pattern(src, _ANY_SETFLAG[target]) > 0


@TARGETS_WITH_PTO
def test_preexisting_barrier_recognition(target):
    """Sibling pass ON + VS ON -> VS does not duplicate V->V barriers.

    The sibling AscendSyncInsert pass runs first and inserts barriers.
    VS should recognize pre-existing ascend_auto_barrier calls and avoid
    duplicating V->V barriers for the same dependency.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.tile.exp(b_ub, b_ub)
            T.copy(b_ub, B[:, :])

    src_full, _ = _compile_and_get_source(main, PASS_FULL, target=target, out_idx=[1])
    src_vs, _ = _compile_and_get_source(main, PASS_VS_ONLY, target=target, out_idx=[1])

    barrier_full = _count_sync(src_full, target, "barrier_v")
    barrier_vs = _count_sync(src_vs, target, "barrier_v")

    assert barrier_vs == 1, f"VS-only should insert exactly 1 V->V barrier, got {barrier_vs}.\nSource:\n{src_vs}"
    assert barrier_full <= barrier_vs + 1, (
        f"With sibling ON, VS should not duplicate barriers. Full={barrier_full}, VS-only={barrier_vs}.\nSource:\n{src_full}"
    )


def test_vs_with_cv_combine_off():
    """VS ON + CV combine OFF -> V->V barrier still inserted (no resource_scope).

    Without CV combine, there are no resource_scope AttrStmt boundaries.
    VS should still insert V->V barriers based on name matching.

    Note: PTO backend has a known compilation issue with CV combine OFF
    (unrelated to VS pass), so this test only runs on ascendc.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.tile.exp(b_ub, b_ub)
            T.copy(b_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_NO_CV, target="ascendc", out_idx=[1])
    _assert_has_sync(src, "ascendc", "barrier_v")


@TARGETS_WITH_PTO
def test_vs_with_memory_planning_off(target):
    """VS ON + memory planning OFF -> V->V barrier still inserted (name-based).

    Without memory planning, address_map/size_map are empty, so
    GetPhysicalAddress returns -1 for all buffers.  Alias detection is
    disabled, but name-based dependency tracking still works.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.tile.exp(b_ub, b_ub)
            T.copy(b_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_NO_PLAN, target=target, out_idx=[1])
    _assert_has_sync(src, target, "barrier_v")


def test_a5_ascendc_pipe_v_v_barrier():
    """Platform A5 + ascendc + V->V -> no PipeBarrier<PIPE_V>."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.tile.exp(b_ub, b_ub)
            T.copy(b_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target="ascendc", platform="A5", out_idx=[1])
    _assert_has_sync(src, "ascendc", "v_v")


def test_a5_still_emits_event_pairs():
    """Platform A5 + S->V dependency -> S_V event pair still inserted.

    A5 only skips PipeBarrier<PIPE_V>.  Event pairs (S_V, etc.) should
    still be emitted.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            a_ub[0, 0] = T.cast(1.0, "float32")
            T.tile.exp(b_ub, a_ub)
            T.copy(b_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target="pto", platform="A5", out_idx=[1])
    _assert_has_sync(src, "pto", "s_v")


def test_ascendc_default_vs_off():
    """AscendC target without explicit VS config -> VS not auto-enabled.

    Unlike PTO, ascendc's _TARGET_PASS_DEFAULTS is empty, so VS is OFF
    by default.  No auto sync should appear.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.tile.exp(b_ub, b_ub)
            T.copy(b_ub, B[:, :])

    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    }
    src, _ = _compile_and_get_source(main, pass_configs, target="ascendc", out_idx=[1])
    _assert_no_auto_sync(src, "ascendc")


def test_platform_auto_inserts_v_barrier():
    """Platform "auto" + V->V -> PipeBarrier<PIPE_V> inserted (non-A5).

    "auto" should resolve to a non-A5 platform, so PIPE_V barriers are
    NOT skipped (unlike A5).
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.tile.exp(b_ub, b_ub)
            T.copy(b_ub, B[:, :])

    src, _ = _compile_and_get_source(main, PASS_VS_ONLY, target="pto", platform="auto", out_idx=[1])
    _assert_has_sync(src, "pto", "barrier_v")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
