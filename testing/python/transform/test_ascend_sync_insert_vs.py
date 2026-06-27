"""Tests for ascend_sync_insert_vs pass.

This test suite verifies the simplified auto sync insertion pass
(tl.ascend_auto_sync_vs) which tracks PIPE_V / PIPE_S / PIPE_MTE2 / PIPE_MTE3.

Test strategy:
  - Build PrimFunc kernels that exercise specific pipeline dependency patterns
  - Lower with the vs pass enabled and inspect the lowered IR for expected
    sync intrinsics (tl.ascend_auto_barrier / set_flag / wait_flag)
  - Also run end-to-end correctness tests on NPU
"""

import pytest

import torch

import tilelang
import tilelang.language as T

tir = tilelang.tvm.tir
tvm = tilelang.tvm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYNC_BARRIER = "tl.ascend_auto_barrier"
SYNC_SET_FLAG = "tl.ascend_auto_set_flag"
SYNC_WAIT_FLAG = "tl.ascend_auto_wait_flag"


def _lower_and_get_ir(func, pass_configs, target="ascendc"):
    """Lower a PrimFunc and return the lowered IR module for inspection.

    Uses tilelang.lower which handles target/platform setup correctly.
    """
    with tvm.transform.PassContext(opt_level=3, config=pass_configs):
        artifact = tilelang.lower(func, target=target, platform="A3")

    mod = artifact.device_mod
    if mod is None:
        mod = artifact.host_mod
    return mod


def _count_sync_intrinsics(ir_text):
    """Count sync intrinsics in IR text."""
    return {
        "barrier": ir_text.count("ascend_auto_barrier"),
        "set_flag": ir_text.count("ascend_auto_set_flag"),
        "wait_flag": ir_text.count("ascend_auto_wait_flag"),
    }


def _has_event_pair(ir_text, event_type):
    """Check if a specific event pair (e.g. 'V_S') appears in set/wait flags."""
    return f'"{event_type}"' in ir_text and "ascend_auto_set_flag" in ir_text and "ascend_auto_wait_flag" in ir_text


def _has_pipe_barrier(ir_text, pipeline):
    """Check if a PipeBarrier for a specific pipeline exists."""
    return f'"{pipeline}"' in ir_text and "ascend_auto_barrier" in ir_text


PASS_CONFIGS_VS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

PASS_CONFIGS_DISABLED = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# ---------------------------------------------------------------------------
# IR-level tests: verify sync insertion logic without NPU execution
# ---------------------------------------------------------------------------


def _make_vadd_then_vsub_dep(M=128, N=128, dtype="float16"):
    """V -> V RAW dependency: vadd writes ubuf, vsub reads same ubuf.

    o1 = vadd(x1, x2)   # PIPE_V write ubuf
    o2 = vsub(o1, x3)    # PIPE_V read ubuf  -> needs PipeBarrier<PIPE_V>
    """
    block_M, block_N = M, N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        D: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)
            d_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[0, 0], a_ub)
            T.copy(B[0, 0], b_ub)

            T.tile.add(c_ub, a_ub, b_ub)
            T.tile.sub(d_ub, c_ub, a_ub)

            T.copy(c_ub, C[0, 0])
            T.copy(d_ub, D[0, 0])

    return main


def _make_vadd_vsub_no_dep(M=128, N=128, dtype="float16"):
    """V -> V no dependency: vadd and vsub operate on different buffers.

    o1 = vadd(x1, x2)    # PIPE_V write c_ub
    o2 = vsub(x3, x4)    # PIPE_V write d_ub (no shared buffer) -> no sync
    """
    block_M, block_N = M, N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        D: T.Tensor((M, N), dtype),
        E: T.Tensor((M, N), dtype),
        F: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)
            d_ub = T.alloc_ub((block_M, block_N), dtype)
            e_ub = T.alloc_ub((block_M, block_N), dtype)
            f_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[0, 0], a_ub)
            T.copy(B[0, 0], b_ub)
            T.copy(C[0, 0], c_ub)
            T.copy(D[0, 0], d_ub)

            T.tile.add(e_ub, a_ub, b_ub)
            T.tile.sub(f_ub, c_ub, d_ub)

            T.copy(e_ub, E[0, 0])
            T.copy(f_ub, F[0, 0])

    return main


def _make_s_to_v(M=128, N=128, dtype="float16"):
    """S -> V dependency: BufferStore (SetValue) then vadd reads same buffer.

    a_ub[0,0] = 0.0   # PIPE_S write a_ub
    vadd(c, a, b)     # PIPE_V read a_ub  -> needs EventPair S_V
    """
    block_M, block_N = M, N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(B[0, 0], b_ub)
            T.copy(A[0, 0], a_ub)

            # PIPE_S write: single scalar store to UB
            a_ub[0, 0] = 0.0

            # PIPE_V read a_ub -> S->V dependency
            T.tile.add(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[0, 0])

    return main


def _make_v_to_s(M=128, N=128, dtype="float16"):
    """V -> S dependency: vadd writes buffer, then scalar read (GetValue).

    vadd(c_ub, a_ub, b_ub)   # PIPE_V write c_ub
    a_ub[0,0] = c_ub[0,0]    # PIPE_S read c_ub (BufferLoad in RHS)  -> V->S
    """
    block_M, block_N = M, N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[0, 0], a_ub)
            T.copy(B[0, 0], b_ub)

            # PIPE_V write c_ub
            T.tile.add(c_ub, a_ub, b_ub)

            # PIPE_S read c_ub via BufferLoad in RHS of BufferStore
            a_ub[0, 0] = c_ub[0, 0]

            T.copy(a_ub, C[0, 0])

    return main


def _make_v_to_s_nested_expr(M=128, N=128, dtype="float16"):
    """V -> S dependency with nested BufferLoad in arithmetic expr.

    vadd(c_ub, a_ub, b_ub)                       # PIPE_V write c_ub
    d_ub[0,0] = c_ub[0,0] + tir.const(1, dtype)  # PIPE_S reads c_ub via Add(BufferLoad, FloatImm) -> V->S
                                                  # d_ub is kept alive (via prior copy) so storage
                                                  # rewrite does not merge it with c_ub; the only
                                                  # V->S dep is from the c_ub read inside Add
    """
    block_M, block_N = M, N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)
            d_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[0, 0], a_ub)
            T.copy(B[0, 0], b_ub)
            # Initialize d_ub so its lifetime overlaps with c_ub, preventing
            # storage rewrite from merging d_ub into c_ub.
            T.copy(A[0, 0], d_ub)

            # PIPE_V write c_ub
            T.tile.add(c_ub, a_ub, b_ub)

            # PIPE_S read c_ub via nested BufferLoad in Add expr of BufferStore RHS.
            # d_ub has a distinct physical address from c_ub, so the only V->S
            # dependency comes from reading c_ub inside the Add expression.
            d_ub[0, 0] = c_ub[0, 0] + tir.const(1, dtype)

            T.copy(d_ub, C[0, 0])

    return main


def _make_s_to_mte3(M=128, N=128, dtype="float16"):
    """S -> MTE3 dependency: BufferStore then copy_ub_to_gm.

    a_ub[0,0] = 0.0   # PIPE_S write a_ub
    copy(a_ub -> C)   # PIPE_MTE3 read a_ub  -> needs EventPair S_MTE3
    """
    block_M, block_N = M, N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[0, 0], a_ub)

            # PIPE_S write: single scalar store to UB
            a_ub[0, 0] = 0.0

            # PIPE_MTE3 read a_ub -> S->MTE3 dependency
            T.copy(a_ub, C[0, 0])

    return main


def _make_mte2_to_s(M=128, N=128, dtype="float16"):
    """MTE2 -> S dependency: copy_gm_to_ub then BufferStore (SetValue) on same buffer.

    copy(A -> a_ub)   # PIPE_MTE2 write a_ub
    a_ub[0,0] = 0.0   # PIPE_S write a_ub  -> needs EventPair MTE2_S (WAW)
    copy(a_ub -> C)   # PIPE_MTE3 read a_ub -> also needs EventPair S_MTE3
    """
    block_M, block_N = M, N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_M, block_N), dtype)

            # PIPE_MTE2 write a_ub
            T.copy(A[0, 0], a_ub)

            # PIPE_S write a_ub -> MTE2->S dependency (WAW on same buffer)
            a_ub[0, 0] = 0.0

            # PIPE_MTE3 read a_ub -> S->MTE3 dependency
            T.copy(a_ub, C[0, 0])

    return main


def _make_if_internal_v_to_s(M=128, N=128, dtype="float16"):
    """IfThenElse internal V->S dependency.

    fill(b_ub, 0.0)         # PIPE_V write b_ub
    if cond:
        b_ub[i, j] = a_ub[i,j]  # PIPE_S write b_ub (reads a_ub) -> V->S WAW on b_ub
    No PIPE_ALL barrier should be inserted.
    """
    block_M, block_N = M, N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[0, 0], a_ub)

            # PIPE_V write b_ub
            T.tile.fill(b_ub, 0.0)

            # IfThenElse with internal S write -> V->S WAW on b_ub
            if block_M > 1:
                b_ub[0, 0] = a_ub[0, 0]

            T.copy(b_ub, C[0, 0])

    return main


def _make_loop_back_edge_v_to_v(M=128, N=128, dtype="float16"):
    """Loop back-edge V->V dependency.

    for i in range(K):
        fill(buf, 0.0)       # PIPE_V write buf
        add(buf, buf, a_ub)  # PIPE_V read+write buf
    # Back-edge: add writes buf -> next iter fill writes buf -> V->V WAW
    # Should insert PipeBarrier<PIPE_V> at beginning of loop body (from re-visit pass)
    """
    block_M, block_N = M, N
    K = 4

    @T.prim_func
    def main(
        A: T.Tensor((K, M, N), dtype),
        C: T.Tensor((K, M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            buf = T.alloc_ub((block_M, block_N), dtype)

            for i in T.serial(K):
                T.copy(A[i, 0, 0], a_ub)
                T.tile.fill(buf, 0.0)
                T.tile.add(buf, buf, a_ub)
                T.copy(buf, C[i, 0, 0])

    return main


def _make_loop_if_back_edge_v_to_v(M=128, N=128, dtype="float16"):
    """Loop with IfThenElse where forward dep and back-edge dep both
    require a barrier at the same position inside the if.

    fill(buf, 0.0)           # V write buf (OUTSIDE loop)
    for i in range(K):
        if i > 0:
            fill(buf, 1.0)   # V write buf (INSIDE if)
                              # forward WAW with fill(buf,0.0) -> first_pass inserts barrier
                              # back-edge WAW with fill(buf,1.0) prev iter -> revisit_pass would duplicate
    copy(buf, C)             # MTE3, after loop

    Without P1 (barrier detection): 2 barriers inside if (forward + duplicate back-edge)
    With P1: 1 barrier inside if (back-edge sees existing barrier, skips)
    """
    block_M, block_N = M, N
    K = 4

    @T.prim_func
    def main(
        C: T.Tensor((1, M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            buf = T.alloc_ub((block_M, block_N), dtype)

            T.tile.fill(buf, 0.0)
            for i in T.serial(K):
                if i > 0:
                    T.tile.fill(buf, 1.0)
            T.copy(buf, C[0, 0, 0])

    return main


# ---------------------------------------------------------------------------
# IR inspection tests
# ---------------------------------------------------------------------------


class TestSyncInsertVsIR:
    def test_v_to_v_raw_inserts_pipe_barrier_v(self):
        """V->V RAW: vadd then vsub on same buffer should insert PipeBarrier<PIPE_V>."""
        func = _make_vadd_then_vsub_dep()
        mod = _lower_and_get_ir(func, PASS_CONFIGS_VS)
        ir_text = mod.script()
        counts = _count_sync_intrinsics(ir_text)
        assert counts["barrier"] > 0, f"Expected PipeBarrier for V->V RAW, but got {counts}\nIR:\n{ir_text}"

    def test_v_to_v_no_dep_no_cross_pipeline_sync(self):
        """V->V with different logical buffers: no V->V dependency, and since
        MTE2->V / V->MTE3 are not handled by this pass, no set/wait flags
        should be inserted. Buffer reuse by storage rewrite may create
        physical address overlaps, but those cross-pipeline pairs (MTE2_V,
        V_MTE3) are outside this pass's scope."""
        func = _make_vadd_vsub_no_dep()
        mod = _lower_and_get_ir(func, PASS_CONFIGS_VS)
        ir_text = mod.script()
        counts = _count_sync_intrinsics(ir_text)
        assert counts["set_flag"] == 0, f"Expected no SetFlag (MTE2->V/V->MTE3 not in scope), but got {counts}\nIR:\n{ir_text}"
        assert counts["wait_flag"] == 0, f"Expected no WaitFlag (MTE2->V/V->MTE3 not in scope), but got {counts}\nIR:\n{ir_text}"

    def test_s_to_v_inserts_event_pair(self):
        """S->V: BufferStore (SetValue) then vadd should insert SetFlag/WaitFlag event pair."""
        func = _make_s_to_v()
        mod = _lower_and_get_ir(func, PASS_CONFIGS_VS)
        ir_text = mod.script()
        counts = _count_sync_intrinsics(ir_text)
        assert counts["set_flag"] > 0, f"Expected SetFlag for S->V, but got {counts}\nIR:\n{ir_text}"
        assert counts["wait_flag"] > 0, f"Expected WaitFlag for S->V, but got {counts}\nIR:\n{ir_text}"

    def test_v_to_s_inserts_event_pair(self):
        """V->S: vadd then scalar read (GetValue) should insert SetFlag/WaitFlag event pair."""
        func = _make_v_to_s()
        mod = _lower_and_get_ir(func, PASS_CONFIGS_VS)
        ir_text = mod.script()
        counts = _count_sync_intrinsics(ir_text)
        assert counts["set_flag"] > 0, f"Expected SetFlag for V->S, but got {counts}\nIR:\n{ir_text}"
        assert counts["wait_flag"] > 0, f"Expected WaitFlag for V->S, but got {counts}\nIR:\n{ir_text}"

    def test_v_to_s_nested_expr_inserts_event_pair(self):
        """V->S with nested BufferLoad in Add expr should still be detected."""
        func = _make_v_to_s_nested_expr()
        mod = _lower_and_get_ir(func, PASS_CONFIGS_VS)
        ir_text = mod.script()
        assert _has_event_pair(ir_text, "V_S"), f"Expected V_S event pair for nested V->S, but not found\nIR:\n{ir_text}"

    def test_s_to_mte3_inserts_event_pair(self):
        """S->MTE3: BufferStore then copy_ub_to_gm should insert SetFlag/WaitFlag event pair."""
        func = _make_s_to_mte3()
        mod = _lower_and_get_ir(func, PASS_CONFIGS_VS)
        ir_text = mod.script()
        counts = _count_sync_intrinsics(ir_text)
        assert counts["set_flag"] > 0, f"Expected SetFlag for S->MTE3, but got {counts}\nIR:\n{ir_text}"
        assert counts["wait_flag"] > 0, f"Expected WaitFlag for S->MTE3, but got {counts}\nIR:\n{ir_text}"

    def test_mte2_to_s_inserts_event_pair(self):
        """MTE2->S: copy_gm_to_ub then scalar read (GetValue) should insert SetFlag/WaitFlag."""
        func = _make_mte2_to_s()
        mod = _lower_and_get_ir(func, PASS_CONFIGS_VS)
        ir_text = mod.script()
        counts = _count_sync_intrinsics(ir_text)
        assert counts["set_flag"] > 0, f"Expected SetFlag for MTE2->S, but got {counts}\nIR:\n{ir_text}"
        assert counts["wait_flag"] > 0, f"Expected WaitFlag for MTE2->S, but got {counts}\nIR:\n{ir_text}"

    def test_if_internal_v_to_s_no_pipe_all(self):
        """IfThenElse internal V->S dependency: should insert EventPair V_S
        inside the if, and should NOT insert any PIPE_ALL barrier."""
        func = _make_if_internal_v_to_s()
        mod = _lower_and_get_ir(func, PASS_CONFIGS_VS)
        ir_text = mod.script()
        counts = _count_sync_intrinsics(ir_text)
        # Should have V_S event pair (set_flag + wait_flag)
        assert counts["set_flag"] > 0, f"Expected SetFlag for V->S inside if, but got {counts}\nIR:\n{ir_text}"
        assert counts["wait_flag"] > 0, f"Expected WaitFlag for V->S inside if, but got {counts}\nIR:\n{ir_text}"
        # Should NOT have PIPE_ALL barrier
        assert not _has_pipe_barrier(ir_text, "PIPE_ALL"), f"PIPE_ALL barrier should not be inserted by this pass\nIR:\n{ir_text}"

    def test_disabled_pass_is_noop(self):
        """When tl.ascend_auto_sync_vs=False, no sync intrinsics should be inserted."""
        func = _make_vadd_then_vsub_dep()
        mod_disabled = _lower_and_get_ir(func, PASS_CONFIGS_DISABLED)
        ir_disabled = mod_disabled.script()
        counts_disabled = _count_sync_intrinsics(ir_disabled)

        mod_enabled = _lower_and_get_ir(func, PASS_CONFIGS_VS)
        ir_enabled = mod_enabled.script()
        counts_enabled = _count_sync_intrinsics(ir_enabled)

        assert counts_disabled == {"barrier": 0, "set_flag": 0, "wait_flag": 0}, (
            f"Disabled pass should insert no syncs, but got {counts_disabled}\nIR:\n{ir_disabled}"
        )
        assert counts_enabled["barrier"] > 0 or counts_enabled["set_flag"] > 0, (
            f"Enabled pass should insert syncs, but got {counts_enabled}"
        )

    def test_loop_back_edge_v_to_v(self):
        """Loop back-edge V->V: fill+add in a loop should insert PipeBarrier<PIPE_V>
        at the beginning of the loop body for the back-edge WAW dependency."""
        func = _make_loop_back_edge_v_to_v()
        mod = _lower_and_get_ir(func, PASS_CONFIGS_VS)
        ir_text = mod.script()
        counts = _count_sync_intrinsics(ir_text)
        assert counts["barrier"] > 0, f"Expected PipeBarrier for loop back-edge V->V, but got {counts}\nIR:\n{ir_text}"

    def test_loop_if_back_edge_no_duplicate_barrier(self):
        """When first_pass inserts a barrier inside an IfThenElse for a forward
        V->V WAW, and revisit_pass encounters a back-edge V->V WAW at the same
        position, the pass should detect the existing barrier and not insert
        a duplicate."""
        func = _make_loop_if_back_edge_v_to_v()
        mod = _lower_and_get_ir(func, PASS_CONFIGS_VS)
        ir_text = mod.script()
        counts = _count_sync_intrinsics(ir_text)
        assert counts["barrier"] == 1, (
            f"Expected exactly 1 barrier (forward dep, no duplicate from back-edge), but got {counts}\nIR:\n{ir_text}"
        )


# ---------------------------------------------------------------------------
# End-to-end correctness tests (require functional NPU hardware)
# ---------------------------------------------------------------------------

PASS_CONFIGS_E2E = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


def _e2e_vadd_vsub(M=256, N=256, block_M=128, block_N=128, dtype="float16"):
    """Kernel: c = a + b; d = c - a  (V->V RAW dependency)."""
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        D: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            d_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.add(c_ub, a_ub, b_ub)
            T.tile.sub(d_ub, c_ub, a_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])
            T.copy(d_ub, D[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def _e2e_elementwise(M=256, N=256, block_M=128, block_N=128, dtype="float16"):
    """Kernel: B = abs(A)  (MTE2->V->MTE3 pipeline)."""
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.abs(b_ub, a_ub)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def _e2e_vadd(M=256, N=256, block_M=128, block_N=128, dtype="float16"):
    """Kernel: C = A + B  (MTE2->V->MTE3 with two inputs)."""
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
            T.tile.add(c_ub, a_ub, b_ub)
            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


class TestSyncInsertVsE2E:
    """End-to-end tests that compile and run kernels on NPU hardware.

    These tests verify that kernels compiled with the vs sync pass produce
    correct numerical results. They require functional NPU hardware.
    """

    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.parametrize("dtype", ["float16"])
    @pytest.mark.xfail(
        reason="AscendSyncInsert (tl.ascend_auto_sync) does not insert sufficient "
        "MTE2->V / V->MTE3 syncs for this kernel with V->V RAW dependency. "
        "Previously masked by the buggy ShouldSync in VS pass which over-synced "
        "V<->MTE2/MTE3. This is a pre-existing AscendSyncInsert deficiency, not a "
        "VS pass regression.",
        strict=True,
    )
    def test_vadd_vsub_raw(self, target, dtype):
        """E2E: V->V RAW dependency kernel should produce correct results."""
        M, N = 256, 256
        func = _e2e_vadd_vsub(M, N, dtype=dtype)
        kernel = tilelang.compile(func, out_idx=[2, 3], pass_configs=PASS_CONFIGS_E2E, target=target)

        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        a = torch.randn(M, N, dtype=torch_dtype).npu()
        b = torch.randn(M, N, dtype=torch_dtype).npu()
        torch.npu.synchronize()

        c, d = kernel(a, b)
        ref_c = a + b
        ref_d = ref_c - a
        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(d, ref_d, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "target",
        [
            "ascendc",
            pytest.param(
                "pto",
                marks=pytest.mark.xfail(
                    reason="PTO codegen does not support sync intrinsics (PipeBarrier/"
                    "SetFlag/WaitFlag) inserted by ascend_sync_insert_vs pass on "
                    "A2/A3. This is a PTO backend limitation, not a VS pass issue.",
                    strict=True,
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("dtype", ["float16", "float"])
    def test_abs_elementwise(self, target, dtype):
        """E2E: MTE2->V->MTE3 elementwise kernel should produce correct results."""
        M, N = 256, 256
        func = _e2e_elementwise(M, N, dtype=dtype)
        kernel = tilelang.compile(func, out_idx=[1], pass_configs=PASS_CONFIGS_E2E, target=target)

        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        a = torch.randn(M, N, dtype=torch_dtype).npu()
        torch.npu.synchronize()

        b = kernel(a)
        ref_b = torch.abs(a)
        torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "target",
        [
            "ascendc",
            pytest.param(
                "pto",
                marks=pytest.mark.xfail(
                    reason="PTO codegen does not support sync intrinsics (PipeBarrier/"
                    "SetFlag/WaitFlag) inserted by ascend_sync_insert_vs pass on "
                    "A2/A3. This is a PTO backend limitation, not a VS pass issue.",
                    strict=True,
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("dtype", ["float16", "float"])
    def test_vadd_two_inputs(self, target, dtype):
        """E2E: MTE2(x2)->V->MTE3 vadd kernel should produce correct results."""
        M, N = 256, 256
        func = _e2e_vadd(M, N, dtype=dtype)
        kernel = tilelang.compile(func, out_idx=[2], pass_configs=PASS_CONFIGS_E2E, target=target)

        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        a = torch.randn(M, N, dtype=torch_dtype).npu()
        b = torch.randn(M, N, dtype=torch_dtype).npu()
        torch.npu.synchronize()

        c = kernel(a, b)
        ref_c = a + b
        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("target", ["ascendc"])
    def test_vs_pass_alone_no_crash(self, target):
        """E2E: VS pass + AscendSyncInsert co-enabled should not crash.

        AscendSyncInsert handles MTE2->V / V->MTE3 intra-core syncs; VS pass
        complements it by handling V->V and S<->others. The two passes are
        complementary (not mutually exclusive) and run sequentially in the
        pipeline. For kernels with cross-pipeline dependencies, AscendSyncInsert
        must be co-enabled.
        """
        configs = {
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        }
        M, N = 256, 256
        func = _e2e_elementwise(M, N, dtype="float16")
        kernel = tilelang.compile(func, out_idx=[1], pass_configs=configs, target=target)

        a = torch.randn(M, N, dtype=torch.float16).npu()
        torch.npu.synchronize()

        b = kernel(a)
        ref_b = torch.abs(a)
        torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)
