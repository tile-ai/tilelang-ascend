# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Reject conflicting cross-core synchronization before device execution."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize
from tvm import tir


AUTO_SYNC = tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC
COMBINE = tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE


def _combine(body, config):
    func = tir.PrimFunc([], body)
    with tvm.transform.PassContext(config=config):
        return tilelang.transform.CombineCV()(tvm.IRModule.from_expr(func))["main"]


def test_auto_sync_requires_combine():
    config = {AUTO_SYNC: True, COMBINE: False}
    with pytest.raises(tvm.error.InternalError, match="AUTO_CV_SYNC=True requires TL_ASCEND_AUTO_CV_COMBINE=True"):
        _combine(tir.Evaluate(0), config)


def _manual_flag(kind):
    return tir.Evaluate(T.set_cross_flag("FIX", 0) if kind == "set" else T.wait_cross_flag(0))


@pytest.mark.parametrize("kind", ["set", "wait"])
@pytest.mark.parametrize("nested", [False, True])
def test_auto_sync_rejects_manual_flags(kind, nested):
    body = _manual_flag(kind)
    if nested:
        index = tir.Var("i", "int32")
        body = tir.For(index, 0, 2, tir.ForKind.SERIAL, tir.IfThenElse(index > 0, body, None))
        body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, body)
    with pytest.raises(tvm.error.InternalError, match="cannot be combined with manual T.set_cross_flag/T.wait_cross_flag"):
        _combine(body, {COMBINE: True, AUTO_SYNC: True})


@pytest.mark.parametrize("auto_sync", [None, False])
@pytest.mark.parametrize("combine", [False, True])
def test_manual_flags_allowed_without_auto_sync(auto_sync, combine):
    config = {COMBINE: combine}
    if auto_sync is not None:
        config[AUTO_SYNC] = auto_sync
    body = tir.SeqStmt(
        [
            tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, _manual_flag("set")),
            tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, _manual_flag("wait")),
        ]
    )
    result = _combine(body, config)
    tvm.ir.assert_structural_equal(result.body, body)


def test_auto_sync_allows_scopes_and_intra_core_sync():
    body = tir.SeqStmt(
        [
            tir.Evaluate(T.set_flag("mte2", "v", 0)),
            tir.Evaluate(T.wait_flag("mte2", "v", 0)),
            tir.Evaluate(T.barrier_all()),
        ]
    )
    body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, body)
    result = _combine(body, {COMBINE: True, AUTO_SYNC: True})
    tvm.ir.assert_structural_equal(result.body, body)


def test_auto_sync_without_manual_flags():
    body = tir.Evaluate(0)
    result = _combine(body, {COMBINE: True, AUTO_SYNC: True})
    tvm.ir.assert_structural_equal(result.body, body)


@pytest.mark.parametrize("combine", [None, True])
def test_auto_sync_inserts_paired_workspace_flags(combine):
    # Same L0C -> GM workspace -> UB handoff as the existing C/V copy tests,
    # lowered through the real DSL rather than constructing internal flags.
    @T.prim_func
    def main(
        a: T.Tensor((16, 16), "float16"),
        b: T.Tensor((16, 16), "float16"),
        workspace: T.Tensor((16, 16), "float32"),
        output: T.Tensor((16, 16), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            a_l1 = T.alloc_L1((16, 16), "float16")
            b_l1 = T.alloc_L1((16, 16), "float16")
            acc = T.alloc_L0C((16, 16), "float32")
            value = T.alloc_ub((8, 16), "float32")
            T.copy(a, a_l1)
            T.copy(b, b_l1)
            T.gemm_v0(a_l1, b_l1, acc, init=True)
            T.copy(acc, workspace)
            T.copy(workspace[vid * 8 : (vid + 1) * 8, :], value)
            T.copy(value, output[vid * 8 : (vid + 1) * 8, :])

    def collect_flags(mod):
        sets, waits = [], []

        def visit(node):
            if not isinstance(node, tir.Call):
                return
            if node.op.same_as(tvm.ir.Op.get("tl.ascend_auto_set_cross_flag")):
                sets.append(node)
            elif node.op.same_as(tvm.ir.Op.get("tl.ascend_auto_wait_cross_flag")):
                waits.append(node)

        tir.stmt_functor.post_order_visit(mod["main"].body, visit)
        return sets, waits

    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    for enabled in (False, True):
        # Rebuild the module for each run: passes may mutate their inputs.
        mod = tvm.IRModule.from_expr(main.with_attr("npu_platform", "A2"))
        config = {AUTO_SYNC: enabled}
        if combine is not None:
            config[COMBINE] = combine
        with tvm.transform.PassContext(config=config):
            lowered = LowerAndLegalize(mod, target)
            assert collect_flags(lowered) == ([], [])
            combined = tilelang.transform.CombineCV()(lowered)
        sets, waits = collect_flags(combined)
        if not enabled:
            assert not sets and not waits
            continue
        assert len(sets) == len(waits) == 1
        assert int(sets[0].args[0]) == 2
        assert sets[0].args[1].value == "FIX"
        assert waits[0].args[1].value == "MTE2"
        tvm.ir.assert_structural_equal(sets[0].args[2], waits[0].args[0])


@pytest.mark.parametrize("combine", [False, True])
def test_lowering_rejects_invalid_auto_sync(combine):
    # Exercise the public DSL and real pass ordering, without launching an
    # invalid synchronization protocol on a device.
    @T.prim_func
    def main(output: T.Tensor((64,), "float32")):
        with T.Kernel(1, is_npu=True):
            value = T.alloc_ub((64,), "float32")
            T.tile.fill(value, 1.0)
            T.set_cross_flag("V", 0)
            T.wait_cross_flag(0)
            T.copy(value, output)

    message = "cannot be combined with manual" if combine else "requires TL_ASCEND_AUTO_CV_COMBINE=True"
    with tvm.transform.PassContext(config={COMBINE: combine, AUTO_SYNC: True}), pytest.raises(tvm.error.InternalError, match=message):
        tilelang.lower(main, target="ascendc", platform="A2")
