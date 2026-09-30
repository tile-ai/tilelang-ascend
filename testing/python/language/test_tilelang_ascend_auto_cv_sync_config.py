# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Reject conflicting cross-core synchronization before device execution."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tvm import tir


AUTO_SYNC = tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC
COMBINE = tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE


def _combine(body, config):
    func = tir.PrimFunc([], body)
    with tvm.transform.PassContext(config=config):
        return tilelang.transform.CombineCV()(tvm.IRModule.from_expr(func))["main"]


@pytest.mark.parametrize("combine", [None, False])
def test_auto_sync_requires_combine(combine):
    config = {AUTO_SYNC: True}
    if combine is not None:
        config[COMBINE] = combine
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
    body = tir.SeqStmt([_manual_flag("set"), _manual_flag("wait")])
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


def test_auto_sync_does_not_reject_generated_flags():
    body = tir.SeqStmt(
        [
            tir.Evaluate(tir.call_intrin("handle", tvm.ir.Op.get("tl.ascend_auto_set_cross_flag"), 2, "FIX", 0)),
            tir.Evaluate(tir.call_intrin("handle", tvm.ir.Op.get("tl.ascend_auto_wait_cross_flag"), 0, "MTE2")),
        ]
    )
    result = _combine(body, {COMBINE: True, AUTO_SYNC: True})
    tvm.ir.assert_structural_equal(result.body, body)


def test_auto_sync_without_manual_flags():
    body = tir.Evaluate(0)
    result = _combine(body, {COMBINE: True, AUTO_SYNC: True})
    tvm.ir.assert_structural_equal(result.body, body)


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
