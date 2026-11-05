"""Retained source device-region annotator stays separate from live pipelines."""

import pytest
from tilelang import tvm


tir = tvm.tirx


def _device_case(key="thread_extent", tag="cthread", kind="single"):
    variable = tir.Var("vid", "int32")
    thread = tir.IterVar(tvm.ir.Range(0, 2), variable, tir.IterVar.ThreadIndex, tag)
    leaf = tir.Evaluate(1)
    node = thread if key == "thread_extent" else tir.const(0, "int32")
    body = tir.AttrStmt(node, key, 2, leaf)
    if kind == "siblings":
        body = tir.SeqStmt([body, tir.AttrStmt(tir.const(0, "int32"), "device_scope", 1, tir.Evaluate(2))])
    elif kind == "outer":
        body = tir.AttrStmt(tir.const(0, "int32"), "user_marker", 7, body)
    elif kind == "nested":
        body = tir.AttrStmt(thread, "thread_extent", 2, body)
    elif kind == "existing":
        body = tir.AttrStmt(tvm.target.Target("llvm"), "target", 0, body)
    return tir.PrimFunc([], body).with_attr("global_symbol", "device_case")


def _target(kind="ascend", host=True):
    attrs = {"kind": kind}
    if kind == "ascend":
        attrs.update(arch="dav-2201", model="A3", keys=["ascend"])
    return tvm.target.Target(attrs, host="llvm" if host else None)


def _run(func, target=None):
    if target is not None:
        func = func.with_attr("target", target)
    factory = tvm.ffi.get_global_func("tl.transform.A2A3AnnotateDeviceRegions")
    return factory()(tvm.IRModule({"main": func}))["main"]


def _regions(func):
    result = []
    tir.stmt_functor.post_order_visit(
        func.body, lambda node: result.append(node) if isinstance(node, tir.AttrStmt) and node.attr_key == "target" else None
    )
    return result


@pytest.mark.parametrize("key", ["thread_extent", "pipeline_exec_scope", "device_scope"])
@pytest.mark.parametrize("tag", ["cthread", "blockIdx.x", "threadIdx.x", "blockIdx.y"])
def test_source_device_only_attribute_contract(key, tag):
    before = _device_case(key, tag)
    target = _target()
    after = _run(before, target)
    assert len(_regions(after)) == 1
    region = _regions(after)[0]
    assert region.node.host is None
    assert region.node.kind.name == "ascend"
    tvm.ir.assert_structural_equal(region.body, before.body)
    assert int(region.value) == 0
    assert after.attrs["target"].host is not None


@pytest.mark.parametrize("key", ["thread_extent", "pipeline_exec_scope", "device_scope", "user_marker"])
def test_hostless_function_is_unchanged(key):
    before = _device_case(key)
    after = _run(before, _target(host=False))
    tvm.ir.assert_structural_equal(before.body, after.body)
    assert not _regions(after)


def test_unrelated_attribute_does_not_mark_device_code():
    before = _device_case("user_marker")
    after = _run(before, _target())
    tvm.ir.assert_structural_equal(before.body, after.body)
    assert not _regions(after)


def test_existing_target_region_is_preserved_and_not_revisited():
    before = _device_case(kind="existing")
    after = _run(before, _target())
    tvm.ir.assert_structural_equal(before.body, after.body)
    assert len(_regions(after)) == 1
    assert _regions(after)[0].node.kind.name == "llvm"


def test_generic_outer_attributes_recurse():
    after = _run(_device_case(kind="outer"), _target())
    assert after.body.attr_key == "user_marker"
    assert int(after.body.value) == 7
    assert len(_regions(after)) == 1
    assert after.body.body.attr_key == "target"


def test_first_device_attribute_owns_nested_region():
    after = _run(_device_case(kind="nested"), _target())
    assert len(_regions(after)) == 1
    assert after.body.attr_key == "target"
    assert after.body.body.attr_key == "thread_extent"
    assert after.body.body.body.attr_key == "thread_extent"


def test_sibling_device_regions_are_independent():
    after = _run(_device_case(kind="siblings"), _target())
    assert len(_regions(after)) == 2
    assert all(item.attr_key == "target" for item in after.body.seq)


def test_c_device_with_host_marks_entire_body():
    before = _device_case("user_marker")
    after = _run(before, _target("c"))
    assert len(_regions(after)) == 1
    assert after.body.node.kind.name == "c"
    assert after.body.node.host is None
    tvm.ir.assert_structural_equal(after.body.body, before.body)


def test_c_device_without_host_keeps_source_behavior():
    before = _device_case("user_marker")
    after = _run(before, _target("c", host=False))
    assert not _regions(after)
    tvm.ir.assert_structural_equal(after.body, before.body)


def test_target_attribute_is_required():
    with pytest.raises(Exception, match="Require the target"):
        _run(_device_case())
