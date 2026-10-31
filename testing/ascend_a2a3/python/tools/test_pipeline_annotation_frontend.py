"""TA's complete Pipelined metadata on the neutral loop frame."""

import inspect

import pytest
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script.builder import ir as B
from tvm_ffi import Array

from tilelang import tvm
import tilelang.ascend.a2a3.language as T
from tilelang.language.loop import Pipelined as shared_pipelined


def _loop(start=7, stop=None, **kwargs):
    with IRBuilder() as builder, B.prim_func():
        B.func_name("pipeline")
        with T.Pipelined(start, stop, **kwargs) as index:
            B.evaluate(index)
    loops = []
    tvm.tirx.stmt_functor.post_order_visit(builder.get().body, lambda node: loops.append(node) if isinstance(node, tvm.tirx.For) else None)
    assert len(loops) == 1
    return loops[0]


@pytest.mark.parametrize("start,stop,minimum,extent", [(7, None, 0, 7), (3, 11, 3, 8)])
def test_source_loop_bounds_and_default_cross_interval(start, stop, minimum, extent):
    loop = _loop(start, stop)
    assert int(loop.min) == minimum and int(loop.extent) == extent
    assert loop.kind == tvm.tirx.ForKind.SERIAL
    assert set(loop.annotations) == {"tl_cross_interval"}
    assert isinstance(loop.annotations["tl_cross_interval"], tvm.tirx.IntImm)
    assert loop.annotations["tl_cross_interval"].dtype == "int32"
    assert int(loop.annotations["tl_cross_interval"]) == 1


def test_source_dtype_of_single_stop_is_preserved():
    loop = _loop(tvm.tirx.IntImm("int64", 7))
    assert all(str(value.dtype) == "int64" for value in [loop.min, loop.extent, loop.loop_var])


@pytest.mark.parametrize("stages", [-1, 0, 1, 2, 4])
def test_original_positive_stage_gate(stages):
    loop = _loop(num_stages=stages)
    assert ("num_stages" in loop.annotations) == (stages > 0)
    if stages > 0:
        assert int(loop.annotations["num_stages"]) == stages


@pytest.mark.parametrize("interval", [-1, 0, 1, 2, 3, 4])
def test_original_positive_cross_interval_gate(interval):
    loop = _loop(cross_interval=interval)
    assert ("tl_cross_interval" in loop.annotations) == (interval > 0)
    if interval > 0:
        assert int(loop.annotations["tl_cross_interval"]) == interval


def test_complete_original_positional_signature_and_typed_nested_arrays():
    sync = [[0, 1], [1, 0]]
    group = [[0, 1], [2, 3]]
    with IRBuilder() as builder, B.prim_func():
        B.func_name("pipeline")
        with T.Pipelined(3, 11, 2, [1, 0], [0, 1], sync, group, 3) as index:
            B.evaluate(index)
    loops = []
    tvm.tirx.stmt_functor.post_order_visit(builder.get().body, lambda node: loops.append(node) if isinstance(node, tvm.tirx.For) else None)
    annotations = loops[0].annotations
    assert [int(value) for value in annotations["tl_pipeline_order"]] == [1, 0]
    assert [int(value) for value in annotations["tl_pipeline_stage"]] == [0, 1]
    for key, expected in [("tl_pipeline_sync", sync), ("tl_pipeline_group", group)]:
        actual = annotations[key]
        assert isinstance(actual, Array) and all(isinstance(row, Array) for row in actual)
        assert [[int(value) for value in row] for row in actual] == expected
        assert all(isinstance(value, tvm.tirx.IntImm) and value.dtype == "int32" for row in actual for value in row)
    assert sync == [[0, 1], [1, 0]] and group == [[0, 1], [2, 3]]
    assert int(annotations["num_stages"]) == 2 and int(annotations["tl_cross_interval"]) == 3


@pytest.mark.parametrize("groups", [None, []])
def test_source_empty_sync_and_group_are_omitted(groups):
    assert set(_loop(sync=groups, group=groups).annotations) == {"tl_cross_interval"}


@pytest.mark.parametrize("kwargs", [{"cross_interval": 1.5}, {"sync": [["invalid"]]}, {"group": [1, 2]}])
def test_source_invalid_argument_shapes_remain_frontend_errors(kwargs):
    with pytest.raises(TypeError):
        _loop(**kwargs)


def test_dialect_override_preserves_shared_signature():
    assert "cross_interval" in inspect.signature(T.Pipelined).parameters
    assert "annotations" not in inspect.signature(T.Pipelined).parameters
    assert "annotations" in inspect.signature(shared_pipelined).parameters
    assert "cross_interval" not in inspect.signature(shared_pipelined).parameters


def test_eager_nested_macros_keep_original_pipeline_metadata():
    @T.macro
    def inner(buffer):
        for index in T.Pipelined(4, num_stages=2, sync=[[0, 1]], group=[[0, 1]], cross_interval=2):
            buffer[index] = buffer[index] + 1

    @T.macro
    def outer(buffer):
        inner(buffer)

    @T.prim_func
    def kernel(Buffer: T.Tensor((4,), "float32")):
        outer(Buffer)

    loops = []
    tvm.tirx.stmt_functor.post_order_visit(kernel.body, lambda node: loops.append(node) if isinstance(node, tvm.tirx.For) else None)
    assert len(loops) == 1
    assert int(loops[0].annotations["tl_cross_interval"]) == 2
    assert int(loops[0].annotations["tl_pipeline_sync"][0][1]) == 1
