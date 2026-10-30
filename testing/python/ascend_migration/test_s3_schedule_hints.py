"""Task/Stage validation and real consumption, without compiling device binaries."""

import pytest

import tilelang
from tilelang import language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize

from testing.python.ascend_migration.test_s2_task_ir import (
    _add,
    _buffer,
    _dma,
    _fill,
    _function,
)

tir = tvm.tir
TARGET = tvm.target.Target("c -keys=ascend -model=ascendc")


def make_task_stage_vector_kernel(*, marked=True, stage=0):
    """An existing copy/add/copy shape, with only its add explicitly grouped."""
    if marked:

        @T.prim_func
        def main(
            A: T.Tensor((64,), "float32"),
            B: T.Tensor((64,), "float32"),
            O: T.Tensor((64,), "float32"),
        ):
            with T.Kernel(1, threads=1, is_npu=True):
                a = T.alloc_ub((64,), "float32")
                b = T.alloc_ub((64,), "float32")
                c = T.alloc_ub((64,), "float32")
                with T.Scope("V"):
                    T.copy(A, a)
                    T.copy(B, b)
                    with T.Stage(stage):  # noqa: SIM117 - exercise both lexical marker boundaries
                        with T.Task():
                            T.tile.add(c, a, b)
                    T.copy(c, O)

    else:

        @T.prim_func
        def main(
            A: T.Tensor((64,), "float32"),
            B: T.Tensor((64,), "float32"),
            O: T.Tensor((64,), "float32"),
        ):
            with T.Kernel(1, threads=1, is_npu=True):
                a = T.alloc_ub((64,), "float32")
                b = T.alloc_ub((64,), "float32")
                c = T.alloc_ub((64,), "float32")
                with T.Scope("V"):
                    T.copy(A, a)
                    T.copy(B, b)
                    T.tile.add(c, a, b)
                    T.copy(c, O)

    return main


def _task(body, metadata=None, value=None):
    return tir.AttrStmt(
        {} if metadata is None else metadata,
        "tl.ascend_task",
        tir.const(1, "int32") if value is None else value,
        body,
    )


def _stage(body, stage=None):
    return tir.AttrStmt(
        tir.const(0, "int64") if stage is None else stage,
        "tl.ascend_stage",
        tir.const(1, "int32"),
        body,
    )


def _module(body=None, *, scope=1):
    a = _buffer("a")
    return tvm.IRModule({"main": _function([a], _task(_fill(a)) if body is None else body(a), scope=scope)})


def _consume(module, target=TARGET, platform="A3"):
    return tilelang.transform.AscendLowerScheduleHints(target, platform)(module)


def _attributes(function):
    nodes = []
    tir.stmt_functor.post_order_visit(function.body, lambda node: nodes.append(node) if isinstance(node, tir.AttrStmt) else None)
    return nodes


def test_task_stage_builders():
    function = make_task_stage_vector_kernel()
    markers = {node.attr_key: node for node in _attributes(function) if "tl.ascend_" in node.attr_key}
    assert set(markers) == {"tl.ascend_task", "tl.ascend_stage"}
    assert len(markers["tl.ascend_task"].node) == 0
    assert markers["tl.ascend_stage"].node.dtype == "int64"
    assert int(markers["tl.ascend_stage"].node) == 0
    assert not hasattr(T, "PerCoreTask")
    for value in (True, False, -1, 1 << 31, 0.0, "0", tir.Var("stage", "int32")):
        with pytest.raises(ValueError, match="nonnegative constant signed int32"):
            T.Stage(value)
    positive = make_task_stage_vector_kernel(stage=1)
    assert any(node.attr_key == "tl.ascend_stage" and int(node.node) == 1 for node in _attributes(positive))
    with pytest.raises(tvm.TVMError, match="constraint mode"):
        _consume(tvm.IRModule.from_expr(positive))
    for kwargs in ({"latency": 0}, {"ii": 1}, {"latency": "bad"}):
        with pytest.raises(ValueError, match="latency/ii overrides"):
            T.Task(**kwargs)


def _invalid_body(case, a):
    fill = _fill(a)
    i = tir.Var("i", "int32")
    if case == "multiple_calls":
        return _task(tir.SeqStmt([fill, fill]))
    if case == "nested_task":
        return _task(_task(fill))
    if case == "task_loop":
        return _task(tir.For(i, 0, 1, tir.ForKind.SERIAL, fill))
    if case == "task_if":
        return _task(tir.IfThenElse(tir.const(True), fill, None))
    if case == "parallel_task":
        return tir.For(i, 0, 1, tir.ForKind.PARALLEL, _task(fill))
    if case == "nested_stage":
        return _stage(_stage(_task(fill)))
    if case == "stage_in_task":
        return _task(_stage(fill))
    if case == "bare_stage":
        return _stage(fill)
    if case == "nonzero_stage":
        return _stage(_task(fill), tir.const(1, "int64"))
    if case == "stage_wrong_dtype":
        return _stage(_task(fill), tir.const(0, "int32"))
    if case == "task_metadata":
        return _task(fill, {"latency": tir.const(1, "int64")})
    if case == "task_bad_metadata":
        return _task(fill, tir.const(0))
    if case == "marker_bad_value":
        return _task(fill, value=tir.const(1, "int64"))
    if case == "percore":
        return tir.AttrStmt({}, "tl.ascend_per_core_task", 1, fill)
    if case in {"num_stages", "enable_offset", "legacy_stage", "legacy_order"}:
        annotations = {
            "num_stages": {"num_stages": 2},
            "enable_offset": {"enable_offset": True},
            "legacy_stage": {"tl_pipeline_stage": [0]},
            "legacy_order": {"software_pipeline_order": [0]},
        }[case]
        return tir.For(i, 0, 2, tir.ForKind.SERIAL, _stage(_task(fill)), annotations=annotations)
    if case == "manual_state":
        return _task(tir.Evaluate(tir.call_extern("handle", "set_flag", 0, 1, 0)))
    if case == "compound":
        return _task(tir.Evaluate(tir.call_extern("handle", "gemm", a.access_ptr("w"))))
    raise ValueError(case)


@pytest.mark.parametrize(
    "case",
    [
        "multiple_calls",
        "nested_task",
        "task_loop",
        "task_if",
        "parallel_task",
        "nested_stage",
        "stage_in_task",
        "bare_stage",
        "nonzero_stage",
        "stage_wrong_dtype",
        "task_metadata",
        "task_bad_metadata",
        "marker_bad_value",
        "percore",
        "num_stages",
        "enable_offset",
        "legacy_stage",
        "legacy_order",
        "manual_state",
        "compound",
    ],
)
def test_task_stage_rejections(case):
    module = _module(lambda a: _invalid_body(case, a))
    original = tvm.ir.save_json(module)
    with pytest.raises(tvm.TVMError):
        _consume(module)
    assert tvm.ir.save_json(module) == original
    with pytest.raises(tvm.TVMError):
        tilelang.transform.AnalyzeAscendSchedule(module["main"], "A3")
    if case not in {"manual_state", "compound"}:
        # This early pass runs before Simplify/Parallel lowering erase structure.
        with pytest.raises(tvm.TVMError):
            LowerAndLegalize(module, TARGET)


def test_analysis_consumes_task_stage_markers():
    module = LowerAndLegalize(tvm.IRModule({"main": make_task_stage_vector_kernel()}), TARGET)
    original = tvm.ir.save_json(module)
    graph = tilelang.transform.AnalyzeAscendSchedule(module["main"], "A3")
    assert tvm.ir.save_json(module) == original
    assert len(graph["tasks"]) == 4
    explicit = [group for group in graph["groups"] if "explicit_task" in group]
    assert len(explicit) == 1
    assert bool(explicit[0]["explicit_task"]) and int(explicit[0]["stage"]) == 0
    assert len(explicit[0]["task_ids"]) == 1
    assert graph["tasks"][int(explicit[0]["task_ids"][0])]["pipe"] == "PIPE_V"
    assert isinstance(explicit[0]["statement"], tir.AttrStmt)
    consumed = _consume(module)
    assert "tl.ascend_schedule_hints_pending" not in consumed["main"].attrs
    assert all(node.attr_key not in {"tl.ascend_task", "tl.ascend_stage"} for node in _attributes(consumed["main"]))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("operation", ["fill", "add", "load", "store"])
def test_single_audited_call_preserves_group_and_pipe(dtype, operation):
    a, b = [_buffer(name, dtype=dtype) for name in "ab"]
    gm = _buffer("gm", "global", dtype)
    statement = {
        "fill": lambda: _fill(a),
        "add": lambda: _add(a, b, b),
        "load": lambda: _dma(gm, a),
        "store": lambda: _dma(a, gm),
    }[operation]()
    module = tvm.IRModule({"main": _function([a, b, gm], _stage(_task(statement)))})
    graph = tilelang.transform.AnalyzeAscendSchedule(module["main"], "A3")
    group = next(group for group in graph["groups"] if "explicit_task" in group)
    assert len(group["task_ids"]) == 1 and int(group["stage"]) == 0
    expected = {"fill": "PIPE_V", "add": "PIPE_V", "load": "PIPE_MTE2", "store": "PIPE_MTE3"}
    assert graph["tasks"][0]["pipe"] == expected[operation]
    consumed = _consume(module)
    plain = tvm.IRModule({"main": _function([a, b, gm], statement)})
    tvm.ir.assert_structural_equal(consumed, plain)


@pytest.mark.parametrize("target", ["ascendc", "auto"])
@pytest.mark.parametrize("automatic_sync", [False, True])
def test_lower_consumes_markers_without_behavior_change(target, automatic_sync):
    with tvm.transform.PassContext(config={"tl.ascend_auto_sync": automatic_sync, "tl.ascend_auto_sync_vs": False}):
        marked = tilelang.lower(make_task_stage_vector_kernel(), target=target, platform="A3")
        plain = tilelang.lower(make_task_stage_vector_kernel(marked=False), target=target, platform="A3")
    assert marked.kernel_source == plain.kernel_source
    assert "tl.ascend_task" not in marked.kernel_source
    assert "tl.ascend_stage" not in marked.kernel_source


@pytest.mark.parametrize("model,platform", [("pto", "A3"), ("ascendc", "A5"), ("ascendc", "A2")])
def test_marked_target_gate_preserves_hintless_identity(model, platform):
    target = tvm.target.Target(f"c -keys=ascend -model={model}")
    with pytest.raises(tvm.TVMError, match="only for A3 AscendC/auto"):
        _consume(_module(), target, platform)
    plain = _module(lambda a: _fill(a))
    result = _consume(plain, target, platform)
    assert result["main"].same_as(plain["main"])
    assert tilelang.transform.AscendValidateScheduleHints()(plain)["main"].same_as(plain["main"])
    if platform == "A2":
        with pytest.raises(tvm.TVMError, match="only for A3"):
            tilelang.transform.AnalyzeAscendSchedule(_module()["main"], "A2")


def test_task_validation_ignores_unmarked_siblings_and_preserves_outer_environment():
    module = _module(
        lambda a: tir.SeqStmt(
            [
                tir.Evaluate(tir.call_extern("handle", "unrelated_raw_or_cube_helper")),
                _stage(_task(_fill(a))),
            ]
        )
    )
    result = _consume(module)
    assert "unrelated_raw_or_cube_helper" in result.script()
    assert "tl.ascend_task" not in result.script()
    assert "tl.ascend_stage" not in result.script()
    # A Task alone retains the established loop pipeline contract.
    i = tir.Var("i", "int32")
    pipeline = _module(lambda a: tir.For(i, 0, 2, tir.ForKind.SERIAL, _task(_fill(a)), annotations={"num_stages": 2}))
    assert "tl.ascend_task" not in _consume(pipeline).script()
    with pytest.raises(tvm.TVMError, match="explicit V owner"):
        _consume(_module(scope=0))


def test_enclosing_guard_is_preserved_and_analyzed():
    condition = tir.Var("condition", "int32")
    a = _buffer("a")
    task = _task(_fill(a))
    body = tir.IfThenElse(condition > 0, task, task)
    function = _function([a], body, scalars=[condition])
    module = tvm.IRModule({"main": function})
    result = _consume(module)
    plain = _function([a], tir.IfThenElse(condition > 0, _fill(a), _fill(a)), scalars=[condition])
    tvm.ir.assert_structural_equal(result["main"], plain)
    graph = tilelang.transform.AnalyzeAscendSchedule(function, "A3")
    assert len([group for group in graph["groups"] if "explicit_task" in group]) == 2
    assert all(not bool(group["schedulable"]) for group in graph["groups"])
    assert len(graph["tasks"]) == 2
    assert not tvm.ir.structural_equal(graph["tasks"][0]["guard"], graph["tasks"][1]["guard"])


@pytest.mark.parametrize("dead_region", ["false_guard", "zero_trip"])
@pytest.mark.parametrize("model,platform", [("pto", "A3"), ("ascendc", "A5")])
def test_dead_markers_still_observe_backend_gate_and_clear_pending(dead_region, model, platform):
    def body(a):
        task = _task(_fill(a))
        if dead_region == "false_guard":
            return tir.IfThenElse(tir.const(False), task, None)
        return tir.For(tir.Var("i", "int32"), 0, 0, tir.ForKind.SERIAL, task)

    module = tilelang.transform.AscendValidateScheduleHints()(_module(body))
    assert bool(module["main"].attrs["tl.ascend_schedule_hints_pending"])
    # This explicitly exercises a later eliminating rewrite, rather than
    # claiming the current LowerAndLegalize necessarily erases these regions.
    module = tir.transform.Simplify()(module)
    module = tir.transform.RemoveNoOp()(module)
    assert all(node.attr_key != "tl.ascend_task" for node in _attributes(module["main"]))
    target = tvm.target.Target(f"c -keys=ascend -model={model}")
    with pytest.raises(tvm.TVMError, match="only for A3 AscendC/auto"):
        _consume(module, target, platform)
    result = _consume(module)
    assert "tl.ascend_schedule_hints_pending" not in result["main"].attrs
