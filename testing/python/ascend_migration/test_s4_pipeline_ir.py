"""Single-stage constraint bridge contracts; no Bisheng or device execution."""

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys

import pytest

import tilelang
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize
from test_s4_kernels import make_vector_kernel, vector_pass_configs

tir = tvm.tir
TASK = "tl.ascend_constraint_task_id"
ITER = "tl.ascend_constraint_iteration"
MANIFEST = "tl.ascend_constraint_plan"


def _before_planning(function):
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    module = tvm.IRModule.from_expr(function.with_attr("npu_platform", "A3"))
    module = LowerAndLegalize(module, target)
    module = tir.transform.PlanAndUpdateBufferAllocationLocation()(module)
    module = tilelang.transform.CrossCorePipeline()(module)
    module = tilelang.transform.CombineCV()(module)
    return tir.transform.PlanAndUpdateBufferAllocationLocation()(module)


def _plan(module, configs=None):
    with tvm.transform.PassContext(config=configs or vector_pass_configs()):
        return tilelang.transform.PipelinePlanning()(module)


def _loops(function):
    found = []
    tir.stmt_functor.post_order_visit(function.body, lambda node: found.append(node) if isinstance(node, tir.For) else None)
    return found


def _fills(
    *,
    stages=1,
    minimum=0,
    extent=3,
    dependent=False,
    dtype="int32",
    annotations=None,
    allocation_shape=(64,),
):
    buffers = [tir.decl_buffer(allocation_shape, "float32", name=name, scope="shared.ub") for name in "abc"]
    statements = []
    for index, buffer in enumerate(buffers):
        pointer = buffers[0] if dependent else buffer
        statements.append(
            tir.Evaluate(
                tir.call_intrin(
                    "handle",
                    tvm.ir.Op.get("tl.ascend_fill"),
                    tir.StringImm("Fill<float>"),
                    pointer.access_ptr("w"),
                    tir.const(index + 1.0, "float32"),
                    64,
                )
            )
        )
    block = tir.Block([], [], [], "units", tir.SeqStmt(statements), alloc_buffers=buffers)
    variable = tir.Var("iteration", dtype)
    attributes = {} if stages is None else {"num_stages": stages}
    attributes.update(annotations or {})
    loop = tir.For(
        variable,
        tir.IntImm(dtype, minimum),
        tir.IntImm(dtype, extent) if isinstance(extent, int) else extent,
        tir.ForKind.SERIAL,
        tir.BlockRealize([], True, block),
        annotations=attributes,
    )
    function = tir.PrimFunc([], tir.AttrStmt(tir.const(0), "resource_scope", 1, loop))
    return tvm.IRModule.from_expr(
        function.with_attr("target", tvm.target.Target({"kind": "llvm", "model": "ascendc"}))
        .with_attr("npu_platform", "A3")
        .with_attr("global_symbol", "main")
    )


@contextmanager
def _solver(callback):
    previous = tvm.get_global_func("tl.ascend.solve_schedule", allow_missing=True)
    tvm.register_func("tl.ascend.solve_schedule", callback, override=True)
    try:
        yield
    finally:
        if previous is None:
            tvm._ffi.registry.remove_global_func("tl.ascend.solve_schedule")
        else:
            tvm.register_func("tl.ascend.solve_schedule", previous, override=True)


def _sat(order):
    return tvm.runtime.convert(
        {
            "schema_version": 1,
            "status": "SAT",
            "reason": "",
            "order": order,
            "start_times": [order.index(i) for i in range(len(order))],
            "ii": 0,
            "buffer_versions": [],
        }
    )


def test_inverse_order_is_written_in_injector_index_space():
    def callback(problem):
        assert str(problem["mode"]) == "straight"
        assert [int(value) for value in problem["latencies"]] == [1, 1, 1]
        assert not list(problem["data_deps"])
        return _sat([2, 0, 1])

    with _solver(callback):
        planned = _plan(_fills())
    loop = _loops(planned["main"])[0]
    assert [int(value) for value in loop.annotations["software_pipeline_order"]] == [1, 2, 0]
    assert [int(value) for value in loop.annotations["software_pipeline_stage"]] == [0, 0, 0]
    assert "num_stages" not in loop.annotations
    assert "software_pipeline_async_stages" not in loop.annotations
    manifest = planned["main"].attrs[MANIFEST]
    assert MANIFEST not in manifest["source_function"].attrs
    assert [int(value) for value in manifest["candidates"][0]["unit_ids"]] == [0, 1, 2]
    injected = tilelang.transform.InjectSoftwarePipeline()(planned)
    assert all(int(value) == 1 for value in injected["main"].attrs["buffer_versions"].values())


@pytest.mark.parametrize(
    "response,error",
    [
        ({"schema_version": 1, "status": "UNSAT", "reason": "test"}, "did not produce SAT"),
        ({"schema_version": 1, "status": "TIMEOUT", "reason": "test"}, "did not produce SAT"),
        ({"schema_version": 2, "status": "SAT", "order": [0, 1, 2]}, "solver schema"),
        ({"schema_version": 1, "status": "SAT", "order": [0, 1]}, "order size"),
        ({"schema_version": 1, "status": "SAT", "order": [0, 0, 2]}, "permutation"),
        ({"schema_version": 1, "status": "SAT", "order": [0, 1, 3]}, "permutation"),
        ({"schema_version": 1, "status": "SAT", "order": [False, 1, 2]}, "integer task IDs"),
    ],
)
def test_malformed_solver_results_cannot_emit_pipeline_abi(response, error):
    with _solver(lambda _: tvm.runtime.convert(response)), pytest.raises(tvm.TVMError, match=error):
        _plan(_fills())


def test_solver_order_cannot_reverse_original_waw():
    with (
        _solver(lambda _: _sat([2, 0, 1])),
        pytest.raises(tvm.TVMError, match="logical dependency"),
    ):
        _plan(_fills(dependent=True))


def test_unknown_strategy_is_rejected_by_wrapper_and_native_dispatch():
    config = vector_pass_configs("unknown")
    with tvm.transform.PassContext(config=config):
        with pytest.raises(ValueError, match="planner"):
            tilelang.transform.PipelinePlanning()
        with pytest.raises(tvm.TVMError, match="Unknown.*strategy"):
            tvm.get_global_func("tl.transform.PipelinePlanning")()(_fills())


@pytest.mark.parametrize("operation", ["opaque", "manual_sync"])
def test_whole_kernel_unknown_effects_are_rejected(operation):
    module = _fills()
    function = module["main"]
    statement = tir.Evaluate(
        tir.call_extern("handle", "unknown_raw_helper")
        if operation == "opaque"
        else tir.call_intrin("handle", tvm.ir.Op.get("tl.ascend_auto_set_flag"), tir.StringImm("MTE2_V"), 0)
    )
    function = function.with_body(tir.SeqStmt([function.body, statement]))
    with pytest.raises(tvm.TVMError, match="Unsupported schedule|explicit V owner"):
        _plan(tvm.IRModule.from_expr(function))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("total", [160, 192])
def test_real_lowered_static_loops_bridge_once_with_one_version(dtype, total):
    module = _before_planning(make_vector_kernel(total, dtype))
    planned = _plan(module)
    candidates = planned["main"].attrs[MANIFEST]["candidates"]
    assert candidates
    requested = [loop for loop in _loops(planned["main"]) if "software_pipeline_stage" in loop.annotations]
    assert len(requested) == len(candidates)
    assert all(all(int(stage) == 0 for stage in loop.annotations["software_pipeline_stage"]) for loop in requested)
    injected = tilelang.transform.InjectSoftwarePipeline()(planned)
    assert not any("software_pipeline_stage" in loop.annotations for loop in _loops(injected["main"]))
    if "buffer_versions" in injected["main"].attrs:
        assert all(int(version) == 1 for version in injected["main"].attrs["buffer_versions"].values())
    with pytest.raises(tvm.TVMError, match="already planned"):
        _plan(planned)


def test_markers_keep_original_order_and_follow_substitute_and_unroll():
    with _solver(lambda _: _sat([2, 0, 1])):
        planned = _plan(_fills(minimum=2))
    source = planned["main"].attrs[MANIFEST]["source_function"]
    source_ids = []
    tir.stmt_functor.post_order_visit(
        source.body,
        lambda node: source_ids.append(int(node.node)) if isinstance(node, tir.AttrStmt) and node.attr_key == TASK else None,
    )
    assert source_ids == [0, 1, 2]
    injected = tilelang.transform.InjectSoftwarePipeline()(planned)

    def unroll(node):
        return tir.For(node.loop_var, node.min, node.extent, tir.ForKind.UNROLLED, node.body)

    function = injected["main"]
    body = tir.stmt_functor.ir_transform(function.body, None, unroll, ["tir.For"])
    module = tir.transform.UnrollLoop()(tvm.IRModule.from_expr(function.with_body(body)))
    observed = []
    tir.stmt_functor.post_order_visit(
        module["main"].body,
        lambda node: (
            observed.append(int(tvm.arith.Analyzer().simplify(node.value)))
            if isinstance(node, tir.AttrStmt) and node.attr_key == ITER
            else None
        ),
    )
    assert observed == [2] * 3 + [3] * 3 + [4] * 3


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"stages": None}, "at least one"),
        ({"stages": 0}, "at least one"),
        ({"stages": 2}, "explicit calibrated cost profile"),
        ({"minimum": 2**31 - 1, "extent": 2}, "representable static"),
        ({"dtype": "int64", "minimum": 2**63 - 1, "extent": 1}, "representable static"),
        ({"dtype": "uint32"}, "signed int32/int64"),
        ({"extent": tir.Var("trips", "int32")}, "representable static"),
        ({"annotations": {"tl_pipeline_order": [0, 1, 2]}}, "manual pipeline"),
        ({"annotations": {"software_pipeline_async_stages": [0]}}, "manual pipeline"),
        ({"annotations": {"enable_offset": True}}, "explicit calibrated cost profile"),
    ],
)
def test_unsupported_loop_and_manual_controls_are_rejected(kwargs, error):
    with pytest.raises(tvm.TVMError, match=error):
        _plan(_fills(**kwargs))


@pytest.mark.parametrize(
    "key,value",
    [
        ("tl.ascend_auto_sync", False),
        ("tl.ascend_memory_planning", False),
        ("tl.ascend_auto_cross_core_sync", True),
        ("tl.ascend_auto_sync_vs", True),
    ],
)
def test_conflicting_configs_fail_explicitly(key, value):
    config = vector_pass_configs()
    config[key] = value
    with pytest.raises(tvm.TVMError, match="requires auto_sync"):
        _plan(_fills(), config)


@pytest.mark.parametrize("shape", [(16384,), (49152,), (700, 17), (2**32 + 1, 2**32 - 1, 2**62 - 64, 64)])
def test_ub_budget_is_rejected_before_invoking_the_old_allocator(shape):
    # (700,17) has a smaller raw span but Flatten2D's 32B row padding would
    # push the three independent allocations over budget as well.
    with pytest.raises(tvm.TVMError, match="A3 UB budget"):
        _plan(_fills(allocation_shape=shape))


def test_allocate_dimensions_are_checked_before_capacity_can_wrap():
    shape = (2**32 + 1, 2**32 - 1, 2**62 - 64, 64)
    function = _fills(allocation_shape=shape)["main"]
    buffers = function.body.body.body.block.alloc_buffers
    body = tir.stmt_functor.ir_transform(
        function.body,
        None,
        lambda block: tir.Block([], [], [], block.name_hint, block.body),
        ["tir.Block"],
    )
    for buffer in reversed(buffers):
        body = tir.Allocate(buffer.data, buffer.dtype, buffer.shape, tir.const(True, "bool"), body)
    with pytest.raises(tvm.TVMError, match="allocation extent/product.*A3 UB budget"):
        _plan(tvm.IRModule.from_expr(function.with_body(body)))


def test_explicit_placement_hint_is_preserved_by_constraint_planning():
    module = _fills()
    function = module["main"]
    data = function.body.body.body.block.alloc_buffers[0].data
    function = function.with_attr("address_map", {data: tir.IntImm("int64", 0)})
    planned = _plan(tvm.IRModule.from_expr(function))
    assert int(planned["main"].attrs["address_map"][data]) == 0
    invalid = function.with_attr("address_map", {data: tir.IntImm("int64", 196352)})
    with pytest.raises(tvm.TVMError, match="explicit address.*UB capacity"):
        _plan(tvm.IRModule.from_expr(invalid))


@pytest.mark.parametrize("platform,model,error", [("A5", "ascendc", "A3 only"), ("A3", "pto", "AscendC")])
def test_unsupported_backend_is_rejected(platform, model, error):
    module = _fills()
    function = module["main"].with_attr("npu_platform", platform).with_attr("target", tvm.target.Target({"kind": "llvm", "model": model}))
    with pytest.raises(tvm.TVMError, match=error):
        _plan(tvm.IRModule.from_expr(function))


def test_default_and_explicit_legacy_plans_are_identical():
    module = _before_planning(make_vector_kernel())
    default = vector_pass_configs("legacy")
    del default["tl.ascend_pipeline_planner"]
    assert tvm.ir.structural_equal(_plan(module, default), _plan(module, vector_pass_configs("legacy")))


def test_real_lowering_invokes_each_planner_and_injector_once():
    @tvm.instrument.pass_instrument
    class CountPasses:
        def __init__(self):
            self.names = []

        def run_before_pass(self, module, info):
            self.names.append(info.name)

    instrument = CountPasses()
    with tvm.transform.PassContext(config=vector_pass_configs(), instruments=[instrument]):
        tilelang.lower(make_vector_kernel(), target="ascendc", platform="A3")
    assert instrument.names.count("tl.PipelinePlanning") == 1
    assert instrument.names.count("tl.InjectSoftwarePipeline") == 1
    assert instrument.names.count("tl.AscendVerifyConstraintSync") == 1


def test_cold_legacy_lowering_never_imports_solver_or_z3():
    source = """import sys
from importlib.abc import MetaPathFinder
class RejectSolver(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'z3' or fullname.startswith('z3.') or 'ascend_z3_scheduler' in fullname:
            raise AssertionError('legacy requested optional solver: ' + fullname)
sys.meta_path.insert(0, RejectSolver())
import tilelang
from tilelang import tvm
from test_s4_kernels import make_vector_kernel, vector_pass_configs
with tvm.transform.PassContext(config=vector_pass_configs('legacy')):
    tilelang.lower(make_vector_kernel(), target='ascendc', platform='A3')
assert tvm.get_global_func('tl.ascend.solve_schedule', allow_missing=True) is None
"""
    env = dict(os.environ)
    directory = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = os.pathsep.join([directory, env.get("PYTHONPATH", "")])
    completed = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
