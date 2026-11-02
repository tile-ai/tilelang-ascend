# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU regression tests over the original factory's materialized pipeline IR."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

import tilelang
from tilelang import tvm
from tvm import tir


_EXAMPLE = Path(__file__).resolve().parents[3] / "examples/pipeline/gemm_v0_pipeline.py"
_CONFIG = {
    "tl.ascend_sync_backend": "upstream",
    "tl.ascend_auto_sync": True,
    "tl.ascend_auto_sync_vs": True,
}
_AVAILABLE = tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable", allow_missing=True)
_OPS = ("copy_gm_to_l1", "gemm_v0", "copy_l0c_to_gm", "pipe_barrier")


def _i(value: int) -> tir.IntImm:
    return tir.IntImm("int32", value)


class _ScheduleVariant(ast.NodeTransformer):
    def __init__(self, stages: int, barriers: bool, output_dtype: str) -> None:
        self.stages, self.barriers, self.output_dtype = stages, barriers, output_dtype

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "Pipelined":
            for keyword in node.keywords:
                if keyword.arg == "num_stages":
                    keyword.value = ast.Constant(self.stages)
        return node

    def visit_Expr(self, node):
        if not self.barriers and isinstance(node.value, ast.Call):
            callee = node.value.func
            if isinstance(callee, ast.Attribute) and callee.attr == "barrier_all":
                return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if node.name == "main":
            argument = next(a for a in node.args.args if a.arg == "C")
            argument.annotation.args[1] = ast.Constant(self.output_dtype)
        return node


def _capture(
    directory: Path,
    target: str,
    trip: int = 8,
    stages: int = 3,
    barriers: bool = True,
    output_dtype: str = "float16",
    grid: tuple[int, int] = (1, 1),
) -> tir.PrimFunc:
    if _AVAILABLE is None or not _AVAILABLE():
        pytest.skip("Upstream synchronization requires Z3 ON")
    original = ast.parse(_EXAMPLE.read_text())
    # Exclude example main, argument parsing, cache clearing, and all tensor allocation.
    nodes = [n for n in original.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    module = _ScheduleVariant(stages, barriers, output_dtype).visit(ast.Module(body=nodes, type_ignores=[]))
    source = ast.unparse(ast.fix_missing_locations(module))
    filename = directory / f"factory_{target}_{stages}_{barriers}_{output_dtype}.py"
    filename.write_text(source)
    namespace = {"__name__": "cpu_pipeline_fixture", "__file__": str(filename)}
    exec(compile(source, str(filename), "exec", dont_inherit=True), namespace)
    function = namespace["matmul"].__wrapped__(32 * grid[0], 48 * grid[1], trip * 32, 32, 48, 32)
    captured = None

    class Captured(Exception):
        pass

    def intercept(_target, _platform):
        def apply(mod):
            nonlocal captured
            captured = mod["main"]
            raise Captured

        return apply

    # Select the declared NPU storage layout without requiring an installed device.
    with (
        patch.object(tilelang.transform, "AscendInsertSync", intercept),
        patch("tilelang.utils.target.check_npu_availability", return_value=True),
        pytest.raises(Captured),
        tvm.transform.PassContext(opt_level=3, config=_CONFIG),
    ):
        tilelang.lower(function, target=target, platform="A3")
    assert captured is not None
    return captured


def _name(call: tir.Call) -> str:
    if call.op.name == "tir.call_extern":
        return call.args[0].value.split("<")[0].replace("tl::ascend::", "")
    return call.op.name.replace("tl.ascend_", "")


def _trace(function: tir.PrimFunc) -> list:
    analyzer = tvm.arith.Analyzer()
    records = []

    def number(expr, env):
        value = tir.stmt_functor.substitute(expr, {v: tir.const(x, v.dtype) for v, x in env.items()})
        result = analyzer.simplify(value)
        assert isinstance(result, tir.IntImm), str(result)
        return int(result)

    def argument(expr, env, ignore_mask=False):
        if isinstance(expr, tir.StringImm):
            return expr.value
        if isinstance(expr, tir.Call) and expr.op.name == "tir.tvm_access_ptr":
            mask = None if ignore_mask else number(expr.args[4], env)
            return (
                expr.args[1].name,
                str(expr.args[0].dtype),
                number(expr.args[2], env),
                number(expr.args[3], env),
                mask,
            )
        return number(expr, env)

    def walk(stmt, env):
        if isinstance(stmt, tir.SeqStmt):
            for child in stmt.seq:
                walk(child, env)
        elif isinstance(stmt, tir.AttrStmt):
            if stmt.attr_key == "thread_extent":
                for index in range(number(stmt.value, env)) if stmt.node.thread_tag == "blockIdx.x" else (0,):
                    walk(stmt.body, {**env, stmt.node.var: index})
            else:
                walk(stmt.body, env)
        elif isinstance(stmt, (tir.Allocate, tir.DeclBuffer)):
            walk(stmt.body, env)
        elif isinstance(stmt, tir.For):
            start, count = number(stmt.min, env), number(stmt.extent, env)
            for index in range(start, start + count):
                walk(stmt.body, {**env, stmt.loop_var: index})
        elif isinstance(stmt, tir.LetStmt):
            walk(stmt.body, {**env, stmt.var: number(stmt.value, env)})
        elif isinstance(stmt, tir.IfThenElse):
            branch = stmt.then_case if number(stmt.condition, env) else stmt.else_case
            if branch is not None:
                walk(branch, env)
        elif isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call) and _name(stmt.value) in _OPS:
            name = _name(stmt.value)
            args = [argument(a, env, name == "gemm_v0" and j == 3) for j, a in enumerate(stmt.value.args)]
            records.append((name, args))

    walk(function.body, {})
    return records


def _run(function: tir.PrimFunc, target: str) -> tir.PrimFunc:
    device = tvm.target.Target({"kind": "llvm", "model": target})
    with tvm.transform.PassContext(opt_level=3, config=_CONFIG):
        return tilelang.transform.AscendInsertSync(device, "A3")(tvm.IRModule({"main": function}))["main"]


def _accept(function: tir.PrimFunc, target: str) -> tir.PrimFunc:
    result = _run(function, target)
    assert "tl.ascend_sync_proof" in result.attrs
    assert tvm.ir.structural_equal(result, _run(result, target))
    assert _trace(function) == _trace(result)

    def check(node):
        if isinstance(node, tir.Call) and _name(node) == "gemm_v0" and not isinstance(node.args[4], tir.IntImm):
            assert int(node.args[3].args[4]) == 3

    tir.stmt_functor.post_order_visit(result.body, check)
    return result


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize(
    "stages,barriers,trip",
    [
        (2, True, 1),
        (3, True, 3),
        (3, True, 8),
        (5, True, 7),
        (4, False, 8),
    ],
)
def test_materialized_pipeline_regressions(tmp_path, target, stages, barriers, trip):
    # Includes actual guard-simplified q versus q%B branches and the former B5 flag exhaustion.
    _accept(_capture(tmp_path, target, trip, stages, barriers), target)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_pipeline_fp32_output_and_grid(tmp_path, target):
    _accept(_capture(tmp_path, target, trip=5, output_dtype="float32", grid=(2, 2)), target)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("defect", ["pointer", "template", "opaque_clear"])
def test_pipeline_rejects_non_equivalent_gemm_branches(tmp_path, target, defect):
    function = _capture(tmp_path, target)
    changed = 0

    def post(node):
        nonlocal changed
        if not isinstance(node, tir.IfThenElse) or not isinstance(node.else_case, tir.Evaluate):
            return None
        call = node.else_case.value
        if not isinstance(call, tir.Call) or _name(call) != "gemm_v0":
            return None
        args = list(call.args)
        if defect == "pointer":
            pointer = args[1]
            fields = list(pointer.args)
            fields[2] = fields[2] + _i(1)
            args[1] = tir.Call(pointer.dtype, pointer.op, fields)
        elif defect == "template":
            args[0] = tir.StringImm(args[0].value.rsplit(",", 1)[0] + ",64>")
        else:
            args[4] = tir.call_extern("bool", "opaque_clear")
        changed += 1
        return tir.IfThenElse(node.condition, node.then_case, tir.Evaluate(tir.Call(call.dtype, call.op, args)))

    body = tir.stmt_functor.ir_transform(function.body, None, post, ["tir.IfThenElse"])
    assert changed > 0
    with pytest.raises(tvm.error.TVMError, match="conditional|pure|opaque|If|GEMM|gemm"):
        _run(function.with_body(body), target)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_pipeline_rejects_forged_ring_stride(tmp_path, target):
    function = _capture(tmp_path, target, trip=4)
    rings = dict(function.attrs["tl.ascend_physical_ring_provenance"].items())
    identity = next(iter(rings))
    fields = list(rings[identity])
    fields[4] = fields[4] - tir.IntImm("int64", 256)
    rings[identity] = fields
    with pytest.raises(tvm.error.TVMError, match="ring|slot|footprint|provenance"):
        _run(function.with_attr("tl.ascend_physical_ring_provenance", rings), target)


def _gm_input_shape(function: tir.PrimFunc, rows: int, columns: int, update_logic: bool) -> tir.PrimFunc:
    parameter = function.params[0]
    old = function.buffer_map[parameter]
    buffers = dict(function.buffer_map.items())
    buffers[parameter] = tir.decl_buffer((rows, columns), old.dtype, data=old.data, name=old.name)

    def post(node):
        if isinstance(node, tir.Call) and _name(node) == "copy_gm_to_l1" and node.args[1].args[1].same_as(old.data):
            args = list(node.args)
            args[3] = _i(columns)
            return tir.Call(node.dtype, node.op, args)
        return None

    body = tir.stmt_functor.ir_transform(function.body, None, post, ["tir.Call"])
    result = tir.PrimFunc(function.params, body, function.ret_type, buffers, function.attrs)
    if update_logic:
        shapes = dict(result.attrs["logic_buffer_shapes"].items())
        shapes[old.data] = [_i(rows), _i(columns)]
        result = result.with_attr("logic_buffer_shapes", shapes)
    return result


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("boundary", ["row_stride", "total_capacity"])
def test_pipeline_respects_backend_native_gm_limits(tmp_path, target, boundary):
    function = _capture(tmp_path, target, trip=4)
    old = function.buffer_map[function.params[0]]
    if boundary == "row_stride":
        rows, columns = int(old.shape[0]), 65536
    else:
        columns = int(old.shape[1])
        rows = (2**31 - 1) // columns + 1
    # AscendC copies only a small prefix. Its native ABI does not consume the
    # old GM logic-shape metadata; PTO consumes a full int32 descriptor.
    changed = _gm_input_shape(function, rows, columns, update_logic=target == "pto")
    if target == "ascendc":
        _accept(changed, target)
    else:
        with pytest.raises(tvm.error.TVMError, match="PTO|int32|descriptor|stride"):
            _run(changed, target)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_pipeline_accepts_eight_physical_versions(tmp_path, target):
    function = _capture(tmp_path, target, trip=1, stages=8)
    assert all(int(value) == 8 for value in function.attrs["buffer_versions"].values())
    _accept(function, target)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_pipeline_rejects_more_than_eight_physical_versions(tmp_path, target):
    function = _capture(tmp_path, target, trip=1, stages=9)
    assert all(int(value) == 9 for value in function.attrs["buffer_versions"].values())
    with pytest.raises(tvm.error.TVMError, match="version|ring|8|slot"):
        _run(function, target)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("defect", ["clock", "dropped_refresh", "duplicate_refresh"])
def test_pipeline_rejects_corrupted_dataflow_certificate(tmp_path, target, defect):
    function = _capture(tmp_path, target, trip=4)
    changed = False

    def post(node):
        nonlocal changed
        if changed:
            return None
        if defect == "clock":
            if isinstance(node, tir.AttrStmt) and node.attr_key == "tl.ascend_physical_ring_clock":
                changed = True
                return tir.AttrStmt(node.node, node.attr_key, node.value + _i(1), node.body)
        elif isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call) and _name(node.value) == "copy_gm_to_l1":
            changed = True
            if defect == "dropped_refresh":
                return tir.Evaluate(_i(0))
            return tir.SeqStmt([node, node])
        return None

    body = tir.stmt_functor.ir_transform(function.body, None, post)
    assert changed
    with pytest.raises(tvm.error.TVMError, match="clock|coverage|effect|order|pipeline"):
        _run(function.with_body(body), target)
