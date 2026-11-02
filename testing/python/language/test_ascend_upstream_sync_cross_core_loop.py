# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""One C-only static prefix loop followed by the existing single publication."""

from collections import Counter

import pytest

import tilelang
from tilelang import language as T
from tilelang import tvm
from tilelang.jit.ascend_runtime_contract import prepare_auto_gm_contract
from test_ascend_upstream_sync_cross_core import _run
from test_ascend_upstream_sync_runtime_contract import _mixed

tir = tvm.tir
_M, _N, _K_TILE = 32, 16, 64
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_SYNC_BACKEND: "upstream",
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


def _developer_kloop_gemm_exp(k_tiles: int = 2, *, explicit_noalias: bool = False):
    total_k = k_tiles * _K_TILE

    @T.prim_func
    def main(
        A: T.Tensor((_M, total_k), "float16"),
        B: T.Tensor((total_k, _N), "float16"),
        C: T.Tensor((_M, _N), "float16"),
    ):
        with T.Kernel(1, threads=2, is_npu=True):
            a_l1 = T.alloc_shared((_M, _K_TILE), "float16")
            b_l1 = T.alloc_shared((_K_TILE, _N), "float16")
            accumulator = T.alloc_fragment((_M, _N), "float32")
            result_ub = T.alloc_shared((_M, _N), "float16")
            for k in T.serial(k_tiles):
                T.copy(A[:, k * _K_TILE : (k + 1) * _K_TILE], a_l1)
                T.copy(B[k * _K_TILE : (k + 1) * _K_TILE, :], b_l1)
                T.gemm_v0(a_l1, b_l1, accumulator, init=(k == 0))
            T.copy(accumulator, result_ub)
            T.tile.exp(result_ub, result_ub)
            T.copy(result_ub, C[:, :])

    if explicit_noalias:
        return main.with_attr("tir.noalias", True)
    return main.without_attr("tir.noalias")


def _pre_sync(program):
    class Captured(Exception):
        pass

    captured = []

    @tvm.instrument.pass_instrument
    class Capture:
        def run_before_pass(self, module, info):
            if info.name == "tl.AscendInsertSync":
                captured.append(module["main"])
                raise Captured

    prepared = prepare_auto_gm_contract(program, "cython", "ascendc")
    try:
        with tvm.transform.PassContext(config=_PASS_CONFIGS, instruments=[Capture()]):
            tilelang.lower(prepared, target="ascendc", platform="A3")
    except Captured:
        pass
    assert len(captured) == 1
    return captured[0]


def _call_name(call):
    if call.op.name == "tir.call_extern":
        return call.args[0].value.split("<", 1)[0].removeprefix("tl::ascend::")
    return call.op.name


def _nodes(statement, kind):
    result = []
    tir.stmt_functor.post_order_visit(statement, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def _simple_prefix(trips, *, mutation=None):
    original = _mixed()
    index = tir.Var("k", "int32")

    def replace(node):
        if not isinstance(node, tir.Evaluate) or not isinstance(node.value, tir.Call):
            return node
        if _call_name(node.value) != "copy_gm_to_l1":
            return node
        body = node
        if mutation == "inside_set":
            flags = [call for call in _nodes(original.body, tir.Call) if call.op.name == "tl.ascend_auto_set_cross_flag"]
            body = tir.SeqStmt([node, tir.Evaluate(flags[0])])
        if mutation == "gm_write":
            output = original.buffer_map[original.params[2]]
            body = tir.SeqStmt([node, tir.BufferStore(output, tir.FloatImm("float32", 1), [0])])
        return tir.For(index, 0, trips, tir.ForKind.SERIAL, body)

    result = original.with_body(tir.stmt_functor.ir_transform(original.body, None, replace, ["tir.Evaluate"]))
    if isinstance(trips, tir.Var):
        result = tir.PrimFunc(
            list(result.params) + [trips],
            result.body,
            buffer_map=result.buffer_map,
            attrs=result.attrs,
        )
    return result


def _executed(function, resource, vector_id):
    """Execute original control shells, including generated flag guards."""
    analyzer = tvm.arith.Analyzer()
    result = []

    def integer(expression, values):
        value = analyzer.simplify(tir.stmt_functor.substitute(expression, values))
        assert isinstance(value, tir.IntImm), value
        return int(value)

    def execute(statement, values):
        if isinstance(statement, tir.SeqStmt):
            for child in statement.seq:
                execute(child, values)
        elif isinstance(statement, tir.AttrStmt):
            if statement.attr_key == "resource_scope" and int(statement.value) != resource:
                return
            if statement.attr_key == "thread_extent":
                value = 0 if statement.node.thread_tag == "blockIdx.x" else vector_id
                values = {**values, statement.node.var: tir.const(value, statement.node.var.dtype)}
            execute(statement.body, values)
        elif isinstance(statement, (tir.Allocate, tir.DeclBuffer)):
            execute(statement.body, values)
        elif isinstance(statement, tir.LetStmt):
            value = integer(statement.value, values)
            execute(statement.body, {**values, statement.var: tir.const(value, statement.var.dtype)})
        elif isinstance(statement, tir.For):
            minimum, extent = integer(statement.min, values), integer(statement.extent, values)
            for index in range(minimum, minimum + extent):
                execute(statement.body, {**values, statement.loop_var: tir.const(index, "int32")})
        elif isinstance(statement, tir.IfThenElse):
            branch = statement.then_case if integer(statement.condition, values) else statement.else_case
            if branch is not None:
                execute(branch, values)
        elif isinstance(statement, tir.Evaluate):
            if isinstance(statement.value, tir.Call):
                result.append((statement.value, values.copy()))
        else:
            raise AssertionError(type(statement))

    execute(function.body, {})
    return [(call, lambda value, scope=scope: integer(value, scope)) for call, scope in result]


def _completion_oracle(function, vector_id, *, omit_c_completion=False):
    """Independent issue/completion clocks for the synthetic C-read fixture.

    A lexical C read does not complete another pipe. Only generated events
    propagate its clock to FIX, the existing cross instruction propagates FIX
    to V's S, and the local engine must propagate S before the V consumer.
    """
    pipes = [(owner, pipe) for owner in (0, 1) for pipe in ("S", "MTE1", "MTE2", "MTE3", "V", "M", "FIX")]
    clocks = {pipe: [0] * len(pipes) for pipe in pipes}
    tokens, release, workspace, loaded, transformed = {}, None, None, None, None
    reads, counts = [], Counter()

    def step(owner, pipe, incoming=None):
        clock = clocks[owner, pipe]
        if incoming is not None:
            clock[:] = [max(a, b) for a, b in zip(clock, incoming)]
        clock[pipes.index((owner, pipe))] += 1
        return tuple(clock)

    def before(first, second):
        assert first is not None and second is not None
        assert all(a <= b for a, b in zip(first, second)), "missing asynchronous completion"

    for owner in (0, 1):
        for call, integer in _executed(function, owner, 0 if owner == 0 else vector_id):
            name = _call_name(call)
            if name in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"):
                event = call.args[0].value
                source, destination = event.split("_")
                if omit_c_completion and owner == 0 and event == "MTE2_FIX":
                    continue
                key = (owner, event, integer(call.args[1]))
                if name.endswith("set_flag"):
                    assert key not in tokens, f"double set: {key}"
                    tokens[key] = step(owner, source)
                else:
                    assert key in tokens, f"wait without producer: {key}"
                    step(owner, destination, tokens.pop(key))
            elif name == "tl.ascend_auto_barrier":
                step(owner, call.args[0].value.removeprefix("PIPE_"))
            elif name == "copy_gm_to_l1":
                reads.append(step(0, "MTE2"))
                counts["c_reads"] += 1
            elif name == "copy_l0c_to_gm":
                workspace = step(0, "FIX")
            elif name == "tl.ascend_auto_set_cross_flag":
                assert owner == 0 and release is None
                release = step(0, "FIX")
                for read in reads:
                    before(read, release)
            elif name == "tl.ascend_auto_wait_cross_flag":
                assert owner == 1 and release is not None
                step(1, "S", release)
            elif name == "copy_gm_to_ub":
                loaded = step(1, "MTE2")
                before(workspace, loaded)
            elif name == "tl.ascend_abs":
                transformed = step(1, "V")
                before(loaded, transformed)
            elif name == "copy_ub_to_gm":
                stored = step(1, "MTE3")
                before(transformed, stored)
                for read in reads:
                    before(read, stored)
            else:
                raise AssertionError(name)
    assert not tokens, f"undrained local event tokens: {tokens}"
    return counts


@pytest.fixture(autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.mark.parametrize("trips", [1, 2, 3])
@pytest.mark.parametrize("vector_id", [0, 1])
def test_c_prefix_keeps_real_loop_and_completes_war_before_publication(trips, vector_id):
    original = _simple_prefix(trips)
    result = _run(original)
    assert len(_nodes(result.body, tir.For)) == 1
    assert _completion_oracle(result, vector_id)["c_reads"] == trips
    with pytest.raises(AssertionError, match="missing asynchronous completion"):
        _completion_oracle(result, vector_id, omit_c_completion=True)


@pytest.mark.parametrize("mutation,diagnostic", [("inside_set", "outside"), ("gm_write", "cannot.*GM")])
def test_c_prefix_cannot_contain_publication_or_gm_overwrite(mutation, diagnostic):
    with pytest.raises(tvm.error.TVMError, match=diagnostic):
        _run(_simple_prefix(2, mutation=mutation))


@pytest.mark.parametrize("trips", [0, -1])
def test_c_prefix_requires_a_positive_static_extent(trips):
    with pytest.raises(tvm.error.TVMError, match="positive"):
        _run(_simple_prefix(trips))


def test_c_prefix_symbolic_extent_is_not_enabled_by_the_static_certificate():
    with pytest.raises(tvm.error.TVMError, match="static positive"):
        _run(_simple_prefix(tir.Var("iterations", "int32")))


def test_complete_c_prefix_must_precede_the_outside_set():
    original = _simple_prefix(2)

    def replace(node):
        if not isinstance(node, tir.SeqStmt):
            return node
        loops = [i for i, child in enumerate(node.seq) if isinstance(child, tir.For)]
        sets = [
            i
            for i, child in enumerate(node.seq)
            if isinstance(child, tir.Evaluate)
            and isinstance(child.value, tir.Call)
            and child.value.op.name == "tl.ascend_auto_set_cross_flag"
        ]
        if not loops or not sets:
            return node
        statements = list(node.seq)
        publication = statements.pop(sets[0])
        statements.insert(loops[0], publication)
        return tir.SeqStmt(statements)

    changed = original.with_body(tir.stmt_functor.ir_transform(original.body, None, replace, ["tir.SeqStmt"]))
    with pytest.raises(tvm.error.TVMError, match="complete C-only prefix For before"):
        _run(changed)


def test_vector_control_is_not_admitted_by_cube_prefix_support():
    original = _simple_prefix(2)

    def replace(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "resource_scope" and int(node.value) == 1:
            return tir.AttrStmt(
                node.node,
                node.attr_key,
                node.value,
                tir.For(tir.Var("v_loop", "int32"), 0, 2, tir.ForKind.SERIAL, node.body),
            )
        return node

    changed = original.with_body(tir.stmt_functor.ir_transform(original.body, None, replace, ["tir.AttrStmt"]))
    with pytest.raises(tvm.error.TVMError, match="flat input"):
        _run(changed)


@pytest.mark.parametrize("k_tiles", [2, 3])
@pytest.mark.parametrize("explicit_noalias", [False, True])
def test_real_developer_kloop_and_single_publication_keep_computation(k_tiles, explicit_noalias):
    before = _pre_sync(_developer_kloop_gemm_exp(k_tiles, explicit_noalias=explicit_noalias))
    after = _run(before)
    assert len(_nodes(after.body, tir.For)) == 1
    original = [call for call in _nodes(before.body, tir.Call) if call.op.name != "tir.tvm_access_ptr"]
    retained = [
        call
        for call in _nodes(after.body, tir.Call)
        if call.op.name
        not in (
            "tir.tvm_access_ptr",
            "tl.ascend_auto_set_flag",
            "tl.ascend_auto_wait_flag",
            "tl.ascend_auto_barrier",
        )
    ]
    assert len(original) == len(retained)
    assert all(tvm.ir.structural_equal(a, b) for a, b in zip(original, retained))
    assert int(_nodes(after.body, tir.For)[0].extent) == k_tiles
