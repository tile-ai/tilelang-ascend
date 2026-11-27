"""New local generation is checked by the existing independent physical/value verifier."""

import pytest

import tilelang
from tilelang import tvm

from test_s5_versions_ir import fixture_profile
from test_s2_task_ir import _add, _buffer, _dma, _function
from test_s6_lifecycle import _analyze
from test_s6_sync_coverage import _union_fixture
from test_s6_sync_coverage import _unit
from test_s7_kernels import make_vector_sync_kernel

tir = tvm.tir
TARGET = tvm.target.Target("c -keys=ascend -model=ascendc")
LOCAL = "tl.ascend_local_sync_plan"


def configs():
    return {
        "tl.ascend_pipeline_planner": "constraint",
        "tl.ascend_sync_strategy": "a3",
        "tl.ascend_pipeline_cost_model": fixture_profile(),
        "tl.ascend_auto_sync": True,
        "tl.ascend_memory_planning": True,
        "tl.ascend_auto_sync_vs": False,
        "tl.ascend_auto_cross_core_sync": False,
    }


def strip_sync(function):
    def remove(node):
        if isinstance(node.value, tir.Call) and str(getattr(node.value.op, "name", "")) in {
            "tl.ascend_auto_barrier",
            "tl.ascend_auto_set_flag",
            "tl.ascend_auto_wait_flag",
        }:
            return tir.Evaluate(0)
        return node

    return function.with_body(tir.stmt_functor.ir_transform(function.body, None, remove, ["tir.Evaluate"]))


def generate(function):
    before = tvm.ir.save_json(function)
    with tvm.transform.PassContext(config=configs()):
        result = tilelang.transform.AscendInsertSyncA3(TARGET, "A3")(tvm.IRModule.from_expr(function))["main"]
    assert tvm.ir.save_json(function) == before
    record = result.attrs[LOCAL]
    assert int(record["schema_version"]) == 1 and str(record["generator"]) == "a3"
    assert str(record["lifetime_model"]) == "wait_all_fence_before_reuse"
    assert tvm.ir.structural_equal(record["source_function"].body, function.body)
    assert int(record["event_budget"]["schema_version"]) == 1
    return result


def verify(function):
    before = tvm.ir.save_json(function)
    with tvm.transform.PassContext(config=configs()):
        result = tilelang.transform.AscendVerifyConstraintSync(TARGET, "A3")(tvm.IRModule.from_expr(function))["main"]
    assert result.same_as(function) and tvm.ir.save_json(function) == before


def events(function):
    result = []

    def visit(node):
        if isinstance(node, tir.Call) and "flag" in str(getattr(node.op, "name", "")):
            result.append(node)

    tir.stmt_functor.post_order_visit(function.body, visit)
    return result


def test_source_partial_union_has_real_generated_completion_and_readonly_verification():
    result = generate(strip_sync(_union_fixture()))
    assert events(result)
    verify(result)


def test_actual_final_alias_cannot_be_hidden_by_the_old_disjoint_logical_graph():
    # Source says these are different allocations; final address_map aliases them
    # and overwrites the value before its read. Balanced synchronization cannot fix values.
    with pytest.raises(tvm.TVMError):
        verify(generate(strip_sync(_union_fixture(physical_alias=True))))


@pytest.mark.parametrize("base", [0, 256, 512])
def test_final_physical_reuse_after_the_read_gets_a_real_completion_path(base):
    function = strip_sync(_union_fixture())
    original = function.attrs["address_map"]
    addresses = {data: (0 if int(address) == 0 else base) for data, address in original.items()}
    function = function.with_attr("address_map", addresses)
    # Keep the old disjoint logical graph: the actual final ranges are exact
    # alias, partial overlap, or disjoint. Overwrite remains after the GM read.
    result = generate(function)
    assert all(int(result.attrs["address_map"][data]) == int(value) for data, value in addresses.items())
    verify(result)


@pytest.mark.parametrize("operation", ["tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"])
def test_removing_generated_completion_is_rejected_independently(operation):
    result = generate(strip_sync(_union_fixture()))
    removed = []

    def remove(node):
        if isinstance(node.value, tir.Call) and getattr(node.value.op, "name", "") == operation:
            removed.append(node)
            return tir.Evaluate(0)
        return node

    broken = result.with_body(tir.stmt_functor.ir_transform(result.body, None, remove, ["tir.Evaluate"]))
    assert removed
    with pytest.raises(tvm.TVMError):
        verify(broken)


@pytest.mark.parametrize("condition", [False, True])
def test_actual_noop_guard_does_not_count_as_a_consumed_wait(condition):
    result = generate(strip_sync(_union_fixture()))
    changed = []

    def guard(node):
        if isinstance(node.value, tir.Call) and getattr(node.value.op, "name", "") == "tl.ascend_auto_wait_flag":
            changed.append(node)
            return tir.IfThenElse(tir.IntImm("bool", condition), node, None)
        return node

    altered = result.with_body(tir.stmt_functor.ir_transform(result.body, None, guard, ["tir.Evaluate"]))
    assert changed
    if condition:
        verify(altered)
    else:
        with pytest.raises(tvm.TVMError):
            verify(altered)


def test_existing_automatic_sync_is_not_silently_double_owned():
    with pytest.raises(tvm.TVMError):
        generate(_union_fixture())


@pytest.mark.parametrize("false_branch", [False, True])
def test_whitelisted_fill_cannot_hide_an_unmodeled_nested_call(false_branch):
    function = strip_sync(_union_fixture())
    changed = []

    def inject(node):
        if not changed and isinstance(node.value, tir.Call) and getattr(node.value.op, "name", "") == "tl.ascend_fill":
            call = node.value
            args = list(call.args)
            unknown = tir.call_extern("float32", "unmodeled_fill_scalar")
            args[2] = (
                tir.Call(
                    "float32",
                    tvm.ir.Op.get("tir.if_then_else"),
                    [tir.IntImm("bool", False), unknown, tir.const(1.0, "float32")],
                )
                if false_branch
                else unknown
            )
            changed.append(call)
            return tir.Evaluate(tir.Call(call.dtype, call.op, args))
        return node

    function = function.with_body(tir.stmt_functor.ir_transform(function.body, None, inject, ["tir.Evaluate"]))
    assert len(changed) == 1
    with pytest.raises(tvm.TVMError):
        generate(function)


def independent_dma_source():
    left, right = _buffer("left", shape=(64,)), _buffer("right", shape=(64,))
    first = _buffer("first", scope="global", shape=(64,))
    second = _buffer("second", scope="global", shape=(64,))
    calls = [
        _dma(first, left),
        _add(left, left, left),
        _dma(second, right),
        _add(right, right, right),
    ]
    source = _function([first, second, left, right], calls, noalias=True)
    contract = _analyze(source)
    final = _function([first, second, left, right], [_unit(call, i) for i, call in enumerate(calls)], noalias=True)
    return (
        final.with_attr("address_map", {left.data: 0, right.data: 256})
        .with_attr("size_map", {left.data: 256, right.data: 256})
        .with_attr("tl.ascend_pipeline_buffer_remap", [])
        .with_attr(
            "tl.ascend_constraint_plan",
            {
                "schema_version": 3,
                "source_function": contract["source_function"],
                "source_graph": contract["source_graph"],
                "lifetime_contract": contract,
                "candidates": [],
            },
        )
    )


def pipe_causality_oracle(function):
    """Independent queue DAG: textual adjacency does not order distinct hardware pipes."""
    edges, last, active, history = [], {}, {}, {}
    reused = 0

    def node(pipe):
        current = len(edges)
        edges.append(set())
        if pipe in last:
            edges[last[pipe]].add(current)
        last[pipe] = current
        return current

    def reaches(start, end):
        todo, seen = [start], set()
        while todo:
            item = todo.pop()
            if item == end:
                return True
            if item not in seen:
                seen.add(item)
                todo.extend(edges[item])
        return False

    def evaluate(call):
        nonlocal reused
        name = str(getattr(call.op, "name", ""))
        if name in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"):
            src, dst = str(call.args[0].value).split("_")
            key = (src, dst, int(call.args[1]))
            is_set = name == "tl.ascend_auto_set_flag"
            current = node(src if is_set else dst)
            if is_set:
                if key in history:
                    previous_wait = history[key][1]
                    assert previous_wait is not None and reaches(previous_wait, current), (
                        "previous destination Wait does not happen before the next source Set"
                    )
                    reused += 1
                assert key not in active, "binary event lease was not retired"
                active[key] = [current, None]
                history[key] = active[key]
            else:
                assert key in active and active[key][1] is None, "unmatched queued Wait"
                edges[active[key][0]].add(current)
                active[key][1] = current
            return
        if name in ("tl.ascend_auto_barrier", "tl.ascend_pipe_barrier"):
            pipe = str(call.args[0].value).removeprefix("PIPE_")
            if pipe == "ALL":
                fence = len(edges)
                edges.append(set())
                for previous in set(last.values()):
                    edges[previous].add(fence)
                for participant in ("MTE2", "V", "MTE3"):
                    last[participant] = fence
                for key in list(active):
                    if active[key][1] is not None:
                        assert reaches(active[key][1], fence)
                        del active[key]
            else:
                node(pipe)
            return
        if name in ("tl.ascend_add", "tl.ascend_fill"):
            node("V")
        elif name == "tir.call_extern" and str(call.args[0].value).startswith("copy_gm_to_ub"):
            node("MTE2")
        else:
            raise AssertionError(f"Unsupported operation in the causal fixture: {name}")

    def walk(statement):
        if isinstance(statement, tir.SeqStmt):
            for child in statement.seq:
                walk(child)
        elif isinstance(statement, tir.BlockRealize):
            walk(statement.block)
        elif isinstance(statement, (tir.Block, tir.AttrStmt, tir.Allocate, tir.LetStmt)):
            walk(statement.body)
        elif isinstance(statement, tir.IfThenElse):
            condition = tvm.arith.Analyzer().simplify(statement.condition)
            assert isinstance(condition, tir.IntImm)
            branch = statement.then_case if int(condition) else statement.else_case
            if branch is not None:
                walk(branch)
        elif isinstance(statement, tir.Evaluate):
            if isinstance(statement.value, tir.Call):
                evaluate(statement.value)
        else:
            raise AssertionError(f"Unsupported statement in the causal fixture: {type(statement)}")

    walk(function.body)
    assert not active, "queued Waits do not retire leases without the full-pipe fence"
    return {"reused_ids": reused, "nodes": len(edges)}


def test_independent_dma_events_have_real_destination_consumption_before_id_reuse():
    result = generate(independent_dma_source())
    assert pipe_causality_oracle(result)["reused_ids"] > 0
    verify(result)


@pytest.mark.parametrize("damage", ["remove_all", "source_pipe_only", "late_all"])
def test_linear_set_wait_pairs_cannot_hide_early_binary_flag_reuse(damage):
    result = generate(independent_dma_source())
    assert pipe_causality_oracle(result)["reused_ids"] > 0
    verify(result)
    changed = []

    def remove(node):
        if (
            isinstance(node.value, tir.Call)
            and getattr(node.value.op, "name", "") == "tl.ascend_auto_barrier"
            and str(node.value.args[0].value) == "PIPE_ALL"
        ):
            changed.append(node)
            return (
                tir.Evaluate(tir.Call("handle", node.value.op, [tir.StringImm("PIPE_MTE2")]))
                if damage == "source_pipe_only"
                else tir.Evaluate(0)
            )
        return node

    broken = result.with_body(tir.stmt_functor.ir_transform(result.body, None, remove, ["tir.Evaluate"]))
    assert changed
    if damage == "late_all":
        # A terminal ALL cannot retroactively prevent the earlier double Set.
        broken = broken.with_body(tir.SeqStmt([broken.body, changed[-1]]))
    with pytest.raises(AssertionError, match="previous destination Wait|retired|queued"):
        pipe_causality_oracle(broken)
    with pytest.raises(tvm.TVMError):
        verify(broken)


@pytest.mark.parametrize("versions,trips,cores", [(1, 0, 1), (1, 1, 1), (1, 3, 1), (2, 8, 8), (3, 8, 8)])
def test_complete_lowering_preserves_actual_version_and_event_contract(versions, trips, cores):
    with tvm.transform.PassContext(config=configs()):
        artifact = tilelang.lower(make_vector_sync_kernel(trips, versions, cores), target="ascendc", platform="A3")
    assert any("PipeBarrier" in line and "PIPE_ALL" in line for line in artifact.kernel_source.splitlines())
    for function in artifact.device_mod.functions.values():
        assert LOCAL in function.attrs
        assert int(function.attrs["tl.ascend_constraint_plan"]["schema_version"]) == 3
        assert str(function.attrs[LOCAL]["generator"]) == "a3"
        if trips >= 3:
            actual = [int(row["versions"]) for row in function.attrs["tl.ascend_pipeline_buffer_remap"]]
            assert actual and max(actual) == versions


@pytest.mark.parametrize("platform", ["A5", "unknown"])
def test_new_local_generation_cannot_relabel_other_platforms(platform):
    with tvm.transform.PassContext(config=configs()), pytest.raises(tvm.TVMError):
        tilelang.transform.AscendInsertSyncA3(TARGET, platform)(tvm.IRModule.from_expr(strip_sync(_union_fixture())))
