# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU execution of the actual injected-ring synchronization protocol.

Control flow and event tokens are interpreted independently of synchronization
analysis. Hardware compute operations are counted without running a kernel or
invoking the AscendC/PTO compiler.
"""

from collections import Counter

import pytest

import tilelang
from tilelang import tvm

from test_ascend_pipeline_ring_provenance import _nodes, _pipelined_abs, _pre_sync

tir = tvm.tir
_SET = "tl.ascend_auto_set_flag"
_WAIT = "tl.ascend_auto_wait_flag"
_PIPELINE_CASES = [(2, 4), (2, 5), (3, 7), (3, 8)]


def _operation(call):
    name = getattr(call.op, "name", None)
    if name == "tir.call_extern":
        return call.args[0].value.split("<", 1)[0].removeprefix("tl::ascend::")
    return name


class _VectorCompletionOracle:
    """Check real UB slot RAW/WAR/WAW ordering with independent pipe clocks.

    A lexical task issue never completes another pipeline. Generated flags
    transfer clocks; the concrete offsets and final allocation addresses select
    bytes independently of logical version annotations or engine dependencies.
    This oracle is specific to the full 64-lane float32 fixture below.
    """

    _PIPES = ("MTE2", "V", "MTE3", "S")

    def __init__(self, function):
        self.addresses = function.attrs["address_map"]
        self.clocks = {pipe: [0] * len(self._PIPES) for pipe in self._PIPES}
        self.tokens = {}
        self.writes = {}
        self.reads = {}

    def _step(self, pipe, incoming=None):
        clock = self.clocks[pipe]
        if incoming is not None:
            clock[:] = [max(own, other) for own, other in zip(clock, incoming)]
        clock[self._PIPES.index(pipe)] += 1
        return tuple(clock)

    @staticmethod
    def _require_before(previous, current, description):
        assert all(before <= after for before, after in zip(previous, current)), f"memory dependency missing: {description}"

    def flag(self, kind, event, event_id):
        source, destination = event.split("_")
        assert source in self._PIPES and destination in self._PIPES
        key = (event, event_id)
        if kind == "set":
            assert key not in self.tokens, f"double set: {key}"
            self.tokens[key] = self._step(source)
        else:
            assert key in self.tokens, f"wait without token: {key}"
            self._step(destination, self.tokens.pop(key))

    def barrier(self, call):
        pipe = call.args[0].value.removeprefix("PIPE_")
        assert pipe in self._PIPES
        self._step(pipe)

    def _lanes(self, pointer, integer, bindings):
        assert pointer.op.name == "tir.tvm_access_ptr"
        assert pointer.args[0].dtype == "float32"
        assert integer(pointer.args[3], bindings) == 64
        base = integer(self.addresses[pointer.args[1]], bindings)
        offset = integer(pointer.args[2], bindings) * 4
        return range(base + offset, base + offset + 64 * 4, 4)

    def _read(self, lanes, pipe, now):
        for address in lanes:
            assert address in self.writes, f"read before initialization: {address}"
            self._require_before(self.writes[address], now, f"RAW at UB byte {address}")
            self.reads.setdefault(address, {})[pipe] = now

    def _write(self, lanes, now):
        for address in lanes:
            if address in self.writes:
                self._require_before(self.writes[address], now, f"WAW at UB byte {address}")
            for reader in self.reads.pop(address, {}).values():
                self._require_before(reader, now, f"WAR at UB byte {address}")
            self.writes[address] = now

    def operation(self, operation, call, integer, bindings):
        if operation == "copy_gm_to_ub":
            self._write(self._lanes(call.args[2], integer, bindings), self._step("MTE2"))
        elif operation == "tl.ascend_abs":
            assert integer(call.args[2], bindings) == 64
            now = self._step("V")
            self._read(self._lanes(call.args[1], integer, bindings), "V", now)
            self._write(self._lanes(call.args[0], integer, bindings), now)
        else:
            assert operation == "copy_ub_to_gm"
            self._read(self._lanes(call.args[1], integer, bindings), "MTE3", self._step("MTE3"))

    def finish(self):
        assert not self.tokens, f"undrained completion flags: {self.tokens}"


def _execute_protocol(function, vector_id):
    """Interpret all control shells; reject unresolved conditions explicitly."""
    events = []
    computations = Counter()
    analyzer = tvm.arith.Analyzer()
    completion_oracle = _VectorCompletionOracle(function)

    def integer(expression, bindings):
        replacements = {var: tir.const(value, var.dtype) for var, value in bindings.items()}
        constant = analyzer.simplify(tir.stmt_functor.substitute(expression, replacements))
        assert isinstance(constant, tir.IntImm), f"unresolved protocol expression: {constant}"
        return int(constant.value)

    def visit(statement, bindings, resource=None):
        if isinstance(statement, tir.SeqStmt):
            for child in statement.seq:
                visit(child, bindings, resource)
        elif isinstance(statement, tir.For):
            assert statement.kind in (tir.ForKind.SERIAL, tir.ForKind.UNROLLED)
            start = integer(statement.min, bindings)
            extent = integer(statement.extent, bindings)
            assert extent >= 0
            for iteration in range(start, start + extent):
                visit(statement.body, {**bindings, statement.loop_var: iteration}, resource)
        elif isinstance(statement, tir.IfThenElse):
            branch = statement.then_case if integer(statement.condition, bindings) else statement.else_case
            if branch is not None:
                visit(branch, bindings, resource)
        elif isinstance(statement, tir.LetStmt):
            value = integer(statement.value, bindings)
            visit(statement.body, {**bindings, statement.var: value}, resource)
        elif isinstance(statement, tir.Allocate):
            assert integer(statement.condition, bindings)
            visit(statement.body, bindings, resource)
        elif isinstance(statement, tir.DeclBuffer):
            visit(statement.body, bindings, resource)
        elif isinstance(statement, tir.AttrStmt):
            if statement.attr_key == "thread_extent":
                tag = statement.node.thread_tag
                assert tag in ("blockIdx.x", "blockIdx.y"), f"unexpected thread domain: {tag}"
                index = vector_id if tag == "blockIdx.y" else 0
                assert 0 <= index < integer(statement.value, bindings)
                bindings = {**bindings, statement.node.var: index}
            elif statement.attr_key == "resource_scope":
                resource = integer(statement.value, bindings)
            visit(statement.body, bindings, resource)
        elif isinstance(statement, tir.Evaluate):
            call = statement.value
            if isinstance(call, tir.IntImm):
                assert int(call.value) == 0
                return
            assert isinstance(call, tir.Call), f"unexpected evaluation: {call}"
            operation = _operation(call)
            if operation in (_SET, _WAIT):
                assert resource == 1, "the Vector protocol escaped its execution resource"
                assert isinstance(call.args[1], tir.IntImm), "event dispatch must use literal IDs"
                event_id = integer(call.args[1], bindings)
                assert 0 <= event_id < 8
                kind = "set" if operation == _SET else "wait"
                events.append((kind, call.args[0].value, event_id))
                completion_oracle.flag(kind, call.args[0].value, event_id)
            elif operation == "tl.ascend_auto_barrier":
                assert resource == 1
                completion_oracle.barrier(call)
            else:
                assert operation in ("copy_gm_to_ub", "tl.ascend_abs", "copy_ub_to_gm"), operation
                computations[operation] += 1
                completion_oracle.operation(operation, call, integer, bindings)
        else:
            raise AssertionError(f"unsupported protocol statement: {type(statement).__name__}")

    visit(function.body, {})
    completion_oracle.finish()
    return events, computations


def _assert_balanced_events(events, tokens):
    assert not tokens, f"kernel entry has stale event tokens: {tokens}"
    assert events, "the actual pipeline must contain synchronization events"
    for position, (operation, event, event_id) in enumerate(events):
        key = (event, event_id)
        if operation == "set":
            assert key not in tokens, f"double set at event {position}: {key}"
            tokens.add(key)
        else:
            assert key in tokens, f"wait without token at event {position}: {key}"
            tokens.remove(key)
    assert not tokens, f"kernel exit has undrained event tokens: {tokens}"


@pytest.fixture(scope="module")
def synchronized_pipelines():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    pipelines = {}
    for num_stages, trip_count in _PIPELINE_CASES:
        for memory_planning in (False, True):
            original, _ = _pre_sync(_pipelined_abs(trip_count, num_stages), memory_planning=memory_planning)
            with tvm.transform.PassContext(config={"tl.ascend_auto_sync": True, "tl.ascend_auto_sync_vs": True}):
                result = tilelang.transform.AscendInsertSync(target, "A3")(tvm.IRModule({"main": original}))["main"]
            pipelines[num_stages, trip_count, memory_planning] = (original, result)
    return pipelines


@pytest.mark.parametrize("num_stages,trip_count", _PIPELINE_CASES)
@pytest.mark.parametrize("memory_planning", [False, True])
@pytest.mark.parametrize("vector_id", [0, 1])
def test_real_injected_ring_protocol_is_balanced(synchronized_pipelines, num_stages, trip_count, memory_planning, vector_id):
    original, result = synchronized_pipelines[num_stages, trip_count, memory_planning]
    original_versions = original.attrs["buffer_versions"]
    result_versions = result.attrs["buffer_versions"]
    assert len(original_versions) == len(result_versions) == 2
    for logical_id, version in original_versions.items():
        assert int(result_versions[logical_id]) == int(version)
    assert sorted(int(version) for version in result_versions.values()) == [1, num_stages]
    for logical_id, record in result.attrs["tl.ascend_physical_ring_provenance"].items():
        assert int(record[1]) == int(result_versions[logical_id]) == num_stages

    literal_calls = [
        (_operation(call), call.args[0].value, int(call.args[1]))
        for call in _nodes(result.body, tir.Call)
        if _operation(call) in (_SET, _WAIT)
    ]
    assert literal_calls
    assert all(0 <= event_id < 8 for _, _, event_id in literal_calls)
    print(f"B={num_stages} count={trip_count} memory_planning={memory_planning} literal_events={literal_calls}")
    tokens = set()
    for invocation in range(3):
        events, computations = _execute_protocol(result, vector_id)
        assert computations == {
            "copy_gm_to_ub": trip_count,
            "tl.ascend_abs": trip_count,
            "copy_ub_to_gm": trip_count,
        }
        _assert_balanced_events(events, tokens)
        if invocation == 0:
            print(f"vid={vector_id} executed_events={events}")


@pytest.mark.parametrize("remove", ["set", "wait"])
def test_protocol_oracle_detects_missing_credits_and_drains(synchronized_pipelines, remove):
    _, result = synchronized_pipelines[2, 4, False]
    events, _ = _execute_protocol(result, 0)
    corrupted = [event for event in events if event[0] != remove]
    with pytest.raises(AssertionError, match="wait without token|double set|undrained"):
        _assert_balanced_events(corrupted, set())


@pytest.mark.parametrize("event", ["MTE2_V", "V_MTE2", "V_MTE3", "MTE3_V"])
def test_completion_oracle_rejects_balanced_flags_with_missing_ring_dependency(synchronized_pipelines, event):
    _, result = synchronized_pipelines[3, 7, False]

    def remove_pair(statement):
        if isinstance(statement, tir.Evaluate) and isinstance(statement.value, tir.Call):
            call = statement.value
            if _operation(call) in (_SET, _WAIT) and call.args[0].value == event:
                return tir.Evaluate(0)
        return None

    corrupted = result.with_body(tir.stmt_functor.ir_transform(result.body, remove_pair, None, ["tir.Evaluate"]))
    with pytest.raises(AssertionError, match="memory dependency missing"):
        _execute_protocol(corrupted, 0)
