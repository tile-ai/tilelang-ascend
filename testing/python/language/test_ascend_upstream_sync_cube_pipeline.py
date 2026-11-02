# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU checks for static Cube SIP with two independent physical L1 rings.

The protocol oracle uses final allocation addresses and actual access_ptr byte
offsets. Injector clocks and logical buffer identities never select its slots.
These tests do not compile AscendC/PTO source or execute a device kernel.
"""

from collections import Counter
from dataclasses import dataclass, field

import pytest

from tilelang import language as T
from tilelang import tvm

from test_ascend_pipeline_ring_provenance import _nodes, _pre_sync
from test_ascend_upstream_sync_composite import (
    _RESERVED,
    _changed_call,
    _events,
    _gemms,
    _rewrite_gemm,
    _run,
)
from test_ascend_upstream_sync_composite_loop import _single_step_helper_events
from test_ascend_upstream_sync_pipeline import _assert_balanced_events

tir = tvm.tir
_RINGS = "tl.ascend_physical_ring_provenance"
_OWNERS = "tl.ascend_physical_pipeline_owners"
_EXTENT = "tl.ascend_physical_pipeline_extent"
_SEGMENT = "tl.ascend_physical_pipeline_segment"
_STAGE = "tl.ascend_physical_pipeline_stage"
_CLOCK = "tl.ascend_physical_ring_clock"
_SET = "tl.ascend_auto_set_flag"
_WAIT = "tl.ascend_auto_wait_flag"
_TRIP_COUNTS = (3, 4, 5, 7)


def _pipelined_k_gemm(trip_count=5, num_stages: int = 2) -> tir.PrimFunc:
    # Follow examples/pipeline/gemm_v0_pipeline.py and _ordinary_k_gemm.
    # Only the K iterator changes from the validated ordinary-loop fixture.
    width = 16
    k_l1 = 64
    full_k = trip_count * k_l1

    @T.prim_func
    def main(
        A: T.Tensor((width, full_k), "float16"),
        B: T.Tensor((full_k, width), "float16"),
        C: T.Tensor((width, width), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            a_l1 = T.alloc_L1((width, k_l1), "float16")
            b_l1 = T.alloc_L1((k_l1, width), "float16")
            c_l0 = T.alloc_L0C((width, width), "float32")
            with T.Scope("C"):
                for k in T.Pipelined(trip_count, num_stages=num_stages):
                    T.copy(A[0, k * k_l1], a_l1)
                    T.copy(B[k * k_l1, 0], b_l1)
                    T.gemm_v0(a_l1, b_l1, c_l0, init=(k == 0), kL0Size=k_l1)
                T.copy(c_l0, C[0, 0])

    return main


def _operation(call: tir.Call) -> str:
    name = getattr(call.op, "name", None)
    if name == "tir.call_extern":
        return call.args[0].value.split("<", 1)[0].removeprefix("tl::ascend::")
    return name


def _attributes(function: tir.PrimFunc, key: str) -> list[tir.AttrStmt]:
    return [node for node in _nodes(function.body, tir.AttrStmt) if node.attr_key == key]


@dataclass
class _Execution:
    events: list = field(default_factory=list)
    computations: Counter = field(default_factory=Counter)
    # A/B actual GM pointer identities and element offsets, then clear.
    gemms: list = field(default_factory=list)
    # (operation, operand number, pool, first byte, byte count).
    accesses: list = field(default_factory=list)


class _CubeCompletionOracle:
    """Audit actual byte ownership with independent hardware pipeline clocks.

    Generated flags transfer clocks between MTE2/MTE1/M/FIX. The audited helper
    contributes its MTE1 acquire -> M completion contract only: lexical helper
    return and issue order never complete another pipeline. L1 reads remain
    live until that M completion, including across a later physical slot wrap.
    """

    _PIPES = ("MTE2", "MTE1", "M", "FIX")

    def __init__(self, function: tir.PrimFunc, execution: _Execution, check_dependencies=True):
        self.addresses = function.attrs["address_map"]
        self.sizes = function.attrs["size_map"]
        self.execution = execution
        self.check_dependencies = check_dependencies
        self.clocks = {pipe: [0] * len(self._PIPES) for pipe in self._PIPES}
        self.tokens = {}
        self.writes = {}
        self.read_completions = {}
        self.contents = {}

    def _step(self, pipe, incoming=None):
        clock = self.clocks[pipe]
        if incoming is not None:
            clock[:] = [max(own, other) for own, other in zip(clock, incoming)]
        clock[self._PIPES.index(pipe)] += 1
        return tuple(clock)

    def _require_before(self, previous, current, description):
        if self.check_dependencies:
            assert all(before <= after for before, after in zip(previous, current)), f"byte dependency missing: {description}"

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

    def _bytes(self, pointer, expected_elements, operation, operand, integer, bindings):
        assert pointer.op.name == "tir.tvm_access_ptr"
        data = pointer.args[1]
        scope = str(data.type_annotation.storage_scope)
        assert scope in ("shared.l1", "wmma.accumulator"), scope
        element_bytes = tvm.DataType(pointer.args[0].dtype).bits // 8
        assert element_bytes == (2 if scope == "shared.l1" else 4)
        assert integer(pointer.args[3], bindings) == expected_elements
        offset = integer(pointer.args[2], bindings) * element_bytes
        byte_count = expected_elements * element_bytes
        assert offset >= 0 and offset + byte_count <= integer(self.sizes[data], bindings)
        first = integer(self.addresses[data], bindings) + offset
        self.execution.accesses.append((operation, operand, scope, first, byte_count))
        # Pool is part of the address: L1 and L0C may both start at byte zero.
        return [(scope, address) for address in range(first, first + byte_count)]

    def _write(self, addresses, now, description, content=None):
        for address in addresses:
            if address in self.writes:
                self._require_before(self.writes[address], now, f"WAW {description} {address}")
            previous = self.read_completions.pop(address, None)
            if previous is not None:
                self._require_before(previous, now, f"WAR {description} {address}")
            self.writes[address] = now
            if content is not None:
                self.contents[address] = content

    def load_l1(self, call, integer, bindings):
        source, destination = call.args[1:3]
        assert source.op.name == "tir.tvm_access_ptr"
        source_tile = (source.args[1], integer(source.args[2], bindings))
        addresses = self._bytes(destination, 16 * 64, "load", 0, integer, bindings)
        self._write(addresses, self._step("MTE2"), "before L1 refresh", source_tile)

    def gemm(self, call, integer, bindings):
        assert call.args[0].value == "gemm_v0<half, float, 16, 16, 64, false, false, 64>"
        acquire = self._step("MTE1")
        operands = []
        sources = []
        for operand in (1, 2):
            addresses = self._bytes(call.args[operand], 16 * 64, "gemm", operand, integer, bindings)
            operands.extend(addresses)
            for address in addresses:
                assert address in self.writes, f"L1 read before initialization: {address}"
                self._require_before(self.writes[address], acquire, f"RAW before GEMM {address}")
            source = self.contents[addresses[0]]
            assert all(self.contents[address] == source for address in addresses)
            sources.extend(source)
        accumulator = self._bytes(call.args[3], 16 * 16, "gemm", 3, integer, bindings)
        clear = integer(call.args[4], bindings)
        for address in accumulator:
            if address in self.writes:
                self._require_before(self.writes[address], acquire, f"before accumulating GEMM acquire {address}")
            else:
                assert clear == 1, "the first GEMM must initialize every L0C byte"
        # This is an explicit helper contract edge, not an inference from its
        # C++ return. External consumers still need a generated completion edge.
        completion = self._step("M", acquire)
        for address in operands:
            self.read_completions[address] = completion
        self._write(accumulator, completion, "before L0C accumulation")
        self.execution.gemms.append((*sources, clear))

    def store_c(self, call, integer, bindings):
        addresses = self._bytes(call.args[1], 16 * 16, "fix", 1, integer, bindings)
        acquire = self._step("FIX")
        for address in addresses:
            assert address in self.writes, f"FIX before L0C initialization: {address}"
            self._require_before(self.writes[address], acquire, f"RAW before final FIX {address}")

    def finish(self):
        assert not self.tokens, f"undrained generated completion flags: {self.tokens}"


def _execute_cube_protocol(function: tir.PrimFunc, *, check_dependencies=True) -> _Execution:
    execution = _Execution()
    analyzer = tvm.arith.Analyzer()
    oracle = _CubeCompletionOracle(function, execution, check_dependencies)

    def integer(expression, bindings):
        substitutions = {var: tir.const(value, var.dtype) for var, value in bindings.items()}
        value = analyzer.simplify(tir.stmt_functor.substitute(expression, substitutions))
        assert isinstance(value, tir.IntImm), f"unresolved protocol expression: {value}"
        return int(value.value)

    def visit(statement, bindings, resource=None):
        if isinstance(statement, tir.SeqStmt):
            for child in statement.seq:
                visit(child, bindings, resource)
        elif isinstance(statement, tir.For):
            assert statement.kind in (tir.ForKind.SERIAL, tir.ForKind.UNROLLED)
            minimum = integer(statement.min, bindings)
            extent = integer(statement.extent, bindings)
            assert extent >= 0
            for index in range(minimum, minimum + extent):
                visit(statement.body, {**bindings, statement.loop_var: index}, resource)
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
                assert statement.node.thread_tag in ("blockIdx.x", "blockIdx.y")
                assert integer(statement.value, bindings) > 0
                bindings = {**bindings, statement.node.var: 0}
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
            assert resource == 0, f"Cube operation escaped its resource: {operation}"
            if operation in (_SET, _WAIT):
                assert isinstance(call.args[1], tir.IntImm), "event IDs require literal dispatch"
                event_id = integer(call.args[1], bindings)
                assert 0 <= event_id < 8
                kind = "set" if operation == _SET else "wait"
                event = call.args[0].value
                execution.events.append((kind, event, event_id))
                oracle.flag(kind, event, event_id)
            elif operation == "tl.ascend_auto_barrier":
                oracle.barrier(call)
            else:
                assert operation in (
                    "copy_gm_to_l1",
                    "tl.ascend_gemm_v0",
                    "copy_l0c_to_gm",
                ), operation
                execution.computations[operation] += 1
                if operation == "copy_gm_to_l1":
                    oracle.load_l1(call, integer, bindings)
                elif operation == "tl.ascend_gemm_v0":
                    oracle.gemm(call, integer, bindings)
                    # Independently audit the concrete helper token sequence,
                    # including its primed but unused second L0AB slot. Hidden
                    # tokens never stand in for generated external clock edges.
                    execution.events.extend(_single_step_helper_events())
                else:
                    oracle.store_c(call, integer, bindings)
        else:
            raise AssertionError(f"unsupported protocol shell: {type(statement).__name__}")

    visit(function.body, {})
    oracle.finish()
    return execution


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.fixture(scope="module")
def cube_pipelines():
    functions = {}
    for trip_count in _TRIP_COUNTS:
        for planning in (False, True):
            before, _ = _pre_sync(
                _pipelined_k_gemm(trip_count).with_attr("tir.noalias", True),
                memory_planning=planning,
            )
            functions[trip_count, planning] = (before, _run(before, target_kind="llvm"))
    return functions


def _assert_maps_and_clocks_preserved(before: tir.PrimFunc, after: tir.PrimFunc) -> None:
    for name in ("address_map", "size_map", "buffer_versions", _RINGS, _OWNERS):
        expected, actual = before.attrs[name], after.attrs[name]
        assert len(expected) == len(actual)
        for key, facts in expected.items():
            assert key in actual
            tvm.ir.assert_structural_equal(actual[key], facts)
    for key in (_EXTENT, _SEGMENT, _STAGE, _CLOCK):
        expected, actual = _attributes(before, key), _attributes(after, key)
        assert len(expected) == len(actual)
        for old, new in zip(expected, actual):
            tvm.ir.assert_structural_equal(old.node, new.node)
            tvm.ir.assert_structural_equal(old.value, new.value)
    assert len(_nodes(before.body, tir.Allocate)) == len(_nodes(after.body, tir.Allocate))


@pytest.mark.parametrize("trip_count", _TRIP_COUNTS)
@pytest.mark.parametrize("memory_planning", [False, True])
def test_static_cube_sip_keeps_true_b2_and_independent_lead_one(cube_pipelines, trip_count: int, memory_planning: bool) -> None:
    before, after = cube_pipelines[trip_count, memory_planning]
    rings = list(before.attrs[_RINGS].values())
    assert len(rings) == 2 and not rings[0][2].same_as(rings[1][2])
    for logical, record in before.attrs[_RINGS].items():
        assert [int(record[index]) for index in (0, 1, 3, 4, 5, 6)] == [1, 2, 0, 16384, 16, 0]
        assert int(before.attrs["buffer_versions"][logical]) == 2
        assert int(before.attrs["size_map"][record[2]]) == 4096
    assert sorted(int(before.attrs["address_map"][record[2]]) for record in rings) == [0, 4096]
    assert len(before.attrs[_OWNERS]) == 1
    owner, owner_record = next(iter(before.attrs[_OWNERS].items()))
    assert [int(value) for value in owner_record] == [1, 1, 0, 3]
    envelope = _attributes(before, _EXTENT)
    assert len(envelope) == 1 and envelope[0].node[0].same_as(owner)
    assert int(envelope[0].value) == trip_count
    loops = _nodes(before.body, tir.For)
    assert len(loops) == 1 and int(loops[0].extent) == trip_count - 1
    assert len(_gemms(before)) == 2, "one steady GEMM and one epilogue GEMM"
    assert not any(_operation(call) == "copy_l0c_to_gm" for call in _nodes(envelope[0].body, tir.Call)), (
        "final FIX must remain outside the SIP owner"
    )
    assert not _events(before)
    _assert_maps_and_clocks_preserved(before, after)
    generated = _events(after)
    assert generated
    assert all(event_id not in _RESERVED.get(event, set()) for _, event, event_id in generated)
    assert not any(var.name.startswith("__ascend_if_proxy_") for var in _nodes(after.body, tir.Var))
    tvm.ir.assert_structural_equal(after, _run(after, target_kind="llvm"))


@pytest.mark.parametrize("trip_count", _TRIP_COUNTS)
@pytest.mark.parametrize("memory_planning", [False, True])
def test_static_cube_sip_orders_actual_l1_bytes_and_c_completion(cube_pipelines, trip_count: int, memory_planning: bool) -> None:
    before, after = cube_pipelines[trip_count, memory_planning]
    source_a, source_b = (before.buffer_map[parameter].data for parameter in before.params[:2])
    tokens = set()
    for _invocation in range(3):
        execution = _execute_cube_protocol(after)
        assert execution.computations == {
            "copy_gm_to_l1": 2 * trip_count,
            "tl.ascend_gemm_v0": trip_count,
            "copy_l0c_to_gm": 1,
        }
        assert len(execution.gemms) == trip_count
        for logical, (a_root, a_offset, b_root, b_offset, clear) in enumerate(execution.gemms):
            assert a_root.same_as(source_a) and b_root.same_as(source_b)
            assert (a_offset, b_offset, clear) == (logical * 64, logical * 1024, int(logical == 0))
        _assert_balanced_events(execution.events, tokens)
        # Both alternating physical slots must be observed for both operands;
        # comparing only buffer Var identities cannot satisfy these checks.
        for operand in (1, 2):
            starts = [
                first
                for operation, index, pool, first, size in execution.accesses
                if operation == "gemm" and index == operand and pool == "shared.l1" and size == 2048
            ]
            assert len(starts) == trip_count
            assert starts == [starts[0] + (logical % 2) * 2048 for logical in range(trip_count)]


def _without_event_pairs(function: tir.PrimFunc, directions: set[str]) -> tir.PrimFunc:
    def rewrite(statement):
        if isinstance(statement, tir.Evaluate) and isinstance(statement.value, tir.Call):
            call = statement.value
            if _operation(call) in (_SET, _WAIT) and call.args[0].value in directions:
                return tir.Evaluate(0)
        return None

    return function.with_body(tir.stmt_functor.ir_transform(function.body, rewrite, None, ["tir.Evaluate"]))


@pytest.mark.parametrize("memory_planning", [False, True])
def test_cube_byte_oracle_detects_balanced_protocol_dependency_cuts(cube_pipelines, memory_planning: bool) -> None:
    _, after = cube_pipelines[5, memory_planning]
    present = {event for _, event, _ in _events(after)}
    # Test each actual direction independently. The engine may supply redundant
    # M->MTE1 and M->MTE2 paths, so a redundant single-direction deletion is
    # allowed; the complete cut of both completion paths must still fail.
    candidates = [
        {"MTE2_MTE1"},
        {"M_MTE2"},
        {"M_MTE1"},
        {"M_FIX"},
        {"M_MTE2", "M_MTE1"},
    ]
    for candidate in candidates:
        directions = candidate & present
        if not directions:
            continue
        corrupted = _without_event_pairs(after, directions)
        unchecked = _execute_cube_protocol(corrupted, check_dependencies=False)
        _assert_balanced_events(unchecked.events, set())
        try:
            _execute_cube_protocol(corrupted)
        except AssertionError as error:
            assert "byte dependency missing" in str(error), str(error)
    cuts = {
        "initial L1 readiness": {event for event in present if event.split("_")[0] == "MTE2"},
        "L1 reuse and accumulation": {event for event in present if event.split("_")[0] == "M" and event.split("_")[1] != "FIX"},
        "final FIX acquire": {event for event in present if event.split("_")[1] == "FIX"},
    }
    for description, directions in cuts.items():
        assert directions, f"no generated completion path for {description}: {present}"
        corrupted = _without_event_pairs(after, directions)
        unchecked = _execute_cube_protocol(corrupted, check_dependencies=False)
        _assert_balanced_events(unchecked.events, set())
        with pytest.raises(AssertionError, match="byte dependency missing"):
            _execute_cube_protocol(corrupted)


def _change_first_clock(function: tir.PrimFunc) -> tir.PrimFunc:
    changed = False

    def rewrite(statement):
        nonlocal changed
        if not changed and isinstance(statement, tir.AttrStmt) and statement.attr_key == _CLOCK:
            changed = True
            return tir.AttrStmt(statement.node, statement.attr_key, statement.value + 2, statement.body)
        return None

    body = tir.stmt_functor.ir_transform(function.body, rewrite, None, ["tir.AttrStmt"])
    assert changed
    return function.with_body(body)


def _merge_l1_roots(function: tir.PrimFunc) -> tir.PrimFunc:
    """Legally merge real allocations while preserving every accessed byte.

    Logical ring IDs and their true B remain distinct. This constructs the
    unsupported shared-root layout explicitly, without disabling the real
    StorageRewrite pass or merely corrupting a provenance root field.
    """
    rings = dict(function.attrs[_RINGS].items())
    entries = sorted(rings.items(), key=lambda item: int(function.attrs["address_map"][item[1][2]]))
    assert len(entries) == 2
    (_, first), (second_id, second) = entries
    first_root, second_root = first[2], second[2]
    assert not first_root.same_as(second_root)
    first_bytes = int(function.attrs["size_map"][first_root])
    second_bytes = int(function.attrs["size_map"][second_root])
    first_address = int(function.attrs["address_map"][first_root])
    assert int(function.attrs["address_map"][second_root]) == first_address + first_bytes
    assert int(first[5]) == int(second[5]) == 16
    assert not _nodes(function.body, tir.DeclBuffer)
    assert not _nodes(function.body, tir.BufferLoad)
    assert not _nodes(function.body, tir.BufferStore)

    def rewrite(node):
        if isinstance(node, tir.Call) and _operation(node) == "tir.tvm_access_ptr":
            if node.args[1].same_as(second_root):
                arguments = list(node.args)
                arguments[1] = first_root
                arguments[2] = arguments[2] + first_bytes // 2
                return tir.Call(node.dtype, node.op, arguments, node.span)
        elif isinstance(node, tir.Allocate):
            if node.buffer_var.same_as(second_root):
                return node.body
            if node.buffer_var.same_as(first_root):
                return tir.Allocate(
                    node.buffer_var,
                    node.dtype,
                    [(first_bytes + second_bytes) // 2],
                    node.condition,
                    node.body,
                    node.annotations,
                    node.span,
                )
        return None

    body = tir.stmt_functor.ir_transform(function.body, None, rewrite, ["tir.Call", "tir.Allocate"])
    addresses = dict(function.attrs["address_map"].items())
    sizes = dict(function.attrs["size_map"].items())
    del addresses[second_root]
    del sizes[second_root]
    sizes[first_root] = first_bytes + second_bytes
    second_fields = list(second)
    second_fields[2] = first_root
    second_fields[3] = second_fields[3] + first_bytes * 8
    rings[second_id] = second_fields
    return function.with_body(body).with_attr("address_map", addresses).with_attr("size_map", sizes).with_attr(_RINGS, rings)


@pytest.mark.parametrize("memory_planning", [False, True])
def test_same_root_logical_cube_rings_remain_rejected(cube_pipelines, memory_planning) -> None:
    before, _ = cube_pipelines[5, memory_planning]
    merged = _merge_l1_roots(before)
    original_execution = _execute_cube_protocol(before, check_dependencies=False)
    merged_execution = _execute_cube_protocol(merged, check_dependencies=False)
    assert original_execution.accesses == merged_execution.accesses
    assert original_execution.computations == merged_execution.computations
    records = list(merged.attrs[_RINGS].values())
    assert records[0][2].same_as(records[1][2])
    assert [int(record[1]) for record in records] == [2, 2]
    with pytest.raises(tvm.error.TVMError, match="(?i)(multiple logical|independent.*root|same.*root)"):
        _run(merged, target_kind="llvm")


@pytest.mark.parametrize("change", ["version", "clock", "slot", "footprint"])
def test_static_cube_sip_rejects_inconsistent_ring_facts(cube_pipelines, change: str) -> None:
    before, _ = cube_pipelines[5, False]
    if change == "version":
        versions = dict(before.attrs["buffer_versions"].items())
        versions[next(iter(before.attrs[_RINGS]))] = 3
        malformed = before.with_attr("buffer_versions", versions)
    elif change == "clock":
        # +B selects the same physical slot. Logical stage/clock agreement
        # must be checked independently of matching a modulo pointer offset.
        malformed = _change_first_clock(before)
    elif change == "slot":

        def move_slot(call):
            arguments = list(call.args)
            pointer = arguments[1]
            fields = list(pointer.args)
            fields[2] = fields[2] + 1024
            arguments[1] = tir.Call(pointer.dtype, pointer.op, fields, pointer.span)
            return tir.Evaluate(tir.Call(call.dtype, call.op, arguments, call.span))

        malformed = _rewrite_gemm(before, move_slot)
    else:
        # The real helper would read 4096B per operand, crossing a 2048B ring
        # slot even though the descriptive access_ptr extent remains short.
        malformed = _rewrite_gemm(
            before,
            lambda call: tir.Evaluate(_changed_call(call, template="gemm_v0<half, float, 16, 16, 128, false, false, 64>")),
        )
    with pytest.raises(tvm.error.TVMError, match="(?i)(version|clock|slot|footprint|extent|stage)"):
        _run(malformed, target_kind="llvm")


@pytest.mark.parametrize("platform,model", [("A2", "ascendc"), ("A5", "ascendc"), ("A2", "pto"), ("A5", "pto")])
def test_static_cube_sip_requires_its_a3_target(cube_pipelines, platform, model) -> None:
    before, _ = cube_pipelines[5, False]
    with pytest.raises(tvm.error.TVMError, match="(?i)(AscendC|A3|composite.*target)"):
        _run(before, platform=platform, model=model, target_kind="llvm")


def test_static_cube_sip_accepts_certified_a3_pto_target(cube_pipelines) -> None:
    before, _ = cube_pipelines[5, False]
    after = _run(before, model="pto", target_kind="llvm")
    assert after.attrs.get("tl.ascend_sync_proof") is not None
    _assert_maps_and_clocks_preserved(before, after)
    original_gemms, synchronized_gemms = _gemms(before), _gemms(after)
    assert original_gemms and len(original_gemms) == len(synchronized_gemms)
    for original, synchronized in zip(original_gemms, synchronized_gemms):
        tvm.ir.assert_structural_equal(original, synchronized)
    tvm.ir.assert_structural_equal(after, _run(after, model="pto", target_kind="llvm"))


@pytest.mark.parametrize("change", ["non_cube", "opaque", "user_l0ab"])
def test_static_cube_sip_keeps_owner_effect_and_hidden_scratch_restrictions(cube_pipelines, change: str) -> None:
    before, _ = cube_pipelines[5, False]
    if change == "non_cube":
        malformed = _rewrite_gemm(
            before,
            lambda call: tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, tir.Evaluate(call)),
        )
    elif change == "opaque":
        malformed = _rewrite_gemm(
            before,
            lambda call: tir.SeqStmt([tir.Evaluate(call), tir.Evaluate(tir.call_extern("handle", "unknown_cube_effect"))]),
        )
    else:
        user_l0 = tir.decl_buffer((64,), "float16", name="user_l0", scope="wmma.matrix_a")
        malformed = before.with_body(tir.Allocate(user_l0.data, user_l0.dtype, [64], tir.const(True, "bool"), before.body))
    with pytest.raises(tvm.error.TVMError, match="(?i)(Cube|resource|opaque|unknown|L0|contract)"):
        _run(malformed, target_kind="llvm")


def test_dynamic_cube_sip_remains_outside_the_static_contract() -> None:
    before, _ = _pre_sync(_pipelined_k_gemm(T.symbolic("dynamic_k_tiles")))
    with pytest.raises(tvm.error.TVMError, match="(?i)(static|dynamic|ring|Cube)"):
        _run(before, target_kind="llvm")


def test_vector_helper_cannot_borrow_the_l1_ring_capability() -> None:
    source = tir.decl_buffer((64,), "float32", name="input")
    ring = tir.decl_buffer((128,), "float32", name="ring", scope="shared.l1")
    copy = tir.Evaluate(
        tir.call_extern(
            "handle",
            "tl::ascend::copy_gm_to_ub<float, 64>",
            source.access_ptr("r"),
            ring.access_ptr("w", extent=64),
            64,
            1,
            64,
            tir.const(0, "float32"),
            64,
        )
    )
    body = tir.AttrStmt(
        tir.IntImm("int32", 0),
        "resource_scope",
        1,
        tir.Allocate(
            ring.data,
            ring.dtype,
            [128],
            tir.const(True, "bool"),
            tir.AttrStmt([ring.data], _CLOCK, 0, copy),
        ),
    )
    function = tir.PrimFunc([source.data], body, buffer_map={source.data: source}).with_attrs(
        {
            "address_map": {ring.data: 0},
            "size_map": {ring.data: 512},
            "buffer_versions": {ring.data: 2},
            _RINGS: {ring.data: [1, 2, ring.data, 0, 2048, 32, 0]},
        }
    )
    with pytest.raises(tvm.error.TVMError, match="certified static Cube pipeline"):
        _run(function, target_kind="llvm")


@pytest.mark.parametrize("clear", ["always_false", "always_true", "opaque"])
def test_cube_sip_clear_is_proven_in_its_actual_logical_clock_domain(cube_pipelines, clear) -> None:
    before, _ = cube_pipelines[5, False]
    value = tir.call_extern("bool", "opaque_clear") if clear == "opaque" else tir.const(clear == "always_true", "bool")
    malformed = _rewrite_gemm(before, lambda call: tir.Evaluate(_changed_call(call, clear=value)))
    with pytest.raises(tvm.error.TVMError, match="(?i)(clear|pure)"):
        _run(malformed, target_kind="llvm")


def test_cube_sip_copy_cannot_shrink_its_full_mask_contract(cube_pipelines) -> None:
    before, _ = cube_pipelines[5, False]
    changed = False

    def rewrite(statement):
        nonlocal changed
        if isinstance(statement, tir.Evaluate) and isinstance(statement.value, tir.Call):
            call = statement.value
            if not changed and _operation(call) == "copy_gm_to_l1":
                changed = True
                arguments = list(call.args)
                arguments[4] = tir.IntImm("int32", 15)
                return tir.Evaluate(tir.Call(call.dtype, call.op, arguments, call.span))
        return None

    body = tir.stmt_functor.ir_transform(before.body, None, rewrite, ["tir.Evaluate"])
    assert changed
    with pytest.raises(tvm.error.TVMError, match="(?i)(full|refresh|mask|contract)"):
        _run(before.with_body(body), target_kind="llvm")


def test_cube_sip_owner_lead_cannot_be_reconstructed_from_storage_b(cube_pipelines) -> None:
    before, _ = cube_pipelines[5, False]
    owners = dict(before.attrs[_OWNERS].items())
    owner = next(iter(owners))
    record = list(owners[owner])
    record[1] = 2  # Real lead is one, independently of storage B=2.
    owners[owner] = record
    with pytest.raises(tvm.error.TVMError, match="(?i)(lead|contract)"):
        _run(before.with_attr(_OWNERS, owners), target_kind="llvm")
