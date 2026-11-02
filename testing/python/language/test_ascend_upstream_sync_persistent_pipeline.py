# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU admission and asynchronous byte tests for Persistent containing SIP.

The executable oracle expands actual final TIR into four independent FIFO
queues. A scalar wait stalls only S. Physical UB bytes come from access_ptr
offsets and final allocation maps, independently of injector clock metadata.
No device compiler or NPU is used by these tests.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import permutations
import random

import pytest

from tilelang import language as T
from tilelang import tvm

from test_ascend_pipeline_ring_provenance import _nodes, _pre_sync
from test_ascend_upstream_sync_composite import _run

tir = tvm.tir
_WORKERS = 4
_WIDTH = 64
_PIPES = ("S", "V", "MTE2", "MTE3")
_ACTIVE = _PIPES[1:]
_SEALS = frozenset([*(("S", pipe, 0) for pipe in _ACTIVE), *((pipe, "S", 0) for pipe in _ACTIVE)])
_RINGS = "tl.ascend_physical_ring_provenance"
_OWNERS = "tl.ascend_physical_pipeline_owners"
_EXTENT = "tl.ascend_physical_pipeline_extent"
_CLOCK = "tl.ascend_physical_ring_clock"
_CASES = ((9, 3), (9, 5), (10, 5), (17, 3))
_ADVERSARIAL = (
    (2, 1, 0, 3),  # MTE2 fast, MTE3 last: expose refresh before delayed store.
    (1, 2, 0, 3),  # Vector fast, MTE3 last: expose result-B1 WAR.
    (0, 3, 2, 1),  # Scalar fast: stress wave entry and reused initializer IDs.
)


def _persistent_pipelined_abs(rows: int = 9, tiles=5) -> tir.PrimFunc:
    # Follow _persistent_abs and _pipelined_abs: only place the existing
    # complete-tile Vector SIP inside the canonical Persistent row iterator.
    @T.prim_func
    def main(
        A: T.Tensor((2, rows, tiles, _WIDTH), "float32"),
        O: T.Tensor((2, rows, tiles, _WIDTH), "float32"),
    ):
        with T.Kernel(_WORKERS, is_npu=True) as (cid, vid):
            for row, _column in T.Persistent([rows, 1], _WORKERS, cid, group_size=1):
                with T.Scope("V"):
                    source = T.alloc_ub((_WIDTH,), "float32")
                    result = T.alloc_ub((_WIDTH,), "float32")
                    for tile in T.Pipelined(tiles, num_stages=2):
                        T.copy(A[vid, row, tile, :], source)
                        T.tile.abs(result, source)
                        T.copy(result, O[vid, row, tile, :])

    return main.with_attr("tir.noalias", True)


def _operation(call: tir.Call) -> str:
    name = getattr(call.op, "name", "")
    if name == "tir.call_extern":
        return call.args[0].value.split("<", 1)[0].removeprefix("tl::ascend::")
    return name


def _attributes(function: tir.PrimFunc, name: str):
    return [node for node in _nodes(function.body, tir.AttrStmt) if node.attr_key == name]


def _has_break(statement: tir.Stmt) -> bool:
    return any(_operation(call) == "tl.loop_break" for call in _nodes(statement, tir.Call))


@dataclass(frozen=True)
class _Command:
    pipe: str
    kind: str
    event: tuple[str, str, int] | None = None
    reads: tuple = ()
    writes: tuple = ()
    global_elements: tuple[int, ...] = ()
    description: str = ""

    def __str__(self):
        return f"{self.pipe} {self.kind} {self.event or self.description}"


@dataclass
class _Program:
    queues: tuple
    computations: Counter


def _commands(function: tir.PrimFunc, cid: int, vid: int, rows: int, tiles: int) -> _Program:
    """Capture all actual loop iterations before executing any hardware pipe.

    This deliberately permits commands for later waves to be enqueued already.
    Only real per-pipe FIFO order and flag set/wait commands can order their
    completion; the lexical For/If traversal supplies no cross-pipeline edge.
    """
    queues = {pipe: [] for pipe in _PIPES}
    computations = Counter()
    analyzer = tvm.arith.Analyzer()
    addresses, sizes = function.attrs["address_map"], function.attrs["size_map"]
    source, output = (function.buffer_map[parameter].data for parameter in function.params[:2])

    class LoopBreak(Exception):
        pass

    def integer(expression, bindings):
        substitutions = {var: tir.const(value, var.dtype) for var, value in bindings.items()}
        value = analyzer.simplify(tir.stmt_functor.substitute(expression, substitutions))
        assert isinstance(value, tir.IntImm), f"unresolved executable expression: {value}"
        return int(value.value)

    def ub_bytes(pointer, bindings):
        assert pointer.op.name == "tir.tvm_access_ptr"
        assert pointer.args[0].dtype == "float32"
        assert integer(pointer.args[3], bindings) == _WIDTH
        data = pointer.args[1]
        scope = str(data.type_annotation.storage_scope)
        assert scope == "shared.ub", scope
        offset = integer(pointer.args[2], bindings) * 4
        assert offset >= 0 and offset + _WIDTH * 4 <= integer(sizes[data], bindings)
        first = integer(addresses[data], bindings) + offset
        return tuple((scope, address) for address in range(first, first + _WIDTH * 4))

    def gm_elements(pointer, expected, bindings):
        assert pointer.op.name == "tir.tvm_access_ptr"
        assert pointer.args[0].dtype == "float32" and pointer.args[1].same_as(expected)
        assert integer(pointer.args[3], bindings) == _WIDTH
        first = integer(pointer.args[2], bindings)
        assert first >= 0 and first + _WIDTH <= 2 * rows * tiles * _WIDTH
        return tuple(range(first, first + _WIDTH))

    def enqueue(pipe, kind, **fields):
        assert pipe in queues, pipe
        queues[pipe].append(_Command(pipe, kind, **fields))

    def visit(statement, bindings, resource=None):
        if isinstance(statement, tir.SeqStmt):
            for child in statement.seq:
                visit(child, bindings, resource)
        elif isinstance(statement, tir.For):
            assert statement.kind in (tir.ForKind.SERIAL, tir.ForKind.UNROLLED)
            minimum, extent = integer(statement.min, bindings), integer(statement.extent, bindings)
            assert extent >= 0
            for iteration in range(minimum, minimum + extent):
                enqueue("S", "scalar", description=f"{statement.loop_var.name}={iteration}")
                try:
                    visit(statement.body, {**bindings, statement.loop_var: iteration}, resource)
                except LoopBreak:
                    break
        elif isinstance(statement, tir.IfThenElse):
            branch = statement.then_case if integer(statement.condition, bindings) else statement.else_case
            if branch is not None:
                visit(branch, bindings, resource)
        elif isinstance(statement, tir.LetStmt):
            enqueue("S", "scalar", description=str(statement.var))
            visit(
                statement.body,
                {**bindings, statement.var: integer(statement.value, bindings)},
                resource,
            )
        elif isinstance(statement, tir.Allocate):
            assert integer(statement.condition, bindings)
            visit(statement.body, bindings, resource)
        elif isinstance(statement, tir.DeclBuffer):
            visit(statement.body, bindings, resource)
        elif isinstance(statement, tir.AttrStmt):
            if statement.attr_key == "thread_extent":
                tag = statement.node.thread_tag
                assert tag in ("blockIdx.x", "blockIdx.y"), tag
                index = cid if tag == "blockIdx.x" else vid
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
            assert isinstance(call, tir.Call), f"unexpected executable evaluation: {call}"
            name = _operation(call)
            if name == "tl.loop_break":
                raise LoopBreak
            assert resource == 1, f"Vector operation escaped its resource: {name}"
            if name in (
                "tl.ascend_auto_set_flag",
                "tl.ascend_auto_wait_flag",
                "tl.ascend_set_flag",
                "tl.ascend_wait_flag",
            ):
                if name.startswith("tl.ascend_auto_"):
                    first, second = call.args[0].value.split("_")
                    event_id = call.args[1]
                else:
                    first, second = (argument.value for argument in call.args[:2])
                    event_id = call.args[2]
                assert isinstance(event_id, tir.IntImm), "event dispatch must use literal IDs"
                assert 0 <= int(event_id) < 8
                event = (first, second, int(event_id))
                kind = "set" if name.endswith("set_flag") else "wait"
                enqueue(first if kind == "set" else second, kind, event=event)
            elif name == "tl.ascend_auto_barrier":
                enqueue(call.args[0].value.removeprefix("PIPE_"), "barrier")
            elif name == "copy_gm_to_ub":
                computations[name] += 1
                enqueue(
                    "MTE2",
                    "copy",
                    writes=ub_bytes(call.args[2], bindings),
                    global_elements=gm_elements(call.args[1], source, bindings),
                )
            elif name == "tl.ascend_abs":
                assert integer(call.args[2], bindings) == _WIDTH
                computations[name] += 1
                enqueue(
                    "V",
                    "abs",
                    reads=ub_bytes(call.args[1], bindings),
                    writes=ub_bytes(call.args[0], bindings),
                )
            else:
                assert name == "copy_ub_to_gm", name
                computations[name] += 1
                enqueue(
                    "MTE3",
                    "store",
                    reads=ub_bytes(call.args[1], bindings),
                    global_elements=gm_elements(call.args[2], output, bindings),
                )
        else:
            raise AssertionError(f"unsupported executable shell: {type(statement).__name__}")

    visit(function.body, {})
    return _Program(tuple(tuple(queues[pipe]) for pipe in _PIPES), computations)


class _Hazard(AssertionError):
    pass


@dataclass
class _Execution:
    outputs: dict
    computations: Counter
    stores_per_byte: Counter
    trace: tuple[str, ...]


def _run_schedule(program: _Program, *, priority=None, seed=None) -> _Execution:
    queues = program.queues
    positions = [0] * len(_PIPES)
    clocks = [[0] * len(_PIPES) for _ in _PIPES]
    tokens, writers, readers, contents, outputs = {}, {}, {}, {}, {}
    computations, stores_per_byte = Counter(), Counter()
    randomizer = random.Random(seed)
    trace = []

    def before(previous, current, description):
        if previous is not None and not all(a <= b for a, b in zip(previous, current)):
            raise _Hazard(f"UB byte dependency missing: {description}")

    def read(command, now):
        for address in command.reads:
            if address not in writers:
                raise _Hazard(f"UB read before initialization: {address}")
            before(writers[address], now, f"RAW at {address} in {command.kind}")
            readers.setdefault(address, {})[command.pipe] = now
        values = []
        for first in range(0, len(command.reads), 4):
            lane = command.reads[first : first + 4]
            value = contents[lane[0]]
            assert all(contents[address] == value for address in lane)
            values.append(value)
        return values

    def write(command, now, values):
        assert len(command.writes) == len(values) * 4
        for index, address in enumerate(command.writes):
            before(writers.get(address), now, f"WAW at {address} in {command.kind}")
            for reader in readers.pop(address, {}).values():
                before(reader, now, f"WAR at {address} in {command.kind}")
            writers[address] = now
            contents[address] = values[index // 4]

    total = sum(map(len, queues))
    for _ in range(total):
        order = list(priority) if priority is not None else list(range(len(_PIPES)))
        if priority is None:
            randomizer.shuffle(order)
        for pipe_index in order:
            if positions[pipe_index] == len(queues[pipe_index]):
                continue
            command = queues[pipe_index][positions[pipe_index]]
            if command.kind != "wait" or command.event in tokens:
                break
        else:
            raise _Hazard(f"deadlock: all remaining destination waits are blocked; {trace[-12:]}")
        positions[pipe_index] += 1
        clock = clocks[pipe_index]
        if command.kind == "wait":
            clock[:] = [max(a, b) for a, b in zip(clock, tokens.pop(command.event))]
        clock[pipe_index] += 1
        now = tuple(clock)
        trace.append(str(command))
        if command.kind == "set":
            if command.event in tokens:
                raise _Hazard(f"double set with undrained token: {command.event}; {trace[-12:]}")
            tokens[command.event] = now
        elif command.kind == "copy":
            write(command, now, [-(element + 1) for element in command.global_elements])
            computations["copy_gm_to_ub"] += 1
        elif command.kind == "abs":
            values = read(command, now)
            write(command, now, [abs(value) for value in values])
            computations["tl.ascend_abs"] += 1
        elif command.kind == "store":
            values = read(command, now)
            for element, value in zip(command.global_elements, values):
                if value != element + 1:
                    raise _Hazard(f"wrong A row/tile reached O element {element}: {value}")
                if element in outputs:
                    raise _Hazard(f"duplicate output element: {element}")
                outputs[element] = value
            stores_per_byte.update(command.reads)
            computations["copy_ub_to_gm"] += 1
        else:
            assert command.kind in ("scalar", "barrier", "wait")
    assert positions == list(map(len, queues))
    if tokens:
        raise _Hazard(f"leftover undrained event tokens: {tokens}")
    assert computations == program.computations
    return _Execution(outputs, computations, stores_per_byte, tuple(trace))


def _expected_elements(rows: int, tiles: int, cid: int, vid: int) -> set[int]:
    return {
        ((vid * rows + row) * tiles + tile) * _WIDTH + lane
        for row in range(cid, rows, _WORKERS)
        for tile in range(tiles)
        for lane in range(_WIDTH)
    }


def _without_pairs(program: _Program, removed) -> _Program:
    queues = tuple(tuple(command for command in queue if command.event not in removed) for queue in program.queues)
    return _Program(queues, program.computations)


def _bad_schedule(program: _Program):
    for priority in permutations(range(len(_PIPES))):
        try:
            _run_schedule(program, priority=priority)
        except _Hazard as error:
            return str(error)
    for seed in range(24):
        try:
            _run_schedule(program, seed=seed)
        except _Hazard as error:
            return str(error)
    return None


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.fixture(scope="module")
def persistent_pipelines():
    result = {}
    for rows, tiles in _CASES:
        for planning in (False, True):
            before, _ = _pre_sync(_persistent_pipelined_abs(rows, tiles), memory_planning=planning)
            result[rows, tiles, planning] = (before, _run(before, target_kind="llvm"))
    return result


@pytest.mark.parametrize("rows,tiles", _CASES)
@pytest.mark.parametrize("memory_planning", [False, True])
def test_persistent_sip_retains_physical_layout_and_true_inner_period(
    persistent_pipelines, rows: int, tiles: int, memory_planning: bool
) -> None:
    before, after = persistent_pipelines[rows, tiles, memory_planning]
    assert _has_break(before.body) and not _has_break(after.body)
    assert len(before.attrs[_RINGS]) == len(before.attrs[_OWNERS]) == 1
    logical, record = next(iter(before.attrs[_RINGS].items()))
    assert [int(record[index]) for index in (0, 1, 3, 4, 5, 6)] == [1, 2, 0, 2048, 32, 0]
    assert int(before.attrs["buffer_versions"][logical]) == 2
    assert int(before.attrs["size_map"][record[2]]) == 512
    assert int(before.attrs["address_map"][record[2]]) == 0
    assert sorted(int(value) for value in before.attrs["size_map"].values()) == [256, 512]
    assert sorted(int(value) for value in before.attrs["address_map"].values()) == [0, 512]
    assert [int(value) for value in next(iter(before.attrs[_OWNERS].values()))] == [1, 2, 0, 3]
    for name in ("address_map", "size_map", "buffer_versions", _RINGS, _OWNERS):
        expected, actual = before.attrs[name], after.attrs[name]
        assert len(expected) == len(actual)
        for key, value in expected.items():
            assert key in actual
            tvm.ir.assert_structural_equal(actual[key], value)
    for name in (_EXTENT, _CLOCK):
        expected, actual = _attributes(before, name), _attributes(after, name)
        assert len(expected) == len(actual)
        for old, new in zip(expected, actual):
            tvm.ir.assert_structural_equal(old.node, new.node)
            tvm.ir.assert_structural_equal(old.value, new.value)
    assert len(_nodes(before.body, tir.Allocate)) == len(_nodes(after.body, tir.Allocate)) == 2
    assert not any(
        _operation(call) in ("copy_gm_to_ub", "tl.ascend_abs", "copy_ub_to_gm")
        for branch in _nodes(after.body, tir.IfThenElse)
        for call in _nodes(branch, tir.Call)
    ), "the normalized static computation must not remain under a semantic If"
    assert len(_nodes(after.body, tir.For)) == 1 + (tiles > 3), (
        "retain actual outer/steady loops; Simplify removes the N3 extent-one steady For"
    )
    assert not any(var.name.startswith("__ascend_if_proxy_") for var in _nodes(after.body, tir.Var))
    tvm.ir.assert_structural_equal(after, _run(after, target_kind="llvm"))


@pytest.mark.parametrize("rows,tiles", _CASES)
@pytest.mark.parametrize("memory_planning", [False, True])
def test_real_persistent_sip_async_queues_preserve_all_worker_bytes(
    persistent_pipelines, rows: int, tiles: int, memory_planning: bool
) -> None:
    _, after = persistent_pipelines[rows, tiles, memory_planning]
    coverage = Counter()
    for cid in range(_WORKERS):
        waves = len(range(cid, rows, _WORKERS))
        for vid in range(2):
            program = _commands(after, cid, vid, rows, tiles)
            expected = _expected_elements(rows, tiles, cid, vid)
            assert program.computations == {
                "copy_gm_to_ub": waves * tiles,
                "tl.ascend_abs": waves * tiles,
                "copy_ub_to_gm": waves * tiles,
            }
            seals = Counter((command.kind, command.event) for queue in program.queues for command in queue if command.event in _SEALS)
            assert seals == Counter({(kind, event): waves for event in _SEALS for kind in ("set", "wait")}), (
                "every executed wave requires exactly one complete six-pair seal"
            )
            for priority in _ADVERSARIAL:
                execution = _run_schedule(program, priority=priority)
                assert set(execution.outputs) == expected
                # All stores read the same B1 result bytes. The oracle must
                # retain their MTE3 lifetime across inner and outer reuse.
                assert len(execution.stores_per_byte) == _WIDTH * 4
                assert set(execution.stores_per_byte.values()) == {waves * tiles}
            coverage.update(expected)
    assert coverage == Counter({element: 1 for element in range(2 * rows * tiles * _WIDTH)})


@pytest.mark.parametrize("memory_planning", [False, True])
def test_real_persistent_sip_survives_every_fifo_priority_and_random_delays(persistent_pipelines, memory_planning: bool) -> None:
    rows, tiles = 9, 5
    _, after = persistent_pipelines[rows, tiles, memory_planning]
    for cid in range(_WORKERS):
        for vid in range(2):
            program = _commands(after, cid, vid, rows, tiles)
            expected = _expected_elements(rows, tiles, cid, vid)
            for priority in permutations(range(len(_PIPES))):
                assert set(_run_schedule(program, priority=priority).outputs) == expected
            for seed in range(4):
                assert set(_run_schedule(program, seed=seed).outputs) == expected


@pytest.mark.parametrize("memory_planning", [False, True])
def test_async_oracle_exposes_removed_whole_wave_boundaries(persistent_pipelines, memory_planning: bool) -> None:
    _, after = persistent_pipelines[9, 5, memory_planning]
    program = _commands(after, 0, 0, 9, 5)
    present = {command.event for queue in program.queues for command in queue if command.event}
    assert present >= _SEALS
    for removed in (
        _SEALS,
        frozenset(event for event in _SEALS if event[0] == "S"),
        frozenset(event for event in _SEALS if event[1] == "S"),
    ):
        witness = _bad_schedule(_without_pairs(program, removed))
        assert witness is not None, f"removing {sorted(removed)} preserved every tested schedule"
        assert "dependency missing" in witness or "undrained" in witness, witness
    # A single dataflow can make individual seal directions redundant. The
    # independent wave-seal primitive tests isolate all six pairs separately.
    for event in sorted(_SEALS):
        witness = _bad_schedule(_without_pairs(program, {event}))
        if witness is not None:
            assert "dependency missing" in witness or "undrained" in witness, witness


def _rewrite_owner(function: tir.PrimFunc, rewrite) -> tir.PrimFunc:
    changed = False

    def visit(statement):
        nonlocal changed
        if isinstance(statement, tir.AttrStmt) and statement.attr_key == _EXTENT:
            assert not changed, "the fixture must contain exactly one owner"
            changed = True
            return tir.AttrStmt(statement.node, statement.attr_key, statement.value, rewrite(statement.body))
        return None

    body = tir.stmt_functor.ir_transform(function.body, visit, None, ["tir.AttrStmt"])
    assert changed
    return function.with_body(body)


def _append_outside_owner(function: tir.PrimFunc, statement: tir.Stmt) -> tir.PrimFunc:
    changed = False

    def visit(node):
        nonlocal changed
        if isinstance(node, tir.AttrStmt) and node.attr_key == _EXTENT:
            assert not changed
            changed = True
            return tir.SeqStmt([node, statement])
        return None

    body = tir.stmt_functor.ir_transform(function.body, visit, None, ["tir.AttrStmt"])
    assert changed
    return function.with_body(body)


@pytest.mark.parametrize("change", ["negative_offset", "ring_root", "small_size", "wide_order"])
def test_closed_owner_rejects_unproven_result_footprint_or_stage(persistent_pipelines, change: str) -> None:
    before, _ = persistent_pipelines[9, 5, False]
    versions = before.attrs["buffer_versions"]
    result = next(root for root, version in versions.items() if int(version) == 1)
    ring = next(iter(before.attrs[_RINGS].values()))[2]
    if change == "small_size":
        sizes = dict(before.attrs["size_map"].items())
        sizes[result] = tir.IntImm("int64", 128)
        malformed = before.with_attr("size_map", sizes)
    elif change == "wide_order":
        changed = False

        def stage(node):
            nonlocal changed
            if isinstance(node, tir.AttrStmt) and node.attr_key.endswith("_pipeline_stage") and not changed:
                changed = True
                fields = list(node.node)
                fields[3] = tir.IntImm("int64", int(fields[3]) + (1 << 32))
                return tir.AttrStmt(fields, node.attr_key, node.value, node.body)
            return None

        malformed = before.with_body(tir.stmt_functor.ir_transform(before.body, stage, None, ["tir.AttrStmt"]))
        assert changed
    else:
        changed = False

        def pointer(node):
            nonlocal changed
            if isinstance(node, tir.Call) and _operation(node) == "tl.ascend_abs" and not changed:
                changed = True
                arguments = list(node.args)
                destination = arguments[0]
                fields = list(destination.args)
                if change == "negative_offset":
                    fields[2] = tir.IntImm(fields[2].dtype, -64)
                else:
                    fields[1] = ring
                arguments[0] = tir.Call(destination.dtype, destination.op, fields)
                return tir.Call(node.dtype, node.op, arguments)
            return None

        malformed = before.with_body(tir.stmt_functor.ir_transform(before.body, None, pointer, ["tir.Call"]))
        assert changed
    with pytest.raises(tvm.error.TVMError, match="(?i)(Persistent|pipeline|footprint|stage)"):
        _run(malformed, target_kind="llvm")


@pytest.mark.parametrize(
    "change",
    [
        "multiple_owner",
        "duplicate_owner_body",
        "mutable",
        "user_flag",
        "external_user_flag",
        "extra_compute",
        "outside_async",
        "header",
    ],
)
def test_persistent_sip_rejects_unproven_outer_forms(persistent_pipelines, change: str) -> None:
    before, _ = persistent_pipelines[9, 5, False]
    if change == "multiple_owner":
        owners = dict(before.attrs[_OWNERS].items())
        owners[tir.Var("unrelated_owner", "int32")] = next(iter(owners.values()))
        malformed = before.with_attr(_OWNERS, owners)
    elif change == "duplicate_owner_body":
        # Reuse the identical Stmt object at two executable occurrences.
        # Identity-based scans must not mistake the two owner protocols for
        # a single occurrence or export flags through a shared anchor.
        owner = _attributes(before, _EXTENT)[0]
        malformed = _append_outside_owner(before, owner)
    elif change == "mutable":
        source = before.buffer_map[before.params[0]]
        read = tir.BufferLoad(source, [0] * len(source.shape))
        malformed = _rewrite_owner(before, lambda body: tir.LetStmt(tir.Var("mutable_saved", "float32"), read, body))
    elif change == "user_flag":
        flags = [
            tir.Evaluate(
                tir.Call(
                    "handle",
                    tvm.ir.Op.get(f"tl.ascend_{kind}_flag"),
                    [tir.StringImm("MTE2"), tir.StringImm("V"), tir.IntImm("int32", 7)],
                )
            )
            for kind in ("set", "wait")
        ]
        malformed = _rewrite_owner(before, lambda body: tir.SeqStmt([*flags, body]))
    elif change == "external_user_flag":
        # A complete forward pair outside the private owner is still a user
        # protocol. It cannot acquire the driver's private wave capability.
        flags = [
            tir.Evaluate(
                tir.Call(
                    "handle",
                    tvm.ir.Op.get(f"tl.ascend_{kind}_flag"),
                    [tir.StringImm("S"), tir.StringImm("MTE2"), tir.IntImm("int32", 1)],
                )
            )
            for kind in ("set", "wait")
        ]
        malformed = _append_outside_owner(before, tir.SeqStmt(flags))
    elif change == "outside_async":
        # Reuse the real prologue's full GM->UB operation, including actual
        # row/vid addresses. It is valid asynchronous hardware work, now
        # duplicated outside the injector-certified owner envelope.
        extra = next(
            statement
            for statement in _nodes(before.body, tir.Evaluate)
            if isinstance(statement.value, tir.Call) and _operation(statement.value) == "copy_gm_to_ub"
        )
        steady = next(loop for loop in _nodes(before.body, tir.For) if not _has_break(loop.body))
        assert not any(var.same_as(steady.loop_var) for var in _nodes(extra, tir.Var))
        malformed = _append_outside_owner(before, extra)
    elif change == "extra_compute":
        versions = before.attrs["buffer_versions"]
        result_root = next(root for root, version in versions.items() if int(version) == 1)
        result = tir.decl_buffer((_WIDTH,), "float32", data=result_root, scope="shared.ub")
        extra = tir.BufferStore(result, tir.const(0, "float32"), [0])

        def append(statement):
            if isinstance(statement, tir.For) and _has_break(statement.body):
                return tir.For(
                    statement.loop_var,
                    statement.min,
                    statement.extent,
                    statement.kind,
                    tir.SeqStmt([statement.body, extra]),
                    annotations=statement.annotations,
                )
            return None

        body = tir.stmt_functor.ir_transform(before.body, append, None, ["tir.For"])
        malformed = before.with_body(body)
    else:
        unknown = tir.Var("unproven_break", "bool")

        def change_header(statement):
            if isinstance(statement, tir.IfThenElse) and _has_break(statement.then_case):
                return tir.IfThenElse(tir.Or(statement.condition, unknown), statement.then_case, statement.else_case)
            return None

        body = tir.stmt_functor.ir_transform(before.body, change_header, None, ["tir.IfThenElse"])
        malformed = tir.PrimFunc([*before.params, unknown], body, before.ret_type, before.buffer_map, before.attrs)
    with pytest.raises(tvm.error.TVMError, match="(?i)(persistent|owner|pipeline|pure|scope|canonical|flag)"):
        _run(malformed, target_kind="llvm")


def test_dynamic_inner_pipeline_remains_outside_the_first_persistent_contract() -> None:
    before, _ = _pre_sync(_persistent_pipelined_abs(9, T.symbolic("dynamic_tiles")))
    with pytest.raises(tvm.error.TVMError, match="(?i)(persistent|static|dynamic|pipeline|outer)"):
        _run(before, target_kind="llvm")


@pytest.mark.parametrize("tiles", [3, 5])
@pytest.mark.parametrize("owners", ["keep", "missing", "empty", "all_owner_shells_removed"])
def test_divisible_persistent_without_break_certificate_cannot_use_generic_ring_path(owners: str, tiles: int) -> None:
    before, _ = _pre_sync(_persistent_pipelined_abs(rows=8, tiles=tiles))
    assert not _has_break(before.body)
    assert len(_nodes(before.body, tir.For)) == 1 + (tiles > 3)
    assert len(_attributes(before, _EXTENT)) == 1
    outer = _nodes(before.body, tir.For)[-1]
    assert _attributes(before.with_body(outer.body), _EXTENT)
    if owners == "missing":
        before = before.without_attr(_OWNERS)
    elif owners == "empty":
        before = before.with_attr(_OWNERS, {})
    elif owners == "all_owner_shells_removed":

        def remove_owner_shell(node):
            if isinstance(node, tir.AttrStmt) and node.attr_key.startswith("tl.ascend_physical_pipeline_"):
                return node.body
            return None

        before = before.without_attr(_OWNERS).with_body(
            tir.stmt_functor.ir_transform(before.body, None, remove_owner_shell, ["tir.AttrStmt"])
        )
        assert not _attributes(before, _EXTENT)
        assert _attributes(before, _CLOCK)
        assert any(int(version) == 2 for version in before.attrs["buffer_versions"].values())
    with pytest.raises(tvm.error.TVMError, match="(?i)(Persistent|pipeline|certif)"):
        _run(before, target_kind="llvm")


def test_generic_ring_cannot_reset_its_logical_clock_inside_a_real_loop() -> None:
    from test_ascend_upstream_sync_physical_rings import _function

    iteration = tir.Var("unmodelled_invocation", "int32")
    before = _function(clock=0, pointer_offset=0, loop_var=iteration)
    with pytest.raises(tvm.error.TVMError, match="(?i)(nested|reset|invocation)"):
        _run(before, target_kind="llvm")
