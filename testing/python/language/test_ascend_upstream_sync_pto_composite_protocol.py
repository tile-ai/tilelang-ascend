# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Actual TIR plus the audited A3 PTO gemm helper, executed on four FIFOs.

The helper expansion follows src/tl_templates/pto/common.h:145-232. Its return
does not transfer completion. Hardware flag tokens and actual byte accesses
alone establish the MTE2 -> MTE1 -> M -> FIX completion chain. No device code
is compiled or executed by this file.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import permutations

import pytest

from tilelang import tvm

from test_ascend_upstream_sync_pto_composite import _gemms, _run
from test_ascend_upstream_sync_pto_composite import pto_flat_gemms as _pto_flat_gemms_fixture

tir = tvm.tir
pto_flat_gemms = _pto_flat_gemms_fixture  # Register the shared module fixture with pytest.
_PIPES = ("MTE2", "MTE1", "M", "FIX")
_DIRECTIONS = ("MTE2_MTE1", "M_MTE1", "FIX_M", "MTE1_M", "MTE1_MTE2", "M_FIX")


@dataclass(frozen=True)
class _Command:
    pipe: str
    kind: str
    event: tuple[str, int] | None = None
    reads: tuple = ()
    writes: tuple = ()
    tag: tuple = ()
    origin: str = "tir"


class _ProtocolError(AssertionError):
    pass


def _helper_commands(emit, chunks, left, right, accumulator, *, legacy_protocol=False):
    """Expand the helper protocol, including its one physical L0 slot.

    The actual-IR oracle supplies every physical byte. The protocol-only
    oracle supplies one representative address per chunk, so it can check
    longer helper loops without claiming support in the compiler profile.
    ``legacy_protocol`` preserves the unsafe helper for negative regression.
    """

    def pair(event):
        source, destination = event.split("_")
        emit(source, "set", event=(event, 1), origin="pto")
        emit(destination, "wait", event=(event, 1), origin="pto")

    assert chunks > 0 and len(left) % chunks == len(right) % chunks == 0
    a_chunk, b_chunk = len(left) // chunks, len(right) // chunks
    hidden_a = tuple(("hidden.l0a", byte) for byte in range(a_chunk))
    hidden_b = tuple(("hidden.l0b", byte) for byte in range(b_chunk))
    pair("MTE2_MTE1")
    for step in range(chunks):
        pair("M_MTE1")
        if step != chunks - 1 and (legacy_protocol or step == 0):
            pair("FIX_M")
        # Each chunk overwrites the same L0A/L0B addresses. Its L1 source
        # advances, independently of event IDs and logical buffer versions.
        for source, destination, width, label in (
            (left, hidden_a, a_chunk, "a"),
            (right, hidden_b, b_chunk, "b"),
        ):
            source = source[step * width : (step + 1) * width]
            emit(
                "MTE1",
                "extract",
                reads=tuple((byte, (label,)) for byte in source),
                writes=destination,
                tag=(label, step),
                origin="pto",
            )
        pair("MTE1_M")
        reads = tuple((byte, ("a", step)) for byte in hidden_a)
        reads += tuple((byte, ("b", step)) for byte in hidden_b)
        if step:
            reads += tuple((byte, ("c", step - 1)) for byte in accumulator)
        emit("M", "mad", reads=reads, writes=accumulator, tag=("c", step), origin="pto")
        if legacy_protocol and step != chunks - 1:
            pair("MTE1_MTE2")
    pair("MTE1_MTE2")
    pair("M_FIX")


def _operation(call):
    name = call.op.name
    if name == "tir.call_extern":
        return call.args[0].value.split("<", 1)[0].removeprefix("tl::ascend::")
    return name


def _program(function, k, *, legacy_protocol=False):
    queues = {pipe: [] for pipe in _PIPES}
    addresses, sizes = function.attrs["address_map"], function.attrs["size_map"]
    analyzer = tvm.arith.Analyzer()
    calculations = Counter()

    def integer(expression, bindings):
        substitutions = {var: tir.const(value, var.dtype) for var, value in bindings.items()}
        value = analyzer.simplify(tir.stmt_functor.substitute(expression, substitutions))
        assert isinstance(value, tir.IntImm), value
        return int(value.value)

    def span(pointer, bindings):
        assert pointer.op.name == "tir.tvm_access_ptr"
        data, offset = pointer.args[1], integer(pointer.args[2], bindings)
        element_bytes = tvm.DataType(pointer.args[0].dtype).bits // 8
        extent = integer(pointer.args[3], bindings) * element_bytes
        base = integer(addresses[data], bindings) + offset * element_bytes
        assert offset >= 0 and offset * element_bytes + extent <= int(sizes[data])
        scope = str(data.type_annotation.storage_scope)
        return tuple((scope, byte) for byte in range(base, base + extent))

    def emit(pipe, kind, **fields):
        queues[pipe].append(_Command(pipe, kind, **fields))

    def helper(call, bindings):
        # These are the real first-profile template parameters. Both buffers
        # use one hidden 4096-byte slot, including the two-chunk K256 case.
        assert call.args[0].value == f"gemm_v0<half, float, 16, 16, {k}, false, false, 128>"
        assert integer(call.args[4], bindings) == 1
        assert integer(call.args[5], bindings) == 16
        left, right, accumulator = (span(call.args[index], bindings) for index in (1, 2, 3))
        assert len(left) == len(right) == 16 * k * 2
        assert len(accumulator) == 16 * 16 * 4
        # M=N=16 makes each full 128-K fractal band contiguous in both
        # actual L1 operands. Each hidden slot spans exactly 4096 bytes.
        _helper_commands(emit, k // 128, left, right, accumulator, legacy_protocol=legacy_protocol)

    def visit(statement, bindings, resource=None):
        if isinstance(statement, tir.SeqStmt):
            for child in statement.seq:
                visit(child, bindings, resource)
        elif isinstance(statement, tir.AttrStmt):
            if statement.attr_key == "thread_extent":
                assert statement.node.thread_tag in ("blockIdx.x", "blockIdx.y")
                bindings = {**bindings, statement.node.var: 0}
            elif statement.attr_key == "resource_scope":
                resource = integer(statement.value, bindings)
            visit(statement.body, bindings, resource)
        elif isinstance(statement, (tir.Allocate, tir.DeclBuffer)):
            visit(statement.body, bindings, resource)
        elif isinstance(statement, tir.LetStmt):
            visit(
                statement.body,
                {**bindings, statement.var: integer(statement.value, bindings)},
                resource,
            )
        elif isinstance(statement, tir.IfThenElse):
            branch = statement.then_case if integer(statement.condition, bindings) else statement.else_case
            if branch is not None:
                visit(branch, bindings, resource)
        elif isinstance(statement, tir.For):
            # A generated dispatch shell is interpreted, never treated as a
            # synchronization edge. Original external loops are gated out.
            for index in range(
                integer(statement.min, bindings),
                integer(statement.min + statement.extent, bindings),
            ):
                visit(statement.body, {**bindings, statement.loop_var: index}, resource)
        elif isinstance(statement, tir.Evaluate):
            call = statement.value
            if isinstance(call, tir.IntImm):
                return
            assert resource == 0
            operation = _operation(call)
            if operation in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"):
                event, event_id = call.args[0].value, integer(call.args[1], bindings)
                assert 0 <= event_id < 8
                source, destination = event.split("_")
                emit(
                    source if operation.endswith("set_flag") else destination,
                    "set" if operation.endswith("set_flag") else "wait",
                    event=(event, event_id),
                )
            elif operation == "tl.ascend_auto_barrier":
                emit(call.args[0].value.removeprefix("PIPE_"), "barrier")
            elif operation == "copy_gm_to_l1":
                calculations[operation] += 1
                pointer = call.args[2]
                gemm = _gemms(function)[0]
                label = "a" if pointer.args[1].same_as(gemm.args[1].args[1]) else "b"
                emit("MTE2", "load", writes=span(pointer, bindings), tag=(label,))
            elif operation == "tl.ascend_gemm_v0":
                calculations[operation] += 1
                helper(call, bindings)
            elif operation == "copy_l0c_to_gm":
                calculations[operation] += 1
                emit(
                    "FIX",
                    "store",
                    reads=tuple((byte, ("c", k // 128 - 1)) for byte in span(call.args[1], bindings)),
                )
            else:
                raise AssertionError(f"unmodelled operation: {operation}")
        else:
            raise AssertionError(f"unmodelled shell: {type(statement)}")

    visit(function.body, {})
    assert calculations == {"copy_gm_to_l1": 2, "tl.ascend_gemm_v0": 1, "copy_l0c_to_gm": 1}
    return tuple(tuple(queues[pipe]) for pipe in _PIPES)


def _execute(queues, priority):
    clocks = [[0] * 4 for _ in range(4)]
    positions = [0] * 4
    tokens, writers, readers = {}, {}, {}

    def before(previous, now, where):
        if not all(a <= b for a, b in zip(previous, now)):
            raise _ProtocolError(f"completion dependency missing: {where}")

    while any(position < len(queue) for position, queue in zip(positions, queues)):
        selected = None
        for index in priority:
            if positions[index] == len(queues[index]):
                continue
            command = queues[index][positions[index]]
            if command.kind != "wait" or command.event in tokens:
                selected = index
                break
        if selected is None:
            raise _ProtocolError(f"wait without available token: {tokens}")
        command = queues[selected][positions[selected]]
        clock = clocks[selected]
        if command.kind == "wait":
            incoming = tokens.pop(command.event)
            clock[:] = [max(a, b) for a, b in zip(clock, incoming)]
        clock[selected] += 1
        now = tuple(clock)
        if command.kind == "set":
            if command.event in tokens:
                raise _ProtocolError(f"double set: {command.event}")
            tokens[command.event] = now
        else:
            for address, expected in command.reads:
                if address not in writers:
                    raise _ProtocolError(f"read before initialized: {address}")
                previous, actual = writers[address]
                before(previous, now, f"before {command.kind} read")
                if actual != expected:
                    raise _ProtocolError(f"wrong physical L0/L1 epoch: {actual} != {expected}")
                readers.setdefault(address, {})[selected] = now
            for address in command.writes:
                if address in writers:
                    before(writers[address][0], now, f"before {command.kind} overwrite")
                for read in readers.get(address, {}).values():
                    before(read, now, "before hidden single-slot reuse")
                writers[address] = now, command.tag
                readers[address] = {}
        positions[selected] += 1
    if tokens:
        raise _ProtocolError(f"undrained event credits: {tokens}")


def _remove_direction(queues, direction):
    return tuple(tuple(command for command in queue if not command.event or command.event[0] != direction) for queue in queues)


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.mark.parametrize("k", [128, 256])
def test_actual_pto_gemm_single_slot_and_full_completion_protocol(pto_flat_gemms, k):
    after = _run(pto_flat_gemms[k], model="pto", target_kind="llvm")
    queues = _program(after, k)
    hidden = Counter((command.kind, command.event) for queue in queues for command in queue if command.origin == "pto" and command.event)
    counts = {
        "MTE2_MTE1": 1,
        "M_MTE1": k // 128,
        "FIX_M": int(k > 128),
        "MTE1_M": k // 128,
        "MTE1_MTE2": 1,
        "M_FIX": 1,
    }
    assert hidden == Counter({(kind, (direction, 1)): count for direction, count in counts.items() for kind in ("set", "wait") if count})
    assert set(counts) == set(_DIRECTIONS)
    # Repeated complete invocations never inherit helper credits. The model
    # does not grant completion at the lexical helper return.
    for _invocation in range(3):
        for priority in permutations(range(4)):
            _execute(queues, priority)


def _abstract_helper_program(chunks, *, legacy_protocol=False):
    """Protocol-only model; it grants no new compiler or shape support."""
    queues = {pipe: [] for pipe in _PIPES}

    def emit(pipe, kind, **fields):
        queues[pipe].append(_Command(pipe, kind, **fields))

    left = tuple(("l1.a", step) for step in range(chunks))
    right = tuple(("l1.b", step) for step in range(chunks))
    accumulator = (("l0.c", 0),)
    emit("MTE2", "load", writes=left, tag=("a",))
    emit("MTE2", "load", writes=right, tag=("b",))
    _helper_commands(emit, chunks, left, right, accumulator, legacy_protocol=legacy_protocol)
    emit("FIX", "store", reads=((accumulator[0], ("c", chunks - 1)),))
    return tuple(tuple(queues[pipe]) for pipe in _PIPES)


@pytest.mark.parametrize("chunks", [1, 2, 3, 4, 8])
def test_single_slot_helper_protocol_drains_with_longer_internal_loops(chunks):
    queues = _abstract_helper_program(chunks)
    for priority in permutations(range(4)):
        _execute(queues, priority)


@pytest.mark.parametrize("chunks,direction", [(2, "MTE1_MTE2"), (3, "FIX_M")])
def test_original_repeated_helper_credits_are_rejected(chunks, direction):
    queues = _abstract_helper_program(chunks, legacy_protocol=True)
    errors = []
    for priority in permutations(range(4)):
        try:
            _execute(queues, priority)
        except _ProtocolError as error:
            errors.append(str(error))
    assert any(f"double set: ('{direction}', 1)" in error for error in errors), errors


def test_original_k256_helper_fails_with_actual_tir_and_inserted_flags(pto_flat_gemms):
    after = _run(pto_flat_gemms[256], model="pto", target_kind="llvm")
    queues = _program(after, 256, legacy_protocol=True)
    errors = []
    for priority in permutations(range(4)):
        try:
            _execute(queues, priority)
        except _ProtocolError as error:
            errors.append(str(error))
    assert any("double set: ('MTE1_MTE2', 1)" in error for error in errors), errors


@pytest.mark.parametrize("k,direction", [(128, "MTE2_MTE1"), (128, "MTE1_M"), (128, "M_FIX"), (256, "M_MTE1")])
def test_async_oracle_catches_deleted_data_or_single_slot_completion_direction(pto_flat_gemms, k, direction):
    after = _run(pto_flat_gemms[k], model="pto", target_kind="llvm")
    broken = _remove_direction(_program(after, k), direction)
    # Remove the entire direction: the helper and generated handshakes can be
    # redundant. Requiring each individual pair here would invent necessity.
    failures = []
    for priority in permutations(range(4)):
        try:
            _execute(broken, priority)
        except _ProtocolError as error:
            failures.append(str(error))
    assert failures, f"removed {direction} but no adversarial FIFO schedule exposed it"
