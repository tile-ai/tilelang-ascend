# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Asynchronous admission model for the six-pair wave seal primitive.

This model has independent FIFO command queues, binary directed event slots
and completion clocks. All waves may already be enqueued: a wait targeting S
does not magically stop another pipe. Entry broadcasts must actually gate the
other queues. No compiler module, TileLang synchronization pass or device is
used here. Passing these tests does not integrate Persistent with SIP.
"""

from dataclasses import dataclass, replace
from itertools import permutations
import random

import pytest

_PIPES = ("S", "V", "MTE2", "MTE3")
_ACTIVE = _PIPES[1:]
_ZERO = (0, 0, 0, 0)
_SEAL = frozenset([*(("S", pipe, 0) for pipe in _ACTIVE), *((pipe, "S", 0) for pipe in _ACTIVE)])
_SENTINEL = -999


@dataclass(frozen=True)
class _Command:
    pipe: str
    kind: str
    wave: int
    event: tuple[str, str, int] | None = None

    def __str__(self):
        return f"wave{self.wave} {self.pipe} {self.kind} {self.event or ''}"


@dataclass(frozen=True)
class _State:
    positions: tuple[int, ...] = (0, 0, 0, 0)
    clocks: tuple[tuple[int, ...], ...] = (_ZERO, _ZERO, _ZERO, _ZERO)
    tokens: tuple[tuple[tuple[str, str, int], tuple[int, ...]], ...] = ()
    writer: tuple[int, ...] | None = None
    readers: tuple[tuple[int, ...] | None, ...] = (None, None, None, None)
    value: int | None = None
    outputs: tuple[int, ...] = ()


class _Hazard(AssertionError):
    pass


@dataclass(frozen=True)
class _Witness:
    reason: str
    trace: tuple[str, ...]


def _queues(waves, *, remove=frozenset(), credit=None):
    """Use the same one-UB copy/abs/store wave as the positive device litmus.

    ``credit=(producer,consumer)`` instead models an inner protocol initializer
    and a delayed drain. It has no memory operations and isolates event reuse
    hazards that a single-buffer dataflow can accidentally hide.
    """
    queues = {pipe: [] for pipe in _PIPES}

    def pair(source, destination, event_id, wave):
        event = (source, destination, event_id)
        if event not in remove:
            queues[source].append(_Command(source, "set", wave, event))
            queues[destination].append(_Command(destination, "wait", wave, event))

    for wave in range(waves):
        for pipe in _ACTIVE:
            pair("S", pipe, 0, wave)
        if credit is None:
            queues["MTE2"].append(_Command("MTE2", "copy", wave))
            pair("MTE2", "V", 1, wave)
            queues["V"].append(_Command("V", "abs", wave))
            pair("V", "MTE3", 1, wave)
            queues["MTE3"].append(_Command("MTE3", "store", wave))
        else:
            source, destination = credit
            for pipe in _ACTIVE:
                queues[pipe].append(_Command(pipe, "nop", wave))
            pair(source, destination, 2, wave)
        for pipe in _ACTIVE:
            pair(pipe, "S", 0, wave)
    return tuple(tuple(queues[pipe]) for pipe in _PIPES)


def _before(previous, current, description):
    if previous is not None and not all(a <= b for a, b in zip(previous, current)):
        raise _Hazard(f"UB byte dependency missing: {description}")


def _step(state, queues, pipe_index):
    position = state.positions[pipe_index]
    if position == len(queues[pipe_index]):
        return None
    command = queues[pipe_index][position]
    tokens = dict(state.tokens)
    clock = state.clocks[pipe_index]
    if command.kind == "wait":
        if command.event not in tokens:
            return None  # Destination FIFO really stalls, other pipes may run.
        clock = tuple(max(a, b) for a, b in zip(clock, tokens.pop(command.event)))
    now = tuple(value + int(index == pipe_index) for index, value in enumerate(clock))
    clocks = list(state.clocks)
    clocks[pipe_index] = now
    positions = list(state.positions)
    positions[pipe_index] += 1
    writer, readers = state.writer, list(state.readers)
    value, outputs = state.value, list(state.outputs)

    def read():
        if writer is None:
            raise _Hazard("UB read before first GM copy")
        _before(writer, now, "RAW")
        readers[pipe_index] = now

    def write():
        _before(writer, now, "WAW")
        for reader in readers:
            _before(reader, now, "WAR")
        return now, [None] * len(_PIPES)

    if command.kind == "set":
        if command.event in tokens:
            raise _Hazard(f"double set with undrained token: {command.event}")
        tokens[command.event] = now
    elif command.kind == "copy":
        writer, readers = write()
        value = -(command.wave + 1)
    elif command.kind == "abs":
        read()
        writer, readers = write()
        value = abs(value)
    elif command.kind == "store":
        read()
        if value != command.wave + 1:
            raise _Hazard(f"wrong wave reached output: {value} for wave {command.wave}")
        outputs[command.wave] = value
    else:
        assert command.kind in ("nop", "wait")
    return replace(
        state,
        positions=tuple(positions),
        clocks=tuple(clocks),
        tokens=tuple(sorted(tokens.items())),
        writer=writer,
        readers=tuple(readers),
        value=value,
        outputs=tuple(outputs),
    )


def _finished(state, queues):
    return all(position == len(queue) for position, queue in zip(state.positions, queues))


def _explore(queues, waves, *, memory=True):
    """Exhaust every reachable FIFO interleaving; return an actual bad trace."""
    initial = _State(outputs=(_SENTINEL,) * waves if memory else ())
    pending = [(initial, ())]
    visited = {initial}
    terminals = 0
    while pending:
        state, trace = pending.pop()
        if _finished(state, queues):
            if state.tokens:
                return _Witness(f"leftover event tokens: {state.tokens}", trace), len(visited)
            if memory:
                assert state.outputs == tuple(range(1, waves + 1))
            terminals += 1
            continue
        ready = False
        for index, queue in enumerate(queues):
            if state.positions[index] == len(queue):
                continue
            command = queue[state.positions[index]]
            try:
                successor = _step(state, queues, index)
            except _Hazard as error:
                return _Witness(str(error), (*trace, str(command))), len(visited)
            if successor is None:
                continue
            ready = True
            if successor not in visited:
                visited.add(successor)
                assert len(visited) <= 200_000, "unexpected queue-state explosion"
                pending.append((successor, (*trace, str(command))))
        if not ready:
            return _Witness("deadlock: all remaining destination waits are blocked", trace), len(visited)
    assert terminals, "no completed schedule was explored"
    return None, len(visited)


def _run_schedule(queues, waves, *, priority=None, seed=None):
    state = _State(outputs=(_SENTINEL,) * waves)
    randomizer = random.Random(seed)
    executed = 0
    while not _finished(state, queues):
        order = list(priority) if priority is not None else list(range(len(_PIPES)))
        if priority is None:
            randomizer.shuffle(order)
        for index in order:
            successor = _step(state, queues, index)
            if successor is not None:
                state = successor
                executed += 1
                break
        else:
            raise _Hazard("deadlock: all remaining destination waits are blocked")
    assert not state.tokens
    assert state.outputs == tuple(range(1, waves + 1))
    assert executed == sum(map(len, queues))


@pytest.mark.parametrize("waves", [0, 1, 3])
def test_six_pair_seal_is_safe_for_every_small_wave_interleaving(waves):
    witness, states = _explore(_queues(waves), waves)
    assert witness is None, witness
    assert states >= 1


@pytest.mark.parametrize("waves", [0, 1, 3, 33])
def test_six_pair_seal_survives_every_pipe_priority_and_long_reuse(waves):
    queues = _queues(waves)
    for order in permutations(range(len(_PIPES))):
        _run_schedule(queues, waves, priority=order)
    for seed in range(12):
        _run_schedule(queues, waves, seed=seed)


def test_removing_all_seals_exposes_cross_wave_alias_or_credit_reuse():
    witness, _ = _explore(_queues(3, remove=_SEAL), 3)
    assert witness is not None
    assert "dependency missing" in witness.reason or "undrained token" in witness.reason


@pytest.mark.parametrize("pipe", _ACTIVE)
@pytest.mark.parametrize("boundary", ["entry", "exit"])
def test_each_pair_closes_a_real_initializer_drain_race(pipe, boundary):
    other = next(candidate for candidate in _ACTIVE if candidate != pipe)
    credit = (pipe, other) if boundary == "entry" else (other, pipe)
    complete = _queues(2, credit=credit)
    witness, _ = _explore(complete, 2, memory=False)
    assert witness is None, witness

    event = ("S", pipe, 0) if boundary == "entry" else (pipe, "S", 0)
    broken = _queues(2, credit=credit, remove={event})
    witness, _ = _explore(broken, 2, memory=False)
    assert witness is not None, f"removing {event} unexpectedly sealed every schedule"
    assert "undrained token" in witness.reason, witness
    assert witness.trace[-1].endswith(str((*credit, 2))), witness
