# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU protocol checks for the ordinary, non-pipelined GEMM K loop.

These tests lower a real kernel and interpret control flow without compiling
device source or executing a kernel. The legacy sync pass is not an oracle.
"""

from collections import Counter

import pytest

from tilelang import language as T
from tilelang import tvm

from test_ascend_pipeline_ring_provenance import _nodes, _pre_sync
from test_ascend_upstream_sync_composite import _events, _gemms, _RESERVED, _run
from test_ascend_upstream_sync_pipeline import _assert_balanced_events

tir = tvm.tir


def _ordinary_k_gemm(trip_count: int, with_barriers: bool = False):
    # Follow examples/gemm/example_gemm.py and _gemm_v0_kernel in the GEMM
    # language suite. The default isolates inserted sync; the barrier variant
    # also covers the original examples' manual synchronization unchanged.
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
                for k in T.serial(trip_count):
                    T.copy(A[0, k * k_l1], a_l1)
                    T.copy(B[k * k_l1, 0], b_l1)
                    if with_barriers:
                        T.barrier_all()
                    T.gemm_v0(a_l1, b_l1, c_l0, init=(k == 0), kL0Size=k_l1)
                    if with_barriers:
                        T.barrier_all()
                T.copy(c_l0, C[0, 0])

    return main


def _operation(call):
    name = getattr(call.op, "name", None)
    if name == "tir.call_extern":
        return call.args[0].value.split("<", 1)[0].removeprefix("tl::ascend::")
    return name


def _helper_events(call):
    # Derive the actual flattened N/K loop from the template. The helper primes
    # and drains both slots even for one step, including K < kL0Size.
    name, _, parameters = call.args[0].value.partition("<")
    fields = [field.strip() for field in parameters.removesuffix(">").split(",")]
    assert name == "gemm_v0" and len(fields) == 8
    element_bytes = {"half": 2, "bfloat16_t": 2, "float": 4}[fields[0]]
    assert fields[1] == "float"
    m, n, k = (int(field) for field in fields[2:5])
    transpose_a, transpose_b = fields[5:7]
    k_l0 = int(fields[7])
    assert min(m, n, k, k_l0) > 0
    assert transpose_a in ("true", "false") and transpose_b in ("true", "false")
    c0 = 32 // element_bytes
    k_stride = ((k_l0 + c0 - 1) // c0) * c0
    n_capacity = (32 * 1024) // (k_stride * element_bytes)
    assert n_capacity > 0
    n_tile = n
    if transpose_b == "false" and ((n + 15) // 16) * 16 > n_capacity:
        # Every uniform N subtile must start on both the source C0 and L0B
        # column-fractal boundaries. Derive the helper's step count while the
        # separate shared-layout probe checks the actual physical intervals.
        n_alignment = max(16, c0)
        candidates = range((n_capacity // n_alignment) * n_alignment, 0, -n_alignment)
        n_tile = next(candidate for candidate in candidates if n % candidate == 0)
    assert n % n_tile == 0
    steps = ((k + k_l0 - 1) // k_l0) * (n // n_tile)
    return _helper_event_sequence(steps)


def _single_step_helper_events():
    # Existing Cube SIP fixtures use one internal step per invocation.
    return _helper_event_sequence(1)


def _helper_event_sequence(steps):
    events = [
        ("set", "MTE2_MTE1", 0),
        ("wait", "MTE2_MTE1", 0),
        ("set", "FIX_M", 0),
        ("wait", "FIX_M", 0),
        ("set", "M_MTE1", 0),
        ("set", "M_MTE1", 1),
    ]
    for step in range(steps):
        slot = step % 2
        events.extend(
            [
                ("wait", "M_MTE1", slot),
                ("set", "MTE1_M", slot),
                ("wait", "MTE1_M", slot),
                ("set", "M_MTE1", slot),
            ],
        )
    events.extend(
        [
            ("wait", "M_MTE1", 0),
            ("wait", "M_MTE1", 1),
            ("set", "MTE1_MTE2", 0),
            ("wait", "MTE1_MTE2", 0),
            ("set", "M_FIX", 0),
            ("wait", "M_FIX", 0),
        ]
    )
    return events


class _CompletionOracle:
    """Check the audited acquire/completion contract with pipeline clocks.

    Only generated flag pairs transfer clocks between pipelines. The helper
    contributes its audited MTE1 acquire -> M completion edge; its lexical
    return never completes M in another pipeline. Whole-allocation A/B effects
    remain live until M completion, matching the conservative composite model.
    """

    _PIPES = ("MTE2", "MTE1", "M", "FIX")

    def __init__(self, *, hidden_pool=False):
        self.clocks = {pipe: [0] * len(self._PIPES) for pipe in self._PIPES}
        self.tokens = {}
        self.l1_writes = {}
        self.l1_read_completions = {}
        self.c_writes = {}
        self.c_reads = {}
        self.hidden_pool = hidden_pool
        self.hidden_completion = None

    def _step(self, pipe, incoming=None):
        clock = self.clocks[pipe]
        if incoming is not None:
            clock[:] = [max(own, other) for own, other in zip(clock, incoming)]
        clock[self._PIPES.index(pipe)] += 1
        return tuple(clock)

    @staticmethod
    def _require_before(previous, current, description):
        assert all(before <= after for before, after in zip(previous, current)), f"completion edge missing {description}"

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
        if pipe == "ALL":
            completed = [max(clock[index] for clock in self.clocks.values()) for index in range(len(self._PIPES))]
            for pipeline in self._PIPES:
                self._step(pipeline, completed)
            return
        assert pipe in self._PIPES
        self._step(pipe)

    def load_l1(self, call):
        destination = call.args[2].args[1]
        now = self._step("MTE2")
        if destination in self.l1_read_completions:
            self._require_before(self.l1_read_completions[destination], now, "before L1 overwrite")
        self.l1_writes[destination] = now

    def gemm(self, call):
        first, second, accumulator = (call.args[index].args[1] for index in (1, 2, 3))
        acquire = self._step("MTE1")
        if self.hidden_pool and self.hidden_completion is not None:
            self._require_before(self.hidden_completion, acquire, "before hidden L0 pool reuse")
        for operand in (first, second):
            assert operand in self.l1_writes
            self._require_before(self.l1_writes[operand], acquire, "before GEMM L1 read")
        if accumulator in self.c_writes:
            self._require_before(self.c_writes[accumulator], acquire, "before accumulating GEMM acquire")
        if accumulator in self.c_reads:
            self._require_before(
                self.c_reads[accumulator],
                acquire,
                "before overwriting a FIX source",
            )
        completion = self._step("M", acquire)
        for operand in (first, second):
            self.l1_read_completions[operand] = completion
        self.c_writes[accumulator] = completion
        self.hidden_completion = completion

    def store_c(self, call):
        source = call.args[1].args[1]
        assert source in self.c_writes
        now = self._step("FIX")
        self._require_before(self.c_writes[source], now, "before final FIX read")
        self.c_reads[source] = now

    def finish(self):
        assert not self.tokens, f"undrained generated completion flags: {self.tokens}"


def _execute_k_protocol(function, bindings=None, *, hidden_pool=False):
    events = []
    computations = Counter()
    clear_conditions = []
    analyzer = tvm.arith.Analyzer()
    completion_oracle = _CompletionOracle(hidden_pool=hidden_pool)

    def integer(expression, bindings):
        substitutions = {var: tir.const(value, var.dtype) for var, value in bindings.items()}
        result = analyzer.simplify(tir.stmt_functor.substitute(expression, substitutions))
        assert isinstance(result, tir.IntImm), f"unresolved protocol expression: {result}"
        return int(result.value)

    def visit(statement, bindings, resource=None):
        if isinstance(statement, tir.SeqStmt):
            for child in statement.seq:
                visit(child, bindings, resource)
        elif isinstance(statement, tir.For):
            assert statement.kind in (tir.ForKind.SERIAL, tir.ForKind.UNROLLED)
            minimum = integer(statement.min, bindings)
            extent = integer(statement.extent, bindings)
            assert extent >= 0
            for iteration in range(minimum, minimum + extent):
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
            assert isinstance(call, tir.Call)
            operation = _operation(call)
            assert resource == 0, f"Cube protocol escaped execution resource: {operation}"
            if operation in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"):
                assert isinstance(call.args[1], tir.IntImm), "event IDs must use literal dispatch"
                event_id = integer(call.args[1], bindings)
                assert 0 <= event_id < 8
                kind = "set" if operation.endswith("set_flag") else "wait"
                events.append((kind, call.args[0].value, event_id))
                completion_oracle.flag(kind, call.args[0].value, event_id)
            elif operation in ("tl.ascend_auto_barrier", "tl.ascend_pipe_barrier"):
                completion_oracle.barrier(call)
            else:
                assert operation in (
                    "copy_gm_to_l1",
                    "tl.ascend_gemm_v0",
                    "copy_l0c_to_gm",
                ), operation
                computations[operation] += 1
                if operation == "tl.ascend_gemm_v0":
                    completion_oracle.gemm(call)
                    clear_conditions.append(integer(call.args[4], bindings))
                    # Compose the audited internal protocol with actual generated
                    # flags, proving hidden reservations prevent collisions.
                    events.extend(_helper_events(call))
                elif operation == "copy_gm_to_l1":
                    completion_oracle.load_l1(call)
                else:
                    completion_oracle.store_c(call)
        else:
            raise AssertionError(f"unsupported protocol shell: {type(statement).__name__}")

    visit(function.body, bindings or {})
    completion_oracle.finish()
    return events, computations, clear_conditions


def _rewrite_loop(function, rewrite):
    return function.with_body(
        tir.stmt_functor.ir_transform(
            function.body,
            lambda node: rewrite(node) if isinstance(node, tir.For) else None,
            None,
            ["tir.For"],
        )
    )


@pytest.fixture(scope="module")
def ordinary_k_loops():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")
    functions = {count: _pre_sync(_ordinary_k_gemm(count))[0] for count in (2, 3)}
    # Preserve an actual For for the one-iteration seed/drain boundary. The
    # normal lowering pipeline simplifies a literal T.serial(1) away earlier.
    functions[1] = _rewrite_loop(functions[2], lambda loop: tir.For(loop.loop_var, loop.min, 1, loop.kind, loop.body))
    return functions


@pytest.mark.parametrize("trip_count", [1, 2, 3])
def test_ordinary_k_loop_preserves_calculation_and_balances_real_protocol(ordinary_k_loops, trip_count):
    original = ordinary_k_loops[trip_count]
    assert len(_nodes(original.body, tir.For)) == 1
    assert len(_gemms(original)) == 1
    assert int(_gemms(original)[0].args[3].args[4]) == 3
    assert not _events(original)
    result = _run(original, target_kind="llvm")
    assert len(_nodes(result.body, tir.For)) == 1
    assert len(_gemms(result)) == 1
    tvm.ir.assert_structural_equal(_gemms(result)[0], _gemms(original)[0])
    assert not result.attrs.get("tl.ascend_physical_ring_provenance")
    generated = _events(result)
    assert generated
    assert all(event_id not in _RESERVED.get(event, set()) for _, event, event_id in generated)
    # M -> MTE2 -> MTE1 transfers completion to both the next L1 refresh and
    # accumulating GEMM acquire. The clock oracle checks that actual chain;
    # an additional, redundant M -> MTE1 event is not required.
    if trip_count > 1:
        assert any(event == "M_MTE2" for _, event, _ in generated)
    assert any(event == "MTE2_MTE1" for _, event, _ in generated)
    assert any(event == "M_FIX" for _, event, _ in generated)
    print(f"trip_count={trip_count} generated_events={generated}")
    tokens = set()
    for _invocation in range(3):
        events, calculations, clear_conditions = _execute_k_protocol(result)
        assert calculations == {
            "copy_gm_to_l1": 2 * trip_count,
            "tl.ascend_gemm_v0": trip_count,
            "copy_l0c_to_gm": 1,
        }
        assert clear_conditions == [1] + [0] * (trip_count - 1)
        _assert_balanced_events(events, tokens)


@pytest.mark.parametrize("trip_count", [1, 2, 3])
def test_k_loop_preserves_explicit_barriers_and_protocol(trip_count):
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")
    original = _pre_sync(_ordinary_k_gemm(max(2, trip_count), with_barriers=True))[0]
    if trip_count == 1:
        original = _rewrite_loop(original, lambda loop: tir.For(loop.loop_var, loop.min, 1, loop.kind, loop.body))
    result = _run(original, target_kind="llvm")

    def original_calls(function):
        loop = _nodes(function.body, tir.For)[0]
        return [
            node.value
            for node in _nodes(loop.body, tir.Evaluate)
            if isinstance(node.value, tir.Call)
            and _operation(node.value) in ("copy_gm_to_l1", "tl.ascend_gemm_v0", "tl.ascend_pipe_barrier")
        ]

    before, after = original_calls(original), original_calls(result)
    assert [_operation(call) for call in before] == [
        "copy_gm_to_l1",
        "copy_gm_to_l1",
        "tl.ascend_pipe_barrier",
        "tl.ascend_gemm_v0",
        "tl.ascend_pipe_barrier",
    ]
    assert len(after) == len(before)
    for old, new in zip(before, after):
        tvm.ir.assert_structural_equal(old, new)
    tokens = set()
    for _ in range(3):
        events, calculations, clear_conditions = _execute_k_protocol(result)
        assert calculations == {"copy_gm_to_l1": 2 * trip_count, "tl.ascend_gemm_v0": trip_count, "copy_l0c_to_gm": 1}
        assert clear_conditions == [1] + [0] * (trip_count - 1)
        _assert_balanced_events(events, tokens)


@pytest.mark.parametrize("change", ["unknown_pipe", "nonliteral", "extra_argument", "fake_extern", "bad_auto_prefix"])
def test_k_loop_rejects_unvalidated_barriers(ordinary_k_loops, change):
    def rewrite(loop):
        operation = tvm.ir.Op.get("tl.ascend_pipe_barrier")
        arguments = [tir.StringImm("ALL")]
        if change == "unknown_pipe":
            arguments = [tir.StringImm("V")]
        elif change == "nonliteral":
            arguments = [tir.IntImm("int32", 0)]
        elif change == "extra_argument":
            arguments.append(tir.IntImm("int32", 0))
        elif change == "fake_extern":
            operation = tvm.ir.Op.get("tir.call_extern")
            arguments = [tir.StringImm("tl.ascend_pipe_barrier"), *arguments]
        elif change == "bad_auto_prefix":
            operation = tvm.ir.Op.get("tl.ascend_auto_barrier")
        barrier = tir.Evaluate(tir.Call("handle", operation, arguments))
        body = tir.SeqStmt([barrier, loop.body])
        return tir.For(loop.loop_var, loop.min, loop.extent, loop.kind, body)

    with pytest.raises(tvm.error.TVMError, match="(?i)(barrier|external For)"):
        _run(_rewrite_loop(ordinary_k_loops[2], rewrite))


@pytest.mark.parametrize("clear", [True, False])
def test_k_loop_preserves_the_programs_pure_clear_expression(ordinary_k_loops, clear):
    from test_ascend_upstream_sync_composite import _changed_call, _rewrite_gemm

    original = _rewrite_gemm(
        ordinary_k_loops[2],
        lambda call: tir.Evaluate(_changed_call(call, clear=tir.const(clear, "bool"))),
    )
    result = _run(original)
    tvm.ir.assert_structural_equal(_gemms(original)[0], _gemms(result)[0])
    assert int(_gemms(result)[0].args[3].args[4]) == 3


@pytest.mark.parametrize(
    "change",
    ["dynamic", "nonzero_min", "parallel"],
)
def test_k_loop_rejects_unvalidated_control_and_storage_forms(ordinary_k_loops, change):
    original = ordinary_k_loops[2]
    dynamic_extent = tir.Var("dynamic_extent", "int32")

    def rewrite(loop):
        minimum, extent, kind, body = loop.min, loop.extent, loop.kind, loop.body
        if change == "dynamic":
            extent = dynamic_extent
        elif change == "nonzero_min":
            minimum = tir.IntImm("int32", 1)
        elif change == "parallel":
            kind = tir.ForKind.PARALLEL
        return tir.For(loop.loop_var, minimum, extent, kind, body)

    original = _rewrite_loop(original, rewrite)
    if change == "dynamic":
        original = tir.PrimFunc(
            [*original.params, dynamic_extent],
            original.body,
            original.ret_type,
            original.buffer_map,
            original.attrs,
        )
    with pytest.raises(
        tvm.error.TVMError,
        match="(?i)(external For|invocation|GM-to-L1|refresh|nonwrapping)",
    ):
        _run(original)


@pytest.mark.parametrize("change", ["nested", "fix_inside", "partial_copy"])
def test_k_loop_preserves_supported_control_store_and_partial_refresh(ordinary_k_loops, change):
    original = ordinary_k_loops[2]
    final_store = next(
        node
        for node in _nodes(original.body, tir.Evaluate)
        if isinstance(node.value, tir.Call) and _operation(node.value) == "copy_l0c_to_gm"
    )

    def rewrite(loop):
        body = loop.body
        if change == "nested":
            body = tir.For(tir.Var("nested", "int32"), 0, 2, tir.ForKind.SERIAL, body)
        elif change == "fix_inside":
            body = tir.SeqStmt([body, final_store])
        else:
            first, *rest = list(body.seq)
            arguments = list(first.value.args)
            arguments[4] = tir.IntImm("int32", 8)
            body = tir.SeqStmt(
                [tir.Evaluate(tir.Call(first.value.dtype, first.value.op, arguments)), *rest],
            )
        return tir.For(loop.loop_var, loop.min, loop.extent, loop.kind, body)

    result = _run(_rewrite_loop(original, rewrite), target_kind="llvm")
    events, computations, _ = _execute_k_protocol(result, hidden_pool=True)
    _assert_balanced_events(events, set())
    assert computations["tl.ascend_gemm_v0"] == (4 if change == "nested" else 2)
    assert computations["copy_l0c_to_gm"] == (3 if change == "fix_inside" else 1)


@pytest.mark.parametrize("remove", ["set", "wait"])
def test_k_protocol_oracle_detects_missing_credits_and_drains(ordinary_k_loops, remove):
    events, _, _ = _execute_k_protocol(_run(ordinary_k_loops[3], target_kind="llvm"))
    corrupted = [event for event in events if event[0] != remove]
    with pytest.raises(AssertionError, match="wait without token|double set|undrained"):
        _assert_balanced_events(corrupted, set())


@pytest.mark.parametrize("event", ["M_MTE2", "MTE2_MTE1", "M_FIX"])
def test_completion_oracle_rejects_balanced_flags_with_a_missing_dependency(ordinary_k_loops, event):
    result = _run(ordinary_k_loops[3], target_kind="llvm")

    def remove_pair(statement):
        if isinstance(statement, tir.Evaluate) and isinstance(statement.value, tir.Call):
            call = statement.value
            if _operation(call) in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag") and call.args[0].value == event:
                return tir.Evaluate(0)
        return None

    corrupted = result.with_body(tir.stmt_functor.ir_transform(result.body, remove_pair, None, ["tir.Evaluate"]))
    with pytest.raises(AssertionError, match="completion edge missing"):
        _execute_k_protocol(corrupted)
