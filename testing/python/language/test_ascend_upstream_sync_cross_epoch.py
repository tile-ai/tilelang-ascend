# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Independent admission and completion checks for reused C/V channels."""

from collections import Counter, defaultdict, deque

import pytest

from tilelang import tvm

from test_ascend_upstream_sync_cross_core import _copy, _intrinsic, _run
from test_ascend_upstream_sync_composite import _run as _run_model
from test_ascend_upstream_sync_cross_core_owner_tree import (
    _actor_calls as _base_actor_calls,
    _call_name,
    _dma,
    _semantic_leaves,
)

tir = tvm.tir
_LOCAL_FLAGS = {"tl.ascend_set_flag", "tl.ascend_wait_flag"}
_AUTO_FLAGS = {"tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"}


@pytest.fixture(autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


def _local_flag(kind, event_id=6):
    # The directed pair and numeric ID deliberately coincide on C and V.
    return _intrinsic(f"tl.ascend_{kind}_flag", "S", "MTE2", event_id)


def _epoch_program(
    *,
    groups=2,
    epochs=3,
    depth=1,
    mutation=None,
    vector_only_handoff=False,
    free_pipe=None,
    readonly_input=False,
    empty_controls=False,
):
    """A ready/free ring with real GM RAW and WAW reuse.

    The L0C is an abstract already-computed value, as in the existing closed
    publication fixtures. The device companion supplies a real GEMM instead.
    """
    rows, columns = 32, 16
    vector_rows = rows // 2
    shape = (groups * depth * rows, columns)
    first = tir.decl_buffer(shape, "float32", name="input")
    shared = tir.decl_buffer(shape, "float32", name="workspace")
    output = tir.decl_buffer(shape, "float32", name="output")
    accumulator = tir.decl_buffer(
        (rows, columns),
        "float32",
        name="accumulator",
        scope="wmma.accumulator",
    )
    l1 = tir.decl_buffer((rows, columns), "float32", name="l1", scope="shared.l1")
    local = tir.decl_buffer((vector_rows, columns), "float32", name="local", scope="shared.ub")
    side = tir.decl_buffer(
        (groups * epochs * vector_rows, columns),
        "float32",
        name="vector_only_gm",
    )
    private = tir.decl_buffer(
        (vector_rows, columns),
        "float32",
        name="side_source",
        scope="shared.ub",
    )
    received = tir.decl_buffer(
        (vector_rows, columns),
        "float32",
        name="side_target",
        scope="shared.ub",
    )
    bx, vid = tir.Var("group", "int32"), tir.Var("subcore", "int32")
    ci, vi = tir.Var("c_epoch", "int32"), tir.Var("v_epoch", "int32")
    pitch = 0 if mutation == "overlapping_groups" else depth * rows * columns
    c_offset = bx * pitch + (ci % depth) * rows * columns
    vector_offset = 0 if mutation == "overlapping_subcores" else vid * vector_rows * columns
    vector_epoch = vi + 1 if mutation == "wrong_slot" else vi
    v_offset = bx * pitch + (vector_epoch % depth) * rows * columns + vector_offset
    free_id, ready_id = 0, 1
    scalar_parameters = []
    if mutation == "dynamic_cross_id":
        ready_id = tir.Var("ready_id", "int32")
        scalar_parameters.append(ready_id)

    def cross_set(pipe, channel):
        return _intrinsic("tl.ascend_set_cross_flag", pipe, channel, 2)

    def cross_wait(channel):
        return _intrinsic("tl.ascend_wait_cross_flag", channel, "MTE2")

    def empty_protocol_controls(iteration):
        other = tir.Var(f"empty_{iteration.name}", "int32")

        def pair():
            return tir.SeqStmt([_local_flag("set", 7), _local_flag("wait", 7)])

        return [
            tir.IfThenElse(iteration % 2 == 0, pair(), None),
            tir.For(other, 0, 2, tir.ForKind.SERIAL, pair()),
        ]

    def cube_read(source):
        return _dma(
            f"tl::ascend::copy_gm_to_l1<float, {rows}, {columns}>",
            source,
            l1,
            columns,
            rows,
            columns,
            rows,
            columns,
            source_offset=c_offset,
            extent=rows * columns,
        )

    cube_write = _dma(
        f"tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, {rows}, {columns}, 0>",
        accumulator,
        shared,
        columns,
        rows,
        columns,
        rows,
        columns,
        0,
        0,
        destination_offset=c_offset,
        extent=rows * columns,
    )
    cube_source = cube_read(first)
    if not readonly_input:
        cube_source = tir.IfThenElse(ci < depth, cube_source, cube_read(shared))
    cube_payload = [
        cross_wait(free_id),
        _local_flag("wait"),
        cube_source,
        cube_write,
        _local_flag("set"),
        cross_set("S" if mutation == "scalar_cross_set" else "FIX", ready_id),
    ]
    if empty_controls:
        cube_payload[2:2] = empty_protocol_controls(ci)
    if mutation == "hardware_owner_conflict":
        wrong = _intrinsic(
            "tl.ascend_abs",
            local.access_ptr("w"),
            local.access_ptr("r"),
            vector_rows * columns,
        )
        cube_payload.insert(2, tir.IfThenElse(ci == 0, wrong, None))
    if mutation == "missing_free_wait":
        cube_payload.pop(0)
    if mutation == "late_free_wait":
        cube_payload.insert(-1, cube_payload.pop(0))
    if mutation == "early_ready":
        cube_payload.insert(2, cube_payload.pop())
    if mutation == "no_gm_publication":
        cube_payload = [s for s in cube_payload if not s.same_as(cube_write)]
    if mutation in ("gm_dependent_control", "unknown_scalar_control"):
        if mutation == "unknown_scalar_control":
            predicate = tir.Var("run_epoch", "int32")
            scalar_parameters.append(predicate)
            condition = predicate != 0
        else:
            condition = tir.BufferLoad(first, [0, 0]) > tir.FloatImm("float32", 0)
        cube_payload = [
            tir.IfThenElse(
                condition,
                tir.SeqStmt(cube_payload),
                None,
            )
        ]
    cube_body = cube_payload[0] if len(cube_payload) == 1 else tir.SeqStmt(cube_payload)
    cube = [
        _local_flag("set"),
        tir.For(ci, 0, epochs, tir.ForKind.SERIAL, cube_body),
        *[cross_wait(free_id) for _ in range(depth)],
        _local_flag("wait"),
    ]
    if mutation in ("missing_drain", "asymmetric_terminal_release"):
        del cube[-2]
    if mutation == "nonuniform_terminal_drain":
        cube[-2] = tir.IfThenElse(bx == 0, cube[-2], None)
    if mutation == "missing_all_drains":
        cube = [*cube[:2], cube[-1]]
    vector_payload = [
        cross_wait(ready_id),
        _local_flag("wait"),
        _dma(
            f"tl::ascend::copy_gm_to_ub<float, {columns}, {vector_rows}>",
            shared,
            local,
            columns,
            vector_rows,
            columns,
            tir.const(0, "float32"),
            vector_rows,
            columns,
            source_offset=v_offset,
            extent=vector_rows * columns,
        ),
        _intrinsic(
            "tl.ascend_abs",
            local.access_ptr("w"),
            local.access_ptr("r"),
            vector_rows * columns,
        ),
    ]
    if empty_controls:
        vector_payload[2:2] = empty_protocol_controls(vi)
    for destination in (shared, output):
        vector_payload.append(
            _dma(
                f"tl::ascend::copy_ub_to_gm<float, {columns}, {vector_rows}>",
                local,
                destination,
                columns,
                vector_rows,
                columns,
                destination_offset=v_offset,
                extent=vector_rows * columns,
            )
        )
    if mutation == "cross_l1_alias":
        l1_view = tir.decl_buffer(
            (vector_rows, columns),
            "float32",
            data=l1.data,
            scope="shared.l1",
        )
        vector_payload.append(_copy("copy_ub_to_l1", local, l1_view))
    vector_prefix = []
    if vector_only_handoff:
        side_load = _dma(
            f"tl::ascend::copy_gm_to_ub<float, {columns}, {vector_rows}>",
            first,
            private,
            columns,
            vector_rows,
            columns,
            tir.const(0, "float32"),
            vector_rows,
            columns,
            source_offset=bx * pitch,
            extent=vector_rows * columns,
        )
        vector_prefix.append(tir.IfThenElse(vid == 0, side_load, None))
        side_read = _dma(
            f"tl::ascend::copy_gm_to_ub<float, {columns}, {vector_rows}>",
            side,
            received,
            columns,
            vector_rows,
            columns,
            tir.const(0, "float32"),
            vector_rows,
            columns,
            source_offset=(bx * epochs + vi - 1) * vector_rows * columns,
            extent=vector_rows * columns,
        )
        vector_payload.insert(2, tir.IfThenElse(tir.And(vid == 1, vi > 0), side_read, None))
        side_write = _dma(
            f"tl::ascend::copy_ub_to_gm<float, {columns}, {vector_rows}>",
            private,
            side,
            columns,
            vector_rows,
            columns,
            destination_offset=(bx * epochs + vi) * vector_rows * columns,
            extent=vector_rows * columns,
        )
        vector_payload.append(tir.IfThenElse(vid == 0, side_write, None))
    release_pipe = free_pipe or ("MTE2" if vector_only_handoff else "MTE3")
    vector_payload += [_local_flag("set"), cross_set(release_pipe, free_id)]
    if mutation == "uncovered_initial_gm":
        cube.insert(1, tir.stmt_functor.substitute(cube_write, {ci: tir.IntImm("int32", 0)}))
        vector_prefix.insert(
            0,
            tir.stmt_functor.substitute(vector_payload[2], {vi: tir.IntImm("int32", 0)}),
        )
    if mutation == "early_release":
        vector_payload.insert(2, vector_payload.pop())
    if mutation == "missing_release":
        vector_payload.pop()
    if mutation == "one_subcore_release":
        vector_payload[-1] = tir.IfThenElse(vid == 0, vector_payload[-1], None)
    if mutation == "asymmetric_terminal_release":
        release_guard = tir.Or(vi < epochs - 1, vid == 0)
        vector_payload[-1] = tir.IfThenElse(release_guard, vector_payload[-1], None)
    seeds = [cross_set(release_pipe, free_id) for _ in range(depth)]
    if mutation == "missing_seed":
        seeds = []
    if mutation == "extra_seed":
        seeds *= 2
    vector = [
        _local_flag("set"),
        *vector_prefix,
        *seeds,
        tir.For(
            vi,
            0,
            epochs - (mutation == "unequal_epochs"),
            tir.ForKind.SERIAL,
            tir.SeqStmt(vector_payload),
        ),
        _local_flag("wait"),
    ]
    body = tir.SeqStmt(
        [
            tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, tir.SeqStmt(cube)),
            tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, tir.SeqStmt(vector)),
        ]
    )
    allocations = (accumulator, l1, local)
    if vector_only_handoff:
        allocations += (private, received)
    for buffer in allocations:
        body = tir.Allocate(buffer.data, buffer.dtype, buffer.shape, tir.const(True), body)
    for variable, tag, extent in ((vid, "blockIdx.y", 2), (bx, "blockIdx.x", groups)):
        body = tir.AttrStmt(
            tir.IterVar(None, variable, tir.IterVar.ThreadIndex, tag),
            "thread_extent",
            extent,
            body,
        )
    buffers = (first, shared, output, side) if vector_only_handoff else (first, shared, output)
    sizes, addresses = {}, {}
    for index, buffer in enumerate(allocations):
        sizes[buffer.data] = int(buffer.shape[0]) * int(buffer.shape[1]) * 4
        addresses[buffer.data] = index * 4096 if buffer.scope() == "shared.ub" else 0
    function = (
        tir.PrimFunc(
            [buffer.data for buffer in buffers] + scalar_parameters,
            body,
            buffer_map={buffer.data: buffer for buffer in buffers},
        )
        .with_attr("tir.noalias", True)
        .with_attr("address_map", addresses)
        .with_attr("size_map", sizes)
    )
    if mutation == "unknown_alias":
        function = function.without_attr("tir.noalias")
    return function


def _coordinate_epoch_program(kind="if_then_else", *, mutation=None):
    function = _epoch_program()
    coordinates = []

    def find_group(node):
        if not isinstance(node, tir.AttrStmt) or node.attr_key != "thread_extent":
            return
        if node.node.thread_tag == "blockIdx.x":
            coordinates.append(node.node.var)

    tir.stmt_functor.post_order_visit(function.body, find_group)
    assert len(coordinates) == 1
    group = coordinates[0]
    condition = group % 2 == 0
    first, second = tir.IntImm("int32", 3), tir.IntImm("int32", 2)
    if mutation in ("selected_overflow", "unselected_overflow"):
        second = tir.Add(tir.IntImm("int32", (1 << 31) - 1), group)
        if mutation == "unselected_overflow":
            condition = group < 2
    if mutation == "unknown_call":
        if kind == "select":
            condition = group < 2
        second = tir.call_pure_extern("int32", "unknown_integer_function", group)
    if mutation == "data_dependent":
        source = function.buffer_map[function.params[0]]
        condition = tir.BufferLoad(source, [0, 0]) > tir.FloatImm("float32", 0)
    if mutation in ("guarded_division", "invalid_division"):
        condition = group == (0 if mutation == "guarded_division" else 1)
        second = tir.floordiv(first, group)
    if mutation == "condition_overflow":
        original_width_sum = tir.Add(tir.IntImm("int32", (1 << 31) - 1), group)
        condition = tir.GE(original_width_sum, tir.IntImm("int32", 0))
    expression = tir.if_then_else if kind == "if_then_else" else tir.Select
    cube_extent = expression(condition, first, second)
    if kind == "if_then_else":
        assert isinstance(cube_extent, tir.Call) and cube_extent.op.name == "tir.if_then_else"
    else:
        assert isinstance(cube_extent, tir.Select)
    vector_extent = cube_extent
    if mutation == "different_owner_counts":
        vector_extent = expression(condition, second, first)

    def replace(node):
        if not isinstance(node, tir.For):
            return None
        extent = {"c_epoch": cube_extent, "v_epoch": vector_extent}.get(node.loop_var.name)
        if extent is None:
            return None
        return tir.For(node.loop_var, node.min, extent, node.kind, node.body)

    return function.with_body(
        tir.stmt_functor.ir_transform(function.body, None, replace, ["tir.For"]),
    )


def _actor_calls(function, owner, group, subcore):
    # Specialize coordinate-only ITE before the generic arithmetic walker:
    # its simplifier otherwise folds division in an unreachable branch first.
    coordinates = {}

    def coordinate(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
            value = group if node.node.thread_tag == "blockIdx.x" else subcore
            coordinates[node.node.var] = tir.const(value, node.node.var.dtype)

    tir.stmt_functor.post_order_visit(function.body, coordinate)
    analyzer = tvm.arith.Analyzer()

    def selected(node):
        if not isinstance(node, tir.Call) or node.op.name != "tir.if_then_else":
            return None
        condition = analyzer.simplify(tir.stmt_functor.substitute(node.args[0], coordinates))
        if isinstance(condition, tir.IntImm):
            return node.args[1 if int(condition) else 2]
        return None

    specialized = function.with_body(
        tir.stmt_functor.ir_transform(function.body, selected, None, ["tir.Call"]),
    )
    return _base_actor_calls(specialized, owner, group, subcore)


def _epoch_byte_oracle(
    function,
    groups,
    epochs,
    *,
    vector_only_handoff=False,
    focus_root=None,
    total_epochs=None,
):
    """Issue independent actor streams and check actual asynchronous clocks.

    A cross wait may block its actor, so C and both AIV streams are scheduled
    independently. Events transmit pipe clocks; lexical adjacency is never
    treated as completion. GM bytes are shared, UB/L1/L0C bytes are actor local.
    """
    actors = []
    for group in range(groups):
        actors.extend((group, owner, subcore) for owner, subcore in ((0, 0), (1, 0), (1, 1)))
    pipes = ("S", "MTE1", "MTE2", "MTE3", "V", "M", "FIX")
    dimensions = [(actor, pipe) for actor in actors for pipe in pipes]
    dimension_index = {key: index for index, key in enumerate(dimensions)}
    clocks = {key: [0] * len(dimensions) for key in dimensions}
    completed = {key: tuple(value) for key, value in clocks.items()}
    blocked_issue = {actor: (0,) * len(dimensions) for actor in actors}
    streams = {}
    for actor in actors:
        streams[actor] = deque(_actor_calls(function, actor[1], actor[0], actor[2]))
    local_tokens, cross_tokens = {}, defaultdict(deque)
    writes, readers = {}, defaultdict(list)
    counts = Counter()
    gm_roots = {buffer.data for buffer in function.buffer_map.values()}
    inputs = {function.buffer_map[function.params[0]].data}
    initialized = set()

    def initialize_accumulator(node):
        if not isinstance(node, tir.Allocate):
            return
        if node.buffer_var.type_annotation.storage_scope == "wmma.accumulator":
            initialized.add(node.buffer_var)

    tir.stmt_functor.post_order_visit(function.body, initialize_accumulator)

    def step(actor, pipe, incoming=None):
        clock = clocks[actor, pipe]
        # CrossCoreWaitFlag blocks subsequent instruction issue on every pipe.
        # It does not complete operations already issued on unrelated pipes.
        clock[:] = [max(a, b) for a, b in zip(clock, blocked_issue[actor])]
        if incoming is not None:
            clock[:] = [max(a, b) for a, b in zip(clock, incoming)]
        clock[dimension_index[actor, pipe]] += 1
        return tuple(clock)

    def before(first, second, actor, root, address):
        detail = f"actor={actor}, root={root.name}, byte={address[-1]}"
        if not all(a <= b for a, b in zip(first, second)):
            raise AssertionError(f"missing asynchronous completion edge: {detail}")

    def span(actor, pointer, integer, rows=None, columns=None, stride=None):
        root = pointer.args[1]
        base, size = integer(pointer.args[2]), tvm.DataType(pointer.args[0].dtype).bits // 8
        elements = (
            range(base, base + integer(pointer.args[3]))
            if rows is None
            else (base + row * stride + column for row in range(rows) for column in range(columns))
        )
        storage = root if root in gm_roots else (actor, root)
        addresses = []
        for element in elements:
            addresses.extend((storage, element * size + byte) for byte in range(size))
        return root, addresses

    def access(actor, ticket, region, write):
        root, addresses = region
        if focus_root is not None and not root.same_as(focus_root):
            return
        for address in addresses:
            if address in writes:
                before(writes[address], ticket, actor, root, address)
            elif not write:
                assert root in inputs or root in initialized, "read of an unproduced byte"
            if write:
                for reader in readers.pop(address, []):
                    before(reader, ticket, actor, root, address)
                writes[address] = ticket
            else:
                readers[address].append(ticket)

    def advance(actor, call, integer):
        group, owner, subcore = actor
        name = _call_name(call)
        if name in _AUTO_FLAGS:
            event, event_id = call.args[0].value, integer(call.args[1])
            source, destination = event.split("_")
            key = (actor, source, destination, event_id)
            if name.endswith("wait_flag"):
                if key not in local_tokens:
                    return False
                step(actor, destination, local_tokens.pop(key))
            else:
                assert key not in local_tokens, "double set of a local token"
                local_tokens[key] = step(actor, source)
        elif name == "tl.ascend_auto_barrier":
            pipe = call.args[0].value.removeprefix("PIPE_")
            if pipe == "ALL":
                joined = [max(clocks[actor, p][i] for p in pipes) for i in range(len(dimensions))]
                for p in pipes:
                    completed[actor, p] = step(actor, p, joined)
            else:
                completed[actor, pipe] = step(actor, pipe)
        elif name.endswith("set_cross_flag"):
            step(actor, "S")
            if name == "tl.ascend_auto_set_cross_flag":
                mode = integer(call.args[0])
                pipe, channel = call.args[1].value, integer(call.args[2])
            else:
                pipe = call.args[0].value
                channel, mode = integer(call.args[1]), integer(call.args[2])
            assert mode == 2
            # CrossCoreSetFlag publishes after its selected pipe completes.
            # A real MTE3->MTE2 WaitFlag therefore contributes to Set(MTE2),
            # whereas an unrelated MTE3 instruction's lexical order cannot.
            release = step(actor, pipe, clocks[actor, "S"])
            completed[actor, pipe] = release
            targets = [(group, 1, v) for v in (0, 1)] if owner == 0 else [(group, 0, 0)]
            for target in targets:
                cross_tokens[target, channel, actor].append(release)
            counts["cross_set"] += 1
        elif name.endswith("wait_cross_flag"):
            channel = integer(call.args[0])
            producers = [(group, 0, 0)] if owner == 1 else [(group, 1, v) for v in (0, 1)]
            keys = [(actor, channel, producer) for producer in producers]
            if not all(cross_tokens[key] for key in keys):
                return False
            for key in keys:
                step(actor, "S", cross_tokens[key].popleft())
            blocked_issue[actor] = tuple(clocks[actor, "S"])
            counts["cross_wait"] += 1
        elif name in ("copy_gm_to_l1", "copy_l0c_to_gm", "copy_gm_to_ub", "copy_ub_to_gm"):
            fields = call.args[0].value.partition("<")[2].removesuffix(">").split(",")
            if name == "copy_l0c_to_gm":
                rows, columns = map(int, fields[-3:-1])
            elif name == "copy_gm_to_l1":
                rows, columns = map(int, fields[1:])
            else:
                columns, rows = int(fields[1]), int(fields[2]) if len(fields) == 3 else 1
            pipe = {"copy_l0c_to_gm": "FIX", "copy_ub_to_gm": "MTE3"}.get(name, "MTE2")
            source = span(actor, call.args[1], integer)
            destination = span(actor, call.args[2], integer)
            if name in ("copy_gm_to_l1", "copy_gm_to_ub"):
                source = span(actor, call.args[1], integer, rows, columns, integer(call.args[3]))
            else:
                destination = span(
                    actor,
                    call.args[2],
                    integer,
                    rows,
                    columns,
                    integer(call.args[3]),
                )
            ticket = step(actor, pipe)
            access(actor, ticket, source, False)
            access(actor, ticket, destination, True)
            counts[name] += 1
        elif name == "tl.ascend_abs":
            ticket = step(actor, "V")
            access(actor, ticket, span(actor, call.args[1], integer), False)
            access(actor, ticket, span(actor, call.args[0], integer), True)
            counts[name] += 1
        else:
            raise AssertionError(name)
        return True

    while any(streams.values()):
        progress = False
        for actor in actors:
            if not streams[actor]:
                continue
            call, integer = streams[actor][0]
            if advance(actor, call, integer):
                streams[actor].popleft()
                progress = True
        assert progress, "cross/local protocol deadlocked"
    assert not local_tokens, "undrained local event tokens"
    assert not any(cross_tokens.values()), "undrained cross-core tokens"
    total = groups * epochs if total_epochs is None else total_epochs
    assert counts["copy_l0c_to_gm"] == total
    assert counts["copy_gm_to_ub"] == (2 + vector_only_handoff) * total
    assert counts["copy_ub_to_gm"] == (4 + vector_only_handoff) * total
    return counts


@pytest.mark.parametrize(
    "groups,epochs,depth",
    [(1, 2, 1), (2, 3, 1), (3, 4, 1), (2, 5, 3), (1, 2, 15)],
)
def test_two_way_epochs_keep_dma_bytes_and_drain_every_actor(groups, epochs, depth):
    original = _epoch_program(groups=groups, epochs=epochs, depth=depth)
    result = _run(original)
    old, retained = _semantic_leaves(original), _semantic_leaves(result)
    # This helper excludes all sync instructions from the semantic projection.
    old = [call for call in old if call.op.name not in _LOCAL_FLAGS]
    assert len(old) == len(retained)
    for first, second in zip(old, retained):
        tvm.ir.assert_structural_equal(first, second)
    counts = _epoch_byte_oracle(result, groups, epochs)
    assert counts["tl.ascend_abs"] == groups * 2 * epochs
    assert not any(call.op.name in _LOCAL_FLAGS for call in retained)
    tvm.ir.assert_structural_equal(result, _run(result))


def test_cross_wait_acquires_fix_without_a_mutable_mte2_bridge():
    # All C-side MTE2 copies read an immutable input. A workspace MTE2 read
    # therefore cannot accidentally supply the acquire needed by the FIX write.
    result = _run(_epoch_program(readonly_input=True))
    _epoch_byte_oracle(result, 2, 3)


def test_epoch_template_dma_does_not_certify_the_pto_metadata_backend():
    # PTO reconstructs DMA geometry from metadata; AscendC helper template
    # dimensions cannot stand in for those distinct runtime memory effects.
    function = _epoch_program()
    with pytest.raises(tvm.error.TVMError, match="supports A2/A3 AscendC typed legacy effects only"):
        _run_model(function, model="pto", target_kind="llvm")


@pytest.mark.parametrize("kind", ["if_then_else", "select"])
def test_coordinate_dependent_epoch_counts_keep_each_workgroup_protocol(kind):
    result = _run(_coordinate_epoch_program(kind))
    _epoch_byte_oracle(result, 2, 3, total_epochs=5)


def test_if_then_else_does_not_evaluate_an_unselected_overflowing_branch():
    result = _run(_coordinate_epoch_program(mutation="unselected_overflow"))
    _epoch_byte_oracle(result, 2, 3)


def test_if_then_else_divisor_is_checked_under_its_actual_branch_guard():
    result = _run(_coordinate_epoch_program(mutation="guarded_division"))
    _epoch_byte_oracle(result, 2, 3)


@pytest.mark.parametrize(
    "kind,mutation,diagnostic",
    [
        ("if_then_else", "selected_overflow", "nonwrapping signed index arithmetic"),
        ("if_then_else", "unknown_call", "finite pure loop extent"),
        ("if_then_else", "data_dependent", "mutable For bounds are unsupported"),
        ("if_then_else", "different_owner_counts", "deadlock or unmatched cross-core Wait"),
        ("select", "unselected_overflow", "nonwrapping signed index arithmetic"),
        ("select", "unknown_call", "finite pure loop extent"),
        ("select", "guarded_division", "division requires a nonzero divisor"),
        ("if_then_else", "invalid_division", "division requires a nonzero divisor"),
        ("if_then_else", "condition_overflow", "nonwrapping signed index arithmetic"),
    ],
)
def test_unchecked_conditional_extents_cannot_authorize_epochs(kind, mutation, diagnostic):
    # Select evaluates both branches; if_then_else evaluates only the selected
    # one. Neither may turn a selected overflow or unknown value into a count.
    function = _coordinate_epoch_program(kind, mutation=mutation)
    with pytest.raises(tvm.error.TVMError, match=diagnostic):
        _run(function)


def test_empty_normalized_controls_inherit_their_explicit_resource_owner():
    result = _run(_epoch_program(empty_controls=True))
    _epoch_byte_oracle(result, 2, 3)


def test_empty_control_owner_inheritance_does_not_hide_real_hardware_effects():
    function = _epoch_program(mutation="hardware_owner_conflict")
    with pytest.raises(tvm.error.TVMError, match="Configured operation conflicts with resource_scope"):
        _run(function)


@pytest.mark.parametrize(
    "mutation,diagnostic",
    [
        ("missing_seed", "deadlock or unmatched cross-core Wait"),
        ("extra_seed", "unordered physical GM RAW/WAR/WAW"),
        ("missing_release", "deadlock or unmatched cross-core Wait"),
        ("missing_free_wait", "unordered physical GM RAW/WAR/WAW"),
        ("late_free_wait", "unordered physical GM RAW/WAR/WAW"),
        ("early_ready", "unordered physical GM RAW/WAR/WAW"),
        ("early_release", "unordered physical GM RAW/WAR/WAW"),
        ("one_subcore_release", "deadlock or unmatched cross-core Wait"),
        ("unequal_epochs", "deadlock or unmatched cross-core Wait"),
        ("overlapping_subcores", "unordered physical GM RAW/WAR/WAW"),
        ("overlapping_groups", "unordered physical GM RAW/WAR/WAW"),
        ("unknown_alias", "requires explicit noalias or a trusted Cython fresh-v2 allocation contract"),
        # The GM-loaded condition is a non-DMA GM effect, so it is rejected
        # during effect import before the finite-control proof is reached.
        ("gm_dependent_control", "requires canonical typed DMA for each real GM effect"),
        ("unknown_scalar_control", "requires a finite pure branch condition"),
        ("dynamic_cross_id", "literal cross event ID"),
        ("no_gm_publication", "requires a real Cube-to-Vector GM publication"),
        ("scalar_cross_set", "cross Set pipe disagrees with its owner"),
        ("cross_l1_alias", "uncovered cross-owner non-GM effect"),
        ("uncovered_initial_gm", "unordered physical GM RAW/WAR/WAW"),
        ("asymmetric_terminal_release", "terminal credits require a symmetric single-credit cleanup"),
        ("nonuniform_terminal_drain", "terminal cleanup differs across workgroups"),
    ],
)
def test_unproven_epoch_protocols_remain_rejected(mutation, diagnostic):
    function = _epoch_program(mutation=mutation)
    with pytest.raises(tvm.error.TVMError, match=diagnostic):
        _run(function)


def test_sixteen_initial_credits_exceed_the_hardware_counter():
    function = _epoch_program(groups=1, epochs=16, depth=16)
    with pytest.raises(tvm.error.TVMError, match="cannot prove hardware counter capacity"):
        _run(function)


def test_multiple_outstanding_terminal_credits_remain_rejected():
    # All original waits can finish and the two slots never race. The terminal
    # signature still exceeds the one-credit cleanup contract.
    function = _epoch_program(groups=1, epochs=2, depth=2, mutation="missing_all_drains")
    with pytest.raises(tvm.error.TVMError, match="terminal credits require a symmetric single-credit cleanup"):
        _run(function)


def test_one_terminal_credit_is_consumed_without_adding_a_producer():
    original = _epoch_program(mutation="missing_drain")
    result = _run(original)

    def cross_calls(function, suffix):
        return [call for call in _semantic_leaves(function) if _call_name(call).endswith(suffix)]

    old_sets = cross_calls(original, "set_cross_flag")
    new_sets = cross_calls(result, "set_cross_flag")
    assert len(old_sets) == len(new_sets)
    for first, second in zip(old_sets, new_sets):
        tvm.ir.assert_structural_equal(first, second)
    old_waits = cross_calls(original, "wait_cross_flag")
    new_waits = cross_calls(result, "wait_cross_flag")
    assert len(new_waits) == len(old_waits) + 1

    def payload(function):
        leaves = []
        for call in _semantic_leaves(function):
            name = _call_name(call)
            if name not in _LOCAL_FLAGS and "cross_flag" not in name:
                leaves.append(call)
        return leaves

    old_payload, new_payload = payload(original), payload(result)
    assert len(old_payload) == len(new_payload)
    for first, second in zip(old_payload, new_payload):
        tvm.ir.assert_structural_equal(first, second)
    for owner in (0, 1):
        old = []
        new = []
        for program, calls in ((original, old), (result, new)):
            for call, integer in _actor_calls(program, owner, 0, 0):
                name = _call_name(call)
                if "cross_flag" in name:
                    channel = call.args[1] if name.endswith("set_cross_flag") else call.args[0]
                    calls.append((name, integer(channel)))
        assert new == old + ([("tl.ascend_wait_cross_flag", 0)] if owner == 0 else [])
    _epoch_byte_oracle(result, 2, 3)
    tvm.ir.assert_structural_equal(result, _run(result))


def test_a_ready_token_cannot_authorize_another_ring_slot():
    function = _epoch_program(groups=1, epochs=5, depth=3, mutation="wrong_slot")
    with pytest.raises(tvm.error.TVMError, match="unordered physical GM RAW/WAR/WAW"):
        _run(function)


def test_independent_oracle_detects_missing_free_completion():
    # Sending FREE from MTE2 must explicitly import the preceding MTE3 writes.
    result = _run(_epoch_program(free_pipe="MTE2"))

    def remove(node):
        if not isinstance(node, tir.Evaluate) or not isinstance(node.value, tir.Call):
            return None
        call, name = node.value, _call_name(node.value)
        source_completion = name == "tl.ascend_auto_barrier" and call.args[0].value in (
            "PIPE_MTE3",
            "PIPE_ALL",
        )
        source_handoff = name in _AUTO_FLAGS and call.args[0].value.startswith("MTE3_")
        return tir.Evaluate(0) if source_completion or source_handoff else None

    broken = result.with_body(
        tir.stmt_functor.ir_transform(result.body, remove, None, ["tir.Evaluate"]),
    )
    with pytest.raises(AssertionError, match="completion.*root=workspace"):
        _epoch_byte_oracle(broken, 2, 3)


def test_cross_fences_cover_mutable_gm_used_only_by_vector_subcores():
    original = _epoch_program(vector_only_handoff=True)
    result = _run(original)
    _epoch_byte_oracle(result, 2, 3, vector_only_handoff=True)
    side_root = original.buffer_map[original.params[-1]].data

    def remove(node):
        if not isinstance(node, tir.Evaluate) or not isinstance(node.value, tir.Call):
            return None
        call, name = node.value, _call_name(node.value)
        barrier = name == "tl.ascend_auto_barrier" and call.args[0].value in (
            "PIPE_MTE3",
            "PIPE_ALL",
        )
        handoff = name in _AUTO_FLAGS and call.args[0].value in ("MTE3_S", "MTE3_MTE2")
        return tir.Evaluate(0) if barrier or handoff else None

    broken = result.with_body(
        tir.stmt_functor.ir_transform(result.body, remove, None, ["tir.Evaluate"]),
    )
    # Focus on the separate GM root so a missing fence for the main C/V
    # workspace cannot make this sensitivity check pass accidentally.
    with pytest.raises(AssertionError, match="completion.*root=vector_only_gm"):
        _epoch_byte_oracle(broken, 2, 3, vector_only_handoff=True, focus_root=side_root)
