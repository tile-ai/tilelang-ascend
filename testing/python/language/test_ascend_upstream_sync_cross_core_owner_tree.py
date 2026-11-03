# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Closed C/V trees: real DMA byte ownership and asynchronous completion.

The host fixtures deliberately need no numerical GEMM implementation. Their C
accumulator represents an already computed value; the independent oracle checks
the actual strided memory effects, original control and generated event order.
"""

from collections import Counter

import pytest

from tilelang import tvm
from tilelang import language as T
from tilelang.jit.ascend_runtime_contract import prepare_auto_gm_contract
from test_ascend_upstream_sync_cross_core import _intrinsic, _run
from test_ascend_pipeline_ring_provenance import _pre_sync
from test_ascend_upstream_sync_composite_invocation import _independent_gemms, _nested_gemm

tir = tvm.tir

_OWNER_TREE_PASS_CONFIGS = {
    "tl.ascend_sync_backend": "upstream",
    "tl.ascend_auto_sync": True,
    "tl.ascend_auto_sync_vs": True,
    "tl.ascend_auto_cv_combine": False,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_memory_planning": False,
}


def _owner_tree_gemm(workgroups=2, *, in_place=False):
    """Expert staging extends the established single-publication GEMM fixture.

    Both outputs and the two explicit workspaces use their actual Cython
    allocation roles. Inputs remain unrestricted and may alias one another.
    """
    tile_m, n, k_tile, k_tiles, tiles = 32, 16, 64, 2, 2
    vector_rows = tile_m // 2
    group_rows, total_k, stride = tile_m * tiles, k_tile * k_tiles, n * 2
    total_rows = workgroups * group_rows

    @T.prim_func
    def main(
        A: T.Tensor((total_rows, total_k), "float16"),
        B: T.Tensor((total_k, n), "float16"),
        Bias: T.Tensor((1, n), "float32"),
        Workspace: T.Tensor((total_rows, stride), "float32"),
        Output: T.Tensor((total_rows, stride), "float32"),
        HalfOutput: T.Tensor((total_rows, stride), "float16"),
        HalfWorkspace: T.Tensor((total_rows, stride), "float16"),
    ):
        with T.Kernel(workgroups, is_npu=True) as (cid, vid):
            a = T.alloc_L1((tile_m, k_tile), "float16")
            b = T.alloc_L1((k_tile, n), "float16")
            c = T.alloc_L0C((tile_m, n), "float32")
            local = T.alloc_ub((vector_rows, n), "float32")
            bias = T.alloc_ub((vector_rows, n), "float32")
            half_local = T.alloc_ub((vector_rows, n), "float16")
            with T.Scope("C"):
                for tile in T.serial(tiles):
                    for k in T.serial(k_tiles):
                        T.copy(A[cid * group_rows + tile * tile_m, k * k_tile], a)
                        T.copy(B[k * k_tile, 0], b)
                        T.gemm_v0(a, b, c, init=(k == 0), kL0Size=64)
                    if in_place:
                        T.copy(c, Output[cid * group_rows + tile * tile_m, 0])
                    else:
                        T.copy(c, Workspace[cid * group_rows + tile * tile_m, 0])
                T.set_cross_flag("FIX", 3)
            with T.Scope("V"):
                # Empty public intent still needs a real S->MTE2 handoff.
                T.wait_cross_flag(3)
                for tile in T.serial(tiles):
                    v_row = cid * group_rows + tile * tile_m + vid * vector_rows
                    if in_place:
                        T.copy(Output[v_row, 0], local)
                    else:
                        T.copy(
                            Workspace[v_row, 0],
                            local,
                        )
                    for row in T.serial(vector_rows):
                        T.copy(Bias[:, :], bias[row : row + 1, :])
                    T.tile.add(local, local, bias)
                    T.copy(local, Output[v_row, 0])
                    T.tile.add(local, local, bias)
                    T.copy(local, Output[v_row, 0])
                    T.copy(local, Workspace[v_row, 0])
                    T.tile.cast(half_local, local, "CAST_NONE", vector_rows * n)
                    T.copy(
                        half_local,
                        HalfOutput[v_row, 0],
                    )
                    T.copy(
                        half_local,
                        HalfWorkspace[v_row, 0],
                    )

    return main.without_attr("tir.noalias")


def _owner_tree_pre_sync(program):
    class Captured(Exception):
        pass

    captured = []

    @tvm.instrument.pass_instrument
    class Capture:
        def run_before_pass(self, module, info):
            if info.name == "tl.AscendInsertSync":
                captured.append(module["main"])
                raise Captured

    import tilelang

    prepared = prepare_auto_gm_contract(
        program,
        "cython",
        "ascendc",
        out_idx=[4, 5],
        workspace_idx=[3, 6],
    )
    try:
        with tvm.transform.PassContext(config=_OWNER_TREE_PASS_CONFIGS, instruments=[Capture()]):
            tilelang.lower(prepared, target="ascendc", platform="A3")
    except Captured:
        pass
    assert len(captured) == 1
    return captured[0]


def _pointer(buffer, mode, offset=0, extent=None):
    return buffer.access_ptr(mode, offset=offset, extent=extent)


def _dma(name, source, destination, *arguments, source_offset=0, destination_offset=0, extent=None):
    return tir.Evaluate(
        tir.call_extern(
            "handle",
            name,
            _pointer(source, "r", source_offset, extent),
            _pointer(destination, "w", destination_offset, extent),
            *arguments,
        )
    )


def _closed_tree(
    *,
    groups=2,
    partition="rows",
    in_place=False,
    kind=tir.ForKind.SERIAL,
    wait_pipe="",
    mutation=None,
    trusted=True,
):
    columns, c_rows, tiles = 16, 16, 2
    c_columns = columns * 2 if partition == "columns" else columns
    vector_rows = c_rows if partition == "columns" else c_rows // 2
    stride = c_columns * 2
    group_rows = tiles * c_rows
    bx, vid = tir.Var("bx", "int32"), tir.Var("vid", "int32")
    ci, ck = tir.Var("c_tile", "int32"), tir.Var("c_read", "int32")
    vi, row = tir.Var("v_tile", "int32"), tir.Var("broadcast_row", "int32")
    shape = (groups * group_rows, stride)
    first = tir.decl_buffer(shape, "float32", name="first")
    second = tir.decl_buffer(shape, "float32", name="second")
    bias = tir.decl_buffer((columns,), "float32", name="bias")
    workspace = tir.decl_buffer(shape, "float32", name="workspace")
    output = tir.decl_buffer(shape, "float32", name="output")
    half_output = tir.decl_buffer(shape, "float16", name="half_output")
    half_workspace = tir.decl_buffer(shape, "float16", name="half_workspace")
    published = output if in_place else workspace
    accumulator = tir.decl_buffer(
        (c_rows, c_columns),
        "float32",
        name="acc",
        scope="wmma.accumulator",
    )
    l1 = tir.decl_buffer((c_rows, c_columns), "float32", name="l1", scope="shared.l1")
    local = tir.decl_buffer((vector_rows, columns), "float32", name="local", scope="shared.ub")
    broadcast = tir.decl_buffer(
        (vector_rows, columns),
        "float32",
        name="broadcast",
        scope="shared.ub",
    )
    half_local = tir.decl_buffer(
        (vector_rows, columns),
        "float16",
        name="half_local",
        scope="shared.ub",
    )
    volume = vector_rows * columns
    group_pitch = 0 if mutation == "overlapping_workgroups" else group_rows * stride
    c_start = bx * group_pitch + ci * c_rows * stride
    if partition == "columns":
        v_start = bx * group_pitch + vi * c_rows * stride + vid * columns
    else:
        v_start = bx * group_pitch + vi * c_rows * stride + vid * vector_rows * stride
    if mutation == "overlapping_subcores":
        v_start = bx * group_pitch + vi * c_rows * stride
    if mutation == "integer_overflow":
        maximum = tir.IntImm("int32", (1 << 31) - 1)
        v_start = tir.Add(v_start, tir.Sub(tir.Add(bx, maximum), maximum))

    release = _intrinsic("tl.ascend_set_cross_flag", "FIX", 3, 2)
    acquire = _intrinsic(
        "tl.ascend_wait_cross_flag",
        3,
        "V" if mutation == "unsupported_wait_intent" else wait_pipe,
    )
    c_reads = tir.SeqStmt(
        [
            _dma(
                f"tl::ascend::copy_gm_to_l1<float, {c_rows}, {c_columns}>",
                source,
                l1,
                stride,
                c_rows,
                c_columns,
                c_rows,
                c_columns,
                source_offset=c_start,
                extent=c_rows * c_columns,
            )
            for source in (first, second)
        ]
    )
    c_write = _dma(
        f"tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, {c_rows}, {c_columns}, 0>",
        accumulator,
        published,
        stride,
        c_rows,
        c_columns,
        c_rows,
        c_columns,
        0,
        0,
        destination_offset=c_start,
        extent=c_rows * c_columns,
    )
    c_body = [tir.For(ck, 0, 2, kind, c_reads), c_write]
    if mutation == "flag_in_loop":
        c_body.append(release)
    cube = [tir.For(ci, 0, tiles, kind, tir.SeqStmt(c_body))]
    if mutation != "flag_in_loop":
        cube.append(release)
    if mutation == "flag_in_condition":
        condition = tir.Var("publish_condition", "bool")
        cube[-1] = tir.IfThenElse(condition, release, None)
    if mutation == "write_after_publication":
        cube.append(tir.stmt_functor.substitute(c_write, {ci: tir.IntImm("int32", 0)}))
    shared_read = _dma(
        f"tl::ascend::copy_gm_to_ub<float, {columns}, {vector_rows}>",
        published,
        local,
        stride,
        vector_rows,
        columns,
        tir.const(0, "float32"),
        vector_rows,
        columns,
        source_offset=v_start,
        extent=volume,
    )
    bias_row = _dma(
        f"tl::ascend::copy_gm_to_ub<float, {columns}>",
        bias,
        broadcast,
        columns,
        1,
        columns,
        tir.const(0, "float32"),
        1,
        columns,
        destination_offset=row * columns,
        extent=columns,
    )
    add = _intrinsic(
        "tl.ascend_add",
        local.access_ptr("w"),
        local.access_ptr("r"),
        broadcast.access_ptr("r"),
        volume,
    )
    output_stride = (1 << 32) if mutation == "wide_dma_stride" else stride
    output_columns = columns - 1 if mutation == "partial_dma_mask" else columns
    store = _dma(
        f"tl::ascend::copy_ub_to_gm<float, {columns}, {vector_rows}>",
        local,
        output,
        tir.IntImm("int64", output_stride),
        vector_rows,
        output_columns,
        destination_offset=v_start,
        extent=volume,
    )
    vector_body = tir.SeqStmt(
        [
            shared_read,
            tir.For(row, 0, vector_rows, kind, bias_row),
            add,
            store,
            add,
            store,
            _intrinsic(
                "tl.ascend_cast",
                half_local.access_ptr("w"),
                local.access_ptr("r"),
                "CAST_NONE",
                volume,
            ),
            _dma(
                f"tl::ascend::copy_ub_to_gm<half, {columns}, {vector_rows}>",
                half_local,
                half_output,
                stride,
                vector_rows,
                columns,
                destination_offset=v_start,
                extent=volume,
            ),
            _dma(
                f"tl::ascend::copy_ub_to_gm<half, {columns}, {vector_rows}>",
                half_local,
                half_workspace,
                stride,
                vector_rows,
                columns,
                destination_offset=v_start,
                extent=volume,
            ),
        ]
    )
    vector = [acquire, tir.For(vi, 0, tiles, kind, vector_body)]
    if mutation == "read_before_wait":
        vector.insert(0, tir.stmt_functor.substitute(shared_read, {vi: tir.IntImm("int32", 0)}))
    if mutation == "wait_in_loop":
        vector = [tir.For(vi, 0, tiles, kind, tir.SeqStmt([acquire, vector_body]))]
    body = tir.SeqStmt(
        [
            tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, tir.SeqStmt(cube)),
            tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, tir.SeqStmt(vector)),
        ]
    )
    allocations = [accumulator, l1, local, broadcast, half_local]
    addresses, sizes, ub_offset = {}, {}, 0
    for buffer in allocations:
        size = int(buffer.shape[0]) * int(buffer.shape[1]) * (tvm.DataType(buffer.dtype).bits // 8)
        addresses[buffer.data] = ub_offset if buffer.scope() == "shared.ub" else 0
        sizes[buffer.data] = size
        if buffer.scope() == "shared.ub":
            ub_offset += size
        body = tir.Allocate(buffer.data, buffer.dtype, buffer.shape, tir.const(True, "bool"), body)
    body = tir.AttrStmt(
        tir.IterVar(None, vid, tir.IterVar.ThreadIndex, "blockIdx.y"),
        "thread_extent",
        2,
        body,
    )
    body = tir.AttrStmt(
        tir.IterVar(None, bx, tir.IterVar.ThreadIndex, "blockIdx.x"),
        "thread_extent",
        groups,
        body,
    )
    buffers = [first, second, bias, workspace, output, half_output, half_workspace]
    handles = [tir.Var(f"arg{index}", "handle") for index in range(len(buffers))]
    if mutation == "flag_in_condition":
        handles.append(condition)
    function = (
        tir.PrimFunc(handles, body, buffer_map=dict(zip(handles, buffers)))
        .with_attr("address_map", addresses)
        .with_attr("size_map", sizes)
        .with_attr("npu_cv_ratio", tir.StringImm("cv_1_2"))
        .with_attr("auto_gm_indices", [])
    )
    if trusted:
        function = prepare_auto_gm_contract(
            function,
            "cython",
            "ascendc",
            out_idx=[4, 5],
            workspace_idx=[3, 6],
        )
    else:
        # Positions alone do not certify actual per-call allocator ownership.
        function = function.with_attr("auto_gm_indices", [3, 4, 5, 6])
    return function


def _call_name(call):
    if call.op.name == "tir.call_extern":
        return call.args[0].value.split("<", 1)[0].removeprefix("tl::ascend::")
    return call.op.name


def _actor_calls(function, owner, workgroup, subcore):
    analyzer = tvm.arith.Analyzer()
    result = []

    def integer(expression, values):
        value = analyzer.simplify(tir.stmt_functor.substitute(expression, values))
        assert isinstance(value, tir.IntImm), value
        return int(value)

    def walk(statement, values):
        if isinstance(statement, tir.SeqStmt):
            for child in statement.seq:
                walk(child, values)
        elif isinstance(statement, tir.AttrStmt):
            if statement.attr_key == "resource_scope" and int(statement.value) != owner:
                return
            if statement.attr_key == "thread_extent":
                coordinate = workgroup if statement.node.thread_tag == "blockIdx.x" else subcore
                values = {
                    **values,
                    statement.node.var: tir.const(coordinate, statement.node.var.dtype),
                }
            walk(statement.body, values)
        elif isinstance(statement, (tir.Allocate, tir.DeclBuffer)):
            walk(statement.body, values)
        elif isinstance(statement, tir.LetStmt):
            walk(
                statement.body,
                {
                    **values,
                    statement.var: tir.const(integer(statement.value, values), statement.var.dtype),
                },
            )
        elif isinstance(statement, tir.For):
            minimum, extent = integer(statement.min, values), integer(statement.extent, values)
            for index in range(minimum, minimum + extent):
                walk(
                    statement.body,
                    {**values, statement.loop_var: tir.const(index, statement.loop_var.dtype)},
                )
        elif isinstance(statement, tir.IfThenElse):
            condition = integer(statement.condition, values)
            branch = statement.then_case if condition else statement.else_case
            if branch is not None:
                walk(branch, values)
        elif isinstance(statement, tir.Evaluate):
            if isinstance(statement.value, tir.Call):
                result.append((statement.value, values.copy()))
        else:
            raise AssertionError(type(statement))

    walk(function.body, {})
    return [(call, lambda expr, scope=scope: integer(expr, scope)) for call, scope in result]


def _byte_completion_oracle(
    function,
    groups,
    *,
    envelope=False,
    missing_internal=None,
    publication=True,
    input_count=3,
):
    """Interpret real row/column bytes and event happens-before, not IR order.

    CrossSet only publishes FIX operations completed by its actual drain. A
    CrossWait completes S; a later MTE2 read needs its own local handoff.
    Other memory dependencies, including UB reuse and repeated GM stores, must
    be carried by actual local events. A real GEMM expands the helper's entry,
    ping-pong and drain protocol; its lexical return transfers no clock.

    ``envelope`` additionally checks each advertised external acquire before
    entering the helper. MTE1 and M gates remain independent at that boundary.
    """
    actors = [(group, 0, 0) for group in range(groups)]
    if publication:
        actors += [(group, 1, vid) for group in range(groups) for vid in range(2)]
    pipes = ("S", "MTE1", "MTE2", "MTE3", "V", "M", "FIX")
    dimensions = [(actor, pipe) for actor in actors for pipe in pipes]
    clocks = {key: [0] * len(dimensions) for key in dimensions}
    completed = {key: tuple(value) for key, value in clocks.items()}
    tokens, published = {}, {}
    writes, readers = {}, {}
    counts = Counter()
    inputs = {function.buffer_map[parameter].data for parameter in function.params[:input_count]}
    gm_roots = {buffer.data for buffer in function.buffer_map.values()}
    initialized_local = set()
    gemm_calls = []

    def find_initial_accumulator(node):
        if isinstance(node, tir.Call) and node.op.name == "tl.ascend_gemm_v0":
            gemm_calls.append(node)
        if not isinstance(node, tir.Allocate):
            return
        if node.buffer_var.type_annotation.storage_scope == "wmma.accumulator":
            initialized_local.add(node.buffer_var)

    tir.stmt_functor.post_order_visit(function.body, find_initial_accumulator)
    if gemm_calls:
        # Real clear/accumulate operations must produce their own L0C bytes.
        initialized_local.clear()

    def before(first, second):
        assert all(a <= b for a, b in zip(first, second)), "missing asynchronous completion edge"

    def step(actor, pipe, incoming=None):
        clock = clocks[actor, pipe]
        if incoming is not None:
            clock[:] = [max(a, b) for a, b in zip(clock, incoming)]
        clock[dimensions.index((actor, pipe))] += 1
        return tuple(clock)

    def bytes_of(pointer, integer, rows=None, columns=None, stride=None):
        root = pointer.args[1]
        first = integer(pointer.args[2])
        item_bytes = tvm.DataType(pointer.args[0].dtype).bits // 8
        if rows is None:
            elements = range(first, first + integer(pointer.args[3]))
        else:
            elements = (first + r * stride + c for r in range(rows) for c in range(columns))
        addresses = []
        for element in elements:
            addresses.extend(range(element * item_bytes, (element + 1) * item_bytes))
        return root, addresses

    def access(actor, ticket, span, write):
        root, addresses = span
        for address in addresses:
            key = (root, address) if root in gm_roots else (actor, root, address)
            if key in writes:
                before(writes[key], ticket)
            elif not write:
                # Only user inputs and the abstract already-computed L0C are
                # initialized. Fresh GM and private UB require real writes.
                assert root in inputs or root in initialized_local, "read of an unproduced byte"
            if write:
                for reader in readers.pop(key, []):
                    before(reader, ticket)
                writes[key] = ticket
            else:
                readers.setdefault(key, []).append(ticket)

    def flag(actor, kind, event, event_id):
        source, destination = event.split("_")
        key = (actor, source, destination, event_id)
        if kind == "set":
            assert key not in tokens, "double set of a local event"
            tokens[key] = step(actor, source)
        else:
            assert key in tokens, "local wait without a producer token"
            step(actor, destination, tokens.pop(key))

    def require_acquire(actor, span, pipe, write=False):
        root, addresses = span
        ticket = clocks[actor, pipe]
        for address in addresses:
            key = (root, address) if root in gm_roots else (actor, root, address)
            if key in writes:
                before(writes[key], ticket)
            if write:
                for reader in readers.get(key, []):
                    before(reader, ticket)

    def gemm(actor, call, integer):
        # This fixture uses one complete internal K/N tile. Interpret the
        # actual common.h gemm_v0 protocol, including both unused-slot credits.
        # Do not merge the external M and MTE1 acquire clocks by assumption.
        name, _, template = call.args[0].value.partition("<")
        fields = [value.strip() for value in template.removesuffix(">").split(",")]
        assert name == "gemm_v0" and len(fields) == 8
        assert fields[:2] == ["half", "float"] and fields[5:7] == ["false", "false"]
        m, n, k = map(int, fields[2:5])
        k_l0 = int(fields[7])
        assert k == k_l0 and all(dimension % 16 == 0 for dimension in (m, n, k))
        assert max(m, n) * k_l0 * 2 <= 32 * 1024
        assert integer(call.args[5]) == n
        first, second, accumulator = (bytes_of(call.args[index], integer) for index in (1, 2, 3))
        clear = integer(call.args[4])
        assert clear in (0, 1)
        if envelope:
            require_acquire(actor, first, "MTE1")
            require_acquire(actor, second, "MTE1")
            require_acquire(actor, accumulator, "M", write=True)

        for kind in ("set", "wait"):
            flag(actor, kind, "MTE2_MTE1", 0)
        if missing_internal != "FIX_M":
            for kind in ("set", "wait"):
                flag(actor, kind, "FIX_M", 0)
        for slot in (0, 1):
            if missing_internal != f"M_MTE1_prime_{slot}":
                flag(actor, "set", "M_MTE1", slot)
        flag(actor, "wait", "M_MTE1", 0)

        # The helper's actual slot-0 L1->L0 loads follow its M_MTE1 wait.
        # These physical byte ranges are private to each Cube, but shared by
        # every helper invocation. No helper-return completion is invented.
        first_l0 = ("helper_l0a", range(m * k_l0 * 2))
        second_l0 = ("helper_l0b", range(n * k_l0 * 2))
        load = step(actor, "MTE1")
        access(actor, load, first, False)
        access(actor, load, first_l0, True)
        load = step(actor, "MTE1")
        access(actor, load, second, False)
        access(actor, load, second_l0, True)
        for kind in ("set", "wait"):
            flag(actor, kind, "MTE1_M", 0)
        step(actor, "M")  # The helper's real PIPE_M barrier.
        mma = step(actor, "M")
        access(actor, mma, first_l0, False)
        access(actor, mma, second_l0, False)
        if not clear:
            access(actor, mma, accumulator, False)
        access(actor, mma, accumulator, True)
        flag(actor, "set", "M_MTE1", 0)
        for slot in (0, 1):
            flag(actor, "wait", "M_MTE1", slot)
        for event in ("MTE1_MTE2", "M_FIX"):
            for kind in ("set", "wait"):
                flag(actor, kind, event, 0)
        counts["tl.ascend_gemm_v0"] += 1
        counts["gemm_clear" if clear else "gemm_accumulate"] += 1

    for actor in actors:
        group, owner, vid = actor
        for call, integer in _actor_calls(function, owner, group, vid):
            name = _call_name(call)
            if name in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"):
                kind = "set" if name.endswith("set_flag") else "wait"
                flag(actor, kind, call.args[0].value, integer(call.args[1]))
            elif name == "tl.ascend_auto_barrier":
                pipe = call.args[0].value.removeprefix("PIPE_")
                completed[actor, pipe] = step(actor, pipe)
            elif name.endswith("set_cross_flag"):
                assert owner == 0
                flag_id = call.args[2] if name == "tl.ascend_auto_set_cross_flag" else call.args[1]
                key = (group, integer(flag_id))
                assert key not in published, "publication is repeated"
                published[key] = [completed[actor, "FIX"], set()]
                counts["cross_set"] += 1
            elif name.endswith("wait_cross_flag"):
                assert owner == 1
                key = (group, integer(call.args[0]))
                release, consumed = published[key]
                assert vid not in consumed, "cross token consumed twice"
                consumed.add(vid)
                step(actor, "S", release)
                counts["cross_wait"] += 1
            elif name in ("copy_gm_to_l1", "copy_l0c_to_gm", "copy_gm_to_ub", "copy_ub_to_gm"):
                symbol = call.args[0].value
                template = symbol.partition("<")[2].removesuffix(">")
                dimensions_in_symbol = [field.strip() for field in template.split(",")]
                if name == "copy_l0c_to_gm":
                    rows, columns = map(int, dimensions_in_symbol[-3:-1])
                elif name == "copy_gm_to_l1":
                    rows, columns = map(int, dimensions_in_symbol[1:])
                else:
                    columns = int(dimensions_in_symbol[1])
                    rows = int(dimensions_in_symbol[2]) if len(dimensions_in_symbol) == 3 else 1
                pipe = {
                    "copy_gm_to_l1": "MTE2",
                    "copy_l0c_to_gm": "FIX",
                    "copy_gm_to_ub": "MTE2",
                    "copy_ub_to_gm": "MTE3",
                }[name]
                ticket = step(actor, pipe)
                source = bytes_of(call.args[1], integer)
                destination = bytes_of(call.args[2], integer)
                if name in ("copy_gm_to_l1", "copy_gm_to_ub"):
                    source = bytes_of(call.args[1], integer, rows, columns, integer(call.args[3]))
                else:
                    destination = bytes_of(
                        call.args[2],
                        integer,
                        rows,
                        columns,
                        integer(call.args[3]),
                    )
                access(actor, ticket, source, False)
                access(actor, ticket, destination, True)
                counts[name] += 1
            elif name in ("tl.ascend_add", "tl.ascend_cast"):
                ticket = step(actor, "V")
                access(actor, ticket, bytes_of(call.args[1], integer), False)
                if name == "tl.ascend_add":
                    access(actor, ticket, bytes_of(call.args[2], integer), False)
                access(actor, ticket, bytes_of(call.args[0], integer), True)
            elif name == "tl.ascend_gemm_v0":
                gemm(actor, call, integer)
            else:
                raise AssertionError(name)
    assert not tokens, "undrained local event tokens"
    assert all(consumed == {0, 1} for _, consumed in published.values())
    expected_sets = groups if publication else 0
    assert counts["cross_set"] == expected_sets and counts["cross_wait"] == expected_sets * 2
    return counts


def _semantic_leaves(function):
    calls = []

    def visit(node):
        if (
            isinstance(node, tir.Evaluate)
            and isinstance(node.value, tir.Call)
            and _call_name(node.value)
            not in (
                "tl.ascend_auto_set_flag",
                "tl.ascend_auto_wait_flag",
                "tl.ascend_auto_barrier",
            )
        ):
            calls.append(node.value)

    tir.stmt_functor.post_order_visit(function.body, visit)
    return calls


@pytest.fixture(autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.mark.parametrize(
    "groups,partition,in_place,kind,wait_pipe",
    [
        (2, "rows", False, tir.ForKind.SERIAL, ""),
        (3, "rows", True, tir.ForKind.UNROLLED, ""),
        (2, "columns", False, tir.ForKind.UNROLLED, "MTE2"),
        (3, "columns", True, tir.ForKind.SERIAL, ""),
    ],
)
def test_closed_trees_preserve_dma_bytes_and_the_channel(
    groups,
    partition,
    in_place,
    kind,
    wait_pipe,
):
    original = _closed_tree(
        groups=groups,
        partition=partition,
        in_place=in_place,
        kind=kind,
        wait_pipe=wait_pipe,
    )
    result = _run(original)
    old, retained = _semantic_leaves(original), _semantic_leaves(result)
    assert len(old) == len(retained)
    for first, second in zip(old, retained):
        tvm.ir.assert_structural_equal(first, second)
    counts = _byte_completion_oracle(result, groups)
    assert counts["copy_l0c_to_gm"] == groups * 2
    assert counts["copy_ub_to_gm"] == groups * 2 * 2 * 4
    assert "tir.noalias" not in result.attrs
    tvm.ir.assert_structural_equal(result, _run(result))


@pytest.mark.parametrize("missing", ["FIX", "S_MTE2"])
def test_independent_byte_oracle_detects_missing_completion_fences(missing):
    result = _run(_closed_tree())

    def remove(node):
        if not isinstance(node, tir.Evaluate) or not isinstance(node.value, tir.Call):
            return node
        name, call = _call_name(node.value), node.value
        drains_fix = name == "tl.ascend_auto_barrier" and call.args[0].value == "PIPE_FIX"
        if missing == "FIX" and drains_fix:
            return tir.Evaluate(0)
        is_auto_flag = name in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag")
        if missing == "S_MTE2" and is_auto_flag and call.args[0].value == missing:
            return tir.Evaluate(0)
        return node

    broken = result.with_body(
        tir.stmt_functor.ir_transform(result.body, None, remove, ["tir.Evaluate"]),
    )
    with pytest.raises(AssertionError, match="completion"):
        _byte_completion_oracle(broken, 2)


@pytest.mark.parametrize(
    "mutation",
    [
        "overlapping_workgroups",
        "overlapping_subcores",
        "write_after_publication",
        "read_before_wait",
        "flag_in_loop",
        "wait_in_loop",
        "flag_in_condition",
        "wide_dma_stride",
        "partial_dma_mask",
        "integer_overflow",
        "unsupported_wait_intent",
    ],
)
def test_closed_tree_rejects_unproven_global_ownership_control_or_dma(mutation):
    with pytest.raises(tvm.error.TVMError):
        _run(_closed_tree(mutation=mutation))


def test_unversioned_fresh_positions_cannot_authorize_closed_publication():
    with pytest.raises(tvm.error.TVMError, match="fresh|noalias"):
        _run(_closed_tree(trusted=False))


@pytest.fixture(scope="module")
def expert_owner_trees():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")
    functions = {}
    for in_place in (False, True):
        original = _owner_tree_pre_sync(_owner_tree_gemm(2, in_place=in_place))
        functions[in_place] = (original, _run(original))
    return functions


@pytest.mark.parametrize("in_place", [False, True])
def test_actual_expert_gemm_owner_trees_preserve_broadcast_and_allocation_roles(
    expert_owner_trees,
    in_place,
):
    original, result = expert_owner_trees[in_place]
    old, retained = _semantic_leaves(original), _semantic_leaves(result)
    assert len(old) == len(retained)
    for first, second in zip(old, retained):
        tvm.ir.assert_structural_equal(first, second)
    assert original.attrs["tl.ascend_auto_gm_fresh_contract"].value == "cython-auto-gm-v2"
    assert [int(index) for index in original.attrs["tl.ascend_jit_result_indices"]] == [4, 5]
    assert [int(index) for index in original.attrs["tl.ascend_jit_workspace_indices"]] == [3, 6]
    assert "tir.noalias" not in result.attrs
    calls = []
    tir.stmt_functor.post_order_visit(
        result.body,
        lambda node: calls.append(node) if isinstance(node, tir.Call) else None,
    )
    auto_sets = [call for call in calls if call.op.name == "tl.ascend_auto_set_flag"]
    assert any(call.args[0].value == "S_MTE2" for call in auto_sets)
    assert any(call.args[0].value == "FIX_M" for call in auto_sets)
    assert any(call.args[0].value == "MTE2_MTE1" for call in auto_sets)
    for call in calls:
        if call.op.name not in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"):
            continue
        event = call.args[0].value
        assert event != "FIX_MTE1"
        if event in ("FIX_M", "MTE2_MTE1"):
            assert int(call.args[1]) != 0, "external events must not reuse the helper's event 0"
    for envelope in (False, True):
        counts = _byte_completion_oracle(result, 2, envelope=envelope)
        assert counts["tl.ascend_gemm_v0"] == 8
        assert counts["gemm_clear"] == counts["gemm_accumulate"] == 4
        assert counts["copy_l0c_to_gm"] == 4
        assert counts["copy_ub_to_gm"] == 40


def _without_external_events(function, event):
    def remove(node):
        if not isinstance(node, tir.Evaluate) or not isinstance(node.value, tir.Call):
            return node
        call = node.value
        is_local_flag = call.op.name in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag")
        if is_local_flag and call.args[0].value == event:
            return tir.Evaluate(0)
        return node

    return function.with_body(
        tir.stmt_functor.ir_transform(function.body, None, remove, ["tir.Evaluate"]),
    )


@pytest.mark.parametrize("event", ["FIX_M", "MTE2_MTE1"])
def test_independent_helper_oracle_distinguishes_external_acquire_gates(
    expert_owner_trees,
    event,
):
    _, result = expert_owner_trees[False]
    broken = _without_external_events(result, event)
    # The concrete helper itself has both event-0 entry handshakes. They can
    # redundantly repair the missing external event; a full model must allow
    # this. The advertised envelope still requires its own independent gate.
    _byte_completion_oracle(broken, 2)
    with pytest.raises(AssertionError, match="completion"):
        _byte_completion_oracle(broken, 2, envelope=True)


@pytest.mark.parametrize("slot", [0, 1])
def test_independent_helper_oracle_requires_both_real_slot_primes(expert_owner_trees, slot):
    _, result = expert_owner_trees[False]
    with pytest.raises(AssertionError, match="producer token"):
        _byte_completion_oracle(result, 2, missing_internal=f"M_MTE1_prime_{slot}")


def test_independent_helper_oracle_rejects_fix_source_overwrite_without_any_handoff(
    expert_owner_trees,
):
    _, result = expert_owner_trees[False]
    broken = _without_external_events(result, "FIX_M")
    with pytest.raises(AssertionError, match="completion"):
        _byte_completion_oracle(broken, 2, missing_internal="FIX_M")


@pytest.mark.parametrize("fresh_v2", [False, True], ids=["v1", "v2"])
@pytest.mark.parametrize(
    "factory,out_indices,input_count,gemms,clears,stores",
    [
        (_independent_gemms, [-2, -1], 4, 4, 4, 4),
        (_nested_gemm, [-1], 2, 6, 3, 3),
    ],
    ids=["independent", "nested"],
)
def test_single_cube_acquires_survive_explicit_fresh_output_roles(
    factory,
    out_indices,
    input_count,
    gemms,
    clears,
    stores,
    fresh_v2,
):
    # Keep the existing real kernels. Ordinary local synchronization retains
    # the same conservative GM aliases with either allocation recipe.
    prepared = prepare_auto_gm_contract(
        factory(),
        "cython",
        "ascendc",
        out_idx=out_indices if fresh_v2 else None,
    )
    original, _ = _pre_sync(prepared)
    result = _run(original)
    marker = "cython-auto-gm-v2" if fresh_v2 else "cython-auto-gm-v1"
    assert result.attrs["tl.ascend_auto_gm_fresh_contract"].value == marker
    assert "tir.noalias" not in result.attrs
    original_leaves, retained_leaves = _semantic_leaves(original), _semantic_leaves(result)
    assert len(original_leaves) == len(retained_leaves)
    for first, second in zip(original_leaves, retained_leaves):
        tvm.ir.assert_structural_equal(first, second)
    calls = []
    tir.stmt_functor.post_order_visit(
        result.body,
        lambda node: calls.append(node) if isinstance(node, tir.Call) else None,
    )
    events = [call.args[0].value for call in calls if call.op.name == "tl.ascend_auto_set_flag"]
    assert "MTE2_MTE1" in events
    assert "FIX_MTE1" not in events
    for call in calls:
        if call.op.name not in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"):
            continue
        if call.args[0].value in ("MTE2_MTE1", "FIX_M"):
            assert int(call.args[1]) != 0
    counts = _byte_completion_oracle(
        result,
        1,
        publication=False,
        input_count=input_count,
    )
    assert counts["tl.ascend_gemm_v0"] == gemms
    assert counts["gemm_clear"] == clears
    assert counts["gemm_accumulate"] == gemms - clears
    assert counts["copy_l0c_to_gm"] == stores
    if fresh_v2:
        legacy = prepare_auto_gm_contract(factory(), "cython", "ascendc")
        legacy_before, _ = _pre_sync(legacy)
        legacy_result = _run(legacy_before)
        tvm.ir.assert_structural_equal(legacy_result.body, result.body, map_free_vars=True)


@pytest.mark.parametrize(
    "factory,out_indices,input_count",
    [(_independent_gemms, [-2, -1], 4), (_nested_gemm, [-1], 2)],
    ids=["independent", "nested"],
)
def test_single_cube_noalias_helper_keeps_both_independent_acquires(
    factory,
    out_indices,
    input_count,
):
    # An explicit valid noalias contract exposes the real FIX/L0C dependency
    # without relying on implicit consumption of an allocation recipe.
    prepared = prepare_auto_gm_contract(
        factory().with_attr("tir.noalias", True),
        "cython",
        "ascendc",
        out_idx=out_indices,
    )
    original, _ = _pre_sync(prepared)
    result = _run(original)
    calls = []
    tir.stmt_functor.post_order_visit(
        result.body,
        lambda node: calls.append(node) if isinstance(node, tir.Call) else None,
    )
    events = [call.args[0].value for call in calls if call.op.name == "tl.ascend_auto_set_flag"]
    assert "FIX_M" in events and "MTE2_MTE1" in events
    assert "FIX_MTE1" not in events
    for call in calls:
        if call.op.name not in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"):
            continue
        if call.args[0].value in ("MTE2_MTE1", "FIX_M"):
            assert int(call.args[1]) != 0
    for envelope in (False, True):
        _byte_completion_oracle(
            result,
            1,
            envelope=envelope,
            publication=False,
            input_count=input_count,
        )
    broken = _without_external_events(result, "FIX_M")
    with pytest.raises(AssertionError, match="completion"):
        _byte_completion_oracle(
            broken,
            1,
            envelope=True,
            publication=False,
            input_count=input_count,
        )
