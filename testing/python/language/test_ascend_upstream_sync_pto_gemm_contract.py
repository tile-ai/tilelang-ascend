# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Host-only regression coverage for ordinary PTO GEMM synchronization."""

from __future__ import annotations

from collections.abc import Callable

import pytest

import tilelang
from tilelang import tvm
from tvm import tir


_AVAILABLE = tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable", allow_missing=True)
_TARGET = tvm.target.Target({"kind": "llvm", "model": "pto"})
_EVENTS = ("MTE2_MTE1", "FIX_M", "M_MTE1", "MTE1_M", "MTE1_MTE2", "M_FIX")


def _i(value: int) -> tir.IntImm:
    return tir.IntImm("int32", value)


def _function(
    m: int = 128,
    n: int = 256,
    k: int = 64,
    kl0: int = 128,
    grid: tuple[int, int] = (1, 1),
    pad_allocations: bool = False,
) -> tir.PrimFunc:
    # The call ABI and descriptors match the materialized ordinary GEMM path.
    # Constructing TIR here avoids a device, a DSL kernel, and memory planning.
    gm_m, gm_n = m * grid[0], n * grid[1]
    inputs = (
        tir.decl_buffer((gm_m, k), "float16", name="A"),
        tir.decl_buffer((k, gm_n), "float16", name="B"),
        tir.decl_buffer((gm_m, gm_n), "float16", name="C"),
    )
    local = (
        tir.decl_buffer((m, k), "float16", name="A_L1", scope="shared.l1"),
        tir.decl_buffer((k, n), "float16", name="B_L1", scope="shared.l1"),
        tir.decl_buffer((m, n), "float32", name="C_L0", scope="wmma.accumulator"),
    )
    cid = tir.Var("cid", "int32")
    bx, by = cid // grid[1], cid % grid[1]
    offsets = (bx * m * k, by * n, bx * m * gm_n + by * n)
    extents = (m * k, k * n, m * n)
    physical_extents = extents
    if pad_allocations:
        padded = []
        for rows, columns in ((m, k), (k, n), (m, n)):
            padded_rows = (rows + 15) // 16 * 16
            padded_columns = (columns + 15) // 16 * 16
            padded.append(padded_rows * padded_columns)
        physical_extents = tuple(padded)
    gm_pointers = []
    for j, buffer in enumerate(inputs):
        mask = "w" if j == 2 else "r"
        gm_pointers.append(buffer.access_ptr(mask, offset=offsets[j], extent=extents[j]))
    statements = [
        tir.Evaluate(
            tir.call_extern(
                "handle",
                f"tl::ascend::copy_gm_to_l1<half, {m}, {k}>",
                gm_pointers[0],
                local[0].access_ptr("w"),
                k,
                m,
                k,
                m,
                k,
            )
        ),
        tir.Evaluate(
            tir.call_extern(
                "handle",
                f"tl::ascend::copy_gm_to_l1<half, {k}, {n}>",
                gm_pointers[1],
                local[1].access_ptr("w"),
                gm_n,
                k,
                n,
                k,
                n,
            )
        ),
        tir.Evaluate(
            tir.Call(
                "handle",
                tvm.ir.Op.get("tl.ascend_gemm_v0"),
                [
                    tir.StringImm(f"gemm_v0<half, float, {m}, {n}, {k}, false, false, {kl0}>"),
                    local[0].access_ptr("r"),
                    local[1].access_ptr("r"),
                    local[2].access_ptr("rw"),
                    tir.const(True),
                    _i(n),
                ],
            )
        ),
        tir.Evaluate(
            tir.call_extern(
                "handle",
                f"tl::ascend::copy_l0c_to_gm<float, half, layout::RowMajor, {m}, {n}, 0>",
                local[2].access_ptr("r"),
                gm_pointers[2],
                gm_n,
                m,
                n,
                m,
                n,
                tir.const(False),
                0,
            )
        ),
    ]
    body = tir.AttrStmt(_i(0), "resource_scope", _i(0), tir.SeqStmt(statements))
    for buffer, extent in reversed(list(zip(local, physical_extents))):
        body = tir.Allocate(buffer.data, buffer.dtype, [_i(extent)], tir.const(True), body)
    thread = tir.IterVar(None, cid, tir.IterVar.ThreadIndex, "blockIdx.x")
    body = tir.AttrStmt(thread, "thread_extent", _i(grid[0] * grid[1]), body)
    params = [tir.Var(f"parameter_{j}", "handle") for j in range(3)]
    function = tir.PrimFunc(params, body, buffer_map=dict(zip(params, inputs)))
    shapes = {b.data: list(b.shape) for b in (*inputs, *local)}
    logical = {b.data: list(b.shape) for b in inputs}
    logical.update({b.data: list(b.shape) * 2 for b in local})
    sizes = {}
    for j, (buffer, extent) in enumerate(zip(local, physical_extents)):
        sizes[buffer.data] = _i(extent * (4 if j == 2 else 2))
    return (
        function.with_attr("global_symbol", "main")
        .with_attr("buffer_shapess", shapes)
        .with_attr("logic_buffer_shapes", logical)
        .with_attr(
            "address_map",
            {
                local[0].data: _i(0),
                local[1].data: _i(2 * physical_extents[0]),
                local[2].data: _i(0),
            },
        )
        .with_attr("size_map", sizes)
    )


def _name(node: tir.Call) -> str:
    if not isinstance(node, tir.Call):
        return ""
    if node.op.name == "tir.call_extern":
        return node.args[0].value.split("<")[0].replace("tl::ascend::", "")
    return node.op.name.replace("tl.ascend_", "")


def _calls(function: tir.PrimFunc) -> list[tir.Call]:
    calls = []

    def visit(node):
        if _name(node) in ("copy_gm_to_l1", "gemm_v0", "copy_l0c_to_gm"):
            calls.append(node)

    tir.stmt_functor.post_order_visit(function.body, visit)
    return calls


def _run(function: tir.PrimFunc) -> tir.PrimFunc:
    if _AVAILABLE is None or not _AVAILABLE():
        pytest.skip("Upstream synchronization requires Z3 ON")
    config = {"tl.ascend_auto_sync": True, "tl.ascend_auto_sync_vs": True}
    with tvm.transform.PassContext(config=config):
        mod = tilelang.transform.AscendInsertSync(_TARGET, "A3")(tvm.IRModule({"main": function}))
    return mod["main"]


def _accept(function: tir.PrimFunc) -> tir.PrimFunc:
    result = _run(function)
    assert "tl.ascend_sync_proof" in result.attrs
    assert tvm.ir.structural_equal(result, _run(result))
    assert "tir.noalias" not in result.attrs
    assert len(_calls(function)) == len(_calls(result)) == 4
    assert all(tvm.ir.structural_equal(a, b) for a, b in zip(_calls(function), _calls(result)))
    return result


def _change_calls(
    function: tir.PrimFunc,
    name: str,
    change: Callable,
    ordinal: int = 0,
) -> tir.PrimFunc:
    seen = 0

    def post(node):
        nonlocal seen
        if _name(node) != name:
            return None
        current, seen = seen, seen + 1
        if current != ordinal:
            return None
        args = list(node.args)
        change(args)
        return tir.Call(node.dtype, node.op, args)

    body = tir.stmt_functor.ir_transform(function.body, None, post, ["tir.Call"])
    return function.with_body(body)


def _pointer(args: list, position: int, field: int, value: tir.PrimExpr) -> None:
    original = args[position]
    fields = list(original.args)
    fields[field] = value
    args[position] = tir.Call(original.dtype, original.op, fields)


def _map(function: tir.PrimFunc, name: str, root: tir.Var, value) -> tir.PrimFunc:
    values = dict(function.attrs[name].items())
    values[root] = value
    return function.with_attr(name, values)


def _assumed(function: tir.PrimFunc, kind: str = "tilelang_assume") -> tir.PrimFunc:
    offset = tir.Var("offset", "int32")
    function = _change_calls(function, "copy_gm_to_l1", lambda a: _pointer(a, 1, 2, offset))
    function = tir.PrimFunc(
        list(function.params) + [offset],
        function.body,
        function.ret_type,
        function.buffer_map,
        function.attrs,
    )
    if kind == "evaluate":

        def post(node):
            if isinstance(node, tir.SeqStmt):
                return tir.SeqStmt([tir.Evaluate(tir.assume(offset == 0))] + list(node.seq))
            return None

        body = tir.stmt_functor.ir_transform(function.body, None, post, ["tir.SeqStmt"])
        return function.with_body(body)
    return function.with_body(tir.AttrStmt(offset == 0, kind, _i(1), function.body))


@pytest.mark.parametrize(
    "m,n,k,kl0,grid",
    [
        (16, 16, 128, 128, (1, 1)),
        (16, 16, 256, 128, (1, 1)),
        (32, 48, 32, 16, (1, 1)),
        (64, 64, 80, 32, (1, 1)),
        (128, 240, 144, 128, (1, 1)),
        (64, 256, 272, 128, (1, 1)),
        (128, 256, 64, 4080, (1, 1)),
        (64, 128, 256, 64, (2, 2)),
        (128, 256, 64, 128, (2, 2)),
        (128, 256, 64, 128, (3, 3)),
        (512, 64, 64, 64, (1, 1)),
        (32, 512, 64, 64, (1, 1)),
        (528, 48, 64, 32, (1, 1)),
    ],
)
def test_generic_pto_gemm(m: int, n: int, k: int, kl0: int, grid: tuple[int, int]) -> None:
    _accept(_function(m, n, k, kl0, grid))


@pytest.mark.parametrize("m,n", [(528, 48), (32, 528), (128, 272)])
def test_pto_gemm_rejects_exhausted_pools(m: int, n: int) -> None:
    with pytest.raises(tvm.error.TVMError, match="footprint exceeds"):
        _run(_function(m, n, 64, 64))


def test_pto_gemm_l1_address_boundaries() -> None:
    function = _function()
    roots = {v.name: v for v in function.attrs["address_map"]}
    last = 524032 - 16384
    _accept(_map(function, "address_map", roots["A_L1"], _i(last)))
    for root, address in ((roots["A_L1"], last + 32), (roots["A_L1"], 1), (roots["B_L1"], 0)):
        with pytest.raises(tvm.error.TVMError, match="physical pool|overlap"):
            _run(_map(function, "address_map", root, _i(address)))


@pytest.mark.parametrize(
    "name,position,ordinal",
    [
        ("copy_gm_to_l1", 1, 0),
        ("copy_gm_to_l1", 1, 1),
        ("copy_l0c_to_gm", 2, 0),
    ],
)
def test_pto_gemm_rejects_grid_pointer_one_element_past(
    name: str,
    position: int,
    ordinal: int,
) -> None:
    function = _function(grid=(3, 3))

    def shift(args):
        _pointer(args, position, 2, args[position].args[2] + _i(1))

    with pytest.raises(tvm.error.TVMError, match="GM|bounds|rectangle"):
        _run(_change_calls(function, name, shift, ordinal))


@pytest.mark.parametrize("kind", ["tilelang_assume", "tl.assume", "evaluate"])
def test_pto_gemm_scoped_assumptions(kind: str) -> None:
    _accept(_assumed(_function(), kind))


def test_pto_gemm_ignores_unsupported_assumption_metadata() -> None:
    with pytest.raises(tvm.error.TVMError, match="GM|bounds|rectangle"):
        _run(_assumed(_function(), "tir.assume"))


def test_pto_gemm_pure_let_uses_thread_bounds() -> None:
    function = _function(grid=(3, 3))

    def post(node):
        if not isinstance(node, tir.SeqStmt):
            return None
        statements = list(node.seq)
        call = statements[0].value
        offset = call.args[1].args[2]
        captured = tir.Var("captured_offset", offset.dtype)
        args = list(call.args)
        _pointer(args, 1, 2, captured)
        statements[0] = tir.Evaluate(tir.Call(call.dtype, call.op, args))
        return tir.LetStmt(captured, offset, tir.SeqStmt(statements))

    body = tir.stmt_functor.ir_transform(function.body, None, post, ["tir.SeqStmt"])
    _accept(function.with_body(body))


def test_pto_gemm_assumption_does_not_escape_sibling_scope() -> None:
    function = _function()
    offset = tir.Var("scoped_offset", "int32")
    seen = 0

    def post(node):
        nonlocal seen
        if not isinstance(node, tir.Evaluate) or _name(node.value) != "copy_gm_to_l1":
            return None
        args = list(node.value.args)
        _pointer(args, 1, 2, offset)
        statement = tir.Evaluate(tir.Call(node.value.dtype, node.value.op, args))
        seen += 1
        if seen == 1:
            return tir.AttrStmt(offset == 0, "tilelang_assume", _i(1), statement)
        return statement

    body = tir.stmt_functor.ir_transform(function.body, None, post, ["tir.Evaluate"])
    function = tir.PrimFunc(
        list(function.params) + [offset],
        body,
        function.ret_type,
        function.buffer_map,
        function.attrs,
    )
    with pytest.raises(tvm.error.TVMError, match="index|GM|bounds|rectangle"):
        _run(function)


@pytest.mark.parametrize("kind", ["unbound", "mutable", "overflow", "lossy_cast", "unsigned_wrap"])
def test_pto_gemm_rejects_unproven_index_arithmetic(kind: str) -> None:
    function = _function()
    dtype = "int64" if kind == "lossy_cast" else "uint32" if kind == "unsigned_wrap" else "int32"
    value = tir.Var("index_value", dtype)
    expression = value
    condition = value == tir.const(0, dtype)
    if kind in ("overflow", "unsigned_wrap"):
        plus = tir.Add(value, tir.const(1, dtype))
        expression = tir.Sub(plus, plus)
        condition = value == tir.const(4294967295 if kind == "unsigned_wrap" else 2147483647, dtype)
    elif kind == "lossy_cast":
        expression = tir.Cast("int32", value)
        condition = value == tir.const(4294967296, dtype)
    function = _change_calls(function, "copy_gm_to_l1", lambda a: _pointer(a, 1, 2, expression))
    body = function.body
    if kind != "unbound":
        body = tir.AttrStmt(condition, "tl.assume", _i(1), function.body)
    params = list(function.params) + [value]
    if kind == "mutable":
        buffer = function.buffer_map[function.params[0]]
        loaded = tir.Cast("int32", tir.BufferLoad(buffer, [_i(0), _i(0)]))
        body = tir.LetStmt(value, loaded, function.body)
        params = function.params
    function = tir.PrimFunc(
        params,
        body,
        function.ret_type,
        function.buffer_map,
        function.attrs,
    )
    with pytest.raises(tvm.error.TVMError, match="index|wrapping|cast|GM|bounds|mutable|opaque"):
        _run(function)


@pytest.mark.parametrize("kind", ["transpose", "clear_false", "outer_loop"])
def test_pto_gemm_rejects_other_invocation_protocols(kind: str) -> None:
    function = _function()
    if kind == "transpose":

        def change(args):
            args[0] = tir.StringImm(args[0].value.replace("false, false", "true, false"))

        function = _change_calls(function, "gemm_v0", change)
    elif kind == "clear_false":
        function = _change_calls(function, "gemm_v0", lambda a: a.__setitem__(4, tir.const(False)))
    else:
        loop = tir.For(tir.Var("outer_k", "int32"), _i(0), _i(1), tir.ForKind.SERIAL, function.body)
        function = function.with_body(loop)
    with pytest.raises(tvm.error.TVMError, match="gemm_v0"):
        _run(function)


@pytest.mark.parametrize("direction", _EVENTS)
def test_pto_gemm_hidden_event_reservations(direction: str) -> None:
    producer, consumer = direction.split("_", 1)

    def with_events(event_id):
        function = _function()
        events = [
            tir.Evaluate(
                tir.Call(
                    "handle",
                    tvm.ir.Op.get(f"tl.ascend_{kind}_flag"),
                    [
                        tir.StringImm("PIPE_" + producer),
                        tir.StringImm("PIPE_" + consumer),
                        _i(event_id),
                    ],
                )
            )
            for kind in ("set", "wait")
        ]

        def post(node):
            if isinstance(node, tir.SeqStmt):
                return tir.SeqStmt(events + list(node.seq))
            return None

        body = tir.stmt_functor.ir_transform(function.body, None, post, ["tir.SeqStmt"])
        return function.with_body(body)

    with pytest.raises(tvm.error.TVMError, match="hidden event pool"):
        _run(with_events(1))
    for event_id in (0, 2):
        _accept(with_events(event_id))


@pytest.mark.parametrize("m,n,k", [(17, 16, 64), (16, 17, 64), (16, 16, 17)])
def test_pto_gemm_rejects_unaligned_descriptors(m: int, n: int, k: int) -> None:
    with pytest.raises(tvm.error.TVMError, match="aligned"):
        _run(_function(m, n, k, pad_allocations=True))


def test_pto_gemm_rejects_malformed_copy_and_allocation() -> None:
    function = _function()
    root = next(v for v in function.attrs["address_map"] if v.name == "A_L1")
    bad_shape = _map(function, "logic_buffer_shapes", root, [_i(128), _i(64), _i(127), _i(64)])
    bad_mask = _change_calls(function, "copy_gm_to_l1", lambda a: _pointer(a, 1, 4, _i(2)))
    short_fix = _change_calls(function, "copy_l0c_to_gm", lambda a: a.__delitem__(slice(6, None)))

    def post(node):
        if isinstance(node, tir.Allocate) and node.buffer_var.same_as(root):
            return tir.Allocate(root, "float32", node.extents, node.condition, node.body)
        return None

    body = tir.stmt_functor.ir_transform(function.body, None, post, ["tir.Allocate"])
    retyped = function.with_body(body)
    for malformed in (bad_shape, bad_mask, short_fix, retyped):
        with pytest.raises(tvm.error.TVMError, match="(?:PTO flat|Composite) gemm_v0"):
            _run(malformed)


def _allocation_root() -> tir.Var:
    return tir.Var("tile", tvm.ir.PointerType(tvm.ir.PrimType("float16"), "shared.l1"))


def _allocation_source(roots: list[tir.Var], addresses: dict[tir.Var, int]) -> list[str]:
    body = tir.Evaluate(_i(0))
    for root in reversed(roots):
        body = tir.Allocate(root, "float16", [_i(256)], tir.const(True), body)
    function = (
        tir.PrimFunc([], body)
        .with_attr("global_symbol", "allocation_identity")
        .with_attr("address_map", addresses)
        .with_attr("logic_buffer_shapes", {root: [16, 16, 16, 16] for root in roots})
    )
    # This entry returns C++ source; it does not compile or launch a kernel.
    module = tvm.get_global_func("target.build.tilelang_ascend_pto")(
        tvm.IRModule({"allocation_identity": function}), tvm.target.Target("llvm"), "A3"
    )
    return [line.strip() for line in module.get_source().splitlines() if "TASSIGN(" in line]


def test_pto_codegen_preserves_allocation_var_identity() -> None:
    first, second = _allocation_root(), _allocation_root()
    assert not first.same_as(second)
    assert _allocation_source([first, second], {first: 0, second: 512}) == [
        "TASSIGN(tile, 0);",
        "TASSIGN(tile_1, 512);",
    ]


@pytest.mark.parametrize("legacy_addresses", [(512,), (512, 512)])
def test_pto_codegen_accepts_unambiguous_legacy_addresses(
    legacy_addresses: tuple[int, ...],
) -> None:
    root = _allocation_root()
    addresses = {_allocation_root(): value for value in legacy_addresses}
    assert _allocation_source([root], addresses) == ["TASSIGN(tile, 512);"]


def test_pto_codegen_rejects_ambiguous_legacy_addresses() -> None:
    root, first, second = _allocation_root(), _allocation_root(), _allocation_root()
    with pytest.raises(tvm.error.TVMError, match="Ambiguous PTO allocation address"):
        _allocation_source([root], {first: 0, second: 512})
