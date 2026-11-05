"""TA storage planning, pointer rewrites and optional target memory metadata."""

import importlib.util
from contextlib import contextmanager
from pathlib import Path

import pytest

from tilelang import tvm

tirx = tvm.tirx


def _run(body, buffers=(), is_npu=True, config=None, attrs=None, target=None):
    func = tirx.PrimFunc([b.data for b in buffers], body, buffer_map={b.data: b for b in buffers})
    func = func.with_attr(
        {"global_symbol": "main", "target": tvm.target.Target(target or {"kind": "ascend", "arch": "dav-2201"}), **(attrs or {})}
    )
    with tvm.transform.PassContext(config=config or {}):
        return tvm.ffi.get_global_func("tl.transform.A2A3AscendStorageRewrite")(is_npu)(tvm.IRModule.from_expr(func))["main"]


def _nodes(body, typ):
    out = []
    tirx.stmt_functor.post_order_visit(body, lambda x: out.append(x) if isinstance(x, typ) else None)
    return out


def _two_lifetimes(scope="shared.ub", dtypes=("float32", "float32"), sizes=(16, 16)):
    a = tirx.decl_buffer((sizes[0],), dtypes[0], name="a", scope=scope)
    b = tirx.decl_buffer((sizes[1],), dtypes[1], name="b", scope=scope)
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(a),
            tirx.AllocBuffer(b),
            tirx.BufferStore(a, tirx.const(1, a.dtype), [0]),
            tirx.Evaluate(tirx.BufferLoad(a, [0])),
            tirx.BufferStore(b, tirx.const(2, b.dtype), [0]),
            tirx.Evaluate(tirx.BufferLoad(b, [0])),
        ]
    )
    return a, b, body


@pytest.mark.parametrize("is_npu,allocations", [(False, 1), (True, 2)])
def test_storage_preserves_npu_reuse_switch_and_flat_alias_declarations(is_npu, allocations):
    _, _, body = _two_lifetimes()
    after = _run(body, is_npu=is_npu)
    assert len(_nodes(after.body, tirx.AllocBuffer)) == allocations
    stores = _nodes(after.body, tirx.BufferStore)
    assert stores[0].buffer.data.same_as(stores[1].buffer.data) == (not is_npu)
    if not is_npu:
        assert _nodes(after.body, tirx.DeclBuffer)


def test_storage_npu_still_preserves_source_inplace_algorithm():
    a = tirx.decl_buffer((16,), "float32", name="a", scope="shared.ub")
    b = tirx.decl_buffer((16,), "float32", name="b", scope="shared.ub")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(a),
            tirx.AllocBuffer(b),
            tirx.For(i, 0, 16, tirx.ForKind.SERIAL, tirx.BufferStore(a, tirx.FloatImm("float32", 1), [i])),
            tirx.For(j, 0, 16, tirx.ForKind.SERIAL, tirx.BufferStore(b, tirx.BufferLoad(a, [j]) + 1, [j])),
            tirx.Evaluate(tirx.BufferLoad(b, [0])),
        ]
    )
    after = _run(body, is_npu=True)
    assert len(_nodes(after.body, tirx.AllocBuffer)) == 1


def test_storage_two_l1_allocations_disable_reuse_even_without_npu_flag():
    _, _, body = _two_lifetimes(scope="shared.l1")
    after = _run(body, is_npu=False)
    assert len(_nodes(after.body, tirx.AllocBuffer)) == 2


def test_storage_preserves_initializer_annotations_and_local_var_allocations():
    a, b, body = _two_lifetimes(scope="local.var", dtypes=("int32", "int32"), sizes=(1, 1))
    attrs = {"tl.local_var_init": {a.data: tirx.IntImm("int32", 7), b.data: tirx.IntImm("int32", 11)}}
    after = _run(body, attrs=attrs)
    allocations = _nodes(after.body, tirx.AllocBuffer)
    assert len(allocations) == 2
    assert sorted(int(x.annotations["tl.local_var_init"]) for x in allocations) == [7, 11]


def test_storage_dynamic_extent_and_conditional_scope_remain_represented():
    n = tirx.Var("n", "int32")
    condition = tirx.Var("condition", "bool")
    a = tirx.decl_buffer((n,), "float32", name="dynamic", scope="shared.ub")
    body = tirx.IfThenElse(condition, tirx.SeqStmt([tirx.AllocBuffer(a), tirx.BufferStore(a, tirx.FloatImm("float32", 1), [0])]), None)
    after = _run(body)
    assert _nodes(after.body, tirx.IfThenElse)[0].condition.same_as(condition)
    allocation = _nodes(after.body, tirx.AllocBuffer)[0]
    assert tvm.arith.Analyzer().can_prove_equal(allocation.buffer.shape[0], n)


def test_storage_bind_scope_preserves_load_and_following_uses():
    a, b, _ = _two_lifetimes()
    value = tirx.Var("value", "float32")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(a),
            tirx.AllocBuffer(b),
            tirx.BufferStore(a, tirx.FloatImm("float32", 3), [0]),
            tirx.Bind(value, tirx.BufferLoad(a, [0])),
            tirx.BufferStore(b, value, [0]),
            tirx.Evaluate(tirx.BufferLoad(b, [0])),
        ]
    )
    after = _run(body, is_npu=False)
    assert _nodes(after.body, tirx.Bind)[0].var.same_as(value)
    assert _nodes(after.body, tirx.Bind)[0].value.buffer.data.same_as(_nodes(after.body, tirx.BufferStore)[0].buffer.data)


def test_storage_rejects_unlowered_vectorized_loop():
    a, _, _ = _two_lifetimes()
    i = tirx.Var("i", "int32")
    body = tirx.SeqStmt(
        [tirx.AllocBuffer(a), tirx.For(i, 0, 4, tirx.ForKind.VECTORIZED, tirx.BufferStore(a, tirx.FloatImm("float32", 1), [i]))]
    )
    with pytest.raises(tvm.error.InternalError, match="VectorizeLoop before"):
        _run(body)


def _memory_info(unit_bits, max_num_bits, max_simd_bits, head_address=None):
    return tvm.ffi.get_global_func("tl.ascend.a2a3.MakeMemoryInfo")(unit_bits, max_num_bits, max_simd_bits, head_address)


@contextmanager
def _metadata(callback):
    name = "tvm.info.mem.shared.a2a3_storage_test"
    previous = tvm.ffi.get_global_func(name, allow_missing=True)
    if previous is not None:
        tvm.ffi.remove_global_func(name)
    if callback is not None:
        tvm.ffi.register_global_func(name, callback)
    try:
        yield
    finally:
        if tvm.ffi.get_global_func(name, allow_missing=True) is not None:
            tvm.ffi.remove_global_func(name)
        if previous is not None:
            tvm.ffi.register_global_func(name, previous)


def _tagged_allocation():
    _, _, body = _two_lifetimes(scope="shared.a2a3_storage_test", dtypes=("uint8", "uint8"), sizes=(3, 2))
    return body


def test_storage_missing_optional_metadata_keeps_source_32bit_alignment():
    with _metadata(None):
        after = _run(_tagged_allocation())
    allocation = _nodes(after.body, tirx.AllocBuffer)[0]
    assert int(allocation.buffer.shape[0]) == 8
    assert int(tvm.arith.Analyzer().simplify(_nodes(after.body, tirx.BufferStore)[1].indices[0])) == 4


def test_storage_metadata_retains_all_fields_alignment_and_bounds():
    info = _memory_info(8, 1024, 64, tirx.IntImm("int64", 128))
    assert int(info.unit_bits) == 8
    assert int(info.max_num_bits) == 1024
    assert int(info.max_simd_bits) == 64
    assert int(info.head_address) == 128
    with _metadata(lambda: info):
        after = _run(_tagged_allocation())
    assert int(_nodes(after.body, tirx.AllocBuffer)[0].buffer.shape[0]) == 16
    with _metadata(lambda: _memory_info(8, 64, 64)), pytest.raises(tvm.error.InternalError, match="Allocation exceed bound"):
        _run(_tagged_allocation())


@pytest.mark.parametrize("value", [17, {"max_simd_bits": 64}])
def test_storage_metadata_wrong_callback_type_is_diagnosed_and_cleaned_up(value):
    name = "tvm.info.mem.shared.a2a3_storage_test"
    before = tvm.ffi.get_global_func(name, allow_missing=True)
    with _metadata(lambda: value), pytest.raises(tvm.error.InternalError, match="must return tl.ascend.a2a3.MemoryInfo"):
        _run(_tagged_allocation())
    assert tvm.ffi.get_global_func(name, allow_missing=True) is before


def test_storage_memory_info_bridge_import_is_lazy_without_native_type_lookup(monkeypatch):
    source = Path(__file__).resolve().parents[5] / "tilelang/ascend/a2a3/transform/memory_info.py"

    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("import must not look up an Ascend native function")

    monkeypatch.setattr(tvm.ffi, "get_global_func", unexpected_lookup)
    spec = importlib.util.spec_from_file_location("a2a3_memory_info_import_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.make_memory_info)


@pytest.mark.parametrize("merge_static,allocations", [(False, 1), (True, 2)])
def test_storage_standard_tirx_config_preserves_original_merge_gate(merge_static, allocations):
    _, _, body = _two_lifetimes()
    after = _run(body, is_npu=False, config={"tirx.merge_static_smem": merge_static})
    assert len(_nodes(after.body, tirx.AllocBuffer)) == allocations


def test_storage_nullable_metadata_callback_keeps_default_alignment():
    with _metadata(lambda: None):
        after = _run(_tagged_allocation())
    assert int(_nodes(after.body, tirx.AllocBuffer)[0].buffer.shape[0]) == 8


def test_storage_readonly_buffer_parameter_keeps_its_declared_type():
    buffer = tirx.decl_buffer((16,), "int8", name="readonly")
    body = tirx.Evaluate(tirx.BufferLoad(buffer, [tirx.Ramp(0, 1, 4)]))
    after = _run(body, buffers=[buffer])
    assert after.buffer_map[buffer.data].same_as(buffer)
    assert str(_nodes(after.body, tirx.BufferLoad)[0].dtype) == "int8x4"


def test_storage_metadata_invalid_alignment_has_a_diagnostic():
    with _metadata(lambda: _memory_info(8, 1024, 0)), pytest.raises(tvm.error.InternalError, match="max_simd_bits must be positive"):
        _run(_tagged_allocation())


def test_storage_zero_simd_metadata_is_allowed_when_alignment_is_not_consumed():
    info = _memory_info(8, 1024, 0)
    assert int(info.max_simd_bits) == 0
    buffer = tirx.decl_buffer((3,), "uint8", name="single", scope="shared.a2a3_storage_test")
    body = tirx.SeqStmt([tirx.AllocBuffer(buffer), tirx.BufferStore(buffer, tirx.const(1, "uint8"), [0])])
    with _metadata(lambda: info):
        after = _run(body)
    assert int(_nodes(after.body, tirx.AllocBuffer)[0].buffer.shape[0]) == 3
