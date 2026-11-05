"""Lexical scope and source ABI for TA's private Persistent producer."""

import importlib
import inspect

import pytest
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script.builder import ir as B

import tilelang
from tilelang import tvm
import tilelang.ascend.a2a3.language as T
from tilelang.ascend.a2a3 import compiler
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
from tilelang.language.loop import Persistent as shared_persistent

NATIVE = tvm.get_global_func("tl.A2A3Persistent", allow_missing=True) is not None
pytestmark = pytest.mark.skipif(not NATIVE, reason="requires the native A2/A3 Persistent producer built with USE_ASCEND=ON")
TARGETS = [{"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]} for dialect in ["ascendc", "pto"]]


def _original_factory():
    module = importlib.import_module("testing.ascend_a2a3.python.language.test_tilelang_ascend_language_persistent")
    return module._persistent_kernel


def _undefined(func):
    definitions = list(func.params) + [buffer.data for buffer in func.buffer_map.values()]
    # Buffer-map shape symbols are implicit PrimFunc inputs. A statement-only
    # visitor needs those exact Var identities supplied along with handles.
    for buffer in func.buffer_map.values():
        for dimension in buffer.shape:
            definitions.extend(tvm.tirx.analysis.undefined_vars(dimension))
    return tvm.tirx.analysis.undefined_vars(func.body, definitions)


@pytest.mark.parametrize("rows", [4, 10, 11, 20, "dynamic"])
def test_original_kernel_frontend_and_json_have_no_undefined_vars(rows):
    extent = T.symbolic("rows") if rows == "dynamic" else rows
    func = _original_factory()(extent)
    assert not _undefined(func)
    restored = tvm.ir.load_json(tvm.ir.save_json(func))
    assert not _undefined(restored)
    tvm.ir.assert_structural_equal(func, restored)


def test_frame_coordinate_identity_is_bound_inside_its_wave():
    with IRBuilder() as builder, B.prim_func():
        B.func_name("coordinates")
        with T.Persistent([3, 4], 2, 0, 2) as coordinates:
            captured = list(coordinates)
            B.evaluate(coordinates[0] + coordinates[1])
    func = builder.get()
    assert not _undefined(func)
    loops = []
    bindings = []
    tvm.tirx.stmt_functor.post_order_visit(
        func.body,
        lambda node: (
            loops.append(node) if isinstance(node, tvm.tirx.For) else bindings.append(node) if isinstance(node, tvm.tirx.Bind) else None
        ),
    )
    assert len(loops) == 1 and len(bindings) == 2
    assert [binding.var.same_as(var) for binding, var in zip(bindings, reversed(captured))] == [True, True]
    body_bindings = []
    tvm.tirx.stmt_functor.post_order_visit(
        loops[0].body, lambda node: body_bindings.append(node) if isinstance(node, tvm.tirx.Bind) else None
    )
    assert len(body_bindings) == 2
    assert all(any(value.same_as(loops[0].loop_var) for value in tvm.tirx.analysis.undefined_vars(binding.value)) for binding in bindings)


def test_original_private_signature_leaves_shared_padding_api_unchanged():
    assert list(inspect.signature(T.Persistent).parameters) == ["domain", "wave_size", "index", "group_size"]
    assert "num_stages" in inspect.signature(shared_persistent).parameters
    assert "annotations" in inspect.signature(shared_persistent).parameters
    assert T.Persistent is not shared_persistent


def test_dynamic_shape_named_like_wave_counter_keeps_distinct_var_identities():
    extent = T.symbolic("w")
    func = _original_factory()(extent)
    assert not _undefined(func)
    loops = []
    tvm.tirx.stmt_functor.post_order_visit(
        func.body,
        lambda node: loops.append(node) if isinstance(node, tvm.tirx.For) and node.thread_binding is None else None,
    )
    assert len(loops) == 1 and not loops[0].loop_var.same_as(extent)
    restored = tvm.ir.load_json(tvm.ir.save_json(func))
    assert not _undefined(restored)
    for target in TARGETS:
        artifact = tilelang.lower(restored, target=target)
        assert "int64_t w" in artifact.kernel_source
        assert [str(dim) for dim in artifact.params[0].shape] == ["w", "2"]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("rows", [4, 10, 11, 20, "dynamic"])
def test_all_original_positive_domains_reach_complete_source(target, rows):
    extent = T.symbolic("rows") if rows == "dynamic" else rows
    artifact = tilelang.lower(_original_factory()(extent), target=target)
    assert len(artifact.params) == 1
    assert [str(dim) for dim in artifact.params[0].shape] == [str(extent), "2"]
    assert "void call(" in artifact.kernel_source
    if rows == "dynamic":
        assert "int64_t rows" in artifact.kernel_source
    if rows in [4, 10]:
        assert "break;" not in artifact.kernel_source
    if rows == 11:
        assert "break;" in artifact.kernel_source


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("rows", [11, "dynamic"])
def test_real_bisheng_compiles_guarded_and_dynamic_direct_carrier_source(target, rows):
    extent = T.symbolic("rows") if rows == "dynamic" else rows
    artifact = tilelang.lower(_original_factory()(extent), target=target)
    try:
        resolved_target = tvm.target.Target(target)
        config = compiler.resolve_compile_config(resolved_target, normalize_a2a3_pass_configs(resolved_target))
    except FileNotFoundError as error:
        pytest.skip(str(error))
    library = compiler.LibraryGenerator(config)
    library.update_lib_code(artifact.kernel_source)
    library.compile_lib(timeout=120)
    assert library.get_lib_path().endswith(".so")
