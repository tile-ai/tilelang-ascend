"""Literal data survives the real A2/A3 pipeline without runtime arguments."""

import numpy as np
import pytest

from tilelang import tvm
import tilelang.ascend.a2a3.language as T
from tilelang.ascend.a2a3 import compiler
from tilelang.ascend.a2a3.transform.constant import normalize_constants
from tilelang.backend import create_backend_context
from tilelang.engine.lower import lower_with_context

try:
    tvm.ir.Op.get("tl.a2a3_const_data")
    _NATIVE = True
except Exception:
    _NATIVE = False
pytestmark = pytest.mark.skipif(not _NATIVE, reason="requires the A2/A3 native literal bridge")
TARGET = {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc"]}


def literal_kernel(dtype="int32", scope="C", annotations=None):
    values = [-7, 11, 23, 37] if dtype == "int32" else [-7.5, 11.25, 23.5, 37.0]

    @T.prim_func
    def literal(Input: T.Tensor((1,), "int32"), Output: T.Tensor((1,), dtype)):
        pointer = T.allocate_const(values, dtype, [4], annotations)
        table = T.decl_buffer((4,), dtype, data=pointer)  # noqa: F841 - raw source name
        with T.Scope(scope):
            T._src_code("Output.SetValue(0, table[Input.GetValue(0) & 3]);", pointer)

    return literal


def _lower(func, configs=None):
    with tvm.transform.PassContext(opt_level=3, config=configs or {}):
        return lower_with_context(func, create_backend_context(TARGET, "c", "cython"))


def _declarations(func):
    nodes = []
    tvm.tirx.stmt_functor.post_order_visit(
        func.body,
        lambda n: (
            nodes.append(n)
            if isinstance(n, tvm.tirx.Bind) and isinstance(n.value, tvm.tirx.Call) and n.value.op.name == "tl.a2a3_const_data"
            else None
        ),
    )
    return nodes


@pytest.mark.parametrize("dtype,scope", [("int32", "C"), ("int32", "V"), ("float32", "C")])
def test_public_literal_survives_full_pipeline_json_and_has_no_new_arguments(dtype, scope):
    func = literal_kernel(dtype, scope)
    original_json = tvm.ir.save_json(func)
    artifact = _lower(func)
    source = artifact.kernel_source
    assert len(artifact.params) == 2
    assert "call(uint8_t* Input_handle, uint8_t* Output_handle, aclrtStream stream)" in source
    assert source.count('section(".rodata.tvm")') == 1
    assert "table[Input.GetValue(0) & 3]" in source
    assert "table.GetValue" not in source
    assert "SetFlag<" not in source and "SetVectorMask" not in source
    final = next(iter(artifact.device_mod.functions.values()))
    declarations = _declarations(final)
    assert len(declarations) == 1
    declaration = declarations[0].value
    assert [str(arg.value) for arg in declaration.args[2:4]] == ["c0", "table"]
    values = np.array([-7, 11, 23, 37] if dtype == "int32" else [-7.5, 11.25, 23.5, 37], dtype=dtype)
    assert bytes.fromhex(declaration.args[1].value) == values.tobytes()
    assert tvm.ir.save_json(func) == original_json
    assert _lower(tvm.ir.load_json(original_json)).kernel_source == source
    restored = tvm.ir.load_json(tvm.ir.save_json(artifact.device_mod))
    context = create_backend_context(TARGET, "c", "cython")
    assert context.codegen_device(restored, compile_device=False).inspect_source() == source


def test_cv_consumers_and_constant_declaration_outside_resource_scope():
    @T.prim_func
    def cv(Input: T.Tensor((1,), "int32"), Output: T.Tensor((2,), "int32")):
        pointer = T.allocate_const([7, 11, 23, 37], "int32", [4])
        table = T.decl_buffer((4,), "int32", data=pointer)  # noqa: F841 - raw source name
        with T.Scope("C"):
            T._src_code("Output.SetValue(0, table[Input.GetValue(0) & 3]);", pointer)
        with T.Scope("V"):
            T._src_code("Output.SetValue(1, table[Input.GetValue(0) & 3]);", pointer)

    artifact = _lower(cv, {"tl.ascend_auto_cv_combine": True, "tl.ascend_auto_sync": True, "tl.ascend_auto_sync_vs": True})
    source = artifact.kernel_source
    assert source.count('section(".rodata.tvm")') == 1
    assert source.count("table[Input.GetValue(0) & 3]") == 2
    assert "if ASCEND_IS_AIC" in source and "if ASCEND_IS_AIV" in source
    assert "SetFlag<" not in source
    assert len(artifact.params) == 2


def test_distinct_tables_nested_scope_and_equal_content_are_not_interned():
    @T.prim_func
    def tables(Input: T.Tensor((1,), "int32"), Output: T.Tensor((1,), "int32")):
        first = T.allocate_const([7, 11, 23, 37], "int32", [4])
        with T.Scope("C"):
            if Input[0] >= 0:
                second = T.allocate_const([7, 11, 23, 37], "int32", [4])
                T._src_code("Output.SetValue(0, first[Input.GetValue(0) & 3] + second[Input.GetValue(0) & 3]);", first, second)

    artifact = _lower(tables)
    assert artifact.kernel_source.count('section(".rodata.tvm")') == 2
    assert "first[4]" in artifact.kernel_source and "second[4]" in artifact.kernel_source


def test_var_alias_chain_preserves_the_source_valid_original_literal_name():
    @T.prim_func
    def aliases(Input: T.Tensor((1,), "int32"), Output: T.Tensor((1,), "int32")):
        table = T.allocate_const([7, 11, 23, 37], "int32", [4])
        alias = T.bind(table)
        other = T.bind(alias)
        with T.Scope("C"):
            T._src_code("Output.SetValue(0, table[Input.GetValue(0) & 3]);", other)

    source = _lower(aliases).kernel_source
    assert "table[4]" in source and "table[Input.GetValue(0) & 3]" in source
    assert "void* alias" not in source


def test_scalar_shape_and_original_ignored_annotations_keep_literal_behavior():
    annotations = {"storage_alignment": 64, "volatile_scope": 1, "pragma_unroll_explicit": 1, "audit.tag": "payload"}
    assert _lower(literal_kernel(annotations=annotations)).kernel_source == _lower(literal_kernel()).kernel_source
    normalized = normalize_constants(literal_kernel(annotations=annotations))
    payload = _declarations(normalized)[0].value.args[4].value
    assert "storage_alignment" in payload and "payload" in payload

    @T.prim_func
    def scalar(Input: T.Tensor((1,), "int32"), Output: T.Tensor((1,), "int32")):
        table = T.allocate_const(7, "int32", [])
        with T.Scope("C"):
            T._src_code("Output.SetValue(0, table[0] + Input.GetValue(0));", table)

    assert "table[1]" in _lower(scalar).kernel_source


def test_source_failed_buffer_load_is_diagnosed_before_any_native_pass():
    @T.prim_func
    def invalid(Input: T.Tensor((1,), "int32"), Output: T.Tensor((1,), "int32")):
        pointer = T.allocate_const([7, 11, 23, 37], "int32", [4])
        table = T.decl_buffer((4,), "int32", data=pointer)  # noqa: F841 - raw source name
        with T.Scope("C"):
            Output[0] = table[Input[0] & 3]

    with pytest.raises(ValueError, match="BufferLoad/BufferStore.*T._src_code"):
        _lower(invalid)


def test_pointer_escape_is_diagnosed():
    @T.prim_func
    def invalid():
        pointer = T.allocate_const([7], "int32", [1])
        T.evaluate(T.call_extern("handle", "escape", pointer))

    with pytest.raises(ValueError, match="pointer escape"):
        normalize_constants(invalid)


@pytest.mark.parametrize("extent", [8, 0, -1])
def test_payload_shape_is_distinct_from_source_declaration_extents(extent):
    @T.prim_func
    def mismatch(Input: T.Tensor((1,), "int32"), Output: T.Tensor((1,), "int32")):
        table = T.allocate_const([7, 11, 23, 37], "int32", [extent])
        with T.Scope("C"):
            T._src_code("Output.SetValue(0, table[Input.GetValue(0) & 3]);", table)

    artifact = _lower(mismatch)
    assert "table[4]" in artifact.kernel_source
    final = next(iter(artifact.device_mod.functions.values()))
    metadata = tvm.ir.load_json(_declarations(final)[0].value.args[4].value)
    assert list(metadata["extents"]) == [extent]


def test_noninteger_shape_and_nonmapping_annotations_are_diagnosed():
    with pytest.raises(ValueError, match="static integer"):
        T.allocate_const([1], "int32", [1.5])
    with pytest.raises(TypeError, match="mapping"):
        T.allocate_const([1], "int32", [1], annotations=[1])


def test_duplicate_literal_names_are_not_silently_renamed():
    func = literal_kernel()
    declaration = _declarations(func)[0]
    other = tvm.tirx.Var("table", declaration.var.type_annotation)
    func = func.with_body(tvm.tirx.SeqStmt([func.body, tvm.tirx.Bind(other, declaration.value)]))
    with pytest.raises(ValueError, match="ambiguous in raw source"):
        normalize_constants(func)


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_real_bisheng_compiles_the_literal_kernel(dtype, monkeypatch):
    try:
        config = compiler.resolve_compile_config(tvm.target.Target(TARGET))
    except (ValueError, FileNotFoundError) as error:
        pytest.skip(str(error))
    artifact = _lower(literal_kernel(dtype))
    library = compiler.LibraryGenerator(config)
    library.update_lib_code(artifact.kernel_source)
    library.compile_lib(timeout=120)
    assert library.libpath.endswith(".so")


def test_codegen_handles_cv_cloned_declaration_identity_without_duplicate_arrays():
    artifact = _lower(literal_kernel(scope="V"))
    final = next(iter(artifact.device_mod.functions.values()))
    original = _declarations(final)[0]
    clone = tvm.tirx.Var("cloned_pointer", original.var.type_annotation)
    raw = tvm.tirx.Evaluate(
        tvm.tirx.call_intrin("handle", tvm.ir.Op.get("tl.ascend_src_code"), "Output.SetValue(0, table[Input.GetValue(0) & 3]);", clone)
    )
    body = tvm.tirx.SeqStmt([final.body, tvm.tirx.Bind(clone, original.value), tvm.tirx.AttrStmt(0, "resource_scope", 1, raw)])
    # This fixture models the result of a statement clone, after IDs have been
    # fixed by frontend normalization. No new normalization runs after cloning.
    mod = tvm.IRModule.from_expr(final.with_body(body))
    source = create_backend_context(TARGET, "c", "cython").codegen_device(mod, compile_device=False).inspect_source()
    assert source.count('section(".rodata.tvm")') == 1
    assert source.count("table[Input.GetValue(0) & 3]") == 2
    assert "cloned_pointer" not in source


def test_direct_multifunction_codegen_has_an_explicit_constant_boundary():
    artifact = _lower(literal_kernel())
    final = next(iter(artifact.device_mod.functions.values()))
    mod = tvm.IRModule({"first": final, "second": final.with_attr("global_symbol", "second")})
    with pytest.raises(Exception, match="single exported kernel"):
        create_backend_context(TARGET, "c", "cython").codegen_device(mod, compile_device=False)


@pytest.mark.parametrize("dtype", ["void", "handle"])
def test_malformed_native_literal_dtype_is_diagnosed_before_size_arithmetic(dtype):
    func = literal_kernel()
    original = _declarations(func)[0]
    args = list(original.value.args)
    args[0] = tvm.tirx.StringImm(dtype)
    value = tvm.tirx.Call("handle", original.value.op, args)
    body = tvm.tirx.stmt_functor.ir_transform(
        func.body,
        None,
        lambda n: tvm.tirx.Bind(n.var, value) if isinstance(n, tvm.tirx.Bind) and n.var.same_as(original.var) else None,
        ["tirx.Bind"],
    )
    with pytest.raises(Exception, match="dtype|bit|type"):
        _lower(func.with_body(body))


def test_pointer_typing_is_restricted_to_live_proven_constant_bindings():
    from tvm.tirx.script.builder import ir as builder

    prepare = tvm.ffi.get_global_func("tl.a2a3.PrepareConstBufferData")
    finished = literal_kernel()
    constant = _declarations(finished)[0]
    before = tvm.ir.save_json(constant.var.type_annotation)
    with pytest.raises(Exception, match="live IR builder"):
        prepare(constant.var)
    assert tvm.ir.save_json(constant.var.type_annotation) == before
    with tvm.script.ir_builder.IRBuilder(), builder.prim_func():
        foreign = builder.bind(tvm.tirx.call_extern("handle", "foreign_pointer"))
        foreign_type = foreign.type_annotation
        assert prepare(foreign).same_as(foreign)
        assert tvm.ir.save_json(foreign.type_annotation) == tvm.ir.save_json(foreign_type)
        # A Var from a finished function is not a declaration of this builder.
        assert prepare(constant.var).same_as(constant.var)
        assert tvm.ir.save_json(constant.var.type_annotation) == before
        conflict = builder.bind(constant.value, tvm.ir.PointerType(tvm.ir.PrimType("float32")))
        with pytest.raises(Exception, match="conflicting element type"):
            prepare(conflict)


def test_alias_decl_buffer_keeps_payload_pointer_type_and_source_view_dtype():
    @T.prim_func
    def alias_view(Input: T.Tensor((1,), "int32"), Output: T.Tensor((1,), "int32")):
        pointer = T.allocate_const([7, 11, 23, 37], "int32", [4])
        alias = T.bind(pointer)
        view = T.decl_buffer((4,), "float32", data=alias)  # noqa: F841 - metadata-only view
        with T.Scope("C"):
            T._src_code("Output.SetValue(0, pointer[Input.GetValue(0) & 3]);", alias)

    # The view is metadata-only: the original array remains int32 and raw code
    # uses the original declaration name, as in the source-positive alias case.
    artifact = _lower(alias_view)
    assert "pointer[4]" in artifact.kernel_source
    assert "pointer[Input.GetValue(0) & 3]" in artifact.kernel_source


def test_numpy_array_input_is_cast_like_the_original_public_builder():
    values = np.array([-7, 11, 23, 37], dtype="int64")

    @T.prim_func
    def array_input(Input: T.Tensor((1,), "int32"), Output: T.Tensor((1,), "int32")):
        table = T.allocate_const(values, "int32", [4])
        with T.Scope("C"):
            T._src_code("Output.SetValue(0, table[Input.GetValue(0) & 3]);", table)

    assert "table[4]" in _lower(array_input).kernel_source


def test_empty_literal_is_preserved_without_an_out_of_bounds_read():
    @T.prim_func
    def empty(Input: T.Tensor((1,), "int32"), Output: T.Tensor((1,), "int32")):
        table = T.allocate_const([], "int32", [0])
        with T.Scope("C"):
            T._src_code("Output.SetValue(0, Input.GetValue(0));", table)

    assert "table[0]" in _lower(empty).kernel_source


def test_live_ambiguous_rebinding_cannot_reuse_a_stale_constant_origin():
    from tvm.tirx.script.builder import ir as builder

    prepare = tvm.ffi.get_global_func("tl.a2a3.PrepareConstBufferData")
    constant = _declarations(literal_kernel())[0]
    with tvm.script.ir_builder.IRBuilder(), builder.prim_func():
        var = builder.bind(constant.value)
        builder.bind(tvm.tirx.call_extern("handle", "foreign_pointer"), var=var)
        before = tvm.ir.save_json(var.type_annotation)
        with pytest.raises(Exception, match="ambiguous Var redefinition"):
            prepare(var)
        assert tvm.ir.save_json(var.type_annotation) == before


def test_constant_bind_inside_vector_scope_is_mask_neutral():
    @T.prim_func
    def inside(Input: T.Tensor((1,), "int32"), Output: T.Tensor((1,), "int32")):
        with T.Scope("V"):
            table = T.allocate_const([7, 11, 23, 37], "int32", [4])
            T._src_code("Output.SetValue(0, table[Input.GetValue(0) & 3]);", table)

    source = _lower(inside).kernel_source
    assert "table[4]" in source and "SetVectorMask" not in source


def test_unused_opaque_literal_declaration_survives_dce():
    @T.prim_func
    def unused(Input: T.Tensor((1,), "int32"), Output: T.Tensor((1,), "int32")):
        table = T.allocate_const([7], "int32", [1])  # noqa: F841 - intentionally no dependency use
        with T.Scope("C"):
            T._src_code("Output.SetValue(0, Input.GetValue(0));")

    assert "table[1]" in _lower(unused).kernel_source


def test_unrelated_duplicate_binding_keeps_normal_decl_buffer_typing_unchanged():
    from tvm.tirx.script.builder import ir as builder

    prepare = tvm.ffi.get_global_func("tl.a2a3.PrepareConstBufferData")
    constant = _declarations(literal_kernel())[0]
    with tvm.script.ir_builder.IRBuilder(), builder.prim_func():
        ambiguous = builder.bind(tvm.tirx.call_extern("handle", "foreign_pointer"))
        builder.bind(tvm.tirx.call_extern("handle", "foreign_pointer"), var=ambiguous)
        foreign = builder.bind(tvm.tirx.call_extern("handle", "other_pointer"))
        before = tvm.ir.save_json(foreign.type_annotation)
        assert prepare(foreign).same_as(foreign)
        assert tvm.ir.save_json(foreign.type_annotation) == before
        live_const = builder.bind(constant.value)
        assert prepare(live_const).same_as(live_const)
        assert isinstance(live_const.type_annotation, tvm.ir.PointerType)


def test_frontend_normalization_preserves_the_literal_binding_source_span():
    from tilelang.ir import get_stmt_span, make_span, set_stmt_span

    func = literal_kernel()
    binding = _declarations(func)[0]
    set_stmt_span(binding, make_span("constant_input.py", 9))
    normalized = normalize_constants(func)
    span = get_stmt_span(_declarations(normalized)[0])
    assert span.source_name.name == "constant_input.py" and span.line == 9


def test_inline_macro_constant_uses_live_parent_builder_bindings():
    @T.macro
    def inline_table():
        pointer = T.allocate_const([7, 11, 23, 37], "int32", [4])
        table = T.decl_buffer((4,), "int32", data=pointer)  # noqa: F841 - raw source name
        with T.Scope("C"):
            T._src_code("Output.SetValue(0, table[Input.GetValue(0) & 3]);", pointer)

    @T.prim_func
    def kernel(Input: T.Tensor((1,), "int32"), Output: T.Tensor((1,), "int32")):
        inline_table()

    assert "table[4]" in _lower(kernel).kernel_source


def test_eager_macro_return_does_not_hide_a_live_constant_from_decl_buffer():
    @T.macro
    def make_table():
        pointer = T.allocate_const([7, 11, 23, 37], "int32", [4])
        return pointer

    @T.prim_func
    def kernel(Input: T.Tensor((1,), "int32"), Output: T.Tensor((1,), "int32")):
        pointer = make_table()
        table = T.decl_buffer((4,), "int32", data=pointer)  # noqa: F841 - raw source name
        with T.Scope("C"):
            T._src_code("Output.SetValue(0, table[Input.GetValue(0) & 3]);", pointer)

    # Eager value-returning macros are newer than TA's statement-only macros.
    # This checks builder scope handling, not a claim of legacy producer support.
    assert "table[4]" in _lower(kernel).kernel_source
