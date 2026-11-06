# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""S3 loop metadata and legacy PackedFunc ABI; no device execution."""

import pytest
import tvm
from tvm import tir
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tir as I

import tilelang
import tilelang.language as T
from tilelang import _ffi_api


def _function(frame_factory):
    with IRBuilder() as builder, I.prim_func():
        I.func_name("main")
        with frame_factory() as variables:
            variable = variables if isinstance(variables, tir.Var) else variables[0]
            I.evaluate(variable)
    return builder.get()


def _loops(function):
    loops = []
    tir.stmt_functor.post_order_visit(function.body, lambda node: loops.append(node) if isinstance(node, tir.For) else None)
    return list(reversed(loops))


def test_parallel_metadata_outermost_and_legacy_coalesced_width():
    metadata = {"user.loop_tag": "outer", "coalesced_width": 4}
    function = _function(lambda: T.Parallel(3, 5, coalesced_width=4, annotations=metadata))
    outer, inner = _loops(function)
    assert outer.kind == inner.kind == tir.ForKind.PARALLEL
    assert str(outer.annotations["user.loop_tag"]) == "outer"
    assert "user.loop_tag" not in inner.annotations
    assert int(outer.annotations["coalesced_width"]) == 4
    assert int(inner.annotations["coalesced_width"]) == 4
    assert metadata == {"user.loop_tag": "outer", "coalesced_width": 4}


def test_parallel_coalesced_width_conflict_is_explicit():
    with pytest.raises(ValueError, match="conflicting coalesced_width"):
        T.Parallel(3, coalesced_width=4, annotations={"coalesced_width": 8})


@pytest.mark.parametrize("num_stages", [0, -1, 2])
def test_pipelined_legacy_arity_and_stage_semantics(num_stages):
    old = _function(lambda: _ffi_api.Pipelined(1, 5, num_stages, [], [], [], [], 1))
    new = _function(lambda: T.Pipelined(1, 5, num_stages))
    assert tvm.ir.structural_equal(old, new)
    loop = _loops(new)[0]
    assert int(loop.min) == 1 and int(loop.extent) == 4
    assert ("num_stages" in loop.annotations) == (num_stages > 0)
    assert int(loop.annotations["tl_cross_interval"]) == 1


def test_pipelined_default_zero_none_and_manual_abi():
    default = _function(lambda: T.Pipelined(5))
    zero = _function(lambda: T.Pipelined(5, num_stages=0))
    assert tvm.ir.structural_equal(default, zero)
    with pytest.raises((TypeError, tvm.TVMError)):
        T.Pipelined(5, num_stages=None)
    metadata = {"user.loop_tag": "manual"}
    function = _function(lambda: T.Pipelined(1, 5, 0, [0], [0], [[0, 1]], [[0]], 3, annotations=metadata))
    annotation = _loops(function)[0].annotations
    assert "num_stages" not in annotation
    assert list(annotation["tl_pipeline_order"]) == [0]
    assert list(annotation["tl_pipeline_stage"]) == [0]
    assert [list(row) for row in annotation["tl_pipeline_sync"]] == [[0, 1]]
    assert [list(row) for row in annotation["tl_pipeline_group"]] == [[0]]
    assert int(annotation["tl_cross_interval"]) == 3
    assert str(annotation["user.loop_tag"]) == "manual"
    assert metadata == {"user.loop_tag": "manual"}


def test_persistent_legacy_arity_and_positive_single_fullwave_metadata():
    old = _function(lambda: _ffi_api.Persistent([4, 2], 8, 0, 8))
    default = _function(lambda: T.Persistent([4, 2], 8, 0))
    zero = _function(lambda: T.Persistent([4, 2], 8, 0, 8, 0))
    # The legacy Persistent builder emits coordinate lets around its wave loop;
    # their pure expressions refer to w before Simplify inlines the lets.
    assert tvm.ir.structural_equal(old, default, map_free_vars=True)
    assert tvm.ir.structural_equal(default, zero, map_free_vars=True)
    metadata = {"user.loop_tag": "wave"}
    positive = _function(lambda: T.Persistent([4, 2], 8, 0, 8, 2, annotations=metadata))
    loop = _loops(positive)[0]
    assert int(loop.annotations["num_stages"]) == 2
    assert str(loop.annotations["user.loop_tag"]) == "wave"
    assert not isinstance(loop.body, tir.IfThenElse)
    assert metadata == {"user.loop_tag": "wave"}


@pytest.mark.parametrize(
    "domain,wave_size,group_size",
    [([4, 2], 20, 8), ([4, 2], 4, 8), ([2, 3], 8, 2)],
)
@pytest.mark.parametrize("direct_ffi", [False, True])
def test_persistent_guarded_positive_stages_require_s5(domain, wave_size, group_size, direct_ffi):
    def frame():
        if direct_ffi:
            return _ffi_api.Persistent(domain, wave_size, 0, group_size, 2, {})
        return T.Persistent(domain, wave_size, 0, group_size, 2)

    function = _function(frame)
    assert bool(_loops(function)[0].annotations["tl.ascend_persistent_guard"])
    with pytest.raises(tvm.TVMError, match="requires the constraint scheduling mode"):
        tilelang.transform.AscendValidateScheduleHints()(tvm.IRModule.from_expr(function))


def test_persistent_symbolic_positive_stages_require_s5():
    function = _function(lambda: T.Persistent([T.symbolic("rows"), 2], 8, 0, num_stages=2))
    assert bool(_loops(function)[0].annotations["tl.ascend_persistent_guard"])
    with pytest.raises(tvm.TVMError, match="requires the constraint scheduling mode"):
        tilelang.transform.AscendValidateScheduleHints()(tvm.IRModule.from_expr(function))


def test_persistent_nondivisible_group_is_padded_and_guarded():
    function = _function(lambda: T.Persistent([2, 3], 5, 0, 2))
    loop = _loops(function)[0]
    assert int(tvm.arith.Analyzer().simplify(loop.extent)) == 2  # padded 8 tiles / wave 5
    assert isinstance(loop.body, tir.SeqStmt)
    assert isinstance(loop.body.seq[0], tir.IfThenElse)  # partial final wave exits
    assert isinstance(loop.body.seq[1], tir.IfThenElse)  # padded tile excluded
    assert "num_stages" not in loop.annotations


def test_persistent_large_positive_domain_uses_wide_schedule_arithmetic():
    function = _function(lambda: T.Persistent([65536, 65536], 20, 0, 8))
    loop = _loops(function)[0]
    assert loop.loop_var.dtype == "int64"
    assert int(tvm.arith.Analyzer().simplify(loop.extent)) == (2**32 + 19) // 20
    condition = loop.body.seq[0].condition
    assert condition.a.dtype == condition.b.dtype == "int64"
    assert int(condition.a) == 2**32
    # The user-facing coordinate variables retain their declared int32 dtype.
    lets = []
    tir.stmt_functor.post_order_visit(function.body, lambda node: lets.append(node) if isinstance(node, tir.LetStmt) else None)
    assert len(lets) == 2
    assert all(binding.var.dtype == binding.value.dtype == "int32" for binding in lets)
    for worker, wave, expected in [(0, 0, (0, 0)), (15, (2**32 - 1) // 20, (65535, 65535))]:
        sample = _function(lambda worker=worker: T.Persistent([65536, 65536], 20, worker, 8))
        sample_loop = _loops(sample)[0]
        bindings = []
        tir.stmt_functor.post_order_visit(
            sample.body,
            lambda node, bindings=bindings: bindings.append(node) if isinstance(node, tir.LetStmt) else None,
        )
        coordinates = {
            str(binding.var.name): int(
                tvm.arith.Analyzer().simplify(tir.stmt_functor.substitute(binding.value, {sample_loop.loop_var: tir.IntImm("int64", wave)}))
            )
            for binding in bindings
        }
        assert (coordinates["v0"], coordinates["v1"]) == expected


def test_persistent_static_padded_product_overflow_is_rejected():
    with pytest.raises(tvm.TVMError, match="padded domain size must fit int64"):
        T.Persistent([tir.IntImm("int64", 2**62), 4], 20, 0)


@pytest.mark.parametrize(
    "domain,wave_size,index,group_size,num_stages,error",
    [
        ([], 8, 0, 8, 0, "domain must not be empty"),
        ([0, 2], 8, 0, 8, 0, "domain extents must be positive"),
        ([4, 2], 0, 0, 8, 0, "wave_size must be positive"),
        ([4, 2], 8, -1, 8, 0, "index must satisfy"),
        ([4, 2], 8, 8, 8, 0, "index must satisfy"),
        ([4, 2], 8, 0, 0, 0, "group_size must be positive"),
        ([4, 2], 8, 0, 8, -1, "num_stages must be nonnegative"),
        ([tir.FloatImm("float32", 4), 2], 8, 0, 8, 0, "integer scalar"),
        ([4, 2], 8, tir.const(True, "bool"), 8, 0, "integer scalar"),
    ],
)
def test_persistent_invalid_domain_is_rejected(domain, wave_size, index, group_size, num_stages, error):
    with pytest.raises(tvm.TVMError, match=error):
        _ffi_api.Persistent(domain, wave_size, index, group_size, num_stages, {})


_RESERVED = [
    "num_stages",
    "tl_pipeline_order",
    "tl_pipeline_stage",
    "tl_pipeline_sync",
    "tl_pipeline_group",
    "tl_cross_interval",
    "software_pipeline_order",
    "software_pipeline_stage",
    "software_pipeline_async_stages",
    "tl.ascend_task",
    "tl.ascend_stage",
    "tl.ascend_storage_alias_provenance",
    "tl.ascend_storage_view_provenance",
    "tl.ascend_storage_source_data",
    "tl.ascend_storage_original_parameters",
    "enable_offset",
    "parallel_loop_layout",
]


@pytest.mark.parametrize("key", _RESERVED)
@pytest.mark.parametrize("builder", ["parallel", "pipelined", "persistent"])
def test_reserved_annotations_rejected_by_python_and_native(key, builder):
    metadata = {key: 1}

    def python():
        if builder == "parallel":
            return T.Parallel(4, annotations=metadata)
        if builder == "pipelined":
            return T.Pipelined(4, annotations=metadata)
        return T.Persistent([4, 2], 8, 0, annotations=metadata)

    def native():
        if builder == "parallel":
            return _ffi_api.Parallel([4], metadata)
        if builder == "pipelined":
            return _ffi_api.Pipelined(0, 4, 0, [], [], [], [], 1, metadata)
        return _ffi_api.Persistent([4, 2], 8, 0, 8, 0, metadata)

    if key == "enable_offset" and builder == "pipelined":
        python_error, native_error = "bool", "boolean"
    else:
        python_error = native_error = "reserved"
    python_exception = TypeError if key == "enable_offset" and builder == "pipelined" else ValueError
    with pytest.raises(python_exception, match=python_error):
        python()
    with pytest.raises(tvm.TVMError, match=native_error):
        native()


@pytest.mark.parametrize("builder", [T.Parallel, T.Pipelined])
def test_annotations_require_string_key_mapping(builder):
    with pytest.raises(TypeError, match="mapping"):
        builder(4, annotations=[("user.tag", 1)])
    with pytest.raises(TypeError, match="string keys"):
        builder(4, annotations={1: "value"})


@pytest.mark.parametrize("builder", ["parallel", "pipelined", "persistent"])
@pytest.mark.parametrize("metadata", [None, [1]])
def test_native_rejects_malformed_annotation_maps(builder, metadata):
    with pytest.raises((TypeError, tvm.TVMError)):
        if builder == "parallel":
            _ffi_api.Parallel([4], metadata)
        elif builder == "pipelined":
            _ffi_api.Pipelined(0, 4, 0, [], [], [], [], 1, metadata)
        else:
            _ffi_api.Persistent([4, 2], 8, 0, 8, 0, metadata)


def test_nondivisible_partial_single_wave_has_no_dangling_break_in_codegen():
    @T.prim_func
    def main(O: T.Tensor((2, 3), "int32")):
        with T.Kernel(20, is_npu=True) as (cid, _):
            for row, col in T.Persistent([2, 3], 20, cid, 2):
                with T.Scope("C"):
                    O[row, col] = row * 3 + col

    artifact = tilelang.lower(main, target="ascendc")
    assert "break;" not in artifact.kernel_source
