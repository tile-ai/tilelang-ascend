"""Source-only contracts for the complete TA AscendC generator."""

import ctypes
import os

import pytest
from tilelang import tvm

if os.environ.get("A2A3_TEST_CODEGEN_LIBRARY"):
    # Optional isolated native module used during migration, before integration.
    _CODEGEN_LIBRARY = ctypes.CDLL(os.environ["A2A3_TEST_CODEGEN_LIBRARY"], mode=ctypes.RTLD_GLOBAL)


def _call(name, *args):
    args = [tvm.tirx.StringImm(arg) if isinstance(arg, str) else arg for arg in args]
    return tvm.tirx.Evaluate(tvm.tirx.Call("handle", tvm.ir.Op.get(name), args))


def _make_func(body, buffers=(), attrs=None, ratio="cv_1_2", imported=None, span=None):
    tirx = tvm.tirx
    buffers = buffers or (tirx.decl_buffer((1,), "float32", name="unused", scope="global"),)
    if imported:
        body = tirx.AttrStmt(tirx.IntImm("int32", 0), "pragma_import_c", tirx.StringImm(imported), body)
    cid, vid = tirx.Var("cid", "int32"), tirx.Var("vid", "int32")
    vector = tirx.IterVar(tvm.ir.Range(0, 2), vid, tirx.IterVar.ThreadIndex, "cthread")
    cube = tirx.IterVar(tvm.ir.Range(0, 4), cid, tirx.IterVar.ThreadIndex, "blockIdx.x")
    body = tirx.AttrStmt(cube, "thread_extent", 4, tirx.AttrStmt(vector, "thread_extent", 2, body))
    params = [tirx.Var("arg" + str(i), "handle") for i in range(len(buffers))]
    func = tirx.PrimFunc(params, body, buffer_map=dict(zip(params, buffers)), span=span)
    for key, value in {"global_symbol": "main", "npu_cv_ratio": tirx.StringImm(ratio), **(attrs or {})}.items():
        func = func.with_attr(key, value)
    return func


def _source(func, platform="A3", target=None, configs=None):
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}) if target is None else target
    with tvm.transform.PassContext(config=configs or {}):
        module = tvm.ffi.get_global_func("target.build.tilelang_ascend_a2a3")(tvm.IRModule.from_expr(func), target, platform)
    return module.inspect_source()


def _allocate(buffers, statements):
    tirx = tvm.tirx
    return tirx.SeqStmt([*(tirx.AllocBuffer(buf) for buf in buffers), *statements])


def _vector(body):
    return tvm.tirx.AttrStmt(tvm.tirx.IntImm("int32", 0), "resource_scope", 1, body)


def test_shared_sync_abi_preserves_ta_vector_barrier_and_cthread():
    body = tvm.tirx.SeqStmt(
        [
            _call("tl.ascend_set_flag", "MTE2_V", 3),
            _call("tl.ascend_wait_flag", "MTE2_V", 3),
            _call("tl.ascend_pipe_barrier", "PIPE_V"),
        ]
    )
    text = _source(_make_func(body))
    assert "AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);" in text
    assert "AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);" in text
    assert "AscendC::PipeBarrier<PIPE_V>();" in text
    assert "auto vid = AscendC::GetSubBlockIdx();" in text
    assert "cid = cid / 2;" in text
    assert "main_kernel<<<4, nullptr, stream>>>" in text
    assert 'extern "C" __global__ __aicore__ void main(' not in text


@pytest.mark.parametrize("ratio,task", [("cv_1_1", "KERNEL_TYPE_MIX_AIC_1_1"), ("cv_1_2", "KERNEL_TYPE_MIX_AIC_1_2")])
def test_ta_cv_ratio_is_retained(ratio, task):
    text = _source(_make_func(tvm.tirx.Evaluate(0), ratio=ratio))
    assert task in text
    assert ("cid = cid / 2;" in text) == (ratio == "cv_1_2")


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_platform_requires_explicit_a2a3_and_fixed_arch(platform):
    func = _make_func(tvm.tirx.Evaluate(0))
    text = _source(func, platform)
    for scope, capacity in [
        ("ascend_l0a", 65536),
        ("ascend_l0b", 65536),
        ("ascend_l1", 524032),
        ("ascend_l0c", 131072),
        ("ascend_ub", 196352),
    ]:
        assert f"pipe.InitBuffer({scope}, {capacity});" in text
    with pytest.raises(tvm.error.InternalError, match="resolved A2/A3 platform"):
        _source(func, "A5")
    with pytest.raises(tvm.error.InternalError, match="arch=dav-2201"):
        _source(func, platform, tvm.target.Target("ascend"))


def test_backend_headers_precede_imports_and_generated_functions():
    imported = "inline int imported_helper(int value) { return value + 1; }"
    text = _source(_make_func(_call("tl.ascend_src_code", "int result = imported_helper(1);"), imported=imported))
    assert text.index('#include "ascend/a2a3/tl_templates/ascend/common.h"') < text.index(imported)
    assert text.index(imported) < text.index('extern "C" __global__ __aicore__')
    assert text.index(imported) < text.index("int result = imported_helper(1);")


@pytest.mark.parametrize(
    "scope,storage",
    [
        ("shared.l1", "ascend_l1"),
        ("shared.ub", "ascend_ub"),
        ("wmma.matrix_a", "ascend_l0a"),
        ("wmma.matrix_b", "ascend_l0b"),
        ("wmma.accumulator", "ascend_l0c"),
    ],
)
def test_allocbuffer_keeps_every_ta_storage_dispatch(scope, storage):
    buf = tvm.tirx.decl_buffer((16, 32), "float16", name="tile", scope=scope)
    text = _source(_make_func(_allocate([buf], [tvm.tirx.Evaluate(0)]), attrs={"address_map": {buf.data: 128}}))
    assert f"{storage}.GetWithOffset<half>(512, 128);" in text


def test_local_var_initializer_and_flat_sequence_use():
    tirx = tvm.tirx
    buf = tirx.decl_buffer((1,), "int32", name="counter", scope="local.var")
    body = tirx.SeqStmt([tirx.AllocBuffer(buf, annotations={"tl.local_var_init": 7}), tirx.BufferStore(buf, 11, [0])])
    text = _source(_make_func(body))
    assert "int counter = 7;" in text
    assert "counter = 11;" in text


def test_selected_vector_terminal_and_payload_validation():
    tirx = tvm.tirx
    a, b, c = [tirx.decl_buffer((64,), "float32", name=name, scope="shared.ub") for name in ("a", "b", "c")]
    args = [
        c.access_ptr("w"),
        a.access_ptr("r"),
        b.access_ptr("r"),
        64,
        0,
        1,
        tirx.IntImm("uint64", (1 << 63) - 1),
        tirx.IntImm("uint64", 0),
    ]
    selected = _call("tl.ascend_add_raw", *args)
    attrs = {"address_map": {buf.data: index * 256 for index, buf in enumerate((a, b, c))}}
    text = _source(_make_func(_vector(_allocate([a, b, c], [selected])), attrs=attrs))
    assert "AscendC::Add<float, false>" in text
    assert "AscendC::MASK_PLACEHOLDER" in text
    with pytest.raises(tvm.error.InternalError, match="explicit AIV"):
        _source(_make_func(_allocate([a, b, c], [selected]), attrs=attrs))
    invalid = list(args)
    invalid[-2] = tirx.IntImm("int32", 3)
    with pytest.raises(tvm.error.InternalError, match="uint64"):
        _source(_make_func(_vector(_allocate([a, b, c], [_call("tl.ascend_add_raw", *invalid)])), attrs=attrs))


@pytest.mark.parametrize("platform,repeat_zero", [("A2", "false"), ("A3", "true")])
def test_reduce2d_retains_a2_a3_repeat_zero_difference(platform, repeat_zero):
    tirx = tvm.tirx
    dst, src, tmp = [
        tirx.decl_buffer((size,), "float32", name=name, scope="shared.ub") for name, size in [("out", 1), ("src", 32), ("tmp", 128)]
    ]
    call = _call(
        "tl.ascend_reduce_fp32_v2", "reduce_sum<float, 1, 32, -1>", dst.access_ptr("w"), src.access_ptr("r"), tmp.access_ptr("rw"), True
    )
    attrs = {"address_map": {buf.data: index * 512 for index, buf in enumerate((dst, src, tmp))}}
    text = _source(_make_func(_vector(_allocate([dst, src, tmp], [call])), attrs=attrs), platform)
    assert f"reduce_2d<float, reduce2d_v2::Reduce2DKind::kSum, true, 1, 32, -1, 32, {repeat_zero}>" in text


def test_vector_catalog_is_closed_and_uses_mapped_ffi():
    names = list(tvm.ffi.get_global_func("tl.transform.A2A3AscendVectorTerminalCatalog")())
    assert len(names) == len(set(names)) == 90
    for name in names:
        assert tvm.ir.Op.get(name).name == name


def test_dynamic_gm_shape_is_preserved_in_host_and_kernel_signatures():
    tirx = tvm.tirx
    size = tirx.Var("dynamic_m", "int32")
    source = tirx.decl_buffer((size, 64), "float32", name="input", scope="global")
    text = _source(_make_func(tirx.Evaluate(0), [source]))
    assert "int64_t dynamic_m" in text
    assert "void main_kernel_tiling(int64_t dynamic_m)" in text
    assert "main_kernel<<<4, nullptr, stream>>>(arg0, dynamic_m, fftsAddr);" in text


@pytest.mark.parametrize("enabled", [False, True])
def test_function_span_reaches_common_line_directive_support(enabled):
    span = tvm.ir.Span(tvm.ir.SourceName("ta_kernel.py"), 12, 12, 1, 5)
    text = _source(_make_func(tvm.tirx.Evaluate(0), span=span), configs={"tl.emit_line_directives": enabled})
    assert ('#line 12 "ta_kernel.py"' in text) == enabled


def test_exception_dump_preamble_and_original_parameter_metadata():
    buf = tvm.tirx.decl_buffer((16, 32), "float32", name="input", scope="global")
    text = _source(_make_func(tvm.tirx.Evaluate(0), [buf]), configs={"tl.ascend_exception_dump": True})
    assert '#include "ascend/a2a3/tl_templates/ascend/exception_dump.h"' in text
    assert "tilelang_register_exception_dump_callback();" in text
    assert "paramSizeInfo.count = 1;" in text
    assert "paramSizeInfo.dataTypes[0] = 0;" in text
    assert "ParamSizeInfo paramSizeInfo" in text


def test_ta_export_metadata_and_kernel_call_use_global_symbol():
    func = _make_func(tvm.tirx.Evaluate(0)).with_attr("global_symbol", "custom_symbol")
    mod = tvm.IRModule({tvm.ir.GlobalVar("module_lookup_key"): func})
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    generated = tvm.ffi.get_global_func("target.build.tilelang_ascend_a2a3")(mod, target, "A3")
    assert list(generated.get_function("get_func_names")()) == ["custom_symbol"]
    text = generated.inspect_source()
    assert "void custom_symbol_kernel(" in text
    assert "custom_symbol_kernel<<<4, nullptr, stream>>>" in text
    assert "void custom_symbol(" not in text
