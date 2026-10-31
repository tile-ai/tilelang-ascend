"""Backend-owned preparation preserves final target-specific physical IR."""

from dataclasses import replace

import pytest

from tilelang import tvm
from tilelang.backend import BackendContext, get_backend
from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.engine.lower import device_codegen, device_codegen_without_compile


def _context(codegen):
    module = get_backend("cpu")
    module = replace(module, device_codegens={kind: codegen for kind in module.target_kinds})
    return BackendContext(module, tvm.target.Target("c"), tvm.target.Target("c"), module.execution_backends[0])


def _module():
    return tvm.IRModule.from_expr(
        tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(0)).with_attr({"global_symbol": "main", "target": tvm.target.Target("c")})
    )


@tvm.instrument.pass_instrument
class _Trace:
    def __init__(self):
        self.names = []

    def run_before_pass(self, _mod, info):
        self.names.append(info.name)


@pytest.mark.parametrize("compile_device", [False, True])
def test_default_device_preparation_keeps_existing_three_pass_order(compile_device):
    built = []

    def build(mod, _target):
        built.append(mod)
        return mod

    context = _context(DeviceCodegen("unit", build=build, build_without_compile=build))
    trace = _Trace()
    with tvm.transform.PassContext(instruments=[trace]):
        result = (device_codegen if compile_device else device_codegen_without_compile)(_module(), context)
    assert result.same_as(built[0])
    # HoistBroadcastValues is a Python PrimFunc pass whose existing name is pass_fn.
    assert trace.names == ["tl.LowerIntrin", "tirx.Simplify", "pass_fn"]


@pytest.mark.parametrize("compile_device", [False, True])
def test_explicit_identity_preparation_runs_zero_tir_transforms(compile_device):
    from tilelang.ascend.a2a3.codegen import _preserve_physical_terminals

    original = _module()
    received = []

    def build(mod, target):
        received.append((mod, target))
        return mod

    context = _context(
        DeviceCodegen("unit-final-terminals", build=build, build_without_compile=build, preprocess=_preserve_physical_terminals)
    )
    trace = _Trace()
    with tvm.transform.PassContext(instruments=[trace]):
        result = (device_codegen if compile_device else device_codegen_without_compile)(original, context)
    assert result.same_as(original)
    assert received[0][0].same_as(original)
    assert received[0][1].same_as(context.target)
    assert not trace.names


@pytest.mark.parametrize("compile_device", [False, True])
def test_direct_device_codegen_lower_retains_builder_only_contract(compile_device):
    original = _module()

    def unexpected_prepare(_mod, _target):
        raise AssertionError("direct lower must not add compiler-entry preparation")

    codegen = DeviceCodegen("direct", build=lambda m, _t: m, build_without_compile=lambda m, _t: m, preprocess=unexpected_prepare)
    assert codegen.lower(original, tvm.target.Target("c"), compile_device=compile_device).same_as(original)


@pytest.mark.parametrize("compile_device", [False, True])
def test_mask_legalized_physical_ir_reaches_builder_without_body_passes(compile_device):
    from tilelang.ascend.a2a3.codegen import _preserve_physical_terminals

    select = tvm.ffi.get_global_func("tl.transform.A2A3AscendVectorInstructionSelection", allow_missing=True)
    legalize = tvm.ffi.get_global_func("tl.transform.A2A3AscendVectorMaskLegalize", allow_missing=True)
    if select is None or legalize is None:
        pytest.skip("A2/A3 native vector passes are not compiled")
    ir = tvm.tirx
    buffers = [ir.decl_buffer((64,), "float32", name=name, scope="shared.ub") for name in ["dst", "src0", "src1"]]
    call = ir.Call(
        "handle",
        tvm.ir.Op.get("tl.ascend_add"),
        [buffers[0].access_ptr("w"), buffers[1].access_ptr("r"), buffers[2].access_ptr("r"), ir.IntImm("int32", 64)],
    )
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "keys": ["ascendc"], "model": "A3"})
    body = ir.AttrStmt(0, "resource_scope", 1, ir.Evaluate(call))
    func = ir.PrimFunc([b.data for b in buffers], body, buffer_map={b.data: b for b in buffers}).with_attr(
        {"global_symbol": "main", "target": target}
    )
    original = legalize(target, "A3")(select(target, "A3")(tvm.IRModule.from_expr(func)))
    physical_names = []
    ir.stmt_functor.post_order_visit(original["main"].body, lambda n: physical_names.append(n.op.name) if isinstance(n, ir.Call) else None)
    assert "tl.ascend_add_raw" in physical_names
    context = _context(
        DeviceCodegen("final-mask", build=lambda m, _t: m, build_without_compile=lambda m, _t: m, preprocess=_preserve_physical_terminals)
    )
    trace = _Trace()
    with tvm.transform.PassContext(instruments=[trace]):
        result = (device_codegen if compile_device else device_codegen_without_compile)(original, context)
    assert result["main"].same_as(original["main"])
    assert result["main"].body.same_as(original["main"].body)
    assert not trace.names
