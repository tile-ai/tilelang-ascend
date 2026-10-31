"""Public lowering can consume pipeline-owned final argument metadata."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tilelang import tvm
from tilelang.ascend.a2a3.codegen import _preserve_physical_terminals
from tilelang.ascend.a2a3.pipeline import ASCENDC_PIPELINE, get_device_kernel
from tilelang.backend import BackendContext, get_backend
from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.engine.lower import lower_with_context
from tvm.ir import CallingConv

ir = tvm.tirx


def _input():
    n = ir.Var("n", "int32")
    a = ir.decl_buffer((n, 16), "float16", name="A")
    func = ir.PrimFunc([a.data, n], ir.Evaluate(0), buffer_map={a.data: a}).with_attr("global_symbol", "main")
    return func, n


def _context(extractor):
    # This shared metadata contract also runs against a real USE_ASCEND=OFF library.
    base = get_backend("cpu")
    target = tvm.target.Target("c")

    def lower(mod, target):
        original = mod["main"]
        temporary = ir.decl_buffer((original.params[1], 2), "uint8", name="automatic_gm")
        func = ir.PrimFunc(
            [*original.params, temporary.data],
            original.body,
            buffer_map={**dict(original.buffer_map), temporary.data: temporary},
            attrs=original.attrs,
        ).with_attr({"calling_conv": CallingConv.DEVICE_KERNEL_LAUNCH, "target": target, "auto_gm_indices": [2]})
        return tvm.IRModule({"main": func})

    codegen = DeviceCodegen(
        "parameter-contract",
        build_without_compile=lambda _mod, _target: SimpleNamespace(inspect_source=lambda: "source-only"),
        preprocess=_preserve_physical_terminals,
    )
    module = replace(
        base,
        pipelines={kind: PassPipeline(kind, lower, parameter_extractor=extractor) for kind in base.target_kinds},
        device_codegens={kind: codegen for kind in base.target_kinds},
    )
    return BackendContext(module, target, tvm.target.Target("c"), module.execution_backends[0])


def test_public_lower_refreshes_final_user_dynamic_and_automatic_gm_parameters():
    original, n = _input()
    artifact = lower_with_context(original, _context(ASCENDC_PIPELINE.parameter_extractor))
    assert [str(p.dtype) for p in artifact.params] == ["float16", "int32", "uint8"]
    assert artifact.params[0].shape[0].same_as(n)
    assert artifact.params[0].shape[1] == 16
    assert artifact.params[1].shape == []
    assert artifact.params[2].shape[0].same_as(n)
    assert artifact.params[2].shape[1] == 2
    func = get_device_kernel(artifact.device_mod)
    assert [p.name for p in func.params] == ["A", "n", "automatic_gm"]
    assert [int(index) for index in func.attrs["auto_gm_indices"]] == [2]
    assert func.body.same_as(original.body)


def test_default_pipeline_keeps_original_input_parameter_extraction():
    original, _ = _input()
    artifact = lower_with_context(original, _context(None))
    assert [str(p.dtype) for p in artifact.params] == ["float16", "int32"]
    assert len(get_device_kernel(artifact.device_mod).params) == 3


def test_runtime_only_does_not_invoke_parameter_extractor():
    def unexpected(_mod):
        raise AssertionError("runtime_only must not extract parameters")

    original, _ = _input()
    artifact = lower_with_context(original, _context(unexpected), runtime_only=True)
    assert artifact.params is None


def test_a2a3_final_parameter_reader_rejects_multiple_exports_and_host_conventions():
    original, _ = _input()
    device = original.with_attr("calling_conv", CallingConv.DEVICE_KERNEL_LAUNCH)
    with pytest.raises(ValueError, match="exactly one exported"):
        get_device_kernel(tvm.IRModule({"main": device, "second": device.with_attr("global_symbol", "second")}))
    with pytest.raises(ValueError, match="host calling conventions"):
        get_device_kernel(tvm.IRModule.from_expr(original.with_attr("calling_conv", CallingConv.C_PACKED_FUNC)))
    with pytest.raises(ValueError, match="global_symbol"):
        get_device_kernel(tvm.IRModule({"main": device.without_attr("global_symbol")}))
