"""Public A2/A3 codegen and platform normalization without a device."""

import pytest

from tilelang import tvm
from tilelang.ascend import target as ascend_target
from tilelang.ascend.a2a3 import codegen
from tilelang.backend import create_backend_context
from tilelang.backend.target import normalize_target_for_execution
from tilelang.cache.kernel_cache import KernelCache
from tilelang.instrumentation import PassInstrumentationTool, compile_pass_instrumentation
from tvm.target import Target

from testing.ascend_a2a3.python.refactor.codegen.test_a2a3_codegen import _call, _make_func


@pytest.fixture(autouse=True)
def no_device(monkeypatch):
    monkeypatch.delenv("TL_PLATFORM", raising=False)
    monkeypatch.setattr(ascend_target, "_get_soc_name", lambda: None)


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("compile_device", [False, True])
def test_public_two_argument_codegen_passes_resolved_native_platform(monkeypatch, platform, compile_device):
    monkeypatch.setenv("TL_PLATFORM", platform)
    context = create_backend_context("ascendc", "c", "auto")
    # Cache identity is frozen in the canonical target, even if the environment
    # changes before code generation.
    monkeypatch.setenv("TL_PLATFORM", "A3" if platform == "A2" else "A2")
    mod = tvm.IRModule.from_expr(_make_func(_call("tl.ascend_pipe_barrier", "PIPE_V")))
    original = tvm.ffi.get_global_func
    received = []

    def lookup(name, *args, **kwargs):
        native = original(name, *args, **kwargs)
        if name != "target.build.tilelang_ascend_a2a3":
            return native

        def invoke(*args):
            received.append(args)
            return native(*args)

        return invoke

    monkeypatch.setattr(tvm.ffi, "get_global_func", lookup)
    result = context.codegen_device(mod, compile_device=compile_device)
    assert len(received) == 1
    native_mod, native_target, native_platform = received[0]
    assert native_mod.same_as(mod)
    assert native_target.same_as(context.target)
    assert native_platform == platform
    assert "AscendC::PipeBarrier<PIPE_V>();" in result.inspect_source()


def test_codegen_records_device_instrumentation(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "A2")
    context = create_backend_context("ascendc", "c", "auto")
    mod = tvm.IRModule.from_expr(_make_func(tvm.tirx.Evaluate(0)))
    events = []

    class Recorder(PassInstrumentationTool):
        def run_codegen(self, event, next_call):
            events.append(event)
            return next_call()

    with compile_pass_instrumentation(tools=[Recorder()], include_default_tools=False):
        context.codegen_device(mod, compile_device=False)
    assert len(events) == 1
    assert events[0].name == "target.build.tilelang_ascend_a2a3"
    assert events[0].role == "device"
    assert events[0].mod.same_as(mod)
    assert events[0].target.same_as(context.target)


def test_unknown_platform_keeps_context_usable_but_blocks_codegen():
    context = create_backend_context("ascendc", "c", "auto")
    assert context.name == "ascend_a2a3"
    with pytest.raises(ValueError, match="TL_PLATFORM|platform"):
        context.codegen_device(tvm.IRModule({}), compile_device=False)


def test_resolved_platform_is_part_of_the_kernel_cache_key(monkeypatch):
    func = _make_func(tvm.tirx.Evaluate(0))
    cache = KernelCache()
    targets, keys = [], []
    for platform in ["A2", "A3"]:
        monkeypatch.setenv("TL_PLATFORM", platform)
        context = create_backend_context("ascendc", "c", "auto")
        targets.append(str(context.target))
        keys.append(cache._generate_key(func, [], "cython", (), target=context.target))
    assert targets[0] != targets[1]
    assert keys[0] != keys[1]


def test_a5_stays_outside_and_pto_freezes_the_same_platform(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "A3")
    a5 = Target({"kind": "ascend", "arch": "dav-3510"})
    pto = Target({"kind": "ascend", "arch": "dav-2201", "keys": ["pto", "ascend"]})
    assert normalize_target_for_execution(a5, "auto").same_as(a5)
    normalized_pto = normalize_target_for_execution(pto, "auto")
    assert normalized_pto.attrs["model"] == "A3"
    assert list(normalized_pto.keys) == list(pto.keys)
    # Preserve the earlier shared PTO execution-name specialization. A2/A3
    # consumes either source dialect through the shared Cython executor.
    with pytest.raises(ValueError, match="Allowed: cython"):
        create_backend_context("ascendc", "c", "pto")
    result = codegen.PTO_CODEGEN.lower(tvm.IRModule.from_expr(_make_func(tvm.tirx.Evaluate(0))), normalized_pto, compile_device=False)
    assert "ascend/a2a3/tl_templates/pto/common.h" in result.inspect_source()
    assert 'extern "C" void call(' in result.inspect_source()
