from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import gmla.runtime as runtime_module
from gmla import _selection
from gmla._contracts import validate_chain_spans, public_prefill_block_n
from gmla._runtime import DEFAULT_SCALE, validate_scale
from gmla.layout import InternalFormatError

from .test_contracts import ShapeTensor, extracted, fail_if_called


NAMES = ("q_content", "q_aux", "latent_kv", "key_aux", "w_k", "w_v", "gate_logits", "metadata")


def fixture(index=0, *, current=0, grad=False):
    device = SimpleNamespace(type="npu", index=index)
    shapes = [(1, 1, 1, 128), (1, 1, 1, 64), (1, 17, 512), (1, 17, 64), (1, 128, 512), (1, 128, 512), (1, 1, 1, 128), (1, 8)]
    values = [ShapeTensor(shape, "i32" if i == 7 else "fp16", device) for i, shape in enumerate(shapes)]
    calls = {"device": 0, "formats": [], "validation": 0, "execution": 0}

    def current_device():
        calls["device"] += 1
        return current

    def format_guard(value, *, name, **kwargs):
        calls["formats"].append(name)
        if getattr(value, "bad_format", False):
            raise InternalFormatError(name + " actual physical format is not admitted")
        return value

    layout = SimpleNamespace(require_dense_base=format_guard)
    fake_torch = SimpleNamespace(
        Tensor=ShapeTensor,
        is_grad_enabled=lambda: grad,
        float16="fp16",
        bfloat16="bf16",
        int32="i32",
        npu=SimpleNamespace(current_device=current_device),
    )
    validate = extracted(
        "gmla/_runtime.py",
        "validate_forward_inputs",
        {
            "torch": fake_torch,
            "validate_scale": validate_scale,
            "DEFAULT_SCALE": DEFAULT_SCALE,
            "validate_chain_spans": validate_chain_spans,
            "default_layout": layout,
        },
    )

    def checked(*args, **kwargs):
        calls["validation"] += 1
        return validate(*args, **kwargs)

    return values, calls, layout, checked


@pytest.mark.parametrize("operand", range(8), ids=NAMES)
def test_every_physical_format_is_rejected_before_bundle_loading(monkeypatch, operand):
    values, calls, _, checked = fixture()
    values[operand].bad_format = True
    monkeypatch.setattr(runtime_module, "validate_forward_inputs", checked)
    monkeypatch.setattr(runtime_module.GatedMLARuntime, "kernel_bundle", property(fail_if_called))
    with pytest.raises(InternalFormatError, match=NAMES[operand]):
        runtime_module.build_runtime().forward(*values)
    assert calls["validation"] == 1 and calls["execution"] == 0


@pytest.mark.parametrize("operand", range(8), ids=NAMES)
def test_logical_wrong_device_rejected_before_current_device_lookup(monkeypatch, operand):
    values, calls, _, checked = fixture()
    values[operand].device = SimpleNamespace(type="cpu", index=None)
    monkeypatch.setattr(runtime_module, "validate_forward_inputs", checked)
    monkeypatch.setattr(runtime_module.GatedMLARuntime, "kernel_bundle", property(fail_if_called))
    with pytest.raises(ValueError):
        runtime_module.build_runtime().forward(*values)
    assert calls["device"] == 0 and calls["formats"] == []


@pytest.mark.parametrize("operand", range(8), ids=NAMES)
def test_all_operand_shape_errors_precede_backend_and_bundle(monkeypatch, operand):
    values, calls, _, checked = fixture()
    values[operand].shape = (*values[operand].shape[:-1], values[operand].shape[-1] + 1)
    monkeypatch.setattr(runtime_module, "validate_forward_inputs", checked)
    monkeypatch.setattr(runtime_module.GatedMLARuntime, "kernel_bundle", property(fail_if_called))
    with pytest.raises(ValueError):
        runtime_module.build_runtime().forward(*values)
    assert calls["device"] == 0 and calls["formats"] == []


def test_mismatched_current_npu_rejected_without_switch_or_bundle(monkeypatch):
    values, calls, _, checked = fixture(index=1, current=0)
    monkeypatch.setattr(runtime_module, "validate_forward_inputs", checked)
    monkeypatch.setattr(runtime_module.GatedMLARuntime, "kernel_bundle", property(fail_if_called))
    with pytest.raises(ValueError, match="current device"):
        runtime_module.build_runtime().forward(*values)
    assert calls["device"] == 1 and calls["formats"] == []


def test_grad_rejected_before_owner_device_format_or_bundle(monkeypatch):
    values, calls, _, checked = fixture(grad=True)
    values[6].requires_grad = True
    monkeypatch.setattr(runtime_module, "validate_forward_inputs", checked)
    monkeypatch.setattr(runtime_module.GatedMLARuntime, "kernel_bundle", property(fail_if_called))
    with pytest.raises(RuntimeError, match="inference-only"):
        runtime_module.build_runtime().forward(*values)
    assert calls["device"] == 0 and calls["formats"] == []


def test_public_runtime_validates_once_then_private_execution(monkeypatch):
    values, calls, _, checked = fixture()
    events = []

    def execute(*args, **kwargs):
        calls["execution"] += 1
        events.append("execute")
        assert args[-1] == (1, 1, 1, 17, DEFAULT_SCALE)
        return "output"

    bundle = SimpleNamespace(_forward_validated=execute, forward=fail_if_called)

    def get_bundle(self):
        events.append("bundle")
        assert calls["validation"] == 1 and calls["formats"] == list(NAMES)
        return bundle

    monkeypatch.setattr(runtime_module, "validate_forward_inputs", checked)
    monkeypatch.setattr(runtime_module.GatedMLARuntime, "kernel_bundle", property(get_bundle))
    assert runtime_module.build_runtime().forward(*values) == "output"
    assert events == ["bundle", "execute"]
    assert calls["validation"] == calls["execution"] == 1
    assert calls["formats"] == list(NAMES)


def test_direct_bundle_forward_uses_same_preflight_once():
    values, calls, layout, checked = fixture()
    bundle = SimpleNamespace(layout_module=layout, block_m=32, block_n=64, max_blocks=20, _forward_validated=lambda *a, **k: "output")
    forward = extracted("gmla/_runtime.py", "forward", {"validate_forward_inputs": checked}, class_name="KernelBundle")
    assert forward(bundle, *values) == "output"
    assert calls["validation"] == 1 and calls["formats"] == list(NAMES)


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((1, 128, 96, 128), 128),
        ((2, 128, 96, 128), 64),
        ((1, 127, 96, 128), 64),
        ((1, 129, 96, 128), 64),
        ((1, 128, 95, 128), 64),
        ((1, 128, 97, 128), 64),
        ((1, 128, 96, 127), 64),
        ((1, 128, 96, 129), 64),
        ((1, 1, 96, 128), 64),
        ((1, 0, 96, 128), 64),
        ((1, 128, 96, 0), 64),
    ],
)
def test_profile_is_exact_shape_only(shape, expected):
    assert public_prefill_block_n(*shape, DEFAULT_SCALE) == expected


def test_custom_scale_retains_old_tile():
    assert public_prefill_block_n(1, 128, 96, 128, DEFAULT_SCALE * 0.75) == 64


def test_profile_preflight_checks_actual_tile_and_explicit_bundle_stays_64():
    device = SimpleNamespace(type="npu", index=0)
    shapes = [(1, 128, 96, 128), (1, 128, 96, 64), (1, 128, 512), (1, 128, 64), (96, 128, 512), (96, 128, 512), (1, 128, 96, 128), (1, 8)]
    inputs = [ShapeTensor(shape, "i32" if i == 7 else "fp16", device) for i, shape in enumerate(shapes)]
    observed = []

    def spans(*args, **kwargs):
        observed.append(kwargs["block_n"])
        return validate_chain_spans(*args, **kwargs)

    fake_torch = SimpleNamespace(
        Tensor=ShapeTensor,
        is_grad_enabled=lambda: False,
        float16="fp16",
        bfloat16="bf16",
        int32="i32",
        npu=SimpleNamespace(current_device=lambda: 0),
    )
    validate = extracted(
        "gmla/_runtime.py",
        "validate_forward_inputs",
        {
            "torch": fake_torch,
            "validate_scale": validate_scale,
            "DEFAULT_SCALE": DEFAULT_SCALE,
            "validate_chain_spans": spans,
            "default_layout": SimpleNamespace(require_dense_base=lambda *a, **k: None),
        },
    )
    assert validate(*inputs, block_n_selector=public_prefill_block_n) == (1, 128, 96, 128, DEFAULT_SCALE)
    validate(*inputs, block_n=64)
    assert observed == [128, 64]


def test_public_profile_uses_its_bundle_after_validation(monkeypatch):
    calls = []
    contract = (1, 128, 96, 128, DEFAULT_SCALE)

    def validate(*args, **kwargs):
        assert kwargs["block_n_selector"] is public_prefill_block_n
        calls.append("validated")
        return contract

    def selected(target):
        assert calls == ["validated"] and target == "pto"
        calls.append("selected128")
        return SimpleNamespace(_forward_validated=lambda *a, **k: "profile output")

    monkeypatch.setattr(runtime_module, "validate_forward_inputs", validate)
    monkeypatch.setattr(_selection, "selected_prefill128_bundle", selected)
    monkeypatch.setattr(runtime_module.GatedMLARuntime, "kernel_bundle", property(fail_if_called))
    assert runtime_module.build_runtime().forward(*([object()] * 8)) == "profile output"
    assert calls == ["validated", "selected128"]


def test_profile_bundle_cache_never_mutates_base():
    @dataclass(frozen=True)
    class Base:
        block_n: int = 64

    base = Base()
    select = extracted("gmla/_selection.py", "selected_prefill128_bundle", {"selected_bundle": lambda target: base})
    first = select("pto")
    assert first is select("pto") and first is not base
    assert first.block_n == 128 and base.block_n == 64
