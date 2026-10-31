"""Hardware-independent concrete-platform and serialized-target contracts."""

import pytest

from tvm.target import Target

from tilelang.ascend import target as ascend_target
from tilelang.ascend.a2a3.platform import normalize_a2a3_platform_target, resolve_a2a3_platform
from tilelang.backend import create_backend_context


@pytest.fixture(autouse=True)
def isolated_platform_detection(monkeypatch):
    monkeypatch.delenv("TL_PLATFORM", raising=False)
    monkeypatch.setattr(ascend_target, "_get_soc_name", lambda: None)


def _target(model=None, **kwargs):
    config = {"kind": "ascend", "arch": "dav-2201", **kwargs}
    if model is not None:
        config["model"] = model
    return Target(config)


def _unexpected_probe():
    pytest.fail("An explicit platform must not probe a device")


@pytest.mark.parametrize(
    ("model", "expected"),
    [("A2", "A2"), ("a3", "A3"), ("Ascend910B1", "A2"), ("ascend910c", "A3"), ("Ascend910_9392", "A3")],
)
def test_explicit_model_has_priority_and_skips_probe(monkeypatch, model, expected):
    monkeypatch.setenv("TL_PLATFORM", "A3" if expected == "A2" else "A2")
    monkeypatch.setattr(ascend_target, "_get_soc_name", _unexpected_probe)
    assert resolve_a2a3_platform(_target(model)) == expected


@pytest.mark.parametrize(("override", "expected"), [("A2", "A2"), (" a3 ", "A3")])
def test_environment_has_priority_over_probe(monkeypatch, override, expected):
    monkeypatch.setenv("TL_PLATFORM", override)
    monkeypatch.setattr(ascend_target, "_get_soc_name", _unexpected_probe)
    assert resolve_a2a3_platform(_target("unknown")) == expected


@pytest.mark.parametrize(
    ("soc", "expected"),
    [("Ascend910B2", "A2"), ("Ascend910C", "A3"), ("Ascend910_9392", "A3")],
)
def test_real_device_names(monkeypatch, soc, expected):
    monkeypatch.setattr(ascend_target, "_get_soc_name", lambda: soc)
    assert resolve_a2a3_platform(_target()) == expected


@pytest.mark.parametrize("soc", ["Ascend950", "Ascend910_9599", "Ascend910Z", "Ascend910A", "unknown", None])
def test_unknown_and_a5_devices_never_guess_a3(monkeypatch, soc):
    monkeypatch.setattr(ascend_target, "_get_soc_name", lambda: soc)
    target = _target()
    assert resolve_a2a3_platform(target, required=False) is None
    with pytest.raises(ValueError, match="target.model=A2/A3 or TL_PLATFORM=A2/A3"):
        resolve_a2a3_platform(target)
    assert normalize_a2a3_platform_target(target) is None


@pytest.mark.parametrize("override", ["A5", "invalid", "Ascend910C"])
def test_invalid_or_a5_override_does_not_fall_back_to_hardware(monkeypatch, override):
    monkeypatch.setenv("TL_PLATFORM", override)
    monkeypatch.setattr(ascend_target, "_get_soc_name", _unexpected_probe)
    target = _target()
    assert resolve_a2a3_platform(target, required=False) is None
    assert normalize_a2a3_platform_target(target) is None
    with pytest.raises(ValueError, match="TL_PLATFORM=.*target.model=A2/A3"):
        resolve_a2a3_platform(target)


def test_explicit_model_ignores_an_invalid_environment(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "invalid")
    monkeypatch.setattr(ascend_target, "_get_soc_name", _unexpected_probe)
    assert resolve_a2a3_platform(_target("A2")) == "A2"


@pytest.mark.parametrize("target", [Target("llvm"), Target("ascend"), Target({"kind": "ascend", "arch": "dav-3510"})])
def test_non_a2a3_targets_are_not_resolved(monkeypatch, target):
    monkeypatch.setenv("TL_PLATFORM", "A3")
    monkeypatch.setattr(ascend_target, "_get_soc_name", _unexpected_probe)
    assert resolve_a2a3_platform(target, required=False) is None
    assert normalize_a2a3_platform_target(target) is None
    with pytest.raises(ValueError, match="ascend target with arch=dav-2201"):
        resolve_a2a3_platform(target)


def test_normalization_preserves_target_and_host(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "A3")
    target = Target({"kind": "ascend", "arch": "dav-2201", "keys": ["pto", "ascend"], "mcpu": "dav-2201"}, host="llvm")
    normalized = normalize_a2a3_platform_target(target, "cython")
    assert normalized.attrs["model"] == "A3"
    assert str(normalized.attrs["arch"]) == "dav-2201"
    assert list(normalized.keys) == list(target.keys)
    assert str(normalized.attrs["mcpu"]) == "dav-2201"
    assert normalized.host.same_as(target.host)
    assert str(target.attrs.get("model", "")) != "A3"


def test_canonical_model_returns_original_target(monkeypatch):
    monkeypatch.setattr(ascend_target, "_get_soc_name", _unexpected_probe)
    target = _target("A2")
    assert normalize_a2a3_platform_target(target) is target


def test_platform_changes_serialized_target_identity(monkeypatch):
    target = _target()
    monkeypatch.setenv("TL_PLATFORM", "A2")
    a2 = normalize_a2a3_platform_target(target)
    monkeypatch.setenv("TL_PLATFORM", "A3")
    a3 = normalize_a2a3_platform_target(target)
    assert a2.export()["model"] == "A2"
    assert a3.export()["model"] == "A3"
    assert str(a2) != str(a3)


def test_device_detection_is_not_cached(monkeypatch):
    devices = iter(["Ascend910B1", "Ascend910C"])
    monkeypatch.setattr(ascend_target, "_get_soc_name", lambda: next(devices))
    target = _target()
    assert resolve_a2a3_platform(target) == "A2"
    assert resolve_a2a3_platform(target) == "A3"


def test_failed_hardware_probe_is_an_unresolved_platform(monkeypatch):
    def unavailable():
        raise RuntimeError("device runtime unavailable")

    monkeypatch.setattr(ascend_target, "_get_soc_name", unavailable)
    target = _target()
    assert resolve_a2a3_platform(target, required=False) is None
    assert normalize_a2a3_platform_target(target) is None
    with pytest.raises(ValueError, match="Cannot confirm.*target.model=A2/A3"):
        resolve_a2a3_platform(target)


def test_soft_resolution_does_not_break_explicit_arch_routing(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "invalid")
    target = _target()
    assert normalize_a2a3_platform_target(target) is None
    assert create_backend_context(target, "c", "auto").name == "ascend_a2a3"


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("execution_backend", ["auto", "cython"])
def test_pto_context_freezes_platform_and_preserves_dialect(monkeypatch, platform, execution_backend):
    monkeypatch.setenv("TL_PLATFORM", platform)
    monkeypatch.setattr(ascend_target, "_get_soc_name", _unexpected_probe)
    target = _target(keys=["pto", "ascend"], mcpu="dav-2201")
    context = create_backend_context(target, "c", execution_backend)
    assert context.name == "pto_a2a3"
    assert context.execution_backend.name == "cython"
    assert context.target.attrs["model"] == platform
    assert list(context.target.keys) == list(target.keys)
    assert context.target.attrs["mcpu"] == "dav-2201"
    assert str(target.attrs.get("model", "")) != platform
    monkeypatch.setenv("TL_PLATFORM", "A3" if platform == "A2" else "A2")
    # Serialized contexts and later codegen must consume the frozen target,
    # including when environment changes between cache lookup and compilation.
    restored = Target(dict(context.target.export()), host=context.target.host)
    assert resolve_a2a3_platform(context.target) == platform
    assert resolve_a2a3_platform(restored) == platform
    assert create_backend_context(restored, "c", "cython").target.attrs["model"] == platform


def test_unknown_pto_platform_keeps_routing_but_strict_consumers_reject():
    target = _target(keys=["pto", "ascend"])
    context = create_backend_context(target, "c", "auto")
    assert context.name == "pto_a2a3"
    assert context.execution_backend.name == "cython"
    with pytest.raises(ValueError, match="Cannot confirm.*target.model=A2/A3"):
        resolve_a2a3_platform(context.target)
