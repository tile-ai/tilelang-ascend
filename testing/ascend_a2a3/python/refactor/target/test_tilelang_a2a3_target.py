"""Hardware-independent coverage for the A2/A3 backend boundary."""

import pytest

from tilelang import tvm
from tilelang.ascend import target as ascend_target
from tilelang.backend import create_backend_context
from tilelang.backend import target as target_registry
from tilelang.ascend.a2a3.target import target_is_ascend_a2a3, target_is_pto_a2a3
from tilelang.ascend.a2a3.pipeline import _REQUIRED_NATIVE_FACTORIES
from tilelang.backend.target import determine_target
from tvm.target import Target

_ASCEND_KIND_AVAILABLE = "ascend" in Target.list_kinds()
_requires_ascend_kind = pytest.mark.skipif(not _ASCEND_KIND_AVAILABLE, reason="TVM was built without the ascend target kind")


@pytest.mark.parametrize(
    ("platform", "expected_arch"),
    [("A2", "dav-2201"), ("a3", "dav-2201"), ("A5", None)],
)
def test_platform_override(monkeypatch, platform, expected_arch):
    monkeypatch.setenv("TL_PLATFORM", platform)
    monkeypatch.setattr(ascend_target, "_get_soc_name", lambda: None)
    assert ascend_target.detect_a2a3_arch() == expected_arch


def test_invalid_platform_override_is_actionable(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "unknown")
    with pytest.raises(ValueError, match="Invalid TL_PLATFORM.*expected A2, A3, or A5"):
        determine_target("ascend", return_object=True)


@pytest.mark.parametrize(
    ("soc", "expected_arch"),
    [
        ("Ascend910B1", "dav-2201"),
        ("Ascend910C", "dav-2201"),
        ("Ascend910_9392", "dav-2201"),
        ("Ascend950PR_9599", None),
        ("Ascend910_9599", None),
        ("unknown", None),
        ("Ascend910Z", None),
        ("Ascend910A", None),
        (None, None),
    ],
)
def test_soc_detection(monkeypatch, soc, expected_arch):
    monkeypatch.delenv("TL_PLATFORM", raising=False)
    monkeypatch.setattr(ascend_target, "_get_soc_name", lambda: soc)
    assert ascend_target.detect_a2a3_arch() == expected_arch


@pytest.mark.parametrize(
    ("target", "expected_backend"),
    [
        ("ascend", "ascend_a2a3"),
        ("asc", "ascend_a2a3"),
        ("pto", "pto_a2a3"),
        ("ascendc", "ascend_a2a3"),
        ({"kind": "ascend", "arch": "dav-2201"}, "ascend_a2a3"),
        ({"kind": "ascend", "arch": "dav-2201", "keys": ["pto", "ascend"]}, "pto_a2a3"),
    ],
)
@_requires_ascend_kind
def test_a2a3_target_forms(monkeypatch, target, expected_backend):
    monkeypatch.setenv("TL_PLATFORM", "A3")
    context = create_backend_context(target, "c", "auto")
    assert context.name == expected_backend
    assert context.execution_backend.name == "cython"
    assert str(context.target.attrs["arch"]) == "dav-2201"


@_requires_ascend_kind
def test_explicit_alias_selects_a2a3_without_device(monkeypatch):
    monkeypatch.delenv("TL_PLATFORM", raising=False)
    monkeypatch.setattr(ascend_target, "_get_soc_name", lambda: None)
    assert create_backend_context("ascendc", "c", "auto").name == "ascend_a2a3"
    assert create_backend_context("ascend", "c", "auto").name == "ascend"
    assert create_backend_context("pto", "c", "auto").name == "pto"


@_requires_ascend_kind
def test_a5_and_a2a3_predicates_are_disjoint(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "A5")
    assert create_backend_context("ascend", "c", "auto").name == "ascend"
    assert create_backend_context("pto", "c", "auto").name == "pto"
    assert create_backend_context(Target({"kind": "ascend", "arch": "dav-2201"}), "c", "auto").name == "ascend_a2a3"


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
@pytest.mark.parametrize("pto", [False, True])
@_requires_ascend_kind
def test_four_backend_predicates_match_exactly_once(arch, pto):
    target = Target({"kind": "ascend", "arch": arch, "keys": ["pto", "ascend"] if pto else ["ascend"]})
    matches = [
        ascend_target.target_is_plain_ascend(target),
        ascend_target.target_is_pto(target),
        target_is_ascend_a2a3(target),
        target_is_pto_a2a3(target),
    ]
    assert sum(matches) == 1
    expected_index = (2 if arch == "dav-2201" else 0) + pto
    assert matches[expected_index]


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
@pytest.mark.parametrize("pto", [False, True])
@pytest.mark.parametrize("as_object", [False, True])
@_requires_ascend_kind
def test_explicit_arch_has_priority_over_detection_and_environment(monkeypatch, arch, pto, as_object):
    monkeypatch.setenv("TL_PLATFORM", "A5" if arch == "dav-2201" else "A3")

    def unexpected_probe():
        pytest.fail("An explicit target must not probe hardware")

    monkeypatch.setattr(ascend_target, "_get_soc_name", unexpected_probe)
    target = {"kind": "ascend", "arch": arch, "keys": ["pto", "ascend"] if pto else ["ascend"]}
    context = create_backend_context(Target(target) if as_object else target, "c", "auto")
    expected = "pto" if pto else "ascend"
    if arch == "dav-2201":
        expected += "_a2a3"
    assert context.name == expected
    assert str(context.target.attrs["arch"]) == arch


@pytest.mark.parametrize(
    ("soc", "a2a3"),
    [("Ascend910B1", True), ("Ascend910C", True), ("Ascend950", False), ("unknown", False), (None, False)],
)
@pytest.mark.parametrize("target", ["ascend", "pto", "ascendc", "auto"])
def test_detected_target_routes(monkeypatch, soc, a2a3, target):
    monkeypatch.delenv("TL_PLATFORM", raising=False)
    monkeypatch.setattr(ascend_target, "_get_soc_name", lambda: soc)
    monkeypatch.setattr(ascend_target, "check_ascend_availability", lambda: soc is not None)
    # Isolate Ascend detection from CUDA/ROCm devices on the test host.
    monkeypatch.setattr(target_registry, "_TARGET_DETECTORS", {"ascend": target_registry._TARGET_DETECTORS["ascend"]})
    if target == "auto" and soc is None:
        with pytest.raises(ValueError, match="No registered target detector"):
            create_backend_context(target, "c", "auto")
        return
    if not _ASCEND_KIND_AVAILABLE:
        pytest.skip("TVM was built without the ascend target kind")
    expected = "pto" if target == "pto" else "ascend"
    if a2a3 or target == "ascendc":
        expected += "_a2a3"
    assert create_backend_context(target, "c", "auto").name == expected


@_requires_ascend_kind
def test_auto_ascend_detector_selects_a2a3(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "A2")
    detected = ascend_target._detect_ascend_target()
    assert detected is not None
    assert create_backend_context(detected, "c", "auto").name == "ascend_a2a3"


@_requires_ascend_kind
def test_pto_execution_backend_is_not_a2a3_pto(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "A3")
    with pytest.raises(ValueError, match="Allowed: cython"):
        create_backend_context("ascend", "c", "pto")


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@_requires_ascend_kind
def test_a2a3_dialects_share_single_export_contract(monkeypatch, target):
    monkeypatch.setenv("TL_PLATFORM", "A3")
    context = create_backend_context(target, "c", "auto")
    assert context.name == ("ascend_a2a3" if target == "ascendc" else "pto_a2a3")
    assert context.execution_backend.name == "cython"
    assert ("pto" in context.target.keys) == (target == "pto")
    native_ready = all(tvm.ffi.get_global_func(name, allow_missing=True) is not None for name in _REQUIRED_NATIVE_FACTORIES)
    if native_ready:
        with pytest.raises(ValueError, match="exactly one exported source"):
            context.lower(tvm.IRModule({}))
    else:
        with pytest.raises(RuntimeError, match="USE_ASCEND=ON"):
            context.lower(tvm.IRModule({}))


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("compile_device", [False, True])
def test_a2a3_native_source_codegen_uses_selected_dialect(monkeypatch, target, compile_device):
    native = "target.build.tilelang_ascend_a2a3" if target == "ascendc" else "target.build.tilelang_pto_a2a3"
    if tvm.ffi.get_global_func(native, allow_missing=True) is None:
        pytest.skip(f"requires USE_ASCEND=ON with {native}")
    monkeypatch.setenv("TL_PLATFORM", "A3")
    context = create_backend_context(target, "c", "auto")
    func = tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(0)).with_attr("global_symbol", "kernel")
    generated = context.codegen_device(tvm.IRModule.from_expr(func), compile_device=compile_device)
    assert "kernel_kernel" in generated.inspect_source()
