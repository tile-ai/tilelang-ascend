"""Original platform scenarios adapted to the approved strict A2/A3 contract.

TA 0deb8b93 falls back to A3 for an unknown or unavailable device. The migrated
backend deliberately requires a confirmed A2/A3 platform; A5 keeps its own
backend. The source scenario/expectation table is in testing_ports.json.
"""

import sys
from unittest.mock import MagicMock

import pytest
from tvm.target import Target

from tilelang.ascend.a2a3.platform import resolve_a2a3_platform
from tilelang.backend import create_backend_context

pytestmark = pytest.mark.skipif("ascend" not in Target.list_kinds(), reason="requires the registered Ascend target kind")


def _a2a3_target(model=None):
    values = {"kind": "ascend", "arch": "dav-2201", "keys": ["ascendc", "ascend"]}
    if model is not None:
        values["model"] = model
    return Target(values)


def _mock_acl(monkeypatch, soc_name=None):
    mock_acl = MagicMock()
    if soc_name is not None:
        mock_acl.get_soc_name.return_value = soc_name
    monkeypatch.setitem(sys.modules, "acl", mock_acl)
    monkeypatch.delenv("TL_PLATFORM", raising=False)


def _mock_torch_fallback(monkeypatch, device_name=None, available=True):
    monkeypatch.setitem(sys.modules, "acl", None)
    mock_torch = MagicMock()
    mock_torch.npu.is_available.return_value = available
    if device_name is not None:
        mock_torch.npu.get_device_name.return_value = device_name
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.delenv("TL_PLATFORM", raising=False)


def test_determine_platform_explicit():
    assert resolve_a2a3_platform(_a2a3_target("A2")) == "A2"
    assert resolve_a2a3_platform(_a2a3_target("A3")) == "A3"
    a5 = Target({"kind": "ascend", "arch": "dav-3510", "model": "A5"})
    assert create_backend_context(a5, execution_backend="cython").module.name == "ascend"
    with pytest.raises(ValueError, match="requires an ascend target with arch=dav-2201"):
        resolve_a2a3_platform(a5)


def test_determine_platform_env(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "A2")
    assert resolve_a2a3_platform(_a2a3_target()) == "A2"
    monkeypatch.setenv("TL_PLATFORM", "A5")
    assert create_backend_context("ascend", execution_backend="cython").module.name == "ascend"
    with pytest.raises(ValueError, match="does not select A2/A3"):
        resolve_a2a3_platform(_a2a3_target())


def test_determine_platform_auto_910B(monkeypatch):
    _mock_acl(monkeypatch, soc_name="Ascend910B")
    assert resolve_a2a3_platform(_a2a3_target()) == "A2"


def test_determine_platform_auto_910C(monkeypatch):
    _mock_acl(monkeypatch, soc_name="Ascend910_93")
    assert resolve_a2a3_platform(_a2a3_target()) == "A3"


def test_determine_platform_auto_950D(monkeypatch):
    _mock_acl(monkeypatch, soc_name="Ascend950")
    assert create_backend_context("ascend", execution_backend="cython").module.name == "ascend"
    with pytest.raises(ValueError, match="Cannot confirm the A2/A3 platform"):
        resolve_a2a3_platform(_a2a3_target())


def test_determine_platform_auto_unknown_device(monkeypatch):
    _mock_acl(monkeypatch, soc_name="AscendUnknown")
    with pytest.raises(ValueError, match="Cannot confirm the A2/A3 platform"):
        resolve_a2a3_platform(_a2a3_target())


def test_determine_platform_auto_910B_torch_fallback(monkeypatch):
    _mock_torch_fallback(monkeypatch, device_name="Ascend910B")
    assert resolve_a2a3_platform(_a2a3_target()) == "A2"


def test_determine_platform_auto_950D_torch_fallback(monkeypatch):
    _mock_torch_fallback(monkeypatch, device_name="Ascend950")
    assert create_backend_context("ascend", execution_backend="cython").module.name == "ascend"
    with pytest.raises(ValueError, match="Cannot confirm the A2/A3 platform"):
        resolve_a2a3_platform(_a2a3_target())


def test_determine_platform_auto_fallback(monkeypatch):
    _mock_torch_fallback(monkeypatch, available=False)
    with pytest.raises(ValueError, match="Cannot confirm the A2/A3 platform"):
        resolve_a2a3_platform(_a2a3_target())
