import sys
from unittest.mock import MagicMock

from tilelang.utils.target import determine_platform


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
    assert determine_platform("A2") == "A2"
    assert determine_platform("A3") == "A3"
    assert determine_platform("A5") == "A5"


def test_determine_platform_env(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "A2")
    assert determine_platform("auto") == "A2"
    monkeypatch.setenv("TL_PLATFORM", "A5")
    assert determine_platform("auto") == "A5"


def test_determine_platform_auto_910B(monkeypatch):
    _mock_acl(monkeypatch, soc_name="Ascend910B")
    assert determine_platform("auto") == "A2"


def test_determine_platform_auto_910C(monkeypatch):
    _mock_acl(monkeypatch, soc_name="Ascend910_93")
    assert determine_platform("auto") == "A3"


def test_determine_platform_auto_950D(monkeypatch):
    _mock_acl(monkeypatch, soc_name="Ascend950")
    assert determine_platform("auto") == "A5"


def test_determine_platform_auto_unknown_device(monkeypatch):
    _mock_acl(monkeypatch, soc_name="AscendUnknown")
    assert determine_platform("auto") == "A3"


def test_determine_platform_auto_910B_torch_fallback(monkeypatch):
    _mock_torch_fallback(monkeypatch, device_name="Ascend910B")
    assert determine_platform("auto") == "A2"


def test_determine_platform_auto_950D_torch_fallback(monkeypatch):
    _mock_torch_fallback(monkeypatch, device_name="Ascend950")
    assert determine_platform("auto") == "A5"


def test_determine_platform_auto_fallback(monkeypatch):
    _mock_torch_fallback(monkeypatch, available=False)
    assert determine_platform("auto") == "A3"
