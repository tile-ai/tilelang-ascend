"""Chip-profile resolution for the Ascend carver arch.

The carver picks its tile candidates from ``Ascend.ub_cap`` / ``l0c_cap`` /
``compute_max_core``, so resolving an Atlas A3 (Ascend910_93 / Ascend910C)
device to the 910A profile silently changes the search space: 910A reports
256 KB UB and 256 KB L0C, while DAV_2201 parts (910B family) have 192 KB UB
and 512 KB L0C.
"""

import types

import pytest

from tilelang.carver.arch import ascend as ascend_arch
from tilelang.carver.arch.ascend import Ascend


class _FakeProps:
    def __init__(self, name, cores=20, l2=16 * 1024 * 1024):
        self.name = name
        self.cube_core_num = cores
        self.l2_cache_size = l2


class _FakeNpu:
    def __init__(self, props):
        self._props = props

    def is_available(self):
        return True

    def device_count(self):
        return 1

    def current_device(self):
        return 0

    def get_device_properties(self, device=None):
        return self._props


@pytest.fixture
def fake_device(monkeypatch):
    def install(name, cores=20):
        monkeypatch.setattr(ascend_arch, "_TORCH_NPU_AVAILABLE", True)
        monkeypatch.setattr(
            ascend_arch,
            "torch",
            types.SimpleNamespace(npu=_FakeNpu(_FakeProps(name, cores))),
            raising=False,
        )

    return install


@pytest.mark.parametrize("device_name", ["Ascend910_9362", "ASCEND910C", "Ascend910C3"])
def test_a3_device_resolves_to_910c_profile(fake_device, device_name):
    fake_device(device_name)
    arch = Ascend()
    assert arch.chip_name == "Ascend910C"
    assert arch.ub_cap == 192 * 1024
    assert arch.l0c_cap == 512 * 1024


def test_a2_device_keeps_910b_profile(fake_device):
    fake_device("Ascend910B3")
    arch = Ascend()
    assert arch.chip_name == "Ascend910B"
    assert arch.ub_cap == 192 * 1024


def test_310p_device_keeps_310p_profile(fake_device):
    fake_device("Ascend310P3")
    arch = Ascend()
    assert arch.chip_name == "Ascend310P"


def test_explicit_910c_chip_name_uses_910b_memory_profile():
    arch = Ascend(chip_name="Ascend910C")
    assert arch.chip_name == "Ascend910C"
    assert arch.ub_cap == 192 * 1024
    assert arch.l0c_cap == 512 * 1024


def test_mcpu_resolves_910c(monkeypatch):
    monkeypatch.setattr(ascend_arch, "_TORCH_NPU_AVAILABLE", False)
    arch = Ascend(target="llvm --keys=ascend -mcpu=ascend910c")
    assert arch.chip_name == "Ascend910C"
