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
from tilelang.carver.arch.ascend import Ascend, is_cube_supported_precision


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


@pytest.mark.parametrize("mcpu", ["ascend910c", "ascend910_93"])
def test_mcpu_resolves_910c(monkeypatch, mcpu):
    monkeypatch.setattr(ascend_arch, "_TORCH_NPU_AVAILABLE", False)
    arch = Ascend(target=f"llvm --keys=ascend -mcpu={mcpu}")
    assert arch.chip_name == "Ascend910C"
    assert arch.chip_profile == "Ascend910B"
    assert is_cube_supported_precision("float16", "float32", arch) is True


@pytest.mark.parametrize("chip_name", ["Ascend910_93", "ascend910c", "Ascend910C3"])
def test_explicit_chip_name_spellings_are_canonicalised(chip_name):
    """Caller-supplied names go through the same resolver as detected ones."""
    arch = Ascend(chip_name=chip_name)
    assert arch.chip_name == "Ascend910C"
    assert arch.chip_profile == "Ascend910B"
    assert arch.ub_cap == 192 * 1024


def test_910c_alias_points_at_910b_profile():
    """Ascend910C has no CHIP_SPECS entry of its own; it reuses 910B's."""
    arch = Ascend(chip_name="Ascend910C")
    assert arch.chip_profile == "Ascend910B"


@pytest.mark.parametrize("device_name", ["Ascend910B3", "Ascend310P3"])
def test_non_aliased_profiles_stay_their_own(fake_device, device_name):
    fake_device(device_name)
    arch = Ascend()
    assert arch.chip_profile == arch.chip_name


@pytest.mark.parametrize(
    "in_dtype, accum_dtype, expected",
    [
        ("float16", "float32", True),
        ("float16", "float16", True),
        ("float16", "bfloat16", True),
        ("bfloat16", "float32", True),
        ("bfloat16", "bfloat16", True),
        ("int8", "int32", False),
        ("float32", "float32", False),
    ],
)
def test_cube_precision_supported_for_910c(in_dtype, accum_dtype, expected):
    """The A3 alias must take the same cube branch as 910B.

    Previously ``is_cube_supported_precision`` enumerated raw chip names, so
    Ascend910C raised ValueError; the caller swallows it and cube hints
    silently degrade to the default policy.
    """
    arch = Ascend(chip_name="Ascend910C")
    assert is_cube_supported_precision(in_dtype, accum_dtype, arch) is expected


@pytest.mark.parametrize("chip_name", ["Ascend910A", "Ascend910B", "Ascend310P"])
def test_cube_precision_unchanged_for_existing_profiles(chip_name):
    arch = Ascend(chip_name=chip_name)
    assert is_cube_supported_precision("float16", "float32", arch) is True
    assert is_cube_supported_precision("int8", "int32", arch) is False


@pytest.mark.parametrize("chip_name", ["Ascend950PR", "Ascend910_95", "Ascend910_9599"])
def test_a5_names_stay_unprofiled_and_raise(chip_name):
    """A5 has no carver profile yet: never silently fall back to 910A.

    The explicit spelling must keep the caller's name (so the capability
    check raises loudly) instead of being canonicalised into the generic
    "910" bucket.
    """
    arch = Ascend(chip_name=chip_name)
    assert arch.chip_name == chip_name
    with pytest.raises(ValueError, match="Unsupported architecture"):
        is_cube_supported_precision("float16", "float32", arch)


def test_explicit_310p_sku_canonicalises_to_profile():
    arch = Ascend(chip_name="Ascend310P3")
    assert arch.chip_name == "Ascend310P"
    assert arch.chip_profile == "Ascend310P"
