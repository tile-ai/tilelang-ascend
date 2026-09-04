import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest import mock

import pytest


def _load_ascend_arch_module():
    """Load the leaf modules without importing carver's torch_npu-only modules."""
    arch_dir = Path(__file__).parents[2] / "tilelang" / "carver" / "arch"
    package_name = "_tilelang_a5_arch_test"
    package = ModuleType(package_name)
    package.__path__ = [str(arch_dir)]

    loaded_names = [package_name, f"{package_name}.arch_base", f"{package_name}.ascend"]
    try:
        sys.modules[package_name] = package
        base_spec = importlib.util.spec_from_file_location(f"{package_name}.arch_base", arch_dir / "arch_base.py")
        base_module = importlib.util.module_from_spec(base_spec)
        sys.modules[base_spec.name] = base_module
        base_spec.loader.exec_module(base_module)

        ascend_spec = importlib.util.spec_from_file_location(f"{package_name}.ascend", arch_dir / "ascend.py")
        ascend_module = importlib.util.module_from_spec(ascend_spec)
        sys.modules[ascend_spec.name] = ascend_module
        ascend_spec.loader.exec_module(ascend_module)
        return ascend_module
    finally:
        for module_name in loaded_names:
            sys.modules.pop(module_name, None)


ascend_arch = _load_ascend_arch_module()


class _FakeNpu:
    def __init__(self, name, cube_core_num, l2_cache_size):
        self._props = SimpleNamespace(
            name=name,
            cube_core_num=cube_core_num,
            L2_cache_size=l2_cache_size,
        )

    def is_available(self):
        return True

    def device_count(self):
        return 1

    def current_device(self):
        return 0

    def get_device_properties(self, _device):
        return self._props


@pytest.mark.parametrize(
    "name, cube_cores",
    [
        ("Ascend950PR_950z", 4),
        ("Ascend950DT_9575", 28),
        ("Ascend910_9599", 36),
    ],
)
def test_a5_resources_use_runtime_sku_properties(name, cube_cores):
    fake_torch = SimpleNamespace(npu=_FakeNpu(name, cube_cores, 96 * 1024 * 1024))
    with (
        mock.patch.object(ascend_arch, "_TORCH_NPU_AVAILABLE", True),
        mock.patch.object(ascend_arch, "torch", fake_torch),
    ):
        arch = ascend_arch.Ascend(SimpleNamespace(mcpu="dav-3510"))

    assert arch.chip_name == "Ascend950"
    assert arch.compute_max_core == cube_cores
    assert arch.ub_cap == 248 * 1024
    assert arch.l1_cap == 512 * 1024
    assert arch.l0c_cap == 256 * 1024
    assert arch.l2_cache_size_bytes == 96 * 1024 * 1024
    assert ascend_arch.is_cube_supported_precision("float16", "float32", arch)
    assert ascend_arch.is_cube_supported_precision("bfloat16", "bfloat16", arch)
    assert not ascend_arch.is_cube_supported_precision("float32", "float32", arch)
    assert not ascend_arch.is_cube_supported_precision("float16", "int32", arch)


def test_a5_without_runtime_core_query_fails_loud():
    with (
        mock.patch.object(ascend_arch, "_TORCH_NPU_AVAILABLE", False),
        pytest.raises(RuntimeError, match="core count is device/SKU-specific"),
    ):
        ascend_arch.Ascend(SimpleNamespace(mcpu="dav-3510"), chip_name="Ascend950")


@pytest.mark.parametrize(
    "mcpu, runtime_name",
    [
        ("dav-3510", "Ascend910B"),
        ("dav-910b", "Ascend950PR_950z"),
        ("dav-910c", "Ascend910B"),
        ("dav-910b", "Ascend910C"),
    ],
)
def test_target_mcpu_profile_rejects_runtime_device_mismatch(mcpu, runtime_name):
    fake_torch = SimpleNamespace(npu=_FakeNpu(runtime_name, 4, 96 * 1024 * 1024))
    with (
        mock.patch.object(ascend_arch, "_TORCH_NPU_AVAILABLE", True),
        mock.patch.object(ascend_arch, "torch", fake_torch),
        pytest.raises(RuntimeError, match="does not match runtime device profile"),
    ):
        ascend_arch.Ascend(SimpleNamespace(mcpu=mcpu))
