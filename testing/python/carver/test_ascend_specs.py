from pathlib import Path
import platform
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tilelang.carver.arch import ascend
from tilelang.carver.arch.ascend import Ascend, is_cube_supported_precision
from tilelang.carver.roller.policy.ascend import AscendDefaultPolicy

SDK_9392 = """[SoCInfo]
cube_core_cnt=24
vector_core_cnt=48
l2_size=201326592
support_bf16=1
[AICoreSpec]
l0_a_size=65536
l0_b_size=65536
l0_c_size=131072
l1_size=524288
ub_size=196608
cube_m_size=16
cube_n_size=16
cube_k_size=16
"""


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    directory = tmp_path / "data/platform_config"
    directory.mkdir(parents=True)
    path = directory / "Ascend910_9392.ini"
    path.write_text(SDK_9392)
    monkeypatch.setattr(ascend, "ASCEND_HOME", str(tmp_path))
    monkeypatch.setattr(ascend, "_TORCH_NPU_AVAILABLE", False)
    return path


@pytest.mark.parametrize("source", ["device", "target", "chip"])
def test_a3_detection(sdk, monkeypatch, source):
    npu = MagicMock()
    npu.is_available.return_value = True
    npu.device_count.return_value = 1
    npu.get_device_properties.return_value = SimpleNamespace(
        name="Ascend910_9392", cube_core_num=24, vector_core_num=48, L2_cache_size=201326592
    )
    monkeypatch.setattr(ascend, "npu", npu, raising=False)
    monkeypatch.setattr(ascend, "_TORCH_NPU_AVAILABLE", source == "device")
    target = "llvm -keys=ascend -mcpu=ascend910_9392" if source == "target" else "llvm -keys=ascend"
    arch = Ascend(
        target,
        chip_name="ascend910_9392" if source == "chip" else None,
        platform_config=sdk if source == "chip" else None,
    )
    assert (
        arch.chip_name,
        arch.compute_max_core,
        arch.compute_max_vector_core,
        arch.l0c_cap,
        arch.l1_cap,
        arch.ub_cap,
        arch.l2_cache_size_bytes,
        arch.raw_memory_capacities["UB"],
        is_cube_supported_precision("bfloat16", "float32", arch),
    ) == ("Ascend910_9392", 24, 48, 131072, 524032, 196352, 201326592, 196608, True)


@pytest.mark.parametrize("m, allowed", [(256, False), (128, True)])
def test_a3_l0_admission(sdk, m, allowed):
    policy = AscendDefaultPolicy(Ascend(chip_name="Ascend910_9392"))
    assert policy._check_l0_usage(m * 64 * 2, 64 * 256 * 2, m * 256 * 4, 2) is allowed


def test_9382_sdk(sdk):
    directory = sdk.parents[2] / f"{platform.machine()}-linux" / "data/platform_config"
    directory.mkdir(parents=True)
    path = directory / "Ascend910_9382.ini"
    path.write_text(SDK_9392.replace("cube_core_cnt=24", "cube_core_cnt=20").replace("support_bf16=1", "support_bf16=0"))
    arch = Ascend(chip_name="Ascend910_9382")
    assert (
        Path(arch.sdk_resources["path"]),
        arch.compute_max_core,
        is_cube_supported_precision("float16", "float32", arch),
        is_cube_supported_precision("bfloat16", "float32", arch),
    ) == (path, 20, True, False)
