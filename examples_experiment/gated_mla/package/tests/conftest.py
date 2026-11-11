import pytest
import torch


@pytest.fixture(scope="session")
def npu():
    try:
        import torch_npu
    except ImportError as error:
        pytest.fail(f"Real NPU tests selected but torch_npu is unavailable: {error}")
    # Establish the pinned native TVM/TileLang libraries before this fixture
    # performs any NPU operation. This calls no kernel factory and launches none.
    from gmla import build_runtime

    _ = build_runtime().kernel_bundle
    assert torch.npu.is_available(), "Real NPU required; no silent skip"
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    return torch_npu


def pytest_configure(config):
    from pathlib import Path
    from _gmla_ci.scoped_ci import configure

    configure(config, Path(__file__).resolve().parents[1])
