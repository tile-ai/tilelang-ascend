"""Keep adopted A2/A3 source tests separate from real A2/A3 device tests."""

import pytest
from tilelang import tvm, env
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
from tilelang.cache.kernel_cache import KernelCache


def pytest_configure(config):
    config.addinivalue_line("markers", "ci_skip: original TA case excluded by its normal CI collector")
    config.addinivalue_line("markers", "a2a3_original_skip_a5: original TA non-A5 requirement, resolved at test time")
    config.addinivalue_line("markers", "low_priority: original TA boundary case reserved for full testing")
    config.addinivalue_line("markers", "a2a3_native: requires the A2/A3 pipeline and native source builders")
    config.addinivalue_line("markers", "a2a3_device: requires an actual supported Ascend A2/A3 NPU")


@pytest.fixture(autouse=True)
def _require_a2a3(request, monkeypatch):
    device = request.node.get_closest_marker("a2a3_device") is not None
    native = request.node.get_closest_marker("a2a3_native") is not None
    if not device and not native:
        return
    for name in ("tl.transform.A2A3HostLegalize", "target.build.tilelang_ascend_a2a3", "target.build.tilelang_pto_a2a3"):
        if tvm.get_global_func(name, allow_missing=True) is None:
            pytest.skip(f"requires USE_ASCEND=ON with native A2/A3 component {name}")
    if not device:
        return
    try:
        import torch
        import torch_npu  # noqa: F401

        from tilelang.ascend.a2a3.platform import _known_platform

        if not torch.npu.is_available():
            pytest.skip("requires an actual Ascend A2/A3 NPU")
        soc = str(torch.npu.get_device_name())
        platform = _known_platform(soc)
        if platform is None:
            pytest.skip(f"requires a confirmed A2/A3 device, found {soc!r}")
    except ImportError:
        pytest.skip("requires torch_npu and an actual Ascend A2/A3 NPU")
    # Keep source target aliases tied to the actual device, scoped to this test.
    monkeypatch.setenv("TL_PLATFORM", platform)


@pytest.fixture
def isolated_a2a3_cache(tmp_path, monkeypatch):
    """Keep per-case disk clearing and memory clearing away from other suites."""
    descriptor = vars(type(env))["TILELANG_CACHE_DIR"]
    with monkeypatch.context() as scoped:
        # Patch the descriptor override itself so its exact prior value (including
        # None/live environment behavior) is restored, rather than forcing an old
        # resolved path into the process after the test.
        scoped.setattr(descriptor, "_forced_value", str(tmp_path / "a2a3-cache"))
        scoped.setattr(KernelCache, "_last_cleaned_staging_root", KernelCache._last_cleaned_staging_root)
        cache = A2A3CythonKernelCache()
        scoped.setattr(cache, "_memory_cache", {})
        yield tmp_path / "a2a3-cache"
