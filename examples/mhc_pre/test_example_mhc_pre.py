import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

# Large shapes (n >= 1024 at h = 2560, or h = 7168) are marked low_priority:
# they only run in full/scheduled CI jobs, keeping the per-PR pytest suite
# fast. The default suite still covers all hc values (1-8), non-divisible h
# (100), and one mid-size representative (512, 2560, 4).
_LOW_PRIORITY_CASES = frozenset(
    {
        (1024, 2560, 4),
        (2048, 2560, 4),
        (4096, 2560, 4),
        (1024, 7168, 4),
    }
)


def _mhc_pre_cases() -> list:
    shapes = [
        (4, 128, 4),
        (16, 256, 4),
        (4, 1280, 4),
        (512, 2560, 4),
        (1024, 2560, 4),
        (2048, 2560, 4),
        (4096, 2560, 4),
        (1024, 7168, 4),
        (4, 100, 4),
        (4, 128, 1),
        (4, 128, 2),
        (4, 128, 3),
        (4, 128, 5),
        (4, 128, 6),
        (4, 128, 7),
        (4, 128, 8),
        (4, 100, 8),
    ]
    return [
        pytest.param(
            n,
            h,
            hc_mult,
            id=f"n{n}_h{h}_hc{hc_mult}",
            marks=pytest.mark.low_priority if (n, h, hc_mult) in _LOW_PRIORITY_CASES else (),
        )
        for (n, h, hc_mult) in shapes
    ]


def _load_mhc_pre_example() -> ModuleType:
    source = Path(__file__).with_name("example_mhc_pre.py")
    spec = importlib.util.spec_from_file_location("_mhc_pre_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv
    return module


@pytest.mark.parametrize(("n", "h", "hc_mult"), _mhc_pre_cases())
def test_mhc_pre_accuracy(n: int, h: int, hc_mult: int) -> None:
    import torch

    example = _load_mhc_pre_example()

    data = example.generate_full_test_data(n, h, hc_mult)
    post_tl, comb_tl, layer_tl = example.mhc_pre(**data)
    post_ref, comb_ref, layer_ref = example.mhc_pre_ref(**data)
    torch.npu.synchronize()

    torch.testing.assert_close(post_tl.cpu(), post_ref.cpu(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(comb_tl.cpu(), comb_ref.cpu(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(layer_tl.cpu(), layer_ref.cpu(), rtol=1e-2, atol=1e-2)


def test_mhc_pre_distinct_params() -> None:
    import torch

    example = _load_mhc_pre_example()

    # pre_eps != sinkhorn_eps and post_mult != 1.0/2.0 exercise the
    # distinct-parameter routing through the three kernels.
    data = example.generate_full_test_data(4, 128, 4, hc_pre_eps=1e-4, hc_sinkhorn_eps=3e-3, hc_post_mult_value=1.7, sinkhorn_repeat=3)
    post_tl, comb_tl, layer_tl = example.mhc_pre(**data)
    post_ref, comb_ref, layer_ref = example.mhc_pre_ref(**data)
    torch.npu.synchronize()

    torch.testing.assert_close(post_tl.cpu(), post_ref.cpu(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(comb_tl.cpu(), comb_ref.cpu(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(layer_tl.cpu(), layer_ref.cpu(), rtol=1e-2, atol=1e-2)
