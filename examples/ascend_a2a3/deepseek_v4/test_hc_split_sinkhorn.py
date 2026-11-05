import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_hc_split_sinkhorn_example() -> ModuleType:
    source = Path(__file__).with_name("hc_split_sinkhorn.py")
    spec = importlib.util.spec_from_file_location("_hc_split_sinkhorn_example_for_test", source)
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


def test_hc_split_sinkhorn_accuracy() -> None:
    import torch

    example = _load_hc_split_sinkhorn_example()

    dtype = torch.float32
    batch = 1
    seq_len = 5
    hc_mult = 4
    sinkhorn_iters = 20
    eps = 1e-6
    mix_hc = (2 + hc_mult) * hc_mult
    n = batch * seq_len

    torch.manual_seed(42)
    mixes = torch.rand((n, mix_hc), dtype=dtype, device="npu")
    hc_scale = torch.rand(3, dtype=dtype, device="npu")
    hc_base = torch.rand(mix_hc, dtype=dtype, device="npu")

    pre = torch.empty((n, hc_mult), dtype=dtype, device="npu")
    post = torch.empty((n, hc_mult), dtype=dtype, device="npu")
    comb = torch.empty((n, hc_mult, hc_mult), dtype=dtype, device="npu")
    torch.npu.synchronize()

    kernel = example.hc_split_sinkhorn(hc=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=eps)

    pre, post, comb = kernel(mixes, hc_scale, hc_base)
    pre_ref, post_ref, comb_ref = example.hc_split_sinkhorn_ref(
        mixes,
        hc_scale,
        hc_base,
        hc_mult,
        sinkhorn_iters,
        eps,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(pre_ref, pre, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(post_ref, post, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(comb_ref, comb, rtol=1e-2, atol=1e-2)
