import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

# Default (per-PR) case set, chosen by which kernel compile keys the cases
# cover rather than by count, following the moe/quant precedent (#1711):
# CI runs pytest with --forked, so every case re-compiles its kernel from
# scratch, and a manifest change triggers the full CI run on top of the
# 60-minute PR job budget.
# sinkhorn_bwd_implicit_cg(n_stream, tilesize) is a single kernel and
# seqlen is handled by the host pad-run-trim adapter, so the compile key
# is (n_stream, tilesize). The three default cases below each cover one
# distinct key:
#   - (100, 16, 8): the n_stream=16 key plus the non-divisible-seqlen
#     host pad path (example keeps the same shape for its simple case)
#   - (256, 32, 8): the n_stream=32 key
#   - (250, 8, 8): the n_stream=8 key plus a second non-divisible seqlen
# The remaining three n_stream=16 shapes (256/250/512) only vary seqlen,
# keep the same compile key, and are marked low_priority for the
# 360-minute scheduled run.
_DEFAULT_CASES = frozenset(
    {
        (100, 16, 8),
        (256, 32, 8),
        (250, 8, 8),
    }
)


def _mhc_bwd_cases() -> list:
    shapes = [
        (256, 16, 8),
        (100, 16, 8),
        (250, 16, 8),
        (512, 16, 8),
        (256, 32, 8),
        (250, 8, 8),
    ]
    return [
        pytest.param(
            seqlen,
            n_stream,
            tilesize,
            id=f"seqlen{seqlen}_ns{n_stream}_ts{tilesize}",
            marks=() if (seqlen, n_stream, tilesize) in _DEFAULT_CASES else pytest.mark.low_priority,
        )
        for (seqlen, n_stream, tilesize) in shapes
    ]


def _load_mhc_bwd_example() -> ModuleType:
    source = Path(__file__).with_name("example_mhc_bwd.py")
    spec = importlib.util.spec_from_file_location("_mhc_bwd_example_for_test", source)
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


@pytest.mark.parametrize(("seqlen", "n_stream", "tilesize"), _mhc_bwd_cases())
def test_mhc_bwd_accuracy(seqlen: int, n_stream: int, tilesize: int) -> None:
    import torch

    example = _load_mhc_bwd_example()

    M = example.generate_test_data(seqlen, n_stream)
    R, _ = example.sinkhorn_forward(M, 20)
    loss_weight = torch.randn_like(R)

    loss_a = (R * loss_weight).sum()
    loss_a.backward()
    grad_M_autograd = M.grad.detach().clone()

    grad_M_implicit = example.sinkhorn_bwd(R.detach(), loss_weight, n_stream, tilesize)

    grad_M_ref = example.sinkhorn_bwd_ref(R.detach().cpu(), loss_weight.cpu(), n_stream, tilesize)
    torch.npu.synchronize()

    # Same thresholds as the historical example test: keep the verdict
    # strength rather than relaxing to the repo-wide assert_close default.
    abs_diff = (grad_M_autograd.cpu() - grad_M_implicit.cpu()).abs()
    assert abs_diff.max().item() < 1e-3, f"autograd max_abs_diff={abs_diff.max().item():.6e}"

    ref_diff = (grad_M_ref - grad_M_implicit.cpu()).abs()
    assert ref_diff.max().item() < 1e-5, f"manual-CG ref max_diff={ref_diff.max().item():.6e}"
