import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

# The cases a pull request runs; everything else is low_priority and waits for
# the scheduled run. Both kernels compile per (num_groups, num_experts_per_group,
# num_topk_groups, num_topk_sum), so these five cover both num_topk_sum paths --
# the only one that changes which reduction is generated -- both num_topk_groups,
# all three num_tokens including the single-token edge, and five of the six
# experts-per-group values, spanning the extremes of the 32-element alignment:
# 6 padded up to 32, 32 exact, and 64 exact.
_SMOKE_CASES = frozenset(
    {
        (1, 72, 12, 1, 2),  # single token, experts_per_group=6 padded to 32, top-1
        (4001, 256, 4, 2, 4),  # experts_per_group=64, top-2, four groups kept
        (4001, 72, 8, 2, 2),  # experts_per_group=9, top-2
        (8001, 256, 8, 1, 4),  # largest token count, experts_per_group=32 exact, top-1
        (1, 256, 16, 2, 2),  # single token, most groups, experts_per_group=16
    }
)


def _test_params() -> list:
    params = []
    for num_tokens in (1, 4001, 8001):
        for num_experts in (72, 256):
            for num_groups in (4, 8, 12, 16):
                if num_experts % num_groups:
                    continue
                for num_group_sum_topk in (1, 2):
                    for num_topk_groups in (2, 4):
                        case = {
                            "num_tokens": num_tokens,
                            "num_experts": num_experts,
                            "num_groups": num_groups,
                            "num_group_sum_topk": num_group_sum_topk,
                            "num_topk_groups": num_topk_groups,
                        }
                        params.append(
                            pytest.param(
                                case,
                                id=(
                                    f"tok={num_tokens}_exp={num_experts}_grp={num_groups}_sumk={num_group_sum_topk}_topk={num_topk_groups}"
                                ),
                                marks=(
                                    ()
                                    if (num_tokens, num_experts, num_groups, num_group_sum_topk, num_topk_groups) in _SMOKE_CASES
                                    else pytest.mark.low_priority
                                ),
                            )
                        )
    return params


@pytest.fixture
def isolated_kernel_cache(tmp_path):
    """Give the case its own kernel cache directory.

    The Example clears the kernel cache when it is imported, and here that
    import happens inside every fork. Against the shared directory that would
    delete kernels the other operator tests in the same Pytest invocation are
    compiling into; against this one it deletes nothing anyone else holds.
    """
    import tilelang.cache

    previous = tilelang.cache.get_cache_dir()
    tilelang.cache.set_cache_dir(str(tmp_path))
    try:
        yield
    finally:
        tilelang.cache.set_cache_dir(str(previous))


def _load_moe_topk_sum_example() -> ModuleType:
    source = Path(__file__).with_name("moe_topk_sum_and_topk_group_idx.py")
    spec = importlib.util.spec_from_file_location("_moe_topk_sum_example_for_test", source)
    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        # moe_topk_sum_and_topk_group_idx.py reads arguments at import time. Hide
        # Pytest's own arguments while loading it, without changing the Example.
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv
    return module


@pytest.mark.parametrize("params", _test_params())
def test_topk_sum_and_topk_group_idx(params: dict, isolated_kernel_cache) -> None:
    import numpy as np
    import torch

    example = _load_moe_topk_sum_example()

    num_groups = params["num_groups"]
    num_group_sum_topk = params["num_group_sum_topk"]
    num_topk_groups = params["num_topk_groups"]
    num_experts_per_group = params["num_experts"] // num_groups

    torch.manual_seed(42)
    scores = torch.randn((params["num_tokens"], num_groups, num_experts_per_group), dtype=torch.float32)
    topk_idx_ref = example.ref_topk_sum_and_topk_group_idx(scores, num_group_sum_topk, num_topk_groups)

    if hasattr(torch, "npu") and torch.npu.is_available():
        scores = scores.to("npu").contiguous()
    elif torch.cuda.is_available():
        scores = scores.to("cuda").contiguous()

    topk_idx = example.topk_sum_and_topk_group_idx(scores, num_group_sum_topk, num_topk_groups)
    np.testing.assert_equal(topk_idx.cpu().numpy(), topk_idx_ref.cpu().numpy())


@pytest.mark.parametrize("params", _test_params())
def test_topk_sum_and_topk_group_idx_backward(params: dict, isolated_kernel_cache) -> None:
    import numpy as np
    import torch

    example = _load_moe_topk_sum_example()

    num_groups = params["num_groups"]
    num_group_sum_topk = params["num_group_sum_topk"]
    num_topk_groups = params["num_topk_groups"]
    num_experts_per_group = params["num_experts"] // num_groups

    torch.manual_seed(42)
    scores = torch.randn((params["num_tokens"], num_groups, num_experts_per_group), dtype=torch.float32)
    grad_out = torch.randn((params["num_tokens"], num_topk_groups), dtype=torch.float32)

    topk_idx = example.ref_topk_sum_and_topk_group_idx(scores, num_group_sum_topk, num_topk_groups)
    grad_scores_ref = example.ref_topk_sum_and_topk_group_idx_backward(grad_out, scores, topk_idx, num_group_sum_topk)

    if hasattr(torch, "npu") and torch.npu.is_available():
        scores = scores.to("npu").contiguous()
        grad_out = grad_out.to("npu").contiguous()
        topk_idx = topk_idx.to("npu").contiguous()
    elif torch.cuda.is_available():
        scores = scores.to("cuda").contiguous()
        grad_out = grad_out.to("cuda").contiguous()
        topk_idx = topk_idx.to("cuda").contiguous()

    grad_scores_tl = example.topk_sum_and_topk_group_idx_backward(grad_out, scores, topk_idx, num_group_sum_topk)

    np.testing.assert_allclose(
        grad_scores_tl.cpu().numpy(),
        grad_scores_ref.cpu().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
