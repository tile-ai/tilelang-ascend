"""Real CPU mask-helper checks; no original NPU kernel/reference is executed."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("dense_last", [False, True])
@pytest.mark.parametrize("kind", ["topk", "threshold"])
def test_seer_masks_real_CPU_indices_triangular_and_input_not_mutated(kind, dense_last):
    import torch

    path = Path(__file__).resolve().parents[4] / "examples_experiment/seer_attention/block_sparse_attn.py"
    spec = importlib.util.spec_from_file_location("_a2a3_seer_mask_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = torch.tensor([[[[4, 1, 3, 2], [1, 4, 3, 2], [4, 3, 2, 1], [1, 2, 3, 4]]]], dtype=torch.float32, device="cpu")
    original = x.clone()
    rng = torch.random.get_rng_state().clone()
    if kind == "topk":
        actual = module.get_sparse_attn_mask_from_topk(x, 2, dense_last)
        expected = [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]
    else:
        actual = module.get_sparse_attn_mask_from_threshold(x, 3.5, dense_last)
        expected = [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    if dense_last:
        expected[-2:] = [[1, 1, 1, 0], [1, 1, 1, 1]]
    expected = torch.tensor([[expected]], dtype=torch.bool, device="cpu")
    assert actual.device.type == "cpu" and actual.dtype == torch.bool
    assert actual.shape == (1, 1, 4, 4) and torch.equal(actual, expected)
    assert torch.equal(x, original)
    assert torch.equal(torch.random.get_rng_state(), rng)
