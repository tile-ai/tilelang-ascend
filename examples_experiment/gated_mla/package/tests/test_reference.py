import pytest
import torch

from gmla import DEFAULT_SCALE, prepare_metadata, reference

from .fixtures import make_inputs, references


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packaged_original_references_ragged_prefill(dtype):
    inputs = make_inputs(dict(B=2, H=3, Tq=3, Tk=5), dtype, 0, "head_specific")
    meta = prepare_metadata([[3, 5, 2], [0, 0, 0]], batch=2, query_capacity=3, cache_capacity=5)
    result = references(inputs, meta, reference, DEFAULT_SCALE)
    assert result["expanded"].shape == (2, 3, 3, 128)
    assert result["expanded"].dtype == torch.float64 and torch.isfinite(result["expanded"]).all()
    assert torch.count_nonzero(result["expanded"][1]) == 0
    torch.testing.assert_close(result["expanded"], result["absorbed"], rtol=2e-12, atol=2e-12)
