"""Mandatory R3 range regression; same existing B1/H96/T17 Value factories."""

import pytest
import torch

from gmla import build_runtime, reference


def _range_fixture(dtype, packed):
    high = 65530.0 if dtype == torch.float16 else torch.finfo(torch.float32).max
    tiny = 2.0**-24 if dtype == torch.float16 else 2.0**-133
    amplification = 65504.0 if dtype == torch.float16 else torch.finfo(torch.bfloat16).max
    z = torch.empty((1, 17, 96, 512), dtype=torch.float32)
    z[:, :, :32] = high
    z[:, :, 32:64] = tiny
    z[:, :, 64:] = 3 * tiny
    z[:, :, 1::2].neg_()
    wv = torch.empty((96, 128, 512), dtype=dtype)
    wv[:32] = 1 / 512
    wv[32:] = amplification
    if packed:
        weight = torch.empty((96, 256, 512), dtype=dtype)
        weight[:, :128] = 0.25  # Finite poison: this K half must never be consumed.
        weight[:, 128:] = wv
    else:
        weight = wv
    gate = torch.zeros((1, 17, 96, 128), dtype=dtype)
    expected = torch.einsum("bthr,hvr->bthv", z.double(), wv.double()) * torch.sigmoid(gate.double())
    assert all(torch.isfinite(t).all() for t in (z, weight, gate, expected))
    assert torch.isfinite(expected.to(dtype)).all()
    assert torch.count_nonzero(expected[:, :, 32:]) == expected[:, :, 32:].numel()
    return z, weight, gate, expected


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True], ids=["separate", "packed"])
def test_value_r3_range_mixed_heads(npu, dtype, packed):
    z, weight, gate, expected = _range_fixture(dtype, packed)
    runtime = build_runtime()  # Default PTO, not an explicit backend bypass.
    provider = runtime.kernel_bundle.projections
    factory = provider.value_gate_packed_kernel if packed else provider.value_gate_kernel
    kernel = factory(1, 17, 96, "float16" if dtype == torch.float16 else "bfloat16")
    assert str(kernel.target) == "pto"
    dz, dw, dg = z.npu(), weight.npu(), gate.npu()
    previous = None
    with torch.no_grad():
        for _ in range(3):
            actual = kernel(dz, dw, dg)
            torch.npu.synchronize()
            assert actual.shape == (1, 17, 96, 128) and actual.dtype == dtype
            reference.assert_fp16_close(actual, expected)
            # Separate group gates stop the huge-Z reference amplitude from
            # concealing small-value loss in a global normalized statistic.
            for begin, end in ((0, 32), (32, 64), (64, 96)):
                reference.assert_fp16_close(actual[:, :, begin:end], expected[:, :, begin:end])
            result = actual.cpu()
            if previous is not None:
                torch.testing.assert_close(result, previous, rtol=0, atol=0)
            previous = result
    for device, original in ((dz, z), (dw, weight), (dg, gate)):
        torch.testing.assert_close(device.cpu(), original, rtol=0, atol=0)
