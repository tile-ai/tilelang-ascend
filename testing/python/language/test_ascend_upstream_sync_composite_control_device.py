# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Runtime conditional GEMM paths with automatic synchronization only."""

import pytest
import torch

import tilelang
from tilelang import tvm

from test_ascend_upstream_sync_composite_control import (
    _conditional_accumulation,
    _conditional_gemms,
    _guarded_gemm,
    _mixed_conditional_gemm,
)
from test_ascend_upstream_sync_device import _PASS_CONFIGS

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


def _compile(program):
    tilelang.disable_cache()
    return tilelang.compile(
        program,
        out_idx=[],
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )


@pytest.mark.parametrize("with_else", [False, True], ids=["optional", "if_else"])
def test_conditional_gemms_alternate_paths_and_drain_between_calls(with_else):
    kernel = _compile(_conditional_gemms(with_else))
    generator = torch.Generator().manual_seed(20261104)
    for invocation, (count, choice) in enumerate(((0, 0), (3, -1), (3, 0), (1, 1), (3, 1), (0, -1), (2, 0))):
        shapes = ((16, 128), (128, 16), (16, 64), (64, 16))
        inputs = [(torch.randn(shape, generator=generator) * (invocation + 1) / 16).half() for shape in shapes]
        first, second, third, fourth = inputs
        outputs = [torch.full((48, 16), -17, dtype=torch.float16, device="npu") for _ in range(2)]
        expected = [torch.full((48, 16), -17, dtype=torch.float16) for _ in range(2)]
        for batch in range(count):
            expected[1][batch * 16 : (batch + 1) * 16] = (third.float() @ fourth.float()).half()
            if batch % 2 == choice or with_else:
                offset = 0 if batch % 2 == choice else 64
                expected[0][batch * 16 : (batch + 1) * 16] = (
                    first[:, offset : offset + 64].float() @ second[offset : offset + 64].float()
                ).half()
        kernel(*(value.npu() for value in inputs), *outputs, count, choice)
        torch.npu.synchronize()
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=1e-2, atol=1e-2)
            untouched = reference == -17
            assert torch.equal(actual.cpu()[untouched], reference[untouched])


def test_nested_guarded_gm_rectangle_keeps_invalid_tiles_untouched():
    kernel = _compile(_guarded_gemm())
    first = ((torch.arange(16 * 128).reshape(16, 128) % 31 - 15) / 16).half()
    second = ((torch.arange(64 * 16).reshape(64, 16) % 23 - 11) / 8).half()
    for tile in (-1, 0, 2, 1, -1, 1):
        output = torch.full((16, 16), -17, dtype=torch.float16, device="npu")
        expected = torch.full((16, 16), -17, dtype=torch.float16)
        if 0 <= tile < 2:
            expected = (first[:, tile * 64 : (tile + 1) * 64].float() @ second.float()).half()
        kernel(first.npu(), second.npu(), output, tile)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-2, atol=1e-2)


def test_optional_accumulation_keeps_the_unconditional_fix_correct():
    kernel = _compile(_conditional_accumulation())
    generator = torch.Generator().manual_seed(20261105)
    for choice in (0, 1, -1, 1, 0):
        first = (torch.randn((16, 128), generator=generator) / 8).half()
        second = (torch.randn((128, 16), generator=generator) / 8).half()
        output = torch.full((16, 16), -17, dtype=torch.float16, device="npu")
        reduction = 128 if choice > 0 else 64
        expected = (first[:, :reduction].float() @ second[:reduction].float()).half()
        kernel(first.npu(), second.npu(), output, choice)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-2, atol=1e-2)


def test_mixed_conditional_refresh_publishes_after_accumulation():
    kernel = _compile(_mixed_conditional_gemm())
    generator = torch.Generator().manual_seed(20261106)
    for _ in range(3):
        first = (torch.randn((64, 128), generator=generator) / 8).half()
        second = (torch.randn((128, 16), generator=generator) / 8).half()
        workspace = torch.full((64, 16), -17, dtype=torch.float32, device="npu")
        output = torch.full((64, 16), -17, dtype=torch.float32, device="npu")
        expected = torch.empty((64, 16), dtype=torch.float32)
        for group in range(2):
            for tile in range(2):
                row = group * 32 + tile * 16
                left = first[row : row + 16]
                if tile % 2:
                    left = torch.cat((left[:, 64:], left[:, :64]), dim=1)
                expected[row : row + 16] = left.float() @ second.float() * 2
        kernel(first.npu(), second.npu(), workspace, output)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-2, atol=1e-2)
