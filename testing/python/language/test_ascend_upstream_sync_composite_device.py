# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Device regressions for the audited flat AscendC gemm_v0 helper contract."""

import pytest
import torch

import tilelang
from tilelang import tvm

from test_ascend_upstream_sync_device import _PASS_CONFIGS
from test_ascend_upstream_sync_composite_loop import _ordinary_k_gemm
from test_ascend_upstream_sync_composite_invocation import (
    _dynamic_nested_gemm,
    _independent_gemms,
    _nested_gemm,
    _padded_refresh_gemm,
    _repeated_tiled_gemm,
)
from test_tilelang_ascend_language_gemm_v0_n_tiling import gemm_nt


pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.mark.parametrize("k", [64, 256])
def test_flat_gemm_internal_event_pools_are_drained_between_invocations(k: int) -> None:
    # The existing factory contains one flat gemm_v0 call, with no manual
    # barriers or flags. K=256 also exercises its internal L0 ping-pong.
    tilelang.disable_cache()
    kernel = tilelang.compile(
        gemm_nt(16, 16, k, "float16", "float"),
        out_idx=[-1],
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    generator = torch.Generator().manual_seed(20261030 + k)
    for _invocation in range(3):
        first = torch.randn((16, k), generator=generator, dtype=torch.float16)
        second = torch.randn((k, 16), generator=generator, dtype=torch.float16)
        expected = (first.float() @ second.float()).half()
        actual = kernel(first.npu(), second.npu())
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("trip_count", [2, 3])
def test_ordinary_k_loop_without_manual_sync(trip_count: int) -> None:
    tilelang.disable_cache()
    kernel = tilelang.compile(
        _ordinary_k_gemm(trip_count),
        out_idx=[-1],
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    full_k = trip_count * 64
    generator = torch.Generator().manual_seed(20261031 + trip_count)
    for _invocation in range(3):
        first = torch.randn((16, full_k), generator=generator, dtype=torch.float16)
        second = torch.randn((full_k, 16), generator=generator, dtype=torch.float16)
        expected = (first.float() @ second.float()).half()
        actual = kernel(first.npu(), second.npu())
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shared_a", [False, True])
def test_two_gemms_reuse_hidden_pools_and_store_inside_the_outer_loop(shared_a):
    tilelang.disable_cache()
    kernel = tilelang.compile(
        _independent_gemms(shared_a=shared_a),
        out_idx=[-2, -1],
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    generator = torch.Generator().manual_seed(2201)
    for _ in range(3):
        shapes = ((16, 64), (64, 16), (16, 64), (64, 16))
        tensors = [(torch.randn(shape, generator=generator) * 0.25).half() for shape in shapes]
        first, second, third, fourth = tensors
        expected = [
            (first.float() @ second.float()).half().repeat(2, 1),
            ((first if shared_a else third).float() @ fourth.float()).half().repeat(2, 1),
        ]
        actual = kernel(*(value.npu() for value in tensors))
        torch.npu.synchronize()
        assert len(actual) == 2
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output.cpu(), reference, rtol=1e-2, atol=1e-2)


def test_nested_k_loops_preserve_accumulation_and_fix_before_reuse():
    tilelang.disable_cache()
    kernel = tilelang.compile(
        _nested_gemm(),
        out_idx=[-1],
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    generator = torch.Generator().manual_seed(2202)
    for _ in range(3):
        first = (torch.randn((16, 128), generator=generator) * 0.25).half()
        second = (torch.randn((128, 16), generator=generator) * 0.25).half()
        expected = (first.float() @ second.float()).half().repeat(3, 1)
        actual = kernel(first.npu(), second.npu())
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)


def test_dynamic_nested_gemm_zero_trip_does_not_store_or_leave_event_credits():
    tilelang.disable_cache()
    kernel = tilelang.compile(
        _dynamic_nested_gemm(),
        out_idx=[],
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    first = torch.ones((16, 128), dtype=torch.float16, device="npu")
    second = torch.ones((128, 16), dtype=torch.float16, device="npu")
    for count in (0, 1, 3, 2, 0):
        output = torch.full((48, 16), -17, dtype=torch.float16, device="npu")
        kernel(first, second, output, count)
        torch.npu.synchronize()
        expected = torch.full((48, 16), -17, dtype=torch.float16)
        expected[: count * 16] = 128
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize(
    "shape,tile_k",
    [((1, 16, 64), 64), ((13, 11, 47), 64), ((13, 11, 47), 47)],
)
def test_partial_gm_l1_refresh_clears_the_whole_poisoned_fractal(dtype, shape, tile_k):
    rows, columns, reduction = shape
    tilelang.disable_cache()
    kernel = tilelang.compile(
        _padded_refresh_gemm(*shape, dtype=dtype, tile_k=tile_k),
        out_idx=[-1],
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    torch_dtype = getattr(torch, dtype)
    for poison, patterned in ((7, False), (-3, True), (11, True)):
        if patterned:
            left = ((torch.arange(rows * reduction) % 23) - 11).reshape(rows, reduction)
            right = ((torch.arange(reduction * columns) % 19) - 9).reshape(reduction, columns)
            left = left.to(torch_dtype) / 16
            right = right.to(torch_dtype) / 8
        else:
            left = torch.ones((rows, reduction), dtype=torch_dtype)
            right = torch.ones((reduction, columns), dtype=torch_dtype)
        # Dyadic inputs make every product and short reduction exact in FP32,
        # while nonuniform values expose permutations hidden by all-ones data.
        expected = torch.zeros((16, 16), dtype=torch.float32)
        expected[:rows, :columns] = left.float() @ right.float()
        inputs = [
            torch.full((16, tile_k), poison, dtype=torch_dtype, device="npu"),
            torch.full((tile_k, 16), poison + 1, dtype=torch_dtype, device="npu"),
            left.npu(),
            right.npu(),
        ]
        actual = kernel(*inputs)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype,m,n,k,k_l0",
    [
        ("float16", 16, 1360, 96, 96),
        ("float32", 16, 1360, 48, 48),
        ("float16", 16, 688, 192, 96),
        ("float32", 16, 688, 96, 48),
        ("float16", 13, 16, 128, 64),
    ],
)
def test_non_power_of_two_k_tiles_keep_both_l0_slots_and_n_subtiles_disjoint(
    dtype,
    m,
    n,
    k,
    k_l0,
):
    # The first two points are the original review counterexamples, without
    # shrinking their logical shape. The other points also alternate K slots.
    tilelang.disable_cache()
    kernel = tilelang.compile(
        _repeated_tiled_gemm(m, n, k, k_l0, dtype),
        out_idx=[-1],
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    torch_dtype = getattr(torch, dtype)
    for phase in (0, 3, 7):
        first = ((torch.arange(m * k) + phase) % 23 - 11).reshape(m, k)
        second = ((torch.arange(k * n) + phase * 5) % 19 - 9).reshape(k, n)
        first = first.to(torch_dtype) / 16
        second = second.to(torch_dtype) / 8
        # Every product and reduction is exactly representable in FP32.
        expected = (first.float() @ second.float()).repeat(2, 1, 1)
        actual = kernel(first.npu(), second.npu())
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
