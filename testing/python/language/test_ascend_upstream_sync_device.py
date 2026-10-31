# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""NPU regressions for the opt-in upstream synchronization engine.

These tests compile AscendC and execute on an Ascend device. They are separate
from the host-only IR tests. There are no manual barriers or flags, and Bisheng
automatic synchronization is disabled so it cannot repair a missing handoff.
"""

from typing import Any

import pytest
import torch

import tilelang
from tilelang import language as T
from tilelang import tvm
from tvm.tir import PrimFunc


pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)

_WIDTH = 64
_VECTOR_CORES = 2
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_SYNC_BACKEND: "upstream",
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}


@pytest.fixture(scope="module", autouse=True)
def _requires_upstream_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


def _compile_device_kernel(program: PrimFunc) -> Any:
    tilelang.disable_cache()
    return tilelang.compile(
        program,
        out_idx=[],
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )


@pytest.fixture(scope="module")
def copy_fill_store_kernel() -> Any:
    @T.prim_func
    def main(
        A: T.Tensor((_WIDTH,), "float32"),
        O: T.Tensor((_VECTOR_CORES, 2, _WIDTH), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, vid):
            original = T.alloc_ub((_WIDTH,), "float32")
            replacement = T.alloc_ub((_WIDTH,), "float32")
            T.annotate_address({original: 0, replacement: 0})
            with T.Scope("V"):
                T.copy(A, original)
                T.copy(original, O[vid, 0, :])
                T.tile.fill(replacement, 3)
                T.copy(replacement, O[vid, 1, :])

    return _compile_device_kernel(main)


def test_upstream_sync_copy_fill_store_preserves_physical_alias_read(
    copy_fill_store_kernel: Any,
) -> None:
    source = torch.arange(_WIDTH, dtype=torch.float32, device="npu") - 32
    output = torch.full((_VECTOR_CORES, 2, _WIDTH), -99.0, dtype=torch.float32, device="npu")
    expected = torch.full((_VECTOR_CORES, 2, _WIDTH), 3.0, dtype=torch.float32)
    expected[:, 0, :] = torch.arange(_WIDTH, dtype=torch.float32) - 32

    copy_fill_store_kernel(source, output)

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.fixture(scope="module")
def optional_alias_loop_kernel() -> Any:
    n = T.symbolic("n")

    @T.prim_func
    def main(
        A: T.Tensor((_WIDTH,), "float32"),
        O: T.Tensor((_VECTOR_CORES, 3, _WIDTH), "float32"),
        Count: T.Tensor((n,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, vid):
            original = T.alloc_ub((_WIDTH,), "float32")
            replacement = T.alloc_ub((_WIDTH,), "float32")
            T.annotate_address({original: 0, replacement: 0})
            with T.Scope("V"):
                T.tile.fill(replacement, 1)
                T.copy(original, O[vid, 0, :])
                for _iteration in T.serial(n):
                    T.tile.fill(replacement, 2)
                    T.copy(A, replacement)
                T.copy(original, O[vid, 1, :])
                T.tile.fill(replacement, 3)
                T.copy(replacement, O[vid, 2, :])

    return _compile_device_kernel(main)


@pytest.mark.parametrize("trip_count", [0, 1, 3])
def test_upstream_sync_optional_loop_preserves_alias_values(optional_alias_loop_kernel: Any, trip_count: int) -> None:
    source = torch.arange(_WIDTH, dtype=torch.float32, device="npu") - 32
    output = torch.full((_VECTOR_CORES, 3, _WIDTH), -99.0, dtype=torch.float32, device="npu")
    count = torch.empty((trip_count,), dtype=torch.int32, device="npu")
    expected = torch.full((_VECTOR_CORES, 3, _WIDTH), 3.0, dtype=torch.float32)
    expected[:, 0, :] = 1.0
    expected[:, 1, :] = torch.arange(_WIDTH, dtype=torch.float32) - 32 if trip_count else 1.0

    optional_alias_loop_kernel(source, output, count)

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("memory_planning", [False, True])
@pytest.mark.parametrize("operation", ["abs", "add", "relu_inplace"])
def test_upstream_sync_existing_developer_operators(operation: str, memory_planning: bool) -> None:
    import test_tilelang_ascend_language_elementwise as fixtures

    generator = torch.Generator().manual_seed(20261029)
    shape = (256,) if operation == "relu_inplace" else (64, 128)
    first = torch.randn(shape, generator=generator, dtype=torch.float32)
    if operation == "abs":
        program = fixtures.vec_abs(64, 128, 64, 128, "float")
        expected, arguments = torch.abs(first), [first.npu()]
    elif operation == "add":
        second = torch.randn(shape, generator=generator, dtype=torch.float32)
        program = fixtures.vec_add_developer(64, 128, 64, 128, "float")
        expected, arguments = first + second, [first.npu(), second.npu()]
    else:
        program = fixtures.activation_inplace("relu", 256, "float")
        expected, arguments = torch.relu(first), [first.npu()]
    tilelang.disable_cache()
    kernel = tilelang.compile(
        program,
        out_idx=[-1],
        target="ascendc",
        platform="A3",
        # Keep the default automatic C/V classification for Developer inputs.
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_SYNC_BACKEND: "upstream",
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: memory_planning,
        },
        compile_flags=["--cce-auto-sync=off"],
    )
    actual = kernel(*arguments)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("ordinary_sync", [False, True])
def test_upstream_sync_scalar_buffer_writes_complete_before_dma(ordinary_sync: bool) -> None:
    from test_tilelang_ascend_language_scalar_index import scalar_index_2d

    tilelang.disable_cache()
    kernel = tilelang.compile(
        scalar_index_2d(4, 8),
        out_idx=[1],
        target="ascendc",
        platform="A3",
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_SYNC_BACKEND: "upstream",
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: ordinary_sync,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
        },
        compile_flags=["--cce-auto-sync=off"],
    )
    source = torch.arange(32, dtype=torch.int32).reshape(4, 8)
    actual = kernel(source.npu())
    torch.testing.assert_close(actual.cpu(), source * 2, rtol=0, atol=0)


def test_upstream_sync_mutable_hardware_operand_is_captured_before_overwrite() -> None:
    @T.prim_func
    def main(
        A: T.Tensor((_WIDTH,), "float32"),
        O: T.Tensor((_VECTOR_CORES, _WIDTH), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, vid):
            source = T.alloc_ub((_WIDTH,), "float32")
            coefficient = T.alloc_ub((_WIDTH,), "float32")
            result = T.alloc_ub((_WIDTH,), "float32")
            with T.Scope("V"):
                T.copy(A, source)
                T.tile.fill(coefficient, 2)
                T.tile.mul(result, source, coefficient[0])
                T.tile.fill(coefficient, 17)
                T.copy(result, O[vid, :])

    kernel = _compile_device_kernel(main)
    source = torch.arange(_WIDTH, dtype=torch.float32) - 32
    output = torch.empty((_VECTOR_CORES, _WIDTH), dtype=torch.float32, device="npu")
    kernel(source.npu(), output)
    torch.testing.assert_close(output.cpu(), (source * 2).expand(2, -1), rtol=0, atol=0)


def test_upstream_sync_indirect_dma_address_is_captured_before_index_overwrite() -> None:
    @T.prim_func
    def main(A: T.Tensor((3, _WIDTH), "float32"), O: T.Tensor((2, 3, _WIDTH), "float32")):
        with T.Kernel(1, is_npu=True) as (_cid, vid):
            values = T.alloc_ub((_WIDTH,), "float32")
            index = T.alloc_ub((_WIDTH,), "int32")
            with T.Scope("V"):
                for row in T.serial(3):
                    T.tile.fill(index, 2 - row)
                    T.copy(A[row, 0], values)
                    T.copy(values, O[vid, index[0], 0])
                    T.tile.fill(index, 17)

    kernel = _compile_device_kernel(main.with_attr("tir.noalias", True))
    source = torch.arange(3 * _WIDTH, dtype=torch.float32).reshape(3, _WIDTH) - 32
    expected = source.flip(0).expand(2, -1, -1)
    for _invocation in range(3):
        output = torch.full(expected.shape, -999.0, dtype=torch.float32, device="npu")
        kernel(source.npu(), output)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.fixture(scope="module")
def optional_load_kernel() -> Any:
    n = T.symbolic("n")

    @T.prim_func
    def main(
        A: T.Tensor((_WIDTH,), "float32"),
        O: T.Tensor((_VECTOR_CORES, _WIDTH), "float32"),
        Count: T.Tensor((n,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, vid):
            value = T.alloc_ub((_WIDTH,), "float32")
            with T.Scope("V"):
                T.tile.fill(value, 1)
                if n > 0:
                    T.copy(A, value)
                T.copy(value, O[vid, :])

    return _compile_device_kernel(main)


@pytest.mark.parametrize("trip_count", [0, 1])
def test_upstream_sync_optional_producer_has_a_matching_handoff(optional_load_kernel: Any, trip_count: int) -> None:
    source = torch.arange(_WIDTH, dtype=torch.float32) - 32
    output = torch.empty((_VECTOR_CORES, _WIDTH), dtype=torch.float32, device="npu")
    count = torch.empty((trip_count,), dtype=torch.int32, device="npu")
    optional_load_kernel(source.npu(), output, count)
    expected = source.expand(_VECTOR_CORES, -1) if trip_count else torch.ones((_VECTOR_CORES, _WIDTH), dtype=torch.float32)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
