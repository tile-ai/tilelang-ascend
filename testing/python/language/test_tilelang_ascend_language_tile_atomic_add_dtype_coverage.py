"""
T.tile.atomic_add dtype coverage tests.

Existing coverage in test_tilelang_ascend_language_tile_atomic_add.py:
- float32/float16 x ascendc/pto (1D UB->GM)
- float32 x ascendc/pto (2D UB->GM)
- float16 x ascendc/pto (2D L0C->GM via GEMM)

This file supplements with:
1. int32 dtype coverage (1D UB->GM, ascendc + pto) — discovered support beyond ftcheck
2. Scope violation tests (dst=UB, src=GM must raise)
3. Dtype mismatch test (dst float32, src float16 must raise)
4. Unsupported dtype compilation errors (uint16/uint32 x pto, int8 x ascendc)

Test suite follows the simplification principle for direct-intrinsic APIs
(mentor z00520135 review on PR4): since atomic_add has no type-specific
processing logic, only representative dtypes are tested.
"""

import pytest
import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

VEC_NUM = 2


def _torch_dtype(dtype):
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "int16": torch.int16,
        "int32": torch.int32,
        "int8": torch.int8,
    }
    return mapping[dtype]


def _fill_value(dtype):
    if dtype in ("int16", "int32", "int8"):
        return 1
    return 1.0


def _make_ub_to_gm_1d(dtype, tile_n=32, num_blocks=4):
    @T.prim_func
    def main(C: T.Tensor((tile_n,), dtype)):  # type: ignore
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((tile_n,), dtype)
            T.tile.fill(src_ub, _fill_value(dtype))
            T.tile.atomic_add(C[0], src_ub)

    return main


def _run_and_check(program, shape, dtype, num_blocks, target):
    kernel = tilelang.compile(program, pass_configs=PASS_CONFIGS, target=target)
    torch_dtype = _torch_dtype(dtype)
    out = torch.zeros(shape, dtype=torch_dtype, device="npu")
    torch.npu.synchronize()
    kernel(out)
    torch.npu.synchronize()
    expected = torch.full(shape, num_blocks * VEC_NUM, dtype=torch_dtype, device="npu")
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Functional tests: int32 dtype (default — core integer type, discovered support)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="tile atomic_add correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize(
    "target",
    [
        pytest.param("ascendc", marks=pytest.mark.low_priority, id="int32-ascendc"),
        pytest.param("pto", id="int32-pto"),
    ],
)
def test_atomic_add_int32_1d(target):
    num_blocks = 4
    tile_n = 32
    program = _make_ub_to_gm_1d("int32", tile_n=tile_n, num_blocks=num_blocks)
    _run_and_check(program, (tile_n,), "int32", num_blocks, target)


# ---------------------------------------------------------------------------
# Exception boundary: scope violations (default — no LP)
# ---------------------------------------------------------------------------


def test_atomic_add_dst_scope_violation_raises():
    """dst must be GM scope; using UB should raise during TIR construction."""

    with pytest.raises(RuntimeError, match="DiagnosticError"):

        @T.prim_func
        def main(C: T.Tensor((32,), "float32")):  # type: ignore
            with T.Kernel(1, is_npu=True) as (cid, _):
                dst_ub = T.alloc_ub((32,), "float32")
                src_ub = T.alloc_ub((32,), "float32")
                T.tile.fill(src_ub, 1.0)
                T.tile.atomic_add(dst_ub[0], src_ub)


def test_atomic_add_src_scope_violation_raises():
    """src must be local scope; using GM should raise during TIR construction."""

    with pytest.raises(RuntimeError, match="DiagnosticError"):

        @T.prim_func
        def main(
            C: T.Tensor((32,), "float32"),  # type: ignore
            D: T.Tensor((32,), "float32"),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                T.tile.atomic_add(C[0], D[0])


# ---------------------------------------------------------------------------
# Exception boundary: dtype mismatch (default — no LP)
# ---------------------------------------------------------------------------


def test_atomic_add_dtype_mismatch_raises():
    """dst and src dtype must match; mismatch should raise at compile time."""

    @T.prim_func
    def main(C: T.Tensor((32,), "float32")):  # type: ignore
        with T.Kernel(1, is_npu=True) as (cid, _):
            src_ub = T.alloc_ub((32,), "float16")
            T.tile.fill(src_ub, 1.0)
            T.tile.atomic_add(C[0], src_ub)

    with pytest.raises(RuntimeError, match="dtype to match"):  # noqa: B017
        tilelang.compile(main, pass_configs=PASS_CONFIGS, target="ascendc")


# ---------------------------------------------------------------------------
# Exception boundary: unsupported dtype (default — no LP)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,target",
    [
        ("uint16", "pto"),
        pytest.param("uint32", "pto", marks=pytest.mark.low_priority, id="uint32-pto"),
        ("int8", "ascendc"),
    ],
)
def test_atomic_add_unsupported_dtype_raises(dtype, target):
    """uint16/uint32 x pto and int8 x ascendc should fail at compile time."""

    @T.prim_func
    def main(C: T.Tensor((32,), dtype)):  # type: ignore
        with T.Kernel(1, is_npu=True) as (cid, _):
            src_ub = T.alloc_ub((32,), dtype)
            T.tile.fill(src_ub, 1)
            T.tile.atomic_add(C[0], src_ub)

    with pytest.raises(RuntimeError, match="Compilation Failed"):  # noqa: B017
        tilelang.compile(main, pass_configs=PASS_CONFIGS, target=target)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
