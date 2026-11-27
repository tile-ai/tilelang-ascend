"""PTO composite helpers must preserve logical valid extents and slice tails."""

import os
import shutil
from pathlib import Path

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang.jit.adapter.libgen import LibraryGenerator


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def mul_add_dst_valid_shape(count, dtype, region):
    # Based on the existing vec_mul_add_dst/static-slice regressions. Each
    # vector subblock owns one GM row, so both can check the complete output.
    total = count if region == "full" else 64
    offset = 16 if region == "offset" else 0

    @T.prim_func
    def main(
        A: T.Tensor((2, total), dtype),
        B: T.Tensor((2, total), dtype),
        C: T.Tensor((2, total), dtype),
        D: T.Tensor((2, total), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a = T.alloc_ub((total,), dtype)
            b = T.alloc_ub((total,), dtype)
            c = T.alloc_ub((total,), dtype)
            T.copy(A[vid, 0], a)
            T.copy(B[vid, 0], b)
            T.copy(C[vid, 0], c)
            if region == "full":
                T.tile.mul_add_dst(c, a, b)
            else:
                T.tile.mul_add_dst(
                    c[offset : offset + count],
                    a[offset : offset + count],
                    b[offset : offset + count],
                )
            T.copy(c, D[vid, 0])

    return main


def _has_bisheng():
    ascend_home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_HOME")
    return shutil.which("bisheng") is not None or (
        ascend_home is not None and (Path(ascend_home) / "compiler/ccec_compiler/bin/bisheng").is_file()
    )


@pytest.mark.skipif(not _has_bisheng(), reason="Bisheng compiler is required")
@pytest.mark.parametrize("count", [31, 32, 33])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("region", ["full", "prefix", "offset"])
def test_mul_add_dst_valid_shape_compile(count, dtype, region):
    # Lowering alone cannot catch a mismatch between the helper template and
    # the generated tile type. Compile the generated kernel without a device.
    with tilelang.tvm.transform.PassContext(config=PASS_CONFIGS):
        artifact = tilelang.lower(mul_add_dst_valid_shape(count, dtype, region), target="pto", platform="A2")
    library = LibraryGenerator("pto", "A2")
    library.update_lib_code(artifact.kernel_source)
    library.compile_lib(timeout=120)
    assert Path(library.libpath).is_file()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU is required")
@pytest.mark.parametrize("count", [31, 32, 33])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("region", ["full", "prefix", "offset"])
def test_mul_add_dst_valid_shape_npu(count, dtype, region):
    total = count if region == "full" else 64
    offset = 16 if region == "offset" else 0
    kernel = tilelang.compile(
        mul_add_dst_valid_shape(count, dtype, region),
        out_idx=[-1],
        pass_configs=PASS_CONFIGS,
        target="pto",
    )
    # Small integer-valued inputs give exact products and sums in both dtypes.
    generator = torch.Generator().manual_seed(0)
    a, b, c = [torch.randint(-3, 4, (2, total), generator=generator).to(getattr(torch, dtype)).npu() for _ in range(3)]
    result = kernel(a, b, c)
    expected = c.clone()
    expected[:, offset : offset + count] += a[:, offset : offset + count] * b[:, offset : offset + count]
    # The entire buffer is checked, including the unchanged prefix and suffix.
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def unary_valid_shape(op, count, dtype, region):
    total = count if region == "full" else 64
    offset = 16 if region == "offset" else 0

    @T.prim_func
    def main(
        A: T.Tensor((2, total), dtype),
        C: T.Tensor((2, total), dtype),
        D: T.Tensor((2, total), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a = T.alloc_ub((total,), dtype)
            c = T.alloc_ub((total,), dtype)
            T.copy(A[vid, 0], a)
            T.copy(C[vid, 0], c)
            if op == "sigmoid":
                T.tile.sigmoid(c[offset : offset + count], a[offset : offset + count])
            elif op == "silu":
                T.tile.silu(c[offset : offset + count], a[offset : offset + count])
            else:
                T.tile.axpy(c[offset : offset + count], a[offset : offset + count], 2.0)
            T.copy(c, D[vid, 0])

    return main


@pytest.mark.skipif(not _has_bisheng(), reason="Bisheng compiler is required")
@pytest.mark.parametrize("op", ["sigmoid", "silu", "axpy"])
@pytest.mark.parametrize("count", [32, 33])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("region", ["full", "offset"])
def test_unary_valid_shape_compile(op, count, dtype, region):
    with tilelang.tvm.transform.PassContext(config=PASS_CONFIGS):
        artifact = tilelang.lower(unary_valid_shape(op, count, dtype, region), target="pto", platform="A2")
    library = LibraryGenerator("pto", "A2")
    library.update_lib_code(artifact.kernel_source)
    library.compile_lib(timeout=120)
    assert Path(library.libpath).is_file()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU is required")
@pytest.mark.parametrize("op", ["sigmoid", "silu", "axpy"])
@pytest.mark.parametrize("count", [32, 33])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("region", ["full", "offset"])
def test_unary_valid_shape_npu(op, count, dtype, region):
    total = count if region == "full" else 64
    offset = 16 if region == "offset" else 0
    kernel = tilelang.compile(
        unary_valid_shape(op, count, dtype, region),
        out_idx=[-1],
        pass_configs=PASS_CONFIGS,
        target="pto",
    )
    generator = torch.Generator().manual_seed(0)
    a, c = [torch.randint(-3, 4, (2, total), generator=generator).to(getattr(torch, dtype)).npu() for _ in range(2)]
    result = kernel(a, c)
    expected = c.clone()
    values = a[:, offset : offset + count]
    if op == "sigmoid":
        expected[:, offset : offset + count] = torch.sigmoid(values)
    elif op == "silu":
        expected[:, offset : offset + count] = torch.nn.functional.silu(values)
    else:
        expected[:, offset : offset + count] += values * 2
    tolerance = 2e-3 if dtype == "float16" else 1e-5
    torch.testing.assert_close(result, expected, rtol=tolerance, atol=tolerance)
    # Keep exact checks for untouched elements even when the activation needs
    # a floating-point tolerance on the updated region.
    torch.testing.assert_close(result[:, :offset], c[:, :offset], rtol=0, atol=0)
    torch.testing.assert_close(result[:, offset + count :], c[:, offset + count :], rtol=0, atol=0)
