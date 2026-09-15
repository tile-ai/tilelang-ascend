import re
from unittest.mock import patch

import pytest

import tilelang
import tilelang.language as T

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

PACKED_ROWS = 8192
BLOCK_N = 128
BAD_ROW_STRIDE = PACKED_ROWS * BLOCK_N
DTYPE = "float32"
ASCENDC_EXPECTED_ARGS = f", {BLOCK_N}, 1, {BLOCK_N}"
PTO_STRIDE_PATTERN = re.compile(r"pto::Stride<([^>]*)>\(\)")


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


def _row_slice_copy_kernel():

    @T.prim_func
    def main(
        src: T.Tensor((PACKED_ROWS, BLOCK_N), DTYPE),  # type: ignore
        dst: T.Tensor((PACKED_ROWS, BLOCK_N), DTYPE),  # type: ignore
    ):
        with T.Kernel(1, threads=1, is_npu=True) as cid:
            row_ub = T.alloc_ub((BLOCK_N,), DTYPE)

            T.copy(src[cid, 0:BLOCK_N], row_ub)
            T.copy(row_ub, dst[cid, 0:BLOCK_N])

    return main


def _row_slice_atomic_add_kernel():

    @T.prim_func
    def main(
        dst: T.Tensor((PACKED_ROWS, BLOCK_N), DTYPE),  # type: ignore
    ):
        with T.Kernel(1, threads=1, is_npu=True) as cid:
            row_ub = T.alloc_ub((BLOCK_N,), DTYPE)

            T.tile.fill(row_ub, 1.0)
            T.tile.atomic_add(dst[cid, 0:BLOCK_N], row_ub)

    return main


def _compile_and_get_source(program, target):
    with (
        patch("tilelang.jit.adapter.libgen.LibraryGenerator.compile_lib") as mock_compile,
        patch(
            "tilelang.jit.adapter.libgen.LibraryGenerator.load_lib",
            return_value=None,
        ),
    ):
        mock_compile.return_value = None
        compiled = tilelang.compile(
            program,
            pass_configs=PASS_CONFIGS,
            target=target,
        )
    return compiled.get_kernel_source()


def _copy_rows(code, op_name):
    return [line.strip() for line in code.splitlines() if op_name in line]


def _assert_pto_row_stride(rows):
    for row in rows:
        match = PTO_STRIDE_PATTERN.search(row)
        assert match, row
        strides = [part.strip() for part in match.group(1).split(",")]
        assert strides[-2:] == [str(BLOCK_N), "1"], row


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_row_slice_copy_uses_last_dimension_as_gm_stride(target):
    code = _compile_and_get_source(_row_slice_copy_kernel(), target)
    copy_rows = _copy_rows(code, "copy_gm_to_ub") + _copy_rows(code, "copy_ub_to_gm")

    assert copy_rows, f"generated {target} code did not contain GM<->UB copy calls:\n{code}"
    if target == "pto":
        _assert_pto_row_stride(copy_rows)
    else:
        assert str(BAD_ROW_STRIDE) not in "\n".join(copy_rows), (
            "row-slice GM<->UB copies should use the last dimension as the GM row "
            f"stride ({BLOCK_N}), not last_dim * leading_dim ({BAD_ROW_STRIDE}):\n" + "\n".join(copy_rows)
        )
        assert all(ASCENDC_EXPECTED_ARGS in row for row in copy_rows), "\n".join(copy_rows)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_row_slice_atomic_add_uses_last_dimension_as_gm_stride(target):
    code = _compile_and_get_source(_row_slice_atomic_add_kernel(), target)
    atomic_rows = _copy_rows(code, "atomic_add_ub_to_gm")

    assert atomic_rows, f"generated {target} code did not contain atomic_add_ub_to_gm:\n{code}"
    if target == "pto":
        _assert_pto_row_stride(atomic_rows)
    else:
        assert str(BAD_ROW_STRIDE) not in "\n".join(atomic_rows), (
            "row-slice UB->GM atomic_add should use the last dimension as the GM row "
            f"stride ({BLOCK_N}), not last_dim * leading_dim ({BAD_ROW_STRIDE}):\n" + "\n".join(atomic_rows)
        )
        assert all(ASCENDC_EXPECTED_ARGS in row for row in atomic_rows), "\n".join(atomic_rows)
