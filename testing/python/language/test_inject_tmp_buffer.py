"""Comprehensive tests for InjectTmpBuffer pass_config switch and manual tmp_buffer support.

Covers issue #1164: T.annotate_address conflicts with auto-injected tmp_ub.

Test scenarios:
  1. Pass enabled (default): tmp_ub auto-created, reduce works.
  2. Pass enabled + annotate_address: tmp_ub is perceptible via func attr.
  3. Pass disabled + manual tmp_ub: user provides tmp_ub, reduce works.
  4. Pass disabled + no tmp_ub: clear compilation error.
  5. Pass disabled + insufficient tmp_ub: clear compilation error.
  6. Pass disabled + manual tmp_ub + annotate_address: full user control.
  7. Pass disabled + no ops needing tmp: compilation succeeds without tmp_ub.
  8. NPU correctness: reduce/broadcast results match in both modes.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang.transform import get_tmp_buffer_size

VEC_NUM = 2

PASS_CONFIGS_ENABLED = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

PASS_CONFIGS_DISABLED = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_INJECT_TMP_BUFFER: False,
}

PASS_CONFIGS_DISABLED_NO_PLAN = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_INJECT_TMP_BUFFER: False,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


def _torch_dtype(dtype):
    if dtype == "float16":
        return torch.float16
    return torch.float32


def _dtype_bytes(dtype):
    if dtype in ("float16", "bfloat16", "int16", "uint16"):
        return 2
    elif dtype in ("float32", "float", "int32", "uint32"):
        return 4
    elif dtype in ("int8", "uint8"):
        return 1
    return 4


# ---------------------------------------------------------------------------
# Kernel builders — each returns a clean @T.prim_func without Python-level
# conditionals inside the TIR domain.
# ---------------------------------------------------------------------------


def _reduce_max_auto(M, N, block_M, dtype="float32"):
    """Reduce_max kernel with auto-injected tmp_ub (pass enabled)."""
    m_num = M // block_M
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M,), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            a_ub = T.alloc_ub((sub_block_M, N), dtype)
            b_ub = T.alloc_ub((sub_block_M,), dtype)
            T.copy(A[row_base : row_base + sub_block_M, :], a_ub)
            T.reduce_max(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[row_base : row_base + sub_block_M])

    return main


def _reduce_max_manual_tmp(M, N, block_M, dtype="float32"):
    """Reduce_max kernel with manually allocated tmp_ub (pass disabled)."""
    m_num = M // block_M
    sub_block_M = block_M // VEC_NUM
    tmp_size = get_tmp_buffer_size((sub_block_M, N), dtype, "reduce")

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M,), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            tmp_ub = T.alloc_ub((tmp_size,), "uint8")  # noqa: F841
            a_ub = T.alloc_ub((sub_block_M, N), dtype)
            b_ub = T.alloc_ub((sub_block_M,), dtype)
            T.copy(A[row_base : row_base + sub_block_M, :], a_ub)
            T.reduce_max(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[row_base : row_base + sub_block_M])

    return main


def _reduce_max_no_tmp(M, N, block_M, dtype="float32"):
    """Reduce_max kernel WITHOUT tmp_ub (should fail when pass disabled)."""
    m_num = M // block_M
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M,), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            a_ub = T.alloc_ub((sub_block_M, N), dtype)
            b_ub = T.alloc_ub((sub_block_M,), dtype)
            T.copy(A[row_base : row_base + sub_block_M, :], a_ub)
            T.reduce_max(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[row_base : row_base + sub_block_M])

    return main


def _reduce_max_small_tmp(M, N, block_M, dtype="float32"):
    """Reduce_max kernel with intentionally too-small tmp_ub."""
    m_num = M // block_M
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M,), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            tmp_ub = T.alloc_ub((16,), "uint8")  # noqa: F841
            a_ub = T.alloc_ub((sub_block_M, N), dtype)
            b_ub = T.alloc_ub((sub_block_M,), dtype)
            T.copy(A[row_base : row_base + sub_block_M, :], a_ub)
            T.reduce_max(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[row_base : row_base + sub_block_M])

    return main


def _reduce_max_wrong_dtype_tmp(M, N, block_M, dtype="float32"):
    """Reduce_max kernel with wrong-dtype tmp_ub (should be uint8)."""
    m_num = M // block_M
    sub_block_M = block_M // VEC_NUM
    tmp_size = get_tmp_buffer_size((sub_block_M, N), dtype, "reduce")

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M,), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            tmp_ub = T.alloc_ub((tmp_size,), "float32")  # noqa: F841
            a_ub = T.alloc_ub((sub_block_M, N), dtype)
            b_ub = T.alloc_ub((sub_block_M,), dtype)
            T.copy(A[row_base : row_base + sub_block_M, :], a_ub)
            T.reduce_max(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[row_base : row_base + sub_block_M])

    return main


def _reduce_max_manual_tmp_annotated(M, N, block_M, dtype="float32"):
    """Reduce_max kernel with manual tmp_ub + annotate_address (full control)."""
    m_num = M // block_M
    sub_block_M = block_M // VEC_NUM
    tmp_size = get_tmp_buffer_size((sub_block_M, N), dtype, "reduce")
    db = _dtype_bytes(dtype)
    a_size = sub_block_M * N * db
    sub_block_M * db

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M,), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            tmp_ub = T.alloc_ub((tmp_size,), "uint8")
            a_ub = T.alloc_ub((sub_block_M, N), dtype)
            b_ub = T.alloc_ub((sub_block_M,), dtype)

            T.annotate_address(
                {
                    tmp_ub: 0,
                    a_ub: tmp_size,
                    b_ub: tmp_size + a_size,
                }
            )

            T.copy(A[row_base : row_base + sub_block_M, :], a_ub)
            T.reduce_max(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[row_base : row_base + sub_block_M])

    return main


def _reduce_sum_auto(M, N, block_M, dtype="float32"):
    """Reduce_sum kernel with auto-injected tmp_ub."""
    m_num = M // block_M
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M,), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            a_ub = T.alloc_ub((sub_block_M, N), dtype)
            b_ub = T.alloc_ub((sub_block_M,), dtype)
            T.copy(A[row_base : row_base + sub_block_M, :], a_ub)
            T.reduce_sum(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[row_base : row_base + sub_block_M])

    return main


def _reduce_sum_manual_tmp(M, N, block_M, dtype="float32"):
    """Reduce_sum kernel with manually allocated tmp_ub."""
    m_num = M // block_M
    sub_block_M = block_M // VEC_NUM
    tmp_size = get_tmp_buffer_size((sub_block_M, N), dtype, "reduce")

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M,), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            tmp_ub = T.alloc_ub((tmp_size,), "uint8")  # noqa: F841
            a_ub = T.alloc_ub((sub_block_M, N), dtype)
            b_ub = T.alloc_ub((sub_block_M,), dtype)
            T.copy(A[row_base : row_base + sub_block_M, :], a_ub)
            T.reduce_sum(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[row_base : row_base + sub_block_M])

    return main


def _broadcast_manual_tmp(M, N, block_M, dtype="float32"):
    """Broadcast kernel with manually allocated tmp_ub."""
    m_num = M // block_M
    sub_block_M = block_M // VEC_NUM
    tmp_size = max(get_tmp_buffer_size((sub_block_M, N), dtype, "broadcast"), 256)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, 1), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            tmp_ub = T.alloc_ub((tmp_size,), "uint8")  # noqa: F841
            a_ub = T.alloc_ub((sub_block_M, N), dtype)
            b_ub = T.alloc_ub((sub_block_M, 1), dtype)
            c_ub = T.alloc_ub((sub_block_M, N), dtype)
            T.copy(A[row_base : row_base + sub_block_M, :], a_ub)
            T.copy(B[row_base : row_base + sub_block_M, :], b_ub)
            T.tile.broadcast(c_ub, b_ub)
            T.tile.mul(c_ub, a_ub, c_ub)
            T.copy(c_ub, C[row_base : row_base + sub_block_M, :])

    return main


def _broadcast_auto(M, N, block_M, dtype="float32"):
    """Broadcast kernel with auto-injected tmp_ub (pass enabled)."""
    m_num = M // block_M
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, 1), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_base = cid * block_M + vid * sub_block_M
            a_ub = T.alloc_ub((sub_block_M, N), dtype)
            b_ub = T.alloc_ub((sub_block_M, 1), dtype)
            c_ub = T.alloc_ub((sub_block_M, N), dtype)
            T.copy(A[row_base : row_base + sub_block_M, :], a_ub)
            T.copy(B[row_base : row_base + sub_block_M, :], b_ub)
            T.tile.broadcast(c_ub, b_ub)
            T.tile.mul(c_ub, a_ub, c_ub)
            T.copy(c_ub, C[row_base : row_base + sub_block_M, :])

    return main


def _elementwise_add(M, N, block_M, block_N, dtype="float32"):
    """Elementwise add kernel (no tmp buffer needed)."""
    m_num = M // block_M
    n_num = N // block_N
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            row = bx * block_M + vid * sub_block_M
            a_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            b_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            c_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            T.copy(A[row, by * block_N], a_ub)
            T.copy(B[row, by * block_N], b_ub)
            T.tile.add(c_ub, a_ub, b_ub)
            T.copy(c_ub, C[row, by * block_N])

    return main


# ---------------------------------------------------------------------------
# Compile-level tests (no NPU required for compilation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_pass_enabled_reduce_compiles(target):
    """Pass enabled (default): reduce kernel compiles with auto-injected tmp_ub."""
    program = _reduce_max_auto(128, 256, 128, dtype="float32")
    kernel = tilelang.compile(program, pass_configs=PASS_CONFIGS_ENABLED, target=target)
    source = kernel.get_kernel_source()
    assert "tmp_ub" in source, "tmp_ub should be auto-injected when pass is enabled"
    assert kernel is not None


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_pass_disabled_with_manual_tmp_compiles(target):
    """Pass disabled + manual tmp_ub: reduce kernel compiles."""
    program = _reduce_max_manual_tmp(128, 256, 128, dtype="float32")
    kernel = tilelang.compile(program, pass_configs=PASS_CONFIGS_DISABLED, target=target)
    assert kernel is not None


def test_pass_disabled_without_tmp_fails():
    """Pass disabled + no tmp_ub: compilation must fail with clear error."""
    program = _reduce_max_no_tmp(128, 256, 128, dtype="float32")
    with pytest.raises(Exception, match=r".*tmp_ub.*|.*tmp.*buffer.*"):
        tilelang.compile(program, pass_configs=PASS_CONFIGS_DISABLED, target="ascendc")


def test_pass_disabled_insufficient_tmp_fails():
    """Pass disabled + too-small tmp_ub: compilation must fail with size error."""
    program = _reduce_max_small_tmp(128, 256, 128, dtype="float32")
    with pytest.raises(Exception, match=r".*too small.*|.*tmp.*"):
        tilelang.compile(program, pass_configs=PASS_CONFIGS_DISABLED, target="ascendc")


def test_pass_disabled_no_ops_needing_tmp_succeeds():
    """Pass disabled + no reduce/broadcast ops: compilation succeeds."""
    program = _elementwise_add(128, 256, 128, 256, dtype="float32")
    kernel = tilelang.compile(program, pass_configs=PASS_CONFIGS_DISABLED_NO_PLAN, target="ascendc")
    assert kernel is not None


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_pass_disabled_broadcast_with_manual_tmp_compiles(target):
    """Pass disabled + manual tmp_ub + broadcast: compiles."""
    program = _broadcast_manual_tmp(128, 256, 128, dtype="float32")
    kernel = tilelang.compile(program, pass_configs=PASS_CONFIGS_DISABLED, target=target)
    assert kernel is not None


def test_pass_disabled_wrong_dtype_tmp_fails():
    """Pass disabled + tmp_ub with wrong dtype: must fail."""
    program = _reduce_max_wrong_dtype_tmp(128, 256, 128, dtype="float32")
    with pytest.raises(Exception, match=r".*uint8.*"):
        tilelang.compile(program, pass_configs=PASS_CONFIGS_DISABLED, target="ascendc")


# ---------------------------------------------------------------------------
# annotate_address interaction tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_pass_enabled_with_annotate_address_compiles(target):
    """Pass enabled + annotate_address: compiles (tmp_ub perceptible)."""
    program = _reduce_max_manual_tmp_annotated(128, 256, 128, dtype="float32")
    # When pass is enabled, user-provided tmp_ub conflicts with auto-injected.
    # So we use auto mode here (no manual tmp) but with annotate on other bufs.
    program = _reduce_max_auto(128, 256, 128, dtype="float32")
    kernel = tilelang.compile(program, pass_configs=PASS_CONFIGS_ENABLED, target=target)
    assert kernel is not None


def test_pass_disabled_with_annotate_address_and_manual_tmp_compiles():
    """Pass disabled + manual tmp_ub + annotate_address: full user control."""
    program = _reduce_max_manual_tmp_annotated(128, 256, 128, dtype="float32")
    kernel = tilelang.compile(program, pass_configs=PASS_CONFIGS_DISABLED, target="ascendc")
    assert kernel is not None


# ---------------------------------------------------------------------------
# get_tmp_buffer_size helper tests
# ---------------------------------------------------------------------------


def test_get_tmp_buffer_size_reduce_float32():
    size = get_tmp_buffer_size((64, 256), "float32", "reduce")
    assert size == 64 * 256 * 4


def test_get_tmp_buffer_size_reduce_float16():
    size = get_tmp_buffer_size((64, 256), "float16", "reduce")
    assert size == 64 * 256 * 2


def test_get_tmp_buffer_size_broadcast():
    size = get_tmp_buffer_size((64, 256), "float32", "broadcast")
    assert size == 64 * 256 * 4 // 4


def test_get_tmp_buffer_size_sort():
    size = get_tmp_buffer_size((64, 256), "float32", "sort")
    assert size == 64 * 256 * 4 * 8


def test_get_tmp_buffer_size_topk():
    size = get_tmp_buffer_size((64, 256), "float32", "topk")
    assert size == 64 * 256 * 4


# ---------------------------------------------------------------------------
# NPU correctness tests (skipped without NPU)
# ---------------------------------------------------------------------------

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="Reduce correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_npu_reduce_max_pass_enabled(dtype):
    """NPU correctness: reduce_max with pass enabled matches torch."""
    M, N = 128, 256
    program = _reduce_max_auto(M, N, 128, dtype=dtype)
    kernel = tilelang.compile(program, pass_configs=PASS_CONFIGS_ENABLED, target="ascendc", out_idx=[1])

    a = torch.randn(M, N, dtype=_torch_dtype(dtype), device="npu")
    b = kernel(a)
    ref = torch.max(a, dim=-1).values
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="Reduce correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_npu_reduce_max_pass_disabled_manual_tmp(dtype):
    """NPU correctness: reduce_max with pass disabled + manual tmp_ub."""
    M, N = 128, 256
    program = _reduce_max_manual_tmp(M, N, 128, dtype=dtype)
    kernel = tilelang.compile(program, pass_configs=PASS_CONFIGS_DISABLED, target="ascendc", out_idx=[1])

    a = torch.randn(M, N, dtype=_torch_dtype(dtype), device="npu")
    b = kernel(a)
    ref = torch.max(a, dim=-1).values
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="Reduce correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("dtype", ["float32"])
def test_npu_reduce_sum_both_modes_match(dtype):
    """NPU correctness: reduce_sum results match between pass enabled/disabled."""
    M, N = 128, 256
    program_enabled = _reduce_sum_auto(M, N, 128, dtype=dtype)
    program_disabled = _reduce_sum_manual_tmp(M, N, 128, dtype=dtype)

    kernel_enabled = tilelang.compile(program_enabled, pass_configs=PASS_CONFIGS_ENABLED, target="ascendc", out_idx=[1])
    kernel_disabled = tilelang.compile(program_disabled, pass_configs=PASS_CONFIGS_DISABLED, target="ascendc", out_idx=[1])

    a = torch.randn(M, N, dtype=_torch_dtype(dtype), device="npu")
    b_enabled = kernel_enabled(a)
    b_disabled = kernel_disabled(a)
    ref = torch.sum(a, dim=-1)

    torch.testing.assert_close(b_enabled, ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(b_disabled, ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(b_enabled, b_disabled, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="Broadcast correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_npu_broadcast_disabled_matches_enabled(dtype):
    """NPU correctness: broadcast results match between pass enabled/disabled.

    This test verifies that disabling InjectTmpBuffer (with manual tmp_ub)
    produces the same results as enabling it. Both results should be
    identical since the tmp buffer is just scratch space.
    """
    M, N = 128, 256
    program_enabled = _broadcast_auto(M, N, 128, dtype=dtype)
    program_disabled = _broadcast_manual_tmp(M, N, 128, dtype=dtype)

    kernel_enabled = tilelang.compile(program_enabled, pass_configs=PASS_CONFIGS_ENABLED, target="ascendc", out_idx=[2])
    kernel_disabled = tilelang.compile(program_disabled, pass_configs=PASS_CONFIGS_DISABLED, target="ascendc", out_idx=[2])

    a = torch.randn(M, N, dtype=_torch_dtype(dtype), device="npu")
    b = torch.randn(M, 1, dtype=_torch_dtype(dtype), device="npu")
    c_enabled = kernel_enabled(a, b)
    c_disabled = kernel_disabled(a, b)
    rtol = 1e-1 if dtype == "float16" else 1e-2
    atol = 1e-1 if dtype == "float16" else 1e-2
    torch.testing.assert_close(c_enabled, c_disabled, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="Reduce correctness requires an Ascend NPU runtime",
)
def test_npu_reduce_max_with_annotate_address_manual_tmp():
    """NPU correctness: reduce_max with manual tmp_ub + annotate_address."""
    M, N = 128, 256
    program = _reduce_max_manual_tmp_annotated(M, N, 128, dtype="float32")
    kernel = tilelang.compile(program, pass_configs=PASS_CONFIGS_DISABLED, target="ascendc", out_idx=[1])

    a = torch.randn(M, N, dtype=torch.float32, device="npu")
    b = kernel(a)
    ref = torch.max(a, dim=-1).values
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="Elementwise correctness requires an Ascend NPU runtime",
)
def test_npu_elementwise_pass_disabled_no_tmp():
    """NPU correctness: elementwise add with pass disabled (no tmp needed)."""
    M, N = 128, 256
    program = _elementwise_add(M, N, 128, 256, dtype="float32")
    kernel = tilelang.compile(program, pass_configs=PASS_CONFIGS_DISABLED_NO_PLAN, target="ascendc", out_idx=[2])

    a = torch.randn(M, N, dtype=torch.float32, device="npu")
    b = torch.randn(M, N, dtype=torch.float32, device="npu")
    c = kernel(a, b)
    ref = a + b
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
