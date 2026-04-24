from pathlib import Path

import pytest
import torch

import tilelang
import tilelang.language as T


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

EXPECTED_FRONTEND_REJECTION_ERRORS = (
    TypeError,
    ValueError,
    AssertionError,
    tilelang.tvm.error.DiagnosticError,
)
VEC_NUM = 2
REPO_ROOT = Path(__file__).resolve().parents[3]
ASCEND_COMMON_H = REPO_ROOT / "src" / "tl_templates" / "ascend" / "common.h"


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


def _compile(program):
    return tilelang.compile(program, pass_configs=PASS_CONFIGS, target="ascendc")


def _tile_atomic_add_kernel(num_blocks=4, tile_n=32, dtype="float32"):
    @T.prim_func
    def main(C: T.Tensor((tile_n,), dtype)):  # type: ignore
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((tile_n,), dtype)

            with T.Scope("V"):
                T.tile.fill(src_ub, 1.0)
                T.barrier_all()
                T.tile.atomic_add(C[0], src_ub)

    return main


def _tile_atomic_add_2d_kernel(tile_m=4, tile_n=32, dtype="float32"):
    @T.prim_func
    def main(C: T.Tensor((tile_m, tile_n), dtype)):  # type: ignore
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((tile_m, tile_n), dtype)

            with T.Scope("V"):
                T.tile.fill(src_ub, 1.0)
                T.barrier_all()
                T.tile.atomic_add(C[0, 0], src_ub)

    return main


def _plain_copy_kernel(tile_n=32, dtype="float32"):
    @T.prim_func
    def main(C: T.Tensor((tile_n,), dtype)):  # type: ignore
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((tile_n,), dtype)

            with T.Scope("V"):
                T.tile.fill(src_ub, 1.0)
                T.barrier_all()
                T.copy(src_ub, C[0])

    return main


def _dst_non_gm_kernel(tile_n=32, dtype="float32"):
    @T.prim_func
    def main(C: T.Tensor((tile_n,), dtype)):  # type: ignore
        with T.Kernel(1, is_npu=True) as (cid, vid):
            dst_ub = T.alloc_ub((tile_n,), dtype)
            src_ub = T.alloc_ub((tile_n,), dtype)

            with T.Scope("V"):
                T.tile.fill(src_ub, 1.0)
                T.tile.atomic_add(dst_ub, src_ub)
                T.copy(dst_ub, C[0])

    return main


def _src_non_local_kernel(tile_n=32, dtype="float32"):
    @T.prim_func
    def main(
        A: T.Tensor((tile_n,), dtype),  # type: ignore
        C: T.Tensor((tile_n,), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid), T.Scope("V"):
            T.tile.atomic_add(C[0], A[0])

    return main


def _main_repo_style_kwargs_kernel(keyword, tile_n=32, dtype="float32"):
    if keyword == "memory_order":

        @T.prim_func
        def main(C: T.Tensor((tile_n,), dtype)):  # type: ignore
            with T.Kernel(1, is_npu=True) as (cid, vid):
                src_ub = T.alloc_ub((tile_n,), dtype)
                with T.Scope("V"):
                    T.tile.atomic_add(C[0], src_ub, memory_order="relaxed")

        return main

    if keyword == "return_prev":

        @T.prim_func
        def main(C: T.Tensor((tile_n,), dtype)):  # type: ignore
            with T.Kernel(1, is_npu=True) as (cid, vid):
                src_ub = T.alloc_ub((tile_n,), dtype)
                with T.Scope("V"):
                    T.tile.atomic_add(C[0], src_ub, return_prev=True)

        return main

    if keyword == "use_tma":

        @T.prim_func
        def main(C: T.Tensor((tile_n,), dtype)):  # type: ignore
            with T.Kernel(1, is_npu=True) as (cid, vid):
                src_ub = T.alloc_ub((tile_n,), dtype)
                with T.Scope("V"):
                    T.tile.atomic_add(C[0], src_ub, use_tma=True)

        return main

    raise ValueError(f"Unsupported keyword case: {keyword}")


def _constant_src_kernel(tile_n=32, dtype="float32"):
    @T.prim_func
    def main(C: T.Tensor((tile_n,), dtype)):  # type: ignore
        with T.Kernel(1, is_npu=True) as (cid, vid), T.Scope("V"):
            T.tile.atomic_add(C[0], T.float32(1.0))

    return main


def _expression_src_kernel(tile_n=32, dtype="float32"):
    @T.prim_func
    def main(C: T.Tensor((tile_n,), dtype)):  # type: ignore
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((tile_n,), dtype)

            with T.Scope("V"):
                T.tile.fill(src_ub, 1.0)
                T.tile.atomic_add(C[0], src_ub[0] + T.float32(1.0))

    return main


def _has_atomic_disable_compat(source):
    lower_source = source.lower()
    helper_markers = (
        "disable_dma_atomic_compat",
        "TL_ASCEND_DISABLE_DMA_ATOMIC",
    )
    helper_like_name = (
        "atomic" in lower_source and "compat" in lower_source and any(marker in lower_source for marker in ("disable", "close"))
    )
    guarded_direct_close = (
        "DisableDmaAtomic" in source and "SetAtomicNone" in source and any(marker in source for marker in ("#if", "#ifdef", "#elif"))
    )
    return any(marker in source for marker in helper_markers) or helper_like_name or guarded_direct_close


def _read_ascend_common_source():
    return ASCEND_COMMON_H.read_text(encoding="utf-8")


def test_tile_atomic_add_compile_only():
    kernel = _compile(_tile_atomic_add_kernel())

    assert kernel.get_kernel_source()


def test_tile_atomic_add_source_uses_atomic_helper_with_disable_compat():
    kernel = _compile(_tile_atomic_add_kernel())
    source = kernel.get_kernel_source()
    common_source = _read_ascend_common_source()
    source_bundle = source + "\n" + common_source

    assert "tl_templates/ascend/common.h" in source
    assert "atomic_add_ub_to_gm" in source
    assert "SetAtomicAdd" in source_bundle
    assert _has_atomic_disable_compat(source_bundle)


def test_tile_atomic_add_compile_2d_region_source_uses_atomic_helper():
    kernel = _compile(_tile_atomic_add_2d_kernel())
    source = kernel.get_kernel_source()

    assert "atomic_add_ub_to_gm" in source


def test_tile_atomic_add_compile_float16_source_uses_half_helper():
    kernel = _compile(_tile_atomic_add_kernel(dtype="float16"))
    source = kernel.get_kernel_source()

    assert "atomic_add_ub_to_gm<half" in source


def test_plain_copy_source_does_not_use_atomic_add_helper():
    kernel = _compile(_plain_copy_kernel())
    source = kernel.get_kernel_source()

    assert "copy_ub_to_gm" in source
    assert "atomic_add_ub_to_gm" not in source


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="tile atomic_add correctness requires an Ascend NPU runtime",
)
def test_tile_atomic_add_correctness_accumulates_multiple_blocks_after_zeroing_gm():
    num_blocks = 4
    tile_n = 32
    kernel = _compile(_tile_atomic_add_kernel(num_blocks=num_blocks, tile_n=tile_n))

    out = torch.empty((tile_n,), dtype=torch.float32, device="npu")
    out.zero_()
    torch.npu.synchronize()

    kernel(out)
    torch.npu.synchronize()

    expected = torch.full(
        (tile_n,),
        num_blocks * VEC_NUM,
        dtype=torch.float32,
        device="npu",
    )
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_tile_atomic_add_rejects_non_gm_dst():
    with pytest.raises(EXPECTED_FRONTEND_REJECTION_ERRORS):
        _compile(_dst_non_gm_kernel())


def test_tile_atomic_add_rejects_non_local_src():
    with pytest.raises(EXPECTED_FRONTEND_REJECTION_ERRORS):
        _compile(_src_non_local_kernel())


@pytest.mark.parametrize("keyword", ["memory_order", "return_prev", "use_tma"])
def test_tile_atomic_add_rejects_main_repo_style_keyword_args(keyword):
    with pytest.raises(EXPECTED_FRONTEND_REJECTION_ERRORS):
        _compile(_main_repo_style_kwargs_kernel(keyword))


@pytest.mark.parametrize(
    "program_factory",
    [
        pytest.param(_constant_src_kernel, id="constant-src"),
        pytest.param(_expression_src_kernel, id="expression-src"),
    ],
)
def test_tile_atomic_add_rejects_constant_or_expression_src(program_factory):
    with pytest.raises(EXPECTED_FRONTEND_REJECTION_ERRORS):
        _compile(program_factory())
