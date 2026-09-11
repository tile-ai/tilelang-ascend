"""Compact fp32 row-reduction contracts and unchanged backend fallbacks."""

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.transform.pass_config import process_default_pass_config

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _kernel(m, n, physical_row, dtype, kind, clear, *, follow_with_add=False):
    reduce_fn = {"sum": T.reduce_sum, "max": T.reduce_max, "min": T.reduce_min}[kind]

    @T.prim_func
    def main(
        a: T.Tensor((m, physical_row), dtype),  # type: ignore
        b: T.Tensor((m,), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            src = T.alloc_ub((m, physical_row), dtype)
            dst = T.alloc_ub((m,), dtype)
            with T.Scope("V"):
                if vid == 0:
                    T.copy(a, src)
                    if not clear:
                        T.tile.fill(dst, 2.0)
                    reduce_fn(src, dst, dim=-1, real_shape=[m, n], clear=clear)
                    if follow_with_add:
                        T.tile.add(dst, dst, dst)
                    T.copy(dst, b)

    return main


def _source(func, target="auto", platform="A3"):
    config = process_default_pass_config(target, PASS_CONFIGS)
    with tvm.transform.PassContext(opt_level=3, config=config):
        return tilelang.lower(func, target=target, platform=platform).kernel_source


def _sum_after_mul_kernel():
    @T.prim_func
    def main(
        a: T.Tensor((8, 128), "float32"),  # type: ignore
        b: T.Tensor((8,), "float32"),  # type: ignore
    ):
        with T.Kernel(8, is_npu=True) as (cid, vid):
            src = T.alloc_ub((1, 128), "float32")
            squared = T.alloc_ub((1, 128), "float32")
            total = T.alloc_ub((1,), "float32")
            with T.Scope("V"):
                if vid == 0:
                    T.copy(a[cid, 0], src)
                    # Keep DMA ownership explicit; BiSheng must order the V-only chain below.
                    T.set_flag("MTE2", "V", 0)
                    T.wait_flag("MTE2", "V", 0)
                    T.tile.mul(squared, src, src)
                    T.reduce_sum(squared, total, dim=-1)
                    T.tile.add(total, total, T.float32(1.0))
                    T.set_flag("V", "MTE3", 0)
                    T.wait_flag("V", "MTE3", 0)
                    T.copy(total, b[cid])

    return main


def test_fp32_row_reduce_preserves_bisheng_auto_sync():
    """Low-level compiler regression: DMA is explicit, Vector ordering is automatic."""
    config = {
        **PASS_CONFIGS,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False,
    }
    compiled = tilelang.compile(
        _sum_after_mul_kernel(),
        out_idx=[1],
        target="ascendc",
        pass_configs=config,
        compile_flags=["-O2", "--cce-auto-sync=on"],
    )
    generator = torch.Generator().manual_seed(20260907)
    host = torch.randn((8, 128), generator=generator)
    result = compiled(host.npu())
    expected = host.square().sum(dim=1) + 1.0
    torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_fp32_row_frontend_uses_backing_shape_and_checks_dtype():
    dst = tvm.tir.decl_buffer((2,), "float32", scope="shared.ub")
    src = tvm.tir.decl_buffer((4, 24), "float32", scope="shared.ub")
    region = tvm.tir.BufferRegion(
        src,
        [tvm.ir.Range.from_min_extent(1, 2), tvm.ir.Range.from_min_extent(8, 9)],
    )
    call = T.reduce_max(region, dst)
    assert call.args[0].value == "reduce_max<float, 2, 9, -1>"
    assert int(call.args[-1]) == 24
    assert int(call.args[2].args[2]) == 32

    row = tvm.tir.decl_buffer((16,), "float32", scope="shared.ub")
    half_scalar = tvm.tir.decl_buffer((1,), "float16", scope="shared.ub")
    with pytest.raises(TypeError, match="dtypes must match"):
        T.reduce_sum(row, half_scalar)


@pytest.mark.parametrize("logical_width", [10, 9])
def test_fp32_row_reduce_rejects_unaligned_multirow_pitch(logical_width):
    func = _kernel(2, logical_width, 10, "float32", "sum", True)
    with pytest.raises(tvm.error.InternalError, match="physical row must be 32-byte aligned when M > 1"):
        _source(func, "ascendc")


def test_fp32_row_reduce_allows_unaligned_single_row():
    func = _kernel(1, 10, 10, "float32", "sum", True)
    source = _source(func, "ascendc")
    assert ", 1, 10, -1, 10, true>(" in source
    compiled = tilelang.compile(func, out_idx=[1], target="ascendc", pass_configs=PASS_CONFIGS)
    host = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    torch.testing.assert_close(compiled(host.npu()).cpu(), host.sum(dim=1), rtol=0, atol=0)


def test_fp32_reduce2d_v2_runtime_smoke():
    """One launch protects M/N tails, poisoned padding, and destination merge."""
    m, n, physical_row = 31, 95, 96
    compiled = tilelang.compile(
        _kernel(m, n, physical_row, "float32", "max", False),
        out_idx=[1],
        target="ascendc",
        pass_configs=PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off", "-O3"],
    )
    generator = torch.Generator().manual_seed(20260823)
    host = torch.full((m, physical_row), 1.0e20, dtype=torch.float32)
    logical = 8.0 * torch.rand((m, n), generator=generator) - 4.0
    host[:, :n] = logical
    expected = torch.maximum(logical.max(dim=1).values, torch.tensor(2.0))
    result = compiled(host.npu())
    torch.npu.synchronize()
    if isinstance(result, (tuple, list)):
        result = result[0]
    torch.testing.assert_close(result.cpu(), expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "m,n,physical_row,scratch_bytes",
    [(247, 63, 184, 8928), (1, 16384, 16384, 9312), (1, 8, 8, 32)],
)
def test_fp32_row_reduce_sizes_used_intermediate_slots(m, n, physical_row, scratch_bytes):
    """Protect one-slot, unequal two-slot, and leaf-only scratch layouts."""
    compiled = tilelang.compile(
        _kernel(m, n, physical_row, "float32", "sum", True),
        out_idx=[1],
        target="ascendc",
        pass_configs=PASS_CONFIGS,
    )
    assert f"tmp_ub = ascend_ub.GetWithOffset<uint8_t>({scratch_bytes}," in compiled.get_kernel_source()
    host = torch.arange(m * physical_row, dtype=torch.float32).reshape(m, physical_row) % 29 - 14
    host[:, n:] = 1.0e6
    expected = host[:, :n].sum(dim=1)
    torch.testing.assert_close(compiled(host.npu()).cpu(), expected, rtol=0, atol=0)


def test_codegen_selects_v2_kinds_and_preserves_other_backends():
    for kind, enum_name in (("sum", "kSum"), ("max", "kMax"), ("min", "kMin")):
        fp32 = _source(
            _kernel(64, 32, 32, "float32", kind, True, follow_with_add=kind == "max"),
            "ascendc",
        )
        assert "tl::ascend::reduce_2d<float" in fp32
        assert f"reduce2d_v2::Reduce2DKind::{enum_name}" in fp32
        if kind == "max":
            assert fp32.count("AscendC::SetMaskNorm();") == 1
            assert fp32.count("AscendC::SetVectorMask<uint8_t>") == 1

    fp16 = _source(_kernel(8, 16, 16, "float16", "sum", True), "ascendc")
    assert "tl::ascend::reduce_sum<float" in fp16
    assert fp16.count("AscendC::Cast(") == 2
    assert "tl::ascend::reduce_sum_half<" not in fp16
    assert "tl::ascend::reduce_2d<" not in fp16

    pto = _source(_kernel(8, 32, 32, "float32", "max", True), "pto")
    assert "TROWMAX(" in pto
    assert "tl::ascend::reduce_2d" not in pto

    default = _source(_kernel(8, 32, 32, "float32", "max", True))
    assert "tl::ascend::reduce_2d<float" in default

    a5 = _source(_kernel(8, 32, 32, "float32", "sum", True), "ascendc", platform="A5")
    assert "tl::ascend::reduce_sum<float" in a5
    assert "tmp_ub.ReinterpretCast<float>" not in a5
    narrow_a5 = _source(_kernel(8, 9, 16, "float32", "max", True), "ascendc", platform="A5")
    assert "tl::ascend::reduce_max_narrow<float>" in narrow_a5
    assert "tmp_ub" not in narrow_a5


def test_fp32_reduce2d_v2_uses_barriers_on_a2():
    func = _kernel(8, 32, 32, "float32", "max", True)
    a2 = _source(func, "ascendc", platform="A2")
    a3 = _source(func, "ascendc", platform="A3")
    assert "tl::ascend::reduce_2d<float" in a2
    assert ", -1, 32, false>(" in a2
    assert ", -1, 32, true>(" in a3
