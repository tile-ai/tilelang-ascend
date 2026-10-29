"""Unary vector regions must use accessed shapes, preserving untouched rows."""

import numpy as np
import pytest
import tilelang
import tilelang.language as T

pytestmark = [pytest.mark.op("vector"), pytest.mark.mode("Expert")]


@T.macro
def wait_for(src, dst):
    with T.rs(src):
        T.set_flag(dst, 0)
    with T.rs(dst):
        T.wait_flag(src, 0)


def unary_region(case):
    src_rows = (
        24 if case == "invalid" else (16 if case in ("src_slice", "both_slices") else 8)
    )
    dst_rows = 24 if case in ("dst_slice", "both_slices", "invalid") else 8
    src_begin = 4 if src_rows > 8 else 0
    dst_begin = 8 if dst_rows == 24 else 0
    dst_extent = 16 if case == "invalid" else 8

    @T.prim_func
    def main(
        A: T.Tensor((src_rows, 8), "float32"),
        Seed: T.Tensor((dst_rows, 8), "float32"),
        B: T.Tensor((dst_rows, 8), "float32"),
    ):
        with T.Kernel(1, is_npu=True):
            a = T.alloc_ub((src_rows, 8), "float32")
            b = T.alloc_ub((dst_rows, 8), "float32")
            T.copy(A, a)
            T.copy(Seed, b)
            wait_for("PIPE_MTE2", "PIPE_V")
            if case == "abs":
                T.vabs(
                    a[src_begin : src_begin + 8, :],
                    b[dst_begin : dst_begin + dst_extent, :],
                )
            else:
                T.vexp(
                    a[src_begin : src_begin + 8, :],
                    b[dst_begin : dst_begin + dst_extent, :],
                )
            wait_for("PIPE_V", "PIPE_MTE3")
            T.copy(b, B)
            wait_for("PIPE_MTE3", "PIPE_MTE2")

    return main


@pytest.mark.parametrize(
    "case", ["dst_slice", "src_slice", "both_slices", "full", "abs", "invalid"]
)
def test_unary_regions(case):
    import torch
    import torch_npu  # noqa: F401

    flags = {
        "npuir.enable_auto_multi_buffer": False,
        "npuir.disable_hivm_auto_inject_sync": True,
        "tl.enable_plan_and_update_buffer_allocation": False,
    }
    if case == "invalid":
        # Equal allocation sizes do not make unequal accessed extents legal.
        with pytest.raises(Exception, match="Check failed|shape|broadcast"):
            tilelang.compile(unary_region(case), target="npuir", pass_configs=flags)
        return
    fn = tilelang.compile(unary_region(case), target="npuir", pass_configs=flags)
    src_rows = 16 if case in ("src_slice", "both_slices") else 8
    dst_rows = 24 if case in ("dst_slice", "both_slices") else 8
    src_begin = 4 if src_rows > 8 else 0
    dst_begin = 8 if dst_rows == 24 else 0
    a = np.linspace(-2, 2, src_rows * 8, dtype=np.float32).reshape(src_rows, 8)
    seed = np.full((dst_rows, 8), -123, dtype=np.float32)
    expected = seed.copy()
    values = a[src_begin : src_begin + 8]
    expected[dst_begin : dst_begin + 8] = (
        np.abs(values) if case == "abs" else np.exp(values)
    )
    da, ds = torch.from_numpy(a).npu(), torch.from_numpy(seed).npu()
    db = torch.empty_like(ds)
    fn(da, ds, db)
    torch.npu.synchronize()
    actual = db.cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert np.array_equal(actual[:dst_begin], seed[:dst_begin])
    assert np.array_equal(actual[dst_begin + 8 :], seed[dst_begin + 8 :])
