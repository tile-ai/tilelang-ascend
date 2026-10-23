import pytest
import tilelang
import tilelang.language as T
import torch

"""
GM -> UB 2D sub-region copy dst-stride regression.

Feature under test
------------------
``T.copy(X[0:H, 0:W], buf[PT:PT+H, PL:PL+W])`` copies an HxW source tile into
a sub-region of a wider UB buffer ``buf`` whose physical row width is SWp > W.
The DataCopyPad dst block stride must advance each destination row by the
buffer's PHYSICAL row width (SWp), not by the copy width W.

Bug pinned here: the codegen emitted the copy template column dim ``dstN``
from the copy-region extent (W) instead of the destination buffer's physical
row width (SWp).  The helper then computes the dst inter-block stride as
``(dstN - maskShapeN) * sizeof(T) / 32 = 0``, so every row after the first is
written contiguously (row width W) instead of strided by SWp -- the tile lands
bit-shifted across rows.  Verified bit-exactly against a torch golden.

These cases execute on real NPU hardware (``.npu()``).
"""

VEC_MIN_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


def _torch_dtype(dtype):
    return {"float": torch.float32, "float16": torch.float16}[dtype]


def subregion_2d(H, W, PT, PL, SWp, Hp, dtype):
    """Copy X[0:H,0:W] into buf[PT:PT+H, PL:PL+W]; store the whole buffer back.

    PL is chosen 32-Byte aligned so the dst base offset is legal; the buffer
    row width SWp > W so the dst rows are strided (the bug regime)."""

    @T.prim_func
    def main(X: T.Tensor((H, W), dtype), Y: T.Tensor((Hp, SWp), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            buf = T.alloc_ub((Hp, SWp), dtype)
            T.tile.clear(buf)
            T.copy(X[0:H, 0:W], buf[PT : PT + H, PL : PL + W])
            T.copy(buf, Y[0:Hp, 0:SWp])

    return main


def run_test_subregion_2d(H, W, PT, PL, SWp, Hp, dtype):
    torch.manual_seed(0)
    func = subregion_2d(H, W, PT, PL, SWp, Hp, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_MIN_CONFIGS, target="ascendc")
    td = _torch_dtype(dtype)
    x = torch.randn(H, W, dtype=td).npu()
    torch.npu.synchronize()
    y = func(x).cpu()
    x = x.cpu()
    # Data region must land at [PT:PT+H, PL:PL+W] bit-exactly.
    torch.testing.assert_close(y[PT : PT + H, PL : PL + W], x, rtol=0, atol=0)
    # Everything outside the data region must stay 0 (cleared).
    assert torch.all(y[0:PT, :] == 0), "top pad rows not zero"
    assert torch.all(y[PT : PT + H, 0:PL] == 0), "left pad cols not zero"
    if SWp > PL + W:
        assert torch.all(y[PT : PT + H, PL + W : SWp] == 0), "right pad not zero"


@pytest.mark.parametrize(
    "H,W,PT,PL,SWp,Hp,dtype",
    [
        # fp32: PL=8 (32 B), W=61, physical row SWp=80 > W -> strided dst rows.
        (61, 61, 1, 8, 80, 63, "float"),
        # fp16: PL=16 (32 B), W=31, physical row SWp=64 > W.
        (16, 31, 2, 16, 64, 20, "float16"),
    ],
)
def test_subregion_2d(H, W, PT, PL, SWp, Hp, dtype):
    run_test_subregion_2d(H, W, PT, PL, SWp, Hp, dtype)


# =============================================================================
# Group 2 - UB -> GM 2D sub-region store (symmetric dst-stride fix)
# Store a sub-region of a wider UB buffer (physical row width SWp > W) out to
# GM.  The src inter-block stride must use the buffer's physical row width.
# =============================================================================
def subregion_2d_store(H, W, PT, PL, SWp, Hp, dtype):
    """Fill buf[PT:PT+H, PL:PL+W] = X, store buf[PT:PT+H, PL:PL+W] -> Y[0:H,0:W].

    The UB->GM store reads a sub-region (col offset PL, width W) of the wider
    buffer; the src rows must advance by SWp, not W."""

    @T.prim_func
    def main(X: T.Tensor((H, W), dtype), Y: T.Tensor((H, W), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            buf = T.alloc_ub((Hp, SWp), dtype)
            T.tile.clear(buf)
            # Fill the sub-region via GM->UB (already fixed) so the store side
            # is exercised in isolation.
            T.copy(X[0:H, 0:W], buf[PT : PT + H, PL : PL + W])
            T.copy(buf[PT : PT + H, PL : PL + W], Y[0:H, 0:W])

    return main


def run_test_subregion_2d_store(H, W, PT, PL, SWp, Hp, dtype):
    torch.manual_seed(0)
    func = subregion_2d_store(H, W, PT, PL, SWp, Hp, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_MIN_CONFIGS, target="ascendc")
    td = _torch_dtype(dtype)
    x = torch.randn(H, W, dtype=td).npu()
    torch.npu.synchronize()
    y = func(x).cpu()
    torch.testing.assert_close(y, x.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "H,W,PT,PL,SWp,Hp,dtype",
    [
        (61, 61, 1, 8, 80, 63, "float"),
        (16, 31, 2, 16, 64, 20, "float16"),
    ],
)
def test_subregion_2d_store(H, W, PT, PL, SWp, Hp, dtype):
    run_test_subregion_2d_store(H, W, PT, PL, SWp, Hp, dtype)
