import pytest
import tilelang
import tilelang.language as T
import torch

"""
Regression test for the non-PTO row-broadcast primitives on the AscendC backend:
``row_expand_div`` / ``row_expand_sub`` / ``row_expand_mul_nd``
(src/tl_templates/ascend/common.h).

Feature under test
------------------
A row-broadcast elementwise op computes ``dst[i, j] = src0[i, j] OP src1[i]`` --
every column of row ``i`` is combined with that row's single ``src1`` scalar. The
faithful equivalent of the Ascend C reference's ``RowDivs`` / ``RowMuls``: ``Brcb``
the ``[M, 1]`` column into an ``[M, blk]`` tile, then ``Div`` / ``Sub`` / ``Mul``
with ``src1BlkStride=0`` / ``src1RepStride=1``. This avoids materialising an
``[M, N]`` broadcast buffer (which overflows the vector UB once the cube/vector
pipeline stops the planner from reusing buffers).

The PTO backend already has a ``row_expand_mul`` (``TROWEXPANDMUL``); the non-PTO
(``tilelang_ascend``) codegen had no div/sub template and ``LOG(FATAL)``'d on the
mul path. These primitives fill that gap:
  * ``row_expand_div`` / ``row_expand_sub`` -- new non-PTO templates + ops.
  * ``row_expand_mul_nd`` -- a DISTINCT op from the PTO ``row_expand_mul``, so the
    PTO binding/codegen and its examples/HISA callers stay byte-identical.

How this test exercises it
--------------------------
A GM->UB load feeds a single row-expand op whose result is stored UB->GM and
checked against ``a OP b[:, None]`` (torch broadcast). ``M`` / ``N`` satisfy the
templates' asserts (``N % (256/sizeof(T)) == 0`` and ``N/MASK <= M``).

This targets the ascendc backend only (the fix is in the non-PTO templates /
codegen; the PTO path is untouched).
"""

TARGET = "ascendc"

VEC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear the tilelang cache before the session (a stale kernel could mask a
    codegen regression -- a rebuilt kernel could otherwise return a cached one)."""
    tilelang.cache.clear_cache()
    yield


def _torch_dtype(dtype):
    return {
        "float": torch.float32,
        "float16": torch.float16,
    }[dtype]


def row_expand_kernel(M, N, op, dtype):
    """Load A [M,N] and per-row scalars B [M] into UB, apply a row-broadcast
    div/sub/mul_nd (dst[i,j] = A[i,j] OP B[i]), store the result. ``tmp`` is the
    [M, 32/sizeof(T)] scratch the templates use for the Brcb."""
    blk = 32 // (4 if dtype == "float" else 2)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M,), dtype),  # type: ignore  per-row scalar
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((M,), dtype)
            tmp = T.alloc_ub((M, blk), dtype)
            c_ub = T.alloc_ub((M, N), dtype)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            if op == "div":
                T.tile.row_expand_div(c_ub, a_ub, b_ub, tmp)
            elif op == "sub":
                T.tile.row_expand_sub(c_ub, a_ub, b_ub, tmp)
            else:  # mul_nd
                T.tile.row_expand_mul_nd(c_ub, a_ub, b_ub, tmp)
            T.copy(c_ub, C)

    return main


def run_test_row_expand(M, N, op, dtype):
    torch.manual_seed(0)
    func = row_expand_kernel(M, N, op, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=TARGET)
    td = _torch_dtype(dtype)
    a = torch.randn(M, N, dtype=td).npu()
    # Per-row scalars kept away from 0 so the div case is well-conditioned.
    b = (torch.randn(M, dtype=td) + 2.0).npu()
    torch.npu.synchronize()
    c = func(a, b)
    bcol = b[:, None].float()
    if op == "div":
        ref = a.float() / bcol
    elif op == "sub":
        ref = a.float() - bcol
    else:  # mul_nd
        ref = a.float() * bcol
    rtol, atol = 1e-2, 1e-2
    torch.testing.assert_close(c.float().cpu(), ref.to(c.dtype).float().cpu(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("op", ["div", "sub", "mul_nd"])
def test_row_expand(op, dtype):
    # M=64, N=256: N % (256/sizeof) == 0 (fp32 64, fp16 128) and N/MASK <= M.
    run_test_row_expand(64, 256, op, dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
