import pytest
import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout
import torch

"""
gemm_v0 runtime n_actual (variable output-column count) correctness suite.

Feature under test
------------------
``T.gemm_v0(..., transpose_B=True, n_actual=win)`` computes only the first
``win`` output columns of the M x N result -- the "QK over the actual window
length" pattern -- instead of the full compile-time ``N``.  ``n_actual``
defaults to ``N``, so every existing caller is byte-identical.  This is the dual
of the runtime ``K`` already threaded through ``mma`` (``mma<M,N>(A,B,C,init,K,
n_actual)``): the template ``M/N/K`` and the physical L0B/L0C layout stay
compile-time; only *how many output columns are actually computed* becomes a
runtime value.

Why an ordinary gemm_v0 cannot express this
-------------------------------------------
Before this change the output-column count is the compile-time template ``N``:
``copy_l1_to_l0b`` loads ``N`` columns and ``mma`` sets ``mmadParams.n = N``.
There is no way to compute only a runtime ``win`` columns; a caller had to
compute the full ``N`` and mask the ``[win:N]`` tail downstream.  ``n_actual``
adds that runtime column count, mirroring runtime ``K``.

Scenarios (ascendc target, transpose_B GEMM ``S = Q @ K^T``)
------------------------------------------------------------
  * ``n_actual = None`` -> defaults to ``N``: full ``S == Q @ K^T`` (compat:
    the default path is byte-identical to before this change).
  * ``n_actual = win < N``: only ``S[:, :win]`` is computed by ``mma``; the
    ``[win:N]`` columns are never marked ready (left at the L0C init value,
    "masked downstream" by the caller), so only the computed band is asserted.

``n_actual`` must be a multiple of 16 (the fractal column granularity), matching
how the operator rounds the runtime window length up to 16 before passing it in.

NOTE: executes on real NPU hardware (``.npu()``); ascendc target only (the PTO
backend has its own gemm_v0 codegen that ignores this trailing arg).
"""

TARGET = "ascendc"

DEV_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before the session."""
    tilelang.cache.clear_cache()
    yield


def _torch_dtype(dtype):
    return {"float": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def qk_gemm(block_M, block_N, dim, dtype, accum_dtype, n_actual):
    """transpose_B QK gemm ``S = Q @ K^T`` computing only ``n_actual`` output
    columns.  ``n_actual=None`` -> full ``N`` (default).  Mirrors
    gm_to_l1::full_copy_annotated with the ``n_actual`` arg added."""

    @T.prim_func
    def main(
        Q: T.Tensor([1, block_M, dim], dtype),  # type: ignore
        K: T.Tensor([1, block_N, dim], dtype),  # type: ignore
        S: T.Tensor([1, block_M, block_N], accum_dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                }
            )
            T.copy(Q[0, :, :], q_l1[:, :])
            T.copy(K[0, :, :], k_l1[:, :])
            T.gemm_v0(q_l1, k_l1, l0c, transpose_B=True, init=True, n_actual=n_actual)
            T.copy(l0c, S[0, :, :])

    return main


def run_qk(block_M, block_N, dim, dtype, accum_dtype, n_actual):
    torch.manual_seed(0)
    win = block_N if n_actual is None else n_actual
    func = qk_gemm(block_M, block_N, dim, dtype, accum_dtype, n_actual)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=DEV_CONFIGS, target=TARGET)
    td = _torch_dtype(dtype)
    q = torch.randn(1, block_M, dim, dtype=td).npu()
    k = torch.randn(1, block_N, dim, dtype=td).npu()
    torch.npu.synchronize()
    s = func(q, k)
    ref = torch.einsum("bqd,bkd->bqk", q.float(), k.float())
    # Only the first `win` output columns are computed; [win:N] are left at the
    # L0C init value (masked downstream), so assert only the computed band.
    torch.testing.assert_close(s[:, :, :win], ref[:, :, :win], rtol=1e-2, atol=1e-2)


# (block_M, block_N, dim, n_actual) -- n_actual is a multiple of 16 (or None).
configs = [
    (64, 128, 128, None),  # default n_actual = N: full result (compat)
    (64, 128, 128, 64),  # window 64 < 128
    (64, 128, 128, 112),  # window 112 (7*16), a non-trivial partial < 128
    (64, 256, 128, 128),  # N=256, window 128 < 256
]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("block_M,block_N,dim,n_actual", configs)
def test_qk_n_actual(block_M, block_N, dim, n_actual, dtype):
    run_qk(block_M, block_N, dim, dtype, "float", n_actual)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
