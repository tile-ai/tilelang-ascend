"""Kernel-driven cube decomposition: a K-accumulating mma with a fused fixpipe.

The gemm helpers own the whole L1->L0->mma->fixpipe sequence internally, which
is fine when the operands are ready up front. A kernel that keeps its own KV
ring, or splits the contraction into chunks that arrive at different times,
needs to drive that sequence itself: load each L0 tile, issue the mma, and copy
the accumulated result out when the last tile lands.

Doing that from the front end needs four runtime arguments, all of which this
test exercises together:

  * ``T.copy(L1->L0A, real_k=)``  -- the fractal's K extent, so a full-width L0
    buffer is loaded as the ``[M, real_k]`` fractal the mma will read.
  * ``T.copy(L1->L0B, real_n=)``  -- the same for L0B's other axis. L0B's
    fractal derives its K-block stride from the column count, so a full-width
    load feeding a shorter mma addresses the wrong K-blocks.
  * ``T.mma(k_actual=, n_actual=, unit_flag=)`` -- contract only part of the
    operands, and mark whether the result stays in L0C (0b10) or is released to
    a paired fixpipe (0b11).
  * ``T.copy(L0C->GM, unit_flag=0b11)`` -- the paired fixpipe.

The kernel computes A @ B^T over K=256 as two hand-walked 128-wide tiles that
accumulate into one L0C slot, with only the second one flushing. That covers
the accumulating path (``cmatrixInitVal == false``), which is also what reads
``cmatrixSource``; leaving that field uninitialised hangs the cube once the mma
carries a unitFlag.

``n_act`` is a runtime value so ``real_n`` / ``n_actual`` take their runtime
path. A single tile would pass with a wrong K-block offset (its offset is 0),
so both tiles are required to make the test meaningful.
"""

import pytest

import torch

import tilelang
import tilelang.language as T

M, N, K = 64, 128, 256
KL0 = 128  # kL0 tile width: K = 2 tiles

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def kernel_driven_mma(dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Tensor([M, K], dtype),
        B: T.Tensor([N, K], dtype),  # K^T layout: the gemm is A @ B^T
        nact: T.Tensor([1], "int32"),  # runtime output-column count
        C: T.Tensor([M, N], accum_dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_l1 = T.alloc_L1([M, K], dtype)
            b_l1 = T.alloc_L1([N, K], dtype)
            a_l0 = T.alloc_L0A([2, M, KL0], dtype)
            b_l0 = T.alloc_L0B([2, N, KL0], dtype)
            c_l0 = T.alloc_L0C([1, M, N], accum_dtype)
            with T.Scope("C"):
                n_act = nact[0]
                T.barrier_all()
                T.copy(A, a_l1)
                T.copy(B, b_l1)
                T.barrier_all()
                # kL0 tile 0: initialise the L0C slot, hold it (0b10).
                T.copy(a_l1[:, 0:KL0], a_l0[0, :, :])
                T.copy(
                    b_l1[:, 0:KL0],
                    b_l0[0, :, :],
                    transpose=True,
                    real_k=KL0,
                    real_n=n_act,
                )
                T.barrier_all()
                T.mma(
                    a_l0[0, :, :],
                    b_l0[0, :, :],
                    c_l0[0, :, :],
                    init=True,
                    k_actual=KL0,
                    n_actual=n_act,
                    unit_flag=0b10,
                )
                T.barrier_all()
                # kL0 tile 1: accumulate into the same slot, then flush (0b11).
                T.copy(a_l1[:, KL0 : 2 * KL0], a_l0[1, :, :])
                T.copy(
                    b_l1[:, KL0 : 2 * KL0],
                    b_l0[1, :, :],
                    transpose=True,
                    real_k=KL0,
                    real_n=n_act,
                )
                T.barrier_all()
                T.mma(
                    a_l0[1, :, :],
                    b_l0[1, :, :],
                    c_l0[0, :, :],
                    init=False,
                    k_actual=KL0,
                    n_actual=n_act,
                    unit_flag=0b11,
                )
                T.barrier_all()
                # The fixpipe paired with the flushing mma. Its column count must
                # equal the mma's n, or it waits on L0C columns no mma marked
                # ready.
                T.copy(c_l0[0, :, :], C[:, 0:n_act], unit_flag=0b11)
                T.barrier_all()

    return main


@pytest.mark.parametrize("n_act", [N, 96, 32])
def test_kernel_driven_mma(n_act):
    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    func = tilelang.compile(
        kernel_driven_mma(),
        out_idx=[-1],
        target="ascendc",
        pass_configs=pass_configs,
    )

    a = (torch.randn(M, K) * 0.1).half()
    b = (torch.randn(N, K) * 0.1).half()
    nact = torch.tensor([n_act], dtype=torch.int32)

    out = func(a.npu(), b.npu(), nact.npu())

    ref = (a.float() @ b.float().T)[:, :n_act]
    torch.testing.assert_close(out.cpu()[:, :n_act], ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
