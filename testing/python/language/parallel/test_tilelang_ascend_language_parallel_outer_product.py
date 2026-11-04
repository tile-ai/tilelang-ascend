"""Regression for 1D x 1D broadcast materialization under a 2D T.Parallel."""

import pytest
import torch

import tilelang
import tilelang.language as T

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
def outer_product(dv: int, dk: int):
    @T.prim_func
    def main(
        v: T.Tensor((dv,), "float32"),
        w: T.Tensor((dk,), "float32"),
        out: T.Tensor((1, dk), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            v_ub = T.alloc_ub((dv,), "float32")
            w_ub = T.alloc_ub((dk,), "float32")
            outer_ub = T.alloc_ub((dv, dk), "float32")
            reduced_ub = T.alloc_ub((dk,), "float32")
            T.copy(v[0], v_ub)
            T.copy(w[0], w_ub)
            for j, i in T.Parallel(dv, dk):
                outer_ub[j, i] = v_ub[j] * w_ub[i]
            T.reduce_sum(outer_ub, reduced_ub, dim=0)
            T.copy(reduced_ub, out[0, 0])

    return main


@pytest.mark.parametrize("dv,dk", [(8, 8), (16, 4), (3, 17)])
def test_1d_operands_are_not_reduced_to_scalar_broadcast(dv, dk):
    func = outer_product(dv, dk)
    v = torch.randn(dv).npu()
    w = torch.randn(dk).npu()
    torch.npu.synchronize()

    out = func(v, w)[0].cpu()
    expected = (v.cpu().unsqueeze(-1) * w.cpu().unsqueeze(0)).sum(0)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
