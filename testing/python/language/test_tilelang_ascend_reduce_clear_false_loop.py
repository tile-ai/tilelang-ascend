"""Coverage for loop-carried T.reduce_sum(clear=False) inside T.Pipelined.

Issue #1924 reported wrong results, but its reproducer compared the (M,)
output against ``x.sum(0)`` of shape (M, N).  Broadcasting that difference
reports ~3e1 for a *correct* kernel.  With the expected reference
``x.sum(dim=(0, 2))`` the accumulation is exact in serial, unrolled and
pipelined forms.  This test locks in the in-loop merge path, which the
existing reduce tests do not cover.
"""

import pytest
import torch
import tilelang
import tilelang.language as T

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

TARGET = "ascendc"
S_, M, N = 2, 64, 64


@pytest.fixture(scope="module", autouse=True)
def _disable_cache():
    tilelang.disable_cache()
    yield
    tilelang.enable_cache()


def _kernel(mode, num_stages=2):
    @T.prim_func
    def main(
        X: T.Tensor((S_, M, N), "float32"),
        out: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            x_ub = T.alloc_shared([M, N], "float32")
            acc = T.alloc_shared([M], "float32")
            T.tile.fill(acc, 0.0)
            if mode == "serial":
                for i in T.serial(S_):
                    T.copy(X[i, 0:M, 0:N], x_ub)
                    T.reduce_sum(x_ub, acc, dim=1, clear=False)
            elif mode == "unroll":
                for i in T.unroll(S_):
                    T.copy(X[i, 0:M, 0:N], x_ub)
                    T.reduce_sum(x_ub, acc, dim=1, clear=False)
            else:
                for i in T.Pipelined(S_, num_stages=num_stages):
                    T.copy(X[i, 0:M, 0:N], x_ub)
                    T.reduce_sum(x_ub, acc, dim=1, clear=False)
            T.copy(acc, out)

    return main


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="reduce accumulation requires an Ascend NPU runtime",
)
@pytest.mark.parametrize(
    "mode,num_stages",
    [("serial", 1), ("unroll", 1), ("pipeline", 1), ("pipeline", 2)],
)
def test_reduce_sum_clear_false_loop_carried(mode, num_stages):
    kernel = tilelang.compile(_kernel(mode, num_stages), pass_configs=PASS_CONFIGS, target=TARGET)
    x = torch.randn(S_, M, N, dtype=torch.float32, device="npu")
    out = torch.zeros(M, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    kernel(x, out)
    torch.npu.synchronize()
    ref = x.sum(dim=(0, 2))
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
