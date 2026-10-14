"""Ascend NPU regression tests for explicit vectorized loops."""

import tilelang
import tilelang.language as T
import torch


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def nonzero_min_vectorized_kernel(size, start, stop, dtype="float32"):

    @T.prim_func
    def main(A: T.Tensor((size,), dtype), B: T.Tensor((size,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((size,), dtype)
            b_ub = T.alloc_ub((size,), dtype)
            with T.Scope("V"):
                T.copy(A, a_ub)
                T.tile.fill(b_ub, 0.0)
                for i in T.vectorized(start, stop):
                    b_ub[i] = a_ub[i] + 1.0
                T.copy(b_ub, B)

    return main


def test_vectorized_nonzero_min():
    size, start, stop = 128, 32, 96
    func = nonzero_min_vectorized_kernel(size, start, stop)
    src = torch.arange(size, dtype=torch.float32).npu()

    torch.npu.synchronize()
    out = func(src)
    ref = torch.zeros_like(src)
    ref[start:stop] = src[start:stop] + 1.0
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_vectorized_nonzero_min()
