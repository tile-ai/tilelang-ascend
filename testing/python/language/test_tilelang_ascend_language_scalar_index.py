"""Scalar element access on a multi-dimensional buffer.

A scalar read/write like ``table[b, i]`` has to flatten the index across every
dimension. Emitting only the innermost index makes every row alias row 0, which
stays silent whenever the leading index happens to be 0 -- a single-batch test
passes while a multi-batch one returns the first batch's data.

The kernel varies both indices on a [B, N] buffer, so a dropped leading index
shows up as rows 1.. holding row 0's values. It exercises the scalar load and
the scalar store together.

The pto codegen carries the same defect, but fixing it there needs a separate
look at how its global-buffer accesses are addressed, so this covers ascendc
only and the pto side is tracked as its own issue.
"""

import pytest

import torch

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


def scalar_index_2d(B, N, dtype="int32"):
    @T.prim_func
    def main(
        table: T.Tensor([B, N], dtype),
        out: T.Tensor([B, N], dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            buf = T.alloc_ub([B, N], dtype)
            if vid == 0:
                for b in range(B):
                    for i in range(N):
                        # 2-D scalar load AND 2-D scalar store
                        buf[b, i] = table[b, i] * 2
                T.copy(buf, out)

    return main


@pytest.mark.parametrize("target", ["ascendc"])
@pytest.mark.parametrize("B,N", [(4, 8), (3, 16), (2, 32)])
def test_scalar_index_2d(B, N, target):
    tilelang.cache.clear_cache()
    func = tilelang.compile(
        scalar_index_2d(B, N),
        out_idx=[1],
        target=target,
        pass_configs=pass_configs,
    )
    table = torch.arange(B * N, dtype=torch.int32).reshape(B, N)
    out = func(table.npu())
    torch.testing.assert_close(out.cpu(), table * 2)


if __name__ == "__main__":
    pytest.main([__file__])
