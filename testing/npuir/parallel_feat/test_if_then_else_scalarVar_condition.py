import pytest
import torch

import tilelang
import tilelang.language as T


pytestmark = [
    pytest.mark.op("parallel_if_then_else"),
    pytest.mark.mode("Developer"),
]

ACCUM_DTYPE = "float32"

CASES = [
    (32, 16),
]


def kernel_parallel_if_then_else(H_per_block, BI):

    @T.prim_func
    def parallel_if_then_else(
        out: T.Tensor((32, H_per_block, BI), ACCUM_DTYPE),
    ):
        with T.Kernel(32, is_npu=True) as (cid, _):
            acc_s = T.alloc_shared((H_per_block, BI), ACCUM_DTYPE)

            for h_i, bi_i in T.Parallel(H_per_block, BI):
                acc_s[h_i, bi_i] = T.if_then_else(
                    cid >= 1,
                    T.float32(0),
                    T.float32("-inf"),
                )

            T.copy(acc_s, out[cid, :, :])

    return parallel_if_then_else


@pytest.mark.parametrize("H_per_block, BI", CASES)
def test_parallel_if_then_else(H_per_block, BI):
    func = kernel_parallel_if_then_else(H_per_block, BI)
    kernel = tilelang.compile(func, target="npuir")

    out = torch.zeros((32, H_per_block, BI), dtype=torch.float32, device="npu")

    ref_output = torch.where(
        torch.arange(32, device="npu").view(32, 1, 1).expand(32, H_per_block, BI) >= 1,
        torch.zeros(32, H_per_block, BI, dtype=torch.float32, device="npu"),
        torch.full(
            (32, H_per_block, BI), float("-inf"), dtype=torch.float32, device="npu"
        ),
    )

    kernel(out)

    torch.testing.assert_close(out, ref_output, rtol=1e-3, atol=1e-3)
