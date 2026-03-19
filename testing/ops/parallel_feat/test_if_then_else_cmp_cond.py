import pytest
import torch

import tilelang
import tilelang.language as T


pytestmark = [
    pytest.mark.op("parallel_if_then_else"),
    pytest.mark.mode("Developer"),
]

DTYPE = "float32"
ACCUM_DTYPE = "float32"

CASES = [
    (32, 64),
]


def kernel_dv_conditional(block_S, block_DV):
    @T.prim_func
    def dv_conditional(
        dv: T.Tensor((block_S, block_DV), ACCUM_DTYPE),
        G_fragment: T.Tensor((block_S,), ACCUM_DTYPE),
        G_fragment_post: T.Tensor((block_S,), ACCUM_DTYPE),
        G_last: T.Tensor((1,), ACCUM_DTYPE),
        out: T.Tensor((block_S, block_DV), ACCUM_DTYPE),
    ):
        with T.Kernel(1, is_npu=True) as (bx, _):
            dv_fragment = T.alloc_shared((block_S, block_DV), ACCUM_DTYPE)
            G_fragment_buf = T.alloc_shared((block_S,), ACCUM_DTYPE)
            G_fragment_post_buf = T.alloc_shared((block_S,), ACCUM_DTYPE)
            out_buf = T.alloc_shared((block_S, block_DV), ACCUM_DTYPE)

            T.copy(dv, dv_fragment)
            T.copy(G_fragment, G_fragment_buf)
            T.copy(G_fragment_post, G_fragment_post_buf)

            for i_s2, i_v in T.Parallel(block_S, block_DV):
                out_buf[i_s2, i_v] = T.if_then_else(
                    G_last[0] - G_fragment_buf[i_s2] <= T.float32(0),
                    dv_fragment[i_s2, i_v] * G_fragment_post_buf[i_s2],
                    T.float32(0),
                )

            T.copy(out_buf, out)

    return dv_conditional


@pytest.mark.parametrize("block_S, block_DV", CASES)
def test_parallel_dv_conditional(block_S, block_DV):
    func = kernel_dv_conditional(block_S, block_DV)
    kernel = tilelang.compile(func, target="npuir")

    dv = torch.randn((block_S, block_DV), dtype=torch.float32, device="npu")
    G_fragment = torch.randn((block_S,), dtype=torch.float32, device="npu")
    G_fragment_post = torch.randn((block_S,), dtype=torch.float32, device="npu")
    G_last = torch.randn((1,), dtype=torch.float32, device="npu")
    out = torch.zeros((block_S, block_DV), dtype=torch.float32, device="npu")

    cond = (G_last[0] - G_fragment) <= 0
    ref_output = torch.where(
        cond.unsqueeze(1),
        dv * G_fragment_post.unsqueeze(1),
        torch.zeros_like(dv),
    )

    kernel(dv, G_fragment, G_fragment_post, G_last, out)

    torch.testing.assert_close(out, ref_output, rtol=1e-3, atol=1e-3)
