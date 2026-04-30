import pytest
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

pytestmark = [
    pytest.mark.op("gqa_decode_parallel"),
    pytest.mark.mode("Developer"),
    pytest.mark.smoke,
]

CASES = [
    pytest.param(
        4, 8, 5, "tileops/kernels/flash_decode/gqa_decode.py:86", id="gqa_decode_l86"
    ),
    pytest.param(
        8,
        8,
        6,
        "tileops/kernels/flash_decode/mha_decode_paged.py:84",
        id="mha_decode_paged_l84",
    ),
    pytest.param(
        8,
        16,
        9,
        "tileops/kernels/flash_decode/gqa_decode_paged.py:95",
        id="gqa_decode_paged_l95",
    ),
    pytest.param(
        16,
        16,
        13,
        "tileops/kernels/flash_decode/gqa_decode.py:193",
        id="gqa_decode_l193",
    ),
]


def kernel_gqa_decode_parallel1(block_h, block_n, valid_block_n):
    @T.prim_func
    def main(
        scores: T.Tensor((block_h, block_n), "float32"),
        logsum: T.Tensor((block_h,), "float32"),
        out: T.Tensor((block_h, block_n), "float32"),
    ):
        with T.Kernel(1, is_npu=True):
            scores_buf = T.alloc_shared((block_h, block_n), "float32")
            out_buf = T.alloc_shared((block_h, block_n), "float32")
            T.copy(scores, scores_buf)
            for i, j in T.Parallel(block_h, block_n):
                scores_buf[i, j] = T.if_then_else(
                    j < valid_block_n, scores_buf[i, j], -T.float32(10000.0)
                )
            for i, j in T.Parallel(block_h, block_n):
                out_buf[i, j] = scores_buf[i, j] / logsum[i]
            T.copy(out_buf, out)

    return main


@pytest.mark.parametrize("block_h, block_n, valid_block_n, source", CASES)
def test_gqa_decode_Parallel1(block_h, block_n, valid_block_n, source):
    kernel = tilelang.compile(
        kernel_gqa_decode_parallel1(block_h, block_n, valid_block_n), target="npuir"
    )
    scores = torch.randn((block_h, block_n), dtype=torch.float32, device="npu")
    logsum = torch.rand((block_h,), dtype=torch.float32, device="npu") + 1.0
    out = torch.zeros((block_h, block_n), dtype=torch.float32, device="npu")
    masked = scores.clone()
    masked[:, valid_block_n:] = -10000.0
    ref = masked / logsum.unsqueeze(1)
    kernel(scores, logsum, out)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
