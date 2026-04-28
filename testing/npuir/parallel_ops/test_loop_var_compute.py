import pytest
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

pytestmark = [pytest.mark.op("gqa_decode_parallel"), pytest.mark.mode("Developer")]

SHAPE_CASES = [
    pytest.param(4, 8, 5, id="4x8_thresh5"),
    pytest.param(8, 16, 9, id="8x16_thresh9"),
]


# ============================================================
# Case 1: Linear combination of loop variables as condition
# Condition: i * block_n + j < threshold (flattened 2D index threshold)
# ============================================================
def kernel_linear_combo_condition(block_h, block_n, valid_block_n):
    threshold = block_h * valid_block_n

    @T.prim_func
    def main(
        scores: T.Tensor((block_h, block_n), "float32"),
        logsum: T.Tensor((block_h,), "float32"),
        out: T.Tensor((block_h, block_n), "float32"),
    ):
        with T.Kernel(1, is_npu=True):
            scores_buf = T.alloc_shared((block_h, block_n), "float32")
            masked_buf = T.alloc_shared((block_h, block_n), "float32")
            out_buf = T.alloc_shared((block_h, block_n), "float32")

            T.copy(scores, scores_buf)

            # mask out elements where flattened index >= threshold
            for i, j in T.Parallel(block_h, block_n):
                masked_buf[i, j] = T.if_then_else(
                    i * block_n + j < threshold,
                    scores_buf[i, j],
                    -T.float32(10000.0),
                )

            for i, j in T.Parallel(block_h, block_n):
                out_buf[i, j] = masked_buf[i, j] / logsum[i]

            T.copy(out_buf, out)

    return main


@pytest.mark.parametrize("block_h, block_n, valid_block_n", SHAPE_CASES)
def test_linear_combo_condition(block_h, block_n, valid_block_n):
    kernel = tilelang.compile(
        kernel_linear_combo_condition(block_h, block_n, valid_block_n),
        target="npuir",
    )
    scores = torch.randn((block_h, block_n), dtype=torch.float32, device="npu")
    logsum = torch.rand((block_h,), dtype=torch.float32, device="npu") + 1.0
    out = torch.zeros((block_h, block_n), dtype=torch.float32, device="npu")

    threshold = block_h * valid_block_n
    flat_idx = torch.arange(block_h * block_n, device="npu").reshape(block_h, block_n)
    masked = torch.where(
        flat_idx < threshold, scores, torch.full_like(scores, -10000.0)
    )
    ref = masked / logsum.unsqueeze(1)

    kernel(scores, logsum, out)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


# ============================================================
# Case 2: Cross-loop-variable comparison: i < j (upper triangle mask)
# Elements where row index < col index are kept; others set to -10000
# ============================================================
def kernel_cross_var_compare(block_h, block_n):
    @T.prim_func
    def main(
        scores: T.Tensor((block_h, block_n), "float32"),
        logsum: T.Tensor((block_h,), "float32"),
        out: T.Tensor((block_h, block_n), "float32"),
    ):
        with T.Kernel(1, is_npu=True):
            scores_buf = T.alloc_shared((block_h, block_n), "float32")
            masked_buf = T.alloc_shared((block_h, block_n), "float32")
            out_buf = T.alloc_shared((block_h, block_n), "float32")

            T.copy(scores, scores_buf)

            # upper triangle: keep elements where i < j
            for i, j in T.Parallel(block_h, block_n):
                masked_buf[i, j] = T.if_then_else(
                    i < j,
                    scores_buf[i, j],
                    -T.float32(10000.0),
                )

            for i, j in T.Parallel(block_h, block_n):
                out_buf[i, j] = masked_buf[i, j] / logsum[i]

            T.copy(out_buf, out)

    return main


@pytest.mark.parametrize("block_h, block_n, valid_block_n", SHAPE_CASES)
def test_cross_var_compare(block_h, block_n, valid_block_n):
    kernel = tilelang.compile(
        kernel_cross_var_compare(block_h, block_n),
        target="npuir",
    )
    scores = torch.randn((block_h, block_n), dtype=torch.float32, device="npu")
    logsum = torch.rand((block_h,), dtype=torch.float32, device="npu") + 1.0
    out = torch.zeros((block_h, block_n), dtype=torch.float32, device="npu")

    row = torch.arange(block_h, device="npu").unsqueeze(1)
    col = torch.arange(block_n, device="npu").unsqueeze(0)
    masked = torch.where(row < col, scores, torch.full_like(scores, -10000.0))
    ref = masked / logsum.unsqueeze(1)

    kernel(scores, logsum, out)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


# ============================================================
# Case 3: Loop variables appear directly in value arithmetic
# out[i, j] = scores[i, j] * (i + 1) + scores[i, j] * j
# Both i and j participate in actual computation, not just condition
# ============================================================
def kernel_affine_value_expr(block_h, block_n, valid_block_n):
    @T.prim_func
    def main(
        scores: T.Tensor((block_h, block_n), "float32"),
        logsum: T.Tensor((block_h,), "float32"),
        out: T.Tensor((block_h, block_n), "float32"),
    ):
        with T.Kernel(1, is_npu=True):
            scores_buf = T.alloc_shared((block_h, block_n), "float32")
            masked_buf = T.alloc_shared((block_h, block_n), "float32")
            out_buf = T.alloc_shared((block_h, block_n), "float32")

            T.copy(scores, scores_buf)

            # scale each element by its row index weight (i + 1)
            # and mask out columns >= valid_block_n
            for i, j in T.Parallel(block_h, block_n):
                masked_buf[i, j] = T.if_then_else(
                    j < valid_block_n,
                    scores_buf[i, j] * (i + T.float32(1.0)),  # i participates in value
                    -T.float32(10000.0),
                )

            # divide by logsum, then add column offset j as position bias
            for i, j in T.Parallel(block_h, block_n):
                out_buf[i, j] = masked_buf[i, j] / logsum[i] + j * T.float32(
                    0.01
                )  # j participates in value

            T.copy(out_buf, out)

    return main


@pytest.mark.parametrize("block_h, block_n, valid_block_n", SHAPE_CASES)
def test_affine_value_expr(block_h, block_n, valid_block_n):
    kernel = tilelang.compile(
        kernel_affine_value_expr(block_h, block_n, valid_block_n),
        target="npuir",
    )
    scores = torch.randn((block_h, block_n), dtype=torch.float32, device="npu")
    logsum = torch.rand((block_h,), dtype=torch.float32, device="npu") + 1.0
    out = torch.zeros((block_h, block_n), dtype=torch.float32, device="npu")

    row = torch.arange(block_h, device="npu").unsqueeze(1).float()
    col = torch.arange(block_n, device="npu").unsqueeze(0).float()

    # first pass: scale by (i + 1), mask columns >= valid_block_n
    masked = torch.where(
        col < valid_block_n,
        scores * (row + 1.0),
        torch.full_like(scores, -10000.0),
    )
    # second pass: divide by logsum, add column position bias
    ref = masked / logsum.unsqueeze(1) + col * 0.01

    kernel(scores, logsum, out)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


# ============================================================
# Case 4: Combined condition: j < valid_block_n AND i <= j
# Causal mask intersected with column threshold
# ============================================================
def kernel_combined_condition(block_h, block_n, valid_block_n):
    @T.prim_func
    def main(
        scores: T.Tensor((block_h, block_n), "float32"),
        logsum: T.Tensor((block_h,), "float32"),
        out: T.Tensor((block_h, block_n), "float32"),
    ):
        with T.Kernel(1, is_npu=True):
            scores_buf = T.alloc_shared((block_h, block_n), "float32")
            masked_buf = T.alloc_shared((block_h, block_n), "float32")
            masked_buf2 = T.alloc_shared((block_h, block_n), "float32")
            out_buf = T.alloc_shared((block_h, block_n), "float32")

            T.copy(scores, scores_buf)

            # first pass: column threshold mask (j < valid_block_n)
            for i, j in T.Parallel(block_h, block_n):
                masked_buf[i, j] = T.if_then_else(
                    j < valid_block_n,
                    scores_buf[i, j],
                    -T.float32(10000.0),
                )

            # second pass: causal mask (i <= j), applied on top
            for i, j in T.Parallel(block_h, block_n):
                masked_buf2[i, j] = T.if_then_else(
                    i <= j,
                    masked_buf[i, j],
                    -T.float32(10000.0),
                )

            for i, j in T.Parallel(block_h, block_n):
                out_buf[i, j] = masked_buf2[i, j] / logsum[i]

            T.copy(out_buf, out)

    return main


@pytest.mark.parametrize("block_h, block_n, valid_block_n", SHAPE_CASES)
def test_combined_condition(block_h, block_n, valid_block_n):
    kernel = tilelang.compile(
        kernel_combined_condition(block_h, block_n, valid_block_n),
        target="npuir",
    )
    scores = torch.randn((block_h, block_n), dtype=torch.float32, device="npu")
    logsum = torch.rand((block_h,), dtype=torch.float32, device="npu") + 1.0
    out = torch.zeros((block_h, block_n), dtype=torch.float32, device="npu")

    row = torch.arange(block_h, device="npu").unsqueeze(1)
    col = torch.arange(block_n, device="npu").unsqueeze(0)
    col_mask = col < valid_block_n
    causal_mask = row <= col
    masked = torch.where(
        col_mask & causal_mask, scores, torch.full_like(scores, -10000.0)
    )
    ref = masked / logsum.unsqueeze(1)

    kernel(scores, logsum, out)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
