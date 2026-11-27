"""Regression tests for issue #2005.

Two independent failures blocked a pipelined FP8 decode loop:

1. A scalar binding in the loop body (``idx = ...``) becomes a ``LetStmt``
   that scopes every stage, and PipelinePlanning rejected the body.
2. ``T.reinterpretcast`` emitted ``LocalTensor<T> dst = src.ReinterpretCast<T>()``
   every time the software pipeline cloned the stage.  The prologue, steady
   state and epilogue share one C++ scope, so bisheng reported
   ``redefinition of 'f32'`` for any loop with more than one iteration.
"""

import pytest
import torch
import tilelang
import tilelang.language as T

TARGET = "ascendc"


@pytest.fixture(scope="module", autouse=True)
def _disable_cache():
    tilelang.disable_cache()
    yield
    tilelang.enable_cache()


def _decode_kernel(k_pages, n, mode):
    """The issue's minimal compile-only reproducer."""

    @T.prim_func
    def main(x: T.Tensor((k_pages, n), "uint8"), y: T.Tensor((k_pages, n), "bfloat16")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            u8 = T.alloc_ub((n,), "uint8")
            f16 = T.alloc_ub((n,), "float16")
            i32 = T.alloc_ub((n,), "int32")
            f32 = T.alloc_ub((n,), "float32")
            out = T.alloc_ub((n,), "bfloat16")

            for k in T.Pipelined(k_pages, num_stages=2):
                if mode == "let":
                    idx = T.alloc_var("int32")
                    idx = T.max(T.min(k, k_pages - 1), 0)
                    T.copy(x[idx, 0:n], u8)
                else:
                    T.copy(x[k, 0:n], u8)

                T.tile.cast(f16, u8, "CAST_NONE", n)
                T.tile.cast(i32, f16, "CAST_RINT", n)
                T.tile.bitwise_lshift(i32, i32, 20)
                T.reinterpretcast(f32, i32, "float")
                T.tile.cast(out, f32, "CAST_RINT", n)
                T.copy(out, y[k, 0:n])

    return main


def test_pipeline_let_stmt_body_compiles():
    """A scalar LetStmt wrapping the staged body is inlined before planning."""
    kernel = tilelang.compile(_decode_kernel(8, 128, "let"), target=TARGET)
    assert kernel is not None


@pytest.mark.parametrize("pages", [1, 2, 8])
def test_pipeline_reinterpretcast_clone_compiles(pages):
    """Cloned stages rebind the pre-allocated LocalTensor instead of
    redeclaring it; pages=1 is the pre-fix anchor that already worked."""
    kernel = tilelang.compile(_decode_kernel(pages, 128, "reinterpret"), target=TARGET)
    assert kernel is not None
    source = kernel.get_kernel_source()
    # A rebind is an assignment, not a second declaration of the same name.
    assert "f32 = i32.ReinterpretCast<float>()" in source
    assert source.count("LocalTensor<float> f32 =") <= 1


def _conditional_kernel(k_pages, n):
    """A scalar assigned at the top level and again inside an ``if``."""

    @T.prim_func
    def main(x: T.Tensor((k_pages, n), "uint8"), y: T.Tensor((k_pages, n), "bfloat16")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            u8 = T.alloc_ub((n,), "uint8")
            f16 = T.alloc_ub((n,), "float16")
            i32 = T.alloc_ub((n,), "int32")
            f32 = T.alloc_ub((n,), "float32")
            out = T.alloc_ub((n,), "bfloat16")
            for k in T.Pipelined(k_pages, num_stages=2):
                idx = T.alloc_var("int32")
                idx = k
                if k % 2 == 0:
                    idx = k_pages - 1
                T.copy(x[idx, 0:n], u8)
                T.tile.cast(f16, u8, "CAST_NONE", n)
                T.tile.cast(i32, f16, "CAST_RINT", n)
                T.tile.bitwise_lshift(i32, i32, 20)
                T.reinterpretcast(f32, i32, "float")
                T.tile.cast(out, f32, "CAST_RINT", n)
                T.copy(out, y[k, 0:n])

    return main


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="correctness check requires an Ascend NPU runtime",
)
def test_pipeline_conditional_scalar_store():
    """A conditional second assignment must not be replaced by the top-level
    value; the scalar statements are grouped with the copy that consumes it."""
    k_pages, n = 8, 128
    kernel = tilelang.compile(_conditional_kernel(k_pages, n), target=TARGET)
    x = torch.randint(0, 256, (k_pages, n), dtype=torch.uint8, device="npu")
    y = torch.zeros(k_pages, n, dtype=torch.bfloat16, device="npu")
    torch.npu.synchronize()
    kernel(x, y)
    torch.npu.synchronize()

    xc = x.cpu()
    rows = []
    for k in range(k_pages):
        src = k_pages - 1 if k % 2 == 0 else k
        v = xc[src].to(torch.float16).to(torch.int32)
        v = torch.bitwise_left_shift(v, 20)
        rows.append(v.view(torch.float32).to(torch.bfloat16))
    ref = torch.stack(rows)
    torch.testing.assert_close(y.cpu().float(), ref.float(), rtol=2e-2, atol=2e-2)
