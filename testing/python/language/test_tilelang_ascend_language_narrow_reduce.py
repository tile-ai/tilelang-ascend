"""Row-reduce over part of a wider row.

A reduce whose logical width is only part of the row the data physically sits
in has to step by the PHYSICAL row width to move between rows. That happens
whenever a region is sliced along the columns (``ub[:, a:b]``) or ``real_shape``
names fewer columns than the buffer holds -- an online softmax over a sliding
window is the usual source of both.

The {M, N} shape alone cannot express it: AscendC's Reduce* walks the source as
one contiguous M x N block, so for a [M, 512] tile reduced over 64 columns it
consumes elements [0, M*64) -- row 0's first eight chunks -- and returns those
as if they were the per-row results. Row 0 lands on the right answer (it starts
at offset 0), every other row does not, and nothing reports an error.

Both spellings are covered, and the padding is filled with a value that changes
the result if it is read, so a wrong walk cannot pass by luck. The first row is
excluded from that poisoning on purpose: it is correct either way, and a test
that only looked at it would pass against the broken walk.
"""

import pytest

import torch

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _reduce_fn(op):
    return {"sum": T.reduce_sum, "max": T.reduce_max, "min": T.reduce_min}[op]


def narrow_via_real_shape(M, buf_n, valid, op, dtype="float"):
    """Whole buffer in, logical width given by real_shape."""
    reduce_fn = _reduce_fn(op)

    @T.prim_func
    def main(src: T.Tensor([M, buf_n], dtype), out: T.Tensor([M], dtype)):
        with T.Kernel(1, is_npu=True) as (_, vid):
            ub = T.alloc_ub([M, buf_n], dtype)
            acc = T.alloc_ub([M], dtype)
            if vid == 0:
                T.copy(src, ub)
                reduce_fn(ub, acc, dim=-1, real_shape=[M, valid])
                T.copy(acc, out)

    return main


def narrow_via_slice(M, buf_n, valid, op, offset, dtype="float"):
    """A column range of the buffer, named by slicing the region."""
    reduce_fn = _reduce_fn(op)

    @T.prim_func
    def main(src: T.Tensor([M, buf_n], dtype), out: T.Tensor([M], dtype)):
        with T.Kernel(1, is_npu=True) as (_, vid):
            ub = T.alloc_ub([M, buf_n], dtype)
            acc = T.alloc_ub([M], dtype)
            if vid == 0:
                T.copy(src, ub)
                reduce_fn(ub[:, offset : offset + valid], acc, dim=-1)
                T.copy(acc, out)

    return main


def _reference(data, op, lo, hi):
    window = data[:, lo:hi]
    if op == "sum":
        return window.sum(dim=-1)
    if op == "max":
        return window.max(dim=-1).values
    return window.min(dim=-1).values


def _data(M, buf_n, op, lo, hi):
    torch.manual_seed(0)
    data = torch.randn(M, buf_n, dtype=torch.float32)
    # Outside the reduced range, plant a value that would dominate the result if
    # it were read. Row 0 is left alone: it is correct under either walk, so
    # poisoning it would hide which rows actually went wrong.
    poison = {"sum": 7.0, "max": 100.0, "min": -100.0}[op]
    data[1:, :lo] = poison
    data[1:, hi:] = poison
    return data


@pytest.mark.parametrize("op", ["sum", "max", "min"])
@pytest.mark.parametrize("valid", [64, 32, 16])
def test_narrow_reduce_real_shape(op, valid):
    M, buf_n = 16, 512
    tilelang.cache.clear_cache()
    func = tilelang.compile(
        narrow_via_real_shape(M, buf_n, valid, op),
        out_idx=[-1],
        target="ascendc",
        pass_configs=pass_configs,
    )
    data = _data(M, buf_n, op, 0, valid)
    out = func(data.npu()).cpu()
    torch.testing.assert_close(out, _reference(data, op, 0, valid), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("op", ["sum", "max", "min"])
@pytest.mark.parametrize("offset", [0, 64, 448])
def test_narrow_reduce_column_slice(op, offset):
    M, buf_n, valid = 16, 512, 64
    tilelang.cache.clear_cache()
    func = tilelang.compile(
        narrow_via_slice(M, buf_n, valid, op, offset),
        out_idx=[-1],
        target="ascendc",
        pass_configs=pass_configs,
    )
    data = _data(M, buf_n, op, offset, offset + valid)
    out = func(data.npu()).cpu()
    torch.testing.assert_close(out, _reference(data, op, offset, offset + valid), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
