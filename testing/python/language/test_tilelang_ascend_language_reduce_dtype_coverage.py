"""
T.reduce_max / T.reduce_min / T.reduce_sum 补充测试：clear=False 2D + 异常边界

不重复 elementwise.py / narrow_reduce.py 中已有的 dim/dtype/real_shape 测试，
仅补充仓库缺失的覆盖：
1. clear=False 2D 累加模式
2. 异常边界：不支持 dtype / shape 不匹配 / dim 越界 / real_shape 错误 / rank 过高
"""

import pytest
import tilelang
import tilelang.language as T
import torch


@pytest.fixture(scope="module", autouse=True)
def _disable_cache():
    tilelang.disable_cache()
    yield
    tilelang.enable_cache()


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

DTYPE_TORCH_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}

REDUCE_FNS = {
    "max": T.reduce_max,
    "min": T.reduce_min,
    "sum": T.reduce_sum,
}

REF_FNS = {
    "max": lambda x, dim: x.max(dim=dim).values,
    "min": lambda x, dim: x.min(dim=dim).values,
    "sum": lambda x, dim: x.sum(dim=dim),
}


def _make_1d_kernel(op, dtype, N=128):
    reduce_fn = REDUCE_FNS[op]

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((1,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((N,), dtype)
            b_ub = T.alloc_ub((1,), dtype)
            T.copy(A[0], a_ub)
            reduce_fn(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[0])

    return main


def _make_clear_false_2d_kernel(op, dtype, init_value, M=16, N=128):
    """clear=False on 2D buffer with non-zero init to verify merge behavior."""
    reduce_fn = REDUCE_FNS[op]

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((M,), dtype)
            T.copy(A[0, 0], a_ub)
            T.tile.fill(b_ub, init_value)
            reduce_fn(a_ub, b_ub, dim=-1, clear=False)
            T.copy(b_ub, B[0])

    return main


# ---------------------------------------------------------------------------
# clear=False on 2D buffer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="reduce correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize(
    "op", ["sum", pytest.param("max", marks=pytest.mark.low_priority), pytest.param("min", marks=pytest.mark.low_priority)]
)
@pytest.mark.parametrize("dtype", ["float32", pytest.param("float16", marks=pytest.mark.low_priority)])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_clear_false_2d(op, dtype, target):
    """clear=False accumulation on 2D buffer with non-zero init.

    Uses deterministic input and init values that distinguish clear=False
    from clear=True: if clear=False is ignored, the result equals the
    plain reduce (without merging init), which differs from the expected
    merge result.
    """
    if op == "sum" and dtype == "float16" and target == "ascendc":
        pytest.skip("CANN limit: ReduceSum<half, AR> rejected by static_assert (clear=False path has no reduce_sum_half workaround)")
    M, N = 16, 128
    torch_dtype = DTYPE_TORCH_MAP[dtype]

    # Deterministic input: arange * 0.1 gives values 0.0, 0.1, ..., 204.7
    a = (torch.arange(M * N, dtype=torch.float32) * 0.1).reshape(M, N).to(torch_dtype)

    # Init values chosen so clear=False and clear=True give different results:
    # - sum: init=1.25 → clear=False: sum(a)+1.25, clear=True: sum(a)
    # - max: init=1e6  → clear=False: 1e6 (all a < 1e6), clear=True: max(a)
    # - min: init=-1e6 → clear=False: -1e6 (all a > -1e6), clear=True: min(a)
    INIT_VALUES = {"sum": 1.25, "max": 1e6, "min": -1e6}
    init_value = INIT_VALUES[op]

    program = _make_clear_false_2d_kernel(op, dtype, init_value, M, N)
    func = tilelang.compile(program, out_idx=[-1], pass_configs=pass_configs, target=target)

    a_npu = a.npu()
    torch.npu.synchronize()
    b = func(a_npu)
    torch.npu.synchronize()

    # Expected: merge reduce result with init value (clear=False semantics)
    ref = REF_FNS[op](a, -1)
    if op == "sum":
        ref = ref + init_value
    elif op == "max":
        ref = torch.maximum(ref, torch.tensor(init_value, dtype=torch_dtype))
    elif op == "min":
        ref = torch.minimum(ref, torch.tensor(init_value, dtype=torch_dtype))
    torch.testing.assert_close(b.cpu(), ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Unsupported dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op", ["sum", pytest.param("max", marks=pytest.mark.low_priority), pytest.param("min", marks=pytest.mark.low_priority)]
)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_unsupported_dtype(op, dtype, target):
    """Unsupported dtype should fail to compile"""
    N = 128
    program = _make_1d_kernel(op, dtype, N)
    with pytest.raises(RuntimeError, match="Compilation Failed"):
        tilelang.compile(program, out_idx=[-1], pass_configs=pass_configs, target=target)


# ---------------------------------------------------------------------------
# Exception boundaries: compile-time ValueError / TypeError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)])
def test_reduce_out_shape_mismatch(target):
    """out shape not matching reduced/keepdim shape should raise ValueError."""

    with pytest.raises(RuntimeError, match="DiagnosticError"):

        @T.prim_func
        def main(A: T.Tensor((16, 128), "float32"), B: T.Tensor((32,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((16, 128), "float32")
                b_ub = T.alloc_ub((32,), "float32")
                T.copy(A[0, 0], a_ub)
                T.reduce_max(a_ub, b_ub, dim=-1)
                T.copy(b_ub, B[0])

        tilelang.compile(main, out_idx=[-1], pass_configs=pass_configs, target=target)


@pytest.mark.low_priority
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_dim_not_int(target):
    """dim as non-integer (tuple) should raise TypeError."""

    with pytest.raises(RuntimeError, match="DiagnosticError"):

        @T.prim_func
        def main(A: T.Tensor((16, 128), "float32"), B: T.Tensor((16,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((16, 128), "float32")
                b_ub = T.alloc_ub((16,), "float32")
                T.copy(A[0, 0], a_ub)
                T.reduce_max(a_ub, b_ub, dim=(0, 1))
                T.copy(b_ub, B[0])

        tilelang.compile(main, out_idx=[-1], pass_configs=pass_configs, target=target)


@pytest.mark.low_priority
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_real_shape_wrong_length(target):
    """real_shape with length != 2 should raise ValueError."""

    with pytest.raises(RuntimeError, match="DiagnosticError"):

        @T.prim_func
        def main(A: T.Tensor((16, 128), "float32"), B: T.Tensor((16,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((16, 128), "float32")
                b_ub = T.alloc_ub((16,), "float32")
                T.copy(A[0, 0], a_ub)
                T.reduce_max(a_ub, b_ub, dim=-1, real_shape=[16, 128, 1])
                T.copy(b_ub, B[0])

        tilelang.compile(main, out_idx=[-1], pass_configs=pass_configs, target=target)


@pytest.mark.low_priority
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_real_shape_exceed_extent(target):
    """real_shape exceeding buffer extent should raise ValueError."""

    with pytest.raises(RuntimeError, match="DiagnosticError"):

        @T.prim_func
        def main(A: T.Tensor((16, 128), "float32"), B: T.Tensor((16,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((16, 128), "float32")
                b_ub = T.alloc_ub((16,), "float32")
                T.copy(A[0, 0], a_ub)
                T.reduce_max(a_ub, b_ub, dim=-1, real_shape=[16, 256])
                T.copy(b_ub, B[0])

        tilelang.compile(main, out_idx=[-1], pass_configs=pass_configs, target=target)


@pytest.mark.low_priority
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_1d_invalid_dim(target):
    """1D buffer with dim=1 should raise ValueError (only 0/-1 valid)."""

    with pytest.raises(RuntimeError, match="DiagnosticError"):

        @T.prim_func
        def main(A: T.Tensor((128,), "float32"), B: T.Tensor((1,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((128,), "float32")
                b_ub = T.alloc_ub((1,), "float32")
                T.copy(A[0], a_ub)
                T.reduce_max(a_ub, b_ub, dim=1)
                T.copy(b_ub, B[0])

        tilelang.compile(main, out_idx=[-1], pass_configs=pass_configs, target=target)


@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)])
def test_reduce_2d_invalid_dim(target):
    """2D buffer with dim=2 should raise ValueError (only 0/1/-1/-2 valid)."""

    with pytest.raises(RuntimeError, match="DiagnosticError"):

        @T.prim_func
        def main(A: T.Tensor((16, 128), "float32"), B: T.Tensor((16,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((16, 128), "float32")
                b_ub = T.alloc_ub((16,), "float32")
                T.copy(A[0, 0], a_ub)
                T.reduce_max(a_ub, b_ub, dim=2)
                T.copy(b_ub, B[0])

        tilelang.compile(main, out_idx=[-1], pass_configs=pass_configs, target=target)


@pytest.mark.low_priority
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_rank_too_high(target):
    """4D buffer should raise ValueError (max rank 3)."""

    with pytest.raises(RuntimeError, match="DiagnosticError"):

        @T.prim_func
        def main(A: T.Tensor((2, 4, 16, 128), "float32"), B: T.Tensor((2, 4, 16), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((2, 4, 16, 128), "float32")
                b_ub = T.alloc_ub((2, 4, 16), "float32")
                T.copy(A[0, 0, 0, 0], a_ub)
                T.reduce_max(a_ub, b_ub, dim=-1)
                T.copy(b_ub, B[0, 0, 0])

        tilelang.compile(main, out_idx=[-1], pass_configs=pass_configs, target=target)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
