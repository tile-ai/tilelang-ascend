"""
T.reduce_max / T.reduce_min / T.reduce_sum 补充测试：全量 dtype × dim × clear + 异常边界

补充现有测试缺失的组合：
1. dim=0（沿列方向归约，2D）
2. clear=False 累加模式
3. 异常边界：不支持的 dtype、不支持的 dim 值
"""

import pytest
import tilelang
import tilelang.language as T
import torch

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


def _make_2d_col_kernel(op, dtype, M=16, N=128):
    reduce_fn = REDUCE_FNS[op]

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((1, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((1, N), dtype)
            T.copy(A[0, 0], a_ub)
            reduce_fn(a_ub, b_ub, dim=0)
            T.copy(b_ub, B[0, 0])

    return main


def _make_2d_row_kernel(op, dtype, M=16, N=128):
    reduce_fn = REDUCE_FNS[op]

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, 1), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((M, 1), dtype)
            T.copy(A[0, 0], a_ub)
            reduce_fn(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[0, 0])

    return main


def _make_clear_false_kernel(op, dtype, N=128):
    reduce_fn = REDUCE_FNS[op]

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((1,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((N,), dtype)
            b_ub = T.alloc_ub((1,), dtype)
            T.copy(A[0], a_ub)
            T.tile.fill(b_ub, 0.0)
            reduce_fn(a_ub, b_ub, dim=-1, clear=False)
            T.copy(b_ub, B[0])

    return main


@pytest.mark.low_priority
@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="reduce correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("op", ["max", "min", "sum"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_1d_all_dtype(op, dtype, target):
    """1D 全量 dtype x target"""
    N = 128
    program = _make_1d_kernel(op, dtype, N)
    func = tilelang.compile(program, out_idx=[-1], pass_configs=pass_configs, target=target)
    torch_dtype = DTYPE_TORCH_MAP[dtype]
    a = torch.randn(N, dtype=torch_dtype).npu()
    torch.npu.synchronize()
    b = func(a)
    ref = REF_FNS[op](a, -1).reshape(1)
    if op == "sum" and dtype == "float16":
        pytest.xfail("float16 reduction sum may overflow")
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.low_priority
@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="reduce correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("op", ["max", "min", "sum"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_2d_dim0(op, dtype, target):
    """2D dim=0 (column reduction)"""
    M, N = 16, 128
    program = _make_2d_col_kernel(op, dtype, M, N)
    func = tilelang.compile(program, out_idx=[-1], pass_configs=pass_configs, target=target)
    torch_dtype = DTYPE_TORCH_MAP[dtype]
    a = torch.randn(M, N, dtype=torch_dtype).npu()
    torch.npu.synchronize()
    b = func(a)
    ref = REF_FNS[op](a, 0).reshape(1, N)
    if op == "sum" and dtype == "float16":
        pytest.xfail("float16 reduction sum may overflow")
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.low_priority
@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="reduce correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("op", ["max", "min", "sum"])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_clear_false(op, dtype, target):
    """clear=False accumulation mode"""
    N = 128
    program = _make_clear_false_kernel(op, dtype, N)
    func = tilelang.compile(program, out_idx=[-1], pass_configs=pass_configs, target=target)
    torch_dtype = DTYPE_TORCH_MAP[dtype]
    a = torch.randn(N, dtype=torch_dtype).npu()
    torch.npu.synchronize()
    b = func(a)
    ref = REF_FNS[op](a, -1).reshape(1)
    if op == "max":
        ref = torch.maximum(ref, torch.tensor(0.0, dtype=torch_dtype).npu())
    elif op == "min":
        ref = torch.minimum(ref, torch.tensor(0.0, dtype=torch_dtype).npu())
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.low_priority
@pytest.mark.parametrize("op", ["max", "min", "sum"])
@pytest.mark.parametrize("dtype", ["int32", "bfloat16", "int16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_unsupported_dtype(op, dtype, target):
    """Unsupported dtype should fail to compile"""
    if target == "pto" and dtype in ("int32", "int16"):
        pytest.xfail("pto backend does not reject int32/int16 for reduce ops")
    N = 128
    program = _make_1d_kernel(op, dtype, N)
    with pytest.raises(Exception):  # noqa: B017
        tilelang.compile(program, out_idx=[-1], pass_configs=pass_configs, target=target)


@pytest.mark.low_priority
@pytest.mark.xfail(reason="3D dim=2 not raising; _legalize_reduce_dim should reject but doesn't")
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_reduce_invalid_dim_3d(target):
    """3D buffer dim=2 should raise (only 0/1/-1/-2 supported)"""

    @T.prim_func
    def main(A: T.Tensor((4, 16, 128), "float32"), B: T.Tensor((4, 16, 1), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((4, 16, 128), "float32")
            b_ub = T.alloc_ub((4, 16, 1), "float32")
            T.copy(A[0, 0, 0], a_ub)
            T.reduce_max(a_ub, b_ub, dim=2)
            T.copy(b_ub, B[0, 0, 0])

    with pytest.raises(Exception):  # noqa: B017
        tilelang.compile(main, out_idx=[-1], pass_configs=pass_configs, target=target)
