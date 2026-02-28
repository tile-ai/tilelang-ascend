import pytest

import torch

import tilelang
import tilelang.language as T

"""
Test suite for verifying that T.tile.add, T.tile.sub, T.tile.mul, T.tile.max,
T.tile.exp, T.tile.broadcast, T.reduce_max, and T.reduce_sum support
BufferRegion (slices) as any operand.
"""

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before tests"""
    tilelang.cache.clear_cache()
    yield


# ======================== T.tile.add with slices ========================

def vec_add_slice_dst_src0(M, N, block_M, block_N, dtype="float"):
    """Test add with BufferRegion as dst and src0."""
    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            buf = T.alloc_ub((2 * block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)

            if vid == 0:
                T.copy(A[0, 0], buf[0:block_M, 0:block_N])
                T.copy(B[0, 0], b_ub)

                # dst=slice, src0=slice, src1=buffer
                T.tile.add(buf[block_M:2 * block_M, 0:block_N],
                           buf[0:block_M, 0:block_N], b_ub)

                T.copy(buf[block_M:2 * block_M, 0:block_N], C[0, 0])

    return main


def run_test_add_slice_dst_src0(M, N, block_M, block_N, target):
    func = vec_add_slice_dst_src0(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).float().npu()
    b = torch.randn(M, N).float().npu()
    torch.npu.synchronize()

    c = func(a, b)
    ref_c = a[:block_M, :block_N] + b[:block_M, :block_N]
    torch.testing.assert_close(c[:block_M, :block_N], ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_add_slice_dst_src0(target):
    run_test_add_slice_dst_src0(128, 128, 64, 128, target=target)


# ======================== T.tile.sub with slices ========================

def vec_sub_slice_all(M, N, block_M, block_N, dtype="float"):
    """Test sub with BufferRegion as all operands."""
    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            buf = T.alloc_ub((3 * block_M, block_N), dtype)

            if vid == 0:
                T.copy(A[0, 0], buf[0:block_M, 0:block_N])
                T.copy(B[0, 0], buf[block_M:2 * block_M, 0:block_N])

                # All operands are slices
                T.tile.sub(buf[2 * block_M:3 * block_M, 0:block_N],
                           buf[0:block_M, 0:block_N],
                           buf[block_M:2 * block_M, 0:block_N])

                T.copy(buf[2 * block_M:3 * block_M, 0:block_N], C[0, 0])

    return main


def run_test_sub_slice_all(M, N, block_M, block_N, target):
    func = vec_sub_slice_all(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).float().npu()
    b = torch.randn(M, N).float().npu()
    torch.npu.synchronize()

    c = func(a, b)
    ref_c = a[:block_M, :block_N] - b[:block_M, :block_N]
    torch.testing.assert_close(c[:block_M, :block_N], ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_sub_slice_all(target):
    run_test_sub_slice_all(128, 128, 64, 128, target=target)


# ======================== T.tile.mul with slices ========================

def vec_mul_slice_src0_src1(M, N, block_M, block_N, dtype="float"):
    """Test mul with BufferRegion as src0 and src1."""
    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            buf = T.alloc_ub((2 * block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)

            if vid == 0:
                T.copy(A[0, 0], buf[0:block_M, 0:block_N])
                T.copy(B[0, 0], buf[block_M:2 * block_M, 0:block_N])

                # src0=slice, src1=slice
                T.tile.mul(c_ub,
                           buf[0:block_M, 0:block_N],
                           buf[block_M:2 * block_M, 0:block_N])

                T.copy(c_ub, C[0, 0])

    return main


def run_test_mul_slice_src0_src1(M, N, block_M, block_N, target):
    func = vec_mul_slice_src0_src1(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).float().npu()
    b = torch.randn(M, N).float().npu()
    torch.npu.synchronize()

    c = func(a, b)
    ref_c = a[:block_M, :block_N] * b[:block_M, :block_N]
    torch.testing.assert_close(c[:block_M, :block_N], ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_mul_slice_src0_src1(target):
    run_test_mul_slice_src0_src1(128, 128, 64, 128, target=target)


# ======================== T.tile.max with slices ========================

def vec_max_slice_dst(M, N, block_M, block_N, dtype="float"):
    """Test max with BufferRegion as dst."""
    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            out_buf = T.alloc_ub((2 * block_M, block_N), dtype)

            if vid == 0:
                T.copy(A[0, 0], a_ub)
                T.copy(B[0, 0], b_ub)

                # dst=slice
                T.tile.max(out_buf[0:block_M, 0:block_N], a_ub, b_ub)

                T.copy(out_buf[0:block_M, 0:block_N], C[0, 0])

    return main


def run_test_max_slice_dst(M, N, block_M, block_N, target):
    func = vec_max_slice_dst(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).float().npu()
    b = torch.randn(M, N).float().npu()
    torch.npu.synchronize()

    c = func(a, b)
    ref_c = torch.max(a[:block_M, :block_N], b[:block_M, :block_N])
    torch.testing.assert_close(c[:block_M, :block_N], ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_max_slice_dst(target):
    run_test_max_slice_dst(128, 128, 64, 128, target=target)


# ======================== T.tile.exp with slices ========================

def vec_exp_slice(M, N, block_M, block_N, dtype="float"):
    """Test exp with BufferRegion as both dst and src0."""
    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            buf = T.alloc_ub((2 * block_M, block_N), dtype)

            if vid == 0:
                T.copy(A[0, 0], buf[0:block_M, 0:block_N])

                # dst=slice, src0=slice
                T.tile.exp(buf[block_M:2 * block_M, 0:block_N],
                           buf[0:block_M, 0:block_N])

                T.copy(buf[block_M:2 * block_M, 0:block_N], B[0, 0])

    return main


def run_test_exp_slice(M, N, block_M, block_N, target):
    func = vec_exp_slice(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).float().npu()
    torch.npu.synchronize()

    b = func(a)
    ref_b = torch.exp(a[:block_M, :block_N])
    torch.testing.assert_close(b[:block_M, :block_N], ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_exp_slice(target):
    run_test_exp_slice(128, 128, 64, 128, target=target)


# ======================== T.tile.broadcast with slices ========================

def vec_broadcast_slice(M, N, block_M, block_N, dtype="float"):
    """Test broadcast with BufferRegion as dst, src, and tmp."""
    @T.prim_func
    def main(
            A: T.Tensor((1, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            buf = T.alloc_ub((block_M + 1, block_N), dtype)
            tmp = T.alloc_ub((2 * block_M, block_N), dtype)

            if vid == 0:
                T.copy(A[0, 0], buf[0:1, 0:block_N])

                # dst=slice, src=slice, tmp=slice
                T.tile.broadcast(buf[1:block_M + 1, 0:block_N],
                                 buf[0:1, 0:block_N],
                                 tmp[0:block_M, 0:block_N])

                T.copy(buf[1:block_M + 1, 0:block_N], B[0, 0])

    return main


def run_test_broadcast_slice(M, N, block_M, block_N, target):
    func = vec_broadcast_slice(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(1, N).float().npu()
    torch.npu.synchronize()

    b = func(a)
    ref_b = a.expand(block_M, block_N)
    torch.testing.assert_close(b[:block_M, :block_N], ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_broadcast_slice(target):
    run_test_broadcast_slice(64, 128, 64, 128, target=target)


# ======================== T.reduce_max with slices ========================

def vec_reduce_max_slice(M, N, block_M, block_N, dtype="float"):
    """Test reduce_max with BufferRegion as buffer operand."""
    from tilelang import DataType

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            buf = T.alloc_ub((2 * block_M, block_N), dtype)
            out_ub = T.alloc_ub((1, block_N), dtype)
            tmp = T.alloc_shared([3 * DataType(dtype).bits // 8 * block_M * block_N], dtype)

            if vid == 0:
                T.copy(A[0, 0], buf[0:block_M, 0:block_N])

                # buffer=slice
                T.reduce_max(buf[0:block_M, 0:block_N], out_ub, tmp, dim=0)

                T.copy(out_ub, B[0, 0])

    return main


def run_test_reduce_max_slice(M, N, block_M, block_N, target):
    func = vec_reduce_max_slice(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).float().npu()
    torch.npu.synchronize()

    b = func(a)
    ref_b = torch.max(a[:block_M, :block_N], dim=0, keepdim=True).values
    torch.testing.assert_close(b[:, :block_N], ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["pto"])
def test_reduce_max_slice(target):
    run_test_reduce_max_slice(128, 128, 64, 128, target=target)


# ======================== T.reduce_sum with slices ========================

def vec_reduce_sum_slice(M, N, block_M, block_N, dtype="float"):
    """Test reduce_sum with BufferRegion as buffer operand."""
    from tilelang import DataType

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            buf = T.alloc_ub((2 * block_M, block_N), dtype)
            out_ub = T.alloc_ub((1, block_N), dtype)
            tmp = T.alloc_shared([3 * DataType(dtype).bits // 8 * block_M * block_N], dtype)

            if vid == 0:
                T.copy(A[0, 0], buf[0:block_M, 0:block_N])

                # buffer=slice
                T.reduce_sum(buf[0:block_M, 0:block_N], out_ub, tmp, dim=0)

                T.copy(out_ub, B[0, 0])

    return main


def run_test_reduce_sum_slice(M, N, block_M, block_N, target):
    func = vec_reduce_sum_slice(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).float().npu()
    torch.npu.synchronize()

    b = func(a)
    ref_b = torch.sum(a[:block_M, :block_N], dim=0, keepdim=True)
    torch.testing.assert_close(b[:, :block_N], ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["pto"])
def test_reduce_sum_slice(target):
    run_test_reduce_sum_slice(128, 128, 64, 128, target=target)
