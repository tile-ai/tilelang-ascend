import random

import pytest

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T

"""
This is an element-wise pytest automation test suite.
All test cases are written and executed at the Developer Level.
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


@pytest.fixture
def setup_random_seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(0)
    yield


def vec_abs(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.abs(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_abs(M, N, block_M, block_N, target):
    func = vec_abs(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu()

    torch.npu.synchronize()
    b = func(a)

    ref_b = torch.abs(a)
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_abs(target, shape):
    M, N = shape
    run_test_abs(M, N, 128, 256, target=target)


def vec_add_auto_copy(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                C[bx * block_M + vid * block_M // VEC_NUM + i, by * block_N + j] = a_ub[i, j] + b_ub[i, j]

    return main


def run_test_add_auto_copy(M, N, block_M, block_N, target):
    func = vec_add_auto_copy(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).float().npu()
    b = torch.randn(M, N).float().npu()

    torch.npu.synchronize()

    c = func(a, b)
    ref_c = a + b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_add_auto_copy(target, shape):
    M, N = shape
    run_test_add_auto_copy(M, N, 128, 128, target=target)


def vec_add_developer(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                c_ub[i, j] = a_ub[i, j] + b_ub[i, j]

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_add_developer(M, N, block_M, block_N, target):
    func = vec_add_developer(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).float().npu()
    b = torch.randn(M, N).float().npu()
    torch.npu.synchronize()

    c = func(a, b)
    ref_c = a + b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_add_developer(target, shape):
    M, N = shape
    run_test_add_developer(M, N, 128, 128, target=target)


def adds(M, N, block_M, block_N, scalar, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.tile.add(b_ub, a_ub, scalar)
            T.copy(b_ub, B[bx * block_M, by * block_N])

    return main


def run_test_adds(M, N, block_M, block_N, scalar, target):
    func = adds(M, N, block_M, block_N, scalar)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).float().npu()
    b = func(a)
    ref_b = a + 2.0
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_adds(target, shape):
    M, N = shape
    run_test_adds(M, N, 64, 32, 2.0, target=target)


def bitwise_and(M, N, block_M, block_N, dtype="int16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
            T.tile.bitwise_and(c_ub, a_ub, b_ub)
            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_bitwise_and(M, N, block_M, block_N, target):
    func = bitwise_and(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randint(0, 10, (M, N), dtype=torch.int16).npu()
    b = torch.randint(0, 10, (M, N), dtype=torch.int16).npu()

    torch.npu.synchronize()

    c = func(a, b)
    ref_c = a & b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_and(target, shape):
    M, N = shape
    run_test_bitwise_and(M, N, 128, 256, target=target)


def axpy(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            T.tile.fill(b_ub, 0)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.axpy(b_ub, a_ub, 2.0)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_axpy(M, N, block_M, block_N, target):
    func = axpy(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = a * 2.0
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_axpy(target, shape):
    M, N = shape
    run_test_axpy(M, N, 128, 256, target)


def bilinear_interpolation(mask, h_repeat, repeat_mode, dst_blk_stride, v_r_offset, v_repeat, src0, src0offset_int,
                           src0offset, src1):
    m_num = 1
    n_num = 1

    VEC_NUM = 1

    @T.prim_func
    def main(
            src0: T.Tensor((src0.shape[0], src0.shape[1]), "float16"),
            src0_offset: T.Tensor((src0offset.shape[0], src0offset.shape[1]), "uint32"),
            src1: T.Tensor((src1.shape[0], src1.shape[1]), "float16"),
            dst: T.Tensor((src0.shape[0], src0.shape[1] // 2), "float16"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            src0_ub = T.alloc_ub((src0.shape[0] // VEC_NUM, src0.shape[1]), "float16")
            src0_offset_ub = T.alloc_ub((src0offset.shape[0] // VEC_NUM, src0offset.shape[1]), "uint32")
            src1_ub = T.alloc_ub((src1.shape[0] // VEC_NUM, src1.shape[1]), "float16")
            dst_ub = T.alloc_ub((src0.shape[0] // VEC_NUM, src0.shape[1] // 2), "float16")
            shared_tmp_buffer_ub = T.alloc_ub((src0.shape[0], src0.shape[1]), "uint8")

            T.copy(src0[0, 0], src0_ub)
            T.copy(src0_offset[0, 0], src0_offset_ub)
            T.copy(src1[0, 0], src1_ub)

            T.tile.bilinear_interpolation(dst_ub, src0_ub, src0_offset_ub, src1_ub, mask, h_repeat,
                                          repeat_mode, dst_blk_stride, v_r_offset, v_repeat, shared_tmp_buffer_ub)

            T.copy(dst_ub, dst[0, 0])

    return main


def fun_ref(a, b, c, hRepeat, vRepeat, repeatMode, vROffset):
    a = a.flatten()
    b = b.flatten()
    c = c.flatten()
    re = []

    if repeatMode:
        for k in range(vRepeat):
            s = torch.zeros(128, dtype=torch.float16).npu()  # 初始化累加器
            r = torch.zeros(128 * hRepeat, dtype=torch.float16).npu()
            for i in range(hRepeat):
                for j in range(8):
                    idx = b[k * 8 * hRepeat + i * 8 + j].to(torch.int64) // 32
                    r[i * 128 + j * 16: i * 128 + (j + 1) * 16] = a[idx * 16: (idx + 1) * 16] * c[
                        k * 8 * hRepeat + i * 8 + j]
                s += r[i * 128: (i + 1) * 128]
            re.append(s)
    else:
        for k in range(vRepeat):
            s = torch.zeros(128, dtype=torch.float16).npu()
            r = torch.zeros(128 * hRepeat, dtype=torch.float16).npu()
            for i in range(hRepeat):
                for j in range(8):
                    idx = b[k * 8 * hRepeat + i * 8 + j].to(torch.int64) // 32
                    r[i * 128 + j * 16: i * 128 + (j + 1) * 16] = a[idx * 16: (idx + 1) * 16] * c[k * hRepeat + i]
                s += r[i * 128: (i + 1) * 128]
            re.append(s)
    return torch.cat(re, dim=0).flatten()


def run_test_bilinear_interpolation(target):
    src0 = torch.arange(1, 513, dtype=torch.float16).reshape(1, -1).npu()
    src0offset_int = torch.arange(0, 1024, 32, dtype=torch.int64).reshape(1, -1).npu()
    src0offset = src0offset_int.to(dtype=torch.uint32)
    src1 = torch.arange(2, 18, dtype=torch.float16).reshape(1, -1).npu()
    hRepeat = 2
    mask1 = 128
    repeatMode = False
    dstBlkStride = 1
    vROffset = 128
    vRepeat = 2
    mask0 = 0
    func = bilinear_interpolation(mask1, hRepeat, repeatMode, dstBlkStride, vROffset, vRepeat, src0, src0offset_int,
                                  src0offset, src1)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch.npu.synchronize()

    c = func(src0, src0offset, src1)

    if vROffset > 128:
        outsize = vRepeat * vROffset
    else:
        outsize = vRepeat * 128

    out = fun_ref(src0, src0offset, src1, hRepeat, vRepeat, repeatMode, vROffset)

    out_real = torch.zeros(outsize, dtype=torch.float16).npu()
    if mask0 == 0:
        for i in range(vRepeat):
            n = mask1 // 16
            l = mask1 % 16
            for j in range(n):
                out_real[i * vROffset + j * 16: i * vROffset + (j + 1) * 16] = out[
                                                                               i * 128 + j * 16: i * 128 + (j + 1) * 16]
            out_real[i * vROffset + n * 16: i * vROffset + n * 16 + l] = out[i * 128 + n * 16: i * 128 + n * 16 + l]

    ref_c = out_real[:vRepeat * 128].unsqueeze(0)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc"])
def test_bilinear_interpolation(target):
    run_test_bilinear_interpolation(target)


def bitwise_lshift(M, N, block_M, block_N, scalarvalue, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.bitwise_lshift(b_ub, a_ub, scalarvalue)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_bitwise_lshift(M, N, block_M, block_N, scalarvalue, target):
    func = bitwise_lshift(M, N, block_M, block_N, scalarvalue)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randint(low=1, high=101, size=(M, N), dtype=torch.int32).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = pow(2, scalarvalue) * a

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_lshift(target, shape):
    M, N = shape
    scalarvalue = random.randint(1, 32)
    run_test_bitwise_lshift(M, N, 128, 256, scalarvalue=scalarvalue, target=target)


def bitwise_not(M, N, block_M, block_N, dtype="int16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.bitwise_not(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_bitwise_not(M, N, block_M, block_N, target):
    func = bitwise_not(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randint(0, 10, (M, N), dtype=torch.int16).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = ~a

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_not(target, shape):
    M, N = shape
    run_test_bitwise_not(M, N, 128, 256, target=target)


def bitwise_rshift(M, N, block_M, block_N, scalarvalue, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.bitwise_rshift(b_ub, a_ub, scalarvalue)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_bitwise_rshift(M, N, block_M, block_N, scalarvalue, target):
    func = bitwise_rshift(M, N, block_M, block_N, scalarvalue)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randint(low=1, high=101, size=(M, N), dtype=torch.int32).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = a // pow(2, scalarvalue)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_rshift(target, shape):
    M, N = shape
    scalarvalue = random.randint(1, 32)
    run_test_bitwise_rshift(M, N, 128, 256, scalarvalue=scalarvalue, target=target)


def bitwise_xor(M, N, block_M, block_N, dtype="int16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            tmp_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.bitwise_xor(c_ub, a_ub, b_ub, tmp_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_bitwise_xor(M, N, block_M, block_N, target):
    func = bitwise_xor(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randint(0, 10, (M, N), dtype=torch.int16).npu()
    b = torch.randint(0, 10, (M, N), dtype=torch.int16).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = torch.bitwise_xor(a, b)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_xor(target, shape):
    M, N = shape
    run_test_bitwise_xor(M, N, 128, 256, target)


def block_reduce_max(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum,
                     dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N // dataBlockNum), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N // dataBlockNum), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.block_reduce_max(b_ub, a_ub, repeat, mask, dstRepStride, srcBlkStride, srcRepStride)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N // dataBlockNum])

    return main


def run_test_block_reduce_max(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
                              dataBlockNum, target):
    func = block_reduce_max(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
                            dataBlockNum)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    num_groups = M * N // dataBlockNum
    ref_b = torch.zeros((1, num_groups)).to(torch.float16)
    a_flag = a.reshape(-1)
    for i in range(num_groups):
        start = i * dataBlockNum
        end = start + dataBlockNum
        group = a_flag[start:end]
        max_val = torch.max(group).item()
        ref_b[0, i] = max_val
    ref_b = ref_b.reshape(M, N // dataBlockNum)
    ref_b = ref_b.npu().to(dtype=torch.float16)
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc"])
def test_block_reduce_max(target):
    M = 2
    N = 512
    block_M = 2
    block_N = 128
    dataBlockNum = 16
    mask = 128
    repeat = 1
    dstRepStride = 1
    srcBlkStride = 1
    srcRepStride = 8
    run_test_block_reduce_max(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
                              dataBlockNum, target)


def block_reduce_min(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum,
                     dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N // dataBlockNum), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N // dataBlockNum), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.block_reduce_min(b_ub, a_ub, repeat, mask, dstRepStride, srcBlkStride, srcRepStride)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N // dataBlockNum])

    return main


def run_test_block_reduce_min(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
                              dataBlockNum, target):
    func = block_reduce_min(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
                            dataBlockNum)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    num_groups = M * N // dataBlockNum
    ref_b = torch.zeros((1, num_groups)).to(torch.float16)
    a_flag = a.reshape(-1)
    for i in range(num_groups):
        start = i * dataBlockNum
        end = start + dataBlockNum
        group = a_flag[start:end]
        min_val = torch.min(group).item()
        ref_b[0, i] = min_val
    ref_b = ref_b.reshape(M, N // dataBlockNum)
    ref_b = ref_b.npu().to(dtype=torch.float16)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc"])
def test_block_reduce_min(target):
    M = 2
    N = 512
    block_M = 2
    block_N = 128
    dataBlockNum = 16
    mask = 128
    repeat = 1
    dstRepStride = 1
    srcBlkStride = 1
    srcRepStride = 8
    run_test_block_reduce_min(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
                              dataBlockNum, target)


def block_reduce_sum(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride, dataBlockNum,
                     dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N // dataBlockNum), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N // dataBlockNum), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.block_reduce_sum(b_ub, a_ub, repeat, mask, dstRepStride, srcBlkStride, srcRepStride)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N // dataBlockNum])

    return main


def run_test_block_reduce_sum(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
                              dataBlockNum, target):
    func = block_reduce_sum(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
                            dataBlockNum)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    num_groups = M * N // dataBlockNum
    ref_b = torch.zeros((1, num_groups)).to(torch.float16)
    a_flag = a.reshape(-1)
    for i in range(num_groups):
        start = i * dataBlockNum
        end = start + dataBlockNum
        group = a_flag[start:end]
        sum_val = torch.sum(group).item()
        ref_b[0, i] = sum_val
    ref_b = ref_b.reshape(M, N // dataBlockNum)
    ref_b = ref_b.npu().to(dtype=torch.float16)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc"])
def test_block_reduce_sum(target):
    M = 2
    N = 512
    block_M = 2
    block_N = 128
    dataBlockNum = 16
    mask = 128
    repeat = 1
    dstRepStride = 1
    srcBlkStride = 1
    srcRepStride = 8
    run_test_block_reduce_sum(M, N, block_M, block_N, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
                              dataBlockNum, target)


def cast(M, N, block_M, block_N, mode, count):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 1

    @T.prim_func
    def main(
            A: T.Tensor((M, N), "float"),
            B: T.Tensor((M, N), "float"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "float")
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "float")

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.cast(b_ub, a_ub, mode, count)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_cast(M, N, block_M, block_N, mode, count, target):
    func = cast(M, N, block_M, block_N, mode, count)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    # without setdeqscale
    a = torch.full((M, N), 0.5, dtype=torch.float).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.full((M, N), 0.0, dtype=torch.float).npu()

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(64, 64)])
def test_cast(target, shape):
    M, N = shape
    run_test_cast(M, N, 16, 16, "CAST_RINT", 4096, target)


def cast_scale(M, N, block_M, block_N, mode, count, scale):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 1

    @T.prim_func
    def main(
            A: T.Tensor((M, N), "int32"),
            B: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "int32")
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "float16")

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.set_deq_scale(scale)

            T.tile.cast(b_ub, a_ub, mode, count)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_cast_scale(M, N, block_M, block_N, mode, count, scale, target):
    func = cast_scale(M, N, block_M, block_N, mode, count, scale)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    # should setdeqscale
    a = torch.full((M, N), 1, dtype=torch.int32).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.full((M, N), 1.0, dtype=torch.float16).npu()

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(64, 64)])
def test_cast_scale(target, shape):
    M, N = shape
    run_test_cast_scale(M, N, 16, 16, "CAST_RINT", 4096, 1.0, target)


def clamp(size, max_val, min_val, dtype="float16"):
    block_size = 64 * 1024
    loop_num = (size + block_size - 1) // block_size

    VEC_NUM = 2

    @T.prim_func
    def main(
            input: T.Tensor([size], dtype),
            output: T.Tensor([size], dtype),
    ):
        with T.Kernel(loop_num, is_npu=True) as (cid, vid):
            idx = cid

            in_ub = T.alloc_ub((block_size // VEC_NUM), dtype)
            tmp_ub = T.alloc_ub((block_size // VEC_NUM), "uint8")

            T.copy(input[idx * block_size // VEC_NUM], in_ub)
            for i in range(size):
                T.tile.clamp(in_ub, in_ub, tmp_ub, min_val, max_val, block_size // VEC_NUM)

            T.copy(in_ub, output[idx * block_size // VEC_NUM])

    return main


def run_test_clamp(size, max_val, min_val, thresh, target):
    if min_val > max_val:
        max_val, min_val = min_val, max_val

    func = clamp(size, max_val, min_val, "float16")
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = (torch.rand([size]) - 0.5) * 2 * thresh
    a = a.half().npu()

    b = func(a)
    ref_b = torch.clamp(a, min_val, max_val)

    torch.testing.assert_close(b, ref_b, rtol=0, atol=0)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_clamp(target):
    size = 10
    thresh = 10000
    max_val = random.uniform(-1 * thresh, thresh)
    min_val = random.uniform(-1 * thresh, thresh)
    run_test_clamp(size, max_val, min_val, thresh, target)


def compare(M, N, block_M, block_N, mode, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N // 8), "uint8"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N // 8), "uint8")

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.compare(c_ub, a_ub, b_ub, mode)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N // 8])

    return main


def compare_and_set_bits(A, B, C):
    """
    compare A to B, and set C's element according to comparison result
    Args:
        A: torch.Tensor, shape (128, 128), float32
        B: torch.Tensor, shape (128, 128), float32
        C: torch.Tensor, shape (128, 16), uint8

    Returns:
        C: torch.Tensor, shape (128, 16), uint8
    """
    assert A.dtype == torch.float32, "A must be float32"
    assert B.dtype == torch.float32, "B must be float32"
    assert C.dtype == torch.uint8, "C must be uint8"

    # set mask's according bit to True when A < B, or to False
    mask = A < B  # shape: (128, 128)

    C_result = torch.zeros(C.size(0), C.size(1), dtype=torch.uint8, device=A.device)

    for i in range(C.size(0)):
        for j in range(C.size(1)):
            start_bit = j * 8
            end_bit = start_bit + 8

            bits = mask[i, start_bit:end_bit]  # shape: (8,)

            byte_value = 0
            for k in range(8):
                if bits[k]:
                    byte_value |= (1 << k)

            C_result[i, j] = byte_value

    return C_result


def run_test_compare(M, N, block_M, block_N, mode, target):
    torch.npu.config.allow_internal_format = True
    # torch.set_printoptions(threshold=np.inf)

    func = compare(M, N, block_M, block_N, mode)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.zeros(M, N).npu()
    b = torch.ones(M, N).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = torch.zeros(M, N // 8, dtype=torch.uint8).npu()
    ref_c = compare_and_set_bits(a, b, ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(256, 256)])
def test_compare(target, shape):
    M, N = shape
    run_test_compare(M, N, 128, 256, "LT", target)


def compare_scalar(M, N, block_M, block_N, mode, b_scalar, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N // 8), "uint8")
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N // 8), "uint8")

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.compare(c_ub, a_ub, b_scalar, mode)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N // 8])

    return main


def compare_with_scalar_and_set_bits(A, b, C):
    """
    compare A's element to b, and set C's element according to comparison result
    Args:
        A: torch.Tensor, shape (128, 128), float32
        b: float, scalar value
        C: torch.Tensor, shape (128, 16), uint8

    Returns:
        C: torch.Tensor, shape (128, 16), uint8
    """
    assert A.dtype == torch.float32, "A must be float32"
    assert C.dtype == torch.uint8, "C must be uint8"

    # set mask position to True or False(position set to True when A < b, else False)
    mask = A < b  # shape: (128, 128)

    C_result = torch.zeros(C.size(0), C.size(1), dtype=torch.uint8, device=A.device)

    for i in range(C.size(0)):
        for j in range(C.size(1)):
            start_bit = j * 8
            end_bit = start_bit + 8

            bits = mask[i, start_bit:end_bit]  # shape: (8,)

            byte_value = 0
            for k in range(8):
                if bits[k]:
                    byte_value |= (1 << k)

            C_result[i, j] = byte_value

    return C_result


def run_test_compare_scalar(M, N, block_M, block_N, mode, b_scalar, target):
    torch.npu.config.allow_internal_format = True
    # torch.set_printoptions(threshold=np.inf)

    func = compare_scalar(M, N, block_M, block_N, mode, b_scalar)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)
    a = torch.zeros(M, N).npu()

    torch.npu.synchronize()

    c = func(a)

    ref_c = torch.zeros(M, N // 8, dtype=torch.uint8).npu()
    ref_c = compare_with_scalar_and_set_bits(a, b_scalar, ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(256, 256)])
def test_compare_scalar(target, shape):
    M, N = shape
    block_M = 128
    block_N = 256
    mode = "LT"
    b_scalar = 1.0
    run_test_compare_scalar(M, N, block_M, block_N, mode, b_scalar, target)


def cos(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
            A: T.Tensor([M, N], dtype),
            B: T.Tensor([M, N], dtype),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            b = T.alloc_ub([sub_block_M, block_N], dtype)
            tmp = T.alloc_ub([2 * sub_block_M * block_N], "uint8")

            T.copy(A[bx * block_M + vid * sub_block_M: bx * block_M + (vid + 1) * sub_block_M,
                   by * block_N: (by + 1) * block_N], a)  # Load input
            T.tile.cos(b, a, tmp)  # Compute cos
            T.copy(b, B[bx * block_M + vid * sub_block_M: bx * block_M + (vid + 1) * sub_block_M,
                      by * block_N: (by + 1) * block_N])  # Store output

    return main


def run_test_cos(target):
    test_configs = [
        (1024, 1024, 128, 128),
    ]

    for M, N, block_M, block_N in test_configs:
        func = cos(M, N, block_M, block_N, dtype="float")
        func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)
        a = torch.randn(M, N).npu()
        b = func(a)
        ref_b = torch.cos(a)
        torch.testing.assert_close(b, ref_b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("target", ["ascendc"])
def test_cos(target):
    run_test_cos(target)


def createvecindex(M, N, block_M, block_N, firstValue, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 1

    @T.prim_func
    def main(
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.tile.createvecindex(c_ub, firstValue)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_createvecindex(M, N, block_M, block_N, firstValue, target):
    func = createvecindex(M, N, block_M, block_N, firstValue)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch.npu.synchronize()

    c = func()

    ref_c = torch.arange(start=firstValue, end=firstValue + block_N, dtype=torch.int32).reshape(M, N)
    ref_c = ref_c.npu()

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_createvecindex(target):
    M = 1
    N = 1024
    block_M = 1
    block_N = 1024
    firstValue = 0
    run_test_createvecindex(M, N, block_M, block_N, firstValue, target)


def vec_div(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.div(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_vec_div(M, N, block_M, block_N, target):
    func = vec_div(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = a / b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_div(target, shape):
    M, N = shape
    run_test_vec_div(M, N, 64, 128, target=target)


def exp(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.exp(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_exp(M, N, block_M, block_N, target):
    func = exp(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.exp(a)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_exp(target, shape):
    M, N = shape
    run_test_exp(M, N, 128, 256, target=target)


def fill(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)

            T.tile.fill(a_ub, 10.0)
            T.copy(a_ub, A[bx * block_M, by * block_N])

    return main


def run_test_fill(M, N, block_M, block_N, target):
    func = fill(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch.npu.synchronize()

    b = func()

    ref_b = torch.full((M, N), 10.0).npu()

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_fill(target, shape):
    M, N = shape
    run_test_fill(M, N, 64, 32, target=target)


def gather(M, N, block_M, block_N, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), "uint32"),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "uint32")
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.gather(c_ub, a_ub, b_ub, 0)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def generate_golden_gather(a, b):
    result = torch.zeros(a.size(0), a.size(1), dtype=torch.int32)
    for i in range(a.size(0)):
        tmp_result = torch.zeros(1, a.size(1), dtype=torch.int32)
        for j in range(a.size(1)):
            index = b[i, j].to(torch.int32) / 4  # 4: sizeof(int)
            index = index.long()
            tmp_result[0, j] = a[i, index]
        result[i:] = tmp_result
    return result


def run_test_gather(M, N, block_M, block_N, target):
    func = gather(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.arange(N, dtype=torch.int32).unsqueeze(0).expand(M, -1).npu()
    all_multiples = torch.arange(0, 4 * N, 4)  # 4: sizeof(int)
    random_indices = torch.randperm(len(all_multiples))[:N]
    random_multiples = all_multiples[random_indices].to(torch.uint32)
    tmp_tensor = random_multiples.reshape(1, N)
    tensor_cpu = tmp_tensor.repeat(M, 1)
    b = tensor_cpu.npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = generate_golden_gather(a, b).npu()

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc"])
@pytest.mark.parametrize("shape", [(128, 1024)])
def test_gather(target, shape):
    M, N = shape
    block_M = 16
    block_N = N
    run_test_gather(M, N, block_M, block_N, target)


def gatherb(M, N, block_M, block_N, b_len, repeat_time, dtype="uint16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, b_len), "uint32"),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, b_len), "uint32")
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * b_len], b_ub)

            T.tile.gatherb(c_ub, a_ub, b_ub, repeat_time, 1, 8)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def generate_golden_gatherb(a, b, N, max_index):
    result = torch.zeros(1, N, dtype=torch.int16)
    for i in range(max_index):
        start = b[0, i].to(torch.int32) // 2  # 2: sizeof(uint16)
        for j in range(16):  # 16: 8 * 32 // (2 * 8) 一个DataBlock有16个数
            result[0, i * 16 + j] = start + j
    return result


def run_test_gatherb(M, N, block_M, block_N, b_len, repeat_time, target):
    func = gatherb(M, N, block_M, block_N, b_len, repeat_time)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.arange(N, dtype=torch.int16).to(torch.uint16).unsqueeze(0).expand(M, -1).npu()
    tmp_tensor = torch.zeros(1, b_len, dtype=torch.uint32)
    for i in range(b_len):
        tmp_tensor[0, b_len - 1 - i] = i * 32
    b = tmp_tensor.expand(M, -1).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = generate_golden_gatherb(a, b, N, b_len).expand(M, -1).npu()

    torch.testing.assert_close(c.to(torch.int16), ref_c.to(torch.int16), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(64, 1024)])
def test_gatherb(target, shape):
    M, N = shape
    block_M = 2
    block_N = N
    b_len = (N * 2 - 1) // 32 + 1
    repeat_time = N * 2 // (32 * 8)
    run_test_gatherb(M, N, block_M, block_N, b_len, repeat_time, target)


def leaky_relu(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.leaky_relu(b_ub, a_ub, 2.0)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_leaky_relu(M, N, block_M, block_N, target):
    func = leaky_relu(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu()

    torch.npu.synchronize()

    b = func(a)

    leaky_relu_golden = nn.LeakyReLU(negative_slope=2.0)
    ref_b = leaky_relu_golden(a)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_leaky_relu(target, shape):
    M, N = shape
    block_M = 128
    block_N = 256
    run_test_leaky_relu(M, N, block_M, block_N, target)


def ln(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.ln(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_ln(M, N, block_M, block_N, target):
    func = ln(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = abs(torch.randn(M, N).npu())

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.log(a)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_ln(target, shape):
    M, N = shape
    block_M = 128
    block_N = 256
    run_test_ln(M, N, block_M, block_N, target)


def vec_max(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.max(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_vec_max(M, N, block_M, block_N, target):
    func = vec_max(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = torch.max(a, b)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_max(target, shape):
    M, N = shape
    run_test_vec_max(M, N, 64, 128, target=target)


def vec_maxs(M, N, block_M, block_N, scalar, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.tile.max(b_ub, a_ub, scalar)
            T.copy(b_ub, B[bx * block_M, by * block_N])

    return main


def run_test_vec_maxs(M, N, block_M, block_N, scalar, target):
    func = vec_maxs(M, N, block_M, block_N, scalar)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).float().npu()
    a = a * 50

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.clamp_min(a, scalar)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_maxs(target, shape):
    M, N = shape
    scalar = 2.0
    run_test_vec_maxs(M, N, 64, 32, scalar=scalar, target=target)


def vec_min(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.min(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_vec_min(M, N, block_M, block_N, target):
    func = vec_min(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = torch.min(a, b)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_min(target, shape):
    M, N = shape
    run_test_vec_min(M, N, 64, 128, target=target)


def vec_mins(M, N, block_M, block_N, scalar, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.tile.min(b_ub, a_ub, scalar)
            T.copy(b_ub, B[bx * block_M, by * block_N])

    return main


def run_test_vec_mins(M, N, block_M, block_N, scalar, target):
    func = vec_mins(M, N, block_M, block_N, scalar)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).float().npu()
    a = a * 50

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.clamp_max(a, scalar)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_mins(target, shape):
    M, N = shape
    scalar = 2.0
    run_test_vec_mins(M, N, 64, 32, scalar=scalar, target=target)


def vec_mul(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.mul(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_vec_mul(M, N, block_M, block_N, target):
    func = vec_mul(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = a * b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_mul(target, shape):
    M, N = shape
    run_test_vec_mul(M, N, 64, 128, target=target)


def vec_muls(M, N, block_M, block_N, scalar, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.tile.mul(b_ub, a_ub, scalar)
            T.copy(b_ub, B[bx * block_M, by * block_N])

    return main


def run_test_vec_muls(M, N, block_M, block_N, scalar, target):
    func = vec_muls(M, N, block_M, block_N, scalar)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).float().npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = a * 2.0

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_muls(target, shape):
    M, N = shape
    scalar = 2.0
    run_test_vec_muls(M, N, 64, 32, scalar=scalar, target=target)


def bitwise_or(M, N, block_M, block_N, dtype="int16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.bitwise_or(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_bitwise_or(M, N, block_M, block_N, target):
    func = bitwise_or(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randint(0, 10, (M, N), dtype=torch.int16).npu()
    b = torch.randint(0, 10, (M, N), dtype=torch.int16).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = a | b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_or(target, shape):
    M, N = shape
    run_test_bitwise_or(M, N, 128, 256, target=target)


def vec_pow(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            # Temporary buffer allocated as uint8
            tmp = T.alloc_ub((2 * (block_M // VEC_NUM), block_N), "uint8")

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.pow(c_ub, a_ub, b_ub, tmp)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_pow(M, N, block_M, block_N, target):
    func = vec_pow(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.rand(M, N).npu() + 0.5  # Base must be positive for pow()
    b = torch.rand(M, N).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = torch.pow(a, b)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_pow(target, shape):
    M, N = shape
    run_test_pow(M, N, 128, 128, target=target)


def reciprocal(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.reciprocal(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_reciprocal(M, N, block_M, block_N, target):
    func = reciprocal(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.rand(M, N).npu() * 0.9 + 0.1

    torch.npu.synchronize()

    b = func(a)

    ref_b = 1 / a

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_reciprocal(target, shape):
    M, N = shape
    run_test_reciprocal(M, N, 128, 128, target=target)


def relu(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.relu(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_relu(M, N, block_M, block_N, target):
    func = relu(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.relu(a)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_relu(target, shape):
    M, N = shape
    run_test_relu(M, N, 128, 256, target=target)


def rsqrt(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.rsqrt(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_rsqrt(M, N, block_M, block_N, target):
    func = rsqrt(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.rsqrt(a)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2, equal_nan=True)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_rsqrt(target, shape):
    M, N = shape
    run_test_rsqrt(M, N, 128, 256, target=target)


def vec_select(M, N, block_M, block_N, mode, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            MASK: T.Tensor((M, N // 8), "uint8"),
            C: T.Tensor((M, N), dtype),

    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            selmask_ub = T.alloc_ub((block_M // VEC_NUM, block_N // 8), "uint8")
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
            T.copy(MASK[bx * block_M + vid * block_M // VEC_NUM, by * block_N // 8], selmask_ub)

            T.tile.select(c_ub, selmask_ub, a_ub, b_ub, mode)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def select_by_mask_bits_select(A, B, M):
    """
    select element from A or B by bits in M, then set it to C

    Args:
        A: torch.Tensor, shape (128, 128), float32
        B: torch.Tensor, shape (128, 128), float32
        M: torch.Tensor, shape (128, 16), uint8,

    Returns:
        C: torch.Tensor, shape (128, 128), float32
    """
    assert A.dtype == torch.float32, "A must be float32"
    assert B.dtype == torch.float32, "B must be float32"
    assert M.dtype == torch.uint8, "M must be uint8"

    C = torch.zeros_like(A).npu()
    M_cpu = M.cpu()

    for i in range(M_cpu.size(0)):
        for j in range(M_cpu.size(1)):
            byte_val = M_cpu[i, j]

            start_col = j * 8
            end_col = start_col + 8

            for bit_pos in range(8):
                col = start_col + bit_pos
                if (byte_val >> bit_pos) & 1:
                    C[i, col] = A[i, col]
                else:
                    C[i, col] = B[i, col]

    return C


def run_test_vec_select(M, N, block_M, block_N, mode, target):
    torch.npu.config.allow_internal_format = True
    # torch.set_printoptions(threshold=np.inf)

    func = vec_select(M, N, block_M, block_N, mode)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)
    a = torch.zeros(M, N).npu()
    b = torch.ones(M, N).npu()
    m = torch.full((M, N // 8), 0xF, dtype=torch.uint8).npu()

    torch.npu.synchronize()

    c = func(a, b, m)

    ref_c = select_by_mask_bits_select(a, b, m)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc"])
@pytest.mark.parametrize("shape", [(256, 256)])
def test_vec_select(target, shape):
    M, N = shape
    block_M = 64
    block_N = 128
    mode = "VSEL_CMPMASK_SPR"
    run_test_vec_select(M, N, block_M, block_N, mode, target)


def vec_select_scalar(M, N, block_M, block_N, mode, b_scalar, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            MASK: T.Tensor((M, N // 8), "uint8"),
            C: T.Tensor((M, N), dtype),

    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            selmask_ub = T.alloc_ub((block_M // VEC_NUM, block_N // 8), "uint8")
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(MASK[bx * block_M + vid * block_M // VEC_NUM, by * block_N // 8], selmask_ub)

            T.tile.select(c_ub, selmask_ub, a_ub, b_scalar, mode)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def select_by_mask_bits_select_scalar(A, b, M):
    """
    select value by mask bits of M from A or B to assign to C

    Args:
        A: torch.Tensor, shape (128, 128), float32
        B: torch.Tensor, shape (128, 128), float32
        M: torch.Tensor, shape (128, 16), uint8, 每个元素存储8个bit位

    Returns:
        C: torch.Tensor, shape (128, 128), float32
    """
    assert A.dtype == torch.float32, "A must be float32"
    assert M.dtype == torch.uint8, "M must be uint8"

    C = torch.zeros_like(A).npu()
    M_cpu = M.cpu()

    for i in range(M_cpu.size(0)):
        for j in range(M_cpu.size(1)):
            byte_val = M_cpu[i, j]

            start_col = j * 8
            end_col = start_col + 8

            for bit_pos in range(8):
                col = start_col + bit_pos
                if (byte_val >> bit_pos) & 1:
                    C[i, col] = A[i, col]
                else:
                    C[i, col] = b

    return C


def run_test_vec_select_scalar(M, N, block_M, block_N, mode, b_scalar, target):
    torch.npu.config.allow_internal_format = True
    # torch.set_printoptions(threshold=np.inf)

    func = vec_select_scalar(M, N, block_M, block_N, mode, b_scalar)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.zeros(M, N).npu()
    m = torch.full((M, N // 8), 0xF, dtype=torch.uint8).npu()

    torch.npu.synchronize()

    c = func(a, m)

    ref_c = select_by_mask_bits_select_scalar(a, b_scalar, m)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc"])
@pytest.mark.parametrize("shape", [(256, 256)])
def test_vec_select_scalar(target, shape):
    M, N = shape
    block_M = 64
    block_N = 128
    mode = "VSEL_TENSOR_SCALAR_MODE"
    b_scalar = 1.0
    run_test_vec_select_scalar(M, N, block_M, block_N, mode, b_scalar, target)


def sin(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
            A: T.Tensor([M, N], dtype),
            B: T.Tensor([M, N], dtype),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            b = T.alloc_ub([sub_block_M, block_N], dtype)
            tmp = T.alloc_ub([2 * sub_block_M * block_N], "uint8")
            T.copy(A[bx * block_M + vid * sub_block_M: bx * block_M + (vid + 1) * sub_block_M,
                   by * block_N: (by + 1) * block_N], a)  # Load input
            T.tile.sin(b, a, tmp)  # Compute sin
            T.copy(b, B[bx * block_M + vid * sub_block_M: bx * block_M + (vid + 1) * sub_block_M,
                      by * block_N: (by + 1) * block_N])  # Store output

    return main


def run_test_sin(target):
    test_configs = [
        (1024, 1024, 128, 128),
    ]

    for M, N, block_M, block_N in test_configs:
        func = sin(M, N, block_M, block_N, dtype="float")
        func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)
        a = torch.randn(M, N).npu()
        b = func(a)
        ref_b = torch.sin(a)
        torch.testing.assert_close(b, ref_b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("target", ["ascendc"])
def test_sin(target):
    run_test_sin(target)


def sort32(M, N, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), "float"),
            B: T.Tensor((M, N), "uint32"),
            C: T.Tensor((M, 2 * N), "float"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N // VEC_NUM), "float")
            b_ub = T.alloc_ub((block_M, block_N // VEC_NUM), "uint32")
            c_ub = T.alloc_ub((block_M, 2 * block_N // VEC_NUM), "float")

            T.copy(A[bx * block_M, by * block_N + vid * block_N // VEC_NUM], a_ub)
            T.copy(B[bx * block_M, by * block_N + vid * block_N // VEC_NUM], b_ub)

            T.tile.sort32(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M, 2 * (by * block_N + vid * block_N // VEC_NUM)])

    return main


def run_test_sort32(M, N, block_M, block_N, target):
    func = sort32(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randint(low=1, high=101, size=(M, N), dtype=torch.float).npu()
    b = torch.zeros((M, N), dtype=torch.uint32).npu()

    torch.npu.synchronize()

    c = func(a, b)
    # 计算ref_c
    group_size = 32
    total_elements = M * N
    b_ref = torch.zeros((M, N), dtype=torch.int32, device="npu")
    src0_flat = a.flatten()
    src1_flat = b_ref.flatten()

    total_groups = (total_elements + group_size - 1) // group_size
    for i in range(total_groups):
        start = i * group_size
        end = min((i + 1) * group_size, total_elements)

        group_src0 = src0_flat[start:end]
        group_src1 = src1_flat[start:end]
        sorted_indices = torch.argsort(group_src0, descending=True)

        src0_flat[start:end] = group_src0[sorted_indices]
        src1_flat[start:end] = group_src1[sorted_indices]

    sorted_src0 = src0_flat.reshape(M, N)
    sorted_src1 = src1_flat.reshape(M, N)

    ref_c = torch.empty((M, 2 * N), dtype=torch.float)
    ref_c[:, ::2] = sorted_src0
    ref_c[:, 1::2] = sorted_src1

    torch.testing.assert_close(c, ref_c.npu(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_sort32(target, shape):
    M, N = shape
    block_M = 64
    block_N = 128  # block_M,block_N与repeatimes有关，repeatimes要求范围[0,255]
    run_test_sort32(M, N, block_M, block_N, target)


def sqrt(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.sqrt(b_ub, a_ub)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_sqrt(M, N, block_M, block_N, target):
    func = sqrt(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.rand(M, N).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = torch.sqrt(a)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_sqrt(target, shape):
    M, N = shape
    run_test_sqrt(M, N, 128, 256, target=target)


def vec_sub(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            T.tile.sub(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_vec_sub(M, N, block_M, block_N, target):
    func = vec_sub(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()

    torch.npu.synchronize()

    c = func(a, b)

    ref_c = a - b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_sub(target, shape):
    M, N = shape
    run_test_vec_sub(M, N, 64, 128, target=target)


def vec_subs(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor[(M, N), dtype],
            B: T.Tensor[(M, N), dtype],
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            T.tile.sub(b_ub, a_ub, 3.0)

            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_vec_subs(M, N, block_M, block_N, target):
    func = vec_subs(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu()

    torch.npu.synchronize()

    b = func(a)

    ref_b = a - 3.0

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_subs(target, shape):
    M, N = shape
    run_test_vec_subs(M, N, 128, 256, target=target)


def transpose(M, N, block_M, block_N, dtype="int16"):
    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((M, N), dtype)

            T.copy(A, a_ub)

            T.tile.transpose(b_ub, a_ub)

            T.copy(b_ub, B)

    return main


def run_test_transpose(M, N, block_M, block_N, target):
    func = transpose(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu().to(torch.int16)

    torch.npu.synchronize()

    b = func(a)

    ref_b = a.T

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(16, 16)])
def test_transpose(target, shape):
    M, N = shape
    run_test_transpose(M, N, 16, 16, target)


def wholereducemax(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride,
                   dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, 2 * N // mask), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, 2 * block_N // mask), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.wholereducemax(b_ub, a_ub, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * 2 * block_N // mask])

    return main


def run_test_wholereducemax(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride,
                            target):
    func = wholereducemax(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    num_groups = M * N // mask
    ref_b = torch.zeros((1, 2 * num_groups)).to(torch.float16)
    a_flag = a.reshape(-1)
    for i in range(num_groups):
        start = i * mask
        end = start + mask
        group = a_flag[start:end]
        max_val = torch.max(group).item()
        max_idx_in_group = torch.argmax(group).item()
        result = torch.tensor([max_idx_in_group], dtype=torch.uint16).view(torch.float16).float().item()
        ref_b[0, 2 * i] = max_val
        ref_b[0, 2 * i + 1] = result
    ref_b = ref_b.reshape(M, 2 * N // mask)
    ref_b = ref_b.npu().to(dtype=torch.float16)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc"])
def test_wholereducemax(target):
    M = 2
    N = 512
    block_M = 2
    block_N = 128
    mask = 64
    repeatTimes = 2
    dstRepStride = 1
    srcBlkStride = 1
    srcRepStride = 4
    run_test_wholereducemax(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, target)


def wholereducemin(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride,
                   dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, 2 * N // mask), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, 2 * block_N // mask), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.wholereducemin(b_ub, a_ub, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * 2 * block_N // mask])

    return main


def run_test_wholereducemin(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride,
                            target):
    func = wholereducemin(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    num_groups = M * N // mask
    ref_b = torch.zeros((1, 2 * num_groups)).to(torch.float16)
    a_flag = a.reshape(-1)
    for i in range(num_groups):
        start = i * mask
        end = start + mask
        group = a_flag[start:end]
        min_val = torch.min(group).item()
        min_idx_in_group = torch.argmin(group).item()
        result = torch.tensor([min_idx_in_group], dtype=torch.uint16).view(torch.float16).float().item()
        ref_b[0, 2 * i] = min_val
        ref_b[0, 2 * i + 1] = result
    ref_b = ref_b.reshape(M, 2 * N // mask)
    ref_b = ref_b.npu().to(dtype=torch.float16)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc"])
def test_wholereducemin(target):
    M = 2
    N = 512
    block_M = 2
    block_N = 128
    mask = 64
    repeatTimes = 2
    dstRepStride = 1
    srcBlkStride = 1
    srcRepStride = 4
    run_test_wholereducemin(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, target)


def wholereducesum(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride,
                   dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N // mask), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N // mask), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.wholereducesum(b_ub, a_ub, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N // mask])

    return main


def run_test_wholereducesum(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride,
                            target):
    func = wholereducesum(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N, dtype=torch.float16).npu()

    torch.npu.synchronize()

    b = func(a)

    num_groups = M * N // mask
    ref_b = torch.zeros((1, num_groups))
    a_flag = a.reshape(-1)
    for i in range(num_groups):
        start = i * mask
        end = start + mask
        group = a_flag[start:end]
        sum_val = torch.sum(group).item()
        ref_b[0, i] = sum_val
    ref_b = ref_b.reshape(M, N // mask)
    ref_b = ref_b.npu().to(dtype=torch.float16)

    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc"])
def test_wholereducesum(target):
    M = 2
    N = 512
    block_M = 2
    block_N = 128
    mask = 64
    repeatTimes = 2
    dstRepStride = 1
    srcBlkStride = 1
    srcRepStride = 4
    run_test_wholereducesum(M, N, block_M, block_N, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, target)


def generate_arithmetic_progression(N, block_size, dtype="int32"):
    num_blocks = N // block_size

    @T.prim_func
    def main(
            output: T.Tensor((N,), dtype),
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, _):
            start_idx = cid * block_size

            seq_ub = T.alloc_shared((block_size,), dtype)

            T.tile.arith_progression(seq_ub, start_idx, 1, block_size)

            T.copy(seq_ub, output[start_idx])

    return main


def run_test_generate_arithmetic_progression(N, block_size, target):
    func = generate_arithmetic_progression(N, block_size)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    output = torch.zeros(N, dtype=torch.int32).npu()

    torch.npu.synchronize()

    result = func(output)

    ref_result = torch.arange(0, N, dtype=torch.int32).npu()

    torch.testing.assert_close(result, ref_result, rtol=0, atol=0)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [1024])
def test_generate_arithmetic_progression(target, shape):
    N = shape
    block_size = 64
    run_test_generate_arithmetic_progression(N, block_size, target)


if __name__ == "__main__":
    pytest.main()


def vec_fused_mul_add(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
            D: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
            T.copy(C[bx * block_M + vid * block_M // VEC_NUM, by * block_N], c_ub)

            # a_ub = b_ub * a_ub + c_ub
            T.tile.fused_mul_add(a_ub, b_ub, c_ub)

            T.copy(a_ub, D[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_fused_mul_add(M, N, block_M, block_N, target):
    func = vec_fused_mul_add(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()
    c = torch.randn(M, N).npu()

    torch.npu.synchronize()

    d = func(a, b, c)

    ref_d = b * a + c
    torch.testing.assert_close(d, ref_d, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_fused_mul_add(target, shape):
    M, N = shape
    run_test_fused_mul_add(M, N, 128, 256, target)


def vec_mul_add_dst(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
            D: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
            T.copy(C[bx * block_M + vid * block_M // VEC_NUM, by * block_N], c_ub)

            # a_ub = b_ub * c_ub + a_ub
            T.tile.mul_add_dst(a_ub, b_ub, c_ub)

            T.copy(a_ub, D[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_mul_add_dst(M, N, block_M, block_N, target):
    func = vec_mul_add_dst(M, N, block_M, block_N)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()
    c = torch.randn(M, N).npu()

    torch.npu.synchronize()

    d = func(a, b, c)

    ref_d = b * c + a
    torch.testing.assert_close(d, ref_d, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_vec_mul_add_dst(target, shape):
    M, N = shape
    run_test_mul_add_dst(M, N, 128, 256, target)
