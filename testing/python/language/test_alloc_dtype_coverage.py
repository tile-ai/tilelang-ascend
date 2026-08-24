import pytest

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


# ---------------------------------------------------------------------------
# alloc_shared dtype coverage
# ---------------------------------------------------------------------------


def _make_alloc_shared_kernel(dtype, shape=(128,)):
    @T.prim_func
    def main(A: T.Tensor(shape, dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            buf = T.alloc_shared(shape, dtype)
            T.tile.fill(buf, 1)
            T.copy(buf, A[0])

    return main


@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
        pytest.param("float16", marks=pytest.mark.low_priority),
        pytest.param("bfloat16", marks=pytest.mark.low_priority),
        pytest.param("int16", marks=pytest.mark.low_priority),
        pytest.param("int32", marks=pytest.mark.low_priority),
    ],
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_alloc_shared_dtype(dtype, target):
    shape = (128,)
    kernel = _make_alloc_shared_kernel(dtype, shape)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)
    torch.npu.synchronize()
    result = func()
    torch.npu.synchronize()

    tdtype = TORCH_DTYPE_MAP[dtype]
    ref = torch.ones(shape, dtype=tdtype)
    torch.testing.assert_close(result.cpu(), ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# alloc_fragment dtype coverage (GEMM accumulator)
# ---------------------------------------------------------------------------


def _make_alloc_fragment_gemm(dtype):
    M, N, K = 128, 128, 128
    if dtype in ("float16", "bfloat16"):
        accum_dtype = "float32"
    elif dtype == "int8":
        accum_dtype = "int32"
    else:
        accum_dtype = "float32"

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((M, K), dtype)
            B_L1 = T.alloc_L1((K, N), dtype)
            C_L0 = T.alloc_fragment((M, N), accum_dtype)
            T.copy(A, A_L1)
            T.copy(B, B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0, init=True)
            T.copy(C_L0, C)

    return main, accum_dtype


@pytest.mark.parametrize(
    "dtype",
    [
        "float16",
        pytest.param("bfloat16", marks=pytest.mark.low_priority),
        pytest.param("int8", marks=pytest.mark.low_priority),
    ],
)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_alloc_fragment_gemm(dtype, target):
    kernel, accum_dtype = _make_alloc_fragment_gemm(dtype)
    func = tilelang.compile(kernel, out_idx=[2], pass_configs=PASS_CONFIGS, target=target)
    torch.npu.synchronize()

    M, N, K = 128, 128, 128
    tdtype = TORCH_DTYPE_MAP[dtype]
    a = torch.ones(M, K, dtype=tdtype).npu()
    b = torch.ones(K, N, dtype=tdtype).npu()
    torch.npu.synchronize()
    result = func(a, b)
    torch.npu.synchronize()

    ref = (a.float() @ b.float()).to(TORCH_DTYPE_MAP[accum_dtype])
    torch.testing.assert_close(result.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# alloc_var dtype coverage (ascendc only — pto has init bug)
# ---------------------------------------------------------------------------


def _make_alloc_var_kernel(dtype):
    @T.prim_func
    def main(A: T.Tensor((16,), "int32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            buf = T.alloc_shared((16,), "int32")
            v = T.alloc_var(dtype, init=1)
            T.tile.fill(buf, 0)
            buf[0] = v
            T.copy(buf, A[0])

    return main


@pytest.mark.parametrize(
    "dtype",
    [
        "int32",
        pytest.param("float32", marks=pytest.mark.low_priority),
        pytest.param("bool", marks=pytest.mark.low_priority),
    ],
)
def test_alloc_var_dtype_ascendc(dtype):
    kernel = _make_alloc_var_kernel(dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")
    torch.npu.synchronize()
    result = func()
    torch.npu.synchronize()

    ref = torch.zeros(16, dtype=torch.int32)
    ref[0] = 1
    torch.testing.assert_close(result.cpu(), ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Exception boundary: single-backend dtype compile failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,target",
    [
        ("int8", "ascendc"),
        ("uint8", "ascendc"),
    ],
)
def test_alloc_shared_unsupported_dtype_raises(dtype, target):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            buf = T.alloc_shared((128,), dtype)
            T.tile.fill(buf, 1)
            T.copy(buf, A[0])

    with pytest.raises(Exception, match="Compilation"):  # noqa: B017
        tilelang.compile(main, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
