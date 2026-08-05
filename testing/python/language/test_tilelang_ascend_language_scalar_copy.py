import pytest
import torch

import tilelang
import tilelang.language as T
import tvm


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


def scalar_copy(dtype):
    @T.prim_func
    def main(
        A: T.Tensor((), dtype),
        B: T.Tensor((1, 32), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            scalar_ub = T.alloc_ub((1,), dtype)
            output_ub = T.alloc_ub((1, 32), dtype)

            T.copy(A, scalar_ub)
            T.tile.fill(output_ub, 0)
            output_ub[0, 0] = scalar_ub[0]
            T.copy(output_ub, B)

    return main


def mismatched_scalar_copy():
    @T.prim_func
    def main(A: T.Tensor((2,), "float32")):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            scalar_ub = T.alloc_ub((1,), "float32")
            T.copy(A, scalar_ub)

    return main


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_scalar_copy(dtype):
    func = tilelang.compile(
        scalar_copy(dtype),
        out_idx=[],
        pass_configs=PASS_CONFIGS,
        target="ascendc",
    )

    a = torch.tensor(2, dtype=getattr(torch, dtype), device="npu")
    b = torch.empty((1, 32), dtype=getattr(torch, dtype), device="npu")
    func(a, b)
    torch.npu.synchronize()

    expected = torch.zeros((1, 32), dtype=getattr(torch, dtype), device="npu")
    expected[0, 0] = a
    torch.testing.assert_close(b, expected, rtol=0, atol=0)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_scalar_copy_codegen(dtype, target):
    with tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        artifact = tilelang.lower(scalar_copy(dtype), target=target)

    assert "scalar_ub.SetValue(0" in artifact.kernel_source
    assert "copy_gm_to_ub" not in artifact.kernel_source


def test_scalar_copy_rejects_mismatched_shape():
    with pytest.raises(tvm.error.DiagnosticError):
        mismatched_scalar_copy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
