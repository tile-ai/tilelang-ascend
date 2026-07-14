import pytest
import torch

import tilelang
import tilelang.language as T

NUMEL = 32

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def if_then_else_buffer_store(dtype):

    @T.prim_func
    def main(
        A: T.Tensor((NUMEL,), dtype),  # type: ignore
        B: T.Tensor((NUMEL,), dtype),  # type: ignore
        Selector: T.Tensor((NUMEL,), "int32"),  # type: ignore
        C: T.Tensor((NUMEL,), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((NUMEL,), dtype)
            b_ub = T.alloc_ub((NUMEL,), dtype)
            selector_ub = T.alloc_ub((NUMEL,), "int32")
            c_ub = T.alloc_ub((NUMEL,), dtype)

            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.copy(Selector, selector_ub)

            for i in T.serial(NUMEL):
                c_ub[i] = T.if_then_else(
                    selector_ub[i] > 0,
                    T.if_then_else(selector_ub[i] > 1, a_ub[i], b_ub[i]),
                    b_ub[i],
                )

            c_ub[T.if_then_else(selector_ub[0] > 0, 0, 1)] = T.if_then_else(selector_ub[1] > 1, a_ub[1], b_ub[1])
            T.copy(c_ub, C)

    return main


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_if_then_else_buffer_store_codegen(dtype, target):
    with tilelang.tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        artifact = tilelang.lower(if_then_else_buffer_store(dtype), target=target)

    source = artifact.kernel_source
    first_binding = source.find("condval")
    store_lines = [line for line in source.splitlines() if ".SetValue(" in line]
    assert first_binding >= 0
    assert len(store_lines) == 2
    assert all("condval" in line and line.rstrip().endswith(");") for line in store_lines)
    assert all(first_binding < source.find(line) for line in store_lines)


@pytest.mark.parametrize(
    "dtype,torch_dtype",
    [("float", torch.float32), ("float16", torch.float16)],
)
def test_if_then_else_buffer_store_npu(dtype, torch_dtype):
    func = tilelang.compile(
        if_then_else_buffer_store(dtype),
        out_idx=[-1],
        pass_configs=PASS_CONFIGS,
        target="ascendc",
    )

    a = torch.tensor([3.0, 1.0] * (NUMEL // 2), dtype=torch_dtype).npu()
    b = torch.tensor([1.0, 4.0] * (NUMEL // 2), dtype=torch_dtype).npu()
    selector = torch.tensor([2, 1, 0, -1] * (NUMEL // 4), dtype=torch.int32).npu()
    torch.npu.synchronize()

    actual = func(a, b, selector)
    torch.npu.synchronize()

    expected = torch.where(selector > 0, torch.where(selector > 1, a, b), b)
    expected[0] = torch.where(selector[1] > 1, a[1], b[1])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
