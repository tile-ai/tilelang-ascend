"""Copy a view of a nonzero-offset UB alias; check exact bytes."""

import numpy as np
import pytest
import tilelang
import tilelang.language as T

pytestmark = [pytest.mark.op("view"), pytest.mark.mode("Expert")]


@T.macro
def wait_for(src, dst):
    with T.rs(src):
        T.set_flag(dst, 0)
    with T.rs(dst):
        T.wait_flag(src, 0)


def offset_kernel(case):
    src_dtype = "float16" if case == "wide" else "float32"
    dst_dtype = "float16" if case == "narrow" else "float32"
    shape = (8, 32) if case == "narrow" else ((4, 16) if case == "wide" else (8, 16))

    @T.prim_func
    def main(A: T.Tensor((16, 32), src_dtype), B: T.Tensor(shape, dst_dtype)):
        with T.Kernel(1, is_npu=True):
            a = T.alloc_ub((16, 32), src_dtype)
            source = T.Buffer(
                (4, 32), src_dtype, data=a.data, elem_offset=128, scope="shared"
            )
            viewed = T.view(source, shape, dtype=dst_dtype)
            T.copy(A, a)
            wait_for("PIPE_MTE2", "PIPE_MTE3")
            T.copy(viewed, B)
            wait_for("PIPE_MTE3", "PIPE_MTE2")

    return main


@pytest.mark.parametrize("case", ["same", "narrow", "wide"])
def test_view_offset_device(case):
    import torch
    import torch_npu  # noqa: F401

    flags = {
        "npuir.enable_auto_multi_buffer": False,
        "npuir.disable_hivm_auto_inject_sync": True,
        "tl.enable_plan_and_update_buffer_allocation": False,
    }
    fn = tilelang.compile(offset_kernel(case), target="npuir", pass_configs=flags)
    src_dtype = "float16" if case == "wide" else "float32"
    dst_dtype = "float16" if case == "narrow" else "float32"
    shape = (8, 32) if case == "narrow" else ((4, 16) if case == "wide" else (8, 16))
    a = np.arange(512, dtype=src_dtype).reshape(16, 32)
    expected = a.ravel()[128:256].view(dst_dtype).reshape(shape).copy()
    da = torch.from_numpy(a).npu()
    db = torch.empty(shape, dtype=getattr(torch, dst_dtype), device="npu")
    fn(da, db)
    torch.npu.synchronize()
    assert db.cpu().numpy().tobytes() == expected.tobytes()
