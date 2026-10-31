"""Device regression for an outstanding alias read across an optional loop."""

import pytest
import torch

import tilelang
import tilelang.ascend.a2a3.language as T
from tilelang import tvm
from tilelang.env import CacheState


@pytest.fixture
def optional_loop_kernel(monkeypatch):
    monkeypatch.setattr(CacheState, "_enabled", False)
    n = T.symbolic("n")

    @T.prim_func
    def main(
        A: T.Tensor((64,), "float32"),
        O: T.Tensor((2, 2, 64), "float32"),
        Count: T.Tensor((n,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a = T.alloc_ub((64,), "float32")
            b = T.alloc_ub((64,), "float32")
            T.annotate_address({a: 0, b: 0})
            with T.Scope("V"):
                T.tile.fill(b, 1)
                T.barrier_all()
                T.copy(a, O[vid, 0, :])
                for _i in T.serial(n):
                    T.tile.fill(b, 2)
                    T.copy(A, b)
                T.tile.fill(b, 3)
                T.copy(b, O[vid, 1, :])

    return tilelang.compile(
        main,
        out_idx=[],
        target=tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]}),
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
        },
        # Exercise the framework's handoffs without BiSheng repairing them.
        compile_flags=["--cce-auto-sync=off"],
    )


@pytest.mark.a2a3_device
@pytest.mark.parametrize("trip_count", [0, 1, 3])
def test_optional_loop_preserves_the_previous_alias_value(optional_loop_kernel, trip_count):
    source = torch.arange(64, dtype=torch.float32, device="npu")
    output = torch.full((2, 2, 64), -99.0, dtype=torch.float32, device="npu")
    count = torch.empty((trip_count,), dtype=torch.int32, device="npu")
    expected = torch.full((2, 2, 64), 3.0, dtype=torch.float32)
    expected[:, 0, :] = 1.0
    optional_loop_kernel(source, output, count)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
