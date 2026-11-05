"""Regression tests for rank-changing T.Parallel buffer projections.

The vector-copy helper assumes matching source and destination ranks. A 2D UB
row projected into a 1D UB buffer must use the scalar serial fallback rather
than indexing the 1D destination shape as if it were 2D.
"""

import pytest
import tilelang
import tilelang.language as T
import torch


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


@pytest.fixture
def setup_random_seed():
    torch.manual_seed(0)
    yield


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def row_projection_kernel(rows, lanes, dtype="float32"):
    @T.prim_func
    def main(
        source: T.Tensor((rows, lanes), dtype),
        destination: T.Tensor((lanes,), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            source_ub = T.alloc_ub((rows, lanes), dtype)
            destination_ub = T.alloc_ub((lanes,), dtype)
            with T.Scope("V"):
                T.copy(source, source_ub)
                for lane in T.Parallel(lanes):
                    destination_ub[lane] = source_ub[rows - 1, lane]
                T.copy(destination_ub, destination)

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def rank_matched_row_kernel(rows, lanes, dtype="float32"):
    @T.prim_func
    def main(
        source: T.Tensor((rows, lanes), dtype),
        destination: T.Tensor((1, lanes), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            source_ub = T.alloc_ub((rows, lanes), dtype)
            destination_ub = T.alloc_ub((1, lanes), dtype)
            with T.Scope("V"):
                T.copy(source, source_ub)
                for lane in T.Parallel(lanes):
                    destination_ub[0, lane] = source_ub[rows - 1, lane]
                T.copy(destination_ub, destination)

    return main


def test_parallel_2d_to_1d_projection_uses_serial_fallback(setup_random_seed):
    rows, lanes = 4, 16
    kernel = row_projection_kernel(rows, lanes)
    source = torch.randn((rows, lanes), dtype=torch.float32, device="npu")
    out = kernel(source)

    torch.testing.assert_close(out, source[-1], rtol=0, atol=0)

    kernel_source = kernel.get_kernel_source()
    assert "GetValue" in kernel_source
    assert "SetValue" in kernel_source


def test_parallel_rank_matched_projection_remains_valid(setup_random_seed):
    rows, lanes = 4, 16
    kernel = rank_matched_row_kernel(rows, lanes)
    source = torch.randn((rows, lanes), dtype=torch.float32, device="npu")
    out = kernel(source)

    torch.testing.assert_close(out[0], source[-1], rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "0"])
