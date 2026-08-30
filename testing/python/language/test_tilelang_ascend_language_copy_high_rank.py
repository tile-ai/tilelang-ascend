"""Ascend GM/UB copy regressions for contiguous higher-rank regions."""

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


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def copy_3d_identity_kernel(dtype="float16"):

    @T.prim_func
    def main(
        A: T.Tensor((2, 2, 16), dtype),
        C: T.Tensor((2, 2, 16), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((2, 2, 16), dtype)
            with T.Scope("V"):
                T.copy(A, a_ub)
                T.copy(a_ub, C)

    return main


def test_contiguous_3d_gm_ub_copy():
    kernel = copy_3d_identity_kernel()
    source = kernel.get_kernel_source()
    assert "copy_gm_to_ub<half, 16, 4>" in source
    assert "copy_ub_to_gm<half, 16, 4>" in source

    a = torch.arange(64, dtype=torch.float16).reshape(2, 2, 16).npu()
    torch.npu.synchronize()
    out = kernel(a)
    torch.testing.assert_close(out, a, rtol=0, atol=0)


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def copy_trailing_singleton_kernel(dtype="float16"):

    @T.prim_func
    def main(
        A: T.Tensor((2, 4, 5, 16), dtype),
        C: T.Tensor((8, 16), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((8, 16), dtype)
            with T.Scope("V"):
                T.copy(A[:, :, 2:3, :], a_ub)
                T.copy(a_ub, C)

    return main


def test_trailing_singleton_gm_dimension_uses_constant_stride():
    kernel = copy_trailing_singleton_kernel()
    a = torch.arange(640, dtype=torch.float16).reshape(2, 4, 5, 16).npu()
    expected = a[:, :, 2, :].reshape(8, 16)
    torch.npu.synchronize()
    out = kernel(a)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@T.prim_func
def noncontiguous_high_rank_copy(
    A: T.Tensor((2, 2, 2, 16), "float16"),
    C: T.Tensor((4, 16), "float16"),
):
    with T.Kernel(1, is_npu=True) as (cid, vid):
        a_ub = T.alloc_ub((4, 16), "float16")
        with T.Scope("V"):
            T.copy(A[:, 0:1, :, :], a_ub)
            T.copy(a_ub, C)


def test_noncontiguous_high_rank_copy_fails_loudly():
    with pytest.raises(
        tilelang.tvm.error.InternalError,
        match="Cannot flatten non-contiguous high-rank Ascend GM<->UB copy",
    ):
        tilelang.lower(noncontiguous_high_rank_copy, target="ascendc")


@T.prim_func
def nonzero_min_full_extent_copy(
    A: T.Tensor((2, 2, 16), "float16"),
    C: T.Tensor((4, 16), "float16"),
):
    with T.Kernel(1, is_npu=True) as (cid, vid):
        a_ub = T.alloc_ub((4, 16), "float16")
        with T.Scope("V"):
            T.copy(A[:, 1:3, :], a_ub)
            T.copy(a_ub, C)


def test_nonzero_min_full_extent_copy_fails_loudly():
    with pytest.raises(
        tilelang.tvm.error.InternalError,
        match="Cannot flatten non-contiguous high-rank Ascend GM<->UB copy",
    ):
        tilelang.lower(nonzero_min_full_extent_copy, target="ascendc")


@T.prim_func
def unaligned_high_rank_copy(
    A: T.Tensor((2, 2, 9), "float16"),
    C: T.Tensor((2, 2, 9), "float16"),
):
    with T.Kernel(1, is_npu=True) as (cid, vid):
        a_ub = T.alloc_ub((2, 2, 9), "float16")
        with T.Scope("V"):
            T.copy(A, a_ub)
            T.copy(a_ub, C)


def test_unaligned_high_rank_copy_supported():
    kernel = tilelang.compile(
        unaligned_high_rank_copy,
        out_idx=[-1],
        pass_configs=pass_configs,
        target="ascendc",
    )
    a = torch.arange(36, dtype=torch.float16).reshape(2, 2, 9).npu()
    torch.npu.synchronize()
    out = kernel(a)
    torch.testing.assert_close(out, a, rtol=0, atol=0)
