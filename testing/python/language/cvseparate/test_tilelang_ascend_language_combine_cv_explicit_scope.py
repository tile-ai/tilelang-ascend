"""Regression tests for CombineCV with explicit C/V scopes.

Explicit ``T.Scope`` regions are authoritative ownership boundaries. CombineCV
must not reclassify manual cross-core flags from neighboring copy operations,
or it can place a consumer wait under an impossible nested AIV/AIC predicate.
"""

import pytest
import tilelang
import tilelang.language as T
import torch


M = 64
N = 64
K = 64

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


@pytest.fixture
def setup_random_seed():
    torch.manual_seed(0)
    yield


def explicit_v_to_c_kernel():
    @T.prim_func
    def main(
        a: T.Tensor((M, K), "float16"),
        b: T.Tensor((K, N), "float16"),
        out: T.Tensor((M, N), "float32"),
        a_workspace: T.Tensor((M, K), "float16"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as _cid:
            a_ub = T.alloc_ub((M, K), "float16")
            a_l1 = T.alloc_L1((M, K), "float16")
            b_l1 = T.alloc_L1((K, N), "float16")
            c_l0 = T.alloc_L0C((M, N), "float32")

            with T.Scope("V"):
                T.copy(a, a_ub)
                for i, j in T.Parallel(M, K):
                    a_ub[i, j] = a_ub[i, j] * 2.0
                T.copy(a_ub, a_workspace)
                T.set_cross_flag("MTE3", 0)

            with T.Scope("C"):
                T.wait_cross_flag(0)
                T.copy(a_workspace, a_l1)
                T.copy(b, b_l1)
                T.barrier_all()
                T.gemm_v0(a_l1, b_l1, c_l0, init=True)
                T.barrier_all()
                T.copy(c_l0, out)

    return main


def explicit_c_to_v_kernel():
    @T.prim_func
    def main(
        a: T.Tensor((M, K), "float16"),
        b: T.Tensor((K, N), "float16"),
        out: T.Tensor((M, N), "float32"),
        c_workspace: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as _cid:
            a_l1 = T.alloc_L1((M, K), "float16")
            b_l1 = T.alloc_L1((K, N), "float16")
            c_l0 = T.alloc_L0C((M, N), "float32")
            c_ub = T.alloc_ub((M, N), "float32")

            with T.Scope("C"):
                T.copy(a, a_l1)
                T.copy(b, b_l1)
                T.barrier_all()
                T.gemm_v0(a_l1, b_l1, c_l0, init=True)
                T.barrier_all()
                T.copy(c_l0, c_workspace)
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                T.wait_cross_flag(0)
                T.copy(c_workspace, c_ub)
                for i, j in T.Parallel(M, N):
                    c_ub[i, j] = c_ub[i, j] + 1.0
                T.copy(c_ub, out)

    return main


def _resource_scopes_at(source, needle):
    """Return active ASCEND_IS_AIC/AIV predicates at the line containing needle."""

    depth = 0
    active_scopes = []
    for line in source.splitlines():
        stripped = line.strip()
        if needle in stripped:
            return [scope for _, scope in active_scopes]

        if "if ASCEND_IS_AIC" in stripped and "{" in stripped:
            active_scopes.append((depth, "AIC"))
        elif "if ASCEND_IS_AIV" in stripped and "{" in stripped:
            active_scopes.append((depth, "AIV"))

        depth += line.count("{") - line.count("}")
        active_scopes = [(open_depth, scope) for open_depth, scope in active_scopes if depth > open_depth]

    raise AssertionError(f"{needle!r} not found in generated source")


@pytest.mark.parametrize(
    "program,expected_wait_scope,expected_set_scope",
    [
        (explicit_v_to_c_kernel, "AIC", "AIV"),
        (explicit_c_to_v_kernel, "AIV", "AIC"),
    ],
)
def test_explicit_scope_keeps_manual_cross_core_flags(
    program,
    expected_wait_scope,
    expected_set_scope,
):
    kernel = tilelang.compile(
        program(),
        out_idx=[2],
        workspace_idx=[3],
        pass_configs=pass_configs,
        target="ascendc",
    )
    source = kernel.get_kernel_source()

    assert source.count("CrossCoreWaitFlag") == 1
    assert source.count("CrossCoreSetFlag") == 1
    assert _resource_scopes_at(source, "CrossCoreWaitFlag") == [expected_wait_scope]
    assert _resource_scopes_at(source, "CrossCoreSetFlag") == [expected_set_scope]


def test_explicit_v_to_c_is_correct(setup_random_seed):
    kernel = tilelang.compile(
        explicit_v_to_c_kernel(),
        out_idx=[2],
        workspace_idx=[3],
        pass_configs=pass_configs,
        target="ascendc",
    )
    a = torch.randn((M, K), dtype=torch.float16, device="npu")
    b = torch.randn((K, N), dtype=torch.float16, device="npu")
    out = kernel(a, b)
    reference = (a * 2.0).float() @ b.float()
    torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-4)


def test_explicit_c_to_v_is_correct(setup_random_seed):
    kernel = tilelang.compile(
        explicit_c_to_v_kernel(),
        out_idx=[2],
        workspace_idx=[3],
        pass_configs=pass_configs,
        target="ascendc",
    )
    a = torch.randn((M, K), dtype=torch.float16, device="npu")
    b = torch.randn((K, N), dtype=torch.float16, device="npu")
    out = kernel(a, b)
    reference = a.float() @ b.float() + 1.0
    torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "0"])
