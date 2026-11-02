# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""NPU acceptance for a real Developer GEMM->exp C/V publication.

No kernel-level flags, barriers, or resource scopes are written manually.
The existing automatic C/V split and workspace publication remain enabled;
the migrated engine supplies local completion synchronization.
"""

import contextlib
import re
from typing import Any

import pytest
import torch

import tilelang
from tilelang import language as T
from tilelang import tvm


pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU execution",
)

_M, _N, _K = 32, 16, 64
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_SYNC_BACKEND: "upstream",
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


def _flat_developer_gemm_exp(*, explicit_noalias: bool = True) -> tvm.tir.PrimFunc:
    # Based on test_ascend_sync_insert_vs::test_resource_scope_no_crash.
    # Developer threads=2 also enables compiler-managed UB row partitioning,
    # so the two Vector cores process disjoint halves of the GEMM result.
    @T.prim_func
    def main(
        A: T.Tensor((_M, _K), "float16"),
        B: T.Tensor((_K, _N), "float16"),
        C: T.Tensor((_M, _N), "float16"),
    ):
        with T.Kernel(1, threads=2, is_npu=True):
            a_l1 = T.alloc_shared((_M, _K), "float16")
            b_l1 = T.alloc_shared((_K, _N), "float16")
            accumulator = T.alloc_fragment((_M, _N), "float32")
            result_ub = T.alloc_shared((_M, _N), "float16")
            T.copy(A[:, :], a_l1)
            T.copy(B[:, :], b_l1)
            T.gemm_v0(a_l1, b_l1, accumulator, init=True)
            T.copy(accumulator, result_ub)
            T.tile.exp(result_ub, result_ub)
            T.copy(result_ub, C[:, :])

    # The test allocates A, B, C separately; automatic GM workspaces are also
    # separate allocations. This states the actual alias contract explicitly.
    if explicit_noalias:
        return main.with_attr("tir.noalias", True)
    return main.without_attr("tir.noalias")


def _capture_pre_sync(program: tvm.tir.PrimFunc) -> tvm.tir.PrimFunc:
    """Run host lowering only, stopping before the new synchronization pass."""

    class Captured(Exception):
        pass

    @tvm.instrument.pass_instrument
    class Capture:
        def __init__(self) -> None:
            self.function = None

        def run_before_pass(self, module: tvm.IRModule, info: Any) -> None:
            if info.name == "tl.AscendInsertSync":
                self.function = module["main"]
                raise Captured

    capture = Capture()
    with (
        tvm.transform.PassContext(config=_PASS_CONFIGS, instruments=[capture]),
        contextlib.suppress(Captured),
    ):
        tilelang.lower(program, target="ascendc", platform="A3")
    assert capture.function is not None, "did not reach the upstream sync boundary"
    return capture.function


def _assert_pre_sync_contract(program: tvm.tir.PrimFunc) -> None:
    ratio = program.attrs["npu_cv_ratio"]
    assert (ratio.value if isinstance(ratio, tvm.tir.StringImm) else str(ratio)) == "cv_1_2"
    calls = []
    subgroup = []

    def visit(statement: tvm.tir.Stmt, owner: int | None = None) -> None:
        if isinstance(statement, tvm.tir.SeqStmt):
            for child in statement.seq:
                visit(child, owner)
        elif isinstance(statement, tvm.tir.AttrStmt):
            if statement.attr_key == "resource_scope":
                owner = int(statement.value)
            elif statement.attr_key == "thread_extent" and statement.node.thread_tag == "threadIdx.x":
                assert int(statement.value) == 2
                subgroup.append(statement.node.var)
            visit(statement.body, owner)
        elif isinstance(statement, (tvm.tir.Allocate, tvm.tir.DeclBuffer, tvm.tir.LetStmt)):
            visit(statement.body, owner)
        elif isinstance(statement, tvm.tir.Evaluate):
            if isinstance(statement.value, tvm.tir.Call):
                calls.append((owner, statement.value))
        else:
            raise AssertionError(f"expected a flat pre-sync program, got {type(statement)}")

    visit(program.body)

    def named(name: str) -> list[tuple[int, tvm.tir.Call]]:
        return [(owner, call) for owner, call in calls if call.op.name == name]

    sets = named("tl.ascend_auto_set_cross_flag")
    waits = named("tl.ascend_auto_wait_cross_flag")
    assert len(sets) == len(waits) == len(subgroup) == 1
    assert sets[0][0] == 0 and waits[0][0] == 1
    release, acquire = sets[0][1], waits[0][1]
    assert int(release.args[0]) == 2 and release.args[1].value == "FIX"
    assert int(release.args[2]) == int(acquire.args[0])
    assert acquire.args[1].value == "MTE2"
    assert len(named("tl.ascend_gemm_v0")) == 1
    assert not named("tl.ascend_set_flag") and not named("tl.ascend_wait_flag")
    assert not named("tl.ascend_pipe_barrier")

    external = [(owner, call) for owner, call in calls if call.op.name == "tir.call_extern"]
    stores = [(owner, call) for owner, call in external if "copy_l0c_to_gm" in call.args[0].value]
    loads = [(owner, call) for owner, call in external if "copy_gm_to_ub" in call.args[0].value]
    assert len(stores) == len(loads) == 1
    assert stores[0][0] == 0 and loads[0][0] == 1
    load = loads[0][1]
    shape = re.fullmatch(r"tl::ascend::copy_gm_to_ub<half,\s*(\d+),\s*(\d+)>", load.args[0].value)
    assert shape is not None
    columns, rows = map(int, shape.groups())
    assert (rows, columns) == (_M // 2, _N)
    assert int(load.args[4]) == rows and int(load.args[5]) == columns
    # Compare real template/mask dimensions, not the conservative source
    # access_ptr extent, which may still describe the full GM workspace.
    assert stores[0][1].args[2].args[1].same_as(load.args[1].args[1])
    exponentials = named("tl.ascend_exp")
    assert len(exponentials) == 1 and exponentials[0][0] == 1
    assert int(exponentials[0][1].args[2]) == _M // 2 * _N
    output_stores = [call for _, call in external if "copy_ub_to_gm" in call.args[0].value]
    assert len(output_stores) == 1
    output_offset = output_stores[0].args[2].args[2]
    assert tvm.arith.Analyzer().can_prove(output_offset == subgroup[0] * (_M // 2 * _N))


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


def test_flat_developer_gemm_exp_publication_and_repeated_invocation() -> None:
    program = _flat_developer_gemm_exp()
    _assert_pre_sync_contract(_capture_pre_sync(program))
    tilelang.disable_cache()
    kernel = tilelang.compile(
        program,
        out_idx=[2],
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    source = kernel.get_kernel_source()
    assert "CrossCoreSetFlag<2, PIPE_FIX>" in source
    assert "CrossCoreWaitFlag(" in source
    assert "HardEvent::S_MTE2" in source
    generator = torch.Generator().manual_seed(20261030)
    for _invocation in range(3):
        first = (torch.randn((_M, _K), generator=generator) * 0.125).half()
        second = (torch.randn((_K, _N), generator=generator) * 0.125).half()
        expected = (first.float() @ second.float()).half().exp().half()
        actual = kernel(first.npu(), second.npu())
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("output_alias", [None, "A", "B"])
def test_cython_fresh_workspace_orders_possible_input_output_alias(
    output_alias: str | None,
) -> None:
    from tilelang.jit.ascend_runtime_contract import (
        AUTO_GM_FRESH_CONTRACT_ATTR,
        AUTO_GM_FRESH_CONTRACT_VERSION,
        prepare_auto_gm_contract,
    )

    program = _flat_developer_gemm_exp(explicit_noalias=False)
    assert "tir.noalias" not in program.attrs
    pre_sync = _capture_pre_sync(prepare_auto_gm_contract(program, "cython", "ascendc"))
    _assert_pre_sync_contract(pre_sync)
    assert pre_sync.attrs[AUTO_GM_FRESH_CONTRACT_ATTR].value == AUTO_GM_FRESH_CONTRACT_VERSION
    assert [int(index) for index in pre_sync.attrs["auto_gm_indices"]] == [3]
    assert "tir.noalias" not in pre_sync.attrs
    tilelang.disable_cache()
    kernel = tilelang.compile(
        program,
        out_idx=[],
        execution_backend="cython",
        target="ascendc",
        platform="A3",
        pass_configs=_PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
    )
    source = kernel.get_kernel_source()
    assert "CrossCoreSetFlag<2, PIPE_FIX>" in source
    assert "CrossCoreWaitFlag(" in source
    assert "HardEvent::S_MTE2" in source
    generator = torch.Generator().manual_seed(20261031)
    for _invocation in range(3):
        first = (torch.randn((_M, _K), generator=generator) * 0.125).half()
        second = (torch.randn((_K, _N), generator=generator) * 0.125).half()
        expected = (first.float() @ second.float()).half().exp().half()
        first_device, second_device = first.npu(), second.npu()
        if output_alias is None:
            output = torch.empty((_M, _N), dtype=torch.float16, device=first_device.device)
        else:
            owner = first_device if output_alias == "A" else second_device
            output = owner.view(-1)[: _M * _N].view(_M, _N)
            assert output.data_ptr() == owner.data_ptr()
        kernel(first_device, second_device, output)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-2, atol=1e-2)
