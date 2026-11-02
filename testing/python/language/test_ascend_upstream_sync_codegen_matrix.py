# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Real Bisheng ABI checks; these tests never invoke the compiled kernels.

Compilation on the available A3 host does not certify execution on A2/A5.
The programs exercise a V-to-V handoff and a physical software-pipeline ring.
"""

import shutil

import pytest

import tilelang
from tilelang import language as T
from tilelang import tvm

from test_ascend_pipeline_ring_provenance import _pipelined_abs
from test_ascend_upstream_sync_device import _PASS_CONFIGS


@pytest.fixture(scope="module", autouse=True)
def _requires_native_compiler_and_proof_service():
    if shutil.which("bisheng") is None:
        pytest.skip("requires the CANN Bisheng compiler")
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


def _same_pipe_abs():
    # The verified Persistent and pipeline fixtures use the same UB/Abs API.
    # Two in-place operations additionally require a real same-V handoff.
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), O: T.Tensor((2, 64), "float32")):
        with T.Kernel(1, is_npu=True) as (_cid, vid):
            local = T.alloc_ub((64,), "float32")
            with T.Scope("V"):
                T.copy(A, local)
                T.tile.abs(local, local)
                T.tile.abs(local, local)
                T.copy(local, O[vid, :])

    return main


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("platform", ["A2", "A3", "A5"])
@pytest.mark.parametrize("program_kind", ["same_pipe", "physical_ring"])
def test_bisheng_compiles_upstream_sync_abi(target, platform, program_kind):
    program = _same_pipe_abs() if program_kind == "same_pipe" else _pipelined_abs(5, 2)
    tilelang.disable_cache()
    try:
        kernel = tilelang.compile(
            program.with_attr("tir.noalias", True),
            out_idx=[],
            target=target,
            platform=platform,
            pass_configs=_PASS_CONFIGS,
            compile_flags=["--cce-auto-sync=off"],
        )
    except tvm.error.TVMError as error:
        # Verified with the legacy backend too. Only this exact pre-existing
        # codegen limitation is expected; synchronization failures still fail.
        if target == "ascendc" and platform == "A5" and "Unresolved call Op(tl.ascend_abs)" in str(error):
            pytest.xfail("Existing A5/AscendC codegen has no tl.ascend_abs lowering")
        raise
    assert kernel.get_kernel_source()
    if program_kind != "same_pipe":
        return
    calls = []
    for _symbol, function in kernel.artifact.device_mod.functions_items():
        if isinstance(function, tvm.tir.PrimFunc):
            tvm.tir.stmt_functor.post_order_visit(
                function.body,
                lambda node: calls.append(node) if isinstance(node, tvm.tir.Call) else None,
            )
    if platform == "A5":
        assert any(getattr(call.op, "name", "") == "tl.ascend_auto_set_flag" and call.args[0].value == "V_V" for call in calls)
    else:
        assert any(getattr(call.op, "name", "") == "tl.ascend_auto_barrier" and call.args[0].value == "PIPE_V" for call in calls)
