# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Conditional GEMM calls retain lexical proofs and asynchronous lifetimes."""

import pytest

from tilelang import language as T
from tilelang import tvm

from test_ascend_pipeline_ring_provenance import _nodes, _pre_sync
from test_ascend_upstream_sync_composite import _events, _run
from test_ascend_upstream_sync_composite_invocation import (
    _assert_calculation_preserved,
    _independent_gemms,
)
from test_ascend_upstream_sync_composite_loop import _execute_k_protocol
from test_ascend_upstream_sync_pipeline import _assert_balanced_events

tir = tvm.tir


def _conditional_gemms(with_else=False):
    # Extend _independent_gemms: disjoint user allocations still share the
    # helper's hidden L0/event pools across an optional call and loop backedge.
    @T.prim_func
    def main(
        A: T.Tensor((16, 128), "float16"),
        B: T.Tensor((128, 16), "float16"),
        D: T.Tensor((16, 64), "float16"),
        E: T.Tensor((64, 16), "float16"),
        O: T.Tensor((48, 16), "float16"),
        P: T.Tensor((48, 16), "float16"),
        count: T.int32,
        choice: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            a = T.alloc_L1((16, 64), "float16")
            b = T.alloc_L1((64, 16), "float16")
            d = T.alloc_L1((16, 64), "float16")
            e = T.alloc_L1((64, 16), "float16")
            c = T.alloc_L0C((16, 16), "float32")
            f = T.alloc_L0C((16, 16), "float32")
            with T.Scope("C"):
                for batch in T.serial(T.min(T.max(count, 0), 3)):
                    T.copy(D, d)
                    T.copy(E, e)
                    T.gemm_v0(d, e, f, init=True, kL0Size=64)
                    T.copy(f, P[batch * 16, 0])
                    if batch % 2 == choice:
                        T.copy(A[0, 0], a)
                        T.copy(B[0, 0], b)
                        T.gemm_v0(a, b, c, init=True, kL0Size=64)
                        T.copy(c, O[batch * 16, 0])
                    else:
                        if with_else:
                            T.copy(A[0, 64], a)
                            T.copy(B[64, 0], b)
                            T.gemm_v0(a, b, c, init=True, kL0Size=64)
                            T.copy(c, O[batch * 16, 0])

    return main


def _guarded_gemm():
    # The input offset is safe only inside the original nested guards. The
    # audit must use their facts without exporting them to any other leaf.
    @T.prim_func
    def main(
        A: T.Tensor((16, 128), "float16"),
        B: T.Tensor((64, 16), "float16"),
        O: T.Tensor((16, 16), "float16"),
        tile: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            a = T.alloc_L1((16, 64), "float16")
            b = T.alloc_L1((64, 16), "float16")
            c = T.alloc_L0C((16, 16), "float32")
            with T.Scope("C"):
                if tile >= 0:  # noqa: SIM102 - nested proof scopes are the regression input
                    if tile < 2:
                        T.copy(A[0, tile * 64], a)
                        T.copy(B, b)
                        T.gemm_v0(a, b, c, init=True, kL0Size=64)
                        T.copy(c, O)

    return main


def _conditional_accumulation():
    # Extend the ordinary K-loop accumulation test with one optional refresh
    # and clear=False call. FIX remains outside the branch on both paths.
    @T.prim_func
    def main(
        A: T.Tensor((16, 128), "float16"),
        B: T.Tensor((128, 16), "float16"),
        O: T.Tensor((16, 16), "float16"),
        choice: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            a = T.alloc_L1((16, 64), "float16")
            b = T.alloc_L1((64, 16), "float16")
            c = T.alloc_L0C((16, 16), "float32")
            with T.Scope("C"):
                T.copy(A[0, 0], a)
                T.copy(B[0, 0], b)
                T.gemm_v0(a, b, c, init=True, kL0Size=64)
                if choice > 0:
                    T.copy(A[0, 64], a)
                    T.copy(B[64, 0], b)
                    T.gemm_v0(a, b, c, init=False, kL0Size=64)
                T.copy(c, O)

    return main


def _mixed_conditional_gemm(groups=2):
    # Extend the owner-tree GEMM publication fixture: odd tiles reverse the
    # A K blocks, retaining a real conditional refresh and a shared accumulator.
    @T.prim_func
    def main(
        A: T.Tensor((groups * 32, 128), "float16"),
        B: T.Tensor((128, 16), "float16"),
        Workspace: T.Tensor((groups * 32, 16), "float32"),
        Output: T.Tensor((groups * 32, 16), "float32"),
    ):
        with T.Kernel(groups, is_npu=True) as (cid, vid):
            a = T.alloc_L1((16, 64), "float16")
            b = T.alloc_L1((64, 16), "float16")
            c = T.alloc_L0C((16, 16), "float32")
            local = T.alloc_ub((8, 16), "float32")
            with T.Scope("C"):
                for tile in T.serial(2):
                    for k in T.serial(2):
                        if tile % 2 == 0:
                            T.copy(A[cid * 32 + tile * 16, k * 64], a)
                        else:
                            T.copy(A[cid * 32 + tile * 16, (1 - k) * 64], a)
                        T.copy(B[k * 64, 0], b)
                        T.gemm_v0(a, b, c, init=(k == 0), kL0Size=64)
                    T.copy(c, Workspace[cid * 32 + tile * 16, 0])
                T.set_cross_flag("FIX", 3)
            with T.Scope("V"):
                T.wait_cross_flag(3)
                for tile in T.serial(2):
                    T.copy(Workspace[cid * 32 + tile * 16 + vid * 8, 0], local)
                    T.tile.add(local, local, local)
                    T.pipe_barrier("V")
                    T.copy(local, Output[cid * 32 + tile * 16 + vid * 8, 0])

    # The device regression supplies four independent allocations. This is an
    # explicit input contract, not a publication certificate or runtime stamp.
    return main.with_attr("tir.noalias", True)


def _scoped_prefix(function, owner, statement):
    def rewrite(node):
        if node.attr_key == "resource_scope" and int(node.value) == owner:
            return tir.AttrStmt(node.node, node.attr_key, node.value, tir.SeqStmt([statement, node.body]), node.span)
        return None

    return function.with_body(tir.stmt_functor.ir_transform(function.body, rewrite, None, ["tir.AttrStmt"]))


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.fixture(scope="module", params=[False, True], ids=["optional", "if_else"])
def conditional_ir(request):
    before = _pre_sync(_conditional_gemms(request.param))[0]
    assert _nodes(before.body, tir.IfThenElse)
    after = _run(before, target_kind="llvm")
    tvm.ir.assert_structural_equal(after, _run(after, target_kind="llvm"))
    return before, after, request.param


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("choice", [-1, 0, 1])
def test_conditional_calls_balance_every_path_and_hidden_pool_reuse(conditional_ir, count, choice):
    before, after, with_else = conditional_ir
    _assert_calculation_preserved(before, after)
    assert _nodes(after.body, tir.IfThenElse), "conditional execution must remain"
    assert len(_nodes(after.body, tir.For)) == len(_nodes(before.body, tir.For)) == 1
    assert not any(var.name.startswith("__ascend_if_proxy_") for var in _nodes(after.body, tir.Var))
    bindings = {parameter: value for parameter, value in zip(after.params[-2:], (count, choice))}
    events, computations, clears = _execute_k_protocol(after, bindings, hidden_pool=True)
    _assert_balanced_events(events, set())
    selected = sum(batch % 2 == choice for batch in range(count))
    invocations = count + (count if with_else else selected)
    assert computations["tl.ascend_gemm_v0"] == invocations
    assert computations["copy_gm_to_l1"] == invocations * 2
    assert computations["copy_l0c_to_gm"] == invocations
    assert clears == [1] * invocations


def test_conditional_hidden_pool_oracle_detects_removed_completion(conditional_ir):
    _before, after, _with_else = conditional_ir
    assert any(event == "M_MTE1" for _, event, _ in _events(after))

    def remove_completion(statement):
        if isinstance(statement, tir.Evaluate) and isinstance(statement.value, tir.Call):
            call = statement.value
            if getattr(call.op, "name", "") in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"):
                source, destination = call.args[0].value.split("_")
                # Cut every completion feedback route into a later acquire.
                # M->FIX remains, so the first store still has its producer;
                # removing only M->MTE1 can leave a redundant route via FIX.
                if source in ("M", "FIX") and destination in ("MTE1", "MTE2", "M"):
                    return tir.Evaluate(tir.IntImm("int32", 0))
        return None

    corrupted = after.with_body(tir.stmt_functor.ir_transform(after.body, remove_completion, None, ["tir.Evaluate"]))
    bindings = dict(zip(after.params[-2:], (3, 0)))
    with pytest.raises(AssertionError, match="completion edge missing before hidden L0 pool reuse"):
        _execute_k_protocol(corrupted, bindings, hidden_pool=True)


def test_nested_branch_facts_prove_the_original_gm_rectangle():
    before = _pre_sync(_guarded_gemm())[0]
    after = _run(before, target_kind="llvm")
    _assert_calculation_preserved(before, after)
    for tile in (-1, 0, 1, 2):
        events, computations, _ = _execute_k_protocol(after, {after.params[-1]: tile}, hidden_pool=True)
        _assert_balanced_events(events, set())
        assert computations["tl.ascend_gemm_v0"] == int(0 <= tile < 2)


def test_optional_accumulation_completes_before_the_unconditional_fix():
    before = _pre_sync(_conditional_accumulation())[0]
    after = _run(before, target_kind="llvm")
    _assert_calculation_preserved(before, after)
    for choice in (-1, 0, 1):
        events, computations, clears = _execute_k_protocol(after, {after.params[-1]: choice}, hidden_pool=True)
        _assert_balanced_events(events, set())
        assert computations["tl.ascend_gemm_v0"] == 1 + int(choice > 0)
        assert computations["copy_l0c_to_gm"] == 1
        assert clears == ([1, 0] if choice > 0 else [1])


def test_conditional_calls_require_ordinary_automatic_synchronization():
    import tilelang

    before = _pre_sync(_conditional_accumulation())[0]
    target = tvm.target.Target("llvm -model=ascendc")
    with (
        tvm.transform.PassContext(config={"tl.ascend_auto_sync": False, "tl.ascend_auto_sync_vs": True}),
        pytest.raises(tvm.error.TVMError, match="General GEMM invocation.*ordinary automatic synchronization"),
    ):
        tilelang.transform.AscendInsertSync(target, "A3")(tvm.IRModule({"main": before}))


def test_existing_implicit_cube_barrier_keeps_its_single_resource_inference():
    before = _pre_sync(_independent_gemms())[0]
    body = tir.stmt_functor.ir_transform(
        before.body,
        lambda node: node.body if node.attr_key == "resource_scope" else None,
        None,
        ["tir.AttrStmt"],
    )
    barrier = tir.Evaluate(tir.Call("handle", tvm.ir.Op.get("tl.ascend_pipe_barrier"), [tir.StringImm("M")]))
    original = before.with_body(tir.SeqStmt([barrier, body]))
    after = _run(original, target_kind="llvm")
    _assert_calculation_preserved(original, after)


def test_mixed_conditional_refresh_uses_complete_epoch_without_manual_flags():
    from test_ascend_upstream_sync_cross_core_owner_tree import _byte_completion_oracle

    before = _pre_sync(_mixed_conditional_gemm())[0]
    after = _run(before, target_kind="llvm")
    _assert_calculation_preserved(before, after)
    for envelope in (False, True):
        counts = _byte_completion_oracle(after, 2, envelope=envelope, input_count=2, cross_wait_blocks_issue=True)
        assert counts["tl.ascend_gemm_v0"] == 8
        assert counts["gemm_clear"] == counts["gemm_accumulate"] == 4


def test_mixed_conditional_refresh_does_not_grant_allocation_authority():
    before = _pre_sync(_mixed_conditional_gemm().without_attr("tir.noalias"))[0]
    with pytest.raises(tvm.error.TVMError, match="epoch publication requires explicit noalias"):
        _run(before, target_kind="llvm")


def test_epoch_wait_issue_model_does_not_invent_producer_completion():
    from test_ascend_upstream_sync_cross_core_owner_tree import _byte_completion_oracle

    after = _run(_pre_sync(_mixed_conditional_gemm())[0], target_kind="llvm")

    def remove_fix_drain(node):
        if isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call):
            call = node.value
            if getattr(call.op, "name", "") == "tl.ascend_auto_barrier" and call.args[0].value == "PIPE_FIX":
                return tir.Evaluate(0)
        return None

    corrupted = after.with_body(tir.stmt_functor.ir_transform(after.body, remove_fix_drain, None, ["tir.Evaluate"]))
    assert not tvm.ir.structural_equal(after, corrupted)
    with pytest.raises(AssertionError, match="missing asynchronous completion edge"):
        _byte_completion_oracle(corrupted, 2, input_count=2, cross_wait_blocks_issue=True)


@pytest.fixture(scope="module")
def mixed_ir():
    return _pre_sync(_mixed_conditional_gemm())[0]


@pytest.mark.parametrize("automatic", [False, True], ids=["manual", "auto"])
@pytest.mark.parametrize("owner,pipe", [(0, "M"), (0, "MTE1"), (0, "FIX"), (0, "ALL"), (1, "V"), (1, "ALL"), (1, "S")])
def test_invocation_barriers_follow_their_actual_owner(mixed_ir, automatic, owner, pipe):
    name = "tl.ascend_auto_barrier" if automatic else "tl.ascend_pipe_barrier"
    spelling = "PIPE_" + pipe if automatic else pipe
    call = tir.Call("handle", tvm.ir.Op.get(name), [tir.StringImm(spelling)])
    before = _scoped_prefix(mixed_ir, owner, tir.Evaluate(call))
    after = _run(before, target_kind="llvm")
    assert any(tvm.ir.structural_equal(node, call) for node in _nodes(after.body, tir.Call))


@pytest.mark.parametrize("change", ["wrong_owner", "unknown", "nonliteral", "extra", "prefix", "extern"])
def test_invocation_barrier_abi_is_not_delegated_to_purity_check(mixed_ir, change):
    arguments = [tir.StringImm("V")]
    operation = tvm.ir.Op.get("tl.ascend_pipe_barrier")
    owner = 1
    if change == "wrong_owner":
        owner = 0
    elif change == "unknown":
        arguments = [tir.StringImm("UNKNOWN")]
    elif change == "nonliteral":
        arguments = [tir.IntImm("int32", 1)]
    elif change == "extra":
        arguments.append(tir.IntImm("int32", 0))
    elif change == "prefix":
        operation = tvm.ir.Op.get("tl.ascend_auto_barrier")
    else:
        operation = tvm.ir.Op.get("tir.call_extern")
        arguments.insert(0, tir.StringImm("tl::ascend::pipe_barrier"))
    invalid = _scoped_prefix(mixed_ir, owner, tir.Evaluate(tir.Call("handle", operation, arguments)))
    with pytest.raises(tvm.error.TVMError, match="Composite gemm_v0 invocation contract: .*barrier"):
        _run(invalid, target_kind="llvm")


@pytest.mark.parametrize("automatic", [False, True])
def test_epoch_request_normalizes_flat_explicit_flags_and_rejects_automatic(mixed_ir, automatic):
    if automatic:
        pair = [
            tir.Evaluate(
                tir.Call("handle", tvm.ir.Op.get("tl.ascend_auto_" + kind + "_flag"), [tir.StringImm("S_MTE2"), tir.IntImm("int32", 6)])
            )
            for kind in ("set", "wait")
        ]
    else:
        from test_ascend_upstream_sync_composite import _manual_flag

        pair = [_manual_flag(kind, "S_MTE2", 6) for kind in ("set", "wait")]
    before = _scoped_prefix(mixed_ir, 0, tir.SeqStmt(pair))
    if automatic:
        with pytest.raises(tvm.error.TVMError, match="epoch composition cannot retain automatic hard flags"):
            _run(before, target_kind="llvm")
    else:
        after = _run(before, target_kind="llvm")
        assert not any(
            getattr(call.op, "name", "") in ("tl.ascend_set_flag", "tl.ascend_wait_flag") for call in _nodes(after.body, tir.Call)
        )
        _assert_calculation_preserved(before, after)


@pytest.mark.parametrize("escape", ["else", "sibling"])
def test_branch_facts_cannot_justify_another_path(escape):
    before = _pre_sync(_guarded_gemm())[0]
    inner = next(node for node in _nodes(before.body, tir.IfThenElse) if not _nodes(node.then_case, tir.IfThenElse))

    def rewrite(node):
        if node.same_as(inner):
            if escape == "else":
                return tir.IfThenElse(node.condition, node.then_case, node.then_case)
            return tir.SeqStmt([node, node.then_case])
        return None

    invalid = before.with_body(tir.stmt_functor.ir_transform(before.body, rewrite, None, ["tir.IfThenElse"]))
    with pytest.raises(tvm.error.TVMError, match="(?i)(nonwrapping|GM rectangle|GM row)"):
        _run(invalid, target_kind="llvm")


def test_predicate_arithmetic_is_audited_before_its_own_assumption():
    before = _pre_sync(_guarded_gemm())[0]
    outer = _nodes(before.body, tir.IfThenElse)[-1]
    tile = before.params[-1]

    def rewrite(node):
        if node.same_as(outer):
            return tir.IfThenElse(tile + 1 > 0, node.then_case, None)
        return None

    invalid = before.with_body(tir.stmt_functor.ir_transform(before.body, rewrite, None, ["tir.IfThenElse"]))
    with pytest.raises(tvm.error.TVMError, match="nonwrapping signed index arithmetic"):
        _run(invalid, target_kind="llvm")
