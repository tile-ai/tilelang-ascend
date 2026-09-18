"""Regressions for resource partitioning that must preserve statement semantics."""

import pytest

import tilelang
import tilelang.language as T
from tvm import IRModule, tir


def _scope(owner, body):
    return tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", owner, body)


def _module(body, *buffers):
    block = tir.Block([], [], [], "tilelang_root", body)
    root = tir.BlockRealize([], True, block)
    return IRModule({"main": tir.PrimFunc([buffer.data for buffer in buffers], root)})


def _combine(module):
    with tilelang.transform.PassContext(config={"tl.ascend_auto_cv_combine": True}):
        combined = tilelang.transform.CombineCV()(module)
    tilelang.transform.AscendResourceScopeVerify()(combined)
    return combined["main"].body.block.body.seq


def _nodes(statement, node_type):
    result = []
    tir.stmt_functor.post_order_visit(statement, lambda node: result.append(node) if isinstance(node, node_type) else None)
    return result


def _calls(statement, name):
    return [call for call in _nodes(statement, tir.Call) if getattr(call.op, "name", None) == name]


def _dcci(buffer):
    return tir.Evaluate(T.tile.datacachecleanandinvalid_experiment(buffer, "SINGLE_CACHE_LINE", "CACHELINE_OUT"))


def test_dcci_explicit_owners_and_ub_restriction():
    gm = tir.decl_buffer((64,), "float32", name="gm", scope="global")
    ub = tir.decl_buffer((64,), "float32", name="ub", scope="shared.ub")
    both = _module(tir.SeqStmt([_scope(0, _dcci(gm)), _scope(1, _dcci(gm))]), gm)
    branches = _combine(both)
    for branch in branches:
        assert len(_calls(branch, "tl.ascend_datacachecleanandinvalid_experiment")) == 1

    tilelang.transform.AscendResourceScopeVerify()(_module(_scope(1, _dcci(ub)), ub))
    with pytest.raises(Exception, match="Vector operation must be inside"):
        tilelang.transform.AscendResourceScopeVerify()(_module(_scope(0, _dcci(ub)), ub))


def test_dcci_uses_context_including_scalar_store_ownership():
    gm = tir.decl_buffer((64,), "float32", name="gm", scope="global")
    ub = tir.decl_buffer((64,), "float32", name="ub", scope="shared.ub")
    cube = tir.BufferStore(gm, tir.FloatImm("float32", 1), [0])
    vector = tir.Evaluate(T.tile.fill(ub, 0.0))
    dcci = _dcci(gm)

    for body, owner in [
        (tir.SeqStmt([cube, dcci]), 0),
        (tir.SeqStmt([vector, dcci, vector]), 1),
        (tir.SeqStmt([vector, cube, dcci, cube]), 0),
    ]:
        branches = _combine(_module(body, gm, ub))
        counts = [len(_calls(branch, "tl.ascend_datacachecleanandinvalid_experiment")) for branch in branches]
        assert counts[owner] == 1
        assert counts[1 - owner] == 0

    for body in [dcci, tir.SeqStmt([vector, cube, dcci, vector])]:
        with pytest.raises(Exception, match="must be inside T.Scope"):
            _combine(_module(body, gm, ub))


def test_shared_scalar_gm_reads_do_not_prove_one_dcci_owner():
    gm = tir.decl_buffer((64,), "float32", name="gm", scope="global")
    ub = tir.decl_buffer((64,), "float32", name="ub", scope="shared.ub")
    value = tir.Var("value", "float32")
    shared = tir.LetStmt(value, gm[0], tir.SeqStmt([tir.Evaluate(T.tile.fill(ub, value)), _dcci(gm)]))
    with pytest.raises(Exception, match="must be inside T.Scope"):
        _combine(_module(shared, gm, ub))

    # This read is evaluated only by the concretely Vector-owned fill call.
    owned_read = tir.Evaluate(T.tile.fill(ub, gm[0]))
    for statement in [owned_read, _scope(1, owned_read)]:
        owned = tir.SeqStmt([statement, _dcci(gm)])
        branches = _combine(_module(owned, gm, ub))
        assert not _calls(branches[0], "tl.ascend_datacachecleanandinvalid_experiment")
        assert len(_calls(branches[1], "tl.ascend_datacachecleanandinvalid_experiment")) == 1


@pytest.mark.parametrize("loop_kind", ["for", "while"])
def test_shared_loop_control_retains_scalar_dependencies_on_both_sides(loop_kind):
    gm = tir.decl_buffer((1,), "float32", name="gm", scope="global")
    ub = tir.decl_buffer((64,), "float32", name="ub", scope="shared.ub")
    state = tir.decl_buffer((1,), "int32", name="state", scope="local.var")
    limit = tir.Var("limit", "int32")
    body = tir.SeqStmt(
        [
            _scope(1, tir.Evaluate(T.tile.fill(ub, 1.0))),
            tir.BufferStore(gm, tir.Cast("float32", state[0]), [0]),
            tir.BufferStore(state, state[0] + 1, [0]),
            tir.IfThenElse(state[0] >= limit, tir.Evaluate(T.loop_break()), None),
        ]
    )
    if loop_kind == "for":
        loop = tir.For(tir.Var("i", "int32"), 0, 10, tir.ForKind.SERIAL, body)
    else:
        loop = tir.While(state[0] < 10, body)
    shared = tir.LetStmt(limit, tir.IntImm("int32", 3), tir.SeqStmt([tir.BufferStore(state, 0, [0]), loop]))
    branches = _combine(_module(shared, gm, ub, state))
    for branch in branches:
        assert len(_nodes(branch, tir.LetStmt)) == 1
        assert len(_nodes(branch, tir.For if loop_kind == "for" else tir.While)) == 1
        assert len(_calls(branch, "tl.loop_break")) == 1
        stores = [store for store in _nodes(branch, tir.BufferStore) if store.buffer.same_as(state)]
        assert len(stores) == 2  # initialization and loop-carried update
    assert not _calls(branches[0], "tl.ascend_fill")
    assert len(_calls(branches[1], "tl.ascend_fill")) == 1


def test_explicit_loop_control_stays_on_its_owner():
    body = tir.For(tir.Var("i", "int32"), 0, 10, tir.ForKind.SERIAL, tir.Evaluate(T.loop_break()))
    branches = _combine(_module(_scope(1, body)))
    assert not _calls(branches[0], "tl.loop_break")
    assert len(_calls(branches[1], "tl.loop_break")) == 1


def test_empty_pipeline_preserves_loop_and_condition_evaluation():
    from tvm import ir
    from tvm.target import Target

    i = tir.Var("i", "int32")
    condition = tir.call_extern("bool", "condition", i)
    branch = tir.IfThenElse(condition, tir.Evaluate(0), tir.Evaluate(0))
    scratch = tir.decl_buffer((16,), "float32", scope="shared.ub")
    body = tir.BlockRealize([], True, tir.Block([], [], [], "empty", branch, alloc_buffers=[scratch]))
    loop = tir.For(
        i,
        tir.call_extern("int32", "loop_min"),
        tir.call_extern("int32", "loop_extent"),
        tir.ForKind.SERIAL,
        body,
        annotations={"num_stages": 2, "software_pipeline_stage": [0], "software_pipeline_order": [0], "preserved": 1},
    )
    function = tir.PrimFunc([], loop).with_attr("target", Target({"kind": "llvm", "model": "ascendc"}))
    planned = tilelang.transform.PipelinePlanning()(IRModule({"main": function}))
    result = planned["main"].body
    ir.assert_structural_equal(result.min, loop.min)
    ir.assert_structural_equal(result.extent, loop.extent)
    ir.assert_structural_equal(result.body, loop.body)
    assert set(result.annotations) == {"preserved"}
    cleaned = tir.transform.RemoveNoOp()(tilelang.transform.InjectSoftwarePipeline()(planned))
    calls = _calls(cleaned["main"].body, "tir.call_extern")
    assert sorted(call.args[0].value for call in calls) == ["condition", "loop_extent", "loop_min"]


@pytest.mark.parametrize("dynamic", [False, True])
def test_pipeline_with_nested_serial_loop_lowers_after_cv_split(dynamic):
    @T.prim_func
    def main(A: T.Tensor((4, 64), "float32"), B: T.Tensor((4, 64), "float32"), repeats: T.int32):
        with T.Kernel(1, threads=1, is_npu=True):
            a = T.alloc_ub((64,), "float32")
            b = T.alloc_ub((64,), "float32")
            for k in T.Pipelined(4, num_stages=2):
                T.copy(A[k, :], a)
                if dynamic:
                    for _r in T.serial(repeats):
                        T.tile.add(b, a, a)
                else:
                    for _r in T.serial(2):
                        T.tile.add(b, a, a)
                T.copy(b, B[k, :])

    config = {
        "tl.ascend_auto_sync": True,
        "tl.ascend_auto_cv_combine": True,
        "tl.ascend_auto_cross_core_sync": True,
        "tl.ascend_memory_planning": True,
    }
    with tilelang.transform.PassContext(config=config):
        source = tilelang.lower(main, target="ascendc", platform="A3").kernel_source
    assert "AscendC::Add<float, false>" in source
    assert "copy_gm_to_ub" in source and "copy_ub_to_gm" in source
    if dynamic:
        assert "< repeats;" in source


def test_nested_empty_pipelines_preserve_control_and_clear_all_scheduling():
    from tvm import ir
    from tvm.target import Target

    n = tir.Var("n", "int32")
    i, j = tir.Var("i", "int32"), tir.Var("j", "int32")
    annotations = {
        "num_stages": 2,
        "tl_pipeline_order": [0],
        "tl_pipeline_stage": [0],
        "software_pipeline_order": [0],
        "software_pipeline_stage": [0],
        "software_pipeline_async_stages": [0],
        "preserved": 7,
    }
    inner = tir.For(j, 1, n, tir.ForKind.SERIAL, tir.Evaluate(0), annotations=annotations)
    body = tir.While(n > 0, inner)
    outer = tir.For(i, 0, 4, tir.ForKind.SERIAL, body, annotations=annotations)
    function = tir.PrimFunc([n], outer).with_attr("target", Target({"kind": "llvm", "model": "ascendc"}))
    planned = tilelang.transform.PipelinePlanning()(IRModule({"main": function}))
    injected = tilelang.transform.InjectSoftwarePipeline()(planned)
    expected_inner = tir.For(j, 1, n, tir.ForKind.SERIAL, tir.Evaluate(0), annotations={"preserved": 7})
    expected = tir.For(i, 0, 4, tir.ForKind.SERIAL, tir.While(n > 0, expected_inner), annotations={"preserved": 7})
    ir.assert_structural_equal(planned["main"].body, expected)
    ir.assert_structural_equal(injected["main"].body, expected)


def test_explicit_blocks_keep_separate_cpp_local_scopes():
    @T.prim_func
    def main(A: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=1, is_npu=True):
            ub = T.alloc_ub((64,), "float32")
            with T.Scope("V"):
                T.tile.fill(ub, 0.0)
                T._src_code("int scalar_value = 1;")
                T._src_code("A.SetValue(0, scalar_value);")
            with T.Scope("V"):
                T.tile.fill(ub, 1.0)
                T._src_code("int scalar_value = 2;")
                T._src_code("A.SetValue(1, scalar_value);")

    with tilelang.transform.PassContext(config={"tl.ascend_auto_cv_combine": True}):
        source = tilelang.lower(main, target="ascendc", platform="A3").kernel_source
    first = source.index("int scalar_value = 1;")
    second = source.index("int scalar_value = 2;")
    between = source[first:second]
    assert "}" in between
    assert "if ASCEND_IS_AIV {" in between


def test_explicit_gm_dcci_reaches_both_codegen_paths():
    @T.prim_func
    def main(A: T.Tensor((16,), "int32")):
        with T.Kernel(1, threads=1, is_npu=True):
            with T.Scope("C"):
                A[0] = 1
                T.tile.datacachecleanandinvalid_experiment(A, "SINGLE_CACHE_LINE", "CACHELINE_OUT")
            with T.Scope("V"):
                A[1] = 2
                T.tile.datacachecleanandinvalid_experiment(A, "SINGLE_CACHE_LINE", "CACHELINE_OUT")

    with tilelang.transform.PassContext(config={"tl.ascend_auto_sync": True}):
        source = tilelang.lower(main, target="ascendc", platform="A3").kernel_source
    assert source.count("DataCacheCleanAndInvalid") == 2
    assert "HardEvent::S_V" not in source
    assert "PipeBarrier<PIPE_S>" not in source
