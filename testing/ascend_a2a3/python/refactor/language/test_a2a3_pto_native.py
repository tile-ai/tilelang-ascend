"""Source-only PTO codegen contracts on the complete migrated component."""

import pytest

from tilelang import tvm
import tilelang.ascend.a2a3.language as T
from tilelang.ascend.a2a3 import pipeline
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

X = tvm.tirx
_BUILDER = tvm.get_global_func("target.build.tilelang_pto_a2a3", allow_missing=True)
pytestmark = pytest.mark.skipif(_BUILDER is None, reason="PTO source builder requires USE_ASCEND=ON")


def _target(platform="A3"):
    return tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": platform, "keys": ["pto", "ascend"]})


def _source(mod, platform="A3"):
    return _BUILDER(mod, _target(platform), platform).inspect_source()


def _func(body, attrs=None, shapes=None):
    shapes = shapes or [(1,), (1,)]
    buffers = [X.decl_buffer(shape, "int32", name=f"B{i}") for i, shape in enumerate(shapes)]
    params = [X.Var(f"B{i}_handle", "handle") for i in range(len(buffers))]
    func = X.PrimFunc(params, body, buffer_map=dict(zip(params, buffers))).with_attr("global_symbol", "main")
    if attrs:
        func = func.with_attr(attrs)
    return tvm.IRModule.from_expr(func)


def _call(op, *args):
    return X.Evaluate(X.call_intrin("handle", tvm.ir.Op.get("tl." + op), *args))


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_shared_manual_and_auto_sync_schemas(platform):
    body = X.SeqStmt(
        [
            _call("ascend_set_flag", "MTE2_V", 2),
            _call("ascend_wait_flag", "MTE2_V", 2),
            _call("ascend_pipe_barrier", "PIPE_V"),
            _call("ascend_auto_set_flag", "MTE2_V", 2),
            _call("ascend_auto_wait_flag", "MTE2_V", 2),
            _call("ascend_wait_cross_flag", 3, ""),
        ]
    )
    source = _source(_func(body), platform)
    assert "tl::ascend_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (2);" in source
    assert "tl::ascend_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (2);" in source
    assert "pipe_barrier(PIPE_V);" in source
    assert "set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);" in source
    assert "wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);" in source
    assert "tl::ascend_pto::wait_cross_flag(3);" in source
    assert "__DAV_C220" not in source  # No resource scope was requested.


def test_old_cross_flag_invalid_arguments_are_diagnostics():
    with pytest.raises(tvm.error.InternalError, match="PTO wait_cross_flag requires"):
        _source(_func(_call("ascend_wait_cross_flag", 3)))
    with pytest.raises(tvm.error.InternalError, match="pipe must be a StringImm"):
        _source(_func(_call("ascend_wait_cross_flag", 3, 0)))


@pytest.mark.parametrize("name", ["_", "vid"])
@pytest.mark.parametrize("mode", [None, 0, 1])
def test_cthread_origin_and_anonymous_vid_guard_survive_json(name, mode):
    vid = X.Var(name, "int32")
    iv = X.IterVar(None, vid, X.IterVar.ThreadIndex, "cthread")
    body = X.AttrStmt(iv, "thread_extent", 2, X.Evaluate(0))
    attrs = {} if mode is None else {"a2a3_cthread_modes": {vid: X.IntImm("int32", mode)}}
    mod = _func(body, attrs)
    source = _source(mod)
    assert source.count("get_subblockid()") == (0 if mode == 1 and name == "_" else 1)
    assert _source(tvm.ir.load_json(tvm.ir.save_json(mod))) == source


def test_cthread_stale_origin_is_rejected_instead_of_guessed():
    vid = X.Var("vid", "int32")
    iv = X.IterVar(None, vid, X.IterVar.ThreadIndex, "cthread")
    stale = X.Var("vid", "int32")
    mod = _func(X.AttrStmt(iv, "thread_extent", 2, X.Evaluate(0)), {"a2a3_cthread_modes": {stale: X.IntImm("int32", 1)}})
    with pytest.raises(tvm.error.InternalError, match="origin metadata is stale"):
        _source(mod)


@pytest.mark.parametrize("reversed_order", [False, True])
def test_recursive_shape_var_host_order(reversed_order):
    m, n = X.Var("M", "int32"), X.Var("N", "int32")
    first = n + m if reversed_order else m * 2 + n
    source = _source(_func(X.Evaluate(0), shapes=[(first,), (m + n,), (m,), (n,), (2,), (16,)]))
    dims = "int64_t N, int64_t M" if reversed_order else "int64_t M, int64_t N"
    assert f"uint8_t *B5_handle, {dims}, void *stream)" in source
    assert "M1" not in source and "N1" not in source


@pytest.mark.parametrize("threads", [None, 1, 2])
@pytest.mark.parametrize("discard", [False, True])
def test_real_kernel_producer_full_pipeline_origin(threads, discard):
    if threads is None and discard:

        @T.prim_func
        def kernel(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
            with T.Kernel(1, is_npu=True) as (cid, _), T.Scope("C"):
                B[cid] = A[cid]
    elif threads is None:

        @T.prim_func
        def kernel(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
            with T.Kernel(1, is_npu=True) as (cid, vid), T.Scope("C"):
                B[vid] = A[cid]
    elif discard:

        @T.prim_func
        def kernel(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
            with T.Kernel(1, threads=threads, is_npu=True) as cid:
                _ = T.get_thread_binding()
                with T.Scope("C"):
                    B[cid] = A[cid]
    else:

        @T.prim_func
        def kernel(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
            with T.Kernel(1, threads=threads, is_npu=True) as cid:
                vid = T.get_thread_binding()
                with T.Scope("C"):
                    B[vid] = A[cid]

    target = _target()
    with tvm.transform.PassContext(config=normalize_a2a3_pass_configs(target)):
        lowered = pipeline._lower_pto(tvm.IRModule.from_expr(kernel), target)
        lowered = X.transform.Simplify()(lowered)
    func = pipeline.get_device_kernel(lowered)
    modes = func.attrs["a2a3_cthread_modes"]
    assert len(modes) == 1 and int(next(iter(modes.values()))) == (1 if threads is None else 0)
    source = _source(lowered)
    # The old parser renamed a discarded default vid to "_" and omitted this
    # declaration. Eager with-as preserves its initial "vid" name, so it emits
    # one unused query. Keep this explicit frontend representation difference;
    # the direct final-IR tests above still protect the original "_" guard.
    assert source.count("get_subblockid()") == 1
    assert _source(tvm.ir.load_json(tvm.ir.save_json(lowered))) == source


def _literal(dtype, cv):
    values = [-7, 11, 23, 37] if dtype == "int32" else [-7.5, 11.25, 23.5, 37.0]
    size = 33 if cv else 1

    @T.prim_func
    def literal(Input: T.Tensor((1,), "int32"), Output: T.Tensor((size,), dtype)):
        table = T.allocate_const(values, dtype, [4])
        with T.Scope("C"):
            T._src_code("Output_handle[0] = table[Input_handle[0] & 3];", table)
        if cv:
            with T.Scope("V"):
                T._src_code("Output_handle[32] = table[Input_handle[0] & 3];", table)

    return literal


@pytest.mark.parametrize("dtype,cv", [("int32", False), ("float32", False), ("int32", True)])
def test_literal_raw_full_pipeline_json_and_zero_extra_parameters(dtype, cv):
    target = _target()
    configs = normalize_a2a3_pass_configs(target, {"tl.ascend_auto_cv_combine": cv, "tl.ascend_auto_sync": cv})
    with tvm.transform.PassContext(config=configs):
        lowered = pipeline._lower_pto(tvm.IRModule.from_expr(_literal(dtype, cv)), target)
        lowered = X.transform.Simplify()(lowered)
    source = _source(lowered)
    assert len(pipeline.get_device_kernel(lowered).params) == 2
    assert "call(uint8_t *Input_handle, uint8_t *Output_handle, void *stream)" in source
    assert source.count('section(".rodata.tvm")') == 1
    assert source.count("table[Input_handle[0] & 3]") == (2 if cv else 1)
    assert "table_handle" not in source
    assert "__DAV_C220_CUBE__" in source
    if cv:
        assert "__DAV_C220_VEC__" in source
    assert _source(tvm.ir.load_json(tvm.ir.save_json(lowered))) == source
