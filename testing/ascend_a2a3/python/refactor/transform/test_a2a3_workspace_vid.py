"""TA workspace policies and vector splitting through direct native passes."""

import warnings

import pytest

from tilelang import tvm
import tilelang.ascend.a2a3.language as T

from _a2a3_vid_source_kernels import (
    buffer_region_kernel,
    gm_gather_skip_set_dual,
    gm_ub_gm_identity,
    gm_ub_gm_identity_t1,
    parallel_elementwise_1d,
    parallel_elementwise_1d_t1,
    tile_cast_ops_kernel,
    tile_ops_kernel,
    tile_parallel_kernel,
    tile_range_kernel,
)


tir = tvm.tirx
_WORKSPACE = [
    "sigmoid",
    "sin",
    "cos",
    "round",
    "pow",
    "bitwise_xor",
    "clamp_max",
    "clamp_min",
    "clamp",
    "sort",
    "merge_sort",
    "topk",
    "gather_mask",
    "select",
    "gather",
    "broadcast",
    "reduce_sum_experiment",
    "reduce_sum_mask_experiment",
    "bilinear_interpolation",
    "reduce",
]
_PTO_UNSUPPORTED = {"sin", "cos", "bilinear_interpolation", "reduce_sum_experiment", "reduce_sum_mask_experiment"}


def _target(backend="ascendc", model="A3"):
    return tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": model, "keys": [backend, "ascend"]})


def _apply(name, func, *args):
    factory = tvm.ffi.get_global_func("tl.transform.A2A3" + name)
    return factory(*args)(tvm.IRModule({"main": func}))["main"]


def _workspace_function(name, dtype="float32", explicit=False):
    def buffer(name, shape=(8, 64), dtype=dtype):
        return tir.decl_buffer(shape, dtype, name=name, scope="shared.ub")

    dst, src, other = [buffer(name) for name in ("dst", "src", "other")]
    rows, wide, flat = buffer("rows", (8,)), buffer("wide", (2048,)), buffer("flat", (512,))
    arena = buffer("arena", (65536,), "uint8")
    mask, offsets = buffer("mask", (512,), "uint8"), buffer("offsets", (512,), "uint32")
    buffers = [dst, src, other, rows, wide, flat, arena, mask, offsets]
    kw = {"tmp": arena} if explicit else {}
    fn = getattr(T.tile, name) if name != "reduce" else T.reduce_sum
    args = {
        "sigmoid": (dst, src),
        "sin": (dst, src),
        "cos": (dst, src),
        "round": (dst, src, 512),
        "pow": (dst, src, other),
        "bitwise_xor": (dst, src, other),
        "clamp_max": (dst, src, 1.0, 512),
        "clamp_min": (dst, src, 1.0, 512),
        "clamp": (dst, src, -1.0, 1.0, 512),
        "sort": (wide, flat, 512),
        "merge_sort": (wide, flat, flat),
        "topk": (wide, flat, 16, 512),
        "gather_mask": (dst, src, "P0101"),
        "select": (dst, mask, src, 1.0, "VSEL_TENSOR_SCALAR_MODE"),
        "gather": (dst, src, offsets, 0),
        "broadcast": (dst, rows),
        "reduce_sum_experiment": (dst, src, 512),
        "reduce_sum_mask_experiment": (dst, src, 64, 1, 1),
        "bilinear_interpolation": (dst, src, offsets, other, 64, 1, False, 1, 0, 1),
        "reduce": (src, rows),
    }
    if name == "broadcast":
        kw["axis"] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        call = fn(*args[name], **kw)
    block = tir.SBlock([], [], [], "tilelang_root", tir.Evaluate(call), alloc_buffers=buffers)
    return tir.PrimFunc([], tir.SBlockRealize([], True, block))


def _calls(func, op=None):
    found = []

    def visit(node):
        if isinstance(node, tir.Call) and (op is None or node.op.name == op):
            found.append(node)

    tir.stmt_functor.post_order_visit(func.body, visit)
    return found


@pytest.mark.parametrize("name", _WORKSPACE)
@pytest.mark.parametrize("backend", ["ascendc", "pto"])
@pytest.mark.parametrize("explicit", [False, True])
def test_all_workspace_operation_policies(name, backend, explicit):
    func = _workspace_function(name, explicit=explicit)
    if backend == "pto" and name in _PTO_UNSUPPORTED:
        with pytest.raises(tvm.error.InternalError, match="not supported by the PTO backend"):
            _apply("InjectTmpBuffer", func, _target(backend), False)
        return
    result = _apply("InjectTmpBuffer", func, _target(backend), False)
    assert _calls(result)[-1].op.same_as(_calls(func)[-1].op)
    allocations = result.body.block.alloc_buffers
    declared = [buffer.data for buffer in allocations]
    for call in _calls(result, "tirx.tvm_access_ptr"):
        assert any(call.args[1].same_as(data) for data in declared)
    # Workspace writes may not become hidden read dependencies.
    for call in _calls(result, "tirx.tvm_access_ptr"):
        if str(call.args[1].name).startswith("tmp_ub"):
            assert int(call.args[4]) == 2


@pytest.mark.parametrize(
    "target", [tvm.target.Target("llvm"), tvm.target.Target("ascend"), tvm.target.Target({"kind": "ascend", "arch": "dav-3510"})]
)
def test_workspace_rejects_other_target_variants(target):
    with pytest.raises(tvm.error.InternalError, match="ascend arch=dav-2201"):
        _apply("InjectTmpBuffer", _workspace_function("sigmoid"), target, False)


@pytest.mark.parametrize("model", ["A2", "A3", "pto"])
def test_platform_model_does_not_select_workspace_backend(model):
    func = _workspace_function("sigmoid")
    ascend = _apply("InjectTmpBuffer", func, _target("ascendc", model), False)
    pto = _apply("InjectTmpBuffer", func, _target("pto", model), False)
    assert len(_calls(ascend, "tl.ascend_sigmoid")[0].args) == 4
    assert len(_calls(pto, "tl.ascend_sigmoid")[0].args) == 3


def _reduce_func(m=4, n=8, threads=2, explicit_bytes=None, for_binding=False):
    src = tir.decl_buffer((m, n), "float32", name="src", scope="shared.ub")
    dst = tir.decl_buffer((m,), "float32", name="dst", scope="shared.ub")
    buffers = [src, dst]
    kw = {}
    if explicit_bytes is not None:
        arena = tir.decl_buffer((explicit_bytes,), "uint8", name="arena", scope="shared.ub")
        buffers.append(arena)
        kw["tmp"] = arena
    call = T.reduce_max(src, dst, **kw)
    block = tir.SBlock([], [], [], "tilelang_root", tir.Evaluate(call), alloc_buffers=buffers)
    body = tir.SBlockRealize([], True, block)
    vid = tir.Var("vid", "int32")
    iv = tir.IterVar(tvm.ir.Range(0, threads), vid, tir.IterVar.ThreadIndex, "cthread")
    if for_binding:
        body = tir.For(vid, 0, threads, tir.ForKind.THREAD_BINDING, body, iv)
    else:
        body = tir.AttrStmt(iv, "thread_extent", threads, body)
    return tir.PrimFunc([], body).with_attr("npu_cv_ratio", "cv_1_2" if threads == 2 else "cv_1_1")


def _workspace_bytes(func):
    ptr = _calls(func, "tl.ascend_reduce")[0].args[3]
    return int(ptr.args[3]) * tvm.DataType(ptr.args[0].dtype).itemsize


@pytest.mark.parametrize("for_binding", [False, True])
@pytest.mark.parametrize(
    ("m", "n", "threads", "before", "after"), [(4, 8, 2, 576, 288), (8, 64, 2, 1664, 832), (512, 128, 2, 36992, 18496), (4, 8, 1, 288, 288)]
)
def test_reduce_workspace_plan_compensates_for_later_vid_split(for_binding, m, n, threads, before, after):
    func = _reduce_func(m, n, threads, for_binding=for_binding)
    injected = _apply("InjectTmpBuffer", func, _target(), True)
    reduced = _apply("AscendVidReduction", injected)
    assert _workspace_bytes(injected) == before
    assert _workspace_bytes(reduced) == after
    assert reduced.attrs["npu_cv_ratio"] == func.attrs["npu_cv_ratio"]


def test_standalone_workspace_does_not_assume_a_vid_split():
    injected = _apply("InjectTmpBuffer", _reduce_func(), _target(), False)
    assert _workspace_bytes(injected) == 288


def test_vid_rejects_an_arena_that_becomes_too_small():
    with pytest.raises(tvm.error.InternalError, match="explicit tmp arena is too small"):
        _apply("InjectTmpBuffer", _reduce_func(explicit_bytes=288), _target(), True)


@pytest.mark.parametrize(
    ("factory", "args"),
    [
        (gm_ub_gm_identity, (1024, 128, 128, 128)),
        (parallel_elementwise_1d, (1024, 128)),
        (gm_gather_skip_set_dual, (1, 256, 1, 128, 64, 64, 64)),
        (tile_ops_kernel, (256, 128, 64, 128)),
        (tile_cast_ops_kernel, (256, 128, 64, 128)),
        (tile_parallel_kernel, (256, 128, 64, 128)),
        (tile_range_kernel, (256, 128, 64, 128)),
        (buffer_region_kernel, (256, 128, 64, 128)),
        (gm_ub_gm_identity_t1, (1024, 128, 128, 128)),
        (parallel_elementwise_1d_t1, (1024, 128)),
    ],
)
def test_original_vid_kernel_factories_trace_and_run_three_passes(factory, args):
    func = factory(*args)
    inferred = _apply("InferAllocScope", func)
    injected = _apply("InjectTmpBuffer", inferred, _target(), True)
    reduced = _apply("AscendVidReduction", injected)
    assert isinstance(reduced, tir.PrimFunc)
    assert len(_calls(reduced)) > 0
    before, after = {}, {}

    def collect(mapping, node):
        if isinstance(node, tir.SBlock):
            for buffer in node.alloc_buffers:
                if buffer.scope() == "shared.ub":
                    mapping[buffer.name] = [int(x) for x in buffer.shape]

    tir.stmt_functor.post_order_visit(inferred.body, lambda node: collect(before, node))
    tir.stmt_functor.post_order_visit(reduced.body, lambda node: collect(after, node))
    assert before
    skipped = set(reduced.attrs.get("buffers_skip_vid_reduction", []))
    threads_one = factory in (gm_ub_gm_identity_t1, parallel_elementwise_1d_t1)
    for name, shape in before.items():
        expected = shape if threads_one or name in skipped else [max(shape[0] // 2, 1), *shape[1:]]
        assert after[name] == expected
    if factory is gm_gather_skip_set_dual:
        assert {"kv_ub", "kv_tail_ub"}.issubset(skipped)
    assert reduced.attrs["npu_cv_ratio"] == func.attrs["npu_cv_ratio"]


def _manual_vid_kernel():
    @T.prim_func
    def main():
        with T.Kernel(1, is_npu=True) as (_, vid):
            src = T.alloc_ub((4, 8), "float32")
            dst = T.alloc_ub((4,), "float32")
            if vid == 0:
                T.reduce_max(src, dst)

    return main


def test_manual_vid_launch_keeps_the_source_non_splitting_mode():
    func = _manual_vid_kernel()
    injected = _apply("InjectTmpBuffer", func, _target(), True)
    reduced = _apply("AscendVidReduction", injected)
    assert _workspace_bytes(injected) == 288
    assert _workspace_bytes(reduced) == 288
    assert _calls(reduced, "tl.ascend_reduce")[0].args[0].value == "reduce_max<float, 4, 8, -1>"


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("for_binding", [False, True])
def test_unmarked_cthread_does_not_require_cv_ratio_attribute(threads, for_binding):
    func = _reduce_func(threads=threads, for_binding=for_binding).without_attr("npu_cv_ratio")
    injected = _apply("InjectTmpBuffer", func, _target(), True)
    reduced = _apply("AscendVidReduction", injected)
    assert _workspace_bytes(injected) == (576 if threads == 2 else 288)
    assert _workspace_bytes(reduced) == 288


def _clear_reduce_function(dtype="float32", dim=-1, explicit=False):
    src = tir.decl_buffer((8, 64), dtype, name="src", scope="shared.ub")
    dst = tir.decl_buffer((8 if dim == -1 else 64,), dtype, name="dst", scope="shared.ub")
    arena = tir.decl_buffer((65536,), "uint8", name="arena", scope="shared.ub")
    kw = {"tmp": arena} if explicit else {}
    call = T.reduce_sum(src, dst, dim=dim, clear=False, **kw)
    block = tir.SBlock([], [], [], "tilelang_root", tir.Evaluate(call), alloc_buffers=[src, dst, arena])
    return tir.PrimFunc([], tir.SBlockRealize([], True, block))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("explicit", [False, True])
def test_pto_clear_false_has_disjoint_workspace_views_and_is_idempotent(dtype, dim, explicit):
    result = _apply("InjectTmpBuffer", _clear_reduce_function(dtype, dim, explicit), _target("pto"), False)
    call = _calls(result, "tl.ascend_reduce")[0]
    views = [arg for arg in call.args[3:] if isinstance(arg, tir.Call)]
    assert views
    assert all(int(view.args[4]) == 2 for view in views)
    if len(views) == 2 and views[0].args[1].same_as(views[1].args[1]):
        assert int(views[1].args[2]) >= int(views[0].args[2]) + int(views[0].args[3])
    again = _apply("InjectTmpBuffer", result, _target("pto"), False)
    tvm.ir.assert_structural_equal(result, again)
