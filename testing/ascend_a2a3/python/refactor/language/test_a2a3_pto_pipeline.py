"""PTO's original 27 native pass chain and backend-local launch metadata."""

import pytest
from tilelang import tvm
import tilelang.ascend.a2a3.language as T
from tilelang.ascend.a2a3 import codegen, pipeline
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
from tilelang.backend import create_backend_context

from _a2a3_source_kernels import indexer, indexer_dynamic, matmul, matmul_add

_NATIVE = tvm.ffi.get_global_func("tl.transform.A2A3AscendResourceScopeVerify", allow_missing=True) is not None
pytestmark = pytest.mark.skipif(not _NATIVE, reason="complete A2/A3 native compiler is not built")

# Factory order from pinned TA engine/phase.py, excluding the two AscendC-only
# managed-mask factories. HostProcesser is the source Python HostLegalize bridge.
_TA_PASSES = [
    "InferAllocScope",
    "InjectTmpBuffer",
    "AscendVidReduction",
    "BufferShapeCollector",
    "HostLegalize",
    "AscendLowerParallelToVector",
    "LayoutInference",
    "CollectBufferShapes",
    "LowerTileOp",
    "AscendTailMaskPropagation",
    "AscendWorkspaceReduction",
    "LegalizeVectorizedLoop",
    "LegalizeSafeMemoryAccess",
    "CrossCorePipeline",
    "CombineCV",
    "PipelinePlanning",
    "InjectSoftwarePipeline",
    "AscendLowerOpaqueBlock",
    "ConfigIndexBitwidth",
    "Flatten2DBuffer",
    "FlattenBuffer",
    "VectorizeLoop",
    "AscendStorageRewrite",
    "AscendMemoryPlanning",
    "AscendSyncInsert",
    "AscendSyncInsertVS",
    "AscendResourceScopeVerify",
]


@tvm.instrument.pass_instrument
class _Trace:
    def __init__(self):
        self.names = []
        self.prefix = None
        self.entries = []
        self.stack = []

    def run_before_pass(self, mod, info):
        name = str(info.name)
        self.names.append(name)
        index = len(self.entries)
        parent = self.stack[-1] if self.stack else None
        self.entries.append({"name": name, "parent": parent, "children": []})
        if parent is not None:
            self.entries[parent]["children"].append(index)
        self.stack.append(index)
        if name == "tl.A2A3InferAllocScope":
            self.prefix = mod

    def run_after_pass(self, mod, info):
        assert self.entries[self.stack.pop()]["name"] == str(info.name)


def _assert_compat_children(trace):
    assert not trace.stack
    for entry in trace.entries:
        children = [trace.entries[index]["name"] for index in entry["children"]]
        if entry["name"] == "tl.A2A3Simplify":
            assert children == ["tirx.Simplify"]
        elif entry["name"] == "tl.A2A3RemoveNoOp":
            # Fixed-point iterations depend on the actual dead-binding chain.
            # Every round must call the genuine configured shared pass.
            assert children and children == ["tirx.RemoveNoOp"] * len(children)


def _assert_complete_pto_trace(trace):
    compat = {"tl.A2A3Simplify", "tl.A2A3RemoveNoOp"}
    native = [name.removeprefix("tl.A2A3") for name in trace.names if name.startswith("tl.A2A3") and name not in compat]
    assert native == _TA_PASSES

    # Keep the original shared phase boundaries as well as all 27 TA native
    # factories. This rejects moving a wrapper within a native-to-native gap.
    before = {
        "HostLegalize": ["tirx.BindTarget"],
        "CrossCorePipeline": ["s_tir.PlanAndUpdateBufferAllocationLocation"],
        "PipelinePlanning": ["s_tir.PlanAndUpdateBufferAllocationLocation"],
        "ConfigIndexBitwidth": ["tirx.NarrowDataType"],
        "AscendMemoryPlanning": [
            "tirx.UnrollLoop",
            "s_tir.RenormalizeSplitPattern",
            "tl.A2A3Simplify",
            "tl.A2A3RemoveNoOp",
            "s_tir.RewriteUnsafeSelect",
            "s_tir.HoistIfThenElse",
        ],
    }
    after = {
        "HostLegalize": ["tl.A2A3Simplify"],
        "LegalizeSafeMemoryAccess": ["tl.A2A3Simplify"],
        "FlattenBuffer": ["tl.A2A3Simplify"],
    }
    expected = []
    for name in _TA_PASSES:
        expected.extend(before.get(name, []))
        expected.append("tl.A2A3" + name)
        expected.extend(after.get(name, []))
    top_level = [entry["name"] for entry in trace.entries if entry["parent"] is None]
    original_start = top_level.index("tl.A2A3InferAllocScope")
    assert top_level[original_start:] == expected
    assert top_level.count("tl.A2A3Simplify") == 4
    assert top_level.count("tl.A2A3RemoveNoOp") == 1
    _assert_compat_children(trace)


def _context(platform="A3"):
    return create_backend_context({"kind": "ascend", "arch": "dav-2201", "model": platform, "keys": ["pto", "ascend"]}, "c", "cython")


@pytest.mark.parametrize(
    "factory,args,configs",
    [
        (matmul, (256, 256, 128, 128, 128, 64), {}),
        (
            matmul_add,
            (256, 256, 128, 128, 128, 64),
            {
                "tl.ascend_auto_sync": True,
                "tl.ascend_auto_cross_core_sync": True,
                "tl.ascend_auto_cv_combine": True,
                "tl.ascend_memory_planning": True,
            },
        ),
        (indexer, (2, 1, 32, 512, 4096, 64, 1024, 512, 4, 128, 256, 64), {"tl.ascend_memory_planning": True}),
        (indexer_dynamic, (1, 32, 64, 1024, 512, 4, 128, 256, 64), {"tl.ascend_memory_planning": True}),
    ],
)
def test_source_factories_run_the_complete_pto_pass_sequence(factory, args, configs):
    context = _context()
    trace = _Trace()
    original = factory(*args)
    effective = normalize_a2a3_pass_configs(context.target, configs)
    with tvm.transform.PassContext(opt_level=3, config=effective, instruments=[trace]):
        lowered = pipeline._lower_pto(tvm.IRModule.from_expr(original), context.target)
    func = pipeline.get_device_kernel(lowered)
    _assert_complete_pto_trace(trace)
    assert func.attrs["npu_platform"] == "A3"
    assert "pto" in func.attrs["target"].keys
    assert "address_map" in func.attrs and "size_map" in func.attrs
    assert len(pipeline._extract_parameters(lowered)) == len(func.params)
    _assert_modes(lowered, [1])
    assert original.attrs.get("npu_platform") is None
    assert "tl.LowerIntrin" not in trace.names


@pytest.mark.parametrize("threads", [None, 1, 2])
@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("discard_vid", [False, True])
def test_real_launch_forms_and_pto_codegen_tail(threads, platform, discard_vid):
    if threads is None and discard_vid:

        @T.prim_func
        def kernel(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
            with T.Kernel(1, is_npu=True) as (cid, _), T.Scope("C"):
                B[cid] = A[cid]
    elif threads is None:

        @T.prim_func
        def kernel(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
            with T.Kernel(1, is_npu=True) as (cid, vid), T.Scope("C"):
                B[vid] = A[cid]
    elif discard_vid:

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

    context = _context(platform)
    with tvm.transform.PassContext(config=normalize_a2a3_pass_configs(context.target)):
        lowered = pipeline._lower_pto(tvm.IRModule.from_expr(kernel), context.target)
    _assert_modes(lowered, [1 if threads is None else 0])
    trace = _Trace()
    with tvm.transform.PassContext(instruments=[trace]):
        prepared = codegen._prepare_pto_codegen(lowered, context.target)
    assert trace.names == ["tl.A2A3Simplify", "tirx.Simplify"]
    assert [entry["name"] for entry in trace.entries if entry["parent"] is None] == ["tl.A2A3Simplify"]
    _assert_compat_children(trace)
    assert pipeline.get_device_kernel(prepared).attrs["npu_platform"] == platform


def _assert_modes(mod, expected, tail=True):
    variants = [mod, codegen._prepare_pto_codegen(mod, None)] if tail else [mod]
    for variant in variants:
        for candidate in [variant, tvm.ir.load_json(tvm.ir.save_json(variant))]:
            func = pipeline.get_device_kernel(candidate)
            bindings = []
            tvm.tirx.stmt_functor.post_order_visit(
                func.body,
                lambda node, bindings=bindings: (
                    bindings.append(node.node.var)
                    if isinstance(node, tvm.tirx.AttrStmt) and node.attr_key == "thread_extent" and node.node.thread_tag == "cthread"
                    else None
                ),
            )
            modes = func.attrs["a2a3_cthread_modes"]
            assert len(bindings) == len(expected)
            assert all(any(key.same_as(var) for key in modes) for var in bindings)
            assert [int(modes[var]) for var in bindings] == expected


@pytest.mark.parametrize("nested", [False, True])
def test_mixed_real_launches_keep_per_binding_modes_through_complete_chain(nested):
    @T.prim_func
    def kernel(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, is_npu=True):
            with T.Scope("C"):
                B[0] = A[0]
            if nested:
                with T.Kernel(1, threads=1, is_npu=True), T.Scope("C"):
                    B[1] = A[1]
        if not nested:
            with T.Kernel(1, threads=1, is_npu=True), T.Scope("C"):
                B[1] = A[1]

    context = _context()
    trace = _Trace()
    # The pinned TA pass explicitly requires exactly one tilelang_root, so
    # preserve its multi-launch rejection instead of changing that algorithm.
    with (
        tvm.transform.PassContext(config=normalize_a2a3_pass_configs(context.target), instruments=[trace]),
        pytest.raises(tvm.error.InternalError, match="Expected exactly one tilelang_root block"),
    ):
        pipeline._lower_pto(tvm.IRModule.from_expr(kernel), context.target)
    # The shared prefix leaves the nested launch as thread-binding For nodes;
    # it never reaches codegen because the original unique-root gate rejects it.
    _assert_modes(trace.prefix, [1] if nested else [1, 0], tail=False)


@pytest.mark.parametrize("inner_for", [False, True])
def test_raw_outer_binding_never_borrows_the_inner_launch_marker(inner_for):
    tir = tvm.tirx
    inner = tir.IterVar(None, tir.Var("_", "int32"), tir.IterVar.ThreadIndex, "cthread")
    outer = tir.IterVar(None, tir.Var("_", "int32"), tir.IterVar.ThreadIndex, "cthread")
    block = tir.SBlock([], [], [], "tilelang_root", tir.Evaluate(0), annotations={"tilelang.is_npu_kernel_frame": True})
    inner_body = tir.SBlockRealize([], True, block)
    inner_body = (
        tir.For(inner.var, 0, 2, tir.ForKind.THREAD_BINDING, inner_body, thread_binding=inner)
        if inner_for
        else tir.AttrStmt(inner, "thread_extent", 2, inner_body)
    )
    body = tir.AttrStmt(outer, "thread_extent", 2, inner_body)
    func = tir.PrimFunc([], body).with_attr({"global_symbol": "main", "calling_conv": tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH})
    _assert_modes(pipeline._record_pto_cthread_modes(tvm.IRModule.from_expr(func)), [0] if inner_for else [1, 0], tail=False)


def test_candidate_chain_rejects_the_other_dialect():
    target = create_backend_context({"kind": "ascend", "arch": "dav-2201", "model": "A3"}, "c", "cython").target
    with pytest.raises(ValueError, match="with the PTO dialect key"):
        pipeline._lower_pto(tvm.IRModule({}), target)


def test_missing_pto_native_codegen_has_a_build_diagnostic(monkeypatch):
    original = tvm.ffi.get_global_func

    def lookup(name, allow_missing=False):
        return None if name == "target.build.tilelang_pto_a2a3" else original(name, allow_missing=allow_missing)

    monkeypatch.setattr(tvm.ffi, "get_global_func", lookup)
    with pytest.raises(RuntimeError, match="USE_ASCEND=ON.*missing target.build.tilelang_pto_a2a3"):
        codegen._build_pto(tvm.IRModule({}), _context().target)
