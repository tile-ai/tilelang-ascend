"""Complete pinned TA lowering for the A2/A3 AscendC and PTO dialects."""

from __future__ import annotations

import tilelang
from tilelang import tvm
from tvm import s_tir, tirx
from tvm.ir import CallingConv

from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.pass_pipeline.pipeline_utils import allow_vectorize

from . import transform as ascend_transform
from .platform import resolve_a2a3_platform
from .transform.constant import normalize_constants
from .target import target_is_ascend_a2a3, target_is_pto_a2a3


def get_device_kernel(mod):
    """Return the single exported A2/A3 device kernel without choosing by order."""
    functions = list(mod.functions.values())
    if len(functions) != 1 or not isinstance(functions[0], tvm.tirx.PrimFunc):
        raise ValueError("A2/A3 lowering requires exactly one exported device PrimFunc; multi-function modules are not supported yet.")
    func = functions[0]
    if not func.attrs or not func.attrs.get("global_symbol"):
        raise ValueError("A2/A3 device kernel requires a global_symbol export.")
    if int(func.attrs.get("calling_conv", CallingConv.DEFAULT)) != int(CallingConv.DEVICE_KERNEL_LAUNCH):
        raise ValueError("A2/A3 lowered kernel must use DEVICE_KERNEL_LAUNCH; explicit host calling conventions are unsupported.")
    return func


def _extract_parameters(mod):
    # Delay the engine import until compilation to keep backend registration
    # independent of the engine's import of BackendContext.
    from tilelang.engine.lower import extrac_params

    return extrac_params(get_device_kernel(mod))


# All 29 source phase factories plus the proposal-retained inactive component
# must be available before the public path starts transforming frontend IR.
_REQUIRED_NATIVE_FACTORIES = (
    "tl.transform.A2A3AnnotateDeviceRegions",
    "tl.transform.A2A3AscendLowerOpaqueBlock",
    "tl.transform.A2A3AscendLowerParallelToVector",
    "tl.transform.A2A3AscendMemoryPlanning",
    "tl.transform.A2A3AscendResourceScopeVerify",
    "tl.transform.A2A3AscendStorageRewrite",
    "tl.transform.A2A3AscendSyncInsert",
    "tl.transform.A2A3AscendSyncInsertVS",
    "tl.transform.A2A3AscendTailMaskPropagation",
    "tl.transform.A2A3AscendVectorInstructionSelection",
    "tl.transform.A2A3AscendVectorMaskLegalize",
    "tl.transform.A2A3AscendVidReduction",
    "tl.transform.A2A3AscendWorkspaceReduction",
    "tl.transform.A2A3BufferShapeCollector",
    "tl.transform.A2A3CollectBufferShapes",
    "tl.transform.A2A3CombineCV",
    "tl.transform.A2A3RemoveNoOp",
    "tl.transform.A2A3Simplify",
    "tl.transform.A2A3ConfigIndexBitwidth",
    "tl.transform.A2A3CrossCorePipeline",
    "tl.transform.A2A3Flatten2DBuffer",
    "tl.transform.A2A3FlattenBuffer",
    "tl.transform.A2A3HostLegalize",
    "tl.transform.A2A3InferAllocScope",
    "tl.transform.A2A3InjectSoftwarePipeline",
    "tl.transform.A2A3InjectTmpBuffer",
    "tl.transform.A2A3LayoutInference",
    "tl.transform.A2A3LegalizeSafeMemoryAccess",
    "tl.transform.A2A3LegalizeVectorizedLoop",
    "tl.transform.A2A3LowerTileOp",
    "tl.transform.A2A3PipelinePlanning",
    "tl.transform.A2A3VectorizeLoop",
)


def _require_native_components():
    missing = [name for name in _REQUIRED_NATIVE_FACTORIES if tvm.ffi.get_global_func(name, allow_missing=True) is None]
    if missing:
        raise RuntimeError("The complete A2/A3 native compiler is required; rebuild with USE_ASCEND=ON. Missing: " + ", ".join(missing))


def _prepare_source_kernel(mod, platform):
    functions = list(mod.functions.items())
    if len(functions) != 1 or not isinstance(functions[0][1], tirx.PrimFunc):
        raise ValueError("A2/A3 lowering requires exactly one exported source PrimFunc; multi-function modules are not supported yet.")
    symbol, func = functions[0]
    if not func.attrs or not func.attrs.get("global_symbol"):
        raise ValueError("A2/A3 source kernel requires a global_symbol export.")
    if "calling_conv" in func.attrs and int(func.attrs["calling_conv"]) != int(CallingConv.DEVICE_KERNEL_LAUNCH):
        raise ValueError("A2/A3 source kernel conflicts with an explicit host calling convention; use a device kernel export.")
    # TA compiles the exported PrimFunc directly, including legal kernels that
    # do not use T.Kernel. Preserve body/params and every existing attribute.
    result = mod.clone()
    result[symbol] = normalize_constants(func).with_attr({"npu_platform": platform, "calling_conv": CallingConv.DEVICE_KERNEL_LAUNCH})
    return result


def _record_pto_cthread_modes(mod):
    """Keep each launch's old vid spelling policy after SBlock legalization."""
    result = mod.clone()
    for symbol, func in mod.functions.items():
        modes = {}

        def collect(node, modes=modes):
            if not isinstance(node, tirx.AttrStmt) or node.attr_key != "thread_extent" or node.node.thread_tag != "cthread":
                return
            mode = None

            def nearest_marker(child):
                nonlocal mode
                if mode is not None:
                    return False
                # An inner launch owns its own marker, even if the outer raw
                # cthread has none. Never borrow a nested launch's policy.
                if isinstance(child, tirx.AttrStmt) and child.attr_key == "thread_extent" and child.node.thread_tag == "cthread":
                    return False
                if isinstance(child, tirx.For) and child.thread_binding is not None and child.thread_binding.thread_tag == "cthread":
                    return False
                if isinstance(child, tirx.SBlock):
                    if "tilelang.is_npu_kernel_frame_dev_mode" in child.annotations:
                        mode = 0
                    elif "tilelang.is_npu_kernel_frame" in child.annotations:
                        mode = 1
                return mode is None

            tirx.stmt_functor.pre_order_visit(node.body, nearest_marker)
            modes[node.node.var] = tirx.IntImm("int32", 0 if mode is None else mode)

        tirx.stmt_functor.post_order_visit(func.body, collect)
        result[symbol] = func.with_attr("a2a3_cthread_modes", modes)
    return result


def LowerAndLegalize(mod, target):
    """Preserve TA 0deb8b93 phase.py LowerAndLegalize order."""
    mod = ascend_transform.AscendInferBufferScope()(mod)
    mod = ascend_transform.InjectTmpBuffer(target, plan_vid_reduction=True)(mod)
    mod = ascend_transform.AscendVidReduction()(mod)
    mod = ascend_transform.BufferShapeCollector()(mod)
    mod = tirx.transform.BindTarget(target)(mod)
    mod = ascend_transform.HostProcesser()(mod)
    mod = ascend_transform.AscendSimplify()(mod)
    mod = ascend_transform.AscendLowerParallelToVector()(mod)
    mod = ascend_transform.LayoutInference()(mod)
    mod = ascend_transform.CollectBufferShapes()(mod)
    mod = ascend_transform.LowerTileOp()(mod)
    rewrite_reduce = target_is_ascend_a2a3(target) or target_is_pto_a2a3(target)
    mod = ascend_transform.AscendTailMaskPropagation(rewrite_reduce=rewrite_reduce)(mod)
    mod = ascend_transform.AscendWorkspaceReduction()(mod)
    mod = ascend_transform.LegalizeVectorizedLoop()(mod)
    mod = ascend_transform.LegalizeSafeMemoryAccess()(mod)
    mod = ascend_transform.AscendSimplify()(mod)
    return mod


def OptimizeForTarget(mod, target, platform):
    """Preserve the complete TA optimizer; mask legalization remains last."""
    pass_ctx = tilelang.transform.get_pass_context()
    managed_vector_mask = target_is_ascend_a2a3(target) and platform in {"A2", "A3"}
    mod = s_tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = ascend_transform.CrossCorePipeline()(mod)
    mod = ascend_transform.CombineCV()(mod)
    mod = s_tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = ascend_transform.PipelinePlanning()(mod)
    mod = ascend_transform.InjectSoftwarePipeline()(mod)
    mod = ascend_transform.AscendLowerOpaqueBlock()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = ascend_transform.ConfigIndexBitwidth()(mod)
    mod = ascend_transform.Flatten2DBuffer()(mod)
    mod = ascend_transform.FlattenBuffer()(mod)
    mod = ascend_transform.AscendSimplify()(mod)
    mod = ascend_transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    # NPU storage semantics follow this resolved target variant, not the host's
    # optional torch.npu availability or simulator state.
    mod = ascend_transform.AscendStorageRewrite(is_npu=True)(mod)
    mod = tirx.transform.UnrollLoop()(mod)
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = ascend_transform.AscendSimplify()(mod)
    mod = ascend_transform.RemoveNoOp()(mod)
    mod = s_tir.transform.RewriteUnsafeSelect()(mod)
    mod = s_tir.transform.HoistIfThenElse()(mod)
    mod = ascend_transform.AscendMemoryPlanning()(mod)
    mod = ascend_transform.AscendSyncInsert(target, platform)(mod)
    mod = ascend_transform.AscendSyncInsertVS(target, platform)(mod)
    if managed_vector_mask:
        mod = ascend_transform.AscendSimplify()(mod)
    mod = ascend_transform.AscendResourceScopeVerify()(mod)
    if managed_vector_mask:
        mod = ascend_transform.AscendVectorInstructionSelection(target, platform)(mod)
        mod = ascend_transform.AscendVectorMaskLegalize(target, platform)(mod)
    return mod


def _lower_ascendc(mod, target):
    if not target_is_ascend_a2a3(target):
        raise ValueError("A2/A3 AscendC lowering requires arch=dav-2201 without the PTO dialect key.")
    return _lower_a2a3(mod, target)


def _lower_pto(mod, target):
    if not target_is_pto_a2a3(target):
        raise ValueError("A2/A3 PTO lowering requires arch=dav-2201 with the PTO dialect key.")
    return _lower_a2a3(mod, target)


def _lower_a2a3(mod, target):
    platform = resolve_a2a3_platform(target)
    _require_native_components()
    mod = _prepare_source_kernel(mod, platform)
    # Required TIRX frontend adaptation precedes the original phase sequence.
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        default_threads=None,
        unsupported_annotations=["cluster_dims"],
        launch_dim_tags=["cthread"],
    )(mod)
    if target_is_pto_a2a3(target):
        mod = _record_pto_cthread_modes(mod)
    mod = LowerAndLegalize(mod, target)
    return OptimizeForTarget(mod, target, platform)


ASCENDC_PIPELINE = PassPipeline("ascend", _lower_ascendc, parameter_extractor=_extract_parameters)
PTO_PIPELINE = PassPipeline("ascend", _lower_pto, parameter_extractor=_extract_parameters)
