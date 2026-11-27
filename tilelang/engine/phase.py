# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
from __future__ import annotations
from tvm import tir, IRModule
from tvm.target import Target
import tilelang
from tilelang.transform import PassContext
from tilelang.contrib.nvcc import have_tma


def allow_warp_specialized(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    # avoid circular import
    from tilelang.jit.adapter.utils import is_cuda_target

    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if not is_cuda_target(target):
        return False
    disable_warp_specialized = pass_ctx.config.get("tl.disable_warp_specialized", False)
    return not disable_warp_specialized


def allow_tma_and_warp_specialized(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    # avoid circular import
    from tilelang.jit.adapter.utils import is_cuda_target

    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if not is_cuda_target(target) or not have_tma(target):
        return False
    disable_tma_lower = pass_ctx.config.get("tl.disable_tma_lower", False)
    return not disable_tma_lower and allow_warp_specialized(pass_ctx=pass_ctx, target=target)


def allow_fence_proxy(target: Target | None = None) -> bool:
    # avoid circular import
    from tilelang.jit.adapter.utils import is_cuda_target

    return is_cuda_target(target) and have_tma(target)


def allow_vectorize(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    disable_vectorize = pass_ctx.config.get("tir.disable_vectorize", False)
    return not disable_vectorize


def _ascend_sync_strategy(pass_ctx: PassContext, target: Target, platform: str | None = None) -> str:
    strategy = str(pass_ctx.config.get("tl.ascend_sync_strategy", "legacy"))
    if strategy not in {"legacy", "a3"}:
        raise ValueError(f"Unknown Ascend synchronization strategy: {strategy!r}")
    if strategy == "a3":
        if target.model not in {"ascendc", "auto"}:
            raise ValueError("The a3 synchronization strategy requires the AscendC backend")
        if platform is not None and platform != "A3":
            raise ValueError("The a3 synchronization strategy requires platform A3")
    return strategy


def _ascend_pipeline_planner(pass_ctx: PassContext, target: Target, platform: str | None = None) -> str:
    planner = str(pass_ctx.config.get("tl.ascend_pipeline_planner", "legacy"))
    if planner not in {"legacy", "constraint", "auto"}:
        raise ValueError(f"Unknown Ascend pipeline planner strategy: {planner!r}")
    if planner == "auto":
        if target.model not in {"ascendc", "auto"} or (platform is not None and platform != "A3"):
            raise ValueError("The auto scheduler requires the A3 AscendC backend")
        required = {
            "tl.ascend_auto_sync": True,
            "tl.ascend_memory_planning": True,
            "tl.ascend_auto_cv_combine": True,
            "tl.ascend_auto_cross_core_sync": False,
            "tl.ascend_auto_sync_vs": False,
        }
        defaults = {"tl.ascend_auto_cv_combine": True}
        for key, expected in required.items():
            if bool(pass_ctx.config.get(key, defaults.get(key, False))) != expected:
                raise ValueError(f"The auto scheduler requires {key}={expected}")
        if str(pass_ctx.config.get("tl.ascend_sync_strategy", "legacy")) != "a3":
            raise ValueError("The auto scheduler requires tl.ascend_sync_strategy='a3'")
        if pass_ctx.config.get("tl.ascend_pipeline_cost_model") is not None:
            raise ValueError("The auto scheduler uses abstract_issue_step_v1; a constraint latency cost model cannot be supplied")
    return planner


def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    pass_ctx = tilelang.transform.get_pass_context()
    _ascend_sync_strategy(pass_ctx, target)
    planner = _ascend_pipeline_planner(pass_ctx, target)
    if planner == "constraint":
        mod = tilelang.transform.AscendCanonicalizeStorageAliases()(mod)
        mod = tilelang.transform.AscendPrepareConstraintPipeline(target)(mod)
    elif planner == "auto":
        mod = tilelang.transform.AscendPrepareAutoSchedule(target)(mod)
    mod = tilelang.transform.AscendValidateScheduleHints()(mod)
    # Workspace sizing must see the same UB scopes as VidReduction.
    mod = tilelang.transform.AscendInferBufferScope()(mod)
    # allocate the tmp buffer for vector api
    mod = tilelang.transform.InjectTmpBuffer(target, plan_vid_reduction=True)(mod)
    # Vid reduction
    mod = tilelang.transform.AscendVidReduction()(mod)
    # Collect buffer shape
    mod = tilelang.transform.BufferShapeCollector()(mod)
    # Bind the target device information to the module
    mod = tir.transform.BindTarget(target)(mod)
    # Identify and filter host tiling data for npu
    mod = tilelang.transform.HostProcesser()(mod)
    # mod = tilelang.transform.FrontendLegalize()(mod)
    # Simplify the IR expressions
    mod = tir.transform.Simplify()(mod)
    # Lower parallel loops to vector instructions for Ascend.
    mod = tilelang.transform.AscendLowerParallelToVector()(mod)
    # Infer memory layouts for fragments and shared memory
    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.CollectBufferShapes()(mod)
    # Lower high-level tile operations to low-level operations
    mod = tilelang.transform.LowerTileOp()(mod)
    # Propagate UB tail valid-regions and rewrite supported vector ops,
    # compare/select, broadcast, and allow-listed reductions to tail-aware
    # variants (must run before passes that reorder copy/vector ops).
    # The pass self-gates on TL_ASCEND_TAIL_MASK (default off) and is a no-op
    # otherwise, so non-tail kernels are unaffected. The pass itself keeps a
    # strict allow-list for float32 sum/max/min over axis 0 of a 2D tile. Both
    # AscendC and PTO lower that contract to backend-native valid-region code;
    # last-axis reduction and all other forms stay on the established full-tile
    # + pad_value path.
    rewrite_reduce = target.model in {"ascendc", "pto", "auto"}
    mod = tilelang.transform.AscendTailMaskPropagation(rewrite_reduce=rewrite_reduce)(mod)
    # Erase manual workspace allocations for virtual CV copy in Ascend
    mod = tilelang.transform.AscendWorkspaceReduction()(mod)
    # Legalize vectorized loops to ensure they are valid
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    # Add safety checks for memory accesses
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    # Simplify again to clean up any duplicated conditions
    # that may have been introduced by safety checks
    mod = tir.transform.Simplify()(mod)
    # Try to vectorize loop with dynamic shape
    # mod = tilelang.transform.LoopVectorizeDynamic()(mod)
    return mod


def OptimizeForTarget(mod: IRModule, target: Target, platform: str) -> IRModule:
    from tilelang.utils.target import check_npu_availability

    pass_ctx = tilelang.transform.get_pass_context()
    planner = _ascend_pipeline_planner(pass_ctx, target, platform)
    sync_strategy = _ascend_sync_strategy(pass_ctx, target, platform)
    if planner == "constraint":
        mod = tilelang.transform.AscendCanonicalizeStorageAliases()(mod)
    if planner != "auto":
        mod = tilelang.transform.AscendLowerScheduleHints(target, platform)(mod)
    managed_vector_mask = target.model in {"ascendc", "auto"} and platform in {"A2", "A3"}
    mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    if planner == "auto":
        mod = tilelang.transform.AscendAutoSchedule(target, platform)(mod)
        mod = tilelang.transform.AscendMaterializeSchedule(target, platform)(mod)
    else:
        if sync_strategy == "a3" and planner == "legacy":
            mod = tilelang.transform.AscendPrepareSyncHandoff(target, platform)(mod)
        mod = tilelang.transform.CrossCorePipeline()(mod)
        mod = tilelang.transform.CombineCV()(mod)
        # Recompute allocation scopes after C/V partitioning drops buffer uses.
        mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.AscendLowerOpaqueBlock()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    # Collect buffer shape and flatten buffer shape to 2D
    mod = tilelang.transform.Flatten2DBuffer()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.AscendStorageRewrite(is_npu=check_npu_availability())(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.RewriteUnsafeSelect()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)
    mod = tilelang.transform.AscendMemoryPlanning()(mod)
    if planner == "auto":
        mod = tilelang.transform.AscendInsertScheduledSync(target, platform)(mod)
    elif sync_strategy == "a3" and planner == "constraint":
        mod = tilelang.transform.AscendInsertSyncA3(target, platform)(mod)
    else:
        mod = tilelang.transform.AscendSyncInsert(target, platform)(mod)
    if planner != "auto":
        mod = tilelang.transform.AscendSyncInsertVS(target, platform)(mod)
    if sync_strategy == "a3" and planner == "legacy":
        mod = tilelang.transform.AscendInsertSyncHandoff(target, platform)(mod)
    if managed_vector_mask:
        # Materialize physical terminals only after every semantic analysis and
        # rewrite pass has completed. No TIR-transforming pass may run after
        # mask legalization.
        mod = tir.transform.Simplify()(mod)
    if planner == "auto":
        mod = tilelang.transform.AscendVerifyAutoSchedule(target, platform)(mod)
    elif planner == "constraint":
        mod = tilelang.transform.AscendVerifyConstraintSync(target, platform)(mod)
    elif sync_strategy == "a3":
        mod = tilelang.transform.AscendVerifySyncHandoff(target, platform)(mod)
    # CombineCV or explicit T.Scope annotations must assign every
    # resource-specific Ascend operation to C or V. Verify after automatic
    # synchronization has materialized its final hardware calls.
    mod = tilelang.transform.AscendResourceScopeVerify()(mod)
    if managed_vector_mask:
        mod = tilelang.transform.AscendVectorInstructionSelection(target, platform)(mod)
        mod = tilelang.transform.AscendVectorMaskLegalize(target, platform)(mod)
    # print(mod)
    return mod
