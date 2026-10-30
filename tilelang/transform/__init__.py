# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Wrapping transformations."""
# pylint: disable=invalid-name, unsupported-binary-operation

from . import _ffi_api
from .simplify import Simplify, simplify_prim_func  # noqa: F401
from .pass_config import PassConfigKey  # noqa: F401
from tilelang import tvm as tvm  # noqa: F401
from tvm.ir.transform import PassContext  # noqa: F401
from tvm.target import Target


def HostProcesser():
    """HostProcesser

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.HostLegalize()  # type: ignore


def get_pass_context():
    """Get the current pass context"""
    return PassContext.current()


def ClusterPlanning():
    """ClusterPlanning

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ClusterPlanning()  # type: ignore


def PipelinePlanning():
    """Plan software-pipeline annotations with the selected Ascend strategy.

    Construct inside the active PassContext. Legacy remains the default; only
    the constraint strategy registers the optional numeric solver callback.
    """
    strategy = str(get_pass_context().config.get(PassConfigKey.TL_ASCEND_PIPELINE_PLANNER, "legacy"))
    if strategy == "auto":
        raise ValueError("The auto scheduler uses AscendAutoSchedule and AscendMaterializeSchedule")
    if strategy not in {"legacy", "constraint"}:
        raise ValueError(f"Unknown Ascend pipeline planner strategy: {strategy!r}")
    if strategy == "constraint":
        from .ascend_schedule_solver_ffi import register_solver

        _validate_constraint_cost_model()
        register_solver()
    return _ffi_api.PipelinePlanning()  # type: ignore


def AscendPrepareAutoSchedule(target: Target):
    """Preserve finite control and requested version facts before early simplification."""
    return _ffi_api.AscendPrepareAutoSchedule(target)  # type: ignore


def AscendAutoSchedule(target: Target, platform: str):
    """Plan the audited finite A3 task DAG using abstract issue steps.

    Optional solver registration is lazy and exclusive to the auto route.
    This path does not consume the constraint planner's nanosecond profile.
    """
    from .ascend_schedule_solver_ffi import register_solver

    register_solver()
    return _ffi_api.AscendAutoSchedule(target, platform)  # type: ignore


def AscendMaterializeSchedule(target: Target, platform: str):
    """Emit finite task instances and owned UB/L1 versions without the legacy injector."""
    return _ffi_api.AscendMaterializeSchedule(target, platform)  # type: ignore


def AscendInsertScheduledSync(target: Target, platform: str):
    """Generate sole-owner local fences and fixed-slot A3 GM handshakes after MP."""
    return _ffi_api.AscendInsertScheduledSync(target, platform)  # type: ignore


def AscendVerifyAutoSchedule(target: Target, platform: str):
    """Independently verify source producers, physical versions and the final protocol."""
    return _ffi_api.AscendVerifyAutoSchedule(target, platform)  # type: ignore


def load_ascend_pipeline_cost_model(path, *, expected_identity=None):
    """Load the complete artifact to pass as TL_ASCEND_PIPELINE_COST_MODEL."""
    from .ascend_pipeline_cost import load_cost_model

    return load_cost_model(path, expected_identity=expected_identity)


def _validate_constraint_cost_model():
    profile = get_pass_context().config.get(PassConfigKey.TL_ASCEND_PIPELINE_COST_MODEL)
    if profile is not None:
        import json
        from .ascend_schedule_solver_ffi import _plain
        from .ascend_pipeline_cost import validate_cost_model

        # Convert old-TVM String/Map/IntImm containers without treating a
        # symbolic expression or a filename as calibrated numeric data.
        validate_cost_model(json.loads(json.dumps(_plain(profile))))


def AscendPrepareConstraintPipeline(target: tvm.target.Target):
    """Preserve requested short-loop pipeline facts before simplification."""
    _validate_constraint_cost_model()
    return _ffi_api.AscendPrepareConstraintPipeline(target)  # type: ignore


def AscendCanonicalizeStorageAliases():
    """Normalize known storage/view aliases before constraint analysis and memory planning.

    The constraint pipeline calls this idempotent pass before hint and operation
    lowering, then again before late hint consumption. The pass operates only
    in a constraint context; legacy returns the input function unchanged.
    Unknown pointer origins and unprovable view offsets are diagnosed.
    """
    return _ffi_api.AscendCanonicalizeStorageAliases()  # type: ignore


def AscendVerifyConstraintSync(target: tvm.target.Target, platform: str):
    """Verify final constraint-mode task identities, dependencies and event lifetimes."""
    return _ffi_api.AscendVerifyConstraintSync(target, platform)  # type: ignore


def AscendInsertSyncA3(target: Target, platform: str):
    """Generate completion events from final physical Vector dependencies on A3.

    Only the audited constraint/a3 route owns this domain. Existing final
    constraint verification independently checks the generated events.
    """
    return _ffi_api.AscendInsertSyncA3(target, platform)  # type: ignore


def AllocateAscendSyncEvents(intervals, platform: str = "A3") -> tvm.ir.Map:
    """Allocate A3 event slots for explicit half-open live intervals without changing IR.

    Requests describe core, directed source/destination PIPE, begin/end and
    optional fixed IDs or reserved IDs. Invalid domains and exhausted pools
    are diagnosed by the same allocator used by the native generator.
    """
    return _ffi_api.AllocateAscendSyncEvents(intervals, platform)  # type: ignore


def AscendPrepareSyncHandoff(target: Target, platform: str):
    """Audit fixed C/V workspace ownership before partitioning and synchronization."""
    return _ffi_api.AscendPrepareSyncHandoff(target, platform)  # type: ignore


def AscendInsertSyncHandoff(target: Target, platform: str):
    """Generate A3 mode2 workspace ready/release after final memory planning."""
    return _ffi_api.AscendInsertSyncHandoff(target, platform)  # type: ignore


def AscendVerifySyncHandoff(target: Target, platform: str):
    """Independently verify final joint C/V workspace completion and event drain."""
    return _ffi_api.AscendVerifySyncHandoff(target, platform)  # type: ignore


def AscendValidateScheduleHints():
    """Check Task/Stage structure before frontend legalization can erase it."""
    return _ffi_api.AscendValidateScheduleHints()  # type: ignore


def AscendLowerScheduleHints(target: tvm.target.Target, platform: str):
    """Audit supported A3 Task effects, then consume validated Task/Stage hints."""
    return _ffi_api.AscendLowerScheduleHints(target, platform)  # type: ignore


def AnalyzeAscendSchedule(func: tvm.tir.PrimFunc, platform: str = "A3") -> tvm.ir.Map:
    """Inspect audited lowered calls without changing IR or invoking a scheduler.

    The returned legacy TVM containers retain symbolic PrimExpr footprints and
    guards. They are not an already numeric solver problem or pipeline ABI.
    Unsupported effects are diagnosed only by this explicit analysis entrypoint.
    """
    return _ffi_api.AnalyzeAscendSchedule(func, platform)  # type: ignore


def AnalyzeAscendStorageOwners(func: tvm.tir.PrimFunc, platform: str = "A3") -> tvm.ir.Map:
    """Inspect canonical storage identity, views and known pointer aliases without rewriting IR.

    Storage identity follows data Vars and proven alias bindings. The report
    retains typed legacy TVM objects and byte offsets, rather than treating
    buffer names or different dtype views as independent allocations.
    """
    return _ffi_api.AnalyzeAscendStorageOwners(func, platform)  # type: ignore


def AnalyzeAscendLifetimes(func: tvm.tir.PrimFunc, platform: str = "A3") -> tvm.ir.Map:
    """Prove complete source read definitions on bounded control paths without changing IR.

    The report records allocation activations and byte-segment producers;
    skipped or zero-trip writes provide no initialization. This explicit
    analysis does not schedule operations or generate synchronization.
    """
    return _ffi_api.AnalyzeAscendLifetimes(func, platform)  # type: ignore


def LayoutInference():
    """LayoutInference

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LayoutInference()  # type: ignore


def LowerTileOp():
    """LowerTileOp

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerTileOp()  # type: ignore


def InjectSoftwarePipeline():
    """InjectSoftwarePipeline

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectSoftwarePipeline()  # type: ignore


def FrontendLegalize():
    """FrontendLegalize

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.FrontendLegalize()  # type: ignore


def LowerHopperIntrin():
    """LowerHopperIntrin

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerHopperIntrin() if hasattr(_ffi_api, "LowerHopperIntrin") else lambda f: f  # type: ignore


def WarpSpecializedPipeline():
    """WarpSpecializedPipeline

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.WarpSpecializedPipeline()  # type: ignore


def RewriteWgmmaSync():
    """RewriteWgmmaSync

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.RewriteWgmmaSync()  # type: ignore


def ThreadSync(storage_scope: str):
    """Insert sync between parallel read/write of shared buffers.

    Parameters
    ----------
    storage_scope: str
        The target storage scope.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ThreadSync(storage_scope)  # type: ignore


def ThreadPartialSync(storage_scope: str):
    """Insert partial sync.

    Parameters
    ----------
    storage_scope: str
        The target storage scope.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ThreadPartialSync(storage_scope)  # type: ignore


def IfStmtBinding():
    """IfStmtBinding

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.IfStmtBinding()  # type: ignore


def MergeIfStmt():
    """MergeIfStmt

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MergeIfStmt()  # type: ignore


def MultiVersionBuffer():
    """WarpSpecializedPipeline

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MultiVersionBuffer()  # type: ignore


def WarpSpecialized():
    """WarpSpecializedPipeline

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.WarpSpecialized()  # type: ignore


def InjectTmaBarrier():
    """InjectTmaBarrier

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectTmaBarrier()  # type: ignore


def InjectFenceProxy():
    """InjectFenceProxy

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectFenceProxy()  # type: ignore


def LegalizeVectorizedLoop():
    """LegalizeLoopVectorize

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LegalizeVectorizedLoop()  # type: ignore


def LegalizeSafeMemoryAccess():
    """LegalizeLoopVectorize

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LegalizeSafeMemoryAccess()  # type: ignore


def MakePackedAPI():
    """MakePackedAPI

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MakePackedAPI()  # type: ignore


def AnnotateDeviceRegions():
    """AnnotateDeviceRegions

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateDeviceRegions()  # type: ignore


def VectorizeLoop(enable_vectorize: bool = True):
    """VectorizeLoop

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.VectorizeLoop(enable_vectorize)  # type: ignore


def InjectPTXAsyncCopy():
    """Rewrite global to shared memory copy on CUDA with asynchronous copy.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectPTXAsyncCopy()  # type: ignore


def LowerDeviceStorageAccessInfo():
    """Lower attached storage access information on device.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass

    Note
    ----
    Run this pass after all storage access analysis finish.
    """
    return _ffi_api.LowerDeviceStorageAccessInfo()  # type: ignore


def LoopVectorizeDynamic():
    """Try to vectorize loop with dynamic shape.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.LoopVectorizeDynamic()  # type: ignore


def ConfigIndexBitwidth():
    """Config index bitwidth.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.ConfigIndexBitwidth()  # type: ignore


def FlattenBuffer():
    """FlattenBuffer

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.FlattenBuffer()  # type: ignore


def EliminateStorageSyncForMBarrier():
    """EliminateStorageSyncForMBarrier"""
    return _ffi_api.EliminateStorageSyncForMBarrier()  # type: ignore


def MergeSharedMemoryAllocations():
    """MergeSharedMemoryAllocations

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MergeSharedMemoryAllocations()  # type: ignore


def AscendMemoryPlanning():
    """Memory planning for Ascend NPU

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.AscendMemoryPlanning()  # type: ignore


def AscendSyncInsert(target: Target, platform: str):
    """Auto insert sync for Ascend.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.AscendSyncInsert(target, platform)  # type: ignore


def AscendSyncInsertVS(target: Target, platform: str):
    """As a supplement to AscendSyncInsert, auto insert sync for Ascend (V→V, S↔Others).

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.AscendSyncInsertVS(target, platform)  # type: ignore


def AscendVectorInstructionSelection(target: Target, platform: str):
    """Select physical Ascend Vector terminals and their mask contracts."""
    return _ffi_api.AscendVectorInstructionSelection(target, platform)  # type: ignore


def AscendVectorMaskLegalize(target: Target, platform: str):
    """Reuse compatible Ascend Vector mask state before code generation."""
    return _ffi_api.AscendVectorMaskLegalize(target, platform)  # type: ignore


def AscendResourceScopeVerify():
    """Verify that resource-specific Ascend operations have a C/V owner."""
    return _ffi_api.AscendResourceScopeVerify()  # type: ignore


def CombineCV():
    """CombineCV

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.CombineCV()  # type: ignore


def CrossCorePipeline():
    """CrossCorePipeline

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.CrossCorePipeline()  # type: ignore


def AscendLowerParallelToVector():
    """Lower parallel loops to vector instructions for Ascend.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.AscendLowerParallelToVector()  # type: ignore


def AscendVidReduction():
    """Reduction Vid for Ascend.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.AscendVidReduction()  # type: ignore


def AscendWorkspaceReduction():
    """Reduction Workspace for Ascend.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.AscendWorkspaceReduction()  # type: ignore


def AscendTailMaskPropagation(rewrite_reduce: bool = True):
    """Propagate UB tail valid-regions and rewrite vector ops to tail-aware
    variants for the Ascend backend.

    Parameters
    ----------
    rewrite_reduce : bool
        Whether allow-listed reduce ops may be rewritten to tail_reduce.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.AscendTailMaskPropagation(rewrite_reduce)  # type: ignore


def AscendInferBufferScope():
    """Infer Buffer Scope for Ascend.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.InferAllocScope()  # type: ignore


def CollectBufferShapes():
    """Infer Buffer Scope for Ascend.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.CollectBufferShapes()  # type: ignore


def BufferShapeCollector():
    """Collect Buffer Shape for Ascend.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.BufferShapeCollector()  # type: ignore


def Flatten2DBuffer():
    """flatten buffer shape to 2D.

    Converts buffer dimensions as follows:
    - 1D [M] -> 2D [1, M]
    - 2D [N, M] -> 2D [N, M] (unchanged)
    - ND [D1, D2, ..., Dn] -> 2D [D1*D2*...*Dn-1, Dn]

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.Flatten2DBuffer()  # type: ignore


def AscendStorageRewrite(is_npu: bool = False):
    """StorageRewrite for Ascend.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.AscendStorageRewrite(is_npu)  # type: ignore


def AscendLowerOpaqueBlock():
    """LowerOpaqueBlock for Ascend.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.AscendLowerOpaqueBlock()  # type: ignore


def InjectTmpBuffer(target: Target, plan_vid_reduction: bool = False):
    """InjectTmpBuffer for Ascend

    Set plan_vid_reduction when AscendVidReduction will run afterwards.
    Buffer scopes must already be inferred in that case.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.InjectTmpBuffer(target, plan_vid_reduction)  # type: ignore
