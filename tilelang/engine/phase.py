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


def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    # allocate the tmp buffer for vector api
    mod = tilelang.transform.InjectTmpBuffer(target)(mod)
    mod = tilelang.transform.AscendInferBufferScope()(mod)
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
    mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.CrossCorePipeline()(mod)
    # print(mod)
    mod = tilelang.transform.CombineCV()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    # print(mod)
    mod = tilelang.transform.AscendLowerOpaqueBlock()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    # Collect buffer shape and flatten buffer shape to 2D
    mod = tilelang.transform.Flatten2DBuffer()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.AscendStorageRewrite(is_npu=check_npu_availability())(mod)
    # print(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    # print(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.RewriteUnsafeSelect()(mod)
    # print(mod)
    mod = tir.transform.HoistIfThenElse()(mod)
    # print(mod)
    mod = tilelang.transform.AscendMemoryPlanning()(mod)
    # print(mod)
    mod = tilelang.transform.AscendSyncInsert(target, platform)(mod)
    # print(mod)
    return mod
