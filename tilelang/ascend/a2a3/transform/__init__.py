"""Backend-local factories for the complete migrated TA components."""

from tilelang import tvm


def BufferShapeCollector():
    return tvm.ffi.get_global_func("tl.transform.A2A3BufferShapeCollector")()


def AscendLowerParallelToVector():
    return tvm.ffi.get_global_func("tl.transform.A2A3AscendLowerParallelToVector")()


def Flatten2DBuffer():
    return tvm.ffi.get_global_func("tl.transform.A2A3Flatten2DBuffer")()


def CollectBufferShapes():
    return tvm.ffi.get_global_func("tl.transform.A2A3CollectBufferShapes")()


def AscendLowerOpaqueBlock():
    return tvm.ffi.get_global_func("tl.transform.A2A3AscendLowerOpaqueBlock")()


def LegalizeSafeMemoryAccess():
    return tvm.ffi.get_global_func("tl.transform.A2A3LegalizeSafeMemoryAccess")()


def ConfigIndexBitwidth():
    return tvm.ffi.get_global_func("tl.transform.A2A3ConfigIndexBitwidth")()


def LegalizeVectorizedLoop():
    return tvm.ffi.get_global_func("tl.transform.A2A3LegalizeVectorizedLoop")()


def FlattenBuffer():
    return tvm.ffi.get_global_func("tl.transform.A2A3FlattenBuffer")()


def AscendInferBufferScope():
    # The native pass calls the backend's default tagged zN layout factory.
    from .. import layout  # noqa: F401

    return tvm.ffi.get_global_func("tl.transform.A2A3InferAllocScope")()


def LayoutInference():
    return tvm.ffi.get_global_func("tl.transform.A2A3LayoutInference")()


def InjectTmpBuffer(target, plan_vid_reduction: bool = False):
    return tvm.ffi.get_global_func("tl.transform.A2A3InjectTmpBuffer")(target, plan_vid_reduction)


def AscendVidReduction():
    return tvm.ffi.get_global_func("tl.transform.A2A3AscendVidReduction")()


def LowerTileOp():
    return tvm.ffi.get_global_func("tl.transform.A2A3LowerTileOp")()


def VectorizeLoop(enable_vectorize=True):
    return tvm.ffi.get_global_func("tl.transform.A2A3VectorizeLoop")(enable_vectorize)


def AscendWorkspaceReduction():
    return tvm.ffi.get_global_func("tl.transform.A2A3AscendWorkspaceReduction")()


def AscendTailMaskPropagation(rewrite_reduce: bool = True):
    return tvm.ffi.get_global_func("tl.transform.A2A3AscendTailMaskPropagation")(rewrite_reduce)


def AscendStorageRewrite(is_npu=False):
    return tvm.ffi.get_global_func("tl.transform.A2A3AscendStorageRewrite")(is_npu)


def AscendMemoryPlanning():
    return tvm.ffi.get_global_func("tl.transform.A2A3AscendMemoryPlanning")()


def CombineCV():
    return tvm.ffi.get_global_func("tl.transform.A2A3CombineCV")()


def AscendResourceScopeVerify():
    return tvm.ffi.get_global_func("tl.transform.A2A3AscendResourceScopeVerify")()


def CrossCorePipeline():
    return tvm.ffi.get_global_func("tl.transform.A2A3CrossCorePipeline")()


def PipelinePlanning():
    return tvm.ffi.get_global_func("tl.transform.A2A3PipelinePlanning")()


def AscendSyncInsert(target, platform):
    return tvm.ffi.get_global_func("tl.transform.A2A3AscendSyncInsert")(target, platform)


def InjectSoftwarePipeline():
    return tvm.ffi.get_global_func("tl.transform.A2A3InjectSoftwarePipeline")()


def AscendSyncInsertVS(target, platform):
    return tvm.ffi.get_global_func("tl.transform.A2A3AscendSyncInsertVS")(target, platform)


def AnnotateDeviceRegions():
    return tvm.ffi.get_global_func("tl.transform.A2A3AnnotateDeviceRegions")()


def AscendVectorInstructionSelection(target, platform):
    return tvm.ffi.get_global_func("tl.transform.A2A3AscendVectorInstructionSelection")(target, platform)


def AscendVectorMaskLegalize(target, platform):
    return tvm.ffi.get_global_func("tl.transform.A2A3AscendVectorMaskLegalize")(target, platform)


def HostProcesser():
    return tvm.ffi.get_global_func("tl.transform.A2A3HostLegalize")()


def RemoveNoOp():
    """Preserve pinned TA dead-binding/effect cleanup on TIRX."""
    return tvm.ffi.get_global_func("tl.transform.A2A3RemoveNoOp")()


def AscendSimplify():
    """Preserve pinned TA eligible dead-Let cleanup at Simplify stages."""
    return tvm.ffi.get_global_func("tl.transform.A2A3Simplify")()
