# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The compiler for TL programs."""

import os
import os.path as osp
from typing import Union, Optional, Callable, List
import tilelang.transform
from tilelang import tvm as tvm
from tvm import tir
from tvm.ir import CallingConv
from tvm.target import Target
from tilelang.contrib import hipcc, nvcc
from tilelang.engine.param import KernelParam, CompiledArtifact
from tilelang.utils.target import determine_target  # noqa: F401
from tilelang.engine.phase import (
    LowerAndLegalize,
    OptimizeForTarget,
)
from tilelang.tladapter import transforms, conversion
from tilelang.tladapter.transforms import hivm as H, mlir as M, bishengir as B
from tilelang.utils.ascend_npu import detect_npu_target


def _build_npuir_pass_pipeline() -> list:
    """Build the full NPUIR pass pipeline (decomposed from bishengir-compile).

    The pipeline mirrors the C++ implementation in PassPipeline.cpp,
    ConvertToHIVMPipeline.cpp and HIVMPipelines.cpp, fully decomposed into
    individual passes so that each step is independently observable.
    """
    TILELANG_ASCEND_MODE = os.environ.get("TILELANG_ASCEND_MODE")
    expert = (
        TILELANG_ASCEND_MODE is None
        or TILELANG_ASCEND_MODE.lower().strip() in ["expert", "exp", "e"]
    )
    disable_tensor = expert

    passes: list = []

    # ── buildBiShengHIRPipeline: front-end passes ──────────────────────
    passes += [
        M.canonicalize(top_down=True),
        B.adapt_triton_kernel,
        B.canonicalize_module,
        B.append_device_spec(target=detect_npu_target()),
    ]

    # ── convert-to-hivm-pipeline (decomposed) ──────────────────────────
    passes += [
        conversion.hfusion_to_hivm(mm_map_mode="macro_instr"),
        conversion.triton_global_kernel_args_to_hivm_op,
        conversion.tensor_to_hivm,
        conversion.to_hivm_op,
    ]

    # ── optimize-hivm-pipeline (decomposed) ────────────────────────────
    passes.append(H.init_entry_kernel)

    if not disable_tensor:
        passes += _hivm_pre_bufferization_passes(expert)
        passes += _hivm_bufferization_passes()

    passes += _hivm_post_bufferization_passes(expert)
    passes.append(M.inline_scope(force_inline=True))
    return passes


def _canonicalization_hivm():
    """canonicalizationHIVMPipeline from HIVMPipelines.cpp."""
    return [
        B.arith_to_affine,
        M.scf_canonicalize_iter_arg,
        B.extended_canonicalizer,
        M.scf_for_loop_canonicalization,
        M.cse,
        B.extended_canonicalizer,      # nested func.func in C++, but module-level is safe
        H.opt_single_point,
        B.extended_canonicalizer,
        M.memref_dse,
    ]


def _hivm_pre_bufferization_passes(expert: bool) -> list:
    """hivmPreBufferizationOptimizationPipeline from HIVMPipelines.cpp."""
    passes: list = []
    passes += [
        H.normalize_matmul,                                     # propagate-reshape first
        M.propagate_reshape(for_hivm=True),
        M.scf_remove_redundant_loop_init,
        H.normalize_matmul,
        H.inline_fixpipe,
        H.tile_batchmm_into_loop,
        H.insert_load_store_for_mix_cv,
    ]
    passes += [
        H.normalize_matmul,
        H.insert_nz2nd_for_debug,
        H.inline_fixpipe,
    ]
    passes += [
        H.insert_load_store_for_mix_cv,
        H.insert_workspace_for_mix_cv,
        H.bind_workspace_arg,
    ]
    passes.append(H.infer_func_core_type)
    # auto_blockify_parallel_loop only runs when enableAutoBlockifyLoop=true
    # (default false), so it is NOT added here.
    passes.append(H.mark_multi_buffer(enable_auto=True))
    passes.append(B.extended_canonicalizer)
    passes.append(H.inline_otf_broadcast)
    passes.append(H.cv_pipelining())
    # PlanMemory with GLOBAL_WORKSPACE_PLAN to assign offsets to workspace allocs
    passes.append(H.plan_memory(mem_plan_mode="global-work-space-plan"))
    passes += _hivm_cross_core_sync_passes()
    passes.append(H.insert_infer_workspace_size_func)
    passes.append(M.lower_memref_ext)
    passes.append(H.insert_infer_task_type_func)
    passes.append(H.split_mix_kernel)
    passes.append(M.inline_scope())
    if not expert:
        passes.append(H.tile_and_bind_sub_block)
    passes.append(M.fold_tensor_empty)
    passes += _canonicalization_hivm()
    passes += [
        M.loop_invariant_code_motion,
        M.loop_invariant_subset_hoisting,
    ]
    passes += [
        H.clone_tensor_empty,
        H.hivm_inline_otf_load_store,
    ]
    return passes


def _hivm_bufferization_passes() -> list:
    """bufferizationPipeline from HIVMPipelines.cpp (triton path)."""
    passes: list = []
    passes += [
        M.optimize_dps_op_with_yielded_insert_slice,
        H.clone_tensor_empty,
    ]
    passes.append(M.one_shot_bufferize(
        bufferize_function_boundaries=True,
        function_boundary_type_conversion="identity-layout-map",
        allow_return_allocs_from_loops=True,
        allow_unknown_ops=True,
    ))
    passes += _canonicalization_hivm()
    passes.append(conversion.to_hivm_op)
    passes.append(M.drop_equivalent_buffer_results)
    passes += _canonicalization_hivm()
    passes.append(M.drop_equivalent_buffer_results)
    return passes


def _hivm_cross_core_sync_passes() -> list:
    """hivmCrossCoreSyncPipeline from HIVMPipelines.cpp."""
    return [
        H.mark_real_core_type(),
        H.inject_block_sync(),
        H.mark_real_core_type(remove_core_type_attrs=True),
    ]


def _hivm_post_bufferization_passes(expert: bool) -> list:
    """hivmPostBufferizationOptimizationPipeline from HIVMPipelines.cpp."""
    passes: list = []
    passes += [
        H.lift_zero_rank,
        M.map_for_to_forall,
        H.hivm_map_forall_to_blocks,
        H.hivm_decompose_op,
        H.sync_block_hoisting,
        H.bind_sync_block_lock_arg,
        H.insert_infer_sync_block_lock_num_and_init_func,
        H.sync_block_lock_lowering,
        H.non_contiguous_reshape_to_copy,
        H.infer_hivm_mem_scope,
        H.hivm_decompose_op,
    ]
    passes.append(H.hivm_aggregated_decompose_op(
        decompose_phase="before-hivm-align"))
    passes.append(H.hivm_recognize_deinterleave_op)
    passes.append(H.hivm_aggregated_decompose_op(
        decompose_phase="after-hivm-recognize-deinterleave"))
    passes.append(H.hivm_aggregated_decompose_op(
        decompose_phase="after-hivm-recognize-broadcast"))

    # alignStoragePipeline
    passes += [
        H.align_alloc_size,
        H.mark_stride_align,
        M.fold_alloc_reshape,
        H.enable_stride_align,
    ]

    passes.append(H.hivm_aggregated_decompose_op(
        decompose_phase="after-hivm-align"))
    passes.append(H.infer_hivm_data_layout)
    passes.append(H.hivm_aggregated_decompose_op(
        decompose_phase="after-infer-hivm-data-layout"))

    passes += [
        B.extended_canonicalizer,
        H.auto_infer_buffer_size,
        B.arith_to_affine,
        H.constantize_buffer_size,
        H.set_buffer_size,
        H.flatten_ops,
    ]
    passes.append(H.hivm_aggregated_decompose_op(
        decompose_phase="after-hivm-flatten-ops"))
    passes += [
        H.reduce_rank_subview,
        H.lift_lowest_stride,
        H.alloc_extra_buffer,
        H.infer_hivm_mem_scope,
    ]
    passes += _canonicalization_hivm()
    passes.append(H.inline_load_copy)

    passes.append(H.mark_multi_buffer(
        enable_auto=True,
        limit_auto_multi_buffer_only_for_local_buffer=True,
    ))
    passes.append(H.plan_memory())

    passes += [
        H.hivm_lower_to_loops,
        H.hivm_decompose_op,
    ]
    passes += [
        H.inject_sync(),
        H.add_ffts_to_sync_block_set_op,
        H.enable_multi_buffer,
        H.lift_lowest_stride,
    ]
    return passes


def is_cpu_device_backend(target: Target):
    return target.kind.name == "c"


def has_device_kernel_launch(attrs) -> bool:
    """Check if the attributes indicate a device kernel launch."""
    return bool(attrs and "calling_conv" in attrs and
                attrs["calling_conv"] == CallingConv.DEVICE_KERNEL_LAUNCH)


def is_device_call_c_device(func: tir.PrimFunc):
    attrs = func.attrs

    # Check if it's a C target
    if "target" in attrs and attrs["target"].kind.name == "c":
        return True

    return has_device_kernel_launch(attrs)


def is_device_call(func: tir.PrimFunc):
    return has_device_kernel_launch(func.attrs)


def get_device_call(is_device_c: bool = False) -> Callable[[tir.PrimFunc], bool]:
    return is_device_call_c_device if is_device_c else is_device_call


def get_host_call(is_device_c: bool = False) -> Callable[[tir.PrimFunc], bool]:
    return lambda func: not get_device_call(is_device_c)(func)


@tvm.register_func("tilelang_callback_cuda_compile", override=True)
def tilelang_callback_cuda_compile(code, target):
    project_root = osp.join(osp.dirname(__file__), "../..")
    if "TL_TEMPLATE_PATH" in os.environ:
        tl_template_path = os.environ["TL_TEMPLATE_PATH"]
    else:
        tl_template_path = osp.abspath(osp.join(project_root, "src"))
    # TODO(lei): this indeed should be renamed into
    # TL_CUTLASS_INCLUDE_PATH in the future
    if "TL_CUTLASS_PATH" in os.environ:
        cutlass_path = os.environ["TL_CUTLASS_PATH"]
    else:
        cutlass_path = osp.abspath(osp.join(project_root, "3rdparty/cutlass/include"))
    compute_version = "".join(nvcc.get_target_compute_version(target).split("."))

    # special handle for Hopper
    if compute_version == "90":
        arch = ["-arch=sm_90a"]
        format = "cubin"
    else:
        arch = [f"-arch=sm_{compute_version}"]
        format = "cubin"

    # printing out number of registers
    debug_option = "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage"
    ptx = nvcc.compile_cuda(
        code,
        format,
        arch,
        options=[
            "-std=c++17",
            debug_option,
            "--use_fast_math",
            "-I" + tl_template_path,
            "-I" + cutlass_path,
        ],
        verbose=False,
    )

    return ptx


@tvm.register_func("tilelang_callback_hip_compile", override=True)
def tilelang_callback_hip_compile(code, target):
    project_root = osp.join(osp.dirname(__file__), "../..")
    tl_template_path = osp.abspath(osp.join(project_root, "src"))

    # TODO(lei): actually this indeed should be renamed into
    # TL_COMPOSABLE_KERNEL_INCLUDE_PATH in the future
    if "TL_COMPOSABLE_KERNEL_PATH" in os.environ:
        ck_path = os.environ["TL_COMPOSABLE_KERNEL_PATH"]
    else:
        ck_path = osp.abspath(osp.join(project_root, "3rdparty/composable_kernel/include"))

    hsaco = hipcc.compile_hip(
        code,
        target_format="hsaco",
        options=[
            "-std=c++17",
            "-I" + tl_template_path,
            "-I" + ck_path,
        ],
        verbose=False,
    )

    return hsaco


def extrac_params(func: tir.PrimFunc) -> List[KernelParam]:
    tensor_types = []
    for var in func.params:
        if var in func.buffer_map:
            tensor_types.append(KernelParam.from_buffer(func.buffer_map[var]))
        else:
            tensor_types.append(KernelParam.from_var(var))
    return tensor_types


def canon_target_host(target: Union[str, Target], target_host: Optional[Union[str, Target]]):

    if not target_host:
        target_host = "llvm" if tvm.runtime.enabled("llvm") else "stackvm"

    return target_host


def host_codegen(host_mod: tvm.IRModule, target_host: Target) -> tvm.IRModule:
    host_mod = tir.transform.BindTarget(target_host)(host_mod)
    host_mod = tir.transform.FP8StorageLegalize()(host_mod)
    host_mod = tir.transform.BF16StorageLegalize()(host_mod)
    host_mod = tir.transform.LowerTVMBuiltin()(host_mod)
    host_mod = tir.transform.LowerCustomDatatypes()(host_mod)
    host_mod = tir.transform.LowerIntrin()(host_mod)
    host_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(host_mod)
    host_mod = tir.transform.CombineContextCall()(host_mod)
    if target_host.kind.name == "llvm":
        host_mod = tvm._ffi.get_global_func("target.build.llvm")(host_mod, target_host)
    elif target_host.kind.name == "c":
        host_mod = tvm._ffi.get_global_func("target.build.c")(host_mod, target_host)
    else:
        raise ValueError(f"Target host {target_host.kind.name} is not supported")
    return host_mod


def device_codegen(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    if target.kind.name == "npuir":
        # device_mod = tvm._ffi.get_global_func("target.build.tilelang_npuir")(device_mod, target)
        TILELANG_ASCEND_MODE = os.environ.get('TILELANG_ASCEND_MODE')
        if TILELANG_ASCEND_MODE is None:
            device_mod = tvm._ffi.get_global_func("target.build.tilelang_npuir_apis")(device_mod, target)
        elif TILELANG_ASCEND_MODE.lower().strip() in ['expert', 'exp', 'e']:
            device_mod = tvm._ffi.get_global_func("target.build.tilelang_npuir_apis")(device_mod, target)
        else:
            device_mod = tvm._ffi.get_global_func("target.build.tilelang_npuir_dev")(device_mod, target)
        return device_mod
    device_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(device_mod)
    device_mod = tir.transform.LowerIntrin()(device_mod)
    device_mod = tir.transform.Simplify()(device_mod)
    if target.kind.name == "cuda":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_cuda")(device_mod, target)
    elif target.kind.name == "hip":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_hip")(device_mod, target)
    else:
        raise ValueError(f"Target {target.kind.name} is not supported")

    return device_mod


def device_codegen_without_compile(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    device_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(device_mod)
    device_mod = tir.transform.LowerIntrin()(device_mod)
    device_mod = tir.transform.Simplify()(device_mod)
    if target.kind.name == "cuda":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_cuda_without_compile")(
            device_mod, target)
    elif target.kind.name == "hip":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_hip_without_compile")(
            device_mod, target)
    elif target.kind.name == "c":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_cpp")(device_mod, target)
    elif target.kind.name == "llvm":
        device_mod = tvm._ffi.get_global_func("target.build.llvm")(device_mod, target)
    elif target.kind.name == "webgpu":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_webgpu")(device_mod, target)
    else:
        raise ValueError(f"Target {target.kind.name} is not supported")

    return device_mod


def lower(
    func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
    target: Union[str, Target] = "auto",
    target_host: Optional[Union[str, Target]] = None,
    runtime_only=False,
    enable_host_codegen=False,
    enable_device_compile=False,
) -> CompiledArtifact:
    '''
        enable_host_codegen: whether to enable host codegen, default is False, as we have our
        own host codegen implementation in jit.
        enable_device_compile: whether to enable device codegen, default is False, as we have our
        own device codegen implementation in jit.
    '''

    mod = func_or_mod
    params = None
    if isinstance(func_or_mod, tir.PrimFunc):
        func = func_or_mod
        params = extrac_params(func) if not runtime_only else None
        mod = tvm.IRModule({func.attrs["global_symbol"]: func})

    if isinstance(target, str):
        target = determine_target(target)

    target_host = canon_target_host(target, target_host)

    target_host = tvm.target.Target.canon_target(target_host)
    target = tvm.target.Target(target, target_host)

    _is_host_call = get_host_call(is_device_c=is_cpu_device_backend(target))
    _is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))
    # Phase 1: Lower and legalize the IR
    mod = LowerAndLegalize(mod, target)

    # Phase 2: Optimize the IR for the target
    mod = OptimizeForTarget(mod, target)

    TILELANG_DUMP_IR = os.environ.get('TILELANG_DUMP_IR', '').lower()
    dump_ir = TILELANG_DUMP_IR in ('true', '1', 'yes', 'on')
    if dump_ir:
        print("====== TVM IR ======")
        print(mod)
        print()
    if target.kind.name == "npuir":
        codegen_mod = device_codegen(mod, target)
        mlir_str = codegen_mod.get_source()
        if dump_ir:
            print("====== npuir ======")
            print(mlir_str)
        tladapter_passes = _build_npuir_pass_pipeline()
        for i, p in enumerate(tladapter_passes):
            mlir_str = p(mlir_str)
            if dump_ir:
                name = getattr(p, "pass_name", None) or f"pass-{i}"
                print(f"====== after {name} ======")
                print(mlir_str)
        return mlir_str

    host_mod = tir.transform.Filter(_is_host_call)(mod)
    device_mod = tir.transform.Filter(_is_device_call)(mod)

    codegen_mod = device_codegen(
        device_mod, target) if enable_device_compile else device_codegen_without_compile(
            device_mod, target)

    if enable_host_codegen:
        host_mod = host_codegen(host_mod, target_host)
        host_mod.import_module(codegen_mod)
        return CompiledArtifact(
            host_mod, device_mod, params, codegen_mod.get_source(), rt_mod=host_mod)

    return CompiledArtifact(host_mod, device_mod, params, codegen_mod.get_source())
