// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/include/tilelangir/Transforms/Passes.td
 * \brief TileLangIR pass TableGen definitions (CV split, vectorize).
 *
 */

#ifndef TILELANGIR_TRANSFORMS_PASSES_TD
#define TILELANGIR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def TileLangIRCVSplit : Pass<"tilelangir-cv-split", "func::FuncOp"> {
  let summary = "TileLangIR CV split pass";
  let description = [{
    This pass is grouping ops in each `scf::ForOp` according to loads and stores connected to workspaces. Each group of ops is wrapped into a `scope::ScopeOp` identifying the `hivm::TCoreType` of the scope along with its results. The pass is applied on `scf::ForOp`s in a post-order manner and it looks at the operations from the current nesting level only.
  }];
  let dependentDialects = [
    "scope::ScopeDialect",
  ];
}

def TileLangIRSpecializeCube : Pass<"tilelangir-specialize-cube", "func::FuncOp"> {
  let summary = "TileLangIR Specialize Cube ops pass";
  let description = [{

  }];
  let dependentDialects = [
    "hivm::HIVMDialect",
  ];
}

def TileLangIRVectorize : Pass<"tilelangir-vectorize", "::mlir::ModuleOp"> {
  let summary = "TileLangIR vectorize pass (custom)";
  let description = [{
    Applies vectorization transformations. Implementation to be added.
  }];
}

def TileLangIRInsertCVSync : Pass<"tilelangir-insert-cv-sync", "::mlir::ModuleOp"> {
  let summary = "TileLangIR Insert CV synchronization pass (custom)";
  let description = [{
    Insert SyncBlockSetOp to synchronize CV cores.
  }];
}

def TileLangIRInsertWorkspace : Pass<"tilelangir-insert-workspace", "::mlir::func::FuncOp"> {
  let summary = "Insert workspace for cross-core (vector/cube) data transfer";
  let description = [{
    Detects memref.alloc buffers that are simultaneously accessed by HIVM ops
    of different core types (vector vs cube). When such a cross-core buffer is
    found (and it is not already marked as gm or memref_ext.alloc_workspace),
    this pass creates a memref_ext.alloc_workspace buffer and redirects the
    writes from one core type through the workspace, inserting the necessary
    memref.copy operations to complete the data transfer.
  }];
  let dependentDialects = [
    "memref::MemRefDialect",
    "bishengir::memref_ext::MemRefExtDialect"
  ];
}

def TileLangIRMarkMultiBuffer : Pass<"tilelangir-mark-multibuffer", "::mlir::func::FuncOp"> {
  let summary = "Mark alloc_workspace ops used in pipelined loops with multi_buffer annotation";
  let description = [{
    Finds scf.for ops carrying the `tilelangir.num_stages` attribute and
    collects all `memref_ext.alloc_workspace` values that are used (directly
    or through view-like chains) inside the loop body.  For each such
    workspace allocation, an `annotation.mark` op is inserted with
    `hivm.multi_buffer = <num_stages>`.
  }];
  let dependentDialects = [
    "annotation::AnnotationDialect",
    "bishengir::memref_ext::MemRefExtDialect"
  ];
}

def TileLangIRInferMemScope : Pass<"tilelangir-infer-mem-scope", "::mlir::func::FuncOp"> {
  let summary = "Infer and propagate memory address spaces";
  let description = [{
    Infers and propagates HIVM memory address spaces (`#hivm.address_space<...>`)
    for all memref values in a device function, based on hardware semantics:

    Phase 1: `memref_ext.alloc_workspace` results → GM.
    Phase 2: `hivm.hir.mmadL1` operands → mA/mB to L1(cbuf), mC to L0C(cc).
    Phase 3: VECTOR-core HIVM ops → all memref operands traced to root alloc → UB.
    Phase 4: Function arguments (memref without scope) → GM; update function type.
    Phase 5: Remaining `memref.alloc` without scope → default based on enclosing
             `scope.scope`'s `hivm.tcore_type` or the function's `hivm.func_core_type`
             (AIC→L1, AIV→UB).
  }];
}

def TileLangIRMergeCopyChains : Pass<"tilelangir-merge-copy-chains", "::mlir::func::FuncOp"> {
  let summary = "Merge memref.copy chains after memscope inference";
  let description = [{
    Merges chains where address spaces match the hardware pattern, i.e. first copy is cc→cbuf, second copy is cbuf→gm.
  }];
  let dependentDialects = [
    "memref::MemRefDialect",
    "bishengir::memref_ext::MemRefExtDialect",
    "hivm::HIVMDialect",
  ];
}

def TileLangIRPlanWorkspaceMemory : Pass<"tilelangir-plan-workspace-memory", "::mlir::func::FuncOp"> {
  let summary = "Assign byte offsets to memref_ext.alloc_workspace ops";
  let description = [{
    Collects all `memref_ext.alloc_workspace` operations that lack offset
    information, groups them by their workspace argument, and assigns
    sequential byte offsets within each group.  
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "bishengir::memref_ext::MemRefExtDialect",
  ];
}

def TileLangIREnableMultiBuffer : Pass<"tilelangir-enable-multi-buffer", "::mlir::ModuleOp"> {
  let summary = "TileLangIR Enable Multi Buffer pass (custom)";
  let description = [{
    This pass implements multi-buffering for crosscorepipeline.

    It performs the following transformations:
    1. Identifies `memref_ext::AllocWorkspaceOp` operations annotated with the
       `hivm.multi_buffer` attribute and expands them by prepending a multi-buffer
       dimension to the original shape (e.g., `memref<64x64xf16>` → `memref<3x64x64xf16>`).

    2. Replaces `scope::ScopeOp` regions inside `scf::ForOp` loops (annotated with
       `num_stage` attribute) with nested `scf::ForOp` loops. The inner loop iterates
       over the multi-buffer dimension, enabling round-robin access to buffer copies.

    3. Adjusts all `memref::SubViewOp` and `memref::CopyOp` operations that reference
       the expanded workspace to use the new multi-buffer dimension. The pass uses
       `memref::CollapseShapeOp` to collapse size-1 dimensions, ensuring rank reduction
       and maintaining type compatibility with downstream operations.
  }];
}

def TileLangIRSplitMixKernel : Pass<"tilelangir-split-mix-kernel", "::mlir::ModuleOp"> {
  let summary = "TileLangIR Split Mix Kernel pass";
  let description = [{
    This pass splits a mixed kernel (containing both cube and vector operations)
    into two separate kernels: one for cube operations (AIC) and one for vector
    operations (AIV). 
    
    The pass identifies functions marked with mix_mode = "aic" and transforms them:
    - Creates two functions: original_name_mix_aic and original_name_mix_aiv
    - Filters operations based on core type (CUBE vs VECTOR)
    - Filters memory allocations based on address space (cbuf/cc for AIC, ub for AIV)
    - Preserves gm (global memory) operations in both functions
    - Preserves scalar operations in both functions
  }];
}

def TileLangIRWrapHostFunction : Pass<"tilelangir-wrap-host-function", "::mlir::ModuleOp"> {
  let summary = "Generate host-side functions encoding kernel signature metadata";
  let description = [{
    For each device entry kernel, creates two host-side functions:
      {baseName}_get_kernel_num_args()        - returns number of user args
      {baseName}_get_kernel_arg_type(int idx) - returns type code for arg idx

    The compiled .so can then be queried at JIT time to extract metadata
    without fragile regex parsing of MLIR text.  This design supports an
    arbitrary number of user arguments.

    Type code: bit 7 set for memref/pointer; bits 0-6 encode the element type
    (0=fp32, 1=fp16, 2=bf16, 3=i8, 4=i16, 5=i32, 6=i64, 7=u32, 8=u64,
     9=fp64, 10=i1).

    User arguments are identified by skipping the first 3 compiler-inserted
    parameters (ffts_base, sync_block_lock, workspace) and the last 6 grid
    parameters.
  }];
}

#endif // TILELANGIR_TRANSFORMS_PASSES_TD
