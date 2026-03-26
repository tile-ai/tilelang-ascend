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

def TileLangIRVectorize : Pass<"tilelangir-vectorize", "::mlir::ModuleOp"> {
  let summary = "TileLangIR vectorize pass (custom)";
  let description = [{
    Applies vectorization transformations. Implementation to be added.
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

#endif // TILELANGIR_TRANSFORMS_PASSES_TD
