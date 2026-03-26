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

#endif // TILELANGIR_TRANSFORMS_PASSES_TD
