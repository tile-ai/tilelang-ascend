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

#endif // TILELANGIR_TRANSFORMS_PASSES_TD
