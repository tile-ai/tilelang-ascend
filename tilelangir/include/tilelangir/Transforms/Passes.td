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

def TileLangIRCVSplit : Pass<"tilelangir-cv-split", "::mlir::ModuleOp"> {
  let summary = "TileLangIR CV split pass (custom)";
  let description = [{
    Splits or transforms operations for CV (e.g. coefficient/variable separation).
    Implementation to be added.
  }];
}

def TileLangIRVectorize : Pass<"tilelangir-vectorize", "::mlir::ModuleOp"> {
  let summary = "TileLangIR vectorize pass (custom)";
  let description = [{
    Applies vectorization transformations. Implementation to be added.
  }];
}

#endif // TILELANGIR_TRANSFORMS_PASSES_TD
