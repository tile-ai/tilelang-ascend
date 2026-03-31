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

def TileLangIRWrapHostFunction : Pass<"tilelangir-wrap-host-function", "::mlir::ModuleOp"> {
  let summary = "Generate host-side functions encoding kernel signature metadata";
  let description = [{
    For each device entry kernel, creates two host-side functions:
      {baseName}_get_kernel_num_args()        - returns number of user args
      {baseName}_get_kernel_arg_type(int idx) - returns type code for arg idx

    Type code: bit 7 set for memref/pointer; bits 0-6 encode the element type
    (0=fp32, 1=fp16, 2=bf16, 3=i8, 4=i16, 5=i32, 6=i64, 7=u32, 8=u64,
     9=fp64, 10=i1).

    User arguments are identified by skipping the first 3 compiler-inserted
    parameters (ffts_base, sync_block_lock, workspace) and the last 6 grid
    parameters.
  }];
}

#endif // TILELANGIR_TRANSFORMS_PASSES_TD
