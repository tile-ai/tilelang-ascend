// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/Vectorize.cpp
 * \brief TileLangIR vectorize pass.
 *
 */

#include "tilelangir/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRVECTORIZE
#include "tilelangir/Transforms/Passes.h.inc"

namespace {
#define DEBUG_TYPE "tilelangir-vectorize"
struct TileLangIRVectorize : impl::TileLangIRVectorizeBase<TileLangIRVectorize> {
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: placeholder, no transformation applied yet\n");
  }
};
#undef DEBUG_TYPE
} // namespace

} // namespace tilelangir
} // namespace mlir
