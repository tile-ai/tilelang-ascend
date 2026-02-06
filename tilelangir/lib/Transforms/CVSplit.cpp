// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/CVSplit.cpp
 * \brief TileLangIR CV split pass.
 *
 */

#include "tilelangir/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRCVSPLIT
#include "tilelangir/Transforms/Passes.h.inc"

namespace {
#define DEBUG_TYPE "tilelangir-cv-split"
struct TileLangIRCVSplit : impl::TileLangIRCVSplitBase<TileLangIRCVSplit> {
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: placeholder, no transformation applied yet\n");
  }
};
#undef DEBUG_TYPE
} // namespace

} // namespace tilelangir
} // namespace mlir
