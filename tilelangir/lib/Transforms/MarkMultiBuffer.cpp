// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "tilelangir/Transforms/Passes.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRMARKMULTIBUFFER
#include "tilelangir/Transforms/Passes.h.inc"

namespace {
#define DEBUG_TYPE "tilelangir-mark-multibuffer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

static Value getRootDef(Value val) {
  while (auto *defOp = val.getDefiningOp()) {
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(defOp)) {
      val = viewLike.getViewSource();
      continue;
    }
    break;
  }
  return val;
}

static bool isAlreadyMarked(Operation *op) {
  for (auto *user : op->getResult(0).getUsers()) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(user)) {
      if (markOp->hasAttr("hivm.multi_buffer"))
        return true;
    }
  }
  return false;
}

struct TileLangIRMarkMultiBuffer
    : impl::TileLangIRMarkMultiBufferBase<TileLangIRMarkMultiBuffer> {

  void runOnOperation() override {
    auto funcOp = getOperation();

    funcOp.walk([&](scf::ForOp forOp) {
      auto numStagesAttr =
          forOp->getAttrOfType<IntegerAttr>("tilelangir.num_stages");
      if (!numStagesAttr)
        return;

      int32_t numStages = numStagesAttr.getInt();
      LLVM_DEBUG(DBGS() << "found scf.for with num_stages=" << numStages
                        << "\n");

      llvm::SmallPtrSet<Operation *, 8> workspaceOps;

      forOp.getBody()->walk([&](Operation *op) {
        for (Value operand : op->getOperands()) {
          Value root = getRootDef(operand);
          auto *defOp = root.getDefiningOp();
          if (!defOp)
            continue;
          if (isa<bishengir::memref_ext::AllocWorkspaceOp>(defOp))
            workspaceOps.insert(defOp);
        }
      });

      OpBuilder builder(funcOp.getContext());
      for (auto *wsOp : workspaceOps) {
        if (isAlreadyMarked(wsOp)) {
          LLVM_DEBUG(DBGS() << "skip already-marked workspace: " << *wsOp
                            << "\n");
          continue;
        }

        LLVM_DEBUG(DBGS() << "marking workspace: " << *wsOp << "\n");
        builder.setInsertionPointAfter(wsOp);
        auto markOp = builder.create<annotation::MarkOp>(
            wsOp->getLoc(), wsOp->getResult(0));
        markOp->setAttr("hivm.multi_buffer",
                         builder.getI32IntegerAttr(numStages));
      }
    });
  }
};

#undef DBGS
#undef DEBUG_TYPE
} // namespace

} // namespace tilelangir
} // namespace mlir
