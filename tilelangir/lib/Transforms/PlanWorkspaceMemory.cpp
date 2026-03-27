// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "tilelangir/Transforms/Passes.h"

#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tilelangir-plan-workspace-memory"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRPLANWORKSPACEMEMORY
#include "tilelangir/Transforms/Passes.h.inc"

namespace {

struct PlanWorkspaceMemoryPass
    : public impl::TileLangIRPlanWorkspaceMemoryBase<
          PlanWorkspaceMemoryPass> {
  using TileLangIRPlanWorkspaceMemoryBase::TileLangIRPlanWorkspaceMemoryBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    SmallVector<bishengir::memref_ext::AllocWorkspaceOp> allocOps;
    funcOp.walk([&](bishengir::memref_ext::AllocWorkspaceOp op) {
      if (op.getOffset().empty())
        allocOps.push_back(op);
    });

    if (allocOps.empty())
      return;

    // Group workspace allocs by their backing workspace argument so that
    // independent workspace pools get independent offset sequences.
    llvm::MapVector<Value,
                    SmallVector<bishengir::memref_ext::AllocWorkspaceOp>>
        argToOps;
    for (auto op : allocOps)
      argToOps[op.getWorkspaceArg()].push_back(op);

    constexpr int64_t bitsPerByte = 8;

    for (auto &[wsArg, ops] : argToOps) {
      int64_t currentByteOffset = 0;

      for (auto op : ops) {
        MemRefType ty = op.getType();
        if (!ty.hasStaticShape()) {
          op.emitError("plan-workspace-memory: dynamic shapes unsupported");
          return signalPassFailure();
        }

        int64_t elemBytes =
            static_cast<int64_t>(ty.getElementTypeBitWidth()) / bitsPerByte;
        int64_t sizeInBytes = elemBytes * ty.getNumElements();

        OpBuilder builder(op);
        Value offsetVal = builder.create<arith::ConstantIndexOp>(
            op.getLoc(), currentByteOffset);

        auto newOp =
            builder.create<bishengir::memref_ext::AllocWorkspaceOp>(
                op.getLoc(), op->getResultTypes(), op.getWorkspaceArg(),
                op.getDynamicSize(), SmallVector<Value>{offsetVal});

        op.replaceAllUsesWith(newOp.getResult());
        op.erase();

        currentByteOffset += sizeInBytes;
      }
    }
  }
};

} // namespace
} // namespace tilelangir
} // namespace mlir
