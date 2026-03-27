// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "tilelangir/Transforms/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRMERGECOPYCHAINS
#include "tilelangir/Transforms/Passes.h.inc"

namespace {
#define DEBUG_TYPE "tilelangir-merge-copy-chains"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using hivm::AddressSpace;

static std::optional<AddressSpace> getAS(Value v) {
  return hivm::getOptionalHIVMAddressSpace(v.getType());
}

static bool staticShapesAndElemMatch(Value src, Value dst) {
  auto t1 = dyn_cast<MemRefType>(src.getType());
  auto t2 = dyn_cast<MemRefType>(dst.getType());
  if (!t1 || !t2)
    return false;
  if (t1.getElementType() != t2.getElementType())
    return false;
  if (!t1.hasStaticShape() || !t2.hasStaticShape())
    return false;
  return t1.getShape() == t2.getShape();
}

static bool isValuePotentiallyModifiedBetween(Operation *from, Operation *to,
                                              Value val) {
  for (OpOperand &use : val.getUses()) {
    Operation *user = use.getOwner();
    if (user == from || user == to)
      continue;

    if (auto copy = dyn_cast<memref::CopyOp>(user))
      if (copy.getSource() == val)
        continue;

    Operation *ancestor = user;
    while (ancestor && ancestor->getBlock() != from->getBlock())
      ancestor = ancestor->getParentOp();
    if (!ancestor)
      continue;

    if (from->isBeforeInBlock(ancestor) && ancestor->isBeforeInBlock(to))
      return true;
  }
  return false;
}

static void mergeInBlock(Block &block, IRRewriter &rewriter) {
  // Phase 1: collect all cc(L0C)→cbuf(L1) copies in this block.
  SmallVector<memref::CopyOp> ccToCbuf;
  for (auto &op : block) {
    auto c = dyn_cast<memref::CopyOp>(&op);
    if (!c)
      continue;
    auto s = getAS(c.getSource()), d = getAS(c.getTarget());
    if (s && d && *s == AddressSpace::L0C && *d == AddressSpace::L1)
      ccToCbuf.push_back(c);
  }

  // Phase 2: for each cc→cbuf copy, fan out to cbuf→gm users.
  for (auto c1 : ccToCbuf) {
    Value cbuf = c1.getTarget();
    Value ccSrc = c1.getSource();

    bool safe = true;
    SmallVector<memref::CopyOp> candidates;

    for (OpOperand &use : cbuf.getUses()) {
      Operation *owner = use.getOwner();
      if (owner == c1.getOperation())
        continue;

      auto userCopy = dyn_cast<memref::CopyOp>(owner);
      if (!userCopy || userCopy.getSource() != cbuf) {
        safe = false;
        break;
      }

      // Cross-block user: legal (cbuf is only read) but we cannot merge it
      // without deeper alias analysis, so just skip it.
      if (owner->getBlock() != c1->getBlock())
        continue;

      auto dstAS = getAS(userCopy.getTarget());
      if (dstAS && *dstAS == AddressSpace::GM &&
          staticShapesAndElemMatch(ccSrc, userCopy.getTarget()))
        candidates.push_back(userCopy);
    }

    if (!safe)
      continue;

    // Phase 3: filter — only keep candidates where cc is unmodified between.
    SmallVector<memref::CopyOp> toMerge;
    for (auto c2 : candidates)
      if (!isValuePotentiallyModifiedBetween(c1, c2, ccSrc))
        toMerge.push_back(c2);

    if (toMerge.empty())
      continue;

    LLVM_DEBUG(DBGS() << "merging " << toMerge.size()
                      << " cbuf→gm copies from: " << *c1 << "\n");

    for (auto c2 : toMerge) {
      rewriter.setInsertionPoint(c2);
      rewriter.create<memref::CopyOp>(c2.getLoc(), ccSrc, c2.getTarget());
      rewriter.eraseOp(c2);
    }

    // Erase c1 only when cbuf has no remaining users.
    bool hasRemainingUses = false;
    for (OpOperand &use : cbuf.getUses()) {
      if (use.getOwner() != c1.getOperation()) {
        hasRemainingUses = true;
        break;
      }
    }
    if (!hasRemainingUses)
      rewriter.eraseOp(c1);
  }
}

struct TileLangIRMergeCopyChains
    : impl::TileLangIRMergeCopyChainsBase<TileLangIRMergeCopyChains> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(&getContext());

    funcOp.walk([&](Block *block) { mergeInBlock(*block, rewriter); });
  }
};

#undef DBGS
#undef DEBUG_TYPE
} // namespace

} // namespace tilelangir
} // namespace mlir
