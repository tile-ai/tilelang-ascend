// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
//
// Known limitations (TODO):
// 1. Ops with TCoreType::CUBE_OR_VECTOR cause a hard error. This case has not
//    been observed in practice, but may arise if the dialect evolves. A proper
//    resolution would require context-dependent core type inference.
// 2. Ops with TCoreType::CUBE_AND_VECTOR are silently skipped (not included in
//    cross-core analysis). This is correct when such ops handle cross-core
//    synchronization internally, but may need revisiting if CUBE_AND_VECTOR
//    ops with external memory ordering requirements are introduced.

#include "tilelangir/Transforms/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRINSERTWORKSPACE
#include "tilelangir/Transforms/Passes.h.inc"

namespace {
#define DEBUG_TYPE "tilelangir-insert-workspace"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

/// Trace a value back to its root allocation through ViewLikeOpInterface
/// chains.
static Value getRootAlloc(Value val) {
  while (auto *defOp = val.getDefiningOp()) {
    if (auto viewLikeOp = dyn_cast<ViewLikeOpInterface>(defOp)) {
      val = viewLikeOp.getViewSource();
    } else {
      break;
    }
  }
  return val;
}

static bool hasGMAddressSpace(Value val) {
  if (auto memrefType = dyn_cast<MemRefType>(val.getType())) {
    auto addrSpace = hivm::getOptionalHIVMAddressSpace(memrefType);
    return addrSpace.has_value() && *addrSpace == hivm::AddressSpace::GM;
  }
  return false;
}

static bool isAllocWorkspace(Value val) {
  auto *defOp = val.getDefiningOp();
  return defOp && isa<bishengir::memref_ext::AllocWorkspaceOp>(defOp);
}

static StringRef coreTypeName(hivm::TCoreType ct) {
  switch (ct) {
  case hivm::TCoreType::VECTOR:
    return "VECTOR";
  case hivm::TCoreType::CUBE:
    return "CUBE";
  case hivm::TCoreType::CUBE_OR_VECTOR:
    return "CUBE_OR_VECTOR";
  case hivm::TCoreType::CUBE_AND_VECTOR:
    return "CUBE_AND_VECTOR";
  }
  llvm_unreachable("unknown TCoreType");
}

/// Find the core type of HIVM ops that write to `buffer`.
/// When all writers share the same core type, returns it directly.
/// When conflicting writers exist and `refOp` is provided, returns the core
/// type of the last writer before `refOp` in program order.
/// Returns nullopt if no writer is found or order cannot be determined.
static std::optional<hivm::TCoreType>
getWriterCoreType(Value buffer, Operation *refOp = nullptr) {
  struct WriterInfo {
    Operation *op;
    hivm::TCoreType coreType;
  };
  SmallVector<WriterInfo> writers;

  // Step 1: Collect all DPS writers of `buffer` (including through aliases
  // like subview/reinterpret_cast).  Skip CUBE_AND_VECTOR / CUBE_OR_VECTOR.
  SmallVector<Value, 4> worklist = {buffer};
  SmallPtrSet<Value, 8> visited;
  SmallPtrSet<Operation *, 16> seenWriters;

  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    if (!visited.insert(val).second)
      continue;

    for (Operation *user : val.getUsers()) {
      if (auto viewLike = dyn_cast<ViewLikeOpInterface>(user)) {
        if (viewLike.getViewSource() == val)
          worklist.push_back(user->getResult(0));
        continue;
      }

      auto coreTypeIface = dyn_cast<hivm::CoreTypeInterface>(user);
      if (!coreTypeIface)
        continue;
      auto dps = dyn_cast<DestinationStyleOpInterface>(user);
      if (!dps)
        continue;

      for (OpOperand &init : dps.getDpsInitsMutable()) {
        if (init.get() == val) {
          auto ct = coreTypeIface.getCoreType();
          if (!ct)
            continue;
          if (*ct == hivm::TCoreType::CUBE_AND_VECTOR ||
              *ct == hivm::TCoreType::CUBE_OR_VECTOR)
            continue;
          if (seenWriters.insert(user).second)
            writers.push_back({user, *ct});
        }
      }
    }
  }

  if (writers.empty())
    return std::nullopt;

  // Step 2: Fast path — all writers agree on core type.
  bool allSame = llvm::all_of(writers, [&](const WriterInfo &w) {
    return w.coreType == writers[0].coreType;
  });
  if (allSame)
    return writers[0].coreType;

  // Step 3: Conflicting writers — resolve by program order.
  // Return the core type of the last writer before refOp.
  // Without refOp we cannot determine order, so give up.
  if (!refOp)
    return std::nullopt;

  Block *refBlock = refOp->getBlock();
  Operation *refAncestor = refBlock->findAncestorOpInBlock(*refOp);
  if (!refAncestor)
    return std::nullopt;

  // Step 3a: Sort writers by program order.  For ops nested in different
  // regions (e.g. inside scf.for), compare their ancestors in refBlock.
  llvm::sort(writers, [&](const WriterInfo &a, const WriterInfo &b) {
    Operation *ancA = refBlock->findAncestorOpInBlock(*a.op);
    Operation *ancB = refBlock->findAncestorOpInBlock(*b.op);
    if (!ancA || !ancB)
      return false;
    if (ancA != ancB)
      return ancA->isBeforeInBlock(ancB);
    if (a.op->getBlock() == b.op->getBlock())
      return a.op->isBeforeInBlock(b.op);
    return false;
  });

  // Step 3b: Scan sorted writers, pick the last one before refOp.
  std::optional<hivm::TCoreType> lastCT;
  for (auto &w : writers) {
    Operation *ancestor = refBlock->findAncestorOpInBlock(*w.op);
    if (!ancestor)
      continue;
    if (ancestor->isBeforeInBlock(refAncestor) || ancestor == refAncestor)
      lastCT = w.coreType;
  }

  LLVM_DEBUG(if (lastCT) {
    DBGS() << "conflicting writers for buffer, resolved to "
           << coreTypeName(*lastCT) << " (last writer before ref op)\n";
  });

  return lastCT;
}

/// Find the core type of HIVM ops that read from `buffer` (or its aliases).
/// A "reader" is any core-typed op that uses the buffer as a non-DPS-init
/// operand, or any core-typed op at all if it doesn't implement DPS.
/// Returns nullopt if no reader is found or if readers have conflicting types.
static std::optional<hivm::TCoreType> getReaderCoreType(Value buffer) {
  SmallVector<Value, 4> worklist = {buffer};
  SmallPtrSet<Value, 8> visited;
  std::optional<hivm::TCoreType> result;

  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    if (!visited.insert(val).second)
      continue;

    for (Operation *user : val.getUsers()) {
      if (auto viewLike = dyn_cast<ViewLikeOpInterface>(user)) {
        if (viewLike.getViewSource() == val)
          worklist.push_back(user->getResult(0));
        continue;
      }

      auto coreTypeIface = dyn_cast<hivm::CoreTypeInterface>(user);
      if (!coreTypeIface)
        continue;
      auto ct = coreTypeIface.getCoreType();
      if (!ct)
        continue;
      if (*ct == hivm::TCoreType::CUBE_AND_VECTOR ||
          *ct == hivm::TCoreType::CUBE_OR_VECTOR)
        continue;

      if (!result)
        result = ct;
      else if (*result != *ct)
        return std::nullopt;
    }
  }
  return result;
}

/// Find the best insertion point for a workspace alloc at function level:
/// before the earliest memref.alloc or the ancestor enclosing the cross-core
/// buffer, whichever comes first.
static Operation *findWorkspaceInsertionPoint(Operation *allocOp,
                                              func::FuncOp funcOp) {
  Block &entryBlock = funcOp.getBody().front();
  Operation *ancestor = entryBlock.findAncestorOpInBlock(*allocOp);
  assert(ancestor && "allocOp must be inside funcOp");
  for (auto &op : entryBlock) {
    if (isa<memref::AllocOp>(&op) || &op == ancestor)
      return &op;
  }
  return ancestor;
}

struct TileLangIRInsertWorkspace
    : impl::TileLangIRInsertWorkspaceBase<TileLangIRInsertWorkspace> {

  struct CoreTypedOp {
    Operation *op;
    hivm::TCoreType coreType;
    bool isWriter;
  };

  struct BufferInfo {
    memref::AllocOp allocOp;
    SmallVector<CoreTypedOp> ops;
  };

  /// Handle a memref.copy user of the analyzed buffer (or one of its aliases).
  ///
  /// memref.copy bridges two buffers, so we infer the core type transitively:
  ///  - Buffer is the DESTINATION of the copy: the writer core type comes from
  ///    the copy's source (via getWriterCoreType).
  ///  - Buffer is the SOURCE of the copy: the reader core type comes from the
  ///    copy's direct target value (via getReaderCoreType).  We intentionally
  ///    use the target value rather than its root alloc to avoid double-counting
  ///    cross-core users that are already tracked by the root alloc's own
  ///    analysis.
  void collectCopyOp(memref::CopyOp copyOp, Value val, BufferInfo &info,
                     SmallPtrSet<Operation *, 16> &addedOps) {
    // Case 1: buffer (or alias) is the copy destination.
    if (copyOp.getTarget() == val &&
        addedOps.insert(copyOp.getOperation()).second) {
      Value sourceRoot = getRootAlloc(copyOp.getSource());
      if (!hasGMAddressSpace(sourceRoot) && !isAllocWorkspace(sourceRoot)) {
        if (auto writerCT =
                getWriterCoreType(sourceRoot, copyOp.getOperation())) {
          LLVM_DEBUG(DBGS() << "  copy-writer (" << coreTypeName(*writerCT)
                            << ",W): " << *copyOp << "\n");
          info.ops.push_back(
              {copyOp.getOperation(), *writerCT, /*isWriter=*/true});
        }
      }
    }

    // Case 2: buffer (or alias) is the copy source.
    if (copyOp.getSource() == val &&
        addedOps.insert(copyOp.getOperation()).second) {
      Value target = copyOp.getTarget();
      Value targetRoot = getRootAlloc(target);
      if (!hasGMAddressSpace(targetRoot) && !isAllocWorkspace(targetRoot)) {
        if (auto readerCT = getReaderCoreType(target)) {
          LLVM_DEBUG(DBGS() << "  copy-reader (" << coreTypeName(*readerCT)
                            << ",R): " << *copyOp << "\n");
          info.ops.push_back(
              {copyOp.getOperation(), *readerCT, /*isWriter=*/false});
        }
      }
    }
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    LLVM_DEBUG(DBGS() << "processing function: " << funcOp.getSymName()
                      << "\n");

    SmallVector<BufferInfo> buffers;

    // Phase 1: For each alloc, collect all core-typed ops that access it.
    auto walkResult = funcOp.walk([&](memref::AllocOp allocOp) -> WalkResult {
      Value buffer = allocOp.getResult();
      if (hasGMAddressSpace(buffer)) {
        LLVM_DEBUG(DBGS() << "skip GM alloc: " << *allocOp << "\n");
        return WalkResult::advance();
      }

      LLVM_DEBUG(DBGS() << "analyzing alloc: " << *allocOp << "\n");

      BufferInfo info;
      info.allocOp = allocOp;

      SmallVector<Value, 4> worklist = {buffer};
      SmallPtrSet<Value, 8> visited;
      SmallPtrSet<Operation *, 16> addedOps;

      while (!worklist.empty()) {
        Value val = worklist.pop_back_val();
        if (!visited.insert(val).second)
          continue;

        for (Operation *user : val.getUsers()) {
          if (auto viewLike = dyn_cast<ViewLikeOpInterface>(user)) {
            if (viewLike.getViewSource() == val)
              worklist.push_back(user->getResult(0));
            continue;
          }

          if (auto coreTypeIface = dyn_cast<hivm::CoreTypeInterface>(user)) {
            if (auto ct = coreTypeIface.getCoreType()) {
              if (*ct == hivm::TCoreType::CUBE_AND_VECTOR) {
                LLVM_DEBUG(DBGS()
                           << "  skip CUBE_AND_VECTOR op: " << *user << "\n");
                continue;
              }
              if (*ct == hivm::TCoreType::CUBE_OR_VECTOR) {
                user->emitError("op has CUBE_OR_VECTOR core type; "
                                "insert-workspace cannot determine core "
                                "affinity");
                return WalkResult::interrupt();
              }
              bool isWriter = false;
              if (auto dps = dyn_cast<DestinationStyleOpInterface>(user)) {
                for (OpOperand &init : dps.getDpsInitsMutable()) {
                  if (init.get() == val) {
                    isWriter = true;
                    break;
                  }
                }
              }
              if (addedOps.insert(user).second) {
                LLVM_DEBUG(DBGS() << "  core-typed user (" << coreTypeName(*ct)
                                  << (isWriter ? ",W" : ",R") << "): " << *user
                                  << "\n");
                info.ops.push_back({user, *ct, isWriter});
              } else if (isWriter) {
                for (auto &entry : info.ops) {
                  if (entry.op == user) {
                    entry.isWriter = true;
                    break;
                  }
                }
              }
            }
            continue;
          }

          if (auto copyOp = dyn_cast<memref::CopyOp>(user))
            collectCopyOp(copyOp, val, info, addedOps);
        }
      }

      if (info.ops.size() < 2)
        return WalkResult::advance();

      bool hasVector = false, hasCube = false;
      for (auto &op : info.ops) {
        if (op.coreType == hivm::TCoreType::VECTOR)
          hasVector = true;
        else if (op.coreType == hivm::TCoreType::CUBE)
          hasCube = true;
      }
      if (!hasVector || !hasCube)
        return WalkResult::advance();

      LLVM_DEBUG(DBGS() << "  => cross-core buffer: " << info.ops.size()
                        << " ops\n");
      buffers.push_back(std::move(info));
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
      return signalPassFailure();

    // Phase 2: For each cross-core buffer, process boundaries.
    for (auto &info : buffers)
      processBuffer(funcOp, info);
  }

  /// For a single cross-core buffer, sort its accessors by program order,
  /// detect core-type transition boundaries, and insert workspace transfers.
  ///
  /// Given a buffer accessed by ops of mixed core types in this order:
  ///   [CUBE_W, CUBE_R, VECTOR_R, VECTOR_R, CUBE_R, ...]
  ///                    ^boundary0           ^boundary1
  ///
  /// A boundary is "real" only when the preceding segment contains at least
  /// one writer.  Read-only segments don't produce new data, so we skip or
  /// reuse previously created buffers instead of inserting redundant copies.
  ///
  /// For each *real* boundary we produce:
  ///   1. newBuffer = memref.alloc (same type as original)
  ///   2. memref.copy currentBuffer -> workspace   (after last writer op)
  ///   3. memref.copy workspace     -> newBuffer   (immediately after)
  ///   4. Replace reader-segment ops' references to original with newBuffer
  ///
  /// For a *soft* boundary (preceding segment read-only):
  ///   - If coreTypeBuffer already has a buffer for readerCoreType, reuse it.
  ///   - Otherwise the reader's core type matches the original writer → no-op.
  ///
  /// `coreTypeBuffer` maps each core type to its dedicated buffer.
  /// `currentBuffer` tracks the latest written buffer for data chaining.
  void processBuffer(func::FuncOp funcOp, BufferInfo &info) {
    memref::AllocOp allocOp = info.allocOp;
    SmallVector<CoreTypedOp> &ops = info.ops;

    // Step 1: Sort ops by program order in the alloc's enclosing block
    Block *block = allocOp->getBlock();
    llvm::sort(ops, [&](const CoreTypedOp &a, const CoreTypedOp &b) {
      Operation *ancA = block->findAncestorOpInBlock(*a.op);
      Operation *ancB = block->findAncestorOpInBlock(*b.op);
      assert(ancA && ancB && "ops must be in scope of alloc's block");
      if (ancA != ancB)
        return ancA->isBeforeInBlock(ancB);
      if (a.op->getBlock() == b.op->getBlock())
        return a.op->isBeforeInBlock(b.op);
      return false;
    });

    LLVM_DEBUG({
      DBGS() << "sorted ops for " << *allocOp << ":\n";
      for (size_t i = 0; i < ops.size(); ++i)
        DBGS() << "  [" << i << "] " << coreTypeName(ops[i].coreType)
               << (ops[i].isWriter ? "(W)" : "(R)") << ": " << *ops[i].op
               << "\n";
    });

    // Step 2: Find cross-core boundaries
    SmallVector<size_t> boundaries;
    for (size_t i = 1; i < ops.size(); ++i) {
      if (ops[i].coreType != ops[i - 1].coreType)
        boundaries.push_back(i);
    }

    if (boundaries.empty())
      return;

    LLVM_DEBUG(DBGS() << boundaries.size() << " cross-core boundary(s)\n");

    // Step 3: Prepare workspace type (identity layout, for GM)
    Value buffer = allocOp.getResult();
    auto bufferType = cast<MemRefType>(buffer.getType());
    auto wsType =
        MemRefType::get(bufferType.getShape(), bufferType.getElementType());

    Value originalBuffer = buffer;
    Value currentBuffer = buffer;
    Value workspace; // lazily created

    // coreTypeBuffer[ct] = the buffer that ops of core type `ct` should use.
    DenseMap<hivm::TCoreType, Value> coreTypeBuffer;
    coreTypeBuffer[ops[0].coreType] = originalBuffer;

    // Collect memref.copy ops that use originalBuffer (or aliases) as SOURCE.
    // These are not in the ops list but need source redirection after split.
    SmallVector<memref::CopyOp> sourceCopies;
    {
      SmallVector<Value, 4> aliasWorklist = {originalBuffer};
      SmallPtrSet<Value, 8> aliasVisited;
      while (!aliasWorklist.empty()) {
        Value val = aliasWorklist.pop_back_val();
        if (!aliasVisited.insert(val).second)
          continue;
        for (Operation *user : val.getUsers()) {
          if (auto viewLike = dyn_cast<ViewLikeOpInterface>(user)) {
            if (viewLike.getViewSource() == val)
              aliasWorklist.push_back(user->getResult(0));
            continue;
          }
          if (auto copyOp = dyn_cast<memref::CopyOp>(user))
            if (copyOp.getSource() == val)
              sourceCopies.push_back(copyOp);
        }
      }
    }

    // segmentTransitions[i] = (anchorOp, buffer) where anchorOp is the last op
    // before boundary i (in alloc's block), and buffer is what the reader
    // segment after that boundary uses.
    SmallVector<std::pair<Operation *, Value>> segmentTransitions;

    for (size_t bIdx = 0; bIdx < boundaries.size(); ++bIdx) {
      size_t segStart = (bIdx == 0) ? 0 : boundaries[bIdx - 1];
      size_t readerStart = boundaries[bIdx];
      size_t readerEnd =
          (bIdx + 1 < boundaries.size()) ? boundaries[bIdx + 1] : ops.size();

      hivm::TCoreType readerCore = ops[readerStart].coreType;

      // Check if the preceding segment [segStart, readerStart) has writers.
      bool precedingHasWriter = false;
      for (size_t i = segStart; i < readerStart; ++i) {
        if (ops[i].isWriter) {
          precedingHasWriter = true;
          break;
        }
      }

      LLVM_DEBUG(DBGS() << "boundary " << bIdx << ": "
                        << coreTypeName(ops[readerStart - 1].coreType) << " -> "
                        << coreTypeName(readerCore)
                        << (precedingHasWriter ? " [real]" : " [soft]")
                        << "\n");

      if (precedingHasWriter) {
        // Real boundary: data changed, insert workspace copy.
        Operation *lastWriterOp = ops[readerStart - 1].op;
        LLVM_DEBUG(DBGS() << "  last writer: " << *lastWriterOp << "\n");

        // Lazily create workspace
        if (!workspace) {
          Operation *wsInsertPt = findWorkspaceInsertionPoint(allocOp, funcOp);
          OpBuilder wsBuilder(wsInsertPt);
          auto wsOp = wsBuilder.create<bishengir::memref_ext::AllocWorkspaceOp>(
              allocOp.getLoc(), wsType, Value(), ValueRange{}, ValueRange{});
          workspace = wsOp.getMemref();
          LLVM_DEBUG(DBGS() << "  workspace: " << wsType << "\n");
        }

        // 4a. Allocate new buffer for reader segment.
        OpBuilder allocBuilder(allocOp->getBlock(),
                               std::next(Block::iterator(allocOp)));
        auto newAlloc =
            allocBuilder.create<memref::AllocOp>(allocOp.getLoc(), bufferType);
        Value newBuffer = newAlloc.getResult();

        LLVM_DEBUG(DBGS() << "  new buffer: " << *newAlloc << "\n");

        // 4b. Insert two copies: currentBuffer -> workspace -> newBuffer
        OpBuilder copyBuilder(lastWriterOp->getBlock(),
                              std::next(Block::iterator(lastWriterOp)));
        auto copy1 = copyBuilder.create<memref::CopyOp>(
            lastWriterOp->getLoc(), currentBuffer, workspace);
        OpBuilder afterCopy1(copy1->getBlock(),
                             std::next(Block::iterator(copy1)));
        auto copy2 = afterCopy1.create<memref::CopyOp>(lastWriterOp->getLoc(),
                                                       workspace, newBuffer);

        LLVM_DEBUG(DBGS() << "  copy1: " << *copy1 << "\n");
        LLVM_DEBUG(DBGS() << "  copy2: " << *copy2 << "\n");

        // 4c. Rewrite reader-segment ops to use newBuffer.
        DenseMap<Value, Value> aliasMap;
        aliasMap[originalBuffer] = newBuffer;
        for (size_t i = readerStart; i < readerEnd; ++i)
          replaceBufferUses(ops[i].op, originalBuffer, newBuffer, aliasMap);

        LLVM_DEBUG(DBGS() << "  replaced " << (readerEnd - readerStart)
                          << " reader op(s)\n");

        coreTypeBuffer[readerCore] = newBuffer;
        currentBuffer = newBuffer;

        segmentTransitions.push_back({lastWriterOp, newBuffer});

      } else {
        // Soft boundary: preceding segment is read-only, no new data.
        // Reuse previously created buffer for this core type if available.
        Value segBuffer = originalBuffer;
        auto it = coreTypeBuffer.find(readerCore);
        if (it != coreTypeBuffer.end() && it->second != originalBuffer) {
          Value existingBuffer = it->second;
          DenseMap<Value, Value> aliasMap;
          aliasMap[originalBuffer] = existingBuffer;
          for (size_t i = readerStart; i < readerEnd; ++i)
            replaceBufferUses(ops[i].op, originalBuffer, existingBuffer,
                              aliasMap);

          LLVM_DEBUG(DBGS() << "  reuse existing buffer for "
                            << coreTypeName(readerCore) << ", replaced "
                            << (readerEnd - readerStart) << " op(s)\n");
          segBuffer = existingBuffer;
        } else {
          LLVM_DEBUG(DBGS() << "  skip: reader core matches original writer, "
                               "no replacement needed\n");
        }
        // Don't advance currentBuffer: no new data was produced.
        segmentTransitions.push_back({ops[readerStart - 1].op, segBuffer});
      }
    }

    // Step 5: Redirect memref.copy ops that read from originalBuffer.
    // Each copy's source should point to the buffer active at its program
    // position (determined by the last segment transition before the copy).
    if (!sourceCopies.empty() && !segmentTransitions.empty()) {
      LLVM_DEBUG(DBGS() << "redirecting " << sourceCopies.size()
                        << " source copy(s)\n");
      for (auto copyOp : sourceCopies) {
        Operation *ancCopy =
            block->findAncestorOpInBlock(*copyOp.getOperation());
        if (!ancCopy)
          continue;

        Value targetBuf = originalBuffer;
        for (auto &[anchorOp, buf] : segmentTransitions) {
          Operation *ancAnchor = block->findAncestorOpInBlock(*anchorOp);
          if (ancAnchor && ancAnchor->isBeforeInBlock(ancCopy))
            targetBuf = buf;
        }

        if (targetBuf != originalBuffer) {
          DenseMap<Value, Value> aliasMap;
          aliasMap[originalBuffer] = targetBuf;
          replaceBufferUses(copyOp.getOperation(), originalBuffer, targetBuf,
                            aliasMap);
          LLVM_DEBUG(DBGS() << "  redirected copy source: " << *copyOp << "\n");
        }
      }
    }

    // Step 6: Erase dead view-like ops (e.g. subview) on originalBuffer.
    // getOrCreateAlias clones subviews onto the new buffer but does not remove
    // the originals; clean up any that have become unused.
    {
      SmallVector<Operation *> deadOps;
      SmallVector<Value, 4> cleanupWorklist = {originalBuffer};
      SmallPtrSet<Value, 8> cleanupVisited;
      while (!cleanupWorklist.empty()) {
        Value val = cleanupWorklist.pop_back_val();
        if (!cleanupVisited.insert(val).second)
          continue;
        for (Operation *user : val.getUsers()) {
          if (auto viewLike = dyn_cast<ViewLikeOpInterface>(user)) {
            if (viewLike.getViewSource() == val) {
              Value result = user->getResult(0);
              cleanupWorklist.push_back(result);
              if (result.use_empty())
                deadOps.push_back(user);
            }
          }
        }
      }
      for (auto *op : llvm::reverse(deadOps))
        op->erase();
    }
  }

  /// Replace uses of oldBuffer (and its view-like aliases) with newBuffer in
  /// the given op.
  void replaceBufferUses(Operation *op, Value oldBuffer, Value newBuffer,
                         DenseMap<Value, Value> &aliasMap) {
    for (OpOperand &operand : op->getOpOperands()) {
      Value val = operand.get();

      if (val == oldBuffer) {
        operand.set(newBuffer);
        continue;
      }

      if (getRootAlloc(val) == oldBuffer) {
        Value newVal = getOrCreateAlias(val, aliasMap);
        if (newVal != val)
          operand.set(newVal);
      }
    }
  }

  /// Lazily clone a view-like chain so that it roots at the new buffer.
  Value getOrCreateAlias(Value oldVal, DenseMap<Value, Value> &aliasMap) {
    auto it = aliasMap.find(oldVal);
    if (it != aliasMap.end())
      return it->second;

    auto *defOp = oldVal.getDefiningOp();

    if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp)) {
      Value newSource = getOrCreateAlias(subviewOp.getSource(), aliasMap);
      OpBuilder builder(subviewOp);
      auto newSubview = builder.create<memref::SubViewOp>(
          subviewOp.getLoc(), newSource, subviewOp.getMixedOffsets(),
          subviewOp.getMixedSizes(), subviewOp.getMixedStrides());
      aliasMap[oldVal] = newSubview.getResult();
      return newSubview.getResult();
    }

    aliasMap[oldVal] = oldVal;
    return oldVal;
  }
};

#undef DBGS
#undef DEBUG_TYPE
} // namespace

} // namespace tilelangir
} // namespace mlir
