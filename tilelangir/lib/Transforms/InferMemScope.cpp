// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "tilelangir/Transforms/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/InferHIVMMemScope.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRINFERMEMSCOPE
#include "tilelangir/Transforms/Passes.h.inc"

namespace {
#define DEBUG_TYPE "tilelangir-infer-mem-scope"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace hivm;

// ===----------------------------------------------------------------------===//
// Replaces AscendNPU-IR's MemScopeInferAndPropagateHelper with a more general
// approach: instead of a whitelist for single-result ops, we propagate the scope
// to *every* memref-typed result of any user op.
// ===----------------------------------------------------------------------===//
class MemScopePropagator {
public:
  LogicalResult run(Value operand, AddressSpaceAttr targetScope) {
    auto memRefType = dyn_cast<BaseMemRefType>(operand.getType());
    if (!memRefType)
      return failure();
    if (memRefType.getMemorySpace())
      return success();

    setBaseMemRefTypeScope(operand, targetScope);
    return propagateToUsers(operand);
  }

private:
  static BlockArgument getTiedWhileBodyIterArg(scf::WhileOp op,
                                               OpOperand *opOperand) {
    auto argsMutable = op.getInitsMutable();
    auto *it = llvm::find(argsMutable, *opOperand);
    if (it == argsMutable.end())
      return {};
    return op.getAfterArguments()[std::distance(argsMutable.begin(), it)];
  }

  LogicalResult propagateToUsers(Value val) {
    auto memrefScope = getHIVMAddressSpaceAttr(val.getType());

    for (OpOperand &use : val.getUses()) {
      Operation *userOp = use.getOwner();
      LogicalResult res =
          TypeSwitch<Operation *, LogicalResult>(userOp)
              .Case<scf::YieldOp>([&](scf::YieldOp op) {
                Operation *parentOp = op->getParentOp();
                Value yieldOperand = op.getOperand(use.getOperandNumber());
                if (!isa<BaseMemRefType>(yieldOperand.getType()))
                  return success();
                Value parentResult =
                    parentOp->getResult(use.getOperandNumber());
                setBaseMemRefTypeScope(parentResult, memrefScope);
                return propagateToUsers(parentResult);
              })
              .Case<scf::ForOp>([&](scf::ForOp op) {
                Value result = op.getTiedLoopResult(&use);
                setBaseMemRefTypeScope(result, memrefScope);
                Value bbArg = op.getTiedLoopRegionIterArg(&use);
                setBaseMemRefTypeScope(bbArg, memrefScope);
                return success(
                    propagateToUsers(bbArg).succeeded() &&
                    propagateToUsers(result).succeeded());
              })
              .Case<scf::WhileOp>([&](scf::WhileOp op) {
                BlockArgument bbArg =
                    cast<BlockArgument>(op.getTiedLoopRegionIterArg(&use));
                auto yield = op.getTiedLoopYieldedValue(bbArg);
                BlockArgument afterArg = getTiedWhileBodyIterArg(op, &use);
                setBaseMemRefTypeScope(bbArg, memrefScope);
                setBaseMemRefTypeScope(yield->get(), memrefScope);
                setBaseMemRefTypeScope(afterArg, memrefScope);
                return success(
                    propagateToUsers(afterArg).succeeded() &&
                    propagateToUsers(bbArg).succeeded() &&
                    propagateToUsers(yield->get()).succeeded());
              })
              .Case<func::CallOp>([&](auto) { return success(); })
              .Default([&](Operation *op) {
                if (op->getNumResults() == 0)
                  return success();
                for (OpResult result : op->getResults()) {
                  if (!isa<BaseMemRefType>(result.getType()))
                    continue;
                  setBaseMemRefTypeScope(result, memrefScope);
                  if (failed(propagateToUsers(result)))
                    return failure();
                }
                return success();
              });
      if (failed(res))
        return failure();
    }
    return success();
  }
};

/// Set address space on a root value and propagate to all users.
static LogicalResult setAllocScope(Value rootVal,
                                   hivm::AddressSpace space) {
  auto memRefType = dyn_cast<BaseMemRefType>(rootVal.getType());
  if (!memRefType)
    return success();
  if (memRefType.getMemorySpace())
    return success();

  auto spaceAttr = AddressSpaceAttr::get(rootVal.getContext(), space);
  MemScopePropagator propagator;
  return propagator.run(rootVal, spaceAttr);
}

/// Infer and set UB scope for all memref operands of a VECTOR-core op.
static LogicalResult handleVectorOp(Operation *op) {
  for (OpOperand &operand : op->getOpOperands()) {
    Value val = operand.get();
    if (!isa<BaseMemRefType>(val.getType()))
      continue;

    auto rootAlloc = utils::tracebackMemRefToAlloc(val);
    if (!rootAlloc.has_value())
      continue;

    if (failed(setAllocScope(*rootAlloc, hivm::AddressSpace::UB))) {
      return op->emitOpError(
          "Failed to infer/propagate UB scope for VECTOR operand");
    }
  }
  return success();
}

/// Determine the default address space for a remaining memref.alloc, based
/// on the enclosing scope.scope's tcore_type or the function's core type.
static std::optional<hivm::AddressSpace>
getDefaultScope(memref::AllocOp allocOp, func::FuncOp funcOp) {
  if (allocOp.getType().getMemorySpace())
    return std::nullopt;

  Operation *parent = allocOp->getParentOp();
  while (parent && parent != funcOp.getOperation()) {
    if (auto scopeOp = dyn_cast<scope::ScopeOp>(parent)) {
      if (auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(
              hivm::TCoreTypeAttr::name)) {
        auto ct = attr.getTcoretype();
        if (ct == hivm::TCoreType::VECTOR)
          return hivm::AddressSpace::UB;
        if (ct == hivm::TCoreType::CUBE)
          return hivm::AddressSpace::L1;
      }
    }
    parent = parent->getParentOp();
  }

  auto funcCoreType = hivm::queryFuncCoreType(funcOp);
  if (funcCoreType.has_value()) {
    if (*funcCoreType == hivm::TFuncCoreType::AIC)
      return hivm::AddressSpace::L1;
    if (*funcCoreType == hivm::TFuncCoreType::AIV)
      return hivm::AddressSpace::UB;
  }

  return std::nullopt;
}

struct TileLangIRInferMemScope
    : impl::TileLangIRInferMemScopeBase<TileLangIRInferMemScope> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    LLVM_DEBUG(DBGS() << "processing function: " << funcOp.getSymName()
                      << "\n");

    // Phase 1: memref_ext.alloc_workspace → GM.
    funcOp.walk([&](bishengir::memref_ext::AllocWorkspaceOp op) {
      LLVM_DEBUG(DBGS() << "Phase 1 workspace: " << *op << "\n");
      if (failed(setAllocScope(op.getMemref(), hivm::AddressSpace::GM)))
        return signalPassFailure();
    });

    // Phase 2: hivm.hir.mmadL1 → mA/mB → L1, mC → L0C.
    funcOp.walk([&](hivm::MmadL1Op op) {
      LLVM_DEBUG(DBGS() << "Phase 2 mmadL1: " << *op << "\n");
      if (failed(hivm::inferAndPropagateMemScopeForMmadL1(op)))
        return signalPassFailure();
    });

    // Phase 3: VECTOR-core HIVM ops → all memref operands → UB.
    auto vectorResult = funcOp.walk([&](Operation *op) -> WalkResult {
      auto ctIface = dyn_cast<hivm::CoreTypeInterface>(op);
      if (!ctIface)
        return WalkResult::advance();
      auto ct = ctIface.getCoreType();
      if (!ct || *ct != hivm::TCoreType::VECTOR)
        return WalkResult::advance();

      LLVM_DEBUG(DBGS() << "Phase 3 VECTOR op: " << *op << "\n");
      if (failed(handleVectorOp(op)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (vectorResult.wasInterrupted())
      return signalPassFailure();

    // Phase 4: Function arguments → GM; update function type.
    LLVM_DEBUG(DBGS() << "Phase 4 func args → GM\n");
    if (failed(hivm::inferAndPropagateMemScopeForFunc(funcOp)))
      return signalPassFailure();

    // Phase 5: Remaining memref.alloc → default scope.
    funcOp.walk([&](memref::AllocOp op) {
      auto scope = getDefaultScope(op, funcOp);
      if (!scope.has_value())
        return;
      LLVM_DEBUG(DBGS() << "Phase 5 remaining alloc → "
                        << hivm::stringifyAddressSpace(*scope) << ": " << *op
                        << "\n");
      if (failed(setAllocScope(op, *scope)))
        signalPassFailure();
    });
  }
};

#undef DBGS
#undef DEBUG_TYPE
} // namespace

} // namespace tilelangir
} // namespace mlir
