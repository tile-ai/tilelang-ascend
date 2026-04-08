// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/CVSplit.cpp
 * \brief TileLangIR CV split pass.
 *
 */

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/IR/HIVMTraits.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "tilelangir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

namespace mlir::tilelangir {

#define GEN_PASS_DEF_TILELANGIRCVSPLIT
#include "tilelangir/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tilelangir-cv-split"
#define LDBG(X)                                                                \
  LLVM_DEBUG(llvm::dbgs() << "[" << DEBUG_TYPE << "] " << X << '\n')

struct TileLangIRCVSplit : impl::TileLangIRCVSplitBase<TileLangIRCVSplit> {

  struct Mapper {
    OpResult def;
    OpOperand &use;

    Mapper(OpResult def, OpOperand &use) : def(def), use(use) {}
  };

  void runOnOperation() override {
    getOperation().walk([this](scf::ForOp forOp) {
      LDBG("Processing " << forOp);
      currentForOp = forOp;
      auto body = forOp.getBody();

      std::size_t groupId = 0;
      DenseMap<const Operation *, std::size_t> opGroupId;
      for (auto copyOp : body->getOps<CopyOpInterface>())
        for (auto val : {copyOp.getSource(), copyOp.getTarget()}) {
          if (!isValFromWorkspace(val))
            continue;
          LDBG("Start from " << *copyOp.getOperation());
          if (visitGroupOfOps(copyOp, [&](const Operation *op) {
                if (opGroupId.find(op) == opGroupId.end()) {
                  opGroupId[op] = groupId;
                  return false;
                }
                return true;
              }))
            break;
          groupId++;
          break;
        }

      SmallVector<hivm::TCoreType> coreType(groupId,
                                            hivm::TCoreType::CUBE_OR_VECTOR);
      SmallVector<SmallVector<Operation *>> groups(groupId);
      SmallVector<std::vector<Mapper>> groupResults(groupId);
      for (auto &op : body->getOperations()) {
        if (opGroupId.find(&op) == opGroupId.end())
          continue;
        const auto id = opGroupId[&op];
        auto opCoreType = hivm::TCoreType::CUBE_OR_VECTOR;
        if (auto hivmOp = dyn_cast<hivm::HIVMStructuredOp>(op);
            hivmOp && hivmOp.getCoreType())
          opCoreType = *hivmOp.getCoreType();
        LDBG("Collecting op " << op << " belonging to group " << id
                              << " with core type "
                              << hivm::stringifyTCoreType(opCoreType));

        if (opCoreType != hivm::TCoreType::CUBE_OR_VECTOR)
          switch (coreType[id]) {
          case hivm::TCoreType::CUBE_OR_VECTOR:
            coreType[id] = opCoreType;
            break;
          case hivm::TCoreType::CUBE:
          case hivm::TCoreType::VECTOR:
            coreType[id] = coreType[id] == opCoreType
                               ? opCoreType
                               : hivm::TCoreType::CUBE_AND_VECTOR;
            break;
          }

        groups[id].push_back(&op);

        for (auto result : op.getResults()) {
          for (auto &use : result.getUses()) {
            if (isa<scf::YieldOp>(use.getOwner()))
              groupResults[id].emplace_back(result, use);
          }
        }
      }

      OpBuilder builder(body->getTerminator());
      for (auto &&[coreType, group, groupResults] :
           llvm::zip(coreType, groups, groupResults)) {
        auto scope = builder.create<scope::ScopeOp>(
            builder.getUnknownLoc(),
            TypeRange(llvm::map_to_vector(groupResults, [](const Mapper &map) {
              return map.use.get().getType();
            })));
        for (auto &&[id, map] : llvm::enumerate(groupResults)) {
          map.use.assign(scope.getResult(id));
        }
        scope->setAttr(hivm::TCoreTypeAttr::name,
                       builder.getAttr<hivm::TCoreTypeAttr>(coreType));

        auto &body = scope.getRegion().emplaceBlock();
        for (auto op : group) {
          op->moveBefore(&body, body.end());
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(&body);
        builder.create<scope::ReturnOp>(
            builder.getUnknownLoc(),
            ValueRange(llvm::map_to_vector(
                groupResults, [](Mapper &map) -> Value { return map.def; })));

        LDBG("Packed a " << hivm::stringifyTCoreType(coreType)
                         << "-core scope:\n"
                         << scope);
      }
    });
  }

private:
  scf::ForOp currentForOp;

  bool visitGroupOfOps(Operation *op,
                       llvm::function_ref<bool(const Operation *)> visitor) {
    if (visitor(op))
      return true;
    LDBG("Visiting " << *op);

    for (auto user : op->getUsers()) {
      if (!currentForOp->isProperAncestor(user))
        continue;
      if (isa<scf::YieldOp>(user))
        continue;
      visitGroupOfOps(user, visitor);
    }

    for (auto operand : op->getOperands()) {
      auto definingOp = operand.getDefiningOp();
      if (!definingOp)
        continue;
      if (isScalarOp(definingOp) ||
          isa<bishengir::memref_ext::AllocWorkspaceOp>(definingOp))
        continue;
      visitGroupOfOps(definingOp, visitor);
    }

    return false;
  }

  bool isValFromWorkspace(Value val) {
    auto definingOp = val.getDefiningOp();
    return definingOp &&
           TypeSwitch<Operation *, bool>(definingOp)
               .Case([&](ViewLikeOpInterface op) {
                 return isValFromWorkspace(op.getViewSource());
               })
               .Case([](bishengir::memref_ext::AllocWorkspaceOp op) {
                 return true;
               })
               .Default(false);
  }

  bool isScalarOp(Operation *op) {
    auto isScalar = [](Value val) {
      return !isa<TensorType, BaseMemRefType, VectorType>(val.getType());
    };
    return llvm::all_of(op->getOperands(), isScalar) &&
           llvm::all_of(op->getResults(), isScalar);
  }
};
#undef DEBUG_TYPE

} // namespace mlir::tilelangir
