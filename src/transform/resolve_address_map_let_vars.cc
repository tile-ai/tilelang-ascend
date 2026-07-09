// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file resolve_address_map_let_vars.cc
 * \brief Resolve let-bound variables in address_map annotations before
 *        Simplify pass eliminates the LetStmt nodes.
 */

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

class LetBindingCollector : public StmtExprVisitor {
public:
  std::unordered_map<const VarNode *, PrimExpr> bindings;
  void VisitStmt_(const LetStmtNode *op) final {
    bindings[op->var.get()] = op->value;
    StmtExprVisitor::VisitStmt_(op);
  }
};

class AddressMapLetVarResolver : public StmtMutator {
public:
  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    BlockRealize realize = Downcast<BlockRealize>(StmtMutator::VisitStmt_(op));
    Block block = realize->block;

    if (block->annotations.count("address_map")) {
      auto raw_map =
          block->annotations.at("address_map").as<Map<Var, PrimExpr>>();
      if (raw_map) {
        std::unordered_map<const VarNode *, PrimExpr> let_bindings;
        LetBindingCollector collector;
        collector(block->body);
        let_bindings = std::move(collector.bindings);

        if (!let_bindings.empty()) {
          Map<Var, PrimExpr> vmap;
          for (const auto &lb : let_bindings) {
            vmap.Set(GetRef<Var>(lb.first), lb.second);
          }
          Map<Var, PrimExpr> resolved_map;
          for (const auto &kv : raw_map.value()) {
            PrimExpr resolved = tir::Substitute(kv.second, vmap);
            resolved_map.Set(kv.first, resolved);
          }
          auto block_ptr = block.CopyOnWrite();
          block_ptr->annotations.Set("address_map", resolved_map);
          auto realize_ptr = realize.CopyOnWrite();
          realize_ptr->block = block;
          return realize;
        }
      }
    }
    return realize;
  }
};

tvm::transform::Pass ResolveAddressMapLetVars() {
  auto pass_func = [=](PrimFunc f, IRModule m,
                       tvm::transform::PassContext ctx) {
    auto *fptr = f.CopyOnWrite();
    fptr->body = AddressMapLetVarResolver()(std::move(fptr->body));
    return f;
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0,
                                            "tl.ResolveAddressMapLetVars", {});
}

TVM_REGISTER_GLOBAL("tl.transform.ResolveAddressMapLetVars")
    .set_body_typed(ResolveAddressMapLetVars);

} // namespace tl
} // namespace tvm
