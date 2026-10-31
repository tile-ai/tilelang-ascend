// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

// Preserve the pinned TA Simplify binding-elimination boundary when the
// shared Simplify operates on flat TIRX Bind rather than nested LetStmt.
#include <unordered_map>
#include <utility>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "support/check.h"

#include "common/binding_cleanup.h"

namespace tvm::tl::a2a3 {
namespace {
using namespace tirx;
using namespace ffi;
using binding_cleanup::VarSet;

// TA 0deb8b93, src/tir/transforms/simplify.cc::CanInlineLetStmt tests the
// original RHS, before VisitExpr simplifies it. In particular, a floating
// expression that later folds to a constant is not newly eligible here.
class OriginalInlineBindings : public StmtExprVisitor {
public:
  static VarSet Collect(const PrimFunc &func) {
    OriginalInlineBindings visitor;
    visitor(func->body);
    return std::move(visitor.eligible_);
  }

private:
  void VisitStmt_(const BindNode *op) final {
    const PrimExpr &value = op->value;
    if (is_const_number(value) || value.as<VarNode>() ||
        (value.dtype().is_int() &&
         SideEffect(value) <= CallEffectKind::kPure)) {
      eligible_.insert(op->var);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  VarSet eligible_;
};

// Buffer definitions were already an exception in TA Simplify. TIRX also
// carries real scalar dependencies in annotations and function metadata.
class DefinitionUses : public StmtExprVisitor {
public:
  static VarSet Collect(const PrimFunc &func) {
    DefinitionUses visitor;
    visitor.uses_.IncludeMetadata(func->attrs);
    visitor.uses_.IncludeMetadata(func->buffer_map);
    visitor.uses_.IncludeMetadata(func->ret_type);
    for (const Var &param : func->params) {
      visitor.uses_.IncludeDefinitionMetadata(param);
    }
    visitor(func->body);
    return std::move(visitor.used_);
  }

private:
  DefinitionUses() : uses_(&used_, std::nullopt, true) {}

  void VisitBufferDef(const Buffer &buffer, bool alloc_data) final {
    uses_.IncludeDefinitionMetadata(buffer);
  }

  void VisitBufferUse(const Buffer &buffer) final {
    uses_.IncludeDefinitionMetadata(buffer);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    uses_.IncludeMetadata(op->annotations);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    uses_.IncludeMetadata(op->annotations);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SBlockNode *op) final {
    uses_.IncludeMetadata(op->annotations);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    uses_.IncludeMetadata(op->node);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    uses_.IncludeMetadata(op->annotations);
    StmtExprVisitor::VisitExpr_(op);
  }

  VarSet used_;
  binding_cleanup::Uses uses_;
};

VarSet StillPureBindings(const PrimFunc &func, const VarSet &original) {
  VarSet result;
  PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (const auto *bind = node.as<BindNode>();
        bind && original.count(bind->var) &&
        SideEffect(bind->value) <= CallEffectKind::kPure) {
      result.insert(bind->var);
    }
  });
  return result;
}

} // namespace

tvm::transform::Pass AscendSimplify() {
  auto pass_func = [](IRModule mod,
                      tvm::transform::PassContext ctx) -> IRModule {
    std::unordered_map<GlobalVar, VarSet, ObjectPtrHash, ObjectPtrEqual>
        eligible;
    std::unordered_map<GlobalVar, VarSet, ObjectPtrHash, ObjectPtrEqual>
        protected_bindings;
    for (const auto &[symbol, base_func] : mod->functions) {
      if (auto func = base_func.as<PrimFunc>()) {
        eligible.emplace(symbol, OriginalInlineBindings::Collect(func.value()));
        protected_bindings.emplace(
            symbol, binding_cleanup::CollectProtectedBindings(func.value()));
        VarSet metadata = DefinitionUses::Collect(func.value());
        protected_bindings.at(symbol).insert(metadata.begin(), metadata.end());
      }
    }

    // Keep the real module, PassContext, config, analyzer and shared pass.
    // Never rebuild a temporary one-function module or move a phase.
    mod = tirx::transform::Simplify()(std::move(mod));
    IRModule result = mod->ShallowCopy();
    for (const auto &[symbol, base_func] : mod->functions) {
      auto func = base_func.as<PrimFunc>();
      if (!func || !eligible.count(symbol)) {
        continue;
      }
      VarSet retained = DefinitionUses::Collect(func.value());
      const VarSet &before = protected_bindings.at(symbol);
      retained.insert(before.begin(), before.end());
      VarSet after = binding_cleanup::CollectProtectedBindings(func.value());
      retained.insert(after.begin(), after.end());
      VarSet candidates = StillPureBindings(func.value(), eligible.at(symbol));
      Stmt body = binding_cleanup::BackwardBindings(
          retained, retained, false, true,
          std::move(candidates))(func.value()->body);
      if (!body.same_as(func.value()->body)) {
        PrimFunc updated = func.value();
        updated.CopyOnWrite()->body = std::move(body);
        result.CopyOnWrite()->Update(symbol, std::move(updated));
      }
    }
    return result;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.A2A3Simplify", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.A2A3Simplify", AscendSimplify);
}

} // namespace tvm::tl::a2a3
