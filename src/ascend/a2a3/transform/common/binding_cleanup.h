// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#ifndef TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_BINDING_CLEANUP_H_
#define TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_BINDING_CLEANUP_H_

#include <optional>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>
#include <unordered_set>
#include <utility>

namespace tvm::tl::a2a3::binding_cleanup {
using namespace tirx;
using namespace ffi;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

VarSet CollectProtectedBindings(const PrimFunc &func);

// Shared with the earlier CV repair. Definition metadata is opt-in so that
// extraction does not broaden the behavior of the existing CV entry points.
class Uses : public StmtExprVisitor {
public:
  explicit Uses(VarSet *vars, Optional<Stmt> excluded = std::nullopt,
                bool deep_definitions = false)
      : deep_definitions_(deep_definitions), vars_(vars),
        excluded_(std::move(excluded)) {}

  void IncludeMetadata(const Any &value) { VisitMetadata(value); }
  void IncludeDefinitionMetadata(const ObjectRef &definition) {
    reflection::ForEachFieldInfo(
        TVMFFIGetTypeInfo(definition->type_index()),
        [&](const TVMFFIFieldInfo *field) {
          VisitMetadata(reflection::FieldGetter(field)(definition));
        });
  }

private:
  void VisitStmt(const Stmt &stmt) final {
    if (!excluded_.defined() || !stmt.same_as(excluded_.value())) {
      StmtExprVisitor::VisitStmt(stmt);
    }
  }
  void VisitExpr_(const VarNode *op) final {
    Var var = GetRef<Var>(op);
    vars_->insert(var);
    if (deep_definitions_ && definitions_.insert(var).second) {
      IncludeDefinitionMetadata(var);
    }
  }
  void VisitBufferUse(const Buffer &buffer) final {
    vars_->insert(buffer->data);
    // A descriptor can carry a scalar shape/stride use without a nearby
    // DeclBuffer.  Counting it conservatively cannot discard a needed bind.
    VisitBufferDef(buffer, false);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    if (op->predicate.defined()) {
      VisitExpr(op->predicate.value());
    }
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    if (op->predicate.defined()) {
      VisitExpr(op->predicate.value());
    }
  }
  void VisitStmt_(const ForNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    VisitMetadata(op->annotations);
  }
  void VisitStmt_(const SBlockNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    VisitMetadata(op->annotations);
  }
  void VisitStmt_(const AllocBufferNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    if (deep_definitions_) {
      VisitMetadata(op->annotations);
    }
  }
  void VisitStmt_(const AttrStmtNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    VisitMetadata(op->node);
  }
  void VisitExpr_(const CallNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    VisitMetadata(op->annotations);
  }
  void VisitMetadata(const Any &value) {
    auto object = value.as<ObjectRef>();
    if (!object || !object->defined() || !metadata_.insert(*object).second) {
      return;
    }
    if (auto expr = object->as<PrimExpr>()) {
      VisitExpr(expr.value());
    } else if (auto array = object->as<ArrayObj>()) {
      for (const Any &item : *array) {
        VisitMetadata(item);
      }
    } else if (auto map = object->as<MapObj>()) {
      for (const auto &item : *map) {
        VisitMetadata(item.first);
        VisitMetadata(item.second);
      }
    } else {
      reflection::ForEachFieldInfo(
          TVMFFIGetTypeInfo((*object)->type_index()),
          [&](const TVMFFIFieldInfo *field) {
            VisitMetadata(reflection::FieldGetter(field)(*object));
          });
    }
  }
  bool deep_definitions_;
  VarSet definitions_;
  VarSet *vars_;
  Optional<Stmt> excluded_;
  std::unordered_set<ObjectRef, ObjectPtrHash, ObjectPtrEqual> metadata_;
};

class BackwardBindings : public StmtMutator {
public:
  explicit BackwardBindings(
      VarSet live = {}, VarSet protected_bindings = {},
      bool emit_unused_effects = false, bool deep_definitions = false,
      std::optional<VarSet> eligible_bindings = std::nullopt,
      bool analysis_only = false)
      : eligible_bindings_(std::move(eligible_bindings)),
        analysis_only_(analysis_only), deep_definitions_(deep_definitions),
        emit_unused_effects_(emit_unused_effects),
        protected_(std::move(protected_bindings)), live_(std::move(live)) {}
  bool RemovedBinding() const { return removed_; }
  const VarSet &LiveIn() const { return live_; }

protected:
  Stmt VisitStmt(const Stmt &stmt) final {
    if (deep_definitions_) {
      if (stmt.as<SeqStmtNode>() || stmt.as<BindNode>() ||
          stmt.as<IfThenElseNode>() || stmt.as<ForNode>() ||
          stmt.as<WhileNode>()) {
        return StmtMutator::VisitStmt(stmt);
      }
      VarSet live_out = live_;
      Uses headers(&live_, std::nullopt, true);
      if (const auto *attr = stmt.as<AttrStmtNode>()) {
        headers.IncludeMetadata(attr->node);
        headers(attr->value);
      } else if (const auto *block = stmt.as<SBlockNode>()) {
        headers.IncludeMetadata(block->annotations);
        headers.IncludeMetadata(block->reads);
        headers.IncludeMetadata(block->writes);
        headers.IncludeMetadata(block->alloc_buffers);
        headers.IncludeMetadata(block->match_buffers);
      } else if (const auto *allocation = stmt.as<AllocBufferNode>()) {
        headers.IncludeMetadata(allocation->annotations);
        headers.IncludeDefinitionMetadata(allocation->buffer);
      }
      Stmt result = StmtMutator::VisitStmt(stmt);
      VarSet dependencies;
      Uses{&dependencies, std::nullopt, true}(result);
      class Definitions : public StmtVisitor {
      public:
        VarSet vars;
        void VisitStmt_(const BindNode *op) final { vars.insert(op->var); }
      } definitions;
      definitions(result);
      for (const Var &var : definitions.vars) {
        dependencies.erase(var);
      }
      live_ = std::move(live_out);
      live_.insert(dependencies.begin(), dependencies.end());
      return result;
    }
    VarSet live_out = live_;
    // Branches and nested scopes may consume an ancestor bind.  Seed their
    // complete use set before mutating children, then recompute after empty
    // controls disappear on the next fixed-point iteration.
    if (!stmt.as<SeqStmtNode>() && !stmt.as<BindNode>()) {
      Uses{&live_, std::nullopt, deep_definitions_}(stmt);
    }
    Stmt result = StmtMutator::VisitStmt(stmt);
    if (!stmt.as<BindNode>()) {
      live_ = std::move(live_out);
      Uses{&live_, std::nullopt, deep_definitions_}(result);
    }
    return result;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> reversed;
    for (auto it = op->seq.rbegin(); it != op->seq.rend(); ++it) {
      reversed.push_back(VisitStmt(*it));
    }
    Array<Stmt> statements;
    for (auto it = reversed.rbegin(); it != reversed.rend(); ++it) {
      statements.push_back(*it);
    }
    return SeqStmt::Flatten(statements);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (!deep_definitions_) {
      return StmtMutator::VisitStmt_(op);
    }
    VarSet live_out = live_;
    Stmt then_case = VisitStmt(op->then_case);
    VarSet then_in = live_;
    live_ = live_out;
    Optional<Stmt> else_case = op->else_case;
    if (else_case.defined()) {
      else_case = VisitStmt(else_case.value());
    }
    live_.insert(then_in.begin(), then_in.end());
    Uses{&live_, std::nullopt, true}(op->condition);
    return IfThenElse(op->condition, then_case, else_case, op->span);
  }

  VarSet LoopLiveOut(const Stmt &body, const VarSet &live_out) const {
    VarSet result = live_out;
    VarSet all_uses;
    Uses{&all_uses, std::nullopt, true}(body);
    // A monotone finite identity-set analysis retains a definition used before
    // it on the next iteration, without mutating/discarding that definition.
    size_t remaining = all_uses.size() + live_out.size() + 1;
    while (true) {
      BackwardBindings analysis(result, protected_, emit_unused_effects_, true,
                                eligible_bindings_, true);
      analysis(body);
      size_t previous = result.size();
      result.insert(analysis.LiveIn().begin(), analysis.LiveIn().end());
      if (result.size() == previous) {
        return result;
      }
      ICHECK_GT(remaining--, 0U)
          << "Binding liveness must converge over finite Var identities";
    }
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (!deep_definitions_) {
      return StmtMutator::VisitStmt_(op);
    }
    VarSet live_out = live_;
    live_ = LoopLiveOut(op->body, live_out);
    Stmt body = VisitStmt(op->body);
    live_.insert(live_out.begin(), live_out.end());
    live_.erase(op->loop_var);
    Uses header(&live_, std::nullopt, true);
    header(op->min);
    header(op->extent);
    if (op->step.defined()) {
      header(op->step.value());
    }
    header.IncludeMetadata(op->annotations);
    For result = GetRef<For>(op);
    result.CopyOnWrite()->body = body;
    return result;
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    if (!deep_definitions_) {
      return StmtMutator::VisitStmt_(op);
    }
    VarSet live_out = live_;
    Uses{&live_out, std::nullopt, true}(op->condition);
    live_ = LoopLiveOut(op->body, live_out);
    Stmt body = VisitStmt(op->body);
    live_.insert(live_out.begin(), live_out.end());
    return While(op->condition, body, op->span);
  }

  static bool CanDiscard(const PrimExpr &value) {
    bool discard = SideEffect(value) <= CallEffectKind::kReadState;
    PostOrderVisit(value, [&](const ObjectRef &node) {
      if (const auto *load = node.as<BufferLoadNode>();
          load && load->predicate.defined()) {
        discard &= CanDiscard(load->predicate.value());
      }
    });
    return discard;
  }

  Stmt VisitStmt_(const BindNode *op) final {
    // Match original TA RemoveNoOp::HasSideEffect: pure and ordinary reads
    // may disappear; update/opaque calls (including const_data) may not.
    if (deep_definitions_) {
      Uses{&live_, std::nullopt, true}.IncludeDefinitionMetadata(op->var);
    }
    bool eligible = !eligible_bindings_ || eligible_bindings_->count(op->var);
    if (eligible && !live_.count(op->var) && !protected_.count(op->var)) {
      if (CanDiscard(op->value)) {
        if (analysis_only_) {
          return GetRef<Stmt>(op);
        }
        removed_ = true;
        return Evaluate(0, op->span);
      }
      if (emit_unused_effects_) {
        if (analysis_only_) {
          Uses{&live_, std::nullopt, true}(op->value);
          return GetRef<Stmt>(op);
        }
        removed_ = true;
        Uses{&live_, std::nullopt, deep_definitions_}(op->value);
        return Evaluate(op->value, op->span);
      }
    }
    live_.erase(op->var);
    Uses{&live_, std::nullopt, deep_definitions_}(op->value);
    return GetRef<Stmt>(op);
  }

  std::optional<VarSet> eligible_bindings_;
  bool analysis_only_;
  bool deep_definitions_;
  bool emit_unused_effects_;
  VarSet protected_;
  VarSet live_;
  bool removed_{false};
};
} // namespace tvm::tl::a2a3::binding_cleanup
#endif // TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_BINDING_CLEANUP_H_
