// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/* Restore the pinned TA LetStmt cleanup at its original pipeline phase.
 * TIRX's shared RemoveNoOp intentionally does not eliminate flat Bind nodes.
 */
#include <algorithm>
#include <functional>
#include <unordered_set>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "tirx/transform/remove_no_op.h"

#include "../op/ascend.h"
#include "../op/builtin.h"
#include "common/binding_cleanup.h"

namespace tvm::tl::a2a3 {
namespace {
using namespace tirx;
using namespace ffi;
using binding_cleanup::VarSet;

bool IsEnabled(const Any &value) {
  if (auto boolean = value.as<bool>()) {
    return boolean.value();
  }
  if (auto integer = value.as<int64_t>()) {
    return integer.value() != 0;
  }
  if (auto object = value.as<ObjectRef>()) {
    if (const auto *integer = object.value().as<IntImmNode>()) {
      return integer->value != 0;
    }
  }
  // Unknown metadata must not authorize removing a volatile read.
  return true;
}

bool HasVolatileMetadata(const Any &value) {
  std::unordered_set<ObjectRef, ObjectPtrHash, ObjectPtrEqual> visited;
  std::function<bool(const Any &)> inspect = [&](const Any &item) -> bool {
    auto object = item.as<ObjectRef>();
    if (!object || !object->defined() || !visited.insert(*object).second) {
      return false;
    }
    if (auto map = object->as<MapObj>()) {
      for (const auto &entry : *map) {
        if (auto key = entry.first.as<String>()) {
          if ((key.value() == "volatile" || key.value() == "volatile_scope") &&
              IsEnabled(entry.second)) {
            return true;
          }
        }
        if (inspect(entry.first) || inspect(entry.second)) {
          return true;
        }
      }
      return false;
    }
    if (auto array = object->as<ArrayObj>()) {
      for (const Any &entry : *array) {
        if (inspect(entry)) {
          return true;
        }
      }
      return false;
    }
    bool result = false;
    reflection::ForEachFieldInfo(
        TVMFFIGetTypeInfo((*object)->type_index()),
        [&](const TVMFFIFieldInfo *field) {
          if (!result) {
            result = inspect(reflection::FieldGetter(field)(*object));
          }
        });
    return result;
  };
  return inspect(value);
}

// Only ascend_src_code has a documented implicit-local-name payload. Ordinary
// extern calls use their explicit operands and registered effects; their names
// are not a reason to keep every binding in the enclosing function.
class ProtectedBindings : public StmtExprVisitor {
public:
  static VarSet Collect(const PrimFunc &func) {
    ProtectedBindings visitor;
    visitor.volatile_scope_ = HasVolatileMetadata(func->attrs);
    visitor(func->body);
    return visitor.protected_;
  }

private:
  void VisitStmt_(const SeqStmtNode *op) final {
    for (const Stmt &stmt : op->seq) {
      VisitStmt(stmt);
    }
  }

  class PredicateEffects : public ExprVisitor {
  public:
    static bool HasEffect(const PrimExpr &expr) {
      PredicateEffects visitor;
      visitor.has_effect_ = SideEffect(expr) > CallEffectKind::kReadState;
      visitor(expr);
      return visitor.has_effect_;
    }

  private:
    void VisitExpr_(const BufferLoadNode *op) final {
      ExprVisitor::VisitExpr_(op);
      if (op->predicate.defined()) {
        has_effect_ |=
            SideEffect(op->predicate.value()) > CallEffectKind::kReadState;
        VisitExpr(op->predicate.value());
      }
    }
    bool has_effect_{false};
  };

  void VisitStmt_(const BindNode *op) final {
    Optional<Var> saved = current_binding_;
    current_binding_ = op->var;
    VisitExpr(op->value);
    current_binding_ = std::move(saved);
    visible_.insert(op->var);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (current_binding_.defined() &&
        (volatile_scope_ || volatile_buffers_.count(op->buffer->data) ||
         HasVolatileMetadata(op->buffer) ||
         (op->predicate.defined() &&
          PredicateEffects::HasEffect(op->predicate.value())))) {
      // Shared RemoveNoOp does not inspect a BufferLoad predicate when
      // classifying Evaluate(value). Keep this binding so its predicate's
      // observable effects cannot disappear on the next cleanup iteration.
      protected_.insert(current_binding_.value());
    }
    StmtExprVisitor::VisitExpr_(op);
    if (op->predicate.defined()) {
      VisitExpr(op->predicate.value());
    }
  }

  void VisitExpr_(const CallNode *op) final {
    if (current_binding_.defined() && HasVolatileMetadata(op->annotations)) {
      protected_.insert(current_binding_.value());
    }
    if (op->op.same_as(ascend_src_code())) {
      protected_.insert(visible_.begin(), visible_.end());
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    VarSet saved_visible = visible_;
    VarSet saved_buffers = volatile_buffers_;
    bool saved_volatile = volatile_scope_;
    if (op->attr_key == "volatile_scope" && !is_zero(op->value)) {
      volatile_scope_ = true;
      if (auto var = op->node.as<Var>()) {
        volatile_buffers_.insert(var.value());
      }
    }
    VisitExpr(op->value);
    VisitStmt(op->body);
    visible_ = std::move(saved_visible);
    volatile_buffers_ = std::move(saved_buffers);
    volatile_scope_ = saved_volatile;
  }

  void VisitStmt_(const ForNode *op) final {
    VarSet saved_visible = visible_;
    VarSet saved_buffers = volatile_buffers_;
    bool saved_volatile = volatile_scope_;
    volatile_scope_ |= HasVolatileMetadata(op->annotations);
    StmtExprVisitor::VisitStmt_(op);
    visible_ = std::move(saved_visible);
    volatile_buffers_ = std::move(saved_buffers);
    volatile_scope_ = saved_volatile;
  }

  void VisitStmt_(const SBlockNode *op) final {
    VarSet saved_visible = visible_;
    VarSet saved_buffers = volatile_buffers_;
    bool saved_volatile = volatile_scope_;
    volatile_scope_ |= HasVolatileMetadata(op->annotations);
    StmtExprVisitor::VisitStmt_(op);
    visible_ = std::move(saved_visible);
    volatile_buffers_ = std::move(saved_buffers);
    volatile_scope_ = saved_volatile;
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    VarSet saved_visible = visible_;
    VarSet saved_buffers = volatile_buffers_;
    VisitExpr(op->condition);
    VisitStmt(op->then_case);
    visible_ = saved_visible;
    volatile_buffers_ = saved_buffers;
    if (op->else_case.defined()) {
      VisitStmt(op->else_case.value());
    }
    visible_ = std::move(saved_visible);
    volatile_buffers_ = std::move(saved_buffers);
  }

  void VisitStmt_(const WhileNode *op) final {
    VarSet saved_visible = visible_;
    VarSet saved_buffers = volatile_buffers_;
    StmtExprVisitor::VisitStmt_(op);
    visible_ = std::move(saved_visible);
    volatile_buffers_ = std::move(saved_buffers);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    if (HasVolatileMetadata(op->annotations)) {
      volatile_buffers_.insert(op->buffer->data);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  Optional<Var> current_binding_;
  bool volatile_scope_{false};
  VarSet volatile_buffers_;
  VarSet visible_;
  VarSet protected_;
};

size_t CountBindings(const Stmt &body) {
  class Counter : public StmtVisitor {
  public:
    size_t count{0};
    void VisitStmt_(const BindNode *op) final { ++count; }
  } visitor;
  visitor(body);
  return visitor.count;
}

std::pair<PrimFunc, bool> PruneBindingsOnce(PrimFunc func) {
  VarSet roots;
  binding_cleanup::Uses uses(&roots, std::nullopt, true);
  uses.IncludeMetadata(func->attrs);
  uses.IncludeMetadata(func->buffer_map);
  for (const Var &param : func->params) {
    uses.IncludeDefinitionMetadata(param);
  }
  VarSet protected_bindings = ProtectedBindings::Collect(func);
  binding_cleanup::BackwardBindings pruner(roots, protected_bindings, true,
                                           true);
  Stmt body = pruner(func->body);
  func.CopyOnWrite()->body = body;
  return {func, pruner.RemovedBinding()};
}

} // namespace

binding_cleanup::VarSet
binding_cleanup::CollectProtectedBindings(const tirx::PrimFunc &func) {
  return ProtectedBindings::Collect(func);
}

tvm::transform::Pass RemoveNoOp() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext ctx) {
    size_t previous_count = 0;
    for (const auto &entry : mod->functions) {
      if (const auto *func = entry.second.as<tirx::PrimFuncNode>()) {
        previous_count += CountBindings(func->body);
      }
    }
    while (true) {
      // Call the genuine shared pass, not its unconfigured Stmt helper. This
      // preserves dataflow-analysis and max-simplification-step settings.
      mod = tirx::transform::RemoveNoOp()(mod);
      Map<GlobalVar, BaseFunc> updates;
      bool removed = false;
      size_t count = 0;
      for (const auto &entry : mod->functions) {
        if (const auto *func = entry.second.as<tirx::PrimFuncNode>()) {
          auto result = PruneBindingsOnce(GetRef<tirx::PrimFunc>(func));
          removed |= result.second;
          count += CountBindings(result.first->body);
          updates.Set(entry.first, result.first);
        }
      }
      mod.CopyOnWrite()->Update(IRModule(updates));
      if (!removed) {
        return mod;
      }
      ICHECK_LT(count, previous_count)
          << "A2/A3 cleanup must remove a statement binding occurrence per "
             "round";
      previous_count = count;
    }
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.A2A3RemoveNoOp",
                                          {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.A2A3RemoveNoOp", RemoveNoOp);
}
} // namespace tvm::tl::a2a3
