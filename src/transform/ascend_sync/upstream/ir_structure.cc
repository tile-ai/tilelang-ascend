/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// Ported from tile-ai/tilelang 1c0f402e166a7d0a055c991d2f82d2175a7917b7.
// Source: src/ascend/transform/auto_schedule/ir_structure.{h,cc}.
// Adaptation: legacy TIR/runtime containers and implicit unit-step For loops;
// leaf effects are supplied by the legacy importer, without tile-op analysis.

#include "ir_structure.h"

#include <tvm/node/structural_equal.h>
#include <tvm/tir/builtin.h>

#include <sstream>

namespace tvm::tl::ascend_sync {

using namespace tir;

namespace {

bool GuardsStructurallyEqual(const PrimExpr &a, const PrimExpr &b) {
  ICHECK(a.defined() && b.defined());
  return a.same_as(b) || StructuralEqual()(a, b);
}

bool GuardsEquivalentWithAnalyzer(ProofAnalyzer *analyzer, const PrimExpr &a,
                                  const PrimExpr &b) {
  ICHECK(a.defined() && b.defined());
  if (GuardsStructurallyEqual(a, b))
    return true;
  return analyzer->CanProveEqual(a, b);
}

bool GuardImpliesWithAnalyzer(ProofAnalyzer *analyzer, const PrimExpr &premise,
                              const PrimExpr &conclusion) {
  ICHECK(premise.defined() && conclusion.defined());
  if (GuardsStructurallyEqual(premise, conclusion))
    return true;
  auto exit_constraint = analyzer->EnterConstraint(premise);
  bool result = analyzer->CanProve(conclusion);
  exit_constraint();
  return result;
}

} // namespace

bool GuardsEquivalent(const PrimExpr &a, const PrimExpr &b,
                      const ConstrSet &outer_ctx) {
  ProofAnalyzer analyzer;
  outer_ctx.Populate(analyzer);
  return GuardsEquivalentWithAnalyzer(&analyzer, a, b);
}

bool GuardImplies(const PrimExpr &premise, const PrimExpr &conclusion,
                  const ConstrSet &outer_ctx) {
  if (is_zero(premise) || is_one(conclusion) ||
      GuardsStructurallyEqual(premise, conclusion))
    return true;
  ProofAnalyzer analyzer;
  outer_ctx.Populate(analyzer);
  return GuardImpliesWithAnalyzer(&analyzer, premise, conclusion);
}

PrimExpr IRStructure::GetConditionGuard() const {
  PrimExpr guard = Bool(true);
  for (const auto &item : GetGuards()) {
    if (!item->IsCondition())
      continue;
    const PrimExpr &condition =
        static_cast<const ConditionGuard *>(item.get())->condition;
    guard = is_one(guard) ? condition : guard && condition;
  }
  return guard;
}

bool IRStructure::GuardsTouchStorage(const Var &storage) const {
  class StorageTouchVisitor : public ExprVisitor {
  public:
    explicit StorageTouchVisitor(Var storage) : storage_(std::move(storage)) {}
    bool touched{false};

  private:
    void VisitExpr_(const VarNode *op) final {
      touched |= storage_.same_as(GetRef<Var>(op));
    }
    void VisitExpr_(const BufferLoadNode *op) final {
      touched |= op->buffer->data.same_as(storage_);
      ExprVisitor::VisitExpr_(op);
    }
    void VisitExpr_(const ProducerLoadNode *op) final {
      // ProducerLoad has no reliable legacy storage-Var association here.
      touched = true;
      ExprVisitor::VisitExpr_(op);
    }
    void VisitExpr_(const CallNode *op) final {
      // An opaque guard call can inspect storage without an access_ptr ABI.
      if (SideEffect(GetRef<PrimExpr>(op)) > CallEffectKind::kPure)
        touched = true;
      ExprVisitor::VisitExpr_(op);
    }
    Var storage_;
  };

  auto touches_storage = [&](const PrimExpr &expression) {
    if (!expression.defined())
      return false;
    StorageTouchVisitor visitor(storage);
    visitor(expression);
    return visitor.touched;
  };
  for (const auto &guard : GetGuards()) {
    if (guard->IsCondition()) {
      const auto *condition = static_cast<const ConditionGuard *>(guard.get());
      if (touches_storage(condition->condition))
        return true;
      continue;
    }
    const auto *attribute = static_cast<const AttributeGuard *>(guard.get());
    if (const auto *expression = attribute->node.as<PrimExprNode>()) {
      if (touches_storage(GetRef<PrimExpr>(expression)))
        return true;
    }
  }
  return false;
}

Stmt WrapWithGuard(Stmt stmt, const PrimExpr &guard) {
  ICHECK(guard.defined());
  return is_one(guard) ? stmt : IfThenElse(guard, std::move(stmt));
}

ConstrSet TaskNode::GetContextBeforeConditionGuards() const {
  ConstrSet result = outer_ctx;
  const GuardList &guards = GetGuards();
  for (auto guard_it = guards.rbegin(); guard_it != guards.rend(); ++guard_it) {
    if (!(*guard_it)->IsCondition())
      continue;
    const PrimExpr &condition =
        static_cast<const ConditionGuard *>(guard_it->get())->condition;
    bool removed = false;
    for (size_t index = result.constrs_.size(); index > 0; --index) {
      const Constr &constraint = result.constrs_[index - 1];
      if (constraint.kind == Constr::kConstr &&
          StructuralEqual()(constraint.value, condition)) {
        result.constrs_.erase(result.constrs_.begin() + index - 1);
        removed = true;
        break;
      }
    }
    ICHECK(removed) << "Task condition is missing from its constraint context: "
                    << condition;
  }
  return result;
}

void ControlNode::InitializeLoopBodyContext() {
  ICHECK(task != nullptr);
  ConstrSet result = task->outer_ctx;
  result.AddConstr(control->loop_var,
                   Range::FromMinExtent(control->min, control->extent));
  result.AddConstr(control->extent > 0);
  // Unlike an upstream flat Bind, a legacy LetStmt only defines its variable
  // inside its body. The importer records these bindings in each dominated
  // task's outer_ctx; hoisting child LetStmt bindings here would assert them
  // outside their lexical lifetime and could incorrectly remove dependencies.
  loop_body_ctx_ = std::move(result);
}

IRStructure *IRStructure::GetLowestCommonAncestor(IRStructure *other) {
  ICHECK(other != nullptr) << "Cannot find an ancestor of a null IRStructure";
  std::unordered_set<IRStructure *> ancestors;
  for (IRStructure *node = this; node != nullptr; node = node->GetParent()) {
    ancestors.insert(node);
  }
  for (IRStructure *node = other; node != nullptr; node = node->GetParent()) {
    if (ancestors.count(node))
      return node;
  }
  return nullptr;
}

std::vector<IRStructure *> IRStructure::PathFrom(IRStructure *ancestor) {
  std::vector<IRStructure *> path;
  for (IRStructure *node = this; node != nullptr; node = node->GetParent()) {
    path.push_back(node);
    if (node == ancestor) {
      std::reverse(path.begin(), path.end());
      return path;
    }
  }
  if (ancestor == nullptr) {
    std::reverse(path.begin(), path.end());
    return path;
  }
  LOG(FATAL) << "The requested IRStructure path start is not an ancestor";
  return {};
}

IRStructure *IRStructure::GetChildOnPathFrom(IRStructure *ancestor) {
  ICHECK(ancestor != nullptr) << "IRStructure path ancestor cannot be null";
  IRStructure *node = this;
  while (node != nullptr && node->GetParent() != ancestor)
    node = node->GetParent();
  ICHECK(node != nullptr)
      << "The requested IRStructure path start is not a strict ancestor";
  return node;
}

int CompareIRStructure(IRStructure *lhs, IRStructure *rhs) {
  if (lhs == rhs)
    return 0;
  if (lhs == nullptr)
    return -2;
  if (rhs == nullptr)
    return 2;

  std::vector<IRStructure *> lhs_path = lhs->PathFrom();
  std::vector<IRStructure *> rhs_path = rhs->PathFrom();
  size_t common_size = std::min(lhs_path.size(), rhs_path.size());
  bool same_nodes = true;
  for (size_t i = 0; i < common_size; ++i) {
    same_nodes &= lhs_path[i] == rhs_path[i];
    if (lhs_path[i]->GetStage() != rhs_path[i]->GetStage())
      return lhs_path[i]->GetStage() < rhs_path[i]->GetStage() ? -1 : 1;
    if (lhs_path[i]->GetIndex() != rhs_path[i]->GetIndex())
      return lhs_path[i]->GetIndex() < rhs_path[i]->GetIndex() ? -1 : 1;
  }
  if (lhs_path.size() == rhs_path.size())
    return 0;
  if (lhs_path.size() < rhs_path.size())
    return same_nodes ? -2 : -1;
  return same_nodes ? 2 : 1;
}

PrimExpr CalculateIterationCount(ControlNode *loop) {
  PrimExpr total_iter = IntImm(DataType::Int(32), 0);
  PrimExpr total_multiplier = IntImm(DataType::Int(32), 1);
  for (; loop != nullptr; loop = loop->GetParentControl()) {
    const ForNode *for_node = loop->control.get();
    PrimExpr step = IntImm(for_node->loop_var.dtype(), 1);
    total_iter =
        total_iter +
        indexdiv(for_node->loop_var - for_node->min, step) * total_multiplier;
    total_multiplier = total_multiplier * ceildiv(for_node->extent, step);
  }
  return total_iter;
}

std::shared_ptr<IRStructure> TaskNode::Clone() const {
  auto new_task = std::make_shared<TaskNode>(CloneExtraInfo());
  // Copy statement
  new_task->stmt = stmt;
  // Copy resource pipe mask
  new_task->SetPipeMask(GetPipeMask());
  new_task->SetHbmMask(GetHbmMask());
  new_task->SetSpecialWriteMask(GetSpecialWriteMask());
  new_task->SetReadsPadValue(ReadsPadValueRegister());
  new_task->SetReadsAtomic(ReadsAtomicRegister());
  if (IsPerCoreTask())
    new_task->MarkPerCoreTask();
  // Copy memory access regions
  new_task->SetReadRegions(GetReadRegions());
  new_task->SetWriteRegions(GetWriteRegions());
  new_task->SetL0Pitches(GetL0Pitches());
  new_task->SetReadVars(GetReadVars());
  new_task->SetWriteVars(GetWriteVars());
  new_task->outer_ctx = outer_ctx;
  new_task->SetCoreMask(GetCoreMask());
  new_task->CopyBaseMetadataFrom(*this);
  return new_task;
}

std::shared_ptr<IRStructure> ControlNode::Clone() const {
  auto new_ctrl = std::make_shared<ControlNode>(CloneExtraInfo());
  // Copy For control (For is a TVM object with reference counting)
  new_ctrl->control = control;
  // Clone children
  new_ctrl->children.reserve(children.size());
  for (size_t i = 0; i < children.size(); ++i) {
    std::shared_ptr<IRStructure> child =
        children[i] ? children[i]->Clone() : nullptr;
    if (child) {
      child->SetIndex(i);
      child->SetParent(new_ctrl.get());
    }
    new_ctrl->children.push_back(std::move(child));
  }
  if (task) {
    new_ctrl->task = std::static_pointer_cast<TaskNode>(task->Clone());
    new_ctrl->task->SetParent(new_ctrl.get());
  }
  new_ctrl->loop_body_ctx_ = loop_body_ctx_;
  new_ctrl->CopyBaseMetadataFrom(*this);
  return new_ctrl;
}

void CollectAllTaskNodes(IRStructure *node,
                         std::vector<TaskNode *> &all_tasks) {
  if (!node)
    return;
  if (node->IsTask()) {
    all_tasks.push_back(static_cast<TaskNode *>(node));
    return;
  }
  ICHECK(node->IsControl());
  auto *control = static_cast<ControlNode *>(node);
  for (const auto &child : control->children)
    CollectAllTaskNodes(child.get(), all_tasks);
}

// Original helper function to print IRStructure (kept for backward
// compatibility)
void PrintIRStructure(const IRStructure *node, int indent) {
  if (!node)
    return;

  std::string indent_str(indent * 2, ' ');

  for (const auto &guard : node->GetGuards()) {
    if (guard->IsAttribute()) {
      const auto *attribute_guard =
          static_cast<const AttributeGuard *>(guard.get());
      LOG(INFO) << indent_str << "  [attr " << attribute_guard->key << "] "
                << attribute_guard->node;
    } else {
      const auto *condition_guard =
          static_cast<const ConditionGuard *>(guard.get());
      LOG(INFO) << indent_str << "  [guard] " << condition_guard->condition;
    }
  }

  if (node->IsTask()) {
    const TaskNode *task = static_cast<const TaskNode *>(node);
    LOG(INFO) << indent_str << "TaskNode:";
    LOG(INFO) << indent_str << "  stmt: " << task->stmt;
    for (auto &region : task->GetReadRegions()) {
      LOG(INFO) << indent_str << "  Read Region: " << region;
    }
    for (auto &region : task->GetWriteRegions()) {
      LOG(INFO) << indent_str << "  Write Region: " << region;
    }
    for (auto &var : task->GetReadVars()) {
      LOG(INFO) << indent_str << "  Read Var: " << var;
    }
    for (auto &var : task->GetWriteVars()) {
      LOG(INFO) << indent_str << "  Write Var: " << var;
    }
    LOG(INFO) << indent_str << "  pipe_mask: 0x" << std::hex
              << task->GetPipeMask() << std::dec;
    LOG(INFO) << indent_str << "  latency: " << task->GetLatency() << " cycles";
    LOG(INFO) << indent_str << "  II: " << task->GetII() << " cycles";
    LOG(INFO) << indent_str
              << "  core_mask: " << static_cast<int>(task->GetCoreMask());
  } else if (node->IsControl()) {
    const ControlNode *control = static_cast<const ControlNode *>(node);
    LOG(INFO) << indent_str << "ControlNode (For loop):";
    LOG(INFO) << indent_str << "  Loop body: " << control->children.size()
              << " children";
    for (size_t i = 0; i < control->children.size(); i++) {
      LOG(INFO) << indent_str << "  Child " << i << ":";
      PrintIRStructure(control->children[i].get(), indent + 2);
    }
  }
}

} // namespace tvm::tl::ascend_sync
