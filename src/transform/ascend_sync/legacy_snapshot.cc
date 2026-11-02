// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

// Snapshot placement follows tile-ai/tilelang commit
// 1c0f402e166a7d0a055c991d2f82d2175a7917b7,
// src/ascend/transform/normalize_control_flow_for_schedule.cc:134-155.
// Adaptation: legacy lexical LetStmt replaces flat tirx::Bind. If/else gains
// one condition binding and two separately anchored optional controls;
// mutable conditions/operands are captured, and mutable For bounds rejected.

#include "legacy_snapshot.h"

#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../common/operation_config.h"

namespace tvm::tl::ascend_sync {

using namespace tir;

namespace {

// Restrict snapshots to ordinary scalar buffer reads and pure arithmetic.
// A call's own effect classification must be pure: an unmodelled read-state
// intrinsic might read a register or memory without exposing a BufferRegion.
class SnapshotValueValidator : public ExprVisitor {
public:
  void Check(const PrimExpr &value, const char *where) {
    ICHECK(value.dtype().lanes() == 1 && !value.dtype().is_handle())
        << "Legacy Ascend snapshot requires a scalar value in " << where;
    ICHECK(SideEffect(value) <= CallEffectKind::kReadState)
        << "Legacy Ascend snapshot cannot evaluate an opaque expression in "
        << where << ": " << value;
    VisitExpr(value);
  }

private:
  void VisitExpr_(const CallNode *op) final {
    const auto *node = op->op.as<OpNode>();
    static auto effects = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
    ICHECK(node != nullptr && effects.count(GetRef<Op>(node)) &&
           effects[GetRef<Op>(node)]->value <=
               static_cast<int>(CallEffectKind::kPure))
        << "Legacy Ascend snapshot requires explicit buffer reads; opaque or "
           "unmodelled state-reading calls are unsupported: "
        << op->op;
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ProducerLoadNode *) final {
    LOG(FATAL) << "Legacy Ascend snapshot requires explicit BufferLoad, not "
                  "ProducerLoad";
  }

  void VisitExpr_(const ReduceNode *) final {
    LOG(FATAL) << "Legacy Ascend snapshot does not support an unlowered Reduce";
  }
};

std::string OperationName(const CallNode *call, size_t *argument_offset) {
  *argument_offset = 0;
  std::string name;
  if (call->op.same_as(builtin::call_extern())) {
    if (call->args.empty() || !call->args[0].as<StringImmNode>())
      return name;
    name = Downcast<StringImm>(call->args[0])->value;
    *argument_offset = 1;
  } else if (const auto *op = call->op.as<OpNode>()) {
    name = op->name;
  }
  if (const size_t begin = name.find('<'); begin != std::string::npos)
    name.erase(begin);
  const std::string prefix = "tl::ascend::";
  if (name.rfind(prefix, 0) == 0)
    name.erase(0, prefix.size());
  return name;
}

class LegacySnapshotNormalizer : public StmtExprMutator {
public:
  LegacySnapshotResult Run(const PrimFunc &function) {
    PrimFunc normalized = function;
    Stmt body = VisitStmt(function->body);
    if (!body.same_as(function->body))
      normalized.CopyOnWrite()->body = std::move(body);
    return {std::move(normalized), std::move(mutable_bindings_)};
  }

private:
  Var SnapshotVar(const PrimExpr &value, const char *where) {
    SnapshotValueValidator().Check(value, where);
    Var var("__ascend_snapshot_" + std::to_string(next_snapshot_++),
            value.dtype());
    if (SideEffect(value) > CallEffectKind::kPure)
      mutable_bindings_.insert(var);
    return var;
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    // Do not recompute a definition, or widen its scope by moving it into the
    // surrounding sequence. Its body alone sees the captured value.
    if (SideEffect(op->value) > CallEffectKind::kPure) {
      SnapshotValueValidator().Check(op->value, "LetStmt definition");
      mutable_bindings_.insert(op->var);
    }
    Stmt body = VisitStmt(op->body);
    return body.same_as(op->body)
               ? GetRef<Stmt>(op)
               : LetStmt(op->var, op->value, std::move(body), op->span);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    Stmt then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case = op->else_case;
    if (else_case.defined())
      else_case = VisitStmt(else_case.value());
    if (else_case.defined()) {
      // Capture even pure conditions once: the two optional controls describe
      // one original branch decision, not two independently evaluated Ifs.
      // Each If has its own occurrence/anchor, so incoming/outgoing events
      // cannot be merged at a shared If/else boundary in reverse order.
      Var condition = SnapshotVar(op->condition, "IfThenElse predicate");
      Stmt first =
          IfThenElse(condition, std::move(then_case), NullOpt, op->span);
      Stmt second =
          IfThenElse(Not(condition), else_case.value(), NullOpt, op->span);
      Stmt branches = SeqStmt({std::move(first), std::move(second)});
      return LetStmt(condition, op->condition, std::move(branches), op->span);
    }
    if (SideEffect(op->condition) <= CallEffectKind::kPure)
      return IfThenElse(op->condition, std::move(then_case), else_case,
                        op->span);
    Var condition = SnapshotVar(op->condition, "IfThenElse predicate");
    Stmt branch =
        IfThenElse(condition, std::move(then_case), else_case, op->span);
    return LetStmt(condition, op->condition, std::move(branch), op->span);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    ICHECK(SideEffect(op->min) <= CallEffectKind::kPure &&
           SideEffect(op->extent) <= CallEffectKind::kPure)
        << "Legacy Ascend snapshot: mutable For bounds are unsupported";
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr)
      return GetRef<Stmt>(op);
    size_t offset = 0;
    const std::string name = OperationName(call, &offset);
    auto configured = GetOperationConfig().find(name);
    if (configured == GetOperationConfig().end())
      return GetRef<Stmt>(op); // The importer diagnoses unknown operations.
    const OperationConfig config = ResolveOperationConfig(call, name, offset);
    std::unordered_set<size_t> buffer_arguments;
    for (const auto &[index, access] : config.buffer_accesses)
      buffer_arguments.insert(offset + index);
    Array<PrimExpr> arguments = call->args;
    std::vector<std::pair<Var, PrimExpr>> bindings;
    for (size_t i = offset; i < arguments.size(); ++i) {
      if (buffer_arguments.count(i)) {
        const auto *pointer = arguments[i].as<CallNode>();
        if (pointer == nullptr ||
            !pointer->op.same_as(builtin::tvm_access_ptr()) ||
            pointer->args.size() != 5)
          continue; // The importer diagnoses malformed pointer ABIs.
        Array<PrimExpr> fields = pointer->args;
        for (size_t field : {2U, 3U}) {
          if (SideEffect(fields[field]) <= CallEffectKind::kPure)
            continue;
          Var snapshot = SnapshotVar(fields[field], "access_ptr offset/extent");
          bindings.emplace_back(snapshot, fields[field]);
          fields.Set(field, snapshot);
        }
        if (!fields.same_as(pointer->args)) {
          Call rewritten = GetRef<Call>(pointer);
          rewritten.CopyOnWrite()->args = std::move(fields);
          arguments.Set(i, std::move(rewritten));
        }
        continue;
      }
      if (SideEffect(arguments[i]) <= CallEffectKind::kPure)
        continue;
      Var snapshot = SnapshotVar(arguments[i], "hardware scalar operand");
      bindings.emplace_back(snapshot, arguments[i]);
      arguments.Set(i, snapshot);
    }
    if (bindings.empty())
      return GetRef<Stmt>(op);
    Call rewritten = GetRef<Call>(call);
    rewritten.CopyOnWrite()->args = std::move(arguments);
    Stmt statement = Evaluate(std::move(rewritten), op->span);
    for (auto it = bindings.rbegin(); it != bindings.rend(); ++it)
      statement =
          LetStmt(it->first, it->second, std::move(statement), op->span);
    return statement;
  }

  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> mutable_bindings_;
  size_t next_snapshot_{0};
};

} // namespace

LegacySnapshotResult NormalizeLegacySnapshots(const PrimFunc &function) {
  return LegacySnapshotNormalizer().Run(function);
}

} // namespace tvm::tl::ascend_sync
