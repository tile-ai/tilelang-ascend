// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#include "legacy_manual_events.h"

#include <tvm/ir/op.h>
#include <tvm/node/structural_equal.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../common/operation_config.h"
#include "legacy_protocol_context.h"

namespace tvm::tl::ascend_sync {

using namespace tir;

namespace {

constexpr const char *kDiagnostic = "Legacy manual event normalization: ";
constexpr size_t kExecutionBudget = 65536;
constexpr size_t kStableConditionLimit = 8;

std::string OperationName(const CallNode *call) {
  if (const auto *operation = call->op.as<OpNode>())
    return operation->name;
  return "";
}

bool IsExplicitFlag(const CallNode *call) {
  const std::string name = OperationName(call);
  return name == "tl.ascend_set_flag" || name == "tl.ascend_wait_flag";
}

using Tokens = std::unordered_set<std::string>;

// The map also lets the executor skip computation-only loops. Their iteration
// counts and conditions cannot affect the token protocol.
class FlagDiscovery : public StmtExprVisitor {
public:
  void Run(const Stmt &body) { VisitStmt(body); }
  bool NeedsNormalization() const { return needs_normalization_; }
  bool HasFlags(const Stmt &statement) const {
    const auto found = counts_.find(statement.get());
    return found != counts_.end() && found->second != 0;
  }
  bool HasAutomaticFlags() const { return automatic_; }
  bool HasCrossFlags() const { return cross_; }

private:
  void VisitStmt(const Stmt &statement) final {
    const size_t previous = count_;
    StmtExprVisitor::VisitStmt(statement);
    counts_[statement.get()] = count_ - previous;
  }

  void VisitStmt_(const ForNode *op) final {
    ++control_depth_;
    StmtExprVisitor::VisitStmt_(op);
    --control_depth_;
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    ++control_depth_;
    StmtExprVisitor::VisitStmt_(op);
    --control_depth_;
  }

  void VisitStmt_(const EvaluateNode *op) final {
    const CallNode *previous = standalone_;
    standalone_ = op->value.as<CallNode>();
    StmtExprVisitor::VisitStmt_(op);
    standalone_ = previous;
  }

  void VisitExpr_(const CallNode *op) final {
    const std::string name = OperationName(op);
    if (IsExplicitFlag(op)) {
      ICHECK(op == standalone_)
          << kDiagnostic << "hard flags must be standalone Evaluate calls";
      ++count_;
      ICHECK_EQ(op->args.size(), 3U) << kDiagnostic << "malformed hard flag";
      needs_normalization_ |=
          control_depth_ > 0 || !op->args[2].as<IntImmNode>();
    }
    automatic_ |=
        name == "tl.ascend_auto_set_flag" || name == "tl.ascend_auto_wait_flag";
    cross_ |= name == "tl.ascend_set_cross_flag" ||
              name == "tl.ascend_wait_cross_flag" ||
              name == "tl.ascend_auto_set_cross_flag" ||
              name == "tl.ascend_auto_wait_cross_flag";
    StmtExprVisitor::VisitExpr_(op);
  }

  std::unordered_map<const StmtNode *, size_t> counts_;
  size_t count_{0};
  int control_depth_{0};
  bool needs_normalization_{false};
  bool automatic_{false};
  bool cross_{false};
  const CallNode *standalone_{nullptr};
};

// Correlation is valid only for immutable SSA values. A reused binder could
// otherwise give the same Var identity two different values in disjoint
// lexical scopes, making structural equality alone an invalid stability proof.
class RepeatedBinders : public StmtExprVisitor {
public:
  void Run(const PrimFunc &function) {
    for (const Var &parameter : function->params)
      Record(parameter);
    VisitStmt(function->body);
  }

  bool ContainsRepeatedFreeVariable(const PrimExpr &value) const {
    for (const Var &variable : UndefinedVars(value)) {
      const auto found = counts_.find(variable);
      if (found != counts_.end() && found->second > 1)
        return true;
    }
    return false;
  }

private:
  void Record(const Var &variable) { ++counts_[variable]; }

  void VisitStmt_(const LetStmtNode *op) final {
    Record(op->var);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    Record(op->loop_var);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "thread_extent" || op->attr_key == "virtual_thread") {
      if (const auto *iteration = op->node.as<IterVarNode>())
        Record(iteration->var);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  std::unordered_map<Var, size_t, ObjectPtrHash, ObjectPtrEqual> counts_;
};

class StableConditions : public StmtExprVisitor {
public:
  StableConditions(const FlagDiscovery &discovery,
                   const RepeatedBinders &binders)
      : discovery_(discovery), binders_(binders) {}

  std::vector<PrimExpr> Run(const Stmt &body) {
    VisitStmt(body);
    return std::move(conditions_);
  }

private:
  void VisitStmt(const Stmt &statement) final {
    if (discovery_.HasFlags(statement))
      StmtExprVisitor::VisitStmt(statement);
  }

  void VisitStmt_(const ForNode *op) final {
    ++loop_depth_;
    StmtExprVisitor::VisitStmt_(op);
    --loop_depth_;
  }

  void VisitStmt_(const LetStmtNode *op) final {
    const auto binding = context_.Bind(op->var, context_.Evaluate(op->value));
    VisitStmt(op->body);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    if (loop_depth_ == 0 &&
        SideEffect(op->condition) <= CallEffectKind::kPure &&
        !binders_.ContainsRepeatedFreeVariable(op->condition) &&
        !context_.Evaluate(op->condition).has_value()) {
      bool present = false;
      for (const PrimExpr &condition : conditions_)
        present |= StructuralEqual()(condition, op->condition);
      if (!present) {
        ICHECK_LT(conditions_.size(), kStableConditionLimit)
            << kDiagnostic << "stable condition path budget exceeded";
        conditions_.push_back(op->condition);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  const FlagDiscovery &discovery_;
  const RepeatedBinders &binders_;
  LegacyProtocolContext context_;
  std::vector<PrimExpr> conditions_;
  size_t loop_depth_{0};
};

class TokenExecutor {
public:
  TokenExecutor(const FlagDiscovery &discovery,
                const std::vector<PrimExpr> &stable_conditions,
                uint32_t choices, size_t *actions, bool cross_core)
      : discovery_(discovery), stable_conditions_(stable_conditions),
        choices_(choices), actions_(actions), cross_core_(cross_core) {}

  void Run(const Stmt &body) {
    Execute(body);
    ICHECK(tokens_.empty())
        << kDiagnostic
        << "unmatched SetFlag at kernel exit; all tokens must "
           "be drained";
  }

  int RequiredOwner() const { return required_owner_; }

private:
  void ConsumeBudget() {
    ICHECK_LT((*actions_)++, kExecutionBudget)
        << kDiagnostic << "static protocol execution budget exceeded";
  }

  void ExecuteLoopBody(const Stmt &body) {
    ++loop_depth_;
    Execute(body);
    --loop_depth_;
  }

  void RequireOwner(int owner) {
    if (owner < 0)
      return;
    if (cross_core_)
      return;
    ICHECK(required_owner_ < 0 || required_owner_ == owner)
        << kDiagnostic << "hard events require conflicting execution owners";
    required_owner_ = owner;
  }

  static std::string Pipe(const PrimExpr &expression) {
    const auto *literal = expression.as<StringImmNode>();
    ICHECK(literal != nullptr)
        << kDiagnostic << "hard-event pipe must be a literal";
    std::string pipe = literal->value;
    if (pipe.rfind("PIPE_", 0) == 0)
      pipe.erase(0, 5);
    static const std::unordered_set<std::string> pipes{
        "S", "V", "M", "MTE1", "MTE2", "MTE3", "FIX"};
    ICHECK(pipes.count(pipe))
        << kDiagnostic << "unsupported hard-event pipe " << pipe;
    return pipe;
  }

  void Execute(const Stmt &statement) {
    if (!discovery_.HasFlags(statement))
      return;
    ConsumeBudget();
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      for (const Stmt &child : sequence->seq)
        Execute(child);
    } else if (const auto *binding = statement.as<LetStmtNode>()) {
      const auto scope =
          context_.Bind(binding->var, context_.Evaluate(binding->value));
      Execute(binding->body);
    } else if (const auto *attribute = statement.as<AttrStmtNode>()) {
      if (attribute->attr_key == "resource_scope") {
        const auto *owner = attribute->value.as<IntImmNode>();
        ICHECK(owner != nullptr && (owner->value == 0 || owner->value == 1))
            << kDiagnostic << "requires a literal Cube or Vector owner";
        const auto scope =
            context_.EnterOwner(static_cast<LegacyProtocolOwner>(owner->value));
        Execute(attribute->body);
      } else {
        Execute(attribute->body);
      }
    } else if (const auto *allocation = statement.as<AllocateNode>()) {
      const auto condition = context_.Evaluate(allocation->condition);
      ICHECK(condition.has_value() && condition.value())
          << kDiagnostic << "conditional allocation around a protocol";
      Execute(allocation->body);
    } else if (const auto *declaration = statement.as<DeclBufferNode>()) {
      Execute(declaration->body);
    } else if (const auto *loop = statement.as<ForNode>()) {
      ICHECK(loop->kind == ForKind::kSerial || loop->kind == ForKind::kUnrolled)
          << kDiagnostic << "requires serial or unrolled protocol loops";
      ICHECK(SideEffect(loop->min) <= CallEffectKind::kPure &&
             SideEffect(loop->extent) <= CallEffectKind::kPure)
          << kDiagnostic << "mutable protocol loop bounds are unsupported";
      const auto minimum = context_.Evaluate(loop->min);
      const auto extent = context_.Evaluate(loop->extent);
      auto binding = context_.Bind(loop->loop_var, std::nullopt);
      if (minimum.has_value() && extent.has_value()) {
        ICHECK_GE(extent.value(), 0)
            << kDiagnostic << "negative protocol loop extent";
        ICHECK_LE(static_cast<uint64_t>(extent.value()), kExecutionBudget)
            << kDiagnostic << "static protocol execution budget exceeded";
        const int bits = loop->loop_var.dtype().bits();
        const __int128 end = __int128(minimum.value()) + extent.value();
        ICHECK(loop->loop_var.dtype().is_scalar() &&
               loop->loop_var.dtype().is_int() && bits <= 64 &&
               end >= -(__int128(1) << (bits - 1)) &&
               end <= (__int128(1) << (bits - 1)) - 1)
            << kDiagnostic << "nonrepresentable protocol loop endpoint";
        for (int64_t iteration = 0; iteration < extent.value(); ++iteration) {
          ConsumeBudget();
          const __int128 index = __int128(minimum.value()) + iteration;
          ICHECK(loop->loop_var.dtype().is_scalar() &&
                 loop->loop_var.dtype().is_int() && bits <= 64 &&
                 index >= -(__int128(1) << (bits - 1)) &&
                 index <= (__int128(1) << (bits - 1)) - 1)
              << kDiagnostic << "nonrepresentable protocol loop index";
          binding.Set(static_cast<int64_t>(index));
          ExecuteLoopBody(loop->body);
        }
      } else {
        // An invariant token set proves every iteration (and zero iterations)
        // without assuming a particular dynamic trip count. Unknown ID/branch
        // expressions still fail conservatively inside Execute.
        const Tokens before = tokens_;
        ExecuteLoopBody(loop->body);
        ICHECK(tokens_ == before)
            << kDiagnostic
            << "dynamic protocol loop must preserve its token "
               "state for every iteration";
      }
    } else if (const auto *branch = statement.as<IfThenElseNode>()) {
      ICHECK(SideEffect(branch->condition) <= CallEffectKind::kReadState)
          << kDiagnostic << "opaque protocol branch condition";
      auto condition = context_.Evaluate(branch->condition);
      if (!condition.has_value() && loop_depth_ == 0) {
        for (size_t index = 0; index < stable_conditions_.size(); ++index) {
          if (StructuralEqual()(stable_conditions_[index], branch->condition)) {
            condition = (choices_ >> index) & 1U;
            break;
          }
        }
      }
      if (condition.has_value()) {
        if (condition.value())
          Execute(branch->then_case);
        else if (branch->else_case.defined())
          Execute(branch->else_case.value());
      } else {
        const Tokens before = tokens_;
        Execute(branch->then_case);
        const Tokens then_state = tokens_;
        tokens_ = before;
        if (branch->else_case.defined())
          Execute(branch->else_case.value());
        ICHECK(tokens_ == then_state)
            << kDiagnostic
            << "unresolved branches must have identical token "
               "states";
      }
    } else if (const auto *evaluate = statement.as<EvaluateNode>()) {
      const auto *call = evaluate->value.as<CallNode>();
      ICHECK(call != nullptr && IsExplicitFlag(call))
          << kDiagnostic << "hard flags must be standalone Evaluate calls";
      const auto id = context_.Evaluate(call->args[2]);
      ICHECK(id.has_value() && id.value() >= 0 && id.value() < 8)
          << kDiagnostic
          << "requires a statically evaluable event ID in "
             "[0, 8) without integer overflow";
      const std::string source = Pipe(call->args[0]);
      const std::string destination = Pipe(call->args[1]);
      const std::string event = source + "_" + destination;
      bool supported = false;
      for (const auto &[mapping, name] : GetEventMapping())
        supported |= name == event;
      ICHECK(supported) << kDiagnostic << "unsupported hardware event "
                        << event;
      const auto cube_pipe = [](const std::string &pipe) {
        return pipe == "M" || pipe == "MTE1" || pipe == "FIX";
      };
      const int resource = static_cast<int>(context_.Owner());
      ICHECK((resource != 0 || (source != "V" && destination != "V")) &&
             (resource != 1 || (!cube_pipe(source) && !cube_pipe(destination))))
          << kDiagnostic << "hard-event pipe disagrees with execution owner";
      ICHECK(!cross_core_ || resource >= 0)
          << kDiagnostic
          << "cross-core publication requires an explicit hard-event owner";
      RequireOwner(resource);
      for (const std::string &pipe : {source, destination}) {
        const int owner = pipe == "V" ? 1 : (cube_pipe(pipe) ? 0 : -1);
        RequireOwner(owner);
      }
      // The old single-owner profile unifies inferred and explicit owners.
      // Epoch candidates require explicit C/V banks before local removal.
      const std::string key =
          (cross_core_ ? std::to_string(resource) + ":" : "") + event + ":" +
          std::to_string(id.value());
      if (OperationName(call) == "tl.ascend_set_flag") {
        ICHECK(tokens_.insert(key).second)
            << kDiagnostic << "duplicate SetFlag without WaitFlag: " << key;
      } else {
        ICHECK_EQ(tokens_.erase(key), 1U)
            << kDiagnostic << "WaitFlag before its producer: " << key;
      }
    } else {
      LOG(FATAL) << kDiagnostic
                 << "unsupported control or nested hard-event expression";
    }
  }

  const FlagDiscovery &discovery_;
  const std::vector<PrimExpr> &stable_conditions_;
  uint32_t choices_;
  size_t *actions_;
  LegacyProtocolContext context_;
  Tokens tokens_;
  size_t loop_depth_{0};
  int required_owner_{-1};
  bool cross_core_{false};
};

class RemoveExplicitFlags : public StmtExprMutator {
private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>();
        call != nullptr && IsExplicitFlag(call))
      return Evaluate(0, op->span);
    return GetRef<Stmt>(op);
  }
};

} // namespace

LegacyManualEventResult NormalizeLegacyManualEvents(const PrimFunc &function) {
  FlagDiscovery discovery;
  discovery.Run(function->body);
  if (!discovery.NeedsNormalization())
    return {function, false};
  ICHECK(!discovery.HasAutomaticFlags())
      << kDiagnostic
      << "mixed explicit and automatic hard flags are "
         "unsupported";
  const bool cross_core = discovery.HasCrossFlags();
  RepeatedBinders binders;
  binders.Run(function);
  const auto conditions =
      StableConditions(discovery, binders).Run(function->body);
  const uint32_t paths = uint32_t(1) << conditions.size();
  size_t actions = 0;
  int required_owner = -1;
  // Different expressions are deliberately independent choices. Enumerating
  // impossible combinations can conservatively reject a legal protocol; it
  // cannot make an unproved protocol pass by guessing a relation between them.
  for (uint32_t choices = 0; choices < paths; ++choices) {
    TokenExecutor executor(discovery, conditions, choices, &actions,
                           cross_core);
    executor.Run(function->body);
    const int owner = executor.RequiredOwner();
    ICHECK(required_owner < 0 || owner < 0 || required_owner == owner)
        << kDiagnostic << "conditional hard events require conflicting owners";
    if (owner >= 0)
      required_owner = owner;
  }
  PrimFunc result = function;
  result.CopyOnWrite()->body = RemoveExplicitFlags()(function->body);
  return {std::move(result), true, required_owner, cross_core};
}

} // namespace tvm::tl::ascend_sync
