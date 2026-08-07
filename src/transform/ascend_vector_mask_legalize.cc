// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_vector_mask_legalize.cc
 * \brief Repair Ascend Vector mask state immediately before target codegen.
 */

#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "arith/ir_mutator_with_analyzer.h"

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/ascend.h"
#include "common/ascend_vector_mask_contract.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

struct Transparency {
  bool mode{true};
  bool lo{true};
  bool hi{true};
};

class ResourceScopeDetector final : public StmtVisitor {
public:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "resource_scope") {
      found = true;
    }
    StmtVisitor::VisitStmt_(op);
  }

  bool found{false};
};

class LoopTransparencySummarizer final : public StmtExprVisitor {
public:
  explicit LoopTransparencySummarizer(arith::Analyzer *analyzer)
      : analyzer_(analyzer) {}

  void VisitExpr_(const CallNode *op) final {
    Call call = GetRef<Call>(op);
    if (IsSelectedVectorTerminal(call)) {
      MaskContract contract = MaskContractOf(call, analyzer_);
      transparency.mode &=
          contract.mode.requirement_kind() == MaskRequirementKind::kAny &&
          contract.mode.ensure_kind() == MaskEnsureKind::kPreserve;
      const bool no_payload_requirement =
          contract.lo.requirement_kind() == MaskRequirementKind::kAny &&
          contract.hi.requirement_kind() == MaskRequirementKind::kAny;
      transparency.lo &= no_payload_requirement &&
                         contract.lo.ensure_kind() == MaskEnsureKind::kPreserve;
      transparency.hi &= no_payload_requirement &&
                         contract.hi.ensure_kind() == MaskEnsureKind::kPreserve;
      return;
    }
    NonTerminalMaskEffect effect = ClassifyNonTerminalMaskEffect(call);
    ICHECK(effect != NonTerminalMaskEffect::kUnclassified)
        << "Unclassified call reached AscendVectorMaskLegalize: " << call;
    if (effect == NonTerminalMaskEffect::kBarrier) {
      transparency = {false, false, false};
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  Transparency transparency;

private:
  arith::Analyzer *analyzer_;
};

AscendMaskMode ParseMode(const PrimExpr &value) {
  const auto *imm = value.as<IntImmNode>();
  ICHECK(imm && (imm->value == 0 || imm->value == 1))
      << "Mask mode must be NORMAL(0) or COUNTER(1), got " << value;
  return imm->value == 0 ? AscendMaskMode::kNormal : AscendMaskMode::kCounter;
}

bool UsesBoundVar(const PrimExpr &expr, const Var &var) {
  if (!expr.defined()) {
    return false;
  }
  return UsesVar(
      expr, [&](const VarNode *candidate) { return candidate == var.get(); });
}

} // namespace

class AscendVectorMaskLegalizer final : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc func, PassContext ctx, Target target,
                             std::string platform) {
    if (!UseCompilerManagedVectorMask(target, platform)) {
      return func;
    }

    ResourceScopeDetector detector;
    detector(func->body);

    arith::Analyzer analyzer;
    PrimExpr shape_constraint = const_true();
    for (const auto &[_, buffer] : func->buffer_map) {
      for (const PrimExpr &dimension : buffer->shape) {
        if (dimension.dtype().is_int() || dimension.dtype().is_uint()) {
          shape_constraint =
              And(shape_constraint, dimension >= make_zero(dimension.dtype()));
        }
      }
    }
    AscendVectorMaskLegalizer legalizer(&analyzer, std::move(target),
                                        std::move(platform), detector.found,
                                        func->params);
    PrimFuncNode *copy = func.CopyOnWrite();
    With<arith::ConstraintContext> constraint(&analyzer, shape_constraint);
    if (detector.found) {
      copy->body = legalizer(func->body);
    } else {
      legalizer.in_vector_region_ = true;
      legalizer.facts_ = MaskFacts::Unknown();
      copy->body = legalizer(func->body);
      legalizer.in_vector_region_ = false;
    }
    return func;
  }

  AscendVectorMaskLegalizer(arith::Analyzer *analyzer, Target target,
                            std::string platform, bool has_resource_scopes,
                            const Array<Var> &parameters)
      : arith::IRMutatorWithAnalyzer(analyzer), target_(std::move(target)),
        platform_(std::move(platform)),
        profile_(AscendVectorMaskTargetProfile::Get(platform_)),
        has_resource_scopes_(has_resource_scopes) {
    for (const Var &parameter : parameters) {
      lexical_scope_.insert(parameter.get());
    }
  }

private:
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> sequence;
    for (const Stmt &stmt : op->seq) {
      Stmt rewritten = VisitStmt(stmt);
      if (const auto *nested = rewritten.as<SeqStmtNode>()) {
        for (const Stmt &item : nested->seq) {
          sequence.push_back(item);
        }
      } else {
        sequence.push_back(rewritten);
      }
    }
    if (sequence.empty()) {
      return Evaluate(0);
    }
    if (sequence.size() == 1) {
      return sequence[0];
    }
    return SeqStmt(sequence);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call_node = op->value.as<CallNode>();
    if (!call_node) {
      return GetRef<Stmt>(op);
    }
    Call call = GetRef<Call>(call_node);
    if (IsSelectedVectorTerminal(call)) {
      ICHECK(in_vector_region_)
          << "Selected Vector terminal escaped its resource_scope=1 region: "
          << call;
      return RewriteConsumer(call);
    }
    ICHECK(!IsVectorMaskSetter(call))
        << "Mask Legalizer input already contains a compiler mask setter: "
        << call;
    NonTerminalMaskEffect effect = ClassifyNonTerminalMaskEffect(call);
    ICHECK(effect != NonTerminalMaskEffect::kUnclassified)
        << "Unclassified call reached AscendVectorMaskLegalize: " << call;
    if (effect == NonTerminalMaskEffect::kBarrier) {
      ICHECK(in_vector_region_ || !has_resource_scopes_)
          << "T._src_code mask barrier is outside resource_scope=1";
      facts_ = MaskFacts::Unknown();
    }
    return GetRef<Stmt>(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);
    MaskFacts incoming = facts_;

    Stmt then_case;
    {
      With<arith::ConstraintContext> context(analyzer_, condition);
      facts_ = incoming;
      then_case = VisitStmt(op->then_case);
    }
    MaskFacts then_out = facts_;

    Optional<Stmt> else_case;
    MaskFacts else_out = incoming;
    if (op->else_case.defined()) {
      With<arith::ConstraintContext> context(
          analyzer_, analyzer_->rewrite_simplify(Not(condition)));
      facts_ = incoming;
      else_case = VisitStmt(op->else_case.value());
      else_out = facts_;
    }
    facts_ = MergeFacts(then_out, else_out);

    if (is_one(condition)) {
      facts_ = then_out;
      return then_case;
    }
    if (is_zero(condition)) {
      facts_ = else_out;
      return else_case.value_or(Evaluate(0));
    }
    return IfThenElse(condition, then_case, else_case, op->span);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    LoopTransparencySummarizer summarizer(analyzer_);
    summarizer(op->body);
    MaskFacts incoming = facts_;
    MaskFacts body_in = incoming;
    if (!summarizer.transparency.mode) {
      body_in.mode.reset();
    }
    if (!summarizer.transparency.lo) {
      body_in.lo.reset();
    }
    if (!summarizer.transparency.hi) {
      body_in.hi.reset();
    }

    facts_ = body_in;
    lexical_scope_.insert(op->loop_var.get());
    Stmt rewritten = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    lexical_scope_.erase(op->loop_var.get());

    facts_ = incoming;
    if (!summarizer.transparency.mode) {
      facts_.mode.reset();
    }
    if (!summarizer.transparency.lo) {
      facts_.lo.reset();
    }
    if (!summarizer.transparency.hi) {
      facts_.hi.reset();
    }
    DropBoundVar(op->loop_var, &facts_);
    return rewritten;
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    lexical_scope_.insert(op->var.get());
    Stmt rewritten = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    lexical_scope_.erase(op->var.get());
    ProjectLet(op->var, op->value,
               IsPureBufferFreeMaskPayload(op->value, lexical_scope_), &facts_);
    return rewritten;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != "resource_scope") {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    const auto *scope_imm = op->value.as<IntImmNode>();
    ICHECK(scope_imm && (scope_imm->value == 0 || scope_imm->value == 1))
        << "resource_scope must be 0 (AIC) or 1 (AIV), got " << op->value;
    int scope = static_cast<int>(scope_imm->value);
    ICHECK(active_resource_scope_ == -1 || active_resource_scope_ == scope)
        << "Overlapping AIC/AIV resource scopes are unsupported";

    int saved_scope = active_resource_scope_;
    bool saved_in_region = in_vector_region_;
    MaskFacts saved_facts = facts_;
    active_resource_scope_ = scope;
    if (saved_scope == -1 && scope == 1) {
      in_vector_region_ = true;
      facts_ = MaskFacts::Unknown();
    }
    Stmt body = VisitStmt(op->body);
    if (saved_scope == -1 && scope == 1) {
      facts_ = saved_facts;
      in_vector_region_ = saved_in_region;
    }
    active_resource_scope_ = saved_scope;
    return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    for (const IterVar &iter : op->iter_vars) {
      lexical_scope_.insert(iter->var.get());
    }
    Stmt rewritten = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    for (const IterVar &iter : op->iter_vars) {
      lexical_scope_.erase(iter->var.get());
      DropBoundVar(iter->var, &facts_);
    }
    return rewritten;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = GetRef<Call>(op);
    ICHECK(ClassifyNonTerminalMaskEffect(call) ==
           NonTerminalMaskEffect::kNeutral)
        << "Non-neutral or unclassified call is nested in an expression "
           "after Selection: "
        << call;
    return arith::IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    ICHECK(false) << "While reached AscendVectorMaskLegalize after Selection";
    throw;
  }

  Stmt RewriteConsumer(const Call &call) {
    MaskContract contract = MaskContractOf(call, analyzer_);
    ValidateConsumerPayloads(call);
    std::vector<Stmt> sequence;

    if (contract.mode.requirement_kind() == MaskRequirementKind::kExact &&
        !ModeSatisfied(contract.mode.required())) {
      PrimExpr required = contract.mode.required();
      sequence.push_back(Evaluate(
          Call(DataType::Handle(), ascend_set_mask_mode(), {required})));
      facts_.mode = ParseMode(required);
    }

    bool repair_lo =
        contract.lo.requirement_kind() == MaskRequirementKind::kExact &&
        !PayloadSatisfied(facts_.lo, contract.lo.required());
    bool repair_hi =
        contract.hi.requirement_kind() == MaskRequirementKind::kExact &&
        !PayloadSatisfied(facts_.hi, contract.hi.required());
    if (repair_lo || repair_hi) {
      PayloadPair pair = CompletePayload(contract);
      sequence.push_back(Evaluate(Call(
          DataType::Handle(), ascend_set_mask_payload(), {pair.lo, pair.hi})));
      facts_.lo = pair.lo;
      facts_.hi = pair.hi;
    }

    sequence.push_back(Evaluate(call));
    ApplyContract(contract, &facts_);
    if (sequence.size() == 1) {
      return sequence[0];
    }
    return SeqStmt(sequence);
  }

  bool ModeSatisfied(const PrimExpr &required) const {
    return facts_.mode.has_value() &&
           facts_.mode.value() == ParseMode(required);
  }

  void ValidateConsumerPayloads(const Call &call) const {
    const auto *spec = SelectedVectorTerminalSpecOf(call);
    ICHECK(spec != nullptr);
    std::vector<PrimExpr> payloads;
    if (spec->contract_kind == SelectedMaskContractKind::kRawCounter) {
      ICHECK(!call->args.empty());
      payloads.push_back(call->args.back());
    } else if (spec->contract_kind ==
                   SelectedMaskContractKind::kRawNormalDynamicPayload ||
               spec->contract_kind == SelectedMaskContractKind::
                                          kSelfContainedNormalDynamicPayload) {
      ICHECK_GE(call->args.size(), 2U);
      payloads.push_back(call->args[call->args.size() - 2]);
      payloads.push_back(call->args.back());
    }
    for (const PrimExpr &payload : payloads) {
      ICHECK(IsPureBufferFreeMaskPayload(payload, lexical_scope_))
          << "Selected Vector terminal has a non-pure, buffer-backed, or "
             "out-of-scope mask payload: "
          << call;
    }
  }

  bool PayloadSatisfied(const std::optional<PrimExpr> &known,
                        const PrimExpr &required) const {
    return known.has_value() && MaskPayloadEqual(*known, required, analyzer_);
  }

  PayloadPair CompletePayload(const MaskContract &contract) {
    uint8_t possible_modes = kPossibleNormal | kPossibleCounter;
    if (contract.mode.requirement_kind() == MaskRequirementKind::kExact) {
      possible_modes =
          ParseMode(contract.mode.required()) == AscendMaskMode::kNormal
              ? kPossibleNormal
              : kPossibleCounter;
    } else if (facts_.mode.has_value()) {
      possible_modes = facts_.mode.value() == AscendMaskMode::kNormal
                           ? kPossibleNormal
                           : kPossibleCounter;
    }

    std::optional<PrimExpr> fixed_lo;
    std::optional<PrimExpr> fixed_hi;
    if (contract.lo.requirement_kind() == MaskRequirementKind::kExact) {
      fixed_lo = NormalizeMaskPayload(contract.lo.required(), analyzer_);
    }
    if (contract.hi.requirement_kind() == MaskRequirementKind::kExact) {
      fixed_hi = NormalizeMaskPayload(contract.hi.required(), analyzer_);
    }

    std::optional<PrimExpr> candidate_lo =
        fixed_lo.has_value() ? fixed_lo : facts_.lo;
    std::optional<PrimExpr> candidate_hi =
        fixed_hi.has_value() ? fixed_hi : facts_.hi;
    if (candidate_lo.has_value() && candidate_hi.has_value() &&
        profile_.LegalPayloadPair(possible_modes, *candidate_lo, *candidate_hi,
                                  analyzer_)) {
      return {NormalizeMaskPayload(*candidate_lo, analyzer_),
              NormalizeMaskPayload(*candidate_hi, analyzer_)};
    }
    return profile_.CanonicalComplete(possible_modes, fixed_lo, fixed_hi,
                                      analyzer_);
  }

  void ApplyContract(const MaskContract &contract, MaskFacts *facts) {
    ApplyModeEnsure(contract.mode, facts);
    ApplyPayloadEnsure(contract.lo, &facts->lo);
    ApplyPayloadEnsure(contract.hi, &facts->hi);
  }

  void ApplyModeEnsure(const MaskFieldContract &field, MaskFacts *facts) {
    switch (field.ensure_kind()) {
    case MaskEnsureKind::kPreserve:
      return;
    case MaskEnsureKind::kExact:
      facts->mode = ParseMode(field.ensured());
      return;
    case MaskEnsureKind::kUnknown:
      facts->mode.reset();
      return;
    }
  }

  void ApplyPayloadEnsure(const MaskFieldContract &field,
                          std::optional<PrimExpr> *fact) {
    switch (field.ensure_kind()) {
    case MaskEnsureKind::kPreserve:
      return;
    case MaskEnsureKind::kExact:
      *fact = NormalizeMaskPayload(field.ensured(), analyzer_);
      return;
    case MaskEnsureKind::kUnknown:
      fact->reset();
      return;
    }
  }

  MaskFacts MergeFacts(const MaskFacts &lhs, const MaskFacts &rhs) {
    MaskFacts result;
    if (lhs.mode.has_value() && rhs.mode.has_value() && lhs.mode == rhs.mode) {
      result.mode = lhs.mode;
    }
    if (lhs.lo.has_value() && rhs.lo.has_value() &&
        MaskPayloadEqual(*lhs.lo, *rhs.lo, analyzer_)) {
      result.lo = lhs.lo;
    }
    if (lhs.hi.has_value() && rhs.hi.has_value() &&
        MaskPayloadEqual(*lhs.hi, *rhs.hi, analyzer_)) {
      result.hi = lhs.hi;
    }
    return result;
  }

  void ProjectLet(const Var &var, const PrimExpr &value, bool legal_value,
                  MaskFacts *facts) {
    auto project = [&](std::optional<PrimExpr> *fact) {
      if (!fact->has_value() || !UsesBoundVar(**fact, var)) {
        return;
      }
      if (!legal_value) {
        fact->reset();
        return;
      }
      PrimExpr substituted =
          tir::Substitute(**fact, {{var, analyzer_->Simplify(value)}});
      if (UsesBoundVar(substituted, var)) {
        fact->reset();
      } else {
        *fact = NormalizeMaskPayload(substituted, analyzer_);
      }
    };
    project(&facts->lo);
    project(&facts->hi);
  }

  void DropBoundVar(const Var &var, MaskFacts *facts) {
    if (facts->lo.has_value() && UsesBoundVar(*facts->lo, var)) {
      facts->lo.reset();
    }
    if (facts->hi.has_value() && UsesBoundVar(*facts->hi, var)) {
      facts->hi.reset();
    }
  }

  Target target_;
  std::string platform_;
  const AscendVectorMaskTargetProfile &profile_;
  bool has_resource_scopes_{false};
  bool in_vector_region_{false};
  int active_resource_scope_{-1};
  MaskFacts facts_;
  std::unordered_set<const VarNode *> lexical_scope_;
};

tvm::transform::Pass AscendVectorMaskLegalize(Target target,
                                              std::string platform) {
  auto pass_func = [=](PrimFunc func, IRModule module, PassContext ctx) {
    return AscendVectorMaskLegalizer::Substitute(std::move(func), ctx, target,
                                                 platform);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendVectorMaskLegalize", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendVectorMaskLegalize")
    .set_body_typed(AscendVectorMaskLegalize);

} // namespace tl
} // namespace tvm
