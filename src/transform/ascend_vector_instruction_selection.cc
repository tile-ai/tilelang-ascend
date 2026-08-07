// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_vector_instruction_selection.cc
 * \brief Select physical Ascend Vector terminals before Phase-2 scheduling.
 */

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "arith/ir_mutator_with_analyzer.h"

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

DataType AccessPtrDtype(const PrimExpr &expr) {
  const auto *access_ptr = expr.as<CallNode>();
  ICHECK(access_ptr && access_ptr->op.same_as(builtin::tvm_access_ptr()) &&
         !access_ptr->args.empty())
      << "Expected tvm_access_ptr, got " << expr;
  const PrimExpr &type_arg = access_ptr->args[0];
  if (const auto *call = type_arg.as<CallNode>()) {
    return call->dtype;
  }
  if (const auto *str = type_arg.as<StringImmNode>()) {
    return DataType(runtime::String2DLDataType(str->value));
  }
  LOG(FATAL) << "Unexpected access_ptr dtype operand " << type_arg;
  throw;
}

std::string Trim(std::string value) {
  auto first = std::find_if_not(value.begin(), value.end(), [](char c) {
    return std::isspace(static_cast<unsigned char>(c));
  });
  auto last = std::find_if_not(value.rbegin(), value.rend(), [](char c) {
                return std::isspace(static_cast<unsigned char>(c));
              }).base();
  return first < last ? std::string(first, last) : std::string();
}

std::vector<std::string> ParseTemplateArguments(const std::string &tag) {
  size_t begin = tag.find('<');
  size_t end = tag.rfind('>');
  ICHECK(begin != std::string::npos && end != std::string::npos && begin < end)
      << "Expected a templated Ascend operation tag, got " << tag;
  std::vector<std::string> result;
  std::stringstream stream(tag.substr(begin + 1, end - begin - 1));
  std::string item;
  while (std::getline(stream, item, ',')) {
    result.push_back(Trim(item));
  }
  return result;
}

std::pair<PrimExpr, PrimExpr> NormalMaskBits(const PrimExpr &length,
                                             DataType dtype) {
  const auto *imm = length.as<IntImmNode>();
  ICHECK(imm) << "NORMAL mask length must be compile-time constant, got "
              << length;
  int64_t len = imm->value;
  ICHECK_GE(len, 0);
  ICHECK_LE(len, 128);
  int64_t type_len = 0;
  if (dtype.is_int() && dtype.bits() == 4) {
    type_len = 64;
  } else {
    ICHECK_GT(dtype.bytes(), 0);
    type_len = 32 / dtype.bytes();
  }

  constexpr uint64_t kFull = std::numeric_limits<uint64_t>::max();
  uint64_t lo = 0;
  uint64_t hi = 0;
  if (len == 64) {
    lo = kFull;
  } else if (len == type_len || len >= 128) {
    lo = kFull;
    hi = kFull;
  } else if (len > 64) {
    lo = kFull;
    hi = (uint64_t{1} << static_cast<uint32_t>(len - 64)) - 1;
  } else if (len > 0) {
    lo = (uint64_t{1} << static_cast<uint32_t>(len)) - 1;
  }
  return {make_const(DataType::UInt(64), lo),
          make_const(DataType::UInt(64), hi)};
}

Call WithSelectedOp(const CallNode *call, const std::string &selected_name,
                    Array<PrimExpr> args) {
  return Call(call->dtype, Op::Get(selected_name), std::move(args), call->span);
}

Call WithSelectedOp(const CallNode *call, const std::string &selected_name) {
  return WithSelectedOp(call, selected_name, call->args);
}

bool IsCopyUbToUb(const CallNode *call) {
  if (!call->op.same_as(builtin::call_extern()) || call->args.empty()) {
    return false;
  }
  const auto *name = call->args[0].as<StringImmNode>();
  if (name == nullptr) {
    return false;
  }
  std::string value = name->value;
  return value.find("copy_ub_to_ub<") != std::string::npos;
}

bool CopyUbToUbIsTypeNeutral(const CallNode *call) {
  std::string name = Downcast<StringImm>(call->args[0])->value;
  std::vector<std::string> params = ParseTemplateArguments(name);
  ICHECK_GE(params.size(), 2U) << "Malformed copy_ub_to_ub tag " << name;
  return params[0] == params[1];
}

bool IsIntegerLiteral(const PrimExpr &value) {
  if (value.as<IntImmNode>() != nullptr) {
    return true;
  }
  const auto *call = value.as<CallNode>();
  if (call == nullptr || !call->op.same_as(tir::builtin::large_uint_imm())) {
    return false;
  }
  return std::all_of(
      call->args.begin(), call->args.end(),
      [](const PrimExpr &part) { return part.as<IntImmNode>() != nullptr; });
}

arith::ConstIntBound InferLetValueBound(const PrimExpr &value,
                                        arith::Analyzer *analyzer) {
  ICHECK(analyzer != nullptr);
  PrimExpr true_value;
  PrimExpr false_value;
  if (const auto *select = value.as<SelectNode>()) {
    true_value = select->true_value;
    false_value = select->false_value;
  } else if (const auto *call = value.as<CallNode>();
             call != nullptr &&
             call->op.same_as(tir::builtin::if_then_else()) &&
             call->args.size() == 3U) {
    true_value = call->args[1];
    false_value = call->args[2];
  } else {
    return analyzer->const_int_bound(value);
  }
  arith::ConstIntBound true_bound = InferLetValueBound(true_value, analyzer);
  arith::ConstIntBound false_bound = InferLetValueBound(false_value, analyzer);
  return arith::ConstIntBound(
      std::min(true_bound->min_value, false_bound->min_value),
      std::max(true_bound->max_value, false_bound->max_value));
}

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

} // namespace

class AscendVectorInstructionSelector final
    : public arith::IRMutatorWithAnalyzer {
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
    AscendVectorInstructionSelector selector(&analyzer, std::move(target),
                                             std::move(platform), func->params,
                                             detector.found);
    PrimFuncNode *copy = func.CopyOnWrite();
    With<arith::ConstraintContext> constraint(&analyzer, shape_constraint);
    copy->body = selector(func->body);
    return func;
  }

  AscendVectorInstructionSelector(arith::Analyzer *analyzer, Target target,
                                  std::string platform,
                                  const Array<Var> &parameters,
                                  bool has_resource_scopes)
      : arith::IRMutatorWithAnalyzer(analyzer), target_(std::move(target)),
        platform_(std::move(platform)),
        has_resource_scopes_(has_resource_scopes) {
    for (const Var &parameter : parameters) {
      lexical_scope_.insert(parameter.get());
    }
  }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (IsCopyUbToUb(call)) {
        ValidateVectorRegion(call);
        ValidateNeutralOperands(call);
        const char *selected =
            CopyUbToUbIsTypeNeutral(call)
                ? "tl.ascend_copy_ub_to_ub_neutral"
                : "tl.ascend_copy_ub_to_ub_self_contained_normal_full";
        Call selected_call = WithSelectedOp(call, selected);
        ValidateSelectedPayloads(call, selected_call);
        return Evaluate(selected_call);
      }
      if (const auto *op_node = call->op.as<OpNode>()) {
        Op semantic_op = GetRef<Op>(op_node);
        if (!semantic_op.same_as(builtin::call_extern()) &&
            !SelectedVectorTerminalSpecsForBase(semantic_op).empty()) {
          ValidateVectorRegion(call);
          ValidateNeutralOperands(call);
          Call selected_call = SelectSemanticTerminal(call);
          ValidateSelectedPayloads(call, selected_call);
          return Evaluate(selected_call);
        }
      }
      if (call->op.same_as(ascend_set_mask_mode()) ||
          call->op.same_as(ascend_set_mask_payload()) ||
          IsSelectedVectorTerminal(GetRef<Call>(call))) {
        ReportUnsupported(call, "internal mask operation appeared before "
                                "instruction selection");
      }
      if (call->op.same_as(ascend_row_expand_mul())) {
        ReportUnsupported(call,
                          "tl.ascend_row_expand_mul has no AscendC emission; "
                          "use the public experiment primitive only on a "
                          "supported backend");
      }
      NonTerminalMaskEffect effect =
          ClassifyNonTerminalMaskEffect(GetRef<Call>(call));
      if (effect == NonTerminalMaskEffect::kBarrier) {
        ValidateVectorRegion(call);
        return GetRef<Stmt>(op);
      }
      if (effect == NonTerminalMaskEffect::kUnclassified) {
        ReportUnsupported(
            call, "call is absent from the audited non-terminal effect table");
      }
    }
    PrimExpr value = VisitExpr(op->value);
    return Evaluate(value);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = GetRef<Call>(op);
    if (const auto *op_node = op->op.as<OpNode>()) {
      Op semantic_op = GetRef<Op>(op_node);
      if (!semantic_op.same_as(builtin::call_extern()) &&
          !SelectedVectorTerminalSpecsForBase(semantic_op).empty()) {
        ReportUnsupported(
            op, "mask-effect Vector call must be the direct value of an "
                "Evaluate statement");
      }
    }
    if (ClassifyNonTerminalMaskEffect(call) !=
        NonTerminalMaskEffect::kNeutral) {
      ReportUnsupported(
          op, "nested call is not an audited mask-neutral expression call");
    }
    return arith::IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  void ValidateNeutralOperands(const CallNode *call) {
    for (const PrimExpr &argument : call->args) {
      VisitExpr(argument);
    }
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    LOG(FATAL) << "AscendVectorInstructionSelection does not support While in "
                  "the managed A2/A3 Vector-mask grammar: "
               << GetRef<Stmt>(op);
    throw;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    lexical_scope_.insert(op->loop_var.get());
    Stmt result = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    lexical_scope_.erase(op->loop_var.get());
    return result;
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    PrimExpr body_constraint = const_true();
    if (SideEffect(value) <= CallEffectKind::kPure) {
      analyzer_->Bind(op->var, value);
    } else if (value.dtype().is_int() || value.dtype().is_uint()) {
      arith::ConstIntBound bound = InferLetValueBound(value, analyzer_);
      if (bound->min_value != arith::ConstIntBound::kNegInf) {
        body_constraint =
            And(body_constraint,
                op->var >= make_const(op->var.dtype(), bound->min_value));
      }
      if (bound->max_value != arith::ConstIntBound::kPosInf) {
        body_constraint =
            And(body_constraint,
                op->var <= make_const(op->var.dtype(), bound->max_value));
      }
    }

    lexical_scope_.insert(op->var.get());
    Stmt body;
    {
      With<arith::ConstraintContext> context(analyzer_, body_constraint);
      body = VisitStmt(op->body);
    }
    lexical_scope_.erase(op->var.get());
    if (value.same_as(op->value) && body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }
    return LetStmt(op->var, value, body, op->span);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != "resource_scope") {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    const auto *scope_imm = op->value.as<IntImmNode>();
    if (scope_imm == nullptr ||
        (scope_imm->value != 0 && scope_imm->value != 1)) {
      LOG(FATAL) << "Unsupported A2/A3 compiler-managed Vector-mask input: "
                    "resource_scope must be 0 (AIC) or 1 (AIV), got "
                 << op->value;
    }
    int scope = static_cast<int>(scope_imm->value);
    if (active_resource_scope_ != -1 && active_resource_scope_ != scope) {
      LOG(FATAL) << "Unsupported A2/A3 compiler-managed Vector-mask input: "
                    "overlapping AIC/AIV resource scopes";
    }
    int saved_scope = active_resource_scope_;
    active_resource_scope_ = scope;
    Stmt body = VisitStmt(op->body);
    active_resource_scope_ = saved_scope;
    return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    if (op->init.defined()) {
      LOG(FATAL) << "AscendVectorInstructionSelection does not support a "
                    "Block init in the managed A2/A3 Vector-mask grammar: "
                 << GetRef<Stmt>(op);
    }
    for (const IterVar &iter_var : op->iter_vars) {
      lexical_scope_.insert(iter_var->var.get());
    }
    Stmt result = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    for (const IterVar &iter_var : op->iter_vars) {
      lexical_scope_.erase(iter_var->var.get());
    }
    return result;
  }

  void ValidateSelectedPayloads(const CallNode *semantic,
                                const Call &selected) const {
    const auto *spec = SelectedVectorTerminalSpecOf(selected);
    ICHECK(spec != nullptr);
    std::vector<PrimExpr> payloads;
    bool counter = false;
    bool normal = false;
    switch (spec->contract_kind) {
    case SelectedMaskContractKind::kRawCounter:
      ICHECK(!selected->args.empty());
      payloads.push_back(selected->args.back());
      counter = true;
      break;
    case SelectedMaskContractKind::kRawNormalDynamicPayload:
    case SelectedMaskContractKind::kSelfContainedNormalDynamicPayload:
      ICHECK_GE(selected->args.size(), 2U);
      payloads.push_back(selected->args[selected->args.size() - 2]);
      payloads.push_back(selected->args.back());
      normal = true;
      break;
    default:
      return;
    }

    for (const PrimExpr &payload : payloads) {
      PrimExpr simplified = analyzer_->Simplify(payload);
      if (!IsPureBufferFreeMaskPayload(simplified, lexical_scope_)) {
        ReportUnsupported(semantic,
                          "mask payload is not pure, buffer-free, and lexical");
      }
      bool non_negative =
          simplified.dtype().is_uint() ||
          simplified.as<SizeVarNode>() != nullptr ||
          analyzer_->CanProve(simplified >= make_zero(simplified.dtype()));
      if (!(simplified.dtype().is_int() || simplified.dtype().is_uint()) ||
          simplified.dtype().bits() > 64 || !non_negative) {
        ReportUnsupported(semantic,
                          "mask payload must be provably non-negative and "
                          "representable as uint64");
      }
      if (normal && !IsIntegerLiteral(simplified)) {
        ReportUnsupported(
            semantic,
            "NORMAL mask payload must simplify to a compile-time constant");
      }
      if (counter) {
        PrimExpr normalized = NormalizeMaskPayload(simplified, analyzer_);
        PrimExpr limit = make_const(DataType::UInt(64),
                                    std::numeric_limits<uint32_t>::max());
        bool fits_counter = analyzer_->CanProve(normalized <= limit) ||
                            simplified.dtype().bits() <= 32;
        if (!fits_counter) {
          ReportUnsupported(semantic,
                            "COUNTER mask payload must fit in uint32");
        }
      }
    }
  }

  void ValidateVectorRegion(const CallNode *call) const {
    if (has_resource_scopes_ && active_resource_scope_ != 1) {
      ReportUnsupported(
          call, "mask-effect call must be inside exactly one resource_scope=1 "
                "Vector execution region");
    }
  }

  Call SelectSemanticTerminal(const CallNode *call) {
    const Op base = Downcast<Op>(call->op);
    const auto &candidates = SelectedVectorTerminalSpecsForBase(base);
    ICHECK(!candidates.empty());

    if (base.same_as(ascend_reduce())) {
      return SelectReduce(call);
    }
    if (base.same_as(ascend_broadcast())) {
      return SelectBroadcast(call);
    }
    if (base.same_as(ascend_gather_mask())) {
      ICHECK_GE(call->args.size(), 4U);
      const bool custom = call->args[3].as<CallNode>() != nullptr;
      return WithSelectedOp(
          call, custom ? "tl.ascend_gather_mask_custom_composite"
                       : "tl.ascend_gather_mask_fixed_self_contained_normal");
    }
    if (base.same_as(ascend_round())) {
      const bool has_tmp =
          call->args.size() >= 4 && call->args[2].as<CallNode>() != nullptr;
      return WithSelectedOp(call, has_tmp ? "tl.ascend_round_advanced_composite"
                                          : "tl.ascend_round_cast_raw_counter");
    }

    const SelectedVectorTerminalSpec &candidate = candidates.front();
    Array<PrimExpr> args = call->args;
    if (candidate.contract_kind ==
        SelectedMaskContractKind::kRawNormalDynamicPayload) {
      AppendNormalPayload(call, base, &args);
    } else if (candidate.contract_kind ==
               SelectedMaskContractKind::kSelfContainedNormalDynamicPayload) {
      AppendSelfContainedPayload(call, base, &args);
    }
    return WithSelectedOp(call, candidate.selected->name, std::move(args));
  }

  Call SelectBroadcast(const CallNode *call) {
    const bool has_tmp =
        call->args.size() >= 4 && call->args[3].as<CallNode>() != nullptr;
    if (has_tmp) {
      return WithSelectedOp(call, "tl.ascend_broadcast_advanced_composite");
    }
    ICHECK_GE(call->args.size(), 4U);
    const auto *dim_imm = call->args[3].as<IntImmNode>();
    ICHECK(dim_imm && dim_imm->value > 0)
        << "Broadcast rank must be a positive constant";
    int64_t dim = dim_imm->value;
    ICHECK_GE(call->args.size(), static_cast<size_t>(4 + 2 * dim));
    PrimExpr count = 1;
    for (int64_t i = 0; i < dim; ++i) {
      count *= call->args[4 + i];
    }
    Array<PrimExpr> args = call->args;
    args.push_back(NormalizeMaskPayload(count, analyzer_));
    return WithSelectedOp(call, "tl.ascend_broadcast_raw_counter",
                          std::move(args));
  }

  Call SelectReduce(const CallNode *call) {
    ICHECK_GE(call->args.size(), 4U);
    std::string tag = Downcast<StringImm>(call->args[0])->value;
    std::vector<std::string> params = ParseTemplateArguments(tag);
    ICHECK_EQ(params.size(), 4U) << "Malformed reduce tag " << tag;

    bool has_physical_row = call->args.size() > 4 &&
                            !call->args.back().dtype().is_bool() &&
                            call->args.back().as<IntImmNode>() != nullptr;
    bool clear = true;
    for (auto it = call->args.rbegin(); it != call->args.rend(); ++it) {
      if ((*it).dtype().is_bool()) {
        clear = !is_zero(*it);
        break;
      }
    }
    bool half_sum = tag.find("reduce_sum<half") != std::string::npos && clear;
    if (!has_physical_row && !half_sum) {
      return WithSelectedOp(call, "tl.ascend_reduce_advanced_composite");
    }

    int64_t m = std::stoll(params[1]);
    int64_t n = std::stoll(params[2]);
    int64_t dim = std::stoll(params[3]);
    int64_t mask = dim == -1 ? n : (dim == 0 ? m : m * n);
    auto bits = NormalMaskBits(IntImm(DataType::Int(32), mask),
                               AccessPtrDtype(call->args[2]));
    Array<PrimExpr> args = call->args;
    args.push_back(bits.first);
    args.push_back(bits.second);
    return WithSelectedOp(call,
                          has_physical_row
                              ? "tl.ascend_reduce_narrow_raw_normal"
                              : "tl.ascend_reduce_half_sum_raw_normal",
                          std::move(args));
  }

  PrimExpr RequireConstantNormalLength(const CallNode *call,
                                       PrimExpr length) const {
    PrimExpr simplified = analyzer_->Simplify(std::move(length));
    const auto *imm = simplified.as<IntImmNode>();
    if (imm == nullptr || imm->value < 0 || imm->value > 128) {
      ReportUnsupported(
          call, "NORMAL mask length must simplify to a constant in [0, 128]");
    }
    return simplified;
  }

  void AppendNormalPayload(const CallNode *call, const Op &base,
                           Array<PrimExpr> *args) {
    size_t mask_index = 0;
    size_t dtype_index = 1;
    if (base.same_as(ascend_block_reduce_max()) ||
        base.same_as(ascend_block_reduce_min()) ||
        base.same_as(ascend_block_reduce_sum())) {
      mask_index = 3;
    } else if (base.same_as(ascend_wholereducemax()) ||
               base.same_as(ascend_wholereducemin()) ||
               base.same_as(ascend_wholereducesum())) {
      mask_index = 2;
    } else if (base.same_as(ascend_fill_experiment())) {
      ICHECK_GE(call->args.size(), 4U);
      args->push_back(NormalizeMaskPayload(call->args[3], analyzer_));
      args->push_back(make_zero(DataType::UInt(64)));
      return;
    } else {
      ReportUnsupported(call, "cataloged NORMAL terminal is missing a payload "
                              "materializer");
    }
    PrimExpr mask_length =
        RequireConstantNormalLength(call, call->args[mask_index]);
    auto bits =
        NormalMaskBits(mask_length, AccessPtrDtype(call->args[dtype_index]));
    args->push_back(bits.first);
    args->push_back(bits.second);
  }

  void AppendSelfContainedPayload(const CallNode *call, const Op &base,
                                  Array<PrimExpr> *args) {
    if (base.same_as(ascend_bilinear_interpolation())) {
      PrimExpr mask_length = RequireConstantNormalLength(call, call->args[4]);
      auto bits = NormalMaskBits(mask_length, AccessPtrDtype(call->args[1]));
      args->push_back(bits.first);
      args->push_back(bits.second);
      return;
    }
    if (base.same_as(ascend_gather_mask_experiment())) {
      args->push_back(NormalizeMaskPayload(call->args[5], analyzer_));
      args->push_back(make_zero(DataType::UInt(64)));
      return;
    }
    ReportUnsupported(call, "cataloged self-contained NORMAL terminal is "
                            "missing a payload materializer");
  }

  [[noreturn]] void ReportUnsupported(const CallNode *call,
                                      const std::string &reason) const {
    const auto *op_node = call->op.as<OpNode>();
    std::string op_name =
        op_node ? std::string(op_node->name) : std::string("<non-Op>");
    LOG(FATAL) << "Unsupported A2/A3 compiler-managed Vector-mask input: op="
               << op_name << ", target=" << target_
               << ", platform=" << platform_ << ", reason=" << reason
               << ", call=" << GetRef<Call>(call);
    throw;
  }

  Target target_;
  std::string platform_;
  std::unordered_set<const VarNode *> lexical_scope_;
  bool has_resource_scopes_{false};
  int active_resource_scope_{-1};
};

tvm::transform::Pass AscendVectorInstructionSelection(Target target,
                                                      std::string platform) {
  auto pass_func = [=](PrimFunc func, IRModule module, PassContext ctx) {
    return AscendVectorInstructionSelector::Substitute(std::move(func), ctx,
                                                       target, platform);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendVectorInstructionSelection",
                            {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendVectorInstructionSelection")
    .set_body_typed(AscendVectorInstructionSelection);

} // namespace tl
} // namespace tvm
