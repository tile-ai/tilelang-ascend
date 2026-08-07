// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_vector_mask_contract.cc
 * \brief Canonical semantics for compiler-managed Ascend Vector mask state.
 */

#include "common/ascend_vector_mask_contract.h"

#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <tvm/runtime/logging.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>

#include "../op/ascend.h"

namespace tvm {
namespace tl {

namespace {

PrimExpr ModeExpr(AscendMaskMode mode) {
  return IntImm(DataType::Int(32), static_cast<int32_t>(mode));
}

PrimExpr FullPayload() {
  return make_const(DataType::UInt(64), std::numeric_limits<uint64_t>::max());
}

const std::vector<SelectedVectorTerminalSpec> &AllSelectedTerminalSpecs() {
  static const std::vector<SelectedVectorTerminalSpec> specs = {
#define TL_ASCEND_SELECTED_VECTOR_OP(Name, Arity, Base, ContractKind)          \
  {Op::Get("tl." #Name), Base(), SelectedMaskContractKind::ContractKind},
#include "../op/ascend_vector_mask_ops.inc"
#undef TL_ASCEND_SELECTED_VECTOR_OP
  };
  return specs;
}

const std::unordered_map<std::string, size_t> &SelectedTerminalIndex() {
  static const std::unordered_map<std::string, size_t> index = [] {
    std::unordered_map<std::string, size_t> result;
    const auto &specs = AllSelectedTerminalSpecs();
    for (size_t i = 0; i < specs.size(); ++i) {
      bool inserted = result.emplace(specs[i].selected->name, i).second;
      ICHECK(inserted) << "Duplicate selected Vector terminal "
                       << specs[i].selected;
    }
    return result;
  }();
  return index;
}

class PureBufferFreePayloadChecker final : public ExprVisitor {
public:
  explicit PureBufferFreePayloadChecker(
      const std::unordered_set<const VarNode *> &lexical_scope)
      : lexical_scope_(lexical_scope) {}

  bool Check(const PrimExpr &value) {
    VisitExpr(value);
    return valid_;
  }

private:
  void VisitExpr_(const VarNode *op) final {
    if (!lexical_scope_.count(op) && !local_scope_.count(op)) {
      valid_ = false;
    }
  }

  void VisitExpr_(const LetNode *op) final {
    VisitExpr(op->value);
    local_scope_.insert(op->var.get());
    VisitExpr(op->body);
    local_scope_.erase(op->var.get());
  }

  void VisitExpr_(const BufferLoadNode *op) final { valid_ = false; }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tir::builtin::large_uint_imm())) {
      ExprVisitor::VisitExpr_(op);
      return;
    }
    // Payloads are deliberately restricted to the arithmetic expression
    // grammar plus TVM's internal encoding for a uint64 literal. Even another
    // pure Call would make later effect classification and lexical projection
    // ambiguous.
    valid_ = false;
  }

  const std::unordered_set<const VarNode *> &lexical_scope_;
  std::unordered_set<const VarNode *> local_scope_;
  bool valid_{true};
};

} // namespace

MaskFieldContract::MaskFieldContract(MaskRequirementKind requirement_kind,
                                     MaskEnsureKind ensure_kind,
                                     PrimExpr required, PrimExpr ensured)
    : requirement_kind_(requirement_kind), ensure_kind_(ensure_kind),
      required_(std::move(required)), ensured_(std::move(ensured)) {
  ICHECK(!(ensure_kind_ == MaskEnsureKind::kPreserve &&
           requirement_kind_ == MaskRequirementKind::kExact))
      << "Exact/Preserve is not a canonical mask contract form";
  ICHECK_EQ(requirement_kind_ == MaskRequirementKind::kExact,
            required_.defined());
  ICHECK_EQ(ensure_kind_ == MaskEnsureKind::kExact, ensured_.defined());
}

MaskFieldContract MaskFieldContract::AnyPreserve() {
  return MaskFieldContract(MaskRequirementKind::kAny, MaskEnsureKind::kPreserve,
                           PrimExpr(), PrimExpr());
}

MaskFieldContract MaskFieldContract::AnyExact(PrimExpr ensured) {
  return MaskFieldContract(MaskRequirementKind::kAny, MaskEnsureKind::kExact,
                           PrimExpr(), std::move(ensured));
}

MaskFieldContract MaskFieldContract::AnyUnknown() {
  return MaskFieldContract(MaskRequirementKind::kAny, MaskEnsureKind::kUnknown,
                           PrimExpr(), PrimExpr());
}

MaskFieldContract MaskFieldContract::ExactExact(PrimExpr required,
                                                PrimExpr ensured) {
  return MaskFieldContract(MaskRequirementKind::kExact, MaskEnsureKind::kExact,
                           std::move(required), std::move(ensured));
}

MaskFieldContract MaskFieldContract::ExactUnknown(PrimExpr required) {
  return MaskFieldContract(MaskRequirementKind::kExact,
                           MaskEnsureKind::kUnknown, std::move(required),
                           PrimExpr());
}

bool UseCompilerManagedVectorMask(const Target &target,
                                  const std::string &platform) {
  if (!target.defined() || (platform != "A2" && platform != "A3")) {
    return false;
  }
  auto model_attr = target->attrs.Get("model");
  if (!model_attr.defined()) {
    return false;
  }
  std::string model = Downcast<String>(model_attr).operator std::string();
  return model == "ascendc" || model == "auto";
}

PrimExpr NormalizeMaskPayload(PrimExpr value, arith::Analyzer *analyzer) {
  ICHECK(value.defined());
  ICHECK(analyzer != nullptr);
  return analyzer->Simplify(cast(DataType::UInt(64), std::move(value)));
}

bool MaskPayloadEqual(const PrimExpr &lhs, const PrimExpr &rhs,
                      arith::Analyzer *analyzer) {
  ICHECK(lhs.defined() && rhs.defined());
  ICHECK(analyzer != nullptr);
  return analyzer->CanProveEqual(analyzer->Simplify(lhs),
                                 analyzer->Simplify(rhs));
}

bool IsPureBufferFreeMaskPayload(
    const PrimExpr &value,
    const std::unordered_set<const VarNode *> &lexical_scope) {
  if (!value.defined()) {
    return false;
  }
  return PureBufferFreePayloadChecker(lexical_scope).Check(value);
}

const AscendVectorMaskTargetProfile &
AscendVectorMaskTargetProfile::Get(const std::string &platform) {
  static const AscendVectorMaskTargetProfile a2("A2");
  static const AscendVectorMaskTargetProfile a3("A3");
  ICHECK(platform == "A2" || platform == "A3")
      << "Compiler-managed Vector mask is only defined for A2/A3, got "
      << platform;
  return platform == "A2" ? a2 : a3;
}

bool AscendVectorMaskTargetProfile::LegalPayloadPair(
    uint8_t possible_modes, const PrimExpr &lo, const PrimExpr &hi,
    arith::Analyzer *analyzer) const {
  ICHECK_NE(possible_modes, 0U);
  ICHECK(lo.defined() && hi.defined());
  ICHECK(analyzer != nullptr);

  if (possible_modes & kPossibleCounter) {
    PrimExpr normalized_lo = NormalizeMaskPayload(lo, analyzer);
    PrimExpr normalized_hi = NormalizeMaskPayload(hi, analyzer);
    PrimExpr limit =
        make_const(DataType::UInt(64), std::numeric_limits<uint32_t>::max());
    PrimExpr zero = make_zero(DataType::UInt(64));
    bool legal_lo = analyzer->CanProve(normalized_lo <= limit);
    if (!legal_lo) {
      // Selection proves signed symbolic lengths non-negative before building
      // selected IR.  Preserve that white-box symbolic form through the
      // uint64 normalization used by contracts and setters.
      if (const auto *cast_node = normalized_lo.as<CastNode>()) {
        DataType source_dtype = cast_node->value.dtype();
        legal_lo = (source_dtype.is_int() || source_dtype.is_uint()) &&
                   source_dtype.bits() <= 32;
      }
    }
    return legal_lo && analyzer->CanProveEqual(normalized_hi, zero);
  }
  return true;
}

PayloadPair AscendVectorMaskTargetProfile::CanonicalComplete(
    uint8_t possible_modes, const std::optional<PrimExpr> &fixed_lo,
    const std::optional<PrimExpr> &fixed_hi, arith::Analyzer *analyzer) const {
  ICHECK_NE(possible_modes, 0U);
  ICHECK(analyzer != nullptr);
  PrimExpr zero = make_zero(DataType::UInt(64));
  PayloadPair result{
      fixed_lo.has_value() ? NormalizeMaskPayload(*fixed_lo, analyzer) : zero,
      fixed_hi.has_value() ? NormalizeMaskPayload(*fixed_hi, analyzer) : zero};
  ICHECK(LegalPayloadPair(possible_modes, result.lo, result.hi, analyzer))
      << "Mask payload requirements have no legal canonical completion for "
      << platform_ << ": lo=" << result.lo << ", hi=" << result.hi;
  return result;
}

bool IsSelectedVectorTerminal(const Call &call) {
  if (!call.defined()) {
    return false;
  }
  const auto *op_node = call->op.as<OpNode>();
  if (op_node == nullptr) {
    return false;
  }
  return SelectedTerminalIndex().count(op_node->name) != 0;
}

bool IsVectorMaskSetter(const Call &call) {
  if (!call.defined()) {
    return false;
  }
  return call->op.same_as(ascend_set_mask_mode()) ||
         call->op.same_as(ascend_set_mask_payload());
}

const SelectedVectorTerminalSpec *
SelectedVectorTerminalSpecOf(const Call &call) {
  if (!call.defined()) {
    return nullptr;
  }
  const auto *op_node = call->op.as<OpNode>();
  if (op_node == nullptr) {
    return nullptr;
  }
  auto it = SelectedTerminalIndex().find(op_node->name);
  if (it == SelectedTerminalIndex().end()) {
    return nullptr;
  }
  return &AllSelectedTerminalSpecs()[it->second];
}

const std::vector<SelectedVectorTerminalSpec> &
SelectedVectorTerminalSpecsForBase(const Op &base) {
  static const std::unordered_map<std::string,
                                  std::vector<SelectedVectorTerminalSpec>>
      by_base = [] {
        std::unordered_map<std::string, std::vector<SelectedVectorTerminalSpec>>
            result;
        for (const auto &spec : AllSelectedTerminalSpecs()) {
          result[spec.base->name].push_back(spec);
        }
        return result;
      }();
  static const std::vector<SelectedVectorTerminalSpec> empty;
  auto it = by_base.find(base->name);
  return it == by_base.end() ? empty : it->second;
}

const Op &BaseOperationOf(const Call &selected) {
  const auto *spec = SelectedVectorTerminalSpecOf(selected);
  ICHECK(spec != nullptr)
      << "BaseOperationOf expects a selected Vector terminal, got " << selected;
  return spec->base;
}

MaskContract MaskContractOf(const Call &selected, arith::Analyzer *analyzer) {
  const auto *spec = SelectedVectorTerminalSpecOf(selected);
  ICHECK(spec != nullptr)
      << "MaskContractOf expects a selected Vector terminal, got " << selected;
  ICHECK(analyzer != nullptr);

  PrimExpr normal = ModeExpr(AscendMaskMode::kNormal);
  PrimExpr counter = ModeExpr(AscendMaskMode::kCounter);
  PrimExpr full = FullPayload();
  PrimExpr zero = make_zero(DataType::UInt(64));
  auto neutral = [] {
    return MaskContract{MaskFieldContract::AnyPreserve(),
                        MaskFieldContract::AnyPreserve(),
                        MaskFieldContract::AnyPreserve()};
  };
  auto unknown_all = [] {
    return MaskContract{MaskFieldContract::AnyUnknown(),
                        MaskFieldContract::AnyUnknown(),
                        MaskFieldContract::AnyUnknown()};
  };
  auto normal_full_composite = [&] {
    return MaskContract{MaskFieldContract::ExactUnknown(normal),
                        MaskFieldContract::ExactUnknown(full),
                        MaskFieldContract::ExactUnknown(full)};
  };

  switch (spec->contract_kind) {
  case SelectedMaskContractKind::kRawCounter: {
    ICHECK(!selected->args.empty());
    PrimExpr count = NormalizeMaskPayload(selected->args.back(), analyzer);
    return MaskContract{MaskFieldContract::ExactExact(counter, counter),
                        MaskFieldContract::ExactExact(count, count),
                        MaskFieldContract::AnyPreserve()};
  }
  case SelectedMaskContractKind::kRawNormalDynamicPayload: {
    ICHECK_GE(selected->args.size(), 2U);
    PrimExpr lo = NormalizeMaskPayload(
        selected->args[selected->args.size() - 2], analyzer);
    PrimExpr hi = NormalizeMaskPayload(selected->args.back(), analyzer);
    return MaskContract{MaskFieldContract::ExactExact(normal, normal),
                        MaskFieldContract::ExactExact(lo, lo),
                        MaskFieldContract::ExactExact(hi, hi)};
  }
  case SelectedMaskContractKind::kRawNormalFull:
    return MaskContract{MaskFieldContract::ExactExact(normal, normal),
                        MaskFieldContract::ExactExact(full, full),
                        MaskFieldContract::ExactExact(full, full)};
  case SelectedMaskContractKind::kNeutral:
    return neutral();
  case SelectedMaskContractKind::kUnknownAll:
    return unknown_all();
  case SelectedMaskContractKind::kCompositeNormalFull:
    return normal_full_composite();
  case SelectedMaskContractKind::kCompositeNormalFullToNormalUnknownPayload:
    return MaskContract{MaskFieldContract::ExactExact(normal, normal),
                        MaskFieldContract::ExactUnknown(full),
                        MaskFieldContract::ExactUnknown(full)};
  case SelectedMaskContractKind::kSelfContainedNormalFull:
    return MaskContract{MaskFieldContract::AnyExact(normal),
                        MaskFieldContract::AnyExact(full),
                        MaskFieldContract::AnyExact(full)};
  case SelectedMaskContractKind::kSelfContainedNormalZero:
    return MaskContract{MaskFieldContract::AnyExact(normal),
                        MaskFieldContract::AnyExact(zero),
                        MaskFieldContract::AnyExact(zero)};
  case SelectedMaskContractKind::kSelfContainedNormalDynamicPayload: {
    ICHECK_GE(selected->args.size(), 2U);
    PrimExpr lo = NormalizeMaskPayload(
        selected->args[selected->args.size() - 2], analyzer);
    PrimExpr hi = NormalizeMaskPayload(selected->args.back(), analyzer);
    return MaskContract{MaskFieldContract::AnyExact(normal),
                        MaskFieldContract::AnyExact(lo),
                        MaskFieldContract::AnyExact(hi)};
  }
  }
  LOG(FATAL) << "Unknown selected Vector mask contract kind for "
             << selected->op;
  throw;
}

NonTerminalMaskEffect ClassifyNonTerminalMaskEffect(const Call &call) {
  if (call->op.same_as(ascend_src_code())) {
    return NonTerminalMaskEffect::kBarrier;
  }
  if (IsVectorMaskSetter(call) || IsSelectedVectorTerminal(call)) {
    return NonTerminalMaskEffect::kUnclassified;
  }
  const auto *op_node = call->op.as<OpNode>();
  if (op_node == nullptr) {
    return NonTerminalMaskEffect::kUnclassified;
  }
  Op op = GetRef<Op>(op_node);
  if (!op.same_as(tir::builtin::call_extern()) &&
      !SelectedVectorTerminalSpecsForBase(op).empty()) {
    return NonTerminalMaskEffect::kUnclassified;
  }

  const std::string op_name = op_node->name;
  if (op_name.rfind("tir.", 0) == 0 &&
      !op.same_as(tir::builtin::call_extern())) {
    static const auto call_effect =
        Op::GetAttrMap<tir::TCallEffectKind>("TCallEffectKind");
    if (!call_effect.count(op)) {
      return NonTerminalMaskEffect::kUnclassified;
    }
    auto effect = static_cast<tir::CallEffectKind>(call_effect[op]->value);
    if (effect == tir::CallEffectKind::kExprAnnotation ||
        effect == tir::CallEffectKind::kPure ||
        effect == tir::CallEffectKind::kReadState ||
        effect == tir::CallEffectKind::kSpecialCallArg ||
        effect == tir::CallEffectKind::kEmbedInfo) {
      return NonTerminalMaskEffect::kNeutral;
    }
    return NonTerminalMaskEffect::kUnclassified;
  }
  if (op.same_as(tir::builtin::call_extern())) {
    if (call->args.empty()) {
      return NonTerminalMaskEffect::kUnclassified;
    }
    const auto *name_imm = call->args[0].as<StringImmNode>();
    if (name_imm == nullptr) {
      return NonTerminalMaskEffect::kUnclassified;
    }
    std::string name = name_imm->value;
    size_t namespace_pos = name.find("tl::ascend::");
    if (namespace_pos != std::string::npos) {
      name = name.substr(namespace_pos + 12);
    }
    size_t template_pos = name.find('<');
    if (template_pos != std::string::npos) {
      name = name.substr(0, template_pos);
    }
    if ((name.rfind("copy_", 0) == 0 && name != "copy_ub_to_ub" &&
         name != "copy_ub_to_ub_Nz" && name != "copy_pipe_to_ub_V") ||
        name.rfind("atomic_add_", 0) == 0 || name == "mma" ||
        name == "gemm_v0" || name == "gemm_v1") {
      return NonTerminalMaskEffect::kNeutral;
    }
    return NonTerminalMaskEffect::kUnclassified;
  }

  static const std::unordered_set<std::string> neutral_ops = {
      "tl.ascend_atomic_add",
      "tl.ascend_copy",
      "tl.region",
      "tl.ascend_set_deq_scale",
      "tl.ascend_reinterpretcast",
      "tl.ascend_wait_cross_flag",
      "tl.ascend_set_cross_flag",
      "tl.ascend_set_flag",
      "tl.ascend_wait_flag",
      "tl.ascend_pipe_barrier",
      "tl.ascend_free_pipe",
      "tl.ascend_sync_all",
      "tl.ascend_gemm_v0",
      "tl.ascend_gemm_v1",
      "tl.ascend_printf",
      "tl.ascend_dump_tensor",
      "tl.ascend_auto_barrier",
      "tl.ascend_auto_set_flag",
      "tl.ascend_auto_wait_flag",
      "tl.ascend_auto_set_cross_flag",
      "tl.ascend_auto_wait_cross_flag",
      "tl.ascend_use_swizzle",
      "tl.ascend_mma",
      "tl.ascend_shmem_put_nbi",
      "tl.ascend_shmem_get_nbi",
      "tl.ascend_shmem_ub_put_nbi",
      "tl.ascend_shmem_ub_get_nbi",
      "tl.ascend_copy_cv_experiment",
      "tl.ascend_copy_vc_experiment",
  };
  if (neutral_ops.count(op_name)) {
    return NonTerminalMaskEffect::kNeutral;
  }
  return NonTerminalMaskEffect::kUnclassified;
}

} // namespace tl
} // namespace tvm
