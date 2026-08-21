// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*! \file ascend_vector_mask_legalize.cc
 *  \brief Insert the minimum required Ascend Vector mask repairs.
 */

#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/node/structural_equal.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/ascend.h"
#include "common/ascend_vector_mask.h"
#include "common/operation_config.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

static constexpr const char *ascendVectorMaskReuse =
    "tl.ascend_vector_mask_reuse";

TVM_REGISTER_PASS_CONFIG_OPTION(ascendVectorMaskReuse, Bool);

namespace {

enum class AscendMaskMode : int32_t { kNormal = 0, kCounter = 1 };
enum class MaskRequirement : uint8_t { kAny, kExact };
enum class MaskEnsure : uint8_t { kPreserve, kExact, kUnknown };

struct MaskFieldContract {
  MaskRequirement requirement{MaskRequirement::kAny};
  MaskEnsure ensure{MaskEnsure::kPreserve};
  PrimExpr required;
  PrimExpr ensured;
};

struct MaskContract {
  MaskFieldContract mode;
  MaskFieldContract lo;
  MaskFieldContract hi;
};

struct MaskFacts {
  std::optional<AscendMaskMode> mode;
  std::optional<PrimExpr> lo;
  std::optional<PrimExpr> hi;
};

MaskFieldContract AnyExact(PrimExpr value) {
  return {MaskRequirement::kAny, MaskEnsure::kExact, PrimExpr(),
          std::move(value)};
}

MaskFieldContract AnyUnknown() {
  return {MaskRequirement::kAny, MaskEnsure::kUnknown, PrimExpr(), PrimExpr()};
}

MaskFieldContract ExactExact(PrimExpr value) {
  return {MaskRequirement::kExact, MaskEnsure::kExact, value, std::move(value)};
}

PrimExpr ModeValue(AscendMaskMode mode) {
  return IntImm(DataType::Int(32), static_cast<int32_t>(mode));
}

PrimExpr FullPayload() {
  return make_const(DataType::UInt(64), std::numeric_limits<uint64_t>::max());
}

PrimExpr ZeroPayload() { return make_zero(DataType::UInt(64)); }

bool UsesHighWord(DataType dtype) {
  return !dtype.is_void() && dtype.bits() < 32;
}

MaskContract NormalFullRequirement(DataType dtype) {
  MaskContract result{ExactExact(ModeValue(AscendMaskMode::kNormal)),
                      ExactExact(FullPayload()),
                      {}};
  if (UsesHighWord(dtype))
    result.hi = ExactExact(FullPayload());
  return result;
}

MaskContract RawNormalRequirement(DataType dtype, PrimExpr lo, PrimExpr hi) {
  MaskContract result{ExactExact(ModeValue(AscendMaskMode::kNormal)),
                      ExactExact(std::move(lo)),
                      {}};
  if (UsesHighWord(dtype))
    result.hi = ExactExact(std::move(hi));
  return result;
}

MaskContract NormalFullPostState() {
  return {AnyExact(ModeValue(AscendMaskMode::kNormal)), AnyExact(FullPayload()),
          AnyExact(FullPayload())};
}

MaskContract UnknownPostState(MaskContract result) {
  result.mode.ensure = MaskEnsure::kUnknown;
  result.lo.ensure = MaskEnsure::kUnknown;
  result.hi.ensure = MaskEnsure::kUnknown;
  return result;
}

bool IsSameTypeCopy(const SelectedCallView &selected) {
  const Array<PrimExpr> &args = selected.semantic_args();
  ICHECK_GE(args.size(), 3U);
  return VectorAccessPtrDtype(args[1]) == VectorAccessPtrDtype(args[2]);
}

enum class RuntimeStridedCopyEffect : uint8_t {
  kNotApplicable,
  kPreserve,
  kNormalFull,
  kPathDependent,
};

RuntimeStridedCopyEffect
ClassifyRuntimeStridedCopy(const SelectedCallView &selected,
                           arith::Analyzer *analyzer) {
  const Array<PrimExpr> &args = selected.semantic_args();
  if (!selected.semantic_spec().base.same_as(tir::builtin::call_extern()) ||
      args.size() != 9 || IsSameTypeCopy(selected)) {
    return RuntimeStridedCopyEffect::kNotApplicable;
  }
  PrimExpr contiguous = (args[4] == args[5]) && (args[7] == args[8]);
  if (analyzer->CanProve(contiguous) || analyzer->CanProve(args[3] > 0)) {
    return RuntimeStridedCopyEffect::kNormalFull;
  }
  if (analyzer->CanProve(!contiguous) && analyzer->CanProve(args[3] == 0)) {
    return RuntimeStridedCopyEffect::kPreserve;
  }
  return RuntimeStridedCopyEffect::kPathDependent;
}

MaskContract ContractOf(const Call &call, arith::Analyzer *analyzer) {
  SelectedCallView selected(call);
  const AscendVectorSemanticOpSpec &semantic = selected.semantic_spec();
  const AscendVectorTerminalVariant &variant = selected.variant();
  const Array<PrimExpr> &semantic_args = selected.semantic_args();
  DataType dtype = selected.vector_dtype();
  auto explicit_mask = [&]() -> std::pair<PrimExpr, PrimExpr> {
    if (variant.payload == PayloadLayout::kMask) {
      return {selected.mask_lo(), selected.mask_hi()};
    }
    if (semantic.base.same_as(ascend_bilinear_interpolation())) {
      const auto *lanes = analyzer->Simplify(semantic_args[4]).as<IntImmNode>();
      ICHECK(lanes && lanes->value >= 0 && lanes->value <= 128);
      return NormalMaskBits(lanes->value);
    }
    ICHECK(semantic.base.same_as(ascend_gather_mask_experiment()));
    return {analyzer->Simplify(semantic_args[5]), ZeroPayload()};
  };
  MaskContract contract;
  if (semantic.base.same_as(tir::builtin::call_extern())) {
    return IsSameTypeCopy(selected) ? contract : NormalFullPostState();
  }
  if (semantic.base.same_as(ascend_gather_mask())) {
    if (!semantic_args[3].as<CallNode>()) {
      return MaskContract{AnyExact(ModeValue(AscendMaskMode::kNormal)),
                          AnyExact(ZeroPayload()), AnyExact(ZeroPayload())};
    }
    return UnknownPostState(NormalFullRequirement(dtype));
  }
  if (variant.emitter != EmitterFamily::kHelper) {
    switch (variant.payload) {
    case PayloadLayout::kCount:
      return {ExactExact(ModeValue(AscendMaskMode::kCounter)),
              ExactExact(analyzer->Simplify(selected.count())),
              {}};
    case PayloadLayout::kMaskSpec: {
      const auto *mode = selected.mask_mode().as<IntImmNode>();
      ICHECK(mode);
      PrimExpr lo = analyzer->Simplify(selected.mask_lo());
      if (mode->value == static_cast<int32_t>(AscendMaskMode::kCounter)) {
        return {ExactExact(ModeValue(AscendMaskMode::kCounter)),
                ExactExact(lo),
                {}};
      }
      return RawNormalRequirement(dtype, lo,
                                  analyzer->Simplify(selected.mask_hi()));
    }
    case PayloadLayout::kMask:
      return RawNormalRequirement(dtype, analyzer->Simplify(selected.mask_lo()),
                                  analyzer->Simplify(selected.mask_hi()));
    case PayloadLayout::kNone:
      if (semantic_args.size() == 5) {
        return MaskContract{ExactExact(ModeValue(AscendMaskMode::kNormal)),
                            AnyExact(FullPayload()), AnyExact(FullPayload())};
      }
      return NormalFullRequirement(dtype);
    }
    LOG(FATAL) << "Unknown raw Ascend Vector mask payload";
  }
  switch (variant.helper_contract) {
  case ContractRecipe::kNeutral:
    return contract;
  case ContractRecipe::kUnknown:
    return MaskContract{AnyUnknown(), AnyUnknown(), AnyUnknown()};
  case ContractRecipe::kPayloadFull:
    // dav-c220 Gatherb and Brcb call ResetMask(): mode is preserved while
    // both payload words become full.
    return MaskContract{{}, AnyExact(FullPayload()), AnyExact(FullPayload())};
  case ContractRecipe::kCompositeNormalFullUnknown:
    return UnknownPostState(NormalFullRequirement(dtype));
  case ContractRecipe::kCompositeNormalFullToNormalUnknownPayload:
    contract = NormalFullRequirement(dtype);
    contract.mode.ensure = MaskEnsure::kExact;
    contract.mode.ensured = ModeValue(AscendMaskMode::kNormal);
    contract.lo.ensure = MaskEnsure::kUnknown;
    contract.hi.ensure = MaskEnsure::kUnknown;
    return contract;
  case ContractRecipe::kSelfContainedNormalFull:
    return NormalFullPostState();
  case ContractRecipe::kSelfContainedNormalExplicit: {
    auto [mask_lo, mask_hi] = explicit_mask();
    PrimExpr lo = analyzer->Simplify(mask_lo);
    PrimExpr hi = analyzer->Simplify(mask_hi);
    // These self-contained helpers overwrite the complete architectural mask
    // state. Record the high word even when the current operation uses b32:
    // a following b16 terminal observes that same global register state.
    contract = {AnyExact(ModeValue(AscendMaskMode::kNormal)), AnyExact(lo),
                AnyExact(hi)};
    return contract;
  }
  case ContractRecipe::kCreateVecIndex: {
    const auto *count = semantic_args.back().as<IntImmNode>();
    if (count == nullptr || dtype.is_void()) {
      return MaskContract{AnyUnknown(), AnyUnknown(), AnyUnknown()};
    }
    // The one-block path uses scalar stores. Larger constants enter a
    // count-form Adds helper which restores NORMAL and both full words.
    if (count->value <= 32 / dtype.bytes()) {
      return contract;
    }
    return NormalFullPostState();
  }
  case ContractRecipe::kGatherCount: {
    // Gather programs a NORMAL payload for its element count, but its dav-c220
    // Level-0 path does not switch the mask mode.  Require NORMAL on entry and
    // publish the payload that Gather leaves behind.
    const auto *count = semantic_args.back().as<IntImmNode>();
    if (count == nullptr || dtype.is_void() || count->value <= 0) {
      return {ExactExact(ModeValue(AscendMaskMode::kNormal)), AnyUnknown(),
              AnyUnknown()};
    }
    int64_t lanes_per_repeat = dtype.bits() == 16 ? 128 : 64;
    int64_t final_lanes = count->value % lanes_per_repeat;
    if (final_lanes == 0) {
      final_lanes = lanes_per_repeat;
    }
    auto [lo, hi] = NormalMaskBits(final_lanes);
    return {ExactExact(ModeValue(AscendMaskMode::kNormal)), AnyExact(lo),
            AnyExact(hi)};
  }
  case ContractRecipe::kNormalExplicitLowArg:
    ICHECK_GE(semantic_args.size(), 4U);
    // Fill_experiment's dav-c220 Level-0 Duplicate overload writes its own
    // two-word mask to {mask0, 0}, but does not switch the mask mode. Require
    // NORMAL on entry and publish the exact state that it leaves behind.
    contract.mode = ExactExact(ModeValue(AscendMaskMode::kNormal));
    contract.lo = AnyExact(analyzer->Simplify(semantic_args[3]));
    contract.hi = AnyExact(ZeroPayload());
    return contract;
  }
  LOG(FATAL) << "Unknown Ascend Vector mask contract recipe";
  throw;
}

bool CallMayAffectVectorMask(const Call &call) {
  if (IsVectorMaskSetter(call)) {
    return true;
  }
  if (call->op.same_as(ascend_pipe_barrier()) ||
      call->op.same_as(ascend_sync_all()) ||
      call->op.same_as(ascend_set_flag()) ||
      call->op.same_as(ascend_wait_flag()) ||
      call->op.same_as(ascend_set_cross_flag()) ||
      call->op.same_as(ascend_wait_cross_flag()) ||
      call->op.same_as(ascend_auto_barrier()) ||
      call->op.same_as(ascend_auto_set_flag()) ||
      call->op.same_as(ascend_auto_wait_flag()) ||
      call->op.same_as(ascend_auto_set_cross_flag()) ||
      call->op.same_as(ascend_auto_wait_cross_flag())) {
    return false;
  }
  if (IsSelectedVectorTerminal(call)) {
    SelectedCallView selected(call);
    if (selected.variant().emitter != EmitterFamily::kHelper)
      return true;
    if (selected.semantic_spec().base.same_as(tir::builtin::call_extern()))
      return !IsSameTypeCopy(selected);
    if (selected.semantic_spec().base.same_as(ascend_gather_mask()))
      return true;
    return selected.variant().helper_contract != ContractRecipe::kNeutral;
  }
  if (call->op.same_as(tir::builtin::call_extern())) {
    const auto *name =
        call->args.empty() ? nullptr : call->args[0].as<StringImmNode>();
    if (name == nullptr) {
      return true;
    }
    std::string value = name->value;
    // GM-to-UB copies may fill a runtime tail with AscendC::Duplicate.  The
    // full-tile path is MTE2-only, while the padding path leaves NORMAL/full
    // mask state, so the call is not unconditionally mask-neutral.
    if (value.find("copy_gm_to_ub") != std::string::npos) {
      return true;
    }
    if (value.find("copy_") != std::string::npos ||
        value.find("atomic_add_") != std::string::npos ||
        value.find("mma") != std::string::npos) {
      return false;
    }
    // Unknown external helpers are opaque. They may contain count-form
    // Vector APIs that rewrite mode/payload, so preserve facts only for the
    // explicitly neutral data-movement/Cube allowlist above.
    return true;
  }
  const auto *op = call->op.as<OpNode>();
  if (op == nullptr || op->name == "tl.ascend_src_code") {
    return true;
  }
  auto it = GetOperationConfig().find(op->name);
  if (it != GetOperationConfig().end()) {
    return it->second.default_pipeline == "PIPE_V";
  }
  return std::string(op->name).rfind("tl.ascend_", 0) == 0;
}

class MaskEffectDetector final : public StmtExprVisitor {
  void VisitExpr_(const CallNode *op) final {
    found |= CallMayAffectVectorMask(GetRef<Call>(op));
    if (!found) {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

public:
  bool found{false};
};

} // namespace

class AscendVectorMaskLegalizer final : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc func, Target target, std::string platform,
                          bool reuse_mask) {
    if (!UseCompilerManagedVectorMask(target, platform)) {
      return func;
    }
    AscendVectorMaskLegalizer legalizer(reuse_mask);
    PrimFuncNode *copy = func.CopyOnWrite();
    copy->body = legalizer.VisitStmt(func->body);
    return func;
  }

private:
  explicit AscendVectorMaskLegalizer(bool reuse_mask)
      : reuse_mask_(reuse_mask) {}

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *node = op->value.as<CallNode>();
    if (node == nullptr || vector_scope_depth_ == 0) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Call call = GetRef<Call>(node);
    ICHECK(!IsVectorMaskSetter(call))
        << "Vector-mask setter appeared before legalization: " << call;
    if (IsSelectedVectorTerminal(call)) {
      if (!reuse_mask_) {
        facts_ = {};
      }
      SelectedCallView selected(call);
      RuntimeStridedCopyEffect copy_effect =
          ClassifyRuntimeStridedCopy(selected, &analyzer_);
      MaskContract contract = ContractOf(call, &analyzer_);
      std::vector<Stmt> sequence;
      Repair(contract, &sequence);
      sequence.push_back(GetRef<Stmt>(op));
      if (!reuse_mask_) {
        facts_ = {};
      } else if (copy_effect == RuntimeStridedCopyEffect::kPathDependent) {
        // The runtime-strided helper either executes at least one count-form
        // Cast and leaves NORMAL/full, or executes zero loop iterations and
        // preserves the incoming state. Keep only facts true on both paths.
        MaskFacts cast_post{AscendMaskMode::kNormal, FullPayload(),
                            FullPayload()};
        facts_ = Meet(facts_, cast_post);
      } else if (copy_effect != RuntimeStridedCopyEffect::kPreserve) {
        Apply(contract);
      }
      return SeqStmt::Flatten(sequence);
    }
    ICHECK(!RequiresSelectedVectorTerminal(call))
        << "Unselected compiler-managed Vector operation reached mask "
           "legalization: "
        << call;
    if (IsGmToUbCopy(call)) {
      // At runtime this helper either performs a pure MTE2 copy and preserves
      // the incoming state, or pads through count-form Duplicate and leaves
      // NORMAL/full.  Keep only facts that hold on both paths.
      MaskFacts padded{AscendMaskMode::kNormal, FullPayload(), FullPayload()};
      facts_ = Meet(facts_, padded);
    } else if (call->op.same_as(ascend_src_code()) ||
               CallMayAffectVectorMask(call)) {
      facts_ = {};
    }
    return GetRef<Stmt>(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);
    MaskFacts incoming = facts_;
    facts_ = incoming;
    Stmt then_case = VisitStmt(op->then_case);
    MaskFacts then_out = facts_;
    facts_ = incoming;
    Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      else_case = VisitStmt(op->else_case.value());
    }
    MaskFacts else_out = facts_;
    facts_ = Meet(then_out, else_out);
    return IfThenElse(condition, then_case, else_case, op->span);
  }

  template <typename LoopNode> Stmt VisitEffectfulLoop(const LoopNode *op) {
    MaskEffectDetector detector;
    detector(op->body);
    if (!detector.found) {
      return GetRef<Stmt>(op);
    }
    facts_ = {};
    Stmt result = StmtExprMutator::VisitStmt_(op);
    facts_ = {};
    return result;
  }

  Stmt VisitStmt_(const ForNode *op) final { return VisitEffectfulLoop(op); }

  Stmt VisitStmt_(const WhileNode *op) final { return VisitEffectfulLoop(op); }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    Stmt body = VisitStmt(op->body);
    DropFactsUsing(op->var);
    return LetStmt(op->var, value, body, op->span);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    Stmt result = StmtExprMutator::VisitStmt_(op);
    for (const IterVar &iter : op->iter_vars) {
      DropFactsUsing(iter->var);
    }
    return result;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != "resource_scope") {
      return StmtExprMutator::VisitStmt_(op);
    }
    const auto *scope = op->value.as<IntImmNode>();
    ICHECK(scope && (scope->value == 0 || scope->value == 1));
    if (scope->value == 0) {
      return GetRef<Stmt>(op);
    }
    bool outermost = vector_scope_depth_++ == 0;
    if (outermost) {
      facts_ = {};
    }
    Stmt body = VisitStmt(op->body);
    --vector_scope_depth_;
    if (outermost) {
      facts_ = {};
    }
    return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
  }

  bool Equal(const std::optional<PrimExpr> &known, const PrimExpr &required) {
    if (!known.has_value()) {
      return false;
    }
    PrimExpr lhs = analyzer_.Simplify(*known);
    PrimExpr rhs = analyzer_.Simplify(required);
    return StructuralEqual()(lhs, rhs) || analyzer_.CanProveEqual(lhs, rhs);
  }

  bool IsGmToUbCopy(const Call &call) const {
    if (!call->op.same_as(tir::builtin::call_extern()) || call->args.empty()) {
      return false;
    }
    const auto *name = call->args[0].as<StringImmNode>();
    if (name == nullptr) {
      return false;
    }
    std::string value = name->value;
    return value.find("copy_gm_to_ub") != std::string::npos;
  }

  bool ModeEqual(const std::optional<AscendMaskMode> &known,
                 const MaskFieldContract &field) {
    if (!known.has_value()) {
      return false;
    }
    const auto *value = field.required.as<IntImmNode>();
    ICHECK(value);
    return static_cast<int32_t>(*known) == value->value;
  }

  void Repair(const MaskContract &contract, std::vector<Stmt> *sequence) {
    if (contract.mode.requirement == MaskRequirement::kExact &&
        !ModeEqual(facts_.mode, contract.mode)) {
      sequence->push_back(
          Evaluate(Call(DataType::Handle(), ascend_set_mask_mode(),
                        {contract.mode.required})));
      facts_.mode = static_cast<AscendMaskMode>(
          Downcast<IntImm>(contract.mode.required)->value);
    }
    bool repair_lo = contract.lo.requirement == MaskRequirement::kExact &&
                     !Equal(facts_.lo, contract.lo.required);
    bool repair_hi = contract.hi.requirement == MaskRequirement::kExact &&
                     !Equal(facts_.hi, contract.hi.required);
    if (!repair_lo && !repair_hi) {
      return;
    }
    // A setter always writes both payload words.  Use a deterministic zero for
    // a sibling that the consumer does not constrain (COUNTER and b32 NORMAL)
    // instead of leaking an unrelated fact from the previous terminal.
    PrimExpr lo = contract.lo.requirement == MaskRequirement::kExact
                      ? analyzer_.Simplify(contract.lo.required)
                      : ZeroPayload();
    PrimExpr hi = contract.hi.requirement == MaskRequirement::kExact
                      ? analyzer_.Simplify(contract.hi.required)
                      : ZeroPayload();
    sequence->push_back(Evaluate(
        Call(DataType::Handle(), ascend_set_mask_payload(), {lo, hi})));
    facts_.lo = lo;
    facts_.hi = hi;
  }

  void ApplyField(const MaskFieldContract &field,
                  std::optional<PrimExpr> *fact) {
    if (field.ensure == MaskEnsure::kExact) {
      *fact = analyzer_.Simplify(field.ensured);
    } else if (field.ensure == MaskEnsure::kUnknown) {
      fact->reset();
    }
  }

  void Apply(const MaskContract &contract) {
    if (contract.mode.ensure == MaskEnsure::kExact) {
      facts_.mode = static_cast<AscendMaskMode>(
          Downcast<IntImm>(contract.mode.ensured)->value);
    } else if (contract.mode.ensure == MaskEnsure::kUnknown) {
      facts_.mode.reset();
    }
    ApplyField(contract.lo, &facts_.lo);
    ApplyField(contract.hi, &facts_.hi);
  }

  MaskFacts Meet(const MaskFacts &lhs, const MaskFacts &rhs) {
    MaskFacts result;
    if (lhs.mode.has_value() && lhs.mode == rhs.mode) {
      result.mode = lhs.mode;
    }
    if (lhs.lo.has_value() && rhs.lo.has_value() && Equal(lhs.lo, *rhs.lo)) {
      result.lo = lhs.lo;
    }
    if (lhs.hi.has_value() && rhs.hi.has_value() && Equal(lhs.hi, *rhs.hi)) {
      result.hi = lhs.hi;
    }
    return result;
  }

  void DropFactsUsing(const Var &var) {
    auto uses = [&](const std::optional<PrimExpr> &fact) {
      return fact.has_value() &&
             UsesVar(*fact, [node = var.get()](const VarNode *candidate) {
               return candidate == node;
             });
    };
    if (uses(facts_.lo)) {
      facts_.lo.reset();
    }
    if (uses(facts_.hi)) {
      facts_.hi.reset();
    }
  }

  const bool reuse_mask_;
  int vector_scope_depth_{0};
  MaskFacts facts_;
  arith::Analyzer analyzer_;
};

tvm::transform::Pass AscendVectorMaskLegalize(Target target,
                                              std::string platform) {
  auto pass_func = [=](PrimFunc func, IRModule, PassContext ctx) {
    bool reuse_mask =
        ctx->GetConfig<Bool>(ascendVectorMaskReuse, Bool(true)).value();
    return AscendVectorMaskLegalizer::Rewrite(std::move(func), target, platform,
                                              reuse_mask);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendVectorMaskLegalize", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendVectorMaskLegalize")
    .set_body_typed(AscendVectorMaskLegalize);

} // namespace tl
} // namespace tvm
