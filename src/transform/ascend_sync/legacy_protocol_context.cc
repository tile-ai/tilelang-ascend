// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#include "legacy_protocol_context.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm::tl::ascend_sync {

using namespace tir;

namespace {

using Constants = LegacyProtocolConstants;
// Preserve the existing diagnostic text while sharing the evaluator.
constexpr const char *kDiagnostic = "Legacy manual event normalization: ";

// Do not let TVM constant folding turn an overflowing ID/bound intermediate
// into an apparently valid event slot. First evaluate a small integer-only
// subset with a wider checked accumulator; only then ask the TVM simplifier to
// confirm the constant result. Unknown values remain unknown, never zero.
class CheckedInteger {
public:
  explicit CheckedInteger(const Constants &constants) : constants_(constants) {}

  std::optional<int64_t> Evaluate(const PrimExpr &value) const {
    const DataType dtype = value.dtype();
    if (!dtype.is_scalar() || !((dtype.is_int() && dtype.bits() <= 64) ||
                                (dtype.is_uint() && dtype.bits() <= 32)))
      return std::nullopt;
    const auto checked = EvaluateImpl(value);
    if (!checked.has_value())
      return std::nullopt;
    ICHECK(Fits(checked.value(), dtype))
        << kDiagnostic << "integer overflow in protocol expression: " << value;
    return static_cast<int64_t>(checked.value());
  }

private:
  static bool Fits(__int128 value, DataType dtype) {
    const __int128 minimum =
        dtype.is_int() ? -(__int128(1) << (dtype.bits() - 1)) : 0;
    const __int128 maximum = dtype.is_int()
                                 ? (__int128(1) << (dtype.bits() - 1)) - 1
                                 : (__int128(1) << dtype.bits()) - 1;
    return value >= minimum && value <= maximum;
  }

  std::optional<__int128> EvaluateImpl(const PrimExpr &value) const {
    if (const auto *literal = value.as<IntImmNode>())
      return literal->value;
    if (const auto *variable = value.as<VarNode>()) {
      const auto found = constants_.find(GetRef<Var>(variable));
      if (found != constants_.end())
        return found->second;
      return std::nullopt;
    }
    if (const auto *conversion = value.as<CastNode>()) {
      const auto source = Evaluate(conversion->value);
      return source.has_value() ? std::optional<__int128>(source.value())
                                : std::nullopt;
    }
    if (const auto *selection = value.as<SelectNode>()) {
      const auto condition = Evaluate(selection->condition);
      const auto true_value = Evaluate(selection->true_value);
      const auto false_value = Evaluate(selection->false_value);
      if (!condition.has_value() || !true_value.has_value() ||
          !false_value.has_value())
        return std::nullopt;
      const auto selected = condition.value() ? true_value : false_value;
      return selected.has_value() ? std::optional<__int128>(selected.value())
                                  : std::nullopt;
    }
    if (const auto *call = value.as<CallNode>()) {
      if (!call->op.same_as(builtin::if_then_else()) || call->args.size() != 3)
        return std::nullopt;
      const auto condition = Evaluate(call->args[0]);
      if (!condition.has_value())
        return std::nullopt;
      const auto selected = Evaluate(call->args[condition.value() ? 1 : 2]);
      return selected.has_value() ? std::optional<__int128>(selected.value())
                                  : std::nullopt;
    }
    if (const auto *negation = value.as<NotNode>()) {
      const auto operand = Evaluate(negation->a);
      return operand.has_value() ? std::optional<__int128>(!operand.value())
                                 : std::nullopt;
    }
    PrimExpr left, right;
    enum class Kind {
      kAdd,
      kSub,
      kMul,
      kDiv,
      kMod,
      kFloorDiv,
      kFloorMod,
      kMin,
      kMax,
      kLT,
      kLE,
      kGT,
      kGE,
      kEQ,
      kNE,
      kAnd,
      kOr
    };
    std::optional<Kind> kind;
#define TL_PROTOCOL_INTEGER_NODE(Type, Tag)                                    \
  if (const auto *operation = value.as<Type##Node>()) {                        \
    left = operation->a;                                                       \
    right = operation->b;                                                      \
    kind = Kind::Tag;                                                          \
  }
    TL_PROTOCOL_INTEGER_NODE(Add, kAdd)
    TL_PROTOCOL_INTEGER_NODE(Sub, kSub)
    TL_PROTOCOL_INTEGER_NODE(Mul, kMul)
    TL_PROTOCOL_INTEGER_NODE(Div, kDiv)
    TL_PROTOCOL_INTEGER_NODE(Mod, kMod)
    TL_PROTOCOL_INTEGER_NODE(FloorDiv, kFloorDiv)
    TL_PROTOCOL_INTEGER_NODE(FloorMod, kFloorMod)
    TL_PROTOCOL_INTEGER_NODE(Min, kMin)
    TL_PROTOCOL_INTEGER_NODE(Max, kMax)
    TL_PROTOCOL_INTEGER_NODE(LT, kLT)
    TL_PROTOCOL_INTEGER_NODE(LE, kLE)
    TL_PROTOCOL_INTEGER_NODE(GT, kGT)
    TL_PROTOCOL_INTEGER_NODE(GE, kGE)
    TL_PROTOCOL_INTEGER_NODE(EQ, kEQ)
    TL_PROTOCOL_INTEGER_NODE(NE, kNE)
    TL_PROTOCOL_INTEGER_NODE(And, kAnd)
    TL_PROTOCOL_INTEGER_NODE(Or, kOr)
#undef TL_PROTOCOL_INTEGER_NODE
    if (!kind.has_value())
      return std::nullopt;
    const auto a = Evaluate(left), b = Evaluate(right);
    if (!a.has_value() || !b.has_value())
      return std::nullopt;
    const __int128 x = a.value(), y = b.value();
    __int128 result;
    switch (kind.value()) {
    case Kind::kAdd:
      result = x + y;
      break;
    case Kind::kSub:
      result = x - y;
      break;
    case Kind::kMul:
      result = x * y;
      break;
    case Kind::kDiv:
    case Kind::kMod:
    case Kind::kFloorDiv:
    case Kind::kFloorMod: {
      ICHECK(y != 0) << kDiagnostic << "zero divisor in protocol expression";
      __int128 quotient = x / y;
      const __int128 remainder = x % y;
      if ((kind == Kind::kFloorDiv || kind == Kind::kFloorMod) &&
          remainder != 0 && ((x < 0) != (y < 0)))
        --quotient;
      ICHECK(Fits(quotient, left.dtype()))
          << kDiagnostic << "integer division overflow in protocol expression";
      result = kind == Kind::kDiv || kind == Kind::kFloorDiv ? quotient
                                                             : x - quotient * y;
      break;
    }
    case Kind::kMin:
      result = x < y ? x : y;
      break;
    case Kind::kMax:
      result = x > y ? x : y;
      break;
    case Kind::kLT:
      result = x < y;
      break;
    case Kind::kLE:
      result = x <= y;
      break;
    case Kind::kGT:
      result = x > y;
      break;
    case Kind::kGE:
      result = x >= y;
      break;
    case Kind::kEQ:
      result = x == y;
      break;
    case Kind::kNE:
      result = x != y;
      break;
    case Kind::kAnd:
      result = x && y;
      break;
    case Kind::kOr:
      result = x || y;
      break;
    }
    ICHECK(Fits(result, value.dtype()))
        << kDiagnostic << "integer overflow in protocol expression: " << value;
    // The operands above are already representable in their original types.
    // Substituting only those checked values is safe for this one operation.
    Map<Var, PrimExpr> bindings;
    for (const auto &[variable, constant] : constants_)
      bindings.Set(variable, make_const(variable.dtype(), constant));
    arith::Analyzer analyzer;
    const auto simplified = analyzer.Simplify(Substitute(value, bindings));
    const auto *literal = simplified.as<IntImmNode>();
    if (!literal || literal->value != result)
      return std::nullopt;
    return result;
  }

  const Constants &constants_;
};

} // namespace

std::optional<int64_t>
EvaluateLegacyProtocolInteger(const PrimExpr &value,
                              const LegacyProtocolConstants &constants) {
  return CheckedInteger(constants).Evaluate(value);
}

LegacyProtocolContext::Binding::Binding(LegacyProtocolContext *context,
                                        const Var &variable,
                                        std::optional<int64_t> value)
    : context_(context), variable_(variable) {
  const auto previous = context_->constants_.find(variable_);
  if (previous != context_->constants_.end())
    previous_ = previous->second;
  Set(value);
}

LegacyProtocolContext::Binding::~Binding() { Set(previous_); }

void LegacyProtocolContext::Binding::Set(std::optional<int64_t> value) {
  if (value.has_value())
    context_->constants_[variable_] = value.value();
  else
    context_->constants_.erase(variable_);
}

LegacyProtocolContext::OwnerScope::OwnerScope(LegacyProtocolContext *context,
                                              LegacyProtocolOwner owner)
    : context_(context), previous_(context->owner_) {
  context_->owner_ = owner;
}

LegacyProtocolContext::OwnerScope::~OwnerScope() {
  context_->owner_ = previous_;
}

std::optional<int64_t>
LegacyProtocolContext::Evaluate(const PrimExpr &value) const {
  return EvaluateLegacyProtocolInteger(value, constants_);
}

LegacyProtocolContext::Binding
LegacyProtocolContext::Bind(const Var &variable, std::optional<int64_t> value) {
  return Binding(this, variable, value);
}

LegacyProtocolContext::OwnerScope
LegacyProtocolContext::EnterOwner(LegacyProtocolOwner owner) {
  return OwnerScope(this, owner);
}

} // namespace tvm::tl::ascend_sync
