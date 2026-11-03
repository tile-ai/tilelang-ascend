// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#pragma once

#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

#include <limits>

#include "upstream/constraints.h"

namespace tvm::tl::ascend_sync {

using namespace tir;

// Proof-only integer validation. Check every original-width intermediate before
// admitting the mathematical value to an analyzer; never rewrite emitted TIR.
inline bool TryLegacyInteger(const PrimExpr &value, ProofAnalyzer *analyzer,
                             const char **failure) {
  auto reject = [&](const char *message) {
    *failure = message;
    return false;
  };
  if (!value.defined() || SideEffect(value) > CallEffectKind::kPure)
    return reject("mutable or opaque expressions are unsupported");
  const DataType dtype = value.dtype();
  if (!dtype.is_scalar() || dtype.is_bool() ||
      !((dtype.is_int() && dtype.bits() <= 64) ||
        (dtype.is_uint() && dtype.bits() <= 32)))
    return reject("requires a value-preserving scalar index");
  if (ContainsUnsignedWrappingArithmetic(value))
    return reject("wrapping unsigned index arithmetic is unsupported");
  bool valid = true;
  PostOrderVisit(value, [&](const ObjectRef &node) {
    if (!valid)
      return;
    const auto *expression = node.as<PrimExprNode>();
    if (!expression || expression->dtype.is_bool() ||
        (!expression->dtype.is_int() && !expression->dtype.is_uint()))
      return;
    const DataType type = expression->dtype;
    const int bits = type.bits();
    if (!type.is_scalar() || bits > (type.is_uint() ? 32 : 64)) {
      valid = reject("requires a value-preserving scalar index");
      return;
    }
    PrimExpr divisor;
    if (const auto *operation = node.as<FloorDivNode>())
      divisor = operation->b;
    else if (const auto *operation = node.as<DivNode>())
      divisor = operation->b;
    else if (const auto *operation = node.as<FloorModNode>())
      divisor = operation->b;
    else if (const auto *operation = node.as<ModNode>())
      divisor = operation->b;
    if (divisor.defined() &&
        !analyzer->z3_prover.CanProve(divisor != make_zero(divisor.dtype()))) {
      valid = reject("index division requires a nonzero divisor");
      return;
    }
    const int64_t minimum =
        type.is_uint() ? 0
                       : (bits == 64 ? std::numeric_limits<int64_t>::min()
                                     : -(int64_t(1) << (bits - 1)));
    const int64_t maximum =
        type.is_uint() ? (int64_t(1) << bits) - 1
                       : (bits == 64 ? std::numeric_limits<int64_t>::max()
                                     : (int64_t(1) << (bits - 1)) - 1);
    if (const auto *conversion = node.as<CastNode>()) {
      const DataType source_type = conversion->value.dtype();
      if (!source_type.is_scalar() ||
          !((source_type.is_int() && source_type.bits() <= 64) ||
            (source_type.is_uint() && source_type.bits() <= 32))) {
        valid = reject("index casts require integer source values");
        return;
      }
      const PrimExpr source = cast(DataType::Int(64), conversion->value);
      if (!analyzer->z3_prover.CanProve(
              source >= IntImm(DataType::Int(64), minimum) &&
              source <= IntImm(DataType::Int(64), maximum))) {
        valid = reject("lossy index casts are unsupported");
        return;
      }
    }
    const PrimExpr wide = cast(DataType::Int(64), GetRef<PrimExpr>(expression));
    // Avoid the legacy folder, which can wrap narrow constant intermediates.
    if (!analyzer->z3_prover.CanProve(
            wide >= IntImm(DataType::Int(64), minimum) &&
            wide <= IntImm(DataType::Int(64), maximum)))
      valid = reject("cannot prove nonwrapping signed index arithmetic");
  });
  return valid;
}

inline bool TryLegacyPredicate(const PrimExpr &predicate,
                               ProofAnalyzer *analyzer, const char **failure) {
  if (!predicate.defined() || !predicate.dtype().is_scalar() ||
      !predicate.dtype().is_bool() ||
      SideEffect(predicate) > CallEffectKind::kPure) {
    *failure = "assumptions require a pure scalar boolean predicate";
    return false;
  }
  bool valid = true;
  PostOrderVisit(predicate, [&](const ObjectRef &node) {
    const auto *expression = node.as<PrimExprNode>();
    if (valid && expression && !expression->dtype.is_bool() &&
        (expression->dtype.is_int() || expression->dtype.is_uint()))
      valid = TryLegacyInteger(GetRef<PrimExpr>(expression), analyzer, failure);
  });
  return valid;
}

// Replay definitions before using their mathematical values. In particular a
// later assumption cannot justify an earlier overflowing Let or loop bound.
inline bool TryPopulateLegacyContext(const ConstrSet &context,
                                     ProofAnalyzer *analyzer,
                                     const char **failure) {
  for (const Constr &constraint : context.constrs_) {
    if (constraint.kind == Constr::kBindRange) {
      if (!TryLegacyInteger(constraint.range->min, analyzer, failure) ||
          !TryLegacyInteger(constraint.range->extent, analyzer, failure) ||
          !TryLegacyInteger(
              Add(constraint.range->min, constraint.range->extent), analyzer,
              failure))
        return false;
    } else if (constraint.value.dtype().is_bool()) {
      if (!TryLegacyPredicate(constraint.value, analyzer, failure))
        return false;
    } else if (!TryLegacyInteger(constraint.value, analyzer, failure)) {
      return false;
    }
    constraint.Populate(*analyzer);
  }
  return true;
}

} // namespace tvm::tl::ascend_sync
