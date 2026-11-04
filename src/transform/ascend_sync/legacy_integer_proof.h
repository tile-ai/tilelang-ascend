// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#pragma once

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <limits>

#include "upstream/constraints.h"

namespace tvm::tl::ascend_sync {

using namespace tir;

// Proof-only validation: preserve original-width arithmetic and actual lazy
// branch evaluation without rewriting the emitted TIR or its DMA predicates.
class LegacyIntegerAudit : public ExprVisitor {
public:
  LegacyIntegerAudit(ProofAnalyzer *analyzer, const char **failure)
      : analyzer_(analyzer), failure_(failure) {}

  bool Run(const PrimExpr &value) {
    VisitExpr(value);
    return valid_;
  }

private:
  void Reject(const char *message) {
    *failure_ = message;
    valid_ = false;
  }

  void VisitExpr(const PrimExpr &value) final {
    if (!valid_)
      return;
    const DataType type = value.dtype();
    if (type.is_uint() && !type.is_bool() &&
        (value.as<AddNode>() || value.as<SubNode>() || value.as<MulNode>())) {
      Reject("wrapping unsigned index arithmetic is unsupported");
      return;
    }
    ExprVisitor::VisitExpr(value);
    if (valid_)
      AuditValue(value);
  }

  void VisitExpr_(const CallNode *op) final {
    if (!op->op.same_as(builtin::if_then_else())) {
      ExprVisitor::VisitExpr_(op);
      return;
    }
    if (op->args.size() != 3 || !op->args[0].dtype().is_scalar() ||
        !op->args[0].dtype().is_bool() || op->args[1].dtype() != op->dtype ||
        op->args[2].dtype() != op->dtype) {
      Reject("malformed lazy integer conditional");
      return;
    }
    const PrimExpr &condition = op->args[0];
    VisitExpr(condition);
    if (!valid_)
      return;
    if (analyzer_->z3_prover.CanProve(condition)) {
      VisitExpr(op->args[1]);
      return;
    }
    if (analyzer_->z3_prover.CanProve(!condition)) {
      VisitExpr(op->args[2]);
      return;
    }
    for (int branch = 1; branch <= 2 && valid_; ++branch) {
      auto exit =
          analyzer_->EnterConstraint(branch == 1 ? condition : !condition);
      VisitExpr(op->args[branch]);
      exit();
    }
  }

  void AuditValue(const PrimExpr &value) {
    const DataType type = value.dtype();
    if (type.is_bool() || (!type.is_int() && !type.is_uint()))
      return;
    const int bits = type.bits();
    if (!type.is_scalar() || bits > (type.is_uint() ? 32 : 64)) {
      Reject("requires a value-preserving scalar index");
      return;
    }
    PrimExpr divisor;
    if (const auto *operation = value.as<FloorDivNode>())
      divisor = operation->b;
    else if (const auto *operation = value.as<DivNode>())
      divisor = operation->b;
    else if (const auto *operation = value.as<FloorModNode>())
      divisor = operation->b;
    else if (const auto *operation = value.as<ModNode>())
      divisor = operation->b;
    if (divisor.defined() &&
        !analyzer_->z3_prover.CanProve(divisor != make_zero(divisor.dtype()))) {
      Reject("index division requires a nonzero divisor");
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
    if (const auto *conversion = value.as<CastNode>()) {
      const DataType source_type = conversion->value.dtype();
      if (!source_type.is_scalar() ||
          !((source_type.is_int() && source_type.bits() <= 64) ||
            (source_type.is_uint() && source_type.bits() <= 32))) {
        Reject("index casts require integer source values");
        return;
      }
      const PrimExpr source = cast(DataType::Int(64), conversion->value);
      if (!analyzer_->z3_prover.CanProve(
              source >= IntImm(DataType::Int(64), minimum) &&
              source <= IntImm(DataType::Int(64), maximum))) {
        Reject("lossy index casts are unsupported");
        return;
      }
    }
    const PrimExpr wide = cast(DataType::Int(64), value);
    // Avoid the legacy folder, which can wrap narrow constant intermediates.
    if (!analyzer_->z3_prover.CanProve(
            wide >= IntImm(DataType::Int(64), minimum) &&
            wide <= IntImm(DataType::Int(64), maximum)))
      Reject("cannot prove nonwrapping signed index arithmetic");
  }

  ProofAnalyzer *analyzer_;
  const char **failure_;
  bool valid_{true};
};

inline bool TryLegacyInteger(const PrimExpr &value, ProofAnalyzer *analyzer,
                             const char **failure) {
  if (!value.defined() || SideEffect(value) > CallEffectKind::kPure) {
    *failure = "mutable or opaque expressions are unsupported";
    return false;
  }
  const DataType dtype = value.dtype();
  if (!dtype.is_scalar() || dtype.is_bool() ||
      !((dtype.is_int() && dtype.bits() <= 64) ||
        (dtype.is_uint() && dtype.bits() <= 32))) {
    *failure = "requires a value-preserving scalar index";
    return false;
  }
  return LegacyIntegerAudit(analyzer, failure).Run(value);
}

inline bool TryLegacyPredicate(const PrimExpr &predicate,
                               ProofAnalyzer *analyzer, const char **failure) {
  if (!predicate.defined() || !predicate.dtype().is_scalar() ||
      !predicate.dtype().is_bool() ||
      SideEffect(predicate) > CallEffectKind::kPure) {
    *failure = "assumptions require a pure scalar boolean predicate";
    return false;
  }
  return LegacyIntegerAudit(analyzer, failure).Run(predicate);
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
