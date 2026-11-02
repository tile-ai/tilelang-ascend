// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

// Original legacy adapter. The recognized guard is emitted by PersistentFor
// in this repository's src/ir.cc; no upstream scheduler or break semantics are
// approximated here.

#include "legacy_persistent_normalize.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace tvm::tl::ascend_sync {
using namespace tir;

namespace {

constexpr const char *kDiagnostic =
    "Upstream Ascend Persistent normalization requires a canonical leading "
    "monotone break guard";

bool IsBreak(const CallNode *call) {
  const auto *op = call->op.as<OpNode>();
  return op != nullptr && op->name == "tl.loop_break";
}

bool IsBreakStatement(const Stmt &statement) {
  const auto *evaluate = statement.as<EvaluateNode>();
  const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
  return call && IsBreak(call) && call->args.empty();
}

class BreakValidator : public StmtExprVisitor {
  void VisitExpr_(const CallNode *op) final {
    ICHECK(!IsBreak(op)) << kDiagnostic << "; unsupported break remains";
    StmtExprVisitor::VisitExpr_(op);
  }
};

struct Worker {
  Var variable;
  PrimExpr extent;
};

struct GuardPayload {
  Stmt payload;
  PrimExpr worker_base;
  int64_t domain;
  int64_t wave;
};

class PersistentNormalizer : public StmtExprMutator {
public:
  LegacyPersistentNormalization Run(const PrimFunc &function) {
    Stmt body = VisitStmt(function->body);
    BreakValidator()(body);
    PrimFunc result = function;
    if (!body.same_as(function->body))
      result.CopyOnWrite()->body = std::move(body);
    return {result, std::move(certificates_)};
  }

private:
  std::optional<GuardPayload> Match(const IfThenElseNode *guard,
                                    const ForNode *loop,
                                    const Map<Var, PrimExpr> &bindings) {
    if (guard->else_case.defined() || !IsBreakStatement(guard->then_case))
      return std::nullopt;
    ICHECK(SideEffect(guard->condition) <= CallEffectKind::kPure)
        << kDiagnostic << "; guard reads mutable state";
    arith::Analyzer analyzer;
    PrimExpr condition =
        analyzer.Simplify(Substitute(guard->condition, bindings));
    PrimExpr threshold, index;
    bool equality = false;
    if (const auto *comparison = condition.as<LENode>()) {
      threshold = comparison->a;
      index = comparison->b;
    } else if (const auto *comparison = condition.as<EQNode>()) {
      equality = true;
      const bool left_constant = comparison->a.as<IntImmNode>() != nullptr;
      threshold = left_constant ? comparison->a : comparison->b;
      index = left_constant ? comparison->b : comparison->a;
    } else {
      return std::nullopt;
    }
    const auto *domain = threshold.as<IntImmNode>();
    const auto *sum = index.as<AddNode>();
    if (!domain || !sum || domain->value <= 0)
      return std::nullopt;
    const MulNode *product = sum->a.as<MulNode>();
    PrimExpr base = sum->b;
    if (!product) {
      product = sum->b.as<MulNode>();
      base = sum->a;
    }
    const VarNode *worker = base.as<VarNode>();
    int64_t divisor = 1;
    if (const auto *division = base.as<FloorDivNode>()) {
      const auto *constant = division->b.as<IntImmNode>();
      if (!constant || constant->value <= 0)
        return std::nullopt;
      divisor = constant->value;
      worker = division->a.as<VarNode>();
    }
    if (!product || !worker)
      return std::nullopt;
    const IntImmNode *wave = nullptr;
    if (product->a.same_as(loop->loop_var))
      wave = product->b.as<IntImmNode>();
    else if (product->b.same_as(loop->loop_var))
      wave = product->a.as<IntImmNode>();
    if (!wave || wave->value <= 0)
      return std::nullopt;
    ICHECK(workers_.size() == 1 &&
           workers_[0].variable.same_as(GetRef<Var>(worker)))
        << kDiagnostic << "; worker must be the enclosing blockIdx.x";
    const auto *worker_extent = workers_[0].extent.as<IntImmNode>();
    ICHECK(wave->value <= std::numeric_limits<int64_t>::max() / divisor &&
           worker_extent && worker_extent->value == wave->value * divisor)
        << kDiagnostic << "; wave size must equal the worker extent";
    const DataType dtype = loop->loop_var.dtype();
    ICHECK(dtype.is_int() && dtype.lanes() == 1 &&
           (dtype.bits() == 32 || dtype.bits() == 64) &&
           loop->extent.dtype() == dtype && worker->dtype == dtype &&
           domain->dtype == dtype && wave->dtype == dtype &&
           base.dtype() == dtype)
        << kDiagnostic << "; requires matching signed 32/64-bit index types";
    const int64_t maximum = dtype.bits() == 32
                                ? std::numeric_limits<int32_t>::max()
                                : std::numeric_limits<int64_t>::max();
    ICHECK(worker_extent->value <= maximum && wave->value <= maximum &&
           domain->value <= maximum - (wave->value - 1))
        << kDiagnostic << "; domain/wave arithmetic could overflow";
    ICHECK(is_zero(loop->min) && loop->kind == ForKind::kSerial &&
           !loop->thread_binding.defined() &&
           SideEffect(loop->extent) <= CallEffectKind::kPure)
        << kDiagnostic << "; requires a serial loop with minimum zero";
    // Header expressions are evaluated outside all Let scopes in the body.
    // Do not let a body rebinding change the original iteration-range proof.
    PrimExpr extent = analyzer.Simplify(Substitute(loop->extent, bindings_));
    const auto *count = extent.as<IntImmNode>();
    const int64_t expected = (domain->value + wave->value - 1) / wave->value;
    ICHECK(count && count->value == expected)
        << kDiagnostic << "; original trip count must equal ceil(domain/wave)";
    // A bounds-aware Simplify can turn >= into == when the threshold is the
    // last reachable index (e.g. D=6,W=4 becomes w*2+cid//2 == 3). Equality is
    // equivalent to the original monotone guard only if no reachable index
    // can exceed that threshold. Do not normalize arbitrary equality breaks.
    ICHECK(!equality || expected * wave->value - 1 <= domain->value)
        << kDiagnostic
        << "; equality guard may miss indices above its threshold";
    // Simplify can divide a common static factor from the comparison:
    // 22 <= w*20+cid becomes 11 <= w*10+floordiv(cid,2). The only admitted
    // divided base is that same nonnegative hardware worker coordinate.
    return GuardPayload{Evaluate(0), base, domain->value, wave->value};
  }

  // Peel only pure lexical shells. Inactive iterations used to reach the
  // leading guard before any payload effect; skipping a mutable Let before
  // that guard would change program evaluation and must be rejected.
  std::optional<GuardPayload> StripLeadingGuard(const Stmt &statement,
                                                const ForNode *loop,
                                                Map<Var, PrimExpr> bindings) {
    if (const auto *branch = statement.as<IfThenElseNode>())
      return Match(branch, loop, bindings);
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      for (size_t i = 0; i != sequence->seq.size(); ++i) {
        const Stmt &child = sequence->seq[i];
        if (const auto *evaluate = child.as<EvaluateNode>();
            evaluate && evaluate->value.as<IntImmNode>())
          continue;
        auto match = StripLeadingGuard(child, loop, bindings);
        if (!match)
          return std::nullopt;
        Array<Stmt> body;
        for (size_t j = 0; j != sequence->seq.size(); ++j)
          body.push_back(j == i ? match->payload : sequence->seq[j]);
        match->payload = SeqStmt(body);
        return match;
      }
      return std::nullopt;
    }
    if (const auto *attribute = statement.as<AttrStmtNode>()) {
      if (attribute->attr_key != "resource_scope" ||
          !attribute->value.as<IntImmNode>())
        return std::nullopt;
      auto match = StripLeadingGuard(attribute->body, loop, bindings);
      if (match)
        match->payload =
            AttrStmt(attribute->node, attribute->attr_key, attribute->value,
                     match->payload, attribute->span);
      return match;
    }
    if (const auto *binding = statement.as<LetStmtNode>()) {
      if (SideEffect(binding->value) > CallEffectKind::kPure)
        return std::nullopt;
      bindings.Set(binding->var, Substitute(binding->value, bindings));
      auto match = StripLeadingGuard(binding->body, loop, bindings);
      if (match)
        match->payload = LetStmt(binding->var, binding->value, match->payload,
                                 binding->span);
      return match;
    }
    if (const auto *allocation = statement.as<AllocateNode>()) {
      if (!is_one(allocation->condition))
        return std::nullopt;
      for (const PrimExpr &extent : allocation->extents) {
        if (!extent.as<IntImmNode>())
          return std::nullopt;
      }
      auto match = StripLeadingGuard(allocation->body, loop, bindings);
      if (match) {
        Allocate result = GetRef<Allocate>(allocation);
        result.CopyOnWrite()->body = match->payload;
        match->payload = std::move(result);
      }
      return match;
    }
    if (const auto *declaration = statement.as<DeclBufferNode>()) {
      auto match = StripLeadingGuard(declaration->body, loop, bindings);
      if (match)
        match->payload =
            DeclBuffer(declaration->buffer, match->payload, declaration->span);
      return match;
    }
    return std::nullopt;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    auto match = StripLeadingGuard(op->body, op, bindings_);
    const ConstrSet saved = context_;
    if (!match) {
      if (SideEffect(op->min) <= CallEffectKind::kPure &&
          SideEffect(op->extent) <= CallEffectKind::kPure) {
        context_.AddConstr(op->loop_var,
                           Range::FromMinExtent(op->min, op->extent));
        context_.AddConstr(op->extent > make_zero(op->extent.dtype()));
      }
      Stmt result = StmtExprMutator::VisitStmt_(op);
      context_ = saved;
      return result;
    }
    const DataType dtype = op->loop_var.dtype();
    // Under 0 <= cid < W this numerator is nonnegative, and its static
    // maximum D+W-1 was checked above before constructing the expression.
    PrimExpr numerator =
        make_const(dtype, match->domain + match->wave - 1) - match->worker_base;
    For result = GetRef<For>(op);
    auto *changed = result.CopyOnWrite();
    changed->extent = floordiv(numerator, make_const(dtype, match->wave));
    context_.AddConstr(op->loop_var,
                       Range::FromMinExtent(op->min, changed->extent));
    context_.AddConstr(changed->extent > make_zero(dtype));
    changed->body = VisitStmt(match->payload);
    context_ = saved;
    certificates_.push_back({result, saved, match->domain, match->wave});
    return result;
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    const Map<Var, PrimExpr> previous = bindings_;
    const ConstrSet saved = context_;
    if (SideEffect(op->value) <= CallEffectKind::kPure) {
      bindings_.Set(op->var, Substitute(op->value, bindings_));
      context_.AddConstr(op->var, op->value);
    }
    Stmt result = StmtExprMutator::VisitStmt_(op);
    bindings_ = previous;
    context_ = saved;
    return result;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    bool worker = false;
    const ConstrSet saved = context_;
    if ((op->attr_key == "thread_extent" || op->attr_key == "virtual_thread") &&
        SideEffect(op->value) <= CallEffectKind::kPure) {
      if (const auto *iv = op->node.as<IterVarNode>())
        context_.AddConstr(
            iv->var,
            Range::FromMinExtent(make_zero(op->value.dtype()), op->value));
    }
    if (op->attr_key == "thread_extent") {
      const auto *iteration = op->node.as<IterVarNode>();
      if (iteration && iteration->thread_tag == "blockIdx.x") {
        workers_.push_back({iteration->var, op->value});
        worker = true;
      }
    }
    Stmt result = StmtExprMutator::VisitStmt_(op);
    if (worker)
      workers_.pop_back();
    context_ = saved;
    return result;
  }

  Map<Var, PrimExpr> bindings_;
  std::vector<Worker> workers_;
  ConstrSet context_;
  std::vector<LegacyPersistentLoopCertificate> certificates_;
};

} // namespace

PrimFunc NormalizeLegacyPersistent(const PrimFunc &function) {
  return NormalizeLegacyPersistentWithCertificates(function).function;
}

LegacyPersistentNormalization
NormalizeLegacyPersistentWithCertificates(const PrimFunc &function) {
  return PersistentNormalizer().Run(function);
}

TVM_REGISTER_GLOBAL("tl.ascend_sync.NormalizeLegacyPersistent")
    .set_body_typed(NormalizeLegacyPersistent);

} // namespace tvm::tl::ascend_sync
