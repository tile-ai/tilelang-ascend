/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// Ported from tile-ai/tilelang 1c0f402e166a7d0a055c991d2f82d2175a7917b7.
// Source: src/transform/common/constr_visitor.h,
// FreshenMutableReads/Constr/ConstrSet. Legacy adaptation: tirx/ffi types
// become tir/runtime types. Proof-only contexts use ProofAnalyzer rather than
// changing TVM's shared Analyzer.

#ifndef TVM_TL_TRANSFORM_ASCEND_SYNC_UPSTREAM_CONSTRAINTS_H_
#define TVM_TL_TRANSFORM_ASCEND_SYNC_UPSTREAM_CONSTRAINTS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/expr.h>
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../z3_proof_service.h"

namespace tvm::tl {

// Legacy arithmetic analysis assumes mathematical integers for these nodes.
bool ContainsUnsignedWrappingArithmetic(const PrimExpr &expr);

// Local proof adapter for the upstream synchronization analyses. Neither its
// contextual rewrites nor fresh variables may be used to rewrite emitted TIR.
class ProofAnalyzer : public arith::Analyzer {
public:
  ascend_sync::Z3ProofService z3_prover;

  ProofAnalyzer() = default;
  ~ProofAnalyzer();
  ProofAnalyzer(const ProofAnalyzer &) = delete;
  ProofAnalyzer &operator=(const ProofAnalyzer &) = delete;

  void Bind(const tir::Var &var, const Range &range,
            bool allow_override = false);
  void Bind(const tir::Var &var, const PrimExpr &value,
            bool allow_override = false);
  std::function<void()> EnterConstraint(const PrimExpr &constraint,
                                        bool is_assume = false);
  bool CanProve(const PrimExpr &predicate);
  bool CanProveEqual(const PrimExpr &lhs, const PrimExpr &rhs);
  PrimExpr Simplify(const PrimExpr &expr, int steps = 2);
  bool CanUseLegacyArithmetic(const PrimExpr &expr) const;

private:
  // Never reuse old legacy bindings after skipping a wrapping override.
  bool legacy_safe_{true};
  // Populate deliberately leaves scopes open. Keep them alive until explicitly
  // exited (in LIFO order) or this analyzer is destroyed.
  std::vector<std::function<void()>> constraint_exits_;
};

/*!
 * \brief Replace mutable reads with unknown variables.
 *
 * `Analyzer::Bind` installs a rewrite `var -> value`, so a definition reading
 * mutable state would outlive a store this does not track, and would make two
 * instances agree where two threads read two different registers.
 *
 * By default, every occurrence gets an independent variable. Snapshot mode
 * reuses variables for structurally equal reads within this mutator instance.
 * Use it only when repeated expressions describe the same captured value, such
 * as the bounds of one access region, and use separate instances for distinct
 * access points. This models a snapshot; it does not capture values at runtime.
 */
class FreshenMutableReads : public tir::ExprMutator {
public:
  using tir::ExprMutator::operator();

  /*! \brief Whether repeated reads denote separate evaluations or one snapshot.
   */
  enum class Mode {
    kPerOccurrence,
    kSnapshot,
  };

  explicit FreshenMutableReads(Mode mode = Mode::kPerOccurrence)
      : mode_(mode) {}

private:
  /*!
   * \brief Replace \p e according to the caller's evaluation model.
   *
   * An opaque call need not return the same value twice (`f() - f()` is not
   * zero), and two reads of one location may be separated by a store. Sharing
   * would assert an equality that need not hold, so reuse requires explicit
   * snapshot mode.
   */
  PrimExpr Fresh(const PrimExpr &e) {
    if (mode_ == Mode::kSnapshot) {
      auto it = memo_.find(e);
      if (it != memo_.end())
        return it->second;
    }
    tir::Var fresh("free" + std::to_string(count_++), e.dtype());
    if (mode_ == Mode::kSnapshot)
      memo_.emplace(e, fresh);
    return fresh;
  }

  PrimExpr VisitExpr_(const tir::BufferLoadNode *op) override {
    return Fresh(GetRef<PrimExpr>(op));
  }
  PrimExpr VisitExpr_(const tir::ProducerLoadNode *op) override {
    return Fresh(GetRef<PrimExpr>(op));
  }
  PrimExpr VisitExpr_(const tir::ReduceNode *op) override {
    return Fresh(GetRef<PrimExpr>(op));
  }
  PrimExpr VisitExpr_(const tir::CallNode *op) override {
    // `SideEffect` covers the arguments as well, so a call reading state
    // anywhere below it becomes a single unknown; only a wholly pure one
    // recurses.
    if (tir::SideEffect(GetRef<PrimExpr>(op)) > tir::CallEffectKind::kPure)
      return Fresh(GetRef<PrimExpr>(op));
    return tir::ExprMutator::VisitExpr_(op);
  }

  Mode mode_;
  int count_{0};
  std::unordered_map<PrimExpr, PrimExpr, StructuralHash, tir::ExprDeepEqual>
      memo_;
};

struct Constr {

  enum Kind {
    kConstr,
    kBindValue,
    kBindRange,
  } kind;
  bool is_assume = false;
  tir::Var var;
  PrimExpr value;
  Range range;

  Constr(PrimExpr constr, bool is_assume = false)
      : kind(kConstr), value(constr), is_assume(is_assume) {};
  Constr(tir::Var var, PrimExpr val)
      : kind(kBindValue), var(var), value(val) {};
  Constr(tir::Var var, Range range)
      : kind(kBindRange), var(var), range(range) {};

  Constr() = default;
  Constr(const Constr &other) = default;
  Constr(Constr &&other) = default;
  Constr &operator=(const Constr &other) = default;

  void Format(std::ostream &os) const {
    os << "Constr(kind=";
    switch (kind) {
    case kConstr:
      os << "kConstr";
      os << ", is_assume=" << (is_assume ? "true" : "false");
      os << ", value=" << value;
      break;
    case kBindValue:
      os << "kBindValue";
      os << ", var=" << var->name_hint;
      os << ", value=" << value;
      break;
    case kBindRange:
      os << "kBindRange";
      os << ", var=" << var->name_hint;
      os << ", range=Range(min=" << range->min;
      os << ", extent=" << range->extent << ")";
      break;
    default:
      os << "Unknown";
    }
    os << ")";
  }

  PrimExpr ToGenericConstr() const {
    switch (kind) {
    case kConstr:
      return value;
    case kBindValue:
      // A vector-typed bind cannot be stated as an equality.
      if (var.dtype().is_vector())
        return Bool(true);
      return var == value;
    case kBindRange:
      return tir::And(var >= range->min, var < (range->min + range->extent));
    }
    LOG(FATAL) << "Unreachable";
    return PrimExpr();
  }
  /*!
   * \brief Rewrite through \p subs, keeping the kind whenever possible, so that
   *        renaming one side of a two-instance comparison keeps its binds.
   */
  Constr Substitute(Map<tir::Var, PrimExpr> subs) const {
    switch (kind) {
    case kConstr:
      return Constr(tir::Substitute(value, subs), is_assume);
    case kBindValue:
    case kBindRange: {
      auto it = subs.find(var);
      const auto *new_var =
          it == subs.end() ? var.get() : (*it).second.as<tir::VarNode>();
      // A bind remapped onto a non-variable is only expressible as a predicate.
      if (new_var == nullptr)
        return Constr(tir::Substitute(ToGenericConstr(), subs), is_assume);
      if (kind == kBindValue)
        return Constr(GetRef<tir::Var>(new_var), tir::Substitute(value, subs));
      return Constr(GetRef<tir::Var>(new_var),
                    Range::FromMinExtent(tir::Substitute(range->min, subs),
                                         tir::Substitute(range->extent, subs)));
    }
    }
    LOG(FATAL) << "Unreachable";
    return Constr();
  }
  /*!
   * \brief A copy whose definition no longer reads mutable state.
   *
   * Only a bind needs it: `Bind` installs a rewrite, so `v == A[i]` would keep
   * being substituted after a store invalidated it. A predicate states a fact
   * that held on entry and stays usable as a premise inside the scope it
   * guards.
   *
   * Every consumer goes through here, so no path can miss the substitution.
   */
  Constr FreshenReads() const {
    FreshenMutableReads freshen;
    switch (kind) {
    case kConstr:
      return *this;
    case kBindValue:
      return Constr(var, freshen(value));
    case kBindRange:
      return Constr(var, Range::FromMinExtent(freshen(range->min),
                                              freshen(range->extent)));
    }
    LOG(FATAL) << "Unreachable";
    return Constr();
  }
  void Populate(ProofAnalyzer &analyzer) const {
    Constr c = FreshenReads();
    switch (c.kind) {
    case kConstr:
      // Simplify against the binds already installed, so a predicate held
      // in a bound boolean var (`cond` where `cond = (i < n)` is a bind)
      // expands into the condition it represents before the analyzer
      // digests it. EnterConstraint evaluates eagerly, exactly like Bind:
      // an opaque boolean entered as-is contributes nothing.
      analyzer.EnterConstraint(analyzer.Simplify(c.value), c.is_assume);
      break;
    case kBindValue:
      analyzer.Bind(c.var, c.value);
      break;
    case kBindRange:
      analyzer.Bind(c.var, c.range);
      break;
    default:
      LOG(FATAL) << "Unreachable";
    }
  }
};

struct ConstrSet {
  ConstrSet Substitute(Map<tir::Var, PrimExpr> subs) const {
    ConstrSet new_set;
    for (const auto &c : constrs_) {
      new_set.constrs_.push_back(c.Substitute(subs));
    }
    return new_set;
  }
  /*!
   * \brief Rename \p from and every bind defined at or after it by appending
   *        \p suffix; binds defined before \p from stay shared.
   *
   * Used when one set is instantiated twice in a single analyzer to model two
   * concurrent executions: two threads in ThreadSync, two logical iterations in
   * VerifyParallelLoop. Sharing a variable private to an execution turns
   * `v == f(a)` and `v == f(b)` into `f(a) == f(b)`, contradicting `a != b` for
   * an injective `f` and leaving a set under which every query holds vacuously.
   *
   * \p subs accumulates the renames, so the caller can apply the same map to
   * expressions held outside the set. \p from has to be an `Optional`, as
   * `tir::Var()` builds a real variable named "v". Pass \p rename_ranges false
   * to keep iteration variables shared, as ThreadSync's loop-carry model needs.
   */
  ConstrSet RenameFrom(const std::string &suffix, Map<tir::Var, PrimExpr> &subs,
                       const Optional<tir::Var> &from = NullOpt,
                       bool rename_ranges = true) const {
    bool active = !from.defined();
    for (const Constr &c : constrs_) {
      if (c.kind != Constr::kBindValue && c.kind != Constr::kBindRange)
        continue;
      if (from.defined() && c.var.same_as(from.value()))
        active = true;
      if (!rename_ranges && c.kind == Constr::kBindRange)
        continue;
      if (active && !subs.count(c.var))
        subs.Set(c.var, tir::Var(c.var->name_hint + suffix, c.var.dtype()));
    }
    return Substitute(subs);
  }

  /*!
   * \brief Union with \p other, dropping duplicates.
   *
   * A variable bound to conflicting values on the two sides is a caller error
   * -- it should have been renamed per side -- and is reported rather than
   * merged. `is_assume` is cleared: an assume is trusted only where it was
   * stated, and being trusted it may reference mutable reads.
   */
  ConstrSet Merge(const ConstrSet &other) const {
    ConstrSet out = *this;
    std::unordered_map<const tir::VarNode *, Constr> bound;
    std::unordered_set<PrimExpr, StructuralHash, tir::ExprDeepEqual> preds;
    for (const Constr &c : out.constrs_) {
      if (c.kind == Constr::kConstr)
        preds.insert(c.value);
      else
        bound.emplace(c.var.get(), c);
    }
    for (const Constr &c : other.constrs_) {
      if (c.kind == Constr::kConstr) {
        if (preds.insert(c.value).second)
          out.constrs_.push_back(c);
        continue;
      }
      auto it = bound.find(c.var.get());
      if (it == bound.end()) {
        bound.emplace(c.var.get(), c);
        out.constrs_.push_back(c);
      } else if (it->second.kind != c.kind ||
                 !tir::ExprDeepEqual()(it->second.ToGenericConstr(),
                                       c.ToGenericConstr())) {
        LOG(WARNING) << "ConstrSet::Merge: var '" << c.var->name_hint
                     << "' bound to conflicting values across merged sets; "
                        "caller should rename per-side-varying vars. Dropping "
                        "the incoming bind. existing="
                     << it->second.ToGenericConstr()
                     << " incoming=" << c.ToGenericConstr();
      }
    }
    for (Constr &c : out.constrs_) {
      c.is_assume = false;
    }
    return out.Normalized();
  }

  /*!
   * \brief Restore the bind-before-use invariant after concatenation.
   *
   * Lexical collection emits a bind before every entry that reads its
   * var, so a single set never needs this. `Merge` concatenates two such
   * sets, and an entry of the first set may read a bind var the second
   * set defines: replaying that order digests the predicate before the
   * definition exists, and the fact is lost -- `EnterConstraint`
   * evaluates eagerly, exactly like `Bind` (see the ordering discussion
   * in issue #3220).
   *
   * Entries stay in their current order, except that a bind is hoisted
   * (with the binds its own definition reads, recursively) directly
   * before the first entry that reads its var. Within one
   * lexically-collected set no entry reads a later bind, so this is a
   * no-op there and the program-order guarantee of #2805 stays intact.
   * A hoisted bind can still miss bounds that a first-set predicate
   * would have tightened; that residue is inherent to eager `Bind` and
   * only a lazily-evaluated bind (issue #3220, option 3) removes it.
   */
  ConstrSet Normalized() const {
    std::unordered_map<const tir::VarNode *, size_t> bind_at;
    for (size_t i = 0; i < constrs_.size(); ++i) {
      const Constr &c = constrs_[i];
      if ((c.kind == Constr::kBindValue || c.kind == Constr::kBindRange) &&
          !bind_at.count(c.var.get())) {
        bind_at.emplace(c.var.get(), i);
      }
    }
    enum : uint8_t { kNew = 0, kVisiting = 1, kEmitted = 2 };
    std::vector<uint8_t> state(constrs_.size(), kNew);
    ConstrSet out;
    out.constrs_.reserve(constrs_.size());
    std::function<void(size_t)> emit = [&](size_t index) {
      if (state[index] == kEmitted) {
        return;
      }
      if (state[index] == kVisiting) {
        // SSA-form collection cannot produce a definition cycle; if one
        // arrives anyway, keep the current relative order instead of
        // crashing the compile.
        LOG(WARNING) << "ConstrSet::Normalized: cyclic bind definitions; "
                        "keeping collection order";
        return;
      }
      state[index] = kVisiting;
      for (const tir::Var &var : ReadVars(constrs_[index])) {
        auto it = bind_at.find(var.get());
        if (it != bind_at.end() && it->second != index) {
          emit(it->second);
        }
      }
      state[index] = kEmitted;
      out.constrs_.push_back(constrs_[index]);
    };
    for (size_t i = 0; i < constrs_.size(); ++i) {
      emit(i);
    }
    return out;
  }

  /*!
   * \brief Lower every bind to a predicate, leaving predicates as they are.
   *
   * Use this when the analyzer binds one of the shared variables itself: a
   * second `Bind` of it trips the re-bind check, while predicates coexist with
   * any bind.
   */
  ConstrSet ToConstraints() const {
    ConstrSet out;
    out.constrs_.reserve(constrs_.size());
    for (const Constr &c : constrs_)
      out.constrs_.push_back(
          Constr(c.FreshenReads().ToGenericConstr(), c.is_assume));
    return out;
  }

  void Populate(ProofAnalyzer &analyzer) const {
    // Keep program order: `Analyzer::Bind` evaluates the bounds and modular set
    // of the value at bind time, so entering the binds first would widen them
    // -- a `v = tx` inside `if tx < 64` would lose its upper bound. Merged
    // sets are additionally normalized (see `Normalized`) so a bind always
    // precedes the entries that read its var.
    for (const auto &c : constrs_) {
      c.Populate(analyzer);
    }
  }

  /*! \brief Free vars an entry's definition or predicate reads (the bind's
   *  own var excluded by construction: it does not occur in its value). */
  static Array<tir::Var> ReadVars(const Constr &c) {
    switch (c.kind) {
    case Constr::kConstr:
    case Constr::kBindValue:
      return tir::UndefinedVars(c.value);
    case Constr::kBindRange: {
      Array<tir::Var> vars = tir::UndefinedVars(c.range->min);
      for (const tir::Var &var : tir::UndefinedVars(c.range->extent)) {
        vars.push_back(var);
      }
      return vars;
    }
    }
    LOG(FATAL) << "Unreachable";
    return {};
  }
  bool CanProve(const PrimExpr &expr) const {
    ProofAnalyzer analyzer;
    Populate(analyzer);
    return analyzer.CanProve(expr);
  }
  template <typename... Args> void AddConstr(Args... args) {
    constrs_.push_back(Constr(args...));
  }

  /*! \brief Convert the constraint set to a conjunction (AND) of all
   * constraints */
  PrimExpr ToConjunction() const {
    if (constrs_.empty())
      return Bool(true);
    PrimExpr result = constrs_[0].ToGenericConstr();
    for (size_t i = 1; i < constrs_.size(); ++i) {
      result = tir::And(result, constrs_[i].ToGenericConstr());
    }
    return result;
  }

  void Format(std::ostream &os) const {
    os << "ConstrSet(size=" << constrs_.size() << ") {\n";
    for (size_t i = 0; i < constrs_.size(); ++i) {
      os << "  [" << i << "] ";
      constrs_[i].Format(os);
      os << "\n";
    }
    os << "}";
  }

  std::vector<Constr> constrs_;
};

} // namespace tvm::tl

#endif // TVM_TL_TRANSFORM_ASCEND_SYNC_UPSTREAM_CONSTRAINTS_H_
