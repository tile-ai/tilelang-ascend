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

// Backported from TileLang TVM 4211874e9e4b8770fd20ad0d98ab80afd1df5028.
// Source: src/target/z3/z3_prover_on.cc. Changes isolate the service from
// Analyzer and adapt public TIR/container APIs. Local safety fixes are marked
// below: scoped memo restoration, explicit binding override, cast capability,
// unsigned bit-vector interpretation, and shift-domain validation.

#include "z3_proof_service.h"
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <climits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tvm/ir/expr.h"
#include "tvm/runtime/data_type.h"
#include "z3++.h"

namespace tvm::tl::ascend_sync {

using namespace tir;

namespace {

// Adaptation guard: upstream's mathematical-integer cast translation is an
// identity. Reject conversions whose machine-integer value is not preserved
// rather than using that identity to prove a false statement about legacy TIR.
class SupportedCastChecker : public ExprVisitor {
public:
  void VisitExpr_(const CastNode *op) final {
    const DataType src = op->value.dtype();
    const DataType dst = op->dtype;
    const bool scalar_integer =
        src.lanes() == 1 && dst.lanes() == 1 &&
        (src.is_int() || src.is_uint() || src.is_bool()) &&
        (dst.is_int() || dst.is_uint() || dst.is_bool());
    if (scalar_integer) {
      const bool same_kind = (src.is_int() == dst.is_int()) &&
                             (src.is_uint() == dst.is_uint()) &&
                             (src.is_bool() == dst.is_bool());
      const bool preserves_value = (same_kind && dst.bits() >= src.bits()) ||
                                   (src.is_bool() && !dst.is_bool()) ||
                                   (src.is_uint() && !src.is_bool() &&
                                    dst.is_int() && dst.bits() > src.bits());
      ICHECK(preserves_value) << "Z3ProofService does not support narrowing or "
                                 "non-value-preserving "
                                 "integer casts: "
                              << src << " -> " << dst;
    }
    ExprVisitor::VisitExpr_(op);
  }
};

void CheckSupportedCasts(const PrimExpr &expr) { SupportedCastChecker()(expr); }

class ShiftDomainCollector : public ExprVisitor {
public:
  std::vector<Call> shifts;

  void VisitExpr_(const CallNode *op) final {
    ExprVisitor::VisitExpr_(op);
    if (op->op.same_as(tir::builtin::shift_left()) ||
        op->op.same_as(tir::builtin::shift_right()))
      shifts.push_back(GetRef<Call>(op));
  }
};

struct Namespace {
  std::unordered_set<std::string> used_names;
  /// @brief Get a new name that is not used before
  /// This function is used to generate z3 variable names
  ///
  /// Z3 may deduplicate variables with the same name, which
  /// causes issues when different TVM variables are mapped to
  /// the same z3 variable.
  ///
  /// This function generates unique names by appending
  /// suffixes to the original expression string representation.
  ///
  /// such as : "x", "x$1", "x$2", ...
  std::string GetNewName(const PrimExpr &expr) {
    std::stringstream ss;
    ss << expr;
    auto name = ss.str();
    if (used_names.count(name) == 0) {
      used_names.insert(name);
      return name;
    }
    int idx = 1;
    std::string check_name = name + "$" + std::to_string(idx);
    while (used_names.count(check_name)) {
      idx++;
      check_name = name + "$" + std::to_string(idx);
    }
    used_names.insert(check_name);
    return check_name;
  }
};

} // namespace

class Z3ProofService::Impl : ExprFunctor<z3::expr(const PrimExpr &)> {
public:
  using Base = ExprFunctor<z3::expr(const PrimExpr &)>;

  /// @brief Z3 context owning every handle of this prover.
  // Null until the solver materializes; Materialize creates a context
  // private to this prover so its handles remain local to one service.
  std::shared_ptr<z3::context> ctx;

  /// @brief Z3 solver instance
  std::optional<z3::solver> solver;

  /// @brief Memoized PrimExpr -> slot in z3_pool_. Holds no z3 handles, so
  /// its pointer-hashed bucket order cannot affect z3 object lifetime.
  std::unordered_map<PrimExpr, size_t, StructuralHash, ExprDeepEqual> memo_;

  /// @brief Slots owning the memoized z3 handles, plus a free-slot stack.
  /// Handles are created and released only at fixed points of the execution
  /// path (never in hash-bucket order), so Z3's ast-id recycling -- and thus
  /// solver behavior under rlimit -- stays deterministic across processes.
  std::vector<std::optional<z3::expr>> z3_pool_;
  std::vector<size_t> free_slots_;

  // Upstream only removes side-effect memo entries when a constraint exits.
  // A Bind(var, constant) inside that scope also changes the memo; retain its
  // previous state so eager and lazy scopes both restore the full environment.
  struct MemoSnapshot {
    std::unordered_map<PrimExpr, size_t, StructuralHash, ExprDeepEqual> memo;
    std::vector<std::optional<z3::expr>> pool;
    std::vector<size_t> free_slots;
  };
  std::vector<std::optional<MemoSnapshot>> scope_memo_snapshots_;

  MemoSnapshot CaptureMemo() const { return {memo_, z3_pool_, free_slots_}; }

  void RestoreMemo(MemoSnapshot snapshot) {
    memo_ = std::move(snapshot.memo);
    z3_pool_ = std::move(snapshot.pool);
    free_slots_ = std::move(snapshot.free_slots);
  }

  void MemoPut(const PrimExpr &e, const z3::expr &ex) {
    auto [it, inserted] = memo_.emplace(e, 0);
    if (!inserted)
      return; // already memoized: keep the existing entry
    if (free_slots_.empty()) {
      it->second = z3_pool_.size();
      z3_pool_.emplace_back(ex);
    } else {
      it->second = free_slots_.back();
      free_slots_.pop_back();
      z3_pool_[it->second] = ex;
    }
  }

  void MemoErase(const PrimExpr &e) {
    auto it = memo_.find(e);
    if (it == memo_.end())
      return;
    z3_pool_[it->second].reset();
    free_slots_.push_back(it->second);
    memo_.erase(it);
  }

  // Upstream accepts allow_override but MemoPut never replaces an entry.
  // Preserve other bound variables' captured values, while dropping compound
  // expression caches that may have been translated using the old binding.
  void InvalidateBinding(const Var &var) {
    std::vector<PrimExpr> erase;
    for (const auto &entry : memo_) {
      if (!entry.first.as<VarNode>() || entry.first.same_as(var))
        erase.push_back(entry.first);
    }
    for (const auto &expr : erase)
      MemoErase(expr);
  }

  const z3::expr *MemoGet(const PrimExpr &e) const {
    auto it = memo_.find(e);
    return it == memo_.end() ? nullptr : &*z3_pool_[it->second];
  }

  bool is_assume = false;

  void RequireShiftAmount(const z3::expr &amount, unsigned bit_width) {
    z3::expr_vector invalid_amount(*ctx);
    invalid_amount.push_back(
        !(amount >= 0 && amount < ctx->int_val(bit_width)));
    ICHECK(solver->check(invalid_amount) == z3::unsat)
        << "Z3ProofService does not support a shift amount unless its range "
           "is proven to be within the operand bit width";
  }

  // Validate shifts against the existing environment before appending a Bind
  // or entering a scope. An invalid lazy input must not poison the journal and
  // surface only when a later, unrelated query materializes the solver.
  void ValidateInput(const PrimExpr &expr) {
    CheckSupportedCasts(expr);
    ShiftDomainCollector collector;
    collector(expr);
    if (collector.shifts.empty())
      return;
    Materialize();
    for (const Call &shift : collector.shifts) {
      ICHECK_EQ(shift->args.size(), 2U);
      if (IsValidDType(shift->args[0].dtype()) &&
          IsValidDType(shift->args[1].dtype())) {
        RequireShiftAmount(ConvertInt(shift->args[1]),
                           shift->args[0].dtype().bits());
      }
    }
  }

  /// @brief Namespace for variable naming
  Namespace ns;

  /// @brief Timeout in milliseconds
  unsigned timeout_ms{UINT_MAX};

  /// @brief Max steps
  unsigned rlimit{UINT_MAX};

  /// @brief Create a z3 solver with custom options
  static z3::solver CreateSolver(z3::context &ctx) {
    z3::solver result(ctx);
    // here we disable model generation to speed up the solving process
    result.set("model", false);
    // ensure determinstic behavior
    result.set("random_seed", (unsigned)42);
    return result;
  }

  Impl() {
    // The solver (and its context) is created lazily (see Materialize).
    // Analyzers are constructed in large numbers as cheap scratch objects,
    // and only a tiny fraction ever reaches the Z3 fallback of CanProve;
    // creating the solver (and translating every Bind into z3 constraints)
    // eagerly made every Analyzer pay Z3's initialization cost up front.
    // Until the first operation that actually needs the solver,
    // Bind/EnterConstraint only journal their arguments into scope_stack_.
    scope_stack_.push_back({});
    scope_side_effects_.push_back({});
    scope_memo_snapshots_.push_back(std::nullopt);
    // default timeout 5ms
    // Z3's implementation of timeout, when setting timeout T ms, it will stop
    // at T - 1 ms SetTimeoutMs(5); use rlimit, not timeout to ensure
    // determinstic behavior
    SetRLimit(1e4);
  }

  /// @brief Create the solver on first use and replay the recorded journal.
  ///
  /// The live scope_stack_ (which also feeds the SMTLIB2 debug output) holds
  /// exactly the binds and constraints an eagerly-built solver would carry:
  /// binds append to the innermost scope and scopes pop LIFO, so walking the
  /// scopes front-to-back replays the surviving entries in the order they
  /// were recorded. Entries from scopes that already exited were popped
  /// together with their scope and are correctly absent. The replay performs
  /// the same MemoPut/solver->add sequence the eager path would have done
  /// for these entries, so CanProve answers are unchanged.
  void Materialize() {
    if (solver)
      return;
    // The solver gets its own context: verdicts under the deterministic
    // rlimit budget depend on the context's accumulated AST state (ids feed
    // the search heuristics), so sharing a context couples every solver's
    // borderline answers to whichever analyzers happened to be created
    // before it. A private context makes each analyzer's answers a pure
    // function of its own journal and query history. Unmaterialized provers
    // hold no context (and no z3 state) at all.
    ctx = std::make_shared<z3::context>();
    solver.emplace(CreateSolver(*ctx));
    if (timeout_ms != UINT_MAX) {
      solver->set("timeout", timeout_ms);
    }
    if (rlimit != UINT_MAX) {
      solver->set("rlimit", rlimit);
    }
    ICHECK_EQ(scope_side_effects_.size(), scope_stack_.size());
    for (size_t si = 0; si < scope_stack_.size(); ++si) {
      for (const Scope &s : scope_stack_[si]) {
        switch (s.kind) {
        case Scope::BindValue:
          ApplyBindValue(s.var, s.value, s.allow_override);
          break;
        case Scope::BindRange:
          ApplyBindRange(s.var, s.min, s.extent, s.allow_override);
          break;
        case Scope::Constraint:
          scope_memo_snapshots_[si] = CaptureMemo();
          ApplyConstraint(s.constraint, s.is_assume, si);
          break;
        }
      }
    }
  }

  /// @brief Create a Free z3 expression from PrimExprNode
  z3::expr Create(const PrimExprNode *op) {
    // Expression translation only happens with a live solver: every public
    // entry point that can reach a visitor materializes first.
    ICHECK(solver.has_value());
    auto ref = GetRef<PrimExpr>(op);
    auto dtype = op->dtype;
    std::string name = ns.GetNewName(ref);
    /// TVM max_val can't handle uint64 max correctly, so we special case it
    /// here
    if (dtype.is_bool()) {
      return ctx->bool_const(name.c_str());
    } else {
      z3::expr e = ctx->int_const(name.c_str());
      if (dtype.is_uint() && dtype.bits() == 64) {
        solver->add(ctx->int_val(0) <= e &&
                    e <= ctx->int_val((uint64_t)UINT64_MAX));
      } else {
        auto min_val = Downcast<IntImm>(min_value(dtype))->value;
        auto max_val = Downcast<IntImm>(max_value(dtype))->value;
        solver->add(ctx->int_val(min_val) <= e && e <= ctx->int_val(max_val));
      }
      return e;
    }
  }

  struct Scope {
    enum Kind {
      BindValue,
      BindRange,
      Constraint,
    } kind;
    Var var;
    PrimExpr value;
    PrimExpr min;
    PrimExpr extent;
    PrimExpr constraint;
    /// Assume-mode constraints keep the free variables they introduce
    /// memoized for the lifetime of their scope; recorded so a deferred
    /// replay reproduces the same memo behavior.
    bool is_assume = false;
    bool allow_override = false;
  };

  /// @brief scope_stack memorizes existing constraint and bindings
  ///        to generate SMTLIB2 representation with comments
  std::vector<std::vector<Scope>> scope_stack_;

  /// @brief Per-scope memoized side-effect exprs (parallel to scope_stack_).
  /// An assume-mode constraint keeps the free variables it introduces in the
  /// memo until its scope exits; the exit callback erases whatever this
  /// stack's top holds, which stays correct whether the constraint was
  /// translated when it was entered or later, during a deferred replay.
  std::vector<std::vector<PrimExpr>> scope_side_effects_;

  /// @brief Enter a constraint scope
  std::function<void()> EnterConstraint(const PrimExpr &constraint,
                                        bool is_assume = false) {
    ValidateInput(constraint);
    if (!IsValidDType(constraint->dtype))
      return nullptr;
    scope_stack_.push_back({});
    scope_stack_.back().push_back(Scope{Scope::Constraint, Var(), PrimExpr(),
                                        PrimExpr(), PrimExpr(), constraint,
                                        is_assume});
    scope_side_effects_.push_back({});
    scope_memo_snapshots_.push_back(std::nullopt);
    if (solver) {
      scope_memo_snapshots_.back() = CaptureMemo();
      try {
        ApplyConstraint(constraint, is_assume, scope_side_effects_.size() - 1);
      } catch (...) {
        RestoreMemo(std::move(*scope_memo_snapshots_.back()));
        scope_memo_snapshots_.pop_back();
        scope_side_effects_.pop_back();
        scope_stack_.pop_back();
        throw;
      }
    }
    // Exit callback: scopes leave LIFO, so the stack tops are this scope's.
    // If the solver materialized while the scope was live, the replay pushed
    // a frame for it (and recorded any assume-kept side effects), so the pop
    // and the memo erasure below stay balanced either way.
    return [this]() {
      if (solver) {
        solver->pop();
        ICHECK(scope_memo_snapshots_.back().has_value());
        RestoreMemo(std::move(*scope_memo_snapshots_.back()));
      }
      scope_memo_snapshots_.pop_back();
      scope_side_effects_.pop_back();
      scope_stack_.pop_back();
    };
  }

  /// @brief Translate a constraint into the solver (push + assert). Assume
  /// mode keeps introduced free vars memoized for the scope's lifetime by
  /// parking them in scope_side_effects_[scope_index].
  void ApplyConstraint(const PrimExpr &constraint, bool is_assume,
                       size_t scope_index) {
    solver->push();
    this->is_assume = is_assume;
    try {
      solver->add(VisitBool(constraint));
    } catch (...) {
      this->is_assume = false;
      ClearTransientReads();
      solver->pop();
      throw;
    }
    this->is_assume = false;
    auto side_effect_exprs = std::move(side_effect_exprs_);
    side_effect_exprs_.clear();
    if (is_assume) {
      auto &keep = scope_side_effects_[scope_index];
      keep.insert(keep.end(), side_effect_exprs.begin(),
                  side_effect_exprs.end());
    } else {
      for (const auto &expr : side_effect_exprs) {
        MemoErase(expr);
      }
    }
  }

  /// @brief Check trivil bad cases, return true if the expr is a bad case
  /// Z3 prover may take a long time to initialize (at least 200us),
  /// This optimization can speedup 30% of the test cases in our unit tests
  bool CheckTrivilBadCases(const PrimExpr &expr) {
    if (IsFreeNode(expr)) {
      return true;
    }
    auto checkTrivilCmp = [this](const PrimExpr &lhs, const PrimExpr &rhs) {
      if (IsFreeNode(lhs) && rhs->IsInstance<IntImmNode>()) {
        return true;
      }
      if (IsFreeNode(rhs) && lhs->IsInstance<IntImmNode>()) {
        return true;
      }
      if (IsFreeNode(lhs) && IsFreeNode(rhs)) {
        return true;
      }
      // cast('xxx', free_var) == constant
      if (auto cast = lhs.as<CastNode>()) {
        if (IsFreeNode(cast->value) && rhs->IsInstance<IntImmNode>()) {
          return true;
        }
      }
      // constant == cast('xxx', free_var)
      if (auto cast = rhs.as<CastNode>()) {
        if (IsFreeNode(cast->value) && lhs->IsInstance<IntImmNode>()) {
          return true;
        }
      }
      return false;
    };
    if (auto eq = expr.as<EQNode>()) {
      auto lhs = eq->a;
      auto rhs = eq->b;
      return checkTrivilCmp(lhs, rhs);
    } else if (auto ne = expr.as<NENode>()) {
      auto lhs = ne->a;
      auto rhs = ne->b;
      return checkTrivilCmp(lhs, rhs);
    }
    return false;
  }

  /// @brief Check if the expression can be proved
  bool CanProve(const PrimExpr &expr) {
    ValidateInput(expr);
    if (!IsValidDType(expr->dtype))
      return false;
    // The trivial-bad-case check below consults the memo (IsFreeNode), so
    // the journal must be replayed first for it to see the bound vars.
    Materialize();
    if (CheckTrivilBadCases(expr))
      return false;
    z3::expr_vector constr(*ctx);
    constr.push_back(!ConvertBool(expr));
    auto result = solver->check(constr);
    constr.pop_back();
    return result == z3::unsat;
  }

  /// @brief Binded
  /// @brief Bind a variable to a value or a range
  /// A bind journaled inside a live constraint scope is dropped with that
  /// scope. The eager path differed only cosmetically: it kept the var's
  /// memo translation after the scope's solver frame (and with it the
  /// var's range asserts) was popped, leaving an unconstrained placeholder
  /// -- observably the same as translating the var as a fresh free
  /// variable on demand, which is what a later query on the journal does.
  void Bind(const Var &var, const PrimExpr &value,
            bool allow_override = false) {
    ValidateInput(value);
    ICHECK(IsValidDType(value.dtype()))
        << "Z3ProofService Bind requires a scalar integer or boolean value";
    if (!IsValidDType(var->dtype))
      return;
    Scope binding{Scope::BindValue, var, value};
    binding.allow_override = allow_override;
    scope_stack_.back().push_back(std::move(binding));
    if (!solver)
      return; // journaled; translated when the solver materializes
    ApplyBindValue(var, value, allow_override);
  }

  void ApplyBindValue(const Var &var, const PrimExpr &value,
                      bool allow_override = false) {
    if (allow_override)
      InvalidateBinding(var);
    // we add the binding whenever the value is pure,
    // because non-pure parts are handling by creating free variables in
    // VisitExpr
    MemoPut(var, ConvertInt(value));
  }

  /// @brief Bind a variable to a range
  void Bind(const Var &var, const Range &range, bool allow_override = false) {
    ValidateInput(range->min);
    ValidateInput(range->extent);
    if (!IsValidDType(var->dtype))
      return;
    Scope binding{Scope::BindRange, var, PrimExpr(), range->min, range->extent};
    binding.allow_override = allow_override;
    scope_stack_.back().push_back(std::move(binding));
    if (!solver)
      return; // journaled; translated when the solver materializes
    ApplyBindRange(var, range->min, range->extent, allow_override);
  }

  void ApplyBindRange(const Var &var, const PrimExpr &min,
                      const PrimExpr &extent, bool allow_override = false) {
    if (allow_override)
      InvalidateBinding(var);
    // 1. Create a placeholder for the var, and save it in the memo
    //    if the var is overrided later, we can just update the memo, and the
    //    old placeholder will be ignored
    auto var_expr = Create(var.as<PrimExprNode>());
    MemoPut(var, var_expr);

    // 2. Add constraint on the placeholder
    //    when min_expr >= max_expr, the range is empty, which is under
    //    undefined behavior instead of adding an unsat constraint, we just skip
    //    the range constraint to leave it a free var
    if (tir::is_const_int(min) && tir::is_const_int(min + extent)) {
      int64_t min_value = *tir::as_const_int(min);
      int64_t max_value = *tir::as_const_int(min + extent);
      if (min_value < max_value) {
        solver->add(ctx->int_val(min_value) <= var_expr);
        solver->add(var_expr < ctx->int_val(max_value));
      }
    } else {
      solver->add(
          ConvertBool(extent <= 0 || (min <= var && var < min + extent)));
    }
  }

  /// @brief Set timeout in milliseconds
  void SetTimeoutMs(unsigned timeout_ms) {
    this->timeout_ms = timeout_ms;
    if (!solver)
      return; // stored; applied when the solver materializes
    solver->set("timeout", timeout_ms);
  }

  /// @brief Set max steps
  void SetRLimit(unsigned rlimit) {
    this->rlimit = rlimit;
    if (!solver)
      return; // stored; applied when the solver materializes
    solver->set("rlimit", rlimit);
  }

  /// @brief Get the SMTLIB2 representation of the current solver state
  String GetSMTLIB2() {
    Materialize();
    std::stringstream ss;
    ss << "(set-option :timeout " << timeout_ms << ")\n";
    AddScopeDebugMsg(ss);
    ss << solver->to_smt2();
    return ss.str();
  }

  void AddScopeDebugMsg(std::ostream &ss) {
    for (const auto &scope : scope_stack_) {
      ss << "; Entering Scope\n";
      for (const auto &s : scope) {
        switch (s.kind) {
        case Scope::Constraint:
          ss << "; constraint: " << s.constraint << "\n";
          break;
        case Scope::BindValue:
          ss << "; bind value: " << s.var << " = " << s.value << "\n";
          break;
        case Scope::BindRange:
          ss << "; bind range: " << s.var << " in [" << s.min << ", "
             << s.min + s.extent << ")\n";
          break;
        }
      }
    }
  }

  /// @brief Get the SMTLIB2 representation of the current solver state with
  /// additional expr trying to prove
  String GetSMTLIB2(const PrimExpr &expr) {
    ValidateInput(expr);
    Materialize();
    std::stringstream ss;
    ss << "(set-option :timeout " << timeout_ms << ")\n";
    AddScopeDebugMsg(ss);
    ss << "; Trying to prove: " << expr << "\n";
    solver->push();
    solver->add(!ConvertBool(expr));
    ss << solver->to_smt2();
    solver->pop();
    return ss.str();
  }

  /// @brief Get the statistics of the solver
  String GetStats() {
    Materialize();
    std::stringstream ss;
    ss << solver->statistics();
    return ss.str();
  }

private:
  using Z3BinOp = z3::expr (*)(const z3::expr &, const z3::expr &);

  std::vector<PrimExpr> side_effect_exprs_;

  void ClearTransientReads() {
    for (const auto &expr : side_effect_exprs_)
      MemoErase(expr);
    side_effect_exprs_.clear();
  }

  z3::expr ConvertBool(const PrimExpr &e, bool is_assume = false) {
    this->is_assume = is_assume;
    try {
      auto res = VisitBool(e);
      ClearTransientReads();
      this->is_assume = false;
      return res;
    } catch (...) {
      ClearTransientReads();
      this->is_assume = false;
      throw;
    }
  }

  z3::expr ConvertInt(const PrimExpr &e, bool is_assume = false) {
    this->is_assume = is_assume;
    try {
      auto res = VisitInt(e);
      ClearTransientReads();
      this->is_assume = false;
      return res;
    } catch (...) {
      ClearTransientReads();
      this->is_assume = false;
      throw;
    }
  }

  /// @brief Visit expression with memoization
  z3::expr VisitExpr(const PrimExpr &e) override {
    if (const z3::expr *hit = MemoGet(e)) {
      return *hit;
    }
    auto res = Base::VisitExpr(e);
    auto side_effect = SideEffect(e);
    if (side_effect <= CallEffectKind::kPure) {
      MemoPut(e, res);
    } else if (side_effect <= CallEffectKind::kReadState) {
      MemoPut(e, res);
      side_effect_exprs_.emplace_back(e);
    } else {
      if (is_assume) {
        MemoPut(e, res);
      }
      side_effect_exprs_.emplace_back(e);
    }
    return res;
  }

  /// @brief Check if the expression is a free node having no constraints
  bool IsFreeNode(const PrimExpr &e) {
    if (memo_.count(e)) {
      return false;
    }
    return e->IsInstance<CallNode>() || e->IsInstance<BufferLoadNode>() ||
           e->IsInstance<ProducerLoadNode>() || e->IsInstance<ReduceNode>() ||
           (e->IsInstance<CastNode>() &&
            !IsValidDType(Downcast<Cast>(e)->value->dtype));
  }

  /// @brief Check if the dtype is valid for z3 integer operations
  static bool IsValidDType(const DataType &dtype) {
    return (dtype.is_int() || dtype.is_uint() || dtype.is_bool()) &&
           dtype.lanes() == 1;
  }

  /// @brief Visit the expression and convert it into z3 integer expression
  z3::expr VisitInt(const PrimExpr &expr) {
    auto e = VisitExpr(expr);
    if (e.is_bool()) {
      return z3::ite(e, ctx->int_val(1), ctx->int_val(0));
    } else {
      return e;
    }
  }

  /// @brief Visit the expression and convert it into z3 boolean expression
  z3::expr VisitBool(const PrimExpr &e) {
    auto expr = VisitExpr(e);
    if (expr.is_bool()) {
      return expr;
    } else {
      return expr != ctx->int_val(0);
    }
  }

  /// @brief Helper function to visit binary arithmetic operations
  z3::expr VisitArith(Z3BinOp signed_op, const PrimExprNode *op,
                      const PrimExpr &a, const PrimExpr &b) {
    if (IsValidDType(a->dtype) && IsValidDType(b->dtype)) {
      auto result = signed_op(VisitInt(a), VisitInt(b));
      // Unsigned arithmetic wraps at its original TIR width. Unbounded
      // integer results can otherwise prove overlapping addresses disjoint.
      // Comparisons have boolean dtype and must keep their boolean result.
      if (op->dtype.is_uint() && !op->dtype.is_bool())
        return z3::bv2int(z3::int2bv(op->dtype.bits(), result), false);
      return result;
    } else {
      return Create(op);
    }
  }

  z3::expr VisitExpr_(const LetNode *op) override {
    if (IsValidDType(op->var->dtype)) {
      MemoPut(op->var, VisitInt(op->value));
    }
    return VisitExpr(op->body);
  }
  z3::expr VisitExpr_(const CastNode *op) override {
    // if the inner dtype is valid, we just visit it
    if (IsValidDType(op->value->dtype) && IsValidDType(op->dtype)) {
      return VisitInt(op->value);
    } else {
      // otherwise, we create a new free z3 variable
      return Create(op);
    }
  }
  z3::expr VisitExpr_(const VarNode *op) override { return Create(op); }
  z3::expr VisitExpr_(const BufferLoadNode *op) override { return Create(op); }
  z3::expr VisitExpr_(const ProducerLoadNode *op) override {
    return Create(op);
  }
  z3::expr VisitExpr_(const ReduceNode *op) override { return Create(op); }
  z3::expr VisitExpr_(const MinNode *op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return z3::ite(a < b, a, b);
  }
  z3::expr VisitExpr_(const MaxNode *op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return z3::ite(a > b, a, b);
  }
  static z3::expr floordiv(const z3::expr &a, const z3::expr &b) {
    return z3::ite(b > 0, a / b, -((-a) / b));
  }
  static z3::expr floormod(const z3::expr &a, const z3::expr &b) {
    return z3::ite(b > 0, a % b, -((-a) % b));
  }
  z3::expr VisitTruncDivMod(const PrimExprNode *op, const PrimExpr &a,
                            const PrimExpr &b, bool remainder) {
    if (!IsValidDType(a.dtype()) || !IsValidDType(b.dtype()))
      return Create(op);
    auto lhs = VisitInt(a);
    auto rhs = VisitInt(b);
    auto valid = rhs != 0;
    if (op->dtype.is_int()) {
      auto minimum = Downcast<IntImm>(min_value(op->dtype))->value;
      valid = valid && !(lhs == ctx->int_val(minimum) && rhs == -1);
    }
    // Z3 integer division is Euclidean, whereas TIR Div/Mod truncate
    // toward zero. Divide magnitudes and restore the quotient's sign.
    auto magnitude = z3::ite(lhs < 0, -lhs, lhs) / z3::ite(rhs < 0, -rhs, rhs);
    auto quotient = z3::ite((lhs < 0) != (rhs < 0), -magnitude, magnitude);
    auto result = remainder ? lhs - quotient * rhs : quotient;
    // Division by zero and signed MIN/-1 have no defined machine result.
    // Keep those branches unknown instead of certifying a concrete value.
    return z3::ite(valid, result, Create(op));
  }
  z3::expr VisitExpr_(const AddNode *op) override {
    return VisitArith(z3::operator+, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const SubNode *op) override {
    return VisitArith(z3::operator-, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const MulNode *op) override {
    return VisitArith(z3::operator*, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const DivNode *op) override {
    return VisitTruncDivMod(op, op->a, op->b, false);
  }
  z3::expr VisitExpr_(const ModNode *op) override {
    return VisitTruncDivMod(op, op->a, op->b, true);
  }
  z3::expr VisitExpr_(const FloorDivNode *op) override {
    return VisitArith(floordiv, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const FloorModNode *op) override {
    return VisitArith(floormod, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const EQNode *op) override {
    return VisitArith(z3::operator==, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const NENode *op) override {
    return VisitArith(z3::operator!=, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const LTNode *op) override {
    return VisitArith(z3::operator<, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const LENode *op) override {
    return VisitArith(z3::operator<=, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const GTNode *op) override {
    return VisitArith(z3::operator>, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const GENode *op) override {
    return VisitArith(z3::operator>=, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const AndNode *op) override {
    return VisitBool(op->a) && VisitBool(op->b);
  }
  z3::expr VisitExpr_(const OrNode *op) override {
    return VisitBool(op->a) || VisitBool(op->b);
  }
  z3::expr VisitExpr_(const NotNode *op) override { return !VisitBool(op->a); }
  z3::expr VisitExpr_(const SelectNode *op) override {
    return z3::ite(VisitBool(op->condition), VisitInt(op->true_value),
                   VisitInt(op->false_value));
  }
  z3::expr VisitExpr_(const IntImmNode *op) override {
    return ctx->int_val(op->value);
  }

  // Bitwise operations
  z3::expr VisitExpr_(const CallNode *op) override {
    // Check if this is a bitwise operation
    if (op->op.same_as(tir::builtin::bitwise_and())) {
      return VisitBitwiseOp(z3::operator&, op);
    } else if (op->op.same_as(tir::builtin::bitwise_or())) {
      return VisitBitwiseOp(z3::operator|, op);
    } else if (op->op.same_as(tir::builtin::bitwise_xor())) {
      return VisitBitwiseOp(z3::operator^, op);
    } else if (op->op.same_as(tir::builtin::bitwise_not())) {
      return VisitBitwiseNotOp(op);
    } else if (op->op.same_as(tir::builtin::shift_left())) {
      return VisitShiftOp(z3::shl, op);
    } else if (op->op.same_as(tir::builtin::shift_right())) {
      if (op->dtype.is_uint())
        return VisitShiftOp(z3::lshr, op);
      return VisitShiftOp(z3::ashr, op);
    } else {
      // For other call nodes, create a free variable
      return Create(op);
    }
  }

  /// @brief Helper function to visit binary bitwise operations
  z3::expr VisitBitwiseOp(z3::expr (*op_func)(const z3::expr &,
                                              const z3::expr &),
                          const CallNode *op) {
    if (op->args.size() != 2) {
      LOG(FATAL) << "Binary bitwise operation expects 2 arguments, got "
                 << op->args.size();
      throw std::logic_error("Unreachable after fatal diagnostic");
    }

    const PrimExpr &a = op->args[0];
    const PrimExpr &b = op->args[1];
    unsigned bit_width =
        std::max(op->args[0].dtype().bits(), op->args[1].dtype().bits());

    if (IsValidDType(a->dtype) && IsValidDType(b->dtype)) {
      // Adaptation fix: upstream always requested a signed bv2int result,
      // which made unsigned high-bit values incorrectly become negative.
      return z3::bv2int(op_func(z3::int2bv(bit_width, VisitInt(a)),
                                z3::int2bv(bit_width, VisitInt(b))),
                        op->dtype.is_int());
    } else {
      return Create(op);
    }
  }

  /// @brief Helper function to visit unary bitwise not operation
  z3::expr VisitBitwiseNotOp(const CallNode *op) {
    if (op->args.size() != 1) {
      LOG(FATAL) << "Bitwise not operation expects 1 argument, got "
                 << op->args.size();
      throw std::logic_error("Unreachable after fatal diagnostic");
    }

    const PrimExpr &a = op->args[0];

    if (IsValidDType(a->dtype)) {
      // Cast integer to bit-vector, apply bitwise not, then cast back.
      unsigned bit_width = a.dtype().bits();
      z3::expr a_int = VisitInt(a);
      z3::expr a_bv = z3::int2bv(bit_width, a_int);
      // Match the result dtype, including unsigned high-bit values.
      return z3::bv2int(~a_bv, op->dtype.is_int());
    } else {
      return Create(op);
    }
  }

  /// @brief Helper function to visit shift operations
  z3::expr VisitShiftOp(z3::expr (*op_func)(const z3::expr &, const z3::expr &),
                        const CallNode *op) {
    if (op->args.size() != 2) {
      LOG(FATAL) << "Shift operation expects 2 arguments, got "
                 << op->args.size();
      throw std::logic_error("Unreachable after fatal diagnostic");
    }

    const PrimExpr &a = op->args[0];
    const PrimExpr &b = op->args[1];

    // Shift operations require integer types for both operands
    if (IsValidDType(a->dtype) && IsValidDType(b->dtype)) {
      // For shift operations, we need to ensure the shift amount is
      // non-negative and within reasonable bounds
      z3::expr a_expr = VisitInt(a);
      z3::expr b_expr = VisitInt(b);

      // Adaptation safety check: upstream asserted the shift precondition,
      // which made a negative amount render the persistent solver UNSAT.
      // Prove it using temporary assumptions instead; an unsupported amount
      // must never become an assumption about the source program.
      unsigned bit_width = a.dtype().bits();
      RequireShiftAmount(b_expr, bit_width);
      z3::expr a_bv = z3::int2bv(bit_width, a_expr);
      z3::expr b_bv = z3::int2bv(bit_width, b_expr);

      // Perform the shift in bit-vector domain, then cast back to int.
      z3::expr result_bv = op_func(a_bv, b_bv);
      return z3::bv2int(result_bv, op->dtype.is_int());
    } else {
      return Create(op);
    }
  }

  z3::expr VisitExprDefault_(const Object *op) override {
    LOG(FATAL) << "Z3ProofService only support integers, but got "
               << op->GetTypeKey() << ".";
    throw std::logic_error("Unreachable after fatal diagnostic");
  }
};

TVM_DLL bool Z3ProofService::CanProve(const PrimExpr &expr) {
  return impl_->CanProve(expr);
}
TVM_DLL void Z3ProofService::Bind(const Var &var, const Range &new_range,
                                  bool allow_override) {
  return impl_->Bind(var, new_range, allow_override);
}
TVM_DLL void Z3ProofService::Bind(const Var &var, const PrimExpr &expr,
                                  bool allow_override) {
  return impl_->Bind(var, expr, allow_override);
}
std::function<void()>
Z3ProofService::EnterConstraint(const PrimExpr &constraint, bool is_assume) {
  return impl_->EnterConstraint(constraint, is_assume);
}
String Z3ProofService::GetSMTLIB2(const Optional<PrimExpr> expr) {
  if (expr.defined()) {
    return impl_->GetSMTLIB2(expr.value());
  } else {
    return impl_->GetSMTLIB2();
  }
}
void Z3ProofService::SetTimeoutMs(unsigned timeout_ms) {
  impl_->SetTimeoutMs(timeout_ms);
}
void Z3ProofService::SetRLimit(unsigned max_step) {
  impl_->SetRLimit(max_step);
}
String Z3ProofService::GetStats() { return impl_->GetStats(); }
bool Z3ProofService::Available() { return true; }

Z3ProofService::Z3ProofService() : impl_(new Impl{}) {}
TVM_DLL Z3ProofService::~Z3ProofService() { delete impl_; }

} // namespace tvm::tl::ascend_sync
