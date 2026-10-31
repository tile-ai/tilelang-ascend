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

// Legacy proof-context bridge for the upstream constraint representation.
// Source algorithms: src/transform/common/constr_visitor.h at
// tile-ai/tilelang 1c0f402e166a7d0a055c991d2f82d2175a7917b7.

#include "constraints.h"

#include <tvm/support/with.h>

#include <utility>

namespace tvm::tl {

namespace {

class UnsignedWrappingArithmeticFinder : public tir::ExprVisitor {
public:
  bool found{false};

  void VisitExpr_(const tir::AddNode *op) final { VisitArithmetic(op); }
  void VisitExpr_(const tir::SubNode *op) final { VisitArithmetic(op); }
  void VisitExpr_(const tir::MulNode *op) final { VisitArithmetic(op); }

private:
  template <typename Node> void VisitArithmetic(const Node *op) {
    if (op->dtype.is_uint() && !op->dtype.is_bool()) {
      found = true;
      return;
    }
    tir::ExprVisitor::VisitExpr_(op);
  }
};

struct ProofConstraintScope {
  std::shared_ptr<With<arith::ConstraintContext>> legacy_scope;
  std::function<void()> z3_exit;

  void Exit() {
    if (z3_exit) {
      auto exit = std::move(z3_exit);
      exit();
    }
    legacy_scope.reset();
  }
};

} // namespace

bool ContainsUnsignedWrappingArithmetic(const PrimExpr &expr) {
  UnsignedWrappingArithmeticFinder finder;
  finder(expr);
  return finder.found;
}

ProofAnalyzer::~ProofAnalyzer() {
  for (auto it = constraint_exits_.rbegin(); it != constraint_exits_.rend();
       ++it)
    (*it)();
}

void ProofAnalyzer::Bind(const tir::Var &var, const Range &range,
                         bool allow_override) {
  legacy_safe_ = CanUseLegacyArithmetic(range->min) &&
                 CanUseLegacyArithmetic(range->extent);
  if (legacy_safe_)
    arith::Analyzer::Bind(var, range, allow_override);
  z3_prover.Bind(var, range, allow_override);
}

void ProofAnalyzer::Bind(const tir::Var &var, const PrimExpr &value,
                         bool allow_override) {
  legacy_safe_ = CanUseLegacyArithmetic(value);
  if (legacy_safe_)
    arith::Analyzer::Bind(var, value, allow_override);
  z3_prover.Bind(var, value, allow_override);
}

std::function<void()> ProofAnalyzer::EnterConstraint(const PrimExpr &constraint,
                                                     bool is_assume) {
  auto scope = std::make_shared<ProofConstraintScope>();
  legacy_safe_ = CanUseLegacyArithmetic(constraint);
  if (legacy_safe_)
    scope->legacy_scope =
        std::make_shared<With<arith::ConstraintContext>>(this, constraint);
  scope->z3_exit = z3_prover.EnterConstraint(constraint, is_assume);
  auto exit = [scope]() { scope->Exit(); };
  constraint_exits_.push_back(exit);
  return exit;
}

bool ProofAnalyzer::CanProve(const PrimExpr &predicate) {
  return (CanUseLegacyArithmetic(predicate) &&
          arith::Analyzer::CanProve(predicate)) ||
         z3_prover.CanProve(predicate);
}

bool ProofAnalyzer::CanProveEqual(const PrimExpr &lhs, const PrimExpr &rhs) {
  return CanProve(lhs == rhs);
}

PrimExpr ProofAnalyzer::Simplify(const PrimExpr &expr, int steps) {
  return CanUseLegacyArithmetic(expr) ? arith::Analyzer::Simplify(expr, steps)
                                      : expr;
}

bool ProofAnalyzer::CanUseLegacyArithmetic(const PrimExpr &expr) const {
  return legacy_safe_ && !ContainsUnsignedWrappingArithmetic(expr);
}

} // namespace tvm::tl
