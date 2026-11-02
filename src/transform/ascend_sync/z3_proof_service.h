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
// Source: include/tvm/arith/analyzer.h (Z3Prover API).

#ifndef TVM_TL_TRANSFORM_ASCEND_SYNC_Z3_PROOF_SERVICE_H_
#define TVM_TL_TRANSFORM_ASCEND_SYNC_Z3_PROOF_SERVICE_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/container/string.h>
#include <tvm/tir/expr.h>

#include <functional>

namespace tvm::tl::ascend_sync {

/*!
 * \brief Upstream Z3 prover adapted to legacy TIR without changing Analyzer.
 *
 * Bindings and constraints are local to each service. Constraint exit callbacks
 * must run in reverse creation order before the service is destroyed. A false
 * CanProve result means the predicate was not proven, including solver unknown.
 * Availability is separate: disabled builds reject operations explicitly.
 */
class Z3ProofService {
public:
  static bool Available();

  Z3ProofService();
  ~Z3ProofService();

  Z3ProofService(const Z3ProofService &) = delete;
  Z3ProofService &operator=(const Z3ProofService &) = delete;

  void Bind(const tir::Var &var, const Range &range,
            bool allow_override = false);
  void Bind(const tir::Var &var, const PrimExpr &value,
            bool allow_override = false);
  std::function<void()> EnterConstraint(const PrimExpr &constraint,
                                        bool is_assume = false);
  bool CanProve(const PrimExpr &predicate);

  void SetRLimit(unsigned limit);
  void SetTimeoutMs(unsigned timeout_ms);
  String GetSMTLIB2(Optional<PrimExpr> predicate = NullOpt);
  String GetStats();

private:
  class Impl;
  Impl *impl_;
};

} // namespace tvm::tl::ascend_sync

#endif // TVM_TL_TRANSFORM_ASCEND_SYNC_Z3_PROOF_SERVICE_H_
