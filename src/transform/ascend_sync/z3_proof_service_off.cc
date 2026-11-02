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

// Adapted from TileLang TVM 4211874e9e4b8770fd20ad0d98ab80afd1df5028.
// Source: src/target/z3/z3_prover_off.cc. Unlike Analyzer's optional fallback,
// this service reports disabled operations instead of silently declining proof.

#include "z3_proof_service.h"

#include <tvm/runtime/logging.h>

#include <stdexcept>

namespace tvm::tl::ascend_sync {

namespace {

[[noreturn]] void Unavailable() {
  LOG(FATAL)
      << "Ascend synchronization Z3ProofService is unavailable: rebuild "
         "TileLang with USE_ASCEND_SYNC_Z3=ON and a supported Z3 library";
  throw std::logic_error("Unreachable after fatal diagnostic");
}

} // namespace

class Z3ProofService::Impl {};

bool Z3ProofService::Available() { return false; }

Z3ProofService::Z3ProofService() : impl_(nullptr) {}
Z3ProofService::~Z3ProofService() = default;

bool Z3ProofService::CanProve(const PrimExpr &) { Unavailable(); }
void Z3ProofService::Bind(const tir::Var &, const Range &, bool) {
  Unavailable();
}
void Z3ProofService::Bind(const tir::Var &, const PrimExpr &, bool) {
  Unavailable();
}
std::function<void()> Z3ProofService::EnterConstraint(const PrimExpr &, bool) {
  Unavailable();
}
void Z3ProofService::SetTimeoutMs(unsigned) { Unavailable(); }
void Z3ProofService::SetRLimit(unsigned) { Unavailable(); }
String Z3ProofService::GetSMTLIB2(Optional<PrimExpr>) { Unavailable(); }
String Z3ProofService::GetStats() { Unavailable(); }

} // namespace tvm::tl::ascend_sync
