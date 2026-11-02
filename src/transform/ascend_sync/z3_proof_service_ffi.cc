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

#include "z3_proof_service.h"

#include <tvm/runtime/registry.h>

#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace tvm::tl::ascend_sync {

namespace {

using runtime::PackedFunc;
using runtime::TVMArgs;
using runtime::TVMRetValue;

unsigned ReadLimit(const TVMArgs &args) {
  ICHECK_EQ(args.size(), 2);
  int64_t limit = args[1];
  ICHECK_GE(limit, 0) << "Z3 proof resource limits must be nonnegative";
  ICHECK_LE(limit, std::numeric_limits<unsigned>::max());
  return static_cast<unsigned>(limit);
}

PackedFunc MakeProofDispatcher(std::shared_ptr<Z3ProofService> service) {
  return PackedFunc([service = std::move(service)](TVMArgs args,
                                                   TVMRetValue *result) {
    ICHECK_GE(args.size(), 1);
    std::string operation = args[0];
    if (operation == "bind_range") {
      ICHECK_EQ(args.size(), 4);
      tir::Var var = args[1];
      Range range = args[2];
      bool allow_override = args[3];
      service->Bind(var, range, allow_override);
    } else if (operation == "bind_expr") {
      ICHECK_EQ(args.size(), 4);
      tir::Var var = args[1];
      PrimExpr value = args[2];
      bool allow_override = args[3];
      service->Bind(var, value, allow_override);
    } else if (operation == "enter_constraint") {
      ICHECK_EQ(args.size(), 3);
      PrimExpr constraint = args[1];
      bool is_assume = args[2];
      auto exit = std::make_shared<std::function<void()>>(
          service->EnterConstraint(constraint, is_assume));
      // Hold the service until its scope exits, and make repeated cleanup
      // harmless.
      *result =
          PackedFunc([service, exit](TVMArgs cleanup_args, TVMRetValue *) {
            ICHECK_EQ(cleanup_args.size(), 0);
            if (*exit) {
              auto callback = std::move(*exit);
              callback();
            }
          });
    } else if (operation == "can_prove") {
      ICHECK_EQ(args.size(), 2);
      PrimExpr predicate = args[1];
      *result = service->CanProve(predicate);
    } else if (operation == "set_rlimit") {
      service->SetRLimit(ReadLimit(args));
    } else if (operation == "set_timeout_ms") {
      service->SetTimeoutMs(ReadLimit(args));
    } else if (operation == "smtlib") {
      ICHECK_LE(args.size(), 2);
      Optional<PrimExpr> predicate = NullOpt;
      if (args.size() == 2 && args[1].type_code() != kTVMNullptr) {
        predicate = args[1].operator PrimExpr();
      }
      *result = service->GetSMTLIB2(predicate);
    } else if (operation == "stats") {
      ICHECK_EQ(args.size(), 1);
      *result = service->GetStats();
    } else {
      LOG(FATAL) << "Unknown internal Z3ProofService operation: " << operation;
    }
  });
}

} // namespace

TVM_REGISTER_GLOBAL("tl.ascend_sync.Z3ProofServiceAvailable")
    .set_body_typed(Z3ProofService::Available);

TVM_REGISTER_GLOBAL("tl.ascend_sync.CreateZ3ProofService").set_body_typed([]() {
  ICHECK(Z3ProofService::Available())
      << "Z3ProofService is unavailable; configure with USE_ASCEND_SYNC_Z3=ON "
         "and a compatible Z3_ROOT";
  return MakeProofDispatcher(std::make_shared<Z3ProofService>());
});

} // namespace tvm::tl::ascend_sync
