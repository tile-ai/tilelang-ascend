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

// Legacy target policy for the ported upstream synchronization engine.
// Hardware instruction semantics remain separate from dependency analysis.

#pragma once

#include "upstream/dependency_analysis.h"
#include <string>

namespace tvm::tl::ascend_sync {

struct SyncTargetPolicy {
  std::string platform{"A3"};
  bool ordinary_sync{true};
  bool vector_scalar_sync{true};
  int intra_core_flag_limit{8};
  ConflictOracle physical_conflict_oracle;

  void Validate() const;
  bool EnabledForPipes(const std::string &producer,
                       const std::string &consumer) const;
  bool HasImplicitSamePipeOrder(const std::string &pipe) const;
  bool UsesFlag(const std::string &producer, const std::string &consumer) const;
  void RequireEvent(const std::string &event) const;
};

} // namespace tvm::tl::ascend_sync
